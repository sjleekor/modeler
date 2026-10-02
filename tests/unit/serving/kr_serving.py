"""Label-free KR daily scoring and briefing JSON adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from datetime import time as datetime_time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from modeler.serving.kr_model import EXCLUDED_FEATURES, LoadedBundle, load_bundle
from modeler.serving.schema import validate_report

SEOUL = ZoneInfo("Asia/Seoul")
KR_EXCHANGE_CODES = frozenset({"KOSPI", "KOSDAQ"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware_decision_at(value: str, report_date: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("decision_at must include a timezone")
    local = parsed.astimezone(SEOUL)
    if local.date().isoformat() != report_date or local.time().replace(tzinfo=None) != datetime_time(10, 0):
        raise ValueError("decision_at must be 10:00 Asia/Seoul on report_date")
    return local


def _require_decision_reached(decision_at: str, report_date: str, *, now: datetime | None = None) -> None:
    decision = _aware_decision_at(decision_at, report_date)
    current = now or datetime.now(SEOUL)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("current time must include a timezone")
    if current.astimezone(SEOUL) < decision:
        raise ValueError("KR scoring cannot run before the 10:00 Asia/Seoul decision time")


def _validate_prepared_completion(prepared_manifest: dict, report_date: str) -> None:
    """Require the fixed native feature snapshot to be complete by D 09:30 KST."""
    cutoff_value = prepared_manifest.get("input_cutoff")
    if not isinstance(cutoff_value, str):
        raise ValueError("prepared manifest must record input_cutoff")
    cutoff = datetime.fromisoformat(cutoff_value)
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("prepared input_cutoff must include a timezone")
    cutoff_local = cutoff.astimezone(SEOUL)
    if cutoff_local.date().isoformat() != report_date or cutoff_local.time().replace(tzinfo=None) != datetime_time(9, 30):
        raise ValueError("prepared input_cutoff must be exactly 09:30 Asia/Seoul on report_date")
    for field in ("raw_snapshot_completed_at", "feature_marts_completed_at", "completed_at"):
        value = prepared_manifest.get(field)
        if not isinstance(value, str):
            raise ValueError(f"prepared manifest must record {field}")
        completed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if completed.tzinfo is None or completed.utcoffset() is None:
            raise ValueError(f"prepared {field} must include a timezone")
        if completed > cutoff:
            raise ValueError(f"prepared {field} is after the 09:30 input cutoff")


def score_cross_section(
    *, bundle: LoadedBundle, panel: pl.DataFrame, report_date: str,
    decision_at: str, feature_asof_date: str, prepared_manifest: dict,
) -> dict:
    """Score every eligible row; publication remains unresolved pending evidence."""
    _aware_decision_at(decision_at, report_date)
    if datetime.fromisoformat(feature_asof_date).date().isoformat() != feature_asof_date:
        raise ValueError("feature_asof_date must use YYYY-MM-DD")
    if feature_asof_date >= report_date:
        raise ValueError("feature_asof_date must precede report_date")
    if prepared_manifest.get("feature_asof_date") != feature_asof_date:
        raise ValueError("feature_asof_date differs from the preparation manifest")
    if prepared_manifest.get("market") != "KR":
        raise ValueError("preparation manifest market is not KR")
    if panel.is_empty():
        raise ValueError("scoring panel has no eligible KR symbols")
    if str(panel.get_column("trade_date").max()) != feature_asof_date or str(panel.get_column("trade_date").min()) != feature_asof_date:
        raise ValueError("scoring panel must contain exactly the declared feature as-of date")
    if set(EXCLUDED_FEATURES) & set(panel.columns):
        panel = panel.drop([name for name in EXCLUDED_FEATURES if name in panel.columns])
    if any(
        name.startswith(("y_", "raw_label", "fwd_ret_", "bench_ret_"))
        for name in panel.columns
    ):
        raise ValueError("label or forward-return fields reached KR serving input")
    required = set(bundle.feature_columns) | {"trade_date", "ticker", "market"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"scoring panel missing required model features: {sorted(missing)}")
    contract = prepared_manifest.get("model_config", {})
    if (contract.get("feature_set"), contract.get("flow_variant"), contract.get("preprocess_profile")) != (
        "FS1h", "lag1", "rank"
    ):
        raise ValueError("prepared panel lag/preprocess contract differs from bundle")
    if contract.get("feature_time_contract", {}).get("serving_applies_additional_lag") is not False:
        raise ValueError("daily serving must not shift already-defined price/flow features again")
    if set(contract.get("excluded_features", ())) != set(EXCLUDED_FEATURES):
        raise ValueError("prepared panel does not record the fixed balance feature exclusion")

    name_by_symbol = (
        dict(zip(panel.get_column("ticker").cast(pl.String).to_list(), panel.get_column("name").cast(pl.String).to_list(), strict=True))
        if "name" in panel.columns else {}
    )
    if panel.select(["ticker", "market"]).is_duplicated().any():
        raise ValueError("scoring panel has duplicate (ticker, market) rows")
    exchange_codes = set(panel.get_column("market").cast(pl.String).unique().to_list())
    if not exchange_codes or not exchange_codes <= KR_EXCHANGE_CODES:
        raise ValueError(f"scoring panel contains unsupported KR exchange codes: {sorted(exchange_codes - KR_EXCHANGE_CODES)}")
    selected = panel.select(["trade_date", "ticker", "market", *bundle.feature_columns]).sort("ticker")
    # The adopted rank profile is stateless. Reuse its frozen definition so
    # cross-sectional ties, null flags, and rank null fill match model training.
    from modeler.etl import preprocess as pp

    transformer = pp.fit(
        selected,
        pp.PreprocessConfig(profile="rank"),
        extra_exclude=(),
    )
    transformed = transformer.transform(selected)
    design = list(bundle.design_columns)
    if set(design) - set(transformed.columns):
        raise ValueError("bundle design columns are absent after the rank transform")
    matrix = transformed.select(design).to_numpy()
    if np.isinf(matrix).any():
        raise ValueError("scoring matrix contains infinity")
    scores = bundle.model.predict_proba(matrix)[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("KR model returned an invalid p_raw score")
    symbols = selected.get_column("ticker").cast(pl.String).to_list()
    names = [name_by_symbol.get(symbol) or symbol for symbol in symbols]
    rows = sorted(
        zip(symbols, names, scores.tolist(), strict=True),
        key=lambda row: (-row[2], row[0]),
    )
    rankings = [
        {"rank": rank, "symbol": symbol, "name": name, "score": float(score)}
        for rank, (symbol, name, score) in enumerate(rows, start=1)
    ]
    halted_col = "px_is_halted"
    halted = int(panel[halted_col].cast(pl.Boolean).fill_null(False).sum()) if halted_col in panel.columns else None
    quality = {
        "status": "review_required",
        "management_filter_available": False,
        "management_state": "unverified",
        "halted_rows_at_K": halted,
        "price_jump_review": "not_checked",
        "eligible_rows_scored": len(rankings),
        "excluded_balance_features": list(EXCLUDED_FEATURES),
    }
    report = {
        "schema_version": "1.0",
        "market": "KR",
        "report_date": report_date,
        "decision_at": _aware_decision_at(decision_at, report_date).isoformat(),
        "feature_asof_date": feature_asof_date,
        "status": "partial",
        "model_id": bundle.manifest["model_id"],
        "model_version": bundle.manifest["model_version"],
        "rankings": rankings,
        "quality": quality,
        "provenance": {
            "bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
            "prepared_manifest_sha256": hashlib.sha256(
                json.dumps(prepared_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "feature_snapshot_date": prepared_manifest.get("snapshot_date"),
            "source_model_run_id": bundle.manifest["source_model_run_id"],
            "target": "y_up_20d; p_raw ranking score, uncalibrated",
            "label_used_for_inference": False,
        },
        "publication": {
            "status": "unresolved",
            "evidence": ["원천·파생 피처의 공개 가능 조건을 확인하지 않았습니다."],
        },
    }
    validate_report(report)
    return report


def run_scoring(
    *, bundle_dir: Path, feature_dir: Path, report_date: str,
    decision_at: str, output_path: Path,
) -> dict:
    prepare_path = feature_dir / "prepare_manifest.json"
    panel_path = feature_dir / "feature_panel.parquet"
    prepared_manifest = json.loads(prepare_path.read_text(encoding="utf-8"))
    if not isinstance(prepared_manifest, dict):
        raise ValueError("prepared feature manifest must be an object")
    _require_decision_reached(decision_at, report_date)
    _validate_prepared_completion(prepared_manifest, report_date)
    expected_panel_hash = prepared_manifest.get("files", {}).get("feature_panel.parquet")
    if expected_panel_hash != _sha256(panel_path):
        raise ValueError("prepared feature panel hash does not match its manifest")
    bundle = load_bundle(bundle_dir, verify_golden=True)
    panel = pl.read_parquet(panel_path)
    payload = score_cross_section(
        bundle=bundle,
        panel=panel,
        report_date=report_date,
        decision_at=decision_at,
        feature_asof_date=str(prepared_manifest["feature_asof_date"]),
        prepared_manifest=prepared_manifest,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        validate_report(existing)
        same_input = (
            existing.get("report_date") == report_date
            and existing.get("provenance", {}).get("bundle_manifest_sha256")
            == payload["provenance"]["bundle_manifest_sha256"]
            and existing.get("provenance", {}).get("prepared_manifest_sha256")
            == payload["provenance"]["prepared_manifest_sha256"]
        )
        if same_input:
            return existing
        raise FileExistsError(f"KR report path contains a different completed input: {output_path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, output_path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--decision-at", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = run_scoring(
        bundle_dir=args.bundle_dir,
        feature_dir=args.feature_dir,
        report_date=args.report_date,
        decision_at=args.decision_at,
        output_path=args.output,
    )
    print(json.dumps({"rows": len(payload["rankings"]), "status": payload["status"], "publication": payload["publication"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
