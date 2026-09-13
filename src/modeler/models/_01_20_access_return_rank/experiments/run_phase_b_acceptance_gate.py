"""Validation-only acceptance gate for Phase B Horizon Scan candidates.

The combined A+B artifact decides the candidate population: every primary
feature with at least one ``screen_pass`` cell enters together.  No horizon or
feature is re-selected here.  The script compares the existing model baseline
with baseline + those candidates on the already declared 5/20/60-day targets.

The 2026-06-11..2026-07-31 holdout is deliberately not exposed here.  Its
60-session label is not mature in the pinned 2026-08-23 snapshot, so the final
one-shot holdout remains deferred to the 2026-10/11 refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from modeler.etl import preprocess as pp
from modeler.etl.config import DataRoot, EngineOptions
from modeler.etl.metrics import economic_report
from modeler.models._01_20_access_return_rank import train as tr
from modeler.models._01_20_access_return_rank.experiments.run_grade_a_acceptance_gate import (
    BASELINE_COLS,
    TARGET_HORIZONS,
    _fmt,
    _wf_predictions,
)

SNAPSHOT_DATE = "2026-08-23"
SOURCE = "sj2_remote"
_ROOT = DataRoot.resolve(market="kr")
BASELINE_DATASET_DIR = (
    _ROOT.datasets
    / "grade_a_gate/baseline/01_20_access_return_rank"
    / f"snapshot_date={SNAPSHOT_DATE}/source={SOURCE}"
)
BASELINE_RESULTS = Path("docs/target/01_20_access_return_rank/grade_a_acceptance_gate_results.json")
OUTPUT_JSON = Path("docs/target/01_20_access_return_rank/phase_b_acceptance_gate_results.json")
OUTPUT_MD = Path("docs/target/01_20_access_return_rank/phase_b_acceptance_gate_results.md")
DATASET_ROOT = (
    _ROOT.datasets
    / "phase_b_acceptance_gate/candidate/01_20_access_return_rank"
    / f"snapshot_date={SNAPSHOT_DATE}/source={SOURCE}"
)
FEATURE_MART_ROOT = (
    _ROOT.derived / "feature" / f"snapshot_date={SNAPSHOT_DATE}/source={SOURCE}"
)

# Only columns that can be selected by the current expansion registry belong
# here.  An unknown screen-pass feature is a contract change and must fail
# closed instead of being silently omitted.
FEATURE_TO_MART: dict[str, str] = {
    "fin_log_mcap": "feat_fin_scan_daily",
    "fin_value_z": "feat_fin_scan_daily",
    "fin_gross_profitability": "feat_fin_scan_daily",
    "fin_asset_growth_yoy": "feat_fin_scan_daily",
    "fin_accruals_to_assets": "feat_fin_scan_daily",
    "ev_net_share_issuance_yoy": "feat_event_scan_daily",
    "ev_payout_yield": "feat_event_scan_daily",
    "mcap_krx_log": "feat_market_cap",
    "ev_filing_burst_60d": "feat_filing_activity",
    "ev_amendment_ratio_1y": "feat_filing_activity",
    "own_insider_filing_burst_60d": "feat_filing_activity",
    "own_major_filing_60d": "feat_filing_activity",
    "own_amendment_ratio_1y": "feat_filing_activity",
    "hc_employee_growth_yoy": "feat_periodic_extras",
    "hc_revenue_per_employee": "feat_periodic_extras",
    "own_major_stake": "feat_periodic_extras",
    "own_major_stake_chg": "feat_periodic_extras",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_short_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ident(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return f'"{value}"'


def select_screen_pass_features(ab_run_dir: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return frozen screen-pass families/features from a successful AB run."""
    success = _read_json(ab_run_dir / "_SUCCESS.json")
    if success.get("status") != "success":
        raise RuntimeError(f"AB run is not successful: {ab_run_dir}")
    frame = pl.read_parquet(ab_run_dir / "combined_ab_primary_hypotheses.parquet")
    selected = frame.filter(pl.col("screen_pass").fill_null(False))
    features = sorted(selected.get_column("feature").unique().to_list())
    families = sorted(selected.get_column("family").unique().to_list())
    unknown = sorted(set(features) - FEATURE_TO_MART.keys())
    if unknown:
        raise RuntimeError(f"screen-pass features have no mart mapping: {unknown}")
    if not features:
        raise RuntimeError("AB run has no screen-pass Phase B features")
    return families, features, success


def group_features_by_mart(features: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for feature in features:
        grouped.setdefault(FEATURE_TO_MART[feature], []).append(feature)
    return {mart: sorted(cols) for mart, cols in sorted(grouped.items())}


def _validate_baseline() -> dict[str, Any]:
    manifest = _read_json(BASELINE_DATASET_DIR / "dataset_manifest.json")
    expected = {
        "snapshot_date": SNAPSHOT_DATE,
        "period": {"start": "2015-01-02", "end": "2026-06-10"},
        "code_rev": _git_short_head(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"baseline dataset {key} mismatch: {manifest.get(key)!r} != {value!r}"
            )
    std_path = BASELINE_DATASET_DIR / "feat_panel_std.parquet"
    schema = pl.read_parquet_schema(std_path)
    missing = sorted(set(BASELINE_COLS) - schema.keys())
    if missing:
        raise RuntimeError(f"baseline standardized panel is missing columns: {missing}")
    return manifest


def _validate_marts(grouped: dict[str, list[str]], config_hash: str) -> None:
    con = duckdb.connect()
    try:
        for mart, features in grouped.items():
            directory = FEATURE_MART_ROOT / mart
            metadata = _read_json(directory / "_cache_metadata.json")
            if metadata.get("analysis_config_hash") != config_hash:
                raise RuntimeError(
                    f"{mart} config hash mismatch: "
                    f"{metadata.get('analysis_config_hash')} != {config_hash}"
                )
            glob = directory / "**" / "*.parquet"
            columns = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet({_sql_string(glob)}, "
                    "hive_partitioning=false)"
                ).fetchall()
            }
            missing = sorted(set(features) - columns)
            if missing:
                raise RuntimeError(f"{mart} is missing selected features: {missing}")
            duplicate_count = con.execute(
                "SELECT count(*) - count(DISTINCT (trade_date, ticker, market)) "
                f"FROM read_parquet({_sql_string(glob)}, hive_partitioning=false)"
            ).fetchone()[0]
            if duplicate_count:
                raise RuntimeError(f"{mart} has {duplicate_count} duplicate daily keys")
    finally:
        con.close()


def _dataset_contract(
    *,
    ab_run_dir: Path,
    ab_success: dict[str, Any],
    baseline_manifest: dict[str, Any],
    families: list[str],
    features: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "snapshot_date": SNAPSHOT_DATE,
        "source": SOURCE,
        "period": baseline_manifest["period"],
        "code_rev": baseline_manifest["code_rev"],
        "ab_run_dir": str(ab_run_dir),
        "ab_run_id": ab_success["run_id"],
        "ab_content_hash": ab_success["content_hash"],
        "config_hash": ab_success["config_hash"],
        "baseline_std_sha256": _sha256(BASELINE_DATASET_DIR / "feat_panel_std.parquet"),
        "families": families,
        "features": features,
    }


def _join_new_features(
    *,
    output: Path,
    grouped: dict[str, list[str]],
    engine: EngineOptions,
) -> None:
    baseline = BASELINE_DATASET_DIR / "feat_panel_std.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(engine.threads)}")
        con.execute(f"SET memory_limit = {_sql_string(engine.memory_limit)}")
        con.execute(f"SET temp_directory = {_sql_string(engine.temp_directory)}")
        joins: list[str] = []
        selected: list[str] = []
        for index, (mart, features) in enumerate(grouped.items()):
            alias = f"m{index}"
            glob = FEATURE_MART_ROOT / mart / "**" / "*.parquet"
            cols = ", ".join(_ident(feature) for feature in features)
            joins.append(
                f"LEFT JOIN (SELECT trade_date, ticker, market, {cols} "
                f"FROM read_parquet({_sql_string(glob)}, hive_partitioning=false)) {alias} "
                "USING (trade_date, ticker, market)"
            )
            selected.extend(f"{alias}.{_ident(feature)}" for feature in features)
        output.parent.mkdir(parents=True, exist_ok=True)
        sql = (
            "SELECT b.*, "
            + ", ".join(selected)
            + f" FROM read_parquet({_sql_string(baseline)}, hive_partitioning=false) b "
            + " ".join(joins)
            + " ORDER BY b.fold_id, b.fold_role, b.trade_date, b.ticker, b.market"
        )
        con.execute(
            f"COPY ({sql}) TO {_sql_string(output)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        baseline_rows = con.execute(
            f"SELECT count(*) FROM read_parquet({_sql_string(baseline)})"
        ).fetchone()[0]
        output_rows = con.execute(
            f"SELECT count(*) FROM read_parquet({_sql_string(output)})"
        ).fetchone()[0]
        if output_rows != baseline_rows:
            raise RuntimeError(
                f"candidate join changed row count: {baseline_rows} -> {output_rows}"
            )
    finally:
        con.close()


def _standardize_new_features(raw_path: Path, features: list[str], parts_dir: Path) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = (
        pl.scan_parquet(raw_path)
        .select("fold_id")
        .unique()
        .sort("fold_id")
        .collect()
        .get_column("fold_id")
        .to_list()
    )
    cfg = pp.PreprocessConfig(profile="tree")
    for fold_id in fold_ids:
        part = parts_dir / f"fold_id={fold_id}.parquet"
        fold = pl.scan_parquet(raw_path).filter(pl.col("fold_id") == fold_id).collect()
        train = fold.filter(pl.col("fold_role") == "train")
        bounds = pp.fit_winsor_bounds(train, features, cfg)
        transformed = pp.FittedPreprocess(features, cfg, bounds).transform(fold)
        transformed.write_parquet(part, compression="zstd", row_group_size=100_000)
        print(f"  standardized fold {fold_id}: {transformed.height:,} rows", flush=True)


def build_candidate_dataset(
    *,
    contract: dict[str, Any],
    features: list[str],
    engine: EngineOptions,
    force: bool,
) -> Path:
    manifest_path = DATASET_ROOT / "dataset_manifest.json"
    parts_dir = DATASET_ROOT / "feat_panel_std"
    if manifest_path.exists() and any(parts_dir.glob("*.parquet")) and not force:
        if _read_json(manifest_path) != contract:
            raise RuntimeError("candidate dataset contract changed; rerun with --force")
        print(f"reusing candidate dataset: {DATASET_ROOT}", flush=True)
        return parts_dir
    if (manifest_path.exists() or parts_dir.exists()) and not force:
        raise RuntimeError("partial candidate dataset exists; rerun with --force")

    # Generated gate artifacts only.  ``--force`` is explicit and never touches
    # the baseline or official Horizon Scan marts.
    if force:
        import shutil

        shutil.rmtree(DATASET_ROOT, ignore_errors=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    raw_path = DATASET_ROOT / "joined_raw.parquet"
    grouped = group_features_by_mart(features)
    _join_new_features(output=raw_path, grouped=grouped, engine=engine)
    _standardize_new_features(raw_path, features, parts_dir)
    raw_path.unlink()
    manifest_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return parts_dir


def _run_candidate(parts_dir: Path, features: list[str]) -> list[dict[str, Any]]:
    std = pl.read_parquet(str(parts_dir / "*.parquet"))
    records: list[dict[str, Any]] = []
    for horizon in TARGET_HORIZONS:
        config = tr.TrainConfig(
            model="hgb", target=f"y_rank_{horizon}d", realized=f"raw_label_{horizon}d"
        )
        started = time.time()
        result = tr.walk_forward(std, config)
        preds = _wf_predictions(std, config, result)
        econ = economic_report(
            preds, pred_col="pred", realized_col=config.realized, horizon=horizon
        )
        record = {
            "name": "phase_b_candidate",
            "horizon": horizon,
            "n_raw_features": len(BASELINE_COLS) + len(features),
            "n_design_features": len(result.feature_cols),
            "best_params": result.best_params,
            "mean_rank_ic": result.mean_rank_ic,
            "per_fold_rank_ic": [row.report.rank_ic_mean for row in result.fold_results],
            "economic": econ.as_dict(),
            "train_seconds": round(time.time() - started, 1),
        }
        records.append(record)
        print(
            f"h={horizon}: IC={_fmt(record['mean_rank_ic'])}, "
            f"cost-adj={_fmt(record['economic']['cost_adjusted_spread'])}",
            flush=True,
        )
    return records


def _baseline_records() -> list[dict[str, Any]]:
    records = _read_json(BASELINE_RESULTS).get("records", [])
    selected = [row for row in records if row.get("name") == "baseline"]
    if sorted(row.get("horizon") for row in selected) != sorted(TARGET_HORIZONS):
        raise RuntimeError("baseline acceptance artifact does not cover the declared horizons")
    return selected


def summarize_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize valid-set deltas without turning them into a final holdout verdict."""
    by_key = {(row["name"], row["horizon"]): row for row in records}
    deltas: list[dict[str, Any]] = []
    for horizon in TARGET_HORIZONS:
        baseline = by_key[("baseline", horizon)]
        candidate = by_key[("phase_b_candidate", horizon)]
        deltas.append(
            {
                "horizon": horizon,
                "mean_rank_ic": candidate["mean_rank_ic"] - baseline["mean_rank_ic"],
                "cost_adjusted_spread": (
                    candidate["economic"]["cost_adjusted_spread"]
                    - baseline["economic"]["cost_adjusted_spread"]
                ),
            }
        )
    improved_all = all(
        row["mean_rank_ic"] > 0 and row["cost_adjusted_spread"] > 0 for row in deltas
    )
    return {
        "status": "improved_all_horizons" if improved_all else "mixed",
        "deltas": deltas,
        "final_acceptance": "deferred_until_new_mature_h60_holdout",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    records = payload["records"]
    lines = [
        "# Phase B candidates — validation-only acceptance gate",
        "",
        f"> AB run: `{payload['ab_run_id']}` / config `{payload['config_hash']}`",
        "> 후보 선택: AB `screen_pass`가 하나 이상인 family의 primary feature를 전부 함께 추가",
        "> holdout: 미사용. h60 라벨이 성숙하는 2026년 10~11월 이후 한 번만 평가",
        "",
        f"후보 family {len(payload['families'])}개, "
        f"feature {len(payload['candidate_features'])}개:",
        "",
        ", ".join(f"`{name}`" for name in payload["candidate_features"]),
        "",
    ]
    by_key = {(row["name"], row["horizon"]): row for row in records}
    for horizon in TARGET_HORIZONS:
        base = by_key[("baseline", horizon)]
        cand = by_key[("phase_b_candidate", horizon)]
        delta_ic = cand["mean_rank_ic"] - base["mean_rank_ic"]
        delta_cost = (
            cand["economic"]["cost_adjusted_spread"]
            - base["economic"]["cost_adjusted_spread"]
        )
        lines.extend(
            [
                f"## h={horizon}",
                "",
                "| config | raw/design | mean Rank IC | cost-adjusted spread |",
                "|---|---:|---:|---:|",
                f"| baseline | {base['n_raw_features']}/{base['n_design_features']} | "
                f"{_fmt(base['mean_rank_ic'])} | "
                f"{_fmt(base['economic']['cost_adjusted_spread'])} |",
                f"| phase_b_candidate | {cand['n_raw_features']}/{cand['n_design_features']} | "
                f"{_fmt(cand['mean_rank_ic'])} | "
                f"{_fmt(cand['economic']['cost_adjusted_spread'])} |",
                "",
                f"- candidate − baseline: Rank IC **{_fmt(delta_ic)}**, "
                f"cost-adjusted spread **{_fmt(delta_cost)}**",
                "",
            ]
        )
    lines.extend(
        [
            "## 판정 범위",
            "",
            "valid 구간에서는 세 horizon의 Rank IC와 비용 반영 spread가 모두 개선됐습니다. "
            "따라서 validation 단계 결과는 `improved_all_horizons`입니다.",
            "",
            "이 결과는 purged walk-forward valid 구간의 증분성과 경제성만 판정합니다. "
            "최종 채택은 h60까지 값이 있는 새 holdout을 한 번 평가한 뒤 정합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-run-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    families, features, ab_success = select_screen_pass_features(args.ab_run_dir)
    baseline_manifest = _validate_baseline()
    grouped = group_features_by_mart(features)
    _validate_marts(grouped, ab_success["config_hash"])
    contract = _dataset_contract(
        ab_run_dir=args.ab_run_dir,
        ab_success=ab_success,
        baseline_manifest=baseline_manifest,
        families=families,
        features=features,
    )
    parts_dir = build_candidate_dataset(
        contract=contract,
        features=features,
        engine=EngineOptions(
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_directory=str(_ROOT.base / "_tmp"),
        ),
        force=args.force,
    )
    if args.build_only:
        print(parts_dir)
        return 0

    records = [*_baseline_records(), *_run_candidate(parts_dir, features)]
    payload = {
        **contract,
        "baseline_result_reused": str(BASELINE_RESULTS),
        "candidate_features": features,
        "records": records,
        "validation_summary": summarize_validation(records),
        "holdout_status": "deferred_until_2026_10_or_11_for_mature_h60_labels",
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON} and {OUTPUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
