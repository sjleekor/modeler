"""KR exploratory h20 model bundle training and verification.

The training path consumes an already frozen model-02 feature panel. It does
not run a search, evaluate a holdout, or read labels from the serving path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import polars as pl

from modeler.etl import preprocess as pp
from modeler.models._01_20_access_return_rank.train import design_columns
from modeler.models._02_updown_prob import features as fx
from modeler.models._02_updown_prob import train as updown_train

MODEL_ID = "kr_daily_h20_v1"
MODEL_VERSION = "1.0.0"
SOURCE_RUN_ID = "E2_h20_FS1h_seed0"
FEATURE_SET = "FS1h"
FLOW_VARIANT = "lag1"
PREPROCESS_PROFILE = "rank"
HORIZON = 20
SEED = 0
TRAIN_FORMATION_CUTOFF = "2025-07-31"
TRAIN_LABEL_TERMINAL_CUTOFF = "2025-07-31"
TARGET = "y_up_20d"
EXCLUDED_FEATURES = ("flow_short_balance_qty", "flow_short_balance_chg_20d")
# Frozen from the adopted E2 run. This is a one-point refit; no grid is run.
ADOPTED_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.03,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_label_end_dates(price_glob: str, *, horizon: int = HORIZON) -> pl.DataFrame:
    """Match the label builder's non-halt per-ticker horizon and its market peer set."""
    if horizon != HORIZON:
        raise ValueError(f"the serving model is fixed at h{HORIZON}, got h{horizon}")
    escaped = price_glob.replace("'", "''")
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    try:
        result = con.execute(
            f"""
            WITH px AS (
                SELECT trade_date, ticker, market,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, market ORDER BY trade_date
                       ) AS d_idx
                FROM read_parquet('{escaped}', hive_partitioning=true)
                WHERE NOT (open = 0 AND high = 0 AND low = 0)
            ), ends AS (
                SELECT a.trade_date, a.ticker, a.market,
                       f.trade_date AS label_end_date
                FROM px a
                JOIN px f
                  ON f.ticker = a.ticker AND f.market = a.market
                 AND f.d_idx = a.d_idx + {horizon}
            ), market_ends AS (
                SELECT trade_date, market, MAX(label_end_date) AS peer_label_end_date
                FROM ends
                GROUP BY trade_date, market
            )
            SELECT e.trade_date, e.ticker, e.market, e.label_end_date,
                   m.peer_label_end_date
            FROM ends e
            JOIN market_ends m USING (trade_date, market)
            """
        ).pl()
    finally:
        con.close()
    if result.is_empty():
        raise ValueError("daily_ohlcv에서 h20 label end date를 만들지 못했습니다")
    return result


def _feature_columns(panel: pl.DataFrame) -> list[str]:
    features = pp.feature_columns(panel)
    blocked = set(EXCLUDED_FEATURES)
    if blocked & set(features):
        features = [name for name in features if name not in blocked]
    if any(name in features for name in EXCLUDED_FEATURES):
        raise AssertionError("excluded balance features reached model inputs")
    return features


def _training_frame(
    panel_path: Path,
    label_ends_path: Path,
    *,
    columns: list[str] | None = None,
    formation_cutoff: str = TRAIN_FORMATION_CUTOFF,
    terminal_cutoff: str = TRAIN_LABEL_TERMINAL_CUTOFF,
) -> tuple[pl.DataFrame, dict[str, int]]:
    expected = {"trade_date", "ticker", "market", TARGET}
    read_columns = list(dict.fromkeys([*(columns or []), *expected]))
    panel = pl.read_parquet(panel_path, columns=read_columns if columns is not None else None)
    missing = expected - set(panel.columns)
    if missing:
        raise ValueError(f"frozen panel missing required columns: {sorted(missing)}")
    if "label_end_date" in panel.columns or "peer_label_end_date" in panel.columns:
        raise ValueError("panel must not supply precomputed terminal dates; use frozen prices")

    label_ends = pl.read_parquet(label_ends_path)
    required_ends = {"trade_date", "ticker", "market", "label_end_date", "peer_label_end_date"}
    if required_ends - set(label_ends.columns):
        raise ValueError("label end table does not satisfy the price-derived terminal contract")

    rows_before = panel.height
    panel = panel.filter(pl.col("trade_date") <= pl.lit(formation_cutoff).str.to_date())
    end_map = label_ends.filter(
        pl.col("label_end_date").is_not_null() & pl.col("peer_label_end_date").is_not_null()
    )
    panel = panel.join(end_map, on=["trade_date", "ticker", "market"], how="left")
    mature_mask = (
        pl.col(TARGET).is_not_null()
        & pl.col("label_end_date").is_not_null()
        & pl.col("peer_label_end_date").is_not_null()
        & (pl.col("label_end_date") <= pl.lit(terminal_cutoff).str.to_date())
        & (pl.col("peer_label_end_date") <= pl.lit(terminal_cutoff).str.to_date())
    )
    panel = panel.with_columns(mature_mask.alias("_training_mature")).drop(
        ["label_end_date", "peer_label_end_date"]
    )
    matured_count = panel.get_column("_training_mature").sum()
    if not matured_count:
        raise ValueError("development cutoffs leave no mature labeled rows")
    counts = {
        "panel_rows_before_filter": rows_before,
        "rows_after_formation_cutoff": panel.height,
        "matured_rows_used": int(matured_count),
        "rows_excluded_by_terminal_or_missing_label": panel.height - int(matured_count),
    }
    return panel, counts


def _model_input(
    panel: pl.DataFrame, declared_features: list[str]
) -> tuple[pl.DataFrame, list[str], list[str]]:
    blocked = set(EXCLUDED_FEATURES)
    features = [name for name in declared_features if name not in blocked]
    missing = set(features) - set(panel.columns)
    if missing:
        raise ValueError(f"frozen panel missing manifest features: {sorted(missing)}")
    if any(name in features for name in EXCLUDED_FEATURES):
        raise AssertionError("excluded balance features reached model inputs")
    # Rank each full formation cross-section before label maturity filtering.
    # This preserves the adopted FS1h rank transform when some rows have not
    # matured or do not have a resolved target.
    model_panel = panel.select(["trade_date", "ticker", "market", *features])
    fitted = pp.fit(model_panel, pp.PreprocessConfig(profile=PREPROCESS_PROFILE))
    transformed = fitted.transform(model_panel)
    design = design_columns(transformed, TARGET)
    if any(c in design for excluded in EXCLUDED_FEATURES for c in (excluded, f"{excluded}_isna")):
        raise AssertionError("excluded balance feature or missing flag reached model design")
    return transformed, design, features


def _input_matrix(frame: pl.DataFrame, design: list[str]) -> np.ndarray:
    return frame.select(design).to_numpy()


def train_bundle(
    *,
    panel_path: Path,
    label_ends_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    price_source: str,
    source_code_revision: str,
) -> dict[str, Any]:
    """Fit the fixed h20 estimator and atomically write a reproducible bundle."""
    panel_path = panel_path.resolve()
    label_ends_path = label_ends_path.resolve()
    dataset_manifest_path = dataset_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"bundle output already exists: {output_dir}")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("snapshot_date") != "2026-08-23":
        raise ValueError("training source must use frozen KR snapshot 2026-08-23")
    if dataset_manifest.get("period") != {"start": "2015-01-02", "end": TRAIN_FORMATION_CUTOFF}:
        raise ValueError("training source does not match the frozen development formation boundary")
    extra = dataset_manifest.get("extra", {})
    if (extra.get("feature_set"), extra.get("flow_variant"), extra.get("preprocess_profile"), extra.get("horizon")) != (
        FEATURE_SET, FLOW_VARIANT, PREPROCESS_PROFILE, HORIZON
    ):
        raise ValueError("training source does not match E2_h20_FS1h_seed0 feature config")

    declared_features = extra.get("feature_columns")
    if not isinstance(declared_features, list) or not declared_features:
        raise ValueError("frozen dataset manifest has no declared feature columns")
    train_rows, row_counts = _training_frame(
        panel_path, label_ends_path, columns=declared_features
    )
    mature_mask = train_rows.get_column("_training_mature").to_numpy()
    fit_keys = train_rows.filter(pl.col("_training_mature")).select(
        ["trade_date", "ticker", "market", TARGET]
    )
    y = fit_keys.get_column(TARGET).cast(pl.Int8).to_numpy()
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("fixed classifier requires both target classes in the matured training rows")
    last_date = fit_keys.get_column("trade_date").max()
    del fit_keys
    transformed, design, feature_columns = _model_input(train_rows, declared_features)
    # The feature-only training frame is enough after rank transformation;
    # release the joined label/maturity frame before materializing fit rows.
    del train_rows
    golden = transformed.filter(pl.col("trade_date") == last_date).sort("ticker")
    golden_input = golden.select(design).to_numpy(order="c")
    golden_symbols = golden.get_column("ticker").to_list()
    fit_frame = transformed.filter(pl.Series("_training_mature", mature_mask))
    del transformed, golden
    matrix = fit_frame.select(design).to_numpy(order="c")
    del fit_frame
    train_config = updown_train.TrainConfig(
        model="hgb_clf", target="y_up", horizon=HORIZON,
        grid=(dict(ADOPTED_PARAMS),), seed=SEED, calibrate="none", monotonic=False,
    )
    if np.isinf(matrix).any():
        raise ValueError("training feature matrix contains infinity")
    if not matrix.flags.c_contiguous:
        raise ValueError("training feature matrix must be C-contiguous")
    model = updown_train.make_model(train_config, dict(ADOPTED_PARAMS), design)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        model.fit(matrix, y)

    golden_scores = model.predict_proba(golden_input)[:, 1]
    ranked = sorted(zip(golden_symbols, golden_scores.tolist()), key=lambda x: (-x[1], x[0]))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        model_path = temp_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        golden_payload = {
            "feature_asof_date": str(last_date),
            "design_columns": design,
            "input": golden_input.tolist(),
            "symbols": golden_symbols,
            "expected_p_raw": golden_scores.tolist(),
            "expected_order": [symbol for symbol, _ in ranked],
        }
        (temp_dir / "golden.json").write_text(
            json.dumps(golden_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "schema_version": "kr-model-bundle.v1",
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "market": "KR",
            "source_model_run_id": SOURCE_RUN_ID,
            "estimator": {"family": "hgb_clf", "target": TARGET, "params": ADOPTED_PARAMS},
            "calibration": "none",
            "score_semantics": "p_raw; ranking score only, not calibrated probability or confidence",
            "feature_contract": {
                "feature_set": FEATURE_SET,
                "flow_variant": FLOW_VARIANT,
                "preprocess_profile": PREPROCESS_PROFILE,
                "feature_columns": feature_columns,
                "design_columns": design,
                "excluded_features": list(EXCLUDED_FEATURES),
                "horizon_sessions": HORIZON,
            },
            "universe": dataset_manifest.get("universe_filter"),
            "training": {
                "snapshot_date": dataset_manifest["snapshot_date"],
                "source": dataset_manifest.get("lake", {}).get("raw"),
                "formation_start": dataset_manifest["period"]["start"],
                "formation_cutoff": TRAIN_FORMATION_CUTOFF,
                "label_terminal_cutoff": TRAIN_LABEL_TERMINAL_CUTOFF,
                "label_maturity_rule": "own h20 terminal date and max same-date/same-market benchmark peer terminal date <= cutoff",
                "label_boundary_source": price_source,
            "rows": row_counts,
                "seed": SEED,
                "grid_search": False,
                "holdout_evaluation": False,
            },
            "runtime": {
                "source_code_revision": source_code_revision,
                "python": os.sys.version.split()[0],
                "numpy": importlib.metadata.version("numpy"),
                "polars": importlib.metadata.version("polars"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
            },
            "created_at": datetime.now(UTC).isoformat(),
            "files": {
                "model.joblib": sha256_file(model_path),
                "golden.json": sha256_file(temp_dir / "golden.json"),
            },
            "input_files": {
                "source_panel_sha256": sha256_file(panel_path),
                "label_end_table_sha256": sha256_file(label_ends_path),
                "source_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            },
        }
        manifest["manifest_sha256"] = _hash_json(manifest)
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        temp_dir.rename(output_dir)
    except Exception:
        for path in temp_dir.iterdir():
            path.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise
    return manifest


@dataclass(frozen=True)
class LoadedBundle:
    directory: Path
    manifest: dict[str, Any]
    model: Any
    golden: dict[str, Any]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(self.manifest["feature_contract"]["feature_columns"])

    @property
    def design_columns(self) -> tuple[str, ...]:
        return tuple(self.manifest["feature_contract"]["design_columns"])


def load_bundle(directory: Path, *, verify_golden: bool = True) -> LoadedBundle:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = manifest.pop("manifest_sha256", None)
    if stored_hash != _hash_json(manifest):
        raise ValueError("model bundle manifest hash mismatch")
    manifest["manifest_sha256"] = stored_hash
    _validate_bundle_contract(manifest)
    for name, expected_hash in manifest.get("files", {}).items():
        if sha256_file(directory / name) != expected_hash:
            raise ValueError(f"model bundle file hash mismatch: {name}")
    golden = json.loads((directory / "golden.json").read_text(encoding="utf-8"))
    model = joblib.load(directory / "model.joblib")
    bundle = LoadedBundle(directory=directory, manifest=manifest, model=model, golden=golden)
    if verify_golden:
        verify_bundle_golden(bundle)
    return bundle


def _validate_bundle_contract(manifest: dict[str, Any]) -> None:
    """Reject a hash-valid bundle that does not match the adopted KR contract."""
    if manifest.get("schema_version") != "kr-model-bundle.v1":
        raise ValueError("unsupported KR model bundle schema")
    if (manifest.get("model_id"), manifest.get("model_version"), manifest.get("market")) != (
        MODEL_ID, MODEL_VERSION, "KR"
    ):
        raise ValueError("model bundle identity differs from the fixed KR model")
    if manifest.get("source_model_run_id") != SOURCE_RUN_ID:
        raise ValueError("model bundle source run differs from the adopted E2 run")
    if set(manifest.get("files", {})) != {"model.joblib", "golden.json"}:
        raise ValueError("model bundle files must be exactly model.joblib and golden.json")
    estimator = manifest.get("estimator", {})
    if estimator != {"family": "hgb_clf", "target": TARGET, "params": ADOPTED_PARAMS}:
        raise ValueError("model bundle estimator differs from the adopted E2 configuration")
    if manifest.get("calibration") != "none":
        raise ValueError("KR serving bundle must not use calibration")
    if manifest.get("score_semantics") != "p_raw; ranking score only, not calibrated probability or confidence":
        raise ValueError("model bundle score semantics differ from the KR ranking contract")
    feature_contract = manifest.get("feature_contract", {})
    expected_features = [
        column for column in fx.feature_columns(FEATURE_SET, HORIZON)
        if column not in EXCLUDED_FEATURES
    ]
    expected_design = [*expected_features, *(f"{column}_isna" for column in expected_features)]
    if (
        feature_contract.get("feature_set"),
        feature_contract.get("flow_variant"),
        feature_contract.get("preprocess_profile"),
        feature_contract.get("horizon_sessions"),
        feature_contract.get("excluded_features"),
        feature_contract.get("feature_columns"),
        feature_contract.get("design_columns"),
    ) != (
        FEATURE_SET, FLOW_VARIANT, PREPROCESS_PROFILE, HORIZON,
        list(EXCLUDED_FEATURES), expected_features, expected_design,
    ):
        raise ValueError("model bundle feature/design contract differs from fixed KR h20 spec")
    training = manifest.get("training", {})
    if (
        training.get("formation_cutoff"), training.get("label_terminal_cutoff"),
        training.get("grid_search"), training.get("holdout_evaluation"),
    ) != (TRAIN_FORMATION_CUTOFF, TRAIN_LABEL_TERMINAL_CUTOFF, False, False):
        raise ValueError("model bundle training boundaries differ from fixed development cutoffs")


def verify_bundle_golden(bundle: LoadedBundle) -> None:
    design = list(bundle.design_columns)
    if design != bundle.golden.get("design_columns"):
        raise ValueError("golden design schema differs from bundle")
    matrix = np.asarray(bundle.golden["input"], dtype=float)
    predictions = bundle.model.predict_proba(matrix)[:, 1]
    expected = np.asarray(bundle.golden["expected_p_raw"], dtype=float)
    if predictions.shape != expected.shape or not np.allclose(predictions, expected, rtol=0.0, atol=1e-12):
        raise ValueError("reloaded KR model does not reproduce golden predictions")
    scores = dict(zip(bundle.golden["symbols"], predictions.tolist(), strict=True))
    order = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    if order != bundle.golden["expected_order"]:
        raise ValueError("reloaded KR model does not reproduce golden ranking")
