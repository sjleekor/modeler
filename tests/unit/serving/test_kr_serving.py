from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.serving.kr_model import (
    ADOPTED_PARAMS,
    EXCLUDED_FEATURES,
    LoadedBundle,
    _model_input,
    _training_frame,
    _validate_bundle_contract,
    load_bundle,
    train_bundle,
)
from modeler.serving.kr_prepare import KrBriefingSpec
from modeler.serving.kr_serving import (
    _require_decision_reached,
    _validate_completion_evidence,
    _validate_prepared_completion,
    score_cross_section,
)


def test_kr_briefing_spec_excludes_both_short_balance_columns() -> None:
    columns = KrBriefingSpec().feature_columns(20)
    assert len(columns) == 47
    assert not set(EXCLUDED_FEATURES) & set(columns)


def _valid_bundle_manifest() -> dict:
    features = list(KrBriefingSpec().feature_columns(20))
    return {
        "schema_version": "kr-model-bundle.v1",
        "model_id": "kr_daily_h20_v1",
        "model_version": "1.0.0",
        "market": "KR",
        "source_model_run_id": "E2_h20_FS1h_seed0",
        "score_semantics": "p_raw; ranking score only, not calibrated probability or confidence",
        "files": {"model.joblib": "model-hash", "golden.json": "golden-hash"},
        "estimator": {"family": "hgb_clf", "target": "y_up_20d", "params": dict(ADOPTED_PARAMS)},
        "calibration": "none",
        "feature_contract": {
            "feature_set": "FS1h",
            "flow_variant": "lag1",
            "preprocess_profile": "rank",
            "horizon_sessions": 20,
            "excluded_features": list(EXCLUDED_FEATURES),
            "feature_columns": features,
            "design_columns": [*features, *(f"{name}_isna" for name in features)],
        },
        "training": {
            "formation_cutoff": "2025-07-31",
            "label_terminal_cutoff": "2025-07-31",
            "grid_search": False,
            "holdout_evaluation": False,
        },
    }


def test_hash_valid_bundle_must_match_full_fixed_model_contract() -> None:
    _validate_bundle_contract(_valid_bundle_manifest())
    mutations = [
        ("model_id", "other-model"),
        ("files", {"../outside.joblib": "hash", "golden.json": "hash"}),
        ("calibration", "isotonic"),
    ]
    for key, value in mutations:
        manifest = _valid_bundle_manifest()
        manifest[key] = value
        with pytest.raises(ValueError):
            _validate_bundle_contract(manifest)

    for key, value in (
        ("horizon_sessions", 60),
        ("preprocess_profile", "tree"),
        ("excluded_features", []),
        ("design_columns", ["px_ret_1d", "px_ret_1d"]),
    ):
        manifest = _valid_bundle_manifest()
        manifest["feature_contract"][key] = value
        with pytest.raises(ValueError):
            _validate_bundle_contract(manifest)

    manifest = _valid_bundle_manifest()
    manifest["training"]["label_terminal_cutoff"] = "2025-08-01"
    with pytest.raises(ValueError, match="training boundaries"):
        _validate_bundle_contract(manifest)


def test_training_rank_transform_uses_full_cross_section_before_maturity() -> None:
    panel = pl.DataFrame(
        {
            "trade_date": [date(2025, 7, 30)] * 3,
            "ticker": ["000001", "000002", "000003"],
            "market": ["KOSPI"] * 3,
            "px_ret_1d": [1.0, 2.0, 3.0],
            "_training_mature": [True, True, False],
        },
        schema_overrides={"trade_date": pl.Date},
    )
    transformed, design, features = _model_input(panel, ["px_ret_1d"])
    assert features == ["px_ret_1d"]
    assert design == ["px_ret_1d", "px_ret_1d_isna"]
    assert transformed.get_column("px_ret_1d").to_list() == [0.0, 0.5, 1.0]


def test_training_cutoff_requires_both_own_and_market_peer_terminal(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    ends_path = tmp_path / "ends.parquet"
    dates = [date(2025, 7, 30), date(2025, 7, 31)]
    panel = pl.DataFrame(
        {
            "trade_date": dates,
            "ticker": ["000001", "000001"],
            "market": ["KOSPI", "KOSPI"],
            "y_up_20d": [1, 1],
        },
        schema_overrides={"trade_date": pl.Date},
    )
    ends = pl.DataFrame(
        {
            "trade_date": dates,
            "ticker": ["000001", "000001"],
            "market": ["KOSPI", "KOSPI"],
            "label_end_date": [date(2025, 7, 31), date(2025, 7, 31)],
            "peer_label_end_date": [date(2025, 7, 31), date(2025, 8, 1)],
        },
        schema_overrides={
            "trade_date": pl.Date,
            "label_end_date": pl.Date,
            "peer_label_end_date": pl.Date,
        },
    )
    panel.write_parquet(panel_path)
    ends.write_parquet(ends_path)
    rows, counts = _training_frame(panel_path, ends_path)
    assert rows.get_column("_training_mature").to_list() == [True, False]
    assert counts["matured_rows_used"] == 1


def test_tiny_fixed_train_bundle_fits_and_reloads_golden(tmp_path: Path) -> None:
    features = list(KrBriefingSpec().feature_columns(20))
    dates = [date(2025, 7, 29)] * 3 + [date(2025, 7, 30)] * 3
    tickers = ["000001", "000002", "000003"] * 2
    panel_values = {
        "trade_date": dates,
        "ticker": tickers,
        "market": ["KOSPI", "KOSPI", "KOSDAQ"] * 2,
        "y_up_20d": [0, 1, 0, 1, 0, 1],
    }
    for index, feature in enumerate(features, start=1):
        panel_values[feature] = [float(index * row) for row in (1, 2, 3, 2, 3, 4)]
    panel_path = tmp_path / "panel.parquet"
    label_ends_path = tmp_path / "label_ends.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"
    output_dir = tmp_path / "bundle"
    pl.DataFrame(panel_values, schema_overrides={"trade_date": pl.Date}).write_parquet(panel_path)
    pl.DataFrame(
        {
            "trade_date": dates,
            "ticker": tickers,
            "market": ["KOSPI", "KOSPI", "KOSDAQ"] * 2,
            "label_end_date": [date(2025, 7, 31)] * 6,
            "peer_label_end_date": [date(2025, 7, 31)] * 6,
        },
        schema_overrides={"trade_date": pl.Date, "label_end_date": pl.Date,
                          "peer_label_end_date": pl.Date},
    ).write_parquet(label_ends_path)
    manifest_path.write_text(
        json.dumps({
            "snapshot_date": "2026-08-23",
            "period": {"start": "2015-01-02", "end": "2025-07-31"},
            "extra": {"feature_set": "FS1h", "flow_variant": "lag1",
                      "preprocess_profile": "rank", "horizon": 20,
                      "feature_columns": features},
            "lake": {"raw": "sj2_remote"},
            "universe_filter": {"membership_reconstruction_available": False},
        }),
        encoding="utf-8",
    )

    bundle_manifest = train_bundle(
        panel_path=panel_path,
        label_ends_path=label_ends_path,
        dataset_manifest_path=manifest_path,
        output_dir=output_dir,
        price_source="tiny test fixture only",
        source_code_revision="test-fixture",
    )
    assert bundle_manifest["training"]["rows"]["matured_rows_used"] == 6
    assert bundle_manifest["training"]["label_terminal_cutoff"] == "2025-07-31"
    loaded = load_bundle(output_dir, verify_golden=True)
    assert loaded.golden["feature_asof_date"] == "2025-07-30"


class _TinyRankModel:
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        score = matrix[:, 0]
        return np.column_stack((1.0 - score, score))


def _bundle() -> LoadedBundle:
    return LoadedBundle(
        directory=Path("/unused"),
        manifest={
            "manifest_sha256": "bundle-hash",
            "model_id": "kr_daily_h20_v1",
            "model_version": "1.0.0",
            "source_model_run_id": "E2_h20_FS1h_seed0",
            "feature_contract": {
                "feature_columns": ["px_ret_1d"],
                "design_columns": ["px_ret_1d", "px_ret_1d_isna"],
            },
        },
        model=_TinyRankModel(),
        golden={},
    )


def test_scoring_is_label_free_and_emits_common_report_contract() -> None:
    panel = pl.DataFrame(
        {
            "trade_date": [date(2026, 9, 28)] * 3,
            "ticker": ["000001", "000002", "000003"],
            "market": ["KOSPI"] * 3,
            "name": ["A", "B", "C"],
            "px_ret_1d": [1.0, 2.0, 3.0],
            "flow_short_balance_qty": [100, 200, 300],
        },
        schema_overrides={"trade_date": pl.Date},
    )
    prepared = {
        "market": "KR",
        "snapshot_date": "2026-09-28",
        "feature_asof_date": "2026-09-28",
        "model_config": {
            "feature_set": "FS1h",
            "flow_variant": "lag1",
            "preprocess_profile": "rank",
            "excluded_features": list(EXCLUDED_FEATURES),
            "feature_time_contract": {"serving_applies_additional_lag": False},
        },
    }
    bundle = _bundle()
    report = score_cross_section(
        bundle=bundle,
        panel=panel,
        report_date="2026-09-29",
        decision_at="2026-09-29T10:00:00+09:00",
        feature_asof_date="2026-09-28",
        prepared_manifest=prepared,
    )
    assert report["status"] == "partial"
    assert report["publication"]["status"] == "unresolved"
    assert report["provenance"]["label_used_for_inference"] is False
    assert [row["symbol"] for row in report["rankings"]] == ["000003", "000002", "000001"]
    assert all(row["name"] for row in report["rankings"])
    assert all("score" in row for row in report["rankings"])


def test_top100_quality_flags_keep_original_ranks_and_prices_at_K() -> None:
    panel = pl.DataFrame(
        {
            "trade_date": [date(2026, 9, 28)] * 3,
            "ticker": ["000001", "000002", "000003"],
            "market": ["KOSPI"] * 3,
            "px_ret_1d": [1.0, 2.0, 3.0],
            "is_halted": [False, False, False],
            "ca_price_jump_suspect": [False, True, False],
            "ca_share_change_confirmed": [False, False, False],
            "ca_rule_applicability_unknown": [False, False, False],
            "simple_ret": [0.01, 0.35, 0.02],
        },
        schema_overrides={"trade_date": pl.Date},
    )
    prepared = {
        "market": "KR", "feature_asof_date": "2026-09-28",
        "model_config": {
            "feature_set": "FS1h", "flow_variant": "lag1", "preprocess_profile": "rank",
            "excluded_features": list(EXCLUDED_FEATURES),
            "feature_time_contract": {"serving_applies_additional_lag": False},
        },
    }
    report = score_cross_section(
        bundle=_bundle(), panel=panel, report_date="2026-09-29",
        decision_at="2026-09-29T10:00:00+09:00",
        feature_asof_date="2026-09-28", prepared_manifest=prepared,
    )
    assert [row["symbol"] for row in report["rankings"]] == ["000003", "000002", "000001"]
    jump = next(row for row in report["rankings"] if row["symbol"] == "000002")
    assert jump["rank"] == 2
    assert jump["quality_review"] is True
    assert "K_price_jump_or_unknown" in jump["quality_reasons"]
    assert jump["K_simple_return"] == 0.35
    assert report["quality"]["top100_quality_review_rows"] == 1


def test_scoring_preserves_kr_exchange_groups_for_cross_section_ranks() -> None:
    panel = pl.DataFrame(
        {
            "trade_date": [date(2026, 9, 28)] * 4,
            "ticker": ["000001", "000002", "000003", "000004"],
            "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ"],
            "px_ret_1d": [1.0, 3.0, 2.0, 4.0],
        },
        schema_overrides={"trade_date": pl.Date},
    )
    prepared = {
        "market": "KR",
        "feature_asof_date": "2026-09-28",
        "model_config": {
            "feature_set": "FS1h",
            "flow_variant": "lag1",
            "preprocess_profile": "rank",
            "excluded_features": list(EXCLUDED_FEATURES),
            "feature_time_contract": {"serving_applies_additional_lag": False},
        },
    }
    report = score_cross_section(
        bundle=_bundle(), panel=panel, report_date="2026-09-29",
        decision_at="2026-09-29T10:00:00+09:00",
        feature_asof_date="2026-09-28", prepared_manifest=prepared,
    )
    assert report["market"] == "KR"
    assert [row["symbol"] for row in report["rankings"]] == [
        "000002", "000004", "000001", "000003"
    ]
    unknown_market = panel.with_columns(pl.lit("OTHER").alias("market"))
    with pytest.raises(ValueError, match="unsupported KR exchange codes"):
        score_cross_section(
            bundle=_bundle(), panel=unknown_market, report_date="2026-09-29",
            decision_at="2026-09-29T10:00:00+09:00",
            feature_asof_date="2026-09-28", prepared_manifest=prepared,
        )


def test_scoring_rejects_nonzero_seconds_at_decision_cutoff() -> None:
    with pytest.raises(ValueError, match="10:00"):
        score_cross_section(
            bundle=_bundle(),
            panel=pl.DataFrame(
                {
                    "trade_date": [date(2026, 9, 28)],
                    "ticker": ["000001"],
                    "market": ["KOSPI"],
                    "px_ret_1d": [1.0],
                },
                schema_overrides={"trade_date": pl.Date},
            ),
            report_date="2026-09-29",
            decision_at="2026-09-29T10:00:01+09:00",
            feature_asof_date="2026-09-28",
            prepared_manifest={"market": "KR", "feature_asof_date": "2026-09-28"},
        )


def test_direct_cli_guards_decision_time_and_prepared_snapshot_completion() -> None:
    with pytest.raises(ValueError, match="before the 10:00"):
        _require_decision_reached(
            "2026-09-29T10:00:00+09:00", "2026-09-29",
            now=datetime.fromisoformat("2026-09-29T09:59:59+09:00"),
        )
    _require_decision_reached(
        "2026-09-29T10:00:00+09:00", "2026-09-29",
        now=datetime.fromisoformat("2026-09-29T10:00:00+09:00"),
    )
    manifest = {
        "input_cutoff": "2026-09-29T09:30:00+09:00",
        "raw_snapshot_completed_at": "2026-09-29T09:15:00+09:00",
        "feature_marts_completed_at": "2026-09-29T09:20:00+09:00",
        "feature_build_completed_at": "2026-09-29T09:25:00+09:00",
    }
    marker = {"verified_available_by": "2026-09-29T09:29:59+09:00"}
    _validate_prepared_completion(manifest, marker, "2026-09-29")
    marker["verified_available_by"] = "2026-09-29T09:30:01+09:00"
    with pytest.raises(ValueError, match="after the 09:30"):
        _validate_prepared_completion(manifest, marker, "2026-09-29")


def test_direct_cli_completion_marker_binds_native_manifest_and_feature_hashes(tmp_path: Path) -> None:
    import hashlib
    import json

    feature_dir = tmp_path / "prepared"
    feature_dir.mkdir()
    feature_path = feature_dir / "feature_panel.parquet"
    feature_path.write_bytes(b"fixed feature artifact")
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    prepared = {"availability_evidence_type": "prepared_features_completion",
                "features_sha256": digest, "input_sha256": digest}
    manifest_path = feature_dir / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(prepared), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker = {"schema_version": "prepared-features-completion.v1",
              "availability_evidence_type": "prepared_features_completion",
              "verified_available_by": "2026-09-29T09:29:00+09:00",
              "features_sha256": digest,
              "native_prepare_manifest_sha256": manifest_sha}
    (feature_dir / "completion.json").write_text(json.dumps(marker), encoding="utf-8")
    loaded, marker_sha = _validate_completion_evidence(feature_dir, prepared)
    assert loaded == marker
    assert len(marker_sha) == 64

    marker["native_prepare_manifest_sha256"] = "0" * 64
    (feature_dir / "completion.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        _validate_completion_evidence(feature_dir, prepared)


def test_scoring_rejects_same_day_feature_asof_and_label_columns() -> None:
    panel = pl.DataFrame(
        {
            "trade_date": [date(2026, 9, 29)],
            "ticker": ["000001"],
            "market": ["KOSPI"],
            "px_ret_1d": [1.0],
            "y_up_20d": [1],
        },
        schema_overrides={"trade_date": pl.Date},
    )
    label_panel = panel.with_columns(pl.lit(date(2026, 9, 28)).alias("trade_date"))
    with pytest.raises(ValueError, match="precede report_date"):
        score_cross_section(
            bundle=_bundle(),
            panel=panel,
            report_date="2026-09-29",
            decision_at="2026-09-29T10:00:00+09:00",
            feature_asof_date="2026-09-29",
            prepared_manifest={"market": "KR", "feature_asof_date": "2026-09-29"},
        )
    with pytest.raises(ValueError, match="label or forward-return"):
        score_cross_section(
            bundle=_bundle(),
            panel=label_panel,
            report_date="2026-09-29",
            decision_at="2026-09-29T10:00:00+09:00",
            feature_asof_date="2026-09-28",
            prepared_manifest={"market": "KR", "feature_asof_date": "2026-09-28"},
        )
