from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from collector.lake import DataRoot
from modeler.serving.us_daily import (
    FEATURES,
    GOLDEN_DATES,
    build_daily_panel,
    compare_frozen_features,
    export_model_bundle,
    input_cutoff_for_report_date,
    load_model,
    transform_daily_features,
    validate_prepared_cutoff,
    validate_inference_time,
    validate_selection_completion,
    write_daily_run,
    _estimator_variant,
    _publish_native_directory,
    _read_native_completion,
)


class MeanModel:
    def __init__(self, variant: str = "unreviewed") -> None:
        self.variant = variant

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return matrix[:, : len(FEATURES)].mean(axis=1)


class FakeLake:
    def __init__(self, tables: dict[str, pl.DataFrame]) -> None:
        self.tables = tables

    def scan(self, table: str) -> pl.LazyFrame:
        return self.tables[table].lazy()


def feature_frame(days: tuple[date, ...], *, include_noneligible: bool = False) -> pl.DataFrame:
    symbols = ["AAA", "BBB"] + (["CCC"] if include_noneligible else [])
    rows = []
    for day_index, day in enumerate(days):
        for row_index, symbol in enumerate(symbols):
            row = {
                "date": day,
                "symbol": symbol,
                "price_ge_5": not (include_noneligible and symbol == "CCC"),
                "close": 10.0 + row_index,
                "adv_20d": 2_000_000.0,
            }
            for column_index, name in enumerate(FEATURES):
                row[name] = (
                    None
                    if name == FEATURES[0] and symbol == "AAA" and day_index == 0
                    else 0.01 * day_index + 0.1 * row_index + 0.001 * column_index
                )
                row[f"{name}_isna"] = row[name] is None
            rows.append(row)
    return pl.DataFrame(rows)


def test_daily_panel_uses_that_months_membership_and_exact_session_prices(monkeypatch) -> None:
    scoring_date = date(2026, 7, 15)
    members = pl.DataFrame(
        {
            "date": [date(2026, 7, 1), date(2026, 7, 1)],
            "symbol": ["AAA", "BBB"],
            "cik": ["1", "2"],
            "sic": ["1234", "4321"],
            "mcap_rank": [1, 2],
            "adv_20d": [2_000_000.0, 1_500_000.0],
            "exchange": ["XNYS", "XNYS"],
            "in_universe": [True, True],
        }
    )
    calendar = pl.DataFrame(
        {
            "date": [date(2026, 7, 1), scoring_date, date(2026, 7, 16)],
            "exchange": ["XNYS", "XNYS", "XNYS"],
            "close_local": [time(16, 0), time(16, 0), time(16, 0)],
        }
    )
    prices = pl.DataFrame(
        {
            "date": [scoring_date, scoring_date, date(2026, 7, 16)],
            "symbol": ["AAA", "BBB", "FUTURE"],
            "close": [25.0, 35.0, 99.0],
        }
    )
    adjusted = pl.DataFrame(
        {
            "date": [scoring_date, scoring_date],
            "symbol": ["AAA", "BBB"],
            "adj_close": [25.0, 35.0],
            "adj_volume": [100.0, 200.0],
        }
    )
    monkeypatch.setattr(
        "modeler.serving.us_daily.adjusted_daily",
        lambda lake, base_date: adjusted.lazy(),
    )
    lake = FakeLake(
        {"universe_daily": members, "trading_calendar": calendar, "prices_daily": prices}
    )
    decision_at = datetime(2026, 7, 16, 14, 0, tzinfo=timezone.utc)
    panel = build_daily_panel(lake, scoring_date, decision_at=decision_at)
    assert panel["date"].unique().to_list() == [scoring_date]
    assert panel["symbol"].to_list() == ["AAA", "BBB"]
    assert panel["price_ge_5"].to_list() == [True, True]
    with pytest.raises(ValueError, match="decision_at 전에 장이 끝나지 않은 세션"):
        build_daily_panel(lake, date(2026, 7, 16), decision_at=decision_at)


def test_transform_preserves_53_features_missing_flags_and_106_column_matrix() -> None:
    frame = feature_frame((date(2026, 7, 15),), include_noneligible=True)
    transformed, matrix = transform_daily_features(frame)
    assert transformed.height == 2
    assert matrix.shape == (2, 106)
    assert transformed["mom_12_1_isna"].to_list() == [True, False]
    assert transformed["mom_12_1_rank"].to_list() == [0.5, 0.0]
    assert "turnover_rank" not in FEATURES


def test_frozen_feature_comparison_checks_keys_masks_and_values() -> None:
    frozen = feature_frame(GOLDEN_DATES)
    actual = frozen.clone()
    report = compare_frozen_features(actual, frozen)
    assert report.height == 53
    assert report["max_abs_delta"].max() == 0.0

    # `iv_isna` is itself an indicator. Match the original model prep, which
    # recomputes its missingness flag from the finite feature value.
    derived_flag_missing = actual.drop("iv_isna_isna")
    frozen_derived_flag_missing = frozen.drop("iv_isna_isna")
    assert compare_frozen_features(
        derived_flag_missing, frozen_derived_flag_missing
    ).height == 53
    with pytest.raises(ValueError, match="missing flag"):
        compare_frozen_features(
            actual.drop("mom_12_1_isna"), frozen.drop("mom_12_1_isna")
        )

    changed = actual.with_columns(
        pl.when((pl.col("date") == GOLDEN_DATES[0]) & (pl.col("symbol") == "AAA"))
        .then(999.0)
        .otherwise(pl.col("mom_6_1"))
        .alias("mom_6_1")
    )
    with pytest.raises(ValueError, match="frozen v2와 다릅니다"):
        compare_frozen_features(changed, frozen)


def test_bundle_reload_and_daily_output_are_hash_idempotent(tmp_path: Path, monkeypatch) -> None:
    models: dict[str, Path] = {}
    manifests: dict[str, Path] = {}
    golden = feature_frame(GOLDEN_DATES)
    for name in ("lightgbm", "ridge"):
        source = tmp_path / f"{name}-source.joblib"
        joblib.dump(
            {
                "model": MeanModel(name),
                "features": FEATURES,
                "train_end": "2025-06-30",
                "max_training_terminal": "2025-06-30",
                "kind": "exploratory",
            },
            source,
        )
        bundle_dir = tmp_path / "bundle" / name
        monkeypatch.setattr(
            "modeler.serving.us_daily._estimator_variant",
            lambda estimator: estimator.variant,
        )
        export_model_bundle(
            source_model=source,
            bundle_dir=bundle_dir,
            model_version="1",
            golden_features=golden,
            golden_source_sha256={
                "us_features_v2": "a" * 64,
                "us_features_flow_v1": "b" * 64,
            },
        )
        model_path = bundle_dir / "model.joblib"
        manifest_path = bundle_dir / "manifest.json"
        loaded, loaded_manifest = load_model(model_path, manifest_path)
        assert loaded["kind"] == "exploratory"
        assert loaded_manifest["model_id"] == "us_exploratory_20260929_r1"
        assert loaded_manifest["model_version"] == "1"
        assert loaded_manifest["variant"] == name
        models[name] = model_path
        manifests[name] = manifest_path

    scoring_date = date(2026, 7, 15)
    frame = feature_frame((scoring_date,))
    root = DataRoot(tmp_path / "stock_data" / "us")
    kwargs = {
        "root": root,
        "scoring_date": scoring_date,
        "features": frame,
        "model_paths": models,
        "model_manifests": manifests,
        "source_revision": {"prices_daily": "2026-07-15", "universe_daily": "2026-07-01"},
        "code_hash": "code-a",
        "report_date": date(2026, 7, 16),
        "decision_at": datetime(2026, 7, 16, 1, 0, tzinfo=timezone.utc),
        "prepared_available_at": "2026-07-15T23:00:00+09:00",
        "input_cutoff": "2026-07-16T09:30:00+09:00",
        "freshness_status": "ok",
        "selection_manifest_sha256": "selection-hash",
    }
    first = write_daily_run(**kwargs)
    second = write_daily_run(**kwargs)
    assert first == second
    for directory in first.values():
        assert (directory / "input_features.parquet").is_file()
        assert (directory / "design_matrix.npy").is_file()
        assert (directory / "rankings.parquet").is_file()
    changed = write_daily_run(**{**kwargs, "code_hash": "code-b"})
    assert all(path != changed[name] for name, path in first.items())


def test_estimator_variant_rejects_unreviewed_object() -> None:
    with pytest.raises(ValueError, match="지원하지 않는 US estimator"):
        _estimator_variant(MeanModel("unreviewed"))


def _native_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True)
    feature_path = directory / "features.parquet"
    feature_path.write_bytes(b"synthetic-feature-input")
    feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    manifest = {
        "market": "US",
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": feature_sha,
        "input_sha256": feature_sha,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path


def test_native_prepared_completion_needs_marker_and_rejects_late_publish(tmp_path: Path) -> None:
    from datetime import timedelta

    from modeler.serving.us_daily import _SEOUL_TZ

    cutoff = datetime(2026, 9, 30, 9, 30, tzinfo=_SEOUL_TZ)
    incomplete_manifest = _native_fixture(tmp_path / "incomplete")
    with pytest.raises(ValueError, match="completion marker"):
        _read_native_completion(incomplete_manifest, input_cutoff=cutoff)

    staged_manifest = _native_fixture(tmp_path / "staged")
    final_dir = tmp_path / "published-late"
    late = cutoff + timedelta(seconds=1)

    def late_after_publish() -> datetime:
        assert (final_dir / "features.parquet").is_file()
        assert (final_dir / "manifest.json").is_file()
        return late

    _publish_native_directory(
        staged_manifest.parent,
        final_dir,
        now_fn=late_after_publish,
    )
    published_manifest = final_dir / "manifest.json"
    completion = _read_native_completion(published_manifest)
    assert completion["verified_available_by"] == late
    with pytest.raises(ValueError, match="마감 뒤에 완료"):
        _read_native_completion(published_manifest, input_cutoff=cutoff)

    marker_path = final_dir / "completion.json"
    marker = json.loads(marker_path.read_text())
    marker["features_sha256"] = hashlib.sha256(b"different-input").hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True))
    with pytest.raises(ValueError, match="marker hash"):
        _read_native_completion(
            published_manifest,
            expected_marker_sha256=completion["marker_sha256"],
        )


def test_prepared_features_must_finish_by_exact_0930_kst_cutoff() -> None:
    report_date = date(2026, 7, 16)
    cutoff = input_cutoff_for_report_date(report_date)
    validate_prepared_cutoff(
        available_at=cutoff,
        input_cutoff=cutoff,
        report_date=report_date,
    )
    with pytest.raises(ValueError, match="09:30 입력 마감"):
        validate_prepared_cutoff(
            available_at=datetime(2026, 7, 16, 9, 30, 1, tzinfo=cutoff.tzinfo),
            input_cutoff=cutoff,
            report_date=report_date,
        )
    with pytest.raises(ValueError, match="시간대"):
        validate_prepared_cutoff(
            available_at=datetime(2026, 7, 16, 9, 30),
            input_cutoff=cutoff,
            report_date=report_date,
        )


def test_selection_can_finish_after_input_cutoff_but_before_inference() -> None:
    report_date = date(2026, 7, 16)
    cutoff = input_cutoff_for_report_date(report_date)
    decision_at = datetime(2026, 7, 16, 10, 0, tzinfo=cutoff.tzinfo)
    validate_selection_completion(
        completed_at=datetime(2026, 7, 16, 9, 30, 1, tzinfo=cutoff.tzinfo),
        decision_at=decision_at,
    )
    validate_selection_completion(completed_at=decision_at, decision_at=decision_at)
    with pytest.raises(ValueError, match="10:00"):
        validate_selection_completion(
            completed_at=datetime(2026, 7, 16, 10, 0, 1, tzinfo=cutoff.tzinfo),
            decision_at=decision_at,
        )
    with pytest.raises(ValueError, match="시간대"):
        validate_selection_completion(
            completed_at=datetime(2026, 7, 16, 9, 30, 1),
            decision_at=decision_at,
        )


def test_inference_waits_until_exact_decision_time() -> None:
    decision_at = datetime(2026, 7, 16, 10, 0, tzinfo=input_cutoff_for_report_date(
        date(2026, 7, 16)
    ).tzinfo)
    with pytest.raises(ValueError, match="이릅니다"):
        validate_inference_time(
            decision_at=decision_at,
            now=datetime(2026, 7, 16, 9, 59, 59, tzinfo=decision_at.tzinfo),
        )
    validate_inference_time(decision_at=decision_at, now=decision_at)
    with pytest.raises(ValueError, match="정확히 10:00:00"):
        validate_inference_time(
            decision_at=decision_at.replace(microsecond=1),
            now=decision_at.replace(microsecond=1),
        )
