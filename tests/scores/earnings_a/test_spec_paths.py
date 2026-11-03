"""Tests for earnings_a spec constants, month arithmetic, and input paths.

All data here is synthetic. No real label or raw values are read.
"""

from __future__ import annotations

import hashlib
import itertools
import shutil
from datetime import date
from pathlib import Path

import duckdb
import pytest

from modeler.scores.earnings_a import paths as paths_mod
from modeler.scores.earnings_a import spec
from modeler.scores.earnings_a.paths import (
    RAW_TABLES,
    DataPaths,
    connect_inputs,
    sha256_file,
    verify_input_hashes,
)


def _write_parquet(target: Path, select_sql: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"COPY ({select_sql}) TO '{target}' (FORMAT PARQUET)")
    finally:
        con.close()


def _build_tree(root: Path) -> DataPaths:
    """Build a synthetic stock_data tree with the real raw layout and tiny marts."""
    paths = DataPaths.resolve({"STOCK_DATA_ROOT": str(root)})
    raw = paths.raw_root
    _write_parquet(
        raw / "daily_ohlcv/schema_version=1/year=2024/month=01/part-000000.parquet",
        "SELECT 1 AS ticker, DATE '2024-01-02' AS trade_date, 100 AS close",
    )
    _write_parquet(
        raw / "daily_market_cap/schema_version=1/year=2024/month=01/part-000000.parquet",
        "SELECT 1 AS ticker, DATE '2024-01-02' AS trade_date, 1000 AS market_cap",
    )
    _write_parquet(
        raw / "stock_master/schema_version=1/part-000000.parquet",
        "SELECT '000001' AS ticker, 'ACTIVE' AS status",
    )
    _write_parquet(
        raw / "dart_corp_master/schema_version=1/part-000000.parquet",
        "SELECT '00000001' AS corp_code, '000001' AS ticker, '64992' AS induty_code",
    )
    _write_parquet(
        raw / "dart_filing_receipt_raw/schema_version=1/part-000000.parquet",
        "SELECT '00000001' AS corp_code, '20240514000001' AS rcept_no, '분기보고서' AS report_nm",
    )
    _write_parquet(
        raw
        / "dart_financial_statement_raw/schema_version=1/bsns_year=2025/reprt_code=11012"
        / "part-000000.parquet",
        "SELECT '00000001' AS corp_code, 2025 AS bsns_year, 1 AS amount",
    )
    _write_parquet(paths.v4p_parquet, "SELECT 'x' AS ticker, 1 AS total_assets")
    # The label mart exists on disk, so the test can check that it is not registered.
    _write_parquet(paths.r2_parquet, "SELECT 'x' AS ticker, 2 AS y_clip_marker")
    return paths


# ---------------------------------------------------------------- spec constants


def test_features_has_21_names_in_spec_order():
    assert len(spec.FEATURES) == 21
    assert spec.FEATURES == (
        "oi_ttm",
        "oi_q0",
        "oi_q1",
        "oi_q2",
        "oi_q3",
        "oi_chg_4q",
        "rev_ttm",
        "margin_ttm",
        "ocf_ttm",
        "accrual_ttm",
        "ni_ttm",
        "asset_growth",
        "equity_ratio",
        "cash_ratio",
        "debt_ratio",
        "capex_ttm",
        "div_ttm",
        "log_assets",
        "loss_ttm",
        "is_ofs",
        "stale_days",
    )
    assert set(spec.BINARY_FEATURES) <= set(spec.FEATURES)


def test_hgb_grid_has_8_configs_in_product_order():
    assert len(spec.HGB_GRID) == 8
    expected = [
        {
            "learning_rate": lr,
            "max_iter": mi,
            "max_leaf_nodes": ml,
            "min_samples_leaf": 100,
            "early_stopping": False,
            "loss": "absolute_error",
        }
        for lr, mi, ml in itertools.product((0.05, 0.1), (100, 300), (15, 31))
    ]
    assert list(spec.HGB_GRID) == expected


def test_ridge_alphas_has_4_values():
    assert spec.RIDGE_ALPHAS == (0.1, 1.0, 10.0, 100.0)
    assert len(spec.RIDGE_ALPHAS) == 4


def test_inner_val_dates_are_13_quarter_ends():
    dates = spec.INNER_VAL_DATES
    assert len(dates) == 13
    assert dates[0] == date(2019, 6, 30)
    assert dates[-1] == date(2022, 6, 30)
    assert all(isinstance(d, date) for d in dates)
    assert all(d.month in spec.QUARTER_MONTHS for d in dates)
    assert all(spec.add_months_end(d, 0) == d for d in dates)


def test_metrics_has_12_metric_codes():
    assert len(spec.METRICS) == 12
    assert spec.METRICS["oi"] == "operating_income"
    assert spec.METRICS["div"] == "dividends_paid"


def test_reason_codes_and_seeds():
    assert len(spec.REASON_CODES) == 7
    assert spec.REASON_CODES[0] == "IMMATURE"
    assert spec.REASON_CODES[-1] == "UNCONFIRMED"
    assert spec.BOOTSTRAP_SEED == 20261103
    assert spec.BOOTSTRAP_DRAWS == 10000


# ---------------------------------------------------------------- month arithmetic


def test_month_index_is_year_times_12_plus_month_minus_1():
    assert spec.month_index(date(2024, 1, 1)) == 2024 * 12
    assert spec.month_index(date(2024, 12, 31)) == 2024 * 12 + 11
    assert spec.month_index(date(2025, 1, 31)) == spec.month_index(date(2024, 12, 1)) + 1


def test_add_months_end_leap_day_plus_12_clamps_to_feb_28():
    assert spec.add_months_end(date(2024, 2, 29), 12) == date(2025, 2, 28)


def test_add_months_end_crosses_into_leap_february():
    assert spec.add_months_end(date(2023, 11, 30), 3) == date(2024, 2, 29)


def test_add_months_end_negative_n():
    assert spec.add_months_end(date(2024, 3, 15), -3) == date(2023, 12, 31)
    assert spec.add_months_end(date(2024, 1, 31), -1) == date(2023, 12, 31)
    assert spec.add_months_end(date(2024, 1, 1), -13) == date(2022, 12, 31)


def test_add_months_end_ignores_day_of_input():
    assert spec.add_months_end(date(2024, 6, 1), 3) == spec.add_months_end(date(2024, 6, 30), 3)
    assert spec.add_months_end(date(2024, 6, 1), 3) == date(2024, 9, 30)


def test_add_months_end_zero_is_month_end():
    assert spec.add_months_end(date(2024, 2, 10), 0) == date(2024, 2, 29)


# ---------------------------------------------------------------- DataPaths


def test_resolve_uses_stock_data_root_from_env(tmp_path):
    p = DataPaths.resolve({"STOCK_DATA_ROOT": str(tmp_path)})
    root = tmp_path.resolve()
    assert p.stock_data_root == root
    assert p.raw_root == (root / "kr/raw/raw_postgres/snapshot_date=2026-10-23/source=sj2_remote")
    assert p.out_root == root / "kr/output/earnings_forecast_a"
    assert p.v4p_parquet == (
        root
        / "kr/output/research_prep_20261027/a_financial_target_v4"
        / "mart_p6prime_pit/derived/feature/snapshot_date=2026-10-23/source=sj2_remote"
        / "fin_quarterly_metric_vintage_dlr_v4p/part-000000.parquet"
    )
    assert p.r2_parquet == (
        root
        / "kr/output/research_prep_20261027/a_financial_target_v4"
        / "mart_r2/derived/feature/snapshot_date=2026-10-23/source=sj2_remote"
        / "fin_quarterly_metric_vintage_dl_v4/part-000000.parquet"
    )


def test_resolve_defaults_to_resolved_relative_stock_data(monkeypatch):
    monkeypatch.delenv("STOCK_DATA_ROOT", raising=False)
    p = DataPaths.resolve({})
    assert p.stock_data_root == Path("../stock_data").resolve()


def test_datapaths_is_frozen(tmp_path):
    p = DataPaths.resolve({"STOCK_DATA_ROOT": str(tmp_path)})
    with pytest.raises(Exception):
        p.out_root = tmp_path  # type: ignore[misc]


def test_expected_sha256_values_are_64_hex_chars():
    assert set(paths_mod.EXPECTED_SHA256) == {"v4p", "r2"}
    for value in paths_mod.EXPECTED_SHA256.values():
        assert len(value) == 64
        int(value, 16)


# ---------------------------------------------------------------- hashing


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "blob.bin"
    data = b"earnings-a" * 300_000  # several chunks
    target.write_bytes(data)
    assert sha256_file(target) == hashlib.sha256(data).hexdigest()
    assert sha256_file(target, chunk_size=7) == hashlib.sha256(data).hexdigest()


def test_verify_input_hashes_passes_when_hashes_match(tmp_path, monkeypatch):
    paths = _build_tree(tmp_path / "data")
    monkeypatch.setitem(paths_mod.EXPECTED_SHA256, "v4p", sha256_file(paths.v4p_parquet))
    monkeypatch.setitem(paths_mod.EXPECTED_SHA256, "r2", sha256_file(paths.r2_parquet))
    verify_input_hashes(paths)  # no exception


def test_verify_input_hashes_names_file_on_mismatch(tmp_path):
    paths = _build_tree(tmp_path / "data")
    # The pinned values are the real mart hashes, so the synthetic files must mismatch.
    with pytest.raises(RuntimeError, match="input hash mismatch: part-000000.parquet"):
        verify_input_hashes(paths)


def test_verify_input_hashes_error_text_is_name_only(tmp_path):
    paths = _build_tree(tmp_path / "data")
    with pytest.raises(RuntimeError) as info:
        verify_input_hashes(paths)
    text = str(info.value)
    assert "v4p" in text
    assert paths.v4p_parquet.name in text
    assert paths_mod.EXPECTED_SHA256["v4p"] not in text
    assert sha256_file(paths.v4p_parquet) not in text


# ---------------------------------------------------------------- connect_inputs


def test_connect_inputs_registers_v4p_and_raw_views_only(tmp_path):
    paths = _build_tree(tmp_path / "data")
    con = connect_inputs(paths)
    try:
        names = {
            row[0]
            for row in con.execute(
                "SELECT view_name FROM duckdb_views() WHERE NOT internal"
            ).fetchall()
        }
        assert names == {"v4p", *RAW_TABLES}
        assert con.execute("SELECT count(*) FROM v4p").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM stock_master").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM dart_financial_statement_raw").fetchone()[0] == 1
    finally:
        con.close()


def test_connect_inputs_never_references_label_mart(tmp_path):
    paths = _build_tree(tmp_path / "data")
    con = connect_inputs(paths)
    try:
        rows = con.execute(
            "SELECT view_name, sql FROM duckdb_views() WHERE NOT internal"
        ).fetchall()
        for name, sql in rows:
            assert "dl_v4" not in name and "dl_v4" not in sql, name
            assert "mart_r2" not in name and "mart_r2" not in sql, name
    finally:
        con.close()


def test_connect_inputs_missing_raw_table_raises(tmp_path):
    paths = _build_tree(tmp_path / "data")
    shutil.rmtree(paths.raw_root / "dart_corp_master")
    with pytest.raises(FileNotFoundError, match="dart_corp_master"):
        connect_inputs(paths)


def test_connect_inputs_sets_statistics_propagation_off(tmp_path):
    paths = _build_tree(tmp_path / "data")
    con = connect_inputs(paths)
    try:
        value = con.execute("SELECT current_setting('disabled_optimizers')").fetchone()[0]
        assert "statistics_propagation" in value
    finally:
        con.close()
