from __future__ import annotations

import math
from datetime import date

import duckdb
import pytest

from modeler.analysis.horizon_scan_daily_ic import CELL_IDENTITY_COLUMNS
from modeler.analysis.horizon_scan_runner import (
    UNIVERSE_SAMPLE_COMBOS,
    assert_lag1_matches_prior_valid_session,
    build_analysis_panel_sql,
    build_broad_quantile_segment_sql,
    build_period_segment_sql,
    family_coverage_stats,
    register_analysis_panel,
    resolve_common_formation_end,
    resolve_horizon_eligible_end,
    run_registry_scan,
)


def _seed_marts(con: duckdb.DuckDBPyConnection, *, duplicate_flow_row: bool = False) -> None:
    con.execute("""
        CREATE TABLE label_scan AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 0.01, true, true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.02, true, false),
            (DATE '2024-01-03', 'A', 'KOSPI', -0.01, true, true)
        ) AS t(trade_date, ticker, market, raw_label_5d, common_formation_120d,
               common_survivor_120d)
    """)
    con.execute("""
        CREATE TABLE feat_price AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 0.10, 0.09),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.20, 0.19),
            (DATE '2024-01-03', 'A', 'KOSPI', 0.11, 0.10)
        ) AS t(trade_date, ticker, market, px_reversal_5d, px_reversal_5d_lag1)
    """)
    flow_rows = """
            (DATE '2024-01-02', 'A', 'KOSPI', 0.5, 'allowed'),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.6, 'banned'),
            (DATE '2024-01-03', 'A', 'KOSPI', 0.4, 'allowed')
    """
    if duplicate_flow_row:
        flow_rows += ", (DATE '2024-01-03', 'A', 'KOSPI', 0.9, 'allowed')"
    con.execute(f"""
        CREATE TABLE feat_flow AS SELECT * FROM (VALUES {flow_rows}
        ) AS t(trade_date, ticker, market, flow_foreign_netbuy_to_volume_20d, short_regime)
    """)
    con.execute("""
        ALTER TABLE feat_flow ADD COLUMN short_balance_is_available BOOLEAN DEFAULT true
    """)
    con.execute("""
        CREATE TABLE dim_universe_broad_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', true, false, true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', true, false, true),
            (DATE '2024-01-03', 'A', 'KOSPI', true, false, true)
        ) AS t(trade_date, ticker, market, in_universe, membership_reconstruction_available,
               management_filter_available)
    """)
    con.execute("""
        CREATE TABLE dim_universe_tradable_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', true, false, true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', false, false, true),
            (DATE '2024-01-03', 'A', 'KOSPI', true, false, true)
        ) AS t(trade_date, ticker, market, in_universe, membership_reconstruction_available,
               management_filter_available)
    """)
    con.execute("""
        CREATE TABLE dim_stock_pit_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 1000.0, 100.0, 900.0, 90000.0, true, false),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 2000.0, 0.0, 2000.0, 400000.0, true, false),
            (DATE '2024-01-03', 'A', 'KOSPI', 1000.0, 100.0, 900.0, 90900.0, true, false)
        ) AS t(trade_date, ticker, market, issued_shares_pit, treasury_shares_pit,
               float_shares_pit, market_cap_pit, shares_is_available, shares_invalid_flag)
    """)
    con.execute("""
        CREATE TABLE dim_price_quality_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 100, false, false, 0.01, 0.00995,
             false, false, 0.0, false, 'allowed', true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 50, false, false, 0.02, 0.0198,
             false, false, 0.0, false, 'banned', true),
            (DATE '2024-01-03', 'A', 'KOSPI', 101, false, false, -0.01, -0.01005,
             false, false, 0.0, false, 'allowed', true)
        ) AS t(trade_date, ticker, market, valid_session_idx, is_halted, volume_zero,
               simple_ret, log_ret, ca_mask, ca_event, ca_event_cumulative,
               ca_rule_applicability_unknown, short_regime, short_balance_is_available)
    """)


def test_panel_join_preserves_label_scan_grain_and_resolves_collisions() -> None:
    con = duckdb.connect()
    _seed_marts(con)
    register_analysis_panel(con)
    rows = con.execute("SELECT * FROM analysis_panel ORDER BY trade_date, ticker").fetchdf()
    assert len(rows) == 3  # matches label_scan's row count, not feat_flow's

    # in_broad/in_tradable both survive under distinct names (no ambiguous-column error).
    assert {"in_broad", "in_tradable"} <= set(rows.columns)
    b_row = rows[(rows.ticker == "B")].iloc[0]
    assert bool(b_row["in_broad"]) is True
    assert bool(b_row["in_tradable"]) is False

    # short_regime/short_balance_is_available come from the quality mart, not flow's.
    assert "short_regime" in rows.columns
    assert rows.columns.tolist().count("short_regime") == 1

    # feature columns from both price and flow are present and unprefixed.
    assert "px_reversal_5d" in rows.columns
    assert "flow_foreign_netbuy_to_volume_20d" in rows.columns
    assert "market_cap_pit" in rows.columns

    # formation_session_idx is a global calendar ordinal shared by every
    # ticker/market on the same trade_date (2024-01-02 -> 1, 2024-01-03 -> 2).
    by_date = dict(zip(rows["trade_date"], rows["formation_session_idx"]))
    assert rows[rows.trade_date == rows.trade_date.min()]["formation_session_idx"].nunique() == 1
    assert sorted(set(by_date.values())) == [1, 2]
    assert "raw_label_5d" in rows.columns


def test_panel_join_raises_on_duplicate_join_partner_keys_instead_of_deduping() -> None:
    con = duckdb.connect()
    _seed_marts(con, duplicate_flow_row=True)
    with pytest.raises(RuntimeError, match="fanned out"):
        register_analysis_panel(con)


def test_build_analysis_panel_sql_accepts_view_name_overrides() -> None:
    sql = build_analysis_panel_sql(price_view="custom_price")
    assert "FROM label_scan l" in sql
    assert "LEFT JOIN custom_price p USING" in sql


def _seed_segment_panel(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE analysis_panel AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A1', 'KOSPI', true, false, 10.0),
            (DATE '2024-01-02', 'A2', 'KOSPI', true, false, 20.0),
            (DATE '2024-01-02', 'A3', 'KOSPI', true, true,  30.0),
            (DATE '2024-01-02', 'A4', 'KOSPI', true, true,  40.0),
            (DATE '2024-01-02', 'A5', 'KOSPI', true, false, 50.0),
            (DATE '2024-01-02', 'A6', 'KOSPI', true, false, 60.0),
            -- a name outside broad must still get labeled from broad's cutpoints
            (DATE '2024-01-02', 'A7', 'KOSPI', false, false, 15.0),
            -- ties: three names tied at 10 must land in the same bucket
            (DATE '2024-01-03', 'B1', 'KOSPI', true, false, 10.0),
            (DATE '2024-01-03', 'B2', 'KOSPI', true, false, 10.0),
            (DATE '2024-01-03', 'B3', 'KOSPI', true, false, 10.0),
            (DATE '2024-01-03', 'B4', 'KOSPI', true, false, 40.0),
            (DATE '2024-01-03', 'B5', 'KOSPI', true, false, 50.0),
            (DATE '2024-01-03', 'B6', 'KOSPI', true, false, 60.0),
            -- a thin cross-section (below min_names) must come back NULL
            (DATE '2024-01-04', 'C1', 'KOSPI', true, false, 10.0),
            (DATE '2024-01-04', 'C2', 'KOSPI', true, false, 20.0),
            -- two markets on the same date: KOSDAQ's much smaller values must
            -- not be pooled into KOSPI's cross-section (or vice versa)
            (DATE '2024-01-05', 'D1', 'KOSPI', true, false, 1000.0),
            (DATE '2024-01-05', 'D2', 'KOSPI', true, false, 2000.0),
            (DATE '2024-01-05', 'D3', 'KOSPI', true, false, 3000.0),
            (DATE '2024-01-05', 'E1', 'KOSDAQ', true, false, 1.0),
            (DATE '2024-01-05', 'E2', 'KOSDAQ', true, false, 2.0),
            (DATE '2024-01-05', 'E3', 'KOSDAQ', true, false, 3.0)
        ) AS t(trade_date, ticker, market, in_broad, in_tradable, market_cap_pit)
    """)


def test_broad_quantile_segment_splits_evenly_and_reuses_cutpoints_on_tradable() -> None:
    con = duckdb.connect()
    _seed_segment_panel(con)
    sql = build_broad_quantile_segment_sql(
        value_col="market_cap_pit", segment_col="size_segment", min_names=3
    )
    rows = {
        r[0]: r[1]
        for r in con.execute(f"SELECT ticker, size_segment FROM ({sql}) t").fetchall()
    }
    assert rows["A1"] == "small" and rows["A2"] == "small"
    assert rows["A3"] == "mid" and rows["A4"] == "mid"
    assert rows["A5"] == "large" and rows["A6"] == "large"
    # A7 (in_broad=false, value 15) still gets a segment from broad's cutpoints —
    # cutpoints are never recomputed within a universe subset.
    assert rows["A7"] == "small"


def test_broad_quantile_segment_keeps_ties_in_the_same_bucket() -> None:
    con = duckdb.connect()
    _seed_segment_panel(con)
    sql = build_broad_quantile_segment_sql(
        value_col="market_cap_pit", segment_col="size_segment", min_names=3
    )
    rows = {
        r[0]: r[1]
        for r in con.execute(f"SELECT ticker, size_segment FROM ({sql}) t").fetchall()
    }
    assert rows["B1"] == rows["B2"] == rows["B3"] == "small"
    assert rows["B4"] == "mid"
    assert rows["B5"] == rows["B6"] == "large"


def test_broad_quantile_segment_computes_each_market_independently() -> None:
    con = duckdb.connect()
    _seed_segment_panel(con)
    sql = build_broad_quantile_segment_sql(
        value_col="market_cap_pit", segment_col="size_segment", min_names=3
    )
    rows = {
        r[0]: r[1]
        for r in con.execute(f"SELECT ticker, size_segment FROM ({sql}) t").fetchall()
    }
    # KOSDAQ's tiny values (1,2,3) must not be pooled with KOSPI's (1000,2000,3000) —
    # each market gets its own small/mid/large split from its own cross-section.
    assert rows["D1"] == "small" and rows["D2"] == "mid" and rows["D3"] == "large"
    assert rows["E1"] == "small" and rows["E2"] == "mid" and rows["E3"] == "large"


def test_broad_quantile_segment_is_null_below_min_names() -> None:
    con = duckdb.connect()
    _seed_segment_panel(con)
    sql = build_broad_quantile_segment_sql(
        value_col="market_cap_pit", segment_col="size_segment", min_names=3
    )
    rows = {
        r[0]: r[1]
        for r in con.execute(f"SELECT ticker, size_segment FROM ({sql}) t").fetchall()
    }
    assert rows["C1"] is None and rows["C2"] is None


def test_period_segment_sql_resolves_placeholders() -> None:
    period_sets = [
        {"id": "2014_2016", "start": date(2014, 6, 1), "end": date(2016, 12, 31)},
        {"id": "2023_11_common_end", "start": date(2023, 11, 1), "end": "common_formation_end"},
    ]
    sql = build_period_segment_sql(period_sets, {"common_formation_end": date(2025, 2, 4)})
    con = duckdb.connect()
    row = con.execute(
        f"SELECT {sql} FROM (SELECT DATE '2024-01-01' AS trade_date) t"
    ).fetchone()
    assert row[0] == "2023_11_common_end"

    row = con.execute(
        f"SELECT {sql} FROM (SELECT DATE '2020-01-01' AS trade_date) t"
    ).fetchone()
    assert row[0] is None  # falls in neither preregistered interval


def test_period_segment_sql_rejects_unresolved_placeholder() -> None:
    period_sets = [{"id": "x", "start": date(2020, 1, 1), "end": "horizon_eligible_end"}]
    with pytest.raises(KeyError, match="horizon_eligible_end"):
        build_period_segment_sql(period_sets, {})


def test_resolve_common_formation_end_reads_the_flag() -> None:
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE analysis_panel AS SELECT * FROM (VALUES
            (DATE '2024-01-01', true),
            (DATE '2024-01-02', true),
            (DATE '2024-01-03', false)
        ) AS t(trade_date, common_formation_120d)
    """)
    assert resolve_common_formation_end(con) == date(2024, 1, 2)


def test_resolve_horizon_eligible_end_stops_before_holdout() -> None:
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE analysis_panel AS SELECT * FROM (VALUES
            (DATE '2024-01-01', DATE '2024-01-05'),
            (DATE '2024-01-02', DATE '2024-01-06'),
            (DATE '2024-01-03', DATE '2026-01-01')
        ) AS t(trade_date, label_end_date_5d)
    """)
    result = resolve_horizon_eligible_end(
        con, "analysis_panel", "label_end_date_5d", holdout_start=date(2025, 8, 1)
    )
    assert result == date(2024, 1, 2)


def test_family_coverage_stats_computes_expected_fractions() -> None:
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE formation AS SELECT * FROM (VALUES
            (DATE '2024-01-01', 1.0, true),
            (DATE '2024-01-01', NULL, true),
            (DATE '2024-01-02', 2.0, false),
            (DATE '2024-01-02', 3.0, true)
        ) AS t(trade_date, px_reversal_5d, label_ok_5d)
    """)
    stats = family_coverage_stats(
        con, formation_view="formation", feature_col="px_reversal_5d", label_ok_col="label_ok_5d"
    )
    assert stats["n_formation"] == 4
    assert stats["feature_coverage"] == pytest.approx(0.75)
    assert stats["label_coverage"] == pytest.approx(0.75)
    assert stats["effective_sample_start"] == date(2024, 1, 1)
    assert stats["effective_sample_end"] == date(2024, 1, 2)


def _seed_lag1_panel(con: duckdb.DuckDBPyConnection, *, break_lag1: bool = False) -> None:
    b_session2_lag1 = 999.0 if break_lag1 else 0.20
    con.execute(f"""
        CREATE TABLE analysis_panel AS SELECT * FROM (VALUES
            ('A', 'KOSPI', 1, 0.10, NULL),
            ('A', 'KOSPI', 2, 0.11, 0.10),
            -- ticker A's day 3 is a halt (no valid_session_idx=3 row at all);
            -- its next valid session is 4, whose lag1 must reference session
            -- 2's native value (0.11), not a raw session-4-minus-1 lookup.
            ('A', 'KOSPI', 4, 0.12, 0.11),
            ('B', 'KOSPI', 1, 0.20, NULL),
            ('B', 'KOSPI', 2, 0.21, {b_session2_lag1})
        ) AS t(ticker, market, valid_session_idx, native_col, lag1_col)
    """)


def test_lag1_shift_invariant_passes_for_correctly_shifted_data() -> None:
    con = duckdb.connect()
    _seed_lag1_panel(con)
    assert_lag1_matches_prior_valid_session(
        con, panel_view="analysis_panel", native_col="native_col", lag1_col="lag1_col"
    )


def test_lag1_shift_invariant_rejects_mismatched_data() -> None:
    con = duckdb.connect()
    _seed_lag1_panel(con, break_lag1=True)
    with pytest.raises(RuntimeError, match="does not equal"):
        assert_lag1_matches_prior_valid_session(
            con, panel_view="analysis_panel", native_col="native_col", lag1_col="lag1_col"
        )


# --- Stage 0: run_registry_scan is the only scan_cell caller with a sink ---


class _RecordingSink:
    def __init__(self) -> None:
        self.cells: list[dict[str, object]] = []
        self.flushed: list[str] = []

    def emit(self, cell, *, daily, market_ic, daily_spread=None, market_spread=None) -> None:
        self.cells.append(cell)

    def flush_feature(self, feature: str) -> None:
        self.flushed.append(feature)


def _seed_two_feature_panel(con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for session in range(1, 31):
        d = f"2024-01-01' + {session - 1}"
        for market in ("KOSPI", "KOSDAQ"):
            for t in range(10):
                wobble = 3.0 * math.sin(t + 0.7 * session)
                label = float(t) * 2.0 + wobble
                rows.append(
                    f"(DATE '{d}, '{market[:1]}{t}', '{market}', {session}, "
                    f"{float(t) + 0.01 * session}, {float(9 - t) + 0.01 * session}, "
                    f"{label}, {label}, true, true, true, true, true, false)"
                )
    con.execute(
        "CREATE TABLE analysis_panel AS SELECT "
        "trade_date, ticker, market, formation_session_idx, "
        "CAST(feat_a AS DOUBLE) AS feat_a, CAST(feat_b AS DOUBLE) AS feat_b, "
        "CAST(y_rank_5d AS DOUBLE) AS y_rank_5d, "
        "CAST(raw_label_5d AS DOUBLE) AS raw_label_5d, "
        "label_ok_5d, in_broad, in_tradable, common_formation_120d, "
        "common_survivor_120d, ca_mask FROM (VALUES "
        + ",".join(rows)
        + ") t(trade_date, ticker, market, formation_session_idx, feat_a, feat_b, "
        "y_rank_5d, raw_label_5d, label_ok_5d, in_broad, in_tradable, "
        "common_formation_120d, common_survivor_120d, ca_mask)"
    )


_TWO_FEATURE_REGISTRY = [
    {
        "hypothesis_id": "fam_a|feat_a|cum|0|5",
        "family": "fam_a",
        "feature": "feat_a",
        "scan_type": "cum",
        "h_start": 0,
        "h_end": 5,
        "expected_sign": "+",
        "hypothesis_role": "primary",
    },
    {
        "hypothesis_id": "fam_b|feat_b|cum|0|5",
        "family": "fam_b",
        "feature": "feat_b",
        "scan_type": "cum",
        "h_start": 0,
        "h_end": 5,
        "expected_sign": "-",
        "hypothesis_role": "primary",
    },
]

_TWO_FEATURE_SCAN_KWARGS = {
    "sample_start": "2024-01-01",
    "min_names": 5,
    "min_names_for_spread": 5,
    "quantile_count": 5,
    "min_dates_per_cell": 5,
}


@pytest.mark.parametrize("reuse_formation_frames", [True, False])
def test_run_registry_scan_rows_are_unchanged_by_a_sink(reuse_formation_frames: bool) -> None:
    con = duckdb.connect()
    _seed_two_feature_panel(con)
    without = run_registry_scan(
        con,
        _TWO_FEATURE_REGISTRY,
        **_TWO_FEATURE_SCAN_KWARGS,
        reuse_formation_frames=reuse_formation_frames,
    )
    with_sink = run_registry_scan(
        con,
        _TWO_FEATURE_REGISTRY,
        **_TWO_FEATURE_SCAN_KWARGS,
        reuse_formation_frames=reuse_formation_frames,
        daily_sink=_RecordingSink(),
    )
    assert with_sink == without


@pytest.mark.parametrize("reuse_formation_frames", [True, False])
def test_run_registry_scan_flushes_every_feature_exactly_once(
    reuse_formation_frames: bool,
) -> None:
    con = duckdb.connect()
    _seed_two_feature_panel(con)
    sink = _RecordingSink()
    run_registry_scan(
        con,
        _TWO_FEATURE_REGISTRY,
        **_TWO_FEATURE_SCAN_KWARGS,
        reuse_formation_frames=reuse_formation_frames,
        daily_sink=sink,
    )
    assert sorted(sink.flushed) == ["feat_a", "feat_b"]
    assert len(sink.cells) == 8  # 2 hypotheses x 4 universe/sample combos


def test_run_registry_scan_normalizes_the_identity_of_every_emitted_cell() -> None:
    con = duckdb.connect()
    _seed_two_feature_panel(con)
    sink = _RecordingSink()
    run_registry_scan(con, _TWO_FEATURE_REGISTRY, **_TWO_FEATURE_SCAN_KWARGS, daily_sink=sink)
    assert all(tuple(cell) == CELL_IDENTITY_COLUMNS for cell in sink.cells)
    assert {(c["hypothesis_id"], c["universe"], c["sample_kind"]) for c in sink.cells} == {
        (hid, universe, sample_kind)
        for hid in ("fam_a|feat_a|cum|0|5", "fam_b|feat_b|cum|0|5")
        for universe, sample_kind in UNIVERSE_SAMPLE_COMBOS
    }


def test_run_registry_scan_normalizes_a_phase_b_shaped_registry() -> None:
    """Phase B hands ``run_registry_scan`` rows carrying ``role``/``cell_type``
    rather than ``hypothesis_role``; the stored identity is the same schema."""
    con = duckdb.connect()
    _seed_two_feature_panel(con)
    sink = _RecordingSink()
    run_registry_scan(
        con,
        [
            {
                "hypothesis_id": "fin_fam|feat_a|cum|0|5",
                "family": "fin_fam",
                "feature": "feat_a",
                "cell_type": "cumulative",
                "scan_type": "cum",
                "h_start": 0,
                "h_end": 5,
                "role": "ready_primary",
            }
        ],
        **_TWO_FEATURE_SCAN_KWARGS,
        daily_sink=sink,
    )
    assert {cell["hypothesis_role"] for cell in sink.cells} == {"ready_primary"}
    assert {cell["scan_type"] for cell in sink.cells} == {"cum"}
