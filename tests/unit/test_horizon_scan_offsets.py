from __future__ import annotations

import math

import duckdb
import pytest

from modeler.analysis.horizon_scan_runner import (
    build_offset_formation_sql,
    run_nonoverlap_offsets,
    scan_offset,
)


def _seed_offset_panel(con: duckdb.DuckDBPyConnection, *, n_sessions: int = 40) -> None:
    """Same shape as test_horizon_scan_scan_cell's fixture: feature and label
    are strongly (not perfectly) rank-correlated per date×market, so daily IC
    varies session to session instead of being a degenerate constant."""
    rows = []
    for session in range(1, n_sessions + 1):
        d = f"2024-01-01' + {session - 1}"
        for market in ("KOSPI", "KOSDAQ"):
            for t in range(10):
                ticker = f"{market[:1]}{t}"
                feature = float(t) + 0.01 * session
                wobble = 3.0 * math.sin(t + 0.7 * session)
                raw_label = float(t) * 2.0 + wobble
                rows.append(
                    f"(DATE '{d}, '{ticker}', '{market}', {session}, "
                    f"{feature}, {raw_label}, {raw_label}, true, "
                    "true, true, true, true, false)"
                )
    con.execute(
        "CREATE TABLE analysis_panel AS SELECT "
        "trade_date, ticker, market, formation_session_idx, "
        "CAST(px_feature AS DOUBLE) AS px_feature, "
        "CAST(y_rank_5d AS DOUBLE) AS y_rank_5d, "
        "CAST(raw_label_5d AS DOUBLE) AS raw_label_5d, "
        "label_ok_5d, in_broad, in_tradable, common_formation_120d, "
        "common_survivor_120d, ca_mask "
        "FROM (VALUES "
        + ",".join(rows)
        + ") t(trade_date, ticker, market, formation_session_idx, "
        "px_feature, y_rank_5d, raw_label_5d, label_ok_5d, "
        "in_broad, in_tradable, common_formation_120d, common_survivor_120d, ca_mask)"
    )


def test_offset_formation_sql_partitions_sessions_disjointly() -> None:
    con = duckdb.connect()
    _seed_offset_panel(con)
    stride = 5
    counts = []
    seen_sessions: set[int] = set()
    for offset in range(stride):
        sql = build_offset_formation_sql(
            panel_view="analysis_panel",
            feature_col="px_feature",
            scan_type="cum",
            h_start=0,
            h_end=5,
            universe="broad",
            sample_kind="common_survivor",
            sample_start="2024-01-01",
            stride=stride,
            offset=offset,
        )
        rows = con.execute(f"SELECT DISTINCT formation_session_idx FROM ({sql})").fetchall()
        sessions = {r[0] for r in rows}
        assert sessions.isdisjoint(seen_sessions)  # each session belongs to exactly one offset
        seen_sessions |= sessions
        counts.append(len(sessions))
    assert seen_sessions == set(range(1, 41))
    assert sum(counts) == 40


def test_scan_offset_reports_valid_with_ic_and_sign_test() -> None:
    con = duckdb.connect()
    _seed_offset_panel(con)
    result = scan_offset(
        con,
        feature_col="px_feature",
        scan_type="cum",
        h_start=0,
        h_end=5,
        universe="broad",
        sample_kind="common_survivor",
        sample_start="2024-01-01",
        stride=5,
        offset=0,
        min_names=5,
        nonoverlap_min_dates=5,
        alignment_sign=1.0,
    )
    assert result["status"] == "valid"
    assert result["n_dates"] == 8  # 40 sessions / stride 5
    assert result["ic_mean"] > 0
    assert result["n_trials"] is not None and result["n_trials"] <= result["n_dates"]
    assert 0.0 <= result["p_sign_test"] <= 1.0


def test_scan_offset_marks_insufficient_below_nonoverlap_min_dates() -> None:
    con = duckdb.connect()
    _seed_offset_panel(con)
    result = scan_offset(
        con,
        feature_col="px_feature",
        scan_type="cum",
        h_start=0,
        h_end=5,
        universe="broad",
        sample_kind="common_survivor",
        sample_start="2024-01-01",
        stride=5,
        offset=0,
        min_names=5,
        nonoverlap_min_dates=100,  # far above the 8 dates this offset actually has
        alignment_sign=1.0,
    )
    assert result["status"] == "insufficient"
    assert result["p_sign_test"] is None


def test_run_nonoverlap_offsets_covers_every_offset_and_never_picks_one_winner() -> None:
    con = duckdb.connect()
    _seed_offset_panel(con)
    summary = run_nonoverlap_offsets(
        con,
        feature_col="px_feature",
        scan_type="cum",
        h_start=0,
        h_end=5,
        universe="broad",
        sample_kind="common_survivor",
        sample_start="2024-01-01",
        min_names=5,
        nonoverlap_min_dates=5,
        alignment_sign=1.0,
    )
    assert summary["n_offsets_total"] == 5
    assert summary["n_offsets_valid"] == 5
    assert summary["offset_status"] == "complete"
    assert len(summary["offsets"]) == 5  # every offset's own result is preserved
    assert summary["offset_sign_agreement_ratio"] > 0.5  # feature is strongly aligned
    assert (
        summary["offset_ic_mean_min"]
        <= summary["offset_ic_mean_median"]
        <= summary["offset_ic_mean_max"]
    )


def test_run_nonoverlap_offsets_marks_some_insufficient_when_a_bucket_is_thin() -> None:
    con = duckdb.connect()
    _seed_offset_panel(con, n_sessions=40)
    summary = run_nonoverlap_offsets(
        con,
        feature_col="px_feature",
        scan_type="cum",
        h_start=0,
        h_end=5,
        universe="broad",
        sample_kind="common_survivor",
        sample_start="2024-01-01",
        min_names=5,
        nonoverlap_min_dates=9,  # each offset has exactly 8 dates -> all insufficient
        alignment_sign=1.0,
    )
    assert summary["n_offsets_valid"] == 0
    assert summary["offset_status"] == "insufficient"


def test_offset_subsampling_never_reaches_the_daily_ic_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§A-5's offsets have their own IC path and never touch ``scan_cell``, so
    a Stage 0 sink can never be threaded into them by accident — an offset's
    subsampled daily IC is a diagnostic, not the cell's stored series."""
    from modeler.analysis import horizon_scan_runner

    def _fail(*args, **kwargs):
        raise AssertionError("run_nonoverlap_offsets must not call scan_cell")

    monkeypatch.setattr(horizon_scan_runner, "scan_cell", _fail)
    con = duckdb.connect()
    _seed_offset_panel(con)
    summary = run_nonoverlap_offsets(
        con,
        feature_col="px_feature",
        scan_type="cum",
        h_start=0,
        h_end=5,
        universe="broad",
        sample_kind="common_survivor",
        sample_start="2024-01-01",
        min_names=5,
        nonoverlap_min_dates=5,
        alignment_sign=1.0,
    )
    assert summary["n_offsets_total"] == 5
