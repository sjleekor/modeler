"""``dim_peer_monthly`` / ``feat_relation_stat`` — F-2 (02_relationship_features §1).

What has to hold for this family to mean anything:

* the peer set is decided by returns up to a month end and by nothing after it
  (the whole claim to being point-in-time, since the industry alternative is
  not);
* the residual it correlates is ``feat_price``'s residual, not a second
  definition of one — and ``feat_price``'s own SQL text, an A0 cache key, is
  untouched;
* the frozen constants (K=20, 252/189, the 15-of-20 NULL rule) are in the cache
  contract, so changing one cannot silently reuse the old mart;
* a session whose peers mostly did not trade is NULL rather than an average of
  three names.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import duckdb
import numpy as np
import pytest

from modeler.etl.features import price, relation_stat
from modeler.etl.trading_panel import build_market_model_sql

# --------------------------------------------------------------------------
# F-2.1 — the shared residual, and feat_price left alone
# --------------------------------------------------------------------------


def test_the_peer_correlation_reads_feat_prices_own_residual() -> None:
    # Not "an equivalent market model": the same text, so px_idio_vol_60d and
    # the peer sets cannot drift into two ideas of a residual return.
    assert build_market_model_sql("valid") in relation_stat.build_resid_panel_sql()


def test_feat_price_sql_is_byte_identical() -> None:
    """F-2.1 requires sharing the residual *without* touching ``feat_price``.

    The hashes are the ones ``test_research_features.py`` freezes; repeated here
    because this module is the change that could have moved them.
    """

    def _hash(**kwargs) -> str:
        return hashlib.sha256(price.build_price_sql(**kwargs).encode()).hexdigest()

    assert _hash() == "6886ddd45d33c6a13f0d1fd1653a8dd37518bfaa5189ae1227f853bb3b2a4bf3"
    assert (
        _hash(quality_view="dim_price_quality_daily")
        == "110617482d64a50d4ef92ad70defcc23b5b276e2a025452f92a845ccbacec228"
    )


# --------------------------------------------------------------------------
# the pairwise-complete correlation
# --------------------------------------------------------------------------


def test_pairs_are_correlated_on_their_overlap_not_on_zeros() -> None:
    # A stock listed halfway through the window has real gaps. Filling them
    # with zero would report a genuine 1.0 co-movement as roughly 0.7.
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    window = np.column_stack([base, base.copy()])
    window[:50, 1] = np.nan

    rho, joint = relation_stat._pairwise_complete_corr(window)

    assert joint[0, 1] == 50
    assert rho[0, 1] == pytest.approx(1.0)


def test_a_constant_series_has_no_correlation_rather_than_a_spurious_one() -> None:
    window = np.column_stack([np.arange(60.0), np.ones(60)])

    rho, _ = relation_stat._pairwise_complete_corr(window)

    assert np.isnan(rho[0, 1])


def test_a_thin_overlap_is_not_a_peer_candidate() -> None:
    rho = np.array([[1.0, 0.9], [0.9, 1.0]])
    joint = np.array([[252, relation_stat.MIN_VALID_SESSIONS - 1], [1, 252]])

    assert list(relation_stat._top_peers(rho, joint, 5)) == []


def test_peers_come_back_in_descending_rho_order() -> None:
    rho = np.array(
        [
            [1.0, 0.1, 0.9, 0.5],
            [0.1, 1.0, 0.2, 0.3],
            [0.9, 0.2, 1.0, 0.4],
            [0.5, 0.3, 0.4, 1.0],
        ]
    )
    joint = np.full((4, 4), 252)

    first = next(iter(relation_stat._top_peers(rho, joint, 2)))
    _, order, rhos, _ = first

    assert list(order) == [2, 3]
    assert list(rhos) == [0.9, 0.5]


def test_a_stock_is_never_its_own_peer() -> None:
    rho = np.array([[1.0, 0.2], [0.2, 1.0]])
    joint = np.full((2, 2), 252)

    for i, order, _, _ in relation_stat._top_peers(rho, joint, 2):
        assert i not in order


# --------------------------------------------------------------------------
# the big-cap rank inside K=20
# --------------------------------------------------------------------------


def test_the_bigcap_rank_is_taken_inside_the_k20_set_only() -> None:
    # Ranking over all 40 stored peers and then filtering to rank<=20 would
    # leave a stock whose largest names sit at rank 21+ with fewer than five.
    n = 45
    rho = np.zeros((n, n))
    rho[0] = np.linspace(0.9, 0.1, n)  # peer rank follows column order
    joint = np.full((n, n), 252)
    caps = np.arange(float(n))  # the *last* columns are the biggest

    batch = relation_stat._collect_month(rho, joint, caps)

    ranked = [
        (rank, mcap_rank)
        for self_pos, rank, mcap_rank in zip(
            batch.self_pos, batch.rank, batch.mcap_rank, strict=True
        )
        if self_pos == 0
    ]
    top5 = sorted(rank for rank, mcap_rank in ranked if mcap_rank is not None and mcap_rank <= 5)
    assert len(top5) == relation_stat.BIGCAP_N
    assert max(top5) <= relation_stat.PEER_K


def test_a_peer_without_a_market_cap_is_never_a_large_cap() -> None:
    n = 25
    rho = np.zeros((n, n))
    rho[0] = np.linspace(0.9, 0.1, n)
    joint = np.full((n, n), 252)
    caps = np.full(float(n).__int__(), np.nan)
    caps[3] = 10.0

    batch = relation_stat._collect_month(rho, joint, caps)

    ranks = [
        mcap_rank
        for self_pos, mcap_rank in zip(batch.self_pos, batch.mcap_rank, strict=True)
        if self_pos == 0
    ]
    assert sum(1 for r in ranks if r is not None) == 1


# --------------------------------------------------------------------------
# end-to-end on a synthetic panel
# --------------------------------------------------------------------------


def _synthetic_ohlcv(
    con: duckdb.DuckDBPyConnection,
    *,
    sessions: int,
    tickers: int,
    seed: int = 7,
    start: dt.date = dt.date(2020, 1, 1),
) -> None:
    """A tradeable panel: no halts, one market, geometric closes from noise."""
    rng = np.random.default_rng(seed)
    dates = [start + dt.timedelta(days=i) for i in range(sessions)]
    rows: list[tuple] = []
    for t in range(tickers):
        # Two blocks that co-move within themselves, so peer sets are not noise.
        block = rng.normal(size=sessions) * 0.01
        own = rng.normal(size=sessions) * 0.004
        shared = rng.normal(size=sessions) * 0.01 if t % 2 == 0 else block
        close = 10_000 * np.exp(np.cumsum(shared + own))
        for i, d in enumerate(dates):
            price_i = int(close[i])
            rows.append((d, f"{t:06d}", "KOSPI", price_i, price_i, price_i, price_i, 1_000))

    con.execute(
        "CREATE OR REPLACE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open BIGINT, high BIGINT, low BIGINT, close BIGINT, volume BIGINT)"
    )
    con.executemany("INSERT INTO daily_ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute(
        "CREATE OR REPLACE TABLE daily_market_cap (trade_date DATE, ticker VARCHAR, "
        "market VARCHAR, market_cap BIGINT)"
    )
    con.execute(
        "INSERT INTO daily_market_cap "
        "SELECT trade_date, ticker, market, close * 1000 FROM daily_ohlcv"
    )
    con.execute("CREATE OR REPLACE TABLE dart_corp_master (ticker VARCHAR, induty_code VARCHAR)")
    con.executemany(
        "INSERT INTO dart_corp_master VALUES (?, ?)",
        [(f"{t:06d}", "26100" if t % 2 == 0 else "58210") for t in range(tickers)],
    )


def _peer_panel(sessions: int = 620, tickers: int = 12) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    _synthetic_ohlcv(con, sessions=sessions, tickers=tickers)
    relation_stat.compute_peer_monthly(con, table_name="dim_peer_monthly")
    return con


def test_the_month_end_axis_is_the_last_session_of_each_month() -> None:
    con = duckdb.connect()
    _synthetic_ohlcv(con, sessions=70, tickers=3)

    ends = [
        row[0]
        for row in con.execute(
            f"SELECT month_end FROM ({relation_stat.build_month_end_sql()})"
        ).fetchall()
    ]

    assert ends[0] == dt.date(2020, 1, 31)
    assert ends[1] == dt.date(2020, 2, 29)


def test_the_bigcap_market_cap_predates_the_month_end() -> None:
    # The endpoint publishes T+1, so the month end's own row is not readable in
    # time; the rank must come from the session before it.
    con = duckdb.connect()
    _synthetic_ohlcv(con, sessions=70, tickers=3)

    rows = con.execute(
        f"SELECT month_end, ticker, market_cap FROM ({relation_stat.build_peer_mcap_sql()}) "
        "WHERE month_end = DATE '2020-01-31' AND ticker = '000000'"
    ).fetchall()
    expected = con.execute(
        "SELECT market_cap FROM daily_market_cap "
        "WHERE ticker = '000000' AND trade_date = DATE '2020-01-30'"
    ).fetchone()

    assert len(rows) == 1
    assert rows[0][2] == pytest.approx(float(expected[0]))


def test_peer_sets_do_not_move_when_returns_after_t0_change() -> None:
    """The PIT test §1.5 asks for: rewrite the future, get the same peers."""
    con = _peer_panel()
    before = con.execute(
        "SELECT month_end, ticker, peer_ticker, peer_rank FROM dim_peer_monthly ORDER BY ALL"
    ).fetchall()
    cutoff = con.execute("SELECT max(month_end) FROM dim_peer_monthly").fetchone()[0]

    # Replace every close after the last peer month end with noise of a
    # completely different correlation structure.
    con.execute(
        "UPDATE daily_ohlcv SET close = close + CAST(1000 * random() AS BIGINT) "
        "WHERE trade_date > ?",
        [cutoff],
    )
    relation_stat.compute_peer_monthly(con, table_name="dim_peer_monthly_after")
    after = con.execute(
        "SELECT month_end, ticker, peer_ticker, peer_rank FROM dim_peer_monthly_after ORDER BY ALL"
    ).fetchall()

    assert after == before
    assert before  # the fixture actually produced peer sets


def test_a_stored_peer_set_holds_k_stored_names_ranked_from_one() -> None:
    con = _peer_panel(tickers=60)

    rows = con.execute(
        "SELECT count(*), min(peer_rank), max(peer_rank) FROM dim_peer_monthly "
        "GROUP BY month_end, ticker, market"
    ).fetchall()

    for count, lo, hi in rows:
        assert lo == 1
        assert count == hi
        assert hi <= relation_stat.PEER_K_STORED


def test_a_stock_is_only_a_candidate_with_enough_usable_residuals() -> None:
    # 252 sessions of residual need 252 sessions of market model before them,
    # so a panel shorter than that produces nothing at all rather than a peer
    # set built on a partial window.
    con = duckdb.connect()
    _synthetic_ohlcv(con, sessions=400, tickers=8)
    relation_stat.compute_peer_monthly(con, table_name="dim_peer_monthly")

    assert con.execute("SELECT count(*) FROM dim_peer_monthly").fetchone()[0] == 0


# --------------------------------------------------------------------------
# the daily mart
# --------------------------------------------------------------------------


def _relation_rows(con: duckdb.DuckDBPyConnection):
    relation_stat.register_relation_stat_view(con)
    return con


def test_the_daily_mart_emits_the_documented_columns() -> None:
    con = _relation_rows(_peer_panel())

    columns = [row[0] for row in con.execute("DESCRIBE feat_relation_stat").fetchall()]

    assert columns[:3] == ["trade_date", "ticker", "market"]
    for name in relation_stat.FEATURE_COLUMNS + relation_stat.COUNT_COLUMNS:
        assert name in columns
    # The scan config requires a variant_columns.lag1 mapping per family, so
    # every preregistered primary needs a real lag1 column to point at.
    for name in relation_stat.PRIMARY_COLUMNS:
        assert f"{name}_lag1" in columns


def test_a_session_reads_the_peer_set_of_an_earlier_month_end() -> None:
    con = _relation_rows(_peer_panel())

    rows = con.execute(
        "SELECT trade_date, rel_peer_month_end FROM feat_relation_stat "
        "WHERE rel_peer_month_end IS NOT NULL"
    ).fetchall()

    assert rows
    for trade_date, month_end in rows:
        assert month_end < trade_date


def test_the_first_session_of_a_peer_month_still_uses_the_previous_set() -> None:
    con = _relation_rows(_peer_panel())
    month_ends = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT month_end FROM dim_peer_monthly ORDER BY 1"
        ).fetchall()
    ]
    assert len(month_ends) >= 2

    # t0 itself is still on the previous month's set — its own return helped
    # choose the new one.
    used = con.execute(
        "SELECT DISTINCT rel_peer_month_end FROM feat_relation_stat WHERE trade_date = ?",
        [month_ends[1]],
    ).fetchall()

    assert [row[0] for row in used] == [month_ends[0]]


def test_own_minus_peer_is_exactly_own_return_minus_the_peer_mean() -> None:
    con = _relation_rows(_peer_panel(tickers=60))

    bad = con.execute(
        "SELECT count(*) FROM feat_relation_stat "
        "WHERE rel_own_minus_peer_20d IS NOT NULL "
        "  AND abs(rel_own_minus_peer_20d - (rel_own_ret_20d - rel_peer_mom_20d)) > 1e-12"
    ).fetchone()[0]

    assert bad == 0


def test_a_thin_peer_day_is_null_rather_than_an_average_of_three_names() -> None:
    con = _relation_rows(_peer_panel(tickers=60))

    violations = con.execute(
        "SELECT count(*) FROM feat_relation_stat "
        f"WHERE rel_peer_n_20d < {relation_stat.MIN_PEER_VALID} "
        "  AND rel_peer_mom_20d IS NOT NULL"
    ).fetchone()[0]
    thin_bigcap = con.execute(
        "SELECT count(*) FROM feat_relation_stat "
        f"WHERE rel_peer_n_bigcap < {relation_stat.MIN_BIGCAP_VALID} "
        "  AND rel_peer_bigcap_lag_ret_5d IS NOT NULL"
    ).fetchone()[0]

    assert violations == 0
    assert thin_bigcap == 0


def test_the_mart_carries_only_valid_sessions() -> None:
    con = _peer_panel()
    con.execute(
        "INSERT INTO daily_ohlcv VALUES (DATE '2021-09-14', '000000', 'KOSPI', 0, 0, 0, 5000, 0)"
    )
    _relation_rows(con)

    halted = con.execute(
        "SELECT count(*) FROM feat_relation_stat "
        "WHERE trade_date = DATE '2021-09-14' AND ticker = '000000'"
    ).fetchone()[0]

    assert halted == 0


def test_the_yearly_parts_and_the_one_shot_query_agree() -> None:
    # materialize_relation_stat writes a year at a time off a pre-built returns
    # panel. If that path diverged from the contract SQL the mart's own cache
    # hash would be describing a query it never ran.
    con = _peer_panel(tickers=40)
    con.execute(
        f"CREATE TEMP TABLE _relation_return_panel AS {relation_stat.build_return_panel_sql()}"
    )

    def _part(year: int) -> str:
        # Exactly the shape materialize_relation_stat builds: read one year
        # wider than emitted so the lag1 window crosses the part boundary.
        return relation_stat.build_relation_stat_sql(
            return_panel_view="_relation_return_panel",
            session_filter=f"year(r.trade_date) IN ({year - 1}, {year})",
            emit_filter=f"year(trade_date) = {year}",
        )

    one_shot = relation_stat.build_relation_stat_sql()
    chunked = " UNION ALL ".join(f"({_part(year)})" for year in (2020, 2021, 2022))
    # Compared with a tolerance, not by EXCEPT: DuckDB's parallel AVG over
    # doubles is not bit-reproducible, so an exact set difference would fail on
    # last-bit noise and say nothing about the partitioning.
    columns = (
        relation_stat.FEATURE_COLUMNS
        + relation_stat.COUNT_COLUMNS
        + tuple(f"{c}_lag1" for c in relation_stat.PRIMARY_COLUMNS)
    )
    worst = ", ".join(
        f"max(abs(coalesce(a.{c}, 0) - coalesce(b.{c}, 0))) AS d_{c}, "
        f"count(*) FILTER (WHERE (a.{c} IS NULL) <> (b.{c} IS NULL)) AS n_{c}"
        for c in columns
    )
    row = con.execute(
        f"SELECT count(*), {worst} FROM ({one_shot}) a "
        f"JOIN ({chunked}) b USING (trade_date, ticker, market)"
    ).fetchone()
    matched = con.execute(f"SELECT count(*) FROM ({chunked})").fetchone()[0]

    assert row[0] == matched > 0
    for value in row[1:]:
        assert value == pytest.approx(0, abs=1e-9)


def test_lag1_is_the_tickers_own_previous_session() -> None:
    con = _relation_rows(_peer_panel())

    mismatched = con.execute(f"""
        SELECT count(*) FROM (
            SELECT rel_peer_mom_20d_lag1,
                   CASE WHEN date_diff('day', LAG(trade_date) OVER w, trade_date)
                             <= {relation_stat.LAG1_MAX_GAP_DAYS}
                        THEN LAG(rel_peer_mom_20d) OVER w END AS expected
            FROM feat_relation_stat
            WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
        )
        WHERE rel_peer_mom_20d_lag1 IS DISTINCT FROM expected
        """).fetchone()[0]

    assert mismatched == 0


def test_the_lag1_gap_bound_is_in_the_sql_and_covers_a_part_lookback() -> None:
    # Why the bound is what makes the yearly parts reproduce the unpartitioned
    # contract: a part reads its own year plus the previous one, so the oldest
    # predecessor the guard admits for the year's first session — 365 days
    # back — is still inside the part. Drop the guard and a stock returning
    # from a multi-year suspension gets a lag1 that depends on nothing but
    # where the partition boundary fell.
    assert relation_stat.LAG1_MAX_GAP_DAYS == 365
    assert (
        f"date_diff('day', LAG(trade_date) OVER w, trade_date)"
        f" <= {relation_stat.LAG1_MAX_GAP_DAYS}" in relation_stat.build_relation_stat_sql()
    )
    earliest_admitted = dt.date(2021, 1, 1) - dt.timedelta(days=relation_stat.LAG1_MAX_GAP_DAYS)
    assert earliest_admitted >= dt.date(2020, 1, 1)


def test_a_part_boundary_does_not_null_out_the_first_sessions_lag1() -> None:
    # The failure this guards is silent: WHERE runs before the window, so a
    # part filtered to its own year alone would hand January's first session a
    # NULL lag1 at every boundary and still look like a complete mart.
    # A longer panel than _peer_panel's: the peer sets need 252 + 252 sessions
    # before the first value, so the default fixture has no year boundary with
    # anything on both sides of it.
    con = duckdb.connect()
    _synthetic_ohlcv(con, sessions=900, tickers=16, start=dt.date(2019, 1, 1))
    relation_stat.compute_peer_monthly(con, table_name="dim_peer_monthly")
    con.execute(
        f"CREATE TEMP TABLE _relation_return_panel AS {relation_stat.build_return_panel_sql()}"
    )

    def _first_session_lag1_nulls(session_filter: str) -> tuple[int, int]:
        sql = relation_stat.build_relation_stat_sql(
            return_panel_view="_relation_return_panel",
            session_filter=session_filter,
            emit_filter="year(trade_date) = 2021",
        )
        return con.execute(f"""
            WITH p AS ({sql}), f AS (SELECT min(trade_date) AS d FROM p)
            SELECT count(*), count(*) FILTER (WHERE p.rel_peer_mom_20d_lag1 IS NULL)
            FROM p JOIN f ON p.trade_date = f.d
            """).fetchone()

    widened_rows, widened_nulls = _first_session_lag1_nulls("year(r.trade_date) IN (2020, 2021)")
    narrow_rows, narrow_nulls = _first_session_lag1_nulls("year(r.trade_date) = 2021")

    assert widened_rows > 0
    assert widened_nulls == 0
    # The same part read only within its own year loses every lag1 on that day.
    assert narrow_nulls == narrow_rows == widened_rows


def test_the_returns_panel_the_parts_read_is_the_contracts_own_text() -> None:
    assert relation_stat.build_return_panel_sql() in relation_stat.build_relation_stat_sql()


# --------------------------------------------------------------------------
# the cache contract
# --------------------------------------------------------------------------


def test_the_frozen_constants_are_in_the_cache_contract() -> None:
    # A changed K or window must invalidate dim_peer_monthly: the peer sets are
    # numpy output, so the contract text is the only place the rules appear.
    sql = relation_stat.peer_monthly_contract_sql()

    for value in (
        relation_stat.FORMULA_VERSION,
        str(relation_stat.CORR_WINDOW_SESSIONS),
        str(relation_stat.MIN_VALID_SESSIONS),
        str(relation_stat.PEER_K),
        str(relation_stat.PEER_K_STORED),
    ):
        assert value in sql


def test_the_null_rules_are_in_the_daily_marts_sql() -> None:
    sql = relation_stat.build_relation_stat_sql()

    assert relation_stat.FORMULA_VERSION in sql
    assert f">= {relation_stat.MIN_PEER_VALID}" in sql
    assert f">= {relation_stat.MIN_BIGCAP_VALID}" in sql
