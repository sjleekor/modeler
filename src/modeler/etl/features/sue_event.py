"""``fin_sue_event`` — Phase B B-6 (04_specific_plan_B.md §4.6, §5.2, B-6).

Grain: ``(ticker, original_rcept_no, event_formation_date, market)`` — one row
per *original* earnings-announcement event, not per calendar day (§5.2). This
is the one Phase B mart that is event-time, not daily-continuous: every other
B-3..B-5 mart broadcasts a PIT value across every trading day until the next
one supersedes it; this mart instead measures the market's reaction to one
specific announcement over a fixed 60-session window after it.

Built entirely from B-2/B-3 outputs plus ``daily_ohlcv`` — no new raw
extraction:

    original-event determination   B-2's ``captured_vintage_status``/
                                     ``is_revision`` (already receipt-relation-
                                     based, §3.5) — an event only exists here
                                     when B-2 could actually confirm it, never
                                     from a bare captured rcept_no (§3.5: "raw에
                                     우연히 남은 최소 rcept_no를 original로
                                     간주하지 않는다").
    same-day revision collapse      B-2's ``same_day_effective_rcept_no``.
    event_formation_date            B-2's ``available_from`` for that vintage
                                     (already "next KRX session after
                                     disclosure").
    quarterly EPS / comparative EPS B-3's standalone ``controlling_net_income``
                                     and ``weighted_avg_shares``, plus B-3's
                                     ``value_lag_4q`` for both.

Comparative EPS — a deliberate scope reduction (see also ``metric_vintages.py``
and ``event_scan.py`` docstrings for the same honesty pattern): §4.6's
*primary* method reconstructs comparative weighted-average shares from a
separate "prior-year" XBRL duration context inside the *same* filing. This
repository does not parse multiple XBRL contexts per concept per filing (B-2
takes one captured value per concept), so that reconstruction is not
buildable without new XBRL-parsing work. This module instead always uses what
§4.6 itself defines as the secondary ``as_was_comparative`` method — both the
net income and the shares exactly as they were reported 4 quarters ago (B-3's
``value_lag_4q``) — labeled ``comparative_policy='as_was_lag4q'`` in the
output so nothing claims to be the primary frmtrm_q_amount-based figure it
is not.

Market-excess convention matches ``research/etl/labels.py``'s ``eqw_market``
benchmark (equal-weighted mean of the same window's return across every
ticker with a valid session that day) — reimplemented here rather than
imported, consistent with every other Phase B mart's independence from
non-Phase-B modules.

The 60-session revision-contamination check (event rule 5) compares *session
index* distance, not calendar-day distance: a later vintage's own
``available_from`` (already next-session-adjusted by B-2) is looked up in the
same per-ticker session index used for returns, so "60 sessions" means
exactly 60 trading sessions regardless of holidays in between.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

SUE_EVENT_TABLE = "fin_sue_event"

# §2.1/§4.6 frozen event-time grid — must match phase_b.event_buckets
# (horizon_scan_config.yaml, validated in horizon_scan_config.py).
EVENT_BUCKETS: tuple[tuple[int, int], ...] = ((0, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 60))
_BUCKET_OFFSETS = sorted({offset for bucket in EVENT_BUCKETS for offset in bucket})

MIN_SUE_HISTORY = 8
PRIMARY_WINDOW_SESSIONS = 60


def _bucket_col(h1: int, h2: int, suffix: str) -> str:
    return f"bucket_{h1}_{h2}_{suffix}"


def _quarter_ordinal_case_sql(col: str) -> str:
    return (
        f"CASE {col} WHEN '11013' THEN 1 WHEN '11012' THEN 2 "
        "WHEN '11014' THEN 3 WHEN '11011' THEN 4 END"
    )


def build_sue_event_sql(
    *,
    vintage_view: str = "stock_metric_vintage_fact",
    quarterly_view: str = "fin_quarterly_metric_vintage",
    price_view: str = "daily_ohlcv",
    quality_view: str = "dim_price_quality_daily",
) -> str:
    """SQL producing ``fin_sue_event`` from B-2/B-3's outputs and daily prices."""
    offset_joins = "\n".join(
        f"""
        LEFT JOIN px p{o}
          ON p{o}.ticker = f.ticker AND p{o}.market = f.market AND p{o}.d_idx = f.d_idx + {o}"""
        for o in _BUCKET_OFFSETS
        if o != 0
    )
    bucket_raw_cols = ",\n            ".join(
        (
            f"p{h2}.close_d / NULLIF(f.close_d, 0) - 1"
            if h1 == 0
            else f"p{h2}.close_d / NULLIF(p{h1}.close_d, 0) - 1"
        )
        + f" AS {_bucket_col(h1, h2, 'raw')}"
        for h1, h2 in EVENT_BUCKETS
    )
    bucket_ca_cols = ",\n            ".join(f"""EXISTS (
                SELECT 1 FROM px_quality pc
                WHERE pc.ticker = f.ticker AND pc.market = f.market
                  AND pc.d_idx > f.d_idx + {h1} AND pc.d_idx <= f.d_idx + {h2}
                  AND pc.ca_mask
            ) AS {_bucket_col(h1, h2, "ca_contaminated")}""" for h1, h2 in EVENT_BUCKETS)
    bench_cols = ",\n            ".join(
        f"AVG({_bucket_col(h1, h2, 'raw')}) OVER (PARTITION BY trade_date, market)"
        f" AS {_bucket_col(h1, h2, 'bench')}"
        for h1, h2 in EVENT_BUCKETS
    )
    final_bucket_cols = ",\n        ".join(
        f"br.{_bucket_col(h1, h2, 'raw')},\n        "
        f"br.{_bucket_col(h1, h2, 'raw')} - br.{_bucket_col(h1, h2, 'bench')}"
        f" AS {_bucket_col(h1, h2, 'excess')},\n        "
        f"br.{_bucket_col(h1, h2, 'ca_contaminated')}"
        for h1, h2 in EVENT_BUCKETS
    )
    any_ca_contaminated = " OR ".join(
        f"br.{_bucket_col(h1, h2, 'ca_contaminated')}" for h1, h2 in EVENT_BUCKETS
    )
    any_bucket_missing = " OR ".join(
        f"br.{_bucket_col(h1, h2, 'raw')} IS NULL" for h1, h2 in EVENT_BUCKETS
    )

    return f"""
    WITH ni_lineage AS (
        -- net_income's own vintage row: disclosed_date/same_day/captured_status
        -- lineage and reprt_code that fin_quarterly_metric_vintage (B-3) does
        -- not carry (B-3 already translated reprt_code into quarter_ordinal).
        SELECT
            ticker, fs_basis, bsns_year, reprt_code,
            {_quarter_ordinal_case_sql("reprt_code")} AS quarter_ordinal,
            rcept_no, disclosed_date, available_from, same_day_effective_rcept_no,
            is_revision, captured_vintage_status
        FROM {vintage_view}
        WHERE metric_code = 'controlling_net_income'
    ),
    -- weighted_avg_shares is XBRL-sourced (B-2 sets fs_basis='' for every
    -- dart_xbrl_fact_raw-derived metric — the raw XBRL fact table carries no
    -- CFS/OFS dimension at all), so it is joined on ticker/quarter only, not
    -- fs_basis; only controlling_net_income (dart_financial_statement_raw)
    -- carries a real CFS/OFS split here.
    shares AS (
        SELECT ticker, bsns_year, quarter_ordinal,
               standalone_value AS shares, value_lag_4q AS shares_lag4q
        FROM {quarterly_view}
        WHERE metric_code = 'weighted_avg_shares'
    ),
    eps AS (
        SELECT
            q.ticker, q.market, q.corp_code, q.bsns_year, l.reprt_code, q.fs_basis, q.seq_key,
            q.statement_period_end, q.rcept_no,
            l.disclosed_date, l.available_from, l.same_day_effective_rcept_no,
            l.is_revision, l.captured_vintage_status,
            q.standalone_value / NULLIF(s.shares, 0) AS quarterly_eps,
            q.value_lag_4q / NULLIF(s.shares_lag4q, 0) AS comparative_eps
        FROM {quarterly_view} q
        JOIN ni_lineage l
          ON l.ticker = q.ticker AND l.fs_basis = q.fs_basis
         AND l.bsns_year = q.bsns_year AND l.quarter_ordinal = q.quarter_ordinal
        JOIN shares s
          ON s.ticker = q.ticker
         AND s.bsns_year = q.bsns_year AND s.quarter_ordinal = q.quarter_ordinal
        WHERE q.metric_code = 'controlling_net_income'
    ),
    -- §3.5: only a receipt-relation-confirmed vintage is an "original event".
    original_events AS (
        SELECT *, (quarterly_eps - comparative_eps) AS seasonal_change
        FROM eps
        WHERE captured_vintage_status = 'original_confirmed_revisions_partial'
          AND is_revision = FALSE
    ),
    with_history AS (
        SELECT
            *,
            STDDEV_SAMP(seasonal_change) OVER (
                PARTITION BY ticker, fs_basis ORDER BY seq_key
                ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
            ) AS history_stddev,
            COUNT(seasonal_change) OVER (
                PARTITION BY ticker, fs_basis ORDER BY seq_key
                ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
            ) AS history_count
        FROM original_events
        WHERE seasonal_change IS NOT NULL
    ),
    -- event rule 5: a later revision of this same filing position is a
    -- contamination risk only if it becomes knowable within 60 *sessions* —
    -- resolved below via session-index distance, not calendar days.
    revisions AS (
        SELECT ticker, fs_basis, bsns_year, reprt_code, available_from AS revision_available_from
        FROM {vintage_view}
        WHERE metric_code = 'controlling_net_income' AND is_revision = TRUE
    ),
    px AS (
        SELECT trade_date, ticker, market, CAST(close AS DOUBLE) AS close_d,
               ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS d_idx
        FROM {price_view}
        WHERE NOT (open = 0 AND high = 0 AND low = 0)
    ),
    px_quality AS (
        SELECT p.trade_date, p.ticker, p.market, p.d_idx, p.close_d,
               COALESCE(q.ca_mask, FALSE) AS ca_mask
        FROM px p
        LEFT JOIN {quality_view} q USING (trade_date, ticker, market)
    ),
    events_with_formation AS (
        SELECT wh.*, f.trade_date AS formation_trade_date, f.d_idx AS formation_d_idx
        FROM with_history wh
        JOIN px_quality f ON f.ticker = wh.ticker AND f.trade_date = wh.available_from
    ),
    bucket_returns AS (
        SELECT
            f.ticker, f.market, f.trade_date, f.d_idx,
            {bucket_raw_cols},
            {bucket_ca_cols}
        FROM px_quality f
        {offset_joins}
    ),
    bucket_scored AS (
        SELECT *,
            {bench_cols}
        FROM bucket_returns
    ),
    revision_check AS (
        SELECT
            ewf.ticker, ewf.fs_basis, ewf.bsns_year, ewf.reprt_code, ewf.formation_d_idx,
            MIN(rp.d_idx) AS earliest_revision_d_idx
        FROM events_with_formation ewf
        JOIN revisions r
          ON r.ticker = ewf.ticker AND r.fs_basis = ewf.fs_basis
         AND r.bsns_year = ewf.bsns_year AND r.reprt_code = ewf.reprt_code
        JOIN px_quality rp ON rp.ticker = r.ticker AND rp.trade_date = r.revision_available_from
        WHERE rp.d_idx > ewf.formation_d_idx
        GROUP BY ewf.ticker, ewf.fs_basis, ewf.bsns_year, ewf.reprt_code, ewf.formation_d_idx
    )
    SELECT
        ewf.ticker, ewf.market, ewf.corp_code, ewf.rcept_no AS original_rcept_no,
        ewf.bsns_year, ewf.reprt_code, ewf.fs_basis,
        ewf.disclosed_date AS original_disclosed_date,
        ewf.available_from AS event_formation_date,
        'as_was_lag4q' AS comparative_policy,
        ewf.quarterly_eps, ewf.comparative_eps, ewf.seasonal_change,
        ewf.history_count AS sue_history_count,
        CASE WHEN ewf.history_count >= {MIN_SUE_HISTORY} AND ewf.history_stddev > 0
             THEN ewf.seasonal_change / ewf.history_stddev
        END AS fin_sue,
        (
            rc.earliest_revision_d_idx IS NOT NULL
            AND rc.earliest_revision_d_idx - ewf.formation_d_idx <= {PRIMARY_WINDOW_SESSIONS}
        ) AS revision_within_60_sessions,
        (
            NOT ({any_bucket_missing})
            AND NOT ({any_ca_contaminated})
            AND NOT (
                rc.earliest_revision_d_idx IS NOT NULL
                AND rc.earliest_revision_d_idx - ewf.formation_d_idx <= {PRIMARY_WINDOW_SESSIONS}
            )
        ) AS is_primary_constant_sample,
        {final_bucket_cols}
    FROM events_with_formation ewf
    LEFT JOIN bucket_scored br
      ON br.ticker = ewf.ticker AND br.trade_date = ewf.formation_trade_date
    LEFT JOIN revision_check rc
      ON rc.ticker = ewf.ticker AND rc.fs_basis = ewf.fs_basis
     AND rc.bsns_year = ewf.bsns_year AND rc.reprt_code = ewf.reprt_code
    """


def register_sue_event_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = SUE_EVENT_TABLE,
    **views: str,
) -> str:
    """Register a DuckDB view over the SQL above (no parquet — tests/parity)."""
    sql = build_sue_event_sql(**views)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_sue_event(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    force: bool = False,
    **views: str,
) -> str:
    """Build + register ``fin_sue_event`` as a cached parquet mart."""
    materialize(
        con,
        config,
        SUE_EVENT_TABLE,
        build_sue_event_sql(**views),
        force=force,
    )
    return register_mart_view(con, config, SUE_EVENT_TABLE)
