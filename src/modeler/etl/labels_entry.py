# ruff: noqa: E501
"""Entry-timing labels (c1 / o1) beside the legacy c0 label — quality-fix plan §3.

``etl/labels.py`` is part of the model02 holdout-opening code hash and must not
change, so the entry logic lives here and wraps :func:`labels.build_label_sql`.

``build_entry_label_sql(spec, entry, ...)`` returns the c0 label (exactly what
``build_label_sql(spec)`` makes) with a second column family added beside it.

Entry / exit
  - Enter at the close (``c1``) / open (``o1``) of the next *market* trading day
    t+1 (calendar = distinct dates of the price view per market; a halt does NOT
    push the entry back). Exit is the h-th non-halt session of the ticker counted
    from the entry day (``o1`` exits at that session's open).
  - Entry halted / missing / locked limit-up -> NULL, kind ``untradable``. t is the
    market's last date (no t+1 yet) -> NULL, kind ``censored``.
  - Exit day locked limit-down (open=high=low=close, ret <= -(limit-1%p)) is NOT
    nulled: the exit moves to the next non-halt, non-locked session of the ticker.
    If none exists the no-exit rule below applies. Limit: 15% before 2015-06-15,
    30% after (same values as ``etl/quality.py``).
  - No such exit session:
      * series ENDED (last row of the ticker is at least ``min_end_gap_sessions``
        market sessions before the market's last date; with ``master_view`` also
        a DELISTED row and no ACTIVE row of the ticker in any market, so a
        market transfer is not a delisting) -> ``terminal``: exit price = the
        series' last non-halt close (also for o1). Exception, the series ends
        with >= ``end_halt_sessions`` halt rows and no trade after them (no
        정리매매 prices) -> exit price = last close * (1 - ``end_halt_haircut``)
        (``terminal_halted``); with ``end_halt_haircut=None`` the label is NULL
        (``terminal_unpriced``), so the caller has to choose explicitly. The
        price shape alone cannot tell distress from a merger or share swap: of
        the 11 KR series ending this way (last row 2016-01..2025-07) only 3 were
        distress delistings, 7 mergers/swaps. Plan
        ``15_quality_fix_plan.md`` §9.3 therefore uses 0.0 as the main value
        and 0.962 (1 - median 0.0379 measured on 124 priced distress series) as
        a sensitivity.
      * series not ended (panel end, or still halted at panel end) -> NULL,
        ``censored``.
  - Corporate-action mask: any ``ca_event`` in [t+1, exit] (needs ``quality_view``;
    without it no mask) -> NULL, ``ca_masked``. Rows of a *terminal tail* are
    ignored: the non-halt rows after the last halt row of an ENDED series, when
    there are 1..``terminal_tail_max`` of them (정리매매 crashes look like events).
    Interpretation: a series with no halt row at all has no tail.

Columns (``tag`` in {c1, o1}; per horizon h)
  ``fwd_ret_{tag}_{h}d``, ``bench_ret_{tag}_{h}d`` (eqw market, terminal included),
  ``raw_label_{tag}_{h}d``, ``y_{rank,reg,cls,up,top}_{tag}_{h}d`` for
  ``spec.outputs``, and
  - ``label_exit_kind_{tag}_{h}d``: regular | late | terminal | terminal_halted |
    terminal_unpriced | censored | untradable | ca_masked
  - ``label_end_date_{tag}_{h}d``: actual exit trade date (terminal kinds: the
    series' last row date). NULL for censored / untradable.
  - ``label_exit_lag_{tag}_{h}d``: market sessions from entry date to that date
    minus h, floored at 0 (a terminal exit can be nearer than h sessions).
    ``regular`` = lag 0, ``late`` = valid label with lag > 0.
  Per row: ``in_terminal_tail`` (row t itself is in a terminal tail; a universe
  exclusion for the next model, not applied here). With ``spec.index_bench`` and
  c1 also ``bench_ret_idx_c1_{h}d`` / ``raw_label_idx_c1_{h}d`` (index close on
  the entry date and on the actual exit date). No index open series exists, so o1
  has no index columns. The legacy ``y_*`` columns still come from c0.

Parameters: ``min_end_gap_sessions`` (1), ``master_view`` (None; ticker, market,
status), ``end_halt_sessions`` (20), ``end_halt_haircut`` (None),
``terminal_tail_max`` (10).

Leakage note: terminal values use knowledge of the delisting, which is allowed
for labels but never for features. A label can end hundreds of sessions after t
(long halts), so training and validation splits must purge by
``label_end_date_*``, not by h.

The row set is every non-halt (t, ticker) row, so the c0 columns are NULL on
rows the c0 inner join would drop. The dataset therefore differs from
``label_daily``: materialize it under its own name (``label_entry_v2``).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.labels import LabelSpec, _index_cte, build_label_sql
from modeler.etl.mart import materialize, register_mart_view

LABEL_ENTRY_TABLE = "label_entry_v2"
_VALID_ENTRY = ("c1", "o1")

# Daily price limit (same values as the inline CASE in etl/quality.py).
_LIMIT_CHANGE_DATE = "2015-06-15"
_LIMIT_BEFORE = 0.15
_LIMIT_AFTER = 0.30
_LIMIT_LOCK_MARGIN = 0.01


def _entry_ctes(
    spec: LabelSpec,
    entry: str,
    price_view: str,
    quality_view: str | None,
    *,
    min_end_gap_sessions: int,
    master_view: str | None,
    end_halt_sessions: int,
    end_halt_haircut: float | None,
    terminal_tail_max: int,
) -> list[str]:
    tag = entry
    pcol = "close_d" if tag == "c1" else "open_d"
    event_expr = "COALESCE(q.ca_event, FALSE)" if quality_view else "FALSE"
    q_join = (
        f"LEFT JOIN {quality_view} q "
        "ON q.trade_date = p.trade_date AND q.ticker = p.ticker AND q.market = p.market"
        if quality_view
        else ""
    )
    lim = (
        f"(CASE WHEN trade_date < DATE '{_LIMIT_CHANGE_DATE}' "
        f"THEN {_LIMIT_BEFORE} ELSE {_LIMIT_AFTER} END - {_LIMIT_LOCK_MARGIN})"
    )
    if master_view:
        master_ctes = [
            f"""
        master AS (
            SELECT ticker,
                   BOOL_OR(UPPER(status) = 'DELISTED') AS has_del,
                   BOOL_OR(UPPER(status) = 'ACTIVE') AS has_act
            FROM {master_view} GROUP BY ticker
        )"""
        ]
        master_join = "LEFT JOIN master m ON m.ticker = s.ticker"
        master_cond = " AND COALESCE(m.has_del AND NOT m.has_act, FALSE)"
    else:
        master_ctes, master_join, master_cond = [], "", ""
    ctes = [
        f"""
        cal0 AS (
            SELECT market, trade_date,
                   ROW_NUMBER() OVER (PARTITION BY market ORDER BY trade_date) AS s_idx
            FROM (SELECT DISTINCT market, trade_date FROM {price_view})
        )""",
        """
        cal AS (
            SELECT market, trade_date, s_idx,
                   LEAD(trade_date) OVER (PARTITION BY market ORDER BY s_idx) AS next_date,
                   LEAD(s_idx) OVER (PARTITION BY market ORDER BY s_idx) AS next_s,
                   MAX(s_idx) OVER (PARTITION BY market) AS mkt_last_s
            FROM cal0
        )""",
        f"""
        pxe0 AS (
            SELECT p.trade_date, p.ticker, p.market,
                   CAST(p.open AS DOUBLE) AS open_d, CAST(p.close AS DOUBLE) AS close_d,
                   (p.open = p.high AND p.high = p.low AND p.low = p.close) AS flat,
                   ROW_NUMBER() OVER (PARTITION BY p.ticker, p.market ORDER BY p.trade_date) AS d_idx,
                   LAG(CAST(p.close AS DOUBLE)) OVER (
                       PARTITION BY p.ticker, p.market ORDER BY p.trade_date
                   ) AS prev_close,
                   CASE WHEN {event_expr} THEN 1 ELSE 0 END AS ca_ev
            FROM {price_view} p {q_join}
            WHERE NOT (p.open = 0 AND p.high = 0 AND p.low = 0)
        )""",
        f"""
        pxe00 AS (
            SELECT trade_date, ticker, market, open_d, close_d, d_idx, ca_ev,
                   COALESCE(flat AND prev_close > 0
                            AND close_d / prev_close - 1 >= {lim}, FALSE) AS up_lock,
                   COALESCE(flat AND prev_close > 0
                            AND close_d / prev_close - 1 <= -{lim}, FALSE) AS dn_lock
            FROM pxe0
        )""",
        f"""
        ser0 AS (
            SELECT ticker, market, MAX(trade_date) AS last_date,
                   MAX(CASE WHEN open = 0 AND high = 0 AND low = 0 THEN trade_date END)
                       AS last_halt_date
            FROM {price_view} GROUP BY ticker, market
        )""",
        """
        nhs AS (
            SELECT ticker, market, MAX(trade_date) AS last_nh_date,
                   ARG_MAX(close_d, trade_date) AS last_close
            FROM pxe00 GROUP BY ticker, market
        )""",
        f"""
        ser1 AS (
            SELECT s.ticker, s.market, s.last_date, s.last_halt_date,
                   h.last_nh_date, h.last_close, COUNT(r.trade_date) AS n_trail
            FROM ser0 s
            JOIN nhs h ON h.ticker = s.ticker AND h.market = s.market
            LEFT JOIN {price_view} r
              ON r.ticker = s.ticker AND r.market = s.market AND r.trade_date > h.last_nh_date
            GROUP BY s.ticker, s.market, s.last_date, s.last_halt_date, h.last_nh_date, h.last_close
        )""",
        """
        ser2 AS (
            SELECT s.ticker, s.market, s.last_date, s.last_halt_date, s.last_nh_date,
                   s.last_close, s.n_trail, COUNT(p.trade_date) AS tail_n
            FROM ser1 s
            LEFT JOIN pxe00 p
              ON p.ticker = s.ticker AND p.market = s.market AND p.trade_date > s.last_halt_date
            GROUP BY s.ticker, s.market, s.last_date, s.last_halt_date, s.last_nh_date,
                     s.last_close, s.n_trail
        )""",
        *master_ctes,
        f"""
        ser AS (
            SELECT s.ticker, s.market, s.last_date, s.last_halt_date, s.last_nh_date,
                   s.last_close, s.n_trail, s.tail_n, cl.s_idx AS last_s,
                   (cl.mkt_last_s - cl.s_idx >= {min_end_gap_sessions}{master_cond}) AS ended
            FROM ser2 s
            JOIN cal cl ON cl.market = s.market AND cl.trade_date = s.last_date
            {master_join}
        )""",
        f"""
        pxe1 AS (
            SELECT p.trade_date, p.ticker, p.market, p.open_d, p.close_d, p.d_idx,
                   p.up_lock, p.dn_lock, c.s_idx,
                   (s.ended AND s.last_halt_date IS NOT NULL
                    AND s.tail_n BETWEEN 1 AND {terminal_tail_max}
                    AND p.trade_date > s.last_halt_date) AS in_tail,
                   p.ca_ev AS ca_ev_raw
            FROM pxe00 p
            JOIN ser s ON s.ticker = p.ticker AND s.market = p.market
            JOIN cal c ON c.market = p.market AND c.trade_date = p.trade_date
        )""",
        """
        pxe AS (
            SELECT trade_date, ticker, market, open_d, close_d, d_idx, up_lock, dn_lock,
                   s_idx, in_tail,
                   CASE WHEN in_tail THEN 0 ELSE ca_ev_raw END AS ca_ev,
                   SUM(CASE WHEN in_tail THEN 0 ELSE ca_ev_raw END) OVER (
                       PARTITION BY ticker, market ORDER BY d_idx
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS ca_cum,
                   MIN(CASE WHEN NOT dn_lock THEN d_idx END) OVER (
                       PARTITION BY ticker, market ORDER BY d_idx
                       ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                   ) AS nf
            FROM pxe1
        )""",
        f"""
        ent AS (
            SELECT a.trade_date, a.ticker, a.market, a.in_tail AS in_terminal_tail,
                   c.next_date AS e_date, c.next_s AS e_s, e.d_idx AS e_idx,
                   CASE WHEN e.trade_date IS NOT NULL AND NOT e.up_lock
                        THEN e.{pcol} END AS e_px,
                   e.ca_cum - e.ca_ev AS e_ca_before,
                   CASE WHEN c.next_date IS NULL THEN 'censored'
                        WHEN e.trade_date IS NULL OR e.up_lock OR e.{pcol} IS NULL
                        THEN 'untradable' END AS kind0
            FROM pxe a
            JOIN cal c ON c.market = a.market AND c.trade_date = a.trade_date
            LEFT JOIN pxe e
              ON e.ticker = a.ticker AND e.market = a.market AND e.trade_date = c.next_date
        )""",
    ]
    hc = "NULL" if end_halt_haircut is None else repr(float(end_halt_haircut))
    for h in spec.horizons:
        use_idx = spec.index_bench and tag == "c1"
        idx_cols = ", ie.idx_close AS ie_close, ix.idx_close AS ix_close" if use_idx else ""
        idx_join = (
            "\n            LEFT JOIN idx ie ON ie.market = n.market AND ie.trade_date = n.e_date"
            "\n            LEFT JOIN idx ix ON ix.market = n.market AND ix.trade_date = "
            "COALESCE(x.trade_date, CASE WHEN s.ended THEN s.last_date END)"
            if use_idx
            else ""
        )
        ctes.append(f"""
        fx{h} AS (
            SELECT n.trade_date, n.ticker, n.market, n.kind0, n.e_px, n.e_s,
                   (x.trade_date IS NOT NULL) AS has_exit,
                   x.{pcol} AS x_px, x.s_idx AS x_s, x.trade_date AS x_date,
                   COALESCE(s.ended, FALSE) AS ended, s.last_close, s.last_date, s.last_s,
                   COALESCE(s.n_trail, 0) AS n_trail,
                   (CASE WHEN x.trade_date IS NOT NULL THEN x.ca_cum ELSE l.ca_cum END)
                       - n.e_ca_before AS ca_n{idx_cols}
            FROM ent n
            LEFT JOIN pxe t0
              ON t0.ticker = n.ticker AND t0.market = n.market AND t0.d_idx = n.e_idx + {h}
            LEFT JOIN pxe x
              ON x.ticker = n.ticker AND x.market = n.market AND x.d_idx = t0.nf
            LEFT JOIN ser s ON s.ticker = n.ticker AND s.market = n.market
            LEFT JOIN pxe l
              ON l.ticker = s.ticker AND l.market = s.market AND l.trade_date = s.last_nh_date{idx_join}
        )""")
        ctes.append(f"""
        fk{h} AS (
            SELECT *,
                   CASE WHEN kind0 IS NOT NULL THEN kind0
                        WHEN has_exit THEN
                            CASE WHEN ca_n > 0 THEN 'ca_masked'
                                 WHEN x_s - e_s - {h} > 0 THEN 'late' ELSE 'regular' END
                        WHEN ended THEN
                            CASE WHEN ca_n > 0 THEN 'ca_masked'
                                 WHEN n_trail >= {end_halt_sessions} AND {hc} IS NULL
                                     THEN 'terminal_unpriced'
                                 WHEN n_trail >= {end_halt_sessions} THEN 'terminal_halted'
                                 ELSE 'terminal' END
                        ELSE 'censored' END AS kind,
                   CASE WHEN kind0 IS NULL
                        THEN COALESCE(x_date, CASE WHEN ended THEN last_date END) END AS end_date,
                   CASE WHEN kind0 IS NULL AND (has_exit OR ended)
                        THEN GREATEST(0, COALESCE(x_s, last_s) - e_s - {h}) END AS exit_lag
            FROM fx{h}
        )""")
        idx_sel = (
            f",\n                   CASE WHEN fwd IS NOT NULL "
            f"THEN ix_close / NULLIF(ie_close, 0) - 1 END AS bench_ret_idx_{tag}_{h}d"
            if use_idx
            else ""
        )
        ctes.append(f"""
        fc{h} AS (
            SELECT trade_date, ticker, market, fwd AS fwd_ret_{tag}_{h}d,
                   kind AS label_exit_kind_{tag}_{h}d,
                   end_date AS label_end_date_{tag}_{h}d,
                   exit_lag AS label_exit_lag_{tag}_{h}d{idx_sel}
            FROM (
                SELECT *,
                       CASE WHEN kind IN ('regular', 'late') THEN x_px / NULLIF(e_px, 0) - 1
                            WHEN kind = 'terminal' THEN last_close / NULLIF(e_px, 0) - 1
                            WHEN kind = 'terminal_halted'
                            THEN last_close * (1 - {hc}) / NULLIF(e_px, 0) - 1 END AS fwd
                FROM fk{h}
            )
        )""")
        if spec.kind == "excess":
            ctes.append(f"""
        ebench{h} AS (
            SELECT trade_date, market, AVG(fwd_ret_{tag}_{h}d) AS bench_ret_{tag}_{h}d
            FROM fc{h} GROUP BY trade_date, market
        )""")
    return ctes


def _entry_select_cols(spec: LabelSpec, entry: str) -> str:
    tag = entry
    lo, hi = spec.winsor
    win = "PARTITION BY base.trade_date, base.market"
    cols: list[str] = []
    for h in spec.horizons:
        cols.append(f"e{h}.fwd_ret_{tag}_{h}d")
        if spec.kind == "excess":
            cols.append(f"eb{h}.bench_ret_{tag}_{h}d")
            raw = f"(e{h}.fwd_ret_{tag}_{h}d - eb{h}.bench_ret_{tag}_{h}d)"
        else:
            raw = f"e{h}.fwd_ret_{tag}_{h}d"
        cols.append(f"{raw} AS raw_label_{tag}_{h}d")
        cols.append(f"e{h}.label_exit_kind_{tag}_{h}d")
        cols.append(f"e{h}.label_end_date_{tag}_{h}d")
        cols.append(f"e{h}.label_exit_lag_{tag}_{h}d")
        if spec.index_bench and tag == "c1":
            cols.append(f"e{h}.bench_ret_idx_{tag}_{h}d")
            cols.append(
                f"(e{h}.fwd_ret_{tag}_{h}d - e{h}.bench_ret_idx_{tag}_{h}d) "
                f"AS raw_label_idx_{tag}_{h}d"
            )
        g = f"{raw} IS NOT NULL"
        r = f"PERCENT_RANK() OVER ({win} ORDER BY {raw})"
        if "rank" in spec.outputs:
            cols.append(f"CASE WHEN {g} THEN {r} END AS y_rank_{tag}_{h}d")
        if "reg" in spec.outputs:
            cols.append(
                f"CASE WHEN {g} THEN LEAST(GREATEST({raw}, "
                f"QUANTILE_CONT({raw}, {lo}) OVER ({win})), "
                f"QUANTILE_CONT({raw}, {hi}) OVER ({win})) END AS y_reg_{tag}_{h}d"
            )
        if "cls" in spec.outputs:
            cols.append(
                f"CASE WHEN {g} THEN (CASE WHEN {r} >= {spec.cls_top} THEN 1 "
                f"WHEN {r} <= {spec.cls_bottom} THEN -1 ELSE 0 END) END AS y_cls_{tag}_{h}d"
            )
        if "up" in spec.outputs:
            cols.append(f"CASE WHEN {g} THEN CAST({raw} > 0 AS TINYINT) END AS y_up_{tag}_{h}d")
        if "top" in spec.outputs:
            cols.append(
                f"CASE WHEN {g} THEN CAST({r} >= {spec.cls_top} AS TINYINT) END AS y_top_{tag}_{h}d"
            )
    return ",\n            ".join(cols)


def build_entry_label_sql(
    spec: LabelSpec,
    entry: str,
    price_view: str = "daily_ohlcv",
    quality_view: str | None = None,
    index_view: str = "common_feature_observation_raw",
    *,
    min_end_gap_sessions: int = 1,
    master_view: str | None = None,
    end_halt_sessions: int = 20,
    end_halt_haircut: float | None = None,
    terminal_tail_max: int = 10,
) -> str:
    """c0 label (``build_label_sql(spec)``) plus the ``entry`` (c1/o1) family."""
    if entry not in _VALID_ENTRY:
        raise ValueError(f"entry must be one of {_VALID_ENTRY}, got {entry!r}")
    if min_end_gap_sessions < 1 or end_halt_sessions < 1 or terminal_tail_max < 1:
        raise ValueError("min_end_gap_sessions, end_halt_sessions, terminal_tail_max must be >= 1")
    if end_halt_haircut is not None and not 0.0 <= end_halt_haircut <= 1.0:
        raise ValueError(f"end_halt_haircut must be in [0, 1] or None, got {end_halt_haircut}")
    c0_sql = build_label_sql(spec, price_view, index_view)
    ctes = _entry_ctes(
        spec,
        entry,
        price_view,
        quality_view,
        min_end_gap_sessions=min_end_gap_sessions,
        master_view=master_view,
        end_halt_sessions=end_halt_sessions,
        end_halt_haircut=end_halt_haircut,
        terminal_tail_max=terminal_tail_max,
    )
    head = [f"c0 AS ({c0_sql})"]
    if spec.index_bench and entry == "c1":
        head.append(_index_cte(index_view))
    # Row set: every non-halt (t, ticker); c0 columns LEFT-joined so rows the c0
    # inner join drops (no exit) survive.
    base = """
        base AS (
            SELECT n.trade_date, n.ticker, n.market, n.in_terminal_tail FROM ent n
        )"""
    all_ctes = ",\n".join(head + ctes + [base])
    joins = "".join(
        f"\n        LEFT JOIN fc{h} AS e{h} USING (trade_date, ticker, market)"
        + (
            f"\n        LEFT JOIN ebench{h} AS eb{h} USING (trade_date, market)"
            if spec.kind == "excess"
            else ""
        )
        for h in spec.horizons
    )
    return f"""
        WITH {all_ctes}
        SELECT
            base.trade_date, base.ticker, base.market,
            c0.* EXCLUDE (trade_date, ticker, market),
            base.in_terminal_tail,
            {_entry_select_cols(spec, entry)}
        FROM base
        LEFT JOIN c0 USING (trade_date, ticker, market){joins}
    """


def materialize_entry_label(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    spec: LabelSpec,
    entry: str,
    *,
    price_view: str = "daily_ohlcv",
    quality_view: str | None = None,
    table: str = LABEL_ENTRY_TABLE,
    force: bool = False,
    **terminal_kwargs: object,
) -> str:
    """Build + register the entry-label mart view (default ``label_entry_v2``).

    ``terminal_kwargs``: ``min_end_gap_sessions``, ``master_view``,
    ``end_halt_sessions``, ``end_halt_haircut``, ``terminal_tail_max``.
    """
    sql = build_entry_label_sql(spec, entry, price_view, quality_view, **terminal_kwargs)  # type: ignore[arg-type]
    materialize(con, config, table, sql, force=force)
    return register_mart_view(con, config, table)
