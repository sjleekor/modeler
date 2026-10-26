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
      * End-of-data guard: a terminal kind (``terminal``, ``terminal_halted``,
        ``terminal_unpriced``) needs the SCHEDULED exit session to exist in the
        market calendar (entry session + h <= the market's last session), i.e.
        the window was observable for survivors too. Otherwise ``censored``
        (NULL); without it the last h+1 sessions would hold only delisted names.
        ``late`` exits (the actual exit was observed) are not guarded.
  - Corporate-action mask: any ``ca_event`` in the window (needs ``quality_view``;
    without it no mask) -> NULL, ``ca_masked``. The window is (t+1, exit] for c1
    (an event dated t+1 is a jump between the t and t+1 closes and does not touch
    a t+1-close entry) and [t+1, exit] for o1 (the t+1 open is after the
    overnight event). Rows of a *terminal tail* are
    ignored: the non-halt rows after the last halt row of an ENDED series, when
    there are 1..``terminal_tail_max`` of them (정리매매 crashes look like events).
    Interpretation: a series with no halt row at all has no tail.

Columns (``tag`` in {c1, o1}; per horizon h)
  ``fwd_ret_{tag}_{h}d``, ``bench_ret_{tag}_{h}d`` (eqw market, terminal included;
  NULL on a (date, market) with fewer than ``min_cross_section`` valid ``fwd``,
  which makes ``raw_label`` and ``y_*`` NULL there while ``fwd`` stays),
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

Fixed benchmark (``fixed_benchmarks``, default none): for each ``(tag, idx_nm)``
(or ``(tag, idx_clss, idx_nm)`` if the name alone is ambiguous) a row of
``fixed_index_view`` (``krx_index_daily``: bas_dd, idx_clss, idx_nm, close_idx,
open_idx) gives one index series for ALL stocks, whatever their market: the
investor's fixed alternative, and the main judgment yardstick G of plan
``16_c1_label_benchmark.md`` §6. Columns per horizon:
``bench_ret_{tag}_{entry}_{h}d`` = index return over the stock's own entry date
and actual exit date (c1: close to close; o1: ``open_idx`` on the entry date to
``open_idx`` on the exit date; a terminal exit uses the series' last row date and
the index CLOSE, like the stock side), and ``raw_label_{tag}_{entry}_{h}d`` =
``fwd - bench``. NULL when the stock label is NULL or the index has no value on
either exact date (no forward fill). The market-specific ``index_bench`` columns
and the eqw benchmark stay as diagnostics. KOSPI 200 in the lake is
``("k200", "코스피 200")`` (idx_group kospi, idx_clss KOSPI; one row per date,
unique by name alone, 2026-10-18).

Parameters: ``min_end_gap_sessions`` (1), ``master_view`` (None; ticker, market,
status), ``end_halt_sessions`` (20), ``end_halt_haircut`` (None),
``terminal_tail_max`` (10), ``min_cross_section`` (30), ``fixed_benchmarks`` (()),
``fixed_index_view`` (``krx_index_daily``).

Leakage note: terminal values use knowledge of the delisting, which is allowed
for labels but never for features. A label can end hundreds of sessions after t
(long halts), so training and validation splits must purge by
``label_end_date_*``, not by h.

The row set is every non-halt (t, ticker) row, so the c0 columns are NULL on
rows the c0 inner join would drop. The dataset therefore differs from
``label_daily``: materialize it under its own name (``label_entry_v2``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.labels import LabelSpec, _index_cte, build_label_sql
from modeler.etl.mart import materialize, register_mart_view

LABEL_ENTRY_TABLE = "label_entry_v2"
_VALID_ENTRY = ("c1", "o1")
_TAG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

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
    min_cross_section: int,
    fixed: Sequence[tuple[str, str | None, str]] = (),
    fixed_index_view: str = "krx_index_daily",
) -> list[str]:
    tag = entry
    pcol = "close_d" if tag == "c1" else "open_d"
    # c1 enters at the t+1 close: an event dated t+1 is already behind the entry.
    # o1 enters at the t+1 open: an event dated t+1 happened overnight, before it.
    ca_before = "e.ca_cum" if tag == "c1" else "e.ca_cum - e.ca_ev"
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
                   {ca_before} AS e_ca_before, c.mkt_last_s,
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
    fpc = "close_i" if tag == "c1" else "open_i"
    for ftag, fclss, fnm in fixed:
        where = f"idx_nm = '{_sql_str(fnm)}'"
        if fclss is not None:
            where += f" AND idx_clss = '{_sql_str(fclss)}'"
        ctes.append(f"""
        fb_{ftag} AS (
            SELECT bas_dd AS trade_date, CAST(close_idx AS DOUBLE) AS close_i,
                   CAST(open_idx AS DOUBLE) AS open_i
            FROM {fixed_index_view} WHERE {where}
        )""")
    for h in spec.horizons:
        use_idx = spec.index_bench and tag == "c1"
        idx_cols = ", ie.idx_close AS ie_close, ix.idx_close AS ix_close" if use_idx else ""
        exit_date = "COALESCE(x.trade_date, CASE WHEN s.ended THEN s.last_date END)"
        for ftag, _, _ in fixed:
            idx_cols += (
                f", fe_{ftag}.{fpc} AS fb_{ftag}_e"
                f", CASE WHEN x.trade_date IS NOT NULL THEN fx_{ftag}.{fpc} "
                f"ELSE fx_{ftag}.close_i END AS fb_{ftag}_x"
            )
        idx_join = (
            "\n            LEFT JOIN idx ie ON ie.market = n.market AND ie.trade_date = n.e_date"
            "\n            LEFT JOIN idx ix ON ix.market = n.market AND ix.trade_date = "
            "COALESCE(x.trade_date, CASE WHEN s.ended THEN s.last_date END)"
            if use_idx
            else ""
        )
        for ftag, _, _ in fixed:
            idx_join += (
                f"\n            LEFT JOIN fb_{ftag} fe_{ftag} ON fe_{ftag}.trade_date = n.e_date"
                f"\n            LEFT JOIN fb_{ftag} fx_{ftag} ON fx_{ftag}.trade_date = {exit_date}"
            )
        ctes.append(f"""
        fx{h} AS (
            SELECT n.trade_date, n.ticker, n.market, n.kind0, n.e_px, n.e_s, n.mkt_last_s,
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
                   CASE WHEN kind IN ('censored', 'untradable') THEN NULL
                        ELSE COALESCE(x_date, last_date) END AS end_date,
                   CASE WHEN kind IN ('censored', 'untradable') THEN NULL
                        ELSE GREATEST(0, COALESCE(x_s, last_s) - e_s - {h}) END AS exit_lag
            FROM (
                SELECT *,
                       CASE WHEN kind0 IS NOT NULL THEN kind0
                            WHEN has_exit THEN
                                CASE WHEN ca_n > 0 THEN 'ca_masked'
                                     WHEN x_s - e_s - {h} > 0 THEN 'late' ELSE 'regular' END
                            WHEN ended THEN
                                CASE WHEN e_s + {h} > mkt_last_s THEN 'censored'
                                     WHEN ca_n > 0 THEN 'ca_masked'
                                     WHEN n_trail >= {end_halt_sessions} AND {hc} IS NULL
                                         THEN 'terminal_unpriced'
                                     WHEN n_trail >= {end_halt_sessions} THEN 'terminal_halted'
                                     ELSE 'terminal' END
                            ELSE 'censored' END AS kind
                FROM fx{h}
            )
        )""")
        idx_sel = (
            f",\n                   CASE WHEN fwd IS NOT NULL "
            f"THEN ix_close / NULLIF(ie_close, 0) - 1 END AS bench_ret_idx_{tag}_{h}d"
            if use_idx
            else ""
        )
        for ftag, _, _ in fixed:
            idx_sel += (
                f",\n                   CASE WHEN fwd IS NOT NULL "
                f"THEN fb_{ftag}_x / NULLIF(fb_{ftag}_e, 0) - 1 END "
                f"AS bench_ret_{ftag}_{tag}_{h}d"
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
            SELECT trade_date, market,
                   CASE WHEN COUNT(fwd_ret_{tag}_{h}d) >= {min_cross_section}
                        THEN AVG(fwd_ret_{tag}_{h}d) END AS bench_ret_{tag}_{h}d
            FROM fc{h} GROUP BY trade_date, market
        )""")
    return ctes


def _sql_str(v: str) -> str:
    return v.replace("'", "''")


def _entry_select_cols(
    spec: LabelSpec, entry: str, fixed: Sequence[tuple[str, str | None, str]] = ()
) -> str:
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
        for ftag, _, _ in fixed:
            cols.append(f"e{h}.bench_ret_{ftag}_{tag}_{h}d")
            cols.append(
                f"(e{h}.fwd_ret_{tag}_{h}d - e{h}.bench_ret_{ftag}_{tag}_{h}d) "
                f"AS raw_label_{ftag}_{tag}_{h}d"
            )
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


def _normalize_fixed(
    fixed_benchmarks: Sequence[tuple[str, ...]],
) -> list[tuple[str, str | None, str]]:
    """``(tag, idx_nm)`` or ``(tag, idx_clss, idx_nm)`` -> ``(tag, idx_clss|None, idx_nm)``."""
    out: list[tuple[str, str | None, str]] = []
    for item in fixed_benchmarks:
        if len(item) == 2:
            tag, nm = item
            clss = None
        elif len(item) == 3:
            tag, clss, nm = item
        else:
            raise ValueError(
                f"fixed benchmark must be (tag, idx_nm) or (tag, idx_clss, idx_nm): {item!r}"
            )
        if not _TAG_RE.match(tag):
            raise ValueError(f"fixed benchmark tag must match {_TAG_RE.pattern}, got {tag!r}")
        if tag in {t for t, _, _ in out}:
            raise ValueError(f"duplicate fixed benchmark tag {tag!r}")
        out.append((tag, clss, nm))
    return out


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
    min_cross_section: int = 30,
    fixed_benchmarks: Sequence[tuple[str, ...]] = (),
    fixed_index_view: str = "krx_index_daily",
) -> str:
    """c0 label (``build_label_sql(spec)``) plus the ``entry`` (c1/o1) family."""
    if entry not in _VALID_ENTRY:
        raise ValueError(f"entry must be one of {_VALID_ENTRY}, got {entry!r}")
    if (
        min_end_gap_sessions < 1
        or end_halt_sessions < 1
        or terminal_tail_max < 1
        or min_cross_section < 1
    ):
        raise ValueError(
            "min_end_gap_sessions, end_halt_sessions, terminal_tail_max, "
            "min_cross_section must be >= 1"
        )
    if end_halt_haircut is not None and not 0.0 <= end_halt_haircut <= 1.0:
        raise ValueError(f"end_halt_haircut must be in [0, 1] or None, got {end_halt_haircut}")
    fixed = _normalize_fixed(fixed_benchmarks)
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
        min_cross_section=min_cross_section,
        fixed=fixed,
        fixed_index_view=fixed_index_view,
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
            {_entry_select_cols(spec, entry, fixed)}
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
    ``end_halt_sessions``, ``end_halt_haircut``, ``terminal_tail_max``,
    ``min_cross_section``, ``fixed_benchmarks``, ``fixed_index_view``.
    """
    sql = build_entry_label_sql(spec, entry, price_view, quality_view, **terminal_kwargs)  # type: ignore[arg-type]
    materialize(con, config, table, sql, force=force)
    return register_mart_view(con, config, table)
