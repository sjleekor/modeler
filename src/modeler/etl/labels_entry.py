# ruff: noqa: E501
"""Entry-timing labels (c1 / o1) beside the legacy c0 label — quality-fix plan §3.

``etl/labels.py`` is part of the model02 holdout-opening code hash and must not
change, so the entry logic lives here and wraps :func:`labels.build_label_sql`.

``build_entry_label_sql(spec, entry)`` returns the c0 label (exactly what
``build_label_sql(spec)`` makes) with a second column family added beside it:

  - Enter at the close (``c1``) / open (``o1``) of the next *market* trading day
    t+1 (calendar = distinct dates of the price view per market; a halt does NOT
    push the entry back) and exit h non-halt sessions of the ticker later,
    counted from the entry day.
  - Columns, ``tag`` in {c1, o1}: ``fwd_ret_{tag}_{h}d``, ``bench_ret_{tag}_{h}d``
    (eqw market over tradeable rows), ``raw_label_{tag}_{h}d``,
    ``label_missing_exit_{tag}_{h}d`` and ``y_{rank,reg,cls,up,top}_{tag}_{h}d``
    for ``spec.outputs``. With ``spec.index_bench`` and c1 also
    ``bench_ret_idx_c1_{h}d`` / ``raw_label_idx_c1_{h}d`` (index return over the
    same entry/exit dates). No index open series exists, so o1 has no index
    columns. The legacy ``y_*`` columns still come from c0.
  - NULL (untradeable): t+1 halted or missing; t+1 locked limit-up
    (open=high=low=close and ret >= limit-1%p); exit locked limit-down; a
    corporate-action event in [t+1, t+1+h] (needs ``quality_view`` with
    ``ca_event``; without it no mask). Limit 15% before 2015-06-15, 30% after.
  - ``label_missing_exit`` = entry tradeable but no exit session (delisting /
    panel end). The row set is every non-halt (t, ticker) row, so the c0 columns
    are NULL on rows the c0 inner join would drop. The dataset therefore differs
    from ``label_daily``: materialize under its own name (``label_entry_v2``).
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
    spec: LabelSpec, entry: str, price_view: str, quality_view: str | None
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
    ctes = [
        f"""
        cal AS (
            SELECT market, trade_date,
                   LEAD(trade_date) OVER (PARTITION BY market ORDER BY trade_date) AS next_date
            FROM (SELECT DISTINCT market, trade_date FROM {price_view})
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
        pxe AS (
            SELECT trade_date, ticker, market, open_d, close_d, d_idx,
                   SUM(ca_ev) OVER (
                       PARTITION BY ticker, market ORDER BY trade_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS ca_cum,
                   ca_ev,
                   COALESCE(flat AND prev_close > 0
                            AND close_d / prev_close - 1 >= {lim}, FALSE) AS up_lock,
                   COALESCE(flat AND prev_close > 0
                            AND close_d / prev_close - 1 <= -{lim}, FALSE) AS dn_lock
            FROM pxe0
        )""",
        f"""
        ent AS (
            SELECT a.trade_date, a.ticker, a.market,
                   e.trade_date AS e_date, e.d_idx AS e_idx,
                   CASE WHEN e.trade_date IS NOT NULL AND NOT e.up_lock
                        THEN e.{pcol} END AS e_px,
                   e.ca_cum - e.ca_ev AS e_ca_before
            FROM pxe a
            JOIN cal c ON c.market = a.market AND c.trade_date = a.trade_date
            LEFT JOIN pxe e
              ON e.ticker = a.ticker AND e.market = a.market AND e.trade_date = c.next_date
        )""",
    ]
    for h in spec.horizons:
        ok = (
            "n.e_px IS NOT NULL AND x.trade_date IS NOT NULL AND NOT x.dn_lock "
            "AND x.ca_cum - n.e_ca_before = 0"
        )
        idx_col = idx_join = ""
        if spec.index_bench and tag == "c1":
            idx_col = (
                f",\n                   CASE WHEN {ok} THEN ix.idx_close / NULLIF(ie.idx_close, 0) - 1 "
                f"END AS bench_ret_idx_{tag}_{h}d"
            )
            idx_join = (
                "\n            LEFT JOIN idx ie ON ie.market = n.market AND ie.trade_date = n.e_date"
                "\n            LEFT JOIN idx ix ON ix.market = n.market AND ix.trade_date = x.trade_date"
            )
        ctes.append(f"""
        fc{h} AS (
            SELECT n.trade_date, n.ticker, n.market,
                   CASE WHEN {ok} THEN x.{pcol} / NULLIF(n.e_px, 0) - 1 END AS fwd_ret_{tag}_{h}d,
                   (n.e_px IS NOT NULL AND x.trade_date IS NULL) AS label_missing_exit_{tag}_{h}d{idx_col}
            FROM ent n
            LEFT JOIN pxe x
              ON x.ticker = n.ticker AND x.market = n.market AND x.d_idx = n.e_idx + {h}{idx_join}
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
        cols.append(f"e{h}.label_missing_exit_{tag}_{h}d")
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
) -> str:
    """c0 label (``build_label_sql(spec)``) plus the ``entry`` (c1/o1) family."""
    if entry not in _VALID_ENTRY:
        raise ValueError(f"entry must be one of {_VALID_ENTRY}, got {entry!r}")
    c0_sql = build_label_sql(spec, price_view, index_view)
    ctes = _entry_ctes(spec, entry, price_view, quality_view)
    head = [f"c0 AS ({c0_sql})"]
    if spec.index_bench and entry == "c1":
        head.append(_index_cte(index_view))
    # Row set: every non-halt (t, ticker); c0 columns LEFT-joined so rows the c0
    # inner join drops (no exit) survive for the missing-exit diagnostic.
    base = """
        base AS (
            SELECT n.trade_date, n.ticker, n.market FROM ent n
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
) -> str:
    """Build + register the entry-label mart view (default ``label_entry_v2``)."""
    sql = build_entry_label_sql(spec, entry, price_view, quality_view)
    materialize(con, config, table, sql, force=force)
    return register_mart_view(con, config, table)
