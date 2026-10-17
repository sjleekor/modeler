"""Vintage marts that keep delisted companies (``_dl`` variants).

``stock_metric_vintage_fact`` (``metric_vintages.py``) joins every raw DART table to
``corp`` -- ``dart_corp_master`` rows with ``market IS NOT NULL``. ``market`` of a corp is
filled only when its ticker matches the *current* ``stock_master``
(``collector/.../sync_dart_corp.py``), so a company that has since been delisted has
``market = NULL`` and drops out of the mart, and with it out of
``fin_quarterly_metric_vintage`` and ``feat_fin_v2`` (15_quality_fix_plan.md §10.2, E1).
That makes the financial features a future-delisting marker: ``fin_roe_v2`` is non-null on
8.9% of delisted rows against 61.2% of surviving ones.

The variant here changes **only the corp relation**. ``market`` becomes

1. ``dart_corp_master.market`` when present (so an active corp's rows are unchanged),
2. else the ``stock_master`` row of that ticker that was seen last, whatever its status
   (``DELISTED`` included; a ticker that moved KOSDAQ -> KOSPI has two rows and the later
   one wins),
3. else the market of the ticker's last ``daily_ohlcv`` row.

A ticker for which none of the three gives a market, or that is not in
``dart_corp_master`` at all (OpenDART never returned a corp code for it), stays out: its
``market`` stays NULL and the unchanged ``market IS NOT NULL`` predicate drops it.

Nothing in the original modules is touched. The corp relation reaches the original
builder through its own ``corp_view`` argument (a parenthesised sub-select works as a
``FROM`` item), so the CTE text, the staged plan and the metric rules are the original's.
Because the original is a hash contract, the swap is **guarded**: the original ``corp`` CTE
must still read exactly the text assumed in :data:`_EXPECTED_CORP_CTE`, with the relation
appearing once, or :class:`CorpCteDriftError` is raised instead of building a mart that
silently differs. (The same applies if the staged plan stops carrying the relation once.)

The marts are new -- ``stock_metric_vintage_fact_dl`` and
``fin_quarterly_metric_vintage_dl`` -- so no frozen mart or cache key is reused or
overwritten. ``feat_fin_v2`` takes ``vintage_view="fin_quarterly_metric_vintage_dl"``; the
join there is on ``ticker`` only, so it needs no other change.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view
from modeler.etl.marts import financial_quarters as _fq
from modeler.etl.marts import metric_vintages as _mv

SMVF_DL_TABLE = "stock_metric_vintage_fact_dl"
FQMV_DL_TABLE = "fin_quarterly_metric_vintage_dl"

SEMANTICS_V1 = _mv.SEMANTICS_V1
SEMANTICS_V2 = _mv.SEMANTICS_V2
PLAN_SINGLE = _mv.PLAN_SINGLE
PLAN_STAGED = _mv.PLAN_STAGED

#: Recorded in the cache metadata (and so in the plan hash) of both ``_dl`` marts.
_VARIANT_METADATA: tuple[tuple[str, object], ...] = (
    ("variant", "keep_delisted_corps"),
    ("corp_market_sources", "dart_corp_master,stock_master,daily_ohlcv"),
)

#: The ``corp`` CTE of ``metric_vintages._vintage_sql_parts``, with ``{corp_view}`` where the
#: relation goes. It is kept as a literal on purpose: this module is only valid for that text.
_EXPECTED_CORP_CTE = """corp AS (
        -- §3.4: no is_active filter — active-status backcast is a survivor bias.
        SELECT ticker, market, corp_code
        FROM {corp_view}
        WHERE ticker IS NOT NULL AND ticker <> ''
          AND market IS NOT NULL
    )"""

_VIEW_KEYS_OWN = ("corp_view", "stock_master_view", "ohlcv_view")


class OriginalSqlDriftError(RuntimeError):
    """An original builder's text is not the one this variant was written for."""


class CorpCteDriftError(OriginalSqlDriftError):
    """The original ``corp`` CTE (or the staged plan's use of it) changed."""


def corp_relation_sql(
    *,
    corp_view: str = "dart_corp_master",
    stock_master_view: str = "stock_master",
    ohlcv_view: str = "daily_ohlcv",
) -> str:
    """A ``(SELECT ...) AS corp_market_src`` relation of ``(ticker, market, corp_code)``.

    One output row per ``dart_corp_master`` row (the two lookups return at most one row per
    ticker, so they cannot multiply it). ``market`` is ``COALESCE`` of the corp's own market,
    the last-seen ``stock_master`` market and the last ``daily_ohlcv`` market; blank strings
    from the two lookups count as missing, the corp's own value is passed through untouched
    so that a corp that was in the original mart keeps exactly its original row.
    """
    return f"""(
        SELECT
            c.ticker,
            COALESCE(c.market, sm.market, om.market) AS market,
            c.corp_code
        FROM {corp_view} c
        LEFT JOIN (
            SELECT ticker, market
            FROM {stock_master_view}
            WHERE ticker IS NOT NULL AND NULLIF(market, '') IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ticker
                ORDER BY last_seen_date DESC NULLS LAST,
                         (status = 'ACTIVE') DESC NULLS LAST,
                         updated_at DESC NULLS LAST,
                         market ASC
            ) = 1
        ) sm ON sm.ticker = c.ticker
        LEFT JOIN (
            SELECT ticker, arg_max(market, trade_date) AS market
            FROM {ohlcv_view}
            WHERE ticker IS NOT NULL AND NULLIF(market, '') IS NOT NULL
            GROUP BY ticker
        ) om ON om.ticker = c.ticker
    ) AS corp_market_src"""


def _split_views(views: dict[str, object]) -> tuple[str, dict[str, object]]:
    """The corp relation for ``views`` and the view arguments left for the original builder."""
    own = {k: views.pop(k) for k in _VIEW_KEYS_OWN if k in views}
    relation = corp_relation_sql(**own)  # type: ignore[arg-type]
    return relation, views


def _assert_corp_cte(texts: Sequence[str], relation: str, *, what: str) -> None:
    """Raise unless the corp CTE with ``relation`` is in exactly one of ``texts``, once."""
    expected = _EXPECTED_CORP_CTE.replace("{corp_view}", relation)
    cte_hits = sum(t.count(expected) for t in texts)
    rel_hits = sum(t.count(relation) for t in texts)
    if cte_hits != 1 or rel_hits != 1:
        raise CorpCteDriftError(
            f"{what}: expected the original corp CTE exactly once with the delisted-corp "
            f"relation substituted, found the CTE {cte_hits}x and the relation {rel_hits}x; "
            "metric_vintages._vintage_sql_parts changed its corp CTE, so this variant must be "
            "rewritten before it can be built"
        )


def build_stock_metric_vintage_fact_dl_sql(
    *, semantics: str = SEMANTICS_V1, **views: object
) -> str:
    """The original ``stock_metric_vintage_fact`` statement over the delisted-keeping corp.

    ``views`` are the original builder's (``financial_view``, ``calendar_table``,
    ``pairing_tolerance`` ...) plus ``corp_view``, ``stock_master_view`` and ``ohlcv_view``
    for the corp relation. Raises :class:`CorpCteDriftError` if the original corp CTE text
    changed.
    """
    relation, rest = _split_views(dict(views))
    sql = _mv.build_stock_metric_vintage_fact_sql(
        corp_view=relation,
        semantics=semantics,
        **rest,  # type: ignore[arg-type]
    )
    _assert_corp_cte([sql], relation, what="stock_metric_vintage_fact_dl contract SQL")
    return sql


def build_stock_metric_vintage_fact_dl_plan(
    *, semantics: str = SEMANTICS_V1, plan: str = PLAN_SINGLE, **views: object
) -> MartPlan:
    """The :class:`MartPlan` of the ``_dl`` fact; unlike the original it is never ``None``.

    The plan always carries the variant metadata, so a cache hit needs the same variant.
    ``plan="staged"`` is the original six-step plan with the corp relation in the winners
    stage.
    """
    relation, rest = _split_views(dict(views))
    contract_sql = build_stock_metric_vintage_fact_dl_sql(
        semantics=semantics,
        **views,  # type: ignore[arg-type]
    )
    base = _mv.build_stock_metric_vintage_fact_plan(
        semantics=semantics,
        plan=plan,
        corp_view=relation,
        **rest,  # type: ignore[arg-type]
    )
    if base is None:
        base = MartPlan(plan_id=PLAN_SINGLE, final_sql=contract_sql, semantics_version=semantics)
    if base.stages:
        _assert_corp_cte(
            [sql for _, sql in base.stages], relation, what="stock_metric_vintage_fact_dl stages"
        )
        if relation in base.final_sql:
            raise CorpCteDriftError(
                "stock_metric_vintage_fact_dl: the final statement of the staged plan reads the "
                "corp relation; the original staged plan changed"
            )
    return dataclasses.replace(base, metadata=_VARIANT_METADATA)


def register_stock_metric_vintage_fact_dl_view(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: Sequence[date],
    view_name: str = SMVF_DL_TABLE,
    semantics: str = SEMANTICS_V1,
    **views: str,
) -> str:
    """Calendar table plus a view over the ``_dl`` SQL (no parquet; tests and parity checks)."""
    _mv._register_calendar(con, trading_days)
    sql = build_stock_metric_vintage_fact_dl_sql(semantics=semantics, **views)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_stock_metric_vintage_fact_dl(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    trading_days: Sequence[date],
    force: bool = False,
    semantics: str = SEMANTICS_V1,
    plan: str = PLAN_SINGLE,
    stage_dir: Path | None = None,
    **views: str,
) -> str:
    """Build and register ``stock_metric_vintage_fact_dl`` as a cached parquet mart.

    Needs the raw DART views of the original plus ``stock_master`` and ``daily_ohlcv``
    registered on ``con``. Arguments as in
    :func:`~modeler.etl.marts.metric_vintages.materialize_stock_metric_vintage_fact`.
    """
    _mv._register_calendar(con, trading_days)
    materialize(
        con,
        config,
        SMVF_DL_TABLE,
        build_stock_metric_vintage_fact_dl_sql(semantics=semantics, **views),
        force=force,
        plan=build_stock_metric_vintage_fact_dl_plan(semantics=semantics, plan=plan, **views),
        stage_dir=stage_dir,
    )
    return register_mart_view(con, config, SMVF_DL_TABLE)


def build_fin_quarterly_metric_vintage_dl_sql(
    *,
    vintage_view: str = SMVF_DL_TABLE,
    pairing_tolerance: float = 0.0,
    semantics: str = SEMANTICS_V1,
) -> str:
    """``fin_quarterly_metric_vintage`` reading ``vintage_view`` (default: the ``_dl`` fact).

    The original builder already takes the vintage relation by name, so this is its text
    unchanged; the guard only checks that the relation is read once and by that name.
    """
    sql = _fq.build_fin_quarterly_metric_vintage_sql(
        vintage_view=vintage_view, pairing_tolerance=pairing_tolerance, semantics=semantics
    )
    if sql.count(f"FROM {vintage_view}\n") != 1:
        raise OriginalSqlDriftError(
            f"fin_quarterly_metric_vintage_dl: expected 'FROM {vintage_view}' exactly once in "
            "the original builder's text; financial_quarters changed how it reads the fact"
        )
    return sql


def register_fin_quarterly_metric_vintage_dl_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FQMV_DL_TABLE,
    vintage_view: str = SMVF_DL_TABLE,
    pairing_tolerance: float = 0.0,
    semantics: str = SEMANTICS_V1,
) -> str:
    """A view over the ``_dl`` quarterly SQL (no parquet; tests and parity checks)."""
    sql = build_fin_quarterly_metric_vintage_dl_sql(
        vintage_view=vintage_view, pairing_tolerance=pairing_tolerance, semantics=semantics
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_fin_quarterly_metric_vintage_dl(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str = SMVF_DL_TABLE,
    pairing_tolerance: float = 0.0,
    force: bool = False,
    semantics: str = SEMANTICS_V1,
) -> str:
    """Build and register ``fin_quarterly_metric_vintage_dl`` as a cached parquet mart.

    Needs ``vintage_view`` (the ``_dl`` fact by default) registered on ``con``. The cache
    metadata records the variant and the upstream relation name.
    """
    sql = build_fin_quarterly_metric_vintage_dl_sql(
        vintage_view=vintage_view, pairing_tolerance=pairing_tolerance, semantics=semantics
    )
    plan = MartPlan(
        plan_id=PLAN_SINGLE,
        final_sql=sql,
        semantics_version=semantics,
        metadata=(*_VARIANT_METADATA, ("vintage_view", vintage_view)),
    )
    materialize(con, config, FQMV_DL_TABLE, sql, force=force, plan=plan)
    return register_mart_view(con, config, FQMV_DL_TABLE)
