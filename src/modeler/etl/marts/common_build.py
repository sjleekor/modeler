"""DuckDB port of ``common build-daily`` -> ``common_feature_daily_fact`` (§3.2).

Recomputes the KRX-session-aligned common feature facts from the raw observation
lake (``common_feature_observation_raw``) instead of the Postgres
``service/build_common_feature_daily_facts.py`` row-by-row path. Catalog and
catalog-inputs come from the pure code definitions in
``collector.kr.definitions.common_features``; the series config (for
``max_stale_business_days`` / ``default_transform``) is read from the lake view
``common_feature_series`` (the one config the collector shares — decision 7).

This is the highest-risk port (plan review "High"): the Python build is *not* a
simple as-of + transform. The pieces reproduced 1:1:

- **period-latest-vintage as-of** (``_asof_history``): among observations with
  ``available_from_date <= feature_date``, pick per *period* (period_end_date,
  else observation_date) the max ``(release_date, available_from_date, fetched_at,
  vintage, raw_id)`` — a period-ordered history, not a single latest row. The
  ``current`` observation is the one with the largest period.
- **stale gate** (``_is_stale``): if ``current.available_from_date`` is more than
  ``max_stale_business_days`` KRX sessions before ``feature_date`` -> value NULL
  (provenance still recorded). Business-day age uses ``bisect_right`` over the
  stale calendar; reproduced with a 1-based dense session index.
- **positional lag** for ret/change/vol (history position -N) vs **calendar
  offset** for yoy/mom (exact 12/1 months prior period).
- **vol_Nd**: sample stddev (ddof=1) of the last N one-step returns.
- **multi-input** spread = long-short, ratio = num/den (den 0 -> NULL), per role.

Numeric path stays in DECIMAL to match Python ``Decimal`` exactly for
level/ret/change/yoy/mom/spread/ratio. ``vol`` ends in a square root: Python uses
``Decimal.sqrt`` while DuckDB ``sqrt``/``stddev_samp`` go through DOUBLE, so vol
parity is to a small relative tolerance (plan §7.4); every other transform exact.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

import duckdb
from collector.kr.shared import (
    CommonFeatureCatalogEntry,
    CommonFeatureSeries,
    default_common_feature_catalog,
    default_common_feature_series,
)

from modeler.etl.lake import _sql_str_literal

CFDF_VIEW = "common_feature_daily_fact"
_CAL_TABLE = "cal_rel"
_FEATURE_DATES_TABLE = "feature_dates_rel"
_BUILD_TABLE = "cfdf_build"

_SCALAR_TRANSFORMS = {"level", "yoy", "mom"}
_WINDOWED_TRANSFORM = re.compile(r"^(ret|change|vol)_(\d+)d$")
_MULTI_INPUT_TRANSFORMS = {
    "spread": ("spread_long", "spread_short"),
    "ratio": ("numerator", "denominator"),
}


def _resolve_transform(feature: CommonFeatureCatalogEntry, default_transform: str) -> str:
    return feature.transform_code or default_transform or "level"


def _parse_windowed(transform_code: str) -> tuple[str, int] | None:
    match = _WINDOWED_TRANSFORM.match(transform_code)
    if match is None:
        return None
    window = int(match.group(2))
    if window <= 0:
        return None
    return match.group(1), window


def _bisect_right_expr(date_expr: str) -> str:
    """SQL for ``bisect_right(stale_calendar, <date>)`` via the session index.

    Count of sessions ``<= date`` = max ``idx`` whose ``d <= date`` (0 below all).
    """
    return f"(SELECT COALESCE(MAX(c.idx), 0) FROM {_CAL_TABLE} c WHERE c.d <= {date_expr})"


# DuckDB's ``/`` over DECIMAL yields DOUBLE; cast back to a high-scale DECIMAL so
# ratio math stays exact vs Python ``Decimal`` (plan §7.4 option a). 12 places is
# well beyond any meaningful precision in these features.
_DIV_DECIMAL = "DECIMAL(38,12)"


def _decimal_div(numerator: str, denominator: str) -> str:
    """``numerator / denominator`` cast back to DECIMAL (DuckDB ``/`` -> DOUBLE)."""
    return (
        f"CAST(CAST({numerator} AS {_DIV_DECIMAL}) "
        f"/ CAST({denominator} AS {_DIV_DECIMAL}) AS {_DIV_DECIMAL})"
    )


def _ratio_minus_one(numerator: str, denominator: str) -> str:
    """``(numerator / denominator) - 1`` in DECIMAL, NULL when denominator is 0."""
    div = _decimal_div(numerator, denominator)
    return f"CASE WHEN {denominator} = 0 THEN NULL ELSE {div} - 1 END"


def _history_ctes(series_id: str, obs_view: str) -> str:
    """CTEs ``obs/hist/ranked/current`` for one series' as-of history.

    Depends on table ``feature_dates_rel(feature_date)`` existing on the conn.
    """
    series_lit = _sql_str_literal(series_id)
    return f"""
        obs AS (
            SELECT raw_id, available_from_date, value_numeric,
                   COALESCE(period_end_date, observation_date) AS period_date,
                   release_date, fetched_at, vintage
            FROM {obs_view}
            WHERE series_id = {series_lit}
              AND available_from_date IS NOT NULL
              AND value_numeric IS NOT NULL
        ),
        hist AS (
            SELECT fd.feature_date, o.*
            FROM {_FEATURE_DATES_TABLE} fd
            JOIN obs o ON o.available_from_date <= fd.feature_date
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY fd.feature_date, o.period_date
                ORDER BY COALESCE(o.release_date, o.available_from_date) DESC,
                         o.available_from_date DESC,
                         o.fetched_at DESC,
                         o.vintage DESC,
                         o.raw_id DESC
            ) = 1
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY feature_date ORDER BY period_date ASC
                   ) AS pos,
                   COUNT(*) OVER (PARTITION BY feature_date) AS hist_len
            FROM hist
        ),
        current AS (
            SELECT * FROM ranked WHERE pos = hist_len
        )
    """


def _stale_expr(max_stale_business_days: int) -> str:
    return (
        f"(({_bisect_right_expr('cur.feature_date')}) "
        f"- ({_bisect_right_expr('cur.available_from_date')})) > {max_stale_business_days}"
    )


def build_single_input_sql(
    feature: CommonFeatureCatalogEntry,
    *,
    default_transform: str,
    max_stale_business_days: int,
    obs_view: str,
) -> str | None:
    """Self-contained SELECT of cfdf rows for one single-input feature.

    Returns ``None`` for an unsupported transform (Python records a partial
    error and emits no facts for it).
    """
    series_id = feature.input_series_ids[0]
    transform_code = _resolve_transform(feature, default_transform)
    feature_code_lit = _sql_str_literal(feature.feature_code)
    series_lit = _sql_str_literal(series_id)
    unit_lit = _sql_str_literal(feature.unit)
    stale = _stale_expr(max_stale_business_days)
    ctes = _history_ctes(series_id, obs_view)

    windowed = _parse_windowed(transform_code)

    # Common projection of the non-value columns.
    tail = f"""
            {unit_lit} AS unit,
            [{series_lit}] AS source_series_ids,
            COALESCE(cur.available_from_date, cur.feature_date) AS asof_available_date,
            cur.vintage AS selected_vintage
    """

    if transform_code == "level":
        return f"""
        WITH {ctes}
        SELECT cur.feature_date, {feature_code_lit} AS feature_code,
            CASE WHEN {stale} THEN NULL ELSE cur.value_numeric END AS value_numeric,
            {tail}
        FROM current cur
        """

    if windowed is not None and windowed[0] in {"ret", "change"}:
        n = windowed[1]
        if windowed[0] == "change":
            value_sql = "(cur.value_numeric - base.value_numeric)"
        else:
            value_sql = _ratio_minus_one("cur.value_numeric", "base.value_numeric")
        return f"""
        WITH {ctes}
        SELECT cur.feature_date, {feature_code_lit} AS feature_code,
            CASE WHEN {stale} THEN NULL ELSE {value_sql} END AS value_numeric,
            {tail}
        FROM current cur
        LEFT JOIN ranked base
          ON base.feature_date = cur.feature_date AND base.pos = cur.pos - {n}
        """

    if windowed is not None and windowed[0] == "vol":
        return _build_vol_sql(
            ctes, feature_code_lit, series_lit, unit_lit, stale, window=windowed[1]
        )

    if transform_code in {"yoy", "mom"}:
        months = 12 if transform_code == "yoy" else 1
        value_sql = _ratio_minus_one("cur.value_numeric", "base.value_numeric")
        return f"""
        WITH {ctes}
        SELECT cur.feature_date, {feature_code_lit} AS feature_code,
            CASE WHEN {stale} THEN NULL ELSE {value_sql} END AS value_numeric,
            {tail}
        FROM current cur
        LEFT JOIN ranked base
          ON base.feature_date = cur.feature_date
         AND (year(base.period_date) * 12 + (month(base.period_date) - 1))
             = (year(cur.period_date) * 12 + (month(cur.period_date) - 1)) - {months}
        """

    return None


def _build_vol_sql(
    ctes: str,
    feature_code_lit: str,
    series_lit: str,
    unit_lit: str,
    stale: str,
    *,
    window: int,
) -> str:
    tail = f"""
            {unit_lit} AS unit,
            [{series_lit}] AS source_series_ids,
            COALESCE(cur.available_from_date, cur.feature_date) AS asof_available_date,
            cur.vintage AS selected_vintage
    """
    if window < 2:
        return f"""
        WITH {ctes}
        SELECT cur.feature_date, {feature_code_lit} AS feature_code,
            CAST(NULL AS DOUBLE) AS value_numeric,
            {tail}
        FROM current cur
        """
    return f"""
    WITH {ctes},
    step_returns AS (
        SELECT r.feature_date, r.pos,
               CASE WHEN prev.value_numeric = 0 THEN NULL
                    ELSE (r.value_numeric / prev.value_numeric) - 1 END AS ret
        FROM ranked r
        JOIN ranked prev
          ON prev.feature_date = r.feature_date AND prev.pos = r.pos - 1
    ),
    vol AS (
        SELECT cur.feature_date,
               stddev_samp(CAST(sr.ret AS DOUBLE)) AS vol_value,
               COUNT(*) AS n_ret,
               SUM(CASE WHEN sr.ret IS NULL THEN 1 ELSE 0 END) AS n_bad
        FROM current cur
        JOIN step_returns sr
          ON sr.feature_date = cur.feature_date
         AND sr.pos BETWEEN cur.pos - {window} + 1 AND cur.pos
        GROUP BY cur.feature_date
    )
    SELECT cur.feature_date, {feature_code_lit} AS feature_code,
        CASE
            WHEN {stale} THEN NULL
            WHEN cur.pos - {window} < 0 THEN NULL
            WHEN v.n_ret < {window} THEN NULL
            WHEN v.n_bad > 0 THEN NULL
            ELSE v.vol_value
        END AS value_numeric,
        {tail}
    FROM current cur
    LEFT JOIN vol v ON v.feature_date = cur.feature_date
    """


def build_multi_input_sql(
    feature: CommonFeatureCatalogEntry,
    *,
    series_max_stale: dict[str, int],
    obs_view: str,
) -> str | None:
    """Self-contained SELECT of cfdf rows for one multi-input feature.

    spread = long - short, ratio = num / den (den 0 -> NULL). Each role's input
    independently computes its as-of current + stale gate; the combined value is
    NULL if any input is NULL or stale.
    """
    required_roles = _MULTI_INPUT_TRANSFORMS[feature.transform_code]
    series_by_role = feature.series_by_role()
    if any(role not in series_by_role for role in required_roles):
        return None

    feature_code_lit = _sql_str_literal(feature.feature_code)
    unit_lit = _sql_str_literal(feature.unit)

    role_ctes: list[str] = []
    role_aliases: list[str] = []
    series_ids: list[str] = []
    for idx, role in enumerate(required_roles):
        series_id = series_by_role[role]
        series_ids.append(series_id)
        max_stale = series_max_stale.get(series_id, 0)
        alias = f"in{idx}"
        role_aliases.append(alias)
        # Per-input current + stale, exposed as one row per feature_date.
        stale = (
            f"(({_bisect_right_expr('cur.feature_date')}) "
            f"- ({_bisect_right_expr('cur.available_from_date')})) > {max_stale}"
        )
        role_ctes.append(f"""
        {alias}_h AS ({_history_ctes_inline(series_id, obs_view)}),
        {alias} AS (
            SELECT cur.feature_date,
                   CASE WHEN {stale} THEN NULL ELSE cur.value_numeric END AS value_numeric,
                   cur.available_from_date,
                   cur.vintage
            FROM {alias}_h cur WHERE cur.pos = cur.hist_len
        )""")

    # spread/ratio combine
    long_alias, short_alias = role_aliases[0], role_aliases[1]
    if feature.transform_code == "spread":
        combined = f"({long_alias}.value_numeric - {short_alias}.value_numeric)"
    else:  # ratio: num/den (NOT minus one), den 0 -> NULL; keep DECIMAL exactness
        num = f"{long_alias}.value_numeric"
        den = f"{short_alias}.value_numeric"
        combined = f"CASE WHEN {den} = 0 THEN NULL ELSE {_decimal_div(num, den)} END"
    series_list = ", ".join(_sql_str_literal(s) for s in series_ids)
    # asof = max over inputs; vintage from first input (index 0).
    asof_terms = ", ".join(
        f"COALESCE({a}.available_from_date, fd.feature_date)" for a in role_aliases
    )
    joins = "\n        ".join(
        f"LEFT JOIN {a} ON {a}.feature_date = fd.feature_date" for a in role_aliases
    )

    return f"""
    WITH {",".join(role_ctes)}
    SELECT
        fd.feature_date,
        {feature_code_lit} AS feature_code,
        {combined} AS value_numeric,
        {unit_lit} AS unit,
        [{series_list}] AS source_series_ids,
        greatest({asof_terms}) AS asof_available_date,
        COALESCE({long_alias}.vintage, '') AS selected_vintage
    FROM {_FEATURE_DATES_TABLE} fd
        {joins}
    """


def _history_ctes_inline(series_id: str, obs_view: str) -> str:
    """Like ``_history_ctes`` but as a single subquery body (for multi-input).

    Returns one row per feature_date with pos/hist_len so the caller can pick
    ``current`` (pos = hist_len).
    """
    series_lit = _sql_str_literal(series_id)
    return f"""
            WITH obs AS (
                SELECT raw_id, available_from_date, value_numeric,
                       COALESCE(period_end_date, observation_date) AS period_date,
                       release_date, fetched_at, vintage
                FROM {obs_view}
                WHERE series_id = {series_lit}
                  AND available_from_date IS NOT NULL
                  AND value_numeric IS NOT NULL
            ),
            hist AS (
                SELECT fd.feature_date, o.*
                FROM {_FEATURE_DATES_TABLE} fd
                JOIN obs o ON o.available_from_date <= fd.feature_date
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY fd.feature_date, o.period_date
                    ORDER BY COALESCE(o.release_date, o.available_from_date) DESC,
                             o.available_from_date DESC,
                             o.fetched_at DESC,
                             o.vintage DESC,
                             o.raw_id DESC
                ) = 1
            )
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY feature_date ORDER BY period_date ASC
                   ) AS pos,
                   COUNT(*) OVER (PARTITION BY feature_date) AS hist_len
            FROM hist
    """


def _series_config(
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    series_view: str = "common_feature_series",
) -> tuple[dict[str, int], dict[str, str]]:
    """Return (max_stale_business_days, default_transform) keyed by series_id.

    Decision 7: the collector and compute share the *same* series rows. When the
    ``common_feature_series`` lake view is registered on ``con`` we read it from
    there (the rows sj2 actually seeded); otherwise we fall back to the code
    definition (which is what seeds that table) — identical by construction, so
    there is no drift branch to gate.
    """
    if con is not None and _view_exists(con, series_view):
        rows = con.execute(
            f"SELECT series_id, max_stale_business_days, default_transform FROM {series_view}"
        ).fetchall()
        max_stale = {r[0]: int(r[1]) for r in rows}
        default_transform = {r[0]: (r[2] or "level") for r in rows}
        return max_stale, default_transform

    series: list[CommonFeatureSeries] = default_common_feature_series()
    max_stale = {s.series_id: s.max_stale_business_days for s in series}
    default_transform = {s.series_id: (s.default_transform or "level") for s in series}
    return max_stale, default_transform


def _view_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return rows is not None


def register_common_feature_daily_fact_view(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: Sequence[date],
    feature_dates: Sequence[date],
    obs_view: str = "common_feature_observation_raw",
    view_name: str = CFDF_VIEW,
    feature_codes: Sequence[str] | None = None,
    active_only: bool = True,
) -> str:
    """Build ``common_feature_daily_fact`` from the raw observation view.

    ``trading_days`` is the KRX session calendar spanning the stale window (same
    one the Python ``_build_stale_calendar`` uses); ``feature_dates`` are the KRX
    sessions to emit facts for. Both are provided by the orchestrator so the mart
    stays calendar-source-agnostic.
    """
    max_stale, default_transform = _series_config(con)

    # Shared calendars as real tables so each feature SQL can reference them.
    con.execute(f"DROP TABLE IF EXISTS {_CAL_TABLE}")
    con.execute(f"CREATE TABLE {_CAL_TABLE} (d DATE, idx BIGINT)")
    con.executemany(
        f"INSERT INTO {_CAL_TABLE} VALUES (?, ?)",
        [(d, i + 1) for i, d in enumerate(trading_days)],
    )
    con.execute(f"DROP TABLE IF EXISTS {_FEATURE_DATES_TABLE}")
    con.execute(f"CREATE TABLE {_FEATURE_DATES_TABLE} (feature_date DATE)")
    con.executemany(
        f"INSERT INTO {_FEATURE_DATES_TABLE} VALUES (?)",
        [(d,) for d in feature_dates],
    )

    catalog = default_common_feature_catalog()
    if feature_codes is not None:
        wanted = set(feature_codes)
        catalog = [f for f in catalog if f.feature_code in wanted]
    if active_only:
        catalog = [f for f in catalog if f.active]

    selects: list[str] = []
    for feature in catalog:
        if feature.transform_code in _MULTI_INPUT_TRANSFORMS:
            sql = build_multi_input_sql(feature, series_max_stale=max_stale, obs_view=obs_view)
        else:
            series_id = feature.input_series_ids[0]
            sql = build_single_input_sql(
                feature,
                default_transform=default_transform.get(series_id, "level"),
                max_stale_business_days=max_stale.get(series_id, 0),
                obs_view=obs_view,
            )
        if sql is not None:
            selects.append(f"SELECT * FROM ({sql})")

    con.execute(f"DROP TABLE IF EXISTS {_BUILD_TABLE}")
    if not selects:
        con.execute(
            f"CREATE TABLE {_BUILD_TABLE} (feature_date DATE, feature_code VARCHAR, "
            "value_numeric DOUBLE, unit VARCHAR, source_series_ids VARCHAR[], "
            "asof_available_date DATE, selected_vintage VARCHAR)"
        )
    else:
        con.execute(f"CREATE TABLE {_BUILD_TABLE} AS {' UNION ALL '.join(selects)}")
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {_BUILD_TABLE}")
    return view_name
