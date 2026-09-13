# ruff: noqa: E501
"""``dim_industry_pit_daily`` — point-in-time industry classification (F-1.5).

Design: ``docs/dev/20260907_additional_feature/01_industry_pit.md`` §2.3.

``dart_corp_master.induty_code`` is an UPSERT target, so it holds *today's* KSIC
code with no history: every industry-neutral variant built on it puts a 2026
classification into a 2015 z-score. ``dart_corp_profile_history`` (F-1.1~F-1.4)
fixes that going forwards by snapshotting ``company.json`` once a month, and this
mart is what turns those monthly snapshots into a daily as-of dimension.

It does **not** fix the past, and it does not pretend to. Sessions before the
first snapshot carry the earliest observed code with ``ind_is_backcast = TRUE``,
which is the look-ahead marker ``01`` §2.3 requires: a consumer decides what to
do with those rows, and both the Horizon Scan families and the model's FS3 treat
them as NULL until D-F1 is decided (``01`` §4).

**Exposure rule — one session later than ``01`` §2.3's ``<= t``.** That section
says "the last ``induty_code`` observed at or before session t". Taken
literally that is unsafe: the seed rows were observed at 23:10 KST and the
2026-09 snapshot at 00:14 KST, so "observed on date t" can mean either before
or after session t's close depending on when the Cronicle event happened to
run. This mart therefore exposes an observation from the **next valid session
strictly after** ``date(observed_at)``, which is the repo-wide rule for a
domestic observation (``00_overview.md`` §4.3) and holds whatever time of day
the monthly run lands on. The cost is one session of delay on a value that
changes a few times a year.

**Fold-up universe.** ``ind_group`` comes from
:func:`collector.kr.definitions.industry_groups.resolve_groups` — the rule
fixed in N2-6 (2-digit KSIC prefix, fold into the section below
``MIN_GROUP_SIZE`` members, then into ``OTHER``), reused rather than
reselected (``01`` §7). It is resolved **once per ``observed_month``**, over
every ticker-bearing corporation in that month's snapshot, including the
delisted ones (``corp_cls = 'E'``). Two consequences worth stating:

- Counting delisted names means a group can clear ``MIN_GROUP_SIZE`` on names
  that no longer trade. Restricting the count to live names was tried and
  rejected: the label map would then be resolved on the 2026 listed universe
  and applied to 2015 sessions, which sends most of the historical panel to
  ``OTHER`` — a worse distortion than the one it removes, and one that lands
  hardest exactly on the delisted cohort F-9.3 showed we cannot backfill.
- A consumer that needs a group whose members are all live re-resolves on its
  own universe; ``resolve_groups`` is a pure function and takes any mapping.

**Change flags.** ``ind_changed_recent_252`` is NULL, not FALSE, until two
*consecutive non-seed* observations have been exposed. The seed rows are
copies of ``dart_corp_master``'s current state, so a difference between the seed
and the first real snapshot may have happened at any time before it (the DDL
comment on ``dart_corp_profile_history.is_seed`` says the same) — dating such a
change to the snapshot that first saw it would be a fabrication, and F-1.4
measured one of the two differences in the first pair to be a *precision*
change (262 -> 26293), not an industry change at all. So a transition is only
counted between two non-seed observations, and "not enough observations to
tell" reads as NULL. The first month this flag can be non-NULL is 2026-11 —
2026-10's snapshot is the first whose predecessor is itself not a seed.

Once it *is* measurable it stays measurable: the guard is a cumulative count,
not a windowed one, so a month with no new observation reads FALSE ("no change
was observed") rather than falling back to NULL.

``ind_group_changed_recent_252`` applies the same rule to ``ind_group``. It is
additive to ``01`` §2.3's five columns because the D-F1 threshold (0.3%,
``01`` §4.2) is stated on the *group* rate, and because the group comparison is
what discriminates a real industry move from the precision artifact above.

**Grain** ``(trade_date, ticker, market)``, valid sessions only — halt rows
(``open=high=low=0``) never enter, exactly as in ``build_valid_session_sql``.
"""

from __future__ import annotations

import json
import logging

import duckdb
from collector.kr.shared import (
    MIN_GROUP_SIZE,
    UNKNOWN_GROUP,
    resolve_groups,
)

from modeler.etl.config import LakeConfig
from modeler.etl.mart import (
    _metadata_path,
    _sql_hash,
    is_materialized,
    materialize,
    register_mart_view,
)
from modeler.etl.trading_panel import build_valid_session_sql

logger = logging.getLogger(__name__)

INDUSTRY_PIT_TABLE = "dim_industry_pit_daily"

#: Bump when the as-of rule, the fold-up universe or a column formula changes.
#: Part of the mart's cache-contract text, so a change forces the rebuild.
#:
#:   industry_pit_v1 — the 2026-09 rules as first built (01 §2.3).
FORMULA_VERSION = "industry_pit_v1"

#: Lookback for the recency flags, in that ticker's own valid sessions (01 §2.3).
CHANGE_LOOKBACK_SESSIONS = 252

#: The columns this mart publishes beyond the grain.
FEATURE_COLUMNS: tuple[str, ...] = (
    "ind_ksic_code",
    "ind_group",
    "ind_observed_at",
    "ind_is_backcast",
    "ind_changed_recent_252",
    "ind_group_changed_recent_252",
)

_STAGING_TABLE = "_industry_pit_staging"


# ---------------------------------------------------------------------------
# stage (a): monthly observations + the per-month fold-up, resolved in Python
# ---------------------------------------------------------------------------


def compute_industry_observations(
    con: duckdb.DuckDBPyConnection,
    *,
    history_view: str = "dart_corp_profile_history",
    table_name: str = _STAGING_TABLE,
    min_group_size: int = MIN_GROUP_SIZE,
) -> str:
    """Stage one row per ``(ticker, observed_month)`` with its fold-up group.

    The fold-up cannot be expressed in SQL because it is data-dependent: whether
    a 2-digit group stands on its own depends on how many members it has that
    month, and whether its section stands depends on the same count one level
    up. That loop runs in Python — the same division of labour as
    ``modeler.etl.industry.register_industry_group_view`` — and its answer is
    handed back to DuckDB as a lookup table.

    Args:
        con: DuckDB connection with *history_view* registered.
        history_view: Raw ``dart_corp_profile_history`` view name.
        table_name: Temp table to write.
        min_group_size: Smallest group allowed to stand on its own.

    Returns:
        The staged table name.

    Raises:
        RuntimeError: When the history view holds no ticker-bearing row. An
            empty industry dimension would make every group ``XX`` and read as
            "industry does not matter", which is the failure
            ``register_industry_group_view`` also refuses to produce silently.
    """
    rows = con.execute(f"""
        SELECT observed_month, ticker, induty_code, observed_at, is_seed
        FROM {history_view}
        WHERE ticker IS NOT NULL AND ticker <> ''
        ORDER BY observed_month, ticker
        """).fetchall()
    if not rows:
        raise RuntimeError(
            f"{history_view} has no ticker-bearing row; the monthly snapshot "
            "(F-1.4) has not reached this lake. Refresh it before building "
            f"{INDUSTRY_PIT_TABLE}."
        )

    by_month: dict[object, dict[str, str | None]] = {}
    for observed_month, ticker, induty_code, _observed_at, _is_seed in rows:
        by_month.setdefault(observed_month, {})[str(ticker)] = induty_code

    groups_by_month = {
        month: resolve_groups(codes, min_group_size=min_group_size)
        for month, codes in by_month.items()
    }

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"""
        CREATE TEMP TABLE {table_name} (
            ticker VARCHAR, observed_month DATE, observed_at TIMESTAMP WITH TIME ZONE,
            observed_date DATE, induty_code VARCHAR, ind_group VARCHAR, is_seed BOOLEAN
        )
        """)
    con.executemany(
        f"INSERT INTO {table_name} VALUES (?, ?, ?, CAST(? AS DATE), ?, ?, ?)",
        [
            (
                str(ticker),
                observed_month,
                observed_at,
                observed_at,
                induty_code,
                groups_by_month[observed_month].get(str(ticker), UNKNOWN_GROUP),
                bool(is_seed),
            )
            for observed_month, ticker, induty_code, observed_at, is_seed in rows
        ],
    )
    logger.info(
        "%s: %d observations over %d months", table_name, len(rows), len(groups_by_month)
    )
    return table_name


# ---------------------------------------------------------------------------
# stage (b): the daily as-of dimension
# ---------------------------------------------------------------------------


def build_industry_pit_sql(
    staging_table: str = _STAGING_TABLE,
    *,
    price_view: str = "daily_ohlcv",
) -> str:
    """SQL producing ``dim_industry_pit_daily`` at ``(trade_date, ticker, market)``.

    The fold-up groups come out of Python, so this text is what stands in for
    the formula in the cache contract: the frozen constants appear in it and
    ``FORMULA_VERSION`` covers the rest.
    """
    return f"""
        -- {FORMULA_VERSION}: min_group_size={MIN_GROUP_SIZE} lookback={CHANGE_LOOKBACK_SESSIONS}
        -- exposure = next valid session strictly after date(observed_at)
        WITH sessions AS (
            SELECT trade_date, ticker, market
            FROM ({build_valid_session_sql(price_view)})
        ),
        obs AS (
            SELECT
                ticker, observed_month, observed_at, observed_date,
                induty_code, ind_group, is_seed,
                -- A transition is only readable between two non-seed
                -- observations; see the module docstring.
                (NOT is_seed AND NOT COALESCE(LAG(is_seed) OVER w, TRUE))
                    AS obs_change_measurable,
                (
                    NOT is_seed
                    AND NOT COALESCE(LAG(is_seed) OVER w, TRUE)
                    AND induty_code IS DISTINCT FROM LAG(induty_code) OVER w
                ) AS obs_code_changed,
                (
                    NOT is_seed
                    AND NOT COALESCE(LAG(is_seed) OVER w, TRUE)
                    AND ind_group IS DISTINCT FROM LAG(ind_group) OVER w
                ) AS obs_group_changed
            FROM {staging_table}
            WINDOW w AS (PARTITION BY ticker ORDER BY observed_month)
        ),
        first_obs AS (
            SELECT
                ticker,
                arg_min(induty_code, observed_month) AS induty_code,
                arg_min(ind_group, observed_month) AS ind_group,
                arg_min(observed_at, observed_month) AS observed_at
            FROM obs
            GROUP BY ticker
        ),
        as_of_obs AS (
            SELECT
                s.trade_date, s.ticker, s.market,
                o.observed_month, o.observed_at, o.induty_code, o.ind_group,
                o.is_seed, o.obs_change_measurable,
                o.obs_code_changed, o.obs_group_changed
            FROM sessions s
            ASOF LEFT JOIN obs o
                ON s.ticker = o.ticker AND s.trade_date > o.observed_date
        ),
        filled AS (
            SELECT
                a.trade_date, a.ticker, a.market,
                COALESCE(a.induty_code, f.induty_code) AS ind_ksic_code,
                COALESCE(a.ind_group, f.ind_group) AS ind_group,
                COALESCE(a.observed_at, f.observed_at) AS ind_observed_at,
                -- NULL when no observation exists at all: "this session predates
                -- the first snapshot" and "this ticker was never profiled" are
                -- different states and must not share a value.
                CASE
                    WHEN a.observed_month IS NOT NULL THEN FALSE
                    WHEN f.observed_at IS NOT NULL THEN TRUE
                END AS ind_is_backcast,
                a.observed_month,
                COALESCE(a.obs_change_measurable, FALSE) AS obs_change_measurable,
                COALESCE(a.obs_code_changed, FALSE) AS obs_code_changed,
                COALESCE(a.obs_group_changed, FALSE) AS obs_group_changed
            FROM as_of_obs a
            LEFT JOIN first_obs f ON a.ticker = f.ticker
        ),
        marked AS (
            SELECT
                *,
                (
                    observed_month IS NOT NULL
                    AND observed_month IS DISTINCT FROM LAG(observed_month) OVER w
                ) AS is_first_session_of_obs
            FROM filled
            WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
        ),
        windowed AS (
            SELECT
                *,
                -- Cumulative, not windowed: once a ticker has had two
                -- consecutive non-seed observations, "no change was observed in
                -- the last {CHANGE_LOOKBACK_SESSIONS} sessions" is a real answer
                -- even when no observation landed inside the window. Counting
                -- only inside it would send the flag back to NULL as soon as the
                -- change that set it scrolled past the edge.
                SUM(CASE WHEN is_first_session_of_obs AND obs_change_measurable THEN 1 ELSE 0 END)
                    OVER (
                        PARTITION BY ticker, market ORDER BY trade_date
                        ROWS UNBOUNDED PRECEDING
                    ) AS measurable_obs_cum,
                MAX(CASE WHEN is_first_session_of_obs AND obs_code_changed THEN 1 ELSE 0 END)
                    OVER w252 AS code_changed_252,
                MAX(CASE WHEN is_first_session_of_obs AND obs_group_changed THEN 1 ELSE 0 END)
                    OVER w252 AS group_changed_252
            FROM marked
            WINDOW w252 AS (
                PARTITION BY ticker, market ORDER BY trade_date
                ROWS BETWEEN {CHANGE_LOOKBACK_SESSIONS - 1} PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            trade_date, ticker, market,
            ind_ksic_code, ind_group, ind_observed_at, ind_is_backcast,
            -- No measurable observation yet means a change is undetectable, not
            -- absent: the only predecessor is the seed row, which cannot date it.
            CASE WHEN measurable_obs_cum >= 1 THEN code_changed_252 = 1 END
                AS ind_changed_recent_252,
            CASE WHEN measurable_obs_cum >= 1 THEN group_changed_252 = 1 END
                AS ind_group_changed_recent_252
        FROM windowed
    """


def _industry_pit_is_current(config: LakeConfig, contract_sql: str) -> bool:
    """True when the on-disk mart was built by these exact rules."""
    if not is_materialized(config, INDUSTRY_PIT_TABLE):
        return False
    path = _metadata_path(config, INDUSTRY_PIT_TABLE)
    if not path.is_file():
        return False
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stored.get("analysis_config_hash") != config.analysis_config_hash:
        return False
    return stored.get("sql_hash") == _sql_hash(contract_sql)


def materialize_industry_pit(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    history_view: str = "dart_corp_profile_history",
    force: bool = False,
) -> str:
    """Build + register the ``dim_industry_pit_daily`` mart view.

    Idempotent: an existing mart built under the same ``FORMULA_VERSION``,
    constants and config hash is reused, and the fold-up is not re-resolved.

    Args:
        con: DuckDB connection with *price_view* and *history_view* registered.
        config: Lake configuration.
        price_view: Raw ``daily_ohlcv`` view name — the session grid.
        history_view: Raw ``dart_corp_profile_history`` view name.
        force: Rebuild even when the mart already exists.

    Returns:
        The registered view name.
    """
    contract = build_industry_pit_sql(price_view=price_view)
    if not force and _industry_pit_is_current(config, contract):
        return register_mart_view(con, config, INDUSTRY_PIT_TABLE)

    staging = compute_industry_observations(con, history_view=history_view)
    materialize(
        con,
        config,
        INDUSTRY_PIT_TABLE,
        build_industry_pit_sql(staging, price_view=price_view),
        force=True,
    )
    con.execute(f"DROP TABLE IF EXISTS {staging}")
    return register_mart_view(con, config, INDUSTRY_PIT_TABLE)
