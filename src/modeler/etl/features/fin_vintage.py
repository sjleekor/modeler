# ruff: noqa: E501
"""Shared strict-PIT vintage primitives for the daily financial marts (F-4.1).

``feat_fin_scan_daily`` worked out how to read ``fin_quarterly_metric_vintage``
onto a daily panel without leaking: TTM for flow metrics and the instant itself
otherwise, one interval per availability date with same-day candidates resolved
to the latest fiscal period, and a single ``fs_basis`` decision per ticker-date
so a day's bundle cannot mix consolidated and separate figures. ``feat_fin_risk``
(F-4) needs exactly those rules on a different set of metrics, so they live here
instead of being written a second time — a second copy is how two marts end up
disagreeing about what "the company's total assets on this date" means.

**The text is the contract.** ``mart._sql_hash`` keys the mart cache on the SQL
string, and every published Phase A/B run records the lineage that follows from
it. So the builders below return ``fin_scan.py``'s text *verbatim*, indentation
included, rather than a tidied-up equivalent: extraction only. That is what
``test_fin_risk.py::test_the_fin_scan_sql_is_byte_identical`` pins, and why
``BASE_OK_SQL`` carries its own continuation indent.

See ``docs/dev/20260907_additional_feature/03_financial_risk_lifecycle_transition.md``
§2 and ``04_specific_plan_B.md`` §3.6 (the shared-basis argument).
"""

from __future__ import annotations

from collections.abc import Sequence

#: The two financial-statement bases every metric is interval-joined under.
FS_BASES: tuple[str, ...] = ("CFS", "OFS")

#: The tradability/quality precondition every daily ratio is gated on. Emitted
#: with its original indentation so the extraction is byte-neutral for
#: ``feat_fin_scan_daily`` (module docstring); a caller places it where the
#: 12-space-indented first line belongs.
BASE_OK_SQL = """(market_cap_pit IS NOT NULL AND market_cap_pit > 0
             AND shares_is_available AND NOT shares_invalid_flag
             AND NOT COALESCE(is_halted, TRUE) AND valid_session_idx IS NOT NULL
            )"""


def build_metric_intervals_cte(
    vintage_view: str,
    metrics: Sequence[str],
    *,
    with_previous: bool = False,
) -> str:
    """The ``metric_points`` / ``_ranked`` / ``metric_intervals`` CTE trio.

    One row per ``(ticker, metric_code, fs_basis, quarter)`` carrying the value
    a daily feature wants — TTM for flow metrics, the instant otherwise — plus
    the ``[available_from, next_available_from)`` interval it is the current
    figure for.

    Args:
        vintage_view: B-3's ``fin_quarterly_metric_vintage``.
        metrics: Metric codes to project, in the order the caller joins them.
        with_previous: Also project the *preceding* interval's value and
            availability date. F-4's transition families are all statements
            about a change between two consecutive vintages
            (``fin_profit_turn``, ``fin_negative_equity_exit``,
            ``fin_lifecycle_transition``), and "the previous vintage" has to
            mean the previous *interval* — not the previous quarter, which a
            catching-up filer may have published on the same day. Off by
            default so the emitted text stays byte-identical for
            ``feat_fin_scan_daily``.
    """
    metrics_in = "(" + ", ".join(f"'{m}'" for m in metrics) + ")"
    previous = (
        ""
        if not with_previous
        else """,
            LAG(daily_value) OVER (
                PARTITION BY ticker, metric_code, fs_basis ORDER BY daily_available_from
            ) AS prev_daily_value,
            LAG(daily_available_from) OVER (
                PARTITION BY ticker, metric_code, fs_basis ORDER BY daily_available_from
            ) AS prev_available_from"""
    )
    return f"""
    metric_points AS (
        SELECT
            ticker, metric_code, fs_basis, seq_key, rcept_no,
            CASE WHEN metric_kind IN ('direct_interim', 'cumulative_reported')
                 THEN ttm_value ELSE standalone_value END AS daily_value,
            CASE WHEN metric_kind IN ('direct_interim', 'cumulative_reported')
                 THEN ttm_available_from ELSE available_from END AS daily_available_from,
            value_lag_4q
        FROM {vintage_view}
        WHERE metric_code IN {metrics_in}
    ),
    -- fin_v3: a filer catching up files several periods at once, so more than
    -- one value can become available on the same day — 6,024 such groups, e.g.
    -- 038530 on 2020-08-10 carrying net income for 2016 Q4 through 2020 Q1.
    -- ``daily_available_from`` alone is then not a total order and the interval
    -- boundaries (and so the value read on every later date) depended on scan
    -- order. Same-day candidates are resolved to the *latest fiscal period* —
    -- the figure describing the company's most recent state, which is what the
    -- daily feature means — and the losers dropped before LEAD runs, so the
    -- interval chain is built from one row per availability date.
    -- See 10_known_issues.md I12.
    metric_points_ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY ticker, metric_code, fs_basis, daily_available_from
                ORDER BY seq_key DESC, rcept_no DESC
            ) AS same_day_rank
        FROM metric_points
        WHERE daily_available_from IS NOT NULL AND daily_value IS NOT NULL
    ),
    metric_intervals AS (
        SELECT
            ticker, metric_code, fs_basis, daily_value, value_lag_4q,
            daily_available_from,
            LEAD(daily_available_from) OVER (
                PARTITION BY ticker, metric_code, fs_basis ORDER BY daily_available_from
            ) AS next_available_from{previous}
        FROM metric_points_ranked
        WHERE same_day_rank = 1
    )
    """


def build_metric_join(metric_code: str, basis: str) -> str:
    """One ``LEFT JOIN metric_intervals`` restricted to the current interval."""
    alias = f"m_{metric_code}_{basis.lower()}"
    return f"""
        LEFT JOIN metric_intervals {alias}
          ON {alias}.ticker = panel.ticker
         AND {alias}.metric_code = '{metric_code}' AND {alias}.fs_basis = '{basis}'
         AND {alias}.daily_available_from <= panel.trade_date
         AND ({alias}.next_available_from IS NULL
              OR panel.trade_date < {alias}.next_available_from)"""


def build_metric_joins(metrics: Sequence[str], bases: Sequence[str] = FS_BASES) -> str:
    """Every ``(metric, basis)`` join, metric-major — the order fin_scan uses."""
    return "\n".join(build_metric_join(metric, basis) for metric in metrics for basis in bases)


def build_wide_intervals_cte(
    metrics: Sequence[str],
    bases: Sequence[str] = FS_BASES,
    *,
    with_previous: bool = False,
    lag_4q_metrics: Sequence[str] = (),
) -> str:
    """``wide_intervals``: every metric's current value on one row per filing date.

    Reads ``metric_intervals`` and returns one row per ``(ticker,
    availability date)`` carrying all ``metrics x bases`` values as of that
    date, so the daily panel needs *one* ASOF join instead of one per
    metric/basis pair.

    **Why this shape.** The obvious form — one equality-plus-range join per
    metric/basis — asks the engine to consider every interval of a ticker
    against every session of that ticker, for each of the 20 pairs
    ``feat_fin_risk`` needs. On the real panel (7.2M sessions, 1.17M intervals)
    that does not complete. Twenty chained ASOF joins do not either. Pivoting
    first moves the work to a ~600k-row grid and leaves a single ASOF.

    **Why it is the same lookup.** Per ``(ticker, metric, basis)`` the
    intervals are contiguous and non-overlapping, so the value at session t is
    the one from the greatest availability date <= t. The grid below is the
    union of *all* metrics' availability dates for that ticker, and each
    column is forward-filled along it, so at the greatest grid date d <= t a
    column holds the value from the greatest date of *its own* metric that is
    <= d. Since that metric's dates are a subset of the grid, that is the same
    as the greatest one <= t. Identical row, both ways —
    ``test_fin_risk.py::test_the_wide_pivot_and_the_range_join_agree`` checks
    it against the range form on staggered filing dates.

    Args:
        metrics: Metric codes to widen.
        bases: Financial-statement bases.
        with_previous: Also carry each metric's preceding interval value and
            availability date (requires ``with_previous=True`` upstream).
        lag_4q_metrics: Metrics whose ``value_lag_4q`` is also carried.
    """
    pairs = [(metric, basis) for metric in metrics for basis in bases]

    def _fields(metric: str, basis: str) -> list[tuple[str, str]]:
        suffix = f"{metric}_{basis.lower()}"
        match = f"metric_code = '{metric}' AND fs_basis = '{basis}'"
        out = [
            (f"v_{suffix}", f"CASE WHEN {match} THEN daily_value END"),
            (f"a_{suffix}", f"CASE WHEN {match} THEN daily_available_from END"),
        ]
        if with_previous:
            out += [
                (f"pv_{suffix}", f"CASE WHEN {match} THEN prev_daily_value END"),
                (f"pa_{suffix}", f"CASE WHEN {match} THEN prev_available_from END"),
            ]
        if metric in lag_4q_metrics:
            out.append((f"l4_{suffix}", f"CASE WHEN {match} THEN value_lag_4q END"))
        return out

    columns = [field for metric, basis in pairs for field in _fields(metric, basis)]
    # A filing date can carry several metrics, so the pivot aggregates; each
    # CASE is non-NULL for exactly one metric/basis, so MAX is a pick, not a
    # choice between competing values.
    picked = ",\n            ".join(f"MAX({expr}) AS {alias}" for alias, expr in columns)
    # Forward-fill along the ticker's own filing dates. UNBOUNDED PRECEDING
    # with IGNORE NULLS is "the most recent non-NULL", which is exactly "the
    # value that was current as of this date".
    filled = ",\n            ".join(
        f"last_value({alias} IGNORE NULLS) OVER w AS {alias}" for alias, _ in columns
    )
    return f"""
    interval_grid AS (
        SELECT
            ticker, daily_available_from AS asof_date,
            {picked}
        FROM metric_intervals
        GROUP BY ticker, daily_available_from
    ),
    wide_intervals AS (
        SELECT
            ticker, asof_date,
            {filled}
        FROM interval_grid
        WINDOW w AS (
            PARTITION BY ticker ORDER BY asof_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    )
    """


def build_wide_intervals_asof_join(alias: str = "iv") -> str:
    """The single ASOF join from a ``panel`` CTE onto ``wide_intervals``."""
    return f"""
        ASOF LEFT JOIN wide_intervals {alias}
          ON {alias}.ticker = panel.ticker
         AND {alias}.asof_date <= panel.trade_date"""


def select_by_basis(metric: str, alias: str | None = None) -> str:
    """``CASE`` picking one metric's value under the ticker-date's chosen basis.

    The basis is decided once per ticker-date by whether ``net_income`` has a
    CFS value, and every other metric follows that decision (§3.6). Resolving
    each metric's basis independently would let a single day's bundle mix bases
    — the failure this rule exists to prevent.
    """
    target = alias or f"{metric}_selected"
    return (
        f"CASE WHEN v_net_income_cfs IS NOT NULL THEN v_{metric}_cfs\n"
        f"                 ELSE v_{metric}_ofs END AS {target}"
    )
