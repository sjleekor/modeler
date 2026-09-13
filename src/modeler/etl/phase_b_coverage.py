"""§7.1 mart-side quality/coverage diagnostics for Phase B (B-10 Stage 2).

The other five of the seven ``*_quality``/``*_coverage`` artifacts the run
directory contract lists. ``phase_b_quality.py`` holds the two that sit
directly on raw and ask "is the source good enough to build on"; these five sit
on the B-2…B-6 marts and ask the next question — "did the mart keep enough of
that source, and does the feature actually exist on enough dates for a scan
over it to mean anything".

``receipt_value_pairing_quality``  per (bsns_year, reprt_code) over
    ``stock_metric_vintage_fact``. How many rows sourced from
    ``dart_financial_statement_raw`` were confirmed against the *same receipt's*
    XBRL value. ``phase_b.receipt_value_pairing_required=verified_same_receipt``
    with ``receipt_value_pairing_error_tolerance=0`` is frozen in
    ``horizon_scan_config.yaml``, so a nonzero ``value_mismatch_rows`` is a
    blocker rather than a warning, and ``unlinked_receipt`` sizes how much XBRL
    the lake is still missing.

``stock_metric_vintage_quality``   per (metric_code, bsns_year, reprt_code,
    fs_basis) over the same mart. Availability fallback, revision ratio, mapping fallback,
    period-end conflicts, and how many positions carry more than one captured
    vintage. §4.2 needs ``mapping_fallback_ratio``/``revision_ratio`` to exist
    before evidence grade "B" can take them into account.

``quarterly_metric_quality``       per (metric_code, bsns_year, quarter) over
    ``fin_quarterly_metric_vintage``. B-3's standalone/TTM reconstruction: how
    often the direct interim value and the cumulative difference disagree, how
    often TTM completes, and how much later TTM becomes available than the
    filing that closed it.

``feature_coverage``               per (feature, variant, market, year) over the
    two daily marts. Nonnull ratio, names per date, first/last date with a
    value, and filing age. A family card's ``effective_start``/``coverage``
    lines come from here, and ``min_names_per_date`` is what decides whether a
    date survives the scan's ``min_names`` filter at all.

``event_coverage``                 per (event_year, market, reprt_code) over
    ``fin_sue_event``. Events, SUE completeness, constant-sample share, and
    per-bucket return availability — the event-grain analogue of the above.

All five are pure SQL over registered views, so the Phase B runner materializes
them exactly the way it materializes everything else in the run directory.

Missing groups are absent rows, never zero-filled — same rule as the raw-side
pair. A year with no filings is a gap you can see, not a row of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from collector.kr.shared import default_metric_mapping_rules
from modeler.etl.features.event_scan import EVENT_SCAN_TABLE
from modeler.etl.features.filing_activity import FILING_ACTIVITY_TABLE
from modeler.etl.features.fin_scan import FIN_SCAN_TABLE
from modeler.etl.features.market_cap import MARKET_CAP_TABLE
from modeler.etl.features.periodic_extras import PERIODIC_EXTRAS_TABLE
from modeler.etl.features.sue_event import (
    EVENT_BUCKETS,
    MIN_SUE_HISTORY,
    SUE_EVENT_TABLE,
    _bucket_col,
)
from modeler.etl.lake import _sql_str_literal
from modeler.etl.marts.financial_quarters import FQMV_TABLE
from modeler.etl.marts.metric_vintages import SMVF_TABLE

RECEIPT_VALUE_PAIRING_QUALITY_TABLE = "receipt_value_pairing_quality"
STOCK_METRIC_VINTAGE_QUALITY_TABLE = "stock_metric_vintage_quality"
QUARTERLY_METRIC_QUALITY_TABLE = "quarterly_metric_quality"
FEATURE_COVERAGE_TABLE = "feature_coverage"
EVENT_COVERAGE_TABLE = "event_coverage"


def build_receipt_value_pairing_quality_sql(*, vintage_view: str = SMVF_TABLE) -> str:
    """One row per (bsns_year, reprt_code) over ``stock_metric_vintage_fact``.

    ``*_ratio`` denominators are the *applicable* rows only — the ones
    ``receipt_value_pairing_status`` can actually judge, i.e. those sourced from
    ``dart_financial_statement_raw``. Dividing by all rows would let a year with
    lots of XBRL-sourced rows look better paired than it is.
    """
    applicable = "receipt_value_pairing_status <> 'not_applicable'"
    verified = "receipt_value_pairing_status = 'verified_same_receipt'"
    unlinked = "receipt_value_pairing_status = 'unlinked_receipt'"
    mismatch = "receipt_value_pairing_status = 'value_mismatch'"
    return f"""
    SELECT
        bsns_year,
        reprt_code,
        ANY_VALUE(period_type) AS period_type,
        COUNT(*) AS rows,
        COUNT(DISTINCT ticker) AS tickers,
        COUNT(*) FILTER (WHERE {applicable}) AS applicable_rows,
        COUNT(*) FILTER (WHERE {verified}) AS verified_rows,
        COUNT(*) FILTER (WHERE {unlinked}) AS unlinked_rows,
        COUNT(*) FILTER (WHERE {mismatch}) AS value_mismatch_rows,
        COUNT(*) FILTER (WHERE {verified})::DOUBLE
            / NULLIF(COUNT(*) FILTER (WHERE {applicable}), 0) AS verified_ratio,
        -- Frozen tolerance is 0, so this must be exactly 0.0 for the year to
        -- clear §1.2's hard gate. It is reported per year rather than as one
        -- boolean so a backfill can be aimed at the years that fail.
        COUNT(*) FILTER (WHERE {mismatch})::DOUBLE
            / NULLIF(COUNT(*) FILTER (WHERE {applicable}), 0) AS value_mismatch_ratio,
        COUNT(DISTINCT rcept_no) FILTER (WHERE {applicable}) AS applicable_receipts,
        COUNT(DISTINCT rcept_no) FILTER (WHERE {verified}) AS verified_receipts,
        COUNT(DISTINCT rcept_no) FILTER (WHERE {mismatch}) AS receipts_with_value_mismatch,
        MAX(pairing_tolerance) AS pairing_tolerance
    FROM {vintage_view}
    GROUP BY 1, 2
    ORDER BY 1, 2
    """


# The fs_basis values a vintage-fact row can carry: the two financial-statement
# bases, plus '' for rows sourced from share counts / shareholder returns /
# XBRL, which have no basis of their own.
_FS_BASES: tuple[str, ...] = ("", "CFS", "OFS")


def _catalog_best_priority_relation() -> str:
    """``(metric_code, fs_basis, catalog_best_priority)`` from the rule catalog.

    Taken from the catalog rather than from whatever the lake happens to hold,
    so ``mapping_fallback_ratio`` means "this row did not come from the metric's
    preferred rule" in an absolute sense. A metric whose preferred rule never
    matches anywhere then honestly reports a ratio of 1.0 instead of silently
    redefining its second-choice rule as the baseline.

    Keyed by ``fs_basis`` too, and that part is not cosmetic. ``net_income``'s
    best rule is priority 10 = (CFS, CIS); an OFS row cannot match it no matter
    how clean the filing is, because the rule names the other basis. Comparing
    every row against one global minimum therefore reported 65.5% fallback for
    ``net_income`` on the real lake — a statement about the rule table's shape,
    not about the data. Per-basis, an OFS row is measured against the best OFS
    rule (priority 30) and only a genuine second-choice match counts.
    """
    best: dict[tuple[str, str], int] = {}
    for rule in default_metric_mapping_rules():
        if not rule.is_active:
            continue
        for basis in _FS_BASES:
            # A rule with no fs_div declared applies to every basis.
            if rule.fs_div not in ("", basis):
                continue
            key = (rule.metric_code, basis)
            current = best.get(key)
            if current is None or rule.priority < current:
                best[key] = rule.priority
    if not best:
        return (
            "SELECT NULL::VARCHAR AS metric_code, NULL::VARCHAR AS fs_basis,"
            " NULL::INTEGER AS catalog_best_priority WHERE FALSE"
        )
    values = ", ".join(
        f"({_sql_str_literal(metric_code)}, {_sql_str_literal(basis)}, {priority})"
        for (metric_code, basis), priority in sorted(best.items())
    )
    return f"SELECT * FROM (VALUES {values}) AS t(metric_code, fs_basis, catalog_best_priority)"


def build_stock_metric_vintage_quality_sql(*, vintage_view: str = SMVF_TABLE) -> str:
    """One row per (metric_code, bsns_year, reprt_code, fs_basis) over B-2.

    ``fs_basis`` is part of the grain because the mapping catalog is: a metric's
    preferred rule can name one basis, so an OFS row must be judged against the
    best OFS rule rather than against a CFS-only one (see
    ``_catalog_best_priority_relation``). Collapsing the bases would make
    ``catalog_best_priority`` ambiguous within a group.

    ``revision_ratio`` is computed over rows whose ``is_revision`` is *known* —
    a row whose receipt never matched ``dart_filing_receipt_raw`` has
    ``is_revision IS NULL``, and counting those as "not a revision" would make
    the ratio look better precisely where receipt coverage is worst.
    """
    catalog = _catalog_best_priority_relation()
    return f"""
    WITH catalog_priority AS ({catalog}),
    positions AS (
        SELECT
            metric_code, bsns_year, reprt_code, fs_basis, ticker, statement_period_end,
            COUNT(DISTINCT rcept_no) AS vintages
        FROM {vintage_view}
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    position_stats AS (
        SELECT
            metric_code, bsns_year, reprt_code, fs_basis,
            COUNT(*) AS positions,
            COUNT(*) FILTER (WHERE vintages > 1) AS multi_vintage_positions,
            COUNT(*) FILTER (WHERE vintages > 1)::DOUBLE
                / NULLIF(COUNT(*), 0) AS multi_vintage_ratio
        FROM positions
        GROUP BY 1, 2, 3, 4
    ),
    row_stats AS (
        SELECT
            v.metric_code, v.bsns_year, v.reprt_code, v.fs_basis,
            ANY_VALUE(v.period_type) AS period_type,
            COUNT(*) AS rows,
            COUNT(DISTINCT v.ticker) AS tickers,
            COUNT(DISTINCT v.rcept_no) AS receipts,
            COUNT(*) FILTER (WHERE v.availability_source = 'synthetic_fallback')
                AS synthetic_availability_rows,
            COUNT(*) FILTER (WHERE v.availability_source = 'synthetic_fallback')::DOUBLE
                / NULLIF(COUNT(*), 0) AS synthetic_availability_ratio,
            COUNT(*) FILTER (WHERE v.captured_vintage_status = 'captured_vintages_only')
                AS captured_vintages_only_rows,
            COUNT(*) FILTER (
                WHERE v.captured_vintage_status = 'original_confirmed_revisions_partial'
            ) AS original_confirmed_rows,
            COUNT(*) FILTER (WHERE v.captured_vintage_status = 'unlinked_receipt')
                AS unlinked_receipt_rows,
            COUNT(*) FILTER (WHERE v.is_revision IS NOT NULL) AS revision_known_rows,
            COUNT(*) FILTER (WHERE v.is_revision) AS revision_rows,
            COUNT(*) FILTER (WHERE v.is_revision)::DOUBLE
                / NULLIF(COUNT(*) FILTER (WHERE v.is_revision IS NOT NULL), 0) AS revision_ratio,
            MIN(v.mapping_priority) AS observed_best_priority,
            ANY_VALUE(cp.catalog_best_priority) AS catalog_best_priority,
            COUNT(*) FILTER (WHERE v.mapping_priority > cp.catalog_best_priority)
                AS mapping_fallback_rows,
            COUNT(*) FILTER (WHERE v.mapping_priority > cp.catalog_best_priority)::DOUBLE
                / NULLIF(COUNT(*), 0) AS mapping_fallback_ratio,
            COUNT(DISTINCT v.mapping_rule_code) AS mapping_rules_used,
            COUNT(DISTINCT v.source_table) AS source_tables_used,
            COUNT(*) FILTER (WHERE v.source_table = 'dart_xbrl_fact_raw') AS xbrl_sourced_rows,
            COUNT(DISTINCT v.period_end_source) AS period_end_sources,
            COUNT(*) FILTER (WHERE v.period_end_conflict) AS period_end_conflict_rows,
            MIN(v.disclosed_date) AS min_disclosed_date,
            MAX(v.disclosed_date) AS max_disclosed_date,
            MIN(v.available_from) AS min_available_from,
            MAX(v.available_from) AS max_available_from
        FROM {vintage_view} v
        LEFT JOIN catalog_priority cp
          ON cp.metric_code = v.metric_code AND cp.fs_basis = v.fs_basis
        GROUP BY 1, 2, 3, 4
    )
    SELECT
        r.*,
        p.positions,
        p.multi_vintage_positions,
        p.multi_vintage_ratio
    FROM row_stats r
    JOIN position_stats p
      ON p.metric_code = r.metric_code
     AND p.bsns_year = r.bsns_year
     AND p.reprt_code = r.reprt_code
     AND p.fs_basis = r.fs_basis
    ORDER BY r.metric_code, r.bsns_year, r.reprt_code, r.fs_basis
    """


def build_quarterly_metric_quality_sql(*, quarterly_view: str = FQMV_TABLE) -> str:
    """One row per (metric_code, bsns_year, quarter) over B-3's output.

    ``standalone_source_conflict`` only exists for ``direct_interim`` metrics
    (the only kind with two independent ways to get the same number), so its
    ratio is taken over rows where the flag is not NULL rather than over every
    row — otherwise a metric kind that *cannot* conflict would dilute the rate
    of the one that can.
    """
    return f"""
    SELECT
        metric_code,
        bsns_year,
        quarter,
        ANY_VALUE(metric_kind) AS metric_kind,
        COUNT(*) AS rows,
        COUNT(DISTINCT ticker) AS tickers,
        COUNT(DISTINCT fs_basis) AS fs_bases,
        COUNT(*) FILTER (WHERE standalone_value IS NOT NULL) AS standalone_rows,
        COUNT(*) FILTER (WHERE standalone_value IS NOT NULL)::DOUBLE
            / NULLIF(COUNT(*), 0) AS standalone_ratio,
        COUNT(*) FILTER (WHERE standalone_source_conflict IS NOT NULL)
            AS conflict_checkable_rows,
        COUNT(*) FILTER (WHERE standalone_source_conflict) AS standalone_conflict_rows,
        COUNT(*) FILTER (WHERE standalone_source_conflict)::DOUBLE
            / NULLIF(COUNT(*) FILTER (WHERE standalone_source_conflict IS NOT NULL), 0)
            AS standalone_conflict_ratio,
        COUNT(*) FILTER (WHERE standalone_value < 0) AS negative_standalone_rows,
        COUNT(*) FILTER (WHERE cumulative_derived_value IS NOT NULL)
            AS cumulative_derived_rows,
        COUNT(*) FILTER (WHERE comparative_q_amount IS NOT NULL) AS comparative_rows,
        COUNT(*) FILTER (WHERE value_lag_4q IS NOT NULL) AS lag_4q_rows,
        COUNT(*) FILTER (WHERE ttm_complete) AS ttm_complete_rows,
        COUNT(*) FILTER (WHERE ttm_complete)::DOUBLE / NULLIF(COUNT(*), 0) AS ttm_complete_ratio,
        -- How much later the TTM becomes usable than the filing that closed it:
        -- TTM waits for the *latest* of its four quarters, so a late revision
        -- to an old quarter pushes the whole window's availability forward.
        AVG(CAST(ttm_available_from - available_from AS BIGINT))
            FILTER (WHERE ttm_complete) AS mean_ttm_availability_lag_days,
        MAX(CAST(ttm_available_from - available_from AS BIGINT))
            FILTER (WHERE ttm_complete) AS max_ttm_availability_lag_days,
        MIN(available_from) AS min_available_from,
        MAX(available_from) AS max_available_from
    FROM {quarterly_view}
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, 3
    """


@dataclass(frozen=True, slots=True)
class FeatureCoverageSpec:
    """One column of one daily mart, as ``feature_coverage`` measures it."""

    feature: str
    variant: str
    source_mart: str
    value_column: str
    age_expr: str | None
    precondition: str
    precondition_expr: str | None


# (feature, age expression, precondition label, precondition expression).
# The precondition is the mart-side condition that has to hold before the
# feature can be non-NULL at all — naming it per feature is what makes
# `precondition_ok_ratio` readable across rows that measure different things.
_FIN_PRIMARY: tuple[tuple[str, str | None, str, str | None], ...] = (
    # Market cap is priced, not filed — it has no filing age.
    ("fin_log_mcap", None, "none", None),
    (
        "fin_value_z",
        "value_fin_age_days",
        "value_components_ge_2",
        "value_component_count >= 2",
    ),
    ("fin_gross_profitability", "profitability_fin_age_days", "none", None),
    ("fin_operating_profitability", "profitability_fin_age_days", "none", None),
    ("fin_asset_growth_yoy", "asset_growth_fin_age_days", "none", None),
    ("fin_accruals_to_assets", "accruals_fin_age_days", "none", None),
)

# The four z-scored inputs behind fin_value_z. Not scan features themselves —
# they are here because "fin_value_z is NULL" is usually a statement about
# which of these four is missing, and that is not answerable from fin_value_z.
_FIN_VALUE_COMPONENTS: tuple[str, ...] = (
    "fin_book_to_market",
    "fin_earnings_yield",
    "fin_cfo_yield",
    "fin_sales_to_price",
)

_EVENT_PRIMARY: tuple[tuple[str, str | None, str, str | None], ...] = (
    (
        "ev_net_share_issuance_yoy",
        "CAST(trade_date - issuance_available_from AS BIGINT)",
        "issuance_identity_and_classification",
        "COALESCE(issuance_identity_ok, FALSE)"
        " AND COALESCE(issuance_classification_complete, FALSE)",
    ),
    (
        "ev_payout_yield",
        "CAST(trade_date - payout_available_from AS BIGINT)",
        "payout_source_known",
        "dividend_source IS NOT NULL",
    ),
)

_EXPANSION_PRIMARY: tuple[tuple[str, str, str | None], ...] = (
    (MARKET_CAP_TABLE, "mcap_krx_log", None),
    (FILING_ACTIVITY_TABLE, "ev_filing_burst_60d", None),
    (FILING_ACTIVITY_TABLE, "ev_amendment_ratio_1y", None),
    (FILING_ACTIVITY_TABLE, "own_insider_filing_burst_60d", None),
    (FILING_ACTIVITY_TABLE, "own_major_filing_60d", None),
    (FILING_ACTIVITY_TABLE, "own_amendment_ratio_1y", None),
    (
        PERIODIC_EXTRAS_TABLE,
        "hc_employee_growth_yoy",
        "CAST(trade_date - hc_employee_growth_available_from AS BIGINT)",
    ),
    (
        PERIODIC_EXTRAS_TABLE,
        "hc_revenue_per_employee",
        "CAST(trade_date - hc_productivity_available_from AS BIGINT)",
    ),
    (
        PERIODIC_EXTRAS_TABLE,
        "own_major_stake",
        "CAST(trade_date - own_major_stake_available_from AS BIGINT)",
    ),
    (
        PERIODIC_EXTRAS_TABLE,
        "own_major_stake_chg",
        "CAST(trade_date - own_major_stake_chg_available_from AS BIGINT)",
    ),
)


def _feature_coverage_specs() -> tuple[FeatureCoverageSpec, ...]:
    specs: list[FeatureCoverageSpec] = []
    for mart, entries in ((FIN_SCAN_TABLE, _FIN_PRIMARY), (EVENT_SCAN_TABLE, _EVENT_PRIMARY)):
        for feature, age_expr, precondition, precondition_expr in entries:
            specs.append(
                FeatureCoverageSpec(
                    feature=feature,
                    variant="native_t",
                    source_mart=mart,
                    value_column=feature,
                    age_expr=age_expr,
                    precondition=precondition,
                    precondition_expr=precondition_expr,
                )
            )
            # The extra-delay variant carries the previous session's value, so
            # its true filing age is one session older than the native_t age
            # column on the same row. Rather than pretend the two are the same
            # number, age is left NULL here and read off the native_t row.
            specs.append(
                FeatureCoverageSpec(
                    feature=f"{feature}_lag1",
                    variant="lag1",
                    source_mart=mart,
                    value_column=f"{feature}_lag1",
                    age_expr=None,
                    precondition=precondition,
                    precondition_expr=precondition_expr,
                )
            )
    for feature in _FIN_VALUE_COMPONENTS:
        specs.append(
            FeatureCoverageSpec(
                feature=feature,
                variant="native_t",
                source_mart=FIN_SCAN_TABLE,
                value_column=feature,
                age_expr="value_fin_age_days",
                precondition="none",
                precondition_expr=None,
            )
        )
    for mart, feature, age_expr in _EXPANSION_PRIMARY:
        for variant, value_column in (
            ("native_t", feature),
            ("lag1", f"{feature}_lag1"),
        ):
            specs.append(
                FeatureCoverageSpec(
                    feature=value_column,
                    variant=variant,
                    source_mart=mart,
                    value_column=value_column,
                    age_expr=age_expr if variant == "native_t" else None,
                    precondition="none",
                    precondition_expr=None,
                )
            )
    return tuple(specs)


FEATURE_COVERAGE_SPECS: tuple[FeatureCoverageSpec, ...] = _feature_coverage_specs()


def build_feature_coverage_sql(
    *,
    fin_scan_view: str = FIN_SCAN_TABLE,
    event_scan_view: str = EVENT_SCAN_TABLE,
    specs: tuple[FeatureCoverageSpec, ...] = FEATURE_COVERAGE_SPECS,
) -> str:
    """One row per (feature, variant, market, year) over the two daily marts.

    Both a per-row and a per-date aggregation are needed and they answer
    different questions: ``coverage_ratio`` is "how much of the panel has a
    value", ``min_names_per_date`` is "would the thinnest date in this year
    survive the scan's ``min_names`` filter". A year can look well covered on
    the first and still be unscannable on the second.
    """
    views = {FIN_SCAN_TABLE: fin_scan_view, EVENT_SCAN_TABLE: event_scan_view}
    branches = []
    for spec in specs:
        age = spec.age_expr if spec.age_expr is not None else "NULL"
        precondition_ok = (
            "TRUE" if spec.precondition_expr is None else f"({spec.precondition_expr})"
        )
        branches.append(f"""
        SELECT
            {_sql_str_literal(spec.feature)} AS feature,
            {_sql_str_literal(spec.variant)} AS variant,
            {_sql_str_literal(spec.source_mart)} AS source_mart,
            trade_date, ticker, market,
            ({spec.value_column} IS NOT NULL) AS has_value,
            CAST({age} AS BIGINT) AS age_days,
            {_sql_str_literal(spec.precondition)} AS precondition,
            {precondition_ok} AS precondition_ok
        FROM {views.get(spec.source_mart, spec.source_mart)}""")
    long_sql = "\n        UNION ALL\n".join(branches)
    return f"""
    WITH long AS ({long_sql}
    ),
    per_date AS (
        SELECT
            feature, variant, source_mart, market,
            CAST(EXTRACT(year FROM trade_date) AS INTEGER) AS year,
            trade_date,
            COUNT(*) FILTER (WHERE has_value) AS names
        FROM long
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    date_stats AS (
        SELECT
            feature, variant, source_mart, market, year,
            COUNT(*) AS dates,
            COUNT(*) FILTER (WHERE names > 0) AS dates_with_value,
            MIN(names) AS min_names_per_date,
            CAST(MEDIAN(names) AS DOUBLE) AS median_names_per_date,
            MAX(names) AS max_names_per_date
        FROM per_date
        GROUP BY 1, 2, 3, 4, 5
    ),
    row_stats AS (
        SELECT
            feature, variant, source_mart, market,
            CAST(EXTRACT(year FROM trade_date) AS INTEGER) AS year,
            ANY_VALUE(precondition) AS precondition,
            COUNT(*) AS panel_rows,
            COUNT(*) FILTER (WHERE has_value) AS nonnull_rows,
            COUNT(*) FILTER (WHERE has_value)::DOUBLE / NULLIF(COUNT(*), 0) AS coverage_ratio,
            COUNT(DISTINCT ticker) AS tickers,
            COUNT(DISTINCT ticker) FILTER (WHERE has_value) AS tickers_with_value,
            COUNT(*) FILTER (WHERE precondition_ok) AS precondition_ok_rows,
            COUNT(*) FILTER (WHERE precondition_ok)::DOUBLE
                / NULLIF(COUNT(*), 0) AS precondition_ok_ratio,
            MIN(trade_date) FILTER (WHERE has_value) AS first_value_date,
            MAX(trade_date) FILTER (WHERE has_value) AS last_value_date,
            AVG(age_days) FILTER (WHERE has_value) AS mean_age_days,
            quantile_cont(age_days, 0.95) FILTER (WHERE has_value) AS p95_age_days
        FROM long
        GROUP BY 1, 2, 3, 4, 5
    )
    SELECT
        r.*,
        d.dates,
        d.dates_with_value,
        d.min_names_per_date,
        d.median_names_per_date,
        d.max_names_per_date
    FROM row_stats r
    JOIN date_stats d
      ON d.feature = r.feature AND d.variant = r.variant
     AND d.source_mart = r.source_mart AND d.market = r.market AND d.year = r.year
    ORDER BY r.feature, r.variant, r.market, r.year
    """


def build_event_coverage_sql(*, sue_event_view: str = SUE_EVENT_TABLE) -> str:
    """One row per (event_year, market, reprt_code) over ``fin_sue_event``.

    The per-bucket counts are the event-grain analogue of ``feature_coverage``'s
    ``dates_with_value``: a bucket whose return is NULL for most events cannot
    be scanned no matter how many events the year has, and the late buckets are
    where delisting and short price history bite first.
    """
    bucket_cols = ",\n        ".join(
        f"COUNT(*) FILTER (WHERE {_bucket_col(h1, h2, 'raw')} IS NOT NULL)"
        f" AS {_bucket_col(h1, h2, 'events')}"
        for h1, h2 in EVENT_BUCKETS
    )
    any_bucket_missing = " OR ".join(
        f"{_bucket_col(h1, h2, 'raw')} IS NULL" for h1, h2 in EVENT_BUCKETS
    )
    any_ca_contaminated = " OR ".join(
        f"{_bucket_col(h1, h2, 'ca_contaminated')}" for h1, h2 in EVENT_BUCKETS
    )
    return f"""
    SELECT
        CAST(EXTRACT(year FROM event_formation_date) AS INTEGER) AS event_year,
        market,
        reprt_code,
        COUNT(*) AS events,
        COUNT(DISTINCT ticker) AS tickers,
        COUNT(DISTINCT event_formation_date) AS formation_dates,
        MIN(event_formation_date) AS first_formation_date,
        MAX(event_formation_date) AS last_formation_date,
        COUNT(*) FILTER (WHERE fin_sue IS NOT NULL) AS events_with_sue,
        COUNT(*) FILTER (WHERE fin_sue IS NOT NULL)::DOUBLE
            / NULLIF(COUNT(*), 0) AS sue_ratio,
        COUNT(*) FILTER (WHERE sue_history_count >= {MIN_SUE_HISTORY})
            AS events_with_full_history,
        AVG(sue_history_count) AS mean_sue_history_count,
        COUNT(*) FILTER (WHERE is_primary_constant_sample) AS primary_constant_sample_events,
        COUNT(*) FILTER (WHERE is_primary_constant_sample)::DOUBLE
            / NULLIF(COUNT(*), 0) AS primary_constant_sample_ratio,
        COUNT(*) FILTER (WHERE revision_within_60_sessions) AS revision_contaminated_events,
        COUNT(*) FILTER (WHERE revision_within_60_sessions)::DOUBLE
            / NULLIF(COUNT(*), 0) AS revision_contaminated_ratio,
        COUNT(*) FILTER (WHERE {any_bucket_missing}) AS events_with_missing_bucket,
        COUNT(*) FILTER (WHERE {any_ca_contaminated}) AS events_with_ca_contamination,
        {bucket_cols}
    FROM {sue_event_view}
    WHERE event_formation_date IS NOT NULL
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, 3
    """


def register_receipt_value_pairing_quality_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = RECEIPT_VALUE_PAIRING_QUALITY_TABLE,
    **views: str,
) -> str:
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"{build_receipt_value_pairing_quality_sql(**views)}"
    )
    return view_name


def register_stock_metric_vintage_quality_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = STOCK_METRIC_VINTAGE_QUALITY_TABLE,
    **views: str,
) -> str:
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"{build_stock_metric_vintage_quality_sql(**views)}"
    )
    return view_name


def register_quarterly_metric_quality_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = QUARTERLY_METRIC_QUALITY_TABLE,
    **views: str,
) -> str:
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS {build_quarterly_metric_quality_sql(**views)}"
    )
    return view_name


def register_feature_coverage_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FEATURE_COVERAGE_TABLE,
    **views: str,
) -> str:
    existing = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM duckdb_tables() "
            "UNION SELECT view_name FROM duckdb_views()"
        ).fetchall()
    }
    specs = tuple(spec for spec in FEATURE_COVERAGE_SPECS if spec.source_mart in existing)
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"{build_feature_coverage_sql(specs=specs, **views)}"
    )
    return view_name


def register_event_coverage_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = EVENT_COVERAGE_TABLE,
    **views: str,
) -> str:
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {build_event_coverage_sql(**views)}")
    return view_name
