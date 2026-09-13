# ruff: noqa: E501
"""``feat_fin_risk`` — leverage, life cycle and state transitions (F-4).

Design: ``docs/dev/20260907_additional_feature/03_financial_risk_lifecycle_transition.md``.

Nine families over the canonical metrics that were already collected, so this
mart adds no ingestion at all:

    leverage / risk   fin_debt_to_assets, fin_net_debt_to_mcap,
                      fin_interest_coverage, fin_ext_finance_to_assets
    life cycle        fin_lifecycle_stage, fin_lifecycle_transition
    transition        fin_profit_turn, fin_dividend_initiation,
                      fin_negative_equity_exit

**PIT rule.** Same strict vintage as ``feat_fin_scan_daily``, imported from
``fin_vintage`` rather than restated (F-4.1): a figure is readable from the
session after its filing was received, TTM for flow metrics and the instant
itself for balance-sheet ones, one interval per availability date with same-day
candidates resolved to the latest fiscal period, and a *single* ``fs_basis``
decision per ticker-date — CFS if ``net_income`` has a consolidated value that
day, else OFS — applied to every metric, so a day's bundle can never mix
consolidated and separate figures. ``fin_dividend_initiation`` is the one
column not read from the vintage: DPS comes from
``dart_shareholder_return_raw`` and is exposed on the session *after* the
annual filing's receipt date, the rule ``ev_payout_yield`` uses.

**Valid from.** TTM needs four consecutive quarters and the three transition
families need a *preceding* vintage on top of that, so nothing is populated
before roughly 2016-2017 — the measured start per column is in
``docs/dev/20260907_additional_feature/results/``.

**"Previous vintage" is per metric.** CFO/CFI/CFF each have their own interval
chain, so a stage transition compares this filing's triple with each metric's
own preceding value. Those three preceding values come from the same filing
almost always but not by construction, so ``fin_lifecycle_prev_aligned`` says
whether they did — the limitation is reported, not hidden.

**Survivorship.** ~1,300 delisted corporations have no financials yet (S-1
remainder, ``06`` §3). Decline-stage and distress conclusions are held until
that backfill lands; the mart is built now and the card carries
"생존편향 미해결".

**Grain** ``(trade_date, ticker, market)``, one row per ``dim_stock_pit_daily``
row — the same layering as ``feat_fin_scan_daily``, with broad/tradable
filtering left downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.features.event_scan import (
    COMMON_STOCK_KINDS,
    DPS_ROW_NAME,
    UNSPLIT_STOCK_KINDS,
)
from modeler.etl.features.fin_vintage import (
    BASE_OK_SQL,
    build_metric_intervals_cte,
    build_wide_intervals_asof_join,
    build_wide_intervals_cte,
    select_by_basis,
)
from modeler.etl.mart import materialize, register_mart_view

FIN_RISK_TABLE = "feat_fin_risk"
_CAL_TABLE = "_fin_risk_calendar"

#: Same role as ``FIN_FEATURE_FORMULA_VERSION``: the ratio definitions, the
#: life-cycle mapping, the transition rules and the interest-coverage clip are
#: covered by neither ``config_hash`` (the scan YAML) nor ``phase_b_code_hash``
#: (the analysis modules), so a change here would otherwise produce different
#: numbers under an identical run-spec fingerprint. Bump it whenever any of
#: them changes, and do not reuse an artifact of the same snapshot across a bump.
#:
#:   fin_risk_v1 — the 2026-09 rules as first built (03 §1).
#:   fin_risk_v2 — F-5.0. No formula changed here; the *inputs* did.
#:            interest_paid, investing_cash_flow, financing_cash_flow and
#:            cash_and_cash_equivalents gained XBRL fallback mapping rules, and
#:            the `ifrs_` spelling those rules add is what covers 2015-2018
#:            (DART switched taxonomy prefix around 2019: 364 facts up to 2018
#:            under ifrs-full_Liabilities against 112,827 under
#:            ifrs_Liabilities). Five of the nine families were confined to
#:            2020+ purely by that gap. The values move for every one of them,
#:            so the fingerprint has to move too — a re-run under fin_risk_v1
#:            would otherwise reuse an artifact of an identical run spec while
#:            producing different numbers.
#:   fin_risk_v3 — adds F-5.4's three derived families on the five metrics
#:            F-5.1/F-5.2 mapped (`current_assets`, `current_liabilities`,
#:            `retained_earnings`, `borrowings_short_term`,
#:            `borrowings_long_term`). The nine v2 families are untouched
#:            formula-for-formula, but the mart's schema and SQL text both move,
#:            so the fingerprint has to.
FORMULA_VERSION = "fin_risk_v3"

#: Interest coverage above this is capped. A coverage of 400x and one of 4,000x
#: say the same thing ("no debt service pressure") and the difference is mostly
#: a near-zero denominator, so the tail is compressed rather than left to
#: dominate a winsorize downstream. Upper only — 03 §1.1.
INTEREST_COVERAGE_CAP = 100

#: Fiscal years of zero/absent dividend required before a positive DPS counts
#: as an initiation (03 §1.3). A year missing from the source is not evidence
#: of a zero dividend, so an incomplete history yields NULL, not 0.
DIVIDEND_INITIATION_LOOKBACK_YEARS = 3

#: The annual report. Dividend disclosure is annual-cadence, so unlike the
#: cash-flow metrics this needs no quarter/TTM machinery.
ANNUAL_REPRT_CODE = "11011"

#: Metrics interval-joined onto the daily panel. ``total_assets`` additionally
#: carries ``value_lag_4q`` for the average-assets denominator, exactly as in
#: ``feat_fin_scan_daily``.
_METRICS: tuple[str, ...] = (
    "total_liabilities",
    "total_assets",
    "total_equity",
    "cash_and_cash_equivalents",
    "net_income",
    "operating_income",
    "interest_paid",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    # F-5.4. `revenue` was already canonical; the five below arrived with
    # F-5.1/F-5.2 and are also registered in `financial_quarters.INSTANT_METRICS`
    # -- a metric absent from that set never reaches this mart, whatever
    # `metric_rules` says about it.
    "revenue",
    "current_assets",
    "current_liabilities",
    "retained_earnings",
    "borrowings_short_term",
    "borrowings_long_term",
)

#: Altman (1968) table 1 for public manufacturers, verbatim. The weights are
#: the paper's and are not this repository's to re-fit (the same rule
#: `LIFECYCLE_STAGES` follows for Dickinson).
#:
#: F-5.1 is what makes the full form possible: `03` §3 left `fin_altman_z`
#: conditional on "becoming the full form rather than a reduced one", and
#: retained earnings -- measured at 0.959 coverage from 2015 -- is the term that
#: decides it. All five are now available.
ALTMAN_WEIGHTS: dict[str, float] = {
    "working_capital_to_assets": 1.2,
    "retained_earnings_to_assets": 1.4,
    "ebit_to_assets": 3.3,
    "market_equity_to_liabilities": 0.6,
    "sales_to_assets": 1.0,
}

#: Dickinson (2011) table 1, verbatim. Stage numbering is the paper's:
#: 1 Introduction, 2 Growth, 3 Mature, 4 Shake-out, 5 Decline. Eight sign
#: combinations map onto five stages and the mapping is *fixed* — it is not a
#: modelling choice this repository gets to revisit (03 §1.2).
LIFECYCLE_STAGES: dict[tuple[int, int, int], int] = {
    (-1, -1, +1): 1,  # Introduction
    (+1, -1, +1): 2,  # Growth
    (+1, -1, -1): 3,  # Mature
    (-1, -1, -1): 4,  # Shake-out
    (+1, +1, +1): 4,  # Shake-out
    (+1, +1, -1): 4,  # Shake-out
    (-1, +1, +1): 5,  # Decline
    (-1, +1, -1): 5,  # Decline
}

#: The nine preregistered primaries, in output order (03 §4).
PRIMARY_COLUMNS: tuple[str, ...] = (
    "fin_debt_to_assets",
    "fin_net_debt_to_mcap",
    "fin_interest_coverage",
    "fin_ext_finance_to_assets",
    "fin_lifecycle_stage",
    "fin_lifecycle_transition",
    "fin_profit_turn",
    "fin_dividend_initiation",
    "fin_negative_equity_exit",
    # F-5.4. Registered in a *later* F-HS config than the nine above (03 §3:
    # "F-5의 파생 family는 F-4와 같은 F-HS config에 넣지 않는다"), so these are
    # built and reported now and preregistered separately.
    "fin_current_ratio",
    "fin_borrowings_to_mcap",
    "fin_altman_z",
)
#: Exploratory decomposition and quality flags. Reported, not preregistered.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "fin_lifecycle_transition_up",
    "fin_lifecycle_transition_down",
    "fin_lifecycle_prev_stage",
    "fin_lifecycle_prev_aligned",
    "fin_interest_coverage_capped",
    # F-5.4. `fin_borrowings_total` is the numerator in won, kept so the
    # short + long sum can be audited against the balance sheet -- the
    # reconciliation F-5.1 §6.3 ran on 10,012 corp-years is a claim about this
    # value. The five Altman terms are emitted individually because a composite
    # score is uninterpretable without them, and because `03` §3 asks whether
    # the full form beats the reduced one.
    "fin_borrowings_total",
    "fin_altman_wc_to_assets",
    "fin_altman_re_to_assets",
    "fin_altman_ebit_to_assets",
    "fin_altman_mve_to_liabilities",
    "fin_altman_sales_to_assets",
    "fs_basis_used",
)
#: One availability date per family group, plus its age in days — the same
#: shape ``feat_fin_scan_daily`` emits, and what a card reports staleness from.
AVAILABILITY_GROUPS: tuple[str, ...] = (
    "leverage",
    "net_debt",
    "coverage",
    "ext_finance",
    "lifecycle",
    "profit_turn",
    "negative_equity",
    "dividend",
    "current_ratio",
    "borrowings",
    "altman",
)

_STOCK_KIND_RANK_SQL = (
    "CASE WHEN sr.stock_knd IN ("
    + ", ".join(f"'{k}'" for k in sorted(COMMON_STOCK_KINDS))
    + ") THEN 0 WHEN sr.stock_knd IS NULL OR sr.stock_knd IN ("
    + ", ".join(f"'{k}'" for k in sorted(UNSPLIT_STOCK_KINDS))
    + ") THEN 1 END"
)


def _sign(column: str) -> str:
    """``+1`` / ``-1`` / NULL. An exact zero is not a Dickinson sign.

    The paper's patterns are over strict signs. A TTM cash flow of exactly zero
    is almost always an artifact (a filer reporting a blank as 0), and calling
    it positive would place the firm in a stage the paper never assigns it. Such
    rows are NULL and counted in the report instead.
    """
    return f"CASE WHEN {column} > 0 THEN 1 WHEN {column} < 0 THEN -1 END"


def _lifecycle_case(cfo: str, cfi: str, cff: str, *, alias: str) -> str:
    """The 8-combination mapping as one CASE over the three sign expressions."""
    branches = "\n                 ".join(
        f"WHEN {cfo} = {signs[0]} AND {cfi} = {signs[1]} AND {cff} = {signs[2]} THEN {stage}"
        for signs, stage in LIFECYCLE_STAGES.items()
    )
    return f"CASE {branches}\n            END AS {alias}"


def build_dividend_history_sql(
    *,
    shareholder_return_view: str = "dart_shareholder_return_raw",
    calendar_table: str = _CAL_TABLE,
) -> str:
    """One row per (ticker, fiscal year) with the common-share DPS and its asof.

    Share class matters and ``stock_knd`` is not in the row key, so a filer
    reporting 보통주 and 우선주 separately puts several rows in one (ticker,
    year) group; picking the preferred row would call a preferred-only dividend
    a common-share initiation. Rank 0 is the common share, rank 1 a filer that
    did not split by class (one class, so the figure is the common one), and
    everything else is excluded rather than ranked — the same rule
    ``ev_payout_yield`` applies, built from the same shared constants.

    ``dps_is_zero`` accepts the literal ``-`` the source writes for "no
    dividend": that is an explicit zero, unlike an absent row.
    """
    return f"""
        WITH dividend_rows AS (
            SELECT
                sr.ticker, sr.bsns_year, sr.rcept_no, sr.value_numeric, sr.value_text,
                sr.stock_knd,
                {_STOCK_KIND_RANK_SQL} AS stock_kind_rank,
                CASE WHEN sr.rcept_no ~ '^[0-9]{{14}}$'
                     THEN strptime(left(sr.rcept_no, 8), '%Y%m%d')::DATE END AS disclosed_date
            FROM {shareholder_return_view} sr
            WHERE sr.statement_type = 'dividend' AND sr.reprt_code = '{ANNUAL_REPRT_CODE}'
              AND sr.metric_code = 'thstrm'
              AND sr.row_name = '{DPS_ROW_NAME}'
              AND sr.ticker IS NOT NULL AND sr.ticker <> ''
        ),
        dividend_winners AS (
            SELECT *
            FROM dividend_rows
            WHERE stock_kind_rank IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ticker, bsns_year
                ORDER BY stock_kind_rank ASC,
                         disclosed_date ASC NULLS LAST, rcept_no ASC,
                         -- total order, so the winner cannot depend on scan
                         -- order (10_known_issues.md I12)
                         value_numeric DESC NULLS LAST, stock_knd ASC NULLS LAST,
                         value_text ASC NULLS LAST
            ) = 1
        )
        SELECT
            ticker, bsns_year,
            value_numeric AS dps_raw,
            (value_numeric IS NOT NULL AND value_numeric > 0) AS dps_is_positive,
            (value_numeric = 0 OR value_text = '-') AS dps_is_zero,
            CASE WHEN disclosed_date IS NOT NULL
                 THEN (SELECT MIN(c.d) FROM {calendar_table} c WHERE c.d > disclosed_date)
            END AS dps_available_from
        FROM dividend_winners
    """


def build_dividend_initiation_sql(
    *,
    shareholder_return_view: str = "dart_shareholder_return_raw",
    calendar_table: str = _CAL_TABLE,
) -> str:
    """One interval per (ticker, filing) carrying ``fin_dividend_initiation``.

    NULL — not 0 — when any of the ``DIVIDEND_INITIATION_LOOKBACK_YEARS``
    preceding fiscal years is absent from the source: a missing year is missing
    evidence, and treating it as a zero dividend would manufacture initiations
    for every company whose early filings were never collected.
    """
    lookback = DIVIDEND_INITIATION_LOOKBACK_YEARS
    prior_joins = "\n".join(
        f"""            LEFT JOIN history p{n}
              ON p{n}.ticker = h.ticker AND p{n}.bsns_year = h.bsns_year - {n}"""
        for n in range(1, lookback + 1)
    )
    prior_known = " AND ".join(f"p{n}.ticker IS NOT NULL" for n in range(1, lookback + 1))
    prior_zero = " AND ".join(f"COALESCE(p{n}.dps_is_zero, FALSE)" for n in range(1, lookback + 1))
    return f"""
        WITH history AS (
            {build_dividend_history_sql(
                shareholder_return_view=shareholder_return_view,
                calendar_table=calendar_table,
            )}
        ),
        judged AS (
            SELECT
                h.ticker, h.bsns_year, h.dps_available_from,
                CASE WHEN {prior_known}
                     THEN CASE WHEN h.dps_is_positive AND {prior_zero} THEN 1 ELSE 0 END
                END AS fin_dividend_initiation
            FROM history h
{prior_joins}
        ),
        positioned AS (
            SELECT *
            FROM judged
            WHERE dps_available_from IS NOT NULL
            -- One position per availability date, resolved to the latest fiscal
            -- year: a filer catching up files several years at once, and the
            -- interval chain below must be built from one row per date.
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ticker, dps_available_from ORDER BY bsns_year DESC
            ) = 1
        )
        SELECT
            ticker, bsns_year, fin_dividend_initiation,
            dps_available_from,
            LEAD(dps_available_from) OVER (
                PARTITION BY ticker ORDER BY dps_available_from
            ) AS next_dps_available_from
        FROM positioned
    """


def build_fin_risk_sql(
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    shareholder_return_view: str = "dart_shareholder_return_raw",
    calendar_table: str = _CAL_TABLE,
) -> str:
    """SQL producing ``feat_fin_risk``.

    Args:
        pit_view: A0's PIT shares/market-cap mart.
        quality_view: A0's price-quality mart.
        vintage_view: B-3's quarterly metric vintage mart.
        shareholder_return_view: Raw ``alotMatter`` rows, for DPS.
        calendar_table: KRX session table, for the dividend exposure date.
            Register it with :func:`register_fin_risk_calendar` first.
    """
    projections = ",\n            ".join(
        f"iv.v_{metric}_{basis}, iv.pv_{metric}_{basis}, "
        f"iv.a_{metric}_{basis}, iv.pa_{metric}_{basis}"
        for metric in _METRICS
        for basis in ("cfs", "ofs")
    )
    selected = ",\n            ".join(select_by_basis(metric) for metric in _METRICS)
    prev_selected = ",\n            ".join(
        f"CASE WHEN v_net_income_cfs IS NOT NULL THEN pv_{metric}_cfs\n"
        f"                 ELSE pv_{metric}_ofs END AS {metric}_prev"
        for metric in _METRICS
    )
    availability = ",\n            ".join(
        f"CASE WHEN v_net_income_cfs IS NOT NULL THEN a_{metric}_cfs\n"
        f"                 ELSE a_{metric}_ofs END AS {metric}_asof"
        for metric in _METRICS
    )
    prev_availability = ",\n            ".join(
        f"CASE WHEN v_net_income_cfs IS NOT NULL THEN pa_{metric}_cfs\n"
        f"                 ELSE pa_{metric}_ofs END AS {metric}_prev_asof"
        for metric in ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow")
    )
    lag1 = ",\n        ".join(
        f"LAG({column}) OVER (PARTITION BY ticker, market ORDER BY trade_date) AS {column}_lag1"
        for column in PRIMARY_COLUMNS
    )
    asof_outputs = ",\n        ".join(
        f"{group}_available_from,\n        (trade_date - {group}_available_from) "
        f"AS {group}_fin_age_days"
        for group in AVAILABILITY_GROUPS
    )
    # Altman's weights, written into the SQL text so a changed weight changes
    # the mart's cache key rather than silently reusing the old score.
    altman_sum = "\n                        + ".join(
        f"{ALTMAN_WEIGHTS[term]} * fin_altman_{alias}"
        for term, alias in (
            ("working_capital_to_assets", "wc_to_assets"),
            ("retained_earnings_to_assets", "re_to_assets"),
            ("ebit_to_assets", "ebit_to_assets"),
            ("market_equity_to_liabilities", "mve_to_liabilities"),
            ("sales_to_assets", "sales_to_assets"),
        )
    )
    dividend_sql = build_dividend_initiation_sql(
        shareholder_return_view=shareholder_return_view,
        calendar_table=calendar_table,
    )

    return f"""
    -- {FORMULA_VERSION}
    WITH {build_metric_intervals_cte(vintage_view, _METRICS, with_previous=True)},
    {build_wide_intervals_cte(_METRICS, with_previous=True, lag_4q_metrics=("total_assets",))},
    dividend_intervals AS ({dividend_sql}),
    panel AS (
        SELECT
            pit.trade_date, pit.ticker, pit.market,
            pit.market_cap_pit, pit.issued_shares_pit,
            pit.shares_is_available, pit.shares_invalid_flag, pit.shares_available_from,
            q.is_halted, q.valid_session_idx
        FROM {pit_view} pit
        LEFT JOIN {quality_view} q USING (trade_date, ticker, market)
    ),
    joined AS (
        SELECT panel.*,
            {projections},
            iv.l4_total_assets_cfs AS v_total_assets_cfs_lag4q,
            iv.l4_total_assets_ofs AS v_total_assets_ofs_lag4q,
            div.fin_dividend_initiation AS dividend_initiation_raw,
            div.dps_available_from AS dividend_available_from
        FROM panel
        {build_wide_intervals_asof_join()}
        LEFT JOIN dividend_intervals div
          ON div.ticker = panel.ticker
         AND div.dps_available_from <= panel.trade_date
         AND (div.next_dps_available_from IS NULL
              OR panel.trade_date < div.next_dps_available_from)
    ),
    resolved AS (
        SELECT
            trade_date, ticker, market,
            market_cap_pit, shares_is_available, shares_invalid_flag,
            is_halted, valid_session_idx,
            dividend_initiation_raw, dividend_available_from,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN 'CFS'
                 WHEN v_net_income_ofs IS NOT NULL THEN 'OFS' END AS fs_basis_used,
            {selected},
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_total_assets_cfs_lag4q
                 ELSE v_total_assets_ofs_lag4q END AS total_assets_lag4q_selected,
            {prev_selected},
            {availability},
            {prev_availability}
        FROM joined
    ),
    scored AS (
        SELECT
            *,
            {BASE_OK_SQL} AS base_ok,
            CASE WHEN total_assets_selected > 0 AND total_assets_lag4q_selected > 0
                 THEN (total_assets_selected + total_assets_lag4q_selected) / 2 END AS avg_assets,
            {_sign("operating_cash_flow_selected")} AS cfo_sign,
            {_sign("investing_cash_flow_selected")} AS cfi_sign,
            {_sign("financing_cash_flow_selected")} AS cff_sign,
            {_sign("operating_cash_flow_prev")} AS cfo_sign_prev,
            {_sign("investing_cash_flow_prev")} AS cfi_sign_prev,
            {_sign("financing_cash_flow_prev")} AS cff_sign_prev,
            greatest(total_liabilities_asof, total_assets_asof) AS leverage_available_from,
            greatest(total_liabilities_asof, cash_and_cash_equivalents_asof)
                AS net_debt_available_from,
            greatest(operating_income_asof, interest_paid_asof) AS coverage_available_from,
            greatest(financing_cash_flow_asof, total_assets_asof) AS ext_finance_available_from,
            greatest(operating_cash_flow_asof, investing_cash_flow_asof,
                     financing_cash_flow_asof) AS lifecycle_available_from,
            net_income_asof AS profit_turn_available_from,
            total_equity_asof AS negative_equity_available_from,
            greatest(current_assets_asof, current_liabilities_asof)
                AS current_ratio_available_from,
            greatest(borrowings_short_term_asof, borrowings_long_term_asof)
                AS borrowings_available_from,
            greatest(current_assets_asof, current_liabilities_asof, retained_earnings_asof,
                     operating_income_asof, revenue_asof, total_assets_asof,
                     total_liabilities_asof) AS altman_available_from
        FROM resolved
    ),
    staged AS (
        SELECT
            *,
            {_lifecycle_case("cfo_sign", "cfi_sign", "cff_sign", alias="lifecycle_stage_raw")},
            {_lifecycle_case(
                "cfo_sign_prev", "cfi_sign_prev", "cff_sign_prev", alias="lifecycle_stage_prev"
            )},
            (operating_cash_flow_prev_asof IS NOT NULL
             AND investing_cash_flow_prev_asof = operating_cash_flow_prev_asof
             AND financing_cash_flow_prev_asof = operating_cash_flow_prev_asof
            ) AS fin_lifecycle_prev_aligned
        FROM scored
    ),
    ratios AS (
        SELECT
            trade_date, ticker, market, fs_basis_used, fin_lifecycle_prev_aligned,
            leverage_available_from, net_debt_available_from, coverage_available_from,
            ext_finance_available_from, lifecycle_available_from,
            profit_turn_available_from, negative_equity_available_from,
            dividend_available_from, current_ratio_available_from,
            borrowings_available_from, altman_available_from,
            -- ``base_ok`` gates only the market-cap-dependent ratio, which is
            -- what ``feat_fin_scan_daily`` does: ``fin_book_to_market`` and the
            -- other price-scaled columns carry it, ``fin_gross_profitability``
            -- and the other purely accounting ones do not. Tradability is the
            -- downstream sample's job, and a leverage ratio is a fact about the
            -- filing regardless of whether the stock traded that session.
            CASE WHEN total_assets_selected > 0
                 THEN total_liabilities_selected / total_assets_selected
            END AS fin_debt_to_assets,
            CASE WHEN base_ok AND total_liabilities_selected IS NOT NULL
                      AND cash_and_cash_equivalents_selected IS NOT NULL
                 THEN (total_liabilities_selected - cash_and_cash_equivalents_selected)
                      / market_cap_pit
            END AS fin_net_debt_to_mcap,
            -- The CASE guard is not decoration: DuckDB's LEAST *skips* NULL
            -- arguments, so a bare LEAST(NULL, 100) returns 100 and every
            -- company with no interest figure would be pinned to the healthiest
            -- end of the cross-section (the fin_v1 failure, 10_known_issues I1).
            CASE WHEN interest_paid_selected > 0 AND operating_income_selected IS NOT NULL
                 THEN LEAST(operating_income_selected / interest_paid_selected,
                            {INTEREST_COVERAGE_CAP})
            END AS fin_interest_coverage,
            -- NULL, not FALSE, where there is no coverage value: "was it
            -- capped" has no answer for a company with no interest line, and
            -- the diagnostic this feeds is "of the values we do have, what
            -- share hit the cap".
            CASE WHEN interest_paid_selected > 0 AND operating_income_selected IS NOT NULL
                 THEN operating_income_selected / interest_paid_selected
                      > {INTEREST_COVERAGE_CAP}
            END AS fin_interest_coverage_capped,
            CASE WHEN avg_assets > 0 AND financing_cash_flow_selected IS NOT NULL
                 THEN financing_cash_flow_selected / avg_assets
            END AS fin_ext_finance_to_assets,
            lifecycle_stage_raw AS fin_lifecycle_stage,
            lifecycle_stage_prev AS fin_lifecycle_prev_stage,
            CASE WHEN lifecycle_stage_raw IS NOT NULL AND lifecycle_stage_prev IS NOT NULL
                 THEN CASE WHEN lifecycle_stage_raw <> lifecycle_stage_prev THEN 1 ELSE 0 END
            END AS fin_lifecycle_transition,
            CASE WHEN lifecycle_stage_raw IS NOT NULL AND lifecycle_stage_prev IS NOT NULL
                 THEN CASE WHEN lifecycle_stage_raw > lifecycle_stage_prev THEN 1 ELSE 0 END
            END AS fin_lifecycle_transition_up,
            CASE WHEN lifecycle_stage_raw IS NOT NULL AND lifecycle_stage_prev IS NOT NULL
                 THEN CASE WHEN lifecycle_stage_raw < lifecycle_stage_prev THEN 1 ELSE 0 END
            END AS fin_lifecycle_transition_down,
            CASE WHEN net_income_selected IS NOT NULL AND net_income_prev IS NOT NULL
                 THEN CASE WHEN net_income_prev < 0 AND net_income_selected > 0 THEN 1
                           WHEN net_income_prev > 0 AND net_income_selected < 0 THEN -1
                           ELSE 0 END
            END AS fin_profit_turn,
            dividend_initiation_raw AS fin_dividend_initiation,
            CASE WHEN total_equity_selected IS NOT NULL AND total_equity_prev IS NOT NULL
                 THEN CASE WHEN total_equity_prev <= 0 AND total_equity_selected > 0 THEN 1
                           ELSE 0 END
            END AS fin_negative_equity_exit,
            -- F-5.4 ---------------------------------------------------------
            CASE WHEN current_liabilities_selected > 0 AND current_assets_selected IS NOT NULL
                 THEN current_assets_selected / current_liabilities_selected
            END AS fin_current_ratio,
            -- Short + long only. The current portion of long-term borrowings
            -- never reached 0.5 coverage (max 0.466) so it is not a required
            -- component; omitting it understates interest-bearing debt by a
            -- median 10.2% and a p90 59.7%, which the card states. The sum was
            -- reconciled first: across 10,012 corp-years with all three parts,
            -- short + long never exceeds total liabilities (F-5.1 §6.3).
            CASE WHEN borrowings_short_term_selected IS NOT NULL
                      AND borrowings_long_term_selected IS NOT NULL
                 THEN borrowings_short_term_selected + borrowings_long_term_selected
            END AS fin_borrowings_total,
            CASE WHEN base_ok
                      AND borrowings_short_term_selected IS NOT NULL
                      AND borrowings_long_term_selected IS NOT NULL
                 THEN (borrowings_short_term_selected + borrowings_long_term_selected)
                      / market_cap_pit
            END AS fin_borrowings_to_mcap,
            -- Altman's five terms, each emitted so the composite is readable.
            CASE WHEN total_assets_selected > 0
                      AND current_assets_selected IS NOT NULL
                      AND current_liabilities_selected IS NOT NULL
                 THEN (current_assets_selected - current_liabilities_selected)
                      / total_assets_selected
            END AS fin_altman_wc_to_assets,
            CASE WHEN total_assets_selected > 0 AND retained_earnings_selected IS NOT NULL
                 THEN retained_earnings_selected / total_assets_selected
            END AS fin_altman_re_to_assets,
            -- EBIT is approximated by operating income. Korean filers report
            -- 영업이익 as a standard line and a true EBIT would need interest and
            -- tax added back from the income statement, which canonical does not
            -- carry -- stated rather than silently equated.
            CASE WHEN total_assets_selected > 0 AND operating_income_selected IS NOT NULL
                 THEN operating_income_selected / total_assets_selected
            END AS fin_altman_ebit_to_assets,
            CASE WHEN base_ok AND total_liabilities_selected > 0
                 THEN market_cap_pit / total_liabilities_selected
            END AS fin_altman_mve_to_liabilities,
            CASE WHEN total_assets_selected > 0 AND revenue_selected IS NOT NULL
                 THEN revenue_selected / total_assets_selected
            END AS fin_altman_sales_to_assets
        FROM staged
    )
    ,
    with_altman AS (
        SELECT
            *,
            -- All five terms or nothing. A partial sum would put a company
            -- missing its sales line into the same distribution as one that
            -- scored low on five real terms, and Altman's cut-offs are
            -- statements about the complete score.
            CASE WHEN fin_altman_wc_to_assets IS NOT NULL
                      AND fin_altman_re_to_assets IS NOT NULL
                      AND fin_altman_ebit_to_assets IS NOT NULL
                      AND fin_altman_mve_to_liabilities IS NOT NULL
                      AND fin_altman_sales_to_assets IS NOT NULL
                 THEN {altman_sum}
            END AS fin_altman_z
        FROM ratios
    )
    SELECT
        trade_date, ticker, market,
        {", ".join(PRIMARY_COLUMNS)},
        {lag1},
        fin_lifecycle_transition_up, fin_lifecycle_transition_down,
        fin_lifecycle_prev_stage, fin_lifecycle_prev_aligned,
        fin_interest_coverage_capped,
        fin_borrowings_total,
        fin_altman_wc_to_assets, fin_altman_re_to_assets, fin_altman_ebit_to_assets,
        fin_altman_mve_to_liabilities, fin_altman_sales_to_assets,
        fs_basis_used,
        {asof_outputs}
    FROM with_altman
    """


def register_fin_risk_calendar(
    con: duckdb.DuckDBPyConnection,
    trading_days: Sequence[date],
    *,
    table: str = _CAL_TABLE,
) -> str:
    """(Re)create the KRX session table the dividend exposure date reads."""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} (d DATE)")
    if trading_days:
        con.executemany(f"INSERT INTO {table} VALUES (?)", [(d,) for d in trading_days])
    return table


def register_fin_risk_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FIN_RISK_TABLE,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    shareholder_return_view: str = "dart_shareholder_return_raw",
    calendar_table: str = _CAL_TABLE,
) -> str:
    """Register a DuckDB view over the SQL above (no parquet — tests/parity)."""
    sql = build_fin_risk_sql(
        pit_view=pit_view,
        quality_view=quality_view,
        vintage_view=vintage_view,
        shareholder_return_view=shareholder_return_view,
        calendar_table=calendar_table,
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_fin_risk(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    trading_days: Sequence[date],
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    shareholder_return_view: str = "dart_shareholder_return_raw",
    force: bool = False,
) -> str:
    """Build + register ``feat_fin_risk`` as a cached parquet mart.

    Requires ``pit_view``, ``quality_view`` (A0), ``vintage_view`` (B-3) and
    ``shareholder_return_view`` (raw) registered on ``con``. ``trading_days``
    is the KRX session calendar the dividend exposure date is resolved against.
    """
    register_fin_risk_calendar(con, trading_days)
    materialize(
        con,
        config,
        FIN_RISK_TABLE,
        build_fin_risk_sql(
            pit_view=pit_view,
            quality_view=quality_view,
            vintage_view=vintage_view,
            shareholder_return_view=shareholder_return_view,
        ),
        force=force,
    )
    return register_mart_view(con, config, FIN_RISK_TABLE)
