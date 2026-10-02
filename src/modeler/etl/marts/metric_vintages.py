"""``stock_metric_vintage_fact`` — Phase B B-2 (04_specific_plan_B.md §3.1-§3.5).

Preserves ``rcept_no`` as its own grain dimension — unlike the legacy
``stock_metric_fact`` (``research/etl/marts/metrics_normalize.py``), which
collapses to one winner per ``(ticker, metric_code, bsns_year, reprt_code)``
and does not keep ``rcept_no`` as an independent column — so availability and
revision lineage per captured filing is never lost.

This module intentionally does **not** share code with ``metrics_normalize.py``:
that module's output is frozen against golden-parity fixtures for model
regression checks (§3.1 "기존 stock_metric_fact ... 는 모델 회귀 검증을 위해
그대로 둔다"). The metric *mapping rules* are still reused from
``collector.kr.definitions.metric_rules`` — only the candidate-matching SQL
that projects ``rcept_no``/availability/lineage is written independently here.

Grain: ``(ticker, metric_code, statement_period_end, fs_basis, rcept_no)``.

What this mart does NOT do (§1.1 condition 3, §3.5):
    - It never claims ``complete_original_and_revisions`` — proving no further
      revision was ever filed needs the full receipt list scoped to a wide
      enough date range, which is a data-completeness question for B-1's
      backfill, not something derivable from whatever raw happens to hold now.
    - It never treats the numerically-smallest captured ``rcept_no`` as
      "original" without ``dart_filing_receipt_raw`` confirmation (§3.5: "raw에
      우연히 남은 최소 rcept_no를 original로 간주하지 않는다").
    - It never applies ``dart_corp_master.is_active=true`` (§3.4) — no survivor
      bias from backcasting current active status.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb
from collector.kr.shared import (
    MetricMappingRule,
    default_metric_catalog,
    default_metric_mapping_rules,
)

from modeler.etl.config import LakeConfig
from modeler.etl.lake import _sql_str_literal
from modeler.etl.mart import MartPlan, materialize, register_mart_view

SMVF_TABLE = "stock_metric_vintage_fact"
_CAL_TABLE = "_metric_vintage_calendar"

# §1.1 condition 3 fallback for filings whose rcept_no cannot be parsed at all
# (empty or not 14 digits) — matches the legacy feat_fin_pit.py PIT lag
# (period_end + 90d annual / 45d quarterly), not a session-adjusted date.
ANNUAL_FALLBACK_DAYS = 90
QUARTERLY_FALLBACK_DAYS = 45

# Exact-match only (phase_b.receipt_value_pairing_error_tolerance: 0, frozen in
# horizon_scan_config.yaml) — kept as a plain default here rather than an import
# so this raw-ETL mart does not depend on the analysis-config module.
DEFAULT_PAIRING_TOLERANCE = 0.0

_PERIOD_TYPE_SQL = (
    "CASE {col} WHEN '11013' THEN 'q1' WHEN '11012' THEN 'half' "
    "WHEN '11014' THEN 'q3' WHEN '11011' THEN 'annual' ELSE 'unknown' END"
)


def _period_type_expr(col: str) -> str:
    return _PERIOD_TYPE_SQL.format(col=col)


def _calendar_period_end_expr(reprt_col: str, year_col: str) -> str:
    return (
        f"CASE {reprt_col} "
        f"WHEN '11013' THEN make_date({year_col}, 3, 31) "
        f"WHEN '11012' THEN make_date({year_col}, 6, 30) "
        f"WHEN '11014' THEN make_date({year_col}, 9, 30) "
        f"WHEN '11011' THEN make_date({year_col}, 12, 31) "
        f"ELSE NULL END"
    )


# Same XBRL-dimension tie-break used by metrics_normalize.py's winner
# selection — kept as an independent literal (see module docstring) so this
# mart never silently drifts if the legacy file's ranking changes.
_XBRL_RANK_SQL = (
    "COALESCE(json_array_length(dimensions), 0) * 10"
    " + (CASE WHEN dimensions LIKE '%ConsolidatedMember%' THEN -5"
    " WHEN dimensions LIKE '%SeparateMember%' THEN 5 ELSE 0 END)"
    " + (CASE WHEN dimensions LIKE '%ReportedAmountMember%' THEN 1 ELSE 0 END)"
    " + (CASE WHEN dimensions LIKE '%OperatingSegmentsMember%' THEN 3 ELSE 0 END)"
)

# Duration of an XBRL context in days; 0 for instant facts, which have no
# period_start. Only meaningful on ``xbrl_scoped``, which projects the column.
_XBRL_DURATION_DAYS = "COALESCE(duration_days, 0)"


# The financial-statement rule predicate, shared by the ``candidates`` join and by
# the staged plan's key stage so the two cannot drift apart. The indentation is the
# one it has inside ``candidates`` (the text is part of the v1 ``sql_hash``).
_FIN_RULE_MATCH_SQL = """r.source_table = 'dart_financial_statement_raw'
         AND (r.fs_div = '' OR f.fs_div = r.fs_div)
         AND (r.sj_div = '' OR f.sj_div = r.sj_div)
         AND (r.account_id = '' OR f.account_id = r.account_id)
         AND (r.account_nm = '' OR f.account_nm = r.account_nm)"""


def _rules_relation_sql(rules: list[MetricMappingRule], unit_by_code: dict[str, str]) -> str:
    """Build a ``(VALUES ...) AS rules(...)`` relation from the code rule list."""
    cols = [
        "rule_code",
        "metric_code",
        "source_table",
        "value_selector",
        "priority",
        "statement_type",
        "fs_div",
        "sj_div",
        "account_id",
        "account_nm",
        "row_name",
        "stock_knd",
        "dim1",
        "dim2",
        "dim3",
        "metric_code_match",
        "unit",
    ]
    rows: list[str] = []
    for r in rules:
        values = [
            _sql_str_literal(r.rule_code),
            _sql_str_literal(r.metric_code),
            _sql_str_literal(r.source_table),
            _sql_str_literal(r.value_selector),
            str(r.priority),
            _sql_str_literal(r.statement_type),
            _sql_str_literal(r.fs_div),
            _sql_str_literal(r.sj_div),
            _sql_str_literal(r.account_id),
            _sql_str_literal(r.account_nm),
            _sql_str_literal(r.row_name),
            _sql_str_literal(r.stock_knd),
            _sql_str_literal(r.dim1),
            _sql_str_literal(r.dim2),
            _sql_str_literal(r.dim3),
            _sql_str_literal(r.metric_code_match),
            _sql_str_literal(unit_by_code.get(r.metric_code, "")),
        ]
        rows.append("(" + ", ".join(values) + ")")
    col_list = ", ".join(cols)
    values_list = ",\n            ".join(rows)
    return f"(VALUES\n            {values_list}\n        ) AS rules({col_list})"


#: ``v1`` is today's SQL and is what every frozen run was written under. ``v2``
#: changes what a few rows *mean* (see :func:`build_stock_metric_vintage_fact_sql`);
#: it is what the serving builder selects. The two are cache-distinct: ``v2``
#: has its own ``sql_hash`` and records ``semantics_version`` in the metadata.
SEMANTICS_V1 = "v1"
SEMANTICS_V2 = "v2"
SEMANTICS_VERSIONS = (SEMANTICS_V1, SEMANTICS_V2)

#: Execution plans. ``single`` is one statement (today). ``staged`` computes the
#: same rows in six steps, each written to parquet and re-read, because the
#: single statement spills over 160GB on the server (see ``_staged_plan``).
PLAN_SINGLE = "single"
PLAN_STAGED = "staged"
PLANS = (PLAN_SINGLE, PLAN_STAGED)

#: Availability source written when the receipt date is known but the next
#: session is not in the calendar (``v2`` only).
AVAILABILITY_BEYOND_CALENDAR = "receipt_beyond_calendar"

_REPORT_ORDER_SQL = (
    "CASE {col} WHEN '11013' THEN 1 WHEN '11012' THEN 2 "
    "WHEN '11014' THEN 3 WHEN '11011' THEN 4 ELSE 5 END"
)


def first_report_order_sql(col: str) -> str:
    """Order of the four periodic reports within a fiscal year (Q1, half, Q3, annual).

    Used by the ``v2`` tie rules: when several reports carry the same period's
    value, the one that *first published* it wins -- lowest ``bsns_year``, then
    the earliest report of that year. It is a rank over the report kind, not a
    raw id, so it does not depend on load order.
    """
    return _REPORT_ORDER_SQL.format(col=col)


def _availability_select(semantics: str) -> str:
    """The ``available_from`` / ``availability_source`` pair of the final SELECT.

    ``v1`` falls back to ``period_end + 45/90d`` whenever the next-session lookup
    returned NULL -- which is also what happens to a receipt filed on the last day
    of the calendar. The fallback then lands *before* the receipt (a 2025 annual
    report received 2026-10-01 became available 2026-03-31; 24,572 rows over 240
    simulated K days), and ``availability_source`` still said ``rcept_no``.
    ``v2`` uses the fallback only when the receipt date itself is missing or
    unparseable; a parseable receipt whose next session is not in the calendar
    gets ``available_from = NULL`` and a source of its own.
    """
    fallback = (
        "w.statement_period_end\n"
        "                + CASE WHEN w.period_type = 'annual' "
        f"THEN INTERVAL '{ANNUAL_FALLBACK_DAYS} days'\n"
        f"                       ELSE INTERVAL '{QUARTERLY_FALLBACK_DAYS} days' END"
    )
    if semantics == SEMANTICS_V1:
        return f"""        CAST(
            COALESCE(
                fl.available_from_from_receipt,
                {fallback}
            ) AS DATE
        ) AS available_from,
        CASE WHEN fl.has_parsed_receipt_date THEN 'rcept_no' ELSE 'synthetic_fallback' END
            AS availability_source,"""
    return f"""        CAST(
            CASE WHEN fl.has_parsed_receipt_date
                 THEN fl.available_from_from_receipt
                 ELSE {fallback}
            END AS DATE
        ) AS available_from,
        CASE
            WHEN fl.has_parsed_receipt_date
            THEN CASE WHEN fl.available_from_from_receipt IS NULL
                      THEN '{AVAILABILITY_BEYOND_CALENDAR}' ELSE 'rcept_no' END
            ELSE 'synthetic_fallback'
        END AS availability_source,"""


def _winner_tie_break(semantics: str) -> str:
    """Extra ``ORDER BY`` terms of the ``winners`` window.

    ``v1`` ends at ``source_key``. When one ``rcept_no`` sits in the raw tables
    under two ``(bsns_year, reprt_code)`` combinations (``20200515000226`` in
    ``dart_share_count_raw``) every key ties and the winner depends on the
    execution plan. ``v2`` lets the earliest report win: ``bsns_year`` and then
    report order, ascending.
    """
    if semantics == SEMANTICS_V1:
        return ""
    return f",\n            bsns_year ASC, {first_report_order_sql('reprt_code')} ASC"


def _vintage_sql_parts(
    *,
    financial_view: str,
    share_count_view: str,
    shareholder_return_view: str,
    xbrl_view: str,
    filing_receipt_view: str,
    corp_view: str,
    calendar_table: str,
    pairing_tolerance: float,
    semantics: str,
    xbrl_keys_source: str | None = None,
    xbrl_pairing_source: str = "xbrl_scoped",
    xbrl_candidate_source: str = "xbrl_scoped",
    winners_source: str = "winners",
) -> tuple[dict[str, str], str]:
    """The named CTE texts and the final SELECT of ``stock_metric_vintage_fact``.

    One source for both execution plans: the single statement joins every CTE
    in order, the staged plan builds each stage from the same texts with a few
    relation names swapped (``xbrl_*_source``, ``winners_source``). With the
    defaults and ``semantics="v1"`` the joined text is byte-identical to the
    statement every existing mart was written under.
    """
    if semantics not in SEMANTICS_VERSIONS:
        raise ValueError(f"unknown stock_metric_vintage_fact semantics {semantics!r}")
    if xbrl_keys_source is None:
        xbrl_keys_source = xbrl_view
    rules = default_metric_mapping_rules()
    unit_by_code = {entry.metric_code: entry.unit for entry in default_metric_catalog()}
    rules_rel = _rules_relation_sql(rules, unit_by_code)

    period_type_fin = _period_type_expr("f.reprt_code")
    period_type_sc = _period_type_expr("s.reprt_code")
    period_type_sr = _period_type_expr("sr.reprt_code")
    period_type_xf = _period_type_expr("x.reprt_code")
    winner_tie_break = _winner_tie_break(semantics)
    availability_select = _availability_select(semantics)

    parts: dict[str, str] = {}
    parts["corp"] = f"""corp AS (
        -- §3.4: no is_active filter — active-status backcast is a survivor bias.
        SELECT ticker, market, corp_code
        FROM {corp_view}
        WHERE ticker IS NOT NULL AND ticker <> ''
          AND market IS NOT NULL
    )"""
    parts["rule_rel"] = f"""rule_rel AS (SELECT * FROM {rules_rel})"""
    parts["filing_keys"] = f"""filing_keys AS (
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no
        FROM {financial_view}
        UNION
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {share_count_view}
        UNION
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {shareholder_return_view}
        UNION
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {xbrl_keys_source}
    )"""
    parts["xbrl_scoped"] = f"""xbrl_scoped AS (
        -- Every XBRL fact plus the two context attributes that decide whether
        -- it belongs to *this filing's own* statement rather than to one of the
        -- comparative years or the other consolidation basis printed alongside
        -- it. Three places below need them, and all three used to ignore them
        -- (08 §4.3.2).
        SELECT
            corp_code, ticker, bsns_year, reprt_code, rcept_no,
            concept_id, concept_name, label_ko, context_id, dimensions,
            value_numeric,
            COALESCE(instant_date, period_end) AS period_end_effective,
            date_diff('day', period_start, period_end) AS duration_days,
            CASE
                WHEN dimensions LIKE '%ConsolidatedMember%' THEN 'CFS'
                WHEN dimensions LIKE '%SeparateMember%' THEN 'OFS'
            END AS xbrl_fs_basis
        FROM {xbrl_view}
    )"""
    parts["xbrl_period_by_filing"] = """xbrl_period_by_filing AS (
        -- §3.3 priority 1: the filing's OWN period end, taken from its XBRL
        -- contexts. A periodic report's XBRL carries the current period *and*
        -- one or two comparative years, so the current period is the LATEST
        -- context, not the earliest — MIN() here put a FY2024 annual filing's
        -- period end on 2022-12-31 for 64,688 of its 67,276 rows, and
        -- statement_period_end is this mart's grain (08 §4.3.2).
        --
        -- Contexts dated after the receipt itself are dropped: a filing does
        -- not report on a period that had not ended when it was submitted, so
        -- such a context is forward-looking, not the statement period.
        SELECT corp_code, bsns_year, reprt_code, rcept_no,
               MAX(period_end_effective) AS xbrl_period_end
        FROM xbrl_scoped
        WHERE period_end_effective IS NOT NULL
          AND (
            NOT rcept_no ~ '^[0-9]{14}$'
            OR period_end_effective <= strptime(left(rcept_no, 8), '%Y%m%d')::DATE
          )
        GROUP BY corp_code, bsns_year, reprt_code, rcept_no
    )"""
    parts["xbrl_pairing"] = f"""xbrl_pairing AS (
        -- §1.2 receipt_value_pairing: the same receipt's XBRL value for the
        -- same concept, period, and consolidation basis. Matching on concept
        -- alone paired every OFS statement row against the consolidated
        -- context (SeparateMember always loses the dimension tie-break) and let
        -- an arbitrary comparative year win among the rest — which is why
        -- value_mismatch_ratio read 0.51-0.97 against a frozen tolerance of 0.
        --
        -- ``duration_pref`` exists because an interim filing prints both the
        -- 3-month and the year-to-date duration ending on the same date, and
        -- which one the financial-statement API's thstrm_amount equals depends
        -- on the statement: IS/CIS report the 3-month figure, CF reports the
        -- cumulative one (the same split fin_quarterly_metric_vintage encodes
        -- as direct_interim vs cumulative_reported). Emitting one winner per
        -- preference lets the join pick by ``sj_div`` instead of guessing.
        SELECT
            corp_code, bsns_year, reprt_code, rcept_no, concept_id,
            xbrl_fs_basis, period_end_effective, duration_pref,
            value_numeric, context_id AS pairing_xbrl_source_key
        FROM {xbrl_pairing_source}
        CROSS JOIN (VALUES ('shortest'), ('longest')) AS p(duration_pref)
        WHERE xbrl_fs_basis IS NOT NULL AND period_end_effective IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY corp_code, bsns_year, reprt_code, rcept_no,
                         concept_id, xbrl_fs_basis, period_end_effective, duration_pref
            ORDER BY
                CASE WHEN duration_pref = 'shortest' THEN COALESCE(duration_days, 0)
                     ELSE -COALESCE(duration_days, 0) END ASC,
                {_XBRL_RANK_SQL} ASC,
                context_id ASC
        ) = 1
    )"""
    parts["stlm_period_by_filing"] = f"""stlm_period_by_filing AS (
        -- §3.3 priority 2: dart_share_count_raw / dart_shareholder_return_raw.stlm_dt.
        SELECT corp_code, bsns_year, reprt_code, rcept_no, MIN(stlm_dt) AS stlm_period_end
        FROM (
            SELECT corp_code, bsns_year, reprt_code, rcept_no, stlm_dt
            FROM {share_count_view} WHERE stlm_dt IS NOT NULL
            UNION ALL
            SELECT corp_code, bsns_year, reprt_code, rcept_no, stlm_dt
            FROM {shareholder_return_view} WHERE stlm_dt IS NOT NULL
        ) t
        GROUP BY corp_code, bsns_year, reprt_code, rcept_no
    )"""
    parts["filing_period_end"] = f"""filing_period_end AS (
        SELECT
            fk.corp_code, fk.bsns_year, fk.reprt_code, fk.rcept_no,
            COALESCE(
                xp.xbrl_period_end, sp.stlm_period_end,
                {_calendar_period_end_expr("fk.reprt_code", "fk.bsns_year")}
            ) AS period_end,
            CASE
                WHEN xp.xbrl_period_end IS NOT NULL THEN 'xbrl'
                WHEN sp.stlm_period_end IS NOT NULL THEN 'stlm'
                ELSE 'calendar_fallback'
            END AS period_end_source,
            (
                xp.xbrl_period_end IS NOT NULL AND sp.stlm_period_end IS NOT NULL
                AND xp.xbrl_period_end <> sp.stlm_period_end
            ) AS period_end_conflict
        FROM filing_keys fk
        LEFT JOIN xbrl_period_by_filing xp USING (corp_code, bsns_year, reprt_code, rcept_no)
        LEFT JOIN stlm_period_by_filing sp USING (corp_code, bsns_year, reprt_code, rcept_no)
    )"""
    parts["filing_availability"] = """filing_availability AS (
        SELECT
            fk.corp_code, fk.bsns_year, fk.reprt_code, fk.rcept_no,
            CASE
                WHEN fk.rcept_no ~ '^[0-9]{14}$'
                THEN strptime(left(fk.rcept_no, 8), '%Y%m%d')::DATE
                ELSE NULL
            END AS disclosed_date
        FROM filing_keys fk
    )"""
    parts["filing_receipt_relation"] = f"""filing_receipt_relation AS (
        -- §3.5: only a *matched* receipt lets us classify original/revision.
        -- report_nm carrying "정정" is OpenDART's own correction annotation.
        SELECT corp_code, rcept_no, (report_nm LIKE '%정정%') AS is_correction_by_report_nm
        FROM {filing_receipt_view}
    )"""
    parts["filing_lineage"] = f"""filing_lineage AS (
        SELECT
            fa.corp_code, fa.bsns_year, fa.reprt_code, fa.rcept_no,
            fa.disclosed_date,
            CASE
                WHEN fa.disclosed_date IS NOT NULL
                THEN (SELECT MIN(c.d) FROM {calendar_table} c WHERE c.d > fa.disclosed_date)
                ELSE NULL
            END AS available_from_from_receipt,
            (fa.disclosed_date IS NOT NULL) AS has_parsed_receipt_date,
            rr.is_correction_by_report_nm,
            (rr.corp_code IS NOT NULL) AS receipt_matched,
            -- §3.2: same-day multiple filings collapse to the numerically-latest
            -- rcept_no (the only one whose value is knowable by next session).
            MAX(fa.rcept_no) OVER (
                PARTITION BY fa.corp_code, fa.bsns_year, fa.reprt_code, fa.disclosed_date
            ) AS same_day_effective_rcept_no
        FROM filing_availability fa
        LEFT JOIN filing_receipt_relation rr
          ON rr.corp_code = fa.corp_code AND rr.rcept_no = fa.rcept_no
    )"""
    parts["candidates"] = f"""candidates AS (
        -- dart_financial_statement_raw (value_selector is always thstrm_amount)
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_fin} AS period_type,
            pe.period_end AS statement_period_end,
            f.bsns_year, f.reprt_code,
            f.fs_div AS fs_basis,
            f.rcept_no,
            CAST(f.thstrm_amount AS DECIMAL(30,4)) AS value_numeric,
            r.unit,
            f.currency,
            'dart_financial_statement_raw' AS source_table,
            concat(f.rcept_no, ':', f.account_id, ':', f.ord) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            0 AS candidate_rank,
            pe.period_end_source, pe.period_end_conflict,
            xr.value_numeric AS pairing_xbrl_value,
            xr.pairing_xbrl_source_key,
            -- B-3 (fin_quarterly_metric_vintage) cross-check inputs — only
            -- meaningful for IS/CIS interim filings, NULL from the other
            -- three branches below.
            CAST(f.thstrm_add_amount AS DECIMAL(30,4)) AS cumulative_value_numeric,
            CAST(f.frmtrm_q_amount AS DECIMAL(30,4)) AS comparative_q_amount
        FROM {financial_view} f
        JOIN corp c ON c.ticker = f.ticker
        JOIN rule_rel r
          ON {_FIN_RULE_MATCH_SQL}
        JOIN filing_period_end pe
          ON pe.corp_code = f.corp_code AND pe.bsns_year = f.bsns_year
         AND pe.reprt_code = f.reprt_code AND pe.rcept_no = f.rcept_no
        LEFT JOIN xbrl_pairing xr
          ON xr.corp_code = f.corp_code AND xr.bsns_year = f.bsns_year
         AND xr.reprt_code = f.reprt_code AND xr.rcept_no = f.rcept_no
         AND xr.concept_id = f.account_id
         AND xr.xbrl_fs_basis = f.fs_div
         AND xr.period_end_effective = pe.period_end
         AND xr.duration_pref = CASE
                WHEN f.sj_div IN ('IS', 'CIS') THEN 'shortest' ELSE 'longest' END
        WHERE f.thstrm_amount IS NOT NULL

        UNION ALL
        -- dart_share_count_raw (no cross-source XBRL pairing target)
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_sc} AS period_type,
            pe.period_end AS statement_period_end,
            s.bsns_year, s.reprt_code,
            '' AS fs_basis,
            s.rcept_no,
            CAST(
                CASE r.value_selector
                    WHEN 'istc_totqy' THEN s.istc_totqy
                    WHEN 'tesstk_co' THEN s.tesstk_co
                END AS DECIMAL(30,4)
            ) AS value_numeric,
            r.unit,
            NULL AS currency,
            'dart_share_count_raw' AS source_table,
            concat(s.rcept_no, ':', s.se) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            0 AS candidate_rank,
            pe.period_end_source, pe.period_end_conflict,
            NULL AS pairing_xbrl_value,
            NULL AS pairing_xbrl_source_key,
            NULL AS cumulative_value_numeric,
            NULL AS comparative_q_amount
        FROM {share_count_view} s
        JOIN corp c ON c.ticker = s.ticker
        JOIN rule_rel r
          ON r.source_table = 'dart_share_count_raw'
         AND (r.row_name = '' OR s.se = r.row_name)
        JOIN filing_period_end pe
          ON pe.corp_code = s.corp_code AND pe.bsns_year = s.bsns_year
         AND pe.reprt_code = s.reprt_code AND pe.rcept_no = s.rcept_no
        WHERE CASE r.value_selector
                  WHEN 'istc_totqy' THEN s.istc_totqy
                  WHEN 'tesstk_co' THEN s.tesstk_co
              END IS NOT NULL

        UNION ALL
        -- dart_shareholder_return_raw (no cross-source XBRL pairing target)
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_sr} AS period_type,
            pe.period_end AS statement_period_end,
            sr.bsns_year, sr.reprt_code,
            '' AS fs_basis,
            sr.rcept_no,
            CAST(sr.value_numeric AS DECIMAL(30,4)) AS value_numeric,
            r.unit,
            NULL AS currency,
            'dart_shareholder_return_raw' AS source_table,
            concat(
                sr.rcept_no, ':', sr.statement_type, ':', sr.row_name, ':',
                sr.stock_knd, ':', sr.dim1, ':', sr.dim2, ':', sr.dim3, ':',
                sr.metric_code
            ) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            0 AS candidate_rank,
            pe.period_end_source, pe.period_end_conflict,
            NULL AS pairing_xbrl_value,
            NULL AS pairing_xbrl_source_key,
            NULL AS cumulative_value_numeric,
            NULL AS comparative_q_amount
        FROM {shareholder_return_view} sr
        JOIN corp c ON c.ticker = sr.ticker
        JOIN rule_rel r
          ON r.source_table = 'dart_shareholder_return_raw'
         AND (r.statement_type = '' OR sr.statement_type = r.statement_type)
         AND (r.row_name = '' OR sr.row_name = r.row_name)
         AND (r.stock_knd = '' OR sr.stock_knd = r.stock_knd)
         AND (r.dim1 = '' OR sr.dim1 = r.dim1)
         AND (r.dim2 = '' OR sr.dim2 = r.dim2)
         AND (r.dim3 = '' OR sr.dim3 = r.dim3)
         AND (r.metric_code_match = '' OR sr.metric_code = r.metric_code_match)
        JOIN filing_period_end pe
          ON pe.corp_code = sr.corp_code AND pe.bsns_year = sr.bsns_year
         AND pe.reprt_code = sr.reprt_code AND pe.rcept_no = sr.rcept_no
        WHERE sr.value_numeric IS NOT NULL

        UNION ALL
        -- dart_xbrl_fact_raw (is itself the pairing target for financial rows)
        --
        -- fs_basis follows the RULE, not the source table (I7). A rule with
        -- ``fs_div = ''`` keeps the historical empty basis; the metrics sourced
        -- that way -- weighted-average shares, depreciation -- have always
        -- lived at ``fs_basis = ''`` and moving them would break parity well
        -- outside I7's scope.
        --
        -- A rule naming CFS or OFS reads the basis off the XBRL dimensions
        -- instead, which is what lets an XBRL fallback compete with a
        -- financial-statement rule at all. The winner window partitions by
        -- fs_basis, so a fallback stranded at '' never competes with a CFS
        -- statement row -- it just adds a second row for the same metric and
        -- period. That is what happened on the first attempt.
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_xf} AS period_type,
            pe.period_end AS statement_period_end,
            x.bsns_year, x.reprt_code,
            CASE WHEN r.fs_div = '' THEN '' ELSE COALESCE(x.xbrl_fs_basis, '') END AS fs_basis,
            x.rcept_no,
            CAST(x.value_numeric AS DECIMAL(30,4)) AS value_numeric,
            r.unit,
            NULL AS currency,
            'dart_xbrl_fact_raw' AS source_table,
            concat(x.rcept_no, ':', x.context_id, ':', x.concept_id) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            -- Prefer the cumulative (longest) duration among the contexts that
            -- survive the period filter: every metric sourced this way is a
            -- cash-flow-statement item, which OpenDART reports year-to-date
            -- (fin_quarterly_metric_vintage's `cumulative_reported` kind).
            -- Same convention the pairing CTE applies to sj_div='CF'.
            -{_XBRL_DURATION_DAYS} * 1000 + {_XBRL_RANK_SQL} AS candidate_rank,
            pe.period_end_source, pe.period_end_conflict,
            NULL AS pairing_xbrl_value,
            NULL AS pairing_xbrl_source_key,
            NULL AS cumulative_value_numeric,
            NULL AS comparative_q_amount
        FROM {xbrl_candidate_source} x
        JOIN corp c ON c.ticker = x.ticker
        JOIN rule_rel r
          ON r.source_table = 'dart_xbrl_fact_raw'
         AND (r.fs_div = '' OR x.xbrl_fs_basis = r.fs_div)
         AND (r.account_id = '' OR x.concept_id = r.account_id)
         AND (r.account_nm = ''
              OR x.label_ko = r.account_nm
              OR x.concept_name = r.account_nm)
        JOIN filing_period_end pe
          ON pe.corp_code = x.corp_code AND pe.bsns_year = x.bsns_year
         AND pe.reprt_code = x.reprt_code AND pe.rcept_no = x.rcept_no
        -- Without this the winner could be a comparative year's fact carrying
        -- the current period's statement_period_end — a two-year-old number
        -- silently presented as this filing's (08 §4.3.2).
        WHERE x.value_numeric IS NOT NULL
          AND x.period_end_effective = pe.period_end
    )"""
    parts["winners"] = f"""winners AS (
        SELECT *
        FROM candidates
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, metric_code, statement_period_end, fs_basis, rcept_no
            ORDER BY priority ASC, candidate_rank ASC, source_key ASC{winner_tie_break}
        ) = 1
    )"""

    final = f"""SELECT
        w.ticker, w.market, w.corp_code, w.metric_code, w.period_type,
        w.statement_period_end, w.bsns_year, w.reprt_code, w.fs_basis, w.rcept_no,
        fl.disclosed_date,
{availability_select}
        fl.same_day_effective_rcept_no,
        w.value_numeric, w.unit, w.currency,
        w.source_table, w.source_key, w.mapping_rule_code, w.priority AS mapping_priority,
        w.period_end_source, w.period_end_conflict,
        CASE
            WHEN NOT fl.receipt_matched THEN NULL
            ELSE NOT fl.is_correction_by_report_nm
        END AS is_original_by_report_nm,
        CASE
            WHEN NOT fl.receipt_matched THEN NULL
            WHEN fl.is_correction_by_report_nm THEN TRUE
            ELSE FALSE
        END AS is_revision,
        CASE
            WHEN fl.receipt_matched AND NOT fl.is_correction_by_report_nm THEN w.rcept_no
            ELSE NULL
        END AS original_rcept_no,
        -- §3.5: the four statuses describe a (ticker, metric, period) vintage
        -- *chain*, not this row in isolation. `complete_original_and_revisions`
        -- is never emitted — proving no further revision was ever filed needs
        -- a full receipt-list scan this mart does not perform (see docstring).
        CASE
            WHEN NOT MAX(fl.receipt_matched) OVER (
                PARTITION BY w.ticker, w.metric_code, w.statement_period_end, w.fs_basis
            ) THEN 'captured_vintages_only'
            WHEN fl.receipt_matched THEN 'original_confirmed_revisions_partial'
            ELSE 'unlinked_receipt'
        END AS captured_vintage_status,
        CASE
            WHEN w.source_table <> 'dart_financial_statement_raw' THEN 'not_applicable'
            WHEN w.pairing_xbrl_value IS NULL THEN 'unlinked_receipt'
            WHEN ABS(w.value_numeric - w.pairing_xbrl_value) <= {pairing_tolerance}
                THEN 'verified_same_receipt'
            ELSE 'value_mismatch'
        END AS receipt_value_pairing_status,
        w.pairing_xbrl_source_key,
        {pairing_tolerance} AS pairing_tolerance,
        w.cumulative_value_numeric, w.comparative_q_amount
    FROM {winners_source} w
    LEFT JOIN filing_lineage fl
      ON fl.corp_code = w.corp_code AND fl.bsns_year = w.bsns_year
     AND fl.reprt_code = w.reprt_code AND fl.rcept_no = w.rcept_no
    """
    return parts, final


_CTE_ORDER = (
    "corp",
    "rule_rel",
    "filing_keys",
    "xbrl_scoped",
    "xbrl_period_by_filing",
    "xbrl_pairing",
    "stlm_period_by_filing",
    "filing_period_end",
    "filing_availability",
    "filing_receipt_relation",
    "filing_lineage",
    "candidates",
    "winners",
)


def _join_ctes(parts: dict[str, str], names: Sequence[str], final: str) -> str:
    """``WITH <names...> <final>`` laid out like the original single statement."""
    return "\n    WITH " + ",\n    ".join(parts[name] for name in names) + "\n    " + final


def build_stock_metric_vintage_fact_sql(
    *,
    financial_view: str = "dart_financial_statement_raw",
    share_count_view: str = "dart_share_count_raw",
    shareholder_return_view: str = "dart_shareholder_return_raw",
    xbrl_view: str = "dart_xbrl_fact_raw",
    filing_receipt_view: str = "dart_filing_receipt_raw",
    corp_view: str = "dart_corp_master",
    calendar_table: str = _CAL_TABLE,
    pairing_tolerance: float = DEFAULT_PAIRING_TOLERANCE,
    semantics: str = SEMANTICS_V1,
) -> str:
    """SQL producing ``stock_metric_vintage_fact`` rows from the raw lake views.

    ``calendar_table`` must already be a real table ``(d DATE, idx BIGINT)`` of
    KRX sessions (see ``register_stock_metric_vintage_fact_view``) — a
    correlated ``MIN(d) WHERE d > disclosed_date`` gives the first KRX session
    strictly after disclosure (§3.2; same-day filings never get intraday
    availability, since no receipt time is recorded).

    ``semantics="v1"`` (default) is the statement existing marts were written
    under. ``semantics="v2"`` differs in two places, both on rows where ``v1``
    is wrong or plan-dependent:

    - *calendar boundary*: a parseable receipt date never falls back to
      ``period_end + 45/90d``. If its next session is not in the calendar,
      ``available_from`` is NULL and ``availability_source`` is
      ``receipt_beyond_calendar``. ``synthetic_fallback`` now means only
      "no usable receipt date".
    - *tie rule*: the ``winners`` window ends with ``bsns_year ASC`` and report
      order, so the report that first published a period's value wins.

    A consumer that takes ``greatest()`` over several ``available_from`` (the
    TTM in ``fin_quarterly_metric_vintage``) must not be handed NULLs: DuckDB's
    ``greatest`` skips them. The serving builder therefore extends the calendar
    past K and fails if any row is still ``receipt_beyond_calendar``.
    """
    parts, final = _vintage_sql_parts(
        financial_view=financial_view,
        share_count_view=share_count_view,
        shareholder_return_view=shareholder_return_view,
        xbrl_view=xbrl_view,
        filing_receipt_view=filing_receipt_view,
        corp_view=corp_view,
        calendar_table=calendar_table,
        pairing_tolerance=pairing_tolerance,
        semantics=semantics,
    )
    return _join_ctes(parts, _CTE_ORDER, final)


_STG_XBRL_KEYS = "_smvf_xbrl_keys"
_STG_XBRL_PERIOD = "_smvf_xbrl_period"
_STG_XBRL_SCOPED = "_smvf_xbrl_scoped"
_STG_PAIRING_IN = "_smvf_pairing_in"
_STG_WINNERS = "_smvf_winners"

_VIEW_DEFAULTS: dict[str, object] = {
    "financial_view": "dart_financial_statement_raw",
    "share_count_view": "dart_share_count_raw",
    "shareholder_return_view": "dart_shareholder_return_raw",
    "xbrl_view": "dart_xbrl_fact_raw",
    "filing_receipt_view": "dart_filing_receipt_raw",
    "corp_view": "dart_corp_master",
    "calendar_table": _CAL_TABLE,
    "pairing_tolerance": DEFAULT_PAIRING_TOLERANCE,
}


def _xbrl_concept_ids() -> list[str]:
    """XBRL concept ids the mapping rules can ever read, derived from the rules.

    Both rule kinds count: an XBRL rule reads the fact itself, and a
    financial-statement rule pairs its row with the XBRL fact of the same
    ``concept_id = account_id``. A rule without an ``account_id`` would match
    every concept, so restricting the XBRL facts by concept would silently drop
    rows; the staged plan refuses to build rather than guess.
    """
    ids: set[str] = set()
    for rule in default_metric_mapping_rules():
        if rule.source_table not in ("dart_financial_statement_raw", "dart_xbrl_fact_raw"):
            continue
        if not rule.account_id:
            raise ValueError(
                f"rule {rule.rule_code!r} has no account_id; the staged plan restricts XBRL "
                "facts to the concepts the rules name and cannot be used with a wildcard rule"
            )
        ids.add(rule.account_id)
    return sorted(ids)


def _staged_plan(semantics: str, **kw: object) -> MartPlan:
    """The measured six-step plan for ``stock_metric_vintage_fact``.

    The single statement hands DuckDB a ``financial rows EXISTS ...`` shape that
    it estimates at 2.8M XBRL rows when the table has 114M; it builds a hash
    table over all of it and the 8-key window sort spills past 160GB. Here every
    step is written to parquet and read back, so the next step is planned on real
    row counts (Mac, 2026-09-29 snapshot, 2 threads, 4GB: about 55 s, temp 4.6GB):

    1. ``_smvf_xbrl_keys``: DISTINCT XBRL filing keys (111,393 rows).
    2. ``_smvf_xbrl_period``: the original ``xbrl_period_by_filing`` text, over
       *all* XBRL facts (the filing period needs every context).
    3. ``_smvf_xbrl_scoped``: XBRL facts of the concepts the rules name (derived
       at run time, see :func:`_xbrl_concept_ids`). NULL values are kept here: the
       pairing picks its winner among them in the single statement, and the
       candidate branch drops them itself.
    4. ``_smvf_pairing_in``: an explicit ``INNER JOIN`` to the DISTINCT keys
       (filing, concept, basis) of financial-statement rows that match a rule.
       Not ``EXISTS``: a join the planner sees the size of. The keys are DISTINCT
       on exactly the join columns, so a fact matches at most one key row and the
       join cannot multiply rows; a plan check also asserts it.
    5. ``_smvf_winners``: ``candidates`` and ``winners`` of the original text,
       reading steps 1, 2, 3 and 4.
    6. The final SELECT over ``_smvf_winners`` and ``filing_lineage``.

    The rows equal the single statement's: the same CTE texts, with only the
    relation an XBRL branch reads swapped for a pre-filtered copy of it.
    """
    parts, final = _vintage_sql_parts(
        **kw,  # type: ignore[arg-type]
        semantics=semantics,
        xbrl_keys_source=_STG_XBRL_KEYS,
        xbrl_pairing_source=_STG_PAIRING_IN,
        xbrl_candidate_source=_STG_XBRL_SCOPED,
        winners_source=_STG_WINNERS,
    )
    xbrl_view = kw["xbrl_view"]
    financial_view = kw["financial_view"]
    concepts = ", ".join(_sql_str_literal(c) for c in _xbrl_concept_ids())
    period_cte = f"xbrl_period_by_filing AS (SELECT * FROM {_STG_XBRL_PERIOD})"

    keys_sql = f"SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {xbrl_view}"
    period_sql = _join_ctes(
        parts, ("xbrl_scoped", "xbrl_period_by_filing"), "SELECT * FROM xbrl_period_by_filing"
    )
    scoped_sql = _join_ctes(
        parts, ("xbrl_scoped",), f"SELECT * FROM xbrl_scoped WHERE concept_id IN ({concepts})"
    )
    pairing_in_sql = _join_ctes(
        parts,
        ("rule_rel",),
        f"""SELECT
        x.corp_code, x.bsns_year, x.reprt_code, x.rcept_no, x.concept_id,
        x.xbrl_fs_basis, x.period_end_effective, x.duration_days, x.dimensions,
        x.context_id, x.value_numeric
    FROM {_STG_XBRL_SCOPED} x
    INNER JOIN (
        SELECT DISTINCT
            f.corp_code, f.bsns_year, f.reprt_code, f.rcept_no,
            f.account_id AS concept_id, f.fs_div AS xbrl_fs_basis
        FROM {financial_view} f
        JOIN rule_rel r
          ON {_FIN_RULE_MATCH_SQL}
        WHERE f.thstrm_amount IS NOT NULL
    ) k
      ON k.corp_code = x.corp_code AND k.bsns_year = x.bsns_year
     AND k.reprt_code = x.reprt_code AND k.rcept_no = x.rcept_no
     AND k.concept_id = x.concept_id AND k.xbrl_fs_basis = x.xbrl_fs_basis
    """,
    )
    parts_staged = dict(parts, xbrl_period_by_filing=period_cte)
    winners_sql = _join_ctes(
        parts_staged,
        (
            "corp",
            "rule_rel",
            "filing_keys",
            "xbrl_period_by_filing",
            "xbrl_pairing",
            "stlm_period_by_filing",
            "filing_period_end",
            "candidates",
            "winners",
        ),
        "SELECT * FROM winners\n    ",
    )
    final_sql = _join_ctes(
        parts,
        ("filing_keys", "filing_availability", "filing_receipt_relation", "filing_lineage"),
        final,
    )
    return MartPlan(
        plan_id=PLAN_STAGED,
        semantics_version=semantics,
        final_sql=final_sql,
        stages=(
            (_STG_XBRL_KEYS, keys_sql),
            (_STG_XBRL_PERIOD, period_sql),
            (_STG_XBRL_SCOPED, scoped_sql),
            (_STG_PAIRING_IN, pairing_in_sql),
            (_STG_WINNERS, winners_sql),
        ),
        checks=(
            (
                "pairing_inner_join_keeps_at_most_the_scoped_rows",
                f"SELECT (SELECT count(*) FROM {_STG_PAIRING_IN})"
                f" <= (SELECT count(*) FROM {_STG_XBRL_SCOPED})",
            ),
        ),
    )


def build_stock_metric_vintage_fact_plan(
    *, semantics: str = SEMANTICS_V1, plan: str = PLAN_SINGLE, **views: object
) -> MartPlan | None:
    """The :class:`MartPlan` for ``(semantics, plan)``, or None for the legacy default.

    ``None`` means "write the contract statement as it always was", so a caller
    that passes neither argument leaves the cache metadata untouched.
    """
    if semantics not in SEMANTICS_VERSIONS:
        raise ValueError(f"unknown stock_metric_vintage_fact semantics {semantics!r}")
    if plan not in PLANS:
        raise ValueError(f"unknown stock_metric_vintage_fact plan {plan!r}; expected {PLANS}")
    unknown = set(views) - set(_VIEW_DEFAULTS)
    if unknown:
        raise TypeError(f"unexpected view arguments: {sorted(unknown)}")
    kw = {**_VIEW_DEFAULTS, **views}
    if plan == PLAN_STAGED:
        return _staged_plan(semantics, **kw)
    if semantics == SEMANTICS_V1:
        return None
    return MartPlan(
        plan_id=PLAN_SINGLE,
        semantics_version=semantics,
        final_sql=build_stock_metric_vintage_fact_sql(**kw, semantics=semantics),  # type: ignore[arg-type]
    )


def _register_calendar(con: duckdb.DuckDBPyConnection, trading_days: Sequence[date]) -> None:
    con.execute(f"DROP TABLE IF EXISTS {_CAL_TABLE}")
    con.execute(f"CREATE TABLE {_CAL_TABLE} (d DATE, idx BIGINT)")
    con.executemany(
        f"INSERT INTO {_CAL_TABLE} VALUES (?, ?)",
        [(d, i + 1) for i, d in enumerate(trading_days)],
    )


def register_stock_metric_vintage_fact_view(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: Sequence[date],
    view_name: str = SMVF_TABLE,
    semantics: str = SEMANTICS_V1,
    **views: str,
) -> str:
    """Register the KRX session calendar table, then a view over the SQL above.

    Lightweight (no parquet) path for tests/parity checks; the orchestrated
    pipeline should materialize instead via
    ``materialize_stock_metric_vintage_fact``.
    """
    _register_calendar(con, trading_days)
    sql = build_stock_metric_vintage_fact_sql(**views, semantics=semantics)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_stock_metric_vintage_fact(
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
    """Build + register ``stock_metric_vintage_fact`` as a cached parquet mart.

    Requires the raw DART views (``dart_financial_statement_raw`` etc.) already
    registered on ``con``. ``trading_days`` is caller-supplied (matches
    ``research/etl/lake.py``'s "calendar-source-agnostic mart" convention) —
    typically ``get_trading_days`` spanning the raw lake's filing date range.

    ``semantics`` is what the mart means and ``plan`` how it is computed (see
    :class:`~modeler.etl.mart.MartPlan`). The defaults are the legacy statement
    and write exactly the metadata they always did. ``plan="staged"`` needs
    ``stage_dir``, a directory owned by this run, so stage files are never reused
    across runs.

    With ``semantics="v2"`` pass a calendar that reaches past the newest receipt:
    a receipt whose next session is missing gets ``available_from = NULL``.
    """
    _register_calendar(con, trading_days)
    materialize(
        con,
        config,
        SMVF_TABLE,
        build_stock_metric_vintage_fact_sql(**views, semantics=semantics),
        force=force,
        plan=build_stock_metric_vintage_fact_plan(semantics=semantics, plan=plan, **views),
        stage_dir=stage_dir,
    )
    return register_mart_view(con, config, SMVF_TABLE)
