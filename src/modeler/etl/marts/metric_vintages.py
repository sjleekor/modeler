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

import duckdb

from collector.kr.shared import (
    MetricMappingRule,
    default_metric_catalog,
    default_metric_mapping_rules,
)
from modeler.etl.config import LakeConfig
from modeler.etl.lake import _sql_str_literal
from modeler.etl.mart import materialize, register_mart_view

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
) -> str:
    """SQL producing ``stock_metric_vintage_fact`` rows from the raw lake views.

    ``calendar_table`` must already be a real table ``(d DATE, idx BIGINT)`` of
    KRX sessions (see ``register_stock_metric_vintage_fact_view``) — a
    correlated ``MIN(d) WHERE d > disclosed_date`` gives the first KRX session
    strictly after disclosure (§3.2; same-day filings never get intraday
    availability, since no receipt time is recorded).
    """
    rules = default_metric_mapping_rules()
    unit_by_code = {entry.metric_code: entry.unit for entry in default_metric_catalog()}
    rules_rel = _rules_relation_sql(rules, unit_by_code)

    period_type_fin = _period_type_expr("f.reprt_code")
    period_type_sc = _period_type_expr("s.reprt_code")
    period_type_sr = _period_type_expr("sr.reprt_code")
    period_type_xf = _period_type_expr("x.reprt_code")

    return f"""
    WITH corp AS (
        -- §3.4: no is_active filter — active-status backcast is a survivor bias.
        SELECT ticker, market, corp_code
        FROM {corp_view}
        WHERE ticker IS NOT NULL AND ticker <> ''
          AND market IS NOT NULL
    ),
    rule_rel AS (SELECT * FROM {rules_rel}),
    filing_keys AS (
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no
        FROM {financial_view}
        UNION
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {share_count_view}
        UNION
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {shareholder_return_view}
        UNION
        SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM {xbrl_view}
    ),
    xbrl_scoped AS (
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
    ),
    xbrl_period_by_filing AS (
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
            NOT rcept_no ~ '^[0-9]{{14}}$'
            OR period_end_effective <= strptime(left(rcept_no, 8), '%Y%m%d')::DATE
          )
        GROUP BY corp_code, bsns_year, reprt_code, rcept_no
    ),
    xbrl_pairing AS (
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
        FROM xbrl_scoped
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
    ),
    stlm_period_by_filing AS (
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
    ),
    filing_period_end AS (
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
    ),
    filing_availability AS (
        SELECT
            fk.corp_code, fk.bsns_year, fk.reprt_code, fk.rcept_no,
            CASE
                WHEN fk.rcept_no ~ '^[0-9]{{14}}$'
                THEN strptime(left(fk.rcept_no, 8), '%Y%m%d')::DATE
                ELSE NULL
            END AS disclosed_date
        FROM filing_keys fk
    ),
    filing_receipt_relation AS (
        -- §3.5: only a *matched* receipt lets us classify original/revision.
        -- report_nm carrying "정정" is OpenDART's own correction annotation.
        SELECT corp_code, rcept_no, (report_nm LIKE '%정정%') AS is_correction_by_report_nm
        FROM {filing_receipt_view}
    ),
    filing_lineage AS (
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
    ),
    candidates AS (
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
          ON r.source_table = 'dart_financial_statement_raw'
         AND (r.fs_div = '' OR f.fs_div = r.fs_div)
         AND (r.sj_div = '' OR f.sj_div = r.sj_div)
         AND (r.account_id = '' OR f.account_id = r.account_id)
         AND (r.account_nm = '' OR f.account_nm = r.account_nm)
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
        FROM xbrl_scoped x
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
    ),
    winners AS (
        SELECT *
        FROM candidates
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, metric_code, statement_period_end, fs_basis, rcept_no
            ORDER BY priority ASC, candidate_rank ASC, source_key ASC
        ) = 1
    )
    SELECT
        w.ticker, w.market, w.corp_code, w.metric_code, w.period_type,
        w.statement_period_end, w.bsns_year, w.reprt_code, w.fs_basis, w.rcept_no,
        fl.disclosed_date,
        CAST(
            COALESCE(
                fl.available_from_from_receipt,
                w.statement_period_end
                + CASE WHEN w.period_type = 'annual' THEN INTERVAL '{ANNUAL_FALLBACK_DAYS} days'
                       ELSE INTERVAL '{QUARTERLY_FALLBACK_DAYS} days' END
            ) AS DATE
        ) AS available_from,
        CASE WHEN fl.has_parsed_receipt_date THEN 'rcept_no' ELSE 'synthetic_fallback' END
            AS availability_source,
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
    FROM winners w
    LEFT JOIN filing_lineage fl
      ON fl.corp_code = w.corp_code AND fl.bsns_year = w.bsns_year
     AND fl.reprt_code = w.reprt_code AND fl.rcept_no = w.rcept_no
    """


def register_stock_metric_vintage_fact_view(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: Sequence[date],
    view_name: str = SMVF_TABLE,
    **views: str,
) -> str:
    """Register the KRX session calendar table, then a view over the SQL above.

    Lightweight (no parquet) path for tests/parity checks; the orchestrated
    pipeline should materialize instead via
    ``materialize_stock_metric_vintage_fact``.
    """
    con.execute(f"DROP TABLE IF EXISTS {_CAL_TABLE}")
    con.execute(f"CREATE TABLE {_CAL_TABLE} (d DATE, idx BIGINT)")
    con.executemany(
        f"INSERT INTO {_CAL_TABLE} VALUES (?, ?)",
        [(d, i + 1) for i, d in enumerate(trading_days)],
    )
    sql = build_stock_metric_vintage_fact_sql(**views)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_stock_metric_vintage_fact(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    trading_days: Sequence[date],
    force: bool = False,
    **views: str,
) -> str:
    """Build + register ``stock_metric_vintage_fact`` as a cached parquet mart.

    Requires the raw DART views (``dart_financial_statement_raw`` etc.) already
    registered on ``con``. ``trading_days`` is caller-supplied (matches
    ``research/etl/lake.py``'s "calendar-source-agnostic mart" convention) —
    typically ``get_trading_days`` spanning the raw lake's filing date range.
    """
    con.execute(f"DROP TABLE IF EXISTS {_CAL_TABLE}")
    con.execute(f"CREATE TABLE {_CAL_TABLE} (d DATE, idx BIGINT)")
    con.executemany(
        f"INSERT INTO {_CAL_TABLE} VALUES (?, ?)",
        [(d, i + 1) for i, d in enumerate(trading_days)],
    )
    materialize(
        con,
        config,
        SMVF_TABLE,
        build_stock_metric_vintage_fact_sql(**views),
        force=force,
    )
    return register_mart_view(con, config, SMVF_TABLE)
