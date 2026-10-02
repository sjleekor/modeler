"""DuckDB port of ``metrics normalize`` -> ``stock_metric_fact`` (refactor §3.1).

Recomputes the canonical financial-metric facts from the raw DART lake instead
of the Postgres ``service/normalize_metrics.py`` row-by-row path. The mapping
rules and catalog come from the pure code definitions in
``collector.kr.definitions.metric_rules`` (no Postgres ``metric_mapping_rule``
table). The output schema is parquet/DuckDB only and is kept compatible with the
old ``stock_metric_fact`` so ``research/etl/features/fin_pit.py`` reads it via the
same ``stock_metric_fact`` view.

Selection semantics reproduced 1:1 from ``_collect_candidates``:

- one candidate per (raw row, matching rule); a rule matches with wildcard
  semantics (``rule.col == '' OR row.col == rule.col``);
- the winner per ``(ticker, metric_code, bsns_year, reprt_code)`` is the minimum
  ``(priority, candidate_rank, source_key)`` — ``candidate_rank`` is 0 for every
  source except XBRL, which ranks by dimension count/flags (``_xbrl_candidate_rank``);
- ``value_text`` mirrors Python ``str(Decimal)``: share-count columns are BIGINT
  so they render without decimals (``'27931470'``); the DECIMAL(30,4) selectors
  render with four places (``'80.0000'``).
"""

from __future__ import annotations

import duckdb
from collector.kr.shared import (
    CURRENT_METRIC_RULES_VERSION,
    MetricMappingRule,
    default_metric_catalog,
    default_metric_mapping_rules,
    metric_rules_content_hash,
    resolve_metric_rules_version,
)

from modeler.etl.lake import _sql_str_literal
from modeler.etl.mart import MartPlan

SMF_VIEW = "stock_metric_fact"

#: Default: one join per source against the whole rule table with ``(col = '' OR ...)``
#: wildcards, then ``QUALIFY ROW_NUMBER`` over every candidate. The text every frozen
#: mart was written under.
PLAN_SINGLE = "single"
#: Same rows: each source joins its equality rules and its wildcard rules separately (so
#: the equality branch is a hash join), and the winner is ``arg_min`` of a ``GROUP BY``
#: instead of a window over all candidates.
PLAN_SPLIT_ARGMIN = "split_argmin"
PLANS = (PLAN_SINGLE, PLAN_SPLIT_ARGMIN)

# reprt_code -> period_type / period_end, matching definitions.metric_rules.
_PERIOD_TYPE_SQL = (
    "CASE {col} WHEN '11013' THEN 'q1' WHEN '11012' THEN 'half' "
    "WHEN '11014' THEN 'q3' WHEN '11011' THEN 'annual' ELSE 'unknown' END"
)


def _period_type_expr(col: str) -> str:
    return _PERIOD_TYPE_SQL.format(col=col)


def _infer_period_end_expr(reprt_col: str, year_col: str) -> str:
    """SQL for ``infer_period_end(bsns_year, reprt_code)`` (NULL on unknown)."""
    return (
        f"CASE {reprt_col} "
        f"WHEN '11013' THEN make_date({year_col}, 3, 31) "
        f"WHEN '11012' THEN make_date({year_col}, 6, 30) "
        f"WHEN '11014' THEN make_date({year_col}, 9, 30) "
        f"WHEN '11011' THEN make_date({year_col}, 12, 31) "
        f"ELSE NULL END"
    )


# XBRL candidate rank == _xbrl_candidate_rank(row): substring checks on the
# joined dimension list are equivalent to LIKE over the raw JSON-array string.
_XBRL_RANK_SQL = (
    "COALESCE(json_array_length(dimensions), 0) * 10"
    " + (CASE WHEN dimensions LIKE '%ConsolidatedMember%' THEN -5"
    " WHEN dimensions LIKE '%SeparateMember%' THEN 5 ELSE 0 END)"
    " + (CASE WHEN dimensions LIKE '%ReportedAmountMember%' THEN 1 ELSE 0 END)"
    " + (CASE WHEN dimensions LIKE '%OperatingSegmentsMember%' THEN 3 ELSE 0 END)"
)


def _rules_relation_sql(
    rules: list[MetricMappingRule],
    unit_by_code: dict[str, str],
) -> str:
    """Build a ``(VALUES ...) AS rules(...)`` relation from the code rule list.

    ``unit`` is resolved from the catalog here so the fact carries the same unit
    the Python path attached via ``unit_by_metric_code``.
    """
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


def build_stock_metric_fact_sql(
    *,
    financial_view: str = "dart_financial_statement_raw",
    share_count_view: str = "dart_share_count_raw",
    shareholder_return_view: str = "dart_shareholder_return_raw",
    xbrl_view: str = "dart_xbrl_fact_raw",
    corp_view: str = "dart_corp_master",
    plan: str = PLAN_SINGLE,
    rules_version: str | None = None,
) -> str:
    """SQL producing ``stock_metric_fact`` rows from the raw lake views.

    ``rules_version`` picks the collector metric rule set (``None`` = current, the text every
    caller got before versions existed; ``"mrv1_20260818"`` = the set the KR model was
    trained under). It changes which rows exist, not how they are chosen.

    ``plan`` picks how the same rows are reached; the default text is unchanged. The one
    place the plans may differ is a tie on the whole winner key ``(priority,
    candidate_rank, source_key)`` for a ``(ticker, metric_code, bsns_year, reprt_code)``
    group, where both pick an arbitrary candidate.
    """
    if plan not in PLANS:
        raise ValueError(f"unknown stock_metric_fact plan {plan!r}; expected {PLANS}")
    split = plan == PLAN_SPLIT_ARGMIN
    # No argument for the default so a caller that substitutes the rule list keeps working.
    if rules_version is None:
        rules, catalog = default_metric_mapping_rules(), default_metric_catalog()
    else:
        rules = default_metric_mapping_rules(rules_version)
        catalog = default_metric_catalog(rules_version)
    unit_by_code = {entry.metric_code: entry.unit for entry in catalog}
    rules_rel = _rules_relation_sql(rules, unit_by_code)

    period_type_fin = _period_type_expr("f.reprt_code")
    period_type_sc = _period_type_expr("s.reprt_code")
    period_type_sr = _period_type_expr("sr.reprt_code")
    period_type_xf = _period_type_expr("x.reprt_code")

    # period_end fallbacks per source (Python ``or`` -> COALESCE).
    pe_fin = _infer_period_end_expr("f.reprt_code", "f.bsns_year")
    pe_sc = f"COALESCE(s.stlm_dt, {_infer_period_end_expr('s.reprt_code', 's.bsns_year')})"
    pe_sr = f"COALESCE(sr.stlm_dt, {_infer_period_end_expr('sr.reprt_code', 'sr.bsns_year')})"
    pe_xf = (
        "COALESCE(x.instant_date, x.period_end, "
        f"{_infer_period_end_expr('x.reprt_code', 'x.bsns_year')})"
    )

    def fin_block(rule_rel: str, lead: str) -> str:
        return f"""-- dart_financial_statement_raw (value_selector is always thstrm_amount)
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_fin} AS period_type,
            {pe_fin} AS period_end,
            f.bsns_year, f.reprt_code,
            f.fs_div AS fs_div,
            CAST(f.thstrm_amount AS DECIMAL(30,4)) AS value_numeric,
            CAST(f.thstrm_amount AS VARCHAR) AS value_text,
            r.unit,
            'dart_financial_statement_raw' AS source_table,
            concat(f.rcept_no, ':', f.account_id, ':', f.ord) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            0 AS candidate_rank
        FROM {financial_view} f
        JOIN corp c ON c.ticker = f.ticker
        JOIN {rule_rel} r
          ON {lead}r.source_table = 'dart_financial_statement_raw'
         AND (r.fs_div = '' OR f.fs_div = r.fs_div)
         AND (r.sj_div = '' OR f.sj_div = r.sj_div)
         AND (r.account_id = '' OR f.account_id = r.account_id)
         AND (r.account_nm = '' OR f.account_nm = r.account_nm)
        WHERE f.thstrm_amount IS NOT NULL"""

    def share_block(rule_rel: str, lead: str) -> str:
        return f"""-- dart_share_count_raw (BIGINT selectors -> no-decimal value_text)
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_sc} AS period_type,
            {pe_sc} AS period_end,
            s.bsns_year, s.reprt_code,
            '' AS fs_div,
            CAST(
                CASE r.value_selector
                    WHEN 'istc_totqy' THEN s.istc_totqy
                    WHEN 'tesstk_co' THEN s.tesstk_co
                END AS DECIMAL(30,4)
            ) AS value_numeric,
            CAST(
                CASE r.value_selector
                    WHEN 'istc_totqy' THEN s.istc_totqy
                    WHEN 'tesstk_co' THEN s.tesstk_co
                END AS VARCHAR
            ) AS value_text,
            r.unit,
            'dart_share_count_raw' AS source_table,
            concat(s.rcept_no, ':', s.se) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            0 AS candidate_rank
        FROM {share_count_view} s
        JOIN corp c ON c.ticker = s.ticker
        JOIN {rule_rel} r
          ON {lead}r.source_table = 'dart_share_count_raw'
         AND (r.row_name = '' OR s.se = r.row_name)
        WHERE CASE r.value_selector
                  WHEN 'istc_totqy' THEN s.istc_totqy
                  WHEN 'tesstk_co' THEN s.tesstk_co
              END IS NOT NULL"""

    def return_block(rule_rel: str, lead: str) -> str:
        return f"""-- dart_shareholder_return_raw
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_sr} AS period_type,
            {pe_sr} AS period_end,
            sr.bsns_year, sr.reprt_code,
            '' AS fs_div,
            CAST(sr.value_numeric AS DECIMAL(30,4)) AS value_numeric,
            CAST(sr.value_numeric AS VARCHAR) AS value_text,
            r.unit,
            'dart_shareholder_return_raw' AS source_table,
            concat(
                sr.rcept_no, ':', sr.statement_type, ':', sr.row_name, ':',
                sr.stock_knd, ':', sr.dim1, ':', sr.dim2, ':', sr.dim3, ':',
                sr.metric_code
            ) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            0 AS candidate_rank
        FROM {shareholder_return_view} sr
        JOIN corp c ON c.ticker = sr.ticker
        JOIN {rule_rel} r
          ON {lead}r.source_table = 'dart_shareholder_return_raw'
         AND (r.statement_type = '' OR sr.statement_type = r.statement_type)
         AND (r.row_name = '' OR sr.row_name = r.row_name)
         AND (r.stock_knd = '' OR sr.stock_knd = r.stock_knd)
         AND (r.dim1 = '' OR sr.dim1 = r.dim1)
         AND (r.dim2 = '' OR sr.dim2 = r.dim2)
         AND (r.dim3 = '' OR sr.dim3 = r.dim3)
         AND (r.metric_code_match = '' OR sr.metric_code = r.metric_code_match)
        WHERE sr.value_numeric IS NOT NULL"""

    def xbrl_block(rule_rel: str, lead: str) -> str:
        return f"""-- dart_xbrl_fact_raw (candidate_rank from dimensions)
        SELECT
            c.ticker, c.market, c.corp_code,
            r.metric_code,
            {period_type_xf} AS period_type,
            {pe_xf} AS period_end,
            x.bsns_year, x.reprt_code,
            '' AS fs_div,
            CAST(x.value_numeric AS DECIMAL(30,4)) AS value_numeric,
            CAST(x.value_numeric AS VARCHAR) AS value_text,
            r.unit,
            'dart_xbrl_fact_raw' AS source_table,
            concat(x.rcept_no, ':', x.context_id, ':', x.concept_id) AS source_key,
            r.rule_code AS mapping_rule_code,
            r.priority AS priority,
            {_XBRL_RANK_SQL} AS candidate_rank
        FROM {xbrl_view} x
        JOIN corp c ON c.ticker = x.ticker
        JOIN {rule_rel} r
          ON {lead}r.source_table = 'dart_xbrl_fact_raw'
         AND (r.account_id = '' OR x.concept_id = r.account_id)
         AND (r.account_nm = ''
              OR x.label_ko = r.account_nm
              OR x.concept_name = r.account_nm)
        WHERE x.value_numeric IS NOT NULL"""

    if split:
        blocks = [
            fin_block("rule_rel_eq_account_id", "r.account_id = f.account_id\n         AND "),
            fin_block("rule_rel_wild_account_id", ""),
            share_block("rule_rel_eq_row_name", "r.row_name = s.se\n         AND "),
            share_block("rule_rel_wild_row_name", ""),
            return_block("rule_rel_eq_row_name", "r.row_name = sr.row_name\n         AND "),
            return_block("rule_rel_wild_row_name", ""),
            xbrl_block("rule_rel_eq_account_id", "r.account_id = x.concept_id\n         AND "),
            xbrl_block("rule_rel_wild_account_id", ""),
        ]
        rule_splits = """,
    rule_rel_eq_account_id AS (SELECT * FROM rule_rel WHERE account_id <> ''),
    rule_rel_wild_account_id AS (SELECT * FROM rule_rel WHERE account_id = ''),
    rule_rel_eq_row_name AS (SELECT * FROM rule_rel WHERE row_name <> ''),
    rule_rel_wild_row_name AS (SELECT * FROM rule_rel WHERE row_name = '')"""
        winner = """SELECT
        ticker, w.market, w.corp_code, metric_code, w.period_type, w.period_end,
        bsns_year, reprt_code, w.fs_div, w.value_numeric, w.value_text, w.unit,
        w.source_table, w.source_key, w.mapping_rule_code
    FROM (
        SELECT
            ticker, metric_code, bsns_year, reprt_code,
            arg_min(
                struct_pack(market := market, corp_code := corp_code,
                            period_type := period_type, period_end := period_end,
                            fs_div := fs_div, value_numeric := value_numeric,
                            value_text := value_text, unit := unit,
                            source_table := source_table, source_key := source_key,
                            mapping_rule_code := mapping_rule_code),
                struct_pack(p := priority, r := candidate_rank, k := source_key)
            ) AS w
        FROM candidates
        GROUP BY ticker, metric_code, bsns_year, reprt_code
    )"""
    else:
        blocks = [
            fin_block("rule_rel", ""),
            share_block("rule_rel", ""),
            return_block("rule_rel", ""),
            xbrl_block("rule_rel", ""),
        ]
        rule_splits = ""
        winner = """SELECT
        ticker, market, corp_code, metric_code, period_type, period_end,
        bsns_year, reprt_code, fs_div, value_numeric, value_text, unit,
        source_table, source_key, mapping_rule_code
    FROM candidates
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY ticker, metric_code, bsns_year, reprt_code
        ORDER BY priority ASC, candidate_rank ASC, source_key ASC
    ) = 1"""
    candidates = "\n\n        UNION ALL\n        ".join(blocks)

    return f"""
    WITH corp AS (
        SELECT ticker, market, corp_code
        FROM {corp_view}
        WHERE is_active = TRUE
          AND ticker IS NOT NULL AND ticker <> ''
          AND market IS NOT NULL
    ),
    rule_rel AS (SELECT * FROM {rules_rel}){rule_splits},
    candidates AS (
        {candidates}
    )
    {winner}
    """


def is_default_stock_metric_fact(plan: str, rules_version: str | None) -> bool:
    """True for the definition every frozen research mart was written under."""
    return plan == PLAN_SINGLE and (
        rules_version is None
        or resolve_metric_rules_version(rules_version) == CURRENT_METRIC_RULES_VERSION
    )


def stock_metric_fact_plan_record(
    plan: str = PLAN_SINGLE, *, rules_version: str | None = None, **views: str
) -> dict[str, str | None]:
    """What a manifest records about how ``stock_metric_fact`` was computed.

    ``stock_metric_fact`` lives in the derived-mart lake, not the feature mart, so it has
    no ``_cache_metadata.json``. The default plan with the current rules records
    ``plan_hash = None`` like every other legacy-text mart and carries no rules keys.
    Anything else also records ``rules_version`` and ``rules_hash`` (sha256 of the rule and
    catalog contents), so a mart built under other rules can never be mistaken for it.
    """
    record: dict[str, str | None] = {
        "semantics_version": "v1", "plan": plan, "plan_hash": None,
    }
    if plan != PLAN_SINGLE:
        record["plan_hash"] = MartPlan(
            plan_id=plan,
            final_sql=build_stock_metric_fact_sql(**views, plan=plan, rules_version=rules_version),
        ).plan_hash
    if not is_default_stock_metric_fact(plan, rules_version):
        record["rules_version"] = resolve_metric_rules_version(rules_version)
        record["rules_hash"] = metric_rules_content_hash(rules_version)
    return record


def register_stock_metric_fact_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = SMF_VIEW,
    plan: str = PLAN_SINGLE,
    rules_version: str | None = None,
    **views: str,
) -> str:
    """Register a DuckDB view computing ``stock_metric_fact`` from raw views.

    Lightweight path (no parquet) for parity checks and direct consumers; the
    orchestrated pipeline materializes instead (heavy QUALIFY once per snapshot).
    """
    sql = build_stock_metric_fact_sql(**views, plan=plan, rules_version=rules_version)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name
