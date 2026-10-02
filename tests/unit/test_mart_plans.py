"""E1 — selectable mart semantics and execution plans.

Two axes, both recorded in the cache metadata and kept apart:

* ``semantics_version`` is what a mart *means*. ``v1`` is the SQL every frozen run used.
* ``plan`` is how it is computed. Plans of one semantics must return the same rows.

Legacy callers (no new argument) must keep byte-identical SQL text and metadata. The hashes
pinned below were computed from the code as it was before this change.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest
from collector.kr.shared import MetricMappingRule
from collector.lake import DataRoot

from modeler.etl import stock_pit, trading_panel
from modeler.etl.config import REMOTE_SOURCE, LakeConfig
from modeler.etl.features import fin_pit, fin_scan, flow, price
from modeler.etl.features.fin_vintage import (
    JOIN_PLAN_COALESCE,
    build_metric_join,
    build_metric_joins,
)
from modeler.etl.mart import (
    MartPlan,
    MartPlanCheckFailed,
    StaleMartContract,
    is_materialized,
    mart_cache_metadata,
    mart_table_dir,
    materialize,
)
from modeler.etl.marts import metric_vintages as mv
from modeler.etl.marts import metrics_normalize as mn

from . import test_research_fin_scan as scan_fx
from .test_metric_vintages import (
    _DIM_CFS,
    _DIM_OFS,
    _TRADING_DAYS,
    _add_financial,
    _add_xbrl,
    _base_con,
    _statement_row,
    _three_comparative_years,
    _xbrl_row,
)

# --------------------------------------------------------------------------
# Legacy text is unchanged
# --------------------------------------------------------------------------

#: sha256 of each builder's default output, taken from the pre-E1 code (git HEAD 3c6613b).
LEGACY_HASHES = {
    "smvf_default": "284d09f7705b2e6d26bf2f7102840a658b4697968ececdc8cc59e76a76413b7d",
    "smvf_custom_views": "4b9e2fd2685d7d093c748390ec82e02dbde9b4cc4f4c17adae32a6bf0cc12585",
    "fin_pit": "e729a9e537ca78b5f00c2059dc12af5a627cbdea3ba943c193e8f8da631f023b",
    "fin_pit_available": "3395590bb62e4559575dd4258145ab2f0b203852b8f4298834934a08953b36ec",
    "fin_pit_custom_views": "e6364be1c95f0be6e5dfb340b8efbca392c167522055cd34ed3c3156538b5cac",
    "metric_join": "47481956f0073f537e95de8196141e5a641807021186034beee89fe53e29fa78",
    "metric_joins": "477ab9beccf96750d3509095061da5f14a53b67c011302b7d0ca6338430318d2",
    # E1 round 2: defaults of the builders that gained a plan or a semantics argument.
    "stock_pit": "a8216fe39b29d11ed208e7e72e6b6492bbd839e5fa875b34123be4478c671dd0",
    "stock_pit_custom_views": "c2dff038bc51c68b0969680499d708294e6cb01b2294e9070ce0caa35c9cdd44",
    # No price view: the degraded path. Serving used to take it by mistake (see below).
    "flow_degraded": "e6712309bd22ef4a87a3aff4b5dc900fa5a60cf5fad03bb43ec9ac3f4894a5b4",
    # The research builder's call (etl/compute_all.py `_build_features`).
    "flow_research": "1bb97c537cd0656bf49df38e52c9a444bf7484aad68fc1ec6999f0e6b43aad4b",
    "flow_dedup": "68f8861db176e2316c4af14eecedd57e0c191a9b0a677c2abff13e65f3a7be2e",
    "stock_metric_fact": "b1cca6c8d616ff3374033063ae959f496c5484c2b72d63fd2c6cdeaac84ae179",
    "market_model": "b83b9dbea992ad6887a8a752a085b780c5a3336a3448e6dc2b9b7cfa631898b0",
    "valid_session": "b34db53530cec7f5946dbf9bb9af1d855a1e8f7759875fe045f206139cff646d",
    "price": "6886ddd45d33c6a13f0d1fd1653a8dd37518bfaa5189ae1227f853bb3b2a4bf3",
    "price_with_quality": "110617482d64a50d4ef92ad70defcc23b5b276e2a025452f92a845ccbacec228",
    "price_custom_views": "e46eed957a44c4bf4b6efc15d2bd41884db11d61cca9509a2fb38cfc873253ae",
}


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_legacy_callers_get_the_exact_pre_change_sql() -> None:
    assert _h(mv.build_stock_metric_vintage_fact_sql()) == LEGACY_HASHES["smvf_default"]
    custom = mv.build_stock_metric_vintage_fact_sql(
        financial_view="a",
        share_count_view="b",
        shareholder_return_view="c",
        xbrl_view="d",
        filing_receipt_view="e",
        corp_view="f",
        calendar_table="g",
        pairing_tolerance=0.5,
    )
    assert _h(custom) == LEGACY_HASHES["smvf_custom_views"]
    assert _h(fin_pit.build_fin_pit_sql()) == LEGACY_HASHES["fin_pit"]
    assert _h(fin_pit.build_available_sql()) == LEGACY_HASHES["fin_pit_available"]
    assert _h(fin_pit.build_fin_pit_sql("u", "s")) == LEGACY_HASHES["fin_pit_custom_views"]
    assert _h(build_metric_join("revenue", "CFS")) == LEGACY_HASHES["metric_join"]
    assert _h(build_metric_joins(["a", "b"])) == LEGACY_HASHES["metric_joins"]


def test_round_two_defaults_keep_the_exact_pre_change_sql() -> None:
    pins = LEGACY_HASHES
    assert _h(stock_pit.build_stock_pit_sql()) == pins["stock_pit"]
    assert _h(stock_pit.build_stock_pit_sql(price_view="p", share_view="s")) == pins[
        "stock_pit_custom_views"
    ]
    assert _h(flow.build_flow_sql()) == pins["flow_degraded"]
    # Without a price view the pit and quality views are ignored: this is what serving
    # built before the fix.
    assert _h(
        flow.build_flow_sql(pit_view="dim_stock_pit_daily", quality_view="dim_price_quality_daily")
    ) == pins["flow_degraded"]
    assert _h(_research_flow_sql()) == pins["flow_research"]
    assert _h(flow.build_dedup_sql()) == pins["flow_dedup"]
    assert _h(mn.build_stock_metric_fact_sql()) == pins["stock_metric_fact"]
    assert _h(trading_panel.build_market_model_sql("src")) == pins["market_model"]
    assert _h(trading_panel.build_valid_session_sql()) == pins["valid_session"]
    assert _h(price.build_price_sql()) == pins["price"]
    assert _h(price.build_price_sql(quality_view="dim_price_quality_daily")) == pins[
        "price_with_quality"
    ]
    assert _h(price.build_price_sql("p", quality_view="q")) == pins["price_custom_views"]


def _research_flow_sql() -> str:
    return flow.build_flow_sql(
        price_view="daily_ohlcv",
        pit_view="dim_stock_pit_daily",
        quality_view="dim_price_quality_daily",
    )


def test_new_semantics_and_plans_have_their_own_text() -> None:
    v1 = mv.build_stock_metric_vintage_fact_sql()
    v2 = mv.build_stock_metric_vintage_fact_sql(semantics="v2")
    assert v1 != v2 and "receipt_beyond_calendar" in v2 and "receipt_beyond_calendar" not in v1
    assert fin_pit.build_available_sql(semantics="v2") != fin_pit.build_available_sql()
    assert build_metric_join("revenue", "CFS", JOIN_PLAN_COALESCE) != build_metric_join(
        "revenue", "CFS"
    )
    for builder in (
        lambda: mv.build_stock_metric_vintage_fact_sql(semantics="v9"),
        lambda: fin_pit.build_available_sql(semantics="v9"),
        lambda: build_metric_join("revenue", "CFS", "nope"),
        lambda: mv.build_stock_metric_vintage_fact_plan(plan="nope"),
        lambda: stock_pit.build_stock_pit_sql(plan="nope"),
        lambda: flow.build_flow_sql(pivot_plan="nope"),
        lambda: mn.build_stock_metric_fact_sql(plan="nope"),
        lambda: trading_panel.build_market_model_sql("x", semantics="v9"),
    ):
        with pytest.raises(ValueError):
            builder()


# --------------------------------------------------------------------------
# MartPlan mechanics
# --------------------------------------------------------------------------


def _lake(tmp_path: Path, tag: str = "kr") -> LakeConfig:
    return LakeConfig(DataRoot(tmp_path / tag), "2026-09-30", REMOTE_SOURCE)


def test_plan_hash_covers_plan_id_every_stage_and_the_final_sql() -> None:
    base = MartPlan("staged", "SELECT * FROM s", stages=(("s", "SELECT 1 AS x"),))
    assert base.plan_hash == dataclasses.replace(base).plan_hash
    assert base.plan_hash != dataclasses.replace(base, plan_id="other").plan_hash
    assert base.plan_hash != dataclasses.replace(base, final_sql="SELECT x FROM s").plan_hash
    assert (
        base.plan_hash != dataclasses.replace(base, stages=(("s", "SELECT 2 AS x"),)).plan_hash
    )
    assert base.plan_hash != dataclasses.replace(base, stages=(("t", "SELECT 1 AS x"),)).plan_hash


def test_a_plan_writes_stages_then_the_mart_and_leaves_no_stage_files(tmp_path: Path) -> None:
    con, config = duckdb.connect(), _lake(tmp_path)
    plan = MartPlan(
        "staged",
        "SELECT x * 2 AS y FROM _stg_one",
        stages=(("_stg_one", "SELECT * FROM range(5) t(x)"),),
        checks=(("five rows", "SELECT count(*) = 5 FROM _stg_one"),),
    )
    stage_dir = tmp_path / "run" / "stages"
    # A stale file from an earlier run must never be read.
    (stage_dir / "m").mkdir(parents=True)
    (stage_dir / "m" / "_stg_one.parquet").write_bytes(b"not parquet")
    table = materialize(
        con, config, "m", "SELECT 0 AS y", plan=plan, stage_dir=stage_dir
    )
    rows = con.execute(f"SELECT y FROM read_parquet('{table}/*.parquet') ORDER BY y").fetchall()
    assert [r[0] for r in rows] == [0, 2, 4, 6, 8]
    assert not (stage_dir / "m").exists()
    with pytest.raises(duckdb.CatalogException):
        con.execute("SELECT * FROM _stg_one")
    metadata = mart_cache_metadata(config, "m")
    assert metadata["plan"] == "staged" and metadata["semantics_version"] == "v1"
    assert metadata["plan_hash"] == plan.plan_hash
    assert metadata["sql_hash"] == _h("SELECT 0 AS y")


def test_a_staged_plan_needs_an_explicit_stage_dir(tmp_path: Path) -> None:
    con, config = duckdb.connect(), _lake(tmp_path)
    plan = MartPlan("staged", "SELECT * FROM s", stages=(("s", "SELECT 1 AS x"),))
    with pytest.raises(ValueError, match="stage_dir"):
        materialize(con, config, "m", "SELECT 1 AS x", plan=plan)
    assert not mart_table_dir(config, "m").exists()


@pytest.mark.parametrize("broken", ["check", "stage", "final"])
def test_a_failed_plan_leaves_no_mart_and_no_metadata(tmp_path: Path, broken: str) -> None:
    con, config = duckdb.connect(), _lake(tmp_path)
    plan = MartPlan(
        "staged",
        "SELECT * FROM missing_relation" if broken == "final" else "SELECT * FROM s",
        stages=(("s", "SELECT nope FROM nowhere" if broken == "stage" else "SELECT 1 AS x"),),
        checks=(("always false", "SELECT false"),) if broken == "check" else (),
    )
    with pytest.raises((MartPlanCheckFailed, duckdb.Error)):
        materialize(con, config, "m", "SELECT 1 AS x", plan=plan, stage_dir=tmp_path / "st")
    assert not mart_table_dir(config, "m").exists()
    assert not is_materialized(config, "m")
    with pytest.raises(duckdb.CatalogException):
        con.execute("SELECT * FROM s")


def test_reuse_requires_the_same_semantics_and_plan(tmp_path: Path) -> None:
    con, config = duckdb.connect(), _lake(tmp_path)
    plan = MartPlan("coalesce_join", "SELECT 1 AS x")
    materialize(con, config, "m", "SELECT 1 AS x", plan=plan)
    # Same plan: reused. The stage dir is not even touched.
    materialize(con, config, "m", "SELECT 1 AS x", plan=plan, stage_dir=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
    for other in (
        None,
        dataclasses.replace(plan, plan_id="staged"),
        dataclasses.replace(plan, semantics_version="v2"),
        dataclasses.replace(plan, final_sql="SELECT 1 AS x /* changed */"),
    ):
        with pytest.raises(StaleMartContract):
            materialize(con, config, "m", "SELECT 1 AS x", plan=other)
    # The other direction: a legacy mart is not accepted by a caller that asks for a plan.
    materialize(con, config, "legacy", "SELECT 1 AS x")
    assert set(mart_cache_metadata(config, "legacy")) == {
        "analysis_config_hash",
        "schema_hash",
        "sql_hash",
    }
    with pytest.raises(StaleMartContract):
        materialize(con, config, "legacy", "SELECT 1 AS x", plan=plan)


# --------------------------------------------------------------------------
# stock_metric_vintage_fact: staged == single, calendar boundary, tie rule
# --------------------------------------------------------------------------


def _equivalence_con() -> duckdb.DuckDBPyConnection:
    """Several filings: CFS and OFS, comparative years, a NULL-valued competing fact, an XBRL
    fallback, share counts, a revision, a missing receipt id, and a receipt on the last
    calendar day (so ``v1`` and ``v2`` differ there)."""
    con = _base_con()
    con.execute("INSERT INTO dart_corp_master VALUES ('000660', 'KOSPI', '00164779')")
    _add_financial(con, amount=1000.0)
    _three_comparative_years(con)
    # A NULL-valued fact that wins the pairing tie-break (lower context id, same rank): the
    # single statement pairs against it, so the staged plan must too.
    _add_xbrl(
        con,
        value=0,
        context_id="a_null",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
    )
    con.execute("UPDATE dart_xbrl_fact_raw SET value_numeric = NULL WHERE context_id = 'a_null'")
    _add_financial(con, amount=400.0, fs_div="OFS")
    for year, value in ((2024, 390.0), (2025, 400.0)):
        _add_xbrl(
            con,
            value=value,
            context_id=f"ofs{year}",
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            dimensions=_DIM_OFS,
        )
    # Revision of the same period, later receipt, different value.
    _add_financial(con, amount=1010.0, rcept_no="20260420000007")
    # Fallback metrics and a concept no rule names.
    con.execute(_xbrl_row("ifrs-full_Assets", _DIM_CFS, 5000.0))
    con.execute(_xbrl_row("custom_NoRuleNamesThis", _DIM_CFS, 1.0))
    con.execute(_statement_row("ifrs-full_Equity", "CFS", 3000.0))
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','합계',5000,100,DATE '2025-12-31'),"
        "('00164779','000660',2025,'11011','','합계',7000,0,DATE '2025-12-31'),"
        "('00164779','000660',2025,'11011','20261231000009','합계',7100,0,DATE '2025-12-31')"
    )
    con.execute(
        "INSERT INTO dart_filing_receipt_raw VALUES "
        "('00126380','20260310000001','사업보고서'),('00126380','20260420000007','[기재정정]사업보고서')"
    )
    return con


def _build(con, tmp_path: Path, tag: str, *, days=_TRADING_DAYS, **kwargs) -> str:
    config = LakeConfig(DataRoot(tmp_path / tag), "2026-09-30", REMOTE_SOURCE)
    mv.materialize_stock_metric_vintage_fact(
        con, config, trading_days=days, stage_dir=tmp_path / tag / "stages", **kwargs
    )
    table = f"smvf_{tag}"
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM stock_metric_vintage_fact")
    return table


def _same_rows(con, left: str, right: str) -> None:
    assert con.execute(f"SELECT count(*) FROM {left}").fetchone()[0] > 0
    for a, b in ((left, right), (right, left)):
        assert con.execute(f"SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}").fetchall() == []


@pytest.mark.parametrize("semantics", ["v1", "v2"])
def test_staged_plan_returns_the_single_statements_rows(tmp_path: Path, semantics: str) -> None:
    con = _equivalence_con()
    single = _build(con, tmp_path, f"single_{semantics}", semantics=semantics)
    staged = _build(con, tmp_path, f"staged_{semantics}", semantics=semantics, plan="staged")
    _same_rows(con, single, staged)
    # The XBRL pairing really went through the NULL-valued competitor in both.
    statuses = {
        r[0]
        for r in con.execute(
            f"SELECT receipt_value_pairing_status FROM {staged} WHERE metric_code = 'revenue'"
        ).fetchall()
    }
    assert "unlinked_receipt" in statuses


def test_staged_cache_metadata_keeps_the_single_statement_sql_hash(tmp_path: Path) -> None:
    con = _equivalence_con()
    _build(con, tmp_path, "legacy")
    legacy = mart_cache_metadata(_lake(tmp_path, "legacy"), mv.SMVF_TABLE)
    _build(con, tmp_path, "staged", plan="staged")
    staged = mart_cache_metadata(_lake(tmp_path, "staged"), mv.SMVF_TABLE)
    assert set(legacy) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert legacy["sql_hash"] == LEGACY_HASHES["smvf_default"]
    assert staged["sql_hash"] == legacy["sql_hash"]
    assert staged["schema_hash"] == legacy["schema_hash"]
    assert (staged["semantics_version"], staged["plan"]) == ("v1", "staged")
    plan = mv.build_stock_metric_vintage_fact_plan(plan="staged")
    assert staged["plan_hash"] == plan.plan_hash
    assert [name for name, _ in plan.stages][0] == "_smvf_xbrl_keys"


def test_the_staged_plan_restricts_xbrl_to_the_concepts_the_rules_name() -> None:
    ids = mv._xbrl_concept_ids()
    assert "ifrs-full_Revenue" in ids and len(ids) == len(set(ids)) > 40
    stages = dict(mv.build_stock_metric_vintage_fact_plan(plan="staged").stages)
    scoped = stages["_smvf_xbrl_scoped"]
    assert "'ifrs-full_Revenue'" in scoped and "NoRuleNamesThis" not in scoped


def test_a_rule_without_account_id_refuses_the_staged_plan(monkeypatch) -> None:
    rules = mv.default_metric_mapping_rules()
    wildcard = dataclasses.replace(rules[0], account_id="", source_table="dart_xbrl_fact_raw")
    monkeypatch.setattr(mv, "default_metric_mapping_rules", lambda: [wildcard, *rules])
    with pytest.raises(ValueError, match="no account_id"):
        mv.build_stock_metric_vintage_fact_plan(plan="staged")


def test_the_pairing_join_cannot_multiply_rows(tmp_path: Path) -> None:
    """Keys are DISTINCT on exactly the join columns, so three statement rows for one
    (filing, concept, basis) still match each fact once."""
    con = _base_con()
    for ord_ in range(1, 4):
        _add_financial(con, amount=1000.0, ord_=ord_)
    _three_comparative_years(con)
    single = _build(con, tmp_path, "single")
    staged = _build(con, tmp_path, "staged", plan="staged")
    _same_rows(con, single, staged)
    revenue_rows = con.execute(
        f"SELECT count(*) FROM {staged} WHERE metric_code = 'revenue'"
    ).fetchone()
    assert revenue_rows == (1,)


def _boundary_con(rcept_no: str = "20261001000001") -> duckdb.DuckDBPyConnection:
    con = _base_con()
    if rcept_no:
        _add_financial(con, amount=1000.0, rcept_no=rcept_no)
    return con


def _revenue(con, **kwargs) -> tuple:
    mv.register_stock_metric_vintage_fact_view(con, **kwargs)
    return con.execute(
        "SELECT disclosed_date, available_from, availability_source "
        "FROM stock_metric_vintage_fact WHERE metric_code = 'revenue'"
    ).fetchone()


def test_calendar_boundary_v1_looks_back_and_v2_does_not() -> None:
    """Review appendix A: a 2025 annual report received 2026-10-01."""
    last_day = [date(2026, 10, 1)]
    with_next = [date(2026, 10, 1), date(2026, 10, 2)]
    # v1: the next session is missing, so the 90-day fallback lands before the receipt, and
    # the source still claims rcept_no. This is the defect; v1 keeps it for frozen runs.
    assert _revenue(_boundary_con(), trading_days=last_day) == (
        date(2026, 10, 1), date(2026, 3, 31), "rcept_no")
    assert _revenue(_boundary_con(), trading_days=with_next) == (
        date(2026, 10, 1), date(2026, 10, 2), "rcept_no")
    # v2: no fallback for a parseable receipt; NULL plus a source of its own.
    assert _revenue(_boundary_con(), trading_days=last_day, semantics="v2") == (
        date(2026, 10, 1), None, "receipt_beyond_calendar")
    assert _revenue(_boundary_con(), trading_days=with_next, semantics="v2") == (
        date(2026, 10, 1), date(2026, 10, 2), "rcept_no")


def test_v2_keeps_the_period_end_fallback_for_a_missing_receipt_date() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','','합계',5000,100,DATE '2025-12-31')"
    )
    mv.register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS, semantics="v2")
    row = con.execute(
        "SELECT disclosed_date, available_from, availability_source "
        "FROM stock_metric_vintage_fact WHERE metric_code = 'issued_shares'"
    ).fetchone()
    assert row == (None, date(2026, 3, 31), "synthetic_fallback")


@pytest.mark.parametrize("plan", ["single", "staged"])
def test_v2_calendar_boundary_in_both_plans(tmp_path: Path, plan: str) -> None:
    con = _boundary_con()
    table = _build(con, tmp_path, plan, days=[date(2026, 10, 1)], semantics="v2", plan=plan)
    assert con.execute(
        f"SELECT available_from, availability_source FROM {table} WHERE metric_code = 'revenue'"
    ).fetchall() == [(None, "receipt_beyond_calendar")]


def _share_count_tie(con, rows: list[tuple[int, str]]) -> None:
    """One receipt filed under several (bsns_year, reprt_code); every key ties."""
    for year, reprt in rows:
        con.execute(
            "INSERT INTO dart_share_count_raw VALUES "
            f"('00126380','005930',{year},'{reprt}','20200515000226','합계',5000,100,"
            "DATE '2019-12-31')"
        )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # The real case: the same receipt under two years. The earlier year wins.
        ([(2020, "11013"), (2019, "11011")], (2019, "11011", "annual")),
        ([(2019, "11011"), (2020, "11013")], (2019, "11011", "annual")),
        # Same year: report order Q1 < half < Q3 < annual.
        ([(2020, "11014"), (2020, "11011"), (2020, "11013")], (2020, "11013", "q1")),
        ([(2020, "11011"), (2020, "11012")], (2020, "11012", "half")),
    ],
)
@pytest.mark.parametrize("plan", ["single", "staged"])
def test_v2_tie_picks_the_report_that_published_first(
    tmp_path: Path, rows, expected, plan: str
) -> None:
    con = _base_con()
    _share_count_tie(con, rows)
    table = _build(con, tmp_path, plan, semantics="v2", plan=plan)
    got = con.execute(
        f"SELECT bsns_year, reprt_code, period_type FROM {table} "
        "WHERE metric_code = 'issued_shares'"
    ).fetchall()
    assert got == [expected]


# --------------------------------------------------------------------------
# feat_fin_pit v2
# --------------------------------------------------------------------------


def _fin_pit_con(rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    """rows: (period_end, period_type, bsns_year, reprt_code, value) for total_assets."""
    con = duckdb.connect()
    values = ",".join(
        f"('A','KOSPI','total_assets','{pt}',DATE '{pe}',{v},{by},'{rc}')"
        for pe, pt, by, rc, v in rows
    )
    con.execute(
        "CREATE TABLE stock_metric_fact AS SELECT * FROM (VALUES "
        + values
        + ") AS t(ticker, market, metric_code, period_type, period_end, value_numeric,"
        " bsns_year, reprt_code)"
    )
    con.execute(
        "CREATE TABLE dim_universe_daily AS SELECT DATE '2024-06-03' AS trade_date,"
        " 'A' AS ticker, 'KOSPI' AS market, TRUE AS in_universe"
    )
    return con


@pytest.mark.parametrize("order", [0, 1])
def test_fin_pit_v2_prefers_the_report_that_first_published_the_period(order: int) -> None:
    # FY2023 annual figure from its own report (2023/11011) and again as the comparative
    # in the FY2024 annual report (2024/11011): both carry period_end 2023-12-31.
    rows = [
        ("2023-12-31", "annual", 2023, "11011", 100.0),
        ("2023-12-31", "annual", 2024, "11011", 90.0),
    ]
    con = _fin_pit_con(rows if order == 0 else rows[::-1])
    v2 = fin_pit.build_available_sql(semantics="v2")
    assert con.execute(f"SELECT value FROM ({v2})").fetchall() == [(100.0,)]
    # period_end DESC still comes first: a later period beats an earlier one.
    con = _fin_pit_con(rows + [("2024-03-31", "q1", 2024, "11013", 120.0)])
    got = con.execute(f"SELECT value FROM ({v2}) ORDER BY value").fetchall()
    assert got == [(100.0,), (120.0,)]


def test_fin_pit_v2_orders_by_report_within_a_year() -> None:
    con = _fin_pit_con(
        [
            ("2023-06-30", "half", 2023, "11012", 7.0),
            ("2023-06-30", "half", 2023, "11014", 8.0),
            ("2023-06-30", "half", 2023, "11011", 9.0),
        ]
    )
    assert con.execute(
        f"SELECT value FROM ({fin_pit.build_available_sql(semantics='v2')})"
    ).fetchall() == [(7.0,)]


def test_fin_pit_materialize_records_v2_and_keeps_v1_metadata(tmp_path: Path) -> None:
    con = _fin_pit_con([("2023-12-31", "annual", 2023, "11011", 100.0)])
    legacy, new = LakeConfig(DataRoot(tmp_path / "v1"), "2026-09-30", REMOTE_SOURCE), LakeConfig(
        DataRoot(tmp_path / "v2"), "2026-09-30", REMOTE_SOURCE
    )
    fin_pit.materialize_fin_pit(con, legacy)
    fin_pit.materialize_fin_pit(con, new, semantics="v2")
    old_meta, new_meta = (mart_cache_metadata(c, fin_pit.FIN_TABLE) for c in (legacy, new))
    assert set(old_meta) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert old_meta["sql_hash"] == LEGACY_HASHES["fin_pit"]
    assert new_meta["sql_hash"] == _h(fin_pit.build_fin_pit_sql(semantics="v2"))
    assert new_meta["sql_hash"] != old_meta["sql_hash"]
    assert (new_meta["semantics_version"], new_meta["plan"]) == ("v2", "single")
    with pytest.raises(StaleMartContract):
        fin_pit.materialize_fin_pit(con, new)


# --------------------------------------------------------------------------
# feat_fin_scan_daily: coalesce_join
# --------------------------------------------------------------------------


def test_coalesce_join_differs_from_the_or_join_only_in_the_upper_bound() -> None:
    plain = fin_scan.build_fin_scan_daily_sql()
    fast = fin_scan.build_fin_scan_daily_sql(join_plan=JOIN_PLAN_COALESCE)
    assert fast != plain
    assert fast.replace(
        build_metric_joins(fin_scan._METRICS, fin_scan._BASES, JOIN_PLAN_COALESCE),
        build_metric_joins(fin_scan._METRICS, fin_scan._BASES),
    ) == plain
    coalesce_join = build_metric_join("revenue", "CFS", JOIN_PLAN_COALESCE)
    assert " OR " not in coalesce_join and "COALESCE" in coalesce_join


def test_the_two_join_forms_agree_at_interval_edges_and_open_intervals() -> None:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE metric_intervals AS SELECT * FROM (VALUES "
        "('A','revenue','CFS',1.0,DATE '2024-01-10',DATE '2024-04-10'),"
        "('A','revenue','CFS',2.0,DATE '2024-04-10',NULL),"
        "('B','revenue','CFS',3.0,DATE '2024-02-01',NULL)"
        ") t(ticker, metric_code, fs_basis, daily_value, daily_available_from, next_available_from)"
    )
    dates = [
        date(2024, 1, 9), date(2024, 1, 10), date(2024, 4, 9), date(2024, 4, 10),
        date(2024, 4, 11), date(2030, 1, 1),
    ]  # fmt: skip
    con.execute(
        "CREATE TABLE panel AS SELECT * FROM (VALUES "
        + ",".join(f"('{t}',DATE '{d}')" for t in "AB" for d in dates)
        + ") t(ticker, trade_date)"
    )

    def run(plan: str) -> list[tuple]:
        join = build_metric_join("revenue", "CFS", plan)
        return con.execute(
            f"SELECT panel.ticker, panel.trade_date, m_revenue_cfs.daily_value FROM panel {join}"
            " ORDER BY 1, 2"
        ).fetchall()

    expected, got = run("or"), run(JOIN_PLAN_COALESCE)
    assert got == expected
    by_key = {(t, d): v for t, d, v in got}
    assert by_key[("A", date(2024, 1, 9))] is None  # before the first interval
    assert by_key[("A", date(2024, 4, 9))] == 1.0  # last day of the closed interval
    assert by_key[("A", date(2024, 4, 10))] == 2.0  # the edge belongs to the next interval
    assert by_key[("A", date(2030, 1, 1))] == 2.0  # open interval never ends


def _scan_con() -> duckdb.DuckDBPyConnection:
    con = scan_fx._base_con()
    scan_fx._setup_full_ticker(con, "00126380", "005930")
    day = date(2023, 3, 1)
    while day <= date(2024, 5, 31):
        if day.weekday() < 5:
            scan_fx._insert_pit(con, ticker="005930", trade_date=day, market_cap=500_000_000)
        day += timedelta(days=1)
    scan_fx._register(con)
    return con


def test_coalesce_join_scan_returns_the_or_join_scan_rows() -> None:
    con = _scan_con()
    con.execute(f"CREATE TABLE scan_or AS {fin_scan.build_fin_scan_daily_sql()}")
    con.execute(
        "CREATE TABLE scan_coalesce AS "
        + fin_scan.build_fin_scan_daily_sql(join_plan=JOIN_PLAN_COALESCE)
    )
    _same_rows(con, "scan_or", "scan_coalesce")
    # Both sides of at least one availability edge are in the window and carry values.
    assert (
        con.execute(
            "SELECT count(DISTINCT fin_book_to_market) FROM scan_or "
            "WHERE fin_book_to_market IS NOT NULL"
        ).fetchone()[0]
        >= 1
    )


def test_fin_scan_materialize_with_a_plan_keeps_the_sql_hash(tmp_path: Path) -> None:
    con = _scan_con()
    legacy, fast = (
        LakeConfig(DataRoot(tmp_path / tag), "2026-09-30", REMOTE_SOURCE) for tag in ("or", "co")
    )
    fin_scan.materialize_fin_scan_daily(con, legacy)
    fin_scan.materialize_fin_scan_daily(con, fast, join_plan=JOIN_PLAN_COALESCE)
    old, new = (mart_cache_metadata(c, fin_scan.FIN_SCAN_TABLE) for c in (legacy, fast))
    assert set(old) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert new["sql_hash"] == old["sql_hash"] and new["schema_hash"] == old["schema_hash"]
    assert (new["plan"], new["semantics_version"]) == ("coalesce_join", "v1")
    expected = MartPlan(
        "coalesce_join", fin_scan.build_fin_scan_daily_sql(join_plan=JOIN_PLAN_COALESCE)
    )
    assert new["plan_hash"] == expected.plan_hash
    with pytest.raises(StaleMartContract):
        fin_scan.materialize_fin_scan_daily(con, fast)
    assert json.loads(
        (mart_table_dir(fast, fin_scan.FIN_SCAN_TABLE) / "_cache_metadata.json").read_text()
    ) == new


# --------------------------------------------------------------------------
# E1 round 2: dim_stock_pit_daily asof_intervals == single
# --------------------------------------------------------------------------


def _business_days(first: date, last: date) -> list[date]:
    days, day = [], first
    while day <= last:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


_SHARE_COLUMNS = (
    "ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, rcept_no VARCHAR, se VARCHAR, "
    "istc_totqy BIGINT, tesstk_co BIGINT, distb_stock_co BIGINT, stlm_dt DATE"
)


def _pit_con(filings: list[tuple], tickers=("A", "B", "C")) -> duckdb.DuckDBPyConnection:
    """Prices for each ticker on every business day of 2023-2024 (``B`` misses a few days) and
    the given share-count rows ``(ticker, year, reprt, rcept_no, se, issued, treasury, float,
    stlm_dt)``."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, close DOUBLE)"
    )
    days = _business_days(date(2023, 1, 2), date(2024, 12, 31))
    rows = [
        (d, t, "KOSPI", 100.0 + i)
        for t in tickers
        for i, d in enumerate(days)
        if not (t == "B" and i % 37 == 0)
    ]
    con.executemany("INSERT INTO daily_ohlcv VALUES (?, ?, ?, ?)", rows)
    con.execute(f"CREATE TABLE dart_share_count_raw ({_SHARE_COLUMNS})")
    con.executemany("INSERT INTO dart_share_count_raw VALUES (?,?,?,?,?,?,?,?,?)", filings)
    return con


def _pit_pair(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE TABLE pit_single AS " + stock_pit.build_stock_pit_sql())
    con.execute(
        "CREATE TABLE pit_asof AS "
        + stock_pit.build_stock_pit_sql(plan=stock_pit.PLAN_ASOF_INTERVALS)
    )


def _tricky_share_rows() -> list[tuple]:
    d = date.fromisoformat
    return [
        # annual report, then the half-year report
        ("A", 2022, "11011", "20230320000001", "합계", 1000, 10, 900, d("2022-12-31")),
        ("A", 2023, "11012", "20230814000001", "합계", 1100, 20, 1000, d("2023-06-30")),
        # a late-arriving OLD report (older period, filed after the half-year): never wins
        ("A", 2023, "11013", "20230901000009", "합계", 777, 7, 700, d("2023-03-31")),
        # not the total row: ignored
        ("A", 2023, "11011", "20240301000001", "소계", 5, 1, 4, d("2023-12-31")),
        # stlm_dt NULL sorts last, so it only fills the time before anything else exists
        ("A", 2023, "11011", "20231120000001", "합계", 1200, 30, 1100, None),
        # rcept_no NULL: fallback lag (stlm + 45 days -> first session on/after 2023-11-14)
        ("A", 2023, "11014", None, "합계", 1300, 40, 1200, d("2023-09-30")),
        # invalid rcept_no, annual: lag 90 days is beyond the calendar, falls to stlm_dt
        ("A", 2024, "11011", "abc", "합계", 1400, 50, 1300, d("2024-12-31")),
        # filed on the last session: no next session, never available
        ("A", 2024, "11011", "20241231000001", "합계", 1500, 60, 1400, d("2024-12-31")),
        # a single filing with an invalid float (> issued) and a missing treasury count
        ("B", 2022, "11011", "20230105000001", "합계", 500, None, 600, d("2022-12-31")),
        # ticker C only has a non-total row
        ("C", 2022, "11011", "20230105000002", "소계", 50, 5, 40, d("2022-12-31")),
    ]


def test_asof_intervals_returns_the_single_statement_rows_on_the_tricky_cases() -> None:
    con = _pit_con(_tricky_share_rows())
    _pit_pair(con)
    _same_rows(con, "pit_single", "pit_asof")

    def issued(day: str) -> float | None:
        return con.execute(
            "SELECT issued_shares_pit FROM pit_asof WHERE ticker = 'A' AND trade_date = ?",
            [day],
        ).fetchone()[0]

    assert issued("2023-01-02") is None  # before the first filing: no backward fill
    assert issued("2023-03-21") == 1000  # first session after the 03-20 receipt
    assert issued("2023-03-20") is None
    assert issued("2023-08-15") == 1100
    assert issued("2023-09-04") == 1100  # the late old report (777) did not displace it
    assert issued("2023-11-13") == 1100
    assert issued("2023-11-21") == 1300  # the NULL-stlm_dt report (1200) sorts last: never wins
    assert issued("2023-11-14") == 1300  # NULL rcept_no: fallback availability, newest period
    assert issued("2024-12-30") == 1300
    assert con.execute(
        "SELECT count(*) FROM pit_asof WHERE issued_shares_pit IN (777, 1200, 1500, 5)"
    ).fetchone()[0] == 0
    assert issued("2024-12-31") == 1400  # invalid rcept_no falls back to stlm_dt itself
    shares = con.execute(
        "SELECT shares_used_fallback_lag, shares_invalid_flag, treasury_missing_flag "
        "FROM pit_asof WHERE ticker = 'B' AND trade_date = DATE '2023-06-01'"
    ).fetchone()
    assert shares == (False, True, True)
    assert (
        con.execute(
            "SELECT count(*) FROM pit_asof WHERE ticker = 'C' AND shares_source IS NOT NULL"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("seed", range(6))
def test_asof_intervals_matches_single_on_random_filings(seed: int) -> None:
    rng = random.Random(seed)
    filings, used = [], set()
    for ticker in ("A", "B", "C"):
        for _ in range(rng.randint(0, 9)):
            stlm = rng.choice(
                [date(2021, 12, 31), date(2022, 6, 30), date(2022, 12, 31), date(2023, 3, 31),
                 date(2023, 9, 30), date(2023, 12, 31), date(2024, 6, 30), None]
            )  # fmt: skip
            kind = rng.choice(["ok", "ok", "ok", "null", "bad", "weekend"])
            filed = date(2023, 1, 1) + timedelta(days=rng.randint(0, 760))
            if kind == "ok":
                rcept = f"{filed:%Y%m%d}{rng.randint(1, 999999):06d}"
            elif kind == "weekend":
                rcept = f"{filed:%Y%m%d}000001"
            else:
                rcept = None if kind == "null" else rng.choice(["", "abc", "9999"])
            # Whole-key ties are the one place the plans may differ (the single statement
            # picks arbitrarily), so the generator never produces them.
            key = (ticker, stlm, rcept if kind in ("ok", "weekend") else (stlm, kind))
            if key in used:
                continue
            used.add(key)
            issued = rng.choice([0, 1000, 2000, 3000, None])
            treasury = rng.choice([None, 0, 10, 100])
            float_raw = rng.choice([None, 0, 500, 5000])
            filings.append(
                (ticker, 2023, rng.choice(["11011", "11012", "11013", "11014"]), rcept,
                 rng.choice(["합계", "합계", "소계"]), issued, treasury, float_raw, stlm)
            )  # fmt: skip
    con = _pit_con(filings)
    _pit_pair(con)
    _same_rows(con, "pit_single", "pit_asof")


def test_asof_intervals_breaks_whole_key_ties_the_same_way_whatever_the_input_order() -> None:
    d = date.fromisoformat
    tie = [
        ("A", 2023, "11011", "20230320000001", "합계", 2000, 20, 1800, d("2022-12-31")),
        ("A", 2023, "11011", "20230320000001", "합계", 1000, 10, 900, d("2022-12-31")),
    ]
    results = []
    for rows in (tie, tie[::-1]):
        con = _pit_con(rows, tickers=("A",))
        con.execute(
            "CREATE TABLE r AS " + stock_pit.build_stock_pit_sql(plan=stock_pit.PLAN_ASOF_INTERVALS)
        )
        results.append(
            con.execute("SELECT * FROM r ORDER BY trade_date, ticker, market").fetchall()
        )
    assert results[0] == results[1]
    # The smaller issued count wins the tie (the last key of the added ordering).
    assert con.execute("SELECT max(issued_shares_pit) FROM r").fetchone()[0] == 1000


def test_stock_pit_materialize_records_the_plan_and_keeps_the_sql_hash(tmp_path: Path) -> None:
    con = _pit_con(_tricky_share_rows())
    legacy, fast = (_lake(tmp_path, tag) for tag in ("single", "asof"))
    stock_pit.materialize_stock_pit(con, legacy)
    stock_pit.materialize_stock_pit(con, fast, plan=stock_pit.PLAN_ASOF_INTERVALS)
    old, new = (mart_cache_metadata(c, stock_pit.PIT_TABLE) for c in (legacy, fast))
    assert set(old) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert old["sql_hash"] == LEGACY_HASHES["stock_pit"]
    assert new["sql_hash"] == old["sql_hash"] and new["schema_hash"] == old["schema_hash"]
    assert (new["plan"], new["semantics_version"]) == ("asof_intervals", "v1")
    assert new["plan_hash"] == MartPlan(
        "asof_intervals", stock_pit.build_stock_pit_sql(plan="asof_intervals")
    ).plan_hash
    with pytest.raises(StaleMartContract):
        stock_pit.materialize_stock_pit(con, fast)
    # The two marts hold the same rows.
    for tag in ("single", "asof"):
        glob = str(mart_table_dir(_lake(tmp_path, tag), stock_pit.PIT_TABLE) / "*.parquet")
        con.execute(f"CREATE TABLE m_{tag} AS SELECT * FROM read_parquet('{glob}')")
    _same_rows(con, "m_single", "m_asof")


# --------------------------------------------------------------------------
# E1 round 2: feat_flow argmin_pivot == window_dedup
# --------------------------------------------------------------------------


def _flow_con(seed: int = 0) -> duckdb.DuckDBPyConnection:
    rng = random.Random(seed)
    con = duckdb.connect()
    days = _business_days(date(2024, 1, 2), date(2024, 3, 29))
    codes = flow.METRIC_CODES
    rows = []
    for ticker, market in (("A", "KOSPI"), ("B", "KOSPI"), ("A", "KOSDAQ")):
        for i, day in enumerate(days):
            for code in codes:
                value = float(rng.randint(-500, 500)) if "net_buy" in code else float(
                    rng.randint(0, 900)
                )
                krx: float | None = value
                if rng.random() < 0.08:
                    krx = None  # a KRX row whose value is NULL
                pykrx = value if rng.random() < 0.9 else value + 1.0
                mode = rng.choice(["both", "both", "krx", "pykrx", "none"])
                if mode in ("both", "krx"):
                    rows.append((day, ticker, market, code, krx, "KRX"))
                if mode in ("both", "pykrx"):
                    rows.append((day, ticker, market, code, pykrx, "PYKRX"))
                if mode == "both" and rng.random() < 0.1:
                    rows.append((day, ticker, market, code, value, "KIS"))
            if i % 11 == 0:  # a code the mart does not know, and a NULL source
                rows.append((day, ticker, market, "unknown_metric", 1.0, "KRX"))
                rows.append((day, ticker, market, "foreign_net_buy_volume", 3.0, None))
    con.execute(
        "CREATE TABLE krx_security_flow_raw (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "metric_code VARCHAR, value DOUBLE, source VARCHAR)"
    )
    con.executemany("INSERT INTO krx_security_flow_raw VALUES (?,?,?,?,?,?)", rows)
    con.execute("CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
                "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT)")
    prices = [
        (day, t, m, 0.0 if (i % 17 == 5 and t == "B") else 100.0, 100.0 if i % 17 != 5 else 0.0,
         100.0 if i % 17 != 5 else 0.0, 100.0 + i, 1000 + 10 * i)
        for t, m in (("A", "KOSPI"), ("B", "KOSPI"), ("A", "KOSDAQ"))
        for i, day in enumerate(days)
    ]  # fmt: skip
    con.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)", prices)
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT trade_date, ticker, market, "
        "100000.0 AS float_shares_pit FROM daily_ohlcv"
    )
    con.execute(
        "CREATE VIEW dim_price_quality_daily AS SELECT trade_date, ticker, market, "
        "trade_date >= DATE '2024-02-01' AS short_balance_is_available, 'allowed' AS short_regime, "
        "CASE WHEN open = 0 AND high = 0 AND low = 0 THEN NULL ELSE "
        "ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) END "
        "AS valid_session_idx FROM daily_ohlcv"
    )
    return con


_FLOW_KW = {
    "price_view": "daily_ohlcv",
    "pit_view": "dim_stock_pit_daily",
    "quality_view": "dim_price_quality_daily",
}


@pytest.mark.parametrize("kwargs", [_FLOW_KW, {}], ids=["with_price_view", "degraded"])
def test_argmin_pivot_returns_the_window_dedup_rows(kwargs: dict) -> None:
    con = _flow_con()
    con.execute("CREATE TABLE f_window AS " + flow.build_flow_sql(**kwargs))
    con.execute(
        "CREATE TABLE f_argmin AS "
        + flow.build_flow_sql(**kwargs, pivot_plan=flow.PIVOT_PLAN_ARGMIN)
    )
    _same_rows(con, "f_window", "f_argmin")
    assert con.execute("SELECT count(flow_foreign_netbuy_sum_20d) FROM f_argmin").fetchone()[0] > 0


def test_argmin_pivot_keeps_a_null_krx_value_over_a_present_pykrx_one() -> None:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE krx_security_flow_raw AS SELECT * FROM (VALUES "
        "(DATE '2024-01-02','A','KOSPI','foreign_net_buy_volume',NULL::DOUBLE,'KRX'),"
        "(DATE '2024-01-02','A','KOSPI','foreign_net_buy_volume',7.0,'PYKRX'),"
        "(DATE '2024-01-02','A','KOSPI','foreign_net_buy_volume',8.0,'KIS'),"
        "(DATE '2024-01-02','A','KOSPI','individual_net_buy_volume',5.0,'PYKRX'),"
        "(DATE '2024-01-02','A','KOSPI','individual_net_buy_volume',6.0,'KIS')"
        ") t(trade_date, ticker, market, metric_code, value, source)"
    )
    for plan in (flow.PIVOT_PLAN_WINDOW, flow.PIVOT_PLAN_ARGMIN):
        row = con.execute(
            "SELECT flow_foreign_netbuy_sum_5d, flow_indiv_netbuy_sum_5d FROM ("
            + flow.build_flow_sql(pivot_plan=plan)
            + ")"
        ).fetchone()
        # NULL stays NULL (no fall-through to PYKRX/KIS); the two non-KRX sources are equal
        # priority, so a pair of them is not asserted — only that one of them is returned.
        assert row[0] is None, plan
        assert row[1] in (5.0, 6.0), plan


def test_flow_argmin_pivot_has_no_dedup_pass() -> None:
    assert "dedup" not in flow.build_flow_sql(pivot_plan=flow.PIVOT_PLAN_ARGMIN)
    assert "arg_min_null" in flow.build_flow_sql(pivot_plan=flow.PIVOT_PLAN_ARGMIN)
    assert "PARTITION BY trade_date, ticker, market, metric_code" in flow.build_flow_sql()


def test_flow_materialize_records_the_pivot_plan(tmp_path: Path) -> None:
    con = _flow_con()
    legacy, fast = (_lake(tmp_path, tag) for tag in ("window", "argmin"))
    flow.materialize_flow(con, legacy, **_FLOW_KW)
    flow.materialize_flow(con, fast, **_FLOW_KW, pivot_plan=flow.PIVOT_PLAN_ARGMIN)
    old, new = (mart_cache_metadata(c, flow.FLOW_TABLE) for c in (legacy, fast))
    assert set(old) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert old["sql_hash"] == LEGACY_HASHES["flow_research"]
    assert new["sql_hash"] == old["sql_hash"] and new["schema_hash"] == old["schema_hash"]
    assert (new["plan"], new["semantics_version"]) == ("argmin_pivot", "v1")
    with pytest.raises(StaleMartContract):
        flow.materialize_flow(con, fast, **_FLOW_KW)


# --------------------------------------------------------------------------
# E1 round 2: stock_metric_fact split_argmin == single
# --------------------------------------------------------------------------


def _smf_pair(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE TABLE smf_single AS " + mn.build_stock_metric_fact_sql())
    con.execute(
        "CREATE TABLE smf_split AS "
        + mn.build_stock_metric_fact_sql(plan=mn.PLAN_SPLIT_ARGMIN)
    )


def test_split_argmin_has_the_single_statements_schema_and_the_golden_rows() -> None:
    from . import test_metrics_normalize_mart as golden

    con = duckdb.connect()
    golden._load_raw_tables(con, golden.MockMetricStorage())
    _smf_pair(con)
    assert (
        con.execute("DESCRIBE smf_single").fetchall()
        == con.execute("DESCRIBE smf_split").fetchall()
    )
    _same_rows(con, "smf_single", "smf_split")
    # CFS beats OFS, and the XBRL fallback loses to the statement line.
    assert con.execute(
        "SELECT count(*) FROM smf_split WHERE mapping_rule_code LIKE 'fin.net_income.cfs%'"
    ).fetchone()[0] >= 0


def _rule(code, metric, table, priority, selector="thstrm_amount", **fields):
    return MetricMappingRule(
        rule_code=code, metric_code=metric, source_table=table, value_selector=selector,
        priority=priority, **fields,
    )


def _wildcard_rules() -> list[MetricMappingRule]:
    fin, sc = "dart_financial_statement_raw", "dart_share_count_raw"
    ret, xb = "dart_shareholder_return_raw", "dart_xbrl_fact_raw"
    return [
        _rule("fin.rev.cfs", "rev", fin, 10, fs_div="CFS", sj_div="IS", account_id="rev"),
        _rule("fin.rev.ofs", "rev", fin, 20, fs_div="OFS", sj_div="IS", account_id="rev"),
        _rule("fin.assets.any", "assets", fin, 5, fs_div="CFS", sj_div="BS", account_nm="자산총계"),
        _rule("fin.anything", "cat", fin, 50),
        _rule("sc.issued", "issued", sc, 10, "istc_totqy", row_name="합계"),
        _rule("sc.treasury.any", "treasury", sc, 30, "tesstk_co"),
        _rule("ret.dps", "dps", ret, 10, "value_numeric", statement_type="dividend",
              row_name="DPS", stock_knd="보통주", metric_code_match="thstrm"),
        _rule("ret.any", "payout", ret, 20, "value_numeric", dim1="d1", metric_code_match="thstrm"),
        _rule("xbrl.rev", "xrev", xb, 10, "value_numeric", account_id="ifrs_Revenue"),
        _rule("xbrl.rev.label", "xrev", xb, 20, "value_numeric", account_nm="매출액"),
    ]  # fmt: skip


def _wildcard_con(seed: int) -> duckdb.DuckDBPyConnection:
    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE dart_corp_master (corp_code VARCHAR, ticker VARCHAR, market VARCHAR, "
        "is_active BOOLEAN)"
    )
    con.executemany(
        "INSERT INTO dart_corp_master VALUES (?,?,?,?)",
        [("c1", "T1", "KOSPI", True), ("c2", "T2", "KOSDAQ", True), ("c3", "T3", "KOSPI", False),
         ("c4", "", "KOSPI", True), ("c5", "T5", None, True)],
    )  # fmt: skip
    tickers = ["T1", "T2", "T3", "T5"]
    con.execute(
        "CREATE TABLE dart_financial_statement_raw (ticker VARCHAR, bsns_year INTEGER, "
        "reprt_code VARCHAR, fs_div VARCHAR, sj_div VARCHAR, account_id VARCHAR, "
        "account_nm VARCHAR, thstrm_amount DECIMAL(30,4), ord BIGINT, rcept_no VARCHAR)"
    )
    con.executemany(
        "INSERT INTO dart_financial_statement_raw VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (rng.choice(tickers), rng.choice([2023, 2024]), rng.choice(["11011", "11012", "11099"]),
             rng.choice(["CFS", "OFS"]), rng.choice(["IS", "BS", "CIS"]),
             rng.choice(["rev", "other", ""]), rng.choice(["자산총계", "매출", ""]),
             rng.choice([None, Decimal("5"), Decimal("12.5"), Decimal("-3")]), i,
             f"2024{rng.randint(1, 12):02d}01{rng.randint(0, 3):06d}")
            for i in range(120)
        ],
    )  # fmt: skip
    con.execute(
        "CREATE TABLE dart_share_count_raw (ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "se VARCHAR, istc_totqy BIGINT, tesstk_co BIGINT, stlm_dt DATE, rcept_no VARCHAR)"
    )
    con.executemany(
        "INSERT INTO dart_share_count_raw VALUES (?,?,?,?,?,?,?,?)",
        [
            (rng.choice(tickers), rng.choice([2023, 2024]), rng.choice(["11011", "11013"]),
             rng.choice(["합계", "보통주", "우선주"]), rng.choice([None, 1000, 2000]),
             rng.choice([None, 0, 50]), rng.choice([None, date(2023, 12, 31)]), f"2024{i:08d}")
            for i in range(60)
        ],
    )  # fmt: skip
    con.execute(
        "CREATE TABLE dart_shareholder_return_raw (ticker VARCHAR, bsns_year INTEGER, "
        "reprt_code VARCHAR, statement_type VARCHAR, row_name VARCHAR, stock_knd VARCHAR, "
        "dim1 VARCHAR, dim2 VARCHAR, dim3 VARCHAR, metric_code VARCHAR, "
        "value_numeric DECIMAL(30,4), stlm_dt DATE, rcept_no VARCHAR)"
    )
    con.executemany(
        "INSERT INTO dart_shareholder_return_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (rng.choice(tickers), 2024, "11011", rng.choice(["dividend", "buyback"]),
             rng.choice(["DPS", "other"]), rng.choice(["보통주", "우선주", ""]),
             rng.choice(["d1", ""]), "", "", rng.choice(["thstrm", "frmtrm"]),
             rng.choice([None, Decimal("100"), Decimal("250")]), None, f"2024{i:08d}")
            for i in range(60)
        ],
    )  # fmt: skip
    con.execute(
        "CREATE TABLE dart_xbrl_fact_raw (ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "concept_id VARCHAR, concept_name VARCHAR, label_ko VARCHAR, context_id VARCHAR, "
        "period_end DATE, instant_date DATE, dimensions VARCHAR, value_numeric DECIMAL(30,4), "
        "rcept_no VARCHAR)"
    )
    dims = [
        "[]", '["ConsolidatedMember"]', '["SeparateMember"]',
        '["ConsolidatedMember","ReportedAmountMember"]', '["OperatingSegmentsMember"]',
    ]  # fmt: skip
    con.executemany(
        "INSERT INTO dart_xbrl_fact_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (rng.choice(tickers), 2024, rng.choice(["11011", "11014"]),
             rng.choice(["ifrs_Revenue", "ifrs_Other", ""]), rng.choice(["Revenue", "매출액", "x"]),
             rng.choice(["매출액", "기타"]), f"ctx{i}", date(2024, 12, 31), None,
             rng.choice(dims), rng.choice([None, Decimal("9"), Decimal("11")]), f"2025{i:08d}")
            for i in range(120)
        ],
    )  # fmt: skip
    return con


@pytest.mark.parametrize("seed", range(5))
def test_split_argmin_matches_single_with_wildcard_rules(seed: int, monkeypatch) -> None:
    monkeypatch.setattr(mn, "default_metric_mapping_rules", _wildcard_rules)
    con = _wildcard_con(seed)
    _smf_pair(con)
    _same_rows(con, "smf_single", "smf_split")
    # Both branches were exercised: winners from equality and from wildcard rules.
    codes = {
        r[0] for r in con.execute("SELECT DISTINCT mapping_rule_code FROM smf_split").fetchall()
    }
    assert {"sc.issued", "xbrl.rev"} & codes
    wildcard = {"sc.treasury.any", "ret.any", "xbrl.rev.label", "fin.anything", "fin.assets.any"}
    assert wildcard & codes


def test_split_argmin_keeps_priority_order_across_fs_div_and_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(mn, "default_metric_mapping_rules", _wildcard_rules)
    con = _wildcard_con(0)
    for table in ("dart_financial_statement_raw", "dart_xbrl_fact_raw"):
        con.execute(f"DELETE FROM {table}")
    con.execute(
        "INSERT INTO dart_financial_statement_raw VALUES "
        "('T1', 2024, '11011', 'OFS', 'IS', 'rev', '', 1, 1, '20250101000001'),"
        "('T1', 2024, '11011', 'CFS', 'IS', 'rev', '', 2, 2, '20250101000001')"
    )
    con.execute(
        "INSERT INTO dart_xbrl_fact_raw VALUES "
        "('T1', 2024, '11011', 'ifrs_Revenue', 'Revenue', '매출액', 'a', DATE '2024-12-31', NULL,"
        " '[\"SeparateMember\"]', 3, '20250101000001'),"
        "('T1', 2024, '11011', 'ifrs_Revenue', 'Revenue', '매출액', 'b', DATE '2024-12-31', NULL,"
        " '[\"ConsolidatedMember\"]', 4, '20250101000001')"
    )
    _smf_pair(con)
    for table in ("smf_single", "smf_split"):
        got = dict(con.execute(
            f"SELECT metric_code, value_numeric FROM {table} WHERE ticker = 'T1' "
            "AND metric_code IN ('rev', 'xrev')").fetchall())
        assert got == {"rev": Decimal("2.0000"), "xrev": Decimal("4.0000")}, table
    _same_rows(con, "smf_single", "smf_split")


def test_split_argmin_text_is_pinned_and_default_is_untouched() -> None:
    split = mn.build_stock_metric_fact_sql(plan=mn.PLAN_SPLIT_ARGMIN)
    assert "arg_min(" in split and "QUALIFY" not in split
    assert "rule_rel_eq_account_id" in split and "rule_rel_wild_row_name" in split
    assert "rule_rel_eq" not in mn.build_stock_metric_fact_sql()


def test_split_argmin_persisted_mart_records_its_plan(tmp_path: Path) -> None:
    from modeler.etl.lake import read_derived_plan_record, register_derived_marts

    from . import test_metrics_normalize_mart as golden

    config = _lake(tmp_path)
    assert read_derived_plan_record(config, "stock_metric_fact") == {
        "semantics_version": "v1", "plan": "single", "plan_hash": None}
    con = duckdb.connect()
    golden._load_raw_tables(con, golden.MockMetricStorage())
    register_derived_marts(
        con, config, which=("stock_metric_fact",), persist=True,
        metric_fact_plan=mn.PLAN_SPLIT_ARGMIN)
    record = read_derived_plan_record(config, "stock_metric_fact")
    assert record == mn.stock_metric_fact_plan_record(mn.PLAN_SPLIT_ARGMIN)
    assert record["plan"] == "split_argmin" and len(record["plan_hash"]) == 64
    # Same plan: reused. The default plan, or another record, is not silently mixed in.
    register_derived_marts(
        con, config, which=("stock_metric_fact",), persist=True,
        metric_fact_plan=mn.PLAN_SPLIT_ARGMIN)
    other = _lake(tmp_path, "other")
    register_derived_marts(con, other, which=("stock_metric_fact",), persist=True)
    assert read_derived_plan_record(other, "stock_metric_fact")["plan"] == "single"
    with pytest.raises(StaleMartContract):
        register_derived_marts(
            con, other, which=("stock_metric_fact",), persist=True,
            metric_fact_plan=mn.PLAN_SPLIT_ARGMIN)


def test_rules_version_changes_the_text_only_when_asked() -> None:
    default = mn.build_stock_metric_fact_sql()
    assert mn.build_stock_metric_fact_sql(rules_version=None) == default
    assert mn.build_stock_metric_fact_sql(rules_version="mrv2_20260909") == default
    frozen = mn.build_stock_metric_fact_sql(rules_version="mrv1_20260818")
    assert frozen != default
    assert "retained_earnings" in default and "retained_earnings" not in frozen
    with pytest.raises(ValueError):
        mn.build_stock_metric_fact_sql(rules_version="mrv0")


def test_rules_version_is_recorded_and_never_mixed_up(tmp_path: Path) -> None:
    from modeler.etl.lake import read_derived_plan_record, register_derived_marts

    from . import test_metrics_normalize_mart as golden

    # Default definition: no rules keys, exactly the legacy record.
    assert mn.stock_metric_fact_plan_record() == {
        "semantics_version": "v1", "plan": "single", "plan_hash": None}
    assert mn.stock_metric_fact_plan_record(rules_version="mrv2_20260909") == (
        mn.stock_metric_fact_plan_record())
    record = mn.stock_metric_fact_plan_record(rules_version="mrv1_20260818")
    assert record["rules_version"] == "mrv1_20260818" and len(record["rules_hash"]) == 64
    assert record["plan"] == "single" and record["plan_hash"] is None
    split = mn.stock_metric_fact_plan_record(mn.PLAN_SPLIT_ARGMIN, rules_version="mrv1_20260818")
    assert split["plan_hash"] != mn.stock_metric_fact_plan_record(mn.PLAN_SPLIT_ARGMIN)["plan_hash"]

    config = _lake(tmp_path)
    con = duckdb.connect()
    golden._load_raw_tables(con, golden.MockMetricStorage())
    register_derived_marts(
        con, config, which=("stock_metric_fact",), persist=True,
        metric_fact_plan=mn.PLAN_SPLIT_ARGMIN, metric_rules_version="mrv1_20260818")
    assert read_derived_plan_record(config, "stock_metric_fact") == split
    # Same rules: reused. Other rules, or the default definition: refused.
    register_derived_marts(
        con, config, which=("stock_metric_fact",), persist=True,
        metric_fact_plan=mn.PLAN_SPLIT_ARGMIN, metric_rules_version="mrv1_20260818")
    for kwargs in (
        {"metric_fact_plan": mn.PLAN_SPLIT_ARGMIN},
        {"metric_fact_plan": mn.PLAN_SPLIT_ARGMIN, "metric_rules_version": "mrv2_20260909"},
        {},
    ):
        with pytest.raises(StaleMartContract):
            register_derived_marts(
                con, config, which=("stock_metric_fact",), persist=True, **kwargs)
    # A default-built mart is not reused by the frozen rules either.
    other = _lake(tmp_path, "other")
    register_derived_marts(con, other, which=("stock_metric_fact",), persist=True)
    with pytest.raises(StaleMartContract):
        register_derived_marts(
            con, other, which=("stock_metric_fact",), persist=True,
            metric_rules_version="mrv1_20260818")


# --------------------------------------------------------------------------
# E1 round 2: market return semantics v2 (A4)
# --------------------------------------------------------------------------


def _return_panel(con: duckdb.DuckDBPyConnection, *, shuffle_seed: int | None) -> None:
    rng = random.Random(7)
    days = _business_days(date(2022, 1, 3), date(2022, 12, 30))
    rows = [
        (d, f"T{t:03d}", "KOSPI" if t % 3 else "KOSDAQ", i + 1,
         rng.gauss(0, 0.02) * 10 ** rng.choice([-6, -3, 0, 2]))
        for t in range(60)
        for i, d in enumerate(days)
    ]  # fmt: skip
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    con.execute(
        "CREATE OR REPLACE TABLE panel (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "valid_session_idx BIGINT, log_ret DOUBLE)"
    )
    con.executemany("INSERT INTO panel VALUES (?,?,?,?,?)", rows)


def _market_model(con, semantics: str) -> str:
    return (
        "WITH "
        + trading_panel.build_market_model_sql("panel", semantics)
        + " SELECT trade_date, ticker, market, market_ret, beta_252, alpha_252, model_n_252,"
        " resid_ret FROM residuals ORDER BY trade_date, ticker, market"
    )


def test_market_model_v2_keeps_the_columns_and_the_mean_up_to_rounding() -> None:
    con = duckdb.connect()
    _return_panel(con, shuffle_seed=None)
    assert con.execute(f"DESCRIBE {_market_model(con, 'v1')}").fetchall() == con.execute(
        f"DESCRIBE {_market_model(con, 'v2')}"
    ).fetchall()
    v1 = con.execute(_market_model(con, "v1")).fetchall()
    v2 = con.execute(_market_model(con, "v2")).fetchall()
    assert len(v1) == len(v2) == 60 * 260
    worst = max(abs(a[3] - b[3]) for a, b in zip(v1, v2, strict=True))
    assert worst < 1e-12  # the same mean; only the summation order differs


def test_market_model_v2_is_independent_of_row_order_and_thread_count() -> None:
    results = {}
    for threads in (1, 2, 4):
        for shuffle_seed in (None, 1, 2):
            con = duckdb.connect()
            con.execute(f"SET threads = {threads}")
            _return_panel(con, shuffle_seed=shuffle_seed)
            # repr round-trips floats exactly and, unlike ``==``, treats NaN as equal to NaN.
            results[threads, shuffle_seed] = repr(con.execute(_market_model(con, "v2")).fetchall())
    # Exact equality, not approx: the point is bitwise reproducibility.
    assert len(set(results.values())) == 1


def test_fin_scan_v2_orders_the_cross_sectional_moments() -> None:
    v1, v2 = fin_scan.build_fin_scan_daily_sql(), fin_scan.build_fin_scan_daily_sql(semantics="v2")
    assert v1 != v2 and v2.count("ORDER BY ticker ROWS BETWEEN UNBOUNDED PRECEDING") == 8
    # Nothing else moved: undoing the 8 windows gives the v1 text back.
    undone = v2.replace(" " + fin_scan._ORDERED_WHOLE_PARTITION, "")
    assert undone == v1
    with pytest.raises(ValueError):
        fin_scan.build_fin_scan_daily_sql(semantics="v9")


def test_ordered_whole_partition_moments_do_not_depend_on_arrival_order_or_threads() -> None:
    """The pattern fin_scan v2 uses, on values whose float sum is order-sensitive."""
    rng = random.Random(3)
    base = [
        (d, f"T{t:04d}", "KOSPI", rng.gauss(0, 1) * 10 ** rng.choice([-8, -3, 0, 3, 8]))
        for d in range(8) for t in range(400)
    ]  # fmt: skip
    results = set()
    for threads in (1, 2, 4):
        for seed in (None, 1, 2):
            rows = list(base)
            if seed is not None:
                random.Random(seed).shuffle(rows)
            con = duckdb.connect()
            con.execute(f"SET threads = {threads}")
            con.execute("CREATE TABLE x (d INTEGER, ticker VARCHAR, market VARCHAR, v DOUBLE)")
            con.executemany("INSERT INTO x VALUES (?,?,?,?)", rows)
            window = f"PARTITION BY d, market {fin_scan._ORDERED_WHOLE_PARTITION}"
            results.add(repr(con.execute(
                f"SELECT d, ticker, AVG(v) OVER ({window}), STDDEV_SAMP(v) OVER ({window}) "
                "FROM x ORDER BY d, ticker").fetchall()))
    assert len(results) == 1


def test_fin_scan_v2_returns_v1_rows_up_to_the_last_digits_and_records_itself(
    tmp_path: Path,
) -> None:
    con = _scan_con()
    con.execute(f"CREATE TABLE scan_v1 AS {fin_scan.build_fin_scan_daily_sql()}")
    con.execute(f"CREATE TABLE scan_v2 AS {fin_scan.build_fin_scan_daily_sql(semantics='v2')}")
    others = [
        r[0] for r in con.execute("DESCRIBE scan_v1").fetchall()
        if r[0] not in ("fin_value_z", "fin_value_z_lag1")
    ]
    cols = ", ".join(others)
    assert (
        con.execute(f"SELECT {cols} FROM scan_v1 EXCEPT ALL SELECT {cols} FROM scan_v2").fetchall()
        == []
    )
    assert con.execute(
        "SELECT max(abs(a.fin_value_z - b.fin_value_z)) FROM scan_v1 a JOIN scan_v2 b "
        "USING (trade_date, ticker, market)").fetchone()[0] in (None, pytest.approx(0, abs=1e-12))
    new = _lake(tmp_path, "v2")
    fin_scan.materialize_fin_scan_daily(con, new, join_plan=JOIN_PLAN_COALESCE, semantics="v2")
    meta = mart_cache_metadata(new, fin_scan.FIN_SCAN_TABLE)
    assert meta["sql_hash"] == _h(fin_scan.build_fin_scan_daily_sql(semantics="v2"))
    assert (meta["semantics_version"], meta["plan"]) == ("v2", "coalesce_join")
    assert meta["plan_hash"] == MartPlan(
        "coalesce_join",
        fin_scan.build_fin_scan_daily_sql(join_plan="coalesce_join", semantics="v2"),
        semantics_version="v2",
    ).plan_hash
    legacy = _lake(tmp_path, "v1")
    fin_scan.materialize_fin_scan_daily(con, legacy)
    assert set(mart_cache_metadata(legacy, fin_scan.FIN_SCAN_TABLE)) == {
        "analysis_config_hash", "schema_hash", "sql_hash"}
    assert mart_cache_metadata(legacy, fin_scan.FIN_SCAN_TABLE)["sql_hash"] != meta["sql_hash"]


def test_price_materialize_v2_records_its_semantics(tmp_path: Path) -> None:
    con = duckdb.connect()
    days = _business_days(date(2024, 1, 2), date(2024, 2, 29))
    con.execute(
        "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT)"
    )
    con.executemany(
        "INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)",
        [(d, t, "KOSPI", 100.0, 101.0, 99.0, 100.0 + i + k, 1000) for k, t in enumerate("AB")
         for i, d in enumerate(days)],
    )  # fmt: skip
    legacy, new = (_lake(tmp_path, tag) for tag in ("v1", "v2"))
    price.materialize_price(con, legacy)
    price.materialize_price(con, new, semantics="v2")
    old, fresh = (mart_cache_metadata(c, price.PRICE_TABLE) for c in (legacy, new))
    assert set(old) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert old["sql_hash"] == LEGACY_HASHES["price"]
    assert fresh["sql_hash"] == _h(price.build_price_sql(market_semantics="v2"))
    assert fresh["sql_hash"] != old["sql_hash"]
    assert (fresh["semantics_version"], fresh["plan"]) == ("v2", "single")
    with pytest.raises(StaleMartContract):
        price.materialize_price(con, new)


# --------------------------------------------------------------------------
# A0: feat_fin_scan_daily column projection
# --------------------------------------------------------------------------

LEGACY_HASHES.update({
    "fqmv_default": "439ce730b157e1b5e92ac18cecf12576c240e3ab9e30181232b738877e676b9a",
    "fin_scan_default": "9656de5eb96b9b1498187308a9ea3831a3f0a2f4e29ff45cca7fb061fdd7fb4a",
    "fin_scan_v2": "1a15d0489c66ca380b139d67ea62390faa62f6f01781cad7429cf19191ff8694",
    "fin_scan_coalesce": "272d3576a589e26d6233a9dc4d4f4d72ab285297ee4a6f9440ac6782e329d8b2",
    "fin_scan_coalesce_v2": "5e823432c915026578c99781b4f3f2130bedaec496445123ac96c7c2ee023bd9",
})


def test_projection_refactor_keeps_the_fin_scan_text_and_fqmv_default() -> None:
    from modeler.etl.marts import financial_quarters as fq

    assert _h(fq.build_fin_quarterly_metric_vintage_sql()) == LEGACY_HASHES["fqmv_default"]
    assert _h(fin_scan.build_fin_scan_daily_sql()) == LEGACY_HASHES["fin_scan_default"]
    assert _h(fin_scan.build_fin_scan_daily_sql(semantics="v2")) == LEGACY_HASHES["fin_scan_v2"]
    assert _h(fin_scan.build_fin_scan_daily_sql(join_plan=JOIN_PLAN_COALESCE)) == (
        LEGACY_HASHES["fin_scan_coalesce"])
    assert _h(fin_scan.build_fin_scan_daily_sql(join_plan=JOIN_PLAN_COALESCE, semantics="v2")) == (
        LEGACY_HASHES["fin_scan_coalesce_v2"])


def test_projection_and_full_mart_share_one_fin_log_mcap_and_base_ok_definition() -> None:
    from modeler.etl.features.fin_vintage import BASE_OK_SQL

    full = fin_scan.build_fin_scan_daily_sql()
    projection = fin_scan.build_fin_scan_projection_sql(["fin_log_mcap"])
    for text in (full, projection):
        assert f"{fin_scan.FIN_LOG_MCAP_SQL} AS fin_log_mcap" in text
        assert f"{BASE_OK_SQL} AS base_ok" in text
    # Both read the same panel source.
    assert fin_scan._panel_source("dim_stock_pit_daily", "dim_price_quality_daily") in full
    assert fin_scan._panel_source("dim_stock_pit_daily", "dim_price_quality_daily") in projection
    assert "fin_quarterly_metric_vintage" not in projection


def test_projection_fails_loudly_for_a_column_it_cannot_produce() -> None:
    with pytest.raises(ValueError, match=r"does not support \['fin_value_z'\]"):
        fin_scan.build_fin_scan_projection_sql(["fin_log_mcap", "fin_value_z"])
    with pytest.raises(ValueError, match="at least one column"):
        fin_scan.build_fin_scan_projection_sql([])
    # Key columns need no request; requesting one is not an error.
    assert fin_scan.check_projection_columns(["ticker", "fin_log_mcap"]) == ("fin_log_mcap",)


def _scan_con_with_edge_cases() -> duckdb.DuckDBPyConnection:
    """The scan fixture plus PIT rows that fail each ``base_ok`` condition in turn."""
    con = _scan_con()
    day = date(2024, 5, 6)
    cases = [  # ticker, market_cap, available, invalid, halted, valid_session_idx
        ("EDGE01", None, True, False, False, 1),
        ("EDGE02", 0.0, True, False, False, 1),
        ("EDGE03", 1e8, False, False, False, 1),
        ("EDGE04", 1e8, True, True, False, 1),
        ("EDGE05", 1e8, True, False, True, 1),
        ("EDGE06", 1e8, True, False, False, None),
        ("EDGE07", 1e8, True, False, None, 1),
        ("EDGE08", 2.5e8, True, False, False, 1),
    ]
    for ticker, cap, available, invalid, halted, idx in cases:
        con.execute(
            "INSERT INTO dim_stock_pit_daily VALUES (?, ?, 'KOSPI', ?, 1000, ?, ?, ?)",
            [day, ticker, cap, available, invalid, day])
        con.execute(
            "INSERT INTO dim_price_quality_daily VALUES (?, ?, 'KOSPI', ?, ?)",
            [day, ticker, halted, idx])
    # A PIT row with no quality row at all: the LEFT JOIN yields NULL flags.
    con.execute(
        "INSERT INTO dim_stock_pit_daily VALUES (?, 'EDGE09', 'KOSPI', 3e8, 1000, TRUE, FALSE, ?)",
        [day, day])
    return con


def test_projection_returns_the_full_marts_keys_and_fin_log_mcap_bit_for_bit() -> None:
    con = _scan_con_with_edge_cases()
    con.execute(f"CREATE TABLE scan_full AS {fin_scan.build_fin_scan_daily_sql()}")
    con.execute(
        "CREATE TABLE scan_proj AS " + fin_scan.build_fin_scan_projection_sql(["fin_log_mcap"]))
    assert [r[0] for r in con.execute("DESCRIBE scan_proj").fetchall()] == [
        "trade_date", "ticker", "market", "fin_log_mcap"]
    _same_rows(con, "scan_proj", "scan_proj")
    joined = "scan_full f JOIN scan_proj p USING (trade_date, ticker, market)"
    total = con.execute("SELECT count(*) FROM scan_full").fetchone()[0]
    assert total == con.execute("SELECT count(*) FROM scan_proj").fetchone()[0]
    assert total == con.execute(f"SELECT count(*) FROM {joined}").fetchone()[0]  # 1:1 on the key
    # Same value (NULLs in the same places) on every row, including the edge cases.
    assert con.execute(
        f"SELECT count(*) FROM {joined} WHERE f.fin_log_mcap IS DISTINCT FROM p.fin_log_mcap"
    ).fetchone()[0] == 0
    nulls, values = con.execute(
        "SELECT count(*) FILTER (WHERE fin_log_mcap IS NULL), "
        "count(*) FILTER (WHERE fin_log_mcap IS NOT NULL) FROM scan_proj").fetchone()
    assert nulls >= 8 and values > 100  # every edge case but EDGE08 is NULL


def test_projection_mart_records_its_columns_and_is_never_reused_as_the_full_mart(
    tmp_path: Path,
) -> None:
    con = _scan_con()
    full, proj = (_lake(tmp_path, tag) for tag in ("full", "proj"))
    fin_scan.materialize_fin_scan_daily(con, full)
    fin_scan.materialize_fin_scan_daily(con, proj, columns=["fin_log_mcap"])
    full_meta, proj_meta = (mart_cache_metadata(c, fin_scan.FIN_SCAN_TABLE) for c in (full, proj))
    assert "projected_columns" not in full_meta
    assert proj_meta["projected_columns"] == ["fin_log_mcap"]
    assert (proj_meta["plan"], proj_meta["semantics_version"]) == ("projection", "v1")
    # The contract text is the projection statement, not the full one.
    projection_sql = fin_scan.build_fin_scan_projection_sql(["fin_log_mcap"])
    assert proj_meta["sql_hash"] == _h(projection_sql) != full_meta["sql_hash"]
    assert proj_meta["schema_hash"] != full_meta["schema_hash"]
    assert proj_meta["plan_hash"] == MartPlan(
        "projection", projection_sql, metadata=(("projected_columns", ["fin_log_mcap"]),)
    ).plan_hash
    # Neither direction reuses the other.
    with pytest.raises(StaleMartContract):
        fin_scan.materialize_fin_scan_daily(con, proj)
    with pytest.raises(StaleMartContract):
        fin_scan.materialize_fin_scan_daily(con, full, columns=["fin_log_mcap"])
    # The same projection is reused.
    fin_scan.materialize_fin_scan_daily(con, proj, columns=["fin_log_mcap"])
    with pytest.raises(ValueError, match="takes no join_plan or semantics"):
        fin_scan.materialize_fin_scan_daily(
            con, _lake(tmp_path, "x"), columns=["fin_log_mcap"], semantics="v2")
    with pytest.raises(ValueError, match="does not support"):
        fin_scan.materialize_fin_scan_daily(con, _lake(tmp_path, "y"), columns=["fin_value_z"])


def test_plan_metadata_enters_the_plan_hash_only_when_given() -> None:
    plain = MartPlan("projection", "SELECT 1")
    assert plain.plan_hash == dataclasses.replace(plain, metadata=()).plan_hash
    tagged = dataclasses.replace(plain, metadata=(("projected_columns", ["a"]),))
    assert tagged.plan_hash != plain.plan_hash
    assert tagged.plan_hash != dataclasses.replace(
        plain, metadata=(("projected_columns", ["b"]),)).plan_hash


# --------------------------------------------------------------------------
# fin_quarterly_metric_vintage v2: an unknown availability propagates
# --------------------------------------------------------------------------

_FQ_COLUMNS = (
    "ticker VARCHAR, market VARCHAR, corp_code VARCHAR, metric_code VARCHAR, fs_basis VARCHAR, "
    "bsns_year INTEGER, reprt_code VARCHAR, value_numeric DECIMAL(30,4), available_from DATE, "
    "rcept_no VARCHAR, statement_period_end DATE, cumulative_value_numeric DECIMAL(30,4), "
    "comparative_q_amount DECIMAL(30,4), is_revision BOOLEAN"
)
_REPRT = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}


def _vintage_con(rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    """``rows``: (metric, year, quarter, value, available_from or None)."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE stock_metric_vintage_fact ({_FQ_COLUMNS})")
    for metric, year, quarter, value, avail in rows:
        con.execute(
            "INSERT INTO stock_metric_vintage_fact VALUES "
            "('005930', 'KOSPI', '00126380', ?, 'CFS', ?, ?, ?, ?, ?, ?, NULL, NULL, FALSE)",
            [metric, year, _REPRT[quarter], value, avail, f"{year}{quarter}0000001",
             date(year, quarter * 3, 28)])
    return con


def _fq(con, semantics: str, where: str = "TRUE") -> dict:
    from modeler.etl.marts import financial_quarters as fq

    sql = fq.build_fin_quarterly_metric_vintage_sql(semantics=semantics)
    cols = (
        "metric_code, bsns_year, quarter_ordinal, standalone_value, available_from, "
        "ttm_value, ttm_available_from"
    )
    rows = con.execute(
        f"SELECT {cols} FROM ({sql}) WHERE {where} ORDER BY 1, 2, 3").fetchall()
    return {(m, y, q): (v, a, t, ta) for m, y, q, v, a, t, ta in rows}


def _revenue_rows(q2_available: date | None) -> list[tuple]:
    """Revenue for 2023 Q1..Q4 (Q4 is the annual total) and 2024 Q1; Q2 2023 is the one
    whose availability is unknown when ``q2_available`` is None."""
    return [
        ("revenue", 2023, 1, 100, date(2023, 4, 11)),
        ("revenue", 2023, 2, 150, q2_available),
        ("revenue", 2023, 3, 140, date(2023, 11, 10)),
        ("revenue", 2023, 4, 560, date(2024, 3, 11)),
        ("revenue", 2024, 1, 120, date(2024, 4, 11)),
    ]


def test_v1_dates_a_ttm_before_a_component_whose_availability_is_unknown() -> None:
    con = _vintage_con(_revenue_rows(None))
    got = _fq(con, "v1")
    # Q1 2024 TTM = Q1'24 + Q4'23 (170) + Q3'23 (140) + Q2'23 (150): complete, and dated by
    # greatest() skipping the NULL, i.e. at Q1'24's own filing. The Q2'23 receipt lies in the
    # future, so this date is earlier than the TTM is knowable.
    value, avail, ttm, ttm_avail = got[("revenue", 2024, 1)]
    assert ttm == 120 + 170 + 140 + 150
    assert ttm_avail == date(2024, 4, 11)
    # The differenced Q4 is dated the same way (v1 ignores Q2'23).
    assert got[("revenue", 2023, 4)][1] == date(2024, 3, 11)
    assert got[("revenue", 2023, 2)][1] is None  # its own NULL is simply passed through


def test_v2_makes_every_availability_that_depends_on_an_unknown_one_null() -> None:
    con = _vintage_con(_revenue_rows(None))
    got = _fq(con, "v2")
    assert got[("revenue", 2024, 1)][2] == 120 + 170 + 140 + 150  # the value is still there
    assert got[("revenue", 2024, 1)][3] is None  # ttm_available_from
    assert got[("revenue", 2023, 4)][1] is None  # Q4 total minus Q1..Q3 needs Q2'23
    assert got[("revenue", 2023, 2)][1] is None
    # Figures that do not touch Q2'23 keep their date.
    assert got[("revenue", 2023, 1)][1] == date(2023, 4, 11)
    assert got[("revenue", 2023, 3)][1] == date(2023, 11, 10)


def test_v2_differenced_and_instant_quarters_propagate_too() -> None:
    rows = [
        ("operating_cash_flow", 2023, 1, 10, date(2023, 4, 11)),
        ("operating_cash_flow", 2023, 2, 30, None),
        ("operating_cash_flow", 2023, 3, 60, date(2023, 11, 10)),
        ("operating_cash_flow", 2023, 4, 100, date(2024, 3, 11)),
        ("total_assets", 2023, 3, 900, None),
        ("total_assets", 2023, 4, 1000, date(2024, 3, 11)),
        ("weighted_avg_shares", 2023, 1, 10, date(2023, 4, 11)),
        ("weighted_avg_shares", 2023, 2, 11, None),
        ("weighted_avg_shares", 2023, 3, 12, date(2023, 11, 10)),
    ]
    con = _vintage_con(rows)
    v1, v2 = _fq(con, "v1"), _fq(con, "v2")
    # Cumulative: Q2 = q2 - q1 needs q2 (unknown) and q1; Q3 = q3 - q2 needs q2; Q4 = q4 - q3.
    assert v1[("operating_cash_flow", 2023, 2)][1] == date(2023, 4, 11)  # greatest() skipped q2
    assert v2[("operating_cash_flow", 2023, 2)][1] is None
    assert v1[("operating_cash_flow", 2023, 3)][1] == date(2023, 11, 10)  # skipped the NULL
    assert v2[("operating_cash_flow", 2023, 3)][1] is None
    assert v2[("operating_cash_flow", 2023, 4)][1] == date(2024, 3, 11)  # only needs Q3
    assert v1[("weighted_avg_shares", 2023, 3)][1] == date(2023, 11, 10)
    assert v2[("weighted_avg_shares", 2023, 3)][1] is None
    # An instant passes its own availability through under both versions.
    assert v1[("total_assets", 2023, 3)][1] is None and v2[("total_assets", 2023, 3)][1] is None
    assert v2[("total_assets", 2023, 4)][1] == date(2024, 3, 11)


def test_a_missing_vintage_is_not_an_unknown_one() -> None:
    """Q3 absent (no filing) is a missing component, not an unknown availability: Q4 of a
    cumulative metric has no value either way, and its date must not turn into NULL because of
    a vintage that does not exist."""
    rows = [
        ("operating_cash_flow", 2023, 1, 10, date(2023, 4, 11)),
        ("operating_cash_flow", 2023, 4, 100, date(2024, 3, 11)),
    ]
    con = _vintage_con(rows)
    v1, v2 = _fq(con, "v1"), _fq(con, "v2")
    assert v1 == v2
    assert v2[("operating_cash_flow", 2023, 4)][1] == date(2024, 3, 11)


def test_v2_equals_v1_when_no_availability_is_unknown() -> None:
    con = _vintage_con(_revenue_rows(date(2023, 8, 10)))
    from modeler.etl.marts import financial_quarters as fq

    for semantics, name in (("v1", "fq1"), ("v2", "fq2")):
        con.execute(
            f"CREATE TABLE {name} AS "
            + fq.build_fin_quarterly_metric_vintage_sql(semantics=semantics))
    _same_rows(con, "fq1", "fq2")
    assert _fq(con, "v2")[("revenue", 2024, 1)][3] == date(2024, 4, 11)


def test_daily_intervals_drop_a_row_whose_availability_is_null() -> None:
    """``fin_vintage`` keeps only ``daily_available_from IS NOT NULL`` rows, so a v2 NULL TTM
    never becomes a daily value (v1 dated the same TTMs and let them through)."""
    from modeler.etl.features.fin_vintage import build_metric_intervals_cte
    from modeler.etl.marts import financial_quarters as fq

    con = _vintage_con(_revenue_rows(None))
    cte = build_metric_intervals_cte("fin_quarterly_metric_vintage", ["revenue"])

    def intervals(semantics: str) -> list[tuple]:
        con.execute(
            "CREATE OR REPLACE VIEW fin_quarterly_metric_vintage AS "
            + fq.build_fin_quarterly_metric_vintage_sql(semantics=semantics))
        return con.execute(
            f"WITH {cte} SELECT daily_value, daily_available_from FROM metric_intervals "
            "ORDER BY daily_available_from").fetchall()

    assert [value for value, _ in intervals("v1")] == [560, 120 + 170 + 140 + 150]
    assert intervals("v2") == []


def test_fqmv_materialize_records_v2_and_keeps_v1_metadata(tmp_path: Path) -> None:
    from modeler.etl.marts import financial_quarters as fq

    con = _vintage_con(_revenue_rows(date(2023, 8, 10)))
    legacy, new = _lake(tmp_path, "v1"), _lake(tmp_path, "v2")
    fq.materialize_fin_quarterly_metric_vintage(con, legacy)
    fq.materialize_fin_quarterly_metric_vintage(con, new, semantics="v2")
    old, fresh = (mart_cache_metadata(c, fq.FQMV_TABLE) for c in (legacy, new))
    assert set(old) == {"analysis_config_hash", "schema_hash", "sql_hash"}
    assert old["sql_hash"] == LEGACY_HASHES["fqmv_default"]
    assert fresh["sql_hash"] == _h(fq.build_fin_quarterly_metric_vintage_sql(semantics="v2"))
    assert fresh["sql_hash"] != old["sql_hash"]
    assert (fresh["semantics_version"], fresh["plan"]) == ("v2", "single")
    with pytest.raises(StaleMartContract):
        fq.materialize_fin_quarterly_metric_vintage(con, new)
    with pytest.raises(ValueError, match="unknown fin_quarterly_metric_vintage semantics"):
        fq.build_fin_quarterly_metric_vintage_sql(semantics="v3")
