"""``stock_metric_vintage_fact`` semantics ``v3`` -- XBRL income metrics read the 3-month context.

``v1``/``v2`` pick the *longest* XBRL context for every XBRL-sourced metric, written for the
cash-flow items. Income metrics arrive through the same branch and are treated downstream as
direct 3-month values, so a half-year / Q3 filing handed over the 6- / 9-month total
(2026-10-27 finding, 05_prep_a_financial_target.md). ``v3`` fixes that and is opt-in; these
tests pin both sides: the old text is byte-identical, the new one is right.
"""

from __future__ import annotations

import hashlib
from datetime import date

import pytest

from modeler.etl.marts import financial_quarters as fq
from modeler.etl.marts import metric_vintages as mv
from modeler.etl.marts import metric_vintages_delisted as dl
from tests.unit.test_metric_vintages import (
    _DIM_CFS,
    _TRADING_DAYS,
    _add_xbrl,
    _base_con,
)

# --- the old text must not move -----------------------------------------------------------
# SHA-256 of the SQL the frozen P6 / serving builders were written under, taken on main
# (0ebe1c8) before this change. A mismatch means a v1/v2 statement changed.

_OLD_HASHES = {
    ("smvf", "v1"): "284d09f7705b2e6d26bf2f7102840a658b4697968ececdc8cc59e76a76413b7d",
    ("staged", "v1"): "7cc7e627477ff35a3d01f12255fac5a816a6afa5a813d6278998683818aae2d1",
    ("dl", "v1"): "7ceb375d7219b4af8b24c3a87565aba4e262999abfcf6d408e4266990fc006bc",
    ("dlstaged", "v1"): "8105023f326eef1f3bfd23be7c424a30d8408feb9659d34e5677e29be3ad9dd2",
    ("fqmv", "v1"): "439ce730b157e1b5e92ac18cecf12576c240e3ab9e30181232b738877e676b9a",
    ("dlr", "v1"): "1a984f0ff6ed81724ba0ddc305e9dd2918ab86aee25d4695ae0d4b9466f36581",
    ("smvf", "v2"): "8f0194208575cd91d8e5130ef9ac83b3f821edbd3bf95ab8a2582f166aa05d3a",
    ("staged", "v2"): "313beb17d3377dd1b998e5b686420ed840835ebd928089ea223c748da72fcbb3",
    ("dl", "v2"): "0e9e83844eb32361392a033c208962cd56cc7f6db88bb0e90ea29cfd55741465",
    ("dlstaged", "v2"): "3112b3cb12e85afd84b1682c27374de9da9d5c8290451f9f6f0c6753e1df57a6",
    ("fqmv", "v2"): "c8ad1366856f98c6bf78ccee122e60aa6c4cf0b3c0e0afcb26c0cba366bfbe1f",
    ("dlr", "v2"): "81ffe7a31b41a5f05519724cf7ef0898576b0dbe4d7bf1b8a9c1ad376233a873",
}


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _plan_text(plan) -> str:
    return plan.final_sql + "".join(sql for _, sql in plan.stages)


def _current_hashes(sem: str) -> dict[str, str]:
    return {
        "smvf": _h(mv.build_stock_metric_vintage_fact_sql(semantics=sem)),
        "staged": _h(
            _plan_text(mv.build_stock_metric_vintage_fact_plan(semantics=sem, plan="staged"))
        ),
        "dl": _h(dl.build_stock_metric_vintage_fact_dl_sql(semantics=sem)),
        "dlstaged": _h(
            _plan_text(dl.build_stock_metric_vintage_fact_dl_plan(semantics=sem, plan="staged"))
        ),
        "fqmv": _h(fq.build_fin_quarterly_metric_vintage_sql(semantics=sem)),
        "dlr": _h(dl.build_fin_quarterly_metric_vintage_dlr_sql(semantics=sem)),
    }


@pytest.mark.parametrize("sem", ["v1", "v2"])
def test_old_semantics_sql_is_byte_identical_to_main(sem: str) -> None:
    got = _current_hashes(sem)
    want = {k: v for (k, s), v in _OLD_HASHES.items() if s == sem}
    assert got == want


def test_default_semantics_is_still_v1_and_v3_is_a_distinct_text() -> None:
    assert mv.build_stock_metric_vintage_fact_sql() == mv.build_stock_metric_vintage_fact_sql(
        semantics="v1"
    )
    assert mv.build_stock_metric_vintage_fact_plan() is None
    v2 = mv.build_stock_metric_vintage_fact_sql(semantics="v2")
    v3 = mv.build_stock_metric_vintage_fact_sql(semantics="v3")
    assert v3 != v2
    assert "NOT BETWEEN 80 AND 100" in v3
    assert "NOT BETWEEN" not in v2


def test_v3_keeps_every_other_v2_difference() -> None:
    # v3 = v2 + the income rule: the availability select and the winners tie-break are v2's.
    v2 = mv.build_stock_metric_vintage_fact_sql(semantics="v2")
    v3 = mv.build_stock_metric_vintage_fact_sql(semantics="v3")
    for text in (mv._availability_select("v2"), mv._winner_tie_break("v2")):
        assert text in v2 and text in v3


def test_unknown_semantics_is_still_refused() -> None:
    with pytest.raises(ValueError):
        mv.build_stock_metric_vintage_fact_sql(semantics="v4")


def test_v3_mart_gets_its_own_name() -> None:
    assert mv.semantics_table_name(mv.SMVF_TABLE, "v1") == "stock_metric_vintage_fact"
    assert mv.semantics_table_name(mv.SMVF_TABLE, "v2") == "stock_metric_vintage_fact"
    assert mv.semantics_table_name(mv.SMVF_TABLE, "v3") == "stock_metric_vintage_fact_v3"
    assert mv.semantics_table_name(dl.SMVF_DL_TABLE, "v3") == "stock_metric_vintage_fact_dl_v3"


# --- values -------------------------------------------------------------------------------

_REVENUE = "ifrs-full_Revenue"
_NET_INCOME = "ifrs-full_ProfitLoss"
_CFO = "ifrs-full_CashFlowsFromUsedInOperatingActivities"

_RCEPT = {
    "11013": "20250515000001",
    "11012": "20250814000001",
    "11014": "20251114000001",
    "11011": "20260320000001",
}
#: (reprt_code, context id, start, end, revenue) -- Q1 100, Q2 110, Q3 120, Q4 140 (FY 470).
_CONTEXTS = [
    ("11013", "q1", date(2025, 1, 1), date(2025, 3, 31), 100.0),
    ("11012", "q2", date(2025, 4, 1), date(2025, 6, 30), 110.0),
    ("11012", "h1", date(2025, 1, 1), date(2025, 6, 30), 210.0),
    ("11014", "q3", date(2025, 7, 1), date(2025, 9, 30), 120.0),
    ("11014", "m9", date(2025, 1, 1), date(2025, 9, 30), 330.0),
    ("11011", "fy", date(2025, 1, 1), date(2025, 12, 31), 470.0),
]


def _xbrl_con(*, skip: frozenset[str] = frozenset(), concept: str = _REVENUE):
    con = _base_con()
    for reprt, ctx, start, end, value in _CONTEXTS:
        if ctx in skip:
            continue
        _add_xbrl(
            con, value=value, context_id=ctx, period_start=start, period_end=end,
            dimensions=_DIM_CFS, concept_id=concept, bsns_year=2025, reprt_code=reprt,
            rcept_no=_RCEPT[reprt],
        )  # fmt: skip
    return con


def _fact_rows(con, sem: str, metric: str) -> dict[str, float]:
    con.execute("DROP VIEW IF EXISTS smvf_t")
    mv.register_stock_metric_vintage_fact_view(
        con, trading_days=_TRADING_DAYS, view_name="smvf_t", semantics=sem
    )
    rows = con.execute(
        "SELECT reprt_code, value_numeric FROM smvf_t WHERE metric_code = ? AND fs_basis = 'CFS'",
        [metric],
    ).fetchall()
    return {r: float(v) for r, v in rows}


def test_v3_interim_income_reads_the_three_month_context() -> None:
    got = _fact_rows(_xbrl_con(), "v3", "revenue")
    assert got == {"11013": 100.0, "11012": 110.0, "11014": 120.0, "11011": 470.0}


@pytest.mark.parametrize("sem", ["v1", "v2"])
def test_old_semantics_still_reads_the_cumulative_context(sem: str) -> None:
    # The defect, kept on purpose: the frozen / serving values do not move.
    got = _fact_rows(_xbrl_con(), sem, "revenue")
    assert got == {"11013": 100.0, "11012": 210.0, "11014": 330.0, "11011": 470.0}


def test_v3_cash_flow_stays_cumulative() -> None:
    con = _xbrl_con(concept=_CFO)
    for sem in ("v2", "v3"):
        got = _fact_rows(con, sem, "operating_cash_flow")
        assert got == {"11013": 100.0, "11012": 210.0, "11014": 330.0, "11011": 470.0}, sem


def test_v3_row_is_dropped_when_no_three_month_context_exists() -> None:
    # Falling back to the 6-month value would reproduce the bug, so the row is left empty.
    got = _fact_rows(_xbrl_con(skip=frozenset({"q2"})), "v3", "revenue")
    assert got == {"11013": 100.0, "11014": 120.0, "11011": 470.0}


def test_v3_context_window_is_80_to_100_days() -> None:
    con = _base_con()
    # 79, 80, 100 and 101 days ending on the half-year end; the 90-day one must win.
    for ctx, days, value in (("a", 79, 1.0), ("b", 100, 2.0), ("c", 101, 3.0), ("d", 90, 4.0)):
        start = date.fromordinal(date(2025, 6, 30).toordinal() - days)
        _add_xbrl(con, value=value, context_id=ctx, period_start=start,
                  period_end=date(2025, 6, 30), dimensions=_DIM_CFS, concept_id=_REVENUE,
                  bsns_year=2025, reprt_code="11012", rcept_no=_RCEPT["11012"])  # fmt: skip
    assert _fact_rows(con, "v3", "revenue") == {"11012": 4.0}


def test_v3_picks_closest_to_91_days_among_valid_contexts() -> None:
    con = _base_con()
    for ctx, days, value in (("a", 82, 1.0), ("b", 92, 2.0), ("c", 98, 3.0)):
        start = date.fromordinal(date(2025, 9, 30).toordinal() - days)
        _add_xbrl(con, value=value, context_id=ctx, period_start=start,
                  period_end=date(2025, 9, 30), dimensions=_DIM_CFS, concept_id=_REVENUE,
                  bsns_year=2025, reprt_code="11014", rcept_no=_RCEPT["11014"])  # fmt: skip
    assert _fact_rows(con, "v3", "revenue") == {"11014": 2.0}


def test_v3_net_income_follows_the_same_rule() -> None:
    got = _fact_rows(_xbrl_con(concept=_NET_INCOME), "v3", "net_income")
    assert got == {"11013": 100.0, "11012": 110.0, "11014": 120.0, "11011": 470.0}


def _quarters(con, sem: str, metric: str) -> dict[str, tuple]:
    mv.register_stock_metric_vintage_fact_view(
        con, trading_days=_TRADING_DAYS, view_name="smvf_t", semantics=sem
    )
    con.execute("DROP VIEW IF EXISTS fqmv_t")
    sql = fq.build_fin_quarterly_metric_vintage_sql(vintage_view="smvf_t", semantics="v2")
    con.execute(f"CREATE VIEW fqmv_t AS {sql}")
    rows = con.execute(
        "SELECT quarter, standalone_value, ttm_value FROM fqmv_t "
        "WHERE metric_code = ? AND fs_basis = 'CFS' ORDER BY quarter_ordinal",
        [metric],
    ).fetchall()
    return {
        q: (None if v is None else float(v), None if t is None else float(t)) for q, v, t in rows
    }


def test_v3_quarterly_mart_gives_standalone_q1_to_q4() -> None:
    q = _quarters(_xbrl_con(), "v3", "revenue")
    assert {k: v[0] for k, v in q.items()} == {"Q1": 100.0, "Q2": 110.0, "Q3": 120.0, "Q4": 140.0}
    # only Q4 has four quarters behind it inside this one fiscal year
    assert q["Q4"][1] == 470.0


def test_old_quarterly_mart_keeps_the_old_wrong_values() -> None:
    q = _quarters(_xbrl_con(), "v2", "revenue")
    # Q2 = 210 (cumulative), Q3 = 330, Q4 = 470 - (100 + 210 + 330) = -170: the defect.
    assert {k: v[0] for k, v in q.items()} == {"Q1": 100.0, "Q2": 210.0, "Q3": 330.0, "Q4": -170.0}


# --- _dl / _dlr -----------------------------------------------------------------------------


def _dl_con(con) -> None:
    con.execute(
        "CREATE TABLE stock_master (ticker VARCHAR, market VARCHAR, last_seen_date DATE, "
        "status VARCHAR, updated_at TIMESTAMP)"
    )
    con.execute("CREATE TABLE daily_ohlcv (ticker VARCHAR, market VARCHAR, trade_date DATE)")


def test_dl_and_dlr_accept_v3_and_old_dl_text_is_untouched() -> None:
    con = _xbrl_con()
    _dl_con(con)
    mv._register_calendar(con, _TRADING_DAYS)
    con.execute(
        f"CREATE VIEW smvf_dl_t AS {dl.build_stock_metric_vintage_fact_dl_sql(semantics='v3')}"
    )
    # the R1 text replacement still finds its anchors over a v3 fact
    sql = dl.build_fin_quarterly_metric_vintage_dlr_sql(vintage_view="smvf_dl_t", semantics="v2")
    con.execute(f"CREATE VIEW dlr_t AS {sql}")
    got = con.execute(
        "SELECT quarter, standalone_value FROM dlr_t "
        "WHERE metric_code = 'revenue' AND fs_basis = 'CFS' ORDER BY quarter_ordinal"
    ).fetchall()
    assert [(q, float(v)) for q, v in got] == [
        ("Q1", 100.0), ("Q2", 110.0), ("Q3", 120.0), ("Q4", 140.0)
    ]  # fmt: skip
    plan = dl.build_stock_metric_vintage_fact_dl_plan(semantics="v3", plan="staged")
    assert plan.semantics_version == "v3"
    assert any("NOT BETWEEN 80 AND 100" in sql for _, sql in plan.stages)


def test_dl_v3_over_the_same_rows_equals_v2_outside_income_xbrl_rows() -> None:
    # A cash-flow-only filing: v2 and v3 facts are identical row for row.
    con = _xbrl_con(concept=_CFO)
    _dl_con(con)
    mv._register_calendar(con, _TRADING_DAYS)
    a = con.execute(
        f"SELECT * FROM ({dl.build_stock_metric_vintage_fact_dl_sql(semantics='v2')}) ORDER BY ALL"
    ).fetchall()
    b = con.execute(
        f"SELECT * FROM ({dl.build_stock_metric_vintage_fact_dl_sql(semantics='v3')}) ORDER BY ALL"
    ).fetchall()
    assert a == b and a
