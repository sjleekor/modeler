"""company_xbrl 시험 (사전등록 20261010_quality_score §5.1·§5.2).

합성 parquet으로 기간·fs_div·우선순위·대소문자를 확인하고, 레이크가 있으면 실데이터 한 건을 본다.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.scores.quality import company_xbrl as X

CS = "ifrs-full:ConsolidatedAndSeparateFinancialStatementsAxis"
SEP = f'["{CS}=ifrs-full:SeparateMember"]'
CON = f'["{CS}=ifrs-full:ConsolidatedMember"]'
SEP_EQ = f'["{CS}=ifrs-full:SeparateMember", "ifrs-full:ComponentsOfEquityAxis=ifrs-full:X"]'


def _row(rcept, concept, ctx, dims, value, *, inst=None, start=None, end=None, nil=False, fy=2024):
    return {
        "rcept_no": rcept,
        "corp_code": "00000001",
        "bsns_year": fy,
        "reprt_code": 11011,
        "concept_id": concept,
        "context_id": ctx,
        "context_type": "instant" if inst else "duration",
        "period_start": start,
        "period_end": end,
        "instant_date": inst,
        "dimensions": dims,
        "value_numeric": value,
        "is_nil": nil,
    }


def _write(tmp_path: Path, rows: list[dict]) -> str:
    df = pl.DataFrame(
        rows,
        schema={
            "rcept_no": pl.Utf8, "corp_code": pl.Utf8, "bsns_year": pl.Int64,
            "reprt_code": pl.Int64, "concept_id": pl.Utf8, "context_id": pl.Utf8,
            "context_type": pl.Utf8, "period_start": pl.Date, "period_end": pl.Date,
            "instant_date": pl.Date, "dimensions": pl.Utf8, "value_numeric": pl.Float64,
            "is_nil": pl.Boolean,
        },
    )  # fmt: skip
    df.write_parquet(tmp_path / "x.parquet")
    return str(tmp_path / "*.parquet")


def _inst3(r, concept, dims, vals, tag="SeparateMember"):
    """C/P/BP instant 세 행 (12월 결산 2024)."""
    out = []
    for pre, d, v in zip(("CFY2024", "PFY2023", "BPFY2022"), (2024, 2023, 2022), vals):
        out.append(_row(r, concept, f"{pre}eFY_{tag}", dims, v, inst=date(d, 12, 31)))
    return out


def _dur3(r, concept, dims, vals, tag="SeparateMember"):
    out = []
    for pre, y, v in zip(("CFY2024", "PFY2023", "BPFY2022"), (2024, 2023, 2022), vals):
        out.append(
            _row(
                r, concept, f"{pre}dFY_{tag}", dims, v,
                start=date(y, 1, 1), end=date(y, 12, 31),
            )
        )  # fmt: skip
    return out


def test_period_and_fsdiv_basic(tmp_path):
    r = "20250321000001"
    rows = (
        _inst3(r, "ifrs-full_Assets", SEP, (300.0, 200.0, 100.0))
        + _inst3(r, "ifrs-full_Assets", CON, (3000.0, 2000.0, 1000.0), "ConsolidatedMember")
        + _dur3(r, "ifrs-full_ProfitLoss", SEP, (-7.0, -5.0, -3.0))
    )
    g = _write(tmp_path, rows)
    df = X.xbrl_values(g, None)
    key = lambda m, fs, p: df.filter(  # noqa: E731
        (pl.col("metric") == m) & (pl.col("fs_div") == fs) & (pl.col("period") == p)
    )
    assert key("ta", "OFS", "C")["value"].to_list() == [300.0]
    assert key("ta", "OFS", "P")["value"].to_list() == [200.0]
    assert key("ta", "OFS", "BP")["value"].to_list() == [100.0]
    assert key("ta", "CFS", "BP")["value"].to_list() == [1000.0]
    assert key("ta", "OFS", "P")["period_date"].to_list() == [date(2023, 12, 31)]
    assert key("ni", "OFS", "C")["value"].to_list() == [-7.0]
    assert key("ni", "OFS", "BP")["value"].to_list() == [-3.0]
    assert df.columns == X.OUT_COLS
    assert df.group_by(["rcept_no", "fs_div", "metric", "period"]).len()["len"].max() == 1


def test_extra_axis_and_empty_dims_dropped(tmp_path):
    r = "20250321000002"
    rows = _inst3(r, "ifrs-full_Equity", SEP, (50.0, 40.0, 30.0))
    rows += _inst3(r, "ifrs-full_Equity", SEP_EQ, (9.0, 9.0, 9.0), "SepEq")  # 축 둘 → 버림
    rows += _inst3(r, "ifrs-full_Assets", "[]", (1.0, 1.0, 1.0), "NoAxis")  # 축 없음 → 버림
    rows += _inst3(r, "ifrs-full_Assets", None, (1.0, 1.0, 1.0), "Null")
    df = X.xbrl_values(_write(tmp_path, rows), None)
    assert set(df["metric"]) == {"te"}
    assert sorted(df["value"].to_list()) == [30.0, 40.0, 50.0]


def test_concept_priority_case_and_prefix(tmp_path):
    r = "20250321000003"
    rows = _dur3(r, "ifrs-full_ProfitLossFromOperatingActivities", SEP, (2.0, 2.0, 2.0), "a")
    rows += _dur3(r, "dart_OperatingIncomeLoss", SEP, (1.0, 1.0, 1.0), "b")  # 1순위
    # ifrs_ 접두어와 대소문자: ifrs-full_ 가 이긴다
    rows += _inst3(r, "ifrs_Assets", SEP, (7.0, 7.0, 7.0), "c")
    rows += _inst3(r, "IFRS-FULL_ASSETS", SEP, (8.0, 8.0, 8.0), "d")
    rows += _inst3(r, "ifrs_Liabilities", SEP, (4.0, 3.0, 2.0), "e")  # ifrs_만 있어도 읽는다
    df = X.xbrl_values(_write(tmp_path, rows), None)
    oi = df.filter(pl.col("metric") == "oi")
    assert set(oi["value"]) == {1.0} and set(oi["concept_used"]) == {"dart_OperatingIncomeLoss"}
    ta = df.filter((pl.col("metric") == "ta") & (pl.col("period") == "C"))
    assert ta["value"].to_list() == [8.0]  # 'IFRS-FULL_ASSETS' (접두어 ifrs-full)가 ifrs_보다 앞
    tl = df.filter((pl.col("metric") == "tl") & (pl.col("period") == "P"))
    assert tl["value"].to_list() == [3.0] and tl["concept_used"].to_list() == ["ifrs_Liabilities"]


def test_duration_must_be_one_year_and_nil_dropped(tmp_path):
    r = "20250321000004"
    rows = _dur3(r, "ifrs-full_Revenue", SEP, (10.0, 9.0, 8.0))
    # 분기(3개월) duration은 버린다
    rows.append(
        _row(r, "ifrs-full_Revenue", "PFY2023dTQQ_x", SEP, 99.0,
             start=date(2023, 10, 1), end=date(2023, 12, 31))
    )  # fmt: skip
    rows.append(
        _row(r, "ifrs-full_Assets", "CFY2024eFY_n", SEP, 5.0, inst=date(2024, 12, 31), nil=True)
    )
    df = X.xbrl_values(_write(tmp_path, rows), None)
    rev = df.filter(pl.col("metric") == "rev").sort("period")
    assert rev["value"].to_list() == [8.0, 10.0, 9.0]  # BP, C, P
    assert "ta" not in set(df["metric"])


def test_non_december_fiscal_year(tmp_path):
    r = "20240930000005"
    rows = []
    for pre, y, v in (("CFY2024", 2024, 30.0), ("PFY2023", 2023, 20.0), ("BPFY2022", 2022, 10.0)):
        rows.append(_row(r, "ifrs-full_Assets", f"{pre}eFY_s", SEP, v, inst=date(y, 6, 30)))
    df = X.xbrl_values(_write(tmp_path, rows), None)
    got = {p: v for p, v in zip(df["period"], df["value"])}
    assert got == {"C": 30.0, "P": 20.0, "BP": 10.0}


def test_prefix_date_mismatch_dropped(tmp_path):
    r = "20250321000006"
    rows = _inst3(r, "ifrs-full_Assets", SEP, (3.0, 2.0, 1.0))
    # 결산기 변경: 접두어는 BP인데 날짜는 12개월 전(P 자리)
    rows.append(_row(r, "ifrs-full_Equity", "BPFY2023eFY_z", SEP, 5.0, inst=date(2023, 12, 31)))
    rows.append(_row(r, "ifrs-full_Equity", "CFY2024eFY_z", SEP, 6.0, inst=date(2024, 12, 31)))
    df = X.xbrl_values(_write(tmp_path, rows), None)
    te = df.filter(pl.col("metric") == "te")
    assert te["period"].to_list() == ["C"]


def test_rcept_and_metric_filter(tmp_path):
    rows = _inst3("20250321000007", "ifrs-full_Assets", SEP, (3.0, 2.0, 1.0))
    rows += _inst3("20250321000008", "ifrs-full_Assets", SEP, (6.0, 5.0, 4.0))
    rows += _inst3("20250321000008", "ifrs-full_Liabilities", SEP, (6.0, 5.0, 4.0), "L")
    g = _write(tmp_path, rows)
    df = X.xbrl_values(g, ["20250321000008"], ["ta"])
    assert set(df["rcept_no"]) == {"20250321000008"} and set(df["metric"]) == {"ta"}
    assert X.xbrl_values(g, [], None).height == 0
    with pytest.raises(KeyError):
        X.xbrl_values(g, None, ["nope"])


def test_metric_table_matches_spec():
    assert X.XBRL_METRICS["oi"][0] == "dart_OperatingIncomeLoss"
    assert X.XBRL_METRICS["ltb"][0] == "dart_LongTermBorrowingsGross"
    assert set(X.XBRL_METRICS) >= {"ip_op", "ip_fin", "ip_inv", "cap", "ta", "ni", "ocf"}


# ---------------------------------------------------------------- 실데이터 (없으면 skip)
_ROOT = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
_XBRL = _ROOT / "kr/raw/raw_postgres/snapshot_date=2026-09-30/source=sj2_remote/dart_xbrl_fact_raw"


@pytest.mark.skipif(not _XBRL.exists(), reason="XBRL 레이크 없음")
def test_real_lake_known_filing():
    df = X.xbrl_values(f"{_XBRL}/**/*.parquet", ["20250321001375"], ["ta", "ni"])
    g = {(r["metric"], r["period"]): r for r in df.iter_rows(named=True)}
    assert {r["fs_div"] for r in g.values()} == {"OFS"}
    assert g[("ta", "C")]["value"] == 151912649154
    assert g[("ta", "P")]["value"] == 114973284037
    assert g[("ta", "BP")]["value"] == 107702060968
    assert g[("ni", "C")]["value"] == -7253006357
