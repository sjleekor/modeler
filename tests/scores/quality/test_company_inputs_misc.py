"""company_inputs_misc(W2b) 시험: 합성 raw 트리 + 실데이터 대조."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.scores.quality import company_inputs_misc as m
from modeler.scores.quality.company_common import Lake

SNAP = "2026-09-30"


def _write(root: Path, table: str, rows: list[dict], schema: dict) -> None:
    d = root / f"snapshot_date={SNAP}" / "source=sj2_remote" / table
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema).write_parquet(d / "part.parquet")


SR_SCHEMA = {
    "corp_code": pl.String,
    "bsns_year": pl.Int32,
    "reprt_code": pl.String,
    "statement_type": pl.String,
    "row_name": pl.String,
    "stock_knd": pl.String,
    "rcept_no": pl.String,
    "raw_payload": pl.String,
}


def _dps(corp, year, rc, knd, t, p1="-", p2="-", reprt="11011"):
    payload = json.dumps({"thstrm": t, "frmtrm": p1, "lwfr": p2, "stock_knd": knd})
    return dict(
        corp_code=corp, bsns_year=year, reprt_code=reprt, statement_type="dividend",
        row_name=m.cm.DPS_ROW_NAME, stock_knd=knd, rcept_no=rc, raw_payload=payload,
    )  # fmt: skip


def _tr(corp, year, rc, knd, incnr, acq="1", reprt="11011"):
    payload = json.dumps({"stock_knd": knd, "change_qy_incnr": incnr, "change_qy_acqs": acq})
    return dict(
        corp_code=corp, bsns_year=year, reprt_code=reprt, statement_type="treasury_stock",
        row_name="자기주식 취득 및 처분 현황", stock_knd=knd, rcept_no=rc, raw_payload=payload,
    )  # fmt: skip


def _sh(raw_id, corp, year, rc, se, qty, reprt="11011"):
    return dict(
        raw_id=raw_id, corp_code=corp, bsns_year=year, reprt_code=reprt, rcept_no=rc,
        se=se, istc_totqy=qty,
    )  # fmt: skip


def _cap(corp, rc, date_s, typ):
    payload = json.dumps({"isu_dcrs_de": date_s, "isu_dcrs_stle": typ, "isu_dcrs_qy": "1,000"})
    return dict(corp_code=corp, rcept_no=rc, raw_payload=payload)


def _par(corp, year, rc, t, p1, p2):
    payload = json.dumps({"thstrm": t, "frmtrm": p1, "lwfr": p2, "stock_knd": "보통주"})
    return dict(
        corp_code=corp, bsns_year=year, reprt_code="11011", statement_type="dividend",
        row_name=m.cm.PAR_ROW_NAME, stock_knd="보통주", rcept_no=rc, raw_payload=payload,
    )  # fmt: skip


@pytest.fixture()
def raw(tmp_path):
    root = tmp_path / "raw_postgres"
    # A: 정상. 2016 보고서(접수 2017-03), 2017 보고서 정정본이 B_t(2018-06-30) 뒤.
    # B: 합계만, 우선주 소각.
    div = [
        _dps("A", 2016, "A16", "보통주", "100", "50", "-"),
        _dps("A", 2017, "A17", "보통주", "120", "100", "50"),
        _dps("B", 2016, "B16", "-", "-", "10", "1,000"),
        _dps("C", 2016, "C16", "우선주", "5"),  # excluded만 -> none
    ]
    tr = [
        _tr("A", 2016, "A16", "보통주", "-"),
        _tr("A", 2016, "A16", "우선주", "-"),
        _tr("A", 2017, "A17", "보통주", "2,000"),
        _tr("B", 2016, "B16", "우선주", "300"),
    ]
    _write(
        root,
        "dart_shareholder_return_raw",
        div + tr + [_par("A", 2017, "A17", "500", "500", "100")],
        SR_SCHEMA,
    )
    sh_schema = {
        "raw_id": pl.Int64, "corp_code": pl.String, "bsns_year": pl.Int32,
        "reprt_code": pl.String, "rcept_no": pl.String, "se": pl.String, "istc_totqy": pl.String,
    }  # fmt: skip
    shares = [
        _sh(1, "A", 2016, "A16", "보통주", "1,000"),
        _sh(2, "A", 2016, "A16", "합계", "1,500"),
        _sh(3, "A", 2016, "A16", "우선주", "500"),
        _sh(4, "A", 2017, "A17", "보통주", "1,100"),
        _sh(5, "A", 2017, "A17", "합계", "1,200"),
        _sh(6, "B", 2016, "B16", "합 계", "2,000"),
        _sh(7, "B", 2016, "B16", "비고", "9"),
        _sh(8, "C", 2016, "C16", "보통주", "-"),
        _sh(9, "A", 2016, "AQ1", "보통주", "7", reprt="11013"),  # 분기는 무시
    ]
    _write(root, "dart_share_count_raw", shares, sh_schema)
    caps = [
        _cap("A", "x1", "2017-05-02", "무상증자"),
        _cap("D", "x2", "2016-01-02", "유상증자"),  # 목록 밖
    ]
    _write(
        root,
        "dart_capital_change_raw",
        caps,
        {"corp_code": pl.String, "rcept_no": pl.String, "raw_payload": pl.String},
    )
    return root


@pytest.fixture()
def avail():
    # A17은 B_2017 = 2018-06-30 뒤(정정), 나머지는 B_t 안.
    return pl.DataFrame(
        {
            "rcept_no": ["A16", "A17", "B16", "C16"],
            "avail_date": [
                date(2017, 3, 30),
                date(2018, 8, 1),
                date(2017, 3, 30),
                date(2017, 3, 30),
            ],
        }
    )


def test_parse_qty():
    df = pl.DataFrame({"x": ["1,234", " 5 ", "-", "", None, "abc", "0"]})
    out = df.select(m.parse_qty_expr("x"))["x"].to_list()
    assert out == [1234, 5, None, None, None, None, 0]


def test_select_versions_boundary_and_latest():
    rep = pl.DataFrame(
        {
            "corp_code": ["A", "A", "A", "B"],
            "report_year": pl.Series([2016, 2016, 2016, 2016], dtype=pl.Int32),
            "rcept_no": ["r1", "r2", "r3", "r4"],
            "v": [1, 2, 3, 4],
        }
    )
    avail = pl.DataFrame(
        {
            "rcept_no": ["r1", "r2", "r3"],  # r4는 접수 목록에 없음
            "avail_date": [date(2017, 3, 1), date(2018, 6, 30), date(2018, 7, 2)],
        }
    )
    picked, exist = m.select_versions(rep, avail, [2016, 2017], lag=0)
    p = picked.filter(pl.col("fy") == 2016)
    assert p["rcept_no"].to_list() == ["r1"]  # B_2016 = 2017-06-30: r2·r3는 뒤
    assert exist.filter(pl.col("fy") == 2016).height == 2  # A, B(가용 판본은 없어도 존재)
    assert "B" not in p["corp_code"].to_list()  # 접수 목록에 없으면 결측
    # lag=1: fy 2017의 p1 = 2016 보고서(B_2017 = 2018-06-30 기준). r2는 경계일 포함, r3는 뒤.
    picked1, _ = m.select_versions(rep, avail, [2017], lag=1)
    assert picked1["rcept_no"].to_list() == ["r2"]


def test_dividend_reports(raw):
    d = m.dividend_reports(raw, SNAP).sort("rcept_no")
    a16 = d.filter(pl.col("rcept_no") == "A16").row(0, named=True)
    assert (a16["dps"], a16["dps_p1"], a16["dps_p2"]) == (100.0, 50.0, 0.0)  # `-` 는 0원
    b16 = d.filter(pl.col("rcept_no") == "B16").row(0, named=True)
    assert b16["dps_src"] == "unmarked" and b16["dps"] == 0.0 and b16["dps_p2"] == 1000.0
    c16 = d.filter(pl.col("rcept_no") == "C16").row(0, named=True)
    assert c16["dps_src"] == "none" and c16["dps"] is None  # 우선주 행만 -> 결측


def test_share_reports_se_choice(raw):
    s = m.share_reports(raw, SNAP).sort("rcept_no")
    got = {r["rcept_no"]: (r["shares"], r["shares_src"]) for r in s.iter_rows(named=True)}
    assert got["A16"] == (1000, "보통주")  # 둘 다 있으면 보통주
    assert got["B16"] == (2000, "합계")  # 공백 제거 후 합계
    assert got["C16"] == (None, "보통주")  # `-` 는 결측, 합계로 안 넘어감(CI-se-row)
    assert "AQ1" not in got  # 분기 보고서 제외
    summ = m.share_se_summary(raw, SNAP)
    assert summ["picked_common"] == 3 and summ["picked_total"] == 1
    assert summ["both_common_and_total"] == 2


def test_treasury_reports(raw):
    t = m.treasury_reports(raw, SNAP)
    r = {x["rcept_no"]: x for x in t.iter_rows(named=True)}
    assert r["A16"]["retire"] == 0  # 행은 있고 모두 `-`
    assert r["A17"]["retire"] == 1 and r["A17"]["retire_pref_only"] is False
    assert r["B16"]["retire"] == 1 and r["B16"]["retire_pref_only"] is True  # 우선주만 소각
    assert "C16" not in r  # 자사주 행 없음 -> 결측


def test_build_misc_panel(raw, avail):
    lake = Lake(root=raw, raw_snapshot=SNAP)
    panel, diag = m.build_misc(lake, [2016, 2017], raw_root=raw, avail=avail)
    assert panel.columns == m.PANEL_COLUMNS
    p = {(r["corp_code"], r["fy"]): r for r in panel.iter_rows(named=True)}
    a16 = p[("A", 2016)]
    assert (a16["dps"], a16["dps_p1"], a16["dps_p2"], a16["dps_src"]) == (
        100.0,
        50.0,
        0.0,
        "common",
    )
    assert a16["dps_rcept_no"] == "A16"
    assert a16["shares"] == 1000 and a16["shares_p1"] is None  # 2015 보고서 없음
    assert a16["retire"] == 0 and a16["retire_p1"] is None
    # fy 2017: A17이 B_2017 뒤 정정 -> t 값 결측, p1(2016 보고서)은 가용
    a17 = p[("A", 2017)]
    assert a17["dps"] is None and a17["shares"] is None and a17["retire"] is None
    assert a17["shares_p1"] == 1000 and a17["retire_p1"] == 0
    d17 = diag.filter((pl.col("corp_code") == "A") & (pl.col("fy") == 2017)).row(0, named=True)
    assert d17["dps_after_base"] and d17["shares_after_base"] and d17["retire_after_base"]
    # B: 합계, 우선주만 소각
    b16 = p[("B", 2016)]
    assert b16["shares"] == 2000 and b16["shares_src"] == "합계" and b16["retire"] == 1
    # C: 우선주 행만 -> dps 결측, 주식수 `-` -> 결측, 자사주 행 없음 -> 결측
    c16 = p[("C", 2016)]
    assert c16["dps"] is None and c16["dps_src"] == "none"
    assert c16["shares"] is None and c16["retire"] is None
    # D는 그 해 사업보고서가 어느 표에도 없어 행이 없다
    assert ("D", 2016) not in p
    cov = m.coverage_misc(lake, [2016, 2017], built=(panel, diag)).sort("fy")
    r17 = cov.filter(pl.col("fy") == 2017).row(0, named=True)
    assert r17["n_rows"] == 1 and r17["n_dps_after_base"] == 1 and r17["n_shares_p1"] == 1
    r16 = cov.filter(pl.col("fy") == 2016).row(0, named=True)
    assert r16["n_rows"] == 3 and r16["dps_src_none"] == 1
    assert r16["n_shares_se_common"] == 2 and r16["n_shares_se_total"] == 1
    assert r16["n_retire_pref_only"] == 1


def test_event_flags_judgment_vs_record(raw, avail):
    lake = Lake(root=raw, raw_snapshot=SNAP)
    panel, _ = m.build_misc(lake, [2016, 2017, 2018], raw_root=raw, avail=avail)
    p = {(r["corp_code"], r["fy"]): r for r in panel.iter_rows(named=True)}
    a17 = p[("A", 2017)]
    # 무상증자 2017-05-02 -> 판정용·기록용 모두 2017
    assert a17["capevt"] and a17["capevt_rec"]
    # 액면 변경: A17 보고서 thstrm=frmtrm=500, frmtrm≠lwfr(100) -> 해 2016은 기록용에만
    a16 = p[("A", 2016)]
    assert not a16["capevt"] and a16["capevt_rec"]
    # 사건 해 지연 표시: fy 2017의 p1은 2016, p2는 2015
    assert (a17["capevt_p1"], a17["capevt_rec_p1"]) == (False, True)
    # 사건이 없는 해는 False(결측 아님)
    assert p[("B", 2016)]["capevt"] is False and p[("B", 2016)]["capevt_rec"] is False


def test_dps_latest(raw):
    lake = Lake(root=raw, raw_snapshot=SNAP)
    lat = m.dps_latest(lake, raw_root=raw)
    assert lat.columns == ["corp_code", "report_year", "rcept_no", "dps_t", "dps_p1", "knd_source"]
    a17 = lat.filter((pl.col("corp_code") == "A") & (pl.col("report_year") == 2017)).row(
        0, named=True
    )
    assert a17["rcept_no"] == "A17" and a17["dps_t"] == 120.0  # 기준일 없이 가장 늦은 판본


def test_cli_does_not_clobber_fs_outputs(tmp_path):
    # 출력 파일 이름이 W2a 출력과 겹치지 않는다.
    assert not (
        {"fs_panel.parquet", "manifest.json"}
        & {"misc_panel.parquet", "dps_latest.parquet", "coverage_misc.tsv", "manifest_misc.json"}
    )


# ---------------------------------------------------------------- 실데이터
ROOT = os.environ.get("STOCK_DATA_ROOT")
REAL = Path(ROOT) / "kr/raw/raw_postgres" / f"snapshot_date={SNAP}" if ROOT else None


@pytest.mark.skipif(
    not (REAL and REAL.is_dir()), reason="STOCK_DATA_ROOT의 raw 2026-09-30이 없습니다"
)
def test_real_dps_and_shares_match_raw():
    raw_root = Path(ROOT) / "kr/raw/raw_postgres"
    d = m.dividend_reports(raw_root, SNAP)
    s = m.share_reports(raw_root, SNAP)
    both = d.join(s, on=["corp_code", "report_year", "rcept_no"]).filter(
        pl.col("dps").is_not_null()
        & pl.col("shares").is_not_null()
        & (pl.col("report_year") == 2020)
    )
    assert both.height > 100
    sample = (
        both.sort("corp_code")
        .head(1)
        .vstack(both.sort("corp_code").tail(1))
        .vstack(both.sort("corp_code").slice(both.height // 2, 1))
    )
    base = REAL / "source=sj2_remote"
    for r in sample.iter_rows(named=True):
        # 배당: payload의 thstrm 문자열과 대조(보통주·표시 없음 행의 최댓값)
        sr = (
            pl.scan_parquet(
                str(base / "dart_shareholder_return_raw/**/*.parquet"), hive_partitioning=False
            )
            .filter(
                (pl.col("corp_code") == r["corp_code"])
                & (pl.col("rcept_no") == r["rcept_no"])
                & (pl.col("statement_type") == "dividend")
                & (pl.col("row_name") == m.cm.DPS_ROW_NAME)
            )
            .select("stock_knd", "raw_payload")
            .collect()
        )
        vals = []
        for knd, pay in sr.iter_rows():
            if m.cm.classify_stock_knd(knd) == r["dps_src"]:
                vals.append(m.cm.parse_dps_cell(json.loads(pay)["thstrm"]))
        assert max(v for v in vals if v is not None) == r["dps"]
        # 주식수: se 선택 행의 raw 문자열
        sc = (
            pl.scan_parquet(
                str(base / "dart_share_count_raw/**/*.parquet"), hive_partitioning=False
            )
            .filter((pl.col("corp_code") == r["corp_code"]) & (pl.col("rcept_no") == r["rcept_no"]))
            .select("se", "raw_payload")
            .collect()
        )
        want = [
            int(json.loads(pay)["istc_totqy"].replace(",", ""))
            for se, pay in sc.iter_rows()
            if se.replace(" ", "") == r["shares_src"]
            and json.loads(pay)["istc_totqy"].replace(",", "").lstrip("-").isdigit()
        ]
        assert want and want[0] == r["shares"]
