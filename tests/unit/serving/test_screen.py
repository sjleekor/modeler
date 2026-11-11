"""거르기 규칙 screen-v1 (06_screen_v1_rules.md): 상수는 규칙 문서 원문 그대로, 조건마다 합성 입력으로 확인합니다.

10-26 단위 재현(맥 사본이 있을 때만): KR 위험 조건 수가 03 §4.1과 같고, 규모 판정일만 03 모의처럼
매일 값(10-21)으로 옮기면 통과 330 · 선별 30 원순위가 03 목록과 같습니다.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.serving import screen as sc

# 06 §2·§3 원문을 그대로 옮긴 것입니다. 코드 상수와 둘 다 바꿔야만 이 시험이 지나갑니다(정의 변경은 screen-v2).
SPEC_INCLUDE = (
    "상장폐지|실질심사|관리종목|개선기간|정리매매|환기종목|의견거절|부적정|감사의견|자본잠식|"
    "자기자본10억원미만|회생절차|파산신청|부도발생|매출액50억원미만|파산선고|회생계획인가"
)
SPEC_EXCLUDE = (
    r"미지정|기각|사유 해소|대상 제외|상장유지|종속회사|자회사|피보증|미진행|해제|취하|철회|조회공시요구|"
    r"이전상장|해외증권|의안상정|우선주|신주인수권|사채|채권|^상장폐지결정$|사유\s*해소|출자법인|"
    r"관리종목지정우려종목\)?\s*\(?(주가|시가총액)|(주가|시가총액)(요건)? ?미달에 따른"
)


def test_constants_are_the_fixed_rule_text() -> None:
    assert sc.RULE_VERSION == "screen-v1" and sc.SCHEMA == "briefing-screen.v1"
    assert sc.KR_FILING_INCLUDE_RE == SPEC_INCLUDE
    assert sc.KR_FILING_EXCLUDE_RE == SPEC_EXCLUDE
    assert sc.KR_SPAC_RE == "스팩|기업인수목적"
    assert sc.KR_REIT_RE == "리츠|인프라$|맥쿼리인프라"
    assert (sc.KR_FILING_DAYS, sc.KR_HALT_DAYS, sc.KR_MIN_CLOSE, sc.KR_MIN_MCAP) == (1095, 20, 1000, 20_000_000_000)
    assert (sc.KR_SIZE_RANK, sc.KR_LIQUIDITY_RANK, sc.KR_SESSION_MIN_TICKERS) == (500, 500, 1000)
    assert (sc.WINDOW_SESSIONS, sc.LIQUIDITY_SESSIONS, sc.SCREEN_TOP) == (252, 20, 30)
    assert sc.US_ACQ_A_FORMS == {"SC 14D9", "SC 14D9/A"}
    assert sc.US_ACQ_B_FORMS == {"SC TO-T", "SC TO-T/A", "SC 13E3", "SC 13E3/A"}
    assert sc.US_ACQ_B_FILE_PREFIX == "005-"
    assert sc.US_ACQ_C_FORMS == {"DEFM14A", "DEFM14A/A", "PREM14A", "PREM14A/A"}
    assert sc.US_ACQ_C_ISSUER_FORMS == {"S-4", "S-4/A", "S-4MEF", "S-4 POS", "F-4", "F-4/A"}
    assert sc.US_ACQ_C_COMPLETION_ITEM == "2.01"
    assert sc.US_ACQ_C_BLOCK_ITEMS == {"5.01", "3.01"} and sc.US_ACQ_C_BLOCK_FORMS == {"15-12B", "15-12G"}
    assert sc.US_LISTING_ITEM == "3.01" and sc.US_LISTING_FORMS == {"15-12G", "15-12G/A"}
    assert "25-NSE" not in sc.US_LISTING_FORMS | sc.US_ACQ_C_BLOCK_FORMS  # 쓰지 않습니다(06 §3)
    assert (sc.US_NARROW_RANGE, sc.US_NARROW_SESSIONS, sc.US_LIQUIDITY_RANK) == (0.015, 20, 1500)


@pytest.mark.parametrize(("title", "hit"), [
    ("주권매매거래정지(상장적격성 실질심사 사유 발생)", True),
    ("상장적격성 실질심사 대상 결정", True),
    ("[기재정정]회생절차개시결정", True),
    ("감사보고서 제출(의견거절)", True),
    ("관리종목지정(자본잠식률 50%이상)", True),
    ("파산선고", True),
    ("부도발생", True),
    ("상장폐지결정", False),  # ^상장폐지결정$
    ("상장적격성 실질심사 사유해소", False),  # 사유\s*해소
    ("관리종목 지정사유 해소", False),
    ("관리종목지정우려종목(시가총액 미달) 안내", False),
    ("주가 미달에 따른 관리종목지정 우려 예고", False),
    ("자회사의 회생절차개시신청", False),
    ("출자법인 회생절차개시결정", False),
    ("해외증권시장 상장폐지", False),
    ("풍문또는보도에대한해명(미확정)", False),
    ("", False),
    (None, False),
])
def test_kr_filing_dictionary(title: str | None, hit: bool) -> None:
    assert sc.kr_filing_hit(title) is hit


def test_kr_kind() -> None:
    assert sc.kr_kind("005930", "삼성전자") == []
    assert sc.kr_kind("005935", "삼성전자우") == ["kind_preferred"]
    assert sc.kr_kind("0001A0", "덕양에너젠") == []
    assert sc.kr_kind("123450", "하나29호스팩") == ["kind_spac"]
    assert sc.kr_kind("123450", "어떤기업인수목적") == ["kind_spac"]
    assert sc.kr_kind("088980", "맥쿼리인프라") == ["kind_reit"]
    assert sc.kr_kind("123450", "한국인프라") == ["kind_reit"]
    assert sc.kr_kind("123450", "인프라웨어") == []  # 인프라$ 는 이름 끝만
    assert sc.kr_kind("123450", "SK리츠") == ["kind_reit"]


def test_rank_and_order_helpers() -> None:
    assert sc.rank_desc([5.0, None, 7.0, 5.0, 1.0]) == [2, None, 1, 2, 4]
    days = [date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2), date(2026, 11, 2), date(2026, 11, 3)]
    assert sc.month_first_sessions(days) == [date(2026, 9, 30), date(2026, 10, 1), date(2026, 11, 2)]
    assert sc.judgement_session(days, date(2026, 11, 3)) == date(2026, 11, 2)
    assert sc.judgement_session(days, date(2026, 10, 31)) == date(2026, 10, 1)
    assert sc.previous_month_first(days, date(2026, 11, 2)) == date(2026, 10, 1)
    rows = [{"rank": i, "symbol": f"S{i}"} for i in range(1, 6)]
    got = sc.screen_order(rows, [False, True, False, True, True], top=2)
    assert got == [{"screen_rank": 1, "rank": 2, "symbol": "S2"}, {"screen_rank": 2, "rank": 4, "symbol": "S4"}]


# --------------------------------------------------------------------------- KR synthetic
def _kr_sessions(asof: date, n: int) -> list[date]:
    days, d = [], asof
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def _kr_world(asof: date):
    """1,001 filler stocks every session (so each day is a session) plus the stocks under test."""
    sessions = _kr_sessions(asof, 300)
    fill = [f"9{i:04d}0" for i in range(1001)]
    tested = ["000010", "000020", "000030", "000040", "000050", "000060", "000070", "000080", "000095"]
    ohlcv, mcap = [], []
    for d in sessions:
        for t in fill:
            ohlcv.append((d, t, 5000, 100))
            mcap.append((d, t, 10_000_000_000, 1))  # tiny: never in the top 500 of the scored set
        for k, t in enumerate(tested):
            close = 900 if t == "000040" else 5000
            volume = 0 if (t == "000030" and d >= sessions[-20]) else 1000
            if t == "000060" and d < sessions[-100]:
                continue  # 100세션 이력
            ohlcv.append((d, t, close, volume))
            cap = 10_000_000_000 if t == "000050" else 100_000_000_000 * (k + 1)
            mcap.append((d, t, cap, 1_000_000_000 * (k + 1)))
    ohlcv = pl.DataFrame(ohlcv, schema=["trade_date", "ticker", "close", "volume"], orient="row")
    mcap = pl.DataFrame(mcap, schema=["trade_date", "ticker", "market_cap", "trading_value"], orient="row")
    filings = pl.DataFrame({"ticker": ["000020", "000070", "000080"],
                            "rcept_dt": [asof - timedelta(days=10), asof - timedelta(days=1095), asof],
                            "rcept_no": ["R1", "R2", "R3"],
                            "report_nm": ["상장적격성 실질심사 대상 결정", "상장폐지", "상장적격성 실질심사 사유해소"]})
    names = {"000010": "좋은회사", "000020": "공시회사", "000030": "정지회사", "000040": "저가회사",
             "000050": "소형회사", "000060": "신규회사", "000070": "오래된공시", "000080": "해소회사",
             "000095": "좋은회사우"}
    rankings = [{"rank": i + 1, "symbol": t, "market": "KOSPI", "name": names[t], "score": 1 - i / 100}
                for i, t in enumerate(tested)]
    return rankings, ohlcv, mcap, filings, sessions


def test_screen_kr_conditions() -> None:
    asof = date(2026, 11, 9)
    rankings, ohlcv, mcap, filings, sessions = _kr_world(asof)
    later = pl.DataFrame({"trade_date": [asof + timedelta(days=1)], "ticker": ["000010"], "close": [1], "volume": [0]})
    block = sc.screen_kr(rankings, asof=asof, ohlcv=pl.concat([ohlcv, later]), mcap=mcap, filings=filings)
    by = {r["symbol"]: r for r in block["rows"]}
    assert by["000010"]["reasons"] == [] and by["000010"]["pass"]
    assert by["000020"]["reasons"] == ["delisting_filing"] and by["000020"]["filings"] == ["R1"]
    assert by["000030"]["reasons"] == ["long_halt"] and by["000030"]["values"]["zero_volume_days"] == 20
    assert by["000040"]["reasons"] == ["close_below_1000"]
    assert by["000050"]["reasons"] == ["mcap_below_20b"]
    assert by["000060"]["reasons"] == ["short_history"]
    assert by["000070"]["reasons"] == []  # K − 1,095일 당일은 창 밖입니다 ((K − 1,095, K])
    assert by["000080"]["reasons"] == []  # 해소 공시는 제외 정규식
    assert by["000095"]["reasons"] == ["kind_preferred"]
    assert block["dates"]["judgement_session"] == "2026-11-02"
    assert block["dates"]["size_market_cap_date"] == "2026-11-02"
    assert block["dates"]["risk_market_cap_date"] == asof.isoformat()
    assert [x["symbol"] for x in block["screened"]] == ["000010", "000070", "000080"]
    assert block["summary"]["passed"] == 3 and block["summary"]["by_reason"]["delisting_filing"] == 1


def test_screen_kr_size_rank_and_missing_judgement_day() -> None:
    asof = date(2026, 11, 3)
    rankings, ohlcv, mcap, filings, sessions = _kr_world(asof)
    # 시총 표가 11-02(판정일)를 아직 갖지 않으면 전달 첫 세션(10-01) 값을 씁니다(06 §2 ③, 05 §6 C2).
    mcap = mcap.filter(pl.col("trade_date") < date(2026, 11, 2))
    block = sc.screen_kr(rankings, asof=asof, ohlcv=ohlcv, mcap=mcap, filings=filings.clear())
    assert block["dates"]["judgement_session"] == "2026-11-02"
    assert block["dates"]["size_market_cap_date"] == "2026-10-01"
    assert block["dates"]["risk_market_cap_date"] == "2026-10-31" or block["dates"]["risk_market_cap_date"] == "2026-10-30"
    # 순위는 점수 단면 안에서 셉니다. 순위 한도를 1로 낮추면 시총이 가장 큰 종목만 남습니다.
    old = sc.KR_SIZE_RANK
    try:
        sc.KR_SIZE_RANK = 1
        block = sc.screen_kr(rankings, asof=asof, ohlcv=ohlcv, mcap=mcap, filings=filings.clear())
    finally:
        sc.KR_SIZE_RANK = old
    sizes = {r["symbol"]: r["values"]["market_cap_rank"] for r in block["rows"]}
    assert sizes["000095"] == 1 and "size_liquidity" in block["rows"][0]["reasons"]


def test_screen_kr_refuses_a_non_session_asof() -> None:
    asof = date(2026, 11, 9)
    rankings, ohlcv, mcap, filings, _ = _kr_world(asof)
    with pytest.raises(sc.ScreenError, match="asof_not_a_session"):
        sc.screen_kr(rankings, asof=date(2026, 11, 8), ohlcv=ohlcv, mcap=mcap, filings=filings)


# --------------------------------------------------------------------------- US
def _filings(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"cik": pl.Int64, "form": pl.String, "filing_date": pl.Date,
                                      "file_number": pl.String, "items": pl.String, "accession": pl.String},
                        orient="row")


def test_us_filing_flags() -> None:
    d = date(2026, 6, 1)
    f = _filings([
        (1, "SC 14D9", d, "005-1", None, "a1"),  # (a) target
        (2, "SC TO-T", d, None, None, "b-acq"),  # acquirer row: no 005-
        (3, "SC TO-T/A", d, "005-9", None, "b1"),  # (b)
        (4, "DEFM14A", d, None, None, "c1"),  # (c) counts
        (5, "DEFM14A", d, None, None, "c2"), (5, "S-4", d, None, None, "s4"),  # stock-issuing acquirer
        (6, "PREM14A", d, None, None, "c3"), (6, "8-K", d + timedelta(days=30), None, "2.01,9.01", "k1"),
        (7, "PREM14A", d, None, None, "c4"), (7, "8-K", d + timedelta(days=30), None, "2.01", "k2"),
        (7, "8-K", d + timedelta(days=31), None, "3.01", "k3"),  # completed but delisting: counts
        (8, "8-K", d, None, "3.01,9.01", "l1"),
        (9, "15-12G/A", d, None, None, "l2"),
        (10, "8-K", d, None, "2.01", "x"), (10, "15-12B", d, None, None, "y"),  # nothing
        (11, "PREM14A", d, None, None, "c5"), (11, "8-K", d, None, "2.01", "k5"),  # same day: not after
    ])
    flags = sc.us_filing_flags(f)
    assert flags[1]["acquisition"] == ["a1"] and flags[1]["acquisition_rules"] == ["a"]
    assert 2 not in flags
    assert flags[3]["acquisition"] == ["b1"]
    assert flags[4]["acquisition"] == ["c1"]
    assert 5 not in flags and 6 not in flags
    assert flags[7]["acquisition"] == ["c4"] and flags[7]["listing"] == ["k3"]
    assert flags[8] == {"acquisition": [], "acquisition_rules": [], "listing": ["l1"]}
    assert flags[9]["listing"] == ["l2"]
    assert 10 not in flags
    assert flags[11]["acquisition"] == ["c5"]


def test_us_cik_map_uses_base_ticker_for_class_shares() -> None:
    uni = pl.DataFrame({"symbol": ["LEN", "LEN.B", "BRK.B", "X"], "cik": [10, None, None, 20]})
    assert sc.us_cik_map(uni, ["LEN.B", "BRK.B", "X", "Z"]) == {"LEN.B": 10, "BRK.B": None, "X": 20, "Z": None}


def _us_world(asof: date):
    days, d = [], asof
    while len(days) < 280:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.sort()
    rows = []
    for i, d in enumerate(days):
        for s in ("GOOD", "FLAT", "GAP", "NEW", "ZERO", "ACQ", "DOT.B"):
            if s == "GAP" and i == 200:
                continue
            if s == "NEW" and i < len(days) - 100:
                continue
            close = 100.0 if s == "FLAT" else 100.0 + (i % 7)
            volume = 0 if s == "ZERO" else 1000
            rows.append((d, s, close, volume))
    prices = pl.DataFrame(rows, schema=["date", "symbol", "close", "volume"], orient="row")
    adjusted = prices.rename({"close": "adj_close", "volume": "adj_volume"})
    universe = pl.DataFrame({"date": [asof] * 3, "symbol": ["ACQ", "GOOD", "DOT"], "cik": [77, 1, 88]})
    filings = _filings([(77, "SC 14D9", asof - timedelta(days=3), "005-1", None, "acq1"),
                        (77, "SC 14D9", asof + timedelta(days=1), "005-1", None, "later"),
                        (88, "8-K", asof - timedelta(days=3), None, "3.01", "dot1")])
    rankings = [{"rank": i + 1, "symbol": s} for i, s in enumerate(("FLAT", "GOOD", "GAP", "NEW", "ZERO", "ACQ", "DOT.B"))]
    return rankings, prices, adjusted, filings, universe


def test_screen_us_conditions() -> None:
    asof = date(2026, 11, 6)
    rankings, prices, adjusted, filings, universe = _us_world(asof)
    block = sc.screen_us(rankings, asof=asof, prices=prices, adjusted=adjusted, filings=filings, universe=universe)
    by = {r["symbol"]: r for r in block["rows"]}
    assert by["FLAT"]["reasons"] == ["narrow_range"] and by["FLAT"]["values"]["range_20"] == 0.0
    assert by["GOOD"]["reasons"] == [] and by["GOOD"]["values"]["range_20"] == pytest.approx(6 / 100 * 100 / 100, abs=0.07)
    assert by["GAP"]["reasons"] == ["trading_gap"]
    assert by["NEW"]["reasons"] == ["short_history"]
    assert by["ZERO"]["reasons"] == ["trading_gap", "liquidity"] and by["ZERO"]["values"]["range_20"] is None
    assert by["ACQ"]["reasons"] == ["acquisition_target"] and by["ACQ"]["filings"] == ["acq1"]
    assert by["DOT.B"]["reasons"] == ["listing_deficiency"] and by["DOT.B"]["values"]["cik"] == 88
    assert block["dates"]["judgement_session"] == "2026-11-02"
    assert block["dates"]["filing_basis_date"] == (asof - timedelta(days=3)).isoformat()
    assert [x["symbol"] for x in block["screened"]] == ["GOOD"]
    assert block["summary"]["filings_unchecked"] == 4  # FLAT, GAP, NEW, ZERO 에는 cik이 없습니다
    old = sc.US_LIQUIDITY_RANK
    try:
        sc.US_LIQUIDITY_RANK = 0
        block = sc.screen_us(rankings, asof=asof, prices=prices, adjusted=adjusted, filings=filings, universe=universe)
    finally:
        sc.US_LIQUIDITY_RANK = old
    assert all("liquidity" in r["reasons"] for r in block["rows"])


def test_document_and_targets() -> None:
    doc = sc.failure_document(unit="2026-11-10", invocation_id="i", report_sha256="r", release="rel", reason="timeout")
    assert doc["status"] == "failed" and set(doc["models"]) == set(sc.MODEL_MARKET)
    assert sc.screened_targets(doc, sc.KR_MODEL_ID) == []
    doc["models"][sc.KR_MODEL_ID] = {"status": "ok", "screened": [{"screen_rank": 1, "rank": 47, "symbol": "039490"}]}
    assert sc.screened_targets(doc, sc.KR_MODEL_ID) == [{"screen_rank": 1, "rank": 47, "symbol": "039490"}]
    assert sc.screened_targets({**doc, "rule_version": "screen-v2"}, sc.KR_MODEL_ID) == []
    assert json.loads(sc.document_bytes(doc))["rule_version"] == "screen-v1"


# --------------------------------------------------------------------------- 10-26 reproduction (Mac copies)
SD = Path("/Users/whishaw/wss_p/stock_data")
D1026 = SD / "kr/output/briefing_explain_p1/D=2026-10-26"
SNAP = SD / "kr/raw/raw_postgres/snapshot_date=2026-10-23/source=sj2_remote"
MOCK_1026 = [81, 97, 109, 144, 152, 160, 164, 165, 166, 190, 215, 216, 222, 240, 248, 283, 287, 289, 296, 298,
             306, 311, 316, 319, 320, 325, 333, 334, 335, 337]  # 03 §4.1 선별 30의 원순위


@pytest.mark.skipif(not (D1026 / "run/report-2026-10-26.json").is_file() or not SNAP.is_dir(),
                    reason="10-26 맥 사본이 없습니다")
def test_reproduces_the_1026_kr_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    report = json.loads((D1026 / "run/report-2026-10-26.json").read_bytes())
    model = next(m for m in report["markets"] if m["market"] == "KR")
    dirs = {"daily_ohlcv": D1026 / "raw_daily_ohlcv_snapshot=2026-10-26",
            "daily_market_cap": SNAP / "daily_market_cap",
            "dart_filing_receipt_raw": SNAP / "dart_filing_receipt_raw", "stock_master": SNAP / "stock_master"}
    asof = date(2026, 10, 23)
    inp = sc.load_kr_inputs(dirs, asof)
    block = sc.screen_kr(model["rankings"], asof=asof, ohlcv=inp.ohlcv, mcap=inp.mcap, filings=inp.filings,
                         master=inp.master)
    s = block["summary"]
    kinds = s["by_reason"]["kind_preferred"] + s["by_reason"]["kind_spac"] + s["by_reason"]["kind_reit"]
    assert s["total"] == 1980 and kinds == 67  # 겹침 없음
    # 03 §4.1: 장기 정지 39 · 1,000원 미만 40 · 200억 미만 57 · 이력 60. 공시는 06 사전으로 148 → 96
    assert [s["by_reason"][k] for k in ("long_halt", "close_below_1000", "mcap_below_20b", "short_history")] \
        == [39, 40, 57, 60]
    assert s["by_reason"]["delisting_filing"] == 96
    assert block["dates"]["judgement_session"] == "2026-10-01" and s["passed"] == 321  # 월 1회 판정(06 §1)
    # 03 모의는 규모를 매일 값(10-21)으로 봤습니다. 판정일만 그 날로 옮기면 03 목록이 그대로 나옵니다.
    monkeypatch.setattr(sc, "judgement_session", lambda sessions, a: date(2026, 10, 21))
    daily = sc.screen_kr(model["rankings"], asof=asof, ohlcv=inp.ohlcv, mcap=inp.mcap, filings=inp.filings,
                         master=inp.master)
    assert daily["summary"]["passed"] == 330 and daily["summary"]["top_n_kept"] == 2
    assert [x["rank"] for x in daily["screened"]] == MOCK_1026
