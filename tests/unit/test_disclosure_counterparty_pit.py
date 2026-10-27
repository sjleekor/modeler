"""상대방 상장사 연결 시점 기준(1.1.0) 단위 시험. 합성 DataFrame만 쓴다."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from modeler.etl.disclosures.counterparty import (
    build_pit_index,
    classify_counterparty,
    has_foreign_region_paren,
)


def _idx(snaps: dict[str, list[tuple[str, str]]], dm_rows, sm_rows):
    sn = pd.DataFrame(
        [
            dict(snapshot_id=f"s{i}", as_of_date=dt.date.fromisoformat(d), fetched_at=i)
            for i, d in enumerate(snaps)
        ]
    )
    items = pd.DataFrame(
        [
            dict(snapshot_id=f"s{i}", ticker=tk, name=nm)
            for i, rows in enumerate(snaps.values())
            for tk, nm in rows
        ]
    )
    dm = pd.DataFrame(dm_rows, columns=["corp_code", "ticker", "corp_name", "stock_name"])
    sm = pd.DataFrame(sm_rows, columns=["ticker", "name"])
    return build_pit_index(dm, sm, sn, items, "t")


def _one(raw, idx, dt_):
    r = classify_counterparty(raw, idx, dt_)
    assert len(r) == 1
    return r[0]


def test_not_listed_before_listing_and_after_delisting():
    fill = [(f"9{k:05d}", f"채움{k}") for k in range(20)]  # 부분 스냅샷 규칙에 안 걸리게
    idx = _idx(
        {
            "2016-01-29": [("000001", "가나다"), *fill],
            "2018-01-31": [("000001", "가나다"), ("000002", "신규산업"), *fill],
            "2020-01-31": [("000001", "가나다"), *fill],
        },
        [("C1", "000001", "가나다", "가나다"), ("C2", "000002", "신규산업", "신규산업")],
        [("000001", "가나다"), ("000002", "신규산업"), *fill],
    )
    pre = _one("신규산업", idx, "20170601")  # 상장 전
    assert pre["cp_class"] == "not_listed_at_date"
    assert pre["matched_corp_code"] == "C2" and pre["matched_ticker"] is None
    assert pre["ticker_ever"] == "000002"
    on = _one("신규산업", idx, "20180201")
    assert on["cp_class"] == "listed_pit_exact" and on["matched_ticker"] == "000002"
    assert on["snapshot_as_of"] == "2018-01-31" and on["name_at_date"] == "신규산업"
    post = _one("신규산업", idx, "20200601")  # 폐지 뒤
    assert post["cp_class"] == "not_listed_at_date"


def test_same_name_other_period_picks_then_ticker():
    # 제일모직형: 옛 종목(001300)은 2014에 폐지, 새 종목(028260)이 같은 이름을 이어받음
    idx = _idx(
        {"2013-12-31": [("001300", "제일모직")], "2015-06-30": [("028260", "제일모직")]},
        [("OLD", "001300", "제일모직", "제일모직"), ("NEW", "028260", "삼성물산", "삼성물산")],
        [("001300", "제일모직"), ("028260", "삼성물산")],
    )
    r = _one("제일모직(주)", idx, "20150715")
    assert r["cp_class"] == "listed_pit_normalized" and r["matched_ticker"] == "028260"
    assert r["name_basis"] == "snapshot_name"
    r0 = _one("제일모직", idx, "20140101")  # 첫 스냅샷 전 날짜는 첫 스냅샷
    assert r0["matched_ticker"] == "001300" and r0["snapshot_as_of"] == "2013-12-31"


def test_renamed_company_old_name_matches_by_then_name():
    idx = _idx(
        {"2015-01-30": [("034020", "두산중공업")], "2024-01-31": [("034020", "두산에너빌리티")]},
        [("D", "034020", "두산에너빌리티", "두산에너빌리티")],
        [("034020", "두산에너빌리티")],
    )
    r = _one("두산중공업", idx, "20160301")
    assert r["matched_ticker"] == "034020" and r["name_at_date"] == "두산중공업"
    assert r["name_basis"] == "snapshot_name"


def test_current_name_only_requires_then_name_compat():
    idx = _idx(
        {"2016-01-29": [("007460", "나라케이아이씨"), ("000020", "한성크린텍")]},
        [("A", "007460", "에이프로젠", "에이프로젠"), ("H", "000020", "한성크린텍", "한성크린텍")],
        [("007460", "에이프로젠"), ("000020", "한성크린텍")],
    )
    r = _one("(주)에이프로젠", idx, "20160301")  # 지금 이름만 맞고 그때 이름이 완전히 다름
    assert r["cp_class"] == "renamed_ticker_unverified"
    assert r["matched_ticker"] is None and r["candidate_tickers"] == "007460"
    assert r["name_at_date"] == "나라케이아이씨"


def test_containment_and_unknown_then_name_accepted():
    idx = _idx(
        {"2016-01-29": [("011780", "금호석유"), ("000030", "000030")]},
        [("K", "011780", "금호석유화학", "금호석유화학"), ("U", "000030", "미지기업", "미지기업")],
        [("011780", "금호석유화학"), ("000030", "미지기업")],
    )
    k = _one("금호석유화학", idx, "20160301")
    assert k["cp_class"] == "listed_pit_exact" and k["name_basis"] == "current_name_confirmed"
    u = _one("미지기업", idx, "20160301")  # 스냅샷에 이름이 없다 → 인정, 표시
    assert u["matched_ticker"] == "000030"
    assert u["match_method"] == "exact+then_name_unknown"
    assert u["name_basis"] == "current_name_then_unknown"


def test_foreign_region_paren_and_english_gloss():
    assert has_foreign_region_paren("삼성전자(중국)")
    assert has_foreign_region_paren("SK이노베이션(헝가리)")
    assert has_foreign_region_paren("현대모비스 (China)")
    assert not has_foreign_region_paren("에어부산 주식회사 (Air Busan Co., Ltd)")
    assert not has_foreign_region_paren("현대제철 (HYUNDAI STEEL COMPANY)")
    idx = _idx(
        {"2016-01-29": [("298690", "에어부산"), ("005930", "삼성전자")]},
        [("B", "298690", "에어부산", "에어부산"), ("S", "005930", "삼성전자", "삼성전자")],
        [("298690", "에어부산"), ("005930", "삼성전자")],
    )
    assert _one("삼성전자(중국)", idx, "20160301")["cp_class"] == "foreign_subsidiary"
    r = _one("에어부산 주식회사 (Air Busan Co., Ltd)", idx, "20160301")
    assert r["cp_class"] == "listed_pit_normalized" and r["matched_ticker"] == "298690"


def test_ambiguous_when_two_listed_same_name():
    idx = _idx(
        {"2016-01-29": [("000100", "동명"), ("000200", "동명")]},
        [],
        [],
    )
    assert _one("동명", idx, "20160301")["cp_class"] == "ambiguous_name"


def test_same_date_duplicate_snapshots_use_latest_fetched():
    sn = pd.DataFrame(
        [
            dict(snapshot_id="a", as_of_date=dt.date(2020, 1, 1), fetched_at=1),
            dict(snapshot_id="b", as_of_date=dt.date(2020, 1, 1), fetched_at=2),
        ]
    )
    items = pd.DataFrame(
        [
            dict(snapshot_id="a", ticker="000001", name="가"),
            dict(snapshot_id="b", ticker="000002", name="나"),
        ]
    )
    idx = build_pit_index(
        pd.DataFrame(columns=["corp_code", "ticker", "corp_name", "stock_name"]),
        pd.DataFrame(columns=["ticker", "name"]),
        sn,
        items,
    )
    assert len(idx.snaps) == 1 and idx.snaps[0].tickers == frozenset({"000002"})


def test_not_listed_with_several_old_tickers_keeps_all_tickers_ever():
    # 국민은행형: 같은 이름의 옛 종목이 둘이고 그때 둘 다 상장 아님 → corp는 비우고 티커는 다 남긴다
    idx = _idx(
        {"2016-01-29": [("000001", "다른회사")]},
        [("C1", "031150", "국민은행", "국민은행")],
        [("031150", "국민은행"), ("060000", "국민은행")],
    )
    r = _one("(주)국민은행", idx, "20160301")
    assert r["cp_class"] == "not_listed_at_date" and r["matched_ticker"] is None
    assert r["ticker_ever"] == "031150,060000"


def test_partial_snapshot_is_skipped():
    full = [(f"{i:06d}", f"종목{i}") for i in range(10)]
    snaps = {
        "2020-01-31": full,
        "2020-02-29": full[:5],  # 이웃 10개의 80% 미만 → 부분 스냅샷
        "2020-03-31": full[:9],
    }
    idx = _idx(snaps, [], [])
    assert [x.as_of for x in idx.snaps] == ["20200131", "20200331"]
    assert idx.excluded == [("20200229", 5, 9)]
    # 2020-03-10 공시는 부분 스냅샷이 아니라 1월 말 스냅샷을 쓴다
    assert idx.snap_at("20200310").as_of == "20200131"
