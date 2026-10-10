"""사전등록 §11.1~11.3 패널 규칙마다 작은 합성 예시 하나씩. 점수·결과와 무관하다."""

import csv
from datetime import date, timedelta
from pathlib import Path

from modeler.scores.quality import etf_panel as ep

COLS = [
    "BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "CMPPREVDD_PRC", "FLUC_RT", "NAV",
    "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP",
    "INVSTASST_NETASST_TOTAMT", "LIST_SHRS", "IDX_IND_NM", "OBJ_STKPRC_IDX",
    "CMPPREVDD_IDX", "FLUC_RT_IDX",
]


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_panel(tmp_path, rows):
    """rows: dict 목록(BAS_DD는 date). 안 준 칸은 빈 값."""
    p = tmp_path / "x.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["BAS_DD"] = r["BAS_DD"].strftime("%Y%m%d")
            w.writerow({c: r.get(c, "") for c in COLS})
    return ep.read_panel(p)


def day_rows(days, etfs):
    """etfs: {isu: {day_index: dict(칸...)}} 가 아니라 {isu: fn(i, d)->dict|None}."""
    rows = []
    for i, d in enumerate(days):
        for isu, fn in etfs.items():
            v = fn(i, d)
            if v is not None:
                rows.append({"BAS_DD": d, "ISU_CD": isu, "ISU_NM": f"N{isu}", **v})
    return rows


def test_closed_days_dropped_and_calendar(tmp_path):
    days = weekdays(date(2024, 1, 1), 6)
    # 3번째 날은 아무도 종가가 없다(휴장일 행)
    rows = day_rows(days, {"A": lambda i, d: {"TDD_CLSPRC": "" if i == 2 else "100"},
                           "B": lambda i, d: {"TDD_CLSPRC": "-" if i == 2 else "50"}})
    pn = make_panel(tmp_path, rows)
    assert pn.n_weekdays_in_file == 6
    assert pn.n_days == 5 and pn.n_closed_days == 1
    assert days[2] not in pn.calendar["date"].to_list()
    assert days[2] not in pn.rows["date"].to_list()
    assert pn.rows["day_idx"].max() == 4


def test_month_ends(tmp_path):
    days = [date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29),
            date(2024, 3, 1)]
    pn = make_panel(tmp_path, day_rows(days, {"A": lambda i, d: {"TDD_CLSPRC": "1"}}))
    # 마지막 날(3월 1일)도 자료 끝이라 그 달의 마지막 거래일로 센다
    assert pn.month_ends["date"].to_list() == [date(2024, 1, 31), date(2024, 2, 29),
                                              date(2024, 3, 1)]
    assert pn.end_date == date(2024, 3, 1)


def _status_for(tmp_path, k):
    days = weekdays(date(2024, 1, 1), 30)
    last = 29 - k
    rows = day_rows(days, {
        "ALIVE": lambda i, d: {"TDD_CLSPRC": "10"},
        "X": lambda i, d: {"TDD_CLSPRC": "10"} if i <= last else None,
    })
    pn = make_panel(tmp_path, rows)
    life = ep.lifecycle(pn).filter(ep.pl.col("isu_cd") == "X")
    assert life["k_after_last"][0] == k
    return life["status"][0]


def test_status_boundaries(tmp_path):
    assert _status_for(tmp_path, 0) == "listed"
    assert _status_for(tmp_path, 1) == "pending"
    assert _status_for(tmp_path, 5) == "pending"
    assert _status_for(tmp_path, 20) == "pending"
    assert _status_for(tmp_path, 21) == "disappeared"
    assert ep.event_status(0) == "listed" and ep.event_status(20) == "pending"
    assert ep.event_status(21) == "disappeared"


def test_mid_gap_counts_no_row_vs_blank_close(tmp_path):
    days = weekdays(date(2024, 1, 1), 12)

    def a(i, d):
        if i in (3, 4):  # 행 없음 2일
            return None
        if i in (7, 8, 9):  # 행은 있고 종가 빈 값 3일
            return {"TDD_CLSPRC": ""}
        return {"TDD_CLSPRC": "10"}

    rows = day_rows(days, {"ALIVE": lambda i, d: {"TDD_CLSPRC": "10"}, "A": a})
    life = ep.lifecycle(make_panel(tmp_path, rows)).filter(ep.pl.col("isu_cd") == "A")
    r = life.row(0, named=True)
    assert r["gap_days"] == 5 and r["gap_no_row"] == 2 and r["gap_blank_close"] == 3
    assert r["max_gap_run"] == 3 and r["has_gap"]
    assert r["n_close_days"] == 7 and r["first_date"] == days[0]
    # 끝에서 종가 없는 날(10, 11번째는 종가 있음) 영향 없음
    life2 = ep.lifecycle(make_panel(tmp_path, rows)).filter(ep.pl.col("isu_cd") == "ALIVE")
    assert not life2["has_gap"][0] and life2["max_gap_run"][0] == 0


def _netasst_case(tmp_path, vals):
    """25거래일(1월 1~31일 근처). 월말 = 마지막 날. vals: {day_index: 순자산}."""
    days = weekdays(date(2024, 1, 1), 23)  # 2024-01-31 까지 23평일
    assert days[-1] == date(2024, 1, 31)
    rows = day_rows(days, {"A": lambda i, d: {
        "TDD_CLSPRC": "10", "NAV": "10", "LIST_SHRS": "1000000",
        "INVSTASST_NETASST_TOTAMT": vals.get(i, "")}})
    pn = make_panel(tmp_path, rows)
    return ep.month_end_netassets(pn).row(0, named=True), days


def test_month_end_netasst_uses_own_value(tmp_path):
    r, _ = _netasst_case(tmp_path, {22: "500", 21: "300"})
    assert r["netasst"] == 500 and r["lag_days"] == 0 and not r["filled"]


def test_month_end_netasst_zero_falls_back_within_5(tmp_path):
    # 월말 0, 직전 5거래일 중 가장 가까운 날(i=19, 3거래일 전)
    r, days = _netasst_case(tmp_path, {22: "0", 21: "0", 20: "", 19: "700", 18: "800"})
    assert r["netasst"] == 700 and r["lag_days"] == 3 and r["filled"]
    assert r["netasst_src_date"] == days[19] and r["raw_zero_or_null"]


def test_month_end_netasst_5th_day_ok_6th_not(tmp_path):
    r5, _ = _netasst_case(tmp_path, {17: "900"})  # 22-5
    assert r5["netasst"] == 900 and r5["lag_days"] == 5
    r6, _ = _netasst_case(tmp_path, {16: "900"})  # 22-6
    assert r6["netasst"] is None and r6["filled"] is False


def test_month_end_netasst_not_nav_times_shares(tmp_path):
    r, _ = _netasst_case(tmp_path, {})  # NAV 10 × 좌수 1,000,000 이 있어도 채우지 않는다
    assert r["netasst"] is None


def test_daily_gap_excludes_nav_zero_and_null(tmp_path):
    days = weekdays(date(2024, 1, 1), 4)
    navs = ["100", "0", "", "50"]
    closes = ["101", "10", "10", "45"]
    rows = day_rows(days, {"A": lambda i, d: {"TDD_CLSPRC": closes[i], "NAV": navs[i]}})
    g = ep.daily_gap(make_panel(tmp_path, rows))
    assert g["date"].to_list() == [days[0], days[3]]
    assert abs(g["gap"][0] - 0.01) < 1e-12 and abs(g["gap"][1] - 0.1) < 1e-12


def test_return_pairs_missing_index_breaks_pair(tmp_path):
    days = weekdays(date(2024, 1, 1), 6)
    nav = ["100", "101", "102", "103", "104", "105"]
    idx = ["10", "11", "-", "13", "", "15"]
    rows = day_rows(days, {"A": lambda i, d: {"TDD_CLSPRC": "9", "NAV": nav[i],
                                              "OBJ_STKPRC_IDX": idx[i]}})
    rp = ep.return_pairs(make_panel(tmp_path, rows))
    # 쌍이 되는 날은 1일(0·1 둘 다 있음)뿐. 2는 지수 '-', 3은 전일 없음, 4 빈 값, 5 전일 없음.
    assert rp["date"].to_list() == [days[1]]
    assert abs(rp["nav_ret"][0] - 0.01) < 1e-12 and abs(rp["idx_ret"][0] - 0.1) < 1e-12


def test_return_pairs_need_adjacent_trading_days(tmp_path):
    days = weekdays(date(2024, 1, 1), 4)
    # 이 ETF는 3번째 날 행이 없다 -> 4번째 날 쌍 없음
    rows = day_rows(days, {"ALIVE": lambda i, d: {"TDD_CLSPRC": "1"},
                           "A": lambda i, d: None if i == 2 else
                           {"TDD_CLSPRC": "9", "NAV": "100", "OBJ_STKPRC_IDX": "5"}})
    rp = ep.return_pairs(make_panel(tmp_path, rows))
    assert rp["date"].to_list() == [days[1]]


def test_listed_one_year_boundary():
    first = date(2024, 2, 29)
    assert not ep.is_listed_one_year(date(2025, 2, 27), first)
    assert ep.is_listed_one_year(date(2025, 2, 28), first)  # 2월 29일 + 12개월 = 2월 28일
    assert not ep.is_listed_one_year(date(2024, 12, 31), date(2024, 1, 4))
    assert ep.is_listed_one_year(date(2025, 1, 31), date(2024, 1, 4))
    # 2010-01-04부터 있던 ETF: 2010-12 월말은 거짓, 2011-01 월말부터 참
    assert not ep.is_listed_one_year(date(2010, 12, 31), date(2010, 1, 4))
    assert ep.is_listed_one_year(date(2011, 1, 31), date(2010, 1, 4))


def test_listed_one_year_expr_matches_scalar():
    pl = ep.pl
    firsts = [date(2024, 2, 29), date(2010, 1, 4), date(2023, 1, 31)]
    mes = [date(2025, 2, 28), date(2011, 1, 31), date(2023, 12, 31), date(2024, 1, 31)]
    df = pl.DataFrame({"first_date": firsts}).join(pl.DataFrame({"month_end": mes}), how="cross")
    got = df.with_columns(ep.listed_one_year_expr().alias("ok"))
    for r in got.iter_rows(named=True):
        assert r["ok"] == ep.is_listed_one_year(r["month_end"], r["first_date"])


# ---------------------------------------------------------------- 경로: 환경변수로 받고 코드에 절대 경로가 없다
def test_default_paths_follow_env(monkeypatch):
    for k in ("STOCK_DATA_ROOT", "MY_ROOT", "QUALITY_INTERP_TABLE", "QUALITY_E_INTERP_TABLE", "QUALITY_PREREG"):
        monkeypatch.delenv(k, raising=False)
    assert ep.default_interp_table() == str(Path("../stock_data") / ep.INTERP_TABLE_REL)
    assert ep.default_prereg() == str(Path("../my") / ep.PREREG_REL)
    monkeypatch.setenv("STOCK_DATA_ROOT", "/x/sd")
    monkeypatch.setenv("MY_ROOT", "/x/my")
    assert ep.default_interp_table() == "/x/sd/" + ep.INTERP_TABLE_REL
    assert ep.default_prereg() == "/x/my/" + ep.PREREG_REL
    monkeypatch.setenv("QUALITY_INTERP_TABLE", "/y/t.md")
    monkeypatch.setenv("QUALITY_PREREG", "/y/p.md")
    assert ep.default_interp_table() == "/y/t.md"
    assert ep.default_prereg() == "/y/p.md"


def test_repo_root_has_uv_lock():
    assert (ep.repo_root() / "uv.lock").exists()


def test_no_absolute_local_paths_in_quality_code():
    import re
    from pathlib import Path as P

    pat = re.compile("/private" + "/tmp|/Users" + "/whishaw|claude" + "-501")
    bad = []
    for d in (P(ep.__file__).parent, P(__file__).parent):
        for f in d.glob("*.py"):
            if f.name == P(__file__).name:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    bad.append(f"{f.name}:{i}")
    assert not bad, bad
