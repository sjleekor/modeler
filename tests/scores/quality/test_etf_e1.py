"""E1 상장 유지 위험(사전등록 §11.1~11.4, 해석 표 I01~I24)의 규칙마다 합성 예시 하나씩."""

import csv
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e1 as e1
from modeler.scores.quality import etf_panel as ep

COLS = [
    "BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "CMPPREVDD_PRC", "FLUC_RT", "NAV",
    "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP",
    "INVSTASST_NETASST_TOTAMT", "LIST_SHRS", "IDX_IND_NM", "OBJ_STKPRC_IDX",
    "CMPPREVDD_IDX", "FLUC_RT_IDX",
]


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def fake_panel(start: date, end: date) -> ep.Panel:
    """달력만 있는 패널(outcome_stats는 월말 달력과 자료 끝 날짜만 쓴다)."""
    days = weekdays(start, end)
    cal = (
        pl.DataFrame({"date": days})
        .with_row_index("day_idx")
        .with_columns(pl.col("day_idx").cast(pl.Int64))
        .with_columns(
            (pl.col("date").dt.month_end() != pl.col("date").shift(-1).dt.month_end())
            .fill_null(True)
            .alias("is_month_end")
        )
    )
    return ep.Panel(pl.DataFrame(), cal, 0, len(days), {})


# ---------------------------------------------------------------- 백분위(I03)
def test_percentile_formula_ties_and_n1():
    df = pl.DataFrame({"g": [1, 1, 1, 2, 3, 3, 3], "v": [1.0, 2.0, 3.0, 9.0, 1.0, 1.0, 3.0]})
    out = df.with_columns(e1.percentile_expr("v", ["g"]).alias("p"))
    assert out["p"].to_list() == [0.0, 50.0, 100.0, 50.0, 25.0, 25.0, 100.0]


def test_percentile_null_excluded_from_n():
    df = pl.DataFrame({"g": [1, 1, 1], "v": [1.0, None, 5.0]})
    out = df.with_columns(e1.percentile_expr("v", ["g"]).alias("p"))
    assert out["p"].to_list() == [0.0, None, 100.0]


# ---------------------------------------------------------------- 상관 쌍 경계(I05)
def _pairs(isu, idxs, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, len(idxs))
    return pl.DataFrame(
        {"isu_cd": isu, "day_idx": idxs, "nav_ret": x, "idx_ret": x + rng.normal(0, 0.002, len(idxs))}
    )


def test_corr_pairs_199_vs_200():
    m = 500
    p199 = _pairs("A", list(range(m - 251, m - 251 + 199)))
    p200 = _pairs("B", list(range(m - 251, m - 251 + 200)))
    grid = pl.DataFrame({"isu_cd": ["A", "B"], "day_idx": [m, m]})
    r = e1.window_corr(grid, pl.concat([p199, p200]))
    assert r["n_pairs"].to_list() == [199, 200]
    assert r["corr"][0] is None and r["corr"][1] is not None and r["corr"][1] > 0.9


def test_corr_window_edges_252_sessions():
    m = 600
    # 창 = [m-251, m]. m-252 와 m+1 은 창 밖이다.
    p = _pairs("A", [m - 252] + list(range(m - 251, m + 1)) + [m + 1])
    r = e1.window_corr(pl.DataFrame({"isu_cd": ["A"], "day_idx": [m]}), p)
    assert r["n_pairs"][0] == 252


def test_corr_constant_series_is_null():
    m = 400
    idx = list(range(m - 250, m + 1))
    p = pl.DataFrame({"isu_cd": "A", "day_idx": idx, "nav_ret": [0.01] * len(idx),
                      "idx_ret": np.linspace(0, 0.02, len(idx))})
    r = e1.window_corr(pl.DataFrame({"isu_cd": ["A"], "day_idx": [m]}), p)
    assert r["corr"][0] is None


# ---------------------------------------------------------------- 월말 점수(합성 패널)
def make_panel(tmp_path, rows):
    p = tmp_path / "x.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["BAS_DD"] = r["BAS_DD"].strftime("%Y%m%d")
            w.writerow({c: r.get(c, "") for c in COLS})
    return ep.read_panel(p)


def build_rows(days, spec):
    """spec: isu -> dict(name, idx, netasst, noise, start(date, optional))."""
    rows = []
    rng = np.random.default_rng(7)
    mkt = rng.normal(0, 0.01, len(days))
    for isu, s in spec.items():
        nav, ix = 100.0, 1000.0
        noise = rng.normal(0, s["noise"], len(days))
        for i, d in enumerate(days):
            nav *= 1 + mkt[i]
            ix *= 1 + mkt[i] + noise[i]
            if d < s.get("start", days[0]):
                continue
            rows.append({
                "BAS_DD": d, "ISU_CD": isu, "ISU_NM": s["name"], "IDX_IND_NM": s["idx"],
                "TDD_CLSPRC": f"{nav:.4f}", "NAV": f"{nav:.4f}", "OBJ_STKPRC_IDX": f"{ix:.4f}",
                "INVSTASST_NETASST_TOTAMT": str(s["netasst"]),
            })
    return rows


SPEC = {
    "D1": dict(name="KODEX 200", idx="코스피 200", netasst=10_000_000_000, noise=0.0005),
    "D2": dict(name="TIGER 200", idx="코스피 200", netasst=20_000_000_000, noise=0.001),
    "D3": dict(name="ACE 200", idx="코스피 200", netasst=30_000_000_000, noise=0.002),
    "D4": dict(name="SOL 200", idx="코스피 200", netasst=40_000_000_000, noise=0.004),
    "A1": dict(name="KODEX 액티브 200", idx="코스피 200", netasst=5_000_000_000, noise=0.01),
    "DZ": dict(name="HANARO 200", idx="코스피 200", netasst=0, noise=0.001),
    "F1": dict(name="TIGER 미국S&P500", idx="S&P 500", netasst=10_000_000_000, noise=0.01),
    "F2": dict(name="KODEX 미국나스닥100", idx="나스닥 100", netasst=90_000_000_000, noise=0.01),
    "F3": dict(name="ACE 미국S&P500", idx="S&P 500", netasst=3_000_000_000, noise=0.01),
    "MAT": dict(name="KODEX 국고채 25-06", idx="코스피 200", netasst=5_000_000_000, noise=0.001),
    "LEV": dict(name="KODEX 레버리지", idx="코스피 200", netasst=5_000_000_000, noise=0.001),
    "UNK": dict(name="XX 이름없음", idx="알수없는지수", netasst=5_000_000_000, noise=0.001),
    "YNG": dict(name="KODEX 신규 200", idx="코스피 200", netasst=7_000_000_000, noise=0.001,
                start=date(2022, 7, 1)),
}


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e1")
    days = weekdays(date(2022, 1, 3), date(2023, 12, 29))
    pn = make_panel(tmp, build_rows(days, SPEC))
    life = ep.lifecycle(pn)
    return pn, life, e1.monthly_scores(pn, life)


def test_classification_of_fixture(scored):
    _, life, _ = scored
    g = {r["isu_cd"]: r for r in life.iter_rows(named=True)}
    assert g["MAT"]["exclude_maturity"] and g["LEV"]["pension_ineligible_candidate"]
    assert g["UNK"]["region"] == "unknown" and g["F1"]["region"] == "foreign"
    assert g["D1"]["region"] == "domestic" and g["A1"]["active"] and not g["D1"]["active"]


def test_universe_exclusions(scored):
    _, _, sc = scored
    assert set(sc["isu_cd"].unique()) == {"D1", "D2", "D3", "D4", "A1", "DZ", "F1", "F2", "F3", "YNG"}


def test_listed_one_year_boundary(scored):
    _, _, sc = scored
    # 기존 ETF: 첫 거래일 2022-01-03 + 12개월 = 2023-01-03 → 월말 2023-01-31 부터
    d1 = sc.filter(pl.col("isu_cd") == "D1")["month_end"]
    assert d1.min() == date(2023, 1, 31)
    # YNG: 첫 거래일 2022-07-01 + 12개월 = 2023-07-01 → 2023-06-30 은 제외, 2023-07-31 부터
    y = sc.filter(pl.col("isu_cd") == "YNG")["month_end"]
    assert y.min() == date(2023, 7, 31)
    assert ep.is_listed_one_year(date(2023, 7, 3), date(2022, 7, 1))
    assert not ep.is_listed_one_year(date(2023, 6, 30), date(2022, 7, 1))


def test_pools_and_percentiles(scored):
    _, _, sc = scored
    m = date(2023, 12, 29)
    d = sc.filter(pl.col("month_end") == m)
    dom = d.filter(pl.col("region") == "domestic")
    # DZ는 순자산이 0(누락) → 행은 있고 풀 밖, 점수 null
    dz = dom.filter(pl.col("isu_cd") == "DZ").row(0, named=True)
    assert not dz["in_pool"] and dz["e1"] is None and dz["alert"] is None
    pool = dom.filter(pl.col("in_pool"))
    assert set(pool["isu_cd"]) == {"D1", "D2", "D3", "D4", "A1", "YNG"}
    assert pool["pool_size"].unique().to_list() == [6]
    p = {r["isu_cd"]: r for r in pool.iter_rows(named=True)}
    # 순자산 순위(풀 6개): A1 5억 < D1 100억 < YNG 70억? 정렬: A1(5)<YNG(7)<D1(10)<D2(20)<D3(30)<D4(40)
    assert p["A1"]["pct_netasst"] == 0.0 and p["D4"]["pct_netasst"] == 100.0
    assert p["YNG"]["pct_netasst"] == pytest.approx(20.0)
    # 액티브는 상관 백분위 풀이 따로 → A1 혼자라 50. 패시브 5개(D1~D4,YNG)끼리만 순위
    assert p["A1"]["pct_corr"] == 50.0
    passives = sorted((r["pct_corr"] for k, r in p.items() if k != "A1"))
    assert passives == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])
    # 잡음이 가장 작은 D1이 상관 최상위
    assert p["D1"]["pct_corr"] == 100.0
    # E1 = 두 백분위 평균
    assert p["D2"]["e1"] == pytest.approx((p["D2"]["pct_netasst"] + p["D2"]["pct_corr"]) / 2)
    # 규정 문턱 거리: 액티브는 −0.7
    assert p["A1"]["corr_gap"] == pytest.approx(p["A1"]["corr"] - 0.7)
    assert p["D1"]["corr_gap"] == pytest.approx(p["D1"]["corr"] - 0.9)
    # 해외형: 순자산 백분위 = E1
    fo = d.filter(pl.col("region") == "foreign")
    f = {r["isu_cd"]: r for r in fo.iter_rows(named=True)}
    assert f["F3"]["e1"] == 0.0 and f["F2"]["e1"] == 100.0 and f["F1"]["e1"] == 50.0
    assert f["F1"]["corr"] is None and f["F1"]["corr_gap"] is None


def test_alert_threshold_pool_of_6_is_not_flagged_except_bottom(scored):
    _, _, sc = scored
    d = sc.filter((pl.col("month_end") == date(2023, 12, 29)) & pl.col("in_pool"))
    # e1_pct는 풀 내 재백분위. 풀 6개면 가장 낮은 하나만 0 → 경보, 다음은 20
    dom = d.filter(pl.col("region") == "domestic").sort("e1_pct")
    assert dom["alert"].to_list()[0] is True and dom["alert"].to_list()[1] is False
    assert (dom["alert"] == (dom["e1_pct"] <= 10)).all()
    assert (dom["baseline_alert"] == (dom["pct_netasst"] <= 10)).all()


def test_no_scores_after_last_month_end(scored):
    pn, life, _ = scored
    sc = e1.monthly_scores(pn, life, last_month_end=date(2023, 6, 30))
    assert sc["month_end"].max() == date(2023, 6, 30)


# ---------------------------------------------------------------- 형성 월말 F(I01)
ME = fake_panel(date(2010, 1, 4), date(2017, 12, 29)).month_ends["date"].to_list()


def test_formation_month_end_cases():
    f = e1.formation_month_end
    assert f(date(2015, 3, 31), ME, 6) == date(2014, 9, 30)  # 정확히 월말
    assert f(date(2015, 3, 15), ME, 6) == date(2014, 8, 29)  # 9/15 이하 마지막 월말
    assert f(date(2015, 8, 31), ME, 6) == date(2015, 2, 27)  # 2월 28일은 토요일 → 27일
    assert f(date(2016, 8, 31), ME, 6) == date(2016, 2, 29)  # 윤년 2월 말
    assert f(date(2016, 3, 31), ME, 12) == date(2015, 3, 31)
    assert f(date(2016, 5, 31), ME, 3) == date(2016, 2, 29)
    assert f(date(2010, 3, 1), ME, 6) is None


# ---------------------------------------------------------------- 결과 통계
def _life(rows):
    base = dict(isu_nm="n", status="disappeared", exclude_maturity=False,
                pension_ineligible_candidate=False, region="domestic", active=False)
    return pl.DataFrame([{**base, **r} for r in rows])


def _scores(rows):
    base = dict(region="domestic", active=False, in_pool=True, alert=False,
                baseline_alert=False, e1_pct=50.0)
    return pl.DataFrame([{**base, **r} for r in rows])


PN = fake_panel(date(2010, 1, 4), date(2016, 12, 30))


def _case():
    life = _life([
        dict(isu_cd="E1", last_date=date(2013, 7, 15)),  # F = 2012-12-31
        dict(isu_cd="E2", last_date=date(2013, 9, 30)),  # F = 2013-03-29
        dict(isu_cd="E3", last_date=date(2013, 11, 20)),  # F = 2013-05-31, 점수 없음
        dict(isu_cd="MT", last_date=date(2013, 7, 15), exclude_maturity=True),
        dict(isu_cd="UK", last_date=date(2013, 7, 15), region="unknown"),
        dict(isu_cd="PN", last_date=date(2013, 7, 15), status="pending"),
        dict(isu_cd="LV", last_date=date(2013, 7, 15), pension_ineligible_candidate=True),
        dict(isu_cd="FO", last_date=date(2014, 7, 1), region="foreign"),  # F = 2013-12-31
        dict(isu_cd="J1", last_date=date(2016, 6, 30)),  # F = 2015-12-31 → 판정 구간
        dict(isu_cd="LIVE", status="listed", last_date=date(2016, 12, 30)),
    ])
    rows = [
        dict(isu_cd="E1", month_end=date(2012, 12, 31), alert=True, baseline_alert=False, e1_pct=5.0),
        dict(isu_cd="E2", month_end=date(2013, 3, 29), alert=False, baseline_alert=True, e1_pct=40.0),
        dict(isu_cd="E3", month_end=date(2013, 5, 31), in_pool=False, alert=None, baseline_alert=None, e1_pct=None),
        dict(isu_cd="FO", month_end=date(2013, 12, 31), region="foreign", alert=True, baseline_alert=True),
    ]
    for d, a in [(date(2012, 1, 31), True), (date(2012, 2, 29), False), (date(2012, 3, 30), False)]:
        rows.append(dict(isu_cd="LIVE", month_end=d, alert=a, baseline_alert=False))
    # E1의 연속 경보: 2012-10-31, 11-30, 12-31, 2013-01..06 (09-28은 경보 아님)
    for d, a in [(date(2012, 9, 28), False), (date(2012, 10, 31), True), (date(2012, 11, 30), True),
                 (date(2013, 1, 31), True), (date(2013, 2, 28), True), (date(2013, 3, 29), True),
                 (date(2013, 4, 30), True), (date(2013, 5, 31), True), (date(2013, 6, 28), True)]:
        rows.append(dict(isu_cd="E1", month_end=d, alert=a, baseline_alert=False))
    return life, _scores(rows)


def test_outcome_hit_miss_unscored_and_scope():
    life, sc = _case()
    st = e1.outcome_stats(sc, life, "dev", 6, panel=PN)["by_region"]
    d = st["domestic"]
    assert d["n_events"] == 3 and d["n_scored"] == 2 and d["n_unscored"] == 1
    assert d["unscored_share"] == pytest.approx(1 / 3)
    assert d["hits"] == 1 and d["hit_rate"] == 0.5
    assert d["hit_rate_unscored_as_miss"] == pytest.approx(1 / 3)
    assert d["baseline_hits"] == 1 and d["baseline_hit_rate"] == 0.5
    assert d["hit_minus_baseline"] == 0.0
    assert d["n_events_out_of_period"] == 1  # J1(F=2015-12)
    assert d["unscored_gt_cap"] and not d["g1_scored_ge_30"]
    f = st["foreign"]
    assert f["n_events"] == 1 and f["hits"] == 1 and f["hit_rate"] == 1.0


def test_event_table_fields():
    life, sc = _case()
    ev = e1.outcome_stats(sc, life, "dev", 6, panel=PN)["events"]
    assert set(ev["isu_cd"]) == {"E1", "E2", "E3", "FO"}
    r = {x["isu_cd"]: x for x in ev.iter_rows(named=True)}
    assert r["E1"]["formation_month_end"] == date(2012, 12, 31) and r["E1"]["alert"] is True
    assert r["E3"]["scored"] is False and r["E3"]["alert"] is None
    assert r["E2"]["formation_month_end"] == date(2013, 3, 29)


def test_lead_distribution_chain():
    life, sc = _case()
    ev = e1.outcome_stats(sc, life, "dev", 6, panel=PN)["events"]
    r = {x["isu_cd"]: x for x in ev.iter_rows(named=True)}
    # E1: L 앞 마지막 월말 2013-06-28(경보) → 거꾸로 2013-01-31까지 이어지고 2012-12-31... 연속 사슬:
    # 2013-06 ~ 2013-01, 2012-12, 2012-11, 2012-10 (09-28은 경보 아님) → 시작 2012-10-31, L=2013-07
    assert r["E1"]["lead_months"] == 9
    # E2: L 앞 마지막 점수 월말(2013-03-29)이 경보 아님 → 0
    assert r["E2"]["lead_months"] == 0
    assert r["E3"]["lead_months"] is None  # E3는 점수 월말이 하나(2013-05-31, 풀 밖)뿐이라 사슬이 없다


def test_lead_chain_breaks_at_unscored_month():
    life, sc = _case()
    sc = sc.filter(~((pl.col("isu_cd") == "E1") & (pl.col("month_end") == date(2013, 3, 29))))
    ev = e1.outcome_stats(sc, life, "dev", 6, panel=PN)["events"]
    r = {x["isu_cd"]: x for x in ev.iter_rows(named=True)}
    assert r["E1"]["lead_months"] == 3  # 3-29가 없어 사슬이 2013-04-30에서 끝 → L 2013-07


def test_horizon_changes_formation():
    life, sc = _case()
    ev3 = e1.outcome_stats(sc, life, "dev", 3, panel=PN)["events"]
    r = {x["isu_cd"]: x for x in ev3.iter_rows(named=True)}
    assert r["E1"]["formation_month_end"] == date(2013, 3, 29)
    assert r["E1"]["alert"] is True and r["E1"]["horizon_months"] == 3


def test_dev_ignores_judgment_period_scores():
    life, sc = _case()
    extra = _scores([dict(isu_cd="E1", month_end=date(2015, 2, 27), alert=False)])
    ev = e1.outcome_stats(pl.concat([sc, extra]), life, "dev", 6, panel=PN)["events"]
    r = {x["isu_cd"]: x for x in ev.iter_rows(named=True)}
    assert r["E1"]["lead_months"] == 9  # 2015 점수는 개발 구간 사슬에 안 들어온다


# ---------------------------------------------------------------- FAR(I09)
def test_far_window_and_pending_exclusion():
    pn = fake_panel(date(2010, 1, 4), date(2013, 6, 28))
    life = _life([
        dict(isu_cd="X", status="listed", last_date=date(2013, 6, 28)),
        dict(isu_cd="Y", status="disappeared", last_date=date(2012, 8, 10)),  # M=2011-12 창 밖(8개월 뒤 → 안)
        dict(isu_cd="P", status="pending", last_date=date(2012, 3, 15)),
        dict(isu_cd="Z", status="disappeared", last_date=date(2012, 1, 31)),  # L = M(2012-01-31)
    ])
    rows = []
    for isu in ("X", "Y", "P", "Z"):
        for d in (date(2011, 12, 30), date(2012, 1, 31), date(2012, 5, 31), date(2012, 6, 29)):
            rows.append(dict(isu_cd=isu, month_end=d, alert=(isu == "X" and d.month == 12)))
    sc = _scores(rows)
    far = e1.far_months(sc, life, pn, "dev")
    got = {(r["isu_cd"], r["month_end"]) for r in far.iter_rows(named=True)}
    # 2012-06-29 → M+12개월 = 2013-06-29 > 자료 끝 2013-06-28 → 전부 제외
    assert not any(m == date(2012, 6, 29) for _, m in got)
    # X는 사건 없음 → 3개월(2011-12-30, 2012-01-31, 2012-05-31)
    assert {m for i, m in got if i == "X"} == {date(2011, 12, 30), date(2012, 1, 31), date(2012, 5, 31)}
    # Y는 L=2012-08-10: 2011-12-30·2012-01-31·2012-05-31 모두 창 안 → 전부 제외
    assert not any(i == "Y" for i, _ in got)
    # P(pending) L=2012-03-15: 2011-12-30·2012-01-31 창 안 → 제외, 2012-05-31은 L이 M보다 앞 → 사건 아님 → 포함
    assert {m for i, m in got if i == "P"} == {date(2012, 5, 31)}
    # Z: L = 2012-01-31 = M → 소멸로 센다(2011-12-30·2012-01-31 제외), 2012-05-31은 L<M → 포함
    assert {m for i, m in got if i == "Z"} == {date(2012, 5, 31)}


def test_far_rate_in_stats():
    life, sc = _case()
    far = e1.outcome_stats(sc, life, "dev", 6, panel=PN)["by_region"]["domestic"]["far"]
    assert far["n_alert"] <= far["n_etf_months"] and far["far"] == far["n_alert"] / far["n_etf_months"]


# ---------------------------------------------------------------- 부트스트랩(I14·I02)
def test_p_value_formula_and_lower_bound():
    a = np.array([0.1, 0.3, 0.5, 0.2])
    assert e1.p_value(a) == pytest.approx((3 + 1) / (4 + 1))
    assert e1.p_value(np.array([0.9] * 9)) == pytest.approx(1 / 10)
    assert e1.lower_bound(np.arange(101, dtype=float), 0.05) == pytest.approx(5.0)
    assert e1.BOOTSTRAP_SEED == 20261010 and e1.BOOTSTRAP_B == 2000 and e1.HIT_NULL == 0.30


def test_bootstrap_seed_reproducible():
    life, sc = _case()
    a = e1.bootstrap(sc, life, "dev", 6, panel=PN, b=200)
    b = e1.bootstrap(sc, life, "dev", 6, panel=PN, b=200)
    c = e1.bootstrap(sc, life, "dev", 6, panel=PN, b=200, seed=1)
    assert np.array_equal(a["domestic"].arrays["hit"], b["domestic"].arrays["hit"], equal_nan=True)
    assert not np.array_equal(a["domestic"].arrays["far"], c["domestic"].arrays["far"], equal_nan=True)
    h = a["domestic"].arrays["hit"]
    assert np.nanmin(h) >= 0 and np.nanmax(h) <= 1 and len(h) == 200
    assert set(a["domestic"].arrays) == {"hit", "far", "base_hit", "base_far", "diff"}


# ---------------------------------------------------------------- judgment 보호
def test_judgment_requires_env(monkeypatch):
    life, sc = _case()
    monkeypatch.delenv(e1.JUDGMENT_ENV, raising=False)
    with pytest.raises(PermissionError):
        e1.outcome_stats(sc, life, "judgment", 6, panel=PN)
    with pytest.raises(PermissionError):
        e1.bootstrap(sc, life, "judgment", 6, panel=PN, b=5)
    monkeypatch.setenv(e1.JUDGMENT_ENV, "   ")
    with pytest.raises(PermissionError):
        e1.outcome_stats(sc, life, "judgment", 6, panel=PN)
    with pytest.raises(ValueError):
        e1.outcome_stats(sc, life, "x", 6, panel=PN)
    # 변수가 있으면 보호만 통과한다(합성 자료라 결과는 의미 없음)
    monkeypatch.setenv(e1.JUDGMENT_ENV, "1")
    assert e1.outcome_stats(sc, life, "judgment", 6, panel=PN)["period"] == "judgment"
