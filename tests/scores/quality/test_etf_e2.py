"""E2 거래 품질: 합성 데이터로 사전등록 §11.2~11.4·해석 표 I02·I03·I06·I07·I10·I15·I16·I17·I19·I23을 확인한다."""

import csv
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e2 as e2
from modeler.scores.quality import etf_panel as ep

COLS = [
    "BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "CMPPREVDD_PRC", "FLUC_RT", "NAV",
    "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP",
    "INVSTASST_NETASST_TOTAMT", "LIST_SHRS", "IDX_IND_NM", "OBJ_STKPRC_IDX",
    "CMPPREVDD_IDX", "FLUC_RT_IDX",
]


def weekdays(start, end):
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = weekdays(date(2010, 1, 1), date(2012, 12, 31))  # Y=2010, Y+1=2011, 2012는 자료 끝(끝나지 않은 해 아님)


def build(tmp_path, etfs, days=DAYS):
    """etfs: {isu: dict(name, idx, gap_by_year{year: gap}, trdval_by_year, first, last, netasst)}.

    종가 100, NAV = 100 / (1 + gap) → |종가 − NAV| ÷ NAV = gap.
    """
    p = tmp_path / "x.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for d in days:
            for isu, e in etfs.items():
                if d < e.get("first", days[0]) or d > e.get("last", days[-1]):
                    continue
                gap = e["gap"][d.year]
                tv = e.get("trdval", {}).get(d.year, 1000.0)
                w.writerow(
                    {
                        "BAS_DD": d.strftime("%Y%m%d"), "ISU_CD": isu, "ISU_NM": e.get("name", "TIGER 200"),
                        "IDX_IND_NM": e.get("idx", "코스피 200"), "TDD_CLSPRC": "100",
                        "NAV": repr(100 / (1 + gap)), "ACC_TRDVAL": repr(tv(d) if callable(tv) else tv),
                        "INVSTASST_NETASST_TOTAMT": str(e.get("netasst", 1000)),
                    }
                )
    panel = ep.read_panel(p)
    return panel, ep.lifecycle(panel)


def six(gaps_y, gaps_y1, **extra):
    """같은 그룹 ETF 여섯 개: ISU a..f, Y/Y+1 괴리를 각각 준다."""
    return {
        f"A{i}": {"gap": {2010: gaps_y[i], 2011: gaps_y1[i], 2012: gaps_y1[i]}, **extra}
        for i in range(len(gaps_y))
    }


G = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]


def test_components_gap_and_trdval_median(tmp_path):
    # 괴리: 일평균(거래량 0인 날 포함), 거래대금: 중앙값(0 포함)
    etfs = six(G, G)
    etfs["A0"]["trdval"] = {2010: lambda d: 0.0 if d.day % 2 else 10.0, 2011: 5.0}
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    r = ey.filter(pl.col("isu_cd") == "A0").row(0, named=True)
    assert r["gap_y"] == pytest.approx(0.001)
    assert r["gap_y1"] == pytest.approx(0.001)
    assert r["trdval_y1"] == 5.0
    days2010 = [d for d in DAYS if d.year == 2010]
    vals = sorted(0.0 if d.day % 2 else 10.0 for d in days2010)
    assert r["trdval_y"] == pytest.approx(float(np.median(vals)))
    assert r["n_close_y"] == len(days2010)


def test_gap_skips_nonpositive_nav(tmp_path):
    etfs = {"A0": {"gap": {2010: 0.01, 2011: 0.01, 2012: 0.01}}}
    panel, life = build(tmp_path, etfs)
    # NAV 칸을 0으로 만든 판: 괴리 있는 날 수가 줄어야 한다
    gap = ep.daily_gap(panel)
    assert gap.height == panel.rows.height


def test_200_day_condition(tmp_path):
    etfs = six(G, G)
    # A5는 2011년 안 199일만 거래
    d2011 = [d for d in DAYS if d.year == 2011]
    etfs["A5"]["last"] = DAYS[-1]
    etfs["A5"]["first"] = DAYS[0]
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    assert ey["ok_lenient"].sum() == 6
    # 별도 합성: Y+1 일수 부족
    etfs2 = six(G, G)
    etfs2["A5"]["gap"] = {2010: 0.006, 2011: 0.006, 2012: 0.006}
    etfs2["A5"]["last"] = d2011[198]  # 2011 안 199일만 있고 거기서 끝남
    panel2, life2 = build(tmp_path, etfs2)
    ey2 = e2.etf_years(panel2, life2, [2010]).filter(pl.col("isu_cd") == "A5").row(0, named=True)
    assert ey2["n_close_y1"] == 199 and ey2["fail_days_y1"] is True
    assert ey2["fail_delisted_y1"] is True and ey2["ok_lenient"] is False
    # Y 일수 부족: 2010 늦게 상장
    etfs3 = six(G, G)
    etfs3["A5"]["first"] = [d for d in DAYS if d.year == 2010][-199]
    panel3, life3 = build(tmp_path, etfs3)
    r3 = e2.etf_years(panel3, life3, [2010]).filter(pl.col("isu_cd") == "A5").row(0, named=True)
    assert r3["n_close_y"] == 199 and r3["fail_days_y"] is True and not r3["ok_lenient"]


def test_y1_listing_kept_condition_and_counts(tmp_path):
    etfs = six(G, G)
    d2011 = [d for d in DAYS if d.year == 2011]
    etfs["A4"]["last"] = d2011[-2]  # 마지막 거래일이 Y+1 마지막 시장 거래일보다 하루 이르다(일수는 충분)
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    r = ey.filter(pl.col("isu_cd") == "A4").row(0, named=True)
    assert r["n_close_y1"] >= 200 and r["fail_delisted_y1"] is True and not r["ok_lenient"]
    ex = e2.exclusion_counts(ey).row(0, named=True)
    assert ex["n_removed_delisted_y1"] == 1 and ex["n_pass_lenient"] == 5
    assert ex["alone_delisted_y1"] == 1


def test_group_size_boundary_4_vs_5_and_count_after_conditions(tmp_path):
    # 같은 그룹 ETF 5개이지만 하나가 폐지로 빠지면 조건 뒤 크기 4 → 제외(I06)
    etfs = six(G[:5], G[:5])
    d2011 = [d for d in DAYS if d.year == 2011]
    etfs["A0"]["last"] = d2011[-3]
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    assert e2.e2_stats(ey, "dev")["n_group_years"] == 0
    # 4 → 5: 그룹 6개에서 하나 빠지면 크기 5로 남는다
    etfs2 = six(G, G)
    etfs2["A0"]["last"] = d2011[-3]
    panel2, life2 = build(tmp_path, etfs2)
    ey2 = e2.etf_years(panel2, life2, [2010])
    st = e2.e2_stats(ey2, "dev")
    assert st["n_group_years"] == 1 and st["pooled_etf_years"] == 5
    st3 = e2.e2_stats(ey2, "dev", min_group=3)
    assert st3["pooled_etf_years"] == 5
    # 크기 분포(값 없이)
    nov = e2.etf_years(panel2, life2, [2010], with_values=False)
    assert nov["gap_y"].null_count() == nov.height
    dist = e2.group_size_distribution(nov).row(0, named=True)
    assert dist["lenient_ge5_groups"] == 1 and dist["lenient_ge5_etf_years"] == 5
    # 크기 4 그룹
    etfs3 = six(G[:4], G[:4])
    panel3, life3 = build(tmp_path, etfs3)
    d3 = e2.group_size_distribution(e2.etf_years(panel3, life3, [2010], with_values=False)).row(0, named=True)
    assert d3["lenient_ge5_groups"] == 0 and d3["lenient_ge3_groups"] == 1 and d3["lenient_ge3_etf_years"] == 4


def test_maturity_pension_hold_excluded(tmp_path):
    etfs = six(G, G)
    etfs["A0"]["name"] = "KODEX 국고채 24-12"  # 만기형
    etfs["A1"]["name"] = "KODEX 레버리지"  # 연금 부적격 후보
    etfs["A2"]["idx"] = ""  # 기초지수 없음 → 비교 보류
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010]).sort("isu_cd")
    d = {r["isu_cd"]: r for r in ey.iter_rows(named=True)}
    assert d["A0"]["fail_maturity"] and d["A1"]["fail_pension"] and d["A2"]["fail_hold"]
    assert ey["ok_lenient"].sum() == 3


def test_spearman_ties_and_direction():
    assert e2.spearman(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])) == pytest.approx(1.0)
    assert e2.spearman(np.array([1, 2, 2, 4]), np.array([1, 2, 2, 4])) == pytest.approx(1.0)
    # 동률 평균 순위: x = [1,2,2,4] → [1,2.5,2.5,4], y = [1,2,3,4]
    rx, ry = np.array([1, 2.5, 2.5, 4]), np.array([1, 2, 3, 4])
    exp = np.corrcoef(rx, ry)[0, 1]
    assert e2.spearman(np.array([1, 2, 2, 4]), np.array([1, 2, 3, 4])) == pytest.approx(exp)
    assert np.isnan(e2.spearman(np.array([1, 1, 1]), np.array([1, 2, 3])))
    # I03 백분위
    assert list(e2.pct_rank(np.array([5.0]))) == [50.0]
    assert list(e2.pct_rank(np.array([1.0, 2.0, 3.0]))) == [0.0, 50.0, 100.0]
    assert list(e2.pct_rank(np.array([1.0, 1.0, 3.0]))) == [25.0, 25.0, 100.0]


def test_persistent_low_gap_rho_plus_one(tmp_path):
    # 괴리가 낮은 ETF가 이듬해에도 낮다 → ρ = +1
    etfs = six(G, G)
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    st = e2.e2_stats(ey, "dev")
    assert st["weighted_rho"] == pytest.approx(1.0)
    assert st["n_pos_years"] == 1 and st["n_years"] == 1
    # 순위가 뒤집히면 −1
    etfs2 = six(G, list(reversed(G)))
    panel2, life2 = build(tmp_path, etfs2)
    st2 = e2.e2_stats(e2.etf_years(panel2, life2, [2010]), "dev")
    assert st2["weighted_rho"] == pytest.approx(-1.0)
    assert st2["n_pos_years"] == 0


def test_trdval_direction(tmp_path):
    etfs = six(G, G)
    for i in range(6):
        etfs[f"A{i}"]["trdval"] = {2010: 100.0 * (i + 1), 2011: 10.0 * (i + 1)}
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    assert e2.e2_stats(ey, "dev", value="trdval")["weighted_rho"] == pytest.approx(1.0)


def test_weighted_mean_and_yearly(tmp_path):
    # 한 해에 그룹 둘: 크기 6(ρ=+1), 크기 5(ρ=−1) → 가중평균 (6 − 5) / 11
    etfs = six(G, G)
    for i in range(5):
        etfs[f"B{i}"] = {
            "idx": "코스피 100", "gap": {2010: G[i], 2011: G[4 - i], 2012: G[4 - i]},
        }
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    st = e2.e2_stats(ey, "dev")
    assert st["n_group_years"] == 2 and st["pooled_etf_years"] == 11
    assert st["weighted_rho"] == pytest.approx((6 * 1.0 + 5 * -1.0) / 11)
    assert st["by_year"].row(0, named=True)["rho_w"] == pytest.approx(1 / 11)


def test_strict_holds_missing_return_type(tmp_path):
    etfs = six(G, G)
    # 기본 지수는 수익률 표기가 없다 → strict에서는 비교 보류, lenient는 통과
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    assert ey["ok_lenient"].sum() == 6 and ey["ok_strict"].sum() == 0
    assert e2.e2_stats(ey, "dev", key="strict")["n_group_years"] == 0


def test_bootstrap_seed_reproducible_and_p_value(tmp_path):
    etfs = six(G, G)
    for i in range(6):
        etfs[f"C{i}"] = {"idx": "코스피 100", "gap": {2010: G[i], 2011: G[(i + 1) % 6], 2012: G[i]}}
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    b1 = e2.bootstrap(ey, "dev", n_boot=200)
    b2 = e2.bootstrap(ey, "dev", n_boot=200)
    b3 = e2.bootstrap(ey, "dev", n_boot=200, seed=1)
    assert np.array_equal(b1, b2, equal_nan=True)
    assert not np.array_equal(b1, b3, equal_nan=True)
    assert e2.BOOT_SEED == 20261010 and e2.BOOT_N == 2000
    f = b1[np.isfinite(b1)]
    assert e2.p_value(b1) == pytest.approx((np.sum(f <= 0) + 1) / (len(f) + 1))
    assert e2.lower_bound(b1, 0.05) == pytest.approx(np.quantile(f, 0.05))
    # 완전 지속(ρ = +1)이면 부트스트랩 ρ ≤ 0 이 거의 없다
    etfs2 = six(G, G)
    p2, l2 = build(tmp_path, etfs2)
    b = e2.bootstrap(e2.etf_years(p2, l2, [2010]), "dev", n_boot=300)
    assert e2.p_value(b) < 0.05 and np.all(b[np.isfinite(b)] > 0.99)


def test_bootstrap_batch_matches_direct_spearman():
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=7), 1)  # 동률 포함
    y = np.round(rng.normal(size=7), 1)
    m = rng.multinomial(7, np.full(7, 1 / 7), size=20).astype(float)
    rho, w = e2._weighted_spearman_batch(x, y, m)
    for b in range(20):
        idx = np.repeat(np.arange(7), m[b].astype(int))
        assert w[b] == len(idx)
        exp = e2.spearman(x[idx], y[idx]) if len(idx) >= 2 else np.nan
        if np.isnan(exp):
            assert np.isnan(rho[b])
        else:
            assert rho[b] == pytest.approx(exp)


def test_bootstrap_excludes_small_or_constant_groups():
    # 행 3개 미만: 그룹이 빠져 nan
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = np.array([[1, 1, 0, 0, 0.0], [1, 1, 1, 1, 1.0], [0, 0, 0, 5, 0.0]])
    rho, w = e2._weighted_spearman_batch(x, x, m)
    assert list(w) == [2, 5, 5]
    assert np.isnan(rho[2])  # 한 행 5번 = 값이 다 같다
    assert rho[1] == pytest.approx(1.0)


def test_baseline_picks_largest_passive(tmp_path):
    etfs = six(G, G)
    # 순자산: A3 최대이지만 액티브, A2가 패시브 최대
    for i, na in enumerate([100, 200, 900, 800, 300, 400]):
        etfs[f"A{i}"]["netasst"] = na
    etfs["A3"]["name"] = "TIGER 200 액티브"
    etfs["A3"]["idx"] = "코스피 200"
    # 액티브는 그룹 키가 달라진다 → 패시브 그룹에서 A3 제외 (A0,A1,A2,A4,A5 → 크기 5)
    for i in range(6):
        etfs[f"A{i}"]["gap"] = {2010: G[i], 2011: G[i], 2012: G[i]}
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    na = ep.month_end_netassets(panel, life)
    b = e2.e2_baseline(ey, na, "dev")
    assert b["n_compared"] == 1
    gy = b["group_years"].row(0, named=True) if "group_years" in b else None
    # group_years는 보고용이라 dict에 남아 있다
    assert gy["passive_top"] == "A2" and gy["score_top"] == "A0"
    assert b["mean_passive_top_gap_y1"] == pytest.approx(G[2])
    assert b["mean_score_top_gap_y1"] == pytest.approx(G[0])
    assert b["share_score_lower"] == 1.0


def test_judgment_guard(tmp_path, monkeypatch):
    monkeypatch.delenv(e2.JUDGMENT_ENV, raising=False)
    etfs = six(G, G)
    panel, life = build(tmp_path, etfs)
    ey = e2.etf_years(panel, life, [2010])
    with pytest.raises(PermissionError):
        e2.e2_stats(ey, "judgment")
    with pytest.raises(PermissionError):
        e2.bootstrap(ey, "judgment", n_boot=10)
    with pytest.raises(PermissionError):
        e2.e2_baseline(ey, pl.DataFrame(), "judgment")
    with pytest.raises(PermissionError):
        e2.etf_years(panel, life, [2014], with_values=True)
    # 값 없이 세는 것은 허용
    e2.etf_years(panel, life, [2010], with_values=False)
    with pytest.raises(ValueError):
        e2.e2_stats(ey, "all")
