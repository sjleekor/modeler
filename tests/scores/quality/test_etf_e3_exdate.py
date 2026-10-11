"""E3 분배락일: 후보 R−1/R0/R+1, 개발 구간 점검, NAV 차감일 정렬, ETF-연 정합성."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e3_dense as dn
from modeler.scores.quality import etf_e3_exdate as ex

from .e3_synth import build_panel

ETFS = {
    "001A00": dict(name="KODEX 합성1", idx="합성지수", alpha=0.0, nav_noise=0.0002),
    "002A00": dict(name="KODEX 합성2", idx="합성지수", alpha=0.0, nav_noise=0.0002),
}


@pytest.fixture()
def world(tmp_path):
    panel, dist = build_panel(tmp_path, ETFS, quarterly_amt=0.01)
    return panel, dist, dn.build_dense(panel)


def test_adjust_days_are_day_before_record_date(world):
    panel, dist, d = world
    adj = ex.krx_adjust_days(d)
    for r in dist.iter_rows(named=True):
        e = int(ex.effective_idx(d, np.array([np.datetime64(r["rgt_std_dt"])]))[0])
        j = d.col[r["isu_cd"]]
        assert adj[e - 1, j] and not adj[e, j] and not adj[e + 1, j]


def test_effective_idx_weekend_and_edges(world):
    _, _, d = world
    sat = np.array([np.datetime64("2014-03-29")])  # 토요일 → 금요일(03-28)
    i = int(ex.effective_idx(d, sat)[0])
    assert d.dates[i] == np.datetime64("2014-03-28")
    out = np.array([np.datetime64("2010-01-01"), np.datetime64("2030-01-01")])
    assert ex.effective_idx(d, out).tolist() == [-1, -1]
    r = ex.rule_idx(d, np.array([np.datetime64("2014-03-31")]), "R-1")[0]
    r0 = ex.rule_idx(d, np.array([np.datetime64("2014-03-31")]), "R0")[0]
    rp = ex.rule_idx(d, np.array([np.datetime64("2014-03-31")]), "R+1")[0]
    assert (r0 - r, rp - r0) == (1, 1)


def test_candidate_rates_choose_r_minus_1(world):
    _, dist, d = world
    dev = ex.dev_distributions(d, dist)  # 2013~2015 분기 분배
    assert dev.height > 0
    rates = ex.candidate_match_rates(d, ex.krx_adjust_days(d), dev)
    c = rates["candidates"]
    assert c["R-1"]["rate_over_krx_etf"] == 1.0
    assert c["R0"]["rate_over_krx_etf"] == 0.0 and c["R+1"]["rate_over_krx_etf"] == 0.0
    ch = ex.choose_rule(rates)
    assert ch["chosen"] == "R-1"


def test_choose_rule_pending_below_90():
    rates = {"candidates": {k: {"rate_over_krx_etf": 0.5} for k in ex.RULE_SHIFT}}
    assert ex.choose_rule(rates)["chosen"] is None


def test_nav_alignment_majority_offset_minus_one(world):
    _, dist, d = world
    dev = ex.dev_distributions(d, dist)
    a = ex.nav_alignment(d, dev, "R-1")
    assert a["n_low_vol_etf"] > 0
    assert a["majority_offset"] == -1 and a["decision"] == "shift:-1"
    assert (
        ex.nav_alignment(d, dev, "R0")["majority_offset"] != 0
    )  # R0 이면 d 기준 NAV 낙폭이 d가 아니다


def test_exdate_matrix_and_consistency(world):
    _, dist, d = world
    p = ex.dist_rule_idx(d, dist, "R-1")
    D, info = ex.exdate_matrix(d, p, -1)
    assert info["n_skipped"] == 0 and (D > 0).sum() == dist.height
    cons = ex.etf_year_consistency(d, p, [2014, 2015])
    assert set(cons["ratio"].to_list()) == {1.0}
    # 한 분배의 기준일을 사흘 뒤로 밀면 그 ETF-연의 비율이 내려간다(8개 중 하나 → 0.875 < 0.9)
    bad = dist.with_columns(
        pl.when((pl.col("isu_cd") == "001A00") & (pl.col("rgt_std_dt") == date(2014, 6, 30)))
        .then(pl.lit(date(2014, 7, 3)))
        .otherwise(pl.col("rgt_std_dt"))
        .alias("rgt_std_dt")
    )
    pb = ex.dist_rule_idx(d, bad, "R-1")
    cb = ex.etf_year_consistency(d, pb, [2014])
    r = cb.filter(pl.col("isu_cd") == "001A00")["ratio"][0]
    assert r == pytest.approx(7 / 8) and r < ex.CONSISTENCY_MIN
    assert cb.filter(pl.col("isu_cd") == "002A00")["ratio"][0] == 1.0


def test_no_distribution_etf_year_is_not_in_consistency_table(world):
    _, dist, d = world
    p = ex.dist_rule_idx(d, dist.filter(pl.col("isu_cd") == "001A00"), "R-1")
    cons = ex.etf_year_consistency(d, p, [2014])
    assert "002A00" not in cons["isu_cd"].to_list()


def test_frozen_constants():
    assert ex.FROZEN_RULE in ex.RULE_SHIFT and ex.FROZEN_NAV_SHIFT in (-1, 0, 1)
