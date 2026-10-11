"""E3 표본·제외 일곱 단계·공통 창 수익률·분배락일 처리 (합성 패널)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e3_returns as rt
from modeler.scores.quality import etf_e3_sample as sm

from .e3_synth import DAYS, build_panel, code_map_frame, fee_frame, make_ctx

BASE = dict(idx="합성지수", nav_noise=0.0005)


def day(y, m, d):
    return date(y, m, d)


def test_seven_exclusions_in_order(tmp_path):
    etfs = {
        "001A00": dict(name="KODEX 정상1", alpha=0.04, **BASE),
        "002A00": dict(name="KODEX 정상2", alpha=0.02, **BASE),
        "003A00": dict(
            name="KODEX 지수변경",
            alpha=0.0,
            idx_before="옛지수",
            idx_change_at=day(2015, 6, 1),
            **BASE,
        ),
        "004A00": dict(name="KODEX 분할", alpha=0.0, split_at=day(2015, 6, 1), **BASE),
        "005A00": dict(
            name="KODEX NAV결측",
            alpha=0.0,
            nav_missing=set(d for d in DAYS if d.year == 2015 and d.month == 5),
            **BASE,
        ),
        "006A00": dict(name="KODEX 이상치", alpha=0.0, outlier_at=day(2015, 6, 10), **BASE),
        "007A00": dict(name="KODEX 정합성", alpha=0.0, **BASE),
        "008A00": dict(name="KODEX 대응없음", alpha=0.0, **BASE),
        "009A00": dict(name="KODEX 보수없음", alpha=0.0, **BASE),
    }
    panel, dist = build_panel(tmp_path, etfs)
    # 정합성: 007의 분배 기준일을 전부 사흘씩 민다
    dist = dist.with_columns(
        pl.when(pl.col("isu_cd") == "007A00")
        .then(pl.col("rgt_std_dt") + pl.duration(days=3))
        .otherwise(pl.col("rgt_std_dt"))
        .alias("rgt_std_dt")
    )
    cm = code_map_frame([(k, f"F{k}") for k in etfs if k != "008A00"])
    fee = fee_frame({f"F{k}": 0.2 for k in etfs if k not in ("008A00", "009A00")})
    ctx = sm.make_ctx(panel, dist, cm, fee)
    s = sm.build_sample(ctx, (2014, 2015, 2016))
    ey = s.ey.filter(pl.col("year") == 2015)
    where = dict(zip(ey["isu_cd"], ey["excluded_at"]))
    assert where["003A00"] == "1_index_change"
    assert where["004A00"] == "2_split_merge"
    assert where["005A00"] == "3_nav_missing"
    assert where["006A00"] == "4_outlier"
    assert where["007A00"] == "5_exdate_consistency"
    assert where["008A00"] == "6_dis_map"
    assert where["009A00"] == "7_fee"
    assert where["001A00"] == "" and where["002A00"] == ""
    c = s.counts.filter(pl.col("year") == 2015).to_dicts()[0]
    assert c["n_start"] == 9 and c["n_final_etf_years"] == 2
    assert [
        c[f"n_removed_{k}"]
        for k in (
            "1_index_change",
            "2_split_merge",
            "3_nav_missing",
            "4_outlier",
            "5_exdate_consistency",
            "6_dis_map",
            "7_fee",
        )
    ] == [1] * 7
    assert c["n_final_group_years"] == 1 and c["n_pairs_all"] == 1


def test_group_year_with_one_survivor_is_dropped(tmp_path):
    etfs = {
        "001A00": dict(name="KODEX 정상1", alpha=0.04, **BASE),
        "002A00": dict(
            name="KODEX 지수변경",
            alpha=0.0,
            idx_before="옛지수",
            idx_change_at=day(2015, 6, 1),
            **BASE,
        ),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    s = sm.build_sample(ctx, (2015,))
    assert s.counts["n_final_etf_years"][0] == 0 and s.gys == []


def test_min_trade_days_condition_from_e2(tmp_path):
    etfs = {
        "001A00": dict(name="KODEX 정상1", alpha=0.0, **BASE),
        "002A00": dict(name="KODEX 늦은상장", alpha=0.0, first=day(2015, 11, 1), **BASE),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    s = sm.build_sample(ctx, (2015,))
    assert s.counts["n_start"][0] == 0  # 크기 2 미만으로 시작부터 빠진다


def test_common_window_return_and_missing_day(tmp_path):
    miss = {day(2015, 3, 12)}
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.05, **BASE),
        "002A00": dict(name="KODEX B", alpha=-0.03, nav_missing=miss, **BASE),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    d = ctx.dense
    i, j = d.col["001A00"], d.col["002A00"]
    ra, rb = rt.pair_rtr(ctx, i, j, 2015)
    s, e = d.year_range(2015)
    f = np.ones(e - s + 1)
    g = np.ones(e - s + 1)
    for t in range(s, e + 1):
        for arr, col in ((f, i), (g, j)):
            a, b = d.nav[t, col], d.nav[t - 1, col]
            if np.isfinite(a) and np.isfinite(b):
                arr[t - s] = (a + ctx.D[t, col]) / b
    both = np.array(
        [
            np.isfinite(d.nav[t, i])
            and np.isfinite(d.nav[t - 1, i])
            and np.isfinite(d.nav[t, j])
            and np.isfinite(d.nav[t - 1, j])
            for t in range(s, e + 1)
        ]
    )
    # 결측일과 그 다음 날은 B의 r_t 가 없다 → 공통일에서 빠진다
    assert both.sum() == (e - s + 1) - 2
    assert ra == pytest.approx(np.prod(f[both]) - 1) and rb == pytest.approx(np.prod(g[both]) - 1)
    own_a, lost_a = rt.own_rtr(ctx, i, 2015)
    own_b, lost_b = rt.own_rtr(ctx, j, 2015)
    assert not lost_a and not lost_b
    assert own_a != ra  # 자기 창은 A 의 그 이틀을 포함한다


def test_exdate_in_non_common_day_drops_pair_year(tmp_path):
    # B 의 NAV 결측일이 A·B 의 NAV 차감일(2015-03 분기 E−2)과 겹치면 쌍-연을 뺀다
    from .e3_synth import quarter_end_idx

    qe = quarter_end_idx(DAYS)
    e2015 = next(e for e in qe if DAYS[e].year == 2015 and DAYS[e].month == 3)
    ex_day = DAYS[e2015 - 2]
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.05, **BASE),
        "002A00": dict(name="KODEX B", alpha=-0.03, nav_missing={ex_day}, **BASE),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    i, j = ctx.dense.col["001A00"], ctx.dense.col["002A00"]
    assert not sm.pair_window_ok(ctx, i, j, 2015)
    assert sm.pair_window_ok(ctx, i, j, 2016)
    s = sm.build_sample(ctx, (2014, 2015))
    c15 = s.counts.filter(pl.col("year") == 2015).to_dicts()[0]
    c14 = s.counts.filter(pl.col("year") == 2014).to_dicts()[0]
    assert c15["n_pairs_exdate_ok"] == 0 and c15["n_pairs_fail_window_Y"] == 1
    assert (
        c14["n_pairs_exdate_ok"] == 0 and c14["n_pairs_fail_window_Y1"] == 1
    )  # 2014 형성의 결과 창이 2015


def test_return_includes_distribution_on_nav_exday(tmp_path):
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.0, nav_noise=0.0, idx="합성지수"),
        "002A00": dict(name="KODEX B", alpha=0.0, nav_noise=0.0, idx="합성지수"),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    i = ctx.dense.col["001A00"]
    own, _ = rt.own_rtr(ctx, i, 2015)
    assert abs(own) < 1e-5  # 알파 0·잡음 0: 분배를 더해야 낙폭이 상쇄되어 0 이 된다(안 더하면 −4%)


def test_g_h_construction_and_pair_symmetry(tmp_path):
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.05, **BASE),
        "002A00": dict(name="KODEX B", alpha=0.0, **BASE),
    }
    ctx = make_ctx(tmp_path, etfs, {"001A00": 0.1, "002A00": 0.5})
    s = sm.build_sample(ctx, (2015,))
    v = rt.gy_values(ctx, s.gys[0])
    ra, rb = rt.pair_rtr(ctx, int(s.gys[0].cols[0]), int(s.gys[0].cols[1]), 2015)
    assert v.G[0, 1] == pytest.approx(ra + 0.1 / 100) and v.G[1, 0] == pytest.approx(rb + 0.5 / 100)
    ra1, rb1 = rt.pair_rtr(ctx, int(s.gys[0].cols[0]), int(s.gys[0].cols[1]), 2016)
    assert v.H[0, 1] == pytest.approx(ra1 + v.A[0] / 100) and v.A[0] == pytest.approx(0.1)
    assert v.R[1, 0] == pytest.approx(rb1)
