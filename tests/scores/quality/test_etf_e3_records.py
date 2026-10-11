"""E3 기록용 칸: R1·R7·R8·R11·R12·F구간별 폐지·R9·R13 (합성 자료)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e3_exdate as ex
from modeler.scores.quality import etf_e3_records as rec
from modeler.scores.quality import etf_e3_returns as rt
from modeler.scores.quality import etf_e3_run as run
from modeler.scores.quality import etf_e3_sample as sm

from .e3_synth import DAYS, build_panel, code_map_frame, fee_frame, make_ctx

BASE = dict(idx="합성지수", nav_noise=0.0003)
YEARS = (2014, 2015, 2016)


def three(extra=None):
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.06, netasst=1_000, **BASE),
        "002A00": dict(name="KODEX B", alpha=0.0, netasst=9_000, **BASE),
        "003A00": dict(name="KODEX C", alpha=-0.04, netasst=5_000, **BASE),
    }
    etfs.update(extra or {})
    return etfs


def test_r1_resid_and_r2():
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=5) * 0.003 for _ in range(30)]
    perfect = rec.r1_unexplained(xs, [0.8 * x + 0.1 for x in xs])
    assert perfect["beta"] == pytest.approx(0.8) and perfect["resid_sd_bp"] < 1e-6
    assert perfect["r2_within"] == pytest.approx(1.0)
    noisy = rec.r1_unexplained(xs, [0.8 * x + rng.normal(size=5) * 0.001 for x in xs])
    assert noisy["resid_sd_bp"] == pytest.approx(10.0, rel=0.2) and 0.0 < noisy["r2_within"] < 1.0
    none = rec.r1_unexplained(xs, [np.zeros(5) for _ in xs])
    assert none["r2_within"] is None or abs(none["r2_within"]) < 1e-9


def test_r7_r8_run_and_close_dist_position(tmp_path):
    etfs = three()
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    s = sm.build_sample(ctx, YEARS)
    r7 = rec.r7_general_account(ctx, s, 50, 1)
    assert r7["factor"] == 0.846 and np.isfinite(r7["C"])
    r8 = rec.r8_close_based(ctx, s, 50, 1)
    assert np.isfinite(r8["C"]) and r8["n_pairs_exdate_ok_close"] > 0
    c2 = rec.close_based_ctx(ctx)
    adj = ex.krx_adjust_days(ctx.dense)
    # 종가 기준 분배는 KRX 조정일(규칙 날짜, 이동 0)에 들어가고 NAV 기준은 하루 앞이다
    assert (c2.D > 0).sum() == (ctx.D > 0).sum()
    assert np.array_equal(c2.D > 0, adj & (c2.D > 0)) and (c2.D > 0)[adj].all() is np.True_
    assert c2.dense.nav is not ctx.dense.nav and np.allclose(
        c2.dense.nav, ctx.dense.close, equal_nan=True
    )
    # D 0.846 배: 총수익은 분배를 덜 더하므로 낮다
    i = ctx.dense.col["002A00"]
    a, _ = rt.own_rtr(ctx, i, 2015)
    import dataclasses

    b, _ = rt.own_rtr(dataclasses.replace(ctx, D=ctx.D * 0.846), i, 2015)
    assert b < a


def delisting_ctx(tmp_path):
    etfs = three(
        {"004A00": dict(name="KODEX D", alpha=0.02, last=date(2016, 10, 14), netasst=4_000, **BASE)}
    )
    panel, dist = build_panel(tmp_path, etfs)
    cm = code_map_frame([(k, f"F{k}") for k in etfs])
    fee = fee_frame({f"F{k}": 0.2 for k in etfs})
    fee = fee.filter(
        ~((pl.col("fund_code") == "F004A00") & (pl.col("base_date") > date(2016, 10, 14)))
    )
    return sm.make_ctx(panel, dist, cm, fee)


def test_r11_includes_delisted_with_partial_window(tmp_path):
    ctx = delisting_ctx(tmp_path)
    main = sm.build_sample(ctx, (2015,))
    part = sm.build_sample(ctx, (2015,), include_delisted_y1=True)
    assert main.counts["n_start"][0] == 3 and part.counts["n_start"][0] == 4
    assert (
        part.counts["n_final_etf_years"][0] == 4 and part.counts["n_removed_3_nav_missing"][0] == 0
    )
    out = rec.r11_delisted_partial(ctx, (2015,), 50, 1)
    assert out["n_delisted_partial_etf_years_in_final"] == 1 and out["n_final_etf_years"] == 4
    v = rt.gy_values(ctx, part.gys[0])
    k = part.gys[0].isu.index("004A00")
    assert v.delisted_y1[k] and np.isfinite(v.A[k]) and np.isfinite(v.H[k, 0] if k else v.H[0, k])
    # 폐지 ETF 쌍은 공통일이 폐지일까지, 다른 ETF의 분배락일이 폐지일 뒤여도 쌍을 안 뺀다
    assert part.gys[0].pair_ok[k].sum() == 3


def test_delisted_by_f_band(tmp_path):
    ctx = delisting_ctx(tmp_path)
    out = rec.delisted_by_f_band(ctx, (2015,))
    b = out["bands_pct"]["0.10<=F<0.30"]  # TER 0.2
    assert b["n_etf_years"] == 4 and b["n_delisted_y1"] == 1
    assert out["bands_pct"]["F<0.10"]["n_etf_years"] == 0


def test_r12_baseline_picks_top_g_vs_largest_passive(tmp_path):
    etfs = three()
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    s = sm.build_sample(ctx, (2015,))
    inp = run.build_inputs(ctx, s)
    out = rec.r12_baseline(ctx, s, inp["vals"])
    # (가) = A(알파 0.06), (나) = 순자산 최대 B(9,000) → A − B 의 H_own 차이는 양수, 약 6%p
    assert out["n_group_years"] == 1 and out["share_positive"] == 1.0
    assert 300 < out["diff_bp_weighted_mean"] < 900
    # 둘이 같으면(최고 G 가 순자산도 최대) 건너뛴다
    etfs2 = three()
    etfs2["001A00"]["netasst"] = 99_000
    ctx2 = make_ctx(
        tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path,
        etfs2,
        {k: 0.2 for k in etfs2},
    )
    s2 = sm.build_sample(ctx2, (2015,))
    out2 = rec.r12_baseline(ctx2, s2, run.build_inputs(ctx2, s2)["vals"])
    assert out2["n_same_etf"] == 1 and out2["n_group_years"] == 0


TR = "S&P 500 Total Return Index"


def test_r9_waits_without_index_values(tmp_path):
    etfs = {
        "379800": dict(name="KODEX S&P", alpha=0.0, nav_noise=0.0, idx="합성지수"),
        "448290": dict(name="TIGER S&P", alpha=0.0, nav_noise=0.0, idx="합성지수"),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    assert rec.r9_index_tracking(ctx)["status"] == "자료 대기"


def test_r9_uses_krx_reported_index_and_cross_values(tmp_path):
    days = {d: i for i, d in enumerate(DAYS)}
    lo, hi = days[date(2015, 1, 5)], days[date(2015, 12, 30)]

    def obj_a(i):
        return 100.0 * (1.0002**i) if lo <= i <= hi else None

    def obj_b(i):  # 448290 은 구간 뒤쪽 절반만 보고하고, 세 날은 값이 다르다
        if not (lo + 100 <= i <= hi):
            return None
        return obj_a(i) * (1.01 if i in (lo + 120, lo + 121, lo + 122) else 1.0)

    etfs = {
        "379800": dict(name="KODEX S&P", alpha=0.0, nav_noise=0.0, idx=TR, obj=obj_a),
        "448290": dict(name="TIGER S&P", alpha=0.0, nav_noise=0.0, idx=TR, obj=obj_b),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    out = rec.r9_index_tracking(ctx)
    assert out["status"] == "계산" and out["circular"] is True
    sr = out["series"]
    assert sr["n_days_both_differ"] == 3 and sr["n_days_both"] == hi - 100 - lo + 1
    y = out["etfs"]["379800"]["combined_series"][2015]
    # 지수 수익 = 값 있는 첫날~끝날 비율, ETF 는 알파 0 이라 총수익 ≈ 0 → 차이 ≈ |지수 수익|
    assert y["index_tr"] == pytest.approx(1.0002 ** (hi - lo) - 1)
    assert y["abs_diff_bp"] == pytest.approx(abs(y["etf_tr"] - y["index_tr"]) * 1e4)
    assert y["signed_diff_bp"] < 0 and y["n_index_days"] == hi - lo + 1
    # 교차 값: 379800 은 448290 이 보고한 날(뒤쪽 절반)만
    c = out["etfs"]["379800"]["cross_other_etf_reported"][2015]
    assert (
        out["etfs"]["379800"]["cross_source"] == "448290"
        and c["n_index_days"] == hi - (lo + 100) + 1
    )
    assert out["etfs"]["448290"]["cross_source"] == "379800"


def test_r13_guard_and_run(tmp_path):
    etfs = three()
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    assert (
        rec.r13_forward(ctx, 50, 1)["status"] == "2027-01 이후"
    )  # 자료 끝 2018-12-31 < 2026-12-30
    out = rec.r13_forward(ctx, 50, 1, form_year=2017, min_end=date(2018, 12, 28))
    assert out["status"] == "계산" and out["sample_counts"][0]["year"] == 2017
    assert np.isfinite(out["C"])


def test_fee_aug_fixed_variant_in_judge_all(tmp_path):
    etfs = {}
    for g in range(4):
        for k, a in enumerate((0.06, 0.02, -0.02)):
            etfs[f"{g}{k}A000"[:6]] = dict(
                name=f"KODEX G{g}{k}", alpha=a, idx=f"합성지수{g}", nav_noise=0.0003
            )
    ctx = make_ctx(tmp_path, etfs, {k: 0.1 + 0.2 * (i % 3) for i, k in enumerate(etfs)})
    res = run.judge_all(ctx, years=YEARS, n_perm=49, n_boot=50)
    r = res["records"]
    for key in (
        "fee_aug_fixed",
        "R1_unexplained",
        "R7_general_account_0.846",
        "R8_close_based",
        "R11_delisted_partial_window",
        "R12_baseline",
        "delisted_merged_by_F_band",
        "R9_index_tracking",
        "R13_forward_confirmation",
    ):
        assert key in r, key
    assert np.isfinite(r["fee_aug_fixed"]["C"])
    assert set(res["not_implemented"]) == {"R9_kospi200_tr", "foreign_exposure_nav_lag"}
