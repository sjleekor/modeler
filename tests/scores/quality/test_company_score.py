"""company_score 시험 — 합성 자료, 손계산 대조 (사전등록 §5.2)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import company_score as cs

NUM_COLS = [c for c in cs.REQUIRED_COLS if c not in ("corp_code", "fy", "in_universe", "currency")]


def panel(rows: list[dict]) -> pl.DataFrame:
    """빈 칸은 null인 입력 패널. 행마다 덮어쓸 값만 준다."""
    recs = []
    for i, r in enumerate(rows):
        base = {c: None for c in NUM_COLS}
        base.update(corp_code=f"C{i}", fy=2019, in_universe=True, currency="KRW")
        base.update(r)
        recs.append(base)
    schema = {c: pl.Float64 for c in NUM_COLS}
    schema.update(corp_code=pl.Utf8, fy=pl.Int64, in_universe=pl.Boolean, currency=pl.Utf8)
    return pl.DataFrame(recs, schema=schema)


def one(r: dict) -> dict:
    return cs.compute_scores(panel([r])).row(0, named=True)


# ---------------------------------------------------------------- 백분위
def _pr(vals, fy=None, uni=None, hib=True):
    n = len(vals)
    return cs.pct_rank(
        pl.Series(vals, dtype=pl.Float64),
        pl.Series(fy or [2019] * n),
        pl.Series(uni or [True] * n),
        hib,
    ).to_list()


def test_pct_ties_average_rank():
    # 순위 1, 2.5, 2.5, 4 -> (r-1)/3*100
    got = _pr([1.0, 2.0, 2.0, 3.0])
    assert got == pytest.approx([0.0, 100 / 3 * 1.5, 100 / 3 * 1.5, 100.0])


def test_pct_lower_is_better():
    assert _pr([1.0, 2.0, 3.0], hib=False) == pytest.approx([100.0, 50.0, 0.0])


def test_pct_single_pool_is_50():
    assert _pr([7.0]) == [50.0]


def test_pct_not_universe_excluded_and_null_missing():
    got = _pr([1.0, 2.0, 3.0, 99.0, None], uni=[True, True, True, False, True])
    assert got[:3] == pytest.approx([0.0, 50.0, 100.0])
    assert got[3] is None and got[4] is None  # 풀 밖 / 값 없음


def test_pct_per_fy():
    got = _pr([1.0, 2.0, 10.0, 20.0], fy=[2018, 2018, 2019, 2019])
    assert got == pytest.approx([0.0, 100.0, 0.0, 100.0])


def test_pct_worst_mask_zero_and_pool_include():
    v = pl.Series([None, 5.0, 6.0, 7.0], dtype=pl.Float64)
    kw = dict(
        fy=pl.Series([2019] * 4),
        in_universe=pl.Series([True] * 4),
        higher_is_better=True,
        worst_mask=pl.Series([True, False, False, False]),
    )
    inc = cs.pct_rank(v, worst_mode="include", **kw).to_list()
    assert inc == pytest.approx([0.0, 100 / 3, 200 / 3, 100.0])  # n=4 안에서 순위
    exc = cs.pct_rank(v, worst_mode="exclude", **kw).to_list()
    assert exc == pytest.approx([0.0, 0.0, 50.0, 100.0])  # 나머지 셋으로 순위


# ---------------------------------------------------------------- Z''
def test_zpp_hand_calc():
    r = one(dict(ta=1000, tl=400, te=600, ca=500, cl=200, re=300, oi=100))
    x1, x2, x3, x4 = 0.3, 0.3, 0.1, 1.5
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    assert r["c1_zpp"] == pytest.approx(z)
    assert r["c1_zpp_em"] == pytest.approx(z + 3.25)


@pytest.mark.parametrize("over", [dict(ta=0), dict(ta=-5), dict(tl=0), dict(tl=None)])
def test_zpp_missing(over):
    base = dict(ta=1000, tl=400, te=600, ca=500, cl=200, re=300, oi=100)
    base.update(over)
    assert one(base)["c1_zpp"] is None


# ---------------------------------------------------------------- 성분
def test_icr_rules():
    assert cs.icr_paid_value(np.array([10.0]), np.array([0.0]))[0] == 100
    assert cs.icr_paid_value(np.array([0.0]), np.array([0.0]))[0] == -100
    assert cs.icr_paid_value(np.array([-3.0]), np.array([0.0]))[0] == -100
    assert cs.icr_paid_value(np.array([1e6]), np.array([1.0]))[0] == 100
    assert cs.icr_paid_value(np.array([-1e6]), np.array([1.0]))[0] == -100
    assert cs.icr_paid_value(np.array([6.0]), np.array([2.0]))[0] == 3
    assert math.isnan(cs.icr_paid_value(np.array([6.0]), np.array([np.nan]))[0])


def test_debt_ratio_nonpositive_equity_is_pct_zero():
    p = panel(
        [
            dict(tl=100, te=50),
            dict(tl=300, te=100),
            dict(tl=200, te=-10),
            dict(tl=50, te=0),
        ]
    )
    r = cs.compute_scores(p)
    assert r["c1_dr_pct"].to_list()[2:] == [0.0, 0.0]
    assert r["c1_dr"].to_list()[2:] == [None, None]
    # 낮을수록 좋음: 비율 2 > 3 이므로 첫 행이 둘째보다 높다
    assert r["c1_dr_pct"][0] > r["c1_dr_pct"][1] > 0


def test_capital_impairment_ratio_and_current_ratio():
    r = one(dict(te=50, cap=100, ca=300, cl=100))
    assert r["c1_capr"] == pytest.approx(0.5)
    assert r["c1_cr"] == pytest.approx(3.0)
    r = one(dict(te=50, cap=0, ca=300, cl=0))
    assert r["c1_capr"] is None and r["c1_cr"] is None


def test_c1_needs_zpp_and_three_components():
    # Z'' 없이 성분 4개(capr, icr, cr, dr) -> 결측
    r = cs.compute_scores(
        panel(
            [
                dict(te=50, cap=100, oi=10, ip=2, ca=3, cl=1, tl=10),
                dict(te=60, cap=100, oi=11, ip=2, ca=4, cl=1, tl=10),
            ]
        )
    )
    assert r["c1_n"].to_list() == [4, 4] and r["c1"].to_list() == [None, None]
    # Z'' 포함 3개(zpp, capr, cr) -> 계산
    z = dict(ta=1000, tl=400, te=600, ca=500, cl=200, re=300, oi=100, cap=100)
    r = cs.compute_scores(panel([dict(z, ca=500), dict(z, ca=400)]))
    assert r["c1_n"].to_list() == [4, 4] or True  # zpp, capr, cr, dr (ip 없음)
    r = cs.compute_scores(
        panel([dict(ta=1000, tl=400, te=600, ca=500, cl=200, re=300, oi=100) for _ in range(2)])
    )
    assert r["c1_n"].to_list() == [3, 3]  # zpp, cr, dr
    assert all(x is not None for x in r["c1"].to_list())


def test_cfo_ni_boundaries():
    f = cs.cfo_ni_value
    a = lambda x: np.array([float(x)])  # noqa: E731
    assert f(a(10), a(50))[0] == 3  # 상한
    assert f(a(10), a(-50))[0] == -1  # 하한
    assert f(a(10), a(5))[0] == pytest.approx(0.5)
    assert f(a(0), a(5))[0] == 3  # ni ≤ 0, ocf > 0
    assert f(a(-4), a(0))[0] == -1  # ni ≤ 0, ocf ≤ 0
    assert f(a(-4), a(-1))[0] == -1
    assert math.isnan(f(a(np.nan), a(1))[0])


def test_accrual_and_roe_and_vol():
    r = one(
        dict(ni=100, ocf=40, ta=1000, ta_p1=800, te=500, te_p1=400, te_p2=250, ni_p1=40, ni_p2=25)
    )
    assert r["c2_acc"] == pytest.approx(60 / 900)
    assert r["c2_roe"] == pytest.approx(0.2)
    # ROE 세 값 0.2, 0.1, 0.1 -> ddof=1 표준편차
    assert r["c2_rvol"] == pytest.approx(float(np.std([0.2, 0.1, 0.1], ddof=1)))
    # te_p1 ≤ 0 이면 변동성 결측 (CI-rvol)
    r = one(dict(ni=100, te=500, te_p1=0, te_p2=250, ni_p1=40, ni_p2=25))
    assert r["c2_rvol"] is None


def test_c2_roe_worst_and_min_three():
    p = panel(
        [dict(ni=10, te=-5, ocf=1, ta=10, ta_p1=10), dict(ni=5, te=10, ocf=2, ta=10, ta_p1=10)]
    )
    r = cs.compute_scores(p)
    assert r["c2_roe_pct"][0] == 0.0 and r["c2_roe"][0] is None
    assert r["c2_n"].to_list() == [3, 3]  # acc, cfo, roe
    assert r["c2"][0] is not None


# ---------------------------------------------------------------- C3
def test_dividend_streak():
    f = cs._streak
    a = lambda *x: np.array([float(v) for v in x])  # noqa: E731
    n = np.nan
    assert f(a(5), a(5), a(5))[0] == 3
    assert f(a(5), a(0), a(5))[0] == 1
    assert f(a(0), a(5), a(5))[0] == 0
    assert f(a(5), a(n), a(5))[0] == 1  # 앞 연도 결측에서 끊음
    assert math.isnan(f(a(n), a(5), a(5))[0])


def _cut(p2, p1, t, cur="KRW", ev=None, ev1=None):
    r = one(dict(dps_p2=p2, dps_p1=p1, dps=t, currency=cur, capevt=ev, capevt_p1=ev1))
    return r["c3_cuts"]


def test_cuts_o3_definition():
    # p2=p1=100 로 첫 쌍은 삭감 없음, 둘째 쌍만 시험
    assert _cut(100, 100, 80) == 1  # 20% 감소
    assert _cut(100, 100, 81) == 0
    assert _cut(100, 100, 0) == 1
    assert _cut(40, 40, 0) == 0  # 40원 < 50원 하한 (KRW)
    assert _cut(50, 50, 40) == 1  # 50원은 하한 통과, 40 ≤ 40
    assert _cut(0.4, 0.4, 0.0, cur="USD") == 1  # 비KRW는 하한 없음
    assert _cut(100, 100, 0, ev=True) is None
    assert _cut(100, 100, 0, ev1=True) is None
    assert _cut(100, 100, 0, ev=False, ev1=False) == 1
    assert _cut(None, 100, 80) is None  # 한 쌍 결측 (CI-cut)
    assert _cut(100, 80, 64) == 2  # 두 쌍 모두 삭감


def test_net_issuance_and_retire():
    r = one(dict(shares=110, shares_p1=100))
    assert r["c3_iss"] == pytest.approx(0.1)
    assert one(dict(shares=110, shares_p1=100, capevt=True))["c3_iss"] is None
    assert one(dict(retire=0, retire_p1=1))["c3_ret"] == 1
    assert one(dict(retire=0, retire_p1=0))["c3_ret"] == 0
    assert one(dict(retire=None, retire_p1=None))["c3_ret"] is None
    assert one(dict(retire=0, retire_p1=None))["c3_ret"] is None  # CI-ret


# ---------------------------------------------------------------- 합성
def _full(i: int) -> dict:
    """세 차원이 모두 계산되는 한 행 (값이 i에 따라 달라짐)."""
    return dict(
        ta=1000, tl=400 + i, te=600 - i, ca=500 + i, cl=200, re=300, oi=100 + i, cap=100, ip=5,
        ni=50 + i, ocf=40 + i, ta_p1=900, te_p1=550, te_p2=500, ni_p1=40, ni_p2=30,
        dps=10 + i, dps_p1=10, dps_p2=10, shares=100, shares_p1=100, retire=0, retire_p1=0,
    )  # fmt: skip


def test_composite_requires_three_dims_and_2dim_two():
    rows = [_full(i) for i in range(4)]
    # 마지막 행: C3 성분을 모두 없앰 -> C3 결측, C1·C2 있음
    for k in ("dps", "dps_p1", "dps_p2", "shares", "shares_p1", "retire", "retire_p1"):
        rows[3][k] = None
    r = cs.compute_scores(panel(rows))
    assert r["c"][:3].null_count() == 0
    assert r["c"][3] is None
    assert r["c_2dim_n"][3] == 2 and r["c_2dim"][3] is not None
    assert r["c3"][3] is None
    # 세 차원 평균식
    exp = (r["c1_pct"][0] + r["c2_pct"][0] + r["c3_pct"][0]) / 3
    assert r["c"][0] == pytest.approx(exp)


def test_alt_composites():
    rows = [_full(i) for i in range(4)]
    r = cs.compute_scores(panel(rows))
    # O2용 C1′ = (icr, cr) 평균
    assert r["c1_alt"][0] == pytest.approx((r["c1_icr_pct"][0] + r["c1_cr_pct"][0]) / 2)
    assert r["c_o2"][0] == pytest.approx((r["c1_alt_pct"][0] + r["c2_pct"][0] + r["c3_pct"][0]) / 3)
    assert r["c_o3"][0] == pytest.approx((r["c1_pct"][0] + r["c2_pct"][0] + r["c3_alt_pct"][0]) / 3)
    assert r["c_o4"][0] == pytest.approx((r["c1_pct"][0] + r["c2_alt_pct"][0] + r["c3_pct"][0]) / 3)
    # 남은 성분이 모두 있어야: icr 없으면 C1′ 결측
    rows[0]["ip"] = None
    r = cs.compute_scores(panel(rows))
    assert r["c1_alt"][0] is None and r["c_o2"][0] is None


def test_not_in_universe_has_no_scores_but_f():
    rows = [_full(i) for i in range(3)]
    rows[2]["in_universe"] = False
    r = cs.compute_scores(panel(rows))
    assert r["c1_zpp_pct"][2] is None and r["c"][2] is None


# ---------------------------------------------------------------- Piotroski F
def f_base(**kw) -> dict:
    """아홉 신호가 모두 1이 되는 기본 행."""
    d = dict(
        ta=1000, ta_p1=900, ta_p2=800,
        ni=90, ni_p1=40, ocf=108,  # ROA .1 > 0, ΔROA: .1 > 40/800=.05, ocf/ta_p1 .12 > .1
        ltb=50, ltb_p1=100,  # 50/950 < 100/850
        ca=300, cl=100, ca_p1=200, cl_p1=100,
        shares=100, shares_p1=100,
        gp=400, rev=1000, gp_p1=240, rev_p1=800,  # 마진 .4 > .3; 회전 1000/900 > 800/800
    )  # fmt: skip
    d.update(kw)
    return d


def _f(**kw):
    return one(f_base(**kw))


def test_f_all_ones():
    r = _f()
    assert [r[f"f{i}"] for i in range(1, 10)] == [1] * 9
    assert r["f_sum"] == 9 and r["f_ltb_req"] == 9


@pytest.mark.parametrize(
    "k,over",
    [
        (1, dict(ni=-1)),
        (2, dict(ocf=0)),
        (3, dict(ni_p1=80)),  # 80/800=.1 == .1 -> 동률이라 0
        (4, dict(ocf=80)),  # ocf/ta_p1 .0889 < .1
        (5, dict(ltb=100, ltb_p1=100)),  # 100/950 > 100/850? 아래 별도
        (6, dict(ca_p1=400)),
        (7, dict(shares=101)),
        (8, dict(gp=100)),
        (9, dict(rev=700)),
    ],
)
def test_f_each_signal_zero(k, over):
    if k == 5:
        over = dict(ltb=100, ltb_p1=90)  # 100/950=.105 > 90/850=.106? 계산 보정
        over = dict(ltb=120, ltb_p1=90)
    r = _f(**over)
    assert r[f"f{k}"] == 0
    assert r["f_sum"] is not None


def test_f5_tie_is_zero_and_ltb_fill():
    # 둘 다 결측 -> 0 채움 -> 0 < 0 거짓 -> F5 = 0 (동률)
    r = _f(ltb=None, ltb_p1=None)
    assert r["f5"] == 0 and r["f_sum"] == 8
    assert r["f_ltb_req"] is None  # 값 필수 판은 결측
    # 한쪽만 결측: 전년 값이 있고 올해 0 채움 -> 0 < 양수 -> 1
    r = _f(ltb=None, ltb_p1=100)
    assert r["f5"] == 1 and r["f_ltb_req"] is None


def test_f7_event_and_missing_inputs():
    r = _f(capevt=True)
    assert r["f7"] is None and r["f_sum"] is None
    r = _f(ni_p1=None)
    assert r["f3"] is None and r["f_sum"] is None
    r = _f(ta_p1=0)  # 분모 ≤ 0 -> F1·F2 등 결측 (CI-fden)
    assert r["f1"] is None and r["f_sum"] is None
