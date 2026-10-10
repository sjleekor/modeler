"""MRS 점수 조립 — 결측 규칙(§3.3), 가용성, 첫 날짜, σ_target, KR 크기 성능 (합성 데이터만)."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from modeler.scores.mrs import components as C
from modeler.scores.mrs import config as cfg
from modeler.scores.mrs.score import asset_sigma20, compute_scores
from tests.scores.mrs.test_vintage import (
    daily_rows,
    grid_dates,
    make_grid,
    random_rows,
)

# 시험용 작은 명세: T 둘(가격 추세 둘), V 하나(VIX), L 하나(us_term)
_SMALL = C.us_components(trend_lag_n=3, ma_window=6, rvol_window=3, median_window=5, credit_lag=3)
_SIGNS_BASE = {
    "us_trend_252": ("T", "+"),
    "us_trend_ma200": ("T", "+"),
    "vix_level": ("V", "+"),
    "us_term": ("L", "+"),
}
_SPECS = {k: _SMALL[k] for k in _SIGNS_BASE}
WARM = 5


def _accel_price(n: int) -> np.ndarray:
    """로그 가격이 볼록하게 늘어 추세 x가 계속 새 최고가를 낸다 -> 백분위 100."""
    i = np.arange(n, dtype=float)
    return 100.0 * np.exp(0.0005 * i * i)


def _inputs(grid, *, price=None, vix=None, term=None, starts=None):
    n = grid.n
    starts = starts or {}
    price = _accel_price(n) if price is None else price
    vix = np.full(n, 20.0) if vix is None else vix
    term = np.linspace(2.0, 1.0, n) if term is None else term
    return {
        C.S_US_PRICE: daily_rows(grid, price, start_idx=starts.get("price", 0)),
        C.S_VIX: daily_rows(grid, vix, start_idx=starts.get("vix", 0)),
        C.S_T10Y2Y: daily_rows(grid, term, start_idx=starts.get("term", 0)),
    }


def _run(grid, inputs, signs=None):
    return compute_scores(
        "US", grid, inputs, warmup=WARM, specs=_SPECS, signs=signs or dict(_SIGNS_BASE)
    )


def test_hand_computed_subscores_signs_and_mrs():
    n = 40
    grid = make_grid(n, saturdays=0.0)
    signs = dict(_SIGNS_BASE)
    signs["vix_level"] = ("V", "-")  # 부호 - : 100 - pct
    res = _run(grid, _inputs(grid), signs)
    f = res.frame
    t = n - 2  # 마지막 결정일
    row = f.row(t, named=True)
    # T: 두 추세 모두 계속 최고 -> pct 100
    assert row["pct_us_trend_252"] == 100.0 and row["pct_us_trend_ma200"] == 100.0
    assert row["sub_T"] == 100.0
    # V: 상수 VIX -> x 전부 0 -> 약한 순위 100, 부호 - 이므로 0
    assert row["pct_vix_level"] == 100.0 and row["sub_V"] == 0.0
    # L: us_term 단조 감소 -> x(t)가 이력 최소 -> pct = 100/n_t
    nv = row["nvalid_us_term"]
    assert nv == t + 1
    assert row["pct_us_term"] == pytest.approx(100.0 / nv)
    assert row["sub_L"] == pytest.approx(100.0 / nv)
    assert row["MRS"] == pytest.approx((100.0 + 0.0 + 100.0 / nv) / 3)
    assert (row["n_components_T"], row["n_components_V"], row["n_components_L"]) == (2, 1, 1)
    # 마지막 격자일은 결정 시각이 없어 값이 전부 null
    last = f.row(n - 1, named=True)
    assert last["decision_at"] is None and last["MRS"] is None and last["n_components_T"] == 0


def test_missing_components_dropped_and_counted():
    """성분이 구간 중간에 들어온다: 살아 있는 것만 평균, 개수는 n_components_*."""
    n = 60
    grid = make_grid(n, saturdays=0.0)
    rng = np.random.default_rng(2)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    res = _run(grid, _inputs(grid, price=price))
    f = res.frame
    for i in range(n - 1):
        r = f.row(i, named=True)
        ts = [
            r[f"pct_{c}"] for c in ("us_trend_252", "us_trend_ma200") if r[f"pct_{c}"] is not None
        ]
        assert r["n_components_T"] == len(ts)
        if ts:
            assert r["sub_T"] == pytest.approx(sum(ts) / len(ts))
        else:
            assert r["sub_T"] is None
    # 두 추세의 첫 백분위가 서로 다른 날에 시작한다 (lag 3 먼저, ma 창 6 나중)
    n_T = f["n_components_T"].to_list()
    assert 0 in n_T and 1 in n_T and 2 in n_T


def test_subscore_without_components_is_dropped_and_mrs_uses_the_rest():
    n = 50
    grid = make_grid(n, saturdays=0.0)
    # VIX는 30번째 격자일부터만 있다 -> 그 전에는 V 하위 점수가 없다
    f = _run(grid, _inputs(grid, starts={"vix": 30})).frame
    for i in range(10, 30):
        r = f.row(i, named=True)
        assert r["n_components_V"] == 0 and r["sub_V"] is None
        subs = [v for v in (r["sub_T"], r["sub_V"], r["sub_L"]) if v is not None]
        assert len(subs) == 2 and r["MRS"] == pytest.approx(sum(subs) / 2)
    r = f.row(45, named=True)
    assert r["n_components_V"] == 1
    assert r["MRS"] == pytest.approx((r["sub_T"] + r["sub_V"] + r["sub_L"]) / 3)


def test_all_three_missing_gives_null_mrs():
    n = 30
    grid = make_grid(n, saturdays=0.0)
    f = _run(grid, _inputs(grid)).frame
    early = f.row(2, named=True)  # warm-up도 lag도 안 찬 날
    assert early["MRS"] is None
    assert (early["n_components_T"], early["n_components_V"], early["n_components_L"]) == (0, 0, 0)
    assert early["sub_T"] is None and early["sub_V"] is None and early["sub_L"] is None
    # 하나라도 살아나는 날부터 MRS가 있다
    assert f["MRS"].drop_nulls().len() > 0


def test_only_one_subscore_alive_equals_that_subscore():
    n = 30
    grid = make_grid(n, saturdays=0.0)
    f = _run(grid, _inputs(grid, starts={"vix": 25, "term": 25})).frame
    r = f.row(15, named=True)
    assert r["sub_T"] is not None and r["sub_V"] is None and r["sub_L"] is None
    assert r["MRS"] == r["sub_T"]


def test_availability_ok_always_true_and_input_max_not_after_decision():
    n = 140
    grid = make_grid(n, seed=3)
    rng = np.random.default_rng(21)
    inputs = {
        C.S_US_PRICE: random_rows(rng, grid, p_obs=0.95, lag_max=1, backfill_idx=90),
        C.S_VIX: random_rows(rng, grid, backfill_idx=60, lag_max=2),
        C.S_T10Y2Y: random_rows(rng, grid, lag_max=4, gap=(100, 115)),
    }
    f = _run(grid, inputs).frame
    assert f["availability_ok"].all()
    both = f.filter(
        pl.col("decision_at").is_not_null() & pl.col("input_available_at_max").is_not_null()
    )
    assert both.height > 100
    assert (both["input_available_at_max"] <= both["decision_at"]).all()
    # 입력 행이 아직 하나도 가용하지 않은 첫 날들은 입력 최댓값이 null이다
    assert f["MRS"].drop_nulls().len() > 20


def test_first_value_and_first_pct_dates():
    n = 50
    grid = make_grid(n, saturdays=0.0)
    ds = grid_dates(grid)
    rng = np.random.default_rng(8)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    res = _run(grid, _inputs(grid, price=price, starts={"vix": 10}))
    fd = {r["component"]: r for r in res.first_dates.to_dicts()}
    # trend lag 3: x는 3번째 격자일부터, 유효 개수 WARM=5 -> 7번째
    assert fd["us_trend_252"]["first_value_date"] == ds[3]
    assert fd["us_trend_252"]["first_pct_date"] == ds[3 + WARM - 1]
    # ma 6: x는 5번째부터 (창 6행)
    assert fd["us_trend_ma200"]["first_value_date"] == ds[5]
    # VIX는 10번째 행부터, 중앙값 창 5 -> x는 14번째부터
    assert fd["vix_level"]["first_value_date"] == ds[10 + 4]
    assert fd["vix_level"]["first_pct_date"] == ds[10 + 4 + WARM - 1]
    assert fd["us_term"]["first_value_date"] == ds[0]
    assert fd["us_trend_252"]["sub"] == "T" and fd["vix_level"]["sub"] == "V"
    # 프레임의 첫 non-null과도 일치
    f = res.frame
    first_pct = f.filter(pl.col("pct_us_term").is_not_null())["date"].min()
    assert first_pct == fd["us_term"]["first_pct_date"] == ds[WARM - 1]


def test_missing_input_series_raises_instead_of_silently_dropping():
    grid = make_grid(20, saturdays=0.0)
    inputs = _inputs(grid)
    del inputs[C.S_VIX]
    with pytest.raises(ValueError, match="VIXCLS"):
        _run(grid, inputs)


def test_vix_proxy_is_just_other_input_rows():
    """vix_proxy protocol: VIX 입력 행만 바꾸면 같은 함수가 다른 V를 낸다."""
    n = 40
    grid = make_grid(n, saturdays=0.0)
    base = _inputs(grid)
    alt = dict(base)
    alt[C.S_VIX] = daily_rows(grid, 20 + np.arange(n) * 0.0 + np.sin(np.arange(n)))
    a = _run(grid, base).frame
    b = _run(grid, alt).frame
    assert a["pct_us_trend_252"].to_list() == b["pct_us_trend_252"].to_list()
    assert a["pct_vix_level"].to_list() != b["pct_vix_level"].to_list()


# --------------------------------------------------------------------------- σ_20 / σ_target
def test_asset_sigma20_and_target_expanding_median_no_warmup():
    n = 120
    grid = make_grid(n, saturdays=0.0)
    rng = np.random.default_rng(5)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    out = asset_sigma20(grid, daily_rows(grid, p))
    sig = C.sigma_20(p)
    s = out["sigma20"].to_numpy().astype(float)
    # 마지막 격자일은 결정이 없다 -> null
    assert np.isnan(s[-1])
    np.testing.assert_allclose(s[:-1], sig[:-1], equal_nan=True)
    tgt = out["sigma_target"].to_numpy().astype(float)
    assert np.isnan(tgt[:20]).all()  # σ_20 이력이 1개 생기기 전
    for t in (20, 21, 50, n - 2):
        h = sig[: t + 1]
        h = h[~np.isnan(h)]
        assert tgt[t] == np.median(h)
    assert out["nvalid"][20] == 1  # warm-up 없음: 이력 1개면 낸다 (MI08)


def test_asset_sigma20_uses_t_vintage_history():
    """가격 한 점이 늦게 개정되면 개정 전 결정일의 σ_target은 그대로, 이후는 달라진다."""
    n = 60
    grid = make_grid(n, saturdays=0.0)
    ds = grid_dates(grid)
    rng = np.random.default_rng(6)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    base = daily_rows(grid, p)
    rev = pl.DataFrame(
        {
            "date": [ds[25]],
            "value": [p[25] * 1.2],
            "available_at": [grid.decision_at.to_list()[40]],
        },
        schema=base.schema,
    )
    a = asset_sigma20(grid, base)
    b = asset_sigma20(grid, pl.concat([base, rev]))
    ta, tb = a["sigma_target"].to_numpy().astype(float), b["sigma_target"].to_numpy().astype(float)
    assert np.array_equal(ta[:40], tb[:40], equal_nan=True)  # 개정 전 vintage 그대로
    assert not np.array_equal(ta[40:58], tb[40:58], equal_nan=True)


# --------------------------------------------------------------------------- 성능 (KR 크기)
def test_kr_sized_all_eight_components_with_late_backfill_performance():
    n, t_bf = 8000, 5200
    grid = make_grid(n, saturdays=0.0)
    ds = grid_dates(grid)
    decs = grid.decision_at.to_list()
    rng = np.random.default_rng(0)

    def walk(scale, start):
        return start * np.exp(np.cumsum(rng.normal(0, scale, n)))

    def lagged_rows(vals, lag_days=0):
        from datetime import timedelta

        from tests.scores.mrs.test_vintage import at, frame

        return frame(
            [(ds[i], float(vals[i]), at(ds[i] + timedelta(days=lag_days))) for i in range(n)]
        )

    vix_vals = np.abs(walk(0.03, 18.0))
    from tests.scores.mrs.test_vintage import at, frame

    vix = frame(
        [(ds[i], float(vix_vals[i]), decs[t_bf] if i < t_bf else at(ds[i])) for i in range(n)]
    )
    inputs = {
        C.S_KR_PRICE: lagged_rows(walk(0.012, 1000.0)),
        C.S_TV_KOSPI: lagged_rows(walk(0.05, 5e6)),
        C.S_TV_KOSDAQ: lagged_rows(walk(0.05, 3e6)),
        C.S_FOREIGN_KOSPI: lagged_rows(rng.normal(0, 1e5, n)),
        C.S_FOREIGN_KOSDAQ: lagged_rows(rng.normal(0, 5e4, n)),
        C.S_FX: lagged_rows(walk(0.004, 1100.0)),
        C.S_KR10Y: lagged_rows(3 + np.cumsum(rng.normal(0, 0.01, n)), lag_days=1),
        C.S_KR3Y: lagged_rows(2.5 + np.cumsum(rng.normal(0, 0.01, n)), lag_days=1),
        C.S_VIX: vix,
    }
    t0 = time.perf_counter()
    res = compute_scores("KR", grid, inputs)
    dt = time.perf_counter() - t0
    print(f"\n[perf] KR 8000격자 x 성분 8 (VIX 5200행 백필): {dt:.1f}s")
    assert dt < 30
    f = res.frame
    assert f["availability_ok"].all()
    assert f["MRS"].drop_nulls().len() > 5000
    # VIX는 백필 당일부터 백분위가 난다 (1,260개를 새로 채우지 않는다)
    first_vix = res.first_dates.filter(pl.col("component") == "vix_level")["first_pct_date"][0]
    assert first_vix == ds[t_bf]
    assert res.diagnostics["vix_level"]["n_recomputed_rows"] >= t_bf
    assert set(res.first_dates["component"]) == set(cfg.KR_COMPONENTS)
