"""t 시점 vintage 엔진 — 증분 엔진 == brute force, warm-up, 이력 가용 의미 (합성 데이터만)."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from modeler.scores.mrs import components as C
from modeler.scores.mrs import config as cfg
from modeler.scores.mrs.vintage import (
    Grid,
    _reference_vintage,
    vintage_component,
    vintage_component_result,
)

KST = timezone(timedelta(hours=9))
SCHEMA = {"date": pl.Date, "value": pl.Float64, "available_at": pl.Datetime("us", "UTC")}


# --------------------------------------------------------------------------- 합성 도구
def make_grid(
    n: int, seed: int = 0, saturdays: float = 0.3, start: date = date(2000, 1, 3)
) -> Grid:
    """평일 + 확률적으로 토요일 세션. 결정 시각 = 다음 격자일 08:30 KST. 마지막 날은 결정 없음."""
    rng = np.random.default_rng(seed)
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5 or (d.weekday() == 5 and rng.random() < saturdays):
            dates.append(d)
        d += timedelta(days=1)
    decs: list[datetime | None] = [
        datetime(b.year, b.month, b.day, 8, 30, tzinfo=KST).astimezone(UTC) for b in dates[1:]
    ]
    decs.append(None)
    return Grid.from_lists(dates, decs)


def frame(rows: list[tuple[date, float, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "value": [r[1] for r in rows],
            "available_at": [r[2] for r in rows],
        },
        schema=SCHEMA,
    )


def at(d: date, hour: int = 9) -> datetime:
    return datetime(d.year, d.month, d.day, hour, 0, tzinfo=UTC)


def grid_dates(grid: Grid) -> list[date]:
    return grid.dates.astype("datetime64[D]").astype(object).tolist()


def daily_rows(
    grid: Grid, values: np.ndarray, *, start_idx: int = 0, hour: int = 9
) -> pl.DataFrame:
    """격자일마다 한 행, 가용 시각 = 그날 ``hour``시 UTC (그 격자일 결정 시각보다 이르다)."""
    ds = grid_dates(grid)
    return frame([(ds[i], float(values[i]), at(ds[i], hour)) for i in range(start_idx, len(ds))])


def random_rows(
    rng: np.random.Generator,
    grid: Grid,
    *,
    p_obs: float = 0.85,
    lag_max: int = 3,
    p_rev: float = 0.08,
    backfill_idx: int | None = None,
    gap: tuple[int, int] | None = None,
    base: float = 100.0,
) -> pl.DataFrame:
    """달력일마다(토·일 포함) 관측 + 늦은 가용 + 옛 관측 개정 + 큰 백필 + 공백 + 동률 가용 시각."""
    ds = grid_dates(grid)
    decs = grid.decision_at.to_list()
    start, end = ds[0] - timedelta(days=40), ds[-1]
    gap_lo = ds[gap[0]] if gap else None
    gap_hi = ds[gap[1]] if gap else None
    cutoff = ds[backfill_idx] if backfill_idx is not None else None
    rows: list[tuple[date, float, datetime]] = []
    level = 0.0
    d = start
    while d <= end:
        level += rng.normal(0, 0.02)
        if rng.random() < p_obs and not (gap_lo and gap_lo <= d <= gap_hi):
            val = base * float(np.exp(level))
            lag = int(rng.integers(0, lag_max + 1))
            a = datetime(d.year, d.month, d.day, int(rng.integers(0, 24)), tzinfo=UTC) + timedelta(
                days=lag
            )
            if cutoff is not None and d < cutoff:
                a = decs[backfill_idx]  # 옛 이력 전체가 한 시각에 가용 (FRED 백필)
            rows.append((d, val, a))
            if rng.random() < p_rev:  # 옛 관측일의 늦은 개정 vintage
                rv = val * float(1 + rng.normal(0, 0.05))
                ra = a + timedelta(days=int(rng.integers(1, 40)), hours=int(rng.integers(0, 24)))
                rows.append((d, rv, ra))
            if rng.random() < 0.03:  # 가용 시각이 같은 두 vintage (뒤 행이 이긴다)
                rows.append((d, val * 1.01, a))
        d += timedelta(days=1)
    order = rng.permutation(len(rows))
    return frame([rows[i] for i in order])


def eq(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert_frame_equal(a, b, check_exact=True)


# --------------------------------------------------------------------------- 공식 + 입력 묶음
def _f_level(a):
    return C.level_vs_median_neg(a["s"], 6)


def _f_trend(a):
    return C.trend_lag(a["s"], 4) + C.trend_ma(a["s"], 5)


def _f_rvol(a):
    return C.rvol_neg(a["s"], 3, 6)


def _f_liq(a):
    return C.liq_20(a["k"], a["d"], 3, 5)


def _f_foreign(a):
    return C.foreign_20(a["fk"], a["fd"], a["k"], a["d"], 3)


def _f_term(a):
    return C.term_spread(a["l"], a["r"])


CASES = {
    "level": (_f_level, 5, ("s",)),
    "trend": (_f_trend, 4, ("s",)),
    "rvol": (_f_rvol, 8, ("s",)),
    "liq_two_series": (_f_liq, 4, ("k", "d")),
    "foreign_four_series": (_f_foreign, 2, ("fk", "fd", "k", "d")),
    "term_diff": (_f_term, 0, ("l", "r")),
}


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("stat,warmup", [("pct", 8), ("median", 1)])
def test_efficient_equals_reference_random(case, seed, stat, warmup):
    formula, lookback, keys = CASES[case]
    n = 130
    grid = make_grid(n, seed=seed)
    rng = np.random.default_rng(100 + seed)
    inputs = {}
    for i, k in enumerate(keys):
        inputs[k] = random_rows(
            rng,
            grid,
            p_obs=[0.9, 0.7, 0.85, 0.8][i % 4],
            lag_max=[0, 3, 1, 5][i % 4],
            # 첫 계열: 큰 늦은 백필(결정일 60에 옛 이력 전체 가용), 둘째: 10일 넘는 공백
            backfill_idx=60 if i == 0 else None,
            gap=(80, 95) if i == 1 else None,
        )
    ref = _reference_vintage(grid, inputs, formula, stat=stat, warmup=warmup)
    res = vintage_component_result(
        grid, inputs, formula, lookback=lookback, stat=stat, warmup=warmup
    )
    eq(res.frame, ref)
    # lookback을 안 주면(매번 전체 재계산) 같은 값
    eq(vintage_component_result(grid, inputs, formula, stat=stat, warmup=warmup).frame, ref)
    # 시험이 실제로 무언가를 건드렸는가
    assert ref["stat_value"].null_count() < n - 5
    assert ref["x"].null_count() > 0
    assert res.n_batches > 10


def test_random_world_exercises_all_features():
    """위 난수 시험이 개정·백필·공백·토요일·동률을 실제로 담는지 점검한다."""
    grid = make_grid(130, seed=0)
    rng = np.random.default_rng(100)
    df = random_rows(rng, grid, backfill_idx=60, gap=(80, 95))
    ds = grid_dates(grid)
    assert df.group_by("date").len()["len"].max() >= 2  # 여러 vintage
    sat = [d for d in df["date"].to_list() if d.weekday() == 5]
    assert sat and any(d not in set(ds) for d in sat)  # 격자 밖 관측일
    dec60 = grid.decision_at.to_list()[60]
    big = df.filter(pl.col("available_at") == dec60)
    assert big.height > 40  # 한꺼번에 가용
    assert df.filter((pl.col("date") >= ds[80]) & (pl.col("date") <= ds[95])).height == 0


def test_engine_recomputes_far_less_than_reference():
    n = 200
    grid = make_grid(n, seed=5)
    rng = np.random.default_rng(7)
    inputs = {"s": random_rows(rng, grid, p_obs=0.95, lag_max=1, p_rev=0.01)}
    res = vintage_component_result(grid, inputs, _f_level, lookback=5, warmup=5)
    full = (n - 1) * n // 2
    assert res.n_recomputed_rows < full / 4


# --------------------------------------------------------------------------- 손 계산 의미 시험
def _identity(a):
    return a["s"].copy()


def test_gap_over_ten_days_is_null_and_resumes():
    grid = make_grid(40, saturdays=0.0)
    ds = grid_dates(grid)
    rows = [(ds[i], float(i + 1), at(ds[i])) for i in list(range(0, 10)) + list(range(30, 39))]
    df = frame(rows)
    out = vintage_component(grid, None, {"s": df}, _identity, stat="pct", warmup=1)
    x = out["x"].to_list()
    for i in range(40):
        if i < 10 or 30 <= i <= 38:
            want = float(i + 1)
        elif i <= 29 and (ds[i] - ds[9]).days <= 10:
            want = 10.0  # 아직 10일 안: 앞 값을 끌어온다
        else:
            want = None  # 10일 넘게 낡으면 null, 마지막 날은 결정이 없다
        assert x[i] == want, (i, x[i], want)
    # 끌어온 칸 수 진단
    carried = out["n_carried"].to_list()
    assert carried[11] == 1 and carried[5] == 0


def test_saturday_observation_not_on_grid_lands_on_next_grid_day():
    grid = make_grid(10, saturdays=0.0, start=date(2000, 1, 3))  # 월~금 + 월~금
    ds = grid_dates(grid)
    sat = ds[4] + timedelta(days=1)  # 첫 금요일 다음 토요일
    df = frame([(sat, 7.0, at(sat, 12))])
    out = vintage_component(grid, None, {"s": df}, _identity, stat="pct", warmup=1)
    x = out["x"].to_list()
    assert x[4] is None  # 금요일 격자: 토요일 관측은 아직 없다 (d <= s)
    assert x[5] == 7.0  # 다음 월요일: 가용하고 나이 2일
    assert out["input_available_at_max"][4] == at(sat, 12)  # 금요일 결정 시각에 이미 가용
    assert out["input_available_at_max"][3] is None


def test_revision_of_old_obs_changes_history_only_from_when_available():
    """개정은 가용해진 결정일부터 이력에 반영된다 (옛 날을 그날 시점 값으로 다시 안 계산)."""
    grid = make_grid(12, saturdays=0.0)
    ds = grid_dates(grid)
    base = [(ds[i], float(i), at(ds[i])) for i in range(10)]
    rev = [(ds[2], 100.0, at(ds[8]))]  # 2번 관측을 8번째 결정 직전에 개정
    out = vintage_component(grid, None, {"s": frame(base + rev)}, _identity, stat="pct", warmup=1)
    p = out["stat_value"].to_list()
    assert p[5] == 100.0  # 개정 전: 5가 최대
    # t=8: 이력 {0,1,100,3,4,5,6,7,8} -> x=8의 약한 순위 8/9
    assert p[8] == pytest.approx(100.0 * 8 / 9)
    assert p[7] == 100.0
    assert out["nvalid"][8] == 9


def test_pct_is_weak_rank_in_zero_hundred():
    grid = make_grid(8, saturdays=0.0)
    ds = grid_dates(grid)
    vals = [3.0, 3.0, 1.0, 2.0, 3.0, 5.0, 0.5]
    out = vintage_component(
        grid,
        None,
        {"s": frame([(ds[i], v, at(ds[i])) for i, v in enumerate(vals)])},
        _identity,
        stat="pct",
        warmup=1,
    )
    p = out["stat_value"].to_list()
    assert p[0] == 100.0 and p[1] == 100.0
    assert p[2] == pytest.approx(100 / 3)  # 가장 작다 -> 1/3
    assert p[3] == pytest.approx(200 / 4)  # {3,3,1,2}에서 <=2 가 2개
    assert p[4] == 100.0  # 3 이하 5개 중 5개 (동률 포함)
    assert p[6] == pytest.approx(100 / 7)
    assert all(0 < v <= 100 for v in p[:7])


def test_median_stat_includes_t_and_has_no_warmup():
    grid = make_grid(6, saturdays=0.0)
    ds = grid_dates(grid)
    vals = [4.0, 1.0, 3.0, 10.0, 2.0]
    out = vintage_component(
        grid,
        None,
        {"s": frame([(ds[i], v, at(ds[i])) for i, v in enumerate(vals)])},
        _identity,
        stat="median",
        warmup=1,
    )
    assert out["stat_value"].to_list()[:5] == [4.0, 2.5, 3.0, 3.5, 3.0]


# --------------------------------------------------------------------------- warm-up
def test_warmup_exact_small_override():
    n, lag, warm = 80, 5, 30
    grid = make_grid(n, saturdays=0.0)
    rng = np.random.default_rng(3)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    out = vintage_component(
        grid,
        None,
        {"s": daily_rows(grid, p)},
        lambda a: C.trend_lag(a["s"], lag),
        lookback=lag,
        stat="pct",
        warmup=warm,
    )
    nv = out["nvalid"].to_numpy()
    pct = out["stat_value"].to_numpy()
    first = (
        int(np.flatnonzero(~np.isnan(pct.astype(float)))[0])
        if out["stat_value"].null_count() < n
        else -1
    )
    assert first == lag + warm - 1  # x는 lag번째부터 유효, 유효 개수가 warm에 닿는 날
    assert nv[first] == warm and nv[first - 1] == warm - 1


def test_warmup_exact_real_1260():
    n = 1530
    grid = make_grid(n, saturdays=0.0)
    rng = np.random.default_rng(4)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    sp = C.kr_components()["kr_trend_252"]
    out = vintage_component(
        grid,
        None,
        {C.S_KR_PRICE: daily_rows(grid, p)},
        sp.fn,
        lookback=sp.lookback,
        stat="pct",
        warmup=cfg.WARMUP_VALID_OBS,
    )
    pct = out["stat_value"].to_list()
    first = next(i for i, v in enumerate(pct) if v is not None)
    assert first == 252 + cfg.WARMUP_VALID_OBS - 1  # 1511
    assert out["nvalid"][first] == cfg.WARMUP_VALID_OBS
    assert pct[first - 1] is None and out["nvalid"][first - 1] == cfg.WARMUP_VALID_OBS - 1
    # 값 자체: 약한 순위
    xs = np.log(p[252:] / p[:-252])
    h = xs[: first - 252 + 1]
    assert pct[first] == pytest.approx(100.0 * np.count_nonzero(h <= h[-1]) / h.size)


# ------------------------------------------------------------------------- "t에 가용이던 이력 전체"
def test_late_backfill_gives_percentile_on_the_backfill_day_not_1260_days_later():
    """VIX 2010-11-22형: 옛 이력 전체가 한 날 가용해지면 그날 백분위가 바로 나온다."""
    n, t_bf = 1700, 1600
    grid = make_grid(n, saturdays=0.0)
    ds = grid_dates(grid)
    decs = grid.decision_at.to_list()
    rng = np.random.default_rng(11)
    vix = 15 + np.cumsum(rng.normal(0, 0.3, n))
    vix = np.abs(vix) + 5
    rows = []
    for i in range(n):
        if i < t_bf:
            rows.append((ds[i], float(vix[i]), decs[t_bf]))  # 한꺼번에 가용
        else:
            rows.append((ds[i], float(vix[i]), at(ds[i])))
    sp = C.kr_components()["vix_level"]
    res = vintage_component_result(
        grid,
        {C.S_VIX: frame(rows)},
        sp.fn,
        lookback=sp.lookback,
        stat="pct",
        warmup=cfg.WARMUP_VALID_OBS,
    )
    out = res.frame
    assert out["stat_value"][:t_bf].null_count() == t_bf  # 가용 전엔 아무것도 없다
    assert out["stat_value"][t_bf] is not None  # 백필 당일에 백분위가 난다
    # 유효 이력: 격자 251번째부터 t_bf까지
    assert out["nvalid"][t_bf] == t_bf + 1 - (cfg.MEDIAN_WINDOW - 1)
    assert out["nvalid"][t_bf] >= cfg.WARMUP_VALID_OBS
    # 값 검증: 전 이력이 이미 가용이므로 x를 직접 계산해 순위를 센다 (미래 vintage 개정 없음)
    x = sp.fn({C.S_VIX: vix[: t_bf + 1]})
    h = x[~np.isnan(x)]
    assert out["stat_value"][t_bf] == pytest.approx(100.0 * np.count_nonzero(h <= x[t_bf]) / h.size)
    # 백필 당일은 앞부분 전체를 다시 계산한 날이다 (엔진 진단)
    assert res.n_recomputed_rows >= t_bf


def test_availability_never_after_decision_time():
    grid = make_grid(120, seed=2)
    rng = np.random.default_rng(9)
    inputs = {k: random_rows(rng, grid, lag_max=4) for k in ("k", "d")}
    out = vintage_component(grid, None, inputs, _f_liq, lookback=4, stat="pct", warmup=5)
    dec = grid.decision_at
    mx = out["input_available_at_max"]
    ok = (mx.is_null() | dec.is_null() | (mx <= dec)).all()
    assert ok


def test_decision_times_must_be_monotone_and_trailing_none():
    d = [date(2000, 1, 3), date(2000, 1, 4), date(2000, 1, 5)]
    a, b = at(d[0]), at(d[1])
    with pytest.raises(ValueError):
        Grid.from_lists(d, [b, a, None])
    with pytest.raises(ValueError):
        Grid.from_lists(d, [None, a, b])


def test_empty_series_gives_all_null():
    grid = make_grid(20, saturdays=0.0)
    empty = frame([])
    out = vintage_component(grid, None, {"s": empty}, _identity, stat="pct", warmup=1)
    assert out["x"].null_count() == 20 and out["stat_value"].null_count() == 20
    eq(out, _reference_vintage(grid, {"s": empty}, _identity, warmup=1))


def test_size_performance_kr_sized_grid_one_component_with_backfill():
    """KR 크기(8,000 격자) 한 성분 + 큰 늦은 백필이 몇 초 안에 끝난다 (8성분은 test_score)."""
    n, t_bf = 8000, 5000
    grid = make_grid(n, saturdays=0.0)
    ds = grid_dates(grid)
    decs = grid.decision_at.to_list()
    rng = np.random.default_rng(1)
    v = 20 + np.cumsum(rng.normal(0, 0.2, n))
    v = np.abs(v) + 1
    rows = [(ds[i], float(v[i]), decs[t_bf] if i < t_bf else at(ds[i])) for i in range(n)]
    sp = C.kr_components()["vix_level"]
    t0 = time.perf_counter()
    res = vintage_component_result(
        grid, {C.S_VIX: frame(rows)}, sp.fn, lookback=sp.lookback, warmup=cfg.WARMUP_VALID_OBS
    )
    dt = time.perf_counter() - t0
    print(f"\n[perf] vix_level 8000격자 + 5000행 백필: {dt:.2f}s, 재계산 {res.n_recomputed_rows}행")
    assert dt < 30
    assert res.frame["stat_value"][t_bf] is not None
