"""t 시점 vintage 이력 엔진 (사양 §3.1, 사전등록 §3.1 "가용 이력의 해석"·§3.2).

결정일 ``t``마다 아래를 **그 시점에 가용이던 데이터만으로** 다시 정의한다 (MI03).

* ``D_t`` = 입력 행 중 ``available_at <= decision_at(t)``. 같은 관측일이 여럿이면
  ``available_at``이 가장 늦은 행(최신 vintage)의 값.
* 격자 배치 ``v_t(s)`` (``s <= t``) = ``D_t``에서 관측일 ``d <= s``인 것 중 가장 늦은 ``d``의 값.
  ``(s - d).days > staleness``이면 null (MI04). 격자 밖 관측일(토요일 등)도 이 규칙으로 붙는다.
* 성분 ``x_t(s)`` = 산식을 ``v_t(.)`` 배열에 적용한 값. 창 안에 null이 있으면 null (MI05).
* 이력 = ``{x_t(s) : s <= t, x_t(s) 유효}`` (격자 첫날부터, MI06). 개수 ``n_t``는 ``t`` 포함.
* ``stat="pct"``: ``n_t >= warmup``이고 ``x_t(t)`` 유효일 때
  ``100 * #{x_t(s) <= x_t(t)} / n_t`` (약한 순위, (0, 100], MI07).
* ``stat="median"``: ``n_t >= warmup``이면 이력 중앙값 (``σ_target``용, warm-up 1, MI08).

"각 과거 날을 그날의 시점 값으로 다시 계산"하는 방식은 쓰지 않는다 — 큰 백필이 한꺼번에 가용해지는
날에는 그날 가용이 된 이력 전체가 그날 즉시 분포에 들어간다 (VIX 2010-11-22, BAA10Y 2014-01-27).

두 구현이 있다.

``_reference_vintage``
    매 ``t``마다 ``D_t``를 처음부터 다시 만들고 전체 격자에 산식을 적용하는 기준 구현. 느리다.
``vintage_component_result`` (실제 엔진)
    ``D_t \\ D_{t-1}``의 최소 관측일 ``d_min`` 앞선 격자일의 ``x``는 그대로이므로, ``d_min`` 이후
    격자일만(산식이 거슬러 보는 ``lookback`` 행을 앞에 붙여) 다시 계산한다.
    평소(관측일 = 결정일)에는 ``t`` 한 칸만, 큰 백필이 오는 날에는
    그 날짜부터 ``t``까지 다시 계산한다.
    시험(``test_vintage.py``)이 두 구현이 정확히 같음을 난수 입력으로 확인한다.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np
import polars as pl

from modeler.scores.mrs import config as cfg

Arrays = dict[str, np.ndarray]
Formula = Callable[[Arrays], np.ndarray]

#: 결정 시각이 없는 격자일(마지막 날)의 표식. ``int64`` 마이크로초 epoch.
NO_DECISION = np.iinfo(np.int64).min
_US_PER_DAY = 86_400_000_000


# --------------------------------------------------------------------------- 격자
@dataclass(frozen=True)
class Grid:
    """결정 격자. ``dates``는 오름차순 ``datetime64[D]``, ``decision_us``는 UTC epoch 마이크로초.

    결정 시각이 없는 날(마지막 격자일 — 다음 세션이 없다)은 ``NO_DECISION``이다. 결정 시각은
    비내림차순이어야 하고 ``NO_DECISION``은 끝에만 올 수 있다.
    """

    dates: np.ndarray
    decision_us: np.ndarray

    def __post_init__(self) -> None:
        if self.dates.shape != self.decision_us.shape:
            raise ValueError("격자 날짜와 결정 시각의 길이가 다르다")
        if self.dates.shape[0] > 1 and not (np.diff(self.dates.astype("int64")) > 0).all():
            raise ValueError("격자 날짜는 엄격히 증가해야 한다")
        valid = self.decision_us != NO_DECISION
        k = int(valid.sum())
        if k and not valid[:k].all():
            raise ValueError("결정 시각 없음은 격자 끝에만 올 수 있다")
        if k > 1 and (np.diff(self.decision_us[:k]) < 0).any():
            raise ValueError("결정 시각은 비내림차순이어야 한다")

    @property
    def n(self) -> int:
        return int(self.dates.shape[0])

    @property
    def n_decisions(self) -> int:
        return int((self.decision_us != NO_DECISION).sum())

    @property
    def decision_at(self) -> pl.Series:
        """결정 시각 (Datetime us, UTC). 없는 날은 null."""
        return _epoch_to_series("decision_at", self.decision_us)

    @classmethod
    def from_lists(cls, dates: Sequence[date], decision_ats: Sequence[datetime | None]) -> Grid:
        """``date`` 목록과 tz-aware ``datetime`` 목록(없으면 ``None``)으로 만든다."""
        d = pl.Series("d", list(dates), dtype=pl.Date).to_numpy().astype("datetime64[D]")
        dec = pl.Series("dec", list(decision_ats), dtype=pl.Datetime("us", "UTC"))
        return cls(d, _series_to_epoch(dec))


def _series_to_epoch(s: pl.Series) -> np.ndarray:
    """Datetime(tz-aware 또는 naive=UTC) 시리즈 -> int64 마이크로초, null은 ``NO_DECISION``."""
    e = s.dt.epoch("us").fill_null(NO_DECISION)
    return e.to_numpy().astype(np.int64)


def _epoch_to_series(name: str, e: np.ndarray) -> pl.Series:
    vals = [None if v == NO_DECISION else v for v in e.tolist()]
    s = pl.Series(name, vals, dtype=pl.Int64)
    return s.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")


# --------------------------------------------------------------------------- 입력 행 준비
@dataclass
class _Rows:
    """한 계열의 입력 행: 관측일(일수)·값·가용 시각. 가용 시각 오름차순(동률은 입력 순서)."""

    days: np.ndarray  # int64, epoch 이후 일수
    value: np.ndarray  # float64
    avail: np.ndarray  # int64, epoch us

    @property
    def n(self) -> int:
        return int(self.days.shape[0])


def _prep_rows(rows: pl.DataFrame) -> _Rows:
    """``date, value, available_at`` 행 -> 정렬된 배열. null이 든 행은 버린다 (MI38)."""
    r = rows.select("date", "value", "available_at").drop_nulls()
    if r.height == 0:
        e = np.empty(0, dtype=np.int64)
        return _Rows(e, np.empty(0), e.copy())
    days = r["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    value = r["value"].cast(pl.Float64).to_numpy().astype(float)
    avail = _series_to_epoch(r["available_at"])
    order = np.argsort(avail, kind="stable")  # 동률은 입력 순서 유지 -> 뒤에 온 행이 이긴다
    return _Rows(days[order], value[order], avail[order])


@dataclass
class VintageResult:
    """엔진 결과와 진단.

    ``frame`` 열: ``date, x, stat_value, nvalid, input_available_at_max, n_carried``.

    * ``x`` = ``x_t(t)`` (결정이 없는 날·무효면 null)
    * ``stat_value`` = ``pct``면 백분위 (0, 100], ``median``이면 이력 중앙값 (없으면 null)
    * ``nvalid`` = t 시점 유효 이력 개수 ``n_t`` (t 포함)
    * ``input_available_at_max`` = ``D_t``에 들어간 입력 행의 ``available_at`` 최댓값 (UTC)
    * ``n_carried`` = ``v_t(t)``가 같은 날 관측이 아니라 앞선 관측을 끌어온 입력 계열 수 (MI11)
    """

    frame: pl.DataFrame
    n_recomputed_rows: int = 0  # 산식을 다시 계산한 격자 행 수 합계(``x`` 칸 기준)
    n_batches: int = 0  # D_t가 바뀐 결정일 수
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- 통계
def _stat_from_hist(hist: np.ndarray, x_t: float, stat: str, warmup: int) -> tuple[float, int]:
    """``hist``는 ``x_t(0..t)`` 전체(NaN 포함). 반환: (통계값 또는 NaN, 유효 개수)."""
    valid = hist[~np.isnan(hist)]
    n = int(valid.shape[0])
    if n < max(1, warmup):
        return np.nan, n
    if stat == "pct":
        if np.isnan(x_t):
            return np.nan, n
        return 100.0 * int(np.count_nonzero(valid <= x_t)) / n, n
    return float(np.median(valid)), n


def _check_stat(stat: str) -> None:
    if stat not in ("pct", "median"):
        raise ValueError(f"stat은 'pct' 또는 'median'이다: {stat!r}")


def _assemble(
    grid: Grid,
    x: np.ndarray,
    stat_value: np.ndarray,
    nvalid: np.ndarray,
    avail_max: np.ndarray,
    n_carried: np.ndarray,
) -> pl.DataFrame:
    has_dec = grid.decision_us != NO_DECISION
    xo = np.where(has_dec, x, np.nan)
    return pl.DataFrame(
        {
            "date": pl.Series(grid.dates).cast(pl.Date),
            "x": pl.Series(xo, dtype=pl.Float64).fill_nan(None),
            "stat_value": pl.Series(stat_value, dtype=pl.Float64).fill_nan(None),
            "nvalid": pl.Series(nvalid, dtype=pl.Int64),
            "input_available_at_max": _epoch_to_series("input_available_at_max", avail_max),
            "n_carried": pl.Series(n_carried, dtype=pl.Int64),
        }
    )


def _as_staleness(staleness_days: int | dict[str, int], keys: Sequence[str]) -> dict[str, int]:
    if isinstance(staleness_days, dict):
        return {k: int(staleness_days.get(k, cfg.MACRO_STALENESS_DAYS)) for k in keys}
    return {k: int(staleness_days) for k in keys}


# ------------------------------------------------------------------------- 기준 구현 (brute force)
def _reference_vintage(
    grid: Grid,
    inputs: dict[str, pl.DataFrame],
    formula: Formula,
    *,
    staleness_days: int | dict[str, int] = cfg.MACRO_STALENESS_DAYS,
    stat: Literal["pct", "median"] = "pct",
    warmup: int = cfg.WARMUP_VALID_OBS,
) -> pl.DataFrame:
    """매 ``t``마다 ``D_t``·``v_t``·``x_t``를 처음부터 다시 만드는 느린 기준 구현 (시험 전용)."""
    _check_stat(stat)
    keys = list(inputs)
    stale = _as_staleness(staleness_days, keys)
    raw: dict[str, list[tuple[int, float, int]]] = {}
    for k, df in inputs.items():
        r = _prep_rows(df)
        # 입력 순서가 아니라 정렬 뒤 순서를 쓰되, 동률 규칙은 "뒤 행이 이김"으로 같다.
        raw[k] = list(zip(r.days.tolist(), r.value.tolist(), r.avail.tolist(), strict=True))

    n = grid.n
    gdays = grid.dates.astype("datetime64[D]").astype(np.int64)
    x_out = np.full(n, np.nan)
    st_out = np.full(n, np.nan)
    nv_out = np.zeros(n, dtype=np.int64)
    am_out = np.full(n, NO_DECISION, dtype=np.int64)
    carried = np.zeros(n, dtype=np.int64)

    for t in range(grid.n_decisions):
        dec = int(grid.decision_us[t])
        arrays: Arrays = {}
        amax = NO_DECISION
        ncar = 0
        for k, rows in raw.items():
            best: dict[int, tuple[int, float]] = {}
            for d, v, a in rows:  # 가용 시각 오름차순이므로 뒤 행이 최신 vintage
                if a <= dec:
                    prev = best.get(d)
                    if prev is None or a >= prev[0]:
                        best[d] = (a, v)
            obs = sorted(best)
            for d in obs:
                amax = max(amax, best[d][0])
            arr = np.full(t + 1, np.nan)
            for s in range(t + 1):
                j = bisect_right(obs, int(gdays[s])) - 1
                if j < 0:
                    continue
                d = obs[j]
                if int(gdays[s]) - d > stale[k]:
                    continue
                arr[s] = best[d][1]
                if s == t and d != int(gdays[t]):
                    ncar += 1
            arrays[k] = arr
        xs = np.asarray(formula(arrays), dtype=float)
        x_out[t] = xs[t]
        st_out[t], nv_out[t] = _stat_from_hist(xs[: t + 1], xs[t], stat, warmup)
        am_out[t] = amax
        carried[t] = ncar
    return _assemble(grid, x_out, st_out, nv_out, am_out, carried)


# --------------------------------------------------------------------------- 실제 엔진 (증분)
class _SeriesState:
    """한 입력 계열의 증분 상태: 관측일 축 ``u``, 가용해진 값, 격자 배치 ``v``."""

    def __init__(self, rows: _Rows, gdays: np.ndarray, stale: int) -> None:
        self.rows = rows
        self.stale = stale
        self.gdays = gdays
        self.u = np.unique(rows.days)  # 관측일(오름차순, 중복 없음)
        nu = self.u.shape[0]
        self.pos = np.searchsorted(self.u, rows.days)  # 각 행의 u 위치
        self.val = np.full(nu, np.nan)
        self.present = np.zeros(nu, dtype=bool)
        self.lastp = np.full(nu, -1, dtype=np.int64)  # u 위치 j 이하에서 가용한 가장 늦은 위치
        # 격자일 s -> 관측일 <= s 인 마지막 u 위치 (가용 여부와 무관)
        self.gpos = np.searchsorted(self.u, gdays, side="right") - 1
        self.v = np.full(gdays.shape[0], np.nan)
        self.ptr = 0
        self.avail_max = NO_DECISION
        self.carried_t = False

    def absorb(self, dec: int) -> int | None:
        """``available_at <= dec`` 행을 받는다. 바뀐 가장 이른 관측일의 u 위치(없으면 None)."""
        r = self.rows
        p1 = int(np.searchsorted(r.avail, dec, side="right"))
        if p1 <= self.ptr:
            return None
        sl = slice(self.ptr, p1)
        self.ptr = p1
        pos_b = self.pos[sl]
        # 같은 관측일이 배치 안에 여럿이면 뒤 행(최신 vintage)이 이긴다.
        _, first_rev = np.unique(pos_b[::-1], return_index=True)
        last_idx = pos_b.shape[0] - 1 - first_rev
        pu = pos_b[last_idx]
        self.val[pu] = r.value[sl][last_idx]
        self.present[pu] = True
        self.avail_max = max(self.avail_max, int(r.avail[p1 - 1]))
        p0 = int(pu.min())
        nu = self.u.shape[0]
        idx = np.where(self.present[p0:], np.arange(p0, nu), -1)
        carry = int(self.lastp[p0 - 1]) if p0 > 0 else -1
        self.lastp[p0:] = np.maximum(np.maximum.accumulate(idx), carry)
        return p0

    def place(self, i0: int, t: int) -> None:
        """격자 ``i0..t``의 ``v``를 다시 붙인다."""
        if self.u.shape[0] == 0:  # 입력 행이 하나도 없는 계열
            self.v[i0 : t + 1] = np.nan
            self.carried_t = False
            return
        gp = self.gpos[i0 : t + 1]
        lp = np.where(gp >= 0, self.lastp[np.where(gp >= 0, gp, 0)], -1)
        ok = lp >= 0
        d = self.u[np.where(ok, lp, 0)]
        ok &= (self.gdays[i0 : t + 1] - d) <= self.stale
        self.v[i0 : t + 1] = np.where(ok, self.val[np.where(ok, lp, 0)], np.nan)
        self.carried_t = bool(ok[-1] and d[-1] != self.gdays[t])


def vintage_component_result(
    grid: Grid,
    inputs: dict[str, pl.DataFrame],
    formula: Formula,
    *,
    lookback: int | None = None,
    staleness_days: int | dict[str, int] = cfg.MACRO_STALENESS_DAYS,
    stat: Literal["pct", "median"] = "pct",
    warmup: int = cfg.WARMUP_VALID_OBS,
) -> VintageResult:
    """t 시점 vintage 성분 이력과 확장창 통계를 증분으로 계산한다.

    Parameters
    ----------
    inputs
        계열 이름 -> ``date, value, available_at`` 행. 산식 ``formula``가 같은 이름으로 받는다.
    formula
        ``{계열: 격자 배열}`` -> ``x`` 배열(같은 길이, NaN = null). **인과적**이어야 한다
        (``x[s]``는 ``s`` 이하 칸만 본다).
    lookback
        ``x[s]``가 거슬러 보는 최대 행 수. ``None``이면 매번 처음부터 다시 계산한다(느림).
    """
    _check_stat(stat)
    keys = list(inputs)
    stale = _as_staleness(staleness_days, keys)
    gdays = grid.dates.astype("datetime64[D]").astype(np.int64)
    n = grid.n
    states = {k: _SeriesState(_prep_rows(df), gdays, stale[k]) for k, df in inputs.items()}

    x_cur = np.full(n, np.nan)
    x_out = np.full(n, np.nan)
    st_out = np.full(n, np.nan)
    nv_out = np.zeros(n, dtype=np.int64)
    am_out = np.full(n, NO_DECISION, dtype=np.int64)
    carried = np.zeros(n, dtype=np.int64)
    n_re = 0
    n_batches = 0

    for t in range(grid.n_decisions):
        dec = int(grid.decision_us[t])
        i0 = t
        changed = False
        for stt in states.values():
            p0 = stt.absorb(dec)
            if p0 is not None:
                changed = True
                i0 = min(i0, int(np.searchsorted(gdays, stt.u[p0], side="left")))
        if changed:
            n_batches += 1
        for stt in states.values():
            stt.place(i0, t)
        start = 0 if lookback is None else max(0, i0 - lookback)
        arrays = {k: stt.v[start : t + 1] for k, stt in states.items()}
        xs = np.asarray(formula(arrays), dtype=float)
        x_cur[i0 : t + 1] = xs[i0 - start :]
        n_re += t + 1 - i0
        x_out[t] = x_cur[t]
        st_out[t], nv_out[t] = _stat_from_hist(x_cur[: t + 1], x_cur[t], stat, warmup)
        am_out[t] = max((s.avail_max for s in states.values()), default=NO_DECISION)
        carried[t] = sum(1 for s in states.values() if s.carried_t)

    frame = _assemble(grid, x_out, st_out, nv_out, am_out, carried)
    return VintageResult(frame=frame, n_recomputed_rows=n_re, n_batches=n_batches)


def vintage_component(
    grid_dates: Sequence[date] | Grid,
    decision_ats: Sequence[datetime | None] | None,
    inputs: dict[str, pl.DataFrame],
    formula: Formula,
    *,
    lookback: int | None = None,
    staleness_days: int | dict[str, int] = cfg.MACRO_STALENESS_DAYS,
    stat: Literal["pct", "median"] = "pct",
    warmup: int = cfg.WARMUP_VALID_OBS,
) -> pl.DataFrame:
    """``vintage_component_result(...).frame``. ``grid_dates``에 ``Grid``를 직접 줘도 된다."""
    grid = (
        grid_dates
        if isinstance(grid_dates, Grid)
        else Grid.from_lists(grid_dates, decision_ats or [])
    )
    return vintage_component_result(
        grid,
        inputs,
        formula,
        lookback=lookback,
        staleness_days=staleness_days,
        stat=stat,
        warmup=warmup,
    ).frame
