"""합성 현금 계정 (사양 01 §2.4).

::

    C(u_next) = C(u) * (1 + y_available_at(u)/100 * calendar_days(u, u_next)/365)
    cash_proxy_return(entry, exit) = C(exit)/C(entry) - 1
    cash_basis = "synthetic_short_rate_act365"

* ``u -> u_next``는 자산 달력의 연속한 두 세션이다. 주말·휴일은 실제 달력 일수로 센다.
* 구간 시작 ``u``에 **이용 가능했던** 가장 최근 금리만 쓴다. ALFRED ``realtime_start <= u``인
  행 중 관측일이 가장 늦은 것(같은 관측일이면 최신 vintage). 관측일 개정이 나중에 와도
  더 최근 관측일의 값은 바뀌지 않는다.
* 그 값의 나이(``u - 관측일``, 달력일)가 ``staleness_days``를 넘으면 그 step의 금리는 null,
  null step을 하나라도 포함한 구간의 현금 수익은 null이다. 채워 넣지 않는다.
* 이 계정의 미래 경로는 **라벨에만** 쓴다. 피쳐용 금리는 결정 시점까지만 본다.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

CASH_BASIS = "synthetic_short_rate_act365"
DEFAULT_STALENESS_DAYS = 14

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_NO_RATE = "no_rate_yet"

#: ``CashAccount.interval_return``의 사유 코드.
CODE_OK, CODE_STALE, CODE_NO_RATE = 0, 1, 2


def rate_available_at(rates: pl.DataFrame, as_of: Sequence[date]) -> pl.DataFrame:
    """각 ``as_of`` 날짜 시점의 최신 이용 가능 금리.

    ``rates``: ``date``(관측일), ``realtime_start``, ``value``(%). 반환: ``as_of,
    rate_pct, rate_obs_date`` (없으면 null).
    """
    rows = rates.drop_nulls(["date", "realtime_start", "value"]).sort(["realtime_start", "date"])
    rs = rows["realtime_start"].to_list()
    ds = rows["date"].to_list()
    vs = rows["value"].to_list()
    # realtime_start 순서로 훑으며 관측일 최댓값(frontier)과 그 날짜의 최신 vintage 값을 유지한다.
    latest_by_date: dict[date, float] = {}
    frontier: date | None = None
    starts: list[date] = []
    fr_date: list[date | None] = []
    fr_val: list[float | None] = []
    i = 0
    n = len(rs)
    while i < n:
        cur = rs[i]
        while i < n and rs[i] == cur:
            latest_by_date[ds[i]] = vs[i]
            if frontier is None or ds[i] > frontier:
                frontier = ds[i]
            i += 1
        starts.append(cur)
        fr_date.append(frontier)
        fr_val.append(latest_by_date[frontier] if frontier is not None else None)
    out_rate: list[float | None] = []
    out_obs: list[date | None] = []
    for u in as_of:
        k = bisect_right(starts, u) - 1
        if k < 0:
            out_rate.append(None)
            out_obs.append(None)
        else:
            out_rate.append(fr_val[k])
            out_obs.append(fr_date[k])
    return pl.DataFrame(
        {"as_of": list(as_of), "rate_pct": out_rate, "rate_obs_date": out_obs},
        schema={"as_of": pl.Date, "rate_pct": pl.Float64, "rate_obs_date": pl.Date},
    )


@dataclass(frozen=True)
class CashAccount:
    series_id: str
    staleness_days: int
    cash_basis: str
    #: 세션별: session, rate_pct, rate_obs_date, rate_age_days, step_days, step_status,
    #: growth(null=불가), cash_index(불가 step은 1배로 넘긴 누적곱), bad_stale_cum, bad_norate_cum
    frame: pl.DataFrame

    def interval_return(
        self, entry_idx: np.ndarray, exit_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """세션 인덱스 배열 -> (현금 수익률, 사유 코드). 불가 구간은 NaN."""
        idx = self.frame["cash_index"].to_numpy()
        st = self.frame["bad_stale_cum"].to_numpy()
        nr = self.frame["bad_norate_cum"].to_numpy()
        n = len(idx)
        valid = (entry_idx >= 0) & (exit_idx < n) & (exit_idx >= entry_idx)
        e = np.where(valid, entry_idx, 0)
        x = np.where(valid, exit_idx, 0)
        ret = idx[x] / idx[e] - 1.0
        code = np.where(
            (st[x] - st[e]) > 0, CODE_STALE, np.where((nr[x] - nr[e]) > 0, CODE_NO_RATE, CODE_OK)
        )
        ret = np.where(valid & (code == CODE_OK), ret, np.nan)
        return ret, code


def build_cash_account(
    rates: pl.DataFrame,
    sessions: Sequence[date],
    *,
    series_id: str,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
) -> CashAccount:
    """``rates``(``date, realtime_start, value``)와 자산 세션 목록에서 현금 계정을 만든다."""
    sess = list(sessions)
    avail = rate_available_at(rates, sess)
    f = (
        avail.rename({"as_of": "session"})
        .with_columns(
            (pl.col("session") - pl.col("rate_obs_date")).dt.total_days().alias("rate_age_days"),
            (pl.col("session").shift(-1) - pl.col("session")).dt.total_days().alias("step_days"),
        )
        .with_columns(
            pl.when(pl.col("rate_pct").is_null())
            .then(pl.lit(STATUS_NO_RATE))
            .when(pl.col("rate_age_days") > staleness_days)
            .then(pl.lit(STATUS_STALE))
            .otherwise(pl.lit(STATUS_OK))
            .alias("step_status")
        )
        .with_columns(
            pl.when(pl.col("step_status") == STATUS_OK)
            .then(1.0 + pl.col("rate_pct") / 100.0 * pl.col("step_days") / 365.0)
            .otherwise(None)
            .alias("growth")
        )
    )
    growth = f["growth"].to_list()
    status = f["step_status"].to_list()
    n = len(sess)
    cash = np.ones(n)
    bad_st = np.zeros(n, dtype=np.int64)
    bad_nr = np.zeros(n, dtype=np.int64)
    for k in range(n - 1):
        g = growth[k]
        cash[k + 1] = cash[k] * (g if g is not None else 1.0)
        bad_st[k + 1] = bad_st[k] + (1 if status[k] == STATUS_STALE else 0)
        bad_nr[k + 1] = bad_nr[k] + (1 if status[k] == STATUS_NO_RATE else 0)
    f = f.with_columns(
        pl.Series("cash_index", cash),
        pl.Series("bad_stale_cum", bad_st),
        pl.Series("bad_norate_cum", bad_nr),
    )
    return CashAccount(series_id, staleness_days, CASH_BASIS, f)


def load_us_rates(lake, series_id: str) -> pl.DataFrame | None:
    """레이크 ``macro_series``의 ``series_id`` 행. 없으면 ``None``."""
    df = (
        lake.scan("macro_series")
        .filter(pl.col("series_id") == series_id)
        .select("date", "realtime_start", "value")
        .collect()
    )
    return df if df.height else None
