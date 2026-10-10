"""MRS 성분 산식 (사양 §3.2, 사전등록 §3.1 표).

격자에 붙은 배열(``numpy``, NaN = null)을 받아 같은 길이의 ``x`` 배열을 돌려주는 순수 함수다.

* lag·rolling은 **격자 행 수**로 센다 (§3.2). 창 안에 NaN이 하나라도 있으면 그 칸은 NaN이다
  (strict, MI05). 앞부분 ``lag``·``window``가 모자란 칸도 NaN이다.
* 창 계산은 ``sliding_window_view`` + 창 단위 축소라, 같은 창이면 배열을 어디서 잘라 계산해도
  비트까지 같은 값이 나온다. 증분 엔진(``vintage.py``)이 앞부분을 잘라 다시 계산해도 기준 구현과
  정확히 같아야 하기 때문이다.
* 각 성분의 ``lookback``은 ``x(s)``가 ``s - lookback`` 행까지만 거슬러 본다는 선언이다.
  엔진은 이 값만큼만 앞을 붙여 다시 계산한다. ``test_components.py``가 선언이 맞는지 확인한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from modeler.scores.mrs import config as cfg

# --------------------------------------------------------------------------- 입력 계열 이름
#: 입력 행 사전(``dict[str, DataFrame]``)의 키. C(inputs.py)가 같은 이름으로 넘긴다.
S_KR_PRICE = "market_kospi_ecos"  # KR 격자·가격 (MI01)
S_US_PRICE = "tr_index"  # US SPY 총수익 지수 (MI09)
S_TV_KOSPI = "trdval_kospi_ecos"
S_TV_KOSDAQ = "trdval_kosdaq_ecos"
S_FOREIGN_KOSPI = "foreign_net_kospi_ecos"
S_FOREIGN_KOSDAQ = "foreign_net_kosdaq_ecos"
S_FX = "fx_usdkrw_ecos"
S_KR10Y = "rate_kr_gov10y"
S_KR3Y = "rate_kr_gov3y"
S_VIX = "VIXCLS"
S_BAA = "BAA10Y"
S_T10Y2Y = "T10Y2Y"  # 문면 원천 칸이 macro_series.T10Y2Y (MI12)

Arrays = dict[str, np.ndarray]


# --------------------------------------------------------------------------- 창 계산 도구
def _rolling(a: np.ndarray, window: int, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """끝이 ``s``인 최근 ``window``행에 ``func``을 적용한다. NaN이 하나라도 있으면 NaN (MI05)."""
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape[0], np.nan)
    if window < 1 or a.shape[0] < window:
        return out
    win = sliding_window_view(a, window)
    bad = np.isnan(win).any(axis=-1)
    res = np.asarray(func(win), dtype=float)
    res = np.where(bad, np.nan, res)
    out[window - 1 :] = res
    return out


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(a, window, lambda w: np.mean(w, axis=-1))


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(a, window, lambda w: np.sum(w, axis=-1))


def rolling_median(a: np.ndarray, window: int) -> np.ndarray:
    return _rolling(a, window, lambda w: np.median(w, axis=-1))


def rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    """표본표준편차(ddof=1)."""
    return _rolling(a, window, lambda w: np.std(w, axis=-1, ddof=1))


def lagged(a: np.ndarray, lag: int) -> np.ndarray:
    """``a[s - lag]`` (앞 ``lag``칸은 NaN)."""
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape[0], np.nan)
    if lag < a.shape[0]:
        out[lag:] = a[: a.shape[0] - lag]
    return out


def _safe_log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``ln(num/den)``. 분자·분모가 양수가 아니면 NaN (정의되지 않음 — MI37)."""
    ok = (num > 0) & (den > 0)
    out = np.full(np.shape(num), np.nan)
    with np.errstate(all="ignore"):
        out = np.where(ok, np.log(np.where(ok, num, 1.0) / np.where(ok, den, 1.0)), np.nan)
    return out


# --------------------------------------------------------------------------- 수익·변동성
def log_returns(p: np.ndarray) -> np.ndarray:
    """``ln(P(s)/P(s-1))`` — MS ``rvol_20``과 같은 로그 수익 (MI10)."""
    p = np.asarray(p, dtype=float)
    return _safe_log_ratio(p, lagged(p, 1))


def sigma_20(p: np.ndarray, window: int = cfg.RVOL_WINDOW) -> np.ndarray:
    """최근 ``window``개 로그 수익의 표본표준편차(ddof=1). 연율화하지 않는다 (MI10)."""
    return rolling_std(log_returns(p), window)


# --------------------------------------------------------------------------- 성분 산식
def trend_lag(p: np.ndarray, lag: int = cfg.TREND_LAG) -> np.ndarray:
    """``ln(P_t / P_{t-lag})`` — ``kr_trend_252``·``us_trend_252``."""
    p = np.asarray(p, dtype=float)
    return _safe_log_ratio(p, lagged(p, lag))


def trend_ma(p: np.ndarray, window: int = cfg.MA_WINDOW) -> np.ndarray:
    """``P_t / mean_window(P) - 1`` — ``*_trend_ma200``."""
    p = np.asarray(p, dtype=float)
    return p / rolling_mean(p, window) - 1.0


def rvol_neg(
    p: np.ndarray, window: int = cfg.RVOL_WINDOW, median_window: int = cfg.MEDIAN_WINDOW
) -> np.ndarray:
    """``-ln(σ_20(t) / median_252(σ_20))`` — ``kr_rvol_20``·``us_rvol_20``."""
    s = sigma_20(p, window)
    return -_safe_log_ratio(s, rolling_median(s, median_window))


def level_vs_median_neg(a: np.ndarray, window: int = cfg.MEDIAN_WINDOW) -> np.ndarray:
    """``-(A_t - median_252(A))`` — ``vix_level``·``credit_baa10y_level``."""
    a = np.asarray(a, dtype=float)
    return -(a - rolling_median(a, window))


def trading_value(tv_kospi: np.ndarray, tv_kosdaq: np.ndarray) -> np.ndarray:
    """TV = KOSPI + KOSDAQ 거래대금. 둘 다 있는 격자일만 (MI11)."""
    return np.asarray(tv_kospi, dtype=float) + np.asarray(tv_kosdaq, dtype=float)


def liq_20(
    tv_kospi: np.ndarray,
    tv_kosdaq: np.ndarray,
    mean_window: int = cfg.LIQ_MEAN_WINDOW,
    median_window: int = cfg.MEDIAN_WINDOW,
) -> np.ndarray:
    """``ln(mean_20(TV) / median_252(TV))`` — ``kr_liq_20``."""
    tv = trading_value(tv_kospi, tv_kosdaq)
    return _safe_log_ratio(rolling_mean(tv, mean_window), rolling_median(tv, median_window))


def foreign_20(
    f_kospi: np.ndarray,
    f_kosdaq: np.ndarray,
    tv_kospi: np.ndarray,
    tv_kosdaq: np.ndarray,
    window: int = cfg.FLOW_WINDOW,
) -> np.ndarray:
    """``Σ_20(외국인 순매수 KOSPI+KOSDAQ) / Σ_20(TV)`` — ``kr_foreign_20``. 분모 0 이하는 NaN."""
    num = rolling_sum(np.asarray(f_kospi, dtype=float) + np.asarray(f_kosdaq, dtype=float), window)
    den = rolling_sum(trading_value(tv_kospi, tv_kosdaq), window)
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def krw_20(fx: np.ndarray, lag: int = cfg.FX_LAG) -> np.ndarray:
    """``-ln(FX_t / FX_{t-20})`` — ``krw_20`` (원화 강세가 +)."""
    fx = np.asarray(fx, dtype=float)
    return -_safe_log_ratio(fx, lagged(fx, lag))


def term_spread(long_rate: np.ndarray, short_rate: np.ndarray) -> np.ndarray:
    """``kr10y - kr3y`` — ``kr_term``."""
    return np.asarray(long_rate, dtype=float) - np.asarray(short_rate, dtype=float)


def credit_chg_neg(baa: np.ndarray, lag: int = cfg.CREDIT_CHG_LAG) -> np.ndarray:
    """``-(BAA_t - BAA_{t-20})`` — ``credit_baa10y_chg``."""
    baa = np.asarray(baa, dtype=float)
    return -(baa - lagged(baa, lag))


def identity(a: np.ndarray) -> np.ndarray:
    """``us_term`` = ``T10Y2Y_t`` 그대로 (MI12)."""
    return np.asarray(a, dtype=float).copy()


# --------------------------------------------------------------------------- 성분 명세
@dataclass(frozen=True)
class ComponentSpec:
    """성분 하나: 이름·입력 계열·산식·거슬러 보는 행 수."""

    name: str
    inputs: tuple[str, ...]
    fn: Callable[[Arrays], np.ndarray]
    lookback: int


def _kr_specs(
    trend_lag_n: int, ma_n: int, rv_n: int, med_n: int, liq_n: int, flow_n: int, fx_n: int
) -> dict[str, ComponentSpec]:
    tvk, tvd = S_TV_KOSPI, S_TV_KOSDAQ
    return {
        "kr_trend_252": ComponentSpec(
            "kr_trend_252",
            (S_KR_PRICE,),
            lambda a: trend_lag(a[S_KR_PRICE], trend_lag_n),
            trend_lag_n,
        ),
        "kr_trend_ma200": ComponentSpec(
            "kr_trend_ma200", (S_KR_PRICE,), lambda a: trend_ma(a[S_KR_PRICE], ma_n), ma_n - 1
        ),
        "vix_level": ComponentSpec(
            "vix_level", (S_VIX,), lambda a: level_vs_median_neg(a[S_VIX], med_n), med_n - 1
        ),
        "kr_rvol_20": ComponentSpec(
            "kr_rvol_20",
            (S_KR_PRICE,),
            lambda a: rvol_neg(a[S_KR_PRICE], rv_n, med_n),
            rv_n + med_n - 1,
        ),
        "kr_liq_20": ComponentSpec(
            "kr_liq_20",
            (tvk, tvd),
            lambda a: liq_20(a[tvk], a[tvd], liq_n, med_n),
            max(liq_n, med_n) - 1,
        ),
        "kr_foreign_20": ComponentSpec(
            "kr_foreign_20",
            (S_FOREIGN_KOSPI, S_FOREIGN_KOSDAQ, tvk, tvd),
            lambda a: foreign_20(a[S_FOREIGN_KOSPI], a[S_FOREIGN_KOSDAQ], a[tvk], a[tvd], flow_n),
            flow_n - 1,
        ),
        "krw_20": ComponentSpec("krw_20", (S_FX,), lambda a: krw_20(a[S_FX], fx_n), fx_n),
        "kr_term": ComponentSpec(
            "kr_term", (S_KR10Y, S_KR3Y), lambda a: term_spread(a[S_KR10Y], a[S_KR3Y]), 0
        ),
    }


def kr_components(
    *,
    trend_lag_n: int = cfg.TREND_LAG,
    ma_window: int = cfg.MA_WINDOW,
    rvol_window: int = cfg.RVOL_WINDOW,
    median_window: int = cfg.MEDIAN_WINDOW,
    liq_window: int = cfg.LIQ_MEAN_WINDOW,
    flow_window: int = cfg.FLOW_WINDOW,
    fx_lag: int = cfg.FX_LAG,
) -> dict[str, ComponentSpec]:
    """한국 성분 8개 명세. 인자는 시험에서 창을 줄일 때만 바꾼다 (기본 = 동결 config)."""
    return _kr_specs(
        trend_lag_n, ma_window, rvol_window, median_window, liq_window, flow_window, fx_lag
    )


def us_components(
    *,
    trend_lag_n: int = cfg.TREND_LAG,
    ma_window: int = cfg.MA_WINDOW,
    rvol_window: int = cfg.RVOL_WINDOW,
    median_window: int = cfg.MEDIAN_WINDOW,
    credit_lag: int = cfg.CREDIT_CHG_LAG,
) -> dict[str, ComponentSpec]:
    """미국 성분 7개 명세 (``us_ftd_20``은 수집 전이라 없다, §3.1)."""
    p = S_US_PRICE
    return {
        "us_trend_252": ComponentSpec(
            "us_trend_252", (p,), lambda a: trend_lag(a[p], trend_lag_n), trend_lag_n
        ),
        "us_trend_ma200": ComponentSpec(
            "us_trend_ma200", (p,), lambda a: trend_ma(a[p], ma_window), ma_window - 1
        ),
        "vix_level": ComponentSpec(
            "vix_level",
            (S_VIX,),
            lambda a: level_vs_median_neg(a[S_VIX], median_window),
            median_window - 1,
        ),
        "us_rvol_20": ComponentSpec(
            "us_rvol_20",
            (p,),
            lambda a: rvol_neg(a[p], rvol_window, median_window),
            rvol_window + median_window - 1,
        ),
        "credit_baa10y_chg": ComponentSpec(
            "credit_baa10y_chg",
            (S_BAA,),
            lambda a: credit_chg_neg(a[S_BAA], credit_lag),
            credit_lag,
        ),
        "credit_baa10y_level": ComponentSpec(
            "credit_baa10y_level",
            (S_BAA,),
            lambda a: level_vs_median_neg(a[S_BAA], median_window),
            median_window - 1,
        ),
        "us_term": ComponentSpec("us_term", (S_T10Y2Y,), lambda a: identity(a[S_T10Y2Y]), 0),
    }


def sigma20_spec(price_series: str = "price", window: int = cfg.RVOL_WINDOW) -> ComponentSpec:
    """변동성 관리 기준선용 ``σ_20`` 명세 (``score.asset_sigma20``)."""
    return ComponentSpec(
        "sigma20", (price_series,), lambda a: sigma_20(a[price_series], window), window
    )
