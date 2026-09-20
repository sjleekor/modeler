"""F3 변동성 — ``rv_20`` · ``rv_60`` · ``idio_vol_60`` · ``beta_252``.

거래일 창은 ``_daily.py``의 관례(종목별 가격 행 순서)를 쓴다. ``rv_20``·``rv_60``은
당일 단순수익률(``ret``)의 20·60일 표준편차 그대로다.

``idio_vol_60``(SPY 회귀 잔차 60일 sd)와 ``beta_252``는 SPY 수익률과의 단순회귀
(절편 포함 OLS)에서 나온다. 매 행마다 60·252개짜리 OLS를 다시 푸는 대신, 표준
분산분해 항등식을 쓴다::

    beta        = Cov(r, m) / Var(m)
    Var(resid)  = Var(r) - beta^2 * Var(m)   (교차항 소거로 나오는 정확한 항등식)
    idio_vol    = sqrt(max(Var(resid), 0))

굴림 공분산·분산은 ``pl.rolling_cov``(대각이면 곧 분산)로 낸다. ``Var(resid)``가
부동소수 오차로 아주 살짝 음수가 나올 수 있어 ``0``으로 잘라낸 뒤 제곱근을 씌운다.

SPY 자체는 패널 종목이 아니므로(ETF는 유니버스에서 빠진다) ``_daily.with_market_return``이
날짜로 별도 조인해 ``ret_spy``를 붙인다 — 그 결과 종목이 실제로 거래된 날짜에서만
SPY와 짝지어 회귀하게 된다(``_daily.py`` 모듈 docstring 참고).

**티커 재사용 경고.** 창이 몇 년치 공백을 건너뛰면(``momentum.py`` 참고 — 검산
5번에서 ``CBK`` 등으로 실제 발견) 굴림 통계가 무관한 두 회사의 수익률을 섞는다.
``_daily.mask_ticker_reuse_gap``으로 창의 가장 오래된 행까지 무효화한다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._daily import (
    daily_prices,
    join_features,
    mask_ticker_reuse_gap,
    panel_symbols,
    with_market_return,
)
from modeler.us.lake import UsLake

_FEATURES = ("rv_20", "rv_60", "idio_vol_60", "beta_252")


def _residual_vol(ret: pl.Expr, ret_spy: pl.Expr, window: int) -> pl.Expr:
    var_stock = ret.rolling_std(window_size=window).over("symbol") ** 2
    var_mkt = ret_spy.rolling_std(window_size=window).over("symbol") ** 2
    cov = pl.rolling_cov(ret, ret_spy, window_size=window).over("symbol")
    beta = cov / var_mkt
    resid_var = (var_stock - beta**2 * var_mkt).clip(lower_bound=0.0)
    return resid_var.sqrt()


def _beta(ret: pl.Expr, ret_spy: pl.Expr, window: int) -> pl.Expr:
    var_mkt = ret_spy.rolling_std(window_size=window).over("symbol") ** 2
    cov = pl.rolling_cov(ret, ret_spy, window_size=window).over("symbol")
    return cov / var_mkt


def add_volatility(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F3 변동성 피쳐 + ``_isna``를 붙인다."""
    daily = with_market_return(daily_prices(lake, symbols=panel_symbols(panel)), lake)
    ret = pl.col("ret")
    ret_spy = pl.col("ret_spy")
    date_col = pl.col("date")

    rv_20 = ret.rolling_std(window_size=20).over("symbol")
    rv_60 = ret.rolling_std(window_size=60).over("symbol")
    idio_vol_60 = _residual_vol(ret, ret_spy, 60)
    beta_252 = _beta(ret, ret_spy, 252)

    features = daily.with_columns(
        mask_ticker_reuse_gap(rv_20, date_col, 19).alias("rv_20"),
        mask_ticker_reuse_gap(rv_60, date_col, 59).alias("rv_60"),
        mask_ticker_reuse_gap(idio_vol_60, date_col, 59).alias("idio_vol_60"),
        mask_ticker_reuse_gap(beta_252, date_col, 251).alias("beta_252"),
    )
    return join_features(panel, features, _FEATURES)
