"""거래 비용 모델 — 스프레드 + 제곱근 시장충격 (``02_labels_universe_benchmark.md`` §5)::

    cost_roundtrip_i = spread_i + 2 · impact_i
    spread_i  = max(2bp, 0.01 / price_i)            # $0.01 틱 / 가격. 1/price가 아니다 (V11)
    impact_i  = k · sigma_i,daily · sqrt(Q_i / ADV_i)   # 제곱근 충격

수수료는 0 (미국 리테일·기관 관행, 거래세 없음). **판정(E·I)에 쓰는 것은
Q=$10M, k=0.1이다** — 나머지는 ``Q_GRID × K_GRID`` 표로 민감도를 같이 낸다.
"""

from __future__ import annotations

from itertools import product

import polars as pl

from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily

#: 스프레드 하한 (비율). 2bp.
MIN_SPREAD = 0.0002

#: $0.01 틱 크기.
TICK_SIZE = 0.01

#: 판정(E·I)에 쓰는 기본값 (``02`` §5).
DEFAULT_Q_DOLLAR = 10_000_000.0
DEFAULT_K = 0.1

#: 민감도 표 격자 — 자금 규모 Q(달러) × 충격 계수 k.
Q_GRID: tuple[float, ...] = (1_000_000.0, 10_000_000.0, 100_000_000.0)
K_GRID: tuple[float, ...] = (0.05, 0.1, 0.2)


def spread(price: pl.Expr) -> pl.Expr:
    """왕복 스프레드 비용(비율). ``max(2bp, 0.01/price)``.

    ``0.01``은 틱 크기다 — ``1/price``가 아니다 (계획 검토 V11). 가격이 낮을수록
    스프레드가 커진다: $5 종목이면 20bp.
    """
    return pl.max_horizontal(pl.lit(MIN_SPREAD), pl.lit(TICK_SIZE) / price)


def impact(sigma_daily: pl.Expr, adv_dollar: pl.Expr, *, q_dollar: float, k: float) -> pl.Expr:
    """제곱근 시장충격(비율). ``k · sigma_daily · sqrt(Q / ADV)``."""
    return k * sigma_daily * (pl.lit(q_dollar) / adv_dollar).sqrt()


def cost_roundtrip(
    price: pl.Expr,
    sigma_daily: pl.Expr,
    adv_dollar: pl.Expr,
    *,
    q_dollar: float = DEFAULT_Q_DOLLAR,
    k: float = DEFAULT_K,
) -> pl.Expr:
    """왕복비용(비율). ``spread + 2*impact``. 수수료는 0이라 더하지 않는다."""
    return spread(price) + 2 * impact(sigma_daily, adv_dollar, q_dollar=q_dollar, k=k)


def daily_volatility(lake: UsLake, *, window: int = 20) -> pl.LazyFrame:
    """종목별 후행 ``window`` 거래일 일별 수익률 표준편차(비율). ``date, symbol, sigma_daily``.

    ``volatility_daily``(옵션 IV·HV)는 커버리지가 약 1,600종목뿐이라
    (``01_data_readiness.md`` §2) 비용 모델 전체 유니버스의 분모로 못 쓴다 —
    조정 종가에서 직접 실현 변동성을 구한다. 첫 ``window`` 거래일은 결측이다.
    """
    prices = adjusted_daily(lake).select("date", "symbol", "adj_close").sort(["symbol", "date"])
    returns = prices.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("_ret")
    )
    return returns.with_columns(
        pl.col("_ret")
        .rolling_std(window_size=window, min_samples=window)
        .over("symbol")
        .alias("sigma_daily")
    ).select("date", "symbol", "sigma_daily")


def cost_grid(
    df: pl.DataFrame,
    *,
    price_col: str = "close",
    sigma_col: str = "sigma_daily",
    adv_col: str = "adv_20d",
    q_grid: tuple[float, ...] = Q_GRID,
    k_grid: tuple[float, ...] = K_GRID,
) -> pl.DataFrame:
    """``Q x k`` 격자에서 왕복비용의 평균·중앙값 (횡단면 전체, ``02`` §5).

    ``df``는 ``price_col``·``sigma_col``·``adv_col``이 있는 (리밸런스, 종목) 표다.
    ``sigma_col``이 결측인 행(변동성 워밍업 안 된 초반)은 그 (Q,k) 계산에서
    제외한다. ``NaN``(레이크의 알려진 조정가 결함 — ``labels.py`` 모듈 docstring
    참고)도 같이 뺀다 — polars의 ``mean()``은 null과 달리 ``NaN``을 건너뛰지
    않아 하나만 섞여도 전체 평균이 ``NaN``이 된다.
    """
    rows: list[dict[str, float | int]] = []
    for q, k in product(q_grid, k_grid):
        cost = df.select(
            cost_roundtrip(
                pl.col(price_col), pl.col(sigma_col), pl.col(adv_col), q_dollar=q, k=k
            ).alias("cost")
        )["cost"]
        cost = cost.filter(cost.is_finite())
        rows.append(
            {
                "q_dollar": q,
                "k": k,
                "n": cost.len(),
                "mean_cost_roundtrip": cost.mean() if cost.len() else None,
                "median_cost_roundtrip": cost.median() if cost.len() else None,
            }
        )
    return pl.DataFrame(rows)
