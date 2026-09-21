"""M6 — 비용 반영 지표 · 분해 · CPCV 과적합 확률.

``05_validation_protocol.md`` §3·§4, ``02_labels_universe_benchmark.md``
§5·§6, ``06_execution_steps.md`` M6이 정본이다. 이 모듈은 이미 저장된 OOF
예측(``output/model_runs/*/predictions``)과 ``us_labels_v1`` 라벨을 입력으로
받아 돈으로 환산한 지표(E·E_ew·I·S·MDD·turnover·breakeven_Q·hit_top)와
Q×k 민감도표를 계산하는 순수 함수를 모은다.

**비용 공식은 ``modeler.us.cost``, 벤치마크는 ``modeler.us.benchmark``를
그대로 쓴다** — 여기서 다시 짜지 않는다(``06`` M6 지시). ``spread()``·
``impact()`` 두 building block만 그대로 가져와 "스프레드 배수" 같은
민감도 변형을 조립한다.

CPCV(``skfolio.CombinatorialPurgedCV``)로 채택 설정 하나를 재학습하는
루프는 이 모듈에 없다 — ``m4_run``의 모델 적합 함수에 묶여 있어
``m6_run.py``가 직접 오케스트레이션한다. 이 모듈은 그 결과(경로별 Sharpe
목록)를 받아 PBO·DSR을 계산하는 부분만 담당한다.

``m7_run.py``(holdout 개봉, ``06`` M7)도 이 모듈을 그대로 재사용한다 —
``excess_over_series``(``excess_over``의 월별 시계열 버전, permutation
귀무 확률 계산용)와 ``s_spread_long_short``/``s_long_short_summary``(``S``의
롱/숏 기여 분해, M6이 처음 손으로 낸 것을 재사용 가능하게 옮겼다),
``sign_flip_permutation_probability``(M8이 쓸 갈래 A 귀무 확률)만 새로
더했다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _scipy_skew

from modeler.us import cost as cost_mod
from modeler.us.scan import DECILE_FRACTION, MIN_NAMES, decile_spread

#: 판정 포트폴리오 크기 — ``02`` §6.
TOP_K = 100

#: 지표 I(``02`` §4)의 거래가능 유니버스 하한 — 조정 전 종가.
PRICE_FLOOR = 5.0

#: Y16이 요구하는 스프레드 2배 민감도.
SPREAD_SENSITIVITY_MULTIPLIER = 2.0

#: ``sigma_daily``가 이 값을 넘으면 ``corp_actions``의 중복 분할 행 결함
#: (``07_risks.md`` Y8c — ``labels.py`` 모듈 docstring이 자세히 적었다)이
#: 조정 종가를 오염시켜 만든 후유증으로 본다. 진짜 종목의 20거래일 롤링
#: 일별 수익률 표준편차가 20%를 넘는 경우는 사실상 없다 — 2026-09-20 실측
#: 유니버스 전체에서 최댓값이 **50,758**(!)까지 나온다(GRPH 2021-07-01).
#: ``labels.py``의 ``MAX_PLAUSIBLE_ABS_L0``과 같은 자리의 방어선이다: 원천
#: 결함은 ``prices.py``에서 고치지 않고(``build_labels.py`` 판단과 같다),
#: 이 값을 쓰는 자리에서 명백히 오염된 입력만 걸러낸다.
MAX_PLAUSIBLE_SIGMA_DAILY = 0.20

__all__ = [
    "TOP_K",
    "PRICE_FLOOR",
    "SPREAD_SENSITIVITY_MULTIPLIER",
    "topk_rows",
    "turnover_by_date",
    "monthly_cost_drag",
    "portfolio_track",
    "excess_over",
    "excess_over_series",
    "hit_rate",
    "s_spread",
    "s_spread_long_short",
    "s_long_short_summary",
    "breakeven_q_dollar",
    "sensitivity_grid",
    "sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "pbo_fraction_nonpositive",
    "sign_flip_permutation_probability",
]


# --- 1. top-k 프레임 ----------------------------------------------------------


def topk_rows(
    df: pl.DataFrame,
    *,
    pred_col: str = "pred",
    k: int = TOP_K,
    date_col: str = "date",
) -> pl.DataFrame:
    """그날 예측 상위 ``k``개 행 — ordinal rank로 동순위를 결정적으로 가른다.

    ``modeler.etl.metrics._topk_ranked``(한국)와 같은 규칙이다: null·비유한
    예측은 애초에 살 수 없으므로 먼저 던다.
    """
    clean = df.filter(pl.col(pred_col).is_not_null() & pl.col(pred_col).is_finite())
    rank = pl.col(pred_col).rank("ordinal", descending=True).over(date_col)
    return clean.with_columns(rank.alias("_topk_rank")).filter(pl.col("_topk_rank") <= k)


def turnover_by_date(
    membership: dict[date, set[str]], ordered_dates: Sequence[date]
) -> dict[date, float]:
    """월별 회전율 — 그 달과 바로 전 달 보유 종목 집합의 비대칭차.

    ``modeler.etl.metrics.portfolio_turnover``와 같은 정의
    (``1 - |A∩B| / max(|A|,|B|)``)를 날짜별로 남긴다(그 함수는 전체 평균
    스칼라 하나만 준다) — 월별 비용 항력을 내려면 달마다 값이 있어야 한다.
    첫 리밸런스는 비교할 전달이 없다 — **전량 신규 진입으로 보고 1.0을
    쓴다** (포트폴리오를 처음 구성하는 비용은 실제로 발생한다).
    """
    present = [d for d in ordered_dates if d in membership]
    out: dict[date, float] = {}
    if present:
        out[present[0]] = 1.0
    for prev, curr in zip(present, present[1:]):
        a, b = membership[prev], membership[curr]
        denom = max(len(a), len(b))
        out[curr] = float("nan") if denom == 0 else 1.0 - len(a & b) / denom
    return out


# --- 2. 비용 항력 --------------------------------------------------------------


def monthly_cost_drag(
    picked: pl.DataFrame,
    *,
    turnover: dict[date, float],
    q_dollar: float,
    k: float,
    spread_multiplier: float = 1.0,
    price_col: str = "close",
    sigma_col: str = "sigma_daily",
    adv_col: str = "adv_20d",
    date_col: str = "date",
) -> pl.DataFrame:
    """월별 비용 항력 = 그달 보유종목 평균 왕복비용 × 그달 회전율.

    실제 매매는 교체된 종목분(``turnover``)에서만 일어난다는 근사다 —
    ``modeler.models._01_20_access_return_rank.experiments.run_topk_cost_check``의
    ``cost = turnover * effective_bps``(한국)와 같은 관례다. 왕복비용 자체는
    ``modeler.us.cost.spread``·``impact``를 그대로 조립한다(``spread_multiplier``
    는 Y16 민감도용 배수, 기본 1.0이면 ``cost.cost_roundtrip``과 같다).
    ``sigma_col``이 ``MAX_PLAUSIBLE_SIGMA_DAILY``를 넘는 행(corp_actions
    결함 Y8c의 후유증)은 그 달 평균에서 뺀다 — 명백히 오염된 값 하나가
    보유종목 100개 평균을 통째로 왜곡하는 것을 막는다.

    반환: ``date, mean_cost_roundtrip, turnover, cost_drag``.
    """
    per_row_cost = (
        spread_multiplier * cost_mod.spread(pl.col(price_col))
        + 2 * cost_mod.impact(pl.col(sigma_col), pl.col(adv_col), q_dollar=q_dollar, k=k)
    ).alias("_cost_roundtrip")
    with_cost = picked.with_columns(per_row_cost).filter(
        pl.col("_cost_roundtrip").is_finite() & (pl.col(sigma_col) <= MAX_PLAUSIBLE_SIGMA_DAILY)
    )
    mean_cost = (
        with_cost.group_by(date_col)
        .agg(pl.col("_cost_roundtrip").mean().alias("mean_cost_roundtrip"))
        .sort(date_col)
    )
    rows = []
    for row in mean_cost.iter_rows(named=True):
        d = row[date_col]
        t = turnover.get(d, float("nan"))
        drag = row["mean_cost_roundtrip"] * t if math.isfinite(t) else float("nan")
        rows.append(
            {
                date_col: d,
                "mean_cost_roundtrip": row["mean_cost_roundtrip"],
                "turnover": t,
                "cost_drag": drag,
            }
        )
    schema = {
        date_col: pl.Date,
        "mean_cost_roundtrip": pl.Float64,
        "turnover": pl.Float64,
        "cost_drag": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


# --- 3. 포트폴리오 궤적 ---------------------------------------------------------


def portfolio_track(
    df: pl.DataFrame,
    *,
    pred_col: str = "pred",
    k: int = TOP_K,
    date_col: str = "date",
    symbol_col: str = "symbol",
    value_col: str = "L0",
    price_col: str = "close",
    sigma_col: str = "sigma_daily",
    adv_col: str = "adv_20d",
    q_dollar: float = cost_mod.DEFAULT_Q_DOLLAR,
    k_impact: float = cost_mod.DEFAULT_K,
    spread_multiplier: float = 1.0,
) -> pl.DataFrame:
    """월별 top-k 궤적 하나: ``date, n_held, gross_return, turnover, cost_drag, net_return``.

    ``df``는 이미 그 유니버스로 걸러진 (date, symbol, pred, value_col,
    price_col, sigma_col, adv_col) 프레임이어야 한다(지표 I는 호출 전에
    ``price_ge_5``로 걸러서 넘긴다).
    """
    picked = topk_rows(df, pred_col=pred_col, k=k, date_col=date_col)
    membership = {
        d: set(grp[symbol_col].to_list())
        for (d,), grp in picked.group_by([date_col], maintain_order=True)
    }
    ordered_dates = sorted(membership)
    t_by_date = turnover_by_date(membership, ordered_dates)

    gross = (
        picked.group_by(date_col)
        .agg(pl.col(value_col).mean().alias("gross_return"), pl.len().alias("n_held"))
        .sort(date_col)
    )
    cost_df = monthly_cost_drag(
        picked,
        turnover=t_by_date,
        q_dollar=q_dollar,
        k=k_impact,
        spread_multiplier=spread_multiplier,
        price_col=price_col,
        sigma_col=sigma_col,
        adv_col=adv_col,
        date_col=date_col,
    )
    return (
        gross.join(cost_df, on=date_col, how="left")
        .with_columns(
            (pl.col("gross_return") - pl.col("cost_drag").fill_null(0.0)).alias("net_return")
        )
        .sort(date_col)
    )


def excess_over(
    track: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    return_col: str = "net_return",
    benchmark_col: str,
    date_col: str = "date",
) -> tuple[float, int]:
    """``track``의 ``return_col``에서 ``benchmark``의 ``benchmark_col``을 뺀 평균과 개월 수."""
    merged = track.join(benchmark, on=date_col, how="inner")
    if merged.height == 0:
        return float("nan"), 0
    diff = (merged[return_col] - merged[benchmark_col]).drop_nulls()
    if diff.len() == 0:
        return float("nan"), 0
    return float(diff.mean()), diff.len()


def excess_over_series(
    track: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    return_col: str = "net_return",
    benchmark_col: str,
    date_col: str = "date",
) -> pl.DataFrame:
    """``excess_over``와 같은 조인이지만 평균 하나가 아니라 ``(date, excess)`` 시계열을 준다.

    ``m7_run``의 부호 뒤섞기(permutation) 귀무 확률 계산(``06`` M8)이 월별
    시계열을 그대로 필요로 한다 — ``excess_over``는 스칼라 평균만 준다.
    """
    merged = track.join(benchmark, on=date_col, how="inner").sort(date_col)
    schema = {date_col: pl.Date, "excess": pl.Float64}
    if merged.height == 0:
        return pl.DataFrame(schema=schema)
    diff = (pl.col(return_col) - pl.col(benchmark_col)).alias("excess")
    return merged.select(date_col, diff).drop_nulls()


def hit_rate(picked: pl.DataFrame, *, value_col: str = "L2") -> float:
    """top-k로 뽑힌 (날짜, 종목) 전체에서 ``value_col`` > 0인 비율."""
    values = picked[value_col].drop_nulls()
    if values.len() == 0:
        return float("nan")
    return float((values > 0).sum() / values.len())


def s_spread(
    df: pl.DataFrame,
    *,
    pred_col: str = "pred",
    value_col: str = "L0",
    date_col: str = "date",
    min_names: int = MIN_NAMES,
    fraction: float = DECILE_FRACTION,
) -> tuple[float, int]:
    """``S`` — 예측 10분위 top - bottom, L0 동일가중 월수익 평균 (``05`` §4).

    ``modeler.us.scan.decile_spread``를 그대로 재사용한다(피쳐 대신 예측을
    ``feature_col``에 넣는다) — 다시 구현하지 않는다.
    """
    return decile_spread(
        df,
        feature_col=pred_col,
        value_col=value_col,
        group_col=date_col,
        min_names=min_names,
        fraction=fraction,
    )


def s_spread_long_short(
    df: pl.DataFrame,
    *,
    pred_col: str = "pred",
    value_col: str = "L0",
    date_col: str = "date",
    min_names: int = MIN_NAMES,
    fraction: float = DECILE_FRACTION,
) -> pl.DataFrame:
    """``S``를 월별로 롱(상위−유니버스)·숏(유니버스−하위) 기여로 가른다.

    ``modeler.us.scan.decile_spread``와 같은 top/bottom 정의(``pred_col``의
    ordinal rank, ``min_names`` 미만인 날짜는 뺀다)를 쓴다 — M6 보고서가
    처음 손으로 낸 분해("S 의 88% 가 숏 쪽")를 재사용 가능한 함수로 옮긴
    것뿐이고 숫자를 다시 정의하지 않는다.

    반환: ``date, top, universe, bottom, spread, long_excess, short_excess``
    (``spread == long_excess + short_excess``, 부동소수점 오차 안에서).
    """
    ranked = df.with_columns(
        pl.col(pred_col).rank("ordinal").over(date_col).alias("_r"),
        pl.len().over(date_col).alias("_n"),
    ).filter(pl.col("_n") >= min_names)
    schema = {
        date_col: pl.Date,
        "top": pl.Float64,
        "universe": pl.Float64,
        "bottom": pl.Float64,
        "spread": pl.Float64,
        "long_excess": pl.Float64,
        "short_excess": pl.Float64,
    }
    if ranked.height == 0:
        return pl.DataFrame(schema=schema)
    ranked = ranked.with_columns(((pl.col("_r") - 1) / (pl.col("_n") - 1)).alias("_pct"))
    top = (
        ranked.filter(pl.col("_pct") >= 1 - fraction)
        .group_by(date_col)
        .agg(pl.col(value_col).mean().alias("top"))
    )
    bottom = (
        ranked.filter(pl.col("_pct") <= fraction)
        .group_by(date_col)
        .agg(pl.col(value_col).mean().alias("bottom"))
    )
    universe = ranked.group_by(date_col).agg(pl.col(value_col).mean().alias("universe"))
    return (
        top.join(bottom, on=date_col, how="inner")
        .join(universe, on=date_col, how="inner")
        .with_columns(
            (pl.col("top") - pl.col("bottom")).alias("spread"),
            (pl.col("top") - pl.col("universe")).alias("long_excess"),
            (pl.col("universe") - pl.col("bottom")).alias("short_excess"),
        )
        .sort(date_col)
    )


def s_long_short_summary(monthly: pl.DataFrame) -> dict[str, float | int]:
    """``s_spread_long_short`` 출력의 시계열 평균과 롱/숏 몫.

    ``long_share + short_share == 1.0``(부동소수점 오차 안에서) — 몫은
    ``long_excess``·``short_excess``의 시계열 평균 합을 분모로 쓴다(각 달의
    비중이 아니라 전체 기간 평균의 비중, M6 보고서와 같은 정의).
    """
    if monthly.height == 0:
        return {
            "S": float("nan"),
            "long_excess_mean": float("nan"),
            "short_excess_mean": float("nan"),
            "long_share": float("nan"),
            "short_share": float("nan"),
            "n_months": 0,
        }
    s_mean = float(monthly["spread"].mean())
    long_mean = float(monthly["long_excess"].mean())
    short_mean = float(monthly["short_excess"].mean())
    total = long_mean + short_mean
    return {
        "S": s_mean,
        "long_excess_mean": long_mean,
        "short_excess_mean": short_mean,
        "long_share": long_mean / total if total else float("nan"),
        "short_share": short_mean / total if total else float("nan"),
        "n_months": monthly.height,
    }


# --- 4. 손익분기 자금규모 · 민감도표 --------------------------------------------


def _mean_cost_drag_at(
    picked: pl.DataFrame,
    turnover: dict[date, float],
    *,
    q_dollar: float,
    k: float,
    spread_multiplier: float,
    price_col: str,
    sigma_col: str,
    adv_col: str,
    date_col: str,
) -> float:
    df = monthly_cost_drag(
        picked,
        turnover=turnover,
        q_dollar=q_dollar,
        k=k,
        spread_multiplier=spread_multiplier,
        price_col=price_col,
        sigma_col=sigma_col,
        adv_col=adv_col,
        date_col=date_col,
    )
    values = df["cost_drag"].drop_nulls() if df.height else df["cost_drag"]
    return float(values.mean()) if values.len() else float("nan")


def breakeven_q_dollar(
    gross_alpha: float,
    picked: pl.DataFrame,
    turnover: dict[date, float],
    *,
    k_impact: float = cost_mod.DEFAULT_K,
    spread_multiplier: float = 1.0,
    price_col: str = "close",
    sigma_col: str = "sigma_daily",
    adv_col: str = "adv_20d",
    date_col: str = "date",
    q_low: float = 1e3,
    q_high: float = 1e13,
    max_iter: int = 200,
) -> float | None:
    """``E(Q) = gross_alpha - mean_cost_drag(Q) = 0``이 되는 ``Q``.

    비용은 ``sqrt(Q)``라 ``Q``에 단조증가한다 — 로그스케일 이분법으로 푼다.
    ``gross_alpha``(비용 전 알파, 즉 top-k 총수익 - 벤치마크)가 이미
    0 이하면 비용 없이도 지는 상태라 손익분기가 없다(``None``). ``q_high``
    에서도 여전히 이득이면 그 범위 안에서 손익분기가 없다는 뜻으로
    ``None``을 돌려준다(기록만 — 무한정 올리지 않는다).
    """
    if not math.isfinite(gross_alpha) or gross_alpha <= 0:
        return None

    def f(q: float) -> float:
        drag = _mean_cost_drag_at(
            picked,
            turnover,
            q_dollar=q,
            k=k_impact,
            spread_multiplier=spread_multiplier,
            price_col=price_col,
            sigma_col=sigma_col,
            adv_col=adv_col,
            date_col=date_col,
        )
        return gross_alpha - drag

    f_low, f_high = f(q_low), f(q_high)
    if f_low <= 0:
        return q_low
    if f_high > 0:
        return None

    lo, hi = q_low, q_high
    mid = hi
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-9 or (hi - lo) / hi < 1e-9:
            return mid
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return mid


def sensitivity_grid(
    picked: pl.DataFrame,
    turnover: dict[date, float],
    *,
    gross_return_by_date: pl.DataFrame,
    benchmark: pl.DataFrame,
    benchmark_col: str,
    q_grid: tuple[float, ...] = cost_mod.Q_GRID,
    k_grid: tuple[float, ...] = cost_mod.K_GRID,
    price_col: str = "close",
    sigma_col: str = "sigma_daily",
    adv_col: str = "adv_20d",
    date_col: str = "date",
) -> pl.DataFrame:
    """``Q × k`` 9칸에서 ``E``(비용 반영 초과수익) — ``02`` §5, Y16.

    ``gross_return_by_date``는 ``date, gross_return``(비용 전 top-k 평균수익).
    각 칸마다 그 ``(Q, k)``에서 다시 비용을 계산해 ``E = mean(gross - drag) -
    mean(benchmark)``를 낸다.
    """
    rows = []
    for q in q_grid:
        for k in k_grid:
            cost_df = monthly_cost_drag(
                picked,
                turnover=turnover,
                q_dollar=q,
                k=k,
                price_col=price_col,
                sigma_col=sigma_col,
                adv_col=adv_col,
                date_col=date_col,
            )
            net = gross_return_by_date.join(cost_df, on=date_col, how="left").with_columns(
                (pl.col("gross_return") - pl.col("cost_drag").fill_null(0.0)).alias("net_return")
            )
            e_value, n = excess_over(net, benchmark, benchmark_col=benchmark_col, date_col=date_col)
            rows.append({"q_dollar": q, "k": k, "E": e_value, "n_months": n})
    return pl.DataFrame(rows)


# --- 5. Sharpe · PBO · DSR -----------------------------------------------------


def sharpe_ratio(returns: Sequence[float], *, periods_per_year: int | None = 12) -> float:
    """연환산(또는 기간당) Sharpe — ``mean/std`` × ``sqrt(periods_per_year)``.

    ``periods_per_year=None``이면 연환산하지 않은 기간당 Sharpe를 돌려준다
    (DSR 공식은 관측치와 같은 빈도의 Sharpe를 요구한다 — Bailey·Lopez de
    Prado 2014).
    """
    values = np.array(
        [r for r in returns if r is not None and isinstance(r, (int, float)) and math.isfinite(r)],
        dtype=float,
    )
    if values.size < 2:
        return float("nan")
    sd = values.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr = values.mean() / sd
    return float(sr * math.sqrt(periods_per_year)) if periods_per_year else float(sr)


_EULER_MASCHERONI = 0.5772156649015328606


def expected_max_sharpe(n_trials: int, sr_var_across_trials: float) -> float:
    """``N``개 시행에서 기대되는 최대 Sharpe(귀무가설: 진짜 알파는 0) — Bailey·
    Lopez de Prado(2014) 식 2. 시행들의 Sharpe가 정규분포 ``N(0, V)``를
    따른다고 보고 그 최댓값의 기댓값을 근사한다(Euler-Mascheroni 상수로
    보정)."""
    if n_trials < 2 or sr_var_across_trials < 0:
        return float("nan")
    sd = math.sqrt(sr_var_across_trials)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sd * ((1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    sr_hat: float,
    *,
    n_trials: int,
    sr_var_across_trials: float,
    skewness: float,
    kurtosis: float,
    n_obs: int,
) -> dict[str, float]:
    """DSR — Bailey·Lopez de Prado(2014) "The Deflated Sharpe Ratio" 식 8·9.

    ``sr_hat``·``sr_var_across_trials``는 **같은 빈도**(이 모듈에서는
    월간, 연환산 전)의 Sharpe여야 ``n_obs``(그 빈도의 관측치 수)와 척도가
    맞는다. ``skewness``·``kurtosis``(첨도, 정규분포=3)는 채택된 전략의
    월별 수익률 분포에서 잰다.

    반환: ``sr0``(다중비교 보정 기준 Sharpe) · ``z`` · ``dsr``(``Φ(z)``,
    진짜 Sharpe가 0보다 클 확률).
    """
    sr0 = expected_max_sharpe(n_trials, sr_var_across_trials)
    if not math.isfinite(sr0) or n_obs < 2:
        return {"sr0_expected_max_sharpe": sr0, "z": float("nan"), "dsr": float("nan")}
    denom = math.sqrt(max(1.0 - skewness * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2, 1e-12))
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1) / denom
    return {"sr0_expected_max_sharpe": sr0, "z": float(z), "dsr": float(norm.cdf(z))}


def sample_skew_kurtosis(returns: Sequence[float]) -> tuple[float, float]:
    """표본 왜도(skewness)·첨도(kurtosis, 정규분포=3) — ``scipy.stats`` 그대로."""
    values = np.array(
        [r for r in returns if r is not None and isinstance(r, (int, float)) and math.isfinite(r)],
        dtype=float,
    )
    if values.size < 3:
        return float("nan"), float("nan")
    return float(_scipy_skew(values, bias=False)), float(_scipy_kurtosis(values, bias=False) + 3.0)


def pbo_fraction_nonpositive(path_values: Sequence[float]) -> float:
    """PBO 프록시 — CPCV 경로 Sharpe 중 0 이하인 비율.

    **고전 Bailey CSCV PBO(2017)와 다르다.** 원래 정의는 여러 후보 전략을
    IS에서 고른 뒤 그 전략의 OOS 순위(중앙값 대비)로 과적합 확률을 잰다 —
    최소 두 개 이상의 후보가 있어야 순위를 매길 수 있다. 이 계획은 채택
    설정 **하나**만 CPCV로 재학습한다(``05`` §3 "고른 설정 하나에 대해서만")
    — 비교할 다른 후보가 없다. 그래서 "그 하나의 설정이 독립적으로 purge된
    OOS 경로 28개 중 몇 개에서 조차 살아남지 못하는가"로 과적합 확률을
    대신 잰다: **PBO = P(경로 Sharpe ≤ 0)**. 표본 밖에서 재현되지 않을
    확률이라는 원래 취지는 같지만 계산 방법이 다르다 — 보고서에 이 차이를
    적는다.
    """
    values = [v for v in path_values if v is not None and math.isfinite(v)]
    if not values:
        return float("nan")
    return sum(1 for v in values if v <= 0) / len(values)


# --- 6. 부호 뒤섞기 permutation — 갈래 A 귀무 확률 -------------------------------


def sign_flip_permutation_probability(
    series_map: dict[str, Sequence[float]],
    *,
    n_perm: int = 1000,
    seed: int,
) -> dict[str, float | int]:
    """``series_map``의 모든 시계열이 동시에 양수 평균이 되는 귀무 확률.

    ``06`` M8 지시: holdout 월수익을 부호 뒤섞기(permutation) ``n_perm``회
    해서 판정 넷(``E``·``E_ew``·``I``·``S``)이 전부 양수로 나오는 비율을
    낸다 — "갈래 A가 잡음만으로도 나올 확률"이다.

    한 시행 안에서는 **같은 부호 벡터 하나**를 ``series_map``의 모든
    시계열에 곱한다(달마다 독립으로 ±1을 하나씩 뽑는다) — 넷은 서로 다른
    벤치마크·정의를 쓰지만 같은 달의 시장 실현 하나를 공유하므로, "그 달의
    실현이 우연히 플러스/마이너스로 뒤집혔다면" 이라는 귀무가설은 네
    시계열에 같은 뒤집힘을 준다는 뜻이다. 시계열마다 독립으로 부호를
    뽑으면 실제로는 상관된 네 지표를 인위적으로 독립시켜 귀무 확률을
    낮게(더 엄격하게) 왜곡한다.

    모든 시계열은 길이가 같아야 한다(``date``로 이미 정렬해 넘긴다는 전제 —
    이 함수는 날짜를 보지 않고 위치로만 대응시킨다).
    """
    keys = list(series_map)
    if not keys:
        raise ValueError("series_map이 비어 있습니다")
    arrays = [np.asarray(series_map[k], dtype=float) for k in keys]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"series 길이가 다릅니다: {dict(zip(keys, (a.size for a in arrays), strict=True))}"
        )
    n = lengths.pop()
    if n == 0:
        return {
            "n_perm": n_perm,
            "n_months": 0,
            "seed": seed,
            "keys": keys,
            "probability_all_positive": float("nan"),
        }

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if all(bool(np.mean(a * signs) > 0) for a in arrays):
            hits += 1
    return {
        "n_perm": n_perm,
        "n_months": n,
        "seed": seed,
        "keys": keys,
        "probability_all_positive": hits / n_perm,
    }
