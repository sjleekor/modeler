"""성분 -> 하위 점수 -> MRS (사양 §3.3, 사전등록 §3.2·§3.3).

::

    pct_i(t)    = vintage.py 의 확장창 백분위 (warm-up 1,260, t 시점 vintage)
    score_sub(t) = mean_i( + 성분은 pct, - 성분은 100 - pct )      (살아 있는 성분만)
    MRS(t)       = mean( T, V, L )                                  (살아 있는 하위 점수만)

성분이 하나도 없는 하위 점수는 뺀다. 셋 다 없으면 MRS는 null이다 (비중 100%는 weights.py의 일).
선행 백필이 안 된 성분을 빼고 돌리는 데 이 규칙을 쓰지 않는다 — 입력 계열이 통째로 없으면
``compute_scores``가 예외를 낸다 (§3.1·§3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from modeler.scores.mrs import config as cfg
from modeler.scores.mrs.components import (
    ComponentSpec,
    kr_components,
    sigma20_spec,
    us_components,
)
from modeler.scores.mrs.vintage import (
    NO_DECISION,
    Grid,
    _epoch_to_series,
    _series_to_epoch,
    vintage_component_result,
)

PRICE_KEY = "price"


@dataclass
class ScoreResult:
    """``frame``: 격자일당 한 행. ``first_dates``: 성분별 첫 날짜. ``diagnostics``: 엔진 진단."""

    frame: pl.DataFrame
    first_dates: pl.DataFrame
    diagnostics: dict = field(default_factory=dict)


def component_specs(market: str) -> dict[str, ComponentSpec]:
    """시장별 동결 성분 명세 (KR 8개, US 7개)."""
    m = market.upper()
    if m == "KR":
        return kr_components()
    if m == "US":
        return us_components()
    raise ValueError(f"market은 KR 또는 US다: {market!r}")


def market_components(market: str) -> dict[str, tuple[str, str]]:
    """성분 -> (하위 점수, 부호) 동결표."""
    return dict(cfg.KR_COMPONENTS if market.upper() == "KR" else cfg.US_COMPONENTS)


def _mean_alive(stack: list[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN을 뺀 평균과 살아 있는 개수. 하나도 없으면 NaN."""
    if not stack:
        return np.full(n, np.nan), np.zeros(n, dtype=np.int64)
    m = np.vstack(stack)
    cnt = (~np.isnan(m)).sum(axis=0).astype(np.int64)
    tot = np.nansum(m, axis=0)
    with np.errstate(all="ignore"):
        mean = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)
    return mean, cnt


def compute_scores(
    market: str,
    grid: Grid,
    inputs: dict[str, pl.DataFrame],
    *,
    warmup: int = cfg.WARMUP_VALID_OBS,
    staleness_days: int | dict[str, int] = cfg.MACRO_STALENESS_DAYS,
    specs: dict[str, ComponentSpec] | None = None,
    signs: dict[str, tuple[str, str]] | None = None,
) -> ScoreResult:
    """한 시장의 성분 백분위·하위 점수·MRS를 격자 전체에 대해 계산한다.

    Parameters
    ----------
    inputs
        입력 계열 이름(``components.S_*``) -> ``date, value, available_at`` 행. 점수 protocol
        ``vix_proxy``는 ``S_VIX`` 행만 근사 행으로 바꿔 같은 함수를 부른다 (MI13).
    specs, signs
        시험에서 창을 줄이거나 성분을 바꿀 때만 쓴다. 기본은 동결 config의 표.

    Returns
    -------
    ScoreResult
        ``frame`` 열: ``date, decision_at, MRS, sub_T, sub_V, sub_L, n_components_T/V/L,
        pct_<성분>, x_<성분>, nvalid_<성분>, availability_ok, input_available_at_max``.
        마지막 격자일(결정 시각 없음)은 값이 전부 null인 행으로 남는다.
    """
    specs = specs if specs is not None else component_specs(market)
    signs = signs if signs is not None else market_components(market)
    missing = sorted({k for sp in specs.values() for k in sp.inputs} - set(inputs))
    if missing:
        raise ValueError(f"입력 계열이 없다: {missing} (성분을 임의로 빼고 돌리지 않는다, §3.1)")

    n = grid.n
    cols: dict[str, pl.Series] = {}
    pct: dict[str, np.ndarray] = {}
    avail_max = np.full(n, NO_DECISION, dtype=np.int64)
    first_rows: list[dict] = []
    diag: dict[str, dict] = {}

    for name, sp in specs.items():
        res = vintage_component_result(
            grid,
            {k: inputs[k] for k in sp.inputs},
            sp.fn,
            lookback=sp.lookback,
            staleness_days=staleness_days,
            stat="pct",
            warmup=warmup,
        )
        f = res.frame
        x = f["x"]
        p = f["stat_value"].fill_null(np.nan).to_numpy().astype(float)
        pct[name] = p
        cols[f"pct_{name}"] = f["stat_value"].alias(f"pct_{name}")
        cols[f"x_{name}"] = x.alias(f"x_{name}")
        cols[f"nvalid_{name}"] = f["nvalid"].alias(f"nvalid_{name}")
        avail_max = np.maximum(avail_max, _series_to_epoch(f["input_available_at_max"]))
        dates = f["date"]
        first_rows.append(
            {
                "component": name,
                "sub": signs[name][0],
                "first_value_date": dates.filter(x.is_not_null()).min(),
                "first_pct_date": dates.filter(f["stat_value"].is_not_null()).min(),
            }
        )
        diag[name] = {
            "n_recomputed_rows": res.n_recomputed_rows,
            "n_batches": res.n_batches,
            "carried_days": int(f["n_carried"].sum()),
            "x_null_days": int(x.is_null().sum()),
        }

    sub_vals: dict[str, np.ndarray] = {}
    sub_cnt: dict[str, np.ndarray] = {}
    for s in cfg.SUB_SCORES:
        stack = []
        for name, (sub, sign) in signs.items():
            if sub != s or name not in pct:
                continue
            stack.append(pct[name] if sign == "+" else 100.0 - pct[name])
        sub_vals[s], sub_cnt[s] = _mean_alive(stack, n)
    mrs, _ = _mean_alive([sub_vals[s] for s in cfg.SUB_SCORES], n)

    dec = grid.decision_us
    # 가용성 확인 (§3.1·§11): 입력 행의 가용 시각이 결정 시각을 넘지 않는다. 구조상 항상 참이다.
    ok = (avail_max == NO_DECISION) | (dec == NO_DECISION) | (avail_max <= dec)

    out: dict[str, pl.Series] = {
        "date": pl.Series("date", grid.dates).cast(pl.Date),
        "decision_at": grid.decision_at,
        "MRS": pl.Series("MRS", mrs, dtype=pl.Float64).fill_nan(None),
    }
    for s in cfg.SUB_SCORES:
        out[f"sub_{s}"] = pl.Series(f"sub_{s}", sub_vals[s], dtype=pl.Float64).fill_nan(None)
    for s in cfg.SUB_SCORES:
        out[f"n_components_{s}"] = pl.Series(f"n_components_{s}", sub_cnt[s], dtype=pl.Int64)
    out.update(cols)
    out["availability_ok"] = pl.Series("availability_ok", ok, dtype=pl.Boolean)
    out["input_available_at_max"] = _epoch_to_series("input_available_at_max", avail_max)

    first = pl.DataFrame(
        first_rows,
        schema={
            "component": pl.String,
            "sub": pl.String,
            "first_value_date": pl.Date,
            "first_pct_date": pl.Date,
        },
    )
    return ScoreResult(frame=pl.DataFrame(out), first_dates=first, diagnostics=diag)


def asset_sigma20(
    grid: Grid,
    price_rows: pl.DataFrame,
    *,
    window: int = cfg.RVOL_WINDOW,
    staleness_days: int = cfg.MACRO_STALENESS_DAYS,
) -> pl.DataFrame:
    """변동성 관리 기준선 입력 (사양 §3.1 마지막 항목, §6).

    ``sigma20(t)`` = t 시점 vintage 가격의 최근 ``window``개 로그 수익 표본표준편차,
    ``sigma_target(t)`` = t 시점 vintage ``σ_20`` 이력(t 포함)의 확장창 중앙값.
    **warm-up을 걸지 않는다** — 유효 ``σ_20`` 이력이 1개 이상이면 낸다 (MI08, 문면에 학습 없음).

    ``price_rows``: ``date, value, available_at`` (자산 가격 또는 총수익 지수).
    반환 열: ``date, sigma20, sigma_target, nvalid``.
    """
    sp = sigma20_spec(PRICE_KEY, window)
    res = vintage_component_result(
        grid,
        {PRICE_KEY: price_rows},
        sp.fn,
        lookback=sp.lookback,
        staleness_days=staleness_days,
        stat="median",
        warmup=1,
    )
    return res.frame.select(
        "date",
        pl.col("x").alias("sigma20"),
        pl.col("stat_value").alias("sigma_target"),
        "nvalid",
    )
