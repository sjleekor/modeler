"""ETF E3 — 보수 선택 규칙 (사양 §5).

- 형성 보수 ``F_i(Y)``: T = Y의 마지막 거래일. 기준일 ≤ T − ``LAG_DAYS``(92)일인 월말 파일 가운데 **가장 늦은 것**의
  값. 그 파일에 그 펀드 행이 없으면 없음(더 이른 파일로 물러서지 않는다). 12월 말 형성이면 8월 말 값, 실효 지연 119~121일.
- 결과 연도 보수 ``A_i(Y+1)``: Y+1의 1~12월 월말 파일 값의 평균(있는 달만, ``A_MIN_MONTHS``(10)개 이상 필요).
- 값 칸은 ``ter``(기본)·``sum_fee``·``ter+brokerage`` 중 하나.

**있는지만 보는 함수**(``formation_available``·``outcome_months``)와 **값을 읽는 함수**(``formation_value``·
``outcome_value``)를 나눴다. 동결 전 입력 존재 개수는 앞의 것만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

LAG_DAYS = 92  # 처음 고른 값(사양 §5)
A_MIN_MONTHS = 10
FIELDS = ("ter", "sum_fee", "ter+brokerage")


@dataclass
class FeeArrays:
    months: np.ndarray  # datetime64[D] 월말 기준일(오름차순)
    funds: list[str]
    fund_col: dict[str, int]
    vals: dict[str, np.ndarray]  # field -> (M, F) float, 없으면 NaN
    operator: dict[str, str]  # 표준코드 -> 운용사(마지막 값)


def build_fee_arrays(df: pl.DataFrame) -> FeeArrays:
    """``etf_e3_inputs.load_dis_fees`` 결과 → 월 × 펀드 행렬."""
    months = np.array(sorted(df["base_date"].unique().to_list()), dtype="datetime64[D]")
    funds = sorted(df["fund_code"].unique().to_list())
    fc = {f: i for i, f in enumerate(funds)}
    mi = {m: i for i, m in enumerate(months)}
    m_idx = np.array([mi[np.datetime64(d)] for d in df["base_date"].to_list()])
    f_idx = np.array([fc[c] for c in df["fund_code"].to_list()])
    vals = {}
    for name, col in (("ter", "ter"), ("sum_fee", "sum_fee"), ("brokerage", "brokerage")):
        m = np.full((len(months), len(funds)), np.nan)
        m[m_idx, f_idx] = df[col].to_numpy().astype(float)
        vals[name] = m
    vals["ter+brokerage"] = vals["ter"] + vals["brokerage"]
    op = {}
    for c, o in zip(df["fund_code"].to_list(), df["operator"].to_list()):
        if o:
            op[c] = o
    return FeeArrays(months, funds, fc, vals, op)


def _field(fa: FeeArrays, field: str) -> np.ndarray:
    if field not in fa.vals:
        raise ValueError(f"field 는 {FIELDS} 중 하나: {field!r}")
    return fa.vals[field]


def formation_month_index(
    fa: FeeArrays, t_last: np.datetime64, lag_days: int = LAG_DAYS, aug_fixed: bool = False
) -> int:
    """기준일 ≤ T − lag 인 월말 파일 가운데 가장 늦은 것의 번호. 없으면 −1.

    ``aug_fixed`` 이면 규칙 대신 T가 속한 해의 8월 월말 파일 하나(기록용 변형, 지연 119일 이상으로 통일).
    """
    if aug_fixed:
        y = int(t_last.astype("datetime64[Y]").astype(int)) + 1970
        ym = fa.months.astype("datetime64[M]")
        hit = np.flatnonzero(ym == np.datetime64(f"{y}-08"))
        return int(hit[0]) if len(hit) else -1
    lim = t_last - np.timedelta64(lag_days, "D")
    k = int(np.searchsorted(fa.months, lim, side="right")) - 1
    return k


def formation_available(
    fa: FeeArrays,
    fund: str | None,
    t_last: np.datetime64,
    lag_days: int = LAG_DAYS,
    field: str = "ter",
    aug_fixed: bool = False,
) -> bool:
    """F가 있는가(값은 읽지 않고 유무만)."""
    if fund is None or fund not in fa.fund_col:
        return False
    k = formation_month_index(fa, t_last, lag_days, aug_fixed)
    if k < 0:
        return False
    return bool(np.isfinite(_field(fa, field)[k, fa.fund_col[fund]]))


def formation_value(
    fa: FeeArrays,
    fund: str | None,
    t_last: np.datetime64,
    lag_days: int = LAG_DAYS,
    field: str = "ter",
    aug_fixed: bool = False,
) -> float:
    """F 값(%). 없으면 NaN. 판정 실행에서만 쓴다."""
    if fund is None or fund not in fa.fund_col:
        return float("nan")
    k = formation_month_index(fa, t_last, lag_days, aug_fixed)
    if k < 0:
        return float("nan")
    return float(_field(fa, field)[k, fa.fund_col[fund]])


def _outcome_cols(fa: FeeArrays, y1: int) -> np.ndarray:
    yrs = fa.months.astype("datetime64[Y]").astype(int) + 1970
    return np.flatnonzero(yrs == y1)


def outcome_months(fa: FeeArrays, fund: str | None, y1: int, field: str = "ter") -> int:
    """Y+1 의 월말 파일 중 그 펀드 값이 있는 달 수(값은 읽지 않는다)."""
    if fund is None or fund not in fa.fund_col:
        return 0
    rows = _outcome_cols(fa, y1)
    return int(np.isfinite(_field(fa, field)[rows, fa.fund_col[fund]]).sum())


def outcome_value(
    fa: FeeArrays, fund: str | None, y1: int, field: str = "ter", min_months: int = A_MIN_MONTHS
) -> float:
    """A 값(%): 있는 달 평균, ``min_months`` 개 미만이면 NaN. 판정 실행에서만 쓴다."""
    if fund is None or fund not in fa.fund_col:
        return float("nan")
    rows = _outcome_cols(fa, y1)
    v = _field(fa, field)[rows, fa.fund_col[fund]]
    v = v[np.isfinite(v)]
    if len(v) < min_months:
        return float("nan")
    return float(v.mean())


def outcome_range(
    fa: FeeArrays, fund: str | None, y1: int, field: str = "ter"
) -> tuple[float, float]:
    """Y+1 월말 값의 (최소, 최대) — 기록용(보수 변경 폭). 판정 실행에서만 쓴다."""
    if fund is None or fund not in fa.fund_col:
        return float("nan"), float("nan")
    v = _field(fa, field)[_outcome_cols(fa, y1), fa.fund_col[fund]]
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.min()), float(v.max())


# ---------------------------------------------------------------- 지연 판독
LAG_MIN_OBS = 6


def lag_summary(measured_lags: list[int] | None, censored_over: int | None = None) -> dict:
    """실측 지연(일) 목록 → 최소·중앙·최대와 상태.

    6개 미만이면 ``status = "잠정(지연 실측 전)"``. **관측 수와 상관없이** 실측 지연 하나라도 ``LAG_DAYS`` 를 넘거나,
    아직 안 나온 기준일의 경과일(``censored_over``)이 ``LAG_DAYS`` 를 넘으면 ``cap = True``
    (E3b′ 등급 상한 C, "지연 규칙 부족"). 메인 정정(10-11): 처음 판은 6개 이상일 때만 상한을 걸어,
    한 달이라도 92일을 넘긴 사실을 판정이 놓칠 수 있었다.
    """
    lags = sorted(int(x) for x in (measured_lags or []))
    n = len(lags)
    out = {
        "n": n,
        "min": None,
        "median": None,
        "max": None,
        "censored_over": censored_over,
        "status": "잠정(지연 실측 전)",
        "cap": False,
    }
    if n:
        out.update({"min": lags[0], "median": float(np.median(lags)), "max": lags[-1]})
    if n >= LAG_MIN_OBS:
        out["status"] = "실측"
    over = (n and lags[-1] > LAG_DAYS) or (censored_over is not None and censored_over > LAG_DAYS)
    out["cap"] = bool(over)
    return out
