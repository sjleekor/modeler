"""ETF 품질 점수 E2 거래 품질 — ETF-연 표, 그룹-연도 스피어만 통계, 클러스터 부트스트랩, 기준선
(사전등록 20261010_quality_score §11.1~11.4, 구현 해석 표 v1 I02·I03·I06·I07·I10·I15·I16·I17·I19·I23).

    STOCK_DATA_ROOT=../stock_data PYTHONPATH=src python -m modeler.scores.quality.etf_e2 --period dev

경로는 환경변수로 받는다(기본 ``../stock_data``). 산출물은 ``stock_data/kr/output/quality_score_etf_dev_20261010/e2/``
에만 쓴다.

정의(사전등록 원문, 해석 표 번호는 상수 옆에):

- 형성 연도 Y의 성분 두 개와 Y+1의 실현 두 개를 ETF마다 한 행에 둔다(``etf_years``).
  괴리 = Y 거래일의 ``|종가 − NAV| ÷ NAV`` 일평균(I07: 종가·NAV가 있고 NAV > 0인 날, 거래량 0인 날 포함),
  거래대금 = Y 종가 있는 거래일 ``ACC_TRDVAL`` 중앙값(I17: 0 포함).
- 연간 통계의 조건(§11.3): Y·Y+1 모두 종가 있는 거래일 200일 이상, Y+1 마지막 시장 거래일까지 상장 유지,
  만기형 아님, 연금 부적격 후보 아님(I10), 비교 보류 아님(lenient 주 규칙). 그룹 크기는 이 조건을 통과한
  ETF로 센다(I06). 그룹 크기 5 이상인 그룹-연도만 주 계산에 쓴다(§11.1).
- 통계 = 그룹-연도 스피어만 순위 상관(동률 평균 순위)의 그룹 크기 가중평균(§11.3).
  괴리는 (괴리 성분 백분위 — 낮을수록 높음, −실현 괴리), 거래대금은 (거래대금 성분 백분위, 실현 거래대금).
- 클러스터 부트스트랩(I15), p값 식(I02), 기준선(I19, 기록용).

**판정 구간 보호.** 형성 Y ≥ 2014 의 성분–실현 값을 잇는 계산(``etf_years(with_values=True)``,
``e2_stats``, ``bootstrap``, ``e2_baseline``)은 환경변수 ``QUALITY_E_JUDGMENT_CONFIRMED`` 가 비어 있지 않을
때만 돈다. 그룹 크기 분포(입력 개수)는 값 없이 센다(``with_values=False``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.quality import etf_panel as ep

# ---------------------------------------------------------------- 상수(처음 고른 값, 사전등록·해석 표)
E2_VERSION = "quality-score-v0/etf_e2/1"

MIN_TRADE_DAYS = 200  # §11.3: Y·Y+1 거래일 200일 이상(처음 고른 값)
MIN_GROUP = 5  # §11.1: 그룹 크기 5 미만 제외(처음 고른 값). I06: 조건을 통과한 ETF로 센다
MIN_GROUP_RECORD = 3  # §6·§14 #25: 기록용 변형(크기 3 이상)
BOOT_N = 2000  # I15: ETF 복원 추출 2,000회
BOOT_SEED = 20261010  # I14·I15: 고정 시드
BOOT_MIN_ROWS = 3  # I15: 뽑힌 행 3개 미만인 그룹-연도 제외
ALPHAS = (0.025, 0.05)  # I18: Holm 단계 α (이 모듈은 분위수 함수만 낸다)
PCT_SINGLE = 50.0  # I03: n = 1이면 백분위 50
NORM_WARN_NOTE = "I02: p = (부트스트랩 ρ ≤ 0 개수 + 1) ÷ (B + 1)"

#: 구간(§7.1). 형성 연도 Y 범위. dev 실현 ≤ 2014, judgment 실현 2016~2025. 2026년은 결과에 안 쓴다.
PERIODS = {"dev": (2010, 2013), "judgment": (2015, 2024)}
ALL_YEARS = (2010, 2024)  # 그룹 크기 분포(입력 개수)를 세는 형성 연도
VALUE_GUARD_FROM_Y = 2014  # 이 연도 이상의 성분–실현 값 연결은 보호 변수가 있어야 한다
JUDGMENT_ENV = "QUALITY_E_JUDGMENT_CONFIRMED"

#: 사전등록 §6 재구현(2015~2024 합, 조건 적용 전, §11.1 실제 키·lenient): 크기 5 이상 ETF-연, 3 이상 ETF-연
SECTION6_REIMPL = {"ge5_etf_years_2015_2024": 495, "ge3_etf_years_2015_2024": 944}

DEFAULT_INPUT_REL = ep.DEFAULT_INPUT_REL
DEFAULT_OUT_REL = "kr/output/quality_score_etf_dev_20261010/e2"
DEFAULT_INTERP_TABLE = (
    "/private/tmp/claude-501/-Users-whishaw-wss-p-my/b1cc46eb-1957-4b1b-b4b5-6208b7ec737c/"
    "scratchpad/interp_table_v1.md"
)
DEFAULT_PREREG = (
    "/Users/whishaw/wss_p/my/milestones/common/scores/20261010_quality_score/01_preregistration.md"
)


# ---------------------------------------------------------------- 보호
def _judgment_confirmed() -> bool:
    return bool(os.environ.get(JUDGMENT_ENV, "").strip())


def _guard_period(period: str) -> tuple[int, int]:
    """period 이름을 형성 연도 범위로 바꾼다. judgment 는 보호 변수가 없으면 예외."""
    if period not in PERIODS:
        raise ValueError(f"period 는 {sorted(PERIODS)} 중 하나여야 한다: {period!r}")
    if period == "judgment" and not _judgment_confirmed():
        raise PermissionError(
            f"판정 구간은 환경변수 {JUDGMENT_ENV} 가 비어 있지 않을 때만 돌린다(사전등록 §12.4)."
        )
    return PERIODS[period]


def _guard_years(years: list[int]) -> None:
    if any(y >= VALUE_GUARD_FROM_Y for y in years) and not _judgment_confirmed():
        raise PermissionError(
            f"형성 Y ≥ {VALUE_GUARD_FROM_Y} 의 성분과 실현 값을 잇는 계산은 {JUDGMENT_ENV} 없이 못 한다."
        )


# ---------------------------------------------------------------- 백분위·순위·스피어만
def pct_rank(x: np.ndarray) -> np.ndarray:
    """I03: (평균 순위 − 1) ÷ (n − 1) × 100, 값이 클수록 높다. n = 1이면 50."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 1:
        return np.array([PCT_SINGLE])
    return (avg_rank(x) - 1.0) / (n - 1.0) * 100.0


def avg_rank(x: np.ndarray) -> np.ndarray:
    """동률은 평균 순위(1부터). 가장 작은 값이 1."""
    x = np.asarray(x, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return (cum - (cnt - 1) / 2.0)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """평균 순위의 피어슨 상관. 한쪽이 전부 같으면 nan."""
    rx, ry = avg_rank(x), avg_rank(y)
    sx, sy = rx.std(), ry.std()
    if len(rx) < 2 or sx == 0 or sy == 0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


# ---------------------------------------------------------------- ETF-연 표
def _year_stats(panel: ep.Panel) -> pl.DataFrame:
    """(ETF, 연도)마다 종가 있는 거래일 수, 거래대금 중앙값(I17), 괴리 일평균(I07)과 괴리 일수."""
    closes = panel.rows.filter(pl.col("TDD_CLSPRC").is_not_null()).with_columns(
        pl.col("date").dt.year().alias("year")
    )
    a = closes.group_by("ISU_CD", "year").agg(
        pl.len().alias("n_close"),
        pl.col("ACC_TRDVAL").median().alias("trdval"),
    )
    g = (
        ep.daily_gap(panel)
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("isu_cd", "year")
        .agg(pl.col("gap").mean().alias("gap"), pl.len().alias("n_gap"))
        .rename({"isu_cd": "ISU_CD"})
    )
    return a.join(g, on=["ISU_CD", "year"], how="left").rename({"ISU_CD": "isu_cd"})


def etf_years(
    panel: ep.Panel,
    life: pl.DataFrame,
    years: list[int] | None = None,
    with_values: bool = True,
) -> pl.DataFrame:
    """형성 연도 Y마다 ETF 한 행(그해 종가가 한 번이라도 있는 ETF).

    칸: ``year``, ``isu_cd``, ``group_key``, ``n_close_y``·``n_close_y1``(종가 거래일 수),
    ``gap_y``·``gap_y1``(괴리 일평균), ``trdval_y``·``trdval_y1``(거래대금 중앙값),
    조건 실패 칸 ``fail_*``, ``ok_lenient``(주 규칙)·``ok_strict``(PR/TR 표기 없으면 보류).

    ``with_values=False`` 이면 값 칸은 null이다(그룹 크기 분포용 — 입력 개수만 센다).
    ``with_values=True`` 이고 Y ≥ 2014 가 들어 있으면 보호 변수가 있어야 한다.
    Y+1 이 자료의 마지막 연도(끝나지 않은 해)면 그 Y는 만들지 않는다.
    """
    if years is None:
        years = list(range(ALL_YEARS[0], ALL_YEARS[1] + 1))
    if with_values:
        _guard_years(years)
    last_full_year = panel.end_date.year - 1  # 끝나지 않은 해는 실현 연도로 안 쓴다
    years = [y for y in years if y + 1 <= last_full_year]
    ys = _year_stats(panel)
    cal = panel.calendar.with_columns(pl.col("date").dt.year().alias("year"))
    year_last = cal.group_by("year").agg(pl.col("day_idx").max().alias("last_idx_year"))

    base = ys.filter(pl.col("year").is_in(years)).select(
        "year",
        "isu_cd",
        pl.col("n_close").alias("n_close_y"),
        pl.col("gap").alias("gap_y"),
        pl.col("trdval").alias("trdval_y"),
    )
    nxt = ys.select(
        (pl.col("year") - 1).alias("year"),
        "isu_cd",
        pl.col("n_close").alias("n_close_y1"),
        pl.col("gap").alias("gap_y1"),
        pl.col("trdval").alias("trdval_y1"),
    )
    lc = life.select(
        "isu_cd",
        "last_idx",
        "group_key",
        "active",
        pl.col("compare_hold").fill_null(True),
        pl.col("compare_hold_strict_rt").fill_null(True),
        pl.col("exclude_maturity").fill_null(False),
        pl.col("pension_ineligible_candidate").fill_null(False),
    )
    out = (
        base.join(nxt, on=["year", "isu_cd"], how="left")
        .join(year_last.with_columns((pl.col("year") - 1).alias("year")), on="year", how="left")
        .join(lc, on="isu_cd", how="left")
        .with_columns(pl.col("n_close_y1").fill_null(0))
        .with_columns(
            (pl.col("n_close_y") < MIN_TRADE_DAYS).alias("fail_days_y"),
            (pl.col("n_close_y1") < MIN_TRADE_DAYS).alias("fail_days_y1"),
            # I23·§11.3: Y+1 마지막 시장 거래일까지 목록에 있음(마지막 거래일 ≥ Y+1의 마지막 거래일)
            (pl.col("last_idx") < pl.col("last_idx_year")).alias("fail_delisted_y1"),
            pl.col("exclude_maturity").alias("fail_maturity"),
            pl.col("pension_ineligible_candidate").alias("fail_pension"),  # I10
            pl.col("compare_hold").alias("fail_hold"),
            pl.col("compare_hold_strict_rt").alias("fail_hold_strict"),
        )
        .with_columns(
            (
                ~(
                    pl.col("fail_days_y")
                    | pl.col("fail_days_y1")
                    | pl.col("fail_delisted_y1")
                    | pl.col("fail_maturity")
                    | pl.col("fail_pension")
                    | pl.col("fail_hold")
                )
            ).alias("ok_lenient"),
        )
        .with_columns(
            (pl.col("ok_lenient") & ~pl.col("fail_hold_strict")).alias("ok_strict"),
        )
        .drop("last_idx_year")
    )
    if not with_values:
        out = out.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(c) for c in ("gap_y", "gap_y1", "trdval_y", "trdval_y1")]
        )
    return out.sort(["year", "isu_cd"])


def exclusion_counts(ey: pl.DataFrame) -> pl.DataFrame:
    """조건별 빠진 수. 연도마다 (a) 조건 하나씩 따로(겹침 포함), (b) 정해진 순서로 순차.

    순차 순서: 만기형 → 연금 부적격 → 비교 보류 → Y 200일 → Y+1 폐지 → Y+1 200일.
    ``n_removed_delisted_y1`` 이 "Y+1 중 폐지돼 빠진 수"(앞 조건을 다 통과한 ETF 중)다.
    """
    rows = []
    for y, g in ey.group_by("year", maintain_order=True):
        y = y[0]
        remain = g
        rec = {"year": y, "n_etf_with_close_in_y": g.height}
        for name in ("maturity", "pension", "hold", "days_y", "delisted_y1", "days_y1"):
            col = f"fail_{name}"
            rec[f"alone_{name}"] = int(g[col].sum())
            hit = remain.filter(pl.col(col))
            rec[f"n_removed_{name}"] = hit.height
            remain = remain.filter(~pl.col(col))
        rec["n_pass_lenient"] = remain.height
        rec["n_pass_strict"] = int(g["ok_strict"].sum())
        assert rec["n_pass_lenient"] == int(g["ok_lenient"].sum())
        rows.append(rec)
    return pl.DataFrame(rows).sort("year")


# ---------------------------------------------------------------- 그룹-연도 구성
def _value_cols(value: str) -> tuple[str, str]:
    if value == "gap":
        return "gap_y", "gap_y1"
    if value == "trdval":
        return "trdval_y", "trdval_y1"
    raise ValueError(f"value 는 gap|trdval: {value!r}")


def _groups(
    ey: pl.DataFrame, period: str, min_group: int, value: str, key: str
) -> tuple[list[dict], dict]:
    """주 계산의 그룹-연도 목록. 각 원소: year, group_key, isu(ndarray), x(성분 백분위), y(실현, 방향 맞춤).

    표본 = period 연도 범위 ∧ 통과 칸(lenient/strict) ∧ 성분·실현 값이 있는 ETF.
    크기 판정(min_group)은 이 표본의 그룹별 행 수다(값이 비어 빠진 ETF는 따로 센다).
    """
    lo, hi = _guard_period(period)
    if key not in ("lenient", "strict"):
        raise ValueError(f"key 는 lenient|strict: {key!r}")
    xc, yc = _value_cols(value)
    ok = "ok_lenient" if key == "lenient" else "ok_strict"
    sub = ey.filter(pl.col("year").is_between(lo, hi) & pl.col(ok))
    n_pass = sub.height
    sub = sub.filter(pl.col(xc).is_not_null() & pl.col(yc).is_not_null())
    info = {"n_pass_conditions": n_pass, "n_dropped_null_value": n_pass - sub.height}
    out = []
    for (y, gk), g in sub.group_by(["year", "group_key"], maintain_order=True):
        if g.height < min_group:
            continue
        g = g.sort("isu_cd")
        x = g[xc].to_numpy().astype(float)
        yv = g[yc].to_numpy().astype(float)
        if value == "gap":
            xs, ys_ = pct_rank(-x), -yv  # 낮은 괴리 = 높은 백분위, 실현은 −괴리
        else:
            xs, ys_ = pct_rank(x), yv
        out.append(
            {
                "year": int(y),
                "group_key": gk,
                "isu": g["isu_cd"].to_list(),
                "x": xs,
                "y": ys_,
                "gap_y": g["gap_y"].to_numpy().astype(float),
                "gap_y1": g["gap_y1"].to_numpy().astype(float),
                "active": g["active"].to_list(),
            }
        )
    return out, info


def e2_stats(
    ey: pl.DataFrame,
    period: str,
    min_group: int = MIN_GROUP,
    value: str = "gap",
    key: str = "lenient",
) -> dict:
    """그룹-연도 스피어만과 가중평균. ``period="judgment"`` 는 보호 변수가 필요하다.

    반환: ``group_years``(year, group_key, size, rho), ``weighted_rho``(가중치 = 그룹 크기,
    ρ가 정의된 그룹-연도만), ``pooled_etf_years``(G1: 크기 min_group 이상 그룹의 ETF-연 합),
    ``by_year``(연도별 가중평균 ρ), ``n_years``·``n_pos_years``(I16: 그룹이 있는 해만).
    """
    gs, info = _groups(ey, period, min_group, value, key)
    rows = []
    for g in gs:
        rows.append(
            {
                "year": g["year"],
                "group_key": g["group_key"],
                "size": len(g["isu"]),
                "rho": spearman(g["x"], g["y"]),
            }
        )
    gy = pl.DataFrame(
        rows,
        schema={"year": pl.Int64, "group_key": pl.String, "size": pl.Int64, "rho": pl.Float64},
    )
    valid = gy.filter(pl.col("rho").is_not_null() & pl.col("rho").is_not_nan())
    wr = _wmean(valid["rho"].to_numpy(), valid["size"].to_numpy())
    by_year = (
        valid.group_by("year")
        .agg(
            (pl.col("rho") * pl.col("size")).sum().alias("_num"),
            pl.col("size").sum().alias("etf_years"),
            pl.len().alias("n_groups"),
        )
        .with_columns((pl.col("_num") / pl.col("etf_years")).alias("rho_w"))
        .drop("_num")
        .sort("year")
    )
    # I16: 크기 min_group 이상 그룹이 하나 이상인 해만(ρ가 정의된 그룹이 있는 해)
    return {
        "period": period,
        "value": value,
        "key": key,
        "min_group": min_group,
        "n_group_years": gy.height,
        "n_group_years_valid_rho": valid.height,
        "n_group_years_constant": gy.height - valid.height,
        "pooled_etf_years": int(gy["size"].sum()) if gy.height else 0,
        "weighted_rho": wr,
        "n_years": by_year.height,
        "n_pos_years": int((by_year["rho_w"] > 0).sum()) if by_year.height else 0,
        "by_year": by_year,
        "group_years": gy.sort(["year", "group_key"]),
        **info,
    }


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(v * w) / np.sum(w)) if len(v) and np.sum(w) > 0 else float("nan")


# ---------------------------------------------------------------- 부트스트랩(I15)
def bootstrap(
    ey: pl.DataFrame,
    period: str,
    min_group: int = MIN_GROUP,
    value: str = "gap",
    key: str = "lenient",
    n_boot: int = BOOT_N,
    seed: int = BOOT_SEED,
) -> np.ndarray:
    """ETF 복원 추출 클러스터 부트스트랩. 가중평균 ρ 배열(길이 n_boot)을 돌려준다.

    ETF를 뽑으면 그 ETF의 모든 연도 행이 같은 횟수로 따라온다. 그룹-연도 구성은 원래대로 두고(원래
    크기 min_group 이상인 것만), 뽑힌 행(중복 포함)으로 스피어만을 구한다. 뽑힌 행이 3개 미만이거나
    한쪽 값이 다 같으면 그 그룹-연도는 그 회차에서 뺀다. 가중치 = 뽑힌 행 수. 전부 빠지면 nan.
    """
    gs, _ = _groups(ey, period, min_group, value, key)
    etfs = sorted({i for g in gs for i in g["isu"]})
    n_e = len(etfs)
    out = np.full(n_boot, np.nan)
    if n_e == 0:
        return out
    pos = {e: k for k, e in enumerate(etfs)}
    rng = np.random.default_rng(seed)
    # 회차 × ETF 뽑힌 횟수(복원 추출 n_e개 = 다항분포)
    counts = rng.multinomial(n_e, np.full(n_e, 1.0 / n_e), size=n_boot).astype(float)
    num = np.zeros(n_boot)
    den = np.zeros(n_boot)
    for g in gs:
        m = counts[:, [pos[i] for i in g["isu"]]]  # B × n
        rho, w = _weighted_spearman_batch(g["x"], g["y"], m)
        ok = np.isfinite(rho) & (w >= BOOT_MIN_ROWS)
        num += np.where(ok, rho * w, 0.0)
        den += np.where(ok, w, 0.0)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _weighted_spearman_batch(x: np.ndarray, y: np.ndarray, m: np.ndarray):
    """행 i가 m[b, i]번 들어간 다중집합의 스피어만(평균 순위 피어슨), 회차마다. (ρ[B], 행 수[B])."""
    w = m.sum(axis=1)

    def mranks(v):
        less = (v[None, :] < v[:, None]).astype(float)  # [i, j] = v_j < v_i
        eq = (v[None, :] == v[:, None]).astype(float)
        return m @ less.T + (m @ eq.T + 1.0) / 2.0  # B × n 평균 순위(다중집합)

    rx, ry = mranks(x), mranks(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        wsum = np.where(w > 0, w, np.nan)
        mx = (m * rx).sum(axis=1) / wsum
        my = (m * ry).sum(axis=1) / wsum
        dx, dy = rx - mx[:, None], ry - my[:, None]
        vx = (m * dx * dx).sum(axis=1)
        vy = (m * dy * dy).sum(axis=1)
        cov = (m * dx * dy).sum(axis=1)
        rho = cov / np.sqrt(vx * vy)
    # 한쪽 값이 다 같으면 분산 0 → nan
    rho = np.where((vx > 1e-12) & (vy > 1e-12), rho, np.nan)
    return rho, w


def p_value(boot: np.ndarray) -> float:
    """I02: (부트스트랩 ρ ≤ 0 개수 + 1) ÷ (B + 1). nan 회차는 B에서 뺀다."""
    b = boot[np.isfinite(boot)]
    return float((np.sum(b <= 0) + 1) / (len(b) + 1))


def lower_bound(boot: np.ndarray, alpha: float) -> float:
    """단측 하한 = 부트스트랩 분포의 α 분위수(Holm 단계 α는 호출하는 쪽이 정한다)."""
    return float(np.quantile(boot[np.isfinite(boot)], alpha))


# ---------------------------------------------------------------- 기준선(I19, 기록용)
def e2_baseline(
    ey: pl.DataFrame,
    netassets: pl.DataFrame,
    period: str,
    min_group: int = MIN_GROUP,
    key: str = "lenient",
) -> dict:
    """그룹-연도마다 (Y 마지막 월말 순자산 1위 패시브 ETF)와 (괴리 성분 1위 ETF)의 Y+1 실현 괴리.

    ``netassets`` 는 ``etf_panel.month_end_netassets`` 결과. 1위 = 최대 순자산(값 있는 ETF 중),
    점수 1위 = Y 괴리 최저(동률이면 코드 순). 점수 1위가 더 낮은 비율은 엄격히 낮은 경우만 센다
    (같으면 ``n_tie``). 패시브 ETF가 그룹에 없거나 순자산이 없으면 그 그룹-연도는 빠진다.
    """
    gs, _ = _groups(ey, period, min_group, "gap", key)
    last_me = (
        netassets.with_columns(pl.col("month_end").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("month_end").max().alias("me"))
    )
    me_of = dict(zip(last_me["year"].to_list(), last_me["me"].to_list()))
    na = {
        (r["isu_cd"], r["month_end"]): r["netasst"]
        for r in netassets.filter(pl.col("netasst").is_not_null()).iter_rows(named=True)
    }
    rows = []
    for g in gs:
        me = me_of.get(g["year"])
        cands = [
            (na[(i, me)], k)
            for k, i in enumerate(g["isu"])
            if not g["active"][k] and (i, me) in na
        ]
        if not cands:
            rows.append({"year": g["year"], "group_key": g["group_key"], "size": len(g["isu"])})
            continue
        kp = max(cands, key=lambda t: (t[0], -t[1]))[1]
        ks = int(np.argmin(g["gap_y"]))  # 동률이면 첫째(코드 순)
        rows.append(
            {
                "year": g["year"],
                "group_key": g["group_key"],
                "size": len(g["isu"]),
                "passive_top": g["isu"][kp],
                "score_top": g["isu"][ks],
                "passive_gap_y1": float(g["gap_y1"][kp]),
                "score_gap_y1": float(g["gap_y1"][ks]),
                "same_etf": kp == ks,
            }
        )
    df = pl.DataFrame(rows, infer_schema_length=None)
    if "passive_top" not in df.columns:
        return {"group_years": df, "n_group_years": df.height, "n_compared": 0}
    cmp = df.filter(pl.col("passive_top").is_not_null())
    n = cmp.height
    lower = int((cmp["score_gap_y1"] < cmp["passive_gap_y1"]).sum()) if n else 0
    tie = int((cmp["score_gap_y1"] == cmp["passive_gap_y1"]).sum()) if n else 0
    return {
        "group_years": df,
        "n_group_years": df.height,
        "n_compared": n,
        "n_no_passive": df.height - n,
        "mean_passive_top_gap_y1": float(cmp["passive_gap_y1"].mean()) if n else None,
        "mean_score_top_gap_y1": float(cmp["score_gap_y1"].mean()) if n else None,
        "mean_all": (
            float((cmp["passive_gap_y1"].mean() + cmp["score_gap_y1"].mean()) / 2) if n else None
        ),
        "n_score_lower": lower,
        "n_tie": tie,
        "n_same_etf": int(cmp["same_etf"].sum()) if n else 0,
        "share_score_lower": lower / n if n else None,
    }


# ---------------------------------------------------------------- 그룹 크기 분포(입력 개수, 전 연도)
def group_size_distribution(ey_nov: pl.DataFrame) -> pl.DataFrame:
    """조건 적용 뒤 그룹 크기 분포(값 없이 센 입력 개수). 연도마다 lenient·strict, 크기 5 이상·3 이상.

    ``ey_nov`` 는 ``etf_years(with_values=False)`` 결과다.
    """
    rows = []
    for y, g in ey_nov.group_by("year", maintain_order=True):
        rec = {"year": y[0], "n_etf_with_close": g.height}
        for key, ok in (("lenient", "ok_lenient"), ("strict", "ok_strict")):
            sizes = g.filter(pl.col(ok)).group_by("group_key").len()["len"].to_numpy()
            rec[f"{key}_n_pass"] = int(sizes.sum())
            rec[f"{key}_ge5_groups"] = int((sizes >= 5).sum())
            rec[f"{key}_ge5_etf_years"] = int(sizes[sizes >= 5].sum())
            rec[f"{key}_ge3_groups"] = int((sizes >= 3).sum())
            rec[f"{key}_ge3_etf_years"] = int(sizes[sizes >= 3].sum())
        rows.append(rec)
    return pl.DataFrame(rows).sort("year")


def group_size_before_conditions(panel: ep.Panel, life: pl.DataFrame) -> pl.DataFrame:
    """§6 재구현과 같은 읽기: 그해 첫 거래일 종가가 있는 ETF, 만기형·비교 보류 제외, 조건(200일 등) 적용 전.

    연금 부적격은 빼지 않는다(§6 재구현 문면에는 없다). 조건 적용 뒤 표와 나란히 두려는 기록이다.
    """
    cal = panel.calendar.with_columns(pl.col("date").dt.year().alias("year"))
    first_day = cal.group_by("year").agg(pl.col("day_idx").min().alias("fi"))
    have = panel.rows.filter(pl.col("TDD_CLSPRC").is_not_null()).select(
        pl.col("ISU_CD").alias("isu_cd"), "day_idx"
    )
    lc = life.select("isu_cd", "group_key", "compare_hold", "exclude_maturity")
    j = (
        first_day.join(have, left_on="fi", right_on="day_idx")
        .join(lc, on="isu_cd")
        .filter(~pl.col("compare_hold") & ~pl.col("exclude_maturity"))
    )
    rows = []
    for y, g in j.group_by("year", maintain_order=True):
        sizes = g.group_by("group_key").len()["len"].to_numpy()
        rows.append(
            {
                "year": y[0],
                "pre_ge5_groups": int((sizes >= 5).sum()),
                "pre_ge5_etf_years": int(sizes[sizes >= 5].sum()),
                "pre_ge3_groups": int((sizes >= 3).sum()),
                "pre_ge3_etf_years": int(sizes[sizes >= 3].sum()),
            }
        )
    return pl.DataFrame(rows).sort("year")


# ---------------------------------------------------------------- 실행
def _sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.check_output(["git", "-C", str(here), *args], text=True).strip()
    except Exception:
        return ""


def _nan_to_none(o):
    """JSON에 NaN을 쓰지 않는다(nan → null)."""
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return o


def _stats_summary(st: dict) -> dict:
    return {
        k: st[k]
        for k in (
            "period", "value", "key", "min_group", "n_group_years", "n_group_years_valid_rho",
            "n_group_years_constant", "pooled_etf_years", "weighted_rho", "n_years", "n_pos_years",
            "n_pass_conditions", "n_dropped_null_value",
        )
    } | {"by_year": st["by_year"].to_dicts()}


def run_dev(a: argparse.Namespace) -> dict:
    period = "dev"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = ep.read_panel(a.input)
    life = ep.lifecycle(panel)
    lo, hi = PERIODS[period]
    dev_years = list(range(lo, hi + 1))

    ey = etf_years(panel, life, dev_years, with_values=True)
    ey_nov = etf_years(panel, life, list(range(ALL_YEARS[0], ALL_YEARS[1] + 1)), with_values=False)
    excl = exclusion_counts(ey)

    variants = {
        "main": dict(min_group=MIN_GROUP, value="gap", key="lenient"),
        "trdval": dict(min_group=MIN_GROUP, value="trdval", key="lenient"),
        "min3": dict(min_group=MIN_GROUP_RECORD, value="gap", key="lenient"),
        "strict": dict(min_group=MIN_GROUP, value="gap", key="strict"),
    }
    stats = {n: e2_stats(ey, period, **kw) for n, kw in variants.items()}
    gy_rows = []
    for n, st in stats.items():
        gy_rows.append(st["group_years"].with_columns(pl.lit(n).alias("variant")))
    gy = pl.concat(gy_rows).select("variant", "year", "group_key", "size", "rho")

    boot = bootstrap(ey, period, **variants["main"])
    boots = {}
    for n in ("trdval", "min3", "strict"):
        boots[n] = bootstrap(ey, period, **variants[n], n_boot=BOOT_N)
    finite = boot[np.isfinite(boot)]

    def bsum(b):
        f = b[np.isfinite(b)]
        if len(f) == 0:
            return None
        return {
            "n_valid": int(len(f)),
            "mean": float(f.mean()),
            "q025": lower_bound(b, 0.025),
            "q05": lower_bound(b, 0.05),
            "q50": lower_bound(b, 0.5),
            "q95": lower_bound(b, 0.95),
            "q975": lower_bound(b, 0.975),
            "p_value": p_value(b),
        }

    netassets = ep.month_end_netassets(
        panel, life
    ).filter(pl.col("month_end").dt.year().is_between(lo, hi))
    base = e2_baseline(ey, netassets, period, **{k: variants["main"][k] for k in ("min_group", "key")})
    base_gy = base.pop("group_years")

    dist = group_size_distribution(ey_nov)
    pre = group_size_before_conditions(panel, life)
    dist = dist.join(pre, on="year", how="left")
    d1524 = dist.filter(pl.col("year").is_between(2015, 2024))
    dist_check = {
        "after_conditions_2015_2024_lenient_ge5_etf_years": int(d1524["lenient_ge5_etf_years"].sum()),
        "after_conditions_2015_2024_lenient_ge3_etf_years": int(d1524["lenient_ge3_etf_years"].sum()),
        "before_conditions_2015_2024_ge5_etf_years": int(d1524["pre_ge5_etf_years"].sum()),
        "before_conditions_2015_2024_ge3_etf_years": int(d1524["pre_ge3_etf_years"].sum()),
        "section6_reimpl": SECTION6_REIMPL,
    }

    gk = ey.filter(pl.col("year").is_between(lo, hi))
    ey_out = gk.select(
        "year", "isu_cd", "group_key", "active", "n_close_y", "n_close_y1", "gap_y", "gap_y1",
        "trdval_y", "trdval_y1", "fail_maturity", "fail_pension", "fail_hold", "fail_hold_strict",
        "fail_days_y", "fail_delisted_y1", "fail_days_y1", "ok_lenient", "ok_strict",
    )
    ey_out.write_csv(out / "etf_years_dev.csv")
    gy.write_csv(out / "group_years_dev.csv")
    excl.write_csv(out / "exclusions_dev.csv")
    dist.write_csv(out / "group_size_distribution_all_years.csv")
    base_gy.write_csv(out / "baseline_group_years_dev.csv")

    summary = {
        "version": E2_VERSION,
        "period": period,
        "formation_years": [lo, hi],
        "note_p": NORM_WARN_NOTE,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "min_rows": BOOT_MIN_ROWS},
        "main": {**_stats_summary(stats["main"]), "bootstrap": bsum(boot)},
        "variants": {
            n: {**_stats_summary(stats[n]), "bootstrap": bsum(boots[n])}
            for n in ("trdval", "min3", "strict")
        },
        "baseline_I19": base,
        "exclusions_dev_total": {
            c: int(excl[c].sum()) for c in excl.columns if c != "year"
        },
        "group_size_check": dist_check,
        "n_bootstrap_nan": int(np.sum(~np.isfinite(boot))),
    }
    (out / "summary.json").write_text(json.dumps(_nan_to_none(summary), ensure_ascii=False, indent=2, default=str))

    src = Path(__file__)
    status = _git("status", "--porcelain")
    inputs = {"input_csv": a.input, "interp_table": a.interp_table, "preregistration": a.prereg}
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "quality-score v0 E2 거래 품질, 개발 구간(형성 Y 2010~2013)만. 판정 구간 값 연결 없음.",
        "inputs": {k: {"path": v, "sha256": _sha256(v)} for k, v in inputs.items() if Path(v).exists()},
        "module": {"version": E2_VERSION, "file": str(src), "sha256": _sha256(src)},
        "etf_panel": {"file": str(Path(ep.__file__)), "sha256": _sha256(ep.__file__)},
        "seed": BOOT_SEED,
        "n_boot": BOOT_N,
        "code": {
            "repo": "modeler",
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": status.splitlines(),
            "worktree_dirty": bool(status),
        },
        "runtime": {"python": sys.version.split()[0], "polars": pl.__version__, "numpy": np.__version__},
        "outputs": sorted(p.name for p in out.glob("*") if p.name != "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"summary": summary, "exclusions": excl, "dist": dist, "gy": gy}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--input", default=str(root / DEFAULT_INPUT_REL))
    ap.add_argument("--out-dir", default=str(root / DEFAULT_OUT_REL))
    ap.add_argument("--interp-table", default=DEFAULT_INTERP_TABLE)
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    ap.add_argument("--period", choices=["dev"], required=True, help="이 CLI는 개발 구간만 돈다")
    a = ap.parse_args(argv)
    res = run_dev(a)
    print(json.dumps(_nan_to_none(res["summary"]), ensure_ascii=False, indent=1, default=str)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
