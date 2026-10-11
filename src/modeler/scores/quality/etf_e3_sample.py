"""ETF E3 — 표본과 ETF-연 단위 제외 (사양 §2). 값(수익률·보수)은 읽지 않고 **있는지·개수만** 다룬다.

순서: E2 조건(동결 파서 키·만기형·연금 부적격·비교 보류·200일·폐지) → 그룹-연도 크기 ≥ 2 →
(1) 기초지수 변경 (2) 분할·병합 (3) NAV 결측 (4) 이상치 (5) 분배락일 정합성 < 90% (6) DIS 대응 높음·확인 아님
(7) 보수: F 없음 또는 결과 연도 월말 TER 10개월 미만. 제외 단계마다 연도별 수를 센다.
마지막에 남은 ETF-연이 2개 미만인 그룹-연도를 뺀다.

처음 고른 값: 10일, 30%, 1.5, [0.9, 1.1], 90%, 10/12 (사양 §2).
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.quality import etf_e2 as e2
from modeler.scores.quality import etf_e3_exdate as ex
from modeler.scores.quality import etf_e3_fees as fees
from modeler.scores.quality import etf_e3_inputs as inp
from modeler.scores.quality import etf_panel as ep
from modeler.scores.quality.etf_e3_dense import Dense, build_dense

NAV_MISS_MAX = 10
OUTLIER = 0.30
SPLIT_R = 1.5
SPLIT_BAND = (0.9, 1.1)
MIN_GROUP = 2
STEPS = (
    "index_change",  # 1
    "split_merge",  # 2
    "nav_missing",  # 3
    "outlier",  # 4
    "exdate_consistency",  # 5
    "dis_map",  # 6
    "fee",  # 7
)
JUDGMENT_YEARS = tuple(range(2015, 2025))
DEV_YEARS = tuple(range(2011, 2015))


@dataclass
class Ctx:
    panel: ep.Panel
    dense: Dense
    life: pl.DataFrame
    dist: pl.DataFrame  # 분배(청산 제외, 금액 > 0): isu_cd, rgt_std_dt, amount, j, ri, match
    D: np.ndarray  # NAV 분배락일 주당 분배금 (T, N)
    rfin: np.ndarray  # bool (T, N): 그날 총수익 r_t 가 정의됨(NAV_t, NAV_{t-1} 둘 다 있음)
    exmask: np.ndarray  # bool (T, N): NAV 분배락일
    code_map: dict[str, str]  # 단축코드 -> 표준코드
    code_level: dict[str, str]  # 단축코드 -> high|manual
    fee: fees.FeeArrays
    input_sha: dict = field(default_factory=dict)
    nav_shift: int = ex.FROZEN_NAV_SHIFT
    rule: str = ex.FROZEN_RULE
    extra: dict = field(default_factory=dict)
    last_idx: np.ndarray | None = None  # ETF 열별 마지막 거래일 번호(상장 유지 판단)


def return_defined(d: Dense) -> np.ndarray:
    r = np.zeros_like(d.has_row)
    ok = np.isfinite(d.nav[1:]) & np.isfinite(d.nav[:-1]) & (d.nav[:-1] > 0)
    r[1:] = ok
    return r


def make_ctx(
    panel: ep.Panel,
    dist_use,
    code_map: pl.DataFrame,
    fee_df: pl.DataFrame,
    rule: str = ex.FROZEN_RULE,
    nav_shift: int = ex.FROZEN_NAV_SHIFT,
    input_sha: dict | None = None,
) -> Ctx:
    """읽어 둔 입력으로 ``Ctx`` 를 만든다(테스트는 합성 입력으로 이 함수를 부른다).

    ``dist_use`` 는 청산분배 제외·금액 > 0 인 분배 표, ``code_map`` 은 ``load_code_map`` 결과.
    """
    life = ep.lifecycle(panel)
    d = build_dense(panel)
    dist = ex.dist_rule_idx(d, dist_use, rule)
    D, _ = ex.exdate_matrix(d, dist, nav_shift)
    return Ctx(
        panel=panel,
        dense=d,
        life=life,
        dist=dist,
        D=D,
        rfin=return_defined(d),
        exmask=D > 0,
        code_map=dict(zip(code_map["short_code"].to_list(), code_map["fund_code"].to_list())),
        code_level=dict(zip(code_map["short_code"].to_list(), code_map["level"].to_list())),
        fee=fees.build_fee_arrays(fee_df),
        input_sha=input_sha or {},
        nav_shift=nav_shift,
        rule=rule,
        last_idx=_last_idx(life, d),
    )


def _last_idx(life: pl.DataFrame, d: Dense) -> np.ndarray:
    out = np.full(len(d.isu), -1)
    for c, f in zip(life["isu_cd"].to_list(), life["last_idx"].to_list()):
        out[d.col[c]] = f
    return out


def load_ctx(
    root: Path | None = None,
    verify: bool = True,
    rule: str = ex.FROZEN_RULE,
    nav_shift: int = ex.FROZEN_NAV_SHIFT,
) -> Ctx:
    root = root or inp.stock_data_root()
    sha = inp.verify_inputs(root) if verify else {}
    panel = ep.read_panel(root / inp.KRX_REL)
    seibro, _ = inp.load_seibro(root / inp.SEIBRO_REL)
    cm = inp.load_code_map(root / inp.CODE_MAP_REL, root / inp.MANUAL_REL)
    fee_df = inp.load_dis_fees(root / inp.DIS_RAW_REL)
    return make_ctx(panel, inp.distributions_for_use(seibro), cm, fee_df, rule, nav_shift, sha)


# ---------------------------------------------------------------- ETF-연 단위 제외 계산(모두 불리언)
def _index_change(ctx: Ctx, y: int) -> dict[str, bool]:
    """Y∪Y+1 안에서 IDX_IND_NM의 null 아닌 값이 둘 이상."""
    r = ctx.panel.rows.filter(
        pl.col("date").dt.year().is_in([y, y + 1]) & pl.col("IDX_IND_NM").is_not_null()
    )
    g = r.group_by("ISU_CD").agg(pl.col("IDX_IND_NM").n_unique().alias("n"))
    return dict(zip(g["ISU_CD"].to_list(), (g["n"] > 1).to_list()))


def outlier_mask(change: np.ndarray) -> np.ndarray:
    """NAV 하루 변화 절대값 ≥ 30%(경계 포함, 부동소수 오차 1e-12 허용)."""
    return np.abs(change) >= OUTLIER - 1e-12


def split_matrix(d: Dense) -> np.ndarray:
    """분할·병합일: LIST_SHRS 비 r ≥ 1.5 또는 ≤ 1/1.5 이고 NAV 비 × r 이 [0.9, 1.1]."""
    m = np.zeros_like(d.has_row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = d.shares[1:] / d.shares[:-1]
        nr = d.nav[1:] / d.nav[:-1]
        big = (r >= SPLIT_R) | (r <= 1.0 / SPLIT_R)
        prod = nr * r
        m[1:] = (
            np.isfinite(r)
            & np.isfinite(nr)
            & big
            & (prod >= SPLIT_BAND[0])
            & (prod <= SPLIT_BAND[1])
        )
    return m


def window_flags(ctx: Ctx, y: int, split: np.ndarray) -> pl.DataFrame:
    """연 y 형성 ETF-연(창 = Y 첫 거래일 ~ Y+1 마지막 거래일)의 제외 1~4 불리언 칸. 값은 내보내지 않는다."""
    d = ctx.dense
    s0, _ = d.year_range(y)
    _, e1 = d.year_range(y + 1)
    N = len(d.isu)
    first_idx = np.full(N, 10**9)
    fi = ctx.life.select("isu_cd", "first_idx")
    for c, f in zip(fi["isu_cd"].to_list(), fi["first_idx"].to_list()):
        first_idx[d.col[c]] = f
    sl = slice(s0, e1 + 1)
    t = np.arange(s0, e1 + 1)
    start = np.maximum(s0, first_idx)
    last_idx = np.full(N, -1)
    li = ctx.life.select("isu_cd", "last_idx")
    for c, f in zip(li["isu_cd"].to_list(), li["last_idx"].to_list()):
        last_idx[d.col[c]] = f
    end = np.minimum(e1, last_idx)  # 폐지된 ETF는 마지막 NAV 날까지(상장 유지 ETF는 e1)
    inwin = (t[:, None] >= start[None, :]) & (t[:, None] <= end[None, :])
    nav = d.nav[sl]
    miss = ((~np.isfinite(nav)) & inwin).sum(axis=0)
    first_nav_missing = ~np.isfinite(d.nav[np.minimum(start, e1), np.arange(N)])
    last_nav_missing = ~np.isfinite(d.nav[np.maximum(end, 0), np.arange(N)])
    nav_missing = (miss > NAV_MISS_MAX) | first_nav_missing | last_nav_missing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pc = np.full_like(d.nav, np.nan)
        pc[1:] = d.nav[1:] / d.nav[:-1] - 1.0
    outl = (outlier_mask(pc[sl]) & ~ctx.exmask[sl]).any(axis=0)
    spl = split[sl].any(axis=0)
    idxc = _index_change(ctx, y)
    return pl.DataFrame(
        {
            "year": [y] * N,
            "isu_cd": d.isu,
            "x_index_change": [idxc.get(c, False) for c in d.isu],
            "x_split_merge": spl.tolist(),
            "x_nav_missing": nav_missing.tolist(),
            "x_outlier": outl.tolist(),
        }
    )


@dataclass
class GroupYear:
    year: int
    key: str
    cols: np.ndarray  # ETF 열 번호(isu 코드순)
    isu: list[str]
    pair_ok: np.ndarray  # (n, n) bool: 두 창(Y, Y+1) 모두 공통일·분배락일 규칙 통과


@dataclass
class Sample:
    ey: pl.DataFrame  # 시작 ETF-연 전부(제외 표시 칸 포함)
    counts: pl.DataFrame  # 연도별 단계별 수
    gys: list[GroupYear]
    aux: dict = field(default_factory=dict)


def year_closed_panel(panel: ep.Panel) -> ep.Panel:
    """달력 끝에 다음 해 첫날 하나를 덧붙여, 마지막 해를 '끝난 해'로 보게 한 패널(``etf_e2.etf_years`` 용).

    ETF 행·거래일 번호는 그대로다. 호출하는 쪽이 자료가 그 해 마지막 거래일까지 있는지 확인해야 한다.
    """
    cal = panel.calendar
    last = cal["date"][-1]
    extra = pl.DataFrame(
        {
            "date": [date(last.year + 1, 1, 4)],
            "day_idx": [int(cal["day_idx"][-1]) + 1],
            "is_month_end": [True],
        },
        schema=cal.schema,
    )
    return dataclasses.replace(panel, calendar=pl.concat([cal, extra]))


def key_of(group_key: str, mode: str = "frozen") -> str:
    """``frozen`` = E2 동결 키 그대로. ``no_rt`` = 수익률 표기 칸(마지막)을 뺀 기록용 R4 키."""
    if mode == "frozen":
        return group_key
    if mode == "no_rt":
        return "|".join(group_key.split("|")[:-1])
    raise ValueError(mode)


def pair_window_ok(ctx: Ctx, i: int, j: int, y: int) -> bool:
    """연 y 창에서 쌍 (i, j)의 공통일이 있고, 두 ETF의 분배락일이 모두 공통일 안에 있다.

    창 끝은 두 ETF가 다 상장해 있는 마지막 날까지다(상장 유지 ETF 쌍이면 연 말일 그대로, 폐지 ETF가 낀 부분 창이면
    그 폐지일까지).
    """
    d = ctx.dense
    s, e = d.year_range(y)
    if ctx.last_idx is not None:
        e = min(e, int(ctx.last_idx[i]), int(ctx.last_idx[j]))
    if e < s:
        return False
    sl = slice(s, e + 1)
    common = ctx.rfin[sl, i] & ctx.rfin[sl, j]
    if common.sum() == 0:
        return False
    if (ctx.exmask[sl, i] & ~common).any() or (ctx.exmask[sl, j] & ~common).any():
        return False
    return True


def build_sample(
    ctx: Ctx,
    years: tuple[int, ...] | list[int] = JUDGMENT_YEARS,
    key_mode: str = "frozen",
    min_group: int = MIN_GROUP,
    fee_field: str = "ter",
    lag_days: int = fees.LAG_DAYS,
    region: str | None = None,
    aug_fixed: bool = False,
    include_delisted_y1: bool = False,
    close_year_end: bool = False,
) -> Sample:
    """표본·제외·개수.

    ``region`` 은 'domestic' 등으로 거르는 기록용 변형(국내 노출만). ``aug_fixed`` 는 F를 Y의 8월 말 파일에서 읽는다.
    ``include_delisted_y1`` 은 Y+1 중 상장폐지된 ETF-연을 Y+1 마지막 NAV 날까지의 부분 창으로 넣는다(R11, 나머지 조건
    같음. 결과 보수 A는 있는 달이 1개 이상이면 된다). ``close_year_end`` 는 자료가 Y+1의 마지막 거래일까지 있을 때
    끝나지 않은 해로 보지 않게 한다(전진 확인 R13).
    """
    d = ctx.dense
    life = ctx.life
    pnl = year_closed_panel(ctx.panel) if close_year_end else ctx.panel
    ey = e2.etf_years(pnl, life, list(years), with_values=False)
    if include_delisted_y1:
        ok_nodel = ~(
            pl.col("fail_days_y")
            | pl.col("fail_days_y1")
            | pl.col("fail_maturity")
            | pl.col("fail_pension")
            | pl.col("fail_hold")
        )
        ey = ey.filter(ok_nodel)
    else:
        ey = ey.filter(pl.col("ok_lenient"))
    ey = ey.with_columns(pl.col("fail_delisted_y1").alias("delisted_y1"))
    ey = ey.with_columns(
        pl.col("group_key")
        .map_elements(lambda k: key_of(k, key_mode), return_dtype=pl.String)
        .alias("gkey")
    )
    if region is not None:
        reg = dict(zip(life["isu_cd"].to_list(), life["region"].to_list()))
        ey = ey.filter(
            pl.col("isu_cd").map_elements(lambda c: reg.get(c) == region, return_dtype=pl.Boolean)
        )
    # 크기 ≥ min_group
    sz = ey.group_by("year", "gkey").len().rename({"len": "gsize"})
    ey = ey.join(sz, on=["year", "gkey"]).filter(pl.col("gsize") >= min_group)
    if ey.height == 0:
        zero = pl.DataFrame(
            {
                "year": [int(y) for y in years],
                "n_start": [0] * len(years),
                "n_final_etf_years": [0] * len(years),
            }
        )
        return Sample(ey=ey, counts=zero, gys=[])

    split = split_matrix(d)
    flags = pl.concat(
        [window_flags(ctx, int(y), split) for y in years if d.year_range(y + 1)[0] >= 0]
    )
    ey = ey.join(flags, on=["year", "isu_cd"], how="left")

    cons = ex.etf_year_consistency(d, ctx.dist, list(years))
    ey = ey.join(
        cons.select("year", "isu_cd", "n_dist", "ratio"), on=["year", "isu_cd"], how="left"
    )
    ey = ey.with_columns(
        (pl.col("ratio").is_not_null() & (pl.col("ratio") < ex.CONSISTENCY_MIN - 1e-12)).alias(
            "x_exdate_consistency"
        ),
        pl.col("n_dist").fill_null(0),
    )

    # 6, 7
    t_last = {y: d.dates[d.year_range(y)[1]] for y in years}
    fund = [ctx.code_map.get(c) for c in ey["isu_cd"].to_list()]
    x_map, f_ok, a_months = [], [], []
    for c, fnd, y in zip(ey["isu_cd"].to_list(), fund, ey["year"].to_list()):
        x_map.append(fnd is None)
        f_ok.append(
            fees.formation_available(ctx.fee, fnd, t_last[y], lag_days, fee_field, aug_fixed)
        )
        a_months.append(fees.outcome_months(ctx.fee, fnd, y + 1, fee_field))
    ey = ey.with_columns(
        pl.Series("fund_code", fund, dtype=pl.String),
        pl.Series("x_dis_map", x_map),
        pl.Series("f_available", f_ok),
        pl.Series("a_months", a_months, dtype=pl.Int64),
    ).with_columns(
        (
            ~pl.col("f_available")
            | (
                pl.col("a_months")
                < pl.when(pl.col("delisted_y1")).then(1).otherwise(fees.A_MIN_MONTHS)
            )
        ).alias("x_fee")
    )

    # 순차 제외 + 개수
    n_start = ey.group_by("year").len().rename({"len": "n_start"})
    rows = {
        int(y): {"year": int(y), "n_start": int(n)}
        for y, n in zip(n_start["year"], n_start["n_start"])
    }
    ey = ey.with_columns(pl.lit("").alias("excluded_at"))
    for k, name in enumerate(STEPS, 1):
        col = f"x_{name}"
        hit = ey[col].fill_null(False) & (ey["excluded_at"] == "")
        ey = ey.with_columns(
            pl.when(hit)
            .then(pl.lit(f"{k}_{name}"))
            .otherwise(pl.col("excluded_at"))
            .alias("excluded_at")
        )
        cnt = ey.filter(hit).group_by("year").len()
        rem = ey.filter(pl.col("excluded_at") == "").group_by("year").len()
        for y in rows:
            rows[y][f"n_removed_{k}_{name}"] = int(cnt.filter(pl.col("year") == y)["len"].sum())
            rows[y][f"n_after_{k}_{name}"] = int(rem.filter(pl.col("year") == y)["len"].sum())
        # 단계 단독 해당 수(겹침 포함)도 남긴다
        al = ey.group_by("year").agg(pl.col(col).fill_null(False).sum().alias("a"))
        for y, a in zip(al["year"], al["a"]):
            rows[int(y)][f"alone_{k}_{name}"] = int(a)

    surv = ey.filter(pl.col("excluded_at") == "")
    gcnt = surv.group_by("year", "gkey").len().rename({"len": "n"})
    keep = gcnt.filter(pl.col("n") >= min_group)
    ey = ey.join(
        keep.select("year", "gkey").with_columns(pl.lit(True).alias("final")),
        on=["year", "gkey"],
        how="left",
    )
    ey = ey.with_columns(
        (pl.col("final").fill_null(False) & (pl.col("excluded_at") == "")).alias("final")
    )
    fin = ey.filter(pl.col("final"))
    gys: list[GroupYear] = []
    n_pairs_all = {}
    n_pairs_ok = {}
    n_pairs_fail_y = {}
    n_pairs_fail_y1 = {}
    for (y, gk), g in fin.group_by(["year", "gkey"], maintain_order=True):
        g = g.sort("isu_cd")
        isu = g["isu_cd"].to_list()
        cols = np.array([d.col[c] for c in isu])
        n = len(cols)
        ok = np.zeros((n, n), dtype=bool)
        for a in range(n):
            for b in range(a + 1, n):
                oy = pair_window_ok(ctx, cols[a], cols[b], int(y))
                oy1 = pair_window_ok(ctx, cols[a], cols[b], int(y) + 1)
                ok[a, b] = ok[b, a] = oy and oy1
                n_pairs_all[y] = n_pairs_all.get(y, 0) + 1
                n_pairs_ok[y] = n_pairs_ok.get(y, 0) + int(oy and oy1)
                n_pairs_fail_y[y] = n_pairs_fail_y.get(y, 0) + int(not oy)
                n_pairs_fail_y1[y] = n_pairs_fail_y1.get(y, 0) + int(not oy1)
        gys.append(GroupYear(int(y), gk, cols, isu, ok))
    for y in rows:
        f_y = fin.filter(pl.col("year") == y)
        rows[y]["n_final_etf_years"] = f_y.height
        rows[y]["n_final_group_years"] = f_y["gkey"].n_unique()
        rows[y]["n_pairs_all"] = n_pairs_all.get(y, 0)
        rows[y]["n_pairs_exdate_ok"] = n_pairs_ok.get(y, 0)
        rows[y]["n_pairs_fail_window_Y"] = n_pairs_fail_y.get(y, 0)
        rows[y]["n_pairs_fail_window_Y1"] = n_pairs_fail_y1.get(y, 0)
        rows[y]["n_etf_years_with_dist"] = int((f_y["n_dist"] > 0).sum())
        rows[y]["n_etf_years_f_available_pre7"] = int(
            ey.filter(pl.col("year") == y)["f_available"].sum()
        )
    counts = pl.DataFrame(sorted(rows.values(), key=lambda r: r["year"]))
    return Sample(ey=ey, counts=counts, gys=gys)
