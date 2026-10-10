"""회사 점수(부분 C) 결과 변수 W4 (사전등록 20261010_quality_score §5.3·§5.4·§5.2·§7.2).

층 A 결과 O1~O4를 프레임에서 만드는 순수 함수 모음이다. 파일·레이크를 읽지 않는다.
로더는 메인이 묶는다. 결과를 만드는 함수는 시작에서 ``guard_years(fys, "outcome")`` 를 부른다
(§7.2: 형성 FY2019 이상은 ``QUALITY_C_JUDGMENT_CONFIRMED`` 가 있어야 한다).
입력 존재 비율(``opinion_missing_rate``)은 "inputs"라 막지 않는다.

출력 형식은 모두 ``[corp_code, fy, status, event]`` 이다.

* status: ``event`` | ``non_event`` | ``excluded_at_t`` | ``unobserved`` | ``excluded_capevt``(O3만)
* event: 1 / 0 / null (제외·미관측은 null)
* ``formation.in_universe`` 가 거짓인 행은 내보내지 않는다.

공통 규칙(§5.3): t에 이미 켜진 사건은 분모에서 뺀다. 결과 값은 가장 늦게 알려진 판본(실현 값).
사전등록이 정하지 않은 선택은 상수로 두고 ``# CI-<이름>`` 주석을 달았다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from modeler.scores.quality.company_common import (
    DEV_YEARS,
    O1_FALLBACK_YEARS,
    O4_JUDGMENT_YEARS,
    guard_years,
)

# ---------------------------------------------------------------- 상태값
EVENT = "event"
NON_EVENT = "non_event"
EXCLUDED_AT_T = "excluded_at_t"
UNOBSERVED = "unobserved"
EXCLUDED_CAPEVT = "excluded_capevt"

NON_CLEAN = "non_clean"
CLEAN = "clean"

OUT_COLS = ["corp_code", "fy", "status", "event"]

# ---------------------------------------------------------------- 상수 (처음 고른 숫자)
O3_MIN_DPS_KRW = 50.0  # §5.3 O3: KRW 보고 회사의 t DPS 하한(처음 고른 값)
O3_CUT_RATIO = 0.8  # §5.3 O3: t+1 DPS ≤ t DPS × 0.8 (20% 이상 삭감, 처음 고른 값)
O3_TOL = 1e-9  # 부동소수 경계(100 × 0.8 등) 흡수
EVEN_YEAR_GAP = 0.03  # §5.3 R03: 짝수 해와 인접 홀수 해의 의견 결측 비율 격차(처음 고른 값)

# CI-d: 한 (회사, fiscal_year)에 여러 보고서·연결/별도 의견이 있을 때.
#   "latest_report"(기본안): 분류가 None이 아닌 행이 있는 가장 늦은 보고서(report_year 최대,
#   같으면 rcept_no 최대) 안에서 non_clean이 하나라도 있으면 non_clean.
#   "any_report"(대안): 모든 보고서의 행 중 non_clean이 있으면 non_clean.
CI_O1_FOLD_MODE = "latest_report"  # CI-d

# CI-e: O1에서 t 의견이 없으면 "t에 이미 비적정"으로 보지 않는다(제외 안 함).
CI_O1_T_MISSING_EXCLUDES = False  # CI-e

# CI-o3-source: "single_report"(기본안): t+1 사업보고서 한 장의 frmtrm(=t DPS)·thstrm(=t+1 DPS).
#   "two_reports"(대안): t DPS는 t 보고서 thstrm, t+1 DPS는 t+1 보고서 thstrm.
CI_O3_SOURCE = "single_report"  # CI-o3-source

# CI-o3-capevt-year: 발행주식수 변동 사건을 보는 해. "t1"(기본안): t+1 해만.
#   "t_or_t1"(대안): t 또는 t+1 해.
CI_O3_CAPEVT_YEAR = "t1"  # CI-o3-capevt-year

# CI-o3-ccy: t DPS의 통화로 문턱(KRW 50원 / 그 밖 0 초과)을 정한다. 통화 null은 KRW로 읽는다
#   (company_score의 CI-currency와 같게 — 메인 검증 10-10).
CI_O3_CURRENCY_FROM = "t_dps_row"  # CI-o3-ccy

# CI-o2-cap: 자본금 ≤ 0 또는 결측이면 그 값을 결측으로 읽는다(형성·t+1 둘 다).
#   형성 시점 결측은 "이미 잠식"으로 보지 않는다(CI-e와 같은 결). t+1 결측은 unobserved.
CI_O2_NONPOS_CAP_IS_MISSING = True  # CI-o2-cap

# CI-o2-full: 완전 잠식(기록용)의 분모 제외는 형성 te < 0(t에 이미 완전 잠식)으로 한다.
CI_O2_FULL_EXCLUDE = "formation_te_negative"  # CI-o2-full

# CI-o4-formation: 형성 ni가 결측이면 "ni ≤ 0"으로 제외하지 않는다(CI-e와 같은 결).
CI_O4_T_MISSING_EXCLUDES = False  # CI-o4-formation

# CI-even-gap: 짝수 해를 인접 홀수 해와 비교할 때 앞뒤 두 홀수 해 비율의 평균과 비교하고
#   한쪽만 있으면 그 해와 비교한다. 격차는 절댓값이고 "3%p를 넘으면"(엄격 초과)이다.
CI_EVEN_COMPARE = "mean_of_adjacent_odd"  # CI-even-gap

_O1_MODES = ("full", "fallback")


# ---------------------------------------------------------------- 공통 보조
def _universe_base(formation: pl.DataFrame, fys: Iterable[int]) -> pl.DataFrame:
    """in_universe이고 fy가 fys인 (corp_code, fy) 유일 행 + 형성 열 전부."""
    fys = list(fys)
    return (
        formation.filter(pl.col("in_universe") & pl.col("fy").is_in(fys))
        .unique(subset=["corp_code", "fy"], keep="last", maintain_order=True)
        .with_columns(pl.col("fy").cast(pl.Int64))
    )


def _empty_out() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "corp_code": pl.Utf8,
            "fy": pl.Int64,
            "status": pl.Utf8,
            "event": pl.Int64,
        }
    )


def _shift(df: pl.DataFrame, k: int, cols: list[str], suffix: str) -> pl.DataFrame:
    """fs_latest 류 프레임을 (corp_code, year-k) 로 당겨 형성 연도 fy 키로 맞춘다."""
    return df.unique(subset=["corp_code", "year"], keep="last", maintain_order=True).select(
        pl.col("corp_code"),
        (pl.col("year") - k).cast(pl.Int64).alias("fy"),
        *[pl.col(c).alias(f"{c}{suffix}") for c in cols],
    )


# ---------------------------------------------------------------- O1 감사의견 비적정 (§5.3 O1)
def opinion_by_year(opinions: pl.DataFrame, mode: str | None = None) -> pl.DataFrame:
    """한 (회사, fiscal_year)의 의견을 하나로 접는다 (CI-d).

    입력 ``[corp_code, rcept_no, report_year, fiscal_year, opinion_class]``. 분류가 None인 행
    (연도·문자열을 못 읽은 행, §5.3 "결측")은 처음부터 버린다.
    출력 ``[corp_code, fiscal_year, opinion]`` — opinion ∈ {"non_clean", "clean"}.

    ``mode="latest_report"``: 가장 늦은 보고서(report_year 최대, 같으면 rcept_no 최대) 안에서
    non_clean이 하나라도 있으면 non_clean. ``mode="any_report"``: 모든 보고서에서 하나라도.
    """
    mode = mode or CI_O1_FOLD_MODE
    if mode not in ("latest_report", "any_report"):
        raise ValueError(f"mode는 latest_report·any_report 중 하나입니다: {mode!r}")
    rows = opinions.filter(pl.col("opinion_class").is_not_null())
    bad = rows.filter(~pl.col("opinion_class").is_in([NON_CLEAN, CLEAN]))
    if bad.height:
        raise ValueError(f"opinion_class는 non_clean·clean·None만 허용합니다: {bad.height}행")
    keys = ["corp_code", "fiscal_year"]
    if mode == "latest_report":
        rows = (
            rows.sort(["report_year", "rcept_no"])
            .with_columns(pl.col("rcept_no").last().over(keys).alias("_last_rcept"))
            .filter(pl.col("rcept_no") == pl.col("_last_rcept"))
        )
    return (
        rows.group_by(keys)
        .agg((pl.col("opinion_class") == NON_CLEAN).any().alias("_nc"))
        .select(
            pl.col("corp_code"),
            pl.col("fiscal_year").cast(pl.Int64),
            pl.when(pl.col("_nc"))
            .then(pl.lit(NON_CLEAN))
            .otherwise(pl.lit(CLEAN))
            .alias("opinion"),
        )
        .sort(keys)
    )


def o1_years(fys: Iterable[int], mode: str = "full") -> list[int]:
    """O1에 쓸 형성 연도. fallback이면 t+1이 홀수인 해만(§5.3 R03 폴백).

    폴백 허용 연도 = ``O1_FALLBACK_YEARS`` 와 개발 구간(DEV_YEARS) 중 t+1이 홀수인 해
    (개발에서는 t=2018, t+1=2019만 해당).
    """
    if mode not in _O1_MODES:
        raise ValueError(f"mode는 full·fallback 중 하나입니다: {mode!r}")
    fys = sorted(set(int(y) for y in fys))
    if mode == "full":
        return fys
    allowed = set(O1_FALLBACK_YEARS) | {y for y in DEV_YEARS if (y + 1) % 2 == 1}
    return [y for y in fys if y in allowed]


def o1(
    formation: pl.DataFrame,
    opinions: pl.DataFrame,
    fys: Iterable[int],
    mode: str = "full",
    fold_mode: str | None = None,
) -> pl.DataFrame:
    """O1 감사의견 비적정 (§5.3).

    사건 = t+1 의견 non_clean. 분모 제외: t 의견이 non_clean(``excluded_at_t``),
    t+1 의견 없음(``unobserved``). t 의견이 없으면 제외하지 않는다(CI-e).
    ``mode="fallback"`` 이면 형성 연도를 ``o1_years`` 로 줄인다(홀수 해 의견만 읽는 폴백).
    """
    fys = list(fys)
    guard_years(fys, "outcome")
    use = o1_years(fys, mode)
    op = opinion_by_year(opinions, fold_mode)
    base = _universe_base(formation, use).select("corp_code", "fy")
    if base.height == 0:
        return _empty_out()
    op_t = op.select(
        "corp_code", pl.col("fiscal_year").alias("fy"), pl.col("opinion").alias("op_t")
    )
    op_t1 = op.select(
        "corp_code", (pl.col("fiscal_year") - 1).alias("fy"), pl.col("opinion").alias("op_t1")
    )
    df = base.join(op_t, on=["corp_code", "fy"], how="left").join(
        op_t1, on=["corp_code", "fy"], how="left"
    )
    t_nc = pl.col("op_t") == NON_CLEAN
    if CI_O1_T_MISSING_EXCLUDES:
        t_nc = t_nc | pl.col("op_t").is_null()
    status = (
        pl.when(t_nc)
        .then(pl.lit(EXCLUDED_AT_T))
        .when(pl.col("op_t1").is_null())
        .then(pl.lit(UNOBSERVED))
        .when(pl.col("op_t1") == NON_CLEAN)
        .then(pl.lit(EVENT))
        .otherwise(pl.lit(NON_EVENT))
    )
    return _finish(df, status)


def _event_expr() -> pl.Expr:
    """status로부터 event를 만든다: event→1, non_event→0, 나머지 null."""
    return (
        pl.when(pl.col("status") == EVENT)
        .then(1)
        .when(pl.col("status") == NON_EVENT)
        .then(0)
        .otherwise(None)
    )


def _finish(df: pl.DataFrame, status: pl.Expr) -> pl.DataFrame:
    """status 식을 달고 event를 status에서 파생해 출력 열만 남긴다."""
    df = df.with_columns(status.alias("status")).with_columns(
        _event_expr().cast(pl.Int64).alias("event")
    )
    return df.select(
        pl.col("corp_code").cast(pl.Utf8),
        pl.col("fy").cast(pl.Int64),
        pl.col("status").cast(pl.Utf8),
        pl.col("event"),
    ).sort(["fy", "corp_code"])


# ---------------------------------------------------------------- O2 자본잠식 (§5.3 O2)
def _clean_cap(col: str) -> pl.Expr:
    c = pl.col(col)
    if CI_O2_NONPOS_CAP_IS_MISSING:
        return pl.when(c > 0).then(c).otherwise(None)
    return c


def o2(
    formation: pl.DataFrame,
    fs_latest: pl.DataFrame,
    fys: Iterable[int],
    kind: str = "partial",
) -> pl.DataFrame:
    """O2 자본잠식 (§5.3).

    ``kind="partial"``: 사건 = t+1 실현 te < 실현 cap(te == cap은 사건 아님). 제외: 형성
    te < 형성 cap. t+1 te·cap 중 결측이면 unobserved.
    ``kind="full"``(기록용 완전 잠식): 사건 = t+1 te < 0, 제외: 형성 te < 0 (CI-o2-full).
    cap ≤ 0은 결측으로 읽는다 (CI-o2-cap).
    """
    if kind not in ("partial", "full"):
        raise ValueError(f"kind는 partial·full 중 하나입니다: {kind!r}")
    fys = list(fys)
    guard_years(fys, "outcome")
    base = _universe_base(formation, fys).select("corp_code", "fy", "te", "cap")
    if base.height == 0:
        return _empty_out()
    nxt = _shift(fs_latest, 1, ["te", "cap"], "_t1")
    df = base.join(nxt, on=["corp_code", "fy"], how="left").with_columns(
        _clean_cap("cap").alias("cap"), _clean_cap("cap_t1").alias("cap_t1")
    )
    if kind == "partial":
        t_hit = pl.col("te") < pl.col("cap")
        t1_missing = pl.col("te_t1").is_null() | pl.col("cap_t1").is_null()
        t1_hit = pl.col("te_t1") < pl.col("cap_t1")
    else:
        t_hit = pl.col("te") < 0
        t1_missing = pl.col("te_t1").is_null()
        t1_hit = pl.col("te_t1") < 0
    status = (
        pl.when(t_hit.fill_null(False))
        .then(pl.lit(EXCLUDED_AT_T))
        .when(t1_missing)
        .then(pl.lit(UNOBSERVED))
        .when(t1_hit)
        .then(pl.lit(EVENT))
        .otherwise(pl.lit(NON_EVENT))
    )
    return _finish(df, status)


# ---------------------------------------------------------------- O3 배당 삭감·중단 (§5.3 O3)
def o3(
    formation: pl.DataFrame,
    dps_latest: pl.DataFrame,
    capevt: pl.DataFrame,
    fys: Iterable[int],
    source: str | None = None,
    capevt_year: str | None = None,
) -> pl.DataFrame:
    """O3 배당 삭감·중단 (§5.3, 발행주식수 변동 사건 목록은 §5.2).

    ``source="single_report"``(CI-o3-source 기본안): t+1 사업보고서(report_year = t+1) 한 장의
    frmtrm(``dps_p1``)을 t DPS, thstrm(``dps_t``)을 t+1 DPS로 쓴다.
    ``source="two_reports"``: t DPS = report_year t 행의 ``dps_t``, t+1 DPS = report_year t+1
    행의 ``dps_t``.

    t DPS 조건: KRW면 50원 이상, 다른 통화는 0 초과. 사건 = t+1 DPS ≤ t DPS × 0.8 (0 포함).
    판정 순서: t DPS가 있는데 조건 미달·0 → excluded_at_t / t+1 DPS 없음(보고서 없음 포함) →
    unobserved / t DPS 결측 → excluded_at_t(행 없음은 결측, §5.2) / t+1 해 발행주식수 변동 사건
    → excluded_capevt / 그 밖 event·non_event.
    ``capevt_year="t1"``(CI-o3-capevt-year 기본안)이면 t+1 해만, ``"t_or_t1"`` 이면 t 해도 본다.
    """
    source = source or CI_O3_SOURCE
    capevt_year = capevt_year or CI_O3_CAPEVT_YEAR
    if source not in ("single_report", "two_reports"):
        raise ValueError(f"source는 single_report·two_reports 중 하나입니다: {source!r}")
    if capevt_year not in ("t1", "t_or_t1"):
        raise ValueError(f"capevt_year는 t1·t_or_t1 중 하나입니다: {capevt_year!r}")
    fys = list(fys)
    guard_years(fys, "outcome")
    base = _universe_base(formation, fys).select("corp_code", "fy")
    if base.height == 0:
        return _empty_out()

    d = dps_latest.unique(subset=["corp_code", "report_year"], keep="last", maintain_order=True)
    r1 = d.select(
        "corp_code",
        (pl.col("report_year") - 1).cast(pl.Int64).alias("fy"),
        pl.col("currency").alias("ccy_r1"),
        pl.col("dps_t").alias("r1_thstrm"),
        pl.col("dps_p1").alias("r1_frmtrm"),
    )
    df = base.join(r1, on=["corp_code", "fy"], how="left")
    if source == "single_report":
        df = df.with_columns(
            pl.col("r1_frmtrm").alias("dps0"),
            pl.col("r1_thstrm").alias("dps1"),
            pl.col("ccy_r1").alias("ccy0"),
        )
    else:
        r0 = d.select(
            "corp_code",
            pl.col("report_year").cast(pl.Int64).alias("fy"),
            pl.col("currency").alias("ccy_r0"),
            pl.col("dps_t").alias("r0_thstrm"),
        )
        df = df.join(r0, on=["corp_code", "fy"], how="left").with_columns(
            pl.col("r0_thstrm").alias("dps0"),
            pl.col("r1_thstrm").alias("dps1"),
            pl.col("ccy_r0").alias("ccy0"),
        )

    # 발행주식수 변동 사건: 사건 있는 (corp, year) 집합
    ev = (
        capevt.select("corp_code", pl.col("year").cast(pl.Int64))
        .unique()
        .with_columns(pl.lit(True).alias("_ev"))
    )
    ev_t1 = ev.select("corp_code", (pl.col("year") - 1).alias("fy"), pl.col("_ev").alias("ev_t1"))
    df = df.join(ev_t1, on=["corp_code", "fy"], how="left")
    if capevt_year == "t_or_t1":
        ev_t = ev.select("corp_code", pl.col("year").alias("fy"), pl.col("_ev").alias("ev_t"))
        df = df.join(ev_t, on=["corp_code", "fy"], how="left")
        cap_hit = pl.col("ev_t1").fill_null(False) | pl.col("ev_t").fill_null(False)
    else:
        cap_hit = pl.col("ev_t1").fill_null(False)

    # CI-o3-ccy: t DPS 행의 통화로 문턱을 정한다.
    passes = (
        pl.when(pl.col("ccy0").fill_null("KRW") == "KRW")
        .then(pl.col("dps0") >= O3_MIN_DPS_KRW)
        .otherwise(pl.col("dps0") > 0)
    )
    t_known = pl.col("dps0").is_not_null()
    t_fail = t_known & ~passes.fill_null(False)
    status = (
        pl.when(t_fail)
        .then(pl.lit(EXCLUDED_AT_T))
        .when(pl.col("dps1").is_null())
        .then(pl.lit(UNOBSERVED))
        .when(~t_known)
        .then(pl.lit(EXCLUDED_AT_T))
        .when(cap_hit)
        .then(pl.lit(EXCLUDED_CAPEVT))
        .when((pl.col("dps1") <= pl.col("dps0") * O3_CUT_RATIO + O3_TOL) | (pl.col("dps1") <= 0))
        .then(pl.lit(EVENT))
        .otherwise(pl.lit(NON_EVENT))
    )
    return _finish(df, status)


# ---------------------------------------------------------------- O4 이익 급감 (§5.3 O4)
def _o4_check_years(fys: list[int]) -> None:
    last = max(O4_JUDGMENT_YEARS)
    over = [y for y in fys if y > last]
    if over:
        raise ValueError(f"O4 형성 연도는 FY{last}까지입니다(t+3 ≤ 2025, §5.3): {over}")


def _ni_wide(df_base: pl.DataFrame, fs_latest: pl.DataFrame, ks: tuple[int, ...]) -> pl.DataFrame:
    out = df_base
    for k in ks:
        out = out.join(_shift(fs_latest, k, ["ni"], f"_t{k}"), on=["corp_code", "fy"], how="left")
    return out


def o4(formation: pl.DataFrame, fs_latest: pl.DataFrame, fys: Iterable[int]) -> pl.DataFrame:
    """O4 이익 급감 (§5.3): t+1~t+3 실현 ni 중 음수가 둘 이상.

    세 해가 모두 관측돼야 하고 아니면 unobserved. 제외: 형성 ni ≤ 0. 형성 연도는
    ``O4_JUDGMENT_YEARS``(2019~2022)까지. 개발 디버깅은 ``o4_dev_t1`` 을 쓴다(§5.4).
    """
    fys = list(fys)
    guard_years(fys, "outcome")
    _o4_check_years(fys)
    base = _universe_base(formation, fys).select("corp_code", "fy", "ni")
    if base.height == 0:
        return _empty_out()
    df = _ni_wide(base, fs_latest, (1, 2, 3))
    cols = [pl.col(f"ni_t{k}") for k in (1, 2, 3)]
    n_neg = pl.sum_horizontal([(c < 0).cast(pl.Int64) for c in cols])
    any_missing = pl.any_horizontal([c.is_null() for c in cols])
    status = (
        pl.when(_o4_t_hit())
        .then(pl.lit(EXCLUDED_AT_T))
        .when(any_missing)
        .then(pl.lit(UNOBSERVED))
        .when(n_neg >= 2)
        .then(pl.lit(EVENT))
        .otherwise(pl.lit(NON_EVENT))
    )
    return _finish(df, status)


def _o4_t_hit() -> pl.Expr:
    hit = pl.col("ni") <= 0
    if CI_O4_T_MISSING_EXCLUDES:
        return hit | pl.col("ni").is_null()
    return hit.fill_null(False)


def o4_relaxed(
    formation: pl.DataFrame, fs_latest: pl.DataFrame, fys: Iterable[int]
) -> pl.DataFrame:
    """O4 완화판 (§5.3 R11, 기록용 — m에 안 센다).

    관측된 해 안에서 적자 2해 이상이면 사건. 관측된 해가 2해 미만이면 제외(unobserved).
    제외 사유는 같다: 형성 ni ≤ 0 → excluded_at_t.
    """
    fys = list(fys)
    guard_years(fys, "outcome")
    _o4_check_years(fys)
    base = _universe_base(formation, fys).select("corp_code", "fy", "ni")
    if base.height == 0:
        return _empty_out()
    df = _ni_wide(base, fs_latest, (1, 2, 3))
    cols = [pl.col(f"ni_t{k}") for k in (1, 2, 3)]
    n_obs = pl.sum_horizontal([c.is_not_null().cast(pl.Int64) for c in cols])
    n_neg = pl.sum_horizontal([(c < 0).fill_null(False).cast(pl.Int64) for c in cols])
    status = (
        pl.when(_o4_t_hit())
        .then(pl.lit(EXCLUDED_AT_T))
        .when(n_obs < 2)
        .then(pl.lit(UNOBSERVED))
        .when(n_neg >= 2)
        .then(pl.lit(EVENT))
        .otherwise(pl.lit(NON_EVENT))
    )
    return _finish(df, status)


def o4_dev_t1(formation: pl.DataFrame, fs_latest: pl.DataFrame, fys: Iterable[int]) -> pl.DataFrame:
    """O4 개발 디버깅판 (§5.4 "O4 디버깅은 t+1만"): 사건 = t+1 실현 ni < 0.

    제외: 형성 ni ≤ 0. t+1 ni 결측이면 unobserved. 판정에는 쓰지 않는다.
    """
    fys = list(fys)
    guard_years(fys, "outcome")
    base = _universe_base(formation, fys).select("corp_code", "fy", "ni")
    if base.height == 0:
        return _empty_out()
    df = _ni_wide(base, fs_latest, (1,))
    status = (
        pl.when(_o4_t_hit())
        .then(pl.lit(EXCLUDED_AT_T))
        .when(pl.col("ni_t1").is_null())
        .then(pl.lit(UNOBSERVED))
        .when(pl.col("ni_t1") < 0)
        .then(pl.lit(EVENT))
        .otherwise(pl.lit(NON_EVENT))
    )
    return _finish(df, status)


# ---------------------------------------------------------------- 분모에서 빠진 수 (§5.3 R11)
def unobserved_scored_counts(
    outcomes: Mapping[str, pl.DataFrame], scored: pl.DataFrame
) -> pl.DataFrame:
    """결과별·연도별로 점수 있는 회사 중 관측 요건에 걸려 분모에서 빠진 수.

    ``outcomes``: {결과 이름: 출력 프레임}. ``scored``: ``[corp_code, fy]``.
    출력 ``[outcome, fy, n_scored, n_unobserved]`` — n_scored = 그 결과 프레임에 행이 있는
    점수 있는 회사 수, n_unobserved = 그중 status가 unobserved인 수. **이유 구분 없이 한 숫자다.**
    상장폐지 여부·사유로 나누는 열을 만들지 않는다(TRS 2단 금지, §14 #27).
    """
    fys = sorted({int(y) for df in outcomes.values() for y in df["fy"].unique().to_list()})
    guard_years(fys, "outcome")
    sc = scored.select("corp_code", pl.col("fy").cast(pl.Int64)).unique()
    parts = []
    for name, df in outcomes.items():
        j = df.join(sc, on=["corp_code", "fy"], how="inner")
        parts.append(
            j.group_by("fy")
            .agg(
                pl.len().alias("n_scored"),
                (pl.col("status") == UNOBSERVED).sum().cast(pl.Int64).alias("n_unobserved"),
            )
            .with_columns(pl.lit(name).alias("outcome"))
            .select("outcome", "fy", pl.col("n_scored").cast(pl.Int64), "n_unobserved")
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "outcome": pl.Utf8,
                "fy": pl.Int64,
                "n_scored": pl.Int64,
                "n_unobserved": pl.Int64,
            }
        )
    return pl.concat(parts).sort(["outcome", "fy"])


# ---------------------------------------------------------------- 짝수 해 의견 결측 비율 (§5.3 R03)
def opinion_missing_rate(universe_years: pl.DataFrame, opinions: pl.DataFrame) -> pl.DataFrame:
    """분모 회사-연도 중 그 fiscal_year 의견(분류 None 아님)이 없는 비율 — 연도별.

    ``universe_years``: ``[corp_code, year]``. 출력 ``[year, n, n_missing, rate]``.
    입력 존재 비율이라 사건 수가 아니다. ``guard_years(..., "inputs")`` 라 판정 구간도 막지 않는다.
    """
    guard_years(universe_years["year"].unique().to_list(), "inputs")
    have = (
        opinions.filter(pl.col("opinion_class").is_not_null())
        .select("corp_code", pl.col("fiscal_year").cast(pl.Int64).alias("year"))
        .unique()
        .with_columns(pl.lit(True).alias("_has"))
    )
    u = universe_years.select("corp_code", pl.col("year").cast(pl.Int64)).unique()
    j = u.join(have, on=["corp_code", "year"], how="left")
    return (
        j.group_by("year")
        .agg(
            pl.len().cast(pl.Int64).alias("n"),
            pl.col("_has").is_null().sum().cast(pl.Int64).alias("n_missing"),
        )
        .with_columns((pl.col("n_missing") / pl.col("n")).alias("rate"))
        .sort("year")
    )


def even_year_flags(rates: pl.DataFrame, threshold: float = EVEN_YEAR_GAP) -> pl.DataFrame:
    """짝수 해 비율이 인접 홀수 해와 ``threshold``(3%p)를 넘게 벌어지는지 (§5.3 R03).

    ``rates``: ``opinion_missing_rate`` 출력. 출력 ``[year, t, rate, ref_rate, gap, demote]`` —
    ``year``는 짝수 해(= t+1), ``t`` = year − 1(O1 형성 연도), ``demote`` 가 True면 그 t를
    O1 판정에서 기록용으로 내린다. 비교 방법은 CI-even-gap: 앞뒤 두 홀수 해 비율의 평균, 한쪽만
    있으면 그 해. 홀수 이웃이 없는 짝수 해는 ref_rate·gap·demote가 null이다.
    """
    r = {int(y): float(v) for y, v in zip(rates["year"].to_list(), rates["rate"].to_list())}
    rows = []
    for y in sorted(r):
        if y % 2 != 0:
            continue
        nb = [r[k] for k in (y - 1, y + 1) if k in r]
        if not nb:
            rows.append((y, y - 1, r[y], None, None, None))
            continue
        ref = sum(nb) / len(nb)
        gap = r[y] - ref
        rows.append((y, y - 1, r[y], ref, gap, abs(gap) > threshold + 1e-12))
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "t": pl.Int64,
            "rate": pl.Float64,
            "ref_rate": pl.Float64,
            "gap": pl.Float64,
            "demote": pl.Boolean,
        },
        orient="row",
    )
