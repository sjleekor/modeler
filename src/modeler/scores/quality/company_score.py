"""회사 품질 점수 계산 (사전등록 20261010_quality_score §5.2, 계획 05 §4 W3).

데이터를 읽지 않는 순수 함수 모듈이다. 입력은 계획 05 §5의 "입력 패널 열 계약"을 따르는
``pl.DataFrame`` 한 장(한 행 = (corp_code, fy))이고, 결과 변수는 다루지 않는다.

입력 가정
- 결측은 null/NaN. ``ip``(지급이자)는 **크기, 0 이상**이다. 부호는 입력 쪽(W2)이 절대값으로
  맞춘다고 가정하고 여기서는 고치지 않는다.
- ``capevt*``는 bool/NaN(발행주식수 변동 사건 해), ``retire*``는 0/1/NaN.
- 백분위 풀은 fy 안에서 ``in_universe`` 이고 그 성분 값이 있는 행이다(§5.1 "백분위").

사전등록 문면이 정하지 않은 선택은 ``# CI-<이름>`` 주석으로 표시했다(메인이 해석 표로 모음).
"""

from __future__ import annotations

import numpy as np
import polars as pl

SCORE_VERSION = "quality-score-company/company_score/1"

# ---------------------------------------------------------------- 상수 (§5.2)
# Altman Z'' 계수 (altman2000.txt:1053). 백분위에는 상수를 넣지 않는다.
ZPP_COEF_X1 = 6.56
ZPP_COEF_X2 = 3.26
ZPP_COEF_X3 = 6.72
ZPP_COEF_X4 = 1.05
ZPP_EM_CONST = 3.25  # 보고용 Z''_EM = Z'' + 3.25

ICR_CAP = 100.0  # 이자보상배율 상하 상한
CFO_NI_LOW = -1.0  # 영업CF/순이익 하한 (순이익 ≤ 0 이고 영업CF ≤ 0 이면 이 값)
CFO_NI_HIGH = 3.0  # 상한 (순이익 ≤ 0 이고 영업CF > 0 이면 이 값)

DPS_FLOOR_KRW = 50.0  # §5.3 O3: KRW 보고 회사의 삭감 판정 하한(원). 처음 고른 값
DPS_CUT_RATIO = 0.8  # 뒤 해 ≤ 앞 해 × 0.8 이면 삭감(20% 이상 감소)
DPS_DEFAULT_CURRENCY = "KRW"  # CI-currency: currency 결측은 KRW로 읽는다

MIN_COMPONENTS = 3  # C1·C2·C3 공통: 성분 3개 이상
MIN_DIMS = 3  # 합성: 세 차원 모두
MIN_DIMS_2DIM = 2  # 기록용 두 차원 이상 합성 (§14 #23 대안 ②)

PCT_SINGLE_POOL = 50.0  # CI-n1: 풀에 값이 하나뿐일 때 백분위 (ETF 해석 I03 선례)

# CI-c: 자본총계 ≤ 0 행의 순위 처리. "include" = 가장 나쁜 값으로 풀에 넣고 순위를 매긴 뒤
# 그 행의 백분위를 0으로 덮는다(기본). "exclude" = 풀에서 빼고 나머지만으로 순위를 매긴 뒤 0을 준다.
TE_NONPOS_MODE = "include"

KEY_COLS = ["corp_code", "fy"]

REQUIRED_COLS = [
    "corp_code", "fy", "in_universe", "currency",
    "ta", "tl", "te", "ca", "cl", "re", "oi", "ni", "ocf", "rev", "gp", "ltb", "ip", "cap",
    "ta_p1", "ta_p2", "te_p1", "te_p2", "ni_p1", "ni_p2", "ocf_p1", "ca_p1", "cl_p1",
    "rev_p1", "gp_p1", "ltb_p1", "dps", "dps_p1", "dps_p2", "shares", "shares_p1",
    "retire", "retire_p1", "capevt", "capevt_p1", "capevt_p2",
]  # fmt: skip

C1_COMPONENTS = ["zpp", "capr", "icr", "cr", "dr"]
C2_COMPONENTS = ["acc", "cfo", "roe", "rvol"]
C3_COMPONENTS = ["streak", "cuts", "iss", "ret"]
# 순환 성분을 뺀 남은 성분 (§5.2 "입력과 결과의 순환")
C1_ALT_COMPONENTS = ["icr", "cr"]  # O2용: 자본잠식률·부채비율·Z'' 제외
C2_ALT_COMPONENTS = ["acc", "cfo"]  # O4용: ROE 수준·변동성 제외
C3_ALT_COMPONENTS = ["iss", "ret"]  # O3용: 연속 배당 연수·삭감 이력 제외


# ---------------------------------------------------------------- 백분위 (§5.1)
def pct_rank(
    values: pl.Series,
    fy: pl.Series,
    in_universe: pl.Series,
    higher_is_better: bool = True,
    worst_mask: pl.Series | None = None,
    worst_mode: str = TE_NONPOS_MODE,
) -> pl.Series:
    """fy 안 횡단면 백분위 = (평균 순위 − 1) ÷ (n − 1) × 100. 높을수록 좋음 = 100.

    풀 = in_universe 이고 값이 있는 행(NaN은 결측). 동률은 평균 순위.
    풀에 한 행뿐이면 ``PCT_SINGLE_POOL``(CI-n1). 풀 밖 행은 null.
    ``worst_mask``가 참인 in_universe 행은 "최하위"(백분위 0)다(§5.2 부채비율·ROE, CI-c).
    그 행의 원값은 호출 쪽이 정하지 않아도 된다(값 대신 최악으로 취급).
    """
    n_rows = len(values)
    v = values.cast(pl.Float64).fill_nan(None)
    uni = in_universe.fill_null(False)
    wm = (
        worst_mask.fill_null(False)
        if worst_mask is not None
        else pl.Series("wm", [False] * n_rows, dtype=pl.Boolean)
    )
    df = pl.DataFrame({"v": v, "fy": fy, "u": uni, "wm": wm})
    key = pl.when(pl.col("v").is_not_null()).then(pl.col("v") if higher_is_better else -pl.col("v"))
    if worst_mode == "include":
        key = pl.when(pl.col("wm")).then(pl.lit(-np.inf)).otherwise(key)
    elif worst_mode != "exclude":
        raise ValueError(f"worst_mode={worst_mode}")
    df = df.with_columns(pl.when(pl.col("u")).then(key).alias("k"))
    df = df.with_columns(
        (pl.col("k").rank("average").over("fy")).alias("r"),
        pl.col("k").count().over("fy").alias("n"),
    )
    df = df.with_columns(
        pl.when(pl.col("k").is_null())
        .then(None)
        .when(pl.col("n") == 1)
        .then(pl.lit(PCT_SINGLE_POOL))
        .otherwise((pl.col("r") - 1) / (pl.col("n") - 1) * 100.0)
        .alias("p")
    )
    df = df.with_columns(
        pl.when(pl.col("u") & pl.col("wm")).then(pl.lit(0.0)).otherwise(pl.col("p")).alias("p")
    )
    return df["p"]


# ---------------------------------------------------------------- 원값 (numpy)
def _f(df: pl.DataFrame, col: str) -> np.ndarray:
    """열을 float64 배열로. 결측(null/NaN)은 NaN."""
    return df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)


def _div(a: np.ndarray, b: np.ndarray, ok: np.ndarray | None = None) -> np.ndarray:
    """a / b. 입력 결측 또는 ok가 거짓이면 NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    bad = np.isnan(a) | np.isnan(b)
    if ok is not None:
        bad = bad | ~ok
    return np.where(bad, np.nan, out)


def zpp_value(ca, cl, ta, re, oi, te, tl) -> np.ndarray:
    """Altman Z'' (§5.2). ta ≤ 0 또는 tl ≤ 0 또는 입력 결측이면 NaN."""
    ok = (ta > 0) & (tl > 0)  # NaN 비교는 False
    x1 = _div(ca - cl, ta, ok)
    x2 = _div(re, ta, ok)
    x3 = _div(oi, ta, ok)
    x4 = _div(te, tl, ok)
    return ZPP_COEF_X1 * x1 + ZPP_COEF_X2 * x2 + ZPP_COEF_X3 * x3 + ZPP_COEF_X4 * x4


def icr_paid_value(oi: np.ndarray, ip: np.ndarray) -> np.ndarray:
    """이자보상배율 = 영업이익 ÷ 지급이자, ±100 상한. 지급이자 0이면 oi>0 → +100, 아니면 −100."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(oi / ip, -ICR_CAP, ICR_CAP)
    out = np.where(ip == 0, np.where(oi > 0, ICR_CAP, -ICR_CAP), ratio)
    return np.where(np.isnan(oi) | np.isnan(ip), np.nan, out)


def cfo_ni_value(ni: np.ndarray, ocf: np.ndarray) -> np.ndarray:
    """영업CF/순이익. ni>0 이면 clip(ocf/ni, −1, 3), ni ≤ 0 이면 ocf>0 → 3, ocf ≤ 0 → −1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.clip(ocf / ni, CFO_NI_LOW, CFO_NI_HIGH)
    out = np.where(ni > 0, pos, np.where(ocf > 0, CFO_NI_HIGH, CFO_NI_LOW))
    return np.where(np.isnan(ni) | np.isnan(ocf), np.nan, out)


def _streak(dps: np.ndarray, dps_p1: np.ndarray, dps_p2: np.ndarray) -> np.ndarray:
    """연속 배당 연수 0~3 (t에서 거꾸로 > 0).

    CI-streak: dps 결측이면 결측. 앞 연도가 결측이면 거기서 끊고 센 값을 쓴다.
    (대안: 앞 연도 결측이면 결측 / 결측을 0으로 읽음)
    """
    n = len(dps)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(dps[i]):
            continue
        k = 0
        for x in (dps[i], dps_p1[i], dps_p2[i]):
            if np.isnan(x) or not x > 0:
                break
            k += 1
        out[i] = k
    return out


def _cut_pair(prev: np.ndarray, nxt: np.ndarray, krw: np.ndarray) -> np.ndarray:
    """삭감 1쌍(앞 해 → 뒤 해): 1/0/NaN (§5.3 O3 정의).

    앞 해 DPS ≥ 50원(KRW; 다른 통화는 > 0)이고 뒤 해 ≤ 앞 해 × 0.8 (0 포함)이면 1.
    앞 해가 하한 미만이면 삭감 아님(0).
    """
    base = np.where(krw, prev >= DPS_FLOOR_KRW, prev > 0)
    cut = base & (nxt <= prev * DPS_CUT_RATIO)
    return np.where(np.isnan(prev) | np.isnan(nxt), np.nan, cut.astype(float))


def cut_history_value(dps, dps_p1, dps_p2, krw, capevt, capevt_p1) -> np.ndarray:
    """삭감 이력 0~2. 두 쌍 모두 계산돼야 하고 사건 해면 결측.

    CI-cut: (a) capevt 또는 capevt_p1이 참이면 결측(사전등록 "같은 해 C3의 삭감 이력도 결측").
    대안: 쌍별(p2→p1은 capevt_p1, p1→t는 capevt)로 결측. capevt_p2는 쓰지 않는다.
    (b) 두 쌍 중 하나라도 결측이면 결측(대안: 있는 쌍만 셈).
    """
    p1 = _cut_pair(dps_p2, dps_p1, krw)
    p2 = _cut_pair(dps_p1, dps, krw)
    out = p1 + p2
    event = (capevt == 1) | (capevt_p1 == 1)
    return np.where(event, np.nan, out)


def retire_value(retire: np.ndarray, retire_p1: np.ndarray) -> np.ndarray:
    """소각: 하나라도 1이면 1, 둘 다 0이면 0, 그 밖(결측 포함)은 NaN.

    CI-ret: 한쪽만 결측이고 다른 쪽이 0이면 결측(대안: 0으로 읽음).
    """
    out = np.where(
        (retire == 1) | (retire_p1 == 1),
        1.0,
        np.where((retire == 0) & (retire_p1 == 0), 0.0, np.nan),
    )
    return out


def piotroski_signals(d: dict[str, np.ndarray], ltb_fill: bool = True) -> dict[str, np.ndarray]:
    """F1~F9 (1/0/NaN). ``d``는 열 이름 → float 배열.

    분모 ≤ 0이면 그 신호는 NaN(CI-fden: 대안은 0 처리). ltb_fill=False면 ltb 결측 → F5 NaN.
    F5는 엄격히 작을 때만 1(동률 0).
    """
    ta, ta1, ta2 = d["ta"], d["ta_p1"], d["ta_p2"]
    ni, ni1, ocf = d["ni"], d["ni_p1"], d["ocf"]
    ok1, ok2 = ta1 > 0, ta2 > 0
    roa = _div(ni, ta1, ok1)
    roa1 = _div(ni1, ta2, ok2)
    cfo = _div(ocf, ta1, ok1)

    def sig(cond: np.ndarray, *vals: np.ndarray) -> np.ndarray:
        bad = np.zeros(len(cond), dtype=bool)
        for x in vals:
            bad |= np.isnan(x)
        return np.where(bad, np.nan, cond.astype(float))

    f = {}
    f["f1"] = sig(roa > 0, roa)
    f["f2"] = sig(cfo > 0, cfo)
    f["f3"] = sig(roa > roa1, roa, roa1)
    f["f4"] = sig(cfo > roa, cfo, roa)

    ltb, ltb1 = d["ltb"], d["ltb_p1"]
    if ltb_fill:  # 결측은 0으로 읽음 (§5.2 장기차입금 결측 처리)
        ltb = np.where(np.isnan(ltb), 0.0, ltb)
        ltb1 = np.where(np.isnan(ltb1), 0.0, ltb1)
    avg_t = (ta + ta1) / 2
    avg_p = (ta1 + ta2) / 2
    lev = _div(ltb, avg_t, avg_t > 0)
    lev1 = _div(ltb1, avg_p, avg_p > 0)
    f["f5"] = sig(lev < lev1, lev, lev1)

    cr = _div(d["ca"], d["cl"], d["cl"] > 0)
    cr1 = _div(d["ca_p1"], d["cl_p1"], d["cl_p1"] > 0)
    f["f6"] = sig(cr > cr1, cr, cr1)

    ev = d["capevt"] == 1
    f7 = sig(d["shares"] <= d["shares_p1"], d["shares"], d["shares_p1"])
    f["f7"] = np.where(ev, np.nan, f7)  # 사건 해는 결측 (F 전체가 결측)

    gm = _div(d["gp"], d["rev"], d["rev"] > 0)
    gm1 = _div(d["gp_p1"], d["rev_p1"], d["rev_p1"] > 0)
    f["f8"] = sig(gm > gm1, gm, gm1)

    tn = _div(d["rev"], ta1, ok1)
    tn1 = _div(d["rev_p1"], ta2, ok2)
    f["f9"] = sig(tn > tn1, tn, tn1)
    return f


# ---------------------------------------------------------------- 차원 합성
def _dim_score(
    df: pl.DataFrame,
    prefix: str,
    comps: list[str],
    must: str | None = None,
    min_n: int = MIN_COMPONENTS,
) -> tuple[pl.Series, pl.Series]:
    """있는 성분 백분위의 평균과 쓴 성분 수. 성분 수 < min_n 이거나 ``must`` 성분이 없으면 null."""
    cols = [pl.col(f"{prefix}_{c}_pct") for c in comps]
    n = pl.sum_horizontal([c.is_not_null() for c in cols])
    mean = pl.mean_horizontal(cols)
    cond = n >= min_n
    if must is not None:
        cond = cond & pl.col(f"{prefix}_{must}_pct").is_not_null()
    out = df.select(
        pl.when(cond).then(mean).alias("s"),
        n.cast(pl.Int64).alias("n"),
    )
    return out["s"], out["n"]


def _alt_score(df: pl.DataFrame, prefix: str, comps: list[str]) -> pl.Series:
    """순환 성분을 뺀 차원 점수: 남은 성분이 **모두** 있어야 평균 (CI-alt)."""
    cols = [pl.col(f"{prefix}_{c}_pct") for c in comps]
    n = pl.sum_horizontal([c.is_not_null() for c in cols])
    return df.select(pl.when(n == len(comps)).then(pl.mean_horizontal(cols)).alias("s"))["s"]


def _mean_of(df: pl.DataFrame, names: list[str], min_n: int) -> tuple[pl.Series, pl.Series]:
    cols = [pl.col(c) for c in names]
    n = pl.sum_horizontal([c.is_not_null() for c in cols])
    out = df.select(
        pl.when(n >= min_n).then(pl.mean_horizontal(cols)).alias("s"),
        n.cast(pl.Int64).alias("n"),
    )
    return out["s"], out["n"]


# ---------------------------------------------------------------- 공개 함수
def compute_scores(panel: pl.DataFrame) -> pl.DataFrame:
    """입력 패널 → 성분 원값·백분위·차원 점수·합성·F·기록용 대안 (키 열 포함).

    출력 열
    - 키: corp_code, fy, in_universe
    - C1: c1_zpp, c1_zpp_em, c1_capr, c1_icr, c1_cr, c1_dr (+ ``_pct``), c1_dr_worst,
      c1, c1_n, c1_pct
    - C2: c2_acc, c2_cfo, c2_roe, c2_rvol (+ ``_pct``), c2_roe_worst, c2, c2_n, c2_pct
    - C3: c3_streak, c3_cuts, c3_iss, c3_ret (+ ``_pct``), c3, c3_n, c3_pct
    - 합성: c (세 차원 필수), c_2dim, c_2dim_n (두 차원 이상)
    - 기록용 대안: c1_alt·c2_alt·c3_alt (순환 성분을 뺀 차원 점수), *_alt_pct,
      c_o2 (C1′ 대체) · c_o3 (C3′ 대체) · c_o4 (C2′ 대체)
    - F: f1~f9, f_sum, f_ltb_req (장기차입금 값 필수 F 합)
    ``c1_dr``·``c2_roe``의 원값은 자본총계 ≤ 0 행에서 null이고 ``*_worst``가 참이다.
    """
    missing = [c for c in REQUIRED_COLS if c not in panel.columns]
    if missing:
        raise ValueError(f"입력 패널 열 누락: {missing}")

    df = panel.select(REQUIRED_COLS)
    fy = df["fy"]
    uni = df["in_universe"].fill_null(False)
    g = {
        c: _f(df, c)
        for c in REQUIRED_COLS
        if c not in ("corp_code", "fy", "in_universe", "currency")
    }
    krw = (
        df["currency"].fill_null(DPS_DEFAULT_CURRENCY).eq(DPS_DEFAULT_CURRENCY).to_numpy()
    )  # CI-currency

    ta, tl, te = g["ta"], g["tl"], g["te"]

    # --- C1 원값
    zpp = zpp_value(g["ca"], g["cl"], ta, g["re"], g["oi"], te, tl)
    capr = _div(te, g["cap"], g["cap"] > 0)  # cap ≤ 0 → 결측 (CI-cap)
    icr = icr_paid_value(g["oi"], g["ip"])
    cr = _div(g["ca"], g["cl"], g["cl"] > 0)  # cl ≤ 0 → 결측 (CI-cr)
    dr = _div(tl, te, te > 0)
    dr_worst = (te <= 0) & ~np.isnan(tl)  # 자본총계 ≤ 0 → 최하위

    # --- C2 원값
    ta_avg = (ta + g["ta_p1"]) / 2
    acc = _div(g["ni"] - g["ocf"], ta_avg, ta_avg > 0)  # CI-acc: 평균 총자산 ≤ 0 → 결측
    cfo = cfo_ni_value(g["ni"], g["ocf"])
    roe = _div(g["ni"], te, te > 0)
    roe_worst = (te <= 0) & ~np.isnan(g["ni"])
    roe1 = _div(g["ni_p1"], g["te_p1"], g["te_p1"] > 0)
    roe2 = _div(g["ni_p2"], g["te_p2"], g["te_p2"] > 0)
    # CI-rvol: 세 te 중 하나라도 ≤ 0 이거나 결측이면 결측 (roe*가 NaN이 되어 전파)
    triple = np.vstack([roe, roe1, roe2])
    rvol = np.where(np.isnan(triple).any(axis=0), np.nan, np.nan_to_num(triple).std(axis=0, ddof=1))

    # --- C3 원값
    streak = _streak(g["dps"], g["dps_p1"], g["dps_p2"])
    cuts = cut_history_value(g["dps"], g["dps_p1"], g["dps_p2"], krw, g["capevt"], g["capevt_p1"])
    iss = _div(g["shares"] - g["shares_p1"], g["shares_p1"], g["shares_p1"] > 0)
    iss = np.where(g["capevt"] == 1, np.nan, iss)  # 사건 해는 결측
    ret = retire_value(g["retire"], g["retire_p1"])

    raw: dict[str, tuple[np.ndarray, bool, np.ndarray | None]] = {
        # 이름: (값, 높을수록 좋음, 최하위 마스크)
        "c1_zpp": (zpp, True, None),
        "c1_capr": (capr, True, None),
        "c1_icr": (icr, True, None),
        "c1_cr": (cr, True, None),
        "c1_dr": (dr, False, dr_worst),
        "c2_acc": (acc, False, None),
        "c2_cfo": (cfo, True, None),
        "c2_roe": (roe, True, roe_worst),
        "c2_rvol": (rvol, False, None),
        "c3_streak": (streak, True, None),
        "c3_cuts": (cuts, False, None),
        "c3_iss": (iss, False, None),
        "c3_ret": (ret, True, None),
    }
    out: dict[str, pl.Series] = {
        "corp_code": df["corp_code"],
        "fy": fy,
        "in_universe": uni,
        "c1_zpp_em": pl.Series(zpp + ZPP_EM_CONST, dtype=pl.Float64).fill_nan(None),
    }
    for name, (val, hib, wm) in raw.items():
        s = pl.Series(name, val, dtype=pl.Float64).fill_nan(None)
        out[name] = s
        out[f"{name}_pct"] = pct_rank(s, fy, uni, hib, None if wm is None else pl.Series(wm)).alias(
            f"{name}_pct"
        )
    out["c1_dr_worst"] = pl.Series(dr_worst & uni.to_numpy())
    out["c2_roe_worst"] = pl.Series(roe_worst & uni.to_numpy())

    res = pl.DataFrame(out)

    # --- 차원 점수 (성분 3개 이상, C1은 Z'' 필수)
    for prefix, comps, must in (
        ("c1", [c for c in C1_COMPONENTS], "zpp"),
        ("c2", C2_COMPONENTS, None),
        ("c3", C3_COMPONENTS, None),
    ):
        s, n = _dim_score(res, prefix, comps, must)
        res = res.with_columns(s.alias(prefix), n.alias(f"{prefix}_n"))
        res = res.with_columns(pct_rank(res[prefix], fy, uni).alias(f"{prefix}_pct"))

    # --- 합성
    comp, comp_n = _mean_of(res, ["c1_pct", "c2_pct", "c3_pct"], MIN_DIMS)
    comp2, comp2_n = _mean_of(res, ["c1_pct", "c2_pct", "c3_pct"], MIN_DIMS_2DIM)
    res = res.with_columns(comp.alias("c"), comp2.alias("c_2dim"), comp2_n.alias("c_2dim_n"))

    # --- 기록용 대안: 순환 성분을 뺀 합성
    for prefix, comps in (
        ("c1", C1_ALT_COMPONENTS),
        ("c2", C2_ALT_COMPONENTS),
        ("c3", C3_ALT_COMPONENTS),
    ):
        a = _alt_score(res, prefix, comps)
        res = res.with_columns(a.alias(f"{prefix}_alt"))
        res = res.with_columns(pct_rank(res[f"{prefix}_alt"], fy, uni).alias(f"{prefix}_alt_pct"))
    for name, rep, others in (
        ("c_o2", "c1_alt_pct", ("c2_pct", "c3_pct")),
        ("c_o3", "c3_alt_pct", ("c1_pct", "c2_pct")),
        ("c_o4", "c2_alt_pct", ("c1_pct", "c3_pct")),
    ):
        s, _ = _mean_of(res, [rep, *others], MIN_DIMS)
        res = res.with_columns(s.alias(name))

    # --- C5 Piotroski F
    fcols: dict[str, pl.Series] = {}
    f = piotroski_signals(g, ltb_fill=True)
    for k, v in f.items():
        fcols[k] = pl.Series(k, v, dtype=pl.Float64).fill_nan(None).cast(pl.Int8)
    fres = pl.DataFrame(fcols)
    f_sum = fres.select(
        pl.when(pl.sum_horizontal([pl.col(c).is_not_null() for c in fres.columns]) == 9)
        .then(pl.sum_horizontal(pl.all()))
        .alias("f_sum")
    )["f_sum"]
    f_req = piotroski_signals(g, ltb_fill=False)
    f_req_df = pl.DataFrame(
        {k: pl.Series(k, v, dtype=pl.Float64).fill_nan(None) for k, v in f_req.items()}
    )
    f_ltb = f_req_df.select(
        pl.when(pl.sum_horizontal([pl.col(c).is_not_null() for c in f_req_df.columns]) == 9)
        .then(pl.sum_horizontal(pl.all()))
        .cast(pl.Int64)
        .alias("f_ltb_req")
    )["f_ltb_req"]
    res = res.with_columns(*fres.get_columns(), f_sum.cast(pl.Int64).alias("f_sum"), f_ltb)
    return res
