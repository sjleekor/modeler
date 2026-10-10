"""시장 국면 점수(MRS) 동결 상수.

사양: ``my/milestones/common/scores/20260924_market_regime_score/01_preregistration_draft.md``
(태그 ``mrs-frozen``, 2026-10-07 동결) 와 정정 블록 §0-2·§0-3·§0-4·§0-5.

**여기 숫자는 사전등록 문면에서 옮긴 것이다. 결과를 보고 바꾸지 않는다.** 문면이 정하지 않아
구현이 고른 값은 ``INTERP_*`` 이름으로 따로 두고, 구현 해석 표(04 문서)에 같은 이름으로 적는다.
"""

from __future__ import annotations

from datetime import date

# --------------------------------------------------------------------------- 점수 (§3)
#: 성분별 유효 관측 수. 이만큼 차기 전에는 그 성분 백분위를 내지 않는다 (§3.2 "5년"의 정의).
WARMUP_VALID_OBS = 1260
#: 롤링 창 (격자 관측 수로 센다, §3.2).
TREND_LAG = 252
MA_WINDOW = 200
RVOL_WINDOW = 20
MEDIAN_WINDOW = 252
LIQ_MEAN_WINDOW = 20
FLOW_WINDOW = 20
FX_LAG = 20
CREDIT_CHG_LAG = 20
#: FRED 계열 나이 상한(달력일). MS0 가용 규칙 "그 나이가 10일을 넘으면 null" (§4).
MACRO_STALENESS_DAYS = 10

SUB_SCORES = ("T", "V", "L")

#: 성분 -> (하위 점수, 부호). 부호 ``+``는 "값이 클수록 위험 선호" (§3.1). 산식 안의 음수는
#: 성분 함수가 처리하고, 여기 부호는 백분위에 곱하는 것이다.
KR_COMPONENTS: dict[str, tuple[str, str]] = {
    "kr_trend_252": ("T", "+"),
    "kr_trend_ma200": ("T", "+"),
    "vix_level": ("V", "+"),
    "kr_rvol_20": ("V", "+"),
    "kr_liq_20": ("L", "+"),
    "kr_foreign_20": ("L", "+"),
    "krw_20": ("L", "+"),
    "kr_term": ("L", "+"),
}
US_COMPONENTS: dict[str, tuple[str, str]] = {
    "us_trend_252": ("T", "+"),
    "us_trend_ma200": ("T", "+"),
    "vix_level": ("V", "+"),
    "us_rvol_20": ("V", "+"),
    "credit_baa10y_chg": ("L", "+"),
    "credit_baa10y_level": ("L", "+"),
    "us_term": ("L", "+"),
    # us_ftd_20 (부호 -)은 수집이 끝날 때까지 표에만 두고 계산하지 않는다 (§3.1).
}

# --------------------------------------------------------------------------- 비중 (§6)
WEIGHT_FLOOR = 0.5
WEIGHT_CAP = 1.0
WEIGHT_LOW_CUT = 20.0
WEIGHT_HIGH_CUT = 80.0
#: MRS가 NULL이면 비중 100% (§3.3 "모르면 하던 대로 한다").
WEIGHT_WHEN_NULL = 1.0
VM_FLOOR = 0.5
VM_CAP = 1.0
#: IRP 위험자산 상한 (§0-3.2).
IRP_CAP = 0.7

# --------------------------------------------------------------------------- 검정 (§5)
G1_MDD_RATIO_MAX = 0.80
G2_RET_RATIO_MIN = 0.90
G3_P_MAX = 0.05
G4_MIN_CORRECT_SIGN = 2
HAC_LAG = 20
FORWARD_HORIZON = 20
QUINTILE = 0.20
PLACEBO_N = 50
PLACEBO_SHIFT_MIN = 20
PLACEBO_SHIFT_MAX = 1000
PLACEBO_SEED = 20260924
TURNOVER_ANNUALIZE = 252

#: 왕복 비용(bp). 편도 = 왕복/2 (§5.3).
COST_RT_BP = {"KR": 60.0, "US": 5.0}
COST_RT_BP_SENS_KR = 120.0
#: 합성 현금 계정 staleness — ``scores/common/cash.py`` 기존 규칙 그대로 (§5.3).
CASH_STALENESS_DAYS = 14
CASH_SERIES = {"KR": "rate_kr_cd91", "US": "DGS3MO"}

# --------------------------------------------------------------------------- 구간 (§5.1)
KR_MAIN_START = date(2000, 1, 3)
KR_MAIN_END = date(2009, 12, 30)
KR_EXPLORE_START = date(2010, 1, 1)
US_EXPLORE_START = date(2016, 1, 1)
#: 동결 전 보호 구간 (§5.1). 승인된 실행 모드 밖에서는 이 구간의 KR 점수·성과를 계산하지 않는다.
PROTECTED_KR_START = KR_MAIN_START
PROTECTED_KR_END = KR_MAIN_END

# --------------------------------------------------------------------------- protocol (§11)
P_MAIN = "main_close_cash"
P_CASH0 = "sens_cash0"
P_COST120 = "sens_cost120"
P_VIX_PROXY = "sens_vix_proxy"
P_IRP = "sens_irp_cap07"
P_KTB_SYNTH = "ktb_synth_cash_sens"
#: 원안 시가 체결. 이 구간에는 정의하지 않는다 (§0-1 M3) — 돌리지 않고 manifest에 이유만 남긴다.
P_ORIG_OPEN = "orig_open"

# --------------------------------------------------------------------------- 합성 국고채 (§0-5)
KTB_SYNTH_MATURITY_YEARS = {"ktb3": 3, "ktb10": 10}
KTB_SYNTH_SERIES = {"ktb3": "rate_kr_gov3y", "ktb10": "rate_kr_gov10y"}
KTB_COUPONS_PER_YEAR = 4
KTB_DAYCOUNT = 365.0
#: P-S1 대용 = 위험자산 30% + 합성 3년 70%, P-S2 대용 = 위험자산 60% + 합성 10년 40% (§0-5.2).
MIX_SYNTH = {"mix_synth_ps1": (0.30, "ktb3"), "mix_synth_ps2": (0.60, "ktb10")}
#: 10년 합성은 2001-01부터 (§0-5.4). 없는 구간은 비워 둔다.
MIX_SYNTH_PS2_START = date(2001, 1, 1)

# --------------------------------------------------------------------------- 선행 백필 (§3.1)
#: 환율·국고채 ECOS 백필이 끝났다고 볼 첫 관측일 상한. 셋이 다 맞아야 실행한다.
BACKFILL_REQUIRED_START = {
    "fx_usdkrw_ecos": date(1990, 1, 3),
    "rate_kr_gov3y": date(1998, 11, 13),
    "rate_kr_gov10y": date(2000, 12, 18),
}
