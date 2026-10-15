"""Frozen rules of the case gallery pilot (my/milestones/common/20261012_feature_explainer/11).

Everything here was fixed before any result was looked at. Changing a value after a run is a
correction block in the plan, not an edit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

SEED = 20261015
PURPOSE = "KR 개발 구간 사례 열람(가설 탐색, 2026-10-15)"

PANEL_REL = (
    "kr/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/"
    "FS1h_h20_lag1_rank/feat_panel.parquet"
)
DATASET_MANIFEST_REL = (
    "kr/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/"
    "FS1h_h20_lag1_rank/dataset_manifest.json"
)
RAW_REL = "kr/raw/raw_postgres/snapshot_date=2026-08-23/source=sj2_remote"

# --- hard date guards (plan section 10, D1) ------------------------------------------------
DATA_END = dt.date(2025, 7, 31)  # the development panel ends here; nothing later is read
FORMATION_START = dt.date(2016, 1, 4)
FORMATION_END = dt.date(2025, 6, 30)
HORIZON = 20
PRE_DAYS = 120  # price chart reaches back this many trading days
YEARS = tuple(range(2016, 2026))
N_DATES = 12

# --- case counts per date ------------------------------------------------------------------
N_GOOD = 3
N_OPP = 3
N_ALL = 2
N_EXTREME = 5
TOP_PCT = 0.9
BOTTOM_PCT = 0.1

# --- chart flags ---------------------------------------------------------------------------
LIMIT_LO = 0.295  # |close/prev - 1| in [LIMIT_LO, LIMIT_HI] counts as a price-limit hit (+-30%)
LIMIT_HI = 0.31
TAX_REF = 0.002  # 0.20% sell tax reference line
CORP_ACTIONS = ("주식분할", "무상증자", "무상감자")

LABEL_LIKE_PREFIXES = ("fwd_", "bench_", "raw_label", "y_")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    direction: int  # +1 high is the good side, -1 low is, 0 none
    basis: str  # where the direction comes from
    window: int | None  # feature window length in trading days (None = no price window)
    card: str  # card file in the explainer directory
    short: str  # one-line meaning (card summary table, verbatim)
    expected: str  # expected sign as the card writes it
    layer3: str  # layer 3 response (09_kr_result.md section 3)
    test: str  # univariate test status
    good_label: str  # what the good side means in words


_NO_TEST = "단변량 검정 없음(FS0). 이 그림이 첫 단변량 관찰이며 판정이 아님"

FEATURES: dict[str, FeatureSpec] = {
    s.name: s
    for s in (
        FeatureSpec(
            "px_idio_vol_60d",
            -1,
            "카드 기대 부호 −",
            60,
            "01_kr_price",
            "시장 움직임으로 설명 안 되는, 그 종목만의 널뛰기",
            "−",
            "SHAP 1위(0.0642) · 방향 ρ −0.96 · 단조 감소 · 안정",
            "scan 단변량 통과. A 6/6, IC −0.1433(0→60). 35개 중 최대(선택 편향 포함). 비용 전",
            "값이 낮은 쪽(변동이 작은 종목)",
        ),
        FeatureSpec(
            "px_turnover_ma20",
            -1,
            "층 3 모델 반응 방향(탐색)",
            20,
            "01_kr_price",
            "최근 한 달 평균 거래대금",
            "미등록",
            "SHAP 2위(0.0354) · 방향 ρ −0.94 · 단조 감소 · 안정",
            _NO_TEST,
            "값이 낮은 쪽(거래대금이 작은 종목) — 모델 반응 방향, 탐색",
        ),
        FeatureSpec(
            "px_dist_52w_high",
            1,
            "계획 §3 표(높은 쪽)",
            252,
            "01_kr_price",
            "1년 최고가에서 얼마나 내려와 있나",
            "미등록. 관련 `px_near_52w_high`는 +",
            "SHAP 4위(0.0319) · 방향 ρ +0.62 · U자 · 안정",
            "관련 family만 검정(A 3/6, IC +·분위 차이 −). 컬럼 자체는 검정 안 함",
            "값이 높은 쪽(52주 고점에 가까운 종목)",
        ),
        FeatureSpec(
            "px_reversal_5d",
            1,
            "카드 기대 부호 +",
            5,
            "01_kr_price",
            "최근 5일 많이 떨어진 종목이 며칠 뒤 되오른다는 생각",
            "+",
            "SHAP 23위(0.0072) · 방향 ρ +0.80 · 기타 · 불안정",
            "scan 단변량 통과. A 7/7(겹치는 셀 수), 효과는 3일 정점. 비용 전·체결 지연 전",
            "값이 높은 쪽(최근 5일 많이 떨어진 종목)",
        ),
        FeatureSpec(
            "fin_roa",
            1,
            "카드 기대 부호 +(문헌, 결과 뒤 기록)",
            None,
            "03_kr_fin_event",
            "가진 자산으로 이익을 얼마나 내는 회사인가",
            "문헌 +(결과 뒤 기록)",
            "SHAP 10위(0.0200) · 방향 ρ +0.92 · 단조 증가 · 안정",
            _NO_TEST,
            "값이 높은 쪽(자산 대비 이익이 큰 종목)",
        ),
        FeatureSpec(
            "fin_roe",
            1,
            "카드 기대 부호 +(문헌, 결과 뒤 기록)",
            None,
            "03_kr_fin_event",
            "주주 몫 자본 대비 이익률. 자본잠식이면 값이 거꾸로",
            "문헌 +(결과 뒤 기록)",
            "SHAP 8위(0.0253) · 방향 ρ +0.91 · 역U자 · 안정",
            _NO_TEST,
            "값이 높은 쪽(자본 대비 이익이 큰 종목). 자본잠식 종목은 따로 표시",
        ),
        FeatureSpec(
            "px_high_low_range_20d",
            1,
            "층 3 모델 반응 방향(탐색)",
            20,
            "01_kr_price",
            "최근 한 달 고가와 저가 폭이 종가의 몇 배인가",
            "미등록",
            "SHAP 5위(0.0297) · 방향 ρ +0.86 · 단조 증가 · 안정",
            _NO_TEST,
            "값이 높은 쪽(20일 고저 폭이 큰 종목) — 모델 반응 방향, 탐색. 정지 사례는 따로 표시",
        ),
        FeatureSpec(
            "flow_individual_netbuy_to_volume_20d",
            1,
            "단변량 관측 부호 +(관측)",
            20,
            "02_kr_flow",
            "20거래일 개인 순매수를 총거래량으로 나눈 비율",
            "미등록(관측 +)",
            "SHAP 15위(0.0132) · 방향 ρ +0.59 · 역U자 · 불안정",
            "scan 단변량 통과(FS1). A, IC +0.0241, 6/6 cell discovery",
            "값이 높은 쪽(개인 순매수 비율이 큰 종목) — 관측 부호, 기대 부호 미등록",
        ),
        FeatureSpec(
            "px_ret_60d",
            0,
            "좋은 쪽 정하지 않음(탐색)",
            60,
            "01_kr_price",
            "최근 석 달 오른 정도",
            "미등록",
            "SHAP 14위(0.0135) · 방향 ρ −0.87 · 단조 감소 · 불안정",
            "검정 안 함. 이웃 `px_mom_12_1`은 D등급이고 기대 방향이 아님",
            "정하지 않음. 상위 10%와 하위 10%를 같은 무게로 보여 줌",
        ),
    )
}

DEFAULT_FEATURES = tuple(FEATURES)
