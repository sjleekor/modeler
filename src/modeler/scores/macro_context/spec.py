"""거시 흐름 섹션의 지표 정의와 고정 상수. 표준 라이브러리만 쓴다.

계획: ``my/milestones/common/20261031_briefing_macro_context/README.md`` §5.1\\~§5.4, §10.2, §11.
점수·순위에 쓰지 않는 표시 전용 섹션이다.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "briefing-macro.v1"

#: 관측 행 기준 기간(세션). 1주 = 5, 1개월 = 20, 3개월 = 60, 6개월 = 120, 1년 = 250.
PERIODS: dict[str, int] = {"1주": 5, "1개월": 20, "3개월": 60, "6개월": 120, "1년": 250}
Q_FLAT = 30
Q_STRONG = 90
HISTORY_YEARS = 10
MIN_HISTORY_OBS = 20
STALE_DAILY_SESSIONS = 3
SUMMARY_MIN, SUMMARY_MAX, SUMMARY_PER_GROUP = 4, 6, 2
VS_PREV_MAX_COUNT = 20

RULES: dict[str, Any] = {
    "periods_obs": PERIODS,
    "flat_quantile_pct": Q_FLAT,
    "strong_quantile_pct": Q_STRONG,
    "history_years": HISTORY_YEARS,
    "quantile_method": "numpy.percentile linear, |change| of same period over end-dates >= D-10y",
    "zero_change": "flat",
}

#: 한국 공통 피쳐 표(common_feature_observation_raw)에서 읽는 series.
KR_IDS: tuple[str, ...] = (
    "rate_kr_cd91",
    "rate_kr_gov3y",
    "rate_kr_gov10y",
    "fx_usdkrw_ecos",
    "foreign_net_kospi_ecos",
    "foreign_net_kosdaq_ecos",
    "trdval_kospi_ecos",
    "trdval_kosdaq_ecos",
    "rate_us2y",
    "rate_us10y",
    "global_vix",
    "commodity_wti_fred",
    # collector에 곧 들어오는 FRED 사본 다섯. 표에 없으면 macro_series만 쓴다.
    "rate_us3m",
    "spread_us_baa10y",
    "fx_usd_broad",
    "commodity_brent_fred",
    "rate_us10y_real",
    "macro_cpi",
    "macro_ppi",
    "macro_m2",
    "macro_consumer_sentiment",
    "macro_unemployment_rate",
    "macro_employment_rate",
)
#: 미국 macro_series에서 읽는 series.
US_IDS: tuple[str, ...] = (
    "DGS3MO",
    "DGS2",
    "DGS10",
    "DFII10",
    "T10Y2Y",
    "BAA10Y",
    "VIXCLS",
    "DTWEXBGS",
    "DCOILWTICO",
    "DCOILBRENTEU",
    "CPIAUCSL",
    "PPIACO",
    "M2SL",
    "UNRATE",
    "PAYEMS",
    "FEDFUNDS",
)

#: 일간 지표. kind: diff(차이 %p) | pct(% 변화) | sum(기간 합) | ratio(20/250 평균비).
#: ``copy``는 한국 표의 사본 series(macro_series 마지막 날 다음 날부터 이음), ``copy_round``는 사본 반올림 자리.
DAILY: tuple[dict[str, Any], ...] = (
    dict(
        id="us_3m",
        name="미 3개월물 금리",
        group="rates",
        us="DGS3MO",
        copy="rate_us3m",
        kind="diff",
        per=["1개월", "3개월", "6개월"],
        fmt="pctlvl",
    ),
    dict(
        id="us_2y",
        name="미 2년물 금리",
        group="rates",
        us="DGS2",
        copy="rate_us2y",
        kind="diff",
        per=["1개월", "3개월", "6개월"],
        fmt="pctlvl",
    ),
    dict(
        id="us_10y",
        name="미 10년물 금리",
        group="rates",
        us="DGS10",
        copy="rate_us10y",
        kind="diff",
        per=["1개월", "3개월", "1년"],
        fmt="pctlvl",
    ),
    dict(
        id="us_10y_real",
        name="미 10년 실질금리",
        group="rates",
        us="DFII10",
        copy="rate_us10y_real",
        kind="diff",
        per=["1개월", "3개월", "1년"],
        fmt="pctlvl",
    ),
    dict(
        id="us_10y2y",
        name="미 10년-2년 금리차",
        group="rates",
        us="T10Y2Y",
        kind="diff",
        per=["3개월", "6개월"],
        fmt="pctlvl",
        spread=True,
    ),
    dict(
        id="kr_cd91",
        name="한 CD91",
        group="rates",
        kr="rate_kr_cd91",
        kind="diff",
        per=["1개월", "3개월", "6개월"],
        fmt="pctlvl",
    ),
    dict(
        id="kr_gov3y",
        name="한 국고 3년",
        group="rates",
        kr="rate_kr_gov3y",
        kind="diff",
        per=["1개월", "3개월", "6개월"],
        fmt="pctlvl",
    ),
    dict(
        id="kr_gov10y",
        name="한 국고 10년",
        group="rates",
        kr="rate_kr_gov10y",
        kind="diff",
        per=["1개월", "3개월", "6개월"],
        fmt="pctlvl",
    ),
    dict(
        id="krus_10y",
        name="한미 10년 금리차 (미-한)",
        group="rates",
        derived="krus",
        kind="diff",
        per=["3개월", "6개월"],
        fmt="pctlvl",
        spread=True,
    ),
    dict(
        id="usdkrw",
        name="원/달러",
        group="fx",
        kr="fx_usdkrw_ecos",
        kind="pct",
        per=["1주", "1개월", "3개월"],
        fmt="krw",
    ),
    dict(
        id="usd_index",
        name="달러지수 (광의)",
        group="fx",
        us="DTWEXBGS",
        copy="fx_usd_broad",
        kind="pct",
        per=["1개월", "3개월", "1년"],
        fmt="idx",
    ),
    dict(
        id="vix",
        name="VIX",
        group="risk",
        us="VIXCLS",
        copy="global_vix",
        copy_round=2,
        kind="pct",
        per=["1주", "1개월"],
        fmt="idx",
        vs_median=True,
    ),
    dict(
        id="credit_baa",
        name="신용 스프레드 (BAA-10년)",
        group="risk",
        us="BAA10Y",
        copy="spread_us_baa10y",
        kind="diff",
        per=["1주", "1개월", "3개월"],
        fmt="pctlvl",
        spread=True,
        level_hidden=True,
    ),
    dict(
        id="wti",
        name="WTI",
        group="commod",
        us="DCOILWTICO",
        copy="commodity_wti_fred",
        kind="pct",
        per=["1주", "1개월", "3개월"],
        fmt="usd",
    ),
    dict(
        id="brent",
        name="브렌트유",
        group="commod",
        us="DCOILBRENTEU",
        copy="commodity_brent_fred",
        kind="pct",
        per=["1주", "1개월", "3개월"],
        fmt="usd",
    ),
    dict(
        id="fn_kospi",
        name="외국인 순매수 KOSPI",
        group="flow",
        kr="foreign_net_kospi_ecos",
        kind="sum",
        per=["1주", "1개월", "3개월"],
        ref_per="6개월",
        fmt="flow",
    ),
    dict(
        id="fn_kosdaq",
        name="외국인 순매수 KOSDAQ",
        group="flow",
        kr="foreign_net_kosdaq_ecos",
        kind="sum",
        per=["1개월", "3개월"],
        fmt="flow",
    ),
    dict(
        id="tv_kospi",
        name="KOSPI 거래대금",
        group="flow",
        kr="trdval_kospi_ecos",
        kind="ratio",
        per=[],
        fmt="tv",
    ),
    dict(
        id="tv_kosdaq",
        name="KOSDAQ 거래대금",
        group="flow",
        kr="trdval_kosdaq_ecos",
        kind="ratio",
        per=[],
        fmt="tv",
    ),
)

GROUPS: tuple[tuple[str, str], ...] = (
    ("rates", "금리"),
    ("fx", "달러·원화"),
    ("risk", "위험 선호"),
    ("commod", "원자재"),
    ("flow", "한국 증시 자금"),
    ("macro", "물가·경기 (월간)"),
)

#: 월간 지표. measure: (라벨, 키, 단위, 방향 있음, 대표).
MONTHLY: tuple[dict[str, Any], ...] = (
    dict(
        id="us_cpi",
        name="미 CPI (계절조정)",
        src="us",
        sid="CPIAUCSL",
        value="yoy",
        vlabel="전년비",
        m=[
            ("전월비", "mom", "%", 1, 0),
            ("3개월 연율", "ann3", "%", 1, 1),
            ("전년비", "yoy", "%", 1, 0),
        ],
    ),
    dict(
        id="us_ppi",
        name="미 PPI (전 품목)",
        src="us",
        sid="PPIACO",
        value="yoy",
        vlabel="전년비",
        m=[("3개월 연율", "ann3", "%", 1, 1), ("전년비", "yoy", "%", 1, 0)],
    ),
    dict(
        id="us_m2",
        name="미 M2",
        src="us",
        sid="M2SL",
        value="yoy",
        vlabel="전년비",
        m=[("3개월 연율", "ann3", "%", 1, 1), ("전년비", "yoy", "%", 1, 0)],
    ),
    dict(
        id="us_unrate",
        name="미 실업률",
        src="us",
        sid="UNRATE",
        value="lvl",
        vlabel="수준",
        m=[("3개월 평균 변화", "d3avg", "%p", 1, 1), ("12개월 저점 대비", "low12", "%p", 0, 0)],
    ),
    dict(
        id="us_payems",
        name="미 비농업 고용",
        src="us",
        sid="PAYEMS",
        value="d1",
        vlabel="월 증감",
        m=[("월 증감", "d1", "천명", 1, 0), ("3개월 평균 증감", "avg3d1", "천명", 1, 1)],
    ),
    dict(
        id="us_fedfunds",
        name="미 기준금리 (FEDFUNDS)",
        src="us",
        sid="FEDFUNDS",
        value="lvl",
        vlabel="수준",
        m=[("3개월 변화", "d3", "%p", 1, 1), ("12개월 변화", "d12", "%p", 1, 0)],
    ),
    dict(
        id="kr_cpi",
        name="한 CPI",
        src="kr",
        sid="macro_cpi",
        value="yoy",
        vlabel="전년비",
        m=[("전년비 3개월 변화", "yoy_d3", "%p", 1, 1)],
    ),
    dict(
        id="kr_ppi",
        name="한 PPI",
        src="kr",
        sid="macro_ppi",
        value="yoy",
        vlabel="전년비",
        m=[("전년비 3개월 변화", "yoy_d3", "%p", 1, 1)],
    ),
    dict(
        id="kr_m2",
        name="한 M2",
        src="kr",
        sid="macro_m2",
        value="yoy",
        vlabel="전년비",
        m=[("3개월 연율", "ann3", "%", 1, 1), ("전년비", "yoy", "%", 1, 0)],
    ),
    dict(
        id="kr_unrate",
        name="한 실업률 (원계열)",
        src="kr",
        sid="macro_unemployment_rate",
        value="yd",
        vlabel="전년 동월 대비",
        m=[("전년 동월 대비", "yd", "%p", 1, 1)],
    ),
    dict(
        id="kr_emprate",
        name="한 고용률 (원계열)",
        src="kr",
        sid="macro_employment_rate",
        value="yd",
        vlabel="전년 동월 대비",
        m=[("전년 동월 대비", "yd", "%p", 1, 1)],
    ),
    dict(
        id="kr_sent",
        name="한 소비자심리",
        src="kr",
        sid="macro_consumer_sentiment",
        value="lvl",
        vlabel="수준(100 기준)",
        m=[("1개월 변화", "d1", "p", 1, 0), ("3개월 변화", "d3", "p", 1, 1)],
    ),
)

GROUP_ORDER = [g for g, _ in GROUPS]
GNAME = dict(GROUPS)
ORDER = {s["id"]: i for i, s in enumerate(DAILY + MONTHLY)}
DAILY_BY_ID = {s["id"]: s for s in DAILY}
MONTHLY_BY_ID = {s["id"]: s for s in MONTHLY}
#: 값에 미국 원천이 들어가는 일간 지표(헤더의 "미국 주간 수집분" 구분용).
US_SOURCE_IDS = frozenset(s["id"] for s in DAILY if "us" in s or s.get("derived"))

FLOW_PLABEL = {"1주": "5세션", "1개월": "20세션", "3개월": "60세션", "6개월": "120세션"}
MINUS = "−"
ARROW = {"up": "↑", "down": "↓", "flat": "→"}

SHORT = {
    "us_3m": "미 3개월물 금리",
    "us_2y": "미 2년물 금리",
    "us_10y": "미 10년물 금리",
    "us_10y_real": "미 10년 실질금리",
    "us_10y2y": "미 10년−2년 금리차",
    "kr_cd91": "한 CD91",
    "kr_gov3y": "한 국고 3년",
    "kr_gov10y": "한 국고 10년",
    "krus_10y": "한미 10년 금리차 (미 − 한)",
    "usdkrw": "원/달러",
    "usd_index": "달러지수 (광의)",
    "vix": "VIX",
    "credit_baa": "신용 스프레드 (BAA − 10년)",
    "wti": "WTI",
    "brent": "브렌트유",
    "fn_kospi": "외국인 순매수 KOSPI",
    "fn_kosdaq": "외국인 순매수 KOSDAQ",
    "tv_kospi": "거래대금 비율 KOSPI",
    "tv_kosdaq": "거래대금 비율 KOSDAQ",
    "us_cpi": "미 CPI",
    "us_ppi": "미 PPI",
    "us_m2": "미 M2",
    "us_unrate": "미 실업률",
    "us_payems": "미 비농업 고용",
    "us_fedfunds": "미 기준금리",
    "kr_cpi": "한 CPI",
    "kr_ppi": "한 PPI",
    "kr_m2": "한 M2",
    "kr_unrate": "한 실업률",
    "kr_emprate": "한 고용률",
    "kr_sent": "한 소비자심리",
}
DISP = dict(SHORT)
DISP.update(
    {
        "fn_kospi": "외국인 순매수 KOSPI (세션 합, 조원)",
        "fn_kosdaq": "외국인 순매수 KOSDAQ (세션 합, 조원)",
        "tv_kospi": "거래대금 비율 KOSPI (20세션 평균 ÷ 250세션 평균)",
        "tv_kosdaq": "거래대금 비율 KOSDAQ (20세션 평균 ÷ 250세션 평균)",
        "us_cpi": "미 CPI 전년비(계절조정)",
        "us_ppi": "미 PPI 전년비(전 품목)",
        "us_m2": "미 M2 전년비",
        "us_unrate": "미 실업률(수준)",
        "us_payems": "미 비농업 고용 월 증감",
        "us_fedfunds": "미 기준금리 (FEDFUNDS) 수준",
        "kr_cpi": "한 CPI 전년비",
        "kr_ppi": "한 PPI 전년비",
        "kr_m2": "한 M2 전년비",
        "kr_unrate": "한 실업률 전년 동월 대비(%p)",
        "kr_emprate": "한 고용률 전년 동월 대비(%p)",
        "kr_sent": "한 소비자심리 (100 기준)",
    }
)

#: 사본 이음 대응표(한국 표 series -> macro_series). 렌더러 부록용.
SPLICE_MAP = {s["copy"]: s["us"] for s in DAILY if "copy" in s}
