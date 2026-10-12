# ruff: noqa: E501
"""KR px_* 18개 해설 페이지 설정. 다른 문서(02~06)는 이 파일을 복사해 FEATURES·THEMES·텍스트·창 모양만 바꾼다."""

DOC_DIR = "milestones/common/20261012_feature_explainer"
OUT_NAME = "01_kr_price.html"
DOC_MD = "01_kr_price.md"
RESULT_MD = "09_kr_result.md"
RUN_ID = "kr_l3_20261012T113305Z"
PANEL_REL = (
    "kr/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/"
    "FS1h_h20_lag1_rank_idxbench__holdout/feat_panel.parquet"
)
RAW_REL = "kr/raw/raw_postgres/snapshot_date=2026-09-30/source=sj2_remote"
REF_DATE = "2026-08-28"
PREFIX = "px_"
MARKET = "kr"
PAGE_TITLE = "KR 가격 피쳐 그림 해설"
GROUP_NAMES = {
    "c01": "5일 수익률·반전",
    "c02": "20일 수익률·이격도",
    "c03": "60일 수익률·모멘텀",
    "c04": "변동성",
    "c05": "회전율·비유동성·공매도",
}
MODELER_COMMIT = "24eb8e0"
N_ICE = 40
EX_DAYS = 260
EX_MARKET = "KOSPI"

THEMES = [
    (
        "수익률·반전",
        "수익률 6개. 같은 가격에서 나온 창만 다릅니다.",
        ["px_ret_1d", "px_ret_5d", "px_ret_20d", "px_ret_60d", "px_mom_20_60", "px_reversal_5d"],
    ),
    (
        "변동성",
        "널뛰기를 재는 5개. 서로 상관이 높아 한 묶음(c04)입니다.",
        ["px_vol_20d", "px_vol_60d", "px_high_low_range_20d", "px_maxret_20d", "px_idio_vol_60d"],
    ),
    (
        "유동성",
        "거래대금과 비유동성. 규모(시가총액)와 겹칩니다.",
        ["px_turnover", "px_turnover_ma20", "px_amihud_20d"],
    ),
    (
        "위치",
        "지금 가격이 평균·고점에서 얼마나 떨어져 있나.",
        ["px_gap_vs_ma20", "px_dist_52w_high"],
    ),
    ("거래정지", "수익 신호가 아니라 품질 표시입니다.", ["px_is_halted", "px_halt_ratio_20d"]),
]

# raw 값 표시 형식: pct(×100,%) / mult(배) / eok(억원) / sci / ratio / bin
FMT = {
    "px_ret_1d": "pct",
    "px_ret_5d": "pct",
    "px_ret_20d": "pct",
    "px_ret_60d": "pct",
    "px_mom_20_60": "pct",
    "px_reversal_5d": "pct",
    "px_vol_20d": "pct",
    "px_vol_60d": "pct",
    "px_high_low_range_20d": "mult",
    "px_maxret_20d": "pct",
    "px_idio_vol_60d": "pct",
    "px_turnover": "eok",
    "px_turnover_ma20": "eok",
    "px_amihud_20d": "sci",
    "px_gap_vs_ma20": "pct",
    "px_dist_52w_high": "pct",
    "px_is_halted": "bin",
    "px_halt_ratio_20d": "ratio",
}

BASIS = {
    "valid": "유효 세션 기준 · 거래정지 행을 뺀 날들 위에서 계산",
    "legacy": "원래 행 기준 (legacy) · 거래정지 행을 포함하고 기업행동 마스킹이 없음",
    "scan": "유효 세션 기준 · 기업행동이 낀 구간은 NULL (scan 피쳐)",
    "row": "당일 행 하나",
}


# 창 모양: axis=보여 줄 일수, rows=[{label, segs:[{a,b,kind,text}]}], marks=[(오프셋,글자)]
def seg(a, b, kind="use", text=""):
    return {"a": a, "b": b, "kind": kind, "text": text}


CARDS = {
    "px_ret_1d": dict(
        title="하루 수익률",
        basis="valid",
        intuition="직전 거래일 대비 오늘 얼마나 올랐거나 내렸는지입니다. 단기 반전 생각의 재료이지만 이 컬럼으로 검정한 적은 없습니다.",
        win=dict(
            axis=8,
            formula="ln( close[t] / close[t−1 유효 세션] )",
            rows=[dict(label="로그수익률 1일", segs=[seg(-1, 0, "use", "1일")])],
            marks=[(-1, "t−1"), (0, "t")],
            note="정지일 행은 ret_1d가 비어 _isna가 켜질 것으로 읽히나 실측은 안 했습니다(미확인).",
        ),
        what="어제 종가 대비 오늘 종가의 로그수익률입니다.",
        model="순위가 오를수록 점수가 내려가는 쪽이지만(ρ −0.55) 곡선 폭이 0.0038로 아주 작습니다.",
        caution="reversal_5d·ret_5d와 정보가 겹칩니다. 기대 부호가 미등록이고, 관련 가설(reversal_1d)은 부호가 반대입니다.",
    ),
    "px_ret_5d": dict(
        title="5거래일 수익률",
        basis="legacy",
        intuition="최근 일주일 동안 얼마나 올랐거나 내렸는지입니다. 많이 오른 종목은 잠깐 되밀린다는 생각의 재료입니다.",
        win=dict(
            axis=30,
            formula="ln( close[t] / close[t−5행] )",
            rows=[dict(label="로그수익률 5행", segs=[seg(-5, 0, "use", "5행")])],
            marks=[(-5, "t−5"), (0, "t")],
            note="",
        ),
        what="5행 전 종가 대비 오늘 종가의 로그수익률입니다. 정지 행을 포함한 원래 행으로 셉니다.",
        model="순위가 오를수록 점수가 내려가고(ρ −0.72) 곡선 폭은 0.0035로 작습니다. 섞기로는 불안정합니다.",
        caution="reversal_5d와 부호만 반대라 상관이 −1.00입니다(묶음 c01). 정지일이 끼면 값이 조금 왜곡됩니다.",
    ),
    "px_ret_20d": dict(
        title="20거래일 수익률",
        basis="legacy",
        intuition="최근 한 달 동안 오른 정도입니다. 추세와 반전 두 이야기가 모두 가능하고, 한국에서는 추세 쪽이 안 맞았다는 이웃 결과가 있습니다.",
        win=dict(
            axis=70,
            formula="ln( close[t] / close[t−20행] )",
            rows=[dict(label="로그수익률 20행", segs=[seg(-20, 0, "use", "20행")])],
            marks=[(-20, "t−20"), (0, "t")],
            note="",
        ),
        what="20행 전 종가 대비 오늘 종가의 로그수익률입니다.",
        model="SHAP은 12위지만 시기에 따라 달라 불안정합니다. 곡선은 U자로 분류됐고 해석하지 않습니다.",
        caution="ret_5d·ret_60d와 창이 겹치고, gap_vs_ma20과는 상관 +0.81로 같은 묶음(c02)입니다.",
    ),
    "px_ret_60d": dict(
        title="60거래일 수익률",
        basis="legacy",
        intuition="최근 석 달 동안 오른 정도입니다. 오른 종목이 계속 오른다는 중기 모멘텀 생각이 우리 한국 데이터로는 안 맞았습니다.",
        win=dict(
            axis=70,
            formula="ln( close[t] / close[t−60행] )",
            rows=[dict(label="로그수익률 60행", segs=[seg(-60, 0, "use", "60행")])],
            marks=[(-60, "t−60"), (0, "t")],
            note="",
        ),
        what="60행 전 종가 대비 오늘 종가의 로그수익률입니다.",
        model="순위가 오를수록 점수가 내려가는 단조 감소 모양(ρ −0.87)이지만 SHAP·섞기 모두 불안정합니다.",
        caution="'한국 중기 모멘텀은 반대'라는 결과는 이웃 피쳐(mom_12_1)의 것이고 이 컬럼의 결과가 아닙니다.",
    ),
    "px_mom_20_60": dict(
        title="20일 수익률 − 60일 수익률",
        basis="legacy",
        intuition="코드로 풀면 '60일 전 종가와 20일 전 종가 사이 변화의 부호 반대'입니다. 문서는 모멘텀 가속이라 불렀지만 최근 20일은 빠져 있습니다.",
        win=dict(
            axis=70,
            formula="ret_20d − ret_60d = ln( close[t−60행] / close[t−20행] )",
            rows=[
                dict(label="ret_20d = ln(c[t]/c[t−20])", segs=[seg(-20, 0, "use", "20행")]),
                dict(label="− ret_60d = ln(c[t]/c[t−60])", segs=[seg(-60, 0, "minus", "60행")]),
                dict(
                    label="결과: close[t]가 소거됨",
                    segs=[seg(-60, -20, "use", "쓰는 구간"), seg(-20, 0, "skip", "안 씀")],
                ),
            ],
            marks=[(-60, "t−60"), (-20, "t−20"), (0, "t")],
            note="최근 20일은 계산에 안 들어가고, 값이 크면 −60→−20 구간에서 가격이 내려온 종목입니다(부호 반대). 이름의 '가속'과 다릅니다.",
        ),
        what="두 로그수익률을 빼면 같은 오늘 종가가 소거되어 60행 전과 20행 전 종가의 비율만 남습니다.",
        model="순위가 오를수록 점수가 오르지만(ρ +0.76) 곡선 폭이 0.0019로 평평 기준을 겨우 넘습니다. 즉 '60→20일 전 수익률이 낮을수록 점수가 높다'입니다.",
        caution="ret_20d·ret_60d와 선형 종속이고 ret_60d와 상관 −0.78(묶음 c03)입니다. 이름과 달리 '가속'이 아닙니다.",
    ),
    "px_reversal_5d": dict(
        title="5일 단기 반전",
        basis="scan",
        intuition="최근 5일 많이 떨어진 종목이 며칠 동안 시장보다 조금 더 오른다는 생각입니다. 부호를 뒤집어 값이 클수록 많이 떨어진 종목입니다.",
        win=dict(
            axis=30,
            formula="−Σ log_ret (최근 5 유효 세션)",
            rows=[dict(label="합계에 −1을 곱함", segs=[seg(-5, 0, "use", "5 유효 세션")])],
            marks=[(-5, "t−5"), (0, "t")],
            note="기업행동이 낀 구간은 NULL입니다. ret_5d와 거의 같은 정보에 부호만 반대입니다.",
        ),
        what="최근 5개 유효 세션 로그수익률 합계에 음수를 붙인 값입니다. 기대 부호는 +입니다.",
        model="순위가 오를수록 점수가 오르지만(ρ +0.80) 폭이 0.0052로 작고 SHAP·섞기 모두 불안정해서 해석하지 않습니다.",
        caution="단변량 효과는 3일이 정점인데 모델 라벨은 20일입니다. ret_5d와 겹칩니다. 저유동 종목의 가짜 반전이 대안 설명입니다.",
    ),
    "px_vol_20d": dict(
        title="20일 실현변동성",
        basis="valid",
        intuition="최근 한 달 동안 하루하루 주가가 얼마나 들쭉날쭉했는지입니다. 값이 클수록 널뛰는 종목입니다.",
        win=dict(
            axis=70,
            formula="STDDEV_SAMP( 일 로그수익률 , 최근 20행 )",
            rows=[dict(label="일 수익률의 표본 표준편차", segs=[seg(-20, 0, "use", "20행")])],
            marks=[(-20, "t−20"), (0, "t")],
            note="공통 전제 표는 유효 세션 기준이라 적었고, 카드는 창이 정지 행을 포함한 행 기준이라 적었습니다. 두 서술이 어긋나며 실측은 안 했습니다.",
        ),
        what="일 로그수익률의 표본 표준편차입니다. 총변동성이라 시장 요인이 섞여 있습니다.",
        model="순위가 오를수록 점수가 오르고(ρ +0.89) 모양은 U자로 분류됐지만 불안정해서 해석하지 않습니다.",
        caution="vol_60d·high_low_range·maxret·idio_vol과 같은 '널뛰기' 축(묶음 c04)입니다.",
    ),
    "px_vol_60d": dict(
        title="60일 실현변동성",
        basis="valid",
        intuition="최근 석 달 동안 주가가 얼마나 들쭉날쭉했는지입니다. 20일 변동성보다 천천히 변하는 '평소 체질' 쪽 신호입니다.",
        win=dict(
            axis=70,
            formula="STDDEV_SAMP( 일 로그수익률 , 최근 60행 )",
            rows=[dict(label="일 수익률의 표본 표준편차", segs=[seg(-60, 0, "use", "60행")])],
            marks=[(-60, "t−60"), (0, "t")],
            note="",
        ),
        what="일 로그수익률의 표본 표준편차를 60행으로 낸 값입니다.",
        model="순위가 오를수록 점수가 오르고(ρ +0.69) U자로 분류됐지만 불안정합니다. 섞으면 IC가 오히려 올랐고 두 절반의 부호가 다릅니다.",
        caution="idio_vol_60d와 같은 창입니다. 이 컬럼에 대한 기대는 문서에 없습니다.",
    ),
    "px_high_low_range_20d": dict(
        title="20일 고저폭",
        basis="legacy",
        intuition="최근 한 달 가장 높은 고가와 가장 낮은 저가의 차이가 현재 종가의 몇 배인지입니다. 한 달 오르내림의 폭을 한 숫자로 본 것입니다.",
        win=dict(
            axis=30,
            formula="( MAX(high, 20행) − MIN(low, 20행) ) / close[t]",
            rows=[
                dict(
                    label="최고 고가 − 최저 저가", segs=[seg(-20, 0, "use", "20행 (정지 행 포함)")]
                )
            ],
            marks=[(-20, "t−20"), (0, "t")],
            note="정지 행은 high=low=0이라 20행 안에 정지일이 있으면 MIN(low)=0이 되고 폭이 max(high)/close로 부풉니다.",
        ),
        what="20행 안의 (최고 고가 − 최저 저가)를 오늘 종가로 나눈 값입니다.",
        model="순위가 오를수록 점수가 오르는 단조 증가(ρ +0.86)이고 SHAP 5위로 안정합니다. 그런데 섞으면 rank IC가 오히려 올랐습니다.",
        caution="정지일이 낀 행의 rank가 평균 0.996으로 쏠리고 그 행의 평균 SHAP이 +0.0673입니다(나머지 +0.0045). 인과는 확인하지 않았습니다.",
    ),
    "px_maxret_20d": dict(
        title="20일 최대 일간수익률",
        basis="scan",
        intuition="최근 한 달 중 하루 가장 크게 오른 날의 상승률입니다. 복권처럼 튀는 종목은 과대평가되어 이후 수익이 낮다는 생각입니다.",
        win=dict(
            axis=30,
            formula="MAX( 단순 일수익률 , 최근 20 유효 세션 )",
            rows=[dict(label="가장 큰 하루 상승률", segs=[seg(-20, 0, "use", "20 유효 세션")])],
            marks=[(-20, "t−20"), (0, "t")],
            note="로그가 아니라 단순수익률입니다. 기업행동이 낀 구간은 NULL입니다.",
        ),
        what="20개 유효 세션의 단순 일수익률 가운데 최댓값입니다. 기대 부호는 −입니다.",
        model="순위가 오를수록 점수가 내려가는 쪽(ρ −0.35)이고 역U자로 분류됐지만 SHAP·섞기 모두 불안정합니다.",
        caution="상한가(±30%)에 값이 뭉칩니다. idio_vol과 같은 복권 선호를 두 번 재는지는 가른 적이 없습니다.",
    ),
    "px_idio_vol_60d": dict(
        title="60일 고유변동성",
        basis="scan",
        intuition="시장 전체가 움직여서 생긴 변동을 빼고, 그 종목만의 사정으로 널뛴 정도입니다. 이런 종목은 고평가가 오래 남아 이후 수익이 낮다는 생각입니다.",
        win=dict(
            axis=270,
            formula="STDEV( 잔차 , 최근 60행 ),  잔차 = 종목 수익률 − 시장모형 예측",
            rows=[
                dict(
                    label="시장모형 적합 (각 날 직전 252세션)",
                    segs=[seg(-252, 0, "fit", "252세션")],
                ),
                dict(label="잔차의 표준편차", segs=[seg(-60, 0, "use", "60행")]),
                dict(
                    label="계산 조건: 최근 126행에 잔차 126개 모두",
                    segs=[seg(-126, 0, "need", "126행")],
                ),
            ],
            marks=[(-252, "t−252"), (-126, "t−126"), (-60, "t−60"), (0, "t")],
            note="개념도입니다. 시장모형은 날마다 직전 세션으로 다시 적합하고(미래 정보 없음), 기업행동 구간은 NULL입니다.",
        ),
        what="시장 대비 회귀의 잔차를 뽑아 그 표준편차를 60행으로 낸 값입니다. 시장과 같이 움직인 부분은 빠집니다.",
        model="순위가 오를수록 점수가 내려가는 단조 감소(ρ −0.96)이고 SHAP 1위입니다. 섞으면 rank IC가 0.0547 줄어, 기대 부호와 같고 두 절반 모두 안정입니다.",
        caution="같은 묶음 c04를 한꺼번에 섞으면 IC 하락이 0.0119로 단독 값보다 작습니다. 이유는 확인하지 않았습니다. 진단이지 판정이 아닙니다.",
    ),
    "px_turnover": dict(
        title="거래대금",
        basis="valid",
        intuition="오늘 종가에 거래량을 곱한 값으로, 그날 얼마나 많은 돈이 오갔는지의 근사입니다. 큰 종목일수록 큽니다.",
        win=dict(
            axis=8,
            formula="close[t] × volume[t]",
            rows=[dict(label="오늘 하루", segs=[seg(-1, 0, "use", "당일")])],
            marks=[(0, "t")],
            note="",
        ),
        what="종가×거래량입니다. 실제 거래대금이 아니라 근사입니다.",
        model="순위가 오를수록 점수가 내려가는 역U자(ρ −0.45)로 분류됐지만 SHAP·섞기 모두 불안정해서 해석하지 않습니다.",
        caution="사전 설계가 '단독 입력 금지'라 한 형태가 그대로 들어 있습니다. 시가총액·amihud와 정보가 겹칩니다.",
    ),
    "px_turnover_ma20": dict(
        title="20일 평균 거래대금",
        basis="legacy",
        intuition="최근 한 달 하루 평균 거래대금입니다. 하루의 들쭉날쭉함을 덜어 낸 '평소 거래 규모'입니다.",
        win=dict(
            axis=70,
            formula="AVG( close × volume , 최근 20행 )",
            rows=[dict(label="거래대금의 평균", segs=[seg(-20, 0, "use", "20행")])],
            marks=[(-20, "t−20"), (0, "t")],
            note="",
        ),
        what="최근 20행 거래대금의 평균입니다.",
        model="순위가 오를수록 점수가 내려가는 단조 감소(ρ −0.94)이고 SHAP 2위로 안정입니다. 섞기는 앞 절반 음, 뒤 절반 양으로 불안정합니다.",
        caution="유니버스 필터(60일 평균 거래대금 1억 원 이상)와 같은 축이라 학습 표본이 이 값으로 잘려 있습니다.",
    ),
    "px_amihud_20d": dict(
        title="Amihud 비유동성",
        basis="valid",
        intuition="거래된 돈에 비해 가격이 얼마나 많이 움직이는지입니다. 값이 클수록 사고팔기 어려운 종목이고, 그 불편의 대가로 수익이 높다는 생각입니다.",
        win=dict(
            axis=70,
            formula="AVG( |ret_1d| / turnover , 최근 20 유효 세션 )",
            rows=[
                dict(label="수익률 ÷ 거래대금의 평균", segs=[seg(-20, 0, "use", "20 유효 세션")])
            ],
            marks=[(-20, "t−20"), (0, "t")],
            note="turnover가 0인 날은 평균에서 뺍니다(NULLIF). 기업행동 마스킹은 없습니다.",
        ),
        what="하루 수익률 절댓값을 거래대금으로 나눈 값의 20일 평균입니다. 기대 부호는 +입니다.",
        model="순위가 오를수록 점수가 오르고(ρ +0.73) U자로 분류됐지만 SHAP·섞기 모두 불안정해서 기대 부호와 비교하지 않았습니다.",
        caution="사실상 시가총액을 뒤집은 값일 수 있습니다(시총과 순위상관 −0.754). 비유동성 프리미엄과 소형주 효과를 구분하지 못합니다.",
    ),
    "px_gap_vs_ma20": dict(
        title="20일 이동평균 이격도",
        basis="legacy",
        intuition="지금 가격이 최근 한 달 평균 가격보다 몇 퍼센트 위(또는 아래)에 있는지입니다. 이 컬럼에 대한 가설 문서는 없습니다.",
        win=dict(
            axis=70,
            formula="close[t] / AVG( close , 최근 20행 ) − 1",
            rows=[dict(label="종가의 평균", segs=[seg(-20, 0, "use", "20행")])],
            marks=[(-20, "t−20"), (0, "t")],
            note="",
        ),
        what="오늘 종가를 최근 20행 평균 종가로 나눈 값에서 1을 뺀 것입니다.",
        model="순위가 오를수록 점수가 내려가는 쪽(ρ −0.95)이지만 곡선 폭이 0.0062로 작고 SHAP은 불안정입니다.",
        caution="사실상 최근 20일 수익률과 비슷한 정보라 ret_20d와 상관 +0.81(묶음 c02)입니다. 기대 근거는 없습니다.",
    ),
    "px_dist_52w_high": dict(
        title="52주 고점 대비 거리",
        basis="legacy",
        intuition="지금 가격이 1년 내 최고 종가에서 얼마나 내려와 있는지입니다. 0이면 신고가이고 음수일수록 고점에서 멀리 떨어진 종목입니다.",
        win=dict(
            axis=270,
            formula="close[t] / MAX( close , 252행 ) − 1   (항상 0 이하)",
            rows=[dict(label="종가의 최댓값", segs=[seg(-252, 0, "use", "252행")])],
            marks=[(-252, "t−252"), (0, "t")],
            note="",
        ),
        what="오늘 종가를 252행 최고 종가로 나눈 값에서 1을 뺀 것입니다.",
        model="순위가 오를수록 점수가 오르지만(ρ +0.62) 곡선은 U자 모양입니다. SHAP 4위, 섞으면 IC가 0.0120 줄어 안정입니다.",
        caution="px_near_52w_high와 '같은 산식'이라는 문서와 아니라는 문서가 갈립니다. 코드는 다릅니다. 그래서 단변량 A등급을 옮기지 않았습니다.",
    ),
    "px_is_halted": dict(
        title="거래정지일 표시",
        basis="row",
        intuition="오늘이 거래정지로 아예 거래가 안 된 날인지 알려 주는 0/1 표시입니다. 수익 신호가 아니라 이 날 가격을 믿기 어렵다는 표시입니다.",
        win=dict(
            axis=8,
            formula="open = 0 AND high = 0 AND low = 0 이면 1",
            rows=[dict(label="오늘 행 하나", segs=[seg(-1, 0, "use", "당일")])],
            marks=[(0, "t")],
            note="순위를 매기지 않는 이진 컬럼입니다. 학습 패널에 값이 1인 행이 있는지는 확인하지 못했습니다.",
        ),
        what="그날 시가·고가·저가가 모두 0이면 참입니다.",
        model="SHAP이 전부 0이고 곡선도 평평합니다. 모델은 이 열을 사실상 쓰지 않습니다. 방향 ρ는 계산되지 않았습니다.",
        caution="수익 근거가 아니라 품질 표시입니다. 다른 px_*는 정지일에 계산되지 않는 구조라고 문서가 적었습니다.",
    ),
    "px_halt_ratio_20d": dict(
        title="최근 20일 중 정지일 비율",
        basis="legacy",
        intuition="최근 한 달 중 거래정지가 며칠이었는지의 비율입니다. 자주 멈추는 종목은 가격 숫자가 왜곡되고 거래 위험도 큽니다.",
        win=dict(
            axis=70,
            formula="AVG( is_halted , 최근 20행 )  (0~1)",
            rows=[dict(label="정지일 수 ÷ 20", segs=[seg(-20, 0, "use", "20행")])],
            marks=[(-20, "t−20"), (0, "t")],
            note="대부분 0이라 순위에 동점이 매우 많을 것으로 읽히나 실측은 없습니다.",
        ),
        what="최근 20행 중 거래정지 행의 비율입니다.",
        model="순위가 오를수록 점수가 오르는 역U자(ρ +0.71)로 분류됐지만 SHAP은 불안정입니다. 곡선 20구간 중 18구간은 값이 그대로입니다. 섞기는 안정하게 음수입니다.",
        caution="수익 근거가 아니라 품질 표시입니다. 정지 종목은 모델이 점수를 매겨도 실제로 살 수 없습니다.",
    ),
}

# 예시 종목: (feature, kind) -> 자동 선택 (시장 내 백분위 ±0.01, 거래대금 큰 순). halt는 20행 안 정지일이 있는 종목.
EXAMPLES = {
    "px_mom_20_60": [dict(kind="p95"), dict(kind="p05")],
    "px_high_low_range_20d": [
        dict(kind="p95"),
        dict(kind="halt", ticker="010780"),
        dict(kind="p05"),
    ],
    "px_idio_vol_60d": [dict(kind="p95"), dict(kind="p05")],
    "px_dist_52w_high": [dict(kind="p95"), dict(kind="p05")],
}
# 예시 창(오프셋, 행 기준; 0=기준일) — 창 그림과 같은 정의
EX_WINDOWS = {
    "px_mom_20_60": [
        dict(a=-60, b=-20, kind="use", text="쓰는 구간"),
        dict(a=-20, b=0, kind="skip", text="안 씀 (최근 20행)"),
    ],
    "px_high_low_range_20d": [dict(a=-19, b=0, kind="use", text="최근 20행")],
    "px_idio_vol_60d": [dict(a=-59, b=0, kind="use", text="60행 창")],
    "px_dist_52w_high": [dict(a=-251, b=0, kind="use", text="252행 창")],
}


# ---------- 쉬운 비유(analogy)·원리 그림(principle) 추가 (2026-10-12) ----------
# 카드의 기존 글·숫자는 그대로 두고 두 칸만 얹는다. 예시 숫자는 전부 '가정'이다.
def _ret_tl(label_old, sub_old, n_text, ret_name, caption):
    return dict(
        type="timeline",
        title="어느 두 종가를 비교하나",
        events=[
            dict(at=-n_text, label=label_old, sub=sub_old),
            dict(at=0, label="오늘 종가", sub="c[t]", kind="today"),
        ],
        spans=[dict(a=-n_text, b=0, label=ret_name)],
        caption=caption,
    )


def _flow3(title, steps, measure):
    return dict(type="flow", title=title, steps=steps, measure=measure)


_ADD = {
    "px_ret_1d": dict(
        analogy="어제 마감 가격표와 오늘 마감 가격표를 나란히 놓고 얼마나 달라졌는지 보는 것과 같습니다.",
        principle=_ret_tl(
            "직전 유효 세션 종가",
            "c[t−1]",
            1,
            "ln( c[t] / c[t−1] )",
            "두 날의 종가만 씁니다. 그 사이 장중 움직임은 들어가지 않습니다.",
        ),
    ),
    "px_ret_5d": dict(
        analogy="지난주 이맘때 가격표와 오늘 가격표를 비교하는 것과 같습니다. 그 사이 오르내림은 보지 않습니다.",
        principle=_ret_tl(
            "5행 전 종가",
            "c[t−5]",
            5,
            "ln( c[t] / c[t−5] )",
            "두 날의 종가만 씁니다. 5일 사이의 오르내림 경로는 값에 안 남습니다.",
        ),
    ),
    "px_ret_20d": dict(
        analogy="한 달 전 가격표와 오늘 가격표를 비교하는 것과 같습니다. 그 사이 오르내림은 보지 않습니다.",
        principle=_ret_tl(
            "20행 전 종가",
            "c[t−20]",
            20,
            "ln( c[t] / c[t−20] )",
            "두 날의 종가만 씁니다. 20일 사이의 오르내림 경로는 값에 안 남습니다.",
        ),
    ),
    "px_ret_60d": dict(
        analogy="석 달 전 가격표와 오늘 가격표를 비교하는 것과 같습니다. 그 사이 오르내림은 보지 않습니다.",
        principle=_ret_tl(
            "60행 전 종가",
            "c[t−60]",
            60,
            "ln( c[t] / c[t−60] )",
            "두 날의 종가만 씁니다. 60일 사이의 오르내림 경로는 값에 안 남습니다.",
        ),
    ),
    "px_mom_20_60": dict(
        analogy="석 달 전 가격에서 한 달 전 가격까지 얼마나 변했는지만 보고, 방향을 뒤집어 적은 성적표와 같습니다. 최근 한 달은 가려져 있습니다.",
        principle=dict(
            type="timeline",
            title="오늘 종가가 지워지면 남는 구간",
            events=[
                dict(at=-60, label="60행 전 종가", sub="c[t−60]"),
                dict(at=-20, label="20행 전 종가", sub="c[t−20]"),
                dict(at=0, label="오늘 종가", sub="c[t] (양쪽에서 소거)", kind="today"),
            ],
            spans=[
                dict(a=-60, b=-20, label="쓰는 구간: ln( c[t−60] / c[t−20] )"),
                dict(a=-20, b=0, label="안 씀"),
            ],
            caption="ret_20d − ret_60d = ln(c[t]/c[t−20]) − ln(c[t]/c[t−60]) = ln(c[t−60]/c[t−20]). 오늘 종가가 양쪽에서 지워져 최근 20행은 안 들어갑니다. 값이 크면 60→20행 전 구간에서 가격이 내려온 종목입니다. 가속이 아닙니다.",
        ),
    ),
    "px_reversal_5d": dict(
        analogy="일주일 동안 값이 얼마나 깎였는지를 '할인폭'으로 적는 것과 같습니다. 많이 깎일수록 큰 숫자가 됩니다.",
        principle=_flow3(
            "부호를 뒤집어 '많이 떨어짐'을 큰 값으로",
            [
                dict(
                    title="5일 동안 많이 떨어짐",
                    text="최근 5 유효 세션 로그수익률을 더하면 음수가 큽니다.",
                ),
                dict(
                    title="부호를 뒤집음",
                    text="합계에 −1을 곱합니다. 예시 숫자(가정): 합계 −0.08이면 +0.08이 됩니다.",
                ),
                dict(
                    title="값이 큼",
                    text="많이 떨어진 종목일수록 큰 값입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            "값이 크다는 것은 최근 5일 많이 떨어졌다는 뜻입니다. 이후에 오른다는 뜻은 아닙니다.",
        ),
    ),
    "px_vol_20d": dict(
        analogy="한 달 동안 매일 체중을 재서 평균 체중이 아니라 얼마나 오르락내리락했는지를 보는 것과 같습니다.",
        principle=_flow3(
            "일 수익률이 변동성이 되기까지",
            [
                dict(title="일 수익률", text="최근 20행의 일 로그수익률을 모읍니다."),
                dict(
                    title="평균에서 벗어난 정도",
                    text="각 날의 값이 평균에서 얼마나 떨어져 있는지 봅니다.",
                ),
                dict(
                    title="표본 표준편차",
                    text="벗어난 정도를 하나로 묶습니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            "값이 클수록 하루하루가 널뛴 종목입니다. 시장 요인이 섞인 총변동성입니다.",
        ),
    ),
    "px_vol_60d": dict(
        analogy="석 달 동안 매일 체중을 재서 얼마나 오르락내리락했는지 보는 것과 같습니다. 한 달 치보다 평소 체질에 가깝습니다.",
        principle=_flow3(
            "일 수익률이 변동성이 되기까지",
            [
                dict(title="일 수익률", text="최근 60행의 일 로그수익률을 모읍니다."),
                dict(
                    title="평균에서 벗어난 정도",
                    text="각 날의 값이 평균에서 얼마나 떨어져 있는지 봅니다.",
                ),
                dict(
                    title="표본 표준편차",
                    text="벗어난 정도를 하나로 묶습니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            "창이 60행이라 20일 변동성보다 천천히 변합니다.",
        ),
    ),
    "px_high_low_range_20d": dict(
        analogy="파도의 가장 높은 곳과 가장 낮은 곳의 높이 차를 재는 것과 같습니다. 그런데 정지일은 가장 낮은 곳이 바닥(0)으로 찍혀 파도가 거대해 보입니다.",
        principle=dict(
            type="ratio",
            title="고저폭 = (최고 고가 − 최저 저가) ÷ 종가 (예시 숫자, 가정)",
            unit="",
            rows=[
                dict(
                    label="정지일이 없는 20행",
                    numLabel="최고 고가 − 최저 저가 (108 − 100)",
                    num=8,
                    denLabel="오늘 종가",
                    den=104,
                    result="8 ÷ 104 ≈ 0.08배",
                ),
                dict(
                    label="20행 안에 정지일이 있는 경우: 정지 행은 low=0",
                    numLabel="최고 고가 − 최저 저가 (108 − 0)",
                    num=108,
                    denLabel="오늘 종가",
                    den=104,
                    result="108 ÷ 104 ≈ 1.04배 → 폭이 부풉니다",
                ),
            ],
            caption="실측(09 §5.2): 20일 안에 정지가 있는 247행은 이 피쳐 rank 평균이 0.996이고 평균 SHAP이 +0.0673입니다(나머지 +0.0045). 인과는 확인하지 않았습니다. 위 숫자는 원리용 가정입니다.",
        ),
    ),
    "px_maxret_20d": dict(
        analogy="한 달 중 가장 크게 뛴 하루만 적어 두는 '최고 기록'과 같습니다. 나머지 날이 조용했는지는 보지 않습니다.",
        principle=_flow3(
            "하루 상승률 20개에서 최댓값 하나",
            [
                dict(
                    title="일 수익률 20개",
                    text="최근 20 유효 세션의 단순 일수익률입니다. 예시 숫자(가정): +1%, −0.5%, +12%, …",
                ),
                dict(
                    title="가장 큰 값 고르기",
                    text="그중 가장 큰 하루를 고릅니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(title="그 값이 피쳐", text="예시라면 12%가 이 종목의 값입니다."),
            ],
            "하루만 크게 뛰어도 값이 큽니다. 상한가(±30%)에 값이 뭉칩니다.",
        ),
    ),
    "px_idio_vol_60d": dict(
        analogy="항구의 배들이 밀물에 같이 오르내린 몫은 빼고, 배 한 척이 따로 흔들린 정도만 재는 것과 같습니다.",
        principle=_flow3(
            "시장 몫을 빼고 남은 흔들림",
            [
                dict(title="종목 수익률", text="그 종목의 일 수익률입니다."),
                dict(
                    title="시장으로 설명되는 부분 빼기",
                    text="직전 252세션으로 적합한 시장모형의 예측을 뺍니다. 남은 것이 잔차입니다.",
                ),
                dict(
                    title="남은 흔들림의 표준편차",
                    text="잔차의 표본 표준편차, 최근 60행입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            "크면 그 종목만의 사정으로 널뛴 종목, 작으면 시장을 따라 움직이는 조용한 종목입니다. 최근 126행에 잔차가 모두 있어야 계산됩니다.",
        ),
    ),
    "px_turnover": dict(
        analogy="장터에서 오늘 오간 돈의 총액을 '가격표 × 팔린 개수'로 어림하는 것과 같습니다.",
        principle=_flow3(
            "종가 × 거래량",
            [
                dict(title="종가", text="오늘 종가 close[t]입니다."),
                dict(title="거래량", text="오늘 거래량 volume[t]입니다."),
                dict(title="곱하기", text="둘을 곱한 값입니다.", hl=True, tag="여기서 재는 값"),
            ],
            "실제 거래대금이 아니라 근사입니다.",
        ),
    ),
    "px_turnover_ma20": dict(
        analogy="하루 매출 대신 한 달 평균 일 매출로 가게 크기를 보는 것과 같습니다. 튀는 날이 평균에 묻힙니다.",
        principle=_flow3(
            "하루 거래대금이 평소 규모가 되기까지",
            [
                dict(title="하루 거래대금", text="종가 × 거래량을 날마다 구합니다."),
                dict(title="최근 20행 모으기", text="20일 치 값을 늘어놓습니다."),
                dict(title="평균", text="20개를 평균합니다.", hl=True, tag="여기서 재는 값"),
            ],
            "하루 튀는 날의 영향이 줄어든 평소 거래 규모입니다.",
        ),
    ),
    "px_amihud_20d": dict(
        analogy="붐비는 시장에서는 큰돈이 오가도 가격이 덜 움직이고, 한적한 시장에서는 작은 거래에도 가격이 출렁입니다. 그 차이를 숫자로 잰 것과 같습니다.",
        principle=dict(
            type="ratio",
            title="하루 비유동성 = |수익률| ÷ 거래대금 (예시 숫자, 가정)",
            unit="",
            rows=[
                dict(
                    label="거래가 붐비는 종목",
                    numLabel="하루 수익률 절댓값(%)",
                    num=4,
                    denLabel="거래대금(억 원)",
                    den=50,
                    result="4 ÷ 50 = 0.08 → 같은 움직임에 값이 작습니다",
                ),
                dict(
                    label="거래가 한적한 종목",
                    numLabel="하루 수익률 절댓값(%)",
                    num=4,
                    denLabel="거래대금(억 원)",
                    den=5,
                    result="4 ÷ 5 = 0.8 → 값이 큽니다",
                ),
            ],
            caption="실제 피쳐는 이 하루 값을 최근 20 유효 세션에서 평균한 것입니다(거래대금이 0인 날은 뺍니다). 값이 클수록 사고팔기 어려운 종목입니다.",
        ),
    ),
    "px_gap_vs_ma20": dict(
        analogy="한 달 평균 기온과 오늘 기온을 견줘 '평소보다 얼마나 더운가'를 보는 것과 같습니다.",
        principle=dict(
            type="ratio",
            title="이격도 = 종가 ÷ 20일 평균 − 1 (예시 숫자, 가정)",
            unit="",
            rows=[
                dict(
                    label="평균보다 위",
                    numLabel="오늘 종가",
                    num=108,
                    denLabel="최근 20행 평균 종가",
                    den=100,
                    result="108 ÷ 100 − 1 = +0.08 (평균보다 8% 위)",
                ),
                dict(
                    label="평균보다 아래",
                    numLabel="오늘 종가",
                    num=95,
                    denLabel="최근 20행 평균 종가",
                    den=100,
                    result="95 ÷ 100 − 1 = −0.05 (평균보다 5% 아래)",
                ),
            ],
        ),
    ),
    "px_dist_52w_high": dict(
        analogy="1년 중 가장 높이 올라갔던 곳의 높이에서 지금 서 있는 곳이 얼마나 내려와 있는지 재는 것과 같습니다.",
        principle=dict(
            type="ratio",
            title="고점 대비 거리 = 종가 ÷ 52주 최고 종가 − 1 (예시 숫자, 가정)",
            unit="",
            rows=[
                dict(
                    label="신고가인 날",
                    numLabel="오늘 종가",
                    num=200,
                    denLabel="252행 최고 종가",
                    den=200,
                    result="200 ÷ 200 − 1 = 0 (고점 그대로)",
                ),
                dict(
                    label="고점에서 내려온 날",
                    numLabel="오늘 종가",
                    num=150,
                    denLabel="252행 최고 종가",
                    den=200,
                    result="150 ÷ 200 − 1 = −0.25 (고점보다 25% 아래)",
                ),
            ],
            caption="항상 0 이하입니다. 0에 가까울수록 고점 근처입니다.",
        ),
    ),
    "px_is_halted": dict(
        analogy="가게가 문을 닫은 날 장부에 '휴업' 도장을 찍는 것과 같습니다. 그날 가격표는 믿기 어렵다는 표시입니다.",
        principle=_flow3(
            "거래정지가 0/1 표시가 되기까지",
            [
                dict(title="거래정지일", text="그날은 거래가 안 됩니다."),
                dict(title="가격이 0인 행", text="시가·고가·저가가 모두 0인 행으로 남습니다."),
                dict(
                    title="표시 = 1",
                    text="시가=고가=저가=0이면 1입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            "수익 신호가 아니라 품질 표시입니다.",
        ),
    ),
    "px_halt_ratio_20d": dict(
        analogy="한 달 달력에서 가게가 문 닫은 날에 동그라미를 치고 몇 칸인지 세는 것과 같습니다.",
        principle=_flow3(
            "정지일 표시가 비율이 되기까지",
            [
                dict(title="거래정지 표시", text="최근 20행의 is_halted(0/1)를 모읍니다."),
                dict(title="정지일 세기", text="1인 날이 며칠인지 셉니다. 예시 숫자(가정): 3일."),
                dict(
                    title="20으로 나눔",
                    text="평균하면 3 ÷ 20 = 0.15입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            "0이면 한 번도 안 멈췄고 1에 가까울수록 자주 멈춘 종목입니다.",
        ),
    ),
}
for _k, _v in _ADD.items():
    CARDS[_k].update(_v)
assert set(_ADD) == set(CARDS) and len(_ADD) == 18
