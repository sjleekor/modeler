# ruff: noqa: E501
"""04 US 모멘텀·변동성·유동성·옵션·지수편입 17개 해설 config."""

DOC_DIR = "milestones/common/20261012_feature_explainer"
OUT_NAME = "04_us_price_liquidity.html"
DOC_MD = "04_us_price_liquidity.md"
RESULT_MD = "10_us_result.md"
RUN_ID = "us_l3_20261012T112931Z"
PANEL_V2_REL = "us/datasets/us_features_v2/part.parquet"
PANEL_FLOW_REL = "us/datasets/us_features_flow_v1/part.parquet"
PRICES_GLOB = "us/derived/snapshots/prices_daily/snapshot_date=*/part.parquet"
COMPANY_META_GLOB = "us/derived/snapshots/company_meta/snapshot_date=*/part.parquet"
MARKET = "us"
REF_DATE = "2026-06-01"
PAGE_TITLE = "US 가격·유동성 피쳐 해설"
MODELER_COMMIT = "d98d5cf"
N_ICE = 40
EX_DAYS = 260

THEMES = [
    (
        "모멘텀·반전",
        "지난 며칠에서 1년까지 얼마나 올랐는지를 여러 길이로 잰 피쳐입니다.",
        ["mom_12_1", "mom_6_1", "mom_1m", "rev_1w", "max_ret_1m"],
    ),
    (
        "변동성·베타",
        "주가가 얼마나 흔들리는지를 잰 피쳐입니다. 시장과 같이 움직이는 정도(베타)도 여기에 둡니다.",
        ["rv_20", "rv_60", "idio_vol_60", "beta_252"],
    ),
    (
        "유동성·규모",
        "거래가 얼마나 활발한지, 회사가 얼마나 큰지를 잰 피쳐입니다.",
        ["log_dvol_20", "amihud_20", "mcap_rank"],
    ),
    (
        "옵션 내재변동성",
        "옵션 가격에 담긴 변동성 예상을 쓰는 피쳐입니다. 자료가 있는 종목이 1,600개 안팎이라 빈 값이 많습니다.",
        ["iv_rank", "iv_hv_spread", "iv_isna"],
    ),
    (
        "지수 편입",
        "S&P 500에 들어 있는지, 들어온 지 얼마나 됐는지를 봅니다.",
        ["sp500_member", "sp500_days_since_add"],
    ),
]
FMT = {
    "mom_12_1": "pct",
    "mom_6_1": "pct",
    "mom_1m": "pct",
    "rev_1w": "pct",
    "max_ret_1m": "pct",
    "rv_20": "pct",
    "rv_60": "pct",
    "idio_vol_60": "pct",
    "beta_252": "ratio",
    "log_dvol_20": "num",
    "amihud_20": "num",
    "mcap_rank": "int",
    "iv_rank": "ratio",
    "iv_hv_spread": "num",
    "iv_isna": "bin",
    "sp500_member": "bin",
    "sp500_days_since_add": "int",
}
BASIS = {"valid": "일봉 종가 기준 · 거래일 단위", "cal": "위키 리비전 기준 · 달력일 단위"}
GROUP_NAMES = {
    "g1": "변동성 묶음",
    "g2": "규모·유동성 묶음",
    "g3": "실적 발표일 묶음",
    "g4": "공매도 묶음",
    "g5": "주문 체결 묶음",
    "g6": "내부자 매수 묶음",
}
SCATTER_LABELS = ["mcap_rank", "rv_20", "idio_vol_60", "rev_1w", "iv_rank"]


def seg(a, b, kind="use", text=""):
    return {"a": a, "b": b, "kind": kind, "text": text}


NOTE_R = "모델 점수의 반응이지 수익 기여가 아닙니다."

CARDS = {
    "mom_12_1": dict(
        title="12개월 모멘텀 (최근 한 달 제외)",
        basis="valid",
        intuition="지난 1년 동안 많이 오른 종목이 앞으로 한두 달도 더 오르는 경향을 노린 피쳐입니다. 우리 데이터에서는 이를 뒷받침하는 단독 결과가 확인되지 않았습니다.",
        analogy="달리기에서 지난 11개월 동안 앞서 달린 선수를 눈여겨보는 것과 같습니다. 바로 직전 한 달은 일부러 빼고 봅니다.",
        win=dict(
            axis=270,
            formula="adj_close[t−21] / adj_close[t−252] − 1",
            rows=[
                dict(
                    label="수익률을 재는 구간",
                    segs=[seg(-252, -21, "use", "t−252 → t−21"), seg(-21, 0, "skip", "안 씀")],
                )
            ],
            marks=[(-252, "t−252"), (-21, "t−21"), (0, "t")],
            note="조정 종가 비율이고 거래일 기준입니다. 최근 한 달은 일부러 뺍니다. 창을 못 채우면(상장 1년 미만 등) 빈 값입니다.",
        ),
        principle=dict(
            type="timeline",
            title="가격이 언제부터 언제까지 쓰이나",
            caption="오늘 t의 값은 21거래일 전 종가까지만 씁니다. 바로 직전 한 달 움직임은 mom_1m이 따로 맡습니다.",
            events=[
                dict(at=-252, label="252거래일 전 종가", sub="기준점"),
                dict(at=-21, label="21거래일 전 종가", sub="끝점"),
                dict(at=0, label="오늘 t", sub="값이 정해지는 날", kind="today"),
            ],
            spans=[
                dict(a=-252, b=-21, label="수익률을 재는 구간 231거래일"),
                dict(a=-21, b=0, label="뺀 구간 21거래일"),
            ],
            known={"from": -252, "to": 0, "label": "모델이 아는 가격"},
        ),
        what="21거래일 전 종가가 252거래일 전 종가보다 몇 % 높은지입니다.",
        model="LightGBM은 SHAP 20위인데 불안정이라 해석하지 않습니다. Ridge는 51위로 작게 쓰고 안정이며 기대와 같은 방향입니다.",
        caution="한국 px_mom_12_1은 D등급이었지만 한 번의 관찰이고 반대 부호도 안정적이지 않았습니다(기간 2/5). 한국 반대를 결론이나 미국 부호의 근거로 쓰지 않습니다. 알려진 요인이라 발표 후 감쇠가 기본값이고, Ridge 51위는 크기(0.00034)가 1위의 약 2%라 '기대와 같음'은 정보가 적습니다. 월간 회전율이 커서 비용 부담이 크다는 우려도 문서에 있습니다.",
    ),
    "mom_6_1": dict(
        title="6개월 모멘텀 (최근 한 달 제외)",
        basis="valid",
        intuition="mom_12_1과 같은 이야기를 반년 창으로 본 것입니다. 두 피쳐는 겹치는 구간이 길어 비슷한 종목을 고를 가능성이 높습니다.",
        analogy="같은 선수를 11개월이 아니라 반년 기록으로만 보는 것과 같습니다. 최근 한 달은 역시 뺍니다.",
        win=dict(
            axis=270,
            formula="adj_close[t−21] / adj_close[t−126] − 1",
            rows=[
                dict(
                    label="수익률을 재는 구간",
                    segs=[seg(-126, -21, "use", "t−126 → t−21"), seg(-21, 0, "skip", "안 씀")],
                )
            ],
            marks=[(-126, "t−126"), (-21, "t−21"), (0, "t")],
            note="mom_12_1과 같은 종가이고 시작점만 126거래일 전입니다.",
        ),
        principle=dict(
            type="timeline",
            title="mom_12_1보다 시작점이 늦습니다",
            caption="끝점은 같고 시작점만 다릅니다. 그래서 두 피쳐가 비슷한 종목을 고를 수 있습니다.",
            events=[
                dict(at=-252, label="252거래일 전", sub="mom_12_1의 시작"),
                dict(at=-126, label="126거래일 전", sub="mom_6_1의 시작"),
                dict(at=-21, label="21거래일 전", sub="둘 다 여기까지"),
                dict(at=0, label="오늘 t", kind="today"),
            ],
            spans=[dict(a=-126, b=-21, label="mom_6_1 구간 105거래일")],
            known={"from": -252, "to": 0, "label": "모델이 아는 가격"},
        ),
        what="21거래일 전 종가가 126거래일 전 종가보다 몇 % 높은지입니다.",
        model="LightGBM은 SHAP 13위인데 불안정이라 해석하지 않습니다. Ridge는 32위, 안정이고 기대와 같은 방향입니다.",
        caution="mom_12_1과의 상관은 문서에 없습니다(미확인). 한국 D등급을 근거로 읽지 않는 점은 mom_12_1과 같습니다. Ridge 32위는 크기(0.00227)가 1위의 약 14%입니다.",
    ),
    "mom_1m": dict(
        title="최근 1개월 수익률",
        basis="valid",
        intuition="지난 한 달에 많이 오른 종목이 다음 달에 되돌림을 받는다는 단기 반전 이야기입니다. 모멘텀 두 피쳐가 뺀 바로 그 구간입니다.",
        analogy="한 달 동안 줄곧 오른 종목을 보고 '이제 좀 쉴 때가 됐다'고 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="adj_close[t] / adj_close[t−21] − 1",
            rows=[dict(label="수익률을 재는 구간", segs=[seg(-21, 0, "use", "21거래일")])],
            marks=[(-21, "t−21"), (0, "t")],
            note="mom_12_1·mom_6_1이 일부러 뺀 구간을 여기서 씁니다.",
        ),
        principle=dict(
            type="timeline",
            title="모멘텀 두 피쳐가 뺀 구간을 씁니다",
            caption="오늘에서 21거래일 전까지가 mom_1m의 몫입니다. 모멘텀 피쳐는 그 앞쪽을 봅니다.",
            events=[
                dict(at=-252, label="252거래일 전", sub="mom_12_1의 시작"),
                dict(at=-21, label="21거래일 전", sub="mom_1m의 시작"),
                dict(at=0, label="오늘 t", kind="today"),
            ],
            spans=[
                dict(a=-252, b=-21, label="mom_12_1 구간"),
                dict(a=-21, b=0, label="mom_1m 구간"),
            ],
        ),
        what="21거래일 전 종가 대비 오늘 종가가 몇 % 올랐는지입니다.",
        model="LightGBM은 SHAP 21위인데 불안정이라 해석하지 않습니다. Ridge는 47위, 안정인데 기대(−)와 반대로 설명하지 못했습니다.",
        caution="rev_1w·max_ret_1m과 같은 단기 반전 계열이라 겹칠 수 있습니다. 상관 수치는 문서에서 확인하지 못했습니다. Ridge 47위는 크기(0.00067)가 작아 '설명 못 함'은 정보가 적고, 단기 신호의 지연 민감도는 측정한 것이 없습니다.",
    ),
    "rev_1w": dict(
        title="최근 1주 수익률",
        basis="valid",
        intuition="지난 일주일 동안 크게 오른 종목은 곧 되돌린다는 아주 단기의 반전입니다. 우리 결과에서는 1차(v1 데이터)의 주가 5달러 이상 종목에서만 약한 신호가 나왔고 3차 이후에는 이어지지 않았습니다.",
        analogy="일주일 사이 확 오른 가격이 평소 자리로 돌아오려는 고무줄처럼 당겨지는 것과 같습니다.",
        win=dict(
            axis=8,
            formula="adj_close[t] / adj_close[t−5] − 1",
            rows=[dict(label="수익률을 재는 구간", segs=[seg(-5, 0, "use", "5거래일")])],
            marks=[(-5, "t−5"), (0, "t")],
            note="",
        ),
        principle=dict(
            type="flow",
            title="한 주 수익률이 값이 되기까지",
            steps=[
                dict(title="5거래일 전 종가", text="일주일 전 종가를 기준으로 잡습니다."),
                dict(
                    title="오늘 종가와 비교",
                    text="오늘 종가가 그 기준보다 몇 % 높은지 구합니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(title="순위로 바꿈", text="그날 종목끼리 비교한 백분위가 모델 입력이 됩니다."),
            ],
            measure="값이 클수록 최근 한 주에 급등한 종목입니다.",
        ),
        what="5거래일 전 종가 대비 오늘 종가의 수익률입니다.",
        model="LightGBM은 SHAP 4위, 안정이고 기대(−)와 같은 방향입니다. Ridge는 17위인데 불안정이라 해석하지 않습니다.",
        caution="소형·저가주에서는 비동기 거래나 호가 반등 때문에 반전처럼 보일 수 있다는 대안 설명이 카드에 있습니다. 1차 B의 t −2.22와 3차 SHORT t 2.39는 모두 다중검정 보정 전입니다. 반전의 수익은 호가를 받아 주는 쪽의 몫이고 월초 종가에 사는 쪽은 내는 쪽일 수 있으며, 지연 민감도는 측정하지 않았습니다.",
    ),
    "max_ret_1m": dict(
        title="한 달 중 최대 일수익률",
        basis="valid",
        intuition="지난 한 달 사이 하루라도 크게 튄 종목, 즉 복권처럼 쓰이는 종목입니다. 우리 결과는 '피하는 쪽은 가설 수준(공식 유니버스 SHORT t 2.21, 보정 전)이고 사는 쪽으로는 못 쓴다'입니다.",
        analogy="한 달 내내 조용하다가 하루만 크게 솟은 종목은, 당첨 한 번이 눈에 띄는 복권 같은 종목입니다.",
        win=dict(
            axis=30,
            formula="MAX( 일 수익률 , 21일 )",
            rows=[
                dict(label="일 수익률 21개 중 최댓값", segs=[seg(-20, 0, "use", "21개(당일 포함)")])
            ],
            marks=[(-20, "t−20"), (0, "t")],
            note="21개 관측치를 못 채우면 빈 값입니다.",
        ),
        principle=dict(
            type="flow",
            title="일 수익률 21개에서 가장 큰 하루 고르기",
            steps=[
                dict(title="일 수익률 21개", text="최근 21거래일의 하루 수익률을 구합니다."),
                dict(
                    title="가장 큰 하루",
                    text="그중 가장 높이 오른 하루의 수익률을 고릅니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(title="순위로 바꿈", text="그날 종목끼리 비교한 백분위가 모델 입력이 됩니다."),
            ],
            measure="값이 클수록 최근 한 달에 크게 튄 날이 있었던 종목입니다.",
        ),
        what="최근 21거래일 중 가장 높았던 하루 수익률입니다.",
        model="LightGBM은 SHAP 27위, 안정이고 기대(−)와 같은 방향입니다. Ridge는 49위인데 방향이 반대(+)이고 설명하지 못했습니다. 두 모델의 방향이 갈립니다.",
        caution="rv_20과 ρ 0.894로 묶입니다. 3차 v2 공식 유니버스 롱은 X(월 −0.16%, t −0.33)이고 SHORT +1.65%(t 2.21)는 문턱(44개 기준 3.254)에 못 미칩니다. 1차 A와 롱 R(월 −0.440%)은 룩어헤드가 있던 v1 데이터입니다. Ridge가 LightGBM과 반대 부호인 것은 변동성 묶음 안 상쇄일 수 있습니다.",
    ),
    "rv_20": dict(
        title="20일 실현변동성",
        basis="valid",
        intuition="최근 한 달 동안 주가가 크게 출렁인 종목입니다. 흔들림이 큰 종목이 이후 수익이 낮다는 '저변동성 이상현상'을 노린 피쳐입니다.",
        analogy="체온을 20일 동안 매일 재서 오르내림의 폭을 보는 것과 같습니다. 평균이 아니라 들쭉날쭉함이 값이 됩니다.",
        win=dict(
            axis=70,
            formula="STD( 일 수익률 , 20일 )",
            rows=[dict(label="일 수익률의 표준편차", segs=[seg(-19, 0, "use", "20일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="연율화하지 않습니다. 순위로 바꾸므로 연율화 여부는 영향이 없습니다.",
        ),
        principle=dict(
            type="flow",
            title="일 수익률이 변동성이 되기까지",
            steps=[
                dict(
                    title="일 수익률", text="매일 종가를 어제 종가와 비교해 몇 % 올랐는지 구합니다."
                ),
                dict(
                    title="평균에서 벗어난 정도",
                    text="20일 평균 수익률에서 얼마나 떨어져 있는지 봅니다.",
                ),
                dict(
                    title="표준편차",
                    text="벗어난 정도를 제곱해 평균 낸 뒤 제곱근을 씌웁니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="값이 클수록 하루하루 널뛰는 종목입니다.",
        ),
        what="일 수익률의 20일 표준편차입니다.",
        model="LightGBM은 SHAP 2위, Ridge는 10위입니다. 둘 다 안정이고 순위가 오르면 점수가 내려가는 쪽으로, 기대(−)와 같습니다.",
        caution="max_ret_1m·idio_vol_60과 강하게 묶입니다. 3차 v2 롱은 X(월 −0.24%), SHORT +1.62%(t 1.85)로 약하고 보정 전입니다. 1차 A·R은 v1 데이터입니다. 기대 부호(−)는 위험 보상이 아니라 행태·제약 후보이고 미검정입니다. Ridge는 rv_20(−)과 rv_60(+)이 상쇄합니다(묶음 합 0.01137 대 멤버 합 0.02451).",
    ),
    "rv_60": dict(
        title="60일 실현변동성",
        basis="valid",
        intuition="rv_20을 석 달 창으로 늘려 더 안정적으로 잰 것입니다.",
        analogy="체온을 20일이 아니라 60일 동안 재는 것과 같습니다. 하루 이틀 튄 값에 덜 흔들립니다.",
        win=dict(
            axis=70,
            formula="STD( 일 수익률 , 60일 )",
            rows=[dict(label="일 수익률의 표준편차", segs=[seg(-59, 0, "use", "60일")])],
            marks=[(-59, "t−59"), (0, "t")],
            note="",
        ),
        principle=dict(
            type="flow",
            title="rv_20과 같은 계산, 창만 석 달입니다",
            steps=[
                dict(title="일 수익률", text="매일 종가를 어제 종가와 비교합니다."),
                dict(
                    title="60일 표준편차",
                    text="최근 60거래일 수익률이 평균에서 벗어난 정도를 잽니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(title="순위로 바꿈", text="그날 종목끼리 비교한 백분위가 모델 입력이 됩니다."),
            ],
            measure="창이 길수록 한 번 튄 날의 영향이 작아집니다.",
        ),
        what="일 수익률의 60일 표준편차입니다.",
        model="LightGBM은 SHAP 30위, 안정인데 기대(−)와 반대로 설명하지 못했습니다. Ridge는 11위인데 불안정이라 해석하지 않습니다.",
        caution="rv_20과의 상관 수치는 문서에 없습니다(미확인). 같은 계열이라 정보가 겹칠 가능성이 높고, Ridge에서는 rv_20과 상쇄하는 쪽이라 단독 부호를 읽지 않습니다.",
    ),
    "idio_vol_60": dict(
        title="자기만의 변동성 (SPY 제거 후 60일)",
        basis="valid",
        intuition="시장 전체 움직임으로 설명되는 부분을 빼고도 남는, 그 종목 고유의 흔들림입니다. 개별 사건에 크게 흔들리는 종목을 가려냅니다.",
        analogy="파도가 센 날에는 모든 배가 흔들립니다. 그 파도를 빼고도 혼자 더 흔들리는 배를 찾는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="sqrt( Var(r) − β² · Var(m) )",
            rows=[dict(label="종목과 SPY의 일 수익률 (60일)", segs=[seg(-59, 0, "use", "60일")])],
            marks=[(-59, "t−59"), (0, "t")],
            note="β = Cov(r, m) / Var(m)이고 m은 SPY 일 수익률입니다. 회귀 대신 분산 항등식으로 계산합니다.",
        ),
        principle=dict(
            type="flow",
            title="시장 몫을 빼고 남는 흔들림",
            steps=[
                dict(
                    title="종목과 SPY 수익률",
                    text="60일 동안 종목 수익률 r과 SPY 수익률 m을 함께 봅니다.",
                ),
                dict(
                    title="시장 몫 계산",
                    text="β = Cov(r, m) / Var(m)으로 시장에 딸려 가는 정도를 구합니다.",
                ),
                dict(
                    title="남는 흔들림",
                    text="전체 분산에서 시장 몫 β²·Var(m)을 빼고 제곱근을 씌웁니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="값이 클수록 시장과 무관하게 혼자 흔들리는 종목입니다.",
        ),
        what="SPY 움직임을 걷어낸 뒤 남는 60일 변동성입니다.",
        model="LightGBM은 SHAP 3위, Ridge는 4위입니다. 둘 다 안정이고 기대(−)와 같은 방향입니다.",
        caution="유니버스에 따라 R과 B로 갈린 사례입니다. rv_20과 ρ 0.863으로 묶입니다. 3차 v2 롱은 X, SHORT +1.80%(t 2.10)입니다. 한국 고유변동성은 K-7에서 강건하지 않아, 한국 A등급을 미국 부호의 근거로 읽지 않습니다.",
    ),
    "beta_252": dict(
        title="시장 베타 (252일)",
        basis="valid",
        intuition="시장(SPY)이 1% 움직일 때 이 종목이 평균 얼마나 같이 움직였는지입니다. 기대 부호가 가장 약한 피쳐입니다.",
        analogy="강물이 흐를 때 배가 물살을 얼마나 타고 가는지를 재는 것과 같습니다. 베타가 크면 시장이 움직일 때 더 크게 따라 움직입니다.",
        win=dict(
            axis=270,
            formula="Cov(r, m) / Var(m)",
            rows=[dict(label="종목과 SPY의 일 수익률", segs=[seg(-251, 0, "use", "252일")])],
            marks=[(-251, "t−251"), (0, "t")],
            note="상장 1년 미만 종목은 창을 못 채워 빈 값입니다.",
        ),
        principle=dict(
            type="flow",
            title="베타가 되기까지",
            steps=[
                dict(
                    title="일 수익률 두 줄",
                    text="252거래일 동안 종목 r과 SPY m의 일 수익률을 모읍니다.",
                ),
                dict(
                    title="같이 움직인 정도",
                    text="두 수익률이 함께 움직인 정도 Cov(r, m)을 구합니다.",
                ),
                dict(
                    title="시장 변동으로 나눔",
                    text="Cov(r, m)을 SPY의 분산 Var(m)으로 나눕니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="1보다 크면 시장보다 크게, 작으면 작게 움직인 종목입니다.",
        ),
        what="252거래일 동안 시장 움직임에 종목이 따라 움직인 정도입니다.",
        model="LightGBM(SHAP 19위)과 Ridge(16위) 모두 불안정이라 해석하지 않습니다.",
        caution="기대 부호 자체가 가장 약합니다. 3차 v2 롱은 X(월 −0.68%, t −1.51)이고 1차(v1)는 R이며 A·B 명단에는 없습니다.",
    ),
    "log_dvol_20": dict(
        title="20일 평균 거래대금(로그)",
        basis="valid",
        intuition="하루에 얼마나 많은 돈이 거래되는 종목인가입니다. 한때 가장 좋은 신호처럼 보였지만 3차에서 입력 오류가 만든 가짜로 판명됐습니다.",
        analogy="가게의 하루 평균 매출을 보는 것과 같습니다. 매출이 큰 가게일수록 손님이 많은 큰 가게입니다.",
        win=dict(
            axis=70,
            formula="log( MEAN( 거래대금 , 20일 ) )",
            rows=[dict(label="조정 거래대금의 평균", segs=[seg(-19, 0, "use", "20일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="거래대금이 0 이하인 날은 평균에서 뺍니다.",
        ),
        principle=dict(
            type="flow",
            title="거래대금이 값이 되기까지",
            steps=[
                dict(
                    title="하루 거래대금",
                    text="종가와 거래량으로 만든 조정 거래대금을 날마다 구합니다.",
                ),
                dict(title="20일 평균", text="최근 20거래일 평균을 냅니다. 0 이하인 날은 뺍니다."),
                dict(
                    title="로그",
                    text="큰 값과 작은 값의 격차를 줄이려고 로그를 씌웁니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="값이 클수록 거래가 활발한 종목입니다.",
        ),
        what="최근 20거래일 하루 평균 거래대금의 로그입니다.",
        model="LightGBM은 SHAP 39위로 작게 씁니다. Ridge는 1위로 가장 크게 기대며, 둘 다 안정이고 기대(−)와 같은 방향입니다.",
        caution="후속 조사에서 롱 A로 보인 신호는 3차에서 부호가 뒤집혀 X가 됐습니다. 유니버스 룩어헤드가 만든 가짜였습니다. 공식 유니버스 롱은 X(월 −0.51%, t −2.26)이고 뒤집힌 −0.988%는 참고 유니버스 값입니다. 타깃 L2가 이 축을 걷어냈고 Ridge는 amihud_20과 계수가 모두 음이라 상쇄하므로, Ridge 1위의 '기대와 같음'은 검증이 아닙니다.",
    ),
    "amihud_20": dict(
        title="아미후드 비유동성",
        basis="valid",
        intuition="거래대금 1달러당 주가가 얼마나 움직이는지입니다. 값이 크면 얇은 거래에도 주가가 잘 밀리는, 사고팔기 어려운 종목입니다.",
        analogy="작은 구멍가게에서는 물건을 한 번만 사도 가격이 들썩이지만, 큰 마트는 같은 양을 사도 그대로인 것과 같습니다.",
        win=dict(
            axis=70,
            formula="MEAN( |일 수익률| / 거래대금 , 20일 )",
            rows=[dict(label="|수익률| ÷ 거래대금의 평균", segs=[seg(-19, 0, "use", "20일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="",
        ),
        principle=dict(
            type="ratio",
            title="같은 가격 변화, 다른 거래대금 (예시 숫자)",
            unit="",
            rows=[
                dict(
                    label="거래가 얇은 종목",
                    numLabel="|수익률|(%)",
                    num=4,
                    denLabel="거래대금($M)",
                    den=2,
                    result="4 ÷ 2 = 2.0 → 값이 큼 (값이 잘 밀림)",
                ),
                dict(
                    label="거래가 두터운 종목",
                    numLabel="|수익률|(%)",
                    num=4,
                    denLabel="거래대금($M)",
                    den=40,
                    result="4 ÷ 40 = 0.1 → 값이 작음",
                ),
            ],
        ),
        what="하루 수익률의 절댓값을 거래대금으로 나눈 값의 20일 평균입니다.",
        model="LightGBM은 SHAP 43위로 작게 쓰지만 Ridge는 2위로 크게 기댑니다. 둘 다 안정이고 기대(+)와 반대로 설명하지 못했습니다.",
        caution="1차에서 R이었습니다. 거래대금·시총 순위가 8년에 걸쳐 서서히 움직이는 추세 때문에 시간 placebo에서 떨어졌습니다. 타깃이 거래대금 축을 걷어냈고 log_dvol_20과 같은 정보를 반대 방향으로 담아 상쇄하므로 Ridge 2위의 '설명 못 함'은 개별 판독용이 아닙니다. 보상이면서 곧 거래 비용입니다.",
    ),
    "mcap_rank": dict(
        title="시가총액 등급",
        basis="valid",
        intuition="전체 보통주를 시가총액 기준으로 10등급으로 나눴을 때 어느 등급인가입니다. 작은 회사일수록 작은 숫자입니다.",
        analogy="학교 전체를 키 순서로 세워 10개 반으로 나눈 뒤, 그 반 번호를 이름표로 다는 것과 같습니다.",
        win=dict(
            axis=8,
            formula="universe_daily.mcap_rank  (그날 값 그대로)",
            rows=[dict(label="당일 등급을 그대로 읽음", segs=[seg(-1, 0, "use", "당일")])],
            marks=[(0, "t")],
            note="새로 계산하지 않고 유니버스 표의 값을 읽습니다. 지연 없이 그날 date로 조인합니다(코드 확인). 값이 그날 시총이라 문제가 없는지는 미확인이고 서빙 10월 값은 비어 있습니다.",
        ),
        principle=dict(
            type="flow",
            title="시가총액이 등급이 되기까지",
            steps=[
                dict(title="보통주만 모음", text="MIDAS 자료에서 보통주만 대상으로 합니다."),
                dict(
                    title="시총 순으로 10등분",
                    text="그날 시총 순으로 세워 등급 1~10을 매깁니다. 각 등급이 10%입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(
                    title="등급을 그대로 입력",
                    text="등급이 10가지뿐이라 순위로 바꿔도 동점이 많습니다.",
                ),
            ],
            measure="숫자가 클수록 대형주입니다. REIT·SPAC·ADR 등은 빈 값입니다.",
        ),
        what="그날 시가총액이 10등급 중 몇 등급인지입니다.",
        model="LightGBM은 SHAP 1위, Ridge는 3위입니다. 둘 다 안정이고 기대(−)와 같은 방향입니다.",
        caution="1차에서 R이었고, 타깃이 이미 사이즈를 걷어낸 순위라 모델에서 하는 일은 층 3 결과로만 봅니다. 타깃 L2의 항에 이 순위가 들어가므로 LightGBM 1위·Ridge 3위의 '기대와 같음'은 소형주 효과의 확인이 아닙니다.",
    ),
    "iv_rank": dict(
        title="내재변동성의 1년 범위 내 위치",
        basis="valid",
        intuition="옵션 시장이 매기는 변동성이 지난 1년 중 최저와 최고 사이 어디쯤인지입니다. 1에 가까우면 시장이 평소보다 큰 흔들림을 예상하는 종목입니다.",
        analogy="1년 동안의 기온 기록에서 오늘 기온이 가장 추웠던 날과 가장 더웠던 날 사이 어디쯤인지를 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="(iv_current − 연저점) / (연고점 − 연저점)",
            rows=[
                dict(
                    label="가장 최근 보고를 이어 쓰는 한도", segs=[seg(-30, 0, "need", "최대 30일")]
                )
            ],
            marks=[(-30, "t−30"), (0, "t")],
            note="보고일이 매일 있지 않아 가장 최근 보고를 30일 안에서만 이어 씁니다. 30일은 이 모듈이 정한 값입니다. 달력일 기준일 수 있어 정확한 단위는 문서에 없습니다.",
        ),
        principle=dict(
            type="ratio",
            title="범위 안에서의 위치 (예시 숫자)",
            unit="",
            rows=[
                dict(
                    label="범위 아래쪽",
                    numLabel="현재−저",
                    num=10,
                    denLabel="범위",
                    den=40,
                    result="(현재−저) 10 ÷ (고−저) 40 = 0.25 → 1년 범위의 아래쪽",
                ),
                dict(
                    label="범위 위쪽",
                    numLabel="현재−저",
                    num=35,
                    denLabel="범위",
                    den=40,
                    result="(현재−저) 35 ÷ (고−저) 40 = 0.875 → 1년 범위의 위쪽",
                ),
            ],
        ),
        what="옵션 시장의 변동성 예상이 지난 1년 범위 중 어디에 있는지입니다.",
        model="LightGBM은 SHAP 37위, Ridge는 5위입니다. 둘 다 안정인데 기대(−)와 반대로 설명하지 못했습니다.",
        caution="옵션 자료가 있는 종목이 약 1,600개뿐이라 빈 값이 약 68%입니다. 빈 값은 무작위가 아니라 소형·비인기 종목에 몰립니다. Ridge 5위 크기의 대부분은 _isna 성분입니다(근사 약 90%, 계산 예·미검증).",
    ),
    "iv_hv_spread": dict(
        title="내재변동성 − 실현변동성",
        basis="valid",
        intuition="옵션 시장이 예상하는 변동성에서 실제로 최근 움직인 변동성을 뺀 값입니다. 다만 부호 자체가 확정되지 않았습니다.",
        analogy="일기예보가 말하는 흔들림과 지난 20일 실제 날씨의 흔들림을 나란히 놓고 그 차이를 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="iv_current − hv_20",
            rows=[
                dict(
                    label="실현변동성 hv_20 (종가로 직접 계산)", segs=[seg(-19, 0, "use", "20일")]
                ),
                dict(label="옵션 쪽 보고 이어 쓰기 한도", segs=[seg(-30, 0, "need", "최대 30일")]),
            ],
            marks=[(-30, "t−30"), (-19, "t−19"), (0, "t")],
            note="hv_20은 원천의 hv_current가 아니라 이 모듈이 조정 종가 일수익률로 20일 표준편차를 내 √252로 연율화한 값입니다. 옵션 쪽 30일 한도의 단위는 문서에 적혀 있지 않습니다.",
        ),
        principle=dict(
            type="flow",
            title="두 변동성의 차이",
            steps=[
                dict(
                    title="옵션 시장의 예상 (IV)",
                    text="옵션 가격에 담긴 변동성 iv_current입니다. 가장 최근 보고를 30일 안에서 이어 씁니다.",
                ),
                dict(
                    title="실제로 움직인 폭 (HV)",
                    text="종가 일수익률의 20일 표준편차를 √252로 연율화합니다.",
                ),
                dict(
                    title="IV에서 HV를 뺌",
                    text="양수면 시장이 실제보다 크게 겁먹은 상태입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="부호가 맞는지는 아직 확인되지 않았습니다. 정의를 뒤집은 문헌이 있습니다.",
        ),
        what="옵션 시장이 예상한 변동성에서 최근 실제 변동성을 뺀 값입니다.",
        model="LightGBM은 SHAP 36위, Ridge는 6위입니다. 둘 다 안정이고 방향은 −지만, 기대 부호가 미확정이라 어느 쪽이든 '맞는' 쪽이 되므로 대조에 정보가 없어 '기대 없음'으로 둡니다.",
        caution="등록 부호(−)가 맞는지 불분명해 문헌 정의 대조가 남아 있습니다. 빈 값은 약 67%입니다. 10-13에 층 3 대조를 '기대와 같음'에서 '기대 없음'으로 고쳤고, Ridge 6위 크기의 대부분도 _isna 성분입니다(근사 약 82%, 계산 예·미검증).",
    ),
    "iv_isna": dict(
        title="옵션 변동성 자료가 없는가",
        basis="valid",
        intuition="옵션 변동성 자료가 없는 종목이라는 표시입니다. 자료가 있는 종목은 주로 크고 유동적이라, 사실상 '옵션이 거래되지 않는 작은 종목' 표지입니다.",
        analogy="가게 앞에 '배달 안 됨' 팻말이 붙었는지 보는 것과 같습니다. 팻말이 붙은 곳은 대개 작은 가게입니다.",
        win=dict(
            axis=30,
            formula="IV 값이 없으면 1, 있으면 0",
            rows=[
                dict(
                    label="이어 쓰기 30일 안에 보고가 있나", segs=[seg(-30, 0, "need", "최대 30일")]
                )
            ],
            marks=[(-30, "t−30"), (0, "t")],
            note="30일 한도에 걸리면 빈 값이 되고 플래그가 켜집니다. 30일의 단위는 문서에 적혀 있지 않습니다.",
        ),
        principle=dict(
            type="flow",
            title="플래그와 그 플래그의 플래그",
            steps=[
                dict(
                    title="옵션 자료가 있나", text="약 1,600개 종목만 옵션 변동성 자료가 있습니다."
                ),
                dict(
                    title="없으면 iv_isna 켜짐",
                    text="30일 이어 쓰기에도 걸리지 않으면 빈 값이고 플래그가 켜집니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(
                    title="iv_isna_isna 열도 생김",
                    text="모든 피쳐에 _isna 열이 붙으므로 이 플래그에도 붙습니다. 빈 값이 없으면 늘 같은 값이라 정보가 없습니다.",
                ),
            ],
            measure="결과에서 iv_isna_isna 성분은 두 모델 모두 기여가 정확히 0.0이었습니다.",
        ),
        what="옵션 변동성 자료가 없으면 켜지는 플래그입니다.",
        model="LightGBM은 SHAP 6위(역U), Ridge는 31위입니다. 둘 다 안정이고 기대 부호가 없습니다. iv_isna_isna 열은 두 모델 모두 기여가 0.0입니다.",
        caution="사실상 이진 변수라 크기·유동성과 겹칩니다. iv_isna_isna 열의 기여는 두 모델 모두 0입니다(10 §6).",
    ),
    "sp500_member": dict(
        title="S&P 500 구성 종목인가",
        basis="cal",
        intuition="그 시점에 S&P 500에 들어 있는 종목인가입니다. 사실상 '대형 우량주' 표시입니다. 처음엔 약한 신호로 보였지만 holdout에서 방향이 뒤집혔습니다.",
        analogy="명문 동아리 회원 명단에 이름이 있는지 보는 것과 같습니다. 명단은 가끔씩만 갱신됩니다.",
        win=dict(
            axis=70,
            formula="t일 이전 가장 최근 위키 리비전에 있으면 True",
            rows=[dict(label="리비전 간격 (달력일)", segs=[seg(-47, 0, "need", "최대 47일")])],
            marks=[(-47, "−47일"), (0, "t")],
            note="거래일이 아니라 달력일입니다. 리비전 표본이 주 1회가 아니고 간격이 최대 47일입니다. 위키 편집 시각은 날짜 단위로만 판정합니다.",
        ),
        principle=dict(
            type="timeline",
            title="명단이 갱신되기까지 (설명용 배치)",
            caption="t일의 값은 t일 이전 가장 최근 리비전이 정합니다. 눈금은 설명용이고 간격은 최대 47일입니다.",
            events=[
                dict(at=0, label="이전 리비전", sub="종목이 명단에 있음"),
                dict(at=30, label="오늘 t", sub="값을 정하는 날", kind="today"),
                dict(at=47, label="다음 리비전", sub="아직 못 봄"),
            ],
            spans=[dict(a=0, b=47, label="리비전 간격 최대 47일")],
            known={"from": 0, "to": 30, "label": "t일의 값을 정하는 리비전"},
        ),
        what="그날 S&P 500 명단에 들어 있는지입니다.",
        model="LightGBM(SHAP 14위)과 Ridge(22위) 모두 불안정이라 해석하지 않습니다.",
        caution="방향을 개발 구간 IC로 정해 낙관 편향이 있었고, holdout 반전은 그 편향과 일치하는 방향입니다. 다만 12개월 한 번의 점추정이고 바스켓이 동점이라 평균 회귀·우연과 구분하지 못합니다. 3차에서는 바스켓 동점 몫이 1.00이라 해석할 수 없습니다.",
    ),
    "sp500_days_since_add": dict(
        title="S&P 500 편입 후 경과일",
        basis="cal",
        intuition="S&P 500에 막 들어온 종목일수록 값이 작습니다. 지수 추종 자금이 몰려 단기로 오른 뒤 되돌린다는 이야기를 노렸지만, 값이 있는 종목이 매우 적습니다.",
        analogy="새 학기 전학생이 전학 온 지 며칠째인지를 세는 것과 같습니다. 원래부터 다니던 학생은 전학 날짜를 알 수 없어 칸이 빕니다.",
        win=dict(
            axis=70,
            formula="t일 − 편입이 관측된 리비전 날짜",
            rows=[dict(label="편입 관측 오차 (달력일)", segs=[seg(-47, 0, "need", "최대 47일")])],
            marks=[(-47, "−47일"), (0, "t")],
            note="편입이 리비전 사이에서 실제로 관측된 경우에만 계산합니다. 첫 리비전(2018-01-04)부터 있던 종목은 편입일을 알 수 없어 빈 값입니다.",
        ),
        principle=dict(
            type="timeline",
            title="경과일은 이렇게 셉니다 (설명용 배치)",
            caption="편입일은 리비전으로만 알 수 있어 최대 47일의 오차를 그대로 물려받습니다. 눈금은 설명용입니다.",
            events=[
                dict(at=0, label="이전 리비전", sub="명단에 없음"),
                dict(at=47, label="편입이 처음 보인 리비전", sub="여기를 편입일로 씀"),
                dict(at=120, label="오늘 t", sub="경과일을 재는 날", kind="today"),
            ],
            spans=[
                dict(a=0, b=47, label="편입일 오차 최대 47일"),
                dict(a=47, b=120, label="경과일 (값)"),
            ],
            known={"from": 47, "to": 120, "label": "경과일을 아는 구간"},
        ),
        what="S&P 500 편입이 관측된 날부터 오늘까지의 일수입니다.",
        model="LightGBM은 SHAP 50위, Ridge는 52위로 거의 쓰지 않습니다. 둘 다 안정이고 기대(−)와 반대로 설명하지 못했습니다.",
        caution="빈 값이 약 97.5%라 사실상 _isna 열이 주된 정보입니다. 왼쪽이 잘려 있고 오차가 최대 47일이라 신호 해상도가 낮습니다. LightGBM 50위(0.00004)와 Ridge 52위(0.00027)는 크기가 거의 0이라 '설명 못 함'은 정보가 적습니다.",
    ),
}

EXAMPLES = {
    "rv_20": [dict(kind="p95"), dict(kind="p05")],
    "rev_1w": [dict(kind="p95"), dict(kind="p05")],
}
EX_WINDOWS = {
    "mcap_rank": [],
    "iv_rank": [],
    "rv_20": [dict(a=-19, b=0, kind="use", text="최근 20일")],
    "rev_1w": [dict(a=-5, b=0, kind="use", text="최근 5거래일")],
}

HTML_OVERRIDE = {
    "LEDE": '미국 서빙 모델 <code>us_exploratory_20260929_r1</code>(LightGBM·Ridge)이 쓰는 피쳐 중 모멘텀·변동성·유동성·옵션·지수 편입 17개를 그림으로 풀었습니다. 각 카드에 쉬운 비유와 계산 원리 그림을 달았습니다. 정의와 직관의 정본은 <a href="04_us_price_liquidity.md"><code>04_us_price_liquidity.md</code></a>이고, 이 페이지는 그 그림판입니다. 모델 반응 그림은 <b>라벨을 읽지 않은 월초 12개 횡단면의 진단이며 모델 점수의 반응이지 수익 기여가 아닙니다.</b> 가로축은 값이 아니라 그날 종목들 사이의 순위(백분위)입니다. 단변량 등급·수치는 비용 전이고 개발 구간을 다섯 번 봤으며 holdout은 이미 열렸습니다. 체결은 월초 종가 진입 가정이고, "안정"은 시드 반복 없이 앞·뒤 절반이 같다는 뜻입니다. 종목 단위 신호라 연금계좌(ETF만)에서는 직접 못 씁니다.',
    "MAPINTRO": "가로는 LightGBM 평균 |SHAP|, 세로는 Ridge 평균 |기여|입니다. 두 모델의 점수 척도가 달라 숫자 크기를 직접 비교하지 않고, <b>선보다 위인지 아래인지</b>만 봅니다. 점 모양은 안정한 모델 수입니다. 색칠한 점이 이 문서의 17개이고 나머지 피쳐는 회색입니다.",
    "MAPCAP": '읽을 점. LightGBM SHAP 1~3위가 <code>mcap_rank</code>·<code>rv_20</code>·<code>idio_vol_60</code>이고, Ridge는 <code>log_dvol_20</code>·<code>amihud_20</code>·<code>mcap_rank</code> 순입니다. 두 모델 모두 규모·유동성과 변동성 쪽에 크게 기댑니다. 다만 규모·유동성은 타깃 L2가 이미 걷어낸 축이라 "기대와 같음"은 규모 프리미엄의 확인이 아니고, Ridge는 묶음 안에서 계수가 상쇄합니다(멤버 합 0.04835 대 묶음 0.01077). 모양과 안정 판정의 기준은 <a href="10_us_result.md"><code>10_us_result.md</code></a> §1에 있습니다.',
    "GROUPSINTRO": "날짜별 순위 상관 |ρ| ≥ 0.7인 피쳐끼리 묶었습니다(12개 날짜 평균). 이 문서의 피쳐는 변동성 묶음(<code>max_ret_1m</code>·<code>rv_20</code>·<code>rv_60</code>·<code>idio_vol_60</code>)과 규모·유동성 묶음(<code>log_dvol_20</code>·<code>amihud_20</code>·<code>mcap_rank</code>)에 듭니다. <b>묶음 안에서는 서로 몫을 나눠 가집니다.</b> Ridge는 계수가 서로 반대 부호로 상쇄하기도 해서, 묶음 합계는 멤버별 크기의 합보다 작을 수 있습니다. 그래서 Ridge는 피쳐 하나씩 읽으면 안 됩니다.",
}
