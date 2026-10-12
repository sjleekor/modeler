# ruff: noqa: E501
"""KR flow_* 15개 해설 페이지 설정 (02번)."""

DOC_DIR = "milestones/common/20261012_feature_explainer"
OUT_NAME = "02_kr_flow.html"
DOC_MD = "02_kr_flow.md"
RESULT_MD = "09_kr_result.md"
RUN_ID = "kr_l3_20261012T113305Z"
PANEL_REL = (
    "kr/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/"
    "FS1h_h20_lag1_rank_idxbench__holdout/feat_panel.parquet"
)
RAW_REL = "kr/raw/raw_postgres/snapshot_date=2026-09-30/source=sj2_remote"
REF_DATE = "2026-08-28"
PREFIX = "flow_"
MARKET = "kr"
PAGE_TITLE = "KR 수급 피쳐 그림 해설"
GROUP_NAMES = {
    "c05": "거래대금·비유동성·공매도 수량",
    "c06": "외국인 20일 순매수·보유 변화",
    "c07": "개인 5일 순매수 수량·비율",
    "c08": "개인 20일 순매수 수량·비율",
}
MODELER_COMMIT = "24eb8e0"
N_ICE = 40
EX_DAYS = 260
EX_MARKET = "KOSPI"
SCATTER_LABELS = ["flow_short_avg_price", "flow_individual_netbuy_to_volume_20d"]

THEMES = [
    (
        "순매수 합 (원시 수량)",
        "외국인·기관·개인이 최근 5일·20일 동안 순매수한 주식 수의 합 6개. 비율이 아니라 수량이라 절댓값에는 규모가 섞이고, 순위에서는 사는 쪽과 파는 쪽 양 끝에 몰립니다.",
        [
            "flow_foreign_netbuy_sum_5d",
            "flow_foreign_netbuy_sum_20d",
            "flow_inst_netbuy_sum_5d",
            "flow_inst_netbuy_sum_20d",
            "flow_indiv_netbuy_sum_5d",
            "flow_indiv_netbuy_sum_20d",
        ],
    ),
    (
        "외국인 보유 주식 수 변화",
        "외국인이 들고 있는 주식 수가 5일·20일 전보다 얼마나 늘었나. 순매수 흐름이 아니라 잔고의 차이입니다.",
        ["flow_foreign_holding_chg_5d", "flow_foreign_holding_chg_20d"],
    ),
    (
        "평소 대비 쏠림 (z)",
        "어제 순매수가 그 종목의 최근 20일 평소 수준에서 얼마나 벗어났나. 거래량으로 나누지 않고 종목 자신과 비교합니다.",
        ["flow_foreign_netbuy_z_20d", "flow_inst_netbuy_z_20d"],
    ),
    (
        "개인 순매수 강도",
        "개인 순매수 합을 같은 기간 총거래량으로 나눈 비율. 분모가 규모를 상쇄합니다. 단변량 검정(scan)은 20일이 primary로 통과했고, 5일은 등록만 되고 실행되지 않았습니다.",
        ["flow_individual_netbuy_to_volume_5d", "flow_individual_netbuy_to_volume_20d"],
    ),
    (
        "공매도",
        "공매도 체결 평균가·거래량·거래대금 3개. 금지 구간을 가리는 마스크가 없어 구간별 결측이 크게 다르고, 값의 분포는 모릅니다.",
        ["flow_short_avg_price", "flow_short_selling_volume", "flow_short_selling_value"],
    ),
]

FMT = {
    "flow_foreign_netbuy_sum_5d": "big",
    "flow_foreign_netbuy_sum_20d": "big",
    "flow_inst_netbuy_sum_5d": "big",
    "flow_inst_netbuy_sum_20d": "big",
    "flow_indiv_netbuy_sum_5d": "big",
    "flow_indiv_netbuy_sum_20d": "big",
    "flow_foreign_holding_chg_5d": "big",
    "flow_foreign_holding_chg_20d": "big",
    "flow_foreign_netbuy_z_20d": "num",
    "flow_inst_netbuy_z_20d": "num",
    "flow_individual_netbuy_to_volume_5d": "pct",
    "flow_individual_netbuy_to_volume_20d": "pct",
    "flow_short_avg_price": "int",
    "flow_short_selling_volume": "big",
    "flow_short_selling_value": "eok",
}

BASIS = {
    "sum5": "lag1 · 직전 유효 거래일까지 최근 5거래일 합 (거래정지일 제외, 창이 다 차는지는 검사 안 함)",
    "sum20": "lag1 · 직전 유효 거래일까지 최근 20거래일 합 (거래정지일 제외, 창이 다 차는지는 검사 안 함)",
    "hold": "lag1 · 오늘 값 − N거래일 전 값 (잔고 차이, 기업행동 마스킹 없음)",
    "z": "lag1 · 하루 값을 최근 20행 평균·표준편차로 표준화 (창에 그날 포함)",
    "full": "lag1 · 마트의 _lag1 컬럼 · 창이 N행 다 차야 계산, 분모 0이면 비움",
    "day": "lag1 · 그날 하루 값 (체결 거래량이지만 공표 시각은 미확인, lag1로 방어)",
}


def seg(a, b, kind="use", text=""):
    return {"a": a, "b": b, "kind": kind, "text": text}


LAG_NOTE = "표의 t는 '직전 유효 거래일'입니다. 서빙은 하루 늦춘 값(lag1)을 입력으로 쓰므로, 날짜 K의 입력은 t까지의 집계입니다."

CARDS = {
    "flow_foreign_netbuy_sum_5d": dict(
        title="5일 외국인 순매수 수량",
        basis="sum5",
        intuition="최근 일주일 동안 외국인이 순매수한 주식 수의 합입니다. 크면 많이 사 모은 종목이고 작거나 음수면 많이 판 종목입니다. 다만 주 수라서 발행주식이 많은 종목은 절댓값이 크게 나옵니다(순위에서는 위쪽이 아니라 양 끝).",
        analogy="가게의 주간 장부에서 '들어온 물건 − 나간 물건'을 합한 순증가 개수와 같습니다. 금액이 아니라 개수라 큰 가게일수록 숫자가 큽니다.",
        win=dict(
            axis=30,
            formula="SUM( foreign_net_buy_volume , 최근 5행 )",
            rows=[dict(label="외국인 순매수 주 수의 단순 합", segs=[seg(-4, 0, "use", "5행")])],
            marks=[(-4, "t−4"), (0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="flow",
            title="하루 순매수가 5일 합이 되기까지",
            steps=[
                dict(
                    title="하루 순매수",
                    text="외국인이 산 주식 수에서 판 주식 수를 뺍니다. 단위는 주입니다.",
                ),
                dict(
                    title="5거래일 쌓기", text="최근 5거래일(거래정지일 제외)의 하루 값을 모읍니다."
                ),
                dict(
                    title="단순 합",
                    text="나누지 않고 그대로 더합니다. 규모가 그대로 섞입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="규모는 절댓값에 섞이고 순위에서는 양 끝에 몰립니다.",
        ),
        what="외국인의 5거래일 순매수 주 수 합입니다. 금액이 아니라 수량입니다.",
        model="평균 |SHAP|이 0.0002로 SHAP 44위입니다. 곡선은 평평하고 섞어도 IC가 거의 안 변해서 모델은 사실상 쓰지 않습니다.",
        caution="단변량 검정을 받은 적이 없고, 관련 비율 결과는 간접 참고일 뿐입니다(관측은 반대 부호 D라 이 가설이 지지되지는 않았습니다). 20일 합과 겹치고, 규모는 순위 양 끝에 몰립니다.",
    ),
    "flow_foreign_netbuy_sum_20d": dict(
        title="20일 외국인 순매수 수량",
        basis="sum20",
        intuition="최근 한 달 동안 외국인이 순매수한 주식 수의 합입니다. 5일 합보다 느리고 완만하게 움직이고, 같은 규모 문제가 있습니다.",
        analogy="한 달 치 장부 합계입니다. 주간 합계보다 흔들림이 작지만, 큰 가게일수록 숫자가 크다는 점은 같습니다.",
        win=dict(
            axis=70,
            formula="SUM( foreign_net_buy_volume , 최근 20행 )",
            rows=[dict(label="외국인 순매수 주 수의 단순 합", segs=[seg(-19, 0, "use", "20행")])],
            marks=[(-19, "t−19"), (0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="ratio",
            title="합계에는 규모가 섞입니다 (가정 예, 단위 만 주)",
            unit="",
            rows=[
                dict(
                    label="대형주 (가정)",
                    numLabel="순매수",
                    num=10,
                    denLabel="거래량",
                    den=500,
                    result="합계 10 · 거래량 대비 10 ÷ 500 = 2%",
                ),
                dict(
                    label="소형주 (가정)",
                    numLabel="순매수",
                    num=1,
                    denLabel="거래량",
                    den=50,
                    result="합계 1 · 거래량 대비 1 ÷ 50 = 2%",
                ),
            ],
        ),
        what="외국인의 20거래일 순매수 주 수 합입니다. 거래량으로 나누지 않습니다.",
        model="SHAP 31위이고 방향 ρ는 −0.19입니다. 곡선은 '기타' 모양이고 섞기 기준으로는 불안정합니다. 크게 의지하지 않습니다.",
        caution="두 종목의 거래 비중이 같아도 합계는 10배 차이가 날 수 있습니다(위 그림은 가정 예). 순위에서는 양 끝에 몰립니다. 외국인 보유 주식 수 변화(c06)와 한 묶음입니다.",
    ),
    "flow_inst_netbuy_sum_5d": dict(
        title="5일 기관 순매수 수량",
        basis="sum5",
        intuition="최근 일주일 기관(연기금·투신 등)이 순매수한 주식 수의 합입니다. 크면 기관이 최근 사들인 종목입니다.",
        analogy="큰손 단체 손님이 일주일 동안 장바구니에 순수하게 더 담은 개수와 같습니다. 큰 가게일수록 숫자가 큽니다.",
        win=dict(
            axis=30,
            formula="SUM( institution_net_buy_volume , 최근 5행 )",
            rows=[dict(label="기관 순매수 주 수의 단순 합", segs=[seg(-4, 0, "use", "5행")])],
            marks=[(-4, "t−4"), (0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="timeline",
            title="집계가 모델 입력이 되기까지 (lag1)",
            events=[
                dict(at=-4, label="합산 시작", sub="5거래일 전"),
                dict(at=0, label="t일", sub="집계 포함 마지막 날"),
                dict(at=1, label="입력 날짜 K", sub="모델이 쓰는 날", kind="today"),
            ],
            spans=[dict(a=-4, b=0, label="5거래일 합"), dict(a=0, b=1, label="하루 늦춤")],
            caption="당일 집계를 장 마감 때 쓸 수 있는지 검증되지 않아서, 하루 늦춘 값을 정본으로 씁니다.",
        ),
        what="기관의 5거래일 순매수 주 수 합입니다.",
        model="SHAP 21위지만 불안정합니다. 곡선은 U자로 분류됐고 해석하지 않습니다.",
        caution="관련 비율(inst_netbuy_to_volume_20d)은 기대 +인데 관측이 −였습니다. 수량 합에서도 같은지는 알 수 없어서 '크면 좋다'를 그대로 믿으면 안 됩니다. 한국 기관은 금융투자·보험 등 8종을 합친 정의입니다.",
    ),
    "flow_inst_netbuy_sum_20d": dict(
        title="20일 기관 순매수 수량",
        basis="sum20",
        intuition="최근 한 달 기관이 순매수한 주식 수의 합입니다. 5일 합보다 천천히 변합니다.",
        analogy="큰손 단체 손님이 한 달 동안 순수하게 더 담은 개수와 같습니다. 일주일 합보다 꾸준히 이어진 쪽이 반영됩니다.",
        win=dict(
            axis=70,
            formula="SUM( institution_net_buy_volume , 최근 20행 )",
            rows=[dict(label="기관 순매수 주 수의 단순 합", segs=[seg(-19, 0, "use", "20행")])],
            marks=[(-19, "t−19"), (0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="timeline",
            title="20거래일 합과 하루 늦춤",
            events=[
                dict(at=-19, label="합산 시작", sub="20거래일 전"),
                dict(at=0, label="t일", sub="집계 포함 마지막 날"),
                dict(at=1, label="입력 날짜 K", sub="모델이 쓰는 날", kind="today"),
            ],
            spans=[dict(a=-19, b=0, label="20거래일 합"), dict(a=0, b=1, label="하루 늦춤")],
            caption="K일의 입력은 t일까지의 집계입니다. t일 뒤의 값은 쓰지 않습니다.",
        ),
        what="기관의 20거래일 순매수 주 수 합입니다.",
        model="SHAP 20위지만 불안정합니다. 곡선은 U자로 분류됐고 해석하지 않습니다.",
        caution="5일 합과 겹치고, 규모는 순위 양 끝에 몰립니다. 관련 비율의 부호가 기대와 반대였지만, 뒤집어 쓰는 것은 결과 뒤 가설이라 후보로 올리지 않습니다(새 사전등록이 필요).",
    ),
    "flow_indiv_netbuy_sum_5d": dict(
        title="5일 개인 순매수 수량",
        basis="sum5",
        intuition="최근 일주일 개인 투자자가 순매수한 주식 수의 합입니다. 개인 매수가 좋은 신호인지 나쁜 신호인지는 연구마다 갈려서 방향을 정하지 않았습니다.",
        analogy="동네 시장에서 일반 손님들이 일주일 동안 순수하게 더 사 간 개수와 같습니다. 이 손님이 많이 온 것이 좋은 일인지는 가게마다 해석이 다릅니다.",
        win=dict(
            axis=30,
            formula="SUM( individual_net_buy_volume , 최근 5행 )",
            rows=[dict(label="개인 순매수 주 수의 단순 합", segs=[seg(-4, 0, "use", "5행")])],
            marks=[(-4, "t−4"), (0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="flow",
            title="가설이 서로 반대입니다",
            steps=[
                dict(title="개인이 몰려 산다", text="최근 5거래일 개인 순매수 합이 큽니다."),
                dict(
                    title="가설 ①② (+)",
                    text="가격 압력, 유동성 공급의 대가 쪽 이야기는 이후 오른다고 봅니다.",
                ),
                dict(
                    title="가설 ③④ (−)",
                    text="추세 추종, 정보 열위 쪽 이야기는 이후 내린다고 봅니다.",
                    hl=True,
                    tag="방향이 갈립니다",
                ),
            ],
            measure="네 이야기가 공존해서 기대 부호를 정하지 않았습니다.",
        ),
        what="개인의 5거래일 순매수 주 수 합입니다.",
        model="평균 |SHAP|이 0.0000이라 SHAP 47위입니다. SHAP이 전부 0이라 방향 ρ는 계산되지 않았고, 모델은 이 열을 쓰지 않습니다.",
        caution="같은 개인 수급을 비율로 잰 individual_netbuy_to_volume_5d와 한 묶음(c07)입니다. 외국인·기관과 거울상일 수 있습니다.",
    ),
    "flow_indiv_netbuy_sum_20d": dict(
        title="20일 개인 순매수 수량",
        basis="sum20",
        intuition="최근 한 달 개인이 순매수한 주식 수의 합입니다.",
        analogy="동네 시장에서 일반 손님이 한 달 동안 순수하게 더 사 간 개수와 같습니다.",
        win=dict(
            axis=70,
            formula="SUM( individual_net_buy_volume , 최근 20행 )",
            rows=[dict(label="개인 순매수 주 수의 단순 합", segs=[seg(-19, 0, "use", "20행")])],
            marks=[(-19, "t−19"), (0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="ratio",
            title="합계와 비중은 다릅니다 (가정 예, 단위 만 주)",
            unit="",
            rows=[
                dict(
                    label="종목 A (가정)",
                    numLabel="순매수",
                    num=10,
                    denLabel="거래량",
                    den=200,
                    result="합계 10 · 거래량 대비 10 ÷ 200 = 5%",
                ),
                dict(
                    label="종목 B (가정)",
                    numLabel="순매수",
                    num=15,
                    denLabel="거래량",
                    den=600,
                    result="합계 15 · 거래량 대비 15 ÷ 600 = 2.5%",
                ),
            ],
            caption="합계는 B가 1.5배 크지만 거래량 대비 비중은 A가 2배입니다. 이 피쳐는 앞의 합계만 봅니다.",
        ),
        what="개인의 20거래일 순매수 주 수 합입니다. 거래량으로 나눈 비율 버전은 아래 to_volume_20d입니다.",
        model="SHAP 39위이고 방향 ρ +0.71, 곡선은 '기타' 모양입니다. SHAP은 안정이지만 섞기는 불안정해서 크게 쓰지 않습니다.",
        caution="비율 버전(to_volume_20d)과 한 묶음(c08)으로 정보가 겹칩니다. 규모는 순위 양 끝에 몰립니다.",
    ),
    "flow_foreign_holding_chg_5d": dict(
        title="5일 외국인 보유 주식 수 변화",
        basis="hold",
        intuition="일주일 전보다 외국인이 들고 있는 주식 수가 얼마나 늘었는지입니다. 주 수라서 큰 종목은 절댓값이 크고(순위에서는 양 끝), 증자로 주식 수가 늘면 외국인이 안 사도 값이 달라질 수 있습니다.",
        analogy="창고 재고표에서 일주일 전 재고와 오늘 재고의 차이를 보는 것과 같습니다. 재고 단위가 바뀌는 일(증자)이 있으면 차이도 흔들립니다.",
        win=dict(
            axis=30,
            formula="foreign_holding_shares[t] − foreign_holding_shares[t−5행]",
            rows=[
                dict(label="오늘 보유 주 수", segs=[seg(-1, 0, "use", "t")]),
                dict(label="− 5행 전 보유 주 수", segs=[seg(-6, -5, "minus", "t−5")]),
                dict(label="결과: 5행 사이의 변화", segs=[seg(-5, 0, "use", "5행 사이")]),
            ],
            marks=[(-5, "t−5"), (0, "t")],
            note=LAG_NOTE + " 20행 전 값이 없으면 비웁니다.",
        ),
        principle=dict(
            type="flow",
            title="잔고 차이로 재는 변화",
            steps=[
                dict(title="5거래일 전 보유", text="외국인이 들고 있던 주식 수(주)입니다."),
                dict(title="오늘 보유", text="오늘 기준 외국인 보유 주식 수입니다."),
                dict(
                    title="차이 = 변화",
                    text="오늘 − 5거래일 전. 순매수 합이 아니라 잔고의 차이입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="증자·감자·분할로 주식 수가 바뀌면 외국인의 행동과 무관한 값이 됩니다(기업행동 마스킹 없음).",
        ),
        what="외국인 보유 주식 수의 5거래일 차분입니다. 분모가 없는 수량입니다.",
        model="SHAP 38위로 작고 곡선은 평평합니다. 섞으면 rank IC가 0.00014 오히려 올랐고, 값이 비었다는 표시(_isna) 쪽 몫이 값 자체보다 큽니다.",
        caution="2026-08 이후 값이 KIS로 바뀌었고 KRX와 17.3% 불일치했습니다. 5일 변화는 대응하는 단변량 결과가 없습니다. 섞기 ΔrankIC가 음수라는 것은 섞는 편이 IC가 높다는 뜻이고, 피쳐를 빼는 근거는 아닙니다.",
    ),
    "flow_foreign_holding_chg_20d": dict(
        title="20일 외국인 보유 주식 수 변화",
        basis="hold",
        intuition="한 달 전보다 외국인 보유 주식 수가 얼마나 늘었는지입니다. 일주일 변화보다 느린 흐름입니다.",
        analogy="창고 재고표에서 한 달 전 재고와 오늘 재고의 차이를 보는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="foreign_holding_shares[t] − foreign_holding_shares[t−20행]",
            rows=[
                dict(label="오늘 보유 주 수", segs=[seg(-1, 0, "use", "t")]),
                dict(label="− 20행 전 보유 주 수", segs=[seg(-21, -20, "minus", "")]),
                dict(label="결과: 20행 사이의 변화", segs=[seg(-20, 0, "use", "20행 사이")]),
            ],
            marks=[(-20, "t−20"), (0, "t")],
            note=LAG_NOTE + " 20행 전 값이 없으면 비웁니다.",
        ),
        principle=dict(
            type="timeline",
            title="두 시점의 잔고를 뺍니다",
            events=[
                dict(at=-20, label="20거래일 전", sub="보유 주 수"),
                dict(at=0, label="t일", sub="보유 주 수"),
                dict(at=1, label="입력 날짜 K", sub="모델이 쓰는 날", kind="today"),
            ],
            spans=[dict(a=-20, b=0, label="이 사이의 변화"), dict(a=0, b=1, label="하루 늦춤")],
            caption="K일의 입력은 t일까지의 집계입니다. t일 뒤의 값은 쓰지 않습니다.",
        ),
        what="외국인 보유 주식 수의 20거래일 차분입니다.",
        model="SHAP 41위이고 방향 ρ −0.70, 곡선은 역U자입니다. SHAP·섞기 모두 안정이지만 ΔrankIC가 +0.00005로 0과 구분되지 않습니다.",
        caution="외국인 20일 순매수 합(c06)과 상관 +0.84로 한 묶음입니다. 같은 외국인을 흐름과 잔고로 두 번 잰 셈입니다.",
    ),
    "flow_foreign_netbuy_z_20d": dict(
        title="외국인 순매수의 평소 대비 쏠림",
        basis="z",
        intuition="어제 외국인의 순매수가 그 종목의 최근 한 달 평소 수준보다 얼마나 튀었는지입니다. 종목 크기와 상관없이 비교됩니다.",
        analogy="평소 하루 매출이 100만 원인 가게가 어제 300만 원을 팔았다면 '평소보다 크게 튄 날'입니다. 가게 크기와 무관하게 평소 대비로 봅니다.",
        win=dict(
            axis=70,
            formula="( x[t] − AVG(x, 20행) ) / STDDEV_SAMP(x, 20행) ,  x = foreign_net_buy_volume",
            rows=[
                dict(label="그날 하루 순매수 x[t]", segs=[seg(-1, 0, "use", "당일")]),
                dict(
                    label="평균·표준편차를 내는 창 (오늘 포함)", segs=[seg(-19, 0, "fit", "20행")]
                ),
            ],
            marks=[(-19, "t−19"), (0, "t")],
            note=LAG_NOTE + " 표준편차가 0이면 비웁니다. 창이 20행 다 찼는지는 검사하지 않습니다.",
        ),
        principle=dict(
            type="flow",
            title="z가 만들어지기까지",
            steps=[
                dict(title="그날 순매수", text="어제 외국인 순매수 주 수 한 개입니다."),
                dict(
                    title="자기 20일과 비교", text="그 종목의 최근 20행 평균과 표준편차를 구합니다."
                ),
                dict(
                    title="z = (그날 − 평균) ÷ 표준편차",
                    text="평소 대비 표준편차의 몇 배인지로 환산합니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="자기 자신과 비교하므로 규모가 상쇄됩니다. 20일 합이 아니라 하루 값 기준입니다.",
        ),
        what="그날 하루 순매수가 같은 종목의 20일 평균에서 표준편차 몇 개만큼 떨어져 있는지입니다. 거래량으로 나누지 않습니다.",
        model="SHAP 37위이고 방향 ρ +0.81, 곡선은 '기타'(보조 판정 증가)입니다. SHAP·섞기 모두 안정이지만 크기가 작습니다. 기대 부호가 없어 대조하지 않았습니다.",
        caution="하루 값이라 잡음이 큽니다. 창에 그날이 들어 있어 |z|는 4.25(=19/√20)를 넘지 못하고, 표준편차가 0이면 비웁니다. 만기일 단일가 같은 제도 수급일이 섞일 수 있습니다(추정). 마트 계산 방식에 따라 끝자리가 달라졌다는 기록이 있어 서빙은 window 방식으로 고정했습니다. 가설도 검정도 없는 피쳐입니다.",
    ),
    "flow_inst_netbuy_z_20d": dict(
        title="기관 순매수의 평소 대비 쏠림",
        basis="z",
        intuition="어제 기관의 순매수가 그 종목의 최근 한 달 평소보다 유난히 많았는지입니다. 외국인 z와 같은 방식입니다.",
        analogy="평소 하루 3만 개씩 사 가던 단체 손님이 어제 9만 개를 사 갔다면 평소 대비 튄 날입니다. 평소 규모가 달라도 같은 눈금으로 비교합니다.",
        win=dict(
            axis=70,
            formula="( x[t] − AVG(x, 20행) ) / STDDEV_SAMP(x, 20행) ,  x = institution_net_buy_volume",
            rows=[
                dict(label="그날 하루 순매수 x[t]", segs=[seg(-1, 0, "use", "당일")]),
                dict(
                    label="평균·표준편차를 내는 창 (오늘 포함)", segs=[seg(-19, 0, "fit", "20행")]
                ),
            ],
            marks=[(-19, "t−19"), (0, "t")],
            note=LAG_NOTE + " 표준편차가 0이면 비웁니다.",
        ),
        principle=dict(
            type="ratio",
            title="수량이 100배 달라도 z는 같습니다 (가정 예, 단위 만 주)",
            unit="",
            rows=[
                dict(
                    label="대형주 (가정)",
                    numLabel="그날−평균",
                    num=200,
                    denLabel="표준편차",
                    den=100,
                    result="z = 200 ÷ 100 = 2.0",
                ),
                dict(
                    label="소형주 (가정)",
                    numLabel="그날−평균",
                    num=2,
                    denLabel="표준편차",
                    den=1,
                    result="z = 2 ÷ 1 = 2.0",
                ),
            ],
            caption="원시 수량은 100배 차이가 나도 z는 같아서 규모가 상쇄됩니다.",
        ),
        what="그날 하루 기관 순매수가 같은 종목의 20일 평균에서 표준편차 몇 개만큼 떨어져 있는지입니다.",
        model="SHAP 42위이고 방향 ρ +0.41, 곡선은 평평합니다. SHAP은 안정이지만 섞기는 불안정이고 크기가 아주 작습니다.",
        caution="가설도 검정도 없습니다. 기관 비율 버전은 관측 부호가 반대였으므로 직관을 그대로 옮기면 안 됩니다. 창에 그날이 들어 있어 |z|는 4.25 이하입니다.",
    ),
    "flow_individual_netbuy_to_volume_5d": dict(
        title="5일 개인 순매수 강도",
        basis="full",
        intuition="최근 일주일 거래된 주식 중 몇 퍼센트를 개인이 순매수했는지입니다. 우리 데이터의 20일 버전에서는 이런 종목이 이후 20일 동안 평균적으로 약간 더 올랐지만 이유는 모릅니다.",
        analogy="장터 전체 거래 중에서 동네 손님이 순수하게 더 사 간 몫이 몇 퍼센트인지 보는 것과 같습니다. 가게 크기가 달라도 같은 눈금입니다.",
        win=dict(
            axis=30,
            formula="SUM( individual_net_buy_volume , 5행 ) / SUM( total_volume , 5행 )",
            rows=[
                dict(label="분자: 개인 순매수 주 수 합", segs=[seg(-4, 0, "use", "5행")]),
                dict(label="분모: 총거래량 합", segs=[seg(-4, 0, "minus", "5행")]),
                dict(label="계산 조건: 5행이 다 있어야 함", segs=[seg(-4, 0, "need", "5행")]),
            ],
            marks=[(-4, "t−4"), (0, "t")],
            note=LAG_NOTE + " 분모가 0이면 비웁니다.",
        ),
        principle=dict(
            type="ratio",
            title="개인 순매수 ÷ 총거래량 (가정 예)",
            unit="",
            rows=[
                dict(
                    label="개인이 순매수",
                    numLabel="순매수 합",
                    num=5,
                    denLabel="거래량 합",
                    den=100,
                    result="5 ÷ 100 = +0.05 → 총거래량의 5%만큼 개인이 순매수",
                ),
                dict(
                    label="개인이 순매도",
                    numLabel="순매수 합",
                    num=-3,
                    denLabel="거래량 합",
                    den=100,
                    result="−3 ÷ 100 = −0.03 → 개인이 팔아 넘긴 쪽",
                ),
            ],
        ),
        what="개인 순매수 5일 합을 같은 기간 총거래량 합으로 나눈 값입니다. 20일 버전이 scan에서 A등급을 받았고, 5일 변형은 등록만 되고 실행되지 않았습니다.",
        model="SHAP 40위이고 방향 ρ +0.62, 곡선은 '기타'(보조 판정 증가)입니다. SHAP은 안정이지만 섞기는 불안정해서 크게 쓰지 않습니다.",
        caution="외국인·기관 비율과 거의 거울상이라 독립 정보인지 불명입니다. 개인 수량 합(c07 짝)과 함께 입력에 들어 있습니다. 20일 결과를 5일에 붙여 읽지 마십시오(5일은 미실행). IC·등급은 개발 구간 값이고 비용 전이며, '6/6 cell'은 겹치는 지평의 셀이라 독립 확인이 아닙니다.",
    ),
    "flow_individual_netbuy_to_volume_20d": dict(
        title="20일 개인 순매수 강도",
        basis="full",
        intuition="최근 한 달 거래된 주식 중 몇 퍼센트를 개인이 순매수했는지입니다. 5일보다 느리게 움직이고, scan에서 primary였습니다.",
        analogy="한 달 동안 장터 전체 거래 중 동네 손님이 순수하게 더 사 간 몫입니다. 일주일 몫보다 오래 이어진 쏠림이 드러납니다.",
        win=dict(
            axis=70,
            formula="SUM( individual_net_buy_volume , 20행 ) / SUM( total_volume , 20행 )",
            rows=[
                dict(label="분자: 개인 순매수 주 수 합", segs=[seg(-19, 0, "use", "20행")]),
                dict(label="분모: 총거래량 합", segs=[seg(-19, 0, "minus", "20행")]),
                dict(label="계산 조건: 20행이 다 있어야 함", segs=[seg(-19, 0, "need", "20행")]),
            ],
            marks=[(-19, "t−19"), (0, "t")],
            note=LAG_NOTE + " 분모가 0이면 비웁니다.",
        ),
        principle=dict(
            type="flow",
            title="두 합을 나눠 비율로 바꿉니다",
            steps=[
                dict(title="개인 순매수 합", text="최근 20거래일 개인 순매수 주 수를 더합니다."),
                dict(
                    title="총거래량 합",
                    text="같은 20거래일 총거래량을 더합니다. 20행이 다 차야 합니다.",
                ),
                dict(
                    title="나눔 = 순매수 강도",
                    text="분모가 규모를 상쇄합니다. 분모가 0이면 비웁니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="0.05면 20일 총거래량의 5%만큼 개인이 순매수했다는 뜻입니다.",
        ),
        what="개인 순매수 20일 합을 20일 총거래량 합으로 나눈 값입니다. 이 문서에서 scan 단변량을 통과(A등급)한 컬럼은 이 20일 primary이고, 5일은 등급을 이어받았을 뿐입니다.",
        model="섞으면 rank IC가 0.00433 줄고(단독 47개 중 7위) 섞기로는 안정입니다. SHAP 15위이지만 SHAP은 불안정하고 곡선은 역U자라 해석하지 않습니다.",
        caution="진단이지 판정이 아닙니다. 개인 20일 수량 합(c08 짝)과 한 묶음이고, 단변량 A등급이 모델 성과를 뜻하지는 않습니다. 층 3 방향 ρ +0.59가 scan 관측 +와 같은 부호인 것은 사후 일치라 확인 근거가 아닙니다.",
    ),
    "flow_short_avg_price": dict(
        title="공매도 체결 평균가",
        basis="day",
        intuition="공매도가 평균 얼마 가격에 체결됐는지(원)입니다. 사실상 그 종목의 주가 수준에 가깝고, 사전 가설은 없습니다. 층 3에서 모델이 크게 기대는 것은 탐색 결과(사후)입니다.",
        analogy="중고장터에서 그날 팔린 물건의 평균 거래가를 보는 것과 같습니다. 비싼 물건이 거래되는 장터는 평균가도 높습니다.",
        win=dict(
            axis=8,
            formula="short_selling_value[t] / NULLIF( short_selling_volume[t] , 0 )",
            rows=[dict(label="공매도 금액 ÷ 공매도 수량 (당일)", segs=[seg(-1, 0, "use", "당일")])],
            marks=[(0, "t")],
            note=LAG_NOTE + " 공매도가 없는 날은 분모가 0이라 비웁니다.",
        ),
        principle=dict(
            type="ratio",
            title="평균가 = 금액 ÷ 수량 (가정 예)",
            unit="",
            rows=[
                dict(
                    label="주가 높은 종목 (가정)",
                    numLabel="금액(백만 원)",
                    num=600,
                    denLabel="수량(천 주)",
                    den=10,
                    result="600백만 ÷ 10천 주 = 6만 원/주",
                ),
                dict(
                    label="주가 낮은 종목 (가정)",
                    numLabel="금액(백만 원)",
                    num=40,
                    denLabel="수량(천 주)",
                    den=20,
                    result="40백만 ÷ 20천 주 = 2천 원/주",
                ),
            ],
            caption="종가 대비 비율이 아니라 가격 레벨이라, 주가가 높은 종목이 위쪽 순위를 차지합니다.",
        ),
        what="그날 공매도 금액을 공매도 거래량으로 나눈 원 단위 평균 체결가입니다.",
        model="평균 |SHAP|으로 SHAP 7위이고, 방향 ρ +0.86의 단조 증가입니다. 섞으면 IC가 0.01426 줄어 단독 2위이지만 앞 절반 +0.03451 / 뒤 절반 −0.00338로 부호가 바뀌어 불안정합니다.",
        caution="주가 수준이나 소형주 여부를 대신 담을 수 있습니다(추론, 실측 없음). 기대 부호가 없고 진단이지 판정이 아닙니다. 모델이 가장 크게 기대는 축 중 하나라 '기대 없음'을 중립으로 읽지 마십시오. 공매도가 없는 날은 비어서 _isna가 저거래 표지일 수 있습니다(추정).",
    ),
    "flow_short_selling_volume": dict(
        title="그날 공매도 거래량",
        basis="day",
        intuition="그날 공매도로 거래된 주식 수입니다. 비율이 아니라 부호 없는 주 수라서 규모·유동성이 큰 종목이 높은 순위를 채울 가능성이 있습니다(추정, 시총과의 상관은 미측정).",
        analogy="그날 빌려서 판 물건의 개수를 세는 것과 같습니다. 큰 장터일수록 개수가 많습니다.",
        win=dict(
            axis=8,
            formula="short_selling_volume[t]  (그대로 통과)",
            rows=[dict(label="그날 하루 값 (합산 없음)", segs=[seg(-1, 0, "use", "당일")])],
            marks=[(0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="flow",
            title="공매도 거래량이 입력이 되기까지",
            steps=[
                dict(title="공매도 체결", text="그날 공매도로 체결된 주식 수(주)입니다."),
                dict(
                    title="그대로 통과",
                    text="합산이나 나눗셈 없이 하루 값을 씁니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(
                    title="하루 늦춤",
                    text="체결 거래량이라 공표 지연은 없고, 다른 수급과 같이 lag1을 겁니다.",
                ),
            ],
            measure="금지 구간(2020-03-16~2021-05-02, 2023-11-06~2025-03-28)을 가리는 마스크가 없습니다.",
        ),
        what="그날 하루 공매도 체결 주 수입니다.",
        model="SHAP 29위이고 곡선은 U자입니다. SHAP은 안정이지만 섞기는 불안정하고, 기대 부호는 비교하지 않았습니다.",
        caution="거래대금·비유동성과 한 묶음(c05)이고 앞 절반 −0.00612 / 뒤 절반 +0.03627로 시기에 따라 부호가 바뀝니다. 두 절반 모두 공매도 재개 뒤라 금지 체제로는 설명되지 않습니다. 마스크가 없어 구간별 결측이 크게 다르고(KOSDAQ 평균가 결측 49.2→79.4→38.4→26.3%), 값의 분포는 모릅니다.",
    ),
    "flow_short_selling_value": dict(
        title="그날 공매도 거래대금",
        basis="day",
        intuition="그날 공매도로 거래된 금액(원)입니다. 거래량 버전과 같은 이야기이고, 주가가 비싼 종목은 금액이 더 크게 나옵니다.",
        analogy="그날 빌려서 판 물건의 영수증 총액입니다. 같은 개수라도 비싼 물건이면 총액이 커집니다.",
        win=dict(
            axis=8,
            formula="short_selling_value[t]  (그대로 통과)",
            rows=[dict(label="그날 하루 값 (합산 없음)", segs=[seg(-1, 0, "use", "당일")])],
            marks=[(0, "t")],
            note=LAG_NOTE,
        ),
        principle=dict(
            type="flow",
            title="수량에 가격이 곱해집니다",
            steps=[
                dict(title="체결 수량", text="그날 공매도로 체결된 주 수입니다."),
                dict(title="× 체결 가격", text="각 체결의 가격이 곱해져 금액이 됩니다."),
                dict(
                    title="= 거래대금(원)",
                    text="하루 합계 금액을 그대로 씁니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="수량이 같아도 주가가 높으면 금액이 커서 가격 수준이 섞입니다.",
        ),
        what="그날 하루 공매도 체결 금액(원)입니다.",
        model="SHAP 25위이고 곡선은 U자입니다. SHAP은 안정이지만 섞기는 불안정하고, 기대 부호는 비교하지 않았습니다.",
        caution="거래량 버전과 같은 c05 묶음이라 정보가 겹칩니다. 이 둘만의 상관은 기록이 없고, 묶음 c05 안 7쌍의 상관은 −0.89~+0.91입니다. 소형·고가주 섞임은 거래량 카드와 같습니다.",
    ),
}

EXAMPLES = {
    "flow_foreign_netbuy_sum_20d": [dict(kind="p95"), dict(kind="p05")],
    "flow_individual_netbuy_to_volume_20d": [dict(kind="p95"), dict(kind="p05")],
    "flow_short_avg_price": [dict(kind="p95"), dict(kind="p05")],
}
EX_WINDOWS = {k: [] for k in EXAMPLES}
EX_NOTES = {
    "flow_foreign_netbuy_sum_20d": "{name}의 20일 외국인 순매수 합은 {value}주이고 코스피 안 순위는 {pct}입니다. 그림은 가격일 뿐 수급 흐름이 아닙니다. 순위가 높다는 것이 '많이 샀다'인지 '원래 큰 종목이다'인지는 이 그림만으로 알 수 없습니다.",
    "flow_individual_netbuy_to_volume_20d": "{name}의 20일 개인 순매수 강도는 {value}이고 코스피 안 순위는 {pct}입니다. 그림은 가격일 뿐 수급 흐름이 아닙니다. 값이 양수면 20일 총거래량 중 그만큼을 개인이 순매수했다는 뜻이고, 음수면 순매도입니다.",
    "flow_short_avg_price": "{name}의 공매도 평균 체결가는 {value}원이고 코스피 안 순위는 {pct}입니다. 그림은 가격이고 공매도 체결 자체가 아닙니다. 이 값은 주가 수준에 가까워서, 순위가 높은 종목은 주가가 높은 쪽이라고 보는 게 자연스럽습니다(추론).",
}

HTML_OVERRIDE = dict(
    LEDE='서빙 모델 <code>kr_daily_h20_v1</code>이 쓰는 수급·공매도 피쳐 <code>flow_*</code> 15개를 그림으로 풀었습니다. 정의와 직관의 정본은 <a href="02_kr_flow.md"><code>02_kr_flow.md</code></a>이고, 이 페이지는 그 그림판입니다. 15개 중 13개(FS0)는 단변량 검정을 받은 적이 없고, 순매수 합 6개는 비율이 아니라 <b>원시 수량</b>이라 절댓값에는 규모가 섞이고 순위에서는 양 끝에 몰립니다. 모델 반응 그림(③·④·⑤)은 이미 열린 holdout(2025-08-01~2026-08-28)으로 만든 <b>진단이지 판정이 아닙니다.</b> 가로축은 값이 아니라 그날 종목들 사이의 순위(백분위)입니다. 카드의 IC·등급은 거래비용 전 값이고, 그날 종가에 진입한다고 가정한 값입니다. 층 3은 공매도 재개·NXT 개시 뒤 한 국면이며 holdout을 세 번째로 쓴 결과입니다. 종목 단위 신호라 연금계좌(ETF만)에서는 직접 못 씁니다.',
    MAPINTRO="가로는 평균 |SHAP|(모델 점수를 얼마나 흔드나), 세로는 그 피쳐 하나만 섞었을 때 rank IC가 줄어든 정도(ΔrankIC)입니다. <b>0선 아래는 섞었더니 IC가 오히려 오른 피쳐</b>입니다(피쳐를 빼는 근거는 아닙니다). 47개 전체 중 <code>flow_*</code> 15개만 색을 칠했고, 나머지 32개는 회색입니다.",
    MAPLEGEND='<span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="5" style="fill:var(--s1)"/></svg>SHAP 안정</span><span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="4" style="fill:var(--surface);stroke:var(--s1)" stroke-width="2"/></svg>SHAP 불안정</span><span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="3.5" style="fill:var(--muted);opacity:.55"/></svg>flow가 아닌 32개 (참고)</span>',
    MAPCAP='읽을 점. 이 문서의 15개는 평균 |SHAP| 합이 0.0782로 47개 합 0.5283의 일부입니다. <code>flow_short_avg_price</code>가 가장 크게 기댑니다(SHAP 7위, 섞으면 rank IC 0.01426 하락으로 단독 2위). 다만 앞 절반 +0.03451 / 뒤 절반 −0.00338로 부호가 바뀌어 섞기로는 불안정합니다. 사전 가설이 없는 탐색 결과(사후)라 "기대 없음"을 중립으로 읽지 마십시오. 섞었더니 IC가 오히려 오른 12개 중 이 문서에는 <code>flow_foreign_holding_chg_5d</code> 하나(−0.00014)만 있습니다. 모양과 안정 판정의 기준은 <a href="09_kr_result.md"><code>09_kr_result.md</code></a> §1에 있습니다.',
    GROUPSINTRO="날짜별 순위 상관 |ρ| ≥ 0.7인 피쳐끼리 묶었습니다(SHAP 날짜 53일 평균). <b>묶음 안에서는 서로 몫을 나눠 가집니다.</b> 이 문서의 수급 피쳐가 든 묶음은 c05~c08입니다. c05는 거래대금·비유동성(가격 쪽 피쳐)과 공매도 거래량·금액이 한 묶음이고(묶음 안 7쌍, ρ −0.89~+0.91), 앞 절반 −0.00612 / 뒤 절반 +0.03627로 시기에 따라 부호가 바뀝니다(두 절반 모두 재개 뒤라 금지 체제로는 설명되지 않습니다). c06~c08은 같은 주체의 수량(또는 보유 변화)과 비율이 짝입니다(+0.84, +0.84, +0.85). 묶음을 한꺼번에 섞은 값은 멤버 단독 값의 합이 되지 않습니다.",
)
