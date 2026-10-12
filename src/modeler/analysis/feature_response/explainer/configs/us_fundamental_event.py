# ruff: noqa: E501
"""05 US 재무·실적·공시 19개 해설 config."""

DOC_DIR = "milestones/common/20261012_feature_explainer"
OUT_NAME = "05_us_fundamental_event.html"
DOC_MD = "05_us_fundamental_event.md"
RESULT_MD = "10_us_result.md"
RUN_ID = "us_l3_20261012T112931Z"
PANEL_V2_REL = "us/datasets/us_features_v2/part.parquet"
PANEL_FLOW_REL = "us/datasets/us_features_flow_v1/part.parquet"
PRICES_GLOB = "us/derived/snapshots/prices_daily/snapshot_date=*/part.parquet"
COMPANY_META_GLOB = "us/derived/snapshots/company_meta/snapshot_date=*/part.parquet"
MARKET = "us"
REF_DATE = "2026-06-01"
PAGE_TITLE = "US 재무·실적·공시 피쳐 해설"
MODELER_COMMIT = "d98d5cf"
N_ICE = 40
EX_DAYS = 260
GROUP_NAMES = {"g3": "실적 발표 달력"}
SCATTER_LABELS = ["roa_ttm", "days_to_earn", "n_estimates", "sp_ttm", "ep_ttm"]

THEMES = [
    (
        "밸류에이션 (F5)",
        "분모가 모두 시총인 네 비율입니다. 클수록 싸다는 쪽으로 같은 방향이라 순위가 많이 겹칩니다.",
        ["bm", "ep_ttm", "cfp_ttm", "sp_ttm"],
    ),
    (
        "수익성 (F6)",
        "같은 순이익·이익을 자산, 자본, 매출로 나눠 본 네 비율입니다.",
        ["roa_ttm", "roe_ttm", "gpa", "opm_ttm"],
    ),
    (
        "재무 변화·자본 배분 (F7·F8)",
        "1년 새 변화(자산·주식 수)와 주주에게 돌려준 몫(배당·자사주)입니다.",
        ["asset_growth", "accruals", "net_issuance", "div_yield", "buyback_yield"],
    ),
    (
        "실적 발표·애널리스트 (F9)",
        "실적 서프라이즈와 발표 달력, 추정치 수입니다. 발표 시각은 8-K 접수 시각으로 보정합니다.",
        ["sue_last", "days_since_earn", "days_to_earn", "n_estimates"],
    ),
    ("공시 (F12)", "SEC 공시 목록에서 센 건수와 제출 지연입니다.", ["n_8k_90", "filing_lag"]),
]
FMT = {
    "bm": "ratio",
    "ep_ttm": "pct",
    "cfp_ttm": "pct",
    "sp_ttm": "ratio",
    "roa_ttm": "pct",
    "roe_ttm": "pct",
    "gpa": "pct",
    "opm_ttm": "pct",
    "asset_growth": "pct",
    "accruals": "pct",
    "net_issuance": "pct",
    "div_yield": "pct",
    "buyback_yield": "pct",
    "sue_last": "num",
    "days_since_earn": "int",
    "days_to_earn": "int",
    "n_estimates": "int",
    "n_8k_90": "int",
    "filing_lag": "int",
}
BASIS = {
    "pit": "SEC 접수일(filed)이 t 이하인 것 중 최신 값 · 접수 시각(16:00 ET) 보정은 없음 · 그림의 거래일 눈금은 근사",
    "ev": "발표·공시 유효일 기준(접수가 16:00 ET 뒤면 다음 거래일) · 그림의 눈금은 근사",
    "cal": "캘린더의 예정일 원값(8-K 보정 없음) · 거래일 수",
    "cash": "배당락일(ex_date) 기준 · 지난 365일",
}


def seg(a, b, kind="use", text=""):
    return {"a": a, "b": b, "kind": kind, "text": text}


def ttm_segs():
    return [
        seg(-251, -189, "use", "분기1"),
        seg(-188, -126, "use", "분기2"),
        seg(-125, -63, "use", "분기3"),
        seg(-62, 0, "use", "분기4"),
    ]


TTM_NOTE = "4분기를 같은 길이로 나눈 그림용 근사입니다. 실제로는 접수된 최근 4분기를 쓰고, 마지막 분기의 끝은 t보다 앞입니다."


def win_ttm(formula, label="최근 4분기 합 (TTM)"):
    return dict(
        axis=270,
        formula=formula,
        rows=[dict(label=label, segs=ttm_segs())],
        marks=[(-251, "약 12개월 전"), (0, "t")],
        note=TTM_NOTE,
    )


def win_ttm_over(formula, label2, label1="순이익 등 최근 4분기 합 (TTM)"):
    return dict(
        axis=270,
        formula=formula,
        rows=[
            dict(label=label1, segs=ttm_segs()),
            dict(label=label2, segs=[seg(-7, 0, "use", "최신 1개")]),
        ],
        marks=[(-251, "약 12개월 전"), (0, "t")],
        note=TTM_NOTE + " 분모는 접수일 이하 중 최신 값 하나입니다.",
    )


def win_pt(formula, label="접수일 ≤ t 중 최신 값"):
    return dict(
        axis=8,
        formula=formula,
        rows=[dict(label=label, segs=[seg(-7, 0, "use", "최신 1개")])],
        marks=[(0, "t")],
        note="값 하나입니다. 새 분기 보고서가 접수되는 날 바뀝니다.",
    )


CARDS = {
    "bm": dict(
        title="장부가 대비 시총 (B/M)",
        basis="pit",
        intuition="회사가 가진 순자산(자본)에 비해 주식 시장이 얼마나 싸게 값을 매겼는지 보는 값입니다. 클수록 자산에 비해 싼 주식입니다.",
        analogy="중고차를 살 때 차 상태(장부 가치)에 비해 부른 값이 얼마나 낮은지 보는 것과 같습니다.",
        win=win_pt("StockholdersEquity(최신) ÷ 시총"),
        principle=dict(
            type="timeline",
            title="재무 숫자가 모델에 오기까지 (SEC 접수일 기준)",
            events=[
                dict(at=0, label="분기 말", sub="보고기간이 끝남"),
                dict(at=40, label="10-Q 접수", sub="보통 한 달 이상 뒤", kind="known"),
                dict(at=55, label="모델이 쓰는 날", sub="접수일 ≤ t인 첫 날부터", kind="today"),
            ],
            spans=[dict(a=0, b=40, label="공시 지연(보통 한 달 이상)")],
            known={"from": 40, "to": 60, "label": "모델이 아는 시점"},
            caption="그림의 일수는 모양만 보이려는 예시입니다. 분기 숫자는 접수일부터 쓰고, 그날 한꺼번에 바뀝니다.",
        ),
        what="순자산을 시총(표지 주식 수 × 그날 종가)으로 나눈 값입니다.",
        model="LightGBM은 31위에서 곡선이 평평하고, Ridge는 순위가 오를수록 점수를 올립니다. 두 모델 모두 안정이고 기대와 같습니다.",
        caution="자본이 음수면 값이 음수가 되어 맨 아래 순위를 받습니다. ep·cfp·sp와 분모가 같아 순위가 겹칩니다. 시총 분모에 분할 보정이 없어 분할 뒤 값이 부풀 수 있습니다. Ridge 크기의 약 57%는 _isna 성분이고, 층 3 '기대와 같음'은 모델이 거의 안 쓰는 값의 방향이라 의미가 약합니다. 층 3 입력은 동결 데이터셋이라 서빙 값과 86셀 다릅니다.",
    ),
    "ep_ttm": dict(
        title="이익 대비 주가 (E/P)",
        basis="pit",
        intuition="최근 1년 순이익을 시총으로 나눈 값입니다. 클수록 번 돈에 비해 주가가 싸고, 적자 회사는 음수라 맨 아래로 갑니다.",
        analogy="가게를 살 때 1년 순이익이 가게값의 몇 %인지 보는 것과 같습니다. 적자 가게는 마이너스입니다.",
        win=win_ttm_over(
            "SUM( NetIncomeLoss , 최근 4분기 ) ÷ 시총", "시총 (최신 표지 주식 수 × 종가)"
        ),
        principle=dict(
            type="timeline",
            title="TTM: 최근 4분기를 합친 12개월치",
            events=[
                dict(at=3, label="분기 1 말", sub=""),
                dict(at=6, label="분기 2 말", sub=""),
                dict(at=9, label="분기 3 말", sub=""),
                dict(at=12, label="분기 4 말", sub="최신 접수분", kind="today"),
            ],
            spans=[dict(a=0, b=12, label="네 분기를 더하면 TTM(12개월)")],
            caption="Q4는 직접 공시되지 않아 연간값 − 9개월 누적으로 만듭니다. 중간에 빠진 분기가 있으면 TTM을 내지 않습니다.",
        ),
        what="최근 4분기 순이익 합을 시총으로 나눈 값입니다.",
        model="LightGBM은 47위에서 U자 곡선을 그리고 기대로 설명하지 못합니다. Ridge는 불안정이라 해석하지 않습니다.",
        caution="적자 종목이 맨 아래에 몰립니다. 일회성 이익이 크면 값이 튀고, 순이익 TTM의 5.3%는 200일 넘게 묵은 값입니다. 시총 분모에 분할 보정이 없습니다. Ridge는 같은 방향 가설인 cfp_ttm·bm과 부호가 갈립니다(불안정).",
    ),
    "cfp_ttm": dict(
        title="영업현금흐름 대비 주가 (CF/P)",
        basis="pit",
        intuition="사업으로 실제 들어온 현금(영업현금흐름)을 시총으로 나눈 값입니다. 장부상 이익과 달리 현금이 기준입니다.",
        analogy="가게의 장부 이익이 아니라 실제로 금고에 들어온 돈이 가게값의 몇 %인지 보는 것과 같습니다.",
        win=win_ttm_over(
            "SUM( 영업현금흐름 , 최근 4분기 ) ÷ 시총",
            "시총 (최신 표지 주식 수 × 종가)",
            "영업현금흐름 최근 4분기 합 (TTM)",
        ),
        principle=dict(
            type="flow",
            title="영업현금이 값이 되기까지",
            steps=[
                dict(
                    title="분기 현금흐름",
                    text="현금흐름표의 영업활동 현금흐름을 분기마다 모읍니다.",
                ),
                dict(title="4분기 합", text="최근 4분기를 더해 12개월치(TTM)를 만듭니다."),
                dict(
                    title="시총으로 나눔",
                    text="시총(표지 주식 수 × 종가)으로 나눕니다. 음수면 아래 순위입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="클수록 현금을 잘 버는 싼 주식입니다.",
        ),
        what="영업현금흐름 4분기 합을 시총으로 나눈 값입니다.",
        model="LightGBM은 32위에서 순위가 오를수록 점수가 오르는 쪽(ρ +0.91)이고 안정이며 기대와 같습니다. Ridge는 불안정이라 해석하지 않습니다.",
        caution="영업현금 TTM의 2.8%는 200일 넘게 묵은 값입니다. accruals와 같은 재료(순이익·영업현금)를 씁니다. 시총 분모에 분할 보정이 없고, Ridge 크기의 약 98%가 _isna 성분이라 Ridge 반응은 값이 아니라 '자료가 있느냐'에 가깝습니다.",
    ),
    "sp_ttm": dict(
        title="매출 대비 시총 (S/P)",
        basis="pit",
        intuition="최근 1년 매출을 시총으로 나눈 값입니다. 적자 회사도 매출은 있어서 같은 줄에 세울 수 있습니다.",
        analogy="가게값 한 푼당 1년 동안 손님이 얼마나 돈을 냈는지 보는 것과 같습니다. 이익이 얼마 남았는지는 따지지 않습니다.",
        win=win_ttm_over(
            "SUM( 매출 , 최근 4분기 ) ÷ 시총",
            "시총 (최신 표지 주식 수 × 종가)",
            "매출 최근 4분기 합 (TTM)",
        ),
        principle=dict(
            type="ratio",
            title="같은 시총, 다른 매출",
            unit="",
            rows=[
                dict(
                    label="매출이 큰 쪽",
                    numLabel="매출 TTM",
                    num=80,
                    denLabel="시총",
                    den=100,
                    result="80 ÷ 100 = 0.8 → 값이 큼(매출에 비해 싸게 평가)",
                ),
                dict(
                    label="매출이 작은 쪽",
                    numLabel="매출 TTM",
                    num=10,
                    denLabel="시총",
                    den=100,
                    result="10 ÷ 100 = 0.1 → 값이 작음(매출에 비해 비싸게 평가)",
                ),
            ],
        ),
        what="매출 4분기 합을 시총으로 나눈 값입니다.",
        model="두 모델 모두 불안정이라 해석하지 않습니다(LightGBM 17위, Ridge 13위).",
        caution="결측이 43.95%입니다. TTM 조립이 틀려 0이 나오는 경우가 있고(U10), 3차에서 공식 유니버스 B(2025-06-30 판 t 2.02, 2026-06-30 판 t 2.19)였지만 44개 Bonferroni 문턱(약 3.25) 아래이고 추정치 0.9~1.0%/월은 부풀었을 수 있습니다. 후보지 증거가 아닙니다. 모델 반응이 불안정해서 롱 B와 모델 반응은 별개이고, Ridge 크기의 약 66%는 _isna 성분입니다. 시총 분모에 분할 보정이 없습니다.",
    ),
    "roa_ttm": dict(
        title="총자산이익률 (ROA)",
        basis="pit",
        intuition="회사가 가진 자산 전체로 1년 동안 순이익을 얼마나 냈는지 보는 값입니다.",
        analogy="같은 크기의 가게 두 곳 중 1년에 더 많이 남긴 곳이 위로 가는 것과 같습니다.",
        win=win_ttm_over(
            "SUM( NetIncomeLoss , 4분기 ) ÷ Assets(최신)", "총자산 (접수일 ≤ t 중 최신)"
        ),
        principle=dict(
            type="ratio",
            title="같은 자산, 다른 순이익",
            unit="",
            rows=[
                dict(
                    label="자산을 잘 쓴 회사",
                    numLabel="순이익 TTM",
                    num=12,
                    denLabel="총자산",
                    den=100,
                    result="12 ÷ 100 = +0.12",
                ),
                dict(
                    label="적자 회사",
                    numLabel="순이익 TTM",
                    num=-8,
                    denLabel="총자산",
                    den=100,
                    result="−8 ÷ 100 = −0.08 → 아래 순위",
                ),
            ],
        ),
        what="순이익 4분기 합을 최신 총자산으로 나눈 값입니다.",
        model="LightGBM은 5위로 크게 기대고 순위가 오를수록 점수가 오르며(ρ +0.76) 안정이고 기대와 같습니다. Ridge는 불안정이라 해석하지 않습니다.",
        caution="일회성 손익에 흔들립니다. roe·opm·gpa와 같은 계열이라 겹칩니다. 층 3 LightGBM 5위·안정·'기대와 같음'이지만 단변량 3차 롱은 t −0.02로 근거가 없고, 같은 개발 구간 표본이라 독립 확인이 아닙니다.",
    ),
    "roe_ttm": dict(
        title="자기자본이익률 (ROE)",
        basis="pit",
        intuition="주주 몫(자기자본)으로 1년에 얼마나 벌었는지 봅니다. 빚이 많으면 같은 이익에도 값이 커집니다.",
        analogy="내 돈 1억을 넣은 가게가 1년에 얼마를 남겼는지 보는 것과 같습니다. 빌린 돈으로 키운 가게는 같은 이익에도 더 높게 나옵니다.",
        win=win_ttm_over(
            "SUM( NetIncomeLoss , 4분기 ) ÷ StockholdersEquity(최신)",
            "자기자본 (접수일 ≤ t 중 최신)",
        ),
        principle=dict(
            type="ratio",
            title="자본이 음수면 부호가 뒤집힙니다",
            unit="",
            rows=[
                dict(
                    label="정상",
                    numLabel="순이익",
                    num=30,
                    denLabel="자본",
                    den=100,
                    result="30 ÷ 100 = +0.3",
                ),
                dict(
                    label="자본이 음수인 흑자 회사",
                    numLabel="순이익",
                    num=30,
                    denLabel="자본",
                    den=-10,
                    result="30 ÷ −10 = −3.0 → 흑자인데 음수가 됩니다",
                ),
                dict(
                    label="자본이 음수인 적자 회사",
                    numLabel="순이익",
                    num=-30,
                    denLabel="자본",
                    den=-10,
                    result="−30 ÷ −10 = +3.0 → 적자인데 양수가 됩니다",
                ),
            ],
        ),
        what="순이익 4분기 합을 최신 자기자본으로 나눈 값입니다.",
        model="두 모델 모두 안정이고 기대와 같습니다. LightGBM은 42위에서 U자, Ridge는 48위에서 순위가 오를수록 점수를 올립니다.",
        caution="코드는 분모가 0이거나 비었을 때만 막습니다. 자본이 음수일 때의 왜곡은 코드에서 읽은 성질이고 데이터로 확인하지 않았습니다. 자본과 이익이 모두 음수면 비율이 양수가 되어 위쪽 순위로 갈 수 있습니다. Ridge 48위는 크기가 작아 '기대와 같음'의 의미가 약합니다.",
    ),
    "gpa": dict(
        title="자산 대비 매출총이익 (GP/A)",
        basis="pit",
        intuition="자산 하나가 매출총이익을 얼마나 만드는지 봅니다. 판매관리비나 세금 같은 잡음을 빼고 본업의 기본 수익력만 보려는 값입니다.",
        analogy="가게 임대료와 인건비를 빼기 전에, 가게 크기 대비 물건을 팔아 남긴 차액이 얼마인지 보는 것과 같습니다.",
        win=win_ttm_over(
            "( GrossProfit 또는 매출 − 매출원가 ) 4분기 합 ÷ Assets(최신)",
            "총자산 (접수일 ≤ t 중 최신)",
            "매출총이익 최근 4분기 합 (TTM)",
        ),
        principle=dict(
            type="flow",
            title="매출총이익이 값이 되기까지",
            steps=[
                dict(
                    title="매출총이익 TTM",
                    text="GrossProfit 태그가 있으면 그대로, 없으면 매출 − 매출원가로 만듭니다.",
                ),
                dict(
                    title="총자산으로 나눔",
                    text="최신 총자산으로 나눕니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="금융 등 매출총이익 개념이 없는 업종은 결측입니다.",
        ),
        what="매출총이익 4분기 합을 최신 총자산으로 나눈 값입니다.",
        model="LightGBM은 22위이지만 불안정이라 해석하지 않습니다. Ridge는 38위에서 순위가 오를수록 점수를 올리며 안정이고 기대와 같습니다.",
        caution="GrossProfit 태그가 있는 행이 31.2%뿐이라 대체식을 씁니다. v1 결측이 70.4%였고(3차 63.2%) 매출 TTM의 틀린 값(U10)에 의존합니다. Ridge 크기의 약 71%가 _isna 성분입니다.",
    ),
    "opm_ttm": dict(
        title="영업이익률",
        basis="pit",
        intuition="매출 100원에서 영업이익이 몇 원 남는지 봅니다. 사업이 얼마나 남는 장사인지를 한 숫자로 보여 줍니다.",
        analogy="치킨 한 마리를 팔 때 재료비·인건비를 빼고 몇 푼이 남는지 보는 것과 같습니다.",
        win=win_ttm(
            "SUM( OperatingIncomeLoss , 4분기 ) ÷ SUM( 매출 , 4분기 )",
            "영업이익과 매출 모두 최근 4분기 합",
        ),
        principle=dict(
            type="ratio",
            title="매출 100에서 남는 몫",
            unit="",
            rows=[
                dict(
                    label="남는 장사",
                    numLabel="영업이익 TTM",
                    num=15,
                    denLabel="매출 TTM",
                    den=100,
                    result="15 ÷ 100 = +0.15",
                ),
                dict(
                    label="영업적자",
                    numLabel="영업이익 TTM",
                    num=-10,
                    denLabel="매출 TTM",
                    den=100,
                    result="−10 ÷ 100 = −0.10",
                ),
            ],
        ),
        what="영업이익 4분기 합을 매출 4분기 합으로 나눈 값입니다.",
        model="두 모델 모두 안정이고 기대와 같습니다. LightGBM은 48위에서 역U자, Ridge는 34위에서 순위가 오를수록 점수를 올립니다.",
        caution="매출이 분모라 매출 TTM의 틀린 0·묵은 값(U10)에 직접 영향받습니다. 매출이 작고 영업이익이 크면 극단값이 됩니다. Ridge 크기의 약 79%가 _isna 성분이고, LightGBM 48위는 크기가 작아 '기대와 같음'의 의미가 약합니다.",
    ),
    "asset_growth": dict(
        title="자산 증가율",
        basis="pit",
        intuition="1년 새 자산을 얼마나 불렸는지 봅니다. 사전등록 부호는 (−)입니다.",
        analogy="1년 전보다 가게를 얼마나 키웠는지 보는 것과 같습니다. 인수로 갑자기 커진 경우와 서서히 키운 경우가 같은 줄에 섭니다.",
        win=dict(
            axis=270,
            formula="Assets(최신) ÷ Assets(1년 전) − 1",
            rows=[
                dict(label="1년 전 보고기간의 총자산", segs=[seg(-262, -254, "minus", "1년 전")]),
                dict(label="최신 총자산", segs=[seg(-7, 0, "use", "최신")]),
            ],
            marks=[(-262, "약 1년 전"), (0, "t")],
            note="코드는 최신 보고기간보다 300~430일 이른 보고기간을 1년 전으로 잡습니다(달력일). 그림의 거래일 위치는 근사입니다.",
        ),
        principle=dict(
            type="ratio",
            title="1년 전 대비 자산",
            unit="",
            rows=[
                dict(
                    label="자산을 늘린 회사",
                    numLabel="최신",
                    num=120,
                    denLabel="1년 전",
                    den=100,
                    result="120 ÷ 100 − 1 = +0.20",
                ),
                dict(
                    label="자산이 줄어든 회사",
                    numLabel="최신",
                    num=90,
                    denLabel="1년 전",
                    den=100,
                    result="90 ÷ 100 − 1 = −0.10",
                ),
            ],
        ),
        what="최신 총자산을 1년 전 총자산으로 나눈 값에서 1을 뺀 값입니다.",
        model="LightGBM은 40위에서 곡선이 평평하고 기대와 같습니다. Ridge는 순위가 오를수록 점수를 올려(ρ +0.70) 기대 부호와 반대 쪽이고, 결과 문서는 이를 설명 못 함으로 적었습니다.",
        caution="인수합병으로 갑자기 느는 경우와 유기적 성장이 섞입니다. LightGBM(기대와 같음)과 Ridge(설명 못 함)가 갈리고 둘 다 크기가 작아, 대조는 사실상 미정입니다.",
    ),
    "accruals": dict(
        title="발생액 비율",
        basis="pit",
        intuition="장부상 순이익 중 현금으로 안 들어온 몫이 자산 대비 얼마인지 봅니다. 사전등록 부호는 (−)입니다.",
        analogy="가게 장부에는 이익이 크게 적혀 있는데 금고의 현금은 그만큼 늘지 않은 정도를 보는 것과 같습니다.",
        win=win_ttm_over(
            "( SUM(NetIncomeLoss) − SUM(영업현금흐름) ) ÷ Assets(최신)",
            "총자산 (접수일 ≤ t 중 최신)",
            "순이익 − 영업현금흐름, 4분기 합",
        ),
        principle=dict(
            type="flow",
            title="이익과 현금의 차이가 값이 되기까지",
            steps=[
                dict(title="순이익 TTM", text="장부상 이익 4분기 합입니다."),
                dict(
                    title="영업현금 TTM을 뺌",
                    text="실제 들어온 현금을 뺍니다. 남는 것이 현금 없는 이익입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
                dict(title="총자산으로 나눔", text="최신 총자산으로 나눕니다."),
            ],
            measure="클수록 이익은 큰데 현금이 안 따라온 회사입니다.",
        ),
        what="(순이익 TTM − 영업현금 TTM)을 최신 총자산으로 나눈 값입니다.",
        model="두 모델 모두 불안정이라 해석하지 않습니다(LightGBM 25위, Ridge 18위).",
        caution="ep와 cfp의 차이를 보는 셈입니다. 재무 11개 중 규칙 A 대조에서 frozen과 다른 셀이 13개로 가장 많았습니다. Ridge 크기의 약 75%가 _isna 성분이고, 평균 자산이 아니라 기말 자산을 쓰는 점이 Sloan 정의와 다릅니다. 1차(v1) IC는 등록 부호(−)와 반대였습니다.",
    ),
    "net_issuance": dict(
        title="순주식발행률",
        basis="pit",
        intuition="1년 새 발행 주식 수가 얼마나 늘었는지 봅니다. 사전등록 부호는 (−)입니다.",
        analogy="피자를 8조각에서 10조각으로 나누면 한 조각이 작아지는 것과 같습니다. 주식 수가 늘면 기존 주주 몫이 묽어집니다.",
        win=dict(
            axis=270,
            formula="발행주식 수(최신) ÷ 발행주식 수(1년 전) − 1",
            rows=[
                dict(
                    label="1년 전 10-K·10-Q 표지의 주식 수",
                    segs=[seg(-262, -254, "minus", "1년 전")],
                ),
                dict(label="최신 표지 주식 수", segs=[seg(-7, 0, "use", "최신")]),
            ],
            marks=[(-262, "약 1년 전"), (0, "t")],
            note="그림의 거래일 위치는 근사입니다. 표지 주식 수는 분기마다 바뀝니다.",
        ),
        principle=dict(
            type="ratio",
            title="1년 전 대비 주식 수",
            unit="",
            rows=[
                dict(
                    label="주식을 새로 찍은 회사",
                    numLabel="최신",
                    num=110,
                    denLabel="1년 전",
                    den=100,
                    result="110 ÷ 100 − 1 = +0.10",
                ),
                dict(
                    label="자사주를 줄인 회사",
                    numLabel="최신",
                    num=95,
                    denLabel="1년 전",
                    den=100,
                    result="95 ÷ 100 − 1 = −0.05",
                ),
            ],
        ),
        what="최신 표지 주식 수를 1년 전 표지 주식 수로 나눈 값에서 1을 뺀 값입니다.",
        model="두 모델 모두 안정이지만 둘 다 기대로 설명하지 못합니다. LightGBM은 46위에서 역U자, Ridge도 46위입니다.",
        caution="주식분할·역분할이 주식 수를 크게 바꾸는데, 이 코드에는 분할 보정이 없습니다(코드로 확인, 빈도·크기는 미측정). 분할이 끼면 값이 분할 배수 −1로 읽힙니다. 3차 SHORT +0.89%/월(t 2.11)로 피할 쪽에서만 t 2를 넘고, Ridge 크기의 약 83%가 _isna 성분입니다.",
    ),
    "div_yield": dict(
        title="배당수익률",
        basis="cash",
        intuition="지난 1년 동안 주당 배당금 합을 주가로 나눈 값입니다. 배당을 안 주는 회사는 0입니다.",
        analogy="예금 이자율처럼, 주식 한 주를 들고 있으면 1년에 현금이 몇 % 들어오는지 보는 것과 같습니다.",
        win=dict(
            axis=270,
            formula="SUM( 주당 배당금 , 지난 365일 ) ÷ 종가[t]",
            rows=[
                dict(
                    label="배당락일이 지난 365일 안에 있는 배당",
                    segs=[seg(-251, 0, "use", "365 달력일")],
                )
            ],
            marks=[(-251, "t−365일"), (0, "t")],
            note="365일은 달력일입니다. 그림은 거래일 약 252행으로 근사했습니다.",
        ),
        principle=dict(
            type="flow",
            title="배당이 값이 되기까지",
            steps=[
                dict(
                    title="배당락일 확인",
                    text="배당락일이 지난 365일 안에 있는 배당만 셉니다. 공개 즉시 알려지는 정보라 지연을 두지 않습니다.",
                ),
                dict(title="주당 금액 합", text="그 배당의 주당 금액을 더합니다."),
                dict(
                    title="종가로 나눔",
                    text="그날 종가로 나눕니다. 무배당은 결측이 아니라 0입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="0인 종목이 한 덩어리로 묶여 있습니다.",
        ),
        what="지난 365일 주당 배당금 합을 그날 종가로 나눈 값입니다.",
        model="두 모델 모두 불안정이라 해석하지 않습니다(LightGBM 23위, Ridge 24위).",
        caution="1차 후속(v1)에서 롱 쪽은 R이었고, 3차 LONG −0.73%/월(t −2.33)도 기대 부호(+)와 반대 방향입니다. 모델 타깃(L2)과 L0 롱 결과는 기대 근거가 다른데, 이 설명이 3차 값까지 닫는지는 미확인이고 고배당이 특정 업종·규모에 몰린 구성 효과일 수 있습니다(가설). 원천징수 등 세후 값은 다르고, 배당액의 분할 보정 여부는 미확인입니다.",
    ),
    "buyback_yield": dict(
        title="자사주 매입 수익률",
        basis="pit",
        intuition="최근 1년간 자사주를 사는 데 쓴 돈을 시총으로 나눈 값입니다.",
        analogy="가게 주인이 자기 가게 지분을 1년 동안 얼마나 다시 사들였는지 가게값 대비로 보는 것과 같습니다.",
        win=win_ttm_over(
            "SUM( PaymentsForRepurchaseOfCommonStock , 4분기 ) ÷ 시총",
            "시총 (최신 표지 주식 수 × 종가)",
            "자사주 매입액 최근 4분기 합 (TTM)",
        ),
        principle=dict(
            type="flow",
            title="자사주 매입이 값이 되기까지",
            steps=[
                dict(
                    title="현금흐름표 태그",
                    text="자사주 매입 지출 태그를 분기마다 모읍니다. 태그가 없으면 값을 못 만듭니다.",
                ),
                dict(title="4분기 합", text="최근 4분기를 더합니다."),
                dict(
                    title="시총으로 나눔",
                    text="시총으로 나눕니다. '안 샀다'와 '태그가 없다'는 구분하지 않고 둘 다 결측입니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="v1에서는 결측이 69.2%였습니다.",
        ),
        what="자사주 매입액 4분기 합을 시총으로 나눈 값입니다.",
        model="두 모델 모두 안정입니다. LightGBM은 51위에서 곡선이 평평하고 기대로 설명하지 못하며, Ridge는 43위에서 기대와 같습니다.",
        caution="결측이 많아 _isna 열이 정보를 상당 부분 나릅니다. Ridge 크기의 약 74%가 _isna 성분이고, 두 모델 방향이 갈리고 크기가 작아 대조는 미정에 가깝습니다. 시총 분모에 분할 보정이 없습니다.",
    ),
    "sue_last": dict(
        title="최근 실적 서프라이즈",
        basis="ev",
        intuition="최근 분기 실적이 시장 예상보다 얼마나 좋았거나 나빴는지입니다.",
        analogy="시험 점수가 모두의 예상 점수보다 몇 % 높거나 낮았는지 보는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="surprise_pct = (EPS − 예상 EPS) ÷ |예상 EPS| × 100",
            rows=[
                dict(label="가장 최근 발표 1건(유효일 ≤ t)", segs=[seg(-30, -29, "use", "발표일")])
            ],
            marks=[(-30, "최근 발표"), (0, "t")],
            note="발표 시점은 그림용 예시입니다. 값은 원천(earnings_calendar)이 준 서프라이즈를 그대로 씁니다.",
        ),
        principle=dict(
            type="flow",
            title="실적 서프라이즈가 값이 되기까지",
            steps=[
                dict(title="실제 EPS", text="발표된 주당순이익입니다."),
                dict(title="예상 EPS와 비교", text="애널리스트 예상과의 차이를 구합니다."),
                dict(
                    title="|예상|로 나눔",
                    text="예상의 절댓값으로 나눠 퍼센트로 만듭니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="표준편차로 나누는 SUE가 아니라 예상 EPS의 절댓값으로 나눈 비율입니다. 예상이 0에 가까우면 극단값이 됩니다.",
        ),
        what="가장 최근 발표의 (실제 EPS − 예상 EPS) ÷ |예상 EPS| × 100입니다.",
        model="두 모델 모두 안정이고 기대와 같습니다. LightGBM은 38위에서 역U자, Ridge는 36위에서 순위가 오를수록 점수를 올립니다.",
        caution="발표 직후 며칠의 신호인데 월간 값에는 직후 구간이 지난 것이 많이 들어갑니다(코드 구조에서 읽은 추정, 데이터로 확인 안 함). 추정이 없는 종목은 결측입니다. 알려진 지 오래된 이상현상이라 약화가 기본값이고 3차 롱 근거가 없습니다. 8-K 접수 시각이 없는 행은 캘린더 날짜 그대로라 장 마감 뒤 발표에는 하루 이른 값이 될 수 있습니다. Ridge 크기의 약 73%가 _isna 성분입니다.",
    ),
    "days_since_earn": dict(
        title="마지막 실적 발표 후 경과일",
        basis="ev",
        intuition="최근 실적을 발표한 지 며칠이 지났는지입니다. 분기 달력에서 지금 어디쯤인지 알려 주는 달력 변수입니다.",
        analogy="학기 중에 지난 시험을 본 지 며칠이 됐는지 세는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="거래일 수( 최근 발표 유효일 → t )",
            rows=[
                dict(
                    label="최근 발표 유효일부터 t까지",
                    segs=[seg(-30, 0, "use", "경과 거래일 = 값")],
                )
            ],
            marks=[(-30, "최근 발표"), (0, "t")],
            note="발표 시점은 그림용 예시입니다. 값은 이 구간의 거래일 수입니다.",
        ),
        principle=dict(
            type="timeline",
            title="발표 유효일에서 오늘까지",
            events=[
                dict(at=0, label="실적 발표", sub="8-K 2.02 접수 시각 기준 유효일"),
                dict(at=30, label="오늘 t", sub="값이 확정되는 날", kind="today"),
            ],
            spans=[dict(a=0, b=30, label="경과 거래일 = 이 값")],
            caption="16:00 ET 뒤에 접수된 발표는 다음 거래일로 올려 셉니다. 일수는 모양만 보이려는 예시입니다.",
        ),
        what="유효일이 t 이하인 가장 최근 발표부터 t까지의 거래일 수입니다.",
        model="LightGBM은 35위에서 순위가 오를수록 점수가 내려가는 쪽(ρ −0.47)이고 안정이며 기대 부호가 없습니다. Ridge는 불안정이라 해석하지 않습니다.",
        caution="부호를 등록하지 않은 변수라 혼자서는 방향이 없고, 다른 피쳐와 조합할 때 의미가 있습니다. days_to_earn과 한 묶음(g3)입니다.",
    ),
    "days_to_earn": dict(
        title="다음 실적 발표까지 남은 날",
        basis="cal",
        intuition="다음 실적 발표 예정일까지 며칠 남았는지입니다. 그 발표일이 그날 이미 공지돼 있었는지는 확인하지 못했습니다. 1차에서는 두 유니버스 모두 A였고, 3차에서는 방향에 따라 롱·숏 양쪽에서 t 3 안팎이 나왔습니다.",
        analogy="시험 날짜를 보고 시험까지 며칠 남았는지 세는 것과 같습니다. 그 날짜를 그때 이미 알 수 있었는지는 따로 확인해야 합니다.",
        win=dict(
            axis=70,
            formula="거래일 수( t → 다음 예정 발표일 )",
            rows=[
                dict(label="t 이전 구간", segs=[seg(-70, 0, "skip", "안 씀 (다음 발표는 t 이후)")])
            ],
            marks=[(0, "t")],
            note="값은 t 이후의 일정이라 가로축에 못 그립니다. 아래 원리 그림에 그렸습니다.",
        ),
        principle=dict(
            type="timeline",
            title="오늘에서 다음 발표 예정일까지",
            events=[
                dict(at=0, label="오늘 t", sub="값이 확정되는 날", kind="today"),
                dict(at=40, label="다음 발표 예정일", sub="공지 여부 미확인"),
            ],
            spans=[dict(a=0, b=40, label="남은 거래일 = 이 값")],
            caption="8-K 보정 없이 캘린더 원값을 씁니다. 일수는 모양만 보이려는 예시입니다. 캘린더에 미래 일정이 없으면 결측이고, 그 결측이 미래 정보일 수 있습니다.",
        ),
        what="캘린더에서 t 다음으로 잡힌 예정 발표일까지의 거래일 수입니다.",
        model="LightGBM은 9위로 크게 기대고 U자 곡선이며 안정이고 기대 부호가 없습니다. Ridge는 불안정이라 해석하지 않습니다.",
        caution="1차에서는 롱·숏 합산 A였는데 롱만 보면 R이라 신호가 숏 쪽에 있었습니다. 부호를 결과를 본 뒤 정해 낙관 편향이 있고, 라벨 기간 안에 다음 발표가 들어올 확률과 직접 이어집니다. 코드는 다음 실제 행 날짜를 시점 필터 없이 붙이므로 사후 캘린더 look-ahead일 수 있고(PIT 미검증), 3차의 양쪽 t 3 안팎 신호도 크기를 그대로 믿기 어렵습니다.",
    ),
    "n_estimates": dict(
        title="애널리스트 추정치 수",
        basis="ev",
        intuition="이 종목의 실적을 예상한 애널리스트(컨센서스)의 수입니다. 사전등록 부호는 (−)입니다.",
        analogy="식당 앞에서 리뷰가 몇 개 달려 있는지 세는 것과 같습니다. 별점이 아니라 리뷰 수입니다.",
        win=dict(
            axis=70,
            formula="가장 최근 발표 행의 컨센서스 추정치 수",
            rows=[
                dict(label="가장 최근 발표 1행(유효일 ≤ t)", segs=[seg(-30, -29, "use", "발표일")])
            ],
            marks=[(-30, "최근 발표"), (0, "t")],
            note="발표 시점은 그림용 예시입니다. 채움률은 78.0%입니다.",
        ),
        principle=dict(
            type="flow",
            title="추정치 수가 값이 되기까지",
            steps=[
                dict(
                    title="실적 캘린더", text="earnings_calendar에서 가장 최근 발표 행을 찾습니다."
                ),
                dict(
                    title="추정치 수 읽기",
                    text="그 행의 컨센서스 추정치 수를 그대로 읽습니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="클수록 많은 사람이 지켜보는 종목입니다. 결측이면 커버리지가 없는 종목입니다.",
        ),
        what="가장 최근 발표 행의 컨센서스 추정치 수입니다.",
        model="두 모델 모두 불안정이라 해석하지 않습니다(LightGBM 15위, Ridge 23위).",
        caution="규모(시총·거래대금)와 얽혀 있을 가능성이 있지만 상관은 확인하지 못했습니다. sue_last와 같은 행에서 와서 결측 패턴이 겹칩니다. Ridge 크기의 약 78%가 _isna 성분입니다.",
    ),
    "n_8k_90": dict(
        title="최근 90일 8-K 건수",
        basis="ev",
        intuition="최근 석 달 동안 회사가 수시공시(8-K)를 몇 번 냈는지입니다. 사건이 많은 회사일수록 큽니다.",
        analogy="최근 석 달 동안 우리 동네 가게가 공지문을 몇 번 붙였는지 세는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="COUNT( 8-K 유효일 ∈ (t−90일, t] )",
            rows=[dict(label="지난 90일(달력일)의 8-K", segs=[seg(-62, 0, "use", "90일 창")])],
            marks=[(-62, "t−90일"), (0, "t")],
            note="90일은 달력일이고 그림은 거래일 약 63행으로 근사했습니다.",
        ),
        principle=dict(
            type="timeline",
            title="90일 창 안의 8-K 세기",
            events=[
                dict(at=-90, label="t−90일", sub="창의 시작"),
                dict(at=-65, label="8-K", sub="1건"),
                dict(at=-30, label="8-K", sub="1건"),
                dict(at=-10, label="8-K", sub="1건"),
                dict(at=0, label="오늘 t", sub="3건 = 값(예시)", kind="today"),
            ],
            spans=[dict(a=-90, b=0, label="90일 창의 건수 = 이 값")],
            caption="8-K 위치와 건수는 모양만 보이려는 예시입니다. 실적 발표도 8-K(item 2.02)라 분기마다 한 건이 들어갑니다.",
        ),
        what="같은 cik가 낸 8-K 중 유효일이 지난 90일 안에 드는 건수입니다.",
        model="두 모델 모두 안정이고 기대와 같습니다. 순위가 오를수록 점수가 내려가고(LightGBM 29위 ρ −0.85, Ridge 44위 ρ −0.99) 두 모델의 방향이 같습니다.",
        caution="cik가 없는 종목은 결측이고, cik가 있으면 8-K가 없을 때 0입니다. 공시의 좋고 나쁨은 구분하지 못합니다.",
    ),
    "filing_lag": dict(
        title="보고서 제출 지연 일수",
        basis="ev",
        intuition="가장 최근 분기·연간 보고서(10-Q·10-K)를 보고기간이 끝나고 며칠 만에 냈는지입니다. 사전등록 부호는 (−)입니다.",
        analogy="과제 마감일이 끝난 뒤 며칠 만에 제출했는지 세는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="filing_date − report_date (가장 최근 10-K·10-Q)",
            rows=[dict(label="보고기간 끝부터 제출까지", segs=[seg(-45, -5, "use", "지연 = 값")])],
            marks=[(-45, "보고기간 끝"), (-5, "제출"), (0, "t")],
            note="일수는 그림용 예시입니다. 정정본(/A)은 제외합니다.",
        ),
        principle=dict(
            type="timeline",
            title="보고기간 끝에서 제출까지",
            events=[
                dict(at=0, label="보고기간 끝", sub="report_date"),
                dict(at=40, label="10-Q·10-K 제출", sub="filing_date", kind="known"),
            ],
            spans=[dict(a=0, b=40, label="filing_lag = 이 일수")],
            caption="일수는 모양만 보이려는 예시입니다. 어떤 공시를 최근 것으로 고를지는 접수 시각 유효일 규칙을 따르고, 값은 두 날짜를 그대로 뺍니다.",
        ),
        what="가장 최근 10-K·10-Q의 제출일에서 보고기간 끝일을 뺀 일수입니다.",
        model="LightGBM은 24위이지만 불안정이라 해석하지 않습니다. Ridge는 29위에서 순위가 오를수록 점수를 내리며(ρ −0.94) 안정이고 기대와 같습니다.",
        caution="롱·숏 합산에서 A, 롱만 보면 R이었다가 3차에서 B로 올라온 불안정한 피쳐입니다. 제출 기한 근처에 값이 몰려 동점 몫이 큽니다. t 2.16도 44개 Bonferroni 문턱(약 3.25) 아래이고, 2026-06-30 판의 B는 열린 holdout을 포함합니다.",
    ),
}
EXAMPLES = {
    "sp_ttm": [dict(kind="p95"), dict(kind="p05")],
    "roa_ttm": [dict(kind="p95"), dict(kind="p05")],
    "days_to_earn": [dict(kind="p95"), dict(kind="p05")],
}
EX_WINDOWS = {"sp_ttm": [], "roa_ttm": [], "days_to_earn": []}

HTML_OVERRIDE = dict(
    LEDE='미국 서빙 모델 <code>us_exploratory_20260929_r1</code>(LightGBM·Ridge)이 쓰는 재무·실적·공시 피쳐 19개를 그림으로 풀었습니다. 정의와 직관의 정본은 <a href="05_us_fundamental_event.md"><code>05_us_fundamental_event.md</code></a>이고, 이 페이지는 그 그림판입니다. 재무 숫자는 SEC 접수일(달력일)부터, 실적·공시는 접수 시각 기준 유효일부터 모델에 들어갑니다. 재무에는 16:00 ET 보정이 없고(영향 크기 미측정) 시총에는 분할 보정이 없으며, <code>days_to_earn</code>은 PIT가 미검증입니다. 모델 반응 그림(③·④·⑤)은 <b>라벨을 읽지 않고</b> 월초 12개 횡단면으로 만든 <b>진단이며 모델 점수의 반응이지 수익 기여가 아닙니다.</b> 가로축은 값이 아니라 그날 종목들 사이의 순위(백분위)입니다. 단변량 수치는 비용 전이고 개발 구간을 다섯 번 봤으며 holdout은 이미 열렸습니다. Ridge 크기에서 <code>_isna</code> 성분 비중이 큰 카드(<code>cfp_ttm</code> 약 98% 등)는 값이 아니라 "자료가 있느냐"에 반응한 것일 수 있습니다. 층 3 입력은 동결 데이터셋이라 서빙 규칙 A 값과 86셀 다르고, 종목 단위 신호라 연금계좌(ETF만)에서는 직접 못 씁니다.',
    MAPCAP="회색 선보다 위에 있는 피쳐는 Ridge 쪽 크기가 합계 비율보다 상대적으로 크고, 아래에 있으면 LightGBM 쪽이 큽니다. 두 모델의 점수 척도가 달라 위치로만 읽습니다. 모양과 안정 판정의 기준은 <a href='10_us_result.md'><code>10_us_result.md</code></a> §1에 있습니다.",
    GROUPSINTRO="날짜별 순위 상관 |ρ| ≥ 0.7인 피쳐끼리 묶은 것 가운데 이 문서의 피쳐가 든 묶음은 <code>days_since_earn</code>과 <code>days_to_earn</code> 하나뿐입니다. 나머지 17개는 단독 피쳐입니다. <b>묶음 안에서는 서로 몫을 나눠 가집니다.</b> 한쪽을 빼도 이웃이 대신하고, Ridge는 계수가 서로 반대 부호로 상쇄하기도 합니다.",
)
_EXTAIL = " 가격 그림은 값의 모습을 짐작하는 참고용이고, 이 피쳐는 가격 창이 아니라서 음영이 없습니다. 가격은 기준일까지의 일봉 종가이고 그 뒤 가격은 쓰지 않았습니다."
EX_NOTES = {
    "sp_ttm": "{name}의 sp_ttm은 {value}(순위 {pct})입니다. 접수된 매출 4분기 합과 시총으로 만든 값입니다."
    + _EXTAIL,
    "roa_ttm": "{name}의 roa_ttm은 {value}(순위 {pct})입니다. 접수된 순이익 4분기 합과 최신 총자산으로 만든 값입니다."
    + _EXTAIL,
    "days_to_earn": "{name}은 다음 실적 발표 예정일까지 {value}거래일 남았습니다(순위 {pct}). 캘린더의 다음 예정일에서 센 값이고, 그날 이미 공지돼 있었는지는 미검증입니다."
    + _EXTAIL,
}
