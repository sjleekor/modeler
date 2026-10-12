# ruff: noqa: E501
"""06 US 수급·보유 피쳐 17개 그림판 config."""

DOC_DIR = "milestones/common/20261012_feature_explainer"
OUT_NAME = "06_us_flow_holders.html"
DOC_MD = "06_us_flow_holders.md"
RESULT_MD = "10_us_result.md"
RUN_ID = "us_l3_20261012T112931Z"
PANEL_V2_REL = "us/datasets/us_features_v2/part.parquet"
PANEL_FLOW_REL = "us/datasets/us_features_flow_v1/part.parquet"
PRICES_GLOB = "us/derived/snapshots/prices_daily/snapshot_date=*/part.parquet"
COMPANY_META_GLOB = "us/derived/snapshots/company_meta/snapshot_date=*/part.parquet"
MARKET = "us"
REF_DATE = "2026-06-01"
PAGE_TITLE = "US 수급·보유 피쳐 그림 해설"
MODELER_COMMIT = "d98d5cf"
N_ICE = 40
EX_DAYS = 260

THEMES = [
    (
        "내부자 거래",
        "임원·대주주가 자기 돈으로 산 주식을 SEC 공시로 센 피쳐 3개입니다. 공시일 기준 90일 창을 씁니다.",
        ["ins_netbuy_90", "ins_cluster_90", "ins_officer_buy_90"],
    ),
    (
        "공매도",
        "빌려서 판 주식이 얼마나 쌓였는지(잔고, 월 2회)와 거래소 밖 공매도 비중(일별)입니다.",
        ["si_ratio", "dtc", "si_chg", "sv_share_20"],
    ),
    (
        "결제 실패(FTD)",
        "SEC 반월 파일의 결제 실패 수량을 20결제일로 본 피쳐 3개입니다. 반월 끝 + 20달력일 뒤부터 씁니다.",
        ["ftd_share_20", "ftd_days_20", "ftd_chg"],
    ),
    (
        "주문 흐름(MIDAS)",
        "SEC MIDAS 분기 파일에서 20거래일 합의 비율로 만든 피쳐 4개입니다. 방향을 등록하지 않았고, 2025년 분기부터 발행이 몇 달씩 늦습니다.",
        ["cancel_ratio_20", "hidden_share_20", "oddlot_share_20", "fill_ratio_20"],
    ),
    (
        "기관 보유(13F)",
        "분기마다 공시되는 13F 보유 보고로 만든 피쳐 3개입니다. 분기말 + 60달력일 뒤부터 씁니다.",
        ["inst_n_log", "inst_breadth_chg", "inst_shares_chg"],
    ),
]
FMT = {
    "ins_netbuy_90": "ratio",
    "ins_cluster_90": "int",
    "ins_officer_buy_90": "int",
    "si_ratio": "num",
    "dtc": "num",
    "si_chg": "num",
    "sv_share_20": "pct",
    "ftd_share_20": "pct",
    "ftd_days_20": "int",
    "ftd_chg": "pct",
    "cancel_ratio_20": "num",
    "hidden_share_20": "pct",
    "oddlot_share_20": "pct",
    "fill_ratio_20": "pct",
    "inst_n_log": "num",
    "inst_breadth_chg": "num",
    "inst_shares_chg": "num",
}
BASIS = {
    "ins": "공시일 기준 · 달력일 90일 · 공시일이 t 이하인 건만",
    "si": "격주 결제일 · 결제일 + 10거래일부터 · 다음 값이 날 때까지 이어 씀",
    "sv": "거래일 20일 평균 · 그날 값까지",
    "ftd": "결제일 격자 20결제일 · 반월 끝 + 20달력일부터",
    "midas": "거래일 20일 합의 비율 · 분기 파일 발행 뒤부터",
    "f13": "분기말 보유 · 분기말 + 60달력일부터 · 분기 내내 같은 값",
}
GROUP_NAMES = {
    "g2": "규모·유동성 (이 문서에서는 inst_n_log)",
    "g4": "내부자 매수 묶음",
    "g5": "공매도 잔고 묶음",
    "g6": "소량 체결·체결 비중 묶음",
}
SCATTER_LABELS = [
    "dtc",
    "si_ratio",
    "sv_share_20",
    "inst_n_log",
    "inst_shares_chg",
    "fill_ratio_20",
]


def seg(a, b, kind="use", text=""):
    return {"a": a, "b": b, "kind": kind, "text": text}


def st(title, text, **kw):
    return dict(title=title, text=text, **kw)


CARDS = {
    # ---------------- 내부자 ----------------
    "ins_netbuy_90": dict(
        title="3개월 내부자 순매수 비율",
        basis="ins",
        intuition="회사 사정을 가장 잘 아는 임원·대주주가 최근 3개월 동안 자기 돈으로 주식을 팔기보다 사고 있다면 앞으로 좋아질 일을 알고 있을 수 있다는 이야기입니다. 우리 데이터에서는 확인되지 않았습니다.",
        analogy="가게 주인이 손님이 되어 자기 가게 물건을 사 가는지, 내다 파는지 지켜보는 것과 같습니다.",
        win=dict(
            axis=270,
            formula="( 매수금액 − 매도금액 ) / ( 매수금액 + 매도금액 ),  공시일이 (t−90일, t] 안인 건",
            rows=[dict(label="공시일 기준 90일", segs=[seg(-90, 0, "use", "90일")])],
            marks=[(-90, "t−90"), (0, "t")],
            note="달력일 기준이고 거래일이 아니라 공시일로 잰 창입니다. 장내 매수(P)·매도(S)만 셉니다. 둘 다 0이면 결측입니다.",
        ),
        principle=dict(
            type="ratio",
            title="순매수 비율은 −1에서 +1 사이입니다",
            unit="",
            rows=[
                dict(
                    label="매수 우위 (가정)",
                    numLabel="순매수",
                    num=20,
                    denLabel="합계",
                    den=100,
                    result="(60 − 40) ÷ (60 + 40) = +0.2",
                ),
                dict(
                    label="매도 우위 (가정)",
                    numLabel="순매수",
                    num=-20,
                    denLabel="합계",
                    den=40,
                    result="(10 − 30) ÷ (10 + 30) = −0.5",
                ),
                dict(
                    label="매수만 있음 (가정)",
                    numLabel="순매수",
                    num=50,
                    denLabel="합계",
                    den=50,
                    result="(50 − 0) ÷ (50 + 0) = +1.0",
                ),
            ],
            caption="숫자는 원리를 보이기 위한 가정입니다.",
        ),
        what="최근 90일에 공시된 내부자 장내 매수와 매도 금액의 차이를 합으로 나눈 값입니다.",
        model="LightGBM에서는 크기가 52위로 작고 곡선이 평평합니다. Ridge는 불안정해서 해석하지 않습니다.",
        caution="스톡옵션 행사·수여·세금 납부는 빠집니다. 결측이 58.0%(1차)로 많아 결측 표시(_isna)가 큰 몫을 합니다.",
    ),
    "ins_cluster_90": dict(
        title="3개월 매수 내부자 수",
        basis="ins",
        intuition="한 사람이 아니라 여러 명이 같은 기간에 사 모았다면 개인 사정보다 회사 전망이 이유일 가능성이 높다는 이야기입니다. 우리 데이터에서 확인된 적은 없습니다.",
        analogy="한 명이 줄을 서면 개인 취향이지만, 여러 명이 같은 가게 앞에 줄을 서면 이유가 있어 보이는 것과 같습니다.",
        win=dict(
            axis=270,
            formula="COUNT DISTINCT( 매수(P) 공시의 신고인 owner_cik ),  공시일 90일",
            rows=[dict(label="공시일 기준 90일", segs=[seg(-90, 0, "use", "90일")])],
            marks=[(-90, "t−90"), (0, "t")],
            note="달력일 기준입니다. 값 대부분이 0이라 같은 값끼리 가장 낮은 순위를 함께 받습니다.",
        ),
        principle=dict(
            type="flow",
            title="공시가 '사 모은 사람 수'가 되기까지",
            steps=[
                st("내부자 공시", "Form 4·5로 올라온 거래 공시를 모읍니다."),
                st("장내 매수만", "공개시장 매수(P) 공시만 남깁니다."),
                st(
                    "신고인 수 세기",
                    "신고인 이름을 겹치지 않게 셉니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="한 공시에 신고인이 최대 10명이라 공동 신고 하나가 수를 키웁니다.",
        ),
        what="최근 90일 매수 공시에 이름이 오른 서로 다른 신고인의 수입니다.",
        model="LightGBM에서는 크기가 44위이고 곡선이 평평합니다. Ridge는 불안정해서 해석하지 않습니다.",
        caution="대부분 0이라 소수만 위 순위로 갑니다. 동점이 많아 상위 100개 바스켓이 임의 추출에 가깝습니다.",
    ),
    "ins_officer_buy_90": dict(
        title="3개월 임원·이사 매수 건수",
        basis="ins",
        intuition="일반 대주주보다 회사 사정을 잘 아는 임원·이사가 직접 샀는지만 따로 세는 피쳐입니다. 방향이 맞을 수는 있어도 우리 데이터에서 확인된 것은 없습니다.",
        analogy="가게 손님 중에서도 점장이나 이사가 직접 장을 봤는지만 따로 세어 보는 것과 같습니다.",
        win=dict(
            axis=270,
            formula="COUNT( 매수(P) 공시 중 신고인에 임원·이사가 한 명이라도 있는 건 ),  공시일 90일",
            rows=[dict(label="공시일 기준 90일", segs=[seg(-90, 0, "use", "90일")])],
            marks=[(-90, "t−90"), (0, "t")],
            note="달력일 기준입니다.",
        ),
        principle=dict(
            type="flow",
            title="공동 신고에서 임원이 한 명만 있어도 셉니다",
            steps=[
                st("매수 공시", "공개시장 매수(P) 공시를 모읍니다."),
                st("신고인 확인", "신고인 중 임원 또는 이사가 한 명이라도 있는지 봅니다."),
                st(
                    "그 건을 셈",
                    "있으면 그 공시 한 건을 전부 셉니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="신고인이 여럿이어도 임원이 한 명 섞여 있으면 한 건으로 셉니다.",
        ),
        what="최근 90일에 임원·이사가 낀 장내 매수 공시의 건수입니다.",
        model="두 모델 모두 안정이고, 순위가 오를수록 점수가 오르는 쪽이라 기대와 같습니다. 크기는 LightGBM 41위, Ridge 30위로 작습니다.",
        caution="ins_netbuy_90·ins_cluster_90과 같은 매수 공시에서 나와 서로 겹칩니다. 겹침 정도는 확인하지 못했습니다.",
    ),
    # ---------------- 공매도 ----------------
    "si_ratio": dict(
        title="거래량 대비 공매도 잔고",
        basis="si",
        intuition="남들이 빌려서 판 채 아직 되사지 않은 주식이 평소 거래량 대비 많을수록 비관적인 투자자가 많은 종목이라는 이야기입니다. 기대 방향은 (−)이고, 우리 개별 검정에서 확인된 기록은 없습니다.",
        analogy="빌린 책을 아직 못 돌려준 사람이 동네에 얼마나 많은지 세는 것과 같습니다. 많을수록 앞으로 책을 사서 돌려줄 사람이 많다는 뜻입니다.",
        win=dict(
            axis=30,
            formula="공매도 잔고 주수 / 평균 일 거래량(주수)",
            rows=[
                dict(label="공표 전에는 모름", segs=[seg(-10, 0, "skip", "10거래일")]),
                dict(
                    label="쓰는 값은 이 앞의 최신 결제일", segs=[seg(-20, -10, "use", "결제일 값")]
                ),
            ],
            marks=[(-20, "t−20"), (-10, "t−10"), (0, "t")],
            note="월 2회 갱신되는 값이라 모델 입력은 2~4주 전 상태입니다. 결제일 + 10거래일부터 쓰고 다음 값이 날 때까지 마지막 값을 이어 씁니다.",
        ),
        principle=dict(
            type="flow",
            title="공매도 잔고는 이렇게 쌓입니다",
            steps=[
                st("주식을 빌림", "공매도 투자자가 증권사에서 주식을 빌립니다."),
                st("시장에 팜", "빌린 주식을 오늘 가격에 먼저 팝니다."),
                st("아직 안 갚음", "나중에 사서 갚아야 합니다. 갚지 않은 수량이 잔고입니다."),
                st(
                    "거래량으로 나눔",
                    "잔고를 평소 일 거래량으로 나눕니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="값이 클수록 거래량에 비해 되사야 할 주식이 많이 쌓인 종목입니다.",
        ),
        what="결제일 기준 공매도 잔고 주수를 평균 일 거래량(주수)으로 나눈 값입니다.",
        model="Ridge가 7위로 크게 기대고, 순위가 오를수록 점수가 내려갑니다(기대와 같음). LightGBM은 28위이고 곡선이 역U입니다.",
        caution="dtc와 정의가 비슷합니다. Ridge에서는 두 피쳐의 계수가 서로 상쇄해 묶음 합계가 작아집니다.",
    ),
    "dtc": dict(
        title="되사는 데 걸리는 날수",
        basis="si",
        intuition="공매도 잔고를 평소 거래량으로 다 사들이려면 며칠이 걸리는지입니다. 길수록 공매도 투자자가 빠져나오기 어렵습니다.",
        analogy="가게 앞에 쌓인 짐을 하루에 나르는 양으로 나누면 며칠 걸리는지 나오는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="FINRA days_to_cover (재계산하지 않고 그대로)",
            rows=[
                dict(label="공표 전에는 모름", segs=[seg(-10, 0, "skip", "10거래일")]),
                dict(
                    label="쓰는 값은 이 앞의 최신 결제일", segs=[seg(-20, -10, "use", "결제일 값")]
                ),
            ],
            marks=[(-20, "t−20"), (-10, "t−10"), (0, "t")],
            note="월 2회 갱신, 결제일 + 10거래일부터 씁니다.",
        ),
        principle=dict(
            type="ratio",
            title="같은 잔고, 다른 거래량",
            unit="",
            rows=[
                dict(
                    label="거래가 활발한 종목 (가정)",
                    numLabel="잔고",
                    num=600,
                    denLabel="거래량",
                    den=300,
                    result="600 ÷ 300 = 2일 만에 되살 수 있음",
                ),
                dict(
                    label="거래가 한산한 종목 (가정)",
                    numLabel="잔고",
                    num=600,
                    denLabel="거래량",
                    den=50,
                    result="600 ÷ 50 = 12일이 걸림",
                ),
            ],
            caption="숫자는 원리를 보이기 위한 가정입니다.",
        ),
        what="공매도 잔고를 평소 일 거래량으로 나눈 일수입니다. FINRA가 계산한 값을 그대로 씁니다.",
        model="두 모델 모두 상위 10위 안(LightGBM 10위, Ridge 9위)이지만 방향이 갈립니다. LightGBM은 순위가 오를수록 점수가 내려가고(기대와 같음), Ridge는 올라갑니다(설명 못 함).",
        caution="1차 후속이 이 한 칸의 근거를 약하다고 결론 냈습니다. 되사기 급등(숏 스퀴즈)은 부호를 반대로 만들 수 있으나 우리 문서에서 다루지 않았습니다.",
    ),
    "si_chg": dict(
        title="공매도 잔고 변화율",
        basis="si",
        intuition="지난 결제일보다 공매도 잔고가 늘었는지입니다. 늘어난 종목은 비관이 새로 쌓인 종목이라는 이야기입니다.",
        analogy="지난번 점검 때보다 창고에 쌓인 짐이 늘었는지 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="FINRA change_percent (직전 결제일 대비 잔고 변화율, %)",
            rows=[
                dict(label="공표 전에는 모름", segs=[seg(-10, 0, "skip", "10거래일")]),
                dict(
                    label="쓰는 값은 이 앞의 최신 결제일", segs=[seg(-20, -10, "use", "결제일 값")]
                ),
            ],
            marks=[(-20, "t−20"), (-10, "t−10"), (0, "t")],
            note="반월 갱신이라 한 달에 값이 두 번 바뀝니다.",
        ),
        principle=dict(
            type="timeline",
            title="공표가 늦어서 값은 한참 묵습니다",
            events=[
                dict(at=0, label="결제일", sub="7영업일째에 FINRA 공표"),
                dict(at=10, label="모델이 쓰는 날", sub="결제일 + 10거래일", kind="today"),
            ],
            spans=[dict(a=0, b=10, label="지연 10거래일")],
            known={"from": 10, "label": "이 값을 쓰는 기간"},
            caption="FINRA가 7영업일째에 공표하고 나스닥이 그날 16:00 ET 뒤 배포하므로 코드는 10거래일 뒤부터 씁니다.",
        ),
        what="직전 결제일보다 공매도 잔고가 몇 % 늘거나 줄었는지입니다.",
        model="두 모델 모두 순위가 오를수록 점수가 오르는 쪽이라 기대 (−)와 반대입니다(설명 못 함). 크기는 LightGBM 34위, Ridge 42위로 작습니다. 반대 결과에 이야기를 붙이지 않았습니다.",
        caution="분모가 작은 종목은 변화율이 크게 튈 수 있습니다. 분모 처리는 FINRA 정의라 확인하지 못했습니다.",
    ),
    "sv_share_20": dict(
        title="거래소 밖 거래 중 공매도 비중",
        basis="sv",
        intuition="거래소 밖에서 체결된 거래량 중 공매도 비중이 평소 높은 종목입니다. 1차에서 처음 보인 신호였지만 시장 전체 공매도가 아니고, 공식 유니버스(5달러 이상)에서는 유지되지 않았습니다.",
        analogy="동네 시장 전체가 아니라 뒷골목 가판에서만 '빌려서 파는 물건'이 얼마나 되는지 세는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="AVG( (short_volume + short_exempt_volume) / total_volume , 20거래일 )",
            rows=[dict(label="일별 비율의 평균", segs=[seg(-19, 0, "use", "20거래일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="그날 값까지 그날에 씁니다(지연 상수 없음). 창이 20일 미만이면 결측입니다.",
        ),
        principle=dict(
            type="ratio",
            title="분모는 시장 전체가 아니라 이 파일의 합계입니다",
            unit="",
            rows=[
                dict(
                    label="이 피쳐 (파일 안 합계로 나눔)",
                    numLabel="공매도",
                    num=20,
                    denLabel="파일 합계",
                    den=40,
                    result="20 ÷ 40 = 0.50",
                ),
                dict(
                    label="시장 전체로 나누면 (가정)",
                    numLabel="공매도",
                    num=20,
                    denLabel="시장",
                    den=100,
                    result="20 ÷ 100 = 0.20, 2.5배 낮게 나옵니다",
                ),
            ],
            caption="2026-08-28 기준 이 파일의 total_volume은 일반 거래량의 중앙값 39.2%입니다. 위 숫자는 이 비율을 보이려는 가정입니다.",
        ),
        what="거래소 밖 공개 거래만 모은 파일에서 공매도 거래량이 차지하는 비중의 20거래일 평균입니다.",
        model="LightGBM에서 7위로 크게 기대고 순위가 오를수록 점수가 내려갑니다(기대와 같음). Ridge는 불안정해서 해석하지 않습니다.",
        caution="1차 전체 유니버스 A였으나 공식(5달러 이상) 유니버스에서는 R입니다. 정의가 문헌과 다르고 시장조성자 헤지 거래도 섞여 있습니다.",
    ),
    # ---------------- 결제 실패 ----------------
    "ftd_share_20": dict(
        title="결제 실패 수량 비율 (20결제일)",
        basis="ftd",
        intuition="최근 20결제일 동안 결제가 안 된 주식이 전체 거래량의 몇 %였느냐입니다. 높을수록 주식을 빌리기 어려운 종목이라는 이야기입니다. 4차에서는 롱 바스켓에서는 안 나오고 피할 쪽에서만 유의했습니다.",
        analogy="택배 주문은 받았는데 창고에 물건이 없어 배송이 밀린 건수가 거래량의 몇 %인지 세는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="Σ 20결제일 결제실패 수량 / Σ 같은 20결제일 거래량",
            rows=[dict(label="20결제일 합", segs=[seg(-19, 0, "use", "20결제일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="결제일 격자는 전 종목 공통이고, 창 안 유효일이 10 미만이면 결측입니다. 값은 결제일이 속한 반월 끝 + 20달력일 뒤부터 씁니다.",
        ),
        principle=dict(
            type="flow",
            title="결제 실패가 값이 되기까지",
            steps=[
                st("주식을 팜", "매도가 체결됩니다."),
                st(
                    "결제일에 못 넘김",
                    "결제일까지 주식을 인도하지 못한 수량이 결제 실패(FTD)입니다.",
                ),
                st("SEC 반월 파일", "반월 끝 + 20달력일부터 모델이 쓸 수 있습니다."),
                st(
                    "거래량으로 나눔",
                    "20결제일 합을 같은 기간 거래량 합으로 나눕니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="SEC 파일은 결제일 기준 누적 미결제 잔고이고, 실패의 나이는 알 수 없습니다.",
        ),
        what="최근 20결제일의 결제 실패 수량 합을 같은 기간 거래량 합으로 나눈 값입니다.",
        model="Ridge에서는 안정이고 순위가 오를수록 점수가 내려갑니다(기대와 같음, 27위). LightGBM은 불안정해서 해석하지 않습니다.",
        caution="4차 X(G1)입니다. 결제 실패가 가장 많은 100개 쪽(반대편)이 시장보다 못했을 뿐 롱 바스켓은 유의하지 않았습니다. 분할이 창에 겹치면 값이 왜곡되고, 문헌 근거도 확인하지 못했습니다.",
    ),
    "ftd_days_20": dict(
        title="결제 실패가 난 날 수 (20결제일)",
        basis="ftd",
        intuition="최근 20결제일 중 결제 실패가 한 번이라도 난 날이 며칠인지입니다. ftd_share_20이 규모라면 이것은 얼마나 자주인지를 봅니다.",
        analogy="배송이 밀린 물량이 많았는지가 아니라, 한 달 중 배송이 밀린 날이 며칠이었는지 세는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="COUNT( 20결제일 중 결제실패 수량 > 0 인 날 )",
            rows=[dict(label="20결제일 창", segs=[seg(-19, 0, "use", "20결제일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="창 안 유효일이 10 미만이면 결측입니다.",
        ),
        principle=dict(
            type="timeline",
            title="SEC 파일은 반월이 끝나고 한참 뒤에 나옵니다",
            events=[
                dict(at=0, label="반월 끝", sub="결제일이 속한 반월"),
                dict(at=20, label="모델이 쓰는 날", sub="반월 끝 + 20달력일", kind="today"),
            ],
            spans=[dict(a=0, b=20, label="지연 20달력일")],
            known={"from": 20, "label": "이 값을 쓰는 기간"},
            caption="2021~2026 파일 138개의 공시 지연은 중앙값 15일, p90 18일입니다. 20일을 넘은 파일은 3개(2.2%)이고, 그 반월만 미래 값이 들어갈 수 있습니다.",
        ),
        what="최근 20결제일 중 결제 실패가 난 날이 며칠인지(0~20)입니다.",
        model="Ridge에서는 안정이고 순위가 오를수록 점수가 내려갑니다(기대와 같음, 37위). LightGBM은 불안정해서 해석하지 않습니다.",
        caution="값이 0~20 정수에 0이 많아 동점이 극심합니다. 4차 X(G1)이고 동점 몫이 0.46이라 바스켓 절반이 임의 추출입니다.",
    ),
    "ftd_chg": dict(
        title="결제 실패 비율의 변화 (21거래일)",
        basis="ftd",
        intuition="지금 결제 실패 비율이 약 한 달 전보다 늘었는지 줄었는지입니다. 등록한 방향은 (−)였는데 결과는 반대로 나왔습니다.",
        analogy="이번 달 배송 지연율에서 한 달 전 지연율을 빼서, 나아졌는지 나빠졌는지 보는 것과 같습니다.",
        win=dict(
            axis=70,
            formula="ftd_share_20[t] − ftd_share_20[t−21행]",
            rows=[
                dict(label="지금의 20결제일 비율", segs=[seg(-19, 0, "use", "20결제일")]),
                dict(
                    label="21행 전의 20결제일 비율 (뺌)", segs=[seg(-40, -21, "minus", "빼는 창")]
                ),
            ],
            marks=[(-40, "t−40"), (-21, "t−21"), (0, "t")],
            note="결제일 격자 기준 21행입니다. 시프트한 값이 비면 같이 빕니다.",
        ),
        principle=dict(
            type="flow",
            title="두 시점의 비율을 뺍니다",
            steps=[
                st("지금 비율", "ftd_share_20의 지금 값입니다."),
                st("한 달 전 비율", "21행 전의 ftd_share_20입니다."),
                st("차이", "지금 값에서 한 달 전 값을 뺍니다.", hl=True, tag="여기서 재는 값"),
            ],
            measure="양수면 결제 실패 비율이 늘었고, 음수면 줄었습니다.",
        ),
        what="ftd_share_20의 지금 값에서 21행 전 값을 뺀 변화입니다.",
        model="두 모델 모두 안정이고 순위가 오를수록 점수가 오르는 쪽이라 기대 (−)와 반대입니다(설명 못 함). 크기는 둘 다 33위입니다.",
        caution="4차에서 등록 방향의 롱 결과가 반대로 나왔습니다. 부호는 사전등록 그대로 두고 뒤집지 않기로 했습니다. 같은 입력에서 값이 실행마다 달라진 적이 있습니다.",
    ),
    # ---------------- MIDAS ----------------
    "cancel_ratio_20": dict(
        title="주문 취소율",
        basis="midas",
        intuition="거래소에 낸 주문 중 거래가 되지 못하고 취소된 비율입니다. 알고리즘 거래가 잦은 종목일수록 높을 수 있습니다. 방향은 모르고, 우리 결과도 방향과 무관하게 안 됐습니다.",
        analogy="식당에서 주문을 넣었다가 취소하는 손님이 많은지, 끝까지 먹는 손님이 많은지 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="Σ 20일 cancels / Σ 20일 lit_trades",
            rows=[dict(label="20거래일 합", segs=[seg(-19, 0, "use", "20거래일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="분기 파일 발행 뒤부터 쓰고, 2025q1부터는 발행이 128~289일 늦습니다. 표에 없는 분기는 분기 끝 + 60일입니다.",
        ),
        principle=dict(
            type="ratio",
            title="같은 거래 수, 다른 취소 수",
            unit="",
            rows=[
                dict(
                    label="취소가 많은 종목 (가정)",
                    numLabel="취소",
                    num=300,
                    denLabel="체결 거래",
                    den=100,
                    result="300 ÷ 100 = 3.0",
                ),
                dict(
                    label="취소가 적은 종목 (가정)",
                    numLabel="취소",
                    num=30,
                    denLabel="체결 거래",
                    den=100,
                    result="30 ÷ 100 = 0.3",
                ),
            ],
            caption="숫자는 원리를 보이기 위한 가정입니다.",
        ),
        what="20거래일 동안의 주문 취소 수를 거래소 체결 거래 수(lit_trades)로 나눈 값입니다.",
        model="두 모델 모두 안정이고 기대 부호가 없습니다. 크기는 LightGBM 49위, Ridge 50위로 작습니다.",
        caution="MIDAS는 REIT·SPAC·ADR 등 다섯 부류를 안 싣습니다. 결측이 30.8%이고 결측 표시(_isna)가 그 부류 지시변수가 됩니다.",
    ),
    "hidden_share_20": dict(
        title="숨은 주문 체결 비중",
        basis="midas",
        intuition="호가창에 보이지 않는 숨은 주문으로 체결된 거래량의 비중입니다. 큰 주문을 감추려는 기관이 많은 종목일 수 있지만 우리 데이터에서는 방향이 확인되지 않았습니다.",
        analogy="진열대에 내놓지 않고 창고에서 바로 꺼내 파는 물건이 전체 판매의 몇 %인지 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="Σ 20일 hidden_vol_k / Σ 20일 trade_vol_for_hidden_k",
            rows=[dict(label="20거래일 합", segs=[seg(-19, 0, "use", "20거래일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="분기 파일 발행 뒤부터 씁니다. 원천의 hidden 열은 음수가 될 수 있습니다.",
        ),
        principle=dict(
            type="ratio",
            title="숨은 주문 체결 비중",
            unit="",
            rows=[
                dict(
                    label="숨은 주문이 많은 종목 (가정)",
                    numLabel="숨은 주문",
                    num=40,
                    denLabel="거래량",
                    den=100,
                    result="40 ÷ 100 = 0.40",
                ),
                dict(
                    label="숨은 주문이 적은 종목 (가정)",
                    numLabel="숨은 주문",
                    num=5,
                    denLabel="거래량",
                    den=100,
                    result="5 ÷ 100 = 0.05",
                ),
            ],
            caption="숫자는 원리를 보이기 위한 가정입니다.",
        ),
        what="20거래일 동안 숨은 주문 체결량을 분모가 되는 거래량으로 나눈 값입니다.",
        model="LightGBM 12위·Ridge 20위로 비교적 위쪽이지만 두 모델 모두 불안정해서 해석하지 않습니다.",
        caution="4차에서 한쪽은 G1, 다른 쪽은 G2(사이즈를 빼면 사라짐)로 탈락했습니다. 결측 30.8%와 발행 지연은 cancel_ratio_20과 같습니다.",
    ),
    "oddlot_share_20": dict(
        title="소량(100주 미만) 체결 비중",
        basis="midas",
        intuition="100주가 안 되는 소량 체결의 비중입니다. 개인 투자자가 많은 종목의 대용치로 쓰려던 피쳐입니다. 고가주는 소량 체결이 기계적으로 많아져 개인 비중과 다를 수 있습니다.",
        analogy="마트에서 낱개로 사 가는 손님의 비중을 세는 것과 같습니다. 단가가 비싼 물건은 낱개로만 팔려서 낱개 비중이 저절로 높아집니다.",
        win=dict(
            axis=30,
            formula="Σ 20일 odd_lot_vol_k / Σ 20일 trade_vol_for_odd_lots_k",
            rows=[dict(label="20거래일 합", segs=[seg(-19, 0, "use", "20거래일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="분기 파일 발행 뒤부터 씁니다.",
        ),
        principle=dict(
            type="ratio",
            title="소량 체결 비중",
            unit="",
            rows=[
                dict(
                    label="소량 체결이 많은 종목 (가정)",
                    numLabel="소량",
                    num=30,
                    denLabel="거래량",
                    den=100,
                    result="30 ÷ 100 = 0.30",
                ),
                dict(
                    label="소량 체결이 적은 종목 (가정)",
                    numLabel="소량",
                    num=5,
                    denLabel="거래량",
                    den=100,
                    result="5 ÷ 100 = 0.05",
                ),
            ],
            caption="가격이 높은 종목도 위로 올라올 수 있습니다. 숫자는 원리를 보이기 위한 가정입니다.",
        ),
        what="20거래일 동안 100주 미만 소량 체결량을 분모가 되는 거래량으로 나눈 값입니다.",
        model="LightGBM은 불안정해서 해석하지 않습니다. Ridge는 안정이고 순위가 오를수록 점수가 내려가지만 기대 부호가 없습니다(41위).",
        caution="문헌이 (−)와 (+)로 갈려 부호를 등록하지 않았습니다. 2025-11 라운드랏 재정의가 이 값에 반영됐는지는 확인하지 못했습니다.",
    ),
    "fill_ratio_20": dict(
        title="주문 대비 체결 비중",
        basis="midas",
        intuition="낸 주문 수량 중 실제로 체결된 비중입니다. 유동성이 얇은 종목일수록 낮을 수 있다고 읽을 수 있지만 우리는 방향을 정하지 않았습니다.",
        analogy="주문서에 적어 낸 양 가운데 실제로 받아 간 양이 얼마나 되는지 보는 것과 같습니다.",
        win=dict(
            axis=30,
            formula="Σ 20일 lit_vol_k / Σ 20일 order_vol_k",
            rows=[dict(label="20거래일 합", segs=[seg(-19, 0, "use", "20거래일")])],
            marks=[(-19, "t−19"), (0, "t")],
            note="분기 파일 발행 뒤부터 씁니다.",
        ),
        principle=dict(
            type="ratio",
            title="낸 주문 중 체결된 몫",
            unit="",
            rows=[
                dict(
                    label="체결이 잘 되는 종목 (가정)",
                    numLabel="체결",
                    num=80,
                    denLabel="낸 주문",
                    den=100,
                    result="80 ÷ 100 = 0.80",
                ),
                dict(
                    label="체결이 잘 안 되는 종목 (가정)",
                    numLabel="체결",
                    num=25,
                    denLabel="낸 주문",
                    den=100,
                    result="25 ÷ 100 = 0.25",
                ),
            ],
            caption="숫자는 원리를 보이기 위한 가정입니다.",
        ),
        what="20거래일 동안 거래소 체결량을 낸 주문량으로 나눈 값입니다.",
        model="두 모델 모두 안정이고 기대 부호가 없으며, 크기는 둘 다 53개 중 53위로 가장 작습니다.",
        caution="롱 쪽은 못 벌었습니다. 분모(주문량)의 정확한 정의는 이번에 열지 않았습니다.",
    ),
    # ---------------- 13F ----------------
    "inst_n_log": dict(
        title="기관 보유자 수 (로그)",
        basis="f13",
        intuition="그 종목을 13F로 보고한 기관이 몇 곳인지입니다. 기관이 거의 안 보는 종목은 공매도가 어려워 고평가가 남았다가 정정된다는 이야기가 있습니다. 4차에서 적게 담긴 쪽이 롱 바스켓으로 가장 크게 못한 것이 가장 강한 개별 증거입니다.",
        analogy="가게마다 단골 도매상이 몇 곳인지 세는 것과 같습니다. 도매상이 거의 없는 가게는 작고 한산한 가게일 가능성이 큽니다.",
        win=dict(
            axis=270,
            formula="LN( 1 + 그 CUSIP을 보유한 고유 filer 수 ),  최근 사용 가능한 분기",
            rows=[
                dict(label="공시 대기 60일", segs=[seg(-60, 0, "skip", "60달력일")]),
                dict(
                    label="쓰는 값은 이 앞의 최신 분기말", segs=[seg(-150, -60, "use", "분기말 값")]
                ),
            ],
            marks=[(-150, "t−150"), (-60, "t−60"), (0, "t")],
            note="분기마다 값이 바뀌고 분기 내내 같은 값이 유지됩니다.",
        ),
        principle=dict(
            type="timeline",
            title="13F는 분기가 끝나고 60일 뒤부터 씁니다",
            events=[
                dict(at=0, label="분기말", sub="보유 기준일"),
                dict(at=45, label="법정 마감", sub="45일"),
                dict(at=60, label="모델이 쓰는 날", sub="분기말 + 60달력일", kind="today"),
            ],
            spans=[dict(a=0, b=60, label="지연 60달력일")],
            known={"from": 60, "to": 150, "label": "쓰는 기간"},
            caption="제출 218,126건의 지연은 중앙값 40일, p90 46일이고, 60일을 넘긴 제출은 3.27%입니다.",
        ),
        what="그 종목을 보유한 13F 기관(고유 filer) 수에 1을 더해 로그를 씌운 값입니다.",
        model="Ridge는 8위로 크게 기대고 안정(순위가 오를수록 점수가 오름, 기대 없음)입니다. LightGBM은 16위이지만 불안정해서 해석하지 않습니다.",
        caution="롱으로 쓰는 신호가 아니라 적게 담긴 종목을 피하라는 신호입니다. 규모와 상관이 높고, 사전 예측이 아니라 사후에 방향이 맞아떨어진 것입니다.",
    ),
    "inst_breadth_chg": dict(
        title="기관 수 변화 (분기 대비)",
        basis="f13",
        intuition="분기 사이에 그 종목을 새로 담은 기관이 얼마나 늘었는지입니다. 기관이 늘어나는 종목이 이후에 좋다는 Chen-Hong-Stein의 가설을 13F로 옮긴 것입니다.",
        analogy="지난 분기보다 단골 도매상이 몇 곳 늘었는지를 전체 도매상 수로 나눠 보는 것과 같습니다.",
        win=dict(
            axis=270,
            formula="( n_holders − 직전 분기 n_holders ) / 그 분기 전체 13F-HR filer 수",
            rows=[
                dict(label="최신 분기 보유자 수", segs=[seg(-150, -60, "use", "이 분기말")]),
                dict(
                    label="직전 분기 보유자 수 (뺌)", segs=[seg(-240, -150, "minus", "직전 분기말")]
                ),
            ],
            marks=[(-240, "t−240"), (-150, "t−150"), (-60, "t−60"), (0, "t")],
            note="직전 분기는 같은 CUSIP의 정확히 3개월 앞 분기이고, 건너뛰면 결측입니다. 값은 분기말 + 60달력일부터 씁니다.",
        ),
        principle=dict(
            type="ratio",
            title="분모가 분기마다 바뀝니다",
            unit="",
            rows=[
                dict(
                    label="전체 filer가 4,673일 때",
                    numLabel="증가",
                    num=200,
                    denLabel="filer",
                    den=4673,
                    result="200 ÷ 4,673 = 0.043",
                ),
                dict(
                    label="전체 filer가 8,856일 때",
                    numLabel="증가",
                    num=200,
                    denLabel="filer",
                    den=8856,
                    result="200 ÷ 8,856 = 0.023",
                ),
            ],
            caption="전체 filer 수가 4,673에서 8,856으로 늘어난 시기가 있었습니다. 증가 200은 설명용 가정입니다.",
        ),
        what="분기 사이 보유 기관 수의 증가분을 그 분기 전체 13F filer 수로 나눈 값입니다.",
        model="두 모델 모두 안정이고 순위가 오를수록 점수가 오르는 쪽이라 기대와 같습니다. 크기는 LightGBM 45위, Ridge 40위로 작습니다.",
        caution="4차 X(G2)입니다. 사이즈를 빼면 사라집니다.",
    ),
    "inst_shares_chg": dict(
        title="기관 보유 주식 수 변화 (거래량 정규화)",
        basis="f13",
        intuition="분기 사이에 기관 전체가 늘리거나 줄인 주식 수를 그 종목의 평소 하루 거래량으로 나눈 값입니다. 평소 거래량에 비해 얼마나 많이 사 모았는지를 봅니다.",
        analogy="도매상들이 이번 분기에 더 쌓은 물량이 그 가게의 평소 하루 판매량의 며칠 치인지 세는 것과 같습니다.",
        win=dict(
            axis=270,
            formula="( 보유 주식 수 합 − 직전 분기 합 ) / 분기말 시점 20일 평균 거래량",
            rows=[
                dict(label="최신 분기 보유 주식 수", segs=[seg(-150, -60, "use", "이 분기말")]),
                dict(
                    label="직전 분기 보유 주식 수 (뺌)",
                    segs=[seg(-240, -150, "minus", "직전 분기말")],
                ),
            ],
            marks=[(-240, "t−240"), (-150, "t−150"), (-60, "t−60"), (0, "t")],
            note="값은 분기말 + 60달력일부터 씁니다. 분모는 분기말 시점의 20일 평균 거래량(조정 전 주식 수)입니다.",
        ),
        principle=dict(
            type="flow",
            title="보유 주식 수 변화를 거래량으로 나눕니다",
            steps=[
                st("이번 분기 합", "기관들의 보유 주식 수를 모두 더합니다."),
                st("직전 분기와 비교", "직전 분기 합을 뺍니다."),
                st(
                    "거래량으로 나눔",
                    "분기말 시점 20일 평균 거래량으로 나눕니다.",
                    hl=True,
                    tag="여기서 재는 값",
                ),
            ],
            measure="평소 거래량 대비 기관이 얼마나 늘렸는지 보는 값입니다.",
        ),
        what="분기 사이 기관 보유 주식 수 합의 변화를 평소 거래량으로 나눈 값입니다.",
        model="LightGBM에서 8위로 비교적 크게 기대고 곡선이 U입니다. 두 모델 모두 안정이고 방향은 순위가 오를수록 점수가 내려가는 쪽이라 기대 (+)와 반대입니다(설명 못 함).",
        caution="shares_total이 분기마다 출렁입니다(AAPL 2026-03-31 79.2억에서 06-30 95.8억). 공동 보유 중복 제거가 거친 탓일 수 있고 이 피쳐 값에 잡음이 들어옵니다.",
    ),
}

EXAMPLES = {
    "si_ratio": [dict(kind="p95"), dict(kind="p05")],
    "sv_share_20": [dict(kind="p95"), dict(kind="p05")],
    "inst_n_log": [dict(kind="p95"), dict(kind="p05")],
}
EX_WINDOWS = {"si_ratio": [], "sv_share_20": [], "inst_n_log": []}
_P = " 위 가격 차트는 참고용이고, 이 피쳐는 가격이 아니라 {src}에서 나옵니다."
EX_NOTES = {
    "si_ratio": "{name}: 기준일 순위 {pct}, 원값 {value}." + _P.format(src="FINRA 공매도 잔고 표"),
    "sv_share_20": "{name}: 기준일 순위 {pct}, 원값 {value}."
    + _P.format(src="FINRA 일별 공매도 거래량 파일"),
    "inst_n_log": "{name}: 기준일 순위 {pct}, 원값 {value}." + _P.format(src="SEC 13F 보유 보고"),
}

_L = '<a href="06_us_flow_holders.md"><code>06_us_flow_holders.md</code></a>'
_R = '<a href="10_us_result.md"><code>10_us_result.md</code></a>'
HTML_OVERRIDE = dict(
    LEDE=f"미국 서빙 모델 <code>us_exploratory_20260929_r1</code>(LightGBM·Ridge)이 쓰는 수급·보유 피쳐 17개를 그림으로 풀었습니다. 내부자 거래·공매도·결제 실패·주문 흐름·13F입니다. 정의와 직관의 정본은 {_L}이고, 이 페이지는 그 그림판입니다. 모델 반응 그림(③·④·⑤)은 <b>라벨을 읽지 않고</b> 월초 12개 횡단면(2025-07-01~2026-06-01)으로 만든 <b>진단이며 모델 점수의 반응이지 수익 기여가 아닙니다.</b> 가로축은 값이 아니라 그날 종목들 사이의 순위(백분위)입니다.",
    MAPINTRO="가로는 LightGBM 평균 |SHAP|, 세로는 Ridge 평균 |기여|입니다. 두 모델의 점수 척도가 달라 숫자 크기를 직접 비교하지 않고, <b>선보다 위인지 아래인지</b>만 봅니다(선은 두 모델 합계 비율입니다). 점 모양은 안정한 모델 수입니다. 전체 53개 중 이 문서의 17개만 색을 칠했고 나머지 36개는 회색입니다.",
    MAPCAP=f"읽을 점. 17개 중 LightGBM 상위 10위 안에는 <code>sv_share_20</code>(7위)·<code>inst_shares_chg</code>(8위)·<code>dtc</code>(10위)가, Ridge 상위 10위 안에는 <code>si_ratio</code>(7위)·<code>inst_n_log</code>(8위)·<code>dtc</code>(9위)가 있습니다. <code>fill_ratio_20</code>은 두 모델 모두 53위로 가장 작습니다. 이 크기는 모델 점수의 반응이고 수익 기여가 아닙니다. 판정 기준은 {_R} §1에 있습니다.",
    GROUPSINTRO="날짜별 순위 상관 |ρ| ≥ 0.7인 피쳐끼리 묶었습니다(12개 날짜 평균). 이 문서의 피쳐가 든 묶음만 그렸습니다: 내부자 매수 3개, <code>si_ratio</code>·<code>dtc</code>, <code>oddlot_share_20</code>·<code>fill_ratio_20</code>, 그리고 규모·유동성 묶음의 <code>inst_n_log</code>. <b>묶음 안에서는 서로 몫을 나눠 가집니다.</b> 같은 정보를 담은 피쳐끼리는 한쪽을 빼도 이웃이 대신하고, 선형 모델(Ridge)은 계수가 서로 반대 부호로 상쇄하기도 합니다. 그래서 묶음 합계는 멤버별 크기의 합보다 작을 수 있습니다. 예를 들어 <code>si_ratio</code>(−1.00)와 <code>dtc</code>(+1.00)는 Ridge에서 서로 상쇄해 묶음 합계가 0.00152, 멤버 합이 0.01828입니다.",
)
