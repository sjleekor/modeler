"""ETF 비교 그룹 분류 파서 (사전등록 20261010_quality_score §11.1, 동결 대상 — §12.1).

입력 칸은 ``ISU_CD``·``ISU_NM``·``IDX_IND_NM``·``BAS_DD``·``TDD_CLSPRC`` 다섯뿐이다.
순자산·NAV·수익률·상관·거래대금은 읽지 않고, 상장폐지와 무엇을 잇지도 않는다.
이 파일 하나가 동결 대상이다(토큰 목록·우선순위·별칭표·국내/해외 규칙이 다 여기 있다).

규칙이 사전등록에 명시되지 않은 곳은 ``AMBIGUITIES`` 에 적어 두었다. 지어낸 규칙이
아니라 가장 좁게 읽은 선택이고, 메인이 사전등록에 확정해야 한다.

기준 날짜: 이름·지수명 파싱은 그 ETF의 **마지막 거래일** 행 한 날짜로 고정한다.
거래일 = 종가(``TDD_CLSPRC``)가 있는 행이 하나라도 있는 날. ETF의 마지막 거래일 = 그 ETF의
종가가 있는 마지막 날.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

PARSER_VERSION = "quality-score-v0/etf_classify/1"

# ---------------------------------------------------------------- 모호한 곳
AMBIGUITIES = {
    "A01": "지수 표기에 PR/TR/NTR 표시가 없을 때(대부분의 지수) 그것이 '못 정함'(비교 보류)인지 "
    "'표기 없음'이라는 하나의 값인지. 기본(lenient)은 후자, strict_rt는 전자로 둘 다 센다.",
    "A02": "통화도 같다. 지수명에 통화 표기가 없으면 'unspecified'라는 값으로 둔다(보류하지 않음).",
    "A03": "액티브 판별 토큰은 이름의 '액티브'·'active'(대소문자 무시)뿐이다. KoAct·TIME 같은 "
    "브랜드나 토큰 없는 액티브 상품은 패시브로 읽힌다.",
    "A04": "환헤지는 이름의 괄호 표기 '(H)'·'(합성 H)'·'(… H)'와 '환헤지'. '(UH)'는 비헤지. "
    "표기 없음도 비헤지로 둔다(환노출 상품과 구분하지 않음).",
    "A05": "레버리지 배율: '레버리지'만 있고 배수 표기 없으면 +2로 가정, '인버스'만 있으면 -1, "
    "'곱버스'는 -2, 'N X'가 있으면 그 배수(인버스면 음수).",
    "A06": "연금 부적격 후보: 사전등록 문구 '레버리지·인버스(음의 배율)'를 레버리지(배율>1)와 "
    "인버스(배율<0) 둘 다로 읽었다. 괄호가 '음의 배율만'이면 레버리지는 남아야 한다.",
    "A07": "옵션 전략 토큰 집합(커버드콜·콜옵션·콜매도·BuyWrite / 프리미엄·옵션·버퍼·풋)과 "
    "이름·지수명 둘 다에서 찾는다는 선택은 사전등록에 없다. 그룹 키에는 'covered_call' / "
    "'premium_option' / 'none' 세 값만 쓴다(행사가·주기 차이는 기초지수명이 가른다고 봄).",
    "A08": "비교 그룹 키의 기초지수는 IDX_IND_NM을 (수익률 표기·통화 표기·공백·괄호·대소문자) "
    "정규화한 문자열이다. 별칭표(같은 지수의 다른 표기 합치기)는 비어 있다 — 지어내지 않았다.",
    "A09": "수익률 표기: 'Gross Return'은 TR로, 'Excess Return'은 ER(별도 값)로 읽는다. "
    "한 이름에 서로 다른 표기가 둘 이상이면 'ambiguous'로 두고 strict에서 보류한다.",
    "A10": "만기형은 ISU_NM의 'YY-MM' 토큰으로만 판별한다. 'KIS 2604만기형' 같은 지수명 표기는 "
    "이름 토큰이 없으면 잡지 않는다. 비교 그룹 키에는 만기를 넣지 않았다(사전등록에 없음).",
    "A11": "국내/해외형: 지수명 정규식 규칙(KR_CONTENT·FOREIGN_STRONG 등)과 소수의 명시 목록으로 "
    "읽었다. 사전등록은 '목록'을 말하지만 목록이 없어 규칙+명시 목록으로 만들었다. 국내·해외가 "
    "섞인 지수, TDF·혼합형, 제공자만 보이고 국가가 안 보이는 지수는 unknown이다.",
    "A12": "IDX_IND_NM도 이름처럼 마지막 거래일 행의 값을 쓴다(사전등록은 이름만 말함).",
    "A13": "상장 ETF 수·비교 그룹 분포의 '연도'는 그 해 첫 거래일 기준 종가가 있는 ETF다. "
    "연도 안에 상장했다 폐지된 ETF는 세지 않는다.",
    "A14": "한 ETF가 첫 거래일에는 있었고 이후 이름·지수가 바뀐 경우 분류는 마지막 거래일 값이라 "
    "그 해 시점의 값과 다를 수 있다(사전등록 그대로, §6 표와 달라지는 원인 중 하나).",
}

# ---------------------------------------------------------------- 이름 토큰
_RE_ACTIVE = re.compile(r"액티브|active", re.I)
_RE_HEDGED = re.compile(r"\((?:[^()]*\s)?H\)|환헤지")
_RE_UNHEDGED = re.compile(r"\((?:[^()]*\s)?UH\)")
_RE_INVERSE = re.compile(r"인버스|inverse|곱버스|(?<![A-Za-z])bear(?![A-Za-z])", re.I)
_RE_LEVERAGE = re.compile(r"레버리지|leverage|곱버스|(?<![A-Za-z])bull(?![A-Za-z])", re.I)
_RE_XMULT = re.compile(r"(?<![A-Za-z0-9.])(\d(?:\.\d)?)[Xx](?![A-Za-z])")
_RE_MATURITY = re.compile(r"(?<!\d)(\d{2})-(0[1-9]|1[0-2])(?!\d)")

_RE_COVERED = re.compile(r"커버드\s?콜|covered\s?call|buywrite|콜매도|콜옵션", re.I)
_RE_PREMIUM = re.compile(r"프리미엄|premium|옵션|option|버퍼|buffer|풋", re.I)

# ---------------------------------------------------------------- 지수명 토큰
_L = r"(?<![A-Za-z])"
_R = r"(?![A-Za-z])"
_RT_PATTERNS = (
    ("NTR", re.compile(rf"{_L}NTR{_R}|Net\s*Total\s*Return|Net\s*Return|순총수익", re.I)),
    ("TR", re.compile(rf"{_L}TR{_R}|Total\s*Ret\w*|총수익|Gross\s*Return")),
    ("PR", re.compile(rf"{_L}PR{_R}|Price\s*Return|price\s*index|시장가격", re.I)),
    ("ER", re.compile(rf"{_L}ER{_R}|Excess\s*Return", re.I)),
)
_CUR_PATTERNS = (
    ("KRW", re.compile(rf"{_L}KRW{_R}|원화")),
    ("USD", re.compile(rf"{_L}USD{_R}|달러")),
    ("JPY", re.compile(rf"{_L}JPY{_R}|엔화|Yen", re.I)),
    ("HKD", re.compile(rf"{_L}HKD{_R}")),
    ("CNY", re.compile(rf"{_L}(?:CNH|CNY|RMB){_R}|위안")),
    ("TWD", re.compile(rf"{_L}TWD{_R}")),
    ("EUR", re.compile(rf"{_L}EUR{_R}(?!O)|유로화")),
    ("INR", re.compile(rf"{_L}INR{_R}")),
)
# 이름 안의 통화·수익률 표기는 기초지수 정규화에서 뺀다(키의 다른 칸이 따로 담는다).
_RE_STRIP = re.compile(
    "|".join(
        [
            r"\(?\s*Total\s*Ret\w*\s*\)?",
            r"\(?\s*Price\s*Return\s*\)?",
            r"\(?\s*Excess\s*Return\s*\)?",
            r"\(?\s*Net\s*Total\s*Return\s*\)?",
            r"\(?\s*Gross\s*Return\s*\)?",
            rf"\(?\s*{_L}(?:NTR|TR|PR|ER){_R}\s*\)?",
            r"\(?\s*총수익\s*지수\s*\)?",
            r"\(?\s*총수익\s*\)?",
            r"\(?\s*시장가격\s*지수\s*\)?",
            r"\(?\s*시장가격\s*\)?",
            r"\(?\s*원화\s*환산\s*\)?",
            rf"\(?\s*{_L}(?:USD|KRW|JPY|HKD|CNH|CNY|TWD|INR|EUR){_R}\s*\)?",
        ]
    ),
    re.I,
)
_RE_PUNCT = re.compile(r"[\s()\[\]\-_,.·]+")

# 별칭표: 같은 지수의 다른 표기(정규화한 문자열 -> 대표 문자열).
# 사전등록에 목록이 없어 비워 둔다(A08).
INDEX_ALIASES: dict[str, str] = {}

# ---------------------------------------------------------------- 국내/해외 규칙
_FOREIGN_STRONG = re.compile(
    "|".join(
        [
            "미국",
            rf"{_L}U\.?S\.?{_R}",
            r"S&P\s?500",
            "나스닥",
            r"nasdaq",
            r"NYSE",
            r"PHLX",
            r"Russell",
            r"CRSP",
            r"Dow\s?Jones\s+(?:Industrial|U\.?S|Internet|Brazil|Target|Taiwan)",
            "다우존스",
            "중국",
            "차이나",
            r"China",
            rf"{_L}CSI{_R}",
            r"STAR\s?50",
            r"SZSE",
            r"Hang\s?Seng",
            r"HSTECH",
            "항셍",
            "홍콩",
            "일본",
            r"Japan",
            r"TOPIX",
            r"Nikkei",
            r"Tokyo",
            "엔선물",
            "엔화",
            "인도",
            r"India",
            r"Nifty",
            "베트남",
            r"VN30",
            "대만",
            r"TAIEX",
            r"Taiwan",
            "유럽",
            r"Europe",
            r"STOXX",
            rf"{_L}DAX{_R}",
            r"EURO",
            "독일",
            r"German",
            "글로벌",
            r"Global",
            r"World",
            r"ACWI",
            rf"{_L}EM{_R}",
            r"Emerging",
            r"EAFE",
            r"BRIC",
            r"Brazil",
            r"Latin",
            r"Mexico",
            r"Indonesia",
            r"Philippines",
            r"Russia",
            r"Singapore",
            r"Asia",
            "아시아",
            r"iEdge",
            r"Carbon",
            r"EUA",
            r"GSCI",
            r"LBMA",
            r"Spot",
            r"Copper",
            r"Treasury",
            r"TIPS",
            r"SOFR",
            r"T-Bills?",
            r"T-Bond",
            "달러",
            "국제",
            "서학개미",
            "Apple",
            "애플",
            r"NVIDIA",
            "엔비디아",
            "테슬라",
            r"Tesla",
            r"TSLA",
            "팔란티어",
            r"Palantir",
            "알리바바",
            "아마존",
            r"Amazon",
            r"Google",
            "구글",
            r"TSMC",
            r"BYD",
            r"Xiaomi",
            r"Berkshire",
            r"Eli Lilly",
            r"Samurai",
            r"Wide Moat",
            r"iBoxx",
            r"Select Sector",
            r"Select Industry",
        ]
    ),
    re.I,
)
_KR_CONTENT = re.compile(
    "|".join(
        [
            r"KRX",
            "코스피",
            "코스닥",
            r"KOSPI",
            r"KOSDAQ",
            rf"{_L}KQ{_R}",
            rf"{_L}KS{_R}",
            "한국",
            "코리아",
            r"Korea",
            rf"{_L}K-",
            rf"{_L}K(?=[가-힣])",
            "국고채",
            "통안채",
            "국공채",
            "국채",
            r"KTB",
            r"KOFR",
            rf"{_L}CD{_R}",
            r"MMF",
            "머니마켓",
            r"Money Market",
            "은행채",
            "회사채",
            "특수채",
            "금융채",
            "크레딧",
            "종합채권",
            "전단채",
            r"KOBI",
            r"KRW Cash",
            r"MKF",
            "스타지수",
            r"KTOP",
            "밸류업",
            "삼성",
            "현대차",
            rf"{_L}SK",
            rf"{_L}LG",
            "카카오",
            "포스코",
            "한화",
            "두산",
            "녹색산업",
            "사회책임",
            "동학개미",
            r"MSB",
            "무위험",
            "단기자금",
        ]
    ),
    re.I,
)
# '미국 국채'·'일본 단기 국채'처럼 외국 수식이 붙은 채권어는 국내 표지로 세지 않는다.
_RE_FOREIGN_BOND = re.compile(
    r"(?:미국|일본|U\.?S\.?|US)\s*(?:단기\s*)?(?:국채|국고채|채|머니마켓|달러채권|TREASURY)", re.I
)
_KR_PROVIDER = re.compile(
    rf"{_L}(?:FnGuide|iSelect|WISE|KAP|KIS|KEDI|NICE|DeepSearch|Akros|MK){_R}|FnGuide-", re.I
)
_RE_MIX = re.compile(
    r"혼합|Blend|50/50|5050|TDF|Target.?Date|Lifetime Allocation|Multi.?Asset", re.I
)
_RE_TDF = re.compile(r"TDF|Target.?Date|Lifetime Allocation", re.I)

# 명시 목록(규칙이 못 정하는 것을 사전등록이 말한 대로 직접 정한다). 키는 IDX_IND_NM 원문.
REGION_OVERRIDES: dict[str, str] = {
    "미국달러선물지수": "unknown",  # KRX 산출 통화선물: 국내·해외 어느 쪽에도 안 넣는다(§11.1).
    "엔선물지수": "unknown",
}


def classify_region(idx_nm: str | None) -> tuple[str, str]:
    """(label, rule). label ∈ domestic / foreign / unknown."""
    if not idx_nm:
        return "unknown", "no_index"
    if idx_nm in REGION_OVERRIDES:
        return REGION_OVERRIDES[idx_nm], "override"
    if _RE_TDF.search(idx_nm):
        return "unknown", "tdf"
    fb = _RE_FOREIGN_BOND.search(idx_nm)
    kr_text = _RE_FOREIGN_BOND.sub(" ", idx_nm)
    cf = bool(_FOREIGN_STRONG.search(idx_nm)) or bool(fb)
    cd = bool(_KR_CONTENT.search(kr_text))
    mix = bool(_RE_MIX.search(idx_nm))
    if cf and cd:
        return "unknown", "mixed_content"
    if cf and mix:
        return "unknown", "foreign_blend"
    if cf:
        return "foreign", "foreign_content"
    if cd:
        return "domestic", "kr_content"
    if _KR_PROVIDER.search(idx_nm):
        return "domestic", "kr_provider_only"
    return "unknown", "no_marker"


# ---------------------------------------------------------------- 파서 본체
@dataclass(frozen=True)
class NameFeatures:
    active: bool
    active_token: str
    hedge: str  # "H" / "UH" / "none"
    hedge_token: str
    multiplier: float  # +N 레버리지, -N 인버스, 1.0 일반
    lev_inv_token: str
    option_name: str
    option_idx: str
    option: str  # covered_call / premium_option / none
    maturity: str  # "YY-MM" 또는 ""
    pension_ineligible_candidate: bool


def _first(rx: re.Pattern[str], s: str) -> str:
    m = rx.search(s)
    return m.group(0) if m else ""


def _option_kind(s: str) -> tuple[str, str]:
    c = _first(_RE_COVERED, s)
    if c:
        return "covered_call", c
    p = _first(_RE_PREMIUM, s)
    if p:
        return "premium_option", p
    return "none", ""


def parse_name(isu_nm: str | None, idx_nm: str | None = None) -> NameFeatures:
    n = isu_nm or ""
    ix = idx_nm or ""
    act = _first(_RE_ACTIVE, n)
    hedge_t = _first(_RE_HEDGED, n)
    uh_t = _first(_RE_UNHEDGED, n)
    if uh_t:
        hedge, hedge_tok = "UH", uh_t
    elif hedge_t:
        hedge, hedge_tok = "H", hedge_t
    else:
        hedge, hedge_tok = "none", ""

    inv = _first(_RE_INVERSE, n)
    lev = _first(_RE_LEVERAGE, n)
    xm = _RE_XMULT.search(n)
    mag = float(xm.group(1)) if xm else None
    tokens = [t for t in (inv, lev, xm.group(0) if xm else "") if t]
    if inv:
        mult = -(mag if mag else (2.0 if "곱버스" in n else 1.0))
    elif lev or mag:
        mult = mag if mag else 2.0
    else:
        mult = 1.0
    if mag == 1.0 and not inv:
        mult = 1.0  # '1X'는 배율 아님

    on_kind, on_tok = _option_kind(n)
    oi_kind, oi_tok = _option_kind(ix)
    option = on_kind if on_kind != "none" else oi_kind
    if on_kind == "premium_option" and oi_kind == "covered_call":
        option = "covered_call"
    mm = _RE_MATURITY.search(n)
    return NameFeatures(
        active=bool(act),
        active_token=act,
        hedge=hedge,
        hedge_token=hedge_tok,
        multiplier=mult,
        lev_inv_token="|".join(tokens),
        option_name=on_tok,
        option_idx=oi_tok,
        option=option,
        maturity=f"{mm.group(1)}-{mm.group(2)}" if mm else "",
        pension_ineligible_candidate=(mult != 1.0),
    )


def parse_index(idx_nm: str | None) -> dict:
    s = idx_nm or ""
    rts = [(k, m.group(0)) for k, rx in _RT_PATTERNS if (m := rx.search(s))]
    kinds = {k for k, _ in rts}
    if not kinds:
        rt, rt_tok = "unspecified", ""
    elif len(kinds) == 1:
        rt, rt_tok = rts[0]
    else:
        rt, rt_tok = "ambiguous", "|".join(t for _, t in rts)
    curs = [(k, m.group(0)) for k, rx in _CUR_PATTERNS if (m := rx.search(s))]
    ck = {k for k, _ in curs}
    if not ck:
        cur, cur_tok = "unspecified", ""
    elif len(ck) == 1:
        cur, cur_tok = curs[0]
    else:
        cur, cur_tok = "multiple", "|".join(t for _, t in curs)
    base = _RE_PUNCT.sub("", _RE_STRIP.sub(" ", s)).casefold()
    base = INDEX_ALIASES.get(base, base)
    region, region_rule = classify_region(idx_nm)
    return {
        "return_type": rt,
        "return_token": rt_tok,
        "currency": cur,
        "currency_token": cur_tok,
        "base_index": base,
        "region": region,
        "region_rule": region_rule,
    }


# ---------------------------------------------------------------- 표 수준
INPUT_COLUMNS = ["BAS_DD", "ISU_CD", "ISU_NM", "IDX_IND_NM", "TDD_CLSPRC"]


def read_input(path: str) -> pl.DataFrame:
    """CSV(gz)에서 다섯 칸만 읽는다. 전부 문자열로 읽고 빈 값은 null."""
    return pl.read_csv(path, columns=INPUT_COLUMNS, infer_schema_length=0, null_values=[""])


def has_close(col: str = "TDD_CLSPRC") -> pl.Expr:
    """거래일 정의: 종가 칸에 숫자가 있다."""
    return pl.col(col).is_not_null() & pl.col(col).str.contains(r"^\d")


def trading_days(df: pl.DataFrame) -> list[str]:
    """거래일 = 종가가 있는 행이 하나라도 있는 날(오름차순 BAS_DD)."""
    return df.filter(has_close()).select("BAS_DD").unique().sort("BAS_DD")["BAS_DD"].to_list()


def classification_table(df: pl.DataFrame) -> pl.DataFrame:
    """ETF별 분류 표. 이름·지수는 그 ETF의 마지막 거래일(종가 있는 마지막 날) 행에서 읽는다."""
    traded = df.filter(has_close()).sort(["ISU_CD", "BAS_DD"])
    per = traded.group_by("ISU_CD", maintain_order=True).agg(
        pl.col("BAS_DD").first().alias("first_trade_date"),
        pl.col("BAS_DD").last().alias("last_trade_date"),
        pl.col("ISU_NM").last().alias("isu_nm"),
        pl.col("IDX_IND_NM").last().alias("idx_ind_nm"),
        pl.len().alias("n_trade_days"),
    )
    rows = []
    for r in per.iter_rows(named=True):
        f = parse_name(r["isu_nm"], r["idx_ind_nm"])
        ix = parse_index(r["idx_ind_nm"])
        hold = []
        if not ix["base_index"]:
            hold.append("no_index")
        if ix["return_type"] == "ambiguous":
            hold.append("return_type_ambiguous")
        if ix["currency"] == "multiple":
            hold.append("currency_multiple")
        strict = list(hold)
        if ix["return_type"] == "unspecified":
            strict.append("return_type_unspecified")
        rows.append(
            {
                "isu_cd": r["ISU_CD"],
                "first_trade_date": r["first_trade_date"],
                "last_trade_date": r["last_trade_date"],
                "n_trade_days": r["n_trade_days"],
                "isu_nm": r["isu_nm"],
                "idx_ind_nm": r["idx_ind_nm"],
                "active": f.active,
                "active_token": f.active_token,
                "hedge": f.hedge,
                "hedge_token": f.hedge_token,
                "multiplier": f.multiplier,
                "lev_inv_token": f.lev_inv_token,
                "option": f.option,
                "option_token_name": f.option_name,
                "option_token_index": f.option_idx,
                "maturity_yymm": f.maturity,
                "is_maturity_type": bool(f.maturity),
                "pension_ineligible_candidate": f.pension_ineligible_candidate,
                "return_type": ix["return_type"],
                "return_token": ix["return_token"],
                "currency": ix["currency"],
                "currency_token": ix["currency_token"],
                "base_index": ix["base_index"],
                "region": ix["region"],
                "region_rule": ix["region_rule"],
                "compare_hold": bool(hold),
                "compare_hold_reason": "|".join(hold),
                "compare_hold_strict_rt": bool(strict),
                "compare_hold_strict_rt_reason": "|".join(strict),
                "group_key": "|".join(
                    [
                        ix["base_index"],
                        "A" if f.active else "P",
                        "H" if f.hedge == "H" else "N",
                        f"x{f.multiplier:g}",
                        f.option,
                        ix["currency"],
                        ix["return_type"],
                    ]
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)
