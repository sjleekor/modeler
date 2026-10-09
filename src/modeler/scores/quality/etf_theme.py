"""ETF 테마 사전과 매칭 (사전등록 20261010_quality_score §11 ETF 부분의 "테마 보기(기록용)").

판단이 아니라 **빠짐없이 덮는 것**이 목적이다. 어느 테마가 유망한지 가리지 않는다.
입력은 ETF의 마지막 거래일 ``isu_nm``·``idx_ind_nm`` 문자열뿐이다. 가격·순자산·수익률·
거래대금은 읽지 않고, 상장폐지와 무엇을 잇지도 않는다. 이 파일 하나가 동결 대상이다
(사전은 아래 ``THEMES`` 상수 표다).

매칭 규칙
- 대상 문자열 = ``isu_nm`` + 공백 + ``idx_ind_nm``, 소문자로 바꾼다.
- 한글이 없는 토큰(영문·숫자)은 공백을 하나로 줄인 문자열에서 찾고, 앞뒤가 영문자가 아니어야
  한다(``ai`` 가 ``aaa`` 나 ``kedi`` 에 걸리지 않게. 숫자는 허용해 ``tech100`` 이 걸린다).
  끝에 ``*`` 를 붙이면 오른쪽 경계를 풀어 ``robot*`` 이 ``robotics`` 에 걸린다.
- 한글이 있는 토큰은 공백을 모두 뺀 문자열에서 부분 문자열로 찾는다(이름이 ``2차전지TOP10`` 처럼
  붙어 있다).
- 제외 토큰은 먼저 그 자리를 공백으로 가린 뒤 포함 토큰을 찾는다(마스킹). 그래서 ``전력인프라`` 를
  가리면 ``인프라`` 테마에 안 걸리고 ``전력`` 테마에는 걸린다.
- 한 ETF가 여러 테마에 들 수 있다. 레버리지·인버스·만기형도 테마 표시는 한다(플래그는 그대로).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

DICT_VERSION = "quality-score-v0/etf_theme/1"

CAT_THEME = "테마"
CAT_STYLE = "스타일/대표지수"
CAT_SINGLE = "단일종목"
CAT_ASSET = "자산군"

AMBIGUITIES = {
    "T01": "사전에 넣는 기준은 '지금 상장(만기형 제외) ETF 3개 이상의 이름·지수명에 나옴'이다. "
    "사용자 일곱 테마는 3개 미만이어도 넣는다.",
    "T02": "'테마'와 '스타일/대표지수'의 경계는 사용자 지시(배당·가치·성장·그룹주·ESG·대표지수는 "
    "스타일/대표지수)를 따랐다. 밸류업·모멘텀·저변동·동일가중·중소형도 스타일로 둔다.",
    "T03": "채권·머니마켓·달러선물·금리 같은 자산군 표시와 삼성전자·하이닉스 같은 단일종목은 "
    "지시한 범주가 아니라 별도 범주('자산군'·'단일종목')로 두었다. 빈도로 정한 것이다.",
    "T04": "금·은·원유는 원자재 테마로 보고 '테마' 범주에 넣었다(자산군이 아님). 판단이 갈린다.",
    "T05": "'테크'(IT·정보기술 포함)는 범위가 넓어 AI·반도체·소프트웨어와 겹친다. 겹침을 허용하고 "
    "따로 하나의 테마로 둔다. 빅테크(미국 대형 기술주)는 별도 테마다.",
    "T06": "이름만 보므로 이름에 테마가 안 드러난 ETF(예: 지수명으로만 알 수 있는 것 외)는 놓친다. "
    "지수명에 있으면 잡는다.",
    "T07": "연도별 상장 수는 현재(마지막 거래일) 이름 기준이다. 그해 첫 거래일에 "
    "first_trade_date<=그날<=last_trade_date 인 만기형 아닌 ETF를 센다. "
    "이름이 나중에 바뀐 ETF는 그 해의 이름과 다를 수 있다(etf_classify A14와 같은 문제).",
    "T08": "후보였지만 지금 상장 3개 미만이라 뺀 것(2026-10-10 실측): 건설 2, 음식료 1, "
    "사이버보안 1, 러셀 1. 맞는 ETF가 새로 생겨도 동결 후에는 사전을 못 고친다.",
}


@dataclass(frozen=True)
class Theme:
    name: str
    category: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    required: bool = False  # 사용자가 반드시 넣으라고 한 테마


T = Theme
THEMES: tuple[Theme, ...] = (
    # ---- 사용자 일곱 테마
    T(
        "반도체",
        CAT_THEME,
        ("반도체", "semiconductor*", "semicondoctor*", "semi", "soxx", "메모리", "hbm"),
        (),
        True,
    ),
    T(
        "의료·헬스케어",
        CAT_THEME,
        (
            "헬스케어",
            "healthcare",
            "health care",
            "바이오",
            "bio*",
            "의료",
            "제약",
            "medical",
            "pharma*",
            "신약",
            "의약",
            "헬스",
            "생명과학",
            "병원",
            "메디컬",
            "치료제",
            "비만",
        ),
        ("바이오매스",),
        True,
    ),
    T("AI", CAT_THEME, ("ai", "인공지능", "artificial intelligence", "피지컬ai"), (), True),
    T(
        "2차전지",
        CAT_THEME,
        (
            "2차전지",
            "이차전지",
            "차전지",
            "배터리",
            "battery",
            "batteries",
            "리튬",
            "lithium",
            "양극재",
            "음극재",
            "전고체",
        ),
        (),
        True,
    ),
    T(
        "방산",
        CAT_THEME,
        ("방산", "방위", "국방", "defense", "defence", "aerospace & defense", "무기"),
        (),
        True,
    ),
    T(
        "원자력",
        CAT_THEME,
        ("원자력", "원전", "smr", "우라늄", "uranium", "nuclear", "핵융합"),
        (),
        True,
    ),
    T("조선", CAT_THEME, ("조선", "shipbuilding", "해운조선"), ("조선일보",), True),
    # ---- 그 밖의 테마 (지금 상장 3개 이상)
    T("로봇", CAT_THEME, ("로봇", "robot*", "휴머노이드", "humanoid", "피지컬"), ()),
    T(
        "우주항공",
        CAT_THEME,
        ("우주", "항공", "space", "aerospace", "위성", "drone", "드론"),
        ("항공사", "항공운송"),
    ),
    T(
        "전력·전력인프라",
        CAT_THEME,
        (
            "전력",
            "electric power",
            "power grid",
            "전선",
            "변압기",
            "grid",
            "전력기기",
            "유틸리티",
            "utilities",
        ),
        (),
    ),
    T("소프트웨어", CAT_THEME, ("소프트웨어", "software", "saas", "sw"), ()),
    T(
        "인터넷·플랫폼",
        CAT_THEME,
        (
            "인터넷",
            "internet",
            "플랫폼",
            "platform",
            "이커머스",
            "e-commerce",
            "ecommerce",
            "커머스",
        ),
        (),
    ),
    T("게임", CAT_THEME, ("게임", "game*", "gaming", "e스포츠"), ()),
    T(
        "엔터·미디어",
        CAT_THEME,
        (
            "엔터",
            "미디어",
            "media",
            "콘텐츠",
            "contents",
            "k-pop",
            "kpop",
            "한류",
            "엔터테인먼트",
            "entertainment",
            "컬처",
            "웹툰",
            "드라마",
        ),
        (),
    ),
    T(
        "금융",
        CAT_THEME,
        ("금융", "은행", "증권", "보험", "financ*", "bank*", "insurance", "핀테크", "fintech"),
        ("금융채", "은행채", "증권사채", "통안채", "금융투자"),
    ),
    T(
        "자동차·전기차",
        CAT_THEME,
        (
            "자동차",
            "전기차",
            "electric vehicle",
            "ev",
            "vehicle*",
            "모빌리티",
            "mobility",
            "자율주행",
            "autonomous",
            "로보택시",
            "스마트카",
        ),
        (),
    ),
    T("화학", CAT_THEME, ("화학", "chemical*"), ()),
    T(
        "철강·소재",
        CAT_THEME,
        ("철강", "steel", "소재", "materials", "비철금속", "구리", "copper"),
        (),
    ),
    T("리츠·부동산", CAT_THEME, ("리츠", "reit*", "부동산", "real estate"), ()),
    T(
        "인프라",
        CAT_THEME,
        ("인프라", "infrastructure"),
        (
            "전력인프라",
            "ai인프라",
            "리츠부동산인프라",
            "리츠부동산",
            "데이터센터인프라",
            "반도체인프라",
        ),
    ),
    T("소비재", CAT_THEME, ("소비재", "소비", "consumer", "필수소비", "경기소비"), ()),
    T("화장품·K뷰티", CAT_THEME, ("화장품", "뷰티", "beauty", "cosmetic*"), ()),
    T(
        "친환경·신재생",
        CAT_THEME,
        (
            "친환경",
            "그린뉴딜",
            "기후변화",
            "탄소",
            "clean",
            "신재생",
            "재생에너지",
            "태양광",
            "풍력",
            "solar",
            "wind",
            "renewable*",
            "climate",
            "esg환경",
            "green",
        ),
        (),
    ),
    T("수소", CAT_THEME, ("수소", "hydrogen", "연료전지"), ()),
    T(
        "에너지",
        CAT_THEME,
        ("에너지", "energy", "정유", "oil", "원유", "crude", "천연가스", "natural gas", "wti"),
        ("신재생에너지", "재생에너지", "에너지저장"),
    ),
    T(
        "금·귀금속",
        CAT_THEME,
        ("금현물", "골드", "gold", "금선물", "silver", "은선물", "은현물", "귀금속", "precious"),
        ("골드만", "goldman"),
    ),
    T("클라우드", CAT_THEME, ("클라우드", "cloud"), ()),
    T("데이터센터", CAT_THEME, ("데이터센터", "data center", "datacenter", "광통신"), ()),
    T("양자", CAT_THEME, ("양자", "quantum"), ()),
    T("빅테크", CAT_THEME, ("빅테크", "big tech", "매그니피센트", "magnificent", "fang", "m7"), ()),
    T(
        "테크·IT",
        CAT_THEME,
        ("테크", "tech", "technology", "it", "정보기술", "information technology"),
        (),
    ),
    T("메타버스", CAT_THEME, ("메타버스", "metaverse"), ()),
    T("소부장", CAT_THEME, ("소부장", "소재부품장비"), ()),
    # ---- 스타일/대표지수
    T("배당", CAT_STYLE, ("배당", "dividend*", "분배금"), ("고정배당",)),
    T(
        "가치·밸류",
        CAT_STYLE,
        ("가치주", "value", "밸류에이션"),
        ("밸류체인", "밸류업", "value chain", "valuechain"),
    ),
    T("성장", CAT_STYLE, ("성장", "growth"), ("배당성장",)),
    T("퀄리티", CAT_STYLE, ("퀄리티", "quality"), ()),
    T("모멘텀", CAT_STYLE, ("모멘텀", "momentum"), ()),
    T("저변동", CAT_STYLE, ("저변동", "low vol*", "minimum volatility", "min vol*"), ()),
    T("동일가중", CAT_STYLE, ("동일가중", "equal weight*", "equalweight"), ()),
    T(
        "중소형",
        CAT_STYLE,
        ("중소형", "소형주", "중형주", "small cap", "smallcap", "mid cap", "midcap"),
        (),
    ),
    T("대형주", CAT_STYLE, ("대형주", "large cap", "largecap", "우량주"), ()),
    T("밸류업", CAT_STYLE, ("밸류업", "value-up", "valueup"), ()),
    T("ESG", CAT_STYLE, ("esg", "sri", "지속가능", "sustainab*"), ()),
    T(
        "그룹주",
        CAT_STYLE,
        (
            "삼성그룹",
            "현대차그룹",
            "sk그룹",
            "lg그룹",
            "한화그룹",
            "포스코그룹",
            "롯데그룹",
            "그룹주",
            "10대그룹",
            "그룹",
        ),
        (),
    ),
    T("코스피·코스피200", CAT_STYLE, ("코스피", "kospi"), ()),
    T("코스닥", CAT_STYLE, ("코스닥", "kosdaq"), ()),
    T("KRX300·KRX", CAT_STYLE, ("krx300", "krx 300"), ()),
    T("S&P500", CAT_STYLE, ("s&p500", "s&p 500", "snp500", "sp500", "s&p"), ()),
    T("나스닥100·나스닥", CAT_STYLE, ("나스닥", "nasdaq*"), ()),
    T("다우존스", CAT_STYLE, ("다우존스", "dow jones"), ()),
    T("MSCI", CAT_STYLE, ("msci",), ()),
    T("차이나·항셍", CAT_STYLE, ("항셍", "hang seng", "csi", "차이나", "china", "중국"), ()),
    T("일본 대표지수", CAT_STYLE, ("topix", "nikkei", "닛케이", "일본"), ()),
    T("인도", CAT_STYLE, ("인도", "india*", "nifty"), ("인도네시아", "indonesia")),
    # ---- 단일종목
    T("삼성전자", CAT_SINGLE, ("삼성전자",), ("삼성전자우",)),
    T("SK하이닉스", CAT_SINGLE, ("하이닉스", "hynix"), ()),
    T("테슬라", CAT_SINGLE, ("테슬라", "tesla"), ()),
    T("엔비디아", CAT_SINGLE, ("엔비디아", "nvidia"), ()),
    T("팔란티어", CAT_SINGLE, ("팔란티어", "palantir"), ()),
    T("현대차", CAT_SINGLE, ("현대차",), ("현대차그룹",)),
    # ---- 자산군
    T(
        "채권",
        CAT_ASSET,
        (
            "채권",
            "국채",
            "국고채",
            "회사채",
            "bond*",
            "treasury",
            "단기채",
            "통안채",
            "공채",
            "중기채",
            "장기채",
            "국공채",
            "은행채",
            "금융채",
            "특수채",
            "하이일드",
            "high yield",
            "tips",
            "물가채",
            "전단채",
        ),
        (),
    ),
    T(
        "머니마켓·금리",
        CAT_ASSET,
        ("머니마켓", "mmf", "cd금리", "금리", "kofr", "sofr", "파킹", "money market"),
        (),
    ),
    T(
        "TDF·멀티에셋",
        CAT_ASSET,
        ("tdf*", "멀티에셋", "multi-asset", "multi asset", "자산배분", "target date"),
        (),
    ),
    T(
        "달러·통화선물",
        CAT_ASSET,
        ("달러선물", "달러", "엔선물", "엔화", "유로", "usd futures", "환율"),
        (),
    ),
)


# ---------------------------------------------------------------- 매칭
_HANGUL = re.compile(r"[가-힣]")
_WS = re.compile(r"\s+")


def _is_ascii_token(tok: str) -> bool:
    return not _HANGUL.search(tok)


def _token_regex(tok: str) -> re.Pattern[str]:
    open_right = tok.endswith("*")
    t = tok.rstrip("*").lower()
    if _is_ascii_token(t):
        t = re.escape(_WS.sub(" ", t.strip())).replace(r"\ ", " ")
        right = r"" if open_right else r"(?![a-z])"
        return re.compile(rf"(?<![a-z]){t}{right}")
    return re.compile(re.escape(_WS.sub("", t)))


def _normalize(name: str | None, index: str | None) -> tuple[str, str]:
    """(공백 하나로 줄인 문자열, 공백 뺀 문자열). 둘 다 소문자."""
    s = _WS.sub(" ", f"{name or ''} {index or ''}".lower()).strip()
    return s, s.replace(" ", "")


def _mask(spaced: str, compact: str, tokens: tuple[str, ...]) -> tuple[str, str]:
    for tok in tokens:
        rx = _token_regex(tok)
        if _is_ascii_token(tok.rstrip("*")):
            spaced = rx.sub(lambda m: " " * len(m.group(0)), spaced)
        else:
            compact = rx.sub(lambda m: " " * len(m.group(0)), compact)
    return spaced, compact


def _hits(spaced: str, compact: str, tokens: tuple[str, ...]) -> list[str]:
    out = []
    for tok in tokens:
        rx = _token_regex(tok)
        hay = spaced if _is_ascii_token(tok.rstrip("*")) else compact
        if rx.search(hay):
            out.append(tok)
    return out


def match_themes(name: str | None, index: str | None) -> list[str]:
    """이름·지수명이 걸리는 테마 이름 목록(사전 순서)."""
    spaced, compact = _normalize(name, index)
    out = []
    for th in THEMES:
        sp, cp = _mask(spaced, compact, th.exclude)
        if _hits(sp, cp, th.include):
            out.append(th.name)
    return out


def matched_tokens(name: str | None, index: str | None, theme: str) -> list[str]:
    th = next(t for t in THEMES if t.name == theme)
    spaced, compact = _normalize(name, index)
    sp, cp = _mask(spaced, compact, th.exclude)
    return _hits(sp, cp, th.include)


# ---------------------------------------------------------------- 산출물
LISTED_LAST_DATE = 20261008  # 마지막 거래일(입력 자료의 끝). 지금 상장 = last_trade_date 가 이 값.
SNAPSHOT_YEARS = (2015, 2020, 2025)

CLASSIFICATION_COLUMNS = (
    "isu_cd", "isu_nm", "idx_ind_nm", "first_trade_date", "last_trade_date",
    "pension_ineligible_candidate", "exclude_maturity", "region", "active", "hedge", "base_index",
)  # fmt: skip


def dictionary_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "theme": [t.name for t in THEMES],
            "category": [t.category for t in THEMES],
            "required_by_user": [t.required for t in THEMES],
            "include_tokens": ["|".join(t.include) for t in THEMES],
            "exclude_tokens": ["|".join(t.exclude) for t in THEMES],
        }
    )


def membership_table(cls: pl.DataFrame, listed_last_date: int = LISTED_LAST_DATE) -> pl.DataFrame:
    """ETF별 테마 목록. 이름·지수명과 구조 플래그만 쓴다."""
    cat = {t.name: t.category for t in THEMES}
    rows = []
    for r in cls.select(CLASSIFICATION_COLUMNS).iter_rows(named=True):
        ths = match_themes(r["isu_nm"], r["idx_ind_nm"])
        rows.append(
            {
                "isu_cd": r["isu_cd"],
                "isu_nm": r["isu_nm"],
                "idx_ind_nm": r["idx_ind_nm"],
                "first_trade_date": r["first_trade_date"],
                "last_trade_date": r["last_trade_date"],
                "listed_now": r["last_trade_date"] == listed_last_date,
                "exclude_maturity": r["exclude_maturity"],
                "pension_ineligible_candidate": r["pension_ineligible_candidate"],
                "region": r["region"],
                "active": r["active"],
                "hedge": r["hedge"],
                "base_index": r["base_index"],
                "n_themes": sum(cat[t] == CAT_THEME for t in ths),
                "themes": "|".join(ths),
                "theme_only": "|".join(t for t in ths if cat[t] == CAT_THEME),
                "style_only": "|".join(t for t in ths if cat[t] == CAT_STYLE),
                "single_stock": "|".join(t for t in ths if cat[t] == CAT_SINGLE),
                "asset_class": "|".join(t for t in ths if cat[t] == CAT_ASSET),
            }
        )
    return pl.DataFrame(rows)


def counts_table(
    mem: pl.DataFrame, first_days: dict[int, int], years: tuple[int, ...] = SNAPSHOT_YEARS
) -> pl.DataFrame:
    """테마별 개수. 지금 상장 = listed_now 이고 만기형 아님."""
    eligible_pool = mem.filter(pl.col("listed_now") & ~pl.col("exclude_maturity"))
    out = []
    for th in THEMES:
        has = pl.col("themes").str.split("|").list.contains(th.name)
        now = eligible_pool.filter(has)
        row = {
            "theme": th.name,
            "category": th.category,
            "n_listed": now.height,
            "n_pension_eligible": now.filter(~pl.col("pension_ineligible_candidate")).height,
            "n_domestic": now.filter(pl.col("region") == "domestic").height,
            "n_foreign": now.filter(pl.col("region") == "foreign").height,
            "n_region_unknown": now.filter(pl.col("region") == "unknown").height,
            "n_distinct_base_index": now["base_index"].n_unique(),
        }
        past = mem.filter(~pl.col("exclude_maturity") & has)
        for y in years:
            d = first_days[y]
            row[f"n_listed_{y}"] = past.filter(
                (pl.col("first_trade_date") <= d) & (pl.col("last_trade_date") >= d)
            ).height
        out.append(row)
    return pl.DataFrame(out)


# ---------------------------------------------------------------- 실행
DEFAULT_OUT_REL = "kr/output/quality_score_prep_20261010"
OUTPUT_FILES = ("etf_theme_dictionary.csv", "etf_theme_membership.csv", "etf_theme_counts.csv")


def _sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.check_output(["git", "-C", str(here), *args], text=True).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    """STOCK_DATA_ROOT=../stock_data python -m modeler.scores.quality.etf_theme"""
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--out-dir", default=str(root / DEFAULT_OUT_REL))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)

    # 읽는 파일은 둘이다. 분류표(이름·지수명·구조 플래그·거래일)와 연도별 첫 거래일 표(날짜만).
    cls = pl.read_csv(out / "etf_classification.csv")
    first_days = {
        int(r["year"]): int(r["first_trading_day"])
        for r in pl.read_csv(out / "etf_group_size_by_year.csv")
        .select("year", "first_trading_day")
        .iter_rows(named=True)
    }
    mem = membership_table(cls)
    cnt = counts_table(mem, first_days)
    dictionary_table().write_csv(out / "etf_theme_dictionary.csv")
    mem.write_csv(out / "etf_theme_membership.csv")
    cnt.write_csv(out / "etf_theme_counts.csv")

    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    src = Path(__file__)
    manifest["theme"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "ETF 테마 사전(사전등록 §11 '테마 보기(기록용)'). 이름·지수명 문자열만 사용. "
        "가격·순자산·수익률을 읽지 않고 결과와 잇지 않는다.",
        "dictionary_version": DICT_VERSION,
        "file": str(src.resolve()),
        "sha256": _sha256(src),
        "inputs": {
            "etf_classification.csv": _sha256(out / "etf_classification.csv"),
            "etf_group_size_by_year.csv": _sha256(out / "etf_group_size_by_year.csv"),
        },
        "code": {
            "repo": "modeler",
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "worktree_dirty": bool(_git("status", "--porcelain", "--", ".")),
        },
        "outputs": {f: _sha256(out / f) for f in OUTPUT_FILES},
        "n_themes": len(THEMES),
        "listed_last_date": LISTED_LAST_DATE,
        "ambiguities": AMBIGUITIES,
    }
    manifest["outputs"] = sorted({*manifest.get("outputs", []), *OUTPUT_FILES})
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest["theme"]["code"] | {"sha256": manifest["theme"]["sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
