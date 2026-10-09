"""테마 사전 시험. 테마마다 합성 이름 예시 하나 이상, 제외 토큰 시험. 점수·결과와 무관하다."""

import polars as pl
import pytest

from modeler.scores.quality import etf_theme as et

EXAMPLES = {
    "반도체": ("KODEX 반도체", "KRX 반도체"),
    "의료·헬스케어": ("TIGER 헬스케어", "FnGuide 헬스케어 지수"),
    "AI": ("RISE AI전력인프라", ""),
    "2차전지": ("KODEX 2차전지산업", "FnGuide 2차전지 지수"),
    "방산": ("PLUS K방산", "FnGuide K-방산 지수"),
    "원자력": ("SOL 원자력SMR", ""),
    "조선": ("HANARO 조선", "FnGuide 조선 지수"),
    "로봇": ("KODEX K-로봇액티브", ""),
    "우주항공": ("KODEX 미국우주항공", ""),
    "전력·전력인프라": ("KODEX AI전력핵심설비", ""),
    "소프트웨어": ("KODEX 미국AI소프트웨어TOP10", ""),
    "인터넷·플랫폼": ("TIGER 인터넷플랫폼", ""),
    "게임": ("KODEX 게임산업", ""),
    "엔터·미디어": ("TIME K컬처액티브", ""),
    "금융": ("KODEX 은행", "KRX 은행"),
    "자동차·전기차": ("KODEX 자동차", "KRX 자동차"),
    "화학": ("TIGER 200 에너지화학", ""),
    "철강·소재": ("KODEX 철강", ""),
    "리츠·부동산": ("TIGER 리츠부동산인프라", ""),
    "인프라": ("PLUS S&P글로벌인프라", "S&P Global Infrastructure Index"),
    "소비재": ("KODEX 필수소비재", ""),
    "화장품·K뷰티": ("TIGER 화장품", ""),
    "친환경·신재생": ("TIGER Fn신재생에너지", ""),
    "수소": ("KODEX 수소", ""),
    "에너지": ("TIGER 원유선물", "S&P GSCI Crude Oil Index"),
    "금·귀금속": ("ACE KRX금현물", ""),
    "클라우드": ("TIGER 글로벌클라우드컴퓨팅", ""),
    "데이터센터": ("RISE 글로벌데이터센터리츠", ""),
    "양자": ("RISE 미국양자컴퓨팅", ""),
    "빅테크": ("TIGER 미국빅테크10", ""),
    "테크·IT": ("TIGER 차이나테크TOP10", "Hang Seng China Tech"),
    "메타버스": ("KODEX 메타버스", ""),
    "소부장": ("KODEX 소부장", ""),
    "배당": ("TIGER 미국배당다우존스", ""),
    "가치·밸류": ("KODEX 가치주", ""),
    "성장": ("ACE 글로벌성장", ""),
    "퀄리티": ("TIGER 퀄리티", ""),
    "모멘텀": ("ACE 모멘텀", ""),
    "저변동": ("KODEX 저변동", ""),
    "동일가중": ("TIGER 동일가중", ""),
    "중소형": ("KODEX 중소형", ""),
    "대형주": ("KODEX 대형주", ""),
    "밸류업": ("KODEX 코리아밸류업", ""),
    "ESG": ("TIGER ESG", ""),
    "그룹주": ("KODEX 삼성그룹", ""),
    "코스피·코스피200": ("KODEX 200", "코스피 200"),
    "코스닥": ("KODEX 코스닥150", ""),
    "KRX300·KRX": ("KODEX KRX300", ""),
    "S&P500": ("TIGER 미국S&P500", ""),
    "나스닥100·나스닥": ("TIGER 미국나스닥100", ""),
    "다우존스": ("TIGER 미국다우존스30", ""),
    "MSCI": ("KODEX MSCI Korea", ""),
    "차이나·항셍": ("TIGER 차이나항셍테크", ""),
    "일본 대표지수": ("ACE 일본Nikkei225", ""),
    "인도": ("KODEX 인도Nifty50", ""),
    "삼성전자": ("KODEX 삼성전자단일종목레버리지", ""),
    "SK하이닉스": ("KODEX 하이닉스단일종목레버리지", ""),
    "테슬라": ("RISE 테슬라고정테크100", ""),
    "엔비디아": ("TIGER 엔비디아미국채커버드콜", ""),
    "팔란티어": ("RISE 팔란티어고정테크100", ""),
    "현대차": ("TIGER 현대차단일종목", ""),
    "채권": ("KODEX 국고채10년", ""),
    "머니마켓·금리": ("KODEX CD금리액티브", ""),
    "TDF·멀티에셋": ("KODEX TDF2050액티브 적격", ""),
    "달러·통화선물": ("KODEX 미국달러선물", ""),
}


def test_every_theme_has_an_example():
    assert set(EXAMPLES) == {t.name for t in et.THEMES}


@pytest.mark.parametrize("theme,ex", list(EXAMPLES.items()))
def test_example_matches_theme(theme, ex):
    assert theme in et.match_themes(*ex)


def test_required_seven_present():
    req = {t.name for t in et.THEMES if t.required}
    assert req == {"반도체", "의료·헬스케어", "AI", "2차전지", "방산", "원자력", "조선"}


def test_ascii_boundary_ai_not_inside_words():
    assert "AI" not in et.match_themes("TIGER AAA채권", "KEDI Aaa Index")
    assert "AI" not in et.match_themes("KODEX 200", "Sustainability Index")
    assert "AI" in et.match_themes("PLUS 미국AI에이전트", "Solactive US AI Agents Index")
    assert "AI" in et.match_themes("X", "Artificial Intelligence Index")


def test_exclude_token_masks_only_that_theme():
    # 전력인프라는 인프라가 아니라 전력 테마다.
    r = et.match_themes("RISE AI전력인프라", "KRX-Akros AI 전력인프라 지수")
    assert "전력·전력인프라" in r and "AI" in r and "인프라" not in r
    # 금융채·특수은행채는 금융 테마가 아니라 채권이다.
    r = et.match_themes("RISE 단기특수은행채액티브", "KAP 단기 특수은행채 지수")
    assert "금융" not in r and "채권" in r
    # 배당성장은 성장 스타일이 아니다. 밸류체인·밸류업은 가치가 아니다.
    assert "성장" not in et.match_themes("TIGER 미국배당성장", "")
    assert "가치·밸류" not in et.match_themes("SOL 우주항공밸류체인", "")
    assert "가치·밸류" not in et.match_themes("KODEX 코리아밸류업", "")
    assert "밸류업" in et.match_themes("KODEX 코리아밸류업", "")
    # 인도네시아는 인도가 아니다. 골드만삭스는 금이 아니다.
    assert "인도" not in et.match_themes("KODEX 인도네시아", "")
    assert "금·귀금속" not in et.match_themes("X", "Goldman Sachs Index")
    # 재생에너지는 에너지(전통)가 아니다.
    assert "에너지" not in et.match_themes("TIGER Fn신재생에너지", "FnGuide 신재생에너지 지수")


def test_spaces_and_case_normalised():
    assert "반도체" in et.match_themes("kodex 미국 반 도 체", "") or "반도체" in et.match_themes(
        "KODEX 미국 반도체", ""
    )
    assert "S&P500" in et.match_themes("TIGER 미국 s&p500", "")
    assert "S&P500" in et.match_themes("X", "S&P 500 Index")
    assert "AI" in et.match_themes("x ai y", "")


def test_multiple_themes_and_no_theme():
    r = et.match_themes("IBK K-AI반도체코어테크", "FnGuide K-AI반도체 코어테크 지수")
    assert {"AI", "반도체", "테크·IT"} <= set(r)
    assert et.match_themes("TIGER 구글밸류체인", "Akros Google Value Chain 지수") == []


def test_dictionary_rules():
    assert {t.category for t in et.THEMES} == {
        et.CAT_THEME,
        et.CAT_STYLE,
        et.CAT_SINGLE,
        et.CAT_ASSET,
    }
    assert len({t.name for t in et.THEMES}) == len(et.THEMES)
    assert all(t.include for t in et.THEMES)
    # 배당·ESG·그룹주·대표지수는 스타일/대표지수 범주다(테마 아님).
    cat = {t.name: t.category for t in et.THEMES}
    for n in ("배당", "가치·밸류", "성장", "그룹주", "ESG", "코스피·코스피200", "S&P500"):
        assert cat[n] == et.CAT_STYLE


def _row(cd, nm, idx, first, last, pens=False, mat=False):
    return (cd, nm, idx, first, last, pens, mat, "domestic", False, "none", "krx반도체")


def _cls(rows):
    cols = {
        "isu_cd": str, "isu_nm": str, "idx_ind_nm": str, "first_trade_date": int,
        "last_trade_date": int, "pension_ineligible_candidate": bool, "exclude_maturity": bool,
        "region": str, "active": bool, "hedge": str, "base_index": str,
    }  # fmt: skip
    return pl.DataFrame(rows, schema=cols, orient="row")


def test_membership_and_counts():
    cls = _cls(
        [
            _row("A", "KODEX 반도체", "KRX 반도체", 20100104, 20261008),
            _row("B", "KODEX 반도체레버리지", "KRX 반도체", 20200102, 20261008, pens=True),
            _row("C", "TIGER 26-12 반도체", "KRX 반도체 만기", 20250102, 20261008, mat=True),
            _row("D", "TIGER 반도체", "KRX 반도체", 20100104, 20180101),
        ]
    )
    mem = et.membership_table(cls)
    assert mem["listed_now"].to_list() == [True, True, True, False]
    # 레버리지·만기형도 테마 표시는 한다. 플래그는 그대로 둔다.
    assert all("반도체" in t for t in mem["themes"].to_list())
    assert mem["pension_ineligible_candidate"].to_list() == [False, True, False, False]
    cnt = et.counts_table(mem, {2015: 20150102, 2020: 20200102, 2025: 20250102}).filter(
        pl.col("theme") == "반도체"
    )
    r = cnt.row(0, named=True)
    assert r["n_listed"] == 2  # 만기형 제외
    assert r["n_pension_eligible"] == 1
    assert r["n_domestic"] == 2
    assert r["n_distinct_base_index"] == 1
    assert r["n_listed_2015"] == 2  # A, D (C는 만기형)
    assert r["n_listed_2020"] == 2  # A, B (D는 2018 폐지)
    assert r["n_listed_2025"] == 2  # A, B (D 폐지, C 만기형)
