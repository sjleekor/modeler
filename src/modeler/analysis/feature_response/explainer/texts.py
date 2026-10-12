# ruff: noqa: E501
"""Default page texts per market. A config overrides pieces through UI and HTML_OVERRIDE."""

from __future__ import annotations

import html
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def code(t: str) -> str:
    return f"<code>{html.escape(t)}</code>"


def kr_defaults(
    C: SimpleNamespace,
    data: dict[str, Any],
    F: list[str],
    DOC_MD: str,
    RES_MD: str,
    rel: Callable[[Any], str],
) -> dict[str, Any]:
    M = data["meta"]
    D = dict(
        KICKER=f"서빙 피쳐 해설 · {DOC_MD[:2]} · 그림판",
        LEDE=f"서빙 모델 {code('kr_daily_h20_v1')}이 쓰는 가격 피쳐 {code('px_*')} {len(F)}개를 그림으로 풀었습니다. 정의와 직관의 정본은 <a href=\"{DOC_MD}\">{code(DOC_MD)}</a>이고, 이 페이지는 그 그림판입니다. 모델 반응 그림(③·④·⑤)은 이미 열린 holdout(2025-08-01~2026-08-28)으로 만든 <b>진단이지 판정이 아닙니다.</b> 가로축은 값이 아니라 그날 종목들 사이의 순위(백분위)입니다.",
        GUIDE="".join(
            f"<li>{x}</li>"
            for x in [
                "<b>가로축은 순위입니다.</b> 모델은 날짜×시장별 백분위(0~1)만 봅니다. 0%는 그날 가장 낮은 쪽, 100%는 가장 높은 쪽입니다.",
                "<b>곡선은 평균 반응(PDP)입니다.</b> 한 피쳐의 순위만 0~100%로 바꾸고 나머지는 그대로 둘 때 예측 상승 확률의 평균입니다. 값이 비었다는 표시(<code>_isna</code>)는 0으로 고정했습니다.",
                "<b>방향 ρ</b>가 +이면 순위가 오를수록 점수가 오릅니다. <b>섞기 ΔrankIC</b>는 날짜 안에서 이 피쳐를 섞었을 때 rank IC가 얼마나 떨어지는지입니다. 음수면 섞었더니 IC가 오른 것입니다.",
                "<b>진단일 뿐입니다.</b> 합격·불합격, 피쳐 선택에 쓰지 않습니다. 기준선 rank IC 0.09986도 평가가 아니라 진단 기준입니다.",
                "<b>원값 눈금은 근사입니다.</b> 2026-08-28 하루, 코스피·코스닥을 합친 분포로 순위를 값으로 바꿨습니다. 실제 순위는 시장별이라 시장별 값은 표에 있습니다.",
            ]
        ),
        MAPTITLE="④ 한 장 지도 — 모델이 어느 피쳐에 기대나",
        MAPINTRO=f"가로는 평균 |SHAP|(모델 점수를 얼마나 흔드나), 세로는 그 피쳐 하나만 섞었을 때 rank IC가 줄어든 정도(ΔrankIC)입니다. <b>0선 아래는 섞었더니 IC가 오히려 오른 피쳐</b>입니다. 47개 전체 중 {code('px_*')} {len(F)}개만 색을 칠했고, 나머지 {47-len(F)}개는 회색입니다.",
        MAPLEGEND='<span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="5" style="fill:var(--s1)"/></svg>SHAP 안정</span><span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="4" style="fill:var(--surface);stroke:var(--s1)" stroke-width="2"/></svg>SHAP 불안정</span>'
        + f'<span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="3.5" style="fill:var(--muted);opacity:.55"/></svg>px가 아닌 {47-len(F)}개 (참고)</span>',
        MAPCAP=f"읽을 점. {code('px_idio_vol_60d')}가 오른쪽 위에 혼자 멀리 있습니다(섞으면 rank IC 0.0547 하락, 기준 0.0999의 55%). {code('px_high_low_range_20d')}는 SHAP 5위인데 섞으면 오히려 오릅니다(단독 47위). px 18개 중 7개가 0선 아래에 있고 구간이 0 위로 올라오지 않습니다. 모양과 안정 판정의 기준은 <a href=\"{RES_MD}\">{code(RES_MD)}</a> §1에 있습니다.",
        GROUPSTITLE="⑤ 상관 묶음 — 서로 몫을 나눠 가집니다",
        GROUPSINTRO=f"날짜별 순위 상관 |ρ| ≥ 0.7인 피쳐끼리 묶었습니다(SHAP 날짜 53일 평균). <b>묶음 안에서는 서로 몫을 나눠 가집니다.</b> 같은 정보를 담은 피쳐 하나를 섞어도 이웃이 대신 알려 주고, SHAP도 둘로 갈립니다. 그래서 묶음을 한꺼번에 섞은 값은 멤버 단독 값의 합이 되지 않습니다. 예를 들어 c04를 한꺼번에 섞으면 IC 하락이 0.0119인데, 멤버 {code('px_idio_vol_60d')} 단독은 0.0547입니다. 이 차이의 원인은 확인하지 않았습니다.",
        FOOT=(
            f"<li><b>입력 파일</b>: 층 3 산출물 {code('stock_data/kr/output/feature_response_l3/'+M['run_id']+'/')}({code('shap_summary')}·{code('importance_summary')}·{code('pdp')}·{code('ice_sample')}·{code('groups.json')}·{code('manifest.json')}) · 원값 눈금은 {code(rel(C.PANEL))}의 {M['ref_date']} 행(라벨 파일은 읽지 않음) · 예시 종목 가격은 {code(rel(C.RAW)+'/daily_ohlcv')}에서 {M['ref_date']} 이하만 · 종목명은 같은 스냅샷의 {code('stock_master')}.</li>"
            f"<li><b>run_id</b> {code(M['run_id'])} · modeler {code('feature-response-l3')} 커밋 {code(C.MODELER_COMMIT)}(clean) · 서빙 번들 {code('kr_daily_h20_v1')} 1.0.0.</li>"
        ),
        UI=dict(
            ref_date=M["ref_date"],
            ylabel="예측 상승 확률",
            total_label="전체 (262일 중 53일 표본)",
            half1="앞 절반 2025-08~2026-01",
            half2="뒤 절반 2026-02~08",
            ex_scope="시장 내",
            ex_pool="KOSPI",
            ex_note="각 종목은 코스피 안에서 해당 피쳐의 순위 백분위가 목표에 가장 가까운 종목 중 평균 거래대금이 큰 쪽입니다(하루 변동 31% 미만, 261거래일 이상). 가격은 정지일 종가를 앞날 값으로 이은 원 단위 종가이고 기업행동 보정은 없습니다.",
        ),
    )
    return D


def us_defaults(
    C: SimpleNamespace,
    data: dict[str, Any],
    F: list[str],
    DOC_MD: str,
    RES_MD: str,
    rel: Callable[[Any], str],
) -> dict[str, Any]:
    M = data["meta"]
    sx = sum(p["x"] for p in data["scatter"])
    sy = sum(p["y"] for p in data["scatter"])
    nall = len(data["scatter"])
    nf = len(F)
    D = dict(
        KICKER=f"서빙 피쳐 해설 · {DOC_MD[:2]} · 그림판",
        LEDE=f"미국 서빙 모델 {code('us_exploratory_20260929_r1')}(LightGBM·Ridge)이 쓰는 피쳐 {nf}개를 그림으로 풀었습니다. 정의와 직관의 정본은 <a href=\"{DOC_MD}\">{code(DOC_MD)}</a>이고, 이 페이지는 그 그림판입니다. 모델 반응 그림(③·④·⑤)은 <b>라벨을 읽지 않고</b> 월초 12개 횡단면({M['dates'][0]}~{M['dates'][-1]})으로 만든 <b>진단이며 모델 점수의 반응이지 수익 기여가 아닙니다.</b> 가로축은 값이 아니라 그날 종목들 사이의 순위(백분위)입니다.",
        GUIDE="".join(
            f"<li>{x}</li>"
            for x in [
                "<b>가로축은 순위입니다.</b> 두 모델 모두 입력을 그날 횡단면 순위(0~1)로 바꿨고 결측은 0.5입니다. 0%는 그날 가장 낮은 쪽, 100%는 가장 높은 쪽입니다.",
                "<b>LightGBM 곡선은 평균 반응(PDP)입니다.</b> 한 피쳐의 순위만 0~100%로 바꾸고 나머지는 그대로 둘 때 모델 점수의 평균입니다. 값이 비었다는 표시(<code>_isna</code>)는 0으로 고정했습니다.",
                "<b>Ridge는 직선입니다.</b> 선형 모델이라 기울기(rank 계수)가 곧 반응입니다. 점수 척도가 LightGBM과 달라 작은 두 번째 패널에 따로 그렸습니다.",
                "<b>방향 ρ</b>가 +이면 순위가 오를수록 점수가 오릅니다. <b>안정</b>은 두 절반(앞 6개월·뒤 6개월) 모두에서 크기 순위와 방향이 같다는 뜻입니다. 이 실행에는 섞기(permutation)가 없습니다.",
                "<b>모델 점수의 반응이지 수익 기여가 아닙니다.</b> 라벨을 읽지 않았고 점수가 맞았는지도 보지 않았습니다. 판정에 쓰지 않습니다.",
                f"<b>원값 눈금은 근사입니다.</b> {M['ref_date']} 하루, 가격 5달러 이상 종목의 분포로 순위를 값으로 바꿨습니다.",
            ]
        ),
        MAPTITLE="④ 한 장 지도 — 두 모델이 어느 피쳐에 기대나",
        MAPINTRO=f"가로는 LightGBM 평균 |SHAP|, 세로는 Ridge 평균 |기여|입니다. 두 모델의 점수 척도가 달라 숫자 크기를 직접 비교하지 않고, <b>선보다 위인지 아래인지</b>만 봅니다(선은 두 모델 합계 비율입니다). 점 모양은 안정한 모델 수입니다. 전체 {nall}개 중 이 문서의 {nf}개만 색을 칠했고 나머지 {nall-nf}개는 회색입니다.",
        MAPLEGEND='<span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="5" style="fill:var(--s1)"/></svg>두 모델 모두 안정</span><span><svg viewBox="0 0 26 12"><path d="M13 0L19 6L13 12L7 6Z" style="fill:var(--s1)"/></svg>한 모델만 안정</span><span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="4" style="fill:var(--surface);stroke:var(--s1)" stroke-width="2"/></svg>둘 다 불안정</span>'
        + f'<span><svg viewBox="0 0 26 10"><circle cx="13" cy="5" r="3.5" style="fill:var(--muted);opacity:.55"/></svg>이 문서 밖 {nall-nf}개 (참고)</span>',
        MAPCAP=f'{nall}개 전체의 평균 크기 합은 LightGBM {sx:.4f}, Ridge {sy:.4f}로 척도가 다릅니다. 모양과 안정 판정의 기준은 <a href="{RES_MD}">{code(RES_MD)}</a> §1에 있습니다.',
        GROUPSTITLE="⑤ 상관 묶음 — 서로 몫을 나눠 가집니다",
        GROUPSINTRO="날짜별 순위 상관 |ρ| ≥ 0.7인 피쳐끼리 묶었습니다(12개 날짜 평균). <b>묶음 안에서는 서로 몫을 나눠 가집니다.</b> 같은 정보를 담은 피쳐끼리는 한쪽을 빼도 이웃이 대신하고, 선형 모델(Ridge)은 계수가 서로 반대 부호로 상쇄하기도 합니다. 그래서 묶음 합계(부호까지 합한 뒤 절댓값)는 멤버별 크기의 합보다 작을 수 있습니다.",
        FOOT=(
            f"<li><b>입력 파일</b>: 층 3 산출물 {code(rel(C.RUN_DIR)+'/')}({code('shap_summary')}·{code('pdp')}·{code('ice_sample')}·{code('ridge_coef')}·{code('groups.json')}·{code('manifest.json')}, 라벨 안 읽음) · 원값 눈금은 {code(rel(C.PANEL_V2))}·{code(rel(C.PANEL_FLOW))}의 {M['ref_date']} 행(가격 5달러 이상, 라벨 파일은 읽지 않음) · 예시 종목 가격은 {code(rel(C.PRICES))}에서 {M['ref_date']} 이하만 · 종목명은 {code(rel(C.COMPANY_META))}.</li>"
            f"<li><b>run_id</b> {code(M['run_id'])} · modeler 커밋 {code(M['commit'])}(실행 당시 작업 트리 dirty, 결과에는 영향 없음 — 결과 문서에서 확인).</li>"
        ),
        UI=dict(
            ref_date=M["ref_date"],
            ylabel="모델 점수",
            total_label="전체 (월초 12개 횡단면)",
            half1="앞 절반 2025-07~2025-12",
            half2="뒤 절반 2026-01~2026-06",
            ex_scope="가격 5달러 이상 종목 내",
            ex_pool="가격 5달러 이상",
            singles_prefix="묶이지 않은 단독 피쳐",
            disclaimer="모델 점수의 반응이지 수익 기여가 아닙니다. 라벨을 읽지 않았습니다.",
            ex_note="각 종목은 기준일 가격 5달러 이상 종목 중 해당 피쳐의 순위 백분위가 목표에 가장 가까운 종목 가운데 20일 평균 거래대금이 큰 쪽입니다(하루 변동 40% 미만, 261거래일 이상). 가격은 일봉 종가이고 분할·배당 보정 여부는 원천 그대로입니다.",
        ),
    )
    return D
