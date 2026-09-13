"""Build the offline feature-performance validation report.

The report consumes one explicitly selected Phase AB run. Phase A/B lineage is
derived from the AB manifest so incompatible runs cannot be combined. All charts
and data are embedded into a single HTML file; no database or network access is
performed.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPORT_SCHEMA_VERSION = "feature_performance_report_v1"
GRADE_ORDER = ("A", "B", "C", "D", "R", "NE")
GRADE_COLORS = {
    "A": "#0f766e",
    "B": "#2563eb",
    "C": "#d97706",
    "D": "#dc2626",
    "R": "#64748b",
    "NE": "#94a3b8",
}
PHASE_COLORS = {"A": "#2563eb", "B": "#7c3aed", "exploratory": "#d97706"}
FEATURE_GROUPS = (
    ("price", "가격·유동성"),
    ("flow", "투자자 수급·공매도"),
    ("event", "공시·자본정책 이벤트"),
    ("financial", "재무·밸류"),
    ("human_capital", "인적자본"),
    ("ownership", "지분·주주 활동"),
)
FEATURE_GUIDES: dict[str, dict[str, str]] = {
    "px_reversal_5d": {
        "group": "price",
        "title": "5일 단기 반전",
        "definition": "최근 5거래일 로그수익률의 부호를 바꾼 값입니다.",
        "reading": "값이 클수록 최근 낙폭이 컸습니다. 이후 반등할수록 양의 IC가 납니다.",
        "caveat": "분할·증자 같은 기업행동이 섞인 구간은 마스킹합니다.",
    },
    "px_mom_12_1": {
        "group": "price",
        "title": "12개월-1개월 모멘텀",
        "definition": "약 12개월 전부터 1개월 전까지의 로그수익률입니다. 최근 1개월은 뺍니다.",
        "reading": "값이 클수록 중장기 상승세가 강했습니다.",
        "caveat": "최근 반전 효과와 섞이지 않도록 마지막 21거래일을 건너뜁니다.",
    },
    "px_resid_mom_12_1": {
        "group": "price",
        "title": "시장요인 제거 모멘텀",
        "definition": "시장모형으로 설명되는 수익률을 뺀 뒤, 12개월-1개월 잔차수익률을 더한 값입니다.",
        "reading": "값이 클수록 시장 전체 움직임으로 설명되지 않는 종목 고유의 상승세가 강했습니다.",
        "caveat": "충분한 시장모형 이력이 있어야 계산되므로 일반 모멘텀보다 표본이 적습니다.",
    },
    "px_near_52w_high": {
        "group": "price",
        "title": "52주 고점 근접도",
        "definition": "현재 종가를 최근 252거래일 최고 종가로 나눈 뒤 1을 뺀 값입니다.",
        "reading": "0에 가까울수록 52주 고점에 가깝고, 더 작을수록 고점에서 멀리 떨어져 있습니다.",
        "caveat": "기존 모델의 `px_dist_52w_high`와 같은 경제 개념입니다.",
    },
    "px_maxret_20d": {
        "group": "price",
        "title": "20일 최대 일간수익률",
        "definition": "최근 20거래일 중 가장 큰 하루 수익률입니다.",
        "reading": "값이 클수록 최근 한 번의 급등이 컸습니다. 고점 추종보다 투기적 급등 뒤 약세를 보는 피처입니다.",
        "caveat": "기업행동이 섞인 20일 구간은 제외합니다.",
    },
    "px_idio_vol_60d": {
        "group": "price",
        "title": "60일 고유변동성",
        "definition": "시장모형 잔차수익률의 최근 60거래일 표준편차입니다.",
        "reading": "값이 클수록 시장 등락을 빼고도 종목 자체의 흔들림이 큽니다.",
        "caveat": "시장모형 추정에 최소 126개 유효 관측치가 필요합니다.",
    },
    "px_amihud_20d": {
        "group": "price",
        "title": "20일 Amihud 비유동성",
        "definition": "최근 20일의 `|수익률| / 거래대금` 평균입니다.",
        "reading": "값이 클수록 적은 거래대금에도 가격이 크게 움직여 유동성이 낮습니다.",
        "caveat": "시가총액과 강하게 겹칠 수 있어 크기 효과와 함께 봐야 합니다.",
    },
    "px_turnover_shock": {
        "group": "price",
        "title": "거래대금 충격",
        "definition": "오늘 거래대금을 직전 60거래일 거래대금 중앙값으로 나눈 로그값입니다.",
        "reading": "값이 클수록 평소보다 거래가 갑자기 몰렸습니다.",
        "caveat": "기준 중앙값에는 오늘 거래대금을 넣지 않습니다.",
    },
    "px_zero_ret_ratio_20d": {
        "group": "price",
        "title": "20일 무수익·무거래 비율",
        "definition": "최근 20일 중 수익률이 0이거나 거래량이 0인 날의 비율입니다.",
        "reading": "값이 클수록 가격이 자주 멈춰 있고 거래가 뜸합니다.",
        "caveat": "독립 알파 가설이 아니라 유동성 상태를 확인하는 reference 피처입니다.",
    },
    "flow_foreign_netbuy_to_volume": {
        "group": "flow",
        "title": "외국인 순매수 강도",
        "definition": "20거래일 외국인 순매수 수량 합계를 같은 기간 전체 거래량으로 나눈 값입니다.",
        "reading": "값이 클수록 거래량 대비 외국인 순매수가 강했습니다.",
        "caveat": "보고서의 공식값은 당일 정보 시점을 피한 `lag1` variant입니다.",
    },
    "flow_inst_netbuy_to_volume": {
        "group": "flow",
        "title": "기관 순매수 강도",
        "definition": "20거래일 기관 순매수 수량 합계를 같은 기간 전체 거래량으로 나눈 값입니다.",
        "reading": "값이 클수록 거래량 대비 기관 순매수가 강했습니다.",
        "caveat": "보고서의 공식값은 당일 정보 시점을 피한 `lag1` variant입니다.",
    },
    "flow_individual_netbuy_to_volume": {
        "group": "flow",
        "title": "개인 순매수 강도",
        "definition": "20거래일 개인 순매수 수량 합계를 같은 기간 전체 거래량으로 나눈 값입니다.",
        "reading": "값이 클수록 거래량 대비 개인 순매수가 강했습니다.",
        "caveat": "외국인·기관·개인 합계는 기타법인을 빼므로 정확히 0이 되지 않습니다.",
    },
    "flow_foreign_holding_ratio_chg": {
        "group": "flow",
        "title": "외국인 보유비율 변화",
        "definition": "외국인 보유주식 수를 PIT 유통주식 수로 나눈 비율의 20거래일 변화입니다.",
        "reading": "값이 클수록 외국인 지분이 최근 늘었습니다.",
        "caveat": "과거 유통주식 수와 외국인 보유 데이터의 가용성에 영향을 받습니다.",
    },
    "flow_short_turnover": {
        "group": "flow",
        "title": "공매도 거래 비중",
        "definition": "최근 20일 공매도 거래량 합계를 전체 거래량 합계로 나눈 값입니다.",
        "reading": "값이 클수록 거래 중 공매도 비중이 높았습니다.",
        "caveat": "공매도 허용 구간만 계산하며 제도 변화에 따른 survival bias 때문에 exploratory로 봅니다.",
    },
    "flow_short_interest": {
        "group": "flow",
        "title": "공매도 잔고 비율",
        "definition": "공매도 잔고 수량을 PIT 유통주식 수로 나눈 값입니다.",
        "reading": "값이 클수록 아직 갚지 않은 공매도 포지션이 큽니다.",
        "caveat": "잔고 데이터가 2016년 중반부터 있어 긴 구간 비교에 제약이 있습니다.",
    },
    "flow_days_to_cover": {
        "group": "flow",
        "title": "공매도 상환 소요일",
        "definition": "공매도 잔고 수량을 최근 20일 평균 거래량으로 나눈 값입니다.",
        "reading": "값이 클수록 평소 거래량으로 공매도 잔고를 정리하는 데 오래 걸립니다.",
        "caveat": "공매도 허용 구간과 잔고 가용 구간만 쓰므로 exploratory 피처입니다.",
    },
    "flow_nat_proxy_20d": {
        "group": "flow",
        "title": "외국인 수급-공매도 균형",
        "definition": "시장 내 외국인 보유비율 변화 순위에서 공매도 잔고비율 순위를 뺀 값입니다.",
        "reading": "값이 클수록 외국인 지분 유입이 공매도 압력보다 상대적으로 강합니다.",
        "caveat": "두 데이터가 모두 있는 공매도 허용 구간에서만 계산합니다.",
    },
    "ev_amendment_ratio": {
        "group": "event",
        "title": "전체 공시 정정 비율",
        "definition": "최근 250거래일 전체 공시 중 정정공시가 차지하는 비율입니다.",
        "reading": "값이 클수록 최근 공시를 다시 고친 비율이 높아 보고 품질 위험이 큽니다.",
        "caveat": "공시는 접수 당일이 아니라 다음 거래일부터 노출합니다.",
    },
    "ev_filing_activity": {
        "group": "event",
        "title": "공시 활동 급증",
        "definition": "최근 60일 공시 건수를 과거 250일 동안의 60일 공시 건수 중앙값으로 나눈 값입니다.",
        "reading": "값이 클수록 평소보다 공시가 갑자기 많아졌습니다.",
        "caveat": "공시의 내용이나 호재·악재 방향이 아니라 빈도 변화만 잽니다.",
    },
    "ev_net_share_issuance_yoy": {
        "group": "event",
        "title": "연간 순주식 발행률",
        "definition": "최근 1년 경제적 발행 증가분에서 감소분을 뺀 뒤 전년 발행주식 수로 나눈 값입니다.",
        "reading": "값이 클수록 기존 주주의 지분 희석이 컸습니다.",
        "caveat": "액면분할·무상증자 같은 기계적 주식 수 변화는 발행에서 뺍니다.",
    },
    "ev_payout_yield": {
        "group": "event",
        "title": "주주환원 수익률",
        "definition": "최근 1년 현금배당과 자기주식 취득액을 더해 PIT 시가총액으로 나눈 값입니다.",
        "reading": "값이 클수록 시가총액 대비 배당·자사주 매입이 많았습니다.",
        "caveat": "배당 총액은 직접 공시값을 우선하고 필요할 때 주당배당금 기반 값을 씁니다.",
    },
    "fin_sue": {
        "group": "event",
        "title": "표준화 이익 서프라이즈",
        "definition": "분기 EPS의 전년 동기 대비 변화를 과거 같은 변화의 표준편차로 나눈 값입니다.",
        "reading": "값이 클수록 과거 변동 폭에 비해 이익 개선이 예상 밖으로 컸습니다.",
        "caveat": "최소 8개 과거 분기가 필요하며 이번 실행에서는 평가할 formation row가 없었습니다.",
    },
    "fin_accruals_to_assets": {
        "group": "financial",
        "title": "자산 대비 발생액",
        "definition": "TTM 순이익에서 영업현금흐름을 뺀 뒤 평균자산으로 나눈 값입니다.",
        "reading": "값이 클수록 이익 중 현금으로 들어오지 않은 부분이 많아 이익의 질이 낮을 수 있습니다.",
        "caveat": "순이익·현금흐름·자산을 같은 재무제표 기준으로 맞춥니다.",
    },
    "fin_asset_growth_yoy": {
        "group": "financial",
        "title": "총자산 증가율",
        "definition": "현재 총자산을 4개 분기 전 총자산으로 나눈 뒤 1을 뺀 값입니다.",
        "reading": "값이 클수록 최근 1년간 자산 투자가 빠르게 늘었습니다.",
        "caveat": "업종별 정상 성장률 차이를 중립화하지 않은 원값입니다.",
    },
    "fin_gross_profitability": {
        "group": "financial",
        "title": "총자산 대비 매출총이익",
        "definition": "TTM 매출총이익을 평균자산으로 나눈 값입니다.",
        "reading": "값이 클수록 보유 자산으로 더 많은 매출총이익을 냅니다.",
        "caveat": "매출총이익 개념이 약한 금융업과 원천 항목 coverage 차이에 주의해야 합니다.",
    },
    "fin_log_mcap": {
        "group": "financial",
        "title": "DART 기반 로그 시가총액",
        "definition": "PIT 발행주식 수와 종가로 구한 시가총액에 자연로그를 취한 값입니다.",
        "reading": "값이 클수록 회사 규모가 큽니다. 음의 IC는 소형주가 상대적으로 강했다는 뜻입니다.",
        "caveat": "공시 주식 수를 쓰므로 KRX 공식 시가총액보다 coverage가 낮습니다.",
    },
    "fin_value_z": {
        "group": "financial",
        "title": "복합 밸류 점수",
        "definition": "B/M·이익수익률·영업현금흐름수익률·매출액/가격을 시장별 z-score로 만든 뒤 평균낸 값입니다.",
        "reading": "값이 클수록 여러 기준에서 주가가 펀더멘털보다 싸게 평가됐습니다.",
        "caveat": "4개 구성요소 중 2개 이상이 있어야 하며 업종 중립화는 하지 않았습니다.",
    },
    "mcap_krx_log": {
        "group": "financial",
        "title": "KRX 로그 시가총액",
        "definition": "KRX Open API의 공식 시가총액에 자연로그를 취한 값입니다.",
        "reading": "값이 클수록 회사 규모가 큽니다. 음의 IC는 소형주 효과를 뜻합니다.",
        "caveat": "`fin_log_mcap`과 경제 개념이 같아 두 값을 동시에 쓸 때 중복을 확인해야 합니다.",
    },
    "hc_employee_growth": {
        "group": "human_capital",
        "title": "직원 수 증가율",
        "definition": "사업보고서 직원 수의 전년 대비 증가율입니다.",
        "reading": "값이 클수록 조직 규모가 빠르게 늘었습니다.",
        "caveat": "합병·분할 연도의 큰 변화를 걸러도 과거 정정 상태를 완전히 재현하지 못해 Grade B가 상한입니다.",
    },
    "hc_productivity": {
        "group": "human_capital",
        "title": "직원당 매출",
        "definition": "연간 매출을 직원 수로 나눈 뒤 자연로그를 취한 값입니다.",
        "reading": "값이 클수록 직원 한 명이 만드는 매출이 많습니다.",
        "caveat": "직원 수와 매출 공시가 모두 나온 뒤부터 쓰며 final-vintage 한계로 Grade B가 상한입니다.",
    },
    "own_amendment_ratio": {
        "group": "ownership",
        "title": "지분 공시 정정 비율",
        "definition": "최근 250거래일 임원·주요주주 및 5% 대량보유 공시 중 정정공시 비율입니다.",
        "reading": "값이 클수록 지분 관련 공시를 다시 고친 비율이 높습니다.",
        "caveat": "공시 접수 이력에서 계산하며 다음 거래일부터 노출합니다.",
    },
    "own_insider_filing_activity": {
        "group": "ownership",
        "title": "임원·주요주주 공시 급증",
        "definition": "최근 60일 임원·주요주주 소유상황 공시 건수를 과거 기준 중앙값으로 나눈 값입니다.",
        "reading": "값이 클수록 내부자 지분 관련 보고가 평소보다 갑자기 많아졌습니다.",
        "caveat": "실제 매수·매도 방향이 아니라 공시 건수만 세므로 신호 해석이 제한됩니다.",
    },
    "own_major_filing_activity": {
        "group": "ownership",
        "title": "5% 대량보유 공시 활동",
        "definition": "최근 60거래일 주식 등의 대량보유상황보고서 접수 건수입니다.",
        "reading": "값이 클수록 주요 보유자의 지분 신고 활동이 많았습니다.",
        "caveat": "지분 증가·감소 방향을 구분하지 않고 활동량만 잽니다.",
    },
    "own_major_stake_change": {
        "group": "ownership",
        "title": "최대주주 지분율 변화",
        "definition": "사업보고서의 최대주주 지분율에서 전년 값을 뺀 퍼센트포인트 변화입니다.",
        "reading": "값이 클수록 최대주주 지분율이 전년보다 높아졌습니다.",
        "caveat": "과거 정정 전 상태를 완전히 재현하지 못해 Grade B가 상한입니다.",
    },
    "own_major_stake_level": {
        "group": "ownership",
        "title": "최대주주 지분율",
        "definition": "사업보고서에 적힌 최대주주 및 특수관계인 합계 지분율 수준입니다.",
        "reading": "값이 클수록 지배주주의 보유 지분이 큽니다.",
        "caveat": "연간 공시의 final-vintage 값을 쓰므로 Grade B가 상한입니다.",
    },
}
HOME_PATH_RE = re.compile(r"/(?:Users|home)/[^/\s<]+")
EXTERNAL_ASSET_RE = re.compile(
    r"<(?:script\b[^>]*\bsrc|link\b[^>]*\bhref|img\b[^>]*\bsrc)\s*=\s*['\"]https?://",
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    r"(?:postgres(?:ql)?://|api[_-]?key\s*[=:]|password\s*[=:]|authorization\s*:)",
    re.IGNORECASE,
)
PLOTLY_RUNTIME_RE = re.compile(
    r"<script data-runtime=['\"]plotly['\"]>.*?</script>",
    re.IGNORECASE | re.DOTALL,
)


class ReportContractError(RuntimeError):
    """Raised when the selected artifacts do not satisfy the report contract."""


@dataclass(frozen=True)
class RunPaths:
    root: Path
    a: Path
    b: Path
    ab: Path
    # Repo checkout holding docs/target/*.json (T1/T2 result files) — a
    # separate base from `root` (the scan root) since S4 (2026-09-13) moved
    # published horizon_scan runs out of the repo into stock_data/.
    repo_root: Path


@dataclass
class ReportBundle:
    paths: RunPaths
    context: dict[str, Any]
    a_cards: list[dict[str, Any]]
    a_primary: pd.DataFrame
    a_exploratory: pd.DataFrame
    b_summary: pd.DataFrame
    ab_cells: pd.DataFrame
    correlations: pd.DataFrame
    phase_a_overlay: pd.DataFrame
    t1_results: dict[str, Any]
    t1_topk: dict[str, Any]
    t1_holdout: dict[str, Any]
    t2_results: dict[str, Any]
    family_rows: list[dict[str, Any]]
    kpis: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportContractError(f"JSON을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise ReportContractError(f"JSON 최상위가 object가 아닙니다: {path}")
    return value


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ReportContractError(f"Parquet이 없습니다: {path}")
    try:
        return pd.read_parquet(path)
    except (ImportError, OSError, ValueError) as exc:
        raise ReportContractError(f"Parquet을 읽을 수 없습니다: {path}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportContractError(message)


def _finite(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if getattr(missing, "shape", ()) == () else False
    except (TypeError, ValueError):
        return False


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if _is_missing_scalar(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return default


def _as_bool(value: Any) -> bool:
    return False if _is_missing_scalar(value) else bool(value)


def _true_count(values: Iterable[Any]) -> int:
    return sum(_as_bool(value) for value in values)


def align_ic(value: Any, expected_sign: str | None) -> float | None:
    """Return IC aligned so a positive number means the expected direction."""
    number = _finite(value)
    if number is None:
        return None
    return -number if expected_sign == "-" else number


def _display_number(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    number = _finite(value)
    if number is None:
        return "—"
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(number, spec)


def _display_int(value: Any) -> str:
    number = _finite(value)
    return "—" if number is None else f"{int(number):,}"


def _as_list(value: Any) -> list[Any]:
    if _is_missing_scalar(value):
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    text = str(value).strip()
    return [] if not text else [text]


def _grade_counts(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values if str(value) not in {"", "None", "nan"})
    return {grade: counts.get(grade, 0) for grade in GRADE_ORDER if counts.get(grade, 0)}


def _grade_counts_text(counts: Mapping[str, int]) -> str:
    return (
        " / ".join(f"{grade}{counts[grade]}" for grade in GRADE_ORDER if counts.get(grade)) or "—"
    )


def _source_for_family(family: str, phase: str, domain: str) -> str:
    if phase == "A":
        return "daily_ohlcv" if domain == "price" else "flow raw"
    if family == "mcap_krx_log":
        return "KRX Open API"
    if family == "fin_log_mcap":
        return "OpenDART + OHLCV"
    return "OpenDART"


def family_name_union(
    a_cards: Sequence[Mapping[str, Any]], b_family_names: Iterable[str]
) -> set[str]:
    return {str(card["family"]) for card in a_cards} | {str(name) for name in b_family_names}


def paired_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    candidate_name: str,
    metric_path: Sequence[str],
) -> float:
    """Calculate candidate-baseline delta for a nested metric."""

    def metric(record: Mapping[str, Any]) -> float:
        value: Any = record
        for key in metric_path:
            value = value[key]
        number = _finite(value)
        if number is None:
            raise ReportContractError(f"metric 값이 유효하지 않습니다: {metric_path}")
        return number

    selected = {
        str(record.get("name")): record
        for record in records
        if int(record.get("horizon", -1)) == horizon
    }
    _require("baseline" in selected, f"h={horizon} baseline record가 없습니다")
    _require(candidate_name in selected, f"h={horizon} {candidate_name} record가 없습니다")
    return metric(selected[candidate_name]) - metric(selected["baseline"])


def t1_h20_decision(topk: Mapping[str, Any]) -> tuple[str, float]:
    delta = paired_delta(
        topk.get("records", []),
        horizon=20,
        candidate_name="candidate",
        metric_path=("topk", "cost_adjusted_return"),
    )
    return ("not_adopted_h20" if delta < 0 else "gate_pass_h20", delta)


def _run_dir(
    root: Path,
    *,
    phase: str,
    snapshot_date: str,
    source: str,
    config_hash: str,
    run_id: str,
) -> Path:
    """``root`` is the scan root (``<STOCK_DATA_ROOT>/kr/output/scan``)."""
    return (
        root
        / "horizon_scan"
        / f"phase={phase}"
        / f"snapshot_date={snapshot_date}"
        / f"source={source}"
        / f"config_hash={config_hash}"
        / f"run_id={run_id}"
    )


def resolve_and_validate_paths(
    root: Path,
    *,
    snapshot_date: str,
    source: str,
    config_hash: str,
    ab_run_id: str,
    repo_root: Path | None = None,
) -> tuple[RunPaths, dict[str, Any]]:
    ab_path = _run_dir(
        root,
        phase="AB",
        snapshot_date=snapshot_date,
        source=source,
        config_hash=config_hash,
        run_id=ab_run_id,
    )
    ab_manifest = _read_json(ab_path / "manifest.json")
    ab_success = _read_json(ab_path / "_SUCCESS.json")
    _require(ab_manifest.get("phase") == "AB", "선택한 manifest의 phase가 AB가 아닙니다")
    if "snapshot_date" in ab_manifest:
        _require(
            ab_manifest.get("snapshot_date") == snapshot_date, "AB snapshot이 요청값과 다릅니다"
        )
    if "source" in ab_manifest:
        _require(ab_manifest.get("source") == source, "AB source가 요청값과 다릅니다")
    _require(ab_manifest.get("config_hash") == config_hash, "AB config hash가 요청값과 다릅니다")
    _require(ab_manifest.get("run_id") == ab_run_id, "AB manifest run id가 요청값과 다릅니다")
    _require(ab_success.get("run_id") == ab_run_id, "AB _SUCCESS run id가 요청값과 다릅니다")
    _require(bool(ab_success.get("content_hash")), "AB _SUCCESS content hash가 없습니다")

    phase_a_run_id = str(ab_manifest.get("phase_a_run_id") or "")
    phase_b_run_id = str(ab_manifest.get("phase_b_run_id") or "")
    _require(bool(phase_a_run_id and phase_b_run_id), "AB manifest에 A/B run id가 없습니다")
    a_path = _run_dir(
        root,
        phase="A",
        snapshot_date=snapshot_date,
        source=source,
        config_hash=config_hash,
        run_id=phase_a_run_id,
    )
    b_path = _run_dir(
        root,
        phase="B",
        snapshot_date=snapshot_date,
        source=source,
        config_hash=config_hash,
        run_id=phase_b_run_id,
    )

    a_manifest = _read_json(a_path / "manifest.json")
    b_manifest = _read_json(b_path / "manifest.json")
    a_success = _read_json(a_path / "_SUCCESS.json")
    b_success = _read_json(b_path / "_SUCCESS.json")
    for phase, manifest in (("A", a_manifest), ("B", b_manifest)):
        _require(manifest.get("snapshot_date") == snapshot_date, f"Phase {phase} snapshot 불일치")
        _require(manifest.get("source") == source, f"Phase {phase} source 불일치")
        _require(manifest.get("config_hash") == config_hash, f"Phase {phase} config 불일치")
        _require(manifest.get("official") is True, f"Phase {phase}가 official run이 아닙니다")
    _require(a_success.get("run_id") == phase_a_run_id, "Phase A _SUCCESS run id 불일치")
    _require(b_success.get("run_id") == phase_b_run_id, "Phase B _SUCCESS run id 불일치")
    _require(
        a_success.get("content_hash") == ab_manifest.get("phase_a_content_hash"),
        "Phase A content hash가 AB lineage와 다릅니다",
    )
    _require(
        b_success.get("content_hash") == ab_manifest.get("phase_b_content_hash"),
        "Phase B content hash가 AB lineage와 다릅니다",
    )

    a_spec = _read_json(a_path / "run_spec.json")
    b_spec = _read_json(b_path / "phase_b_run_spec.json")
    context = {
        "snapshot_date": snapshot_date,
        "source": source,
        "config_hash": config_hash,
        "generated_at": ab_manifest.get("generated_at"),
        "ab_manifest": ab_manifest,
        "success": {"A": a_success, "B": b_success, "AB": ab_success},
        "run_specs": {
            "A": {
                "git_commit": a_spec.get("git_commit"),
                "git_dirty": a_spec.get("git_dirty"),
                "command_line": a_spec.get("command_line"),
            },
            "B": {
                "git_commit": b_spec.get("git_commit"),
                "git_dirty": b_spec.get("git_dirty"),
                "command_line": b_spec.get("command_line"),
            },
        },
    }
    return RunPaths(
        root=root, a=a_path, b=b_path, ab=ab_path, repo_root=repo_root if repo_root is not None else root
    ), context


def _load_target_json(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    path = root / "docs/target/01_20_access_return_rank" / name
    return path, _read_json(path)


def _validate_target_inputs(bundle_context: Mapping[str, Any], root: Path) -> dict[str, Any]:
    t1_results_path, t1_results = _load_target_json(root, "grade_a_acceptance_gate_results.json")
    t1_topk_path, t1_topk = _load_target_json(root, "topk_cost_check.json")
    t1_holdout_path, t1_holdout = _load_target_json(root, "grade_a_acceptance_gate_holdout.json")
    t2_path, t2_results = _load_target_json(root, "phase_b_acceptance_gate_results.json")
    success = bundle_context["success"]["AB"]
    _require(
        t2_results.get("snapshot_date") == bundle_context["snapshot_date"], "T2 snapshot 불일치"
    )
    _require(t2_results.get("source") == bundle_context["source"], "T2 source 불일치")
    _require(t2_results.get("config_hash") == bundle_context["config_hash"], "T2 config 불일치")
    _require(t2_results.get("ab_run_id") == success.get("run_id"), "T2 AB run id 불일치")
    _require(
        t2_results.get("ab_content_hash") == success.get("content_hash"),
        "T2 AB content hash 불일치",
    )
    ab_run_dir = str(t2_results.get("ab_run_dir") or "")
    _require(f"run_id={success.get('run_id')}" in ab_run_dir, "T2 AB run dir 불일치")
    _require(len(t2_results.get("families", [])) == 14, "T2 family 수가 14가 아닙니다")
    _require(len(t2_results.get("features", [])) == 14, "T2 feature 수가 14가 아닙니다")
    return {
        "t1_results": t1_results,
        "t1_topk": t1_topk,
        "t1_holdout": t1_holdout,
        "t2_results": t2_results,
        "paths": {
            "t1_results": t1_results_path,
            "t1_topk": t1_topk_path,
            "t1_holdout": t1_holdout_path,
            "t2_results": t2_path,
        },
    }


def _phase_a_cell_sets(a_horizon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = (a_horizon["universe"] == "broad") & (a_horizon["sample_kind"] == "common_survivor")
    primary = a_horizon[common & (a_horizon["hypothesis_role"] == "primary")].copy()
    exploratory = a_horizon[
        common & (a_horizon["hypothesis_role"] == "exploratory_short_regime")
    ].copy()
    return primary, exploratory


def _family_rows(
    a_cards: Sequence[Mapping[str, Any]],
    a_primary: pd.DataFrame,
    a_exploratory: pd.DataFrame,
    b_summary: pd.DataFrame,
    ab_cells: pd.DataFrame,
    t2_families: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in a_cards:
        family = str(card["family"])
        primary_cells = a_primary[a_primary["family"] == family]
        exploratory_cells = a_exploratory[a_exploratory["family"] == family]
        is_exploratory = bool(card.get("exploratory_short_regime"))
        is_reference = str(card.get("evidence_grade")) == "R"
        if is_exploratory:
            current = "exploratory_short_regime"
        elif is_reference:
            current = "reference_only"
        elif card.get("screen_pass"):
            current = "screen_pass"
        else:
            current = "screened_out"
        ic_values = [_finite(value) for value in primary_cells.get("ic_mean", [])]
        ic_values = [value for value in ic_values if value is not None]
        n_obs_mean = (
            _finite(primary_cells["n_obs_mean"].median()) if not primary_cells.empty else None
        )
        rows.append(
            {
                "phase": "A",
                "domain": str(card.get("domain") or "—"),
                "source": _source_for_family(family, "A", str(card.get("domain") or "")),
                "family": family,
                "primary_feature": str(card.get("primary_feature") or "—"),
                "expected_sign": str(card.get("expected_sign") or "양방향"),
                "observed_sign": str(card.get("observed_sign") or "—"),
                "horizon_band": _horizon_band_text(card.get("candidate_horizon_band")),
                "ic_min": min(ic_values) if ic_values else None,
                "ic_max": max(ic_values) if ic_values else None,
                "q_min": _finite(card.get("q_fdr_global")) if primary_cells.shape[0] else None,
                "discovery_count": (
                    len(card.get("primary_discoveries") or []) if primary_cells.shape[0] else None
                ),
                "screen_pass_count": (
                    int(bool(card.get("screen_pass"))) if primary_cells.shape[0] else None
                ),
                "n_obs_mean": n_obs_mean,
                "n_dates": int(primary_cells["n_dates"].max()) if not primary_cells.empty else None,
                "n_obs": int(primary_cells["n_obs"].max()) if not primary_cells.empty else None,
                "coverage_ratio": None,
                "grade": str(card.get("evidence_grade") or "NE"),
                "grade_cap": "—",
                "grade_counts": {str(card.get("evidence_grade") or "NE"): 1},
                "failed_gates": list(card.get("warnings") or [])
                + list(card.get("limitations") or []),
                "model_input": "T1 candidate/baseline" if card.get("screen_pass") else "—",
                "current_status": current,
                "primary_cells": int(primary_cells.shape[0]),
                "exploratory_cells": int(exploratory_cells.shape[0]),
            }
        )

    for _, summary in b_summary.sort_values("family").iterrows():
        family = str(summary["family"])
        cells = ab_cells[ab_cells["family"] == family]
        valid = cells[cells["status"] == "valid"]
        ic_values = [_finite(value) for value in valid.get("ic_mean", [])]
        ic_values = [value for value in ic_values if value is not None]
        q_values = [_finite(value) for value in cells.get("q_fdr_global_ab", [])]
        q_values = [value for value in q_values if value is not None]
        grade_counts = _grade_counts(cells.get("evidence_grade", []))
        grade_caps = sorted(
            {
                text
                for value in cells.get("source_quality_grade_cap", [])
                if (text := _first_text(value))
            }
        )
        screen_count = _true_count(cells.get("screen_pass", pd.Series(dtype=bool)))
        insufficient = int((cells.get("status", pd.Series(dtype=str)) == "insufficient").sum())
        if family in t2_families:
            current = "validation_improved_bundle"
        elif insufficient == len(cells) and len(cells):
            current = "insufficient"
        elif screen_count:
            current = "screen_pass"
        else:
            current = "screened_out"
        failed = Counter(
            str(gate)
            for value in cells.get("failed_gates", [])
            for gate in _as_list(value)
            if str(gate)
        )
        rows.append(
            {
                "phase": "B",
                "domain": str(summary.get("fdr_family") or "—"),
                "source": _source_for_family(family, "B", str(summary.get("fdr_family") or "")),
                "family": family,
                "primary_feature": str(summary.get("primary_feature") or "—"),
                "expected_sign": str(summary.get("expected_sign") or "양방향"),
                "observed_sign": _observed_sign(ic_values),
                "horizon_band": _cell_band(cells),
                "ic_min": min(ic_values) if ic_values else None,
                "ic_max": max(ic_values) if ic_values else None,
                "q_min": min(q_values) if q_values else None,
                "discovery_count": int(
                    _true_count(cells.get("primary_discovery_ab", pd.Series(dtype=bool)))
                ),
                "screen_pass_count": screen_count,
                "n_obs_mean": _finite(valid["n_obs_mean"].median()) if not valid.empty else None,
                "n_dates": int(valid["n_dates"].max()) if not valid.empty else None,
                "n_obs": int(valid["n_obs"].max()) if not valid.empty else None,
                "coverage_ratio": _finite(summary.get("coverage_ratio")),
                "grade": _grade_counts_text(grade_counts),
                "grade_cap": ", ".join(grade_caps) or "—",
                "grade_counts": grade_counts,
                "failed_gates": [f"{name} ({count})" for name, count in failed.most_common()],
                "model_input": "T2 14-feature bundle" if family in t2_families else "—",
                "current_status": current,
                "primary_cells": int(cells.shape[0]),
                "exploratory_cells": 0,
            }
        )
    return sorted(rows, key=lambda row: (row["phase"], row["domain"], row["family"]))


def _horizon_band_text(value: Any) -> str:
    items = _as_list(value)
    return "—" if not items else "–".join(str(item) for item in items)


def _cell_band(cells: pd.DataFrame) -> str:
    if cells.empty:
        return "—"
    starts = [_finite(value) for value in cells["h_start"]]
    ends = [_finite(value) for value in cells["h_end"]]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    return "—" if not starts or not ends else f"{int(min(starts))}–{int(max(ends))}"


def _observed_sign(values: Sequence[float]) -> str:
    if not values:
        return "—"
    median = float(pd.Series(values).median())
    return "+" if median > 0 else "−" if median < 0 else "0"


def load_report_bundle(
    root: Path,
    *,
    snapshot_date: str,
    source: str,
    config_hash: str,
    ab_run_id: str,
    repo_root: Path | None = None,
) -> ReportBundle:
    """``root`` is the scan root; ``repo_root`` (default: this repo) holds ``docs/target/*``."""
    repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    paths, context = resolve_and_validate_paths(
        root,
        snapshot_date=snapshot_date,
        source=source,
        config_hash=config_hash,
        ab_run_id=ab_run_id,
        repo_root=repo_root,
    )
    a_cards_raw = json.loads((paths.a / "cards/family_cards.json").read_text(encoding="utf-8"))
    _require(isinstance(a_cards_raw, list), "Phase A family cards가 list가 아닙니다")
    a_cards = [dict(card) for card in a_cards_raw]
    a_horizon = _read_parquet(paths.a / "core/horizon_ic.parquet")
    a_primary, a_exploratory = _phase_a_cell_sets(a_horizon)
    b_summary = _read_parquet(paths.b / "core/family_summary.parquet")
    ab_cells = _read_parquet(paths.ab / "combined_ab_primary_hypotheses.parquet")
    correlations = _read_parquet(paths.ab / "primary_feature_rank_correlation.parquet")
    overlay = _read_parquet(paths.ab / "phase_a_card_overlay.parquet")
    targets = _validate_target_inputs(context, repo_root)

    names = family_name_union(a_cards, b_summary["family"].astype(str))
    _require(len(a_cards) == 17, f"Phase A family가 17개가 아닙니다: {len(a_cards)}")
    _require(b_summary["family"].nunique() == 18, "Phase B family가 18개가 아닙니다")
    _require(len(names) == 35, f"family 합집합이 35개가 아닙니다: {len(names)}")
    _require(
        names == set(FEATURE_GUIDES),
        "피처 안내와 실행 family가 다릅니다: "
        f"missing={sorted(names - set(FEATURE_GUIDES))}, extra={sorted(set(FEATURE_GUIDES) - names)}",
    )
    _require(len(a_primary) == 75, f"Phase A primary cell이 75개가 아닙니다: {len(a_primary)}")
    _require(
        len(a_exploratory) == 28,
        f"Phase A exploratory cell이 28개가 아닙니다: {len(a_exploratory)}",
    )
    _require(len(ab_cells) == 153, f"AB cell이 153개가 아닙니다: {len(ab_cells)}")
    _require(
        not a_primary["hypothesis_id"].duplicated().any(), "Phase A primary hypothesis_id 중복"
    )
    _require(
        not a_exploratory["hypothesis_id"].duplicated().any(),
        "Phase A exploratory hypothesis_id 중복",
    )
    _require(not ab_cells["hypothesis_id"].duplicated().any(), "AB hypothesis_id 중복")
    _require((a_primary["status"] == "valid").sum() == 75, "Phase A valid cell 불일치")
    _require(int(b_summary["ready_cells"].sum()) == 78, "Phase B ready cell 불일치")
    _require(ab_cells["family"].nunique() == 30, "AB family가 30개가 아닙니다")
    _require((ab_cells["status"] == "valid").sum() == 147, "AB valid cell 불일치")
    _require((ab_cells["status"] == "insufficient").sum() == 6, "AB insufficient cell 불일치")
    _require(_true_count(ab_cells["primary_discovery_ab"]) == 87, "AB discovery 불일치")
    _require(_true_count(ab_cells["screen_pass"]) == 40, "Phase B screen-pass 불일치")

    decision, t1_topk_delta = t1_h20_decision(targets["t1_topk"])
    t1_decile_delta = paired_delta(
        targets["t1_results"].get("records", []),
        horizon=20,
        candidate_name="candidate",
        metric_path=("economic", "cost_adjusted_spread"),
    )
    permutation = context["ab_manifest"].get("combined_cross_sectional_permutation", {})
    kpis = {
        "family_count": len(names),
        "a_primary_cells": len(a_primary),
        "a_exploratory_cells": len(a_exploratory),
        "b_ready_cells": int(b_summary["ready_cells"].sum()),
        "ab_hypotheses": len(ab_cells),
        "ab_valid": int((ab_cells["status"] == "valid").sum()),
        "ab_insufficient": int((ab_cells["status"] == "insufficient").sum()),
        "ab_discoveries": _true_count(ab_cells["primary_discovery_ab"]),
        "phase_b_screen_pass": _true_count(ab_cells["screen_pass"]),
        "phase_b_grade_counts": _grade_counts(ab_cells["evidence_grade"]),
        "a_grade_counts": _grade_counts(card.get("evidence_grade") for card in a_cards),
        "p_empirical_count": _finite(permutation.get("p_empirical_count")),
        "phase_a_discovery_changes": int(
            overlay["discovery_changed_vs_phase_a_only"].fillna(False).sum()
        ),
        "high_correlation_pairs": int((correlations["mean_rank_corr"].abs() >= 0.7).sum()),
        "t1_decision": decision,
        "t1_decile_h20_delta": t1_decile_delta,
        "t1_topk_h20_delta": t1_topk_delta,
        "t2_status": targets["t2_results"].get("validation_summary", {}).get("status"),
        "holdout_status": targets["t2_results"].get("holdout_status"),
    }
    family_rows = _family_rows(
        a_cards,
        a_primary,
        a_exploratory,
        b_summary,
        ab_cells,
        set(map(str, targets["t2_results"].get("families", []))),
    )
    _require(len(family_rows) == 35, "family 표 row가 35개가 아닙니다")
    _require(len({row["family"] for row in family_rows}) == 35, "family 표에 중복이 있습니다")

    context["target_paths"] = targets["paths"]
    return ReportBundle(
        paths=paths,
        context=context,
        a_cards=a_cards,
        a_primary=a_primary,
        a_exploratory=a_exploratory,
        b_summary=b_summary,
        ab_cells=ab_cells,
        correlations=correlations,
        phase_a_overlay=overlay,
        t1_results=targets["t1_results"],
        t1_topk=targets["t1_topk"],
        t1_holdout=targets["t1_holdout"],
        t2_results=targets["t2_results"],
        family_rows=family_rows,
        kpis=kpis,
    )


def _plot_imports() -> tuple[Any, Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.offline import get_plotlyjs
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ReportContractError(
            "Plotly가 없습니다. `uv run --extra analysis`로 실행하십시오"
        ) from exc
    return go, make_subplots, get_plotlyjs


def _chart_config() -> dict[str, Any]:
    return {
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


def _figure_html(figure: Any) -> str:
    webgl_types = {"scattergl", "heatmapgl", "pointcloud"}
    used_webgl = sorted({str(getattr(trace, "type", "")) for trace in figure.data} & webgl_types)
    _require(not used_webgl, f"WebGL trace는 사용할 수 없습니다: {', '.join(used_webgl)}")
    return figure.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=_chart_config(),
    )


def _base_layout(figure: Any, *, height: int, title: str) -> Any:
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template=None,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        colorway=["#2563eb", "#7c3aed", "#0f766e", "#d97706", "#dc2626"],
        height=height,
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        font={"family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "size": 12},
        hoverlabel={"font_size": 12},
    )
    figure.update_xaxes(gridcolor="#e7ecf2", zerolinecolor="#cbd5e1", automargin=True)
    figure.update_yaxes(gridcolor="#e7ecf2", zerolinecolor="#cbd5e1", automargin=True)
    return figure


def _flow_figure(bundle: ReportBundle) -> Any:
    go, make_subplots, _ = _plot_imports()
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Phase A", "Phase B"))
    a_labels = ["가설 (cell)", "BH pass (cell)", "discovery (cell)", "screen-pass (family)"]
    a_values = [
        len(bundle.a_primary),
        _true_count(bundle.a_primary["bh_pass"]),
        _true_count(bundle.a_primary["primary_discovery"]),
        _true_count(card.get("screen_pass") for card in bundle.a_cards),
    ]
    b_labels = ["가설 (cell)", "discovery (cell)", "screen-pass (cell)", "Grade A/B (cell)"]
    b_families = set(bundle.b_summary["family"].astype(str))
    b_cells = bundle.ab_cells[bundle.ab_cells["family"].astype(str).isin(b_families)]
    b_values = [
        len(b_cells),
        _true_count(b_cells["primary_discovery_ab"]),
        _true_count(b_cells["screen_pass"]),
        int(b_cells["evidence_grade"].isin(["A", "B"]).sum()),
    ]
    for col, labels, values, color in (
        (1, a_labels, a_values, PHASE_COLORS["A"]),
        (2, b_labels, b_values, PHASE_COLORS["B"]),
    ):
        figure.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=color,
                text=values,
                textposition="outside",
                hovertemplate="%{y}: %{x}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    figure.update_yaxes(autorange="reversed")
    figure.add_annotation(
        text="노드의 cell/family 단위를 확인하십시오. 직선 funnel이 아닙니다.",
        x=0.5,
        y=-0.2,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#64748b"},
    )
    return _base_layout(figure, height=430, title="A/B 평가 흐름")


def _cell_label(row: Mapping[str, Any]) -> str:
    cell_type = _first_text(row.get("cell_type"), row.get("scan_type"), default="cell")
    start_value = _finite(row.get("h_start"))
    end_value = _finite(row.get("h_end"))
    start = int(start_value) if start_value is not None else 0
    end = int(end_value) if end_value is not None else 0
    return f"{cell_type} {start}–{end}"


def _heatmap_records(bundle: ReportBundle) -> pd.DataFrame:
    a_frames: list[pd.DataFrame] = []
    for data, role in ((bundle.a_primary, "primary"), (bundle.a_exploratory, "exploratory")):
        frame = data.copy()
        frame["report_phase"] = "A"
        frame["report_role"] = role
        frame["report_q"] = frame.get("q_fdr_global")
        frame["report_discovery"] = frame.get("primary_discovery", False)
        frame["report_screen"] = False
        frame["report_grade"] = frame["family"].map(
            {card["family"]: card.get("evidence_grade") for card in bundle.a_cards}
        )
        frame["report_failed"] = [[] for _ in range(len(frame))]
        a_frames.append(frame)
    b_frame = bundle.ab_cells.copy()
    b_frame["report_phase"] = b_frame["family"].map(
        lambda family: "A" if family in set(bundle.a_primary["family"]) else "B"
    )
    b_frame = b_frame[b_frame["report_phase"] == "B"].copy()
    b_frame["report_role"] = "primary"
    b_frame["report_q"] = b_frame.get("q_fdr_global_ab")
    b_frame["report_discovery"] = b_frame.get("primary_discovery_ab", False)
    b_frame["report_screen"] = b_frame.get("screen_pass", False)
    b_frame["report_grade"] = b_frame.get("evidence_grade")
    b_frame["report_failed"] = b_frame.get("failed_gates")
    columns = sorted(set(a_frames[0].columns) | set(a_frames[1].columns) | set(b_frame.columns))
    combined = pd.concat(
        [frame.reindex(columns=columns) for frame in [*a_frames, b_frame]],
        ignore_index=True,
    )
    combined["row_label"] = combined.apply(
        lambda row: f"{row['report_phase']} · {row['family']} · {row['feature']}", axis=1
    )
    combined["cell_label"] = combined.apply(_cell_label, axis=1)
    combined["aligned_ic"] = combined.apply(
        lambda row: align_ic(row.get("ic_mean"), row.get("expected_sign")), axis=1
    )
    return combined


def _heatmap_figure(bundle: ReportBundle) -> Any:
    go, _, _ = _plot_imports()
    records = _heatmap_records(bundle)
    order = (
        records[["cell_label", "h_start", "h_end"]]
        .drop_duplicates()
        .sort_values(["h_start", "h_end", "cell_label"])["cell_label"]
        .tolist()
    )
    rows = records["row_label"].drop_duplicates().tolist()
    pivot = records.pivot_table(
        index="row_label", columns="cell_label", values="ic_mean", aggfunc="first"
    ).reindex(index=rows, columns=order)
    z_values = pivot.to_numpy(dtype=float)
    max_abs = max(0.01, float(pd.Series(z_values.ravel()).abs().max(skipna=True)))
    figure = go.Figure(
        go.Heatmap(
            z=z_values,
            x=order,
            y=rows,
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            colorscale="RdBu_r",
            colorbar={"title": "raw IC"},
            hovertemplate="%{y}<br>%{x}<br>IC=%{z:.4f}<extra></extra>",
            hoverongaps=False,
        )
    )
    discovery_mask = records["report_discovery"].map(_as_bool)
    screen_mask = records["report_screen"].map(_as_bool)
    markers = records[
        discovery_mask
        | screen_mask
        | (records["status"] == "insufficient")
        | (records["report_role"] == "exploratory")
    ]
    symbol_map = {
        "screen": "diamond-open",
        "discovery": "circle-open",
        "insufficient": "x",
        "exploratory": "triangle-up-open",
    }
    for kind in ("screen", "discovery", "insufficient", "exploratory"):
        marker_discovery = markers["report_discovery"].map(_as_bool)
        marker_screen = markers["report_screen"].map(_as_bool)
        if kind == "screen":
            selected = markers[marker_screen]
        elif kind == "discovery":
            selected = markers[marker_discovery & ~marker_screen]
        elif kind == "insufficient":
            selected = markers[markers["status"] == "insufficient"]
        else:
            selected = markers[markers["report_role"] == "exploratory"]
        if selected.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=selected["cell_label"],
                y=selected["row_label"],
                mode="markers",
                name=kind,
                marker={
                    "symbol": symbol_map[kind],
                    "size": 8,
                    "color": "#111827" if kind != "exploratory" else PHASE_COLORS["exploratory"],
                    "line": {"width": 1.4},
                },
                hovertemplate=f"{kind}<extra></extra>",
            )
        )
    figure.update_xaxes(tickangle=-40)
    return _base_layout(figure, height=max(820, 25 * len(rows) + 220), title="family × horizon IC")


def _sample_scatter_figure(bundle: ReportBundle) -> Any:
    go, _, _ = _plot_imports()
    records = _heatmap_records(bundle)
    records = records[(records["status"] == "valid") & records["n_obs_mean"].notna()].copy()
    figure = go.Figure()
    for phase, role, color, symbol in (
        ("A", "primary", PHASE_COLORS["A"], "circle"),
        ("B", "primary", PHASE_COLORS["B"], "diamond"),
        ("A", "exploratory", PHASE_COLORS["exploratory"], "triangle-up-open"),
    ):
        selected = records[(records["report_phase"] == phase) & (records["report_role"] == role)]
        if selected.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=selected["n_obs_mean"],
                y=selected["ic_mean"],
                mode="markers",
                name=f"{phase} {role}",
                marker={"color": color, "symbol": symbol, "size": 8, "opacity": 0.72},
                customdata=selected[["family", "feature", "n_dates", "n_obs"]].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]} · %{customdata[1]}<br>"
                    "날짜당 평균 표본=%{x:,.0f}<br>IC=%{y:.4f}<br>"
                    "n_dates=%{customdata[2]:,.0f}<br>n_obs=%{customdata[3]:,.0f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
    figure.update_xaxes(title="cell n_obs_mean (날짜당 평균 표본 수)")
    figure.update_yaxes(title="raw IC")
    return _base_layout(figure, height=560, title="표본 규모와 IC")


def _grade_gate_figure(bundle: ReportBundle) -> Any:
    go, make_subplots, _ = _plot_imports()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("등급 분포", "Phase B 주요 실패 게이트"),
        column_widths=(0.42, 0.58),
    )
    a_counts = bundle.kpis["a_grade_counts"]
    b_counts = bundle.kpis["phase_b_grade_counts"]
    for grade in GRADE_ORDER:
        if grade == "NE":
            continue
        figure.add_trace(
            go.Bar(
                x=["A family", "B cell"],
                y=[a_counts.get(grade, 0), b_counts.get(grade, 0)],
                name=grade,
                marker_color=GRADE_COLORS[grade],
                text=[a_counts.get(grade, 0) or "", b_counts.get(grade, 0) or ""],
                hovertemplate=f"Grade {grade}<br>%{{x}}: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    failed = Counter(
        str(gate)
        for value in bundle.ab_cells["failed_gates"]
        for gate in _as_list(value)
        if str(gate)
    )
    top_failed = failed.most_common(9)
    figure.add_trace(
        go.Bar(
            x=[count for _, count in top_failed][::-1],
            y=[name for name, _ in top_failed][::-1],
            orientation="h",
            name="failed gates",
            marker_color="#f59e0b",
            text=[count for _, count in top_failed][::-1],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title="cell/family 수", row=1, col=1)
    figure.update_xaxes(title="실패 cell 수", row=1, col=2)
    return _base_layout(figure, height=540, title="등급과 게이트")


def _correlation_figure(bundle: ReportBundle) -> Any:
    go, _, _ = _plot_imports()
    corr = bundle.correlations
    matrix = corr.pivot(index="family_a", columns="family_b", values="mean_rank_corr")
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu_r",
            colorbar={"title": "mean ρ"},
            hovertemplate="A %{y}<br>B %{x}<br>ρ=%{z:.3f}<extra></extra>",
        )
    )
    high = corr[corr["mean_rank_corr"].abs() >= 0.7]
    if not high.empty:
        figure.add_trace(
            go.Scatter(
                x=high["family_b"],
                y=high["family_a"],
                mode="markers+text",
                text=high["mean_rank_corr"].map(lambda value: f"{value:+.2f}"),
                textposition="top center",
                name="|ρ| ≥ 0.7",
                marker={"symbol": "diamond-open", "size": 13, "color": "#111827"},
                hovertemplate="%{y} ↔ %{x}<br>ρ=%{text}<extra></extra>",
            )
        )
    figure.update_xaxes(tickangle=-45)
    return _base_layout(
        figure,
        height=650,
        title="A primary 12 × B primary 17 교차 rank correlation",
    )


def _records_by_name(data: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for record in data.get("records", []):
        result.setdefault(str(record["name"]), []).append(record)
    for records in result.values():
        records.sort(key=lambda record: int(record["horizon"]))
    return result


def _validation_figure(bundle: ReportBundle) -> Any:
    go, make_subplots, _ = _plot_imports()
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "T1 Rank IC",
            "T1 decile 비용 반영 spread",
            "T2 Rank IC",
            "T2 decile 비용 반영 spread",
        ),
        vertical_spacing=0.18,
    )
    panels = [
        (bundle.t1_results, "candidate", "mean_rank_ic", 1, 1),
        (bundle.t1_results, "candidate", "economic.cost_adjusted_spread", 1, 2),
        (bundle.t2_results, "phase_b_candidate", "mean_rank_ic", 2, 1),
        (bundle.t2_results, "phase_b_candidate", "economic.cost_adjusted_spread", 2, 2),
    ]
    for data, candidate_name, metric, row, col in panels:
        grouped = _records_by_name(data)
        for name, color in (("baseline", "#94a3b8"), (candidate_name, "#0f766e")):
            records = grouped[name]
            values: list[float | None] = []
            for record in records:
                value: Any = record
                for key in metric.split("."):
                    value = value[key]
                values.append(_finite(value))
            figure.add_trace(
                go.Bar(
                    x=[f"h{record['horizon']}" for record in records],
                    y=values,
                    name="baseline" if name == "baseline" else "candidate",
                    legendgroup=f"{row}-{name}",
                    showlegend=(col == 1),
                    marker_color=color,
                    text=[_display_number(value, 4) for value in values],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
    figure.update_layout(barmode="group")
    return _base_layout(figure, height=780, title="모델 validation — baseline 대 candidate")


def _detail_curve_figure(bundle: ReportBundle) -> Any:
    go, _, _ = _plot_imports()
    records = _heatmap_records(bundle)
    families = [row["family"] for row in bundle.family_rows]
    figure = go.Figure()
    trace_ranges: dict[str, list[int]] = {}
    for family in families:
        selected = records[records["family"] == family].sort_values(["h_end", "h_start"])
        if selected.empty:
            continue
        trace_ranges[family] = []
        for kind, symbol in (("cum", "circle"), ("bucket", "square")):
            part = selected[selected.apply(lambda row: kind in _cell_label(row), axis=1)]
            if part.empty:
                continue
            trace_ranges[family].append(len(figure.data))
            figure.add_trace(
                go.Scatter(
                    x=part["h_end"],
                    y=part["ic_mean"],
                    mode="lines+markers",
                    name=kind,
                    visible=False,
                    marker={"symbol": symbol, "size": 8},
                    customdata=part[["cell_label", "report_q", "n_obs_mean", "status"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>IC=%{y:.4f}<br>q=%{customdata[1]}<br>"
                        "n_obs_mean=%{customdata[2]:,.0f}<br>status=%{customdata[3]}<extra></extra>"
                    ),
                )
            )
    available = [family for family in families if trace_ranges.get(family)]
    if available:
        for index in trace_ranges[available[0]]:
            figure.data[index].visible = True
    buttons = []
    for family in available:
        visible = [False] * len(figure.data)
        for index in trace_ranges[family]:
            visible[index] = True
        buttons.append(
            {
                "label": family,
                "method": "update",
                "args": [{"visible": visible}, {"title": f"family 상세 curve — {family}"}],
            }
        )
    figure.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ]
    )
    figure.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
    figure.update_xaxes(title="horizon end (거래일)")
    figure.update_yaxes(title="raw IC")
    title = f"family 상세 curve — {available[0]}" if available else "family 상세 curve"
    return _base_layout(figure, height=570, title=title)


def _cell_details(bundle: ReportBundle, family: str, phase: str) -> list[dict[str, Any]]:
    if phase == "A":
        frames = []
        for source, role in (
            (bundle.a_primary, "primary"),
            (bundle.a_exploratory, "exploratory_short_regime"),
        ):
            selected = source[source["family"] == family].copy()
            if selected.empty:
                continue
            selected["report_role"] = role
            frames.append(selected)
        cells = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        cells = bundle.ab_cells[bundle.ab_cells["family"] == family].copy()
        if not cells.empty:
            cells["report_role"] = "primary"
    if cells.empty:
        return []
    details = []
    for _, cell in cells.sort_values(["h_start", "h_end", "feature"]).iterrows():
        is_b = phase == "B"
        details.append(
            {
                "role": str(cell.get("report_role") or "primary"),
                "feature": _first_text(cell.get("feature"), default="—"),
                "cell": _cell_label(cell),
                "ic": _finite(cell.get("ic_mean")),
                "aligned_ic": align_ic(cell.get("ic_mean"), cell.get("expected_sign")),
                "q": _finite(cell.get("q_fdr_global_ab" if is_b else "q_fdr_global")),
                "discovery": _as_bool(
                    cell.get("primary_discovery_ab" if is_b else "primary_discovery")
                ),
                "screen_pass": _as_bool(cell.get("screen_pass")) if is_b else None,
                "grade": (
                    _first_text(cell.get("evidence_grade"), default="—") if is_b else "family card"
                ),
                "n_dates": _finite(cell.get("n_dates")),
                "n_obs_mean": _finite(cell.get("n_obs_mean")),
                "status": _first_text(cell.get("status"), default="—"),
                "failed_gates": ", ".join(map(str, _as_list(cell.get("failed_gates")))) or "—",
            }
        )
    return details


def _status_label(status: str) -> str:
    return {
        "validation_improved_bundle": "bundle validation 개선",
        "screen_pass": "screen-pass",
        "screened_out": "screening 탈락",
        "exploratory_short_regime": "primary cell 없음 · exploratory",
        "reference_only": "primary cell 없음 · reference",
        "insufficient": "평가 불가",
    }.get(status, status)


def _rich_text(value: str) -> str:
    escaped = html.escape(value)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def _feature_result_text(row: Mapping[str, Any]) -> str:
    status = str(row["current_status"])
    expected = str(row["expected_sign"])
    observed = str(row["observed_sign"])
    ic_min = _finite(row.get("ic_min"))
    ic_max = _finite(row.get("ic_max"))
    parts: list[str] = []
    if ic_min is not None and ic_max is not None:
        parts.append(
            f"primary IC는 {_display_number(ic_min, 4, signed=True)}부터 "
            f"{_display_number(ic_max, 4, signed=True)}까지입니다."
        )
        if expected in {"+", "-"}:
            normalized_observed = "-" if observed == "−" else observed
            parts.append(
                "기대 방향과 맞았습니다."
                if normalized_observed == expected
                else "기대 방향과 반대였습니다."
            )
        elif observed in {"+", "-", "−"}:
            parts.append(f"관측 부호는 {'−' if observed in {'-', '−'} else '+'}입니다.")

    discovery = row.get("discovery_count")
    screen = row.get("screen_pass_count")
    if discovery is not None:
        if row["phase"] == "A":
            parts.append(
                f"discovery {_display_int(discovery)}개, family screen-pass "
                f"{'통과' if _finite(screen) else '미통과'}입니다."
            )
        else:
            parts.append(
                f"discovery {_display_int(discovery)}개, screen-pass {_display_int(screen)}개이며 "
                f"등급은 {row['grade']}입니다."
            )

    if status == "screen_pass":
        parts.append(
            "Phase A screening은 통과했지만 이 피처들이 들어간 T1 후보 묶음은 h20 비용 성과 때문에 비채택입니다."
        )
    elif status == "validation_improved_bundle":
        parts.append(
            "T2 14-feature bundle에 들어갔고 묶음 전체는 h5·h20·h60에서 개선됐습니다. 개별 기여도는 따로 확인하지 않았습니다."
        )
    elif status == "screened_out":
        parts.append("이번 screening에서는 채택 근거가 부족했습니다.")
    elif status == "exploratory_short_regime":
        parts.append(
            "관측 부호는 참고할 수 있지만 공매도 제도 변화와 survival bias 때문에 진단용으로만 봅니다."
        )
    elif status == "reference_only":
        parts.append("독립 가설로 검정하지 않은 reference 피처라 성능 판정을 내리지 않았습니다.")
    elif status == "insufficient":
        parts.append("유효 formation row가 없어 이번 실행에서는 성능을 판단할 수 없습니다.")
    return " ".join(parts)


def _feature_guide_html(bundle: ReportBundle) -> str:
    rows = {str(row["family"]): row for row in bundle.family_rows}
    groups = []
    for group, label in FEATURE_GROUPS:
        families = [
            family for family, guide in FEATURE_GUIDES.items() if guide["group"] == group
        ]
        cards = []
        for family in families:
            guide = FEATURE_GUIDES[family]
            row = rows[family]
            status = str(row["current_status"])
            cards.append(
                "<article class='feature-card' data-testid='feature-guide-card'>"
                "<div class='feature-card-head'>"
                f"<div><div class='feature-title'>{html.escape(guide['title'])}</div>"
                f"<div class='feature-code'><code>{html.escape(family)}</code> · primary "
                f"<code>{html.escape(str(row['primary_feature']))}</code></div></div>"
                f"<div class='feature-badges'><span class='phase phase-{str(row['phase']).lower()}'>{html.escape(str(row['phase']))}</span>"
                f"<span class='status status-{html.escape(status)}'>{html.escape(_status_label(status))}</span></div></div>"
                "<dl class='feature-definition'>"
                f"<div><dt>계산</dt><dd>{_rich_text(guide['definition'])}</dd></div>"
                f"<div><dt>값 읽기</dt><dd>{_rich_text(guide['reading'])}</dd></div>"
                f"<div><dt>주의</dt><dd>{_rich_text(guide['caveat'])}</dd></div>"
                "</dl>"
                f"<div class='feature-verdict'><strong>이번 결과</strong><p>{html.escape(_feature_result_text(row))}</p></div>"
                "</article>"
            )
        groups.append(
            f"<details class='feature-group' open><summary>{html.escape(label)}"
            f"<span>{len(cards)}개 family</span></summary>"
            f"<div class='feature-grid'>{''.join(cards)}</div></details>"
        )
    return "".join(groups)


def _chart_commentary(bundle: ReportBundle) -> dict[str, str]:
    b_families = set(bundle.b_summary["family"].astype(str))
    b_cells = bundle.ab_cells[bundle.ab_cells["family"].astype(str).isin(b_families)]
    flow = (
        f"Phase A는 {len(bundle.a_primary)}개 cell 중 BH를 {_true_count(bundle.a_primary['bh_pass'])}개가 통과했지만, "
        f"최종 screen-pass는 {_true_count(card.get('screen_pass') for card in bundle.a_cards)}개 family로 줄었습니다. "
        f"Phase B는 {len(b_cells)}개 cell 중 {_true_count(b_cells['primary_discovery_ab'])}개가 discovery, "
        f"{_true_count(b_cells['screen_pass'])}개가 screen-pass였습니다. 통계적 유의성만으로 채택하지 않고 방향·거래가능성·강건성 gate가 많이 걸러냈습니다."
    )

    ranked_rows = sorted(
        (
            row
            for row in bundle.family_rows
            if _finite(row.get("ic_min")) is not None and _finite(row.get("ic_max")) is not None
        ),
        key=lambda row: max(abs(float(row["ic_min"])), abs(float(row["ic_max"]))),
        reverse=True,
    )
    strongest = []
    for row in ranked_rows[:4]:
        values = [float(row["ic_min"]), float(row["ic_max"])]
        value = max(values, key=abs)
        strongest.append(f"{row['family']} ({_display_number(value, 4, signed=True)})")
    heatmap = (
        f"절댓값 기준으로 두드러진 신호는 {', '.join(strongest)}입니다. "
        "색이 진해도 marker가 없으면 discovery나 screen-pass를 뜻하지 않습니다. 여러 horizon에서 같은 방향이 이어지는지 함께 봐야 합니다."
    )

    records = _heatmap_records(bundle)
    valid = records[(records["status"] == "valid") & records["n_obs_mean"].notna()].copy()
    valid["abs_ic"] = valid["ic_mean"].abs()
    sample_rho = valid[["n_obs_mean", "abs_ic"]].corr(method="spearman").iloc[0, 1]
    scatter = (
        f"날짜당 표본 수와 |IC|의 Spearman 상관은 {sample_rho:+.2f}로, 표본이 큰 피처가 자동으로 강한 신호를 보인 패턴은 약합니다. "
        "다만 표본이 작은 점의 큰 IC는 불확실성이 더 크므로 n_dates·coverage·강건성 gate를 같이 봐야 합니다."
    )

    failed = Counter(
        str(gate)
        for value in bundle.ab_cells["failed_gates"]
        for gate in _as_list(value)
        if str(gate)
    )
    failed_text = ", ".join(f"{name} {count}개" for name, count in failed.most_common(3))
    grades = (
        f"Phase A는 {_grade_counts_text(bundle.kpis['a_grade_counts'])}, Phase B cell은 "
        f"{_grade_counts_text(bundle.kpis['phase_b_grade_counts'])}입니다. "
        f"Phase B에서 가장 자주 막힌 항목은 {failed_text}입니다. 따라서 유의한 IC가 나와도 기간별 재현성이 부족하면 C로 남는 경우가 많습니다."
    )

    high = bundle.correlations[bundle.correlations["mean_rank_corr"].abs() >= 0.7]
    high_text = ", ".join(
        f"{row.family_a}↔{row.family_b} {row.mean_rank_corr:+.2f}"
        for row in high.itertuples(index=False)
    )
    correlation = (
        f"|ρ|≥0.7인 쌍은 {high_text} 두 쌍뿐입니다. 둘 다 Amihud 비유동성과 시가총액의 음의 상관으로, "
        "저유동성 효과와 소형주 효과가 겹친다는 뜻입니다. 나머지 A×B 조합은 대체로 다른 정보를 담습니다."
    )

    deltas = {
        int(item["horizon"]): item for item in bundle.t2_results["validation_summary"]["deltas"]
    }
    validation = (
        f"T1은 h20 비용 반영 spread가 decile에서 {_display_number(bundle.kpis['t1_decile_h20_delta'], 4, signed=True)}, "
        f"k=100에서 {_display_number(bundle.kpis['t1_topk_h20_delta'], 4, signed=True)}로 나빠져 비채택입니다. "
        f"T2 bundle은 세 horizon을 모두 개선했지만 Rank IC 증분은 h60에서 {_display_number(deltas[60]['mean_rank_ic'], 4, signed=True)}로 작습니다. "
        "최종 판단은 새 h60 holdout 전까지 보류해야 합니다."
    )

    fully_screened = [
        str(row["family"])
        for row in bundle.family_rows
        if row["phase"] == "B"
        and _finite(row.get("screen_pass_count")) == _finite(row.get("primary_cells"))
        and (_finite(row.get("primary_cells")) or 0) > 0
    ]
    curves = (
        f"B에서 등록된 primary cell이 모두 screen-pass한 family는 {', '.join(fully_screened)}입니다. "
        "상세 curve에서는 한 점의 최대 IC보다 cumulative와 bucket이 같은 방향을 유지하는지, 긴 horizon에서 신호가 급격히 약해지지 않는지를 먼저 확인해야 합니다."
    )
    return {
        "flow": flow,
        "heatmap": heatmap,
        "scatter": scatter,
        "grades": grades,
        "correlation": correlation,
        "validation": validation,
        "curves": curves,
    }


def _chart_note(title: str, text: str) -> str:
    return (
        "<div class='chart-note'><div class='chart-note-label'>그래프 총평</div>"
        f"<strong>{html.escape(title)}</strong><p>{html.escape(text)}</p></div>"
    )


def _family_table_html(bundle: ReportBundle) -> str:
    body: list[str] = []
    for row in bundle.family_rows:
        family = row["family"]
        details = _cell_details(bundle, family, row["phase"])
        gate_text = ", ".join(row["failed_gates"][:3]) or "—"
        status = str(row["current_status"])
        body.append(
            "<tr class='family-row' "
            f"data-phase='{html.escape(row['phase'])}' data-status='{html.escape(status)}' "
            f"data-grade='{html.escape(row['grade'])}' data-search='{html.escape((family + ' ' + row['primary_feature'] + ' ' + row['domain']).lower())}'>"
            f"<td><span class='phase phase-{row['phase'].lower()}'>{row['phase']}</span></td>"
            f"<td><code>{html.escape(family)}</code></td>"
            f"<td>{html.escape(row['domain'])}</td><td>{html.escape(row['source'])}</td>"
            f"<td><code>{html.escape(row['primary_feature'])}</code></td>"
            f"<td>{html.escape(row['expected_sign'])} / {html.escape(row['observed_sign'])}</td>"
            f"<td>{html.escape(row['horizon_band'])}</td>"
            f"<td>{_display_number(row['ic_min'], 4, signed=True)} … {_display_number(row['ic_max'], 4, signed=True)}</td>"
            f"<td>{_display_number(row['q_min'], 4)}</td>"
            f"<td>{_display_int(row['discovery_count'])} / {_display_int(row['screen_pass_count'])}</td>"
            f"<td>{_display_int(row['n_obs_mean'])}</td>"
            f"<td>{_display_int(row['n_dates'])}</td>"
            f"<td>{_display_int(row['n_obs'])}</td>"
            f"<td>{_display_number(row['coverage_ratio'], 3)}</td>"
            f"<td><span class='grade grade-{html.escape(str(row['grade'])[0].lower())}'>{html.escape(str(row['grade']))}</span></td>"
            f"<td>{html.escape(row['grade_cap'])}</td>"
            f"<td title='{html.escape(gate_text)}'>{html.escape(gate_text)}</td>"
            f"<td>{html.escape(row['model_input'])}</td>"
            f"<td><span class='status status-{html.escape(status)}'>{html.escape(_status_label(status))}</span></td>"
            "</tr>"
        )
        detail_rows = "".join(
            "<tr>"
            f"<td>{html.escape(detail['role'])}</td><td><code>{html.escape(detail['feature'])}</code></td>"
            f"<td>{html.escape(detail['cell'])}</td>"
            f"<td>{_display_number(detail['ic'], 4, signed=True)}</td>"
            f"<td>{_display_number(detail['aligned_ic'], 4, signed=True)}</td>"
            f"<td>{_display_number(detail['q'], 4)}</td>"
            f"<td>{'✓' if detail['discovery'] else '—'}</td>"
            f"<td>{'✓' if detail['screen_pass'] else '—'}</td>"
            f"<td>{html.escape(detail['grade'])}</td>"
            f"<td>{_display_int(detail['n_dates'])}</td>"
            f"<td>{_display_int(detail['n_obs_mean'])}</td>"
            f"<td>{html.escape(detail['status'])}</td>"
            f"<td>{html.escape(detail['failed_gates'])}</td>"
            "</tr>"
            for detail in details
        )
        if not detail_rows:
            detail_rows = "<tr><td colspan='13'>계산된 cell이 없습니다.</td></tr>"
        body.append(
            "<tr class='detail-row'><td colspan='19'><details>"
            f"<summary>{len(details)}개 cell 상세 보기</summary>"
            "<div class='table-scroll'><table class='cell-table'><thead><tr>"
            "<th>role</th><th>feature</th><th>cell</th><th>IC</th><th>aligned IC</th>"
            "<th>q</th><th>discovery</th><th>screen</th><th>grade</th><th>n_dates</th>"
            "<th>n_obs_mean</th><th>status</th><th>failed gates</th>"
            f"</tr></thead><tbody>{detail_rows}</tbody></table></div></details></td></tr>"
        )
    return "".join(body)


def _topk_table(bundle: ReportBundle) -> str:
    grouped = _records_by_name(bundle.t1_topk)
    rows = []
    for baseline, candidate in zip(grouped["baseline"], grouped["candidate"], strict=True):
        horizon = int(baseline["horizon"])
        base = _finite(baseline["topk"]["cost_adjusted_return"])
        cand = _finite(candidate["topk"]["cost_adjusted_return"])
        delta = None if base is None or cand is None else cand - base
        rows.append(
            f"<tr><td>h{horizon}</td><td>{_display_number(base, 4)}</td>"
            f"<td>{_display_number(cand, 4)}</td><td>{_display_number(delta, 4, signed=True)}</td>"
            f"<td>{'통과' if delta is not None and delta >= 0 else '미통과'}</td></tr>"
        )
    return "".join(rows)


def _validation_delta_table(bundle: ReportBundle) -> str:
    rows = []
    for item in bundle.t2_results["validation_summary"]["deltas"]:
        rows.append(
            f"<tr><td>h{int(item['horizon'])}</td>"
            f"<td>{_display_number(item['mean_rank_ic'], 4, signed=True)}</td>"
            f"<td>{_display_number(item['cost_adjusted_spread'], 4, signed=True)}</td></tr>"
        )
    return "".join(rows)


def _report_css() -> str:
    return """
:root{--ink:#152238;--muted:#617083;--paper:#f5f7fb;--card:#fff;--line:#dbe2ea;
--blue:#2563eb;--purple:#7c3aed;--teal:#0f766e;--amber:#d97706;--red:#dc2626}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}
.shell{max-width:1480px;margin:0 auto;padding:28px}.hero{background:linear-gradient(130deg,#10243e,#183b66 60%,#245578);
color:#fff;border-radius:20px;padding:34px;box-shadow:0 16px 40px #10243e22}.eyebrow{font-size:12px;
letter-spacing:.12em;text-transform:uppercase;color:#b9d7f7}.hero h1{margin:.35rem 0;font-size:clamp(28px,4vw,48px);
line-height:1.12}.hero p{max-width:960px;color:#dbeafe}.badge{display:inline-flex;padding:6px 11px;border:1px solid #fcd34d;
border-radius:999px;background:#78350f55;color:#fde68a;font-weight:750;font-size:12px}.lineage{margin-top:18px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#bfdbfe;overflow-wrap:anywhere}
.nav{position:sticky;top:0;z-index:5;margin:16px 0;padding:10px 14px;background:#ffffffee;backdrop-filter:blur(12px);
border:1px solid var(--line);border-radius:12px;display:flex;gap:14px;overflow-x:auto}.nav a{color:#334155;
text-decoration:none;white-space:nowrap;font-size:13px}.nav a:hover{color:var(--blue)}.kpis{display:grid;
grid-template-columns:repeat(6,minmax(140px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:16px;box-shadow:0 5px 18px #0f172a0a}.kpi .label{font-size:12px;color:var(--muted)}
.kpi .value{font-size:26px;font-weight:780;margin-top:4px}.kpi .note{font-size:11px;color:var(--muted)}
.callout{padding:16px 18px;border-radius:12px;border-left:4px solid var(--amber);background:#fffbeb;margin:16px 0}
.callout.good{border-color:var(--teal);background:#ecfdf5}.callout.bad{border-color:var(--red);background:#fef2f2}
section{margin:20px 0;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;
box-shadow:0 8px 24px #0f172a08}section h2{margin:0 0 6px;font-size:23px}section h3{margin-top:22px}
.section-note{color:var(--muted);font-size:14px;margin-top:0}.chart{width:100%;overflow:hidden}.chart-note{margin:4px 0 22px;
padding:15px 17px;border:1px solid #bfdbfe;border-radius:12px;background:#eff6ff;color:#1e3a5f}.chart-note-label{font-size:11px;
font-weight:800;letter-spacing:.08em;color:#2563eb}.chart-note strong{display:block;margin-top:2px}.chart-note p{margin:4px 0 0;font-size:14px}
.feature-intro{display:grid;grid-template-columns:1.1fr .9fr;gap:14px;margin:16px 0}.feature-intro>div{padding:14px 16px;
border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0;font-size:13px}.feature-group{margin-top:14px;border:1px solid var(--line);
border-radius:14px;background:#f8fafc;overflow:hidden}.feature-group>summary{cursor:pointer;padding:14px 16px;font-weight:780;background:#f1f5f9}
.feature-group>summary span{margin-left:8px;color:var(--muted);font-size:12px;font-weight:600}.feature-grid{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:12px}.feature-card{background:#fff;border:1px solid #e2e8f0;
border-radius:12px;padding:15px;min-width:0}.feature-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.feature-title{font-size:16px;font-weight:780}.feature-code{font-size:11px;color:var(--muted);overflow-wrap:anywhere;margin-top:2px}
.feature-badges{display:flex;justify-content:flex-end;gap:5px;flex-wrap:wrap}.feature-definition{margin:13px 0 0}.feature-definition>div{
display:grid;grid-template-columns:52px 1fr;gap:7px;padding:6px 0;border-top:1px solid #eef2f7}.feature-definition dt{font-size:12px;
font-weight:750;color:#475569}.feature-definition dd{margin:0;font-size:13px}.feature-verdict{margin-top:10px;padding:10px 12px;border-radius:10px;
background:#f8fafc;border-left:3px solid var(--teal);font-size:13px}.feature-verdict p{margin:3px 0 0}.grid-2{display:grid;
grid-template-columns:1fr 1fr;gap:16px}.metric-table,.family-table,.cell-table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border-bottom:1px solid #e7ecf2;padding:8px 9px;text-align:left;vertical-align:top}th{background:#f8fafc;
color:#475569;position:sticky;top:0;z-index:1}.table-scroll{overflow:auto;max-width:100%}.family-table{min-width:1900px}
.family-row:hover{background:#f8fbff}.detail-row>td{background:#f8fafc;padding:0 12px 10px}.detail-row summary{cursor:pointer;
color:var(--blue);padding:8px 0}.cell-table{min-width:1150px;background:#fff}.cell-table th{position:static}.phase,.grade,.status{
display:inline-flex;border-radius:999px;padding:3px 8px;font-weight:700;white-space:nowrap}.phase-a{background:#dbeafe;color:#1d4ed8}
.phase-b{background:#ede9fe;color:#6d28d9}.grade-a{background:#ccfbf1;color:#115e59}.grade-b{background:#dbeafe;color:#1d4ed8}
.grade-c{background:#fef3c7;color:#92400e}.grade-d{background:#fee2e2;color:#991b1b}.grade-r,.grade-n{background:#e2e8f0;color:#475569}
.status{background:#eef2ff;color:#3730a3;font-size:11px}.status-screened_out,.status-insufficient{background:#f1f5f9;color:#64748b}
.status-exploratory_short_regime{background:#fff7ed;color:#9a3412}.status-reference_only{background:#f1f5f9;color:#475569}
.status-validation_improved_bundle{background:#dcfce7;color:#166534}.filters{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.filters label{font-size:12px;color:var(--muted)}.filters input,.filters select{display:block;margin-top:4px;border:1px solid var(--line);
border-radius:8px;padding:8px 10px;background:#fff;min-width:160px}.small{font-size:12px;color:var(--muted)}code{font-family:ui-monospace,
SFMono-Regular,Menlo,monospace;font-size:.92em}.method-list{columns:2;column-gap:32px}.method-list li{break-inside:avoid;margin-bottom:8px}
.footer{font-size:12px;color:var(--muted);padding:18px 4px 40px;overflow-wrap:anywhere}
@media(max-width:1000px){.kpis{grid-template-columns:repeat(3,1fr)}.grid-2,.feature-intro,.feature-grid{grid-template-columns:1fr}.method-list{columns:1}}
@media(max-width:600px){.shell{padding:12px}.hero{padding:23px;border-radius:14px}.kpis{grid-template-columns:repeat(2,1fr)}
section{padding:14px}.kpi .value{font-size:22px}.nav{border-radius:8px}.filters label,.filters input,.filters select{width:100%}
.feature-card-head{display:block}.feature-badges{justify-content:flex-start;margin-top:8px}.feature-definition>div{grid-template-columns:1fr}.feature-definition dd{margin-top:-3px}}
@media print{body{background:#fff}.shell{max-width:none;padding:0}.hero{box-shadow:none;background:#10243e!important;-webkit-print-color-adjust:exact;
print-color-adjust:exact}.nav,.filters,.modebar{display:none!important}section,.kpi{box-shadow:none;break-inside:avoid}.chart{break-inside:avoid}
.family-table{font-size:9px;min-width:0}.detail-row{display:none}.kpis{grid-template-columns:repeat(3,1fr)}.feature-group{break-inside:auto}
.feature-card{break-inside:avoid}.feature-grid{grid-template-columns:repeat(2,1fr)}}
"""


def _report_js() -> str:
    return """
const search=document.getElementById('family-search');
const phase=document.getElementById('phase-filter');
const status=document.getElementById('status-filter');
function applyFilters(){
  const q=search.value.trim().toLowerCase(),p=phase.value,s=status.value;
  document.querySelectorAll('#family-table tbody tr.family-row').forEach(row=>{
    const show=(!q||row.dataset.search.includes(q))&&(!p||row.dataset.phase===p)&&(!s||row.dataset.status===s);
    row.style.display=show?'':'none';
    const detail=row.nextElementSibling;if(detail&&detail.classList.contains('detail-row'))detail.style.display=show?'':'none';
  });
}
[search,phase,status].forEach(el=>el.addEventListener(el.tagName==='INPUT'?'input':'change',applyFilters));
document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{
  const body=document.querySelector('#family-table tbody');
  const pairs=[];let row=body.firstElementChild;
  while(row){const detail=row.nextElementSibling;pairs.push([row,detail]);row=detail?detail.nextElementSibling:null;}
  const key=Number(button.dataset.sort),asc=button.dataset.direction!=='asc';button.dataset.direction=asc?'asc':'desc';
  pairs.sort((a,b)=>a[0].children[key].innerText.localeCompare(b[0].children[key].innerText,'ko',{numeric:true})*(asc?1:-1));
  pairs.forEach(pair=>pair.forEach(node=>node&&body.appendChild(node)));
}));
"""


def _git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ReportContractError(f"repo 밖의 경로를 manifest에 넣을 수 없습니다: {path}") from exc


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp | date):
        return value.isoformat()
    if hasattr(value, "item"):
        return _sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _report_manifest(bundle: ReportBundle) -> dict[str, Any]:
    root = bundle.paths.root
    scan_paths = {
        "a_success": bundle.paths.a / "_SUCCESS.json",
        "a_run_spec": bundle.paths.a / "run_spec.json",
        "a_family_cards": bundle.paths.a / "cards/family_cards.json",
        "a_horizon_ic": bundle.paths.a / "core/horizon_ic.parquet",
        "b_success": bundle.paths.b / "_SUCCESS.json",
        "b_run_spec": bundle.paths.b / "phase_b_run_spec.json",
        "b_family_summary": bundle.paths.b / "core/family_summary.parquet",
        "ab_success": bundle.paths.ab / "_SUCCESS.json",
        "ab_manifest": bundle.paths.ab / "manifest.json",
        "ab_cells": bundle.paths.ab / "combined_ab_primary_hypotheses.parquet",
        "ab_correlations": bundle.paths.ab / "primary_feature_rank_correlation.parquet",
    }
    inputs = {
        name: {"path": _relative(root, path), "sha256": sha256_file(path)}
        for name, path in scan_paths.items()
    }
    inputs.update(
        {
            name: {"path": _relative(bundle.paths.repo_root, path), "sha256": sha256_file(path)}
            for name, path in bundle.context["target_paths"].items()
        }
    )
    success = bundle.context["success"]
    return _sanitize_json(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "snapshot_date": bundle.context["snapshot_date"],
            "source": bundle.context["source"],
            "config_hash": bundle.context["config_hash"],
            "runs": {
                phase: {
                    "run_id": success[phase].get("run_id"),
                    "content_hash": success[phase].get("content_hash"),
                    **(bundle.context["run_specs"].get(phase, {}) if phase in {"A", "B"} else {}),
                }
                for phase in ("A", "B", "AB")
            },
            "report_code": _git_identity(root),
            "inputs": inputs,
            "row_counts": {
                "a_primary": len(bundle.a_primary),
                "a_exploratory": len(bundle.a_exploratory),
                "b_family_summary": len(bundle.b_summary),
                "ab_cells": len(bundle.ab_cells),
                "correlations": len(bundle.correlations),
                "family_table": len(bundle.family_rows),
            },
            "kpis": bundle.kpis,
            "t2_deltas": bundle.t2_results["validation_summary"]["deltas"],
            "holdout_status": bundle.kpis["holdout_status"],
        }
    )


def _assert_safe_output(index_html: str, manifest_text: str) -> None:
    report_owned_html = PLOTLY_RUNTIME_RE.sub("", index_html)
    combined = report_owned_html + "\n" + manifest_text
    if HOME_PATH_RE.search(combined):
        raise ReportContractError("산출물에 사용자 홈 절대경로가 들어 있습니다")
    if SECRET_RE.search(combined):
        raise ReportContractError("산출물에 비밀정보로 보이는 문자열이 들어 있습니다")
    if EXTERNAL_ASSET_RE.search(index_html):
        raise ReportContractError("HTML에 외부 asset URL이 들어 있습니다")


def _render_html(bundle: ReportBundle) -> str:
    _, _, get_plotlyjs = _plot_imports()
    flow_html = _figure_html(_flow_figure(bundle))
    heatmap_html = _figure_html(_heatmap_figure(bundle))
    scatter_html = _figure_html(_sample_scatter_figure(bundle))
    grade_html = _figure_html(_grade_gate_figure(bundle))
    correlation_html = _figure_html(_correlation_figure(bundle))
    validation_html = _figure_html(_validation_figure(bundle))
    detail_html = _figure_html(_detail_curve_figure(bundle))
    success = bundle.context["success"]
    report_date = str(bundle.context.get("generated_at") or "2026-08-28")[:10]
    config_short = bundle.context["config_hash"][:12]
    grade_text = _grade_counts_text(bundle.kpis["phase_b_grade_counts"])
    a_grade_text = _grade_counts_text(bundle.kpis["a_grade_counts"])
    chart_commentary = _chart_commentary(bundle)
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>피쳐 성능 평가 — {html.escape(report_date)} validation</title><style>{_report_css()}</style>
<script data-runtime="plotly">{plotly_js}</script></head><body><main class="shell">
<header class="hero" data-testid="report-hero"><div class="eyebrow">SDC · PRIVATE RESEARCH</div>
<span class="badge">FINAL HOLDOUT PENDING</span><h1>피쳐 성능 평가 — {html.escape(report_date)} validation 기준</h1>
<p>35개 family의 screening, 강건성, coverage와 모델 validation을 한 화면에 정리했습니다.
이 문서는 2026년 10~11월 h60 최종 holdout 전의 validation 보고서이며 투자자문 자료가 아닙니다.</p>
<div class="lineage">snapshot={bundle.context['snapshot_date']} · source={bundle.context['source']} ·
config={config_short}…<br>A={success['A']['run_id']} · B={success['B']['run_id']} · AB={success['AB']['run_id']}</div></header>
<nav class="nav" aria-label="보고서 목차"><a href="#summary">요약</a><a href="#features">피처 안내</a><a href="#screening">Screening</a>
<a href="#coverage">표본·강건성</a><a href="#redundancy">중복</a><a href="#validation">모델 validation</a>
<a href="#curves">상세 curve</a><a href="#families">전체 family</a><a href="#method">해석 기준</a></nav>
<section id="summary"><h2>한 장 요약</h2><p class="section-note">AB 153개는 primary 가설입니다. Phase A exploratory 28개는 따로 표시합니다.</p>
<div class="kpis"><div class="kpi"><div class="label">전체 family</div><div class="value">35</div><div class="note">A 17 + B 18</div></div>
<div class="kpi"><div class="label">AB 가설</div><div class="value">153</div><div class="note">A 75 + B 78</div></div>
<div class="kpi"><div class="label">AB discovery</div><div class="value">87</div><div class="note">permutation p={_display_number(bundle.kpis['p_empirical_count'],4)}</div></div>
<div class="kpi"><div class="label">B screen-pass</div><div class="value">40</div><div class="note">{grade_text}</div></div>
<div class="kpi"><div class="label">T1 h20</div><div class="value">비채택</div><div class="note">k=100 Δ {_display_number(bundle.kpis['t1_topk_h20_delta'],4,signed=True)}</div></div>
<div class="kpi"><div class="label">T2 bundle</div><div class="value">개선</div><div class="note">h5·h20·h60 valid</div></div></div>
<div class="callout good"><strong>T2:</strong> 14개 feature 묶음은 valid 구간에서 세 horizon의 Rank IC와 비용 반영 spread를 모두 높였습니다.
개별 feature 14개가 각각 모델 성능을 높였다는 뜻은 아닙니다.</div>
<div class="callout bad"><strong>남은 gate:</strong> 새 h60 라벨이 성숙한 뒤 holdout을 한 번만 엽니다. 현재 결과로 최종 채택을 선언하지 않습니다.</div></section>
<section id="features" data-testid="feature-guide"><h2>피처 안내와 이번 결과</h2>
<p class="section-note">각 family가 재는 값, 숫자가 커질 때의 뜻, 이번 screening 결과를 함께 정리했습니다.</p>
<div class="feature-intro"><div><strong>코드 읽기:</strong> family는 가설과 gate를 묶는 단위이고, primary feature는 실제 계산에 쓴 컬럼입니다. <code>lag1</code>은 전 거래일 값을 썼다는 뜻입니다.</div>
<div><strong>결과 읽기:</strong> 양의 IC는 피처 값이 큰 종목의 이후 수익률 순위가 높았다는 뜻입니다. 기대 부호가 <code>-</code>이면 음의 raw IC가 가설과 맞습니다.</div></div>
{_feature_guide_html(bundle)}</section>
<section id="screening"><h2>Screening 결과</h2><p class="section-note">A와 B는 단위가 달라 하나의 funnel로 연결하지 않습니다.</p>
<div class="chart">{flow_html}</div>{_chart_note('A/B 평가 흐름', chart_commentary['flow'])}
<div class="chart">{heatmap_html}</div>{_chart_note('family × horizon IC', chart_commentary['heatmap'])}</section>
<section id="coverage"><h2>표본과 강건성</h2><p class="section-note">x축은 두 phase에 공통으로 있는 cell 날짜당 평균 표본 수입니다. A coverage ratio는 계산하지 않았습니다.</p>
<div class="chart">{scatter_html}</div>{_chart_note('표본 규모와 IC', chart_commentary['scatter'])}
<div class="chart">{grade_html}</div>{_chart_note('등급과 게이트', chart_commentary['grades'])}</section>
<section id="redundancy"><h2>A×B 정보 중복</h2><p class="section-note">A×A와 B×B는 이 입력에 없으므로 판단하지 않습니다. |ρ|≥0.7은 {bundle.kpis['high_correlation_pairs']}쌍입니다.</p>
<div class="chart">{correlation_html}</div>{_chart_note('A×B 교차 rank correlation', chart_commentary['correlation'])}</section>
<section id="validation"><h2>모델 validation</h2><p class="section-note">T1은 2026-08-24 decile, T2는 2026-08-28 decile입니다. T1 k=100은 별도 2026-08-12 gate 기록입니다.</p>
<p class="small">T1 8월 24일 decile 비용 반영 spread Δ(h20)는 {_display_number(bundle.kpis['t1_decile_h20_delta'],4,signed=True)}입니다. 아래 k=100 판정과는 별도 실행입니다.</p>
<div class="chart">{validation_html}</div>{_chart_note('baseline 대 candidate', chart_commentary['validation'])}<div class="grid-2"><div><h3>T1 k=100 비용 확인</h3>
<table class="metric-table"><thead><tr><th>horizon</th><th>baseline</th><th>candidate</th><th>Δ</th><th>gate</th></tr></thead>
<tbody>{_topk_table(bundle)}</tbody></table><p class="small">h20 Δ가 0보다 작아 20일 모델은 비채택입니다. 이 표를 8월 24일 decile과 같은 실행으로 해석하지 않습니다.</p></div>
<div><h3>T2 14-feature bundle Δ</h3><table class="metric-table"><thead><tr><th>horizon</th><th>Rank IC Δ</th><th>비용 spread Δ</th></tr></thead>
<tbody>{_validation_delta_table(bundle)}</tbody></table><p class="small">최종 holdout: {html.escape(str(bundle.kpis['holdout_status']))}</p></div></div></section>
<section id="curves"><h2>Family 상세 curve</h2><p class="section-note">dropdown에서 family를 선택하십시오. 공매도 exploratory는 진단값이며 BH·채택 대상이 아닙니다.</p>
<div class="chart">{detail_html}</div>{_chart_note('family별 horizon curve', chart_commentary['curves'])}</section>
<section id="families"><h2>전체 35 family</h2><p class="section-note">표의 `—`는 계산하지 않은 값이며 0이 아닙니다. 열 제목을 누르면 정렬됩니다.</p>
<div class="filters"><label>검색<input id="family-search" data-testid="family-search" type="search" placeholder="family 또는 feature"></label>
<label>Phase<select id="phase-filter" data-testid="phase-filter"><option value="">전체</option><option value="A">A</option><option value="B">B</option></select></label>
<label>상태<select id="status-filter" data-testid="status-filter"><option value="">전체</option><option value="validation_improved_bundle">bundle validation 개선</option>
<option value="screen_pass">screen-pass</option><option value="screened_out">screening 탈락</option><option value="exploratory_short_regime">exploratory</option>
<option value="reference_only">reference</option><option value="insufficient">평가 불가</option></select></label></div>
<div class="table-scroll"><table id="family-table" class="family-table" data-testid="family-table"><thead><tr>
<th><button data-sort="0">phase</button></th><th><button data-sort="1">family</button></th><th>domain</th><th>source</th><th>primary feature</th>
<th>기대/관측</th><th>horizon</th><th>primary IC 범위</th><th>min q</th><th>discovery/screen</th><th>n_obs_mean</th>
<th>n_dates</th><th>n_obs</th><th>coverage</th><th>grade</th><th>grade cap</th><th>주요 gate</th><th>모델 입력</th>
<th>현재 판정</th></tr></thead><tbody>{_family_table_html(bundle)}</tbody></table></div></section>
<section id="method"><h2>해석 기준과 한계</h2><ul class="method-list">
<li><strong>Rank IC</strong>: 그날 feature 순위와 이후 수익률 순위의 Spearman 상관입니다.</li>
<li><strong>ICIR</strong>: 기간별 IC 평균을 IC 변동성으로 나눈 안정성 지표입니다.</li>
<li><strong>BH q</strong>: 여러 사전등록 가설을 함께 검정한 뒤의 오탐 통제값입니다.</li>
<li><strong>screen-pass</strong>: 방향·기간·tradable·delay·강건성 gate까지 통과한 cell입니다.</li>
<li><strong>Grade A/B</strong>: screening 근거 등급이며 모델 채택 상태가 아닙니다.</li>
<li><strong>ready와 evaluable</strong>: B 78개가 ready지만 fin_sue 6개는 formation row가 없어 insufficient입니다.</li>
<li><strong>T1 C와 T2 C</strong>: A의 C는 exploratory/보류일 수 있고 B의 C는 강건성 또는 availability 실패입니다.</li>
<li><strong>PIT</strong>: 재무·이벤트 값은 시장에서 알 수 있었던 날짜 이후에만 노출했습니다.</li>
<li><strong>final-vintage</strong>: 인적자본·지분 일부는 과거 시점 재현에 한계가 있어 grade cap이 있습니다.</li>
<li><strong>공매도 4 family</strong>: survival bias 때문에 exploratory로만 봅니다.</li>
<li><strong>상관 범위</strong>: A primary 12 × B primary 17 교차만 포함합니다.</li>
<li><strong>T2 증분성</strong>: 14개를 한 번에 추가한 bundle 결과라 개별 기여도를 말할 수 없습니다.</li>
<li><strong>T1 과거 holdout</strong>: 평가일 16일·h20 리밸런싱 1회라 결론에 쓰지 않았습니다.</li>
<li><strong>holdout</strong>: 기존 구간을 재사용하지 않고 2026년 10~11월 새 h60 구간을 한 번 평가합니다.</li>
</ul></section>
<footer class="footer">비공개 개인 연구용 · report schema {REPORT_SCHEMA_VERSION}<br>
Phase A grade {a_grade_text} · AB status valid {bundle.kpis['ab_valid']} / insufficient {bundle.kpis['ab_insufficient']} ·
Phase A discovery change {bundle.kpis['phase_a_discovery_changes']}<br>
입력 lineage는 같은 디렉터리의 <code>report_manifest.json</code>에 있습니다.</footer>
</main><script>{_report_js()}</script></body></html>"""


def generate_report(bundle: ReportBundle, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _report_manifest(bundle)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    index_html = _render_html(bundle)
    _assert_safe_output(index_html, manifest_text)
    size_limit = 10 * 1024 * 1024
    if len(index_html.encode("utf-8")) > size_limit:
        raise ReportContractError("index.html이 10MB 상한을 넘었습니다")

    index_path = output_dir / "index.html"
    manifest_path = output_dir / "report_manifest.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as tmp:
        tmp.write(index_html)
        tmp_index = Path(tmp.name)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as tmp:
        tmp.write(manifest_text)
        tmp_manifest = Path(tmp.name)
    os.replace(tmp_index, index_path)
    os.replace(tmp_manifest, manifest_path)
    return index_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="피쳐 성능 self-contained HTML 보고서 생성")
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--source", default="sj2_remote")
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--ab-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from modeler.etl.config import DataRoot

    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    scan_root = DataRoot.resolve(market="kr").output / "scan"
    try:
        bundle = load_report_bundle(
            scan_root,
            snapshot_date=args.snapshot_date,
            source=args.source,
            config_hash=args.config_hash,
            ab_run_id=args.ab_run_id,
        )
        index_path, manifest_path = generate_report(bundle, root / args.output_dir)
    except ReportContractError as exc:
        raise SystemExit(f"report contract error: {exc}") from exc
    print(f"HTML: {_relative(root, index_path)}")
    print(f"manifest: {_relative(root, manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
