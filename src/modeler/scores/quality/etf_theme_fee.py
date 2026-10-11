"""ETF 테마 카드 총보수 설명값 — DIS 월말 보수를 카드에 붙인다 (2026-10-11).

원천은 금융투자협회 DIS 펀드별 보수비용비교 월말 사본이다(사용자 결정 2026-10-11).
**설명값이다.** 상품을 고르는 근거로 쓰지 않는다 — E3 판정은 "(전진 확인 대기)"·"잠정"이고,
사전등록 v2 §8.1이 전진 확인이 같은 방향일 때부터 선택 근거로 허용한다.

- DIS 행: ``<tmpV1>``\\~``<tmpV16>`` 묶음. tmpV9 합계 보수(A, %), tmpV12 TER(A+B, %), tmpV15 펀드 표준코드.
  같은 달 같은 코드 중복 행은 첫 행(E3와 같은 규칙).
- 대응표: 신뢰도 ``high`` + 수동 확인 20쌍만 쓴다(E3와 같은 규칙).
- 기준일 = 컷오프(기본 2026-10-10) 이하인 월말 파일 중 가장 늦은 것.

상태 ``fee_status``: 값 있음 / 대응 미확인(대응표에 있으나 high·수동확인 아님) /
그 달 DIS에 없음(대응은 됐는데 기준일 파일에 그 펀드 행이 없거나 값이 둘 다 비어 있음) /
대응표에 없음(대응표에 단축코드가 없음).
"""

from __future__ import annotations

import gzip
import re
from datetime import date, datetime
from pathlib import Path

import polars as pl

FEE_VERSION = "quality-score-v0/etf_theme_fee/1"

DIS_DIR_REL = "kr/output/etf_dis_fee_research_pull_20261011"
CODE_MAP_REL = "kr/output/etf_dis_fee_code_map_20261011/etf_dis_fee_code_map.csv"
MANUAL_REL = "kr/output/etf_holding_cost_e3_inputs_20261011/dis_map_manual_confirm.csv"

DEFAULT_CUTOFF = date(2026, 10, 10)  # 카드 첫 판 날짜. 이날까지 공개·수집된 월말만 쓴다

ST_OK = "값 있음"
ST_UNCONFIRMED = "대응 미확인"
ST_NOT_IN_MONTH = "그 달 DIS에 없음"
ST_NOT_IN_MAP = "대응표에 없음"
STATUSES = (ST_OK, ST_UNCONFIRMED, ST_NOT_IN_MONTH, ST_NOT_IN_MAP)

FEE_COLUMNS = ["fee_total_pct", "ter_pct", "fee_basis_date", "fee_status"]

_RE_ROW = re.compile(r"<selectMeta>(.*?)</selectMeta>", re.S)
_RE_TAG = re.compile(r"<tmpV(\d+)>([^<]*)</tmpV\d+>")


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_dis_xml(text: str) -> list[dict]:
    """DIS XML 문자열 → 행(fund_code, sum_fee, ter). 표준코드 없는 행은 버린다. 자기닫힘 태그는 None."""
    rows = []
    for m in _RE_ROW.finditer(text):
        tags = {int(k): v for k, v in _RE_TAG.findall(m.group(1))}
        code = (tags.get(15) or "").strip()
        if not code:
            continue
        rows.append({"fund_code": code, "sum_fee": _num(tags.get(9)), "ter": _num(tags.get(12))})
    return rows


def first_per_code(rows: list[dict]) -> dict[str, dict]:
    """같은 표준코드 중복 행은 첫 행."""
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["fund_code"], r)
    return out


def available_months(raw_dir: str | Path) -> list[date]:
    ds = []
    for p in Path(raw_dir).glob("*.xml.gz"):
        ds.append(datetime.strptime(p.name.split(".")[0], "%Y%m%d").date())
    return sorted(ds)


def pick_basis_date(months: list[date], cutoff: date = DEFAULT_CUTOFF) -> date:
    """컷오프 이하인 월말 중 가장 늦은 기준일. 없으면 ValueError."""
    ok = [d for d in months if d <= cutoff]
    if not ok:
        raise ValueError(f"컷오프 {cutoff} 이하 DIS 월말 파일이 없습니다")
    return max(ok)


def load_month(raw_dir: str | Path, basis: date) -> dict[str, dict]:
    p = Path(raw_dir) / f"{basis:%Y%m%d}.xml.gz"
    text = gzip.open(p, "rt", encoding="utf-8").read()
    return first_per_code(parse_dis_xml(text))


def load_code_map(map_csv: str | Path, manual_csv: str | Path) -> tuple[pl.DataFrame, set[str]]:
    """(확정 대응 short_code, fund_code, level) 표와, 대응표에 이름이 오른 모든 단축코드 집합.

    확정 = confidence ``high`` + 수동 확인(E3와 같은 규칙). 높음이 우선이고 단축코드마다 한 행이다.
    """
    m = pl.read_csv(str(map_csv), infer_schema_length=0)
    hi = (
        m.filter(pl.col("confidence") == "high")
        .select("short_code", "fund_code")
        .with_columns(pl.lit("high").alias("level"))
    )
    man = (
        pl.read_csv(str(manual_csv), infer_schema_length=0)
        .select("short_code", "fund_code")
        .with_columns(pl.lit("manual").alias("level"))
    )
    out = pl.concat([hi, man]).unique(subset=["short_code"], keep="first", maintain_order=True)
    if out["fund_code"].n_unique() != out.height:
        raise ValueError("대응표가 1:1이 아닙니다: 같은 표준코드가 단축코드 둘 이상에 붙었습니다")
    known = {x for x in m["short_code"].to_list() if x}
    return out, known


def classify(
    short_code: str,
    confirmed: dict[str, str],
    known: set[str],
    month: dict[str, dict],
) -> tuple[str, float | None, float | None]:
    """(상태, 합계 보수 %, TER %)."""
    fc = confirmed.get(short_code)
    if fc is None:
        return (ST_UNCONFIRMED if short_code in known else ST_NOT_IN_MAP), None, None
    row = month.get(fc)
    if row is None or (row["sum_fee"] is None and row["ter"] is None):
        return ST_NOT_IN_MONTH, None, None
    return ST_OK, row["sum_fee"], row["ter"]


def fee_table(
    short_codes: list[str],
    confirmed: dict[str, str],
    known: set[str],
    month: dict[str, dict],
    basis: date,
) -> pl.DataFrame:
    """단축코드 목록 → isu_cd + FEE_COLUMNS."""
    recs = []
    for sc in short_codes:
        st, a, t = classify(sc, confirmed, known, month)
        recs.append({"isu_cd": sc, "fee_total_pct": a, "ter_pct": t, "fee_basis_date": basis, "fee_status": st})
    return pl.DataFrame(
        recs,
        schema={"isu_cd": pl.String, "fee_total_pct": pl.Float64, "ter_pct": pl.Float64,
                "fee_basis_date": pl.Date, "fee_status": pl.String},
    )  # fmt: skip


def attach_fees(cards: pl.DataFrame, fees: pl.DataFrame) -> pl.DataFrame:
    """카드 뒤에 FEE_COLUMNS를 붙인다. 행 순서·기존 칸은 그대로. 코드마다 한 행이어야 한다."""
    if fees["isu_cd"].n_unique() != fees.height:
        raise ValueError("fees 표에 같은 isu_cd가 둘 이상입니다")
    n = cards.height
    out = cards.join(fees.select(["isu_cd", *FEE_COLUMNS]), on="isu_cd", how="left", maintain_order="left")
    if out.height != n or out["fee_status"].null_count():
        raise ValueError("수수료 표를 붙이는 중 행이 늘거나 빠졌습니다")
    return out


def summarize_status(cards: pl.DataFrame) -> dict:
    rows = cards["fee_status"].value_counts().to_dicts()
    etfs = cards.unique(subset=["isu_cd"])["fee_status"].value_counts().to_dicts()
    return {
        "rows": {r["fee_status"]: r["count"] for r in rows},
        "unique_etfs": {r["fee_status"]: r["count"] for r in etfs},
    }
