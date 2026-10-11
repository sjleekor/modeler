"""ETF E3 보유 비용 — 입력 읽기 (사양 e3_v2_spec §1·§5, 동결 전 구현).

네 가지 입력을 읽는다. 값을 계산하거나 점수와 잇지 않는다.

- KRX 일별 CSV(``etf_panel.read_panel``)와 동결 분류 파서(``etf_classify``)는 그대로 쓴다.
- DIS 월말 원문 XML(``raw/YYYYMMDD.xml.gz``): 행 = ``<tmpV1>``\\~``<tmpV16>`` 묶음.
  tmpV1 운용사, tmpV9 합계 보수, tmpV12 TER(연 %), tmpV15 펀드 표준코드, tmpV16 매매중개수수료율.
  자기닫힘 태그(``<tmpV17 />``)와 빈 태그는 값 없음.
- DIS 표준코드 → KRX 단축코드 대응표(높음) + 확인 목록 20행.
- SEIBro 분배금(청산분배 제외). ``ISIN[3:9]`` = KRX 단축코드.

경로는 ``STOCK_DATA_ROOT`` 환경변수(기본 ``../stock_data``)에서 받는다. 입력 sha256은 manifest의
값과 맞는지 확인하고 실행 manifest에 남긴다(``verify_inputs``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import polars as pl

E3_VERSION = "etf-holding-cost-e3/1"

# ---------------------------------------------------------------- 경로(상대 경로, 기본값만 ../stock_data)
KRX_REL = "kr/output/etf_krx_research_pull_20261009/etf_daily_2010_20261008.csv.gz"
DIS_DIR_REL = "kr/output/etf_dis_fee_research_pull_20261011"
DIS_RAW_REL = DIS_DIR_REL + "/raw"
DIS_SUMS_REL = DIS_DIR_REL + "/SHA256SUMS"
CODE_MAP_REL = "kr/output/etf_dis_fee_code_map_20261011/etf_dis_fee_code_map.csv"
MANUAL_REL = "kr/output/etf_holding_cost_e3_inputs_20261011/dis_map_manual_confirm.csv"
SEIBRO_REL = "kr/output/etf_seibro_dist_research_copy_20261011/seibro_etf_dist_all.json"

# manifest에 적힌 기대 sha256(사양 §1, 각 manifest.json). 다르면 거절한다.
EXPECTED_SHA256 = {
    KRX_REL: "3495e91ec5ed268e1199a290d7139ac8a6fbe83e8ce1a76374311eac9ed274ed",
    CODE_MAP_REL: "f49bd8676f5fff042f1e905d20442c8aac21004ef0d18140a348944723dde2b3",
    MANUAL_REL: "87e4b3b609e7f64f0619e491dc60f4a340ca9fc636419eff57b735045201adf9",
    SEIBRO_REL: "f49e47ab5ef7fb15419297928a6e8a48d3c0715b0f3bf3a19046042384d2c7a1",
    DIS_SUMS_REL: "f0132304def6f16bba36e08aa4fce424d5b7d37b8e6314003de4dbd52345ca45",
}

DIS_FIELDS = {
    "operator": 1,
    "sum_fee": 9,
    "ter": 12,
    "fund_code": 15,
    "brokerage": 16,
}
LIQUIDATION = "청산분배"


def stock_data_root() -> Path:
    return Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_inputs(root: Path | None = None, check_dis_raw: bool = True) -> dict:
    """입력 파일 sha256을 계산해 기대값과 맞추고 {상대경로: sha256}을 돌려준다. 다르면 ValueError."""
    root = root or stock_data_root()
    out: dict[str, str] = {}
    for rel, want in EXPECTED_SHA256.items():
        got = sha256_file(root / rel)
        if got != want:
            raise ValueError(f"sha256 불일치: {rel} 기대 {want[:8]}… 실제 {got[:8]}…")
        out[rel] = got
    if check_dis_raw:
        sums = {}
        for line in (root / DIS_SUMS_REL).read_text().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                sums[name.strip()] = h
        bad = []
        for name, h in sums.items():
            if sha256_file(root / DIS_DIR_REL / name) != h:
                bad.append(name)
        if bad:
            raise ValueError(f"DIS raw sha256 불일치 {len(bad)}개: {bad[:3]}")
        out["dis_raw_files_verified"] = str(len(sums))
    return out


# ---------------------------------------------------------------- DIS XML
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
    """DIS XML 문자열 하나에서 행을 읽는다. 자기닫힘 태그는 값 없음(None).

    ``tmpV1``(운용사)과 ``tmpV15``(표준코드)는 문자열, 나머지는 숫자(없으면 None).
    ``<tmpV15>`` 가 없는 행은 버린다(``n_rows_no_code`` 로 따로 세지 않는다 — 실측에서 0).
    """
    rows = []
    for m in _RE_ROW.finditer(text):
        tags = {int(k): v for k, v in _RE_TAG.findall(m.group(1))}
        code = (tags.get(DIS_FIELDS["fund_code"]) or "").strip()
        if not code:
            continue
        rows.append(
            {
                "fund_code": code,
                "operator": (tags.get(DIS_FIELDS["operator"]) or "").strip() or None,
                "sum_fee": _num(tags.get(DIS_FIELDS["sum_fee"])),
                "ter": _num(tags.get(DIS_FIELDS["ter"])),
                "brokerage": _num(tags.get(DIS_FIELDS["brokerage"])),
            }
        )
    return rows


def load_dis_fees(raw_dir: str | Path) -> pl.DataFrame:
    """월말 파일 전부 → (base_date, fund_code, operator, sum_fee, ter, brokerage). 같은 달 같은 코드 중복은 첫 행."""
    recs = []
    for p in sorted(Path(raw_dir).glob("*.xml.gz")):
        d = datetime.strptime(p.name.split(".")[0], "%Y%m%d").date()
        text = gzip.open(p, "rt", encoding="utf-8").read()
        for r in parse_dis_xml(text):
            r["base_date"] = d
            recs.append(r)
    df = pl.DataFrame(
        recs,
        schema={
            "fund_code": pl.String,
            "operator": pl.String,
            "sum_fee": pl.Float64,
            "ter": pl.Float64,
            "brokerage": pl.Float64,
            "base_date": pl.Date,
        },
    )
    return df.unique(subset=["base_date", "fund_code"], keep="first", maintain_order=True).sort(
        ["fund_code", "base_date"]
    )


# ---------------------------------------------------------------- 대응표
def load_code_map(map_csv: str | Path, manual_csv: str | Path) -> pl.DataFrame:
    """DIS 표준코드 ↔ KRX 단축코드. 높음(confidence=high) + 확인 목록. 칸: short_code, fund_code, level.

    level: ``high``(대응표 높음) | ``manual``(확인 목록). 둘 다이면 ``high``. 단축코드·표준코드는 1:1 이어야 한다.
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
        raise ValueError("대응표가 1:1이 아니다: 같은 표준코드가 단축코드 둘 이상에 붙었다")
    return out


# ---------------------------------------------------------------- SEIBro
def load_seibro(path: str | Path) -> tuple[pl.DataFrame, dict]:
    """분배금 표. 칸: isu_cd(= ISIN[3:9]), rgt_std_dt(Date), amount(주당, 원), kind.

    청산분배는 ``liquidation`` 칸이 True 로 남는다(수집 쪽에서 걸러 쓰도록 ``kind`` 를 같이 둔다).
    반환 두 번째는 ``RGT_RSN_DTAIL_NM`` 분류별 개수(개수만).
    """
    raw = json.loads(Path(path).read_text())
    cats: dict[str, int] = {}
    rows = []
    for x in raw:
        k = x.get("RGT_RSN_DTAIL_NM") or ""
        cats[k] = cats.get(k, 0) + 1
        amt = _num(x.get("ESTM_STDPRC"))
        rows.append(
            {
                "isu_cd": x["ISIN"][3:9],
                "rgt_std_dt": datetime.strptime(x["RGT_STD_DT"], "%Y%m%d").date(),
                "amount": amt,
                "kind": k,
                "liquidation": k == LIQUIDATION,
            }
        )
    df = pl.DataFrame(
        rows,
        schema={
            "isu_cd": pl.String,
            "rgt_std_dt": pl.Date,
            "amount": pl.Float64,
            "kind": pl.String,
            "liquidation": pl.Boolean,
        },
    )
    return df, cats


def distributions_for_use(seibro: pl.DataFrame) -> pl.DataFrame:
    """분배 표본: 청산분배 제외, 금액 > 0."""
    return seibro.filter(~pl.col("liquidation") & (pl.col("amount") > 0))


def month_key(d: date) -> tuple[int, int]:
    return d.year, d.month
