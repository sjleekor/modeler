"""자료 읽기: ETF 일봉·거래일·분배금·CD91 현금 계정 (09b I1~I4).

경로는 환경변수 ``STOCK_DATA_ROOT``로 받는다. 없으면 이 저장소 위쪽에서 ``stock_data``
형제 디렉터리를 찾는다(기본값만 ../stock_data 기준).
"""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

SPLIT_LO, SPLIT_HI = 0.6, 1.6
SHARES_MATCH_LOG_TOL = 0.2  # 가격비 x 상장좌수비가 1에서 이 로그 거리 안이면 "비슷한 배수"
JUMP_LIMIT = 0.25

ETF_DIR = "kr/output/etf_krx_research_pull_20261009"
ETF_FILE = "etf_daily_2010_20261008.csv.gz"
MANIFEST_FILE = "manifest_files.tsv"
DIST_FILE = "kr/output/etf_seibro_dist_research_copy_20261011/seibro_etf_dist_all.json"
MAP_FILE = "kr/output/etf_dis_fee_code_map_20261011/etf_dis_fee_code_map.csv"
CD91_SNAPSHOT = "2026-09-30"
CD91_SERIES = "rate_kr_cd91"
OUTPUT_DIR = "kr/output/etf_baseline_20261018"


class InputError(RuntimeError):
    """멈추고 보고해야 하는 자료 문제."""


def stock_data_root() -> Path:
    env = os.environ.get("STOCK_DATA_ROOT")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p.parent / "stock_data").is_dir():
            return p.parent / "stock_data"
    raise InputError("STOCK_DATA_ROOT를 지정하십시오 (stock_data 디렉터리를 못 찾았습니다)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- 거래일
def load_trading_days(manifest_tsv: Path) -> list[date]:
    """``day_kind=trading``인 날짜 (파일 이름 ``etf_YYYYMMDD.json.gz``에서)."""
    m = pl.read_csv(manifest_tsv, separator="\t", infer_schema_length=0)
    m = m.filter(pl.col("day_kind") == "trading")
    days = [
        datetime.strptime(f.split("_")[1].split(".")[0], "%Y%m%d").date()
        for f in m["file"].to_list()
    ]
    days = sorted(set(days))
    return days


# --------------------------------------------------------------------------- 일봉
def _num(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.replace_all(",", "")
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(col)
    )


def load_etf_frames(csv_gz: Path, codes: list[str]) -> dict[str, pl.DataFrame]:
    """코드별 ``date, close, shares`` (날짜 오름차순, 종가가 있는 행만)."""
    raw = pl.read_csv(
        csv_gz,
        columns=["BAS_DD", "ISU_CD", "TDD_CLSPRC", "LIST_SHRS"],
        infer_schema_length=0,
    ).filter(pl.col("ISU_CD").is_in(codes))
    df = raw.with_columns(
        pl.col("BAS_DD").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        _num("TDD_CLSPRC"),
        _num("LIST_SHRS"),
    ).rename({"TDD_CLSPRC": "close", "LIST_SHRS": "shares", "ISU_CD": "code"})
    out = {}
    for c in codes:
        s = df.filter(pl.col("code") == c).select("date", "close", "shares").sort("date")
        if s["date"].n_unique() != s.height:
            raise InputError(f"{c}: 같은 날짜가 두 번 이상 있습니다")
        out[c] = s.filter(pl.col("close").is_not_null() & (pl.col("close") > 0))
    return out


# --------------------------------------------------------------------------- 분배금
@dataclass
class DistInfo:
    #: 코드별 연도별 이익분배 건수
    counts: dict[str, dict[int, int]] = field(default_factory=dict)
    rows: pl.DataFrame | None = None  # code, record_date, amount
    isin_by_code: dict[str, str] = field(default_factory=dict)
    duplicates_dropped: int = 0
    bad_amount_rows: int = 0
    liquidation_rows_ignored: int = 0
    record_date_field: str = "RGT_STD_DT"


def load_distributions(dist_json: Path, map_csv: Path, codes: list[str]) -> DistInfo:
    m = pl.read_csv(map_csv, infer_schema_length=0)
    isin_by_code = {}
    for c in codes:
        r = m.filter(pl.col("short_code") == c)
        if r.height != 1:
            raise InputError(f"{c}: 분배금 대응표에서 ISIN을 못 찾았습니다 (행 {r.height})")
        isin_by_code[c] = r["isin"][0]
    code_by_isin = {v: k for k, v in isin_by_code.items()}
    data = json.loads(Path(dist_json).read_text())
    info = DistInfo(isin_by_code=isin_by_code)
    rows, seen = [], set()
    for x in data:
        c = code_by_isin.get(x.get("ISIN"))
        if c is None:
            continue
        if x.get("RGT_RSN_DTAIL_NM") != "이익분배":
            info.liquidation_rows_ignored += 1
            continue
        key = (c, x["RGT_STD_DT"], x.get("ESTM_STDPRC"), x.get("TH1_PAY_TERM_BEGIN_DT"))
        if key in seen:
            info.duplicates_dropped += 1
            continue
        seen.add(key)
        try:
            amt = float(str(x["ESTM_STDPRC"]).replace(",", ""))
        except (TypeError, ValueError):
            info.bad_amount_rows += 1
            continue
        rows.append((c, datetime.strptime(x["RGT_STD_DT"], "%Y%m%d").date(), amt))
    df = pl.DataFrame(
        rows,
        schema={"code": pl.String, "record_date": pl.Date, "amount": pl.Float64},
        orient="row",
    ).sort("code", "record_date")
    info.rows = df
    for c in codes:
        s = df.filter(pl.col("code") == c)
        cnt: dict[int, int] = {}
        for d in s["record_date"].to_list():
            cnt[d.year] = cnt.get(d.year, 0) + 1
        info.counts[c] = dict(sorted(cnt.items()))
    return info


# --------------------------------------------------------------------------- 수익
@dataclass
class ReturnPanel:
    days: list[date]
    codes: list[str]
    #: 총수익(분배금 포함, 분할 보정). 상장 전은 NaN.
    ret: dict[str, np.ndarray]
    first_date: dict[str, date]
    last_date: dict[str, date]
    missing_days: dict[str, list[date]]  # 상장 뒤 거래일인데 종가가 없는 날
    split_candidates: list[dict]  # I2 분할·병합 후보(보정에 쓴 것)
    jumps: list[dict]  # 보정 뒤에도 |r| > 25%인 날
    dist_unplaced: list[dict]  # 직전 거래일이 없어 못 넣은 분배금
    dist_added: dict[str, int]


def build_return_panel(
    days: list[date],
    frames: dict[str, pl.DataFrame],
    dist: pl.DataFrame | None,
) -> ReturnPanel:
    """거래일 달력에 맞춘 총수익. 분배금은 기준일 직전 거래일의 수익에 더한다 (I3)."""
    n = len(days)
    pos = {d: i for i, d in enumerate(days)}
    ret, first, last, missing = {}, {}, {}, {}
    splits: list[dict] = []
    jumps: list[dict] = []
    unplaced: list[dict] = []
    added: dict[str, int] = {}
    for c, f in frames.items():
        close = np.full(n, np.nan)
        shr = np.full(n, np.nan)
        for d, p, s in zip(f["date"].to_list(), f["close"].to_list(), f["shares"].to_list()):
            if d in pos:
                close[pos[d]] = p
                shr[pos[d]] = s if s is not None else np.nan
        have = np.where(~np.isnan(close))[0]
        if len(have) == 0:
            raise InputError(f"{c}: 거래일에 종가가 하나도 없습니다")
        i0, i1 = int(have[0]), int(have[-1])
        first[c], last[c] = days[i0], days[i1]
        miss = [days[i] for i in range(i0, n) if np.isnan(close[i])]
        missing[c] = miss
        div = np.zeros(n)
        added[c] = 0
        if dist is not None:
            sub = dist.filter(pl.col("code") == c)
            for rd, amt in zip(sub["record_date"].to_list(), sub["amount"].to_list()):
                k = bisect_left(days, rd) - 1  # 기준일 직전 거래일
                if k < i0 + 1 or k > i1:
                    unplaced.append({"code": c, "record_date": rd, "amount": amt})
                    continue
                div[k] += amt
                added[c] += 1
        r = np.full(n, np.nan)
        for i in range(i0 + 1, n):
            if np.isnan(close[i]) or np.isnan(close[i - 1]):
                continue
            prev = close[i - 1]
            ratio = close[i] / prev
            adj = 1.0
            if ratio < SPLIT_LO or ratio > SPLIT_HI:
                s0, s1 = shr[i - 1], shr[i]
                if not (np.isnan(s0) or np.isnan(s1)) and s0 > 0:
                    sr = s1 / s0
                    if abs(np.log(sr * ratio)) < SHARES_MATCH_LOG_TOL and abs(np.log(sr)) > 0.3:
                        adj = sr
                        splits.append(
                            {"code": c, "date": days[i], "price_ratio": ratio, "shares_ratio": sr}
                        )
            r[i] = (close[i] + div[i]) / (prev / adj) - 1.0
            if abs(r[i]) > JUMP_LIMIT:
                jumps.append({"code": c, "date": days[i], "ret": float(r[i]), "price_ratio": ratio})
        ret[c] = r
    return ReturnPanel(
        days, list(frames), ret, first, last, missing, splits, jumps, unplaced, added
    )


# --------------------------------------------------------------------------- 현금
def cd91_files(root: Path) -> list[Path]:
    base = (
        root
        / "kr/raw/raw_postgres"
        / f"snapshot_date={CD91_SNAPSHOT}"
        / "source=sj2_remote/common_feature_observation_raw"
    )
    return sorted(base.rglob("*.parquet"))


def load_cash_account(root: Path, sessions: list[date]):
    """MRS와 같은 경로: ``KrLake`` + ``load_kr_rates`` + ``build_cash_account``."""
    from modeler.etl.config import DataRoot
    from modeler.scores.common.cash import build_cash_account
    from modeler.scores.common.kr_inputs import COMMON_OBS_TABLE, KrLake, load_kr_rates

    lake = KrLake.resolve(
        DataRoot(base=root / "kr"), snapshot_date=CD91_SNAPSHOT, tables=(COMMON_OBS_TABLE,)
    )
    rates = load_kr_rates(lake, CD91_SERIES, sessions)
    if rates is None:
        raise InputError("rate_kr_cd91이 없습니다")
    return build_cash_account(rates, sessions, series_id=CD91_SERIES)


def daily_cash_returns(acct, n: int) -> np.ndarray:
    """세션 i의 값 = (i-1 -> i) 현금 수익. 불가(null) 구간과 i=0은 NaN."""
    out = np.full(n, np.nan)
    if n > 1:
        ret, _code = acct.interval_return(np.arange(0, n - 1), np.arange(1, n))
        out[1:] = ret
    return out
