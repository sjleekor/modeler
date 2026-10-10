"""회사 점수(부분 C) 공통 모듈 W0 (사전등록 20261010_quality_score §5.1·§5.3·§5.4·§7.2).

경로, 구간 상수, 기준일, 거래일 달력, 접수 목록 기반 가용일, 판정 구간 보호, 해시를 둔다.
결과(부실 사건)·점수는 다루지 않는다. 입력 존재·결측 세기만 한다.

사전등록이 정하지 않은 선택은 상수로 두고 ``# CI-<이름>`` 주석을 달았다.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------- 구간 (§5.4·§5.3)
FIRST_FIN_YEAR = 2015
DEV_YEARS = (2017, 2018)
JUDGMENT_YEARS = tuple(range(2019, 2025))
O4_JUDGMENT_YEARS = (2019, 2020, 2021, 2022)
O1_FALLBACK_YEARS = (2020, 2022, 2024)

# CI-avail-missing: 접수 목록에 없는 rcept_no. 사전등록에 없다. 기본안은 "가용하지 않음(결측)".
CI_AVAIL_MISSING_RECEIPT = "exclude"  # CI-avail-missing

# CI-market: 달력은 KOSPI 기준(07 §3 시험과 같다).
CI_CALENDAR_MARKET = "KOSPI"  # CI-market

# CI-dup-receipt: 한 rcept_no에 행이 여럿이면 rcept_dt가 가장 늦은 행 하나(보수적),
# 같으면 raw_id가 큰 쪽.
CI_DUP_RECEIPT_RULE = "latest_rcept_dt"  # CI-dup-receipt

# §7.2 판정 구간 보호
JUDGMENT_ENV = "QUALITY_C_JUDGMENT_CONFIRMED"
_GUARDED_PURPOSES = ("outcome", "link")
_OPEN_PURPOSES = ("inputs",)
_JUDGMENT_FROM_YEAR = 2019  # 형성 FY2019부터 판정 구간


# ---------------------------------------------------------------- 경로
def stock_data_root() -> Path:
    """환경변수 ``STOCK_DATA_ROOT``(기본 ``../stock_data``)."""
    return Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))


def my_root() -> Path:
    """환경변수 ``MY_ROOT``(기본 ``../my``) — 문서 저장소.

    ETF 쪽 ``etf_panel.my_root`` 와 같은 관례.
    """
    return Path(os.environ.get("MY_ROOT", "../my"))


PREREG_REL = "milestones/common/scores/20261010_quality_score/01_preregistration.md"


def default_prereg() -> Path:
    """사전등록 문서 경로. 환경변수 ``QUALITY_PREREG``, 없으면 ``my_root()/PREREG_REL``."""
    v = os.environ.get("QUALITY_PREREG")
    return Path(v) if v else my_root() / PREREG_REL


@dataclass(frozen=True)
class Lake:
    """raw·파생 snapshot 날짜를 따로 받는다. 판정은 raw 2026-10-18로 돈다."""

    root: Path
    raw_snapshot: str = "2026-09-30"
    derived_snapshot: str = "2026-09-29"

    @classmethod
    def from_env(cls, **kw: str) -> Lake:
        return cls(root=stock_data_root(), **kw)

    def raw_dir(self, table: str) -> Path:
        return (
            Path(self.root)
            / "kr/raw/raw_postgres"
            / f"snapshot_date={self.raw_snapshot}"
            / "source=sj2_remote"
            / table
        )

    def derived_dir(self, table: str) -> Path:
        return (
            Path(self.root)
            / "kr/derived/feature"
            / f"snapshot_date={self.derived_snapshot}"
            / "source=sj2_remote"
            / table
        )

    @staticmethod
    def _glob(d: Path) -> str:
        if not d.is_dir():
            raise FileNotFoundError(
                f"레이크 경로가 없습니다: {d}. STOCK_DATA_ROOT와 snapshot 날짜를 확인하십시오."
            )
        if not any(d.rglob("*.parquet")):
            raise FileNotFoundError(f"parquet 파일이 없습니다: {d}")
        return str(d / "**" / "*.parquet")

    def raw_glob(self, table: str) -> str:
        return self._glob(self.raw_dir(table))

    def derived_glob(self, table: str) -> str:
        """파티션이 없는 표(달력)도, 하위 파티션이 있는 표도 ``**/*.parquet`` 로 읽힌다."""
        return self._glob(self.derived_dir(table))

    def output_dir(self, name: str) -> Path:
        return Path(self.root) / "kr/output" / name


# ---------------------------------------------------------------- 기준일 (§5.1)
def base_date(fy: int) -> date:
    """형성 사업연도 t의 기준일 B_t = (t+1)년 6월 30일."""
    return date(fy + 1, 6, 30)


# ---------------------------------------------------------------- 거래일 달력
class TradingCalendar:
    """정렬된 거래일 배열. 다음 거래일은 d보다 엄격히 뒤인 첫 거래일이다."""

    def __init__(self, days: np.ndarray):
        self._days = np.unique(np.asarray(days, dtype="datetime64[D]"))
        if self._days.size == 0:
            raise ValueError("거래일 달력이 비어 있습니다.")

    @classmethod
    def from_lake(cls, lake: Lake, market: str = CI_CALENDAR_MARKET) -> TradingCalendar:
        df = (
            pl.scan_parquet(lake.derived_glob("dim_trading_calendar"))
            .filter(pl.col("market") == market)
            .select("trade_date")
            .collect()
        )
        if df.height == 0:
            raise ValueError(f"달력에 market={market} 행이 없습니다.")
        return cls(df["trade_date"].to_numpy())

    @property
    def last_day(self) -> date:
        return self._days[-1].astype("datetime64[D]").item()

    def next_trading_day(self, d: date) -> date | None:
        """d보다 엄격히 뒤인 첫 거래일. 달력 끝을 넘으면 None."""
        i = int(np.searchsorted(self._days, np.datetime64(d, "D"), side="right"))
        return None if i >= self._days.size else self._days[i].astype("datetime64[D]").item()

    def next_trading_days(self, s: pl.Series) -> pl.Series:
        """벡터 판. null 입력·달력 끝을 넘는 값은 null."""
        name = s.name
        valid = s.is_not_null()
        arr = s.drop_nulls().to_numpy().astype("datetime64[D]")
        idx = np.searchsorted(self._days, arr, side="right")
        ok = idx < self._days.size
        out = np.full(arr.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
        out[ok] = self._days[idx[ok]]
        res = pl.Series(name, out, dtype=pl.Date)
        if valid.all():
            return res
        full = pl.Series(name, [None] * len(s), dtype=pl.Date)
        return full.scatter(valid.arg_true(), res)


# ---------------------------------------------------------------- 가용일 (§5.1)
def filing_availability(lake: Lake, cal: TradingCalendar) -> pl.DataFrame:
    """접수 목록 → [rcept_no, corp_code, rcept_dt, rcept_no_date, avail_date].

    avail_date = rcept_dt의 다음 거래일("가용일 한 규칙"). rcept_no_date = 접수번호 앞 8자리.
    한 rcept_no에 행이 여럿이면 ``CI_DUP_RECEIPT_RULE`` 로 하나만 남긴다
    (raw 09-30 기준 중복 0건).
    """
    df = (
        pl.scan_parquet(lake.raw_glob("dart_filing_receipt_raw"))
        .select("raw_id", "rcept_no", "corp_code", "rcept_dt")
        .collect()
    )
    df = df.sort(["rcept_no", "rcept_dt", "raw_id"], nulls_last=False).unique(
        subset="rcept_no", keep="last", maintain_order=True
    )
    df = df.with_columns(
        pl.col("rcept_no")
        .str.slice(0, 8)
        .str.to_date("%Y%m%d", strict=False)
        .alias("rcept_no_date")
    )
    avail = cal.next_trading_days(df["rcept_dt"])
    return df.with_columns(avail.alias("avail_date")).select(
        "rcept_no", "corp_code", "rcept_dt", "rcept_no_date", "avail_date"
    )


def count_duplicate_receipts(lake: Lake) -> tuple[int, int]:
    """(전체 행 수, 중복으로 줄어든 행 수). 보고용."""
    df = pl.scan_parquet(lake.raw_glob("dart_filing_receipt_raw")).select("rcept_no").collect()
    return df.height, df.height - df["rcept_no"].n_unique()


def is_available(avail_date: date | None, fy: int) -> bool:
    """avail_date ≤ base_date(fy). avail_date가 없으면 False."""
    return avail_date is not None and avail_date <= base_date(fy)


# 상태값
STATUS_OK = "ok"
STATUS_AFTER_BASE = "after_base"  # 접수는 됐으나 가용일이 기준일 뒤
STATUS_MISSING_RECEIPT = "missing_receipt"  # 접수 목록에 없음
STATUS_NO_AVAIL = "no_avail_date"  # 접수일은 있으나 달력 끝을 넘어 가용일 없음


def available_on_or_before(rcept_nos: pl.Series, fy: int, avail: pl.DataFrame) -> pl.DataFrame:
    """접수번호 목록의 가용 여부. [rcept_no, avail_date, status, available].

    "목록에 없음"은 status로 따로 표시한다. ``CI_AVAIL_MISSING_RECEIPT == "exclude"`` 이면
    available=False(결측)이다.
    """
    left = pl.DataFrame({"rcept_no": rcept_nos})
    # 목록에 있는지는 avail 쪽 표시 열로 본다(avail_date null과 구분). 조인 한 번으로 해서
    # 행 순서에 기대지 않는다.
    j = left.join(
        avail.select("rcept_no", "avail_date").with_columns(pl.lit(True).alias("_in")),
        on="rcept_no",
        how="left",
        maintain_order="left",
    )
    b = base_date(fy)
    status = (
        pl.when(pl.col("_in").is_null())
        .then(pl.lit(STATUS_MISSING_RECEIPT))
        .when(pl.col("avail_date").is_null())
        .then(pl.lit(STATUS_NO_AVAIL))
        .when(pl.col("avail_date") <= b)
        .then(pl.lit(STATUS_OK))
        .otherwise(pl.lit(STATUS_AFTER_BASE))
    )
    j = j.with_columns(status.alias("status")).drop("_in")
    ok = pl.col("status") == STATUS_OK
    if CI_AVAIL_MISSING_RECEIPT != "exclude":
        raise NotImplementedError("CI-avail-missing은 'exclude'만 구현했습니다.")
    return j.with_columns(ok.alias("available"))


def availability_mismatch_counts(lake: Lake, cal: TradingCalendar) -> dict[str, int]:
    """vintage ``available_from`` 과 rcept_dt 다음 거래일이 다른 접수번호 수 (07 §3, 입력 존재).

    키 설명:
    - receipt_rows / receipt_date_ne_rcept_dt: 접수 목록 전체와 접수번호 날짜 ≠ rcept_dt 행 수.
    - annual_receipts / annual_date_ne_rcept_dt: report_nm에 '사업보고서'가 든 접수 행 수와 그중
      접수번호 날짜 ≠ rcept_dt (사전등록의 "사업보고서 369/40,533"에 해당하는 기준 후보).
    - vintage_rcept / vintage_mismatch: vintage 고유 접수번호 중 접수 목록에 있는 수와
      available_from ≠ 다음 거래일(rcept_dt)인 수 (07의 "423건" 기준).
    - vintage_annual_rcept / vintage_annual_mismatch: 위에서 reprt_code='11011'만.
    - vintage_not_in_receipts: vintage 접수번호 중 접수 목록에 없는 수.
    """
    rec = (
        pl.scan_parquet(lake.raw_glob("dart_filing_receipt_raw"))
        .select("rcept_no", "report_nm", "rcept_dt")
        .collect()
    )
    rec = rec.with_columns(
        pl.col("rcept_no").str.slice(0, 8).str.to_date("%Y%m%d", strict=False).alias("nd")
    )
    ne = pl.col("nd") != pl.col("rcept_dt")
    annual = rec.filter(pl.col("report_nm").str.contains("사업보고서"))
    vint = (
        pl.scan_parquet(lake.derived_glob("stock_metric_vintage_fact"))
        .select("rcept_no", "reprt_code", "available_from")
        .unique()
        .collect()
    )
    v = vint.group_by("rcept_no").agg(
        pl.col("available_from").min().alias("af"),
        (pl.col("reprt_code") == "11011").any().alias("is_annual"),
    )
    vj = v.join(rec.select("rcept_no", "rcept_dt"), on="rcept_no", how="left")
    in_rec = vj.filter(pl.col("rcept_dt").is_not_null())
    nxt = cal.next_trading_days(in_rec["rcept_dt"])
    in_rec = in_rec.with_columns(nxt.alias("nxt")).with_columns(
        (pl.col("af") != pl.col("nxt")).alias("mm")
    )
    ann = in_rec.filter(pl.col("is_annual"))
    return {
        "receipt_rows": rec.height,
        "receipt_date_ne_rcept_dt": int(rec.filter(ne).height),
        "annual_receipts": annual.height,
        "annual_date_ne_rcept_dt": int(annual.filter(ne).height),
        "vintage_rcept": in_rec.height,
        "vintage_mismatch": int(in_rec["mm"].sum()),
        "vintage_annual_rcept": ann.height,
        "vintage_annual_mismatch": int(ann["mm"].sum()),
        "vintage_not_in_receipts": v.height - in_rec.height,
    }


# ---------------------------------------------------------------- 판정 구간 보호 (§7.2)
def guard_years(years, purpose: str) -> None:
    """결과 변수 계산·점수-결과 연결에서 형성 FY2019 이상을 쓰려면 확인 환경변수가 있어야 한다.

    purpose="inputs"(입력 존재·결측 세기)는 항상 허용한다. 알 수 없는 purpose는 막는다.
    """
    if purpose in _OPEN_PURPOSES:
        return
    if purpose not in _GUARDED_PURPOSES:
        raise ValueError(f"알 수 없는 purpose입니다: {purpose!r} (inputs·outcome·link 중 하나)")
    judged = sorted(y for y in years if y >= _JUDGMENT_FROM_YEAR)
    if judged and not os.environ.get(JUDGMENT_ENV):
        raise PermissionError(
            f"판정 구간(형성 FY{_JUDGMENT_FROM_YEAR} 이상) {judged}을(를) '{purpose}' 용도로 "
            f"쓰려면 사전등록 동결 확인 뒤 환경변수 {JUDGMENT_ENV}를 설정해야 합니다 (§7.2)."
        )


# ---------------------------------------------------------------- 해시
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(module_file: str | Path) -> str:
    """모듈 소스 파일(``__file__``)의 sha256."""
    return sha256_file(module_file)


def git_head(repo_dir: str | Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
