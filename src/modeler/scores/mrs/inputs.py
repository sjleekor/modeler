"""시장 국면 점수(MRS) 입력층: 격자, PIT 입력 행, 현금 계정, 실현 가격 배열.

사양 §2·§5.1·§5.2. 점수 엔진(``vintage``·``components``·``score``)은 여기서 만든 값만 받는다.
**여기서는 점수·수익을 계산하지 않는다** — 레이크 행을 ``date, value, available_at``으로 맞추고
격자·현금·실현 배열을 붙일 뿐이다.

입력 행 (PIT rows)
    ``date``(관측일, Date), ``value``(Float64), ``available_at``(Datetime us UTC, tz-aware).
    FRED 계열은 vintage가 여럿일 수 있어 같은 ``date``에 ``available_at``이 다른 행이 여럿이다.
    ECOS 계열은 관측일당 한 행이다.

격자 (Grid)
    * KR: ECOS ``market_kospi_ecos`` 관측일 전부(1995-01-03~, 토요일 세션 포함). 달력은
      ``SessionCalendar.from_observed_dates("XKRX", ...)`` (MI01). 개장 09:00 KST.
    * US: ``exchange_calendars`` XNYS 세션을 1993-01-29(SPY 첫 거래일)부터, 끝은 레이크
      ``trading_calendar``에서 SPY 마지막 가격일 **다음** 세션까지. 레이크 달력과 겹치는 구간은
      날짜·개장·폐장 시각이 같아야 하고, 다르면 ``GridMismatchError`` (MI02).
    결정 시각 = 다음 격자 세션 개장 30분 전(``SessionCalendar.decision_at``). 마지막 격자일은 다음
    세션이 없어 ``None``이다 — 결정에서 뺀다.

KR 가용 시각 (MI29)
    ``kr_inputs._load_series`` + ``_with_available_at`` 그대로다. 저장된 ``available_from_date``의
    08:30 KST를 쓴다. 1995~2000 토요일 세션에서 이 값이 격자의 다음 세션(토요일)보다 늦은 건이
    있다(금요일 값이 월요일 08:30 가용) → 그 금요일 결정은 목요일 값을 쓴다. 보수적 기본안이다.
    건수는 ``readiness``가 센다.

VIX 근사 (MI13, ``sens_vix_proxy``)
    관측일당 **가장 이른 vintage**(최소 ``realtime_start``) 값 하나. ``available_at`` = 관측일
    다음(엄격히 뒤) KR 격자 세션 08:30 KST. ``realtime_start``는 무시한다. KR 격자에 뒤 세션이
    없는 관측일(마지막 격자일 이후)은 어떤 결정에도 못 쓰므로 뺀다(MI44).

실현 가격 배열 (MI45)
    r_bh용이다. 격자일에 **같은 날 관측**이 있으면 그 값, 없으면 NaN. 앞 값을 끌어오지 않는다.
    KR은 ECOS 값, US는 SPY 총수익 지수 ``tr_index``(MI09)다.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.scores.common.calendar import UTC_TS, CalendarError, SessionCalendar
from modeler.scores.common.cash import CashAccount, build_cash_account, load_us_rates
from modeler.scores.common.inputs import PinnedScopedLake, sha256_file
from modeler.scores.common.kr_inputs import (
    COMMON_OBS_TABLE,
    KrLake,
    KrNotSyncedError,
    kr_available_at,
    load_kr_rates,
    load_kr_series,
)
from modeler.scores.common.panel import PRICE_AVAILABILITY_BUFFER
from modeler.scores.common.total_return import load_us_total_return
from modeler.scores.market_sector.features import load_us_macro
from modeler.scores.mrs import config

logger = logging.getLogger(__name__)

#: KR 격자를 정하는 ECOS 계열(KOSPI 지수). 관측일 전부가 격자다 (MI01).
KR_GRID_SERIES = "market_kospi_ecos"
#: KR PIT 행으로 읽는 ECOS 계열 아홉 (사양 §5.1). 첫째가 격자, 둘째가 KOSDAQ 자산 가격(MI30).
KR_PIT_SERIES: tuple[str, ...] = (
    "market_kospi_ecos",
    "market_kosdaq_ecos",
    "trdval_kospi_ecos",
    "trdval_kosdaq_ecos",
    "foreign_net_kospi_ecos",
    "foreign_net_kosdaq_ecos",
    "fx_usdkrw_ecos",
    "rate_kr_gov3y",
    "rate_kr_gov10y",
)
#: 실현 가격 배열 키 -> KR ECOS 계열.
KR_REALIZED_SERIES = {"kr_kospi": "market_kospi_ecos", "kr_kosdaq": "market_kosdaq_ecos"}
US_REALIZED_KEY = "us_spx"

US_MACRO_SERIES: tuple[str, ...] = ("VIXCLS", "BAA10Y", "T10Y2Y")
US_MACRO_TABLE = "macro_series"
#: US 입력이 읽는 표. KR 점수도 ``macro_series``(VIX)는 US 레이크에서 읽는다.
US_TABLES_MRS: tuple[str, ...] = (
    "prices_daily",
    "corp_actions",
    "macro_series",
    "trading_calendar",
)
US_TABLES_FOR_KR: tuple[str, ...] = (US_MACRO_TABLE,)
SPY_SYMBOL = "SPY"
#: US 격자 시작일 = SPY 첫 거래일 (MI02).
US_GRID_START = date(1993, 1, 29)

TR_INDEX_KEY = "tr_index"


class GridMismatchError(ValueError):
    """격자 달력이 레이크 ``trading_calendar``와 겹치는 구간에서 다르다 (MI02)."""


# --------------------------------------------------------------------------- 격자
@dataclass(frozen=True)
class Grid:
    """결정 격자. ``dates``는 오름차순이고 ``decision_at``과 길이가 같다(마지막 칸은 ``None``)."""

    market: str
    dates: tuple[date, ...]
    decision_at: tuple[datetime | None, ...]
    calendar: SessionCalendar

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def last_decidable(self) -> date:
        """결정을 낼 수 있는 마지막 격자일(다음 세션이 있는 마지막 날)."""
        return self.dates[-2]

    def decision_frame(self) -> pl.DataFrame:
        """``date, decision_at``. 마지막 격자일은 뺀다(결정 시각이 없다)."""
        n = len(self.dates) - 1
        return pl.DataFrame(
            {
                "date": list(self.dates[:n]),
                "decision_at": pl.Series(list(self.decision_at[:n]), dtype=UTC_TS),
            }
        )

    def next_session_after(self, d: date) -> date | None:
        """``d`` 다음(엄격히 뒤) 첫 격자 세션. 없으면 ``None``."""
        k = bisect_right(self.dates, d)
        return self.dates[k] if k < len(self.dates) else None


def make_grid(market: str, cal: SessionCalendar, dates: Sequence[date]) -> Grid:
    """``dates``(``cal``의 세션)에서 결정 시각을 붙인다. 마지막 날은 ``None``."""
    ds = tuple(dates)
    if len(ds) < 2:
        raise ValueError("격자는 두 날짜 이상이어야 합니다")
    at: list[datetime | None] = [cal.decision_at(d) for d in ds[:-1]] + [None]
    return Grid(market=market, dates=ds, decision_at=tuple(at), calendar=cal)


def build_kr_grid(lake: KrLake) -> Grid:
    """KR 격자: ECOS ``market_kospi_ecos`` 관측일 전부 (MI01). 시리즈가 없으면 예외."""
    raw = load_kr_series(lake, KR_GRID_SERIES, ())
    if raw is None:
        raise KrNotSyncedError(f"{COMMON_OBS_TABLE}에 {KR_GRID_SERIES} 행이 없습니다")
    dates = sorted(set(raw["date"].to_list()))
    cal = SessionCalendar.from_observed_dates("XKRX", dates)
    return make_grid("KR", cal, dates)


def assert_us_calendars_match(grid_cal: SessionCalendar, lake_cal: SessionCalendar) -> int:
    """겹치는 구간에서 날짜·개장·폐장 시각이 같아야 한다 (MI02). 겹친 세션 수를 돌려준다.

    다르면 ``GridMismatchError`` (처음 다른 날짜를 메시지에 적는다).
    """
    lo = max(grid_cal.sessions[0], lake_cal.sessions[0])
    hi = min(grid_cal.sessions[-1], lake_cal.sessions[-1])
    if lo > hi:
        raise GridMismatchError("격자 달력과 레이크 달력이 겹치지 않습니다")

    def window(cal: SessionCalendar) -> list[tuple[date, datetime, datetime]]:
        return [
            (s, o, c)
            for s, o, c in zip(cal.sessions, cal.opens, cal.closes, strict=True)
            if lo <= s <= hi
        ]

    g, k = window(grid_cal), window(lake_cal)
    if g != k:
        gd, kd = {x[0]: x for x in g}, {x[0]: x for x in k}
        only_g = sorted(set(gd) - set(kd))
        only_k = sorted(set(kd) - set(gd))
        diff = sorted(d for d in set(gd) & set(kd) if gd[d] != kd[d])

        def iso(ds: list[date]) -> list[str]:
            return [d.isoformat() for d in ds[:3]]

        raise GridMismatchError(
            f"US 격자 달력이 레이크 trading_calendar와 다릅니다 [{lo} ~ {hi}]: "
            f"격자에만 {len(only_g)}일 {iso(only_g)}, 레이크에만 {len(only_k)}일 {iso(only_k)}, "
            f"개장·폐장 시각 불일치 {len(diff)}일 {iso(diff)}"
        )
    return len(g)


def last_spy_session(lake: PinnedScopedLake, symbol: str = SPY_SYMBOL) -> date:
    """레이크 ``prices_daily``의 SPY 마지막 가격일(종가가 있는 행만). 날짜 하나만 읽는다."""
    out = (
        lake.scan_raw("prices_daily")
        .filter((pl.col("symbol") == symbol) & (pl.col("close").cast(pl.Float64) > 0))
        .select(pl.col("date").max())
        .collect()
        .item()
    )
    if out is None:
        raise ValueError(f"prices_daily에 {symbol} 가격이 없습니다")
    return out


def build_us_grid(lake: PinnedScopedLake) -> tuple[Grid, SessionCalendar]:
    """US 격자: XNYS 세션 1993-01-29 ~ (레이크 달력에서 SPY 마지막 가격일 다음 세션) (MI02).

    ``(격자, 레이크 달력)``을 돌려준다. 레이크 달력은 SPY 경로의 세션 필터에 쓴다.
    ``exchange_calendars``가 없으면 예외다(격자를 레이크 달력만으로 만들지 않는다 — 1993~2010이
    레이크에 없다).
    """
    lake_cal = SessionCalendar.from_us_lake(lake)
    last = last_spy_session(lake)
    try:
        end = lake_cal.session_at(lake_cal.index_of(last) + 1)
    except CalendarError as exc:
        raise GridMismatchError(
            f"SPY 마지막 가격일 {last}의 다음 세션을 레이크 달력에서 찾지 못했습니다: {exc}"
        ) from exc
    cal = SessionCalendar.from_exchange_calendars("XNYS", US_GRID_START, end)
    if cal is None:
        raise RuntimeError("exchange_calendars가 없어 US 격자를 만들 수 없습니다")
    if cal.sessions[0] != US_GRID_START or cal.sessions[-1] != end:
        raise GridMismatchError(
            f"격자 범위가 어긋났습니다: {cal.sessions[0]}~{cal.sessions[-1]} (기대 "
            f"{US_GRID_START}~{end})"
        )
    n = assert_us_calendars_match(cal, lake_cal)
    logger.info(
        "US 격자 %s ~ %s (%d 세션), 레이크 달력과 %d 세션 일치", cal.sessions[0], end, len(cal), n
    )
    return make_grid("US", cal, cal.sessions), lake_cal


# --------------------------------------------------------------------------- 실현 가격
def realized_on_grid(grid_dates: Sequence[date], rows: pl.DataFrame) -> np.ndarray:
    """격자일의 같은 날 관측 값(``date, value`` 행), 없으면 NaN. 앞 값을 끌어오지 않는다 (MI45)."""
    sub = (
        rows.drop_nulls(["date", "value"])
        .select("date", "value")
        .unique(subset=["date"], keep="last", maintain_order=True)
    )
    g = pl.DataFrame({"date": list(grid_dates)}, schema={"date": pl.Date})
    out = g.join(sub, on="date", how="left")
    return out["value"].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)


# --------------------------------------------------------------------------- 입력 파일
@dataclass(frozen=True)
class InputFile:
    """manifest용 입력 파일 한 줄. ``sha256``은 ``hash_files=False``면 ``None``."""

    table: str
    path: str
    bytes: int
    sha256: str | None


def collect_input_files(
    files_by_table: Mapping[str, Sequence[Path]], *, hash_files: bool = True
) -> tuple[InputFile, ...]:
    out = []
    for table in sorted(files_by_table):
        for p in sorted(files_by_table[table]):
            out.append(
                InputFile(
                    table=table,
                    path=str(p),
                    bytes=p.stat().st_size,
                    sha256=sha256_file(p) if hash_files else None,
                )
            )
    return tuple(out)


# --------------------------------------------------------------------------- 레이크 고르기
def resolve_kr_lake(root: DataRoot, snapshot_date: str | None = None) -> KrLake:
    """``common_feature_observation_raw``가 있는 KR 스냅샷. 안 주면 최신 완전 스냅샷."""
    return KrLake.resolve(root, snapshot_date=snapshot_date, tables=(COMMON_OBS_TABLE,))


def pin_us_lake(
    root: DataRoot,
    tables: Sequence[str] = US_TABLES_MRS,
    *,
    snapshot_date: str | None = None,
    snapshots: Mapping[str, str] | None = None,
    symbols: Sequence[str] = (SPY_SYMBOL,),
) -> PinnedScopedLake:
    """표마다 스냅샷을 못 박은 US 레이크(``build_panel.build_us``와 같은 규칙).

    ``snapshots``에 있는 표는 그 날짜, 아니면 ``snapshot_date``가 있고 그 디렉터리가 있으면 그것,
    아니면 최신이다.
    """
    base = PinnedScopedLake(root=root, snapshots={}, symbols=tuple(symbols))
    snaps: dict[str, str] = {}
    for t in tables:
        if snapshots and t in snapshots:
            snaps[t] = snapshots[t]
        elif (
            snapshot_date
            and (root.derived / "snapshots" / t / f"snapshot_date={snapshot_date}").is_dir()
        ):
            snaps[t] = snapshot_date
        else:
            snaps[t] = base.latest_snapshot(t).isoformat()
    return PinnedScopedLake(root=root, snapshots=snaps, symbols=tuple(symbols))


# --------------------------------------------------------------------------- US 거시·VIX
def load_us_macro_rows(lake: PinnedScopedLake, series_ids: Sequence[str]) -> pl.DataFrame:
    """``series_id, date, realtime_start, value, available_at``(MS0 규칙, 값이 있는 행만)."""
    rows = load_us_macro(lake, tuple(series_ids))
    return rows.drop_nulls(["date", "value", "available_at"]).sort(
        ["series_id", "available_at", "date"]
    )


def pit_rows_from_macro(macro: pl.DataFrame, series_id: str) -> pl.DataFrame:
    """한 계열의 PIT 행 ``date, value, available_at``. 같은 관측일의 vintage를 모두 남긴다."""
    return macro.filter(pl.col("series_id") == series_id).select(
        "date", pl.col("value").cast(pl.Float64), pl.col("available_at").cast(UTC_TS)
    )


def vix_proxy_rows(vix_macro: pl.DataFrame, kr_grid: Grid) -> tuple[pl.DataFrame, int]:
    """``sens_vix_proxy`` VIX 입력 (MI13). ``(행, 뺀 행 수)``.

    관측일당 가장 이른 ``realtime_start`` 행의 값 하나. ``available_at`` = 관측일 다음(엄격히
    뒤) KR 격자 세션 08:30 KST. 뒤 세션이 없는 관측일은 뺀다 (MI44).
    """
    first = (
        vix_macro.filter(pl.col("series_id") == "VIXCLS")
        .drop_nulls(["date", "realtime_start", "value"])
        .sort(["date", "realtime_start"])
        .unique(subset=["date"], keep="first", maintain_order=True)
    )
    dates, vals, ats = [], [], []
    dropped = 0
    for d, v in zip(first["date"].to_list(), first["value"].to_list(), strict=True):
        nxt = kr_grid.next_session_after(d)
        if nxt is None:
            dropped += 1
            continue
        dates.append(d)
        vals.append(float(v))
        ats.append(kr_available_at(nxt))
    out = pl.DataFrame(
        {"date": dates, "value": vals, "available_at": pl.Series(ats, dtype=UTC_TS)},
        schema={"date": pl.Date, "value": pl.Float64, "available_at": UTC_TS},
    )
    return out, dropped


# --------------------------------------------------------------------------- 번들
@dataclass(frozen=True)
class KrInputs:
    """KR 입력 번들. ``series``는 KR PIT 아홉 + (있으면) ``VIXCLS``."""

    grid: Grid
    series: dict[str, pl.DataFrame]
    vix_proxy: pl.DataFrame
    realized: dict[str, np.ndarray]
    cash: CashAccount | None
    #: 표 -> 스냅샷 날짜 (``common_feature_observation_raw``는 KR, ``macro_series``는 US).
    snapshot: dict[str, str]
    input_files: tuple[InputFile, ...]
    #: 레이크에 없는 계열 이름. 실행(run)은 비어 있어야 한다 — 판단은 phase 2가 한다.
    missing_series: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsInputs:
    """US 입력 번들. ``series``는 ``tr_index``·``VIXCLS``·``BAA10Y``·``T10Y2Y``."""

    grid: Grid
    series: dict[str, pl.DataFrame]
    realized: dict[str, np.ndarray]
    cash: CashAccount | None
    snapshot: dict[str, str]
    input_files: tuple[InputFile, ...]
    missing_series: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def load_kr_inputs(
    lake: KrLake,
    us_lake: PinnedScopedLake | None,
    *,
    hash_files: bool = True,
) -> KrInputs:
    """KR PIT 행·VIX(+근사)·CD91 현금·실현 배열을 읽는다.

    ``us_lake``는 ``macro_series``(VIXCLS)용이다. 없으면 VIX가 빠진다(``missing_series``).
    """
    grid = build_kr_grid(lake)
    series: dict[str, pl.DataFrame] = {}
    missing: list[str] = []
    for sid in KR_PIT_SERIES:
        df = load_kr_series(lake, sid, grid.dates)
        if df is None:
            missing.append(sid)
        else:
            series[sid] = df
    diag: dict[str, Any] = {"rows": {k: v.height for k, v in series.items()}}

    snapshot = {COMMON_OBS_TABLE: lake.snapshot_date}
    files = {COMMON_OBS_TABLE: lake.table_files(COMMON_OBS_TABLE)}
    vix_proxy = pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64, "available_at": UTC_TS})
    if us_lake is not None:
        macro = load_us_macro_rows(us_lake, ("VIXCLS",))
        if macro.height:
            series["VIXCLS"] = pit_rows_from_macro(macro, "VIXCLS")
            vix_proxy, dropped = vix_proxy_rows(macro, grid)
            diag["vix_proxy_dropped_no_later_session"] = dropped
        else:
            missing.append("VIXCLS")
        snapshot[US_MACRO_TABLE] = us_lake.latest_snapshot(US_MACRO_TABLE).isoformat()
        files.update(us_lake.input_files((US_MACRO_TABLE,)))
    else:
        missing.append("VIXCLS")

    realized = {
        key: realized_on_grid(grid.dates, series[sid])
        for key, sid in KR_REALIZED_SERIES.items()
        if sid in series
    }
    rates = load_kr_rates(lake, config.CASH_SERIES["KR"], grid.dates)
    cash = (
        build_cash_account(
            rates,
            grid.dates,
            series_id=config.CASH_SERIES["KR"],
            staleness_days=config.CASH_STALENESS_DAYS,
        )
        if rates is not None
        else None
    )
    if cash is None:
        missing.append(config.CASH_SERIES["KR"])
    return KrInputs(
        grid=grid,
        series=series,
        vix_proxy=vix_proxy,
        realized=realized,
        cash=cash,
        snapshot=snapshot,
        input_files=collect_input_files(files, hash_files=hash_files),
        missing_series=tuple(missing),
        diagnostics=diag,
    )


def load_us_inputs(lake: PinnedScopedLake, *, hash_files: bool = True) -> UsInputs:
    """US SPY 총수익 경로·거시 PIT 행·DGS3MO 현금·실현 배열을 읽는다 (``build_us`` 방식)."""
    grid, lake_cal = build_us_grid(lake)
    paths, diags = load_us_total_return(
        lake, {US_REALIZED_KEY: SPY_SYMBOL}, sessions=frozenset(lake_cal.sessions)
    )
    path = paths[US_REALIZED_KEY]
    # 가용 시각 = 그 세션 폐장 + 60분 (panel.PRICE_AVAILABILITY_BUFFER)
    avail = [grid.calendar.close_at(s) + PRICE_AVAILABILITY_BUFFER for s in path["session"]]
    tr_rows = pl.DataFrame(
        {
            "date": path["session"],
            "value": path["tr_index"].cast(pl.Float64),
            "available_at": pl.Series(avail, dtype=UTC_TS),
        }
    )
    series: dict[str, pl.DataFrame] = {TR_INDEX_KEY: tr_rows}
    missing: list[str] = []
    macro = load_us_macro_rows(lake, US_MACRO_SERIES)
    for sid in US_MACRO_SERIES:
        sub = pit_rows_from_macro(macro, sid)
        if sub.height:
            series[sid] = sub
        else:
            missing.append(sid)
    rates = load_us_rates(lake, config.CASH_SERIES["US"])
    cash = (
        build_cash_account(
            rates,
            grid.dates,
            series_id=config.CASH_SERIES["US"],
            staleness_days=config.CASH_STALENESS_DAYS,
        )
        if rates is not None
        else None
    )
    if cash is None:
        missing.append(config.CASH_SERIES["US"])
    snapshot = {t: lake.latest_snapshot(t).isoformat() for t in US_TABLES_MRS}
    files = lake.input_files(US_TABLES_MRS)
    return UsInputs(
        grid=grid,
        series=series,
        realized={US_REALIZED_KEY: realized_on_grid(grid.dates, tr_rows)},
        cash=cash,
        snapshot=snapshot,
        input_files=collect_input_files(files, hash_files=hash_files),
        missing_series=tuple(missing),
        diagnostics={
            "rows": {k: v.height for k, v in series.items()},
            "spy_path": {k: v for k, v in diags[US_REALIZED_KEY].items()},
            "calendar_basis": grid.calendar.calendar_basis,
            "lake_calendar_basis": lake_cal.calendar_basis,
        },
    )


__all__ = [
    "GridMismatchError",
    "Grid",
    "InputFile",
    "KR_GRID_SERIES",
    "KR_PIT_SERIES",
    "KrInputs",
    "UsInputs",
    "assert_us_calendars_match",
    "build_kr_grid",
    "build_us_grid",
    "collect_input_files",
    "last_spy_session",
    "load_kr_inputs",
    "load_us_inputs",
    "make_grid",
    "pin_us_lake",
    "realized_on_grid",
    "resolve_kr_lake",
    "vix_proxy_rows",
]
