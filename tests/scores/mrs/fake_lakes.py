"""MRS 입력층·readiness 시험용 가짜 레이크 (tmp_path 아래).

KR: ``common_feature_observation_raw`` 한 표. 격자는 1996-01-02 ~ 1996-03-30의 월~토(토요일 세션
포함). 저장된 ``available_from_date``는 관측일 다음 **평일**이다(금요일·토요일 값은 월요일) — 토요일
세션이 있는 금요일은 그 값이 격자의 다음 세션(토요일)보다 늦다(MI29).
US: ``prices_daily``(SPY)·``corp_actions``·``macro_series``·``trading_calendar``. 거래 달력은
2024-01-02 ~ 2024-06-30의 XNYS(조기 폐장 없는 구간)다.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.common.calendar import SessionCalendar

KR_SNAP = "2026-09-30"
US_SNAP = "2026-10-06"
KR_FIRST = date(1996, 1, 2)
KR_LAST = date(1996, 3, 30)  # 토요일
SENTINEL = 987654.321  # 값 통계가 출력에 새는지 보는 표지 값
US_FIRST_PRICE = date(2024, 1, 2)
US_LAST_PRICE = date(2024, 6, 27)
US_CAL_END = date(2024, 6, 30)

ALL_KR_SERIES = (
    "market_kospi_ecos",
    "market_kosdaq_ecos",
    "trdval_kospi_ecos",
    "trdval_kosdaq_ecos",
    "foreign_net_kospi_ecos",
    "foreign_net_kosdaq_ecos",
    "fx_usdkrw_ecos",
    "rate_kr_gov3y",
    "rate_kr_gov10y",
    "rate_kr_cd91",
)


def kr_grid_dates() -> list[date]:
    out, d = [], KR_FIRST
    while d <= KR_LAST:
        if d.weekday() < 6:
            out.append(d)
        d += timedelta(days=1)
    return out


def next_weekday(d: date) -> date:
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def kr_obs_frame(
    series_dates: dict[str, list[date]],
    *,
    value: float = SENTINEL,
    extra_rows: list[tuple] | None = None,
) -> pl.DataFrame:
    rows = []
    for sid, ds in series_dates.items():
        for i, d in enumerate(ds):
            rows.append(
                (
                    sid,
                    d,
                    Decimal(f"{value + i * 0.001:.8f}"),
                    next_weekday(d),
                    datetime(2026, 9, 29, tzinfo=UTC),
                )
            )
    rows += extra_rows or []
    return pl.DataFrame(
        rows,
        schema={
            "series_id": pl.String,
            "observation_date": pl.Date,
            "value_numeric": pl.Decimal(20, 8),
            "available_from_date": pl.Date,
            "fetched_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


def write_kr_lake(
    root: Path,
    series_dates: dict[str, list[date]] | None = None,
    *,
    snap: str = KR_SNAP,
    value: float = SENTINEL,
    extra_rows: list[tuple] | None = None,
) -> None:
    """``root/kr/raw/raw_postgres/snapshot_date=<snap>/source=sj2_remote`` 아래에 쓴다."""
    grid = kr_grid_dates()
    if series_dates is None:
        series_dates = {sid: list(grid) for sid in ALL_KR_SERIES}
    table = "common_feature_observation_raw"
    base = root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={snap}" / "source=sj2_remote"
    d = base / table
    d.mkdir(parents=True, exist_ok=True)
    kr_obs_frame(series_dates, value=value, extra_rows=extra_rows).write_parquet(
        d / "part-0.parquet"
    )
    (base / "_manifests").mkdir(parents=True, exist_ok=True)
    (base / "_manifests" / "_SUCCESS.json").write_text(json.dumps({"tables": {table: {"rows": 1}}}))


# --------------------------------------------------------------------------- US
def _write_us(root: Path, table: str, df: pl.DataFrame, snap: str = US_SNAP) -> None:
    d = root / "us" / "derived" / "snapshots" / table / f"snapshot_date={snap}"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part.parquet")


def xnys_sessions(start: date, end: date) -> SessionCalendar:
    cal = SessionCalendar.from_exchange_calendars("XNYS", start, end)
    assert cal is not None, "exchange_calendars가 있어야 합니다"
    return cal


def vix_vintage_rows() -> list[tuple]:
    """KR 격자(1996 1분기)에 걸치는 VIX vintage 사례. ``(series, date, realtime_start, value)``."""
    return [
        # 개정: 처음 15.0, 나중 99.0 — 근사는 처음 vintage
        ("VIXCLS", date(1996, 2, 1), date(1996, 2, 1), 15.0),
        ("VIXCLS", date(1996, 2, 1), date(1996, 3, 1), 99.0),
        # 금요일 관측 -> 다음 KR 격자 세션은 토요일
        ("VIXCLS", date(1996, 2, 2), date(1996, 2, 2), 16.0),
        # 늦게 백필된 관측: realtime_start가 2010-11-22 (근사는 이 값을 무시)
        ("VIXCLS", date(1996, 2, 5), date(2010, 11, 22), 17.0),
        # 마지막 격자일(토요일) 직전 금요일: 다음 세션 토요일이 있어 남는다
        ("VIXCLS", date(1996, 3, 29), date(1996, 3, 29), 18.0),
        # 격자 뒤 관측: 뒤 KR 세션이 없어 근사에서 빠진다
        ("VIXCLS", date(1996, 4, 1), date(1996, 4, 1), 19.0),
    ]


def write_us_lake(
    root: Path,
    *,
    drop_cal_session: date | None = None,
    close_override: dict[date, time] | None = None,
    macro_extra: list[tuple] | None = None,
    with_prices: bool = True,
) -> None:
    cal = xnys_sessions(US_FIRST_PRICE, US_CAL_END)
    sessions = list(cal.sessions)
    price_sessions = [s for s in sessions if s <= US_LAST_PRICE]
    rng = np.random.default_rng(3)
    px = 400.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(price_sessions))))
    if with_prices:
        prices = pl.DataFrame(
            [
                (
                    d,
                    "SPY",
                    Decimal(f"{v:.4f}"),
                    Decimal(f"{v:.4f}"),
                    Decimal(f"{v:.4f}"),
                    Decimal(f"{v:.4f}"),
                    1_000_000,
                )
                for d, v in zip(price_sessions, px, strict=True)
            ],
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "open": pl.Decimal(14, 4),
                "high": pl.Decimal(14, 4),
                "low": pl.Decimal(14, 4),
                "close": pl.Decimal(14, 4),
                "volume": pl.Int64,
            },
            orient="row",
        )
        actions = pl.DataFrame(
            [("SPY", date(2024, 3, 15), "dividend", None, None, Decimal("0.50000"))],
            schema={
                "symbol": pl.String,
                "ex_date": pl.Date,
                "kind": pl.String,
                "to_factor": pl.Decimal(10, 5),
                "for_factor": pl.Decimal(10, 5),
                "amount": pl.Decimal(10, 5),
            },
            orient="row",
        )
        _write_us(root, "prices_daily", prices)
        _write_us(root, "corp_actions", actions)
    macro_rows: list[tuple] = []
    weekdays = [
        date(2023, 6, 1) + timedelta(days=i)
        for i in range(0, (date(2024, 6, 28) - date(2023, 6, 1)).days + 1)
    ]
    weekdays = [d for d in weekdays if d.weekday() < 5]
    for k, sid in enumerate(("VIXCLS", "BAA10Y", "T10Y2Y", "DGS3MO")):
        r = np.random.default_rng(10 + k)
        for d in weekdays:
            macro_rows.append((sid, d, d, float(abs(r.normal(3.0, 0.3)) + 0.1)))
    macro_rows += vix_vintage_rows()
    macro_rows += macro_extra or []
    macro = pl.DataFrame(
        macro_rows, schema=["series_id", "date", "realtime_start", "value"], orient="row"
    )
    _write_us(root, "macro_series", macro)
    cal_sessions = [s for s in sessions if s != drop_cal_session]
    closes = [(close_override or {}).get(s, time(16, 0)) for s in cal_sessions]
    _write_us(
        root,
        "trading_calendar",
        pl.DataFrame(
            {
                "date": cal_sessions,
                "exchange": ["XNYS"] * len(cal_sessions),
                "close_local": closes,
            }
        ),
    )


# --------------------------------------------------------------------------- 긴 가짜 레이크 (e2e)
LONG_KR_FIRST = date(1995, 1, 3)
LONG_KR_LAST = date(2014, 12, 31)
LONG_US_FIRST_PRICE = date(2011, 1, 3)
LONG_US_LAST_PRICE = date(2021, 12, 30)
LONG_US_CAL_END = date(2022, 1, 10)


def _weekdays(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def long_kr_grid() -> list[date]:
    """1995-01-03 ~ 2014-12-31 평일 + (2000년까지) 토요일."""
    out, d = [], LONG_KR_FIRST
    while d <= LONG_KR_LAST:
        if d.weekday() < 5 or (d.weekday() == 5 and d.year <= 2000):
            out.append(d)
        d += timedelta(days=1)
    return out


def write_kr_lake_long(root: Path, *, seed: int = 5, backfilled: bool = True) -> None:
    """실제 config(warm-up 1,260)로 점수가 나오는 합성 KR. 백필 계열은 필요 시작일부터 있다."""
    rng = np.random.default_rng(seed)
    grid = long_kr_grid()
    req = {
        "fx_usdkrw_ecos": date(1990, 1, 3) if backfilled else date(2014, 6, 13),
        "rate_kr_gov3y": date(1998, 11, 13) if backfilled else date(2014, 6, 13),
        "rate_kr_gov10y": date(2000, 12, 18) if backfilled else date(2014, 6, 13),
    }
    series: dict[str, list[date]] = {sid: list(grid) for sid in ALL_KR_SERIES}
    for sid, start in req.items():
        series[sid] = _weekdays(start, LONG_KR_LAST)
    rows = []
    for sid, ds in series.items():
        n = len(ds)
        if sid in ("market_kospi_ecos", "market_kosdaq_ecos"):
            vals = 1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
        elif sid.startswith("trdval"):
            vals = np.exp(rng.normal(10.0, 0.3, n))
        elif sid.startswith("foreign_net"):
            vals = rng.normal(0.0, 100.0, n)
        elif sid == "fx_usdkrw_ecos":
            vals = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
        else:  # 금리(%)
            vals = 4.0 + 0.5 * np.sin(np.arange(n) / 200.0) + rng.normal(0.0, 0.05, n)
        for d, v in zip(ds, vals, strict=True):
            rows.append(
                (sid, d, Decimal(f"{v:.8f}"), next_weekday(d), datetime(2026, 9, 29, tzinfo=UTC))
            )
    frame = pl.DataFrame(
        rows,
        schema={
            "series_id": pl.String,
            "observation_date": pl.Date,
            "value_numeric": pl.Decimal(24, 8),
            "available_from_date": pl.Date,
            "fetched_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    table = "common_feature_observation_raw"
    base = root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={KR_SNAP}" / "source=sj2_remote"
    (base / table).mkdir(parents=True, exist_ok=True)
    frame.write_parquet(base / table / "part-0.parquet")
    (base / "_manifests").mkdir(parents=True, exist_ok=True)
    (base / "_manifests" / "_SUCCESS.json").write_text(json.dumps({"tables": {table: {"rows": 1}}}))


def write_us_lake_long(root: Path, *, seed: int = 7) -> None:
    """SPY 2011-01-03~2021-12-30, 거시 1993~(일별, vintage 하나), 실제 XNYS 달력(조기 폐장 포함)."""
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    cal = xnys_sessions(LONG_US_FIRST_PRICE, LONG_US_CAL_END)
    sessions = list(cal.sessions)
    price_sessions = [s for s in sessions if s <= LONG_US_LAST_PRICE]
    rng = np.random.default_rng(seed)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(price_sessions))))
    prices = pl.DataFrame(
        [
            (d, "SPY", Decimal(f"{v:.4f}"), Decimal(f"{v:.4f}"), Decimal(f"{v:.4f}"),
             Decimal(f"{v:.4f}"), 1_000_000)
            for d, v in zip(price_sessions, px, strict=True)
        ],
        schema={"date": pl.Date, "symbol": pl.String, "open": pl.Decimal(14, 4),
                "high": pl.Decimal(14, 4), "low": pl.Decimal(14, 4), "close": pl.Decimal(14, 4),
                "volume": pl.Int64},
        orient="row",
    )  # fmt: skip
    actions = pl.DataFrame(
        [("SPY", date(2015, 3, 20), "dividend", None, None, Decimal("0.50000"))],
        schema={"symbol": pl.String, "ex_date": pl.Date, "kind": pl.String,
                "to_factor": pl.Decimal(10, 5), "for_factor": pl.Decimal(10, 5),
                "amount": pl.Decimal(10, 5)},
        orient="row",
    )  # fmt: skip
    _write_us(root, "prices_daily", prices)
    _write_us(root, "corp_actions", actions)
    macro_rows = []
    days = _weekdays(date(1993, 1, 1), LONG_US_LAST_PRICE)
    for k, sid in enumerate(("VIXCLS", "BAA10Y", "T10Y2Y", "DGS3MO")):
        r = np.random.default_rng(100 + k)
        base = {"VIXCLS": 18.0, "BAA10Y": 2.5, "T10Y2Y": 1.0, "DGS3MO": 2.0}[sid]
        vals = base + np.cumsum(r.normal(0.0, 0.05, len(days)))
        vals = np.abs(vals) + 0.1
        macro_rows += [(sid, d, d, float(v)) for d, v in zip(days, vals, strict=True)]
    _write_us(
        root,
        "macro_series",
        pl.DataFrame(macro_rows, schema=["series_id", "date", "realtime_start", "value"],
                     orient="row"),
    )  # fmt: skip
    _write_us(
        root,
        "trading_calendar",
        pl.DataFrame(
            {
                "date": sessions,
                "exchange": ["XNYS"] * len(sessions),
                "close_local": [c.astimezone(ny).time() for c in cal.closes],
            }
        ),
    )
