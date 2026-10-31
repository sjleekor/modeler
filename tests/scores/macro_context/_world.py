"""합성 한국 raw snapshot(common_feature_observation_raw)·미국 macro_series와 selection. 실제 데이터는 쓰지 않는다.

리포트 날짜 D = 2026-10-28(수). 한국 일간은 10-28까지 값이 있고 가용일은 관측일 다음 날이라 10-28 관측은
D에 쓸 수 없다. 미국 macro_series는 10-21까지(주간 수집), 한국 표의 FRED 사본은 10-27까지다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from modeler.scores.market_sector import inputs_pin as ip

SEOUL = ZoneInfo("Asia/Seoul")
D = date(2026, 10, 28)
KR_SNAP = "2026-10-28"
US_SNAP = "2026-10-26"
KR_DONE = "2026-10-28T04:29:15+09:00"
US_MACRO_LAST = date(2026, 10, 21)
COPY_LAST = date(2026, 10, 27)
KR_LAST = date(2026, 10, 28)
START = date(2013, 6, 3)

#: 한국 표 일간 series: (기준값, 보폭, 로그 걷기 여부)
KR_DAILY = {
    "rate_kr_cd91": (3.0, 0.02),
    "rate_kr_gov3y": (3.2, 0.03),
    "rate_kr_gov10y": (3.5, 0.03),
    "fx_usdkrw_ecos": (1300.0, 6.0),
    "foreign_net_kospi_ecos": (0.0, 2500.0),
    "foreign_net_kosdaq_ecos": (0.0, 800.0),
    "trdval_kospi_ecos": (150000.0, 12000.0),
    "trdval_kosdaq_ecos": (80000.0, 7000.0),
}
#: 한국 표의 FRED 사본(macro_series와 겹치다가 이어진다). 앞의 넷은 지금 있고 뒤의 다섯은 collector에 곧 들어온다.
COPIES_NOW = {
    "rate_us2y": "DGS2",
    "rate_us10y": "DGS10",
    "global_vix": "VIXCLS",
    "commodity_wti_fred": "DCOILWTICO",
}
COPIES_NEW = {
    "rate_us3m": "DGS3MO",
    "spread_us_baa10y": "BAA10Y",
    "fx_usd_broad": "DTWEXBGS",
    "commodity_brent_fred": "DCOILBRENTEU",
    "rate_us10y_real": "DFII10",
}
US_DAILY = {
    "DGS3MO": (3.0, 0.02),
    "DGS2": (3.5, 0.03),
    "DGS10": (4.0, 0.03),
    "DFII10": (1.5, 0.03),
    "T10Y2Y": (0.5, 0.02),
    "BAA10Y": (2.0, 0.02),
    "VIXCLS": (18.0, 0.8),
    "DTWEXBGS": (110.0, 0.4),
    "DCOILWTICO": (70.0, 1.2),
    "DCOILBRENTEU": (75.0, 1.2),
}
US_MONTHLY = {
    "CPIAUCSL": (240.0, 0.3),
    "PPIACO": (200.0, 0.4),
    "M2SL": (15000.0, 40.0),
    "UNRATE": (5.0, 0.1),
    "PAYEMS": (140000.0, 150.0),
    "FEDFUNDS": (2.0, 0.05),
}
KR_MONTHLY = {
    "macro_cpi": (100.0, 0.2),
    "macro_ppi": (100.0, 0.3),
    "macro_m2": (2000000.0, 8000.0),
    "macro_consumer_sentiment": (100.0, 1.5),
    "macro_unemployment_rate": (3.5, 0.2),
    "macro_employment_rate": (60.0, 0.3),
}


def weekdays(a: date, b: date) -> list[date]:
    n = (b - a).days + 1
    return [a + timedelta(days=i) for i in range(n) if (a + timedelta(days=i)).weekday() < 5]


def months(a: date, b: date) -> list[date]:
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _walk(
    seed: int,
    n: int,
    base: float,
    step: float,
    *,
    positive: bool = True,
    mean_reverting: bool = False,
):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    if mean_reverting:
        return base + step * z
    x = base + np.cumsum(z * step)
    return np.maximum(x, base * 0.2) if positive else x


@dataclass
class World:
    root: Path
    kr_snap: str
    us_snap: str

    @property
    def kr_dir(self) -> Path:
        return (
            self.root
            / "kr"
            / "raw"
            / "raw_postgres"
            / f"snapshot_date={self.kr_snap}"
            / "source=sj2_remote"
        )

    @property
    def us_file(self) -> Path:
        return (
            self.root
            / "us"
            / "derived"
            / "snapshots"
            / "macro_series"
            / f"snapshot_date={self.us_snap}"
            / "part.parquet"
        )

    @property
    def kr_file(self) -> Path:
        return (
            self.kr_dir
            / "common_feature_observation_raw"
            / "schema_version=1"
            / "source=ECOS"
            / "part-000000.parquet"
        )


def build_world(
    root: Path,
    *,
    with_new_copies: bool = False,
    kr_extra: list[tuple] | None = None,
    us_extra: list[tuple] | None = None,
    jump_fx_last: float | None = None,
) -> World:
    """``kr_extra``: 한국 표에 더할 행 ``(series_id, 관측일, 값, 가용일)``. ``us_extra``: 미국 행 ``(series_id, 날짜, rs, 값)``."""
    kr_rows: list[tuple] = []
    days = weekdays(START, KR_LAST)
    for k, (sid, (base, step)) in enumerate(KR_DAILY.items()):
        if sid.startswith("foreign_net"):
            vals = _walk(100 + k, len(days), base, step, positive=False, mean_reverting=True)
        elif sid.startswith("trdval"):
            vals = _walk(100 + k, len(days), base, step, mean_reverting=True)
            vals = np.maximum(vals, base * 0.3)
        else:
            vals = _walk(100 + k, len(days), base, step)
        if sid == "fx_usdkrw_ecos" and jump_fx_last is not None:
            vals[-2] = vals[-3] * (
                1 + jump_fx_last
            )  # 10-27 관측: D에는 가용, 직전 평일 P(10-27)에는 아직 아니다
        for d, v in zip(days, vals, strict=True):
            kr_rows.append((sid, d, float(v), d + timedelta(days=1)))
    copies = dict(COPIES_NOW) | (COPIES_NEW if with_new_copies else {})
    cdays = weekdays(date(2026, 9, 1), COPY_LAST)
    for k, (cid, usid) in enumerate(copies.items()):
        base, step = US_DAILY[usid]
        # macro_series와 같은 시드로 같은 값 위에서 이어 간다(겹치는 날은 같은 값)
        full = _walk(500 + list(US_DAILY).index(usid), len(weekdays(START, COPY_LAST)), base, step)
        full_days = weekdays(START, COPY_LAST)
        by = dict(zip(full_days, full, strict=True))
        for d in cdays:
            v = (
                float(by[d]) + 0.0123456
            )  # 사본은 macro_series와 소수 아래가 다르다(VIX 반올림 시험용)
            kr_rows.append((cid, d, v, d + timedelta(days=1)))
    for k, (sid, (base, step)) in enumerate(KR_MONTHLY.items()):
        ms = months(date(2008, 1, 1), date(2026, 9, 1))
        vals = _walk(700 + k, len(ms), base, step, positive=True)
        for m, v in zip(ms, vals, strict=True):
            end = (date(m.year + (m.month == 12), m.month % 12 + 1, 1)) - timedelta(days=1)
            kr_rows.append((sid, m, float(v), end + timedelta(days=20)))
    for row in kr_extra or []:
        kr_rows.append(row)
    kr_df = pl.DataFrame(
        {
            "series_id": [r[0] for r in kr_rows],
            "observation_date": [r[1] for r in kr_rows],
            "value_numeric": pl.Series([r[2] for r in kr_rows], dtype=pl.Float64).cast(
                pl.Decimal(30, 8)
            ),
            "available_from_date": [r[3] for r in kr_rows],
        },
        schema_overrides={"observation_date": pl.Date, "available_from_date": pl.Date},
    )
    src = root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={KR_SNAP}" / "source=sj2_remote"
    d = src / "common_feature_observation_raw" / "schema_version=1" / "source=ECOS"
    d.mkdir(parents=True)
    kr_df.write_parquet(d / "part-000000.parquet")
    idx = src / "krx_index_daily" / "schema_version=1" / "index_group=kospi"
    idx.mkdir(parents=True)
    pl.DataFrame(
        {"bas_dd": [date(2026, 10, 27)], "idx_nm": ["코스피"], "close_idx": [1.0]}
    ).write_parquet(idx / "part-000000.parquet")
    man = src / "_manifests"
    (man / "table_manifests").mkdir(parents=True)
    pg = "00000017-00000127-1"
    tables = {}
    for t in ("krx_index_daily", "common_feature_observation_raw"):
        tm = man / "table_manifests" / f"{t}.json"
        tm.write_text(
            json.dumps(
                {"source": {"name": "sj2_remote", "pg_snapshot_id": pg}, "table": {"name": t}}
            )
        )
        tables[t] = {"manifest_path": str(tm), "rows_exported": 1}
    (man / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "route": "remote",
                "started_at": KR_DONE,
                "finished_at": KR_DONE,
                "pg_snapshot_id": pg,
                "tables": tables,
            }
        )
    )
    # 미국 macro_series
    us_rows: list[tuple] = []
    udays = weekdays(START, US_MACRO_LAST)
    for k, (sid, (base, step)) in enumerate(US_DAILY.items()):
        vals = _walk(500 + k, len(weekdays(START, COPY_LAST)), base, step)[: len(udays)]
        for d_, v in zip(udays, vals, strict=True):
            us_rows.append((sid, d_, d_ + timedelta(days=1), float(v)))
    for k, (sid, (base, step)) in enumerate(US_MONTHLY.items()):
        ms = months(date(2008, 1, 1), date(2026, 8, 1))
        vals = _walk(900 + k, len(ms), base, step)
        for m, v in zip(ms, vals, strict=True):
            val: float | None = float(v)
            if sid == "CPIAUCSL" and m == date(2025, 10, 1):
                val = None  # FRED에 값이 빈 달
            us_rows.append((sid, m, m + timedelta(days=40), val))
    for row in us_extra or []:
        us_rows.append(row)
    us_df = pl.DataFrame(
        {
            "series_id": [r[0] for r in us_rows],
            "date": [r[1] for r in us_rows],
            "realtime_start": [r[2] for r in us_rows],
            "value": pl.Series([r[3] for r in us_rows], dtype=pl.Float64),
        },
        schema_overrides={"date": pl.Date, "realtime_start": pl.Date},
    )
    uf = (
        root
        / "us"
        / "derived"
        / "snapshots"
        / "macro_series"
        / f"snapshot_date={US_SNAP}"
        / "part.parquet"
    )
    uf.parent.mkdir(parents=True)
    us_df.write_parquet(uf)
    done = datetime(2026, 10, 27, 23, 40, tzinfo=SEOUL).timestamp()
    os.utime(uf, (done, done))
    return World(root, KR_SNAP, US_SNAP)


def make_selection(world: World, report_date: date = D) -> dict:
    """``select_market_sector_inputs``와 같은 모양의 selection(거시 섹션에 필요한 표만 고른다)."""
    kr_root, us_root = world.root / "kr", world.root / "us"
    cutoff = ip.input_cutoff(report_date)
    return {
        "schema_version": ip.SCHEMA,
        "report_date": report_date.isoformat(),
        "selected_at": (cutoff + timedelta(minutes=1)).isoformat(),
        "selection_mode": "scheduled",
        "input_cutoff": cutoff.isoformat(),
        "bundle": {},
        "roots": {"kr": str(kr_root), "us": str(us_root)},
        "limits": {},
        "kr": {
            "status": "selected",
            **ip.describe_kr_snapshot(kr_root, world.kr_snap),
            "skipped": [],
        },
        "us": {
            "status": "selected",
            "tables": {
                "macro_series": {
                    "status": "selected",
                    **ip.describe_us_table(us_root, "macro_series", world.us_snap),
                }
            },
        },
    }
