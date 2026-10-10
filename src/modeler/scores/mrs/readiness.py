"""MRS 입력 준비 상태 확인 (사양 §5.3). **개수와 날짜만 본다 — 값은 읽어도 곧바로 버린다.**

계열마다 있는지, 행 수, 최소·최대 관측일, 기간별(1995~1999 / 2000~2009 / 2010~) 격자일 중 같은 날
관측이 없는 수, 10일 넘는 공백의 수와 최대 일수, 저장된 ``available_from_date``가 다음 격자
세션보다 늦은 건수(MI29)를 센다. 그리고 선행 백필 확인(문면 §3.1)을 PASS/FAIL로 낸다.

* KR 값 열은 ``load_kr_series_raw`` 직후 ``drop("value")``로 버린다. US 거시는 값 열을 아예
  ``select`` 하지 않는다(``value is not null`` 필터만 건다). SPY는 날짜만 읽는다. 출력 JSON·표에
  값 통계가 없다는 것은 시험이 지킨다.
* 백필 확인(MI31): ``config.BACKFILL_REQUIRED_START``의 계열 각각이 그 날짜 **+3 달력일** 이하에서
  시작하고, 그 시작일부터 ``2014-06-13``(맥 레이크의 옛 시작일)까지 10일 넘는 공백이 없어야 한다.
  셋이 다 맞아야 전체 PASS다. 문면: "셋을 한 번에 백필하고, 일부만 된 상태로 시작하지 않는다".
* 예상 첫 백분위일은 ``expected_first_dates``다: 실제 관측일·가용 시각에 **시드 난수 값**을 얹어
  점수 엔진을 돌린다. 실제 값은 쓰지 않는다.

CLI::

    python -m modeler.scores.mrs.readiness --market kr|us|all [--kr-snapshot D]

산출물: ``stock_data/<market>/output/regime_score_readiness/<KR 스냅샷>_<YYYYMMDDTHHMM>/``의
``readiness.json``(경로는 ``DataRoot.resolve``). 실제 값으로는 아무것도 계산하지 않는다.
종료 코드는 0이다(백필 FAIL도 정보다). ``--no-expected``로 가짜 값 엔진 실행을 건너뛴다.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.scores.common.calendar import UTC_TS
from modeler.scores.common.kr_inputs import (
    KrLake,
    KrNotSyncedError,
    _with_available_at,
    load_kr_series_raw,
)
from modeler.scores.common.panel import PRICE_AVAILABILITY_BUFFER
from modeler.scores.market_sector.features import fred_available_at
from modeler.scores.mrs import config
from modeler.scores.mrs.inputs import (
    KR_PIT_SERIES,
    SPY_SYMBOL,
    US_MACRO_SERIES,
    US_TABLES_FOR_KR,
    US_TABLES_MRS,
    Grid,
    build_kr_grid,
    build_us_grid,
    pin_us_lake,
    resolve_kr_lake,
)

logger = logging.getLogger(__name__)

#: 10일 넘는 공백 기준(달력일). MS0 계열 나이 상한과 같은 값이다 (MI04).
GAP_DAYS = config.MACRO_STALENESS_DAYS
#: 백필 확인의 끝 날짜 — 맥 레이크 환율·국고채의 옛 시작일 (문면 §3.1, 사양 §5.3).
BACKFILL_WINDOW_END = date(2014, 6, 13)
#: 백필 시작 허용 오차(달력일) (MI31).
INTERP_BACKFILL_TOLERANCE_DAYS = 3
#: 기간 구간(이름, 시작, 끝 포함·None=끝없음). US 격자는 1993부터라 앞 구간 하나를 더 둔다 (MI46).
PERIODS_KR: tuple[tuple[str, date, date | None], ...] = (
    ("1995-1999", date(1995, 1, 1), date(1999, 12, 31)),
    ("2000-2009", date(2000, 1, 1), date(2009, 12, 31)),
    ("2010-", date(2010, 1, 1), None),
)
PERIODS_US: tuple[tuple[str, date, date | None], ...] = (
    ("1993-1994", date(1993, 1, 1), date(1994, 12, 31)),
    *PERIODS_KR,
)
#: readiness가 보는 KR 계열: PIT 아홉 + 현금(CD91) + VIX(US 레이크).
KR_READINESS_SERIES: tuple[str, ...] = (*KR_PIT_SERIES, config.CASH_SERIES["KR"], "VIXCLS")
US_READINESS_SERIES: tuple[str, ...] = (SPY_SYMBOL, *US_MACRO_SERIES, config.CASH_SERIES["US"])


# --------------------------------------------------------------------------- 순수 계산
def _iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def gap_stats(obs_dates: Sequence[date]) -> tuple[int, int]:
    """관측일 연속 쌍의 간격(달력일) 중 ``GAP_DAYS`` 초과 개수와 최대 간격."""
    ds = sorted(set(obs_dates))
    gaps = [(b - a).days for a, b in zip(ds, ds[1:], strict=False)]
    return sum(1 for g in gaps if g > GAP_DAYS), max(gaps, default=0)


def missing_by_period(
    obs_dates: Sequence[date],
    grid_dates: Sequence[date],
    periods: Sequence[tuple[str, date, date | None]],
) -> dict[str, int]:
    """기간마다 격자일 중 **같은 날 관측이 없는** 날의 수."""
    have = set(obs_dates)
    out: dict[str, int] = {}
    for name, lo, hi in periods:
        out[name] = sum(
            1 for d in grid_dates if d >= lo and (hi is None or d <= hi) and d not in have
        )
    return out


def late_available_count(obs_dates: Sequence[date], avail_dates: Sequence[date], grid: Grid) -> int:
    """저장된 가용일이 관측일 **다음 격자 세션**보다 늦은 건수 (MI29).

    다음 격자 세션이 없는 관측일(마지막 격자일 이후)은 센 대상이 아니다.
    """
    n = 0
    for d, a in zip(obs_dates, avail_dates, strict=True):
        nxt = grid.next_session_after(d)
        if nxt is not None and a > nxt:
            n += 1
    return n


def series_report(
    obs_dates: Sequence[date] | None,
    grid_dates: Sequence[date],
    periods: Sequence[tuple[str, date, date | None]],
    *,
    rows: int | None = None,
    late_available: int | None = None,
    first_available_date: date | None = None,
) -> dict[str, Any]:
    """한 계열의 개수·날짜 보고. ``obs_dates``가 ``None``이면 레이크에 없다."""
    if obs_dates is None:
        return {"exists": False}
    ds = sorted(set(obs_dates))
    n_gaps, max_gap = gap_stats(ds)
    out: dict[str, Any] = {
        "exists": True,
        "rows": rows if rows is not None else len(obs_dates),
        "obs_dates": len(ds),
        "min_date": _iso(ds[0]) if ds else None,
        "max_date": _iso(ds[-1]) if ds else None,
        "missing_same_date": missing_by_period(ds, grid_dates, periods),
        "gaps_gt10": n_gaps,
        "max_gap_days": max_gap,
        "late_available_from": late_available,
    }
    if first_available_date is not None:
        out["first_available_date"] = _iso(first_available_date)
    return out


def backfill_check(
    obs_dates_by_series: Mapping[str, Sequence[date] | None],
    required: Mapping[str, date] = config.BACKFILL_REQUIRED_START,
    *,
    window_end: date = BACKFILL_WINDOW_END,
    tolerance_days: int = INTERP_BACKFILL_TOLERANCE_DAYS,
) -> dict[str, Any]:
    """선행 백필 확인 (문면 §3.1, MI31). 계열별 PASS/FAIL과 전체 결과.

    계열 각각: ① 첫 관측일이 ``required + tolerance_days`` 이하, ② 필요 시작일부터
    ``window_end``를 덮는 첫 관측일까지 관측일 간격이 10일을 넘지 않는다(``window_end`` 이상
    관측이 없으면 마지막 관측일에서 ``window_end``까지의 간격도 센다). 계열이 없으면 FAIL.
    """
    per: dict[str, Any] = {}
    for sid, req in required.items():
        raw = obs_dates_by_series.get(sid)
        row: dict[str, Any] = {"required_start": _iso(req)}
        if not raw:
            row.update({"pass": False, "reason": "series_missing", "first_date": None})
            per[sid] = row
            continue
        ds = sorted(set(raw))
        first = ds[0]
        limit = req.toordinal() + tolerance_days
        seq = [req] + [d for d in ds if d >= req]
        covered = next((d for d in seq[1:] if d >= window_end), None)
        if covered is not None:
            seq = [x for x in seq if x <= covered]
        else:
            seq.append(window_end)
        gaps = [(b - a).days for a, b in zip(seq, seq[1:], strict=False)]
        max_gap = max(gaps, default=0)
        n_gaps = sum(1 for g in gaps if g > GAP_DAYS)
        reasons = []
        if first.toordinal() > limit:
            reasons.append("first_date_later_than_required_plus_tolerance")
        if n_gaps:
            reasons.append("gap_gt10_in_window")
        row.update(
            {
                "first_date": _iso(first),
                "tolerance_limit": _iso(date.fromordinal(limit)),
                "window_end": _iso(window_end),
                "gaps_gt10_in_window": n_gaps,
                "max_gap_days_in_window": max_gap,
                "pass": not reasons,
                "reason": ",".join(reasons) if reasons else None,
            }
        )
        per[sid] = row
    return {
        "series": per,
        "tolerance_days": tolerance_days,
        "window_end": _iso(window_end),
        "overall": "PASS" if per and all(r["pass"] for r in per.values()) else "FAIL",
    }


# --------------------------------------------------------------------------- 예상 첫 백분위일
#: 가짜 값 시드. 값은 날짜 계산에 영향이 없다(시험이 시드를 바꿔 확인한다).
EXPECTED_FAKE_SEED = 20261010
EXPECTED_NOTE = "가짜 값으로 낸 예상 날짜, 실제 기록은 실행에서"


def expected_first_dates(
    market: str,
    grid: Grid,
    pit_frames: Mapping[str, pl.DataFrame],
    *,
    seed: int = EXPECTED_FAKE_SEED,
    warmup: int = config.WARMUP_VALID_OBS,
) -> dict[str, Any]:
    """[훅] 가짜 값으로 점수 엔진을 돌려 성분별 예상 ``first_value_date``·``first_pct_date``를 낸다.

    ``pit_frames``: 입력 계열 이름 -> ``date, available_at`` 행(관측일·가용 시각은 **실제 레이크
    것**, 값 열은 없거나 무시한다). 여기서 값은 시드 난수 양수(1~100)로 새로 채운다 — 실제 값은
    읽지도 쓰지도 않는다. 같은 날짜·가용 시각이면 시드와 상관없이 같은 날짜가 나온다(엔진의
    유효 관측 수·창 규칙만 보기 때문). 계열이 빠지면 ``status="missing_inputs"``다.
    """
    from modeler.scores.mrs.score import compute_scores
    from modeler.scores.mrs.vintage import Grid as EngineGrid

    rng = np.random.default_rng(seed)
    fake: dict[str, pl.DataFrame] = {}
    for sid in sorted(pit_frames):
        rows = pit_frames[sid].select("date", "available_at")
        fake[sid] = rows.with_columns(
            pl.Series("value", rng.uniform(1.0, 100.0, rows.height), dtype=pl.Float64)
        ).select("date", "value", "available_at")
    try:
        res = compute_scores(
            market.upper(), EngineGrid.from_lists(grid.dates, grid.decision_at), fake, warmup=warmup
        )
    except ValueError as exc:
        return {"status": "missing_inputs", "note": str(exc)}
    first = [
        {
            "component": r["component"],
            "sub": r["sub"],
            "first_value_date": _iso(r["first_value_date"]),
            "first_pct_date": _iso(r["first_pct_date"]),
        }
        for r in res.first_dates.iter_rows(named=True)
    ]
    return {
        "status": "computed_with_fake_values",
        "note": EXPECTED_NOTE,
        "warmup": warmup,
        "first_dates": first,
        "engine_diagnostics": res.diagnostics,
    }


# --------------------------------------------------------------------------- 레이크 읽기 (날짜만)
def _kr_dates(lake: KrLake, sid: str, grid: Grid) -> tuple[list[date], list[date]] | None:
    """``(관측일, 저장된 가용일)``. 값 열은 읽자마자 버린다. 없으면 ``None``."""
    raw = load_kr_series_raw(lake, sid, grid.dates)
    if raw is None:
        return None
    df = raw.drop("value").sort("date")
    return df["date"].to_list(), df["avail_date"].to_list()


def _us_macro_frame(us_lake, sid: str) -> pl.DataFrame | None:
    """``date, realtime_start``(값이 있는 행만). 값 열은 선택하지 않는다. 없으면 ``None``."""
    rows = (
        us_lake.scan("macro_series")
        .filter((pl.col("series_id") == sid) & pl.col("value").is_not_null())
        .select("date", "realtime_start")
        .collect()
    )
    return rows if rows.height else None


def _us_macro_dates(us_lake, sid: str) -> tuple[list[date], int, date | None] | None:
    """``(관측일, 행 수, 가장 이른 가용일)``. 값 열은 읽지 않는다."""
    rows = _us_macro_frame(us_lake, sid)
    if rows is None:
        return None
    first_at = fred_available_at(rows)["available_at"].min()
    return (
        sorted(set(rows["date"].to_list())),
        rows.height,
        first_at.date() if first_at is not None else None,
    )


def _spy_dates(us_lake) -> list[date]:
    sub = (
        us_lake.scan_raw("prices_daily")
        .filter((pl.col("symbol") == SPY_SYMBOL) & (pl.col("close").cast(pl.Float64) > 0))
        .select("date")
        .collect()
    )
    return sorted(set(sub["date"].to_list()))


# --------------------------------------------------------------------------- 시장별 보고
def _macro_pit_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """``date, realtime_start`` -> ``date, available_at``(MS0 규칙). vintage 행을 그대로 둔다."""
    return fred_available_at(rows).select("date", pl.col("available_at").cast(UTC_TS))


def readiness_kr(kr_lake: KrLake, us_lake=None, *, expected: bool = True) -> dict[str, Any]:
    """KR readiness 보고(JSON 가능한 dict). 값 통계는 없다."""
    grid = build_kr_grid(kr_lake)
    report: dict[str, Any] = {
        "market": "kr",
        "kr_snapshot": kr_lake.snapshot_date,
        "us_snapshots": {},
        "grid": {"first": _iso(grid.dates[0]), "last": _iso(grid.dates[-1]), "n": len(grid)},
        "series": {},
    }
    obs_for_backfill: dict[str, list[date] | None] = {}
    pit: dict[str, pl.DataFrame] = {}
    for sid in (*KR_PIT_SERIES, config.CASH_SERIES["KR"]):
        got = _kr_dates(kr_lake, sid, grid)
        if got is None:
            report["series"][sid] = series_report(None, grid.dates, PERIODS_KR)
            obs_for_backfill[sid] = None
            continue
        obs, avail = got
        obs_for_backfill[sid] = obs
        report["series"][sid] = series_report(
            obs,
            grid.dates,
            PERIODS_KR,
            late_available=late_available_count(obs, avail, grid),
        )
        if sid in KR_PIT_SERIES:
            # 값 자리는 가짜로 채워 가용 시각 규칙(08:30 KST, 누적 최댓값)만 되살린다
            dummy = pl.DataFrame(
                {"date": obs, "value": [1.0] * len(obs), "avail_date": avail},
                schema={"date": pl.Date, "value": pl.Float64, "avail_date": pl.Date},
            )
            pit[sid] = _with_available_at(dummy).select("date", "available_at")
    vix = None
    if us_lake is not None:
        try:
            vix = _us_macro_frame(us_lake, "VIXCLS")
            report["us_snapshots"] = {
                t: us_lake.latest_snapshot(t).isoformat() for t in US_TABLES_FOR_KR
            }
        except FileNotFoundError:
            vix = None
    if vix is None:
        report["series"]["VIXCLS"] = series_report(None, grid.dates, PERIODS_KR)
    else:
        first_at = fred_available_at(vix)["available_at"].min()
        report["series"]["VIXCLS"] = series_report(
            sorted(set(vix["date"].to_list())),
            grid.dates,
            PERIODS_KR,
            rows=vix.height,
            first_available_date=first_at.date(),
        )
        pit["VIXCLS"] = _macro_pit_frame(vix)
    report["backfill"] = backfill_check(obs_for_backfill)
    report["expected_first_dates"] = _expected_section("kr", grid, pit, expected)
    return report


def readiness_us(us_lake, *, expected: bool = True) -> dict[str, Any]:
    """US readiness 보고. 백필 확인은 KR 전용이다(``backfill`` = ``None``)."""
    grid, _ = build_us_grid(us_lake)
    report: dict[str, Any] = {
        "market": "us",
        "kr_snapshot": None,
        "us_snapshots": {t: us_lake.latest_snapshot(t).isoformat() for t in US_TABLES_MRS},
        "grid": {"first": _iso(grid.dates[0]), "last": _iso(grid.dates[-1]), "n": len(grid)},
        "series": {},
    }
    # 달력에 없는 날(성금요일 등)의 가격 행은 경로에서 빠진다(load_us_total_return과 같다).
    sess = set(grid.dates)
    spy_all = _spy_dates(us_lake)
    spy = [d for d in spy_all if d in sess]
    report["series"][SPY_SYMBOL] = series_report(spy, grid.dates, PERIODS_US)
    report["series"][SPY_SYMBOL]["off_calendar_dates"] = len(spy_all) - len(spy)
    # SPY tr_index 가용 시각 = 그 세션 폐장 + 60분 (값 없이 날짜만으로 만든다)
    pit: dict[str, pl.DataFrame] = {
        "tr_index": pl.DataFrame(
            {
                "date": spy,
                "available_at": pl.Series(
                    [grid.calendar.close_at(d) + PRICE_AVAILABILITY_BUFFER for d in spy],
                    dtype=UTC_TS,
                ),
            },
            schema={"date": pl.Date, "available_at": UTC_TS},
        )
    }
    for sid in (*US_MACRO_SERIES, config.CASH_SERIES["US"]):
        frame = _us_macro_frame(us_lake, sid)
        if frame is None:
            report["series"][sid] = series_report(None, grid.dates, PERIODS_US)
            continue
        first_at = fred_available_at(frame)["available_at"].min()
        report["series"][sid] = series_report(
            sorted(set(frame["date"].to_list())),
            grid.dates,
            PERIODS_US,
            rows=frame.height,
            first_available_date=first_at.date(),
        )
        if sid in US_MACRO_SERIES:
            pit[sid] = _macro_pit_frame(frame)
    report["backfill"] = None
    report["expected_first_dates"] = _expected_section("us", grid, pit, expected)
    return report


def _expected_section(
    market: str, grid: Grid, pit: Mapping[str, pl.DataFrame], enabled: bool
) -> dict[str, Any]:
    """``expected_first_dates`` 훅 결과. 꺼 두면 ``status="skipped"``."""
    if not enabled:
        return {"status": "skipped", "note": EXPECTED_NOTE}
    return expected_first_dates(market, grid, pit)


# --------------------------------------------------------------------------- 표·파일
def format_table(report: Mapping[str, Any]) -> str:
    """콘솔 표. 개수·날짜만 나온다."""
    periods = list(next(iter(report["series"].values()), {}).get("missing_same_date", {}) or [])
    if not periods:
        periods = [p[0] for p in (PERIODS_KR if report["market"] == "kr" else PERIODS_US)]
    head = ["series", "rows", "min_date", "max_date", *[f"miss {p}" for p in periods]]
    head += ["gaps>10", "maxgap", "late_avail"]
    lines = [
        f"[{report['market']}] KR 스냅샷={report.get('kr_snapshot')} "
        f"US 스냅샷={report.get('us_snapshots')} 격자 {report['grid']['first']}~"
        f"{report['grid']['last']} ({report['grid']['n']}일)",
        " | ".join(head),
    ]
    for sid, r in report["series"].items():
        if not r["exists"]:
            lines.append(f"{sid} | 없음")
            continue
        miss = [str(r["missing_same_date"].get(p, "-")) for p in periods]
        late = "-" if r["late_available_from"] is None else str(r["late_available_from"])
        lines.append(
            " | ".join(
                [
                    sid,
                    str(r["rows"]),
                    str(r["min_date"]),
                    str(r["max_date"]),
                    *miss,
                    str(r["gaps_gt10"]),
                    str(r["max_gap_days"]),
                    late,
                ]
            )
        )
    bf = report.get("backfill")
    if bf:
        lines.append(f"백필 확인 (끝 {bf['window_end']}, 허용 +{bf['tolerance_days']}일):")
        for sid, r in bf["series"].items():
            lines.append(
                f"  {sid}: {'PASS' if r['pass'] else 'FAIL'} 필요 {r['required_start']} "
                f"첫 관측 {r['first_date']}" + (f" ({r['reason']})" if r.get("reason") else "")
            )
        lines.append(f"  전체: {bf['overall']}")
    exp = report.get("expected_first_dates", {})
    lines.append(f"예상 첫 백분위일: {exp.get('status')} — {exp.get('note')}")
    for r in exp.get("first_dates", []):
        lines.append(
            f"  {r['component']} ({r['sub']}): 첫 값 {r['first_value_date']} "
            f"첫 백분위 {r['first_pct_date']}"
        )
    return "\n".join(lines)


def output_dir(market: str, kr_label: str, now: datetime) -> Path:
    """``stock_data/<market>/output/regime_score_readiness/<KR 스냅샷>_<YYYYMMDDTHHMM>``."""
    return (
        DataRoot.resolve(market).output / "regime_score_readiness" / f"{kr_label}_{now:%Y%m%dT%H%M}"
    )


def write_report(report: Mapping[str, Any], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "readiness.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def run(
    market: str,
    *,
    kr_snapshot: str | None = None,
    now: datetime | None = None,
    expected: bool = True,
) -> list[tuple[dict[str, Any], Path]]:
    """``market``(kr|us|all)의 readiness를 돌려 파일로 쓴다. ``[(보고, 경로)]``."""
    now = now or datetime.now()
    out: list[tuple[dict[str, Any], Path]] = []
    kr_lake: KrLake | None = None
    try:
        kr_lake = resolve_kr_lake(DataRoot.resolve("kr"), kr_snapshot)
    except (KrNotSyncedError, FileNotFoundError):
        if market != "us":
            raise
    # US 단독 실행에서 KR 스냅샷을 못 찾으면 이름표만 "none"이다 (MI47)
    kr_label = kr_lake.snapshot_date if kr_lake is not None else (kr_snapshot or "none")
    if market in ("kr", "all"):
        assert kr_lake is not None
        try:
            us_lake = pin_us_lake(DataRoot.resolve("us"), US_TABLES_FOR_KR, symbols=())
        except FileNotFoundError:
            us_lake = None
        rep = readiness_kr(kr_lake, us_lake, expected=expected)
        out.append((rep, write_report(rep, output_dir("kr", kr_label, now))))
    if market in ("us", "all"):
        us_lake = pin_us_lake(DataRoot.resolve("us"))
        rep = readiness_us(us_lake, expected=expected)
        rep["kr_snapshot"] = kr_lake.snapshot_date if kr_lake is not None else None
        out.append((rep, write_report(rep, output_dir("us", kr_label, now))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--market", choices=["kr", "us", "all"], required=True)
    ap.add_argument("--kr-snapshot", default=None, help="KR 스냅샷 날짜(없으면 최신 완전 스냅샷)")
    ap.add_argument(
        "--no-expected",
        action="store_true",
        help="예상 첫 백분위일(가짜 값 엔진 실행)을 건너뛴다",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        results = run(args.market, kr_snapshot=args.kr_snapshot, expected=not args.no_expected)
    except KrNotSyncedError as exc:
        print(f"KR 레이크 없음: {exc}")
        return 2
    for rep, path in results:
        print(format_table(rep))
        print(f"-> {path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
