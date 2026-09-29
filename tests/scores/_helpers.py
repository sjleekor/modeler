"""합성 경로·달력 도우미."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from modeler.scores.common.calendar import SessionCalendar


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_cal(n: int = 30, start: date = date(2024, 1, 2), cid: str = "XNYS") -> SessionCalendar:
    return SessionCalendar.from_sessions(cid, weekdays(start, n), calendar_basis="synthetic")


def make_path(
    cal: SessionCalendar, tr: dict[int, float] | list[float], basis: str = "total_return"
) -> pl.DataFrame:
    """``tr``: 달력 인덱스 -> tr_index (dict) 또는 0부터 연속 리스트."""
    items = tr.items() if isinstance(tr, dict) else enumerate(tr)
    rows = [(cal.sessions[i], v) for i, v in items]
    return pl.DataFrame(
        {
            "session": [r[0] for r in rows],
            "px_raw": [float(r[1]) * 100 for r in rows],
            "px_adj": [float(r[1]) * 100 for r in rows],
            "tr_index": [float(r[1]) for r in rows],
            "return_basis": [basis] * len(rows),
        }
    )
