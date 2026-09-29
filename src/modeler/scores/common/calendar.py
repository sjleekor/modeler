"""자산별 거래 세션과 결정 시각.

사양 01 §5: 기준 세션 ``t``의 종가를 쓰고 **다음 자산 거래 세션 개장 30분 전**을
``decision_at``으로 둔다. 진입은 ``t+1`` 종가, 만기는 진입에서 60세션 뒤 종가.

* US: 레이크 ``trading_calendar``(exchange_calendars 4.x로 만든 표, 조기 폐장 포함).
* KR: ``exchange_calendars`` XKRX가 설치돼 있으면 그것, 없으면 관측된 가격일에서
  만든다(``calendar_basis``에 표시).

모든 시각은 tz-aware UTC로 낸다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

DECISION_LEAD = timedelta(minutes=30)
HORIZON_SESSIONS = 60

TZ_BY_CALENDAR = {"XNYS": "America/New_York", "XKRX": "Asia/Seoul"}
DEFAULT_OPEN = {"XNYS": time(9, 30), "XKRX": time(9, 0)}
DEFAULT_CLOSE = {"XNYS": time(16, 0), "XKRX": time(15, 30)}

UTC_TS = pl.Datetime("us", "UTC")


class CalendarError(ValueError):
    """세션이 아니거나 달력 끝을 넘었다."""


def _local_to_utc(d: date, t: time, tz: str) -> datetime:
    return datetime.combine(d, t, tzinfo=ZoneInfo(tz)).astimezone(UTC)


@dataclass(frozen=True)
class SessionCalendar:
    calendar_id: str
    calendar_basis: str
    sessions: tuple[date, ...]
    opens: tuple[datetime, ...]
    closes: tuple[datetime, ...]
    _index: dict[date, int] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not (len(self.sessions) == len(self.opens) == len(self.closes)):
            raise ValueError("sessions/opens/closes 길이가 다릅니다")
        if any(b <= a for a, b in zip(self.sessions, self.sessions[1:], strict=False)):
            raise ValueError("sessions는 오름차순·중복 없음이어야 합니다")
        object.__setattr__(self, "_index", {d: i for i, d in enumerate(self.sessions)})

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_sessions(
        cls,
        calendar_id: str,
        sessions: Sequence[date],
        *,
        calendar_basis: str,
        close_local: Sequence[time | None] | None = None,
    ) -> SessionCalendar:
        """세션 날짜 목록에서 개장/폐장 시각을 채운다(현지 시각 -> UTC)."""
        tz = TZ_BY_CALENDAR[calendar_id]
        ds = sorted(set(sessions))
        closes_local = close_local if close_local is not None else [None] * len(ds)
        if len(closes_local) != len(ds):
            raise ValueError("close_local 길이가 세션 수와 다릅니다")
        opens = tuple(_local_to_utc(d, DEFAULT_OPEN[calendar_id], tz) for d in ds)
        closes = tuple(
            _local_to_utc(d, c or DEFAULT_CLOSE[calendar_id], tz)
            for d, c in zip(ds, closes_local, strict=True)
        )
        return cls(calendar_id, calendar_basis, tuple(ds), opens, closes)

    @classmethod
    def from_us_lake(cls, lake, *, exchange: str = "XNYS") -> SessionCalendar:
        """레이크 ``trading_calendar``(lake는 스냅샷이 고정된 ``UsLake``)."""
        tab = (
            lake.scan_raw("trading_calendar")
            .filter(pl.col("exchange") == exchange)
            .select("date", "close_local")
            .unique(subset=["date"], keep="last")
            .sort("date")
            .collect()
        )
        snap = lake.latest_snapshot("trading_calendar").isoformat()
        return cls.from_sessions(
            exchange,
            tab["date"].to_list(),
            calendar_basis=f"lake_trading_calendar@{snap}",
            close_local=tab["close_local"].to_list(),
        )

    @classmethod
    def from_exchange_calendars(
        cls, calendar_id: str, start: date, end: date
    ) -> SessionCalendar | None:
        """``exchange_calendars``가 없으면 ``None``."""
        try:
            import exchange_calendars as xcals
        except ImportError:
            return None
        cal = xcals.get_calendar(calendar_id, start=start.isoformat(), end=end.isoformat())
        sched = cal.schedule
        sessions = tuple(ts.date() for ts in sched.index)
        opens = tuple(ts.to_pydatetime().astimezone(UTC) for ts in sched["open"])
        closes = tuple(ts.to_pydatetime().astimezone(UTC) for ts in sched["close"])
        ver = getattr(xcals, "__version__", "?")
        return cls(calendar_id, f"exchange_calendars=={ver}", sessions, opens, closes)

    @classmethod
    def from_observed_dates(cls, calendar_id: str, dates: Sequence[date]) -> SessionCalendar:
        """달력 라이브러리가 없을 때: 관측된 가격일을 세션으로 본다(휴장 구분 불가)."""
        return cls.from_sessions(calendar_id, dates, calendar_basis="observed_price_dates")

    # -- lookups --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sessions)

    def index_of(self, session: date) -> int:
        try:
            return self._index[session]
        except KeyError:
            raise CalendarError(f"{self.calendar_id} 세션이 아닙니다: {session}") from None

    def session_at(self, idx: int) -> date:
        if not 0 <= idx < len(self.sessions):
            raise CalendarError(
                f"{self.calendar_id} 달력({self.sessions[0]}~{self.sessions[-1]}) 밖입니다"
            )
        return self.sessions[idx]

    def open_at(self, session: date) -> datetime:
        return self.opens[self.index_of(session)]

    def close_at(self, session: date) -> datetime:
        return self.closes[self.index_of(session)]

    def entry_session(self, t: date) -> date:
        """``t+1`` 세션(진입 종가일)."""
        return self.session_at(self.index_of(t) + 1)

    def decision_at(self, t: date) -> datetime:
        """다음 세션 개장 30분 전(UTC). 조기 폐장·서머타임은 세션 표가 처리한다."""
        nxt = self.index_of(t) + 1
        if nxt >= len(self):
            raise CalendarError(f"{t} 다음 세션이 달력에 없습니다")
        return self.opens[nxt] - DECISION_LEAD

    def exit_session(self, entry: date, h: int = HORIZON_SESSIONS) -> date:
        return self.session_at(self.index_of(entry) + h)

    # -- frame ----------------------------------------------------------------
    def session_table(self) -> pl.DataFrame:
        """``idx, session, open_at, close_at, decision_at`` (마지막 세션의 decision_at은 null)."""
        n = len(self)
        decisions = [self.opens[i + 1] - DECISION_LEAD for i in range(n - 1)] + [None]
        return pl.DataFrame(
            {
                "idx": list(range(n)),
                "session": list(self.sessions),
                "open_at": pl.Series(list(self.opens), dtype=UTC_TS),
                "close_at": pl.Series(list(self.closes), dtype=UTC_TS),
                "decision_at": pl.Series(decisions, dtype=UTC_TS),
            }
        )
