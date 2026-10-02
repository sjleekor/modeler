"""Explicit exchange calendars. Dates outside declared coverage stay unknown."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Session:
    day: date
    open_at: time
    close_at: time
    confirmed: bool = True


@dataclass(frozen=True)
class SessionCalendar:
    market: str
    sessions: frozenset[date]
    coverage_start: date
    coverage_end: date
    timezone: ZoneInfo
    open_time: time
    close_time: time
    overrides: dict[date, Session] | None = None
    unconfirmed_dates: frozenset[date] = frozenset()

    @classmethod
    def from_dates(cls, market: str, dates: Iterable[date], *, coverage_start: date,
                   coverage_end: date, timezone: ZoneInfo | None = None,
                   open_time: time | None = None, close_time: time | None = None,
                   overrides: dict[date, Session] | None = None,
                   unconfirmed_dates: Iterable[date] = ()) -> "SessionCalendar":
        defaults = ((time(9), time(15, 30)) if market == "KR" else (time(9, 30), time(16)))
        return cls(market, frozenset(dates), coverage_start, coverage_end,
                   timezone or (SEOUL if market == "KR" else NEW_YORK), open_time or defaults[0],
                   close_time or defaults[1], overrides, frozenset(unconfirmed_dates))

    def knows(self, day: date) -> bool:
        return self.coverage_start <= day <= self.coverage_end

    def is_session(self, day: date) -> bool | None:
        return day in self.sessions if self.knows(day) else None

    def previous_session(self, day: date) -> date | None:
        if day > self.coverage_end or day <= self.coverage_start:
            return None
        cursor = day - timedelta(days=1)
        while cursor >= self.coverage_start:
            if cursor in self.sessions:
                return cursor
            cursor -= timedelta(days=1)
        return None

    def session_distance(self, earlier: date, later: date) -> int | None:
        if not self.knows(earlier) or not self.knows(later):
            return None
        if earlier == later:
            return 0
        sign = 1 if earlier < later else -1
        left, right = sorted((earlier, later))
        return sign * sum(left < d <= right for d in self.sessions)

    def latest_completed_before(self, instant: datetime) -> date | None:
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("instant must include a timezone")
        local_date = instant.astimezone(self.timezone).date()
        if not self.knows(local_date):
            return None
        for day in sorted(self.sessions, reverse=True):
            if day > local_date:
                continue
            session = self.session(day)
            if session is None:
                continue
            if not session.confirmed:
                return None
            close = datetime.combine(day, session.close_at, self.timezone)
            if close <= instant:
                return day
        return None

    @classmethod
    def from_manifest(cls, manifest: dict[str, object]) -> "SessionCalendar":
        """Parse an explicit, versioned session list carried by a prepared-input manifest."""
        market = manifest.get("market")
        zone_name = manifest.get("timezone")
        dates = manifest.get("sessions")
        if market not in {"KR", "US"} or not isinstance(zone_name, str) or not isinstance(dates, list):
            raise ValueError("calendar manifest requires market, timezone, and sessions")
        zone = ZoneInfo(zone_name)
        start = date.fromisoformat(str(manifest.get("coverage_start", "")))
        end = date.fromisoformat(str(manifest.get("coverage_end", "")))
        expected_zone = "Asia/Seoul" if market == "KR" else "America/New_York"
        if start > end or zone_name != expected_zone:
            raise ValueError("calendar coverage or market timezone is invalid")
        sessions = [date.fromisoformat(str(item)) for item in dates]
        if len(sessions) != len(set(sessions)):
            raise ValueError("calendar session list contains duplicate dates")
        override_rows = manifest.get("overrides", [])
        if not isinstance(override_rows, list):
            raise ValueError("calendar overrides must be a list")
        overrides: dict[date, Session] = {}
        for item in override_rows:
            if not isinstance(item, dict):
                raise ValueError("calendar override must be an object")
            day = date.fromisoformat(str(item.get("date", "")))
            if day in overrides:
                raise ValueError("calendar override dates must be unique")
            confirmed = item.get("confirmed", False)
            if not isinstance(confirmed, bool):
                raise ValueError("calendar confirmed value must be boolean")
            overrides[day] = Session(day, time.fromisoformat(str(item["open_at"])),
                                     time.fromisoformat(str(item["close_at"])),
                                     confirmed)
        unconfirmed = manifest.get("unconfirmed_dates", [])
        if not isinstance(unconfirmed, list):
            raise ValueError("unconfirmed_dates must be a list")
        unconfirmed_days = [date.fromisoformat(str(item)) for item in unconfirmed]
        if len(unconfirmed_days) != len(set(unconfirmed_days)):
            raise ValueError("unconfirmed_dates must be unique")
        if any(overrides.get(day) is not None and overrides[day].confirmed
               for day in unconfirmed_days):
            raise ValueError("confirmed override conflicts with unconfirmed_dates")
        calendar = cls.from_dates(market, sessions, coverage_start=start, coverage_end=end,
            timezone=zone, open_time=time.fromisoformat(str(manifest["default_open_at"])),
            close_time=time.fromisoformat(str(manifest["default_close_at"])), overrides=overrides,
            unconfirmed_dates=unconfirmed_days)
        times = [calendar.open_time, calendar.close_time,
                 *(instant for row in overrides.values() for instant in (row.open_at, row.close_at))]
        if (any(item.tzinfo is not None or item.utcoffset() is not None for item in times) or
                calendar.open_time >= calendar.close_time or
                any(row.open_at >= row.close_at for row in overrides.values())):
            raise ValueError("calendar session hours must be ordered local wall times")
        if (any(not calendar.knows(day) for day in sessions) or
                any(day not in calendar.sessions for day in overrides) or
                any(day not in calendar.sessions for day in calendar.unconfirmed_dates)):
            raise ValueError("calendar entries fall outside declared sessions/coverage")
        return calendar

    def session(self, day: date) -> Session | None:
        if self.is_session(day) is not True:
            return None
        if self.overrides and day in self.overrides:
            return self.overrides[day]
        return Session(day, self.open_time, self.close_time, day not in self.unconfirmed_dates)

    def cutoff(self, day: date, at: time) -> datetime:
        if not self.knows(day):
            raise ValueError("date falls outside calendar coverage")
        return datetime.combine(day, at, self.timezone)
