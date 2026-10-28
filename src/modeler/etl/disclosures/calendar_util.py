"""접수일 → 사용 가능 시점(다음 거래일). 시각이 없으므로 당일 장중 공시도 당일에는 쓰지 않는다."""

from __future__ import annotations

import bisect
import datetime as dt


class NextTradingDay:
    """KRX 거래일 목록에서 ``d`` 다음 거래일을 돌려준다.

    목록 마지막 날 이후는 평일을 거래일로 가정하고 ``assumed=True``로 표시한다
    (공휴일을 모르므로 보수적이지 않을 수 있다).
    """

    def __init__(self, trade_dates: list[dt.date]) -> None:
        self.days = sorted(set(trade_dates))
        if not self.days:
            raise ValueError("empty trading calendar")

    def next_after(self, d: dt.date) -> tuple[dt.date, bool]:
        if d >= self.days[-1]:
            x = d + dt.timedelta(days=1)
            while x.weekday() >= 5:
                x += dt.timedelta(days=1)
            return x, True
        i = bisect.bisect_right(self.days, d)
        return self.days[i], False
