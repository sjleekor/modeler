from datetime import date

import numpy as np
import polars as pl

from modeler.scores.common.cash import CASH_BASIS, build_cash_account, rate_available_at
from tests.scores._helpers import weekdays


def _rates(rows):
    return pl.DataFrame(
        rows, schema={"date": pl.Date, "realtime_start": pl.Date, "value": pl.Float64}, orient="row"
    )


def test_constant_rate_365_days_single_step():
    d0 = date(2024, 1, 1)
    d1 = date(2024, 12, 31)  # 365일 뒤
    assert (d1 - d0).days == 365
    acct = build_cash_account(_rates([(d0, d0, 3.65)]), [d0, d1], series_id="T")
    ret, code = acct.interval_return(np.array([0]), np.array([1]))
    assert code[0] == 0 and abs(ret[0] - 0.0365) < 1e-12
    assert acct.cash_basis == CASH_BASIS == "synthetic_short_rate_act365"


def test_weekend_counts_three_calendar_days():
    fri, mon = date(2024, 1, 5), date(2024, 1, 8)
    acct = build_cash_account(_rates([(fri, fri, 3.65)]), [fri, mon], series_id="T")
    assert acct.frame["step_days"][0] == 3
    ret, _ = acct.interval_return(np.array([0]), np.array([1]))
    assert abs(ret[0] - 0.0365 * 3 / 365) < 1e-15


def test_daily_compounding_of_steps():
    sess = weekdays(date(2024, 1, 1), 261)  # 대략 1년
    acct = build_cash_account(
        _rates([(date(2023, 12, 29), date(2023, 12, 29), 3.65)]),
        sess,
        series_id="T",
        staleness_days=10_000,
    )
    ret, _ = acct.interval_return(np.array([0]), np.array([len(sess) - 1]))
    days = (sess[-1] - sess[0]).days
    assert (
        abs(ret[0] - 0.0365 * days / 365) < 1e-3
    )  # 단순이자 대비 복리 효과(약 +0.07%p)만큼만 크다


def test_stale_rate_makes_interval_null():
    d = [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15), date(2024, 1, 22)]
    # 금리는 1/1 관측 한 번뿐. staleness 7일(명시) -> 1/1->1/8 step은 나이 0 ok, 1/8->1/15는
    # 나이 7 ok, 1/15->1/22는 나이 14 -> stale
    acct = build_cash_account(_rates([(d[0], d[0], 5.0)]), d, series_id="T", staleness_days=7)
    assert acct.frame["step_status"].to_list() == ["ok", "ok", "stale", "no_rate_yet"][:3] + [
        "ok" if acct.frame["step_status"][3] == "ok" else "stale"
    ]
    ok_ret, ok_code = acct.interval_return(np.array([0]), np.array([2]))
    bad_ret, bad_code = acct.interval_return(np.array([0]), np.array([3]))
    assert ok_code[0] == 0 and not np.isnan(ok_ret[0])
    assert bad_code[0] == 1 and np.isnan(bad_ret[0])


def test_rate_only_available_from_realtime_start():
    d = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    # 1/2 관측치는 1/4에야 공표(realtime_start) -> 1/2, 1/3 시점엔 금리 없음
    r = _rates([(date(2024, 1, 2), date(2024, 1, 4), 4.0)])
    a = rate_available_at(r, d)
    assert a["rate_pct"].to_list() == [None, None, 4.0]
    acct = build_cash_account(r, d, series_id="T")
    assert acct.frame["step_status"].to_list()[:2] == ["no_rate_yet", "no_rate_yet"]
    ret, code = acct.interval_return(np.array([0]), np.array([1]))
    assert code[0] == 2 and np.isnan(ret[0])


def test_revision_of_older_date_does_not_replace_latest_observation():
    r = _rates(
        [
            (date(2024, 1, 2), date(2024, 1, 3), 4.0),
            (date(2024, 1, 3), date(2024, 1, 4), 4.1),
            (date(2024, 1, 2), date(2024, 1, 10), 3.9),  # 이전 관측일의 뒤늦은 개정
        ]
    )
    a = rate_available_at(r, [date(2024, 1, 5), date(2024, 1, 11)])
    assert a["rate_pct"].to_list() == [4.1, 4.1]
    assert a["rate_obs_date"].to_list() == [date(2024, 1, 3)] * 2


def test_default_staleness_is_14_days():
    from modeler.scores.common.cash import DEFAULT_STALENESS_DAYS
    from modeler.scores.market_sector.config import MsConfig

    assert DEFAULT_STALENESS_DAYS == 14 and MsConfig().cash_staleness_days == 14
