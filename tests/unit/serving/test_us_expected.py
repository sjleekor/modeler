"""US E table: D-1 15:00 KST rule, calendar edges and the daily_inputs contract."""

import json
from datetime import date, datetime, time
from pathlib import Path

import pytest

from modeler.serving import daily_inputs, us_expected
from modeler.serving.calendar_sources import nyse_equity_manifest
from modeler.serving.calendars import SEOUL, SessionCalendar

CSV = (Path(__file__).parents[4] / "collector" / "src" / "collector" / "kr" / "infra"
       / "calendar" / "data" / "holidays_krx.csv")
GENERATED = datetime(2026, 9, 30, 23, 0, tzinfo=SEOUL)


def us_cal(start=date(2026, 8, 1), end=date(2026, 12, 31)):
    return SessionCalendar.from_manifest(nyse_equity_manifest(start, end))


def table(start, end, **kw):
    return us_expected.build_table(start, end, holiday_csv=CSV, generated_at=GENERATED, **kw)


def e(day, **kw):
    return us_expected.expected_session(us_cal(), day, **kw)


def test_weekday_expected_is_two_calendar_days_back_session():
    # D=Wed 10-07: D-1 Tue 15:00 KST, Mon 10-05 session closed Tue 05:00 KST.
    assert e(date(2026, 10, 7)) == date(2026, 10, 5)
    assert e(date(2026, 10, 8)) == date(2026, 10, 6)


def test_monday_expected_is_previous_friday():
    # D=Mon 10-12: D-1 Sun 15:00 KST, Friday 10-09 session closed Sat 05:00 KST.
    assert e(date(2026, 10, 12)) == date(2026, 10, 9)
    assert e(date(2026, 10, 13)) == date(2026, 10, 9)


def test_day_after_us_holiday_and_thanksgiving():
    # Labor Day 09-07: D=Tue 09-08 sees Friday 09-04.
    assert e(date(2026, 9, 8)) == date(2026, 9, 4)
    assert e(date(2026, 9, 9)) == date(2026, 9, 4)
    # Thanksgiving 11-26 closed: D=Fri 11-27 (D-1 Thu 15:00 KST) -> Wed 11-25 closed Thu 06:00 KST.
    assert e(date(2026, 11, 27)) == date(2026, 11, 25)
    assert e(date(2026, 11, 30)) == date(2026, 11, 27)


def test_early_close_uses_1300_new_york():
    cal = us_cal()
    assert us_expected.close_kst(cal, date(2026, 11, 27)).isoformat() == "2026-11-28T03:00:00+09:00"
    assert us_expected.close_kst(cal, date(2026, 11, 25)).isoformat() == "2026-11-26T06:00:00+09:00"
    # 04:00 KST on 11-28: the early close (03:00) has passed, a regular close (06:00) would not have.
    assert e(date(2026, 11, 29), collection_at=time(4, 0)) == date(2026, 11, 27)
    # Christmas Eve early close, Dec 25 holiday.
    assert e(date(2026, 12, 28)) == date(2026, 12, 24)


def test_dst_boundaries_shift_close_in_kst():
    cal = us_cal()
    assert us_expected.close_kst(cal, date(2026, 10, 30)).isoformat() == "2026-10-31T05:00:00+09:00"
    assert us_expected.close_kst(cal, date(2026, 11, 2)).isoformat() == "2026-11-03T06:00:00+09:00"
    # After DST ends (11-01), Monday 11-02 is not closed at 05:30 KST on 11-03.
    assert e(date(2026, 11, 4), collection_at=time(5, 30)) == date(2026, 10, 30)
    assert e(date(2026, 11, 4), collection_at=time(6, 30)) == date(2026, 11, 2)
    # Before DST ends the same 05:30 KST is after the 05:00 close.
    assert e(date(2026, 10, 28), collection_at=time(5, 30)) == date(2026, 10, 26)
    # Start of DST in March: EDT close is 05:00 KST.
    cal_mar = us_cal(date(2026, 2, 20), date(2026, 3, 31))
    assert us_expected.close_kst(cal_mar, date(2026, 3, 9)).isoformat() == "2026-03-10T05:00:00+09:00"
    assert us_expected.close_kst(cal_mar, date(2026, 3, 6)).isoformat() == "2026-03-07T06:00:00+09:00"


def test_kr_holidays_are_excluded_and_table_shape():
    result = table(date(2026, 10, 1), date(2026, 12, 31))
    days = result["expected_session_by_report_date"]
    for holiday in ("2026-10-03", "2026-10-05", "2026-10-09", "2026-12-25", "2026-12-31", "2026-10-04"):
        assert holiday not in days
    assert "2026-10-01" in days and "2026-10-06" in days and "2026-12-30" in days
    assert result["report_date_range"]["count"] == len(days)
    assert days["2026-10-01"] == "2026-09-29"
    assert all(date.fromisoformat(v) < date.fromisoformat(k) for k, v in days.items())


def test_table_is_provisional_and_documents_its_rule():
    result = table(date(2026, 10, 1), date(2026, 10, 9))
    assert result["schema_version"] == "us-expected-source.v1"
    assert result["reviewed_status"] == "unreviewed" and result["provisional"] is True
    assert result["market_lag_limit_sessions"] == 2
    assert result["rule"] == us_expected.RULE_NAME
    assert result["collection_assumption"]["collection_time"] == "15:00"
    assert "sha256=" in result["source_reference"] and result["source_reference"]
    assert result["calendar_sources"]["kr"]["holiday_csv_sha256"].startswith("3328e398")
    assert result["generated_at"] == GENERATED.isoformat()
    assert table(date(2026, 10, 1), date(2026, 10, 9)) == result


def test_confirmed_needs_note_and_status_is_restricted():
    with pytest.raises(ValueError, match="review note"):
        table(date(2026, 10, 1), date(2026, 10, 2), reviewed_status="confirmed")
    with pytest.raises(ValueError, match="reviewed_status"):
        table(date(2026, 10, 1), date(2026, 10, 2), reviewed_status="synthetic_fixture")
    ok = table(date(2026, 10, 1), date(2026, 10, 2), reviewed_status="confirmed", review_note="user 2026-10-01")
    assert ok["reviewed_status"] == "confirmed" and ok["review_note"] == "user 2026-10-01"
    with pytest.raises(ValueError, match="nonnegative"):
        table(date(2026, 10, 1), date(2026, 10, 2), market_lag_limit_sessions=-1)


def test_outside_calendar_coverage_stops():
    with pytest.raises(ValueError, match="coverage"):
        table(date(2025, 12, 1), date(2025, 12, 31))  # KR file covers only 2026
    with pytest.raises(ValueError, match="coverage"):
        table(date(2027, 1, 4), date(2027, 1, 8))
    with pytest.raises(ValueError, match="coverage"):
        table(date(2026, 1, 2), date(2026, 1, 2))  # E would be 2025-12-31, before NYSE coverage
    with pytest.raises(ValueError, match="start"):
        table(date(2026, 10, 2), date(2026, 10, 1))


def test_document_six_days_table_a_upper_bound():
    """us_arrival_history.md table A 'raw 상한' column, one by one."""
    documented = {"2026-09-21": "2026-09-18", "2026-09-22": "2026-09-18", "2026-09-23": "2026-09-21",
                  "2026-09-28": "2026-09-25", "2026-09-29": "2026-09-25", "2026-09-30": "2026-09-28"}
    got = table(date(2026, 9, 21), date(2026, 9, 30))["expected_session_by_report_date"]
    assert {k: got[k] for k in documented} == documented
    assert "2026-09-24" not in got and "2026-09-25" not in got  # Chuseok


def test_write_is_atomic_and_never_overwrites(tmp_path):
    result = table(date(2026, 10, 1), date(2026, 10, 2))
    out = us_expected.write_table(result, tmp_path / "e.json")
    assert json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        us_expected.write_table(result, out)
    assert [p.name for p in tmp_path.iterdir()] == ["e.json"]


def test_cli_build_and_failure_exit(tmp_path, capsys):
    out = tmp_path / "e.json"
    args = ["build", "--start", "2026-10-01", "--end", "2026-10-02", "--output", str(out),
            "--holiday-csv", str(CSV)]
    assert us_expected.main(args) == 0
    assert json.loads(capsys.readouterr().out)["report_dates"] == 2
    assert us_expected.main(args) == 1  # exists
    assert us_expected.main(["build", "--start", "2027-01-04", "--end", "2027-01-05",
                             "--output", str(tmp_path / "x.json"), "--holiday-csv", str(CSV)]) == 1
    assert not (tmp_path / "x.json").exists()


def test_daily_inputs_contract_unreviewed_blocks_and_confirmed_reads(tmp_path):
    """The file is what daily_inputs validates: only a confirmed copy may be read."""
    unreviewed = table(date(2026, 10, 1), date(2026, 10, 2))
    confirmed = table(date(2026, 10, 1), date(2026, 10, 2), reviewed_status="confirmed", review_note="fixture")
    for source, allowed in ((unreviewed, False), (confirmed, True)):
        assert source["schema_version"] == "us-expected-source.v1"
        assert isinstance(source["source_reference"], str) and source["source_reference"]
        assert isinstance(source["market_lag_limit_sessions"], int)
        assert (source["reviewed_status"] in {"confirmed"}) is allowed
        assert date.fromisoformat(source["expected_session_by_report_date"]["2026-10-01"]) == date(2026, 9, 29)
    assert daily_inputs.select  # the reader this table is written for


def test_session_command_prints_e_for_any_calendar_day(capsys):
    assert us_expected.main(["session", "--report-date", "2026-10-01"]) == 0
    assert capsys.readouterr().out.strip() == "2026-09-29"
    # Sunday: the prepare on Saturday 17:30 KST targets E(Sunday) = Friday.
    assert us_expected.main(["session", "--report-date", "2026-10-04"]) == 0
    assert capsys.readouterr().out.strip() == "2026-10-02"
    assert us_expected.main(["session", "--report-date", "2026-01-02"]) == 1
    assert us_expected.main(["session", "--report-date", "2028-01-03"]) == 1
