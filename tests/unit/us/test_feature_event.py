"""M2 이벤트·수급 피쳐(F9·F10·F11·F12·F14·F15) 단위 테스트.

``test_prices.py``·``test_panel.py``와 같은 관례로 ``tmp_path``에 합성
parquet을 쓴다. **실제 레이크는 읽지 않는다.**

PIT 테스트가 중심이다 — "t에 만든 값이 t 뒤 원천 행을 쓰지 않았는가"를
피쳐마다 확인한다. 특히 세 곳(지시문 §7)은 경계값까지 본다:

- ``filings_index`` ET 16:00 경계 (F9·F12가 같이 쓰는
  ``features._trading_days.filings_effective_date``)
- ``short_interest`` 공표 지연 10거래일 (F11)
- ``insider_trans`` ``filing_date`` 기준 자르기, ``transaction_date``가 아니다 (F10)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.features._trading_days import (
    ceil_to_trading_day,
    filings_effective_date,
    shift_trading_days,
    trading_days_between,
)
from modeler.us.features.earnings import add_earnings
from modeler.us.features.filing_activity import add_filing_activity
from modeler.us.features.index_membership import add_index_membership
from modeler.us.features.insider import add_insider
from modeler.us.features.market import add_market
from modeler.us.features.short import add_short
from modeler.us.lake import UsLake

# --- 공용 픽스처 -----------------------------------------------------------


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _weekday_trading_days(start: date, end: date) -> list[date]:
    """토·일만 뺀 평일 목록 — 공휴일 없는 단순 달력. 거래일 산수를 예측 가능하게 한다."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _write_trading_calendar(tmp_path: Path, dates: list[date], exchange: str = "XNYS") -> None:
    frame = pl.DataFrame({"date": dates, "exchange": [exchange] * len(dates)})
    _write_snapshot(tmp_path, "trading_calendar", "2026-09-19", frame)


def _panel(rows: list[dict]) -> pl.DataFrame:
    """테스트용 최소 패널 — date·symbol·cik만 있으면 모든 family가 동작한다."""
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "symbol": pl.String, "cik": pl.Int64},
    )


def _et(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0, *, dst: bool) -> datetime:
    """ET 시각을 UTC datetime으로 만든다. ``dst=True``면 EDT(UTC-4), 아니면 EST(UTC-5)."""
    offset = 4 if dst else 5
    return datetime(y, m, d, hh, mm, ss, tzinfo=UTC) + timedelta(hours=offset)


# --- _trading_days ----------------------------------------------------------


def test_ceil_to_trading_day_returns_self_when_already_trading_day(
    tmp_path: Path, lake: UsLake
) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    lf = pl.LazyFrame({"d": [date(2024, 1, 3)]})  # 수요일, 거래일

    result = ceil_to_trading_day(lake, lf, "d").collect()

    assert result["d_trading"].to_list() == [date(2024, 1, 3)]


def test_ceil_to_trading_day_rolls_forward_over_weekend(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    lf = pl.LazyFrame({"d": [date(2024, 1, 6)]})  # 토요일

    result = ceil_to_trading_day(lake, lf, "d").collect()

    assert result["d_trading"].to_list() == [date(2024, 1, 8)]  # 다음 월요일


def test_shift_trading_days_forward_and_backward(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    lf = pl.LazyFrame({"d": [date(2024, 1, 8)]})  # 월요일

    forward = shift_trading_days(lake, lf, "d", 2).collect()
    backward = shift_trading_days(lake, lf, "d", -2).collect()

    assert forward["d_shift+2"].to_list() == [date(2024, 1, 10)]  # 수요일
    assert backward["d_shift-2"].to_list() == [date(2024, 1, 4)]  # 전주 목요일


def test_trading_days_between_counts_exact_trading_days(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    lf = pl.LazyFrame({"a": [date(2024, 1, 2)], "b": [date(2024, 1, 9)]})

    result = trading_days_between(lake, lf, "a", "b").collect()

    assert result["trading_days"].to_list() == [5]


def test_filings_effective_date_16_00_boundary(tmp_path: Path, lake: UsLake) -> None:
    """15:59 ET는 당일, 16:00 ET 정각과 16:01 ET는 다음 거래일로 민다."""
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    # 2024-01-03(수)는 EST(겨울, UTC-5) 구간이다.
    lf = pl.LazyFrame(
        {
            "acceptance_datetime": [
                _et(2024, 1, 3, 15, 59, dst=False),
                _et(2024, 1, 3, 16, 0, dst=False),
                _et(2024, 1, 3, 16, 1, dst=False),
            ]
        },
        schema={"acceptance_datetime": pl.Datetime("us", "UTC")},
    )

    result = (
        filings_effective_date(lake, lf, "acceptance_datetime")
        .sort("acceptance_datetime")
        .collect()
    )

    assert result["acceptance_datetime_effective_date"].to_list() == [
        date(2024, 1, 3),  # 15:59 -> 당일
        date(2024, 1, 4),  # 16:00 정각 -> 다음 거래일
        date(2024, 1, 4),  # 16:01 -> 다음 거래일
    ]


def test_filings_effective_date_dst_boundary(tmp_path: Path, lake: UsLake) -> None:
    """같은 UTC 20:59라도 서머타임 여부에 따라 ET 시각이 달라져 경계 판정이 갈린다.

    EDT(여름, UTC-4)면 UTC 20:59 -> ET 16:59(마감 뒤, 다음 거래일).
    EST(겨울, UTC-5)면 UTC 20:59 -> ET 15:59(마감 전, 당일).
    """
    _write_trading_calendar(
        tmp_path,
        _weekday_trading_days(date(2024, 6, 1), date(2024, 6, 30))
        + _weekday_trading_days(date(2024, 12, 1), date(2024, 12, 31)),
    )
    summer = datetime(2024, 6, 5, 20, 59, tzinfo=UTC)  # EDT 구간
    winter = datetime(2024, 12, 4, 20, 59, tzinfo=UTC)  # EST 구간
    lf = pl.LazyFrame(
        {"acceptance_datetime": [summer, winter]},
        schema={"acceptance_datetime": pl.Datetime("us", "UTC")},
    )

    result = (
        filings_effective_date(lake, lf, "acceptance_datetime")
        .sort("acceptance_datetime")
        .collect()
    )
    by_input = dict(
        zip(
            result["acceptance_datetime"].to_list(),
            result["acceptance_datetime_effective_date"].to_list(),
        )
    )

    assert by_input[summer] == date(2024, 6, 6)  # EDT라 마감 뒤 -> 다음 거래일
    assert by_input[winter] == date(2024, 12, 4)  # EST라 마감 전 -> 당일


# --- F9 earnings.py ----------------------------------------------------------


def _write_earnings_calendar(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "eps": pl.Float64,
            "eps_forecast": pl.Float64,
            "surprise_pct": pl.Float64,
            "n_estimates": pl.Int32,
        },
    )
    _write_snapshot(tmp_path, "earnings_calendar", "2026-09-18", frame)


def _write_universe_daily_cik(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.String, "cik": pl.Int64})
    _write_snapshot(tmp_path, "universe_daily", "2026-09-18", frame)


def _write_filings_index(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "cik": pl.Int64,
            "form": pl.String,
            "items": pl.String,
            "filing_date": pl.Date,
            "report_date": pl.Date,
            "acceptance_datetime": pl.Datetime("us", "UTC"),
        },
    )
    _write_snapshot(tmp_path, "filings_index", "2026-09-18", frame)


def test_add_earnings_uses_most_recent_release_and_future_schedule(
    tmp_path: Path, lake: UsLake
) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 4, 30)))
    _write_universe_daily_cik(tmp_path, [])
    _write_filings_index(tmp_path, [])
    _write_earnings_calendar(
        tmp_path,
        [
            {
                "date": date(2024, 1, 10),
                "symbol": "AAA",
                "eps": 1.0,
                "eps_forecast": 0.9,
                "surprise_pct": 11.1,
                "n_estimates": 12,
            },
            {
                "date": date(2024, 4, 10),
                "symbol": "AAA",
                "eps": None,
                "eps_forecast": None,
                "surprise_pct": None,
                "n_estimates": 15,
            },
        ],
    )
    panel = _panel([{"date": date(2024, 1, 22), "symbol": "AAA", "cik": None}])  # 1/10 + 8거래일

    result = add_earnings(panel, lake).row(0, named=True)

    assert result["sue_last"] == pytest.approx(11.1)
    assert result["days_since_earn"] == 8
    assert result["days_to_earn"] is not None
    assert result["sue_last_isna"] is False


def test_add_earnings_before_any_release_is_isna(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 4, 30)))
    _write_universe_daily_cik(tmp_path, [])
    _write_filings_index(tmp_path, [])
    _write_earnings_calendar(
        tmp_path,
        [
            {
                "date": date(2024, 4, 10),
                "symbol": "AAA",
                "eps": 1.0,
                "eps_forecast": 0.9,
                "surprise_pct": 5.0,
                "n_estimates": 10,
            }
        ],
    )
    panel = _panel([{"date": date(2024, 1, 3), "symbol": "AAA", "cik": None}])

    result = add_earnings(panel, lake).row(0, named=True)

    assert result["sue_last"] is None
    assert result["sue_last_isna"] is True
    assert result["days_to_earn"] is not None  # 미래 발표는 사전 공지라 안다


def test_add_earnings_pit_8k_after_close_delays_effective_date(
    tmp_path: Path, lake: UsLake
) -> None:
    """실적 발표 당일 8-K가 ET 16:00 뒤 접수되면, 그날 t에는 아직 못 쓰고 다음 거래일부터 쓴다."""
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    _write_universe_daily_cik(tmp_path, [{"date": date(2024, 1, 10), "symbol": "AAA", "cik": 42}])
    _write_filings_index(
        tmp_path,
        [
            {
                "cik": 42,
                "form": "8-K",
                "items": "2.02,9.01",
                "filing_date": date(2024, 1, 10),
                "report_date": None,
                "acceptance_datetime": _et(2024, 1, 10, 17, 0, dst=False),  # 16:00 뒤
            }
        ],
    )
    _write_earnings_calendar(
        tmp_path,
        [
            {
                "date": date(2024, 1, 10),
                "symbol": "AAA",
                "eps": 1.0,
                "eps_forecast": 0.9,
                "surprise_pct": 11.1,
                "n_estimates": 12,
            }
        ],
    )
    panel = _panel(
        [
            {"date": date(2024, 1, 10), "symbol": "AAA", "cik": 42},  # 발표 당일
            {"date": date(2024, 1, 11), "symbol": "AAA", "cik": 42},  # 다음 거래일
        ]
    )

    result = add_earnings(panel, lake).sort("date")

    same_day, next_day = result.row(0, named=True), result.row(1, named=True)
    assert same_day["sue_last_isna"] is True  # 아직 마감 전 접수로 안 밀렸으니 모른다
    assert next_day["sue_last"] == pytest.approx(11.1)
    assert next_day["sue_last_isna"] is False


# --- F10 insider.py -----------------------------------------------------------


def _write_insider_trans(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "accession": pl.String,
            "issuer_cik": pl.Int64,
            "filing_date": pl.Date,
            "trans_date": pl.Date,
            "trans_code": pl.String,
            "trans_shares": pl.Float64,
            "trans_pricepershare": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "insider_trans", "2026-09-18", frame)


def _write_insider_owners(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "accession": pl.String,
            "owner_cik": pl.Int64,
            "is_officer": pl.Boolean,
            "is_director": pl.Boolean,
        },
    )
    _write_snapshot(tmp_path, "insider_owners", "2026-09-18", frame)


def test_add_insider_uses_filing_date_not_transaction_date(tmp_path: Path, lake: UsLake) -> None:
    """거래일(``trans_date``)이 t 이전이어도 ``filing_date``가 t 뒤면 그 거래는 안 쓰인다."""
    table_start = date(2018, 7, 2)
    _write_insider_trans(
        tmp_path,
        [
            {
                "accession": "A1",
                "issuer_cik": 7,
                "filing_date": table_start,  # 워밍업 최소 조건을 맞추는 더미 행
                "trans_date": table_start,
                "trans_code": "S",
                "trans_shares": 1.0,
                "trans_pricepershare": 1.0,
            },
            {
                "accession": "A2",
                "issuer_cik": 7,
                "filing_date": date(2024, 2, 10),  # t(2024-01-15)보다 뒤
                "trans_date": date(2023, 6, 1),  # 거래 자체는 훨씬 이전
                "trans_code": "P",
                "trans_shares": 1000.0,
                "trans_pricepershare": 10.0,
            },
        ],
    )
    _write_insider_owners(
        tmp_path, [{"accession": "A2", "owner_cik": 1, "is_officer": False, "is_director": False}]
    )
    panel = _panel(
        [
            {"date": date(2024, 1, 15), "symbol": "AAA", "cik": 7},  # filing_date 이전
            {"date": date(2024, 2, 20), "symbol": "AAA", "cik": 7},  # filing_date 이후
        ]
    )

    result = add_insider(panel, lake).sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    # t=2024-01-15: A2는 filing_date(2024-02-10)가 t 뒤라 아직 안 보인다 ->
    # 90일 창 안에 아무 거래도 없어 결측이다.
    assert before["ins_netbuy_90_isna"] is True
    # t=2024-02-20: filing_date(2024-02-10)가 t 이전이라 이제 보인다 -> 순매수 +1.
    assert after["ins_netbuy_90_isna"] is False
    assert after["ins_netbuy_90"] == pytest.approx(1.0)


def test_add_insider_cluster_and_officer_counts(tmp_path: Path, lake: UsLake) -> None:
    _write_insider_trans(
        tmp_path,
        [
            {
                # 표 전체의 워밍업 기준(``_table_start``)을 채우는 더미 행 — 다른
                # cik라 B1·B2의 90일 창 계산에는 영향이 없다.
                "accession": "DUMMY",
                "issuer_cik": 1,
                "filing_date": date(2018, 7, 2),
                "trans_date": date(2018, 7, 2),
                "trans_code": "S",
                "trans_shares": 1.0,
                "trans_pricepershare": 1.0,
            },
            {
                "accession": "B1",
                "issuer_cik": 9,
                "filing_date": date(2024, 1, 5),
                "trans_date": date(2024, 1, 5),
                "trans_code": "P",
                "trans_shares": 100.0,
                "trans_pricepershare": 10.0,
            },
            {
                "accession": "B2",
                "issuer_cik": 9,
                "filing_date": date(2024, 1, 20),
                "trans_date": date(2024, 1, 20),
                "trans_code": "P",
                "trans_shares": 50.0,
                "trans_pricepershare": 10.0,
            },
        ],
    )
    _write_insider_owners(
        tmp_path,
        [
            {"accession": "B1", "owner_cik": 100, "is_officer": True, "is_director": False},
            {"accession": "B2", "owner_cik": 200, "is_officer": False, "is_director": False},
        ],
    )
    panel = _panel([{"date": date(2024, 2, 1), "symbol": "BBB", "cik": 9}])

    result = add_insider(panel, lake).row(0, named=True)

    assert result["ins_cluster_90"] == 2  # 서로 다른 신고인 둘(100, 200)
    assert result["ins_officer_buy_90"] == 1  # 임원(B1)만 1건
    assert result["ins_netbuy_90"] == pytest.approx(1.0)  # 매수만 있다


# --- F11 short.py --------------------------------------------------------------


def _write_short_interest(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "settlement_date": pl.Date,
            "symbol": pl.String,
            "current_short_qty": pl.Int64,
            "avg_daily_volume_qty": pl.Int64,
            "days_to_cover": pl.Float64,
            "change_percent": pl.Float64,
            "revision_flag": pl.Boolean,
        },
    )
    _write_snapshot(tmp_path, "short_interest", "2026-09-18", frame)


def _write_short_volume(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "short_volume": pl.Float64,
            "short_exempt_volume": pl.Float64,
            "total_volume": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "short_volume", "2026-09-18", frame)


def test_add_short_settlement_lag_pit_boundary(tmp_path: Path, lake: UsLake) -> None:
    """결제일 + 9거래일은 아직 못 쓰고, + 10거래일부터 쓸 수 있다 (``ASOF_LAG_TRADING_DAYS``)."""
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 3, 31)))
    settlement = date(2024, 1, 15)  # 월요일
    _write_short_interest(
        tmp_path,
        [
            {
                "settlement_date": settlement,
                "symbol": "AAA",
                "current_short_qty": 1000,
                "avg_daily_volume_qty": 500,
                "days_to_cover": 2.0,
                "change_percent": 3.5,
                "revision_flag": False,
            }
        ],
    )
    _write_short_volume(tmp_path, [])
    cal = _weekday_trading_days(date(2024, 1, 1), date(2024, 3, 31))
    idx = cal.index(settlement)
    t_plus_9 = cal[idx + 9]
    t_plus_10 = cal[idx + 10]
    panel = _panel(
        [
            {"date": t_plus_9, "symbol": "AAA", "cik": None},
            {"date": t_plus_10, "symbol": "AAA", "cik": None},
        ]
    )

    result = add_short(panel, lake).sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    assert before["si_ratio_isna"] is True
    assert after["si_ratio_isna"] is False
    assert after["si_ratio"] == pytest.approx(2.0)  # 1000 / 500
    assert after["dtc"] == pytest.approx(2.0)
    assert after["si_chg"] == pytest.approx(3.5)


def test_add_short_sv_share_20_needs_full_window(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 3, 31)))
    _write_short_interest(tmp_path, [])
    days = _weekday_trading_days(date(2024, 1, 2), date(2024, 1, 31))[:20]
    rows = [
        {
            "date": d,
            "symbol": "AAA",
            "short_volume": 30.0,
            "short_exempt_volume": 10.0,
            "total_volume": 100.0,
        }
        for d in days
    ]
    _write_short_volume(tmp_path, rows)
    panel = _panel(
        [
            {"date": days[-2], "symbol": "AAA", "cik": None},  # 19일치만 있음 -> 결측
            {"date": days[-1], "symbol": "AAA", "cik": None},  # 20일치 -> 값 있음
        ]
    )

    result = add_short(panel, lake).sort("date")
    short_window, full_window = result.row(0, named=True), result.row(1, named=True)

    assert short_window["sv_share_20_isna"] is True
    assert full_window["sv_share_20_isna"] is False
    assert full_window["sv_share_20"] == pytest.approx(0.4)  # (30+10)/100


# --- F12 filing_activity.py -----------------------------------------------------


def test_add_filing_activity_16_00_boundary_delays_filing_lag(tmp_path: Path, lake: UsLake) -> None:
    """마감 뒤 접수된 최신 10-Q는 그날 t에는 아직 반영되지 않는다."""
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 31)))
    _write_filings_index(
        tmp_path,
        [
            {
                "cik": 5,
                "form": "10-Q",
                "items": None,
                "filing_date": date(2024, 1, 10),
                "report_date": date(2023, 12, 1),  # lag = 40일
                "acceptance_datetime": _et(2024, 1, 10, 15, 0, dst=False),  # 마감 전
            },
            {
                "cik": 5,
                "form": "10-Q",
                "items": None,
                "filing_date": date(2024, 1, 12),
                "report_date": date(2023, 12, 1),  # lag = 42일
                "acceptance_datetime": _et(2024, 1, 12, 17, 0, dst=False),  # 마감 뒤
            },
        ],
    )
    panel = _panel(
        [
            {"date": date(2024, 1, 12), "symbol": "AAA", "cik": 5},  # 두 번째 공시 접수 당일
            {"date": date(2024, 1, 16), "symbol": "AAA", "cik": 5},  # 그다음 거래일 이후
        ]
    )

    result = add_filing_activity(panel, lake).sort("date")
    same_day, later = result.row(0, named=True), result.row(1, named=True)

    assert same_day["filing_lag"] == 40  # 마감 뒤 접수분은 아직 안 보인다
    assert later["filing_lag"] == 42


def test_add_filing_activity_n_8k_90_counts_within_window(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _weekday_trading_days(date(2024, 1, 1), date(2024, 6, 30)))
    rows = [
        {
            "cik": 5,
            "form": "8-K",
            "items": "7.01",
            "filing_date": d,
            "report_date": None,
            "acceptance_datetime": _et(d.year, d.month, d.day, 10, 0, dst=False),
        }
        for d in [date(2024, 1, 5), date(2024, 2, 5), date(2024, 5, 1)]  # 마지막은 90일 창 밖
    ]
    _write_filings_index(tmp_path, rows)
    panel = _panel([{"date": date(2024, 3, 1), "symbol": "AAA", "cik": 5}])

    result = add_filing_activity(panel, lake).row(0, named=True)

    assert result["n_8k_90"] == 2
    assert result["n_8k_90_isna"] is False


# --- F14 index_membership.py ----------------------------------------------------


def _write_index_constituents(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={"index_id": pl.String, "as_of": pl.Datetime("us", "UTC"), "symbol": pl.String},
    )
    _write_snapshot(tmp_path, "index_constituents", "2026-09-18", frame)


def _as_of(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 12, 0, tzinfo=UTC)


def test_add_index_membership_add_date_and_left_censoring(tmp_path: Path, lake: UsLake) -> None:
    first_revision = date(2018, 1, 4)
    added_revision = date(2024, 2, 1)
    revisions = [first_revision, date(2024, 1, 1), added_revision]
    rows = []
    for r in revisions:
        rows.append({"index_id": "SP500", "as_of": _as_of(r), "symbol": "OLD"})  # 첫날부터 계속
    # NEW는 2024-02-01 리비전에서 처음 등장 -> 실제 관측된 편입
    rows.append({"index_id": "SP500", "as_of": _as_of(added_revision), "symbol": "NEW"})
    _write_index_constituents(tmp_path, rows)
    panel = _panel(
        [
            {"date": date(2024, 3, 1), "symbol": "OLD", "cik": None},
            {"date": date(2024, 3, 1), "symbol": "NEW", "cik": None},
            {"date": date(2024, 3, 1), "symbol": "NEVER", "cik": None},
        ]
    )

    result = add_index_membership(panel, lake).sort("symbol")
    new_row = result.filter(pl.col("symbol") == "NEW").row(0, named=True)
    old_row = result.filter(pl.col("symbol") == "OLD").row(0, named=True)
    never_row = result.filter(pl.col("symbol") == "NEVER").row(0, named=True)

    assert new_row["sp500_member"] is True
    assert new_row["sp500_days_since_add_isna"] is False
    assert new_row["sp500_days_since_add"] == (date(2024, 3, 1) - added_revision).days

    assert old_row["sp500_member"] is True  # 계속 멤버이긴 하다
    assert old_row["sp500_days_since_add_isna"] is True  # 다만 편입일은 왼쪽 잘려 모른다

    assert never_row["sp500_member"] is False
    assert never_row["sp500_days_since_add_isna"] is True


# --- F15 market.py --------------------------------------------------------------


def _write_macro_series(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows, schema={"series_id": pl.String, "realtime_start": pl.Date, "value": pl.Float64}
    )
    _write_snapshot(tmp_path, "macro_series", "2026-09-18", frame)


def _write_corp_actions_empty(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        schema={
            "symbol": pl.String,
            "ex_date": pl.Date,
            "kind": pl.String,
            "to_factor": pl.Float64,
            "for_factor": pl.Float64,
        }
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


def _write_prices_daily(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "prices_daily", "2026-09-18", frame)


def test_add_market_vintage_and_ret_1m(tmp_path: Path, lake: UsLake) -> None:
    trading_days = _weekday_trading_days(date(2024, 1, 1), date(2024, 3, 31))
    _write_trading_calendar(tmp_path, trading_days)
    _write_macro_series(
        tmp_path,
        [
            # t 시점에는 아직 개정 전 값(20.0)만 알려져 있어야 한다.
            {"series_id": "VIXCLS", "realtime_start": date(2024, 1, 2), "value": 20.0},
            {
                "series_id": "VIXCLS",
                "realtime_start": date(2024, 2, 15),
                "value": 21.0,
            },  # t 이후 개정
            {"series_id": "DGS10", "realtime_start": date(2024, 1, 2), "value": 4.0},
            {"series_id": "DGS2", "realtime_start": date(2024, 1, 2), "value": 4.5},
            {"series_id": "BAMLH0A0HYM2", "realtime_start": date(2024, 1, 2), "value": 3.0},
        ],
    )
    _write_corp_actions_empty(tmp_path)
    idx = trading_days.index(date(2024, 2, 1))
    spy_rows = [
        {
            "date": d,
            "symbol": "SPY",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0 + i,
            "volume": 1000.0,
        }
        for i, d in enumerate(trading_days)
    ]
    _write_prices_daily(tmp_path, spy_rows)
    t = trading_days[idx]
    t_minus_21 = trading_days[idx - 21]
    panel = _panel([{"date": t, "symbol": "AAA", "cik": None}])

    result = add_market(panel, lake).row(0, named=True)

    assert result["mkt_vix"] == pytest.approx(20.0)  # 2/15 개정은 아직 안 보인다
    assert result["mkt_term"] == pytest.approx(4.0 - 4.5)
    assert result["mkt_hy"] == pytest.approx(3.0)
    now_close = 100.0 + idx
    past_close = 100.0 + (idx - 21)
    assert result["mkt_ret_1m"] == pytest.approx(now_close / past_close - 1.0)
    assert t_minus_21 in trading_days  # 계산 전제 확인
