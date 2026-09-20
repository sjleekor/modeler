"""``modeler.us.labels`` 단위 테스트. ``tmp_path``에 합성 parquet을 쓴다."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.labels import (
    DISTRESS_SHOCK,
    MAX_PLAUSIBLE_ABS_L0,
    OTHER_SIC2,
    UNCLASSIFIED_SIC2,
    add_l2,
    bucket_sic2,
    build_labels,
    neutralize_cross_section,
    trading_day_offsets,
)
from modeler.us.lake import UsLake


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _write_trading_calendar(tmp_path: Path, dates: list[date], exchange: str = "XNYS") -> None:
    frame = pl.DataFrame({"date": dates, "exchange": [exchange] * len(dates)})
    _write_snapshot(tmp_path, "trading_calendar", "2026-09-19", frame)


def _write_corp_actions(tmp_path: Path, rows: list[dict] | None = None) -> None:
    schema = {
        "symbol": pl.String,
        "ex_date": pl.Date,
        "kind": pl.String,
        "to_factor": pl.Float64,
        "for_factor": pl.Float64,
    }
    frame = pl.DataFrame(rows or [], schema=schema)
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


def _write_prices(tmp_path: Path, rows: list[dict]) -> None:
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


def _write_listing_snapshots(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={"as_of": pl.Date, "symbol": pl.String, "financial_status": pl.String},
    )
    _write_snapshot(tmp_path, "listing_snapshots", "2026-09-18", frame)


_PANEL_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.String,
    "cik": pl.Int64,
    "sic": pl.String,
    "sic2": pl.String,
    "mcap_rank": pl.Int32,
    "adv_20d": pl.Float64,
    "exchange": pl.String,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "adj_volume": pl.Float64,
    "price_ge_5": pl.Boolean,
}


def _panel_row(d: date, symbol: str, **overrides: object) -> dict:
    row = {
        "date": d,
        "symbol": symbol,
        "cik": None,
        "sic": None,
        "sic2": None,
        "mcap_rank": None,
        "adv_20d": 1_000_000.0,
        "exchange": "XNYS",
        "close": 10.0,
        "adj_close": 10.0,
        "adj_volume": 1_000.0,
        "price_ge_5": True,
    }
    row.update(overrides)
    return row


def _panel_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_PANEL_SCHEMA)


def _price_row(d: date, symbol: str, close: float) -> dict:
    return {
        "date": d,
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000.0,
    }


# 21거래일짜리 창(t=day0, t+21=day21)을 만드는 데 넉넉한 달력.
_T = date(2020, 1, 1)
_CALENDAR = [_T + timedelta(days=i) for i in range(40)]
_T21 = _CALENDAR[21]


# --- trading_day_offsets --------------------------------------------------------


def test_trading_day_offsets_returns_nth_future_trading_day(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR)
    result = trading_day_offsets(lake, [_T], 21)
    assert result[_T] == _T21


def test_trading_day_offsets_none_when_beyond_calendar(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR[:5])
    result = trading_day_offsets(lake, [_T], 21)
    assert result[_T] is None


# --- bucket_sic2 -----------------------------------------------------------------


def test_bucket_sic2_null_becomes_unclassified() -> None:
    df = pl.DataFrame({"sic2": [None, "10", "10"]})
    result = bucket_sic2(df, min_group_size=2)
    assert result["sic2_bucket"].to_list() == [UNCLASSIFIED_SIC2, "10", "10"]


def test_bucket_sic2_groups_below_min_size_as_other() -> None:
    sic2 = ["10"] * 25 + ["20"] * 5
    df = pl.DataFrame({"sic2": sic2})
    result = bucket_sic2(df)  # 기본 min_group_size=20

    counted = result.group_by("sic2_bucket").agg(pl.len().alias("n"))
    counts = {row["sic2_bucket"]: row["n"] for row in counted.to_dicts()}
    assert counts["10"] == 25
    assert counts[OTHER_SIC2] == 5


def test_bucket_sic2_preserves_row_order() -> None:
    """행 순서가 join으로 어긋나지 않아야 한다 (직접 겪은 버그, ``labels.py`` 참고)."""
    sic2 = ["big"] * 25 + [None] + ["big"] * 3
    df = pl.DataFrame({"sic2": sic2, "marker": list(range(len(sic2)))})
    result = bucket_sic2(df)
    assert result.height == df.height
    assert result["marker"].to_list() == list(range(len(sic2)))
    assert result["sic2_bucket"][25] == UNCLASSIFIED_SIC2


# --- neutralize_cross_section ----------------------------------------------------


def test_neutralize_cross_section_residual_mean_is_zero_for_pure_group_effects() -> None:
    """노이즈 없이 순수 decile·sic2 그룹 효과만 있으면 잔차가 거의 0이어야 한다."""
    adv_effect = {i: i * 0.01 for i in range(10)}
    sic_effect = {"A": 0.05, "B": -0.05}
    rows = []
    for decile in range(10):
        for i in range(25):  # 데실마다 25종목, sic2는 둘 다 20종목 넘게
            sic = "A" if i % 2 == 0 else "B"
            adv = math.exp(decile + i * 1e-4)  # decile 순서를 그대로 보존
            rows.append(
                {
                    "adv_20d": adv,
                    "sic2": sic,
                    "L1": adv_effect[decile] + sic_effect[sic],
                }
            )
    df = pl.DataFrame(rows)

    result = neutralize_cross_section(df)

    assert abs(result["L2"].sum()) < 1e-8
    assert result["L2"].abs().max() < 1e-6


def test_neutralize_cross_section_minority_sic2_does_not_perfectly_fit() -> None:
    """20종목 미만 sic2는 '기타'로 묶여야 한다 — 안 묶이면 그 종목들의 잔차가 0이 된다."""
    rows = [{"adv_20d": 1_000.0 + i, "sic2": "BIG", "L1": 0.0} for i in range(30)]
    tiny_l1 = [0.5, -0.3, 0.9]
    rows += [{"adv_20d": 2_000.0 + i, "sic2": "TINY", "L1": v} for i, v in enumerate(tiny_l1)]
    df = pl.DataFrame(rows)

    result = neutralize_cross_section(df)

    tiny = result.filter(pl.col("sic2") == "TINY")
    assert tiny["sic2_bucket"].to_list() == [OTHER_SIC2] * 3
    # "기타"로 묶였으니 그룹 평균만 제거되고, 서로 다른 잔차가 남아야 한다
    # (안 묶였으면 회귀가 완전 적합돼 셋 다 잔차 0이 된다).
    residuals = {round(v, 9) for v in tiny["L2"].to_list()}
    assert len(residuals) > 1


def test_add_l2_applies_independently_per_date() -> None:
    rows = []
    for d, shift in [(date(2020, 1, 1), 0.0), (date(2020, 2, 1), 10.0)]:
        for i in range(25):
            rows.append(
                {
                    "date": d,
                    "adv_20d": 1_000.0 + i,
                    "sic2": "A" if i % 2 == 0 else "B",
                    "L1": shift + (i % 3) * 0.01,
                }
            )
    df = pl.DataFrame(rows)

    result = add_l2(df)

    assert result["date"].n_unique() == 2
    means = result.group_by("date").agg(pl.col("L2").mean().alias("m"))
    assert means["m"].abs().max() < 1e-8


# --- build_labels: 종가·종가 수익 --------------------------------------------------


def test_build_labels_priced_continuation(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            _price_row(_T, "AAA", 10.0),
            _price_row(_T21, "AAA", 11.0),
            # MKT는 횡단면 종목 수를 2 이상으로 만들려는 보조 종목이다 — 단일
            # 종목(n=1) 횡단면은 중립화 더미가 완전 포화돼(자유도가 없어져)
            # numpy가 별도로 처리해야 하는 퇴화 사례라 실제 데이터에는 없다
            # (리밸런스마다 항상 수천 종목이다).
            _price_row(_T, "MKT", 100.0),
            _price_row(_T21, "MKT", 101.0),
        ],
    )
    panel = _panel_df(
        [
            _panel_row(_T, "AAA", close=10.0, adj_close=10.0),
            _panel_row(_T, "MKT", close=100.0, adj_close=100.0),
        ]
    )

    labels, diag = build_labels(lake, panel=panel)

    row = labels.filter(pl.col("symbol") == "AAA")
    assert row.height == 1
    assert row["L0"][0] == pytest.approx(11.0 / 10.0 - 1)
    assert row["close_reason"][0] is None
    assert diag["closed_by_reason"] == {
        "distress_delisted": 0,
        "other": 0,
        "ticker_reuse_gap": 0,
    }


def test_build_labels_closed_other_uses_last_price_without_shock(
    tmp_path: Path, lake: UsLake
) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    # 부실 지표가 정상("N")이면 "그 외"로 닫는다.
    _write_listing_snapshots(tmp_path, [{"as_of": _T, "symbol": "AAA", "financial_status": "N"}])
    _write_prices(
        tmp_path,
        [
            _price_row(_T, "AAA", 10.0),
            # AAA는 t+21에 가격이 없다. MKT는 있어서 전체 가격 데이터의 최대일이
            # t+21 이상이 되게 하고(리밸런스가 데이터 밖으로 빠지지 않게), AAA의
            # "닫힘"이 개별 종목 사유이지 데이터 경계 문제가 아님을 보장한다.
            _price_row(_T, "MKT", 100.0),
            _price_row(_T21, "MKT", 101.0),
        ],
    )
    panel = _panel_df(
        [
            _panel_row(_T, "AAA", close=10.0, adj_close=10.0),
            _panel_row(_T, "MKT", close=100.0, adj_close=100.0),
        ]
    )

    labels, diag = build_labels(lake, panel=panel)

    row = labels.filter(pl.col("symbol") == "AAA")
    assert row["close_reason"][0] == "other"
    assert row["L0"][0] == pytest.approx(10.0 / 10.0 - 1)  # 마지막 체결가 그대로, 충격 없음
    assert diag["closed_by_reason"] == {
        "distress_delisted": 0,
        "other": 1,
        "ticker_reuse_gap": 0,
    }


def test_build_labels_closed_distress_applies_shumway_shock(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [{"as_of": _T, "symbol": "AAA", "financial_status": "D"}])
    _write_prices(
        tmp_path,
        [
            _price_row(_T, "AAA", 10.0),
            _price_row(_T, "MKT", 100.0),
            _price_row(_T21, "MKT", 101.0),
        ],
    )
    panel = _panel_df(
        [
            _panel_row(_T, "AAA", close=10.0, adj_close=10.0),
            _panel_row(_T, "MKT", close=100.0, adj_close=100.0),
        ]
    )

    labels, diag = build_labels(lake, panel=panel)

    row = labels.filter(pl.col("symbol") == "AAA")
    assert row["close_reason"][0] == "distress_delisted"
    assert row["L0"][0] == pytest.approx(10.0 * (1 - DISTRESS_SHOCK) / 10.0 - 1)
    assert row["L0"][0] == pytest.approx(-DISTRESS_SHOCK)
    assert diag["closed_by_reason"] == {
        "distress_delisted": 1,
        "other": 0,
        "ticker_reuse_gap": 0,
    }


def test_build_labels_drops_rebalance_when_t21_beyond_price_data(
    tmp_path: Path, lake: UsLake
) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    # 가격 데이터가 t까지만 있고 t+21에는 어떤 종목도 값이 없다 — 데이터 경계다.
    _write_prices(tmp_path, [_price_row(_T, "AAA", 10.0)])
    panel = _panel_df([_panel_row(_T, "AAA", close=10.0, adj_close=10.0)])

    labels, diag = build_labels(lake, panel=panel)

    assert diag["dropped_rebalance_dates"] == [_T.isoformat()]
    assert diag["rebalance_dates_usable"] == 0
    assert labels.height == 0


def test_build_labels_excludes_non_finite_l0_from_bad_split_factor(
    tmp_path: Path, lake: UsLake
) -> None:
    """t 시점 ``adj_close``가 non-finite면(레이크의 corp_actions ``to_factor=0``
    분할 결함이 실제로 만드는 값 — ``labels.py`` 모듈 docstring 참고) 라벨에서 뺀다.

    ``build_labels``에 panel을 직접 주입하는 이 테스트에서는 panel의 ``adj_close``가
    ``daily``와 독립이라, 결함을 재현하려면 panel 쪽에 직접 non-finite 값을 넣어야
    한다 — 실제 파이프라인(``build_panel(lake)``)에서는 이 값이 같은 뿌리
    (``prices.adjusted_daily``)에서 나온다.
    """
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            _price_row(_T, "BAD", 10.0),
            _price_row(_T21, "BAD", 11.0),
            _price_row(_T, "MKT", 100.0),
            _price_row(_T21, "MKT", 101.0),
            _price_row(_T, "MKT2", 50.0),
            _price_row(_T21, "MKT2", 51.0),
        ],
    )
    panel = _panel_df(
        [
            _panel_row(_T, "BAD", close=10.0, adj_close=math.inf),
            # BAD를 뺀 뒤에도 횡단면이 n=2로 남게 보조 종목을 둘 둔다(n=1 퇴화
            # 사례 회피 — ``test_build_labels_priced_continuation`` 주석 참고).
            _panel_row(_T, "MKT", close=100.0, adj_close=100.0),
            _panel_row(_T, "MKT2", close=50.0, adj_close=50.0),
        ]
    )

    labels, diag = build_labels(lake, panel=panel)

    assert "BAD" not in labels["symbol"].to_list()
    assert diag["excluded_non_finite_l0"] == {"count": 1, "symbols": ["BAD"]}


def test_build_labels_excludes_implausible_l0(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [_price_row(_T, "WILD", 1.0), _price_row(_T21, "WILD", 20.0)],  # L0=19, 문턱 10 초과
    )
    panel = _panel_df([_panel_row(_T, "WILD", close=1.0, adj_close=1.0)])

    labels, diag = build_labels(lake, panel=panel)

    assert labels.height == 0
    assert diag["excluded_implausible_l0"]["count"] == 1
    assert diag["excluded_implausible_l0"]["symbols"] == ["WILD"]
    assert diag["excluded_implausible_l0"]["threshold_abs_l0"] == MAX_PLAUSIBLE_ABS_L0


def test_build_labels_ticker_reuse_gap_blocks_cross_company_return(
    tmp_path: Path, lake: UsLake
) -> None:
    """공백 뒤 같은 티커에 붙은 값을 이어진 것으로 보지 않는다 (티커 재사용 방어).

    ``prices_daily``는 상폐 뒤 같은 티커를 다른 회사가 쓰는 계열을 구분 없이
    담고 있다(실측 — 패널 종목의 6.8%가 1년 넘는 가격 공백을 갖는다, 예:
    JONE 2018-11-26 $2.13 -> 2026-09-03 $9.84). 정확히 t+21에 값이 있어도
    그 사이 간격이 ``MAX_TICKER_GAP_DAYS``를 넘으면 이어짐으로 보지 않는다.
    """
    # 거래일을 週 단위로 듬성듬성 두면(가짜 달력) 21거래일 뒤(t21)가 t에서
    # 147일 뒤가 된다 — 문턱(60일)을 넘는 진짜 공백을 만들면서도, 매주 계속
    # 거래되는 종목(MKT)의 인접 관측 간격은 7일로 문턱 밑에 둘 수 있다.
    t = date(2020, 1, 1)
    calendar = [t + timedelta(weeks=i) for i in range(22)]
    t21 = calendar[21]

    _write_trading_calendar(tmp_path, calendar)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            _price_row(t, "REUSED", 10.0),
            _price_row(t21, "REUSED", 500.0),  # 공백 뒤 다른 회사가 이어받았다고 가정
            # MKT는 매 거래일(주 단위) 계속 거래돼(간격 7일) REUSED와 대비된다.
            *[_price_row(d, "MKT", 100.0 + i) for i, d in enumerate(calendar)],
        ],
    )
    panel = _panel_df(
        [
            _panel_row(t, "REUSED", close=10.0, adj_close=10.0),
            _panel_row(t, "MKT", close=100.0, adj_close=100.0),
        ]
    )

    labels, diag = build_labels(lake, panel=panel)

    row = labels.filter(pl.col("symbol") == "REUSED")
    assert row["close_reason"][0] == "ticker_reuse_gap"
    # t21의 값(500.0)을 쓰지 않는다 — 공백 앞 마지막 값(t 자신, 10.0)으로 닫혀 L0=0.
    assert row["L0"][0] == pytest.approx(0.0)
    assert diag["closed_by_reason"]["ticker_reuse_gap"] == 1
