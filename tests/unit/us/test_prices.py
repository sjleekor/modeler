"""``modeler.us.prices`` 단위 테스트.

``test_lake.py``와 같은 관례로 ``tmp_path``에 합성 parquet을 쓴다. 실제 레이크는
읽지 않는다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily, split_factors


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _write_corp_actions(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "ex_date": pl.Date,
            "kind": pl.String,
            "to_factor": pl.Float64,
            "for_factor": pl.Float64,
        },
    )
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


# --- split_factors ------------------------------------------------------------


def test_split_factors_cumulates_two_overlapping_splits(tmp_path: Path, lake: UsLake) -> None:
    """AAPL 7:1(2014-06-09) · 4:1(2020-08-31) 예시. 겹치는 구간에서 곱해져야 한다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": date(2014, 6, 9),
                "kind": "split",
                "to_factor": 7.0,
                "for_factor": 1.0,
            },
            {
                "symbol": "AAPL",
                "ex_date": date(2020, 8, 31),
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            },
        ],
    )

    result = split_factors(lake, base_date=date(2026, 1, 1)).collect().sort("date")

    factors = dict(zip(result["date"].to_list(), result["split_factor"].to_list()))
    # 2014-06-09 행: 자신(1/7)과 그 뒤 2020-08-31 분할(1/4)이 겹쳐 곱해진다.
    assert factors[date(2014, 6, 9)] == pytest.approx(1 / 7 * 1 / 4)
    # 2020-08-31 행: 그 뒤 분할이 없으니 자신만.
    assert factors[date(2020, 8, 31)] == pytest.approx(1 / 4)


def test_split_factors_excludes_splits_after_base_date(tmp_path: Path, lake: UsLake) -> None:
    """base_date(T) 뒤에 일어난 분할은 T 시점엔 알 수 없으므로 제외한다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": date(2014, 6, 9),
                "kind": "split",
                "to_factor": 7.0,
                "for_factor": 1.0,
            },
            {
                "symbol": "AAPL",
                "ex_date": date(2020, 8, 31),
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            },
        ],
    )

    # T를 2020-08-31 이전으로 두면 그 분할은 안 보인다.
    result = split_factors(lake, base_date=date(2015, 1, 1)).collect()

    assert result["date"].to_list() == [date(2014, 6, 9)]
    assert result["split_factor"].item() == pytest.approx(1 / 7)


def test_split_factors_drops_zero_to_factor(tmp_path: Path, lake: UsLake) -> None:
    """``to_factor=0``(2026-09-20 실측 결함)은 가격을 보지 않고 무조건 버린다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AIV",
                "ex_date": date(2019, 2, 20),
                "kind": "split",
                "to_factor": 0.0,
                "for_factor": 0.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2019, 2, 19),
                "symbol": "AIV",
                "open": 40.0,
                "high": 40.0,
                "low": 40.0,
                "close": 40.0,
                "volume": 100.0,
            },
            {
                "date": date(2019, 2, 20),
                "symbol": "AIV",
                "open": 40.0,
                "high": 40.0,
                "low": 40.0,
                "close": 40.0,
                "volume": 100.0,
            },
        ],
    )

    diagnostics: dict[str, int] = {}
    result = split_factors(lake, base_date=date(2026, 1, 1), diagnostics=diagnostics).collect()

    assert result.height == 0
    assert diagnostics["invalid_factor"] == 1
    assert diagnostics["confirmed"] == 0
    assert diagnostics["rejected_price_mismatch"] == 0
    assert diagnostics["unverifiable"] == 0


def test_split_factors_drops_price_mismatched_split(tmp_path: Path, lake: UsLake) -> None:
    """가짜 ex_date(2026-09-20 실측 — 같은 배율 중복 분할)는 가격이 안 맞아 버려진다.

    AMZN 2022-05-26 사례를 축소했다 — 20:1 분할 행이 있는데 그날 가격은 정상
    상승(약 4%)이라 20배 근처와 전혀 안 맞는다.
    """
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "FAKE",
                "ex_date": date(2022, 5, 26),
                "kind": "split",
                "to_factor": 1.0,
                "for_factor": 20.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2022, 5, 25),
                "symbol": "FAKE",
                "open": 2135.50,
                "high": 2135.50,
                "low": 2135.50,
                "close": 2135.50,
                "volume": 100.0,
            },
            {
                "date": date(2022, 5, 26),
                "symbol": "FAKE",
                "open": 2221.55,
                "high": 2221.55,
                "low": 2221.55,
                "close": 2221.55,
                "volume": 100.0,
            },
        ],
    )

    diagnostics: dict[str, int] = {}
    result = split_factors(lake, base_date=date(2026, 1, 1), diagnostics=diagnostics).collect()

    assert result.height == 0
    assert diagnostics["rejected_price_mismatch"] == 1
    assert diagnostics["confirmed"] == 0
    assert diagnostics["unverifiable"] == 0


def test_split_factors_keeps_price_confirmed_forward_split(tmp_path: Path, lake: UsLake) -> None:
    """진짜 분할(가격이 배율만큼 실제로 움직인다)은 살아남는다.

    AMZN 2022-06-06 20:1 사례를 축소했다 — 그날 가격이 20분의 1로 떨어진다.
    """
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "REAL",
                "ex_date": date(2022, 6, 6),
                "kind": "split",
                "to_factor": 20.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2022, 6, 3),
                "symbol": "REAL",
                "open": 2447.00,
                "high": 2447.00,
                "low": 2447.00,
                "close": 2447.00,
                "volume": 100.0,
            },
            {
                "date": date(2022, 6, 6),
                "symbol": "REAL",
                "open": 124.97,
                "high": 124.97,
                "low": 124.97,
                "close": 124.97,
                "volume": 100.0,
            },
        ],
    )

    diagnostics: dict[str, int] = {}
    result = split_factors(lake, base_date=date(2026, 1, 1), diagnostics=diagnostics).collect()

    assert result.height == 1
    assert result["split_factor"].item() == pytest.approx(1 / 20)
    assert diagnostics["confirmed"] == 1
    assert diagnostics["rejected_price_mismatch"] == 0
    assert diagnostics["unverifiable"] == 0


def test_split_factors_keeps_price_confirmed_reverse_split(tmp_path: Path, lake: UsLake) -> None:
    """액면병합(가격이 배율만큼 오른다)도 같은 규칙으로 살아남는다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "MERGE",
                "ex_date": date(2023, 1, 10),
                "kind": "split",
                "to_factor": 1.0,
                "for_factor": 10.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2023, 1, 9),
                "symbol": "MERGE",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
            },
            {
                "date": date(2023, 1, 10),
                "symbol": "MERGE",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
            },
        ],
    )

    diagnostics: dict[str, int] = {}
    result = split_factors(lake, base_date=date(2026, 1, 1), diagnostics=diagnostics).collect()

    assert result.height == 1
    assert result["split_factor"].item() == pytest.approx(10.0)
    assert diagnostics["confirmed"] == 1


def test_split_factors_keeps_unverifiable_split(tmp_path: Path, lake: UsLake) -> None:
    """이 심볼의 원시 가격이 아예 없어 검증이 안 되면 버리지 않고 살린다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "NOPRICE",
                "ex_date": date(2020, 1, 15),
                "kind": "split",
                "to_factor": 2.0,
                "for_factor": 1.0,
            }
        ],
    )
    # NOPRICE의 가격은 하나도 없다 — 다른 심볼만 있다.
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2020, 1, 15),
                "symbol": "OTHER",
                "open": 50.0,
                "high": 50.0,
                "low": 50.0,
                "close": 50.0,
                "volume": 100.0,
            }
        ],
    )

    diagnostics: dict[str, int] = {}
    result = split_factors(lake, base_date=date(2026, 1, 1), diagnostics=diagnostics).collect()

    assert result.height == 1
    assert result["symbol"].item() == "NOPRICE"
    assert result["split_factor"].item() == pytest.approx(1 / 2)
    assert diagnostics["unverifiable"] == 1
    assert diagnostics["confirmed"] == 0
    assert diagnostics["rejected_price_mismatch"] == 0


def test_split_factors_price_tolerance_boundary(tmp_path: Path, lake: UsLake) -> None:
    """1.5배 허용 경계 — ``|log(ratio/f)| < log(1.5)``는 엄격 부등호다.

    ``f=2``(2:1 분할)일 때 허용 구간은 ``ratio/f`` 기준 (1/1.5, 1.5)다.
    ``ratio/f = 1.49``는 안(확인) · ``ratio/f = 1.5`` 정각은 밖(버림)이다.
    """
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "IN",
                "ex_date": date(2021, 3, 1),
                "kind": "split",
                "to_factor": 1.0,
                "for_factor": 2.0,
            },
            {
                "symbol": "OUT",
                "ex_date": date(2021, 3, 1),
                "kind": "split",
                "to_factor": 1.0,
                "for_factor": 2.0,
            },
        ],
    )
    prev_close = 100.0
    # ratio/f = 1.49 -> ratio = 2 * 1.49 = 2.98 -> ex_close = 100 * 2.98 = 298.0
    in_bound_close = prev_close * 2.0 * 1.49
    # ratio/f = 1.5 정각 -> ratio = 3.0 -> ex_close = 300.0 (경계, 버림)
    out_bound_close = prev_close * 2.0 * 1.5

    _write_prices(
        tmp_path,
        [
            {
                "date": date(2021, 2, 26),
                "symbol": "IN",
                "open": prev_close,
                "high": prev_close,
                "low": prev_close,
                "close": prev_close,
                "volume": 100.0,
            },
            {
                "date": date(2021, 3, 1),
                "symbol": "IN",
                "open": in_bound_close,
                "high": in_bound_close,
                "low": in_bound_close,
                "close": in_bound_close,
                "volume": 100.0,
            },
            {
                "date": date(2021, 2, 26),
                "symbol": "OUT",
                "open": prev_close,
                "high": prev_close,
                "low": prev_close,
                "close": prev_close,
                "volume": 100.0,
            },
            {
                "date": date(2021, 3, 1),
                "symbol": "OUT",
                "open": out_bound_close,
                "high": out_bound_close,
                "low": out_bound_close,
                "close": out_bound_close,
                "volume": 100.0,
            },
        ],
    )

    diagnostics: dict[str, int] = {}
    result = split_factors(lake, base_date=date(2026, 1, 1), diagnostics=diagnostics).collect()

    assert result["symbol"].to_list() == ["IN"]
    assert diagnostics["confirmed"] == 1
    assert diagnostics["rejected_price_mismatch"] == 1


def test_split_factors_ignores_dividend_rows(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ex_date": [date(2020, 8, 31), date(2020, 9, 1)],
            "kind": ["split", "dividend"],
            "to_factor": [4.0, None],
            "for_factor": [1.0, None],
            "amount": [None, 0.5],
        }
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)

    result = split_factors(lake, base_date=date(2026, 1, 1)).collect()

    assert result["date"].to_list() == [date(2020, 8, 31)]


# --- adjusted_daily -------------------------------------------------------------


def test_adjusted_daily_is_identity_when_no_splits(tmp_path: Path, lake: UsLake) -> None:
    _write_corp_actions(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2020, 1, 2),
                "symbol": "MSFT",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
            }
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect()

    row = result.row(0, named=True)
    assert row["adj_close"] == pytest.approx(10.5)
    assert row["adj_volume"] == pytest.approx(1000.0)
    assert row["adj_dollar_volume"] == pytest.approx(10.5 * 1000.0)


def test_adjusted_daily_ohlc_share_the_same_factor(tmp_path: Path, lake: UsLake) -> None:
    """OHLC 네 컬럼에 같은 계수가 걸리는가."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "NVDA",
                "ex_date": date(2024, 6, 10),
                "kind": "split",
                "to_factor": 10.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2024, 6, 7),
                "symbol": "NVDA",
                "open": 1200.0,
                "high": 1250.0,
                "low": 1180.0,
                "close": 1210.0,
                "volume": 500.0,
            }
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect().row(0, named=True)

    assert result["adj_open"] == pytest.approx(1200.0 / 10)
    assert result["adj_high"] == pytest.approx(1250.0 / 10)
    assert result["adj_low"] == pytest.approx(1180.0 / 10)
    assert result["adj_close"] == pytest.approx(1210.0 / 10)


def test_adjusted_daily_ex_date_itself_is_not_adjusted(tmp_path: Path, lake: UsLake) -> None:
    """``t < ex_date <= T`` 경계 — ex_date 당일 가격은 이미 조정된 가격이라
    그날 자신의 분할 계수가 다시 걸리면 안 된다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": date(2020, 8, 31),
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2020, 8, 28),  # 분할 직전 거래일
                "symbol": "AAPL",
                "open": 500.0,
                "high": 500.0,
                "low": 500.0,
                "close": 500.0,
                "volume": 100.0,
            },
            {
                "date": date(2020, 8, 31),  # 분할 당일 — 이미 조정된 가격
                "symbol": "AAPL",
                "open": 125.0,
                "high": 125.0,
                "low": 125.0,
                "close": 125.0,
                "volume": 400.0,
            },
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect().sort("date")

    before, on_ex_date = result.row(0, named=True), result.row(1, named=True)
    # 직전 거래일은 4:1로 나뉜다.
    assert before["adj_close"] == pytest.approx(500.0 / 4)
    # 당일은 원시값 그대로 — 계수가 다시 걸리면 125/4가 되어 이 값과 어긋난다.
    assert on_ex_date["adj_close"] == pytest.approx(125.0)
    # 조정 뒤 두 값이 이어진다(연속) — 점프가 없다.
    assert before["adj_close"] == pytest.approx(on_ex_date["adj_close"], rel=1e-6)


def test_adjusted_daily_volume_factor_is_inverse_of_price_factor(
    tmp_path: Path, lake: UsLake
) -> None:
    """거래량 계수가 가격과 반대 방향인가 — 틀리면 거래대금이 조용히 망가진다.

    2:1 분할 전후로 주가는 반토막, 주식 수(거래량)는 두 배가 되는 것이 정상이므로
    원시 달러거래량은 분할 전후로 거의 그대로다. 조정 뒤에도 이 성질이 이어져야
    한다 — 같은 계수를 가격·거래량에 다 걸면 여기서 제곱만큼 틀어진다.
    """
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "T",
                "ex_date": date(2021, 6, 1),
                "kind": "split",
                "to_factor": 2.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2021, 5, 28),  # 분할 전: 종가 100, 거래량 1000
                "symbol": "T",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            {
                "date": date(2021, 6, 1),  # 분할 후: 종가 50, 거래량 2000 (달러거래량 그대로)
                "symbol": "T",
                "open": 50.0,
                "high": 50.0,
                "low": 50.0,
                "close": 50.0,
                "volume": 2000.0,
            },
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect().sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    assert before["adj_close"] == pytest.approx(50.0)
    assert before["adj_volume"] == pytest.approx(2000.0)
    assert before["adj_dollar_volume"] == pytest.approx(after["adj_dollar_volume"], rel=1e-9)
    assert before["adj_dollar_volume"] == pytest.approx(100_000.0)
