"""F5(valuation)·F6(profitability)·F7(investment)·F8(payout) 단위 테스트.

``tests/unit`` 관례대로 진짜 레이크는 읽지 않는다 — ``tmp_path``에 작은 합성
``fundamentals``·``corp_actions`` parquet을 쓰고, 패널은 필요한 컬럼(``date,
symbol, cik, close``)만 가진 최소 ``pl.DataFrame``으로 직접 만든다.

각 섹션:

- ``fundamentals_ttm`` — TTM 분기화 공용 엔진 자체의 정확성·PIT(가장 중요한
  부분. Q4 = FY − Q1 − Q2 − Q3 복원, 정정본이 filed로 원본을 덮는 것, 미래
  filed 행을 안 쓰는 것, 분기 사이가 비면 TTM을 내지 않는 것)
- ``add_valuation``·``add_profitability``·``add_investment``·``add_payout`` —
  피쳐마다 최소 하나씩의 PIT 테스트(``06_execution_steps.md`` M2 요구)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.features.fundamentals_ttm import (
    flow_ttm,
    instant_latest,
    instant_yoy_pair,
    market_cap,
    safe_ratio,
)
from modeler.us.features.investment import add_investment
from modeler.us.features.payout import add_payout
from modeler.us.features.profitability import add_profitability
from modeler.us.features.valuation import add_valuation
from modeler.us.lake import UsLake


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _fundamentals(rows: list[dict]) -> pl.DataFrame:
    """최소 컬럼(``cik, tag, fp, start, end, val, filed, form``)의 합성 프레임.

    ``lake.scan("fundamentals")``의 정제 규칙(``_clean_fundamentals``)이 보는
    건 ``end``·``form``뿐이라 나머지 컬럼(``taxonomy``·``unit``·``accn`` 등)은
    안 넣는다 — ``test_lake.py`` 관례 그대로다.
    """
    return pl.DataFrame(
        rows,
        schema={
            "cik": pl.Int64,
            "tag": pl.String,
            "fp": pl.String,
            "start": pl.Date,
            "end": pl.Date,
            "val": pl.Float64,
            "filed": pl.Date,
            "form": pl.String,
        },
    )


def _quarterly_rows(
    cik: int,
    tag: str,
    *,
    fy_start: date,
    q1: float,
    q2cum: float,
    q3cum: float,
    fy_val: float,
    filed_lag: tuple[int, int, int, int] = (45, 45, 45, 60),
) -> list[dict]:
    """한 회계연도의 4행(Q1·Q2·Q3 누적, FY) — 값은 전부 "회계연도 시작부터의
    누적"이다(``fundamentals_ttm.py`` 모듈 docstring 표). ``end``는 91/181/272/365일
    뒤로 고정해 span 판정 범위 안에 깔끔히 들어가게 한다."""
    q1_end = fy_start + timedelta(days=90)
    q2_end = fy_start + timedelta(days=180)
    q3_end = fy_start + timedelta(days=271)
    fy_end = fy_start + timedelta(days=364)
    return [
        {
            "cik": cik,
            "tag": tag,
            "fp": "Q1",
            "start": fy_start,
            "end": q1_end,
            "val": q1,
            "filed": q1_end + timedelta(days=filed_lag[0]),
            "form": "10-Q",
        },
        {
            "cik": cik,
            "tag": tag,
            "fp": "Q2",
            "start": fy_start,
            "end": q2_end,
            "val": q2cum,
            "filed": q2_end + timedelta(days=filed_lag[1]),
            "form": "10-Q",
        },
        {
            "cik": cik,
            "tag": tag,
            "fp": "Q3",
            "start": fy_start,
            "end": q3_end,
            "val": q3cum,
            "filed": q3_end + timedelta(days=filed_lag[2]),
            "form": "10-Q",
        },
        {
            "cik": cik,
            "tag": tag,
            "fp": "FY",
            "start": fy_start,
            "end": fy_end,
            "val": fy_val,
            "filed": fy_end + timedelta(days=filed_lag[3]),
            "form": "10-K",
        },
    ]


def _panel(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "symbol": pl.String, "cik": pl.Int64, "close": pl.Float64},
    )


# =============================================================================
# fundamentals_ttm — 공용 엔진
# =============================================================================


def test_flow_ttm_reconstructs_q4_via_subtraction(tmp_path: Path, lake: UsLake) -> None:
    """Q4 = FY - Q1 - Q2 - Q3. FY2019 10-K가 filed된 직후 TTM은 FY 값과 같다."""
    rows = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 100.0}])

    result = flow_ttm(panel, lake, ["NetIncomeLoss"])

    row = result.row(0, named=True)
    assert row["isna"] is False
    assert row["value"] == pytest.approx(50.0)  # Q1(10)+Q2(12)+Q3(13)+Q4(15) = FY(50)


def test_flow_ttm_ignores_future_filed_row(tmp_path: Path, lake: UsLake) -> None:
    """t 시점에 아직 filed되지 않은 분기는 TTM에 안 들어간다.

    FY2019 4분기 + FY2020 Q1·Q2·Q3까지만 filed된 시점(2020-11-20, Q3-2020
    filed 2020-11-14 직후)의 TTM은 [Q4-2019..Q3-2020] 네 분기 합이어야 하고,
    아직 filed 전인 FY2020 연간행(있다면 미래 정보)은 쓰지 않는다.
    """
    fy2019 = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    fy2020 = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2020, 1, 1), q1=12, q2cum=25, q3cum=40, fy_val=58
    )
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(fy2019 + fy2020))
    panel = _panel([{"date": date(2020, 11, 20), "symbol": "AAA", "cik": 1, "close": 100.0}])

    result = flow_ttm(panel, lake, ["NetIncomeLoss"])

    row = result.row(0, named=True)
    # Q4-2019(15) + Q1-2020(12) + Q2-2020(13) + Q3-2020(15) = 55.
    # FY2020(58, filed 2021-03-01)은 t=2020-11-20에는 아직 안 보인다 — 썼다면
    # Q4-2020(18)까지 들어가 다른 값이 나왔을 것이다.
    assert row["value"] == pytest.approx(55.0)
    assert row["isna"] is False


def test_flow_ttm_amendment_supersedes_after_its_own_filed_date(
    tmp_path: Path, lake: UsLake
) -> None:
    """정정본(10-Q/A)은 그 자신의 filed 시점부터만 원본을 덮는다."""
    fy2019 = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    fy2020 = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2020, 1, 1), q1=12, q2cum=25, q3cum=40, fy_val=58
    )
    amendment = [
        {
            "cik": 1,
            "tag": "NetIncomeLoss",
            "fp": "Q3",
            "start": date(2020, 1, 1),
            "end": date(2020, 1, 1) + timedelta(days=271),
            "val": 45.0,  # 원본 40 -> 45로 상향 정정
            "filed": date(2020, 12, 1),
            "form": "10-Q/A",
        }
    ]
    _write_snapshot(
        tmp_path, "fundamentals", "2026-09-19", _fundamentals(fy2019 + fy2020 + amendment)
    )

    panel = _panel(
        [
            {"date": date(2020, 11, 20), "symbol": "AAA", "cik": 1, "close": 100.0},  # 정정 전
            {"date": date(2020, 12, 5), "symbol": "AAA", "cik": 1, "close": 100.0},  # 정정 후
        ]
    )

    result = flow_ttm(panel, lake, ["NetIncomeLoss"]).sort("date")

    before = result.row(0, named=True)
    after = result.row(1, named=True)
    assert before["value"] == pytest.approx(55.0)  # 원본 Q3=15로 계산
    assert after["value"] == pytest.approx(60.0)  # 정정 Q3=45-25=20으로 계산 (15->20, +5)


def test_flow_ttm_null_when_quarter_gap_exists(tmp_path: Path, lake: UsLake) -> None:
    """분기 사이에 구멍이 있으면(Q2 데이터 없음) 4분기가 안 맞아 TTM을 null로 둔다."""
    rows = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    rows = [r for r in rows if r["fp"] != "Q2"]  # Q2 행 제거 -> Q2·Q3 둘 다 파생 불가
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 100.0}])

    result = flow_ttm(panel, lake, ["NetIncomeLoss"])

    row = result.row(0, named=True)
    assert row["value"] is None
    assert row["isna"] is True


def test_flow_ttm_tag_fallback_prefers_first_present_tag(tmp_path: Path, lake: UsLake) -> None:
    """폴백 우선순위: 앞 태그가 있으면 그것을, 없으면 뒤 태그를 쓴다."""
    primary_only = _quarterly_rows(
        1, "Revenues", fy_start=date(2019, 1, 1), q1=100, q2cum=210, q3cum=330, fy_val=460
    )
    fallback_only = _quarterly_rows(
        2,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        fy_start=date(2019, 1, 1),
        q1=50,
        q2cum=105,
        q3cum=165,
        fy_val=230,
    )
    _write_snapshot(
        tmp_path, "fundamentals", "2026-09-19", _fundamentals(primary_only + fallback_only)
    )
    panel = _panel(
        [
            {"date": date(2020, 3, 5), "symbol": "PRIMARY", "cik": 1, "close": 100.0},
            {"date": date(2020, 3, 5), "symbol": "FALLBACK", "cik": 2, "close": 100.0},
        ]
    )

    result = flow_ttm(
        panel, lake, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
    ).sort("symbol")

    values = dict(zip(result["symbol"].to_list(), result["value"].to_list()))
    assert values["PRIMARY"] == pytest.approx(460.0)  # Revenues가 있으니 그것을 쓴다
    assert values["FALLBACK"] == pytest.approx(230.0)  # Revenues가 없어 폴백 태그를 쓴다


def test_instant_latest_pit_ignores_future_filed_row(tmp_path: Path, lake: UsLake) -> None:
    rows = [
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 1000.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "Q1",
            "start": None,
            "end": date(2020, 3, 31),
            "val": 1100.0,
            "filed": date(2020, 5, 15),
            "form": "10-Q",
        },
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel(
        [
            {"date": date(2020, 4, 1), "symbol": "AAA", "cik": 1, "close": 100.0},  # Q1 filed 전
            {"date": date(2020, 5, 20), "symbol": "AAA", "cik": 1, "close": 100.0},  # Q1 filed 후
        ]
    )

    result = instant_latest(panel, lake, "Assets").sort("date")

    assert result["value"].to_list() == [1000.0, 1100.0]
    assert result["isna"].to_list() == [False, False]


def test_instant_yoy_pair_finds_prior_year_within_tolerance_window(
    tmp_path: Path, lake: UsLake
) -> None:
    rows = [
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 1000.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2020, 12, 31),
            "val": 1200.0,
            "filed": date(2021, 3, 1),
            "form": "10-K",
        },
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel([{"date": date(2021, 3, 10), "symbol": "AAA", "cik": 1, "close": 100.0}])

    result = instant_yoy_pair(panel, lake, "Assets")

    row = result.row(0, named=True)
    assert row["cur_val"] == pytest.approx(1200.0)
    assert row["prior_val"] == pytest.approx(1000.0)


def test_market_cap_multiplies_latest_shares_by_close(tmp_path: Path, lake: UsLake) -> None:
    rows = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 1_000_000.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel([{"date": date(2020, 4, 1), "symbol": "AAA", "cik": 1, "close": 25.0}])

    result = market_cap(panel, lake)

    row = result.row(0, named=True)
    assert row["mcap"] == pytest.approx(25_000_000.0)
    assert row["isna"] is False


def test_safe_ratio_null_when_denominator_zero_or_null() -> None:
    frame = pl.DataFrame({"num": [10.0, 10.0, None], "den": [2.0, 0.0, 5.0]})
    result = frame.select(safe_ratio(pl.col("num"), pl.col("den")).alias("r"))
    assert result["r"].to_list() == [5.0, None, None]


# =============================================================================
# add_valuation (F5)
# =============================================================================


def test_add_valuation_bm_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    rows = [
        {
            "cik": 1,
            "tag": "StockholdersEquity",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 500.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "StockholdersEquity",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 999.0,  # 미래 정정 — t=2020-03-10에는 안 보여야 한다
            "filed": date(2020, 6, 1),
            "form": "10-K/A",
        },
    ]
    panel = _panel([{"date": date(2020, 3, 10), "symbol": "AAA", "cik": 1, "close": 10.0}])
    # mcap 계산에 EntityCommonStockSharesOutstanding이 필요하다 — 없으면 bm이
    # 결측이 되어 이 테스트의 본 목적(정정 필터링)을 확인할 수 없다.
    shares = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 100.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows + shares))

    result = add_valuation(panel, lake)

    row = result.row(0, named=True)
    # mcap = 100주 * 10 = 1000. bm = equity(500, 정정 전) / mcap(1000) = 0.5
    assert row["bm"] == pytest.approx(0.5)
    assert row["bm_isna"] is False


def test_add_valuation_ep_ttm_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    ni_rows = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    shares_rows = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 100.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]
    future_correction = [
        {
            "cik": 1,
            "tag": "NetIncomeLoss",
            "fp": "FY",
            "start": date(2019, 1, 1),
            "end": date(2019, 12, 31),
            "val": 999.0,
            "filed": date(2021, 1, 1),  # panel 날짜보다 뒤 -> 안 보여야 한다
            "form": "10-K/A",
        }
    ]
    _write_snapshot(
        tmp_path,
        "fundamentals",
        "2026-09-19",
        _fundamentals(ni_rows + shares_rows + future_correction),
    )
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 5.0}])

    result = add_valuation(panel, lake)

    row = result.row(0, named=True)
    # mcap = 100 * 5 = 500. ep_ttm = 50(정정 전 FY 값) / 500 = 0.1
    assert row["ep_ttm"] == pytest.approx(0.1)
    assert row["ep_ttm_isna"] is False


def test_add_valuation_cfp_ttm_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    ocf_rows = _quarterly_rows(
        1,
        "NetCashProvidedByUsedInOperatingActivities",
        fy_start=date(2019, 1, 1),
        q1=8,
        q2cum=18,
        q3cum=30,
        fy_val=44,
    )
    shares_rows = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 100.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(ocf_rows + shares_rows))
    before = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 4.0}])

    result = add_valuation(before, lake)

    row = result.row(0, named=True)
    assert row["cfp_ttm"] == pytest.approx(44.0 / 400.0)
    assert row["cfp_ttm_isna"] is False


def test_add_valuation_sp_ttm_pit_with_revenue_fallback(tmp_path: Path, lake: UsLake) -> None:
    """매출 태그 폴백 + PIT — ``Revenues``가 없는 회사는
    ``RevenueFromContractWithCustomerExcludingAssessedTax``로 대신한다."""
    revenue_rows = _quarterly_rows(
        1,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        fy_start=date(2019, 1, 1),
        q1=100,
        q2cum=210,
        q3cum=330,
        fy_val=460,
    )
    shares_rows = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 46.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]
    _write_snapshot(
        tmp_path, "fundamentals", "2026-09-19", _fundamentals(revenue_rows + shares_rows)
    )
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_valuation(panel, lake)

    row = result.row(0, named=True)
    # mcap = 46 * 10 = 460. sp_ttm = 460(FY 매출, 폴백 태그) / 460 = 1.0
    assert row["sp_ttm"] == pytest.approx(1.0)
    assert row["sp_ttm_isna"] is False


# =============================================================================
# add_profitability (F6)
# =============================================================================


def _profitability_shares_and_assets(cik: int, assets_val: float) -> list[dict]:
    return [
        {
            "cik": cik,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": assets_val,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]


def test_add_profitability_roa_ttm_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    ni_rows = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    assets_rows = _profitability_shares_and_assets(1, 500.0)
    future_ni_correction = [
        {
            "cik": 1,
            "tag": "NetIncomeLoss",
            "fp": "FY",
            "start": date(2019, 1, 1),
            "end": date(2019, 12, 31),
            "val": 999.0,
            "filed": date(2021, 1, 1),
            "form": "10-K/A",
        }
    ]
    _write_snapshot(
        tmp_path,
        "fundamentals",
        "2026-09-19",
        _fundamentals(ni_rows + assets_rows + future_ni_correction),
    )
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_profitability(panel, lake)

    row = result.row(0, named=True)
    assert row["roa_ttm"] == pytest.approx(50.0 / 500.0)
    assert row["roa_ttm_isna"] is False


def test_add_profitability_roe_ttm_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    ni_rows = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    equity_rows = [
        {
            "cik": 1,
            "tag": "StockholdersEquity",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 200.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "StockholdersEquity",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 999.0,
            "filed": date(2021, 1, 1),  # 미래 정정
            "form": "10-K/A",
        },
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(ni_rows + equity_rows))
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_profitability(panel, lake)

    row = result.row(0, named=True)
    assert row["roe_ttm"] == pytest.approx(50.0 / 200.0)
    assert row["roe_ttm_isna"] is False


def test_add_profitability_gpa_pit_with_fallback(tmp_path: Path, lake: UsLake) -> None:
    """``GrossProfit``이 없는 회사는 ``Revenues − CostOfRevenue``로 대신한다.

    PIT: 매출·매출원가 중 하나라도 미래에 filed된 정정이면 안 쓴다.
    """
    revenue_rows = _quarterly_rows(
        1, "Revenues", fy_start=date(2019, 1, 1), q1=100, q2cum=210, q3cum=330, fy_val=460
    )
    cost_rows = _quarterly_rows(
        1, "CostOfRevenue", fy_start=date(2019, 1, 1), q1=60, q2cum=126, q3cum=198, fy_val=276
    )
    assets_rows = _profitability_shares_and_assets(1, 400.0)
    future_cost_correction = [
        {
            "cik": 1,
            "tag": "CostOfRevenue",
            "fp": "FY",
            "start": date(2019, 1, 1),
            "end": date(2019, 12, 31),
            "val": 1.0,
            "filed": date(2021, 1, 1),
            "form": "10-K/A",
        }
    ]
    _write_snapshot(
        tmp_path,
        "fundamentals",
        "2026-09-19",
        _fundamentals(revenue_rows + cost_rows + assets_rows + future_cost_correction),
    )
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_profitability(panel, lake)

    row = result.row(0, named=True)
    # GrossProfit 태그가 아예 없어 폴백: (460 - 276) / 400 = 0.46
    assert row["gpa"] == pytest.approx((460.0 - 276.0) / 400.0)
    assert row["gpa_isna"] is False


def test_add_profitability_opm_ttm_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    oper_rows = _quarterly_rows(
        1, "OperatingIncomeLoss", fy_start=date(2019, 1, 1), q1=20, q2cum=42, q3cum=66, fy_val=92
    )
    revenue_rows = _quarterly_rows(
        1, "Revenues", fy_start=date(2019, 1, 1), q1=100, q2cum=210, q3cum=330, fy_val=460
    )
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(oper_rows + revenue_rows))
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_profitability(panel, lake)

    row = result.row(0, named=True)
    assert row["opm_ttm"] == pytest.approx(92.0 / 460.0)
    assert row["opm_ttm_isna"] is False


# =============================================================================
# add_investment (F7)
# =============================================================================


def test_add_investment_asset_growth_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    rows = [
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 1000.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2020, 12, 31),
            "val": 1200.0,
            "filed": date(2021, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "Assets",
            "fp": "FY",
            "start": None,
            "end": date(2020, 12, 31),
            "val": 9999.0,  # 미래 정정
            "filed": date(2022, 1, 1),
            "form": "10-K/A",
        },
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel([{"date": date(2021, 3, 10), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_investment(panel, lake)

    row = result.row(0, named=True)
    assert row["asset_growth"] == pytest.approx(1200.0 / 1000.0 - 1.0)
    assert row["asset_growth_isna"] is False


def test_add_investment_accruals_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    ni_rows = _quarterly_rows(
        1, "NetIncomeLoss", fy_start=date(2019, 1, 1), q1=10, q2cum=22, q3cum=35, fy_val=50
    )
    ocf_rows = _quarterly_rows(
        1,
        "NetCashProvidedByUsedInOperatingActivities",
        fy_start=date(2019, 1, 1),
        q1=8,
        q2cum=18,
        q3cum=30,
        fy_val=44,
    )
    assets_rows = _profitability_shares_and_assets(1, 400.0)
    _write_snapshot(
        tmp_path, "fundamentals", "2026-09-19", _fundamentals(ni_rows + ocf_rows + assets_rows)
    )
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_investment(panel, lake)

    row = result.row(0, named=True)
    assert row["accruals"] == pytest.approx((50.0 - 44.0) / 400.0)
    assert row["accruals_isna"] is False


def test_add_investment_net_issuance_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    rows = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 100.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2020, 12, 31),
            "val": 110.0,
            "filed": date(2021, 3, 1),
            "form": "10-K",
        },
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2020, 12, 31),
            "val": 500.0,  # 미래 정정
            "filed": date(2022, 1, 1),
            "form": "10-K/A",
        },
    ]
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals(rows))
    panel = _panel([{"date": date(2021, 3, 10), "symbol": "AAA", "cik": 1, "close": 10.0}])

    result = add_investment(panel, lake)

    row = result.row(0, named=True)
    assert row["net_issuance"] == pytest.approx(110.0 / 100.0 - 1.0)
    assert row["net_issuance_isna"] is False


# =============================================================================
# add_payout (F8)
# =============================================================================


def _write_corp_actions(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={"symbol": pl.String, "ex_date": pl.Date, "kind": pl.String, "amount": pl.Float64},
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


def test_add_payout_div_yield_zero_when_no_dividend_and_pit_on_ex_date(
    tmp_path: Path, lake: UsLake
) -> None:
    """무배당은 결측(0)이고, ``ex_date``가 t 뒤인 배당은 그 12개월 합에 안 낀다."""
    _write_corp_actions(
        tmp_path,
        [
            {"symbol": "AAA", "ex_date": date(2020, 1, 15), "kind": "dividend", "amount": 0.5},
            {"symbol": "AAA", "ex_date": date(2020, 6, 15), "kind": "dividend", "amount": 0.5},
            # t 뒤 배당 -- 미래 정보라 12개월 합에 들어가면 안 된다
            {"symbol": "AAA", "ex_date": date(2020, 12, 15), "kind": "dividend", "amount": 0.5},
            {"symbol": "NODIV", "ex_date": date(2010, 1, 1), "kind": "split", "amount": None},
        ],
    )
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", _fundamentals([]))
    panel = _panel(
        [
            {"date": date(2020, 9, 1), "symbol": "AAA", "cik": 1, "close": 20.0},
            {"date": date(2020, 9, 1), "symbol": "NODIV", "cik": 2, "close": 20.0},
        ]
    )

    result = add_payout(panel, lake).sort("symbol")

    values = dict(zip(result["symbol"].to_list(), result["div_yield"].to_list()))
    isna = dict(zip(result["symbol"].to_list(), result["div_yield_isna"].to_list()))
    assert values["AAA"] == pytest.approx(1.0 / 20.0)  # 0.5+0.5만 — 12/15 배당은 미래라 빠진다
    assert isna["AAA"] is False
    assert values["NODIV"] == pytest.approx(0.0)  # 무배당은 0
    assert isna["NODIV"] is False


def test_add_payout_buyback_yield_pit_ignores_future_filed(tmp_path: Path, lake: UsLake) -> None:
    buyback_rows = _quarterly_rows(
        1,
        "PaymentsForRepurchaseOfCommonStock",
        fy_start=date(2019, 1, 1),
        q1=5,
        q2cum=9,
        q3cum=14,
        fy_val=20,
    )
    shares_rows = [
        {
            "cik": 1,
            "tag": "EntityCommonStockSharesOutstanding",
            "fp": "FY",
            "start": None,
            "end": date(2019, 12, 31),
            "val": 10.0,
            "filed": date(2020, 3, 1),
            "form": "10-K",
        }
    ]
    _write_snapshot(
        tmp_path, "fundamentals", "2026-09-19", _fundamentals(buyback_rows + shares_rows)
    )
    _write_corp_actions(tmp_path, [])
    panel = _panel([{"date": date(2020, 3, 5), "symbol": "AAA", "cik": 1, "close": 8.0}])

    result = add_payout(panel, lake)

    row = result.row(0, named=True)
    # mcap = 10 * 8 = 80. buyback_yield = 20(FY 자사주 매입 TTM) / 80 = 0.25
    assert row["buyback_yield"] == pytest.approx(0.25)
    assert row["buyback_yield_isna"] is False
