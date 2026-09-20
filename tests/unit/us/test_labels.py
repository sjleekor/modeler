"""``modeler.us.labels`` 단위 테스트. ``tmp_path``에 합성 parquet을 쓴다."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
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


def test_neutralize_cross_section_residual_zero_for_any_monotonic_size_effect() -> None:
    """``L1``이 ``adv_20d``의 순수 단조함수면(모양과 무관하게) 잔차가 거의 0이어야 한다.

    2026-09-20 순위 공간 정정 전에는 이 테스트가 "10분위 더미 + sic2 더미로
    걷을 수 있는 순수 계단형 효과"를 박아 뒀다 — 그 정의(수준 공간, 10분위
    더미)를 이제 안 쓴다. 새 정의에서는 ``x=percentile_rank(log(adv_20d))``·
    ``y=percentile_rank(L1)``이라, ``L1``이 ``adv_20d``에 대해 **단조증가**면
    동순위가 없는 한 ``y``가 ``x``와 정확히 같아진다(둘 다 같은 순서를 그대로
    반영하는 백분위이므로) — 그러면 회귀의 ``x`` 항 하나만으로 정확히 설명되고
    (계수 1, 나머지 0인 해가 존재), ``x^2``·sic2 더미와 무관하게 잔차가 0이
    된다. 사이즈 효과가 선형이든 3차든 로그든 상관없다는 뜻이라, 예전 계단형
    가정보다 오히려 더 일반적인 보장이다. sic2 더미가 이 항등식을 깨지 않도록
    여기서는 sic2를 하나로 고정한다(더미 분리 자체는
    ``test_neutralize_cross_section_minority_sic2_does_not_perfectly_fit``가 딴다).
    """
    n = 200
    rows = []
    for k in range(n):
        adv = math.exp(k * 1e-3)  # 오름차순 고유값 -> rank(log(adv))-1 == k
        l1 = math.log1p(k) ** 3 + 7.0  # adv에 대해 순수 단조(모양은 무관하다)
        rows.append({"adv_20d": adv, "mcap_rank": None, "sic2": "A", "L1": l1})
    df = pl.DataFrame(rows)

    result = neutralize_cross_section(df)

    assert abs(result["L2"].sum()) < 1e-8
    assert result["L2"].abs().max() < 1e-8


def test_neutralize_cross_section_removes_pure_linear_size_effect_rank_correlation() -> None:
    """``L1``이 사이즈에 순수 선형이면, ``L2``와 ``log(adv_20d)``의 순위상관이 0에
    가까워야 한다 — 게이트(``mean|ρ| < 0.03``)가 실제로 재는 지표다.

    ``x``와 완전히 동순위 없는 순수 단조 관계만 주면(잡음 없음) 회귀가
    정확히 적합돼(``test_neutralize_cross_section_residual_zero_for_any_monotonic_size_effect``
    참고) 잔차가 부동소수점 오차(1e-16 수준) 뿐이다 — 이 크기의 "신호"에
    순위상관을 재면 진짜 관계가 아니라 반올림 오차의 우연한 패턴을 재는
    꼴이라(``mcap_rank`` 항 추가로 설계행렬 열이 하나 더 늘기만 해도 반올림
    패턴이 바뀌어 값이 크게 흔들린다 — 2026-09-20 실측), 아주 작은(``0.1``)
    독립 잡음을 더해 잔차가 실제 크기를 갖게 한다.
    """
    n = 500
    rng = np.random.default_rng(0)
    x = np.array([k / (n - 1) for k in range(n)])
    l1 = 3.0 * x + 0.7 + rng.normal(scale=0.1, size=n)
    df = pl.DataFrame(
        {
            "adv_20d": [math.exp(k * 1e-3) for k in range(n)],
            "mcap_rank": [None] * n,
            "sic2": ["A" if k % 2 == 0 else "B" for k in range(n)],
            "L1": l1.tolist(),
        }
    )

    result = neutralize_cross_section(df)

    rho = np.corrcoef(
        result["L2"].rank(method="average").to_numpy(),
        result["adv_20d"].log().rank(method="average").to_numpy(),
    )[0, 1]
    assert abs(rho) < 0.05


def test_neutralize_cross_section_quadratic_term_reduces_residual_for_hump_shaped_effect() -> None:
    """``x^2`` 항이 실제로 회귀에 들어가는지 확인한다.

    순위 공간에서 뚜렷한 오목(가운데가 높은, hump형) 크기효과를 주면 1차항
    (``x``)만으로는 못 걷는다 — [0,1]에서 ``x``만으로 표현 가능한 함수는
    전부 단조라, 오목한 모양 자체를 설명할 수 없기 때문이다. 실제 구현(``x``
    와 ``x^2``를 같이 회귀)과, ``x^2``을 뺀 대조군을 직접 풀어 비교한다.
    """
    n = 400
    x = np.array([k / (n - 1) for k in range(n)])
    sic = np.array(["A" if k % 2 == 0 else "B" for k in range(n)])
    # 오목한 패턴 + 동순위를 막는 아주 작은(1e-9) 선형 항.
    l1 = (x - x**2) + 1e-9 * np.arange(n)
    df = pl.DataFrame(
        {
            "adv_20d": [math.exp(k * 1e-3) for k in range(n)],
            "mcap_rank": [None] * n,
            "sic2": sic.tolist(),
            "L1": l1.tolist(),
        }
    )

    result = neutralize_cross_section(df)
    quad_sse = float((result["L2"].to_numpy() ** 2).sum())

    # 대조군: x^2 없이(절편 + x + sic2 더미만) 같은 y를 직접 회귀한다.
    n_rows = df.height
    denom = max(n_rows - 1, 1)
    y = (pl.Series(l1.tolist()).rank(method="min").cast(pl.Float64) - 1) / denom
    dummy = np.array([1.0 if s == "B" else 0.0 for s in sic])
    design_linear = np.column_stack([np.ones(n_rows), x, dummy])
    beta_lin, *_ = np.linalg.lstsq(design_linear, y.to_numpy(), rcond=None)
    linear_sse = float(((y.to_numpy() - design_linear @ beta_lin) ** 2).sum())

    assert quad_sse < linear_sse * 0.5


def test_neutralize_cross_section_minority_sic2_does_not_perfectly_fit() -> None:
    """20종목 미만 sic2는 '기타'로 묶여야 한다 — 안 묶이면 그 종목들의 잔차가 0이 된다."""
    rows = [
        {"adv_20d": 1_000.0 + i, "mcap_rank": None, "sic2": "BIG", "L1": 0.0} for i in range(30)
    ]
    tiny_l1 = [0.5, -0.3, 0.9]
    rows += [
        {"adv_20d": 2_000.0 + i, "mcap_rank": None, "sic2": "TINY", "L1": v}
        for i, v in enumerate(tiny_l1)
    ]
    df = pl.DataFrame(rows)

    result = neutralize_cross_section(df)

    tiny = result.filter(pl.col("sic2") == "TINY")
    assert tiny["sic2_bucket"].to_list() == [OTHER_SIC2] * 3
    # "기타"로 묶였으니 그룹 평균만 제거되고, 서로 다른 잔차가 남아야 한다
    # (안 묶였으면 회귀가 완전 적합돼 셋 다 잔차 0이 된다).
    residuals = {round(v, 9) for v in tiny["L2"].to_list()}
    assert len(residuals) > 1


# --- neutralize_cross_section: mcap_rank 항 (2026-09-20 추가) ---------------------


def test_neutralize_cross_section_mcap_rank_ranked_among_present_rows_only() -> None:
    """``m``은 그날 ``mcap_rank``가 있는 행끼리만 순위를 매겨야 한다.

    전체 횡단면(결측 포함)으로 매기면 분모(n-1)가 부풀어 있는 값들의 상대
    순위가 압축된다 — 여기서는 결측 20개, 있는 값 5개(균등 간격)를 섞어서,
    있는 값의 백분위가 **5개 기준**(분모 4)으로 나오는지 본다. 전체 25개
    기준(분모 24)으로 나오면 버그다. 결측 행은 ``mcap_rank_pct=0``·
    ``has_mcap=False``를 받아야 한다 — 회귀의 ``m``·``m^2`` 항이 그 행에는
    구조적으로 0을 곱하는 셈이라, 어떤 계수가 나오든 그 항의 영향을 안 받는다.
    """
    present_ranks = [10, 20, 30, 40, 50]
    n_missing = 20
    rows = [
        {"adv_20d": 1_000.0 + i, "mcap_rank": r, "sic2": "A", "L1": float(i)}
        for i, r in enumerate(present_ranks)
    ]
    rows += [
        {"adv_20d": 2_000.0 + i, "mcap_rank": None, "sic2": "A", "L1": float(i) * 0.1}
        for i in range(n_missing)
    ]
    df = pl.DataFrame(rows)

    result = neutralize_cross_section(df)

    present = result.filter(pl.col("mcap_rank").is_not_null()).sort("mcap_rank")
    assert present["mcap_rank_pct"].to_list() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert present["has_mcap"].to_list() == [True] * 5

    missing = result.filter(pl.col("mcap_rank").is_null())
    assert missing["mcap_rank_pct"].to_list() == [0.0] * n_missing
    assert missing["has_mcap"].to_list() == [False] * n_missing


def test_neutralize_cross_section_removes_pure_linear_mcap_size_effect_rank_correlation() -> None:
    """``L1``이 ``mcap_rank``에 순수 선형이면, ``L2``와 ``mcap_rank``의 순위상관이
    0에 가까워야 한다 — ``m``·``m^2`` 항을 추가한 이유 그 자체를 검증한다
    (``02`` §2 실측 — ADV 축만으로는 ``mean|ρ(L2, mcap_rank)|`` 0.0764로 미달).

    작은(``0.1``) 독립 잡음을 더한다 — 잡음이 아예 없으면 회귀가 정확히
    적합돼 잔차가 부동소수점 오차 수준이라, 그 순위상관이 반올림 오차의
    우연한 패턴이 된다(ADV 축의 같은 꼴 테스트 참고).
    """
    n = 500
    mcap = np.arange(n)
    rng = np.random.default_rng(0)
    l1 = 3.0 * (mcap / (n - 1)) + 0.7 + rng.normal(scale=0.1, size=n)
    adv = rng.permutation(n).astype(float) + 1.0  # mcap 순서와 무관한 순열
    df = pl.DataFrame(
        {
            "adv_20d": adv.tolist(),
            "mcap_rank": mcap.tolist(),
            "sic2": ["A" if k % 2 == 0 else "B" for k in range(n)],
            "L1": l1.tolist(),
        }
    )

    result = neutralize_cross_section(df)

    rho = np.corrcoef(
        result["L2"].rank(method="average").to_numpy(),
        result["mcap_rank"].rank(method="average").to_numpy(),
    )[0, 1]
    assert abs(rho) < 0.05


def test_add_l2_applies_independently_per_date() -> None:
    rows = []
    for d, shift in [(date(2020, 1, 1), 0.0), (date(2020, 2, 1), 10.0)]:
        for i in range(25):
            rows.append(
                {
                    "date": d,
                    "adv_20d": 1_000.0 + i,
                    "mcap_rank": None,
                    "sic2": "A" if i % 2 == 0 else "B",
                    "L1": shift + (i % 3) * 0.01,
                }
            )
    df = pl.DataFrame(rows)

    result = add_l2(df)

    assert result["date"].n_unique() == 2
    means = result.group_by("date").agg(pl.col("L2").mean().alias("m"))
    assert means["m"].abs().max() < 1e-8


def test_add_l2_ranks_within_each_date_not_pooled_globally() -> None:
    """순위 변환이 그날 횡단면 '안'에서만 되는지 확인한다.

    두 날짜를 합친 뒤 날짜 구분 없이 전역으로 순위를 매기면(잘못된 구현) 각
    날짜를 따로 계산한 것과 다른 ``L2``가 나온다 — ``adv_20d`` 절대 스케일이
    날짜마다 크게 다르면(여기서는 1.0 vs 1000.0) 전역 순위가 각 날짜 안의
    상대 순위를 압축해 x 범위가 [0,1]이 아니게 되기 때문이다. ``add_l2``는
    날짜별로 따로 계산해야(``partition_by("date")``) 이 문제가 없다.
    """
    n = 20
    rows = []
    for d, scale in [(date(2020, 1, 1), 1.0), (date(2020, 2, 1), 1_000.0)]:
        for k in range(n):
            rows.append(
                {
                    "date": d,
                    "adv_20d": scale + k,
                    "mcap_rank": None,
                    "sic2": "A" if k % 2 == 0 else "B",
                    "L1": math.sin(k * 0.3),  # 임의의 비단조 패턴
                }
            )
    combined = pl.DataFrame(rows)

    per_date_result = add_l2(combined)

    # 날짜별로 직접 호출해 이어붙인 것과 완전히 같아야 한다.
    expected_parts = [
        neutralize_cross_section(combined.filter(pl.col("date") == d).drop("date"))
        for d in combined["date"].unique(maintain_order=True)
    ]
    expected_l2 = np.concatenate([p["L2"].to_numpy() for p in expected_parts])
    assert per_date_result["L2"].to_numpy() == pytest.approx(expected_l2)

    # 대조군: 두 날짜를 구분 없이 하나의 횡단면으로 (잘못) 합쳐 순위를 매기면
    # 값이 달라진다 — add_l2가 실제로 날짜별로 분리해서 도는지의 증거다.
    pooled = neutralize_cross_section(combined.drop("date"))
    assert not np.allclose(pooled["L2"].to_numpy(), per_date_result["L2"].to_numpy())


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


def test_build_labels_l0_and_l1_unchanged_by_rank_space_l2(tmp_path: Path, lake: UsLake) -> None:
    """``L2`` 중립화를 순위 공간으로 바꿔도 ``L0``·``L1``은 그대로여야 한다.

    ``L0``(개별 종목 수익)·``L1``(그날 유니버스 동일가중 초과수익)은 중립화
    구현과 독립이다 — 백테스트 수익이 이 둘에서 나오므로 바뀌면 안 된다.
    """
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            _price_row(_T, "AAA", 10.0),
            _price_row(_T21, "AAA", 12.0),  # L0 = 0.20
            _price_row(_T, "BBB", 20.0),
            _price_row(_T21, "BBB", 19.0),  # L0 = -0.05
            _price_row(_T, "CCC", 5.0),
            _price_row(_T21, "CCC", 5.5),  # L0 = 0.10
        ],
    )
    panel = _panel_df(
        [
            _panel_row(_T, "AAA", close=10.0, adj_close=10.0),
            _panel_row(_T, "BBB", close=20.0, adj_close=20.0),
            _panel_row(_T, "CCC", close=5.0, adj_close=5.0),
        ]
    )

    labels, _ = build_labels(lake, panel=panel)

    l0 = {row["symbol"]: row["L0"] for row in labels.to_dicts()}
    assert l0["AAA"] == pytest.approx(0.20)
    assert l0["BBB"] == pytest.approx(-0.05)
    assert l0["CCC"] == pytest.approx(0.10)

    expected_mean = (0.20 + (-0.05) + 0.10) / 3
    l1 = {row["symbol"]: row["L1"] for row in labels.to_dicts()}
    for symbol, l0_value in l0.items():
        assert l1[symbol] == pytest.approx(l0_value - expected_mean)


def test_build_labels_l0_and_l1_unchanged_when_mcap_rank_term_added(
    tmp_path: Path, lake: UsLake
) -> None:
    """``mcap_rank`` 항(``m``·``m^2``·``has_mcap``)을 더해도 ``L0``·``L1``은 그대로여야 한다.

    사이즈 항은 ``L2`` 중립화에만 들어간다 — ``L0``·``L1``은 애초에 ``mcap_rank``를
    안 쓴다. ``mcap_rank``가 있는 종목과 없는 종목을 섞어서 확인한다.
    """
    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            _price_row(_T, "AAA", 10.0),
            _price_row(_T21, "AAA", 12.0),  # L0 = 0.20
            _price_row(_T, "BBB", 20.0),
            _price_row(_T21, "BBB", 19.0),  # L0 = -0.05
            _price_row(_T, "CCC", 5.0),
            _price_row(_T21, "CCC", 5.5),  # L0 = 0.10
        ],
    )
    panel = _panel_df(
        [
            _panel_row(_T, "AAA", close=10.0, adj_close=10.0, mcap_rank=1),
            _panel_row(_T, "BBB", close=20.0, adj_close=20.0, mcap_rank=2),
            _panel_row(_T, "CCC", close=5.0, adj_close=5.0, mcap_rank=None),  # 결측도 섞는다
        ]
    )

    labels, _ = build_labels(lake, panel=panel)

    l0 = {row["symbol"]: row["L0"] for row in labels.to_dicts()}
    assert l0["AAA"] == pytest.approx(0.20)
    assert l0["BBB"] == pytest.approx(-0.05)
    assert l0["CCC"] == pytest.approx(0.10)

    expected_mean = (0.20 + (-0.05) + 0.10) / 3
    l1 = {row["symbol"]: row["L1"] for row in labels.to_dicts()}
    for symbol, l0_value in l0.items():
        assert l1[symbol] == pytest.approx(l0_value - expected_mean)


def test_build_labels_counts_constant_has_mcap_months(tmp_path: Path, lake: UsLake) -> None:
    """``has_mcap``이 그 달 전체에서 상수(전부 1이거나 전부 0)인 달 수를 센다.

    그 열이 상수면 절편과 완전공선이 된다(``labels.py`` ``neutralize_cross_section``
    참고) — ``lstsq``가 최소노름해로 처리해 죽지는 않지만, 몇 달인지는 세서
    보고해야 한다.
    """
    date1 = _T
    date2 = _CALENDAR[1]
    date1_t21 = _CALENDAR[21]
    date2_t21 = _CALENDAR[22]

    _write_trading_calendar(tmp_path, _CALENDAR)
    _write_corp_actions(tmp_path)
    _write_listing_snapshots(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            _price_row(date1, "A1", 10.0),
            _price_row(date1_t21, "A1", 11.0),
            _price_row(date1, "A2", 20.0),
            _price_row(date1_t21, "A2", 21.0),
            _price_row(date2, "B1", 30.0),
            _price_row(date2_t21, "B1", 31.0),
            _price_row(date2, "B2", 40.0),
            _price_row(date2_t21, "B2", 41.0),
        ],
    )
    panel = _panel_df(
        [
            # date1: mcap_rank가 전부 있다 (has_mcap 전부 True — 상수).
            _panel_row(date1, "A1", close=10.0, adj_close=10.0, mcap_rank=1),
            _panel_row(date1, "A2", close=20.0, adj_close=20.0, mcap_rank=2),
            # date2: mcap_rank가 전부 없다 (has_mcap 전부 False — 상수).
            _panel_row(date2, "B1", close=30.0, adj_close=30.0, mcap_rank=None),
            _panel_row(date2, "B2", close=40.0, adj_close=40.0, mcap_rank=None),
        ]
    )

    _, diag = build_labels(lake, panel=panel)

    assert diag["constant_has_mcap_months"] == 2
