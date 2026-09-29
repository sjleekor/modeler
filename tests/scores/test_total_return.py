from datetime import date

import polars as pl

from modeler.etl.config import DataRoot
from modeler.scores.common.assets import get_asset
from modeler.scores.common.inputs import PinnedScopedLake
from modeler.scores.common.total_return import (
    compute_total_return_path,
    dividend_coverage,
    load_us_total_return,
)
from modeler.scores.market_sector.labels import compute_labels
from tests.scores._helpers import make_cal, weekdays


def _px(n: int, price: float = 100.0) -> pl.DataFrame:
    ds = weekdays(date(2024, 1, 2), n)
    return pl.DataFrame({"session": ds, "px_adj": [price] * n})


def test_dividend_reinvested_at_ex_date_close():
    px = _px(5)
    div = pl.DataFrame({"ex_date": [px["session"][2]], "div_adj": [2.0]})
    path, _ = compute_total_return_path(px, div)
    assert path["return_basis"][0] == "total_return"
    assert path["tr_index"].to_list() == [1.0, 1.0, 1.02, 1.02, 1.02]


def test_price_only_when_no_dividends():
    path, _ = compute_total_return_path(_px(3), None)
    assert path["return_basis"].unique().to_list() == ["price_only"]
    assert path["tr_index"].to_list() == [1.0, 1.0, 1.0]


def test_entry_day_dividend_not_received_but_next_day_is():
    cal = make_cal(10)
    px = pl.DataFrame({"session": cal.sessions, "px_adj": [100.0] * 10})
    # 라벨: t=세션0, 진입=세션1, 만기=세션2 (horizon=1)
    on_entry = pl.DataFrame({"ex_date": [cal.sessions[1]], "div_adj": [2.0]})
    after_entry = pl.DataFrame({"ex_date": [cal.sessions[2]], "div_adj": [2.0]})
    p1, _ = compute_total_return_path(px, on_entry)
    p2, _ = compute_total_return_path(px, after_entry)
    l1 = compute_labels("a", cal, p1, cash=None, horizon=1).filter(
        pl.col("session") == cal.sessions[0]
    )
    l2 = compute_labels("a", cal, p2, cash=None, horizon=1).filter(
        pl.col("session") == cal.sessions[0]
    )
    assert l1["total_return_60d"][0] == 0.0  # 진입일 배당은 받지 않는다
    assert abs(l2["total_return_60d"][0] - 0.02) < 1e-12  # 진입 뒤 배당은 재투자


def test_dividend_on_non_session_attributed_to_next_and_out_of_range_dropped():
    px = _px(6)
    sat = date(2024, 1, 6)  # 토요일 -> 다음 세션(월 01-08)
    div = pl.DataFrame({"ex_date": [sat, date(2030, 1, 1)], "div_adj": [1.0, 5.0]})
    path, diag = compute_total_return_path(px, div)
    assert diag["dividends_shifted"] == 1 and diag["dividends_unattributed"] == 1
    assert path.filter(pl.col("session") == date(2024, 1, 8))["div_adj"][0] == 1.0


def test_dividend_coverage_flags_stale_recent():
    ex = [date(2025, 3, 20), date(2025, 6, 20), date(2025, 9, 19), date(2025, 12, 19)]
    cov = dividend_coverage(pl.DataFrame({"ex_date": ex}), last_price_session=date(2026, 6, 30))
    assert cov["dividend_rows"] == 4 and cov["possibly_missing_recent"] is True
    ok = dividend_coverage(pl.DataFrame({"ex_date": ex}), last_price_session=date(2026, 1, 5))
    assert ok["possibly_missing_recent"] is False
    assert dividend_coverage(None, last_price_session=None)["dividend_rows"] == 0


def _write(root, table, snap, df: pl.DataFrame):
    d = root.derived / "snapshots" / table / f"snapshot_date={snap}"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part.parquet")


def test_split_adjusted_dividend_is_continuous_across_split(tmp_path, monkeypatch):
    """2:1 분할(ex 01-10) 전 배당 amount(분할 전 주식 수 기준)는 0.5배로 조정된다."""
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    root = DataRoot.resolve("us")
    ds = weekdays(date(2024, 1, 2), 12)
    close = [100.0] * 6 + [50.0] * 6  # 분할 ex_date = ds[6]
    _write(
        root,
        "prices_daily",
        "2024-02-01",
        pl.DataFrame(
            {
                "date": ds,
                "symbol": ["ETF"] * 12,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": [1000] * 12,
            }
        ).with_columns(pl.exclude("date", "symbol", "volume").cast(pl.Decimal(14, 4))),
    )
    corp = pl.DataFrame(
        {
            "symbol": ["ETF", "ETF"],
            "ex_date": [ds[3], ds[6]],
            "kind": ["dividend", "split"],
            "to_factor": [None, 2.0],
            "for_factor": [None, 1.0],
            "amount": [2.0, None],
        }
    ).with_columns(pl.col("to_factor", "for_factor", "amount").cast(pl.Decimal(10, 5)))
    _write(root, "corp_actions", "2024-02-01", corp)
    lake = PinnedScopedLake(
        root=root,
        snapshots={"prices_daily": "2024-02-01", "corp_actions": "2024-02-01"},
        symbols=("ETF",),
    )
    paths, diags = load_us_total_return(lake, {"x": "ETF"})
    p = paths["x"].sort("session")
    assert diags["x"]["dividend_rows"] == 1
    assert p["px_adj"].to_list() == [50.0] * 12  # 분할 조정
    # 배당 2.0(분할 전 기준) -> D* = 1.0 on 조정가 50 -> 2%
    assert abs(p.filter(pl.col("session") == ds[3])["div_adj"][0] - 1.0) < 1e-9
    assert abs(p["tr_index"][3] - 1.02) < 1e-9
    assert abs(p["tr_index"][-1] - 1.02) < 1e-9  # 분할일에 수익 점프가 없다
    assert p["return_basis"][0] == "total_return"
    _ = get_asset  # 레지스트리 import 확인용 참조
