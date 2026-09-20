"""``modeler.us.features``의 F1·F2·F3·F4·F13·F16(가격계) 단위 테스트.

``test_lake.py``·``test_prices.py``와 같은 관례로 ``tmp_path``에 합성 parquet을
쓴다. 실제 레이크는 읽지 않는다.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.features._daily import daily_prices, mask_ticker_reuse_gap, with_market_return
from modeler.us.features.calendar import add_calendar
from modeler.us.features.liquidity import add_liquidity
from modeler.us.features.momentum import add_momentum
from modeler.us.features.options_iv import add_options_iv
from modeler.us.features.reversal import add_reversal
from modeler.us.features.volatility import add_volatility
from modeler.us.lake import UsLake

# --- 공용 헬퍼 -------------------------------------------------------------------


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


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


def _write_no_splits(tmp_path: Path) -> None:
    _write_corp_actions(tmp_path, [])


def _write_trading_calendar(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "exchange": pl.String, "is_early_close": pl.Boolean},
    )
    _write_snapshot(tmp_path, "trading_calendar", "2026-09-19", frame)


def _write_midas(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "security_type": pl.String,
            "turn_rank": pl.Int32,
        },
    )
    _write_snapshot(tmp_path, "midas_security_daily", "2026-09-18", frame)


def _write_volatility(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "iv_current": pl.Float64,
            "iv_year_high": pl.Float64,
            "iv_year_low": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "volatility_daily", "2026-09-18", frame)


def _price_series(
    symbol: str, start: date, closes: list[float], *, day_step: int = 1
) -> list[dict]:
    """``symbol``의 (date, close) 연속 시리즈. open/high/low는 close와 같게,
    volume은 1,000으로 고정한다 — 가격계 테스트는 종가·거래량만 본다."""
    return [
        {
            "date": start + timedelta(days=i * day_step),
            "symbol": symbol,
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1_000.0,
        }
        for i, c in enumerate(closes)
    ]


def _panel(rows: list[dict], *, mcap_rank: dict[tuple, int | None] | None = None) -> pl.DataFrame:
    """``(date, symbol)`` 최소 패널. ``mcap_rank``를 주면(F4 테스트용) 그 컬럼도 만든다."""
    df = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.String})
    if mcap_rank is not None:
        df = df.with_columns(
            pl.Series(
                "mcap_rank",
                [mcap_rank.get((r["date"], r["symbol"])) for r in rows],
                dtype=pl.Int32,
            )
        )
    return df


# --- momentum (F1) ---------------------------------------------------------------


def test_add_momentum_matches_hand_computed_ratios(tmp_path: Path, lake: UsLake) -> None:
    """일정한 일수익률 계열로 mom_12_1·mom_6_1·mom_1m을 손으로 검산한다."""
    n = 300
    closes = [100.0 * (1.001**i) for i in range(n)]
    start = date(2020, 1, 1)
    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)

    t_idx = 280
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])

    result = add_momentum(panel, lake).row(0, named=True)

    assert result["mom_12_1"] == pytest.approx(1.001**231 - 1.0, rel=1e-9)
    assert result["mom_6_1"] == pytest.approx(1.001**105 - 1.0, rel=1e-9)
    assert result["mom_1m"] == pytest.approx(1.001**21 - 1.0, rel=1e-9)
    assert not result["mom_12_1_isna"]
    assert not result["mom_6_1_isna"]
    assert not result["mom_1m_isna"]


def test_add_momentum_excludes_last_21_days_from_mom_12_1_and_mom_6_1(
    tmp_path: Path, lake: UsLake
) -> None:
    """t 당일 가격만 바꾸면 mom_1m은 바뀌어도 mom_12_1·mom_6_1은 바뀌면 안 된다
    — 두 피쳐 다 ``shift(21)``까지만 보고 최근 한 달은 빼기 때문이다."""
    n = 300
    start = date(2020, 1, 1)
    base_closes = [100.0 * (1.001**i) for i in range(n)]
    spiked_closes = list(base_closes)
    spiked_closes[-1] *= 5.0  # 마지막 날(t)만 5배로 튄다

    t_idx = n - 1
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])

    _write_prices(tmp_path, _price_series("AAA", start, base_closes))
    _write_no_splits(tmp_path)
    baseline = add_momentum(panel, lake).row(0, named=True)

    _write_prices(tmp_path, _price_series("AAA", start, spiked_closes))
    spiked = add_momentum(panel, lake).row(0, named=True)

    assert spiked["mom_12_1"] == pytest.approx(baseline["mom_12_1"], rel=1e-9)
    assert spiked["mom_6_1"] == pytest.approx(baseline["mom_6_1"], rel=1e-9)
    assert spiked["mom_1m"] != pytest.approx(baseline["mom_1m"], rel=1e-6)


def test_add_momentum_uses_adjusted_price_across_a_forward_split(
    tmp_path: Path, lake: UsLake
) -> None:
    """4:1 분할이 낀 계열에서 mom_1m이 원시 가격을 썼을 때 나오는 −75% 근처
    값이 아니어야 한다 — 조정 가격을 쓰면 분할은 수익률에 점프를 만들지 않는다."""
    start = date(2020, 1, 1)
    # 분할 전 25일은 완만한 상승, ex_date에 4:1, 이후 15일도 완만한 상승.
    pre = [400.0 * (1.001**i) for i in range(25)]
    post_start = pre[-1] * 1.001 / 4.0  # 분할 다음날 원시 가격(연속적인 조정가 유지)
    post = [post_start * (1.001**i) for i in range(15)]
    closes = pre + post
    ex_date = start + timedelta(days=25)

    _write_prices(tmp_path, _price_series("AAPL", start, closes))
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": ex_date,
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            }
        ],
    )

    t_idx = 35  # 분할 열흘 뒤 — mom_1m 창(t-21~t)이 분할을 가로지른다
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAPL"}])

    result = add_momentum(panel, lake).row(0, named=True)

    assert result["mom_1m"] > -0.5  # 원시 가격을 썼다면 -75% 근처가 나왔을 것


def test_add_momentum_isna_when_warmup_window_not_filled(tmp_path: Path, lake: UsLake) -> None:
    n = 15
    start = date(2020, 1, 1)
    closes = [100.0 + i for i in range(n)]
    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)

    panel = _panel([{"date": start + timedelta(days=10), "symbol": "AAA"}])
    result = add_momentum(panel, lake).row(0, named=True)

    assert result["mom_12_1_isna"]
    assert result["mom_6_1_isna"]
    assert result["mom_1m_isna"]
    assert result["mom_12_1"] is None
    assert result["mom_6_1"] is None
    assert result["mom_1m"] is None


def test_add_momentum_masks_ticker_reuse_gap(tmp_path: Path, lake: UsLake) -> None:
    """같은 티커가 몇 년 뒤 재사용된 것처럼 큰 달력 공백이 낀 계열에서,
    shift가 그 공백을 건너뛰면 mom_12_1·mom_6_1이 null이 돼야 한다(``mom_1m``은
    공백 안쪽만 보므로 영향이 없어야 한다) — 검산 5번에서 실제로 발견한 문제
    (``CBK`` 등)를 그대로 재현한다.
    """
    era_a_start = date(2010, 1, 1)
    era_a = _price_series("REUSED", era_a_start, [100.0 + i for i in range(300)])

    era_b_start = era_a_start + timedelta(days=300 + 1000)  # 1,000일 공백
    era_b = _price_series("REUSED", era_b_start, [1_000.0 + i for i in range(300)])

    _write_prices(tmp_path, era_a + era_b)
    _write_no_splits(tmp_path)

    panel_date = era_b_start + timedelta(days=100)  # era_b의 101번째 행(전체 401번째)
    panel = _panel([{"date": panel_date, "symbol": "REUSED"}])

    result = add_momentum(panel, lake).row(0, named=True)

    assert result["mom_12_1"] is None
    assert result["mom_12_1_isna"]
    assert result["mom_6_1"] is None
    assert result["mom_6_1_isna"]
    # mom_1m은 t-21이 era_b 안쪽(공백을 건너뛰지 않는다)이라 유효해야 한다.
    assert result["mom_1m"] is not None
    assert not result["mom_1m_isna"]


def test_add_momentum_pit_future_rows_do_not_change_past_value(
    tmp_path: Path, lake: UsLake
) -> None:
    """t 시점 값이 t 뒤 가격 행에 영향을 받으면 안 된다."""
    n = 300
    start = date(2020, 1, 1)
    closes = [100.0 * (1.001**i) for i in range(n)]
    t_idx = 280
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])

    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)
    baseline = add_momentum(panel, lake).row(0, named=True)

    # t 뒤에 완전히 다른(극단적인) 미래 가격을 붙인다.
    future_closes = closes + [9999.0, 1.0, 50000.0]
    _write_prices(tmp_path, _price_series("AAA", start, future_closes))
    with_future = add_momentum(panel, lake).row(0, named=True)

    assert with_future["mom_12_1"] == pytest.approx(baseline["mom_12_1"], rel=1e-9)
    assert with_future["mom_6_1"] == pytest.approx(baseline["mom_6_1"], rel=1e-9)
    assert with_future["mom_1m"] == pytest.approx(baseline["mom_1m"], rel=1e-9)


# --- reversal (F2) ----------------------------------------------------------------


def test_add_reversal_rev_1w_uses_t_minus_5(tmp_path: Path, lake: UsLake) -> None:
    n = 30
    start = date(2020, 1, 1)
    closes = [100.0 * (1.01**i) for i in range(n)]
    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)

    t_idx = 25
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])
    result = add_reversal(panel, lake).row(0, named=True)

    assert result["rev_1w"] == pytest.approx(1.01**5 - 1.0, rel=1e-9)


def test_add_reversal_max_ret_1m_is_rolling_max_over_21_days_including_today(
    tmp_path: Path, lake: UsLake
) -> None:
    """``max_ret_1m``이 당일 포함 21개 관측치의 최댓값인가 — 창 밖(22일 전)의
    더 큰 값은 무시해야 한다."""
    n = 30
    start = date(2020, 1, 1)
    closes = [100.0] * n
    # 창 밖(t-25, 21개 창에 안 들어옴): 아주 큰 튐.
    closes[4] = 100.0 * 3.0
    # 창 안(t-10): 이 창에서 최댓값이어야 하는 튐.
    closes[19] = 100.0 * 1.5
    for i in range(5, len(closes)):
        if i != 19:
            closes[i] = closes[i - 1] if i != 5 else 100.0

    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)

    t_idx = 29
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])
    result = add_reversal(panel, lake).row(0, named=True)

    assert result["max_ret_1m"] == pytest.approx(0.5, rel=1e-6)


def test_add_reversal_isna_when_warmup_window_not_filled(tmp_path: Path, lake: UsLake) -> None:
    start = date(2020, 1, 1)
    closes = [100.0, 101.0, 99.0]
    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)

    panel = _panel([{"date": start + timedelta(days=2), "symbol": "AAA"}])
    result = add_reversal(panel, lake).row(0, named=True)

    assert result["rev_1w_isna"]
    assert result["max_ret_1m_isna"]


def test_add_reversal_pit_future_rows_do_not_change_past_value(
    tmp_path: Path, lake: UsLake
) -> None:
    n = 30
    start = date(2020, 1, 1)
    closes = [100.0 * (1.01**i) for i in range(n)]
    t_idx = 25
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])

    _write_prices(tmp_path, _price_series("AAA", start, closes))
    _write_no_splits(tmp_path)
    baseline = add_reversal(panel, lake).row(0, named=True)

    _write_prices(tmp_path, _price_series("AAA", start, closes + [1.0, 99999.0]))
    with_future = add_reversal(panel, lake).row(0, named=True)

    assert with_future["rev_1w"] == pytest.approx(baseline["rev_1w"], rel=1e-9)
    assert with_future["max_ret_1m"] == pytest.approx(baseline["max_ret_1m"], rel=1e-9)


# --- volatility (F3) ---------------------------------------------------------------


def _returns_to_closes(returns: list[float], base: float = 100.0) -> list[float]:
    closes = [base]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    return closes


def test_add_volatility_idio_vol_60_is_zero_when_stock_perfectly_tracks_market(
    tmp_path: Path, lake: UsLake
) -> None:
    """잡음 없이 ``ret = 1.5 * ret_spy``이면 idio_vol_60은 0에 가깝고
    beta_252는 1.5에 가까워야 한다."""
    n_returns = 260
    spy_returns = [0.01, -0.01, 0.02, -0.005][: n_returns % 4 or 4] * (n_returns // 4 + 1)
    spy_returns = spy_returns[:n_returns]
    stock_returns = [1.5 * r for r in spy_returns]

    start = date(2019, 1, 1)
    _write_prices(
        tmp_path,
        _price_series("AAA", start, _returns_to_closes(stock_returns))
        + _price_series("SPY", start, _returns_to_closes(spy_returns)),
    )
    _write_no_splits(tmp_path)

    t_idx = n_returns  # 마지막 행(가장 최근)
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])
    result = add_volatility(panel, lake).row(0, named=True)

    assert result["beta_252"] == pytest.approx(1.5, rel=1e-6)
    assert result["idio_vol_60"] == pytest.approx(0.0, abs=1e-8)
    assert not result["beta_252_isna"]
    assert not result["idio_vol_60_isna"]


def test_add_volatility_rv_20_and_rv_60_are_stdev_of_daily_returns(
    tmp_path: Path, lake: UsLake
) -> None:
    n_returns = 260
    returns = [0.01 if i % 2 == 0 else -0.008 for i in range(n_returns)]
    start = date(2019, 1, 1)
    _write_prices(
        tmp_path,
        _price_series("AAA", start, _returns_to_closes(returns))
        + _price_series("SPY", start, _returns_to_closes(returns)),
    )
    _write_no_splits(tmp_path)

    t_idx = n_returns
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])
    result = add_volatility(panel, lake).row(0, named=True)

    expected_rv20 = statistics.stdev(returns[-20:])
    expected_rv60 = statistics.stdev(returns[-60:])
    assert result["rv_20"] == pytest.approx(expected_rv20, rel=1e-6)
    assert result["rv_60"] == pytest.approx(expected_rv60, rel=1e-6)


def test_add_volatility_isna_when_warmup_window_not_filled(tmp_path: Path, lake: UsLake) -> None:
    n_returns = 10
    start = date(2019, 1, 1)
    returns = [0.01] * n_returns
    _write_prices(
        tmp_path,
        _price_series("AAA", start, _returns_to_closes(returns))
        + _price_series("SPY", start, _returns_to_closes(returns)),
    )
    _write_no_splits(tmp_path)

    panel = _panel([{"date": start + timedelta(days=n_returns), "symbol": "AAA"}])
    result = add_volatility(panel, lake).row(0, named=True)

    assert result["rv_20_isna"]
    assert result["rv_60_isna"]
    assert result["idio_vol_60_isna"]
    assert result["beta_252_isna"]


def test_with_market_return_fills_missing_spy_date_without_cascading_null(
    tmp_path: Path, lake: UsLake
) -> None:
    """SPY가 특정 하루만 행이 없어도 ``ret_spy``는 그날만 0.0이어야 한다.

    실측(2026-09-20)으로 SPY가 다른 종목은 거래한 날에 행이 없는 날이 5번
    있었고, 처음엔 그 결측 하나가 그 뒤 ``window_size`` 거래일 전체를 null로
    밀어버리는 버그가 있었다(polars ``rolling_std``/``rolling_cov``가 창 안의
    null 하나로 창 전체를 null 처리한다) — 이 테스트가 그 회귀를 막는다.
    """
    n = 30
    start = date(2020, 1, 1)
    aaa_rows = _price_series("AAA", start, [100.0 + i for i in range(n)])
    spy_rows = _price_series("SPY", start, [200.0 + i for i in range(n)])
    missing_date = start + timedelta(days=10)
    spy_rows = [r for r in spy_rows if r["date"] != missing_date]

    _write_prices(tmp_path, aaa_rows + spy_rows)
    _write_no_splits(tmp_path)

    daily = with_market_return(daily_prices(lake, symbols=["AAA"]), lake).collect()
    row = daily.filter(pl.col("date") == missing_date).row(0, named=True)

    assert row["ret_spy"] == pytest.approx(0.0)
    # 그 날짜 하나만 채워졌지 다른 날짜까지 0으로 덮이면 안 된다.
    other = daily.filter(pl.col("date") == missing_date + timedelta(days=1)).row(0, named=True)
    assert other["ret_spy"] != 0.0


def test_add_volatility_pit_future_rows_do_not_change_past_value(
    tmp_path: Path, lake: UsLake
) -> None:
    n_returns = 260
    returns = [0.01, -0.007, 0.003, -0.012][:4] * (n_returns // 4 + 1)
    returns = returns[:n_returns]
    start = date(2019, 1, 1)
    t_idx = n_returns
    panel = _panel([{"date": start + timedelta(days=t_idx), "symbol": "AAA"}])

    _write_prices(
        tmp_path,
        _price_series("AAA", start, _returns_to_closes(returns))
        + _price_series("SPY", start, _returns_to_closes(returns)),
    )
    _write_no_splits(tmp_path)
    baseline = add_volatility(panel, lake).row(0, named=True)

    future_returns = returns + [0.5, -0.5, 0.9]
    _write_prices(
        tmp_path,
        _price_series("AAA", start, _returns_to_closes(future_returns))
        + _price_series("SPY", start, _returns_to_closes(future_returns)),
    )
    with_future = add_volatility(panel, lake).row(0, named=True)

    assert with_future["rv_20"] == pytest.approx(baseline["rv_20"], rel=1e-9)
    assert with_future["beta_252"] == pytest.approx(baseline["beta_252"], rel=1e-6)
    assert with_future["idio_vol_60"] == pytest.approx(baseline["idio_vol_60"], abs=1e-8)


# --- liquidity (F4) -----------------------------------------------------------------


def test_add_liquidity_amihud_20_formula(tmp_path: Path, lake: UsLake) -> None:
    """amihud_20 = mean(|ret| / adj_dollar_volume) over 20일 — 손으로 검산한다."""
    n = 25
    start = date(2020, 1, 1)
    closes = [100.0 * (1.0 + 0.01 * (-1 if i % 2 == 0 else 1)) ** i for i in range(n)]
    volumes = [1_000.0 + i * 10 for i in range(n)]
    rows = [
        {
            "date": start + timedelta(days=i),
            "symbol": "AAA",
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": v,
        }
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]
    _write_prices(tmp_path, rows)
    _write_no_splits(tmp_path)
    _write_midas(tmp_path, [])

    t_idx = n - 1
    panel = _panel(
        [{"date": start + timedelta(days=t_idx), "symbol": "AAA"}],
        mcap_rank={(start + timedelta(days=t_idx), "AAA"): 3},
    )
    result = add_liquidity(panel, lake).row(0, named=True)

    ret = [closes[i] / closes[i - 1] - 1.0 for i in range(1, n)]
    dvol = [closes[i] * volumes[i] for i in range(n)]
    window_rets = ret[-20:]
    window_dvols = dvol[-20:]
    expected_amihud = sum(abs(r) / d for r, d in zip(window_rets, window_dvols)) / 20
    expected_log_dvol = pytest.approx(math.log(sum(window_dvols) / 20), rel=1e-9)

    assert result["amihud_20"] == pytest.approx(expected_amihud, rel=1e-6)
    assert result["log_dvol_20"] == expected_log_dvol
    assert result["mcap_rank"] == 3
    assert not result["mcap_rank_isna"]


def test_add_liquidity_turnover_rank_joins_via_ticker_column(tmp_path: Path, lake: UsLake) -> None:
    """midas의 원천 컬럼명은 ``ticker``다 — ``symbol``로 바뀌어 조인돼야 한다."""
    d = date(2020, 1, 1)
    _write_prices(tmp_path, _price_series("AAA", d, [100.0, 101.0]))
    _write_no_splits(tmp_path)
    _write_midas(
        tmp_path,
        [
            {"date": d, "ticker": "AAA", "security_type": "Stock", "turn_rank": 7},
            {"date": d, "ticker": "AAA", "security_type": "ETF", "turn_rank": 2},  # 걸러져야 한다
        ],
    )

    panel = _panel([{"date": d, "symbol": "AAA"}], mcap_rank={(d, "AAA"): 5})
    result = add_liquidity(panel, lake).row(0, named=True)

    assert result["turnover_rank"] == 7
    assert not result["turnover_rank_isna"]


def test_add_liquidity_mcap_rank_isna_when_panel_value_missing(
    tmp_path: Path, lake: UsLake
) -> None:
    d = date(2020, 1, 1)
    _write_prices(tmp_path, _price_series("AAA", d, [100.0]))
    _write_no_splits(tmp_path)
    _write_midas(tmp_path, [])

    panel = _panel([{"date": d, "symbol": "AAA"}], mcap_rank={})
    result = add_liquidity(panel, lake).row(0, named=True)

    assert result["mcap_rank"] is None
    assert result["mcap_rank_isna"]
    assert result["turnover_rank_isna"]


def test_add_liquidity_isna_when_warmup_window_not_filled(tmp_path: Path, lake: UsLake) -> None:
    d0 = date(2020, 1, 1)
    _write_prices(tmp_path, _price_series("AAA", d0, [100.0, 101.0, 99.0]))
    _write_no_splits(tmp_path)
    _write_midas(tmp_path, [])

    panel = _panel(
        [{"date": d0 + timedelta(days=2), "symbol": "AAA"}],
        mcap_rank={(d0 + timedelta(days=2), "AAA"): 1},
    )
    result = add_liquidity(panel, lake).row(0, named=True)

    assert result["log_dvol_20_isna"]
    assert result["amihud_20_isna"]


# --- options_iv (F13) ---------------------------------------------------------------


def test_add_options_iv_rank_formula(tmp_path: Path, lake: UsLake) -> None:
    d = date(2020, 6, 1)
    n = 25
    start = d - timedelta(days=n - 1)
    _write_prices(tmp_path, _price_series("AAA", start, [100.0 + i * 0.1 for i in range(n)]))
    _write_no_splits(tmp_path)
    _write_volatility(
        tmp_path,
        [
            {
                "date": d,
                "symbol": "AAA",
                "iv_current": 0.30,
                "iv_year_high": 0.50,
                "iv_year_low": 0.10,
            }
        ],
    )

    panel = _panel([{"date": d, "symbol": "AAA"}])
    result = add_options_iv(panel, lake).row(0, named=True)

    assert result["iv_rank"] == pytest.approx((0.30 - 0.10) / (0.50 - 0.10))
    assert not result["iv_isna"]
    assert not result["iv_rank_isna"]


def test_add_options_iv_forward_fills_within_tolerance_but_not_beyond(
    tmp_path: Path, lake: UsLake
) -> None:
    report_date = date(2020, 6, 1)
    n = 60
    start = report_date - timedelta(days=5)
    _write_prices(tmp_path, _price_series("AAA", start, [100.0 + i * 0.1 for i in range(n)]))
    _write_no_splits(tmp_path)
    _write_volatility(
        tmp_path,
        [
            {
                "date": report_date,
                "symbol": "AAA",
                "iv_current": 0.40,
                "iv_year_high": 0.60,
                "iv_year_low": 0.20,
            }
        ],
    )

    within_tolerance = report_date + timedelta(days=10)
    beyond_tolerance = report_date + timedelta(days=45)
    panel = _panel(
        [
            {"date": within_tolerance, "symbol": "AAA"},
            {"date": beyond_tolerance, "symbol": "AAA"},
        ]
    )
    result = add_options_iv(panel, lake).sort("date")

    near = result.row(0, named=True)
    far = result.row(1, named=True)

    assert near["iv_rank"] == pytest.approx((0.40 - 0.20) / (0.60 - 0.20))
    assert not near["iv_isna"]
    assert far["iv_rank"] is None
    assert far["iv_isna"]


def test_add_options_iv_pit_future_report_does_not_leak_into_past(
    tmp_path: Path, lake: UsLake
) -> None:
    """``iv_rank``가 미래 창(그 다음 보고)을 보면 안 된다 — asof backward라
    과거 패널 날짜는 그 날짜 이전 보고만 봐야 한다."""
    past_report = date(2020, 1, 10)
    panel_date = date(2020, 1, 12)
    future_report = date(2020, 6, 1)  # panel_date 훨씬 뒤 — 극단값으로 오염 시도

    n = 30
    start = date(2020, 1, 1)
    _write_prices(tmp_path, _price_series("AAA", start, [100.0 + i * 0.1 for i in range(n)]))
    _write_no_splits(tmp_path)
    _write_volatility(
        tmp_path,
        [
            {
                "date": past_report,
                "symbol": "AAA",
                "iv_current": 0.25,
                "iv_year_high": 0.50,
                "iv_year_low": 0.10,
            },
            {
                "date": future_report,
                "symbol": "AAA",
                "iv_current": 0.99,  # 미래의 극단값
                "iv_year_high": 0.99,
                "iv_year_low": 0.01,
            },
        ],
    )

    panel = _panel([{"date": panel_date, "symbol": "AAA"}])
    result = add_options_iv(panel, lake).row(0, named=True)

    assert result["iv_rank"] == pytest.approx((0.25 - 0.10) / (0.50 - 0.10))


# --- calendar (F16) -----------------------------------------------------------------


def test_add_calendar_counts_early_closes_in_trailing_20_day_window(
    tmp_path: Path, lake: UsLake
) -> None:
    start = date(2020, 1, 1)
    n = 25
    early_close_indices = {3, 10, 19}  # 창(마지막 20일, index 5~24) 안에는 10·19만 있다
    rows = [
        {
            "date": start + timedelta(days=i),
            "exchange": "XNYS",
            "is_early_close": i in early_close_indices,
        }
        for i in range(n)
    ]
    _write_trading_calendar(tmp_path, rows)

    panel = _panel([{"date": start + timedelta(days=n - 1), "symbol": "AAA"}])
    result = add_calendar(panel, lake).row(0, named=True)

    assert result["early_close_in_window"] == 2
    assert not result["early_close_in_window_isna"]


def test_add_calendar_broadcasts_same_value_to_every_symbol(tmp_path: Path, lake: UsLake) -> None:
    start = date(2020, 1, 1)
    n = 25
    rows = [
        {"date": start + timedelta(days=i), "exchange": "XNYS", "is_early_close": i == 20}
        for i in range(n)
    ]
    _write_trading_calendar(tmp_path, rows)

    d = start + timedelta(days=n - 1)
    panel = _panel([{"date": d, "symbol": "AAA"}, {"date": d, "symbol": "BBB"}])
    result = add_calendar(panel, lake).sort("symbol")

    assert result["early_close_in_window"].to_list() == [1, 1]


def test_add_calendar_pit_future_early_close_does_not_affect_past_window(
    tmp_path: Path, lake: UsLake
) -> None:
    start = date(2020, 1, 1)
    n = 25
    d = start + timedelta(days=n - 1)
    panel = _panel([{"date": d, "symbol": "AAA"}])

    rows = [
        {"date": start + timedelta(days=i), "exchange": "XNYS", "is_early_close": False}
        for i in range(n)
    ]
    _write_trading_calendar(tmp_path, rows)
    baseline = add_calendar(panel, lake).row(0, named=True)

    future_rows = rows + [
        {"date": start + timedelta(days=n), "exchange": "XNYS", "is_early_close": True}
    ]
    _write_trading_calendar(tmp_path, future_rows)
    with_future = add_calendar(panel, lake).row(0, named=True)

    assert with_future["early_close_in_window"] == baseline["early_close_in_window"]


# --- _daily.mask_ticker_reuse_gap (헬퍼 단위 테스트) --------------------------------


def test_mask_ticker_reuse_gap_nulls_value_when_calendar_span_too_large() -> None:
    df = pl.DataFrame(
        {
            "symbol": ["A"] * 3,
            "date": [date(2010, 1, 1), date(2010, 1, 2), date(2020, 1, 1)],
            "value": [1.0, 2.0, 3.0],
        }
    ).lazy()

    result = df.with_columns(
        mask_ticker_reuse_gap(pl.col("value"), pl.col("date"), 1).alias("masked")
    ).collect()

    # row0->row(-1) 없음(null 유지), row1(2010-01-02 vs 2010-01-01, 1일 차): 정상.
    # row2(2020-01-01 vs 2010-01-02, ~10년 차): 무효화돼야 한다.
    assert result["masked"].to_list() == [None, 2.0, None]
