"""종목 구간(``security_id``) 단위 계산 — 유니버스 v2 설계 §3·T6 (``20261006_universe_v2``).

합성 레이크에 **같은 심볼의 구간 둘**(사이에 공백이 있고, 구간마다 분할이 하나씩 있다)을 만든다.
켜짐 모드(``lake.with_security_boundaries()``)에서 확인하는 것:

* 롤링 창이 앞 구간 행을 읽지 않는다 — 구간 하나만 담은 **기준 레이크**(꺼짐 모드)로 같은 피쳐를
  계산한 값과 같다. 구간 첫 (창−1)행은 비어 있다.
* 분할 조정이 그 구간 안의 분할만 쓴다.
* 지평이 구간 끝을 넘는 라벨은 빠진다.
* 수급 표(FTD·MIDAS·공매도·13F)의 롤링도 구간 안에서만 굴러간다.

꺼짐 모드(기본)는 구간이 없는 데이터에서 켜짐과 같고, 구간이 있는 데이터에서는 지금처럼 앞 구간을
읽는다(검정력 확인).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from modeler.etl.config import DataRoot
from modeler.us.features.ftd import add_ftd
from modeler.us.features.institutional import add_institutional
from modeler.us.features.liquidity import add_liquidity
from modeler.us.features.momentum import add_momentum
from modeler.us.features.order_flow import add_order_flow
from modeler.us.features.reversal import add_reversal
from modeler.us.features.short import add_short
from modeler.us.features.volatility import add_volatility
from modeler.us.labels import build_labels
from modeler.us.lake import SegmentedUsLake, UsLake
from modeler.us.prices import adjusted_daily
from modeler.us.segments import attach_security_id, group_key, pit_segments

# --- 합성 세계 -------------------------------------------------------------------

#: 구간 1: 2020-02-20 ~ 2020-12-14 (300일), 공백 9일, 구간 2: 2020-12-24 ~ 2021-10-19 (300일).
#: 구간 경계를 분기 끝(12-31) 바로 앞에 둔다 — 분기 단위 공표 지연(MIDAS·13F) 표가 구간 2의
#: 첫 행들을 읽게 되는 자리다.
SEG1_END = date(2020, 12, 14)
SEG1 = [SEG1_END - timedelta(days=299 - i) for i in range(300)]
SEG2_START = date(2020, 12, 24)
SEG2 = [SEG2_START + timedelta(days=i) for i in range(300)]
ALL_DATES = [SEG1[0] + timedelta(days=i) for i in range((SEG2[-1] - SEG1[0]).days + 1)]

VOL1 = 100_000.0
VOL2 = 1_000.0
SPLIT_IDX1 = 150
SPLIT_IDX2 = 120


def _snap(root: Path, table: str, frame: pl.DataFrame, snapshot_date: str = "2026-09-18") -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


_PRICE_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def _aaa_rows(dates: list[date], base: float, volume: float, split_idx: int) -> list[dict]:
    """``split_idx``부터 종가가 절반이 되는 AAA 시리즈 (2:1 분할)."""
    rows = []
    for i, d in enumerate(dates):
        close = base + 4.0 * math.sin(i / 5.0) + 0.02 * i
        if i >= split_idx:
            close *= 0.5
        rows.append(
            {
                "date": d,
                "symbol": "AAA",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
            }
        )
    return rows


def _spy_rows() -> list[dict]:
    rows = []
    for i, d in enumerate(ALL_DATES):
        close = 300.0 + 3.0 * math.sin(i / 9.0)
        rows.append(
            {
                "date": d,
                "symbol": "SPY",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 5_000_000.0,
            }
        )
    return rows


def _split_row(ex_date: date) -> dict:
    return {
        "symbol": "AAA",
        "ex_date": ex_date,
        "kind": "split",
        "to_factor": 2.0,
        "for_factor": 1.0,
    }


def _write_segments(root: Path, segs: list[tuple[date | None, date | None]]) -> None:
    """PIT 보기에 ``segs``(구간 목록)를, 사후 보기에는 **일부러 다른** 단일 구간을 쓴다.

    사후 보기를 읽으면 구간이 하나로 보여 아래 시험이 깨진다 — PIT만 읽는지의 보증이다.
    """
    rows = []
    for no, (start, end) in enumerate(segs, start=1):
        rows.append(
            {
                "symbol": "AAA",
                "segment_no": no,
                "security_id": f"AAA#{no}",
                "view": "pit",
                "seg_start": start,
                "seg_end": end,
            }
        )
    rows.append(
        {
            "symbol": "AAA",
            "segment_no": 1,
            "security_id": "AAA#1",
            "view": "post",
            "seg_start": None,
            "seg_end": None,
        }
    )
    rows.append(
        {
            "symbol": "SPY",
            "segment_no": 1,
            "security_id": "SPY#1",
            "view": "pit",
            "seg_start": None,
            "seg_end": None,
        }
    )
    _snap(
        root,
        "security_segments",
        pl.DataFrame(
            rows,
            schema={
                "symbol": pl.String,
                "segment_no": pl.Int32,
                "security_id": pl.String,
                "view": pl.String,
                "seg_start": pl.Date,
                "seg_end": pl.Date,
            },
        ),
    )


def _write_price_world(root: Path, *, seg1: bool, seg2: bool) -> None:
    prices = _spy_rows()
    splits = []
    if seg1:
        prices += _aaa_rows(SEG1, 100.0, VOL1, SPLIT_IDX1)
        splits.append(_split_row(SEG1[SPLIT_IDX1]))
    if seg2:
        prices += _aaa_rows(SEG2, 40.0, VOL2, SPLIT_IDX2)
        splits.append(_split_row(SEG2[SPLIT_IDX2]))
    _snap(root, "prices_daily", pl.DataFrame(prices, schema=_PRICE_SCHEMA))
    _snap(
        root,
        "corp_actions",
        pl.DataFrame(
            splits,
            schema={
                "symbol": pl.String,
                "ex_date": pl.Date,
                "kind": pl.String,
                "to_factor": pl.Float64,
                "for_factor": pl.Float64,
            },
        ),
    )
    _snap(
        root,
        "midas_security_daily",
        pl.DataFrame(
            schema={
                "date": pl.Date,
                "ticker": pl.String,
                "security_type": pl.String,
                "turn_rank": pl.Int32,
            }
        ),
    )


@pytest.fixture()
def world(tmp_path: Path) -> dict[str, Path]:
    """합친 세계(구간 둘)와 구간 하나씩만 담은 기준 세계 둘."""
    roots = {name: tmp_path / name for name in ("both", "ref1", "ref2")}
    _write_price_world(roots["both"], seg1=True, seg2=True)
    _write_segments(roots["both"], [(None, SEG2_START - timedelta(days=1)), (SEG2_START, None)])
    _write_price_world(roots["ref1"], seg1=True, seg2=False)
    _write_price_world(roots["ref2"], seg1=False, seg2=True)
    return roots


def _lake(root: Path, *, segmented: bool) -> UsLake:
    base = UsLake(root=DataRoot(base=root))
    return base.with_security_boundaries() if segmented else base


def _panel(dates: list[date], *, mcap: bool = False) -> pl.DataFrame:
    df = pl.DataFrame({"date": dates, "symbol": ["AAA"] * len(dates)})
    if mcap:
        df = df.with_columns(pl.lit(1, dtype=pl.Int32).alias("mcap_rank"))
    return df


# --- 리더·도우미 -------------------------------------------------------------------


def test_segmented_lake_is_a_subclass_with_flag_and_default_is_off(tmp_path: Path) -> None:
    base = UsLake(root=DataRoot(base=tmp_path))
    assert base.security_boundaries is False
    on = base.with_security_boundaries()
    assert isinstance(on, SegmentedUsLake) and on.security_boundaries is True
    assert on.root == base.root
    assert group_key(base) == "symbol" and group_key(on) == "security_id"


def test_attach_security_id_assigns_segment_by_date_and_uses_pit_view_only(
    world: dict[str, Path],
) -> None:
    lake = _lake(world["both"], segmented=True)
    frame = pl.DataFrame(
        {
            "date": [SEG1[0], SEG1_END, SEG2_START - timedelta(days=5), SEG2_START, SEG2[-1]],
            "symbol": ["AAA"] * 5,
        }
    )
    out = attach_security_id(frame.lazy(), lake).collect().sort("date")
    assert out["security_id"].to_list() == ["AAA#1", "AAA#1", "AAA#1", "AAA#2", "AAA#2"]
    # 구간 표에 없는 심볼은 첫 구간으로 둔다.
    unknown = attach_security_id(
        pl.DataFrame({"date": [SEG1[0]], "symbol": ["ZZZ"]}).lazy(), lake
    ).collect()
    assert unknown["security_id"].to_list() == ["ZZZ#1"]
    # 사후 보기(단일 구간)는 읽지 않는다 — PIT 구간 표는 구간이 둘이다.
    assert pit_segments(lake).filter(pl.col("symbol") == "AAA").height == 2


def test_pit_segments_refuses_a_table_without_view_column(tmp_path: Path) -> None:
    _snap(
        tmp_path,
        "security_segments",
        pl.DataFrame(
            {
                "symbol": ["AAA"],
                "segment_no": [1],
                "security_id": ["AAA#1"],
                "seg_start": [None],
                "seg_end": [None],
            },
            schema_overrides={"seg_start": pl.Date, "seg_end": pl.Date},
        ),
    )
    with pytest.raises(ValueError, match="view"):
        pit_segments(_lake(tmp_path, segmented=True))


# --- 분할 조정 -----------------------------------------------------------------------


def test_split_adjustment_uses_only_splits_inside_the_segment(world: dict[str, Path]) -> None:
    on = (
        adjusted_daily(_lake(world["both"], segmented=True))
        .filter(pl.col("symbol") == "AAA")
        .collect()
        .sort("date")
    )
    off = (
        adjusted_daily(_lake(world["both"], segmented=False))
        .filter(pl.col("symbol") == "AAA")
        .collect()
        .sort("date")
    )
    ex1, ex2 = SEG1[SPLIT_IDX1], SEG2[SPLIT_IDX2]

    def factor(df: pl.DataFrame, d: date) -> float:
        row = df.filter(pl.col("date") == d)
        return row["adj_close"][0] / row["close"][0]

    # 켜짐: 구간 1 행은 구간 1의 분할(ex1)만 — 구간 2의 분할(ex2)이 곱해지지 않는다.
    assert factor(on, SEG1[0]) == pytest.approx(0.5)
    assert factor(on, ex1) == pytest.approx(1.0)
    assert factor(on, SEG1_END) == pytest.approx(1.0)
    # 구간 2 행은 구간 2의 분할만.
    assert factor(on, SEG2[0]) == pytest.approx(0.5)
    assert factor(on, ex2) == pytest.approx(1.0)
    # 꺼짐: 지금처럼 심볼 전체에 두 분할이 모두 걸린다 (검정력 확인).
    assert factor(off, SEG1[0]) == pytest.approx(0.25)
    assert factor(off, ex1) == pytest.approx(0.5)
    assert factor(off, SEG2[0]) == pytest.approx(0.5)
    # 켜짐은 security_id 열을 더하고 꺼짐은 지금 열 그대로다.
    assert "security_id" in on.columns and "security_id" not in off.columns
    assert on["security_id"].to_list()[:1] == ["AAA#1"]
    assert on.filter(pl.col("date") >= SEG2_START)["security_id"].unique().to_list() == ["AAA#2"]


def test_panel_adjustment_ignores_the_boundary_flag(world: dict[str, Path]) -> None:
    """패널은 켜짐 리더로 불러도 심볼 단위 조정을 쓴다 — 패널 열은 두 모드에서 같아야 한다."""
    plain = adjusted_daily(_lake(world["both"], segmented=False), security_boundaries=False)
    forced = adjusted_daily(_lake(world["both"], segmented=True), security_boundaries=False)
    assert_frame_equal(
        plain.collect().sort(["symbol", "date"]), forced.collect().sort(["symbol", "date"])
    )


# --- 롤링 피쳐 -----------------------------------------------------------------------

#: (피쳐 이름, 창) — 구간 첫 (창−1)행은 비어야 한다. 창은 거래일 수다.
_WARMUP = {
    "mom_1m": 21,
    "mom_6_1": 126,
    "mom_12_1": 252,
    "rev_1w": 5,
    "max_ret_1m": 21,
    "rv_20": 20,
    "rv_60": 60,
    "idio_vol_60": 60,
    "beta_252": 252,
    "log_dvol_20": 20,
    "amihud_20": 20,
}

_PRICE_FAMILIES = (
    ("momentum", add_momentum, False),
    ("reversal", add_reversal, False),
    ("volatility", add_volatility, False),
    ("liquidity", add_liquidity, True),
)


def _compute(fn, lake: UsLake, dates: list[date], mcap: bool) -> pl.DataFrame:
    return fn(_panel(dates, mcap=mcap), lake).sort(["date", "symbol"])


@pytest.mark.parametrize(("name", "fn", "mcap"), _PRICE_FAMILIES)
def test_rolling_features_per_segment_equal_the_single_segment_reference(
    world: dict[str, Path], name: str, fn, mcap: bool
) -> None:
    """구간 2의 값이 구간 2만 담은 레이크에서 계산한 값과 같다 = 앞 구간 행을 안 읽는다."""
    on = _compute(fn, _lake(world["both"], segmented=True), SEG1 + SEG2, mcap)
    ref1 = _compute(fn, _lake(world["ref1"], segmented=False), SEG1, mcap)
    ref2 = _compute(fn, _lake(world["ref2"], segmented=False), SEG2, mcap)

    assert_frame_equal(on.filter(pl.col("date").is_in(SEG1)), ref1, check_exact=False)
    assert_frame_equal(on.filter(pl.col("date").is_in(SEG2)), ref2, check_exact=False)


def test_first_window_minus_one_rows_of_each_segment_are_empty(world: dict[str, Path]) -> None:
    lake = _lake(world["both"], segmented=True)
    frames = [_compute(fn, lake, SEG1 + SEG2, mcap) for _, fn, mcap in _PRICE_FAMILIES]
    for seg_dates in (SEG1, SEG2):
        for feature, window in _WARMUP.items():
            df = next(f for f in frames if feature in f.columns)
            values = df.filter(pl.col("date").is_in(seg_dates)).sort("date")[feature].to_list()
            assert all(v is None for v in values[: window - 1]), (
                feature,
                "앞 (창−1)행이 비어야 함",
            )
            # 창을 채운 뒤에는 값이 나온다.
            assert any(v is not None for v in values[window + 5 :]), (feature, "값이 나와야 함")


def test_off_mode_still_reads_the_previous_segment_across_the_boundary(
    world: dict[str, Path],
) -> None:
    """검정력 확인: 꺼짐(기본)은 지금처럼 구간 2 첫 행들에서 앞 구간 값을 읽는다."""
    off = _compute(add_momentum, _lake(world["both"], segmented=False), SEG1 + SEG2, False)
    on = _compute(add_momentum, _lake(world["both"], segmented=True), SEG1 + SEG2, False)
    early = SEG2[3]
    off_value = off.filter(pl.col("date") == early)["mom_1m"][0]
    on_value = on.filter(pl.col("date") == early)["mom_1m"][0]
    assert off_value is not None  # 심볼 단위라 21행 전이 구간 1에 있다
    assert on_value is None


@pytest.mark.parametrize(("name", "fn", "mcap"), _PRICE_FAMILIES)
def test_without_any_boundary_on_equals_off(tmp_path: Path, name: str, fn, mcap: bool) -> None:
    """구간이 하나뿐인 데이터에서는 켜짐과 꺼짐이 같다 — 켜짐이 다른 것을 바꾸지 않는다."""
    root = tmp_path / "single"
    _write_price_world(root, seg1=True, seg2=False)
    _write_segments(root, [(None, None)])
    on = _compute(fn, _lake(root, segmented=True), SEG1, mcap)
    off = _compute(fn, _lake(root, segmented=False), SEG1, mcap)
    assert_frame_equal(on, off, check_exact=False)


# --- 라벨 ----------------------------------------------------------------------------

_CAL = [date(2020, 1, 1) + timedelta(days=i) for i in range(40)]


def _label_world(root: Path) -> None:
    _snap(
        root,
        "trading_calendar",
        pl.DataFrame({"date": _CAL, "exchange": ["XNYS"] * len(_CAL)}),
        "2026-09-19",
    )
    _snap(
        root,
        "corp_actions",
        pl.DataFrame(
            schema={
                "symbol": pl.String,
                "ex_date": pl.Date,
                "kind": pl.String,
                "to_factor": pl.Float64,
                "for_factor": pl.Float64,
            }
        ),
    )
    _snap(
        root,
        "listing_snapshots",
        pl.DataFrame(schema={"as_of": pl.Date, "symbol": pl.String, "financial_status": pl.String}),
    )
    rows = []
    for sym, base in (("AAA", 10.0), ("BBB", 50.0), ("MKT", 100.0)):
        for i, d in enumerate(_CAL):
            close = base + i * 0.1
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000.0,
                }
            )
    _snap(root, "prices_daily", pl.DataFrame(rows, schema=_PRICE_SCHEMA))
    segs = [
        # AAA는 5번째 날(index 4)부터 두 번째 구간이다.
        ("AAA", 1, None, _CAL[3]),
        ("AAA", 2, _CAL[4], None),
        ("BBB", 1, None, None),
        ("MKT", 1, None, None),
    ]
    _snap(
        root,
        "security_segments",
        pl.DataFrame(
            [
                {
                    "symbol": s,
                    "segment_no": n,
                    "security_id": f"{s}#{n}",
                    "view": "pit",
                    "seg_start": a,
                    "seg_end": b,
                }
                for s, n, a, b in segs
            ],
            schema={
                "symbol": pl.String,
                "segment_no": pl.Int32,
                "security_id": pl.String,
                "view": pl.String,
                "seg_start": pl.Date,
                "seg_end": pl.Date,
            },
        ),
    )


def _label_panel(dates_symbols: list[tuple[date, str]]) -> pl.DataFrame:
    rows = []
    for d, sym in dates_symbols:
        i = _CAL.index(d)
        base = {"AAA": 10.0, "BBB": 50.0, "MKT": 100.0}[sym]
        close = base + i * 0.1
        rows.append(
            {
                "date": d,
                "symbol": sym,
                "cik": None,
                "sic": None,
                "sic2": None,
                "mcap_rank": None,
                "adv_20d": 1_000_000.0,
                "exchange": "XNYS",
                "close": close,
                "adj_close": close,
                "adj_volume": 1_000.0,
                "price_ge_5": True,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
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
        },
    )


def test_labels_are_empty_when_horizon_crosses_the_segment_end(tmp_path: Path) -> None:
    _label_world(tmp_path)
    t0, t6 = _CAL[0], _CAL[6]
    panel = _label_panel(
        [(t0, "AAA"), (t0, "BBB"), (t0, "MKT"), (t6, "AAA"), (t6, "BBB"), (t6, "MKT")]
    )
    on_lake = _lake(tmp_path, segmented=True)
    off_lake = _lake(tmp_path, segmented=False)

    on, on_diag = build_labels(on_lake, panel=panel, horizon=5)
    off, _ = build_labels(off_lake, panel=panel, horizon=5)

    # t0 + 5거래일(index 5)은 AAA#2 — AAA#1(끝 index 3)을 넘으므로 켜짐에서는 라벨이 빠진다.
    assert on.filter((pl.col("symbol") == "AAA") & (pl.col("date") == t0)).height == 0
    assert off.filter((pl.col("symbol") == "AAA") & (pl.col("date") == t0)).height == 1
    # 구간 2 안(index 6 → 11)은 켜짐에서도 계산된다. 값은 구간 2 가격만으로 나온다.
    row = on.filter((pl.col("symbol") == "AAA") & (pl.col("date") == t6))
    assert row.height == 1
    assert row["L0"][0] == pytest.approx((10.0 + 11 * 0.1) / (10.0 + 6 * 0.1) - 1)
    # 구간이 하나인 종목은 영향이 없다.
    assert on.filter(pl.col("symbol") == "BBB").height == 2
    assert on_diag["security_boundaries"] == {"enabled": True, "dropped_segment_end": 1}
    # 라벨 스키마는 꺼짐과 같다 (security_id·seg_end가 새지 않는다).
    assert on.columns == off.columns
    assert "security_boundaries" not in build_labels(off_lake, panel=panel, horizon=5)[1]


# --- 수급·외부 표 ----------------------------------------------------------------------


def _write_flow_tables(root: Path) -> None:
    """구간 1은 값이 크고 구간 2는 작다 — 앞 구간을 읽으면 크기로 바로 드러난다."""
    ftd = []
    for d in ALL_DATES:
        ftd.append({"settlement_date": d, "symbol": "ZZZ", "quantity": 1})  # 전역 결제일 축
    ftd += [{"settlement_date": d, "symbol": "AAA", "quantity": 1_000_000} for d in SEG1]
    ftd += [{"settlement_date": d, "symbol": "AAA", "quantity": 10} for d in SEG2]
    _snap(
        root,
        "ftd_fails",
        pl.DataFrame(
            ftd,
            schema={"settlement_date": pl.Date, "symbol": pl.String, "quantity": pl.Int64},
        ),
    )

    midas = []
    for d, cancels in [(d, 500) for d in SEG1] + [(d, 50) for d in SEG2]:
        midas.append(
            {
                "date": d,
                "ticker": "AAA",
                "security_type": "Stock",
                "cancels": cancels,
                "lit_trades": 100,
                "hidden_vol_k": 1.0,
                "trade_vol_for_hidden_k": 2.0,
                "odd_lot_vol_k": 1.0,
                "trade_vol_for_odd_lots_k": 2.0,
                "lit_vol_k": 1.0,
                "order_vol_k": 2.0,
            }
        )
    _snap(
        root,
        "midas_security_daily",
        pl.DataFrame(
            midas,
            schema={
                "date": pl.Date,
                "ticker": pl.String,
                "security_type": pl.String,
                "cancels": pl.Int64,
                "lit_trades": pl.Int64,
                "hidden_vol_k": pl.Float64,
                "trade_vol_for_hidden_k": pl.Float64,
                "odd_lot_vol_k": pl.Float64,
                "trade_vol_for_odd_lots_k": pl.Float64,
                "lit_vol_k": pl.Float64,
                "order_vol_k": pl.Float64,
            },
        ),
    )

    short_volume = [
        {
            "date": d,
            "symbol": "AAA",
            "short_volume": 90.0 if d in set(SEG1) else 10.0,
            "short_exempt_volume": 0.0,
            "total_volume": 100.0,
        }
        for d in SEG1 + SEG2
    ]
    _snap(
        root,
        "short_volume",
        pl.DataFrame(
            short_volume,
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "short_volume": pl.Float64,
                "short_exempt_volume": pl.Float64,
                "total_volume": pl.Float64,
            },
        ),
    )
    _snap(
        root,
        "short_interest",
        pl.DataFrame(
            schema={
                "settlement_date": pl.Date,
                "symbol": pl.String,
                "current_short_qty": pl.Float64,
                "avg_daily_volume_qty": pl.Float64,
                "days_to_cover": pl.Float64,
                "change_percent": pl.Float64,
                "revision_flag": pl.Boolean,
            }
        ),
    )
    _snap(
        root,
        "trading_calendar",
        pl.DataFrame({"date": ALL_DATES, "exchange": ["XNYS"] * len(ALL_DATES)}),
        "2026-09-19",
    )

    _snap(
        root,
        "inst_holdings_q",
        pl.DataFrame(
            [
                {
                    "cusip": "111111111",
                    "period_of_report": date(2020, 9, 30),
                    "n_holders": 100,
                    "shares_total": 1_000_000,
                    "n_filers_total_that_period": 5000,
                },
                {
                    "cusip": "111111111",
                    "period_of_report": date(2020, 12, 31),
                    "n_holders": 110,
                    "shares_total": 1_200_000,
                    "n_filers_total_that_period": 5000,
                },
            ],
            schema={
                "cusip": pl.String,
                "period_of_report": pl.Date,
                "n_holders": pl.Int32,
                "shares_total": pl.Int64,
                "n_filers_total_that_period": pl.Int32,
            },
        ),
    )
    _snap(
        root,
        "cusip_symbol_pit",
        pl.DataFrame(
            [
                {
                    "cusip": "111111111",
                    "symbol": "AAA",
                    "first_seen": date(2018, 1, 1),
                    "last_seen": date(2022, 1, 1),
                    "n_settlement_dates": 100,
                }
            ],
            schema={
                "cusip": pl.String,
                "symbol": pl.String,
                "first_seen": pl.Date,
                "last_seen": pl.Date,
                "n_settlement_dates": pl.Int32,
            },
        ),
    )


@pytest.fixture()
def flow_world(world: dict[str, Path]) -> Path:
    # prices_daily를 상수 거래량 시리즈로 다시 쓴다 (FTD 비율을 정확히 계산하려고).
    root = world["both"]
    _write_flow_tables(root)
    return root


def _seg2_values(df: pl.DataFrame, column: str) -> list[float | None]:
    return df.filter(pl.col("date").is_in(SEG2)).sort("date")[column].to_list()


def test_ftd_rolling_window_stays_inside_the_segment(flow_world: Path) -> None:
    on = add_ftd(_panel(SEG1 + SEG2), _lake(flow_world, segmented=True))
    off = add_ftd(_panel(SEG1 + SEG2), _lake(flow_world, segmented=False))

    # 켜짐: 구간 2 값은 구간 2의 수량(10)과 거래량(1,000)만으로 나온다 (= 0.01).
    # 구간 1의 값(10.0)이 안 섞인다.
    on_vals = [v for v in _seg2_values(on, "ftd_share_20") if v is not None]
    assert on_vals and all(v == pytest.approx(0.01) for v in on_vals)
    # 구간 2에는 아직 20결제일이 안 쌓인 날이 있어 그 사이 값은 빈다 — 첫 (창−1)행이 비는 규칙.
    assert any(v is None for v in _seg2_values(on, "ftd_share_20"))
    # 구간 1의 값은 구간 1 수량·거래량 그대로다.
    on1 = [
        v for v in on.filter(pl.col("date").is_in(SEG1))["ftd_share_20"].to_list() if v is not None
    ]
    assert on1 and all(v == pytest.approx(10.0) for v in on1)
    # 꺼짐(검정력 확인): 같은 구간 2 행에서 앞 구간이 섞인 값이 나온다.
    off_vals = [v for v in _seg2_values(off, "ftd_share_20") if v is not None]
    assert any(abs(v - 0.01) > 1e-6 for v in off_vals)


def test_order_flow_rolling_window_stays_inside_the_segment(flow_world: Path) -> None:
    on = add_order_flow(_panel(SEG1 + SEG2), _lake(flow_world, segmented=True))
    off = add_order_flow(_panel(SEG1 + SEG2), _lake(flow_world, segmented=False))

    on_vals = [v for v in _seg2_values(on, "cancel_ratio_20") if v is not None]
    assert on_vals and all(v == pytest.approx(0.5) for v in on_vals)  # 구간 2: 50/100
    off_vals = [v for v in _seg2_values(off, "cancel_ratio_20") if v is not None]
    assert any(abs(v - 0.5) > 1e-6 for v in off_vals)  # 앞 구간(5.0)이 섞인다


def test_short_volume_rolling_stays_inside_the_segment(flow_world: Path) -> None:
    on = add_short(_panel(SEG1 + SEG2), _lake(flow_world, segmented=True)).sort("date")
    off = add_short(_panel(SEG1 + SEG2), _lake(flow_world, segmented=False)).sort("date")

    seg2_on = _seg2_values(on, "sv_share_20")
    assert all(v is None for v in seg2_on[:19])  # 구간 첫 19행
    assert all(v == pytest.approx(0.1) for v in seg2_on[19:])  # 구간 2: 10/100
    seg2_off = _seg2_values(off, "sv_share_20")
    assert seg2_off[5] is not None and seg2_off[5] > 0.2  # 꺼짐은 앞 구간 0.9가 섞인다


def test_institutional_volume_window_stays_inside_the_segment(flow_world: Path) -> None:
    panel = _panel([d for d in SEG2 if d >= date(2021, 3, 5)][:5])
    on = add_institutional(panel, _lake(flow_world, segmented=True))
    off = add_institutional(panel, _lake(flow_world, segmented=False))

    # 2020-12-31은 구간 2의 8번째 행이라 구간 안 20일 평균 거래량이 없다.
    assert on["inst_shares_chg"].null_count() == on.height
    # 꺼짐은 앞 구간 거래량이 섞인 평균으로 값을 낸다: (1.2M-1.0M)/mean(12*1e5 + 8*1e3)
    assert off["inst_shares_chg"].null_count() == 0
    assert off["inst_shares_chg"][0] == pytest.approx(200_000.0 / ((12 * VOL1 + 8 * VOL2) / 20.0))
    # 롤링이 아닌 보유 값은 두 모드가 같다.
    assert on["inst_n_log"].to_list() == off["inst_n_log"].to_list()
    assert on["inst_n_log"][0] == pytest.approx(math.log(111.0))


def test_warmup_is_a_position_mask_not_a_full_window_requirement(flow_world: Path) -> None:
    """창 안에 거래량 빈 값이 있어도 구간 안 위치가 창−1 이상이면 값이 나온다 (FTD).

    구간 2의 2021-01-23 가격 행을 지운다 — 그날 결제일 격자 행은 수량 10, 거래량은 빈 값이다.
    그 날을 품은 창(2021-01-12~31)을 읽는 2021-02-25 행은 ``min_samples``=20이면 null이지만,
    위치 마스크(위치 38 ≥ 19)에서는 10×20 / (1,000×19) 로 계산된다.
    """
    prices_dir = flow_world / "derived" / "snapshots" / "prices_daily" / "snapshot_date=2026-09-18"
    path = prices_dir / "part.parquet"
    frame = pl.read_parquet(path)
    removed = date(2021, 1, 23)
    assert removed in set(SEG2)
    frame.filter(~((pl.col("symbol") == "AAA") & (pl.col("date") == removed))).write_parquet(path)

    probe = _panel([date(2021, 2, 25)])
    on = add_ftd(probe, _lake(flow_world, segmented=True))
    assert on["ftd_share_20"][0] == pytest.approx(200.0 / 19_000.0)
    assert on["ftd_days_20"][0] == 20


def test_warmup_positions_are_zero_based_window_minus_one(flow_world: Path) -> None:
    """구간 격자 위치 18은 비고 19는 값이 나온다 (창 20) — 같은 규칙을 모든 구간에 쓴다."""
    # 구간 2 격자: 2020-12-24가 위치 0. 위치 19 = 2021-01-12. 그 반월(1/1~15)의 마지막 결제일
    # 1/15(위치 22)를 읽는 2021-02-04 행은 값이 있고, 직전 반월(12/16~31, 마지막 12/31 = 위치 7)을
    # 읽는 2021-01-25 행은 비어 있다.
    on = add_ftd(_panel([date(2021, 1, 25), date(2021, 2, 4)]), _lake(flow_world, segmented=True))
    on = on.sort("date")
    assert on["ftd_share_20"][0] is None
    assert on["ftd_share_20"][1] == pytest.approx(0.01)
