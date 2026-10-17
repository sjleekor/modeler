"""Entry timing (c1/o1) of the label generator — quality-fix plan §3.

Synthetic DuckDB tables only. The c0 path must stay byte-identical to the
generator at commit aaa0f54 (pinned by sha256 of the SQL text, and, when git can
read that commit, by direct string comparison).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from modeler.etl.labels import (
    LabelSpec,
    build_label_scan_sql,
    build_label_sql,
    build_risk_label_sql,
)
from modeler.etl.labels_entry import build_entry_label_sql, materialize_entry_label

# --- c0 byte identity -----------------------------------------------------

_SPECS = {
    "default": LabelSpec(),
    "idx": LabelSpec(index_bench=True),
    "abs": LabelSpec(kind="abs", horizons=(5,), outputs=("reg", "rank", "cls", "up", "top")),
    "multi": LabelSpec(horizons=(1, 2, 3), outputs=("up", "top")),
    "risk": LabelSpec(include_risk=True, risk_horizons=(5, 20)),
}
_SHA = {
    "default": "bd927b316d37c87b9cedec686fd8b5b4f3f3e6bfa06b1cf5bf210631161b41b0",
    "idx": "7e088c7f113e1f6087075e23423feea4217e68f01ae184f9ed249a0a231bf83a",
    "abs": "587b6c7ba536eb66bf4c4e6f999541602b361df30bfaf2ca6a02b7cdab78c2f9",
    "multi": "233f4ea5b8fb80c19c1138503cb4ff64cebe98fa3d78afee4700a3e54444fa80",
    "risk": "6bd818739f9090bcf30706902293e6567881c0f0ad271bd586ffe54db60b7623",
}
_SHA_RISK_SQL = "ee52c810867b5deb8fd675d7e8ff4686f0cc716d44ded809b8309b677e2daf7b"
_SHA_SCAN = "d14db75cc6f6d296e76c98036e19f5d7184df8387d307e374959c44c8e0ee58c"
_SHA_SCAN_Q = "d8221afb7a3cc5854f52751429068c1a2e93de1fd2edaa0d940af8bee2e041aa"


def _h(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


@pytest.mark.parametrize("name", list(_SPECS))
def test_c0_sql_is_byte_identical_snapshot(name: str) -> None:
    assert _h(build_label_sql(_SPECS[name])) == _SHA[name]


def test_c0_other_builders_unchanged() -> None:
    assert _h(build_risk_label_sql(_SPECS["risk"])) == _SHA_RISK_SQL
    assert _h(build_label_scan_sql()) == _SHA_SCAN
    q = build_label_scan_sql(quality_view="quality", horizons=(1,), buckets=((0, 1),))
    assert _h(q) == _SHA_SCAN_Q


def test_c0_matches_original_module_when_git_has_it(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    try:
        src = subprocess.run(
            ["git", "show", "aaa0f54:src/modeler/etl/labels.py"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("commit aaa0f54 not readable")
    f = tmp_path / "labels_orig.py"
    f.write_text(src)
    sp = importlib.util.spec_from_file_location("labels_orig_aaa0f54", f)
    assert sp and sp.loader
    mod = importlib.util.module_from_spec(sp)
    sys.modules["labels_orig_aaa0f54"] = mod
    sp.loader.exec_module(mod)
    for name, new_spec in _SPECS.items():
        old_spec = mod.LabelSpec(
            **{k: getattr(new_spec, k) for k in new_spec.__dataclass_fields__ if k != "entry"}
        )
        assert build_label_sql(new_spec) == mod.build_label_sql(old_spec), name


def test_entry_validation() -> None:
    assert not hasattr(LabelSpec(), "entry")
    for bad in ("c0", "c2"):
        with pytest.raises(ValueError):
            build_entry_label_sql(LabelSpec(), bad)


# --- c1 / o1 behaviour ----------------------------------------------------


def _con(rows: list[tuple], quality: list[tuple] | None = None, index: list[tuple] | None = None):
    """rows: (date, ticker, open, high, low, close); market KOSPI."""
    con = duckdb.connect()
    values = ",".join(
        f"(DATE '{d}', '{t}', 'KOSPI', {o}, {h}, {lo}, {c}, 1)" for d, t, o, h, lo, c in rows
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + values
        + ") AS t(trade_date, ticker, market, open, high, low, close, volume)"
    )
    if quality is not None:
        v = ",".join(f"(DATE '{d}', '{t}', 'KOSPI', {str(e).upper()})" for d, t, e in quality)
        con.execute(
            "CREATE VIEW quality AS SELECT * FROM (VALUES "
            + v
            + ") AS t(trade_date, ticker, market, ca_event)"
        )
    if index is not None:
        v = ",".join(f"(DATE '{d}', '{s}', {x})" for d, s, x in index)
        con.execute(
            "CREATE VIEW common_feature_observation_raw AS SELECT * FROM (VALUES "
            + v
            + ") AS t(observation_date, series_id, value_numeric)"
        )
    return con


def _bar(c: float) -> tuple[float, float, float, float]:
    """A normal (non-flat) bar around close ``c``: open, high, low, close."""
    return (c - 1, c + 1, c - 2, c)


def _series(ticker: str, start: str, closes: list[float | None]) -> list[tuple]:
    """Consecutive calendar days; None = halted day (0,0,0,0)."""
    first = datetime.date.fromisoformat(start)
    out = []
    for i, c in enumerate(closes):
        d = (first + datetime.timedelta(days=i)).isoformat()
        bar = (0, 0, 0, 0) if c is None else _bar(c)
        out.append((d, ticker, *bar))
    return out


def _run(con, spec: LabelSpec, quality_view: str | None = None, entry: str = "c1"):
    sql = build_entry_label_sql(spec, entry, quality_view=quality_view)
    return (
        con.execute(f"SELECT * FROM ({sql}) ORDER BY ticker, trade_date")
        .fetch_df()
        .set_index(["ticker", "trade_date"])
    )


def _d(s: str):
    import pandas as pd

    return pd.Timestamp(s)


def test_c1_shift_matches_hand_calculation_and_keeps_c0() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132, 132, 150, 165])
    rows += _series("B", "2020-01-02", [50, 50, 55, 66, 66, 99])
    con = _con(rows)
    spec = LabelSpec(horizons=(1, 2), outputs=("rank", "up"))
    df = _run(con, spec)
    a1 = df.loc[("A", _d("2020-01-02"))]
    # c1 h=1: enter close d2 (110), exit close d3 (132)
    assert a1["fwd_ret_c1_1d"] == pytest.approx(132 / 110 - 1)
    # c1 h=2: exit close d4 (132)
    assert a1["fwd_ret_c1_2d"] == pytest.approx(132 / 110 - 1)
    # c0 is untouched: enter d1 (100), exit d2 (110)
    assert a1["fwd_ret_1d"] == pytest.approx(0.10)
    b1 = df.loc[("B", _d("2020-01-02"))]
    assert b1["fwd_ret_c1_1d"] == pytest.approx(55 / 50 - 1)
    # eqw market benchmark over both tickers, excess, both columns present
    bench = (132 / 110 - 1 + 55 / 50 - 1) / 2
    assert a1["bench_ret_c1_1d"] == pytest.approx(bench)
    assert a1["raw_label_c1_1d"] == pytest.approx(132 / 110 - 1 - bench)
    assert a1["y_up_c1_1d"] == 1 and b1["y_up_c1_1d"] == 0
    assert not a1["label_missing_exit_c1_1d"]


def test_halted_entry_day_is_null_not_pushed_back() -> None:
    rows = _series("A", "2020-01-02", [100, None, 120, 130, 140, 150])
    rows += _series("B", "2020-01-02", [10, 10, 10, 10, 10, 10])  # keeps d2 in the calendar
    df = _run(_con(rows), LabelSpec(horizons=(1,)))
    a = df.loc[("A", _d("2020-01-02"))]
    assert a["fwd_ret_c1_1d"] != a["fwd_ret_c1_1d"]  # NaN
    # c0 skipped the halt: enter d1 100 -> next live session d3 120
    assert a["fwd_ret_1d"] == pytest.approx(0.20)
    # not a missing-exit case
    assert not a["label_missing_exit_c1_1d"]


def test_limit_up_locked_entry_is_null_and_15pct_before_2015_06_15() -> None:
    def flat(c):  # locked bar
        return (c, c, c, c)

    # 2020: limit 30% -> 115 flat is +15%: tradeable. 130 flat is +30%: locked.
    r = _series("A", "2020-01-02", [100, 100, 100])
    r[1] = ("2020-01-03", "A", *flat(115))
    r2 = _series("C", "2020-01-02", [100, 100, 100])
    r2[1] = ("2020-01-03", "C", *flat(130))
    df = _run(_con(r + r2), LabelSpec(horizons=(1,), kind="abs"))
    assert df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_1d"] == pytest.approx(100 / 115 - 1)
    assert (
        df.loc[("C", _d("2020-01-02"))]["fwd_ret_c1_1d"]
        != df.loc[("C", _d("2020-01-02"))]["fwd_ret_c1_1d"]
    )

    # before 2015-06-15 the limit is 15%: 115 flat is locked
    p = _series("A", "2015-01-02", [100, 100, 100])
    p[1] = ("2015-01-03", "A", *flat(115))
    df = _run(_con(p), LabelSpec(horizons=(1,), kind="abs"))
    v = df.loc[("A", _d("2015-01-02"))]["fwd_ret_c1_1d"]
    assert v != v
    # the 15% -> 30% switch is on 2015-06-15 itself: that day's 115 is tradeable
    p = _series("A", "2015-06-13", [100, 100, 100, 100])
    p[1] = ("2015-06-14", "A", *flat(115))  # still 15%
    p[2] = ("2015-06-15", "A", *flat(115))  # 30% regime, +0%
    df = _run(_con(p), LabelSpec(horizons=(1,), kind="abs"))
    v = df.loc[("A", _d("2015-06-13"))]["fwd_ret_c1_1d"]
    assert v != v
    v = df.loc[("A", _d("2015-06-14"))]["fwd_ret_c1_1d"]
    assert v == pytest.approx(100 / 115 - 1)


def test_limit_down_locked_exit_is_null() -> None:
    r = _series("A", "2020-01-02", [100, 100, 100, 100])
    r[2] = ("2020-01-04", "A", 70, 70, 70, 70)  # -30% flat on exit day
    df = _run(_con(r), LabelSpec(horizons=(1,), kind="abs"))
    v = df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_1d"]
    assert v != v
    assert not df.loc[("A", _d("2020-01-02"))]["label_missing_exit_c1_1d"]


def test_corporate_action_in_shifted_window_masks() -> None:
    dates = ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]
    r = _series("A", "2020-01-02", [100, 100, 100, 100, 100, 100])

    def q(event_day: str | None):
        return [(d, "A", d == event_day) for d in dates]

    spec = LabelSpec(horizons=(2,), kind="abs")
    # t = d1, entry d2, exit d4: window [d2, d4]
    for ev, masked in [
        ("2020-01-02", False),  # on t itself: outside
        ("2020-01-03", True),  # on the entry day: inside
        ("2020-01-04", True),
        ("2020-01-05", True),  # exit day
        ("2020-01-06", False),  # after exit
        (None, False),
    ]:
        df = _run(_con(r, quality=q(ev)), spec, quality_view="quality")
        v = df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_2d"]
        assert bool(v != v) is masked, ev
    # no quality view: no mask
    df = _run(_con(r), spec)
    assert df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_2d"] == pytest.approx(0.0)


def test_missing_exit_is_flagged_and_row_kept() -> None:
    # A is delisted after d4; B lives on. c0 drops A's last rows, c1 flags them.
    rows = _series("A", "2020-01-02", [100, 100, 100, 100])
    rows += _series("B", "2020-01-02", [10, 10, 10, 10, 10, 10])
    df = _run(_con(rows), LabelSpec(horizons=(2,), kind="abs"))
    # t=d3: entry d4 ok, exit would be d6 -> missing
    a = df.loc[("A", _d("2020-01-04"))]
    assert bool(a["label_missing_exit_c1_2d"])
    assert a["fwd_ret_c1_2d"] != a["fwd_ret_c1_2d"]
    # t=d4 (last day): no entry at all -> not flagged
    assert not df.loc[("A", _d("2020-01-05"))]["label_missing_exit_c1_2d"]
    # t=d1: exit d4 exists
    assert df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_2d"] == pytest.approx(0.0)
    # the row set is wider than c0's (c0 inner join has no d3 of A for h=2)
    c0 = _con(rows).execute(build_label_sql(LabelSpec(horizons=(2,), kind="abs"))).fetch_df()
    c0 = c0.set_index(["ticker", "trade_date"])
    assert ("A", _d("2020-01-04")) not in c0.index


def test_o1_uses_open_prices_and_has_no_index_columns() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132, 132])  # open = close - 1
    idx = [
        (f"2020-01-0{i}", "market_kospi_krx", v) for i, v in zip(range(2, 6), [100, 200, 300, 330])
    ]
    spec = LabelSpec(horizons=(1,), kind="abs", index_bench=True)
    df = _run(_con(rows, index=idx), spec, entry="o1")
    a = df.loc[("A", _d("2020-01-02"))]
    assert a["fwd_ret_o1_1d"] == pytest.approx(131 / 109 - 1)
    assert not any("idx_o1" in c for c in df.columns)
    assert "bench_ret_idx_1d" in df.columns  # the c0 index column is unchanged


def test_c1_index_bench_uses_entry_and_exit_dates() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132, 132])
    idx = [
        (f"2020-01-0{i}", "market_kospi_krx", v) for i, v in zip(range(2, 6), [100, 200, 300, 330])
    ]
    spec = LabelSpec(horizons=(1,), kind="abs", index_bench=True)
    df = _run(_con(rows, index=idx), spec)
    a = df.loc[("A", _d("2020-01-02"))]
    assert a["bench_ret_idx_c1_1d"] == pytest.approx(300 / 200 - 1)
    assert a["raw_label_idx_c1_1d"] == pytest.approx(132 / 110 - 1 - 0.5)
    # c0 index return (d1 -> d2) is still there
    assert a["bench_ret_idx_1d"] == pytest.approx(1.0)


def test_c1_outputs_and_materialize_signature() -> None:
    sql = build_entry_label_sql(LabelSpec(horizons=(1,)), "c1")
    for col in ("y_rank_c1_1d", "y_reg_c1_1d", "y_cls_c1_1d", "label_missing_exit_c1_1d"):
        assert col in sql
    assert "quality_view" in materialize_entry_label.__code__.co_varnames


def test_entry_output_carries_the_c0_columns_unchanged() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132, 132, 150])
    rows += _series("B", "2020-01-02", [50, 50, 55, 66, 66])
    con = _con(rows)
    spec = LabelSpec(horizons=(1,), outputs=("rank", "up"))
    c0 = con.execute(f"SELECT * FROM ({build_label_sql(spec)}) ORDER BY 2, 1").fetch_df()
    both = _run(con, spec).reset_index()
    for col in [c for c in c0.columns if c not in ("trade_date", "ticker", "market")]:
        got = both.set_index(["ticker", "trade_date"])[col]
        for _, r in c0.iterrows():
            v = got.loc[(r["ticker"], r["trade_date"])]
            assert v == pytest.approx(r[col]) or (v != v and r[col] != r[col])
