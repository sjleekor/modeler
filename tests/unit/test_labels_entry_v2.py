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
import pandas as pd
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


def _con(
    rows: list[tuple],
    quality: list[tuple] | None = None,
    index: list[tuple] | None = None,
    master: list[tuple] | None = None,
    krx_index: list[tuple] | None = None,
):
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
    if krx_index is not None:
        # (date, idx_nm, open_idx, close_idx); a decoy index rides along
        v = ",".join(
            f"(DATE '{d}', 'kospi', 'KOSPI', '{nm}', {o}, {c})" for d, nm, o, c in krx_index
        )
        con.execute(
            "CREATE VIEW krx_index_daily AS SELECT * FROM (VALUES "
            + v
            + ") AS t(bas_dd, index_group, idx_clss, idx_nm, open_idx, close_idx)"
        )
    if master is not None:
        v = ",".join(f"('{t}', '{m}', '{st}')" for t, m, st in master)
        con.execute(
            "CREATE VIEW master AS SELECT * FROM (VALUES " + v + ") AS t(ticker, market, status)"
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


def _series(ticker: str, start: str, closes: list) -> list[tuple]:
    """Consecutive calendar days; None = halted day (0,0,0,0)."""
    first = datetime.date.fromisoformat(start)
    out = []
    for i, c in enumerate(closes):
        d = (first + datetime.timedelta(days=i)).isoformat()
        bar = (0, 0, 0, 0) if c is None else (c if isinstance(c, tuple) else _bar(c))
        out.append((d, ticker, *bar))
    return out


def _run(con, spec: LabelSpec, quality_view: str | None = None, entry: str = "c1", **kw):
    kw.setdefault("min_cross_section", 1)  # synthetic panels are tiny
    sql = build_entry_label_sql(spec, entry, quality_view=quality_view, **kw)
    return (
        con.execute(f"SELECT * FROM ({sql}) ORDER BY ticker, trade_date")
        .fetch_df()
        .set_index(["ticker", "trade_date"])
    )


def _d(s: str):
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
    assert a1["label_exit_kind_c1_1d"] == "regular"
    assert a1["label_exit_lag_c1_1d"] == 0
    assert a1["label_end_date_c1_1d"] == _d("2020-01-04")
    assert not a1["in_terminal_tail"]


def test_halted_entry_day_is_null_not_pushed_back() -> None:
    rows = _series("A", "2020-01-02", [100, None, 120, 130, 140, 150])
    rows += _series("B", "2020-01-02", [10, 10, 10, 10, 10, 10])  # keeps d2 in the calendar
    df = _run(_con(rows), LabelSpec(horizons=(1,)))
    a = df.loc[("A", _d("2020-01-02"))]
    assert a["fwd_ret_c1_1d"] != a["fwd_ret_c1_1d"]  # NaN
    # c0 skipped the halt: enter d1 100 -> next live session d3 120
    assert a["fwd_ret_1d"] == pytest.approx(0.20)
    assert a["label_exit_kind_c1_1d"] == "untradable"


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


def test_limit_down_locked_exit_moves_to_next_free_session() -> None:
    r = _series("A", "2020-01-02", [100, 100, (70, 70, 70, 70), 100])
    r += _series("B", "2020-01-02", [10] * 6)
    df = _run(_con(r), LabelSpec(horizons=(1,), kind="abs"))
    a = df.loc[("A", _d("2020-01-02"))]
    # entry d2 (100); scheduled exit d3 is locked limit-down -> next free session d4 (100)
    assert a["fwd_ret_c1_1d"] == pytest.approx(0.0)
    assert a["label_exit_kind_c1_1d"] == "late"
    assert a["label_end_date_c1_1d"] == _d("2020-01-05")
    assert a["label_exit_lag_c1_1d"] == 1


def test_corporate_action_window_c1_excludes_entry_day_o1_includes_it() -> None:
    dates = ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]
    r = _series("A", "2020-01-02", [100, 100, 100, 100, 100, 100])

    def q(event_day: str | None):
        return [(d, "A", d == event_day) for d in dates]

    spec = LabelSpec(horizons=(2,), kind="abs")
    # t = d1, entry d2, exit d4 (d2 = 01-03, d4 = 01-05)
    cases = [
        # event day, masked for c1, masked for o1
        ("2020-01-02", False, False),  # on t itself: outside
        ("2020-01-03", False, True),  # entry day: after a c1 entry, before an o1 entry
        ("2020-01-04", True, True),  # the day after entry
        ("2020-01-05", True, True),  # exit day
        ("2020-01-06", False, False),  # after exit
        (None, False, False),
    ]
    for ev, m_c1, m_o1 in cases:
        for entry, masked in (("c1", m_c1), ("o1", m_o1)):
            df = _run(_con(r, quality=q(ev)), spec, quality_view="quality", entry=entry)
            row = df.loc[("A", _d("2020-01-02"))]
            v = row[f"fwd_ret_{entry}_2d"]
            assert bool(v != v) is masked, (ev, entry)
            assert (row[f"label_exit_kind_{entry}_2d"] == "ca_masked") is masked, (ev, entry)
    # no quality view: no mask
    df = _run(_con(r), spec)
    assert df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_2d"] == pytest.approx(0.0)


def test_terminal_needs_the_scheduled_exit_inside_the_calendar() -> None:
    spec = LabelSpec(horizons=(3,), kind="abs")
    a = _series("A", "2020-01-02", [100, 100, 100, 90])  # delisted after d4
    t = ("A", _d("2020-01-04"))  # t=d3: entry d4 (session 4), scheduled exit session 7
    # the calendar ends at session 6: survivors are censored here, so is A
    df = _run(_con(a + _series("B", "2020-01-02", [10] * 6)), spec)
    assert df.loc[t]["label_exit_kind_c1_3d"] == "censored"
    assert df.loc[t]["fwd_ret_c1_3d"] != df.loc[t]["fwd_ret_c1_3d"]
    assert pd.isna(df.loc[t]["label_end_date_c1_3d"])
    # the calendar reaches session 7: the window was observable, A is terminal
    df = _run(_con(a + _series("B", "2020-01-02", [10] * 7)), spec)
    assert df.loc[t]["label_exit_kind_c1_3d"] == "terminal"
    assert df.loc[t]["fwd_ret_c1_3d"] == pytest.approx(0.0)
    # survivors with the same window are censored at session 6, exit observed at 7
    df = _run(_con(a + _series("B", "2020-01-02", [10] * 7)), spec)
    assert df.loc[("B", _d("2020-01-04"))]["label_exit_kind_c1_3d"] == "regular"


def test_min_cross_section_nulls_bench_raw_and_y_but_keeps_fwd() -> None:
    rows = _series("A", "2020-01-02", [100, 100, 110, 110, 110])
    rows += _series("B", "2020-01-02", [100, 100, 100, 100, 100])
    rows += _series("C", "2020-01-02", [100, 100, None, 100, 100])
    spec = LabelSpec(horizons=(1,), outputs=("rank", "up"))
    # t=d2: C's entry day d3 is halted -> only A and B are valid
    t2 = [("A", _d("2020-01-03")), ("B", _d("2020-01-03"))]
    t1 = ("A", _d("2020-01-02"))
    df = _run(_con(rows), spec, min_cross_section=3)
    for t in t2:
        r = df.loc[t]
        assert not pd.isna(r["fwd_ret_c1_1d"])  # fwd stays
        for c in ("bench_ret_c1_1d", "raw_label_c1_1d", "y_rank_c1_1d", "y_up_c1_1d"):
            assert pd.isna(r[c]), c
    r = df.loc[t1]  # three valid rows: bench exists
    assert not pd.isna(r["bench_ret_c1_1d"])
    df = _run(_con(rows), spec, min_cross_section=2)
    assert not pd.isna(df.loc[t2[0]]["bench_ret_c1_1d"])


def test_ended_series_without_exit_is_terminal_at_last_close_and_row_kept() -> None:
    rows = _series("A", "2020-01-02", [100, 100, 100, 90])
    rows += _series("B", "2020-01-02", [10] * 6)  # A ended: its last row is before the market's
    spec = LabelSpec(horizons=(2,), kind="abs")
    df = _run(_con(rows), spec)
    # t=d3: entry d4 (90), exit would be d6 -> terminal at the last close (90)
    a = df.loc[("A", _d("2020-01-04"))]
    assert a["label_exit_kind_c1_2d"] == "terminal"
    assert a["fwd_ret_c1_2d"] == pytest.approx(0.0)
    assert a["label_end_date_c1_2d"] == _d("2020-01-05")
    assert a["label_exit_lag_c1_2d"] == 0  # floored: the end is nearer than h sessions
    # t=d4 (last row): no entry day in A -> untradable (entry missing)
    assert df.loc[("A", _d("2020-01-05"))]["label_exit_kind_c1_2d"] == "untradable"
    # t=d1: exit d4 exists
    assert df.loc[("A", _d("2020-01-02"))]["fwd_ret_c1_2d"] == pytest.approx(90 / 100 - 1)
    # the row set is wider than c0's (c0 inner join has no d3 of A for h=2)
    c0 = _con(rows).execute(build_label_sql(LabelSpec(horizons=(2,), kind="abs"))).fetch_df()
    c0 = c0.set_index(["ticker", "trade_date"])
    assert ("A", _d("2020-01-04")) not in c0.index


def test_panel_end_and_still_halted_are_censored() -> None:
    # both series run to the panel end: no exit -> censored, never terminal
    rows = _series("A", "2020-01-02", [100] * 6) + _series("B", "2020-01-02", [10] * 6)
    df = _run(_con(rows), LabelSpec(horizons=(5,), kind="abs"))
    a = df.loc[("A", _d("2020-01-04"))]  # entry d4, exit idx 9 beyond the panel
    assert a["label_exit_kind_c1_5d"] == "censored" and a["fwd_ret_c1_5d"] != a["fwd_ret_c1_5d"]
    last = df.loc[("A", _d("2020-01-07"))]  # no t+1 yet
    assert last["label_exit_kind_c1_5d"] == "censored"
    # A is still halted at the panel end (its last row is the market's last date)
    rows = _series("A", "2020-01-02", [100, 100, 100, 100, None, None, None])
    rows += _series("B", "2020-01-02", [10] * 7)
    df = _run(_con(rows), LabelSpec(horizons=(5,), kind="abs"))
    a = df.loc[("A", _d("2020-01-02"))]
    assert a["label_exit_kind_c1_5d"] == "censored"
    assert a["fwd_ret_c1_5d"] != a["fwd_ret_c1_5d"]


def _delisted_after_long_halt(tail_closes: list[float]) -> list[tuple]:
    rows = _series("A", "2020-01-02", [100, 101, 102, 103, 104] + [None] * 30 + tail_closes)
    return rows + _series("B", "2020-01-02", [10] * 60)


def test_terminal_tail_after_long_halt_is_not_masked_and_is_flagged() -> None:
    tail = [104 * 0.6**k for k in range(1, 8)]  # -40% a day: 정리매매
    rows = _delisted_after_long_halt(tail)
    quality = [(r[0], "A", True) for r in rows if r[1] == "A"][-7:]  # every tail row is an event
    spec = LabelSpec(horizons=(10, 5), kind="abs")
    df = _run(_con(rows, quality=quality), spec, quality_view="quality")
    a = df.loc[("A", _d("2020-01-05"))]  # t=d4 (103): entry d5 (104), exit idx 15 > 12
    assert a["label_exit_kind_c1_10d"] == "terminal"
    assert a["fwd_ret_c1_10d"] == pytest.approx(0.6**7 - 1)
    assert a["label_end_date_c1_10d"] == _d("2020-02-12")  # the series' last row (day 42)
    assert a["label_exit_lag_c1_10d"] == 42 - 5 - 10
    # t=d1, h=5: exit idx 7 is inside the tail; tail events do not mask -> late
    b = df.loc[("A", _d("2020-01-02"))]
    assert b["label_exit_kind_c1_5d"] == "late"
    assert b["fwd_ret_c1_5d"] == pytest.approx(104 * 0.6**2 / 101 - 1)
    # in_terminal_tail: only the 7 tail rows
    flags = df.loc["A"]["in_terminal_tail"]
    assert int(flags.sum()) == 7 and bool(flags.iloc[-1]) and not bool(flags.iloc[4])
    assert not df.loc["B"]["in_terminal_tail"].any()


def test_event_outside_the_tail_still_masks_a_terminal_label() -> None:
    rows = _delisted_after_long_halt([104 * 0.6**k for k in range(1, 8)])
    quality = [("2020-01-04", "A", True)]  # on d3, between entry d2.. and exit
    df = _run(
        _con(rows, quality=quality),
        LabelSpec(horizons=(10,), kind="abs"),
        quality_view="quality",
    )
    a = df.loc[("A", _d("2020-01-02"))]
    assert a["label_exit_kind_c1_10d"] == "ca_masked"
    assert a["fwd_ret_c1_10d"] != a["fwd_ret_c1_10d"]


def test_long_halt_end_needs_an_explicit_haircut() -> None:
    rows = _series("A", "2020-01-02", [100, 101, 102, 103, 104] + [None] * 25)
    rows += _series("B", "2020-01-02", [10] * 60)
    spec = LabelSpec(horizons=(3,), kind="abs")
    con = _con(rows)
    t = ("A", _d("2020-01-05"))  # t=d4: entry d5 (104), exit idx 8 missing
    df = _run(con, spec)
    assert df.loc[t]["label_exit_kind_c1_3d"] == "terminal_unpriced"
    assert df.loc[t]["fwd_ret_c1_3d"] != df.loc[t]["fwd_ret_c1_3d"]
    assert df.loc[t]["label_end_date_c1_3d"] == _d("2020-01-31")  # last halt row, day 30
    df = _run(con, spec, end_halt_haircut=0.5)
    assert df.loc[t]["label_exit_kind_c1_3d"] == "terminal_halted"
    assert df.loc[t]["fwd_ret_c1_3d"] == pytest.approx(0.5 * 104 / 104 - 1)
    # o1 exits at the last close too: enter at the open of d5 (103)
    df = _run(con, spec, entry="o1", end_halt_haircut=0.5)
    assert df.loc[t]["fwd_ret_o1_3d"] == pytest.approx(0.5 * 104 / 103 - 1)
    # a shorter halt than the end-halt threshold is an ordinary terminal
    df = _run(con, spec, end_halt_sessions=30)
    assert df.loc[t]["label_exit_kind_c1_3d"] == "terminal"
    assert df.loc[t]["fwd_ret_c1_3d"] == pytest.approx(0.0)
    # the entry day itself is halted for t=d5: untradable
    assert df.loc[("A", _d("2020-01-06"))]["label_exit_kind_c1_3d"] == "untradable"


def test_market_transfer_is_not_a_delisting_with_master_view() -> None:
    rows = _series("A", "2020-01-02", [100, 101, 102, 103]) + _series("B", "2020-01-02", [10] * 8)
    spec = LabelSpec(horizons=(2,), kind="abs")
    t = ("A", _d("2020-01-04"))  # entry d4, exit missing
    transfer = [("A", "KOSDAQ", "DELISTED"), ("A", "KOSPI", "ACTIVE"), ("B", "KOSPI", "ACTIVE")]
    df = _run(_con(rows, master=transfer), spec, master_view="master")
    assert df.loc[t]["label_exit_kind_c1_2d"] == "censored"
    delisted = [("A", "KOSPI", "DELISTED"), ("B", "KOSPI", "ACTIVE")]
    df = _run(_con(rows, master=delisted), spec, master_view="master")
    assert df.loc[t]["label_exit_kind_c1_2d"] == "terminal"
    # without a master view the gap rule alone decides, and min_end_gap_sessions tunes it
    df = _run(_con(rows), spec)
    assert df.loc[t]["label_exit_kind_c1_2d"] == "terminal"
    df = _run(_con(rows), spec, min_end_gap_sessions=5)
    assert df.loc[t]["label_exit_kind_c1_2d"] == "censored"


def test_parameter_validation() -> None:
    spec = LabelSpec()
    for kw in (
        {"min_end_gap_sessions": 0},
        {"end_halt_sessions": 0},
        {"terminal_tail_max": 0},
        {"min_cross_section": 0},
        {"end_halt_haircut": 1.5},
    ):
        with pytest.raises(ValueError):
            build_entry_label_sql(spec, "c1", **kw)


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
    for col in (
        "y_rank_c1_1d",
        "y_reg_c1_1d",
        "y_cls_c1_1d",
        "label_exit_kind_c1_1d",
        "label_end_date_c1_1d",
        "label_exit_lag_c1_1d",
    ):
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


# --- fixed benchmark (KOSPI 200) ----------------------------------------------

_K200 = (("k200", "코스피 200"),)


def _k200_rows(start: str, opens: list, closes: list, name: str = "코스피 200") -> list[tuple]:
    first = datetime.date.fromisoformat(start)
    return [
        ((first + datetime.timedelta(days=i)).isoformat(), name, o, c)
        for i, (o, c) in enumerate(zip(opens, closes, strict=True))
        if o is not None
    ]


def _krx(rows: list[tuple]) -> list[tuple]:
    decoy = _k200_rows("2020-01-02", [1] * 12, [1] * 12, name="코스피")  # must be ignored
    return rows + decoy


def test_default_entry_sql_is_unchanged_by_the_fixed_benchmark_option() -> None:
    pins = {
        "a": (
            LabelSpec(),
            "c1",
            {},
            "cabd562dbb32d20203349edd7f7969b0b1dc8dda6efc1abace5cc63d074a1f4d",
        ),
        "b": (
            LabelSpec(index_bench=True, horizons=(5, 20)),
            "c1",
            {"quality_view": "q"},
            "d769328c096a0772d64dd697b9698c212dc15e2d8eb699370e432b6f512b6b19",
        ),
        "c": (
            LabelSpec(kind="abs"),
            "o1",
            {"end_halt_haircut": 0.5, "master_view": "m"},
            "f36f91e0e089422f59be463c3d1ab1389525a1f8aed4ae5e33b5157c20421239",
        ),
    }
    for sp, entry, kw, sha in pins.values():
        assert _h(build_entry_label_sql(sp, entry, **kw)) == sha
        assert build_entry_label_sql(sp, entry, fixed_benchmarks=(), **kw) == build_entry_label_sql(
            sp, entry, **kw
        )


def test_fixed_benchmark_c1_and_o1_match_hand_calculation() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132, 132])  # open = close - 1
    rows += _series("B", "2020-01-02", [10] * 4)
    kx = _krx(_k200_rows("2020-01-02", [199, 209, 230, 249], [200, 210, 231, 250]))
    spec = LabelSpec(horizons=(1,), kind="abs")
    t = ("A", _d("2020-01-02"))
    df = _run(_con(rows, krx_index=kx), spec, fixed_benchmarks=_K200)
    r = df.loc[t]
    assert r["bench_ret_k200_c1_1d"] == pytest.approx(231 / 210 - 1)
    assert r["raw_label_k200_c1_1d"] == pytest.approx(132 / 110 - 1 - (231 / 210 - 1))
    df = _run(_con(rows, krx_index=kx), spec, entry="o1", fixed_benchmarks=_K200)
    r = df.loc[t]
    assert r["fwd_ret_o1_1d"] == pytest.approx(131 / 109 - 1)
    assert r["bench_ret_k200_o1_1d"] == pytest.approx(230 / 209 - 1)
    assert r["raw_label_k200_o1_1d"] == pytest.approx(131 / 109 - 1 - (230 / 209 - 1))
    # the stock label and the eqw benchmark are unchanged by adding the option
    base = _run(_con(rows, krx_index=kx), spec, entry="o1")
    assert base.loc[t]["fwd_ret_o1_1d"] == pytest.approx(r["fwd_ret_o1_1d"])


def test_fixed_benchmark_terminal_uses_series_last_date_and_close() -> None:
    # A trades to d4 (last close 90), has one halt row d5, then is gone; B runs on.
    rows = _series("A", "2020-01-02", [100, 100, 100, 90, None])
    rows += _series("B", "2020-01-02", [10] * 8)
    kx = _krx(
        _k200_rows(
            "2020-01-02",
            [199, 199, 199, 199, 209, 209, 209, 209],
            [200, 200, 200, 200, 220, 220, 220, 220],
        )
    )
    spec = LabelSpec(horizons=(2,), kind="abs")
    t = ("A", _d("2020-01-04"))  # t=d3: entry d4, exit missing, series ended at d5
    df = _run(_con(rows, krx_index=kx), spec, fixed_benchmarks=_K200)
    assert df.loc[t]["label_exit_kind_c1_2d"] == "terminal"
    assert df.loc[t]["label_end_date_c1_2d"] == _d("2020-01-06")
    assert df.loc[t]["bench_ret_k200_c1_2d"] == pytest.approx(220 / 200 - 1)  # d4 -> d5 close
    # o1: stock enters at the d4 open (89) and exits at the last CLOSE (90);
    # the index goes from the d4 open (199) to the d5 CLOSE (220)
    df = _run(_con(rows, krx_index=kx), spec, entry="o1", fixed_benchmarks=_K200)
    assert df.loc[t]["fwd_ret_o1_2d"] == pytest.approx(90 / 89 - 1)
    assert df.loc[t]["bench_ret_k200_o1_2d"] == pytest.approx(220 / 199 - 1)


def test_fixed_benchmark_missing_index_date_is_null_not_filled() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132, 132, 140])
    rows += _series("B", "2020-01-02", [10] * 5)
    # the index has no row for d3 (2020-01-04)
    kx = _krx(_k200_rows("2020-01-02", [199, 209, None, 249, 259], [200, 210, None, 250, 260]))
    spec = LabelSpec(horizons=(1,), kind="abs")
    df = _run(_con(rows, krx_index=kx), spec, fixed_benchmarks=_K200)
    a = df.xs("A", level="ticker")
    # t=d1: exit d3 missing -> NULL bench/raw, fwd kept; t=d2: entry d3 missing -> same
    for d in ("2020-01-02", "2020-01-03"):
        r = a.loc[_d(d)]
        assert not pd.isna(r["fwd_ret_c1_1d"])
        assert pd.isna(r["bench_ret_k200_c1_1d"]) and pd.isna(r["raw_label_k200_c1_1d"])
    # t=d3 (entry d4, exit d5) is fine: no forward fill from d2
    assert a.loc[_d("2020-01-04")]["bench_ret_k200_c1_1d"] == pytest.approx(260 / 250 - 1)
    n_null = int(a["bench_ret_k200_c1_1d"].isna().sum() - a["fwd_ret_c1_1d"].isna().sum())
    assert n_null == 2


def test_fixed_benchmark_three_part_key_and_validation() -> None:
    rows = _series("A", "2020-01-02", [100, 110, 132]) + _series("B", "2020-01-02", [10] * 3)
    kx = _krx(_k200_rows("2020-01-02", [199, 209, 230], [200, 210, 231]))
    df = _run(
        _con(rows, krx_index=kx),
        LabelSpec(horizons=(1,), kind="abs"),
        fixed_benchmarks=(("k200", "KOSPI", "코스피 200"),),
    )
    assert df.loc[("A", _d("2020-01-02"))]["bench_ret_k200_c1_1d"] == pytest.approx(231 / 210 - 1)
    for bad in ((("K 200", "x"),), (("a", "x"), ("a", "y")), (("a",),)):
        with pytest.raises(ValueError):
            build_entry_label_sql(LabelSpec(), "c1", fixed_benchmarks=bad)
