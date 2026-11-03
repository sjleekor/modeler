"""Synthetic-only tests for the stage 2 database glue and tranche assembly."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a import stage2 as S
from modeler.scores.earnings_a import stage2_run as R

PKG = Path(seal.__file__).parent


def _days(n=30, start=date(2025, 7, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    days = _days()
    # 000010 liquid common, 000015 its preferred, 000020 penny (raw close 500), 000030 halted
    # on the last day, 000040 liquid but with a liquidation notice
    rows = []
    for d in days:
        for tk, close in (
            ("000010", 5000),
            ("000015", 3000),
            ("000020", 500),
            ("000030", 5000),
            ("000040", 5000),
        ):
            halted = tk == "000030" and d == days[-1]
            rows.append(
                (d, tk, "KOSPI", 0 if halted else close, 0 if halted else close,
                 0 if halted else close, close, 0 if halted else 100_000)
            )  # fmt: skip
    c.execute(
        "CREATE TABLE daily_ohlcv(trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open BIGINT, high BIGINT, low BIGINT, close BIGINT, volume BIGINT)"
    )
    c.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
    mc = [
        (d, tk, "KOSPI", close, close * 1000, 100_000, close * 1000, 100_000)
        for d in days
        for tk, close in (("000010", 5000), ("000015", 3000), ("000020", 500), ("000030", 5000), ("000040", 5000))
    ]  # fmt: skip
    c.execute(
        "CREATE TABLE daily_market_cap(trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "source_close BIGINT, market_cap BIGINT, trading_value BIGINT, listed_shares BIGINT, "
        "volume BIGINT)"
    )
    c.executemany("INSERT INTO daily_market_cap VALUES (?,?,?,?,?,?,?,?)", mc)
    c.execute(
        "CREATE TABLE dart_filing_receipt_raw(ticker VARCHAR, rcept_dt DATE, rcept_no VARCHAR, "
        "report_nm VARCHAR)"
    )
    c.execute(
        "INSERT INTO dart_filing_receipt_raw VALUES (?, ?, ?, ?)",
        ["000040", days[-3], days[-3].strftime("%Y%m%d") + "000001", "정리매매개시"],
    )
    c.execute(
        "CREATE TABLE krx_index_daily(bas_dd DATE, idx_nm VARCHAR, close_idx DECIMAL(30,4), "
        "fetched_at TIMESTAMP)"
    )
    c.executemany(
        "INSERT INTO krx_index_daily VALUES (?,?,?,?)",
        [
            (days[0], "코스피 200", 300.0, "2026-01-01 00:00:00"),
            (days[0], "코스피 200", 310.0, "2026-02-01 00:00:00"),  # later fetch wins
            (days[0], "코스피 200 중소형주", 1.0, "2026-01-01 00:00:00"),
            (days[5], "코스피 200", 330.0, "2026-01-01 00:00:00"),
        ],
    )
    c.execute(
        "CREATE TABLE etf_daily(ISU_CD VARCHAR, bas_date DATE, tdd_clsprc_num BIGINT, obs_seq BIGINT)"
    )
    c.executemany(
        "INSERT INTO etf_daily VALUES (?,?,?,?)",
        [("069500", days[0], 100, 1), ("069500", days[5], 110, 1), ("069500", days[0], 101, 2), ("999999", days[0], 7, 1)],
    )  # fmt: skip
    c.execute(
        "CREATE TABLE etf_distribution(stock_code VARCHAR, rgt_std_date DATE, per_share_amount DOUBLE)"
    )
    c.execute("INSERT INTO etf_distribution VALUES ('069500', ?, 1.0)", [days[3]])
    return c, days


def test_signal_day_info_uses_the_signal_day_itself(con):
    c, days = con
    u = days[-1]
    info = R.signal_day_info(c, [u]).set_index("ticker")
    assert set(info["signal_day"]) == {u}
    assert bool(info.loc["000030", "halted_u"]) and not bool(info.loc["000010", "halted_u"])
    assert bool(info.loc["000010", "p6_tradable_u"])
    assert not bool(info.loc["000020", "p6_tradable_u"])  # raw close 500 < 1,000
    assert info.loc["000010", "raw_close_u"] == 5000
    # the liquidation notice (3 sessions earlier) is on at u for 000040 only
    assert bool(info.loc["000040", "liq_notice_u"]) and not bool(info.loc["000010", "liq_notice_u"])


def test_mcap_lines_and_closes_and_entry_status(con):
    c, days = con
    lines = R.mcap_lines(c, [days[2]])
    assert len(lines) == 5 and lines["market_cap"].max() == 5000 * 1000
    cl = R.load_closes(c, ["000030", "000010"], days[0], days[-1])
    assert len(cl) == 2 * len(days)
    assert bool(
        cl[(cl["ticker"] == "000030") & (cl["trade_date"] == pd.Timestamp(days[-1]))][
            "is_halted"
        ].iloc[0]
    )
    es = R.entry_status(c, [days[-1]]).set_index("ticker")
    assert bool(es.loc["000030", "halted_entry"]) and not bool(es.loc["000010", "halted_entry"])
    assert R.load_closes(c, [], days[0], days[-1]).empty


def test_index_and_etf_series(con):
    c, days = con
    idx = R.index_series(c)
    assert list(idx.values) == [310.0, 330.0]  # exact name only, latest fetch on duplicates
    px, dist = R.etf_series(c)
    assert list(px.values) == [101.0, 110.0] and len(dist) == 1
    r = S.index_return(idx, days[0], days[5])
    assert r == pytest.approx(330 / 310 - 1)
    tr = S.etf_total_return(px, dist, days[0], days[5])
    assert tr == pytest.approx(110 / 101 - 1) or tr > 110 / 101 - 1  # record date in the window


def _tranche_inputs():
    tk = [f"{i:06d}" for i in range(10, 10 + 70 * 10, 10)]
    n = len(tk)
    feats = pd.DataFrame(
        {
            "ticker": tk,
            "corp_code": tk,
            "input_eligible": True,
            "a_t": 1000.0,
            "oi_ttm_raw": np.linspace(0, 0.2, n),
            "div_ttm": 0.01,
            "ksic2": "26",
            "debt_st_amt": 0.0,
            "debt_lt_amt": 0.0,
            "cash_amt": 10.0,
        }
    )
    preds = pd.DataFrame(
        {
            "ticker": tk,
            "y_hat_hgb": np.linspace(0.2, 0.0, n),
            "y_hat_pers": np.linspace(0.0, 0.2, n),
        }
    )
    info = pd.DataFrame(
        {
            "ticker": tk,
            "halted_u": False,
            "p6_tradable_u": True,
            "liq_notice_u": False,
            "raw_close_u": 5000.0,
        }
    )
    info.loc[0, "p6_tradable_u"] = False  # the best M name is not tradable at u
    lines = pd.DataFrame({"ticker": tk, "market_cap": np.full(n, 1e9)})
    return tk, feats, preds, info, lines


def test_build_tranche_counts_baskets_and_no_returns():
    tk, feats, preds, info, lines = _tranche_inputs()
    sch = S.Schedule(date(2025, 6, 30), date(2025, 7, 14), date(2025, 7, 21), date(2026, 7, 21))
    baskets, rec = R.build_tranche(date(2025, 6, 30), sch, feats, preds, info, lines)
    assert rec["n_eligible_t"] == 70 and rec["n_universe"] == 69
    assert rec["drop_p6_not_tradable"] == 1
    assert tk[0] not in set(baskets[baskets["variant"] == "M"]["ticker"])
    assert (baskets["variant"] == "M").sum() == 30
    assert baskets[baskets["variant"] == "M_N50"].shape[0] == 50
    assert not any("ret" in c or "gross" in c for c in baskets.columns)
    assert rec["overlap_M_B1"] >= 0 and "M_size_quintile" in rec
    # M and B1 rank opposite ends: the top M names are the lowest-index names
    m = baskets[baskets["variant"] == "M"].sort_values("rank")["ticker"].tolist()
    assert m[0] == tk[1]


def test_check_row_counts_without_predictions():
    tk, feats, _, info, lines = _tranche_inputs()
    info["raw_close_u"] = 400_000.0
    info.loc[3:, "raw_close_u"] = 500_000.0  # 67 names above 416,667
    lines = pd.concat([lines, pd.DataFrame({"ticker": [tk[2][:5] + "5"], "market_cap": [2e8]})])
    sch = S.Schedule(date(2025, 6, 30), date(2025, 7, 14), date(2025, 7, 21), None)
    eh = pd.DataFrame({"ticker": tk, "halted_entry": [True] + [False] * (len(tk) - 1)})
    row = R.check_row(date(2025, 6, 30), sch, feats, info, lines, eh)
    assert row["exit"] == "beyond data"
    assert row["n_eligible_t"] == 70 and row["n_eligible_listed_tradable_u"] == 69
    assert row["n_raw_close_gt_416667"] == 67  # names 3..69 (all tradable) are above the limit
    assert row["n_entry_halted"] == 0  # the halted-at-entry name is the untradable one
    assert row["n_pref_addon"] == 1 and row["n_pref_gt_10pct"] == 1
    assert row["n_all_common_tradable_u"] == 69


def test_stage2_modules_never_touch_the_label_mart_directly():
    for name in ("stage2", "stage2_run"):
        src = (PKG / f"{name}.py").read_text(encoding="utf-8")
        assert "r2_parquet" not in src and "mart_r2" not in src
        assert "fin_quarterly_metric_vintage_dl_v4" not in src
    # the pure core imports no label, training or evaluation code
    tree = ast.parse((PKG / "stage2.py").read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and "earnings_a" in (n.module or ""):
            names = {a.name for a in n.names}
            assert not names & {"labels", "train", "evaluate", "seal"}, names


def test_stage2_run_label_reads_only_through_guarded_stage1_functions():
    tree = ast.parse((PKG / "stage2_run.py").read_text(encoding="utf-8"))
    called = {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not called & {"load_label_rows", "build_labels", "build_label_quarters"}
    assert "_load_labels" in called  # stage 1 loader, itself guarded by the grant


def test_cli_registers_stage2_subcommands():
    src = (PKG / "cli.py").read_text(encoding="utf-8")
    for want in ("stage2-check", "stage2-past", "stage2-forward", "stage2-score", "stage2-repro"):
        assert f'"{want}"' in src
    # none of them is a bare open/seal/confirm command
    assert "stage2-open" not in src
