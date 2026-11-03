"""End-to-end synthetic runs of stage2 past / forward / score.

Synthetic DuckDB lake, a temporary git repository with the annotated tag, a temporary output
root. The heavy stage 1 layers (feature build, label mart, hashes) are replaced by small
synthetic stand-ins; everything in ``stage2_run`` and ``seal`` runs for real.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import manifest, seal
from modeler.scores.earnings_a import paths as P
from modeler.scores.earnings_a import stage2_run as R
from modeler.scores.earnings_a.sealed_run import SealedAbort
from modeler.scores.earnings_a.spec import FEATURES
from modeler.scores.earnings_a.train import ModelSpec

N = 60
TICKERS = [f"{i * 10:06d}" for i in range(1, N + 1)]


class _Con:
    """Connection stand-in whose ``close`` does nothing (the runners close it)."""

    def __init__(self, con):
        self._c = con

    def __getattr__(self, k):
        return getattr(self._c, k)

    def close(self):
        pass


def _lake() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(
        """
        CREATE TABLE daily_ohlcv AS
        SELECT d AS trade_date, t AS ticker, 'KOSPI' AS market,
               CAST(5000 + i * 100 + 50 * sin(i * 1.0 + datediff('day', DATE '2018-01-01', d) / 30.0) AS BIGINT) AS close,
               CAST(0 AS BIGINT) AS volume, 0 AS i
        FROM (SELECT CAST(x AS DATE) AS d FROM range(DATE '2018-01-01', DATE '2028-04-01', INTERVAL 1 DAY) r(x)
              WHERE dayofweek(x) BETWEEN 1 AND 5) dd
        CROSS JOIN (SELECT lpad(CAST(k * 10 AS VARCHAR), 6, '0') AS t, k AS i FROM range(1, 61) q(k)) tt
        """.replace("0 AS i\n", "i\n")
    )
    c.execute("ALTER TABLE daily_ohlcv ADD COLUMN open BIGINT")
    c.execute("ALTER TABLE daily_ohlcv ADD COLUMN high BIGINT")
    c.execute("ALTER TABLE daily_ohlcv ADD COLUMN low BIGINT")
    c.execute("UPDATE daily_ohlcv SET open = close, high = close, low = close, volume = 100000")
    c.execute("DELETE FROM daily_ohlcv WHERE ticker = '000030' AND trade_date > DATE '2026-03-02'")
    c.execute(
        """
        CREATE TABLE daily_market_cap AS
        SELECT trade_date, ticker, market, close AS source_close, close * 1000 AS market_cap,
               close * volume AS trading_value, 1000 AS listed_shares, volume
        FROM daily_ohlcv
        """
    )
    c.execute(
        "INSERT INTO daily_market_cap SELECT trade_date, '000015', 'KOSPI', 3000, 3000000, 0, 1000, 1 "
        "FROM (SELECT DISTINCT trade_date FROM daily_ohlcv)"
    )
    c.execute(
        "CREATE TABLE dart_filing_receipt_raw(ticker VARCHAR, rcept_dt DATE, rcept_no VARCHAR, report_nm VARCHAR)"
    )
    c.execute(
        "CREATE TABLE krx_index_daily AS SELECT trade_date AS bas_dd, '코스피 200' AS idx_nm, "
        "CAST(300 + datediff('day', DATE '2018-01-01', trade_date) / 20.0 AS DECIMAL(30,4)) AS close_idx, "
        "TIMESTAMP '2026-01-01 00:00:00' AS fetched_at FROM (SELECT DISTINCT trade_date FROM daily_ohlcv)"
    )
    c.execute(
        "CREATE TABLE etf_daily AS SELECT '069500' AS ISU_CD, trade_date AS bas_date, "
        "CAST(30000 + datediff('day', DATE '2018-01-01', trade_date) AS BIGINT) AS tdd_clsprc_num, "
        "CAST(1 AS BIGINT) AS obs_seq FROM (SELECT DISTINCT trade_date FROM daily_ohlcv)"
    )
    c.execute(
        "CREATE TABLE etf_distribution(stock_code VARCHAR, rgt_std_date DATE, per_share_amount DOUBLE)"
    )
    c.execute("INSERT INTO etf_distribution VALUES ('069500', DATE '2027-06-30', 100.0)")
    return c


def _feats_for(dates) -> pd.DataFrame:
    parts = []
    for d in dates:
        rng = np.random.default_rng(d.toordinal())
        df = pd.DataFrame({"ticker": TICKERS})
        df["decision_date"] = d
        df["corp_code"] = "C" + df["ticker"]
        df["basis"] = "CFS"
        df["ksic2"] = "26"
        for f in FEATURES:
            df[f] = rng.normal(0.0, 0.05, N)
        df["oi_ttm_raw"] = df["oi_ttm"] * 1.1
        df["a_t"] = 1e9 * (1 + rng.random(N))
        df["div_ttm"] = rng.random(N) * 0.02
        df["debt_st_amt"] = 1e7
        df["debt_lt_amt"] = np.nan
        df["cash_amt"] = 2e7
        df["input_eligible"] = True
        df["input_eligible_incl_holding"] = True
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _labels_for(feats: pd.DataFrame, dates) -> pd.DataFrame:
    sub = feats[feats["decision_date"].isin(set(dates))]
    lab = sub[["decision_date", "ticker", "basis"]].copy()
    rng = np.random.default_rng(1)
    lab["label_ok"] = True
    lab["y"] = rng.normal(0.0, 0.05, len(lab))
    lab["y_clip"] = lab["y"]
    lab["label_pub_date"] = [
        pd.Timestamp(seal._to_date(d)) + pd.DateOffset(months=15) for d in lab["decision_date"]
    ]
    lab["missing_kind"] = None
    return lab


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "x")
    git("tag", "-a", seal.STAGE2_TAG, "-m", "frozen")
    out_root = tmp_path / "out"
    out_root.mkdir()
    paths = SimpleNamespace(out_root=out_root, stock_data_root=tmp_path, raw_root=tmp_path)
    lake = _lake()
    calls: list[str] = []
    state = {"f4_fail": False}

    def fake_prepare(con, dates):
        feats = _feats_for(sorted(set(dates)))
        return {"feats": feats, "rows": pd.DataFrame(), "receipts": pd.DataFrame()}

    def fake_labels(paths_, feats, receipts, dates, target, grant):
        return _labels_for(feats, dates), None

    def fake_f4(paths_, lf, mode, path):
        calls.append("f4")
        if state["f4_fail"]:
            raise SealedAbort("F4 digest differs")
        return {"key": mode.f4_key, "match": True}

    hgb = ModelSpec(
        "HGB",
        {"max_iter": 5, "min_samples_leaf": 5, "learning_rate": 0.1, "max_leaf_nodes": 7,
         "early_stopping": False, "loss": "absolute_error"},
    )  # fmt: skip
    monkeypatch.setattr(R, "connect_stage2", lambda p: _Con(lake))
    monkeypatch.setattr(R, "prepare", fake_prepare)
    monkeypatch.setattr(R, "_load_labels", fake_labels)
    monkeypatch.setattr(R, "_f4_check", fake_f4)
    monkeypatch.setattr(
        R, "_common_checks", lambda *a, **k: calls.append("checks") or {"HGB": hgb, "sha256": "x"}
    )
    monkeypatch.setattr(
        manifest,
        "git_state",
        lambda r: {"head": "h", "branch": "b", "dirty": False, "describe": "d"},
    )
    monkeypatch.setattr(manifest, "describe_inputs", lambda *a, **k: {})
    monkeypatch.setattr(P, "verify_input_hashes", lambda p: None)
    ledger = out_root / "seal_ledger.jsonl"
    return SimpleNamespace(
        paths=paths, repo=repo, ledger=ledger, calls=calls, state=state, lake=lake
    )


def _ledger(env):
    if not env.ledger.exists():
        return []
    return [json.loads(x) for x in env.ledger.read_text().splitlines()]


def test_past_runs_end_to_end_opens_once_and_checks_f4_first(env):
    kw = dict(ledger_path=env.ledger, repo_dir=env.repo)
    # a failing F4 digest stops the run before the seal is opened (no ledger, no output dir)
    env.state["f4_fail"] = True
    with pytest.raises(SealedAbort):
        R.run_past(env.paths, None, **kw)
    assert _ledger(env) == []
    assert not list(env.paths.out_root.glob("stage2_past_*"))
    env.state["f4_fail"] = False
    env.calls.clear()
    out_dir, mpath = R.run_past(env.paths, None, **kw)
    assert env.calls == ["checks", "f4"] or env.calls[:2] == ["checks", "f4"]
    led = _ledger(env)
    assert len(led) == 1 and led[0]["phase"] == "STAGE2_PAST" and led[0]["reopen"] is False
    for f in ("baskets.tsv", "tranches.tsv", "tranche_records.tsv", "summary.json"):
        assert (out_dir / f).is_file()
    m = json.loads(mpath.read_text())
    assert m["kind"] == "STAGE2_PAST" and m["f4_check"]["match"] is True
    assert set(m["output_sha256"]) == {"baskets", "tranches", "tranche_records", "summary"}
    tab = pd.read_csv(out_dir / "tranches.tsv", sep="\t")
    assert len(tab) == 25 and tab["M_net"].notna().all() and tab["B0"].notna().all()
    assert np.allclose(tab["M_gross"] - tab["M_net"], 0.0065)
    summ = json.loads((out_dir / "summary.json").read_text())
    assert summ["n_tranches"] == 25 and summ["seed"] == 20261103
    # second open is refused and leaves no empty output directory behind
    before = set(env.paths.out_root.glob("stage2_past_*"))
    with pytest.raises(seal.SealViolation, match="already opened once"):
        R.run_past(env.paths, None, **kw)
    assert set(env.paths.out_root.glob("stage2_past_*")) == before
    assert len(_ledger(env)) == 1


def test_forward_then_score_uses_calendar_entry_and_drops_late_tranche(env):
    kw = dict(ledger_path=env.ledger, repo_dir=env.repo)
    # 2026-09-30, committed after the entry day -> late
    d1 = date(2026, 9, 30)
    out1, _ = R.run_forward(env.paths, d1, None, today=date(2026, 11, 3), **kw)
    meta1 = json.loads((out1 / "meta_2026-09-30.json").read_text())
    assert meta1["late"] is True and meta1["main_estimate"] is False
    assert meta1["signal_day"] == "2026-10-14" and meta1["entry_planned"] == "2026-10-21"
    bfile = out1 / meta1["basket_file"]
    import hashlib

    assert hashlib.sha256(bfile.read_bytes()).hexdigest() == meta1["basket_sha256"]
    baskets = pd.read_csv(bfile, sep="\t", dtype={"ticker": str})
    assert not any("ret" in c or "gross" in c for c in baskets.columns)
    assert (baskets["variant"] == "M").sum() == 30
    with pytest.raises(seal.SealViolation, match="already granted once"):
        R.run_forward(env.paths, d1, None, today=date(2026, 11, 3), **kw)
    # 2026-12-31, committed before the entry day -> counts; earlier forward date is readable
    d2 = date(2026, 12, 31)
    out2, mpath2 = R.run_forward(env.paths, d2, None, today=date(2027, 1, 15), **kw)
    meta2 = json.loads((out2 / "meta_2026-12-31.json").read_text())
    assert meta2["late"] is False and meta2["main_estimate"] is True
    assert meta2["entry_planned"] == "2027-01-21" and meta2["entry_is_estimate"] is False
    rec = [r for r in _ledger(env) if r["decision_date"] == "2026-12-31"][0]
    assert "2026-09-30" in rec["allowed_dates"] and "2026-12-31" in rec["allowed_dates"]
    assert "2025-09-30" not in rec["allowed_dates"]  # CONFIRM stays closed
    assert json.loads(mpath2.read_text())["kind"] == "STAGE2_FORWARD"
    # a tampered basket file is refused at scoring
    out_s, _ = R.run_score(
        env.paths, [out1 / "meta_2026-09-30.json", out2 / "meta_2026-12-31.json"], None
    )
    tab = pd.read_csv(out_s / "tranches.tsv", sep="\t")
    row1 = tab[tab["decision_date"] == "2026-09-30"].iloc[0]
    row2 = tab[tab["decision_date"] == "2026-12-31"].iloc[0]
    assert row1["entry"] == "2026-11-04"  # late: first session after created_date
    assert bool(row1["entry_matches_plan"]) is False
    assert row2["entry"] == "2027-01-21" and bool(row2["entry_matches_plan"]) is True
    assert row2["exit"] == "2028-01-21"
    summ = json.loads((out_s / "score_summary.json").read_text())
    assert summ["main_estimate"]["n_tranches"] == 1
    assert summ["late_or_early_record_only"]["decision_dates"] == ["2026-09-30"]
    assert summ["pending_meta_files"] == []
    assert (out_s / "manifest_stage2_score.json").is_file()
    bfile.write_text(bfile.read_text() + "x\n")
    with pytest.raises(SealedAbort, match="sha256"):
        R.run_score(env.paths, [out1 / "meta_2026-09-30.json"], None)


def test_score_leaves_unmatured_tranche_pending(env):
    out1, _ = R.run_forward(
        env.paths, date(2026, 12, 31), None, today=date(2027, 1, 15),
        ledger_path=env.ledger, repo_dir=env.repo,
    )  # fmt: skip
    env.lake.execute("DELETE FROM daily_ohlcv WHERE trade_date > DATE '2027-06-30'")
    env.lake.execute("DELETE FROM daily_market_cap WHERE trade_date > DATE '2027-06-30'")
    out_s, _ = R.run_score(env.paths, [out1 / "meta_2026-12-31.json"], None)
    summ = json.loads((out_s / "score_summary.json").read_text())
    assert summ["n_scored"] == 0 and summ["pending_meta_files"] == ["meta_2026-12-31.json"]


def test_forward_without_the_tag_writes_nothing(env, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    subprocess.run(["git", "-C", str(bare), "init", "-q"], check=True)
    with pytest.raises(seal.SealViolation):
        R.run_forward(
            env.paths, date(2026, 9, 30), None, today=date(2026, 11, 3),
            ledger_path=env.ledger, repo_dir=bare,
        )  # fmt: skip
    assert not env.ledger.exists()
    assert not list(env.paths.out_root.glob("stage2_forward_*"))
