"""Tests for the B.3 universe retest (``universe_retest``).

Plan: ``my/milestones/kr/modeling/plan/20261006_universe_retest.md`` §8, "구현 시험 넷".

Four of these are the plan's: reproduction, benchmark, seal, invariance. The two
that touch real data (reproduction, benchmark, code hash) read only and compute no
U return. Everything that does use the U mask runs on synthetic frames — the plan
forbids looking at the U return on real data before the day it is run (§8, §9).
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path

import polars as pl
import pytest

from modeler.models._02_updown_prob.experiments import holdout_run as hr
from modeler.models._02_updown_prob.experiments import universe_retest as ur

REAL_RUN = hr.HOLDOUT_DIR / hr.ADOPTED_RUN_ID / "run_spec.json"


def _real_data_available() -> bool:
    if not REAL_RUN.exists():
        return False
    try:
        import json

        spec = json.loads(REAL_RUN.read_text())
        return (
            Path(spec["predictions_path"]).exists()
            and (Path(spec["dataset_dir"]) / "label_daily.parquet").exists()
        )
    except Exception:
        return False


real_data = pytest.mark.skipif(not _real_data_available(), reason="stock_data 가 없다")


# --- helpers --------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _my_repo(tmp_path: Path, *, tagged: bool) -> Path:
    repo = tmp_path / "my"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "plan.md").write_text("x")
    _git(repo, "add", "plan.md")
    _git(repo, "commit", "-q", "-m", "plan")
    if tagged:
        _git(repo, "tag", ur.FROZEN_TAG)
    return repo


def _adopted(**overrides) -> hr.AdoptedRun:
    base = dict(
        stage="E2",
        run_id="E2_h20_FS1h_seed0",
        horizon=20,
        k=2,
        cost_bps_roundtrip=60.0,
        tau=0.6,
        seed=0,
        model="hgb_clf",
        target="y_up",
        calibrate="none",
        monotonic=False,
        best_params={"max_iter": 10},
        primary_pred_col="p_raw",
        feature_set="FS1h",
        flow_variant="lag1",
        preprocess_profile="rank",
        dataset_dir=Path("/nonexistent"),
        predictions_path=Path("/nonexistent/p.parquet"),
        summary={},
        run_spec={},
    )
    base.update(overrides)
    return hr.AdoptedRun(**base)


def _dates(n: int) -> list[dt.date]:
    out, day = [], dt.date(2025, 8, 1)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


# ticker: (market, mcap, unreliable, tradable, pred, fwd_ret)
SPEC = {
    "K1": ("KOSPI", 100.0, False, True, 0.9, 0.10),  # in U
    "K2": ("KOSPI", 90.0, False, True, 0.8, 0.06),  # in U
    "K3": ("KOSPI", 80.0, False, True, 0.1, 0.02),  # in U (3rd of 3)
    "K4": ("KOSPI", 70.0, False, True, 0.95, 0.50),  # rank 4 -> out of U, best score
    "K5": ("KOSPI", 999.0, True, True, 0.99, 0.90),  # unreliable mcap -> dropped before ranking
    "Q1": ("KOSDAQ", 50.0, False, True, 0.5, 0.04),  # in U
    "Q2": ("KOSDAQ", 40.0, False, False, 0.6, 0.30),  # rank 2 but not tradable -> out, no refill
    "Q3": ("KOSDAQ", 30.0, False, True, 0.4, 0.08),  # rank 3 > 2 -> out (not refilled)
}


def _frame(n_dates: int = 21) -> pl.DataFrame:
    rows = []
    for d in _dates(n_dates):
        for ticker, (mkt, mcap, unrel, trad, pred, fwd) in SPEC.items():
            rows.append(
                {
                    "trade_date": d,
                    "ticker": ticker,
                    "market": mkt,
                    "mcap_krx": mcap,
                    "mcap_unreliable": unrel,
                    "tradable": trad,
                    "p_raw": pred,
                    "fwd_ret_20d": fwd,
                    "raw_label_20d": fwd - 0.05,
                    "raw_label_idx_20d": fwd - 0.03,
                    "raw_label_20d_ld": fwd - 0.05,
                    "fold_id": 1,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def small_u(monkeypatch):
    monkeypatch.setattr(ur, "U_SIZE", {"KOSPI": 3, "KOSDAQ": 2})


@pytest.fixture
def sealed(tmp_path) -> Path:
    return _my_repo(tmp_path, tagged=True)


# --- 봉인 -----------------------------------------------------------------------


def test_seal_refuses_without_the_tag(tmp_path, small_u):
    repo = _my_repo(tmp_path, tagged=False)
    with pytest.raises(ur.SealError):
        ur.assert_sealed(repo)
    with pytest.raises(ur.SealError):
        ur.add_u_mask(_frame(), repo)
    with pytest.raises(ur.SealError):
        ur.u_gate(_frame(), _adopted(), repo)
    with pytest.raises(ur.SealError):
        ur.definition_check(_frame(), _adopted(), repo)


def test_seal_opens_with_the_tag_and_names_the_commit(sealed, small_u):
    commit = ur.assert_sealed(sealed)
    assert len(commit) == 40
    assert "in_u" in ur.add_u_mask(_frame(), sealed).columns


def test_run_refuses_before_reading_anything_when_unsealed(tmp_path, monkeypatch):
    repo = _my_repo(tmp_path, tagged=False)
    monkeypatch.setenv("WSS_MY_REPO", str(repo))
    with pytest.raises(ur.SealError):
        ur.run(out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_retest_never_writes_into_the_opening_folder():
    adopted = _adopted()
    with pytest.raises(ValueError):
        ur.assert_not_holdout_folder(hr.HOLDOUT_DIR / adopted.run_id, adopted)
    with pytest.raises(ValueError):
        ur.assert_not_holdout_folder(hr.HOLDOUT_DIR / adopted.run_id / "sub", adopted)
    ur.assert_not_holdout_folder(ur.RETEST_DIR / adopted.run_id, adopted)


# --- U 마스크 (합성) ---------------------------------------------------------------


def test_u_is_the_top_n_per_market_then_tradable_without_refill(sealed, small_u):
    out = ur.add_u_mask(_frame(), sealed)
    members = set(out.filter(pl.col("in_u"))["ticker"].unique())
    # K5 is unreliable (dropped before ranking), K4 is rank 4 of 3. Q2 is rank 2 but not
    # tradable and its seat is NOT given to Q3.
    assert members == {"K1", "K2", "K3", "Q1"}


def test_u_ranks_every_day_and_breaks_ties_by_ticker(sealed, monkeypatch):
    monkeypatch.setattr(ur, "U_SIZE", {"KOSPI": 1, "KOSDAQ": 1})
    d1, d2 = _dates(2)
    rows = []
    for d, caps in ((d1, {"A": 5.0, "B": 5.0, "C": 1.0}), (d2, {"A": 1.0, "B": 5.0, "C": 9.0})):
        for t, c in caps.items():
            rows.append(
                {
                    "trade_date": d,
                    "ticker": t,
                    "market": "KOSPI",
                    "mcap_krx": c,
                    "mcap_unreliable": False,
                    "tradable": True,
                    "fwd_ret_20d": 0.0,
                }
            )
    out = ur.add_u_mask(pl.DataFrame(rows), sealed)
    picked = {
        d: out.filter((pl.col("trade_date") == d) & pl.col("in_u"))["ticker"].to_list()
        for d in (d1, d2)
    }
    assert picked == {d1: ["A"], d2: ["C"]}  # tie on d1 goes to the smaller ticker


def test_u_unreliable_or_missing_mcap_never_enters(sealed, small_u):
    frame = _frame().with_columns(
        pl.when(pl.col("ticker") == "K1").then(None).otherwise(pl.col("mcap_krx")).alias("mcap_krx")
    )
    out = ur.add_u_mask(frame, sealed)
    assert "K1" not in set(out.filter(pl.col("in_u"))["ticker"])


def test_the_three_benchmarks_are_means_over_their_own_rows(sealed, small_u):
    out = ur.add_u_mask(_frame(), sealed)
    day = out.filter(pl.col("trade_date") == out["trade_date"].min())

    def bench(market: str, name: str) -> float:
        return float(day.filter(pl.col("market") == market)[name][0])

    kospi = {k: v[5] for k, v in SPEC.items() if v[0] == "KOSPI"}
    assert bench("KOSPI", "bench_base") == pytest.approx(sum(kospi.values()) / 5)
    assert bench("KOSPI", "bench_trad") == pytest.approx(sum(kospi.values()) / 5)
    assert bench("KOSPI", "bench_u") == pytest.approx((0.10 + 0.06 + 0.02) / 3)
    assert bench("KOSDAQ", "bench_base") == pytest.approx((0.04 + 0.30 + 0.08) / 3)
    assert bench("KOSDAQ", "bench_trad") == pytest.approx((0.04 + 0.08) / 2)
    assert bench("KOSDAQ", "bench_u") == pytest.approx(0.04)


def test_u_gate_scores_only_u_names_against_the_u_equal_weight(sealed, small_u):
    # one market so the hand number is one line: U = K1, K2, K3; top-2 by score = K1, K2.
    frame = _frame().filter(pl.col("market") == "KOSPI")
    gate, framed = ur.u_gate(frame, _adopted(k=2), sealed)
    bench_u = (0.10 + 0.06 + 0.02) / 3
    assert gate.e == pytest.approx((0.10 + 0.06) / 2 - bench_u)
    # K4/K5 have the best scores and huge returns; they must not leak into U.
    assert gate.n_obs_base == framed.filter(pl.col("in_u")).height
    # I_U reads the index label of the same two names: fwd - 0.03 on each.
    assert gate.i == pytest.approx((0.10 + 0.06) / 2 - 0.03)
    assert gate.n_rebalances_e == 2


def test_u_gate_keeps_the_original_label_beside_the_new_one(sealed, small_u):
    out = ur.swap_label(ur.add_u_mask(_frame(), sealed), 20, "bench_u")
    assert "raw_label_20d_orig" in out.columns
    assert out["raw_label_20d_orig"].to_list() == _frame()["raw_label_20d"].to_list()


def test_decide_prime_carries_the_prime_and_the_same_order():
    assert ur.decide_prime(-0.01, 0.02, 0.02)[0] == "C′"
    assert ur.decide_prime(0.01, 0.02, 0.02)[0] == "A′"
    assert ur.decide_prime(0.01, -0.02, 0.02)[0] == "B′"
    assert ur.decide_prime(float("nan"), 0.0, 0.0)[0] == "미판정"


def test_definition_check_reports_literal_signs(sealed):
    frame = _frame().filter(pl.col("market") == "KOSPI")
    res = ur.definition_check(frame, _adopted(k=2), sealed)
    # runner label here is fwd - 0.05 on every row, so E_runner = top2 mean(fwd) - 0.05
    assert res["E_runner"] == pytest.approx((0.90 + 0.50) / 2 - 0.05)
    # literal: bench = mean fwd over all five default-universe rows
    bench = (0.10 + 0.06 + 0.02 + 0.50 + 0.90) / 5
    assert res["E_literal"] == pytest.approx((0.90 + 0.50) / 2 - bench)
    assert res["E_same_sign"] is True
    assert res["needs_correction_block"] is False


def test_jump_count_flags_a_split_inside_the_label_window_only():
    ds = _dates(30)
    rows = []
    for i, d in enumerate(ds):
        close = 100.0 if i < 10 else 40.0  # one-day ratio 0.4 on session 10
        rows.append(
            {
                "trade_date": d,
                "ticker": "X",
                "market": "KOSPI",
                "close": close,
                "label_end_date": ds[min(i + 20, 29)],
            }
        )
    out = ur.count_label_window_jumps(pl.DataFrame(rows), 20)
    # grid = session 0 and 20; only session 0's window (1..20) contains session 10.
    assert out["pairs"] == 1
    assert out["worst_multiple"] == pytest.approx(2.5)


def test_jump_on_a_day_the_name_left_u_still_counts():
    # The crash itself pushes the name out of U; the held row (session 0) is in U.
    ds = _dates(30)
    rows = []
    for i, d in enumerate(ds):
        rows.append(
            {
                "trade_date": d,
                "ticker": "X",
                "market": "KOSPI",
                "close": 100.0 if i < 10 else 40.0,
                "label_end_date": ds[min(i + 20, 29)],
                "in_u": i < 10,
            }
        )
    out = ur.count_label_window_jumps(pl.DataFrame(rows), 20, held_mask="in_u")
    assert out["pairs"] == 1
    assert out["worst_multiple"] == pytest.approx(2.5)


def test_jump_after_the_last_prediction_date_is_seen_through_prices():
    # The frame stops at session 9; the label window and the price series run on.
    ds = _dates(30)
    frame = pl.DataFrame(
        [
            {
                "trade_date": d,
                "ticker": "X",
                "market": "KOSPI",
                "close": 100.0,
                "label_end_date": ds[min(i + 20, 29)],
            }
            for i, d in enumerate(ds[:10])
        ]
    )
    prices = pl.DataFrame(
        [
            {"trade_date": d, "ticker": "X", "market": "KOSPI", "close": 100.0 if i < 15 else 40.0}
            for i, d in enumerate(ds)
        ]
    )
    assert ur.count_label_window_jumps(frame, 20)["pairs"] == 0
    out = ur.count_label_window_jumps(frame, 20, prices=prices)
    assert out["pairs"] == 1
    assert out["worst_multiple"] == pytest.approx(2.5)


# --- 재현 · 벤치 · 불변 (실제 데이터, U 없음) -------------------------------------------


@real_data
def test_reproduction_gives_the_published_gate_without_u():
    res = ur.reproduce()
    assert res["pass"], res
    assert res["verdict"] == "B"
    v = res["values"]
    assert v["e"] == pytest.approx(0.0081, abs=5e-5)
    assert v["e_prime"] == pytest.approx(0.0070, abs=5e-5)
    assert v["i"] == pytest.approx(-0.0249, abs=5e-5)
    assert v["s"] == pytest.approx(-0.0278, abs=5e-5)
    assert v["d"] == pytest.approx(-0.4973, abs=5e-5)


@real_data
def test_benchmark_is_the_all_stocks_equal_weight_rebuilt_from_prices():
    import json

    spec = json.loads(REAL_RUN.read_text())
    dataset_dir = Path(spec["dataset_dir"])
    raw_root = json.loads((dataset_dir / "dataset_manifest.json").read_text())["lake"]["raw"]
    labels = ur.load_label_daily(dataset_dir / "label_daily.parquet")
    labels = labels.filter(pl.col("trade_date") >= dt.date(2025, 8, 1))
    res = ur.bench_check(labels, raw_root)
    assert res["pass"], res
    assert res["cells"] > 400
    assert res["max_within_cell_std"] < 1e-12


def test_benchmark_check_uses_names_the_label_does_not_store(tmp_path):
    days = _dates(22)
    rows = []
    for i, d in enumerate(days):
        for t, growth in (("A", 1.01), ("B", 1.03)):
            px = 100.0 * growth**i
            rows.append(
                {"trade_date": d, "ticker": t, "market": "KOSPI", "open": px, "high": px,
                 "low": px, "close": px}
            )  # fmt: skip
    (tmp_path / "daily_ohlcv").mkdir()
    pl.DataFrame(rows).write_parquet(tmp_path / "daily_ohlcv" / "part.parquet")
    fwd_a, fwd_b = 1.01**20 - 1, 1.03**20 - 1
    bench = (fwd_a + fwd_b) / 2

    def label(raw_bench: float) -> pl.DataFrame:  # the label stores A only
        return pl.DataFrame(
            {
                "trade_date": [days[0]],
                "ticker": ["A"],
                "market": ["KOSPI"],
                "fwd_ret_20d": [fwd_a],
                "raw_label_20d": [fwd_a - raw_bench],
            }
        )

    assert ur.bench_check(label(bench), tmp_path)["pass"]
    # a benchmark averaged over the stored name only (A) is the "default universe" mistake
    assert not ur.bench_check(label(fwd_a), tmp_path)["pass"]


def test_model_code_hash_matches_the_openings_record():
    if not REAL_RUN.exists():
        pytest.skip("개봉 기록이 없다")
    recorded = ur.holdout_record(hr.load_adopted())["run_spec"]["model_code_hash"]
    assert ur.assert_code_unchanged() == recorded


def test_code_changed_is_refused():
    with pytest.raises(ur.CodeChangedError):
        ur.assert_code_unchanged("0" * 16)


# --- the whole run, on synthetic data (the real one is B.3's, on its day) ----------


def _full_frame() -> pl.DataFrame:
    days = _dates(41)
    return (
        _frame(41)
        .with_columns(
            pl.lit(100.0).alias("close"),
            pl.col("trade_date").map_elements(
                lambda d: days[min(days.index(d) + 20, len(days) - 1)], return_dtype=pl.Date
            ).alias("label_end_date"),
        )
    )  # fmt: skip


def test_run_end_to_end_on_synthetic_frames(tmp_path, monkeypatch, small_u):
    repo = _my_repo(tmp_path, tagged=True)
    monkeypatch.setenv("WSS_MY_REPO", str(repo))
    adopted = _adopted(k=2)
    frame = _full_frame()
    ctx = {
        "summary": {"dataset_dir": "synthetic"},
        "labels": None,
        "record": {"run_spec": {"predictions_path": "p", "dataset_dir": "d"}, "gate": {}},
    }
    monkeypatch.setattr(ur.hr, "load_adopted", lambda *a, **k: adopted)
    monkeypatch.setattr(ur, "assert_code_unchanged", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(ur, "load_holdout_frame", lambda a: (frame, ctx))
    monkeypatch.setattr(ur, "load_formation_frame", lambda *a: (frame, 3))
    monkeypatch.setattr(ur, "label_raw_root", lambda ctx: "unused")
    monkeypatch.setattr(
        ur,
        "load_close_series",
        lambda raw_root, start: frame.select("trade_date", "ticker", "market", "close"),
    )
    out = tmp_path / "out"
    result = ur.run(out_dir=out)
    # S is NaN on eight names, so the synthetic verdict is "미판정"; only the plumbing is checked
    assert result["verdict"] in {"A′", "B′", "C′", "미판정"}
    assert result["provenance"]["refit"] is False
    assert result["formation_record_1"]["label_rows_differing_from_label_daily"] == 3
    assert "pooled" in result["formation_record_1"]
    assert result["records"]["u_names_per_day_median"] == 4.0
    for name in ("result.json", "summary.md", "rebalance_returns_eqw_u.parquet"):
        assert (out / name).exists()
