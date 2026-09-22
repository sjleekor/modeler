"""Unit tests for the model 02 holdout runner.

The runner's own rehearsal (``--rehearse-fold``) proves the *fit* path: it
refits a recorded fold and compares to the predictions the adopted run wrote.
What that cannot reach is the part the holdout adds — the date arithmetic, the
two purges, the gate arithmetic and the branch rule — because none of it runs
inside a validation fold. That is what these tests are for, on synthetic data
where the right answer can be counted by hand.

The wall is checked from both sides: ``_widen_to_holdout`` must refuse a date
that is not past it, and ``ModelSpec`` must still refuse the same date on its
own. A test that only proved the bypass works would be testing that the guard
is gone.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot, LakeConfig
from modeler.models._02_updown_prob.experiments import holdout_run as hr
from modeler.models._02_updown_prob.spec import HOLDOUT_START, PERIOD_END, ModelSpec

HOLDOUT = HOLDOUT_START  # "2025-08-01"


# --- helpers --------------------------------------------------------------------


def _sessions(start: dt.date, count: int) -> list[dt.date]:
    out: list[dt.date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _write_prices(
    config: LakeConfig, sessions: list[dt.date], *, halted: dict[str, set[dt.date]] | None = None
) -> None:
    """A two-ticker price table, optionally with halt days for one of them.

    Halt days are ``open=high=low=0``, which is what ``labels._forward_cte``
    filters on — so a halted ticker's ``d_idx`` really does skip them here, and
    the runner's row-level purge has something to find.
    """
    halted = halted or {}
    rows = []
    for day in sessions:
        for ticker, market in (("A001", "KOSPI"), ("A002", "KOSDAQ")):
            is_halt = day in halted.get(ticker, set())
            rows.append(
                {
                    "trade_date": day,
                    "ticker": ticker,
                    "market": market,
                    "open": 0 if is_halt else 1_000,
                    "high": 0 if is_halt else 1_000,
                    "low": 0 if is_halt else 1_000,
                    "close": 1_000,
                    "volume": 0 if is_halt else 2_000,
                }
            )
    path = config.raw_root / "daily_ohlcv"
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path / "part-000000.parquet")


def _lake(tmp_path: Path) -> LakeConfig:
    return LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-08-23", source="test")


def _adopted(tmp_path: Path, **overrides) -> hr.AdoptedRun:
    base = dict(
        stage="E2",
        run_id="E2_h20_FS1h_seed0",
        horizon=20,
        k=100,
        cost_bps_roundtrip=60.0,
        tau=0.6,
        seed=0,
        model="hgb_clf",
        target="y_up",
        calibrate="none",
        monotonic=False,
        best_params={"max_iter": 200, "learning_rate": 0.03},
        primary_pred_col="p_raw",
        feature_set="FS1h",
        flow_variant="lag1",
        preprocess_profile="rank",
        dataset_dir=tmp_path / "ds",
        predictions_path=tmp_path / "ds" / "p.parquet",
        summary={},
        run_spec={},
    )
    base.update(overrides)
    return hr.AdoptedRun(**base)


# --- the branch rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("e", "i", "s", "expected"),
    [
        (-0.01, 0.02, 0.02, "C"),  # C wins over A — the case §3.2 ordered for
        (0.0, 0.02, 0.02, "C"),  # E <= 0 is C, not "almost A"
        (0.01, 0.02, 0.02, "A"),
        (0.01, -0.01, 0.02, "B"),
        (0.01, 0.02, -0.01, "B"),
        (0.01, 0.0, 0.02, "B"),  # I == 0 is not "> 0"
        (0.01, 0.02, 0.0, "B"),
    ],
)
def test_decide_follows_the_frozen_order(e, i, s, expected):
    verdict, reason = hr.decide(e, i, s)
    assert verdict == expected
    assert reason


def test_decide_refuses_to_judge_a_nan():
    verdict, reason = hr.decide(float("nan"), 0.01, 0.01)
    assert verdict == "미판정"
    assert "NaN" in reason


# --- the two purges -------------------------------------------------------------


def test_last_clean_train_formation_is_h_sessions_before_the_wall():
    sessions = _sessions(dt.date(2025, 6, 2), 60)
    before = [d for d in sessions if str(d) < HOLDOUT]
    last, purge = hr.last_clean_train_formation(sessions, 20, HOLDOUT)
    assert purge == 20
    assert last == before[-21]
    # and the label of that date closes strictly before the wall
    assert str(sessions[sessions.index(last) + 20]) < HOLDOUT


def test_last_clean_train_formation_refuses_a_window_it_cannot_purge():
    sessions = _sessions(dt.date(2025, 7, 21), 30)  # only ~9 sessions before the wall
    with pytest.raises(ValueError, match="cannot purge"):
        hr.last_clean_train_formation(sessions, 20, HOLDOUT)


def test_purge_train_by_label_end_drops_only_the_crossing_rows():
    frame = pl.DataFrame(
        {
            "trade_date": [dt.date(2022, 3, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "label_end_date": [
                dt.date(2022, 3, 29),  # normal, well before the wall
                dt.date(2025, 8, 4),  # halted across the wall
                None,  # never closed — keeps, it is not a leak
            ],
        }
    )
    kept, dropped = hr.purge_train_by_label_end(frame, HOLDOUT)
    assert dropped == 1
    assert sorted(kept.get_column("ticker").to_list()) == ["A", "C"]


def test_purge_train_by_label_end_needs_the_column():
    with pytest.raises(ValueError, match="label_end_date"):
        hr.purge_train_by_label_end(pl.DataFrame({"trade_date": [dt.date(2022, 1, 3)]}), HOLDOUT)


def test_assert_no_overlap_catches_a_shared_and_a_touching_boundary():
    train = pl.DataFrame({"trade_date": [dt.date(2025, 7, 1), dt.date(2025, 7, 2)]})
    after = pl.DataFrame({"trade_date": [dt.date(2025, 8, 1)]})
    hr.assert_no_overlap(train, after)  # fine

    with pytest.raises(AssertionError):
        hr.assert_no_overlap(train, pl.DataFrame({"trade_date": [dt.date(2025, 7, 2)]}))
    with pytest.raises(AssertionError):
        hr.assert_no_overlap(after, train)  # train ends after eval starts


# --- the wall -------------------------------------------------------------------


def test_model_spec_still_refuses_the_holdout_on_its_own():
    """The bypass must not have loosened the guard it bypasses."""
    with pytest.raises(ValueError, match="holdout boundary"):
        ModelSpec(period_end="2026-09-16")


def test_widen_to_holdout_refuses_a_date_inside_the_wall():
    spec = ModelSpec()
    with pytest.raises(ValueError, match="nothing to open"):
        hr._widen_to_holdout(spec, PERIOD_END)


def test_widen_to_holdout_moves_only_the_period_end():
    spec = ModelSpec(feature_set="FS1h", preprocess_profile="rank").for_horizon(20)
    widened = hr._widen_to_holdout(spec, "2026-09-16")
    assert widened.period_end == "2026-09-16"
    assert widened.period_start == spec.period_start
    assert widened.feature_set == spec.feature_set
    assert widened.preprocess_profile == spec.preprocess_profile
    assert widened.universe == spec.universe
    assert widened.topk == spec.topk


# --- it cannot search a grid ----------------------------------------------------


def test_adopted_train_config_carries_one_grid_point(tmp_path):
    config = _adopted(tmp_path).train_config()
    assert len(config.grid) == 1
    assert config.grid[0] == {"max_iter": 200, "learning_rate": 0.03}


def test_module_does_not_import_a_grid_or_the_sweep():
    """Structural, on purpose: a sweep that cannot be named cannot be run.

    Parsed rather than grepped — the module docstring says the words, and a
    text search would pass or fail on prose instead of on what is imported.
    """
    import ast

    tree = ast.parse(Path(hr.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    for forbidden in (
        "HGB_CLF_GRID",
        "HGB_CLF_GRID_E1",
        "HGB_REG_GRID",
        "LOGIT_GRID",
        "walk_forward",
        "evaluate_run",
    ):
        assert forbidden not in imported, f"{forbidden} imported by the holdout runner"

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "walk_forward" not in called


# --- the three dates, on a synthetic lake ---------------------------------------


def test_resolve_boundary_counts_the_grid_and_what_has_matured(tmp_path):
    # 260 sessions from 2025-06-02: ~43 before the wall, the rest after.
    sessions = _sessions(dt.date(2025, 6, 2), 260)
    config = _lake(tmp_path)
    _write_prices(config, sessions)
    adopted = _adopted(tmp_path)

    boundary = hr.resolve_boundary(adopted, config, eval_last_formation=None)

    after = [d for d in sessions if str(d) >= HOLDOUT]
    assert boundary.eval_first_formation == str(after[0])
    assert boundary.eval_last_formation == str(sessions[-1])
    assert boundary.n_eval_sessions == len(after)
    assert boundary.n_rebalances_held == len(after[::20])
    # a rebalance matures only if 20 more sessions exist after it
    expected_matured = sum(1 for d in after[::20] if sessions.index(d) + 20 <= len(sessions) - 1)
    assert boundary.n_rebalances_matured == expected_matured
    assert boundary.n_rebalances_matured < boundary.n_rebalances_held
    assert boundary.unmatured_rebalance_dates
    # the training wall
    before = [d for d in sessions if str(d) < HOLDOUT]
    assert boundary.train_last_formation == str(before[-21])
    assert boundary.train_last_label_end < HOLDOUT


def test_resolve_boundary_notes_a_cutoff_past_the_data(tmp_path):
    sessions = _sessions(dt.date(2025, 6, 2), 120)
    config = _lake(tmp_path)
    _write_prices(config, sessions)

    boundary = hr.resolve_boundary(_adopted(tmp_path), config, eval_last_formation="2027-01-04")
    assert boundary.eval_last_formation == str(sessions[-1])
    assert any("가격 마지막 날" in note for note in boundary.notes)


def test_resolve_boundary_finds_the_halted_rows_the_calendar_purge_misses(tmp_path):
    """A name halted across the wall is the whole reason the row purge exists."""
    sessions = _sessions(dt.date(2025, 1, 2), 260)
    # A002 is halted for the 60 sessions leading into the wall, so its d_idx
    # skips them and formation dates well before the calendar purge still land
    # past 2025-08-01.
    wall_index = next(i for i, d in enumerate(sessions) if str(d) >= HOLDOUT)
    halt_days = set(sessions[wall_index - 60 : wall_index])
    config = _lake(tmp_path)
    _write_prices(config, sessions, halted={"A002": halt_days})

    boundary = hr.resolve_boundary(_adopted(tmp_path), config, eval_last_formation=None)
    assert boundary.n_train_rows_dropped_by_label_end > 0
    assert boundary.n_train_tickers_dropped_by_label_end == 1

    # The contrast is the point: the same calendar with no halt has nothing for
    # the row purge to do, so what it found above is the halt and only the halt.
    clean = _lake(tmp_path / "clean")
    _write_prices(clean, sessions)
    clean_boundary = hr.resolve_boundary(_adopted(tmp_path), clean, eval_last_formation=None)
    assert clean_boundary.n_train_rows_dropped_by_label_end == 0
    assert clean_boundary.train_last_formation == boundary.train_last_formation


def test_resolve_boundary_is_json_round_trippable(tmp_path):
    """``boundary.json`` is the record; it has to survive being written."""
    sessions = _sessions(dt.date(2025, 6, 2), 200)
    config = _lake(tmp_path)
    _write_prices(config, sessions)
    boundary = hr.resolve_boundary(_adopted(tmp_path), config, eval_last_formation=None)

    from dataclasses import asdict

    text = json.dumps(asdict(boundary), default=str)
    assert json.loads(text)["holdout_start"] == HOLDOUT


# --- the gate -------------------------------------------------------------------


N_GATE_NAMES = 20
UNTRADABLE_RANK = 2  # third by score, so dropping it does not move a top-2 list


def _gate_frame() -> pl.DataFrame:
    """Forty-one sessions, twenty names, hand-set returns.

    Forty-one so the h20 grid has three points and turnover is a real 0 rather
    than an unmeasurable NaN. Twenty names because S needs a cross-section to
    cut into deciles.
    ``p_raw`` ranks the names identically on both dates, so the buy list is
    stable (turnover 0) and the cost term drops out — which leaves E, E' and I
    as plain averages that can be checked by eye.
    """
    dates = _sessions(dt.date(2025, 8, 1), 41)  # 3 points on the h20 grid
    rows = []
    for day in dates:
        for i in range(N_GATE_NAMES):
            rows.append(
                {
                    "trade_date": day,
                    "ticker": f"T{i:03d}",
                    "market": "KOSPI",
                    "p_raw": 1.0 - 0.01 * i,
                    "tradable": i != UNTRADABLE_RANK,
                    "raw_label_20d": 0.02 - 0.001 * i,
                    "raw_label_idx_20d": 0.03 - 0.001 * i,
                }
            )
    return pl.DataFrame(rows)


def test_gate_metrics_reads_the_two_benchmarks_and_two_universes(tmp_path):
    adopted = _adopted(tmp_path, k=2, cost_bps_roundtrip=0.0)
    gate = hr.gate_metrics(_gate_frame(), adopted)

    # E: top-2 by score -> T000 (0.020) and T001 (0.019), eqw label
    assert gate.e == pytest.approx(0.0195)
    # E': the untradable name ranks third, so the top-2 list is unchanged
    assert gate.e_prime == pytest.approx(0.0195)
    # I: same list, index label -> 0.030 and 0.029
    assert gate.i == pytest.approx(0.0295)
    assert gate.s > 0
    assert gate.verdict == "A"
    assert gate.n_obs_base == 41 * N_GATE_NAMES
    assert gate.n_obs_tradable == 41 * (N_GATE_NAMES - 1)
    assert gate.n_rebalances_i == 3
    assert gate.turnover_tradable == pytest.approx(0.0)


def test_gate_metrics_refuses_a_panel_without_the_index_label(tmp_path):
    frame = _gate_frame().drop("raw_label_idx_20d")
    with pytest.raises(ValueError, match="index_bench"):
        hr.gate_metrics(frame, _adopted(tmp_path, k=2))


def test_gate_metrics_verdict_is_c_when_e_is_not_positive(tmp_path):
    frame = _gate_frame().with_columns(pl.col("raw_label_20d") - 0.5)
    gate = hr.gate_metrics(frame, _adopted(tmp_path, k=2, cost_bps_roundtrip=0.0))
    assert gate.e < 0
    assert gate.i > 0  # the index side is still positive — A and C would collide
    assert gate.s > 0
    assert gate.verdict == "C"
