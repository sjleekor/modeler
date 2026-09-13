"""`03` §3.3's conditional ECE recheck: which run it builds, and how it judges.

The recheck is not one of the preregistered runs, so two things have to stay
true: it takes its config from the stage's own selection (never from a guess),
and it lands outside the stage directory the selection reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from modeler.models._02_updown_prob import evaluate as ev
from modeler.models._02_updown_prob.experiments import recheck_calibration as rc
from modeler.models._02_updown_prob.experiments import registry as rg
from modeler.models._02_updown_prob.experiments import run_matrix as rm

ADOPTED = {
    "run_id": "E1_h120_y_up-logit_seed0",
    "label": "y_up",
    "model": "logit",
    "feature_set": "FS0",
    "preprocess_profile": "rank",
    "flow_variant": "lag1",
    "seed": 0,
    "grid": "LOGIT_GRID",
}


@pytest.fixture
def results_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    stage = tmp_path / "E2"
    stage.mkdir()
    (stage / "selection.json").write_text(
        json.dumps({"stage": "E2", "by_horizon": {"120": ADOPTED}}), encoding="utf-8"
    )
    return tmp_path


def _fold_metrics(eces: list[float], *, calibrated_col: str = "p_cal") -> pl.DataFrame:
    rows = []
    for fold, ece in enumerate(eces, start=1):
        rows.append({"fold_id": fold, "pred_col": "p_raw", "ece": ece + 0.01, "log_loss": 0.66})
        rows.append({"fold_id": fold, "pred_col": calibrated_col, "ece": ece, "log_loss": 0.65})
    return pl.DataFrame(rows)


def test_the_recheck_takes_its_config_from_the_stages_selection(results_root: Path) -> None:
    run = rc.recheck_run("E2", 120)
    assert (run.label, run.model, run.feature_set) == ("y_up", "logit", "FS0")
    assert (run.preprocess_profile, run.flow_variant, run.seed) == ("rank", "lag1", 0)
    assert run.grid == "LOGIT_GRID"
    # the one thing it changes, and the reason it exists
    assert run.calibrate == "isotonic"
    assert run.is_resolved()


def test_the_recheck_run_id_says_what_it_is(results_root: Path) -> None:
    assert rc.recheck_run("E2", 120).run_id == "E2_h120_recal-isotonic_seed0"


def test_a_missing_selection_stops_instead_of_guessing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        rc.recheck_run("E2", 120)


def test_a_horizon_the_selection_does_not_cover_stops(results_root: Path) -> None:
    with pytest.raises(SystemExit):
        rc.recheck_run("E2", 60)


def test_the_verdict_reads_the_calibrated_column_not_the_raw_one() -> None:
    verdict = rc.calibrated_verdict(_fold_metrics([0.02, 0.02, 0.02, 0.02, 0.02]))
    assert verdict["pred_col"] == "p_cal"
    assert verdict["ece"] == pytest.approx(0.02)
    assert verdict["log_loss"] == pytest.approx(0.65)
    assert verdict["n_folds"] == 5


def test_inside_the_ceiling_clears_and_over_it_does_not() -> None:
    assert rc.calibrated_verdict(_fold_metrics([0.02] * 5))["clears_ceiling"] is True
    assert rc.calibrated_verdict(_fold_metrics([0.04] * 5))["clears_ceiling"] is False
    # the ceiling itself is `03` §3.3's, not a local copy
    assert rc.calibrated_verdict(_fold_metrics([0.02] * 5))["ece_ceiling"] == ev.ECE_CEILING


def test_the_ceiling_is_the_mean_but_the_fold_max_is_reported() -> None:
    verdict = rc.calibrated_verdict(_fold_metrics([0.01, 0.01, 0.01, 0.01, 0.055]))
    assert verdict["ece"] == pytest.approx(0.019)
    assert verdict["clears_ceiling"] is True
    assert verdict["ece_fold_max"] == pytest.approx(0.055)


def test_a_run_without_a_calibrated_column_is_an_error_not_a_pass() -> None:
    with pytest.raises(ValueError):
        rc.calibrated_verdict(_fold_metrics([0.02] * 5, calibrated_col="p_raw2"))
    with pytest.raises(ValueError):
        rc.calibrated_verdict(pl.DataFrame({"fold_id": [1], "ece": [0.02]}))


def test_the_record_lands_outside_the_stage_directory(results_root: Path) -> None:
    run = rc.recheck_run("E2", 120)
    out_root = rm.RESULTS_ROOT / rm.RECHECK_DIR / "E2"
    assert rm.stage_dir("E2", smoke=False) not in (out_root / run.run_id).parents


def _write_recheck(root: Path, stage: str, horizon: int, *, clears: bool, ece: float) -> Path:
    run = rg.Run(
        stage=stage,
        variant=rc.RECHECK_VARIANT,
        horizon=horizon,
        label="y_up",
        model="logit",
        feature_set="FS0",
        calibrate="isotonic",
    )
    out_dir = root / rm.RECHECK_DIR / stage / run.run_id
    out_dir.mkdir(parents=True)
    (out_dir / "run_spec.json").write_text(
        json.dumps({"run": rm._as_json(run.__dict__)}), encoding="utf-8"
    )
    (out_dir / "recheck.json").write_text(
        json.dumps({"run_id": run.run_id, "ece": ece, "clears_ceiling": clears}), encoding="utf-8"
    )
    return out_dir


def test_load_rechecks_keys_the_verdicts_by_horizon(results_root: Path) -> None:
    _write_recheck(results_root, "E2", 120, clears=False, ece=0.0323)
    _write_recheck(results_root, "E2", 60, clears=True, ece=0.028)
    verdicts = rm.load_rechecks("E2")
    assert sorted(verdicts) == [60, 120]
    assert verdicts[120]["clears_ceiling"] is False
    assert verdicts[60]["ece"] == pytest.approx(0.028)


def test_load_rechecks_ignores_a_directory_without_a_run_spec(results_root: Path) -> None:
    out_dir = _write_recheck(results_root, "E2", 120, clears=False, ece=0.0323)
    (out_dir / "run_spec.json").unlink()
    assert rm.load_rechecks("E2") == {}


def test_no_rechecks_is_an_empty_mapping_not_an_error(results_root: Path) -> None:
    assert rm.load_rechecks("E4") == {}


def test_a_dropped_horizon_is_skipped_instead_of_resolved(results_root: Path, capsys) -> None:
    for stage in ("E0", "E1", "E2"):
        directory = results_root / stage
        directory.mkdir(exist_ok=True)
        horizons = (5, 20, 60, 120) if stage != "E2" else (5, 20, 60)
        payload: dict = {
            "stage": stage,
            "by_horizon": {
                str(h): {**ADOPTED, "run_id": f"{stage}_h{h}"} for h in horizons
            },
        }
        if stage == "E2":
            payload["dropped_horizons"] = {
                "120": {"reason": "ECE 0.0321 > 0.03, and 0.0323 after isotonic"}
            }
        (directory / "selection.json").write_text(json.dumps(payload), encoding="utf-8")
    runs = rm.resolve_stage("E3", horizon=None, smoke=False)
    assert {run.horizon for run in runs} == {5, 20, 60}
    assert "h120: skipped" in capsys.readouterr().out
    # and asking for that horizon alone is an empty plan, not a crash
    assert rm.resolve_stage("E3", horizon=120, smoke=False) == []


def test_out_root_redirects_the_cache_lookup(results_root: Path) -> None:
    """A recheck must not read (or be read as) the stage's own run of that id."""
    run = rc.recheck_run("E2", 120)
    spec, train_cfg = rm.spec_for(run, smoke=False), rm.train_config_for(run, smoke=False)
    out_root = rm.RESULTS_ROOT / rm.RECHECK_DIR / "E2"
    out_dir = out_root / run.run_id
    out_dir.mkdir(parents=True)
    (out_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "run": rm._as_json(rg.replace(run, inherits=()).__dict__),
                "spec": rm._as_json(rm._spec_payload(spec)),
                "train_config": rm._as_json(train_cfg.__dict__),
                "model_code_hash": rm.model_code_hash(),
                "smoke": False,
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps({"ece": 0.02}), encoding="utf-8")
    assert rm.cached_record(run, spec, train_cfg, smoke=False, out_root=out_root) is not None
    # the same id under the stage directory is a different record
    assert rm.cached_record(run, spec, train_cfg, smoke=False) is None
