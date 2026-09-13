"""Unit tests for the run matrix, its selection rules, and the runner (PR7).

The registry is a preregistration, so the first test reads the run counts out of
`05_experiment_matrix.md` itself: if the document and the code disagree about
how many runs a stage has, one of them was edited without the other.

The selection tests matter for the same reason. Every rule in `05` §2 is a
sentence that has to mean one thing — "the lower log-loss inside a label, the
economic metric across labels, ties to L-A" — and these pin which one.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import REPO_ROOT
from modeler.models._02_updown_prob.evaluate import ECE_CEILING, PRIMARY_ECONOMIC
from modeler.models._02_updown_prob.experiments import registry as rg
from modeler.models._02_updown_prob.experiments import run_matrix as rm
from modeler.models._02_updown_prob.experiments import selection as sel

MATRIX_DOC = REPO_ROOT / "docs" / "dev" / "20260907_model_experiment" / "05_experiment_matrix.md"


def _doc_run_counts() -> dict[str, int]:
    """The run count column of `05` §1's stage table, per stage.

    A range ("4~8") is a required count plus optional runs, so the low end is
    what the registry must produce by default.
    """
    counts: dict[str, int] = {}
    pattern = re.compile(r"^\|\s*\*{0,2}(E\d)\*{0,2}\s*\|.*?\|\s*([\d~]+)\s*\|")
    for line in MATRIX_DOC.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            counts[match.group(1)] = int(match.group(2).split("~")[0])
    return counts


# --- the matrix matches the document ----------------------------------------


def test_run_counts_match_the_preregistration_document() -> None:
    doc = _doc_run_counts()
    assert doc, f"could not parse the stage table in {MATRIX_DOC}"
    for stage, expected in doc.items():
        assert len(rg.stage_runs(stage)) == expected, f"{stage} run count drifted from `05` §1"
    assert rg.REQUIRED_RUN_COUNTS == doc


def test_the_required_total_is_the_documents_fifty_two() -> None:
    total = sum(len(rg.stage_runs(s)) for s in ("E0", "E1", "E2", "E3", "E4"))
    assert total == 52
    with_optional = sum(
        len(rg.stage_runs(s, include_optional=True)) for s in ("E0", "E1", "E2", "E3", "E4")
    )
    assert with_optional == 60  # `05` §1's "약 52~60"


def test_every_stage_covers_every_horizon() -> None:
    for stage in rg.STAGES:
        horizons = {run.horizon for run in rg.stage_runs(stage)}
        assert horizons == set(rg.HORIZONS), f"{stage} does not cover every h"


def test_run_ids_are_unique_within_a_stage() -> None:
    for stage in rg.STAGES:
        ids = [run.run_id for run in rg.stage_runs(stage, include_optional=True)]
        assert len(ids) == len(set(ids)), f"{stage} has colliding run ids"


def test_only_inheriting_runs_carry_unresolved_fields() -> None:
    for stage in rg.STAGES:
        for run in rg.stage_runs(stage, include_optional=True):
            if run.inherits:
                continue
            assert run.is_resolved(), f"{run.run_id} has no inheritance but is unresolved"
    # E0 answers the profile question, so it cannot inherit one
    assert all(not run.inherits for run in rg.stage_runs("E0"))
    assert {run.preprocess_profile for run in rg.stage_runs("E0")} == {"tree", "rank"}


def test_e1_runs_the_four_label_model_pairs_per_horizon() -> None:
    per_h = [run for run in rg.stage_runs("E1") if run.horizon == 20]
    assert {(r.label, r.model) for r in per_h} == {
        ("y_up", "hgb_clf"),
        ("y_up", "logit"),
        ("y_top", "hgb_clf"),
        ("y_top", "logit"),
    }
    # E1 runs the half grid (`04` §2.1)
    assert {r.grid for r in per_h if r.model == "hgb_clf"} == {rg.GRID_HGB_CLF_E1}


def test_e2_runs_the_three_feature_sets_and_e5_three_seeds() -> None:
    assert {r.feature_set for r in rg.stage_runs("E2") if r.horizon == 5} == {
        "FS1",
        "FS1h",
        "FS2",
    }
    assert {r.seed for r in rg.stage_runs("E5") if r.horizon == 5} == {0, 1, 2}
    assert {r.seed for r in rg.stage_runs("E4") if r.variant == "E4a-seed"} == {1, 2}


def test_every_grid_name_used_is_one_the_runner_knows() -> None:
    used = {
        run.grid
        for stage in rg.STAGES
        for run in rg.stage_runs(stage, include_optional=True)
        if run.grid != rg.SELECTED
    }
    assert used <= set(rm.GRIDS)
    assert set(sel.FULL_GRID.values()) <= set(rm.GRIDS)


# --- resolution ---------------------------------------------------------------


def _selection(stage: str, **entry) -> dict:
    base = {
        "run_id": f"{stage}_x",
        "label": "y_up",
        "model": "hgb_clf",
        "feature_set": "FS0",
        "preprocess_profile": "rank",
        "grid": "HGB_CLF_GRID",
    }
    base.update(entry)
    return {"stage": stage, "by_horizon": {str(h): base for h in rg.HORIZONS}}


def test_a_stage_can_be_narrowed_to_a_horizon_a_variant_or_one_run() -> None:
    """The chunking a long stage needs: E2's six HGB runs are 1.5h each."""
    assert len(rg.stage_runs("E2")) == 12
    assert len(rg.stage_runs("E2", horizon=20)) == 3
    assert len(rg.stage_runs("E2", variant="FS2")) == 4
    one = rg.stage_runs("E2", horizon=20, variant="FS2")
    assert len(one) == 1
    assert one[0].run_id == "E2_h20_FS2_seed0"
    with pytest.raises(ValueError, match="unknown variant"):
        rg.stage_runs("E2", variant="FS9")


def test_narrowing_never_writes_a_selection(tmp_path: Path, capsys, monkeypatch) -> None:
    """A partial stage says what is left instead of deciding on part of it."""
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    for stage in ("E0", "E1"):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "selection.json").write_text(json.dumps(_selection(stage)), encoding="utf-8")
    rm.run_stage("E2", horizon=20, variant="FS2", dry_run=True)
    out = capsys.readouterr().out
    assert "E2: 1 runs" in out
    assert "E2_h20_FS2_seed0" in out
    assert not (tmp_path / "E2" / "selection.json").exists()


def test_resolve_fills_inherited_fields() -> None:
    run = rg.stage_runs("E2", horizon=20)[0]
    assert not run.is_resolved()
    resolved = rg.resolve(run, {"E0": _selection("E0"), "E1": _selection("E1")})
    assert resolved.is_resolved()
    assert (resolved.label, resolved.model, resolved.grid) == ("y_up", "hgb_clf", "HGB_CLF_GRID")
    assert resolved.feature_set == "FS1"  # E2's own axis is not inherited


def test_a_pinned_model_drags_its_own_grid_along() -> None:
    """E4b at h60: the adopted config is a logit, the constraint needs HGB.

    Left alone, the run inherited ``LOGIT_GRID`` — a list of ``C`` values —
    and handed it to ``HistGradientBoostingClassifier``, which raises
    ``TypeError``. The grid has to follow the model the run pinned.
    """
    run = rg.stage_runs("E4", horizon=60, variant="E4b-monotonic")[0]
    assert (run.model, run.grid) == ("hgb_clf", rg.SELECTED)
    logit_at_h60 = _selection("E2", model="logit", grid="LOGIT_GRID")
    resolved = rg.resolve(run, {s: logit_at_h60 for s in ("E0", "E1", "E2")})
    assert resolved.model == "hgb_clf"
    assert resolved.grid == "HGB_CLF_GRID"
    assert resolved.monotonic is True


def test_an_inherited_model_keeps_the_inherited_grid() -> None:
    """The correction is for a pinned model only — E4a must not be touched."""
    run = rg.stage_runs("E4", horizon=60, variant="E4a-seed")[0]
    assert (run.model, run.grid) == (rg.SELECTED, rg.SELECTED)
    logit_at_h60 = _selection("E2", model="logit", grid="LOGIT_GRID")
    resolved = rg.resolve(run, {s: logit_at_h60 for s in ("E0", "E1", "E2")})
    assert (resolved.model, resolved.grid) == ("logit", "LOGIT_GRID")


def test_a_pinned_grid_survives_the_correction() -> None:
    """E1 screens the families on a deliberately small grid (`04` §2.1)."""
    run = next(r for r in rg.stage_runs("E1", horizon=20) if r.model == "hgb_clf")
    assert run.grid == "HGB_CLF_GRID_E1"
    resolved = rg.resolve(run, {"E0": _selection("E0")})
    assert resolved.grid == "HGB_CLF_GRID_E1"


def test_e5_adds_fs3_to_the_inherited_set_instead_of_replacing_it() -> None:
    """E5 is "adopted + 11", so it resolves to ``<inherited>_FS3``."""
    run = rg.stage_runs("E5", horizon=5)[0]
    assert (run.variant, run.feature_set) == ("FS3", rg.SELECTED)
    resolved = rg.resolve(run, {s: _selection(s) for s in ("E0", "E1", "E2")})
    assert resolved.feature_set == "FS0_FS3"

    scoped = rg.stage_runs("E5", horizon=20)[0]
    on_fs1h = {s: _selection(s, feature_set="FS1h") for s in ("E0", "E1", "E2")}
    assert rg.resolve(scoped, on_fs1h).feature_set == "FS1h_FS3"


def test_only_e5_gets_the_fs3_suffix() -> None:
    """E2 names its own feature sets, so the suffix must not touch them."""
    e2 = rg.stage_runs("E2", horizon=20, variant="FS2")[0]
    resolved = rg.resolve(e2, {s: _selection(s) for s in ("E0", "E1")})
    assert resolved.feature_set == "FS2"


def test_resolve_refuses_a_missing_selection_instead_of_defaulting() -> None:
    run = rg.stage_runs("E1", horizon=20)[0]
    with pytest.raises(ValueError, match="selection is missing"):
        rg.resolve(run, {})
    with pytest.raises(ValueError, match="no entry for h"):
        rg.resolve(run, {"E0": {"stage": "E0", "by_horizon": {}}})


# --- is_improvement (`03` §3.3) ----------------------------------------------


def _record(
    *,
    stage: str = "E2",
    horizon: int = 20,
    probability: float,
    economic: float,
    variant: str = "FS1",
    label: str = "y_up",
    model: str = "hgb_clf",
    feature_set: str = "FS1",
    profile: str = "rank",
    seed: int = 0,
    ece: float = 0.01,
    folds: dict[int, float] | None = None,
    fold_std: float = 0.001,
    economic_std: float = 0.001,
) -> sel.RunRecord:
    run = rg.Run(
        stage=stage,
        variant=variant,
        horizon=horizon,
        label=label,
        model=model,
        feature_set=feature_set,
        preprocess_profile=profile,
        seed=seed,
    )
    return sel.RunRecord(
        run=run,
        summary={
            "probability_value": probability,
            "economic_value": economic,
            "probability_fold_std": fold_std,
            "economic_fold_std": economic_std,
            "ece": ece,
            "rank_ic_mean": 0.1,
        },
        fold_log_loss=folds or {},
    )


def test_improvement_needs_log_loss_down_economics_not_down_and_a_fold_majority() -> None:
    incumbent = _record(probability=0.690, economic=0.010, folds={1: 0.69, 2: 0.69, 3: 0.69})
    better = _record(probability=0.680, economic=0.011, folds={1: 0.68, 2: 0.68, 3: 0.69})
    verdict = sel.is_improvement(better, incumbent)
    assert verdict.improved
    assert verdict.folds_better == 2 and verdict.folds_compared == 3

    worse_economics = _record(probability=0.680, economic=0.005, folds={1: 0.68, 2: 0.68, 3: 0.68})
    assert not sel.is_improvement(worse_economics, incumbent).improved
    assert "economics fell" in sel.is_improvement(worse_economics, incumbent).reason

    higher_loss = _record(probability=0.700, economic=0.020, folds={1: 0.70, 2: 0.70, 3: 0.70})
    assert "log-loss did not fall" in sel.is_improvement(higher_loss, incumbent).reason

    one_fold = _record(probability=0.689, economic=0.011, folds={1: 0.68, 2: 0.70, 3: 0.70})
    verdict = sel.is_improvement(one_fold, incumbent)
    assert not verdict.improved
    assert "folds improved" in verdict.reason


def test_an_improvement_smaller_than_the_fold_spread_is_flagged_weak() -> None:
    incumbent = _record(
        probability=0.690, economic=0.010, folds={1: 0.69, 2: 0.69, 3: 0.69}, fold_std=0.05
    )
    candidate = _record(probability=0.689, economic=0.011, folds={1: 0.68, 2: 0.68, 3: 0.69})
    verdict = sel.is_improvement(candidate, incumbent)
    assert verdict.improved and verdict.weak
    assert "weak" in verdict.reason


# --- E0: the profile choice ---------------------------------------------------


def _e0_records(rank_ic_rank: float, rank_ic_tree: float, horizon: int = 20) -> list[sel.RunRecord]:
    out = []
    for profile, value in (("rank", rank_ic_rank), ("tree", rank_ic_tree)):
        record = _record(
            stage="E0",
            horizon=horizon,
            variant=f"MA-{profile}",
            label="y_rank",
            model="hgb_reg",
            feature_set="FS0",
            profile=profile,
            probability=0.69,
            economic=0.01,
        )
        record.summary["rank_ic_mean"] = value
        out.append(record)
    return out


def test_e0_adopts_the_rank_profile_within_the_tolerance() -> None:
    payload = sel.select_e0(_e0_records(0.1400, 0.1436))  # 0.0036 worse, inside 0.005
    assert payload["by_horizon"]["20"]["preprocess_profile"] == "rank"
    assert sel.E0_RANK_IC_TOLERANCE == 0.005


def test_e0_keeps_the_tree_profile_when_rank_loses_too_much() -> None:
    payload = sel.select_e0(_e0_records(0.1300, 0.1436))
    entry = payload["by_horizon"]["20"]
    assert entry["preprocess_profile"] == "tree"
    assert "tolerance" in entry["reason"]


def test_e0_needs_both_profiles() -> None:
    with pytest.raises(ValueError, match="both profiles"):
        sel.select_e0(_e0_records(0.14, 0.14)[:1])


# --- E1: label x model --------------------------------------------------------


def _e1_records(overrides: dict | None = None) -> list[sel.RunRecord]:
    values = {
        ("y_up", "hgb_clf"): (0.680, 0.010),
        ("y_up", "logit"): (0.685, 0.009),
        ("y_top", "hgb_clf"): (0.450, 0.012),
        ("y_top", "logit"): (0.460, 0.008),
    }
    values.update(overrides or {})
    return [
        _record(
            stage="E1",
            variant=f"{label}-{model}",
            label=label,
            model=model,
            feature_set="FS0",
            probability=probability,
            economic=economic,
            economic_std=0.0005,
        )
        for (label, model), (probability, economic) in values.items()
    ]


def test_e1_picks_the_lower_log_loss_inside_a_label() -> None:
    payload = sel.select_e1(_e1_records())
    entry = payload["by_horizon"]["20"]
    assert entry["model_by_label"]["y_up"]["model"] == "hgb_clf"
    assert entry["model_by_label"]["y_top"]["model"] == "hgb_clf"


def test_e1_compares_labels_on_the_economic_metric_not_log_loss() -> None:
    """L-B's log-loss is lower by construction (0.2 positives) — not comparable."""
    payload = sel.select_e1(_e1_records())
    entry = payload["by_horizon"]["20"]
    assert entry["label"] == "y_top"  # its cost-adjusted return is higher
    assert "L-B wins" in entry["reason"]
    assert PRIMARY_ECONOMIC[20] in entry["reason"]


def test_e1_breaks_a_tie_toward_l_a() -> None:
    records = _e1_records({("y_top", "hgb_clf"): (0.450, 0.0102)})  # +0.0002, spread 0.0005
    entry = sel.select_e1(records)["by_horizon"]["20"]
    assert entry["label"] == "y_up"
    assert "inside the fold spread" in entry["reason"]


def test_e1_keeps_l_a_when_it_wins_outright() -> None:
    records = _e1_records({("y_top", "hgb_clf"): (0.450, 0.004), ("y_top", "logit"): (0.46, 0.004)})
    entry = sel.select_e1(records)["by_horizon"]["20"]
    assert entry["label"] == "y_up"
    assert "favours L-A" in entry["reason"]


# --- E2: the feature-set chain ------------------------------------------------


def test_e2_walks_the_chain_and_stacks_fs2_on_the_better_base() -> None:
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.690,
        economic=0.010,
        folds={1: 0.69, 2: 0.69, 3: 0.69},
    )
    e1 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id}}}
    records = [
        _record(
            variant="FS1",
            feature_set="FS1",
            probability=0.688,
            economic=0.011,
            folds={1: 0.688, 2: 0.688, 3: 0.69},
        ),
        _record(
            variant="FS1h",
            feature_set="FS1h",
            probability=0.686,
            economic=0.012,
            folds={1: 0.686, 2: 0.686, 3: 0.689},
        ),
        _record(
            variant="FS2",
            feature_set="FS2",
            probability=0.684,
            economic=0.013,
            folds={1: 0.684, 2: 0.684, 3: 0.688},
        ),
    ]
    payload = sel.select_e2(records, e1, [e1_record])
    entry = payload["by_horizon"]["20"]
    assert entry["feature_set"] == "FS2"
    chain = entry["chain"]
    assert [step["candidate"] for step in chain] == ["FS1", "FS1h", "FS2"]
    assert chain[1]["against"] == "FS1"  # FS1h was compared to the FS1 that had won
    assert chain[2]["against"] == "FS1h"  # ...and FS2 stacked on that winner


def test_e2_keeps_the_previous_set_when_nothing_improves() -> None:
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.680,
        economic=0.020,
        folds={1: 0.68, 2: 0.68, 3: 0.68},
    )
    e1 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id}}}
    records = [
        _record(
            variant="FS1",
            feature_set="FS1",
            probability=0.690,
            economic=0.010,
            folds={1: 0.69, 2: 0.69, 3: 0.69},
        ),
        _record(
            variant="FS2",
            feature_set="FS2",
            probability=0.695,
            economic=0.005,
            folds={1: 0.70, 2: 0.70, 3: 0.70},
        ),
    ]
    entry = sel.select_e2(records, e1, [e1_record])["by_horizon"]["20"]
    assert entry["feature_set"] == "FS0"
    assert all(not step["improved"] for step in entry["chain"])


def test_e2_flags_a_winner_over_the_ece_ceiling_for_recheck() -> None:
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.690,
        economic=0.010,
        folds={1: 0.69},
        ece=0.01,
    )
    e1 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id}}}
    records = [
        _record(
            variant="FS1",
            feature_set="FS1",
            probability=0.680,
            economic=0.011,
            folds={1: 0.68},
            ece=0.05,
        ),
    ]
    payload = sel.select_e2(records, e1, [e1_record])
    assert payload["pending_calibration_recheck"]
    assert f"> {ECE_CEILING}" in payload["pending_calibration_recheck"][0]
    # pending, not dropped: without a recheck the rule has not been applied yet
    assert payload["by_horizon"]["20"]["feature_set"] == "FS1"
    assert payload["dropped_horizons"] == {}


def _over_ceiling_e2() -> tuple[list[sel.RunRecord], dict, sel.RunRecord]:
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.690,
        economic=0.010,
        folds={1: 0.69},
        ece=0.01,
    )
    e1 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id}}}
    records = [
        _record(
            variant="FS1",
            feature_set="FS1",
            probability=0.680,
            economic=0.011,
            folds={1: 0.68},
            ece=0.05,
        ),
    ]
    return records, e1, e1_record


def test_e2_drops_a_horizon_whose_recheck_is_still_over_the_ceiling() -> None:
    records, e1, e1_record = _over_ceiling_e2()
    verdict = {"run_id": "E2_h20_recal-isotonic_seed0", "ece": 0.041, "clears_ceiling": False}
    payload = sel.select_e2(records, e1, [e1_record], {20: verdict})
    # a horizon with no calibrated probability has nothing for a later stage to inherit
    assert "20" not in payload["by_horizon"]
    dropped = payload["dropped_horizons"]["20"]
    assert dropped["feature_set"] == "FS1"
    assert dropped["recheck_run_id"] == "E2_h20_recal-isotonic_seed0"
    assert "`03` §3.3" in dropped["reason"]
    assert payload["pending_calibration_recheck"] == []


def test_e2_keeps_a_horizon_whose_recheck_clears_the_ceiling() -> None:
    records, e1, e1_record = _over_ceiling_e2()
    verdict = {"run_id": "E2_h20_recal-isotonic_seed0", "ece": 0.028, "clears_ceiling": True}
    payload = sel.select_e2(records, e1, [e1_record], {20: verdict})
    entry = payload["by_horizon"]["20"]
    assert entry["feature_set"] == "FS1"
    assert entry["calibration_recheck"] == {
        "run_id": "E2_h20_recal-isotonic_seed0",
        "ece": 0.028,
        "cleared": True,
    }
    assert payload["dropped_horizons"] == {}


def test_a_recheck_for_a_horizon_inside_the_ceiling_changes_nothing() -> None:
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.690,
        economic=0.010,
        folds={1: 0.69},
        ece=0.01,
    )
    e1 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id}}}
    records = [
        _record(
            variant="FS1",
            feature_set="FS1",
            probability=0.680,
            economic=0.011,
            folds={1: 0.68},
            ece=0.01,
        ),
    ]
    verdict = {"run_id": "stale", "ece": 0.9, "clears_ceiling": False}
    payload = sel.select_e2(records, e1, [e1_record], {20: verdict})
    assert payload["by_horizon"]["20"]["feature_set"] == "FS1"
    assert payload["dropped_horizons"] == {}


# --- E3 records, E4 measures the seed spread ---------------------------------


def test_e3_records_the_delta_and_leaves_the_selection_alone() -> None:
    e2_record = _record(
        stage="E2", variant="FS2", feature_set="FS2", probability=0.680, economic=0.012
    )
    e2 = {"by_horizon": {"20": {"run_id": e2_record.run.run_id, "feature_set": "FS2"}}}
    records = [
        _record(
            stage="E3",
            variant="flow-native_t",
            feature_set="FS2",
            probability=0.670,
            economic=0.020,
        )
    ]
    payload = sel.select_e3(records, e2, [e2_record])
    assert payload["by_horizon"] == e2["by_horizon"]  # unchanged
    recorded = payload["recorded"]["h20_flow-native_t"]
    assert recorded["delta_probability"] == pytest.approx(-0.010)
    assert recorded["delta_economic"] == pytest.approx(0.008)


def test_e3_finds_an_incumbent_that_an_earlier_stage_recorded() -> None:
    """E2 adopting nothing leaves the winner in E1's directory, not E2's."""
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.6800,
        economic=0.0120,
    )
    e2 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id, "feature_set": "FS0"}}}
    records = [
        _record(
            stage="E3",
            variant="flow-native_t",
            feature_set="FS0",
            probability=0.6790,
            economic=0.0130,
        )
    ]
    # E2's own records do not contain the winner — the delta has to come from E1's
    recorded = sel.select_e3(records, e2, [], [e1_record])["recorded"]["h20_flow-native_t"]
    assert recorded["delta_probability"] == pytest.approx(-0.001)
    assert recorded["delta_economic"] == pytest.approx(0.001)


def test_e3_reports_nan_when_the_incumbent_record_is_nowhere() -> None:
    e2 = {"by_horizon": {"20": {"run_id": "E1_h20_gone", "feature_set": "FS0"}}}
    records = [
        _record(stage="E3", variant="flow-native_t", probability=0.679, economic=0.013)
    ]
    recorded = sel.select_e3(records, e2, [], [])["recorded"]["h20_flow-native_t"]
    assert recorded["delta_probability"] != recorded["delta_probability"]  # NaN


def test_adopted_records_skips_a_dropped_horizon() -> None:
    kept = _record(stage="E1", variant="y_up-logit", horizon=60, probability=0.66, economic=0.02)
    selection = {
        "by_horizon": {"60": {"run_id": kept.run.run_id}},
        "dropped_horizons": {"120": {"reason": "ECE over the ceiling after isotonic"}},
    }
    found = sel.adopted_records(selection, [kept])
    assert sorted(found) == [60]


def test_e4_does_not_raise_when_the_yardstick_is_an_earlier_stages_run() -> None:
    e1_record = _record(
        stage="E1",
        variant="y_up-hgb_clf",
        feature_set="FS0",
        probability=0.6800,
        economic=0.0120,
    )
    e2 = {"by_horizon": {"20": {"run_id": e1_record.run.run_id}}}
    records = [
        _record(
            stage="E4", variant="E4a-seed", seed=seed, feature_set="FS0",
            probability=probability, economic=0.0120,
        )
        for seed, probability in ((1, 0.6810), (2, 0.6795))
    ]
    payload = sel.select_e4(records, e2, [], [e1_record])
    assert payload["by_horizon"]["20"]["run_id"] == e1_record.run.run_id


def test_e4_measures_the_seed_spread_and_keeps_the_config_by_default() -> None:
    e2_record = _record(
        stage="E2", variant="FS2", feature_set="FS2", probability=0.6800, economic=0.0120
    )
    e2 = {"by_horizon": {"20": {"run_id": e2_record.run.run_id}}}
    records = [
        _record(
            stage="E4",
            variant="E4a-seed",
            seed=1,
            feature_set="FS2",
            probability=0.6810,
            economic=0.0115,
        ),
        _record(
            stage="E4",
            variant="E4a-seed",
            seed=2,
            feature_set="FS2",
            probability=0.6795,
            economic=0.0125,
        ),
        _record(
            stage="E4",
            variant="E4b-monotonic",
            feature_set="FS2",
            probability=0.6798,
            economic=0.0121,
        ),
    ]
    entry = sel.select_e4(records, e2, [e2_record])["by_horizon"]["20"]
    assert entry["n_seeds"] == 3
    assert entry["seed_std_probability"] > 0
    # the monotonic variant is inside the seed spread -> no replacement
    assert entry["replaced_by"] is None
    assert entry["feature_set"] == "FS2"


def test_e4_replaces_only_when_both_primaries_clear_the_seed_spread() -> None:
    e2_record = _record(
        stage="E2", variant="FS2", feature_set="FS2", probability=0.6800, economic=0.0120
    )
    e2 = {"by_horizon": {"20": {"run_id": e2_record.run.run_id}}}
    records = [
        _record(
            stage="E4",
            variant="E4a-seed",
            seed=1,
            feature_set="FS2",
            probability=0.6801,
            economic=0.0120,
        ),
        _record(
            stage="E4",
            variant="E4a-seed",
            seed=2,
            feature_set="FS2",
            probability=0.6799,
            economic=0.0120,
        ),
        _record(
            stage="E4",
            variant="E4b-monotonic",
            feature_set="FS2",
            probability=0.6600,
            economic=0.0200,
        ),
    ]
    entry = sel.select_e4(records, e2, [e2_record])["by_horizon"]["20"]
    assert entry["replaced_by"] == "E4b-monotonic"


# --- E5 ----------------------------------------------------------------------


def test_e5_adopts_fs3_only_beyond_the_seed_spread() -> None:
    e4 = {
        "by_horizon": {
            "20": {
                "run_id": "E2_h20_FS2_seed0",
                "label": "y_up",
                "model": "hgb_clf",
                "feature_set": "FS2",
                "preprocess_profile": "rank",
                "probability_value": 0.6800,
                "economic_value": 0.0120,
                "seed_std_probability": 0.0005,
                "seed_std_economic": 0.0003,
            }
        }
    }
    weak = [
        _record(
            stage="E5",
            variant="FS3",
            feature_set="FS3",
            seed=s,
            probability=0.6798,
            economic=0.0121,
        )
        for s in (0, 1, 2)
    ]
    entry = sel.select_e5(weak, e4)["by_horizon"]["20"]
    assert entry["fs3_adopted"] is False
    assert entry["feature_set"] == "FS2"

    strong = [
        _record(
            stage="E5",
            variant="FS3",
            feature_set="FS3",
            seed=s,
            probability=0.6700,
            economic=0.0200,
        )
        for s in (0, 1, 2)
    ]
    entry = sel.select_e5(strong, e4)["by_horizon"]["20"]
    assert entry["fs3_adopted"] is True
    assert entry["feature_set"] == "FS3"


def test_an_adopted_fs3_entry_names_the_run_it_came_from() -> None:
    """The record has to describe the config it now carries the numbers of.

    Leaving ``run_id`` and ``ece`` on the superseded run made E5's h60 entry
    claim ``E1_h60_y_up-logit_seed0`` at ECE 0.0188 while reporting FS3's
    log-loss — and that entry is what the holdout preregistration reads.
    """
    e4 = {
        "by_horizon": {
            "60": {
                "run_id": "E1_h60_y_up-logit_seed0",
                "label": "y_up",
                "model": "logit",
                "feature_set": "FS0",
                "preprocess_profile": "rank",
                "seed": 0,
                "probability_value": 0.6620,
                "economic_value": 0.0216,
                "ece": 0.0188,
                "seed_std_probability": 0.0,
                "seed_std_economic": 0.0,
            }
        }
    }
    group = [
        _record(
            stage="E5",
            variant="FS3",
            horizon=60,
            feature_set="FS0_FS3",
            seed=s,
            probability=0.6617,
            economic=0.0227,
            ece=0.0211,
        )
        for s in (0, 1, 2)
    ]
    entry = sel.select_e5(group, e4)["by_horizon"]["60"]
    assert entry["fs3_adopted"] is True
    assert entry["run_id"] == "E5_h60_FS3_seed0"
    assert entry["seed"] == 0
    assert entry["ece"] == pytest.approx(0.0211)
    assert entry["feature_set"] == "FS0_FS3"


def test_e5_compares_three_seeds_against_three_seeds() -> None:
    """`05` §2 E5 names the adopted config's three seeds as the comparison.

    Against seed 0 alone this FS3 would lose on probability; against the mean
    of the three it wins, so the basis is not a detail.
    """
    e4 = {
        "by_horizon": {
            "20": {
                "run_id": "E2_h20_FS1h_seed0",
                "label": "y_up",
                "model": "hgb_clf",
                "feature_set": "FS1h",
                "preprocess_profile": "rank",
                "seed": 0,
                "ece": 0.0127,
                "probability_value": 0.6680,  # the best of the three seeds
                "economic_value": 0.0120,
                "seed_probability_values": [0.6680, 0.6700, 0.6720],  # mean 0.6700
                "seed_economic_values": [0.0120, 0.0100, 0.0080],  # mean 0.0100
                "seed_std_probability": 0.0,
                "seed_std_economic": 0.0,
            }
        }
    }
    group = [
        _record(stage="E5", variant="FS3", feature_set="FS1h_FS3", seed=s,
                probability=0.6690, economic=0.0110)
        for s in (0, 1, 2)
    ]
    payload = sel.select_e5(group, e4)["by_horizon"]["20"]
    assert payload["compared_against"]["probability_value"] == pytest.approx(0.6700)
    assert payload["compared_against"]["economic_value"] == pytest.approx(0.0100)
    assert payload["fs3_adopted"] is True


def test_e4_records_the_economic_seed_values_e5_needs() -> None:
    e2_record = _record(
        stage="E2", variant="FS2", feature_set="FS2", probability=0.6800, economic=0.0120
    )
    e2 = {"by_horizon": {"20": {"run_id": e2_record.run.run_id}}}
    seeds = [
        _record(
            stage="E4", variant="E4a-seed", seed=s, feature_set="FS2",
            probability=0.6800, economic=e,
        )
        for s, e in ((1, 0.0130), (2, 0.0110))
    ]
    entry = sel.select_e4(seeds, e2, [e2_record])["by_horizon"]["20"]
    assert entry["seed_economic_values"] == pytest.approx([0.0120, 0.0130, 0.0110])


# --- the runner ---------------------------------------------------------------


def test_spec_and_train_config_follow_the_run_definition() -> None:
    run = rg.resolve(
        rg.stage_runs("E2", horizon=60)[0], {"E0": _selection("E0"), "E1": _selection("E1")}
    )
    spec = rm.spec_for(run, smoke=False)
    config = rm.train_config_for(run, smoke=False)
    assert spec.feature_set == "FS1"
    assert spec.preprocess_profile == "rank"
    assert spec.period_end == "2025-07-31"  # D-2, never shortened for an official run
    assert config.model == "hgb_clf"
    assert config.horizon == 60
    assert len(config.grid) == 16  # the full grid from E2 onward


def test_smoke_shrinks_the_run_but_marks_it_unofficial() -> None:
    run = rg.stage_runs("E0", horizon=20)[0]
    spec = rm.spec_for(run, smoke=True)
    config = rm.train_config_for(run, smoke=True)
    assert spec.period_end == rm.SMOKE_PERIOD_END
    assert spec.n_folds == rm.SMOKE_FOLDS
    assert len(config.grid) == 1
    assert rm.stage_dir("E0", smoke=True).name == rm.SMOKE_DIR
    assert rm.stage_dir("E0", smoke=False).name == "E0"


def test_an_unimplemented_variant_says_why_instead_of_running() -> None:
    run = rg.resolve(
        [r for r in rg.stage_runs("E3", include_optional=True) if r.optional][0],
        {"E0": _selection("E0"), "E1": _selection("E1"), "E2": _selection("E2")},
    )
    with pytest.raises(NotImplementedError, match="F-9.1"):
        rm.execute_run(run, config=rm.lake_config())


def test_dry_run_lists_the_resolved_plan_without_touching_the_lake(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    selection_dir = tmp_path / "E0"
    selection_dir.mkdir()
    (selection_dir / "selection.json").write_text(json.dumps(_selection("E0")), encoding="utf-8")

    assert rm.run_stage("E1", horizon=20, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "E1: 4 runs" in out
    assert "E1_h20_y_up-hgb_clf_seed0" in out
    assert "fits" in out


def test_a_stage_refuses_to_start_without_its_prior_selection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="selection.json is missing"):
        rm.run_stage("E2", dry_run=True)


def _record_a_finished_run(directory: Path, run: rg.Run, *, smoke: bool = False) -> Path:
    """Write what a finished run leaves behind, so the cache has something to find."""
    spec = rm.spec_for(run, smoke=smoke)
    out_dir = directory / (rm.SMOKE_DIR if smoke else run.stage) / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "run": dataclasses.asdict(run),
                "spec": rm._spec_payload(spec),
                "train_config": dataclasses.asdict(rm.train_config_for(run, smoke=smoke)),
                "code_hash": rm.code_hash(),
                "model_code_hash": rm.model_code_hash(),
                "smoke": smoke,
            },
            default=str,
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"probability_metric": "log_loss", "probability_value": 0.5, "run_id": run.run_id}
        ),
        encoding="utf-8",
    )
    return out_dir


def test_a_finished_run_is_reused_so_a_stage_can_be_split(tmp_path: Path, monkeypatch) -> None:
    """`06` §4: a stage can be run a horizon at a time, or resumed after a stop.

    This is what makes an eight-hour stage survivable on a laptop that has to
    move: the runs already recorded are read back, not re-fit.
    """
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    run = rg.stage_runs("E0", horizon=20)[0]
    spec = rm.spec_for(run, smoke=False)
    assert rm.cached_record(run, spec, rm.train_config_for(run, smoke=False), smoke=False) is None

    _record_a_finished_run(tmp_path, run)
    cached = rm.cached_record(run, spec, rm.train_config_for(run, smoke=False), smoke=False)
    assert cached is not None
    assert cached.run.run_id == run.run_id


def test_an_interrupted_run_is_not_treated_as_finished(tmp_path: Path, monkeypatch) -> None:
    """run_spec is written before the fit, summary after — only both means done."""
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    run = rg.stage_runs("E0", horizon=20)[0]
    out_dir = _record_a_finished_run(tmp_path, run)
    (out_dir / "summary.json").unlink()
    assert (
        rm.cached_record(
            run, rm.spec_for(run, smoke=False), rm.train_config_for(run, smoke=False), smoke=False
        )
        is None
    )


def test_the_cache_refuses_a_result_from_different_code_or_a_different_spec(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    run = rg.stage_runs("E0", horizon=20)[0]
    out_dir = _record_a_finished_run(tmp_path, run)
    spec = rm.spec_for(run, smoke=False)

    stored = json.loads((out_dir / "run_spec.json").read_text(encoding="utf-8"))
    stored["model_code_hash"] = "0" * 16
    (out_dir / "run_spec.json").write_text(json.dumps(stored, default=str), encoding="utf-8")
    assert rm.cached_record(run, spec, rm.train_config_for(run, smoke=False), smoke=False) is None

    stored["model_code_hash"] = rm.model_code_hash()
    stored["spec"]["period_end"] = "2024-12-31"  # a different selection window
    (out_dir / "run_spec.json").write_text(json.dumps(stored, default=str), encoding="utf-8")
    assert rm.cached_record(run, spec, rm.train_config_for(run, smoke=False), smoke=False) is None


def test_a_smoke_result_never_satisfies_an_official_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    run = rg.stage_runs("E0", horizon=20)[0]
    _record_a_finished_run(tmp_path, run, smoke=True)
    assert (
        rm.cached_record(
            run, rm.spec_for(run, smoke=False), rm.train_config_for(run, smoke=False), smoke=False
        )
        is None
    )
    assert (
        rm.cached_record(
            run, rm.spec_for(run, smoke=True), rm.train_config_for(run, smoke=True), smoke=True
        )
        is not None
    )


def test_the_dry_run_counts_the_fits_the_sweep_and_refit_actually_do(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """The sweep fits the grid on every fold; only the winner is refit per fold."""
    monkeypatch.setattr(rm, "RESULTS_ROOT", tmp_path)
    selection_dir = tmp_path / "E0"
    selection_dir.mkdir()
    (selection_dir / "selection.json").write_text(json.dumps(_selection("E0")), encoding="utf-8")

    rm.run_stage("E1", horizon=20, dry_run=True)
    out = capsys.readouterr().out
    # hgb_clf: grid 4 x 5 folds + 5 refits = 25 ; logit: 3 x 5 + 5 = 20
    assert "-> 25 fits" in out
    assert "-> 20 fits" in out


def test_the_cache_key_covers_the_model_code_but_not_the_runner() -> None:
    """A run's numbers come from the model code; the runner only drives it.

    Hashing ``experiments/`` into the key would mean an edit to a CLI flag threw
    away a half-finished stage's fits — and it would buy nothing, because what
    the runner *decided* (the resolved run, spec and train config) is compared
    directly.
    """
    from modeler.etl.config import REPO_ROOT

    assert rm.model_code_hash() == rm.model_code_hash()
    assert len(rm.model_code_hash()) == 16
    for name in rm.MODEL_CODE_FILES:
        assert (REPO_ROOT / name).is_file(), name
    assert not [name for name in rm.MODEL_CODE_FILES if "experiments/" in name]
    for name in ("train.py", "features.py", "build_dataset.py", "spec.py"):
        assert any(n.endswith(f"/{name}") for n in rm.MODEL_CODE_FILES), name


def test_code_hash_is_stable_and_covers_the_shared_libraries(monkeypatch) -> None:
    first = rm.code_hash()
    assert first == rm.code_hash()
    assert len(first) == 16
    # the broad hash is provenance, so it must differ from the narrow key
    assert first != rm.model_code_hash()


def test_fold_log_loss_reads_the_primary_prediction_column() -> None:
    frame = pl.DataFrame(
        {
            "fold_id": [1, 1, 2, 2],
            "pred_col": ["p_raw", "p_cal", "p_raw", "p_cal"],
            "log_loss": [0.68, 0.67, 0.69, 0.66],
        }
    )
    assert rm._fold_log_loss(frame) == {1: 0.68, 2: 0.69}
    calibrated_only = frame.filter(pl.col("pred_col") == "p_cal")
    assert rm._fold_log_loss(calibrated_only) == {1: 0.67, 2: 0.66}
