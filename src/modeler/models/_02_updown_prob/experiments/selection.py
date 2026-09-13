"""selection — the between-stage rules, as code that writes ``selection.json``.

Plan: ``docs/dev/20260907_model_experiment/05_experiment_matrix.md`` §2 (each
stage's rule), `03` §3.3 (what counts as an improvement), §3.2 (the two primary
metrics).

The rules live here, not in a reader's head, because the whole matrix rests on
"the answer to stage k is fixed before stage k+1 runs". Every function below
takes the stage's run summaries and returns both the decision and the sentence
that justifies it, so ``selection.json`` explains itself and a later reader
cannot re-derive a different winner from the same numbers.

Nothing here reads a run it was not given, and no rule has a magnitude
threshold that was not preregistered: the only numbers in this file are
``E0_RANK_IC_TOLERANCE`` (`05` §2 E0) and the fold-count majority (`03` §3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from modeler.models._02_updown_prob.evaluate import ECE_CEILING, PRIMARY_ECONOMIC
from modeler.models._02_updown_prob.experiments.registry import FULL_GRID, Run

# `05` §2 E0: adopt the rank profile unless it loses more than this in Rank IC.
E0_RANK_IC_TOLERANCE = 0.005


@dataclass(frozen=True)
class RunRecord:
    """One finished run: its definition, its summary, and its per-fold log-loss."""

    run: Run
    summary: dict
    fold_log_loss: dict[int, float] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return self.run.horizon

    @property
    def probability(self) -> float:
        return float(self.summary.get("probability_value", float("nan")))

    @property
    def economic(self) -> float:
        return float(self.summary.get("economic_value", float("nan")))

    @property
    def ece(self) -> float:
        return float(self.summary.get("ece", float("nan")))


@dataclass(frozen=True)
class Improvement:
    """Whether a candidate improves on an incumbent, and why (`03` §3.3)."""

    improved: bool
    weak: bool
    delta_probability: float
    delta_economic: float
    folds_better: int
    folds_compared: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "improved": self.improved,
            "weak": self.weak,
            "delta_probability": self.delta_probability,
            "delta_economic": self.delta_economic,
            "folds_better": self.folds_better,
            "folds_compared": self.folds_compared,
            "reason": self.reason,
        }


def is_improvement(candidate: RunRecord, incumbent: RunRecord) -> Improvement:
    """`03` §3.3: log-loss down, economics not down, and a fold majority.

    "Not down" for the economic metric is ``>= 0`` deliberately — the primary
    probability metric is what has to move, and the economic one is there to
    stop an improvement that is paid for in cost-adjusted return.

    A gap smaller than the incumbent's fold standard deviation is reported as
    ``weak`` rather than rejected: the plan sets no magnitude threshold, it
    asks for the spread to be printed beside the delta.
    """
    delta_probability = candidate.probability - incumbent.probability
    delta_economic = candidate.economic - incumbent.economic
    shared = sorted(set(candidate.fold_log_loss) & set(incumbent.fold_log_loss))
    folds_better = sum(1 for f in shared if candidate.fold_log_loss[f] < incumbent.fold_log_loss[f])
    majority = max(1, len(shared) - 1) if shared else 0
    spread = float(incumbent.summary.get("probability_fold_std", float("nan")))

    checks = []
    if not delta_probability < 0:
        checks.append(f"log-loss did not fall ({delta_probability:+.5f})")
    if not delta_economic >= 0:
        checks.append(f"economics fell ({delta_economic:+.5f})")
    if shared and folds_better < majority:
        checks.append(f"only {folds_better}/{len(shared)} folds improved (need {majority})")

    improved = not checks
    weak = improved and spread == spread and abs(delta_probability) < spread
    reason = (
        "improvement" + (" (weak: gap below the fold spread)" if weak else "")
        if improved
        else "; ".join(checks)
    )
    return Improvement(
        improved=improved,
        weak=weak,
        delta_probability=delta_probability,
        delta_economic=delta_economic,
        folds_better=folds_better,
        folds_compared=len(shared),
        reason=reason,
    )


def _by_horizon(records: list[RunRecord]) -> dict[int, list[RunRecord]]:
    out: dict[int, list[RunRecord]] = {}
    for record in records:
        out.setdefault(record.horizon, []).append(record)
    return out


def _entry(record: RunRecord, **extra) -> dict:
    """The inheritable part of a decision — what ``registry.resolve`` reads."""
    run = record.run
    return {
        "run_id": run.run_id,
        "label": run.label,
        "model": run.model,
        "feature_set": run.feature_set,
        "preprocess_profile": run.preprocess_profile,
        "flow_variant": run.flow_variant,
        "seed": run.seed,
        "grid": FULL_GRID.get(run.model, run.grid),
        "probability_value": record.probability,
        "economic_value": record.economic,
        "ece": record.ece,
        **extra,
    }


def select_e0(records: list[RunRecord]) -> dict:
    """E0: one preprocess profile for everything after it (`05` §2 E0).

    ``rank`` wins ties and near-ties: the tolerance exists because the point of
    E0 is to unify the preprocessing if it costs nothing, and 0.005 of Rank IC
    is the "nothing" the plan wrote down before seeing a number.
    """
    by_horizon: dict[str, dict] = {}
    notes: list[str] = []
    for horizon, group in sorted(_by_horizon(records).items()):
        profiles = {r.run.preprocess_profile: r for r in group}
        if "rank" not in profiles or "tree" not in profiles:
            raise ValueError(f"E0 h={horizon} needs both profiles, got {sorted(profiles)}")
        rank_ic = {
            profile: float(record.summary.get("rank_ic_mean", float("nan")))
            for profile, record in profiles.items()
        }
        chosen = "rank" if rank_ic["rank"] >= rank_ic["tree"] - E0_RANK_IC_TOLERANCE else "tree"
        by_horizon[str(horizon)] = _entry(
            profiles[chosen],
            preprocess_profile=chosen,
            rank_ic_rank_profile=rank_ic["rank"],
            rank_ic_tree_profile=rank_ic["tree"],
            reason=(
                f"rank profile Rank IC {rank_ic['rank']:.4f} vs tree {rank_ic['tree']:.4f}; "
                f"tolerance {E0_RANK_IC_TOLERANCE}"
            ),
        )
        notes.append(f"h{horizon}: {chosen}")
    return {"stage": "E0", "rule": "`05` §2 E0", "by_horizon": by_horizon, "notes": notes}


def select_e1(records: list[RunRecord]) -> dict:
    """E1: one (label, model) per horizon (`05` §2 E1).

    Two comparisons, in this order. Inside a label, the lower valid log-loss
    wins — same label, same positive rate, so the numbers are comparable.
    Across labels they are not (0.5 vs 0.2 positives), so the economic primary
    decides, and a tie inside the fold spread goes to L-A.
    """
    by_horizon: dict[str, dict] = {}
    notes: list[str] = []
    for horizon, group in sorted(_by_horizon(records).items()):
        per_label: dict[str, RunRecord] = {}
        for label in ("y_up", "y_top"):
            candidates = [r for r in group if r.run.label == label]
            if not candidates:
                continue
            per_label[label] = min(candidates, key=lambda r: _nan_high(r.probability))
        if not per_label:
            raise ValueError(f"E1 h={horizon} produced no runs")

        if len(per_label) == 1:
            chosen_label, winner = next(iter(per_label.items()))
            reason = f"only {chosen_label} ran"
        else:
            up, top = per_label["y_up"], per_label["y_top"]
            spread = float(up.summary.get("economic_fold_std", float("nan")))
            gap = top.economic - up.economic
            tie = spread == spread and abs(gap) < spread
            if tie or not gap > 0:
                winner, chosen_label = up, "y_up"
                reason = f"L-A kept: {PRIMARY_ECONOMIC[horizon]} gap {gap:+.5f} " + (
                    "inside the fold spread" if tie else "favours L-A"
                )
            else:
                winner, chosen_label = top, "y_top"
                reason = f"L-B wins on {PRIMARY_ECONOMIC[horizon]} by {gap:+.5f}"

        by_horizon[str(horizon)] = _entry(
            winner,
            model_by_label={
                label: {"model": r.run.model, "log_loss": r.probability}
                for label, r in per_label.items()
            },
            reason=reason,
        )
        notes.append(f"h{horizon}: {chosen_label} + {winner.run.model}")
    return {"stage": "E1", "rule": "`05` §2 E1", "by_horizon": by_horizon, "notes": notes}


def select_e2(
    records: list[RunRecord],
    e1: dict,
    e1_records: list[RunRecord],
    rechecks: dict[int, dict] | None = None,
) -> dict:
    """E2: walk the FS0 -> FS1 -> {FS1h} -> FS2 chain (`05` §2 E2).

    Partial adoption is not available: if FS2's twelve columns do not improve
    as a block, the previous set stands and the per-pair question becomes E4d.

    ``rechecks`` carries `03` §3.3's calibration verdicts, keyed by horizon. A
    winner over the ECE ceiling is *pending* until its recheck exists, and
    **dropped** once a recheck says it is still over — a horizon with no
    calibrated probability has no adopted config, so it inherits nothing.
    """
    e1_by_horizon = {r.horizon: r for r in e1_records if _is_e1_winner(r, e1)}
    verdicts = rechecks or {}
    by_horizon: dict[str, dict] = {}
    dropped: dict[str, dict] = {}
    notes: list[str] = []
    pending: list[str] = []
    for horizon, group in sorted(_by_horizon(records).items()):
        incumbent = e1_by_horizon.get(horizon)
        if incumbent is None:
            raise ValueError(f"E2 h={horizon} has no E1 winner to compare against")
        sets = {r.run.feature_set: r for r in group}
        steps: list[dict] = []

        base, base_id = incumbent, "FS0"
        for candidate_id in ("FS1", "FS1h"):
            candidate = sets.get(candidate_id)
            if candidate is None:
                continue
            verdict = is_improvement(candidate, base)
            steps.append({"candidate": candidate_id, "against": base_id, **verdict.as_dict()})
            if verdict.improved:
                base, base_id = candidate, candidate_id

        fs2 = sets.get("FS2")
        if fs2 is not None:
            verdict = is_improvement(fs2, base)
            steps.append({"candidate": "FS2", "against": base_id, **verdict.as_dict()})
            if verdict.improved:
                base, base_id = fs2, "FS2"

        extra: dict = {}
        if base.ece > ECE_CEILING:
            verdict = verdicts.get(horizon)
            if verdict is None:
                pending.append(f"h{horizon} {base_id} ECE {base.ece:.4f} > {ECE_CEILING}")
            elif not verdict.get("clears_ceiling"):
                dropped[str(horizon)] = {
                    "feature_set": base_id,
                    "run_id": base.run.run_id,
                    "ece": base.ece,
                    "recheck_run_id": verdict.get("run_id"),
                    "recheck_ece": verdict.get("ece"),
                    "reason": (
                        f"ECE {base.ece:.4f} > {ECE_CEILING}, and {verdict.get('ece'):.4f} "
                        f"after isotonic — dropped by `03` §3.3"
                    ),
                }
                notes.append(f"h{horizon}: dropped (no calibrated probability inside the ceiling)")
                continue
            else:
                extra["calibration_recheck"] = {
                    "run_id": verdict.get("run_id"),
                    "ece": verdict.get("ece"),
                    "cleared": True,
                }
        by_horizon[str(horizon)] = _entry(base, feature_set=base_id, chain=steps, **extra)
        notes.append(f"h{horizon}: {base_id}")
    return {
        "stage": "E2",
        "rule": "`05` §2 E2",
        "by_horizon": by_horizon,
        "dropped_horizons": dropped,
        "notes": notes,
        "pending_calibration_recheck": pending,
    }


def select_e3(
    records: list[RunRecord],
    e2: dict,
    e2_records: list[RunRecord],
    earlier_records: list[RunRecord] | None = None,
) -> dict:
    """E3: records, never adopts (`05` §2 E3).

    A same-session flow value that looks better is not a result, it is an
    unverified data path — the size of the gap is what the collection stream
    would have to justify.

    ``earlier_records`` carries the stages before E2, because an adopted config
    E2 did not change was recorded by E1 (see :func:`adopted_records`).
    """
    e2_by_horizon = adopted_records(e2, e2_records, earlier_records or [])
    deltas: dict[str, dict] = {}
    for horizon, group in sorted(_by_horizon(records).items()):
        incumbent = e2_by_horizon.get(horizon)
        for record in group:
            key = f"h{horizon}_{record.run.variant}"
            deltas[key] = {
                "probability_value": record.probability,
                "economic_value": record.economic,
                "delta_probability": (
                    record.probability - incumbent.probability if incumbent else float("nan")
                ),
                "delta_economic": (
                    record.economic - incumbent.economic if incumbent else float("nan")
                ),
            }
    return {
        "stage": "E3",
        "rule": "`05` §2 E3 — record only",
        "by_horizon": e2.get("by_horizon", {}),
        "recorded": deltas,
        "notes": ["E3 does not change the adopted config"],
    }


def select_e4(
    records: list[RunRecord],
    e2: dict,
    e2_records: list[RunRecord],
    earlier_records: list[RunRecord] | None = None,
) -> dict:
    """E4: the seed yardstick, and the one variant allowed to replace (`05` §2 E4).

    E4a's three seeds give the standard deviation every later claim is measured
    against — including E5's. E4b/c/e replace the adopted config only by beating
    it on *both* primaries by more than that spread.

    ``earlier_records`` as in :func:`select_e3`: at a horizon where E2 adopted
    nothing the yardstick is an E1 run, and without it this raised instead.
    """
    e2_by_horizon = adopted_records(e2, e2_records, earlier_records or [])
    by_horizon: dict[str, dict] = {}
    notes: list[str] = []
    for horizon, group in sorted(_by_horizon(records).items()):
        adopted = e2_by_horizon.get(horizon)
        if adopted is None:
            raise ValueError(f"E4 h={horizon} has no E2 winner")
        seeds = [r for r in group if r.run.variant == "E4a-seed"]
        seed_values = [adopted.probability, *[r.probability for r in seeds]]
        seed_economic = [adopted.economic, *[r.economic for r in seeds]]
        seed_std = _std(seed_values)
        seed_std_economic = _std(seed_economic)

        winner, replaced_by = adopted, None
        for record in group:
            if record.run.variant == "E4a-seed":
                continue
            beats_probability = record.probability < winner.probability - _nan_zero(seed_std)
            beats_economic = record.economic > winner.economic + _nan_zero(seed_std_economic)
            if beats_probability and beats_economic:
                winner, replaced_by = record, record.run.variant

        by_horizon[str(horizon)] = _entry(
            winner,
            seed_std_probability=seed_std,
            seed_std_economic=seed_std_economic,
            n_seeds=len(seed_values),
            seed_probability_values=seed_values,
            # E5 compares three seeds against three seeds (`05` §2 E5), so it
            # needs the economic side of the spread too, not only its sd.
            seed_economic_values=seed_economic,
            replaced_by=replaced_by,
            variants={
                r.run.variant: {
                    "probability_value": r.probability,
                    "economic_value": r.economic,
                }
                for r in group
                if r.run.variant != "E4a-seed"
            },
        )
        notes.append(
            f"h{horizon}: seed std {seed_std:.5f}"
            + (f", replaced by {replaced_by}" if replaced_by else ", adopted config kept")
        )
    return {"stage": "E4", "rule": "`05` §2 E4", "by_horizon": by_horizon, "notes": notes}


def select_e5(records: list[RunRecord], e4: dict) -> dict:
    """E5: FS3 is adopted only if it clears E4a's seed spread on both primaries."""
    by_horizon: dict[str, dict] = {}
    notes: list[str] = []
    for horizon, group in sorted(_by_horizon(records).items()):
        adopted = e4.get("by_horizon", {}).get(str(horizon))
        if adopted is None:
            raise ValueError(f"E5 h={horizon} has no E4 selection")
        seed_std = float(adopted.get("seed_std_probability", float("nan")))
        seed_std_economic = float(adopted.get("seed_std_economic", float("nan")))
        probability = _mean([r.probability for r in group])
        economic = _mean([r.economic for r in group])
        # `05` §2 E5: "the comparison is the adopted config's same three seeds",
        # so both sides are three-seed means. Falling back to the entry's single
        # value keeps an older selection.json readable.
        seed_probabilities = adopted.get("seed_probability_values") or []
        seed_economics = adopted.get("seed_economic_values") or []
        against_probability = (
            _mean(seed_probabilities) if seed_probabilities else adopted["probability_value"]
        )
        against_economic = _mean(seed_economics) if seed_economics else adopted["economic_value"]
        beats_probability = probability < against_probability - _nan_zero(seed_std)
        beats_economic = economic > against_economic + _nan_zero(seed_std_economic)
        adopt = bool(beats_probability and beats_economic)
        entry = dict(adopted)
        entry["compared_against"] = {
            "probability_value": against_probability,
            "economic_value": against_economic,
            "basis": "three-seed mean of the adopted config",
        }
        if adopt:
            # The adopted config is now the FS3 run, so the entry has to name it.
            # Leaving run_id and ece on the superseded run made the record claim
            # a config whose numbers it no longer carried.
            winner = min(group, key=lambda r: r.run.seed)
            entry["run_id"] = winner.run.run_id
            entry["seed"] = winner.run.seed
            entry["ece"] = winner.ece
            entry["feature_set"] = winner.run.feature_set
            entry["probability_value"] = probability
            entry["economic_value"] = economic
        entry["fs3_adopted"] = adopt
        entry["fs3_probability_value"] = probability
        entry["fs3_economic_value"] = economic
        entry["reason"] = (
            "FS3 adopted: both primaries beyond the seed spread"
            if adopt
            else "FS3 recorded only: did not clear the seed spread on both primaries"
        )
        by_horizon[str(horizon)] = entry
        notes.append(f"h{horizon}: {'FS3 adopted' if adopt else 'FS3 recorded'}")
    return {"stage": "E5", "rule": "`05` §2 E5", "by_horizon": by_horizon, "notes": notes}


def build_selection(
    stage: str,
    records: list[RunRecord],
    prior: dict[str, dict] | None = None,
    prior_records: dict[str, list[RunRecord]] | None = None,
    rechecks: dict[int, dict] | None = None,
) -> dict:
    """Dispatch to the stage's rule. ``prior`` holds earlier ``selection.json``s."""
    prior = prior or {}
    prior_records = prior_records or {}
    if stage == "E0":
        return select_e0(records)
    if stage == "E1":
        return select_e1(records)
    if stage == "E2":
        return select_e2(records, prior["E1"], prior_records.get("E1", []), rechecks)
    earlier = [*prior_records.get("E1", []), *prior_records.get("E0", [])]
    if stage == "E3":
        return select_e3(records, prior["E2"], prior_records.get("E2", []), earlier)
    if stage == "E4":
        return select_e4(records, prior["E2"], prior_records.get("E2", []), earlier)
    if stage == "E5":
        return select_e5(records, prior["E4"])
    raise ValueError(f"unknown stage {stage!r}")


def adopted_records(selection: dict, *pools: list[RunRecord]) -> dict[int, RunRecord]:
    """The record behind each horizon's adopted run, wherever it was recorded.

    A stage that adopts nothing leaves the incumbent standing, and that
    incumbent's record sits in the directory of whichever stage ran it — E2's
    h5 and h60 winners are E1 runs. Looking the run up by id across every pool
    is what lets a later stage compare against it; searching only the previous
    stage's own directory finds nothing and silently reports a NaN delta.

    A horizon the selection dropped (`03` §3.3) has no ``run_id`` and is simply
    absent from the result.
    """
    wanted = {
        int(horizon): entry["run_id"]
        for horizon, entry in selection.get("by_horizon", {}).items()
        if entry.get("run_id")
    }
    by_run_id = {run_id: horizon for horizon, run_id in wanted.items()}
    found: dict[int, RunRecord] = {}
    for pool in pools:
        for record in pool:
            horizon = by_run_id.get(record.run.run_id)
            if horizon is not None:
                found.setdefault(horizon, record)
    return found


def _is_e1_winner(record: RunRecord, e1: dict) -> bool:
    entry = e1.get("by_horizon", {}).get(str(record.horizon))
    return bool(entry) and entry["run_id"] == record.run.run_id


def _is_e2_winner(record: RunRecord, e2: dict) -> bool:
    entry = e2.get("by_horizon", {}).get(str(record.horizon))
    return bool(entry) and entry["run_id"] == record.run.run_id


def _nan_high(value: float) -> float:
    """Sort NaN last when minimizing."""
    return float("inf") if value != value else value


def _nan_zero(value: float) -> float:
    return 0.0 if value != value else value


def _mean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return float(np.mean(finite)) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
