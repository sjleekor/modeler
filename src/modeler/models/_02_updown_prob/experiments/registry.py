"""registry — the preregistered run matrix E0-E5, fixed in code.

Plan: ``docs/dev/20260907_model_experiment/05_experiment_matrix.md`` §1 (the
stage table), §2 (each stage's runs and its selection rule).

Why the matrix is code and not a table someone reads: a stage's runs and the
rule that picks its winner both have to exist *before* the numbers do. A run
list assembled at the prompt is a search; this list is a preregistration, and
``test_updown_registry`` fails if it stops matching the document.

Later stages do not repeat the earlier stages' answers — they inherit them.
A field set to :data:`SELECTED` is filled in from the previous stage's
``selection.json`` by :func:`resolve`, so E2 cannot silently run a label E1 did
not choose. Nothing here reads a metric; picking the winner is
``selection.py``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from modeler.models._02_updown_prob import features as fx

SELECTED = "@selected"

STAGES: tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4", "E5")
HORIZONS: tuple[int, ...] = (5, 20, 60, 120)

# Grid names, resolved to the tuples in ``train.py`` by the runner. Named rather
# than inlined so a run definition stays a description of *what* runs.
GRID_HGB_REG = "HGB_REG_GRID"
GRID_HGB_CLF_E1 = "HGB_CLF_GRID_E1"
GRID_HGB_CLF = "HGB_CLF_GRID"
GRID_LOGIT = "LOGIT_GRID"

# Each family's full grid from E2 onward (`04` §2.1). A run that pins its own
# model instead of inheriting it has to bring the matching grid along: E4b
# forces ``hgb_clf`` because a monotonic constraint is an HGB feature, and at
# h60 the inherited config is a logit whose grid is a list of ``C`` values.
FULL_GRID: dict[str, str] = {"hgb_clf": GRID_HGB_CLF, "logit": GRID_LOGIT}

# E5's only variant. Named because :func:`resolve` has to recognise it.
FS3_VARIANT = "FS3"


@dataclass(frozen=True)
class Run:
    """One run: a horizon, a config, and what it is there to answer."""

    stage: str
    variant: str
    horizon: int
    label: str  # "y_rank" | "y_up" | "y_top"
    model: str  # "hgb_reg" | "hgb_clf" | "logit"
    feature_set: str
    flow_variant: str = "lag1"
    preprocess_profile: str = "rank"
    seed: int = 0
    grid: str = GRID_HGB_CLF_E1
    calibrate: str = "none"
    cal_target: str | None = None
    monotonic: bool = False
    optional: bool = False
    inherits: tuple[str, ...] = ()  # which stage's selection fills SELECTED fields

    @property
    def run_id(self) -> str:
        """Stable directory name — every knob that distinguishes the run."""
        parts = [
            self.stage,
            f"h{self.horizon}",
            self.variant,
            f"seed{self.seed}",
        ]
        return "_".join(parts)

    def is_resolved(self) -> bool:
        return SELECTED not in (self.label, self.model, self.feature_set, self.preprocess_profile)


def _e0_runs() -> tuple[Run, ...]:
    """E0 (8): model 01's baseline in this frame, on both preprocess profiles.

    Both profiles run so the ``rank`` profile can be adopted (or not) on
    evidence rather than assumption — everything after E0 uses one of them. The
    isotonic calibration onto ``y_up`` is what makes this the *probability*
    baseline the E1 classifiers are compared against.
    """
    runs: list[Run] = []
    for horizon in HORIZONS:
        for profile in ("tree", "rank"):
            runs.append(
                Run(
                    stage="E0",
                    variant=f"MA-{profile}",
                    horizon=horizon,
                    label="y_rank",
                    model="hgb_reg",
                    feature_set="FS0",
                    preprocess_profile=profile,
                    grid=GRID_HGB_REG,
                    calibrate="isotonic",
                    cal_target="y_up",
                )
            )
    return tuple(runs)


def _e1_runs() -> tuple[Run, ...]:
    """E1 (16): which label and which model, per horizon. FS0 fixed."""
    runs: list[Run] = []
    for horizon in HORIZONS:
        for label in ("y_up", "y_top"):
            for model, grid in (("hgb_clf", GRID_HGB_CLF_E1), ("logit", GRID_LOGIT)):
                runs.append(
                    Run(
                        stage="E1",
                        variant=f"{label}-{model}",
                        horizon=horizon,
                        label=label,
                        model=model,
                        feature_set="FS0",
                        preprocess_profile=SELECTED,
                        grid=grid,
                        inherits=("E0",),
                    )
                )
    return tuple(runs)


def _e2_runs() -> tuple[Run, ...]:
    """E2 (12): does a screened feature set improve on FS0?

    FS2 stacks on whichever of FS1/FS1h wins — the runner resolves that from
    E2's own partial results, so the stacking is mechanical (`05` §2).
    """
    runs: list[Run] = []
    for horizon in HORIZONS:
        for feature_set in ("FS1", "FS1h", "FS2"):
            runs.append(
                Run(
                    stage="E2",
                    variant=feature_set,
                    horizon=horizon,
                    label=SELECTED,
                    model=SELECTED,
                    feature_set=feature_set,
                    preprocess_profile=SELECTED,
                    grid=SELECTED,
                    inherits=("E0", "E1"),
                )
            )
    return tuple(runs)


def _e3_runs() -> tuple[Run, ...]:
    """E3 (4 + 4 optional): does the timing rule change the answer?

    E3-a swaps the flow columns to their same-session value. E3-b (the strict
    ``fin_pit`` rewrite) is marked optional because its input does not exist
    yet — F-9.1 in the parallel stream — and `05` §2 says E3 records rather
    than adopts either way.
    """
    runs: list[Run] = []
    for horizon in HORIZONS:
        runs.append(
            Run(
                stage="E3",
                variant="flow-native_t",
                horizon=horizon,
                label=SELECTED,
                model=SELECTED,
                feature_set=SELECTED,
                preprocess_profile=SELECTED,
                flow_variant="native_t",
                grid=SELECTED,
                inherits=("E0", "E1", "E2"),
            )
        )
    for horizon in HORIZONS:
        runs.append(
            Run(
                stage="E3",
                variant="fin_pit-strict",
                horizon=horizon,
                label=SELECTED,
                model=SELECTED,
                feature_set=SELECTED,
                preprocess_profile=SELECTED,
                grid=SELECTED,
                optional=True,
                inherits=("E0", "E1", "E2"),
            )
        )
    return tuple(runs)


def _e4_runs() -> tuple[Run, ...]:
    """E4 (12 + optional): how stable is the adopted config?

    E4a's two extra seeds set the yardstick every later improvement is measured
    against, so they are required; E4b (monotonic constraints) is the one
    variant that can *replace* the adopted config. E4c/d/e are optional in the
    plan and stay optional here.
    """
    runs: list[Run] = []
    for horizon in HORIZONS:
        for seed in (1, 2):
            runs.append(
                Run(
                    stage="E4",
                    variant="E4a-seed",
                    horizon=horizon,
                    label=SELECTED,
                    model=SELECTED,
                    feature_set=SELECTED,
                    preprocess_profile=SELECTED,
                    seed=seed,
                    grid=SELECTED,
                    inherits=("E0", "E1", "E2"),
                )
            )
    for horizon in HORIZONS:
        runs.append(
            Run(
                stage="E4",
                variant="E4b-monotonic",
                horizon=horizon,
                label=SELECTED,
                model="hgb_clf",  # a monotonic constraint is an HGB feature
                feature_set=SELECTED,
                preprocess_profile=SELECTED,
                grid=SELECTED,
                monotonic=True,
                inherits=("E0", "E1", "E2"),
            )
        )
    for horizon in HORIZONS:
        runs.append(
            Run(
                stage="E4",
                variant="E4e-no_isna",
                horizon=horizon,
                label=SELECTED,
                model=SELECTED,
                feature_set=SELECTED,
                preprocess_profile=SELECTED,
                grid=SELECTED,
                optional=True,
                inherits=("E0", "E1", "E2"),
            )
        )
    return tuple(runs)


def _e5_runs() -> tuple[Run, ...]:
    """E5 (12): the parallel stream's 11 columns, on three seeds.

    Compared against the same three seeds of the adopted config, which E4a
    already produced. Needs the four 08-23 marts rebuilt first (`02` §1.5).
    """
    runs: list[Run] = []
    for horizon in HORIZONS:
        for seed in (0, 1, 2):
            runs.append(
                Run(
                    stage="E5",
                    variant="FS3",
                    horizon=horizon,
                    label=SELECTED,
                    model=SELECTED,
                    # Resolved to "<E2's winner>_FS3" — the adopted columns
                    # plus FS3's eleven, horizon-subset iff the base is.
                    feature_set=SELECTED,
                    preprocess_profile=SELECTED,
                    seed=seed,
                    grid=SELECTED,
                    inherits=("E0", "E1", "E2"),
                )
            )
    return tuple(runs)


STAGE_RUNS: dict[str, tuple[Run, ...]] = {
    "E0": _e0_runs(),
    "E1": _e1_runs(),
    "E2": _e2_runs(),
    "E3": _e3_runs(),
    "E4": _e4_runs(),
    "E5": _e5_runs(),
}

# `05` §1's run counts, excluding the optional runs. The test compares against
# these, and they are what the document's table says.
REQUIRED_RUN_COUNTS: dict[str, int] = {"E0": 8, "E1": 16, "E2": 12, "E3": 4, "E4": 12, "E5": 12}


def stage_runs(
    stage: str,
    *,
    horizon: int | None = None,
    variant: str | None = None,
    include_optional: bool = False,
) -> list[Run]:
    """The runs of one stage, optionally narrowed to a horizon and/or a variant.

    The two filters are what makes a long stage divisible: ``--h`` takes one
    horizon, ``--variant`` one column of the matrix, and the pair takes a single
    run. Narrowing never changes what a run *is* — the selection is still
    written only when the whole stage is present.
    """
    if stage not in STAGE_RUNS:
        raise ValueError(f"unknown stage {stage!r}; valid: {STAGES}")
    runs = [r for r in STAGE_RUNS[stage] if include_optional or not r.optional]
    if horizon is not None:
        if horizon not in HORIZONS:
            raise ValueError(f"unknown horizon {horizon}; valid: {HORIZONS}")
        runs = [r for r in runs if r.horizon == horizon]
    if variant is not None:
        known = {r.variant for r in STAGE_RUNS[stage]}
        if variant not in known:
            raise ValueError(f"unknown variant {variant!r} for {stage}; valid: {sorted(known)}")
        runs = [r for r in runs if r.variant == variant]
    return runs


def resolve(run: Run, selections: dict[str, dict]) -> Run:
    """Fill a run's :data:`SELECTED` fields from earlier stages' selections.

    ``selections`` maps a stage id to its ``selection.json`` payload, whose
    ``by_horizon`` section carries one entry per horizon. A missing selection is
    an error rather than a default: running E2 without E1's answer would be
    choosing the label by omission.
    """
    key = str(run.horizon)
    inherited: dict[str, object] = {}
    for stage in run.inherits:
        payload = selections.get(stage)
        if payload is None:
            raise ValueError(
                f"{run.run_id} inherits from {stage}, whose selection is missing — "
                f"run --stage {stage} first"
            )
        entry = payload.get("by_horizon", {}).get(key)
        if entry is None:
            raise ValueError(f"{stage} selection has no entry for h={run.horizon}")
        inherited.update(entry)

    updates: dict[str, object] = {}
    for field_name in ("label", "model", "feature_set", "preprocess_profile", "grid"):
        if getattr(run, field_name) != SELECTED:
            continue
        if field_name not in inherited:
            raise ValueError(
                f"{run.run_id} needs {field_name!r} from {run.inherits}, which did not provide it"
            )
        updates[field_name] = inherited[field_name]
    resolved = replace(run, **updates) if updates else run

    # An inherited grid belongs to the inherited *model*. A run that overrode
    # the model has to take that family's grid with it, or the runner hands
    # ``C`` values to a gradient-booster and dies on TypeError — which is what
    # E4b at h60 did, where the adopted config is a logit.
    if run.grid == SELECTED and run.model != SELECTED:
        expected = FULL_GRID.get(resolved.model)
        if expected is not None and resolved.grid != expected:
            resolved = replace(resolved, grid=expected)

    # E5 asks "the adopted config, plus FS3's eleven columns" (`05` §2 E5), so
    # it inherits the feature set and then adds to it. Plain "FS3" would not do:
    # it is FS2 + 11, and at h5 and h60 the adopted config is FS0, so it would
    # smuggle back the 28 columns E2 rejected and confound the comparison.
    if run.variant == FS3_VARIANT and run.feature_set == SELECTED:
        resolved = replace(resolved, feature_set=f"{resolved.feature_set}{fx.FS3_SUFFIX}")
    return resolved
