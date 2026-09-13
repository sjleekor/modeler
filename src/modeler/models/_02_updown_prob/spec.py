"""ModelSpec for model 02 — the knobs M-0 fixed, with the holdout nailed shut.

Plan: ``docs/dev/20260907_model_experiment/00_overview.md`` §4 (decisions D-1 to
D-6), `01` §2 (horizons), `03` §1 (splits), `05` §3 (snapshot).

Every default here is a decision taken on 2026-09-07, before any result was
seen. The one that is enforced rather than merely defaulted is D-2: selection
stops at formation 2025-07-31 and everything after it is holdout, opened once in
2026-10/11 together with the T1/T2 h60 one-shot. ``__post_init__`` refuses a
later ``period_end`` — the plan calls a mistaken edit here the main risk to the
whole experiment (`06` §8), and a spec that cannot express the mistake is
cheaper than remembering not to make it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.labels import LabelSpec
from modeler.etl.universe import UniverseFilter
from modeler.models._02_updown_prob.features import (
    FEATURE_SET_IDS,
    FLOW_VARIANTS,
    HORIZONS,
    feature_columns,
)

MODEL_ID = "02_updown_prob"

# The pinned lake (`05` §3 / `00` §6.5): every E0-E5 run reads the same marts
# the horizon scan calls canonical. A new snapshot is a holdout-day decision.
SNAPSHOT_DATE = "2026-08-23"
SOURCE = "sj2_remote"

PERIOD_START = "2015-01-02"  # flow<->price join is 100% from here (etl_00 §1.1)
PERIOD_END = "2025-07-31"  # D-2. Not a default to be tuned — a wall.
HOLDOUT_START = "2025-08-01"  # == horizon scan sample.holdout_start

# Primary first, as LabelSpec expects; h20 leads because it is the deployed
# horizon of model 01 and the one comparable to its gate numbers.
SPEC_HORIZONS: tuple[int, ...] = (20, 5, 60, 120)

LABEL_OUTPUTS: tuple[str, ...] = ("rank", "up", "top")  # L-A + L-B + the rank read
PREPROCESS_PROFILES: tuple[str, ...] = ("rank", "tree")
TARGETS: tuple[str, ...] = ("y_up", "y_top", "y_rank")

# D-4 economics. k and the cost are model 01's, so the two are comparable; the
# threshold portfolio is the probability model's own read.
TOPK = 100
COST_BPS_ROUNDTRIP = 60.0
TAU = 0.6


def lake_config(*, analysis_config_hash: str | None = None) -> LakeConfig:
    """The pinned snapshot, as a LakeConfig.

    ``analysis_config_hash`` has to match what the marts were stamped with to
    read them (A0 stamps its contract hash onto every mart it writes), so the
    builder resolves it from the snapshot manifest rather than hard-coding it.
    """
    return LakeConfig(
        root=DataRoot.resolve(market="kr"),
        snapshot_date=SNAPSHOT_DATE,
        source=SOURCE,
        analysis_config_hash=analysis_config_hash,
    )


@dataclass(frozen=True)
class ModelSpec:
    """Build + validation parameters for one model-02 configuration."""

    model_id: str = MODEL_ID
    horizons: tuple[int, ...] = SPEC_HORIZONS
    label: LabelSpec = field(
        default_factory=lambda: LabelSpec(horizons=SPEC_HORIZONS, outputs=LABEL_OUTPUTS)
    )
    feature_set: str = "FS0"
    flow_variant: str = "lag1"  # D-6
    preprocess_profile: str = "rank"
    universe: UniverseFilter = field(default_factory=UniverseFilter)  # D-5
    period_start: str = PERIOD_START
    period_end: str = PERIOD_END
    n_folds: int = 5
    seed: int = 0
    topk: int = TOPK
    cost_bps_roundtrip: float = COST_BPS_ROUNDTRIP
    tau: float = TAU

    def __post_init__(self) -> None:
        if self.period_end > PERIOD_END:
            raise ValueError(
                f"period_end {self.period_end!r} reaches past the holdout boundary "
                f"{PERIOD_END!r} (D-2). Formation dates from {HOLDOUT_START} are opened "
                "once, in 2026-10/11, with T1/T2 — not by editing a spec."
            )
        if self.period_start >= self.period_end:
            raise ValueError(
                f"period_start {self.period_start!r} must precede period_end {self.period_end!r}"
            )
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        unknown = tuple(h for h in self.horizons if h not in HORIZONS)
        if unknown:
            raise ValueError(f"horizons {unknown} outside the preregistered set {HORIZONS} (D-3)")
        if self.feature_set not in FEATURE_SET_IDS:
            raise ValueError(f"feature_set {self.feature_set!r} unknown; valid: {FEATURE_SET_IDS}")
        if self.flow_variant not in FLOW_VARIANTS:
            raise ValueError(f"flow_variant {self.flow_variant!r} unknown; valid: {FLOW_VARIANTS}")
        if self.preprocess_profile not in PREPROCESS_PROFILES:
            raise ValueError(
                f"preprocess_profile {self.preprocess_profile!r} unknown; "
                f"valid: {PREPROCESS_PROFILES}"
            )
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if tuple(self.label.horizons) != tuple(self.horizons):
            raise ValueError(
                f"label horizons {tuple(self.label.horizons)} must equal spec horizons "
                f"{tuple(self.horizons)} — the panel carries one label table for all h"
            )
        if not (0.0 < self.tau < 1.0):
            raise ValueError(f"tau must be in (0,1), got {self.tau}")

    @property
    def primary_horizon(self) -> int:
        return self.horizons[0]

    def for_horizon(self, horizon: int) -> ModelSpec:
        """The same config with ``horizon`` primary — folds are per-h (`03` §1)."""
        if horizon not in self.horizons:
            raise ValueError(f"horizon {horizon} not in {self.horizons}")
        ordered = (horizon, *(h for h in self.horizons if h != horizon))
        return replace(
            self,
            horizons=ordered,
            label=replace(self.label, horizons=ordered),
        )

    def fold_kwargs(self, horizon: int | None = None) -> dict:
        """``walk_forward_splits`` arguments for one horizon.

        ``embargo = purge = h`` and ``holdout_len = 0``: the trailing holdout is
        already outside ``period_end``, so reserving another slice inside the
        selection window would only shrink it.
        """
        h = self.primary_horizon if horizon is None else horizon
        if h not in self.horizons:
            raise ValueError(f"horizon {h} not in {self.horizons}")
        return {
            "horizon": h,
            "embargo": h,
            "purge": h,
            "n_folds": self.n_folds,
            "holdout_len": 0,
        }

    def feature_columns(self, horizon: int | None = None) -> tuple[str, ...]:
        """Feature columns for this config at one horizon."""
        h = self.primary_horizon if horizon is None else horizon
        return feature_columns(self.feature_set, h)

    def label_column(self, target: str, horizon: int | None = None) -> str:
        """Label column name for a target, e.g. ``("y_up", 20) -> "y_up_20d"``."""
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; valid: {TARGETS}")
        h = self.primary_horizon if horizon is None else horizon
        if h not in self.horizons:
            raise ValueError(f"horizon {h} not in {self.horizons}")
        return f"{target}_{h}d"

    def realized_column(self, horizon: int | None = None) -> str:
        """The return-unit label the ranking and economic metrics read (`01` §5)."""
        h = self.primary_horizon if horizon is None else horizon
        if h not in self.horizons:
            raise ValueError(f"horizon {h} not in {self.horizons}")
        return f"raw_label_{h}d"
