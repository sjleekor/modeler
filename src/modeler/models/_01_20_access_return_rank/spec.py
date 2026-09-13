"""Model 01 spec — feature groups, label spec, universe filter, period.

Declarative knobs for this model's dataset build (00_shared §8: "새 모델 = 피처
그룹 선택 + 라벨 1개 정의 + 유니버스/분할 지정"). Milestone A uses price + flow
only; fin/common/event are added in P7/P8 by extending ``feature_groups``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modeler.etl.labels import LabelSpec
from modeler.etl.splits import walk_forward_splits  # noqa: F401 (re-exported intent)
from modeler.etl.universe import UniverseFilter

# First-experiment window (etl_00 §1.1): flow<->price join is 100% from 2015.
PERIOD_START = "2015-01-02"
PERIOD_END = "2026-06-10"


@dataclass(frozen=True)
class ModelSpec:
    """End-to-end build parameters for model 01."""

    model_id: str = "01_20_access_return_rank"
    feature_groups: tuple[str, ...] = ("px", "flow", "fin")  # expanded from milestone A
    model_type: str = "hgb"  # "ridge" | "elasticnet" | "hgb"
    preprocess_profile: str = "tree"  # "linear" | "tree"
    label: LabelSpec = field(default_factory=lambda: LabelSpec(horizons=(20, 5, 60)))
    universe: UniverseFilter = field(default_factory=UniverseFilter)
    period_start: str = PERIOD_START
    period_end: str = PERIOD_END
    # walk-forward
    n_folds: int = 5
    embargo: int = 20  # = label horizon (etl_00 §5)
    purge: int = 20
    holdout_len: int = 120  # reserve trailing holdout for final eval

    @property
    def primary_horizon(self) -> int:
        return self.label.horizons[0]
