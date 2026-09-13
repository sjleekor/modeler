"""dataset_manifest — per-model reproducibility record (00_shared §4).

Every per-model dataset (L2b) writes a ``dataset_manifest.json`` pinning exactly
what produced it: snapshot, lake roots, feature groups, label spec, universe
filter, period, code revision, and row count. This is the single source of truth
for "this model == this snapshot + this mart + this label spec" — for comparison,
audit, and rebuild (00_shared §4).

See ``00_shared`` §4 and ``etl_03_implementation_plan.md`` §4 (P5).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from modeler.etl.config import LakeConfig


def current_git_sha(default: str = "unknown") -> str:
    """Best-effort short git SHA of the working tree (HEAD); ``default`` on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or default
    except (subprocess.SubprocessError, OSError):
        return default


@dataclass
class DatasetManifest:
    """Reproducibility metadata for one model dataset build (00_shared §4)."""

    model_id: str
    snapshot_date: str
    lake: dict[str, str]
    feature_groups: list[str]
    label_spec: dict
    universe_filter: dict
    period: dict[str, str]
    code_rev: str
    row_count: int
    created_at: str | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str, ensure_ascii=False)

    def write(self, path: Path) -> Path:
        """Write the manifest as pretty JSON, creating parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path


def build_manifest(
    *,
    model_id: str,
    config: LakeConfig,
    feature_groups: list[str],
    label_spec: dict,
    universe_filter: dict,
    period: dict[str, str],
    row_count: int,
    mart_root: str | None = None,
    created_at: str | None = None,
    code_rev: str | None = None,
    extra: dict | None = None,
) -> DatasetManifest:
    """Assemble a :class:`DatasetManifest` from a build's parameters.

    ``created_at`` is accepted as a parameter (rather than stamped here) so the
    caller controls timestamp provenance and tests stay deterministic.
    """
    lake = {
        "raw": str(config.raw_root),
    }
    if mart_root is not None:
        lake["feature_mart"] = mart_root
    return DatasetManifest(
        model_id=model_id,
        snapshot_date=config.snapshot_date,
        lake=lake,
        feature_groups=feature_groups,
        label_spec=label_spec,
        universe_filter=universe_filter,
        period=period,
        code_rev=code_rev if code_rev is not None else current_git_sha(),
        row_count=row_count,
        created_at=created_at,
        extra=extra or {},
    )
