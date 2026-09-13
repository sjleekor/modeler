"""Dependency-aware snapshot selection for research lake inputs.

The selector deliberately treats raw and derived inputs as separate dependency
sets. Phase A0/ A only need the newest complete raw capture; Phase B can opt in
to the intersection with derived-mart snapshots. No caller should silently fall
back to the historical default date.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from modeler.etl.config import DataRoot, EngineOptions, LakeConfig

_SNAPSHOT_RE = re.compile(r"^snapshot_date=(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class SnapshotResolution:
    """The selected snapshot and the evidence used to select it."""

    snapshot_date: str
    source: str
    required_inputs: tuple[str, ...]
    require_derived: bool
    auto_selected: bool
    raw_marker: Path
    derived_tables: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_date": self.snapshot_date,
            "source": self.source,
            "required_inputs": list(self.required_inputs),
            "require_derived": self.require_derived,
            "auto_selected": self.auto_selected,
            "raw_marker": str(self.raw_marker),
            "derived_tables": list(self.derived_tables),
        }


def _read_marker(marker: Path) -> dict:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid snapshot completion marker: {marker}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot completion marker must be an object: {marker}")
    return payload


def _raw_candidate_dirs(root: DataRoot, source: str) -> list[tuple[date, Path]]:
    parent = root.raw / "raw_postgres"
    source_dirs = [p for p in parent.glob("snapshot_date=*/source=" + source) if p.is_dir()]
    candidates: list[tuple[date, Path]] = []
    for source_dir in source_dirs:
        snapshot_dir = source_dir.parent.name
        match = _SNAPSHOT_RE.match(snapshot_dir)
        if match:
            candidates.append((date.fromisoformat(match.group(1)), source_dir))
    return sorted(candidates, reverse=True)


def _derived_available(root: DataRoot, source: str, snapshot: str, table: str) -> bool:
    mart_dir = root.derived / "metric" / f"snapshot_date={snapshot}" / f"source={source}" / table
    return mart_dir.is_dir() and any(mart_dir.rglob("*.parquet"))


def _validate_candidate(
    source_dir: Path,
    snapshot: str,
    required_inputs: tuple[str, ...],
    root: DataRoot,
    source: str,
    require_derived: bool,
) -> tuple[bool, Path, tuple[str, ...]]:
    marker = source_dir / "_manifests" / "_SUCCESS.json"
    if not marker.is_file():
        return False, marker, ()
    payload = _read_marker(marker)
    tables = payload.get("tables", {})
    if not isinstance(tables, dict) or not set(required_inputs).issubset(tables):
        return False, marker, ()
    derived = tuple(
        table
        for table in ("stock_metric_fact", "common_feature_daily_fact")
        if _derived_available(root, source, snapshot, table)
    )
    if require_derived and set(derived) != {"stock_metric_fact", "common_feature_daily_fact"}:
        return False, marker, derived
    return True, marker, derived


def resolve_snapshot(
    root: DataRoot,
    source: str,
    *,
    required_inputs: list[str] | tuple[str, ...],
    require_derived: bool = False,
    snapshot_date: str | None = None,
    official: bool = True,
) -> SnapshotResolution:
    """Resolve a complete snapshot, newest-first unless explicitly overridden.

    Explicit dates are useful for fixtures and reproduction. They are marked
    ``auto_selected=False`` so official scan orchestration can enforce the
    policy without forbidding tests.
    """
    required = tuple(dict.fromkeys(required_inputs))
    if not required:
        raise ValueError("required_inputs must not be empty")

    if snapshot_date is not None:
        candidate = (
            root.raw / "raw_postgres" / f"snapshot_date={snapshot_date}" / f"source={source}"
        )
        candidates = [(date.fromisoformat(snapshot_date), candidate)]
    else:
        candidates = _raw_candidate_dirs(root, source)

    failures: list[str] = []
    for _, source_dir in candidates:
        snapshot = source_dir.parent.name.removeprefix("snapshot_date=")
        try:
            valid, marker, derived = _validate_candidate(
                source_dir, snapshot, required, root, source, require_derived
            )
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if valid:
            return SnapshotResolution(
                snapshot_date=snapshot,
                source=source,
                required_inputs=required,
                require_derived=require_derived,
                auto_selected=snapshot_date is None,
                raw_marker=marker,
                derived_tables=derived,
            )
        failures.append(f"{source_dir}: incomplete required inputs or derived marts")

    mode = "auto-selected" if snapshot_date is None else f"requested {snapshot_date}"
    detail = "; ".join(failures[-5:])
    raise FileNotFoundError(
        f"no complete {source!r} raw snapshot ({mode}) for {required}"
        + (f"; {detail}" if detail else "")
    )


def resolve_config(
    root: DataRoot,
    source: str,
    *,
    required_inputs: list[str] | tuple[str, ...],
    require_derived: bool = False,
    snapshot_date: str | None = None,
    engine: EngineOptions | None = None,
    analysis_config_hash: str | None = None,
) -> tuple[LakeConfig, SnapshotResolution]:
    """Return a snapshot-pinned config plus its selection evidence."""
    resolution = resolve_snapshot(
        root,
        source,
        required_inputs=required_inputs,
        require_derived=require_derived,
        snapshot_date=snapshot_date,
    )
    return (
        LakeConfig(
            root=root,
            snapshot_date=resolution.snapshot_date,
            source=source,
            engine=engine if engine is not None else EngineOptions(),
            analysis_config_hash=analysis_config_hash,
        ),
        resolution,
    )
