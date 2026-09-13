"""Checkpoint fingerprints and coordinator locks for Horizon Scan runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registry_hash(registry: list[dict[str, Any]]) -> str:
    fields = [
        {
            key: row.get(key)
            for key in (
                "hypothesis_id",
                "family",
                "feature",
                "scan_type",
                "cell_type",
                "h_start",
                "h_end",
                "expected_sign",
                "status",
            )
        }
        for row in sorted(registry, key=lambda row: row["hypothesis_id"])
    ]
    return canonical_hash(fields)


def build_checkpoint_fingerprint(
    *,
    registry: list[dict[str, Any]],
    a0_manifest_hash: str | None,
    readiness_population_hash: str | None,
    smoke_family: str | None,
    requested_replicates: int,
    include_holdout: bool,
    holdout_start: str | None,
    scan_engine: str,
    row_order_contract: str,
    sue_nw_order_contract: str,
    sue_permutation_order_contract: str,
    mapping_contract_version: str,
    analysis_kernel_hash: str,
    duckdb_version: str,
    polars_version: str,
    numpy_version: str,
) -> dict[str, Any]:
    return {
        "registry_hash": registry_hash(registry),
        "a0_manifest_hash": a0_manifest_hash,
        "readiness_population_hash": readiness_population_hash,
        "smoke_family": smoke_family,
        "requested_replicates": requested_replicates,
        "include_holdout": include_holdout,
        "holdout_start": holdout_start,
        "scan_engine": scan_engine,
        "row_order_contract": row_order_contract,
        "sue_nw_order_contract": sue_nw_order_contract,
        "sue_permutation_order_contract": sue_permutation_order_contract,
        "mapping_contract_version": mapping_contract_version,
        "analysis_kernel_hash": analysis_kernel_hash,
        "duckdb_version": duckdb_version,
        "polars_version": polars_version,
        "numpy_version": numpy_version,
    }


def validate_checkpoint_fingerprint(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if not actual:
        raise ValueError("checkpoint has no fingerprint; refusing unsafe resume")
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"checkpoint fingerprint is missing fields: {missing}")
    mismatches = {
        key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint fingerprint mismatch: {mismatches}")


def checkpoint_namespace(
    root: Path,
    *,
    phase: str,
    snapshot_date: str,
    source: str,
    config_hash: str,
    experiment: str,
    contract: str,
) -> Path:
    return (
        root
        / f"phase={phase}"
        / f"snapshot_date={snapshot_date}"
        / f"source={source}"
        / f"config_hash={config_hash}"
        / f"experiment={experiment}"
        / f"contract={contract}"
    )


@contextmanager
def coordinator_lock(namespace: Path) -> Iterator[Path]:
    """Hold a non-blocking namespace lock for one coordinator process."""
    namespace.mkdir(parents=True, exist_ok=True)
    lock_path = namespace / ".checkpoint.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"checkpoint namespace is already running: {namespace}") from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_replicate_checkpoint(
    namespace: Path, *, replicate: int, fingerprint: dict[str, Any], payload: dict[str, Any]
) -> Path:
    """Write one replicate atomically; workers never append to a shared file."""
    namespace.mkdir(parents=True, exist_ok=True)
    target = namespace / f"replicate={replicate:03d}.json"
    temp = target.with_suffix(target.suffix + ".tmp")
    body = {"fingerprint": fingerprint, "payload": payload}
    temp.write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(target)
    return target


def load_replicate_checkpoints(
    namespace: Path, *, fingerprint: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    loaded: dict[int, dict[str, Any]] = {}
    if not namespace.exists():
        return loaded
    for path in sorted(namespace.glob("replicate=*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A process can be interrupted between creating and replacing a
            # checkpoint. Ignore only malformed JSON so that replicate is
            # recomputed; a valid checkpoint with the wrong fingerprint must
            # still fail loudly below.
            continue
        validate_checkpoint_fingerprint(body.get("fingerprint", {}), fingerprint)
        payload = body.get("payload")
        if not isinstance(payload, dict) or "replicate" not in payload:
            raise ValueError(f"invalid replicate checkpoint: {path}")
        replicate = int(payload["replicate"])
        if replicate in loaded:
            raise ValueError(f"duplicate replicate checkpoint: {replicate}")
        loaded[replicate] = payload
    return loaded
