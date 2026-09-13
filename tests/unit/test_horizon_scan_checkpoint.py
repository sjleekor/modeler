from __future__ import annotations

import pytest
from modeler.analysis.horizon_scan_checkpoint import (
    coordinator_lock,
    load_replicate_checkpoints,
    write_replicate_checkpoint,
)


def test_corrupt_checkpoint_is_recomputed_instead_of_blocking_resume(tmp_path) -> None:
    namespace = tmp_path / "checkpoints"
    fingerprint = {"contract": "v1"}
    (namespace).mkdir()
    (namespace / "replicate=000.json").write_text('{"fingerprint":', encoding="utf-8")
    write_replicate_checkpoint(
        namespace,
        replicate=1,
        fingerprint=fingerprint,
        payload={"replicate": 1, "value": "kept"},
    )

    loaded = load_replicate_checkpoints(namespace, fingerprint=fingerprint)

    assert set(loaded) == {1}
    assert loaded[1]["value"] == "kept"


def test_coordinator_lock_rejects_a_second_owner(tmp_path) -> None:
    namespace = tmp_path / "checkpoints"
    with coordinator_lock(namespace):
        with pytest.raises(RuntimeError, match="already running"):
            with coordinator_lock(namespace):
                pass
