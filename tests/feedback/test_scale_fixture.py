"""Small invariants for the synthetic benchmark; no large benchmark in CI."""

from __future__ import annotations

import json

import pytest
from synthetic_scale import (
    KNOWN_WARNING,
    UNKNOWN_WARNING,
    run_benchmark,
    synthetic_days,
    synthetic_mapping,
)

from modeler.feedback.core import canonical_bytes, sha256_bytes
from modeler.feedback.store import commit_batch, load_index


def test_fixture_is_deterministic_and_has_disjoint_business_day_identities():
    first = list(synthetic_days(6, 125))
    second = list(synthetic_days(6, 125))
    assert [key for key, _ in first] == [key for key, _ in second]
    assert [key for key, _ in first][-1] == "synthetic:2026-01-12"
    refs = set()
    for (_, batch), (_, replay) in zip(first, second, strict=True):
        assert canonical_bytes(batch.predictions) == canonical_bytes(replay.predictions)
        assert batch.artifacts == replay.artifacts
        assert len(batch.predictions) == 125
        assert len(batch.artifacts) == 126
        assert len(batch.publications[0]["prediction_refs"]) == 100
        assert all(row["synthetic_fixture"] for row in batch.predictions)
        assert batch.predictions[0]["provenance"] is batch.predictions[-1]["provenance"]
        assert all(sha256_bytes(raw) == digest for digest, raw in batch.artifacts.items())
        refs.update(row["prediction_ref"] for row in batch.predictions)
        assert batch.predictions[-1]["publication_status"] == "generated"
        assert batch.predictions[-1]["quality_review"] == "not_checked"
    assert len(refs) == 750


def test_warning_mapping_and_rerun_are_stable(tmp_path):
    source_key, batch = next(synthetic_days(1, 25))
    root = tmp_path / "store"
    options = dict(
        mapping=synthetic_mapping(),
        source_key=source_key,
        source_roots=[],
        observed_at=batch.predictions[0]["observed_at"],
        issue_document_ref="synthetic-document",
    )
    commit_batch(root, batch, **options)
    before = load_index(root)
    commit_batch(root, batch, **options)
    after = load_index(root)
    assert before["occurrences"] == after["occurrences"]
    for field, ref in (("predictions", "prediction_ref"), ("publications", "publication_ref")):
        assert [(row[ref], row["first_observed_at"]) for row in before[field]] == [
            (row[ref], row["first_observed_at"]) for row in after[field]
        ]
    assert len(after["predictions"]) == 25
    assert len(after["publications"]) == 1
    assert len(after["occurrences"]) == 20
    mapped = [row for row in after["occurrences"] if row["mapping_status"] == "mapped"]
    unknown = [row for row in after["occurrences"] if row["mapping_status"] == "unmapped"]
    assert len(mapped) == len(unknown) == 10
    assert {row["reason_code"] for row in mapped} == {KNOWN_WARNING}
    assert {row["reason_code"] for row in unknown} == {UNKNOWN_WARNING}
    assert all(row["issue_id"] is None for row in unknown)


def test_checkpoint_retains_all_days_and_reports_bytes(tmp_path):
    root = tmp_path / "scale"
    result = run_benchmark(days=3, rows=4, output_root=root, checkpoints=[1, 3])
    assert result["synthetic_fixture"] is True
    assert result["mode"] == "checkpoint-accumulated-batches"
    assert [cell["rows"] for cell in result["cells"]] == [4, 12]
    assert [cell["occurrences"] for cell in result["cells"]] == [4, 12]
    assert result["cells"][-1]["cumulative_stored_bytes"] > result["cells"][-1]["index_bytes"]
    assert all(cell["peak_rss_bytes"] > 0 for cell in result["cells"])
    index = load_index(root)
    assert len(index["source_states"]) == 2  # Deliberately checkpoint source granularity.
    assert len({row["synthetic_source_key"] for row in index["predictions"]}) == 3
    assert json.loads((root / "SYNTHETIC_FIXTURE.json").read_bytes())["synthetic_fixture"]


def test_size_cap_and_invalid_checkpoint_fail_before_output(tmp_path):
    root = tmp_path / "too-large"
    with pytest.raises(ValueError, match="exceed cap"):
        run_benchmark(days=120, rows=1970, output_root=root, max_estimated_bytes=1)
    assert not root.exists()
    with pytest.raises(ValueError, match="checkpoints"):
        run_benchmark(days=3, rows=4, output_root=root, checkpoints=[1, 2])
    assert not root.exists()
