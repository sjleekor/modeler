from __future__ import annotations

import json

import pytest
from synthetic_scale import run_benchmark, synthetic_days, synthetic_mapping

from modeler.feedback import partition
from modeler.feedback.core import (
    SOURCE_IDENTITY_VERSION,
    Batch,
    FeedbackError,
    identity,
    sha256_bytes,
)
from modeler.feedback.store import load_index


def commit(root, key, batch, **kwargs):
    return partition.commit_day_batch(
        root,
        batch,
        day=key.split(":", 1)[1],
        mapping=synthetic_mapping(),
        source_key=key,
        source_roots=[],
        observed_at="2026-01-05T10:40:00+09:00",
        issue_document_ref="synthetic-document",
        **kwargs,
    )


def test_two_days_keep_independent_generations_and_evidence(tmp_path):
    root = tmp_path / "partitioned"
    batches = list(synthetic_days(2, 4))
    key, batch = batches[0]
    first = commit(root, key, batch)
    old_catalog = partition.load_catalog(root)
    commit(root, *batches[1])
    catalog = partition.load_catalog(root)
    assert len(catalog["days"]) == 2
    assert catalog["days"]["2026-01-05"] == old_catalog["days"]["2026-01-05"]
    for day, entry in catalog["days"].items():
        assert entry["counts"]["predictions"] == 4
        manifest = root / entry["root"] / "runs" / entry["generation"] / "manifest.json"
        assert sha256_bytes(manifest.read_bytes()) == entry["manifest_sha256"]
        assert len(load_index(root / entry["root"])["predictions"]) == 4
    commit(root, key, batch)
    assert (root / "days/2026-01-05/index/generations" / first["run_id"]).exists()
    assert len(partition.load_catalog(root)["days"]) == 2
    assert len(list((root / "days/2026-01-05/evidence").iterdir())) == 5


@pytest.mark.parametrize("replayed_day", [0, 1])
def test_interrupted_catalog_preserves_old_snapshot_and_replay_repairs(
    tmp_path, monkeypatch, replayed_day
):
    root = tmp_path / "partitioned"
    batches = list(synthetic_days(2, 3))
    commit(root, *batches[0])
    before = (root / "catalog.json").read_bytes()
    original = partition._atomic

    def interrupted(path, raw):
        if path.name == "catalog.json":
            raise OSError("catalog interrupted")
        return original(path, raw)

    monkeypatch.setattr(partition, "_atomic", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        commit(root, *batches[replayed_day])
    assert (root / "catalog.json").read_bytes() == before
    shard = root / "days" / batches[replayed_day][0].split(":", 1)[1]
    assert (shard / "index/current.json").exists()
    if replayed_day == 0:
        old_entry = json.loads(before)["days"]["2026-01-05"]
        current = json.loads((shard / "index/current.json").read_bytes())
        assert current["generation"] != old_entry["generation"]
        old_index = shard / "index/generations" / old_entry["generation"] / "index.json"
        assert sha256_bytes(old_index.read_bytes()) == old_entry["index_sha256"]
    monkeypatch.setattr(partition, "_atomic", original)
    commit(root, *batches[replayed_day])
    assert len(partition.load_catalog(root)["days"]) == replayed_day + 1
    assert len(load_index(shard)["predictions"]) == 3


@pytest.mark.parametrize("field", ["predictions", "publications"])
def test_mixed_day_is_rejected_before_writing(tmp_path, field):
    key, batch = next(synthetic_days(1, 2))
    getattr(batch, field)[0]["report_date"] = "2026-01-06"
    root = tmp_path / "partitioned"
    with pytest.raises(FeedbackError, match="differs"):
        commit(root, key, batch)
    assert not root.exists()


def test_undated_shadow_requires_recomputable_identity(tmp_path):
    fields = {
        "bundle_hash": "a" * 64,
        "record_sha256": "b" * 64,
        "ticker": "SYNTHETIC",
        "market": "KOSPI",
    }
    row = {
        "prediction_ref": identity("shadow", fields),
        "kind": "shadow",
        "source_identity_version": SOURCE_IDENTITY_VERSION,
        "bundle_hash": fields["bundle_hash"],
        "record_sha256": fields["record_sha256"],
        "symbol": fields["ticker"],
        "security_market": fields["market"],
    }
    commit(tmp_path / "valid", "synthetic:2026-01-05", Batch(predictions=[row]))
    row["record_sha256"] = "c" * 64
    with pytest.raises(FeedbackError, match="identity mismatch"):
        commit(tmp_path / "invalid", "synthetic:2026-01-05", Batch(predictions=[row]))
    assert not (tmp_path / "invalid").exists()


def test_unbound_publication_and_unpartitioned_root_are_rejected(tmp_path):
    key, batch = next(synthetic_days(1, 2))
    batch.publications[0]["prediction_refs"] = ["f" * 64]
    with pytest.raises(FeedbackError, match="same-batch"):
        commit(tmp_path / "unbound", key, batch)
    root = tmp_path / "existing"
    (root / "index").mkdir(parents=True)
    key, batch = next(synthetic_days(1, 2))
    with pytest.raises(FeedbackError, match="migration"):
        commit(root, key, batch)
    assert not (root / "days").exists()


def test_source_disjoint_guard_applies_to_partition_parent(tmp_path):
    root = tmp_path / "source"
    key, batch = next(synthetic_days(1, 2))
    with pytest.raises(FeedbackError, match="disjoint"):
        partition.commit_day_batch(
            root / "output",
            batch,
            day="2026-01-05",
            mapping=synthetic_mapping(),
            source_key=key,
            source_roots=[root],
            observed_at="2026-01-05T10:40:00+09:00",
            issue_document_ref="synthetic-document",
        )
    assert not root.exists()


def test_partitioned_benchmark_checkpoints_report_all_daily_commits(tmp_path):
    root = tmp_path / "scale"
    summary = run_benchmark(days=3, rows=3, output_root=root, checkpoints=[1, 3], partitioned=True)
    assert summary["mode"] == "partitioned-daily"
    assert [cell["rows"] for cell in summary["cells"]] == [3, 9]
    assert [cell["occurrences"] for cell in summary["cells"]] == [3, 9]
    assert len(partition.load_catalog(root)["days"]) == 3
    last = summary["cells"][-1]
    assert last["index_bytes"] > last["largest_shard_index_bytes"] > 0
    marker = json.loads((root / "SYNTHETIC_FIXTURE.json").read_bytes())
    assert marker["mode"] == "partitioned-daily"


def test_catalog_writer_lock_prevents_day_write(tmp_path):
    root = tmp_path / "locked"
    key, batch = next(synthetic_days(1, 2))
    with partition._writer(root):
        with pytest.raises(FeedbackError, match="writer is locked"):
            commit(root, key, batch)
    assert not (root / "days").exists()
    assert not (root / "catalog.json").exists()


def test_streaming_guard_rejects_descendant_symlinks(tmp_path):
    root = tmp_path / "partitioned"
    root.mkdir()
    (root / "unrelated-day").symlink_to(tmp_path / "outside", target_is_directory=True)
    key, batch = next(synthetic_days(1, 2))
    with pytest.raises(FeedbackError, match="symlinks"):
        commit(root, key, batch)
    assert not (root / "days").exists()
