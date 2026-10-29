"""Reproducible metadata-only store benchmark; no source/label/evaluation adapters.

Run from the repository root with PYTHONPATH=src python tests/feedback/synthetic_scale.py
--days 20 --rows 1970 --checkpoint-days 1,5,20
--output-root /private/tmp/feedback-scale-20
--summary-file /private/tmp/feedback-scale-20-summary.json

Daily mode measures retained generations. --checkpoint-days 1,5,20 commits only
selected accumulated batches, reducing quadratic disk writes. Its source-state
count and elapsed time are deliberately NOT equivalent to daily ingestion.
With --partitioned, checkpoints select reporting only: each day commits once
under days/YYYY-MM-DD and no multi-day Batch is held in memory. Use a fresh
process per run for comparable RSS. Fixtures model code schema shape, not
measured production data distributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

from modeler.feedback.core import (
    ADAPTER_VERSION,
    SOURCE_IDENTITY_VERSION,
    Batch,
    canonical_bytes,
    identity,
    sha256_bytes,
)
from modeler.feedback.issues import compile_mapping
from modeler.feedback.partition import commit_day_batch
from modeler.feedback.store import check_output_root, commit_batch

START = date(2026, 1, 5)
MODEL = "kr_daily_h20_v1"
KNOWN_WARNING = "W20_halt_in_window"
UNKNOWN_WARNING = "SYNTHETIC_unmapped_warning"
DOCUMENT = (
    "| issue_id | 문제·분류 / 대상 | reason_code |\n|---|---|---|\n"
    f"| SYNTHETIC-001 | {MODEL} | {KNOWN_WARNING} |\n"
).encode()
PROVENANCE_KEYS = (
    "bundle_manifest_sha256",
    "prepared_manifest_sha256",
    "bundle_manifest_content_sha256",
    "native_prepare_manifest_content_sha256",
    "input_sha256",
    "native_prepare_manifest_sha256",
    "code_inventory_sha256",
)


def synthetic_mapping() -> dict:
    blob = hashlib.sha1(b"blob " + str(len(DOCUMENT)).encode() + b"\0" + DOCUMENT).hexdigest()
    return compile_mapping(DOCUMENT, my_commit="0" * 40, expected_blob_sha1=blob)


def _digest(text: str) -> str:
    return sha256_bytes(text.encode())


def synthetic_days(days: int, rows: int = 1970) -> Iterator[tuple[str, Batch]]:
    """Yield one deterministic business-day batch at a time, sharing provenance.

    Top 100 have synthetic publication linkage; first 10 have a mapped warning,
    next 10 an unmapped warning. Remaining rows retain unverified internal status.
    Seven provenance digests plus report digest reproduce eight hash fields from
    serving/adapters.py and feedback/briefing.py, with minimal freshness metadata.
    """
    if days < 1 or rows < 1:
        raise ValueError("days and rows must be positive")
    day = START
    for _ in range(days):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        stamp = day.isoformat()
        at = stamp + "T10:40:00+09:00"
        source_key = "synthetic:" + stamp
        provenance = {key: _digest(stamp + key) for key in PROVENANCE_KEYS}
        provenance["freshness"] = {"status": "ok", "synthetic_fixture": True}
        report_hash = _digest(stamp + ":report")
        batch = Batch()
        published_refs = []
        for rank in range(1, rows + 1):
            fields = {
                "market": "KR",
                "security_market": "KOSPI" if rank % 2 else "KOSDAQ",
                "model_id": MODEL,
                "bundle_hash": provenance["bundle_manifest_sha256"],
                "report_sha256": report_hash,
                "invocation_id": _digest(stamp + ":invocation")[:32],
                "symbol": f"SYNTHETIC-{rank:06d}",
            }
            ref = identity("prediction", fields)
            reasons = [KNOWN_WARNING] if rank <= 10 else [UNKNOWN_WARNING] if rank <= 20 else []
            evidence_raw = canonical_bytes(
                {
                    "synthetic_fixture": True,
                    "report_sha256": report_hash,
                    "model_index": 0,
                    "row_index": rank - 1,
                    "row": {
                        "symbol": fields["symbol"],
                        "market": fields["security_market"],
                        "rank": rank,
                        "score": round((rows - rank) / rows, 8),
                        "quality_review": True if rank <= 100 else "not_checked",
                        "quality_reasons": reasons,
                    },
                }
            )
            evidence = sha256_bytes(evidence_raw)
            batch.artifacts[evidence] = evidence_raw
            batch.predictions.append(
                {
                    **fields,
                    "prediction_ref": ref,
                    "kind": "briefing",
                    "model_version": "synthetic.v1",
                    "rank": rank,
                    "score": round((rows - rank) / rows, 8),
                    "report_date": stamp,
                    "feature_asof_date": (day - timedelta(days=1)).isoformat(),
                    "decision_at": stamp + "T10:00:00+09:00",
                    "observed_at": at,
                    "historical_link": False,
                    "quality_reasons": reasons,
                    "quality_review": True if rank <= 100 else "not_checked",
                    "quality_validation": "valid" if rank <= 100 else "not_checked",
                    "checker_version": "synthetic-checker.v1",
                    "source_evidence_sha256": evidence,
                    "evaluation_status": "not_defined",
                    "publication_status": "published_evidence" if rank <= 100 else "generated",
                    "source_identity_version": SOURCE_IDENTITY_VERSION,
                    "adapter_version": ADAPTER_VERSION,
                    "provenance": provenance,
                    "bundle_hash_kind": "file_bytes_sha256",
                    "explain_sha256": None,
                    "synthetic_fixture": True,
                    "synthetic_source_key": source_key,
                }
            )
            if rank <= 100:
                published_refs.append(ref)
        batch.publications.append(
            {
                "publication_ref": identity("synthetic-publication", {"day": stamp}),
                "unit_binding_verified": True,
                "revision": 1,
                "prediction_refs": published_refs,
                "status": "published_evidence",
                "observed_at": at,
                "synthetic_fixture": True,
            }
        )
        source_raw = canonical_bytes(
            {"synthetic_fixture": True, "day": stamp, "rows": rows, "provenance": provenance}
        )
        digest = sha256_bytes(source_raw)
        batch.artifacts[digest] = source_raw
        batch.sources.append(
            {
                "path": f"synthetic/{stamp}/metadata.json",
                "sha256": digest,
                "hash_kind": "bytes_sha256",
                "bytes": len(source_raw),
                "synthetic_fixture": True,
            }
        )
        yield source_key, batch
        day += timedelta(days=1)


def estimate_retained_bytes(days: int, rows: int, checkpoints: list[int]) -> int:
    """Conservative planning envelope, not a prediction of production storage.

    16 KiB per retained prediction generation includes index+diagnostic+warning
    duplication; 4 KiB per row/day covers fixture evidence. No dataset is read.
    """
    return rows * (sum(checkpoints) * 16384 + days * 4096) + len(checkpoints) * 1048576


def _rss_bytes() -> int:
    # macOS reports bytes; Linux and other supported Unix targets report KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _stored_bytes(root: Path) -> int:
    # Match the partition guard: counting evidence must not cache whole-tree paths.
    return sum(
        (Path(directory) / name).stat().st_size
        for directory, _, files in os.walk(root)
        for name in files
    )


def run_benchmark(
    *,
    days: int,
    rows: int,
    output_root: Path,
    checkpoints: list[int] | None = None,
    max_estimated_bytes: int = 2 * 1024**3,
    partitioned: bool = False,
) -> dict:
    if days < 1 or rows < 1:
        raise ValueError("days and rows must be positive")
    selected = list(range(1, days + 1)) if checkpoints is None else checkpoints
    if not selected or selected != sorted(set(selected)) or selected[0] < 1 or selected[-1] != days:
        raise ValueError("checkpoints must be increasing, unique, positive, and end at days")
    estimate = estimate_retained_bytes(days, rows, [1] * days if partitioned else selected)
    if estimate > max_estimated_bytes:
        raise ValueError(f"estimated retained bytes {estimate} exceed cap {max_estimated_bytes}")
    repo = Path(__file__).resolve().parents[2]
    root = check_output_root(output_root, [repo])
    if root.exists() and any(root.iterdir()):
        raise ValueError("benchmark output must be empty to avoid mixing runs")
    mapping = synthetic_mapping()
    summary = {
        "schema": "feedback.synthetic-scale.v1",
        "synthetic_fixture": True,
        "data_origin": "code-schema-only; no real data or evaluation inputs",
        "mode": (
            "partitioned-daily"
            if partitioned
            else "daily" if checkpoints is None else "checkpoint-accumulated-batches"
        ),
        "checkpoint_semantics": (
            "report only; every day commits separately"
            if partitioned
            else "selected accumulated batches commit"
        ),
        "rss_scope": "process lifetime high-water mark, includes fixture and store",
        "estimated_retained_bytes": estimate,
        "max_estimated_bytes": max_estimated_bytes,
        "days": days,
        "rows_per_day": rows,
        "checkpoints": selected,
        "cells": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "SYNTHETIC_FIXTURE.json").write_bytes(canonical_bytes(summary))
    pending = Batch()
    started = time.perf_counter()
    total_index_bytes = 0
    largest_shard_index_bytes = 0
    cumulative_occurrences = 0
    for count, (source_key, batch) in enumerate(synthetic_days(days, rows), 1):
        if not partitioned:
            pending.extend(batch)
            if count not in selected:
                continue
        commit_started = time.perf_counter()
        day = batch.predictions[0]["report_date"]
        commit = commit_day_batch if partitioned else commit_batch
        manifest = commit(
            root,
            batch if partitioned else pending,
            **({"day": day} if partitioned else {}),
            mapping=mapping,
            source_key=(
                source_key
                if partitioned or checkpoints is None
                else f"synthetic:checkpoint:{count}"
            ),
            source_roots=[],
            observed_at=batch.predictions[0]["observed_at"],
            issue_document_ref="synthetic-fixed-document:" + mapping["document_sha256"],
        )
        commit_seconds = time.perf_counter() - commit_started
        pending = Batch()
        shard = root / "days" / day if partitioned else root
        index_path = shard / "index/generations" / manifest["run_id"] / "index.json"
        if partitioned:
            total_index_bytes += index_path.stat().st_size
            largest_shard_index_bytes = max(largest_shard_index_bytes, index_path.stat().st_size)
            cumulative_occurrences += manifest["counts"]["occurrences"]
            if count not in selected:
                continue
        summary["cells"].append(
            {
                "days": count,
                "rows": count * rows if partitioned else manifest["counts"]["predictions"],
                "commit_seconds": commit_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes": _rss_bytes(),
                "index_bytes": total_index_bytes if partitioned else index_path.stat().st_size,
                "largest_shard_index_bytes": largest_shard_index_bytes if partitioned else None,
                "cumulative_stored_bytes": _stored_bytes(root),
                "occurrences": (
                    cumulative_occurrences if partitioned else manifest["counts"]["occurrences"]
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--rows", type=int, default=1970)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--partitioned", action="store_true", help="commit one shard per day")
    parser.add_argument("--checkpoint-days", help="comma-separated commit days, ending at --days")
    parser.add_argument("--max-estimated-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    try:
        summary_file = check_output_root(args.summary_file, [Path(__file__).resolve().parents[2]])
        if summary_file.exists():
            raise ValueError("summary file must not already exist")
        checkpoints = (
            None
            if args.checkpoint_days is None
            else [int(x) for x in args.checkpoint_days.split(",")]
        )
        summary = run_benchmark(
            days=args.days,
            rows=args.rows,
            output_root=args.output_root,
            checkpoints=checkpoints,
            max_estimated_bytes=args.max_estimated_bytes,
            partitioned=args.partitioned,
        )
    except ValueError as exc:
        parser.error(str(exc))
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
