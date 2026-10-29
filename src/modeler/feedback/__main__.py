"""Explicit-input CLI. No globbing, performance reads, or source writes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .core import Batch, FeedbackError, canonical_bytes, read_bytes, sha256_bytes
from .issues import _validate_mapping, compile_mapping
from .store import _immutable, check_output_root, commit_batch


def _verify_document_commit(document: Path, commit: str, blob: str) -> None:
    """The supplied commit must actually contain the pinned document blob."""
    repo = subprocess.run(
        ["git", "-C", str(document.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if repo.returncode:
        raise FeedbackError("issues document must belong to its pinned Git repository")
    root = Path(repo.stdout.strip()).resolve()
    relative = document.resolve().relative_to(root).as_posix()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{commit}:{relative}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode or result.stdout.strip() != blob:
        raise FeedbackError("my commit does not contain the pinned issue document blob")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "observe", "mapping"))
    parser.add_argument("--kind", choices=("briefing", "shadow"), default="briefing")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=os.environ.get("PREDICTION_FEEDBACK_ROOT")
    )
    document_group = parser.add_mutually_exclusive_group(required=True)
    document_group.add_argument("--issues-document", type=Path)
    document_group.add_argument("--mapping-file", type=Path)
    parser.add_argument("--mapping-sha256")
    parser.add_argument("--my-commit", required=True)
    parser.add_argument("--issues-blob", required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_root is None:
            raise FeedbackError("--output-root or PREDICTION_FEEDBACK_ROOT is required")
        if args.command == "observe" and (args.kind != "briefing" or args.checkout is None):
            raise FeedbackError("observe requires briefing source and --checkout")
        if args.command != "mapping" and args.source_root is None:
            raise FeedbackError("--source-root is required for build/observe")
        sources = [p for p in (args.source_root, args.issues_document, args.mapping_file) if p]
        if args.checkout is not None:
            sources.append(args.checkout)
        check_output_root(args.output_root, sources)
        document_batch = Batch()
        if args.mapping_file:
            if args.command == "mapping":
                raise FeedbackError("mapping export requires --issues-document")
            raw = read_bytes(args.mapping_file, root=args.mapping_file.parent, batch=document_batch)
            if sha256_bytes(raw) != args.mapping_sha256:
                raise FeedbackError("mapping file SHA-256 mismatch")
            mapping = json.loads(raw)
            _validate_mapping(mapping)
            if (
                mapping["my_commit"] != args.my_commit
                or mapping["document_blob_sha1"] != args.issues_blob
            ):
                raise FeedbackError("mapping document pins mismatch")
        else:
            document = read_bytes(
                args.issues_document, root=args.issues_document.parent, batch=document_batch
            )
            mapping = compile_mapping(
                document, my_commit=args.my_commit, expected_blob_sha1=args.issues_blob
            )
            _verify_document_commit(args.issues_document, args.my_commit, args.issues_blob)
        if args.command == "mapping":
            raw = canonical_bytes(mapping)
            _immutable(args.output_root / "mapping.json", raw)
            print(json.dumps({"mapping_sha256": sha256_bytes(raw), "rules": len(mapping["rules"])}))
            return 0
        observed_at = datetime.now(UTC).isoformat()
        if args.command == "observe":
            from .receipt import observe_publication

            batch, _ = observe_publication(args.source_root, args.checkout)
        elif args.kind == "briefing":
            from .briefing import read_briefing

            batch = read_briefing(
                args.source_root,
                observed_at=observed_at,
                historical_link=True,
                checkout=args.checkout,
            )
        else:
            from .shadow import read_shadow

            batch = read_shadow(args.source_root, observed_at=observed_at, historical_link=True)
        # The server gets a read-only rules snapshot, not a second human-state document.
        mapping_raw = canonical_bytes(mapping)
        mapping_digest = sha256_bytes(mapping_raw)
        batch.artifacts[mapping_digest] = mapping_raw
        batch.sources.append(
            {
                "path": "issue-mapping",
                "sha256": mapping_digest,
                "hash_kind": "bytes_sha256",
                "bytes": len(mapping_raw),
            }
        )
        source_key = f"{args.kind}:{args.source_root.resolve()}"
        manifest = commit_batch(
            args.output_root,
            batch,
            mapping=mapping,
            source_key=source_key,
            source_roots=sources,
            observed_at=observed_at,
            issue_document_ref=f"{args.my_commit}:{args.issues_blob}",
        )
        print(json.dumps({"run_id": manifest["run_id"], "counts": manifest["counts"]}))
        return 2 if batch.errors else 0
    except (FeedbackError, ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"feedback: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
