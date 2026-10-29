"""Fresh remote observation separated from read-only historical artifact linking."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .core import FeedbackError, canonical_bytes, identity, require_hash, sha256_bytes

REMOTE = "git@github.com:sjleekor/stock_reports.git"
SCHEMA = "prediction-feedback-publication-receipt.v1"
METHOD = "git-ls-remote-ancestry-and-unit-v1"


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], capture_output=True, text=True, check=False, timeout=30
    )


def observe_publication(day_dir: Path, checkout: Path, *, git=_git, clock=None):
    """Verify now; never stamp an old monitor result with a fictional past time.

    This explicit command may contact the configured stock_reports SSH remote. It does
    not fetch, push, change refs, run prediction code, or calculate outcomes.
    """
    from .briefing import _git_unit, read_briefing

    clock = clock or (lambda: datetime.now(UTC))
    started = clock().isoformat()
    before = read_briefing(day_dir, observed_at=started, historical_link=True, checkout=checkout)
    candidates = [p for p in before.publications if p.get("status") == "published_evidence"]
    if before.errors or len(candidates) != 1:
        raise FeedbackError("receipt requires one complete, locally bound publication")
    publication = candidates[0]
    commit = require_hash(publication.get("reports_commit"), length=40)
    actual_remote = git(checkout, "remote", "get-url", "origin")
    if actual_remote.returncode or actual_remote.stdout.strip() != REMOTE:
        raise FeedbackError("receipt remote is not the pinned reports repository")
    listed = git(checkout, "ls-remote", REMOTE, "refs/heads/main")
    if listed.returncode:
        raise FeedbackError("remote verification unavailable")
    fields = listed.stdout.strip().split()
    if len(fields) != 2 or fields[1] != "refs/heads/main":
        raise FeedbackError("unexpected remote verification response")
    head = require_hash(fields[0], length=40)
    if head != commit:
        contained = git(checkout, "merge-base", "--is-ancestor", commit, head)
        if contained.returncode:
            raise FeedbackError(
                "remote head is unavailable locally or does not contain publication"
            )
        bound = _git_unit(
            checkout,
            head,
            publication["report_date"],
            publication["report_sha256"],
            publication["release"],
            publication["unit_sha256"],
            before,
        )
        if bound is None or bound[0] != publication["revision"]:
            raise FeedbackError("remote head no longer contains the exact publication unit")
    verified = clock().isoformat()
    if datetime.fromisoformat(verified) < datetime.fromisoformat(started):
        raise FeedbackError("observation clock moved backwards")
    after = read_briefing(day_dir, observed_at=verified, historical_link=True, checkout=checkout)
    repeated = [
        p
        for p in after.publications
        if p.get("publication_ref") == publication["publication_ref"]
        and p.get("status") == "published_evidence"
    ]
    if after.errors or len(repeated) != 1:
        raise FeedbackError("publication sources changed during observation")
    receipt = {
        "schema": SCHEMA,
        "method": METHOD,
        "verified_by": verified,
        "observation_started_at": started,
        "coverage_status": "snapshot_only",
        "remote_head": head,
        "remote": REMOTE,
        "publication_ref": publication["publication_ref"],
        "collector_code_sha256": sha256_bytes(Path(__file__).read_bytes()),
        **{
            k: publication.get(k)
            for k in (
                "report_date",
                "report_sha256",
                "invocation_id",
                "unit",
                "revision",
                "unit_sha256",
                "reports_commit",
            )
        },
    }
    receipt["receipt_ref"] = identity("publication_receipt", receipt)
    raw = canonical_bytes(receipt)
    digest = sha256_bytes(raw)
    after.artifacts[digest] = raw
    after.sources.append(
        {
            "path": "fresh-publication-receipt",
            "sha256": digest,
            "hash_kind": "bytes_sha256",
            "bytes": len(raw),
        }
    )
    target = repeated[0]
    target.update(
        status="published_verified",
        verified_by=verified,
        receipt_ref=receipt["receipt_ref"],
        receipt_sha256=digest,
        coverage_status="snapshot_only",
    )
    target["verification_observations"] = [
        {
            "receipt_ref": receipt["receipt_ref"],
            "receipt_sha256": digest,
            "verified_by": verified,
            "remote_head": head,
        }
    ]
    target["publication_observation_ref"] = receipt["receipt_ref"]
    verified_refs = set(target.get("prediction_refs", []))
    for prediction in after.predictions:
        if prediction["prediction_ref"] in verified_refs:
            prediction.update(
                publication_status="published_verified",
                verified_by=verified,
                receipt_ref=receipt["receipt_ref"],
                coverage_status="snapshot_only",
            )
    # snapshot_only proves visibility now, not the first valid publication of that day.
    return after, receipt
