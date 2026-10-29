"""Fresh observations use synthetic local git objects and a fake remote response."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime, timedelta

import pytest
from test_briefing import build_case
from test_store import mapping

from modeler.feedback import receipt
from modeler.feedback.briefing import read_briefing
from modeler.feedback.core import FeedbackError
from modeler.feedback.store import commit_batch, load_index


def runner(checkout, *args):
    if args[:2] == ("remote", "get-url"):
        return subprocess.CompletedProcess(args, 0, receipt.REMOTE + "\n", "")
    if args[:1] == ("ls-remote",):
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return subprocess.CompletedProcess(args, 0, head + "\trefs/heads/main\n", "")
    if args[:1] == ("merge-base",):
        return subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True, text=True, check=False
        )
    raise AssertionError(f"unexpected git mutation or network call: {args}")


def clock():
    values = iter(
        [datetime(2026, 10, 30, 1, tzinfo=UTC), datetime(2026, 10, 30, 1, 0, 1, tzinfo=UTC)]
    )
    return lambda: next(values)


def fingerprint(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_fresh_receipt_does_not_rewrite_sources_or_historical_time(tmp_path):
    day, checkout = build_case(tmp_path)
    before = fingerprint(day)
    batch, record = receipt.observe_publication(day, checkout, git=runner, clock=clock())
    assert record["verified_by"] == "2026-10-30T01:00:01+00:00"
    assert record["coverage_status"] == "snapshot_only"
    assert batch.publications[0]["status"] == "published_verified"
    assert batch.predictions[0]["receipt_ref"] == record["receipt_ref"]
    assert before == fingerprint(day)
    assert (
        read_briefing(day, observed_at=record["verified_by"], checkout=checkout).publications[0][
            "verified_by"
        ]
        is None
    )


@pytest.mark.parametrize("failure", ["remote", "missing", "ancestry"])
def test_remote_failure_never_creates_success_receipt(tmp_path, failure):
    day, checkout = build_case(tmp_path)

    def fail(repo, *args):
        if failure == "remote" and args[:1] == ("ls-remote",):
            return subprocess.CompletedProcess(args, 1, "", "unavailable")
        if failure in {"missing", "ancestry"} and args[:1] == ("ls-remote",):
            return subprocess.CompletedProcess(args, 0, "f" * 40 + "\trefs/heads/main\n", "")
        if args[:1] == ("merge-base",):
            return subprocess.CompletedProcess(args, 128 if failure == "missing" else 1, "", "")
        return runner(repo, *args)

    with pytest.raises(FeedbackError):
        receipt.observe_publication(day, checkout, git=fail, clock=clock())
    assert not (day / "publication-receipt.json").exists()


def test_receipt_survives_historical_reingest_with_no_new_timestamp(tmp_path):
    day, checkout = build_case(tmp_path)
    batch, record = receipt.observe_publication(day, checkout, git=runner, clock=clock())
    root = tmp_path / "output"
    kwargs = dict(
        mapping=mapping(),
        source_key="day",
        source_roots=[day, checkout],
        observed_at=record["verified_by"],
        issue_document_ref="fixed-document",
    )
    commit_batch(root, batch, **kwargs)
    commit_batch(
        root, read_briefing(day, observed_at=record["verified_by"], checkout=checkout), **kwargs
    )
    index = load_index(root)
    assert len(index["predictions"]) == len(index["publications"]) == 1
    assert index["publications"][0]["verified_by"] == record["verified_by"]
    assert len(index["publications"][0]["verification_observations"]) == 1
    assert index["predictions"][0]["receipt_ref"] == record["receipt_ref"]


def test_backwards_clock_is_rejected(tmp_path):
    day, checkout = build_case(tmp_path)
    now = datetime(2026, 10, 30, tzinfo=UTC)
    times = iter([now, now - timedelta(seconds=1)])
    with pytest.raises(FeedbackError, match="backwards"):
        receipt.observe_publication(day, checkout, git=runner, clock=lambda: next(times))


def test_source_change_during_remote_verification_is_rejected(tmp_path):
    day, checkout = build_case(tmp_path)

    def race(repo, *args):
        result = runner(repo, *args)
        if args[:1] == ("ls-remote",):
            report = day / f"report-{day.name}.json"
            report.write_bytes(report.read_bytes() + b" ")
        return result

    with pytest.raises(FeedbackError, match="changed"):
        receipt.observe_publication(day, checkout, git=race, clock=clock())


@pytest.mark.parametrize("change_unit", [True, False])
def test_remote_descendant_must_keep_exact_unit(tmp_path, change_unit):
    day, checkout = build_case(tmp_path)
    if change_unit:
        subprocess.run(
            ["git", "-C", str(checkout), "rm", "-r", "reports"], check=True, capture_output=True
        )
    else:
        (checkout / "note.txt").write_text("synthetic later commit")
        subprocess.run(["git", "-C", str(checkout), "add", "note.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "synthetic descendant",
        ],
        check=True,
        capture_output=True,
    )
    if change_unit:
        with pytest.raises(FeedbackError, match="exact publication"):
            receipt.observe_publication(day, checkout, git=runner, clock=clock())
    else:
        batch, _ = receipt.observe_publication(day, checkout, git=runner, clock=clock())
        assert batch.publications[0]["status"] == "published_verified"
