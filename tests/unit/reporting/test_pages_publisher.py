from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "deploy" / "pages"))
import publish_site
from modeler.reporting.site import SiteBuilder
from modeler.serving.schema import report_template

SEOUL = ZoneInfo("Asia/Seoul")


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _fixture_site(day: date, previous: dict[str, bytes] | None = None) -> dict[str, bytes]:
    report = report_template(
        market="KR", report_date=day.isoformat(),
        decision_at=datetime.combine(day, datetime.min.time().replace(hour=10), SEOUL),
        feature_asof_date=(day.replace(day=max(1, day.day - 1))).isoformat(),
        model_id="kr_daily_h20_v1", model_version="1.0.0",
    )
    report.update(status="partial", synthetic_fixture=True)
    return SiteBuilder(base_path="/market-briefing/", synthetic_fixture=True).render(
        report_date=day, reports=[report],
        opening={"status": "unavailable", "publication": {"status": "unresolved", "evidence": []}},
        previous_files=previous,
    )


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "pages.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    seed.mkdir()
    _run("git", "init", "--bare", "--initial-branch=site", str(bare))
    _run("git", "init", "--initial-branch=site", str(seed))
    _run("git", "-C", str(seed), "config", "user.name", "Publisher Test")
    _run("git", "-C", str(seed), "config", "user.email", "publisher-test@example.invalid")
    public = seed / "public"
    public.mkdir()
    first = _fixture_site(date(2026, 9, 28))
    for name, content in first.items():
        path = public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _run("git", "-C", str(seed), "add", "public")
    _run("git", "-C", str(seed), "commit", "-m", "Initial public fixture")
    _run("git", "-C", str(seed), "remote", "add", "origin", str(bare))
    _run("git", "-C", str(seed), "push", "origin", "site:site")
    _run("git", "clone", "--branch", "site", str(bare), str(checkout))
    _run("git", "-C", str(checkout), "config", "user.name", "Publisher Test")
    _run("git", "-C", str(checkout), "config", "user.email", "publisher-test@example.invalid")
    return bare, checkout


def _config(checkout: Path, projection: Path, bare: Path) -> dict[str, object]:
    return {
        "repository_confirmed": True,
        "checkout_dir": str(checkout),
        "projection_dir": str(projection),
        "remote_name": "origin",
        "expected_remote_url": str(bare),
        "branch": "site",
        "base_path": "/market-briefing/",
    }


def test_publisher_retries_its_journaled_public_commit_after_push_failure(tmp_path: Path, monkeypatch) -> None:
    bare, checkout = _checkout(tmp_path)
    projection = tmp_path / "projection"
    projection.mkdir()
    previous = {path.relative_to(checkout / "public").as_posix(): path.read_bytes()
                for path in (checkout / "public").rglob("*") if path.is_file()}
    rendered = _fixture_site(date(2026, 9, 29), previous)
    for name, content in rendered.items():
        path = projection / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    config = _config(checkout, projection, bare)

    real_git = publish_site.git
    failed = False

    def fail_first_push(path: Path, *args: str, capture: bool = True) -> str:
        nonlocal failed
        if args and args[0] == "push" and not failed:
            failed = True
            raise ValueError("simulated isolated bare remote push failure")
        return real_git(path, *args, capture=capture)

    monkeypatch.setattr(publish_site, "git", fail_first_push)
    with pytest.raises(ValueError, match="simulated isolated bare remote"):
        publish_site.publish(config, allow_synthetic=True)
    monkeypatch.setattr(publish_site, "git", real_git)
    journal = publish_site._state_path(checkout)
    recorded = json.loads(journal.read_text(encoding="utf-8"))
    commit = recorded["commit"]
    assert recorded["parent"] == _run("git", "-C", str(checkout), "rev-parse", "HEAD^")
    assert _run("git", "-C", str(checkout), "status", "--porcelain") == ""

    publish_site.publish(config, allow_synthetic=True)
    assert not journal.exists()
    assert _run("git", "-C", str(checkout), "rev-parse", "HEAD") == commit
    assert _run("git", "--git-dir", str(bare), "rev-parse", "refs/heads/site") == commit


def test_publisher_rejects_unjournaled_local_ahead_commit(tmp_path: Path) -> None:
    bare, checkout = _checkout(tmp_path)
    projection = tmp_path / "projection"
    projection.mkdir()
    for name, content in _fixture_site(date(2026, 9, 29)).items():
        path = projection / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (checkout / "notes.txt").write_text("unreviewed local commit\n", encoding="utf-8")
    _run("git", "-C", str(checkout), "add", "notes.txt")
    _run("git", "-C", str(checkout), "commit", "-m", "Unrelated local commit")
    with pytest.raises(ValueError, match="unjournaled commit"):
        publish_site.publish(_config(checkout, projection, bare), allow_synthetic=True)
    assert not publish_site._state_path(checkout).exists()
