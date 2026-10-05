"""Coordinator wiring for the private (owner-only) projection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import test_daily_coordinator_e2e as e2e
import test_daily_multiday_publish as mp
from modeler.serving import daily_coordinator

RANKED_ADAPTER = '''from modeler.serving.schema import report_template

def _infer(context):
    report = report_template(market=context.market, report_date=context.report_date.isoformat(),
        decision_at=context.decision_at, feature_asof_date=context.feature_asof_date,
        model_id=context.model_id, model_version=context.model_version)
    report["status"] = "partial"
    report["synthetic_fixture"] = True
    report["rankings"] = [{"rank": 1, "symbol": "PRIV001", "name": "비공개종목", "score": 0.123456}]
    return report

def infer_kr_daily(context):
    return _infer(context)

def infer_us_model(context):
    return _infer(context)
'''


def _setup_private(tmp_path: Path, monkeypatch, *, private_root: Path | None = None):
    monkeypatch.setattr(e2e, "ADAPTER", RANKED_ADAPTER)
    release, config = e2e._setup(tmp_path)
    body = json.loads(config.read_text())
    body["private_projection_root"] = str(private_root or tmp_path / "private")
    e2e._write(config, body)
    return release, config


def _tree_text(root: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "replace") for p in root.rglob("*") if p.is_file())


def test_render_writes_private_view_and_state(tmp_path: Path, monkeypatch) -> None:
    release, config = _setup_private(tmp_path, monkeypatch)
    mp._day(release, config, mp.D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    private = tmp_path / "private" / "2026-09-29"
    page = (private / "reports" / "2026-09-29" / "index.html").read_text()
    assert "PRIV001" in page and "0.1235" in page and "비공개 — 게시 금지" in page
    assert (private / "PRIVATE_DO_NOT_PUBLISH.txt").is_file()
    state = json.loads((tmp_path / "runs" / "2026-09-29" / "coordinator-render.json").read_text())
    assert state["private_projection_dir"] == str(private)
    assert state["private_manifest_sha256"] == hashlib.sha256((private / "private-manifest.json").read_bytes()).hexdigest()
    # The coordinator no longer builds a public projection at all.
    assert not (tmp_path / "site").exists() and "projection_dir" not in state


def test_private_disabled_by_default(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    mp._day(release, config, mp.D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    state = json.loads((tmp_path / "runs" / "2026-09-29" / "coordinator-render.json").read_text())
    assert state["private_projection_dir"] is None and state["private_manifest_sha256"] is None
    assert not (tmp_path / "private").exists()


def test_overlapping_private_root_is_rejected(tmp_path: Path, monkeypatch) -> None:
    release, config = _setup_private(tmp_path, monkeypatch)
    world = mp._enable_publisher(tmp_path, config)
    runs, checkout = tmp_path / "runs", world.checkout
    base = json.loads(config.read_text())
    assert daily_coordinator._config(config)["private_projection_root"] == str(tmp_path / "private")
    candidates = (runs, runs / "x", tmp_path, checkout, checkout / "private",
                  tmp_path / "private-link")
    for bad in candidates:
        if bad.name == "private-link":
            bad.symlink_to(checkout)
        body = {**base, "private_projection_root": str(bad)}
        path = e2e._write(tmp_path / "bad.json", body)
        with pytest.raises(ValueError):
            daily_coordinator._config(path)
    inside = e2e._write(tmp_path / "bad.json", {**base, "private_projection_root": str(checkout / "p")})
    mp._cli(release, inside, "select", mp.D1, "2026-09-29T09:30:00+09:00", expect=1)


def test_two_days_private_list_and_reports_repository_never_sees_the_private_view(
        tmp_path: Path, monkeypatch) -> None:
    release, config = _setup_private(tmp_path, monkeypatch)
    world = mp._enable_publisher(tmp_path, config)
    mp._day(release, config, mp.D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert mp._cli(release, config, "publish", mp.D1, "2026-09-29T10:00:00+09:00")["status"] == "published"
    mp._prepare_d2(tmp_path, config)
    mp._day(release, config, mp.D2, "2026-09-30T09:30:00+09:00", "2026-09-30T10:00:00+09:00")
    assert mp._cli(release, config, "publish", mp.D2, "2026-09-30T10:00:00+09:00")["status"] == "published"
    private2 = tmp_path / "private" / "2026-09-30"
    archive = (private2 / "archive" / "index.html").read_text()
    assert "2026-09-29" in archive and "2026-09-30" in archive
    manifest = json.loads((private2 / "private-manifest.json").read_text())
    assert [r["report_date"] for r in manifest["reports"]] == ["2026-09-29", "2026-09-30"]
    render2 = json.loads((tmp_path / "runs" / "2026-09-30" / "coordinator-render.json").read_text())
    assert render2["private_previous_projection_dir"] == str(tmp_path / "private" / "2026-09-29")
    for day in ("2026-09-29", "2026-09-30"):
        daily = json.loads((tmp_path / "runs" / day / "reports-publisher-config.json").read_text())
        assert str(tmp_path / "private") not in json.dumps(daily)
    # The repository gets the markdown unit (top N, 4-decimal scores), never the private HTML view.
    unit = world.remote_file(f"{mp._unit(mp.D1)}/kr-stocks.md")
    assert "PRIV001" in unit and "0.1235" in unit and "0.123456" not in unit
    clone = world.clone_for_checks()
    assert not list(clone.rglob("private-manifest.json")) and not list(clone.rglob("*.html"))
    worktree = "\n".join(f.read_text(encoding="utf-8") for f in clone.rglob("*.md"))
    assert "PRIVATE_DO_NOT_PUBLISH" not in worktree and "비공개 — 게시 금지" not in worktree
    # publish_stage source never references the private root
    import inspect
    assert "private_projection" not in inspect.getsource(daily_coordinator.publish_stage)
