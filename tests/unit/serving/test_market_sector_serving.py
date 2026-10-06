"""R4: the market-sector section in the daily serving flow (select pin, scoring stage, release)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import ms_fixtures as F
import pytest
import test_daily_coordinator_e2e as e2e

from modeler.reporting import markdown
from modeler.serving import daily_coordinator as dc
from modeler.serving import release_build as rb
from modeler.serving.daily_inputs import MS_BUNDLE_DIR, MS_BUNDLE_PATH, _release_market_sector

D = F.D
KST = dc.SEOUL
REPORTING_TESTS = Path(__file__).resolve().parents[1] / "reporting"  # reports_world, md_fixtures
if str(REPORTING_TESTS) not in sys.path:
    sys.path.append(str(REPORTING_TESTS))


def _cli(release: Path, config: Path, stage: str, now: str, *, ok: bool = True) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    result = subprocess.run(
        [sys.executable, "-m", "modeler.serving.daily_coordinator", stage, "--config", str(config),
         "--report-date", D.isoformat(), "--fixture-now", now],
        cwd=release, env=env, text=True, capture_output=True, timeout=240)
    if ok:
        assert result.returncode == 0, (stage, result.stdout, result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {}


def _wrapper(release: Path, config: Path, stage: str, now: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    return subprocess.run(
        [sys.executable, "-m", "modeler.serving.daily_wrapper", stage, "--config", str(config),
         "--report-date", D.isoformat(), "--fixture-now", now],
        cwd=release, env=env, text=True, capture_output=True, timeout=240)


def _selection_state(tmp_path: Path) -> dict:
    path = tmp_path / "prepared/selections" / D.isoformat() / "selection-state.json"
    return json.loads(path.read_text())


def _ms_doc(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "runs" / D.isoformat() / f"market-sector-{D}.json").read_text())


def _context(tmp_path: Path):
    run = tmp_path / "runs" / D.isoformat()
    ms = run / f"market-sector-{D}.json"
    return markdown.build_context(
        tmp_path / "no-reports-checkout", (run / f"report-{D}.json").read_bytes(),
        ms.read_bytes() if ms.is_file() else None, "frozen-release", None, 100,
        lambda message: None)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    return F.build_release_world(tmp_path, monkeypatch)


def test_select_pins_the_market_sector_inputs_beside_the_three_jobs(tmp_path, world):
    release, config, data = world
    assert _cli(release, config, "select", F.SELECT_AT)["status"] == "selected"
    state = _selection_state(tmp_path)
    assert len(state["jobs"]) == 3  # the section is not a fourth ranking job
    block = state["market_sector"]
    assert block["status"] == "selected" and set(block["markets"]) == {"KR", "US"}
    assert block["markets"]["KR"] == {"status": "selected", "limit_session": "2026-10-06"}
    selection_path = Path(block["selection"])
    assert selection_path.name == "ms-selection.json"
    assert selection_path.parent.name == D.isoformat()
    selection = json.loads(selection_path.read_text())
    assert selection["selected_at"] == F.SELECT_AT and selection["selection_mode"] == "scheduled"
    assert selection["input_cutoff"] == "2026-10-07T09:30:00+09:00"
    assert selection["kr"]["export_finished_at"] == "2026-10-07T04:29:15+09:00"
    assert selection["bundle"]["sha256"] == block["bundle_sha256"]
    # a second select returns the same immutable selection; a changed ms-selection is refused
    assert _selection_state(tmp_path) == state
    assert _cli(release, config, "select", F.SELECT_AT)["status"] == "selected"
    selection_path.write_text(selection_path.read_text() + " ")
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    again = subprocess.run(
        [sys.executable, "-m", "modeler.serving.daily_coordinator", "select",
         "--config", str(config), "--report-date", D.isoformat(), "--fixture-now", F.SELECT_AT],
        cwd=release, env=env, text=True, capture_output=True, timeout=120)
    assert again.returncode == 1 and "ValueError" in again.stderr


def test_a_kr_export_finished_after_0930_is_not_selected(tmp_path, world):
    from tests.scores import ms_world as W

    release, config, data = world
    W.write_kr_snapshot(data.root, "2026-10-08", "2026-10-07T09:40:00+0900", data.kr_sessions,
                        seed=9)
    assert _cli(release, config, "select", F.SELECT_AT)["status"] == "selected"
    selection_path = Path(_selection_state(tmp_path)["market_sector"]["selection"])
    selection = json.loads(selection_path.read_text())
    assert selection["kr"]["snapshot_date"] == W.KR_SNAP
    assert selection["kr"]["skipped"][0]["reason"] == "completed_after_cutoff"


def test_run_scores_the_section_and_the_unit_renders_it(tmp_path, world):
    release, config, data = world
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    done = _wrapper(release, config, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    assert stages["market_sector"] == "market_sector_ready" and stages["inference"] == "inferred"
    doc = _ms_doc(tmp_path)
    assert doc["report_date"] == D.isoformat() and len(doc["assets"]) == 14
    assert doc["status"] == "ok"
    run = tmp_path / "runs" / D.isoformat()
    state = json.loads((run / "coordinator-market-sector.json").read_text())
    assert state["status"] == "market_sector_ready"
    assert state["markets"] == {"KR": "ok", "US": "ok"}
    assert not list(run.glob("*.superseded-*"))
    ctx = _context(tmp_path)
    assert ctx["ms"]["status"] == "ok" and len(ctx["ms"]["assets"]) == 14
    files = markdown.render_unit(ctx)
    assert "지수 종가" in files["market-sector.md"] and "SPY" in files["market-sector.md"]
    assert "5432" not in files["market-sector.md"]


def test_rerun_keeps_the_same_bytes_and_sets_the_old_document_aside(tmp_path, world):
    release, config, data = world
    _cli(release, config, "select", F.SELECT_AT)
    for _ in range(2):
        assert _cli(release, config, "market-sector", F.RUN_AT)["status"] == "market_sector_ready"
    run = tmp_path / "runs" / D.isoformat()
    doc = run / f"market-sector-{D}.json"
    (old,) = run.glob(f"market-sector-{D}.superseded-*.json")
    assert old.read_bytes() == doc.read_bytes()  # the scoring is deterministic: same bytes


def test_input_changed_after_select_fails_the_section_and_the_units_still_render(tmp_path, world):
    release, config, data = world
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    macro = data.us_table_file("macro_series", "2026-10-05")  # both markets read it
    macro.write_bytes(macro.read_bytes() + b"0")
    done = _wrapper(release, config, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr  # the ranking models still ran
    stages = json.loads(done.stdout)["stages"]
    assert stages["market_sector"] == "market_sector_failed" and stages["inference"] == "inferred"
    assert stages["render"] == "rendered"
    doc = _ms_doc(tmp_path)
    assert doc["status"] == "failed" and doc["assets"] == []
    assert doc["failure"]["stage"] == "score" and doc["failure"]["reason"] == "all_markets_failed"
    assert doc["failure"]["error_class"] == "MarketSectorExitNonzero"
    assert {m: f["reason"] for m, f in doc["failure"]["markets"].items()} == {
        "US": "input_changed", "KR": "input_changed"}
    ctx = _context(tmp_path)
    assert ctx["ms"]["status"] == "failed"
    text = "\n".join(ctx["ms"]["reason"])
    assert "채점" in text and "input_changed" in text
    assert ctx["kr"]["status"] != "failed" or ctx["us"]["status"] != "failed"  # rankings unaffected


def test_a_timeout_makes_a_failed_document_with_the_timeout_flag(tmp_path, world, monkeypatch):
    release, config_path, data = world
    _cli(release, config_path, "select", F.SELECT_AT)
    config = dc._config(config_path)

    def hang(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(dc, "run_group", hang)
    state = dc.market_sector_stage(config, D, datetime.fromisoformat(F.RUN_AT))
    assert state["status"] == "market_sector_failed" and state["timed_out"] is True
    doc = _ms_doc(tmp_path)
    assert doc["failure"] == {
        "stage": "score", "error_class": "MarketSectorTimeout", "reason": "timeout",
        "exit_code": 124, "timed_out": True, "markets": {}}
    run = tmp_path / "runs" / D.isoformat()
    assert not (run / f".market-sector-{D}.new").exists()
    reasons = markdown.ms_failure_reasons(doc["failure"])
    assert any("timeout" in r for r in reasons)


def test_a_missing_selection_before_1000_is_a_failed_document_not_a_crash(tmp_path, world):
    release, config_path, data = world
    config = dc._config(config_path)
    state = dc.market_sector_stage(config, D, datetime.fromisoformat("2026-10-07T09:45:00+09:00"))
    assert state["status"] == "market_sector_failed" and state["stage"] == "select"
    assert _ms_doc(tmp_path)["failure"]["reason"] == "selection_unavailable"


def test_unavailable_kr_inputs_leave_a_partial_section(tmp_path, world):
    release, config, data = world
    shutil.rmtree(data.kr_snapshot_dir().parent)
    assert _cli(release, config, "select", F.SELECT_AT)["status"] == "selected"
    block = _selection_state(tmp_path)["market_sector"]
    assert block["status"] == "partial"
    assert block["markets"]["KR"]["status"] == "unavailable"
    assert _cli(release, config, "market-sector", F.RUN_AT)["status"] == "market_sector_partial"
    doc = _ms_doc(tmp_path)
    assert doc["status"] == "partial" and {a["market"] for a in doc["assets"]} == {"US"}
    assert doc["failures"]["KR"]["reason"] == "input_unavailable"


def test_without_the_section_nothing_changes(tmp_path, world):
    release, config_path, data = world
    body = json.loads(config_path.read_text())
    body["market_sector_kr_root"] = body["market_sector_us_root"] = None
    config = e2e._write(tmp_path / "ops-off.json", body)
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    assert "market_sector" not in _selection_state(tmp_path)
    done = _wrapper(release, config, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    assert "market_sector" not in json.loads(done.stdout)["stages"]
    run = tmp_path / "runs" / D.isoformat()
    assert not list(run.glob("market-sector-*"))
    assert json.loads((run / "coordinator-market-sector.json").read_text())["status"] == "disabled"
    assert _context(tmp_path)["ms"]["status"] == "failed"  # "입력 없음", as before


def test_config_needs_both_roots_and_a_release_with_the_bundle(tmp_path, world):
    release, config_path, data = world
    base = json.loads(config_path.read_text())
    with pytest.raises(ValueError, match="set together"):
        dc._config(e2e._write(tmp_path / "a.json", {**base, "market_sector_us_root": None}))
    with pytest.raises(ValueError, match="does not identify"):
        missing = {**base, "market_sector_kr_root": str(tmp_path / "x")}
        dc._config(e2e._write(tmp_path / "b.json", missing))
    manifest = json.loads(Path(base["release_manifest"]).read_text())
    del manifest["market_sector"]
    bare = e2e._write(release / "release.json", manifest)  # a release without the section
    with pytest.raises(ValueError, match="has no bundle"):
        dc._config(e2e._write(tmp_path / "c.json", {**base, "release_manifest": str(bare)}))


def test_release_check_rejects_a_changed_bundle_or_code(tmp_path, world):
    release, config_path, data = world
    manifest = Path(json.loads(config_path.read_text())["release_manifest"])
    pinned = _release_market_sector(manifest, jobs={})
    assert pinned["bundle_path"].endswith(MS_BUNDLE_PATH)
    model = release / MS_BUNDLE_DIR / "us/models/live/p_opp_ridge.joblib"
    model.chmod(0o644)
    model.write_bytes(model.read_bytes() + b"0")
    with pytest.raises(ValueError, match="바뀌었습니다"):
        _release_market_sector(manifest, jobs={})


# --------------------------------------------------------------------------- release_build
def test_release_build_pins_the_bundle_and_requires_the_code(tmp_path):
    import test_release_build as trb

    ms_bundle = F.fake_ms_bundle(tmp_path / "msb")
    modeler, collector = tmp_path / "modeler", tmp_path / "collector"
    listing = {}

    def add(repo, root, rel, data):
        listing[f"{repo}/{rel}"] = trb._write(root / rel, data)

    add("modeler", modeler, "src/modeler/__init__.py", b"")
    add("modeler", modeler, "src/modeler/serving/__init__.py", b"")
    add("modeler", modeler, "src/modeler/serving/adapters.py", b"# adapters\n")
    add("collector", collector, "src/collector/__init__.py", b"")
    add("collector", collector, trb.CSV, b"date\n2026-01-01\n")
    for rel, data in (("deploy/prod/model-cards.json", b"[]"), ("deploy/prod/runtime.json", b"{}"),
                      ("uv.lock", b"lock")):
        add("modeler", modeler, rel, data)
    manifest = tmp_path / "SHA256SUMS"

    def save():
        manifest.write_text("".join(f"{d}  {n}\n" for n, d in sorted(listing.items())))

    save()
    kwargs = dict(
        modeler_root=modeler, collector_root=collector,
        kr_bundle=trb._bundle(tmp_path / "b", "kr", "1.0.0"),
        us_lightgbm_bundle=trb._bundle(tmp_path / "b", "lightgbm", "1"),
        us_ridge_bundle=trb._bundle(tmp_path / "b", "ridge", "1"),
        model_cards=modeler / "deploy/prod/model-cards.json",
        runtime_manifest=modeler / "deploy/prod/runtime.json", uv_lock=modeler / "uv.lock",
        source_manifest=manifest)
    (tmp_path / "out").mkdir()
    # the scoring code is part of the copied source: without score_daily.py the build stops
    with pytest.raises(rb.BuildError, match="score_daily.py is absent"):
        rb.build_release(**kwargs, ms_bundle=ms_bundle, output=tmp_path / "out" / "r0")
    assert not (tmp_path / "out" / "r0").exists()
    add("modeler", modeler, "src/modeler/scores/__init__.py", b"")
    add("modeler", modeler, "src/modeler/scores/market_sector/__init__.py", b"")
    add("modeler", modeler, "src/modeler/scores/market_sector/score_daily.py", b"# scoring\n")
    save()
    summary = rb.build_release(**kwargs, ms_bundle=ms_bundle, output=tmp_path / "out" / "r1")
    release = tmp_path / "out" / "r1"
    body = json.loads((release / "release.json").read_text())
    block = body["market_sector"]
    assert block["bundle_path"] == MS_BUNDLE_PATH and block["code_sha256"] == summary["code_sha256"]
    assert summary["market_sector"]["bundle_sha256"] == block["bundle_sha256"]
    assert (release / MS_BUNDLE_DIR / "us/models/live/p_opp_ridge.joblib").is_file()
    assert (release / "src/modeler/scores/market_sector/score_daily.py").is_file()
    assert _release_market_sector(release / "release.json") is not None
    # a bundle file that no longer matches bundle.json is refused before anything is copied
    broken = F.fake_ms_bundle(tmp_path / "msb2")
    (broken / "kr/manifest.json").write_bytes(b"tampered")
    with pytest.raises(Exception, match="바뀌었습니다"):
        rb.build_release(**kwargs, ms_bundle=broken, output=tmp_path / "out" / "r2")
    assert not (tmp_path / "out" / "r2").exists()
    # without --ms-bundle the release has no section (as before)
    rb.build_release(**kwargs, output=tmp_path / "out" / "r3")
    assert "market_sector" not in json.loads((tmp_path / "out" / "r3" / "release.json").read_text())


def test_the_publisher_renders_the_unit_from_the_coordinators_document(tmp_path, world):
    """publish_reports reads runs/D/market-sector-D.json as the coordinator left it."""
    from reports_world import CARDS, World, pr

    release, config, data = world
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    assert _wrapper(release, config, "run", F.RUN_AT).returncode == 0
    run = tmp_path / "runs" / D.isoformat()
    render = json.loads((run / "coordinator-render.json").read_text())
    reports = World(tmp_path / "reports-world")
    local = pr.local_step({
        "audience": "owner_only", "checkout_dir": str(reports.checkout), "remote_name": "origin",
        "expected_remote_url": str(reports.bare), "branch": "main", "release": "frozen-release",
        "run_dir": str(run), "report_date": D.isoformat(), "report_sha256": render["report_sha256"],
        "invocation_id": render["invocation_id"], "model_cards_path": str(CARDS),
        "generated_at": f"{D}T10:03:12+09:00"}, allow_synthetic=True)
    assert local["status"] == "local_ready", local
    unit = Path(local["markdown_dir"]) / "unit"
    text = (unit / "market-sector.md").read_text()
    assert "status: ok" in text and "SPY (S&P 500)" in text and "목표별 판정" in text
    manifest = json.loads((Path(local["markdown_dir"]) / "manifest.json").read_text())
    ms_path = run / f"market-sector-{D}.json"
    assert manifest["market_sector_path"] == str(ms_path)
    assert manifest["market_sector_sha256"] == hashlib.sha256(ms_path.read_bytes()).hexdigest()
