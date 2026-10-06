"""Market-sector (R4) fixtures for the serving tests.

A stdlib-only fake bundle and a synthetic release.

``fake_ms_bundle`` writes a structurally valid ``bundle.json`` over junk bytes (release build and
provisioning only check hashes and structure).  ``build_release_world`` builds a frozen synthetic
release (the real ``src/modeler`` tree plus the synthetic ranking adapters) that also pins a *real*
synthetic market-sector bundle, with ops config pointing at the synthetic KR/US data roots, for
report date 2026-10-07.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import test_daily_coordinator_e2e as e2e
from r3_world import ADAPTER_R3, _calendar, write_native

if str(e2e.PROJECT) not in sys.path:  # ``tests.scores.ms_world`` lives in the project package
    sys.path.append(str(e2e.PROJECT))

from modeler.scores.market_sector.bundle import (
    CALENDAR_COLUMNS,
    CALENDAR_SCHEMA,
    SCHEMA,
    sessions_sha256,
)
from modeler.serving import release_build as rb
from modeler.serving.daily_inputs import MS_BUNDLE_DIR, MS_BUNDLE_PATH, MS_CODE_PATH, MS_ENTRYPOINT
from modeler.serving.orchestration import code_inventory_sha256
from modeler.serving.runtime_contract import PACKAGES, probe_code

D = date(2026, 10, 7)
SELECT_AT = "2026-10-07T09:30:00+09:00"
RUN_AT = "2026-10-07T10:00:00+09:00"
MODEL_IDS = (("KR", "kr_daily_h20_v1"), ("US", "us_exploratory_20260929_r1_lightgbm"),
             ("US", "us_exploratory_20260929_r1_ridge"))
SRC = Path(__file__).resolve().parents[3] / "src"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runtime_manifest(target: Path, extra: tuple[str, ...] = ()) -> Path:
    """A runtime manifest of the running interpreter (model packages, plus ``extra`` if given)."""
    probe = subprocess.run([sys.executable, "-c", probe_code((*PACKAGES, *extra))],
                           capture_output=True, text=True, check=True)
    body = {"schema_version": "daily-briefing-runtime.v1", **json.loads(probe.stdout)}
    target.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _fake_calendar(directory: Path, market: str, files: dict[str, str]) -> dict:
    """A small valid calculation-calendar file (three sessions) and its bundle.json entry."""
    sessions = ["2026-09-28", "2026-09-29", "2026-09-30"]
    rows = [[d, f"{d}T00:00:00+00:00", f"{d}T06:30:00+00:00"] for d in sessions]
    sha = sessions_sha256(sessions)
    body = {"schema_version": CALENDAR_SCHEMA, "market": market, "calendar_id": "FAKE",
            "calendar_basis": "fake", "range_end": "2027-12-31", "first_session": sessions[0],
            "last_session": sessions[-1], "n_sessions": len(sessions), "sessions_sha256": sha,
            "columns": CALENDAR_COLUMNS, "sessions": rows}
    rel = f"{market.lower()}/calendar.json"
    (directory / rel).write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    files[rel] = _sha((directory / rel).read_bytes())
    return {"calendar_id": "FAKE", "basis": "fake", "file": rel, "range_end": "2027-12-31",
            "file_n_sessions": len(sessions), "file_sessions_sha256": sha,
            "first_session": sessions[0], "last_session": sessions[-1], "n_sessions": len(sessions),
            "sessions_sha256": sha}


def fake_ms_bundle(directory: Path) -> Path:
    """A bundle.json over junk files: enough for release build / provisioning, not for scoring."""
    directory.mkdir(parents=True)
    files: dict[str, str] = {}
    markets: dict[str, dict] = {}
    for market in ("US", "KR"):
        m = market.lower()
        rel = {"run_manifest": f"{m}/manifest.json", "oof": f"{m}/oof_predictions.parquet",
               "latest_scores": f"{m}/latest_scores.json"}
        models = {name: f"{m}/models/live/{name}.joblib"
                  for name in ("p_opp_ridge", "p_stab_logit", "b_stab_logit_rvol")}
        for path in (*rel.values(), *models.values()):
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"junk {market} {path}".encode())
            files[path] = _sha(target.read_bytes())
        calendar = _fake_calendar(directory, market, files)
        markets[market] = {"run_id": f"fake_{m}", **rel, "models": models, "calendar": calendar}
    manifest = {"schema_version": SCHEMA, "files": files, "markets": markets}
    (directory / "bundle.json").write_text(json.dumps(manifest, sort_keys=True))
    return directory


def build_release_world(tmp_path: Path, monkeypatch, *, kr_snapshot_done: str | None = None):
    """(release, config_path, world) for D=2026-10-07 with a real synthetic market-sector bundle."""
    from tests.scores import ms_world as W

    monkeypatch.setattr(e2e, "ADAPTER", ADAPTER_R3)
    world = W.build_world(tmp_path / "data")
    bundle = W.make_bundle(world, tmp_path / "ms-bundle")
    release = tmp_path / "frozen-release"
    for path in (SRC / "modeler").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        target = release / "src" / path.relative_to(SRC)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    adapter = release / "src" / "modeler" / "serving" / "adapters.py"
    adapter.write_text(ADAPTER_R3)
    shutil.copy2(e2e.PROJECT / "uv.lock", release / "uv.lock")
    runtime_manifest(release / "runtime.json")
    cards = e2e._write(release / "model-cards.json", {
        model_id: {"title": model_id, "summary": "Synthetic fixture model card."}
        for _, model_id in MODEL_IDS})
    files = tuple((p, e2e._sha(p)) for p in (release / "src").rglob("*.py"))
    inventory = [{"path": str(p.relative_to(release)), "sha256": digest} for p, digest in files]
    code_sha = code_inventory_sha256(files)
    jobs = []
    for market, model_id in MODEL_IDS:
        bundle_json = e2e._write(release / "bundles" / (model_id + ".json"),
                                 {"synthetic_fixture": True, "model_id": model_id})
        jobs.append({"market": market, "model_id": model_id, "model_version": "1",
                     "entrypoint": "modeler.serving.adapters:infer_kr_daily" if market == "KR"
                     else "modeler.serving.adapters:infer_us_model",
                     "bundle_path": str(bundle_json.relative_to(release)),
                     "bundle_sha256": e2e._sha(bundle_json),
                     "code_path": str(adapter.relative_to(release)),
                     "code_path_sha256": e2e._sha(adapter),
                     "code_files": inventory, "code_sha256": code_sha})
    ms_sha = rb.copy_ms_bundle(bundle, release / MS_BUNDLE_DIR)
    ms_code = release / MS_CODE_PATH
    release_manifest = e2e._write(release / "release.json", {
        "schema_version": "daily-briefing-release.v1", "frozen": True, "synthetic_fixture": True,
        "release_root": str(release), "jobs": jobs,
        "market_sector": {"entrypoint": MS_ENTRYPOINT, "bundle_path": MS_BUNDLE_PATH,
                          "bundle_sha256": ms_sha, "code_path": MS_CODE_PATH,
                          "code_path_sha256": e2e._sha(ms_code), "code_sha256": code_sha}})
    prepared = tmp_path / "prepared"
    write_native(prepared, "KR", "2026-10-06", "2026-10-07T09:20:00+09:00")
    write_native(prepared, "US", "2026-10-06", "2026-10-07T09:20:00+09:00")
    kr_calendar = e2e._write(tmp_path / "kr-calendar.json",
                             _calendar("KR", date(2026, 9, 14), date(2026, 10, 9)))
    us_calendar = e2e._write(tmp_path / "us-calendar.json",
                             _calendar("US", date(2026, 9, 14), date(2026, 10, 9)))
    expected = e2e._write(tmp_path / "us-expected.json", {
        "schema_version": "us-expected-source.v1", "reviewed_status": "synthetic_fixture",
        "source_reference": "synthetic-fixture-only",
        "expected_session_by_report_date": {D.isoformat(): "2026-10-06"},
        "market_lag_limit_sessions": 1})
    ops = {
        "schema_version": "daily-briefing-ops.v1",
        "prepared_root": str(prepared), "selection_root": str(prepared / "selections"),
        "run_root": str(tmp_path / "runs"), "release_manifest": str(release_manifest),
        "python": sys.executable, "python_sha256": e2e._sha(Path(sys.executable)),
        "runtime_lock": str(release / "uv.lock"),
        "runtime_lock_sha256": e2e._sha(release / "uv.lock"),
        "runtime_manifest": str(release / "runtime.json"),
        "runtime_manifest_sha256": e2e._sha(release / "runtime.json"),
        "model_cards_path": str(cards), "model_cards_sha256": e2e._sha(cards),
        "kr_calendar": str(kr_calendar), "us_calendar": str(us_calendar),
        "us_expected_source": str(expected), "opening_artifact": None,
        "market_sector_kr_root": str(world.kr_root), "market_sector_us_root": str(world.us_root),
        "publisher_enabled": False, "external_verification_enabled": False,
        "reports_publisher": None, "reports_publisher_sha256": None, "reports_checkout": None,
        "reports_remote_url": None, "reports_repository": None, "reports_audience": None,
        "reports_branch": None, "reports_top_n": 100}
    config = e2e._write(tmp_path / "ops.json", ops)
    return release, config, world
