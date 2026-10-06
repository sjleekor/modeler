"""provision_serving.py against a copy of the real source tree and fake bundles."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

import collector
from modeler.serving import release_build as rb
from modeler.serving import us_expected
from modeler.serving.daily_coordinator import _config
from modeler.serving.daily_inputs import _release_jobs
from modeler.serving.runtime_contract import PROBE

MODELER = Path(__file__).resolve().parents[3]
COLLECTOR = Path(collector.__file__).resolve().parents[2]
SCRIPT = MODELER / "deploy" / "prod" / "provision_serving.py"
spec = importlib.util.spec_from_file_location("provision_serving", SCRIPT)
ps = importlib.util.module_from_spec(spec)
sys.modules["provision_serving"] = ps
spec.loader.exec_module(ps)

IGNORE = shutil.ignore_patterns("__pycache__", "._*", "*.pyc")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle(root: Path, kind: str, version: str) -> Path:
    directory = root / kind
    directory.mkdir(parents=True)

    def put(name: str, data: bytes) -> str:
        (directory / name).write_bytes(data)
        return _sha(data)

    if kind == "kr":
        manifest = {"market": "KR", "model_version": version,
                    "files": {"golden.json": put("golden.json", b"{}"),
                              "model.joblib": put("model.joblib", b"kr")}}
    else:
        manifest = {"variant": kind, "model_version": version,
                    "model_sha256": put("model.joblib", kind.encode()),
                    "golden_features_sha256": put("golden_features.parquet", b"gf"),
                    "golden_predictions_sha256": put("golden_predictions.parquet", kind.encode() + b"gp")}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def _run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ps.main(args)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prov")
    modeler, collector_root = tmp / "modeler", tmp / "collector"
    shutil.copytree(MODELER / "src", modeler / "src", ignore=IGNORE)
    shutil.copytree(COLLECTOR / "src", collector_root / "src", ignore=IGNORE)
    (modeler / "deploy/prod").mkdir(parents=True)
    (modeler / "deploy/reports").mkdir(parents=True)
    shutil.copy(MODELER / "deploy/prod/model-cards.json", modeler / "deploy/prod/model-cards.json")
    for name in ps.PUBLISHER_FILES:
        shutil.copy(MODELER / "deploy/reports" / name, modeler / "deploy/reports" / name)
    lock = MODELER / "uv.lock"
    (modeler / "uv.lock").write_bytes(lock.read_bytes() if lock.is_file() else b"lock")
    venv = tmp / "venv_src"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    probe = json.loads(subprocess.run([str(python), "-c", PROBE], capture_output=True, text=True,
                                      check=True).stdout)
    (modeler / "deploy/prod/runtime.json").write_text(
        json.dumps({"schema_version": "daily-briefing-runtime.v1", **probe}))
    listing = {}
    for repo, root in (("modeler", modeler), ("collector", collector_root)):
        for path in (root / "src").rglob("*.py"):
            listing[f"{repo}/{path.relative_to(root).as_posix()}"] = _sha(path.read_bytes())
    data = rb.DATA_FILES[0]
    listing[f"collector/{data}"] = _sha((collector_root / data).read_bytes())
    for rel in ("deploy/prod/model-cards.json", "deploy/prod/runtime.json", "uv.lock",
                *(f"deploy/reports/{n}" for n in ps.PUBLISHER_FILES)):
        listing[f"modeler/{rel}"] = _sha((modeler / rel).read_bytes())
    manifest = tmp / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(f"{d}  {n}\n" for n, d in sorted(listing.items())))
    table = us_expected.build_table(date(2026, 10, 1), date(2026, 10, 30), reviewed_status="confirmed",
                                    review_note="unit test")
    expected = us_expected.write_table(table, tmp / "us-expected.confirmed.json")
    evidence = tmp / "parity.json"
    evidence.write_text('{"status": "score_equivalent"}')
    reports_checkout = tmp / "reports-checkout"
    for name in ("raw", "derived"):
        (tmp / "lake" / name).mkdir(parents=True)
    base = ["--modeler-root", str(modeler), "--collector-root", str(collector_root),
            "--source-manifest", str(manifest),
            "--kr-bundle", str(_bundle(tmp / "b", "kr", "1.0.0")),
            "--us-lightgbm-bundle", str(_bundle(tmp / "b", "lightgbm", "1")),
            "--us-ridge-bundle", str(_bundle(tmp / "b", "ridge", "1")),
            "--venv-source", str(venv), "--us-expected-source", str(expected),
            "--us-expected-sha256", _sha(expected.read_bytes()),
            "--parity-evidence", str(evidence), "--parity-evidence-sha256", _sha(evidence.read_bytes()),
            "--reports-checkout", str(reports_checkout),
            "--us-lake", str(tmp / "lake"), "--runtime-manifest", str(modeler / "deploy/prod/runtime.json")]
    root = tmp / "serving"
    code, out, err = _run(["--serving-root", str(root), "--release-id", "r1", *base])
    assert code == 0, err
    yield {"tmp": tmp, "root": root, "base": base, "summary": json.loads(out), "expected": expected,
           "evidence": evidence, "lake": tmp / "lake"}
    for current, _, _ in os.walk(root):
        os.chmod(current, 0o700)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_layout_modes_and_links(world):
    root = world["root"]
    expected_dirs = {"": 0o750, "config": 0o750, "publisher": 0o750, "releases": 0o750, "runs": 0o700,
                     "locks": 0o700, "private-projection": 0o700, "logs": 0o750,
                     "prepared": 0o750, "prepared/kr": 0o750, "prepared/selections": 0o750,
                     "stock_data/us/output": 0o750}
    for rel, mode in expected_dirs.items():
        assert _mode(root / rel) == mode, rel
    assert (root / "venv/bin/python").is_file()
    assert not any((root / "prepared/kr").iterdir())
    for name in ("raw", "derived"):
        link = root / "stock_data/us" / name
        assert link.is_symlink() and link.resolve() == (world["lake"] / name).resolve()
    native = root / "stock_data/us/output/us_scoring_daily_v1/prepared"
    assert native.is_dir() and not native.is_symlink()
    assert (root / "prepared/us").is_symlink() and (root / "prepared/us").resolve() == native.resolve()
    assert not (root / "stock_data/us/output").is_symlink()
    assert _mode(root / "config/ops.json") == 0o640 and _mode(root / "config/pins.json") == 0o440
    assert _mode(root / "publisher/publish_reports.py") == 0o440
    assert _mode(root / "publisher/validate_reports.py") == 0o440
    assert not (root / "projection").exists()  # the public projection is gone
    release = root / "releases/r1"
    assert _mode(release) == 0o550 and _mode(release / "release.json") == 0o440


def test_ops_json_fields(world):
    root = world["root"]
    ops = json.loads((root / "config/ops.json").read_text())
    release = root / "releases/r1"
    assert ops["schema_version"] == "daily-briefing-ops.v1"
    assert ops["publisher_enabled"] is False and ops["external_verification_enabled"] is False
    assert all(ops[key] is None for key in ("opening_artifact", "opening_snapshot_root", "opening_output_root",
                                            "opening_max_age_seconds"))
    assert ops["private_projection_root"] == str(root / "private-projection")
    assert ops["prepared_root"] == str(root / "prepared") and ops["selection_root"] == str(root / "prepared/selections")
    # Pages keys are gone; the stock_reports publisher keys are set but the publisher stays off.
    for gone in ("base_path", "actions_repository", "actions_workflow", "public_manifest_url",
                 "projection_root", "previous_projection_dir", "publisher_script",
                 "publisher_script_sha256", "publisher_config", "site_checkout"):
        assert gone not in ops, gone
    assert ops["reports_repository"] == "sjleekor/stock_reports"
    assert ops["reports_audience"] == "owner_only"
    assert ops["reports_branch"] == "main" and ops["reports_top_n"] == 100
    assert ops["reports_remote_url"] == "git@github.com:sjleekor/stock_reports.git"
    assert ops["reports_checkout"] == str(world["tmp"] / "reports-checkout")
    assert ops["reports_publisher"] == str(root / "publisher/publish_reports.py")
    publisher = root / "publisher/publish_reports.py"
    assert ops["reports_publisher_sha256"] == _sha(publisher.read_bytes())
    assert ops["runtime_lock"] == str(release / "uv.lock") and ops["model_cards_path"] == str(release / "model-cards.json")
    assert ops["us_expected_source"] == str(root / "config" / world["expected"].name)
    assert ops["python"] == str(root / "venv/bin/python")
    # security names for the kind column: the US lake link always, KR only with --ms-bundle
    assert ops["security_names_us_root"] == str(root / "stock_data" / "us")
    assert ops["security_names_kr_root"] is None
    assert json.loads((release / "release.json").read_text())["synthetic_fixture"] is False
    pins = json.loads((root / "config/pins.json").read_text())
    assert pins["us_expected_source"]["sha256"] == _sha(world["expected"].read_bytes())
    assert pins["parity_evidence"]["sha256"] == _sha(world["evidence"].read_bytes())
    assert set(pins["publisher_scripts"]) == set(ps.PUBLISHER_FILES)


def test_coordinator_accepts_the_config(world):
    ops = world["root"] / "config/ops.json"
    config = _config(ops)
    jobs, fixture = _release_jobs(Path(config["release_manifest"]))
    assert len(jobs) == 3 and fixture is False
    summary = world["summary"]
    assert summary["validation"]["config_ok"] is True and summary["select_executed"] is False
    assert summary["validation"]["calendars"]["kr_calendar"]["coverage"] == ["2026-01-01", "2026-12-31"]
    assert summary["validation"]["calendars"]["us_calendar"]["coverage"] == ["2026-01-01", "2027-12-31"]


def test_rerun_is_refused_and_changes_nothing(world):
    ops = world["root"] / "config/ops.json"
    before = ops.read_bytes()
    code, _, err = _run(["--serving-root", str(world["root"]), "--release-id", "r1", *world["base"]])
    assert code == 1 and "already exists" in err
    code, _, err = _run(["--serving-root", str(world["root"]), "--release-id", "r2", *world["base"]])
    assert code == 1 and "ops.json already exists" in err
    assert ops.read_bytes() == before and not (world["root"] / "releases/r2").exists()


def test_dry_run_writes_nothing(world):
    root = world["tmp"] / "dry"
    code, out, err = _run(["--serving-root", str(root), "--release-id", "r1", "--dry-run", *world["base"]])
    assert code == 0, err
    body = json.loads(out)
    assert body["status"] == "dry_run" and any("ops.json" in item for item in body["would_create"])
    assert not root.exists()


def test_wrong_pin_stops_before_writing(world):
    root = world["tmp"] / "bad_pin"
    args = list(world["base"])
    args[args.index("--us-expected-sha256") + 1] = "0" * 64
    code, _, err = _run(["--serving-root", str(root), "--release-id", "r1", *args])
    assert code == 1 and "US expected source SHA-256 differs" in err and not root.exists()


def test_bad_release_id(world):
    code, _, err = _run(["--serving-root", str(world["tmp"] / "x"), "--release-id", "../x", "--dry-run", *world["base"]])
    assert code == 1 and "release id" in err


def test_release_contains_the_markdown_renderer_and_it_imports_without_serving_code(world):
    release = world["root"] / "releases/r1"
    assert (release / "src/modeler/reporting/markdown.py").is_file()
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import modeler.reporting.markdown as m; "
            "assert 'modeler.serving' not in sys.modules and 'pandas' not in sys.modules; "
            "print(m.FAMILY)")
    out = subprocess.run([sys.executable, "-c", code, str(release / "src")], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "daily-briefing"


def test_publisher_scripts_in_the_serving_root_are_the_reviewed_ones(world):
    pins = json.loads((world["root"] / "config/pins.json").read_text())
    for name in ps.PUBLISHER_FILES:
        placed = world["root"] / "publisher" / name
        assert _sha(placed.read_bytes()) == pins["publisher_scripts"][name]["sha256"]
        assert _sha(placed.read_bytes()) == _sha((MODELER / "deploy/reports" / name).read_bytes())


def test_reports_checkout_and_url_are_checked(world):
    base = list(world["base"])
    at = base.index("--reports-checkout")
    inside = world["tmp"] / "x1" / "reports-checkout"
    args = base[:at] + ["--reports-checkout", str(inside)] + base[at + 2:]
    root = str(world["tmp"] / "x1")
    code, _, err = _run(["--serving-root", root, "--release-id", "r1", "--dry-run", *args])
    assert code == 1 and "overlap the serving root" in err
    https = ["--reports-remote-url", "https://github.com/sjleekor/stock_reports.git"]
    root = str(world["tmp"] / "x2")
    code, _, err = _run(["--serving-root", root, "--release-id", "r1", "--dry-run", *base, *https])
    assert code == 1 and "SSH URL" in err


def _ms_args(world, tmp_path):
    """Provisioning arguments with the section on (the unchanged runtime manifest)."""
    import ms_fixtures

    ms_bundle = ms_fixtures.fake_ms_bundle(tmp_path / "ms-bundle")
    kr_root = tmp_path / "kr-data"
    (kr_root / "raw").mkdir(parents=True)
    return ["--ms-bundle", str(ms_bundle), "--ms-kr-root", str(kr_root)], kr_root


def test_config_checks_the_names_roots_but_not_that_they_exist(world, tmp_path):
    """An unreadable lake only drops the kind column; the config must not stop a run for it."""
    ops = json.loads((world["root"] / "config/ops.json").read_text())

    def load(**changes):
        path = tmp_path / "ops-names.json"
        path.write_text(json.dumps({**ops, **changes}))
        return _config(path)

    missing = "/does/not/exist"
    assert load(security_names_us_root=missing)["security_names_us_root"] == missing
    assert load(security_names_kr_root=None)["security_names_kr_root"] is None
    with pytest.raises(ValueError, match="absolute"):
        load(security_names_kr_root="relative/kr")


def test_provision_with_the_market_sector_section(world, tmp_path):
    """--ms-bundle pins the bundle in the release and turns the section on in ops.json."""
    extra, kr_root = _ms_args(world, tmp_path)
    root = tmp_path / "serving-ms"
    args = ["--serving-root", str(root), "--release-id", "r2", *world["base"], *extra]
    try:
        code, out, err = _run([*args, "--dry-run"])
        assert code == 0, err
        assert json.loads(out)["checks"]["market_sector"] is True
        code, out, err = _run(args)
        assert code == 0, err
        ops = json.loads((root / "config/ops.json").read_text())
        assert ops["market_sector_kr_root"] == str(kr_root)
        assert ops["market_sector_us_root"] == str(root / "stock_data" / "us")
        assert ops["security_names_us_root"] == str(root / "stock_data" / "us")
        assert ops["security_names_kr_root"] == str(kr_root)
        release = root / "releases/r2"
        block = json.loads((release / "release.json").read_text())["market_sector"]
        assert block["bundle_path"] == "bundles/market_sector/bundle.json"
        assert (release / "bundles/market_sector/us/models/live/p_opp_ridge.joblib").is_file()
        config = _config(root / "config/ops.json")
        assert config["market_sector_kr_root"] == str(kr_root)
        assert json.loads(out)["validation"]["config_ok"] is True
        # the calculation calendar is a bundle file: the serving venv needs no exchange_calendars
        assert "exchange_calendars" not in json.loads(out)["validation"]["packages"]
        assert (release / "bundles/market_sector/us/calendar.json").is_file()
        # the first (no --ms-bundle) provisioning keeps the section off
        first = json.loads((world["root"] / "config/ops.json").read_text())
        assert first["market_sector_kr_root"] is None and first["market_sector_us_root"] is None
    finally:
        for current, _, _ in os.walk(root):
            os.chmod(current, 0o700)


def test_the_section_refuses_a_bundle_without_a_calendar_file(world, tmp_path):
    """A bundle that lacks its calculation-calendar file never reaches a release."""
    extra, _ = _ms_args(world, tmp_path)
    bundle = Path(extra[1])
    manifest = json.loads((bundle / "bundle.json").read_text())
    del manifest["files"]["kr/calendar.json"]
    manifest["markets"]["KR"]["calendar"].pop("file")
    (bundle / "kr/calendar.json").unlink()
    (bundle / "bundle.json").write_text(json.dumps(manifest))
    root = tmp_path / "serving-ms-no-calendar"
    code, _, err = _run(["--serving-root", str(root), "--release-id", "r3",
                         *world["base"], *extra])
    assert code == 1 and "계산 달력 파일 항목이 없습니다" in err
    assert not (root / "releases/r3").exists()


def test_runtime_contract_does_not_require_exchange_calendars(tmp_path):
    """The serving venv has no exchange_calendars (2026-10-06): the section must not need it."""
    import ms_fixtures

    from modeler.serving.runtime_contract import PACKAGES, verify_runtime

    plain = ms_fixtures.runtime_manifest(tmp_path / "plain.json")
    python = Path(sys.executable)
    verified = verify_runtime(python, plain)
    assert set(verified["packages"]) == set(PACKAGES)
    assert "exchange_calendars" not in verified["packages"]
    # a manifest still has to pin every model dependency
    body = json.loads(plain.read_text())
    del body["packages"]["polars"]
    short = tmp_path / "short.json"
    short.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="does not pin"):
        verify_runtime(python, short)
    body = json.loads(plain.read_text())
    body["packages"]["numpy"] = "0.0.1"
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="differ"):
        verify_runtime(python, wrong)
