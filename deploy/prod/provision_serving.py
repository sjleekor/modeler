#!/usr/bin/env python3
"""Install the production serving root for the daily market briefing (run once).

Standard library only, so the system ``python3`` can run it.  It builds a fresh
``--serving-root`` from reviewed inputs, never touches the lake (it only links to
it) and never runs ``select``, ``run`` or ``monitor``.  Running it again for the
same release id or over an existing ``config/ops.json`` is refused: a release is
bound to its absolute path, so the only supported "redo" is a new release id in
a new serving root (or deleting the failed root by hand).

Layout written under ``--serving-root`` (modes in parentheses, owner-only on
purpose; Cronicle runs as the same user and nothing else needs access):

    venv/                        copy of the verified venv                  (as copied)
    releases/<id>/               frozen non-synthetic release, read-only    (0550 dirs / 0440 files)
    config/                      ops.json (0640), pins.json, E table, parity evidence,
                                 calendars/ (all 0440)                      (0750)
    publisher/                   publish_site.py, validate_public_site.py   (0750, files 0440)
    stock_data/us/raw|derived    read-only symlinks to the operational lake
    stock_data/us/output/        real directory, the only place prepare writes  (0750)
    prepared/kr/                 empty                                     (0750)
    prepared/us                  symlink -> stock_data/us/output/us_scoring_daily_v1/prepared
    prepared/selections/         D selections (must live inside prepared/)  (0750)
    runs/ locks/                 coordinator state                          (0700)
    projection/                  public projection, re-creatable            (0750)
    private-projection/          private views, never published             (0700)
    logs/                        free for wrappers                          (0750)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

OPS_SCHEMA = "daily-briefing-ops.v1"
PINS_SCHEMA = "daily-briefing-serving-pins.v1"
RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
SCORING_VERSION = "us_scoring_daily_v1"
PUBLISHER_FILES = ("publish_site.py", "validate_public_site.py")
KR_CALENDAR = ("KR", date(2026, 1, 1), date(2026, 12, 31))
US_CALENDAR = ("US", date(2026, 1, 1), date(2027, 12, 31))
DIR_MODES = {"": 0o750, "config": 0o750, "config/calendars": 0o750, "publisher": 0o750,
             "releases": 0o750, "stock_data": 0o750, "stock_data/us": 0o750,
             "stock_data/us/output": 0o750, "prepared": 0o750, "prepared/kr": 0o750,
             "prepared/selections": 0o750, "projection": 0o750, "logs": 0o750,
             "runs": 0o700, "locks": 0o700, "private-projection": 0o700}


class ProvisionError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ProvisionError(message)


def _regular(path: Path, label: str) -> Path:
    _need(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    return path


def _directory(path: Path, label: str) -> Path:
    _need(path.is_dir() and not path.is_symlink(), f"{label} is not a directory: {path}")
    return path


def _pin(value: str, label: str) -> str:
    _need(bool(HEX.fullmatch(value)), f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _run(argv: list[str], *, cwd: Path, pythonpath: str, timeout: int = 900) -> str:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(PYTHONPATH=pythonpath, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, check=False, timeout=timeout)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-12:])
        raise ProvisionError(f"command failed ({done.returncode}): {' '.join(argv[:4])} ...\n{tail}")
    return done.stdout


def _reviewed_key(path: Path, modeler_root: Path) -> str:
    try:
        return "modeler/" + path.resolve(strict=True).relative_to(modeler_root.resolve(strict=True)).as_posix()
    except (ValueError, OSError):
        raise ProvisionError(f"file is not under modeler_root: {path}") from None


def _read_manifest(path: Path) -> dict[str, str]:
    listed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](\S.*)", line)
        if match:
            listed[match.group(2)] = match.group(1)
    _need(bool(listed), "source manifest is empty")
    return listed


class Plan:
    """Validated inputs and every target path; nothing here writes."""

    def __init__(self, args: argparse.Namespace) -> None:
        _need(bool(RELEASE_ID.fullmatch(args.release_id)), "release id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        self.args = args
        self.root = Path(os.path.abspath(args.serving_root))
        self.modeler = Path(os.path.abspath(args.modeler_root))
        self.collector = Path(os.path.abspath(args.collector_root))
        self.release = self.root / "releases" / args.release_id
        self.ops = self.root / "config" / "ops.json"
        self.venv = self.root / "venv"
        self.expected_target = self.root / "config" / Path(args.us_expected_source).name
        self.evidence_target = self.root / "config" / Path(args.parity_evidence).name
        self.publisher_dir = self.root / "publisher"
        self.lake_output = self.root / "stock_data" / "us" / "output"
        self.native_root = self.lake_output / SCORING_VERSION / "prepared"
        self.prepared = self.root / "prepared"
        self.checks: dict[str, Any] = {}

    def check(self) -> None:
        a = self.args
        _need(not self.root.is_symlink(), "serving root cannot be a symlink")
        _need(not os.path.lexists(self.release), f"release path already exists: {self.release}")
        _need(not os.path.lexists(self.ops), f"ops.json already exists: {self.ops}")
        for source in (self.modeler / "src", self.collector / "src"):
            _directory(source, "source tree")
        manifest = _regular(Path(a.source_manifest), "source manifest")
        listed = _read_manifest(manifest)
        self.checks["source_manifest_entries"] = len(listed)
        self.model_cards = _regular(Path(a.model_cards) if a.model_cards else
                                    self.modeler / "deploy/prod/model-cards.json", "model cards")
        self.runtime_manifest = _regular(Path(a.runtime_manifest) if a.runtime_manifest else
                                         self.modeler / "deploy/prod/runtime-verified-sj2-20260930.json",
                                         "runtime manifest")
        self.uv_lock = _regular(Path(a.uv_lock) if a.uv_lock else self.modeler / "uv.lock", "uv.lock")
        for bundle in (a.kr_bundle, a.us_lightgbm_bundle, a.us_ridge_bundle):
            _directory(Path(bundle), "model bundle")
        expected = _regular(Path(a.us_expected_source), "US expected source")
        evidence = _regular(Path(a.parity_evidence), "parity evidence")
        _need(sha256_file(expected) == _pin(a.us_expected_sha256, "--us-expected-sha256"),
              "US expected source SHA-256 differs from the pinned value")
        _need(sha256_file(evidence) == _pin(a.parity_evidence_sha256, "--parity-evidence-sha256"),
              "parity evidence SHA-256 differs from the pinned value")
        body = json.loads(expected.read_text(encoding="utf-8"))
        _need(body.get("schema_version") == "us-expected-source.v1" and body.get("reviewed_status") == "confirmed",
              "US expected source must be us-expected-source.v1 with reviewed_status=confirmed")
        self.checks["us_expected_dates"] = len(body.get("expected_session_by_report_date", {}))
        self.publisher_sources = {}
        pages_dir = Path(a.publisher_source_dir) if a.publisher_source_dir else self.modeler / "deploy/pages"
        for name in PUBLISHER_FILES:
            path = _regular(pages_dir / name, f"publisher script {name}")
            key = _reviewed_key(path, self.modeler)
            _need(listed.get(key) == sha256_file(path), f"publisher script is not the reviewed one: {key}")
            self.publisher_sources[name] = path
        self.pages_config = _regular(Path(a.pages_config), "pages base config")
        self.site_checkout = _directory(Path(a.site_checkout), "site checkout")
        self.lake = {name: _directory(Path(a.us_lake) / name, f"US lake {name}") for name in ("raw", "derived")}
        venv_python = self.venv / "bin" / "python"
        if self.venv.exists():
            _need(venv_python.exists(), f"existing venv has no bin/python: {self.venv}")
        else:
            _need((Path(a.venv_source) / "bin" / "python").exists(), "venv source has no bin/python")
        for target in (self.expected_target, self.evidence_target):
            if target.exists():
                source = expected if target == self.expected_target else evidence
                _need(sha256_file(target) == sha256_file(source), f"config file exists with other content: {target}")
        self.checks["publisher_scripts_reviewed"] = sorted(self.publisher_sources)

    def planned(self) -> list[str]:
        rel = [f"{name}/" for name in ("venv", "config", "config/calendars", "publisher", "runs", "projection",
                                       "private-projection", "locks", "logs", "prepared/kr", "prepared/selections")]
        rel += [f"releases/{self.args.release_id}/", "config/ops.json", "config/pins.json",
                f"config/{self.expected_target.name}", f"config/{self.evidence_target.name}",
                "config/calendars/calendar-KR-<hash>.json", "config/calendars/calendar-US-<hash>.json",
                "publisher/publish_site.py", "publisher/validate_public_site.py",
                "stock_data/us/raw -> " + str(self.lake["raw"]), "stock_data/us/derived -> " + str(self.lake["derived"]),
                f"stock_data/us/output/{SCORING_VERSION}/prepared/",
                "prepared/us -> stock_data/us/output/" + SCORING_VERSION + "/prepared"]
        return [f"{self.root}/{item}" for item in rel]


def _mkdir(path: Path, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, mode)


def _copy_file(source: Path, target: Path, mode: int) -> str:
    digest = sha256_file(source)
    if target.exists():
        _need(sha256_file(target) == digest, f"refusing to overwrite {target}")
    else:
        shutil.copyfile(source, target)
    os.chmod(target, mode)
    _need(sha256_file(target) == digest, f"SHA-256 differs after copy: {target}")
    return digest


def _write_json(path: Path, body: dict[str, Any], mode: int) -> str:
    raw = (json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    descriptor, temp = tempfile.mkstemp(prefix=".new-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, mode)
        os.link(temp, path)   # never replaces an existing file
    finally:
        Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _freeze(release: Path) -> None:
    for current, _, files in os.walk(release):
        for name in files:
            os.chmod(Path(current) / name, 0o440)
    for current, _, _ in os.walk(release, topdown=False):
        os.chmod(current, 0o550)


def _verify_runtime(plan: Plan, python: Path) -> None:
    code = ("import sys; from pathlib import Path; "
            "from modeler.serving.runtime_contract import verify_runtime; "
            "verify_runtime(Path(sys.argv[1]), Path(sys.argv[2])); print('ok')")
    _run([str(python), "-c", code, str(python), str(plan.runtime_manifest)],
         cwd=plan.modeler, pythonpath=str(plan.modeler / "src"), timeout=120)


def _calendar(plan: Plan, python: Path, release: Path, market: str, start: date, end: date) -> dict[str, Any]:
    argv = [str(python), "-m", "modeler.serving.calendar_sources", "--market", market,
            "--start", start.isoformat(), "--end", end.isoformat(),
            "--output-dir", str(plan.root / "config" / "calendars"),
            "--as-of", date.today().isoformat()]
    if market == "KR":
        argv += ["--holiday-csv", str(release / "src/collector/kr/infra/calendar/data/holidays_krx.csv")]
    out = _run(argv, cwd=release, pythonpath=str(release / "src"), timeout=120)
    info = json.loads(out.strip().splitlines()[-1])
    path = Path(info["path"])
    os.chmod(path, 0o440)
    info["sha256"] = sha256_file(path)
    return info


VALIDATE = r"""
import json, sys
from pathlib import Path
from modeler.serving import daily_coordinator as coordinator, daily_inputs as inputs
from modeler.serving.calendars import SessionCalendar
from modeler.serving.runtime_contract import verify_runtime
ops = Path(sys.argv[1])
config = coordinator._config(ops)
jobs, fixture = inputs._release_jobs(Path(config["release_manifest"]))
assert fixture is False, "release must not be synthetic"
runtime = verify_runtime(Path(config["python"]), Path(config["runtime_manifest"]))
cards = json.loads(Path(config["model_cards_path"]).read_text())
assert set(cards) == {model for _, model in jobs}, "model cards do not match the release jobs"
calendars = {}
for key in ("kr_calendar", "us_calendar"):
    manifest = json.loads(Path(config[key]).read_text())
    SessionCalendar.from_manifest(manifest)
    calendars[key] = {"coverage": [manifest["coverage_start"], manifest["coverage_end"]],
                      "sessions": len(manifest["sessions"])}
expected = json.loads(Path(config["us_expected_source"]).read_text())
table = expected["expected_session_by_report_date"]
print(json.dumps({"config_ok": True, "jobs": sorted(model for _, model in jobs),
                  "python": runtime["python"], "packages": runtime["packages"],
                  "calendars": calendars, "expected_dates": [min(table), max(table), len(table)],
                  "publisher_enabled": config["publisher_enabled"],
                  "external_verification_enabled": config["external_verification_enabled"]}, sort_keys=True))
"""


def _structure(plan: Plan) -> dict[str, Any]:
    prepared = plan.prepared
    _need(prepared.is_dir() and not prepared.is_symlink(), "prepared root must be a real directory")
    _need((prepared / "kr").is_dir() and not (prepared / "kr").is_symlink() and not any((prepared / "kr").iterdir()),
          "prepared/kr must be an empty real directory")
    _need((prepared / "us").is_symlink() and (prepared / "us").resolve() == plan.native_root.resolve(),
          "prepared/us must link to the native prepared directory")
    selections = prepared / "selections"
    _need(selections.is_dir() and prepared.resolve() in selections.resolve().parents,
          "selection root must live inside prepared root")
    for name in ("raw", "derived"):
        link = plan.root / "stock_data" / "us" / name
        _need(link.is_symlink() and link.resolve() == plan.lake[name].resolve(), f"lake link {name} is wrong")
    _need(plan.lake_output.is_dir() and not plan.lake_output.is_symlink(), "lake output must be a real directory")
    private = plan.root / "private-projection"
    _need(private.is_dir() and (private.stat().st_mode & 0o077) == 0, "private-projection must be owner-only")
    return {"prepared_us_link": str(os.readlink(prepared / "us")), "selections_inside_prepared": True,
            "lake_links": {name: os.readlink(plan.root / "stock_data/us" / name) for name in ("raw", "derived")}}


def provision(plan: Plan) -> dict[str, Any]:
    a = plan.args
    for rel, mode in DIR_MODES.items():
        _mkdir(plan.root / rel if rel else plan.root, mode)
    # venv: copy once, otherwise only verify what is already there.
    if not plan.venv.exists():
        shutil.copytree(a.venv_source, plan.venv, symlinks=True)
    python = plan.venv / "bin" / "python"
    _verify_runtime(plan, python)
    # frozen release
    _run([str(python), "-m", "modeler.serving.release_build",
          "--modeler-root", str(plan.modeler), "--collector-root", str(plan.collector),
          "--kr-bundle", a.kr_bundle, "--us-lightgbm-bundle", a.us_lightgbm_bundle,
          "--us-ridge-bundle", a.us_ridge_bundle, "--model-cards", str(plan.model_cards),
          "--runtime-manifest", str(plan.runtime_manifest), "--uv-lock", str(plan.uv_lock),
          "--source-manifest", a.source_manifest, "--output", str(plan.release)],
         cwd=plan.modeler, pythonpath=f"{plan.modeler / 'src'}:{plan.collector / 'src'}")
    release = plan.release.resolve(strict=True)
    # config: pinned inputs, calendars
    expected_sha = _copy_file(Path(a.us_expected_source), plan.expected_target, 0o440)
    evidence_sha = _copy_file(Path(a.parity_evidence), plan.evidence_target, 0o440)
    calendars = {"kr": _calendar(plan, python, release, *KR_CALENDAR),
                 "us": _calendar(plan, python, release, *US_CALENDAR)}
    # publisher copies (publisher stays disabled in ops.json)
    publisher = {}
    for name, source in plan.publisher_sources.items():
        target = plan.publisher_dir / name
        publisher[name] = {"path": str(target), "sha256": _copy_file(source, target, 0o440)}
    # stock_data links and native directory
    for name, target in plan.lake.items():
        link = plan.root / "stock_data" / "us" / name
        if not os.path.lexists(link):
            os.symlink(target, link)
    _mkdir(plan.native_root, 0o750)
    _mkdir(plan.native_root.parent, 0o750)
    if not os.path.lexists(plan.prepared / "us"):
        os.symlink(plan.native_root, plan.prepared / "us")
    release_json = release / "release.json"
    pins = {"schema_version": PINS_SCHEMA, "release_id": a.release_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "release_json_sha256": sha256_file(release_json),
            "source_manifest_sha256": sha256_file(Path(a.source_manifest)),
            "us_expected_source": {"path": str(plan.expected_target), "sha256": expected_sha},
            "parity_evidence": {"path": str(plan.evidence_target), "sha256": evidence_sha},
            "publisher_scripts": publisher}
    pins_sha = _write_json(plan.root / "config" / "pins.json", pins, 0o440)
    ops = {"schema_version": OPS_SCHEMA,
           "prepared_root": str(plan.prepared), "selection_root": str(plan.prepared / "selections"),
           "run_root": str(plan.root / "runs"), "projection_root": str(plan.root / "projection"),
           "private_projection_root": str(plan.root / "private-projection"),
           "previous_projection_dir": None,
           "release_manifest": str(release_json),
           "python": str(python), "python_sha256": sha256_file(python),
           "runtime_lock": str(release / "uv.lock"), "runtime_lock_sha256": sha256_file(release / "uv.lock"),
           "runtime_manifest": str(release / "runtime.json"),
           "runtime_manifest_sha256": sha256_file(release / "runtime.json"),
           "model_cards_path": str(release / "model-cards.json"),
           "model_cards_sha256": sha256_file(release / "model-cards.json"),
           "kr_calendar": calendars["kr"]["path"], "us_calendar": calendars["us"]["path"],
           "us_expected_source": str(plan.expected_target),
           "opening_artifact": None, "opening_snapshot_root": None, "opening_output_root": None,
           "opening_max_age_seconds": None,
           "base_path": "/market-briefing/",
           "publisher_enabled": False, "external_verification_enabled": False,
           "publisher_script": publisher["publish_site.py"]["path"],
           "publisher_script_sha256": publisher["publish_site.py"]["sha256"],
           "publisher_config": str(plan.pages_config), "site_checkout": str(plan.site_checkout),
           "actions_repository": a.actions_repository, "actions_workflow": a.actions_workflow,
           "public_manifest_url": a.public_manifest_url}
    ops_sha = _write_json(plan.ops, ops, 0o640)
    validation = json.loads(_run([str(python), "-c", VALIDATE, str(plan.ops)], cwd=release,
                                 pythonpath=str(release / "src"), timeout=300).strip().splitlines()[-1])
    structure = _structure(plan)
    _freeze(release)
    return {"status": "provisioned", "serving_root": str(plan.root), "release_id": a.release_id,
            "release_json": str(release_json), "release_json_sha256": pins["release_json_sha256"],
            "ops_json": str(plan.ops), "ops_json_sha256": ops_sha, "pins_json_sha256": pins_sha,
            "python": str(python), "python_sha256": ops["python_sha256"],
            "calendars": {key: {"path": value["path"], "sha256": value["sha256"],
                                "sessions": value["sessions"]} for key, value in calendars.items()},
            "publisher": publisher, "validation": validation, "structure": structure,
            "select_executed": False}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serving-root", required=True)
    p.add_argument("--release-id", required=True)
    p.add_argument("--modeler-root", required=True, help="canonical modeler source (has src/, deploy/, uv.lock)")
    p.add_argument("--collector-root", required=True, help="canonical collector source (has src/)")
    p.add_argument("--source-manifest", required=True, help="reviewed sha256sum list covering both repos")
    p.add_argument("--kr-bundle", required=True)
    p.add_argument("--us-lightgbm-bundle", required=True)
    p.add_argument("--us-ridge-bundle", required=True)
    p.add_argument("--venv-source", required=True, help="verified venv, copied with cp -a semantics")
    p.add_argument("--us-expected-source", required=True, help="confirmed E table")
    p.add_argument("--us-expected-sha256", required=True)
    p.add_argument("--parity-evidence", required=True)
    p.add_argument("--parity-evidence-sha256", required=True)
    p.add_argument("--pages-config", required=True, help="publisher base config (read, not copied)")
    p.add_argument("--site-checkout", required=True, help="Pages checkout (read, not copied)")
    p.add_argument("--us-lake", default="/home/whi/data/stock_data/us", help="operational lake (read-only link target)")
    p.add_argument("--model-cards")
    p.add_argument("--runtime-manifest")
    p.add_argument("--uv-lock")
    p.add_argument("--publisher-source-dir", help="default: <modeler-root>/deploy/pages")
    p.add_argument("--actions-repository", default="sjleekor/market-briefing")
    p.add_argument("--actions-workflow", default="pages.yml")
    p.add_argument("--public-manifest-url", default="https://sjleekor.github.io/market-briefing/site-manifest.json")
    p.add_argument("--dry-run", action="store_true", help="check inputs and list targets; write nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = Plan(args)
        plan.check()
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "serving_root": str(plan.root), "checks": plan.checks,
                              "would_create": plan.planned()}, indent=2, sort_keys=True))
            return 0
        print(json.dumps(provision(plan), indent=2, sort_keys=True))
        return 0
    except (ProvisionError, OSError, ValueError, KeyError, json.JSONDecodeError,
            subprocess.TimeoutExpired) as error:
        print(f"provision_serving failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
