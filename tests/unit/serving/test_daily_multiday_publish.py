"""Second-day behaviour: automatic previous projection and date-scoped publisher config."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

import test_daily_coordinator_e2e as e2e
from modeler.serving import daily_coordinator

D1 = date(2026, 9, 29)
D2 = date(2026, 9, 30)
PAGES = Path(e2e.PROJECT) / "deploy" / "pages"
WRAPPER = f'''import json, sys
from pathlib import Path
sys.path.insert(0, {str(PAGES)!r})
import publish_site
args = sys.argv[1:]
flag = Path({{flag!r}})
if flag.exists():
    flag.unlink()
    raise SystemExit(3)
config = json.loads(Path(args[args.index("--config") + 1]).read_text())
assert set(config) == publish_site.CONFIG_KEYS
publish_site.publish(config, allow_synthetic="--allow-synthetic" in args)
'''


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout.strip()


def _cli(release: Path, config: Path, stage: str, day: date, now: str, *,
         attempt: int | None = None, expect: int | None = 0) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    cmd = [sys.executable, "-m", "modeler.serving.daily_coordinator", stage, "--config", str(config),
           "--report-date", day.isoformat(), "--fixture-now", now]
    if attempt is not None:
        cmd += ["--attempt", str(attempt)]
    done = subprocess.run(cmd, cwd=release, env=env, text=True, capture_output=True, timeout=120)
    if expect is not None:
        assert done.returncode == expect, (stage, day, done.stdout, done.stderr)
    return json.loads(done.stdout) if done.stdout else {"returncode": done.returncode}


def _prepare_d2(tmp_path: Path, config: Path) -> None:
    for market in ("kr", "us"):
        name = "KR" if market == "kr" else "US"
        directory = tmp_path / "prepared" / market / "score_date=2026-09-29" / "prep_id=fixture"
        directory.mkdir(parents=True)
        feature = directory / ("feature_panel.parquet" if name == "KR" else "features.parquet")
        feature.write_bytes(b"synthetic model input D2: " + name.encode())
        native = e2e._write(directory / ("prepare_manifest.json" if name == "KR" else "manifest.json"),
            {"market": name, "feature_asof_date": "2026-09-29", "input_sha256": e2e._sha(feature),
             "features_sha256": e2e._sha(feature), "synthetic_fixture": True,
             "availability_evidence_type": "prepared_features_completion"})
        e2e._write(directory / "completion.json", {"schema_version": "prepared-features-completion.v1",
            "verified_available_by": "2026-09-30T09:20:00+09:00",
            "availability_evidence_type": "prepared_features_completion",
            "features_sha256": e2e._sha(feature), "native_prepare_manifest_sha256": e2e._sha(native)})
    body = json.loads(config.read_text())
    for key in ("kr_calendar", "us_calendar"):
        path = Path(body[key])
        calendar = json.loads(path.read_text())
        calendar["sessions"].append("2026-09-30")
        calendar["coverage_end"] = "2026-09-30"
        e2e._write(path, calendar)
    path = Path(body["us_expected_source"])
    expected = json.loads(path.read_text())
    expected["expected_session_by_report_date"][D2.isoformat()] = "2026-09-29"
    e2e._write(path, expected)


def _enable_publisher(tmp_path: Path, config: Path, *, flag: Path | None = None) -> tuple[Path, Path]:
    bare, seed, checkout = tmp_path / "pages.git", tmp_path / "seed", tmp_path / "checkout"
    seed.mkdir()
    _git("init", "--bare", "--initial-branch=site", str(bare))
    _git("init", "--initial-branch=site", str(seed))
    _git("-C", str(seed), "config", "user.name", "Publisher Test")
    _git("-C", str(seed), "config", "user.email", "publisher-test@example.invalid")
    (seed / "README.txt").write_text("seed\n")
    _git("-C", str(seed), "add", "README.txt")
    _git("-C", str(seed), "commit", "-m", "seed")
    _git("-C", str(seed), "remote", "add", "origin", str(bare))
    _git("-C", str(seed), "push", "origin", "site:site")
    _git("clone", "--branch", "site", str(bare), str(checkout))
    _git("-C", str(checkout), "config", "user.name", "Publisher Test")
    _git("-C", str(checkout), "config", "user.email", "publisher-test@example.invalid")
    script = tmp_path / "publisher_wrapper.py"
    script.write_text(WRAPPER.replace("{flag!r}", repr(str(flag or tmp_path / "no-such-flag"))))
    base = e2e._write(tmp_path / "publisher-base.json", {
        "repository_confirmed": True, "checkout_dir": str(checkout), "remote_name": "origin",
        "expected_remote_url": str(bare), "branch": "site", "base_path": "/market-briefing/"})
    body = json.loads(config.read_text())
    body.update({"publisher_enabled": True, "publisher_script": str(script),
                 "publisher_script_sha256": e2e._sha(script), "publisher_config": str(base),
                 "site_checkout": str(checkout), "actions_repository": "sjleekor/market-briefing",
                 "actions_workflow": "pages.yml", "public_manifest_url": "https://example.invalid/site-manifest.json"})
    e2e._write(config, body)
    return bare, base


def _day(release: Path, config: Path, day: date, now_select: str, now_run: str) -> None:
    assert _cli(release, config, "select", day, now_select)["status"] == "selected"
    assert _cli(release, config, "infer", day, now_run)["status"] == "inferred"
    assert _cli(release, config, "render", day, now_run)["status"] == "rendered"


def _two_days(tmp_path: Path):
    release, config = e2e._setup(tmp_path)
    bare, base = _enable_publisher(tmp_path, config)
    base_before = base.read_bytes()
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00")["status"] == "published"
    _prepare_d2(tmp_path, config)
    _day(release, config, D2, "2026-09-30T09:30:00+09:00", "2026-09-30T10:00:00+09:00")
    return release, config, bare, base, base_before


def test_second_day_selects_first_day_and_publishes_fast_forward(tmp_path: Path) -> None:
    release, config, bare, base, base_before = _two_days(tmp_path)
    runs = tmp_path / "runs"
    render2 = json.loads((runs / "2026-09-30" / "coordinator-render.json").read_text())
    render1 = json.loads((runs / "2026-09-29" / "coordinator-render.json").read_text())
    assert render1["previous_projection_dir"] is None
    assert render1["previous_site_manifest_sha256"] is None
    assert render2["previous_projection_dir"] == str(tmp_path / "site" / "2026-09-29")
    assert render2["previous_site_manifest_sha256"] == render1["site_manifest_sha256"]
    manifest = json.loads((tmp_path / "site" / "2026-09-30" / "site-manifest.json").read_text())
    assert [row["report_date"] for row in manifest["reports"]] == ["2026-09-30", "2026-09-29"] or \
        {row["report_date"] for row in manifest["reports"]} == {"2026-09-29", "2026-09-30"}
    assert (tmp_path / "site" / "2026-09-30" / "archive" / "index.html").is_file()
    archive = (tmp_path / "site" / "2026-09-30" / "archive" / "index.html").read_text()
    assert "2026-09-29" in archive and "2026-09-30" in archive

    first = _git("--git-dir", str(bare), "rev-parse", "refs/heads/site")
    assert _cli(release, config, "publish", D2, "2026-09-30T10:00:00+09:00")["status"] == "published"
    second = _git("--git-dir", str(bare), "rev-parse", "refs/heads/site")
    assert second != first
    assert _git("--git-dir", str(bare), "merge-base", "--is-ancestor", first, second) == ""
    assert _git("--git-dir", str(bare), "rev-parse", second + "^") == first

    c1 = runs / "2026-09-29" / "pages-publisher-config.json"
    c2 = runs / "2026-09-30" / "pages-publisher-config.json"
    j1, j2 = json.loads(c1.read_text()), json.loads(c2.read_text())
    assert j1["projection_dir"] == str(tmp_path / "site" / "2026-09-29")
    assert j2["projection_dir"] == str(tmp_path / "site" / "2026-09-30")
    assert {k: v for k, v in j1.items() if k != "projection_dir"} == \
        {k: v for k, v in j2.items() if k != "projection_dir"}
    assert base.read_bytes() == base_before
    publication = json.loads((runs / "2026-09-30" / "coordinator-publication.json").read_text())
    assert publication["publisher_config_path"] == str(c2)
    assert publication["publisher_config_sha256"] == hashlib.sha256(c2.read_bytes()).hexdigest()
    assert publication["site_commit"] == second


def test_base_publisher_config_with_projection_dir_is_rejected(tmp_path: Path) -> None:
    release, config_path = e2e._setup(tmp_path)
    _enable_publisher(tmp_path, config_path)
    _day(release, config_path, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    base = Path(json.loads(config_path.read_text())["publisher_config"])
    body = json.loads(base.read_text())
    body["projection_dir"] = str(tmp_path / "site" / "2026-09-29")
    e2e._write(base, body)
    result = _cli(release, config_path, "publish", D1, "2026-09-29T10:00:00+09:00", expect=1)
    assert result["returncode"] == 1
    assert not (tmp_path / "runs" / "2026-09-29" / "pages-publisher-config.json").exists()
    body["projection_dir"] = None
    e2e._write(base, body)
    assert _cli(release, config_path, "publish", D1, "2026-09-29T10:00:00+09:00")["status"] == "published"


def test_previous_projection_fixture_mismatch_stops_render(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    manifest_path = tmp_path / "site" / "2026-09-29" / "site-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["synthetic_fixture"] = False
    manifest_path.write_text(json.dumps(manifest))
    _prepare_d2(tmp_path, config)
    assert _cli(release, config, "select", D2, "2026-09-30T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config, "infer", D2, "2026-09-30T10:00:00+09:00")["status"] == "inferred"
    _cli(release, config, "render", D2, "2026-09-30T10:00:00+09:00", expect=1)
    assert not (tmp_path / "site" / "2026-09-30").exists()


def _manifest_dir(root: Path, name: str, latest: str | None = None, synthetic: bool = True) -> Path:
    path = root / name
    path.mkdir(parents=True)
    e2e._write(path / "site-manifest.json", {"latest_report_date": latest or name,
                                               "synthetic_fixture": synthetic})
    return path


def test_previous_selection_ignores_later_and_inconsistent_directories(tmp_path: Path) -> None:
    root = tmp_path / "site"
    older = _manifest_dir(root, "2026-09-27")
    _manifest_dir(root, "2026-09-28", latest="2026-09-27")  # name and manifest disagree
    (root / "2026-09-26").mkdir()  # no manifest
    _manifest_dir(root, "2026-10-05")  # later than D
    _manifest_dir(root, "not-a-date")
    config = {"projection_root": str(root)}
    chosen, files = daily_coordinator._previous_projection(config, D1, True)
    assert chosen == older
    assert "site-manifest.json" in files
    assert daily_coordinator._previous_projection(config, date(2026, 9, 27), True)[0] is None
    assert daily_coordinator._previous_projection({"projection_root": str(tmp_path / "none")}, D1, True) == (None, None)


def test_previous_selection_rejects_symlink_projection(tmp_path: Path) -> None:
    root = tmp_path / "site"
    target = _manifest_dir(tmp_path / "elsewhere", "2026-09-28")
    root.mkdir()
    (root / "2026-09-28").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        daily_coordinator._previous_projection({"projection_root": str(root)}, D1, True)


def test_previous_selection_explicit_path_is_used(tmp_path: Path) -> None:
    root = tmp_path / "site"
    _manifest_dir(root, "2026-09-28")
    explicit = _manifest_dir(tmp_path / "import", "2026-09-20")
    chosen, _ = daily_coordinator._previous_projection(
        {"projection_root": str(root), "previous_projection_dir": str(explicit)}, D1, True)
    assert chosen == explicit


def test_monitor_retry_republishes_with_same_date_scoped_config(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    flag = tmp_path / "fail-once"
    flag.write_text("x")
    bare, base = _enable_publisher(tmp_path, config, flag=flag)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00", expect=1)["status"] == "publisher_failed"
    daily = tmp_path / "runs" / "2026-09-29" / "pages-publisher-config.json"
    first_bytes = daily.read_bytes()
    monitor = _cli(release, config, "monitor", D1, "2026-09-29T10:15:00+09:00", attempt=1, expect=1)
    assert monitor["status"] == "actions_pending"
    publication = json.loads((tmp_path / "runs" / "2026-09-29" / "coordinator-publication.json").read_text())
    assert publication["status"] == "published"
    assert daily.read_bytes() == first_bytes
    assert publication["publisher_config_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert _git("--git-dir", str(bare), "rev-parse", "refs/heads/site") == publication["site_commit"]
