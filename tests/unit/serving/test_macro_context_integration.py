"""Macro-flow section, real modules end to end (plan 20261031_briefing_macro_context §9 M2-d).

The three pieces were built apart and tested against stand-ins (``macro_fakes``).  Here nothing is
faked: the coordinator's ``macro_context_stage`` starts the real child
(``modeler.serving.macro_context_run``), which calls the real ``build_document`` on synthetic KR
and US macro tables; the real renderer (``reporting.macro``) turns the document into ``macro.md``
and ``assets/macro-*.svg`` inside the unit; and the real publisher and ``validate_reports`` take
that unit.  Synthetic data only.

The release world is the market-sector one (``ms_fixtures``), whose 09:30 select pinned a
selection document for the market-sector bundle's own tables.  Those tables carry no macro series,
so after the market-sector stage has used its selection the test swaps in a
``market-sector-selection.v1`` document for the macro world (``tests/scores/macro_context/_world``,
moved to the release's D) -- the same shape the select stage writes, pointing at lake roots that
hold ``common_feature_observation_raw`` and ``macro_series``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import ms_fixtures as F
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reporting"))
import test_daily_coordinator_e2e as e2e  # noqa: E402
import test_daily_multiday_publish as mp  # noqa: E402
from reports_world import REPORTS_DIR, pr  # noqa: E402
from tests.scores.macro_context import _world as W  # noqa: E402

from modeler.reporting import macro as real_render  # noqa: E402
from modeler.reporting import markdown as md  # noqa: E402
from modeler.scores import macro_context as real_calc  # noqa: E402
from modeler.serving import daily_coordinator as dc  # noqa: E402

D = F.D
NOW = datetime.fromisoformat(F.RUN_AT)
UNIT_DIR = f"reports/daily-briefing/{D:%Y}/{D:%m}/{D}"
PAGE = md.MACRO_FILE
FIVE = tuple(md.SECTION_FILES.values())
FIGURE_LINK = re.compile(r"!\[[^\]]*\]\((assets/[^)]+)\)")
PINNED_RUN_FILES = ("report-{D}.json", "market-sector-{D}.json", "explain-{D}.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / D.isoformat()


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / "prepared" / "selections" / D.isoformat() / "selection-state.json"


def _block(tmp_path: Path) -> dict:
    return json.loads(_state_file(tmp_path).read_text())["market_sector"]


def _set(config: Path, **changes) -> None:
    body = json.loads(config.read_text())
    body.update(changes)
    config.write_text(json.dumps(body, sort_keys=True))


def _cli(release: Path, config: Path, stage: str, now: str = F.RUN_AT, expect: int | None = 0):
    return mp._cli(release, config, stage, D, now, expect=expect)


def _doc(tmp_path: Path) -> dict:
    return json.loads((_run_dir(tmp_path) / f"macro-{D}.json").read_text())


def _stage(config: Path) -> dict:
    """The coordinator's macro stage in this process; the calculation runs in the real child."""
    return dc.macro_context_stage(dc._config(config), D, NOW)


def _unit_tree(root: Path) -> dict[str, bytes]:
    unit = root / UNIT_DIR
    return {
        p.relative_to(unit).as_posix(): p.read_bytes()
        for p in sorted(unit.rglob("*"))
        if p.is_file()
    }


def _front(text: str) -> dict:
    meta, _body = md.parse_front_matter(text)
    assert meta is not None
    return meta


def _shifted_macro_world(root: Path, monkeypatch, *, drop_kr_series: str | None = None) -> W.World:
    """The calculation tests' synthetic lake, moved 21 days back so its D is the release's D."""
    shift = D - W.D

    def snap(text: str) -> str:
        return (date.fromisoformat(text) + shift).isoformat()

    monkeypatch.setattr(W, "D", D)
    monkeypatch.setattr(W, "KR_SNAP", snap(W.KR_SNAP))
    monkeypatch.setattr(W, "US_SNAP", snap(W.US_SNAP))
    monkeypatch.setattr(W, "KR_DONE", f"{snap('2026-10-28')}T04:29:15+09:00")
    monkeypatch.setattr(W, "US_MACRO_LAST", W.US_MACRO_LAST + shift)
    monkeypatch.setattr(W, "COPY_LAST", W.COPY_LAST + shift)
    monkeypatch.setattr(W, "KR_LAST", W.KR_LAST + shift)
    world = W.build_world(root)
    if drop_kr_series is not None:  # before the selection pins the file
        frame = pl.read_parquet(world.kr_file)
        frame.filter(pl.col("series_id") != drop_kr_series).write_parquet(world.kr_file)
    done = datetime(D.year, D.month, D.day, tzinfo=W.SEOUL) - timedelta(minutes=20)
    os.utime(
        world.us_file, (done.timestamp(), done.timestamp())
    )  # the US table finished before 09:30
    return world


def _swap_in_macro_selection(tmp_path: Path, world: W.World) -> Path:
    """Point the pinned market-sector block at a selection document for the macro world."""
    path = tmp_path / "macro-selection" / "ms-selection.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(W.make_selection(world, D), sort_keys=True))
    state = json.loads(_state_file(tmp_path).read_text())
    state["market_sector"] = {
        **state["market_sector"],
        "selection": str(path),
        "selection_sha256": _sha(path),
    }
    _state_file(tmp_path).write_text(json.dumps(state, sort_keys=True))
    return path


def make_world(tmp_path: Path, monkeypatch, **world_options):
    """Real release (no stand-ins), 09:30 select, inference and market-sector run, publisher on a
    local bare remote, macro switch still off.  Returns (release, config, reports, macro selection,
    the market-sector selection 09:30 select pinned)."""
    release, config, _data = F.build_release_world(tmp_path, monkeypatch)
    reports = mp._enable_publisher(tmp_path, config)
    assert _cli(release, config, "select", F.SELECT_AT)["status"] == "selected"
    assert _cli(release, config, "infer")["status"] == "inferred"
    assert _cli(release, config, "market-sector")["status"] in {
        "market_sector_ready",
        "market_sector_partial",
    }
    own_selection = Path(_block(tmp_path)["selection"])
    macro_world = _shifted_macro_world(tmp_path / "macro-data", monkeypatch, **world_options)
    return release, config, reports, _swap_in_macro_selection(tmp_path, macro_world), own_selection


@pytest.fixture()
def world(tmp_path, monkeypatch):
    return make_world(tmp_path, monkeypatch)


def _publish_unit(release: Path, config: Path) -> dict:
    assert _cli(release, config, "render")["status"] == "rendered"
    return _cli(release, config, "publish")


# ----------------------------------------------------------------------------- the ok path
def test_the_real_child_computes_an_ok_document_from_the_pinned_selection(tmp_path, world):
    _release, config, _reports, selection, _own = world
    _set(config, macro_context_enabled=True)
    state = _stage(config)
    run = _run_dir(tmp_path)
    doc = _doc(tmp_path)
    inference = json.loads((run / "coordinator-inference.json").read_text())
    assert state["status"] == "ok" and doc["state"] == "ok"
    assert doc["schema"] == real_calc.SCHEMA == "briefing-macro.v1" and doc["unit"] == D.isoformat()
    assert doc["invocation_id"] == inference["invocation_id"] == state["invocation_id"]
    assert doc["release"] == "frozen-release"
    assert doc["selection_sha256"] == _block(tmp_path)["selection_sha256"] == _sha(selection)
    assert state["selection_sha256"] == doc["selection_sha256"]
    assert state["document_sha256"] == _sha(run / f"macro-{D}.json")
    assert (
        doc["inputs"]["selection"]["report_date"] == D.isoformat()
    )  # the pinned document was read
    assert doc["checks"]["overall_pass"] is True and doc["checks"]["violations"] == []
    assert len(doc["figures"]) == 21 and len(doc["indicators"]) == 31
    raw = (run / f"macro-{D}.json").read_bytes()
    assert raw == real_calc.document_bytes(json.loads(raw))  # canonical bytes
    assert not list(run.glob(".macro-*")) and not list(run.glob("macro-*.superseded-*"))


def test_the_ok_unit_carries_page_figures_links_and_passes_every_check(tmp_path, world):
    release, config, reports, _selection, _own = world
    _set(config, macro_context_enabled=True)
    assert _stage(config)["status"] == "ok"
    body = _doc(tmp_path)
    published = _publish_unit(release, config)
    assert published["status"] == "published", published
    run = _run_dir(tmp_path)
    tree = _unit_tree(reports.checkout)
    figures = {f"assets/macro-{f['name']}.svg" for f in body["figures"]}
    assert len(figures) == 21 and set(tree) == {*FIVE, PAGE, *figures}
    page = tree[PAGE].decode()
    meta = _front(page)
    assert meta["section"] == "macro" and meta["status"] == "ok" and meta["unit"] == D.isoformat()
    assert meta["source"]["macro_sha256"] == _sha(run / f"macro-{D}.json")
    assert meta["source"]["release"] == "frozen-release" and meta["synthetic_fixture"] is True
    # The page is what the real renderer makes of the document, front matter aside.
    text, svgs = real_render.render(body)
    assert page.split("\n---\n", 1)[1].strip() == text.strip()
    assert {f"assets/{name}" for name in svgs} == figures
    # Links: every figure the page shows is in the unit, and every figure of the unit is shown.
    assert set(FIGURE_LINK.findall(page)) == figures
    asof = body["header"]["daily_asof_max"]
    assert f"| [거시 흐름](macro.md) | 정상 | {asof} |" in tree["README.md"].decode()
    assert "거시 흐름은 [macro.md](macro.md)" in tree["market-sector.md"].decode()
    # SVG: name, size, no script / event / external reference.
    for name in figures:
        raw = tree[name]
        assert re.fullmatch(r"assets/macro-[A-Za-z0-9._\-]+\.svg", name)
        assert 0 < len(raw) <= md.SVG_MAX_BYTES and md.svg_violations(name, raw) == []
        assert raw == svgs[name.removeprefix("assets/")].encode()
    # Publisher output == what the local step stored; the pushed tree passes the repository checks.
    local = run / "markdown" / "unit"
    assert {
        p.relative_to(local).as_posix(): p.read_bytes() for p in local.rglob("*") if p.is_file()
    } == tree
    clone = reports.clone_for_checks()
    assert md.validate(clone, allow_synthetic=True) == []
    done = subprocess.run(
        [
            sys.executable,
            str(REPORTS_DIR / "validate_reports.py"),
            "--repo",
            str(clone),
            "--allow-synthetic",
        ],
        env={**os.environ, "PYTHONPATH": str(release / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = json.loads((run / "markdown" / "manifest.json").read_text())
    assert manifest["macro_state"] == "ok" and manifest["macro_sha256"] == _sha(
        run / f"macro-{D}.json"
    )


# ------------------------------------------------------------------ invariants
def test_macro_leaves_report_market_sector_explain_and_selection_bytes_alone(tmp_path, world):
    release, config, _reports, selection, own_selection = world
    _set(config, macro_context_enabled=True, explain_enabled=True)
    explained = _cli(
        release, config, "explain", expect=None
    )  # synthetic bundles: a failed document
    assert explained["status"] == "failed"
    run = _run_dir(tmp_path)
    pinned = [run / name.format(D=D) for name in PINNED_RUN_FILES]
    pinned += [selection, own_selection, _state_file(tmp_path), run / "run-state.json"]
    assert all(path.is_file() for path in pinned)
    before = {path: _sha(path) for path in pinned}
    assert _stage(config)["status"] == "ok"
    assert {path: _sha(path) for path in pinned} == before
    assert _sha(selection) == _block(tmp_path)["selection_sha256"]


def test_a_rerun_gives_the_same_document_bytes_and_the_same_unit_text(tmp_path, world):
    _release, config, _reports, _selection, _own = world
    _set(config, macro_context_enabled=True)
    assert _stage(config)["status"] == "ok"
    run = _run_dir(tmp_path)
    first = (run / f"macro-{D}.json").read_bytes()
    assert _stage(config)["status"] == "ok"  # same invocation, same pinned selection
    (old,) = run.glob(f"macro-{D}.superseded-*.json")  # the first one, set aside
    assert old.read_bytes() == first == (run / f"macro-{D}.json").read_bytes()
    body = json.loads(first)
    assert real_render.render(body) == real_render.render(json.loads(first))


def test_off_the_unit_is_byte_for_byte_the_unit_without_the_feature(tmp_path, world):
    release, config, reports, _selection, _own = world
    assert _stage(config) == {"report_date": D.isoformat(), "status": "disabled"}
    assert not (_run_dir(tmp_path) / f"macro-{D}.json").exists()
    assert _cli(release, config, "render")["status"] == "rendered"
    run = _run_dir(tmp_path)
    daily = dc._reports_publisher_input(
        dc._config(config), D, json.loads((run / "coordinator-render.json").read_text()), run
    )
    daily["generated_at"] = f"{D}T10:03:12+09:00"
    plain = pr.local_step(daily, allow_synthetic=True)
    assert plain["status"] == "local_done" or plain.get("local_done") is True, plain
    unit = run / "markdown" / "unit"
    names = sorted(p.relative_to(unit).as_posix() for p in unit.rglob("*") if p.is_file())
    assert names == sorted(FIVE)
    plain_tree = {name: (unit / name).read_bytes() for name in names}
    plain_manifest = (run / "markdown" / "manifest.json").read_bytes()
    joined = b"".join(plain_tree.values())
    assert (
        b"macro.md" not in joined
        and b"assets/macro-" not in joined
        and "[거시 흐름]".encode() not in joined
    )
    assert not any(key.startswith("macro_") for key in json.loads(plain_manifest))
    # A macro document that exists but is switched off is not read: the same bytes.
    _set(config, macro_context_enabled=True)
    assert _stage(config)["status"] == "ok"
    _set(config, macro_context_enabled=False)
    off = pr.local_step({**daily, "macro_context_enabled": False}, allow_synthetic=True)
    assert off["unit_sha256"] == plain["unit_sha256"]
    assert {n: (unit / n).read_bytes() for n in names} == plain_tree
    assert (run / "markdown" / "manifest.json").read_bytes() == plain_manifest
    # ...and switched on, the unit changes.
    on = pr.local_step({**daily, "macro_context_enabled": True}, allow_synthetic=True)
    assert on["unit_sha256"] != plain["unit_sha256"]
    assert (unit / PAGE).is_file()


# ----------------------------------------------------------------- failure paths
def test_a_missing_series_makes_the_real_child_fail_and_the_unit_still_publishes(
    tmp_path, monkeypatch
):
    release, config, reports, _selection, _own = make_world(
        tmp_path, monkeypatch, drop_kr_series="rate_kr_gov10y"
    )
    _set(config, macro_context_enabled=True)
    state = _stage(config)
    doc = _doc(tmp_path)
    assert (
        state["status"] == "failed"
        and state["reason"] == "exit_nonzero"
        and state["exit_code"] == 1
    )
    assert doc["state"] == "failed" and doc["reason"] == "exit_nonzero"
    assert doc["invocation_id"] == state["invocation_id"]
    assert doc["selection_sha256"] == _block(tmp_path)["selection_sha256"]
    assert "figures" not in doc and "indicators" not in doc
    assert not list(_run_dir(tmp_path).glob(".macro-*"))
    assert _publish_unit(release, config)["status"] == "published"
    tree = _unit_tree(reports.checkout)
    assert set(tree) == {*FIVE, PAGE}  # no figures, no half page
    page = tree[PAGE].decode()
    assert "거시 흐름을 이번 실행에서 만들지 못했습니다(사유 코드: exit_nonzero)." in page
    assert _front(page)["status"] == "failed"
    assert "| [거시 흐름](macro.md) | 실패 |" in tree["README.md"].decode()
    assert md.validate(reports.clone_for_checks(), allow_synthetic=True) == []


def test_a_selection_whose_input_changed_after_pinning_is_not_computed(tmp_path, world):
    """The calculation re-checks the lake against the pins: a file changed after 09:30 fails it."""
    release, config, reports, _selection, _own = world
    victim = next((tmp_path / "macro-data" / "us").rglob("part.parquet"))
    victim.write_bytes(victim.read_bytes() + b"\0")
    _set(config, macro_context_enabled=True)
    state = _stage(config)
    assert state["status"] == "failed" and state["reason"] == "exit_nonzero"
    assert _doc(tmp_path)["state"] == "failed"
    assert _publish_unit(release, config)["status"] == "published"
    assert set(_unit_tree(reports.checkout)) == {*FIVE, PAGE}


def test_a_failure_report_run_still_gets_the_section(tmp_path, world):
    """A failure report (no model result) does not stop the section: it needs the selection only."""
    _release, config, _reports, _selection, _own = world
    _set(config, macro_context_enabled=True)
    run = _run_dir(tmp_path)
    infer = json.loads((run / "coordinator-inference.json").read_text())
    dc._atomic(run / "coordinator-inference.json", {**infer, "failure_report": True})
    assert _stage(config)["status"] == "ok" and _doc(tmp_path)["state"] == "ok"


def test_the_wrapper_run_with_real_modules_publishes_the_whole_section(tmp_path, monkeypatch):
    """The scheduled entry (``daily_wrapper run``): stages in order, no help from the test."""
    release, config, reports, _selection, _own = make_world(tmp_path, monkeypatch)
    _set(config, macro_context_enabled=True)
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "modeler.serving.daily_wrapper",
            "run",
            "--config",
            str(config),
            "--report-date",
            D.isoformat(),
            "--fixture-now",
            F.RUN_AT,
        ],
        cwd=release,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    stages = json.loads(done.stdout)["stages"]
    assert stages["macro_context"] == "ok" and stages["publication"] == "published"
    tree = _unit_tree(reports.checkout)
    assert PAGE in tree and sum(name.startswith("assets/macro-") for name in tree) == 21


# -------------------------------------------------------- the e2e release's module list
def test_the_e2e_release_ships_the_real_modules_and_a_missing_selection_is_a_notice(tmp_path):
    """``e2e._setup`` (explicit MODULES list) now includes the real macro modules: with the switch
    on and no market-sector section the real failure document and real notice page go out."""
    release, config = e2e._setup(tmp_path)
    src = Path(__file__).resolve().parents[3] / "src"
    for relative in (
        "modeler/reporting/macro.py",
        "modeler/scores/macro_context/build.py",
        "modeler/scores/macro_context/load.py",
        "modeler/scores/__init__.py",
    ):
        assert (release / "src" / relative).read_bytes() == (src / relative).read_bytes()
    reports = mp._enable_publisher(tmp_path, config)
    _set(config, macro_context_enabled=True)
    stages = mp._wrapper_run(release, config, mp.D1, "2026-09-29T10:00:00+09:00")
    assert stages["macro_context"] == "failed" and stages["publication"] == "published"
    run = tmp_path / "runs" / mp.D1.isoformat()
    doc = json.loads((run / f"macro-{mp.D1}.json").read_text())
    assert doc["state"] == "failed" and doc["reason"] == "not_in_selection"
    assert doc["selection_sha256"] is None and doc["schema"] == real_calc.SCHEMA
    page = mp._unit_files(reports, mp.D1)[PAGE].decode()
    assert "거시 흐름을 이번 실행에서 만들지 못했습니다(사유 코드: not_in_selection)." in page
    assert not any(name.startswith("assets/macro-") for name in mp._unit_files(reports, mp.D1))
