"""Release builder of the shadow serving (``shadow_release.py``). Synthetic bundle only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.models._03_kr_next import shadow as sh
from modeler.models._03_kr_next import shadow_release as sr
from tests.unit.test_kr_next_shadow import SESSIONS, bundle_dir, make_panel  # noqa: F401

MODELER_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("rev") / "sources.sha256"
    path.write_text(sr.source_manifest_text(MODELER_ROOT), encoding="utf-8")
    return path


def test_closure_contains_what_shadow_needs_and_nothing_of_serving():
    files = sr.import_closure(MODELER_ROOT)
    for need in (
        "src/modeler/__init__.py",
        "src/modeler/models/__init__.py",
        "src/modeler/models/_03_kr_next/__init__.py",
        "src/modeler/models/_03_kr_next/shadow.py",
        "src/modeler/models/_03_kr_next/dataset.py",
        "src/modeler/models/_03_kr_next/evaluate.py",
        "src/modeler/models/_03_kr_next/bundle.py",
        "src/modeler/models/_03_kr_next/preprocess.py",
        "src/modeler/models/_03_kr_next/feature_set.py",
        "src/modeler/etl/universe_terminal.py",
        "src/modeler/etl/labels_entry.py",
    ):
        assert need in files, need
    assert not [f for f in files if "/serving/" in f]
    assert "src/modeler/models/_03_kr_next/shadow_release.py" in files  # ships its own verifier
    assert not [f for f in files if f.endswith("train.py")]


def test_release_is_built_verified_and_importable_from_its_own_dir(tmp_path, reviewed, bundle_dir):  # noqa: F811
    out = tmp_path / "rel"
    summary = sr.build_release(
        modeler_root=MODELER_ROOT, source_manifest=reviewed, output=out, bundle_dir=bundle_dir
    )
    assert summary["bundle_verified"] is True and summary["python_file_count"] > 20
    meta = json.loads((out / "release.json").read_text())
    assert meta["entrypoint"] == "modeler.models._03_kr_next.shadow:main"
    assert sr.verify_release(out)["code_sha256"] == meta["code_sha256"]

    # import in a clean subprocess: every modeler module must come from the release
    code = (
        "import sys, json\n"
        "import modeler.models._03_kr_next.shadow as s\n"
        "import modeler.models._03_kr_next.dataset, modeler.models._03_kr_next.evaluate\n"
        "mods = {n: getattr(m, '__file__', None) for n, m in sys.modules.items()\n"
        "        if n == 'modeler' or n.startswith('modeler.')}\n"
        "print(json.dumps(mods))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(out / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    mods = json.loads(res.stdout.strip().splitlines()[-1])
    assert mods and all(f and f.startswith(str(out / "src")) for f in mods.values()), mods

    # and a real day scored through the release's CLI, bundle taken from the release
    d = SESSIONS[5]
    t = sh.previous_session(SESSIONS, d)
    panel = tmp_path / "panel.parquet"
    make_panel(t, n=40, seed=2).write_parquet(panel)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"sessions": [s.isoformat() for s in SESSIONS]}))
    cli = [
        sys.executable,
        "-m",
        "modeler.models._03_kr_next.shadow",
        "score",
        "--date",
        d.isoformat(),
        "--panel",
        str(panel),
        "--bundle-dir",
        str(out / "bundle"),
        "--out-root",
        str(tmp_path / "o"),
        "--ledger",
        str(tmp_path / "o" / "ledger.jsonl"),
        "--sessions",
        str(cal),
    ]
    res = subprocess.run(cli, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode in (0, 3), res.stderr  # 3 = written after the 15:00 deadline
    rec = json.loads((tmp_path / "o" / f"D={d.isoformat()}" / "record.json").read_text())
    shipped = {i["path"].rsplit("/", 1)[-1] for i in meta["files"]}
    assert set(rec["code"]["files"]) <= shipped  # the record names the release's own files
    assert rec["bundle"]["manifest_sha256"] == meta["bundle"]["manifest_sha256"]
    assert sh.verify_ledger(tmp_path / "o" / "ledger.jsonl")["ok"]


def test_refuses_existing_output_and_leaves_it_alone(tmp_path, reviewed):
    out = tmp_path / "rel"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(sr.ShadowReleaseError, match="already exists"):
        sr.build_release(modeler_root=MODELER_ROOT, source_manifest=reviewed, output=out)
    assert (out / "keep.txt").read_text() == "x"
    assert [p.name for p in tmp_path.iterdir()] == ["rel"]


def test_unreviewed_or_changed_source_stops_the_build(tmp_path, reviewed):
    lines = reviewed.read_text().splitlines()
    shadow_line = next(i for i, x in enumerate(lines) if x.endswith("_03_kr_next/shadow.py"))
    short = tmp_path / "short.sha256"
    short.write_text("\n".join(x for i, x in enumerate(lines) if i != shadow_line) + "\n")
    with pytest.raises(sr.ShadowReleaseError, match="not in the reviewed"):
        sr.build_release(modeler_root=MODELER_ROOT, source_manifest=short, output=tmp_path / "a")
    bad = list(lines)
    bad[shadow_line] = "0" * 64 + bad[shadow_line][64:]
    wrong = tmp_path / "wrong.sha256"
    wrong.write_text("\n".join(bad) + "\n")
    with pytest.raises(sr.ShadowReleaseError, match="differs from the reviewed"):
        sr.build_release(modeler_root=MODELER_ROOT, source_manifest=wrong, output=tmp_path / "b")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["short.sha256", "wrong.sha256"]


def test_verify_detects_a_changed_file_and_an_incomplete_closure(tmp_path, reviewed):
    out = tmp_path / "rel"
    sr.build_release(modeler_root=MODELER_ROOT, source_manifest=reviewed, output=out)
    target = out / "src/modeler/models/_03_kr_next/preprocess.py"
    original = target.read_text()
    target.write_text(original + "\n# edited\n")
    with pytest.raises(sr.ShadowReleaseError, match="changed"):
        sr.verify_release(out)
    target.write_text(original)
    sr.verify_release(out)
    # a release that lost a module it imports is not closed
    meta = json.loads((out / "release.json").read_text())
    gone = "src/modeler/etl/universe_terminal.py"
    (out / gone).unlink()
    meta["files"] = [i for i in meta["files"] if i["path"] != gone]
    pairs = [(i["path"], i["sha256"]) for i in meta["files"]]
    meta["code_sha256"] = sr.code_inventory_sha256(pairs)
    (out / "release.json").write_text(json.dumps(meta))
    with pytest.raises(sr.ShadowReleaseError, match="not closed"):
        sr.verify_release(out)


def test_bundle_copy_checks_the_manifest_pins(tmp_path, reviewed, bundle_dir):  # noqa: F811
    broken = tmp_path / "b"
    broken.mkdir()
    for f in bundle_dir.iterdir():
        (broken / f.name).write_bytes(f.read_bytes())
    pl.read_parquet(broken / "golden_preds.parquet").with_columns(
        pl.col("score") + 1
    ).write_parquet(broken / "golden_preds.parquet")
    with pytest.raises(sr.ShadowReleaseError, match="differs before copy"):
        sr.build_release(
            modeler_root=MODELER_ROOT,
            source_manifest=reviewed,
            output=tmp_path / "r",
            bundle_dir=broken,
        )
    assert not (tmp_path / "r").exists()


def test_date_in_module_docstring_is_not_used():
    assert date(2026, 11, 2) in SESSIONS
