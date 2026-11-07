"""P6′ shadow serving (``_04_kr_next_p6prime.shadow*``): v4p input, P6′ bundle and root only,
P6's record format. Synthetic raw lake and synthetic bundles only; no label is built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from modeler.etl.marts import financial_quarters_v4p as fq4p
from modeler.models._03_kr_next import build_panel as p6_bp
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import preprocess as pp
from modeler.models._03_kr_next import shadow as p6_sh
from modeler.models._03_kr_next import shadow_prepare as p6_sp
from modeler.models._03_kr_next import shadow_release as p6_sr
from modeler.models._03_kr_next import views as p6_views
from modeler.models._04_kr_next_p6prime import inputs, tags
from modeler.models._04_kr_next_p6prime import shadow as sh
from modeler.models._04_kr_next_p6prime import shadow_prepare as sp
from modeler.models._04_kr_next_p6prime import shadow_release as sr

from .test_kr_next_shadow import COLS, make_panel
from .test_kr_next_shadow_prepare import SESSIONS, SMALL_UNIVERSE, D, T, make_raw, sessions_file
from .test_kr_next_shadow_prepare import _wrapper_env as p6_wrapper_env
from .test_kr_next_shadow_release import COLLECTOR_ROOT, MODELER_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "deploy/prod/bin/kr-next-p6prime-shadow.sh"
FIN = inputs.FIN_FILE
NOW = p6_sh.deadline_at(D) - timedelta(hours=5)  # 10:00 KST of D: a valid write


def _bundle(base: Path, prereg_tag: str) -> Path:
    """A small bundle (same model for any tag, so P6 and P6′ fixtures score alike)."""
    frames = [
        make_panel(date(2026, 10, 1) + timedelta(days=i), n=80, seed=100 + i) for i in range(6)
    ]
    big = pl.concat(frames)
    ranked = pp.rank_transform(big.with_columns([pl.col(c).cast(pl.Float64) for c in COLS]), COLS)
    y = np.random.default_rng(5).normal(size=ranked.height)
    model = HistGradientBoostingRegressor(max_iter=10, random_state=0).fit(
        bundle_mod.model_matrix(ranked, COLS), y
    )
    ridge = Ridge(alpha=100.0).fit(bundle_mod.ridge_matrix(ranked, COLS), y)
    return bundle_mod.write_bundle(
        base,
        model=model,
        ridge=ridge,
        ridge_alpha=100.0,
        feature_cols=COLS,
        golden_frame=ranked,
        params={"learning_rate": 0.1},
        max_iter=10,
        score_matrix={"scores": [[0.0]]},
        determinism={"identical": True},
        snapshot_id="snap-test",
        prereg_tag=prereg_tag,
        prereg_commit="abc",
        modeler_commit="def",
        prereg_doc_sha256="0" * 64,
        preprocessing={"rank": "per-date"},
    )


@pytest.fixture(scope="module")
def bundles(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("bundles")
    return {
        "p6": _bundle(base / "p6", tags.P6_PREREG_TAG),
        "p6prime": _bundle(base / "p6prime", tags.PREREG_TAG),
    }


def _small_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    real = p6_bp.build_panel_file
    monkeypatch.setattr(
        p6_bp,
        "build_panel_file",
        lambda con, **kw: real(con, **{**kw, "universe_filter": SMALL_UNIVERSE}),
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> dict:
    """One raw lake; the P6′ prepare (``run`` path) and, for comparison, the P6 prepare."""
    base = tmp_path_factory.mktemp("p6prime_prepare")
    lake = base / "lake"
    raw = make_raw(lake)
    root = base / "shadow_p6prime"
    with pytest.MonkeyPatch.context() as mp:
        _small_universe(mp)
        res = sp.prepare_from_lake(
            D, raw_lake=lake / "raw" / "raw_postgres", out_root=root, sessions=SESSIONS,
            memory_limit="2GB", threads=2,
        )  # fmt: skip
    # P6 reference: the P6 view build and P6 prepare of the same snapshot, in a P6-style root
    p6_root = base / "shadow"
    p6_vdir = base / "p6_views"
    p6_views.build_views(raw, p6_vdir, memory_limit="2GB", threads=2)
    p6_res = p6_sp.prepare_day(
        D, raw_root=raw, views_dir=p6_vdir, out_root=p6_root, sessions=SESSIONS,
        memory_limit="2GB", threads=2, universe_filter=SMALL_UNIVERSE,
    )  # fmt: skip
    return {"base": base, "raw": raw, "root": root, "p6_root": p6_root, "p6_vdir": p6_vdir,
            "p6": p6_res, **res}  # fmt: skip


def _cols(path: Path) -> list[str]:
    return list(pl.read_parquet_schema(path))


# ----------------------------------------------------------------------- prepare: v4p is used


def test_panel_reads_the_v4p_mart_and_the_p6_price_views(prepared) -> None:
    root = prepared["root"]
    p6p_dir, p6_dir = sp.views_dir_for(root, D), sp.p6_views_dir_for(root, D)
    vman = json.loads((p6p_dir / "manifest.json").read_text())
    assert vman["p6prime"]["fin_file_holds"] == fq4p.FQMV_DLR_V4P_TABLE
    assert vman["p6prime"]["p6_views_dir"] == str(p6_dir)
    # the financial file under the P6 name is the v4p mart (its own columns), not P6's _dlr
    fin_p6prime, fin_p6 = p6p_dir / f"{FIN}.parquet", p6_dir / f"{FIN}.parquet"
    assert {"fiscal_year_end", "standalone_null_reason"} <= set(_cols(fin_p6prime))
    assert not {"fiscal_year_end", "standalone_null_reason"} & set(_cols(fin_p6))
    p6man = json.loads((p6_dir / "manifest.json").read_text())
    # PIT and quality: P6's files byte for byte
    for name in inputs.COPIED_FROM_P6:
        assert vman["views"][name]["sha256"] == p6man["views"][name]["sha256"]
    # the panel was built from the P6′ folder: its manifest names the v4p file
    pm = prepared["manifest"]
    assert pm["views_dir"] == str(p6p_dir)
    assert pm["derived_views"][FIN]["sha256"] == vman["views"][FIN]["sha256"]
    assert pm["derived_views"][FIN]["sha256"] != p6man["views"][FIN]["sha256"]
    assert pm["schema"] == p6_sp.SCHEMA and pm["mode"] == "serving" and pm["t"] == T.isoformat()
    # sidecar: same folder manifest, v4p hashes, the P6′ code
    sc = json.loads((sp.panel_dir(root, D) / sh.SIDECAR_FILE).read_text())
    assert sc == prepared["sidecar"]
    assert sc["schema"] == sh.SIDECAR_SCHEMA and sc["fin_file_holds"] == fq4p.FQMV_DLR_V4P_TABLE
    assert sc["views_manifest_sha256"] == pm["views_manifest_sha256"]
    assert sc["fin_sha256"] == pm["derived_views"][FIN]["sha256"]
    assert {"shadow.py", "shadow_prepare.py", "inputs.py"} <= set(sc["code"]["files"])
    assert set(prepared["views_seconds_by_step"]) == {"p6_views", "p6prime_views"}


def test_panel_equals_the_p6_panel_outside_the_financial_columns(prepared) -> None:
    a = pl.read_parquet(prepared["panel"])
    b = pl.read_parquet(prepared["p6"]["panel"])
    assert a.columns == b.columns and a.height == b.height > 0
    keys = ["trade_date", "ticker", "market"]
    a, b = a.sort(keys), b.sort(keys)
    non_fin = [c for c in a.columns if not c.startswith("fin_")]
    assert a.select(non_fin).equals(b.select(non_fin))


def test_views_are_reused_once_finished(prepared, monkeypatch) -> None:
    root = prepared["root"]
    calls: list[str] = []
    monkeypatch.setattr(p6_views, "build_views", lambda *a, **k: calls.append("p6"))
    monkeypatch.setattr(inputs, "build_p6prime_views", lambda *a, **k: calls.append("p6prime"))
    built = sp.ensure_views(D, prepared["raw"], root, memory_limit="2GB", threads=2)
    assert calls == [] and built["views_dir"] == sp.views_dir_for(root, D)


def test_prepare_refuses_the_p6_view_folder(prepared, tmp_path) -> None:
    root = tmp_path / "shadow_p6prime"
    with pytest.raises(sp.PrepareRefused, match="not a P6′ view folder"):
        sp.prepare_day(D, raw_root=prepared["raw"], views_dir=prepared["p6_vdir"],
                       out_root=root, sessions=SESSIONS)  # fmt: skip
    assert not root.exists()


def test_prepare_is_write_once(prepared) -> None:
    with pytest.raises(sp.PrepareRefused, match="already exists"):
        sp.prepare_day(D, raw_root=prepared["raw"], views_dir=sp.views_dir_for(prepared["root"], D),
                       out_root=prepared["root"], sessions=SESSIONS)  # fmt: skip


def test_prepare_never_writes_into_a_non_p6prime_root(prepared, tmp_path, capsys) -> None:
    cal = sessions_file(tmp_path / "cal.json")
    p6_like = tmp_path / "shadow"  # the P6 layout name
    argv = ["run", "--date", D.isoformat(), "--raw-lake", str(prepared["raw"].parent.parent),
            "--out-root", str(p6_like), "--sessions", str(cal)]  # fmt: skip
    assert sp.main(argv) == 2
    assert "not a P6′ path" in capsys.readouterr().err
    assert not p6_like.exists()
    with pytest.raises(sp.PrepareRefused):
        sp.prepare_day(D, raw_root=prepared["raw"], views_dir=sp.views_dir_for(prepared["root"], D),
                       out_root=p6_like, sessions=SESSIONS)  # fmt: skip
    assert not p6_like.exists()
    # the guard is on the resolved path: a ".." detour into the P6 root does not pass
    with pytest.raises(sh.P6RootRefused):
        sh.require_p6prime_root(tmp_path / "shadow_p6prime" / ".." / "shadow")


def test_missing_raw_snapshot_and_t_incomplete_keep_p6_exit_codes(prepared, tmp_path) -> None:
    cal = sessions_file(tmp_path / "cal.json")
    root = tmp_path / "shadow_p6prime"
    argv = ["run", "--date", "2025-01-02", "--raw-lake", str(prepared["raw"].parent.parent),
            "--out-root", str(root), "--sessions", str(cal)]  # fmt: skip
    assert sp.main(argv) == p6_sp.EXIT_RAW_MISSING
    assert sp.EXIT_T_INCOMPLETE == p6_sp.EXIT_T_INCOMPLETE == 33


# ----------------------------------------------------------------------------- score guards


def _score_argv(prepared, bundle: Path, out: Path, *extra: str) -> list[str]:
    cal = sessions_file(prepared["base"] / "cal.json")
    return ["score", "--date", D.isoformat(), "--panel", str(prepared["panel"]),
            "--panel-manifest", str(prepared["manifest_path"]), "--bundle-dir", str(bundle),
            "--out-root", str(out), "--ledger", str(out / "ledger.jsonl"), "--sessions", str(cal),
            *extra]  # fmt: skip


def test_score_refuses_a_p6_bundle(prepared, bundles, tmp_path, capsys) -> None:
    out = tmp_path / "shadow_p6prime" / "records"
    assert sh.main(_score_argv(prepared, bundles["p6"], out)) == 1
    assert "not a P6′ bundle" in capsys.readouterr().err
    assert not out.exists()


def test_score_refuses_a_non_p6prime_out_root(prepared, bundles, tmp_path, capsys) -> None:
    out = tmp_path / "shadow" / "records"
    assert sh.main(_score_argv(prepared, bundles["p6prime"], out)) == 1
    assert "not a P6′ path" in capsys.readouterr().err
    assert not (tmp_path / "shadow").exists()
    root = tmp_path / "shadow_p6prime" / "records"
    with pytest.raises(sh.P6RootRefused, match="not inside"):
        sh.score_day(D, bundle_dir=bundles["p6prime"], out_root=root,
                     ledger_path=tmp_path / "elsewhere" / "ledger.jsonl", sessions=SESSIONS,
                     panel=pl.read_parquet(prepared["panel"]), panel_manifest=prepared["manifest"],
                     sidecar=prepared["sidecar"], now=NOW)  # fmt: skip


def test_score_refuses_a_panel_without_the_p6prime_sidecar(prepared, bundles, tmp_path) -> None:
    out = tmp_path / "shadow_p6prime" / "records"
    kw = dict(bundle_dir=bundles["p6prime"], out_root=out, ledger_path=out / "ledger.jsonl",
              sessions=SESSIONS, now=NOW)  # fmt: skip
    p6_panel = pl.read_parquet(prepared["p6"]["panel"])
    with pytest.raises(sh.ShadowError, match="sidecar"):  # the P6 prepare writes no sidecar
        sh.score_day(D, panel=p6_panel, panel_manifest=prepared["p6"]["manifest"], sidecar=None,
                     **kw)  # fmt: skip
    with pytest.raises(sh.ShadowError, match="different view folders"):  # P6 manifest, P6′ sidecar
        sh.score_day(D, panel=p6_panel, panel_manifest=prepared["p6"]["manifest"],
                     sidecar=prepared["sidecar"], **kw)  # fmt: skip
    with pytest.raises(sh.ShadowError, match="panel manifest"):
        sh.score_day(D, panel=p6_panel, panel_manifest=None, sidecar=prepared["sidecar"], **kw)
    other_d = {**prepared["sidecar"], "D": "2025-01-02"}
    with pytest.raises(sh.ShadowError, match="not of D"):
        sh.score_day(D, panel=pl.read_parquet(prepared["panel"]),
                     panel_manifest=prepared["manifest"], sidecar=other_d, **kw)  # fmt: skip
    assert not out.exists()


# ------------------------------------------------------------------ record format = P6's


def test_record_and_ledger_have_the_p6_format(prepared, bundles, tmp_path) -> None:
    panel = pl.read_parquet(prepared["panel"])
    out = tmp_path / "shadow_p6prime" / "records"
    res = sh.score_day(D, bundle_dir=bundles["p6prime"], out_root=out,
                       ledger_path=out / "ledger.jsonl", sessions=SESSIONS, panel=panel,
                       panel_manifest=prepared["manifest"], sidecar=prepared["sidecar"],
                       now=NOW)  # fmt: skip
    ref_out = tmp_path / "p6ref"
    ref = p6_sh.score_day(D, bundle_dir=bundles["p6"], out_root=ref_out,
                          ledger_path=ref_out / "ledger.jsonl", sessions=SESSIONS, panel=panel,
                          panel_manifest=prepared["manifest"], now=NOW)  # fmt: skip
    rec, rrec = res["record"], ref["record"]
    assert set(rec) == set(rrec) and rec["schema_version"] == p6_sh.SCHEMA_RECORD
    assert set(res["ledger_line"]) == set(ref["ledger_line"])
    assert res["ledger_line"]["schema"] == p6_sh.SCHEMA_LEDGER
    assert rec["valid"] is True and rec["d0"] == D.isoformat() and rec["is_grid_date"] is True
    for k in ("counts", "files", "code", "lagged_cols", "feature_set", "t", "basket"):
        if k == "files":
            assert set(rec[k]) == set(rrec[k])
        elif k == "basket":
            assert (rec[k] is None) == (rrec[k] is None)
        else:
            assert rec[k] == rrec[k], k
    # the same model under two tags scores the same: same rows, ranks, tail flags, basket
    for f in (p6_sh.FILE_PREDICTIONS, p6_sh.FILE_BASKET):
        assert pl.read_parquet(res["dir"] / f).equals(pl.read_parquet(ref["dir"] / f))
    assert rec["bundle"]["prereg_tag"] == tags.PREREG_TAG
    assert rrec["bundle"]["prereg_tag"] == tags.P6_PREREG_TAG
    # panel_manifest: P6's manifest verbatim plus the one p6prime block
    pm = rec["panel_manifest"]
    assert {k: v for k, v in pm.items() if k != sh.PANEL_MANIFEST_KEY} == prepared["manifest"]
    assert pm[sh.PANEL_MANIFEST_KEY]["panel"] == prepared["sidecar"]
    score = pm[sh.PANEL_MANIFEST_KEY]["score"]
    assert score["bundle_tag"] == tags.BUNDLE_TAG and score["prereg_tag"] == tags.PREREG_TAG
    assert score["bundle_manifest_sha256"] == rec["bundle"]["manifest_sha256"]
    assert sh.verify_ledger(out / "ledger.jsonl", out)["ok"]
    assert p6_sh.verify_ledger(out / "ledger.jsonl", out)["ok"]  # P6's own check passes too


def test_cli_score_reads_the_sidecar_next_to_the_panel_and_verify(prepared, bundles, tmp_path,
                                                                   capsys) -> None:  # fmt: skip
    out = tmp_path / "shadow_p6prime" / "records"
    rc = sh.main(_score_argv(prepared, bundles["p6prime"], out))
    assert rc in (0, 3)  # 3 = written after the 15:00 deadline of a 2025 date: invalid, kept
    rec = json.loads((out / f"D={D.isoformat()}" / "record.json").read_text())
    assert rec["panel_manifest"]["p6prime"]["panel"] == prepared["sidecar"]
    capsys.readouterr()
    assert sh.main(["verify", "--ledger", str(out / "ledger.jsonl"), "--out-root", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["p6prime_breaks"] == []


def test_verify_flags_a_p6_record_in_a_p6prime_ledger(prepared, bundles, tmp_path) -> None:
    out = tmp_path / "shadow_p6prime" / "records"
    p6_sh.score_day(D, bundle_dir=bundles["p6"], out_root=out, ledger_path=out / "ledger.jsonl",
                    sessions=SESSIONS, panel=pl.read_parquet(prepared["panel"]),
                    panel_manifest=prepared["manifest"], now=NOW)  # fmt: skip
    rep = sh.verify_ledger(out / "ledger.jsonl", out)
    assert rep["ok"] is False and rep["breaks"] == []
    assert any("prereg_tag" in b for b in rep["p6prime_breaks"])
    assert any("v4p" in b for b in rep["p6prime_breaks"])
    with pytest.raises(sh.P6RootRefused):
        sh.verify_ledger(tmp_path / "shadow" / "ledger.jsonl")


def test_p6_code_is_called_not_copied() -> None:
    """The P6′ modules hold no scoring or panel logic of their own (wrapping only)."""
    src = (Path(sh.__file__).read_text(), Path(sp.__file__).read_text())
    for text in src:
        assert "rank_transform" not in text and "build_basket" not in text
        assert "build_kr_next_panel" not in text
    assert "p6.score_day(" in src[0] and "p6.prepare_day(" in src[1]


# ------------------------------------------------------------------------------- wrapper


def _p6prime_wrapper_env(tmp_path: Path, *, manifest: bool = True):
    env, p6_root, scored = p6_wrapper_env(tmp_path, ["2024-12-31"])
    root = tmp_path / "shadow_p6prime"
    panel = root / "panels_in" / "D=2025-01-02"
    panel.mkdir(parents=True)
    (panel / "panel.parquet").write_bytes(b"x")
    if manifest:
        (panel / "panel_manifest.json").write_text("{}")
    env["SHADOW_ROOT"] = str(root)
    return env, root, scored


def _run(env):
    return subprocess.run(["bash", str(WRAPPER), "--report-date", "2025-01-02"], env=env,
                          capture_output=True, text=True)  # fmt: skip


def test_wrapper_scores_with_the_p6prime_modules_and_its_own_lock(tmp_path) -> None:
    env, root, scored = _p6prime_wrapper_env(tmp_path)
    r = _run(env)
    assert r.returncode == 0, r.stderr
    args = scored.read_text()
    assert "modeler.models._04_kr_next_p6prime.shadow score" in args
    assert "--panel-manifest" in args and f"--out-root {root}/records" in args
    assert (root / "locks" / "kr-next-p6prime-shadow.lock").exists()
    assert not (tmp_path / "shadow" / "locks").exists()  # the P6 root of the fake env


def test_wrapper_refuses_a_root_that_is_not_p6prime(tmp_path) -> None:
    env, _, scored = _p6prime_wrapper_env(tmp_path)
    env["SHADOW_ROOT"] = str(tmp_path / "shadow")
    r = _run(env)
    assert r.returncode == 10 and "not a P6′ path" in r.stderr
    assert not scored.exists() and not (tmp_path / "shadow" / "locks").exists()


def test_wrapper_needs_the_panel_manifest(tmp_path) -> None:
    env, _, scored = _p6prime_wrapper_env(tmp_path, manifest=False)
    r = _run(env)
    assert r.returncode == 31 and "panel manifest missing" in r.stderr
    assert not scored.exists()


def test_wrapper_differs_from_the_p6_wrapper_only_where_intended() -> None:
    p6 = (REPO_ROOT / "deploy/prod/bin/kr-next-shadow.sh").read_text().splitlines()
    p6p = WRAPPER.read_text().splitlines()
    code = [x for x in p6p if x.strip() and not x.lstrip().startswith("#")]
    assert not [x for x in code if "_03_kr_next" in x]
    assert "SHADOW_ROOT=${SHADOW_ROOT:-/home/whi/apps/market-briefing/shadow_p6prime}" in code
    assert len(p6p) - len(p6) < 20  # a copy with a few lines added, not a rewrite


# ------------------------------------------------------------------------------- release


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("rev_p6prime") / "sources.sha256"
    path.write_text(sr.source_manifest_text(MODELER_ROOT, COLLECTOR_ROOT), encoding="utf-8")
    return path


def test_closure_is_the_p6_closure_plus_the_p6prime_code() -> None:
    files, data = sr.closure(MODELER_ROOT, COLLECTOR_ROOT)
    p6_files, p6_data = p6_sr.closure(MODELER_ROOT, COLLECTOR_ROOT)
    assert set(p6_files) <= set(files) and data == p6_data
    extra = set(files) - set(p6_files)
    for need in (
        "src/modeler/models/_04_kr_next_p6prime/__init__.py",
        "src/modeler/models/_04_kr_next_p6prime/shadow.py",
        "src/modeler/models/_04_kr_next_p6prime/shadow_prepare.py",
        "src/modeler/models/_04_kr_next_p6prime/shadow_release.py",
        "src/modeler/models/_04_kr_next_p6prime/inputs.py",
        "src/modeler/models/_04_kr_next_p6prime/tags.py",
        "src/modeler/etl/marts/financial_quarters_v4p.py",
    ):
        assert need in extra, need
    assert not [f for f in files if "/serving/" in f]


def test_release_built_with_the_p6prime_bundle_runs_from_its_own_dir(
    tmp_path, reviewed, bundles, prepared
) -> None:
    out = tmp_path / "rel"
    summary = sr.build_release(modeler_root=MODELER_ROOT, collector_root=COLLECTOR_ROOT,
                               source_manifest=reviewed, output=out,
                               bundle_dir=bundles["p6prime"])  # fmt: skip
    assert summary["bundle_verified"] is True
    meta = json.loads((out / "release.json").read_text())
    assert meta["schema_version"] == sr.SCHEMA and meta["model"] == sh.MODEL
    assert meta["entrypoint"] == sr.ENTRYPOINT and meta["bundle"]["prereg_tag"] == tags.PREREG_TAG
    # isolated import: the P6′ and P6 modules and the collector all come from the release
    iso = (
        "import sys, json\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import collector\n"
        "from modeler.models._04_kr_next_p6prime import shadow, shadow_prepare, shadow_release\n"
        "from modeler.models._03_kr_next import views\n"
        "from modeler.etl.marts import financial_quarters_v4p\n"
        "h, _ = views.load_holidays()\n"
        "mods = [m.__file__ for n, m in sys.modules.items()\n"
        "        if n.split('.')[0] in ('modeler', 'collector') and getattr(m, '__file__', None)]\n"
        "print(json.dumps({'mods': mods, 'n': len(h)}))\n"
    )
    res = subprocess.run([sys.executable, "-I", "-c", iso, str(out / "src")], cwd=tmp_path,
                         env={"PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                         text=True)  # fmt: skip
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got["mods"] and all(f.startswith(str(out / "src")) for f in got["mods"])
    assert got["n"] > 100
    # a day scored through the release's P6′ CLI with the release's bundle
    o = tmp_path / "shadow_p6prime" / "records"
    env = {**os.environ, "PYTHONPATH": str(out / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    cli = [sys.executable, "-m", "modeler.models._04_kr_next_p6prime.shadow",
           *_score_argv(prepared, out / "bundle", o)]  # fmt: skip
    res = subprocess.run(cli, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode in (0, 3), res.stderr
    rec = json.loads((o / f"D={D.isoformat()}" / "record.json").read_text())
    assert rec["bundle"]["manifest_sha256"] == meta["bundle"]["manifest_sha256"]
    shipped = {i["path"].rsplit("/", 1)[-1] for i in meta["files"]}
    assert set(rec["panel_manifest"]["p6prime"]["score"]["code"]["files"]) <= shipped
    # the P6 verifier refuses this release and the P6′ verifier refuses a P6 release
    with pytest.raises(p6_sr.ShadowReleaseError, match="schema"):
        p6_sr.verify_release(out)
    p6_rev = tmp_path / "p6.sha256"
    p6_rev.write_text(p6_sr.source_manifest_text(MODELER_ROOT, COLLECTOR_ROOT))
    p6_out = tmp_path / "p6rel"
    p6_sr.build_release(modeler_root=MODELER_ROOT, collector_root=COLLECTOR_ROOT,
                        source_manifest=p6_rev, output=p6_out)  # fmt: skip
    with pytest.raises(sr.ShadowReleaseError, match="not a P6′ shadow release"):
        sr.verify_release(p6_out)


def test_release_build_refuses_a_p6_bundle(tmp_path, reviewed, bundles) -> None:
    out = tmp_path / "rel"
    with pytest.raises(sr.ShadowReleaseError, match="not a P6′ bundle"):
        sr.build_release(modeler_root=MODELER_ROOT, collector_root=COLLECTOR_ROOT,
                         source_manifest=reviewed, output=out, bundle_dir=bundles["p6"])  # fmt: skip
    assert not out.exists() and not list(tmp_path.glob(".rel.build-*"))


def test_release_build_needs_every_file_reviewed(tmp_path, reviewed) -> None:
    lines = reviewed.read_text().splitlines()
    partial = tmp_path / "partial.sha256"
    partial.write_text("\n".join(x for x in lines if "_04_kr_next_p6prime/shadow.py" not in x))
    with pytest.raises(sr.ShadowReleaseError, match="not in the reviewed"):
        sr.build_release(modeler_root=MODELER_ROOT, collector_root=COLLECTOR_ROOT,
                         source_manifest=partial, output=tmp_path / "rel")  # fmt: skip
    assert not (tmp_path / "rel").exists()
