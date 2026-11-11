"""거르기(`rank_screen_enabled`) 단계: 스위치 모양, 꺼지면 아무것도 쓰지 않음, 실패해도 실행은 계속,
explain은 켠 날의 쓸 만한 screen 문서만 받음, 선별 행은 상위 N 근거와 따로 붙음(05 §6 C5)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import ms_fixtures as F
import numpy as np
import pytest
import test_explain as te
import test_summary_v2_switch as sw

from modeler.serving import daily_coordinator as dc
from modeler.serving import explain as ex
from modeler.serving import screen as sc

D = F.D
NOW = datetime.fromisoformat(F.RUN_AT)
KEY = "rank_screen_enabled"


def _set(config: Path, **changes) -> None:
    body = json.loads(config.read_text())
    body.update(changes)
    config.write_text(json.dumps(body, sort_keys=True))


def _wrapper(release: Path, config: Path, stage: str, now: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    return subprocess.run(
        [sys.executable, "-m", "modeler.serving.daily_wrapper", stage, "--config", str(config),
         "--report-date", D.isoformat(), "--fixture-now", now],
        cwd=release, env=env, text=True, capture_output=True, timeout=240)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    release, config, _data = F.build_release_world(tmp_path, monkeypatch)
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    return release, config


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / D.isoformat()


# --------------------------------------------------------------------------- switch
def test_config_switch_is_off_when_absent_and_a_bool_when_present(tmp_path: Path) -> None:
    config_path, body = sw._setup(tmp_path)
    body.pop(KEY, None)
    config_path.write_text(json.dumps(body))
    assert KEY not in dc._config(config_path)
    for bad in ("yes", 1, None):
        config_path.write_text(json.dumps({**body, KEY: bad}))
        with pytest.raises(ValueError, match="rank_screen_enabled must be true or false"):
            dc._config(config_path)
    config_path.write_text(json.dumps({**body, KEY: True}))
    assert dc._config(config_path)[KEY] is True


def test_publisher_input_carries_the_switch_only_when_on(tmp_path: Path) -> None:
    _config_path, base = sw._setup(tmp_path)
    render = {"report_sha256": "a" * 64, "invocation_id": "i"}
    assert dc._reports_publisher_input({**base, KEY: True}, sw.D1, render, tmp_path / "run")[KEY] is True
    for off in ({**base, KEY: False}, {k: v for k, v in base.items() if k != KEY}):
        assert KEY not in dc._reports_publisher_input(off, sw.D1, render, tmp_path / "run")


def test_publisher_before_screening_accepts_the_input_while_off(tmp_path: Path) -> None:
    old = sw._publisher_before(tmp_path)
    _config_path, base = sw._setup(tmp_path)
    render = {"report_sha256": "a" * 64, "invocation_id": "inv"}
    for off in ({**base, KEY: False}, {k: v for k, v in base.items() if k != KEY}):
        old.check_config(dc._reports_publisher_input(off, sw.D1, render, tmp_path / "run"),
                         need_local=True, strict_remote=False)


# --------------------------------------------------------------------------- stage
def test_off_touches_nothing_in_the_run_directory(tmp_path, world):
    release, config_path = world
    run = _run_dir(tmp_path)
    run.mkdir(parents=True, exist_ok=True)
    stale = run / f"screen-{D}.json"
    stale.write_text('{"invocation_id": "old"}')
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    assert "screen" not in json.loads(done.stdout)["stages"]
    assert stale.read_text() == '{"invocation_id": "old"}'  # not even set aside
    assert not (run / "coordinator-screen.json").exists() and not list(run.glob(".screen-*"))
    assert not list(run.glob(f"screen-{D}.superseded-*"))


def test_on_a_failing_screen_never_fails_the_run(tmp_path, world):
    release, config_path = world
    _set(config_path, rank_screen_enabled=True)
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    # the synthetic release has no raw inputs for the models: every model fails, the document says so
    assert stages["screen"] == "failed" and stages["render"] == "rendered"
    run = _run_dir(tmp_path)
    doc = json.loads((run / f"screen-{D}.json").read_text())
    report = run / f"report-{D}.json"
    assert doc["schema"] == sc.SCHEMA and doc["rule_version"] == "screen-v1"
    assert doc["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert set(doc["models"]) == set(sc.MODEL_MARKET)
    assert all(m["status"] == "failed" and m["reason"] for m in doc["models"].values())
    state = json.loads((run / "coordinator-screen.json").read_text())
    assert state["status"] == "failed" and state["invocation_id"] == doc["invocation_id"]


def test_explain_gets_the_screen_only_when_on_and_usable(tmp_path, world, monkeypatch):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    config = dc._config(config_path)
    infer = dc.infer_stage(config, D, NOW)
    run = _run_dir(tmp_path)
    seen = []
    orig = dc.run_group

    def spy(argv, **kw):
        seen.append(list(argv))
        return orig([sys.executable, "-c", "raise SystemExit(3)"], **kw)

    monkeypatch.setattr(dc, "run_group", spy)
    doc = run / f"screen-{D}.json"
    doc.write_bytes(b'{"x": 1}\n')
    good = {"report_date": D.isoformat(), "status": "ok", "invocation_id": infer["invocation_id"],
            "report_sha256": infer["report_sha256"], "document_sha256": hashlib.sha256(doc.read_bytes()).hexdigest()}
    (run / "coordinator-screen.json").write_text(json.dumps(good))
    dc.explain_stage(config, D, NOW)  # off: a usable document is still not passed
    assert "--screen" not in seen[-1]
    config = {**config, KEY: True}
    dc.explain_stage(config, D, NOW)
    assert seen[-1][seen[-1].index("--screen") + 1] == str(doc)
    assert seen[-1][seen[-1].index("--screen-sha256") + 1] == good["document_sha256"]
    for bad in ({"status": "failed"}, {"invocation_id": "other"}, {"document_sha256": "0" * 64}):
        (run / "coordinator-screen.json").write_text(json.dumps({**good, **bad}))
        dc.explain_stage(config, D, NOW)
        assert "--screen" not in seen[-1]


# --------------------------------------------------------------------------- explain rows (C5)
def test_screened_rows_are_explained_after_the_top_n_and_stay_out_of_the_mean():
    from sklearn.linear_model import Ridge

    X, y = te._world()
    est = Ridge().fit(X, y)
    plain, keys, order = te._section(est, X, top_n=5)
    extra = [keys[order[20]], keys[order[30]]]
    targets = [keys[i] for i in order[:5]] + extra
    scores = est.predict(X)
    wide = ex.explain_cross_section(
        predict=est.predict, X=X, design=te.DESIGN, features=list(te.FEATURES), feature_group=te.GROUP, keys=keys,
        symbols=keys, comparison=np.zeros(len(X), dtype=int), report_keys=[keys[i] for i in order],
        report_scores=[float(scores[i]) for i in order], names={f: f"label {f}" for f in te.FEATURES},
        scope_for=lambda _: "scope", top_n=5, target_keys=targets)
    assert [keys[i] for i in wide.top_index] == targets
    np.testing.assert_array_equal(wide.delta[:5], plain.delta)  # top N rows are exactly as before
    assert wide.top_feats[:5] == plain.top_feats

    def block(section, n_mean=None):
        top = [{"rank": order.index(i) + 1, "symbol": keys[i], "score": 0.5} for i in section.top_index]
        return ex._assemble(model_id="m", market="US", asof="2026-10-23", inputs={}, top=top,
                            group_ids=[g for g in ex.GID if g in set(te.GROUP.values())], section=section,
                            explain_error=None, prices=None, price_error="x", n_mean=n_mean)

    base, more = block(plain), block(wide, n_mean=5)
    assert more["top100_mean_delta"] == base["top100_mean_delta"]
    assert [r["rank"] for r in more["rows"]] == [1, 2, 3, 4, 5, 21, 31]
    assert all("screen_target" not in r for r in base["rows"] + more["rows"][:5])
    assert all(r["screen_target"] is True for r in more["rows"][5:])


def test_top_of_report_adds_only_screened_rows_below_the_top_n():
    model = {"rankings": [{"rank": k, "symbol": f"S{k}", "score": 1 - k / 100} for k in range(1, 50)]}
    assert [r["rank"] for r in ex._top_of_report(model, 3)] == [1, 2, 3]
    assert [r["rank"] for r in ex._top_of_report(model, 3, [2, 40, 7, 40, 99])] == [1, 2, 3, 7, 40]


def test_explain_cli_requires_the_screen_hash_with_the_screen(tmp_path):
    base = ["run", "--report-date", "2026-11-10", "--report", "r", "--report-sha256", "x", "--invocation-id", "i",
            "--release", "r", "--jobs-config", "j", "--jobs-config-sha256", "y", "--prepared-root", str(tmp_path),
            "--output", str(tmp_path / "o")]
    assert ex.main(base + ["--screen", str(tmp_path / "s")]) == 2
