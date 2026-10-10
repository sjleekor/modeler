"""MRS run.py 시험: 권한 확인(거부 경로)과 가짜 레이크 end-to-end (합성 데이터만)."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date

import polars as pl
import pytest

from modeler.scores.mrs import backtest as bt
from modeler.scores.mrs import config
from modeler.scores.mrs import run as mrs_run
from modeler.scores.mrs import score as mrs_score

from .fake_lakes import (
    KR_SNAP,
    US_SNAP,
    write_kr_lake,
    write_kr_lake_long,
    write_us_lake,
    write_us_lake_long,
)

TODAY = date(2026, 10, 11)


def _no_compute(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("compute_scores가 불렸다")

    monkeypatch.setattr(mrs_run, "compute_scores", boom)
    monkeypatch.setattr(mrs_score, "compute_scores", boom)


def _env(root, monkeypatch, *, approve=True):
    """env·오늘·git을 맞추고 승인 파일을 만든다. ``(승인 파일 경로, sha)``."""
    monkeypatch.setenv("STOCK_DATA_ROOT", str(root))
    interp = root / "interp.md"
    interp.write_text("구현 해석 표 (합성 시험)")
    sha = hashlib.sha256(interp.read_bytes()).hexdigest()
    monkeypatch.setattr(mrs_run, "APPROVED_INTERP_SHA256", sha if approve else None)
    monkeypatch.setattr(mrs_run, "_today_kst", lambda: TODAY)
    monkeypatch.setattr(mrs_run, "git_commit", lambda *a, **k: "abc123")
    return interp, sha


def _argv(interp, *, confirm=TODAY.isoformat(), market="all"):
    return [
        "run", "--market", market, "--kr-snapshot", KR_SNAP,
        "--approved-interp", str(interp), "--confirm-run", confirm,
    ]  # fmt: skip


def _assert_nothing_created(root):
    for m in ("kr", "us"):
        assert not (root / m / "output" / "regime_score").exists()


# --------------------------------------------------------------------------- 거부 경로
def test_refused_when_constant_is_none(tmp_path, monkeypatch, caplog):
    interp, _ = _env(tmp_path, monkeypatch, approve=False)
    write_kr_lake(tmp_path)
    _no_compute(monkeypatch)
    assert mrs_run.main(_argv(interp)) == 2
    assert "APPROVED_INTERP_SHA256이 None" in caplog.text
    _assert_nothing_created(tmp_path)


def test_refused_when_sha_mismatch(tmp_path, monkeypatch):
    interp, _ = _env(tmp_path, monkeypatch)
    interp.write_text("바뀐 파일")
    _no_compute(monkeypatch)
    assert mrs_run.main(_argv(interp)) == 2
    _assert_nothing_created(tmp_path)


def test_refused_when_confirm_date_is_not_today(tmp_path, monkeypatch, caplog):
    interp, _ = _env(tmp_path, monkeypatch)
    _no_compute(monkeypatch)
    assert mrs_run.main(_argv(interp, confirm="2026-10-10")) == 2
    assert "오늘(KST)" in caplog.text
    _assert_nothing_created(tmp_path)


def test_refused_when_backfill_fails(tmp_path, monkeypatch, caplog):
    interp, _ = _env(tmp_path, monkeypatch)
    write_kr_lake(tmp_path)  # 1996에 시작 -> 백필 FAIL
    write_us_lake(tmp_path)
    _no_compute(monkeypatch)
    assert mrs_run.main(_argv(interp)) == 2
    assert "선행 백필 확인 FAIL" in caplog.text
    _assert_nothing_created(tmp_path)


def test_refused_when_tree_dirty(tmp_path, monkeypatch):
    interp, _ = _env(tmp_path, monkeypatch)

    def dirty(*a, **k):
        raise mrs_run.DirtyWorktreeError("dirty\n M x.py")

    monkeypatch.setattr(mrs_run, "git_commit", dirty)
    _no_compute(monkeypatch)
    assert mrs_run.main(_argv(interp)) == 2
    _assert_nothing_created(tmp_path)


def test_real_cli_is_refused_without_approval(tmp_path, monkeypatch):
    """상수가 None인 기본 상태: 어떤 인자로도 거부된다."""
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    assert mrs_run.APPROVED_INTERP_SHA256 is None
    assert mrs_run.main(["run", "--market", "all", "--kr-snapshot", KR_SNAP]) == 2
    _assert_nothing_created(tmp_path)


# --------------------------------------------------------------------------- e2e
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("mrs_e2e")
    mp = pytest.MonkeyPatch()
    interp, _ = _env(root, mp)
    write_kr_lake_long(root)
    write_us_lake_long(root)
    t = time.perf_counter()
    code = mrs_run.main(_argv(interp))
    wall = time.perf_counter() - t
    yield {
        "root": root,
        "interp": interp,
        "code": code,
        "wall": wall,
        "kr": root / "kr" / "output" / "regime_score" / KR_SNAP,
        "us": root / "us" / "output" / "regime_score" / US_SNAP,
    }
    mp.undo()


FILES_KR = {"scores.parquet", "first_dates.json", "ledger.parquet", "tests.parquet",
            "tests.json", "synth.parquet", "manifest.json", "report.md"}  # fmt: skip


def test_e2e_files_written(e2e):
    print(f"e2e wall {e2e['wall']:.1f}s")
    assert e2e["code"] == 0
    assert {p.name for p in e2e["kr"].iterdir()} == FILES_KR
    assert {p.name for p in e2e["us"].iterdir()} == FILES_KR - {"synth.parquet"}


def test_e2e_scores_available_and_protocols(e2e):
    sc = pl.read_parquet(e2e["kr"] / "scores.parquet")
    assert set(sc["score_protocol"].unique()) == {"main", "vix_proxy"}
    assert sc["availability_ok"].all()
    assert sc.filter(pl.col("score_protocol") == "main")["MRS"].drop_nulls().len() > 1000
    us = pl.read_parquet(e2e["us"] / "scores.parquet")
    assert set(us["score_protocol"].unique()) == {"main"} and us["availability_ok"].all()
    fd = json.loads((e2e["kr"] / "first_dates.json").read_text())
    assert set(fd) == {"main", "vix_proxy"}
    assert {r["component"] for r in fd["main"]} == set(config.KR_COMPONENTS)


def test_e2e_tests_combinations(e2e):
    kr = pl.read_parquet(e2e["kr"] / "tests.parquet")
    combos = {(r["period"], r["asset"], r["protocol"], r["rule"]) for r in kr.iter_rows(named=True)}
    protos = [config.P_MAIN, config.P_CASH0, config.P_COST120, config.P_VIX_PROXY, config.P_IRP]
    expect = set()
    for rule in bt.RULES:
        for p in [*protos, config.P_KTB_SYNTH]:
            expect.add(("main", "kr_kospi", p, rule))
        for a in ("kr_kospi", "kr_kosdaq"):
            for p in protos:
                expect.add(("explore", a, p, rule))
    assert combos == expect
    assert config.P_ORIG_OPEN not in set(kr["protocol"])
    us = pl.read_parquet(e2e["us"] / "tests.parquet")
    assert {(r["protocol"], r["rule"]) for r in us.iter_rows(named=True)} == {
        (p, r) for p in (config.P_MAIN, config.P_CASH0) for r in bt.RULES
    }
    assert set(us["period"]) == {"explore"} and set(us["market"]) == {"US"}


def test_e2e_synth_rows_have_no_gates_and_grade_only_on_official_row(e2e):
    kr = pl.read_parquet(e2e["kr"] / "tests.parquet")
    syn = kr.filter(pl.col("protocol") == config.P_KTB_SYNTH)
    assert syn.height == 3 and syn["synthetic"].all()
    for c in ("g1", "g2", "g3", "g4", "g5", "gate_class", "placebo_p", "grade"):
        assert syn[c].null_count() == syn.height
    assert not kr.filter(pl.col("protocol") != config.P_KTB_SYNTH)["synthetic"].any()
    graded = kr.filter(pl.col("grade").is_not_null())
    assert graded.height == 1
    row = graded.row(0, named=True)
    assert (row["period"], row["asset"], row["protocol"], row["rule"]) == (
        "main", "kr_kospi", config.P_MAIN, "mrs",
    )  # fmt: skip
    assert row["grade"] in set("ABCDRSX")
    us = pl.read_parquet(e2e["us"] / "tests.parquet")
    assert us["grade"].null_count() == us.height
    # 주 판정 mrs 행은 placebo까지 계산됐다 (세션 > 1000)
    assert row["placebo_p"] is not None and row["n_sessions"] > 1000


def test_e2e_synth_tables_and_ledger(e2e):
    syn = pl.read_parquet(e2e["kr"] / "synth.parquet")
    assert set(syn["table"]) == {"mix_synth_ps1", "mix_synth_ps2"}
    assert set(syn["series"]) == {"mix", "buy_hold", "mrs_main_close_cash"}
    ps2 = syn.filter(pl.col("table") == "mix_synth_ps2")
    assert ps2["first_session"].min() >= date(2001, 1, 1)
    assert syn.filter(pl.col("series") == "mix")["synthetic"].all()
    led = pl.read_parquet(e2e["kr"] / "ledger.parquet")
    assert {"market", "asset", "protocol", "date", "r_mrs", "a_mrs", "w_vm"} <= set(led.columns)
    assert set(led["protocol"]) == {
        config.P_MAIN, config.P_CASH0, config.P_COST120, config.P_VIX_PROXY, config.P_IRP,
        config.P_KTB_SYNTH,
    }  # fmt: skip


def test_e2e_manifest_and_report(e2e):
    man = json.loads((e2e["kr"] / "manifest.json").read_text())
    assert man["placebo_shifts"] == bt.placebo_shifts() and len(man["placebo_shifts"]) == 50
    assert man["protocols_not_run"][config.P_ORIG_OPEN].startswith("이 구간에 정의하지 않음")
    assert config.P_KTB_SYNTH in man["protocols_run"]
    assert man["modeler_git_commit"] == "abc123"
    assert man["approved_interp"]["sha256"] == mrs_run.APPROVED_INTERP_SHA256
    assert man["config"]["WARMUP_VALID_OBS"] == 1260
    assert man["official_grade"] in set("ABCDRSX")
    assert man["snapshot"]["common_feature_observation_raw"] == KR_SNAP
    assert all(f["sha256"] for f in man["input_files"])
    assert set(man["engine_diagnostics"]) == {"main", "vix_proxy"}
    assert man["wall_time_sec"] > 0
    rep = (e2e["kr"] / "report.md").read_text()
    assert "합성(추정)" in rep and mrs_run.SYNTH_LIMITATION in rep
    assert "공식 등급" in rep
    usman = json.loads((e2e["us"] / "manifest.json").read_text())
    assert usman["protocols_not_run"][config.P_COST120].startswith("KR만")
    assert "합성(추정)" not in (e2e["us"] / "report.md").read_text()


def test_e2e_rerun_into_existing_dir_is_refused(e2e, monkeypatch, tmp_path):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(e2e["root"]))
    monkeypatch.setattr(mrs_run, "APPROVED_INTERP_SHA256",
                        hashlib.sha256(e2e["interp"].read_bytes()).hexdigest())  # fmt: skip
    monkeypatch.setattr(mrs_run, "_today_kst", lambda: TODAY)
    monkeypatch.setattr(mrs_run, "git_commit", lambda *a, **k: "abc123")
    before = (e2e["kr"] / "manifest.json").read_text()
    _no_compute(monkeypatch)
    assert mrs_run.main(_argv(e2e["interp"])) == 2
    assert (e2e["kr"] / "manifest.json").read_text() == before


def test_single_market_run_leaves_grade_null(tmp_path, monkeypatch):
    """--market us 단독: 공식 등급을 정하지 않고 보고서가 이유를 적는다."""
    interp, _ = _env(tmp_path, monkeypatch)
    write_us_lake_long(tmp_path)
    assert mrs_run.main(_argv(interp, market="us")) == 0
    out = tmp_path / "us" / "output" / "regime_score" / US_SNAP
    assert not (tmp_path / "kr" / "output").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["official_grade"] is None
    assert "공식 등급 없음" in (out / "report.md").read_text()
