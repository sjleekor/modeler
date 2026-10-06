"""score_daily: 합성 레이크·합성 bundle로 입력 고정·계산 달력·창 피쳐·문서를 시험한다.

동결 run과 같은 점수를 내는지(차이 1e-9 이하)는 실제 ``stock_data``가 필요해서 이 시험이 아니라
``python -m modeler.scores.market_sector.score_daily verify-frozen``이 본다.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from modeler.scores.common.calendar import SessionCalendar
from modeler.scores.market_sector import inputs_pin as ip
from modeler.scores.market_sector import score_daily as sd
from modeler.scores.market_sector.bundle import BundleError, verify_bundle_dir
from modeler.scores.market_sector.config import MsConfig

from . import ms_world as W

CUTOFF = datetime(2026, 10, 7, 9, 31, tzinfo=W.SEOUL)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> W.World:
    tmp = tmp_path_factory.mktemp("msworld")
    w = W.build_world(tmp)
    W.make_bundle(w, tmp / "bundle")
    return w


@pytest.fixture(scope="module")
def bundle_dir(world: W.World) -> Path:
    return world.root.parent / "bundle"


def _copy_world(world: W.World, tmp_path: Path) -> W.World:
    shutil.copytree(world.root, tmp_path / "world")
    shutil.copytree(world.root.parent / "bundle", tmp_path / "bundle")
    return W.World(tmp_path / "world", world.cfg, world.us_sessions, world.kr_sessions)


def _select(world: W.World, bundle: Path, out: Path, *, at: datetime = CUTOFF, mode="scheduled",
            limits=None) -> tuple[Path, dict]:
    selection = ip.select_market_sector_inputs(
        report_date=W.D, selected_at=at, mode=mode, kr_root=world.kr_root, us_root=world.us_root,
        limits=limits or {"KR": W.K, "US": W.A}, bundle_path=bundle / "bundle.json",
        bundle_sha256=ip.sha256_file(bundle / "bundle.json"))
    path = out / "ms-selection.json"
    path.write_text(json.dumps(selection, sort_keys=True))
    return path, selection


def _score(world: W.World, bundle: Path, tmp: Path, **kw) -> tuple[dict, Path]:
    sel_path, _ = _select(world, bundle, tmp, **kw)
    out = tmp / "market-sector.json"
    summary = sd.run_score(
        report_date=W.D, selection_path=sel_path, selection_sha256=None, bundle_path=bundle,
        bundle_sha256=ip.sha256_file(bundle / "bundle.json"), output=out)
    return summary, out


# --------------------------------------------------------------------------- bundle
def test_bundle_rejects_changed_extra_and_wrong_pin(world, bundle_dir, tmp_path):
    verify_bundle_dir(bundle_dir)
    copy = tmp_path / "b"
    shutil.copytree(bundle_dir, copy)
    model = copy / "us/models/live/p_opp_ridge.joblib"
    model.write_bytes(model.read_bytes() + b"x")
    with pytest.raises(BundleError, match="바뀌었습니다"):
        verify_bundle_dir(copy)
    copy2 = tmp_path / "b2"
    shutil.copytree(bundle_dir, copy2)
    (copy2 / "us/extra.txt").write_text("x")
    with pytest.raises(BundleError, match="없는 파일"):
        verify_bundle_dir(copy2)
    with pytest.raises(BundleError, match="고정된 값"):
        verify_bundle_dir(bundle_dir, expected_sha256="0" * 64)


def test_load_bundle_rejects_other_config(world, bundle_dir):
    other = MsConfig(ridge_alpha=11.0)
    with pytest.raises(BundleError, match="설정 해시"):
        sd.load_bundle(bundle_dir, cfg=other)


# --------------------------------------------------------------------------- 입력 고정
def test_select_pins_both_markets_with_hashes(world, bundle_dir, tmp_path):
    _, sel = _select(world, bundle_dir, tmp_path)
    assert ip.market_status(sel) == {"KR": {"status": "selected", "limit_session": "2026-10-06"},
                                     "US": {"status": "selected", "limit_session": "2026-10-06"}}
    kr = sel["kr"]
    assert kr["snapshot_date"] == W.KR_SNAP and kr["pg_snapshot_id"] == "00000017-00000127-1"
    assert kr["export_finished_at"] == "2026-10-07T04:29:15+09:00"
    for table in ip.KR_TABLES:
        assert len(kr["tables"][table]["manifest_sha256"]) == 64
        assert all(len(f["sha256"]) == 64 for f in kr["tables"][table]["files"].values())
    assert set(kr["tables"]["krx_index_daily"]["files"]) == {
        f"schema_version=1/index_group={g}/part-000000.parquet" for g in ("kosdaq", "kospi", "krx")}
    for table in ip.US_TABLES:
        rec = sel["us"]["tables"][table]
        assert rec["status"] == "selected" and rec["completed_at_basis"] == "latest_file_mtime"
        assert len(rec["files"]["part.parquet"]["sha256"]) == 64
    assert sel["input_cutoff"] == "2026-10-07T09:30:00+09:00"
    assert sel["bundle"]["sha256"] == ip.sha256_file(bundle_dir / "bundle.json")


def test_snapshot_completed_after_0930_is_not_chosen(world, bundle_dir, tmp_path):
    w = _copy_world(world, tmp_path)
    # 09:40에 끝난 KR export와 09:40에 쓰인 US prices snapshot이 더 새 날짜로 있다
    W.write_kr_snapshot(w.root, "2026-10-08", "2026-10-07T09:40:00+0900", w.kr_sessions, seed=5)
    import polars as pl_

    prices = pl_.read_parquet(w.us_table_file("prices_daily", W.US_SNAP))
    W.write_us_table(w.root, "prices_daily", "2026-10-07", prices,
                     datetime(2026, 10, 7, 9, 40, tzinfo=W.SEOUL))
    _, sel = _select(w, tmp_path / "bundle", tmp_path)
    assert sel["kr"]["snapshot_date"] == W.KR_SNAP
    assert [s["snapshot_date"] for s in sel["kr"]["skipped"]] == ["2026-10-08"]
    assert sel["kr"]["skipped"][0]["reason"] == "completed_after_cutoff"
    assert sel["us"]["tables"]["prices_daily"]["snapshot_date"] == W.US_SNAP
    assert sel["us"]["tables"]["prices_daily"]["skipped"][0]["snapshot_date"] == "2026-10-07"


def test_only_late_snapshot_means_unavailable(world, bundle_dir, tmp_path):
    w = _copy_world(world, tmp_path)
    marker = w.kr_snapshot_dir() / "_manifests" / "_SUCCESS.json"
    body = json.loads(marker.read_text())
    body["finished_at"] = "2026-10-07T09:40:00+0900"
    marker.write_text(json.dumps(body))
    W.set_mtime(w.us_table_file("prices_daily", W.US_SNAP),
                datetime(2026, 10, 7, 9, 41, tzinfo=W.SEOUL))
    _, sel = _select(w, tmp_path / "bundle", tmp_path)
    status = ip.market_status(sel)
    assert status["KR"] == {"status": "unavailable", "reason": ip.NO_SNAPSHOT_REASON}
    assert status["US"] == {"status": "unavailable", "reason": "us_prices_daily_unavailable"}
    summary, out = _score(w, tmp_path / "bundle", tmp_path)
    assert summary["status"] == "failed" and not out.exists()
    assert {m: f["reason"] for m, f in summary["failures"].items()} == {
        "US": "input_unavailable", "KR": "input_unavailable"}


def test_selection_before_0930_is_refused(world, bundle_dir, tmp_path):
    with pytest.raises(ip.InputPinError, match="09:30"):
        _select(world, bundle_dir, tmp_path, at=datetime(2026, 10, 7, 9, 29, tzinfo=W.SEOUL))


@pytest.mark.parametrize("market", ["US", "KR"])
def test_verify_pins_rejects_change_after_selection(world, bundle_dir, tmp_path, market):
    w = _copy_world(world, tmp_path)
    _, sel = _select(w, tmp_path / "bundle", tmp_path)
    ip.verify_pins(sel, market)
    if market == "US":
        victim = w.us_table_file("macro_series", W.MACRO_SNAP)
    else:
        victim = next((w.kr_snapshot_dir() / "krx_index_daily").rglob("*.parquet"))
    victim.write_bytes(victim.read_bytes() + b"0")
    with pytest.raises(ip.InputChangedError, match="바뀌었습니다"):
        ip.verify_pins(sel, market)


def test_verify_pins_rejects_added_and_removed_files(world, bundle_dir, tmp_path):
    w = _copy_world(world, tmp_path)
    _, sel = _select(w, tmp_path / "bundle", tmp_path)
    extra = w.us_table_file("trading_calendar", W.CAL_SNAP).with_name("late.parquet")
    shutil.copy(w.us_table_file("trading_calendar", W.CAL_SNAP), extra)
    with pytest.raises(ip.InputChangedError, match="목록"):
        ip.verify_pins(sel, "US")
    extra.unlink()
    ip.verify_pins(sel, "US")
    gone = next((w.kr_snapshot_dir() / "common_feature_observation_raw").rglob("*.parquet"))
    gone.unlink()
    with pytest.raises(ip.InputPinError):
        ip.verify_pins(sel, "KR")


@pytest.mark.parametrize("what", ["marker", "table_manifest"])
def test_verify_pins_rejects_changed_kr_manifests(world, bundle_dir, tmp_path, what):
    w = _copy_world(world, tmp_path)
    _, sel = _select(w, tmp_path / "bundle", tmp_path)
    src = w.kr_snapshot_dir() / "_manifests"
    target = src / "_SUCCESS.json"
    if what != "marker":
        target = src / "table_manifests" / "krx_index_daily.json"
    body = json.loads(target.read_text())
    body["note"] = "edited"
    target.write_text(json.dumps(body))
    with pytest.raises(ip.InputChangedError):
        ip.verify_pins(sel, "KR")


def test_selection_file_change_and_date_mismatch_are_refused(world, bundle_dir, tmp_path):
    sel_path, _ = _select(world, bundle_dir, tmp_path)
    sha = ip.sha256_file(sel_path)
    pin = ip.sha256_file(bundle_dir / "bundle.json")
    with pytest.raises(ip.InputChangedError, match="고정한 뒤에 바뀌었습니다"):
        sd.run_score(report_date=W.D, selection_path=sel_path, selection_sha256="0" * 64,
                     bundle_path=bundle_dir, bundle_sha256=pin, output=tmp_path / "o.json")
    with pytest.raises(ip.InputPinError, match="날짜"):
        sd.run_score(report_date=date(2026, 10, 8), selection_path=sel_path, selection_sha256=sha,
                     bundle_path=bundle_dir, bundle_sha256=pin, output=tmp_path / "o.json")
    with pytest.raises(BundleError, match="고정된 값"):
        sd.run_score(report_date=W.D, selection_path=sel_path, selection_sha256=sha,
                     bundle_path=bundle_dir, bundle_sha256="1" * 64, output=tmp_path / "o.json")


# --------------------------------------------------------------------------- 채점 문서
def _walk_keys(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path + "/" + str(key)
            yield from _walk_keys(value, path + "/" + str(key))
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item, path)


def test_document_shape_values_and_no_index_levels(world, bundle_dir, tmp_path):
    summary, out = _score(world, bundle_dir, tmp_path)
    assert summary["status"] == "ok" and summary["markets"] == {"US": "ok", "KR": "ok"}
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == "market-sector-daily.v1" and doc["report_date"] == "2026-10-07"
    assert doc["historical_replay"] is False and doc["failures"] == {}
    assert doc["asof"] == {"KR": "2026-10-02", "US": "2026-10-06", "US_macro": "2026-10-05"}
    # K=10-06, 지수는 10-02까지(10-05는 대체공휴일)
    assert doc["asof_detail"]["KR"]["lag_sessions"] == 1
    assert doc["asof_detail"]["US"]["lag_sessions"] == 0
    assets = doc["assets"]
    assert [a["market"] for a in assets] == ["US"] * 7 + ["KR"] * 7
    assert {a["return_basis"] for a in assets if a["market"] == "US"} == {"total_return"}
    assert {a["return_basis"] for a in assets if a["market"] == "KR"} == {"price_only"}
    for a in assets:
        for key in ("ret_1", "ret_5", "ret_20", "ret_60", "rvol_20", "dd_252", "cash_rate",
                    "b_opp_mean_pct", "b_stab_prob_pct", "opportunity_score", "stability_score"):
            assert isinstance(a[key], float), (a["asset_id"], key)
        assert 0.0 <= a["opportunity_score"] <= 100.0 and a["dd_252"] <= 0.0
        assert a["cash_status"] == "ok" and a["name"]
    # 지수 종가 수준은 어디에도 없다(Q1)
    banned = ("close", "tr_index", "px_", "level")
    assert not [k for k in _walk_keys(doc) if any(b in k.lower() for b in banned)]
    assert doc["verdicts"]["US"] == {
        "opportunity": "실패", "sector_relative": "보류", "stability": "실패"}
    assert doc["verdicts"]["KR"] == {
        "opportunity": "실패", "sector_relative": "실패", "stability": "실패"}
    assert doc["provenance"]["calendar_basis"]["KR"] == sd.xcals_basis()
    assert doc["provenance"]["selection"]["selection_mode"] == "scheduled"
    assert doc["provenance"]["window_check"]["US"]["max_abs_diff"] <= sd.WINDOW_TOLERANCE
    assert doc["status"] == "ok"


def test_document_is_byte_identical_on_rerun(world, bundle_dir, tmp_path):
    _, first = _score(world, bundle_dir, tmp_path)
    other = tmp_path / "again"
    other.mkdir()
    _, second = _score(world, bundle_dir, other)
    assert first.read_bytes() == second.read_bytes()


def test_kr_lag_beyond_nominal_marks_the_section_stale(tmp_path):
    w = W.build_world(tmp_path, kr_last=date(2026, 9, 28))
    W.make_bundle(w, tmp_path / "bundle")
    summary, out = _score(w, tmp_path / "bundle", tmp_path)
    doc = json.loads(out.read_text())
    assert summary["status"] == "stale" and doc["status"] == "stale"
    assert doc["asof_detail"]["KR"]["lag_sessions"] == 5
    assert any("5세션 늦습니다" in n for n in doc["notes"])


def test_stability_null_reason_from_the_frozen_run_is_kept(tmp_path):
    w = W.build_world(tmp_path)
    W.make_bundle(w, tmp_path / "bundle", null_stability_asset="us_ind")
    _, out = _score(w, tmp_path / "bundle", tmp_path)
    doc = json.loads(out.read_text())
    row = next(a for a in doc["assets"] if a["asset_id"] == "us_ind")
    assert row["stability_score"] is None and row["b_stab_prob_pct"] is None
    assert isinstance(row["opportunity_score"], float)
    assert any("us_ind: Stability null (events_in_train<30)" in n for n in doc["notes"])


# --------------------------------------------------------------------------- 창 피쳐
def _frames(world, bundle_dir, market="US"):
    bundle = sd.load_bundle(bundle_dir)
    _, sel = _select(world, bundle_dir, bundle_dir.parent)
    return sd.build_frames(market, sel, bundle.markets[market], bundle.cfg), bundle


@pytest.mark.parametrize("market", ["US", "KR"])
def test_window_features_equal_full_panel_features(world, bundle_dir, market):
    frames, bundle = _frames(world, bundle_dir, market)
    cfg = bundle.cfg
    limit = W.A if market == "US" else W.K
    full = sd.full_price_frame(frames, cfg)
    feats_w = sd.window_features(frames, limit, cfg)
    diff = sd.window_vs_full_diff(feats_w, full, last_n=sd.WINDOW_MARGIN_SESSIONS)
    assert diff["rows"] == 7 * sd.WINDOW_MARGIN_SESSIONS
    assert diff["max_abs_diff"] <= sd.WINDOW_TOLERANCE
    assert diff["null_mismatch"] == 0 and diff["feature_ready_mismatch"] == 0
    # 창은 전체 이력보다 훨씬 짧다
    per_asset = feats_w.group_by("asset_id").len()["len"].unique().to_list()
    assert per_asset == [cfg.max_lookback_sessions + 1 + sd.WINDOW_MARGIN_SESSIONS]
    market_asset = "us_spx" if market == "US" else "kr_kospi"
    assert full.filter(pl.col("asset_id") == market_asset).height > per_asset[0]


def test_a_window_shorter_than_the_lookback_is_refused(world, bundle_dir, monkeypatch):
    frames, bundle = _frames(world, bundle_dir)
    monkeypatch.setattr(sd, "WINDOW_MARGIN_SESSIONS", -150)
    with pytest.raises(sd.WindowMismatchError):
        sd.score_market(frames, bundle.markets["US"], bundle.cfg, W.A)


def test_b_opp_mean_is_the_pit_mean_over_the_whole_history(world, bundle_dir):
    frames, bundle = _frames(world, bundle_dir)
    cfg = bundle.cfg
    res = sd.score_market(frames, bundle.markets["US"], cfg, W.A)
    full = sd.full_price_frame(frames, cfg)
    for aid in ("us_spx", "us_tech"):
        rows = full.filter(pl.col("asset_id") == aid).sort("session")
        last = rows.row(rows.height - 1, named=True)
        pool = rows.filter(
            pl.col("label_matured") & pl.col("feature_ready")
            & (pl.col("session") >= cfg.train_start["US"])
            & (pl.col("label_end_at") < last["decision_at"])
            & pl.col("excess_return_60d_vs_cash").is_not_null())
        assert pool.height > 100
        expected = pool["excess_return_60d_vs_cash"].mean()
        assert res.raw[aid]["b_opp_mean"] == pytest.approx(expected, abs=1e-12)
    # 성숙한 라벨만 쓴다: 마지막 60세션의 라벨은 아직 만기되지 않았다
    assert full.filter(~pl.col("label_matured")).height >= 60


# --------------------------------------------------------------------------- 계산 달력
def _kr_spec(bundle_dir):
    return sd.load_bundle(bundle_dir).markets["KR"].calendar


def test_observed_price_date_calendar_is_rejected(world, bundle_dir):
    spec = _kr_spec(bundle_dir)
    observed = SessionCalendar.from_observed_dates("XKRX", world.kr_sessions)
    with pytest.raises(sd.CalendarMismatchError, match="observed_price_dates"):
        sd.verify_calendar(observed, spec)


def test_calendar_version_and_session_hash_must_match(world, bundle_dir):
    spec = _kr_spec(bundle_dir)
    cal = sd.kr_computation_calendar(spec, world.kr_sessions[-1])
    assert sd.verify_calendar(cal, spec)["frozen_sessions"] == spec["n_sessions"]
    with pytest.raises(sd.CalendarMismatchError, match="exchange_calendars==0.0.1"):
        sd.verify_calendar(cal, {**spec, "basis": "exchange_calendars==0.0.1"})
    with pytest.raises(sd.CalendarMismatchError, match="세션 목록이 동결 때와 다릅니다"):
        sd.verify_calendar(cal, {**spec, "sessions_sha256": "0" * 64})
    # 동결 구간 안에서 세션이 하나 빠지거나 늘어도 거부한다
    dropped = cal.sessions[300]
    shifted = SessionCalendar.from_sessions(
        "XKRX", [s for s in cal.sessions if s != dropped], calendar_basis=cal.calendar_basis)
    with pytest.raises(sd.CalendarMismatchError):
        sd.verify_calendar(shifted, spec)


def test_kr_scoring_without_exchange_calendars_is_refused_and_us_still_scores(
        world, bundle_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(SessionCalendar, "from_exchange_calendars",
                        classmethod(lambda cls, *a, **k: None))
    summary, out = _score(world, bundle_dir, tmp_path)
    assert summary["markets"] == {"US": "ok", "KR": "calendar_mismatch"}
    doc = json.loads(out.read_text())
    assert doc["status"] == "partial"
    assert doc["failures"] == {"KR": {"error_class": "CalendarMismatchError",
                                      "reason": "calendar_mismatch"}}
    assert {a["market"] for a in doc["assets"]} == {"US"}


def test_us_calendar_changed_since_the_freeze_is_refused(world, bundle_dir, tmp_path):
    w = _copy_world(world, tmp_path)
    cal = pl.read_parquet(w.us_table_file("trading_calendar", W.CAL_SNAP))
    dropped = cal.filter(pl.col("date") != w.us_sessions[300])
    W.write_us_table(w.root, "trading_calendar", "2026-09-23", dropped,
                     datetime(2026, 9, 23, 0, 20, tzinfo=W.SEOUL))
    summary, out = _score(w, tmp_path / "bundle", tmp_path)
    assert summary["markets"]["US"] == "calendar_mismatch" and summary["markets"]["KR"] == "ok"
    assert json.loads(out.read_text())["status"] == "partial"


# --------------------------------------------------------------------------- 선택 뒤 변경
def test_input_changed_after_selection_fails_that_market_only(world, bundle_dir, tmp_path):
    w = _copy_world(world, tmp_path)
    sel_path, _ = _select(w, tmp_path / "bundle", tmp_path)
    prices = w.us_table_file("prices_daily", W.US_SNAP)
    frame = pl.read_parquet(prices)
    frame.with_columns(pl.col("close") * 1.01).write_parquet(prices)  # 같은 자리에서 값이 바뀜
    out = tmp_path / "doc.json"
    summary = sd.run_score(
        report_date=W.D, selection_path=sel_path, selection_sha256=None,
        bundle_path=tmp_path / "bundle", bundle_sha256=None, output=out)
    assert summary["markets"] == {"US": "input_changed", "KR": "ok"}
    doc = json.loads(out.read_text())
    assert doc["status"] == "partial" and {a["market"] for a in doc["assets"]} == {"KR"}
    assert doc["failures"]["US"]["error_class"] == "InputChangedError"


def test_cli_exit_codes(world, bundle_dir, tmp_path, capsys):
    w = _copy_world(world, tmp_path)
    sel_path, _ = _select(w, tmp_path / "bundle", tmp_path)
    out = tmp_path / "doc.json"
    argv = ["score", "--report-date", "2026-10-07", "--selection", str(sel_path), "--bundle",
            str(tmp_path / "bundle"), "--output", str(out)]
    assert sd.main(argv) == 0 and out.is_file()
    out.unlink()
    for victim in (w.us_table_file("macro_series", W.MACRO_SNAP),
                   next((w.kr_snapshot_dir() / "krx_index_daily").rglob("*.parquet"))):
        victim.write_bytes(victim.read_bytes() + b"0")
    assert sd.main(argv) == 1 and not out.exists()  # 두 시장 모두 실패: 문서 없음
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["status"] == "failed" and set(summary["failures"]) == {"US", "KR"}
    assert sd.main([*argv[:-1], str(out), "--selection-sha256", "0" * 64]) == 2


# --------------------------------------------------------------------------- 실제 동결 run (선택)
@pytest.mark.skipif(not os.environ.get("MS_FROZEN_ROOT"),
                    reason="MS_FROZEN_ROOT(동결 run이 있는 stock_data)가 있을 때만 돕니다")
def test_frozen_run_is_reproduced_from_the_frozen_inputs(tmp_path):
    """동결 run과 같은 입력 snapshot으로 채점한 점수가 latest_scores.json과 1e-9 안에서 같다."""
    root = Path(os.environ["MS_FROZEN_ROOT"])
    built = sd.build_bundle(root, tmp_path / "bundle")
    report = sd.verify_frozen(root, tmp_path / "bundle")
    assert built["files"] == 12
    for market in ("US", "KR"):
        assert report[market]["max_abs_diff"] <= 1e-9, market
        assert report[market]["window_pass"], market
    assert report["overall_pass"]
