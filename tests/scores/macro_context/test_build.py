"""build_document: 합성 한국 raw snapshot·미국 macro_series로 문서 전체를 시험한다."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from modeler.scores import macro_context as mc
from modeler.scores.market_sector import inputs_pin as ip

from . import _world as W

UNIT = W.D.isoformat()
KW = dict(unit=UNIT, invocation_id="inv-1", release="v1.2.3", selection_sha256="ab" * 32)


def build(world: W.World, selection: dict | None = None, **over) -> dict:
    return mc.build_document(selection or W.make_selection(world), **{**KW, **over})


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> W.World:
    return W.build_world(tmp_path_factory.mktemp("macro"))


@pytest.fixture(scope="module")
def body(world) -> dict:
    return build(world)


def by_id(body: dict) -> dict:
    return {o["id"]: o for o in body["indicators"]}


# --------------------------------------------------------------------------- 모양
def test_body_shape_and_echoed_arguments(body):
    assert list(sorted(body)) == sorted(
        [
            "schema",
            "state",
            "unit",
            "invocation_id",
            "release",
            "selection_sha256",
            "inputs",
            "header",
            "indicators",
            "summary",
            "vs_prev",
            "figures",
            "checks",
            "rules",
        ]
    )
    assert body["schema"] == mc.SCHEMA == "briefing-macro.v1" and body["state"] == "ok"
    assert (body["unit"], body["invocation_id"], body["release"]) == (UNIT, "inv-1", "v1.2.3")
    assert body["selection_sha256"] == "ab" * 32
    assert len(body["indicators"]) == 31 and len(body["figures"]) == 21
    assert [o["id"] for o in body["indicators"]][:3] == ["us_3m", "us_2y", "us_10y"]
    assert sum(o["kind"] == "monthly" for o in body["indicators"]) == 12
    assert all("transition" not in o for o in body["indicators"])
    assert all(o["short_name"] and o["display_name"] for o in body["indicators"])
    assert body["checks"]["overall_pass"] is True and body["checks"]["violations"] == []
    assert 4 <= len(body["summary"]["rows"]) <= 6
    assert body["vs_prev"]["previous_report_date"] == "2026-10-27"
    assert body["rules"]["flat_quantile_pct"] == 30 and body["rules"]["strong_quantile_pct"] == 90
    assert len(body["rules"]["periods"]) == 31
    json.dumps(body, allow_nan=False)  # NaN·numpy 값이 없다


def test_figures_are_renderable_from_body_alone(body):
    for f in body["figures"]:
        assert {"name", "group", "title", "kind", "unit", "ref", "shade_from", "lines"} <= set(f)
        assert f["kind"] in ("daily", "monthly") and f["ref"] in (0, 1, 100, None)
        for ln in f["lines"]:
            assert {"label", "points", "last", "segments", "fmt"} <= set(ln)
            pts = ln["points"]
            assert ln["last"] == {"date": pts[-1][0], "value": pts[-1][1]}
            dates = [p[0] for p in pts]
            assert dates == sorted(dates) and len(set(dates)) == len(dates)
            assert all(round(p[1], 4) == p[1] for p in pts)
            assert ln["segments"][0][0] == 0 and ln["segments"][-1][1] == len(pts)
        if f["kind"] == "daily":
            assert f["shade_from"] and all(200 <= len(ln["points"]) <= 262 for ln in f["lines"])
        else:
            assert f["shade_from"] is None and all(len(ln["points"]) <= 60 for ln in f["lines"])
    names = [f["name"] for f in body["figures"]]
    assert len(set(names)) == 21 and "risk_credit" in names and "flow_trdval_ratio" in names


def test_deterministic_bytes_and_selection_not_mutated(world):
    sel = W.make_selection(world)
    before = copy.deepcopy(sel)
    a = mc.document_bytes(build(world, sel))
    b = mc.document_bytes(build(world, sel))
    assert a == b and a.endswith(b"}\n") and sel == before
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_inputs_record_pins_and_relative_paths(body, world):
    kr = body["inputs"]["kr"]
    assert kr["snapshot_date"] == W.KR_SNAP and kr["table"] == "common_feature_observation_raw"
    assert kr["dir"].startswith("raw/raw_postgres/snapshot_date=2026-10-28/source=sj2_remote/")
    rel, sha = next(iter(kr["files"].items()))
    assert (
        sha
        == hashlib.sha256(
            (world.kr_dir / "common_feature_observation_raw" / rel).read_bytes()
        ).hexdigest()
    )
    us = body["inputs"]["us"]
    assert (
        us["snapshot_date"] == W.US_SNAP
        and us["files"]["part.parquet"] == hashlib.sha256(world.us_file.read_bytes()).hexdigest()
    )
    assert str(world.root) not in json.dumps(body["inputs"])  # 절대 경로는 싣지 않는다


# --------------------------------------------------------------------------- 표시 정책
def test_credit_level_hidden_and_trading_value_ratio_only(body):
    ind = by_id(body)
    assert ind["credit_baa"]["display"] == {"level": "hide"}
    assert all(o["display"] == {"level": "show"} for i, o in ind.items() if i != "credit_baa")
    credit_fig = next(f for f in body["figures"] if f["name"] == "risk_credit")
    ln = credit_fig["lines"][0]
    assert credit_fig["ref"] == 0 and ln["points"][0][1] == 0.0  # 1년 창 시작 대비 변화
    for tv in ("tv_kospi", "tv_kosdaq"):
        assert set(ind[tv]["value"]) == {"level", "display"} and ind[tv]["value"][
            "display"
        ].endswith("배")
    text = json.dumps({k: v for k, v in body.items() if k != "rules"}, ensure_ascii=False)
    assert "avg20" not in text and "avg250" not in text and "평균 조원" not in text
    ratio = next(f for f in body["figures"] if f["name"] == "flow_trdval_ratio")
    assert (
        ratio["unit"] == "배"
        and ratio["ref"] == 1
        and [ln["label"] for ln in ratio["lines"]] == ["KOSPI", "KOSDAQ"]
    )


# --------------------------------------------------------------------------- 사본 이음
def test_copy_splice_with_and_without_copy_series(body):
    ind = by_id(body)
    for iid, cid in (
        ("us_2y", "rate_us2y"),
        ("us_10y", "rate_us10y"),
        ("vix", "global_vix"),
        ("wti", "commodity_wti_fred"),
    ):
        s = ind[iid]["splice"]
        assert s["copy_series_id"] == cid and s["copy_rows"] > 0
        assert s["macro_series_last_date"] == "2026-10-21" and s["copy_first_date"] > "2026-10-21"
        assert ind[iid]["asof"] == "2026-10-27"
    assert ind["vix"]["splice"]["copy_rounding"] == 2
    # 한국 표에 다섯 사본 series가 없으면 macro_series만 쓴다
    for iid, cid in (
        ("us_3m", "rate_us3m"),
        ("credit_baa", "spread_us_baa10y"),
        ("usd_index", "fx_usd_broad"),
        ("brent", "commodity_brent_fred"),
        ("us_10y_real", "rate_us10y_real"),
    ):
        s = ind[iid]["splice"]
        assert (
            s["copy_series_id"] == cid and s["copy_rows"] == 0 and ind[iid]["asof"] == "2026-10-21"
        )
        assert s["splice_point"] == "이을 사본 행 없음 (macro_series만)"
    assert (
        round(ind["vix"]["value"]["level"], 2) == ind["vix"]["value"]["level"]
    )  # 사본으로 이은 값은 둘째 자리


def test_five_new_copies_are_spliced_when_present(tmp_path):
    w = W.build_world(tmp_path, with_new_copies=True)
    b = build(w)
    ind = by_id(b)
    for iid in ("us_3m", "credit_baa", "usd_index", "brent", "us_10y_real"):
        assert ind[iid]["splice"]["copy_rows"] > 0 and ind[iid]["asof"] == "2026-10-27"
    fig = next(f for f in b["figures"] if f["name"] == "rates_us_nominal")
    assert fig["lines"][0]["last"]["date"] == "2026-10-27"
    # VIX 사본만 소수 둘째 자리로 반올림, 다른 사본은 그대로
    vix = ind["vix"]["value"]["level"]
    assert round(vix, 2) == vix
    assert ind["us_3m"]["value"]["level"] != round(ind["us_3m"]["value"]["level"], 2)


# --------------------------------------------------------------------------- 가용일·vintage
def test_availability_and_vintage_rules(tmp_path):
    kr_extra = [
        ("rate_kr_cd91", dt.date(2026, 10, 29), 99.0, dt.date(2026, 10, 30)),  # D 이후 가용
        ("fx_usdkrw_ecos", dt.date(2026, 10, 20), 5000.0, dt.date(2026, 10, 29)),
    ]  # 늦게 백필된 개정
    us_extra = [
        ("UNRATE", dt.date(2026, 8, 1), dt.date(2026, 10, 5), 8.88),  # D 이전 개정 -> 최신 vintage
        ("UNRATE", dt.date(2026, 8, 1), dt.date(2026, 11, 15), 99.9),  # D 이후 개정 -> 못 쓴다
        ("DGS2", dt.date(2026, 10, 21), dt.date(2026, 10, 29), 77.0),
    ]  # D 이후 vintage
    w = W.build_world(tmp_path, kr_extra=kr_extra, us_extra=us_extra)
    b = build(w)
    ind = by_id(b)
    assert ind["kr_cd91"]["asof"] == "2026-10-27" and ind["kr_cd91"]["value"]["level"] < 10
    assert ind["usdkrw"]["value"]["level"] < 3000
    assert b["checks"]["kr_rows_excluded_by_available_from_gt_D"] >= 3
    assert (
        b["checks"]["kr_max_available_from_used"] <= UNIT
        and b["checks"]["us_max_realtime_start_used"] <= UNIT
    )
    un = ind["us_unrate"]
    assert un["value"]["level"] == 8.88 and un["asof"] == "2026-08"
    assert un["release_date"] == "2026-09-10"  # 첫 vintage의 가용일(발표일)
    assert ind["us_2y"]["splice"]["macro_series_last_date"] == "2026-10-21"
    assert b["checks"]["all_available_from_le_D"] and b["checks"]["all_realtime_start_le_D"]


def test_missing_us_month_is_skipped_and_cuts_the_line(body):
    cpi = by_id(body)["us_cpi"]
    assert cpi["missing_months_10y"] == ["2025-10"] and cpi["missing_months_inside"] == 1
    fig = next(f for f in body["figures"] if f["name"] == "macro_prices_us")
    segs = fig["lines"][0]["segments"]
    assert len(segs) == 2  # 결측 달(2025-10)에서 선이 끊긴다
    pts = fig["lines"][0]["points"]
    assert all(not (p[0] == "2025-10-01") for p in pts)
    assert len(fig["lines"][1]["segments"]) == 1  # PPI는 끊기지 않는다


def test_monthly_fields(body):
    m = by_id(body)["us_unrate"]
    assert m["kind"] == "monthly" and m["asof"] == "2026-08" and m["release_date"] == "2026-09-10"
    # 발표 지연 중앙값 40일: 10-28이면 9월 값(10-11 발표 예정)이 나왔어야 하는데 8월까지라 지연 1개월
    assert (
        m["status"] == "지연 1개월" and m["stale"] is True and m["expected_last_month"] == "2026-09"
    )
    assert (
        by_id(body)["kr_sent"]["status"] == "정상"
    )  # 한국은 월말 + 20일 가용, 9월 값이 10-21에 나왔다
    assert {"prev_value", "missing_months_10y", "measures", "percentile", "streak", "stale"} <= set(
        m
    )
    assert by_id(body)["kr_unrate"]["level_note"].startswith("수준 ")


# --------------------------------------------------------------------------- vs_prev
def test_new_condition_appears_when_only_today_has_the_move(tmp_path):
    w = W.build_world(tmp_path, jump_fx_last=0.20)
    b = build(w)
    new = [(e["id"], e["period"], e["condition"]) for e in b["vs_prev"]["new"]]
    assert ("usdkrw", "1주", "급등") in new
    cur = by_id(b)["usdkrw"]
    assert any(
        p["label"] == "1주" and p["strong"] and p["direction"] == "up" for p in cur["periods"]
    )
    assert all(e["streak"] <= 20 and e["streak"] >= 2 for e in b["vs_prev"]["continued"])
    assert b["vs_prev"]["recomputed_dates"] >= 1 and b["vs_prev"]["max_count"] == 20


# --------------------------------------------------------------------------- 입력 고정
def test_changed_file_is_refused_and_reason_code(tmp_path):
    w = W.build_world(tmp_path)
    sel = W.make_selection(w)
    raw = w.kr_file.read_bytes()
    w.kr_file.write_bytes(raw + b"\0")
    with pytest.raises(ip.InputChangedError):
        build(w, sel)
    try:
        build(w, sel)
    except Exception as exc:
        assert mc.failure_reason(exc) == "input_changed"
    w.kr_file.write_bytes(raw)
    build(w, sel)  # 원래대로 돌려 놓으면 다시 열린다
    uraw = w.us_file.read_bytes()
    w.us_file.write_bytes(uraw + b"\0")
    with pytest.raises(ip.InputChangedError, match="macro_series"):
        build(w, sel)


def test_pin_errors_are_refused(tmp_path):
    w = W.build_world(tmp_path)
    sel = W.make_selection(w)
    bad = copy.deepcopy(sel)
    bad["schema_version"] = "other"
    with pytest.raises(ip.InputPinError, match="스키마"):
        build(w, bad)
    with pytest.raises(ip.InputPinError, match="리포트 날짜"):
        build(w, sel, unit="2026-10-27")
    bad = copy.deepcopy(sel)
    bad["us"]["tables"] = {}
    with pytest.raises(ip.InputPinError, match="macro_series"):
        build(w, bad)
    bad = copy.deepcopy(sel)
    bad["kr"] = {
        "status": "unavailable",
        "reason": "no_snapshot_completed_before_cutoff",
        "skipped": [],
    }
    with pytest.raises(ip.InputPinError, match="KR snapshot"):
        build(w, bad)
    bad = copy.deepcopy(sel)
    bad["kr"]["export_finished_at"] = "2026-10-28T09:31:00+09:00"  # D 09:30 이후에 끝난 export
    with pytest.raises(ip.InputPinError, match="09:30"):
        build(w, bad)
    bad = copy.deepcopy(sel)
    bad["us"]["tables"]["macro_series"]["completed_at"] = "2026-10-28T10:00:00+09:00"
    with pytest.raises(ip.InputPinError, match="09:30"):
        build(w, bad)
    bad = copy.deepcopy(sel)
    bad["kr"]["success_marker_sha256"] = "0" * 64
    with pytest.raises(ip.InputChangedError, match="marker"):
        build(w, bad)
    # 고정하지 않은 파일이 snapshot 디렉터리에 더해져도 거부한다
    extra = w.kr_file.with_name("part-000001.parquet")
    extra.write_bytes(w.kr_file.read_bytes())
    with pytest.raises(ip.InputChangedError, match="파일 목록"):
        build(w, sel)


def test_missing_series_surfaces_as_input_missing(tmp_path):
    w = W.build_world(tmp_path)
    import polars as pl

    df = pl.read_parquet(w.kr_file).filter(pl.col("series_id") != "trdval_kosdaq_ecos")
    df.write_parquet(w.kr_file)
    sel = W.make_selection(w)  # 고쳐 쓴 파일을 다시 고정한다
    with pytest.raises(Exception) as ei:
        build(w, sel)
    assert type(ei.value).__name__ == "MacroInputError" and "trdval_kosdaq_ecos" in str(ei.value)
    assert mc.failure_reason(ei.value) == "input_missing"


# --------------------------------------------------------------------------- 실패 문서·바이트
def test_failure_document_and_bytes():
    doc = mc.failure_document(
        unit=UNIT,
        invocation_id="inv-1",
        release="v1",
        selection_sha256=None,
        reason="input_changed",
    )
    assert doc == {
        "schema": "briefing-macro.v1",
        "state": "failed",
        "unit": UNIT,
        "invocation_id": "inv-1",
        "release": "v1",
        "selection_sha256": None,
        "reason": "input_changed",
    }
    raw = mc.document_bytes(doc)
    assert (
        raw.endswith(b"\n")
        and raw.count(b"\n") == 1
        and b" " not in raw.replace("한".encode(), b"")
    )
    assert json.loads(raw) == doc
    assert list(json.loads(raw)) == sorted(doc)  # 키 정렬
    with pytest.raises(ValueError):
        mc.document_bytes({"x": float("nan")})
    assert mc.document_bytes({"b": "한글", "a": 1}) == '{"a":1,"b":"한글"}\n'.encode()


def test_failure_reason_codes():
    assert mc.failure_reason(ip.InputChangedError("x")) == "input_changed"
    assert mc.failure_reason(ip.InputPinError("x")) == "input_unavailable"
    assert mc.failure_reason(FileNotFoundError("x")) == "input_unavailable"
    assert mc.failure_reason(ValueError("x")) == "compute_error"


def test_failure_path_needs_no_heavy_imports():
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[3] / "src"
    code = (
        "import sys; import modeler.scores.macro_context as m; "
        "d = m.failure_document(unit='u', invocation_id='i', release='r', selection_sha256=None, reason='x'); "
        "m.document_bytes(d); print(any(k in sys.modules for k in ('numpy', 'polars')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(src), "PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_vs_prev_recompute_excludes_us_rows_published_on_that_us_date(tmp_path):
    """A US vintage with realtime_start == P (previous weekday) came out on US date P, after the
    P 09:30 KST report: the recomputed P state must not see it. D itself still uses <= D."""
    from modeler.scores.macro_context.build import compute_at
    from modeler.scores.macro_context.load import load_macro_data
    from modeler.scores.macro_context.summary import prev_weekday

    P = prev_weekday(W.D)
    us_extra = [("UNRATE", dt.date(2026, 8, 1), P, 8.88)]  # revision released on US date P
    w = W.build_world(tmp_path, us_extra=us_extra)
    data = load_macro_data(W.make_selection(w), W.D)

    def unrate(res) -> float:
        return next(o for o in res.mons if o["id"] == "us_unrate")["value"]["level"]

    assert unrate(compute_at(data, P)) == 8.88  # the old rule (<= P) would have used it
    assert unrate(compute_at(data, P, us_before=True)) != 8.88
    assert unrate(compute_at(data, W.D)) == 8.88  # D: realtime_start <= D, unchanged
    b = build(w)
    assert by_id(b)["us_unrate"]["value"]["level"] == 8.88
    assert b["checks"]["vs_prev_us_rule"] == "realtime_start < d"
    assert "realtime_start < d" in b["rules"]["vs_prev"]["availability"]
