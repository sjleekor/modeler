"""E6: the failed and stale sections say why the 09:30 select found no input (or an old one).

On 2026-10-06 the US section of the live unit only said the cause class ``MissingInference``.  The
envelope now carries ``selection_summary`` (what select itself recorded per market) and the renderer
writes a "select 사유" line in failed and stale sections.  An envelope without the field renders
byte for byte as before; a healthy section never shows the line.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import md_fixtures as mf

from modeler.reporting import markdown as md

NO_REPO = Path("/nonexistent-stock-reports")
DAY = "2026-10-06"
US_NATIVE_MISSING = {"status": "unavailable", "reason": "producer_completion_missing"}
KR_OK = {"status": "ok", "feature_asof_date": "2026-10-02", "lag_sessions": 0}


def _render(env: dict, ms: dict | None = None) -> dict:
    """The unit's files, with the envelope's own sha256 (printed in the unit) masked.

    The sha256 differs once the envelope carries one more key, which these tests do not compare.
    """
    env_bytes = json.dumps(env).encode()
    ms_bytes = None if ms is None else json.dumps(ms).encode()
    ctx = md.build_context(NO_REPO, env_bytes, ms_bytes, "r1",
                           "2026-10-06T10:03:00+09:00", 100, lambda message: None)
    masked = md.sha256_bytes(env_bytes)
    return {name: text.replace(masked, "<envelope sha256>")
            for name, text in md.render_unit(ctx).items()}


def _us_unavailable_day(summary: object = None) -> tuple[dict, dict]:
    """The 10-06 situation: KR ranked, no US native input, both US models MissingInference."""
    reports = [mf.kr_report(DAY, "2026-10-02")]
    failures = [{"market": "US", "model_id": mf.LGB_ID, "error_class": "MissingInference"},
                {"market": "US", "model_id": mf.RDG_ID, "error_class": "MissingInference"}]
    extra = None if summary is None else {"selection_summary": summary}
    env = mf.envelope(DAY, reports, failures, extra=extra)
    ms = mf.ms_input(DAY, kr_asof="2026-10-02", us_asof="2026-10-02", macro_asof="2026-10-01")
    return env, ms


def test_us_unavailable_day_shows_the_select_reason_next_to_the_cause_class():
    env, ms = _us_unavailable_day({"KR": KR_OK, "US": US_NATIVE_MISSING})
    files = _render(env, ms)
    us = files["us-stocks.md"]
    assert "원인 클래스: `MissingInference`" in us
    line = ("select 사유: 상태 `unavailable`(고른 입력 없음), 사유 `producer_completion_missing`"
            "(D 09:30 전에 끝난 prepare 입력(native)이 없습니다).")
    assert line in us
    assert us.count(line) == 2  # one per US model section
    status = files["data-status.md"]
    assert line in status  # the section table and the warnings
    # KR ranked fine: its file carries no select line
    assert "select 사유" not in files["kr-stocks.md"]
    assert "select 사유" not in files["market-sector.md"]


def test_without_the_field_nothing_changes():
    env, ms = _us_unavailable_day()
    assert "selection_summary" not in env
    plain = _render(env, ms)
    assert all("select 사유" not in text for text in plain.values())
    # a field that is not an object, or names no market, is ignored the same way
    for junk in ("x", [], {}, {"US": "x"}, {"US": {}}, {"US": {"status": 5, "lag_sessions": "3"}}):
        junked = copy.deepcopy(env)
        junked["selection_summary"] = junk
        assert _render(junked, ms) == plain, junk


def test_healthy_sections_are_unchanged_by_the_field():
    env, ms = mf.scenario_normal()
    plain = _render(env, ms)
    with_summary = copy.deepcopy(env)
    with_summary["selection_summary"] = {
        "KR": {"status": "ok", "feature_asof_date": "2026-10-06", "lag_sessions": 0},
        "US": {"status": "ok", "feature_asof_date": "2026-10-06", "lag_sessions": 0}}
    assert _render(with_summary, ms) == plain


def test_stale_sections_show_what_select_recorded():
    day = "2026-10-08"
    kr = mf.kr_report(day, "2026-10-02", lag=3, status="stale", fresh_status="stale")
    reports = [
        kr,
        mf.us_report(day, mf.LGB_ID, "2026-10-02", "2026-10-07", "2026-10-07", lag=3,
                     status="stale"),
        mf.us_report(day, mf.RDG_ID, "2026-10-02", "2026-10-07", "2026-10-07", lag=3,
                     status="stale"),
    ]
    summary = {
        "KR": {"status": "stale", "feature_asof_date": "2026-10-02", "lag_sessions": 3,
               "freshness_reason": "KR feature session does not match K"},
        "US": {"status": "stale", "feature_asof_date": "2026-10-02", "lag_sessions": 3}}
    env = mf.envelope(day, reports, [], extra={"selection_summary": summary})
    ms = mf.ms_input(day, kr_asof="2026-10-06", us_asof="2026-10-07", macro_asof="2026-10-04")
    files = _render(env, ms)
    kr_line = ("select 사유: 상태 `stale`(자료 지연), 신선도 사유 KR 기준일이 K와 다릅니다, "
               "기준일 2026-10-02, 지연 3세션.")
    assert "> " + kr_line in files["kr-stocks.md"]
    assert "> select 사유: 상태 `stale`(자료 지연), 기준일 2026-10-02, 지연 3세션." in files[
        "us-stocks.md"]
    assert kr_line in files["data-status.md"]  # the reason cell of the section table
    assert ("US LightGBM·Ridge: 입력이 3세션 늦습니다. select 사유: 상태 `stale`(자료 지연), "
            "기준일 2026-10-02, 지연 3세션.") in files["README.md"]
    assert "KR: 입력이 3세션 늦습니다. " + kr_line in files["README.md"]  # the summary's warnings
    plain = _render(mf.envelope(day, reports, []), ms)
    assert all("select 사유" not in text for text in plain.values())


def test_a_failed_model_with_a_selected_input_still_shows_the_select_state():
    """Select found an input but inference failed (runner side): the line says what was selected."""
    env, ms = _us_unavailable_day(
        {"US": {"status": "ok", "feature_asof_date": "2026-10-02", "lag_sessions": 0}})
    us = _render(env, ms)["us-stocks.md"]
    assert "select 사유: 상태 `ok`(정상), 기준일 2026-10-02, 지연 0세션." in us


def test_unknown_values_are_escaped_and_not_trusted():
    env, ms = _us_unavailable_day(
        {"US": {"status": "unavailable", "reason": "a|b`c <script>",
                "feature_asof_date": "not-a-date", "lag_sessions": True}})
    us = _render(env, ms)["us-stocks.md"]
    assert "<script>" not in us and "a\\|b\\`c \\<script\\>" in us
    assert "not-a-date" not in us and "지연" not in us.split("select 사유")[1].split("\n")[0]


def test_select_reason_text_wording():
    text = md.select_reason_text
    assert text(None) is None and text({}) is None and text("x") is None
    assert text({"reason": "kr_prepared_newer_than_k"}) == (
        "select 사유: 사유 `kr_prepared_newer_than_k`"
        "(KR prepare 입력의 기준일이 K보다 늦어 쓰지 않았습니다).")
    # a freshness sentence recorded as the reason is translated like the freshness reasons
    assert text({"status": "unavailable", "reason": "US delivery lag reached the stop limit"}) == (
        "select 사유: 상태 `unavailable`(고른 입력 없음), 사유 "
        "US delivery lag reached the stop limit(미국 도착 지연이 중단 한도에 닿았습니다).")
