"""R4: market-sector failure documents and the R3 fields on ``data-status.md``.

* A failed or partial ``market-sector-D.json`` (written by the coordinator or by ``score_daily``)
  turns into readable reasons in the section, never into a crash or a silent empty table.
* The R3 fields (selection mode and times, KR reference evidence, the failure of a unit the
  coordinator made itself) show up in ``data-status.md`` only when the report carries them; a
  report without them renders exactly as before.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import md_fixtures as mf

from modeler.reporting import markdown as md
from modeler.scores.market_sector.daily_doc import failure_document

NO_REPO = Path("/nonexistent-stock-reports")


def _ctx(env: dict, ms: dict | None):
    ms_bytes = None if ms is None else json.dumps(ms).encode()
    return md.build_context(NO_REPO, json.dumps(env).encode(), ms_bytes, "r1",
                            "2026-10-07T10:03:00+09:00", 100, lambda message: None)


def _normal():
    return mf.scenario_normal()


def _status_text(env, ms):
    return md.render_unit(_ctx(env, ms))["data-status.md"]


def _fail_doc(day: str, **kwargs):
    from datetime import date

    return json.loads(json.dumps(failure_document(date.fromisoformat(day), **kwargs)))


# --------------------------------------------------------------------------- market-sector failures
def test_failure_document_becomes_the_sections_reasons():
    env, ms = _normal()
    doc = _fail_doc("2026-10-07", stage="score", error_class="MarketSectorExitNonzero",
                    reason="all_markets_failed", exit_code=1,
                    markets={"US": {"reason": "input_changed", "error_class": "InputChangedError"},
                             "KR": {"reason": "calendar_mismatch",
                                    "error_class": "CalendarMismatchError"}})
    ctx = _ctx(env, doc)
    assert ctx["ms"]["status"] == "failed" and ctx["ms"]["assets"] == []
    text = "\n".join(ctx["ms"]["reason"])
    assert "단계 채점" in text and "`MarketSectorExitNonzero`" in text
    assert "`all_markets_failed`" in text
    assert "US 시장·섹터 계산이 실패했습니다: 사유 `input_changed`" in text
    assert "KR 시장·섹터 계산이 실패했습니다: 사유 `calendar_mismatch`" in text
    files = md.render_unit(ctx)
    assert "시장·섹터를 내지 못했습니다." in files["market-sector.md"]
    assert "input_changed" in files["data-status.md"]
    assert ctx["status"] == "partial"  # the ranking sections still came out


def test_timeout_and_select_failures_name_their_stage():
    env, _ = _normal()
    timeout = _fail_doc("2026-10-07", stage="score", error_class="MarketSectorTimeout",
                        reason="timeout", exit_code=124, timed_out=True)
    reasons = _ctx(env, timeout)["ms"]["reason"]
    assert any("timeout" in r for r in reasons)
    select = _fail_doc("2026-10-07", stage="select", error_class="MarketSectorInputsUnavailable",
                       reason="inputs_unavailable",
                       markets={"KR": {"status": "unavailable",
                                       "reason": "no_snapshot_completed_before_cutoff"},
                                "US": {"status": "selected", "limit_session": "2026-10-06"}})
    text = "\n".join(_ctx(env, select)["ms"]["reason"])
    assert "단계 입력 선택" in text
    assert "KR 시장·섹터 계산이 실패했습니다: 사유 `no_snapshot_completed_before_cutoff`" in text
    assert "US 시장·섹터" not in text  # a market that was fine is not listed as failed


def test_failure_document_for_another_day_is_still_refused():
    env, _ = _normal()
    wrong_day = _fail_doc("2026-10-06", stage="score", error_class="X", reason="y")
    assert "입력 날짜 불일치" in "\n".join(_ctx(env, wrong_day)["ms"]["reason"])


def test_partial_document_names_the_failed_market():
    env, ms = _normal()
    ms = copy.deepcopy(ms)
    ms["assets"] = [a for a in ms["assets"] if a["market"] == "US"]
    ms["failures"] = {"KR": {"reason": "calendar_mismatch", "error_class": "CalendarMismatchError"}}
    ctx = _ctx(env, ms)
    assert ctx["ms"]["status"] == "partial"
    assert ctx["ms"]["reason"] == [
        "KR 시장·섹터 계산이 실패했습니다: 사유 `calendar_mismatch`, "
        "원인 클래스 `CalendarMismatchError`"]
    # without failure info the old wording stays
    del ms["failures"]
    assert _ctx(env, ms)["ms"]["reason"] == ["KR 자산 행이 없습니다"]


# --------------------------------------------------------------------------- R3 fields
R3_US = {"selection_mode": "scheduled", "selected_at": "2026-10-07T09:30:02+09:00",
         "producer_completed_at": "2026-10-07T03:41:10+09:00", "lag_sessions": 0}
R3_KR = {**R3_US, "reference_verdict": "fallback_K_prime", "reference_k": "2026-10-06",
         "reference_date": "2026-10-02", "reference_lag_sessions": 1, "k_ticker_ratio": 0.4213,
         "export_gate_verdict": "not_ready", "dart_chain_ended_at": "2026-10-07T02:10:00+09:00"}


def _with_r3(env: dict, kr: dict | None, us: dict | None) -> dict:
    env = copy.deepcopy(env)
    for report in env["markets"]:
        extra = kr if report["market"] == "KR" else us
        if extra:
            report["quality"] = {**report["quality"], **extra}
    return env


def test_reports_without_r3_fields_render_as_before():
    env, ms = _normal()
    text = _status_text(env, ms)
    for heading in ("## 입력 선택", "## KR 기준일 증거"):
        assert heading not in text
    assert "run이 대신 select함" not in text
    # an unrelated `failure`-less envelope adds nothing to the failure section either
    assert "실패 단계" not in text


def test_selection_and_kr_reference_evidence_are_shown():
    env, ms = _normal()
    text = _status_text(_with_r3(env, R3_KR, R3_US), ms)
    assert "## 입력 선택" in text and "정시 select" in text
    assert "2026-10-07 09:30:02 KST" in text and "2026-10-07 03:41:10 KST" in text
    assert "run이 대신 select함" not in text
    assert "## KR 기준일 증거" in text
    for expected in ("K보다 앞선 세션(K′)으로 내려감", "| K (직전 KR 세션) | 2026-10-06 |",
                     "| 기준일 | 2026-10-02 |", "| K와의 차이(세션) | 1 |", "42.1%",
                     "| export gate 판정 (참고용) | not_ready |",
                     "2026-10-07 02:10:00 KST"):
        assert expected in text, expected
    assert "export gate와 DART chain은 참고 기록이며 기준일을 막지 않습니다." in text


def test_a_run_fallback_selection_is_said_in_one_line():
    env, ms = _normal()
    fallback = {**R3_KR, "selection_mode": "run_fallback",
                "selected_at": "2026-10-07T10:00:41+09:00"}
    text = _status_text(_with_r3(env, fallback, {**R3_US, "selection_mode": "run_fallback"}), ms)
    assert text.count("| run이 대신 select함 |") == 3  # KR and both US models
    assert text.count("run이 대신 select함:") == 1  # the one explaining line
    assert "09:30 select가 돌지 않아 run 단계가 같은 규칙으로 직접 입력을 골랐습니다." in text
    assert "D 09:30 뒤에 끝난 입력은 이때도 쓰지 않았습니다." in text


def test_a_coordinator_made_failure_unit_shows_stage_and_timeout():
    env, ms = _normal()
    env = copy.deepcopy(env)
    env["markets"] = []
    env["failures"] = [{"market": "KR", "model_id": mf.KR_ID, "error_class": "RunnerTimeout"}]
    env["failure"] = {"stage": "infer", "error_class": "RunnerTimeout", "runner_exit_code": 124,
                      "timed_out": True, "synthesized_by": "coordinator"}
    text = _status_text(env, ms)
    assert "| 실패 단계 | 원인 클래스 | runner 종료 코드 | timeout |" in text
    assert "| 추론 | `RunnerTimeout` | 124 | 예 |" in text
    assert "이번 실행이 실패한 단계를 기록하고 직접 만든 실패 단위입니다." in text
    # the stage can also be the selection
    env["failure"] = {**env["failure"], "stage": "select", "runner_exit_code": None,
                      "timed_out": False}
    assert "| 입력 선택 | `RunnerTimeout` | - | 아니오 |" in _status_text(env, ms)


def test_odd_values_in_r3_fields_do_not_break_the_page():
    env, ms = _normal()
    odd = {"selection_mode": 7, "selected_at": "not a time", "producer_completed_at": None,
           "lag_sessions": "x", "reference_verdict": ["a"], "reference_k": "x",
           "k_ticker_ratio": "x", "export_gate_verdict": 1}
    text = _status_text(_with_r3(env, odd, odd), ms)
    assert "미기록" in text and "## 입력 선택" in text
