import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from modeler.feedback_outcomes import output
from modeler.feedback_outcomes.__main__ import main
from modeler.feedback_outcomes.contracts import OutcomeError
from modeler.feedback_outcomes.pipeline import evaluate
from modeler.feedback_outcomes.synthetic import fixture


def evaluate_fixture(data):
    return evaluate(
        data["calendar"], data["events"], data["extract"], cutoff=data["cutoff"], synthetic=True
    )


def test_manual_rows_cohorts_and_repeated_recommendations():
    result = evaluate_fixture(fixture())
    assert len(result["rows"]) == 200
    assert len(result["cohorts"]) == 4
    assert all(row["net"] == pytest.approx(0.094) for row in result["rows"])
    assert all(row["benchmark"] == pytest.approx(0.05) for row in result["rows"])
    assert all(row["excess"] == pytest.approx(0.044) for row in result["rows"])
    assert all(row["market"] == "KR" for row in result["rows"])
    assert result["summary"]["publication_count"] == 2
    assert result["summary"]["unique_symbol_count"] == 100
    assert result["summary"]["repeated_recommendation_count"] == 100
    assert result["summary"]["top20"]["insufficient_sample"] is True


def test_missing_exit_price_is_never_filled_and_invalidates_whole_original_cohort():
    data = fixture()
    data["extract"]["daily_ohlcv"] = [
        r
        for r in data["extract"]["daily_ohlcv"]
        if not (r["ticker"] == "000001" and r["trade_date"] == "2026-12-02")
    ]
    result = evaluate_fixture(data)
    row = next(r for r in result["rows"] if r["report_date"] == "2026-11-04" and r["rank"] == 1)
    assert row["net"] is None
    for cohort in result["cohorts"]:
        if cohort["report_date"] == "2026-11-04":
            assert cohort["net"] is None and cohort["benchmark"] is None
            assert cohort["unresolved_weight"] == pytest.approx(1 / cohort["size"])


def test_unmatured_metadata_does_not_even_construct_price_adapter(monkeypatch):
    from modeler.feedback_outcomes import pipeline

    data = fixture()
    monkeypatch.setattr(
        pipeline, "EndpointExtract", lambda *a, **kw: pytest.fail("opened premature prices")
    )
    result = evaluate(
        data["calendar"], data["events"], None, cutoff="2026-11-06T13:00:00+09:00", synthetic=True
    )
    assert all(row["status"] == "pending" and row["net"] is None for row in result["rows"])


def test_explicit_halt_requires_real_status_proof_and_keeps_original_cash_weight():
    data = fixture()
    cap = data["extract"]["daily_market_cap"][0]
    cap["explicit_untradeable"] = True
    unresolved = evaluate_fixture(data)
    assert unresolved["rows"][0]["status"] == "entry_unresolved"
    cap.update(untradeable_evidence_sha256="e" * 64, untradeable_basis="explicit_krx_status")
    result = evaluate_fixture(data)
    assert result["rows"][0]["status"] == "not_entered"
    assert result["rows"][0]["net"] == 0
    cohort = result["cohorts"][0]
    assert cohort["net"] == pytest.approx(19 / 20 * 0.094)
    assert cohort["benchmark"] == pytest.approx(0.05)


def test_unentered_cash_does_not_need_identity_after_entry_date():
    data = fixture()
    data["extract"]["listing_periods"][0]["valid_to"] = "2026-11-20"
    data["extract"]["daily_market_cap"][0].update(
        explicit_untradeable=True,
        untradeable_evidence_sha256="e" * 64,
        untradeable_basis="explicit_krx_status",
    )
    result = evaluate_fixture(data)
    assert result["rows"][0]["status"] == "not_entered"
    assert result["rows"][0]["net"] == 0
    assert result["cohorts"][0]["net"] == pytest.approx(19 / 20 * 0.094)


def test_immutable_outputs_roundtrip_and_repeat_do_not_add_a_run(tmp_path):
    result = evaluate_fixture(fixture())
    first = output.write_run(tmp_path, result, inputs={"synthetic_fixture": True})
    second = output.write_run(tmp_path, result, inputs={"synthetic_fixture": True})
    assert first == second
    path = Path(first["path"])
    assert pq.read_table(path / "rows.parquet").num_rows == 200
    assert pq.read_table(path / "cohorts.parquet").num_rows == 4
    manifest = json.loads((path / "run_manifest.json").read_bytes())
    assert manifest["synthetic_fixture"] is True
    assert "실제 예측·가격·성과가 아닙니다" in (path / "summary.md").read_text()
    assert len(list(path.parent.iterdir())) == 1


def test_changed_input_preserves_previous_revision(tmp_path):
    data = fixture()
    original = output.write_run(
        tmp_path, evaluate_fixture(data), inputs={"revision": "synthetic-v1"}
    )
    data["extract"]["adjustment_revision"] = "synthetic-v2"
    revised = output.write_run(
        tmp_path, evaluate_fixture(data), inputs={"revision": "synthetic-v2"}
    )
    assert original["run_id"] != revised["run_id"]
    assert (Path(original["path"]) / "run_manifest.json").is_file()


def test_real_cli_and_output_gate_stop_before_price_or_output_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        output, "_parquet", lambda rows: pytest.fail("serialized premature actual returns")
    )
    result = evaluate_fixture(fixture())
    result["synthetic_fixture"] = False
    with pytest.raises(OutcomeError, match="registered Friday"):
        output.write_run(tmp_path, result, inputs={})
    assert (
        main(
            [
                "real",
                "--read-plan",
                "/must-not-open/US/labels/read-plan.json",
                "--output-root",
                str(tmp_path / "actual"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "actual").exists()


def test_synthetic_cli_accepts_no_arbitrary_input_path(tmp_path):
    assert (
        main(
            [
                "synthetic",
                "--read-plan",
                "/not-opened/read-plan.json",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert not (tmp_path / "synthetic").exists()


def test_actual_evaluation_stays_blocked_without_source_bound_withdrawal_schema(monkeypatch):
    from modeler.feedback_outcomes import pipeline

    data = fixture()
    data["calendar"]["synthetic_fixture"] = False
    for event in data["events"]:
        event["synthetic_fixture"] = False
        event["explicit_withdrawal_adapter"] = "not_implemented_no_original_schema"
    monkeypatch.setattr(
        pipeline,
        "EndpointExtract",
        lambda *a, **kw: pytest.fail("opened actual prices before source contract"),
    )
    with pytest.raises(OutcomeError, match="withdrawal evidence adapter"):
        evaluate(
            data["calendar"],
            data["events"],
            None,
            cutoff=data["cutoff"],
            now=datetime.fromisoformat("2026-12-04T14:00:00+09:00"),
        )
