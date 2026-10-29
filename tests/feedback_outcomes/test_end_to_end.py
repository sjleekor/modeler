"""Real F3 capture APIs fed entirely by generated report/git/price fixtures."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from test_metadata import OBSERVED, synthetic_capture

from modeler.feedback_outcomes.metadata import export_capture_metadata, normalize_publications
from modeler.feedback_outcomes.output import write_run
from modeler.feedback_outcomes.pipeline import evaluate
from modeler.feedback_outcomes.synthetic import fixture


def test_generated_f3_bytes_to_first_sample_to_parquet_and_idempotent_manifest(tmp_path):
    source = tmp_path / "producer-source"
    config, _, _, receipt = synthetic_capture(source)
    bundle = export_capture_metadata(config)
    events = normalize_publications(bundle, observed_at=OBSERVED, synthetic=True)
    data = fixture()
    sessions = {"2026-11-04", "2026-12-02"}
    for key, column in (
        ("daily_ohlcv", "trade_date"),
        ("daily_market_cap", "trade_date"),
        ("krx_index_daily", "bas_dd"),
    ):
        data["extract"][key] = [r for r in data["extract"][key] if r[column] in sessions]
    result = evaluate(data["calendar"], events, data["extract"], cutoff=OBSERVED, synthetic=True)
    assert len(result["rows"]) == 100 and len(result["cohorts"]) == 2
    assert result["cohorts"][0]["receipt_ref"] == receipt["capture_ref"]
    assert result["cohorts"][0]["net"] == pytest.approx(0.094)
    assert result["cohorts"][0]["excess"] == pytest.approx(0.044)
    assert result["summary"]["publication_count"] == 1
    assert result["rows"][2]["quality_reasons"] == ["synthetic-quality-warning"]
    first = write_run(tmp_path / "result", result, inputs={"fixture": True}, source_roots=[source])
    second = write_run(tmp_path / "result", result, inputs={"fixture": True}, source_roots=[source])
    assert first == second
    target = Path(first["path"])
    assert pq.read_table(target / "rows.parquet").num_rows == 100
    assert pq.read_table(target / "cohorts.parquet").num_rows == 2
    manifest = json.loads((target / "run_manifest.json").read_bytes())
    assert manifest["rule"]["round_trip_cost"] == 0.006
    assert manifest["calendar_code_sha256"]
    assert manifest["synthetic_fixture"] is True


def test_capture_with_missing_freshness_is_not_promoted_into_a_sample(tmp_path):
    config, *_ = synthetic_capture(tmp_path / "source", missing_lag=True)
    events = normalize_publications(
        export_capture_metadata(config), observed_at=OBSERVED, synthetic=True
    )
    data = fixture()
    result = evaluate(data["calendar"], events, None, cutoff=OBSERVED, synthetic=True)
    assert result["rows"] == [] and result["cohorts"] == []
    assert result["decisions"][0]["reasons"] == ["stale_signal"]
