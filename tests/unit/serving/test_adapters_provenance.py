"""KR and US reports share provenance keys with the same meaning (pinned file SHA-256)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from modeler.serving import adapters, us_daily
from modeler.serving.schema import report_template

BUNDLE_FILE_SHA = "a" * 64
SELECTION_FILE_SHA = "b" * 64
NATIVE_SHA = "c" * 64
INNER_BUNDLE_HASH = "d" * 64
INNER_PREPARED_HASH = "e" * 64


def _context(tmp_path: Path, market: str, model_id: str, prepared: Path) -> SimpleNamespace:
    bundle_dir = tmp_path / "bundle" / ("lightgbm" if market == "US" else "kr")
    bundle_dir.mkdir(parents=True)
    freshness = {"latest_us_session": "2026-09-29"}
    return SimpleNamespace(
        report_date=date(2026, 9, 30), decision_at=datetime(2026, 9, 30, 1, 0, tzinfo=timezone.utc),
        market=market, model_id=model_id, model_version="1", prepared_input=prepared,
        bundle_manifest=bundle_dir / "manifest.json", input_sha256="f" * 64,
        prepared_manifest_sha256=SELECTION_FILE_SHA, native_manifest_sha256=NATIVE_SHA,
        bundle_sha256=BUNDLE_FILE_SHA, code_sha256="1" * 64, feature_asof_date="2026-09-29",
        freshness_status="ok", freshness=freshness, fixture_mode=True,
        selection={"market": market, "actual_us_session": "2026-09-29",
                   "native_prepare_manifest_sha256": NATIVE_SHA, "input_sha256": "f" * 64},
        native_preparation={"market": market, "feature_asof_date": "2026-09-29",
                            "scoring_date": "2026-09-29", "code_hash": "code"})


def test_kr_report_uses_pinned_file_shas_and_keeps_internal_hashes(tmp_path, monkeypatch) -> None:
    import modeler.serving.kr_model as kr_model
    import modeler.serving.kr_serving as kr_serving

    prepared = tmp_path / "panel.parquet"
    pl.DataFrame({"ticker": ["A"]}).write_parquet(prepared)
    context = _context(tmp_path, "KR", "kr_daily_h20_v1", prepared)
    monkeypatch.setattr(kr_model, "load_bundle", lambda *_a, **_k: SimpleNamespace(
        manifest={"model_id": "kr_daily_h20_v1", "model_version": "1"}))

    def fake_score(**_kwargs):
        report = report_template(
            market="KR", report_date="2026-09-30", decision_at=context.decision_at,
            feature_asof_date="2026-09-29", model_id="kr_daily_h20_v1", model_version="1")
        report["quality"] = {}
        report["provenance"] = {"bundle_manifest_sha256": INNER_BUNDLE_HASH,
                                "prepared_manifest_sha256": INNER_PREPARED_HASH}
        return report

    monkeypatch.setattr(kr_serving, "score_cross_section", fake_score)
    provenance = adapters.infer_kr_daily(context)["provenance"]
    assert provenance["bundle_manifest_sha256"] == BUNDLE_FILE_SHA
    assert provenance["prepared_manifest_sha256"] == SELECTION_FILE_SHA
    assert provenance["bundle_manifest_content_sha256"] == INNER_BUNDLE_HASH
    assert provenance["native_prepare_manifest_content_sha256"] == INNER_PREPARED_HASH
    assert provenance["native_prepare_manifest_sha256"] == NATIVE_SHA


def test_us_report_bundle_manifest_sha_is_the_pinned_file_sha(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "features.parquet"
    pl.DataFrame({"date": [date(2026, 9, 29)], "symbol": ["AAA"]}).write_parquet(prepared)
    context = _context(tmp_path, "US", f"{us_daily.MODEL_ID}_lightgbm", prepared)
    monkeypatch.setattr(us_daily, "code_tree_hash", lambda: "code")
    monkeypatch.setattr(us_daily, "load_model", lambda *_: (
        {"model": object()}, {"model_id": us_daily.MODEL_ID, "variant": "lightgbm", "model_version": "1"}))
    monkeypatch.setattr(us_daily, "rank_daily", lambda *_: pl.DataFrame(
        {"rank": [1], "symbol": ["AAA"], "score": [0.5]}))
    provenance = adapters.infer_us_model(context)["provenance"]
    assert provenance["bundle_manifest_sha256"] == BUNDLE_FILE_SHA
    assert provenance["native_prepare_manifest_sha256"] == NATIVE_SHA


def test_kr_report_carries_the_reference_session_check_as_quality_facts(tmp_path, monkeypatch) -> None:
    """The data-status page needs the verdict, K's ticker ratio and the gate's advisory verdict."""
    import modeler.serving.kr_model as kr_model
    import modeler.serving.kr_serving as kr_serving

    prepared = tmp_path / "panel.parquet"
    pl.DataFrame({"ticker": ["A"]}).write_parquet(prepared)
    context = _context(tmp_path, "KR", "kr_daily_h20_v1", prepared)
    reference = {
        "verdict": "fallback_K_prime", "k": "2026-09-30", "reference_date": "2026-09-29",
        "lag_sessions": 1,
        "candidates": [{"session": "2026-09-30", "complete": False, "price": {
            "ticker_count": 1656, "ticker_ratio": 0.6, "previous_session": "2026-09-29"}}],
        "gate": {"verdict": "not_yet", "dart_chain_ended_at": None, "advisory": True},
    }
    context.native_preparation["reference_selection"] = reference
    monkeypatch.setattr(kr_model, "load_bundle", lambda *_a, **_k: SimpleNamespace(
        manifest={"model_id": "kr_daily_h20_v1", "model_version": "1"}))

    def fake_score(**_kwargs):
        report = report_template(
            market="KR", report_date="2026-09-30", decision_at=context.decision_at,
            feature_asof_date="2026-09-29", model_id="kr_daily_h20_v1", model_version="1")
        report["quality"] = {}
        report["provenance"] = {"bundle_manifest_sha256": INNER_BUNDLE_HASH,
                                "prepared_manifest_sha256": INNER_PREPARED_HASH}
        return report

    monkeypatch.setattr(kr_serving, "score_cross_section", fake_score)
    report = adapters.infer_kr_daily(context)
    assert report["quality"]["reference_verdict"] == "fallback_K_prime"
    assert report["quality"]["k_ticker_ratio"] == 0.6 and report["quality"]["k_ticker_count"] == 1656
    assert report["quality"]["export_gate_verdict"] == "not_yet"
    assert "dart_chain_ended_at" not in report["quality"]  # None values are left out
    assert report["provenance"]["reference_selection"] == reference
    # A prepared input from before this change has no evidence: nothing is invented.
    del context.native_preparation["reference_selection"]
    plain = adapters.infer_kr_daily(context)
    assert "reference_verdict" not in plain["quality"]
    assert plain["provenance"]["reference_selection"] is None


def test_us_report_reads_the_lags_from_the_freshness_assessment_keys(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "features.parquet"
    pl.DataFrame({"date": [date(2026, 9, 29)], "symbol": ["AAA"]}).write_parquet(prepared)
    context = _context(tmp_path, "US", f"{us_daily.MODEL_ID}_lightgbm", prepared)
    context.freshness = {"latest_us_session": "2026-09-29", "market_lag": 3, "delivery_lag": 2}
    context.selection["market_lag_limit_sessions"] = 2
    monkeypatch.setattr(us_daily, "code_tree_hash", lambda: "code")
    monkeypatch.setattr(us_daily, "load_model", lambda *_: (
        {"model": object()}, {"model_id": us_daily.MODEL_ID, "variant": "lightgbm", "model_version": "1"}))
    monkeypatch.setattr(us_daily, "rank_daily", lambda *_: pl.DataFrame(
        {"rank": [1], "symbol": ["AAA"], "score": [0.5]}))
    quality = adapters.infer_us_model(context)["quality"]
    assert quality["market_lag_sessions"] == 3 and quality["delivery_lag_sessions"] == 2
    assert quality["market_lag_limit_sessions"] == 2
