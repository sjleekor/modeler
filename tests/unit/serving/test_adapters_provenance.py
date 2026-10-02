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
