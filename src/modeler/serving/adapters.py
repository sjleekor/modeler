"""Pure adapters from validated daily selections to the adopted KR/US scorers."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from modeler.serving.orchestration import InferenceContext
from modeler.serving.schema import report_template, validate_report


def _reference_quality(reference: Any) -> dict[str, Any]:
    """Scalar facts of the KR reference-session check, for the data-status page."""
    if not isinstance(reference, dict):
        return {}
    candidates = reference.get("candidates")
    k_check = candidates[0] if isinstance(candidates, list) and candidates else {}
    price = k_check.get("price", {}) if isinstance(k_check, dict) else {}
    gate = reference.get("gate") if isinstance(reference.get("gate"), dict) else {}
    facts = {
        "reference_verdict": reference.get("verdict"),
        "reference_k": reference.get("k"),
        "reference_date": reference.get("reference_date"),
        "reference_lag_sessions": reference.get("lag_sessions"),
        "k_ticker_count": price.get("ticker_count"),
        "k_ticker_ratio": price.get("ticker_ratio"),
        "k_previous_session": price.get("previous_session"),
        "export_gate_verdict": gate.get("verdict"),
        "dart_chain_ended_at": gate.get("dart_chain_ended_at"),
    }
    return {key: value for key, value in facts.items() if value is not None}


def infer_kr_daily(context: InferenceContext) -> dict[str, Any]:
    """Score one pinned KR panel with the bundle named by a hash-pinned manifest."""
    if (context.market, context.model_id) != ("KR", "kr_daily_h20_v1"):
        raise ValueError("KR adapter received an unexpected model identity")
    from modeler.serving.kr_model import load_bundle
    from modeler.serving.kr_serving import score_cross_section

    bundle = load_bundle(context.bundle_manifest.parent, verify_golden=True)
    if (bundle.manifest.get("model_id"), str(bundle.manifest.get("model_version"))) != (
        context.model_id, context.model_version
    ):
        raise ValueError("KR bundle identity differs from the pinned job")
    selection = context.selection
    panel = pl.read_parquet(context.prepared_input)
    report = score_cross_section(
        bundle=bundle,
        panel=panel,
        report_date=context.report_date.isoformat(),
        decision_at=context.decision_at.isoformat(),
        feature_asof_date=context.feature_asof_date,
        prepared_manifest=context.native_preparation,
    )
    report["status"] = "partial"
    report["synthetic_fixture"] = context.fixture_mode
    report["quality"] = {**report["quality"], "freshness_status": context.freshness_status,
                         **_reference_quality(context.native_preparation.get("reference_selection"))}
    inner = report["provenance"]
    report["provenance"] = {**inner,
        # The completeness check that chose this session (complete_K / fallback_K_prime), with the
        # collector gate's advisory verdict (2026-10-05 change 1); absent on older prepared inputs.
        "reference_selection": context.native_preparation.get("reference_selection"),
        # Shared keys carry the same meaning as US: sha256 of the pinned files.
        "bundle_manifest_sha256": context.bundle_sha256,
        "prepared_manifest_sha256": context.prepared_manifest_sha256,
        # KR-internal content hashes keep their earlier values under explicit names.
        "bundle_manifest_content_sha256": inner["bundle_manifest_sha256"],
        "native_prepare_manifest_content_sha256": inner["prepared_manifest_sha256"],
        "input_sha256": context.input_sha256,
        "native_prepare_manifest_sha256": context.native_manifest_sha256,
        "code_inventory_sha256": context.code_sha256,
        "freshness": context.freshness}
    validate_report(report)
    return report


def infer_us_model(context: InferenceContext) -> dict[str, Any]:
    """Score one US model from the pinned feature artifact without querying the lake."""
    if context.market != "US" or context.freshness_status not in {"ok", "stale"}:
        raise ValueError("US adapter received an invalid market or freshness state")
    from modeler.serving import us_daily

    selection = context.selection
    native = context.native_preparation
    if (native.get("market") != "US"
            or native.get("feature_asof_date") != context.feature_asof_date
            or native.get("scoring_date") != context.feature_asof_date
            or native.get("code_hash") != us_daily.code_tree_hash()
            or selection.get("market") != "US"
            or selection.get("actual_us_session") != context.feature_asof_date
            or selection.get("native_prepare_manifest_sha256") != context.native_manifest_sha256
            or selection.get("input_sha256") != context.input_sha256):
        raise ValueError("US prepared/native selection identity does not match the pinned input")
    model_path = context.bundle_manifest.parent / "model.joblib"
    model, manifest = us_daily.load_model(model_path, context.bundle_manifest)
    variant = context.model_id.rsplit("_", 1)[-1]
    expected_id = f"{us_daily.MODEL_ID}_{variant}"
    if (context.model_id != expected_id or manifest.get("model_id") != us_daily.MODEL_ID
            or manifest.get("variant") != variant
            or str(manifest.get("model_version")) != context.model_version
            or context.bundle_manifest.parent.name != variant
            or variant not in {"lightgbm", "ridge"}):
        raise ValueError("US bundle identity differs from the pinned job")
    features = pl.read_parquet(context.prepared_input)
    if "date" not in features.columns or set(features.get_column("date").unique().to_list()) != {
        date.fromisoformat(context.feature_asof_date)
    }:
        raise ValueError("US scoring input must contain exactly the declared feature session")
    label_prefixes = ("y_", "raw_label", "fwd_ret_", "bench_ret_")
    if any(name.startswith(label_prefixes) for name in features.columns):
        raise ValueError("US scoring input contains label or forward-return columns")
    ranked = us_daily.rank_daily(model["model"], features)
    report = report_template(
        market="US", report_date=context.report_date.isoformat(),
        decision_at=context.decision_at, feature_asof_date=context.feature_asof_date,
        model_id=context.model_id, model_version=context.model_version)
    report.update(
        status=context.freshness_status,
        rankings=[{"rank": int(row["rank"]), "symbol": row["symbol"],
                   "name": row["symbol"], "score": float(row["score"])}
                  for row in ranked.iter_rows(named=True)],
        quality={"freshness_status": context.freshness_status,
                 "feature_count": len(us_daily.FEATURES),
                 "design_column_count": len(us_daily.DESIGN_COLUMNS),
                 "eligible_rows": ranked.height,
                 "monthly_membership_month": date.fromisoformat(context.feature_asof_date).strftime("%Y-%m"),
                 "latest_us_session": context.freshness.get("latest_us_session"),
                 "expected_us_session": context.freshness.get("expected_us_session"),
                 "actual_us_session": context.freshness.get("actual_us_session"),
                 "delivery_lag_sessions": context.freshness.get(
                     "delivery_lag_sessions", context.freshness.get("delivery_lag")),
                 "market_lag_sessions": context.freshness.get(
                     "market_lag_sessions", context.freshness.get("market_lag")),
                 "market_lag_limit_sessions": context.freshness.get(
                     "market_lag_limit_sessions", context.selection.get("market_lag_limit_sessions"))},
        provenance={"scoring_version": us_daily.SCORING_VERSION,
                    "input_sha256": context.input_sha256,
                    "bundle_manifest_sha256": context.bundle_sha256,
                    "native_prepare_manifest_sha256": context.native_manifest_sha256,
                    "code_inventory_sha256": context.code_sha256,
                    "freshness": context.freshness,
                    "score_semantics": "raw model ranking score; not a probability or confidence",
                    "display_name_source": "symbol fallback; company-name mapping is not part of scoring",
                    "labels_read": False},
        synthetic_fixture=context.fixture_mode)
    validate_report(report)
    return report


def model_entrypoint(context: InferenceContext) -> dict[str, Any]:
    """Config-friendly dispatcher; market/model identity remains pinned by the job."""
    if context.market == "KR":
        return infer_kr_daily(context)
    if context.market == "US":
        return infer_us_model(context)
    raise ValueError("unsupported inference market")
