"""Pure references to a preregistered P6 aggregate check, never an evaluation job.

The frozen producer schema is ``_03_kr_next/forward.py::build_check``. It has
no model ID or bundle-byte hash: those are assertions of the separately pinned
registration and its expected *entire check bytes* hash, not facts recovered
from the check. This module never imports the producer, reads paths, opens
points/labels, recomputes a statistic, or creates a registration on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import date, datetime
from typing import Any

from modeler.feedback_outcomes.contracts import SEOUL, OutcomeError

CHECK_SCHEMA = "kr-next-forward-check.v1"
REGISTRATION_SCHEMA = "p6-aggregate-check-reference-registration.v1"
ARTIFACT_KIND = "model_aggregate_check"
P6_MODEL_ID = "kr-next"
P6_BUNDLE_SHA256 = "ae4ffb72346d540d2380d7c9732bda16e62059c14a5e304280b89d02492c1b74"
PREREG_DOCUMENT_SHA256 = "6ef70b739b20cbd610b36964397f71ce439f8395e5acba392061aaef4c97b6c5"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REGISTRATION_KEYS = {
    "schema",
    "model_id",
    "bundle_sha256",
    "registered_check_date",
    "not_before",
    "expected_check_sha256",
    "registration_sha256",
    "allowed_artifact_kind",
}
_CHECK_KEYS = {
    "schema",
    "check_date",
    "data_date",
    "decision",
    "points",
    "missing",
    "turnover_warning",
    "requires_user_confirmation",
    "cell_stats",
    "note",
    "inputs",
}
_POINT_KEYS = {
    "grid_date",
    "matured_at",
    "g_at_check",
    "tail_diff_at_check",
    "rank_ic_at_check",
    "n_pending_at_check",
    "pending_tickers_at_check",
    "excluded_tail_at_check",
    "excluded_rest_at_check",
    "final",
    "final_revision",
    "differs_from_final",
    "g_final",
    "g_final_differs",
    "turnover",
}
_POINT_OUTPUT_KEYS = (
    "grid_date",
    "matured_at",
    "g_at_check",
    "tail_diff_at_check",
    "rank_ic_at_check",
    "n_pending_at_check",
    "excluded_tail_at_check",
    "excluded_rest_at_check",
    "turnover",
)


def _day(value: Any) -> date:
    if not isinstance(value, str):
        raise OutcomeError("check dates require canonical ISO strings")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise OutcomeError("invalid check date") from exc
    if result.isoformat() != value:
        raise OutcomeError("check dates require canonical ISO strings")
    return result


def _instant(value: Any) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise OutcomeError("check observation requires an ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise OutcomeError("check observation requires a timezone")
    return result


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise OutcomeError("check registration requires lowercase SHA-256 hashes")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutcomeError("duplicate keys in check JSON")
        result[key] = value
    return result


def _invalid_constant(value):
    raise OutcomeError(f"non-finite check JSON constant: {value}")


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise OutcomeError("non-finite check JSON number")
    if isinstance(value, dict):
        for nested in value.values():
            _finite_tree(nested)
    elif isinstance(value, list):
        for nested in value:
            _finite_tree(nested)


def _aggregate_keys(check: dict) -> None:
    """Do not let unregistered security results hide inside aggregate containers."""
    for field in ("significance_g", "significance_tail"):
        significance = check["decision"][field]
        if not isinstance(significance, dict) or set(significance) != {"n24", "n36"}:
            raise OutcomeError("significance cells differ from the frozen aggregate schema")
        for cell in significance.values():
            if not isinstance(cell, dict) or set(cell) not in (
                {"status"},
                {"status", "n_skipped_null", "n", "mean", "sd", "t", "p", "significant"},
            ):
                raise OutcomeError("unregistered significance values")
    stats = check["cell_stats"]
    if set(stats) != {"g", "tail_diff", "rank_ic", "record_only"}:
        raise OutcomeError("unregistered aggregate statistic cells")
    for name, keys in {
        "g": {"n", "mean", "power_base", "power_scenario", "lower_975_one_sided"},
        "tail_diff": {"n", "mean", "power_base", "power_scenario", "upper_975_one_sided"},
        "rank_ic": {"n", "mean", "ci95_two_sided"},
    }.items():
        if not isinstance(stats[name], dict) or set(stats[name]) != keys:
            raise OutcomeError("unregistered aggregate statistic values")
        if name != "rank_ic":
            scenario = stats[name]["power_scenario"]
            if not isinstance(scenario, dict) or set(scenario) != {"effect", "sd", "alpha", "test"}:
                raise OutcomeError("unregistered power scenario values")


def _check_shape(check: Any, expected_date: date) -> None:
    if not isinstance(check, dict) or set(check) != _CHECK_KEYS:
        raise OutcomeError("check keys differ from the frozen producer schema")
    if check["schema"] != CHECK_SCHEMA or _day(check["check_date"]) != expected_date:
        raise OutcomeError("check schema or registered date mismatch")
    data_date = _day(check["data_date"])
    if data_date > expected_date:
        raise OutcomeError("check data_date exceeds its registered check date")
    decision_keys = {
        "check_date",
        "n_matured",
        "mean_g",
        "mean_rank_ic",
        "mean_tail_diff",
        "alpha",
        "model",
        "stop_proposal",
        "tail",
        "significance_g",
        "significance_tail",
    }
    if (
        not isinstance(check["decision"], dict)
        or set(check["decision"]) != decision_keys
        or check["decision"].get("check_date") != check["check_date"]
    ):
        raise OutcomeError("aggregate decision check date mismatch")
    if (
        not isinstance(check["cell_stats"], dict)
        or check["cell_stats"].get("record_only") is not True
    ):
        raise OutcomeError("check cell_stats must remain record-only")
    _aggregate_keys(check)
    if not isinstance(check["points"], list) or not isinstance(check["missing"], list):
        raise OutcomeError("check grid records must be lists")
    seen = set()
    for point in check["points"]:
        if not isinstance(point, dict) or set(point) != _POINT_KEYS:
            raise OutcomeError("grid point keys differ from the frozen producer schema")
        grid_date = _day(point["grid_date"])
        matured_at = _day(point["matured_at"])
        if not grid_date < matured_at <= expected_date or point["grid_date"] in seen:
            raise OutcomeError("grid point date is duplicated, immature, or invalid")
        seen.add(point["grid_date"])
    for missing in check["missing"]:
        if not isinstance(missing, dict) or set(missing) != {"grid_date", "reason"}:
            raise OutcomeError("invalid missing grid record")
        if _day(missing["grid_date"]) > expected_date or missing["grid_date"] in seen:
            raise OutcomeError("invalid or duplicated missing grid date")
        seen.add(missing["grid_date"])
    inputs = check["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "bundle_tag_time",
        "prereg_tag",
        "ledger_lines",
        "code_inventory_sha256",
    }:
        raise OutcomeError("check inputs differ from the frozen producer schema")
    if inputs["prereg_tag"] != "kr-next-frozen":
        raise OutcomeError("check belongs to another preregistration")
    _instant(inputs["bundle_tag_time"])
    _hash(inputs["code_inventory_sha256"])
    _finite_tree(check)


def link_registered_check(
    registration: dict, check_raw: bytes | None, *, observed_at: str | datetime
) -> dict:
    """Link exact already-produced check bytes to an explicit registration, with no I/O.

    ``registration_sha256`` pins the *source preregistration document bytes*, not
    a newly created registration or a self-authenticating JSON hash. The check
    date, availability cutoff and expected bytes hash must be explicitly supplied
    from a separately approved registration. No actual date or check hash is
    supplied by this module. Only synthetic registrations are used by its tests.

    Before ``not_before`` this function neither hashes nor parses ``check_raw``.
    Later it preserves the producer's aggregate at-check values, without exposing
    pending tickers or replacing them with the check's later ``final`` snapshot.
    It accepts no points file, revision override, price, label, or arbitrary path.
    """
    if not isinstance(registration, dict) or set(registration) != _REGISTRATION_KEYS:
        raise OutcomeError("registration keys differ from the explicit reference contract")
    if (
        registration["schema"] != REGISTRATION_SCHEMA
        or registration["model_id"] != P6_MODEL_ID
        or registration["bundle_sha256"] != P6_BUNDLE_SHA256
        or registration["allowed_artifact_kind"] != ARTIFACT_KIND
        or registration["registration_sha256"] != PREREG_DOCUMENT_SHA256
    ):
        raise OutcomeError("registration does not match the pinned P6 identity or preregistration")
    expected_hash = _hash(registration["expected_check_sha256"])
    check_date = _day(registration["registered_check_date"])
    available = _instant(registration["not_before"])
    observed = _instant(observed_at)
    if available.astimezone(SEOUL).date() < check_date:
        raise OutcomeError("not_before precedes the registered check date")
    result = {
        "kind": ARTIFACT_KIND,
        "per_security_outcome": False,
        "model_id": P6_MODEL_ID,
        "bundle_sha256": P6_BUNDLE_SHA256,
        "registered_check_date": check_date.isoformat(),
        "registration_sha256": PREREG_DOCUMENT_SHA256,
        "expected_check_sha256": expected_hash,
        "status": "awaiting_check",
        "check_sha256": None,
        "values": None,
        "model_binding": "registration_and_exact_check_bytes",
        "model_binding_in_check": False,
        "recomputed": False,
    }
    if observed < available:
        return {**result, "reason": "registered_time_not_reached"}
    if check_raw is None:
        return {**result, "reason": "check_not_supplied"}
    if not isinstance(check_raw, bytes):
        raise OutcomeError("check artifact must be exact bytes")
    digest = hashlib.sha256(check_raw).hexdigest()
    if digest != expected_hash:
        raise OutcomeError("check byte hash differs from the registration")
    try:
        check = json.loads(check_raw, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise OutcomeError("invalid check JSON bytes") from exc
    _check_shape(check, check_date)
    values = {
        key: deepcopy(check[key])
        for key in (
            "check_date",
            "data_date",
            "decision",
            "missing",
            "turnover_warning",
            "requires_user_confirmation",
            "cell_stats",
            "note",
            "inputs",
        )
    }
    values["points"] = [
        {key: deepcopy(point[key]) for key in _POINT_OUTPUT_KEYS} for point in check["points"]
    ]
    return {**result, "status": "linked", "reason": None, "check_sha256": digest, "values": values}
