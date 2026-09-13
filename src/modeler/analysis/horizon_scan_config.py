"""Load and validate the preregistered horizon-scan contract."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).with_name("horizon_scan_config.yaml")
SCAN_ENGINES = ("legacy", "polars_native_v1")
DEFAULT_SCAN_ENGINE = "polars_native_v1"


@dataclass(frozen=True)
class HorizonScanConfig:
    raw: dict[str, Any]
    config_hash: str
    path: Path

    @property
    def families(self) -> list[dict[str, Any]]:
        return list(self.raw["families"])

    @property
    def horizons(self) -> list[int]:
        return list(self.raw["horizons"])

    @property
    def buckets(self) -> list[list[int]]:
        return list(self.raw["buckets"])

    @property
    def primary_hypothesis_count(self) -> int:
        return sum(
            len(f["primary_horizon_set"]) + len(bucket_primary_cells(f, self.buckets))
            for f in self.families
            if f.get("fdr_include", False)
        )


def bucket_primary_cells(family: dict[str, Any], buckets: list[list[int]]) -> list[tuple[int, int]]:
    """Buckets whose end falls in the family's primary cumulative horizon set.

    §2.1: "bucket은 h_end가 family의 primary horizon에 포함될 때만 해당 family의 주
    검정 셀로 간주한다" — bucket count is NOT simply the cumulative horizon count;
    it is the intersection of the fixed bucket grid's ends with the family's
    preregistered primary horizons (e.g. px_reversal_5d's {1,2,3,5,10} only meets
    the grid at {5,10}, giving 2 bucket cells, not 5).
    """
    if not family.get("include_bucket_primary", True):
        return []
    horizon_set = set(family["primary_horizon_set"])
    return [(b[0], b[1]) for b in buckets if b[1] in horizon_set]


def _canonical_hash(raw: dict[str, Any]) -> str:
    # The kernel is a runtime implementation choice, not a statistical
    # contract. Keeping it out of the config hash lets legacy/native parity
    # runs use the same replicate seeds and permutation mappings while the
    # selected engine is still recorded in run_spec/checkpoint metadata.
    hash_raw = dict(raw)
    execution = raw.get("execution")
    if isinstance(execution, dict) and "scan_engine" in execution:
        hash_raw["execution"] = {
            key: value for key, value in execution.items() if key != "scan_engine"
        }

    def jsonable(value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [jsonable(v) for v in value]
        return value

    encoded = json.dumps(
        jsonable(hash_raw), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_scan_engine(scan_engine: str) -> str:
    if scan_engine not in SCAN_ENGINES:
        raise ValueError(f"unknown scan engine: {scan_engine!r}; expected one of {SCAN_ENGINES}")
    return scan_engine


_PHASE_A_FIXED_PROTOCOL_VALUES = {
    ("discovery", "universe"): "broad",
    ("discovery", "sample_kind"): "common_survivor",
    ("stats", "nw_p_value_distribution"): "asymptotic_normal",
    ("stats", "bh_missing_p_value"): 1.0,
    ("stats", "nonoverlap_all_offsets"): True,
    ("stats", "nonoverlap_inference"): "exact_sign_test",
    ("placebo", "cross_sectional_block"): "date_market",
    ("placebo", "temporal_min_nw_lag"): 59,
}
_EVIDENCE_GRADE_KEYS = {"evaluation_order", "A", "B", "C", "D", "R"}

# 04_specific_plan_B.md §2.1/§2.4 — frozen before any Phase B outcome is computed.
_PHASE_B_FIXED_PROTOCOL_VALUES = {
    "phase_a_primary_count": 75,
    "readiness_freeze_before_label_join": True,
    "preflight_blocked_role": "blocked_exploratory",
    "post_freeze_blocked_p_for_bh": 1.0,
    "receipt_value_pairing_required": "verified_same_receipt",
    "receipt_value_pairing_error_tolerance": 0,
}
_PHASE_B_EVENT_BUCKETS = [[0, 3], [3, 5], [5, 10], [10, 20], [20, 40], [40, 60]]

# 01_design/03_stage1b §3.2 — the conditional-IC contract's own frozen shape.
_PHASE_C_PRIMARY_PAIR_COUNT = 15
_PHASE_C_REFERENCE_PAIR_COUNT = 2
_PHASE_C_PAIR_ROLES = {"primary", "reference"}


def _validate_phase_c(raw: dict[str, Any]) -> None:
    """Check the Phase C block, when one carries preregistered pairs.

    The base config and the expansion overlay both have a ``phase_c`` block
    with nothing but an open policy and dormant N8 regime candidates; only a
    layer that actually registers ``pairs`` is a conditional-IC contract, and
    only that one is checked here.

    What these checks protect is the BH population. ``m`` for Phase C is
    exactly the 15 primary pairs — a pair silently dropped by a typo'd family
    name, or an extra one added later, changes every q-value in the run. The
    same applies to the regimes: a pair naming a regime that does not exist
    would be scanned against nothing.
    """
    phase_c = raw.get("phase_c")
    if not isinstance(phase_c, dict) or "pairs" not in phase_c:
        return

    registered = raw.get("families", [])
    horizons_by_family: dict[str, set[int]] = {
        f["family"]: set(f.get("primary_horizon_set", [])) for f in registered
    }

    regimes = phase_c.get("regimes", [])
    regime_ids = [r.get("id") for r in regimes]
    if len(set(regime_ids)) != len(regime_ids) or any(not rid for rid in regime_ids):
        raise ValueError("phase_c.regimes ids must be unique and non-empty")
    regime_id_set = set(regime_ids)

    pairs = phase_c["pairs"]
    pair_ids = [p.get("id") for p in pairs]
    if len(set(pair_ids)) != len(pair_ids) or any(not pid for pid in pair_ids):
        raise ValueError("phase_c.pairs ids must be unique and non-empty")

    role_counts = {
        role: sum(1 for p in pairs if p.get("role") == role) for role in _PHASE_C_PAIR_ROLES
    }
    unknown_roles = sorted({p.get("role") for p in pairs} - _PHASE_C_PAIR_ROLES)
    if unknown_roles:
        raise ValueError(f"phase_c.pairs has unknown roles: {unknown_roles}")
    if role_counts["primary"] != _PHASE_C_PRIMARY_PAIR_COUNT:
        raise ValueError(
            f"phase_c must register exactly {_PHASE_C_PRIMARY_PAIR_COUNT} primary pairs, "
            f"got {role_counts['primary']}"
        )
    if role_counts["reference"] != _PHASE_C_REFERENCE_PAIR_COUNT:
        raise ValueError(
            f"phase_c must register exactly {_PHASE_C_REFERENCE_PAIR_COUNT} reference pairs, "
            f"got {role_counts['reference']}"
        )

    def _check_cell(where: str, entry: dict[str, Any]) -> None:
        family = entry.get("family")
        if family not in horizons_by_family:
            raise ValueError(f"{where}: unknown family {family!r}")
        regime = entry.get("regime")
        if regime not in regime_id_set:
            raise ValueError(f"{where}: unknown regime {regime!r}")
        cell = entry.get("cell", {})
        if cell.get("scan_type") != "cum":
            raise ValueError(f"{where}: only cumulative cells are registered, got {cell!r}")
        # §3.1: the cell is picked from the family's own preregistered primary
        # horizons, so a Phase C result can never rest on a horizon Phase A/B
        # never judged.
        if cell.get("h_end") not in horizons_by_family[family]:
            raise ValueError(
                f"{where}: h_end={cell.get('h_end')} is not in {family}'s primary_horizon_set "
                f"{sorted(horizons_by_family[family])}"
            )

    for pair in pairs:
        _check_cell(f"phase_c.pairs[{pair['id']}]", pair)
        if pair.get("direction") not in {"+", "-", None}:
            raise ValueError(f"phase_c.pairs[{pair['id']}]: direction must be '+', '-' or null")
    for index, entry in enumerate(phase_c.get("exploratory_grid", {}).get("extra", [])):
        _check_cell(f"phase_c.exploratory_grid.extra[{index}]", entry)

    # N8's dormant regime candidates are a different list with a different
    # role; a pair pointing at one would quietly open a regime the expansion
    # round deliberately left closed (00_overview §1.4).
    candidate_codes = {c.get("feature_code") for c in phase_c.get("regime_candidates", [])}
    overlap = candidate_codes & regime_id_set
    if overlap:
        raise ValueError(
            f"phase_c.regime_candidates must stay dormant, but {sorted(overlap)} is also a regime"
        )

    if not isinstance(phase_c.get("sample_start"), date):
        raise ValueError("phase_c.sample_start must be a date")
    if not isinstance(phase_c.get("placebo", {}).get("seed"), int):
        raise ValueError("phase_c.placebo.seed must be an integer")


def validate_config(raw: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "horizons",
        "buckets",
        "families",
        "quality",
        "sample",
        "stats",
        "decision",
        "placebo",
        "discovery",
        "evidence_grade",
        "phase_b",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"horizon scan config missing keys: {sorted(missing)}")
    if raw["horizons"] != [1, 2, 3, 5, 10, 20, 40, 60, 120]:
        raise ValueError("horizon grid differs from the preregistered A0 grid")
    if raw["buckets"] != [[0, 5], [5, 10], [10, 20], [20, 40], [40, 60], [60, 120]]:
        raise ValueError("bucket grid differs from the preregistered A0 grid")
    for (section, key), expected in _PHASE_A_FIXED_PROTOCOL_VALUES.items():
        actual = raw.get(section, {}).get(key)
        if actual != expected:
            raise ValueError(
                f"{section}.{key} must be the fixed Phase A protocol value {expected!r}, "
                f"got {actual!r}"
            )
    missing_grade_keys = _EVIDENCE_GRADE_KEYS - raw["evidence_grade"].keys()
    if missing_grade_keys:
        raise ValueError(f"evidence_grade missing keys: {sorted(missing_grade_keys)}")
    if raw["evidence_grade"]["evaluation_order"] != ["R", "C", "A", "B", "D"]:
        raise ValueError("evidence_grade.evaluation_order must check R/C before A/B/D")
    schema_version = int(raw["schema_version"])
    if schema_version not in {4, 5}:
        raise ValueError(f"unsupported horizon scan schema_version: {schema_version}")
    for key, expected in _PHASE_B_FIXED_PROTOCOL_VALUES.items():
        actual = raw["phase_b"].get(key)
        if actual != expected:
            raise ValueError(
                f"phase_b.{key} must be the fixed Phase B protocol value {expected!r}, "
                f"got {actual!r}"
            )
    if raw["phase_b"].get("event_buckets") != _PHASE_B_EVENT_BUCKETS:
        raise ValueError("phase_b.event_buckets differs from the preregistered event grid")
    family_names = {f.get("family") for f in raw["families"]}
    sparse_families = raw["stats"].get("sparse_primary_grid_families", [])
    unknown_sparse = [f for f in sparse_families if f not in family_names]
    if unknown_sparse:
        raise ValueError(
            f"sparse_primary_grid_families references unknown families: {unknown_sparse}"
        )
    families = raw["families"]
    if not isinstance(families, list):
        raise ValueError("families must be a list")
    if schema_version == 4 and len(families) != 25:
        raise ValueError("schema v4 must register exactly 25 families")
    if schema_version == 5 and len(families) <= 25:
        raise ValueError("schema v5 expansion must append at least one family")
    names = [f.get("family") for f in families]
    if len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("family names must be unique and non-empty")
    for family in families:
        features = family.get("features", [])
        primary = [f for f in features if f.get("role") == "primary"]
        if len(primary) != 1:
            raise ValueError(f"{family['family']}: exactly one primary feature is required")
        for key in ("primary_horizon_set", "exploratory_horizon_set", "official_feature_variant"):
            if key not in family:
                raise ValueError(f"{family['family']}: missing {key}")
        if "expected_sign" not in family:
            raise ValueError(f"{family['family']}: expected_sign must be explicit")
        if set(family["primary_horizon_set"]) & set(family["exploratory_horizon_set"]):
            raise ValueError(f"{family['family']}: primary and exploratory horizons overlap")
        all_h = set(family["primary_horizon_set"]) | set(family["exploratory_horizon_set"])
        if not all_h.issubset(set(raw["horizons"])):
            raise ValueError(f"{family['family']}: horizon outside fixed grid")
        if family.get("fdr_include") and family.get("role") == "reference_only":
            raise ValueError(f"{family['family']}: reference family cannot enter FDR")
        mapping = family.get("variant_columns", {})
        if not {"native_t", "lag1"}.issubset(mapping):
            raise ValueError(f"{family['family']}: native_t and lag1 mappings are required")
        if "event_buckets" in family and family["event_buckets"] != _PHASE_B_EVENT_BUCKETS:
            raise ValueError(f"{family['family']}: event_buckets differs from the frozen grid")
        if (
            schema_version == 4
            and family.get("role") == "ready"
            and family.get("family") != "flow_individual_netbuy_to_volume"
            and family.get("expected_sign") not in {"+", "-"}
        ):
            raise ValueError(f"{family['family']}: ready families require expected_sign '+' or '-'")
    short = [f for f in families if f.get("role") == "exploratory_short_regime"]
    if len(short) != 4:
        raise ValueError("exactly four short-selling families must be exploratory")
    if sum(1 for f in families if f.get("phase") == "A") != 17:
        raise ValueError("Phase A must contain 17 families")
    phase_b_family_count = sum(1 for f in families if f.get("phase") == "B")
    if schema_version == 4 and phase_b_family_count != 8:
        raise ValueError("schema v4 Phase B must contain 8 families")
    if schema_version == 5 and phase_b_family_count <= 8:
        raise ValueError("schema v5 Phase B must append at least one family")
    fdr_count = sum(
        len(f["primary_horizon_set"]) + len(bucket_primary_cells(f, raw["buckets"]))
        for f in families
        if f.get("fdr_include", False)
    )
    phase_a_primary_count = int(raw["phase_b"]["phase_a_primary_count"])
    if fdr_count != phase_a_primary_count:
        raise ValueError(
            f"global primary hypothesis count must be {phase_a_primary_count}, got {fdr_count}"
        )
    short_count = sum(
        len(f["primary_horizon_set"]) + len(bucket_primary_cells(f, raw["buckets"])) for f in short
    )
    if short_count != 28:
        raise ValueError(f"short exploratory cell count must be 28, got {short_count}")
    phase_b_families = [f for f in families if f.get("phase") == "B"]
    phase_b_count = sum(
        (
            len(f["event_buckets"])
            if "event_buckets" in f
            else len(f["primary_horizon_set"]) + len(bucket_primary_cells(f, raw["buckets"]))
        )
        for f in phase_b_families
    )
    max_candidates = raw["phase_b"].get("primary_candidate_count_max")
    if not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("phase_b.primary_candidate_count_max must be a positive integer")
    if schema_version == 4 and max_candidates != 38:
        raise ValueError("schema v4 phase_b.primary_candidate_count_max must be 38")
    if phase_b_count != max_candidates:
        raise ValueError(f"Phase B candidate count must be {max_candidates}, got {phase_b_count}")
    _validate_registered_at(raw)
    _validate_phase_c(raw)


def _validate_registered_at(raw: dict[str, Any]) -> None:
    """A preregistration layer's date must be a real date, not a placeholder.

    ``registered_at`` goes into the config hash, so committing a layer that
    still reads "TBD" (or an ISO string YAML never parsed) would freeze a
    placeholder into the contract's identity (review M7).
    """
    prereg = raw.get("preregistration")
    if not isinstance(prereg, dict):
        return
    if not isinstance(prereg.get("registered_at"), date):
        raise ValueError(
            "preregistration.registered_at must be a date, got " f"{prereg.get('registered_at')!r}"
        )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key in {"extends", "families_append"}:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    appended = overlay.get("families_append", [])
    if appended:
        if not isinstance(appended, list):
            raise ValueError("families_append must be a list")
        merged["families"] = [*merged.get("families", []), *deepcopy(appended)]
    return merged


def _load_raw(path: Path, *, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"horizon scan config extends cycle: {resolved}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("horizon scan YAML root must be an object")
    parent = raw.get("extends")
    if parent is None:
        return raw
    parent_path = (path.parent / str(parent)).resolve()
    base = _load_raw(parent_path, seen=seen | {resolved})
    return _deep_merge(base, raw)


def load_config(path: Path | str = CONFIG_PATH) -> HorizonScanConfig:
    path = Path(path)
    raw = _load_raw(path)
    validate_config(raw)
    return HorizonScanConfig(raw=raw, config_hash=_canonical_hash(raw), path=path)
