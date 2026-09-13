"""Phase C: preregistered conditional-IC tests (Stage 1b).

Design: ``docs/dev/20260829_macro_features/01_design/03_stage1b_conditional_ic_phase_c.md``.
Contract: ``research/analysis/horizon_scan_macro_20260829.yaml`` ``phase_c``
(hash recorded in ``05_preregistration_record.md``).

The question is not "does this feature predict returns" — Phase A and B
answered that — but "does its rank IC differ between two market regimes". For
one preregistered (family, cell, regime) pair::

    IC_t = alpha + delta * s_t + e_t

so ``delta`` is the difference of the two conditional mean ICs, estimated with
the same gap-aware HAC the cell's own ``t_nw`` uses.

**This phase reads files. It never rebuilds the panel.** Its inputs are two
published runs' ``daily_ic.parquet`` (Stage 0) and a regime series built on the
KRX session grid. That is what makes it cheap enough to rerun and impossible
for it to quietly disagree with the run it is conditioning.

**G4 is the load-bearing gate.** A regime holds the same value for tens of
sessions (``liq_high``: ~60 on the real sample) and ``IC_t`` is itself strongly
autocorrelated, so two unrelated persistent series can produce a large
``delta`` by chance. HAC only corrects within its lag — 19 for an h=20 cell —
which is a third of that. The circular-shift placebo preserves both series'
autocorrelation while destroying their alignment, and it is required for every
pair regardless of horizon (§5.1).

Which A/B runs supply the daily IC is a CLI argument, not config: a run id is
an execution fact, and putting it in the contract would force a hash-exclusion
rule (§7.1). It is recorded in ``phase_c_run_spec.json`` with each file's
sha256 instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_daily_ic import DAILY_IC_DIR_NAME
from modeler.analysis.horizon_scan_phase_c_regimes import REGIME_IDS
from modeler.etl.metrics import (
    benjamini_hochberg,
    choose_nw_lag,
    newey_west_ols,
)

PHASE_C_CONTRACT = "conditional_ic_v1"
CONDITIONAL_IC_TABLE = "conditional_ic"
PLACEBO_SUMMARY_TABLE = "regime_placebo_summary"
SUBPERIOD_TABLE = "subperiod_conditional_ic"
PHASE_C_RUN_SPEC_NAME = "phase_c_run_spec.json"
PHASE_C_REPORT_NAME = "03c_conditional_ic_results.md"

DISCOVERY_UNIVERSE = "broad"
RETENTION_UNIVERSE = "tradable"
DISCOVERY_SAMPLE_KIND = "common_survivor"

# The placebo's seed key. Same construction as
# `horizon_scan_mapping.mapping_seed_sequence`, but its own key — that one is
# keyed on (trade_date, market), which has no meaning here (review §6.6).
PLACEBO_SEED_CONTRACT = "regime_shift_v1"


@dataclass(frozen=True)
class PairSpec:
    """One preregistered (family, cell, regime) test."""

    pair_id: str
    role: str
    family: str
    scan_type: str
    h_start: int
    h_end: int
    regime_id: str
    direction: str | None

    @property
    def nw_lag(self) -> int:
        return choose_nw_lag(
            scan_type=self.scan_type,
            horizon=self.h_end if self.scan_type == "cum" else None,
            bucket_width=(self.h_end - self.h_start) if self.scan_type == "bucket" else None,
        )


def load_pairs(config_raw: dict[str, Any]) -> list[PairSpec]:
    """The registered pairs, in contract order. ``validate_config`` has already
    checked every family/cell/regime reference by the time this runs."""
    return [
        PairSpec(
            pair_id=pair["id"],
            role=pair["role"],
            family=pair["family"],
            scan_type=pair["cell"]["scan_type"],
            h_start=int(pair["cell"]["h_start"]),
            h_end=int(pair["cell"]["h_end"]),
            regime_id=pair["regime"],
            direction=pair.get("direction"),
        )
        for pair in config_raw["phase_c"]["pairs"]
    ]


# --- inputs ---


def load_daily_ic(run_dirs: list[Path]) -> pl.DataFrame:
    """Read the Stage 0 daily IC of one or more published runs.

    Phase A keeps it under ``core/``, Phase B at the run root; both are
    accepted so one call covers a (Phase A run, Phase B run) pair. The
    concatenated frame keeps its ``family``/``universe``/``sample_kind``
    identity columns, which is what the pair lookup below joins on.
    """
    frames: list[pl.DataFrame] = []
    for run_dir in run_dirs:
        for root in (run_dir / DAILY_IC_DIR_NAME, run_dir / "core" / DAILY_IC_DIR_NAME):
            files = (
                sorted(p for p in root.rglob("*.parquet") if p.is_file()) if root.is_dir() else []
            )
            frames.extend(pl.read_parquet(path) for path in files)
    if not frames:
        raise FileNotFoundError(f"no daily_ic.parquet found under {[str(d) for d in run_dirs]}")
    return pl.concat(frames, how="vertical")


def daily_ic_sha256(run_dirs: list[Path]) -> dict[str, str]:
    """One content hash per run directory's daily IC tree, for the run spec."""
    digests: dict[str, str] = {}
    for run_dir in run_dirs:
        hasher = hashlib.sha256()
        for root in (run_dir / DAILY_IC_DIR_NAME, run_dir / "core" / DAILY_IC_DIR_NAME):
            if not root.is_dir():
                continue
            for path in sorted(p for p in root.rglob("*.parquet") if p.is_file()):
                hasher.update(path.relative_to(root).as_posix().encode())
                hasher.update(path.read_bytes())
        digests[run_dir.name] = hasher.hexdigest()
    return digests


def select_cell_ic(
    daily_ic: pl.DataFrame, pair: PairSpec, *, universe: str, sample_kind: str
) -> pl.DataFrame:
    """The one daily IC series a pair conditions on.

    A family can have several cells in ``daily_ic``; the pair names exactly one
    (§3.1), and it must resolve to a single hypothesis or the contract is
    ambiguous.
    """
    frame = daily_ic.filter(
        (pl.col("family") == pair.family)
        & (pl.col("scan_type") == pair.scan_type)
        & (pl.col("h_start") == pair.h_start)
        & (pl.col("h_end") == pair.h_end)
        & (pl.col("universe") == universe)
        & (pl.col("sample_kind") == sample_kind)
    )
    hypotheses = frame["hypothesis_id"].unique().to_list() if frame.height else []
    if len(hypotheses) > 1:
        raise ValueError(
            f"{pair.pair_id}: cell resolves to {len(hypotheses)} hypotheses {hypotheses}"
        )
    return frame.sort("trade_date")


# --- §4 estimation ---


def _empty_result(pair: PairSpec, universe: str, reason: str) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "family": pair.family,
        "scan_type": pair.scan_type,
        "h_start": pair.h_start,
        "h_end": pair.h_end,
        "universe": universe,
        "sample_kind": DISCOVERY_SAMPLE_KIND,
        "regime_id": pair.regime_id,
        "regime_role": pair.role,
        "direction_preregistered": pair.direction,
        "nw_lag": pair.nw_lag,
        "sample_start": None,
        "sample_end": None,
        "n_dates": 0,
        "n_dates_s1": 0,
        "n_dates_s0": 0,
        "share_s1": None,
        "n_regime_transitions": None,
        "mean_run_length_s1": None,
        "mean_run_length_s0": None,
        "ic_mean_s1": None,
        "ic_mean_s0": None,
        "delta": None,
        "se_nw": None,
        "t_nw": None,
        "p_nw": None,
        "status": "insufficient",
        "status_reason": reason,
    }


def estimate_pair(
    pair: PairSpec,
    cell_ic: pl.DataFrame,
    regimes: pl.DataFrame,
    *,
    universe: str = DISCOVERY_UNIVERSE,
    sample_start: date | None = None,
    min_dates_per_regime: int = 250,
    min_share_per_regime: float = 0.20,
) -> dict[str, Any]:
    """One pair's conditional IC difference at one universe coordinate.

    G1 (occupancy) is applied here rather than downstream because a regime that
    barely varies has no ``delta`` worth reporting: the pair stays in the BH
    population at ``p=1.0`` (§5) so ``m`` never shrinks, but its estimate is
    withheld instead of being published as if it meant something.
    """
    regime_column = f"s_{pair.regime_id}"
    if regime_column not in regimes.columns:
        return _empty_result(pair, universe, f"unknown_regime:{pair.regime_id}")
    if cell_ic.is_empty():
        return _empty_result(pair, universe, "no_daily_ic")

    joined = cell_ic.select(["trade_date", "formation_session_idx", "rank_ic"]).join(
        regimes.select(["trade_date", regime_column]), on="trade_date", how="inner"
    )
    joined = joined.filter(pl.col(regime_column).is_not_null())
    if sample_start is not None:
        joined = joined.filter(pl.col("trade_date") >= sample_start)
    joined = joined.sort("formation_session_idx")
    if joined.is_empty():
        return _empty_result(pair, universe, "no_overlapping_dates")

    flags = joined[regime_column].to_numpy().astype(bool)
    values = joined["rank_ic"].to_numpy()
    sessions = joined["formation_session_idx"].to_numpy()
    n_dates = int(flags.size)
    n_s1 = int(flags.sum())
    n_s0 = n_dates - n_s1
    share = n_s1 / n_dates

    result = _empty_result(pair, universe, "")
    persistence = _persistence(flags)
    result.update(
        {
            "sample_start": joined["trade_date"].min(),
            "sample_end": joined["trade_date"].max(),
            "n_dates": n_dates,
            "n_dates_s1": n_s1,
            "n_dates_s0": n_s0,
            "share_s1": share,
            **persistence,
        }
    )
    if (
        n_s1 < min_dates_per_regime
        or n_s0 < min_dates_per_regime
        or not (min_share_per_regime <= share <= 1 - min_share_per_regime)
    ):
        result["status_reason"] = f"g1_occupancy:s1={n_s1},s0={n_s0},share={share:.3f}"
        return result

    fit = newey_west_ols(values, flags.astype(float), sessions, pair.nw_lag)
    result.update(
        {
            "ic_mean_s1": float(values[flags].mean()),
            "ic_mean_s0": float(values[~flags].mean()),
            "delta": fit["delta"],
            "se_nw": fit["se_delta"],
            "t_nw": fit["t_delta"],
            "p_nw": fit["p_delta"],
            "status": "valid" if math.isfinite(fit["t_delta"]) else "insufficient",
            "status_reason": "" if math.isfinite(fit["t_delta"]) else "degenerate_hac_variance",
        }
    )
    return result


def _persistence(flags: np.ndarray) -> dict[str, Any]:
    """§6.5: transitions and mean run length, on the pair's own sample."""
    values = flags.tolist()
    transitions = sum(1 for a, b in zip(values, values[1:], strict=False) if a != b)
    runs: dict[bool, list[int]] = {True: [], False: []}
    previous: bool | None = None
    for value in values:
        if value == previous:
            runs[value][-1] += 1
        else:
            runs[value].append(1)
        previous = value
    return {
        "n_regime_transitions": transitions,
        "mean_run_length_s1": (sum(runs[True]) / len(runs[True]) if runs[True] else None),
        "mean_run_length_s0": (sum(runs[False]) / len(runs[False]) if runs[False] else None),
    }


# --- §6.1 G4 circular-shift placebo ---


def placebo_seed(
    *, replicate_index: int, config_hash: str, pair_id: str, universe: str, base_seed: int
) -> int:
    """Seed fixed by the contract and the replicate index alone.

    Same construction as ``mapping_seed_sequence`` but its own key: that one is
    keyed on ``(trade_date, market)``, which means nothing for a date-level
    regime shift (review §6.6). Never wall-clock or run order, so a rerun
    reproduces the identical null.
    """
    key = "|".join(
        [
            PLACEBO_SEED_CONTRACT,
            str(base_seed),
            str(replicate_index),
            config_hash,
            pair_id,
            universe,
        ]
    ).encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % (2**32)


def run_regime_placebo(
    pair: PairSpec,
    values: np.ndarray,
    flags: np.ndarray,
    sessions: np.ndarray,
    *,
    config_hash: str,
    universe: str,
    repeats: int = 100,
    min_shift: int = 120,
    p_max: float = 0.10,
    base_seed: int = 20260829,
) -> dict[str, Any]:
    """Shift the regime, not the IC, and refit (§6.1).

    Rolling ``s_t`` around the sample keeps its run-length distribution and
    ``IC_t``'s autocorrelation exactly as they are, and destroys only the
    alignment between them — which is the null this test needs. Shifting the IC
    instead would answer a different question, and permuting either would
    destroy the persistence that is the whole problem.

    ``p = (1 + #{|t_k| >= |t_real|}) / (repeats + 1)``, matching Phase A's own
    minimum-p convention.
    """
    n = int(flags.size)
    if n <= 2 * min_shift:
        return {
            "placebo_p": None,
            "placebo_repeats": 0,
            "placebo_pass": None,
            "placebo_status": f"sample_too_short_for_min_shift:{n}<={2 * min_shift}",
        }
    real = newey_west_ols(values, flags.astype(float), sessions, pair.nw_lag)["t_delta"]
    if not math.isfinite(real):
        return {
            "placebo_p": None,
            "placebo_repeats": 0,
            "placebo_pass": None,
            "placebo_status": "real_t_not_finite",
        }

    at_least = 0
    shifts: list[int] = []
    for replicate in range(repeats):
        seed = placebo_seed(
            replicate_index=replicate,
            config_hash=config_hash,
            pair_id=pair.pair_id,
            universe=universe,
            base_seed=base_seed,
        )
        rng = np.random.default_rng(seed)
        shift = int(min_shift + rng.integers(0, n - 2 * min_shift + 1))
        shifts.append(shift)
        shifted = np.roll(flags, shift)
        t_shift = newey_west_ols(values, shifted.astype(float), sessions, pair.nw_lag)["t_delta"]
        if math.isfinite(t_shift) and abs(t_shift) >= abs(real):
            at_least += 1
    p = (1 + at_least) / (repeats + 1)
    return {
        "placebo_p": p,
        "placebo_repeats": repeats,
        "placebo_pass": bool(p <= p_max),
        "placebo_status": "ok",
        "placebo_shift_min": min(shifts),
        "placebo_shift_max": max(shifts),
    }


# --- §4.3 BH, §5 gates and grades ---


def apply_phase_c_bh(
    rows: list[dict[str, Any]], *, q_threshold: float = 0.10
) -> list[dict[str, Any]]:
    """One BH pass over the 15 primary pairs. Reference and exploratory pairs
    are reported but never in ``m`` — they are not part of the claim.

    An ``insufficient`` pair keeps its slot at ``p=1.0`` (§3.2), so a regime
    that failed G1 costs power without shrinking the correction.
    """
    primary = sorted((r for r in rows if r["regime_role"] == "primary"), key=lambda r: r["pair_id"])
    p_values = np.array(
        [
            (
                r["p_nw"]
                if r["status"] == "valid" and r["p_nw"] is not None and math.isfinite(r["p_nw"])
                else 1.0
            )
            for r in primary
        ]
    )
    q_values = benjamini_hochberg(p_values) if p_values.size else np.array([])
    q_by_pair = {
        r["pair_id"]: (float(q) if math.isfinite(q) else None)
        for r, q in zip(primary, q_values, strict=True)
    }

    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        q = q_by_pair.get(row["pair_id"])
        row["q_fdr_phase_c"] = q
        bh_pass = row["regime_role"] == "primary" and q is not None and q < q_threshold
        direction = row["direction_preregistered"]
        delta = row["delta"]
        if direction in ("+", "-") and delta is not None and math.isfinite(delta):
            sign = 1.0 if direction == "+" else -1.0
            row["direction_pass"] = bool(sign * delta > 0)
        else:
            row["direction_pass"] = None
        row["discovery"] = bool(
            row["status"] == "valid" and bh_pass and row["direction_pass"] is not False
        )
        out.append(row)
    return out


def compute_screen_pass(row: dict[str, Any]) -> dict[str, Any]:
    """§5: ``screen_pass = discovery ∧ G1 ∧ G2 ∧ G3 ∧ G4``.

    G1 is folded into ``status`` (an occupancy failure is ``insufficient``), so
    what remains here is the three gates computed from other coordinates:
    period consistency, tradable retention, and the regime placebo.
    """
    if row["regime_role"] != "primary":
        return {"screen_pass": False, "failed_gates": [], "not_applicable_role": True}
    checks = {
        "discovery": bool(row.get("discovery")),
        "g1_occupancy": row["status"] == "valid",
        "g2_period_sign": bool(row.get("period_sign_pass")),
        "g3_tradable_retention": bool(row.get("tradable_pass")),
        "g4_regime_placebo": bool(row.get("placebo_pass")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {"screen_pass": not failed, "failed_gates": failed, "not_applicable_role": False}


def assign_evidence_grade(row: dict[str, Any]) -> str:
    """§5's A/B/C/D/R rubric, checking role and insufficiency before pass/fail.

    A needs a clean screen *and* both non-fatal signals: at least four valid
    subperiods, and the alternative cut (G6) agreeing in sign. Neither can
    fail a pair on its own — they only separate A from B.
    """
    if row["regime_role"] == "reference":
        return "R"
    if row["regime_role"] == "exploratory" or row["status"] != "valid":
        return "C"
    if not row.get("screen_pass"):
        return "D"
    warning = (row.get("valid_subperiods") or 0) < 4 or row.get("alt_cut_sign_agree") is False
    return "B" if warning else "A"


def compute_period_sign_pass(
    subperiod_deltas: list[float | None], delta: float | None
) -> dict[str, Any]:
    """G2: valid subperiods agreeing in sign with the full-sample ``delta``,
    ceil(valid/2) or more (§5). ``None`` entries are subperiods where one side
    of the regime was too thin to estimate — neither help nor hurt.
    """
    valid = [d for d in subperiod_deltas if d is not None and math.isfinite(d)]
    if not valid or delta is None or not math.isfinite(delta) or delta == 0:
        return {
            "valid_subperiods": len(valid),
            "sign_consistent_subperiods": 0,
            "period_sign_pass": False,
        }
    consistent = sum(1 for d in valid if (d > 0) == (delta > 0))
    return {
        "valid_subperiods": len(valid),
        "sign_consistent_subperiods": consistent,
        "period_sign_pass": bool(consistent >= math.ceil(len(valid) / 2)),
    }


def compute_tradable_pass(
    *, delta_broad: float | None, delta_tradable: float | None, min_retention: float = 0.50
) -> dict[str, Any]:
    """G3: the same pair at the tradable coordinate must keep its sign and at
    least half the magnitude. A zero or non-finite broad ``delta`` cannot
    anchor a ratio, so the gate fails rather than reporting "N/A"."""
    if (
        delta_broad is None
        or delta_tradable is None
        or not math.isfinite(delta_broad)
        or not math.isfinite(delta_tradable)
        or delta_broad == 0
    ):
        return {
            "tradable_delta": delta_tradable,
            "tradable_retention": None,
            "tradable_pass": False,
        }
    retention = abs(delta_tradable) / abs(delta_broad)
    same_sign = (delta_broad > 0) == (delta_tradable > 0)
    return {
        "tradable_delta": delta_tradable,
        "tradable_retention": retention,
        "tradable_pass": bool(same_sign and retention >= min_retention),
    }


def assert_regime_columns_present(regimes: pl.DataFrame) -> None:
    """Every contract regime must be in the series, or a pair would silently
    score ``unknown_regime`` and drop out of the run's own population."""
    missing = [rid for rid in REGIME_IDS if f"s_{rid}" not in regimes.columns]
    if missing:
        raise ValueError(f"regime series is missing columns for: {missing}")


def build_phase_c_run_spec(
    *,
    config_hash: str,
    phase_c_block: dict[str, Any],
    phase_a_run_dir: Path,
    phase_b_run_dir: Path | None,
    snapshot_date: str,
    source: str,
    sessions: list[date],
    started_at: str,
    command_line: list[str],
) -> dict[str, Any]:
    """§7.1's immutable first artifact.

    The two input run ids and their ``daily_ic`` hashes live here rather than
    in the contract: a run id is an execution fact, and putting it in config
    would force a hash-exclusion rule. The session-list hash pins the grid the
    regimes were built on.
    """
    run_dirs = [d for d in (phase_a_run_dir, phase_b_run_dir) if d is not None]
    session_hash = hashlib.sha256("|".join(d.isoformat() for d in sessions).encode()).hexdigest()
    return {
        "phase": "C",
        "contract": phase_c_block.get("contract", PHASE_C_CONTRACT),
        "config_hash": config_hash,
        "phase_c_block_hash": hashlib.sha256(
            json.dumps(phase_c_block, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "snapshot_date": snapshot_date,
        "source": source,
        "phase_a_run_id": phase_a_run_dir.name,
        "phase_b_run_id": phase_b_run_dir.name if phase_b_run_dir else None,
        "daily_ic_sha256": daily_ic_sha256(run_dirs),
        "session_grid_sha256": session_hash,
        "n_sessions": len(sessions),
        "placebo_seed": phase_c_block.get("placebo", {}).get("seed"),
        "started_at": started_at,
        "command_line": command_line,
        "run_id": f"{started_at[:19].replace(':', '').replace('-', '')}-phasec",
    }


# --- orchestration ---


def run_pair(
    pair: PairSpec,
    daily_ic: pl.DataFrame,
    regimes: pl.DataFrame,
    *,
    config_hash: str,
    stats: dict[str, Any],
    placebo_cfg: dict[str, Any],
    period_sets: list[dict[str, Any]],
    placeholders: dict[str, date],
    sample_start: date | None,
) -> dict[str, Any]:
    """One pair, end to end: estimate, then every gate that needs another view.

    The tradable coordinate and the subperiod splits are re-estimations of the
    same pair on a narrower slice of the *same* stored daily IC — no rescan, no
    panel. That is what §7.1 means by "panel을 다시 만들지 않는다".
    """
    broad_ic = select_cell_ic(
        daily_ic, pair, universe=DISCOVERY_UNIVERSE, sample_kind=DISCOVERY_SAMPLE_KIND
    )
    row = estimate_pair(
        pair,
        broad_ic,
        regimes,
        universe=DISCOVERY_UNIVERSE,
        sample_start=sample_start,
        min_dates_per_regime=int(stats["min_dates_per_regime"]),
        min_share_per_regime=float(stats["min_share_per_regime"]),
    )

    tradable_ic = select_cell_ic(
        daily_ic, pair, universe=RETENTION_UNIVERSE, sample_kind=DISCOVERY_SAMPLE_KIND
    )
    tradable = estimate_pair(
        pair,
        tradable_ic,
        regimes,
        universe=RETENTION_UNIVERSE,
        sample_start=sample_start,
        min_dates_per_regime=int(stats["min_dates_per_regime"]),
        min_share_per_regime=float(stats["min_share_per_regime"]),
    )
    row.update(
        compute_tradable_pass(
            delta_broad=row["delta"],
            delta_tradable=tradable["delta"],
            min_retention=float(stats["tradable_min_abs_delta_retention"]),
        )
    )

    subperiods = subperiod_deltas(
        pair,
        broad_ic,
        regimes,
        period_sets,
        placeholders=placeholders,
        sample_start=sample_start,
        min_dates_per_regime=int(stats["subperiod_min_dates_per_regime"]),
    )
    row.update(compute_period_sign_pass([s["delta"] for s in subperiods], row["delta"]))
    row["subperiods"] = subperiods

    row.update(alt_cut_delta(pair, broad_ic, regimes, sample_start=sample_start))

    if row["status"] == "valid":
        joined = _join(broad_ic, regimes, pair.regime_id, sample_start)
        row.update(
            run_regime_placebo(
                pair,
                joined["rank_ic"].to_numpy(),
                joined[f"s_{pair.regime_id}"].to_numpy().astype(bool),
                joined["formation_session_idx"].to_numpy(),
                config_hash=config_hash,
                universe=DISCOVERY_UNIVERSE,
                repeats=int(placebo_cfg["regime_circular_shift_repeats"]),
                min_shift=int(placebo_cfg["min_shift_sessions"]),
                p_max=float(placebo_cfg["p_max"]),
                base_seed=int(placebo_cfg["seed"]),
            )
        )
    else:
        row.update(
            {
                "placebo_p": None,
                "placebo_repeats": 0,
                "placebo_pass": None,
                "placebo_status": "skipped_not_valid",
            }
        )
    return row


def _join(
    cell_ic: pl.DataFrame, regimes: pl.DataFrame, regime_id: str, sample_start: date | None
) -> pl.DataFrame:
    column = f"s_{regime_id}"
    joined = cell_ic.select(["trade_date", "formation_session_idx", "rank_ic"]).join(
        regimes.select(["trade_date", column]), on="trade_date", how="inner"
    )
    joined = joined.filter(pl.col(column).is_not_null())
    if sample_start is not None:
        joined = joined.filter(pl.col("trade_date") >= sample_start)
    return joined.sort("formation_session_idx")


def subperiod_deltas(
    pair: PairSpec,
    cell_ic: pl.DataFrame,
    regimes: pl.DataFrame,
    period_sets: list[dict[str, Any]],
    *,
    placeholders: dict[str, date],
    sample_start: date | None,
    min_dates_per_regime: int = 40,
) -> list[dict[str, Any]]:
    """G2's per-subperiod ``delta``. A window where one side of the regime is
    below the floor gets ``None`` — it neither helps nor hurts the sign count,
    because a one-sided window cannot show a sign at all."""
    joined = _join(cell_ic, regimes, pair.regime_id, sample_start)
    column = f"s_{pair.regime_id}"
    rows: list[dict[str, Any]] = []
    for period in period_sets:
        start = _resolve_bound(period["start"], placeholders)
        end = _resolve_bound(period["end"], placeholders)
        window = joined.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        flags = window[column].to_numpy().astype(bool) if window.height else np.array([], bool)
        n_s1 = int(flags.sum())
        n_s0 = int(flags.size) - n_s1
        entry = {
            "pair_id": pair.pair_id,
            "period_id": period["id"],
            "n_dates": int(flags.size),
            "n_dates_s1": n_s1,
            "n_dates_s0": n_s0,
            "delta": None,
        }
        if n_s1 >= min_dates_per_regime and n_s0 >= min_dates_per_regime:
            fit = newey_west_ols(
                window["rank_ic"].to_numpy(),
                flags.astype(float),
                window["formation_session_idx"].to_numpy(),
                pair.nw_lag,
            )
            entry["delta"] = fit["delta"] if math.isfinite(fit["delta"]) else None
        rows.append(entry)
    return rows


def _resolve_bound(value: Any, placeholders: dict[str, date]) -> date:
    if isinstance(value, str):
        if value in placeholders:
            return placeholders[value]
        return date.fromisoformat(value)
    return value


def alt_cut_delta(
    pair: PairSpec,
    cell_ic: pl.DataFrame,
    regimes: pl.DataFrame,
    *,
    sample_start: date | None,
) -> dict[str, Any]:
    """G6 (§6.2): the same pair against ``z_t > median_252(z)`` instead of
    ``z_t > 0``. A non-fatal diagnostic — it separates grade A from B, and
    never fails a pair. Its job is to show whether the zero threshold is doing
    the work or the result survives a different one."""
    column = f"alt_s_{pair.regime_id}"
    if column not in regimes.columns:
        return {"alt_cut_delta": None}
    joined = cell_ic.select(["trade_date", "formation_session_idx", "rank_ic"]).join(
        regimes.select(["trade_date", column]), on="trade_date", how="inner"
    )
    joined = joined.filter(pl.col(column).is_not_null())
    if sample_start is not None:
        joined = joined.filter(pl.col("trade_date") >= sample_start)
    joined = joined.sort("formation_session_idx")
    if joined.height < 3:
        return {"alt_cut_delta": None}
    fit = newey_west_ols(
        joined["rank_ic"].to_numpy(),
        joined[column].to_numpy().astype(float),
        joined["formation_session_idx"].to_numpy(),
        pair.nw_lag,
    )
    # The sign comparison itself belongs with the registered delta, which is
    # only final after BH — see `finalize_rows`.
    return {"alt_cut_delta": fit["delta"] if math.isfinite(fit["delta"]) else None}


def finalize_rows(rows: list[dict[str, Any]], *, q_threshold: float) -> list[dict[str, Any]]:
    """BH, then the gates that depend on it, then grades."""
    scored = apply_phase_c_bh(rows, q_threshold=q_threshold)
    out: list[dict[str, Any]] = []
    for row in scored:
        row = dict(row)
        delta = row.get("delta")
        alt = row.get("alt_cut_delta")
        row["alt_cut_sign_agree"] = (
            None
            if delta is None or alt is None or not math.isfinite(delta) or not math.isfinite(alt)
            else bool((delta > 0) == (alt > 0))
        )
        row.update(compute_screen_pass(row))
        row["evidence_grade"] = assign_evidence_grade(row)
        out.append(row)
    return out


# --- CLI ---


def run_phase_c(
    *,
    phase_a_run_dir: Path,
    phase_b_run_dir: Path | None,
    snapshot_date: str,
    source: str,
    output_root: Path,
    command_line: list[str],
    config_path: Path | str | None = None,
    root: DataRoot | None = None,
) -> Path:
    """Read two published runs' daily IC, condition on the regimes, publish.

    Nothing is rescanned and no panel is built: the only heavy input is
    parquet that two earlier runs already committed to. That is what lets this
    be rerun against a different (A, B) pair without disturbing anything —
    each run keeps its own directory and both stay on disk (§5's "바꿀 수
    있는 것").
    """
    from dataclasses import replace as _replace

    from modeler.analysis.horizon_scan_config import CONFIG_PATH, load_config
    from modeler.analysis.horizon_scan_phase_c_regimes import (
        build_regime_series,
        regime_occupancy,
    )
    from modeler.analysis.horizon_scan_run_spec import kst_now_iso, publish_run
    from modeler.etl.config import DataRoot, LakeConfig
    from modeler.etl.lake import connect, register_persisted_derived_mart
    from modeler.etl.mart import mart_root, register_mart_view

    config = load_config(config_path or CONFIG_PATH)
    phase_c = config.raw.get("phase_c", {})
    if "pairs" not in phase_c:
        raise ValueError(f"{config.path} registers no phase_c.pairs; not a conditional-IC contract")

    root = root or DataRoot.resolve(market="kr")
    base = LakeConfig(root=root, snapshot_date=snapshot_date, source=source)
    manifest = json.loads(
        (mart_root(base) / "_manifests" / "_SUCCESS.json").read_text(encoding="utf-8")
    )
    lake = _replace(base, analysis_config_hash=manifest["config_hash"])
    con = connect(lake)
    register_mart_view(con, lake, "label_scan")
    register_persisted_derived_mart(con, lake, "common_feature_daily_fact")

    sample_start = phase_c.get("sample_start")
    regimes = build_regime_series(con, sample_start=sample_start)
    assert_regime_columns_present(regimes)
    (common_end,) = con.execute(
        "SELECT max(trade_date) FROM label_scan WHERE common_formation_120d"
    ).fetchone()

    run_dirs = [d for d in (phase_a_run_dir, phase_b_run_dir) if d is not None]
    daily_ic = load_daily_ic(run_dirs)
    sessions = regimes["trade_date"].to_list()

    started_at = kst_now_iso()
    run_spec = build_phase_c_run_spec(
        config_hash=config.config_hash,
        phase_c_block=phase_c,
        phase_a_run_dir=phase_a_run_dir,
        phase_b_run_dir=phase_b_run_dir,
        snapshot_date=snapshot_date,
        source=source,
        sessions=sessions,
        started_at=started_at,
        command_line=command_line,
    )

    stats = phase_c["stats"]
    rows = [
        run_pair(
            pair,
            daily_ic,
            regimes,
            config_hash=config.config_hash,
            stats=stats,
            placebo_cfg=phase_c["placebo"],
            period_sets=config.raw["sample"]["period_sets"]["common"],
            placeholders={"common_formation_end": common_end},
            sample_start=sample_start,
        )
        for pair in load_pairs(config.raw)
    ]
    rows = finalize_rows(rows, q_threshold=float(stats["bh_q"]))

    run_dir_root = (
        output_root
        / "phase=C"
        / f"snapshot_date={snapshot_date}"
        / f"source={source}"
        / f"config_hash={config.config_hash}"
    )
    tmp_run_dir = run_dir_root / f"run_id={run_spec['run_id']}.tmp"
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    (tmp_run_dir / PHASE_C_RUN_SPEC_NAME).write_text(
        json.dumps(run_spec, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    subperiods = [entry for row in rows for entry in row.get("subperiods", [])]
    flat = [{k: v for k, v in row.items() if k != "subperiods"} for row in rows]
    pl.DataFrame(flat, infer_schema_length=None).write_parquet(
        tmp_run_dir / f"{CONDITIONAL_IC_TABLE}.parquet"
    )
    if subperiods:
        pl.DataFrame(subperiods, infer_schema_length=None).write_parquet(
            tmp_run_dir / f"{SUBPERIOD_TABLE}.parquet"
        )
    pl.DataFrame(
        [
            {
                "pair_id": row["pair_id"],
                "regime_id": row["regime_id"],
                "placebo_p": row.get("placebo_p"),
                "placebo_repeats": row.get("placebo_repeats"),
                "placebo_pass": row.get("placebo_pass"),
                "placebo_status": row.get("placebo_status"),
                "placebo_shift_min": row.get("placebo_shift_min"),
                "placebo_shift_max": row.get("placebo_shift_max"),
            }
            for row in rows
        ],
        infer_schema_length=None,
    ).write_parquet(tmp_run_dir / f"{PLACEBO_SUMMARY_TABLE}.parquet")
    regimes.write_parquet(tmp_run_dir / "regime_series.parquet")

    manifest_payload = {
        "phase": "C",
        "run_id": run_spec["run_id"],
        "config_hash": config.config_hash,
        "snapshot_date": snapshot_date,
        "source": source,
        "n_pairs": len(rows),
        "n_primary": sum(1 for r in rows if r["regime_role"] == "primary"),
        "n_screen_pass": sum(1 for r in rows if r.get("screen_pass")),
    }
    (tmp_run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    from modeler.analysis.horizon_scan_phase_c_report import (
        build_phase_c_report_context,
        write_phase_c_report,
    )

    write_phase_c_report(
        tmp_run_dir / PHASE_C_REPORT_NAME,
        build_phase_c_report_context(
            run_spec=run_spec,
            rows=rows,
            regime_summary=regime_occupancy(
                regimes.filter(pl.col("trade_date") <= common_end),
                min_dates=int(stats["min_dates_per_regime"]),
                min_share=float(stats["min_share_per_regime"]),
            ),
            q_threshold=float(stats["bh_q"]),
        ),
    )

    return publish_run(
        tmp_run_dir,
        run_dir_root / f"run_id={run_spec['run_id']}",
        run_spec=run_spec,
        required_artifacts=(
            PHASE_C_RUN_SPEC_NAME,
            "manifest.json",
            f"{CONDITIONAL_IC_TABLE}.parquet",
            PHASE_C_REPORT_NAME,
        ),
        content_hash_exclude_names=frozenset(
            {PHASE_C_RUN_SPEC_NAME, "_SUCCESS.json", PHASE_C_REPORT_NAME}
        ),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-run-dir", type=Path, required=True)
    parser.add_argument("--phase-b-run-dir", type=Path, default=None)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--source", default="sj2_remote")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    # F-9.10, same bug as horizon_scan.main: see the comment there.
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(effective_argv)

    from modeler.etl.config import DataRoot

    root = DataRoot.resolve(market="kr")
    output_root = args.output_root or (root.output / "scan" / "horizon_scan")

    published = run_phase_c(
        phase_a_run_dir=args.phase_a_run_dir,
        phase_b_run_dir=args.phase_b_run_dir,
        snapshot_date=args.snapshot_date,
        source=args.source,
        output_root=output_root,
        command_line=["horizon_scan_phase_c", *effective_argv],
        config_path=args.config,
        root=root,
    )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
