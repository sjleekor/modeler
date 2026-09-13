"""Phase A decay summary and family pattern classification (A-7, §A-7).

Operates on already-scanned cell rows (the output of
``horizon_scan_runner.run_registry_scan`` plus ``apply_global_bh``) for one
family — it does not touch the database itself. A cell's ``ic_mean`` is
aligned to the family's expected direction before any peak/onset/half-life
computation (§A-7: two-sided families like individual net-buy use the raw
observed sign instead — see ``_aligned_ic`` in horizon_scan_runner.py, which
this module mirrors rather than imports to keep decay logic reusable outside
the scan/BH pipeline).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # A-9 plots are rendered headless (run directory artifacts, not a UI)
import matplotlib.pyplot as plt  # noqa: E402

from modeler.analysis.horizon_scan_permutation import (  # noqa: E402
    TEMPORAL_MARGINAL_SE_MULTIPLE,
)


def _aligned(ic_mean: float | None, expected_sign: str | None) -> float | None:
    if ic_mean is None or not math.isfinite(ic_mean):
        return None
    if expected_sign == "-":
        return -ic_mean
    return ic_mean  # "+" and two-sided (None) read the observed IC directly


def compute_decay_summary(
    cum_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    *,
    expected_sign: str | None,
    half_life_fraction: float = 0.50,
) -> dict[str, Any]:
    """One family's decay curve summary (§A-7 table).

    ``cum_rows``/``bucket_rows`` each need at minimum ``h_end`` (and
    ``h_start`` for buckets), ``ic_mean``, ``q_fdr_global`` (may be ``None``
    for non-primary/exploratory cells). Buckets are read in ``h_start`` order
    to walk the decay curve chronologically.
    """
    cum_sorted = sorted(cum_rows, key=lambda r: r["h_end"])
    bucket_sorted = sorted(bucket_rows, key=lambda r: r["h_start"])

    summary: dict[str, Any] = {
        "peak_h_cum": None,
        "peak_h_cum_reason": None,
        "peak_bucket": None,
        "peak_bucket_reason": None,
        "onset_h": None,
        "onset_h_reason": "no_cell_passed_bh",
        "half_life_bucket": None,
        "half_life_bucket_reason": None,
        "sign_flip_bucket": None,
    }

    # peak_h_cum: cumulative h with the largest |aligned IC|.
    cum_aligned = [
        (row["h_end"], _aligned(row.get("ic_mean"), expected_sign)) for row in cum_sorted
    ]
    cum_valid = [(h, a) for h, a in cum_aligned if a is not None]
    if cum_valid:
        summary["peak_h_cum"] = max(cum_valid, key=lambda pair: abs(pair[1]))[0]
    else:
        summary["peak_h_cum_reason"] = "no_valid_cumulative_cell"

    # peak_bucket: bucket with the largest aligned IC (not absolute — §A-7
    # says "aligned_ic 최대 bucket", unlike the cumulative peak's abs()).
    bucket_aligned = [
        ((row["h_start"], row["h_end"]), _aligned(row.get("ic_mean"), expected_sign))
        for row in bucket_sorted
    ]
    bucket_valid = [(b, a) for b, a in bucket_aligned if a is not None]
    if bucket_valid:
        summary["peak_bucket"] = max(bucket_valid, key=lambda pair: pair[1])[0]
    else:
        summary["peak_bucket_reason"] = "no_valid_bucket_cell"

    # onset_h: first cumulative h (in grid order) with q_fdr_global < 0.10
    # AND the right sign — the earliest horizon at which the family "shows up."
    for row in cum_sorted:
        q = row.get("q_fdr_global")
        aligned = _aligned(row.get("ic_mean"), expected_sign)
        if q is not None and q < 0.10 and aligned is not None and aligned > 0:
            summary["onset_h"] = row["h_end"]
            summary["onset_h_reason"] = None
            break

    # half_life_bucket: first bucket *after* the peak bucket whose aligned IC
    # drops below half_life_fraction * peak's aligned IC. Never forced if
    # there's no valid peak or no bucket after it (§A-7: "peak가 0 이하이거나
    # 유효 후속 bucket이 없으면 half-life를 억지로 만들지 않는다").
    if bucket_valid:
        peak_bucket_key, peak_value = max(bucket_valid, key=lambda pair: pair[1])
        if peak_value <= 0:
            summary["half_life_bucket_reason"] = "peak_not_positive"
        else:
            peak_index = next(
                i for i, (b, _a) in enumerate(bucket_valid) if b == peak_bucket_key
            )
            threshold = half_life_fraction * peak_value
            following = bucket_valid[peak_index + 1 :]
            if not following:
                summary["half_life_bucket_reason"] = "no_subsequent_bucket"
            else:
                for b, a in following:
                    if a < threshold:
                        summary["half_life_bucket"] = b
                        break
                else:
                    summary["half_life_bucket_reason"] = "decay_never_reaches_half_life"
    else:
        summary["half_life_bucket_reason"] = "no_valid_bucket_cell"

    # sign_flip_bucket: first bucket (in order) whose aligned sign differs
    # from the *first* valid bucket's sign.
    if len(bucket_valid) >= 2:
        first_sign = bucket_valid[0][1] > 0
        for b, a in bucket_valid[1:]:
            if (a > 0) != first_sign:
                summary["sign_flip_bucket"] = b
                break

    return summary


PATTERN_NO_SIGNAL = "no_signal"
PATTERN_IMMEDIATE = "immediate"
PATTERN_DELAYED = "delayed"
PATTERN_SIGN_REVERSAL = "sign_reversal"
PATTERN_SEGMENT_LIMITED = "segment_limited"
PATTERN_EXPLORATORY_ONLY = "exploratory_only"


def classify_pattern_auto(
    *,
    has_primary_discovery: bool,
    has_exploratory_significant: bool,
    peak_bucket: tuple[int, int] | None,
    sign_flip_bucket: tuple[int, int] | None,
    segment_gates_all_pass: bool | None,
) -> str:
    """§A-7 automatic pattern candidates — evaluated in the order that avoids
    a family qualifying for two labels at once (sign_reversal and
    segment_limited are checked before the coarser no_signal/exploratory_only
    fallbacks so a discovery with a real reversal isn't mislabeled "delayed").
    """
    if not has_primary_discovery:
        return PATTERN_NO_SIGNAL if not has_exploratory_significant else PATTERN_EXPLORATORY_ONLY
    if sign_flip_bucket is not None:
        return PATTERN_SIGN_REVERSAL
    if segment_gates_all_pass is False:
        return PATTERN_SEGMENT_LIMITED
    if peak_bucket is not None and peak_bucket[0] == 0 and peak_bucket[1] == 5:
        return PATTERN_IMMEDIATE
    return PATTERN_DELAYED


# --- A-8 screening decision and evidence grade (§A-8) ---


def compute_screen_pass(
    *,
    role: str,
    primary_discovery: bool,
    tradable_pass: bool,
    period_sign_pass: bool,
    isolated_spike: bool,
    available_direction_pass: bool | None,
    delay_required: bool,
    delay_pass: bool | None,
    temporal_null_required: bool,
    temporal_null_pass: bool | None,
) -> dict[str, Any]:
    """§A-8: every listed condition must hold for a ``role="ready"`` family's
    candidate cell. ``delay_required``/``temporal_null_required`` reflect
    whether *this* cell is h<=5 (or bucket (0,5]) / nw_lag>=59 — when a gate
    doesn't apply to the cell it is simply omitted from the check, not
    treated as passed or failed. Exploratory/reference/secondary cells never
    screen-pass (§2.2): they're diagnostics, not discoveries.
    """
    if role != "ready":
        return {"screen_pass": False, "not_applicable_role": True, "failed_gates": []}

    checks: dict[str, bool] = {
        "primary_discovery": primary_discovery,
        "tradable_pass": tradable_pass,
        "period_sign_pass": period_sign_pass,
        "isolated_spike_clear": not isolated_spike,
    }
    if available_direction_pass is not None:
        checks["available_direction_pass"] = available_direction_pass
    if delay_required:
        checks["delay_pass"] = bool(delay_pass)
    if temporal_null_required:
        checks["temporal_null_pass"] = bool(temporal_null_pass)

    failed_gates = [name for name, ok in checks.items() if not ok]
    return {
        "screen_pass": not failed_gates,
        "not_applicable_role": False,
        "failed_gates": failed_gates,
    }


_EVIDENCE_GRADE_EVALUATION_ORDER = ("R", "C", "A", "B", "D")


def assign_evidence_grade(
    *,
    role: str,
    screen_pass: bool,
    has_nonfatal_warning: bool = False,
    all_offsets_evaluable: bool = True,
    available_sign_flip: bool = False,
) -> str:
    """One of ``R/C/A/B/D`` per the config's ``evidence_grade`` rubric,
    checked in its ``evaluation_order`` (R, C before A, B, D — a reference or
    exploratory cell is graded R/C even if it would otherwise look like a
    screen pass, since those roles never enter the candidate pool at all).

    ``role`` is the cell's ``feature_role``/``hypothesis_role``: pass
    ``"reference"`` for the reference-only family, one of
    ``"exploratory_short_regime"``/``"exploratory_horizon"``/
    ``"secondary_feature"`` for non-primary cells, or ``"ready"`` for a
    primary/candidate cell.
    """
    if role == "reference":
        return "R"
    if role in ("exploratory_short_regime", "exploratory_horizon", "secondary_feature"):
        return "C"
    if available_sign_flip:
        return "C"
    if screen_pass and not has_nonfatal_warning and all_offsets_evaluable:
        return "A"
    if screen_pass:
        return "B"
    return "D"


# --- family conclusion card assembly (§5 A-8 card schema) ---

def _marginal_note(source: dict) -> str:
    """`" (marginal: p_max within Ns)"` when the placebo verdict is MC noise.

    F-9.11. The temporal placebo runs 100 replicates, so at p around 0.10 the
    Monte Carlo error of p is 0.030 -- and because the shift seed is derived
    from ``config_hash``, each new preregistration layer redraws the null. Two
    cells changed side between the 2026-08-30 and 2026-09-09 layers with every
    other statistic bit-identical. The verdict still stands as preregistered;
    this only stops it being read as evidence.
    """
    if not source.get("temporal_null_marginal"):
        return ""
    band = source.get("temporal_marginal_band")
    if band is None:
        return " (marginal)"
    return f" (marginal: within {TEMPORAL_MARGINAL_SE_MULTIPLE:.0f} MC SE of p_max, ±{band:.3f})"


_FAMILY_CARD_FIELDS = (
    "family",
    "domain",
    "primary_feature",
    "secondary_features",
    "expected_sign",
    "observed_sign",
    "pattern_auto",
    "pattern_reviewed",
    "review_status",
    "primary_discoveries",
    "candidate_horizon_band",
    "target_label_candidates",
    "peak_h_cum",
    "peak_bucket",
    "onset_h",
    "half_life_bucket",
    "sign_flip_bucket",
    "broad_ic",
    "tradable_ic",
    "tradable_retention",
    "valid_subperiods",
    "sign_consistent_subperiods",
    "native_ic",
    "lag1_ic",
    "delay_pass",
    "common_survivor_ic",
    "available_ic",
    "attrition_warning",
    "nonoverlap_offset_summary",
    "kospi_weight_mean",
    "kosdaq_weight_mean",
    "p_temporal_nw",
    "temporal_null_pass",
    "p_temporal_nw_se",
    "temporal_marginal_band",
    "temporal_null_marginal",
    "q_fdr_global",
    "evidence_grade",
    "screen_pass",
    "sparse_primary_grid",
    "exploratory_short_regime",
    "warnings",
    "limitations",
    "next_action",
)


def build_family_card(
    *,
    family: str,
    domain: str,
    primary_feature: str,
    secondary_features: list[str] | None = None,
    expected_sign: str | None,
    observed_sign: str | None,
    decay_summary: dict[str, Any],
    pattern_auto: str,
    primary_discoveries: list[str],
    candidate_horizon_band: tuple[int, int] | None,
    broad_ic: float | None,
    tradable_ic: float | None,
    tradable_retention: float | None,
    valid_subperiods: int,
    sign_consistent_subperiods: int,
    native_ic: float | None,
    lag1_ic: float | None,
    delay_pass: bool | None,
    common_survivor_ic: float | None,
    available_ic: float | None,
    attrition_warning: bool,
    nonoverlap_offset_summary: dict[str, Any] | None,
    kospi_weight_mean: float | None,
    kosdaq_weight_mean: float | None,
    p_temporal_nw: float | None,
    temporal_null_pass: bool | None,
    p_temporal_nw_se: float | None = None,
    temporal_marginal_band: float | None = None,
    temporal_null_marginal: bool | None = None,
    q_fdr_global: float | None,
    evidence_grade: str,
    screen_pass: bool,
    sparse_primary_grid: bool = False,
    exploratory_short_regime: bool = False,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    next_action: str | None = None,
    target_label_candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble one family's conclusion card (§5 A-8 schema).

    Always created with ``review_status="unreviewed"`` and
    ``pattern_reviewed=None`` — a human reviewer's override lives in a
    separate file and never mutates this card in place (§A-7: "원 parquet를
    수정하지 않는다").
    """
    card = {
        "family": family,
        "domain": domain,
        "primary_feature": primary_feature,
        "secondary_features": secondary_features or [],
        "expected_sign": expected_sign,
        "observed_sign": observed_sign,
        "pattern_auto": pattern_auto,
        "pattern_reviewed": None,
        "review_status": "unreviewed",
        "primary_discoveries": primary_discoveries,
        "candidate_horizon_band": candidate_horizon_band,
        "target_label_candidates": target_label_candidates or [],
        "peak_h_cum": decay_summary.get("peak_h_cum"),
        "peak_bucket": decay_summary.get("peak_bucket"),
        "onset_h": decay_summary.get("onset_h"),
        "half_life_bucket": decay_summary.get("half_life_bucket"),
        "sign_flip_bucket": decay_summary.get("sign_flip_bucket"),
        "broad_ic": broad_ic,
        "tradable_ic": tradable_ic,
        "tradable_retention": tradable_retention,
        "valid_subperiods": valid_subperiods,
        "sign_consistent_subperiods": sign_consistent_subperiods,
        "native_ic": native_ic,
        "lag1_ic": lag1_ic,
        "delay_pass": delay_pass,
        "common_survivor_ic": common_survivor_ic,
        "available_ic": available_ic,
        "attrition_warning": attrition_warning,
        "nonoverlap_offset_summary": nonoverlap_offset_summary,
        "kospi_weight_mean": kospi_weight_mean,
        "kosdaq_weight_mean": kosdaq_weight_mean,
        "p_temporal_nw": p_temporal_nw,
        "temporal_null_pass": temporal_null_pass,
        "p_temporal_nw_se": p_temporal_nw_se,
        "temporal_marginal_band": temporal_marginal_band,
        "temporal_null_marginal": temporal_null_marginal,
        "q_fdr_global": q_fdr_global,
        "evidence_grade": evidence_grade,
        "screen_pass": screen_pass,
        "sparse_primary_grid": sparse_primary_grid,
        "exploratory_short_regime": exploratory_short_regime,
        "warnings": warnings or [],
        "limitations": limitations or [],
        "next_action": next_action,
    }
    assert set(card) == set(_FAMILY_CARD_FIELDS), "family card schema drifted from §5 A-8"
    return card


# --- A-9 family plots (§A-9: 7 plot kinds per family, headless PNG output) ---


def _finish_plot(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def _empty_axis(ax: plt.Axes, message: str = "no data") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_cumulative_ic_curve(
    curves: dict[str, list[dict[str, Any]]], *, family: str, output_path: Path
) -> Path:
    """§A-9 plot 1: cumulative IC vs. h for broad/tradable x common-survivor/available.

    ``curves`` maps a variant label (e.g. ``"broad_common_survivor"``) to its
    ``{h_end, ic_mean}`` rows; a missing or empty variant (e.g. a
    reference-only family with no tradable variant) is skipped, not an error.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, rows in curves.items():
        ordered = sorted(
            (r for r in rows if r.get("ic_mean") is not None), key=lambda r: r["h_end"]
        )
        if not ordered:
            continue
        ax.plot(
            [r["h_end"] for r in ordered], [r["ic_mean"] for r in ordered], marker="o", label=label
        )
        plotted = True
    if plotted:
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("h (days)")
        ax.set_ylabel("IC (mean)")
        ax.legend(fontsize=8)
    else:
        _empty_axis(ax)
    ax.set_title(f"{family}: cumulative IC curve")
    return _finish_plot(fig, output_path)


def plot_bucket_ic_bar(
    bucket_rows: list[dict[str, Any]],
    *,
    family: str,
    expected_sign: str | None,
    output_path: Path,
) -> Path:
    """§A-9 plot 2: aligned bucket IC with an approximate NW 95% interval.

    Bars clearing the global BH q-threshold are colored distinctly from those
    that don't — segment/unadjusted p-values are never given this same
    visual weight elsewhere (§A-9: "q 통과는 색으로 표시하되 segment의
    unadjusted p-value를 발견처럼 강조하지 않는다").
    """
    ordered = sorted(bucket_rows, key=lambda r: r["h_start"])
    fig, ax = plt.subplots(figsize=(6, 4))
    if not ordered:
        _empty_axis(ax)
        ax.set_title(f"{family}: bucket IC")
        return _finish_plot(fig, output_path)

    labels = [f"({r['h_start']},{r['h_end']}]" for r in ordered]
    aligned = [_aligned(r.get("ic_mean"), expected_sign) or 0.0 for r in ordered]
    errors = []
    for r in ordered:
        t_nw, ic_mean = r.get("t_nw"), r.get("ic_mean")
        if t_nw and ic_mean is not None and math.isfinite(t_nw) and t_nw != 0:
            se = abs(ic_mean / t_nw)
        else:
            se = 0.0
        errors.append(1.96 * se)
    colors = [
        "#2a6f2a" if (r.get("q_fdr_global") is not None and r["q_fdr_global"] < 0.10) else "#888888"
        for r in ordered
    ]
    ax.bar(labels, aligned, yerr=errors, color=colors, capsize=3)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("aligned IC")
    ax.set_title(f"{family}: bucket IC (green = BH q<0.10)")
    return _finish_plot(fig, output_path)


def plot_native_vs_lag1(
    native_rows: list[dict[str, Any]],
    lag1_rows: list[dict[str, Any]],
    *,
    family: str,
    output_path: Path,
) -> Path:
    """§A-9 plot 3: native-timing vs. one-session-delayed IC curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    native_ordered = sorted(
        (r for r in native_rows if r.get("ic_mean") is not None), key=lambda r: r["h_end"]
    )
    lag1_ordered = sorted(
        (r for r in lag1_rows if r.get("ic_mean") is not None), key=lambda r: r["h_end"]
    )
    if not native_ordered and not lag1_ordered:
        _empty_axis(ax)
    else:
        if native_ordered:
            ax.plot(
                [r["h_end"] for r in native_ordered],
                [r["ic_mean"] for r in native_ordered],
                marker="o",
                label="native",
            )
        if lag1_ordered:
            ax.plot(
                [r["h_end"] for r in lag1_ordered],
                [r["ic_mean"] for r in lag1_ordered],
                marker="s",
                label="lag1",
            )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("h (days)")
        ax.set_ylabel("IC (mean)")
        ax.legend(fontsize=8)
    ax.set_title(f"{family}: native vs lag1")
    return _finish_plot(fig, output_path)


def plot_subperiod_heatmap(
    period_rows: list[dict[str, Any]], *, family: str, output_path: Path
) -> Path:
    """§A-9 plot 4: period id x horizon aligned-IC heatmap.

    ``period_rows`` entries need ``period_id``, ``h_end``, ``ic_mean``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    if not period_rows:
        _empty_axis(ax)
        ax.set_title(f"{family}: subperiod heatmap")
        return _finish_plot(fig, output_path)

    periods = sorted({r["period_id"] for r in period_rows})
    horizons = sorted({r["h_end"] for r in period_rows})
    grid = np.full((len(horizons), len(periods)), np.nan)
    period_index = {p: i for i, p in enumerate(periods)}
    horizon_index = {h: i for i, h in enumerate(horizons)}
    for r in period_rows:
        if r.get("ic_mean") is not None:
            grid[horizon_index[r["h_end"]], period_index[r["period_id"]]] = r["ic_mean"]
    scale = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    scale = scale or 1.0
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels(horizons)
    ax.set_ylabel("h (days)")
    fig.colorbar(im, ax=ax, label="IC (mean)")
    ax.set_title(f"{family}: subperiod heatmap")
    return _finish_plot(fig, output_path)


def plot_segment_dot(
    segment_rows: list[dict[str, Any]], *, family: str, output_path: Path
) -> Path:
    """§A-9 plot 5: one dot per (segment_axis, segment) — diagnostic only,
    since segment multiplicity never earns its own q-value (§A-4)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if not segment_rows:
        _empty_axis(ax)
        ax.set_title(f"{family}: segment IC")
        return _finish_plot(fig, output_path)
    labels = [f"{r['segment_axis']}:{r['segment']}" for r in segment_rows]
    values = [r.get("ic_mean") or 0.0 for r in segment_rows]
    ax.scatter(range(len(labels)), values)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("IC (mean)")
    ax.set_title(f"{family}: segment IC (diagnostic only)")
    return _finish_plot(fig, output_path)


def plot_coverage_curve(
    coverage_rows: list[dict[str, Any]], *, family: str, output_path: Path
) -> Path:
    """§A-9 plot 6: feature/label coverage and survival-to-h vs. horizon."""
    ordered = sorted(coverage_rows, key=lambda r: r["h_end"])
    fig, ax = plt.subplots(figsize=(6, 4))
    if not ordered:
        _empty_axis(ax)
    else:
        h = [r["h_end"] for r in ordered]
        for key, label in (
            ("feature_coverage", "feature"),
            ("label_coverage", "label"),
            ("survival_to_h", "survival"),
        ):
            series = [r.get(key) for r in ordered]
            if any(v is not None for v in series):
                ax.plot(h, series, marker="o", label=label)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("h (days)")
        ax.set_ylabel("ratio")
        ax.legend(fontsize=8)
    ax.set_title(f"{family}: coverage / survival")
    return _finish_plot(fig, output_path)


def plot_offset_distribution(
    offset_summary: dict[str, Any], *, family: str, output_path: Path
) -> Path:
    """§A-9 plot 7: distribution of per-offset IC means — never a single
    "best" offset highlighted as representative (§A-5)."""
    offsets = [o for o in offset_summary.get("offsets", []) if o.get("status") == "valid"]
    fig, ax = plt.subplots(figsize=(6, 4))
    if not offsets:
        _empty_axis(ax)
    else:
        values = [o["ic_mean"] for o in offsets]
        ax.hist(values, bins=min(10, len(values)))
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("offset IC mean")
        ax.set_ylabel("count")
    ax.set_title(f"{family}: non-overlap offset distribution ({len(offsets)} valid)")
    return _finish_plot(fig, output_path)


def render_family_plots(
    *,
    family: str,
    output_dir: Path,
    cumulative_curves: dict[str, list[dict[str, Any]]],
    bucket_rows: list[dict[str, Any]],
    expected_sign: str | None,
    native_rows: list[dict[str, Any]],
    lag1_rows: list[dict[str, Any]],
    period_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    offset_summary: dict[str, Any],
) -> dict[str, Path]:
    """Render all 7 §A-9 plots for one family into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cumulative_ic_curve": plot_cumulative_ic_curve(
            cumulative_curves, family=family, output_path=output_dir / f"{family}_cumulative_ic.png"
        ),
        "bucket_ic_bar": plot_bucket_ic_bar(
            bucket_rows,
            family=family,
            expected_sign=expected_sign,
            output_path=output_dir / f"{family}_bucket_ic.png",
        ),
        "native_vs_lag1": plot_native_vs_lag1(
            native_rows,
            lag1_rows,
            family=family,
            output_path=output_dir / f"{family}_native_vs_lag1.png",
        ),
        "subperiod_heatmap": plot_subperiod_heatmap(
            period_rows, family=family, output_path=output_dir / f"{family}_subperiod_heatmap.png"
        ),
        "segment_dot": plot_segment_dot(
            segment_rows, family=family, output_path=output_dir / f"{family}_segment_dot.png"
        ),
        "coverage_curve": plot_coverage_curve(
            coverage_rows, family=family, output_path=output_dir / f"{family}_coverage.png"
        ),
        "offset_distribution": plot_offset_distribution(
            offset_summary,
            family=family,
            output_path=output_dir / f"{family}_offset_distribution.png",
        ),
    }


# --- A-9 markdown report renderer (§A-9's fixed 11-section order) ---

_REPORT_SECTION_TITLES = (
    "1. Run identity and A0 preflight",
    "2. Sample, coverage, and holdout seal",
    "3. Global BH summary and short-exploratory coverage",
    "4. Real discovery count vs. cross-sectional permutation null",
    "5. Long-cell NW t vs. temporal placebo null",
    "6. Price overview and family cards",
    "7. Flow overview and family cards",
    "8. Segment, delay, attrition, and offset warnings",
    "9. Acceptance-gate handoff list",
    "10. No-signal, deferred, and exploratory candidates",
    "11. Limitations (survivorship, management-designation, publication lag)",
)

_REPORT_CONTEXT_KEYS = frozenset(
    {
        "run_identity",
        "preflight",
        "sample_coverage",
        "bh_summary",
        "short_exploratory_summary",
        "permutation_summary",
        "temporal_summary",
        "price_cards",
        "flow_cards",
        "warnings",
        "acceptance_gate",
        "deferred_candidates",
        "limitations",
    }
)


def _render_family_card_md(card: dict[str, Any]) -> str:
    lines = [f"#### {card['family']} ({card['evidence_grade']})", ""]
    lines.append(f"- primary feature: `{card['primary_feature']}`")
    lines.append(f"- expected/observed sign: {card['expected_sign']} / {card['observed_sign']}")
    lines.append(f"- pattern: {card['pattern_auto']} (review: {card['review_status']})")
    lines.append(f"- screen_pass: {card['screen_pass']}")
    lines.append(f"- primary discoveries: {', '.join(card['primary_discoveries']) or '-'}")
    band = card.get("candidate_horizon_band")
    lines.append(f"- candidate horizon band: {band if band else '-'}")
    lines.append(
        f"- broad/tradable IC: {card['broad_ic']} / {card['tradable_ic']} "
        f"(retention {card['tradable_retention']})"
    )
    lines.append(
        f"- common-survivor/available IC: {card['common_survivor_ic']} / {card['available_ic']}"
        f"{' (attrition warning)' if card['attrition_warning'] else ''}"
    )
    lines.append(
        f"- native/lag1 IC: {card['native_ic']} / {card['lag1_ic']} "
        f"(delay_pass={card['delay_pass']})"
    )
    if card.get("p_temporal_nw") is not None:
        lines.append(
            f"- temporal placebo: p={card['p_temporal_nw']:.3f} "
            f"pass={card['temporal_null_pass']}{_marginal_note(card)}"
        )
    if card["warnings"]:
        lines.append(f"- warnings: {'; '.join(card['warnings'])}")
    if card["limitations"]:
        lines.append(f"- limitations: {'; '.join(card['limitations'])}")
    if card.get("next_action"):
        lines.append(f"- next action: {card['next_action']}")
    return "\n".join(lines)


def render_markdown_report(context: dict[str, Any]) -> str:
    """Assemble ``03a_horizon_scan_results.md`` in the fixed §A-9 section order.

    Every key in ``_REPORT_CONTEXT_KEYS`` is required (an empty list/dict for
    a section with nothing to say — never omitted, so a missing key is a
    caller bug surfaced immediately, not a silently blank section).
    ``warnings``/``acceptance_gate``/``deferred_candidates``/``limitations``
    are lists of already-formatted one-line strings.
    """
    missing = _REPORT_CONTEXT_KEYS - context.keys()
    if missing:
        raise ValueError(f"report context missing keys: {sorted(missing)}")

    ri = context["run_identity"]
    pf = context["preflight"]
    sc = context["sample_coverage"]
    bh = context["bh_summary"]
    short = context["short_exploratory_summary"]
    perm = context["permutation_summary"]
    temp = context["temporal_summary"]

    sections = [
        "\n".join(
            [
                f"## {_REPORT_SECTION_TITLES[0]}",
                "",
                f"- run_id: `{ri.get('run_id')}`",
                f"- snapshot_date: {ri.get('snapshot_date')} / source: {ri.get('source')}",
                f"- config_hash: `{ri.get('config_hash')}`",
                f"- official: {ri.get('official')} (started {ri.get('started_at')}, "
                f"finished {ri.get('finished_at')})",
                f"- A0 preflight: {pf.get('status', 'unknown')}",
            ]
        ),
        "\n".join(
            [
                f"## {_REPORT_SECTION_TITLES[1]}",
                "",
                f"- holdout_start: {sc.get('holdout_start')}",
                f"- effective sample range: {sc.get('effective_sample_start')} .. "
                f"{sc.get('effective_sample_end')}",
                f"- common_formation_end: {sc.get('common_formation_end')}",
            ]
        ),
        "\n".join(
            [
                f"## {_REPORT_SECTION_TITLES[2]}",
                "",
                f"- primary hypotheses: {bh.get('n_hypotheses', 75)}, "
                f"valid: {bh.get('n_valid')}, bh_pass: {bh.get('n_bh_pass')}, "
                f"primary_discovery: {bh.get('n_primary_discovery')} "
                f"(q<{bh.get('q_threshold', 0.10)})",
                f"- short exploratory cells: {short.get('n_cells', 28)}, "
                f"valid: {short.get('n_valid')}",
            ]
        ),
        "\n".join(
            [
                f"## {_REPORT_SECTION_TITLES[3]}",
                "",
                f"- real discovery count: {perm.get('real_discovery_count')}",
                f"- p_empirical_count: {perm.get('p_empirical_count')} "
                f"({perm.get('n_replicates')} replicates)",
            ]
        ),
    ]
    long_cell_lines = [
        f"  - `{hid}`: p_temporal_nw={v.get('p_temporal_nw')}, "
        f"pass={v.get('temporal_null_pass')}{_marginal_note(v)}"
        for hid, v in temp.get("per_cell", {}).items()
    ] or ["  - (no nw_lag>=59 primary cells)"]
    sections.append(
        "\n".join(
            [f"## {_REPORT_SECTION_TITLES[4]}", "", f"- {temp.get('n_replicates')} replicates"]
            + long_cell_lines
        )
    )
    sections.append(
        "\n".join(
            [f"## {_REPORT_SECTION_TITLES[5]}", ""]
            + [_render_family_card_md(c) for c in context["price_cards"]]
        )
    )
    sections.append(
        "\n".join(
            [f"## {_REPORT_SECTION_TITLES[6]}", ""]
            + [_render_family_card_md(c) for c in context["flow_cards"]]
        )
    )
    warnings = context["warnings"] or ["(none)"]
    sections.append(
        "\n".join([f"## {_REPORT_SECTION_TITLES[7]}", ""] + [f"- {w}" for w in warnings])
    )
    gate = context["acceptance_gate"] or ["(none)"]
    sections.append("\n".join([f"## {_REPORT_SECTION_TITLES[8]}", ""] + [f"- {g}" for g in gate]))
    deferred = context["deferred_candidates"] or ["(none)"]
    sections.append(
        "\n".join([f"## {_REPORT_SECTION_TITLES[9]}", ""] + [f"- {d}" for d in deferred])
    )
    limitations = context["limitations"] or ["(none)"]
    sections.append(
        "\n".join([f"## {_REPORT_SECTION_TITLES[10]}", ""] + [f"- {lim}" for lim in limitations])
    )
    return "\n\n".join(sections) + "\n"


def write_markdown_report(output_path: Path, context: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(context), encoding="utf-8")
    return output_path
