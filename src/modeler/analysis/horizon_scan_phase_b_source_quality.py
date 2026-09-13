"""§9 B-9's "source 비치명 경고" — the grade-A cap from source-layer quality.

``08_phase_b_implementation_log.md`` §4.2 left this open with an explicit
reason: the three signals the plan names (``mapping_fallback_ratio``,
``revision_ratio``, segment diagnostics) had no computed values anywhere, and
``compute_phase_b_evidence_grade``'s docstring says to wire them in "once they
exist rather than fabricating a warning signal that doesn't exist yet". B-10
Stage 2 produced the first two on 2026-08-12, so this module connects them.

What it does *not* do: turn a source warning into a failure. These are
non-fatal — a family that trips one is still a discovery, it just cannot claim
the strongest grade. Nothing here can move a cell below B.

**Thresholds are pre-registered in ``04_specific_plan_B.md`` §2.5 and fixed
before the first Phase B run that will be judged by them.** They were chosen
from the meaning of each ratio, not from the per-family numbers — see the plan
section for the reasoning and for an honest note on what was already measured
when they were written.

Unmeasurable is not the same as clean. A ratio that cannot be computed —
``revision_ratio`` before any receipt history exists, for instance — caps the
grade exactly as a breached threshold does. Grade A is the strongest claim in
this pipeline, and "we could not check a known risk" is not a basis for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# --- pre-registered thresholds (04_specific_plan_B.md §2.5) ---

# More than half the rows resolved through a rule that is not the metric's
# preferred one: the metric's identity then rests mainly on fallbacks.
MAPPING_FALLBACK_WARN = 0.5

# Revisions are rare by nature, so the line sits far lower than the mapping
# one. Above this, restatement is a systematic feature of the metric rather
# than an occasional correction.
REVISION_WARN = 0.10

# Cross-source pairing is an identity check between two views of the same
# receipt; they should agree essentially always. Above 1% the disagreement is
# systematic rather than incidental filer error.
PAIRING_MISMATCH_WARN = 0.01

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_UNMEASURED = "unmeasured"
STATUS_NOT_APPLICABLE = "not_applicable"

# Which canonical metrics each Phase B family's feature actually reads, taken
# from the feature SQL rather than guessed:
#   fin_scan.py `resolved`/`ratios`   -> the five fin_* families
#   sue_event.py `eps`/`shares`       -> fin_sue
#   event_scan.py build_payout_sql    -> ev_payout_yield
# net_income appears in every fin_* set because fin_scan picks the CFS/OFS
# basis off it, so a problem there moves every one of those features.
#
# Two families are deliberately empty rather than absent:
#   fin_log_mcap               is priced, not filed — no metric to warn about.
#   ev_net_share_issuance_yoy  reads dart_capital_change_raw / share counts
#                              straight from raw, never through the metric
#                              layer this module measures. Its own source
#                              quality is capital_change_quality's job.
FAMILY_METRIC_DEPENDENCIES: dict[str, frozenset[str]] = {
    "fin_log_mcap": frozenset(),
    "fin_value_z": frozenset(
        {"total_equity", "controlling_net_income", "operating_cash_flow", "revenue", "net_income"}
    ),
    "fin_gross_profitability": frozenset(
        {"gross_profit", "revenue", "cogs", "operating_income", "total_assets", "net_income"}
    ),
    "fin_asset_growth_yoy": frozenset({"total_assets", "net_income"}),
    "fin_accruals_to_assets": frozenset({"net_income", "operating_cash_flow", "total_assets"}),
    "fin_sue": frozenset({"controlling_net_income", "weighted_avg_shares"}),
    "ev_net_share_issuance_yoy": frozenset(),
    "ev_payout_yield": frozenset(
        {"issued_shares", "treasury_shares", "treasury_share_acquisition_amount"}
    ),
    "mcap_krx_log": frozenset(),
    "ev_filing_activity": frozenset(),
    "ev_amendment_ratio": frozenset(),
    "own_insider_filing_activity": frozenset(),
    "own_major_filing_activity": frozenset(),
    "own_amendment_ratio": frozenset(),
    "hc_employee_growth": frozenset(),
    "hc_productivity": frozenset(),
    "own_major_stake_level": frozenset(),
    "own_major_stake_change": frozenset(),
    # F-2. Built from prices, the PIT market cap and dart_corp_master's
    # industry code — nothing that goes through the metric layer this module
    # measures, so empty rather than absent (same reason as fin_log_mcap).
    "rel_peer_mom": frozenset(),
    "rel_own_minus_peer": frozenset(),
    "rel_peer_bigcap_lag": frozenset(),
    "rel_peer_dispersion": frozenset(),
    # F-4. Taken from fin_risk.py's own formulas, not guessed. net_income is in
    # every set for the same reason as the fin_scan families: fin_risk picks the
    # CFS/OFS basis off it, so a problem there moves each of these features.
    "fin_debt_to_assets": frozenset({"total_liabilities", "total_assets", "net_income"}),
    "fin_net_debt_to_mcap": frozenset(
        {"total_liabilities", "cash_and_cash_equivalents", "net_income"}
    ),
    "fin_interest_coverage": frozenset({"operating_income", "interest_paid", "net_income"}),
    "fin_ext_finance": frozenset({"financing_cash_flow", "total_assets", "net_income"}),
    "fin_lifecycle_stage": frozenset(
        {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "net_income"}
    ),
    "fin_lifecycle_transition": frozenset(
        {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "net_income"}
    ),
    "fin_profit_turn": frozenset({"net_income"}),
    "fin_negative_equity_exit": frozenset({"total_equity", "net_income"}),
    # dps comes straight from dart_shareholder_return_raw, never through the
    # metric layer — the ev_net_share_issuance_yoy case.
    "fin_dividend_initiation": frozenset(),
}


def _weighted_ratio(
    rows: Sequence[dict[str, Any]], *, numerator: str, denominator: str
) -> float | None:
    """Row-weighted ratio over the metric's own rows, not a mean of ratios.

    Averaging per-(year, report) ratios would give a business year with 40 rows
    the same say as one with 60,000.
    """
    total = 0
    hits = 0
    seen = False
    for row in rows:
        denom = row.get(denominator)
        if denom is None:
            continue
        seen = True
        total += int(denom)
        hits += int(row.get(numerator) or 0)
    if not seen or total == 0:
        return None
    return hits / total


def _worst_metric_ratio(
    metrics: frozenset[str],
    by_metric: dict[str, list[dict[str, Any]]],
    *,
    numerator: str,
    denominator: str,
) -> tuple[float | None, str | None]:
    """Highest per-metric ratio across a family's inputs, and which metric it is.

    ``None`` when no input yields a computable ratio — an unmeasurable input is
    not a clean one, and the caller turns that into its own status.
    """
    worst: float | None = None
    worst_metric: str | None = None
    for metric in sorted(metrics):
        ratio = _weighted_ratio(
            by_metric.get(metric, []), numerator=numerator, denominator=denominator
        )
        if ratio is None:
            continue
        if worst is None or ratio > worst:
            worst, worst_metric = ratio, metric
    return worst, worst_metric


def compute_family_source_quality(
    *,
    vintage_quality_rows: Sequence[dict[str, Any]] = (),
    pairing_quality_rows: Sequence[dict[str, Any]] = (),
    family_metric_dependencies: dict[str, frozenset[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """One verdict per family from B-10 Stage 2's two vintage-side diagnostics.

    ``pairing_quality_rows`` has no ``metric_code`` — ``receipt_value_pairing_quality``
    is grained by (bsns_year, reprt_code) — so the pairing ratio is lake-wide
    and applies to every family that reads the metric layer at all. That is
    the honest granularity: a broken cross-source check is a statement about
    the extraction path, not about one metric.
    """
    dependencies = family_metric_dependencies or FAMILY_METRIC_DEPENDENCIES
    pairing_ratio = _weighted_ratio(
        pairing_quality_rows, numerator="value_mismatch_rows", denominator="applicable_rows"
    )

    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in vintage_quality_rows:
        by_metric.setdefault(row.get("metric_code", ""), []).append(row)

    verdicts: dict[str, dict[str, Any]] = {}
    for family, metrics in dependencies.items():
        if not metrics:
            verdicts[family] = {
                "source_quality_status": STATUS_NOT_APPLICABLE,
                "mapping_fallback_ratio": None,
                "mapping_fallback_worst_metric": None,
                "revision_ratio": None,
                "revision_worst_metric": None,
                "pairing_mismatch_ratio": None,
                "source_quality_reasons": "",
            }
            continue

        # Worst metric, not the pooled average. A feature is a ratio of its
        # inputs, so one wholly fallback-mapped metric compromises it however
        # clean the other four are — pooling would dilute exactly the signal
        # this warning exists to catch. Row-weighting still applies *within* a
        # metric, across its business years.
        mapping_ratio, mapping_metric = _worst_metric_ratio(
            metrics, by_metric, numerator="mapping_fallback_rows", denominator="rows"
        )
        revision_ratio, revision_metric = _worst_metric_ratio(
            metrics, by_metric, numerator="revision_rows", denominator="revision_known_rows"
        )
        missing_metrics = sorted(m for m in metrics if not by_metric.get(m))

        reasons: list[str] = []
        if missing_metrics:
            reasons.append("no_vintage_quality_rows")
        if mapping_ratio is None:
            if not missing_metrics:
                reasons.append("mapping_fallback_unmeasured")
        elif mapping_ratio >= MAPPING_FALLBACK_WARN:
            reasons.append("mapping_fallback")
        if revision_ratio is None:
            if not missing_metrics:
                reasons.append("revision_unmeasured")
        elif revision_ratio >= REVISION_WARN:
            reasons.append("revision")
        if pairing_ratio is None:
            reasons.append("pairing_unmeasured")
        elif pairing_ratio >= PAIRING_MISMATCH_WARN:
            reasons.append("pairing_mismatch")

        if not reasons:
            status = STATUS_OK
        elif any(reason.endswith("unmeasured") or reason.startswith("no_") for reason in reasons):
            # Unmeasured dominates: it is the weaker claim of the two, and the
            # card should say "could not check" rather than "checked, warned".
            status = STATUS_UNMEASURED
        else:
            status = STATUS_WARN

        verdicts[family] = {
            "source_quality_status": status,
            "mapping_fallback_ratio": mapping_ratio,
            "mapping_fallback_worst_metric": mapping_metric,
            "revision_ratio": revision_ratio,
            "revision_worst_metric": revision_metric,
            "pairing_mismatch_ratio": pairing_ratio,
            "source_quality_reasons": ",".join(reasons),
        }
    return verdicts


def source_quality_allows_grade_a(status: str | None) -> bool:
    """``None`` means the caller had no diagnostic to consult at all.

    Treated like ``unmeasured`` rather than like ``ok`` — a run that did not
    produce the diagnostics cannot use their absence as evidence of quality.
    """
    return status in (STATUS_OK, STATUS_NOT_APPLICABLE)
