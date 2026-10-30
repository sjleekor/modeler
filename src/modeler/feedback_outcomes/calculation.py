"""Pure, fixed-rule KR h20 diagnostics; this module never loads a price source.

Inputs must already be normalized by the guarded adapter. Unknown outcomes stay
null. Decimal arithmetic prevents accidental boolean/nonfinite numeric inputs.
All reported returns are price returns; no dividends or account model are used.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from modeler.feedback_outcomes.contracts import INDEX_KEYS, ROUND_TRIP_COST, SEOUL

_KST = SEOUL
_COST = Decimal(str(ROUND_TRIP_COST))
_INDEX = {key: name for key, name in INDEX_KEYS.values()}
_RESOLVED = {"available", "not_entered", "withdrawn_before_entry"}


def _number(value: Any, *, positive: bool = False) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and (not positive or result > 0) else None


def _aware(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timezone-aware timestamp required")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return result


def _session(value: Any) -> tuple[str, datetime]:
    if isinstance(value, Mapping):
        day = str(value.get("session", value.get("date", "")))
        close = value.get("close_at")
    else:
        day, close = str(value), None
    parsed = date.fromisoformat(day)
    if parsed.isoformat() != day:
        raise ValueError("session requires canonical YYYY-MM-DD")
    close_at = (
        _aware(close).astimezone(_KST) if close else datetime.combine(parsed, time(15, 30), _KST)
    )
    if close_at.date() != parsed or not time(9) <= close_at.time() <= time(18):
        raise ValueError("session close must be on the same KR date within 09:00-18:00 KST")
    return day, close_at


def _market(value: Any) -> str:
    return str(value or "").strip().lower()


def _slot_market(slot: Mapping[str, Any]) -> str:
    return _market(slot.get("security_market", slot.get("market")))


def _ratio(end: Any, start: Any) -> float | None:
    a, b = _number(end, positive=True), _number(start, positive=True)
    if a is None or b is None:
        return None
    with localcontext() as context:
        context.prec = 34
        result = a / b - 1
    converted = float(result)
    return converted if _number(converted) is not None else None


def _known_at(record: Mapping[str, Any], cutoff: datetime) -> bool:
    try:
        return _aware(record.get("available_at")) <= cutoff
    except (ValueError, TypeError):
        return False


def _execution_reason(record: Mapping[str, Any]) -> str | None:
    if record.get("tradeability_conflict") is True:
        return "tradeability_conflict"
    if record.get("explicit_untradeable") is True:
        return "explicit_untradeable"
    fields = ("source_open", "source_high", "source_low", "source_close", "volume", "trading_value")
    numbers = {key: _number(record.get(key), positive=True) for key in fields}
    if any(value is None for value in numbers.values()):
        return "missing_or_invalid_tradeability_fields"
    op, high, low, close = (numbers[key] for key in fields[:4])
    if not (low <= op <= high and low <= close <= high and high > low):
        return "invalid_ohlc_or_locked_price"
    return None


def _identity_reason(
    slot: Mapping[str, Any],
    entry: Mapping[str, Any],
    exit_: Mapping[str, Any],
    observations: Mapping[str, Any],
    entry_day: str,
    exit_day: str,
) -> str | None:
    if observations.get("identity_verified") is not True:
        return "identity_not_verified"
    entity = entry.get("entity_id")
    if not entity or exit_.get("entity_id") != entity:
        return "entity_changed_or_missing"
    market, symbol = _slot_market(slot), slot.get("symbol")
    for record, day in ((entry, entry_day), (exit_, exit_day)):
        if _market(record.get("market")) != market or record.get("symbol") != symbol:
            return "market_or_symbol_changed"
        start, end = record.get("listing_valid_from"), record.get("listing_valid_to")
        try:
            if not start or day < date.fromisoformat(str(start)).isoformat():
                return "listing_period_not_verified"
            if end and day > date.fromisoformat(str(end)).isoformat():
                return "outside_listing_period"
        except ValueError:
            return "listing_period_not_verified"
    return None


def _benchmark(
    slot: Mapping[str, Any],
    observations: Mapping[str, Any],
    entry_day: str,
    exit_day: str,
    cutoff: datetime,
) -> tuple[float | None, str | None]:
    market = _slot_market(slot)
    if market not in _INDEX:
        return None, "benchmark_market_unknown"
    pair = []
    for key, day in (("benchmark_entry", entry_day), ("benchmark_exit", exit_day)):
        record = observations.get(key)
        if not isinstance(record, Mapping):
            return None, "benchmark_source_missing"
        if record.get("index_group") != market or record.get("idx_nm") != _INDEX[market]:
            return None, "benchmark_key_mismatch"
        if record.get("session", record.get("bas_dd")) != day:
            return None, "benchmark_session_mismatch"
        if record.get("unique") is False:
            return None, "benchmark_not_unique"
        if not _known_at(record, cutoff):
            return None, "benchmark_unavailable_at_cutoff"
        pair.append(record.get("close_idx"))
    value = _ratio(pair[1], pair[0])
    return value, None if value is not None else "benchmark_price_invalid"


def calculate_row(
    slot: Mapping[str, Any],
    *,
    entry_session: Any,
    exit_session: Any,
    observations: Mapping[str, Any],
    cutoff: str | datetime,
    withdrawn_before_entry: bool = False,
) -> dict[str, Any]:
    """Calculate one original rank slot without I/O or fallback prices.

    Sessions are ISO dates or mappings with ``session``/``date`` and optional
    timezone-aware ``close_at``. Date-only KR sessions close at 15:30 KST.
    Provenance is retained verbatim under ``observations`` and original fields.
    """
    cutoff_at = _aware(cutoff)
    entry_day, entry_close = _session(entry_session)
    exit_day, exit_close = _session(exit_session)
    if exit_close <= entry_close:
        raise ValueError("exit session must follow entry session")
    result = dict(slot)
    result.update(
        entry_session=entry_day,
        exit_session=exit_day,
        calculation_at=cutoff_at.isoformat(),
        observations=deepcopy(dict(observations)),
        gross=None,
        net=None,
        benchmark=None,
        excess=None,
        cost=None,
        executed=False,
        entry_executed=False,
        exit_executed=False,
        mature=cutoff_at > exit_close,
        status="pending",
        reason="exit_session_not_complete",
        benchmark_reason=None,
    )
    if cutoff_at <= exit_close:
        return result
    benchmark, benchmark_reason = _benchmark(slot, observations, entry_day, exit_day, cutoff_at)
    result.update(benchmark=benchmark, benchmark_reason=benchmark_reason)

    def finish(status: str, reason: str) -> dict[str, Any]:
        result.update(status=status, reason=reason)
        if status in {"not_entered", "withdrawn_before_entry"}:
            result.update(gross=0.0, net=0.0, cost=0.0)
        if result["net"] is not None and benchmark is not None:
            result["excess"] = float(Decimal(str(result["net"])) - Decimal(str(benchmark)))
        return result

    if withdrawn_before_entry:
        return finish("withdrawn_before_entry", "explicit_withdrawal_before_entry")
    if observations.get("status") == "source_missing" or observations.get("source_missing") is True:
        return finish(
            "source_missing", str(observations.get("reason", "adapter_contract_unavailable"))
        )
    entry, exit_ = observations.get("entry"), observations.get("exit")
    if not isinstance(entry, Mapping):
        return finish("entry_unresolved", "entry_source_missing")
    if not _known_at(entry, cutoff_at):
        return finish("entry_unresolved", "entry_unavailable_at_cutoff")
    identity_reason = _identity_reason(slot, entry, entry, observations, entry_day, entry_day)
    if identity_reason:
        return finish("identity_unresolved", identity_reason)
    entry_reason = _execution_reason(entry)
    if entry_reason == "explicit_untradeable":
        return finish("not_entered", entry_reason)
    if entry_reason:
        return finish("entry_unresolved", entry_reason)
    # Entry can be known even when the scheduled exit cannot be resolved.
    if (
        entry.get("regular_session_corrected") is not True
        or not entry.get("snapshot_id")
        or not entry.get("adjustment_revision")
        or _number(entry.get("adjusted_close"), positive=True) is None
    ):
        return finish("source_missing", "entry_adjustment_proof_not_verified")
    result["entry_executed"] = True
    if not isinstance(exit_, Mapping):
        return finish("exit_unresolved", "exit_source_missing")
    if not _known_at(exit_, cutoff_at):
        return finish("exit_unresolved", "exit_unavailable_at_cutoff")
    identity_reason = _identity_reason(slot, entry, exit_, observations, entry_day, exit_day)
    if identity_reason:
        return finish("identity_unresolved", identity_reason)
    for record in (entry, exit_):
        if record.get("corporate_action", "none") not in {"none", "split", "reverse_split"}:
            return finish("corporate_action_unresolved", "unsupported_corporate_action")
    if (
        observations.get("same_adjustment_vintage") is not True
        or not entry.get("snapshot_id")
        or entry.get("snapshot_id") != exit_.get("snapshot_id")
        or not entry.get("adjustment_revision")
        or entry.get("adjustment_revision") != exit_.get("adjustment_revision")
        or any(record.get("regular_session_corrected") is not True for record in (entry, exit_))
    ):
        return finish("source_missing", "adjustment_chain_not_verified")
    exit_reason = _execution_reason(exit_)
    if exit_reason:
        return finish("exit_unresolved", exit_reason)
    gross = _ratio(exit_.get("adjusted_close"), entry.get("adjusted_close"))
    if gross is None:
        return finish("source_missing", "adjusted_price_missing_or_invalid")
    result.update(
        gross=gross,
        net=float(Decimal(str(gross)) - _COST),
        cost=float(_COST),
        executed=True,
        exit_executed=True,
    )
    return finish("available", "resolved")


def _average(values: Sequence[Any]) -> float | None:
    numbers = [_number(value) for value in values]
    if not numbers or any(value is None for value in numbers):
        return None
    with localcontext() as context:
        context.prec = 34
        result = float(sum(numbers, Decimal(0)) / len(numbers))
    return result if _number(result) is not None else None


def _size(cohort: Mapping[str, Any]) -> int:
    value = cohort.get("size", cohort.get("top_n", cohort.get("n_slots")))
    if isinstance(value, bool) or not isinstance(value, int) or value not in (20, 100):
        raise ValueError("cohort size must be the original top20 or top100")
    return int(value)


def aggregate_cohort(
    cohort: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate original equal weights, never average only surviving slots."""
    size = _size(cohort)
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("rank"), int)
        and not isinstance(row.get("rank"), bool)
        and 1 <= row["rank"] <= size
    ]
    rank_counts = Counter(row["rank"] for row in selected)
    missing = size - len(rank_counts)
    duplicate = sum(count > 1 for count in rank_counts.values())
    invalid = len(rows) - len(selected)
    # A duplicated rank has no unambiguous original row. Do not pick the first
    # candidate or count both executions against the same original weight.
    unique = [row for row in selected if rank_counts[row["rank"]] == 1]
    complete = missing == duplicate == invalid == 0
    unknown = (
        size
        - len(unique)
        + sum(
            _number(row.get("net")) is None or _number(row.get("benchmark")) is None
            for row in unique
        )
    )
    result = dict(cohort)
    result.update(
        size=size,
        slot_count=len(selected),
        original_slot_count=size,
        slot_weight=1 / size,
        missing_rank_count=missing,
        duplicate_rank_count=duplicate,
        invalid_rank_row_count=invalid,
        mature=complete and all(row.get("mature") is True for row in selected),
        net=None,
        benchmark=None,
        excess=None,
        gross=None,
        cost=None,
        status="source_incomplete",
        reason="original_slots_incomplete",
        resolved_slot_count=sum(
            row.get("status") in _RESOLVED
            and _number(row.get("net")) is not None
            and _number(row.get("benchmark")) is not None
            for row in unique
        ),
        unresolved_weight=unknown / size,
        nominal_buy_weight=sum(row.get("entry_executed") is True for row in unique) / size,
        nominal_sell_weight=sum(row.get("exit_executed") is True for row in unique) / size,
    )
    if not complete:
        return result
    if unknown == 0 and result["mature"]:
        result.update(
            net=_average([row.get("net") for row in selected]),
            benchmark=_average([row.get("benchmark") for row in selected]),
            gross=_average([row.get("gross") for row in selected]),
            cost=_average([row.get("cost") for row in selected]),
        )
    if result["net"] is not None and result["benchmark"] is not None:
        result["excess"] = float(Decimal(str(result["net"])) - Decimal(str(result["benchmark"])))
    if not result["mature"]:
        result.update(status="pending", reason="exit_session_not_complete")
    elif result["net"] is not None and result["benchmark"] is not None:
        result.update(status="available", reason="all_original_slots_resolved")
    else:
        result.update(status="unresolved", reason="original_slot_or_benchmark_unresolved")
    return result


def summarize(
    cohorts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Descriptive conditional means and counts only; cohorts are not independent."""
    publications = set()
    recommendation_keys = set()
    recommendations = []
    missing_ref_count = 0
    for offset, row in enumerate(rows):
        publication = row.get("report_sha256") or row.get("publication_ref")
        if isinstance(publication, Mapping):
            publication = publication.get("sha256") or publication.get("receipt_sha256")
        if not isinstance(publication, str) or not publication:
            missing_ref_count += 1
            # Unknown publication identities cannot safely be joined or deduplicated.
            publication = None
            key = ("unknown_publication", offset)
        else:
            publications.add(publication)
            key = (publication, row.get("symbol"), _slot_market(row))
        if key not in recommendation_keys:
            recommendation_keys.add(key)
            recommendations.append(row)
    output: dict[str, Any] = {
        "interpretation": "Conditional means of mature fully resolved cohorts only; "
        "overlapping cohorts are not independent samples and do not describe the full sample.",
        "return_definition": "Price returns excluding dividends; net uses fixed round-trip 60bp; "
        "price-index benchmark has zero cost.",
        "row_count": len(rows),
        "publication_count": len(publications),
        "missing_publication_reference_row_count": missing_ref_count,
        "recommendation_count": len(recommendations),
    }
    entities = Counter((row.get("symbol"), _slot_market(row)) for row in recommendations)
    output.update(
        unique_symbol_count=len(entities),
        repeated_recommendation_count=sum(n - 1 for n in entities.values()),
        repeated_symbol_count=sum(n > 1 for n in entities.values()),
    )
    for size, label in ((20, "top20"), (100, "top100")):
        group = [cohort for cohort in cohorts if _size(cohort) == size]
        mature = [cohort for cohort in group if cohort.get("mature") is True]
        resolved = [
            cohort
            for cohort in mature
            if cohort.get("status") == "available"
            and _number(cohort.get("net")) is not None
            and _number(cohort.get("benchmark")) is not None
            and _number(cohort.get("excess")) is not None
        ]
        weights = [_number(cohort.get("unresolved_weight")) for cohort in group]
        output[label] = {
            "role": "primary" if size == 20 else "secondary",
            "total_cohort_count": len(group),
            "mature_cohort_count": len(mature),
            "resolved_cohort_count": len(resolved),
            "unresolved_cohort_count": len(group) - len(resolved),
            "pending_cohort_count": sum(cohort.get("status") == "pending" for cohort in group),
            "source_incomplete_cohort_count": sum(
                cohort.get("status") == "source_incomplete" for cohort in group
            ),
            "unresolved_weight": (
                float(sum(weights, Decimal(0))) if all(w is not None for w in weights) else None
            ),
            "mean_net": _average([cohort["net"] for cohort in resolved]),
            "mean_benchmark": _average([cohort["benchmark"] for cohort in resolved]),
            "mean_excess": _average([cohort["excess"] for cohort in resolved]),
            "insufficient_sample": len(mature) < 20,
            "sample_status": "insufficient_sample" if len(mature) < 20 else "descriptive_only",
        }
    return output
