"""Pure adapter for explicit KR endpoint extracts, never a lake/DB reader.

Existing daily_ohlcv.close alone is insufficient: the extract must carry pinned
regular-session correction, adjustment, identity and action evidence. Missing
proof is source_missing; an unadjusted or Naver-only close is never substituted.
"""

from __future__ import annotations

import math
from datetime import date

from modeler.feedback.core import require_hash

from .contracts import INDEX_KEYS, RULE_ID, RULE_SHA256, OutcomeError, stamp

EXTRACT_SCHEMA = "kr-v1-outcome-endpoint-extract.v1"
PROOF_KEYS = (
    "source_snapshot_sha256",
    "adjustment_code_sha256",
    "extraction_code_sha256",
    "regular_close_binding_sha256",
    "identity_registry_sha256",
    "capital_actions_sha256",
)


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise OutcomeError("extract session must be a canonical date")
    return value


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _unique(rows, keys, cutoff):
    if not isinstance(rows, list):
        raise OutcomeError("extract rows must be explicit lists")
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            raise OutcomeError("invalid extract row")
        key = tuple(row.get(k) for k in keys)
        if any(not isinstance(k, str) or not k for k in key) or key in output:
            raise OutcomeError("duplicate or incomplete extract identity")
        if stamp(row.get("available_at")) > cutoff:
            raise OutcomeError("extract includes evidence unavailable at cutoff")
        output[key] = dict(row)
    return output


class EndpointExtract:
    """Validate a prepared snapshot in memory, then select only exact E/X rows."""

    def __init__(self, value, *, cutoff, synthetic=False, sessions=None):
        self.cutoff = stamp(cutoff)
        if not isinstance(value, dict) or value.get("schema") != EXTRACT_SCHEMA:
            raise OutcomeError("registered endpoint extract required")
        if value.get("synthetic_fixture") is not synthetic:
            raise OutcomeError("fixture and real extracts cannot be mixed")
        if value.get("rule_id") != RULE_ID or value.get("rule_sha256") != RULE_SHA256:
            raise OutcomeError("extract rule pin mismatch")
        if value.get("market") != "KR" or value.get("source_tables") != [
            "daily_ohlcv",
            "daily_market_cap",
            "krx_index_daily",
        ]:
            raise OutcomeError("only the registered KR tables are permitted")
        self.proof = value.get("proof", {})
        self.complete = (
            value.get("regular_session_corrected") is True
            and value.get("factor_convention") == "adjusted_close=source_close*factor"
            and isinstance(value.get("snapshot_id"), str)
            and bool(value["snapshot_id"])
            and isinstance(value.get("adjustment_revision"), str)
            and bool(value["adjustment_revision"])
        )
        for key in PROOF_KEYS:
            try:
                require_hash(self.proof.get(key))
            except (ValueError, AttributeError):
                self.complete = False
        try:
            require_hash(self.proof.get("identity_registry_sha256"))
            self.identity_complete = True
        except (ValueError, AttributeError):
            self.identity_complete = False
        self.snapshot = value.get("snapshot_id")
        self.revision = value.get("adjustment_revision")
        self.prices = _unique(
            value.get("daily_ohlcv"), ("ticker", "market", "trade_date"), self.cutoff
        )
        self.caps = _unique(
            value.get("daily_market_cap"), ("ticker", "market", "trade_date"), self.cutoff
        )
        self.indices = _unique(
            value.get("krx_index_daily"), ("index_group", "idx_nm", "bas_dd"), self.cutoff
        )
        self.periods = value.get("listing_periods")
        self.actions = value.get("corporate_actions")
        if not isinstance(self.periods, list) or not isinstance(self.actions, list):
            raise OutcomeError("explicit identity and capital-action evidence required")
        allowed = set(sessions) if sessions is not None else None
        for rows in (self.prices, self.caps):
            for key, row in rows.items():
                if key[1] not in INDEX_KEYS:
                    raise OutcomeError("non-KR security market in extract")
                day = _day(row["trade_date"])
                if allowed is not None and day not in allowed:
                    raise OutcomeError("extract includes a session outside the read plan")
        for key, row in self.indices.items():
            day = _day(row["bas_dd"])
            if key[:2] not in INDEX_KEYS.values() or (allowed is not None and day not in allowed):
                raise OutcomeError("unexpected index or session in extract")

    def _identity(self, slot, first, last):
        if not self.identity_complete:
            return None
        first = min(_day(slot.get("report_date", first)), first)
        rows = [
            p
            for p in self.periods
            if isinstance(p, dict)
            and p.get("ticker") == slot["symbol"]
            and p.get("market") == slot["security_market"]
            and _day(p.get("valid_from")) <= first
            and (p.get("valid_to") is None or _day(p["valid_to"]) >= last)
        ]
        if (
            len(rows) != 1
            or not isinstance(rows[0].get("entity_id"), str)
            or not rows[0]["entity_id"]
        ):
            return None
        period = rows[0]
        if stamp(period.get("available_at")) > self.cutoff:
            return None
        # Any overlapping second listing identity is unresolved, even if one spans E/X.
        overlaps = [
            p
            for p in self.periods
            if isinstance(p, dict)
            and p.get("ticker") == slot["symbol"]
            and _day(p.get("valid_from")) <= last
            and (p.get("valid_to") is None or _day(p["valid_to"]) >= first)
        ]
        return period if len(overlaps) == 1 else None

    def observations(self, slot, entry_session, exit_session):
        first, last = _day(entry_session), _day(exit_session)
        entry_period = self._identity(slot, first, first)
        common = {
            "identity_verified": entry_period is not None,
            "same_adjustment_vintage": self.complete,
            "extract_proof": self.proof,
        }
        for label, day in (("entry", first), ("exit", last)):
            period = entry_period if label == "entry" else self._identity(slot, first, last)
            key = (slot["symbol"], slot["security_market"], day)
            price, cap = self.prices.get(key), self.caps.get(key)
            if cap is not None or price is not None:
                cap, price = cap or {}, price or {}
                raw_close, factor, adjusted = (
                    cap.get("source_close"),
                    price.get("adjustment_factor"),
                    price.get("close"),
                )
                bound = self.complete and all(_positive(v) for v in (raw_close, factor, adjusted))
                bound = bound and price.get("snapshot_id", self.snapshot) == self.snapshot
                bound = bound and price.get("adjustment_revision", self.revision) == self.revision
                if bound:
                    bound = math.isclose(adjusted, raw_close * factor, rel_tol=1e-9, abs_tol=1e-8)
                actions = [
                    a
                    for a in self.actions
                    if isinstance(a, dict)
                    and a.get("ticker") == slot["symbol"]
                    and a.get("market") == slot["security_market"]
                    and first < _day(a.get("effective_date")) <= last
                ]
                action_kind = "none"
                for action in actions:
                    if stamp(action.get("available_at")) > self.cutoff:
                        action_kind = "unresolved"
                    elif action.get("kind") not in {"none", "split", "reverse_split"}:
                        action_kind = action.get("kind", "unresolved")
                    elif action_kind == "none":
                        action_kind = action["kind"]
                common[label] = {
                    **{
                        k: cap.get(k)
                        for k in (
                            "source_open",
                            "source_high",
                            "source_low",
                            "source_close",
                            "volume",
                            "trading_value",
                        )
                    },
                    "adjusted_close": adjusted,
                    "adjustment_factor": factor,
                    "explicit_untradeable": cap.get("explicit_untradeable", False),
                    "tradeability_conflict": cap.get("tradeability_conflict", False),
                    "snapshot_id": price.get("snapshot_id", self.snapshot),
                    "adjustment_revision": price.get("adjustment_revision", self.revision),
                    "regular_session_corrected": bound,
                    "corporate_action": action_kind,
                    "entity_id": period.get("entity_id") if period else None,
                    "listing_valid_from": period.get("valid_from") if period else None,
                    "listing_valid_to": period.get("valid_to") if period else None,
                    "market": slot["security_market"],
                    "symbol": slot["symbol"],
                    "session": day,
                    "available_at": max(
                        [v for v in (cap.get("available_at"), price.get("available_at")) if v],
                        key=stamp,
                    ),
                    "source_ref": {
                        "ohlcv": price.get("source_ref"),
                        "market_cap": cap.get("source_ref"),
                        **self.proof,
                    },
                }
                if cap.get("explicit_untradeable") is True:
                    try:
                        require_hash(cap.get("untradeable_evidence_sha256"))
                        trusted = cap.get("untradeable_basis") == "explicit_krx_status"
                    except ValueError:
                        trusted = False
                    if not trusted:
                        common[label].update(explicit_untradeable=False, tradeability_conflict=True)
            group, name = INDEX_KEYS[slot["security_market"]]
            index = self.indices.get((group, name, day))
            if index:
                common["benchmark_" + label] = {**index, "session": day}
        return common
