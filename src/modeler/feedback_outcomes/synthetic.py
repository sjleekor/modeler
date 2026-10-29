"""Code-generated fixtures only. Calendar dates and all prices are invented."""

from __future__ import annotations

from datetime import date, timedelta

from modeler.feedback.core import canonical_bytes, identity, sha256_bytes

from .contracts import BUNDLE_SHA256, MODEL_ID, MODEL_VERSION, RULE_ID, RULE_SHA256
from .prices import EXTRACT_SCHEMA, PROOF_KEYS


def fixture():
    start, end = date(2026, 11, 2), date(2027, 2, 26)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    calendar = {
        "market": "KR",
        "timezone": "Asia/Seoul",
        "coverage_start": str(start),
        "coverage_end": str(end),
        "sessions": [str(d) for d in days if d.weekday() < 5],
        "default_open_at": "09:00",
        "default_close_at": "15:30",
        "unconfirmed_dates": [],
        "overrides": [],
        "synthetic_fixture": True,
        "source": {"basis": "invented weekday-only fixture, not an operating calendar"},
    }
    calendar["calendar_sha256"] = sha256_bytes(canonical_bytes(calendar))
    events = []
    for day, previous in (("2026-11-04", "2026-11-03"), ("2026-11-05", "2026-11-04")):
        report_hash = sha256_bytes(f"synthetic report {day}".encode())
        invocation = "synthetic-" + day
        rows = []
        for rank in range(1, 101):
            symbol, market = f"{rank:06d}", "KOSPI" if rank % 2 else "KOSDAQ"
            fields = {
                "market": "KR",
                "security_market": market,
                "model_id": MODEL_ID,
                "bundle_hash": BUNDLE_SHA256,
                "report_sha256": report_hash,
                "invocation_id": invocation,
                "symbol": symbol,
            }
            rows.append(
                {
                    "prediction_ref": identity("prediction", fields),
                    "symbol": symbol,
                    "security_market": market,
                    "rank": rank,
                    "score": 1 - rank / 200,
                    "quality_reasons": ["W20_halt_in_window"] if rank == 1 else [],
                }
            )
        events.append(
            {
                "kind": "publication",
                "market": "KR",
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "bundle_hash": BUNDLE_SHA256,
                "input_sha256": sha256_bytes(f"synthetic input {day}".encode()),
                "feature_asof_date": previous,
                "freshness_lag_sessions": 0,
                "model_status": "partial",
                "report_date": day,
                "decision_at": day + "T10:00:00+09:00",
                "verified_by": day + "T10:03:00+09:00",
                "coverage_status": "covered",
                "collection_started_at": day + "T10:01:00+09:00",
                "publication_ref": sha256_bytes(f"synthetic publication {day}".encode()),
                "receipt_ref": sha256_bytes(f"synthetic receipt {day}".encode()),
                "report_sha256": report_hash,
                "invocation_id": invocation,
                "reports_commit": "a" * 40,
                "unit_sha256": "b" * 64,
                "revision": 1,
                "synthetic_fixture": True,
                "historical_replay": False,
                "rows": rows,
            }
        )
    from .selection import Calendar, select_cohorts

    cohorts = select_cohorts(events, Calendar.from_manifest(calendar), synthetic=True)["cohorts"]
    sessions = sorted({c[k] for c in cohorts for k in ("entry_session", "exit_session")})
    extract = {
        "schema": EXTRACT_SCHEMA,
        "synthetic_fixture": True,
        "market": "KR",
        "rule_id": RULE_ID,
        "rule_sha256": RULE_SHA256,
        "source_tables": ["daily_ohlcv", "daily_market_cap", "krx_index_daily"],
        "snapshot_id": "synthetic-snapshot",
        "adjustment_revision": "synthetic-v1",
        "factor_convention": "adjusted_close=source_close*factor",
        "regular_session_corrected": True,
        "proof": {k: sha256_bytes(("synthetic " + k).encode()) for k in PROOF_KEYS},
        "daily_ohlcv": [],
        "daily_market_cap": [],
        "krx_index_daily": [],
        "listing_periods": [],
        "corporate_actions": [],
    }
    for rank in range(1, 101):
        symbol, market = f"{rank:06d}", "KOSPI" if rank % 2 else "KOSDAQ"
        extract["listing_periods"].append(
            {
                "ticker": symbol,
                "market": market,
                "entity_id": "synthetic-entity-" + symbol,
                "valid_from": "2026-01-01",
                "valid_to": None,
                "available_at": "2026-11-02T16:00:00+09:00",
            }
        )
        for day in sessions:
            close = 100 if day < "2026-12-01" else 110
            fields = {
                "ticker": symbol,
                "market": market,
                "trade_date": day,
                "available_at": day + "T17:00:00+09:00",
            }
            extract["daily_ohlcv"].append(
                {
                    **fields,
                    "close": close,
                    "adjustment_factor": 1,
                    "source_ref": "synthetic-ohlcv-" + day + symbol,
                }
            )
            extract["daily_market_cap"].append(
                {
                    **fields,
                    "source_open": close - 1,
                    "source_low": close - 2,
                    "source_high": close + 2,
                    "source_close": close,
                    "volume": 10000,
                    "trading_value": close * 10000,
                    "source_ref": "synthetic-market-cap-" + day + symbol,
                }
            )
    for day in sessions:
        for group, name in (("kospi", "코스피"), ("kosdaq", "코스닥")):
            extract["krx_index_daily"].append(
                {
                    "index_group": group,
                    "idx_nm": name,
                    "bas_dd": day,
                    "close_idx": 1000 if day < "2026-12-01" else 1050,
                    "available_at": day + "T17:00:00+09:00",
                    "source_ref": "synthetic-index-" + day + group,
                }
            )
    return {
        "calendar": calendar,
        "events": events,
        "extract": extract,
        "cutoff": "2026-12-04T13:00:00+09:00",
        "synthetic_fixture": True,
        "data_origin": "generated-in-code; no real source input",
    }
