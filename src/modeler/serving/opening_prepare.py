"""Map private KIS slots to a calendar-checked, immutable opening artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

from collector.kr.adapters.kis_intraday import SEOUL, assess_session_status


def _bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _aware(value: str) -> datetime:
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("opening timestamps must include a timezone")
    return instant.astimezone(SEOUL)


def build_opening_artifact(*, report_date: date, calendar_manifest: dict[str, Any],
                           snapshots: list[dict[str, Any]], decision_at: datetime,
                           max_age_seconds: int) -> dict[str, Any]:
    """Map private 09:00/09:20/09:30 observations to SiteBuilder's opening schema.

    The KIS current-index responses do not currently prove exchange sample time.
    Received time is never substituted for that missing source timestamp.
    """
    from modeler.serving.calendars import SessionCalendar

    calendar = SessionCalendar.from_manifest(calendar_manifest)
    if calendar.market != "KR":
        raise ValueError("opening artifact requires a KR calendar")
    decision = decision_at.astimezone(SEOUL) if decision_at.tzinfo else None
    if decision is None or decision.date() != report_date:
        raise ValueError("decision_at must be aware and match report_date")
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    session = calendar.session(report_date)
    if session is None:
        calendar_status = "unavailable_closed_or_unknown_session"
    elif not session.confirmed:
        calendar_status = "unavailable_calendar_unverified"
    else:
        calendar_status = "confirmed"
    usable = []
    evidence = []
    for snapshot in snapshots:
        if snapshot.get("market") != "KR" or not isinstance(snapshot.get("observations"), list):
            raise ValueError("slot snapshot is not a KR observation")
        received_at = _aware(str(snapshot["received_at"]))
        if received_at.date() != report_date or received_at > decision:
            raise ValueError("slot snapshot is outside report date or after decision")
        digest = hashlib.sha256(_bytes(snapshot)).hexdigest()
        evidence.append({"received_at": received_at.isoformat(), "snapshot_sha256": digest,
                         "observation_count": len(snapshot["observations"])})
        usable.append((received_at, snapshot, digest))
    usable.sort(key=lambda item: item[0])
    selected = usable[-1] if usable else None
    status = "unavailable"
    assessment = "no_slot_snapshot"
    indices: list[dict[str, Any]] = []
    industries: list[dict[str, Any]] = []
    if selected is not None:
        received_at, snapshot, digest = selected
        open_at = (datetime.combine(report_date, session.open_at, SEOUL)
                   if session and session.confirmed else None)
        close_at = (datetime.combine(report_date, session.close_at, SEOUL)
                    if session and session.confirmed else None)
        assessment = assess_session_status(
            snapshot, calendar_open=session.confirmed if session else None,
            session_open_at=open_at, session_close_at=close_at,
            slot_at=received_at, max_age_seconds=max_age_seconds,
        )
        if calendar_status != "confirmed":
            assessment = calendar_status
        elif (decision - received_at).total_seconds() > max_age_seconds:
            assessment = "stale_slot_snapshot"
        status = "ok" if assessment == "open_session_observed" else "unavailable"
        for row in snapshot["observations"]:
            if not isinstance(row, dict):
                raise ValueError("observation must be an object")
            mapped = {"code": row.get("code"), "name": row.get("name"),
                      "last": row.get("price"), "change_pct": row.get("change_percent"),
                      "observed_at": row.get("source_observed_at"),
                      "publication": {"status": "unresolved", "evidence": []}}
            (indices if row.get("kind") == "index" else industries).append(mapped)
    return {
        "schema_version": 1, "market": "KR", "report_date": report_date.isoformat(),
        "synthetic_fixture": False,
        "status": status, "session_state": "regular" if status == "ok" else "unavailable",
        "assessment": assessment, "calendar_status": calendar_status,
        "decision_at": decision.isoformat(), "max_age_seconds": max_age_seconds,
        "selected_snapshot_sha256": selected[2] if selected else None,
        "received_at": selected[0].isoformat() if selected else None,
        "observed_at": (min(_aware(str(row["source_observed_at"]))
                            for row in selected[1]["observations"]).isoformat()
                        if selected and status == "ok" else None),
        "slot_evidence": evidence, "indices": indices, "industries": industries,
        "publication": {"status": "unresolved", "evidence": []},
    }


def publish_opening_artifact(*, artifact: dict[str, Any], output_root: Path) -> Path:
    report_date = date.fromisoformat(str(artifact["report_date"]))
    raw = _bytes(artifact)
    digest = hashlib.sha256(raw).hexdigest()
    directory = output_root / f"report_date={report_date.isoformat()}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"opening-{digest[:16]}.json"
    if target.exists():
        if target.read_bytes() != raw:
            raise FileExistsError("opening artifact digest collision")
        return target
    descriptor, name = tempfile.mkstemp(prefix=".opening-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", type=date.fromisoformat, required=True)
    parser.add_argument("--calendar-json", type=Path, required=True)
    parser.add_argument("--snapshot-json", type=Path, action="append", default=[])
    parser.add_argument("--decision-at", type=datetime.fromisoformat, required=True)
    parser.add_argument("--max-age-seconds", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = build_opening_artifact(
        report_date=args.report_date,
        calendar_manifest=json.loads(args.calendar_json.read_text()),
        snapshots=[json.loads(path.read_text()) for path in args.snapshot_json],
        decision_at=args.decision_at, max_age_seconds=args.max_age_seconds,
    )
    path = publish_opening_artifact(artifact=artifact, output_root=args.output_root)
    print(json.dumps({"path": str(path), "status": artifact["status"],
                      "assessment": artifact["assessment"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
