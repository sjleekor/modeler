"""Opening mapper keeps exchange sample time separate from HTTP receipt."""

from __future__ import annotations

from datetime import date, datetime

from modeler.serving.opening_prepare import build_opening_artifact, publish_opening_artifact


def _calendar(confirmed=True):
    return {
        "market": "KR", "timezone": "Asia/Seoul",
        "coverage_start": "2026-09-30", "coverage_end": "2026-09-30",
        "default_open_at": "09:00:00", "default_close_at": "15:30:00",
        "sessions": ["2026-09-30"], "overrides": [],
        "unconfirmed_dates": [] if confirmed else ["2026-09-30"],
    }


def _snapshot(sample_time=None):
    return {
        "market": "KR", "received_at": "2026-09-30T09:30:05+09:00",
        "observations": [
            {"kind": "index", "code": "0001", "name": "KOSPI", "price": "3210.4",
             "change_percent": "0.3", "source_observed_at": sample_time},
            {"kind": "sector", "code": "101", "name": "Sector", "price": "111.2",
             "change_percent": "0.1", "source_observed_at": sample_time},
        ],
    }


def _build(snapshot, calendar=None):
    return build_opening_artifact(
        report_date=date(2026, 9, 30), calendar_manifest=calendar or _calendar(),
        snapshots=[snapshot], decision_at=datetime.fromisoformat("2026-09-30T09:31:00+09:00"),
        max_age_seconds=120,
    )


def test_opening_requires_source_time_and_keeps_publication_unresolved(tmp_path):
    missing = _build(_snapshot())
    assert missing["status"] == "unavailable"
    assert missing["assessment"] == "observation_time_unverified"
    assert missing["observed_at"] is None
    assert missing["received_at"] == "2026-09-30T09:30:05+09:00"
    assert missing["publication"]["status"] == "unresolved"
    path = publish_opening_artifact(artifact=missing, output_root=tmp_path)
    assert path.is_file()
    assert publish_opening_artifact(artifact=missing, output_root=tmp_path) == path

    sourced = _build(_snapshot("2026-09-30T09:30:00+09:00"))
    assert sourced["status"] == "ok"
    assert sourced["observed_at"] == "2026-09-30T09:30:00+09:00"
    assert sourced["synthetic_fixture"] is False
    assert sourced["publication"]["status"] == "unresolved"
    unconfirmed = _build(_snapshot("2026-09-30T09:30:00+09:00"), _calendar(False))
    assert unconfirmed["status"] == "unavailable"
