"""A synthetic frozen release with a wider calendar and any set of prepared inputs (R3 scenarios).

``test_daily_coordinator_e2e._setup`` builds the frozen release, the ops config and one KR and one
US prepared input for 2026-09-28.  This module replaces the prepared inputs and the calendars so a
scenario can say "KR prepared only up to 09-23" or "US prepared input finished at 09:40".
Report date D is 2026-09-29 (Tue): K = 09-28 (Mon), the US session before D is 09-28.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import test_daily_coordinator_e2e as e2e

D = e2e.D
SELECT_AT = "2026-09-29T09:30:00+09:00"
RUN_AT = "2026-09-29T10:00:00+09:00"
MODELS = (
    ("KR", "kr_daily_h20_v1"),
    ("US", "us_exploratory_20260929_r1_lightgbm"),
    ("US", "us_exploratory_20260929_r1_ridge"),
)

#: Like the real adapters: the section takes the freshness status ("ok" or "stale") and carries
#: three ranked rows, so a withheld ranking is visible.  ``R3_SLOW_FLAG`` makes inference hang.
ADAPTER_R3 = """import os
import time

from modeler.serving.schema import report_template


def _infer(context):
    flag = os.environ.get("R3_SLOW_FLAG")
    if flag and os.path.exists(flag):
        with open(flag + ".pid", "w") as handle:
            handle.write(str(os.getpid()))
        time.sleep(120)
    report = report_template(market=context.market, report_date=context.report_date.isoformat(),
        decision_at=context.decision_at, feature_asof_date=context.feature_asof_date,
        model_id=context.model_id, model_version=context.model_version)
    report["status"] = context.freshness_status
    report["synthetic_fixture"] = True
    report["rankings"] = [{"rank": rank, "symbol": f"S{rank}", "name": f"Name {rank}",
                           "score": 1.0 / rank} for rank in (1, 2, 3)]
    report["provenance"] = {"freshness": context.freshness}
    return report


def infer_kr_daily(context):
    return _infer(context)


def infer_us_model(context):
    return _infer(context)
"""


def weekdays(start: date, end: date) -> list[date]:
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _calendar(market: str, start: date, end: date) -> dict:
    return {
        "market": market,
        "timezone": "Asia/Seoul" if market == "KR" else "America/New_York",
        "coverage_start": start.isoformat(),
        "coverage_end": end.isoformat(),
        "sessions": [d.isoformat() for d in weekdays(start, end)],
        "default_open_at": "09:00" if market == "KR" else "09:30",
        "default_close_at": "15:30" if market == "KR" else "16:00",
        "overrides": [],
        "unconfirmed_dates": [],
    }


def write_native(
    prepared: Path, market: str, asof: str, ready_at: str, *, extra_native: dict | None = None
) -> Path:
    """One prepared input (features, native manifest, completion marker) for ``asof``."""
    directory = prepared / market.lower() / f"score_date={asof}" / f"prep_id=r3-{asof}"
    directory.mkdir(parents=True)
    feature = directory / ("feature_panel.parquet" if market == "KR" else "features.parquet")
    feature.write_bytes(f"synthetic model input {market} {asof}".encode())
    native = e2e._write(
        directory / ("prepare_manifest.json" if market == "KR" else "manifest.json"),
        {
            "market": market,
            "feature_asof_date": asof,
            "input_sha256": e2e._sha(feature),
            "features_sha256": e2e._sha(feature),
            "synthetic_fixture": True,
            "availability_evidence_type": "prepared_features_completion",
            **(extra_native or {}),
        },
    )
    e2e._write(
        directory / "completion.json",
        {
            "schema_version": "prepared-features-completion.v1",
            "verified_available_by": ready_at,
            "availability_evidence_type": "prepared_features_completion",
            "features_sha256": e2e._sha(feature),
            "native_prepare_manifest_sha256": e2e._sha(native),
        },
    )
    return directory


def build(
    tmp_path: Path,
    monkeypatch,
    *,
    kr: tuple = (),
    us: tuple = (),
    calendar_start: date = date(2026, 9, 14),
    lag_limit: int = 1,
) -> tuple[Path, Path]:
    """``kr`` / ``us``: (asof, completed_at) pairs of the prepared inputs that exist."""
    monkeypatch.setattr(e2e, "ADAPTER", ADAPTER_R3)
    release, config_path = e2e._setup(tmp_path)
    prepared = tmp_path / "prepared"
    shutil.rmtree(prepared / "kr")
    shutil.rmtree(prepared / "us")
    for market, items in (("KR", kr), ("US", us)):
        for asof, ready_at in items:
            write_native(prepared, market, asof, ready_at)
    config = json.loads(config_path.read_text())
    for key, market in (("kr_calendar", "KR"), ("us_calendar", "US")):
        e2e._write(Path(config[key]), _calendar(market, calendar_start, D))
    expected = json.loads(Path(config["us_expected_source"]).read_text())
    expected["market_lag_limit_sessions"] = lag_limit
    e2e._write(Path(config["us_expected_source"]), expected)
    return release, config_path


def wrapper(release: Path, config: Path, stage: str, now: str):
    """The real ``daily_wrapper`` in a subprocess, as Cronicle runs it (fixture clock)."""
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "modeler.serving.daily_wrapper",
            stage,
            "--config",
            str(config),
            "--report-date",
            D.isoformat(),
            "--fixture-now",
            now,
        ],
        cwd=release,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / D.isoformat()


def report(tmp_path: Path) -> dict:
    return json.loads((run_dir(tmp_path) / f"report-{D}.json").read_text())


def section(rep: dict, model_id: str) -> dict:
    return next(item for item in rep["markets"] if item["model_id"] == model_id)
