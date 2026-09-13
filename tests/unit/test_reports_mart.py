"""Golden parity: DuckDB coverage/readiness reports == frozen Postgres reports.

The Postgres report services are removed at refactor P5, so their output on the
fixture scenario is frozen into ``golden/common_feature_reports.json`` and the
DuckDB ``reports`` mart is checked against that. The freshness gate is a mart-only
unit test (no oracle). Regenerate after an intentional change:

    SDC_UPDATE_GOLDEN=1 uv run pytest tests/unit/test_reports_mart.py

See refactor plan §4, §7.4.
"""

from __future__ import annotations

import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest
from modeler.etl.marts import common_build, reports

from ._common_fixtures import (
    MockCommonFeatureBuildStorage,
    _feature,
    _krx_days,
    _obs,
    _series,
)

_GOLDEN_PATH = Path(__file__).parent / "golden" / "common_feature_reports.json"


def _scenario():
    series = [_series("market_kospi", max_stale_business_days=2)]
    catalog = [_feature("market_kospi_close", transform_code="level")]
    # 2 days fresh, then a gap that turns stale -> nulls + missing in the window.
    obs = [
        _obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500"),
        _obs(2, "market_kospi", date(2026, 1, 6), date(2026, 1, 7), "2520"),
    ]
    return series, catalog, obs, date(2026, 1, 6), date(2026, 1, 12)


def _load_observations(con: duckdb.DuckDBPyConnection, observations) -> None:
    con.execute(
        "CREATE TABLE common_feature_observation_raw ("
        "raw_id BIGINT, source VARCHAR, series_id VARCHAR, observation_date DATE, "
        "period_end_date DATE, release_date DATE, available_from_date DATE, "
        "vintage VARCHAR, value_numeric DECIMAL(30,8), fetched_at TIMESTAMP, "
        "frequency VARCHAR)"
    )
    for o in observations:
        con.execute(
            "INSERT INTO common_feature_observation_raw VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                o.raw_id,
                o.source.value,
                o.series_id,
                o.observation_date,
                o.period_end_date,
                o.release_date,
                o.available_from_date,
                o.vintage,
                o.value_numeric,
                o.fetched_at.replace(tzinfo=None),
                o.frequency,
            ],
        )


def _build_mart_con(observations, series_list, catalog, start, end, monkeypatch):
    monkeypatch.setattr(common_build, "default_common_feature_series", lambda: series_list)
    monkeypatch.setattr(common_build, "default_common_feature_catalog", lambda: catalog)
    monkeypatch.setattr(reports, "default_common_feature_catalog", lambda: catalog)
    feature_dates = _krx_days(start, end)
    first_avail = min(
        (o.available_from_date for o in observations if o.available_from_date is not None),
        default=start,
    )
    trading_days = _krx_days(min(first_avail, start), end)
    con = duckdb.connect()
    _load_observations(con, observations)
    common_build.register_common_feature_daily_fact_view(
        con, trading_days=trading_days, feature_dates=feature_dates
    )
    return con, feature_dates


def _oracle_reports() -> dict:
    """Postgres coverage + readiness for the scenario (regen only; lazy import)."""
    try:
        from collector.kr.service.build_common_feature_daily_facts import (
            build_common_feature_daily_facts,
        )
        from collector.kr.service.report_common_feature_coverage import (
            build_common_feature_coverage_report,
        )
        from collector.kr.service.report_common_feature_readiness import (
            build_common_feature_readiness_report,
        )
    except ModuleNotFoundError as exc:  # P5 removed the oracle
        raise RuntimeError(
            "report services were decommissioned (refactor P5); the golden is now "
            "the source of truth — edit the mart and review the golden diff manually."
        ) from exc

    series, catalog, obs, start, end = _scenario()
    storage = MockCommonFeatureBuildStorage(series=series, catalog=catalog, observations=obs)
    build_common_feature_daily_facts(storage, start, end, krx_trading_days=_krx_days)
    storage.get_common_feature_daily_facts = lambda start, end, feature_codes=None: [  # type: ignore[attr-defined]
        f for f in storage.facts if start <= f.feature_date <= end
    ]
    cov = build_common_feature_coverage_report(storage, start, end, krx_trading_days=_krx_days)
    rdy = build_common_feature_readiness_report(storage, start, end, krx_trading_days=_krx_days)
    return {
        "coverage": {
            r.feature_code: {
                "target_count": r.target_count,
                "fact_count": r.fact_count,
                "non_null_count": r.non_null_count,
                "null_count": r.null_count,
                "missing_count": r.missing_count,
                "pit_violation_count": r.pit_violation_count,
                "coverage_ratio": str(r.coverage_ratio),
            }
            for r in cov.rows
        },
        "readiness": {r.feature_code: r.ready for r in rdy.rows},
    }


def _maybe_update_golden() -> None:
    if os.environ.get("SDC_UPDATE_GOLDEN") != "1":
        return
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN_PATH.write_text(json.dumps(_oracle_reports(), indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def golden():
    _maybe_update_golden()
    return json.loads(_GOLDEN_PATH.read_text())


def test_coverage_matches_golden(golden, monkeypatch):
    series, catalog, obs, start, end = _scenario()
    con, feature_dates = _build_mart_con(obs, series, catalog, start, end, monkeypatch)
    mart = {r.feature_code: r for r in reports.coverage_report(con, feature_dates=feature_dates)}
    g = golden["coverage"]

    assert set(g) == set(mart)
    for code, exp in g.items():
        m = mart[code]
        assert m.target_count == exp["target_count"], code
        assert m.fact_count == exp["fact_count"], code
        assert m.non_null_count == exp["non_null_count"], code
        assert m.null_count == exp["null_count"], code
        assert m.missing_count == exp["missing_count"], code
        assert m.pit_violation_count == exp["pit_violation_count"], code
        assert Decimal(str(m.coverage_ratio)) == Decimal(exp["coverage_ratio"]), code


def test_readiness_matches_golden(golden, monkeypatch):
    series, catalog, obs, start, end = _scenario()
    con, feature_dates = _build_mart_con(obs, series, catalog, start, end, monkeypatch)
    mart = {
        r.feature_code: r.ready for r in reports.readiness_report(con, feature_dates=feature_dates)
    }
    assert {k: bool(v) for k, v in golden["readiness"].items()} == mart


def test_readiness_includes_catalog_feature_with_zero_facts(monkeypatch):
    series = [_series("market_kospi", max_stale_business_days=2)]
    catalog = [
        _feature("market_kospi_close", transform_code="level"),
        _feature("missing_feature", transform_code="level", series_id="missing_series"),
    ]
    obs = [_obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500")]
    con, feature_dates = _build_mart_con(
        obs,
        series,
        catalog,
        date(2026, 1, 6),
        date(2026, 1, 7),
        monkeypatch,
    )

    rows = {r.feature_code: r for r in reports.readiness_report(con, feature_dates=feature_dates)}

    assert "missing_feature" in rows
    assert rows["missing_feature"].ready is False
    assert any("missing_count=2" == blocker for blocker in rows["missing_feature"].blockers)


def test_freshness_gate_flags_stale_series():
    # A daily series whose latest observation lags far behind `end` must violate.
    obs = [_obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500")]
    con = duckdb.connect()
    _load_observations(con, obs)
    # Register a minimal common_feature_series table for the gate.
    con.execute(
        "CREATE TABLE common_feature_series (series_id VARCHAR, frequency VARCHAR, "
        "manual_lag_days INTEGER, max_stale_business_days INTEGER, active BOOLEAN)"
    )
    con.execute("INSERT INTO common_feature_series VALUES ('market_kospi', 'D', 0, 2, TRUE)")
    res = reports.freshness_violations(con, end=date(2026, 2, 1))
    assert not res.ok
    assert res.violations[0].series_id == "market_kospi"


def test_freshness_gate_passes_when_fresh():
    series_obs = [_obs(1, "market_kospi", date(2026, 1, 30), date(2026, 1, 31), "2500")]
    con = duckdb.connect()
    _load_observations(con, series_obs)
    con.execute(
        "CREATE TABLE common_feature_series (series_id VARCHAR, frequency VARCHAR, "
        "manual_lag_days INTEGER, max_stale_business_days INTEGER, active BOOLEAN)"
    )
    con.execute("INSERT INTO common_feature_series VALUES ('market_kospi', 'D', 0, 5, TRUE)")
    res = reports.freshness_violations(con, end=date(2026, 2, 2))
    assert res.ok
