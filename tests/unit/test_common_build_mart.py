"""Golden parity test: DuckDB common_feature_daily_fact mart == frozen build output.

The DuckDB common-build mart must reproduce the Postgres build for every
transform (level/ret/change/vol/stale/yoy/mom/spread/ratio/latest-vintage). The
oracle (``build_common_feature_daily_facts``) was removed at refactor P5, so its
output on each synthetic scenario was frozen once into
``golden/common_feature_daily_fact.json`` and the mart is checked against that —
the golden is the source of truth here; there is no regen path (modeler/ has no
Postgres access by design, S6 §2.7). Edit the mart and review the golden diff
manually for an intentional build change.

See ``docs/dev/20260728_refactor_pipeline/00_refactor_plan.md`` §3.2, §7.4.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest
from collector.kr.shared import Source

from ._common_fixtures import (
    _feature,
    _krx_days,
    _multi_feature,
    _obs,
    _series,
)

# vol ends in a square root (Decimal.sqrt vs DuckDB DOUBLE) -> small relative
# tolerance; every other transform is exact (plan §7.4).
_VOL_REL_TOL = Decimal("1e-9")
_GOLDEN_PATH = Path(__file__).parent / "golden" / "common_feature_daily_fact.json"
_ECOS = Source.ECOS


def _d(days: int) -> timedelta:
    return timedelta(days=days)


@dataclass(frozen=True)
class Scenario:
    name: str
    build: Callable  # () -> (series, catalog, obs, start, end)
    is_vol: bool = False


# --- scenario definitions (single source of inputs for oracle + mart) ------


def _sc_level():
    series = [_series("market_kospi")]
    catalog = [_feature("market_kospi_close", transform_code="level")]
    obs = [
        _obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500"),
        _obs(2, "market_kospi", date(2026, 1, 6), date(2026, 1, 7), "2520"),
    ]
    return series, catalog, obs, date(2026, 1, 6), date(2026, 1, 9)


def _sc_return():
    series = [_series("market_kospi")]
    catalog = [_feature("market_kospi_ret_1d", transform_code="ret_1d")]
    obs = [
        _obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500"),
        _obs(2, "market_kospi", date(2026, 1, 6), date(2026, 1, 7), "2550"),
        _obs(3, "market_kospi", date(2026, 1, 7), date(2026, 1, 8), "2601"),
    ]
    return series, catalog, obs, date(2026, 1, 6), date(2026, 1, 9)


def _sc_change():
    series = [_series("market_kospi")]
    catalog = [_feature("market_kospi_change_1d", transform_code="change_1d")]
    obs = [
        _obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500"),
        _obs(2, "market_kospi", date(2026, 1, 6), date(2026, 1, 7), "2550"),
        _obs(3, "market_kospi", date(2026, 1, 7), date(2026, 1, 8), "2530"),
    ]
    return series, catalog, obs, date(2026, 1, 6), date(2026, 1, 9)


def _sc_vol():
    series = [_series("market_kospi")]
    catalog = [_feature("market_kospi_vol_3d", transform_code="vol_3d")]
    obs = [
        _obs(i + 1, "market_kospi", date(2026, 1, 1) + _d(i), date(2026, 1, 2) + _d(i), v)
        for i, v in enumerate(["2400", "2450", "2410", "2480", "2500", "2520"])
    ]
    return series, catalog, obs, date(2026, 1, 8), date(2026, 1, 9)


def _sc_stale():
    series = [_series("market_kospi", max_stale_business_days=2)]
    catalog = [_feature("market_kospi_close", transform_code="level")]
    obs = [_obs(1, "market_kospi", date(2026, 1, 5), date(2026, 1, 6), "2500")]
    return series, catalog, obs, date(2026, 1, 6), date(2026, 1, 15)


def _sc_yoy():
    series = [_series("macro_cpi", source=_ECOS, max_stale_business_days=400)]
    catalog = [_feature("macro_cpi_yoy", transform_code="yoy", series_id="macro_cpi")]
    obs = [
        _obs(
            1,
            "macro_cpi",
            date(2024, 1, 31),
            date(2024, 2, 20),
            "100",
            period_end_date=date(2024, 1, 31),
            source=_ECOS,
        ),
        _obs(
            2,
            "macro_cpi",
            date(2025, 1, 31),
            date(2025, 2, 20),
            "103",
            period_end_date=date(2025, 1, 31),
            source=_ECOS,
        ),
    ]
    return series, catalog, obs, date(2025, 2, 20), date(2025, 2, 20)


def _sc_yoy_null():
    series = [_series("macro_cpi", source=_ECOS, max_stale_business_days=400)]
    catalog = [_feature("macro_cpi_yoy", transform_code="yoy", series_id="macro_cpi")]
    obs = [
        _obs(
            1,
            "macro_cpi",
            date(2024, 2, 29),
            date(2024, 3, 20),
            "100",
            period_end_date=date(2024, 2, 29),
            source=_ECOS,
        ),
        _obs(
            2,
            "macro_cpi",
            date(2025, 1, 31),
            date(2025, 2, 20),
            "103",
            period_end_date=date(2025, 1, 31),
            source=_ECOS,
        ),
    ]
    return series, catalog, obs, date(2025, 2, 20), date(2025, 2, 20)


def _sc_mom():
    series = [_series("macro_cpi", source=_ECOS, max_stale_business_days=400)]
    catalog = [_feature("macro_cpi_mom", transform_code="mom", series_id="macro_cpi")]
    obs = [
        _obs(
            1,
            "macro_cpi",
            date(2024, 12, 31),
            date(2025, 1, 20),
            "100",
            period_end_date=date(2024, 12, 31),
            source=_ECOS,
        ),
        _obs(
            2,
            "macro_cpi",
            date(2025, 1, 31),
            date(2025, 2, 20),
            "102",
            period_end_date=date(2025, 1, 31),
            source=_ECOS,
        ),
    ]
    return series, catalog, obs, date(2025, 2, 20), date(2025, 2, 20)


def _sc_spread():
    series = [_series("rate_kr_gov10y", source=_ECOS), _series("rate_kr_gov3y", source=_ECOS)]
    catalog = [
        _multi_feature(
            "rate_kr_term_spread_10y_3y",
            transform_code="spread",
            inputs=(("rate_kr_gov10y", "spread_long"), ("rate_kr_gov3y", "spread_short")),
        )
    ]
    obs = [
        _obs(1, "rate_kr_gov10y", date(2026, 6, 8), date(2026, 6, 9), "3.40", source=_ECOS),
        _obs(2, "rate_kr_gov3y", date(2026, 6, 9), date(2026, 6, 10), "3.25", source=_ECOS),
    ]
    return series, catalog, obs, date(2026, 6, 9), date(2026, 6, 10)


def _sc_ratio():
    series = [_series("series_num", source=_ECOS), _series("series_den", source=_ECOS)]
    catalog = [
        _multi_feature(
            "some_ratio",
            transform_code="ratio",
            inputs=(("series_num", "numerator"), ("series_den", "denominator")),
        )
    ]
    obs = [
        _obs(1, "series_num", date(2026, 6, 8), date(2026, 6, 9), "6", source=_ECOS),
        _obs(2, "series_den", date(2026, 6, 8), date(2026, 6, 9), "3", source=_ECOS),
        _obs(3, "series_num", date(2026, 6, 9), date(2026, 6, 10), "6", source=_ECOS),
        _obs(4, "series_den", date(2026, 6, 9), date(2026, 6, 10), "0", source=_ECOS),
    ]
    return series, catalog, obs, date(2026, 6, 9), date(2026, 6, 10)


def _sc_vintage():
    series = [_series("macro_cpi", source=_ECOS, max_stale_business_days=400)]
    catalog = [_feature("macro_cpi_level", transform_code="level", series_id="macro_cpi")]
    obs = [
        _obs(
            1,
            "macro_cpi",
            date(2025, 1, 31),
            date(2025, 2, 20),
            "100",
            period_end_date=date(2025, 1, 31),
            release_date=date(2025, 2, 20),
            vintage="v1",
            source=_ECOS,
        ),
        _obs(
            2,
            "macro_cpi",
            date(2025, 1, 31),
            date(2025, 2, 25),
            "101",
            period_end_date=date(2025, 1, 31),
            release_date=date(2025, 2, 25),
            vintage="v2",
            source=_ECOS,
        ),
    ]
    return series, catalog, obs, date(2025, 2, 25), date(2025, 2, 26)


SCENARIOS = [
    Scenario("level", _sc_level),
    Scenario("return", _sc_return),
    Scenario("change", _sc_change),
    Scenario("volatility", _sc_vol, is_vol=True),
    Scenario("stale", _sc_stale),
    Scenario("yoy", _sc_yoy),
    Scenario("yoy_null", _sc_yoy_null),
    Scenario("mom", _sc_mom),
    Scenario("spread", _sc_spread),
    Scenario("ratio", _sc_ratio),
    Scenario("latest_vintage", _sc_vintage),
]


# --- mart side -------------------------------------------------------------


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


def _run_mart(scenario: Scenario, monkeypatch) -> dict[str, str | None]:
    series, catalog, obs, start, end = scenario.build()
    import modeler.etl.marts.common_build as mod

    monkeypatch.setattr(mod, "default_common_feature_series", lambda: series)
    monkeypatch.setattr(mod, "default_common_feature_catalog", lambda: catalog)

    feature_dates = _krx_days(start, end)
    first_avail = min(
        (o.available_from_date for o in obs if o.available_from_date is not None),
        default=start,
    )
    trading_days = _krx_days(min(first_avail, start), end)

    con = duckdb.connect()
    _load_observations(con, obs)
    mod.register_common_feature_daily_fact_view(
        con, trading_days=trading_days, feature_dates=feature_dates
    )
    rows = con.execute(
        "SELECT feature_date, feature_code, value_numeric FROM common_feature_daily_fact"
    ).fetchall()
    return {f"{r[0]}|{r[1]}": (None if r[2] is None else str(r[2])) for r in rows}


@pytest.fixture(scope="module")
def golden():
    return json.loads(_GOLDEN_PATH.read_text())


def _assert_parity(expected: dict, mart: dict, *, is_vol: bool) -> None:
    assert set(expected) == set(mart), (
        f"key mismatch: golden-only={set(expected) - set(mart)} "
        f"mart-only={set(mart) - set(expected)}"
    )
    mismatches = []
    for key, exp in expected.items():
        got = mart[key]
        if exp is None or got is None:
            if exp != got:
                mismatches.append(f"{key}: golden={exp!r} mart={got!r}")
            continue
        exp_d, got_d = Decimal(exp), Decimal(got)
        if is_vol:
            denom = abs(exp_d) if exp_d != 0 else Decimal("1")
            if abs(exp_d - got_d) / denom > _VOL_REL_TOL:
                mismatches.append(f"{key}: golden={exp_d} mart={got_d} (rel-tol)")
        elif exp_d != got_d:
            mismatches.append(f"{key}: golden={exp_d} mart={got_d}")
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_mart_matches_golden(scenario, golden, monkeypatch):
    expected = golden[scenario.name]
    mart = _run_mart(scenario, monkeypatch)
    _assert_parity(expected, mart, is_vol=scenario.is_vol)


def test_adding_a_catalog_feature_only_adds_rows(monkeypatch):
    """PR-1a-1's contract: a new catalog entry adds fact rows and moves none.

    ``build_common_feature_facts_sql`` emits one independent SELECT per feature
    and UNIONs them, so a feature cannot reach another feature's rows — this
    pins that structural property down rather than trusting it, and is why the
    two golden fixtures above did not have to move when
    ``commodity_wti_spot_level`` was added.
    """
    series, catalog, obs, start, end = _sc_level()
    added = _feature("market_kospi_ret_1d", transform_code="ret_1d")

    def _rows(active_catalog):
        import modeler.etl.marts.common_build as mod

        monkeypatch.setattr(mod, "default_common_feature_series", lambda: series)
        monkeypatch.setattr(mod, "default_common_feature_catalog", lambda: active_catalog)
        con = duckdb.connect()
        _load_observations(con, obs)
        mod.register_common_feature_daily_fact_view(
            con,
            trading_days=_krx_days(date(2026, 1, 5), end),
            feature_dates=_krx_days(start, end),
        )
        return {
            (r[0], r[1]): (r[2], r[3], r[4])
            for r in con.execute(
                "SELECT feature_date, feature_code, value_numeric, unit, "
                "asof_available_date FROM common_feature_daily_fact"
            ).fetchall()
        }

    before = _rows(catalog)
    after = _rows([*catalog, added])

    assert before, "the baseline scenario must produce rows for this to mean anything"
    assert {k: v for k, v in after.items() if k in before} == before
    assert {code for _date, code in after} - {code for _date, code in before} == {
        "market_kospi_ret_1d"
    }
