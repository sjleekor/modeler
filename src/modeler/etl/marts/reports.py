"""Coverage / readiness / freshness checks on the lake (refactor §4).

These move the Postgres operational gates onto the parquet/DuckDB compute node:

- :func:`coverage_report` / :func:`readiness_report` — ports of
  ``service/report_common_feature_coverage.py`` and ``report_common_feature_readiness.py``,
  computed over the ``common_feature_daily_fact`` mart and the KRX feature-date
  calendar instead of the dropped canonical table.
- :func:`freshness_violations` — raw-observation freshness gate, run on the lake
  (``common_feature_observation_raw`` + ``common_feature_series``) so the compute
  pipeline can refuse to run on stale raw (decision 6). It does not check source
  run age because remote ``ingestion_runs`` is not part of the raw mirror.

All functions take a DuckDB connection with the needed views already registered
and return plain dataclasses the orchestrator turns into a non-zero exit + stderr
summary (no Postgres writes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import duckdb
from collector.kr.shared import default_common_feature_catalog


@dataclass(frozen=True, slots=True)
class CoverageRow:
    feature_code: str
    target_count: int
    fact_count: int
    non_null_count: int
    null_count: int
    missing_count: int
    coverage_ratio: float
    pit_violation_count: int


@dataclass(frozen=True, slots=True)
class ReadinessRow:
    feature_code: str
    coverage_ratio: float
    ready: bool
    blockers: tuple[str, ...]
    #: First feature date this feature was judged over. ``None`` when the whole
    #: calendar was used, which is also what a feature with no facts gets.
    window_start: date | None = None
    #: Grid dates before ``window_start`` that carry a fact with a NULL value —
    #: the derivation warm-up excluded from the judgement. Reported rather than
    #: hidden: a YoY feature needing twelve months is expected to have one, and
    #: a large number on a feature that should have none is a real finding.
    warmup_null_count: int = 0


@dataclass(frozen=True, slots=True)
class FreshnessViolation:
    series_id: str | None
    check: str
    message: str


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    end: date
    checked_series: int
    violations: list[FreshnessViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def coverage_report(
    con: duckdb.DuckDBPyConnection,
    *,
    feature_dates: Sequence[date],
    cfdf_view: str = "common_feature_daily_fact",
    feature_codes: Sequence[str] | None = None,
) -> list[CoverageRow]:
    """Per-feature coverage over the KRX feature-date calendar.

    ``target_count`` = number of feature dates; ``coverage_ratio`` =
    non_null / target. Mirrors the Postgres coverage report 1:1 by evaluating
    every active catalog feature, including features with zero generated facts.
    """
    target_count = len(feature_dates)
    codes = _feature_codes(feature_codes)
    if not codes:
        return []

    rows = con.execute(f"""
        WITH feature_codes(feature_code) AS ({_values_list(codes)}),
        feature_dates(feature_date) AS ({_date_values_list(feature_dates)})
        SELECT
            c.feature_code,
            count(f.feature_code) AS fact_count,
            count(f.value_numeric) AS non_null_count,
            count(*) FILTER (
                WHERE f.feature_code IS NOT NULL AND f.value_numeric IS NULL
            ) AS null_count,
            count(*) FILTER (
                WHERE f.feature_code IS NOT NULL AND f.asof_available_date > fd.feature_date
            ) AS pit_violation_count
        FROM feature_codes c
        LEFT JOIN feature_dates fd ON TRUE
        LEFT JOIN {cfdf_view} f
          ON f.feature_code = c.feature_code
         AND f.feature_date = fd.feature_date
        GROUP BY c.feature_code
        """).fetchall()
    out: list[CoverageRow] = []
    for feature_code, fact_count, non_null, null_count, pit_violations in rows:
        missing = max(target_count - fact_count, 0)
        ratio = round(non_null / target_count, 4) if target_count > 0 else 0.0
        out.append(
            CoverageRow(
                feature_code=feature_code,
                target_count=target_count,
                fact_count=fact_count,
                non_null_count=non_null,
                null_count=null_count,
                missing_count=missing,
                coverage_ratio=ratio,
                pit_violation_count=pit_violations,
            )
        )
    out.sort(key=lambda r: (r.coverage_ratio, r.feature_code), reverse=True)
    return out


def _first_feature_dates(
    con: duckdb.DuckDBPyConnection,
    *,
    cfdf_view: str,
    codes: Sequence[str],
) -> dict[str, date]:
    """Earliest date each feature has a non-NULL value for (F-9.9).

    Not the first *fact*: a derived feature emits a row from its input's first
    date but cannot carry a value until its window fills, so a YoY series has a
    structurally NULL run at the front (measured: ``macro_cpi_yoy_latest``, 253
    dates). Judging from the first fact charged every such feature with those
    nulls forever, which no amount of collection can fix.

    A leading NULL run is a warm-up, not a hole. Holes *inside* the span still
    count, and the excluded run is reported as ``warmup_null_count`` so a
    surprisingly long one is visible rather than silently forgiven.
    """
    rows = con.execute(f"""
        WITH feature_codes(feature_code) AS ({_values_list(list(codes))})
        SELECT c.feature_code, min(f.feature_date) FILTER (WHERE f.value_numeric IS NOT NULL)
        FROM feature_codes c
        LEFT JOIN {cfdf_view} f ON f.feature_code = c.feature_code
        GROUP BY c.feature_code
        """).fetchall()
    return {code: first for code, first in rows if first is not None}


def _window_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    cfdf_view: str,
    feature_dates: Sequence[date],
    window_starts: dict[str, date],
) -> dict[str, tuple[int, int, int, int]]:
    """Per-feature ``(fact, non_null, null, pit_violation)`` counts inside the window.

    ``coverage_report``'s counts span the whole calendar, so they cannot be
    reused once the window moves off the first fact: the leading NULL rows are
    real facts and would keep ``null_count`` positive. Recomputed here rather
    than by changing ``coverage_report``, whose output is frozen in
    ``golden/common_feature_reports.json`` and is a separate report.
    """
    if not window_starts:
        return {}
    starts = ", ".join(
        f"('{code}', DATE '{start.isoformat()}')" for code, start in sorted(window_starts.items())
    )
    rows = con.execute(f"""
        WITH windows(feature_code, window_start) AS (VALUES {starts}),
        feature_dates(feature_date) AS ({_date_values_list(feature_dates)})
        SELECT
            w.feature_code,
            count(f.feature_code) AS fact_count,
            count(f.value_numeric) AS non_null_count,
            count(*) FILTER (
                WHERE f.feature_code IS NOT NULL AND f.value_numeric IS NULL
            ) AS null_count,
            count(*) FILTER (
                WHERE f.feature_code IS NOT NULL AND f.asof_available_date > fd.feature_date
            ) AS pit_violation_count
        FROM windows w
        JOIN feature_dates fd ON fd.feature_date >= w.window_start
        LEFT JOIN {cfdf_view} f
          ON f.feature_code = w.feature_code
         AND f.feature_date = fd.feature_date
        GROUP BY w.feature_code
        """).fetchall()
    return {code: (facts, non_null, nulls, pit) for code, facts, non_null, nulls, pit in rows}


def readiness_report(
    con: duckdb.DuckDBPyConnection,
    *,
    feature_dates: Sequence[date],
    required_coverage_ratio: float = 1.0,
    cfdf_view: str = "common_feature_daily_fact",
    feature_codes: Sequence[str] | None = None,
    per_feature_window: bool = True,
) -> list[ReadinessRow]:
    """Strict readiness: full coverage, no nulls, no missing dates, no PIT
    violations — each feature judged over its own history.

    ``per_feature_window`` is what makes this gate usable, and it is on by
    default because the alternative measured 4 of 37 ready and stayed there.

    The calendar spans the earliest availability of ANY series through the
    latest, so four monthly ECOS series that start a year before everything
    else (2013-06-20 against 2014-06-16) set the window for all 37. The other
    33 were then charged ``missing_count=257`` for not existing yet — which is
    not a defect, it is a shorter history. Only those same four passed, and a
    gate that fails 33 of 37 every run is a gate nobody reads.

    Judging each feature from its own first fact keeps every real defect: a hole
    inside a feature's own span still counts, which is how 2017-10-10 (the
    Chuseok closure that outran ``max_stale_business_days``) stays visible.

    Set it ``False`` to reproduce the original single-window behaviour.
    """
    codes = _feature_codes(feature_codes)
    first_dates = (
        _first_feature_dates(con, cfdf_view=cfdf_view, codes=codes) if per_feature_window else {}
    )
    ordered_dates = sorted(feature_dates)
    window_counts = _window_counts(
        con,
        cfdf_view=cfdf_view,
        feature_dates=ordered_dates,
        window_starts=first_dates,
    )

    out: list[ReadinessRow] = []
    for row in coverage_report(
        con,
        feature_dates=feature_dates,
        cfdf_view=cfdf_view,
        feature_codes=feature_codes,
    ):
        target_count = row.target_count
        missing_count = row.missing_count
        coverage_ratio = row.coverage_ratio
        null_count = row.null_count
        pit_violation_count = row.pit_violation_count
        window_start: date | None = None
        warmup_null_count = 0

        first = first_dates.get(row.feature_code)
        if first is not None:
            window_start = first
            target_count = sum(1 for day in ordered_dates if day >= first)
            facts, non_null, nulls, pit = window_counts.get(row.feature_code, (0, 0, 0, 0))
            missing_count = max(target_count - facts, 0)
            coverage_ratio = round(non_null / target_count, 4) if target_count > 0 else 0.0
            null_count = nulls
            pit_violation_count = pit
            # Everything the whole-calendar pass saw minus what the window sees:
            # the derivation warm-up, reported rather than charged.
            warmup_null_count = max(row.null_count - nulls, 0)

        blockers: list[str] = []
        if target_count == 0:
            blockers.append("target_count=0")
        if coverage_ratio < required_coverage_ratio:
            blockers.append(f"coverage_ratio={coverage_ratio} < required={required_coverage_ratio}")
        if null_count > 0:
            blockers.append(f"null_count={null_count}")
        if missing_count > 0:
            blockers.append(f"missing_count={missing_count}")
        if pit_violation_count > 0:
            blockers.append(f"pit_violation_count={pit_violation_count}")
        out.append(
            ReadinessRow(
                feature_code=row.feature_code,
                coverage_ratio=coverage_ratio,
                ready=not blockers,
                blockers=tuple(blockers),
                window_start=window_start,
                warmup_null_count=warmup_null_count,
            )
        )
    out.sort(key=lambda r: (not r.ready, r.feature_code))
    return out


def _feature_codes(feature_codes: Sequence[str] | None) -> list[str]:
    if feature_codes is not None:
        return sorted(set(feature_codes))
    return sorted({item.feature_code for item in default_common_feature_catalog() if item.active})


def _sql_str_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _values_list(values: Sequence[str]) -> str:
    return "VALUES " + ", ".join(f"({_sql_str_literal(value)})" for value in values)


def _date_values_list(values: Sequence[date]) -> str:
    if not values:
        return "SELECT CAST(NULL AS DATE) WHERE FALSE"
    return "VALUES " + ", ".join(f"(DATE '{value.isoformat()}')" for value in values)


def freshness_violations(
    con: duckdb.DuckDBPyConnection,
    *,
    end: date,
    daily_max_lag_days: int = 2,
    macro_max_lag_days: int = 45,
    obs_view: str = "common_feature_observation_raw",
    series_view: str = "common_feature_series",
) -> FreshnessResult:
    """Raw-input freshness gate (port of ``assert_common_freshness``).

    For each active series, the latest observation must be within the per-series
    allowed lag (max of the frequency default, manual_lag_days, and
    max_stale_business_days). Series with no observations violate. Run BEFORE the
    marts so compute refuses stale raw (decision 6).
    """
    rows = con.execute(f"""
        WITH latest AS (
            SELECT series_id, max(observation_date) AS latest_obs
            FROM {obs_view}
            GROUP BY series_id
        )
        SELECT
            s.series_id,
            s.frequency,
            s.manual_lag_days,
            s.max_stale_business_days,
            l.latest_obs
        FROM {series_view} s
        LEFT JOIN latest l USING (series_id)
        WHERE s.active = TRUE
        """).fetchall()

    violations: list[FreshnessViolation] = []
    for series_id, frequency, manual_lag, max_stale, latest_obs in rows:
        base = daily_max_lag_days if (frequency or "").upper() == "D" else macro_max_lag_days
        max_lag = max(base, int(manual_lag or 0), int(max_stale or 0))
        if latest_obs is None:
            violations.append(
                FreshnessViolation(
                    series_id=series_id,
                    check="latest_observation",
                    message="no raw observations found for active common source series",
                )
            )
            continue
        lag_days = (end - latest_obs).days
        if lag_days > max_lag:
            violations.append(
                FreshnessViolation(
                    series_id=series_id,
                    check="latest_observation",
                    message=(
                        f"latest observation {latest_obs.isoformat()} is {lag_days} days "
                        f"behind {end.isoformat()} (allowed {max_lag})"
                    ),
                )
            )

    return FreshnessResult(end=end, checked_series=len(rows), violations=violations)
