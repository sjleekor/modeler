"""Shared common-feature build test fixtures (service-independent).

Extracted from test_build_common_feature_daily_facts.py so the DuckDB mart parity
tests can reuse MockCommonFeatureBuildStorage + synthetic observations without
importing the Postgres build service (removed at refactor P5).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from collector.kr.shared import (
    CommonFeatureCatalogEntry,
    CommonFeatureDailyFact,
    CommonFeatureObservation,
    CommonFeatureSeries,
    IngestionRun,
    Source,
    UpsertResult,
)


def _krx_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _series(
    series_id: str = "market_kospi",
    *,
    source: Source = Source.PYKRX,
    max_stale_business_days: int = 5,
    active: bool = True,
) -> CommonFeatureSeries:
    return CommonFeatureSeries(
        series_id=series_id,
        source=source,
        source_series_key=series_id,
        category="market",
        frequency="D",
        name_kr=series_id,
        unit="point",
        availability_policy="next_krx_session",
        max_stale_business_days=max_stale_business_days,
        active=active,
    )


def _feature(
    feature_code: str = "market_kospi_close",
    *,
    transform_code: str = "level",
    series_id: str = "market_kospi",
    active: bool = True,
) -> CommonFeatureCatalogEntry:
    return CommonFeatureCatalogEntry(
        feature_code=feature_code,
        feature_name_kr=feature_code,
        category="market",
        unit="point",
        transform_code=transform_code,
        input_series_ids=(series_id,),
        active=active,
    )


def _obs(
    raw_id: int,
    series_id: str,
    observation_date: date,
    available_from_date: date,
    value: str,
    *,
    source: Source = Source.PYKRX,
    period_end_date: date | None = None,
    release_date: date | None = None,
    vintage: str = "",
) -> CommonFeatureObservation:
    return CommonFeatureObservation(
        raw_id=raw_id,
        source=source,
        series_id=series_id,
        observation_date=observation_date,
        period_end_date=period_end_date or observation_date,
        release_date=release_date,
        available_from_date=available_from_date,
        vintage=vintage,
        value_numeric=Decimal(value),
        unit="point",
        frequency="D",
        fetched_at=datetime(2026, 6, 9, 8, 0, tzinfo=UTC),
    )


class MockCommonFeatureBuildStorage:
    def __init__(
        self,
        *,
        series: list[CommonFeatureSeries],
        catalog: list[CommonFeatureCatalogEntry],
        observations: list[CommonFeatureObservation],
    ) -> None:
        self.series = series
        self.catalog = catalog
        self.observations = observations
        self.facts: list[CommonFeatureDailyFact] = []
        self.runs: list[IngestionRun] = []
        self.observation_query: (
            tuple[
                list[str] | None,
                date | None,
                date | None,
                Source | None,
                date | None,
            ]
            | None
        ) = None
        self.catalog_query: tuple[list[str] | None, bool] | None = None
        self.series_query: tuple[list[Source] | None, list[str] | None, bool] | None = None

    def record_run(self, run: IngestionRun) -> None:
        self.runs.append(run)

    def get_common_feature_catalog(
        self,
        feature_codes: list[str] | None = None,
        active_only: bool = True,
    ) -> list[CommonFeatureCatalogEntry]:
        self.catalog_query = (feature_codes, active_only)
        return [
            feature
            for feature in self.catalog
            if (not feature_codes or feature.feature_code in feature_codes)
            and (not active_only or feature.active)
        ]

    def get_common_feature_series(
        self,
        sources: list[Source] | None = None,
        series_ids: list[str] | None = None,
        active_only: bool = True,
    ) -> list[CommonFeatureSeries]:
        self.series_query = (sources, series_ids, active_only)
        return [
            series
            for series in self.series
            if (not sources or series.source in sources)
            and (not series_ids or series.series_id in series_ids)
            and (not active_only or series.active)
        ]

    def get_common_feature_observations(
        self,
        series_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        source: Source | None = None,
        available_from_end: date | None = None,
    ) -> list[CommonFeatureObservation]:
        self.observation_query = (series_ids, start, end, source, available_from_end)
        return [
            observation
            for observation in self.observations
            if (not series_ids or observation.series_id in series_ids)
            and (start is None or observation.observation_date >= start)
            and (end is None or observation.observation_date <= end)
            and (source is None or observation.source == source)
            and (
                available_from_end is None
                or observation.available_from_date is None
                or observation.available_from_date <= available_from_end
            )
        ]

    def upsert_common_feature_daily_facts(
        self,
        records: list[CommonFeatureDailyFact],
    ) -> UpsertResult:
        self.facts = records
        return UpsertResult(updated=len(records))


def _multi_feature(
    feature_code: str,
    *,
    transform_code: str,
    inputs: tuple[tuple[str, str], ...],
) -> CommonFeatureCatalogEntry:
    return CommonFeatureCatalogEntry(
        feature_code=feature_code,
        feature_name_kr=feature_code,
        category="rate",
        unit="pctp",
        transform_code=transform_code,
        input_series_ids=tuple(series_id for series_id, _ in inputs),
        input_roles=tuple(role for _, role in inputs),
    )
