"""KR 입력 경로: KRX 지수 종가, KR 현금(CD91), KR 거시(USD/KRW·외국인 순매수·거래대금).

로컬 KR 레이크는 Postgres를 내보낸 parquet다::

    $STOCK_DATA_ROOT/kr/raw/raw_postgres/snapshot_date=<d>/source=sj2_remote/<table>/**/*.parquet

읽는 표는 둘이다(``KR_TABLES_USED``): ``krx_index_daily``(KRX Open API 지수 일별)와
``common_feature_observation_raw``(ECOS 등 공통 피쳐 원자료). 스냅샷은
``etl.snapshot.resolve_snapshot``이 고른다 — ``_manifests/_SUCCESS.json``에 두 표가 다 있는 가장
새 스냅샷이다. 없으면 ``KrNotSyncedError``로 멈춘다(0이나 빈 값으로 채우지 않는다).

가격 기준
    KR 지수는 **가격지수**다. 배당을 더하지 않고 ``return_basis="price_only"``로 표시한다
    (``compute_total_return_path(prices, None)``). 배당 가산 근거가 없다(03 §4.1).

키
    자산 -> ``(index_group, idx_nm)`` (``assets.kr_index_key``). ``idx_nm``은 그룹을 넘어 유일하지
    않다(``건설`` 등 20개가 kospi·kosdaq에 모두 있다) — 항상 그룹과 같이 찾는다.

가용 시각 (PIT)
    KRX 지수 행의 ``available_at``은 US와 같은 규칙이다: **그 세션의 폐장 시각 + 60분**
    (``available_at_basis="session_close_plus_60min"``). 폐장 시각은 XKRX 달력의 실제 값이다
    (2016-08-01 전 15:00 KST, 이후 15:30 KST). 상수로 쓰지 않는다.
    근거: 지수 종가는 폐장 시점에 공개된 정보다. KRX Open API가 T+1로 공표하는 것(``bas_dd`` 당일
    행이 23:00 KST에도 없다, 2026-09-29 실측)은 운영 문제(MS5: 매일 결정 시각에 그날 종가를 받는
    경로)이지 시점 정합성(PIT) 문제가 아니다. 옛 T+1 규칙(다음 달력일 08:30 KST,
    ``krx_openapi_t_plus_1_0830_kst``)은 ``krx_index_available_at_collection_time``에 남겨 두었고
    기본으로는 쓰지 않는다.

    ECOS 계열(USD/KRW·외국인 순매수·거래대금·CD91)은 행의 ``available_from_date``(없으면 관측일
    다음 XKRX 세션)의 **08:30 KST**를 가용 시각으로 쓴다. ``available_from_date``가 관측일 이하이면
    관측일 다음 날로 올린다(당일 종가를 당일에 알 수는 없다). 날짜순으로 ``available_at``이
    단조가 아니면(늦게 백필된 옛 관측) 누적 최댓값으로 올린다 — 늦추는 방향이라 PIT에 안전하다.

달력
    XKRX. ``exchange_calendars``가 있으면 그것(세션 표는 지수 첫 세션 ~ 마지막 세션 + 20일),
    없으면 관측된 지수 날짜(``calendar_basis``에 표시). 달력에 없는 지수 행은 경로에서 빼고
    진단에 센다(이웃 날짜로 옮기지 않는다).
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from modeler.etl.config import REMOTE_SOURCE, DataRoot, LakeConfig
from modeler.etl.snapshot import resolve_snapshot
from modeler.scores.common.calendar import UTC_TS, SessionCalendar
from modeler.scores.common.panel import AVAILABLE_AT_BASIS, PRICE_AVAILABILITY_BUFFER
from modeler.scores.common.total_return import compute_total_return_path, dividend_coverage

logger = logging.getLogger(__name__)

KRX_INDEX_TABLE = "krx_index_daily"
COMMON_OBS_TABLE = "common_feature_observation_raw"
KR_TABLES_USED = (KRX_INDEX_TABLE, COMMON_OBS_TABLE)

#: KRX 지수 행의 가용 시각 기준 이름. 패널 ``available_at_basis`` 열에 들어간다(US와 같다).
KRX_AVAILABLE_AT_BASIS = AVAILABLE_AT_BASIS
#: 옛 규칙(수집 시각 기준 T+1 08:30 KST)의 기준 이름. 기본으로 쓰지 않는다.
KRX_COLLECTION_TIME_BASIS = "krx_openapi_t_plus_1_0830_kst"
#: ECOS 계열 가용 시각 기준 이름(피쳐 manifest용).
ECOS_AVAILABLE_AT_BASIS = "available_from_date_0830_kst"
AVAILABILITY_TIME_KST = time(8, 30)
_KST = ZoneInfo("Asia/Seoul")

SERIES_CD91 = "rate_kr_cd91"
SERIES_FX = "fx_usdkrw_ecos"
SERIES_FOREIGN_NET = "foreign_net_kospi_ecos"
SERIES_TRDVAL = "trdval_kospi_ecos"
#: ``market_kospi_ecos``·``market_kosdaq_ecos``는 KRX 지수의 fallback 원천이다(아직 안 쓴다).
KR_MACRO_SERIES = (SERIES_FX, SERIES_FOREIGN_NET, SERIES_TRDVAL)


class KrNotSyncedError(RuntimeError):
    """KR 지수·CD91·거시 데이터가 아직 맥에 sync되지 않았다(또는 표가 빠졌다)."""


# --------------------------------------------------------------------------- 시각
def kr_available_at(d: date) -> datetime:
    """``d``일 08:30 KST (tz-aware UTC)."""
    return datetime.combine(d, AVAILABILITY_TIME_KST, tzinfo=_KST).astimezone(UTC)


def krx_index_available_at(session: date, cal: SessionCalendar) -> datetime:
    """``bas_dd``가 ``session``인 KRX 지수 행의 가용 시각: 그 세션 폐장 + 60분 (tz-aware UTC).

    US와 같은 규칙이다(``panel.AVAILABLE_AT_BASIS``). 폐장 시각은 ``cal``(XKRX)의 실제 값이라
    2016-08-01 전(15:00 KST)과 후(15:30 KST)가 다르다. 지수 종가는 폐장 때 공개된 정보다.
    KRX Open API의 T+1 공표는 운영 문제(MS5)이지 PIT 문제가 아니다.
    """
    return cal.close_at(session) + PRICE_AVAILABILITY_BUFFER


def krx_index_available_at_collection_time(session: date) -> datetime:
    """옛 규칙: 다음 달력일 08:30 KST (KRX Open API T+1 공표 기준). **기본으로 쓰지 않는다.**

    ``available_at_basis``는 ``KRX_COLLECTION_TIME_BASIS``를 쓴다.
    """
    return kr_available_at(session + timedelta(days=1))


def _next_session_date(d: date, sessions: Sequence[date]) -> date:
    """``d`` 다음(엄격히 뒤) 첫 세션. 달력 끝을 넘으면 ``d + 1일``."""
    k = bisect_right(sessions, d)
    return sessions[k] if k < len(sessions) else d + timedelta(days=1)


# --------------------------------------------------------------------------- 레이크
@dataclass(frozen=True)
class KrLake:
    """스냅샷을 못 박은 KR raw 레이크 뷰."""

    root: DataRoot
    snapshot_date: str
    source: str = REMOTE_SOURCE

    @classmethod
    def resolve(
        cls,
        root: DataRoot,
        *,
        snapshot_date: str | None = None,
        source: str = REMOTE_SOURCE,
        tables: Sequence[str] = KR_TABLES_USED,
    ) -> KrLake:
        """``tables``가 다 있는 완전한 스냅샷. 없으면 ``KrNotSyncedError``.

        ``snapshot_date``를 주면 그 스냅샷이 완전할 때만 쓴다(아니면 예외).
        """
        try:
            res = resolve_snapshot(
                root, source, required_inputs=tuple(tables), snapshot_date=snapshot_date
            )
        except FileNotFoundError as exc:
            raise KrNotSyncedError(
                f"KR raw 스냅샷이 없습니다 (필요한 표 {list(tables)}, source={source}). "
                "collector 백필 뒤 `collector db sync-remote`로 받으십시오. "
                f"원인: {exc}"
            ) from exc
        lake = cls(root=root, snapshot_date=res.snapshot_date, source=source)
        for t in tables:
            if not lake.table_files(t):
                raise KrNotSyncedError(
                    f"{t}: 스냅샷 {res.snapshot_date}에 parquet 파일이 없습니다 "
                    f"({lake.table_dir(t)}). sync를 다시 하십시오."
                )
        return lake

    @property
    def config(self) -> LakeConfig:
        return LakeConfig(root=self.root, snapshot_date=self.snapshot_date, source=self.source)

    def table_dir(self, table: str) -> Path:
        return self.config.raw_root / table

    def table_files(self, table: str) -> list[Path]:
        return sorted(self.table_dir(table).rglob("*.parquet"))

    def scan(self, table: str) -> pl.LazyFrame:
        files = self.table_files(table)
        if not files:
            raise KrNotSyncedError(f"{table}: parquet 파일이 없습니다 ({self.table_dir(table)})")
        return pl.scan_parquet([str(f) for f in files], hive_partitioning=False)

    def input_files(self, tables: Sequence[str]) -> dict[str, list[Path]]:
        return {t: self.table_files(t) for t in tables}


# --------------------------------------------------------------------------- 달력
def kr_session_calendar(index_sessions: Sequence[date]) -> SessionCalendar:
    """XKRX 달력. ``exchange_calendars``가 없거나 실패하면 관측된 지수 날짜."""
    ds = sorted(set(index_sessions))
    if not ds:
        raise ValueError("지수 세션이 비었습니다")
    try:
        cal = SessionCalendar.from_exchange_calendars("XKRX", ds[0], ds[-1] + timedelta(days=20))
    except Exception as exc:  # exchange_calendars가 범위를 거부하는 경우 등
        logger.warning(
            "exchange_calendars XKRX 실패 (%s) — 관측된 지수 날짜를 세션으로 씁니다", exc
        )
        cal = None
    if cal is None:
        return SessionCalendar.from_observed_dates("XKRX", ds)
    return cal


# --------------------------------------------------------------------------- 지수 경로
def load_kr_index_paths(
    lake: KrLake,
    keys_by_asset: Mapping[str, tuple[str, str]],
    *,
    sessions: frozenset[date] | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, Any]]]:
    """자산별 가격지수 경로와 진단.

    ``keys_by_asset``: ``asset_id -> (index_group, idx_nm)``. 반환 경로는
    ``session, px_raw, px_adj, div_adj, tr_index, return_basis`` (``return_basis="price_only"``,
    ``px_raw == px_adj == close_idx``). 종가가 null이거나 0 이하인 행, 달력에 없는 행은 뺀다
    (수는 진단). 지수 행이 하나도 없으면 ``ValueError``.
    """
    raw = (
        lake.scan(KRX_INDEX_TABLE)
        .select(
            pl.col("index_group").cast(pl.String),
            pl.col("idx_nm").cast(pl.String),
            pl.col("bas_dd").cast(pl.Date).alias("session"),
            pl.col("close_idx").cast(pl.Float64).alias("close"),
        )
        .filter(
            pl.any_horizontal(
                [
                    (pl.col("index_group") == g) & (pl.col("idx_nm") == n)
                    for g, n in keys_by_asset.values()
                ]
            )
        )
        .collect()
    )
    paths: dict[str, pl.DataFrame] = {}
    diags: dict[str, dict[str, Any]] = {}
    for asset_id, (group, name) in keys_by_asset.items():
        sub = raw.filter((pl.col("index_group") == group) & (pl.col("idx_nm") == name))
        if sub.height == 0:
            raise ValueError(f"{asset_id}: {KRX_INDEX_TABLE}에 ({group}, {name}) 행이 없습니다")
        n_rows = sub.height
        dup = n_rows - sub["session"].n_unique()
        bad = sub.filter(pl.col("close").is_null() | (pl.col("close") <= 0))
        p = sub.filter(pl.col("close").is_not_null() & (pl.col("close") > 0)).select(
            "session", pl.col("close").alias("px_raw"), pl.col("close").alias("px_adj")
        )
        off_dates: list[date] = []
        if sessions is not None:
            off = p.filter(~pl.col("session").is_in(list(sessions)))
            off_dates = sorted(off["session"].unique().to_list())
            p = p.filter(pl.col("session").is_in(list(sessions)))
        if p.height == 0:
            raise ValueError(f"{asset_id}: ({group}, {name}) 사용 가능한 종가가 없습니다")
        path, diag = compute_total_return_path(p, None)
        last = path["session"].max()
        diag.update(
            {
                "index_group": group,
                "idx_nm": name,
                "price_rows": n_rows,
                "duplicate_price_dates": dup,
                "null_or_nonpositive_close_rows": bad.height,
                "prices_off_calendar": len(off_dates),
                "prices_off_calendar_dates": [d.isoformat() for d in off_dates],
                **dividend_coverage(None, last_price_session=last),
            }
        )
        paths[asset_id] = path
        diags[asset_id] = diag
        logger.info(
            "%s(%s/%s): 지수 %d행 [%s ~ %s] · price_only",
            asset_id,
            group,
            name,
            n_rows,
            path["session"].min(),
            last,
        )
    return paths, diags


# --------------------------------------------------------------------------- ECOS 계열
def _load_series(lake: KrLake, series_id: str, sessions: Sequence[date]) -> pl.DataFrame | None:
    """``date, value, avail_date`` (관측일당 한 행). 시리즈가 없으면 ``None``.

    ``avail_date``: ``available_from_date``가 있으면 그것(관측일 이하면 관측일 + 1일), 없거나
    null이면 관측일 다음 XKRX 세션.
    """
    lf = lake.scan(COMMON_OBS_TABLE)
    names = set(lf.collect_schema().names())
    cols = [
        pl.col("observation_date").cast(pl.Date).alias("date"),
        pl.col("value_numeric").cast(pl.Float64).alias("value"),
        (
            pl.col("available_from_date").cast(pl.Date)
            if "available_from_date" in names
            else pl.lit(None, dtype=pl.Date)
        ).alias("avail_date"),
        (
            pl.col("fetched_at")
            if "fetched_at" in names
            else pl.lit(None, dtype=pl.Datetime("us", "UTC"))
        ).alias("_fetched"),
    ]
    df = (
        lf.filter(pl.col("series_id") == series_id)
        .select(cols)
        .collect()
        .drop_nulls(["date", "value"])
    )
    if df.height == 0:
        return None
    # 같은 관측일이 여러 행이면 가장 늦게 받은 행(fetched_at 없으면 마지막 행)
    df = df.sort(["date", "_fetched"], nulls_last=False).unique(
        subset=["date"], keep="last", maintain_order=True
    )
    sess = sorted(sessions)
    fallback = [_next_session_date(d, sess) for d in df["date"].to_list()]
    df = df.with_columns(
        pl.coalesce(pl.col("avail_date"), pl.Series(fallback, dtype=pl.Date)).alias("avail_date")
    )
    too_early = df.filter(pl.col("avail_date") <= pl.col("date")).height
    if too_early:
        logger.warning(
            "%s: available_from_date <= 관측일인 행 %d개 — 관측일 다음 날로 올립니다",
            series_id,
            too_early,
        )
    df = df.with_columns(
        pl.when(pl.col("avail_date") <= pl.col("date"))
        .then(pl.col("date") + pl.duration(days=1))
        .otherwise(pl.col("avail_date"))
        .alias("avail_date")
    )
    return df.select("date", "value", "avail_date").sort("date")


def _with_available_at(df: pl.DataFrame) -> pl.DataFrame:
    """``avail_date`` -> ``available_at`` (08:30 KST, 날짜순 단조로 누적 최댓값)."""
    at = pl.Series([kr_available_at(d) for d in df["avail_date"].to_list()], dtype=UTC_TS)
    out = df.with_columns(at.alias("available_at")).with_columns(
        pl.col("available_at").cum_max().alias("available_at")
    )
    return out.select("date", "value", "available_at")


@dataclass(frozen=True)
class KrMacro:
    """KR 피쳐 함수 입력: 각 ``date, value, available_at`` long 프레임(tz-aware)."""

    fx: pl.DataFrame
    foreign_net: pl.DataFrame
    trdval: pl.DataFrame


def load_kr_macro(lake: KrLake, sessions: Sequence[date]) -> KrMacro:
    """KR 피쳐 둘의 입력(``usdkrw_ret_60``, ``kr_foreign_net_20_over_trdval``).

    시리즈가 없으면 ``KrNotSyncedError``.
    """
    frames = {}
    for sid in KR_MACRO_SERIES:
        raw = _load_series(lake, sid, sessions)
        if raw is None:
            raise KrNotSyncedError(
                f"{COMMON_OBS_TABLE}에 {sid} 행이 없습니다 — 백필·sync를 확인하십시오"
            )
        frames[sid] = _with_available_at(raw)
    return KrMacro(
        fx=frames[SERIES_FX],
        foreign_net=frames[SERIES_FOREIGN_NET],
        trdval=frames[SERIES_TRDVAL],
    )


def load_kr_rates(
    lake: KrLake, series_id: str = SERIES_CD91, sessions: Sequence[date] = ()
) -> pl.DataFrame | None:
    """``cash.build_cash_account``의 입력 ``date, realtime_start, value``(연 %). 없으면 ``None``.

    ``realtime_start`` = ``available_from_date``(없으면 관측일 다음 XKRX 세션). ``sessions``는
    그 규칙에 쓴다(``available_from_date``가 다 있으면 안 써도 된다).
    """
    raw = _load_series(lake, series_id, sessions)
    if raw is None:
        return None
    return raw.select("date", pl.col("avail_date").alias("realtime_start"), "value")
