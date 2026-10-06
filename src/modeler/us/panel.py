"""(리밸런스일, 종목) 월간 패널 조립.

패널 1행 = (리밸런스일, 종목). ``universe_daily.in_universe = 1``인 것만 담는다.
조정 가격은 ``prices.py``에서 계산한 것을 그대로 붙인다.

**유니버스 버전** (``build_panel(universe_version=...)``, 기본 ``v1``). ``v2``는 멤버십
(``in_universe``)만 ``universe_daily_v2``의 PIT 판정으로 바꾼다 — 유니버스 v2 설계 §4·전진 등록
§10의 "멤버십만 v2". 나머지 열(``cik``·``sic``·``mcap_rank``·``adv_20d``·``exchange``, 가격,
``price_ge_5``)은 v1과 같은 원천(``universe_daily``의 같은 ``(date, symbol)`` 행·``prices_daily``)과
같은 규칙으로 채운다. 그래서 두 버전의 패널 스키마가 같고(``security_id`` 열 없음) 동결 피쳐
코드가 그대로 읽는다. v2에만 있는 멤버(주로 ADR)도 ``universe_daily``에 같은 날 행이 있으면 그 값을
얻고, 없으면 빈 값이다(v1과 같은 규칙).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import polars as pl

from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily

UNIVERSE_VERSIONS = ("v1", "v2")

#: ``trading_calendar``에 실제로 있는 거래소는 하나뿐이다 — 2026-09-20 실측
#: (수집 계획 ``03_schema_and_pit.md`` §4.15: "실측 — XNYS · 2011-01-03~2026-12-31
#: · 4,023행"). 여러 거래소가 있었다면 어느 것을 쓸지 정해야 했겠지만, 지금은
#: 고를 필요가 없다. 나중에 표에 다른 거래소가 추가되면 이 상수를 다시 본다.
_EXCHANGE = "XNYS"


def month_first_trading_days(lake: UsLake, start: date, end: date) -> list[date]:
    """``[start, end]`` 구간에서 달마다 첫 거래일.

    ``trading_calendar``를 ``_EXCHANGE``(XNYS)로 필터링하고, 연-월로 묶어 그 안의
    최소 ``date``를 고른다 — 최소값을 쓰기 때문에 신정처럼 월초에 낀 휴장일이
    있어도 자동으로 그다음 거래일이 뽑힌다.
    """
    calendar = lake.scan("trading_calendar").filter(
        (pl.col("exchange") == _EXCHANGE) & (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    result = (
        calendar.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("_year_month"))
        .group_by("_year_month")
        .agg(pl.col("date").min().alias("date"))
        .sort("date")
        .collect()
    )
    return result["date"].to_list()


def _pit_only(lf: pl.LazyFrame) -> pl.LazyFrame:
    """``view`` 열이 있는 표는 ``pit`` 행만 남긴다. 사후 정정 보기(``post``)는 판정에 안 쓴다 (T11).

    ``universe_daily_v2``에는 지금 ``view`` 열이 없다 — collector가 PIT 보기로만 만드는 표다
    (``collector/us/universe/v2/build.py`` 모듈 docstring). 나중에 열이 생겨도 사후 보기가
    섞여 들어오지 않게 여기서 한 번 더 건다.
    """
    if "view" in lf.collect_schema().names():
        return lf.filter(pl.col("view") == "pit")
    return lf


def _members_v1(lake: UsLake, rebalance_dates: list[date]) -> pl.LazyFrame:
    return (
        lake.scan("universe_daily")
        .filter(pl.col("in_universe") & pl.col("date").is_in(rebalance_dates))
        .select(["date", "symbol", "cik", "sic", "mcap_rank", "adv_20d", "exchange"])
    )


def _members_v2(lake: UsLake, rebalance_dates: list[date]) -> pl.LazyFrame:
    """v2 멤버(PIT)의 ``(date, symbol)``에 v1 ``universe_daily``의 나머지 열을 붙인다.

    멤버 키는 ``(date, symbol)``이어야 한다 — ``universe_daily_v2``의 키는
    ``(date, security_id)``라 한 날 한 심볼에 구간 둘이 멤버면 패널 키가 겹친다. 겹치면 멈춘다.
    """
    members = (
        _pit_only(lake.scan("universe_daily_v2"))
        .filter(pl.col("in_universe") & pl.col("date").is_in(rebalance_dates))
        .select("date", "symbol")
        .collect()
    )
    dup = members.group_by(["date", "symbol"]).len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(
            f"universe_daily_v2 멤버의 (date, symbol)이 {dup.height}건 겹칩니다 — 패널 키가 "
            "유일하지 않습니다."
        )
    attrs = lake.scan("universe_daily").select(
        ["date", "symbol", "cik", "sic", "mcap_rank", "adv_20d", "exchange"]
    )
    return members.lazy().join(attrs, on=["date", "symbol"], how="left")


def build_panel(
    lake: UsLake,
    *,
    start: date = date(2018, 9, 7),
    end: date | None = None,
    universe_version: str = "v1",
) -> pl.DataFrame:
    """월간 패널을 만든다.

    ``end``를 안 주면 스냅샷이 주는 유니버스 표(v1 ``universe_daily``, v2
    ``universe_daily_v2``)의 최대 ``date``까지다.

    ``universe_version``: ``"v1"``(기본, 지금 동작) 또는 ``"v2"``(멤버십만 v2, 모듈 docstring).

    컬럼: ``date, symbol``(키) · ``cik, sic, sic2, mcap_rank, adv_20d,
    exchange``(``universe_daily``) · ``close``(``prices_daily`` 원시 종가) ·
    ``adj_close, adj_volume``(``prices.py``) · ``price_ge_5``(조정 전 종가 >= $5).

    ``sic2``는 ``sic`` 앞 두 자리다. ``sic``이 없으면 ``sic2``도 null로 둔다 —
    "미분류"로 바꾸는 것은 M1(라벨) 몫이다 (``02`` §2 유니버스 SIC 결측 처리 참고).

    조정 가격의 기준 시점 T는 ``adjusted_daily``의 기본값(``prices_daily`` 최대
    ``date``)을 그대로 쓴다 — ``end``와 다르게 둬도 된다. 분할 조정으로 얻는 것은
    "같은 T로 정규화된 계열의 두 시점 사이 비율(수익률)이 T 선택과 무관하다"는
    성질이라, 패널의 ``end``와 T를 굳이 맞출 필요가 없다.

    패널의 조정 가격은 ``lake.security_boundaries``와 상관없이 **심볼 단위 조정**이다
    (``adjusted_daily(security_boundaries=False)``) — 패널 열은 두 모드에서 같아야 한다.
    """
    if universe_version not in UNIVERSE_VERSIONS:
        raise ValueError(
            f"universe_version은 {UNIVERSE_VERSIONS} 중 하나여야 합니다: {universe_version!r}"
        )
    universe_table = "universe_daily" if universe_version == "v1" else "universe_daily_v2"

    if end is None:
        end_row = (
            _pit_only(lake.scan(universe_table)).select(pl.col("date").max().alias("d")).collect()
        )
        end = end_row.item()
        if end is None:
            raise ValueError(f"{universe_table}가 비어 있어 end를 정할 수 없습니다.")

    rebalance_dates = month_first_trading_days(lake, start, end)

    members = (
        _members_v1(lake, rebalance_dates)
        if universe_version == "v1"
        else _members_v2(lake, rebalance_dates)
    )
    universe = members.with_columns(
        pl.when(pl.col("sic").is_not_null())
        .then(pl.col("sic").str.slice(0, 2))
        .otherwise(None)
        .alias("sic2")
    )

    raw_close = lake.scan("prices_daily").select(
        "date", "symbol", pl.col("close").cast(pl.Float64).alias("close")
    )
    adjusted = adjusted_daily(lake, security_boundaries=False).select(
        "date", "symbol", "adj_close", "adj_volume"
    )

    panel = (
        universe.join(raw_close, on=["date", "symbol"], how="left")
        .join(adjusted, on=["date", "symbol"], how="left")
        # price_ge_5는 조정 전(원시) 종가를 본다 — 02 §4 지표 I의 거래가능
        # 유니버스 정의 그대로다. raw_close가 없는(join 실패) 행은 null로 남는다.
        .with_columns((pl.col("close") >= 5).alias("price_ge_5"))
        .select(
            "date",
            "symbol",
            "cik",
            "sic",
            "sic2",
            "mcap_rank",
            "adv_20d",
            "exchange",
            "close",
            "adj_close",
            "adj_volume",
            "price_ge_5",
        )
        .sort(["date", "symbol"])
    )
    return panel.collect()


#: v2 completion에서 manifest로 옮기는 항목. 월별 통계(``month_stats``) 같은 큰 것은 뺀다.
_COMPLETION_KEYS = (
    "table",
    "snapshot_date",
    "mode",
    "start",
    "end",
    "rule_version",
    "rule_versions",
    "snapshot_sha256",
    "input_snapshots",
    "input_snapshot_sha256",
    "dolt_stocks",
    "security_segments_snapshot",
    "security_master_snapshot",
)


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def universe_manifest(
    lake: UsLake,
    universe_version: str,
    *,
    security_boundaries: bool = False,
    require_completion: bool = True,
) -> dict[str, object]:
    """데이터셋 manifest에 적는 유니버스 출처 — 버전, 읽은 표와 ``snapshot_date``, v2 completion.

    v2면 ``universe_daily_v2`` 스냅샷 옆 ``completion.json``에서 규칙 버전·입력 스냅샷·입력
    sha256·dolt 커밋을 옮기고, 그 completion이 적은 ``snapshot_sha256``이 읽는 parquet의 실제
    sha256과 같은지 확인한다(다르면 ``ValueError``). ``require_completion=False``면 completion이
    없어도 넘어간다(합성 데이터 시험용) — 실데이터 빌드는 항상 True다.
    ``security_boundaries``가 켜져 있으면 ``security_segments`` 스냅샷 날짜도 적는다.
    """
    if universe_version not in UNIVERSE_VERSIONS:
        raise ValueError(
            f"universe_version은 {UNIVERSE_VERSIONS} 중 하나여야 합니다: {universe_version!r}"
        )
    tables = ["universe_daily"] + (["universe_daily_v2"] if universe_version == "v2" else [])
    result: dict[str, object] = {
        "universe_version": universe_version,
        "universe_tables": {t: lake.latest_snapshot(t).isoformat() for t in tables},
        "security_boundaries": security_boundaries,
    }
    if security_boundaries:
        result["security_segments_snapshot"] = lake.latest_snapshot("security_segments").isoformat()
    if universe_version == "v2":
        snapshot = lake.snapshot_dir("universe_daily_v2")
        completion_path = snapshot / "completion.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text())
            actual = _sha256_file(snapshot / "part.parquet")
            if completion.get("snapshot_sha256") != actual:
                raise ValueError(
                    f"{completion_path}의 snapshot_sha256이 읽는 parquet의 sha256과 다릅니다."
                )
            result["universe_v2_completion"] = {
                k: completion[k] for k in _COMPLETION_KEYS if k in completion
            }
        elif require_completion:
            raise FileNotFoundError(f"universe_daily_v2의 completion.json이 없습니다: {snapshot}")
    return result
