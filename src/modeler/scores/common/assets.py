"""시장·섹터 자산 레지스트리 (코드로 관리한다, YAML 아님).

사양: ``my/milestones/common/scores/20260929_market_sector_modeling/01_data_and_architecture.md``
§6 "자산 정의" — ``asset_id, asset_type, market, currency, calendar, parent benchmark,
proxy, inception, history type, definition version``.

**여기 항목을 바꾸면 ``ASSET_REGISTRY_VERSION``을 올린다.** 데이터셋 manifest가
버전과 항목 해시를 같이 남긴다.

KR 섹터는 일부러 비워 뒀다 — KRX Open API 지수 목록을 백필한 뒤 수익률 결과를 보지
않고 고른다(README §5). 아래 ``KR_SECTOR_TODO``에 표시만 있다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

AssetType = Literal["market", "style", "sector", "theme"]
Market = Literal["KR", "US"]
Source = Literal["us_prices_daily", "kr_krx_index_daily", "kr_ecos"]
HistoryType = Literal["live", "backfilled"]

ASSET_REGISTRY_VERSION = "ms_assets_v1"

CALENDAR_BY_MARKET: dict[str, str] = {"US": "XNYS", "KR": "XKRX"}


@dataclass(frozen=True)
class Asset:
    asset_id: str
    asset_type: AssetType
    market: Market
    currency: str
    calendar: Literal["XKRX", "XNYS"]
    parent_benchmark: str | None
    #: US는 ETF 심볼, KR은 KRX 지수 이름.
    proxy: str
    source: Source
    #: 공개된 상장/기준일. 레이크에서 확인한 값이 아니다 — 실제 첫 관측 세션은
    #: readiness 표가 따로 낸다. KR은 백필 뒤 확인하므로 ``None``.
    inception: str | None
    history_type: HistoryType
    definition_version: str = "v1"
    #: 주 원천이 막힐 때 쓸 대체 계열(KR: ECOS ``market_kospi_ecos`` 등).
    fallback_proxy: str | None = None
    fallback_source: Source | None = None


_US_COMMON = dict(
    market="US",
    currency="USD",
    calendar="XNYS",
    source="us_prices_daily",
    history_type="live",
)

ASSETS: tuple[Asset, ...] = (
    Asset(
        asset_id="us_spx",
        asset_type="market",
        parent_benchmark=None,
        proxy="SPY",
        inception="1993-01-22",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    # Nasdaq100은 style, 부모 벤치마크는 S&P500 (README §5). IXIC(Composite)이 아니다.
    Asset(
        asset_id="us_ndx",
        asset_type="style",
        parent_benchmark="us_spx",
        proxy="QQQ",
        inception="1999-03-10",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    Asset(
        asset_id="us_fin",
        asset_type="sector",
        parent_benchmark="us_spx",
        proxy="XLF",
        inception="1998-12-16",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    Asset(
        asset_id="us_hlth",
        asset_type="sector",
        parent_benchmark="us_spx",
        proxy="XLV",
        inception="1998-12-16",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    Asset(
        asset_id="us_ind",
        asset_type="sector",
        parent_benchmark="us_spx",
        proxy="XLI",
        inception="1998-12-16",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    Asset(
        asset_id="us_enrg",
        asset_type="sector",
        parent_benchmark="us_spx",
        proxy="XLE",
        inception="1998-12-16",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    Asset(
        asset_id="us_tech",
        asset_type="sector",
        parent_benchmark="us_spx",
        proxy="XLK",
        inception="1998-12-16",
        **_US_COMMON,  # type: ignore[arg-type]
    ),
    Asset(
        asset_id="kr_kospi",
        asset_type="market",
        market="KR",
        currency="KRW",
        calendar="XKRX",
        parent_benchmark=None,
        proxy="코스피",
        source="kr_krx_index_daily",
        inception=None,
        history_type="live",
        fallback_proxy="market_kospi_ecos",
        fallback_source="kr_ecos",
    ),
    Asset(
        asset_id="kr_kosdaq",
        asset_type="market",
        market="KR",
        currency="KRW",
        calendar="XKRX",
        parent_benchmark=None,
        proxy="코스닥",
        source="kr_krx_index_daily",
        inception=None,
        history_type="live",
    ),
)

#: TODO: KR 섹터 3~5개. KRX Open API ``idx/kospi_dd_trd`` 지수 목록을 백필한 뒤
#: (지수 이름·업종별 시작일·이름 변경 확인) **수익률을 보기 전에** 고른다.
#: 임의로 만들어 채우지 않는다. 부모 벤치마크는 ``kr_kospi``(또는 ``kr_kosdaq``).
#: HBM·방산 같은 테마는 ``asset_type="theme"``로 구분하고 sector와 섞지 않는다.
KR_SECTOR_TODO: tuple[str, ...] = ()

_BY_ID: dict[str, Asset] = {a.asset_id: a for a in ASSETS}
if len(_BY_ID) != len(ASSETS):  # pragma: no cover - 레지스트리 편집 실수 방어
    raise RuntimeError("asset_id가 중복됐습니다")
for _a in ASSETS:
    if _a.parent_benchmark is not None and _a.parent_benchmark not in _BY_ID:  # pragma: no cover
        raise RuntimeError(f"{_a.asset_id}의 parent_benchmark가 레지스트리에 없습니다")


def get_asset(asset_id: str) -> Asset:
    try:
        return _BY_ID[asset_id]
    except KeyError:
        raise KeyError(f"등록되지 않은 자산입니다: {asset_id!r}") from None


def assets_for_market(market: str) -> tuple[Asset, ...]:
    return tuple(a for a in ASSETS if a.market == market.upper())


def registry_hash(assets: tuple[Asset, ...] = ASSETS) -> str:
    """항목 전체의 sha256. 항목을 고치고 버전을 안 올린 실수를 manifest 비교로 잡는다."""
    payload = json.dumps([asdict(a) for a in assets], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
