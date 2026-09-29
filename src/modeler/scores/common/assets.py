"""시장·섹터 자산 레지스트리 (코드로 관리한다, YAML 아님).

사양: ``my/milestones/common/scores/20260929_market_sector_modeling/01_data_and_architecture.md``
§6 "자산 정의" — ``asset_id, asset_type, market, currency, calendar, parent benchmark,
proxy, inception, history type, definition version``.

**여기 항목을 바꾸면 ``ASSET_REGISTRY_VERSION``을 올린다.** 데이터셋 manifest가
버전과 항목 해시를 같이 남긴다.

KR 섹터 다섯은 ``ms_assets_v2``부터 활성이다(2026-09-30 승인, P-7). KRX 300 계열과 KRX
유틸리티·정보기술·경기소비재·필수소비재·K콘텐츠는 기준일이 데이터 시작일(2010-01-04)이고
그날 종가가 정확히 1000.0000이라 소급 계산 이력이다 — 탈락(``KR_SECTOR_REJECTED_KRX300``에 기록만
남긴다). 기준일이 2010 이전이라 2010-01-04 값이 1000이 아닌 옛 KRX 업종지수 다섯을 쓴다.
US 대응은 근사다(은행만 = XLF보다 좁다, 기계장비 = 자본재, 에너지화학 = 에너지+화학,
반도체 = 정보기술보다 좁다).

KR 자산의 ``(index_group, idx_nm)``은 ``kr_index_key(asset_id)``가 준다. ``Asset``에 필드를
더하지 않은 것은 ``registry_hash``를 안 바꾸려는 것이다(US 동결 manifest와 비교되기 때문).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, NamedTuple

AssetType = Literal["market", "style", "sector", "theme"]
Market = Literal["KR", "US"]
Source = Literal["us_prices_daily", "kr_krx_index_daily", "kr_ecos"]
HistoryType = Literal["live", "backfilled", "live_since_before_2010_by_base_date"]

ASSET_REGISTRY_VERSION = "ms_assets_v2"

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


class KrSectorCandidate(NamedTuple):
    asset_id: str
    index_group: str
    idx_nm: str
    note: str


#: **탈락한** KRX 300 섹터 후보(기록용, 코드가 안 쓴다). 이 지수들은 2010-01-04 종가가 정확히
#: 1000.0000이라 기준일 = 데이터 시작일인 소급 계산 이력이다(2026-09-30 실측).
KR_SECTOR_REJECTED_KRX300: tuple[KrSectorCandidate, ...] = (
    KrSectorCandidate("kr_fin", "krx", "KRX 300 금융", "탈락 · 소급 이력"),
    KrSectorCandidate("kr_hlth", "krx", "KRX 300 헬스케어", "탈락 · 소급 이력"),
    KrSectorCandidate("kr_ind", "krx", "KRX 300 산업재", "탈락 · 소급 이력"),
    KrSectorCandidate("kr_tech", "krx", "KRX 300 정보기술", "탈락 · 소급 이력"),
)

_KR_SECTOR_COMMON = dict(
    asset_type="sector",
    market="KR",
    currency="KRW",
    calendar="XKRX",
    parent_benchmark="kr_kospi",
    source="kr_krx_index_daily",
    inception=None,
    # 기준일이 2010 이전이라 2010-01-04 값이 1000이 아니다(실측). 소급 이력이 아니라고 본다.
    history_type="live_since_before_2010_by_base_date",
)

#: KR 섹터 다섯(P-7). 프록시는 ``(index_group="krx", idx_nm)``. US 대응은 근사다.
#: 2010-01-04 종가(실측): 은행 923.92 · 헬스케어 1151.62 · 기계장비 1305.36 ·
#: 에너지화학 1616.74 · 반도체 1601.89.
_KR_SECTORS: tuple[Asset, ...] = (
    Asset(asset_id="kr_fin", proxy="KRX 은행", **_KR_SECTOR_COMMON),  # type: ignore[arg-type]
    Asset(asset_id="kr_hlth", proxy="KRX 헬스케어", **_KR_SECTOR_COMMON),  # type: ignore[arg-type]
    Asset(asset_id="kr_ind", proxy="KRX 기계장비", **_KR_SECTOR_COMMON),  # type: ignore[arg-type]
    Asset(asset_id="kr_enrg", proxy="KRX 에너지화학", **_KR_SECTOR_COMMON),  # type: ignore[arg-type]
    Asset(asset_id="kr_tech", proxy="KRX 반도체", **_KR_SECTOR_COMMON),  # type: ignore[arg-type]
)
ASSETS = ASSETS + _KR_SECTORS
KR_SECTOR_PARENT = "kr_kospi"

#: 대표지수 두 개의 KRX 키. 섹터는 ``("krx", proxy)``다.
_KR_MARKET_KEYS: dict[str, tuple[str, str]] = {
    "kr_kospi": ("kospi", "코스피"),
    "kr_kosdaq": ("kosdaq", "코스닥"),
}

_BY_ID: dict[str, Asset] = {a.asset_id: a for a in ASSETS}
if len(_BY_ID) != len(ASSETS):  # pragma: no cover - 레지스트리 편집 실수 방어
    raise RuntimeError("asset_id가 중복됐습니다")
for _a in ASSETS:
    if _a.parent_benchmark is not None and _a.parent_benchmark not in _BY_ID:  # pragma: no cover
        raise RuntimeError(f"{_a.asset_id}의 parent_benchmark가 레지스트리에 없습니다")


def active_assets() -> tuple[Asset, ...]:
    return ASSETS


def registry_version() -> str:
    return ASSET_REGISTRY_VERSION


def kr_index_key(asset_id: str) -> tuple[str, str]:
    """KR 자산의 ``krx_index_daily`` 키 ``(index_group, idx_nm)``.

    ``idx_nm``은 그룹을 넘어 유일하지 않다(``건설`` 등 20개가 kospi·kosdaq에 모두 있다).
    """
    if asset_id in _KR_MARKET_KEYS:
        return _KR_MARKET_KEYS[asset_id]
    for a in _KR_SECTORS:
        if a.asset_id == asset_id:
            return "krx", a.proxy
    raise KeyError(f"KRX 지수 키가 없는 자산입니다: {asset_id!r}")


def get_asset(asset_id: str) -> Asset:
    if asset_id in _BY_ID:
        return _BY_ID[asset_id]
    raise KeyError(f"등록되지 않은 자산입니다: {asset_id!r}")


def assets_for_market(market: str) -> tuple[Asset, ...]:
    return tuple(a for a in active_assets() if a.market == market.upper())


def registry_hash(assets: tuple[Asset, ...] | None = None) -> str:
    """항목 전체의 sha256. 항목을 고치고 버전을 안 올린 실수를 manifest 비교로 잡는다.

    ``assets``를 안 주면 ``ASSETS`` 전체다.
    """
    use = active_assets() if assets is None else assets
    payload = json.dumps([asdict(a) for a in use], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
