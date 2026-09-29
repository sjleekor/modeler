"""시장·섹터 자산 레지스트리 (코드로 관리한다, YAML 아님).

사양: ``my/milestones/common/scores/20260929_market_sector_modeling/01_data_and_architecture.md``
§6 "자산 정의" — ``asset_id, asset_type, market, currency, calendar, parent benchmark,
proxy, inception, history type, definition version``.

**여기 항목을 바꾸면 ``ASSET_REGISTRY_VERSION``을 올린다.** 데이터셋 manifest가
버전과 항목 해시를 같이 남긴다.

KR 섹터는 **아직 활성 등록부에 없다.** 후보 5개는 ``KR_SECTOR_CANDIDATES``에 있고
(``my/.../20260929_market_sector_modeling/04_kr_sector_candidates.md``, 사용자 승인 대기),
``ASSETS``·``ASSET_REGISTRY_VERSION``은 그대로다. 승인 뒤에는 두 가지 중 하나를 한다.

* 실험: ``build_panel --market kr --kr-sectors kr_fin kr_hlth ...`` 처럼 프로세스 안에서
  ``activate_kr_sectors(ids)``를 부른다. 활성화한 id는 패널 manifest에 남고 ``run``이 같은 것을
  다시 켠다. 이때 ``registry_version()``은 ``ms_assets_v1+kr_sectors``로 갈라져 US 등록부와
  섞이지 않는다.
* 동결: 후보를 ``ASSETS``로 옮기고 ``ASSET_REGISTRY_VERSION``을 올린다.

KR 자산의 ``(index_group, idx_nm)``은 ``kr_index_key(asset_id)``가 준다. ``Asset``에 필드를
더하지 않은 것은 ``registry_hash``를 안 바꾸려는 것이다(US 동결 manifest와 비교되기 때문).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, NamedTuple

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


class KrSectorCandidate(NamedTuple):
    asset_id: str
    index_group: str
    idx_nm: str
    note: str


#: KR 섹터 후보 5개 (2026-09-30 실측, ``04_kr_sector_candidates.md`` §3). **승인 전이라 활성
#: 등록부(``ASSETS``)에 없다.** 수익률로 고르지 않았다. 부모 벤치마크는 ``kr_kospi``다.
#: HBM·방산 같은 테마는 ``asset_type="theme"``로 구분하고 sector와 섞지 않는다.
KR_SECTOR_CANDIDATES: tuple[KrSectorCandidate, ...] = (
    KrSectorCandidate("kr_fin", "krx", "KRX 300 금융", "XLF 대응 · 은행·증권·보험"),
    KrSectorCandidate("kr_hlth", "krx", "KRX 300 헬스케어", "XLV 대응"),
    KrSectorCandidate("kr_ind", "krx", "KRX 300 산업재", "XLI 대응"),
    KrSectorCandidate(
        "kr_enrg", "krx", "KRX 에너지화학", "XLE 대응 · 순수 에너지 지수가 없어 에너지화학"
    ),
    KrSectorCandidate("kr_tech", "krx", "KRX 300 정보기술", "XLK 대응 · 반도체 포함"),
)
KR_SECTOR_PARENT = "kr_kospi"

#: 대표지수 두 개의 KRX 키. 섹터는 ``KR_SECTOR_CANDIDATES``에서 온다.
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


#: 프로세스 안에서 켠 KR 섹터. ``activate_kr_sectors``만 채운다.
_ACTIVATED: dict[str, Asset] = {}


def _candidate_to_asset(c: KrSectorCandidate) -> Asset:
    return Asset(
        asset_id=c.asset_id,
        asset_type="sector",
        market="KR",
        currency="KRW",
        calendar="XKRX",
        parent_benchmark=KR_SECTOR_PARENT,
        proxy=c.idx_nm,
        source="kr_krx_index_daily",
        inception=None,
        # KRX 300 계열의 실제 출시일은 미확인 — 소급 계산 이력일 수 있다.
        history_type="backfilled",
    )


def activate_kr_sectors(ids: Sequence[str]) -> tuple[Asset, ...]:
    """승인된 KR 섹터를 이 프로세스의 활성 등록부에 넣는다(멱등).

    ``ids``는 ``KR_SECTOR_CANDIDATES``의 ``asset_id``다. 후보에 없으면 ``KeyError``.
    활성화하면 ``registry_version()``이 ``<버전>+kr_sectors``로 바뀌고 ``registry_hash()``가
    활성 항목을 포함한다. 동결할 때는 후보를 ``ASSETS``로 옮기고 버전을 올린다.
    """
    by_id = {c.asset_id: c for c in KR_SECTOR_CANDIDATES}
    out = []
    for i in ids:
        if i not in by_id:
            raise KeyError(f"KR 섹터 후보가 아닙니다: {i!r} (후보: {sorted(by_id)})")
        _ACTIVATED.setdefault(i, _candidate_to_asset(by_id[i]))
        out.append(_ACTIVATED[i])
    return tuple(out)


def deactivate_kr_sectors() -> None:
    """활성화한 KR 섹터를 모두 뺀다(테스트·재실행용)."""
    _ACTIVATED.clear()


def active_kr_sector_ids() -> tuple[str, ...]:
    return tuple(_ACTIVATED)


def active_assets() -> tuple[Asset, ...]:
    return ASSETS + tuple(_ACTIVATED.values())


def registry_version() -> str:
    """활성 항목 기준 등록부 버전. KR 섹터를 켜지 않았으면 ``ASSET_REGISTRY_VERSION`` 그대로."""
    return f"{ASSET_REGISTRY_VERSION}+kr_sectors" if _ACTIVATED else ASSET_REGISTRY_VERSION


def kr_index_key(asset_id: str) -> tuple[str, str]:
    """KR 자산의 ``krx_index_daily`` 키 ``(index_group, idx_nm)``.

    ``idx_nm``은 그룹을 넘어 유일하지 않다(``건설`` 등 20개가 kospi·kosdaq에 모두 있다).
    """
    if asset_id in _KR_MARKET_KEYS:
        return _KR_MARKET_KEYS[asset_id]
    for c in KR_SECTOR_CANDIDATES:
        if c.asset_id == asset_id:
            return c.index_group, c.idx_nm
    raise KeyError(f"KRX 지수 키가 없는 자산입니다: {asset_id!r}")


def get_asset(asset_id: str) -> Asset:
    if asset_id in _BY_ID:
        return _BY_ID[asset_id]
    if asset_id in _ACTIVATED:
        return _ACTIVATED[asset_id]
    raise KeyError(f"등록되지 않은 자산입니다: {asset_id!r}")


def assets_for_market(market: str) -> tuple[Asset, ...]:
    return tuple(a for a in active_assets() if a.market == market.upper())


def registry_hash(assets: tuple[Asset, ...] | None = None) -> str:
    """항목 전체의 sha256. 항목을 고치고 버전을 안 올린 실수를 manifest 비교로 잡는다.

    ``assets``를 안 주면 활성 등록부(``ASSETS`` + 켠 KR 섹터)다.
    KR 섹터를 안 켰으면 전과 같은 값이다.
    """
    use = active_assets() if assets is None else assets
    payload = json.dumps([asdict(a) for a in use], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
