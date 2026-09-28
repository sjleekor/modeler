"""``us_features_flow_v1`` 데이터셋을 만든다 — F17(FTD)·F18(주문흐름)·F19(기관보유).

    uv run python -m modeler.us.build_flow_features

``us4_flow_features/00_draft.md`` U-D3 그대로다: **``us_panel_v2``의 키(``date``,
``symbol``)만 읽어** 새 family 셋을 붙인다. ``build_panel()``을 다시 불러 패널을
새로 짓지 않는다 — 레이크가 그새 바뀌었으면 동결된 ``us_features_v2``(44개)와
키 집합이 달라질 수 있어, 이미 만들어 둔 동결 패널의 키를 그대로 재사용해야
44개·새 family가 같은 (date, symbol) 축 위에 있다고 보장할 수 있다.

``build_features.py``의 관례(더러운 트리 거부, ``git_commit``·``write_dataset``
manifest)를 그대로 따르되, 둘 더 넣는다 — **출력 디렉터리가 이미 있으면 멈춘다**
(``us_features_v2`` 같은 기존 데이터셋을 실수로 덮지 않는다) · 지연 상수 셋을
manifest에 적는다(``02_lag_constants.md``).

**F19는 조건부다(U-D8).** 그래도 이 스크립트는 F19가 레이크에 없다고 조용히
건너뛰지 않는다 — ``FAMILY_ORDER``에 그대로 셋째로 올라 있고,
``inst_holdings_q``·``cusip_symbol_pit``(혹은 ``prices_daily``) 표가
``$STOCK_DATA_ROOT``에 없으면 ``add_institutional``이 ``FileNotFoundError``를
던져 그 자리에서 멈춘다. U-D8의 "10/11까지 표가 안 굳으면 뺀다"는 판단은
**사람이 그 시점에 이 튜플에서 F19 행을 지우는 방식**으로 하는 것이지, 코드가
표 유무를 보고 스스로 빼지 않는다(조용한 실패를 피하려는 이 저장소의 관례).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.build_features import _missing_rate
from modeler.us.dataset import git_commit, write_dataset
from modeler.us.features.ftd import LAG_FTD_DAYS, add_ftd
from modeler.us.features.institutional import LAG_13F_DAYS, add_institutional
from modeler.us.features.order_flow import (
    MIDAS_AVAILABLE_FROM,
    MIDAS_FALLBACK_LAG_DAYS,
    add_order_flow,
)
from modeler.us.lake import UsLake

DATASET_NAME = "us_features_flow_v1"
SOURCE_PANEL_NAME = "us_panel_v2"

#: (family 이름, add_<family> 함수) — F17·F18·F19. F19는 조건부(U-D8)라도
#: 표가 없으면 에러로 멈추게 그대로 둔다(모듈독스트링 참고) — 조용히 빼지 않는다.
FAMILY_ORDER: tuple[tuple[str, object], ...] = (
    ("F17_ftd", add_ftd),
    ("F18_order_flow", add_order_flow),
    ("F19_institutional", add_institutional),
)


class OutputExistsError(RuntimeError):
    """출력 데이터셋 디렉터리가 이미 있다 — 덮지 않고 멈춘다."""


def _read_panel_keys(root: DataRoot) -> pl.DataFrame:
    """``us_panel_v2``에서 ``(date, symbol)`` 키만 읽는다. 다른 컬럼은 안 가져온다."""
    panel_dir = root.datasets / SOURCE_PANEL_NAME
    part = panel_dir / "part.parquet"
    if not part.exists():
        raise FileNotFoundError(
            f"{SOURCE_PANEL_NAME} 데이터셋이 없습니다: {part}. 먼저 build_panel을 돌려야 합니다."
        )
    keys = pl.read_parquet(part, columns=["date", "symbol"]).unique().sort(["date", "symbol"])
    return keys


def _source_panel_manifest(root: DataRoot) -> dict[str, object]:
    manifest_path = root.datasets / SOURCE_PANEL_NAME / "manifest.json"
    return json.loads(manifest_path.read_text())


def _lag_constants_manifest() -> dict[str, object]:
    return {
        "LAG_FTD_DAYS": LAG_FTD_DAYS,
        "MIDAS_AVAILABLE_FROM": {
            f"{year}q{quarter}": available_from.isoformat()
            for (year, quarter), available_from in sorted(MIDAS_AVAILABLE_FROM.items())
        },
        "MIDAS_FALLBACK_LAG_DAYS": MIDAS_FALLBACK_LAG_DAYS,
        "LAG_13F_DAYS": LAG_13F_DAYS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default=DATASET_NAME, help=f"데이터셋 이름 (기본: {DATASET_NAME})"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="커밋 안 된 트리로도 만든다. **버리는 실험용이다** — "
        "manifest 의 커밋으로 다시 만들 수 없게 된다.",
    )
    args = parser.parse_args(argv)

    root = DataRoot.resolve(market="us")
    dataset_dir = root.datasets / args.name
    if dataset_dir.exists():
        print(
            f"{dataset_dir} 가 이미 있습니다 — 덮지 않고 멈춥니다. "
            "다른 이름(--name)을 쓰거나 기존 디렉터리를 사람이 직접 지우십시오.",
            file=sys.stderr,
        )
        return 1

    lake = UsLake.resolve()
    panel_keys = _read_panel_keys(root)
    panel_columns = list(panel_keys.columns)

    features = panel_keys
    family_order: list[dict[str, object]] = []
    feature_columns: list[str] = []
    for family_name, add_family in FAMILY_ORDER:
        before = set(features.columns)
        features = add_family(features, lake)
        added = [c for c in features.columns if c not in before]
        family_order.append({"family": family_name, "columns_added": added})
        feature_columns.extend(added)

    missing_rate = _missing_rate(features, feature_columns)

    modeler_repo = Path(__file__).resolve().parents[3]
    collector_repo = modeler_repo.parent / "collector"

    manifest = {
        "input_table_snapshots": lake.snapshot_manifest(),
        "modeler_git_commit": git_commit(modeler_repo, allow_dirty=args.allow_dirty),
        "collector_git_commit": git_commit(collector_repo, allow_dirty=args.allow_dirty),
        "source_panel": SOURCE_PANEL_NAME,
        "source_panel_manifest": _source_panel_manifest(root),
        "panel_columns": panel_columns,
        "panel_start": str(features["date"].min()),
        "panel_end": str(features["date"].max()),
        "rebalance_dates": features["date"].n_unique(),
        "lag_constants": _lag_constants_manifest(),
        "family_order": family_order,
        "feature_columns_total": len(feature_columns),
        "feature_missing_rate": missing_rate,
    }
    written_dir = write_dataset(features, root, args.name, manifest=manifest)
    print(f"{written_dir}  {features.height:,}행 · {features.width}열")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
