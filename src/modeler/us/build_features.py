"""``us_features_v1`` 데이터셋을 만든다.

    uv run python -m modeler.us.build_features

``build_panel.py``와 같은 꼴이다 — 데이터셋은 지우면 다시 만들 수 있어야 하고,
manifest의 ``modeler_git_commit``이 가리키는 코드로 이 명령을 돌리면 같은
``content_hash``가 나오는 것이 재현의 뜻이다.

패널을 읽고 ``features.FAMILY_ORDER``의 16개 family를 순서대로 적용해 한 장으로
합친다. **피쳐를 더하거나 정의를 바꾸지 않는다** — ``us-features-frozen`` 태그
이후는 이미 있는 F1\\~F16을 모아 데이터셋으로 쓰는 작업이다(R6 동결).

**유니버스 v2** (``--universe-version v2``, 또는 저장된 패널을 ``--panel-name``으로 받으면 그
패널 manifest의 ``universe_version``). v2 데이터셋 이름에는 ``_u2``가 있어야 하고 v1에는 없어야
한다(``dataset.check_dataset_name``). ``--security-boundaries``는 롤링·분할 조정을 종목 구간
(``security_id``) 단위로 하는 **켜야만 동작하는** 모드다 — v2 패널에서만 쓰고, 이름에 ``_fwd``가
있으면 거부한다(전진 등록 §10의 보조 판정은 동결 코드 = 꺼짐). 같은 이름이 이미 있으면 쓰기를
거부한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.build_panel import universe_filter
from modeler.us.dataset import (
    assert_dataset_absent,
    check_dataset_name,
    git_commit,
    manifest_universe_version,
    read_panel_dataset,
    write_dataset,
)
from modeler.us.features import FAMILY_ORDER
from modeler.us.lake import UsLake
from modeler.us.panel import UNIVERSE_VERSIONS, build_panel, universe_manifest

DATASET_NAME = "us_features_v1"



def _missing_rate(df: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    """``columns`` 각각의 결측률(널 비율). ``df``가 비면 전부 ``NaN``으로 둔다."""
    if df.height == 0:
        return {c: float("nan") for c in columns}
    return {c: float(df[c].is_null().mean()) for c in columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=None,
        help=f"데이터셋 이름 (v1 기본: {DATASET_NAME}. v2는 _u2가 든 이름을 직접 준다)",
    )
    parser.add_argument(
        "--universe-version",
        choices=UNIVERSE_VERSIONS,
        default=None,
        help="유니버스 버전 (기본: v1). --panel-name을 주면 그 패널 manifest 값을 따른다.",
    )
    parser.add_argument(
        "--panel-name",
        default=None,
        help="저장된 패널 데이터셋을 입력으로 쓴다(없으면 패널을 새로 만든다).",
    )
    parser.add_argument(
        "--security-boundaries",
        action="store_true",
        help="롤링·분할 조정을 종목 구간(security_id) 단위로 한다. v2 패널에서만, _fwd 이름 불가.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="커밋 안 된 트리로도 만든다. **버리는 실험용이다** — "
        "manifest 의 커밋으로 다시 만들 수 없게 된다.",
    )
    args = parser.parse_args(argv)

    root = DataRoot.resolve(market="us")
    lake = UsLake.resolve()
    if args.panel_name:
        panel, panel_manifest = read_panel_dataset(root, args.panel_name)
        universe_version = manifest_universe_version(panel_manifest)
        if args.universe_version not in (None, universe_version):
            parser.error(
                f"--universe-version {args.universe_version}이 패널 manifest의 "
                f"{universe_version}과 다릅니다."
            )
    else:
        panel_manifest = None
        universe_version = args.universe_version or "v1"
    name = args.name or (DATASET_NAME if universe_version == "v1" else None)
    if name is None:
        parser.error("유니버스 v2는 --name(_u2가 든 이름)이 필요합니다.")
    check_dataset_name(
        name, universe_version=universe_version, security_boundaries=args.security_boundaries
    )
    assert_dataset_absent(root, name)
    if not args.panel_name:
        panel = build_panel(lake, universe_version=universe_version)
    panel_columns = list(panel.columns)
    feature_lake = lake.with_security_boundaries() if args.security_boundaries else lake

    features = panel
    family_order: list[dict[str, object]] = []
    feature_columns: list[str] = []
    for family_name, add_family in FAMILY_ORDER:
        before = set(features.columns)
        features = add_family(features, feature_lake)
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
        "universe_filter": universe_filter(universe_version),
        **universe_manifest(lake, universe_version, security_boundaries=args.security_boundaries),
        "source_panel": args.panel_name,
        "source_panel_content_hash": panel_manifest["content_hash"] if panel_manifest else None,
        "panel_start": str(panel["date"].min()),
        "panel_end": str(panel["date"].max()),
        "rebalance_dates": panel["date"].n_unique(),
        "panel_columns": panel_columns,
        "family_order": family_order,
        "feature_columns_total": len(feature_columns),
        "feature_missing_rate": missing_rate,
    }
    dataset_dir = write_dataset(features, root, name, manifest=manifest)
    print(f"{dataset_dir}  {features.height:,}행 · {features.width}열")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
