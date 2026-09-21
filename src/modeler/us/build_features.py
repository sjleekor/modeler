"""``us_features_v1`` 데이터셋을 만든다.

    uv run python -m modeler.us.build_features

``build_panel.py``와 같은 꼴이다 — 데이터셋은 지우면 다시 만들 수 있어야 하고,
manifest의 ``modeler_git_commit``이 가리키는 코드로 이 명령을 돌리면 같은
``content_hash``가 나오는 것이 재현의 뜻이다.

패널을 읽고 ``features.FAMILY_ORDER``의 16개 family를 순서대로 적용해 한 장으로
합친다. **피쳐를 더하거나 정의를 바꾸지 않는다** — ``us-features-frozen`` 태그
이후는 이미 있는 F1\\~F16을 모아 데이터셋으로 쓰는 작업이다(R6 동결).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.build_panel import UNIVERSE_FILTER
from modeler.us.dataset import git_commit, write_dataset
from modeler.us.features import FAMILY_ORDER
from modeler.us.lake import UsLake
from modeler.us.panel import build_panel

DATASET_NAME = "us_features_v1"



def _missing_rate(df: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    """``columns`` 각각의 결측률(널 비율). ``df``가 비면 전부 ``NaN``으로 둔다."""
    if df.height == 0:
        return {c: float("nan") for c in columns}
    return {c: float(df[c].is_null().mean()) for c in columns}


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

    lake = UsLake.resolve()
    panel = build_panel(lake)
    panel_columns = list(panel.columns)

    features = panel
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
        "universe_filter": UNIVERSE_FILTER,
        "panel_start": str(panel["date"].min()),
        "panel_end": str(panel["date"].max()),
        "rebalance_dates": panel["date"].n_unique(),
        "panel_columns": panel_columns,
        "family_order": family_order,
        "feature_columns_total": len(feature_columns),
        "feature_missing_rate": missing_rate,
    }
    dataset_dir = write_dataset(
        features, DataRoot.resolve(market="us"), args.name, manifest=manifest
    )
    print(f"{dataset_dir}  {features.height:,}행 · {features.width}열")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
