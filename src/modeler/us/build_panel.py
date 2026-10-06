"""``us_panel_v1`` 데이터셋을 만든다.

    uv run python -m modeler.us.build_panel
    uv run python -m modeler.us.build_panel --universe-version v2 --name us_panel_v3_u2

``--universe-version v2``는 멤버십만 ``universe_daily_v2``(PIT)로 바꾼 패널이다 — 스키마는 v1과
같고 이름에 ``_u2``가 있어야 한다(``dataset.check_dataset_name``). 같은 이름의 데이터셋이 이미
있으면 쓰기를 거부한다(동결 데이터셋을 덮지 않는다).

**임시 스크립트로 만들지 않는다.** 데이터셋은 지우면 다시 만들 수 있어야 하고
(``05_validation_protocol.md`` §6), 그러려면 만드는 방법이 저장소에 남아 있어야
한다. manifest의 ``modeler_git_commit``이 가리키는 코드로 이 명령을 돌리면 같은
``content_hash``가 나오는 것이 재현의 뜻이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modeler.etl.config import DataRoot
from modeler.us.dataset import (
    assert_dataset_absent,
    check_dataset_name,
    git_commit,
    write_dataset,
)
from modeler.us.lake import UsLake
from modeler.us.panel import UNIVERSE_VERSIONS, build_panel, universe_manifest

DATASET_NAME = "us_panel_v1"

#: 유니버스 필터 — manifest에 사람이 읽을 수 있게 남긴다 (``02`` §4).
UNIVERSE_FILTER = (
    "universe_daily.in_universe = 1 (수집 D7: ETF·test issue 제외, 증권종류 배제, "
    "ADV >= $1M 진입 / $0.7M 유지, 월 재판정). 주가 $5 하한은 여기서 거르지 않고 "
    "price_ge_5 플래그로만 낸다 — 지표 E는 하한 없이, I는 하한 있게 둘 다 내야 한다"
)

#: v2 패널의 필터 설명 — 멤버십만 다르고 나머지 열은 v1과 같다 (유니버스 v2 설계 §4).
UNIVERSE_FILTER_V2 = (
    "universe_daily_v2.in_universe = 1 (PIT 보기, 설계 02_design.md §2.3). 멤버십만 v2이고 "
    "cik·sic·mcap_rank·adv_20d·exchange·가격 열은 v1과 같은 원천·규칙이다. 주가 $5 하한은 "
    "여기서 거르지 않고 price_ge_5 플래그로만 낸다"
)


def universe_filter(universe_version: str) -> str:
    return UNIVERSE_FILTER if universe_version == "v1" else UNIVERSE_FILTER_V2




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe-version",
        choices=UNIVERSE_VERSIONS,
        default="v1",
        help="유니버스 버전 (기본: v1). v2는 멤버십만 universe_daily_v2(PIT).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=f"데이터셋 이름 (v1 기본: {DATASET_NAME}. v2는 _u2가 든 이름을 직접 준다)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="커밋 안 된 트리로도 만든다. **버리는 실험용이다** — "
        "manifest 의 커밋으로 다시 만들 수 없게 된다.",
    )
    args = parser.parse_args(argv)

    name = args.name or (DATASET_NAME if args.universe_version == "v1" else None)
    if name is None:
        parser.error("--universe-version v2는 --name(_u2가 든 이름)이 필요합니다.")
    check_dataset_name(name, universe_version=args.universe_version)
    root = DataRoot.resolve(market="us")
    assert_dataset_absent(root, name)

    lake = UsLake.resolve()
    panel = build_panel(lake, universe_version=args.universe_version)

    modeler_repo = Path(__file__).resolve().parents[3]
    collector_repo = modeler_repo.parent / "collector"

    manifest = {
        "input_table_snapshots": lake.snapshot_manifest(),
        "modeler_git_commit": git_commit(modeler_repo, allow_dirty=args.allow_dirty),
        "collector_git_commit": git_commit(collector_repo, allow_dirty=args.allow_dirty),
        "universe_filter": universe_filter(args.universe_version),
        **universe_manifest(lake, args.universe_version),
        "panel_start": str(panel["date"].min()),
        "panel_end": str(panel["date"].max()),
        "rebalance_dates": panel["date"].n_unique(),
    }
    dataset_dir = write_dataset(panel, root, name, manifest=manifest)
    print(f"{dataset_dir}  {panel.height:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
