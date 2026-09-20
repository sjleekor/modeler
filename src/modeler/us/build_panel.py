"""``us_panel_v1`` 데이터셋을 만든다.

    uv run python -m modeler.us.build_panel

**임시 스크립트로 만들지 않는다.** 데이터셋은 지우면 다시 만들 수 있어야 하고
(``05_validation_protocol.md`` §6), 그러려면 만드는 방법이 저장소에 남아 있어야
한다. manifest의 ``modeler_git_commit``이 가리키는 코드로 이 명령을 돌리면 같은
``content_hash``가 나오는 것이 재현의 뜻이다.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from modeler.etl.config import DataRoot
from modeler.us.dataset import write_dataset
from modeler.us.lake import UsLake
from modeler.us.panel import build_panel

DATASET_NAME = "us_panel_v1"

#: 유니버스 필터 — manifest에 사람이 읽을 수 있게 남긴다 (``02`` §4).
UNIVERSE_FILTER = (
    "universe_daily.in_universe = 1 (수집 D7: ETF·test issue 제외, 증권종류 배제, "
    "ADV >= $1M 진입 / $0.7M 유지, 월 재판정). 주가 $5 하한은 여기서 거르지 않고 "
    "price_ge_5 플래그로만 낸다 — 지표 E는 하한 없이, I는 하한 있게 둘 다 내야 한다"
)


def _git_commit(repo: Path) -> str:
    """``repo``의 HEAD 커밋. 작업 트리가 더러우면 ``-dirty``를 붙인다.

    더러운 트리로 만든 데이터셋은 어느 코드로 만들었는지 커밋만으로는 못 찾는다.
    그 사실을 manifest에 남겨야 나중에 재현이 안 될 때 원인을 안다.
    """
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default=DATASET_NAME, help=f"데이터셋 이름 (기본: {DATASET_NAME})"
    )
    args = parser.parse_args(argv)

    lake = UsLake.resolve()
    panel = build_panel(lake)

    modeler_repo = Path(__file__).resolve().parents[3]
    collector_repo = modeler_repo.parent / "collector"

    manifest = {
        "input_table_snapshots": lake.snapshot_manifest(),
        "modeler_git_commit": _git_commit(modeler_repo),
        "collector_git_commit": _git_commit(collector_repo),
        "universe_filter": UNIVERSE_FILTER,
        "panel_start": str(panel["date"].min()),
        "panel_end": str(panel["date"].max()),
        "rebalance_dates": panel["date"].n_unique(),
    }
    dataset_dir = write_dataset(panel, DataRoot.resolve(market="us"), args.name, manifest=manifest)
    print(f"{dataset_dir}  {panel.height:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
