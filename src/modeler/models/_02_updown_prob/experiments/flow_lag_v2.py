"""flow_lag_v2 — 잔고 두 컬럼에 공표 지연을 넣은 ``feat_flow`` 를 따로 만든다 (10월 계획 4.5).

Plan: ``my/milestones/kr/modeling/plan/20260927_october_plan/01_schedule.md`` §4 표 4.5,
``20260927_short_balance_correction.md`` §5.

채택 config 의 ``flow_short_balance_qty`` · ``flow_short_balance_chg_20d`` 는 측정일 값을
``lag1`` 만 밀어 쓴다. 잔고는 측정일 + 2~3세션 뒤 저녁에야 레이크에 들어오므로 formation
시점에 아직 모르는 값이 들어간다. 이 모듈은 그 두 컬럼만 ``LAG_SESSIONS`` 세션 더 미룬
``feat_flow`` 를 **별도 lake root** 에 만든다.

* 공유 lake 의 ``feat_flow`` 는 건드리지 않는다. 나머지 마트는 전부 상대 경로 symlink 다
  (``isolated_lake`` 의 ``_e5`` 와 같은 방식).
* ``MODEL_CODE_FILES`` 에 든 파일은 하나도 안 바뀐다. ``build_dataset`` 은 root 만 다른
  같은 이름의 ``feat_flow`` 를 읽고, 데이터셋은 이 root 아래 ``datasets/`` 에 새로 생긴다
  (루트 이름이 ``_v2`` 이므로 동결본 ``FS1h_h20_lag1_rank`` 와 경로가 겹치지 않는다).
* 이 모듈은 데이터셋을 만들지 않는다. ``--link`` 와 ``--build`` 는 마트까지다.

    uv run python -m modeler.models._02_updown_prob.experiments.flow_lag_v2 --link
    uv run python -m modeler.models._02_updown_prob.experiments.flow_lag_v2 --build
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import replace

from modeler.etl.config import DataRoot, LakeConfig
from modeler.models._02_updown_prob.experiments import isolated_lake as iso
from modeler.models._02_updown_prob.spec import SNAPSHOT_DATE, SOURCE, lake_config

#: 정정 문서 §5 · 계획 4.5. 3 으로 정했다(사용자 결정 2026-10-07): 9월 이후 측정일 21건이 모두
#: 3세션 안에 들어왔고 2세션 기준은 8건만 맞았다. 장 마감 전 적재는 0건이라 lag1 경로와 합쳐
#: 측정일 D 값은 D+4 행부터 보인다(TRS 와 같다) — 03_probe §8.1. 값을 바꿀 때는 root 이름도 바꾼다.
LAG_SESSIONS = 3
ROOT_NAME = "_flow_lag3_v2"
FLOW_MART = "feat_flow"


@functools.lru_cache(maxsize=1)
def lagged_root() -> DataRoot:
    return DataRoot(base=iso._shared_root().derived / ROOT_NAME)


def lagged_lake() -> LakeConfig:
    """공유 snapshot 과 같은 ``LakeConfig`` 에서 root 만 바꾼 것."""
    return replace(lake_config(), root=lagged_root())


def link() -> int:
    """공유 lake 를 relative symlink 로 잇고 ``feat_flow`` 자리만 비워 둔다."""
    shared_root, root = iso._shared_root(), lagged_root()
    shared_marts = iso.feature_dir(shared_root)
    if not shared_marts.is_dir():
        raise SystemExit(f"shared snapshot missing: {shared_marts}")
    for base_name, lake in iso.LINKED_LAKES:
        target = (
            getattr(shared_root, base_name) / lake
            / f"snapshot_date={SNAPSHOT_DATE}" / f"source={SOURCE}"
        )
        if not target.is_dir():
            print(f"  {lake}: absent on the shared snapshot, skipped")
            continue
        dest = getattr(root, base_name) / lake / f"snapshot_date={SNAPSHOT_DATE}"
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / f"source={SOURCE}").exists():
            iso._symlink_relative(dest / f"source={SOURCE}", target)
    marts = iso.feature_dir(root)
    marts.mkdir(parents=True, exist_ok=True)
    linked = 0
    for entry in sorted(shared_marts.iterdir()):
        dest = marts / entry.name
        if entry.name == FLOW_MART:
            if dest.is_symlink():
                dest.unlink()
            continue
        if not dest.exists():
            iso._symlink_relative(dest, entry)
            linked += 1
    print(f"  feature: {linked} marts linked, {FLOW_MART} left to build")
    print(f"lagged lake: {root.base}")
    return 0


def build() -> int:
    """``feat_flow`` 만 ``LAG_SESSIONS`` 로 새로 쓴다. 공유 lake 는 읽기만 한다."""
    from modeler.etl.features.flow import materialize_flow  # noqa: PLC0415
    from modeler.etl.lake import connect, register_views  # noqa: PLC0415
    from modeler.etl.quality import QUALITY_TABLE  # noqa: PLC0415
    from modeler.etl.stock_pit import PIT_TABLE  # noqa: PLC0415
    from modeler.models._02_updown_prob.build_dataset import (  # noqa: PLC0415
        register_read_only,
    )

    config = lagged_lake()
    if iso.feature_dir(lagged_root()).joinpath(FLOW_MART).is_symlink():
        raise SystemExit(f"{FLOW_MART} is a symlink into the shared lake - run --link first")
    con = connect(config)
    register_views(con, config, tables=["daily_ohlcv", "krx_security_flow_raw"])
    register_read_only(con, config, [PIT_TABLE, QUALITY_TABLE])
    materialize_flow(
        con,
        config,
        price_view="daily_ohlcv",
        pit_view=PIT_TABLE,
        quality_view=QUALITY_TABLE,
        short_balance_lag_sessions=LAG_SESSIONS,
    )
    con.close()
    print(f"  {FLOW_MART}: done (lag {LAG_SESSIONS}) -> {lagged_root().base}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link", action="store_true", help="create the symlinked root")
    parser.add_argument("--build", action="store_true", help="materialize feat_flow with the lag")
    args = parser.parse_args(argv)
    if not (args.link or args.build):
        parser.error("one of --link / --build is required")
    if args.link:
        link()
    if args.build:
        return build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
