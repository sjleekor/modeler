"""``us_labels_v1`` 데이터셋을 만든다.

    uv run python -m modeler.us.build_labels
    uv run python -m modeler.us.build_labels --horizon 5
    uv run python -m modeler.us.build_labels --horizon 63

``build_panel.py``와 같은 꼴이다 — 데이터셋은 지우면 다시 만들 수 있어야 하고,
manifest의 ``modeler_git_commit``이 가리키는 코드로 이 명령을 돌리면 같은
``content_hash``가 나오는 것이 재현의 뜻이다.

``--horizon``은 ``labels.build_labels()``의 ``horizon`` 인자를 그대로 넘긴다
(``04_feature_test_plan.md`` §4 — 단일피쳐 검정이 h5·h63의 IC 감쇠를 본다).
기본값(21, h21)일 때는 데이터셋 이름이 기존 그대로 ``us_labels_v1``이다 —
h21의 ``content_hash``가 이 파라미터화 전과 같아야 재현이 깨지지 않은
것이다. 다른 horizon은 ``us_labels_h{N}_v1``로 이름이 갈린다.

**유니버스 v2** (``--universe-version v2`` 또는 ``--panel-name``으로 받은 패널의 manifest). v2
데이터셋 이름에는 ``_u2``가 있어야 하고 v1에는 없어야 한다. ``--security-boundaries``는 라벨을
종목 구간 안에서만 계산하고 지평이 구간 끝을 넘는 행은 뺀다 — v2 패널에서만, ``_fwd`` 이름 불가
(``build_features.py`` docstring 참고). 같은 이름이 이미 있으면 쓰기를 거부한다.

**SPY 벤치마크 비교는 h21에서만 낸다.** ``benchmark.spy_monthly_return``이
``labels.HORIZON_TRADING_DAYS``(h21)를 그대로 쓰기 때문에(SPY 쪽은 이번
파라미터화 대상이 아니다 — ``05`` M3 검정 범위 밖), h5·h63 라벨과 그대로
짝지으면 만기가 다른 두 수익률을 빼는 게 된다. 그래서 h21이 아니면 SPY
비교 필드를 아예 안 내고 그 사실을 manifest에 남긴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modeler.etl.config import DataRoot
from modeler.us.benchmark import ew_minus_spy_monthly, spy_total_return_daily
from modeler.us.build_panel import universe_filter
from modeler.us.cost import DEFAULT_K, DEFAULT_Q_DOLLAR, cost_grid, daily_volatility
from modeler.us.dataset import (
    assert_dataset_absent,
    check_dataset_name,
    git_commit,
    manifest_universe_version,
    read_panel_dataset,
    write_dataset,
)
from modeler.us.labels import HORIZON_TRADING_DAYS, build_labels
from modeler.us.lake import UsLake
from modeler.us.panel import UNIVERSE_VERSIONS, build_panel, universe_manifest

DATASET_NAME = "us_labels_v1"


def _dataset_name(horizon: int) -> str:
    """horizon에 따른 기본 데이터셋 이름. h21은 지금 이름 그대로 둔다."""
    if horizon == HORIZON_TRADING_DAYS:
        return DATASET_NAME
    return f"us_labels_h{horizon}_v1"



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON_TRADING_DAYS,
        help=f"라벨 만기까지 거래일 수 (기본: {HORIZON_TRADING_DAYS}, h21)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="데이터셋 이름 (v1 기본: h21이면 us_labels_v1, 아니면 us_labels_h{horizon}_v1. "
        "v2는 _u2가 든 이름을 직접 준다)",
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
        help="라벨을 종목 구간(security_id) 안에서만 계산한다. v2 패널에서만, _fwd 이름 불가.",
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
    dataset_name = args.name or (
        _dataset_name(args.horizon) if universe_version == "v1" else None
    )
    if dataset_name is None:
        parser.error("유니버스 v2는 --name(_u2가 든 이름)이 필요합니다.")
    check_dataset_name(
        dataset_name,
        universe_version=universe_version,
        security_boundaries=args.security_boundaries,
    )
    assert_dataset_absent(root, dataset_name)
    if not args.panel_name:
        panel = build_panel(lake, universe_version=universe_version)

    label_lake = lake.with_security_boundaries() if args.security_boundaries else lake
    labels, diagnostics = build_labels(label_lake, panel=panel, horizon=args.horizon)

    sigma = daily_volatility(label_lake).select("date", "symbol", "sigma_daily").collect()
    with_sigma = labels.join(sigma, on=["date", "symbol"], how="left")
    grid = cost_grid(with_sigma)

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
        "horizon_trading_days": args.horizon,
        "labels_start": str(labels["date"].min()) if labels.height else None,
        "labels_end": str(labels["date"].max()) if labels.height else None,
        "rebalance_dates_total": diagnostics["rebalance_dates_total"],
        "rebalance_dates_usable": diagnostics["rebalance_dates_usable"],
        "dropped_rebalance_dates": diagnostics["dropped_rebalance_dates"],
        "closed_by_reason": diagnostics["closed_by_reason"],
        "cost_grid_default": {"q_dollar": DEFAULT_Q_DOLLAR, "k": DEFAULT_K},
        "cost_grid": grid.to_dicts(),
    }

    if args.horizon == HORIZON_TRADING_DAYS:
        ew_spy = ew_minus_spy_monthly(lake, labels)
        spy_tr = spy_total_return_daily(lake)
        spy_last = spy_tr.sort("date").tail(1)
        manifest.update(
            {
                "spy_last_date": str(spy_last["date"].item()),
                "spy_last_close": spy_last["close"].item(),
                "spy_last_tr_adj": spy_last["tr_adj"].item(),
                "ew_minus_spy_mean_monthly": ew_spy["ew_minus_spy"].mean(),
            }
        )
    else:
        manifest["ew_minus_spy_mean_monthly"] = None
        manifest["ew_minus_spy_note"] = (
            "benchmark.spy_monthly_return이 HORIZON_TRADING_DAYS(h21)를 고정으로 써서 "
            f"horizon={args.horizon} 라벨과 만기가 안 맞는다 — 여기서는 계산하지 않는다 "
            "(build_features.py 작업 보고 참고, 05 M3 범위 밖)."
        )

    dataset_dir = write_dataset(labels, root, dataset_name, manifest=manifest)
    print(f"{dataset_dir}  {labels.height:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
