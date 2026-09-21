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

**SPY 벤치마크 비교는 h21에서만 낸다.** ``benchmark.spy_monthly_return``이
``labels.HORIZON_TRADING_DAYS``(h21)를 그대로 쓰기 때문에(SPY 쪽은 이번
파라미터화 대상이 아니다 — ``05`` M3 검정 범위 밖), h5·h63 라벨과 그대로
짝지으면 만기가 다른 두 수익률을 빼는 게 된다. 그래서 h21이 아니면 SPY
비교 필드를 아예 안 내고 그 사실을 manifest에 남긴다.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from modeler.etl.config import DataRoot
from modeler.us.benchmark import ew_minus_spy_monthly, spy_total_return_daily
from modeler.us.cost import DEFAULT_K, DEFAULT_Q_DOLLAR, cost_grid, daily_volatility
from modeler.us.dataset import write_dataset
from modeler.us.labels import HORIZON_TRADING_DAYS, build_labels
from modeler.us.lake import UsLake

DATASET_NAME = "us_labels_v1"


def _dataset_name(horizon: int) -> str:
    """horizon에 따른 기본 데이터셋 이름. h21은 지금 이름 그대로 둔다."""
    if horizon == HORIZON_TRADING_DAYS:
        return DATASET_NAME
    return f"us_labels_h{horizon}_v1"


def _git_commit(repo: Path) -> str:
    """``repo``의 HEAD 커밋. 트리가 더러우면 ``-dirty``를 붙인다 (``build_panel.py``와 같다)."""
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
        "--horizon",
        type=int,
        default=HORIZON_TRADING_DAYS,
        help=f"라벨 만기까지 거래일 수 (기본: {HORIZON_TRADING_DAYS}, h21)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="데이터셋 이름 (기본: h21이면 us_labels_v1, 아니면 us_labels_h{horizon}_v1)",
    )
    args = parser.parse_args(argv)
    dataset_name = args.name or _dataset_name(args.horizon)

    lake = UsLake.resolve()
    labels, diagnostics = build_labels(lake, horizon=args.horizon)

    sigma = daily_volatility(lake).select("date", "symbol", "sigma_daily").collect()
    with_sigma = labels.join(sigma, on=["date", "symbol"], how="left")
    grid = cost_grid(with_sigma)

    modeler_repo = Path(__file__).resolve().parents[3]
    collector_repo = modeler_repo.parent / "collector"

    manifest = {
        "input_table_snapshots": lake.snapshot_manifest(),
        "modeler_git_commit": _git_commit(modeler_repo),
        "collector_git_commit": _git_commit(collector_repo),
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

    dataset_dir = write_dataset(
        labels, DataRoot.resolve(market="us"), dataset_name, manifest=manifest
    )
    print(f"{dataset_dir}  {labels.height:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
