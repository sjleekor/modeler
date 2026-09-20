"""``us_labels_v1`` 데이터셋을 만든다.

    uv run python -m modeler.us.build_labels

``build_panel.py``와 같은 꼴이다 — 데이터셋은 지우면 다시 만들 수 있어야 하고,
manifest의 ``modeler_git_commit``이 가리키는 코드로 이 명령을 돌리면 같은
``content_hash``가 나오는 것이 재현의 뜻이다.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from modeler.etl.config import DataRoot
from modeler.us.benchmark import ew_minus_spy_monthly, spy_total_return_daily
from modeler.us.cost import DEFAULT_K, DEFAULT_Q_DOLLAR, cost_grid, daily_volatility
from modeler.us.dataset import write_dataset
from modeler.us.labels import build_labels
from modeler.us.lake import UsLake

DATASET_NAME = "us_labels_v1"


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
        "--name", default=DATASET_NAME, help=f"데이터셋 이름 (기본: {DATASET_NAME})"
    )
    args = parser.parse_args(argv)

    lake = UsLake.resolve()
    labels, diagnostics = build_labels(lake)

    sigma = daily_volatility(lake).select("date", "symbol", "sigma_daily").collect()
    with_sigma = labels.join(sigma, on=["date", "symbol"], how="left")
    grid = cost_grid(with_sigma)

    ew_spy = ew_minus_spy_monthly(lake, labels)
    spy_tr = spy_total_return_daily(lake)
    spy_last = spy_tr.sort("date").tail(1)

    modeler_repo = Path(__file__).resolve().parents[3]
    collector_repo = modeler_repo.parent / "collector"

    manifest = {
        "input_table_snapshots": lake.snapshot_manifest(),
        "modeler_git_commit": _git_commit(modeler_repo),
        "collector_git_commit": _git_commit(collector_repo),
        "horizon_trading_days": 21,
        "labels_start": str(labels["date"].min()),
        "labels_end": str(labels["date"].max()),
        "rebalance_dates_total": diagnostics["rebalance_dates_total"],
        "rebalance_dates_usable": diagnostics["rebalance_dates_usable"],
        "dropped_rebalance_dates": diagnostics["dropped_rebalance_dates"],
        "closed_by_reason": diagnostics["closed_by_reason"],
        "cost_grid_default": {"q_dollar": DEFAULT_Q_DOLLAR, "k": DEFAULT_K},
        "cost_grid": grid.to_dicts(),
        "spy_last_date": str(spy_last["date"].item()),
        "spy_last_close": spy_last["close"].item(),
        "spy_last_tr_adj": spy_last["tr_adj"].item(),
        "ew_minus_spy_mean_monthly": ew_spy["ew_minus_spy"].mean(),
    }
    dataset_dir = write_dataset(labels, DataRoot.resolve(market="us"), args.name, manifest=manifest)
    print(f"{dataset_dir}  {labels.height:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
