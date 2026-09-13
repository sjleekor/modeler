"""k=100 buy-list cost check — the open condition left by the Grade A gate.

``07_phase1_acceptance_gate.md`` §6 adopted ``px_reversal_5d``, ``px_maxret_20d``,
``px_idio_vol_60d`` and ``flow_individual_netbuy_to_volume_{5,20}d``
*conditionally*: incrementality (⑥) was clear, but economics (⑤) had only been
measured on the top decile — roughly 260 names, which turns over far more gently
than the k=100 list ``predict.py`` actually trades. The condition is whether the
improvement survives on that real list, net of its turnover.

This re-derives the same walk-forward predictions the gate used and runs them
through *both* reports, so the decile numbers act as a reproduction control for
the k=100 numbers sitting next to them.

What it deliberately does not do:

* no hyperparameter search — Phase 1's selected params are reused, exactly as
  ``run_holdout`` does;
* no dataset rebuild — it reads the panel already on disk;
* no holdout access — only ``fold_role in (train, valid)`` rows are used, so
  fold 6 (2026-06-11 ~ 2026-07-31) is never touched.

    uv run python -m modeler.models._01_20_access_return_rank.experiments.run_topk_cost_check
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import time
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.metrics import economic_report, topk_economic_report
from modeler.models._01_20_access_return_rank import train as tr
from modeler.models._01_20_access_return_rank.experiments.run_grade_a_acceptance_gate import (
    CONFIGS,
    DOCS_DIR,
    SNAPSHOT_DATE,
    SOURCE,
    TARGET_HORIZONS,
    WALK_FORWARD_PERIOD_END,
    _base_spec,
    _config_dataset_dir,
    _load_phase1_best_params,
    _wf_predictions,
)
from modeler.models._01_20_access_return_rank.experiments.run_grade_a_acceptance_gate import (
    RESULTS_JSON as GATE_RESULTS_JSON,
)

RESULTS_JSON = DOCS_DIR / "topk_cost_check.json"
RESULTS_MD = DOCS_DIR / "topk_cost_check.md"
DEFAULT_K = 100
DEFAULT_COST_BPS = 60.0


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def _std_panel_path(cfg: LakeConfig, name: str, model_id: str) -> Path:
    """Where ``_build_one`` would have written this config's standardized panel."""
    cfg_run = dataclasses.replace(cfg, datasets_root_override=_config_dataset_dir(cfg, name))
    return cfg_run.dataset_dir(model_id) / "feat_panel_std.parquet"


def _gate_decile_spread() -> dict[tuple[str, int], float]:
    """Phase 1's recorded cost-adjusted decile spread, used as a reproduction check."""
    if not GATE_RESULTS_JSON.exists():
        return {}
    payload = json.loads(GATE_RESULTS_JSON.read_text())
    return {
        (r["name"], r["horizon"]): r["economic"]["cost_adjusted_spread"]
        for r in payload["records"]
    }


def run(
    *,
    k: int,
    cost_bps: float,
    horizons: tuple[int, ...],
    cfg: LakeConfig | None = None,
) -> list[dict]:
    cfg = cfg or LakeConfig(
        root=DataRoot.resolve(market="kr"), snapshot_date=SNAPSHOT_DATE, source=SOURCE
    )
    spec = _base_spec(smoke=False, period_end=WALK_FORWARD_PERIOD_END, holdout_len=0)
    best_params = _load_phase1_best_params()
    gate_spread = _gate_decile_spread()

    records: list[dict] = []
    for name, _cols in CONFIGS:
        path = _std_panel_path(cfg, name, spec.model_id)
        if not path.is_file():
            raise SystemExit(
                f"{path} not found — the Grade A gate's dataset for config={name} is gone. "
                "Re-run run_grade_a_acceptance_gate to rebuild it."
            )
        print(f"\n=== config={name} === reading {path}", flush=True)
        std = pl.read_parquet(path)
        for h in horizons:
            config = tr.TrainConfig(
                model=spec.model_type, target=f"y_rank_{h}d", realized=f"raw_label_{h}d"
            )
            result = tr.TrainResult(
                config=config,
                fold_results=[],
                best_params=best_params[(name, h)],
                mean_rank_ic=float("nan"),  # reused from Phase 1, not re-selected here
                feature_cols=tr.design_columns(std, config.target),
            )
            t0 = time.time()
            preds = _wf_predictions(std, config, result)
            fit_s = time.time() - t0
            decile = economic_report(
                preds,
                pred_col="pred",
                realized_col=config.realized,
                horizon=h,
                cost_bps_roundtrip=cost_bps,
            )
            topk = topk_economic_report(
                preds,
                pred_col="pred",
                realized_col=config.realized,
                horizon=h,
                k=k,
                cost_bps_roundtrip=cost_bps,
            )
            records.append(
                {
                    "name": name,
                    "horizon": h,
                    "best_params": result.best_params,
                    "decile": decile.as_dict(),
                    "topk": topk.as_dict(),
                    "gate_decile_cost_adjusted_spread": gate_spread.get((name, h)),
                    "fit_seconds": round(fit_s, 1),
                }
            )
            print(
                f"  h={h}d: decile net={_fmt(decile.cost_adjusted_spread)} "
                f"(gate {_fmt(gate_spread.get((name, h)))}), "
                f"k={k} net={_fmt(topk.cost_adjusted_return)} "
                f"turnover={_fmt(topk.turnover, 3)} ({fit_s:.0f}s)",
                flush=True,
            )
        del std
        gc.collect()
    return records


def render_markdown(records: list[dict], *, k: int, cost_bps: float) -> str:
    by_key = {(r["name"], r["horizon"]): r for r in records}
    horizons = sorted({r["horizon"] for r in records})

    lines = [
        f"# k={k} 매수 리스트 비용 확인 (acceptance gate §6 잔여 조건)",
        "",
        f"> walk-forward 5-fold, {WALK_FORWARD_PERIOD_END}까지. holdout(fold 6)은 읽지 않는다.",
        f"> 왕복 거래비용 {cost_bps:.0f}bp 가정, decile은 대조군.",
        "",
        "## 재현 확인 — decile 경로가 게이트 값을 재현하는가",
        "",
        "| config | h | 이번 decile net | 게이트 decile net | 차이 |",
        "|---|---|---|---|---|",
    ]
    for h in horizons:
        for name, _cols in CONFIGS:
            rec = by_key.get((name, h))
            if not rec:
                continue
            now = rec["decile"]["cost_adjusted_spread"]
            gate = rec["gate_decile_cost_adjusted_spread"]
            diff = None if gate is None else now - gate
            lines.append(
                f"| {name} | {h}d | {_fmt(now)} | {_fmt(gate)} | {_fmt(diff)} |"
            )

    lines += [
        "",
        f"## k={k} 리스트",
        "",
        "| config | h | 보유 종목 | 평균수익 | turnover | 비용차감 후 |",
        "|---|---|---|---|---|---|",
    ]
    for h in horizons:
        for name, _cols in CONFIGS:
            rec = by_key.get((name, h))
            if not rec:
                continue
            t = rec["topk"]
            lines.append(
                f"| {name} | {h}d | {_fmt(t['mean_names_held'], 1)} | "
                f"{_fmt(t['grid_topk_mean_return'])} | {_fmt(t['turnover'], 3)} | "
                f"{_fmt(t['cost_adjusted_return'])} |"
            )

    lines += [
        "",
        "## 판정 — candidate - baseline",
        "",
        f"| h | Δ decile net | Δ k={k} net | k={k}에서 개선 유지 |",
        "|---|---|---|---|",
    ]
    verdicts = []
    for h in horizons:
        base = by_key.get(("baseline", h))
        cand = by_key.get(("candidate", h))
        if not base or not cand:
            continue
        d_dec = cand["decile"]["cost_adjusted_spread"] - base["decile"]["cost_adjusted_spread"]
        d_topk = cand["topk"]["cost_adjusted_return"] - base["topk"]["cost_adjusted_return"]
        held = d_topk > 0
        verdicts.append(held)
        lines.append(f"| {h}d | {_fmt(d_dec)} | {_fmt(d_topk)} | {'예' if held else '아니오'} |")

    if verdicts:
        n_held = sum(verdicts)
        lines += [
            "",
            f"{len(verdicts)}개 horizon 중 {n_held}개에서 k={k} 비용차감 후 candidate가 "
            "baseline보다 높다.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(TARGET_HORIZONS),
        help="subset of horizons to run (default: all three)",
    )
    args = parser.parse_args(argv)

    records = run(k=args.k, cost_bps=args.cost_bps, horizons=tuple(args.horizons))
    markdown = render_markdown(records, k=args.k, cost_bps=args.cost_bps)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {"k": args.k, "cost_bps_roundtrip": args.cost_bps, "records": records},
            indent=2,
            default=str,
        )
    )
    RESULTS_MD.write_text(markdown)
    print("\n" + markdown)
    print(f"wrote {RESULTS_JSON} and {RESULTS_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
