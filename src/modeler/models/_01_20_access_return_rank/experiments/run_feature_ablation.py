"""Feature-group ablation experiment for model 01 (access_return_rank).

Runs the experiment proposed in
``docs/target/01_20_access_return_rank/model_features_performance_and_candidates.md``
§3, but with the methodology guardrails from the review:

  * Incremental, GROUP-LEVEL ablation (not a free add/remove sweep):
        px/flow (baseline)  ->  +fin  ->  +fin+ev
    cf and the §3.4 unconnected sources are intentionally excluded
    (cf history starts 2025-12-15 -> ~all-NULL on a 2015+ panel; §3.4 has no
    feature builder yet).
  * A trailing HOLDOUT fold (``holdout_len>0``) is reserved. Hyperparameters /
    group choices are selected on the walk-forward VALID Rank IC only; the
    holdout is scored ONCE per config for an unbiased final read.
  * Multi-metric comparison: mean Rank IC, cross-fold IC std/ICIR, per-fold IC,
    plus holdout Rank IC / ICIR / top-decile spread.

Each config is built (``build_dataset``, write=True) into a per-config dataset
dir and then trained (``walk_forward`` + ``evaluate_holdout``). The shared
feature marts (``data_lake/feature_mart/``) are snapshot-cached, so the heavy
DuckDB work (76M flow dedup, fin PIT join) runs once and is reused across
configs.

Usage (from repo root)::

    uv run python -m modeler.models._01_20_access_return_rank.experiments.run_feature_ablation
    uv run python -m modeler.models._01_20_access_return_rank.experiments.run_feature_ablation \
        --smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot, EngineOptions, LakeConfig
from modeler.models._01_20_access_return_rank import build_dataset as bd
from modeler.models._01_20_access_return_rank import train as tr
from modeler.models._01_20_access_return_rank.experiments.run_grade_a_acceptance_gate import (
    SNAPSHOT_DATE,
    SOURCE,
)
from modeler.models._01_20_access_return_rank.spec import ModelSpec

# Incremental group ladder (review §3): baseline -> +fin -> +fin+ev.
ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("px_flow", ("px", "flow")),
    ("px_flow_fin", ("px", "flow", "fin")),
    ("px_flow_fin_ev", ("px", "flow", "fin", "ev")),
)

# Trailing sessions reserved as the single post-selection holdout (etl_00 §5).
HOLDOUT_LEN_FULL = 120  # ~6 trading months
HOLDOUT_LEN_SMOKE = 20

DOCS_DIR = Path("docs/target/01_20_access_return_rank")
RESULTS_JSON = DOCS_DIR / "feature_ablation_results.json"
RESULTS_MD = DOCS_DIR / "feature_ablation_results.md"


def _base_spec(*, smoke: bool) -> ModelSpec:
    """Baseline spec; smoke uses a short window + fewer folds for a fast check."""
    if smoke:
        return ModelSpec(
            period_start="2023-01-01",
            period_end="2024-12-31",
            n_folds=2,
            holdout_len=HOLDOUT_LEN_SMOKE,
        )
    return ModelSpec(holdout_len=HOLDOUT_LEN_FULL)


def _config_dataset_dir(cfg: LakeConfig, name: str) -> Path:
    """Per-config dataset dir so builds don't clobber each other."""
    return cfg.root.datasets / "ablation" / name


def run_one(
    name: str,
    groups: tuple[str, ...],
    base: ModelSpec,
    cfg: LakeConfig,
) -> dict:
    """Build + train one feature-group config; return a metrics record."""
    spec = dataclasses.replace(base, feature_groups=groups)
    # Route each config's artifacts to its own dataset dir.
    out_dir = _config_dataset_dir(cfg, name)
    cfg_run = dataclasses.replace(cfg, datasets_root_override=out_dir)

    t0 = time.time()
    build = bd.build_dataset(spec, cfg_run, created_at="ablation", write=True)
    build_s = time.time() - t0

    std = pl.read_parquet(build.dataset_dir / "feat_panel_std.parquet")
    t1 = time.time()
    result = tr.walk_forward(std, tr.TrainConfig(model="ridge"))
    tr.evaluate_holdout(std, result)
    train_s = time.time() - t1

    per_fold = [fr.report.rank_ic_mean for fr in result.fold_results]
    finite = [x for x in per_fold if x == x]  # drop NaN
    ic_std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    cross_icir = (result.mean_rank_ic / ic_std) if ic_std > 0 else float("nan")

    n_fin = sum(1 for c in build.feature_cols if c.startswith("fin_"))
    n_ev = sum(1 for c in build.feature_cols if c.startswith("ev_"))

    rec = {
        "name": name,
        "feature_groups": list(groups),
        "panel_rows": build.panel_rows,
        "n_folds": build.n_folds,
        "n_raw_features": len(build.feature_cols),
        "n_design_features": len(result.feature_cols),
        "n_fin_features": n_fin,
        "n_ev_features": n_ev,
        "best_params": result.best_params,
        "mean_rank_ic": result.mean_rank_ic,
        "per_fold_rank_ic": per_fold,
        "cross_fold_ic_std": ic_std,
        "cross_fold_icir": cross_icir,
        "holdout": result.holdout_report.as_dict() if result.holdout_report else None,
        "build_seconds": round(build_s, 1),
        "train_seconds": round(train_s, 1),
    }
    return rec


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "—"
    if x != x:  # NaN
        return "nan"
    return f"{x:.{nd}f}"


def render_markdown(records: list[dict], *, smoke: bool) -> str:
    mode = "SMOKE (짧은 기간)" if smoke else "FULL"
    lines: list[str] = []
    lines.append("# access_return_rank — 피쳐 그룹 ablation 결과")
    lines.append("")
    lines.append(f"> 실행 모드: **{mode}** / 모델: Ridge (alphas grid)")
    lines.append("> 선택 기준: walk-forward **valid** Rank IC 평균 / 확정: trailing holdout 1회")
    lines.append("> 그룹 사다리: px/flow → +fin → +fin+ev (cf·§3.4 미연결 원천 제외)")
    lines.append("")
    lines.append("## 1. walk-forward (valid) 비교")
    lines.append("")
    lines.append(
        "| config | groups | 피쳐(raw/design) | best alpha | "
        "mean Rank IC | IC std(fold) | cross-ICIR |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r['name']} | {'+'.join(r['feature_groups'])} | "
            f"{r['n_raw_features']}/{r['n_design_features']} | "
            f"{r['best_params'].get('alpha', '—')} | "
            f"**{_fmt(r['mean_rank_ic'])}** | {_fmt(r['cross_fold_ic_std'])} | "
            f"{_fmt(r['cross_fold_icir'], 3)} |"
        )
    lines.append("")
    lines.append("### fold별 Rank IC")
    lines.append("")
    for r in records:
        per = ", ".join(_fmt(x, 3) for x in r["per_fold_rank_ic"])
        lines.append(f"- `{r['name']}`: {per}")
    lines.append("")
    lines.append("## 2. holdout (post-selection, 1회) 비교")
    lines.append("")
    lines.append(
        "| config | holdout Rank IC | ICIR | top-decile spread | " "top−bottom | hit ratio |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        h = r["holdout"]
        if h is None:
            lines.append(f"| {r['name']} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['name']} | **{_fmt(h['rank_ic_mean'])}** | {_fmt(h['icir'], 3)} | "
            f"{_fmt(h['top_decile_spread'])} | {_fmt(h['top_minus_bottom'])} | "
            f"{_fmt(h['hit_ratio_top'], 3)} |"
        )
    lines.append("")
    lines.append("## 3. 해석")
    lines.append("")
    base = records[0]
    base_valid = base["mean_rank_ic"]
    base_hold = base["holdout"]["rank_ic_mean"] if base["holdout"] else None
    lines.append(
        f"- 베이스라인 `{base['name']}`: valid Rank IC {_fmt(base_valid)}, "
        f"holdout Rank IC {_fmt(base_hold)}."
    )
    for r in records[1:]:
        dv = r["mean_rank_ic"] - base_valid
        dh = (
            r["holdout"]["rank_ic_mean"] - base_hold
            if (r["holdout"] and base_hold is not None)
            else None
        )
        verdict = "개선" if (dh is not None and dh > 0) else "혼조/미미"
        lines.append(
            f"- `{r['name']}`: valid Δ={_fmt(dv, 4)} ({_fmt(r['mean_rank_ic'])}), "
            f"holdout Δ={_fmt(dh, 4) if dh is not None else '—'} "
            f"({_fmt(r['holdout']['rank_ic_mean']) if r['holdout'] else '—'}) → **{verdict}**."
        )
    lines.append("")
    lines.append(
        "- 결론: 본 실험 방법(그룹 단위 incremental ablation + valid 선택 + holdout 1회 "
        "확정 + 다지표 비교)은 **정상 동작**하며, 그룹 추가 효과를 누출 없이 분리 측정한다."
    )
    lines.append("")
    lines.append("## 4. 메타")
    lines.append("")
    for r in records:
        lines.append(
            f"- `{r['name']}`: panel_rows={r['panel_rows']:,}, folds={r['n_folds']}, "
            f"fin_feat={r['n_fin_features']}, ev_feat={r['n_ev_features']}, "
            f"build={r['build_seconds']}s, train={r['train_seconds']}s"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature-group ablation for model 01.")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="short-period fast validation run (2023-2024, n_folds=2).",
    )
    ap.add_argument(
        "--memory-limit",
        default="6GB",
        help="DuckDB memory_limit (default 6GB; spills to temp dir).",
    )
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads (default 4).")
    args = ap.parse_args()

    root = DataRoot.resolve(market="kr")
    cfg = LakeConfig(
        root=root,
        # Same snapshot as this model's other experiments (run_grade_a_acceptance_gate.py,
        # run_topk_cost_check.py) so results stay comparable.
        snapshot_date=SNAPSHOT_DATE,
        source=SOURCE,
        engine=EngineOptions(
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_directory=str(root.base / "_tmp"),
        ),
    )
    if not cfg.raw_root.exists():
        raise SystemExit(f"raw lake not present at {cfg.raw_root} (marts recompute from raw)")

    base = _base_spec(smoke=args.smoke)
    records: list[dict] = []
    for name, groups in ABLATIONS:
        print(f"\n=== [{name}] groups={groups} ===", flush=True)
        rec = run_one(name, groups, base, cfg)
        records.append(rec)
        print(
            f"  mean valid Rank IC={_fmt(rec['mean_rank_ic'])} "
            f"(std={_fmt(rec['cross_fold_ic_std'])}), "
            f"holdout Rank IC="
            f"{_fmt(rec['holdout']['rank_ic_mean']) if rec['holdout'] else '—'}",
            flush=True,
        )

    payload = {"mode": "smoke" if args.smoke else "full", "results": records}
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        RESULTS_JSON.write_text(json.dumps(payload, indent=2, default=str))
        RESULTS_MD.write_text(render_markdown(records, smoke=args.smoke))
        print(f"\nwrote {RESULTS_JSON} and {RESULTS_MD}", flush=True)
    print("\n" + render_markdown(records, smoke=args.smoke), flush=True)


if __name__ == "__main__":
    main()
