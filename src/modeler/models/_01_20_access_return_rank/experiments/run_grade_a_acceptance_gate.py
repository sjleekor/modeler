"""Phase 1 acceptance-gate experiment for the Grade A raw-feature candidates.

Closes the acceptance-gate criteria that Phase A horizon-scan screening
explicitly deferred (``docs/dev/20260731_raw_features/01_feature_candidate/
02_feature_candidate.md`` §6.1, ``04_specific_plan_A.md`` §11): ⑤ economic
significance (decile spread, turnover, cost-adjusted spread) and ⑥ incremental
value over the existing baseline model. ⑧ (single holdout look) is closed
separately, gated behind ``--confirm-holdout``.

Two of the six Grade A families are already part of the existing baseline
feature set: ``px_amihud_20d`` verbatim, and ``px_near_52w_high`` is the same
formula as the existing ``px_dist_52w_high`` (see
``docs/dev/20260731_raw_features/01_feature_candidate/02_feature_candidate.md``
§4). Only the four genuinely new columns are tested here: ``px_reversal_5d``,
``px_maxret_20d``, ``px_idio_vol_60d``, and
``flow_individual_netbuy_to_volume_{5,20}d`` (the officially registered
primary+secondary variants; ``_60d`` was not part of Phase A's registry).

Two phases, run separately:

  * Phase 1 (default, safe to re-run freely): walk-forward VALID-only
    baseline-vs-candidate comparison across the model's 3 declared target
    horizons (5/20/60d) — no holdout data is touched.
  * Phase 2 (``--confirm-holdout``, ONE-SHOT): the model's existing trailing
    holdout (ending 2026-06-10) has already been opened twice for unrelated
    decisions (fin/ev group choice, Ridge->HGB model choice — see
    ``docs/target/01_20_access_return_rank/feature_ablation_results.md``,
    ``improvement_results.md``), so re-using it a third time here would look
    at already-spent data. Phase 2 instead opens a FRESH window
    (2026-06-11..2026-07-31) that has never been used for any decision on this
    model, exactly once. Refuses to overwrite an existing result unless
    ``--force`` is passed.

Usage (from repo root)::

    uv run python -m modeler.models._01_20_access_return_rank.experiments.\
run_grade_a_acceptance_gate
    uv run python -m modeler.models._01_20_access_return_rank.experiments.\
run_grade_a_acceptance_gate --smoke
    uv run python -m modeler.models._01_20_access_return_rank.experiments.\
run_grade_a_acceptance_gate --confirm-holdout
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import duckdb
import polars as pl

from modeler.analysis.horizon_scan_config import load_config
from modeler.etl.config import DataRoot, EngineOptions, LakeConfig
from modeler.etl.metrics import economic_report
from modeler.models._01_20_access_return_rank import build_dataset as bd
from modeler.models._01_20_access_return_rank import train as tr
from modeler.models._01_20_access_return_rank.spec import ModelSpec

# --- baseline (pre-Phase-A0) feature columns --------------------------------
# Verbatim from docs/target/01_20_access_return_rank/model_features_performance_
# and_candidates.md §1.1/§1.2/§3.1 — the columns that existed before the
# 20260731 raw-feature research track added anything to feat_price/feat_flow.
BASELINE_PX_COLS: tuple[str, ...] = (
    "px_ret_1d",
    "px_ret_5d",
    "px_ret_20d",
    "px_ret_60d",
    "px_mom_20_60",
    "px_vol_20d",
    "px_vol_60d",
    "px_high_low_range_20d",
    "px_turnover",
    "px_turnover_ma20",
    "px_amihud_20d",
    "px_gap_vs_ma20",
    "px_dist_52w_high",
    "px_is_halted",
    "px_halt_ratio_20d",
)
BASELINE_FLOW_COLS: tuple[str, ...] = (
    "flow_foreign_netbuy_sum_5d",
    "flow_foreign_netbuy_sum_20d",
    "flow_inst_netbuy_sum_5d",
    "flow_inst_netbuy_sum_20d",
    "flow_indiv_netbuy_sum_5d",
    "flow_indiv_netbuy_sum_20d",
    "flow_foreign_holding_chg_5d",
    "flow_foreign_holding_chg_20d",
    "flow_short_balance_chg_20d",
    "flow_foreign_netbuy_z_20d",
    "flow_inst_netbuy_z_20d",
    "flow_short_avg_price",
    "flow_short_selling_volume",
    "flow_short_selling_value",
    "flow_short_balance_qty",
)
BASELINE_FIN_COLS: tuple[str, ...] = (
    "fin_roa",
    "fin_roe",
    "fin_operating_margin",
    "fin_debt_to_equity",
    "fin_equity_ratio",
    "fin_ocf_to_assets",
    "fin_cash_ratio",
    "fin_asset_turnover",
    "fin_is_negative_equity",
    "fin_has_fs",
)
BASELINE_COLS: tuple[str, ...] = (*BASELINE_PX_COLS, *BASELINE_FLOW_COLS, *BASELINE_FIN_COLS)

# The 4 net-new Grade A columns under test (px_amihud_20d / px_dist_52w_high
# are already in BASELINE_COLS — see module docstring).
NEW_CANDIDATE_COLS: tuple[str, ...] = (
    "px_reversal_5d",
    "px_maxret_20d",
    "px_idio_vol_60d",
    "flow_individual_netbuy_to_volume_5d",
    "flow_individual_netbuy_to_volume_20d",
)
CANDIDATE_COLS: tuple[str, ...] = (*BASELINE_COLS, *NEW_CANDIDATE_COLS)

CONFIGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", BASELINE_COLS),
    ("candidate", CANDIDATE_COLS),
)

TARGET_HORIZONS: tuple[int, ...] = (5, 20, 60)  # 20d is the model's deployed target

SNAPSHOT_DATE = "2026-08-01"  # Phase A's snapshot; feat_price/feat_flow already
SOURCE = "sj2_remote"  # materialize every Grade A column here.

# Trailing window already spent on 2 prior decisions (fin/ev choice; Ridge->HGB
# choice) — see docs/dev/20260731_raw_features/01_feature_candidate/
# 07_phase1_acceptance_gate.md §2. Phase 1 stops before this window; Phase 2
# opens the untouched window right after it, once.
SPENT_HOLDOUT_START = "2025-12-01"  # approximate; not used directly, documentation only
WALK_FORWARD_PERIOD_END = "2026-06-10"
FRESH_HOLDOUT_START = "2026-06-11"
FRESH_HOLDOUT_END = "2026-07-31"

DOCS_DIR = Path("docs/target/01_20_access_return_rank")
RESULTS_JSON = DOCS_DIR / "grade_a_acceptance_gate_results.json"
RESULTS_MD = DOCS_DIR / "grade_a_acceptance_gate_results.md"
HOLDOUT_JSON = DOCS_DIR / "grade_a_acceptance_gate_holdout.json"
HOLDOUT_MD = DOCS_DIR / "grade_a_acceptance_gate_holdout.md"


def _git_commit() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=repo_root
        ).strip()
    except Exception:  # pragma: no cover - best-effort provenance only
        return "unknown"


def _config_dataset_dir(cfg: LakeConfig, name: str) -> Path:
    """Per-config dataset dir so baseline/candidate builds don't clobber each other."""
    return cfg.root.datasets / "grade_a_gate" / name


def _base_spec(*, smoke: bool, period_end: str, holdout_len: int) -> ModelSpec:
    if smoke:
        return ModelSpec(
            feature_groups=("px", "flow", "fin"),
            period_start="2023-01-01",
            period_end="2024-12-31",
            n_folds=2,
            holdout_len=0,
        )
    return ModelSpec(
        feature_groups=("px", "flow", "fin"),
        period_end=period_end,
        holdout_len=holdout_len,
    )


def _build_one(
    name: str, cols: tuple[str, ...], spec: ModelSpec, cfg: LakeConfig
) -> bd.BuildResult:
    out_dir = _config_dataset_dir(cfg, name)
    cfg_run = dataclasses.replace(cfg, datasets_root_override=out_dir)
    return bd.build_dataset(
        spec,
        cfg_run,
        created_at="grade_a_acceptance_gate",
        write=True,
        reuse_existing_marts=True,
        feature_cols_override=list(cols),
    )


def _wf_predictions(
    std: pl.DataFrame, config: tr.TrainConfig, result: tr.TrainResult
) -> pl.DataFrame:
    """Re-derive per-row walk-forward predictions from the already-selected params.

    ``train.walk_forward`` only returns aggregated ``RankICReport``s per fold;
    ``economic_report`` needs the raw per-row (ticker, pred, realized) frame.
    Re-run ``train._fit_predict`` once per fold with the SAME selected
    ``best_params`` (no new hyperparameter search) and concatenate — fold valid
    slices tile a disjoint region of the trading calendar (``research/etl/
    splits.py``), so concatenation does not double-count any date.
    """
    wf = std.filter(pl.col("fold_role").is_in(["train", "valid"]))
    frames = []
    for fid in sorted(wf.get_column("fold_id").unique().to_list()):
        train_df = wf.filter((pl.col("fold_id") == fid) & (pl.col("fold_role") == "train"))
        valid_df = wf.filter((pl.col("fold_id") == fid) & (pl.col("fold_role") == "valid"))
        if train_df.height == 0 or valid_df.height == 0:
            continue
        frames.append(
            tr._fit_predict(train_df, valid_df, result.feature_cols, config, result.best_params)
        )
    return pl.concat(frames, how="vertical_relaxed") if frames else wf.clear()


def run_one_horizon(name: str, build: bd.BuildResult, horizon: int, model_type: str) -> dict:
    """Walk-forward train + economic report for one (config, horizon) pair."""
    std = pl.read_parquet(build.dataset_dir / "feat_panel_std.parquet")
    config = tr.TrainConfig(
        model=model_type, target=f"y_rank_{horizon}d", realized=f"raw_label_{horizon}d"
    )
    t0 = time.time()
    result = tr.walk_forward(std, config)
    train_s = time.time() - t0

    preds = _wf_predictions(std, config, result)
    econ = economic_report(preds, pred_col="pred", realized_col=config.realized, horizon=horizon)

    per_fold = [fr.report.rank_ic_mean for fr in result.fold_results]
    return {
        "name": name,
        "horizon": horizon,
        "n_raw_features": len(build.feature_cols),
        "n_design_features": len(result.feature_cols),
        "best_params": result.best_params,
        "mean_rank_ic": result.mean_rank_ic,
        "per_fold_rank_ic": per_fold,
        "economic": econ.as_dict(),
        "train_seconds": round(train_s, 1),
    }


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "—"
    if x != x:  # NaN
        return "nan"
    return f"{x:.{nd}f}"


def render_walkforward_markdown(records: list[dict], *, smoke: bool) -> str:
    mode = "SMOKE" if smoke else "FULL"
    lines: list[str] = []
    lines.append("# Grade A candidates — Phase 1 acceptance-gate walk-forward 결과")
    lines.append("")
    lines.append(f"> 실행 모드: **{mode}** / 모델: hgb")
    lines.append(
        "> baseline vs candidate(= baseline + px_reversal_5d/px_maxret_20d/px_idio_vol_60d/"
    )
    lines.append(
        "> flow_individual_netbuy_to_volume_5d/_20d) / holdout 미사용(walk-forward valid만)"
    )
    lines.append("")
    for h in TARGET_HORIZONS:
        rows = [r for r in records if r["horizon"] == h]
        if not rows:
            continue
        lines.append(f"## target = y_rank_{h}d")
        lines.append("")
        lines.append(
            "| config | 피쳐(raw/design) | mean Rank IC | 경제성(grid top-decile spread) | "
            "turnover | cost-adj spread |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            e = r["economic"]
            lines.append(
                f"| {r['name']} | {r['n_raw_features']}/{r['n_design_features']} | "
                f"**{_fmt(r['mean_rank_ic'])}** | {_fmt(e['grid_top_decile_spread'])} | "
                f"{_fmt(e['turnover'], 3)} | {_fmt(e['cost_adjusted_spread'])} |"
            )
        lines.append("")
        lines.append("fold별 Rank IC:")
        for r in rows:
            per = ", ".join(_fmt(x, 3) for x in r["per_fold_rank_ic"])
            lines.append(f"- `{r['name']}`: {per}")
        base = next((r for r in rows if r["name"] == "baseline"), None)
        cand = next((r for r in rows if r["name"] == "candidate"), None)
        if base and cand:
            d_ic = cand["mean_rank_ic"] - base["mean_rank_ic"]
            d_cost = (
                cand["economic"]["cost_adjusted_spread"] - base["economic"]["cost_adjusted_spread"]
            )
            lines.append("")
            lines.append(
                f"- Δ mean Rank IC = {_fmt(d_ic, 4)}, Δ cost-adjusted spread = {_fmt(d_cost, 4)}"
            )
        lines.append("")
    lines.append("## 메타")
    lines.append("")
    for r in records:
        lines.append(
            f"- `{r['name']}`/h={r['horizon']}d: design_feat={r['n_design_features']}, "
            f"train={r['train_seconds']}s, n_rebalances={r['economic']['n_rebalances']}"
        )
    lines.append("")
    return "\n".join(lines)


def run_walkforward(*, smoke: bool, cfg: LakeConfig) -> list[dict]:
    spec = _base_spec(smoke=smoke, period_end=WALK_FORWARD_PERIOD_END, holdout_len=0)
    records: list[dict] = []
    for name, cols in CONFIGS:
        print(f"\n=== [walk-forward] config={name} ===", flush=True)
        t0 = time.time()
        build = _build_one(name, cols, spec, cfg)
        build_s = time.time() - t0
        print(
            f"  build: {build.panel_rows:,} rows, {len(build.feature_cols)} raw features "
            f"({build_s:.1f}s)",
            flush=True,
        )
        for h in TARGET_HORIZONS:
            rec = run_one_horizon(name, build, h, spec.model_type)
            records.append(rec)
            print(
                f"  h={h}d: mean Rank IC={_fmt(rec['mean_rank_ic'])}, "
                f"cost-adj spread={_fmt(rec['economic']['cost_adjusted_spread'])}",
                flush=True,
            )
    return records


def _count_fresh_holdout_sessions(cfg: LakeConfig, start: str, end: str) -> int:
    """Trading-session count in [start, end] from the already-materialized feat_price mart."""
    path = cfg.feature_mart_root / "feat_price" / "*.parquet"
    con = duckdb.connect()
    try:
        n = con.execute(
            "SELECT count(DISTINCT trade_date) FROM read_parquet(?) "
            "WHERE trade_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)",
            [str(path), start, end],
        ).fetchone()[0]
    finally:
        con.close()
    return int(n)


def render_holdout_markdown(records: list[dict], *, holdout_len: int) -> str:
    lines: list[str] = []
    lines.append("# Grade A candidates — Phase 2 확정 holdout 결과 (1회 한정)")
    lines.append("")
    lines.append(
        f"> holdout: {FRESH_HOLDOUT_START} ~ {FRESH_HOLDOUT_END} ({holdout_len} 거래일) — "
        "이 모델에서 어떤 결정에도 쓰인 적 없는 신규 구간"
    )
    lines.append(f"> git commit at run time: `{_git_commit()}`")
    lines.append("")
    lines.append(
        "| config | horizon | holdout Rank IC | ICIR | top-decile spread | "
        "cost-adj spread(60bp 왕복 가정) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        h = r["holdout_report"]
        e = r["economic"]
        lines.append(
            f"| {r['name']} | {r['horizon']}d | **{_fmt(h['rank_ic_mean'])}** | "
            f"{_fmt(h['icir'], 3)} | {_fmt(h['top_decile_spread'])} | "
            f"{_fmt(e['cost_adjusted_spread'])} |"
        )
    lines.append("")
    lines.append(
        "> 이 결과 파일이 존재하는 한 재실행하지 않는다(`--force` 없이는 거부) — "
        "동일 holdout을 두 번 열어보면 다음 결정에 선택 편향이 생긴다."
    )
    lines.append("")
    return "\n".join(lines)


HOLDOUT_PARTIAL_JSON = DOCS_DIR / "grade_a_acceptance_gate_holdout.partial.json"


def _load_partial_records() -> list[dict]:
    if not HOLDOUT_PARTIAL_JSON.exists():
        return []
    return json.loads(HOLDOUT_PARTIAL_JSON.read_text())["records"]


def _save_partial_records(records: list[dict]) -> None:
    """Checkpoint after every (config, horizon) combo.

    This run has repeatedly been killed by external system load partway
    through (observed: 4 attempts, always around combo 2-3 of 6, with no
    Python traceback — an OS/infra-level kill, not a bug in this script).
    Checkpointing after each combo means a restart resumes instead of
    re-running from scratch. Resuming a deterministic pipeline (fixed
    random_state + the ORDER BY fix in build_dataset._panel_sql) from where
    it stopped is the SAME single holdout look, not a second one — nothing
    about the config or the already-computed combos changes on resume.
    """
    HOLDOUT_PARTIAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_PARTIAL_JSON.write_text(json.dumps({"records": records}, indent=2, default=str))


def _load_phase1_best_params() -> dict[tuple[str, int], dict]:
    """Hyperparameters already selected by Phase 1's walk-forward (no holdout data).

    Reusing these instead of re-running the grid search here keeps the two
    phases cleanly separated (select on validation only; fit once and evaluate
    on holdout with the params already chosen) and skips ~20 redundant fits per
    (config, horizon) — the grid search itself never touches holdout data
    either way, but redoing it inside ``run_holdout`` was pure waste.
    """
    if not RESULTS_JSON.exists():
        raise SystemExit(f"{RESULTS_JSON} not found — run Phase 1 (walk-forward) first.")
    payload = json.loads(RESULTS_JSON.read_text())
    return {(r["name"], r["horizon"]): r["best_params"] for r in payload["records"]}


def run_holdout(*, cfg: LakeConfig, force: bool) -> None:
    if HOLDOUT_JSON.exists() and not force:
        raise SystemExit(
            f"{HOLDOUT_JSON} already exists — this holdout has already been opened once. "
            "Re-running would look at already-spent data; pass --force only if you are "
            "deliberately re-confirming a documented exception."
        )
    phase1_best_params = _load_phase1_best_params()
    n_sessions = _count_fresh_holdout_sessions(cfg, FRESH_HOLDOUT_START, FRESH_HOLDOUT_END)
    if n_sessions < 5:
        raise SystemExit(
            f"only {n_sessions} sessions found in {FRESH_HOLDOUT_START}..{FRESH_HOLDOUT_END} "
            "in this snapshot — refusing to run a near-empty holdout."
        )
    print(f"fresh holdout window has {n_sessions} trading sessions", flush=True)

    spec = _base_spec(smoke=False, period_end=FRESH_HOLDOUT_END, holdout_len=n_sessions)
    records: list[dict] = _load_partial_records()
    done = {(r["name"], r["horizon"]) for r in records}
    if records:
        print(
            f"resuming from checkpoint: {len(records)} combo(s) already done: {sorted(done)}",
            flush=True,
        )
    for name, cols in CONFIGS:
        if all((name, h) in done for h in TARGET_HORIZONS):
            print(
                f"\n=== [holdout] config={name} === (all horizons already checkpointed)", flush=True
            )
            continue
        print(f"\n=== [holdout] config={name} ===", flush=True)
        build = _build_one(name, cols, spec, cfg)
        std = pl.read_parquet(build.dataset_dir / "feat_panel_std.parquet")
        for h in TARGET_HORIZONS:
            if (name, h) in done:
                print(f"  h={h}d: already checkpointed, skipping", flush=True)
                continue
            config = tr.TrainConfig(
                model=spec.model_type, target=f"y_rank_{h}d", realized=f"raw_label_{h}d"
            )
            best_params = phase1_best_params[(name, h)]
            result = tr.TrainResult(
                config=config,
                fold_results=[],
                best_params=best_params,
                mean_rank_ic=float("nan"),  # not re-selected here; see Phase 1 results for this
                feature_cols=tr.design_columns(std, config.target),
            )
            holdout_report = tr.evaluate_holdout(std, result)
            if holdout_report is None:
                raise SystemExit(f"no holdout slice produced for config={name} h={h}d")
            preds = tr.predict_holdout_frame(std, result)
            econ = economic_report(preds, pred_col="pred", realized_col=config.realized, horizon=h)
            records.append(
                {
                    "name": name,
                    "horizon": h,
                    "n_raw_features": len(build.feature_cols),
                    "holdout_report": holdout_report.as_dict(),
                    "economic": econ.as_dict(),
                }
            )
            _save_partial_records(records)
            print(
                f"  h={h}d: holdout Rank IC={_fmt(holdout_report.rank_ic_mean)}, "
                f"cost-adj spread={_fmt(econ.cost_adjusted_spread)}",
                flush=True,
            )

    payload = {
        "holdout_start": FRESH_HOLDOUT_START,
        "holdout_end": FRESH_HOLDOUT_END,
        "n_sessions": n_sessions,
        "git_commit": _git_commit(),
        "baseline_cols": list(BASELINE_COLS),
        "candidate_cols": list(CANDIDATE_COLS),
        "records": records,
    }
    HOLDOUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    HOLDOUT_MD.write_text(render_holdout_markdown(records, holdout_len=n_sessions))
    HOLDOUT_PARTIAL_JSON.unlink(missing_ok=True)  # done — no longer a resumable checkpoint
    print(f"\nwrote {HOLDOUT_JSON} and {HOLDOUT_MD}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 acceptance gate for Grade A candidates.")
    ap.add_argument("--smoke", action="store_true", help="short-period fast validation run.")
    ap.add_argument(
        "--confirm-holdout",
        action="store_true",
        help="run the ONE-SHOT fresh-holdout confirmation (phase 2) instead of walk-forward.",
    )
    ap.add_argument(
        "--force", action="store_true", help="allow re-running an already-written holdout result."
    )
    ap.add_argument(
        "--snapshot-date",
        default=SNAPSHOT_DATE,
        help=f"lake snapshot date (default: {SNAPSHOT_DATE})",
    )
    ap.add_argument(
        "--source",
        default=SOURCE,
        help=f"lake source name (default: {SOURCE})",
    )
    ap.add_argument("--memory-limit", default="6GB", help="DuckDB memory_limit (default 6GB).")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads (default 4).")
    args = ap.parse_args()

    # feat_price/feat_flow under this snapshot were materialized by the Phase A
    # horizon-scan run and are cache-locked to ITS analysis_config_hash
    # (research/etl/mart.py) — declaring the same hash here lets this script
    # read those marts read-only instead of force-rebuilding (and clobbering)
    # the artifacts the official Phase A results and 06_grade_a_deep_dive/ rely on.
    horizon_scan_hash = load_config().config_hash
    root = DataRoot.resolve(market="kr")
    cfg = LakeConfig(
        root=root,
        snapshot_date=args.snapshot_date,
        source=args.source,
        analysis_config_hash=horizon_scan_hash,
        engine=EngineOptions(
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_directory=str(root.base / "_tmp"),
        ),
    )
    if not cfg.raw_root.exists():
        raise SystemExit(f"raw lake not present at {cfg.raw_root}")

    if args.confirm_holdout:
        run_holdout(cfg=cfg, force=args.force)
        return

    records = run_walkforward(smoke=args.smoke, cfg=cfg)
    md = render_walkforward_markdown(records, smoke=args.smoke)
    if not args.smoke:
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON.write_text(json.dumps({"records": records}, indent=2, default=str))
        RESULTS_MD.write_text(md)
        print(f"\nwrote {RESULTS_JSON} and {RESULTS_MD}", flush=True)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
