"""predict — top-k buy-candidate selection for model 01 (access_return_rank).

Turns the trained ranking model into an actual ranked buy list: for each
``trade_date`` it sorts names by the model score (``pred``) and keeps the top
``k`` (1..k). It reuses the existing standardized panel + walk-forward selection
(``train.walk_forward``) and the single post-selection holdout fit
(``train.predict_holdout_frame``), so there is NO leakage and NO rebuild here.

Two outputs are produced from the SAME holdout-fitted model (etl_00 §5, §6):

  * BACKTEST (evaluation): the per-date top-k over the whole holdout window,
    carrying the realized 20d excess return (``raw_label_20d``) where available,
    plus a realized-performance summary (mean top-k return, hit ratio). This is
    the "if we had bought that day's top-k" check.
  * LATEST (live): the top-k for the most recent ``trade_date`` in the panel.
    Its realized label is still null (t+20 in the future), so it is the actual
    forward-looking buy list.

The recommended feature combo is ``px+flow+fin`` (see
``docs/target/01_20_access_return_rank/feature_ablation_results.md`` §4), which
must have been built with ``holdout_len>0`` (a trailing holdout is required).

Usage (from repo root)::

    python -m modeler.models._01_20_access_return_rank.predict
    python -m modeler.models._01_20_access_return_rank.predict --k 50 \
        --dataset-dir data/datasets/ablation/px_flow_fin/01_20_access_return_rank/\
snapshot_date=2026-06-19
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from modeler.models._01_20_access_return_rank import train as tr
from modeler.models._01_20_access_return_rank.enrich import enrich_topk
from modeler.models._01_20_access_return_rank.spec import ModelSpec

DEFAULT_K = 100

# Recommended combo (px+flow+fin) built with a trailing holdout (ablation run).
DEFAULT_DATASET_DIR = Path(
    "data/datasets/ablation/px_flow_fin/01_20_access_return_rank/snapshot_date=2026-06-19"
)
PRED_ROOT = Path("data/predictions/01_20_access_return_rank")
DOCS_DIR = Path("docs/target/01_20_access_return_rank")
SUMMARY_MD = DOCS_DIR / "topk_selection_results.md"


def select_topk(
    preds: pl.DataFrame,
    k: int,
    *,
    date_col: str = "trade_date",
    pred_col: str = "pred",
) -> pl.DataFrame:
    """Per-date top-k by ``pred`` (descending). Adds ``rank`` (1..k) + percentile.

    Ranking is cross-sectional (within each ``trade_date``), matching how the
    model is evaluated (metrics.py). ``rank=1`` is the highest-scored name; ties
    are broken deterministically by ``pred`` order ("ordinal").
    """
    n_in_date = pl.len().over(date_col)
    rank = pl.col(pred_col).rank("ordinal", descending=True).over(date_col)
    ranked = preds.with_columns(
        rank.cast(pl.Int32).alias("rank"),
        (rank / n_in_date).alias("pred_percentile"),
    )
    return ranked.filter(pl.col("rank") <= k).sort([date_col, "rank"])


def _with_meta(df: pl.DataFrame, *, model: str, alpha, groups: str) -> pl.DataFrame:
    """Attach reproducibility metadata columns to a top-k frame."""
    return df.with_columns(
        pl.lit(groups).alias("model_groups"),
        pl.lit(model).alias("model_name"),
        pl.lit(str(alpha)).alias("best_alpha"),
        pl.lit(datetime.now(UTC).isoformat(timespec="seconds")).alias("generated_at"),
    )


def realized_summary(topk: pl.DataFrame, *, realized_col: str) -> dict:
    """Realized performance of the backtest top-k (rows with a known label).

    Computes the mean realized 20d excess return of the selected names and the
    hit ratio (fraction with positive realized excess), over the dates whose
    label is already known (the most recent ~H sessions have null labels).
    """
    known = topk.filter(pl.col(realized_col).is_not_null() & pl.col(realized_col).is_finite())
    if known.height == 0:
        return {"n_eval_rows": 0, "n_eval_dates": 0, "mean_realized": None, "hit_ratio": None}
    agg = known.select(
        pl.col(realized_col).mean().alias("mean_realized"),
        (pl.col(realized_col) > 0).mean().alias("hit_ratio"),
    ).row(0, named=True)
    return {
        "n_eval_rows": known.height,
        "n_eval_dates": known.get_column("trade_date").n_unique(),
        "mean_realized": float(agg["mean_realized"]),
        "hit_ratio": float(agg["hit_ratio"]),
    }


def _write_frame(df: pl.DataFrame, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write a top-k frame as both parquet (machine) and csv (human)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"{stem}.parquet"
    csv = out_dir / f"{stem}.csv"
    df.write_parquet(pq)
    df.write_csv(csv)
    return pq, csv


def generate_topk(
    dataset_dir: Path,
    *,
    k: int = DEFAULT_K,
    groups_label: str = "px+flow+fin",
) -> dict:
    """Train/select -> holdout-fit -> per-date top-k backtest + latest buy list.

    Returns a record with the selected hyperparameters, output paths and the
    realized-performance summary. Writes parquet/csv artifacts under
    ``data/predictions/`` and a markdown summary under ``docs/target/``.
    """
    panel_std = pl.read_parquet(dataset_dir / "feat_panel_std.parquet")
    spec = ModelSpec()
    config = tr.TrainConfig(model=spec.model_type)

    result = tr.walk_forward(panel_std, config)
    tr.evaluate_holdout(panel_std, result)  # populates holdout_report (metrics)
    preds = tr.predict_holdout_frame(panel_std, result)
    if preds is None:
        raise SystemExit(
            f"no holdout slice in {dataset_dir} — build the dataset with holdout_len>0"
        )

    realized_col = config.realized
    # best_params could be {'alpha': ...} or {'max_iter': ..., 'learning_rate': ...}
    best_p_str = ", ".join(f"{k}={v}" for k, v in result.best_params.items())

    # Per-date top-k, then attach display-only reference columns (name / market
    # cap / financials, PIT-safe) before stamping reproducibility metadata.
    topk = enrich_topk(select_topk(preds, k, date_col=config.date_col))
    topk = _with_meta(topk, model=config.model, alpha=best_p_str, groups=groups_label)

    latest_date = topk.get_column("trade_date").max()
    latest = topk.filter(pl.col("trade_date") == latest_date)

    backtest_dir = PRED_ROOT / "holdout"
    latest_dir = PRED_ROOT / "latest"
    bt_pq, bt_csv = _write_frame(topk, backtest_dir, "topk_holdout")
    lt_pq, lt_csv = _write_frame(latest, latest_dir, f"topk_{latest_date}")

    summary = realized_summary(topk, realized_col=realized_col)
    record = {
        "dataset_dir": str(dataset_dir),
        "groups": groups_label,
        "k": k,
        "model": config.model,
        "best_params": best_p_str,
        "mean_valid_rank_ic": result.mean_rank_ic,
        "holdout": result.holdout_report.as_dict() if result.holdout_report else None,
        "n_holdout_dates": topk.get_column("trade_date").n_unique(),
        "latest_date": str(latest_date),
        "latest_n": latest.height,
        "realized_summary": summary,
        "outputs": {
            "backtest_parquet": str(bt_pq),
            "backtest_csv": str(bt_csv),
            "latest_parquet": str(lt_pq),
            "latest_csv": str(lt_csv),
        },
    }
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(_render_markdown(record, latest))
    (PRED_ROOT / "topk_selection_results.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    return record


def _render_markdown(rec: dict, latest: pl.DataFrame) -> str:
    h = rec["holdout"] or {}
    s = rec["realized_summary"]
    lines: list[str] = []
    lines.append("# access_return_rank — 매수 후보 Top-k 선정 결과")
    lines.append("")
    lines.append(f"> 피쳐 조합: **{rec['groups']}** / 모델: {rec['model']} ({rec['best_params']})")
    lines.append(f"> k={rec['k']} / 데이터셋: `{rec['dataset_dir']}`")
    lines.append("")
    lines.append("## 1. 모델 요약")
    lines.append("")
    lines.append(f"- walk-forward valid mean Rank IC: **{rec['mean_valid_rank_ic']:.4f}**")
    if h:
        lines.append(
            f"- holdout Rank IC: **{h.get('rank_ic_mean'):.4f}** "
            f"(ICIR {h.get('icir'):.3f}, top-decile spread {h.get('top_decile_spread'):.4f})"
        )
    lines.append("")
    lines.append("## 2. holdout 백테스트 (per-date Top-k)")
    lines.append("")
    lines.append(
        f"- holdout 일자 수: {rec['n_holdout_dates']}, " f"출력: `{rec['outputs']['backtest_csv']}`"
    )
    if s["mean_realized"] is not None:
        lines.append(
            f"- 라벨 확정 구간 Top-{rec['k']} 실현 20일 초과수익 평균: "
            f"**{s['mean_realized']:.4f}** (적중률 {s['hit_ratio']:.3f}, "
            f"{s['n_eval_dates']}일 / {s['n_eval_rows']}행)"
        )
    lines.append("")
    lines.append(f"## 3. 최신일 매수 후보 ({rec['latest_date']}, 라벨 미확정)")
    lines.append("")
    lines.append(f"- 출력: `{rec['outputs']['latest_csv']}` (총 {rec['latest_n']}종목)")
    lines.append("")
    head = latest.select(
        [
            "rank",
            "ticker",
            "market",
            "name",
            "market_cap_eok",
            "revenue_eok",
            "operating_income_eok",
            "pred",
        ]
    ).head(20)
    lines.append("| rank | ticker | market | 종목명 | 시총(억) | 매출(억) | 영업이익(억) | pred |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def _fmt(v: object) -> str:
        return "-" if v is None else f"{v:,.0f}"

    for row in head.iter_rows(named=True):
        lines.append(
            f"| {row['rank']} | {row['ticker']} | {row['market']} | "
            f"{row['name'] or '-'} | {_fmt(row['market_cap_eok'])} | "
            f"{_fmt(row['revenue_eok'])} | {_fmt(row['operating_income_eok'])} | "
            f"{row['pred']:.4f} |"
        )
    lines.append("")
    lines.append("> 상위 20종목만 표기. 전체 목록은 위 CSV/Parquet 참조.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-k buy-candidate selection for model 01.")
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="built dataset dir with feat_panel_std.parquet (holdout_len>0 required).",
    )
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="top-k to select (default 100).")
    ap.add_argument(
        "--groups-label",
        default="px+flow+fin",
        help="feature-combo label recorded in outputs (metadata only).",
    )
    args = ap.parse_args()

    if not (args.dataset_dir / "feat_panel_std.parquet").exists():
        raise SystemExit(f"feat_panel_std.parquet not found under {args.dataset_dir}")

    rec = generate_topk(args.dataset_dir, k=args.k, groups_label=args.groups_label)
    print(json.dumps(rec, indent=2, default=str), flush=True)
    print(f"\nwrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
