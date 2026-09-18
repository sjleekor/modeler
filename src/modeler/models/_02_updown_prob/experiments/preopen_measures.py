"""preopen_measures — the three measurements the holdout gate asks for (S2).

R3 (strategy drawdown), T1-6 (buy-hold hysteresis) and T1-2 (square-root impact
cost) all read the same thing: a finished run's validation predictions. None of
them refits anything, so all three run on the adopted configs as they stand and
none can change what was adopted — they are additional preregistrations, and the
plan says so (`20260916_holdout_gate_plan.md` §2.1).

They are one module because they share the whole input path — predictions, the
fold map, the per-name mart columns and the close — and splitting that three
ways would mean maintaining it three times.

Outputs, one set per run, under ``results/``::

    drawdown/{run_id}_drawdown.parquet      fold x (max_drawdown, longest)
    hysteresis/{run_id}_hyst.parquet        fold x k_out
    cost_model/{run_id}_cost_curve.parquet  fold x capital x impact_k
    cost_model/{run_id}_cost_curve.png      the h20 curve, if matplotlib is here

``fold_metrics.parquet`` does not carry the drawdown columns yet even though
:class:`TopKEconomicReport` now has them: that frame is written by a run, and
re-fitting three runs to fill in two numbers this module can compute directly
would cost two hours. The next run — the holdout — writes them.

Usage::

    uv run python -m modeler.models._02_updown_prob.experiments.preopen_measures
    uv run python -m modeler.models._02_updown_prob.experiments.preopen_measures \\
        --run E2_h20_FS1h_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import polars as pl

from modeler.etl.metrics import (
    CostModel,
    per_name_cost_bps,
    topk_economic_report,
    topk_hysteresis_report,
    topk_rebalance_series,
)
from modeler.models._02_updown_prob.experiments.run_matrix import RESULTS_ROOT

# The three adopted configs (`results/README.md` §1), as (stage, run_id).
ADOPTED: tuple[tuple[str, str], ...] = (
    ("E1", "E1_h5_y_up-hgb_clf_seed0"),
    ("E2", "E2_h20_FS1h_seed0"),
    ("E5", "E5_h60_FS3_seed0"),
)

# T1-6 fixes these before any result is seen. 100 == k_in is the plain rule, the
# baseline every band is read against.
K_OUT: tuple[int, ...] = (100, 150, 200, 300)

# T1-2 fixes these too. The capitals bracket the plan's premise (personal money,
# ceiling around 2bn KRW); impact_k 1.0 is the headline and 0.5/1.5 the practice
# range, reported beside it rather than chosen after the fact.
CAPITALS_KRW: tuple[float, ...] = (1e8, 3e8, 1e9, 3e9)
IMPACT_K: tuple[float, ...] = (0.5, 1.0, 1.5)

ADV_COL, VOL_COL, CLOSE_COL = "px_turnover_ma20", "px_vol_20d", "close"


def _load_run(stage: str, run_id: str) -> tuple[dict, pl.DataFrame]:
    """A run's summary and its validation predictions, tagged with ``fold_id``."""
    summary = json.loads((RESULTS_ROOT / stage / run_id / "summary.json").read_text())
    predictions = pl.read_parquet(summary["predictions_path"])
    folds = pl.read_parquet(Path(summary["dataset_dir"]) / "split_folds.parquet")

    # The predictions frame carries no fold: rebuild it from the valid windows,
    # which do not overlap (purged walk-forward).
    fold_of = pl.lit(None, dtype=pl.Int64)
    for row in folds.iter_rows(named=True):
        inside = pl.col("trade_date").is_between(row["valid_start"], row["valid_end"])
        fold_of = pl.when(inside).then(pl.lit(row["fold_id"])).otherwise(fold_of)
    predictions = predictions.with_columns(fold_of.alias("fold_id")).filter(
        pl.col("fold_id").is_not_null()
    )
    return summary, predictions


def _with_cost_inputs(summary: dict, predictions: pl.DataFrame) -> pl.DataFrame:
    """Join the three columns the cost model needs (ADV, sigma, close)."""
    dataset_dir = Path(summary["dataset_dir"])
    panel = pl.scan_parquet(dataset_dir / "feat_panel.parquet").select(
        ["trade_date", "ticker", "market", ADV_COL, VOL_COL]
    )
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    raw_root = manifest["lake"]["raw"]
    con = duckdb.connect()
    close = pl.from_arrow(
        con.execute(
            f"SELECT trade_date, ticker, market, CAST(close AS DOUBLE) AS {CLOSE_COL} "
            f"FROM read_parquet('{raw_root}/daily_ohlcv/**/*.parquet')"
        ).arrow()
    )
    con.close()
    return (
        predictions.lazy()
        .join(panel, on=["trade_date", "ticker", "market"], how="left")
        .join(close.lazy(), on=["trade_date", "ticker", "market"], how="left")
        .collect()
    )


def _drawdown_rows(frame: pl.DataFrame, summary: dict) -> list[dict]:
    """R3 — the strategy's worst fall and longest wait, per fold."""
    rows = []
    for fold_id, fold in _by_fold(frame):
        report = topk_economic_report(fold, **_topk_kwargs(summary))
        series = topk_rebalance_series(fold, **_topk_kwargs(summary))
        rows.append(
            {
                "fold_id": fold_id,
                "pred_col": summary["primary_pred_col"],
                **report.as_dict(),
                "n_rebalances_scored": int(series["gross_return"].drop_nulls().len()),
            }
        )
    return rows


def _hysteresis_rows(frame: pl.DataFrame, summary: dict) -> list[dict]:
    """T1-6 — the same economics under each preregistered band."""
    rows = []
    for fold_id, fold in _by_fold(frame):
        for k_out in K_OUT:
            report = topk_hysteresis_report(
                fold,
                pred_col=summary["primary_pred_col"],
                realized_col=f"raw_label_{summary['horizon']}d",
                horizon=summary["horizon"],
                k_in=summary["k"],
                k_out=k_out,
                cost_bps_roundtrip=summary["cost_bps_roundtrip"],
            )
            rows.append({"fold_id": fold_id, **report.as_dict()})
    return rows


def _cost_curve_rows(frame: pl.DataFrame, summary: dict) -> list[dict]:
    """T1-2 — what the same list costs as the order grows.

    Capital and ``impact_k`` move the cost only; the buy list, the gross return
    and the turnover are the same at every point on the curve. So the gross side
    is computed once per fold and the curve is the cost sweep over it. The first
    point is cross-checked against ``topk_economic_report(cost_model=...)`` so
    this shortcut cannot drift from the function the tests pin.
    """
    rows = []
    for fold_id, fold in _by_fold(frame):
        base = topk_economic_report(fold, **_topk_kwargs(summary))
        picks = _bought_names(fold, summary)
        for capital in CAPITALS_KRW:
            for impact_k in IMPACT_K:
                model = CostModel(
                    capital_krw=capital, k_names=summary["k"], impact_k=impact_k
                )
                per_name = (
                    per_name_cost_bps(
                        picks, model=model, adv_col=ADV_COL, vol_col=VOL_COL,
                        close_col=CLOSE_COL,
                    )
                    .drop_nulls()
                    .drop_nans()
                )
                mean_bps = float(per_name.mean()) if per_name.len() else float("nan")
                net = base.grid_topk_mean_return - base.turnover * mean_bps / 10_000.0
                rows.append(
                    {
                        "fold_id": fold_id,
                        "horizon": summary["horizon"],
                        "k": summary["k"],
                        "capital_krw": capital,
                        "impact_k": impact_k,
                        "mean_cost_bps": mean_bps,
                        "n_priced": int(per_name.len()),
                        "n_bought": int(picks.height),
                        "turnover": base.turnover,
                        "grid_topk_mean_return": base.grid_topk_mean_return,
                        "cost_adjusted_return": net,
                        "breakeven_capital_krw": _breakeven_capital(
                            picks, summary, base.grid_topk_mean_return, base.turnover, impact_k
                        )
                        if capital == CAPITALS_KRW[0]
                        else None,
                    }
                )
        _check_shortcut(fold, summary, base, rows[-1])
    return rows


def _breakeven_capital(
    picks: pl.DataFrame, summary: dict, gross: float, turnover: float, impact_k: float
) -> float:
    """The capital at which the cost eats the whole return (T1-2's 손익분기 C).

    Bisected on the real per-name cost rather than solved on a fitted curve: the
    mean cost is an average over the names actually bought, and their ADV
    distribution is what makes the number interesting. ``nan`` when the strategy
    is already under water at the smallest size, or still above it at the
    largest — saying "below 1천만원" or "above 1조" is honest, extrapolating is
    not.
    """
    if not (gross == gross and turnover == turnover):
        return float("nan")

    def net(capital: float) -> float:
        model = CostModel(capital_krw=capital, k_names=summary["k"], impact_k=impact_k)
        per_name = (
            per_name_cost_bps(
                picks, model=model, adv_col=ADV_COL, vol_col=VOL_COL, close_col=CLOSE_COL
            )
            .drop_nulls()
            .drop_nans()
        )
        if not per_name.len():
            return float("nan")
        return gross - turnover * float(per_name.mean()) / 10_000.0

    lo, hi = 1e7, 1e12  # 1천만원 .. 1조
    if not (net(lo) > 0 > net(hi)):
        return float("nan")
    for _ in range(60):
        mid = (lo * hi) ** 0.5  # geometric: the cost moves with sqrt(capital)
        if net(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float((lo * hi) ** 0.5)


def _check_shortcut(
    fold: pl.DataFrame, summary: dict, base, last: dict
) -> None:
    """The sweep must land where the report itself would."""
    model = CostModel(
        capital_krw=last["capital_krw"], k_names=summary["k"], impact_k=last["impact_k"]
    )
    full = topk_economic_report(fold, **_topk_kwargs(summary), cost_model=model)
    if abs(full.cost_adjusted_return - last["cost_adjusted_return"]) > 1e-12:
        raise AssertionError(
            f"cost sweep drifted from topk_economic_report: "
            f"{last['cost_adjusted_return']!r} vs {full.cost_adjusted_return!r}"
        )


def _bought_names(fold: pl.DataFrame, summary: dict) -> pl.DataFrame:
    """The rows actually bought on the rebalance grid — what the cost applies to."""
    from modeler.etl.metrics import _topk_ranked, rebalance_grid

    grid = rebalance_grid(fold["trade_date"].to_list(), summary["horizon"])
    picked = _topk_ranked(
        fold,
        pred_col=summary["primary_pred_col"],
        k=summary["k"],
        date_col="trade_date",
        ticker_col="ticker",
    ).filter(pl.col("trade_date").is_in(grid))
    return picked.join(
        fold.select(["trade_date", "ticker", ADV_COL, VOL_COL, CLOSE_COL]),
        on=["trade_date", "ticker"],
        how="inner",
    )


def _topk_kwargs(summary: dict) -> dict:
    return {
        "pred_col": summary["primary_pred_col"],
        "realized_col": f"raw_label_{summary['horizon']}d",
        "horizon": summary["horizon"],
        "k": summary["k"],
        "cost_bps_roundtrip": summary["cost_bps_roundtrip"],
    }


def _by_fold(frame: pl.DataFrame):
    for fold_id in sorted(frame["fold_id"].unique().to_list()):
        yield fold_id, frame.filter(pl.col("fold_id") == fold_id)


def _write(rows: list[dict], path: Path) -> pl.DataFrame:
    frame = pl.DataFrame(rows, infer_schema_length=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    print(f"  {path.relative_to(RESULTS_ROOT)}: {frame.height} rows")
    return frame


def _plot_cost_curve(frame: pl.DataFrame, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib absent — the curve is in the parquet)")
        return

    mean = (
        frame.group_by(["capital_krw", "impact_k"])
        .agg(pl.col("cost_adjusted_return").mean())
        .sort(["impact_k", "capital_krw"])
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for impact_k in sorted(mean["impact_k"].unique().to_list()):
        part = mean.filter(pl.col("impact_k") == impact_k)
        ax.plot(
            part["capital_krw"] / 1e8,
            part["cost_adjusted_return"] * 100,
            marker="o",
            label=f"k = {impact_k}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("capital (100M KRW)")  # ASCII: the bundled font has no Hangul
    ax.set_ylabel("cost-adjusted return per rebalance (%)")
    ax.set_title("T1-2: square-root impact vs order size")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  {path.relative_to(RESULTS_ROOT)}")


def measure(stage: str, run_id: str) -> None:
    print(f"\n{run_id}")
    summary, predictions = _load_run(stage, run_id)
    frame = _with_cost_inputs(summary, predictions)

    _write(_drawdown_rows(frame, summary), RESULTS_ROOT / "drawdown" / f"{run_id}_drawdown.parquet")
    _write(
        _hysteresis_rows(frame, summary),
        RESULTS_ROOT / "hysteresis" / f"{run_id}_hyst.parquet",
    )
    curve = _write(
        _cost_curve_rows(frame, summary),
        RESULTS_ROOT / "cost_model" / f"{run_id}_cost_curve.parquet",
    )
    if summary["horizon"] == 20:
        _plot_cost_curve(curve, RESULTS_ROOT / "cost_model" / f"{run_id}_cost_curve.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="one run_id (default: all three adopted)")
    args = parser.parse_args(argv)
    todo = [(s, r) for s, r in ADOPTED if args.run in (None, r)]
    if not todo:
        raise SystemExit(f"unknown run {args.run!r}; known: {[r for _s, r in ADOPTED]}")
    for stage, run_id in todo:
        measure(stage, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
