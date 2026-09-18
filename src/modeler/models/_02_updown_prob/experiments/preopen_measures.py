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
    _quantile_stats,
    _topk_ranked,
    classification_report,
    neutralize_predictions,
    per_date_rank_ic,
    per_name_cost_bps,
    rebalance_grid,
    topk_economic_report,
    topk_hysteresis_report,
    topk_rebalance_series,
)
from modeler.etl.metrics import evaluate as rank_evaluate
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

# R1/R2 inputs, read as they lie from the snapshot the run was built on.
TRADABLE_MART = "dim_universe_tradable_daily"
MCAP_MART = "feat_market_cap"
INDEX_SERIES = {"market_kospi_krx": "KOSPI", "market_kosdaq_krx": "KOSDAQ"}

# T1-1B. The first pair is what tier1 §1 registered; the second is the same
# measurement with a size proxy that is not 11% missing, reported beside it so a
# reader can see whether filling that 11% made the answer. The verdict is read
# off the registered pair — the sensitivity is not an alternative to choose.
NEUT_PREREG: tuple[str, ...] = ("fin_log_mcap", "px_amihud_20d")
NEUT_SENSITIVITY: tuple[str, ...] = ("mcap_krx_log", "px_amihud_20d")
NEUT_RATIOS: tuple[float, ...] = (0.5, 1.0)


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
    have = pl.scan_parquet(dataset_dir / "feat_panel.parquet").head(0).collect().columns
    wanted = ["trade_date", "ticker", "market", ADV_COL, VOL_COL]
    # the neutralization exposures ride along when this feature set has them:
    # h5's FS0 does not carry fin_log_mcap, and T1-1B is an h20 measurement.
    wanted += [c for c in NEUT_PREREG if c in have and c not in wanted]
    panel = pl.scan_parquet(dataset_dir / "feat_panel.parquet").select(wanted)
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


def _mart_glob(summary: dict, mart: str) -> str:
    manifest = json.loads((Path(summary["dataset_dir"]) / "dataset_manifest.json").read_text())
    return f"{manifest['lake']['feature_mart']}/{mart}/**/*.parquet"


def _raw_glob(summary: dict, table: str) -> str:
    manifest = json.loads((Path(summary["dataset_dir"]) / "dataset_manifest.json").read_text())
    return f"{manifest['lake']['raw']}/{table}/**/*.parquet"


def _regime_by_market_year(summary: dict, frame: pl.DataFrame) -> pl.DataFrame:
    """§5.2 — bull/bear per (market, year), on the sessions actually evaluated.

    Not the calendar year: the 2025 rows stop at the formation boundary, and a
    full-year index return would reach past it into the holdout. Measuring only
    the sessions inside the evaluated span keeps the label bounded by the data
    it describes, and keeps working when the holdout ends mid-year.
    """
    lo, hi = frame["trade_date"].min(), frame["trade_date"].max()
    pairs = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in INDEX_SERIES.items())
    con = duckdb.connect()
    rows = pl.from_arrow(
        con.execute(f"""
            WITH px AS (
                SELECT CASE series_id {pairs} END AS market,
                       observation_date AS d,
                       CAST(value_numeric AS DOUBLE) AS c,
                       year(observation_date) AS y
                FROM read_parquet('{_raw_glob(summary, "common_feature_observation_raw")}')
                WHERE series_id IN ({", ".join(f"'{k}'" for k in INDEX_SERIES)})
                  AND value_numeric IS NOT NULL
                  AND observation_date BETWEEN DATE '{lo}' AND DATE '{hi}'
            ), bounds AS (
                SELECT market, y, min(d) AS d0, max(d) AS d1 FROM px GROUP BY 1, 2
            )
            SELECT b.market, b.y AS year, p1.c / p0.c - 1 AS index_return
            FROM bounds b
            JOIN px p0 ON p0.market = b.market AND p0.d = b.d0
            JOIN px p1 ON p1.market = b.market AND p1.d = b.d1
        """).arrow()
    )
    con.close()
    return rows.with_columns(
        pl.when(pl.col("index_return") > 0)
        .then(pl.lit("bull"))
        .otherwise(pl.lit("bear"))
        .alias("regime")
    )


def _with_universe_and_buckets(summary: dict, frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the tradable flag (R1) and the three bucket keys (R2, §5.3)."""
    con = duckdb.connect()
    tradable = pl.from_arrow(
        con.execute(
            "SELECT trade_date, ticker, market, in_universe AS tradable, "
            "management_filter_available "
            f"FROM read_parquet('{_mart_glob(summary, TRADABLE_MART)}')"
        ).arrow()
    )
    mcap = pl.from_arrow(
        con.execute(
            "SELECT trade_date, ticker, market, mcap_krx, mcap_krx_log, mcap_unreliable "
            f"FROM read_parquet('{_mart_glob(summary, MCAP_MART)}')"
        ).arrow()
    )
    con.close()

    out = (
        frame.join(tradable, on=["trade_date", "ticker", "market"], how="left")
        .join(mcap, on=["trade_date", "ticker", "market"], how="left")
        .with_columns(pl.col("tradable").fill_null(False))
    )
    regimes = _regime_by_market_year(summary, frame)
    out = out.with_columns(pl.col("trade_date").dt.year().alias("year")).join(
        regimes.select(["market", "year", "regime", "index_return"]),
        on=["market", "year"],
        how="left",
    )
    # Buckets are cut per date across BOTH markets, because the buy list is
    # ranked that way (`_topk_ranked` groups by date alone). Cutting them per
    # market would describe a portfolio nobody holds.
    return out.with_columns(
        pl.col("mcap_krx")
        .qcut([1 / 3, 2 / 3], labels=["small", "mid", "large"], allow_duplicates=True)
        .over("trade_date")
        .cast(pl.Utf8)
        .alias("size_tertile"),
        pl.when(pl.col(ADV_COL) >= pl.col(ADV_COL).median().over("trade_date"))
        .then(pl.lit("liquid"))
        .otherwise(pl.lit("illiquid"))
        .alias("liquidity_half"),
    )


def _tradable_rows(frame: pl.DataFrame, summary: dict) -> list[dict]:
    """R1 — the adopted run re-scored on the tradable universe (1,000원 floor)."""
    pred_col = summary["primary_pred_col"]
    realized = f"raw_label_{summary['horizon']}d"
    rows = []
    for fold_id, fold in _by_fold(frame):
        for scope, part in (("base", fold), ("tradable", fold.filter(pl.col("tradable")))):
            if part.is_empty():
                continue
            topk = topk_economic_report(part, **_topk_kwargs(summary))
            rank = rank_evaluate(part, pred_col=pred_col, realized_col=realized)
            clf = classification_report(
                part, pred_col=pred_col, y_col=f"y_up_{summary['horizon']}d", k=summary["k"]
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "scope": scope,
                    "n_obs": int(part.height),
                    "log_loss": clf.log_loss,
                    "ece": clf.ece,
                    **rank.as_dict(),
                    **{f"topk_{key}": value for key, value in topk.as_dict().items()},
                }
            )
    return rows


def _bucket_rows(frame: pl.DataFrame, summary: dict) -> list[dict]:
    """R2 — signal and picks per (size, liquidity, regime), on the tradable set."""
    pred_col = summary["primary_pred_col"]
    realized = f"raw_label_{summary['horizon']}d"
    tradable = frame.filter(pl.col("tradable"))
    grid = rebalance_grid(tradable["trade_date"].to_list(), summary["horizon"])
    picks = _topk_ranked(
        tradable, pred_col=pred_col, k=summary["k"], date_col="trade_date", ticker_col="ticker"
    ).filter(pl.col("trade_date").is_in(grid))
    picked = tradable.join(
        picks.select(["trade_date", "ticker"]).with_columns(pl.lit(True).alias("_picked")),
        on=["trade_date", "ticker"],
        how="left",
    ).with_columns(pl.col("_picked").fill_null(False))
    n_picked_total = int(picked["_picked"].sum())

    rows = []
    keys = ["size_tertile", "liquidity_half", "regime"]
    for bucket, part in picked.group_by(keys, maintain_order=True):
        if any(v is None for v in bucket):
            continue
        ic = per_date_rank_ic(part, pred_col=pred_col, realized_col=realized)
        ics = ic["rank_ic"].drop_nulls().to_numpy()
        spread, tmb, hit, _ = _quantile_stats(
            part, pred_col=pred_col, realized_col=realized, date_col="trade_date"
        )
        chosen = part.filter(pl.col("_picked"))
        realized_ok = chosen.filter(pl.col(realized).is_not_null())
        rows.append(
            {
                **dict(zip(keys, bucket)),
                "n_obs": int(part.height),
                "n_dates": int(ic.height),
                "n_unreliable_mcap": int(part["mcap_unreliable"].sum() or 0),
                "rank_ic_mean": float(ics.mean()) if ics.size else float("nan"),
                "top_decile_spread": spread,
                "hit_ratio_top": hit,
                "topk_share": chosen.height / n_picked_total if n_picked_total else float("nan"),
                "topk_mean_return": float(realized_ok[realized].mean())
                if realized_ok.height
                else float("nan"),
                "index_return_mean": float(part["index_return"].mean()),
            }
        )
    return rows


def _neutralization_rows(frame: pl.DataFrame, summary: dict) -> list[dict]:
    """T1-1B — how much of the score we already have is size and liquidity.

    Nothing is refit. The adopted probabilities are regressed on the exposures
    inside each ``(date, market)`` and the residual is scored with the same
    metrics, so the gap is the part of the alpha that size explains.
    """
    pred_col = summary["primary_pred_col"]
    realized = f"raw_label_{summary['horizon']}d"
    variants: list[tuple[str, tuple[str, ...] | None, float]] = [("raw", None, 0.0)]
    for label, columns in (("prereg", NEUT_PREREG), ("sensitivity", NEUT_SENSITIVITY)):
        if not all(c in frame.columns for c in columns):
            print(f"  중립화 {label}: 컬럼 없음 {columns} — 건너뛴다")
            continue
        variants += [(f"{label}_{ratio}", columns, ratio) for ratio in NEUT_RATIOS]

    rows = []
    for fold_id, fold in _by_fold(frame):
        part = fold.filter(pl.col("tradable"))
        for name, columns, ratio in variants:
            scored = part
            column = pred_col
            if columns is not None:
                scored = part.with_columns(
                    neutralize_predictions(
                        part, pred_col=pred_col, on=list(columns), ratio=ratio
                    ).alias("p_neut")
                )
                column = "p_neut"
            topk = topk_economic_report(
                scored,
                pred_col=column,
                realized_col=realized,
                horizon=summary["horizon"],
                k=summary["k"],
                cost_bps_roundtrip=summary["cost_bps_roundtrip"],
            )
            rank = rank_evaluate(scored, pred_col=column, realized_col=realized)
            rows.append(
                {
                    "fold_id": fold_id,
                    "variant": name,
                    "exposures": ",".join(columns) if columns else "",
                    "ratio": ratio,
                    "rank_ic_mean": rank.rank_ic_mean,
                    "top_decile_spread": rank.top_decile_spread,
                    "topk_grid_topk_mean_return": topk.grid_topk_mean_return,
                    "topk_cost_adjusted_return": topk.cost_adjusted_return,
                    "topk_turnover": topk.turnover,
                    "topk_max_drawdown": topk.max_drawdown,
                }
            )
    return rows


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

    frame = _with_universe_and_buckets(summary, frame)
    if frame["management_filter_available"].any():
        print("  관리종목 필터가 생겼다 — R1의 전제를 다시 보라")
    else:
        print("  관리종목: 원천 없음 (management_filter_available = FALSE)")

    _write(
        _tradable_rows(frame, summary),
        RESULTS_ROOT / "tradable" / f"{run_id}_fold_metrics_tradable.parquet",
    )
    _write(_bucket_rows(frame, summary), RESULTS_ROOT / "by_bucket" / f"{run_id}_by_bucket.parquet")
    _write(
        _neutralization_rows(frame, summary),
        RESULTS_ROOT / "neutralization" / f"{run_id}_neut.parquet",
    )
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
