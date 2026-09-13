"""Financial PIT features vs forward-return labels correlation EDA.

This is a local research script. It reads snapshot-pinned parquet tables,
materializes `label_daily` if absent, and writes plots/tables under reports/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modeler.etl.config import DEFAULT_SOURCE, DataRoot, LakeConfig
from modeler.etl.labels import LabelSpec, materialize_label
from modeler.etl.lake import connect, register_views
from modeler.etl.mart import is_materialized, register_mart_view

FEATURES: dict[str, dict[str, object]] = {
    "fin_roa": {"clip": True, "label": "ROA"},
    "fin_roe": {"clip": True, "label": "ROE"},
    "fin_debt_to_equity": {"clip": True, "label": "Debt/Equity"},
    "fin_equity_ratio": {"clip": True, "label": "Equity Ratio"},
    "fin_ocf_to_assets": {"clip": True, "label": "OCF/Assets"},
    "fin_cash_ratio": {"clip": True, "label": "Cash/Assets"},
    "fin_asset_turnover": {"clip": True, "label": "Asset Turnover"},
    "fin_operating_margin": {"clip": True, "label": "Operating Margin"},
    "fin_is_negative_equity": {"clip": False, "label": "Negative Equity"},
    "fin_has_fs": {"clip": False, "label": "Has FS"},
}

TARGETS = (
    "fwd_ret_5d",
    "fwd_ret_20d",
    "fwd_ret_60d",
    "raw_label_5d",
    "raw_label_20d",
    "raw_label_60d",
)


@dataclass(frozen=True)
class RunContext:
    snapshot_date: str
    source: str
    report_date: str
    report_dir: Path
    primary_target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--report-date",
        default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d"),
        help="Output subdirectory under reports/analysis/fin_vs_price_corr.",
    )
    parser.add_argument("--primary-target", default="raw_label_20d", choices=TARGETS)
    parser.add_argument("--force-label", action="store_true")
    parser.add_argument("--scatter-sample", type=int, default=20_000)
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def setup_connection(config: LakeConfig, *, force_label: bool) -> object:
    con = connect(config)
    register_views(con, config, tables=["daily_ohlcv"])
    register_mart_view(con, config, "feat_fin_pit")
    register_mart_view(con, config, "feat_price")

    if force_label or not is_materialized(config, "label_daily"):
        materialize_label(con, config, LabelSpec(), force=force_label)
    else:
        register_mart_view(con, config, "label_daily")
    return con


def create_analysis_views(con: object) -> pd.DataFrame:
    feature_select = []
    for feature, meta in FEATURES.items():
        source = f"f.{quote_ident(feature)}"
        if meta["clip"]:
            feature_select.append(f"CAST({source} AS DOUBLE) AS {quote_ident(feature)}")
        else:
            feature_select.append(f"CAST({source} AS DOUBLE) AS {quote_ident(feature)}")

    target_select = [f"CAST(l.{target} AS DOUBLE) AS {target}" for target in TARGETS]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW fin_price_corr_base_raw AS
        SELECT
            f.trade_date,
            f.ticker,
            f.market,
            EXTRACT(year FROM f.trade_date)::INTEGER AS year,
            {", ".join(feature_select)},
            {", ".join(target_select)}
        FROM feat_fin_pit AS f
        JOIN label_daily AS l
          USING (trade_date, ticker, market)
        LEFT JOIN feat_price AS p
          USING (trade_date, ticker, market)
        WHERE p.px_is_halted IS NOT TRUE
        """)

    quantiles: dict[str, tuple[float | None, float | None]] = {}
    for feature, meta in FEATURES.items():
        if not meta["clip"]:
            continue
        row = con.execute(f"""
            SELECT
                quantile_cont({quote_ident(feature)}, 0.01),
                quantile_cont({quote_ident(feature)}, 0.99)
            FROM fin_price_corr_base_raw
            WHERE {quote_ident(feature)} IS NOT NULL
            """).fetchone()
        quantiles[feature] = row if row else (None, None)

    select_cols = ["trade_date", "ticker", "market", "year"]
    for feature, meta in FEATURES.items():
        ident = quote_ident(feature)
        if meta["clip"]:
            lo, hi = quantiles.get(feature, (None, None))
            if lo is None or hi is None:
                select_cols.append(ident)
            else:
                select_cols.append(f"LEAST(GREATEST({ident}, {lo}), {hi}) AS {ident}")
        else:
            select_cols.append(ident)
    select_cols.extend(TARGETS)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW fin_price_corr_base AS
        SELECT {", ".join(select_cols)}
        FROM fin_price_corr_base_raw
        """)

    rows = []
    for feature, (lo, hi) in quantiles.items():
        rows.append({"feature": feature, "winsor_p01": lo, "winsor_p99": hi})
    return pd.DataFrame(rows)


def compute_correlations(con: object) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        f = quote_ident(feature)
        for target in TARGETS:
            pearson = con.execute(f"""
                SELECT count(*) AS n, corr({f}, {target}) AS pearson
                FROM fin_price_corr_base
                WHERE {f} IS NOT NULL AND {target} IS NOT NULL
                """).fetchone()
            spearman = con.execute(f"""
                WITH ranked AS (
                    SELECT
                        percent_rank() OVER (ORDER BY {f}) AS rx,
                        percent_rank() OVER (ORDER BY {target}) AS ry
                    FROM fin_price_corr_base
                    WHERE {f} IS NOT NULL AND {target} IS NOT NULL
                )
                SELECT corr(rx, ry) AS spearman
                FROM ranked
                """).fetchone()
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURES[feature]["label"],
                    "target": target,
                    "n": int(pearson[0] or 0),
                    "pearson": pearson[1],
                    "spearman": spearman[0] if spearman else None,
                }
            )
    return pd.DataFrame(rows)


def plot_corr_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    methods = ("pearson", "spearman")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, method in zip(axes, methods, strict=True):
        pivot = (
            corr_df.pivot(index="feature_label", columns="target", values=method)
            .reindex([str(v["label"]) for v in FEATURES.values()])
            .reindex(columns=list(TARGETS))
        )
        values = pivot.to_numpy(dtype=float)
        max_abs = np.nanmax(np.abs(values))
        lim = max(0.03, float(max_abs)) if np.isfinite(max_abs) else 0.03
        im = ax.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(method.capitalize())
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Financial PIT features vs forward-return labels")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def top_features(corr_df: pd.DataFrame, primary_target: str, top_n: int) -> list[str]:
    ratio_features = [feature for feature, meta in FEATURES.items() if meta["clip"]]
    primary = corr_df[
        (corr_df["target"] == primary_target) & (corr_df["feature"].isin(ratio_features))
    ].copy()
    primary["abs_spearman"] = primary["spearman"].abs()
    primary = primary.sort_values(["abs_spearman", "n"], ascending=[False, False])
    return primary.head(top_n)["feature"].tolist()


def plot_scatter_top(
    con: object, features: list[str], target: str, sample_size: int, out_path: Path
) -> None:
    if not features:
        return
    fig, axes = plt.subplots(
        1, len(features), figsize=(5 * len(features), 4), constrained_layout=True
    )
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features, strict=True):
        f = quote_ident(feature)
        df = con.execute(f"""
            SELECT {f} AS x, {target} AS y
            FROM fin_price_corr_base
            WHERE {f} IS NOT NULL AND {target} IS NOT NULL
            ORDER BY hash(ticker, trade_date)
            LIMIT {int(sample_size)}
            """).df()
        ax.scatter(df["x"], df["y"], s=4, alpha=0.18, linewidths=0)
        y_lo, y_hi = df["y"].quantile([0.005, 0.995])
        if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo < y_hi:
            ax.set_ylim(float(y_lo), float(y_hi))
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(str(FEATURES[feature]["label"]))
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
    fig.suptitle("Sampled scatter, y-axis clipped to 0.5%-99.5% for display")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_quintiles(con: object, features: list[str], primary_target: str) -> pd.DataFrame:
    frames = []
    target_exprs = ", ".join(f"avg({target}) AS avg_{target}" for target in TARGETS)
    for feature in features:
        f = quote_ident(feature)
        df = con.execute(f"""
            WITH q AS (
                SELECT
                    ntile(5) OVER (ORDER BY {f}) AS quintile,
                    {f} AS feature_value,
                    {", ".join(TARGETS)}
                FROM fin_price_corr_base
                WHERE {f} IS NOT NULL AND {primary_target} IS NOT NULL
            )
            SELECT
                '{feature}' AS feature,
                quintile,
                count(*) AS n,
                avg(feature_value) AS avg_feature,
                {target_exprs}
            FROM q
            GROUP BY quintile
            ORDER BY quintile
            """).df()
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_quintiles(quintile_df: pd.DataFrame, target: str, out_path: Path) -> None:
    if quintile_df.empty:
        return
    features = quintile_df["feature"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1, len(features), figsize=(4.5 * len(features), 4), constrained_layout=True
    )
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features, strict=True):
        sub = quintile_df[quintile_df["feature"] == feature]
        ax.plot(sub["quintile"], sub[f"avg_{target}"], marker="o", linewidth=1.5)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(str(FEATURES[feature]["label"]))
        ax.set_xlabel("Quintile")
        ax.set_ylabel(f"Avg {target}")
        ax.set_xticks([1, 2, 3, 4, 5])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_regressions(con: object, features: list[str], target: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        f = quote_ident(feature)
        df = con.execute(f"""
            SELECT {f} AS x, {target} AS y
            FROM fin_price_corr_base
            WHERE {f} IS NOT NULL AND {target} IS NOT NULL
            """).df()
        if len(df) < 3:
            continue
        model = LinearRegression()
        x = df[["x"]].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        model.fit(x, y)
        rows.append(
            {
                "feature": feature,
                "target": target,
                "n": len(df),
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "r2": float(model.score(x, y)),
            }
        )
    return pd.DataFrame(rows)


def compute_group_stability(con: object, features: list[str], target: str) -> pd.DataFrame:
    frames = []
    for feature in features:
        f = quote_ident(feature)
        df = con.execute(f"""
            SELECT
                '{feature}' AS feature,
                year,
                market,
                count(*) AS n,
                corr({f}, {target}) AS pearson
            FROM fin_price_corr_base
            WHERE {f} IS NOT NULL AND {target} IS NOT NULL
            GROUP BY year, market
            HAVING count(*) >= 100
            ORDER BY feature, year, market
            """).df()
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_summary(
    ctx: RunContext,
    base_stats: tuple[int, str, str, int],
    corr_df: pd.DataFrame,
    regressions: pd.DataFrame,
    top: list[str],
) -> None:
    primary = corr_df[corr_df["target"] == ctx.primary_target].copy()
    primary["abs_spearman"] = primary["spearman"].abs()
    primary = primary.sort_values("abs_spearman", ascending=False)
    best = primary.iloc[0]
    max_r2 = regressions["r2"].max() if not regressions.empty else np.nan
    top_list = ", ".join(top)
    conclusion = (
        f"Primary target `{ctx.primary_target}` 기준 최상위 단일 재무지표는 "
        f"`{best['feature']}`이며 Spearman={best['spearman']:.4f}, "
        f"Pearson={best['pearson']:.4f}입니다. 상위 {len(top)}개 지표({top_list})의 "
        f"단순 선형회귀 R^2 최대값은 {max_r2:.6f}로, 현재 스냅샷에서는 재무비율 단독의 "
        "선형 설명력은 매우 약합니다. 분위수/그룹 안정성 CSV를 함께 보며 특정 구간에서만 "
        "나타나는 신호인지 확인해야 합니다."
    )
    summary = f"""# fin_vs_price_corr summary

- snapshot_date: {ctx.snapshot_date}
- source: {ctx.source}
- base rows: {base_stats[0]:,}
- date range: {base_stats[1]} ~ {base_stats[2]}
- tickers: {base_stats[3]:,}
- primary_target: {ctx.primary_target}

## Conclusion

{conclusion}

## Outputs

- `corr_table.csv`
- `corr_heatmap.png`
- `scatter_top3.png`
- `quintile_returns.csv`
- `quintile_returns.png`
- `regression_summary.csv`
- `group_stability.csv`
- `winsor_thresholds.csv`
"""
    (ctx.report_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = DataRoot.resolve(market="kr")
    report_root = root.output / "reports" / "analysis" / "fin_vs_price_corr"
    report_dir = report_root / args.report_date
    report_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        snapshot_date=args.snapshot_date,
        source=args.source,
        report_date=args.report_date,
        report_dir=report_dir,
        primary_target=args.primary_target,
    )
    config = LakeConfig(root=root, snapshot_date=args.snapshot_date, source=args.source)
    con = setup_connection(config, force_label=args.force_label)
    winsor_df = create_analysis_views(con)

    base_stats = con.execute("""
        SELECT
            count(*) AS n,
            min(trade_date)::VARCHAR AS min_trade_date,
            max(trade_date)::VARCHAR AS max_trade_date,
            count(DISTINCT ticker) AS tickers
        FROM fin_price_corr_base
        """).fetchone()

    corr_df = compute_correlations(con)
    top = top_features(corr_df, args.primary_target, args.top_n)
    quintile_df = compute_quintiles(con, top, args.primary_target)
    regressions = compute_regressions(con, top, args.primary_target)
    stability = compute_group_stability(con, top, args.primary_target)

    winsor_df.to_csv(report_dir / "winsor_thresholds.csv", index=False)
    corr_df.to_csv(report_dir / "corr_table.csv", index=False)
    quintile_df.to_csv(report_dir / "quintile_returns.csv", index=False)
    regressions.to_csv(report_dir / "regression_summary.csv", index=False)
    stability.to_csv(report_dir / "group_stability.csv", index=False)

    plot_corr_heatmap(corr_df, report_dir / "corr_heatmap.png")
    plot_scatter_top(
        con, top, args.primary_target, args.scatter_sample, report_dir / "scatter_top3.png"
    )
    plot_quintiles(quintile_df, args.primary_target, report_dir / "quintile_returns.png")
    write_summary(ctx, base_stats, corr_df, regressions, top)

    print(f"Wrote {report_dir}")
    print(f"Top features for {args.primary_target}: {', '.join(top)}")


if __name__ == "__main__":
    main()
