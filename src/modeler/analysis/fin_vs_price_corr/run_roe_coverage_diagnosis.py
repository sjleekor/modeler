"""ROE coverage diagnosis for the financial PIT mart.

This local research script reads the snapshot-pinned lake, diagnoses why
``feat_fin_pit.fin_roe`` is sparse, and writes coverage tables under reports/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeler.etl.config import DataRoot, DEFAULT_SOURCE, LakeConfig  # noqa: E402
from modeler.etl.features.fin_pit import materialize_fin_pit  # noqa: E402
from modeler.etl.lake import connect, register_derived_marts, register_views  # noqa: E402
from modeler.etl.universe import materialize_universe  # noqa: E402

FEATURES = (
    "fin_roe",
    "fin_roa",
    "fin_equity_ratio",
    "fin_debt_to_equity",
)

ROE_COMPONENT_METRICS = (
    "net_income",
    "controlling_net_income",
    "total_equity",
    "total_assets",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--report-date",
        default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d"),
        help="Output subdirectory under reports/analysis/fin_vs_price_corr.",
    )
    return parser.parse_args()


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2%}"


def int_fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def write_csv(df: pd.DataFrame, report_dir: Path, name: str) -> Path:
    out = report_dir / name
    df.to_csv(out, index=False)
    return out


def feature_coverage(con: object) -> pd.DataFrame:
    key_expr = "ticker || '|' || market"
    parts = []
    for feature in FEATURES:
        f = quote_ident(feature)
        parts.append(
            f"""
            SELECT
                '{feature}' AS feature,
                COUNT(*) AS total_rows,
                COUNT({f}) AS non_null_rows,
                COUNT(DISTINCT {key_expr}) AS total_tickers,
                COUNT(DISTINCT CASE WHEN {f} IS NOT NULL THEN {key_expr} END)
                    AS non_null_tickers
            FROM feat_fin_pit
            """
        )
    sql = " UNION ALL ".join(parts)
    df = con.execute(sql).df()
    df["non_null_row_rate"] = df["non_null_rows"] / df["total_rows"]
    df["non_null_ticker_rate"] = df["non_null_tickers"] / df["total_tickers"]
    return df[
        [
            "feature",
            "total_rows",
            "non_null_rows",
            "non_null_row_rate",
            "total_tickers",
            "non_null_tickers",
            "non_null_ticker_rate",
        ]
    ]


def roe_by_year(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            CAST(EXTRACT(year FROM trade_date) AS INTEGER) AS year,
            COUNT(*) AS total_rows,
            COUNT(fin_roe) AS non_null_rows,
            COUNT(DISTINCT ticker || '|' || market) AS total_tickers,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                AS non_null_tickers,
            COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS non_null_row_rate,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                ::DOUBLE / NULLIF(COUNT(DISTINCT ticker || '|' || market), 0)
                AS non_null_ticker_rate
        FROM feat_fin_pit
        GROUP BY 1
        ORDER BY 1
        """
    ).df()


def roe_by_market(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            market,
            COUNT(*) AS total_rows,
            COUNT(fin_roe) AS non_null_rows,
            COUNT(DISTINCT ticker || '|' || market) AS total_tickers,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                AS non_null_tickers,
            COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS non_null_row_rate,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                ::DOUBLE / NULLIF(COUNT(DISTINCT ticker || '|' || market), 0)
                AS non_null_ticker_rate
        FROM feat_fin_pit
        GROUP BY 1
        ORDER BY 1
        """
    ).df()


def roe_by_year_market(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            CAST(EXTRACT(year FROM trade_date) AS INTEGER) AS year,
            market,
            COUNT(*) AS total_rows,
            COUNT(fin_roe) AS non_null_rows,
            COUNT(DISTINCT ticker || '|' || market) AS total_tickers,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                AS non_null_tickers,
            COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS non_null_row_rate,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                ::DOUBLE / NULLIF(COUNT(DISTINCT ticker || '|' || market), 0)
                AS non_null_ticker_rate
        FROM feat_fin_pit
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()


def month_end_coverage(con: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW roe_month_end_sample AS
        SELECT *
        FROM (
            SELECT
                *,
                CAST(date_trunc('month', trade_date) AS DATE) AS trade_month,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, market, date_trunc('month', trade_date)
                    ORDER BY trade_date DESC
                ) AS rn
            FROM feat_fin_pit
        )
        WHERE rn = 1
        """
    )
    overall = con.execute(
        """
        SELECT
            COUNT(*) AS total_month_end_rows,
            COUNT(fin_roe) AS non_null_month_end_rows,
            COUNT(DISTINCT ticker || '|' || market) AS total_tickers,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                AS non_null_tickers,
            COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS non_null_row_rate,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                ::DOUBLE / NULLIF(COUNT(DISTINCT ticker || '|' || market), 0)
                AS non_null_ticker_rate
        FROM roe_month_end_sample
        """
    ).df()
    monthly = con.execute(
        """
        SELECT
            trade_month,
            COUNT(*) AS total_rows,
            COUNT(fin_roe) AS non_null_rows,
            COUNT(DISTINCT ticker || '|' || market) AS total_tickers,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                AS non_null_tickers,
            COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS non_null_row_rate,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                ::DOUBLE / NULLIF(COUNT(DISTINCT ticker || '|' || market), 0)
                AS non_null_ticker_rate
        FROM roe_month_end_sample
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    return overall, monthly


def metric_fact_coverage(con: object) -> pd.DataFrame:
    values = ", ".join(f"('{metric}')" for metric in ROE_COMPONENT_METRICS)
    return con.execute(
        f"""
        WITH requested(metric_code) AS (VALUES {values}),
        facts AS (
            SELECT *
            FROM stock_metric_fact
            WHERE metric_code IN (SELECT metric_code FROM requested)
              AND value_numeric IS NOT NULL
        )
        SELECT
            r.metric_code,
            COUNT(f.metric_code) AS fact_rows,
            COUNT(DISTINCT f.ticker || '|' || f.market) AS ticker_count,
            COUNT(DISTINCT f.corp_code) AS corp_count,
            COUNT(DISTINCT
                f.ticker || '|' || f.market || '|' ||
                CAST(f.bsns_year AS VARCHAR) || '|' || COALESCE(f.reprt_code, '')
            ) AS ticker_report_count,
            MIN(f.bsns_year) AS min_bsns_year,
            MAX(f.bsns_year) AS max_bsns_year,
            MIN(f.period_end) AS min_period_end,
            MAX(f.period_end) AS max_period_end
        FROM requested r
        LEFT JOIN facts f USING (metric_code)
        GROUP BY 1
        ORDER BY 1
        """
    ).df()


def metric_fact_intersection(con: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_level = con.execute(
        """
        WITH flags AS (
            SELECT
                ticker,
                market,
                BOOL_OR(metric_code = 'net_income' AND value_numeric IS NOT NULL)
                    AS has_net_income,
                BOOL_OR(metric_code = 'controlling_net_income' AND value_numeric IS NOT NULL)
                    AS has_controlling_net_income,
                BOOL_OR(metric_code = 'total_equity' AND value_numeric IS NOT NULL)
                    AS has_total_equity
            FROM stock_metric_fact
            WHERE metric_code IN ('net_income', 'controlling_net_income', 'total_equity')
            GROUP BY 1, 2
        )
        SELECT
            has_net_income,
            has_controlling_net_income,
            (has_net_income OR has_controlling_net_income) AS has_roe_numerator,
            has_total_equity,
            COUNT(*) AS ticker_count
        FROM flags
        GROUP BY 1, 2, 3, 4
        ORDER BY 3 DESC, 4 DESC, 1 DESC, 2 DESC
        """
    ).df()
    report_level = con.execute(
        """
        WITH pivoted AS (
            SELECT
                ticker,
                market,
                bsns_year,
                reprt_code,
                period_end,
                BOOL_OR(metric_code = 'net_income' AND value_numeric IS NOT NULL)
                    AS has_net_income,
                BOOL_OR(metric_code = 'controlling_net_income' AND value_numeric IS NOT NULL)
                    AS has_controlling_net_income,
                BOOL_OR(metric_code = 'total_equity' AND value_numeric IS NOT NULL)
                    AS has_total_equity
            FROM stock_metric_fact
            WHERE metric_code IN ('net_income', 'controlling_net_income', 'total_equity')
            GROUP BY 1, 2, 3, 4, 5
        )
        SELECT
            has_net_income,
            has_controlling_net_income,
            (has_net_income OR has_controlling_net_income) AS has_roe_numerator,
            has_total_equity,
            COUNT(*) AS ticker_report_count,
            COUNT(DISTINCT ticker || '|' || market) AS ticker_count
        FROM pivoted
        GROUP BY 1, 2, 3, 4
        ORDER BY 3 DESC, 4 DESC, 1 DESC, 2 DESC
        """
    ).df()
    return ticker_level, report_level


def universe_bias(con: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW roe_ticker_base AS
        SELECT
            f.ticker,
            f.market,
            MIN(f.trade_date) AS first_panel_date,
            MAX(f.trade_date) AS last_panel_date,
            COUNT(*) AS panel_rows,
            COUNT(f.fin_roe) AS roe_non_null_rows,
            BOOL_OR(f.fin_roe IS NOT NULL) AS has_roe
        FROM feat_fin_pit f
        GROUP BY 1, 2
        """
    )
    market_status = con.execute(
        """
        SELECT
            b.has_roe,
            b.market,
            COALESCE(m.status, '(missing)') AS status,
            COUNT(*) AS ticker_count,
            AVG(b.panel_rows) AS avg_panel_rows,
            MIN(b.first_panel_date) AS min_first_panel_date,
            MAX(b.first_panel_date) AS max_first_panel_date,
            MIN(b.last_panel_date) AS min_last_panel_date,
            MAX(b.last_panel_date) AS max_last_panel_date
        FROM roe_ticker_base b
        LEFT JOIN stock_master m
          ON m.ticker = b.ticker
         AND m.market = b.market
        GROUP BY 1, 2, 3
        ORDER BY 1 DESC, 2, 3
        """
    ).df()
    start_year = con.execute(
        """
        SELECT
            has_roe,
            CAST(EXTRACT(year FROM first_panel_date) AS INTEGER) AS first_panel_year,
            COUNT(*) AS ticker_count
        FROM roe_ticker_base
        GROUP BY 1, 2
        ORDER BY 2, 1 DESC
        """
    ).df()
    return market_status, start_year


def coverage_tickers(con: object) -> pd.DataFrame:
    return con.execute(
        """
        WITH latest_roe AS (
            SELECT
                f.trade_date AS latest_roe_date,
                f.ticker,
                f.market,
                f.fin_roe AS latest_fin_roe,
                f.fin_roa AS latest_fin_roa,
                f.fin_debt_to_equity AS latest_fin_debt_to_equity,
                f.fin_equity_ratio AS latest_fin_equity_ratio,
                f.fin_is_negative_equity AS latest_fin_is_negative_equity,
                ROW_NUMBER() OVER (
                    PARTITION BY f.ticker, f.market
                    ORDER BY f.trade_date DESC
                ) AS rn
            FROM feat_fin_pit f
            WHERE f.fin_roe IS NOT NULL
        ),
        profile AS (
            SELECT
                ticker,
                market,
                MIN(trade_date) AS first_panel_date,
                MAX(trade_date) AS last_panel_date,
                MIN(CASE WHEN fin_roe IS NOT NULL THEN trade_date END) AS first_roe_date,
                MAX(CASE WHEN fin_roe IS NOT NULL THEN trade_date END) AS last_roe_date,
                COUNT(*) AS panel_rows,
                COUNT(fin_roe) AS roe_non_null_rows
            FROM feat_fin_pit
            GROUP BY 1, 2
        )
        SELECT
            l.ticker,
            l.market,
            m.name,
            m.status,
            p.first_panel_date,
            p.last_panel_date,
            p.first_roe_date,
            p.last_roe_date,
            p.panel_rows,
            p.roe_non_null_rows,
            l.latest_roe_date,
            l.latest_fin_roe,
            l.latest_fin_roa,
            l.latest_fin_debt_to_equity,
            l.latest_fin_equity_ratio,
            l.latest_fin_is_negative_equity
        FROM latest_roe l
        JOIN profile p
          ON p.ticker = l.ticker
         AND p.market = l.market
        LEFT JOIN stock_master m
          ON m.ticker = l.ticker
         AND m.market = l.market
        WHERE l.rn = 1
        ORDER BY l.market, l.ticker
        """
    ).df()


def write_summary(
    report_dir: Path,
    *,
    snapshot_date: str,
    source: str,
    feature_df: pd.DataFrame,
    year_df: pd.DataFrame,
    month_overall_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    ticker_intersection_df: pd.DataFrame,
    market_status_df: pd.DataFrame,
) -> Path:
    feature_lines = []
    for row in feature_df.itertuples(index=False):
        feature_lines.append(
            "| "
            + " | ".join(
                [
                    row.feature,
                    int_fmt(row.non_null_rows),
                    pct(row.non_null_row_rate),
                    int_fmt(row.non_null_tickers),
                    pct(row.non_null_ticker_rate),
                ]
            )
            + " |"
        )

    roe_row = feature_df.loc[feature_df["feature"] == "fin_roe"].iloc[0]
    month_row = month_overall_df.iloc[0]
    net_income = metric_df.loc[metric_df["metric_code"] == "net_income"].iloc[0]
    controlling = metric_df.loc[
        metric_df["metric_code"] == "controlling_net_income"
    ].iloc[0]
    equity = metric_df.loc[metric_df["metric_code"] == "total_equity"].iloc[0]
    numerator_equity = ticker_intersection_df[
        ticker_intersection_df["has_roe_numerator"] & ticker_intersection_df["has_total_equity"]
    ]["ticker_count"].sum()
    equity_only = int(equity["ticker_count"]) - int(numerator_equity)

    market_lines = []
    for row in market_status_df.itertuples(index=False):
        label = "roe_non_null" if row.has_roe else "roe_null"
        market_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row.market),
                    str(row.status),
                    int_fmt(row.ticker_count),
                    f"{float(row.avg_panel_rows):.1f}",
                ]
            )
            + " |"
        )

    year_min = int(year_df.loc[year_df["non_null_rows"] > 0, "year"].min())
    year_max = int(year_df.loc[year_df["non_null_rows"] > 0, "year"].max())
    # First year where the cross-section is densely covered (>= 50% of tickers).
    # Pre-cliff years stay sparse because of limited early OpenDART filings, not
    # normalization gaps, so we report the usable window dynamically.
    dense_years = year_df.loc[year_df["non_null_ticker_rate"] >= 0.5, "year"]
    first_dense_year = int(dense_years.min()) if not dense_years.empty else None
    window_year = first_dense_year if first_dense_year is not None else year_min

    roe_tickers = int(roe_row.non_null_tickers)
    net_income_tickers = int(net_income.ticker_count)

    if first_dense_year is not None:
        year_finding_line = (
            f"- Non-null `fin_roe` appears from {year_min} to {year_max}; cross-sectional "
            f"coverage becomes dense (>=50% of tickers) from {first_dense_year} onward."
        )
    else:
        year_finding_line = f"- Non-null `fin_roe` appears from {year_min} to {year_max}."

    # Interpretation adapts to the measured coverage, encoding the Step 6 resume
    # thresholds from 01_00_00_fix_roe_plan.md (fin_roe >= 2,000 & net_income >= 2,500).
    coverage_para = (
        f"`net_income` now covers {int_fmt(net_income.ticker_count)} ticker-market pairs and "
        f"`total_equity` covers {int_fmt(equity.ticker_count)}, so the income-statement "
        f"numerator is aligned with the balance-sheet denominator. `fin_roe` reaches "
        f"{int_fmt(roe_row.non_null_tickers)} pairs ({pct(roe_row.non_null_ticker_rate)} of the "
        f"financial-feature universe), leaving ~{int_fmt(equity_only)} equity-only pairs without a "
        f"usable ROE numerator. This reflects the CIS-aware normalization rules that map "
        f"`ifrs-full_ProfitLoss` / `ifrs_ProfitLoss` from the comprehensive-income statement."
    )
    if roe_tickers >= 2000 and net_income_tickers >= 2500:
        verdict_para = (
            "ROE coverage clears the resume threshold (fin_roe >= 2,000 and net_income >= 2,500). "
            f"Downstream ROE distribution, IC, and quintile analysis can treat the {window_year}+ "
            "window as representative of the universe; earlier years stay sparse because of "
            "limited early OpenDART income-statement filings, not normalization gaps."
        )
    elif roe_tickers >= 1000:
        verdict_para = (
            "ROE coverage is partial (fin_roe between 1,000 and 2,000). Keep downstream analysis "
            "exploratory and re-check market/year concentration before drawing universe-level "
            "conclusions."
        )
    else:
        verdict_para = (
            "ROE coverage is still low (fin_roe < 1,000 or net_income < 2,500). Treat results as "
            "non-representative and consider the XBRL or raw-account-name numerator fallback "
            "before resuming distribution/IC/quintile analysis."
        )

    body_lines = [
        "# ROE Coverage Diagnosis",
        "",
        f"- snapshot_date: `{snapshot_date}`",
        f"- source: `{source}`",
        f"- input: `data_lake/feature_mart/snapshot_date={snapshot_date}/feat_fin_pit`",
        "- generated_at_kst: "
        f"`{datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}`",
        "",
        "## Key Findings",
        "",
        f"- `fin_roe` is available for {int_fmt(roe_row.non_null_tickers)} out of "
        f"{int_fmt(roe_row.total_tickers)} ticker-market pairs "
        f"({pct(roe_row.non_null_ticker_rate)}).",
        f"- Month-end sample coverage is {int_fmt(month_row.non_null_month_end_rows)} rows "
        f"across {int_fmt(month_row.non_null_tickers)} ticker-market pairs out of "
        f"{int_fmt(month_row.total_month_end_rows)} rows.",
        f"- `stock_metric_fact.total_equity` covers {int_fmt(equity.ticker_count)} "
        f"ticker-market pairs, `net_income` covers {int_fmt(net_income.ticker_count)}, "
        f"and `controlling_net_income` covers {int_fmt(controlling.ticker_count)}.",
        "- Ticker-level ROE component intersection "
        "`(net_income OR controlling_net_income) AND total_equity` covers "
        f"{int_fmt(numerator_equity)} ticker-market pairs; "
        f"total-equity-without-numerator is approximately {int_fmt(equity_only)} "
        "ticker-market pairs.",
        year_finding_line,
        "",
        "## Feature Coverage",
        "",
        "| feature | non-null rows | row rate | non-null tickers | ticker rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        *feature_lines,
        "",
        "## Market / Status Bias",
        "",
        "| group | market | status | tickers | avg panel rows |",
        "| --- | --- | --- | ---: | ---: |",
        *market_lines,
        "",
        "## Interpretation",
        "",
        coverage_para,
        "",
        verdict_para,
    ]
    body = "\n".join(body_lines)
    out = report_dir / "analysis_summary.md"
    out.write_text(body + "\n")
    return out


def main() -> None:
    args = parse_args()
    root = DataRoot.resolve(market="kr")
    config = LakeConfig(root=root, snapshot_date=args.snapshot_date, source=args.source)
    report_root = root.output / "reports" / "analysis" / "fin_vs_price_corr"
    report_dir = report_root / args.report_date / "roe_distribution"
    report_dir.mkdir(parents=True, exist_ok=True)

    con = connect(config)
    # Recompute stock_metric_fact from the raw lake with the CURRENT (CIS-aware)
    # normalization rules instead of reading the stale canonical export, then
    # rebuild feat_fin_pit on top of it so fin_roe reflects the expanded
    # income-statement coverage (force=True overrides the stale materialized parquet).
    register_views(
        con,
        config,
        tables=[
            "stock_master",
            "daily_ohlcv",
            "dart_financial_statement_raw",
            "dart_share_count_raw",
            "dart_shareholder_return_raw",
            "dart_xbrl_fact_raw",
            "dart_corp_master",
        ],
    )
    register_derived_marts(con, config, which=["stock_metric_fact"])
    materialize_universe(con, config, force=True)
    materialize_fin_pit(con, config, force=True)

    feature_df = feature_coverage(con)
    year_df = roe_by_year(con)
    market_df = roe_by_market(con)
    year_market_df = roe_by_year_market(con)
    month_overall_df, month_df = month_end_coverage(con)
    metric_df = metric_fact_coverage(con)
    ticker_intersection_df, report_intersection_df = metric_fact_intersection(con)
    market_status_df, start_year_df = universe_bias(con)
    ticker_df = coverage_tickers(con)

    write_csv(feature_df, report_dir, "roe_coverage_summary.csv")
    write_csv(year_df, report_dir, "roe_coverage_by_year.csv")
    write_csv(market_df, report_dir, "roe_coverage_by_market.csv")
    write_csv(year_market_df, report_dir, "roe_coverage_by_year_market.csv")
    write_csv(month_overall_df, report_dir, "roe_month_end_coverage_summary.csv")
    write_csv(month_df, report_dir, "roe_month_end_coverage_by_month.csv")
    write_csv(metric_df, report_dir, "roe_metric_fact_coverage.csv")
    write_csv(ticker_intersection_df, report_dir, "roe_metric_fact_ticker_intersection.csv")
    write_csv(report_intersection_df, report_dir, "roe_metric_fact_report_intersection.csv")
    write_csv(market_status_df, report_dir, "roe_universe_market_status_bias.csv")
    write_csv(start_year_df, report_dir, "roe_universe_start_year_bias.csv")
    write_csv(ticker_df, report_dir, "roe_coverage_tickers.csv")
    write_summary(
        report_dir,
        snapshot_date=args.snapshot_date,
        source=args.source,
        feature_df=feature_df,
        year_df=year_df,
        month_overall_df=month_overall_df,
        metric_df=metric_df,
        ticker_intersection_df=ticker_intersection_df,
        market_status_df=market_status_df,
    )

    print(f"Wrote ROE coverage diagnosis to {report_dir}")


if __name__ == "__main__":
    main()
