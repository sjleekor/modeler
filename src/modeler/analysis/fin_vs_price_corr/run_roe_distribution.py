"""ROE distribution analysis for the financial PIT mart (plan 01_00_roe_distribution).

Recomputes ``stock_metric_fact`` from the raw lake with the current (CIS-aware)
normalization rules, rebuilds ``feat_fin_pit``, then profiles ``fin_roe`` along
the axes of plan ``01_00_roe_distribution_plan.md`` §5:

    5.1 overall distribution (month-end panel + company-latest samples)
    5.2 time-varying cross-sectional distribution
    5.3 company-level ROE level / volatility / value-change profile
    5.4 rank persistence (12m calendar lag + filing-to-filing value-change)
    5.5 market (KOSPI/KOSDAQ) distribution
    5.7 outliers / data-quality cases
    (5.6 sector analysis is out of scope: no sector mapping in this snapshot)

Outputs CSV tables + 3 PNG figures + analysis_summary.md under
``reports/analysis/fin_vs_price_corr/<report-date>/roe_distribution/``.

Sampling principles (plan §4): the daily mart repeats the same filing across many
trade days, so distribution work uses a *month-end panel* (last trade day per
ticker/market/month) and a *company-latest* (last non-null-ROE row per name)
sample. ``fin_has_fs = 0`` names are excluded from the ROE distribution;
``fin_is_negative_equity = 1`` is already ROE-null in the mart (clipped).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeler.etl.config import DEFAULT_SOURCE, DataRoot, LakeConfig  # noqa: E402
from modeler.etl.features.fin_pit import materialize_fin_pit  # noqa: E402
from modeler.etl.lake import connect, register_derived_marts, register_views  # noqa: E402
from modeler.etl.universe import materialize_universe  # noqa: E402

# Persistence horizons (calendar months). 1m is reported as an artifact reference
# only (the mart repeats annual filings, so adjacent months are largely identical).
PERSISTENCE_HORIZONS = (1, 3, 6, 12)


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


def write_csv(df: pd.DataFrame, report_dir: Path, name: str) -> Path:
    out = report_dir / name
    df.to_csv(out, index=False)
    return out


# --- reproducible setup ------------------------------------------------------


def setup_views(con: object, config: LakeConfig) -> None:
    """Recompute SMF from raw (current CIS rules) and rebuild feat_fin_pit.

    Mirrors run_roe_coverage_diagnosis.py: read the snapshot-pinned raw lake,
    recompute the derived facts in DuckDB rather than reading the stale canonical
    export, and force-rebuild the PIT mart so fin_roe reflects expanded coverage.
    """
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


def build_samples(con: object) -> None:
    """Create the two analysis samples + derived ROE columns as TEMP views.

    ``roe_me_all`` — month-end sample over names with a balance sheet
    (fin_has_fs), ROE nullable, for missing-rate / coverage-by-month.
    ``roe_me`` — the non-null-ROE distribution sample with winsorized value,
    cross-sectional percentile ranks, and flag columns.
    ``roe_latest`` — last non-null-ROE row per (ticker, market) for company
    distribution and outlier inspection.
    """
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW roe_me_all AS
        SELECT *, CAST(date_trunc('month', trade_date) AS DATE) AS trade_month
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, market, date_trunc('month', trade_date)
                    ORDER BY trade_date DESC
                ) AS _rn
            FROM feat_fin_pit
            WHERE fin_has_fs
        )
        WHERE _rn = 1
        """
    )
    # Winsorized column uses a GLOBAL p1/p99 (plan §4: monthly winsor is unstable).
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW roe_me AS
        WITH base AS (
            SELECT ticker, market, trade_date, trade_month,
                   fin_roe, fin_roa, fin_debt_to_equity, fin_equity_ratio,
                   fin_is_negative_equity
            FROM roe_me_all
            WHERE fin_roe IS NOT NULL
        ),
        thr AS (
            SELECT quantile_cont(fin_roe, 0.01) AS p1,
                   quantile_cont(fin_roe, 0.99) AS p99
            FROM base
        )
        SELECT
            base.*,
            LEAST(GREATEST(base.fin_roe, thr.p1), thr.p99) AS roe_winsor_1p,
            PERCENT_RANK() OVER (
                PARTITION BY base.trade_month, base.market ORDER BY base.fin_roe
            ) AS roe_rank_market,
            PERCENT_RANK() OVER (
                PARTITION BY base.trade_month ORDER BY base.fin_roe
            ) AS roe_rank_all,
            (base.fin_roe < 0) AS is_roe_negative,
            (base.fin_roe > 1.0) AS is_roe_extreme_high,
            (base.fin_roe < -0.5) AS is_roe_extreme_low,
            (base.fin_roe > 0.3) AS is_roe_high,
            (base.fin_roe <> LAG(base.fin_roe) OVER (
                PARTITION BY base.ticker, base.market ORDER BY base.trade_month
            )) AS roe_value_changed
        FROM base CROSS JOIN thr
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW roe_latest AS
        SELECT * EXCLUDE (_rn) FROM (
            SELECT f.ticker, f.market, f.trade_date,
                   f.fin_roe, f.fin_roa, f.fin_debt_to_equity,
                   f.fin_equity_ratio, f.fin_is_negative_equity,
                   m.name, m.status,
                   ROW_NUMBER() OVER (
                       PARTITION BY f.ticker, f.market ORDER BY f.trade_date DESC
                   ) AS _rn
            FROM feat_fin_pit f
            LEFT JOIN stock_master m
              ON m.ticker = f.ticker AND m.market = f.market
            WHERE f.fin_roe IS NOT NULL
        )
        WHERE _rn = 1
        """
    )


# --- §5.1 overall distribution ----------------------------------------------

_DIST_STATS = """
    COUNT(*) AS rows,
    COUNT(DISTINCT ticker || '|' || market) AS tickers,
    AVG(fin_roe) AS mean,
    median(fin_roe) AS median,
    stddev_samp(fin_roe) AS std,
    MIN(fin_roe) AS min,
    MAX(fin_roe) AS max,
    quantile_cont(fin_roe, 0.01) AS p1,
    quantile_cont(fin_roe, 0.05) AS p5,
    quantile_cont(fin_roe, 0.10) AS p10,
    quantile_cont(fin_roe, 0.25) AS p25,
    quantile_cont(fin_roe, 0.75) AS p75,
    quantile_cont(fin_roe, 0.90) AS p90,
    quantile_cont(fin_roe, 0.95) AS p95,
    quantile_cont(fin_roe, 0.99) AS p99,
    AVG(CASE WHEN fin_roe < 0 THEN 1.0 ELSE 0 END) AS neg_ratio,
    AVG(CASE WHEN fin_roe > 0.3 THEN 1.0 ELSE 0 END) AS gt_30_ratio,
    AVG(CASE WHEN fin_roe > 0.5 THEN 1.0 ELSE 0 END) AS gt_50_ratio,
    AVG(CASE WHEN fin_roe > 1.0 THEN 1.0 ELSE 0 END) AS gt_100_ratio,
    AVG(CASE WHEN fin_roe < -0.5 THEN 1.0 ELSE 0 END) AS lt_neg50_ratio,
    AVG(CASE WHEN fin_roe < -1.0 THEN 1.0 ELSE 0 END) AS lt_neg100_ratio
"""


def overall_distribution(con: object) -> pd.DataFrame:
    me = con.execute(f"SELECT 'month_end' AS sample, {_DIST_STATS} FROM roe_me").df()
    latest = con.execute(
        f"SELECT 'company_latest' AS sample, {_DIST_STATS} FROM roe_latest"
    ).df()
    return pd.concat([me, latest], ignore_index=True)


def winsor_thresholds(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            quantile_cont(fin_roe, 0.01) AS global_p1,
            quantile_cont(fin_roe, 0.99) AS global_p99,
            MIN(roe_winsor_1p) AS winsor_min,
            MAX(roe_winsor_1p) AS winsor_max
        FROM roe_me
        """
    ).df()


# --- §5.7 numerator-period caveat: annual-only ROE baseline ------------------
#
# feat_fin_pit.fin_roe divides the latest *available* net_income by equity, but
# the latest available filing is a partial-year YTD figure (Q1/half/Q3) for ~3/4
# of the calendar. The numerator is therefore un-annualized for most trade dates,
# which systematically understates the ROE level (median ~1% vs ~2.6% annual).
# This baseline recomputes ROE from ANNUAL (reprt_code 11011) filings only, so the
# distribution work can compare the mixed-period mart ROE against a clean annual ROE.


def annual_roe_baseline(con: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Company-latest + per-year ROE from annual (11011) filings only.

    Reads the recomputed ``stock_metric_fact`` view directly (not the PIT mart)
    and uses ``COALESCE(controlling_net_income, net_income) / total_equity`` to
    match the ``fin_pit`` ROE definition, restricted to annual reports.
    """
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW annual_roe AS
        WITH ni AS (
            SELECT ticker, bsns_year,
                   COALESCE(
                       MAX(CASE WHEN metric_code='controlling_net_income'
                                THEN value_numeric END),
                       MAX(CASE WHEN metric_code='net_income' THEN value_numeric END)
                   ) AS numer,
                   MAX(CASE WHEN metric_code='total_equity' THEN value_numeric END) AS equity
            FROM stock_metric_fact
            WHERE reprt_code = '11011'
              AND metric_code IN ('net_income', 'controlling_net_income', 'total_equity')
            GROUP BY ticker, bsns_year
        )
        SELECT ticker, bsns_year, numer / NULLIF(equity, 0) AS roe_annual
        FROM ni
        WHERE equity > 0 AND numer IS NOT NULL
        """
    )
    company_latest = con.execute(
        """
        WITH latest AS (
            SELECT ticker, bsns_year, roe_annual,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY bsns_year DESC) AS rn
            FROM annual_roe
        )
        SELECT
            'annual_company_latest' AS sample,
            COUNT(*) AS tickers,
            median(roe_annual) AS median,
            AVG(roe_annual) AS mean,
            quantile_cont(roe_annual, 0.10) AS p10,
            quantile_cont(roe_annual, 0.25) AS p25,
            quantile_cont(roe_annual, 0.75) AS p75,
            quantile_cont(roe_annual, 0.90) AS p90,
            AVG(CASE WHEN roe_annual < 0 THEN 1.0 ELSE 0 END) AS neg_ratio
        FROM latest WHERE rn = 1
        """
    ).df()
    by_year = con.execute(
        """
        SELECT bsns_year,
               COUNT(*) AS tickers,
               median(roe_annual) AS median,
               quantile_cont(roe_annual, 0.25) AS p25,
               quantile_cont(roe_annual, 0.75) AS p75,
               AVG(CASE WHEN roe_annual < 0 THEN 1.0 ELSE 0 END) AS neg_ratio
        FROM annual_roe
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    return company_latest, by_year


# --- §5.2 time-varying distribution -----------------------------------------


def monthly_distribution(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            trade_month,
            COUNT(*) AS total_rows,
            COUNT(fin_roe) AS roe_rows,
            COUNT(DISTINCT CASE WHEN fin_roe IS NOT NULL THEN ticker || '|' || market END)
                AS roe_tickers,
            1 - COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS missing_rate,
            median(fin_roe) AS median,
            quantile_cont(fin_roe, 0.10) AS p10,
            quantile_cont(fin_roe, 0.25) AS p25,
            quantile_cont(fin_roe, 0.75) AS p75,
            quantile_cont(fin_roe, 0.90) AS p90,
            AVG(CASE WHEN fin_roe < 0 THEN 1.0 ELSE 0 END) AS neg_ratio,
            AVG(CASE WHEN fin_is_negative_equity THEN 1.0 ELSE 0 END) AS neg_equity_ratio
        FROM roe_me_all
        GROUP BY 1
        ORDER BY 1
        """
    ).df()


def monthly_distribution_by_market(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            trade_month, market,
            COUNT(*) AS total_rows,
            COUNT(fin_roe) AS roe_rows,
            1 - COUNT(fin_roe)::DOUBLE / NULLIF(COUNT(*), 0) AS missing_rate,
            median(fin_roe) AS median,
            quantile_cont(fin_roe, 0.25) AS p25,
            quantile_cont(fin_roe, 0.75) AS p75,
            AVG(CASE WHEN fin_roe < 0 THEN 1.0 ELSE 0 END) AS neg_ratio
        FROM roe_me_all
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()


# --- §5.3 company-level profile ---------------------------------------------


def company_profile(con: object) -> pd.DataFrame:
    return con.execute(
        """
        WITH prof AS (
            SELECT
                e.ticker, e.market,
                COUNT(*) AS obs_months,
                AVG(e.fin_roe) AS roe_mean,
                median(e.fin_roe) AS roe_median,
                stddev_samp(e.fin_roe) AS roe_std,
                quantile_cont(e.fin_roe, 0.75) - quantile_cont(e.fin_roe, 0.25) AS roe_iqr,
                quantile_cont(e.fin_roe, 0.10) AS roe_p10,
                quantile_cont(e.fin_roe, 0.90) AS roe_p90,
                AVG(CASE WHEN e.fin_roe < 0 THEN 1.0 ELSE 0 END) AS neg_roe_ratio,
                AVG(CASE WHEN e.fin_is_negative_equity THEN 1.0 ELSE 0 END) AS neg_equity_ratio,
                MIN(e.trade_month) AS first_roe_month,
                MAX(e.trade_month) AS last_roe_month,
                SUM(CASE WHEN e.roe_value_changed THEN 1 ELSE 0 END) AS roe_value_changes
            FROM roe_me e
            GROUP BY 1, 2
        )
        SELECT
            p.*, l.name, l.status,
            l.fin_roe AS latest_roe,
            l.fin_roa AS latest_roa
        FROM prof p
        LEFT JOIN roe_latest l ON l.ticker = p.ticker AND l.market = p.market
        ORDER BY p.market, p.ticker
        """
    ).df()


# --- §5.4 persistence --------------------------------------------------------


def rank_persistence(con: object) -> pd.DataFrame:
    """Rank correlation between roe_rank_market at month t and t-h (calendar lag)."""
    parts = []
    for h in PERSISTENCE_HORIZONS:
        parts.append(
            f"""
            SELECT {h} AS horizon_months,
                   corr(a.roe_rank_market, b.roe_rank_market) AS rank_corr,
                   COUNT(*) AS pairs
            FROM roe_me a
            JOIN roe_me b
              ON a.ticker = b.ticker AND a.market = b.market
             AND a.trade_month = b.trade_month + INTERVAL '{h}' MONTH
            """
        )
    df = con.execute(" UNION ALL ".join(parts) + " ORDER BY horizon_months").df()
    # Consecutive-month identical-ROE ratio (stale-repeat artifact magnitude).
    same = con.execute(
        """
        SELECT AVG(CASE WHEN a.fin_roe = b.fin_roe THEN 1.0 ELSE 0 END) AS same_ratio
        FROM roe_me a
        JOIN roe_me b
          ON a.ticker = b.ticker AND a.market = b.market
         AND a.trade_month = b.trade_month + INTERVAL '1' MONTH
        """
    ).fetchone()[0]
    df["consecutive_month_same_roe_ratio"] = same
    return df


def persistence_12m_by_group(con: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_market = con.execute(
        """
        SELECT a.market,
               corr(a.roe_rank_market, b.roe_rank_market) AS rank_corr_12m,
               COUNT(*) AS pairs
        FROM roe_me a
        JOIN roe_me b
          ON a.ticker = b.ticker AND a.market = b.market
         AND a.trade_month = b.trade_month + INTERVAL '12' MONTH
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_year = con.execute(
        """
        SELECT CAST(EXTRACT(year FROM a.trade_month) AS INTEGER) AS year,
               corr(a.roe_rank_market, b.roe_rank_market) AS rank_corr_12m,
               COUNT(*) AS pairs
        FROM roe_me a
        JOIN roe_me b
          ON a.ticker = b.ticker AND a.market = b.market
         AND a.trade_month = b.trade_month + INTERVAL '12' MONTH
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    return by_market, by_year


def value_change_persistence(con: object) -> pd.DataFrame:
    """Filing-to-filing rank persistence: only rows where ROE actually changed.

    Ranks each value-change event within (event_month, market), then correlates
    consecutive events per company — the real-update persistence with the
    stale-repeat artifact removed.
    """
    return con.execute(
        """
        WITH events AS (
            SELECT ticker, market, trade_month, fin_roe,
                   PERCENT_RANK() OVER (
                       PARTITION BY trade_month, market ORDER BY fin_roe
                   ) AS event_rank
            FROM roe_me
            WHERE roe_value_changed OR roe_value_changed IS NULL  -- first obs counts
        ),
        seq AS (
            SELECT *, ROW_NUMBER() OVER (
                       PARTITION BY ticker, market ORDER BY trade_month
                   ) AS event_seq
            FROM events
        )
        SELECT
            corr(a.event_rank, b.event_rank) AS event_rank_corr,
            COUNT(*) AS event_pairs,
            COUNT(DISTINCT a.ticker || '|' || a.market) AS tickers
        FROM seq a
        JOIN seq b
          ON a.ticker = b.ticker AND a.market = b.market
         AND a.event_seq = b.event_seq + 1
        """
    ).df()


# --- §5.5 market distribution ------------------------------------------------


def market_distribution(con: object) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            market,
            COUNT(*) AS rows,
            COUNT(DISTINCT ticker || '|' || market) AS tickers,
            median(fin_roe) AS median,
            quantile_cont(fin_roe, 0.25) AS p25,
            quantile_cont(fin_roe, 0.75) AS p75,
            quantile_cont(fin_roe, 0.75) - quantile_cont(fin_roe, 0.25) AS iqr,
            AVG(CASE WHEN fin_roe < 0 THEN 1.0 ELSE 0 END) AS neg_ratio
        FROM roe_me
        GROUP BY 1 ORDER BY 1
        """
    ).df()


# --- §5.7 outliers / data quality -------------------------------------------


def outlier_tables(con: object) -> dict[str, pd.DataFrame]:
    cols = (
        "ticker, market, name, status, trade_date, "
        "fin_roe, fin_roa, fin_debt_to_equity, fin_equity_ratio, fin_is_negative_equity"
    )
    high = con.execute(
        f"SELECT {cols} FROM roe_latest ORDER BY fin_roe DESC LIMIT 100"
    ).df()
    low = con.execute(
        f"SELECT {cols} FROM roe_latest ORDER BY fin_roe ASC LIMIT 100"
    ).df()
    high_lev = con.execute(
        f"""
        SELECT {cols} FROM roe_latest
        WHERE fin_roe > 0.3 AND fin_debt_to_equity > 2.0
        ORDER BY fin_debt_to_equity DESC LIMIT 100
        """
    ).df()
    high_roe_low_roa = con.execute(
        f"""
        SELECT {cols}, (fin_roe - fin_roa) AS roe_minus_roa FROM roe_latest
        WHERE fin_roe > 0.3 AND fin_roa IS NOT NULL AND fin_roa < 0.05
        ORDER BY roe_minus_roa DESC LIMIT 100
        """
    ).df()
    # Negative-equity names are ROE-null by construction; record as missingness.
    neg_equity = con.execute(
        """
        SELECT f.ticker, f.market, m.name, m.status,
               COUNT(*) AS panel_rows,
               COUNT(f.fin_roe) AS roe_non_null_rows,
               BOOL_OR(f.fin_is_negative_equity) AS ever_negative_equity
        FROM feat_fin_pit f
        LEFT JOIN stock_master m ON m.ticker = f.ticker AND m.market = f.market
        WHERE f.fin_is_negative_equity
        GROUP BY 1, 2, 3, 4
        ORDER BY panel_rows DESC
        """
    ).df()
    return {
        "roe_outliers_high.csv": high,
        "roe_outliers_low.csv": low,
        "roe_high_leverage_cases.csv": high_lev,
        "roe_high_roe_low_roa_cases.csv": high_roe_low_roa,
        "roe_negative_equity_missing.csv": neg_equity,
    }


# --- plots -------------------------------------------------------------------


def plot_histogram(con: object, report_dir: Path) -> None:
    df = con.execute("SELECT fin_roe AS raw, roe_winsor_1p AS winsor FROM roe_me").df()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    lo, hi = df["winsor"].min(), df["winsor"].max()
    axes[0].hist(df["raw"].clip(lo, hi), bins=80, color="#4c72b0")
    axes[0].set_title("Raw ROE (display-clipped to winsor range)")
    axes[0].set_xlabel("fin_roe")
    axes[1].hist(df["winsor"], bins=80, color="#55a868")
    axes[1].set_title("Winsorized ROE (global p1-p99)")
    axes[1].set_xlabel("roe_winsor_1p")
    fig.suptitle("Month-end ROE distribution")
    fig.savefig(report_dir / "roe_histogram.png", dpi=160)
    plt.close(fig)


def plot_monthly_bands(monthly: pd.DataFrame, report_dir: Path) -> None:
    d = monthly.copy()
    d["trade_month"] = pd.to_datetime(d["trade_month"])
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9), constrained_layout=True, sharex=True
    )
    ax1.fill_between(d["trade_month"], d["p10"], d["p90"], alpha=0.2,
                     color="#4c72b0", label="p10-p90")
    ax1.fill_between(d["trade_month"], d["p25"], d["p75"], alpha=0.35,
                     color="#4c72b0", label="p25-p75")
    ax1.plot(d["trade_month"], d["median"], color="#c44e52", lw=1.5, label="median")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("ROE")
    ax1.set_title("Monthly cross-sectional ROE distribution")
    ax1.legend(loc="upper right")
    ax2.plot(d["trade_month"], d["missing_rate"], color="#8172b3", label="missing rate")
    ax2.plot(d["trade_month"], d["neg_ratio"], color="#c44e52", label="negative ROE ratio")
    ax2.plot(d["trade_month"], d["neg_equity_ratio"], color="#937860",
             label="negative-equity ratio")
    ax2.set_ylabel("ratio")
    ax2.set_xlabel("trade_month")
    ax2.legend(loc="upper right")
    fig.savefig(report_dir / "roe_monthly_bands.png", dpi=160)
    plt.close(fig)


def plot_company_volatility(profile: pd.DataFrame, report_dir: Path) -> None:
    d = profile.dropna(subset=["roe_median", "roe_std"]).copy()
    mx, mn = d["roe_median"].quantile(0.99), d["roe_median"].quantile(0.01)
    sx = d["roe_std"].quantile(0.99)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    axes[0].hist(d["roe_median"].clip(mn, mx), bins=60, color="#4c72b0")
    axes[0].set_title("Company median ROE")
    axes[0].set_xlabel("roe_median")
    axes[1].hist(d["roe_std"].clip(0, sx), bins=60, color="#dd8452")
    axes[1].set_title("Company ROE volatility (std)")
    axes[1].set_xlabel("roe_std")
    axes[2].scatter(d["roe_median"].clip(mn, mx), d["roe_std"].clip(0, sx),
                    s=6, alpha=0.3, color="#55a868")
    axes[2].set_title("Median ROE vs volatility")
    axes[2].set_xlabel("roe_median")
    axes[2].set_ylabel("roe_std")
    fig.suptitle("Company-level ROE level / volatility (month-end sample)")
    fig.savefig(report_dir / "roe_company_volatility.png", dpi=160)
    plt.close(fig)


# --- summary -----------------------------------------------------------------


def num(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_summary(
    report_dir: Path,
    *,
    snapshot_date: str,
    source: str,
    overall: pd.DataFrame,
    persistence: pd.DataFrame,
    vc_persistence: pd.DataFrame,
    market: pd.DataFrame,
    profile: pd.DataFrame,
    annual_latest: pd.DataFrame,
) -> Path:
    me = overall[overall["sample"] == "month_end"].iloc[0]
    cl = overall[overall["sample"] == "company_latest"].iloc[0]
    an = annual_latest.iloc[0]
    p12 = persistence[persistence["horizon_months"] == 12]
    p12_corr = float(p12["rank_corr"].iloc[0]) if not p12.empty else float("nan")
    p1_row = persistence[persistence["horizon_months"] == 1]
    p1_corr = float(p1_row["rank_corr"].iloc[0]) if not p1_row.empty else float("nan")
    same_ratio = float(persistence["consecutive_month_same_roe_ratio"].iloc[0])
    vc_corr = float(vc_persistence["event_rank_corr"].iloc[0])

    if p12_corr >= 0.5:
        persist_verdict = (
            f"12-month rank persistence is moderate-to-strong (corr={num(p12_corr)}); "
            "ROE rank is stable enough to consider as a model feature."
        )
    elif p12_corr >= 0.3:
        persist_verdict = (
            f"12-month rank persistence is weak-to-moderate (corr={num(p12_corr)}); "
            "usable but noisy — prefer sector/size controls before relying on it."
        )
    else:
        persist_verdict = (
            f"12-month rank persistence is low (corr={num(p12_corr)}); ROE rank is "
            "not stable enough to use directly as a standalone feature."
        )

    rec = ("winsorized ROE" if abs(float(me["max"])) > 1.0 or abs(float(me["min"])) > 1.0
           else "raw ROE")

    market_lines = [
        "| " + " | ".join([
            str(r.market), f"{int(r.tickers):,}", num(r.median), num(r.iqr), num(r.neg_ratio),
        ]) + " |"
        for r in market.itertuples(index=False)
    ]

    body = [
        "# ROE Distribution Analysis",
        "",
        f"- snapshot_date: `{snapshot_date}`",
        f"- source: `{source}`",
        "- input: `feat_fin_pit` recomputed from raw (current CIS-aware rules)",
        "- generated_at_kst: "
        f"`{datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}`",
        "- plan: `research/dev/20260627_correlation_fs_raw/01_00_roe_distribution_plan.md`",
        "",
        "## 5.1 Overall distribution",
        "",
        "| sample | tickers | mean | median | std | p10 | p90 | neg ratio | >0.3 | >1.0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| month-end | "
        f"{int(me['tickers']):,} | {num(me['mean'])} | {num(me['median'])} | "
        f"{num(me['std'])} | {num(me['p10'])} | {num(me['p90'])} | {num(me['neg_ratio'])} | "
        f"{num(me['gt_30_ratio'])} | {num(me['gt_100_ratio'])} |",
        "| company-latest | "
        f"{int(cl['tickers']):,} | {num(cl['mean'])} | {num(cl['median'])} | "
        f"{num(cl['std'])} | {num(cl['p10'])} | {num(cl['p90'])} | {num(cl['neg_ratio'])} | "
        f"{num(cl['gt_30_ratio'])} | {num(cl['gt_100_ratio'])} |",
        "",
        f"Recommended base representation for downstream return analysis: **{rec}** "
        "(rank-based for cross-sectional IC/quintile work). Extreme tails "
        f"(month-end min={num(me['min'])}, max={num(me['max'])}) confirm winsorize/rank is "
        "needed before any level-based use.",
        "",
        "## 5.7 Numerator-period caveat (annual vs mixed)",
        "",
        "`feat_fin_pit.fin_roe` divides the latest *available* net_income by equity. "
        "For ~3/4 of the calendar the latest filing is a partial-year YTD figure "
        "(Q1/half/Q3), so the numerator is un-annualized and the ROE *level* is "
        "systematically understated. An annual-only (reprt_code 11011) recomputation "
        "shifts the company-latest median up:",
        "",
        "| basis | tickers | median | mean | p25 | p75 | neg ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| mixed-period (feat_fin_pit) | "
        f"{int(cl['tickers']):,} | {num(cl['median'])} | {num(cl['mean'])} | "
        f"{num(cl['p25'])} | {num(cl['p75'])} | {num(cl['neg_ratio'])} |",
        "| annual-only (11011) | "
        f"{int(an['tickers']):,} | {num(an['median'])} | {num(an['mean'])} | "
        f"{num(an['p25'])} | {num(an['p75'])} | {num(an['neg_ratio'])} |",
        "",
        "Use **annual-only ROE** (or a TTM/annualized numerator) for level-based "
        "analysis; the mixed-period mart ROE is acceptable only for cross-sectional "
        "*rank* work within a single trade date. See `roe_annual_by_year.csv`.",
        "",
        "## 5.4 Persistence",
        "",
        f"- 1-month rank persistence (artifact reference): corr={num(p1_corr)}, with "
        f"{same_ratio:.1%} of consecutive month-end pairs holding an *identical* ROE "
        "(stale annual-filing repeat).",
        f"- 12-month rank persistence (primary): corr={num(p12_corr)}.",
        f"- Filing-to-filing (value-change) rank persistence: corr={num(vc_corr)} over "
        f"{int(vc_persistence['event_pairs'].iloc[0]):,} events.",
        "",
        persist_verdict,
        "",
        "## 5.5 Market distribution",
        "",
        "| market | tickers | median | IQR | neg ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
        *market_lines,
        "",
        "## Notes",
        "",
        "- §5.6 sector analysis remains out of scope: no sector mapping in this snapshot.",
        f"- Company profiles cover {len(profile):,} ticker-market pairs "
        "(`roe_company_profile.csv`).",
        "- `fin_is_negative_equity = 1` names are ROE-null by construction (clipped); "
        "recorded as missingness in `roe_negative_equity_missing.csv`, not as extremes.",
    ]
    out = report_dir / "analysis_summary.md"
    out.write_text("\n".join(body) + "\n")
    return out


def main() -> None:
    args = parse_args()
    root = DataRoot.resolve(market="kr")
    config = LakeConfig(root=root, snapshot_date=args.snapshot_date, source=args.source)
    report_root = root.output / "reports" / "analysis" / "fin_vs_price_corr"
    report_dir = report_root / args.report_date / "roe_distribution"
    report_dir.mkdir(parents=True, exist_ok=True)

    con = connect(config)
    setup_views(con, config)
    build_samples(con)

    overall = overall_distribution(con)
    winsor = winsor_thresholds(con)
    monthly = monthly_distribution(con)
    monthly_market = monthly_distribution_by_market(con)
    profile = company_profile(con)
    persistence = rank_persistence(con)
    persist_market, persist_year = persistence_12m_by_group(con)
    vc_persistence = value_change_persistence(con)
    market = market_distribution(con)
    annual_latest, annual_by_year = annual_roe_baseline(con)
    outliers = outlier_tables(con)

    write_csv(overall, report_dir, "roe_overall_distribution.csv")
    write_csv(winsor, report_dir, "roe_winsor_thresholds.csv")
    write_csv(monthly, report_dir, "roe_monthly_distribution.csv")
    write_csv(monthly_market, report_dir, "roe_monthly_distribution_by_market.csv")
    write_csv(profile, report_dir, "roe_company_profile.csv")
    write_csv(persistence, report_dir, "roe_persistence.csv")
    write_csv(persist_market, report_dir, "roe_persistence_12m_by_market.csv")
    write_csv(persist_year, report_dir, "roe_persistence_12m_by_year.csv")
    write_csv(vc_persistence, report_dir, "roe_value_change_persistence.csv")
    write_csv(market, report_dir, "roe_market_distribution.csv")
    write_csv(annual_latest, report_dir, "roe_annual_company_latest.csv")
    write_csv(annual_by_year, report_dir, "roe_annual_by_year.csv")
    # company-latest distribution is the company-latest row of overall; also dump rows
    write_csv(
        con.execute("SELECT * FROM roe_latest ORDER BY market, ticker").df(),
        report_dir,
        "roe_latest_company_distribution.csv",
    )
    for name, df in outliers.items():
        write_csv(df, report_dir, name)

    plot_histogram(con, report_dir)
    plot_monthly_bands(monthly, report_dir)
    plot_company_volatility(profile, report_dir)

    write_summary(
        report_dir,
        snapshot_date=args.snapshot_date,
        source=args.source,
        overall=overall,
        persistence=persistence,
        vc_persistence=vc_persistence,
        market=market,
        profile=profile,
        annual_latest=annual_latest,
    )
    print(f"Wrote ROE distribution analysis to {report_dir}")


if __name__ == "__main__":
    main()
