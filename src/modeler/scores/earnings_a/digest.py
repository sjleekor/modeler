"""F4: the hash-only run. Builds label tables under a hash grant and keeps digests only.

``seal.digest_only`` is the only way labels are touched here; the frames never leave
the builders. Every exact column is turned into a canonical string inside the builder
so the digest does not depend on dtype, row order or pandas version. Float columns
(``y``, ``y_clip``) get their own digest (``%.17g``). Output holds hex digests, counts
and column names only (JSON scalars and dicts of ints).

Builders run under ``warnings.simplefilter("ignore")``: pandas warnings quote values,
and nothing here may print one.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from modeler.scores.earnings_a import features as feat
from modeler.scores.earnings_a import labels, reasons, seal, universe
from modeler.scores.earnings_a.paths import DataPaths
from modeler.scores.earnings_a.spec import RAW_MAX_DATE, SNAPSHOT

WINDOWS: dict[str, tuple[date, ...]] = {
    "DEV": seal.DEV_DATES,
    "GAP": seal.GAP_DATES,
    "SEALED": seal.SEALED_DATES,
}
ALL_DATES: tuple[date, ...] = (
    seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES + seal.CONFIRM_DATES
)
# Two-year labels must be mature (prereg 7.3): DEV labels end <= 2023-06-30, SEALED <= 2024-06-30.
Y2_DATE_MAX: dict[str, date | None] = {
    "DEV": date(2021, 6, 30),
    "GAP": None,
    "SEALED": date(2024, 6, 30),
}
TARGET_ORDER = ("oi_y1", "ocf_y1", "oi_y2")

EXACT_COLS = [
    "decision_date",
    "ticker",
    "basis",
    "a_t_dec",
    "oi_sum_dec",
    "label_pub_date",
    "label_last_quarter_end",
    "rcept_no_1",
    "rcept_no_2",
    "rcept_no_3",
    "rcept_no_4",
    "label_ok",
    "missing_kind",
]
FLOAT_COLS = ["y", "y_clip"]
ANCHOR_COLS = ["decision_date", "ticker", "corp_code", "basis", "a_t_dec", "input_eligible"]
# Label tables and reasons cover input-eligible rows plus the record-only holding-company
# variant (§16 E-3). The clip pool stays `input_eligible`: labels._clip_y uses the anchor's
# input_eligible column.
ANCHOR_ROW_FILTER = "input_eligible_incl_holding"


def prepare_inputs(paths: DataPaths, con) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label-free inputs for all 42 decision dates: (universe, rows, clipped features)."""
    del paths  # the connection already holds the views; kept for a stable signature
    uni = universe.build_universe(con, list(ALL_DATES))
    rows = feat.load_input_rows(con)
    anchor = feat.build_anchor(rows, uni)
    feats = feat.build_features(rows, anchor)
    return uni, rows, feat.clip_features(feats)


def load_receipts(con) -> pd.DataFrame:
    return con.execute(
        "SELECT CAST(rcept_no AS VARCHAR) AS rcept_no, CAST(corp_code AS VARCHAR) AS corp_code, "
        "CAST(report_nm AS VARCHAR) AS report_nm FROM dart_filing_receipt_raw"
    ).fetchdf()


def load_stock_master(con) -> pd.DataFrame:
    return con.execute(
        "SELECT CAST(ticker AS VARCHAR) AS ticker, CAST(status AS VARCHAR) AS status, "
        "last_seen_date FROM stock_master"
    ).fetchdf()


def load_last_trade(con) -> pd.DataFrame:
    return con.execute(
        "SELECT CAST(ticker AS VARCHAR) AS ticker, max(trade_date) AS last_trade_date "
        "FROM daily_ohlcv GROUP BY ticker"
    ).fetchdf()


def _canon(v: Any) -> str:
    if v is None or v is pd.NA or v is pd.NaT:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    if isinstance(v, bool | np.bool_):
        return "True" if bool(v) else "False"
    if isinstance(v, pd.Timestamp | datetime | date | np.datetime64):
        return seal._to_date(v).isoformat()
    return str(v)


def _canonicalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = [_canon(v) for v in out[c]]
    return out


def _in_dates(df: pd.DataFrame, dates: tuple[date, ...]) -> pd.DataFrame:
    keep = set(dates)
    return df.loc[[seal._to_date(d) in keep for d in df["decision_date"]]]


def _anchor_rows(df: pd.DataFrame, dates: tuple[date, ...]) -> pd.DataFrame:
    sub = _in_dates(df, dates)
    keep = sub[ANCHOR_ROW_FILTER].fillna(False).astype(bool)
    return sub.loc[keep.to_numpy()]


def _label_digest(
    paths: DataPaths,
    feats: pd.DataFrame,
    receipts: pd.DataFrame,
    dates: tuple[date, ...],
    target: str,
) -> dict[str, Any]:
    metric, horizon = labels.TARGETS[target]
    anchor = _anchor_rows(feats, dates)[ANCHOR_COLS].reset_index(drop=True)
    exact = [*EXACT_COLS, "input_eligible"]

    def builder(g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = labels.load_label_rows(
                paths, decision_dates=dates, grant=g, horizon=horizon, metrics=(metric,)
            )
            df = labels.build_labels(
                r2, anchor, receipts, decision_dates=dates, grant=g, target=target
            )
            # build_labels does not carry input_eligible; take it from the anchor row.
            flag = anchor[["decision_date", "ticker", "basis", "input_eligible"]]
            df = df.merge(
                flag, on=["decision_date", "ticker", "basis"], how="left", validate="many_to_one"
            )
            return _canonicalize(df, exact)

    return seal.digest_only(
        builder,
        exact_cols=exact,
        float_cols=FLOAT_COLS,
        count_by=("decision_date", "label_ok", "input_eligible"),
    )


def _reason_digest(
    paths: DataPaths,
    feats: pd.DataFrame,
    receipts: pd.DataFrame,
    stock_master: pd.DataFrame,
    last_trade: pd.DataFrame,
    dates: tuple[date, ...],
    run_at: datetime,
) -> dict[str, Any]:
    metric, horizon = labels.TARGETS["oi_y1"]
    sub = _anchor_rows(feats, dates)
    anchor = sub[ANCHOR_COLS].reset_index(drop=True)
    fye = sub[["decision_date", "ticker", "fye_month"]].reset_index(drop=True)
    cols = [c for c in reasons.OUTPUT_COLUMNS if c != "confirmed_at"]

    def builder(g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = labels.load_label_rows(
                paths, decision_dates=dates, grant=g, horizon=horizon, metrics=(metric,)
            )
            lq = labels.build_label_quarters(
                r2, anchor, receipts, decision_dates=dates, grant=g, metric=metric, horizon=horizon
            )
            presence = labels.presence_view(lq).merge(
                fye, on=["decision_date", "ticker"], how="left"
            )
            df = reasons.build_reason_dictionary(
                presence,
                receipts,
                stock_master,
                last_trade,
                run_at=run_at,
                source_snapshot=SNAPSHOT,
                raw_max_date=date.fromisoformat(RAW_MAX_DATE),
                metric=metric,
            )
            return _canonicalize(df, cols)

    return seal.digest_only(
        builder,
        exact_cols=cols,
        float_cols=(),
        count_by=("decision_date", "reason_code", "reason_detail"),
    )


def run_f4(paths: DataPaths, con, *, run_at: datetime) -> dict[str, Any]:
    """Digest every (window, target) label table and the reason table per window."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, feats = prepare_inputs(paths, con)
        receipts = load_receipts(con)
        stock_master = load_stock_master(con)
        last_trade = load_last_trade(con)
        raw_max = last_trade["last_trade_date"].max()
    digests: dict[str, Any] = {}
    reason_digests: dict[str, Any] = {}
    for window, wdates in WINDOWS.items():
        for target in TARGET_ORDER:
            dates = wdates
            if target == "oi_y2" and Y2_DATE_MAX[window] is not None:
                dates = tuple(d for d in wdates if d <= Y2_DATE_MAX[window])
            if not dates:
                continue
            digests[f"{window}.{target}"] = _label_digest(paths, feats, receipts, dates, target)
        reason_digests[window] = _reason_digest(
            paths, feats, receipts, stock_master, last_trade, wdates, run_at
        )
    return {
        "digests": digests,
        "reason_digests": reason_digests,
        "raw_max_trade_date": seal._to_date(raw_max).isoformat(),
    }
