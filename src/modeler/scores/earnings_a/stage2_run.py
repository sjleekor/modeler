"""Research A stage 2: database glue and the four runners behind the CLI.

* :func:`run_check`    F3 technical check: calendar and universe counts, no returns, no predictions.
* :func:`run_past`     one-shot past run (opens ``STAGE2_PAST``; refuses a dirty tree).
* :func:`run_forward`  one forward basket file (opens ``FORWARD`` for one decision date).
* :func:`run_score`    scores matured forward tranches from committed basket files.
* :func:`reproduce_dev` reproduction check of the stage-1 DEV per-date MAE (DEV grant only).

Nothing here writes inside the repository: every output goes under the output root of
``DataPaths`` (``$STOCK_DATA_ROOT/kr/output/earnings_forecast_a``). Pure logic lives in
``stage2.py``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from modeler.etl.universe import tradable_universe_filter
from modeler.etl.universe_raw_price import build_universe_raw_price_sql
from modeler.models._03_kr_next.liquidation_notice import build_liquidation_notice_sql
from modeler.scores.earnings_a import dev_select, digest, manifest, seal, stage2, train
from modeler.scores.earnings_a import features as feat
from modeler.scores.earnings_a import paths as P
from modeler.scores.earnings_a.paths import DataPaths, _quote
from modeler.scores.earnings_a.sealed_run import (
    CHOSEN_REL,
    CHOSEN_SHA256,
    F4_REL,
    SealedAbort,
    _f4_check,
    _load_chosen,
    _load_labels,
    ksic_section,
    make_mode,
)
from modeler.scores.earnings_a.sealed_run import build_dataset as build_train_dataset
from modeler.scores.earnings_a.spec import INNER_VAL_DATES
from modeler.scores.earnings_a.train import ModelSpec

PAST_DATES = seal.STAGE2_PAST_DATES
TRAIN_DATES = seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES
CHECK_DATES = PAST_DATES + (seal.FORWARD_START,)
_LABEL_COLS = ["label_ok", "y", "y_clip", "label_pub_date", "missing_kind"]


# ---------------------------------------------------------------------------
# Connection and small readers
# ---------------------------------------------------------------------------


def connect_stage2(paths: DataPaths) -> duckdb.DuckDBPyConnection:
    """Stage-1 input views plus ``krx_index_daily``, ``etf_daily`` and ``etf_distribution``."""
    con = P.connect_inputs(paths)
    extra = {
        "krx_index_daily": paths.raw_root / "krx_index_daily",
        "etf_daily": paths.stock_data_root / "kr/raw/krx_baseline/etf_daily",
        "etf_distribution": paths.stock_data_root / "kr/raw/krx_baseline/etf_distribution",
    }
    for name, d in extra.items():
        if not d.is_dir() or not any(d.rglob("*.parquet")):
            raise FileNotFoundError(f"input has no parquet files: {name}")
        glob = str(d / "**" / "*.parquet")
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet("
            f"{_quote(glob)}, hive_partitioning=false, union_by_name=true)"
        )
    return con


def load_sessions(con) -> list[date]:
    return sorted(
        r[0] for r in con.execute("SELECT DISTINCT trade_date FROM daily_ohlcv").fetchall()
    )


def signal_day_info(con, u_dates: list[date]) -> pd.DataFrame:
    """Per (signal_day, ticker): halted, P6 tradable, liquidation notice, raw close.

    Evaluated on the signal day ``u`` itself (stage 1 evaluates at ``s_t``). One row per line
    that has a ``daily_ohlcv`` row on ``u``. ``raw_close_u`` is the KRX close of that day.
    """
    con.execute("CREATE OR REPLACE TEMP TABLE _s2_days(trade_date DATE)")
    con.executemany("INSERT INTO _s2_days VALUES (?)", [(d,) for d in sorted(set(u_dates))])
    flt = tradable_universe_filter()
    df = con.execute(
        f"""
        WITH o AS (
            SELECT trade_date, ticker, bool_or(open = 0 AND high = 0 AND low = 0) AS halted_u,
                   max(close) AS adj_close
            FROM daily_ohlcv
            WHERE trade_date IN (SELECT trade_date FROM _s2_days)
            GROUP BY trade_date, ticker
        ),
        liq AS (
            SELECT trade_date, ticker, bool_or(liq_notice_flag) AS liq_notice_u
            FROM ({build_liquidation_notice_sql()})
            WHERE trade_date IN (SELECT trade_date FROM _s2_days)
            GROUP BY trade_date, ticker
        ),
        p6 AS (
            SELECT trade_date, ticker, bool_or(tradable_ok) AS p6_tradable_u
            FROM ({build_universe_raw_price_sql(flt)})
            WHERE trade_date IN (SELECT trade_date FROM _s2_days)
            GROUP BY trade_date, ticker
        ),
        m AS (
            SELECT trade_date, ticker, max(source_close) AS source_close
            FROM daily_market_cap
            WHERE trade_date IN (SELECT trade_date FROM _s2_days)
            GROUP BY trade_date, ticker
        )
        SELECT o.trade_date AS signal_day, o.ticker, o.halted_u,
               COALESCE(liq.liq_notice_u, FALSE) AS liq_notice_u,
               COALESCE(p6.p6_tradable_u, FALSE) AS p6_tradable_u,
               CASE WHEN m.source_close > 0 THEN m.source_close ELSE o.adj_close END AS raw_close_u
        FROM o
        LEFT JOIN liq ON liq.trade_date = o.trade_date AND liq.ticker = o.ticker
        LEFT JOIN p6 ON p6.trade_date = o.trade_date AND p6.ticker = o.ticker
        LEFT JOIN m ON m.trade_date = o.trade_date AND m.ticker = o.ticker
        """  # noqa: S608
    ).df()
    con.execute("DROP TABLE IF EXISTS _s2_days")
    df["signal_day"] = pd.to_datetime(df["signal_day"]).dt.date
    return df


def mcap_lines(con, u_dates: list[date]) -> pd.DataFrame:
    """All ``daily_market_cap`` lines (every listed ticker) of the signal days."""
    con.execute("CREATE OR REPLACE TEMP TABLE _s2_days(trade_date DATE)")
    con.executemany("INSERT INTO _s2_days VALUES (?)", [(d,) for d in sorted(set(u_dates))])
    df = con.execute(
        """
        SELECT trade_date AS signal_day, ticker, max(market_cap) AS market_cap
        FROM daily_market_cap
        WHERE trade_date IN (SELECT trade_date FROM _s2_days)
        GROUP BY trade_date, ticker
        """
    ).df()
    con.execute("DROP TABLE IF EXISTS _s2_days")
    df["signal_day"] = pd.to_datetime(df["signal_day"]).dt.date
    return df


def entry_status(con, entry_days: list[date]) -> pd.DataFrame:
    """Per (entry day, ticker): halted flag on the entry session (no returns)."""
    con.execute("CREATE OR REPLACE TEMP TABLE _s2_days(trade_date DATE)")
    con.executemany("INSERT INTO _s2_days VALUES (?)", [(d,) for d in sorted(set(entry_days))])
    df = con.execute(
        """
        SELECT trade_date AS entry_day, ticker, bool_or(open = 0 AND high = 0 AND low = 0) AS halted_entry
        FROM daily_ohlcv WHERE trade_date IN (SELECT trade_date FROM _s2_days)
        GROUP BY trade_date, ticker
        """
    ).df()
    con.execute("DROP TABLE IF EXISTS _s2_days")
    df["entry_day"] = pd.to_datetime(df["entry_day"]).dt.date
    return df


def load_closes(con, tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Adjusted ``daily_ohlcv`` closes and halt flags of ``tickers`` for ``start..end``."""
    if not tickers:
        return pd.DataFrame(columns=["trade_date", "ticker", "close", "is_halted"])
    con.execute("CREATE OR REPLACE TEMP TABLE _s2_tk(ticker VARCHAR)")
    con.executemany("INSERT INTO _s2_tk VALUES (?)", [(t,) for t in sorted(set(tickers))])
    df = con.execute(
        """
        SELECT trade_date, ticker, max(close) AS close,
               bool_or(open = 0 AND high = 0 AND low = 0) AS is_halted
        FROM daily_ohlcv
        WHERE ticker IN (SELECT ticker FROM _s2_tk) AND trade_date BETWEEN ? AND ?
        GROUP BY trade_date, ticker
        """,
        [start, end],
    ).df()
    con.execute("DROP TABLE IF EXISTS _s2_tk")
    return df


def index_series(con, name: str = stage2.INDEX_NAME) -> pd.Series:
    """Close of one KRX index by date (the latest fetch wins on duplicates)."""
    df = con.execute(
        """
        SELECT bas_dd, arg_max(CAST(close_idx AS DOUBLE), fetched_at) AS c
        FROM krx_index_daily WHERE idx_nm = ? GROUP BY bas_dd ORDER BY bas_dd
        """,
        [name],
    ).df()
    return pd.Series(df["c"].to_numpy(dtype=float), index=pd.to_datetime(df["bas_dd"]))


def etf_series(con, code: str = stage2.KODEX200) -> tuple[pd.Series, pd.DataFrame]:
    """Close series and distribution table of one ETF (KODEX 200 by default)."""
    px = con.execute(
        """
        SELECT bas_date, arg_max(CAST(tdd_clsprc_num AS DOUBLE), obs_seq) AS c
        FROM etf_daily WHERE ISU_CD = ? GROUP BY bas_date ORDER BY bas_date
        """,
        [code],
    ).df()
    dist = con.execute(
        "SELECT rgt_std_date, per_share_amount FROM etf_distribution WHERE stock_code = ?",
        [code],
    ).df()
    return pd.Series(px["c"].to_numpy(dtype=float), index=pd.to_datetime(px["bas_date"])), dist


# ---------------------------------------------------------------------------
# Inputs and predictions
# ---------------------------------------------------------------------------


def ev_amounts(rows: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Borrowings and cash amounts (won) at the anchor period, ``available_from <= t``.

    Same fetch as the stage-1 features (``features._fetch``); returns ``debt_st_amt,
    debt_lt_amt, cash_amt`` aligned to ``feats``.
    """
    src = feat._metric_sources(feat._prep_rows(rows) if "pe_month" not in rows.columns else rows)
    df = feats.reset_index(drop=True)
    key = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "basis": df["basis"],
            "pm0": feat._mi(pd.to_datetime(df["pe0"])).astype("Int64"),
            "t": pd.to_datetime(df["decision_date"]),
        }
    )
    out = pd.DataFrame(index=df.index)
    for short, col in (
        ("debt_st", "debt_st_amt"),
        ("debt_lt", "debt_lt_amt"),
        ("cash", "cash_amt"),
    ):
        out[col] = feat._fetch(df, key, src[short], 0)["st"].astype(float)
    return out


def prepare(con, dates: list[date]) -> dict[str, pd.DataFrame]:
    """Label-free inputs for ``dates``: clipped features with EV amounts, rows, receipts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from modeler.scores.earnings_a import universe

        uni = universe.build_universe(con, list(dates))
        rows = feat.load_input_rows(con)
        anchor = feat.build_anchor(rows, uni)
        feats = feat.clip_features(feat.build_features(rows, anchor))
        feats = feats.reset_index(drop=True)
        feats = pd.concat([feats, ev_amounts(rows, feats)], axis=1)
        receipts = digest.load_receipts(con)
    return {"feats": feats, "rows": rows, "receipts": receipts}


def eval_frame(feats: pd.DataFrame, t: date) -> pd.DataFrame:
    """Anchor rows of ``t`` with empty label columns: feature rows with no label."""
    sub = digest._anchor_rows(feats, (t,)).copy()
    sub["label_ok"] = False
    sub["y"] = np.nan
    sub["y_clip"] = np.nan
    sub["label_pub_date"] = pd.NaT
    sub["missing_kind"] = None
    return sub


def predict(
    ds: pd.DataFrame,
    eval_dates: list[date],
    decision_dates: list[date],
    grant: seal.SealGrant,
    hgb: ModelSpec,
) -> pd.DataFrame:
    """PERS and the chosen HGB for every input-eligible row of ``eval_dates`` (stage 1 code).

    Returns ``decision_date, ticker, corp_code, y_hat_pers, y_hat_hgb``. Training rows are
    ``label_pub_date <= t`` (rule A20, inside ``train``); labels of ``eval_dates`` are never read.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = train.walk_forward(
            ds,
            eval_dates=list(eval_dates),
            specs=[ModelSpec("PERS"), hgb],
            decision_dates=list(decision_dates),
            grant=grant,
        )
    pers = pred[pred["model"] == "PERS"].rename(columns={"y_hat": "y_hat_pers"})
    hgbp = pred[pred["model"] == "HGB"].rename(columns={"y_hat": "y_hat_hgb"})
    keys = ["decision_date", "ticker", "corp_code"]
    return pers[[*keys, "y_hat_pers"]].merge(hgbp[[*keys, "y_hat_hgb"]], on=keys, how="outer")


def reproduce_dev(paths: DataPaths, con, chosen: dict[str, Any], per_date_tsv: Path) -> dict:
    """Reproduction check: stage-2 prediction path under the DEV grant vs the dev per-date MAE.

    Returns the largest absolute MAE difference for HGB and PERS over the 13 inner validation
    dates (aggregated MAE only; no label or prediction value leaves this function).
    """
    grant = seal.grant_dev()
    inputs = prepare(con, list(seal.DEV_DATES))
    feats = inputs["feats"]
    lab = dev_select.load_dev_labels(paths, feats, inputs["receipts"], grant)
    ds = dev_select.build_dataset(feats, lab)
    pred = predict(ds, list(INNER_VAL_DATES), list(seal.DEV_DATES), grant, chosen["HGB"])
    j = pred.merge(
        ds[["decision_date", "ticker", "corp_code", "label_ok", "y_clip"]],
        on=["decision_date", "ticker", "corp_code"],
        how="left",
    )
    j = j[stage2.as_bool(j["label_ok"]) & j["y_clip"].notna()]
    ref = pd.read_csv(per_date_tsv, sep="\t")
    out: dict[str, Any] = {}
    for name, col, sid in (
        ("HGB", "y_hat_hgb", chosen["HGB"].spec_id),
        ("PERS", "y_hat_pers", ModelSpec("PERS").spec_id),
    ):
        got = j.groupby("decision_date").apply(
            lambda g, c=col: float((g["y_clip"] - g[c]).abs().mean()), include_groups=False
        )
        got.index = [seal._to_date(d).isoformat() for d in got.index]
        r = ref[ref["spec_id"] == sid].set_index("decision_date")["mae"]
        if set(got.index) != set(r.index):
            raise AssertionError(f"{name}: dates differ from the dev per-date table")
        diff = (got - r.reindex(got.index)).abs()
        out[name] = {"max_abs_diff": float(diff.max()), "n_dates": int(len(diff))}
    return out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _iso(d: Any) -> Any:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    if isinstance(d, date | datetime | pd.Timestamp):
        return seal._to_date(d).isoformat()
    return d


def _py(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_py(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float):
        return None if np.isnan(o) else o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, date | datetime | pd.Timestamp):
        return seal._to_date(o).isoformat()
    return o


def _write_tsv(path: Path, df: pd.DataFrame) -> Path:
    with open(path, "x", encoding="utf-8") as f:
        df.to_csv(f, sep="\t", index=False, float_format="%.12g", lineterminator="\n")
    return path


def _new_dir(paths: DataPaths, name: str, out: str | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=False)
        return d
    # Two runs in the same second must not collide (the later one would fail before its own
    # seal check); add a counter instead of reusing a directory.
    base = paths.out_root / f"{name}_{stamp}"
    d, k = base, 1
    while True:
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            k += 1
            d = base.with_name(f"{base.name}_{k}")


def _refuse_dirty() -> dict[str, Any]:
    git = manifest.git_state(manifest.repo_root())
    if git["dirty"]:
        raise SystemExit("refused: earnings_a code or tests are dirty")
    return git


# ---------------------------------------------------------------------------
# F3 technical check (no returns, no predictions)
# ---------------------------------------------------------------------------


def check_row(
    t: date,
    sch: stage2.Schedule,
    feats_t: pd.DataFrame,
    info_u: pd.DataFrame,
    mcap_u_lines: pd.DataFrame,
    entry_halt: pd.DataFrame | None,
) -> dict[str, Any]:
    """One row of the stage-2 check: calendar and universe counts for decision date ``t``."""
    co = stage2.company_mcap(mcap_u_lines)
    uni = stage2.assemble(feats_t, info_u, co, preds=None)
    ok = uni[uni["in_stage2"]]
    elig = stage2.as_bool(feats_t["input_eligible"])
    # all common tickers (no stage-1 eligibility) that pass the signal-day filter only
    c = info_u[info_u["ticker"].str[-1] == "0"]
    all_common = c[
        (~stage2.as_bool(c["halted_u"]))
        & stage2.as_bool(c["p6_tradable_u"])
        & ~stage2.as_bool(c["liq_notice_u"])
    ]
    row: dict[str, Any] = {
        "decision_date": t,
        "signal_day": sch.signal_day,
        "entry": sch.entry,
        "exit": sch.exit if sch.exit is not None else "beyond data",
        "n_eligible_t": int(elig.sum()),
        "n_eligible_listed_tradable_u": int(len(ok)),
        "n_all_common_tradable_u": int(len(all_common)),
        "n_raw_close_gt_416667": int((ok["raw_close_u"] > stage2.PRICE_PER_NAME_LIMIT).sum()),
        "n_entry_halted": None,
        "n_entry_missing": None,
        "n_pref_addon": int(stage2.as_bool(ok["has_pref"]).sum()),
        "n_pref_gt_10pct": int((ok["pref_ratio"] > 0.10).sum()),
    }
    for reason in stage2.DROP_ORDER[:5]:
        row[f"drop_{reason}"] = int((uni["drop_reason"] == reason).sum())
    if entry_halt is not None:
        m = ok[["ticker"]].merge(entry_halt[["ticker", "halted_entry"]], on="ticker", how="left")
        row["n_entry_halted"] = int(stage2.as_bool(m["halted_entry"]).sum())
        row["n_entry_missing"] = int(m["halted_entry"].isna().sum())
    return row


def run_check(paths: DataPaths, out: str | None) -> tuple[Path, Path]:
    """Calendar and universe counts for the 25 past dates and 2026-09-30. No returns."""
    con = connect_stage2(paths)
    try:
        sessions = load_sessions(con)
        schedules = {t: stage2.schedule(sessions, t) for t in CHECK_DATES}
        u_days = [s.signal_day for s in schedules.values() if s.signal_day is not None]
        entries = [s.entry for s in schedules.values() if s.entry is not None]
        info = signal_day_info(con, u_days)
        lines = mcap_lines(con, u_days)
        ent = entry_status(con, entries) if entries else None
        inputs = prepare(con, list(CHECK_DATES))
    finally:
        con.close()
    feats = inputs["feats"]
    rows = []
    for t in CHECK_DATES:
        sch = schedules[t]
        if sch.signal_day is None:
            continue
        f_t = feats[pd.to_datetime(feats["decision_date"]).dt.date == t]
        i_u = info[info["signal_day"] == sch.signal_day]
        m_u = lines[lines["signal_day"] == sch.signal_day]
        e_h = ent[ent["entry_day"] == sch.entry] if (ent is not None and sch.entry) else None
        rows.append(check_row(t, sch, f_t, i_u, m_u, e_h))
    table = pd.DataFrame(rows)
    out_dir = _new_dir(paths, "stage2_check", out)
    tsv = _write_tsv(out_dir / "stage2_check.tsv", table)
    git = manifest.git_state(manifest.repo_root())
    extra = {
        "rules": {
            "signal_day": f"last session <= t + {stage2.SIGNAL_LAG_DAYS} days",
            "entry": f"{stage2.ENTRY_SESSIONS}th session after the signal day",
            "exit": f"last session <= entry + {stage2.HOLD_MONTHS} months; beyond data when the data ends first",
        },
        "decision_dates": [_iso(d) for d in CHECK_DATES],
        "sessions_max": _iso(sessions[-1]),
        "table_sha256": _sha_bytes(tsv.read_bytes()),
        "dirty_recorded": bool(git["dirty"]),
        "returns_computed": False,
        "predictions_computed": False,
    }
    m = manifest.build_manifest(
        "STAGE2_CHECK",
        paths,
        code=git,
        inputs=manifest.describe_inputs(paths, include_r2=False),
        extra=_py(extra),
    )
    return out_dir, manifest.write_once(out_dir / "manifest_stage2_check.json", m)


# ---------------------------------------------------------------------------
# Tranche building shared by past and forward
# ---------------------------------------------------------------------------


def build_tranche(
    t: date,
    sch: stage2.Schedule,
    feats_t: pd.DataFrame,
    preds_t: pd.DataFrame,
    info_u: pd.DataFrame,
    lines_u: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Baskets (all variants) and a count record for one tranche. No returns."""
    co = stage2.company_mcap(lines_u)
    uni = stage2.assemble(feats_t, info_u, co, preds_t)
    ok = uni[uni["in_stage2"]]
    baskets = stage2.build_baskets(ok)
    baskets.insert(0, "decision_date", t)
    rec: dict[str, Any] = {
        "decision_date": t,
        "signal_day": sch.signal_day,
        "n_eligible_t": int(stage2.as_bool(feats_t["input_eligible"]).sum()),
        "n_universe": int(len(ok)),
        "n_pref_addon": int(stage2.as_bool(ok["has_pref"]).sum()),
        "n_pref_gt_10pct": int((ok["pref_ratio"] > 0.10).sum()),
        "n_raw_close_gt_416667": int((ok["raw_close_u"] > stage2.PRICE_PER_NAME_LIMIT).sum()),
        "n_ev_signal": int(ok["M_EV"].notna().sum()),
    }
    for reason in stage2.DROP_ORDER:
        rec[f"drop_{reason}"] = int((uni["drop_reason"] == reason).sum())
    rec.update(stage2.tranche_exposure(ok, baskets, ksic_section))
    rec["n_over_price_limit_in_M"] = int(
        baskets[baskets["variant"] == "M"]["ticker"]
        .map(ok.set_index("ticker")["raw_close_u"])
        .gt(stage2.PRICE_PER_NAME_LIMIT)
        .sum()
    )
    return baskets, rec


# Stub (record-only item). Quality score C (company score) is stored only as 2026-09-30 and
# 2026-10-18 snapshots (stock_data/kr/output/quality_score_company_*), not as a point-in-time
# history for 2019..2025, so the "drop the bottom 20% by score C" variant cannot be joined for
# the past tranches. Left as a stub; no value is produced.
def quality_c_exclusion_variant(*_args: Any, **_kwargs: Any) -> None:
    return None


def _training_inputs(
    paths: DataPaths,
    inputs: dict[str, pd.DataFrame],
    grant: seal.SealGrant,
    train_dates: tuple[date, ...],
    extra_date: date | None,
) -> tuple[pd.DataFrame, list[date]]:
    """Training dataset (labels of ``train_dates``, guarded) plus the eval rows of ``extra_date``.

    Which training rows count at an evaluation date is decided by rule A20 inside ``train``
    (``label_pub_date <= t``), not here.
    """
    feats = inputs["feats"]
    lab, _ = _load_labels(paths, feats, inputs["receipts"], tuple(train_dates), "oi_y1", grant)
    ds = build_train_dataset(feats, lab, train_dates)
    dates = list(train_dates)
    if extra_date is not None:
        ds = pd.concat([ds, eval_frame(feats, extra_date)], ignore_index=True)
        dates.append(extra_date)
    return ds, dates


def _common_checks(
    paths: DataPaths, chosen_path: Path | None, chosen_sha256: str
) -> dict[str, Any]:
    P.verify_input_hashes(paths)
    return _load_chosen(chosen_path or paths.out_root / CHOSEN_REL, chosen_sha256)


def _ledger(paths: DataPaths, ledger_path: str | Path | None) -> Path:
    ledger = Path(ledger_path) if ledger_path else seal.default_ledger_path()
    if ledger.parent.resolve() != paths.out_root.resolve():
        raise SealedAbort("seal ledger is not under the output root of the data paths")
    return ledger


# ---------------------------------------------------------------------------
# Past run (one shot)
# ---------------------------------------------------------------------------


def run_past(
    paths: DataPaths,
    out: str | None,
    *,
    allow_reopen: bool = False,
    ledger_path: str | Path | None = None,
    repo_dir: str | Path | None = None,
    git_runner: seal.GitRunner | None = None,
    chosen_path: Path | None = None,
    chosen_sha256: str = CHOSEN_SHA256,
    f4_manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    """Open ``STAGE2_PAST`` once and compute the 25 past tranches with returns."""
    t_start = datetime.now(UTC)
    git = _refuse_dirty()
    chosen = _common_checks(paths, chosen_path, chosen_sha256)
    ledger = _ledger(paths, ledger_path)
    con = connect_stage2(paths)
    try:
        sessions = load_sessions(con)
        scheds = {t: stage2.schedule(sessions, t) for t in PAST_DATES}
        missing = [t.isoformat() for t, s in scheds.items() if s.exit is None]
        if missing:
            raise SealedAbort(f"exit beyond data for: {', '.join(missing)}")
        u_days = [s.signal_day for s in scheds.values()]
        info = signal_day_info(con, u_days)
        lines = mcap_lines(con, u_days)
        inputs = prepare(con, list(TRAIN_DATES))
        index_close = index_series(con)
        etf_close, etf_dist = etf_series(con)

        # integrity check of the stage 1 sealed label table, still before the seal is opened
        f4 = _f4_check(
            paths,
            {"feats": inputs["feats"], "receipts": inputs["receipts"]},
            make_mode(False),
            f4_manifest_path or paths.out_root / F4_REL,
        )
        out_dir = _new_dir(paths, "stage2_past", out)
        try:
            grant = seal.open_seal_grant(
                seal.Phase.STAGE2_PAST,
                repo_dir=repo_dir or manifest.repo_root(),
                ledger_path=ledger,
                allow_reopen=allow_reopen,
                git_runner=git_runner,
            )
        except BaseException:
            if not any(out_dir.iterdir()):
                out_dir.rmdir()
            raise
        ds, dates = _training_inputs(paths, inputs, grant, TRAIN_DATES, None)
        preds = predict(ds, list(PAST_DATES), dates, grant, chosen["HGB"])
        feats = inputs["feats"]
        feat_date = pd.to_datetime(feats["decision_date"]).dt.date
        pred_date = pd.to_datetime(preds["decision_date"]).dt.date
        baskets_all, recs, tranche_rows = [], [], []
        for t in PAST_DATES:
            sch = scheds[t]
            f_t = feats[feat_date == t]
            p_t = preds[pred_date == t]
            i_u = info[info["signal_day"] == sch.signal_day]
            l_u = lines[lines["signal_day"] == sch.signal_day]
            baskets, rec = build_tranche(t, sch, f_t, p_t, i_u, l_u)
            tickers = sorted(set(baskets["ticker"]))
            closes = stage2.closes_frame(load_closes(con, tickers, sch.entry, sch.exit))
            sc = stage2.score_baskets(
                baskets,
                closes,
                sessions,
                sch.entry,
                sch.exit,
                index_close=index_close,
                etf_close=etf_close,
                etf_dist=etf_dist,
            )
            baskets_all.append(baskets)
            recs.append({**rec, "entry": sch.entry, "exit": sch.exit})
            tranche_rows.append({"decision_date": t, "signal_day": sch.signal_day, **sc})
    finally:
        con.close()
    tab = pd.DataFrame(tranche_rows)
    files = {
        "baskets": _write_tsv(out_dir / "baskets.tsv", pd.concat(baskets_all, ignore_index=True)),
        "tranches": _write_tsv(out_dir / "tranches.tsv", tab),
        "tranche_records": _write_tsv(
            out_dir / "tranche_records.tsv", pd.json_normalize(_py(recs))
        ),
    }
    summary = stage2.summarize_tranches(tab)
    for suffix in ("_EV", "_cap", "_N15", "_N50"):
        summary[f"M-B1{suffix}"] = stage2.summarize_tranches(
            tab, m=f"M{suffix}_net", b1=f"B1{suffix}_net"
        )
    summary["M-B1_net100"] = stage2.summarize_tranches(tab, m="M_net100", b1="B1_net100")
    summary["M-B1_net130"] = stage2.summarize_tranches(tab, m="M_net130", b1="B1_net130")
    summary["M-B1_div"] = stage2.summarize_tranches(tab, m="M_net_div", b1="B1_net_div")
    summary["M-B1_B1u"] = stage2.summarize_tranches(tab, m="M_net", b1="B1u_net")
    summary["M-B0_tr"] = stage2.summarize_tranches(tab, m="M_net", b1="B1_net", b0="B0_tr")
    sp = out_dir / "summary.json"
    manifest.write_once(sp, _py(summary))
    files["summary"] = sp
    extra = {
        "mode": "stage2_past",
        "run_at": t_start.isoformat(timespec="seconds"),
        "elapsed_s": round((datetime.now(UTC) - t_start).total_seconds(), 1),
        "ledger_record": grant.record,
        "chosen": {"sha256": chosen["sha256"], "HGB": chosen["HGB"].spec_id},
        "f4_check": f4,
        "decision_dates": [_iso(d) for d in PAST_DATES],
        "train_dates": [_iso(d) for d in TRAIN_DATES],
        "quality_c_variant": "stub: not computed (no point-in-time score C history for 2019..2025)",
        "output_sha256": {k: _sha_bytes(p.read_bytes()) for k, p in files.items()},
        "dirty_recorded": bool(git["dirty"]),
    }
    m = manifest.build_manifest("STAGE2_PAST", paths, code=git, extra=_py(extra))
    return out_dir, manifest.write_once(out_dir / "manifest_stage2_past.json", m)


# ---------------------------------------------------------------------------
# Forward basket file
# ---------------------------------------------------------------------------


def run_forward(
    paths: DataPaths,
    decision_date: date,
    out: str | None,
    *,
    today: date | None = None,
    allow_reopen: bool = False,
    ledger_path: str | Path | None = None,
    repo_dir: str | Path | None = None,
    git_runner: seal.GitRunner | None = None,
    chosen_path: Path | None = None,
    chosen_sha256: str = CHOSEN_SHA256,
) -> tuple[Path, Path]:
    """Write the basket file of one forward tranche (no returns) with its sha256 and plan."""
    today = today or datetime.now(UTC).date()
    git = _refuse_dirty()
    chosen = _common_checks(paths, chosen_path, chosen_sha256)
    ledger = _ledger(paths, ledger_path)
    con = connect_stage2(paths)
    try:
        sessions = load_sessions(con)
        u = stage2.signal_day(sessions, decision_date)
        if u is None or not stage2.signal_day_known(sessions, decision_date):
            raise SealedAbort("the data does not reach the signal day yet; take a newer snapshot")
        entry, est = stage2.entry_day_estimate(sessions, u)
        sch = stage2.Schedule(decision_date, u, entry, stage2.exit_day(sessions, entry))
        info = signal_day_info(con, [u])
        lines = mcap_lines(con, [u])
        train_dates = seal.forward_train_dates(decision_date, today=today, ledger_path=ledger)
        inputs = prepare(con, [*train_dates, decision_date])
        out_dir = _new_dir(paths, f"stage2_forward_{decision_date.isoformat()}", out)
        try:
            grant = seal.open_forward_grant(
                decision_date,
                repo_dir=repo_dir or manifest.repo_root(),
                ledger_path=ledger,
                today=today,
                allow_reopen=allow_reopen,
                git_runner=git_runner,
            )
        except BaseException:
            if not any(out_dir.iterdir()):
                out_dir.rmdir()
            raise
        ds, dates = _training_inputs(paths, inputs, grant, train_dates, decision_date)
        preds = predict(ds, [decision_date], dates, grant, chosen["HGB"])
    finally:
        con.close()
    feats = inputs["feats"]
    f_t = feats[pd.to_datetime(feats["decision_date"]).dt.date == decision_date]
    baskets, rec = build_tranche(decision_date, sch, f_t, preds, info, lines)
    path = out_dir / f"baskets_{decision_date.isoformat()}.tsv"
    _write_tsv(path, baskets)
    sha = _sha_bytes(path.read_bytes())
    is_late = stage2.late_flag(today, entry)
    meta = {
        "kind": "STAGE2_FORWARD_BASKET",
        "decision_date": decision_date,
        "signal_day": u,
        "entry_planned": entry,
        "entry_is_estimate": est,
        "exit_planned": stage2.exit_target(entry),
        "created_date": today,
        "late": is_late,
        "late_rule": "entry = first session after created_date; excluded from the main estimate",
        "main_estimate": bool((not is_late) and decision_date >= stage2.FORWARD_FIRST_MAIN),
        "basket_file": path.name,
        "basket_sha256": sha,
        "counts": rec,
        "ledger_record": grant.record,
        "chosen_sha256": chosen["sha256"],
    }
    meta_path = manifest.write_once(out_dir / f"meta_{decision_date.isoformat()}.json", _py(meta))
    m = manifest.build_manifest(
        "STAGE2_FORWARD",
        paths,
        code=git,
        extra=_py({"meta_sha256": _sha_bytes(meta_path.read_bytes())}),
    )
    return out_dir, manifest.write_once(
        out_dir / f"manifest_stage2_forward_{decision_date.isoformat()}.json", m
    )


# ---------------------------------------------------------------------------
# Scoring committed forward baskets
# ---------------------------------------------------------------------------


def score_meta(
    con, meta_path: Path, sessions: list[date], index_close: pd.Series, etf_close, etf_dist
) -> dict[str, Any] | None:
    """Score one committed basket file; None while the exit day is beyond the data."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bpath = meta_path.parent / meta["basket_file"]
    if _sha_bytes(bpath.read_bytes()) != meta["basket_sha256"]:
        raise SealedAbort(f"basket file sha256 differs from its meta: {bpath.name}")
    t = date.fromisoformat(meta["decision_date"])
    u = date.fromisoformat(meta["signal_day"])
    if meta["late"]:
        created = date.fromisoformat(meta["created_date"])
        nxt = [s for s in sessions if s > created]
        entry = nxt[0] if nxt else None
    else:
        entry = stage2.entry_day(sessions, u)
    exit_ = stage2.exit_day(sessions, entry) if entry is not None else None
    if entry is None or exit_ is None:
        return None
    baskets = pd.read_csv(bpath, sep="\t", dtype={"ticker": str, "corp_code": str})
    closes = stage2.closes_frame(load_closes(con, sorted(set(baskets["ticker"])), entry, exit_))
    sc = stage2.score_baskets(
        baskets, closes, sessions, entry, exit_,
        index_close=index_close, etf_close=etf_close, etf_dist=etf_dist,
    )  # fmt: skip
    return {
        "decision_date": t,
        "signal_day": u,
        "late": bool(meta["late"]),
        "main_estimate": bool(meta["main_estimate"]),
        "entry_matches_plan": entry == date.fromisoformat(meta["entry_planned"]),
        **sc,
    }


def run_score(
    paths: DataPaths, meta_files: list[Path], out: str | None, past_summary: Path | None = None
) -> tuple[Path, Path]:
    """Score matured forward tranches from committed basket files; write one table and summary."""
    git = manifest.git_state(manifest.repo_root())
    con = connect_stage2(paths)
    try:
        sessions = load_sessions(con)
        index_close = index_series(con)
        etf_close, etf_dist = etf_series(con)
        rows, pending = [], []
        for mp in sorted(meta_files):
            r = score_meta(con, mp, sessions, index_close, etf_close, etf_dist)
            (pending if r is None else rows).append(mp.name if r is None else r)
    finally:
        con.close()
    out_dir = _new_dir(paths, "stage2_score", out)
    tab = pd.DataFrame(rows)
    files: dict[str, Path] = {}
    summary: dict[str, Any] = {"pending_meta_files": pending, "n_scored": len(rows)}
    if len(tab):
        files["tranches"] = _write_tsv(out_dir / "tranches.tsv", tab)
        main = tab[tab["main_estimate"]]
        summary["main_estimate"] = stage2.summarize_tranches(main)
        summary["late_or_early_record_only"] = {
            "n": int(len(tab) - len(main)),
            "decision_dates": [_iso(d) for d in tab.loc[~tab["main_estimate"], "decision_date"]],
        }
        if past_summary is not None:
            past = json.loads(past_summary.read_text(encoding="utf-8"))
            for k in ("M-B1", "M-B0"):
                f, p = summary["main_estimate"][k]["mean"], past[k]["mean"]
                summary[f"decay_ratio_{k}"] = stage2.decay_ratio(f, p)
    sp = out_dir / "score_summary.json"
    manifest.write_once(sp, _py(summary))
    files["summary"] = sp
    extra = {
        "meta_files": [p.name for p in meta_files],
        "output_sha256": {k: _sha_bytes(p.read_bytes()) for k, p in files.items()},
        "dirty_recorded": bool(git["dirty"]),
    }
    m = manifest.build_manifest("STAGE2_SCORE", paths, code=git, extra=_py(extra))
    return out_dir, manifest.write_once(out_dir / "manifest_stage2_score.json", m)
