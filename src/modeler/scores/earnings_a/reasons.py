"""Label-missing reason dictionary (05 §4, decision A24).

One output row per presence row that is not ``ok`` (``ok`` = row_present, no
null_reason, not corr-only). The input is ``labels.presence_view(...)``: it has
no value columns, and this module refuses any frame that does.

Rule order, first match wins:

1. IMMATURE: deadline = target_pe + 45d (interim) / +90d (annual) is after
   ``raw_max_date``. Applies to every non-ok row except corr-only rows.
2. SRC_MISSING: ticker traded on/after the deadline AND the receipt list has a
   periodic report (사업/반기/분기보고서, ``(YYYY.MM)`` == target_pe) for the
   period, yet the row is unusable. Corr-only rows are excluded here (their
   receipt is the correction itself) so rule 3 can fire. Detail:
   ``MART_NULL:<reason>`` / ``BASIS_CHANGED`` / ``NO_ROW``.
3. ORIG_MISSING_CORR_ONLY: row_present and is_corr_only.
4. MERGED: stock_master DELISTED (no ACTIVE row for the ticker) and an
   own-company merger receipt (회사합병결정|합병결정|합병등종료보고서, names with
   자회사|종속회사|출자법인|피보증 excluded) dated in
   [last_seen_date-365d, last_seen_date+30d]; the earliest such receipt is the
   evidence.
5. DELISTED: stock_master DELISTED (no ACTIVE row), last_seen_date < deadline.
6. REPORT_STOPPED: last_trade_date >= deadline but no periodic receipt.
7. UNCONFIRMED.

``reason_detail`` carries ``MART_NULL:<x>`` / ``BASIS_CHANGED`` for rows that
fall to later codes too, when informative. Counts only leave via
``summarize_reasons``.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from modeler.scores.earnings_a.spec import DEADLINE_DAYS, SNAPSHOT

FORBIDDEN_COLUMNS = ("value_dec", "y", "y_clip", "oi_sum_dec", "standalone_value", "ttm_value")

OUTPUT_COLUMNS = [
    "ticker",
    "corp_code",
    "target_quarter_end",
    "target_kind",
    "decision_date",
    "k",
    "is_complete",
    "reason_code",
    "reason_detail",
    "reason_evidence",
    "evidence_date",
    "effective_date",
    "confirmed_at",
    "source_snapshot",
]

_PERIODIC_RE = r"(?:사업|반기|분기)보고서.*\((\d{4})\.(\d{2})\)"
_MERGER_RE = r"회사합병결정|합병결정|합병등종료보고서"
_NOT_OWN_RE = r"자회사|종속회사|출자법인|피보증"


def _iso(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d").where(s.notna(), None)


def _receipt_dates(receipts: pd.DataFrame) -> pd.Series:
    if "rcept_dt" in receipts.columns:
        return pd.to_datetime(receipts["rcept_dt"])
    return pd.to_datetime(receipts["rcept_no"].astype(str).str[:8], format="%Y%m%d")


def build_reason_dictionary(
    presence: pd.DataFrame,
    receipts: pd.DataFrame,
    stock_master: pd.DataFrame,
    last_trade: pd.DataFrame,
    *,
    run_at: datetime,
    source_snapshot: str = SNAPSHOT,
    raw_max_date: date = date(2026, 10, 22),
    metric: str = "operating_income",
) -> pd.DataFrame:
    """Return the reason dictionary for all non-ok presence rows."""
    del metric  # reserved: one dictionary per metric, rules do not depend on it
    bad_cols = [c for c in FORBIDDEN_COLUMNS if c in presence.columns]
    if bad_cols:
        raise ValueError(f"presence must not carry value columns: {bad_cols}")

    ok = (
        presence["row_present"].astype(bool)
        & presence["null_reason"].isna()
        & ~presence["is_corr_only"].astype(bool)
    )
    p = presence.loc[~ok].copy().reset_index(drop=True)
    if p.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pe = pd.to_datetime(p["target_pe"])
    fye = p["fye_month"] if "fye_month" in p.columns else pd.Series(12, index=p.index)
    fye = fye.fillna(12).astype(int)
    annual = pe.dt.month.to_numpy() == fye.to_numpy()
    kind = np.where(annual, "annual", "interim")
    days = np.where(annual, DEADLINE_DAYS["annual"], DEADLINE_DAYS["interim"])
    deadline = pe + pd.to_timedelta(days, unit="D")

    present = p["row_present"].astype(bool)
    other = p["other_basis_present"].astype(bool)
    has_null = present & p["null_reason"].notna()
    detail = pd.Series(None, index=p.index, dtype=object)
    detail[has_null] = "MART_NULL:" + p.loc[has_null, "null_reason"].astype(str)
    detail[~present & other] = "BASIS_CHANGED"
    detail[~present & ~other] = "NO_ROW"
    detail_late = detail.where(detail != "NO_ROW", None)

    lt = last_trade.drop_duplicates("ticker").set_index("ticker")["last_trade_date"]

    # periodic receipts per (corp_code, year-month)
    rc = receipts.copy()
    rc["_dt"] = _receipt_dates(rc)
    ext = rc["report_nm"].astype(str).str.extract(_PERIODIC_RE)
    per = rc.loc[ext[0].notna()].copy()
    per["_ym"] = ext.loc[per.index, 0].astype(int) * 12 + ext.loc[per.index, 1].astype(int)
    per = per.sort_values(["rcept_no"]).drop_duplicates(["corp_code", "_ym"])
    per = per[["corp_code", "_ym", "rcept_no", "report_nm", "_dt"]].rename(
        columns={"rcept_no": "_prc", "report_nm": "_prn", "_dt": "_prd"}
    )
    p["_ym"] = pe.dt.year * 12 + pe.dt.month
    p = p.merge(per, on=["corp_code", "_ym"], how="left")
    p = p.reset_index(drop=True)
    has_receipt = p["_prc"].notna().to_numpy()

    # stock master: DELISTED without any ACTIVE row
    sm = stock_master.copy()
    g = sm.groupby("ticker")
    any_active = g["status"].apply(lambda s: bool((s == "ACTIVE").any()))
    any_del = g["status"].apply(lambda s: bool((s == "DELISTED").any()))
    last_seen = g["last_seen_date"].max()
    delisted_t = any_del & ~any_active
    is_del = p["ticker"].map(delisted_t).astype("boolean").fillna(False).to_numpy(dtype=bool)
    seen = pd.to_datetime(p["ticker"].map(last_seen))

    # own-company merger receipts
    mg = rc[
        rc["report_nm"].astype(str).str.contains(_MERGER_RE, regex=True)
        & ~rc["report_nm"].astype(str).str.contains(_NOT_OWN_RE, regex=True)
    ][["corp_code", "rcept_no", "report_nm", "_dt"]]
    cand = p.loc[is_del, ["ticker", "corp_code"]].drop_duplicates()
    cand["_seen"] = pd.to_datetime(cand["ticker"].map(last_seen))
    cand = cand.merge(mg, on="corp_code", how="inner")
    cand = cand[
        (cand["_dt"] >= cand["_seen"] - pd.Timedelta(365, "D"))
        & (cand["_dt"] <= cand["_seen"] + pd.Timedelta(30, "D"))
    ]
    cand = cand.sort_values(["ticker", "_dt", "rcept_no"]).drop_duplicates("ticker")
    merg = cand.set_index("ticker")
    mrc = p["ticker"].map(merg["rcept_no"]) if len(merg) else pd.Series(None, index=p.index)
    mdt = p["ticker"].map(merg["_dt"]) if len(merg) else pd.Series(pd.NaT, index=p.index)
    has_merger = mrc.notna().to_numpy() & is_del

    deadline = deadline.reset_index(drop=True)
    pe = pe.reset_index(drop=True)
    present = p["row_present"].astype(bool).to_numpy()
    corr = p["is_corr_only"].astype(bool).to_numpy()
    detail = detail.reset_index(drop=True)
    detail_late = detail_late.reset_index(drop=True)
    traded_after = (pd.to_datetime(p["ticker"].map(lt)) >= deadline).fillna(False).to_numpy()

    immature = (deadline > pd.Timestamp(raw_max_date)).to_numpy() & ~corr
    src_missing = traded_after & has_receipt & ~corr
    delisted = is_del & (seen < deadline).fillna(False).to_numpy()
    stopped = traded_after & ~has_receipt

    code = np.select(
        [immature, src_missing, corr, has_merger, delisted, stopped],
        [
            "IMMATURE",
            "SRC_MISSING",
            "ORIG_MISSING_CORR_ONLY",
            "MERGED",
            "DELISTED",
            "REPORT_STOPPED",
        ],
        default="UNCONFIRMED",
    )
    code_s = pd.Series(code, index=p.index)

    out_detail = detail.where(code_s == "SRC_MISSING", detail_late)

    evidence = pd.Series(None, index=p.index, dtype=object)
    ev_date = pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns]")
    m = (code_s == "SRC_MISSING").to_numpy()
    evidence[m] = p.loc[m, "_prc"] + " | " + p.loc[m, "_prn"].astype(str)
    ev_date[m] = pd.to_datetime(p.loc[m, "_prd"])
    m = (code_s == "ORIG_MISSING_CORR_ONLY").to_numpy()
    evidence[m] = p.loc[m, "rcept_no"]
    if "available_from" in p.columns:
        ev_date[m] = pd.to_datetime(p.loc[m, "available_from"])
    m = (code_s == "MERGED").to_numpy()
    evidence[m] = mrc[m]
    ev_date[m] = pd.to_datetime(mdt[m])
    m = (code_s == "DELISTED").to_numpy()
    evidence[m] = "stock_master:status=DELISTED"
    ev_date[m] = seen[m]

    out = pd.DataFrame(
        {
            "ticker": p["ticker"],
            "corp_code": p["corp_code"],
            "target_quarter_end": p["target_pe"],
            "target_kind": kind,
            "decision_date": p["decision_date"],
            "k": p["k"],
            "is_complete": False,
            "reason_code": code_s,
            "reason_detail": out_detail,
            "reason_evidence": evidence,
            "evidence_date": _iso(ev_date),
            "effective_date": None,
            "confirmed_at": run_at.isoformat(),
            "source_snapshot": source_snapshot,
        }
    )
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def summarize_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per (decision_date, reason_code); counts only."""
    if df.empty:
        return pd.DataFrame(columns=["decision_date", "reason_code", "n"])
    out = df.groupby(["decision_date", "reason_code"]).size().rename("n").reset_index()
    out["n"] = out["n"].astype(int)
    return out
