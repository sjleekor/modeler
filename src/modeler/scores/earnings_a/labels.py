"""Labels for research A stage 1: next-4-quarter standalone sums from the dl_v4 mart (r2).

This is the ONLY module that opens ``paths.r2_parquet``. Every public function is
guarded by ``seal.guarded`` (genuine grant plus allowed ``decision_dates``). Values travel
as exact decimal strings (``value_dec``) and are summed with ``Decimal``; only the final
ratio ``y`` is a float.

Quarters are matched by calendar month index (year*12+month) of ``statement_period_end``,
never by ``seq_key`` (the mart renumbers it per fiscal year).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

import duckdb
import numpy as np
import pandas as pd

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.paths import DataPaths
from modeler.scores.earnings_a.seal import Phase

CLIP_Q = (0.01, 0.99)
_QUANT = Decimal("0.0001")
_BASES = ("CFS", "OFS")
TARGETS: dict[str, tuple[str, int]] = {
    "oi_y1": ("operating_income", 1),
    "ocf_y1": ("operating_cash_flow", 1),
    "oi_y2": ("operating_income", 2),
}


def _midx(d: date) -> int:
    return d.year * 12 + d.month - 1


def _month_end(idx: int) -> date:
    y, m = divmod(idx, 12)
    return (pd.Timestamp(year=y, month=m + 1, day=1) + pd.offsets.MonthEnd(0)).date()


def _fmt(x: Decimal) -> str:
    return format(x.quantize(_QUANT), "f")


def _quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _norm_dates(dates) -> list[date]:
    return sorted({seal._to_date(d) for d in dates})


def _isnone(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT


# ---------------------------------------------------------------------------
# Reading (the only place that opens r2)
# ---------------------------------------------------------------------------


def _read_r2(paths: DataPaths, metrics: tuple[str, ...], lo: int, hi: int) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET disabled_optimizers='statistics_propagation'")
        mlist = ", ".join(_quote(m) for m in metrics)
        sql = f"""
            SELECT ticker, corp_code, metric_code, fs_basis,
                   statement_period_end,
                   (year(statement_period_end) * 12 + month(statement_period_end) - 1) AS pe_month,
                   rcept_no, available_from,
                   CAST(standalone_value AS VARCHAR) AS value_dec,
                   standalone_null_reason
            FROM read_parquet({_quote(str(paths.r2_parquet))})
            WHERE metric_code IN ({mlist})
              AND fs_basis IN ('CFS', 'OFS')
              AND (year(statement_period_end) * 12 + month(statement_period_end) - 1)
                  BETWEEN {int(lo)} AND {int(hi)}
        """
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    df["value_dec"] = [None if _isnone(v) else _fmt(Decimal(v)) for v in df["value_dec"]]
    df["statement_period_end"] = [seal._to_date(v) for v in df["statement_period_end"]]
    df["available_from"] = [None if _isnone(v) else seal._to_date(v) for v in df["available_from"]]
    return df


@seal.guarded
def load_label_rows(
    paths: DataPaths,
    *,
    decision_dates,
    grant,
    horizon: int = 1,
    metrics: tuple[str, ...] = ("operating_income", "operating_cash_flow", "total_assets"),
) -> pd.DataFrame:
    """Read r2 rows whose period end falls in the target window of ``decision_dates``."""
    if horizon not in (1, 2):
        raise ValueError("horizon must be 1 or 2")
    ds = _norm_dates(decision_dates)
    if not ds:
        raise ValueError("decision_dates is empty")
    lo = _midx(ds[0]) + 3
    hi = _midx(ds[-1]) + 12 * horizon
    if grant.phase is Phase.DEV:
        hi = min(hi, _midx(seal.DEV_LABEL_END_MAX))
    return _read_r2(paths, tuple(metrics), lo, hi)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _with_pe_month(r2: pd.DataFrame) -> pd.DataFrame:
    if "pe_month" in r2.columns:
        return r2
    out = r2.copy()
    out["pe_month"] = [_midx(seal._to_date(v)) for v in out["statement_period_end"]]
    return out


@seal.guarded
def build_label_quarters(
    r2_rows: pd.DataFrame,
    anchor: pd.DataFrame,
    receipts: pd.DataFrame,
    *,
    decision_dates,
    grant,
    metric: str = "operating_income",
    horizon: int = 1,
) -> pd.DataFrame:
    """One row per (decision_date, ticker, k=1..4) with presence and availability facts."""
    if horizon not in (1, 2):
        raise ValueError("horizon must be 1 or 2")
    keep = set(_norm_dates(decision_dates))
    a = anchor.loc[[seal._to_date(d) in keep for d in anchor["decision_date"]]]
    a = a[["decision_date", "ticker", "corp_code", "basis"]].copy()
    a["_t_idx"] = [_midx(seal._to_date(d)) for d in a["decision_date"]]

    base = a.loc[a.index.repeat(4)].reset_index(drop=True)
    base["k"] = np.tile(np.arange(1, 5), len(a))
    off = 12 * (horizon - 1)
    base["pe_month"] = base["_t_idx"] + off + 3 * base["k"]
    base["target_pe"] = [_month_end(int(m)) for m in base["pe_month"]]
    base = base.drop(columns="_t_idx")

    r = _with_pe_month(r2_rows)
    r = r.loc[r["metric_code"] == metric]
    key = ["ticker", "fs_basis", "pe_month"]
    if r.duplicated(key).any():
        raise ValueError("r2 rows are not unique on (ticker, fs_basis, pe_month)")
    cols = ["ticker", "fs_basis", "pe_month", "value_dec", "available_from", "rcept_no"]
    cols.append("standalone_null_reason")
    r = r[cols].rename(
        columns={
            "fs_basis": "basis",
            "standalone_null_reason": "null_reason",
            "pe_month": "pe_month",
        }
    )
    r["row_present"] = True

    out = base.merge(r, on=["ticker", "basis", "pe_month"], how="left")
    out["row_present"] = out["row_present"].eq(True)

    other = r[["ticker", "basis", "pe_month"]].copy()
    other["basis"] = other["basis"].map({"CFS": "OFS", "OFS": "CFS"})
    other["other_basis_present"] = True
    out = out.merge(other, on=["ticker", "basis", "pe_month"], how="left")
    out["other_basis_present"] = out["other_basis_present"].eq(True)

    rc = receipts[["rcept_no", "report_nm"]].drop_duplicates("rcept_no")
    corr = dict(
        zip(
            rc["rcept_no"], rc["report_nm"].fillna("").astype(str).str.contains("정정"), strict=True
        )
    )
    out["is_corr_only"] = [
        bool(corr.get(x, False)) if out_p else False
        for x, out_p in zip(out["rcept_no"], out["row_present"], strict=True)
    ]
    out["null_reason"] = [None if _isnone(v) or v == "" else v for v in out["null_reason"]]
    for c in ("value_dec", "available_from", "rcept_no"):
        out[c] = [None if _isnone(v) else v for v in out[c]]
    out = out.drop(columns="pe_month")
    return out[
        [
            "decision_date",
            "ticker",
            "corp_code",
            "basis",
            "k",
            "target_pe",
            "row_present",
            "value_dec",
            "available_from",
            "rcept_no",
            "null_reason",
            "is_corr_only",
            "other_basis_present",
        ]
    ].reset_index(drop=True)


def _missing_kind(row) -> str:
    if not row.row_present:
        return "BASIS_CHANGED" if row.other_basis_present else "NO_ROW"
    if row.null_reason is not None:
        return f"MART_NULL:{row.null_reason}"
    if row.value_dec is None:
        return "MART_NULL:NULL_VALUE"
    return "CORR_ONLY"


def _quarter_ok(row) -> bool:
    return bool(
        row.row_present
        and row.value_dec is not None
        and row.null_reason is None
        and not row.is_corr_only
    )


def _clip_y(
    y: pd.Series, decision_date: pd.Series, eligible: pd.Series, ok: pd.Series
) -> pd.Series:
    """Clip y per decision date at the 1%/99% bounds of eligible label_ok rows."""
    out = pd.Series(np.nan, index=y.index, dtype=float)
    ydf = pd.DataFrame(
        {
            "y": y.astype(float),
            "d": decision_date,
            "ok": ok.astype(bool),
            "el": eligible.astype(bool),
        }
    )
    for _, g in ydf.groupby("d", sort=False):
        pool = g.loc[g["ok"] & g["el"], "y"].to_numpy(dtype=float)
        if np.isnan(pool).all():
            continue
        lo, hi = np.nanquantile(pool, CLIP_Q, method="linear")
        sel = g.loc[g["ok"]]
        out.loc[sel.index] = sel["y"].clip(lo, hi)
    return out


@seal.guarded
def build_labels(
    r2_rows: pd.DataFrame,
    anchor: pd.DataFrame,
    receipts: pd.DataFrame,
    *,
    decision_dates,
    grant,
    target: str = "oi_y1",
) -> pd.DataFrame:
    """Per (decision_date, ticker) label: four-quarter sum divided by A_t."""
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    metric, horizon = TARGETS[target]
    ds = _norm_dates(decision_dates)
    lq = build_label_quarters(
        r2_rows, anchor, receipts, decision_dates=ds, grant=grant, metric=metric, horizon=horizon
    )
    keys = ["decision_date", "ticker"]
    a = anchor.loc[[seal._to_date(d) in set(ds) for d in anchor["decision_date"]]].copy()
    if "input_eligible" not in a.columns:
        a["input_eligible"] = True
    a = a[[*keys, "a_t_dec", "input_eligible"]]

    lq = lq.copy()
    lq["_ok"] = [_quarter_ok(r) for r in lq.itertuples(index=False)]
    rows = []
    for (d, tk), g in lq.groupby(keys, sort=False):
        g = g.sort_values("k")
        first = g.iloc[0]
        bad = g.loc[~g["_ok"]]
        ok = bad.empty
        rec = {
            "decision_date": d,
            "ticker": tk,
            "corp_code": first["corp_code"],
            "basis": first["basis"],
            "label_last_quarter_end": g.iloc[-1]["target_pe"],
            "label_ok": ok,
            "target": target,
            "horizon": horizon,
            "missing_k": None,
            "missing_kind": None,
            "oi_sum_dec": None,
            "label_pub_date": None,
        }
        for i, rn in enumerate(g["rcept_no"].tolist(), 1):
            rec[f"rcept_no_{i}"] = rn
        if ok:
            with localcontext() as ctx:
                ctx.prec = 60
                rec["oi_sum_dec"] = _fmt(sum((Decimal(v) for v in g["value_dec"]), Decimal(0)))
            rec["label_pub_date"] = max(g["available_from"])
        else:
            b = bad.iloc[0]
            rec["missing_k"] = int(b["k"])
            rec["missing_kind"] = _missing_kind(b)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out.merge(a, on=keys, how="left")

    ys = []
    for s, at in zip(out["oi_sum_dec"], out["a_t_dec"], strict=True):
        if s is None or _isnone(at) or Decimal(at) <= 0:
            ys.append(np.nan)
        else:
            with localcontext() as ctx:
                ctx.prec = 60
                ys.append(float(Decimal(s) / Decimal(at)))
    out["y"] = ys
    out["y_clip"] = _clip_y(
        out["y"], out["decision_date"], out["input_eligible"], out["label_ok"] & out["y"].notna()
    )
    cols = [
        "decision_date",
        "ticker",
        "corp_code",
        "basis",
        "a_t_dec",
        "oi_sum_dec",
        "y",
        "y_clip",
        "label_pub_date",
        "label_last_quarter_end",
        "label_ok",
        "target",
        "horizon",
        "rcept_no_1",
        "rcept_no_2",
        "rcept_no_3",
        "rcept_no_4",
        "missing_k",
        "missing_kind",
    ]
    return out[cols].reset_index(drop=True)


def presence_view(lq: pd.DataFrame) -> pd.DataFrame:
    """Drop value columns; what the reason dictionary may see."""
    return lq.drop(columns=[c for c in ("value_dec",) if c in lq.columns]).copy()
