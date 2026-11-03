"""Input features for earnings forecast A (preregistration section 5.3).

Label-free: this module reads only the v4p view and the universe frame. It never
touches the dl_v4 / r2 label mart. Every value used for a decision date ``t`` must
have an availability date on or before ``t``; a component that is not available
stays NaN. Nothing is filled with zero.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from modeler.scores.earnings_a.spec import (
    A_T_MAX_STALE_MONTHS,
    BINARY_FEATURES,
    CLIP_Q,
    FEATURES,
    METRICS,
    QUARTER_MONTHS,
)

BASES: tuple[str, ...] = ("CFS", "OFS")
_SHORT_BY_CODE = {code: short for short, code in METRICS.items()}
_UNIVERSE_PASSTHROUGH = (
    "session_date",
    "market",
    "ksic2",
    "is_financial",
    "is_holding",
    "p6_tradable",
    "in_universe",
    "exclusion_reason",
)
_ROW_COLS = (
    "ticker",
    "corp_code",
    "metric_code",
    "fs_basis",
    "statement_period_end",
    "available_from",
    "standalone_value",
    "ttm_value",
    "ttm_complete",
    "ttm_available_from",
    "ttm_includes_conflict",
    "fiscal_year_end",
)


def _mi(s: pd.Series) -> pd.Series:
    """Month index (year * 12 + month - 1) of a datetime Series; NaN for NaT."""
    return s.dt.year * 12 + s.dt.month - 1


def load_input_rows(con) -> pd.DataFrame:
    """Read the 12 METRICS rows of view ``v4p`` for fs_basis CFS and OFS.

    ``value_dec`` is the exact decimal string of standalone_value, kept for
    total_assets only (NULL for the other metrics).
    """
    codes = ", ".join(f"'{c}'" for c in METRICS.values())
    sql = f"""
        SELECT ticker, corp_code, metric_code, fs_basis,
               statement_period_end, available_from,
               CAST(standalone_value AS DOUBLE) AS standalone_value,
               CAST(ttm_value AS DOUBLE) AS ttm_value,
               ttm_complete, ttm_available_from, ttm_includes_conflict,
               fiscal_year_end,
               CASE WHEN metric_code = 'total_assets'
                    THEN CAST(CAST(standalone_value AS DECIMAL(38,4)) AS VARCHAR) END AS value_dec
        FROM v4p
        WHERE metric_code IN ({codes}) AND fs_basis IN ('CFS', 'OFS')
    """
    rows = con.execute(sql).fetchdf()
    return _prep_rows(rows)


def _prep_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and add ``pe_month``. Returns a copy."""
    out = rows.copy()
    for col in ("statement_period_end", "available_from", "ttm_available_from", "fiscal_year_end"):
        out[col] = pd.to_datetime(out[col])
    out["pe_month"] = _mi(out["statement_period_end"]).astype("int64")
    return out


def _dec_str(value: float) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    return format(Decimal(repr(float(value))).quantize(Decimal("0.0001")), "f")


def _best_total_assets(ta: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Latest-period total_assets row per ticker with available_from <= t (ta sorted by pe_month)."""
    sub = ta[ta["available_from"] <= t]
    best = sub.drop_duplicates("ticker", keep="last").copy()
    tm = t.year * 12 + t.month - 1
    gap = tm - best["pe_month"]
    val = best["standalone_value"]
    best["status"] = np.select(
        [gap > A_T_MAX_STALE_MONTHS, ~(val > 0)], ["stale", "nonpositive"], default="ok"
    )
    return best


def build_anchor(rows: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Anchor each (decision_date, ticker) on its latest valid total_assets row (A8, A10).

    Only common-ticker rows (``only_preferred_listed`` false) are kept. pe0 is the
    latest statement_period_end among rows available on or before the decision date,
    per basis. The basis whose pe0 is latest is chosen (CFS when both share the same
    pe0). Stale (> 12 months) and nonpositive are judged on that basis only, with no
    fallback to the other one.
    """
    rows = rows if "pe_month" in rows.columns else _prep_rows(rows)
    uni = universe.copy()
    if "only_preferred_listed" in uni.columns:
        uni = uni[~uni["only_preferred_listed"].fillna(False).astype(bool)]
    uni = uni.reset_index(drop=True)
    uni["_t"] = pd.to_datetime(uni["decision_date"])

    ta_all = rows[rows["metric_code"] == METRICS["ta"]]
    per_basis: dict[str, pd.DataFrame] = {}
    for basis in BASES:
        ta = ta_all[ta_all["fs_basis"] == basis].sort_values("pe_month").copy()
        if "value_dec" not in ta.columns:
            ta["value_dec"] = ta["standalone_value"].map(_dec_str)
        ta = ta[
            [
                "ticker",
                "pe_month",
                "statement_period_end",
                "available_from",
                "standalone_value",
                "value_dec",
                "fiscal_year_end",
            ]
        ]
        frames = []
        for t in uni["_t"].unique():
            best = _best_total_assets(ta, pd.Timestamp(t))
            best["_t"] = pd.Timestamp(t)
            frames.append(best)
        cat = (
            pd.concat(frames, ignore_index=True) if frames else ta.iloc[:0].assign(status=[], _t=[])
        )
        per_basis[basis] = cat.add_suffix("_" + basis[0].lower()).rename(
            columns={f"_t_{basis[0].lower()}": "_t", f"ticker_{basis[0].lower()}": "ticker"}
        )

    out = uni.merge(per_basis["CFS"], on=["_t", "ticker"], how="left").merge(
        per_basis["OFS"], on=["_t", "ticker"], how="left"
    )
    # A8: the basis with the latest pe0 wins (CFS on a tie). Validity is judged on
    # that basis only; there is no fallback to the other basis.
    pe_c = pd.to_datetime(out["statement_period_end_c"])
    pe_o = pd.to_datetime(out["statement_period_end_o"])
    use_c = pe_c.notna() & (pe_o.isna() | (pe_c >= pe_o))
    use_o = ~use_c & pe_o.notna()

    def pick(name: str):
        c, o = out[f"{name}_c"], out[f"{name}_o"]
        return c.where(use_c, o.where(use_o))

    res = pd.DataFrame(index=out.index)
    res["decision_date"] = out["decision_date"]
    res["ticker"] = out["ticker"]
    res["corp_code"] = out["corp_code"]
    for col in _UNIVERSE_PASSTHROUGH:
        if col in out.columns:
            res[col] = out[col]
    res["basis"] = np.where(use_c, "CFS", np.where(use_o, "OFS", None))
    res["pe0"] = pd.to_datetime(pick("statement_period_end"))
    res["a_t"] = pick("standalone_value").astype(float)
    res["a_t_dec"] = pick("value_dec")
    res["a_t_available_from"] = pd.to_datetime(pick("available_from"))
    fye = pd.to_datetime(pick("fiscal_year_end")).dt.month
    res["fye_month"] = fye.astype("Int64")
    res["fye_on_grid"] = res["fye_month"].isin(list(QUARTER_MONTHS)).astype(bool)
    status = pick("status")
    res["a_t_status"] = status.where(status.notna(), "absent").astype(object)
    res["basis_switched"] = _basis_switched(res, out["_t"])
    return res.reset_index(drop=True)


def _basis_switched(res: pd.DataFrame, t: pd.Series) -> pd.Series:
    """True when basis differs from the same ticker's basis at the date 3 months earlier."""
    cur = pd.DataFrame(
        {
            "ticker": res["ticker"].to_numpy(),
            "mi": _mi(t).to_numpy(),
            "basis": res["basis"].to_numpy(),
        }
    )
    prev = cur.drop_duplicates(["ticker", "mi"]).copy()
    prev["mi"] = prev["mi"] + 3
    merged = cur[["ticker", "mi"]].merge(
        prev.rename(columns={"basis": "prev_basis"}), on=["ticker", "mi"], how="left"
    )
    prev_basis = merged["prev_basis"].to_numpy(dtype=object)
    basis = cur["basis"].to_numpy(dtype=object)
    switched = pd.notna(prev_basis) & pd.notna(basis) & (prev_basis != basis)
    return pd.Series(switched.astype(bool), index=res.index)


def _metric_sources(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    src: dict[str, pd.DataFrame] = {}
    keep = [
        "ticker",
        "fs_basis",
        "pe_month",
        "available_from",
        "standalone_value",
        "ttm_value",
        "ttm_complete",
        "ttm_available_from",
        "ttm_includes_conflict",
    ]
    for code, short in _SHORT_BY_CODE.items():
        sub = rows.loc[(rows["metric_code"] == code) & rows["fs_basis"].isin(BASES), keep]
        if sub.duplicated(["ticker", "fs_basis", "pe_month"]).any():
            raise ValueError(f"duplicate (ticker, fs_basis, period) rows for {code}")
        src[short] = sub
    return src


def _fetch(anchor: pd.DataFrame, key: pd.DataFrame, src: pd.DataFrame, offset: int) -> pd.DataFrame:
    """Standalone ('st') and TTM ('ttm') values at pe0 + offset months, PIT-gated at t.

    ``key`` has ticker, basis, pm0 (Int64) and t. Returns a frame aligned to ``anchor``
    with columns st, ttm, ttm_conflict.
    """
    ok = key["pm0"].notna() & key["basis"].notna()
    left = key.loc[ok, ["ticker", "basis", "pm0", "t"]].copy()
    left["pe_month"] = left["pm0"].astype("int64") + offset
    left = left.rename(columns={"basis": "fs_basis"}).drop(columns="pm0")
    m = left.reset_index().merge(src, on=["ticker", "fs_basis", "pe_month"], how="left")
    m = m.set_index("index")
    st = m["standalone_value"].where(m["available_from"] <= m["t"])
    ttm_ok = m["ttm_complete"].astype("boolean").fillna(False).astype(bool) & (
        m["ttm_available_from"] <= m["t"]
    )
    ttm = m["ttm_value"].where(ttm_ok)
    conflict = m["ttm_includes_conflict"].astype("boolean").where(ttm_ok)
    out = pd.DataFrame({"st": st, "ttm": ttm, "ttm_conflict": conflict})
    return out.reindex(anchor.index)


def build_features(rows: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Anchor columns + the 21 raw FEATURES + oi_ttm_raw, ttm_includes_conflict, input_eligible."""
    rows = rows if "pe_month" in rows.columns else _prep_rows(rows)
    src = _metric_sources(rows)
    df = anchor.reset_index(drop=True).copy()
    t = pd.to_datetime(df["decision_date"])
    key = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "basis": df["basis"],
            "pm0": _mi(pd.to_datetime(df["pe0"])).astype("Int64"),
            "t": t,
        }
    )
    a = df["a_t"].where(df["a_t"] > 0)

    def get(short: str, offset: int = 0) -> pd.DataFrame:
        return _fetch(df, key, src[short], offset)

    oi0, oi1, oi2, oi3 = (get("oi", o) for o in (0, -3, -6, -9))
    oi_prev = get("oi", -12)
    rev, ni, ocf = get("rev"), get("ni"), get("ocf")
    ta_prev = get("ta", -12)
    te, cash = get("te"), get("cash")
    dst, dlt = get("debt_st"), get("debt_lt")
    cpp, cin, div = get("capex_ppe"), get("capex_int"), get("div")

    f = pd.DataFrame(index=df.index)
    f["oi_ttm"] = oi0["ttm"] / a
    f["oi_q0"] = oi0["st"] / a
    f["oi_q1"] = oi1["st"] / a
    f["oi_q2"] = oi2["st"] / a
    f["oi_q3"] = oi3["st"] / a
    f["oi_chg_4q"] = (oi0["ttm"] - oi_prev["ttm"]) / a
    f["rev_ttm"] = rev["ttm"] / a
    f["margin_ttm"] = (oi0["ttm"] / rev["ttm"]).where(rev["ttm"] > 0)
    f["ocf_ttm"] = ocf["ttm"] / a
    f["accrual_ttm"] = (ni["ttm"] - ocf["ttm"]) / a
    f["ni_ttm"] = ni["ttm"] / a
    f["asset_growth"] = a / ta_prev["st"].where(ta_prev["st"] > 0) - 1.0
    f["equity_ratio"] = te["st"] / a
    f["cash_ratio"] = cash["st"] / a
    f["debt_ratio"] = (dst["st"] + dlt["st"]) / a
    f["capex_ttm"] = (cpp["ttm"].abs() + cin["ttm"].abs()) / a
    f["div_ttm"] = div["ttm"].abs() / a
    f["log_assets"] = np.log(a)
    raw = f["oi_ttm"].copy()
    f["loss_ttm"] = np.where(raw.isna(), np.nan, (raw < 0).astype(float))
    f["is_ofs"] = np.where(df["basis"].isna(), np.nan, (df["basis"] == "OFS").astype(float))
    f["stale_days"] = (t - pd.to_datetime(df["a_t_available_from"])).dt.days.astype(float)

    out = df.copy()
    for name in FEATURES:
        out[name] = f[name].astype(float)
    out["oi_ttm_raw"] = raw.astype(float)
    out["ttm_includes_conflict"] = oi0["ttm_conflict"].where(raw.notna())

    def flag(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return out[col].fillna(False).astype(bool)

    base = out["fye_on_grid"].astype(bool) & (out["a_t_status"] == "ok") & raw.notna()
    in_u = flag("in_universe")
    holding_only = flag("is_holding")
    if "exclusion_reason" in out.columns:
        holding_only = holding_only & (out["exclusion_reason"] == "financial")
    else:
        holding_only = holding_only & False
    out["input_eligible"] = (in_u & base).astype(bool)
    out["input_eligible_incl_holding"] = ((in_u | holding_only) & base).astype(bool)
    return out


def clip_features(
    df: pd.DataFrame,
    cols: tuple[str, ...] | list[str] | None = None,
    eligible_col: str = "input_eligible",
    q: tuple[float, float] = CLIP_Q,
) -> pd.DataFrame:
    """Clip ``cols`` per decision date to quantiles of the eligible rows.

    Bounds use ``numpy.nanquantile(..., method='linear')`` over eligible rows only
    and are applied to every row of that date. Binary features and ``oi_ttm_raw``
    are not clipped unless named in ``cols``.
    """
    if cols is None:
        cols = [c for c in FEATURES if c not in BINARY_FEATURES]
    out = df.copy()
    elig = out[eligible_col].fillna(False).astype(bool)
    for _, idx in out.groupby("decision_date", sort=False).groups.items():
        sel = out.index.isin(idx)
        for col in cols:
            vals = out.loc[sel & elig.to_numpy(), col].to_numpy(dtype=float)
            if vals.size == 0 or np.isnan(vals).all():
                continue
            lo, hi = np.nanquantile(vals, q, method="linear")
            out.loc[sel, col] = out.loc[sel, col].clip(lo, hi)
    return out
