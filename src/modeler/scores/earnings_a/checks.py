"""Earnings forecast A, F3: the label-free technical report.

This module counts the universe and the input rows per decision date. It never reads
the labels and never looks at feature values beyond whether they are missing. The
report is a pre-freeze check, so it carries no quantiles, means, sums, or signs of
feature values. The only value-derived output is the share of missing values per feature.

``checks.py`` must not import ``labels``, ``train``, ``evaluate``, or ``reasons``.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.spec import FEATURES

SIGN_METRICS: tuple[str, ...] = ("capex_ppe", "capex_intangible", "dividends_paid")

_WINDOWS: tuple[tuple[str, tuple[date, ...]], ...] = (
    ("DEV", seal.DEV_DATES),
    ("GAP", seal.GAP_DATES),
    ("SEALED", seal.SEALED_DATES),
    ("CONFIRM", seal.CONFIRM_DATES),
)

_COUNT_COLUMNS: tuple[str, ...] = (
    "n_companies",
    "n_kospi",
    "n_kosdaq",
    "n_halted",
    "n_excl_only_preferred",
    "n_excl_financial",
    "n_holding_64992",
    "n_excl_spac",
    "n_excl_liq_notice",
    "n_excl_non_krw",
    "n_spac_flag",
    "n_in_universe",
    "n_p6_tradable_in_universe",
    "n_a_t_ok",
    "n_a_t_stale",
    "n_a_t_nonpositive",
    "n_a_t_absent",
    "n_fye_offgrid",
    "n_fye_dec",
    "n_fye_other_grid",
    "n_basis_cfs",
    "n_basis_ofs",
    "n_basis_switched",
    "n_oi_ttm_present",
    "n_input_eligible",
    "n_input_eligible_incl_holding",
    "n_ttm_includes_conflict",
    "n_p6_tradable_eligible",
)

F3_COLUMNS: list[str] = (
    ["decision_date", "window", "session_date"]
    + list(_COUNT_COLUMNS)
    + [f"miss_{name}" for name in FEATURES]
)

SIGN_COLUMNS: list[str] = [
    "metric_code",
    "group_by",
    "group_value",
    "n_lt0",
    "n_eq0",
    "n_gt0",
    "n_nan",
]


def window_of(decision_date: date) -> str:
    """Return DEV, GAP, SEALED, CONFIRM, or OTHER for a decision date."""
    for name, dates in _WINDOWS:
        if decision_date in dates:
            return name
    return "OTHER"


def _iso(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _flag(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype(bool)


def _first_iso(s: pd.Series) -> str | None:
    for v in s:
        if not pd.isna(v):
            return _iso(v)
    return None


def _universe_counts(ud: pd.DataFrame) -> dict[str, int]:
    reason = ud["exclusion_reason"]
    uin = _flag(ud["in_universe"])
    return {
        "n_companies": int(len(ud)),
        "n_kospi": int(ud["market"].eq("KOSPI").sum()),
        "n_kosdaq": int(ud["market"].eq("KOSDAQ").sum()),
        "n_halted": int(_flag(ud["is_halted"]).sum()),
        "n_excl_only_preferred": int(reason.eq("only_preferred").sum()),
        "n_excl_financial": int(reason.eq("financial").sum()),
        "n_holding_64992": int(_flag(ud["is_holding"]).sum()),
        "n_excl_spac": int(reason.eq("spac").sum()),
        "n_excl_liq_notice": int(reason.eq("liq_notice").sum()),
        "n_excl_non_krw": int(reason.eq("non_krw").sum()),
        "n_spac_flag": int(_flag(ud["is_spac"]).sum()),
        "n_in_universe": int(uin.sum()),
        "n_p6_tradable_in_universe": int((uin & _flag(ud["p6_tradable"])).sum()),
    }


def _feature_counts(fd: pd.DataFrame) -> dict[str, int]:
    fi = fd[_flag(fd["in_universe"])]
    status = fi["a_t_status"]
    elig = _flag(fi["input_eligible"])
    fye_month = fi["fye_month"]
    return {
        "n_a_t_ok": int(status.eq("ok").sum()),
        "n_a_t_stale": int(status.eq("stale").sum()),
        "n_a_t_nonpositive": int(status.eq("nonpositive").sum()),
        "n_a_t_absent": int(status.eq("absent").sum()),
        "n_fye_offgrid": int((~_flag(fi["fye_on_grid"]) & fye_month.notna()).sum()),
        "n_fye_dec": int(fye_month.isin([12]).sum()),
        "n_fye_other_grid": int(fye_month.isin([3, 6, 9]).sum()),
        "n_basis_cfs": int(fi["basis"].eq("CFS").sum()),
        "n_basis_ofs": int(fi["basis"].eq("OFS").sum()),
        "n_basis_switched": int(_flag(fi["basis_switched"]).sum()),
        "n_oi_ttm_present": int(fi["oi_ttm_raw"].notna().sum()),
        "n_input_eligible": int(elig.sum()),
        "n_input_eligible_incl_holding": int(_flag(fd["input_eligible_incl_holding"]).sum()),
        "n_ttm_includes_conflict": int(_flag(fi.loc[elig, "ttm_includes_conflict"]).sum()),
        "n_p6_tradable_eligible": int((elig & _flag(fi["p6_tradable"])).sum()),
    }


def _miss_shares(fd: pd.DataFrame) -> dict[str, float]:
    el = fd[_flag(fd["input_eligible"])]
    out: dict[str, float] = {}
    for name in FEATURES:
        if len(el) == 0:
            out[f"miss_{name}"] = float("nan")
        else:
            out[f"miss_{name}"] = round(float(el[name].isna().mean()), 4)
    return out


def f3_report(universe: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per universe decision date (sorted). Counts and missing shares only.

    ``universe`` carries the columns of ``universe.COLUMNS``. ``features`` is the output of
    ``features.build_features``. Feature values are never summarized, only their NaN share.
    """
    u = universe.assign(_d=universe["decision_date"].map(_iso))
    f = features.assign(_d=features["decision_date"].map(_iso))
    dates = sorted(d for d in u["_d"].dropna().unique())

    rows: list[dict] = []
    for d in dates:
        ud = u[u["_d"] == d]
        fd = f[f["_d"] == d]
        row: dict = {
            "decision_date": d,
            "window": window_of(date.fromisoformat(d)),
            "session_date": _first_iso(ud["session_date"]),
        }
        row.update(_universe_counts(ud))
        row.update(_feature_counts(fd))
        row.update(_miss_shares(fd))
        rows.append(row)

    return pd.DataFrame(rows, columns=F3_COLUMNS)


def sign_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Sign counts of the TTM value for capex and dividends, on complete TTM rows only.

    ``rows`` is the output of ``features.load_input_rows``. Groups by ``bsns_year`` when
    the column is present, else by the year of ``statement_period_end``. Counts only.
    """
    r = rows[rows["metric_code"].isin(SIGN_METRICS)]
    r = r[_flag(r["ttm_complete"])]
    if "bsns_year" in r.columns:
        group_by = "bsns_year"
        key = r["bsns_year"]
    else:
        group_by = "period_year"
        key = pd.to_datetime(r["statement_period_end"]).dt.year
    work = pd.DataFrame(
        {
            "metric_code": r["metric_code"].to_numpy(),
            "group_value": key.map(lambda v: str(v)).to_numpy(),
            "ttm_value": pd.to_numeric(r["ttm_value"], errors="coerce").to_numpy(dtype=float),
        }
    )
    out: list[dict] = []
    for (metric, group), sub in work.groupby(["metric_code", "group_value"], sort=True):
        v = sub["ttm_value"]
        out.append(
            {
                "metric_code": metric,
                "group_by": group_by,
                "group_value": group,
                "n_lt0": int((v < 0).sum()),
                "n_eq0": int((v == 0).sum()),
                "n_gt0": int((v > 0).sum()),
                "n_nan": int(v.isna().sum()),
            }
        )
    return pd.DataFrame(out, columns=SIGN_COLUMNS)


def _repo_root_of(path: Path) -> Path | None:
    for p in (path, *path.parents):
        if (p / ".git").exists():
            return p
    return None


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_f3(report: pd.DataFrame, signs: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Write the report and the sign table into a new directory (write-once).

    ``out_dir`` must not exist yet and must not sit under a git worktree or repository
    (any ancestor with a ``.git`` entry). Files: ``f3_report.tsv``, ``f3_report.json``
    (list of records), ``f3_signs.tsv``.
    """
    target = Path(out_dir).expanduser().resolve()
    repo = _repo_root_of(target)
    if repo is not None:
        raise ValueError(f"out_dir {target} is inside the repository {repo}; write outside it")
    target.mkdir(parents=True, exist_ok=False)

    paths = {
        "report_tsv": target / "f3_report.tsv",
        "report_json": target / "f3_report.json",
        "signs_tsv": target / "f3_signs.tsv",
    }
    report.to_csv(paths["report_tsv"], sep="\t", index=False, na_rep="")
    records = [{k: _py(v) for k, v in rec.items()} for rec in report.to_dict(orient="records")]
    paths["report_json"].write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    signs.to_csv(paths["signs_tsv"], sep="\t", index=False, na_rep="")
    return paths
