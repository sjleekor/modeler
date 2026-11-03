"""One-shot SEALED verdict runner for research A stage 1 (preregistration 6.3, 7.1-7.3), with a
DEV dry-run mode that walks the identical code path on development data.

Only composes frozen functions (``walk_forward``, ``h1_h2``, label builders, ``digest``); no
rule lives here. Everything runs in ONE process so the seal is opened exactly once:

1. refuse a dirty tree (real mode), verify input hashes, verify ``chosen.json`` sha256,
2. label-free preparation, then the pre-open integrity check (F4 label digest of the window
   recomputed hash-only and compared with the frozen manifest) - abort before opening on a
   mismatch,
3. ``seal.open_seal_grant(SEALED_RUN)`` (real) or ``seal.grant_dev()`` (dry run),
4. primary verdict (H1/H2) written FIRST as ``verdict.json``,
5. record-only items (7.3), each wrapped so that an error is recorded and the run continues.

Output is aggregated numbers only. Row-level labels and predictions never leave memory.
stdout (in the cli) carries paths only.

Readings of the rules that the preregistration leaves open (also listed in the manifest):

* E-3 holding companies: the universe widens to ``input_eligible_incl_holding`` for training and
  evaluation (all three predictors retrained); clip bounds stay those of the frozen pipeline
  (computed over ``input_eligible`` rows).
* E-10 P6 tradable: evaluation rows restricted to ``p6_tradable``; models are the primary ones
  (trained on the full universe), no retraining.
* KSIC 대분류: the dataset only has the 2-digit division (``ksic2``); it is mapped to the KSIC
  rev.10 section letter.
* OCF target: PERS is reported with ``oi_ttm`` (the frozen PERS) and with ``ocf_ttm``.
* 2-year horizon: training/label dates are DEV <= 2021-06-30 and SEALED <= 2024-06-30
  (``digest.Y2_DATE_MAX``); no GAP date.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeler.scores.earnings_a import digest, evaluate, labels, manifest, reasons, seal, train
from modeler.scores.earnings_a import paths as P
from modeler.scores.earnings_a.paths import DataPaths
from modeler.scores.earnings_a.seal import Phase
from modeler.scores.earnings_a.spec import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    HGB_GRID,
    RAW_MAX_DATE,
    RIDGE_ALPHAS,
    SNAPSHOT,
)
from modeler.scores.earnings_a.train import ModelSpec

CHOSEN_SHA256 = "e2369a92cbde3ec42656cfb14d585e3f252eb368782caddbc230b2ea47314412"
CHOSEN_REL = "dev_20261103T031237Z/chosen.json"
F4_REL = "f4_20261103T024518Z/manifest_f4.json"
PREDICTORS = evaluate.PREDICTORS  # ("PERS", "RIDGE", "HGB")
KEYS = ["decision_date", "ticker", "corp_code"]
JOIN_KEYS = ["decision_date", "ticker", "basis"]
N_EVAL = 9


class SealedAbort(RuntimeError):
    """A pre-open check failed. The seal has not been opened."""


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    dry_run: bool
    kind: str
    cmd: str
    decision_dates: tuple[date, ...]
    eval_dates: tuple[date, ...]
    y2_dates: tuple[date, ...]
    y2_eval: tuple[date, ...]
    f4_key: str
    f4_dates: tuple[date, ...]


def y2_dates(decision_dates: Sequence[date]) -> tuple[date, ...]:
    """Dates with a mature 2-year label (``digest.Y2_DATE_MAX``); GAP has none."""
    out = []
    for d in decision_dates:
        for window, wdates in digest.WINDOWS.items():
            cap = digest.Y2_DATE_MAX[window]
            if d in wdates and cap is not None and d <= cap:
                out.append(d)
    return tuple(out)


def make_mode(dry_run: bool) -> Mode:
    if dry_run:
        dd = tuple(seal.DEV_DATES)
        ev = dd[-N_EVAL:]
        y2 = y2_dates(dd)
        return Mode(
            True,
            "SEALED_DRYRUN_DEV",
            "sealed_dryrun_dev",
            dd,
            ev,
            y2,
            tuple(d for d in ev if d in set(y2)),
            "DEV.oi_y1",
            tuple(seal.DEV_DATES),
        )
    dd = tuple(seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES)
    ev = tuple(seal.SEALED_DATES)
    y2 = y2_dates(dd)
    return Mode(
        False,
        "SEALED_VERDICT",
        "sealed",
        dd,
        ev,
        y2,
        tuple(d for d in ev if d in set(y2)),
        "SEALED.oi_y1",
        tuple(seal.SEALED_DATES),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _py(o: Any) -> Any:
    """JSON-safe copy: numpy scalars to python, non-finite floats to None, dates to ISO."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_py(v) for v in o]
    if isinstance(o, np.bool_ | bool):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, date | pd.Timestamp):
        return seal._to_date(o).isoformat()
    return o


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _iso(d: Any) -> str:
    return seal._to_date(d).isoformat()


_SECTIONS: tuple[tuple[str, int, int], ...] = (
    ("A", 1, 3),
    ("B", 5, 9),
    ("C", 10, 34),
    ("D", 35, 35),
    ("E", 36, 39),
    ("F", 41, 42),
    ("G", 45, 47),
    ("H", 49, 52),
    ("I", 55, 56),
    ("J", 58, 63),
    ("K", 64, 66),
    ("L", 68, 68),
    ("M", 70, 73),
    ("N", 74, 76),
    ("O", 84, 84),
    ("P", 85, 85),
    ("Q", 86, 87),
    ("R", 90, 91),
    ("S", 94, 96),
    ("T", 97, 98),
    ("U", 99, 99),
)


def ksic_section(ksic2: Any) -> str:
    """KSIC rev.10 대분류 letter of a 2-digit division; ``NA`` when empty or unmapped."""
    try:
        n = int(str(ksic2)[:2])
    except (TypeError, ValueError):
        return "NA"
    for letter, lo, hi in _SECTIONS:
        if lo <= n <= hi:
            return letter
    return "NA"


# ---------------------------------------------------------------------------
# Seams (module-level so tests can replace them)
# ---------------------------------------------------------------------------


def _git_state() -> dict[str, Any]:
    return manifest.git_state(manifest.repo_root())


def _verify_inputs(paths: DataPaths) -> None:
    P.verify_input_hashes(paths)


def _label_free(paths: DataPaths) -> dict[str, pd.DataFrame]:
    """All label-free inputs: clipped features, receipts, stock master, last trade."""
    con = P.connect_inputs(paths)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, feats = digest.prepare_inputs(paths, con)
            return {
                "feats": feats,
                "receipts": digest.load_receipts(con),
                "stock_master": digest.load_stock_master(con),
                "last_trade": digest.load_last_trade(con),
            }
    finally:
        con.close()


def _load_chosen(path: Path, expected_sha: str) -> dict[str, Any]:
    """Verify the sha256 of ``chosen.json`` and that its specs are members of the frozen grid."""
    if not path.is_file():
        raise SealedAbort(f"chosen.json not found: {path.name}")
    if _sha(path) != expected_sha:
        raise SealedAbort("chosen.json sha256 differs from the committed value")
    obj = json.loads(path.read_text(encoding="utf-8"))
    hgb = ModelSpec("HGB", dict(obj["HGB"]["params"]))
    ridge = ModelSpec("RIDGE", dict(obj["RIDGE"]["params"]))
    if hgb.spec_id != obj["HGB"]["spec_id"] or ridge.spec_id != obj["RIDGE"]["spec_id"]:
        raise SealedAbort("chosen.json spec_id does not match its params")
    if hgb.params not in [dict(g) for g in HGB_GRID]:
        raise SealedAbort("chosen HGB params are not in the frozen grid")
    if float(ridge.params["alpha"]) not in [float(a) for a in RIDGE_ALPHAS]:
        raise SealedAbort("chosen RIDGE alpha is not in the frozen grid")
    return {"HGB": hgb, "RIDGE": ridge, "sha256": expected_sha}


def _f4_check(
    paths: DataPaths, lf: dict[str, pd.DataFrame], mode: Mode, f4_manifest_path: Path
) -> dict[str, Any]:
    """Recompute the hash-only F4 digest of the window's ``oi_y1`` labels; abort on a mismatch.

    Uses ``seal.digest_only`` (a hash grant); the ledger is not touched.
    """
    m = json.loads(f4_manifest_path.read_text(encoding="utf-8"))
    exp = m["digests"][mode.f4_key]
    got = digest._label_digest(paths, lf["feats"], lf["receipts"], mode.f4_dates, "oi_y1")
    ok = (
        got["exact_sha256"] == exp["exact_sha256"]
        and got.get("float_sha256") == exp.get("float_sha256")
        and got["n_rows"] == exp["n_rows"]
    )
    if not ok:
        raise SealedAbort(f"F4 digest {mode.f4_key} differs from the frozen manifest")
    cur = manifest.package_file_sha256()
    old = m.get("code", {}).get("package_file_sha256", {})
    changed = sorted(k for k in cur if old.get(k) != cur[k])
    return {
        "key": mode.f4_key,
        "match": True,
        "exact_sha256": got["exact_sha256"],
        "float_sha256": got.get("float_sha256"),
        "n_rows": got["n_rows"],
        "f4_manifest": f4_manifest_path.name,
        "package_files_changed_since_f4": changed,
    }


def _load_labels(
    paths: DataPaths,
    feats: pd.DataFrame,
    receipts: pd.DataFrame,
    dates: tuple[date, ...],
    target: str,
    grant: seal.SealGrant,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Labels of ``target`` for ``dates`` plus the raw r2 rows (for the reason table)."""
    metric, horizon = labels.TARGETS[target]
    anchor = digest._anchor_rows(feats, dates)[digest.ANCHOR_COLS].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = labels.load_label_rows(
            paths, decision_dates=dates, grant=grant, horizon=horizon, metrics=(metric,)
        )
        lab = labels.build_labels(
            r2, anchor, receipts, decision_dates=dates, grant=grant, target=target
        )
    return lab, r2


def _reason_table(ctx: Ctx) -> pd.DataFrame:
    metric, horizon = labels.TARGETS["oi_y1"]
    dates = ctx.mode.decision_dates
    sub = digest._anchor_rows(ctx.feats, dates)
    anchor = sub[digest.ANCHOR_COLS].reset_index(drop=True)
    fye = sub[["decision_date", "ticker", "fye_month"]].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lq = labels.build_label_quarters(
            ctx.r2,
            anchor,
            ctx.receipts,
            decision_dates=dates,
            grant=ctx.grant,
            metric=metric,
            horizon=horizon,
        )
        presence = labels.presence_view(lq).merge(fye, on=["decision_date", "ticker"], how="left")
        df = reasons.build_reason_dictionary(
            presence,
            ctx.receipts,
            ctx.stock_master,
            ctx.last_trade,
            run_at=ctx.run_at,
            source_snapshot=SNAPSHOT,
            raw_max_date=date.fromisoformat(RAW_MAX_DATE),
            metric=metric,
        )
    return df


# ---------------------------------------------------------------------------
# Dataset, joins, row checks
# ---------------------------------------------------------------------------


def build_dataset(feats: pd.DataFrame, lab: pd.DataFrame, dates: Sequence[date]) -> pd.DataFrame:
    """Features of ``dates`` (anchor rows) with labels joined: the frame ``walk_forward`` takes."""
    allowed = set(dates)
    sub = digest._anchor_rows(feats, tuple(dates))
    if any(seal._to_date(d) not in allowed for d in sub["decision_date"]):
        raise seal.SealViolation("feature rows outside the requested dates")
    cols = [*JOIN_KEYS, "label_ok", "y", "y_clip", "label_pub_date", "missing_kind"]
    ds = sub.merge(lab[cols], on=JOIN_KEYS, how="left", validate="one_to_one")
    ds["label_ok"] = ds["label_ok"].eq(True)
    return ds.reset_index(drop=True)


_J_COLS = [
    "label_ok",
    "y",
    "y_clip",
    "ksic2",
    "a_t",
    "p6_tradable",
    "oi_ttm",
    "is_holding",
    "input_eligible",
]


def join_labels(pred: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _J_COLS if c in ds.columns]
    j = pred.merge(ds[[*KEYS, *cols]], on=KEYS, how="left", validate="m:1")
    j["label_ok"] = j["label_ok"].fillna(False).astype(bool)
    return j


def _scored(j: pd.DataFrame, y_col: str = "y_clip") -> pd.DataFrame:
    return j.loc[j["label_ok"] & j[y_col].notna() & j["y_hat"].notna()]


def assert_same_rows(j: pd.DataFrame, models: Sequence[str] = PREDICTORS) -> None:
    """PERS, RIDGE and HGB predict the same eval rows and are scored on the same rows."""
    for name, frame in (("eval", j), ("scored", _scored(j))):
        sets: dict[Any, dict[str, frozenset]] = {}
        for (d, m), g in frame.groupby(["decision_date", "model"], sort=False):
            if m in models:
                sets.setdefault(d, {})[m] = frozenset(zip(g["ticker"], g["corp_code"], strict=True))
        for d, by_model in sets.items():
            if set(by_model) != set(models) or len({*by_model.values()}) != 1:
                raise AssertionError(f"{name} rows differ across predictors at {_iso(d)}")
        if not sets:
            raise AssertionError(f"no {name} rows")
    scored = _scored(j)
    for m in ("RIDGE", "HGB"):
        if scored.loc[scored["model"] == m, "y_hat"].isna().any():
            raise AssertionError(f"{m}: NaN prediction on a scored row")


# ---------------------------------------------------------------------------
# Judging (record items use the same arithmetic as evaluate.h1_h2)
# ---------------------------------------------------------------------------


def judge(pred: pd.DataFrame, lab: pd.DataFrame, y_col: str = "y_clip") -> dict[str, Any]:
    """H1 and H2 numbers (both always computed) with the preregistered conditions.

    Uses ``evaluate._prepare`` / ``_compare`` - the functions ``h1_h2`` itself uses - so the
    numbers equal ``h1_h2`` for the same inputs. Win thresholds apply to 9 dates only.
    """
    df = evaluate._prepare(pred, lab, y_col)
    n = int(df["_d"].nunique())
    h1, _, _ = evaluate._compare(df, "PERS", y_col, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    h2, _, _ = evaluate._compare(df, "RIDGE", y_col, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    nine = n == evaluate.N_DATES
    h1["conditions"] = {
        "ratio_le_0.95": bool(h1["mae_ratio"] <= evaluate.H1_MAX_MAE_RATIO),
        "wins_ge_7_of_9": bool(h1["wins"] >= evaluate.H1_MIN_WINS) if nine else None,
        "lower_gt_0": bool(h1["lower"] > 0),
    }
    h2["conditions"] = {
        "wins_ge_6_of_9": bool(h2["wins"] >= evaluate.H2_MIN_WINS) if nine else None,
        "lower_gt_0": bool(h2["lower"] > 0),
    }
    return {
        "n_rows": int(len(df)),
        "n_firms": int(df["corp_code"].nunique()),
        "n_dates": n,
        "y_col": y_col,
        "h1": h1,
        "h2": h2,
    }


def _verdict_doc(ver: dict[str, Any], mode: Mode, chosen: dict[str, Any]) -> dict[str, Any]:
    h1, h2 = ver["h1"], ver["h2"]
    h1["conditions"] = {
        "ratio_le_0.95": bool(h1["mae_ratio"] <= evaluate.H1_MAX_MAE_RATIO),
        "wins_ge_7_of_9": bool(h1["wins"] >= evaluate.H1_MIN_WINS),
        "lower_gt_0": bool(h1["lower"] > 0),
    }
    if h2 is not None:
        h2["conditions"] = {
            "wins_ge_6_of_9": bool(h2["wins"] >= evaluate.H2_MIN_WINS),
            "lower_gt_0": bool(h2["lower"] > 0),
        }
    outcome = (
        "H1_FAIL"
        if not h1["pass"]
        else ("H1_PASS_H2_PASS" if h2 and h2["pass"] else "H1_PASS_H2_FAIL")
    )
    return {
        **ver,
        "kind": mode.kind,
        "outcome": outcome,
        "not_a_verdict": mode.dry_run,
        "eval_dates": [_iso(d) for d in mode.eval_dates],
        "chosen": {
            "HGB": {"spec_id": chosen["HGB"].spec_id},
            "RIDGE": {"spec_id": chosen["RIDGE"].spec_id},
            "sha256": chosen["sha256"],
        },
    }


# ---------------------------------------------------------------------------
# Run context and record items
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    paths: DataPaths
    mode: Mode
    grant: seal.SealGrant
    chosen: dict[str, Any]
    out_dir: Path
    run_at: datetime
    feats: pd.DataFrame
    receipts: pd.DataFrame
    stock_master: pd.DataFrame
    last_trade: pd.DataFrame
    lab: pd.DataFrame | None = None
    r2: pd.DataFrame | None = None
    ds: pd.DataFrame | None = None
    pred: pd.DataFrame | None = None
    pred_ind: pd.DataFrame | None = None
    files: dict[str, Path] = field(default_factory=dict)

    # -- output (write-once) --
    def write_json(self, name: str, obj: Any) -> Path:
        p = manifest.write_once(self.out_dir / name, _py(obj))
        self.files[name] = p
        return p

    def write_tsv(self, name: str, df: pd.DataFrame) -> Path:
        p = self.out_dir / name
        with open(p, "x", encoding="utf-8") as f:
            df.to_csv(f, sep="\t", index=False, float_format="%.12g")
        self.files[name] = p
        return p

    def walk(
        self,
        ds: pd.DataFrame,
        specs: Sequence[ModelSpec],
        eval_dates: Sequence[date],
        decision_dates: Sequence[date],
    ) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return train.walk_forward(
                ds,
                eval_dates=list(eval_dates),
                specs=list(specs),
                decision_dates=list(decision_dates),
                grant=self.grant,
            )

    def primary_specs(self) -> list[ModelSpec]:
        return [ModelSpec("PERS"), self.chosen["RIDGE"], self.chosen["HGB"]]


def _models_pred(ctx: Ctx) -> pd.DataFrame:
    assert ctx.pred is not None
    if ctx.pred_ind is not None:
        return pd.concat([ctx.pred, ctx.pred_ind], ignore_index=True)
    return ctx.pred


def _per_model_mae(sc: pd.DataFrame, y_col: str) -> pd.DataFrame:
    e = sc.assign(_ae=(sc[y_col] - sc["y_hat"]).abs())
    g = e.groupby(["decision_date", "model"]).agg(n=("_ae", "size"), mae=("_ae", "mean"))
    return g.reset_index()


def _with_ratios(per: pd.DataFrame) -> pd.DataFrame:
    p = per.copy()
    for base in ("PERS", "RIDGE"):
        ref = p.loc[p["model"] == base].set_index("decision_date")["mae"]
        p[f"ratio_vs_{base.lower()}"] = p["mae"] / p["decision_date"].map(ref)
    return p


def rec_per_date(ctx: Ctx) -> dict[str, Any]:
    """Per-date MAE and ratios per model (7.3), pooled rows (``ALL``) last."""
    j = join_labels(_models_pred(ctx), ctx.ds)
    sc = _scored(j).assign(_ae=lambda x: (x["y_clip"] - x["y_hat"]).abs())
    per = sc.groupby(["decision_date", "model"]).agg(n=("_ae", "size"), mae=("_ae", "mean"))
    per = per.reset_index()
    per["decision_date"] = [_iso(d) for d in per["decision_date"]]
    pooled = sc.groupby("model").agg(n=("_ae", "size"), mae=("_ae", "mean")).reset_index()
    pooled["decision_date"] = "ALL"
    out = pd.concat([per, pooled[per.columns]], ignore_index=True)
    for base in ("PERS", "RIDGE"):
        ref = out.loc[out["model"] == base].set_index("decision_date")["mae"]
        out[f"ratio_vs_{base.lower()}"] = out["mae"] / out["decision_date"].map(ref)
    ctx.write_tsv("record_per_date.tsv", out)
    return {"rows": len(out)}


def rec_by_group(ctx: Ctx) -> dict[str, Any]:
    """MAE by KSIC 대분류 (mapped from ksic2) and by total-assets quintile, per date (7.3)."""
    j = join_labels(_models_pred(ctx), ctx.ds)
    sc = _scored(j).copy()
    sc["_ae"] = (sc["y_clip"] - sc["y_hat"]).abs()
    sc["ksic_section"] = sc["ksic2"].map(ksic_section)
    base = sc.loc[sc["model"] == "PERS", [*KEYS, "a_t"]].copy()
    base["assets_quintile"] = (
        base.groupby("decision_date")["a_t"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        .astype(float)
        + 1
    )
    sc = sc.merge(base[[*KEYS, "assets_quintile"]], on=KEYS, how="left")
    parts = []
    for dim in ("ksic_section", "assets_quintile"):
        sub = sc.dropna(subset=[dim])
        for by, label in ((["decision_date", dim, "model"], None), ([dim, "model"], "ALL")):
            g = sub.groupby(by).agg(n=("_ae", "size"), mae=("_ae", "mean")).reset_index()
            g = g.rename(columns={dim: "group"})
            g["dimension"] = dim
            if label:
                g["decision_date"] = label
            else:
                g["decision_date"] = [_iso(d) for d in g["decision_date"]]
            g["group"] = g["group"].map(lambda v: str(int(v)) if isinstance(v, float) else str(v))
            parts.append(g[["dimension", "group", "decision_date", "model", "n", "mae"]])
    ctx.write_tsv("record_by_group.tsv", pd.concat(parts, ignore_index=True))
    return {"ksic_level": "section letter mapped from ksic2 (KSIC rev.10)"}


def rec_top1(ctx: Ctx) -> dict[str, Any]:
    """Count of |error| in the top 1% (7.3): own pooled p99 per model and PERS's p99 for all."""
    j = join_labels(_models_pred(ctx), ctx.ds)
    sc = _scored(j).copy()
    sc["_ae"] = (sc["y_clip"] - sc["y_hat"]).abs()
    thr_own = sc.groupby("model")["_ae"].quantile(0.99)
    thr_pers = float(sc.loc[sc["model"] == "PERS", "_ae"].quantile(0.99))
    sc["own"] = sc["_ae"] > sc["model"].map(thr_own)
    sc["pers_thr"] = sc["_ae"] > thr_pers
    g = (
        sc.groupby(["decision_date", "model"])
        .agg(n=("_ae", "size"), n_top1_own=("own", "sum"), n_over_pers_p99=("pers_thr", "sum"))
        .reset_index()
    )
    g["decision_date"] = [_iso(d) for d in g["decision_date"]]
    tot = (
        sc.groupby("model")
        .agg(n=("_ae", "size"), n_top1_own=("own", "sum"), n_over_pers_p99=("pers_thr", "sum"))
        .reset_index()
    )
    tot["decision_date"] = "ALL"
    ctx.write_tsv("record_top1.tsv", pd.concat([g, tot[g.columns]], ignore_index=True))
    return {"pers_p99_abs_error": thr_pers, "own_p99": {k: float(v) for k, v in thr_own.items()}}


def rec_spearman(ctx: Ctx) -> dict[str, Any]:
    """Per-date Spearman of (y_hat - oi_ttm, y - oi_ttm), clipped y and oi_ttm (7.3)."""
    j = join_labels(_models_pred(ctx), ctx.ds)
    sc = _scored(j)
    rows = []
    for (d, m), g in sc.groupby(["decision_date", "model"]):
        if m == "PERS":
            continue
        a = g["y_hat"] - g["oi_ttm"]
        b = g["y_clip"] - g["oi_ttm"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = a.corr(b, method="spearman")
        rows.append(
            {
                "decision_date": _iso(d),
                "model": m,
                "n": len(g),
                "spearman": rho,
            }
        )
    out = pd.DataFrame(rows)
    ctx.write_tsv("record_spearman.tsv", out)
    return {"rows": len(out)}


def rec_ind(ctx: Ctx) -> dict[str, Any]:
    """IND industry-shrinkage baseline, same eval rows (7.3). Feeds the other record tables."""
    pred = ctx.walk(ctx.ds, [ModelSpec("IND")], ctx.mode.eval_dates, ctx.mode.decision_dates)
    ctx.pred_ind = pred
    j = join_labels(pred, ctx.ds)
    per = _per_model_mae(_scored(j), "y_clip")
    per["decision_date"] = [_iso(d) for d in per["decision_date"]]
    ctx.write_tsv("record_ind.tsv", per)
    return {"rows": len(per)}


def rec_unclipped(ctx: Ctx) -> dict[str, Any]:
    """MAE and H1/H2 numbers with the unclipped y (E-1). Training stays on the clipped y."""
    j = join_labels(ctx.pred, ctx.ds)
    sc = _scored(j, "y")
    per = _per_model_mae(sc, "y")
    per["decision_date"] = [_iso(d) for d in per["decision_date"]]
    ctx.write_tsv("record_unclipped_mae.tsv", per)
    res = judge(ctx.pred, ctx.ds, "y")
    res["pooled_mae"] = {
        m: float((g["y"] - g["y_hat"]).abs().mean()) for m, g in sc.groupby("model")
    }
    ctx.write_json("record_unclipped.json", res)
    return {}


def rec_p6(ctx: Ctx) -> dict[str, Any]:
    """H1 three conditions on the P6-tradable subset of the evaluation rows (E-10)."""
    ds = ctx.ds
    keep = ds.loc[ds["p6_tradable"].fillna(False).astype(bool), KEYS + ["label_ok", "y_clip"]]
    pred = ctx.pred.merge(keep[KEYS], on=KEYS, how="inner")
    res = judge(pred, keep, "y_clip")
    res["note"] = "evaluation rows restricted to p6_tradable; models trained on the full universe"
    ctx.write_json("record_p6_tradable.json", res)
    return {}


def rec_holding(ctx: Ctx) -> dict[str, Any]:
    """H1 three conditions with holding companies (KSIC 64992) in the universe (E-3).

    Reading: the universe widens to ``input_eligible_incl_holding`` for training AND evaluation,
    all three predictors retrained. Clip bounds stay those of the frozen pipeline.
    """
    ds = ctx.ds.copy()
    extra = int((ds["input_eligible_incl_holding"].fillna(False) & ~ds["input_eligible"]).sum())
    ds["input_eligible"] = ds["input_eligible_incl_holding"].fillna(False).astype(bool)
    pred = ctx.walk(ds, ctx.primary_specs(), ctx.mode.eval_dates, ctx.mode.decision_dates)
    res = judge(pred, ds, "y_clip")
    res["extra_rows_all_dates"] = extra
    res["note"] = "universe widened for training and evaluation; predictors retrained"
    ctx.write_json("record_holding.json", res)
    return {}


def _scored_ocf_pers(
    pred: pd.DataFrame, ds: pd.DataFrame, eval_dates: Sequence[date]
) -> pd.DataFrame:
    """PERS predictions from ``ocf_ttm`` (persistence of the OCF target) on the eval rows."""
    keep = {seal._to_date(d) for d in eval_dates}
    ev = ds.loc[
        ds["input_eligible"].fillna(False).astype(bool)
        & ds["decision_date"].map(seal._to_date).isin(keep)
    ]
    pers = pd.DataFrame(
        {
            "decision_date": ev["decision_date"].to_numpy(),
            "ticker": ev["ticker"].to_numpy(),
            "corp_code": ev["corp_code"].to_numpy(),
            "model": "PERS",
            "spec_id": "PERS|ocf_ttm",
            "y_hat": ev["ocf_ttm"].to_numpy(dtype=float),
        }
    )
    return pd.concat([pred.loc[pred["model"] != "PERS"], pers], ignore_index=True)


def rec_ocf(ctx: Ctx) -> dict[str, Any]:
    """Operating-cash-flow target, same specs and procedure (7.3)."""
    dates = ctx.mode.decision_dates
    lab, _ = _load_labels(ctx.paths, ctx.feats, ctx.receipts, dates, "ocf_y1", ctx.grant)
    ds = build_dataset(ctx.feats, lab, dates)
    pred = ctx.walk(ds, ctx.primary_specs(), ctx.mode.eval_dates, dates)
    j = join_labels(pred, ds)
    assert_same_rows(j)
    res = {
        "pers_oi_ttm": judge(pred, ds, "y_clip"),
        "pers_ocf_ttm": judge(_scored_ocf_pers(pred, ds, ctx.mode.eval_dates), ds, "y_clip"),
        "label_ok_rate": _label_rate(ds, ctx.mode.eval_dates),
        "note": "PERS reported with oi_ttm (frozen definition) and with ocf_ttm",
    }
    per = _per_model_mae(_scored(j), "y_clip")
    per["decision_date"] = [_iso(d) for d in per["decision_date"]]
    ctx.write_tsv("record_ocf_per_date.tsv", per)
    ctx.write_json("record_ocf.json", res)
    return {}


def _label_rate(ds: pd.DataFrame, dates: Sequence[date]) -> dict[str, Any]:
    keep = {seal._to_date(d) for d in dates}
    e = ds.loc[
        ds["input_eligible"].fillna(False).astype(bool)
        & ds["decision_date"].map(seal._to_date).isin(keep)
    ]
    return {"n_eligible": int(len(e)), "n_label_ok": int(e["label_ok"].sum())}


def rec_y2(ctx: Ctx) -> dict[str, Any]:
    """Two-year horizon (7.3): mature dates only, specs retrained on the 2-year labels."""
    m = ctx.mode
    if not m.y2_eval:
        return {"skipped": "no evaluation date has a mature 2-year label"}
    lab, _ = _load_labels(ctx.paths, ctx.feats, ctx.receipts, m.y2_dates, "oi_y2", ctx.grant)
    ds = build_dataset(ctx.feats, lab, m.y2_dates)
    pred = ctx.walk(ds, ctx.primary_specs(), m.y2_eval, m.y2_dates)
    j = join_labels(pred, ds)
    assert_same_rows(j)
    res = judge(pred, ds, "y_clip")
    res["eval_dates"] = [_iso(d) for d in m.y2_eval]
    res["train_label_dates"] = [_iso(d) for d in m.y2_dates]
    res["note"] = "win thresholds are defined for 9 dates and not applied"
    per = _per_model_mae(_scored(j), "y_clip")
    per["decision_date"] = [_iso(d) for d in per["decision_date"]]
    ctx.write_tsv("record_y2_per_date.tsv", per)
    ctx.write_json("record_y2.json", res)
    return {}


def rec_completeness(ctx: Ctx) -> dict[str, Any]:
    """Label completeness per date and missing kind of input-eligible rows (5.4, 7.3)."""
    ds = ctx.ds
    el = ds.loc[ds["input_eligible"].fillna(False).astype(bool)]
    g = el.groupby("decision_date").agg(
        n_eligible=("label_ok", "size"), n_label_ok=("label_ok", "sum")
    )
    g["completeness"] = g["n_label_ok"] / g["n_eligible"]
    g["n_incl_holding"] = ds.groupby("decision_date").size()
    miss = (
        el.loc[~el["label_ok"]]
        .assign(missing_kind=lambda x: x["missing_kind"].fillna("NA"))
        .groupby(["decision_date", "missing_kind"])
        .size()
        .unstack(fill_value=0)
    )
    miss.columns = [f"missing:{c}" for c in miss.columns]
    out = g.join(miss).fillna(0).reset_index()
    out["decision_date"] = [_iso(d) for d in out["decision_date"]]
    ctx.write_tsv("record_completeness.tsv", out)
    return {}


def rec_reasons(ctx: Ctx) -> dict[str, Any]:
    """Reason-code counts per decision date (5.4); counts only."""
    df = _reason_table(ctx)
    if df.empty:
        out = pd.DataFrame(columns=["decision_date", "reason_code", "reason_detail", "n"])
    else:
        out = (
            df.groupby(["decision_date", "reason_code", "reason_detail"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        out["decision_date"] = [_iso(d) for d in out["decision_date"]]
    ctx.write_tsv("record_reasons.tsv", out)
    return {"rows": len(out)}


RECORD_ITEMS: tuple[tuple[str, Callable[[Ctx], dict[str, Any]]], ...] = (
    ("ind", rec_ind),
    ("per_date", rec_per_date),
    ("by_group", rec_by_group),
    ("top1", rec_top1),
    ("spearman", rec_spearman),
    ("unclipped", rec_unclipped),
    ("p6_tradable", rec_p6),
    ("holding", rec_holding),
    ("y2", rec_y2),
    ("ocf", rec_ocf),
    ("completeness", rec_completeness),
    ("reasons", rec_reasons),
)


def _run_record(ctx: Ctx, name: str, fn: Callable[[Ctx], dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    try:
        info = fn(ctx) or {}
        return {"status": "ok", "seconds": round(time.time() - t0, 1), **_py(info)}
    except Exception as e:  # record items never stop the run
        return {
            "status": "error",
            "seconds": round(time.time() - t0, 1),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_out_dir(paths: DataPaths, mode: Mode, out: str | None) -> Callable[[], Path]:
    def make() -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        d = Path(out) if out else paths.out_root / f"{mode.cmd}_{stamp}"
        d.mkdir(parents=True, exist_ok=False)
        return d

    return make


def run_sealed(
    paths: DataPaths,
    make_out: Callable[[], Path],
    *,
    dry_run_dev: bool,
    ledger_path: str | Path | None = None,
    repo_dir: str | Path | None = None,
    git_runner: seal.GitRunner | None = None,
    chosen_path: Path | None = None,
    chosen_sha256: str = CHOSEN_SHA256,
    f4_manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    """Run the verdict (or its DEV dry run). Returns (output dir, manifest path)."""
    t_start = time.time()
    run_at = datetime.now(UTC)
    mode = make_mode(dry_run_dev)

    # 1. pre-open checks - nothing below touches the seal
    git = _git_state()
    if git["dirty"] and not mode.dry_run:
        raise SystemExit("refused: earnings_a code or tests are dirty")
    _verify_inputs(paths)
    chosen = _load_chosen(chosen_path or paths.out_root / CHOSEN_REL, chosen_sha256)
    ledger = Path(ledger_path) if ledger_path else seal.default_ledger_path()
    if not mode.dry_run and ledger.parent.resolve() != paths.out_root.resolve():
        raise SealedAbort("seal ledger is not under the output root of the data paths")

    # 2. label-free preparation and the F4 integrity check, still before opening
    lf = _label_free(paths)
    f4 = _f4_check(paths, lf, mode, f4_manifest_path or paths.out_root / F4_REL)

    # 3. open (once)
    out_dir = make_out()
    if mode.dry_run:
        grant, ledger_rec = seal.grant_dev(), None
    else:
        grant = seal.open_seal_grant(
            Phase.SEALED_RUN,
            repo_dir=repo_dir or manifest.repo_root(),
            ledger_path=ledger,
            git_runner=git_runner,
        )
        ledger_rec = grant.record
    ctx = Ctx(
        paths,
        mode,
        grant,
        chosen,
        out_dir,
        run_at,
        lf["feats"],
        lf["receipts"],
        lf["stock_master"],
        lf["last_trade"],
    )
    sections: dict[str, Any] = {}
    status = "primary_failed"
    try:
        # 4. primary verdict, written first
        t0 = time.time()
        ctx.lab, ctx.r2 = _load_labels(
            paths, ctx.feats, ctx.receipts, mode.decision_dates, "oi_y1", grant
        )
        ctx.ds = build_dataset(ctx.feats, ctx.lab, mode.decision_dates)
        ctx.pred = ctx.walk(ctx.ds, ctx.primary_specs(), mode.eval_dates, mode.decision_dates)
        j = join_labels(ctx.pred, ctx.ds)
        assert_same_rows(j)
        lab_eval = ctx.ds.loc[
            ctx.ds["decision_date"]
            .map(seal._to_date)
            .isin({seal._to_date(d) for d in mode.eval_dates})
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ver = evaluate.h1_h2(
                ctx.pred, lab_eval, decision_dates=list(mode.eval_dates), grant=grant
            )
        doc = _verdict_doc(ver, mode, chosen)
        ctx.write_json("verdict.json", doc)
        status = "primary_written"
        sections["primary"] = {
            "status": "ok",
            "seconds": round(time.time() - t0, 1),
            "outcome": doc["outcome"],
        }
        # 5. record-only items
        for name, fn in RECORD_ITEMS:
            sections[name] = _run_record(ctx, name, fn)
        try:
            chk = judge(ctx.pred, lab_eval, "y_clip")
            sections["judge_matches_h1_h2"] = bool(
                chk["h1"]["mae_ratio"] == ver["h1"]["mae_ratio"]
                and chk["h1"]["lower"] == ver["h1"]["lower"]
            )
        except Exception as e:
            sections["judge_matches_h1_h2"] = f"error: {type(e).__name__}"
        status = "complete"
    except BaseException as e:
        sections["primary_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if status == "primary_failed":
            ctx.write_json("verdict_FAILED.json", {"error": sections["primary_error"]})
        raise
    finally:
        extra = {
            "mode": "dry_run_dev" if mode.dry_run else "sealed_run",
            "status": status,
            "run_at": run_at.isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - t_start, 1),
            "ledger_record": ledger_rec,
            "tag_commit": (ledger_rec or {}).get("tag_commit"),
            "chosen": {
                "sha256": chosen["sha256"],
                "HGB": chosen["HGB"].spec_id,
                "RIDGE": chosen["RIDGE"].spec_id,
            },
            "f4_check": f4,
            "sections": sections,
            "eval_dates": [_iso(d) for d in mode.eval_dates],
            "decision_dates": [_iso(d) for d in mode.decision_dates],
            "y2_dates": [_iso(d) for d in mode.y2_dates],
            "readings": [
                "E-3: universe widened (train+eval), predictors retrained, clip bounds frozen",
                "E-10: eval rows restricted to p6_tradable, models trained on the full universe",
                "KSIC 대분류: section letter mapped from ksic2 (KSIC rev.10)",
                "OCF target: PERS with oi_ttm and with ocf_ttm",
                "y2: DEV <= 2021-06-30 and SEALED <= 2024-06-30, no GAP",
            ],
            "dirty_recorded": bool(git["dirty"]),
            "output_sha256": {k: _sha(p) for k, p in ctx.files.items() if p.exists()},
        }
        m = manifest.build_manifest(mode.kind, paths, code=git, extra=_py(extra))
        mpath = manifest.write_once(out_dir / "manifest_sealed.json", m)
    return out_dir, mpath
