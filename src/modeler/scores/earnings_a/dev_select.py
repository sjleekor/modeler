"""Development-window spec selection (preregistration section 6.2).

Inner walk-forward over ``spec.INNER_VAL_DATES`` under the DEV grant only. For every spec
the per-date MAE (label_ok rows, ``y_clip``) is averaged over the 13 dates; the HGB spec and
the Ridge alpha are picked with ``train.select_spec`` (ties go to grid order). PERS and IND
are scored for the record. Output is aggregated MAE per spec and date; no label value and no
prediction row leaves this module.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from modeler.scores.earnings_a import digest, labels, seal, train
from modeler.scores.earnings_a.paths import DataPaths
from modeler.scores.earnings_a.spec import HGB_GRID, INNER_VAL_DATES, RIDGE_ALPHAS, SEEDS
from modeler.scores.earnings_a.train import ModelSpec

TARGET = "oi_y1"
Y_COL = "y_clip"
JOIN_KEYS = ["decision_date", "ticker", "basis"]


def spec_lists() -> dict[str, list[ModelSpec]]:
    return {
        "HGB": [ModelSpec("HGB", dict(g)) for g in HGB_GRID],
        "RIDGE": [ModelSpec("RIDGE", {"alpha": float(a)}) for a in RIDGE_ALPHAS],
        "PERS": [ModelSpec("PERS")],
        "IND": [ModelSpec("IND")],
    }


def load_dev_labels(
    paths: DataPaths, feats: pd.DataFrame, receipts: pd.DataFrame, grant: seal.SealGrant
) -> pd.DataFrame:
    """DEV labels for ``oi_y1`` (same anchor rows as the F4 digest), guarded by ``grant``."""
    dates = tuple(seal.DEV_DATES)
    metric, horizon = labels.TARGETS[TARGET]
    anchor = digest._anchor_rows(feats, dates)[digest.ANCHOR_COLS].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = labels.load_label_rows(
            paths, decision_dates=dates, grant=grant, horizon=horizon, metrics=(metric,)
        )
        return labels.build_labels(
            r2, anchor, receipts, decision_dates=dates, grant=grant, target=TARGET
        )


def build_dataset(feats: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Join DEV labels onto the clipped features: the frame ``train.walk_forward`` expects."""
    dates = set(seal.DEV_DATES)
    sub = digest._anchor_rows(feats, tuple(seal.DEV_DATES))
    if any(seal._to_date(d) not in dates for d in sub["decision_date"]):
        raise seal.SealViolation("feature rows outside DEV dates")
    keep = lab[[*JOIN_KEYS, "label_ok", Y_COL, "label_pub_date"]]
    ds = sub.merge(keep, on=JOIN_KEYS, how="left", validate="one_to_one")
    ds["label_ok"] = ds["label_ok"].eq(True)
    return ds.reset_index(drop=True)


def _digest_ds(ds: pd.DataFrame) -> str:
    cols = ["decision_date", "ticker", "input_eligible", "label_ok"]
    return hashlib.sha256(
        pd.util.hash_pandas_object(ds[cols].astype(str), index=False).to_numpy().tobytes()
    ).hexdigest()


def score_specs(
    ds: pd.DataFrame,
    *,
    grant: seal.SealGrant,
    eval_dates: Sequence = INNER_VAL_DATES,
    decision_dates: Sequence = tuple(seal.DEV_DATES),
    specs: dict[str, list[ModelSpec]] | None = None,
) -> dict[str, Any]:
    """Walk forward, score, select. Returns aggregated numbers only."""
    specs = specs or spec_lists()
    flat = [s for group in ("HGB", "RIDGE", "PERS", "IND") for s in specs[group]]
    pred = train.walk_forward(
        ds,
        eval_dates=list(eval_dates),
        specs=flat,
        decision_dates=list(decision_dates),
        grant=grant,
    )
    lab = ds[["decision_date", "ticker", "corp_code", "label_ok", Y_COL]]
    j = pred.merge(lab, on=["decision_date", "ticker", "corp_code"], how="left", validate="m:1")
    j["label_ok"] = j["label_ok"].fillna(False).astype(bool)

    # Every spec is scored on the same rows per date.
    keysets: dict[Any, set] = {}
    for (d, sid), g in j.groupby(["decision_date", "spec_id"], sort=False):
        ks = set(zip(g["ticker"], g["corp_code"], strict=True))
        if d in keysets and keysets[d] != ks:
            raise AssertionError(f"specs scored on different rows at {d}: {sid}")
        keysets.setdefault(d, ks)
    scored = j.loc[j["label_ok"] & j[Y_COL].notna()]
    for group in ("HGB", "RIDGE"):
        sub = scored.loc[scored["model"] == group]
        if sub["y_hat"].isna().any():
            raise AssertionError(f"{group}: NaN predictions on scored rows")
    ref_sets: dict[Any, set] = {}
    for (d, sid), g in scored.loc[scored["model"].isin(["HGB", "RIDGE"])].groupby(
        ["decision_date", "spec_id"], sort=False
    ):
        ks = set(zip(g["ticker"], g["corp_code"], strict=True))
        if d in ref_sets and ref_sets[d] != ks:
            raise AssertionError(f"scored rows differ across specs at {d}: {sid}")
        ref_sets.setdefault(d, ks)

    long_rows = []
    order = {s.spec_id: i for i, s in enumerate(flat)}
    for (sid, d), g in scored.groupby(["spec_id", "decision_date"], sort=False):
        g = g.loc[g["y_hat"].notna()]
        long_rows.append(
            {
                "spec_id": sid,
                "model": g["model"].iloc[0] if len(g) else "",
                "decision_date": seal._to_date(d).isoformat(),
                "n_eval_rows": len(keysets[d]),
                "n_scored": len(g),
                "mae": float((g[Y_COL] - g["y_hat"]).abs().mean()) if len(g) else float("nan"),
            }
        )
    per_date = pd.DataFrame(long_rows)
    per_date["_o"] = per_date["spec_id"].map(order)
    per_date = per_date.sort_values(["_o", "decision_date"]).drop(columns="_o")
    per_date = per_date.reset_index(drop=True)
    n_dates = len(list(eval_dates))
    if per_date.groupby("spec_id")["decision_date"].nunique().min() != n_dates:
        raise AssertionError("a spec has no scored rows on some eval date")

    grid = []
    for s in flat:
        g = per_date.loc[per_date["spec_id"] == s.spec_id]
        grid.append(
            {
                "spec_id": s.spec_id,
                "model": s.name,
                "params": json.dumps(s.params, sort_keys=True),
                "mean_mae": float(g["mae"].mean()),
                "n_dates": int(len(g)),
            }
        )
    grid_df = pd.DataFrame(grid)

    chosen: dict[str, ModelSpec] = {}
    for group in ("HGB", "RIDGE"):
        chosen[group] = train.select_spec(j, specs[group], y_col=Y_COL)
        ids = [s.spec_id for s in specs[group]]
        sub = grid_df.set_index("spec_id").loc[ids, "mean_mae"]
        if chosen[group].spec_id != sub.idxmin():  # idxmin returns the first minimum
            raise AssertionError(f"{group}: select_spec disagrees with the grid table")
    return {
        "per_date": per_date,
        "grid": grid_df,
        "chosen": chosen,
        "eval_rows_by_date": {seal._to_date(d).isoformat(): len(k) for d, k in keysets.items()},
        "scored_rows_by_date": {seal._to_date(d).isoformat(): len(k) for d, k in ref_sets.items()},
    }


def run_dev_selection(paths: DataPaths, con, *, run_at: datetime) -> dict[str, Any]:
    """Build DEV inputs and labels, score every spec, pick the HGB spec and Ridge alpha."""
    del run_at  # recorded by the caller's manifest
    grant = seal.grant_dev()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, feats = digest.prepare_inputs(paths, con)
        receipts = digest.load_receipts(con)
    lab = load_dev_labels(paths, feats, receipts, grant)
    ds = build_dataset(feats, lab)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = score_specs(ds, grant=grant)
    res["n_dev_rows"] = int(len(ds))
    res["dataset_key_sha256"] = _digest_ds(ds)
    res["seeds"] = list(SEEDS)
    return res


def write_outputs(out_dir: Path, res: dict[str, Any]) -> dict[str, Path]:
    """Write grid TSV, per-date long TSV and chosen.json (write-once)."""
    from modeler.scores.earnings_a import manifest

    grid, per_date = res["grid"], res["per_date"]
    wide = per_date.pivot(index="spec_id", columns="decision_date", values="mae")
    wide = wide.reindex(grid["spec_id"])
    grid_out = pd.concat([grid.set_index("spec_id"), wide], axis=1).reset_index()
    p_grid, p_long = out_dir / "grid.tsv", out_dir / "per_date.tsv"
    for p, df in ((p_grid, grid_out), (p_long, per_date)):
        with open(p, "x", encoding="utf-8") as f:
            df.to_csv(f, sep="\t", index=False, float_format="%.12g")
    chosen = {
        k: {"spec_id": s.spec_id, "name": s.name, "params": s.params}
        for k, s in res["chosen"].items()
    }
    chosen["eval_rows_by_date"] = res["eval_rows_by_date"]
    chosen["scored_rows_by_date"] = res["scored_rows_by_date"]
    p_ch = manifest.write_once(out_dir / "chosen.json", chosen)
    return {"grid": p_grid, "per_date": p_long, "chosen": p_ch}


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


__all__ = ["run_dev_selection", "score_specs", "write_outputs"]
