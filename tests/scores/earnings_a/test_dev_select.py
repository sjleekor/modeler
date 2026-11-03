"""Synthetic tests for dev_select (no real data)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import cli, dev_select, labels, seal, train
from modeler.scores.earnings_a.spec import FEATURES, INNER_VAL_DATES

DEV = list(seal.DEV_DATES)


def _hgb(lr, leaves):
    return train.ModelSpec(
        "HGB",
        {
            "learning_rate": lr,
            "max_iter": 5,
            "max_leaf_nodes": leaves,
            "min_samples_leaf": 5,
            "early_stopping": False,
            "loss": "absolute_error",
        },
    )


SPECS = {
    "HGB": [_hgb(0.05, 15), _hgb(0.1, 15)],
    "RIDGE": [train.ModelSpec("RIDGE", {"alpha": a}) for a in (0.1, 1.0, 100.0)],
    "PERS": [train.ModelSpec("PERS")],
    "IND": [train.ModelSpec("IND")],
}


def make_ds(n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in DEV:
        for i in range(n):
            f = {c: rng.normal() for c in FEATURES}
            rows.append(
                {
                    "decision_date": d,
                    "ticker": f"T{i:03d}0",
                    "corp_code": f"C{i:03d}",
                    "ksic2": str(10 + i % 3),
                    **f,
                    "input_eligible": True,
                    "label_ok": i % 7 != 0,
                    "y_clip": f["oi_ttm"] + rng.normal(scale=0.5),
                    "label_pub_date": d + timedelta(days=30),
                }
            )
    return pd.DataFrame(rows)


def test_selection_and_same_rows():
    ds = make_ds()
    res = dev_select.score_specs(ds, grant=seal.grant_dev(), specs=SPECS)
    grid = res["grid"].set_index("spec_id")
    for g in ("HGB", "RIDGE"):
        ids = [s.spec_id for s in SPECS[g]]
        assert res["chosen"][g].spec_id == grid.loc[ids, "mean_mae"].idxmin()
    assert len(grid) == 2 + 3 + 2
    assert (grid["n_dates"] == len(INNER_VAL_DATES)).all()
    pd_ = res["per_date"]
    # label_ok rows only; same rows for HGB and Ridge specs per date
    n = pd_.loc[pd_["model"].isin(["HGB", "RIDGE"])].groupby("decision_date")["n_scored"].nunique()
    assert (n == 1).all()
    assert set(res["eval_rows_by_date"].values()) == {30}
    assert set(res["scored_rows_by_date"].values()) == {30 - 5}  # i in {0,7,14,21,28} unlabeled


def test_ridge_tie_goes_to_grid_order():
    ds = make_ds()
    dup = {**SPECS, "RIDGE": [train.ModelSpec("RIDGE", {"alpha": 1.0})] * 2}
    res = dev_select.score_specs(ds, grant=seal.grant_dev(), specs=dup)
    assert res["chosen"]["RIDGE"].params == {"alpha": 1.0}


def test_same_rows_assertion_fires(monkeypatch):
    ds = make_ds()
    orig = train.walk_forward

    def drop_one(*a, **k):
        out = orig(*a, **k)
        bad = out.index[(out["model"] == "RIDGE") & (out["spec_id"].str.contains("0.1"))][0]
        return out.drop(index=bad)

    monkeypatch.setattr(train, "walk_forward", drop_one)
    with pytest.raises(AssertionError):
        dev_select.score_specs(ds, grant=seal.grant_dev(), specs=SPECS)


def test_refuses_non_dev_dates_and_grant():
    ds = make_ds()
    with pytest.raises(seal.SealViolation):
        dev_select.score_specs(
            ds,
            grant=seal.grant_dev(),
            specs=SPECS,
            decision_dates=[*DEV, seal.SEALED_DATES[0]],
            eval_dates=[seal.SEALED_DATES[0]],
        )
    with pytest.raises(seal.SealViolation):
        dev_select.score_specs(ds, grant=object(), specs=SPECS)  # type: ignore[arg-type]
    late = ds.copy()
    late.loc[late.index[0], "decision_date"] = seal.GAP_DATES[0]
    with pytest.raises(seal.SealViolation):
        dev_select.score_specs(late, grant=seal.grant_dev(), specs=SPECS)


def test_no_sealed_date_requested(monkeypatch):
    seen: list = []
    ow = train.walk_forward

    def rec_wf(*a, **k):
        seen.append(("wf", list(k["decision_dates"]), list(k["eval_dates"])))
        return ow(*a, **k)

    monkeypatch.setattr(train, "walk_forward", rec_wf)
    dev_select.score_specs(make_ds(), grant=seal.grant_dev(), specs=SPECS)
    allowed = set(DEV)
    for _, dd, ed in seen:
        assert set(dd) <= allowed and set(ed) <= set(INNER_VAL_DATES) <= allowed


def test_load_dev_labels_uses_only_dev_dates(monkeypatch):
    calls = []

    def fake_load(paths, *, decision_dates, grant, horizon, metrics):
        calls.append(list(decision_dates))
        return pd.DataFrame()

    def fake_build(r2, anchor, receipts, *, decision_dates, grant, target):
        calls.append(list(decision_dates))
        assert target == "oi_y1"
        return pd.DataFrame()

    monkeypatch.setattr(labels, "load_label_rows", fake_load)
    monkeypatch.setattr(labels, "build_labels", fake_build)
    feats = make_ds(3)
    feats["input_eligible_incl_holding"] = True
    feats["basis"] = "CFS"
    feats["a_t_dec"] = "1"
    out = dev_select.load_dev_labels(None, feats, pd.DataFrame(), seal.grant_dev())  # type: ignore[arg-type]
    assert out.empty and calls and all(set(c) == set(DEV) for c in calls)


def test_build_dataset_join():
    feats = make_ds(4)
    feats["input_eligible_incl_holding"] = True
    feats["basis"] = "CFS"
    feats = feats.drop(columns=["label_ok", "y_clip", "label_pub_date"])
    lab = feats[["decision_date", "ticker", "basis"]].copy()
    lab["label_ok"] = True
    lab["y_clip"] = 0.1
    lab["label_pub_date"] = lab["decision_date"]
    lab = lab.iloc[:-1]
    ds = dev_select.build_dataset(feats, lab)
    assert len(ds) == len(feats) and ds["label_ok"].sum() == len(feats) - 1
    late = feats.copy()
    late.loc[late.index[0], "decision_date"] = seal.SEALED_DATES[0]
    late["input_eligible_incl_holding"] = True
    assert len(dev_select.build_dataset(late, lab.iloc[:0])) == len(late) - 1  # filtered, not read


def test_cli_has_dev_subcommand_and_refuses_dirty(monkeypatch):
    monkeypatch.setattr(
        cli.manifest, "git_state", lambda _: {"head": "x", "dirty": True, "describe": "d"}
    )
    with pytest.raises(SystemExit):
        cli.run_dev_cmd(None, None, rehearsal=False)  # type: ignore[arg-type]
