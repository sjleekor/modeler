"""Synthetic tests for train.py and evaluate.py (no real data)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import evaluate, seal, train
from modeler.scores.earnings_a.spec import FEATURES, SEEDS

DATES = list(seal.DEV_DATES[4:10])  # six DEV dates
SPEC_HGB = train.ModelSpec(
    "HGB", {"max_iter": 5, "min_samples_leaf": 5, "early_stopping": False, "loss": "absolute_error"}
)


def make_ds(n_firms=40, dates=DATES, seed=0, lag_days=200):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for i in range(n_firms):
            feats = {f: rng.normal() for f in FEATURES}
            rows.append(
                {
                    "decision_date": d,
                    "ticker": f"T{i:03d}0",
                    "corp_code": f"C{i:03d}",
                    "ksic2": str(10 + i % 3),
                    **feats,
                    "input_eligible": True,
                    "label_ok": True,
                    "y_clip": feats["oi_ttm"] + rng.normal(scale=0.5),
                    "label_pub_date": d + timedelta(days=lag_days),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def grant():
    return seal.grant_dev()


def kw(grant, dates=DATES):
    return {"decision_dates": list(dates), "grant": grant}


def test_missing_grant_raises():
    ds = make_ds(5)
    with pytest.raises(seal.SealViolation):
        train.fit_predict_date(
            ds, eval_date=DATES[-1], spec=train.ModelSpec("PERS"), decision_dates=DATES
        )


def test_dev_grant_with_sealed_rows_raises_before_fit(grant, monkeypatch):
    called = []
    monkeypatch.setattr(train, "_fit_predict", lambda *a, **k: called.append(1))
    ds = make_ds(5, dates=DATES + [seal.SEALED_DATES[0]])
    # grant rejects the passed sealed date
    with pytest.raises(seal.SealViolation):
        train.walk_forward(
            ds,
            eval_dates=[DATES[-1]],
            specs=[train.ModelSpec("PERS")],
            **kw(grant, DATES + [seal.SEALED_DATES[0]]),
        )
    # passed list is clean but ds holds a sealed date
    with pytest.raises(seal.SealViolation):
        train.fit_predict_date(
            ds, eval_date=DATES[-1], spec=train.ModelSpec("PERS"), **kw(grant, DATES)
        )
    assert called == []


def test_train_rows_respect_pub_date_and_date(grant, monkeypatch):
    ds = make_ds(10, lag_days=100)  # quarter-end gap 91 days -> only some dates published
    # make the last date's neighbours unpublished: lag 100 > 91
    seen = {}
    orig = train._split

    def spy(d, e):
        tr, ev = orig(d, e)
        seen["tr"], seen["ev"], seen["e"] = tr, ev, e
        return tr, ev

    monkeypatch.setattr(train, "_split", spy)
    out = train.fit_predict_date(
        ds, eval_date=DATES[-1], spec=train.ModelSpec("RIDGE", {"alpha": 1.0}), **kw(grant)
    )
    tr, e = seen["tr"], pd.Timestamp(seen["e"])
    assert len(tr) > 0
    assert (pd.to_datetime(tr["label_pub_date"]) <= e).all()
    assert (pd.to_datetime(tr["decision_date"]) < e).all()
    # lag 100 days: the immediately preceding date (91 days earlier) is published only at +100
    assert DATES[-2] not in set(tr["decision_date"])
    assert DATES[-3] in set(tr["decision_date"])
    assert set(out["decision_date"]) == {DATES[-1]}


def test_three_predictors_share_eval_rows(grant):
    ds = make_ds(30)
    ds.loc[ds.index[-3:], "input_eligible"] = False
    specs = [
        train.ModelSpec("PERS"),
        train.ModelSpec("RIDGE", {"alpha": 1.0}),
        SPEC_HGB,
        train.ModelSpec("IND"),
    ]
    out = train.walk_forward(ds, eval_dates=DATES[-2:], specs=specs, **kw(grant))
    keys = {
        m: set(zip(g["decision_date"], g["ticker"], strict=True)) for m, g in out.groupby("model")
    }
    assert len(keys) == 4
    assert len({frozenset(v) for v in keys.values()}) == 1
    assert len(keys["PERS"]) == 2 * 30 - 3  # three ineligible rows on the last date
    assert list(out.columns) == [
        "decision_date",
        "ticker",
        "corp_code",
        "model",
        "spec_id",
        "y_hat",
    ]
    pers = out[out.model == "PERS"].merge(ds, on=["decision_date", "ticker"])
    assert np.allclose(pers["y_hat"], pers["oi_ttm"])


def test_hgb_averages_five_seeds(grant, monkeypatch):
    seeds = []

    class Fake:
        def __init__(self, **kw):
            self.rs = kw["random_state"]
            seeds.append(self.rs)

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.full(len(x), float(self.rs))

    monkeypatch.setattr(train, "HistGradientBoostingRegressor", Fake)
    out = train.fit_predict_date(ds=make_ds(10), eval_date=DATES[-1], spec=SPEC_HGB, **kw(grant))
    assert seeds == list(SEEDS) and len(seeds) == 5
    assert np.allclose(out["y_hat"], np.mean(SEEDS))


def test_hgb_real_fit_runs(grant):
    out = train.fit_predict_date(ds=make_ds(40), eval_date=DATES[-1], spec=SPEC_HGB, **kw(grant))
    assert out["y_hat"].notna().all() and len(out) == 40


def test_hgb_all_nan_train_column_dropped(grant):
    """C1: a feature with no value in the training rows is left out of that fit only."""
    ds = make_ds(40)
    last = ds["decision_date"] == DATES[-1]
    ds.loc[~last, "capex_ttm"] = np.nan  # all-NaN in training, observed on the eval date
    out = train.fit_predict_date(ds=ds, eval_date=DATES[-1], spec=SPEC_HGB, **kw(grant))
    assert out["y_hat"].notna().all() and len(out) == 40
    ds2 = ds.copy()
    ds2.loc[last, "capex_ttm"] = 1e9  # eval values of a dropped column cannot move predictions
    out2 = train.fit_predict_date(ds=ds2, eval_date=DATES[-1], spec=SPEC_HGB, **kw(grant))
    assert np.allclose(out["y_hat"], out2["y_hat"])


def test_ridge_matrix_flags_and_fill():
    ds = make_ds(6, dates=DATES[:1])
    ds.loc[ds.index[0], "oi_q1"] = np.nan
    ds.loc[ds.index[1], "oi_q1"] = np.nan
    x = train._ridge_matrix(ds)
    assert x.shape == (6, 2 * len(FEATURES))
    assert list(x.columns[len(FEATURES) :]) == [f"{f}_isna" for f in FEATURES]
    assert x["oi_q1_isna"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert (x["oi_q1"].iloc[:2] == 0.5).all()
    assert x.isna().sum().sum() == 0
    assert x["oi_ttm"].min() == 0.0 and x["oi_ttm"].max() == 1.0
    assert (x["oi_ttm_isna"] == 0).all()


def test_ridge_uses_matrix(grant, monkeypatch):
    shapes = []
    orig = train.Ridge

    class Spy(orig):
        def fit(self, X, y):
            shapes.append(X.shape)
            return super().fit(X, y)

    monkeypatch.setattr(train, "Ridge", Spy)
    train.fit_predict_date(
        ds=make_ds(12),
        eval_date=DATES[-1],
        spec=train.ModelSpec("RIDGE", {"alpha": 1.0}),
        **kw(grant),
    )
    assert shapes[0][1] == 2 * len(FEATURES)


def test_ind_fallback(grant):
    n_a = 25
    rows = []
    d0, d1 = DATES[0], DATES[-1]
    for i in range(n_a + 5 + 3):
        grp = "A" if i < n_a else ("B" if i < n_a + 5 else None)
        rows.append((i, grp))
    base = make_ds(1, dates=[d0]).iloc[0].to_dict()
    recs = []
    for i, grp in rows:
        r = dict(base, ticker=f"T{i}", corp_code=f"C{i}", ksic2=grp, oi_ttm=0.0)
        r.update(decision_date=d0, y_clip=2.0 if grp == "A" else 100.0)
        recs.append(r)
    ev = []
    for key, grp in (("a", "A"), ("b", "B"), ("n", None)):
        ev.append(
            dict(
                base,
                ticker=f"E{key}",
                corp_code=f"E{key}",
                ksic2=grp,
                oi_ttm=0.0,
                decision_date=d1,
                label_ok=False,
                y_clip=np.nan,
            )
        )
    ds = pd.DataFrame(recs + ev)
    ds["label_pub_date"] = ds["decision_date"] + timedelta(days=100)
    out = train.fit_predict_date(
        ds=ds, eval_date=d1, spec=train.ModelSpec("IND"), **kw(grant, [d0, d1])
    ).set_index("ticker")["y_hat"]
    overall = (n_a * 2.0 + 8 * 100.0) / (n_a + 8)
    assert out["Ea"] == pytest.approx(2.0)
    assert out["Eb"] == pytest.approx(overall)
    assert out["En"] == pytest.approx(overall)


# ---------------------------------------------------------------- evaluate


def make_eval(hgb_noise, n_firms=30, n_dates=4, seed=1, ridge_noise=0.6, pers_noise=1.0):
    rng = np.random.default_rng(seed)
    dates = DATES[:n_dates]
    rows, labs = [], []
    for d in dates:
        for i in range(n_firms):
            y = rng.normal()
            labs.append({"decision_date": d, "ticker": f"T{i}", "label_ok": True, "y_clip": y})
            for m, nz in (("PERS", pers_noise), ("RIDGE", ridge_noise), ("HGB", hgb_noise)):
                yh = y + (nz if nz is not None else 0) * rng.choice([-1, 1]) * (0.5 + rng.random())
                rows.append(
                    {
                        "decision_date": d,
                        "ticker": f"T{i}",
                        "corp_code": f"C{i}",
                        "model": m,
                        "y_hat": yh,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(labs), dates


def run_eval(preds, labels, dates, grant, **extra):
    return evaluate.h1_h2(
        preds,
        labels,
        decision_dates=dates,
        grant=grant,
        n_dates=len(dates),
        min_wins_h1=3,
        min_wins_h2=2,
        draws=400,
        **extra,
    )


def test_h1_passes_when_hgb_better_and_h2_runs(grant):
    preds, labels, dates = make_eval(hgb_noise=0.1)
    res = run_eval(preds, labels, dates, grant)
    assert res["h1"]["pass"] and res["h1"]["wins"] == 4 and res["h1"]["mae_ratio"] < 0.95
    assert res["h2"] is not None and res["h2"]["pass"]
    assert res["n_rows"] == 120 and res["n_firms"] == 30
    assert set(res["per_date_mae"]["HGB"]) == {str(d) for d in dates}


def test_h1_fails_on_equal_predictions_and_h2_skipped(grant):
    preds, labels, dates = make_eval(hgb_noise=1.0)
    preds.loc[preds.model == "HGB", "y_hat"] = preds.loc[preds.model == "PERS", "y_hat"].to_numpy()
    res = run_eval(preds, labels, dates, grant)
    assert not res["h1"]["pass"] and res["h1"]["wins"] == 0
    assert res["h1"]["mae_ratio"] == pytest.approx(1.0)
    assert res["h2"] is None and "RIDGE" not in res["per_date_mae"]


def test_h2_fails_when_ridge_equals_hgb(grant):
    preds, labels, dates = make_eval(hgb_noise=0.1)
    preds.loc[preds.model == "RIDGE", "y_hat"] = preds.loc[preds.model == "HGB", "y_hat"].to_numpy()
    res = run_eval(preds, labels, dates, grant)
    assert res["h1"]["pass"] and res["h2"] is not None and not res["h2"]["pass"]


def test_bootstrap_reproducible_and_seed_sensitive(grant):
    preds, labels, dates = make_eval(hgb_noise=0.8)
    a = run_eval(preds, labels, dates, grant, seed=5)
    b = run_eval(preds, labels, dates, grant, seed=5)
    c = run_eval(preds, labels, dates, grant, seed=6)
    assert a == b
    assert a["h1"]["lower"] != c["h1"]["lower"]


def test_rows_restricted_to_common_label_ok(grant):
    preds, labels, dates = make_eval(hgb_noise=0.1)
    labels.loc[0, "label_ok"] = False
    preds = preds[
        ~((preds.model == "RIDGE") & (preds.ticker == "T1") & (preds.decision_date == dates[1]))
    ]
    res = run_eval(preds, labels, dates, grant)
    assert res["n_rows"] == 118


def test_h1_h2_guard(grant):
    preds, labels, dates = make_eval(hgb_noise=0.1)
    with pytest.raises(seal.SealViolation):
        evaluate.h1_h2(preds, labels, decision_dates=dates)
    with pytest.raises(seal.SealViolation):
        evaluate.h1_h2(preds, labels, decision_dates=dates[:2], grant=grant)


def test_select_spec_tie_rule():
    s = [train.ModelSpec("RIDGE", {"alpha": a}) for a in (0.1, 1.0, 10.0)]
    rows = []
    for d in DATES[:2]:
        for spec, err in zip(s, (2.0, 1.0, 1.0), strict=True):
            rows += [
                {
                    "decision_date": d,
                    "spec_id": spec.spec_id,
                    "y_hat": err * sg,
                    "y_clip": 0.0,
                    "label_ok": True,
                }
                for sg in (1, -1)
            ]
    assert train.select_spec(pd.DataFrame(rows), s).params["alpha"] == 1.0  # tie -> first
    assert train.select_spec(pd.DataFrame(rows), s[::-1]).params["alpha"] == 10.0
