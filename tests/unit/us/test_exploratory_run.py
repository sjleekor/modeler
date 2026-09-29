from datetime import date

import joblib
import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.exploratory_run import (
    DATASETS,
    FEATURES,
    load_inputs,
    make_model,
    prepare_features,
    rankings,
    score_model,
    training_rows,
)


def test_late_labels_never_reach_training_or_evaluation(tmp_path):
    dates = [date(2025, 5, 1), date(2025, 6, 2), date(2025, 7, 1), date(2026, 9, 1)]
    base = pl.DataFrame(
        {
            "date": dates,
            "symbol": ["A"] * 4,
            "price_ge_5": [True] * 4,
            "close": [10.0] * 4,
            "adv_20d": [1e7] * 4,
            **{c: [1.0] * 4 for c in FEATURES[:-10]},
        }
    )
    flow = pl.DataFrame(
        {"date": dates, "symbol": ["A"] * 4, **{c: [2.0] * 4 for c in FEATURES[-10:]}}
    )
    labels = pl.DataFrame(
        {
            "date": dates,
            "symbol": ["A"] * 4,
            "terminal_date": [
                date(2025, 6, 2),
                date(2025, 7, 2),
                date(2025, 8, 1),
                date(2026, 10, 1),
            ],
            "L0": [0.1, 999.0, 999.0, 999.0],
            "L2": [0.05, 999.0, 999.0, 999.0],
            "y_rank": [0.7, 999.0, 999.0, 999.0],
        }
    )
    root = DataRoot(tmp_path)
    for name, frame in zip(DATASETS, [base, flow, labels]):
        folder = root.datasets / name
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text("{}")
        frame.write_parquet(folder / "part.parquet")
    core, latest, _ = load_inputs(root)
    assert core["date"].to_list() == [date(2025, 5, 1)]
    assert core["y_rank"].to_list() == [0.7]
    assert latest["date"].unique().to_list() == [date(2026, 9, 1)]
    assert "y_rank" not in latest.columns


def test_training_split_purges_labels_reaching_validation():
    d = date(2024, 1, 2)
    frame = pl.DataFrame(
        {
            "date": [d] * 3,
            "symbol": ["A", "B", "C"],
            "terminal_date": [date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 2)],
        }
    )
    assert training_rows(frame, [d], date(2024, 2, 1))["symbol"].to_list() == ["A"]


def test_saved_scorer_handles_missing_values_and_stable_ties(tmp_path):
    raw = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 4,
            "symbol": ["C", "A", "B", "D"],
            "price_ge_5": [True] * 4,
            "close": [10.0] * 4,
            "adv_20d": [1e7] * 4,
            "x": [float("nan"), 1.0, 1.0, None],
        }
    )
    prepared = prepare_features(raw, ["x"])
    assert prepared["x_rank"].is_finite().all()
    assert prepared.filter(pl.col("symbol").is_in(["C", "D"]))["x_isna"].all()
    from modeler.us.m4_transform import to_design_arrays

    x, _ = to_design_arrays(prepared, ["x"])
    model = make_model("ridge").fit(x, np.array([0.1, 0.1, 0.9, 0.9]))
    original = score_model(model, prepared, ["x"])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = score_model(joblib.load(path), prepared, ["x"])
    np.testing.assert_allclose(original["pred"].to_numpy(), restored["pred"].to_numpy())
    ranks = rankings(prepared.with_columns(pl.lit(1.0).alias("pred")))
    assert ranks["symbol"].to_list() == ["A", "B", "C", "D"]
    assert ranks["rank"].to_list() == [1, 2, 3, 4]


def test_missing_benchmark_month_is_reported_without_dropping_strategy_month():
    from modeler.us.exploratory_run import evaluate

    dates = [date(2024, 1, 2), date(2024, 2, 1), date(2024, 3, 1)]
    rows = []
    for month, d in enumerate(dates):
        for i in range(25):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{i:02d}",
                    "close": 10.0,
                    "adv_20d": 1e7,
                    "rv_20": 0.02,
                    "L0": 0.01 + month * 0.005,
                    "pred": float(i),
                    "L2": float((i + month * 7) % 25),
                }
            )
    spy = pl.DataFrame({"date": [dates[0], dates[2]], "spy_h21_return": [0.02, 0.03]})
    summary, track, _ = evaluate(pl.DataFrame(rows), spy)
    assert summary["n_months"] == 3
    assert summary["spy_comparison_months"] == 2
    assert summary["spy_missing_dates"] == [str(dates[1])]
    assert track.height == 3
    assert track["spy_h21_return"].null_count() == 1
