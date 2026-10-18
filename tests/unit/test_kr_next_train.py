"""Training runner, bundle and CLI guard of the next KR model — synthetic panels only."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import train as tr

SMALL = ["f_a", "f_b", "f_c", "f_d"]


def make_panel(
    n_dates: int = 260,
    n_tickers: int = 40,
    cols: list[str] | None = None,
    seed: int = 0,
    label_name: str = "label_end_date",
    extra_flag: bool = False,
) -> pl.DataFrame:
    """Fake panel: weekday dates, two markets, signal in the first columns, c1-style labels."""
    cols = cols or SMALL
    rng = np.random.default_rng(seed)
    days: list[date] = []
    d = date(2017, 1, 2)
    while len(days) < n_dates:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = n_dates * n_tickers
    ti = np.tile(np.arange(n_tickers), n_dates)
    di = np.repeat(np.arange(n_dates), n_tickers)
    X = rng.normal(size=(rows, len(cols)))
    raw = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=1.0, size=rows)
    end_idx = di + 21
    has_label = end_idx < n_dates
    df = pl.DataFrame(
        {
            "trade_date": [days[i] for i in di],
            "market": ["KOSPI" if t < n_tickers // 2 else "KOSDAQ" for t in ti],
            "ticker": [f"T{t:03d}" for t in ti],
            "raw": raw,
            "has": has_label,
            label_name: [days[e] if e < n_dates else None for e in end_idx],
        }
    )
    for j, c in enumerate(cols):
        v = X[:, j].copy()
        v[rng.random(rows) < 0.1] = np.nan
        df = df.with_columns(pl.Series(c, v).fill_nan(None))
    df = df.with_columns(
        pl.when(pl.col("has")).then(pl.col("raw")).otherwise(None).alias("raw2")
    ).with_columns(
        pl.col("has").alias("trainable"),
        (
            (pl.col("raw2").rank("average").over("trade_date") - 1)
            / (pl.col("raw2").count().over("trade_date") - 1)
        ).alias("y"),
    )
    if extra_flag:
        df = df.with_columns((pl.Series(rng.random(rows) < 0.2)).alias("ttm_includes_conflict"))
    return df.drop("raw", "raw2", "has")


# ----------------------------------------------------------------------------- grid


def test_grid_has_16_unique_points():
    assert len(tr.GRID) == 16
    assert len({tuple(sorted(p.items())) for p in tr.GRID}) == 16
    assert {p["learning_rate"] for p in tr.GRID} == {0.03, 0.1}
    assert {p["max_leaf_nodes"] for p in tr.GRID} == {15, 31}
    assert {p["min_samples_leaf"] for p in tr.GRID} == {200, 1000}
    assert {p["l2_regularization"] for p in tr.GRID} == {0.0, 1.0}
    assert tr.MAX_ITER == 500


# ----------------------------------------------------------------------------- folds / purge


def test_folds_are_five_and_train_respects_label_end_purge():
    panel = make_panel()
    # a row early in time whose label ends very late must never be trained on in an early fold
    late = panel["label_end_date"].max()
    panel = panel.with_columns(
        pl.when((pl.col("ticker") == "T000") & (pl.col("trade_date") == date(2017, 1, 10)))
        .then(pl.lit(late))
        .otherwise(pl.col("label_end_date"))
        .alias("label_end_date")
    )
    prep = tr.prepare_panel(panel, SMALL)
    ctx = tr._context(prep, SMALL)
    assert len(ctx.folds) == 5
    dates = prep["trade_date"].to_numpy().astype("datetime64[D]")
    ends = prep["label_end_date"].to_numpy().astype("datetime64[D]")
    for fi in ctx.folds:
        b = np.datetime64(fi.fold.boundary)
        rows = np.concatenate([fi.fit, fi.inner_val])
        assert (dates[rows] < b).all()
        assert (ends[rows] < b).all()  # purge on label_end_date, not on a fixed horizon
        assert (dates[fi.valid] >= np.datetime64(fi.fold.valid_start)).all()
        assert (dates[fi.valid] <= np.datetime64(fi.fold.valid_end)).all()
        # inner split: validation is later than the fit rows, labels do not overlap
        assert dates[fi.fit].max() < dates[fi.inner_val].min()
        assert ends[fi.fit].max() < dates[fi.inner_val].min()
        cal = np.unique(dates)
        n_between = ((cal > dates[fi.fit].max()) & (cal < dates[fi.inner_val].min())).sum()
        assert n_between >= tr.INNER_GAP_SESSIONS  # 21 for c1 labels (label-end rule)
    # the late-label row sits in no fold's fit set
    pos = np.flatnonzero(
        (prep["ticker"] == "T000").to_numpy() & (dates == np.datetime64(date(2017, 1, 10)))
    )[0]
    assert all(pos not in set(fi.fit) | set(fi.inner_val) for fi in ctx.folds)


def test_inner_validation_is_last_20_percent_of_training_dates():
    panel = make_panel()
    prep = tr.prepare_panel(panel, SMALL)
    ctx = tr._context(prep, SMALL)
    fi = ctx.folds[0]
    tdates = np.unique(ctx.dates[np.concatenate([fi.fit, fi.inner_val])])
    n_val_dates = len(np.unique(ctx.dates[fi.inner_val]))
    total = len(np.unique(ctx.dates[np.concatenate([fi.fit, fi.inner_val])]))
    assert n_val_dates / (total + tr.INNER_GAP_SESSIONS) == pytest.approx(0.2, abs=0.05)
    assert tdates[-1] == ctx.dates[fi.inner_val].max()


# ----------------------------------------------------------------------------- selection


def _matrix(scores: list[list[float]], points=None) -> tr.ScoreMatrix:
    points = points or tr.GRID[: len(scores)]
    s = np.array(scores, dtype=float)
    return tr.ScoreMatrix(
        points=[dict(p) for p in points],
        fold_ids=list(range(1, s.shape[1] + 1)),
        scores=s,
        best_iter=np.full(s.shape, 50),
    )


def test_selection_picks_max_when_alone():
    scores = [[0.01] * 5 for _ in range(16)]
    scores[3] = [0.05] * 5  # zero spread -> SE 0 -> no tie
    sel = tr.select_point(_matrix(scores))
    assert sel["index"] == 3


def test_selection_tie_rule_prefers_regularized_within_one_se():
    # best point 0 (lr 0.03, leaves 15, msl 200, l2 0). fold SE = std/sqrt(5)
    base = [0.040, 0.050, 0.030, 0.045, 0.035]  # mean 0.040, std ~0.00791, SE ~0.00354
    scores = [[x - 0.05 for x in base] for _ in range(16)]
    scores[0] = [x + 0.0 for x in base]
    pts = list(tr.GRID)
    idx = {tuple(p.values()): i for i, p in enumerate(pts)}
    key = lambda lr, leaves, msl, l2: idx[(lr, leaves, msl, l2)]  # noqa: E731
    # inside one SE: l2=1 only, and msl=1000 only (l2=0); l2 must win over msl
    scores[key(0.1, 31, 200, 1.0)] = [x - 0.002 for x in base]
    scores[key(0.03, 15, 1000, 0.0)] = [x - 0.001 for x in base]
    sel = tr.select_point(_matrix(scores, pts))
    assert sel["point"] == {
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
    }
    # msl=1000 beats leaves=15 beats lr=0.03 when l2 is equal
    scores2 = [[x - 0.05 for x in base] for _ in range(16)]
    scores2[key(0.03, 15, 200, 0.0)] = list(base)
    scores2[key(0.1, 15, 200, 0.0)] = [x - 0.001 for x in base]
    scores2[key(0.03, 31, 1000, 0.0)] = [x - 0.001 for x in base]
    sel2 = tr.select_point(_matrix(scores2, pts))
    assert sel2["point"]["min_samples_leaf"] == 1000
    scores3 = [[x - 0.05 for x in base] for _ in range(16)]
    scores3[key(0.03, 31, 200, 0.0)] = list(base)
    scores3[key(0.1, 15, 200, 0.0)] = [x - 0.001 for x in base]
    sel3 = tr.select_point(_matrix(scores3, pts))
    assert sel3["point"]["max_leaf_nodes"] == 15
    scores4 = [[x - 0.05 for x in base] for _ in range(16)]
    scores4[key(0.1, 31, 200, 0.0)] = list(base)
    scores4[key(0.03, 31, 200, 0.0)] = [x - 0.001 for x in base]
    sel4 = tr.select_point(_matrix(scores4, pts))
    assert sel4["point"]["learning_rate"] == 0.03


def test_selection_outside_one_se_is_not_taken():
    base = [0.040, 0.050, 0.030, 0.045, 0.035]
    scores = [[x - 0.05 for x in base] for _ in range(16)]
    scores[0] = list(base)
    idx = {tuple(p.values()): i for i, p in enumerate(tr.GRID)}
    scores[idx[(0.1, 31, 1000, 1.0)]] = [x - 0.02 for x in base]  # far more regularized, too low
    assert tr.select_point(_matrix(scores, list(tr.GRID)))["index"] == 0


def test_final_max_iter_is_median_of_best_iterations():
    m = _matrix([[0.0] * 5])
    m.best_iter = np.array([[10, 30, 20, 100, 40]])
    assert tr.final_max_iter(m, 0) == 30


# --------------------------------------------------------------- CV / determinism / fits

POINTS2 = [tr.GRID[0], tr.GRID[-1]]


def test_cv_grid_shape_and_signal():
    panel = make_panel()
    m = tr.cv_grid(panel, SMALL, points=POINTS2)
    assert m.scores.shape == (2, 5)
    assert m.best_iter.shape == (2, 5)
    assert (m.best_iter >= 1).all() and (m.best_iter <= tr.MAX_ITER).all()
    assert np.nanmean(m.scores) > 0.1  # the fake signal is learnable
    d = m.to_dict()
    json.dumps(d)  # serializable


def test_cv_valid_label_end_max_limits_validation_rows():
    panel = make_panel()
    prep = tr.prepare_panel(panel, SMALL)
    cutoff = date(2017, 12, 29)
    ctx = tr._context(prep, SMALL, valid_label_end_max=cutoff)
    ends = prep["label_end_date"].to_numpy().astype("datetime64[D]")
    for fi in ctx.folds:
        assert (ends[fi.valid] <= np.datetime64(cutoff)).all()


def test_determinism_check_passes_across_seeds_0_to_4():
    panel = make_panel()
    out = tr.determinism_check(panel, tr.GRID[0], range(5), SMALL)
    assert out["identical"] is True
    assert out["seeds"] == [0, 1, 2, 3, 4]


def test_determinism_check_detects_a_seed_dependent_model(monkeypatch):
    panel = make_panel()
    real = tr._fit_early

    class Noisy:
        def __init__(self, m, seed):
            self.m, self.seed = m, seed
            self.validation_score_ = m.validation_score_

        def staged_predict(self, X):
            for p in self.m.staged_predict(X):
                yield p + self.seed * 1e-3

    def fake(ctx, fi, point, seed=0):
        m, best = real(ctx, fi, point, seed=seed)
        return Noisy(m, seed), best

    monkeypatch.setattr(tr, "_fit_early", fake)
    with pytest.raises(AssertionError):
        tr.determinism_check(panel, tr.GRID[0], range(3), SMALL)


def test_fit_final_uses_fixed_max_iter_without_early_stopping():
    panel = make_panel()
    model = tr.fit_final(panel, tr.GRID[0], 7, SMALL)
    assert model.n_iter_ == 7
    assert model.early_stopping is False
    model2 = tr.fit_final(panel, tr.GRID[0], 23, SMALL)
    assert model2.n_iter_ == 23


def test_fit_final_uses_only_trainable_rows_from_start_date():
    panel = make_panel()
    prep = tr.prepare_panel(panel, SMALL)
    rows = tr._final_rows(prep, date(2017, 6, 1), None)
    assert rows["trade_date"].min() >= date(2017, 6, 1)
    assert rows["trainable"].all() and rows["y"].null_count() == 0
    assert rows["label_end_date"].null_count() == 0


def test_null_mask_nulls_columns_where_flag_true_before_ranking():
    panel = make_panel(extra_flag=True)
    prep = tr.prepare_panel(panel, SMALL, null_mask=("ttm_includes_conflict", ["f_a", "f_b"]))
    flag = panel["ttm_includes_conflict"]
    assert prep.filter(flag)["f_a"].null_count() == flag.sum()
    assert prep.filter(flag)["f_b"].null_count() == flag.sum()
    assert prep.filter(~flag)["f_c"].null_count() == panel.filter(~flag)["f_c"].null_count()
    assert "ttm_includes_conflict" not in prep.columns
    with pytest.raises(ValueError):
        tr.prepare_panel(panel, SMALL, null_mask=("ttm_includes_conflict", ["zzz"]))
    # the refit path accepts the mask too
    tr.cv_point(panel, tr.GRID[0], SMALL, null_mask=("ttm_includes_conflict", ["f_a"]))


def test_fit_ridge_and_cv_ridge():
    panel = make_panel()
    r = tr.fit_ridge(panel, SMALL)
    assert r.alpha == 100.0
    scores = tr.cv_ridge(panel, SMALL)
    assert len(scores) == 5 and np.nanmean(scores) > 0.05


# ----------------------------------------------------------------------------- bundle


def _bundle_inputs(panel: pl.DataFrame, cols: list[str]):
    prep = tr.prepare_panel(panel, cols)
    model = tr.fit_final(prep, tr.GRID[0], 10, cols, prepared=True)
    ridge = tr.fit_ridge(prep, cols, prepared=True)
    return prep, model, ridge


def _write(out: Path, prep, model, ridge, cols) -> Path:
    return bundle_mod.write_bundle(
        out,
        model=model,
        ridge=ridge,
        ridge_alpha=100.0,
        feature_cols=cols,
        golden_frame=prep.filter(pl.col("y").is_not_null()),
        params=dict(tr.GRID[0]),
        max_iter=10,
        score_matrix={"scores": [[0.1]]},
        determinism={"identical": True},
        snapshot_id="snap-test",
        prereg_tag="kr-next-frozen",
        prereg_commit="abc123",
        modeler_commit="def456",
        prereg_doc_sha256="0" * 64,
        preprocessing={"rank": "x"},
    )


def test_bundle_round_trip_and_manifest_contents(tmp_path):
    panel = make_panel()
    prep, model, ridge = _bundle_inputs(panel, SMALL)
    out = _write(tmp_path / "b", prep, model, ridge, SMALL)
    assert {p.name for p in out.iterdir()} == {
        "model.joblib",
        "ridge.joblib",
        "golden_inputs.parquet",
        "golden_preds.parquet",
        "manifest.json",
    }
    b = bundle_mod.load_bundle(out)
    man = b.manifest
    assert man["feature_cols"] == SMALL
    assert man["feature_list_sha256"] == bundle_mod.feature_list_hash(SMALL)
    assert man["max_iter"] == 10 and man["snapshot_id"] == "snap-test"
    assert man["prereg_tag"] == "kr-next-frozen" and man["prereg_commit"] == "abc123"
    assert man["golden_rows"] == 200
    assert set(man["libraries"]) >= {"scikit-learn", "polars", "duckdb", "numpy"}
    assert set(man["files"]) == {
        "model.joblib",
        "ridge.joblib",
        "golden_inputs.parquet",
        "golden_preds.parquet",
    }
    gi = pl.read_parquet(out / "golden_inputs.parquet")
    assert gi.height == 200
    np.testing.assert_allclose(
        b.predict(gi), pl.read_parquet(out / "golden_preds.parquet")["score"]
    )
    assert b.predict_ridge(gi).shape == (200,)


def test_bundle_detects_hash_and_golden_mismatch(tmp_path):
    panel = make_panel()
    prep, model, ridge = _bundle_inputs(panel, SMALL)
    out = _write(tmp_path / "b", prep, model, ridge, SMALL)

    # 1) altered model file -> sha256 mismatch even without golden verification
    bad = tmp_path / "bad_model"
    import shutil

    shutil.copytree(out, bad)
    with (bad / "model.joblib").open("ab") as f:
        f.write(b"x")
    with pytest.raises(bundle_mod.BundleError, match="sha256"):
        bundle_mod.load_bundle(bad, verify_golden=False)

    # 2) golden preds changed AND the manifest hash updated -> only the golden check can catch it
    bad2 = tmp_path / "bad_golden"
    shutil.copytree(out, bad2)
    gp = pl.read_parquet(bad2 / "golden_preds.parquet").with_columns(pl.col("score") + 0.01)
    gp.write_parquet(bad2 / "golden_preds.parquet")
    man = json.loads((bad2 / "manifest.json").read_text())
    man["files"]["golden_preds.parquet"] = bundle_mod.sha256_file(bad2 / "golden_preds.parquet")
    (bad2 / "manifest.json").write_text(json.dumps(man))
    bundle_mod.load_bundle(bad2, verify_golden=False)  # hashes fine
    with pytest.raises(bundle_mod.BundleError, match="golden"):
        bundle_mod.load_bundle(bad2, verify_golden=True)


def test_bundle_is_write_once(tmp_path):
    panel = make_panel()
    prep, model, ridge = _bundle_inputs(panel, SMALL)
    out = _write(tmp_path / "b", prep, model, ridge, SMALL)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(bundle_mod.BundleError, match="exists"):
        _write(out, prep, model, ridge, SMALL)
    assert (out / "manifest.json").read_bytes() == before
    # no stray temp dirs left behind
    assert [p.name for p in tmp_path.iterdir()] == ["b"]


# ----------------------------------------------------------------------------- CLI guard / e2e


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def _prereg_repo(path: Path, tag: bool) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    (path / "doc.md").write_text("prereg")
    _git(path, "add", "doc.md")
    _git(path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    if tag:
        _git(path, "-c", "tag.gpgsign=false", "tag", "kr-next-frozen")
    return path


def test_cli_refuses_without_tag_and_reads_nothing(tmp_path, capsys):
    repo = _prereg_repo(tmp_path / "prereg", tag=False)
    out = tmp_path / "out"
    rc = tr.main(
        [
            "--lake",
            str(tmp_path / "no_lake"),
            "--snapshot",
            "s",
            "--out",
            str(out),
            "--prereg-repo",
            str(repo),
        ]
    )
    assert rc == 2
    assert not out.exists()
    assert "kr-next-frozen" in capsys.readouterr().err
    # a similarly named but different tag does not count
    _git(repo, "-c", "tag.gpgsign=false", "tag", "kr-next-frozen-x")
    assert (
        tr.main(["--lake", "x", "--snapshot", "s", "--out", str(out), "--prereg-repo", str(repo)])
        == 2
    )
    with pytest.raises(tr.FrozenTagMissing):
        tr.require_frozen_tag(repo)


def test_cli_refuses_missing_repo_and_existing_out(tmp_path):
    out = tmp_path / "out"
    assert (
        tr.main(
            [
                "--lake",
                "x",
                "--snapshot",
                "s",
                "--out",
                str(out),
                "--prereg-repo",
                str(tmp_path / "nope"),
            ]
        )
        == 2
    )
    repo = _prereg_repo(tmp_path / "prereg", tag=True)
    out.mkdir()
    (out / "keep").write_text("1")
    assert (
        tr.main(["--lake", "x", "--snapshot", "s", "--out", str(out), "--prereg-repo", str(repo)])
        == 2
    )
    assert [p.name for p in out.iterdir()] == ["keep"]


def test_require_frozen_tag_returns_tag_commit(tmp_path):
    repo = _prereg_repo(tmp_path / "prereg", tag=True)
    assert tr.require_frozen_tag(repo) == _git(repo, "rev-parse", "HEAD")


def test_cli_end_to_end_on_synthetic_panel(tmp_path):
    repo = _prereg_repo(tmp_path / "prereg", tag=True)
    cols = list(fs.NOFLOW_COLS)
    assert len(cols) == 30
    panel = make_panel(cols=cols, label_name="label_end_date_c1_20d", extra_flag=True, n_dates=230)
    pp = tmp_path / "panel.parquet"
    panel.write_parquet(pp)
    doc = tmp_path / "prereg.md"
    doc.write_text("doc")
    out = tmp_path / "bundle"
    rc = tr.main(
        [
            "--lake",
            str(tmp_path),
            "--snapshot",
            "snap-synth",
            "--out",
            str(out),
            "--prereg-repo",
            str(repo),
            "--panel",
            str(pp),
            "--prereg-doc",
            str(doc),
            "--r1-sensitivity",
        ]
    )
    assert rc == 0
    b = bundle_mod.load_bundle(out)
    m = b.manifest
    assert m["prereg_commit"] == _git(repo, "rev-parse", "HEAD")
    assert m["prereg_tag"] == "kr-next-frozen"
    assert m["feature_cols"] == cols
    assert m["snapshot_id"] == "snap-synth"
    assert len(m["score_matrix"]["points"]) == 16
    assert np.array(m["score_matrix"]["scores"]).shape == (16, 5)
    assert m["determinism"]["identical"] is True
    assert m["prereg_doc_sha256"] and len(m["prereg_doc_sha256"]) == 64
    assert m["trials"] == {"hgb_grid_points": 16, "ridge": 1}
    assert "r1_sensitivity" in m and len(m["r1_sensitivity"]["fold_scores"]) == 5
    assert b.model.n_iter_ == m["max_iter"]
    # second run into the same --out is refused
    assert (
        tr.main(
            [
                "--lake",
                str(tmp_path),
                "--snapshot",
                "s",
                "--out",
                str(out),
                "--prereg-repo",
                str(repo),
                "--panel",
                str(pp),
            ]
        )
        == 2
    )
