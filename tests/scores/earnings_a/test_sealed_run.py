"""Synthetic tests for the one-shot sealed runner (no real data, never the real ledger)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import cli, seal, sealed_run, train
from modeler.scores.earnings_a.paths import DataPaths
from modeler.scores.earnings_a.spec import FEATURES

ALL38 = tuple(seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES)
HGB_SPEC = train.ModelSpec(
    "HGB",
    {
        "learning_rate": 0.1,
        "max_iter": 5,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 5,
        "early_stopping": False,
        "loss": "absolute_error",
    },
)
RIDGE_SPEC = train.ModelSpec("RIDGE", {"alpha": 10.0})
N = 40


def make_feats(dates=ALL38, n=N, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for i in range(n):
            f = {c: rng.normal() for c in FEATURES}
            rows.append(
                {
                    "decision_date": d,
                    "ticker": f"T{i:03d}0",
                    "corp_code": f"C{i:03d}",
                    "basis": "CFS",
                    "ksic2": str(10 + i % 5),
                    "a_t": float(np.exp(f["log_assets"] + 10)),
                    "a_t_dec": "1",
                    "fye_month": 12,
                    "p6_tradable": i % 2 == 0,
                    "is_holding": i == 3,
                    "ocf_ttm": f["ocf_ttm"],
                    **{k: v for k, v in f.items() if k != "ocf_ttm"},
                    "input_eligible": i != 3,
                    "input_eligible_incl_holding": True,
                }
            )
    return pd.DataFrame(rows)


def synth_labels(feats, dates, target):
    sub = feats.loc[feats["decision_date"].isin(list(dates))]
    rng = np.random.default_rng(1 if target == "oi_y1" else 2)
    y = sub["oi_ttm"].to_numpy() + rng.normal(scale=0.5, size=len(sub))
    ok = (np.arange(len(sub)) % 7) != 0
    lab = sub[["decision_date", "ticker", "corp_code", "basis"]].copy()
    lab["label_ok"] = ok
    lab["y"] = np.where(ok, y * 3, np.nan)
    lab["y_clip"] = np.where(ok, y, np.nan)
    lab["label_pub_date"] = [d + timedelta(days=30) for d in lab["decision_date"]]
    lab["missing_kind"] = np.where(ok, None, "NO_ROW")
    return lab


class Env:
    """Patches every seam; records the order of calls."""

    def __init__(self, tmp_path, monkeypatch, dirty=False, f4_ok=True):
        self.log: list = []
        self.label_dates: list = []
        self.opened = 0
        self.tmp = tmp_path
        feats = make_feats()
        self.feats = feats
        self.paths = DataPaths(tmp_path, tmp_path, tmp_path / "a", tmp_path / "b", tmp_path / "out")
        self.paths.out_root.mkdir()
        chosen = {
            "HGB": {
                "name": "HGB",
                "params": HGB_SPEC.params,
                "spec_id": HGB_SPEC.spec_id,
            },
            "RIDGE": {"name": "RIDGE", "params": RIDGE_SPEC.params, "spec_id": RIDGE_SPEC.spec_id},
        }
        self.chosen_path = tmp_path / "chosen.json"
        self.chosen_path.write_text(json.dumps(chosen))
        self.chosen_sha = sealed_run._sha(self.chosen_path)
        self.f4_path = tmp_path / "f4.json"
        self.f4_path.write_text("{}")
        self.ledger = self.paths.out_root / "seal_ledger.jsonl"
        mp = monkeypatch
        mp.setattr(sealed_run, "HGB_GRID", (HGB_SPEC.params,))
        mp.setattr(sealed_run, "RIDGE_ALPHAS", (10.0,))
        mp.setattr(
            sealed_run,
            "_git_state",
            lambda: (
                self._log("git") or {"head": "h", "branch": "b", "describe": "d", "dirty": dirty}
            ),
        )
        mp.setattr(sealed_run, "_verify_inputs", lambda p: self._log("hashes"))
        mp.setattr(sealed_run, "_load_chosen", self._chosen(sealed_run._load_chosen))
        mp.setattr(sealed_run, "_label_free", self._label_free)
        mp.setattr(sealed_run, "_f4_check", self._f4(f4_ok))
        mp.setattr(sealed_run, "_load_labels", self._labels)
        mp.setattr(sealed_run, "_reason_table", lambda ctx: self._reasons())
        orig = seal.open_seal_grant

        def spy(*a, **k):
            self._log("open_seal")
            self.opened += 1
            return orig(*a, **k)

        mp.setattr(seal, "open_seal_grant", spy)
        mp.setattr(sealed_run.manifest, "describe_inputs", lambda paths, **k: {})

    def _log(self, name):
        self.log.append(name)

    def _chosen(self, real):
        def f(path, sha):
            self._log("chosen")
            return real(path, sha)

        return f

    def _label_free(self, paths):
        self._log("label_free")
        return {
            "feats": self.feats,
            "receipts": pd.DataFrame(),
            "stock_master": pd.DataFrame(),
            "last_trade": pd.DataFrame(),
        }

    def _f4(self, ok):
        def f(paths, lf, mode, path):
            self._log("f4")
            if not ok:
                raise sealed_run.SealedAbort("F4 digest differs")
            return {"match": True}

        return f

    def _labels(self, paths, feats, receipts, dates, target, grant):
        self._log(f"labels:{target}")
        self.label_dates.append((target, list(dates)))
        return synth_labels(feats, dates, target), pd.DataFrame()

    def _reasons(self):
        return pd.DataFrame(
            {
                "decision_date": [seal.DEV_DATES[0]],
                "reason_code": ["UNCONFIRMED"],
                "reason_detail": [None],
            }
        )

    def git_runner(self):
        def run(args):
            if args[0] == "tag":
                return 0, "earnings-a-frozen\n"
            if args[0] == "cat-file":
                return 0, "tag\n"
            return 0, "abc123\n"

        return run

    def run(self, dry):
        mode = sealed_run.make_mode(dry)
        return sealed_run.run_sealed(
            self.paths,
            sealed_run.make_out_dir(self.paths, mode, None),
            dry_run_dev=dry,
            ledger_path=self.ledger,
            repo_dir=self.tmp,
            git_runner=self.git_runner(),
            chosen_path=self.chosen_path,
            chosen_sha256=self.chosen_sha,
            f4_manifest_path=self.f4_path,
        )


def test_modes():
    r, d = sealed_run.make_mode(False), sealed_run.make_mode(True)
    assert r.eval_dates == seal.SEALED_DATES and len(r.decision_dates) == 38
    assert d.eval_dates == seal.DEV_DATES[-9:] and set(d.decision_dates) == set(seal.DEV_DATES)
    assert r.y2_eval == tuple(x for x in seal.SEALED_DATES if x <= date(2024, 6, 30))
    assert len(r.y2_eval) == 5
    assert not set(r.y2_dates) & set(seal.GAP_DATES)
    assert max(x for x in r.y2_dates if x in seal.DEV_DATES) == date(2021, 6, 30)
    assert d.y2_eval == tuple(x for x in d.eval_dates if x <= date(2021, 6, 30))
    assert d.kind == "SEALED_DRYRUN_DEV" and r.kind == "SEALED_VERDICT"


def test_ksic_section():
    assert ksic_ok("10", "C") and ksic_ok("26", "C") and ksic_ok("62", "J")
    assert ksic_ok("64", "K") and ksic_ok(None, "NA") and ksic_ok("04", "NA")


def ksic_ok(v, want):
    return sealed_run.ksic_section(v) == want


def test_real_run_order_and_single_open(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    out_dir, mpath = env.run(dry=False)
    i = env.log.index("open_seal")
    for step in ("git", "hashes", "chosen", "label_free", "f4"):
        assert env.log.index(step) < i
    assert env.log.count("open_seal") == 1 and env.opened == 1
    assert all(not s.startswith("labels:") for s in env.log[:i])
    ledger = [json.loads(x) for x in env.ledger.read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]["phase"] == "SEALED_RUN"
    v = json.loads((out_dir / "verdict.json").read_text())
    assert v["n_dates"] == 9 and v["outcome"] in {"H1_FAIL", "H1_PASS_H2_FAIL", "H1_PASS_H2_PASS"}
    assert v["not_a_verdict"] is False
    m = json.loads(mpath.read_text())
    assert m["status"] == "complete" and m["ledger_record"]["phase"] == "SEALED_RUN"
    assert m["sections"]["judge_matches_h1_h2"] is True
    bad = {k: s for k, s in m["sections"].items() if isinstance(s, dict) and s["status"] != "ok"}
    assert not bad, bad
    for name in (
        "record_per_date.tsv",
        "record_by_group.tsv",
        "record_top1.tsv",
        "record_spearman.tsv",
        "record_ind.tsv",
        "record_unclipped.json",
        "record_p6_tradable.json",
        "record_holding.json",
        "record_y2.json",
        "record_ocf.json",
        "record_completeness.tsv",
        "record_reasons.tsv",
    ):
        assert (out_dir / name).is_file(), name
    # y2 never asks for GAP dates or SEALED dates past 2024-06-30
    y2 = [d for t, d in env.label_dates if t == "oi_y2"][0]
    assert not set(y2) & set(seal.GAP_DATES) and max(y2) <= date(2024, 6, 30)
    # a second run cannot open again
    with pytest.raises(seal.SealViolation):
        env.run(dry=False)
    assert env.opened == 2  # attempted, refused by the ledger; still one ledger line
    assert len(env.ledger.read_text().splitlines()) == 1


def test_dry_run_never_opens_and_never_requests_sealed(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, dirty=True)  # dirty allowed in dry run
    out_dir, mpath = env.run(dry=True)
    assert env.opened == 0 and "open_seal" not in env.log
    assert not env.ledger.exists()
    allowed = set(seal.DEV_DATES)
    assert env.label_dates
    for _, dates in env.label_dates:
        assert set(dates) <= allowed
    m = json.loads(mpath.read_text())
    assert m["kind"] == "SEALED_DRYRUN_DEV" and m["dirty_recorded"] is True
    assert m["status"] == "complete"
    v = json.loads((out_dir / "verdict.json").read_text())
    assert v["not_a_verdict"] is True
    assert v["eval_dates"][0] == "2020-06-30" and v["eval_dates"][-1] == "2022-06-30"
    assert (out_dir / "record_y2.json").is_file()
    assert out_dir.name.startswith("sealed_dryrun_dev_")


def test_dirty_tree_refused_before_anything(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, dirty=True)
    with pytest.raises(SystemExit):
        env.run(dry=False)
    assert env.log == ["git"] and env.opened == 0


def test_f4_mismatch_aborts_before_open(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, f4_ok=False)
    with pytest.raises(sealed_run.SealedAbort):
        env.run(dry=False)
    assert env.opened == 0 and "open_seal" not in env.log
    assert not env.ledger.exists()
    assert not list(env.paths.out_root.glob("sealed_*"))  # no output dir either


def test_chosen_hash_mismatch_aborts_before_open(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.chosen_sha = "0" * 64
    with pytest.raises(sealed_run.SealedAbort):
        env.run(dry=False)
    assert env.opened == 0 and "f4" not in env.log and "label_free" not in env.log


def test_chosen_spec_outside_grid_aborts(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    monkeypatch.setattr(sealed_run, "RIDGE_ALPHAS", (1.0,))
    with pytest.raises(sealed_run.SealedAbort):
        env.run(dry=False)
    assert env.opened == 0


def test_f4_check_compares_exact_float_and_rows(tmp_path, monkeypatch):
    mode = sealed_run.make_mode(False)
    exp = {"exact_sha256": "a" * 64, "float_sha256": "b" * 64, "n_rows": 5}
    f4 = tmp_path / "f4.json"
    f4.write_text(json.dumps({"digests": {"SEALED.oi_y1": exp}, "code": {}}))
    seen = {}

    def fake(paths, feats, receipts, dates, target):
        seen["dates"], seen["target"] = dates, target
        return dict(exp)

    monkeypatch.setattr(sealed_run.digest, "_label_digest", fake)
    lf = {"feats": None, "receipts": None}
    out = sealed_run._f4_check(None, lf, mode, f4)
    assert out["match"] and seen["target"] == "oi_y1" and seen["dates"] == seal.SEALED_DATES
    for k, v in (("exact_sha256", "c" * 64), ("float_sha256", "d" * 64), ("n_rows", 6)):
        monkeypatch.setattr(sealed_run.digest, "_label_digest", lambda *a, k=k, v=v: {**exp, k: v})
        with pytest.raises(sealed_run.SealedAbort):
            sealed_run._f4_check(None, lf, mode, f4)


def test_record_error_does_not_stop_verdict(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    orig = sealed_run._load_labels

    def flaky(paths, feats, receipts, dates, target, grant):
        if target in ("ocf_y1", "oi_y2"):
            raise ValueError("boom 123.456")
        return orig(paths, feats, receipts, dates, target, grant)

    monkeypatch.setattr(sealed_run, "_load_labels", flaky)
    monkeypatch.setattr(sealed_run, "rec_by_group", lambda ctx: 1 / 0)
    monkeypatch.setattr(
        sealed_run,
        "RECORD_ITEMS",
        tuple(
            (n, (sealed_run.rec_by_group if n == "by_group" else f))
            for n, f in sealed_run.RECORD_ITEMS
        ),
    )
    out_dir, mpath = env.run(dry=False)
    assert (out_dir / "verdict.json").is_file()
    m = json.loads(mpath.read_text())
    s = m["sections"]
    assert s["primary"]["status"] == "ok"
    assert s["ocf"]["status"] == "error" and "ValueError" in s["ocf"]["error"]
    assert s["y2"]["status"] == "error" and s["by_group"]["status"] == "error"
    assert s["holding"]["status"] == "ok" and s["reasons"]["status"] == "ok"
    assert m["status"] == "complete"


def test_verdict_written_before_records(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    seen = {}

    def rec(ctx):
        seen["verdict_exists"] = (ctx.out_dir / "verdict.json").is_file()
        return {}

    monkeypatch.setattr(sealed_run, "RECORD_ITEMS", (("probe", rec),))
    env.run(dry=False)
    assert seen["verdict_exists"] is True


def test_primary_failure_after_open_writes_failed_marker(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("primary broke")

    monkeypatch.setattr(sealed_run.evaluate, "h1_h2", boom)
    with pytest.raises(RuntimeError):
        env.run(dry=False)
    d = next(env.paths.out_root.glob("sealed_*"))
    assert (d / "verdict_FAILED.json").is_file() and (d / "manifest_sealed.json").is_file()
    assert not (d / "verdict.json").exists()


def test_three_predictors_same_rows(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    grant = seal.grant_dev()
    ds = sealed_run.build_dataset(
        env.feats, synth_labels(env.feats, seal.DEV_DATES, "oi_y1"), seal.DEV_DATES
    )
    pred = train.walk_forward(
        ds,
        eval_dates=list(seal.DEV_DATES[-3:]),
        specs=[train.ModelSpec("PERS"), RIDGE_SPEC, HGB_SPEC],
        decision_dates=list(seal.DEV_DATES),
        grant=grant,
    )
    j = sealed_run.join_labels(pred, ds)
    sealed_run.assert_same_rows(j)
    bad = j.drop(index=j.index[(j["model"] == "HGB")][0])
    with pytest.raises(AssertionError):
        sealed_run.assert_same_rows(bad)
    nan = j.copy()
    idx = nan.index[(nan["model"] == "RIDGE") & nan["label_ok"]][0]
    nan.loc[idx, "y_hat"] = np.nan
    with pytest.raises(AssertionError):
        sealed_run.assert_same_rows(nan)


def test_holding_variant_adds_rows(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    out_dir, _ = env.run(dry=True)
    base = json.loads((out_dir / "verdict.json").read_text())["n_rows"]
    h = json.loads((out_dir / "record_holding.json").read_text())
    assert h["n_rows"] > base and h["extra_rows_all_dates"] > 0
    p6 = json.loads((out_dir / "record_p6_tradable.json").read_text())
    assert 0 < p6["n_rows"] < base


def test_cli_wiring(tmp_path, monkeypatch):
    seen = {}

    def fake(paths, make_out, *, dry_run_dev):
        seen["dry"] = dry_run_dev
        return Path("o"), Path("m")

    monkeypatch.setattr(sealed_run, "run_sealed", fake)
    monkeypatch.setattr(cli.P.DataPaths, "resolve", classmethod(lambda cls, env=None: None))
    assert cli.main(["sealed", "--dry-run-dev"]) == 0
    assert seen["dry"] is True
    assert cli.main(["sealed"]) == 0 and seen["dry"] is False
