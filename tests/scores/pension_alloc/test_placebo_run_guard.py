import subprocess

import numpy as np
import pytest

from modeler.scores.etf_baseline import lines as L
from modeler.scores.etf_baseline.ledger import simulate
from modeler.scores.pension_alloc import guard
from modeler.scores.pension_alloc import placebo as P
from modeler.scores.pension_alloc import run as RUN
from tests.scores.pension_alloc._synth import bdays, month_ends


def test_circular_shifts_bounds():
    s = P.circular_shifts(24)
    assert s[0] == 6 and s[-1] == 18 and len(s) == 13
    assert P.circular_shifts(11) == []


def test_pvalue_formula():
    assert P.placebo_pvalue(1.0, [0.5, 1.0, 2.0, 0.1]) == pytest.approx((1 + 2) / (1 + 4))
    assert P.placebo_pvalue(5.0, [1.0, 2.0]) == pytest.approx(1 / 3)
    assert P.placebo_pvalue(float("nan"), [1.0]) == 1.0
    assert P.placebo_pvalue(1.0, [float("nan"), 2.0]) == pytest.approx(2 / 3)


def test_placebo_calmar_uses_all_shifts():
    n = 24
    w = np.arange(n, dtype=float)[:, None]
    calls = []

    def fn(path):
        calls.append(path[0, 0])
        return float(path[0, 0])

    out = P.placebo_calmar(w, fn)
    assert out["n_shifts"] == 13 and len(calls) == 14
    # np.roll(path, s)[0] = w[n-s]
    assert sorted(calls[1:]) == sorted(float(n - s) for s in range(6, 19))
    actual = 0.0
    wins = sum(1 for v in out["shifted"] if v >= actual)
    assert out["p"] == pytest.approx((1 + wins) / 14)


def test_holm():
    out = P.holm({"JM": 0.01, "SP": 0.04})
    assert out["JM"]["adj_p"] == pytest.approx(0.02) and out["JM"]["reject"]
    assert out["SP"]["adj_p"] == pytest.approx(0.04) and out["SP"]["reject"]
    out = P.holm({"JM": 0.03, "SP": 0.04})
    assert out["JM"]["adj_p"] == pytest.approx(0.06) and not out["JM"]["reject"]
    assert out["SP"]["adj_p"] == pytest.approx(0.06) and not out["SP"]["reject"]  # 단조
    out = P.holm({"JM": 0.06, "SP": 0.001})
    assert out["SP"]["reject"] and not out["JM"]["reject"]


def test_cost_accounting_matches_baseline_ledger():
    # 자산 하나 + 현금. 0일에 [1,0] 시작(비용 없음), 3일에 [0.5,0.5] -> 4일 종가에 회전 0.5
    T = 8
    ret = np.zeros((T, 1))
    ret[2:, 0] = 0.01
    cash = np.full(T, 0.0)
    cash[0] = np.nan
    ow = 0.0030
    decisions = [0, 3]
    W = np.array([[1.0, 0.0], [0.5, 0.5]])
    a = RUN.run_ledger(ret, cash, decisions, W, ow, T - 1)
    b = simulate(ret, cash, 1, T - 1, {1: W[0], 4: W[1]}, ow)
    assert np.array_equal(a.nav, b.nav)
    # 손계산: 4일 종가까지 1.01^3 (2·3·4일), 회전 |0.5 − 1|=0.5 -> 비용 0.5·ow
    assert a.rebalances[-1]["turnover"] == pytest.approx(0.5)
    assert a.nav[2] == pytest.approx(1.01**3 * (1 - 0.5 * ow))


def test_jm_weights_use_b4_skeleton():
    spec = L.line_specs(L.US_MAIN)["B4"]
    w = spec.weights(0.5)
    assert np.allclose(w, [0.15, 0.15, 0.4, 0.3])


def test_evaluate_candidate_smoke_and_b5r():
    T = 400
    days = bdays(T)
    rng = np.random.default_rng(0)
    ret = 0.0004 + 0.01 * rng.standard_normal((T, 3))
    cash = np.full(T, 0.0001)
    cash[0] = np.nan
    dec = [i for i in month_ends(days) if i >= 10][:-1]
    spec = L.line_specs(L.US_MAIN)["B4"]
    m = rng.choice([0.5, 1.0], len(dec))
    W = np.array([spec.weights(v) for v in m])
    out = RUN.evaluate_candidate(days, ret, cash, dec, W, dec[-1] + 1)
    for c in ("low", "high"):
        o = out[c]
        assert o["b5r_weights"].sum() == pytest.approx(1.0)
        assert 0.0 <= o["placebo"]["p"] <= 1.0
        assert o["placebo"]["n_shifts"] == len(dec) - 11
    # 비용이 높으면 후보의 종가 NAV가 낮다
    assert out["high"]["metrics"]["terminal_nav"] < out["low"]["metrics"]["terminal_nav"]
    # 상수 비중 후보는 모든 이동이 같은 값이라 p=1
    const = np.tile(W[0], (len(dec), 1))
    o2 = RUN.evaluate_candidate(days, ret, cash, dec, const, dec[-1] + 1)
    assert o2["low"]["placebo"]["p"] == 1.0


# ------------------------------------------------------------------ 가드
def _git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)


def _repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "a@b.c")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-qm", "c")
    return tmp_path


def test_guard_requires_both_tags_and_clean_modeler_tree(tmp_path):
    mod = _repo(tmp_path / "mod")
    my = _repo(tmp_path / "my")
    with pytest.raises(guard.NotFrozenError, match="modeler.*태그"):
        guard.require_frozen_tag(mod, my)
    _git(mod, "tag", guard.FROZEN_TAG)
    with pytest.raises(guard.NotFrozenError, match="my.*태그"):
        guard.require_frozen_tag(mod, my)
    _git(my, "tag", guard.FROZEN_TAG)
    out = guard.require_frozen_tag(mod, my)
    assert len(out["code_sha"]) == 40 and out["my_tag"]["commit"] != ""
    assert out["modeler_tag"]["tag_object"] and out["my_tag"]["tag_object"]
    (mod / "f.txt").write_text("changed")
    with pytest.raises(guard.NotFrozenError, match="미커밋"):
        guard.require_frozen_tag(mod, my)
    # 태그가 HEAD가 아닌 옛 커밋에 있으면 실패
    _git(mod, "commit", "-qam", "d")
    with pytest.raises(guard.NotFrozenError):
        guard.require_frozen_tag(mod, my)
    # my에 새 커밋이 생겨도 실패
    _git(mod, "tag", "-f", guard.FROZEN_TAG)
    guard.require_frozen_tag(mod, my)
    (my / "f.txt").write_text("y")
    _git(my, "commit", "-qam", "e")
    with pytest.raises(guard.NotFrozenError, match="my"):
        guard.require_frozen_tag(mod, my)


def test_main_refuses_before_touching_data(monkeypatch):
    import pathlib

    import modeler.scores.etf_baseline.run as base_run

    def boom(*a, **k):
        raise AssertionError("실제 자료를 읽었습니다")

    monkeypatch.setattr(base_run, "prepare", boom)
    monkeypatch.setattr(RUN.D, "load_inputs", boom)
    monkeypatch.setattr(RUN.D, "schema_only", boom)
    monkeypatch.setattr(guard, "_repo_root", lambda: pathlib.Path("/nonexistent-x"))
    with pytest.raises(guard.NotFrozenError):
        RUN.main([])


def test_schema_only_runs_without_guard(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("가드를 불렀습니다")

    monkeypatch.setattr(guard, "require_frozen_tag", boom)
    monkeypatch.setattr(RUN.D, "schema_only", lambda root: "SCHEMA")
    monkeypatch.setattr(RUN.I, "stock_data_root", lambda: None)
    assert RUN.main(["--schema-only"]) == 0
    assert capsys.readouterr().out.strip() == "SCHEMA"


def test_main_calls_guard_first_then_loads(monkeypatch):
    order = []

    def g():
        order.append("guard")
        return {"code_sha": "x"}

    def load(root):
        order.append("load")
        raise RuntimeError("stop")

    monkeypatch.setattr(guard, "require_frozen_tag", g)
    monkeypatch.setattr(RUN.D, "load_inputs", load)
    monkeypatch.setattr(RUN.I, "stock_data_root", lambda: None)
    with pytest.raises(RuntimeError):
        RUN.main([])
    assert order == ["guard", "load"]


def test_decided_constants_and_config():
    assert RUN.LAMBDAS == (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
    assert RUN.GAMMAS == (0.001, 0.01, 0.1)
    assert RUN.JM_SEEDS == tuple(range(20261018, 20261028)) and RUN.JM_MAX_ITER == 100
    cfg = RUN.jm_config()
    assert cfg.feature_warmup == 252 and cfg.label_rule == "compound"
    assert cfg.lambdas == RUN.LAMBDAS and cfg.window_len == 3000 and cfg.val_months == 96
    import os

    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_cash_returns_forward_fill_rate_with_staleness_limit():
    from datetime import date

    import polars as pl

    from modeler.scores.pension_alloc import data as D

    days = [
        date(2020, 1, 1) + __import__("datetime").timedelta(days=k) for k in (0, 1, 2, 5, 6, 30, 31)
    ]
    rates = pl.DataFrame(
        {"date": [date(2019, 12, 31)], "realtime_start": [date(2019, 12, 31)], "value": [3.65]}
    )
    r = D.cash_returns(rates, days)
    assert np.isnan(r[0])
    # 금리 3.65%/365 = 0.01%/일, 달력일 수로 복리: 0->1 하루, 2->5 사흘 (앞 값으로 채움)
    assert r[1] == pytest.approx(1e-4 * 1) and r[3] == pytest.approx(1e-4 * 3)
    # 구간 시작 세션의 관측이 14일 넘게 오래되면 NaN: 1월 7일(7일)은 되고 1월 31일(31일)은 안 된다
    assert r[5] == pytest.approx(1e-4 * 24) and np.isnan(r[6])
