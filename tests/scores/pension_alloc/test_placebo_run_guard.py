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
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "a@b.c")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-qm", "c")
    return tmp_path


def test_guard_requires_tag_and_clean_tree(tmp_path):
    r = _repo(tmp_path)
    with pytest.raises(guard.NotFrozenError, match="태그"):
        guard.require_frozen_tag(r)
    _git(r, "tag", guard.FROZEN_TAG)
    sha = guard.require_frozen_tag(r)
    assert len(sha) == 40
    (r / "f.txt").write_text("changed")
    with pytest.raises(guard.NotFrozenError, match="미커밋"):
        guard.require_frozen_tag(r)
    # 태그가 HEAD가 아닌 옛 커밋에 있으면 실패
    _git(r, "commit", "-qam", "d")
    with pytest.raises(guard.NotFrozenError):
        guard.require_frozen_tag(r)


def test_main_refuses_before_touching_data(monkeypatch):
    import modeler.scores.etf_baseline.run as base_run

    def boom(*a, **k):
        raise AssertionError("실제 자료를 읽었습니다")

    monkeypatch.setattr(base_run, "prepare", boom)
    monkeypatch.setattr(guard, "_repo_root", lambda: __import__("pathlib").Path("/nonexistent-x"))
    with pytest.raises(guard.NotFrozenError):
        RUN.main()
