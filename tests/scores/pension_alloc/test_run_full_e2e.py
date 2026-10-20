"""``run.run_full`` 끝까지 시험 (사전등록 §10.5 P1). 합성 자료만 쓴다.

실제 자료 첫 실행이 곧 진단 결과라서, 동결 전에 파일 묶음(입력 파일·출력 폴더)을 임시 폴더에
만들어 ``run_full``을 처음부터 끝까지 돌린다. 동결 상수 F1~F5(λ 격자·시드·반복 상한·예열)는
코드에서 바꾸지 않고, 시험에서만 ``jm_config``·격자를 줄인다(``test_frozen_constants``가 진짜
상수를 따로 확인한다). 줄이는 것은 JM 창 길이·검증 월 수·λ 점·시드 수뿐이다.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import lines as L
from modeler.scores.pension_alloc import data as D
from modeler.scores.pension_alloc import run as RUN
from modeler.scores.pension_alloc.jump_model import JMConfig, JumpModelRule
from tests.scores.pension_alloc._synth import bdays, slot_returns, two_regime

N_DAYS = 2150  # 약 8.5년
LISTING = (0, 20, 0, 50, 10)  # ETF 슬롯별 상장 세션(그 전은 NaN)
T_LAMBDAS = (0.0, 15.0, 150.0)
T_SEEDS = (11, 12, 13)
T_GAMMAS = (0.001, 0.01, 0.1)
T_WINDOW = 300
T_VAL_MONTHS = 24


def _small_jm_config() -> JMConfig:
    """진입점 설정(F3·F4·F5)에서 창·검증 월 수·λ 점·시드 수만 줄인 시험용."""
    return JMConfig(
        lambdas=T_LAMBDAS,
        seeds=T_SEEDS,
        max_iter=RUN.JM_MAX_ITER,
        feature_warmup=RUN.JM_FEATURE_WARMUP,  # 252, F4 그대로
        label_rule=RUN.JM_LABEL_RULE,  # compound, F5 그대로
        window_len=T_WINDOW,
        val_months=T_VAL_MONTHS,
    )


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """합성 입력 묶음. 파일은 임시 폴더에 실제로 만든다(manifest의 sha256 때문)."""
    monkeypatch.setattr(RUN, "jm_config", _small_jm_config)
    monkeypatch.setattr(RUN, "LAMBDAS", T_LAMBDAS)
    monkeypatch.setattr(RUN, "JM_SEEDS", T_SEEDS)
    monkeypatch.setattr(RUN, "GAMMAS", T_GAMMAS)

    days = bdays(N_DAYS, date(2010, 1, 4))
    kospi_ret, _ = two_regime(N_DAYS, block=180, seed=1)
    kospi_ret = kospi_ret.copy()
    kospi_ret[0] = np.nan
    etf = slot_returns(N_DAYS, seed=2)[:, :5].copy()
    for k, s in enumerate(LISTING):
        etf[:s, k] = np.nan
    cash = np.full(N_DAYS, 0.025 / 252)
    cash[0] = np.nan
    cash[900:906] = np.nan  # 금리 오래된 값 한도를 넘은 구간(F7)
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    files = {}
    for name in ("kr_obs_part0.parquet", "etf_daily.csv", "manifest_files.tsv"):
        p = in_dir / name
        p.write_bytes(f"synthetic:{name}".encode())
        files[name.split(".")[0]] = p
    inp = D.RealInputs(
        days=days,
        kospi_ret=kospi_ret,
        cash_ret=cash,
        etf_ret=etf,
        end_idx=N_DAYS - 1,
        month_ends=L.month_end_indices(days),
        kospi_available_ok=np.r_[np.ones(N_DAYS - 1, dtype=bool), False],
        files=files,
        snapshots={D.OBS_TABLE: "synthetic"},
        diagnostics={"kospi_sessions": N_DAYS},
        stops=[],
    )
    guard_info = {"code_sha": "deadbeef", "modeler_tag": {}, "my_tag": {}}
    return inp, guard_info, tmp_path


def _walk_numbers(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk_numbers(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_numbers(v, f"{path}[{i}]")
    elif isinstance(o, float):
        yield path, o


def _stable(result_path):
    """finished·run_id만 빼고 비교한다."""
    d = json.loads(result_path.read_text())
    d["manifest"].pop("finished")
    d["manifest"].pop("run_id")
    return d


@pytest.fixture
def run1(bundle):
    inp, gi, tmp = bundle
    out = tmp / "out" / "run_a"
    result = RUN.run_full(inp, out, gi)
    return inp, gi, tmp, out, result


def test_frozen_constants():
    """F1~F5: 시험용 축소와 별개로 진입점 상수는 결정값 그대로다."""
    assert RUN.LAMBDAS == (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
    assert RUN.GAMMAS == (0.001, 0.01, 0.1)
    assert RUN.JM_SEEDS == tuple(range(20261018, 20261028))
    assert RUN.JM_MAX_ITER == 100 and RUN.JM_FEATURE_WARMUP == 252
    assert RUN.JM_LABEL_RULE == "compound"
    cfg = RUN.jm_config()
    assert cfg.lambdas == RUN.LAMBDAS and cfg.seeds == RUN.JM_SEEDS
    assert cfg.window_len == 3000 and cfg.val_months == 96


def test_run_full_outputs_exist_and_schema(run1):
    inp, gi, _, out, result = run1
    assert sorted(p.name for p in out.iterdir()) == [
        "jm_m.npy",
        "result.json",
        "rsp_seed_weights.npy",
        "rsp_weights.npy",
    ]
    assert set(result) == {"candidates", "holm_placebo", "baselines", "manifest"}
    assert set(result["candidates"]) == {"R-JM", "R-SP", "R-SP-IRP"}
    for cand, per_cost in result["candidates"].items():
        assert set(per_cost) == {"low", "high"}
        for c, v in per_cost.items():
            assert set(v) == {"metrics", "b5r_weights", "b5r_metrics", "placebo"}
            assert "calmar" in v["metrics"] and "calmar" in v["b5r_metrics"]
            assert (v["placebo"] is None) == (cand == "R-SP-IRP")
    assert set(result["holm_placebo"]) == {"low", "high"}
    for c in ("low", "high"):
        assert set(result["holm_placebo"][c]) == {"R-JM", "R-SP"}
        for h in result["holm_placebo"][c].values():
            assert 0.0 < h["adj_p"] <= 1.0 and isinstance(h["reject"], bool)
    for c in ("low", "high"):
        assert set(result["baselines"][c]) == {"B1", "B2", "B3", "B4", "B5"}
    # 디스크의 result.json은 반환값과 같은 내용이다
    on_disk = json.loads((out / "result.json").read_text())
    assert set(on_disk) == set(result)


def test_run_full_no_nan_where_not_allowed(run1):
    _, _, _, out, result = run1
    on_disk = json.loads((out / "result.json").read_text())
    for cand in ("R-JM", "R-SP", "R-SP-IRP"):
        for c in ("low", "high"):
            v = on_disk["candidates"][cand][c]
            for key in ("metrics", "b5r_metrics", "b5r_weights"):
                for path, x in _walk_numbers(v[key], key):
                    assert np.isfinite(x), f"{cand}/{c}/{path}={x}"
            if v["placebo"] is not None:
                assert np.isfinite(v["placebo"]["actual"])
                assert all(np.isfinite(s) for s in v["placebo"]["shifted"])
    for c in ("low", "high"):
        for line, m in on_disk["baselines"][c].items():
            for path, x in _walk_numbers(m, line):
                assert np.isfinite(x), f"baseline {c}/{path}={x}"
    for name in ("rsp_weights.npy", "rsp_seed_weights.npy", "jm_m.npy"):
        assert np.isfinite(np.load(out / name)).all(), name


def test_run_full_weight_invariants(run1):
    _, _, _, out, result = run1
    w = np.load(out / "rsp_weights.npy")
    sw = np.load(out / "rsp_seed_weights.npy")
    m = np.load(out / "jm_m.npy")
    n = result["manifest"]["diagnostics"]["n_decisions"]
    assert n >= 15  # placebo 이동(최소 6)이 여럿 나올 만큼
    assert w.shape == (n, 6) and sw.shape == (n, len(RUN.RSP_SEEDS), 6) and m.shape == (n,)
    assert (w >= 0).all() and (w <= 1).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)
    assert (sw >= 0).all() and (sw <= 1).all()
    np.testing.assert_allclose(sw.sum(axis=2), 1.0, atol=1e-9)
    np.testing.assert_allclose(sw.mean(axis=1), w, atol=1e-12)  # 결정 비중 = 시드 평균
    assert set(np.unique(m)) <= {0.5, 1.0}
    for cost in ("low", "high"):
        for cand in ("R-JM", "R-SP", "R-SP-IRP"):
            b5 = np.array(result["candidates"][cand][cost]["b5r_weights"])
            assert (b5 >= 0).all()
            assert b5.sum() == pytest.approx(1.0)
        pl = result["candidates"]["R-SP"][cost]["placebo"]
        assert pl["n_shifts"] == n - 2 * 6 + 1 == len(pl["shifted"])
        assert 0.0 < pl["p"] <= 1.0
    # IRP 구조는 안전 슬롯(114260·148070·현금)에 항상 0.3 이상(Q7: 0.7·softmax + 0.3·u)
    irp_diag = result["manifest"]["diagnostics"]["rsp_irp"]
    assert irp_diag["mean_risk_exposure"] <= 0.7 + 1e-9


def test_run_full_start_after_warmup_and_dates(run1):
    inp, _, _, out, result = run1
    diag = result["manifest"]["diagnostics"]
    days = inp.days
    start_idx = days.index(diag["diagnostic_start"])
    # JM: 유효한 날이 창+예열(252)만큼 쌓인 뒤, 거기서 다시 검증 월 수만큼 지나서 첫 결정
    n_valid = np.cumsum(~np.isnan(inp.kospi_ret - inp.cash_ret))
    ready = int(np.searchsorted(n_valid, T_WINDOW + RUN.JM_FEATURE_WARMUP, side="left"))
    assert days.index(diag["first_decision_jm"]) > ready
    assert n_valid[days.index(diag["first_decision_jm"])] > T_WINDOW + RUN.JM_FEATURE_WARMUP
    # R-SP: 마지막 ETF 상장 + 63거래일 + 학습 쌍 60개월 뒤
    first_ok = max(LISTING) + 63
    assert days.index(diag["first_decision_rsp"]) >= first_ok + 60 * 20
    # 진단 시작은 둘 중 늦은 쪽, 모든 결정은 그 뒤 월말이며 마지막 세션 전이다
    assert start_idx == max(
        days.index(diag["first_decision_jm"]), days.index(diag["first_decision_rsp"])
    )
    me = set(inp.month_ends)
    assert start_idx in me and start_idx >= RUN.JM_FEATURE_WARMUP
    assert diag["n_decisions"] == len([d for d in me if start_idx <= d <= inp.end_idx - 1])


def test_run_full_lambda_one_month_lag(run1):
    """F6: 월말 k의 λ는 월말 k−1에서 골랐다. 진입점이 쓴 설정으로 다시 만들어 대조한다."""
    inp, _, _, out, result = run1
    diag = result["manifest"]["diagnostics"]
    start = inp.days.index(diag["diagnostic_start"])
    me = inp.month_ends
    decisions = [d for d in me if d >= start and d + 1 <= inp.end_idx]
    jm = JumpModelRule(inp.kospi_ret, inp.cash_ret, me, RUN.jm_config())
    m_saved = np.load(out / "jm_m.npy")
    picks: dict[str, int] = {}
    for i, t in enumerate(decisions):
        o = jm.decide(t)
        prev = me[me.index(t) - 1]
        assert o["lambda_selected_at"] == prev < t
        assert o["lambda"] in T_LAMBDAS
        # 검증창은 prev에서 끝난다: 검증 샤프 계산에 쓴 월말은 모두 prev 이하
        assert jm.val_month_ends(prev)[-1] == prev
        assert o["m"] == m_saved[i]
        picks[str(o["lambda"])] = picks.get(str(o["lambda"]), 0) + 1
    assert picks == result["manifest"]["diagnostics"]["jm"]["lambda_pick_counts"]
    # 첫 결정의 λ도 첫 결정 직전 월말에서 골랐으므로, 그 월말은 검증 월 수를 채운다
    assert len(jm.val_month_ends(me[me.index(decisions[0]) - 1])) == T_VAL_MONTHS


def test_run_full_manifest_records(run1):
    inp, gi, _, out, result = run1
    man = result["manifest"]
    assert man["run_id"] == "run_a" and man["code_sha"] == "deadbeef"
    assert man["guard"] == gi and man["snapshots"] == inp.snapshots
    p = man["params"]
    assert tuple(p["lambdas"]) == T_LAMBDAS and tuple(p["gammas"]) == T_GAMMAS
    assert tuple(p["jm_seeds"]) == T_SEEDS and tuple(p["rsp_seeds"]) == RUN.RSP_SEEDS
    assert p["jm_max_iter"] == 100 and p["feature_warmup"] == 252
    assert p["label_rule"] == "compound" and p["sharpe_eps"] == 0.005
    assert p["costs_one_way"] == {"low": 0.0005, "high": 0.0030}
    assert p["cash_staleness_days"] == 14 and str(p["end_date"]) == D.END_DATE.isoformat()
    assert set(man["input_paths"]) == set(inp.files)
    assert man["input_paths"] == {k: str(v) for k, v in inp.files.items()}
    assert man["input_sha256"] == {k: I.sha256_file(v) for k, v in inp.files.items()}
    # 진단 기록: 쌍을 뺀 달·λ 선택·현금 비중이 들어 있다
    d = man["diagnostics"]
    assert d["jm"]["n_decisions"] == d["n_decisions"] == d["rsp"]["n_decisions"]
    assert sum(d["jm"]["lambda_pick_counts"].values()) == d["n_decisions"]
    assert d["rsp"]["n_retrains"] >= 1 and 0.0 <= d["rsp"]["mean_cash_weight"] <= 1.0
    assert sum(d["rsp"]["gamma_pick_counts"].values()) == d["rsp"]["n_retrains"]
    # F7: 현금이 빈 6일이 낀 달은 학습에서 빠진다 → 뺀 달 합이 0보다 크다
    assert sum(d["rsp"]["dropped_months_by_retrain"].values()) > 0


def test_run_full_deterministic(bundle):
    inp, gi, tmp = bundle
    RUN.run_full(inp, tmp / "out" / "a", gi)
    RUN.run_full(inp, tmp / "out" / "b", gi)
    for name in ("jm_m.npy", "rsp_weights.npy", "rsp_seed_weights.npy"):
        assert (tmp / "out" / "a" / name).read_bytes() == (tmp / "out" / "b" / name).read_bytes()
    assert _stable(tmp / "out" / "a" / "result.json") == _stable(tmp / "out" / "b" / "result.json")


def test_run_full_stops_before_compute(bundle):
    """입력 stops나 KOSPI 가용성 위반이 있으면 결정 계산 전에 멈춘다(결과 파일 없음)."""
    inp, gi, tmp = bundle
    inp.stops.append("테스트: 상장 뒤 종가 없는 세션")
    with pytest.raises(I.InputError, match="STOP"):
        RUN.run_full(inp, tmp / "out" / "s1", gi)
    assert not (tmp / "out" / "s1").exists()
    inp.stops.clear()
    inp.kospi_available_ok[N_DAYS - 100 : N_DAYS - 1] = False  # 마지막 월말들 가용 시각 위반
    with pytest.raises(I.InputError, match="가용하지 않은"):
        RUN.run_full(inp, tmp / "out" / "s2", gi)
    assert not (tmp / "out" / "s2").exists()
