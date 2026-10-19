# ruff: noqa: E402  (BLAS 스레드 환경변수를 numpy보다 먼저 정해야 한다)
"""평가 하네스와 실제 자료 진입점 (사전등록 §4.1·§6·§10 Q6·§10.4 F1~F11).

* :func:`evaluate_candidate`는 배열만 받는다(합성 자료로 시험 가능). 실제 파일을 열지 않는다.
  기존 기준선 장부(``etf_baseline.ledger.simulate``)와 지표(``compute_metrics``)를 그대로 쓴다.
* :func:`main`은 실제 자료를 읽는 유일한 진입점이다. 전체 실행은 **첫 호출로** 동결 태그 가드
  (modeler·my 두 저장소 태그, 깨끗한 modeler 트리)를 부르고, 실패하면 자료를 열기 전에 멈춘다.
  ``--schema-only``만 가드 없이 돈다(경로·열 이름·자료형·첫날·끝날·행 수만 출력, 값은 안 읽음).
* BLAS 스레드는 1개로 고정한다(비트 단위 재현). 아래 환경변수는 numpy를 불러오기 **전에**
  정해야 하므로 이 파일 맨 위에 둔다. 다른 모듈이 먼저 numpy를 불러온 경우를 대비해
  :func:`main`이 ``threadpoolctl``로도 1개로 묶는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import lines as L
from modeler.scores.etf_baseline.ledger import LedgerResult, simulate
from modeler.scores.etf_baseline.metrics import compute_metrics
from modeler.scores.pension_alloc import data as D
from modeler.scores.pension_alloc import guard
from modeler.scores.pension_alloc.jump_model import JMConfig, JumpModelRule
from modeler.scores.pension_alloc.placebo import holm, placebo_calmar
from modeler.scores.pension_alloc.rsp import DEFAULT_SEEDS, SHARPE_EPS, SoftmaxPolicyRule

# 사전등록 §10.4 결정값 (진입점이 쓴다. 함수 인자는 시험용으로 열어 둔다)
LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)  # F1: 원문 Table 3 여섯 값
GAMMAS = (0.001, 0.01, 0.1)  # F2
JM_SEEDS = tuple(range(20261018, 20261028))  # F3
JM_MAX_ITER = 100  # F3
JM_FEATURE_WARMUP = 252  # F4
JM_LABEL_RULE = "compound"  # F5
RSP_SEEDS = DEFAULT_SEEDS
OUTPUT_DIR = "kr/output/pension_alloc_20261019"

COSTS = {"low": 0.0005, "high": 0.0030}  # 편도 (왕복 10bp·60bp, §4.1 E4)


def targets_from_weights(decisions: list[int], weights: np.ndarray) -> dict[int, np.ndarray]:
    """결정일 d의 목표 비중 -> 체결일 d+1 종가(§4.1 E3)."""
    return {d + 1: np.asarray(w, dtype=float) for d, w in zip(decisions, weights, strict=True)}


def run_ledger(
    asset_ret: np.ndarray,
    cash_ret: np.ndarray,
    decisions: list[int],
    weights: np.ndarray,
    one_way: float,
    end_idx: int,
) -> LedgerResult:
    tg = targets_from_weights(decisions, weights)
    return simulate(asset_ret, cash_ret, decisions[0] + 1, end_idx, tg, one_way)


def _metrics(days: list[date], res: LedgerResult, cash_ret: np.ndarray, start_idx: int) -> dict:
    dates_ = [days[start_idx]] + [days[i] for i in res.idx]
    nav = np.concatenate([[1.0], res.nav])
    tos = [(days[r["idx"]], r["turnover"]) for r in res.rebalances]
    return compute_metrics(dates_, nav, cash_ret[res.idx], tos)


def b5r_weights(res: LedgerResult) -> np.ndarray:
    """B5_R: 후보 비중의 기간 시간평균(슬롯별, 흘러간 비중 기준 — 09b I9와 같은 정의)."""
    w = res.weights_held.mean(axis=0)
    return w / w.sum()


def evaluate_candidate(
    days: list[date],
    asset_ret: np.ndarray,
    cash_ret: np.ndarray,
    decisions: list[int],
    weights: np.ndarray,
    end_idx: int,
    costs: dict[str, float] | None = None,
    placebo: bool = True,
) -> dict:
    """후보 하나를 비용마다 평가한다. ``weights``: (결정 수, K+1) 목표 비중(마지막 열 현금).

    반환 ``{비용이름: {"metrics", "b5r_metrics", "b5r_weights", "placebo"}}``.
    B5_R과 placebo는 비용 수준마다 따로 구한다(§10 Q6 ②·③).
    """
    out: dict = {}
    start_idx = decisions[0] + 1
    for name, ow in (costs or COSTS).items():
        res = run_ledger(asset_ret, cash_ret, decisions, weights, ow, end_idx)
        const = b5r_weights(res)
        res_b5 = run_ledger(
            asset_ret, cash_ret, decisions, np.tile(const, (len(decisions), 1)), ow, end_idx
        )

        def calmar(path: np.ndarray, ow: float = ow) -> float:
            r = run_ledger(asset_ret, cash_ret, decisions, path, ow, end_idx)
            return _metrics(days, r, cash_ret, start_idx)["calmar"]

        out[name] = {
            "metrics": _metrics(days, res, cash_ret, start_idx),
            "b5r_weights": const,
            "b5r_metrics": _metrics(days, res_b5, cash_ret, start_idx),
            "placebo": placebo_calmar(weights, calmar) if placebo else None,
        }
    return out


def jm_config() -> JMConfig:
    """F1·F3·F4·F5가 정한 진입점 설정."""
    return JMConfig(
        lambdas=LAMBDAS,
        seeds=JM_SEEDS,
        max_iter=JM_MAX_ITER,
        feature_warmup=JM_FEATURE_WARMUP,
        label_rule=JM_LABEL_RULE,
    )


def _share(counts: dict, n: int) -> dict:
    return {str(k): v / n for k, v in sorted(counts.items())} if n else {}


def jm_diagnostics(rule: JumpModelRule, outs: list[dict]) -> dict:
    """F1·F3·F5 기록: 끝점 λ 비율, 반복 상한·빈 상태 수, 이름 규칙 불일치, 약세 일수 비율."""
    n = len(outs)
    lam_counts: dict[float, int] = {}
    for o in outs:
        lam_counts[o["lambda"]] = lam_counts.get(o["lambda"], 0) + 1
    ends = (min(rule.cfg.lambdas), max(rule.cfg.lambdas))
    sel = [o["diag"] for o in outs if o["diag"] is not None]
    allf = list(rule.diag.values())
    return {
        "n_decisions": n,
        "lambda_pick_counts": {str(k): v for k, v in sorted(lam_counts.items())},
        "lambda_pick_share": _share(lam_counts, n),
        "lambda_endpoint_share": sum(lam_counts.get(e, 0) for e in ends) / n if n else None,
        "n_fits_total": len(allf),
        "n_fits_with_seed_hit_max_iter": sum(1 for d in allf if d["n_hit_max"] > 0),
        "n_seed_fits_hit_max_iter": sum(d["n_hit_max"] for d in allf),
        "n_windows_empty_state": sum(1 for d in allf if d["empty_state"]),
        "selected_label_disagree_share": (
            sum(d["label_disagree"] for d in sel) / len(sel) if sel else None
        ),
        "selected_bear_day_share_mean": (
            float(np.mean([d["bear_share"] for d in sel])) if sel else None
        ),
        "risk_state_share_over_decisions": (sum(o["is_risk"] for o in outs) / n if n else None),
    }


def rsp_diagnostics(outs: list[dict]) -> dict:
    """F2·F7·F11 기록: γ 선택 비율, 뺀 달, 현금 비중·위험 노출, 시드 sd."""
    by_r = {o["retrain_k"]: o for o in outs}
    g_counts: dict[float, int] = {}
    for o in by_r.values():
        g_counts[o["gamma"]] = g_counts.get(o["gamma"], 0) + 1
    w = np.array([o["weights"] for o in outs])
    sd = np.array([o["seed_weights"].std(axis=0) for o in outs])
    return {
        "n_decisions": len(outs),
        "n_retrains": len(by_r),
        "gamma_pick_counts": {str(k): v for k, v in sorted(g_counts.items())},
        "gamma_pick_share": _share(g_counts, len(by_r)),
        "dropped_months_by_retrain": {str(r): o["n_dropped"] for r, o in sorted(by_r.items())},
        "mean_cash_weight": float(w[:, 5].mean()),
        "mean_risk_exposure": float(w[:, [0, 1, 4]].sum(axis=1).mean()),
        "mean_weight_by_slot": w.mean(axis=0).tolist(),
        "mean_seed_sd_by_slot": sd.mean(axis=0).tolist(),
    }


def baseline_lines(
    inp: D.RealInputs, decisions: list[int], costs: dict[str, float]
) -> dict[str, dict]:
    """B1~B5를 같은 시작·같은 마스터 달력에서 (09b와 같은 함수·장부)."""
    col = {L.DOM: 0, L.US_MAIN: 1, L.BOND3: 2, L.BOND10: 3, L.GOLD: 4}
    specs = L.line_specs(L.US_MAIN)
    sig = L.risk_share_signal(inp.etf_ret[:, 0], inp.etf_ret[:, 1])
    start_idx = decisions[0] + 1
    out: dict[str, dict] = {}
    for cname, ow in costs.items():
        held_b4 = None
        for name in ["B1", "B2", "B3", "B4", "B5"]:
            spec = specs[name]
            w_fix = None
            if name == "B5":
                w_fix = float((held_b4[:, 0] + held_b4[:, 1]).mean())
            tg = L.build_targets(spec, decisions, sig.w_vm, w_fix)
            mat = np.column_stack([inp.etf_ret[:, col[c]] for c in spec.assets])
            res = simulate(mat, inp.cash_ret, start_idx, inp.end_idx, tg, ow)
            if name == "B4":
                held_b4 = res.weights_held
            out.setdefault(cname, {})[name] = _metrics(inp.days, res, inp.cash_ret, start_idx)
    return out


def run_full(inp: D.RealInputs, out_dir: Path, guard_info: dict) -> dict:
    """후보 R-JM·R-SP(주 protocol, IRP 구조는 기록용)와 기준선을 평가해 ``out_dir``에 쓴다."""
    me = inp.month_ends
    stops: list[str] = list(inp.stops)
    cfg = jm_config()
    jm = JumpModelRule(inp.kospi_ret, inp.cash_ret, me, cfg)
    t_jm0 = jm.first_decision_index()
    slot = np.column_stack([inp.etf_ret, inp.cash_ret])
    first_ok = max(int(np.flatnonzero(~np.isnan(inp.etf_ret[:, k]))[0]) for k in range(5)) + 63
    me_rsp = [d for d in me if d >= first_ok]
    rsp = SoftmaxPolicyRule(slot, me_rsp, GAMMAS, RSP_SEEDS, "main")
    rsp_irp = SoftmaxPolicyRule(slot, me_rsp, GAMMAS, RSP_SEEDS, "irp")
    t_rsp0 = me_rsp[rsp.first_k]
    start = max(t_jm0, t_rsp0)
    decisions = [d for d in me if d >= start and d + 1 <= inp.end_idx]
    # 값이 t까지 가용한가 (F9)
    used = [d for d in me if d >= jm.first_ready_index() and d <= decisions[-1]]
    late = [d for d in used if not inp.kospi_available_ok[d]]
    if late:
        stops.append(
            f"KOSPI 값이 결정 시각까지 가용하지 않은 월말 {len(late)}개 (첫 {inp.days[late[0]]})"
        )
    if stops:
        raise I.InputError("STOP: " + "; ".join(stops))

    ret_jm = inp.etf_ret[:, [0, 1, 3]]
    b4 = L.line_specs(L.US_MAIN)["B4"]
    jm_outs = [jm.decide(t) for t in decisions]
    W_jm = np.array([b4.weights(o["m"]) for o in jm_outs])
    pos = {d: k for k, d in enumerate(me_rsp)}
    rsp_outs = [rsp.decide(pos[t]) for t in decisions]
    W_rsp = np.array([o["weights"] for o in rsp_outs])
    irp_outs = [rsp_irp.decide(pos[t]) for t in decisions]
    W_irp = np.array([o["weights"] for o in irp_outs])

    ev = {
        "R-JM": evaluate_candidate(inp.days, ret_jm, inp.cash_ret, decisions, W_jm, inp.end_idx),
        "R-SP": evaluate_candidate(
            inp.days, inp.etf_ret, inp.cash_ret, decisions, W_rsp, inp.end_idx
        ),
        "R-SP-IRP": evaluate_candidate(
            inp.days, inp.etf_ret, inp.cash_ret, decisions, W_irp, inp.end_idx, placebo=False
        ),
    }
    holm_by_cost = {c: holm({k: ev[k][c]["placebo"]["p"] for k in ("R-JM", "R-SP")}) for c in COSTS}
    base = baseline_lines(inp, decisions, COSTS)
    diag = {
        "diagnostic_start": inp.days[decisions[0]],
        "first_decision_jm": inp.days[t_jm0],
        "first_decision_rsp": inp.days[t_rsp0],
        "n_decisions": len(decisions),
        "input": inp.diagnostics,
        "jm": jm_diagnostics(jm, jm_outs),
        "rsp": rsp_diagnostics(rsp_outs),
        "rsp_irp": rsp_diagnostics(irp_outs),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rsp_seed_weights.npy", np.array([o["seed_weights"] for o in rsp_outs]))
    np.save(out_dir / "rsp_weights.npy", W_rsp)
    np.save(out_dir / "jm_m.npy", np.array([o["m"] for o in jm_outs]))
    manifest = {
        "run_id": out_dir.name,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "guard": guard_info,
        "code_sha": guard_info["code_sha"],
        "snapshots": inp.snapshots,
        "input_paths": {k: str(v) for k, v in inp.files.items()},
        "input_sha256": {k: I.sha256_file(v) for k, v in inp.files.items()},
        "params": {
            "lambdas": LAMBDAS,
            "gammas": GAMMAS,
            "jm_seeds": JM_SEEDS,
            "rsp_seeds": RSP_SEEDS,
            "jm_max_iter": JM_MAX_ITER,
            "feature_warmup": JM_FEATURE_WARMUP,
            "label_rule": JM_LABEL_RULE,
            "costs_one_way": COSTS,
            "cash_staleness_days": D.CASH_STALENESS_DAYS,
            "end_date": D.END_DATE,
            "sharpe_eps": SHARPE_EPS,
        },
        "diagnostics": diag,
    }
    result = {
        "candidates": ev,
        "holm_placebo": holm_by_cost,
        "baselines": base,
        "manifest": manifest,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=_json_default)
    )
    return result


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> int:
    """``--schema-only``는 가드 없이 스키마만 출력한다. 그 밖에는 가드가 첫 호출이다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    if a.schema_only:
        print(D.schema_only(I.stock_data_root()))
        return 0
    guard_info = guard.require_frozen_tag()  # 반드시 첫 호출 — 이 줄 앞에 자료 접근이 없다
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        limiter = None
    root = I.stock_data_root()
    inp = D.load_inputs(root)
    run_id = a.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_full(inp, root / OUTPUT_DIR / run_id, guard_info)
    del limiter
    print("출력:", root / OUTPUT_DIR / run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
