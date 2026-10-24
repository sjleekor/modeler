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
import warnings
from datetime import date, datetime, timedelta
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

# 전진 평가 (동결 뒤 실제 월말 결정). 진단 경로의 상수와 무관하다.
# 첫 전진 결정일: 10-27~29 동결 뒤 처음 오는 월말(사전등록 §5). 그 전 월말은 쓰지 않는다.
FORWARD_FIRST_DECISION = date(2026, 10, 30)
FORWARD_OUTPUT_DIR = "kr/output/pension_alloc_forward"


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
            # JM 값은 실제로 쓴 cfg에서 적는다(상수와 어긋나도 기록이 실행을 따른다)
            "lambdas": cfg.lambdas,
            "gammas": GAMMAS,
            "jm_seeds": cfg.seeds,
            "rsp_seeds": RSP_SEEDS,
            "jm_max_iter": cfg.max_iter,
            "feature_warmup": cfg.feature_warmup,
            "label_rule": cfg.label_rule,
            "jm_window_len": cfg.window_len,
            "jm_val_months": cfg.val_months,
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


# =========================================================================== 전진(forward) 평가
def _is_last_weekday_of_month(d: date) -> bool:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.month != d.month


def forward_month_ends(days: list[date], end_idx: int) -> list[int]:
    """마스터 월말 인덱스. 끝 날짜가 그 달의 마지막 평일이면 끝 날짜도 월말로 센다.

    ``month_end_indices``는 다음 세션이 있어야 월말을 알아서 달력 마지막 날을 빼 버린다. 월말 당일
    (예: 10-30 종가 뒤)에 그날의 결정 비중을 얻으려면 끝 날짜 자체가 월말인지 알아야 한다. 끝 날짜가
    거래일이고 그 뒤 첫 평일이 다른 달이면 끝 날짜가 월 마지막 거래일이다. 달 끝 평일이 아닌데
    그 사이가 모두 휴장인 경우는 이 판정이 놓치고, 다음 세션이 생긴 뒤 다시 돌리면 잡힌다.
    """
    me = L.month_end_indices(days)
    if end_idx not in me and _is_last_weekday_of_month(days[end_idx]):
        me = [*me, end_idx]
    return me


def _clean(o):
    """JSON용: NaN·inf를 None으로 바꾼다."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def _forward_manifest(
    inp, out_dir: Path, guard_info: dict, first_on_or_after: date, **extra
) -> dict:
    cfg = jm_config()
    return {
        "run_id": out_dir.name,
        "kind": "forward",
        "finished": datetime.now().isoformat(timespec="seconds"),
        "guard": guard_info,
        "code_sha": guard_info["code_sha"],
        "kr_snapshot": inp.kr_snapshot,
        "snapshots": inp.snapshots,
        "obs_view": inp.obs_view,  # first = 그 시점에 알려진 최초 관측본
        "end_date": inp.end_date,
        "first_forward_decision_on_or_after": first_on_or_after,
        "canonical": inp.canonical,
        "input_paths": {k: str(v) for k, v in inp.files.items()},
        "input_sha256": {k: I.sha256_file(v) for k, v in inp.files.items()},
        "params": {
            "lambdas": cfg.lambdas,
            "gammas": GAMMAS,
            "jm_seeds": cfg.seeds,
            "rsp_seeds": RSP_SEEDS,
            "jm_max_iter": cfg.max_iter,
            "feature_warmup": cfg.feature_warmup,
            "label_rule": cfg.label_rule,
            "jm_window_len": cfg.window_len,
            "jm_val_months": cfg.val_months,
            "costs_one_way": COSTS,
            "cash_staleness_days": D.CASH_STALENESS_DAYS,
            "sharpe_eps": SHARPE_EPS,
            "placebo": False,  # placebo는 진단 전용(사전등록)
        },
        "input_diagnostics": inp.diagnostics,
        **extra,
    }


def run_forward(
    inp: D.ForwardInputs,
    out_dir: Path,
    guard_info: dict,
    first_decision_on_or_after: date = FORWARD_FIRST_DECISION,
) -> dict:
    """전진 구간(첫 전진 결정일 이후 월말 결정)을 평가해 ``out_dir``에 쓴다. placebo는 안 한다.

    규칙 객체는 끝 날짜까지의 전체 자료로 만들지만 ``decide(t)``는 t까지만 본다(절단 시험이 보증).
    결정은 월말 t 중 ``days[t] >= first_decision_on_or_after``인 것이고, 체결(t+1)이 끝 날짜 안인
    것(``evaluated``)만 장부로 평가한다. 끝 날짜 자체가 월말이면 그날의 결정 비중은 평가 없이
    ``pending_decision``으로 기록한다(실제 주문용 비중). 결정이 하나도 없으면 그 사실만 manifest에
    적고 정상 종료한다. 비어 있지 않은 출력 폴더는 거부한다.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"출력 폴더가 비어 있지 않습니다: {out_dir}")
    if inp.stops:
        raise I.InputError("STOP: " + "; ".join(inp.stops))
    days = inp.days
    me = forward_month_ends(days, inp.end_idx)
    cand = [d for d in me if days[d] >= first_decision_on_or_after]
    evaluated = [d for d in cand if d + 1 <= inp.end_idx]
    pending = [d for d in cand if d + 1 > inp.end_idx]
    dec_dates = {
        "evaluated_decision_dates": [days[d] for d in evaluated],
        "pending_decision_date": days[pending[0]] if pending else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cand:
        man = _forward_manifest(
            inp,
            out_dir,
            guard_info,
            first_decision_on_or_after,
            status="no_decision_yet",
            **dec_dates,
        )
        result = {"status": "no_decision_yet", "manifest": man}
        (out_dir / "forward_result.json").write_text(
            json.dumps(_clean(result), ensure_ascii=False, indent=1, default=_json_default)
        )
        return result

    cfg = jm_config()
    jm = JumpModelRule(inp.kospi_ret, inp.cash_ret, me, cfg)
    slot = np.column_stack([inp.etf_ret, inp.cash_ret])
    first_ok = max(int(np.flatnonzero(~np.isnan(inp.etf_ret[:, k]))[0]) for k in range(5)) + 63
    me_rsp = [d for d in me if d >= first_ok]
    rsp = SoftmaxPolicyRule(slot, me_rsp, GAMMAS, RSP_SEEDS, "main")
    rsp_irp = SoftmaxPolicyRule(slot, me_rsp, GAMMAS, RSP_SEEDS, "irp")
    try:
        t_jm0 = jm.first_decision_index()
        t_rsp0 = me_rsp[rsp.first_k]
    except ValueError as exc:
        raise I.InputError(f"STOP: 규칙 첫 결정일을 못 구했습니다: {exc}") from exc
    if cand[0] < max(t_jm0, t_rsp0):
        raise I.InputError(
            f"STOP: 첫 전진 결정 {days[cand[0]]}이 규칙 첫 결정일(JM {days[t_jm0]}, "
            f"RSP {days[t_rsp0]})보다 이릅니다"
        )
    all_dec = [*evaluated, *pending]
    # 값이 t까지 가용한가 (F9). 끝 날짜 자체는 다음 세션이 없어 확인할 수 없다(manifest에 적는다).
    used = [d for d in me if jm.first_ready_index() <= d <= all_dec[-1] and d != inp.end_idx]
    late = [d for d in used if not inp.kospi_available_ok[d]]
    if late:
        raise I.InputError(
            f"STOP: KOSPI 값이 결정 시각까지 가용하지 않은 월말 {len(late)}개 (첫 {days[late[0]]})"
        )

    b4 = L.line_specs(L.US_MAIN)["B4"]
    jm_outs = [jm.decide(t) for t in all_dec]
    W_jm = np.array([b4.weights(o["m"]) for o in jm_outs])
    pos = {d: k for k, d in enumerate(me_rsp)}
    rsp_outs = [rsp.decide(pos[t]) for t in all_dec]
    W_rsp = np.array([o["weights"] for o in rsp_outs])
    irp_outs = [rsp_irp.decide(pos[t]) for t in all_dec]
    W_irp = np.array([o["weights"] for o in irp_outs])
    n_ev = len(evaluated)

    ev = base = None
    if n_ev:
        ret_jm = inp.etf_ret[:, [0, 1, 3]]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # 짧은 구간의 표준편차 등
            kw = {"placebo": False}
            ev = {
                "R-JM": evaluate_candidate(
                    days, ret_jm, inp.cash_ret, evaluated, W_jm[:n_ev], inp.end_idx, **kw
                ),
                "R-SP": evaluate_candidate(
                    days, inp.etf_ret, inp.cash_ret, evaluated, W_rsp[:n_ev], inp.end_idx, **kw
                ),
                "R-SP-IRP": evaluate_candidate(
                    days, inp.etf_ret, inp.cash_ret, evaluated, W_irp[:n_ev], inp.end_idx, **kw
                ),
            }
            base = baseline_lines(inp, evaluated, COSTS)
    pend = None
    if pending:
        pend = {
            "date": days[pending[0]],
            "weights_note": "R-JM: DOM·US_MAIN·BOND10·현금, R-SP·R-SP-IRP: ETF_CODES 순서+현금",
            "R-JM": W_jm[-1],
            "R-SP": W_rsp[-1],
            "R-SP-IRP": W_irp[-1],
            "kospi_value_available_checked": False,
        }
    diag = {
        "first_decision_jm": days[t_jm0],
        "first_decision_rsp": days[t_rsp0],
        "n_evaluated": n_ev,
        "n_pending": len(pending),
        "jm": jm_diagnostics(jm, jm_outs),
        "rsp": rsp_diagnostics(rsp_outs),
        "rsp_irp": rsp_diagnostics(irp_outs),
    }
    np.save(out_dir / "rsp_seed_weights.npy", np.array([o["seed_weights"] for o in rsp_outs]))
    np.save(out_dir / "rsp_weights.npy", W_rsp)
    np.save(out_dir / "rsp_irp_weights.npy", W_irp)
    np.save(out_dir / "jm_weights.npy", W_jm)
    np.save(out_dir / "jm_m.npy", np.array([o["m"] for o in jm_outs]))
    man = _forward_manifest(
        inp,
        out_dir,
        guard_info,
        first_decision_on_or_after,
        status="evaluated" if n_ev else "pending_only",
        diagnostics=diag,
        weights_rows_dates=[days[d] for d in all_dec],
        **dec_dates,
    )
    result = {
        "status": man["status"],
        "candidates": ev,
        "baselines": base,
        "pending_decision": pend,
        "manifest": man,
    }
    (out_dir / "forward_result.json").write_text(
        json.dumps(_clean(result), ensure_ascii=False, indent=1, default=_json_default)
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(_clean(man), ensure_ascii=False, indent=1, default=_json_default)
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
    """``--schema-only``는 가드 없이 스키마만 출력한다. 그 밖에는 가드가 첫 호출이다.

    ``--forward --end-date D --kr-snapshot S``는 전진 평가(동결 뒤 월말 결정)다. 두 값은
    ``--forward``와 같이만 쓰고 ``--forward``에는 둘 다 필수다. 진단 경로는 이 인자가 없을 때와
    같다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--end-date", type=date.fromisoformat, default=None)
    ap.add_argument("--kr-snapshot", default=None)
    a = ap.parse_args(argv)
    if a.forward and (a.end_date is None or a.kr_snapshot is None):
        ap.error("--forward에는 --end-date와 --kr-snapshot이 모두 필요합니다")
    if not a.forward and (a.end_date is not None or a.kr_snapshot is not None):
        ap.error("--end-date·--kr-snapshot은 --forward와 같이만 씁니다")
    if a.schema_only:
        if a.forward:
            print(D.schema_only_forward(I.stock_data_root(), a.end_date, a.kr_snapshot))
        else:
            print(D.schema_only(I.stock_data_root()))
        return 0
    guard_info = guard.require_frozen_tag()  # 반드시 첫 호출 — 이 줄 앞에 자료 접근이 없다
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        limiter = None
    root = I.stock_data_root()
    if a.forward:
        inp = D.load_inputs_forward(root, a.end_date, a.kr_snapshot)
        run_id = a.run_id or datetime.now().strftime(f"fwd_{a.end_date:%Y%m%d}_%H%M%S")
        out_dir = root / FORWARD_OUTPUT_DIR / run_id
        run_forward(inp, out_dir, guard_info)
        del limiter
        print("출력:", out_dir)
        return 0
    inp = D.load_inputs(root)
    run_id = a.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_full(inp, root / OUTPUT_DIR / run_id, guard_info)
    del limiter
    print("출력:", root / OUTPUT_DIR / run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
