"""T1-5 계측: 16개 run 의 리밸런스 수익으로 DSR·PBO 를 계산한다.

입력  ``<out>/runs/<run_id>/rebalance_returns.parquet`` 16개 (``t15_regen`` 이 만든다).
출력  ``<out>/result/{returns_matrix.parquet, dsr_pbo.json, cscv_combinations.parquet}``.
``--copy-to-res`` 가 있으면 앞 둘만 ``RES/dsr_pbo/`` 에 복사한다 (정본은 ``stock_data``).

판정이 아니라 계측이다. 채택(``E2_h20_FS1h_seed0``)과 갈래를 바꾸지 않는다.
검사에 걸리면 계산하지 않고 멈춘다 (종료 코드 2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeler.etl.multiple_testing import (
    annualize_sharpe,
    deflated_sharpe,
    pbo_cscv,
    sharpe,
    strategy_is_best_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
RES = REPO_ROOT / "docs" / "dev" / "20260907_model_experiment" / "results"
RES_COPY_DIR = RES / "dsr_pbo"

ADOPTED = "E2_h20_FS1h_seed0"
RUN_IDS = [
    "E0_h20_MA-rank_seed0",
    "E0_h20_MA-tree_seed0",
    "E1_h20_y_top-hgb_clf_seed0",
    "E1_h20_y_top-logit_seed0",
    "E1_h20_y_up-hgb_clf_seed0",
    "E1_h20_y_up-logit_seed0",
    "E2_h20_FS1_seed0",
    "E2_h20_FS1h_seed0",
    "E2_h20_FS2_seed0",
    "E3_h20_flow-native_t_seed0",
    "E4_h20_E4a-seed_seed1",
    "E4_h20_E4a-seed_seed2",
    "E4_h20_E4b-monotonic_seed0",
    "E5_h20_FS3_seed0",
    "E5_h20_FS3_seed1",
    "E5_h20_FS3_seed2",
]
N_SPLITS = 8
REQUIRED_COLS = ("fold_id", "pred_col", "rebalance_date", "horizon", "gross_return", "net_return")


class T15Error(RuntimeError):
    """검사 실패. 계산하지 않고 멈춘다."""


def default_out() -> Path:
    root = os.environ.get("STOCK_DATA_ROOT")
    if not root:
        raise T15Error("--out 이 없고 STOCK_DATA_ROOT 도 없다")
    return Path(root) / "kr" / "output" / "t15_dsr_pbo"


def _find_summary(run_id: str, res_root: Path) -> dict:
    # ``smoke/``·``holdout/`` 에도 같은 run_id 폴더가 있다. 단계 폴더(E0~E5)만 본다 — t15_regen.find_run_dir 와 같다.
    hits = sorted(p for p in res_root.glob(f"E[0-9]/{run_id}/summary.json"))
    if len(hits) != 1:
        raise T15Error(f"{run_id}: RES 아래 summary.json {len(hits)}개 {hits}")
    return json.loads(hits[0].read_text())


def _has_fail(obj) -> bool:
    """regen_spec.json 안에 FAIL 이 기록됐는지. 키 이름을 모르므로 값 전체를 훑는다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("passed", "pass") and v is False:
                return True
            if _has_fail(v):
                return True
        return False
    if isinstance(obj, (list, tuple)):
        return any(_has_fail(v) for v in obj)
    return isinstance(obj, str) and obj.strip().upper() == "FAIL"


def load_runs(out: Path, res_root: Path = RES, allow_failed: bool = False, run_ids=RUN_IDS) -> tuple[dict, list[str]]:
    """16개 run 을 읽고 검사한다. ``(frames, warnings)`` 를 돌려준다."""
    runs_dir = out / "runs"
    missing = [r for r in run_ids if not (runs_dir / r / "rebalance_returns.parquet").is_file()]
    if missing:
        raise T15Error(f"rebalance_returns.parquet 이 없는 run {len(missing)}개: {missing}")

    warnings: list[str] = []
    failed = []
    for r in run_ids:
        spec = runs_dir / r / "regen_spec.json"
        if spec.is_file() and _has_fail(json.loads(spec.read_text())):
            failed.append(r)
    if failed:
        msg = f"regen_spec.json 에 FAIL 이 기록된 run {len(failed)}개: {failed}"
        if not allow_failed:
            raise T15Error(msg + " (--allow-failed 가 없어 멈춘다)")
        warnings.append("!!! 경고 !!! " + msg + " (--allow-failed 로 계속한다)")

    frames = {}
    for r in run_ids:
        df = pd.read_parquet(runs_dir / r / "rebalance_returns.parquet")
        absent = [c for c in REQUIRED_COLS if c not in df.columns]
        if absent:
            raise T15Error(f"{r}: 필수 열 없음 {absent}")
        pred_cols = df["pred_col"].unique().tolist()
        if len(pred_cols) != 1:
            raise T15Error(f"{r}: pred_col 이 하나가 아니다 {pred_cols}")
        want = _find_summary(r, res_root).get("primary_pred_col")
        if pred_cols[0] != want:
            raise T15Error(f"{r}: pred_col {pred_cols[0]!r} != summary.primary_pred_col {want!r}")
        frames[r] = df.reset_index(drop=True)

    ref_id = run_ids[0]
    ref = frames[ref_id]
    for r in run_ids[1:]:
        df = frames[r]
        for col in ("rebalance_date", "fold_id"):
            if len(df) != len(ref) or not df[col].reset_index(drop=True).equals(ref[col]):
                raise T15Error(f"{r}: {col} 열이 {ref_id} 와 행 순서까지 같지 않다 (행 {len(df)} 대 {len(ref)})")
    if ref["rebalance_date"].duplicated().any():
        raise T15Error("rebalance_date 에 중복이 있다")
    if not ref["rebalance_date"].is_monotonic_increasing:
        raise T15Error("rebalance_date 가 시간 순서가 아니다. CSCV 블록이 깨진다")
    return frames, warnings


def build_matrix(frames: dict, col: str, run_ids=RUN_IDS) -> pd.DataFrame:
    ref = frames[run_ids[0]]
    m = pd.DataFrame({r: frames[r][col].to_numpy(dtype=float) for r in run_ids}, index=pd.Index(ref["rebalance_date"], name="rebalance_date"))
    return m


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def compute(frames: dict, run_ids=RUN_IDS, adopted: str = ADOPTED) -> dict:
    net = build_matrix(frames, "net_return", run_ids)
    gross = build_matrix(frames, "gross_return", run_ids)
    horizon = int(frames[adopted]["horizon"].iloc[0])
    sr_net = {r: sharpe(net[r].to_numpy()) for r in run_ids}
    sr_gross = {r: sharpe(gross[r].to_numpy()) for r in run_ids}
    dsr = deflated_sharpe(net[adopted].to_numpy(), list(sr_net.values()), n_trials=len(run_ids), horizon=horizon)
    cscv = pbo_cscv(net, n_splits=N_SPLITS)
    is_best = strategy_is_best_summary(cscv, adopted)

    sr_table = pd.DataFrame(
        {
            "sr_gross_annual": {r: annualize_sharpe(v, horizon) for r, v in sr_gross.items()},
            "sr_net_annual": {r: annualize_sharpe(v, horizon) for r, v in sr_net.items()},
            "mean_net": net.mean(),
            "mean_gross": gross.mean(),
        }
    )
    order = sr_table["sr_net_annual"].sort_values(ascending=False).index.tolist()
    sr_table["rank_net"] = [order.index(r) + 1 for r in sr_table.index]
    lam = cscv.lambdas
    result = {
        "adopted": adopted,
        "n_runs": len(run_ids),
        "n_obs": int(len(net)),
        "horizon": horizon,
        "cost_basis": "net_return (비용 반영)",
        "dsr": dsr.to_dict(),
        "pbo": {
            "pbo": cscv.pbo,
            "n_splits": cscv.n_splits,
            "block_sizes": cscv.block_sizes,
            "n_combinations": cscv.n_combinations,
            "n_lambda_positive": int((lam > 0).sum()),
            "n_lambda_nonpositive": int((lam <= 0).sum()),
            "lambda_median": float(np.median(lam)),
        },
        "adopted_is_best": is_best,
        "adopted_rank_net_of_n": int(sr_table.loc[adopted, "rank_net"]),
        "sr_table": sr_table.reset_index(names="run_id").to_dict(orient="records"),
        "pbo_overfit_note": bool(cscv.pbo > 0.5),
    }
    return {"result": _clean(result), "net": net, "gross": gross, "cscv": cscv, "sr_table": sr_table}


def summarize(res: dict, warnings: list[str] | None = None) -> str:
    d, p, b = res["dsr"], res["pbo"], res["adopted_is_best"]
    L = list(warnings or [])
    L.append(f"T1-5 DSR·PBO (계측, 채택·갈래 불변)  채택 {res['adopted']}  N={res['n_runs']}  T={res['n_obs']}  h={res['horizon']}")
    L.append(f"DSR  SR^ 연환산 {d['sr_hat_annual']:.3f}  SR0 연환산 {d['sr0_annual']:.3f}  DSR {d['dsr']:.4f}  p-value(1-DSR) {d['p_value']:.4f}")
    L.append(f"     skew {d['skew']:.3f}  kurtosis(4차 적률 비) {d['kurtosis']:.3f}  SR 분산(per-period) {d['sr_var']:.5f}")
    L.append(
        f"PBO  {p['pbo']:.3f}  (조합 {p['n_combinations']}개, λ>0 {p['n_lambda_positive']} / λ<=0 {p['n_lambda_nonpositive']}, "
        f"블록 {p['block_sizes']})"
    )
    if b["n_is_best"]:
        L.append(
            f"채택 run 이 IS 1위인 조합 {b['n_is_best']}/{b['n_combinations']}  OOS 순위(위에서) 중앙 {b['oos_rank_from_top_median']:.1f}"
            f"  범위 {b['oos_rank_from_top_best']:.1f}~{b['oos_rank_from_top_worst']:.1f} (/{b['n_strategies']})"
        )
    else:
        L.append(
            f"채택 run 이 IS 1위인 조합: 해당 없음 (0/{b['n_combinations']}). 전체 조합 순위(위에서) IS 중앙 "
            f"{b['is_rank_from_top_median_all']:.1f} 범위 {b['is_rank_from_top_range_all']}, OOS 중앙 "
            f"{b['oos_rank_from_top_median_all']:.1f} 범위 {b['oos_rank_from_top_range_all']}"
        )
    L.append(f"채택 run 비용 반영 연환산 SR 순위 {res['adopted_rank_net_of_n']}/{res['n_runs']}")
    L.append("run                              SR(gross)  SR(net)  순위")
    for row in sorted(res["sr_table"], key=lambda r: r["rank_net"]):
        mark = " *" if row["run_id"] == res["adopted"] else ""
        L.append(f"{row['run_id']:<32} {row['sr_gross_annual']:>9.3f} {row['sr_net_annual']:>8.3f} {row['rank_net']:>5}{mark}")
    if res["pbo_overfit_note"]:
        L.append("PBO > 0.5: 선택 과정의 과적합 확률이 높다 (참고, 채택·갈래 불변).")
    return "\n".join(L)


def write_outputs(computed: dict, out: Path, copy_to_res: bool = False, res_copy_dir: Path = RES_COPY_DIR) -> list[Path]:
    rdir = out / "result"
    rdir.mkdir(parents=True, exist_ok=True)
    mpath, jpath, cpath = rdir / "returns_matrix.parquet", rdir / "dsr_pbo.json", rdir / "cscv_combinations.parquet"
    computed["net"].to_parquet(mpath)
    jpath.write_text(json.dumps(computed["result"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    t = computed["cscv"].table.copy()
    t["lam"] = t["lam"].replace([np.inf, -np.inf], np.nan)  # parquet 에는 ±inf 대신 null (현 정의에서는 안 나온다)
    t.to_parquet(cpath)
    written = [mpath, jpath, cpath]
    if copy_to_res:
        res_copy_dir.mkdir(parents=True, exist_ok=True)
        for src in (mpath, jpath):
            shutil.copy2(src, res_copy_dir / src.name)
            written.append(res_copy_dir / src.name)
    return written


def run_report(out: Path, res_root: Path = RES, allow_failed: bool = False, copy_to_res: bool = False, res_copy_dir: Path = RES_COPY_DIR) -> str:
    frames, warnings = load_runs(out, res_root, allow_failed)
    computed = compute(frames)
    write_outputs(computed, out, copy_to_res, res_copy_dir)
    return summarize(computed["result"], warnings)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="기본 $STOCK_DATA_ROOT/kr/output/t15_dsr_pbo")
    ap.add_argument("--allow-failed", action="store_true", help="regen_spec.json 에 FAIL 이 있어도 계속")
    ap.add_argument("--copy-to-res", action="store_true", help="returns_matrix.parquet·dsr_pbo.json 을 RES/dsr_pbo/ 에 복사")
    a = ap.parse_args(argv)
    try:
        out = a.out or default_out()
        print(run_report(out, allow_failed=a.allow_failed, copy_to_res=a.copy_to_res))
    except (T15Error, ValueError) as e:
        print(f"중단: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
