"""CLI: ``python -m modeler.scores.etf_baseline.run check|run`` (09b).

``check``는 계산 전 점검만 출력한다(결과 지표를 계산하지 않는다). ``run``은 점검에 문제가
없을 때만 주 구간·U2 x 비용 두 값 x 줄 9개를 계산해 출력 디렉터리에 쓴다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

import modeler
from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import lines as L
from modeler.scores.etf_baseline.ledger import simulate
from modeler.scores.etf_baseline.metrics import compute_metrics, yearly_table

DESIGN_COMMIT = "88fc81c"
END_DATE = date(2026, 10, 8)
FIRST_DECISION = {"main": date(2012, 7, 31), "u2": date(2020, 8, 31)}
COSTS = {"low": 0.0005, "high": 0.0030}  # 편도 (왕복 10bp·60bp)
CODES = [L.DOM, L.US_MAIN, L.BOND3, L.BOND10, L.GOLD, L.US_U2]
JUMP_STOP_FROM = date(2011, 7, 19)


@dataclass
class Prepared:
    root: Path
    days: list[date]
    panel: I.ReturnPanel
    dist: I.DistInfo
    cash_ret: np.ndarray
    signal: L.RiskShareSignal
    end_idx: int
    files: dict[str, Path]


def prepare() -> Prepared:
    root = I.stock_data_root()
    mf = root / I.ETF_DIR / I.MANIFEST_FILE
    csv = root / I.ETF_DIR / I.ETF_FILE
    djson, dmap = root / I.DIST_FILE, root / I.MAP_FILE
    days = [d for d in I.load_trading_days(mf) if d <= END_DATE]
    frames = I.load_etf_frames(csv, CODES)
    dist = I.load_distributions(djson, dmap, CODES)
    panel = I.build_return_panel(days, frames, dist.rows)
    acct = I.load_cash_account(root, days)
    cash_ret = I.daily_cash_returns(acct, len(days))
    sig = L.risk_share_signal(panel.ret[L.DOM], panel.ret[L.US_MAIN])
    files = {
        "etf_daily": csv,
        "manifest_files": mf,
        "seibro_dist": djson,
        "dist_code_map": dmap,
        **{f"cd91_{p.parent.name}": p for p in I.cd91_files(root)},
    }
    return Prepared(root, days, panel, dist, cash_ret, sig, days.index(END_DATE), files)


def check(p: Prepared) -> tuple[dict, list[str]]:
    """계산 전 점검. (보고 dict, STOP 사유 목록)."""
    stops: list[str] = []
    rep: dict = {"etf": {}, "trading_days": len(p.days)}
    pos = {d: i for i, d in enumerate(p.days)}
    for c in CODES:
        starts = [FIRST_DECISION["main"]] + ([FIRST_DECISION["u2"]] if c == L.US_U2 else [])
        win_start = min(s for s in starts) if c != L.US_U2 else FIRST_DECISION["u2"]
        miss = [d for d in p.panel.missing_days[c] if d >= win_start]
        rep["etf"][c] = {
            "first_date": p.panel.first_date[c],
            "last_date": p.panel.last_date[c],
            "missing_trading_days_after_listing": len(p.panel.missing_days[c]),
            "missing_trading_days_in_window": len(miss),
            "dist_rows_added": p.panel.dist_added[c],
            "isin": p.dist.isin_by_code[c],
        }
        if miss:
            stops.append(f"{c}: 구간 안 종가 없는 거래일 {len(miss)}개 (첫 {miss[0]})")
        if p.panel.last_date[c] != END_DATE:
            stops.append(f"{c}: 마지막 날 {p.panel.last_date[c]} != {END_DATE}")
    rep["i2_split_candidates"] = p.panel.split_candidates
    rep["jumps_over_25pct_after_adjust"] = p.panel.jumps
    for j in p.panel.jumps:
        if j["date"] >= JUMP_STOP_FROM:
            stops.append(f"{j['code']} {j['date']} 하루 수익 {j['ret']:.1%}")
    rep["dist_counts_by_year"] = p.dist.counts
    rep["dist_unplaced"] = p.panel.dist_unplaced
    rep["dist_duplicates_dropped"] = p.dist.duplicates_dropped
    rep["dist_bad_amount_rows"] = p.dist.bad_amount_rows
    rep["dist_liquidation_rows_ignored"] = p.dist.liquidation_rows_ignored
    rep["dist_record_date_field"] = p.dist.record_date_field + " (분배 기준일), 금액 ESTM_STDPRC"
    for u, fd in FIRST_DECISION.items():
        s = pos[fd]
        win = np.arange(s + 1, p.end_idx + 1)
        null_win = int(np.isnan(p.cash_ret[win]).sum())
        ready = bool(not np.isnan(p.signal.sigma_target[s]))
        rep[f"window_{u}"] = {
            "first_decision": fd,
            "perf_days": len(win),
            "cd91_null_sessions_in_window": null_win,
            "sigma20_obs_at_first_decision": int(p.signal.n_obs[s]),
            "sigma_target_ready": ready,
            "w_vm_at_first_decision": None if not ready else float(p.signal.w_vm[s]),
        }
        if not ready:
            stops.append(
                f"{u}: 첫 결정일 {fd} σ_target 준비 안 됨 (σ_20 관측 {p.signal.n_obs[s]}개)"
            )
        if null_win:
            stops.append(f"{u}: CD91 null 세션 {null_win}개 (구간 안)")
    rep["cd91_null_sessions_total"] = int(np.isnan(p.cash_ret[1:]).sum())
    return rep, stops


def _git(*args: str) -> str:
    wt = Path(modeler.__file__).resolve().parents[2].parent
    return subprocess.run(
        ["git", "-C", str(wt), *args], capture_output=True, text=True
    ).stdout.strip()


def run_all(p: Prepared, out_dir: Path) -> dict:
    days = p.days
    results, nav_cols, yearly, rebs, wvm_rows = [], {}, [], [], []
    wfix_report: dict = {}
    for u, fd in FIRST_DECISION.items():
        us = L.US_MAIN if u == "main" else L.US_U2
        specs = L.line_specs(us)
        decisions = L.decision_schedule(days, fd, p.end_idx)
        start_idx = decisions[0] + 1
        for d in decisions:
            wvm_rows.append(
                {
                    "universe": u,
                    "decision_date": days[d],
                    "sigma20": p.signal.sigma20[d],
                    "sigma_target": p.signal.sigma_target[d],
                    "w_vm": p.signal.w_vm[d],
                }
            )
        for cname, one_way in COSTS.items():
            res_by_line = {}
            w_fix = None
            for name in ["B4", "B1", "B2", "B3", "R1", "R1i", "R2", "R2i", "B5"]:
                spec = specs[name]
                if name == "B5":
                    w4 = res_by_line["B4"].weights_held
                    w_fix = float((w4[:, 0] + w4[:, 1]).mean())
                    wfix_report[f"{u}_{cname}"] = w_fix
                tg = L.build_targets(spec, decisions, p.signal.w_vm, w_fix)
                mat = np.column_stack([p.panel.ret[c] for c in spec.assets])
                res = simulate(mat, p.cash_ret, start_idx, p.end_idx, tg, one_way)
                res_by_line[name] = res
                dates_ = [days[start_idx]] + [days[i] for i in res.idx]
                nav = np.concatenate([[1.0], res.nav])
                tos = [(days[r["idx"]], r["turnover"]) for r in res.rebalances]
                m = compute_metrics(dates_, nav, p.cash_ret[res.idx], tos)
                extra = {}
                if name == "B4":
                    extra["avg_w_vm_held"] = float(
                        (res.weights_held[:, 0] + res.weights_held[:, 1]).mean() / 0.6
                    )
                    extra["avg_equity_weight"] = float(
                        (res.weights_held[:, 0] + res.weights_held[:, 1]).mean()
                    )
                if name == "B5":
                    extra["w_fix"] = w_fix
                results.append(
                    {
                        "universe": u,
                        "cost": cname,
                        "round_trip_bp": one_way * 2e4,
                        "line": name,
                        **m,
                        **extra,
                    }
                )
                for y in yearly_table(dates_, nav):
                    yearly.append({"universe": u, "cost": cname, "line": name, **y})
                nav_cols[f"{u}|{cname}|{name}"] = (dates_, nav)
                for r in res.rebalances:
                    rebs.append(
                        {
                            "universe": u,
                            "cost_name": cname,
                            "line": name,
                            "date": days[r["idx"]],
                            "turnover_sum_abs_dw": r["turnover"],
                            "cost_frac": r["cost"],
                        }
                    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(out_dir / "metrics.csv")
    pl.DataFrame(yearly).write_csv(out_dir / "yearly.csv")
    pl.DataFrame(rebs).write_csv(out_dir / "rebalances.csv")
    pl.DataFrame(wvm_rows).write_csv(out_dir / "w_vm_signal.csv")
    # 일간 NAV: 긴 형태
    parts = []
    for key, (ds, nav) in nav_cols.items():
        u, c, ln = key.split("|")
        parts.append(pl.DataFrame({"universe": u, "cost": c, "line": ln, "date": ds, "nav": nav}))
    pl.concat(parts).write_parquet(out_dir / "nav_daily.parquet")
    return {"results": results, "wfix": wfix_report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    print("modeler.__file__ =", modeler.__file__)
    p = prepare()
    rep, stops = check(p)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    if stops:
        print("STOP 사유:")
        for s in stops:
            print(" -", s)
        return 2
    print("점검 통과 (STOP 사유 없음)")
    if a.cmd == "check":
        return 0
    started = datetime.now()
    run_id = a.run_id or started.strftime("run_%Y%m%d_%H%M%S")
    out_dir = p.root / I.OUTPUT_DIR / run_id
    out = run_all(p, out_dir)
    status = _git(
        "status",
        "--porcelain",
        "--",
        "src/modeler/scores/etf_baseline",
        "tests/scores/etf_baseline",
    )
    manifest = {
        "run_id": run_id,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "design_doc_commit": DESIGN_COMMIT,
        "code_commit": _git("rev-parse", "HEAD"),
        "code_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_dirty": bool(status),
        "modeler_file": modeler.__file__,
        "input_sha256": {k: I.sha256_file(v) for k, v in p.files.items()},
        "input_paths": {k: str(v) for k, v in p.files.items()},
        "params": {
            "end_date": END_DATE.isoformat(),
            "first_decision": {k: v.isoformat() for k, v in FIRST_DECISION.items()},
            "one_way_cost": COSTS,
            "split_ratio_lo_hi": [I.SPLIT_LO, I.SPLIT_HI],
            "shares_match_log_tol": I.SHARES_MATCH_LOG_TOL,
            "jump_limit": I.JUMP_LIMIT,
            "sigma_window": L.SIGMA_WINDOW,
            "sigma_ddof": 1,
            "sigma_target_min_obs": L.SIGMA_TARGET_MIN_OBS,
            "w_vm_clip": [L.VM_LO, L.VM_HI],
            "cd91_snapshot": I.CD91_SNAPSHOT,
            "cash_staleness_days": 14,
            "codes": CODES,
            "trading_days": len(p.days),
            "w_fix": out["wfix"],
        },
        "precheck": rep,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, default=str)
    )
    print("출력:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
