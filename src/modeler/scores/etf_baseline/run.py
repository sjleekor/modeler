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
from modeler.scores.etf_baseline import canonical as C
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
SOURCES = ("research", "canonical")
CANON_OUTPUT_DIR = "kr/output/etf_baseline_canonical"  # 정본 결과. 조사 사본 결과 폴더와 다르다


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
    end_date: date = END_DATE
    #: 원천 종류·정본 manifest 지문 등. 결과 manifest의 ``source``에 그대로 들어간다.
    source: dict | None = None
    #: 정본에서 읽은 parquet의 ``{상대 경로: sha256}`` (manifest 대조 뒤의 값)
    canonical_files: dict[str, str] | None = None


def prepare(
    source: str = "research",
    *,
    end_date: date = END_DATE,
    canonical_dir: Path | None = None,
    dist_source: str | None = None,
    obs_view: str = "latest",
    cd91_snapshot: str = I.CD91_SNAPSHOT,
) -> Prepared:
    """입력을 읽어 수익 패널까지 만든다.

    ``source=research``(기본)는 조사 사본, ``canonical``은 collector ``krx_baseline`` 정본에서
    ETF 일봉·거래일(요청 기록의 ``day_kind``)·분배금을 읽는다. ``dist_source``로 분배금만
    따로 고를 수 있다(기본은 ``source``와 같다). CD91 현금은 두 경우 모두 기존 경로다.
    """
    if source not in SOURCES:
        raise ValueError(f"source는 {SOURCES} 중 하나입니다: {source!r}")
    dist_source = dist_source or source
    if dist_source not in SOURCES:
        raise ValueError(f"dist_source는 {SOURCES} 중 하나입니다: {dist_source!r}")
    root = I.stock_data_root()
    mf = root / I.ETF_DIR / I.MANIFEST_FILE
    csv = root / I.ETF_DIR / I.ETF_FILE
    djson, dmap = root / I.DIST_FILE, root / I.MAP_FILE
    files: dict[str, Path] = {}
    src: dict = {"kind": source, "dist_kind": dist_source, "end_date": end_date.isoformat()}
    store = None
    if source == "canonical" or dist_source == "canonical":
        store = C.CanonicalStore(C.canonical_base(root, canonical_dir))
        src.update(
            canonical_base=str(store.base),
            canonical_manifests=store.manifest_digest(),
            obs_view=obs_view,
        )
    if source == "research":
        days = [d for d in I.load_trading_days(mf) if d <= end_date]
        frames = I.load_etf_frames(csv, CODES)
        files.update({"etf_daily": csv, "manifest_files": mf})
    else:
        days, cal = C.load_trading_days(store, end_date)
        frames = C.load_etf_frames(store, CODES, obs_view)
        src["calendar"] = cal
    if dist_source == "research":
        dist = I.load_distributions(djson, dmap, CODES)
        files.update({"seibro_dist": djson, "dist_code_map": dmap})
    else:
        dist, win = C.load_distributions(store, CODES, end_date, obs_view)
        src["dist_windows"] = win
    if end_date not in set(days):
        raise I.InputError(f"끝 날짜 {end_date}가 거래일 목록에 없습니다")
    panel = I.build_return_panel(days, frames, dist.rows)
    acct = I.load_cash_account(root, days, cd91_snapshot)
    cash_ret = I.daily_cash_returns(acct, len(days))
    sig = L.risk_share_signal(panel.ret[L.DOM], panel.ret[L.US_MAIN])
    files.update({f"cd91_{p.parent.name}": p for p in I.cd91_files(root, cd91_snapshot)})
    src["cd91_snapshot"] = cd91_snapshot
    return Prepared(
        root,
        days,
        panel,
        dist,
        cash_ret,
        sig,
        days.index(end_date),
        files,
        end_date,
        src,
        None if store is None else store.verified,
    )


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
        if p.panel.last_date[c] != p.end_date:
            stops.append(f"{c}: 마지막 날 {p.panel.last_date[c]} != {p.end_date}")
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
    wt = Path(modeler.__file__).resolve().parents[2]
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


def add_source_args(ap: argparse.ArgumentParser) -> None:
    """입력 원천을 고르는 인자. 비교 도구(compare.py)와 공유한다."""
    ap.add_argument(
        "--source",
        choices=SOURCES,
        default="research",
        help="ETF 일봉·거래일 원천. research=조사 사본(기본), canonical=krx_baseline 정본",
    )
    ap.add_argument(
        "--dist-source",
        choices=SOURCES,
        default=None,
        help="분배금 원천. 기본은 --source와 같다 (혼합 모드는 따로 고른다)",
    )
    ap.add_argument(
        "--canonical-dir",
        type=Path,
        default=None,
        help="정본 저장소 경로. 기본은 $STOCK_DATA_ROOT/kr/raw/krx_baseline",
    )
    ap.add_argument(
        "--obs-view",
        choices=C.VIEWS,
        default="latest",
        help="정본 관측판. latest=키별 최신(기본), first=obs_seq=1 최초 관측본",
    )
    ap.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=END_DATE,
        help=f"기준선 끝 날짜(YYYY-MM-DD). 기본 {END_DATE}",
    )
    ap.add_argument(
        "--cd91-snapshot", default=I.CD91_SNAPSHOT, help="CD91 raw_postgres 스냅샷 날짜"
    )


def prepare_from_args(a: argparse.Namespace) -> Prepared:
    return prepare(
        a.source,
        end_date=a.end_date,
        canonical_dir=a.canonical_dir,
        dist_source=a.dist_source,
        obs_view=a.obs_view,
        cd91_snapshot=a.cd91_snapshot,
    )


def resolve_out_dir(p: Prepared, run_id: str, output_dir: Path | None) -> Path:
    """결과 폴더. 정본 모드는 조사 사본 결과 폴더와 다른 곳이고, 이미 있으면 덮지 않는다."""
    canonical = p.source is not None and (
        p.source["kind"] == "canonical" or p.source["dist_kind"] == "canonical"
    )
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = p.root / (CANON_OUTPUT_DIR if canonical else I.OUTPUT_DIR) / run_id
    if canonical:
        frozen = (p.root / I.OUTPUT_DIR).resolve()
        if out.resolve() == frozen or frozen in out.resolve().parents:
            raise I.InputError(f"정본 결과를 조사 사본 결과 폴더({frozen}) 안에 쓸 수 없습니다")
        if out.exists() and any(out.iterdir()):
            raise I.InputError(f"{out}: 이미 결과가 있습니다. 다른 --run-id를 쓰십시오")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--output-dir", type=Path, default=None, help="결과 폴더를 직접 지정 (기본은 원천별 폴더)"
    )
    add_source_args(ap)
    a = ap.parse_args(argv)
    print("modeler.__file__ =", modeler.__file__)
    p = prepare_from_args(a)
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
    canonical = p.source["kind"] == "canonical" or p.source["dist_kind"] == "canonical"
    run_id = a.run_id or started.strftime(
        "canon_%Y%m%d_%H%M%S" if canonical else "run_%Y%m%d_%H%M%S"
    )
    out_dir = resolve_out_dir(p, run_id, a.output_dir)
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
        "source": p.source,
        "input_sha256": {
            **{k: I.sha256_file(v) for k, v in p.files.items()},
            **{f"canonical:{k}": v for k, v in (p.canonical_files or {}).items()},
        },
        "input_paths": {k: str(v) for k, v in p.files.items()},
        "params": {
            "end_date": p.end_date.isoformat(),
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
