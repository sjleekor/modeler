"""MRS v1 실행기: 권한 확인, KR·US 점수와 검정 조립, 산출물 (사양 §4.5·§4.6·§5.4).

CLI::

    python -m modeler.scores.mrs.run readiness --market kr|us|all [--kr-snapshot D]
    python -m modeler.scores.mrs.run run --market kr|us|all --kr-snapshot D [--us-snapshot D] \\
        --approved-interp PATH --confirm-run YYYY-MM-DD [--allow-dirty]

**권한 확인이 먼저다 (MI32).** 값 열을 읽기 전에 아래를 모두 통과해야 하고, 하나라도 어긋나면
아무것도 계산하지 않고 출력 디렉터리도 만들지 않은 채 종료 코드 2로 끝난다.

1. ``APPROVED_INTERP_SHA256``가 ``None``이 아니다 (메인이 구현 해석 표를 승인하면 채운다).
2. ``--approved-interp`` 파일의 sha256이 그 상수와 같다.
3. ``--confirm-run``이 오늘(Asia/Seoul) 날짜다.
4. 코드 트리가 깨끗하다(``--allow-dirty``면 manifest에 ``-dirty`` 표시).
5. (kr·all) 선행 백필 확인이 PASS다 — 관측일만 보는 ``readiness.backfill_check``.
6. 출력 디렉터리가 아직 없다 (덮어쓰지 않는다, D-8).

산출물: ``stock_data/{kr,us}/output/regime_score/<스냅샷>/`` — KR은 KR 스냅샷, US는 US 가격 스냅샷.
``scores.parquet``, ``first_dates.json``, ``ledger.parquet``, ``tests.parquet``, ``tests.json``,
``synth.parquet``(KR), ``manifest.json``, ``report.md``.

공식 등급(MI22)은 ``--market all``에서만 채운다(KR 주 판정 행에). 한쪽만 돌리면 미국 탐색·한국 탐색
행이 없어 등급을 못 정하므로 ``grade``는 null이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from modeler.etl.config import REPO_ROOT, DataRoot
from modeler.scores.common.kr_inputs import KrNotSyncedError
from modeler.scores.mrs import backtest as bt
from modeler.scores.mrs import config, synth_ktb
from modeler.scores.mrs import readiness as rd
from modeler.scores.mrs.inputs import (
    US_TABLES_FOR_KR,
    KrInputs,
    UsInputs,
    load_kr_inputs,
    load_us_inputs,
    pin_us_lake,
    resolve_kr_lake,
)
from modeler.scores.mrs.score import ScoreResult, asset_sigma20, compute_scores
from modeler.scores.mrs.vintage import Grid as EngineGrid
from modeler.us.dataset import DirtyWorktreeError, git_commit

logger = logging.getLogger(__name__)

#: 승인된 구현 해석 표(04 문서)의 sha256. ``None``이면 실행은 항상 거부된다 (MI32).
APPROVED_INTERP_SHA256: str | None = None

_KST = ZoneInfo("Asia/Seoul")
EXIT_REFUSED = 2
OUTPUT_SUBDIR = "regime_score"

#: 합성(추정) 칸의 한계 문장 (문면 §0-5.6, 그대로 옮긴다).
SYNTH_LIMITATION = (
    "위험자산 쪽은 KOSPI 가격지수라 배당·보수·추적오차가 없다. "
    "국고채 쪽은 금리 합성이라 롤다운·재투자·지표물 차이가 없다. 두 쪽 모두 수준 대용이다."
)
SYNTH_TAG = "합성(추정)"
ORIG_OPEN_REASON = "이 구간에 정의하지 않음 (§0-1 M3)"

#: (protocol, 왕복 비용 bp, 현금 종류, 점수 protocol, IRP 상한). 현금 ``zero``는 ``sens_cash0``.
KR_PROTOCOLS: tuple[tuple[str, float, str, str, float | None], ...] = (
    (config.P_MAIN, config.COST_RT_BP["KR"], "main", "main", None),
    (config.P_CASH0, config.COST_RT_BP["KR"], "zero", "main", None),
    (config.P_COST120, config.COST_RT_BP_SENS_KR, "main", "main", None),
    (config.P_VIX_PROXY, config.COST_RT_BP["KR"], "main", "vix_proxy", None),
    (config.P_IRP, config.COST_RT_BP["KR"], "main", "main", config.IRP_CAP),
)
US_PROTOCOLS: tuple[tuple[str, float, str, str, float | None], ...] = (
    (config.P_MAIN, config.COST_RT_BP["US"], "main", "main", None),
    (config.P_CASH0, config.COST_RT_BP["US"], "zero", "main", None),
)


class RunRefused(RuntimeError):
    """실행을 거부한다. 메시지가 사유다 (종료 코드 2)."""


@dataclass
class MarketRun:
    """한 시장의 실행 결과 묶음 (파일로 쓰기 전)."""

    market: str
    grid_dates: list[date]
    scores: pl.DataFrame
    first_dates: dict[str, list[dict[str, Any]]]
    diagnostics: dict[str, Any]
    tests: list[dict[str, Any]]
    ledgers: list[pl.DataFrame]
    synth: list[dict[str, Any]] = field(default_factory=list)
    protocols_run: list[str] = field(default_factory=list)
    protocols_not_run: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- 권한
def _today_kst() -> date:
    return datetime.now(_KST).date()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def authorize(args: argparse.Namespace) -> tuple[str, list[str]]:
    """권한 확인 1~5. 통과하면 ``(git 커밋 문자열, [])``, 어긋나면 ``RunRefused``.

    값 열을 읽기 전이다(백필 확인은 관측일만 남기고 값을 바로 버린다).
    """
    if APPROVED_INTERP_SHA256 is None:
        raise RunRefused("APPROVED_INTERP_SHA256이 None이다 — 해석 표 승인 전에는 실행하지 않는다")
    ap = Path(args.approved_interp) if args.approved_interp else None
    if ap is None or not ap.is_file():
        raise RunRefused(f"--approved-interp 파일이 없다: {args.approved_interp}")
    got = sha256_path(ap)
    if got != APPROVED_INTERP_SHA256:
        raise RunRefused(f"승인 파일 sha256이 다르다 ({got[:12]} != {APPROVED_INTERP_SHA256[:12]})")
    today = _today_kst()
    if args.confirm_run != today.isoformat():
        raise RunRefused(
            f"--confirm-run {args.confirm_run}이 오늘(KST) {today.isoformat()}와 다르다"
        )
    try:
        commit = git_commit(REPO_ROOT, allow_dirty=args.allow_dirty)
    except DirtyWorktreeError as exc:
        raise RunRefused(f"코드 트리가 깨끗하지 않다: {str(exc).splitlines()[0]}") from exc
    if args.market in ("kr", "all"):
        try:
            lake = resolve_kr_lake(DataRoot.resolve("kr"), args.kr_snapshot)
        except (KrNotSyncedError, FileNotFoundError) as exc:
            raise RunRefused(f"KR 레이크를 열 수 없다: {exc}") from exc
        obs: dict[str, list[date] | None] = {}
        grid = rd.build_kr_grid(lake)
        for sid in config.BACKFILL_REQUIRED_START:
            got_d = rd._kr_dates(lake, sid, grid)
            obs[sid] = got_d[0] if got_d else None
        bf = rd.backfill_check(obs)
        if bf["overall"] != "PASS":
            failed = [k for k, v in bf["series"].items() if not v["pass"]]
            raise RunRefused(f"선행 백필 확인 FAIL: {failed}")
    return commit, []


# --------------------------------------------------------------------------- 작은 도구
def _arr(s: pl.Series) -> np.ndarray:
    return s.cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)


def _score_arrays(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "mrs": _arr(frame["MRS"]),
        "subs": {k: _arr(frame[f"sub_{k}"]) for k in config.SUB_SCORES},
        "avail": frame["availability_ok"].fill_null(True).to_numpy().astype(bool),
    }


def _first_dates_json(res: ScoreResult) -> list[dict[str, Any]]:
    return [
        {k: (v.isoformat() if isinstance(v, date) else v) for k, v in row.items()}
        for row in res.first_dates.iter_rows(named=True)
    ]


def _json_default(o: Any) -> Any:
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON으로 못 쓴다: {type(o)}")


def config_dump() -> dict[str, Any]:
    """``config``의 대문자 상수 전체."""
    return {k: v for k, v in vars(config).items() if k.isupper() and not k.startswith("_")}


def _rows_to_frame(rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    return pl.from_dicts(list(rows), infer_schema_length=None)


def _sigma(vg: EngineGrid, price_rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = asset_sigma20(vg, price_rows)
    return _arr(s["sigma20"]), _arr(s["sigma_target"])


def _ledger_frame(led: bt.Ledger, market: str, asset: str, protocol: str) -> pl.DataFrame:
    return led.to_frame().with_columns(
        pl.lit(market).alias("market"),
        pl.lit(asset).alias("asset"),
        pl.lit(protocol).alias("protocol"),
    )


def _concat_scores(results: dict[str, ScoreResult]) -> pl.DataFrame:
    return pl.concat(
        [r.frame.with_columns(pl.lit(k).alias("score_protocol")) for k, r in results.items()],
        how="vertical_relaxed",
    )


# --------------------------------------------------------------------------- 시장별 실행
def _evaluate(
    *,
    market: str,
    asset: str,
    period: str,
    protocol: str,
    start: date,
    end: date,
    cost_bp: float,
    cash: np.ndarray,
    score: dict[str, Any],
    dates: list[date],
    price: np.ndarray,
    sigma: tuple[np.ndarray, np.ndarray],
    irp_cap: float | None = None,
    with_gates: bool = True,
    synthetic: bool = False,
) -> list[dict[str, Any]]:
    return bt.evaluate_window(
        dates=dates,
        price=price,
        mrs=score["mrs"],
        sub_scores=score["subs"],
        sigma20=sigma[0],
        sigma_target=sigma[1],
        r_cash=cash,
        start=start,
        end=end,
        cost_rt_bp=cost_bp,
        irp_cap=irp_cap,
        with_gates=with_gates,
        availability_ok=score["avail"],
        meta={
            "market": market,
            "asset": asset,
            "period": period,
            "protocol": protocol,
            "synthetic": synthetic,
        },
    )


def run_kr(kr: KrInputs) -> MarketRun:
    """KR 점수(main·vix_proxy), 구간·자산·protocol별 검정, 합성(추정) 혼합 칸."""
    if kr.missing_series:
        raise RunRefused(f"KR 입력 계열이 없다: {list(kr.missing_series)}")
    dates = list(kr.grid.dates)
    n = len(dates)
    vg = EngineGrid.from_lists(dates, kr.grid.decision_at)
    results = {
        "main": compute_scores("KR", vg, kr.series),
        "vix_proxy": compute_scores("KR", vg, {**kr.series, "VIXCLS": kr.vix_proxy}),
    }
    scores = {k: _score_arrays(r.frame) for k, r in results.items()}
    sig = {
        "kr_kospi": _sigma(vg, kr.series["market_kospi_ecos"]),
        "kr_kosdaq": _sigma(vg, kr.series["market_kosdaq_ecos"]),  # MI16
    }
    price = {a: kr.realized[a] for a in ("kr_kospi", "kr_kosdaq")}
    cash = {"main": bt.cash_returns(kr.cash, n), "zero": bt.zero_cash(n)}
    last = dates[-1]
    windows = [
        ("main", config.KR_MAIN_START, config.KR_MAIN_END, ("kr_kospi",)),
        ("explore", config.KR_EXPLORE_START, last, ("kr_kospi", "kr_kosdaq")),
    ]
    tests: list[dict[str, Any]] = []
    ledgers: list[pl.DataFrame] = []
    for asset in ("kr_kospi", "kr_kosdaq"):
        for proto, cost, cash_kind, sp, irp in KR_PROTOCOLS:
            led = bt.build_ledger(
                dates=dates,
                price=price[asset],
                mrs=scores[sp]["mrs"],
                sigma20=sig[asset][0],
                sigma_target=sig[asset][1],
                r_cash=cash[cash_kind],
                cost_rt_bp=cost,
                irp_cap=irp,
            )
            ledgers.append(_ledger_frame(led, "KR", asset, proto))
    for period, start, end, assets in windows:
        for asset in assets:
            for proto, cost, cash_kind, sp, irp in KR_PROTOCOLS:
                tests += _evaluate(
                    market="KR", asset=asset, period=period, protocol=proto, start=start, end=end,
                    cost_bp=cost, cash=cash[cash_kind], score=scores[sp], dates=dates,
                    price=price[asset], sigma=sig[asset], irp_cap=irp,
                )  # fmt: skip
    # 합성(추정) 국고채 현금 민감도: 주 판정 구간·KOSPI만, 게이트·등급 없음 (MI25)
    r_ktb3 = synth_ktb.synth_returns(dates, kr.series["rate_kr_gov3y"], 3)
    r_ktb10 = synth_ktb.synth_returns(dates, kr.series["rate_kr_gov10y"], 10)
    ledgers.append(_ledger_frame(
        bt.build_ledger(
            dates=dates,
            price=price["kr_kospi"],
            mrs=scores["main"]["mrs"],
            sigma20=sig["kr_kospi"][0],
            sigma_target=sig["kr_kospi"][1],
            r_cash=r_ktb3,
            cost_rt_bp=config.COST_RT_BP["KR"],
            ),  # fmt: skip
            "KR",
            "kr_kospi",
            config.P_KTB_SYNTH,
        )
    )
    tests += _evaluate(
        market="KR", asset="kr_kospi", period="main", protocol=config.P_KTB_SYNTH,
        start=config.KR_MAIN_START, end=config.KR_MAIN_END, cost_bp=config.COST_RT_BP["KR"],
        cash=r_ktb3, score=scores["main"], dates=dates, price=price["kr_kospi"],
        sigma=sig["kr_kospi"], with_gates=False, synthetic=True,
    )  # fmt: skip
    synth = _synth_tables(dates, price["kr_kospi"], scores["main"]["mrs"], sig["kr_kospi"],
                          cash["main"], r_ktb3, r_ktb10)  # fmt: skip
    return MarketRun(
        market="KR",
        grid_dates=dates,
        scores=_concat_scores(results),
        first_dates={k: _first_dates_json(r) for k, r in results.items()},
        diagnostics={k: r.diagnostics for k, r in results.items()},
        tests=tests,
        ledgers=ledgers,
        synth=synth,
        protocols_run=[p[0] for p in KR_PROTOCOLS] + [config.P_KTB_SYNTH],
        protocols_not_run={config.P_ORIG_OPEN: ORIG_OPEN_REASON},
    )


def _synth_tables(
    dates: list[date],
    price: np.ndarray,
    mrs: np.ndarray,
    sigma: tuple[np.ndarray, np.ndarray],
    r_cash: np.ndarray,
    r_ktb3: np.ndarray,
    r_ktb10: np.ndarray,
) -> list[dict[str, Any]]:
    """합성(추정) 혼합 칸: 혼합·보유·MRS(main)의 MDD·CAGR·Calmar를 같은 세션 집합에서 낸다."""
    r_bh = bt.forward_returns(price)
    led = bt.build_ledger(
        dates=dates, price=price, mrs=mrs, sigma20=sigma[0], sigma_target=sigma[1],
        r_cash=r_cash, cost_rt_bp=config.COST_RT_BP["KR"],
    )  # fmt: skip
    specs = [
        ("mix_synth_ps1", r_ktb3, config.KR_MAIN_START),
        ("mix_synth_ps2", r_ktb10, config.MIX_SYNTH_PS2_START),
    ]
    out: list[dict[str, Any]] = []
    for name, r_ktb, start in specs:
        mix = synth_ktb.mix_synth(name, dates, r_bh, r_ktb)
        _, idx = bt.window_session_set(dates, mrs, start, config.KR_MAIN_END)
        ok = np.isfinite(mix[idx]) & np.isfinite(r_bh[idx]) & np.isfinite(r_cash[idx])
        s = idx[ok]
        if s.size == 0:
            continue
        first, last_next = dates[int(s[0])], dates[int(s[-1]) + 1]
        for series, r in (("mix", mix), ("buy_hold", r_bh), ("mrs_main_close_cash", led.r["mrs"])):
            m = bt.path_metrics(r[s], first, last_next)
            out.append(
                {
                    "table": name,
                    "series": series,
                    "synthetic": series == "mix",
                    "MDD": m["MDD"],
                    "CAGR": m["CAGR"],
                    "Calmar": m["Calmar"],
                    "n_sessions": int(s.size),
                    "n_excluded": int(idx.size - s.size),
                    "first_session": first,
                    "last_session": dates[int(s[-1])],
                }
            )
    return out


def run_us(us: UsInputs) -> MarketRun:
    """US 점수(main)와 탐색 구간 검정 (protocol은 main·cash0만, MI24)."""
    if us.missing_series:
        raise RunRefused(f"US 입력 계열이 없다: {list(us.missing_series)}")
    dates = list(us.grid.dates)
    n = len(dates)
    vg = EngineGrid.from_lists(dates, us.grid.decision_at)
    res = compute_scores("US", vg, us.series)
    score = _score_arrays(res.frame)
    sig = _sigma(vg, us.series["tr_index"])
    price = us.realized["us_spx"]
    cash = {"main": bt.cash_returns(us.cash, n), "zero": bt.zero_cash(n)}
    tests: list[dict[str, Any]] = []
    ledgers: list[pl.DataFrame] = []
    for proto, cost, cash_kind, _sp, irp in US_PROTOCOLS:
        led = bt.build_ledger(
            dates=dates, price=price, mrs=score["mrs"], sigma20=sig[0], sigma_target=sig[1],
            r_cash=cash[cash_kind], cost_rt_bp=cost, irp_cap=irp,
        )  # fmt: skip
        ledgers.append(_ledger_frame(led, "US", "us_spx", proto))
        tests += _evaluate(
            market="US", asset="us_spx", period="explore", protocol=proto,
            start=config.US_EXPLORE_START, end=dates[-1], cost_bp=cost, cash=cash[cash_kind],
            score=score, dates=dates, price=price, sigma=sig,
        )  # fmt: skip
    not_run = {
        p: "KR만 돌린다 (MI24)"
        for p in (config.P_COST120, config.P_VIX_PROXY, config.P_IRP, config.P_KTB_SYNTH)
    }
    not_run[config.P_ORIG_OPEN] = ORIG_OPEN_REASON
    return MarketRun(
        market="US",
        grid_dates=dates,
        scores=_concat_scores({"main": res}),
        first_dates={"main": _first_dates_json(res)},
        diagnostics={"main": res.diagnostics},
        tests=tests,
        ledgers=ledgers,
        protocols_run=[p[0] for p in US_PROTOCOLS],
        protocols_not_run=not_run,
    )


def _find(
    rows: Sequence[dict[str, Any]], market: str, asset: str, period: str, protocol: str
) -> dict[str, Any] | None:
    for r in rows:
        if (
            r["market"] == market and r["asset"] == asset and r["period"] == period
            and r["protocol"] == protocol and r["rule"] == "mrs"
        ):  # fmt: skip
            return r
    return None


def apply_official_grade(kr: MarketRun, us: MarketRun) -> str | None:
    """KR 주 판정 행에 공식 등급을 채운다 (MI22). 등급을 돌려준다."""
    kr_main = _find(kr.tests, "KR", "kr_kospi", "main", config.P_MAIN)
    kr_exp = _find(kr.tests, "KR", "kr_kospi", "explore", config.P_MAIN)
    us_exp = _find(us.tests, "US", "us_spx", "explore", config.P_MAIN)
    g = bt.official_grade(kr_main, us_exp, kr_exp)
    if kr_main is not None:
        kr_main["grade"] = g
    return g


# --------------------------------------------------------------------------- 보고서
def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _tests_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    head = ["구간", "자산", "protocol", "규칙", "MDD비", "수익비", "Calmar비", "회전율"]
    head += ["placebo p", "등급문자", "공식등급", "세션"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        proto = r["protocol"] + (f" ({SYNTH_TAG})" if r.get("synthetic") else "")
        lines.append(
            "| " + " | ".join([
                r["period"], r["asset"], proto, r["rule"], _f(r["MDD_ratio"]), _f(r["RET_ratio"]),
                _f(r["Calmar_ratio"]), _f(r["turnover"], 2), _f(r["placebo_p"]),
                _f(r["gate_class"]), _f(r.get("grade")), _f(r["n_sessions"]),
            ]) + " |"
        )  # fmt: skip
    return lines


def render_report(mr: MarketRun, *, grade_line: str, snapshot: dict[str, str]) -> str:
    L = [f"# MRS v1 실행 결과 — {mr.market}", ""]
    L += [f"입력 스냅샷: {json.dumps(snapshot, ensure_ascii=False)}", ""]
    L += ["## 공식 등급", "", grade_line, ""]
    L += ["## 검정 (mrs / vm / vm_nofloor 규칙, protocol별)", ""]
    L += _tests_table(mr.tests) + [""]
    L += ["## 성분별 첫 날짜", ""]
    for proto, rows in mr.first_dates.items():
        L += [f"점수 protocol `{proto}`", ""]
        L += ["| 성분 | 하위 | 첫 값 | 첫 백분위 |", "|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['component']} | {r['sub']} | {r['first_value_date']} "
                f"| {r['first_pct_date']} |"
            )
        L.append("")
    if mr.synth:
        L += [f"## 혼합 대용 — {SYNTH_TAG}", ""]
        for name in ("mix_synth_ps1", "mix_synth_ps2"):
            sub = [r for r in mr.synth if r["table"] == name]
            if not sub:
                continue
            L += [f"### {name} ({SYNTH_TAG})", "",
                  f"| 계열 ({SYNTH_TAG} 표) | MDD | CAGR | Calmar | 세션 | 첫 세션 | 끝 세션 |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
            for r in sub:
                tag = f" ({SYNTH_TAG})" if r["synthetic"] else ""
                L.append(
                    f"| {r['series']}{tag} | {_f(r['MDD'])} | {_f(r['CAGR'])} | {_f(r['Calmar'])} "
                    f"| {r['n_sessions']} | {r['first_session']} | {r['last_session']} |"
                )
            L.append("")
        L += [SYNTH_LIMITATION, ""]
    L += ["## 돌리지 않은 protocol", ""]
    L += [f"- `{k}`: {v}" for k, v in mr.protocols_not_run.items()] + [""]
    return "\n".join(L)


# --------------------------------------------------------------------------- 산출물
def write_outputs(
    mr: MarketRun,
    out_dir: Path,
    *,
    manifest: dict[str, Any],
    report: str,
) -> Path:
    """``out_dir``이 있으면 거부. 임시 디렉터리에 다 쓴 뒤 옮긴다."""
    if out_dir.exists():
        raise RunRefused(f"출력 디렉터리가 이미 있다 (덮어쓰지 않는다): {out_dir}")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mr.scores.write_parquet(tmp / "scores.parquet")
    (tmp / "first_dates.json").write_text(
        json.dumps(mr.first_dates, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    )
    pl.concat(mr.ledgers, how="vertical_relaxed").write_parquet(tmp / "ledger.parquet")
    tests = _rows_to_frame(mr.tests)
    tests.write_parquet(tmp / "tests.parquet")
    (tmp / "tests.json").write_text(
        json.dumps(mr.tests, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    )
    if mr.synth:
        _rows_to_frame(mr.synth).write_parquet(tmp / "synth.parquet")
    (tmp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        + "\n"
    )
    (tmp / "report.md").write_text(report)
    tmp.rename(out_dir)
    return out_dir


def _manifest(
    mr: MarketRun,
    *,
    snapshot: dict[str, str],
    files: Sequence[Any],
    commit: str,
    args: argparse.Namespace,
    wall: float,
    grade: str | None,
    grade_note: str,
) -> dict[str, Any]:
    return {
        "market": mr.market,
        "snapshot": snapshot,
        "input_files": [
            {"table": f.table, "path": f.path, "bytes": f.bytes, "sha256": f.sha256} for f in files
        ],
        "modeler_git_commit": commit,
        "config": config_dump(),
        "approved_interp": {
            "path": str(args.approved_interp),
            "sha256": APPROVED_INTERP_SHA256,
        },
        "confirm_run": args.confirm_run,
        "placebo_shifts": bt.placebo_shifts(),
        "engine_diagnostics": mr.diagnostics,
        "protocols_run": mr.protocols_run,
        "protocols_not_run": mr.protocols_not_run,
        "excluded_sessions": {
            f"{r['asset']}|{r['period']}|{r['protocol']}|{r['rule']}": r["n_excluded"]
            for r in mr.tests
        },
        "official_grade": grade,
        "official_grade_note": grade_note,
        "wall_time_sec": round(wall, 2),
    }


# --------------------------------------------------------------------------- 실행
def run_all(args: argparse.Namespace) -> list[Path]:
    """권한 확인 -> 입력 -> 계산 -> 파일. 거부는 ``RunRefused``."""
    t_start = time.perf_counter()
    commit, _ = authorize(args)
    markets = ["kr", "us"] if args.market == "all" else [args.market]
    kr_root, us_root = DataRoot.resolve("kr"), DataRoot.resolve("us")
    kr_lake = resolve_kr_lake(kr_root, args.kr_snapshot) if "kr" in markets else None
    if "us" in markets:
        us_lake = pin_us_lake(us_root, snapshot_date=args.us_snapshot)
    else:
        us_lake = pin_us_lake(us_root, US_TABLES_FOR_KR, snapshot_date=args.us_snapshot, symbols=())
    out_dirs: dict[str, Path] = {}
    if kr_lake is not None:
        out_dirs["kr"] = kr_root.output / OUTPUT_SUBDIR / kr_lake.snapshot_date
    if "us" in markets:
        out_dirs["us"] = us_root.output / OUTPUT_SUBDIR / us_lake.snapshots["prices_daily"]
    for d in out_dirs.values():
        if d.exists():
            raise RunRefused(f"출력 디렉터리가 이미 있다 (덮어쓰지 않는다): {d}")

    runs: dict[str, MarketRun] = {}
    snaps: dict[str, dict[str, str]] = {}
    files: dict[str, Sequence[Any]] = {}
    if kr_lake is not None:
        kin = load_kr_inputs(kr_lake, us_lake)
        runs["kr"], snaps["kr"], files["kr"] = run_kr(kin), kin.snapshot, kin.input_files
    if "us" in markets:
        uin = load_us_inputs(us_lake)
        runs["us"], snaps["us"], files["us"] = run_us(uin), uin.snapshot, uin.input_files

    grade: str | None = None
    if args.market == "all":
        grade = apply_official_grade(runs["kr"], runs["us"])
        note = "KR 주 판정·US 탐색·KR 탐색 행으로 정했다 (MI22)"
    else:
        note = "--market all이 아니라 한쪽 탐색 행이 없어 공식 등급을 정하지 않는다 (MI22)"
    wall = time.perf_counter() - t_start
    written = []
    for m, mr in runs.items():
        line = (
            f"공식 등급: **{grade}**" if (m == "kr" and grade is not None)
            else ("공식 등급은 KR 결과에 있다." if grade is not None
                  else f"공식 등급 없음 — {note}")
        )  # fmt: skip
        man = _manifest(mr, snapshot=snaps[m], files=files[m], commit=commit, args=args,
                        wall=wall, grade=grade, grade_note=note)  # fmt: skip
        written.append(
            write_outputs(mr, out_dirs[m], manifest=man,
                          report=render_report(mr, grade_line=line, snapshot=snaps[m]))  # fmt: skip
        )
    return written


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("readiness", help="입력 준비 상태 (값 없이 개수·날짜만)")
    r.add_argument("--market", choices=["kr", "us", "all"], required=True)
    r.add_argument("--kr-snapshot", default=None)
    r.add_argument("--no-expected", action="store_true")
    x = sub.add_parser("run", help="실제 실행 (권한 확인 필요)")
    x.add_argument("--market", choices=["kr", "us", "all"], required=True)
    x.add_argument("--kr-snapshot", default=None)
    x.add_argument("--us-snapshot", default=None)
    x.add_argument("--approved-interp", default=None)
    x.add_argument("--confirm-run", default=None)
    x.add_argument("--allow-dirty", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "readiness":
        return rd.main(
            ["--market", args.market]
            + (["--kr-snapshot", args.kr_snapshot] if args.kr_snapshot else [])
            + (["--no-expected"] if args.no_expected else [])
        )
    try:
        paths = run_all(args)
    except RunRefused as exc:
        logger.error("실행 거부: %s", exc)
        print(f"실행 거부: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for p in paths:
        print(f"-> {p}")
    return 0


__all__ = ["APPROVED_INTERP_SHA256", "RunRefused", "main", "run_all"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
