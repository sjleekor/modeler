"""universe_retest — B.3: the adopted run's saved predictions, re-read on a large-cap universe.

Plan: ``my/milestones/kr/modeling/plan/20261006_universe_retest.md`` (frozen, tag
``kr-universe-retest-frozen`` in the ``my`` repository). Read §4 (U), §6 (what is
recorded), §7 (the "universe equal-weight" definition check) and §8 (this file)
before changing anything here.

**Nothing is refit.** The holdout predictions the adopted run saved are filtered
to U — KOSPI top 200 and KOSDAQ top 150 by ``mcap_krx``, ranked per trading day,
then the existing ``tradable`` flag on top — and the top-100 return is measured
again. ``holdout_run`` and the files in ``MODEL_CODE_FILES`` are imported, never
edited; ``gate_metrics`` is reused as it is, with the U equal-weight excess put in
the slot of the equal-weight label and the original kept as ``raw_label_*_orig``.

**It refuses to compute a U return before the plan is frozen.** Every function that
turns the U mask (or the §7 definition check) into a return calls ``assert_sealed``
first, which looks for the tag in the ``my`` repository. Writing the code is
allowed before the freeze; reading its answer is not.

Modes (the default fits nothing and computes no return)::

    # seal, code hash and input paths only
    python -m modeler.models._02_updown_prob.experiments.universe_retest --check
    # the plain-`tradable` reproduction of E, E', I, S, D (no U). Published numbers.
    python -m modeler.models._02_updown_prob.experiments.universe_retest --reproduce
    # B.3. Needs the flag; writes under results/universe_retest/, never holdout/.
    python -m modeler.models._02_updown_prob.experiments.universe_retest \\
        --run --i-am-running-the-retest
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

import duckdb
import polars as pl

from modeler.etl import labels as labels_mod
from modeler.etl.manifest import current_git_sha
from modeler.etl.metrics import rebalance_grid, topk_economic_report, topk_rebalance_series
from modeler.models._02_updown_prob.experiments import holdout_run as hr
from modeler.models._02_updown_prob.experiments import preopen_measures as pm
from modeler.models._02_updown_prob.experiments.run_matrix import (
    MODEL_CODE_FILES,
    RESULTS_ROOT,
    code_hash,
    model_code_hash,
)

FROZEN_TAG = "kr-universe-retest-frozen"

#: §4. Per-market counts, taken from the index names (KOSPI200, KOSDAQ150) — not tuned.
U_SIZE: dict[str, int] = {"KOSPI": 200, "KOSDAQ": 150}

#: §8: a folder of its own. Never ``holdout/E2_h20_FS1h_seed0/``.
RETEST_DIR = RESULTS_ROOT / "universe_retest"

#: §6 record 5: a one-day close ratio outside this band inside a label window.
JUMP_UP, JUMP_DOWN = 1.5, 1.0 / 1.5

PLAN_FILE = "milestones/kr/modeling/plan/20261006_universe_retest.md"


class SealError(RuntimeError):
    """The plan is not frozen (tag missing), so no U return may be computed."""


class CodeChangedError(RuntimeError):
    """``MODEL_CODE_FILES`` no longer hash to what the adopted run recorded."""


# --- 0. the seal and the invariants ---------------------------------------------


def find_my_repo() -> Path:
    """The ``my`` repository: ``$WSS_MY_REPO``, else a sibling ``my/`` above this checkout."""
    env = os.environ.get("WSS_MY_REPO")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "my"
        if (candidate / ".git").exists():
            return candidate
    raise SealError("`my` 저장소를 찾지 못했다. WSS_MY_REPO 에 경로를 준다")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30, check=False
    )
    if out.returncode != 0:
        raise SealError(f"git {' '.join(args)} 실패: {out.stderr.strip()}")
    return out.stdout.strip()


def assert_sealed(my_repo: Path | None = None) -> str:
    """Return the tagged commit, or raise if ``kr-universe-retest-frozen`` is absent."""
    repo = my_repo or find_my_repo()
    if not _git(repo, "tag", "-l", FROZEN_TAG):
        raise SealError(
            f"`my` 저장소에 태그 {FROZEN_TAG} 가 없다. 사전등록이 동결되기 전에는 "
            "U 마스크와 정의 대조의 수익을 계산하지 않는다 (§8 봉인)"
        )
    return _git(repo, "rev-list", "-n", "1", FROZEN_TAG)


def holdout_record(adopted: hr.AdoptedRun) -> dict:
    """What the opening wrote: ``run_spec.json`` and ``gate.json`` of the holdout folder."""
    folder = hr.HOLDOUT_DIR / adopted.run_id
    return {
        "run_spec": json.loads((folder / "run_spec.json").read_text()),
        "gate": json.loads((folder / "gate.json").read_text()),
    }


def assert_code_unchanged(recorded: str | None = None) -> str:
    """§8 불변: the model code hashes to what the adopted run's opening recorded."""
    if recorded is None:
        recorded = holdout_record(hr.load_adopted())["run_spec"]["model_code_hash"]
    now = model_code_hash()
    if now != recorded:
        raise CodeChangedError(
            f"model_code_hash 가 다르다: 기록 {recorded} · 지금 {now} "
            f"({len(MODEL_CODE_FILES)}개 파일 중 하나가 바뀌었다)"
        )
    return now


def assert_not_holdout_folder(out_dir: Path, adopted: hr.AdoptedRun) -> None:
    """The retest never writes into the folder the opening wrote."""
    protected = (hr.HOLDOUT_DIR / adopted.run_id).resolve()
    target = out_dir.resolve()
    if target == protected or protected in target.parents:
        raise ValueError(f"결과 폴더 {out_dir} 가 개봉 결과 폴더 {protected} 안이다")


# --- 1. benchmarks --------------------------------------------------------------


def market_eqw_bench(raw_root: Path | str, horizon: int = 20) -> pl.DataFrame:
    """All-stocks equal weight per (date, market), rebuilt from ``daily_ohlcv``.

    §7: the runner's benchmark is the mean forward return of *every* name with a
    price (halt days excluded), not of the stored rows — ``label_daily`` holds the
    default universe only (~2,300 names a day), so it cannot recompute this. The
    SQL is ``labels._forward_cte``, the one that built the label.
    """
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW ohlcv AS SELECT * FROM read_parquet('{raw_root}/daily_ohlcv/**/*.parquet')"
    )
    rows = con.execute(f"""
        WITH {labels_mod._forward_cte("ohlcv")}
        SELECT a.trade_date, a.market,
               AVG(f.close_d / NULLIF(a.close_d, 0) - 1) AS bench_recomputed
        FROM px a
        JOIN px f ON f.ticker = a.ticker AND f.market = a.market AND f.d_idx = a.d_idx + {horizon}
        GROUP BY a.trade_date, a.market
    """).arrow()
    con.close()
    return pl.from_arrow(rows)


def runner_bench(label_daily: pl.DataFrame, horizon: int = 20) -> pl.DataFrame:
    """The benchmark the runner actually used: ``fwd_ret - raw_label``, one value per cell."""
    fwd, raw = f"fwd_ret_{horizon}d", f"raw_label_{horizon}d"
    return (
        label_daily.filter(pl.col(fwd).is_not_null() & pl.col(raw).is_not_null())
        .with_columns((pl.col(fwd) - pl.col(raw)).alias("_b"))
        .group_by(["trade_date", "market"])
        .agg(pl.col("_b").mean().alias("bench_all"), pl.col("_b").std().alias("bench_all_std"))
    )


def masked_eqw_bench(frame: pl.DataFrame, mask: str | None, name: str, fwd: str) -> pl.DataFrame:
    """Per (date, market) mean ``fwd`` over the rows where ``mask`` holds (all rows if None)."""
    rows = frame if mask is None else frame.filter(pl.col(mask))
    return (
        rows.filter(pl.col(fwd).is_not_null() & pl.col(fwd).is_finite())
        .group_by(["trade_date", "market"])
        .agg(pl.col(fwd).mean().alias(name))
    )


def swap_label(frame: pl.DataFrame, horizon: int, bench: str) -> pl.DataFrame:
    """Put ``fwd_ret - bench`` in the equal-weight label slot, keep the old one beside it."""
    eqw = hr.EQW_LABEL.format(h=horizon)
    fwd = f"fwd_ret_{horizon}d"
    return frame.with_columns(
        pl.col(eqw).alias(f"{eqw}_orig"), (pl.col(fwd) - pl.col(bench)).alias(eqw)
    )


# --- 2. the U mask --------------------------------------------------------------


def add_u_mask(frame: pl.DataFrame, my_repo: Path | None = None) -> pl.DataFrame:
    """§4: ``in_u``, plus the three equal-weight benchmarks the plan needs.

    ``frame`` needs ``trade_date, ticker, market, mcap_krx, mcap_unreliable,
    tradable, fwd_ret_20d``. Rank is per trading day inside a market, by
    ``mcap_krx`` descending, ties by ticker (ordinal); unreliable or missing
    market caps are left out before ranking; then ``tradable`` is applied and the
    freed seats are **not** refilled. Refuses to run before the freeze.
    """
    assert_sealed(my_repo)
    eligible = frame.filter(
        pl.col("mcap_krx").is_not_null() & ~pl.col("mcap_unreliable").fill_null(False)
    )
    ranked = (
        eligible.select(["trade_date", "ticker", "market", "mcap_krx"])
        .sort(
            ["trade_date", "market", "mcap_krx", "ticker"], descending=[False, False, True, False]
        )
        .with_columns(
            (pl.int_range(pl.len()).over(["trade_date", "market"]) + 1).alias("mcap_rank")
        )
    )
    sizes = pl.DataFrame(
        {"market": list(U_SIZE), "u_size": list(U_SIZE.values())},
        schema={"market": frame.schema["market"], "u_size": pl.Int64},
    )
    out = (
        frame.join(
            ranked.select(["trade_date", "ticker", "market", "mcap_rank"]),
            on=["trade_date", "ticker", "market"],
            how="left",
        )
        .join(sizes, on="market", how="left")
        .with_columns(
            (
                pl.col("mcap_rank").is_not_null()
                & (pl.col("mcap_rank") <= pl.col("u_size").fill_null(0))
                & pl.col("tradable")
            ).alias("in_u")
        )
        .drop("u_size")
    )
    fwd = "fwd_ret_20d"
    for mask, name in ((None, "bench_base"), ("tradable", "bench_trad"), ("in_u", "bench_u")):
        out = out.join(
            masked_eqw_bench(out, mask, name, fwd), on=["trade_date", "market"], how="left"
        )
    return out


# --- 3. the frame ---------------------------------------------------------------


def _label_daily_path(adopted: hr.AdoptedRun, record: dict) -> Path:
    return Path(record["run_spec"]["dataset_dir"]) / "label_daily.parquet"


def load_label_daily(path: Path) -> pl.DataFrame:
    cols = [
        "trade_date",
        "ticker",
        "market",
        "fwd_ret_20d",
        "raw_label_20d",
        "raw_label_idx_20d",
        "label_end_date",
    ]
    return pl.read_parquet(path, columns=cols)


def _join_labels(frame: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Add the three label_daily columns; keep the label the predictions carried."""
    ld = labels.select(
        "trade_date",
        "ticker",
        "market",
        "fwd_ret_20d",
        "raw_label_idx_20d",
        "label_end_date",
        pl.col("raw_label_20d").alias("raw_label_20d_ld"),
    )
    return frame.join(ld, on=["trade_date", "ticker", "market"], how="left")


def load_holdout_frame(adopted: hr.AdoptedRun) -> tuple[pl.DataFrame, dict]:
    """Holdout predictions with cost inputs, ``tradable``, ``mcap_krx`` and the labels.

    The same two ``preopen_measures`` calls the opening used (§8 step 2); the labels
    are joined from ``label_daily`` (§8 step 1). Reads only.
    """
    record = holdout_record(adopted)
    spec = record["run_spec"]
    summary = {**adopted.summary, "dataset_dir": spec["dataset_dir"]}
    predictions = pl.read_parquet(spec["predictions_path"])
    frame = pm._with_cost_inputs(summary, predictions.with_columns(pl.lit(1).alias("fold_id")))
    frame = pm._with_universe_and_buckets(summary, frame)
    labels = load_label_daily(_label_daily_path(adopted, record))
    frame = _join_labels(frame, labels)
    return frame, {"summary": summary, "labels": labels, "record": record}


def _attach_universe(summary: dict, frame: pl.DataFrame) -> pl.DataFrame:
    """``tradable`` and the market-cap columns only (no bucket keys, no ADV needed)."""
    con = duckdb.connect()
    tradable = pl.from_arrow(
        con.execute(
            "SELECT trade_date, ticker, market, in_universe AS tradable "
            f"FROM read_parquet('{pm._mart_glob(summary, pm.TRADABLE_MART)}')"
        ).arrow()
    )
    mcap = pl.from_arrow(
        con.execute(
            "SELECT trade_date, ticker, market, mcap_krx, mcap_unreliable "
            f"FROM read_parquet('{pm._mart_glob(summary, pm.MCAP_MART)}')"
        ).arrow()
    )
    con.close()
    return (
        frame.join(tradable, on=["trade_date", "ticker", "market"], how="left")
        .join(mcap, on=["trade_date", "ticker", "market"], how="left")
        .with_columns(pl.col("tradable").fill_null(False))
    )


def load_formation_frame(adopted: hr.AdoptedRun, holdout_summary: dict, labels: pl.DataFrame):
    """§5 record 1: formation predictions (5 folds), labels unified on ``label_daily``.

    Marts come from the holdout snapshot (09-29), the same as the main window.
    Returns the frame and the number of rows whose prediction-file label differs
    from ``label_daily`` (``raw_label_20d``, null-aware, 1e-12).
    """
    _summary, predictions = pm._load_run(adopted.stage, adopted.run_id)
    frame = _attach_universe(holdout_summary, predictions)
    frame = _join_labels(frame, labels)
    differs = frame.filter(
        (pl.col("raw_label_20d").is_null() != pl.col("raw_label_20d_ld").is_null())
        | ((pl.col("raw_label_20d") - pl.col("raw_label_20d_ld")).abs() > 1e-12)
    ).height
    return frame, differs


# --- 4. reproduction (no U) -----------------------------------------------------


def reproduce(adopted: hr.AdoptedRun | None = None, *, tol: float = 1e-9) -> dict:
    """§8 재현: the plain ``tradable`` mask and the saved labels give the published gate."""
    adopted = adopted or hr.load_adopted()
    frame, ctx = load_holdout_frame(adopted)
    gate = hr.gate_metrics(frame, adopted)
    published = ctx["record"]["gate"]
    keys = ("e", "e_prime", "i", "s", "d")
    diffs = {k: getattr(gate, k) - float(published[k]) for k in keys}
    return {
        "values": {k: getattr(gate, k) for k in keys},
        "published": {k: float(published[k]) for k in keys},
        "abs_diff": {k: abs(v) for k, v in diffs.items()},
        "pass": all(abs(v) <= tol for v in diffs.values()) and gate.verdict == published["verdict"],
        "verdict": gate.verdict,
    }


def bench_check(labels: pl.DataFrame, raw_root: Path | str, *, tol: float = 1e-12) -> dict:
    """§8 벤치 계산: the all-stocks equal weight equals ``fwd_ret - raw_label`` of the label."""
    joined = runner_bench(labels).join(
        market_eqw_bench(raw_root), on=["trade_date", "market"], how="inner"
    )
    worst = float((joined["bench_all"] - joined["bench_recomputed"]).abs().max())
    spread = float(joined["bench_all_std"].fill_null(0.0).max())
    return {
        "cells": joined.height,
        "max_abs_diff": worst,
        "max_within_cell_std": spread,
        "pass": worst <= tol,
    }


# --- 5. the U measurements ------------------------------------------------------


def _topk(part: pl.DataFrame, adopted: hr.AdoptedRun, realized: str, cost: float | None = None):
    return topk_economic_report(
        part,
        pred_col=adopted.primary_pred_col,
        realized_col=realized,
        horizon=adopted.horizon,
        k=adopted.k,
        cost_bps_roundtrip=adopted.cost_bps_roundtrip if cost is None else cost,
    )


def u_gate(frame: pl.DataFrame, adopted: hr.AdoptedRun, my_repo: Path | None = None):
    """E_U, I_U, S_U, D_U — ``gate_metrics`` on U with the U equal-weight excess as label.

    ``gate_metrics`` also returns an ``e_prime`` — on U it is the same number as
    ``e`` (U is already inside ``tradable``) and carries no information.
    """
    framed = add_u_mask(frame, my_repo)
    u_rows = swap_label(framed.filter(pl.col("in_u")), adopted.horizon, "bench_u")
    return hr.gate_metrics(u_rows.with_columns(pl.lit(True).alias("tradable")), adopted), framed


def _rebalance_mean(series: pl.DataFrame) -> dict:
    """Mean net rebalance return, and the same without the best single rebalance."""
    net = series.filter(pl.col("net_return").is_not_null())
    if net.height < 2:
        return {
            "mean": float("nan"),
            "mean_ex_max": float("nan"),
            "max_date": None,
            "n": net.height,
        }
    top = net.sort("net_return", descending=True).row(0, named=True)
    rest = net.filter(pl.col("rebalance_date") != top["rebalance_date"])
    return {
        "mean": float(net["net_return"].mean()),
        "mean_ex_max": float(rest["net_return"].mean()),
        "max_date": str(top["rebalance_date"]),
        "max_value": float(top["net_return"]),
        "n": net.height,
    }


def label_raw_root(ctx: dict) -> str:
    """The raw lake the labels were built from (``dataset_manifest.json`` ``lake.raw``)."""
    dataset_dir = Path(ctx["record"]["run_spec"]["dataset_dir"])
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    return manifest["lake"]["raw"]


def load_close_series(raw_root: Path | str, start) -> pl.DataFrame:
    """Every session's close from ``daily_ohlcv`` on or after ``start`` (``close > 0``)."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT trade_date, ticker, market, close "
        f"FROM read_parquet('{raw_root}/daily_ohlcv/**/*.parquet') "
        f"WHERE trade_date >= ? AND close > 0",
        [start],
    ).arrow()
    con.close()
    return pl.from_arrow(rows)


def count_label_window_jumps(
    frame: pl.DataFrame,
    horizon: int,
    held_mask: str | None = None,
    prices: pl.DataFrame | None = None,
) -> dict:
    """§6 record 5: (name, rebalance) pairs with a one-day close jump inside the label window.

    Window is ``(trade_date, label_end_date]``. Jump is a close ratio above 1.5 or below
    1/1.5 against the name's previous available close. This approximates the probe's
    count (03_probe §7.2); the plan asks for "the number", not a re-derivation.

    Jumps are found on every row of ``frame`` and only the held rows (rebalance dates,
    and ``held_mask`` when given) are counted: a name that falls out of U on the day of
    a crash must still show the crash. ``prices`` (``trade_date, ticker, market, close``
    from ``daily_ohlcv``, see :func:`load_close_series`) replaces the frame's closes: the
    frame stops at the last prediction date while the last label window runs past it,
    and it skips the days a name is outside the default universe.
    """
    grid = rebalance_grid(frame["trade_date"].unique().to_list(), horizon)
    keep = ["trade_date", "ticker", "market", "close", "label_end_date"]
    if held_mask is not None:
        keep.append(held_mask)
    series = frame.select(keep) if prices is None else prices
    ordered = (
        series.select("trade_date", "ticker", "market", "close")
        .sort(["ticker", "market", "trade_date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over(["ticker", "market"])).alias("_ratio")
        )
    )
    jumps = (
        ordered.filter((pl.col("_ratio") > JUMP_UP) | (pl.col("_ratio") < JUMP_DOWN))
        .select("trade_date", "ticker", "market", "_ratio")
        .rename({"trade_date": "jump_date"})
    )
    held = frame.select(keep).filter(
        pl.col("trade_date").is_in(grid) & pl.col("label_end_date").is_not_null()
    )
    if held_mask is not None:
        held = held.filter(pl.col(held_mask))
    hit = (
        held.select("trade_date", "ticker", "market", "label_end_date")
        .join(jumps, on=["ticker", "market"], how="inner")
        .filter(
            (pl.col("jump_date") > pl.col("trade_date"))
            & (pl.col("jump_date") <= pl.col("label_end_date"))
        )
    )
    pairs = hit.select("trade_date", "ticker", "market").unique().height
    worst = (
        float(
            hit.select(
                pl.when(pl.col("_ratio") >= 1)
                .then(pl.col("_ratio"))
                .otherwise(1 / pl.col("_ratio"))
            ).max()[0, 0]
        )
        if hit.height
        else 1.0
    )
    return {"pairs": pairs, "worst_multiple": worst, "n_rebalances": len(grid)}


def u_records(
    framed: pl.DataFrame, adopted: hr.AdoptedRun, gate, prices: pl.DataFrame | None = None
) -> dict:
    """§6 "같이 적는 기록" 다섯 — nothing here enters the verdict."""
    h = adopted.horizon
    eqw, idx = adopted.eqw_label, adopted.idx_label
    u_rows = framed.filter(pl.col("in_u"))
    swapped = swap_label(u_rows, h, "bench_u")

    gross_gate = hr.gate_metrics(
        swapped.with_columns(pl.lit(True).alias("tradable")),
        dataclasses.replace(adopted, cost_bps_roundtrip=0.0),
    )
    # (U equal weight - index) of the held names, gross; I_U = E_U + this at equal cost.
    gap = swapped.with_columns((pl.col(idx) - pl.col(eqw)).alias("u_eqw_minus_idx"))
    gap_report = _topk(gap, adopted, "u_eqw_minus_idx", cost=0.0)

    def series(realized: str) -> pl.DataFrame:
        return topk_rebalance_series(
            swapped,
            pred_col=adopted.primary_pred_col,
            realized_col=realized,
            horizon=h,
            k=adopted.k,
            cost_bps_roundtrip=adopted.cost_bps_roundtrip,
        )

    # the label the opening used (all-stocks equal-weight excess), straight from label_daily
    old_bench = _topk(u_rows.with_columns(pl.col("raw_label_20d_ld").alias(eqw)), adopted, eqw)
    u_eqw_path = _topk(swapped, adopted, eqw)
    return {
        "u_eqw_minus_idx_gross": gap_report.cost_adjusted_return,
        "ex_largest_rebalance": {
            "E_U": _rebalance_mean(series(eqw)),
            "I_U": _rebalance_mean(series(idx)),
        },
        "gross": {"E_U": gross_gate.e, "I_U": gross_gate.i, "S_U": gross_gate.s},
        "E_U_old_bench": old_bench.cost_adjusted_return,
        "D_U_vs_u_eqw": {
            "max_drawdown": u_eqw_path.max_drawdown,
            "longest_rebalances": u_eqw_path.longest_drawdown_rebalances,
        },
        "D_U_vs_index": {
            "max_drawdown": gate.d,
            "longest_rebalances": gate.d_longest_rebalances,
        },
        "price_contamination": count_label_window_jumps(framed, h, "in_u", prices),
        # The same count over the plain tradable universe, to set beside the published
        # 4 pairs (2.4x) of 03_probe §7.2, whose exact definition this only approximates.
        "price_contamination_tradable_reference": count_label_window_jumps(
            framed, h, "tradable", prices
        ),
        "u_rows": u_rows.height,
        "u_names_per_day_median": float(u_rows.group_by("trade_date").len()["len"].median()),
    }


def definition_check(
    frame: pl.DataFrame, adopted: hr.AdoptedRun, my_repo: Path | None = None
) -> dict:
    """§7: E and E' measured literally — default-universe and tradable equal weights.

    Not a re-judgement. If the literal E is not positive the plan asks for a
    correction block and a question to the user; this only reports the signs.
    """
    assert_sealed(my_repo)
    benches = frame.join(
        masked_eqw_bench(frame, None, "bench_base", "fwd_ret_20d"),
        on=["trade_date", "market"],
        how="left",
    ).join(
        masked_eqw_bench(frame, "tradable", "bench_trad", "fwd_ret_20d"),
        on=["trade_date", "market"],
        how="left",
    )
    runner = hr.gate_metrics(frame, adopted)
    e_lit = _topk(swap_label(benches, adopted.horizon, "bench_base"), adopted, adopted.eqw_label)
    e_prime_lit = _topk(
        swap_label(benches.filter(pl.col("tradable")), adopted.horizon, "bench_trad"),
        adopted,
        adopted.eqw_label,
    )
    return {
        "E_runner": runner.e,
        "E_literal": e_lit.cost_adjusted_return,
        "E_prime_runner": runner.e_prime,
        "E_prime_literal": e_prime_lit.cost_adjusted_return,
        "E_same_sign": (runner.e > 0) == (e_lit.cost_adjusted_return > 0),
        "E_prime_same_sign": (runner.e_prime > 0) == (e_prime_lit.cost_adjusted_return > 0),
        "needs_correction_block": not e_lit.cost_adjusted_return > 0,
    }


# --- 6. verdict -----------------------------------------------------------------


def decide_prime(e_u: float, i_u: float, s_u: float) -> tuple[str, str]:
    """§6: C' first, then A', then B'. Signs only; same rule as ``holdout_run.decide``."""
    verdict, reason = hr.decide(e_u, i_u, s_u)
    return (verdict + "′" if verdict in ("A", "B", "C") else verdict), reason


# --- 7. one run -----------------------------------------------------------------


def _fold_gates(frame: pl.DataFrame, adopted: hr.AdoptedRun) -> dict:
    """Record 1 on formation: gate numbers per fold, and over one pooled grid. Not a verdict."""
    out: dict = {}
    framed = add_u_mask(frame)
    u_rows = framed.filter(pl.col("in_u"))
    for fold in sorted(u_rows["fold_id"].unique().to_list()):
        part = swap_label(u_rows.filter(pl.col("fold_id") == fold), adopted.horizon, "bench_u")
        g = hr.gate_metrics(part.with_columns(pl.lit(True).alias("tradable")), adopted)
        out[f"fold_{fold}"] = {k: getattr(g, k) for k in ("e", "i", "s", "d", "n_rebalances_i")}
    pooled = swap_label(u_rows, adopted.horizon, "bench_u")
    g = hr.gate_metrics(pooled.with_columns(pl.lit(True).alias("tradable")), adopted)
    out["pooled"] = {k: getattr(g, k) for k in ("e", "i", "s", "d", "n_rebalances_i")}
    return out


def run(out_dir: Path | None = None, my_repo: Path | None = None) -> dict:
    """B.3 in one pass: seal, code hash, holdout U gate, records, §7 check, formation record 1."""
    tag_commit = assert_sealed(my_repo)
    adopted = hr.load_adopted()
    code = assert_code_unchanged()
    out_dir = out_dir or (RETEST_DIR / adopted.run_id)
    assert_not_holdout_folder(out_dir, adopted)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    frame, ctx = load_holdout_frame(adopted)
    gate, framed = u_gate(frame, adopted, my_repo)
    verdict, reason = decide_prime(gate.e, gate.i, gate.s)
    prices = load_close_series(label_raw_root(ctx), framed["trade_date"].min())
    records = u_records(framed, adopted, gate, prices)
    definition = definition_check(frame, adopted, my_repo)
    print(f"  holdout {time.time() - started:.0f}s · 갈래 {verdict}")

    for label, realized in (("eqw_u", adopted.eqw_label), ("idx", adopted.idx_label)):
        topk_rebalance_series(
            swap_label(framed.filter(pl.col("in_u")), adopted.horizon, "bench_u").with_columns(
                pl.lit(True).alias("tradable")
            ),
            pred_col=adopted.primary_pred_col,
            realized_col=realized,
            horizon=adopted.horizon,
            k=adopted.k,
            cost_bps_roundtrip=adopted.cost_bps_roundtrip,
        ).write_parquet(out_dir / f"rebalance_returns_{label}.parquet")

    formation_frame, differs = load_formation_frame(adopted, ctx["summary"], ctx["labels"])
    formation = {
        "label_rows_differing_from_label_daily": differs,
        **_fold_gates(formation_frame, adopted),
    }

    result = {
        "verdict": verdict,
        "verdict_reason": reason,
        "gate_u": gate.as_dict(),
        "records": records,
        "definition_check": definition,
        "formation_record_1": formation,
        "provenance": {
            "git_sha": current_git_sha(),
            "code_hash": code_hash(),
            "model_code_hash": code,
            "frozen_tag": FROZEN_TAG,
            "frozen_tag_commit": tag_commit,
            "adopted_run": adopted.run_id,
            "holdout_predictions": ctx["record"]["run_spec"]["predictions_path"],
            "label_daily": str(_label_daily_path(adopted, ctx["record"])),
            "marts": ctx["summary"]["dataset_dir"],
            "formation_predictions": str(adopted.predictions_path),
            "refit": False,
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "summary.md").write_text(render_summary(result))
    print(f"  {out_dir}")
    return result


def render_summary(result: dict) -> str:
    g = result["gate_u"]
    rec = result["records"]
    return f"""# 유니버스 재검정 결과 — U (KOSPI 200 + KOSDAQ 150)

**갈래 {result["verdict"]}** — {result["verdict_reason"]}

원인 분석이다. 독립 검증이 아니다 (사전등록 §0).

| 기호 | 값 |
|---|---:|
| E_U | {g["e"]:+.6f} |
| I_U | {g["i"]:+.6f} |
| S_U | {g["s"]:+.6f} |
| D_U (지수) | {g["d"]:+.6f} |

같이 적는 기록과 정의 대조, formation 기록 ①은 `result.json`에 있다.
U 동일가중 − 지수 (gross): {rec["u_eqw_minus_idx_gross"]:+.6f}
"""


# --- 8. cli ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="봉인·코드 해시·입력 경로만 본다 (기본)")
    mode.add_argument(
        "--reproduce", action="store_true", help="U 없이 공개된 E·E′·I·S·D를 재현한다"
    )
    mode.add_argument("--run", action="store_true", help="B.3 본 실행")
    parser.add_argument("--i-am-running-the-retest", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=None, help="결과 폴더 (기본 results/universe_retest/<run>)"
    )
    args = parser.parse_args(argv)

    adopted = hr.load_adopted()
    if args.reproduce:
        res = reproduce(adopted)
        print(json.dumps(res, indent=2))
        return 0 if res["pass"] else 1
    if args.run:
        if not args.i_am_running_the_retest:
            parser.error("--run 에는 --i-am-running-the-retest 가 필요하다")
        run(args.out)
        return 0
    try:
        tag_commit = assert_sealed()
    except SealError as exc:
        print(f"봉인: 닫힘 — {exc}")
        tag_commit = None
    print(f"봉인: {'열림 ' + tag_commit[:8] if tag_commit else '닫힘'}")
    print(f"model_code_hash: {assert_code_unchanged()} (개봉 기록과 같다)")
    record = holdout_record(adopted)
    for key in ("predictions_path",):
        path = Path(record["run_spec"][key])
        print(f"{key}: {'있다' if path.exists() else '없다'} {path}")
    label_path = _label_daily_path(adopted, record)
    print(f"label_daily: {'있다' if label_path.exists() else '없다'} {label_path}")
    print(f"결과 폴더: {args.out or RETEST_DIR / adopted.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
