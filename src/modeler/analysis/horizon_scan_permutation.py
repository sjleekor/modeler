"""Phase A null experiments: cross-sectional permutation and temporal
placebo (A-6, §A-6a/§A-6b).

Two distinct null experiments, not interchangeable (§A-6 intro): cross-
sectional permutation breaks the feature-label link within each date×market
block (diagnoses join/leakage and the 75-cell search size); circular
date-shift placebo rotates the whole feature calendar against the label
calendar (diagnoses whether gap-aware NW under-corrects long-horizon overlap).
Both need bit-for-bit reproducibility regardless of execution order or
parallelism, hence the deterministic seed derivation below.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_checkpoint import (
    load_replicate_checkpoints,
    write_replicate_checkpoint,
)
from modeler.analysis.horizon_scan_mapping import (
    JOINT_CS_MAPPING_CONTRACT,
    build_and_apply_group_permutation,
)
from modeler.analysis.horizon_scan_runner import _target_columns, apply_global_bh, scan_cell
from modeler.etl.metrics import choose_nw_lag

_CROSS_PROCESS_CONTEXT: dict[str, Any] = {}


def derive_replicate_seed(*, placebo_kind: str, replicate_index: int, config_hash: str) -> int:
    """A seed fixed by (placebo kind, replicate index, config_hash) alone —
    never by wall-clock time or run order — so a resumed or re-parallelized
    run reproduces the exact same replicate (§A-6: "seed는 ... 결정해 실행
    순서·병렬 수에 관계없이 같다").
    """
    payload = f"{placebo_kind}|{replicate_index}|{config_hash}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**32)


def permute_within_groups(
    df: pl.DataFrame,
    *,
    group_cols: list[str],
    permute_cols: list[str],
    seed: int,
) -> pl.DataFrame:
    """Apply one shared row permutation per group to ``permute_cols`` only.

    §A-6a: "같은 permutation mapping을 모든 primary feature column에 공동
    적용한다" — every column in ``permute_cols`` moves together as a single
    row vector (preserving cross-feature correlation and NULL patterns),
    while every other column (date, market, ticker, labels) stays fixed to
    its original row. Groups are processed in a canonical sort order and each
    draws from the same ``seed``-derived generator in that order, so the
    result is identical regardless of the input frame's row order or how the
    groups happen to be iterated upstream.
    """
    missing = set(group_cols) | set(permute_cols)
    unknown = missing - set(df.columns)
    if unknown:
        raise ValueError(f"columns not present in frame: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    fixed_cols = [c for c in df.columns if c not in permute_cols]
    # Sort by every fixed column (not just group_cols) so the within-group row
    # order — and therefore which shuffled index lands on which row — never
    # depends on the input frame's original row order (e.g. "ticker" among
    # fixed_cols disambiguates rows that tie on group_cols alone).
    ordered = df.sort(fixed_cols)

    parts: list[pl.DataFrame] = []
    for _keys, group in ordered.group_by(group_cols, maintain_order=True):
        idx = np.arange(group.height)
        rng.shuffle(idx)
        permuted = group.select(permute_cols)[idx.tolist()]
        parts.append(pl.concat([group.select(fixed_cols), permuted], how="horizontal"))
    return pl.concat(parts, how="vertical").select(ordered.columns)


# --- A-6b circular date-shift temporal placebo ---


def select_circular_shift_distance(*, seed: int, total_sessions: int, min_shift: int = 120) -> int:
    """Pick one shift distance in ``[min_shift, total_sessions - min_shift]`` (§A-6b).

    The range excludes shifts smaller than ``min_shift`` sessions on either
    side of a full wrap so the placebo can't degenerate into an
    (almost-)identity shift — a shift of a few sessions would barely disturb
    the feature-label alignment and understate the null's spread.
    """
    span = total_sessions - 2 * min_shift
    if span < 0:
        raise ValueError(
            f"total_sessions={total_sessions} is too short for min_shift={min_shift} on both sides"
        )
    rng = np.random.default_rng(seed)
    return int(min_shift + rng.integers(0, span + 1))


def circular_shift_session_index(
    session_idx: np.ndarray, *, shift: int, total_sessions: int
) -> np.ndarray:
    """Map each 1-indexed session onto its circularly-shifted position.

    ``total_sessions`` is the full non-holdout KRX calendar length ``T``
    (§A-6b step 1) — shifting by ``T`` is a no-op, so callers must derive
    ``shift`` from :func:`select_circular_shift_distance`, not an arbitrary
    integer, to land in the required ``[min_shift, T-min_shift]`` band.
    """
    zero_based = (np.asarray(session_idx, dtype=np.int64) - 1 + shift) % total_sessions
    return zero_based + 1


def apply_circular_feature_shift(
    df: pl.DataFrame,
    *,
    session_col: str,
    shift: int,
    total_sessions: int,
) -> pl.DataFrame:
    """Relabel every row's session index by one circular shift (§A-6b steps
    2-3): every ticker/market's own feature time series and same-date
    cross-sectional structure travel together (only the row's session label
    changes), so joining this back onto the *original* label frame by the
    shifted session index pairs each real date's label with a different
    date's feature snapshot — the same shift for every name.
    """
    if session_col not in df.columns:
        raise ValueError(f"{session_col!r} not present in frame")
    shifted = circular_shift_session_index(
        df[session_col].to_numpy(), shift=shift, total_sessions=total_sessions
    )
    return df.with_columns(pl.Series(session_col, shifted))


# --- empirical p-values shared by both null experiments (§A-6a/§A-6b) ---


def empirical_discovery_count_p(
    real_discovery_count: int, null_discovery_counts: list[int]
) -> float:
    """§A-6a: ``p_empirical_count = (1 + #{null >= real}) / (repeats + 1)``.

    Interpreted only as a join/leakage and global search-size diagnostic —
    the cross-sectional permutation does not preserve the daily-IC serial
    autocorrelation that overlap requires, so it cannot validate the NW
    overlap correction itself (§A-6 intro).
    """
    if not null_discovery_counts:
        raise ValueError("null_discovery_counts must be non-empty")
    at_least = sum(1 for n in null_discovery_counts if n >= real_discovery_count)
    return (1 + at_least) / (len(null_discovery_counts) + 1)


#: How many Monte Carlo standard errors from ``p_max`` still counts as "the
#: threshold decided this, not the data" (F-9.11). Two, so the band is the
#: conventional 95% one; fixed here rather than per call site so a marginal cell
#: means the same thing in every layer.
TEMPORAL_MARGINAL_SE_MULTIPLE = 2.0


def temporal_placebo_p(
    real_abs_t_nw: float, shifted_abs_t_nw: list[float], *, p_max: float = 0.10
) -> dict[str, Any]:
    """§A-6b: ``p_temporal_nw = (1 + #{|t_shift| >= |t_real|}) / (repeats + 1)``,
    ``temporal_null_pass = p_temporal_nw < p_max``. Only meaningful for the
    long (``nw_lag >= 59``) primary cells this placebo is run against.

    Also reports the Monte Carlo uncertainty of ``p`` itself (F-9.11), because
    the pass/fail alone overstates what 100 replicates can resolve. The shift
    seed is derived from ``config_hash``, so every new preregistration layer
    draws a fresh null and a cell sitting near ``p_max`` changes side between
    layers on nothing but that redraw. Measured on the 2026-09-09 run:
    ``ev_payout_yield|bucket|60|120`` went 0.0891 -> 0.1287 (pass -> fail) and
    ``mcap_krx_log|cum|0|120`` went 0.1386 -> 0.0495 (fail -> pass), with every
    other statistic on both cells bit-identical.

    ``temporal_null_marginal`` says the decision fell inside that noise. It is a
    *diagnostic*, deliberately not an input to ``temporal_null_pass`` or to
    ``screen_pass``: the rule and the replicate count are preregistered
    (``temporal_p_max``, ``temporal_long_cell_repeats``) and changing either
    after seeing results is what preregistration exists to prevent. What was
    missing is the reader being told the flip is a coin toss.
    """
    if not shifted_abs_t_nw:
        raise ValueError("shifted_abs_t_nw must be non-empty")
    repeats = len(shifted_abs_t_nw)
    at_least = sum(1 for t in shifted_abs_t_nw if t >= real_abs_t_nw)
    p = (1 + at_least) / (repeats + 1)
    # Two standard errors, both binomial, and they answer different questions.
    #
    # `p_temporal_nw_se` is the error of *this* estimate, taken at the observed
    # p -- what a reader wants next to the number.
    #
    # The marginal band is taken at `p_max`, not at the observed p, because the
    # question it answers is "could a redraw put this cell on the other side of
    # the threshold?" -- a statement about the neighbourhood of p_max. Using the
    # observed p instead shrinks the band exactly where p is small, and it
    # misses a real flip: mcap_krx_log|cum|0|120 moved 0.1386 -> 0.0495, and
    # 0.0495 is 0.0505 away from 0.10 against a 2-SE-at-p-hat band of 0.0434.
    # At p_max=0.10 with 100 replicates the band is 0.060, spanning 0.04..0.16,
    # which contains both measured flips.
    standard_error = math.sqrt(max(p * (1.0 - p), 0.0) / repeats)
    threshold_se = math.sqrt(max(p_max * (1.0 - p_max), 0.0) / repeats)
    marginal_band = TEMPORAL_MARGINAL_SE_MULTIPLE * threshold_se
    return {
        "p_temporal_nw": p,
        "temporal_null_pass": bool(p < p_max),
        "temporal_repeats": repeats,
        "p_temporal_nw_se": standard_error,
        "temporal_marginal_band": marginal_band,
        "temporal_null_marginal": bool(abs(p - p_max) <= marginal_band),
    }


# --- A-6a/A-6b replicate-loop orchestration (checkpoint/resume-safe) ---
#
# Both loops reuse `scan_cell`/`apply_global_bh` verbatim against a
# reduced, permuted/shifted view of the broad/common-survivor core panel —
# "각 replicate의 75개 primary 셀에 실제와 같은 NW·BH를 적용한다" (§A-6a step 6)
# means the *same* formation/IC/NW code path, not a re-derived shortcut.


def select_long_horizon_hypotheses(
    registry: list[dict[str, Any]], *, min_nw_lag: int = 59
) -> list[dict[str, Any]]:
    """§A-6b: the ``nw_lag >= 59`` primary cells only (13 of the 75 — cumulative
    60d/120d and bucket ``(60,120]``), selected by the same lag rule
    (``choose_nw_lag``) the real scan uses, never a hand-picked family list.
    """
    selected = []
    for hyp in registry:
        lag = choose_nw_lag(
            scan_type=hyp["scan_type"],
            horizon=hyp["h_end"] if hyp["scan_type"] == "cum" else None,
            bucket_width=(hyp["h_end"] - hyp["h_start"]) if hyp["scan_type"] == "bucket" else None,
        )
        if lag >= min_nw_lag:
            selected.append(hyp)
    return selected


def _registry_target_columns(registry: list[dict[str, Any]]) -> list[str]:
    cols: set[str] = set()
    for hyp in registry:
        cols.update(
            _target_columns(scan_type=hyp["scan_type"], h_start=hyp["h_start"], h_end=hyp["h_end"])
        )
    return sorted(cols)


def _load_checkpoint(
    path: Path | None, *, fingerprint: dict[str, Any] | None = None
) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    if path.is_dir():
        if fingerprint is None:
            raise ValueError("directory checkpoints require a fingerprint")
        return load_replicate_checkpoints(path, fingerprint=fingerprint)
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if fingerprint is not None:
            if row.get("_fingerprint") != fingerprint:
                raise ValueError(
                    "checkpoint fingerprint is missing or incompatible; refusing resume"
                )
        out[row["replicate"]] = row
    return out


def _append_checkpoint(
    path: Path | None, row: dict[str, Any], *, fingerprint: dict[str, Any] | None = None
) -> None:
    if path is None:
        return
    if path.is_dir() or (fingerprint is not None and path.suffix == ""):
        if fingerprint is None:
            raise ValueError("directory checkpoints require a fingerprint")
        write_replicate_checkpoint(
            path,
            replicate=int(row["replicate"]),
            fingerprint=fingerprint,
            payload=row,
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fingerprint is not None:
        row = {**row, "_fingerprint": fingerprint}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def fetch_broad_common_survivor_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    extra_cols: list[str],
    sample_start: str,
) -> pl.DataFrame:
    """§A-6a step 1 / §A-6b: the broad/common-survivor core panel, restricted
    to the columns a replicate loop actually needs. NULL patterns and every
    other column travel with the row untouched — no feature/label eligibility
    filtering happens here, exactly as in the real scan's own formation SQL.
    """
    cols = ", ".join(
        dict.fromkeys(
            ["trade_date", "ticker", "market", "formation_session_idx", "ca_mask"] + extra_cols
        )
    )
    sql = f"""
        SELECT {cols}
        FROM {panel_view}
        WHERE in_broad AND common_formation_120d AND common_survivor_120d
          AND trade_date >= DATE '{sample_start}'
    """
    frame = con.execute(sql).pl()
    return frame.with_columns(
        pl.lit(True).alias("in_broad"),
        pl.lit(True).alias("common_formation_120d"),
        pl.lit(True).alias("common_survivor_120d"),
    )


def _scan_registry_once(
    con: duckdb.DuckDBPyConnection,
    registry: list[dict[str, Any]],
    *,
    panel_view: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str = "legacy",
    frame: pl.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if frame is None:
        frame = fetch_broad_common_survivor_frame(
            con,
            panel_view=panel_view,
            extra_cols=sorted(
                {
                    col
                    for hyp in registry
                    for col in _registry_target_columns([hyp]) + [hyp["feature"]]
                }
            ),
            sample_start=sample_start,
        )
    return [
        {
            **hyp,
            **scan_cell(
                con,
                panel_view=panel_view,
                feature_col=hyp["feature"],
                scan_type=hyp["scan_type"],
                h_start=hyp["h_start"],
                h_end=hyp["h_end"],
                universe="broad",
                sample_kind="common_survivor",
                sample_start=sample_start,
                min_names=min_names,
                min_names_for_spread=min_names_for_spread,
                quantile_count=quantile_count,
                min_dates_per_cell=min_dates_per_cell,
                expected_sign=hyp.get("expected_sign"),
                compute_spread=False,
                scan_engine=scan_engine,
                formation_frame=frame,
            ),
        }
        for hyp in registry
    ]


def _cross_sectional_replicate_summary(
    replicate_index: int,
    seed: int,
    bh_rows: list[dict[str, Any]],
    *,
    mapping_hash: str | None = None,
) -> dict[str, Any]:
    valid = [r for r in bh_rows if r["status"] == "valid"]
    p_nw_valid = [r["p_nw"] for r in valid if r["p_nw"] is not None and math.isfinite(r["p_nw"])]
    t_nw_valid = [
        abs(r["t_nw"]) for r in valid if r["t_nw"] is not None and math.isfinite(r["t_nw"])
    ]
    q_valid = [r["q_fdr_global"] for r in bh_rows if r["q_fdr_global"] is not None]
    return {
        "replicate": replicate_index,
        "seed": seed,
        "n_valid_hypotheses": len(valid),
        "n_bh_pass": sum(1 for r in bh_rows if r["bh_pass"]),
        "n_primary_discovery": sum(1 for r in bh_rows if r["primary_discovery"]),
        "min_p_nw": min(p_nw_valid) if p_nw_valid else None,
        "min_q_fdr_global": min(q_valid) if q_valid else None,
        "max_abs_t_nw": max(t_nw_valid) if t_nw_valid else None,
        "mapping_hash": mapping_hash,
    }


def _compute_cross_sectional_replicate(
    con: duckdb.DuckDBPyConnection,
    *,
    replicate: int,
    base_frame: pl.DataFrame,
    primary_registry: list[dict[str, Any]],
    panel_view: str,
    config_hash: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    q_threshold: float,
    scan_engine: str,
    mapping_contract_version: str,
    collect_cell_stats: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = derive_replicate_seed(
        placebo_kind="cross_sectional", replicate_index=replicate, config_hash=config_hash
    )
    feature_cols = sorted({hyp["feature"] for hyp in primary_registry})
    if mapping_contract_version == JOINT_CS_MAPPING_CONTRACT:
        permuted, mapping_hash = build_and_apply_group_permutation(
            base_frame,
            permute_cols=feature_cols,
            replicate_index=replicate,
            config_hash=config_hash,
            mapping_contract_version=mapping_contract_version,
        )
    elif mapping_contract_version == "v1":
        permuted = permute_within_groups(
            base_frame,
            group_cols=["trade_date", "market"],
            permute_cols=feature_cols,
            seed=seed,
        )
        mapping_hash = None
    else:
        raise ValueError(f"unknown mapping contract: {mapping_contract_version!r}")
    rows = _scan_registry_once(
        con,
        primary_registry,
        panel_view=panel_view,
        sample_start=sample_start,
        min_names=min_names,
        min_names_for_spread=min_names_for_spread,
        quantile_count=quantile_count,
        min_dates_per_cell=min_dates_per_cell,
        scan_engine=scan_engine,
        frame=permuted,
    )
    bh_rows = apply_global_bh(rows, q_threshold=q_threshold)
    summary = _cross_sectional_replicate_summary(
        replicate, seed, bh_rows, mapping_hash=mapping_hash
    )
    cell_stats = (
        [
            {
                "mapping_contract_version": mapping_contract_version,
                "replicate": replicate,
                "mapping_hash": mapping_hash,
                "hypothesis_id": row.get("hypothesis_id"),
                "family": row.get("family"),
                "feature": row.get("feature"),
                "scan_type": row.get("scan_type"),
                "h_start": row.get("h_start"),
                "h_end": row.get("h_end"),
                "expected_sign": row.get("expected_sign"),
                "status": row.get("status"),
                "ic_mean": row.get("ic_mean"),
                "t_nw": row.get("t_nw"),
                "p_nw": row.get("p_nw"),
                "n_dates": row.get("n_dates"),
                "n_obs": row.get("n_obs"),
            }
            for row in bh_rows
        ]
        if collect_cell_stats
        else []
    )
    if collect_cell_stats:
        summary["cell_stats"] = cell_stats
    return summary, cell_stats


def _init_cross_sectional_process_worker(
    lake: Any,
    panel_view: str,
    primary_registry: list[dict[str, Any]],
    sample_start: str,
) -> None:
    """Open one independent DuckDB/frame context for a process worker."""
    from modeler.analysis.horizon_scan_run_spec import REQUIRED_A0_MARTS
    from modeler.analysis.horizon_scan_runner import register_analysis_panel
    from modeler.etl.lake import connect
    from modeler.etl.mart import register_mart_view

    con = connect(lake)
    for name in REQUIRED_A0_MARTS:
        register_mart_view(con, lake, name)
    register_analysis_panel(con, view_name=panel_view)
    feature_cols = sorted({hyp["feature"] for hyp in primary_registry})
    target_cols = _registry_target_columns(primary_registry)
    frame = fetch_broad_common_survivor_frame(
        con,
        panel_view=panel_view,
        extra_cols=feature_cols + target_cols,
        sample_start=sample_start,
    )
    _CROSS_PROCESS_CONTEXT.clear()
    _CROSS_PROCESS_CONTEXT.update({"con": con, "frame": frame})


def _compute_cross_sectional_process_replicate(
    replicate: int, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kwargs = {key: value for key, value in kwargs.items() if key != "base_frame"}
    return _compute_cross_sectional_replicate(
        _CROSS_PROCESS_CONTEXT["con"],
        replicate=replicate,
        base_frame=_CROSS_PROCESS_CONTEXT["frame"],
        **kwargs,
    )


def run_cross_sectional_permutation(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    primary_registry: list[dict[str, Any]],
    real_discovery_count: int,
    config_hash: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    n_replicates: int = 100,
    q_threshold: float = 0.10,
    checkpoint_path: Path | None = None,
    scan_engine: str = "legacy",
    checkpoint_fingerprint: dict[str, Any] | None = None,
    mapping_contract_version: str = "v1",
    collect_cell_stats: bool = False,
    workers: int = 1,
    worker_lake: Any | None = None,
) -> dict[str, Any]:
    """§A-6a: 100 joint date×market block permutations of every primary
    feature, each rescanned/BH'd exactly like the real 75 cells.

    Resume-safe: completed replicates are read from ``checkpoint_path``
    (JSONL, one row per replicate) and never recomputed; a fresh call with an
    empty/absent checkpoint reproduces the same summaries bit-for-bit given
    the same ``config_hash`` (seed is derived from replicate index + config
    hash alone — never wall-clock or run order).
    """
    feature_cols = sorted({hyp["feature"] for hyp in primary_registry})
    target_cols = _registry_target_columns(primary_registry)
    base_frame = fetch_broad_common_survivor_frame(
        con, panel_view=panel_view, extra_cols=feature_cols + target_cols, sample_start=sample_start
    )

    checkpoint = _load_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint)
    summaries: list[dict[str, Any]] = [checkpoint[i] for i in sorted(checkpoint)]
    cell_stats: list[dict[str, Any]] = []
    if collect_cell_stats:
        for summary in summaries:
            cell_stats.extend(summary.get("cell_stats", []))
    if workers < 1:
        raise ValueError("workers must be >= 1")
    pending = [i for i in range(n_replicates) if i not in checkpoint]
    kwargs = {
        "base_frame": base_frame,
        "primary_registry": primary_registry,
        "panel_view": panel_view,
        "config_hash": config_hash,
        "sample_start": sample_start,
        "min_names": min_names,
        "min_names_for_spread": min_names_for_spread,
        "quantile_count": quantile_count,
        "min_dates_per_cell": min_dates_per_cell,
        "q_threshold": q_threshold,
        "scan_engine": scan_engine,
        "mapping_contract_version": mapping_contract_version,
        "collect_cell_stats": collect_cell_stats,
    }
    if workers == 1:
        computed = (_compute_cross_sectional_replicate(con, replicate=i, **kwargs) for i in pending)
        computed_rows = zip(pending, computed)
    elif worker_lake is not None:
        # ``ProcessPoolExecutor`` has no ``initkwargs`` argument. Keep the
        # initializer positional so the real Phase A process path is usable.
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cross_sectional_process_worker,
            initargs=(worker_lake, panel_view, primary_registry, sample_start),
        )
        # The child builds its own frame in the initializer. Removing this
        # value before submit is essential: removing it in the child is too
        # late because pickle has already copied it across the process pipe.
        process_kwargs = {key: value for key, value in kwargs.items() if key != "base_frame"}
        futures = [
            executor.submit(_compute_cross_sectional_process_replicate, i, process_kwargs)
            for i in pending
        ]
        computed_rows = zip(pending, (future.result() for future in futures))
    else:
        # Unit/synthetic callers do not have a LakeConfig to initialize in a
        # child process. Keep their large frame in-process; Polars releases the
        # GIL in the hot kernels and checkpoint writes remain coordinator ordered.
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [
            executor.submit(_compute_cross_sectional_replicate, con, replicate=i, **kwargs)
            for i in pending
        ]
        computed_rows = zip(pending, (future.result() for future in futures))
    try:
        for _replicate, (summary, replicate_cell_stats) in computed_rows:
            if collect_cell_stats:
                cell_stats.extend(replicate_cell_stats)
            _append_checkpoint(checkpoint_path, summary, fingerprint=checkpoint_fingerprint)
            summaries.append(summary)
    finally:
        if workers > 1:
            executor.shutdown(wait=True)

    summaries.sort(key=lambda s: s["replicate"])
    replicate_ids = {s["replicate"] for s in summaries}
    if len(summaries) != n_replicates or len(replicate_ids) != n_replicates:
        raise RuntimeError(
            f"cross-sectional permutation replicate set is incomplete or duplicated: "
            f"expected {n_replicates} unique replicates, got {len(replicate_ids)}"
        )

    null_discovery_counts = [s["n_primary_discovery"] for s in summaries]
    p_empirical_count = empirical_discovery_count_p(real_discovery_count, null_discovery_counts)
    return {
        "replicate_summaries": summaries,
        "real_discovery_count": real_discovery_count,
        "null_discovery_counts": null_discovery_counts,
        "p_empirical_count": p_empirical_count,
        "n_replicates": n_replicates,
        "cell_stats": cell_stats,
    }


def run_temporal_placebo(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    long_horizon_registry: list[dict[str, Any]],
    real_t_nw_by_id: dict[str, float | None],
    config_hash: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    n_replicates: int = 100,
    min_shift_sessions: int = 120,
    p_max: float = 0.10,
    checkpoint_path: Path | None = None,
    scan_engine: str = "legacy",
    checkpoint_fingerprint: dict[str, Any] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """§A-6b: 100 circular date-shift placebos of the ``nw_lag>=59`` primary
    cells, each rescanned with the *same* gap-aware NW as the real cell.

    Every long-horizon feature is shifted by the *same* draw per replicate
    (never independently), then rejoined onto the fixed label frame by
    ``(formation_session_idx, ticker, market)`` — only the feature's calendar
    alignment breaks; its own time series, cross-section, and cross-feature
    correlation travel together. Resume-safe via ``checkpoint_path`` (JSONL),
    same reproducibility guarantee as :func:`run_cross_sectional_permutation`.
    """
    feature_cols = sorted({hyp["feature"] for hyp in long_horizon_registry})
    if workers < 1:
        raise ValueError("workers must be >= 1")
    target_cols = _registry_target_columns(long_horizon_registry)
    label_frame = fetch_broad_common_survivor_frame(
        con, panel_view=panel_view, extra_cols=target_cols, sample_start=sample_start
    )
    feature_cols_sql = ", ".join(feature_cols)
    feature_frame = con.execute(f"""
        SELECT ticker, market, formation_session_idx, {feature_cols_sql}
        FROM {panel_view}
        WHERE in_broad AND common_formation_120d AND common_survivor_120d
          AND trade_date >= DATE '{sample_start}'
        """).pl()
    total_sessions = int(feature_frame["formation_session_idx"].max())

    checkpoint = _load_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint)
    replicate_meta: list[dict[str, Any]] = [checkpoint[i] for i in sorted(checkpoint)]
    pending = [i for i in range(n_replicates) if i not in checkpoint]
    compute_kwargs = {
        "con": con,
        "panel_view": panel_view,
        "long_horizon_registry": long_horizon_registry,
        "config_hash": config_hash,
        "sample_start": sample_start,
        "min_names": min_names,
        "min_names_for_spread": min_names_for_spread,
        "quantile_count": quantile_count,
        "min_dates_per_cell": min_dates_per_cell,
        "scan_engine": scan_engine,
        "label_frame": label_frame,
        "feature_frame": feature_frame,
        "total_sessions": total_sessions,
        "min_shift_sessions": min_shift_sessions,
    }
    if workers == 1:
        computed = (_compute_temporal_replicate(i, **compute_kwargs) for i in pending)
        computed_rows = zip(pending, computed)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [
            executor.submit(_compute_temporal_replicate, i, **compute_kwargs) for i in pending
        ]
        computed_rows = zip(pending, (future.result() for future in futures))
    try:
        for _replicate, meta in computed_rows:
            _append_checkpoint(checkpoint_path, meta, fingerprint=checkpoint_fingerprint)
            replicate_meta.append(meta)
    finally:
        if workers > 1:
            executor.shutdown(wait=True)

    replicate_meta.sort(key=lambda m: m["replicate"])
    replicate_ids = {m["replicate"] for m in replicate_meta}
    if len(replicate_meta) != n_replicates or len(replicate_ids) != n_replicates:
        raise RuntimeError(
            f"temporal placebo replicate set is incomplete or duplicated: "
            f"expected {n_replicates} unique replicates, got {len(replicate_ids)}"
        )

    per_cell: dict[str, dict[str, Any]] = {}
    for hyp in long_horizon_registry:
        hid = hyp["hypothesis_id"]
        real_t = real_t_nw_by_id.get(hid)
        shifted = [
            m["abs_t_nw_by_id"][hid]
            for m in replicate_meta
            if m["abs_t_nw_by_id"].get(hid) is not None
        ]
        if real_t is None or not math.isfinite(real_t) or not shifted:
            per_cell[hid] = {"p_temporal_nw": None, "temporal_null_pass": False}
            continue
        per_cell[hid] = temporal_placebo_p(abs(real_t), shifted, p_max=p_max)

    return {
        "replicate_meta": replicate_meta,
        "per_cell": per_cell,
        "n_replicates": n_replicates,
        "total_sessions": total_sessions,
    }


def _compute_temporal_replicate(
    replicate: int,
    *,
    con: duckdb.DuckDBPyConnection,
    panel_view: str,
    long_horizon_registry: list[dict[str, Any]],
    config_hash: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str,
    label_frame: pl.DataFrame,
    feature_frame: pl.DataFrame,
    total_sessions: int,
    min_shift_sessions: int,
) -> dict[str, Any]:
    """Compute one temporal replicate; checkpoint writes stay coordinator-only."""
    seed = derive_replicate_seed(
        placebo_kind="temporal", replicate_index=replicate, config_hash=config_hash
    )
    shift = select_circular_shift_distance(
        seed=seed, total_sessions=total_sessions, min_shift=min_shift_sessions
    )
    shifted_features = apply_circular_feature_shift(
        feature_frame,
        session_col="formation_session_idx",
        shift=shift,
        total_sessions=total_sessions,
    )
    combined = label_frame.join(
        shifted_features, on=["formation_session_idx", "ticker", "market"], how="inner"
    )
    abs_t_nw_by_id: dict[str, float | None] = {}
    for hyp in long_horizon_registry:
        cell = scan_cell(
            con,
            panel_view=panel_view,
            feature_col=hyp["feature"],
            scan_type=hyp["scan_type"],
            h_start=hyp["h_start"],
            h_end=hyp["h_end"],
            universe="broad",
            sample_kind="common_survivor",
            sample_start=sample_start,
            min_names=min_names,
            min_names_for_spread=min_names_for_spread,
            quantile_count=quantile_count,
            min_dates_per_cell=min_dates_per_cell,
            expected_sign=hyp.get("expected_sign"),
            compute_spread=False,
            scan_engine=scan_engine,
            formation_frame=combined,
        )
        t_nw = cell["t_nw"]
        abs_t_nw_by_id[hyp["hypothesis_id"]] = (
            abs(t_nw) if t_nw is not None and math.isfinite(t_nw) else None
        )
    return {
        "replicate": replicate,
        "seed": seed,
        "shift": shift,
        "n_rows_after_join": combined.height,
        "n_tickers_after_join": combined["ticker"].n_unique(),
        "abs_t_nw_by_id": abs_t_nw_by_id,
    }


def run_lookahead_canary(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str = "analysis_panel",
    sample_start: str,
    min_names: int,
    min_dates_per_cell: int,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """A-6 synthetic look-ahead canary — never part of the 75-cell BH set or
    any official artifact (§A-6 completion note).

    Uses ``fwd_ret_1d`` (the *raw*, non-excess 1-day forward return) as the
    feature and compares it against ``label_scan``'s own excess-rank h=1
    target. ``raw_label_1d = fwd_ret_1d - bench_ret_1d`` subtracts a value
    that is constant within a given (date, market) cross-section, so it
    cannot change any name's rank there — the Spearman rank IC between the
    two must be ``1.0 ± tolerance`` in any non-degenerate cross-section. A
    broken join, an off-by-one session lag, or an accidental de-ranking
    anywhere in the label/panel pipeline would show up here first.
    """
    cell = scan_cell(
        con,
        panel_view=panel_view,
        feature_col="fwd_ret_1d",
        scan_type="cum",
        h_start=0,
        h_end=1,
        universe="broad",
        sample_kind="available",
        sample_start=sample_start,
        min_names=min_names,
        min_names_for_spread=min_names,
        quantile_count=5,
        min_dates_per_cell=min_dates_per_cell,
        compute_spread=False,
    )
    canary_pass = (
        cell["status"] == "valid"
        and cell["ic_mean"] is not None
        and abs(cell["ic_mean"] - 1.0) <= tolerance
    )
    return {**cell, "canary_pass": canary_pass}
