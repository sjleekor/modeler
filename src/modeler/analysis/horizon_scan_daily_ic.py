"""Stage 0: persist the per-date IC series the scan currently computes and drops.

``scan_cell`` builds ``market_ic`` (one row per date x market), collapses it to
one n-weighted ``daily`` IC per date, and returns only the scalar summary
(``ic_mean``/``t_nw``/``q5_spread_raw``/...). The daily series itself is
discarded, which is why no year-by-year IC table, no daily-IC family
correlation, and no regime-conditional (Phase C) statistic can be produced
from a published run.

This module is the side channel that keeps it. Nothing here computes a new
statistic: ``ParquetDailyIcSink.emit`` writes down frames ``scan_cell``
already built, and ``reconcile_daily_ic`` re-derives the summary from those
stored rows purely to prove the two agree.

Design: ``docs/dev/20260829_macro_features/01_design/01_stage0_daily_ic_persistence.md``.

Two contracts matter and are enforced by the tests:

1. **Default behaviour is unchanged.** Callers that pass no sink get exactly
   the old code path — the replicate loops, the period-split scans, and the
   direct lag1 calls never pass one, so they are excluded automatically.
2. **IC and spread are stored in two files, not one.** They have different
   date sets (``min_names`` vs. ``min_names_for_spread``, and a date whose
   cross-section is degenerate enough to give a NaN IC can still have a
   spread), so joining them would silently drop spread dates and break
   ``mean(spread) == q5_spread_raw``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from modeler.etl.metrics import choose_nw_lag, n_hac_pairs, newey_west_tstat

DAILY_IC_DIR_NAME = "daily_ic.parquet"
DAILY_SPREAD_DIR_NAME = "daily_spread.parquet"

MARKETS: tuple[str, ...] = ("KOSPI", "KOSDAQ")

# One normalized identity for a scan cell, written onto every daily row so a
# consumer never has to re-join the config to know what it is reading.
CELL_IDENTITY_COLUMNS: tuple[str, ...] = (
    "hypothesis_id",
    "family",
    "feature",
    "scan_type",
    "h_start",
    "h_end",
    "universe",
    "sample_kind",
    "hypothesis_role",
)

DAILY_IC_COLUMNS: tuple[str, ...] = (
    *CELL_IDENTITY_COLUMNS,
    "trade_date",
    "formation_session_idx",
    "rank_ic",
    "n_obs",
    "rank_ic_kospi",
    "n_kospi",
    "rank_ic_kosdaq",
    "n_kosdaq",
)

DAILY_SPREAD_COLUMNS: tuple[str, ...] = (
    *CELL_IDENTITY_COLUMNS,
    "trade_date",
    "spread",
    "n_spread_kospi",
    "n_spread_kosdaq",
)

# The cell key inside a stored file: `hypothesis_id` alone is not unique
# because `run_registry_scan` scans every hypothesis once per (universe,
# sample_kind) combo.
CELL_KEY_COLUMNS: tuple[str, ...] = ("hypothesis_id", "universe", "sample_kind")

_IDENTITY_DTYPES: dict[str, pl.DataType] = {
    "hypothesis_id": pl.Utf8,
    "family": pl.Utf8,
    "feature": pl.Utf8,
    "scan_type": pl.Utf8,
    "h_start": pl.Int64,
    "h_end": pl.Int64,
    "universe": pl.Utf8,
    "sample_kind": pl.Utf8,
    "hypothesis_role": pl.Utf8,
}

_PHASE_B_CELL_TYPE_TO_SCAN_TYPE = {"cumulative": "cum", "bucket": "bucket"}


class DailyIcSink(Protocol):
    """Where ``scan_cell`` hands off the per-date frames it just built.

    ``emit`` is called once per *valid* cell, after the summary statistics are
    final. An ``insufficient`` cell is never emitted: its ``daily`` frame is
    below ``min_dates_required``, so there is no summary to reconcile it
    against, and ``status_reason`` already records why.
    """

    def emit(
        self,
        cell: dict[str, Any],
        *,
        daily: pl.DataFrame,
        market_ic: pl.DataFrame,
        daily_spread: pl.DataFrame | None = None,
        market_spread: pl.DataFrame | None = None,
    ) -> None: ...

    def flush_feature(self, feature: str) -> None: ...


def normalize_cell_identity(
    hypothesis: dict[str, Any], *, universe: str, sample_kind: str
) -> dict[str, Any]:
    """Project a Phase A or Phase B registry row onto one identity schema.

    The two registries were built independently and disagree on two fields:
    Phase A rows carry ``hypothesis_role`` (``primary`` /
    ``exploratory_short_regime``) and always a ``scan_type``; Phase B rows
    carry ``role`` (``ready_primary``) and ``cell_type``
    (``cumulative``/``bucket``), with ``scan_type`` added on the way into
    ``run_registry_scan``. Normalizing here — in front of the sink — keeps
    that difference out of the stored schema.
    """
    scan_type = hypothesis.get("scan_type")
    if scan_type is None:
        cell_type = hypothesis.get("cell_type")
        scan_type = _PHASE_B_CELL_TYPE_TO_SCAN_TYPE.get(cell_type, cell_type)
    role = hypothesis.get("hypothesis_role") or hypothesis.get("role")
    return {
        "hypothesis_id": hypothesis["hypothesis_id"],
        "family": hypothesis["family"],
        "feature": hypothesis["feature"],
        "scan_type": scan_type,
        "h_start": int(hypothesis["h_start"]),
        "h_end": int(hypothesis["h_end"]),
        "universe": universe,
        "sample_kind": sample_kind,
        "hypothesis_role": role,
    }


def _identity_literals(cell: dict[str, Any]) -> list[pl.Expr]:
    return [
        pl.lit(cell.get(name), dtype=_IDENTITY_DTYPES[name]).alias(name)
        for name in CELL_IDENTITY_COLUMNS
    ]


def _market_slice(
    frame: pl.DataFrame, market: str, *, value_col: str, value_alias: str | None, n_alias: str
) -> pl.DataFrame:
    """One market's rows of a per-(date, market) frame, widened to columns.

    Done by filter-and-join rather than ``DataFrame.pivot`` so the output
    column names are fixed by this function instead of by the Polars version's
    multi-value pivot naming.
    """
    selects = [pl.col("trade_date")]
    if value_alias is not None:
        selects.append(pl.col(value_col).cast(pl.Float64).alias(value_alias))
    selects.append(pl.col("n").cast(pl.Int64).alias(n_alias))
    return frame.filter(pl.col("market") == market).select(selects)


def build_daily_ic_frame(
    cell: dict[str, Any], *, daily: pl.DataFrame, market_ic: pl.DataFrame
) -> pl.DataFrame:
    """The stored per-date IC rows for one cell.

    ``rank_ic``/``n_obs`` are the weighted series ``ic_mean`` is the mean of;
    the per-market columns are kept so (a) the weighting can be re-derived —
    that check is part of :func:`reconcile_daily_ic` — and (b) Phase C can
    read a market-conditional IC as a diagnostic without a rescan.
    """
    base = daily.select(
        pl.col("trade_date"),
        pl.col("formation_session_idx").cast(pl.Int64),
        pl.col("rank_ic").cast(pl.Float64),
        pl.col("n").cast(pl.Int64).alias("n_obs"),
    )
    for market in MARKETS:
        suffix = market.lower()
        base = base.join(
            _market_slice(
                market_ic,
                market,
                value_col="rank_ic",
                value_alias=f"rank_ic_{suffix}",
                n_alias=f"n_{suffix}",
            ),
            on="trade_date",
            how="left",
        )
    return (
        base.with_columns(
            *(pl.col(f"n_{market.lower()}").fill_null(0) for market in MARKETS),
        )
        .with_columns(_identity_literals(cell))
        .select(DAILY_IC_COLUMNS)
        .sort("trade_date")
    )


def build_daily_spread_frame(
    cell: dict[str, Any], *, daily_spread: pl.DataFrame, market_spread: pl.DataFrame
) -> pl.DataFrame:
    """The stored per-date quantile-spread rows for one cell."""
    base = daily_spread.select(
        pl.col("trade_date"),
        pl.col("spread").cast(pl.Float64),
    )
    for market in MARKETS:
        base = base.join(
            _market_slice(
                market_spread,
                market,
                value_col="spread",
                value_alias=None,
                n_alias=f"n_spread_{market.lower()}",
            ),
            on="trade_date",
            how="left",
        )
    return (
        base.with_columns(
            *(pl.col(f"n_spread_{market.lower()}").fill_null(0) for market in MARKETS),
        )
        .with_columns(_identity_literals(cell))
        .select(DAILY_SPREAD_COLUMNS)
        .sort("trade_date")
    )


@dataclass(frozen=True)
class DailyIcArtifactSummary:
    """File count / row count / content hash of one written artifact tree."""

    file_count: int
    row_count: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"file_count": self.file_count, "row_count": self.row_count, "sha256": self.sha256}


@dataclass(frozen=True)
class DailyIcSummary:
    """What :meth:`ParquetDailyIcSink.finalize` reports for ``manifest.json``."""

    daily_ic: DailyIcArtifactSummary
    daily_spread: DailyIcArtifactSummary

    def as_manifest_artifacts(self) -> dict[str, Any]:
        return {"daily_ic": self.daily_ic.as_dict(), "daily_spread": self.daily_spread.as_dict()}


def _tree_sha256(root: Path) -> str:
    """SHA256 over every file's (relative path, bytes), sorted by path."""
    hasher = hashlib.sha256()
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


class ParquetDailyIcSink:
    """Buffer per-cell daily frames, then write one parquet file per feature.

    Layout (``01_stage0_daily_ic_persistence.md`` §2.3, and the name
    ``04_specific_plan_B.md`` §7.1 originally reserved)::

        <out_dir>/daily_ic.parquet/family=<family>/<feature>.parquet
        <out_dir>/daily_spread.parquet/family=<family>/<feature>.parquet

    Buffering is per feature because that is the unit ``run_registry_scan``
    already works in: a feature's whole formation frame is live only while its
    hypotheses are being scanned, so flushing on the same boundary keeps peak
    memory at one feature's daily rows rather than the whole run's.

    ``run_id``/``config_hash``/``snapshot_date``/``source`` are in the run
    directory path and are deliberately not repeated as columns.
    """

    def __init__(self, out_dir: Path, *, compression: str = "zstd") -> None:
        self.daily_ic_root = Path(out_dir) / DAILY_IC_DIR_NAME
        self.daily_spread_root = Path(out_dir) / DAILY_SPREAD_DIR_NAME
        self._compression = compression
        self._ic_buffer: dict[tuple[str, str], list[pl.DataFrame]] = {}
        self._spread_buffer: dict[tuple[str, str], list[pl.DataFrame]] = {}
        self._ic_rows = 0
        self._ic_files = 0
        self._spread_rows = 0
        self._spread_files = 0

    def emit(
        self,
        cell: dict[str, Any],
        *,
        daily: pl.DataFrame,
        market_ic: pl.DataFrame,
        daily_spread: pl.DataFrame | None = None,
        market_spread: pl.DataFrame | None = None,
    ) -> None:
        key = (str(cell["family"]), str(cell["feature"]))
        self._ic_buffer.setdefault(key, []).append(
            build_daily_ic_frame(cell, daily=daily, market_ic=market_ic)
        )
        if daily_spread is None or market_spread is None or daily_spread.is_empty():
            # No spread was computed (or every cross-section fell below
            # min_names_for_spread) — q5_spread_raw is None for this cell too,
            # so writing an empty group would only invent a date set.
            return
        self._spread_buffer.setdefault(key, []).append(
            build_daily_spread_frame(cell, daily_spread=daily_spread, market_spread=market_spread)
        )

    def flush_feature(self, feature: str) -> None:
        """Write and drop every buffered group for ``feature``.

        A feature may appear under more than one family, so this writes one
        file per (family, feature) pair it finds buffered.
        """
        for key in [k for k in self._ic_buffer if k[1] == feature]:
            frames = self._ic_buffer.pop(key)
            rows = self._write_group(self.daily_ic_root, key, frames, DAILY_IC_COLUMNS)
            self._ic_rows += rows
            self._ic_files += 1
        for key in [k for k in self._spread_buffer if k[1] == feature]:
            frames = self._spread_buffer.pop(key)
            rows = self._write_group(self.daily_spread_root, key, frames, DAILY_SPREAD_COLUMNS)
            self._spread_rows += rows
            self._spread_files += 1

    def finalize(self) -> DailyIcSummary:
        """Flush whatever is still buffered, then summarize both trees."""
        pending = {k[1] for k in self._ic_buffer} | {k[1] for k in self._spread_buffer}
        for feature in sorted(pending):
            self.flush_feature(feature)
        return DailyIcSummary(
            daily_ic=DailyIcArtifactSummary(
                file_count=self._ic_files,
                row_count=self._ic_rows,
                sha256=_tree_sha256(self.daily_ic_root),
            ),
            daily_spread=DailyIcArtifactSummary(
                file_count=self._spread_files,
                row_count=self._spread_rows,
                sha256=_tree_sha256(self.daily_spread_root),
            ),
        )

    def _write_group(
        self,
        root: Path,
        key: tuple[str, str],
        frames: list[pl.DataFrame],
        columns: tuple[str, ...],
    ) -> int:
        family, feature = key
        target_dir = root / f"family={family}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{feature}.parquet"
        if target.exists():
            raise FileExistsError(
                f"daily IC file already written for family={family} feature={feature}: {target}"
            )
        frame = pl.concat(frames, how="vertical").select(columns)
        frame.write_parquet(target, compression=self._compression)
        return frame.height


# --- §4.1 reconciliation: rebuild the summary from what was stored ---


@dataclass(frozen=True)
class _Check:
    name: str
    stored: float | int | None
    summary: float | int | None
    tolerance: float


def _tree_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def _cell_groups(frame: pl.DataFrame) -> dict[tuple[Any, ...], pl.DataFrame]:
    return {tuple(key): group for key, group in frame.group_by(list(CELL_KEY_COLUMNS))}


def _both_nan(a: float | int | None, b: float | int | None) -> bool:
    return isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)


def _diff(check: _Check) -> float | None:
    """Absolute difference, or ``None`` when the pair is trivially equal."""
    if check.stored is None and check.summary is None:
        return None
    if check.stored is None or check.summary is None:
        return float("inf")
    if _both_nan(check.stored, check.summary):
        return None
    stored = float(check.stored)
    summary = float(check.summary)
    if not math.isfinite(stored) or not math.isfinite(summary):
        return float("inf")
    return abs(stored - summary)


def _cell_checks(
    group: pl.DataFrame, row: dict[str, Any], spread: pl.DataFrame | None
) -> list[_Check]:
    ordered = group.sort("formation_session_idx")
    values = ordered["rank_ic"].to_numpy()
    sessions = ordered["formation_session_idx"].to_numpy()
    n_dates = ordered.height
    lag = choose_nw_lag(
        scan_type=row["scan_type"],
        horizon=row["h_end"] if row["scan_type"] == "cum" else None,
        bucket_width=(row["h_end"] - row["h_start"]) if row["scan_type"] == "bucket" else None,
    )
    ic_mean = float(values.mean())
    ic_std = float(values.std(ddof=1)) if n_dates > 1 else float("nan")
    finite_std = bool(ic_std) and math.isfinite(ic_std) and ic_std != 0
    icir = ic_mean / ic_std if finite_std else float("nan")
    t_naive = ic_mean / (ic_std / math.sqrt(n_dates)) if finite_std else float("nan")

    # `n_obs*` in the summary are statistics of the per-(date, market) counts,
    # not of the per-date totals — reassemble that series from the two market
    # columns (a market absent that date is stored as 0, never as a real group).
    n_kospi = ordered["n_kospi"].to_numpy()
    n_kosdaq = ordered["n_kosdaq"].to_numpy()
    per_market = np.concatenate([n_kospi[n_kospi > 0], n_kosdaq[n_kosdaq > 0]])
    total = ordered["n_obs"].to_numpy()
    kospi_weight = (n_kospi / total).mean() if total.size else float("nan")

    weighted = (
        np.nan_to_num(ordered["rank_ic_kospi"].to_numpy(), nan=0.0) * n_kospi
        + np.nan_to_num(ordered["rank_ic_kosdaq"].to_numpy(), nan=0.0) * n_kosdaq
    ) / total
    weighting_error = float(np.abs(weighted - values).max()) if n_dates else 0.0

    checks = [
        _Check("n_dates", n_dates, row.get("n_dates"), 0.0),
        _Check("ic_mean", ic_mean, row.get("ic_mean"), 1e-12),
        _Check("ic_std", ic_std, row.get("ic_std"), 1e-12),
        _Check("icir", icir, row.get("icir"), 1e-9),
        _Check("t_naive", t_naive, row.get("t_naive"), 1e-9),
        _Check("t_nw", newey_west_tstat(values, sessions, lag), row.get("t_nw"), 1e-9),
        _Check(
            "n_hac_pairs_min",
            n_hac_pairs(sessions, lag) if lag > 0 else 0,
            row.get("n_hac_pairs_min"),
            0.0,
        ),
        _Check("n_obs", int(per_market.sum()), row.get("n_obs"), 0.0),
        _Check("n_obs_min", int(per_market.min()), row.get("n_obs_min"), 0.0),
        _Check("n_obs_mean", float(per_market.mean()), row.get("n_obs_mean"), 1e-12),
        _Check("n_obs_median", float(np.median(per_market)), row.get("n_obs_median"), 1e-12),
        _Check("kospi_weight_mean", float(kospi_weight), row.get("kospi_weight_mean"), 1e-12),
        _Check("market_weighted_ic", weighting_error, 0.0, 1e-12),
    ]
    stored_spread = float(spread["spread"].mean()) if spread is not None and spread.height else None
    checks.append(_Check("q5_spread_raw", stored_spread, row.get("q5_spread_raw"), 1e-12))
    return checks


def reconcile_daily_ic(
    scanned_rows: list[dict[str, Any]],
    *,
    daily_ic_dir: Path,
    daily_spread_dir: Path,
) -> dict[str, Any]:
    """Rebuild every valid cell's summary from the stored daily rows.

    This is Stage 0's whole acceptance criterion: what was stored has to
    reproduce ``horizon_ic.parquet``'s own numbers, or one of the two is
    wrong and the run must not publish. Per-item tolerances follow
    ``01_stage0_daily_ic_persistence.md`` §4.1.

    Files are read one at a time. A cell never spans two files (one file per
    family/feature, and a hypothesis belongs to exactly one feature), so this
    loses nothing and keeps peak memory at one feature's rows rather than a
    whole phase's ~1M.
    """
    daily_ic_dir = Path(daily_ic_dir)
    daily_spread_dir = Path(daily_spread_dir)
    valid_by_key = {
        tuple(row[name] for name in CELL_KEY_COLUMNS): row
        for row in scanned_rows
        if row.get("status") == "valid"
    }

    mismatches: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    seen: set[tuple[Any, ...]] = set()
    spread_files_used: set[Path] = set()

    for ic_file in _tree_files(daily_ic_dir):
        spread_file = daily_spread_dir / ic_file.relative_to(daily_ic_dir)
        spread_groups: dict[tuple[Any, ...], pl.DataFrame] = {}
        if spread_file.is_file():
            spread_files_used.add(spread_file)
            spread_groups = _cell_groups(pl.read_parquet(spread_file))
        for key, group in _cell_groups(pl.read_parquet(ic_file)).items():
            seen.add(key)
            row = valid_by_key.get(key)
            if row is None:
                mismatches.append({"cell": key, "check": "unexpected_daily_ic", "diff": None})
                continue
            for check in _cell_checks(group, row, spread_groups.pop(key, None)):
                diff = _diff(check)
                if diff is None:
                    continue
                max_abs_diff = max(max_abs_diff, diff)
                if diff > check.tolerance:
                    mismatches.append({"cell": key, "check": check.name, "diff": diff})
        for key in sorted(spread_groups, key=str):
            mismatches.append({"cell": key, "check": "unexpected_daily_spread", "diff": None})

    for key in sorted(valid_by_key.keys() - seen, key=str):
        mismatches.append({"cell": key, "check": "missing_daily_ic", "diff": None})
        max_abs_diff = float("inf")

    for spread_file in _tree_files(daily_spread_dir):
        if spread_file in spread_files_used:
            continue
        for key in sorted(_cell_groups(pl.read_parquet(spread_file)), key=str):
            mismatches.append({"cell": key, "check": "unexpected_daily_spread", "diff": None})

    return {
        "reconciled": not mismatches,
        "n_cells": len(valid_by_key),
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches,
    }


def assert_daily_ic_reconciled(result: dict[str, Any]) -> dict[str, Any]:
    """Raise unless every stored cell rebuilt its summary; return the result.

    A failure here is a run failure, not a warning: the stored series and the
    published summary disagree, so at least one of them is wrong and the run
    cannot be official.
    """
    if not result["reconciled"]:
        head = result["mismatches"][:10]
        raise RuntimeError(
            f"daily_ic reconciliation failed for {len(result['mismatches'])} check(s); "
            f"first: {head}"
        )
    return result


def daily_ic_success_fields(result: dict[str, Any]) -> dict[str, Any]:
    """The two fields Stage 0 adds to ``_SUCCESS.json`` (§2.5)."""
    return {
        "daily_ic_reconciled": bool(result["reconciled"]),
        "daily_ic_reconcile_max_abs_diff": float(result["max_abs_diff"]),
    }
