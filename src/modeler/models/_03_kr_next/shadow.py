"""Shadow serving of the next KR model: daily scoring into an append-only, hash-chained record.

Spec: ``my/milestones/kr/modeling/plan/20261018_kr_next_prereg/01_preregistration_draft.md``
§9-§11 (main proposals G-2..G-5, G-7, G-16, G-22). This module is *separate* from the briefing
serving (``modeler.serving``): it imports nothing from there, has its own lock, its own
directory and its own ledger.

One call = one report date ``D`` (decision 10:00 KST, entry at the D close):

1. ``t`` = the session before ``D`` (``sessions`` = KOSPI sessions, ascending).
2. Panel of ``t`` in serving mode (``dataset.build_kr_next_panel(..., include_labels=False)``,
   or a prebuilt panel). It must carry ``liq_notice_flag`` and ``liq_notice_rcept_no``
   (the 정리매매 serving rule G-21, produced by the panel builder of the branch kr-next-data).
3. Candidates = panel rows (universe and one-per-company are already applied by the panel)
   that are not 정리매매-flagged and have a score. Flagged names are removed *before* the
   rank transform: training dropped the same names from the universe before ranking, so the
   cross-section the ranks are taken over stays the same kind of set.
4. ``preprocess.rank_transform`` exactly as training, then the frozen bundle scores
   (HGB ``score``, Ridge ``ridge_score``). The bundle is loaded with ``verify_golden=True``.
5. Rank = score desc, trailing 20-session KRX trading value desc, ticker asc.
   The trading value is ``px_turnover_ma20_v2``: the mean of the KRX ``trading_value`` over
   20 sessions (``feat_price_v2``), read one valid session late like every KRX raw column
   (C3-4). That is exactly "trailing 20-session KRX trading value", so no extra input column
   is needed.
6. Tail flags (G-7) on the candidate set, from this panel's values.
7. On grid dates (``D0 + 20k`` sessions, ``D0`` = the ``D`` of the first *valid* ledger record,
   or ``D`` itself when the ledger has none and this record is valid) the 30-name target
   basket (G-5) with the previous counting grid record's target as holdings and the lagged
   KRX market cap ``exp(mcap_krx_log)``. Drifted weights are the forward job's business.
8. Files under ``out_root/D=<D>/`` (or ``.../D=<D>/attempt-<n>/``): ``panel.parquet``,
   ``predictions.parquet``, ``basket.parquet`` (grid dates), ``record.json``; one line is
   appended to the ledger.

Ledger line (``ledger.jsonl``, one JSON object per line, sorted keys, compact)::

    {schema, D, attempt, dir, record_sha256, valid, written_at, has_basket,
     prev_ledger_sha256}

``prev_ledger_sha256`` = sha256 of the previous line's exact text (without the newline),
``"0" * 64`` for the first line, so editing, dropping or reordering any line breaks the chain.
``record_sha256`` = sha256 of ``record.json``'s bytes; ``record.json`` lists the sha256 of the
other files. :func:`verify_ledger` recomputes all of it.

Only the *first valid attempt* of a ``D`` counts (``counting_records``): a rerun of the same
``D`` is written to ``D=<D>/attempt-<n>/`` and recorded as its own ledger line, never
overwrites, and never changes holdings or ``D0``. ``valid`` = ``written_at <= D 15:00 KST``
(G-2: the write has to precede the entry at the D close).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time as time_mod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import polars as pl

from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import evaluate as ev
from modeler.models._03_kr_next import preprocess as pp

SEOUL = ZoneInfo("Asia/Seoul")
SCHEMA_RECORD = "kr-next-shadow-record.v1"
SCHEMA_LEDGER = "kr-next-shadow-ledger.v1"
GENESIS_HASH = "0" * 64
DECISION_TIME = time(10, 0)
DEADLINE_TIME = time(15, 0)

LIQ_FLAG_COL = "liq_notice_flag"
LIQ_RCEPT_COL = "liq_notice_rcept_no"
KEYS = ("trade_date", "ticker", "market")
#: raw (pre-rank) inputs of the avoid-tail cell (G-7)
TAIL_INPUT_COLS = ("px_vol_20d_v2", "px_idio_vol_60d", "fin_roa_v2", "fin_capital_impaired_v2")
TV20_COL = "px_turnover_ma20_v2"
MCAP_LOG_COL = "mcap_krx_log"
FILE_PANEL = "panel.parquet"
FILE_PREDICTIONS = "predictions.parquet"
FILE_BASKET = "basket.parquet"
FILE_RECORD = "record.json"
LOCK_TIMEOUT_SECONDS = 900.0

_HEX = re.compile(r"^[0-9a-f]{64}$")


class ShadowError(RuntimeError):
    """The shadow record cannot be written (or the ledger is not intact)."""


class ShadowContractError(ShadowError):
    """The panel does not carry what the shadow serving needs."""


# --------------------------------------------------------------------------- small helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return bundle_mod.sha256_file(Path(path))


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent else (",", ":"),
        default=str,
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """tmp file in the same directory + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _lock(path: Path, timeout: float = LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Exclusive flock with a time limit (the pattern of ``serving.orchestration._run_lock``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        deadline = time_mod.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time_mod.monotonic() >= deadline:
                    raise ShadowError(f"shadow lock busy for {timeout:.0f}s: {path}") from None
                time_mod.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def code_files(package_dir: Path | None = None) -> dict[str, str]:
    """``{file name: sha256}`` of every ``*.py`` of this package (``_03_kr_next``)."""
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    return {p.name: sha256_file(p) for p in sorted(root.glob("*.py"))}


def code_inventory_sha256(files: Mapping[str, str]) -> str:
    return sha256_bytes(_json_dumps(sorted(files.items())).encode("utf-8"))


def decision_at(d: date) -> datetime:
    return datetime.combine(d, DECISION_TIME, tzinfo=SEOUL)


def deadline_at(d: date) -> datetime:
    return datetime.combine(d, DEADLINE_TIME, tzinfo=SEOUL)


def previous_session(sessions: Sequence[date], d: date) -> date:
    ordered = sorted(set(sessions))
    if d not in ordered:
        raise ShadowError(f"{d} is not a session of the given calendar")
    i = ordered.index(d)
    if i == 0:
        raise ShadowError(f"{d} is the first session of the calendar, no previous session")
    return ordered[i - 1]


def dir_name(d: date, attempt: int) -> str:
    """Directory of a record, relative to ``out_root`` (POSIX)."""
    base = f"D={d.isoformat()}"
    return base if attempt == 1 else f"{base}/attempt-{attempt}"


# --------------------------------------------------------------------------- ledger


def _read_ledger_lines(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if text == "":
        return []
    if not text.endswith("\n"):
        raise ShadowError("ledger does not end with a newline (torn write?)")
    return text[:-1].split("\n")


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Parsed ledger lines (no verification; see :func:`verify_ledger`)."""
    out = []
    for n, line in enumerate(_read_ledger_lines(Path(path)), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ShadowError(f"ledger line {n} is not JSON") from exc
    return out


def counting_records(lines: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """``{D: ledger line}``: the first *valid* attempt of each ``D`` (only it counts)."""
    out: dict[str, Mapping[str, Any]] = {}
    for line in lines:
        if line.get("valid") and line["D"] not in out:
            out[line["D"]] = line
    return out


def _append_ledger(path: Path, line_obj: dict[str, Any]) -> str:
    """Append one line by rewriting the file atomically; returns the new line's sha256."""
    lines = _read_ledger_lines(path)
    text = _json_dumps(line_obj)
    new = "\n".join([*lines, text]) + "\n"
    _atomic_write_bytes(path, new.encode("utf-8"))
    return sha256_bytes(text.encode("utf-8"))


def verify_ledger(path: str | Path, out_root: str | Path | None = None) -> dict[str, Any]:
    """Recompute the chain and every file hash. ``out_root`` defaults to the ledger's folder.

    Returns ``{"ok": bool, "n_lines": int, "breaks": [str, ...], "orphans": [str, ...]}``.
    ``breaks`` name the first problems found (line number and what differs); ``orphans`` are
    ``D=*`` directories on disk that no ledger line points to (a crash between the directory
    rename and the ledger append, or a deleted line).
    """
    path = Path(path)
    root = Path(out_root) if out_root is not None else path.parent
    breaks: list[str] = []
    try:
        lines = _read_ledger_lines(path)
    except ShadowError as exc:
        return {"ok": False, "n_lines": 0, "breaks": [str(exc)], "orphans": []}
    prev = GENESIS_HASH
    seen_dirs: set[str] = set()
    last_d = ""
    attempts: dict[str, int] = {}
    for n, text in enumerate(lines, 1):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            breaks.append(f"line {n}: not JSON")
            prev = sha256_bytes(text.encode("utf-8"))
            continue
        if obj.get("prev_ledger_sha256") != prev:
            breaks.append(f"line {n}: prev_ledger_sha256 does not match the previous line")
        prev = sha256_bytes(text.encode("utf-8"))
        if obj.get("schema") != SCHEMA_LEDGER:
            breaks.append(f"line {n}: unknown schema {obj.get('schema')!r}")
        d, attempt = obj.get("D"), obj.get("attempt")
        if not isinstance(d, str) or not isinstance(attempt, int):
            breaks.append(f"line {n}: D/attempt malformed")
            continue
        if d < last_d:
            breaks.append(f"line {n}: D {d} is earlier than the previous line's {last_d}")
        last_d = max(last_d, d)
        if attempts.get(d, 0) + 1 != attempt:
            breaks.append(f"line {n}: attempt {attempt} of D={d} is out of sequence")
        attempts[d] = attempt
        rel = obj.get("dir")
        if rel != dir_name(date.fromisoformat(d), attempt):
            breaks.append(f"line {n}: dir {rel!r} does not match D/attempt")
            continue
        seen_dirs.add(rel)
        rec_path = root / rel / FILE_RECORD
        if not rec_path.is_file():
            breaks.append(f"line {n}: {rel}/{FILE_RECORD} is missing")
            continue
        rec_bytes = rec_path.read_bytes()
        if sha256_bytes(rec_bytes) != obj.get("record_sha256"):
            breaks.append(f"line {n}: {rel}/{FILE_RECORD} sha256 differs from the ledger")
            continue
        try:
            rec = json.loads(rec_bytes)
        except json.JSONDecodeError:
            breaks.append(f"line {n}: {rel}/{FILE_RECORD} is not JSON")
            continue
        if (rec.get("D"), rec.get("attempt"), rec.get("valid")) != (d, attempt, obj.get("valid")):
            breaks.append(f"line {n}: record.json D/attempt/valid differ from the ledger line")
        for name, want in sorted((rec.get("files") or {}).items()):
            f = root / rel / name
            if not f.is_file():
                breaks.append(f"line {n}: {rel}/{name} is missing")
            elif sha256_file(f) != want:
                breaks.append(f"line {n}: {rel}/{name} sha256 differs from record.json")
        if bool(obj.get("has_basket")) != (FILE_BASKET in (rec.get("files") or {})):
            breaks.append(f"line {n}: has_basket differs from record.json")
    orphans: list[str] = []
    if root.is_dir():
        for dd in sorted(root.glob("D=*")):
            if not dd.is_dir():
                continue
            if dd.name not in seen_dirs:
                orphans.append(dd.name)
            for ad in sorted(dd.glob("attempt-*")):
                rel = f"{dd.name}/{ad.name}"
                if rel not in seen_dirs:
                    orphans.append(rel)
    return {
        "ok": not breaks and not orphans,
        "n_lines": len(lines),
        "breaks": breaks,
        "orphans": orphans,
    }


# --------------------------------------------------------------------------- panel


def build_serving_panel(
    con: duckdb.DuckDBPyConnection,
    t: date,
    *,
    feature_set: ds.FeatureSet = "noflow",
    views: ds.KrNextViews | None = None,
    mart_overrides: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """The serving-mode panel of the signal date ``t`` (no label column is built)."""
    return ds.build_kr_next_panel(
        con,
        feature_set,
        views=views,
        include_labels=False,
        start_date=t.isoformat(),
        end_date=t.isoformat(),
        mart_overrides=mart_overrides,
    )


def _check_panel(panel: pl.DataFrame, feature_cols: Sequence[str]) -> None:
    missing_liq = [c for c in (LIQ_FLAG_COL, LIQ_RCEPT_COL) if c not in panel.columns]
    if missing_liq:
        raise ShadowContractError(
            f"panel lacks {missing_liq}: the 정리매매 serving rule (G-21) columns come from the "
            "kr-next-data panel builder; the shadow serving does not guess them"
        )
    need = [*KEYS, *feature_cols, *TAIL_INPUT_COLS, TV20_COL, MCAP_LOG_COL]
    missing = sorted(set(c for c in need if c not in panel.columns))
    if missing:
        raise ShadowContractError(f"panel lacks columns: {missing}")
    if panel.select(pl.struct(["ticker", "market"]).is_duplicated().any()).item():
        raise ShadowContractError("panel has a (ticker, market) twice on the signal date")
    if panel["ticker"].n_unique() != panel.height:
        raise ShadowContractError("a ticker appears in two markets on the signal date")


# --------------------------------------------------------------------------- scoring


def _predictions(
    panel_t: pl.DataFrame, bundle: bundle_mod.LoadedBundle, lagged: Sequence[str], d: date
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Candidates -> scores -> rank -> tail flags. Returns (predictions, counts)."""
    feature_cols = bundle.feature_cols
    n_flag_null = int(panel_t[LIQ_FLAG_COL].null_count())
    cands = panel_t.filter(~pl.col(LIQ_FLAG_COL).fill_null(False))
    n_flagged = panel_t.height - cands.height
    if cands.height == 0:
        raise ShadowError("no candidates after the 정리매매 exclusion")

    raw = cands.select(
        [
            *KEYS,
            *TAIL_INPUT_COLS,
            pl.col(TV20_COL).cast(pl.Float64).alias("tv20"),
            pl.col(MCAP_LOG_COL).cast(pl.Float64).exp().alias("mcap_krx"),
            pl.col(LIQ_FLAG_COL).fill_null(False).alias(LIQ_FLAG_COL),
            pl.col(LIQ_RCEPT_COL),
            *[pl.col(c).is_null().alias(f"lagnull_{c}") for c in lagged],
        ]
    )
    ranked = pp.rank_transform(
        cands.select([*KEYS, *feature_cols]).with_columns(
            [pl.col(c).cast(pl.Float64) for c in feature_cols]
        ),
        feature_cols,
    )
    score = bundle.predict(ranked)
    ridge = bundle.predict_ridge(ranked) if bundle.ridge is not None else None
    pred = raw.with_columns(
        pl.Series("score", score, dtype=pl.Float64),
        pl.Series("ridge_score", ridge, dtype=pl.Float64)
        if ridge is not None
        else pl.lit(None, dtype=pl.Float64).alias("ridge_score"),
    )
    n_all = pred.height
    pred = pred.filter(pl.col("score").is_not_null() & pl.col("score").is_not_nan())
    n_unscored = n_all - pred.height
    if pred.height == 0:
        raise ShadowError("no candidate has a score")
    pred = (
        pred.sort(["score", "tv20", "ticker"], descending=[True, True, False], nulls_last=True)
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("rank").cast(pl.Int64))
    )
    pred = ev.tail_flags(pred).with_columns(pl.lit(d).alias("decision_date"))
    cols = [
        "trade_date",
        "decision_date",
        "ticker",
        "market",
        "score",
        "rank",
        "ridge_score",
        "tail1",
        "tail2",
        "tail",
        *TAIL_INPUT_COLS,
        "px_vol_20d_v2_pct",
        "px_idio_vol_60d_pct",
        "fin_roa_v2_pct",
        "tv20",
        "mcap_krx",
        LIQ_FLAG_COL,
        LIQ_RCEPT_COL,
        *[f"lagnull_{c}" for c in lagged],
    ]
    counts = {
        "panel_rows": panel_t.height,
        "liq_flag_null_rows": n_flag_null,
        "liq_flagged_excluded": n_flagged,
        "unscored_excluded": n_unscored,
        "candidates": pred.height,
    }
    return pred.select(cols), counts


def _previous_target(
    ledger: Sequence[Mapping[str, Any]], out_root: Path, d: date
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """Target weights of the latest counting grid record before ``d`` (or ``None``)."""
    counting = counting_records(ledger)
    prior = sorted(
        k for k, line in counting.items() if line.get("has_basket") and k < d.isoformat()
    )
    if not prior:
        return None, None
    line = counting[prior[-1]]
    rel = Path(line["dir"])
    rec = json.loads((out_root / rel / FILE_RECORD).read_bytes())
    f = out_root / rel / FILE_BASKET
    if sha256_file(f) != rec["files"][FILE_BASKET]:
        raise ShadowError(f"previous basket {rel}/{FILE_BASKET} differs from its record.json")
    basket = pl.read_parquet(f)
    weights = dict(zip(basket["ticker"].to_list(), basket["target_weight"].to_list(), strict=True))
    return weights, {"D": line["D"], "dir": line["dir"], "record_sha256": line["record_sha256"]}


def _basket_frame(
    target: Mapping[str, float], pred: pl.DataFrame, prev: Mapping[str, float] | None
) -> pl.DataFrame:
    held = set(prev or {})
    sel = pred.filter(pl.col("ticker").is_in(list(target))).select(
        "ticker", "market", "rank", "score", "mcap_krx", "tv20"
    )
    return (
        sel.with_columns(
            pl.col("ticker")
            .replace_strict(dict(target), return_dtype=pl.Float64)
            .alias("target_weight"),
            pl.col("ticker").is_in(list(held)).alias("was_held"),
        )
        .sort("rank")
        .select(
            "ticker", "market", "target_weight", "rank", "score", "mcap_krx", "tv20", "was_held"
        )
    )


# --------------------------------------------------------------------------- the daily call


def score_day(
    d: date,
    *,
    lake: duckdb.DuckDBPyConnection | None = None,
    bundle_dir: str | Path,
    out_root: str | Path,
    ledger_path: str | Path,
    sessions: Sequence[date],
    now: datetime | None = None,
    panel: pl.DataFrame | None = None,
    panel_manifest: Mapping[str, Any] | None = None,
    views: ds.KrNextViews | None = None,
    mart_overrides: Mapping[str, str] | None = None,
    feature_set: ds.FeatureSet = "noflow",
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Score report date ``d`` with the frozen bundle and append the shadow record.

    ``lake`` = a DuckDB connection with the lake views registered (as ``dataset`` expects);
    ``panel`` = a prebuilt panel (tests, or a panel the prepare step wrote) that is used
    instead. ``panel_manifest`` is recorded verbatim (view SQL hashes G-22, the prepare
    manifest hash, snapshot id...). ``now`` is the write time (default: the clock, taken after
    scoring); ``valid`` is ``now <= d 15:00 KST``. Returns ``{"dir", "record", "ledger_line"}``.
    """
    out_root, ledger_path = Path(out_root), Path(ledger_path)
    if panel is None and lake is None:
        raise ValueError("pass either a lake connection or a prebuilt panel")
    t = previous_session(sessions, d)
    with _lock(ledger_path.with_name(ledger_path.name + ".lock"), lock_timeout):
        ledger = read_ledger(ledger_path)
        chain = verify_ledger(ledger_path, out_root)
        if chain["breaks"]:
            raise ShadowError(f"ledger chain is broken, refusing to append: {chain['breaks'][:3]}")

        bundle = bundle_mod.load_bundle(bundle_dir, verify_golden=True)
        if bundle.feature_cols != list(ds.feature_columns(feature_set)):
            raise ShadowError("bundle feature list differs from the feature set of the panel")
        if panel is None:
            assert lake is not None
            panel = build_serving_panel(
                lake, t, feature_set=feature_set, views=views, mart_overrides=mart_overrides
            )
        _check_panel(panel, bundle.feature_cols)
        panel_t = panel.filter(pl.col("trade_date") == t).sort(["ticker", "market"])
        if panel_t.height == 0:
            raise ShadowError(f"panel has no rows on the signal date {t}")
        lagged = ds.lagged_columns(feature_set)
        pred, counts = _predictions(panel_t, bundle, lagged, d)

        written_at = (now or datetime.now(SEOUL)).astimezone(SEOUL)
        valid = written_at <= deadline_at(d)
        counting = counting_records(ledger)
        d0_s = min(counting) if counting else (d.isoformat() if valid else None)
        d0 = date.fromisoformat(d0_s) if d0_s else None
        grid = False
        grid_k = None
        if d0 is not None and d >= d0:
            grid_list = ev.grid_dates(sessions, d0)
            grid = d in grid_list
            grid_k = grid_list.index(d) if grid else None

        prev_weights: dict[str, float] | None = None
        prev_ref: dict[str, Any] | None = None
        basket_df: pl.DataFrame | None = None
        if grid:
            prev_weights, prev_ref = _previous_target(ledger, out_root, d)
            target = ev.build_basket(
                prev_weights,
                pred.select("ticker", "score", "tv20", pl.col("mcap_krx").alias("mcap")),
            )
            basket_df = _basket_frame(target, pred, prev_weights)

        attempt = 1
        if (out_root / dir_name(d, attempt)).exists():
            attempt = 2
            while (out_root / dir_name(d, attempt)).exists():
                attempt += 1
        rel = dir_name(d, attempt)
        final = out_root / rel
        out_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-D={d.isoformat()}-", dir=out_root))
        try:
            panel_t.write_parquet(tmp / FILE_PANEL)
            pred.write_parquet(tmp / FILE_PREDICTIONS)
            files = {
                FILE_PANEL: sha256_file(tmp / FILE_PANEL),
                FILE_PREDICTIONS: sha256_file(tmp / FILE_PREDICTIONS),
            }
            if basket_df is not None:
                basket_df.write_parquet(tmp / FILE_BASKET)
                files[FILE_BASKET] = sha256_file(tmp / FILE_BASKET)
            cfiles = code_files()
            bm = bundle.manifest
            record = {
                "schema_version": SCHEMA_RECORD,
                "D": d.isoformat(),
                "t": t.isoformat(),
                "attempt": attempt,
                "decision_at": decision_at(d).isoformat(),
                "deadline_at": deadline_at(d).isoformat(),
                "written_at": written_at.isoformat(timespec="seconds"),
                "valid": valid,
                "is_grid_date": grid,
                "grid_k": grid_k,
                "d0": d0.isoformat() if d0 else None,
                "feature_set": feature_set,
                "lagged_cols": list(lagged),
                "counts": counts,
                "files": files,
                "bundle": {
                    "manifest_sha256": sha256_file(Path(bundle_dir) / bundle_mod.MANIFEST_FILE),
                    "files": bm["files"],
                    "feature_list_sha256": bm["feature_list_sha256"],
                    "snapshot_id": bm.get("snapshot_id"),
                    "prereg_tag": bm.get("prereg_tag"),
                    "prereg_doc_sha256": bm.get("prereg_doc_sha256"),
                    "modeler_commit": bm.get("modeler_commit"),
                },
                "code": {"files": cfiles, "inventory_sha256": code_inventory_sha256(cfiles)},
                "panel_manifest": dict(panel_manifest) if panel_manifest else None,
                "panel_manifest_sha256": sha256_bytes(_json_dumps(dict(panel_manifest)).encode())
                if panel_manifest
                else None,
                "basket": None
                if basket_df is None
                else {
                    "n": basket_df.height,
                    "first_build": prev_weights is None,
                    "previous": prev_ref,
                    "fewer_than_k_candidates": counts["candidates"] < ev.BASKET_K,
                },
            }
            rec_bytes = (_json_dumps(record, indent=2) + "\n").encode("utf-8")
            (tmp / FILE_RECORD).write_bytes(rec_bytes)
            os.rename(tmp, final)  # atomic; fails when ``final`` exists as a non-empty directory
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        prev_hash = (
            sha256_bytes(_read_ledger_lines(ledger_path)[-1].encode("utf-8"))
            if ledger
            else GENESIS_HASH
        )
        line = {
            "schema": SCHEMA_LEDGER,
            "D": d.isoformat(),
            "attempt": attempt,
            "dir": rel,
            "record_sha256": sha256_bytes(rec_bytes),
            "valid": valid,
            "written_at": record["written_at"],
            "has_basket": basket_df is not None,
            "prev_ledger_sha256": prev_hash,
        }
        _append_ledger(ledger_path, line)
    return {"dir": final, "record": record, "ledger_line": line}


# --------------------------------------------------------------------------- command line


def _load_sessions(path: Path) -> list[date]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["sessions"] if isinstance(data, dict) else data
    return [date.fromisoformat(str(x)) for x in items]


def main(argv: Sequence[str] | None = None) -> int:
    """``score`` a day from a prebuilt panel parquet, or ``verify`` the ledger."""
    p = argparse.ArgumentParser(prog="kr_next_shadow", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--date", required=True, type=date.fromisoformat, help="report date D")
    s.add_argument("--panel", required=True, type=Path, help="serving-mode panel parquet (day t)")
    s.add_argument("--panel-manifest", type=Path, default=None)
    s.add_argument("--bundle-dir", required=True, type=Path)
    s.add_argument("--out-root", required=True, type=Path)
    s.add_argument("--ledger", required=True, type=Path)
    s.add_argument("--sessions", required=True, type=Path, help="JSON list or calendar manifest")
    v = sub.add_parser("verify")
    v.add_argument("--ledger", required=True, type=Path)
    v.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        if args.cmd == "verify":
            report = verify_ledger(args.ledger, args.out_root)
            print(_json_dumps(report, indent=2))
            return 0 if report["ok"] else 1
        manifest = (
            json.loads(args.panel_manifest.read_text(encoding="utf-8"))
            if args.panel_manifest
            else None
        )
        res = score_day(
            args.date,
            bundle_dir=args.bundle_dir,
            out_root=args.out_root,
            ledger_path=args.ledger,
            sessions=_load_sessions(args.sessions),
            panel=pl.read_parquet(args.panel),
            panel_manifest=manifest,
        )
    except (ShadowError, bundle_mod.BundleError, ValueError, OSError) as exc:
        print(f"kr_next_shadow failed: {exc}", file=sys.stderr)
        return 1
    rec = res["record"]
    print(
        _json_dumps(
            {
                "dir": str(res["dir"]),
                "valid": rec["valid"],
                "attempt": rec["attempt"],
                "is_grid_date": rec["is_grid_date"],
                "counts": rec["counts"],
            },
            indent=2,
        )
    )
    return 0 if rec["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
