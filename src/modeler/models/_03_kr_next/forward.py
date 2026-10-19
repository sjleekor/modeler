"""Forward evaluation job of the next KR model: shadow ledger -> judged forward points.

Spec: C4 spec C-5, G-2..G-9, G-18, G-24 and the preregistration draft §9 (judgment cells) and §10
(forward sample and checks) of ``my/milestones/kr/modeling/plan/20261018_kr_next_prereg``.

The job reads the shadow ledger (``shadow.py``) and the lake, and turns every *grid point* of the
ledger into one row of ``points.parquet`` once the point has matured. It never trains, never reads
a label of a date that the frozen bundle had not scored before the date happened
(:func:`require_forward_only`), and it only appends.

Definitions fixed here (all from the draft; the open choices are marked *choice*):

* **Counting records** = ``shadow.counting_records`` (first valid attempt per ``D``).
  ``D0`` = the earliest counting ``D``; grid dates = ``evaluate.grid_dates(sessions, D0)``.
  A grid date up to the last lake date without a counting record is a **missing point** (G-4):
  one row with ``status = "missing"`` and a reason, no substitute date.
* **G (G-9)** of point ``k``: basket = ``basket.parquet`` target of ``D_k`` (the ledger record);
  holding window ``D_k`` close -> ``D_{k+1}`` close, where ``D_{k+1}`` = ``D0 + 20(k+1)`` sessions
  even when that record is missing; ``evaluate.holding_returns`` on the daily closes of the lake
  (halted/delisted frozen at the last non-halt close, untradable at entry excluded and the weights
  renormalized; "untradable at entry" is the c1 label kind ``untradable`` of the signal date, so an
  upper-limit lock at the ``D_k`` close counts, counted in ``n_untradable_entry``; a basket name
  without any label row keeps the ``holding_returns`` rule and is counted in ``n_basket_no_label``);
  benchmark = KOSPI 200 close to close (``krx_index_daily``, ``idx_nm`` = the fixed
  benchmark of ``feature_set``) on the same two dates; one-way turnover of ``D_k`` =
  ``one_way_turnover(target_k, drift(previous counted basket over D_prev -> D_k))`` (first build:
  1.0; when grid points in between are missing the basket was carried, so the drift runs over the
  longer span); cost 65bp x turnover (``evaluate.g_point``).
* **Tail (G-7)**: the tail flags recorded in ``predictions.parquet`` of ``D_k`` (never
  recomputed); returns = ``fwd_ret_c1_20d`` / ``label_exit_kind_c1_20d`` of the signal date
  ``t_k`` from ``labels_entry`` with exactly the parameters of ``dataset.py`` (c1, h20, haircut 0,
  K200 fixed benchmark); excluded rows are counted per group (``evaluate.tail_point``).
* **Rank IC**: ``evaluate.rank_ic(recorded score, raw_label_k200_c1_20d)`` over the recorded
  candidates whose label passes ``label_gate`` (and has a raw label).
* **Random basket (G-18)**: ``evaluate.random_basket`` on the recorded candidates over the same
  holding window; the percentile of the model basket's gross excess return is recorded.
* **Maturity** (:func:`maturity`, draft §9.1 / spec G-9 read literally): a point is *matured*
  when its PRICE inputs are in: ``D_{k+1}`` is a session of the calendar, the ``D_{k+1}`` stock
  closes and the KOSPI 200 closes of ``D_k`` and ``D_{k+1}`` are in the lake, and
  ``matured_at <= as_of`` (the last lake date, or the check date for a check).
  ``matured_at`` = the session ``INDEX_LAG_SESSIONS`` (1, §10.2 "코스피 200은 1세션 늦음") after
  ``D_{k+1}``. **Label decisions do not gate maturity**: a candidate that is halted for months
  cannot block a point.
* **Pending labels (G-7, rank IC)**: a candidate whose label for ``t_k`` is not decided
  (kind ``censored``, NULL kind, or no label row) is an excluded row of kind ``"pending"`` in the
  tail cell (counted per group and kind, like the other excluded kinds); the cell and the rank IC
  are computed from the decided rows. ``n_pending`` is recorded per point.
* **G at maturity (G-9)**: halted / undecided names are valued at the last non-halt close (what
  ``holding_returns`` does). That value, together with the tail cell, the rank IC and
  ``n_pending`` *as computed at the check* (lake truncated at the check date), is **frozen in the
  check JSON** (``g_at_check``, ``tail_diff_at_check``, ``rank_ic_at_check``,
  ``n_pending_at_check``) and is what the check means and the fixed-n tests use. A check JSON is
  never rewritten; a later check shows its own at-check values and whether they differ from the
  latest final values of ``points.parquet``.
* **Revisions**: ``points.parquet`` rows are keyed by ``(grid_date, revision)``. A new row is
  appended when a point first matures (revision 1, with its pending counts) or when a later
  computation differs from its latest row in a value column (``revision + 1``), e.g. when pending
  labels get decided or a close is revised. Rows are never rewritten
  (the file is rewritten atomically with the old rows verified byte-equal, there is no in-place
  edit). A missing point is appended once.
* **KOSPI 200 close missing (decisions #9.4)**: while the K200 close of ``D_k`` or ``D_{k+1}`` is
  absent the point stays unmatured; once the lake has stock data ``BENCH_MISSING_SESSIONS`` (20)
  sessions after ``D_{k+1}`` and a K200 close is still absent, the point becomes a missing point
  with ``reason = "bench_missing"`` (appended once, like every other missing point). A missing
  point is final: a later matured computation of the same grid date (late data) is not appended
  and is only reported as ``late_after_missing`` in the run summary.
* **Checks**: ``checks/<check_date>.json`` from ``evaluate.check_decision`` on the points that are
  matured **as of the check date** (the lake truncated at the check date, so a later value cannot
  leak in). Write-once.

The bundle manifest is write-once and written before the bundle tag exists, so the tag time comes
from a sidecar ``freeze_info.json`` (``--freeze-info``) that the main session writes at the moment
of the ``kr-next-bundle-frozen`` tag: ``{"bundle_tag": "kr-next-bundle-frozen",
"bundle_tag_time": "<ISO with offset>", "prereg_tag": "kr-next-frozen", "prereg_commit": "<sha>"}``.
It is overlaid on the manifest (:func:`read_bundle_manifest`).

Safety (:func:`require_forward_only`): the job refuses to compute anything unless the bundle
manifest names a preregistration tag and the freeze time of ``kr-next-bundle-frozen``, and every
valid ledger record was *decided and written* after that time. It can therefore only ever judge
forward points; it cannot be used to open a historical label.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
from scipy import stats

from modeler.etl.label_gate import DEFAULT_ALLOWED_KINDS, apply_label_gate
from modeler.etl.labels import LabelSpec
from modeler.etl.labels_entry import build_entry_label_sql
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import evaluate as ev
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import shadow as sh

SCHEMA_CHECK = "kr-next-forward-check.v1"
BUNDLE_TAG_NAME = "kr-next-bundle-frozen"
POINTS_FILE = "points.parquet"
CHECKS_DIR = "checks"
INDEX_LAG_SESSIONS = 1
#: sessions after D_{k+1} (stock data in the lake) after which a missing K200 close = missing point
BENCH_MISSING_SESSIONS = 20
N_DRAWS = 1000

#: the label parameters of ``dataset.build_kr_next_panel_sql`` (a test pins them to its defaults)
LABEL_PARAMS: dict[str, int] = {
    "min_end_gap_sessions": 1,
    "end_halt_sessions": 20,
    "terminal_tail_max": 10,
    "min_cross_section": 30,
}
END_HALT_HAIRCUT = 0.0
PENDING_KINDS = ("censored",)
PENDING = "pending"  # excluded-row kind of an undecided label
BENCH_MISSING = "bench_missing"  # reason of a point whose K200 close never arrived
#: kinds whose decision needs the end of the series: matured the session after the end date


_LABEL = ds.label_columns(ds.HORIZON)
FWD, KIND, END, RAW = _LABEL["fwd"], _LABEL["kind"], _LABEL["end_date"], _LABEL["raw"]

SCHEMA: dict[str, Any] = {
    "grid_k": pl.Int64,
    "grid_date": pl.Date,
    "signal_date": pl.Date,
    "next_grid_date": pl.Date,
    "status": pl.String,
    "reason": pl.String,
    "revision": pl.Int64,
    "matured_at": pl.Date,
    "g_ready_at": pl.Date,
    "g": pl.Float64,
    "gross": pl.Float64,
    "bench": pl.Float64,
    "cost": pl.Float64,
    "turnover": pl.Float64,
    "turnover_warning": pl.Boolean,
    "excluded_weight": pl.Float64,
    "n_excluded": pl.Int64,
    "n_untradable_entry": pl.Int64,
    "n_basket_no_label": pl.Int64,
    "n_basket": pl.Int64,
    "first_build": pl.Boolean,
    "carried_span": pl.Boolean,
    "prev_basket_date": pl.Date,
    "tail_diff": pl.Float64,
    "tail_mean": pl.Float64,
    "rest_mean": pl.Float64,
    "n_tail": pl.Int64,
    "n_rest": pl.Int64,
    "excluded_tail": pl.String,
    "excluded_rest": pl.String,
    "rank_ic": pl.Float64,
    "n_ic": pl.Int64,
    "n_pending": pl.Int64,
    "model_excess": pl.Float64,
    "random_percentile": pl.Float64,
    "random_mean": pl.Float64,
    "random_sd": pl.Float64,
    "random_n_draws": pl.Int64,
    "random_seed": pl.Int64,
    "n_candidates": pl.Int64,
    "exit_kind_counts": pl.String,
    "computed_data_date": pl.Date,
    "record_dir": pl.String,
    "record_sha256": pl.String,
    "code_inventory_sha256": pl.String,
}
#: columns whose change makes a new revision (the rest is bookkeeping)
VALUE_COLS: tuple[str, ...] = (
    "matured_at",
    "g_ready_at",
    "g",
    "gross",
    "bench",
    "cost",
    "turnover",
    "excluded_weight",
    "n_excluded",
    "n_untradable_entry",
    "n_basket_no_label",
    "n_basket",
    "first_build",
    "carried_span",
    "prev_basket_date",
    "tail_diff",
    "tail_mean",
    "rest_mean",
    "n_tail",
    "n_rest",
    "excluded_tail",
    "excluded_rest",
    "rank_ic",
    "n_ic",
    "n_pending",
    "model_excess",
    "random_percentile",
    "random_mean",
    "random_sd",
    "random_n_draws",
    "n_candidates",
    "exit_kind_counts",
)


class ForwardError(RuntimeError):
    """The forward job cannot run (ledger broken, inputs missing, file would be rewritten)."""


class ForwardOnlyError(ForwardError):
    """The guard refused: the job would touch a point that is not a forward point."""


# --------------------------------------------------------------------------- guard


def _parse_dt(value: Any, what: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ForwardOnlyError(f"{what} is not an ISO datetime: {value!r}") from exc
    if dt.tzinfo is None:
        raise ForwardOnlyError(f"{what} has no time zone: {value!r}")
    return dt


def bundle_tag_time(bundle_manifest: Mapping[str, Any]) -> datetime:
    """Freeze time of ``kr-next-bundle-frozen`` from the manifest (key ``bundle_tag_time``).

    Raises :class:`ForwardOnlyError` when the manifest has no prereg tag, no bundle tag time,
    or names another bundle tag.
    """
    tag = bundle_manifest.get("prereg_tag")
    if not isinstance(tag, str) or not tag.strip():
        raise ForwardOnlyError("bundle manifest has no prereg_tag: refusing to evaluate")
    name = bundle_manifest.get("bundle_tag", BUNDLE_TAG_NAME)
    if name != BUNDLE_TAG_NAME:
        raise ForwardOnlyError(f"bundle tag is {name!r}, expected {BUNDLE_TAG_NAME!r}")
    raw = bundle_manifest.get("bundle_tag_time")
    if raw in (None, ""):
        raise ForwardOnlyError(
            f"bundle manifest has no bundle_tag_time (commit time of {BUNDLE_TAG_NAME}): "
            "refusing to evaluate"
        )
    return _parse_dt(raw, "bundle_tag_time")


def require_forward_only(
    ledger: Sequence[Mapping[str, Any]],
    bundle_manifest: Mapping[str, Any],
    out_root: str | Path | None = None,
) -> datetime:
    """Refuse unless every valid ledger record is a forward record of the frozen bundle.

    ``ledger`` = parsed ledger lines (``shadow.read_ledger``). Rules, all raising
    :class:`ForwardOnlyError`:

    * the manifest has a prereg tag and ``bundle_tag_time`` (see :func:`bundle_tag_time`);
    * every valid ledger line has ``decision_at(D) >= bundle_tag_time`` and ``written_at >=
      bundle_tag_time`` (a record written before the tag, or about a date whose decision time was
      before it, is a historical record);
    * with ``out_root``: the record's ``bundle.prereg_tag`` and ``bundle.feature_list_sha256``
      equal the manifest's.

    Returns the tag time. Pure: reads nothing but ``record.json`` files when ``out_root`` is given.
    """
    tag_time = bundle_tag_time(bundle_manifest)
    for line in ledger:
        if not line.get("valid"):
            continue
        d = date.fromisoformat(line["D"])
        assert_forward_point(d, tag_time)
        written = _parse_dt(line.get("written_at"), f"written_at of D={line['D']}")
        if written < tag_time:
            raise ForwardOnlyError(
                f"ledger record D={line['D']} was written at {written.isoformat()}, before the "
                f"bundle tag time {tag_time.isoformat()}: not a forward record"
            )
        if out_root is not None:
            rec = json.loads((Path(out_root) / line["dir"] / sh.FILE_RECORD).read_bytes())
            b = rec.get("bundle") or {}
            if b.get("prereg_tag") != bundle_manifest["prereg_tag"]:
                raise ForwardOnlyError(
                    f"record D={line['D']} was scored by prereg tag {b.get('prereg_tag')!r}, "
                    f"manifest says {bundle_manifest['prereg_tag']!r}"
                )
            fl = bundle_manifest.get("feature_list_sha256")
            if fl is not None and b.get("feature_list_sha256") != fl:
                raise ForwardOnlyError(
                    f"record D={line['D']} used another feature list than the bundle manifest"
                )
    return tag_time


def assert_forward_point(d: date, tag_time: datetime) -> None:
    """Raise unless the decision of grid date ``d`` (10:00 KST) is not before ``tag_time``."""
    if sh.decision_at(d) < tag_time:
        raise ForwardOnlyError(
            f"grid date {d} is earlier than the bundle tag time {tag_time.isoformat()}: "
            "refusing to evaluate a point that is not a forward point"
        )


# --------------------------------------------------------------------------- calendar


def _shift(ordered: Sequence[date], d: date, n: int) -> date | None:
    """The session ``n`` sessions after ``d`` (``d`` must be a session), or None past the end."""
    if d not in ordered:
        return None
    i = list(ordered).index(d) + n
    return ordered[i] if i < len(ordered) else None


def maturity(
    sessions: Sequence[date], next_grid_date: date, index_lag: int = INDEX_LAG_SESSIONS
) -> date | None:
    """``matured_at`` of a point whose price inputs are in: ``index_lag`` sessions after D_{k+1}.

    None when ``next_grid_date`` is not a session or the calendar does not reach that session.
    Labels play no part (draft §9.1).
    """
    return _shift(sorted(set(sessions)), next_grid_date, index_lag)


def pending_kind(kind: str | None) -> str:
    """The kind a label counts as at a check: undecided (NULL / censored) -> ``"pending"``."""
    return PENDING if kind is None or kind in PENDING_KINDS else kind


# --------------------------------------------------------------------------- lake access


def _asof_views(
    con: duckdb.DuckDBPyConnection, views: ds.KrNextViews, as_of: date
) -> ds.KrNextViews:
    """Temp views of the three dated inputs truncated at ``as_of`` (a check sees no later data)."""
    lit = f"DATE '{as_of.isoformat()}'"
    spec = {
        "daily_ohlcv": ("_kfwd_ohlcv", "trade_date"),
        "quality_view": ("_kfwd_quality", "trade_date"),
        "krx_index_daily": ("_kfwd_index", "bas_dd"),
    }
    repl: dict[str, str] = {}
    for field, (name, col) in spec.items():
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW {name} AS "
            f"SELECT * FROM {getattr(views, field)} WHERE {col} <= {lit}"
        )
        repl[field] = name
    return dataclasses.replace(views, **repl)


def lake_last_date(con: duckdb.DuckDBPyConnection, views: ds.KrNextViews) -> date:
    row = con.execute(f"SELECT max(trade_date) FROM {views.daily_ohlcv}").fetchone()
    if row is None or row[0] is None:
        raise ForwardError("daily_ohlcv is empty")
    return row[0]


def _closes(
    con: duckdb.DuckDBPyConnection,
    views: ds.KrNextViews,
    start: date,
    end: date,
    tickers: Sequence[str],
) -> pl.DataFrame:
    """``trade_date, ticker, close, is_halted`` of the tickers in ``[start, end]`` (one vintage).

    Halted = open = high = low = 0, as in ``labels_entry``. ``close`` is the lake's ``close``
    column read once from this connection, so a corporate-action factor cancels in a ratio.
    """
    return con.execute(
        f"""
        SELECT trade_date, ticker, CAST(close AS DOUBLE) AS close,
               (open = 0 AND high = 0 AND low = 0) AS is_halted
        FROM {views.daily_ohlcv}
        WHERE trade_date BETWEEN ? AND ? AND list_contains(?, ticker)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_date, ticker ORDER BY market) = 1
        """,
        [start, end, list(tickers)],
    ).pl()


def _k200(
    con: duckdb.DuckDBPyConnection, views: ds.KrNextViews, dates: Sequence[date]
) -> dict[date, float]:
    name = fs.FIXED_BENCHMARK[1]
    rows = con.execute(
        f"SELECT bas_dd, CAST(close_idx AS DOUBLE) FROM {views.krx_index_daily} "
        "WHERE idx_nm = ? AND list_contains(?, bas_dd) AND close_idx IS NOT NULL",
        [name, list(dates)],
    ).fetchall()
    out: dict[date, float] = {}
    for d, c in rows:
        if d in out:
            raise ForwardError(f"{name} has two rows on {d}: the benchmark is ambiguous")
        out[d] = c
    return out


def compute_labels(
    con: duckdb.DuckDBPyConnection, views: ds.KrNextViews, signal_dates: Sequence[date]
) -> pl.DataFrame:
    """c1 / h20 labels of the signal dates, with the exact parameters of ``dataset.py``.

    Columns: ``trade_date, ticker`` + ``fwd_ret_c1_20d, label_exit_kind_c1_20d,
    label_end_date_c1_20d, raw_label_k200_c1_20d``. One row per (date, ticker).
    """
    if not signal_dates:
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ticker": pl.String,
                FWD: pl.Float64,
                KIND: pl.String,
                END: pl.Date,
                RAW: pl.Float64,
            }
        )
    spec = LabelSpec(horizons=(ds.HORIZON,), kind="excess", bench="eqw_market", outputs=("rank",))
    sql = build_entry_label_sql(
        spec,
        ds.TAG,
        views.daily_ohlcv,
        views.quality_view,
        master_view=views.stock_master,
        end_halt_haircut=END_HALT_HAIRCUT,
        fixed_benchmarks=(fs.FIXED_BENCHMARK,),
        fixed_index_view=views.krx_index_daily,
        **LABEL_PARAMS,
    )
    q = (
        f"SELECT trade_date, ticker, market, {FWD}, {KIND}, {END}, {RAW} FROM ({sql}) "
        "WHERE list_contains(?, trade_date)"
    )
    df = con.execute(q, [list(signal_dates)]).pl()
    return (
        df.sort(["trade_date", "ticker", "market"])
        .unique(subset=["trade_date", "ticker"], keep="first", maintain_order=True)
        .drop("market")
    )


# --------------------------------------------------------------------------- per-point maths


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _num(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _load_record(out_root: Path, line: Mapping[str, Any]) -> dict[str, Any]:
    root = out_root / line["dir"]
    rec = json.loads((root / sh.FILE_RECORD).read_bytes())
    preds = pl.read_parquet(root / sh.FILE_PREDICTIONS)
    basket = None
    if line.get("has_basket") and (root / sh.FILE_BASKET).is_file():
        basket = pl.read_parquet(root / sh.FILE_BASKET)
    return {"line": line, "record": rec, "predictions": preds, "basket": basket}


def _weights(basket: pl.DataFrame) -> dict[str, float]:
    return dict(zip(basket["ticker"].to_list(), basket["target_weight"].to_list(), strict=True))


def _compute(
    con: duckdb.DuckDBPyConnection,
    views: ds.KrNextViews,
    *,
    counting: Mapping[str, Mapping[str, Any]],
    out_root: Path,
    sessions: Sequence[date],
    tag_time: datetime,
    code_sha: str,
    as_of: date | None = None,
    n_draws: int = N_DRAWS,
) -> dict[str, Any]:
    """All grid points as of ``as_of`` (default: the last lake date).

    Returns ``{"rows": [...], "data_date": date}``; each row is a dict over :data:`SCHEMA` plus
    ``state`` in ``matured | pending | missing`` (``pending`` = prices not in yet) and, for a
    matured point, ``pending_tickers`` (candidates whose label is undecided).
    """
    ordered = sorted(set(sessions))
    d0 = date.fromisoformat(min(counting))
    full_last = lake_last_date(con, views)
    v = views
    if as_of is not None and as_of < full_last:
        v = _asof_views(con, views, as_of)
    try:
        data_date = lake_last_date(con, v)
        grid = ev.grid_dates(ordered, d0)

        # a counting record on a date has to agree with the calendar about the grid
        grid_set = set(grid)
        for ds_, line in counting.items():
            d = date.fromisoformat(ds_)
            if d < d0:
                continue
            rec = json.loads((out_root / line["dir"] / sh.FILE_RECORD).read_bytes())
            if bool(rec.get("is_grid_date")) != (d in grid_set):
                raise ForwardError(
                    f"record D={ds_}: is_grid_date={rec.get('is_grid_date')} disagrees with the "
                    "sessions calendar given to the forward job"
                )

        # ---- which points have their price inputs ------------------------------------------
        plan: list[dict[str, Any]] = []
        for k, dk in enumerate(grid):
            nxt = grid[k + 1] if k + 1 < len(grid) else None
            plan.append({"k": k, "D": dk, "next": nxt})
        k200_dates = sorted({p["D"] for p in plan} | {p["next"] for p in plan if p["next"]})
        k200 = _k200(con, v, [d for d in k200_dates if d <= data_date])
        for p in plan:
            counted = counting.get(p["D"].isoformat())
            p["counted"] = counted is not None
            p["matured_at"] = maturity(ordered, p["next"]) if p["next"] else None
            lim = _shift(ordered, p["next"], BENCH_MISSING_SESSIONS) if p["next"] else None
            p["bench_missing"] = bool(
                counted is not None
                and lim is not None
                and lim <= data_date
                and not (p["D"] in k200 and p["next"] in k200)
            )
            p["ready"] = bool(
                counted is not None
                and p["next"] is not None
                and p["matured_at"] is not None
                and p["matured_at"] <= data_date
                and p["next"] <= data_date
                and p["D"] in k200
                and p["next"] in k200
            )
        # ---- labels of the ready points, one query ------------------------------------------
        loaded: dict[date, dict[str, Any]] = {}
        sig_dates: list[date] = []
        for p in plan:
            if p["D"] > data_date or not p["counted"]:
                continue
            loaded[p["D"]] = _load_record(out_root, counting[p["D"].isoformat()])
            if p["ready"]:
                sig_dates.append(date.fromisoformat(loaded[p["D"]]["record"]["t"]))
        labels_all = compute_labels(con, v, sorted(set(sig_dates)))

        rows: list[dict[str, Any]] = []
        prev: tuple[date, int, dict[str, float]] | None = None  # (D, grid index, target)
        turnovers: list[float] = []
        for p in plan:
            k, dk, nxt = p["k"], p["D"], p["next"]
            if dk > data_date:
                break  # the entry close of D_k is not in the lake: a future grid date
            assert_forward_point(dk, tag_time)
            if not p["counted"]:
                rows.append(_missing_row(k, dk, nxt, data_date, code_sha))
                continue
            if p["bench_missing"]:
                rows.append(_missing_row(k, dk, nxt, data_date, code_sha, BENCH_MISSING))
                continue
            ld = loaded[dk]
            rec, preds, basket = ld["record"], ld["predictions"], ld["basket"]
            if basket is None:
                raise ForwardError(f"grid record D={dk} has no basket.parquet")
            target = _weights(basket)
            first_build = prev is None
            carried = prev is not None and (k - prev[1]) > 1
            if prev is None:
                turnover = 1.0
            else:
                cl = _closes(con, v, prev[0], dk, list(prev[2]))
                hr = ev.holding_returns(cl, prev[0], dk, list(prev[2]))
                turnover = ev.one_way_turnover(target, ev.drift(prev[2], hr))
            turnovers.append(turnover)
            row = _base_row(k, dk, nxt, rec, ld["line"], data_date, code_sha)
            row.update(
                turnover=turnover,
                turnover_warning=ev.turnover_warning(turnovers),
                first_build=first_build,
                carried_span=carried,
                prev_basket_date=prev[0] if prev else None,
                n_basket=len(target),
                n_candidates=preds.height,
            )
            prev = (dk, k, target)
            if not p["ready"]:
                row["state"] = "pending"
                rows.append(row)
                continue
            assert nxt is not None
            cands = preds["ticker"].to_list()
            cl = _closes(con, v, dk, nxt, cands)
            hold = ev.holding_returns(cl, dk, nxt, cands)
            bench = k200[nxt] / k200[dk] - 1.0
            t_k = date.fromisoformat(rec["t"])
            lab = labels_all.filter(pl.col("trade_date") == t_k).drop("trade_date")
            # the label rule decides entry: a basket name whose c1 label kind for t_k is
            # ``untradable`` (halt / no row / upper-limit lock at the D_k close) is excluded
            # from G like the label does; ``holding_returns`` alone misses the upper-limit lock
            kind_of = dict(zip(lab["ticker"].to_list(), lab[KIND].to_list(), strict=True))
            untr = sorted(t for t in target if kind_of.get(t) == "untradable")
            no_label = [t for t in target if t not in kind_of]
            hold_g = {**hold, **dict.fromkeys(untr)}
            gp = ev.g_point(dict(target), hold_g, bench, turnover)
            row.update(
                g=_num(gp["g"]),
                gross=_num(gp["gross"]),
                bench=bench,
                cost=gp["cost"],
                excluded_weight=gp["excluded_weight"],
                n_excluded=int(gp["n_excluded"]),
                n_untradable_entry=len(untr),
                n_basket_no_label=len(no_label),
            )
            row["g_ready_at"] = p["matured_at"]
            row["matured_at"] = p["matured_at"]
            row["state"] = "matured"
            # random baskets draw from the same candidates under the same entry rule as the model
            # basket: every untradable candidate (not only basket names) has no holding return
            untr_all = [t for t, k in kind_of.items() if k == "untradable"]
            hold_rb = {**hold, **dict.fromkeys(untr_all)}
            row["pending_tickers"] = _label_cells(row, preds, lab, hold_rb, gp, bench, dk, n_draws)
            rows.append(row)
        return {"rows": rows, "data_date": data_date}
    finally:
        if v is not views:
            for name in ("_kfwd_ohlcv", "_kfwd_quality", "_kfwd_index"):
                con.execute(f"DROP VIEW IF EXISTS {name}")


def _base_row(
    k: int,
    dk: date,
    nxt: date | None,
    rec: Mapping[str, Any],
    line: Mapping[str, Any],
    data_date: date,
    code_sha: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in SCHEMA}
    row.update(
        grid_k=k,
        grid_date=dk,
        signal_date=date.fromisoformat(rec["t"]),
        next_grid_date=nxt,
        status="matured",
        revision=1,
        computed_data_date=data_date,
        record_dir=line["dir"],
        record_sha256=line["record_sha256"],
        code_inventory_sha256=code_sha,
    )
    return row


def _missing_row(
    k: int,
    dk: date,
    nxt: date | None,
    data_date: date,
    code_sha: str,
    reason: str = "no_valid_record",
) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in SCHEMA}
    row.update(
        grid_k=k,
        grid_date=dk,
        next_grid_date=nxt,
        status="missing",
        reason=reason,
        revision=1,
        computed_data_date=data_date,
        code_inventory_sha256=code_sha,
        state="missing",
    )
    return row


def _label_cells(
    row: dict[str, Any],
    preds: pl.DataFrame,
    lab: pl.DataFrame,
    hold: Mapping[str, float | None],
    gp: Mapping[str, Any],
    bench: float,
    dk: date,
    n_draws: int,
) -> list[str]:
    """Tail cell, rank IC and random basket percentile of one point (fills ``row``).

    Undecided labels are excluded rows of kind ``"pending"``. Returns the pending tickers.
    """
    df = (
        preds.select("ticker", "score", "tail", "mcap_krx")
        .join(lab.select("ticker", FWD, KIND, RAW), on="ticker", how="left")
        .with_columns(
            pl.col(KIND)
            .map_elements(pending_kind, return_dtype=pl.String, skip_nulls=False)
            .alias(KIND)
        )
    )
    tail = ev.tail_point(
        df.select(
            "tail", pl.col(FWD).alias("fwd_ret_c1_20d"), pl.col(KIND).alias("label_exit_kind")
        ),
        DEFAULT_ALLOWED_KINDS,
    )
    row.update(
        tail_diff=_num(tail["diff"]),
        tail_mean=_num(tail["tail_mean"]),
        rest_mean=_num(tail["rest_mean"]),
        n_tail=int(tail["n_tail"]),  # type: ignore[arg-type]
        n_rest=int(tail["n_rest"]),  # type: ignore[arg-type]
        excluded_tail=_json(tail["excluded_tail"]),
        excluded_rest=_json(tail["excluded_rest"]),
    )
    gated = apply_label_gate(df, ds.TAG, ds.HORIZON, also_require=[RAW])
    ic = ev.rank_ic(gated["score"].to_list(), gated[RAW].to_list())
    pending = sorted(df.filter(pl.col(KIND) == PENDING)["ticker"].to_list())
    counts: dict[str, int] = {}
    for kk in df[KIND].to_list():
        counts[kk] = counts.get(kk, 0) + 1
    row.update(
        rank_ic=_num(ic),
        n_ic=gated.height,
        n_pending=len(pending),
        exit_kind_counts=_json(counts),
    )
    if gp["gross"] is not None:
        excess = float(gp["gross"]) - bench
        rb = ev.random_basket(
            preds.select("ticker", pl.col("mcap_krx").alias("mcap")),
            dk,
            hold,
            bench,
            model_excess=excess,
            n_draws=n_draws,
        )
        row.update(
            model_excess=excess,
            random_percentile=_num(rb["percentile"]),
            random_mean=_num(rb["mean"]),
            random_sd=_num(rb["sd"]),
            random_n_draws=int(rb["n_draws"]),  # type: ignore[arg-type]
            random_seed=int(rb["seed"]),  # type: ignore[arg-type]
        )
    return pending


def _missing_reasons(lines: Sequence[Mapping[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Say why a grid date is missing: no line at all, or only invalid (late) attempts."""
    by_d: dict[str, list[Mapping[str, Any]]] = {}
    for line in lines:
        by_d.setdefault(line["D"], []).append(line)
    for r in rows:
        if r["state"] == "missing" and r["reason"] != BENCH_MISSING:
            att = by_d.get(r["grid_date"].isoformat(), [])
            r["reason"] = "only_invalid_attempts" if att else "no_record"


# --------------------------------------------------------------------------- points.parquet


def read_points(path: Path) -> pl.DataFrame:
    if not path.is_file():
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(path)


def _frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    data = [{c: r.get(c) for c in SCHEMA} for r in rows]
    return pl.DataFrame(data, schema=SCHEMA)


def plan_appends(
    existing: pl.DataFrame,
    computed: Sequence[Mapping[str, Any]],
    late_after_missing: list[str] | None = None,
) -> list[dict[str, Any]]:
    """New rows to append: first maturity, a changed final value, or a first missing point.

    A grid date that already has a missing row is final (decisions: no revival by late data, so
    inclusion cannot depend on the outcome): a later matured computation for it is not appended;
    its grid date is added to ``late_after_missing`` (run log only, never ``points.parquet``).
    """
    latest: dict[Any, dict[str, Any]] = {}
    for r in existing.sort(["grid_date", "revision"]).to_dicts():
        latest[(r["grid_date"], r["status"] == "missing")] = r
    out: list[dict[str, Any]] = []
    for r in computed:
        if r["state"] == "missing":
            if (r["grid_date"], True) not in latest:
                out.append({**r, "revision": 1})
            continue
        if r["state"] != "matured":
            continue
        if (r["grid_date"], True) in latest:
            if late_after_missing is not None:
                late_after_missing.append(r["grid_date"].isoformat())
            continue
        old = latest.get((r["grid_date"], False))
        if old is None:
            out.append({**r, "revision": 1})
        elif not all(_same(old[c], r.get(c)) for c in VALUE_COLS):
            out.append({**r, "revision": int(old["revision"]) + 1})
    return out


def append_points(path: Path, new_rows: Sequence[Mapping[str, Any]]) -> int:
    """Append rows to ``points.parquet``: old rows are kept and verified equal, never edited."""
    if not new_rows:
        return 0
    old = read_points(path)
    new = _frame(new_rows)
    merged = pl.concat([old.select(list(SCHEMA)), new], how="vertical")
    if old.height and not merged.head(old.height).equals(old.select(list(SCHEMA))):
        raise ForwardError("refusing to write: the existing points rows would change")
    buf = io.BytesIO()
    merged.write_parquet(buf)
    sh._atomic_write_bytes(path, buf.getvalue())
    return new.height


# --------------------------------------------------------------------------- checks


def check_frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """``grid_date, matured_at, g, tail_diff, rank_ic`` for ``evaluate.check_decision``.

    Matured points carry their values as of the check (frozen there). Missing and not-yet-matured
    points are absent (G-4).
    """
    data = [
        {
            "grid_date": r["grid_date"],
            "matured_at": r["matured_at"],
            "g": r["g"],
            "tail_diff": r["tail_diff"],
            "rank_ic": r["rank_ic"],
        }
        for r in rows
        if r["state"] == "matured"
    ]
    return pl.DataFrame(
        data,
        schema={
            "grid_date": pl.Date,
            "matured_at": pl.Date,
            "g": pl.Float64,
            "tail_diff": pl.Float64,
            "rank_ic": pl.Float64,
        },
    )


#: base effect scenario of the Q7 power column (assumptions, not external estimates; draft ST-15)
POWER_SCENARIO: dict[str, dict[str, float]] = {
    "g": {"effect": 0.005, "sd": 0.04},  # +0.5% / month, sd 4%
    "tail_diff": {"effect": -0.01, "sd": 0.03},  # -1% / month, sd 3%
}
STAT_ALPHA = 0.025


def _clean_values(values: Sequence[Any]) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def one_sided_power(n: int, effect: float, sd: float, alpha: float = STAT_ALPHA) -> float | None:
    """Power of the one-sided one-sample t-test at ``n`` points (noncentral t).

    Direction follows the sign of ``effect`` (H1 mean > 0 for positive, < 0 for negative).
    ``None`` for ``n < 2`` or ``sd <= 0``.
    """
    if n < 2 or sd <= 0:
        return None
    crit = float(stats.t.ppf(1 - alpha, n - 1))
    ncp = abs(effect) * math.sqrt(n) / sd
    return float(stats.nct.sf(crit, n - 1, ncp))


def cell_stats(values: Sequence[Any], cell: str) -> dict[str, Any]:
    """Record-only numbers of one cell over the matured points (decisions #6, Q7).

    ``cell`` is ``"g"`` (one-sided 97.5% t LOWER bound), ``"tail_diff"`` (one-sided 97.5% t UPPER
    bound) or ``"rank_ic"`` (two-sided 95% t interval). Always the cumulative ``mean`` and ``n``
    (non-null values); bounds are ``None`` for ``n < 2``. ``power`` is the power at that ``n`` of
    the one-sided t-test under the base scenario (:data:`POWER_SCENARIO`); the rank IC has none.
    This never feeds ``evaluate.check_decision``.
    """
    v = _clean_values(values)
    n = len(v)
    out: dict[str, Any] = {"n": n, "mean": (sum(v) / n) if n else None}
    if cell in POWER_SCENARIO:
        sc = POWER_SCENARIO[cell]
        out["power_base"] = one_sided_power(n, sc["effect"], sc["sd"])
        out["power_scenario"] = {**sc, "alpha": STAT_ALPHA, "test": "one-sided t"}
    se = None
    if n >= 2:
        se = float(np.std(v, ddof=1)) / math.sqrt(n)
    if cell == "g":
        crit = float(stats.t.ppf(1 - STAT_ALPHA, n - 1)) if n >= 2 else None
        out["lower_975_one_sided"] = out["mean"] - crit * se if crit is not None else None
    elif cell == "tail_diff":
        crit = float(stats.t.ppf(1 - STAT_ALPHA, n - 1)) if n >= 2 else None
        out["upper_975_one_sided"] = out["mean"] + crit * se if crit is not None else None
    elif cell == "rank_ic":
        crit = float(stats.t.ppf(0.975, n - 1)) if n >= 2 else None
        out["ci95_two_sided"] = (
            [out["mean"] - crit * se, out["mean"] + crit * se] if crit is not None else None
        )
    else:
        raise ValueError(f"unknown cell {cell!r}")
    return out


NOTE = (
    "이 판정은 검정력 50% 미만의 탐색 판정입니다. 모델 중단은 제안이며 사용자 확인 뒤에만 합니다."
)


def build_check(
    check_date: date,
    rows: Sequence[Mapping[str, Any]],
    final_points: pl.DataFrame,
    *,
    tag_time: datetime,
    prereg_tag: str,
    ledger_lines: int,
    data_date: date,
    code_sha: str,
) -> dict[str, Any]:
    """The check report of ``check_date`` from the points as computed as of that date.

    A grid date that ``points.parquet`` already declares missing (any reason) is final: it is
    never counted, even if the as-of computation would mature it (late data); it is listed under
    ``missing`` with the reason of the points row.
    """
    declared = {
        r["grid_date"]: r["reason"]
        for r in final_points.filter(pl.col("status") == "missing")
        .sort(["grid_date", "revision"])
        .to_dicts()
    }
    rows = [
        {**r, "state": "missing", "reason": declared[r["grid_date"]]}
        if r["state"] == "matured" and r["grid_date"] in declared
        else r
        for r in rows
    ]
    frame = check_frame(rows)
    decision = ev.check_decision(frame, check_date)
    final_latest: dict[Any, dict[str, Any]] = {}
    for r in (
        final_points.filter(pl.col("status") == "matured")
        .sort(["grid_date", "revision"])
        .to_dicts()
    ):
        final_latest[r["grid_date"]] = r
    pts, miss = [], []
    for r in rows:
        if r["state"] == "matured":
            fin = final_latest.get(r["grid_date"])
            at = {
                "g": r["g"],
                "tail_diff": r["tail_diff"],
                "rank_ic": r["rank_ic"],
                "n_pending": r["n_pending"],
            }
            differs = {k: bool(fin) and not _same(fin[k], v) for k, v in at.items()}
            pts.append(
                {
                    "grid_date": r["grid_date"],
                    "matured_at": r["matured_at"],
                    "g_at_check": at["g"],
                    "tail_diff_at_check": at["tail_diff"],
                    "rank_ic_at_check": at["rank_ic"],
                    "n_pending_at_check": at["n_pending"],
                    "pending_tickers_at_check": r.get("pending_tickers") or [],
                    "excluded_tail_at_check": r["excluded_tail"],
                    "excluded_rest_at_check": r["excluded_rest"],
                    "final": {k: fin[k] for k in at} if fin else None,
                    "final_revision": fin["revision"] if fin else None,
                    "differs_from_final": differs,
                    "g_final": fin["g"] if fin else None,
                    "g_final_differs": differs["g"],
                    "turnover": r["turnover"],
                }
            )
        elif r["state"] == "missing":
            miss.append({"grid_date": r["grid_date"], "reason": r["reason"]})
    turn = [r["turnover"] for r in rows if r.get("turnover") is not None]
    return {
        "schema": SCHEMA_CHECK,
        "check_date": check_date,
        "data_date": data_date,
        "decision": decision,
        "points": pts,
        "missing": miss,
        "turnover_warning": ev.turnover_warning(turn),
        "requires_user_confirmation": bool(decision["stop_proposal"]),
        "cell_stats": {
            "g": cell_stats(frame["g"].to_list(), "g"),
            "tail_diff": cell_stats(frame["tail_diff"].to_list(), "tail_diff"),
            "rank_ic": cell_stats(frame["rank_ic"].to_list(), "rank_ic"),
            "record_only": True,
        },
        "note": NOTE,
        "inputs": {
            "bundle_tag_time": tag_time,
            "prereg_tag": prereg_tag,
            "ledger_lines": ledger_lines,
            "code_inventory_sha256": code_sha,
        },
    }


# --------------------------------------------------------------------------- the job


def read_bundle_manifest(path: str | Path, freeze_info: str | Path | None = None) -> dict[str, Any]:
    """Bundle ``manifest.json`` (directory or file) overlaid with an optional freeze-info JSON.

    The write-once bundle manifest cannot hold the time of a tag made after it; the freeze info
    (``{"bundle_tag": "kr-next-bundle-frozen", "bundle_tag_time": "<ISO>"}``, ISO with offset)
    carries it.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    manifest = json.loads(p.read_text(encoding="utf-8"))
    if freeze_info is not None:
        manifest.update(json.loads(Path(freeze_info).read_text(encoding="utf-8")))
    return manifest


def run_forward(
    *,
    con: duckdb.DuckDBPyConnection,
    out_root: str | Path,
    ledger_path: str | Path,
    bundle_manifest: Mapping[str, Any],
    sessions: Sequence[date],
    views: ds.KrNextViews | None = None,
    forward_dir: str | Path | None = None,
    check_dates: Sequence[date] = (),
    allowed_check_dates: Sequence[date] | None = None,
    n_draws: int = N_DRAWS,
) -> dict[str, Any]:
    """Evaluate the matured forward points and write the check reports of ``check_dates``.

    Order: verify the ledger (breaks -> :class:`ForwardError`), the forward-only guard, then the
    computation; nothing is read from the lake before both pass. ``forward_dir`` defaults to
    ``<out_root>/forward``. ``allowed_check_dates`` defaults to
    ``evaluate.check_dates(sessions=...)``.
    """
    out_root, ledger_path = Path(out_root), Path(ledger_path)
    views = views or ds.KrNextViews()
    fdir = Path(forward_dir) if forward_dir else out_root / "forward"
    lines = sh.read_ledger(ledger_path)
    chain = sh.verify_ledger(ledger_path, out_root)
    if chain["breaks"]:
        raise ForwardError(f"ledger chain is broken, refusing to evaluate: {chain['breaks'][:3]}")
    tag_time = require_forward_only(lines, bundle_manifest, out_root)
    counting = sh.counting_records(lines)
    summary: dict[str, Any] = {
        "n_counting": len(counting),
        "orphans": chain["orphans"],
        "new_rows": 0,
        "revisions": [],
        "checks_written": [],
        "checks_existing": [],
    }
    if not counting:
        return summary
    cfiles = sh.code_files()
    code_sha = sh.code_inventory_sha256(cfiles)
    with sh._lock(fdir / ".forward.lock"):
        points_path = fdir / POINTS_FILE
        res = _compute(
            con,
            views,
            counting=counting,
            out_root=out_root,
            sessions=sessions,
            tag_time=tag_time,
            code_sha=code_sha,
            n_draws=n_draws,
        )
        _missing_reasons(lines, res["rows"])
        existing = read_points(points_path)
        late: list[str] = []
        new_rows = plan_appends(existing, res["rows"], late)
        summary["late_after_missing"] = late
        summary["new_rows"] = append_points(points_path, new_rows)
        summary["revisions"] = [
            (r["grid_date"].isoformat(), r["status"], r["revision"]) for r in new_rows
        ]
        summary["data_date"] = res["data_date"]
        allowed = list(
            allowed_check_dates
            if allowed_check_dates is not None
            else ev.check_dates(sessions=sessions)
        )
        for cd in check_dates:
            if cd not in allowed:
                raise ForwardError(f"{cd} is not a check date (allowed: {allowed[:4]}...)")
            if cd > res["data_date"]:
                raise ForwardError(
                    f"check date {cd} is after the last lake date {res['data_date']}"
                )
            target = fdir / CHECKS_DIR / f"{cd.isoformat()}.json"
            if target.exists():
                summary["checks_existing"].append(str(target))
                continue
            asof = _compute(
                con,
                views,
                counting=counting,
                out_root=out_root,
                sessions=sessions,
                tag_time=tag_time,
                code_sha=code_sha,
                as_of=cd,
                n_draws=n_draws,
            )
            _missing_reasons(lines, asof["rows"])
            report = build_check(
                cd,
                asof["rows"],
                read_points(points_path),
                tag_time=tag_time,
                prereg_tag=str(bundle_manifest["prereg_tag"]),
                ledger_lines=len(lines),
                data_date=asof["data_date"],
                code_sha=code_sha,
            )
            sh._atomic_write_bytes(
                target, (sh._json_dumps(report, indent=2) + "\n").encode("utf-8")
            )
            summary["checks_written"].append(str(target))
    return summary


# --------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kr_next_forward", description=__doc__.split("\n")[0])
    p.add_argument("--out-root", required=True, type=Path, help="shadow out_root")
    p.add_argument("--ledger", type=Path, default=None, help="default: <out-root>/ledger.jsonl")
    p.add_argument("--lake", required=True, type=Path, help="DuckDB file with the lake views")
    p.add_argument("--bundle", required=True, type=Path, help="bundle dir or manifest.json")
    p.add_argument("--freeze-info", type=Path, default=None, help="JSON with bundle_tag_time")
    p.add_argument("--sessions", type=Path, default=None, help="JSON list/calendar; else the lake")
    p.add_argument("--forward-dir", type=Path, default=None)
    p.add_argument("--check-date", type=date.fromisoformat, action="append", default=[])
    args = p.parse_args(argv)
    try:
        manifest = read_bundle_manifest(args.bundle, args.freeze_info)
        con = duckdb.connect(str(args.lake), read_only=True)
        views = ds.KrNextViews()
        if args.sessions:
            sessions = sh._load_sessions(args.sessions)
        else:
            sessions = [
                r[0]
                for r in con.execute(
                    f"SELECT DISTINCT trade_date FROM {views.daily_ohlcv} "
                    "WHERE market = 'KOSPI' ORDER BY 1"
                ).fetchall()
            ]
        res = run_forward(
            con=con,
            out_root=args.out_root,
            ledger_path=args.ledger or args.out_root / "ledger.jsonl",
            bundle_manifest=manifest,
            sessions=sessions,
            views=views,
            forward_dir=args.forward_dir,
            check_dates=args.check_date,
        )
    except (ForwardError, sh.ShadowError, ValueError, OSError, duckdb.Error) as exc:
        print(f"kr_next_forward failed: {exc}", file=sys.stderr)
        return 1
    print(sh._json_dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
