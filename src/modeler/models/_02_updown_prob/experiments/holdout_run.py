"""holdout_run — model 02's final fit and its one holdout evaluation.

Plan: ``my/milestones/kr/modeling/plan/20260916_holdout_gate_plan.md`` §3 (the
frozen A/B/C rule), ``20260922_remaining_work.md`` §1-3 (this runner),
``assessment/20260922_direction_review.md`` §2.2-2.3 (what it has to check).

Model 01's counterpart is ``modeler.us.m7_run``; this is the same shape for the
Korean model, and the three properties it is built to have are the same.

**It cannot search a grid.** The adopted parameters are read from the run's
``summary.json`` and handed to one ``_fit_fold`` call. This module does not
import ``HGB_CLF_GRID`` or ``walk_forward``, so "re-tune on the holdout" is not
a mistake that can be made here — it is a thing the file cannot express.

**It separates three dates that the plan kept conflating.** The snapshot date
(which vintage of the source is pinned), the last evaluated formation date
(whose prediction is the last one scored), and the per-horizon label end date
(when that prediction's return closes). A mid-October snapshot does not by
itself make the last formation 2026-09-16; ``--eval-last-formation`` does, and
it is recorded.

**It purges the training set by label end date, not by formation date.**
``build_dataset`` stops the panel at formation 2025-07-31, which is not the same
wall: with ``h=20`` the labels of the last 20 formation sessions close inside
the holdout window (measured: formation 2025-07-04 closes 2025-08-01, formation
2025-07-31 closes 2025-08-29). ``walk_forward_splits`` already trims the final
train end by ``purge`` sessions, and this runner uses that; on top of it there
is a row-level filter, because ``d_idx`` skips halt days and a suspended name's
"+20 sessions" can land years later (measured on the 2026-08-23 snapshot: 539
rows over 60 tickers at h20, worst case 1,090 market sessions late).

Three modes, and only one of them opens anything::

    # 1. dates, boundaries and maturity. Fits nothing, reads no holdout label.
    uv run python -m modeler.models._02_updown_prob.experiments.holdout_run \\
        --preflight --eval-last-formation 2026-09-16

    # 2. rehearsal: rebuild a recorded fold through THIS runner's fit path and
    #    compare to the predictions the adopted run wrote. Touches only the
    #    frozen validation dataset (formation <= 2025-07-31).
    uv run python -m modeler.models._02_updown_prob.experiments.holdout_run \\
        --rehearse-fold 5

    # 3. the opening. Refuses without both flags.
    uv run python -m modeler.models._02_updown_prob.experiments.holdout_run \\
        --open --eval-last-formation 2026-09-16 --i-am-opening-the-holdout

``--open`` is the only mode that reads a holdout label, and the only one that
needs a dataset built past the D-2 wall.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import polars as pl

from modeler.etl import labels as labels_mod
from modeler.etl import preprocess as pp
from modeler.etl.config import LakeConfig
from modeler.etl.lake import register_views
from modeler.etl.manifest import current_git_sha
from modeler.etl.metrics import (
    _quantile_stats,
    rebalance_grid,
    topk_economic_report,
    topk_rebalance_series,
)
from modeler.etl.splits import Fold, walk_forward_splits
from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob import train as tr
from modeler.models._02_updown_prob.experiments import preopen_measures as pm
from modeler.models._02_updown_prob.experiments.run_matrix import (
    RESULTS_ROOT,
    code_hash,
    model_code_hash,
)
from modeler.models._02_updown_prob.spec import (
    HOLDOUT_START,
    PERIOD_END,
    PERIOD_START,
    ModelSpec,
    lake_config,
)

# The one candidate that survived selection (`20260916_holdout_gate_plan.md`
# §3.1). h5 was dropped on its cost structure and h60 is recorded, not judged —
# neither is opened here.
ADOPTED_STAGE = "E2"
ADOPTED_RUN_ID = "E2_h20_FS1h_seed0"

HOLDOUT_DIR = RESULTS_ROOT / "holdout"

#: Where the holdout panel goes. A suffix, not the adopted dataset's directory:
#: that one is what the frozen numbers were computed on and it stays as it is.
HOLDOUT_DATASET_SUFFIX = "__holdout"
#: And where a dress rehearsal's panel goes. A different suffix on purpose:
#: a rehearsal's artefacts must never be mistakable for the opening's.
REHEARSAL_DATASET_SUFFIX = "__rehearsal"

#: The gate reads five numbers off two benchmarks (`§3.1`). ``raw_label_{h}d``
#: is the universe-equal-weight excess; ``raw_label_idx_{h}d`` is the index one,
#: which only exists when the panel was built with ``LabelSpec.index_bench``.
EQW_LABEL = "raw_label_{h}d"
IDX_LABEL = "raw_label_idx_{h}d"


# --- 0. the adopted run ---------------------------------------------------------


@dataclass(frozen=True)
class AdoptedRun:
    """What the adopted run recorded, read back rather than retyped.

    Every number the runner needs to reproduce the model is in ``summary.json``
    and ``run_spec.json``. Nothing here is a literal in this file, because a
    literal is a place for the two to drift apart.
    """

    stage: str
    run_id: str
    horizon: int
    k: int
    cost_bps_roundtrip: float
    tau: float
    seed: int
    model: str
    target: str
    calibrate: str
    monotonic: bool
    best_params: dict
    primary_pred_col: str
    feature_set: str
    flow_variant: str
    preprocess_profile: str
    dataset_dir: Path
    predictions_path: Path
    summary: dict
    run_spec: dict

    @property
    def eqw_label(self) -> str:
        return EQW_LABEL.format(h=self.horizon)

    @property
    def idx_label(self) -> str:
        return IDX_LABEL.format(h=self.horizon)

    def train_config(self) -> tr.TrainConfig:
        """The adopted ``TrainConfig`` with a one-point grid.

        The grid is the winner alone. ``TrainConfig`` requires a non-empty grid
        and this runner never sweeps one, so the winner *is* the grid — and a
        reader of ``run_spec.json`` can see that the holdout searched nothing.
        """
        return tr.TrainConfig(
            model=self.model,
            target=self.target,
            horizon=self.horizon,
            grid=(dict(self.best_params),),
            seed=self.seed,
            calibrate=self.calibrate,
            monotonic=self.monotonic,
        )

    def spec(self) -> ModelSpec:
        """The adopted ``ModelSpec``, still inside the D-2 wall."""
        return ModelSpec(
            feature_set=self.feature_set,
            flow_variant=self.flow_variant,
            preprocess_profile=self.preprocess_profile,
            seed=self.seed,
            topk=self.k,
            cost_bps_roundtrip=self.cost_bps_roundtrip,
            tau=self.tau,
        ).for_horizon(self.horizon)


def load_adopted(stage: str = ADOPTED_STAGE, run_id: str = ADOPTED_RUN_ID) -> AdoptedRun:
    run_dir = RESULTS_ROOT / stage / run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    run_spec = json.loads((run_dir / "run_spec.json").read_text())
    spec_payload = run_spec.get("spec", {})
    return AdoptedRun(
        stage=stage,
        run_id=run_id,
        horizon=int(summary["horizon"]),
        k=int(summary["k"]),
        cost_bps_roundtrip=float(summary["cost_bps_roundtrip"]),
        tau=float(summary["tau"]),
        seed=int(summary["seed"]),
        model=str(summary["model"]),
        target=str(summary["target"]).rsplit("_", 1)[0],
        calibrate=str(summary["calibrate"]),
        monotonic=bool(summary["monotonic"]),
        best_params=dict(summary["best_params"]),
        primary_pred_col=str(summary["primary_pred_col"]),
        feature_set=str(spec_payload.get("feature_set", "FS1h")),
        flow_variant=str(spec_payload.get("flow_variant", "lag1")),
        preprocess_profile=str(spec_payload.get("preprocess_profile", "rank")),
        dataset_dir=Path(summary["dataset_dir"]),
        predictions_path=Path(summary["predictions_path"]),
        summary=summary,
        run_spec=run_spec,
    )


# --- 1. the three dates ---------------------------------------------------------


@dataclass
class Boundary:
    """Resolved dates, and what crosses them. Written as ``boundary.json``.

    The point of this frame is that every date below is *measured* on the
    snapshot the run will use, not assumed from a calendar. ``13~14 회`` was a
    calculation; ``n_rebalances_matured`` is a count.
    """

    snapshot_date: str
    horizon: int
    holdout_start: str
    # training
    train_first_formation: str
    train_last_formation: str
    train_last_label_end: str
    train_last_label_end_halted: str
    purge_sessions: int
    n_train_rows_dropped_by_label_end: int = 0
    n_train_tickers_dropped_by_label_end: int = 0
    # evaluation
    eval_first_formation: str = ""
    eval_last_formation: str = ""
    eval_last_label_end: str = ""
    n_eval_sessions: int = 0
    n_rebalances_held: int = 0
    n_rebalances_matured: int = 0
    rebalance_dates: list[str] = field(default_factory=list)
    unmatured_rebalance_dates: list[str] = field(default_factory=list)
    closed_names_by_rebalance: dict[str, int] = field(default_factory=dict)
    # source
    price_data_last_date: str = ""
    notes: list[str] = field(default_factory=list)


def session_calendar(config: LakeConfig) -> list:
    """Market sessions, as the label CTE counts them (halt days excluded).

    The union over tickers, which is what ``rebalance_grid`` walks and what a
    formation date means. It is *not* the per-ticker ``d_idx`` — that one is
    below, and the gap between the two is the halted-name problem.
    """
    con = duckdb.connect()
    register_views(con, config, tables=[bd.PRICE_TABLE])
    rows = con.execute(
        f"SELECT DISTINCT trade_date FROM {bd.PRICE_TABLE} "
        "WHERE NOT (open = 0 AND high = 0 AND low = 0) ORDER BY 1"
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def label_end_dates(config: LakeConfig, horizon: int) -> pl.DataFrame:
    """``(ticker, market, trade_date) -> label_end_date`` for one horizon.

    The same join ``labels._forward_cte`` makes — per-ticker, halt days removed,
    ``d_idx + h`` — pulled out so the boundary can be checked without building a
    panel. ``build_label_scan_sql`` computes the same column for the horizon
    scan; this is that idea at model 02's grain.
    """
    con = duckdb.connect()
    register_views(con, config, tables=[bd.PRICE_TABLE])
    frame = pl.from_arrow(con.execute(f"""
            WITH px AS (
                SELECT trade_date, ticker, market,
                       ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS d_idx
                FROM {bd.PRICE_TABLE}
                WHERE NOT (open = 0 AND high = 0 AND low = 0)
            )
            SELECT a.trade_date, a.ticker, a.market, f.trade_date AS label_end_date
            FROM px a JOIN px f
              ON f.ticker = a.ticker AND f.market = a.market AND f.d_idx = a.d_idx + {horizon}
        """).arrow())
    con.close()
    return frame


def last_clean_train_formation(
    sessions: list, horizon: int, holdout_start: str
) -> tuple[object, int]:
    """The last formation date whose *calendar* label end is before the wall.

    ``walk_forward_splits`` expresses this as ``purge``; the same arithmetic is
    here so the preflight can print it without building folds. Returns the date
    and the number of sessions purged.
    """
    before = [d for d in sessions if str(d) < holdout_start]
    if len(before) <= horizon:
        raise ValueError(
            f"only {len(before)} sessions before {holdout_start}; cannot purge {horizon}"
        )
    return before[-1 - horizon], horizon


def resolve_boundary(
    adopted: AdoptedRun,
    config: LakeConfig,
    *,
    eval_last_formation: str | None,
    holdout_start: str = HOLDOUT_START,
) -> Boundary:
    """Resolve every date the opening needs, and count what crosses each one."""
    h = adopted.horizon
    sessions = session_calendar(config)
    ends = label_end_dates(config, h)

    train_last, purge = last_clean_train_formation(sessions, h, holdout_start)
    # Two ends for the same formation date. The market one is what the calendar
    # purge was sized against; the other is the worst halted name, and the gap
    # between them is exactly what the row-level purge exists for.
    train_end_market = sessions[sessions.index(train_last) + h]
    train_end_halted = (
        ends.filter(pl.col("trade_date") == train_last).get_column("label_end_date").max()
    )

    # The rows the calendar purge does not catch: a name halted for months has
    # a d_idx that skips the halt, so its "+h sessions" is a later date than the
    # market's. Those rows are dropped by value, not by date.
    stale = ends.filter(
        (pl.col("trade_date") <= train_last)
        & (pl.col("trade_date") >= pl.lit(PERIOD_START).str.to_date())
        & (pl.col("label_end_date") >= pl.lit(holdout_start).str.to_date())
    )

    boundary = Boundary(
        snapshot_date=config.snapshot_date,
        horizon=h,
        holdout_start=holdout_start,
        train_first_formation=PERIOD_START,
        train_last_formation=str(train_last),
        train_last_label_end=str(train_end_market),
        train_last_label_end_halted=str(train_end_halted),
        purge_sessions=purge,
        n_train_rows_dropped_by_label_end=int(stale.height),
        n_train_tickers_dropped_by_label_end=int(stale.get_column("ticker").n_unique()),
        price_data_last_date=str(sessions[-1]),
    )

    eval_sessions = [d for d in sessions if str(d) >= holdout_start]
    if eval_last_formation:
        if eval_last_formation > str(sessions[-1]):
            boundary.notes.append(
                f"요청한 평가 끝 {eval_last_formation} 이 이 snapshot 의 가격 마지막 날 "
                f"{sessions[-1]} 보다 뒤다 — 데이터가 있는 데까지만 잘렸다"
            )
        eval_sessions = [d for d in eval_sessions if str(d) <= eval_last_formation]
    if not eval_sessions:
        boundary.notes.append("평가 구간에 세션이 없다 — eval_last_formation 을 확인하라")
        return boundary

    boundary.eval_first_formation = str(eval_sessions[0])
    boundary.eval_last_formation = str(eval_sessions[-1])
    boundary.n_eval_sessions = len(eval_sessions)

    grid = rebalance_grid(eval_sessions, h)
    boundary.rebalance_dates = [str(d) for d in grid]
    boundary.n_rebalances_held = len(grid)

    # A rebalance is scored only if its label closed inside the data we hold.
    # Two readings, because they answer different questions. The market one is
    # the wall: a formation date needs ``h`` sessions after it in the calendar
    # or no name's label can have closed. The per-name one is the texture:
    # ``ends`` is an inner join on ``d_idx + h``, so a name missing from it at
    # date ``d`` is a name whose label has not closed — halted names lag here.
    # ``topk_economic_report`` silently drops both; counting them is what turns
    # "13~14 회" into a measured number.
    index_of = {d: i for i, d in enumerate(sessions)}
    horizon_reach = len(sessions) - 1
    closed = (
        ends.filter(pl.col("trade_date").is_in(grid))
        .group_by("trade_date")
        .agg(
            pl.col("label_end_date").max().alias("last_end"),
            pl.len().alias("n_closed"),
        )
    )
    closed_by_date = {row["trade_date"]: row for row in closed.iter_rows(named=True)}
    matured, unmatured = [], []
    for d in grid:
        if index_of[d] + h <= horizon_reach and d in closed_by_date:
            matured.append(d)
        else:
            unmatured.append(d)
    boundary.n_rebalances_matured = len(matured)
    boundary.unmatured_rebalance_dates = [str(d) for d in unmatured]
    boundary.closed_names_by_rebalance = {
        str(d): int(closed_by_date[d]["n_closed"]) if d in closed_by_date else 0 for d in grid
    }
    if matured:
        boundary.eval_last_label_end = str(sessions[index_of[matured[-1]] + h])
    if unmatured:
        boundary.notes.append(
            f"라벨이 안 닫힌 리밸런스 {len(unmatured)}개는 개수로만 기록하고 점수에 넣지 않는다"
        )
    return boundary


# --- 2. the holdout panel -------------------------------------------------------


def _widen_to_holdout(spec: ModelSpec, eval_last_formation: str) -> ModelSpec:
    """Push the adopted spec's formation ceiling out to the evaluation end.

    ``ModelSpec.__post_init__`` refuses this value (D-2), and that refusal is
    doing its job everywhere else: the selection window must not be able to
    reach past 2025-07-31 by accident. Opening the holdout *is* crossing that
    wall, so the crossing lives in one named function, is reachable only from
    ``--open``, and is written into ``boundary.json`` when it happens.
    """
    if eval_last_formation <= PERIOD_END:
        raise ValueError(
            f"eval_last_formation {eval_last_formation!r} is not past the wall "
            f"{PERIOD_END!r} — nothing to open"
        )
    widened = ModelSpec(
        model_id=spec.model_id,
        horizons=spec.horizons,
        label=spec.label,
        feature_set=spec.feature_set,
        flow_variant=spec.flow_variant,
        preprocess_profile=spec.preprocess_profile,
        universe=spec.universe,
        period_start=spec.period_start,
        period_end=PERIOD_END,
        n_folds=spec.n_folds,
        seed=spec.seed,
        topk=spec.topk,
        cost_bps_roundtrip=spec.cost_bps_roundtrip,
        tau=spec.tau,
    )
    object.__setattr__(widened, "period_end", eval_last_formation)
    return widened


def holdout_dataset_dir(
    adopted: AdoptedRun, spec: ModelSpec, *, suffix: str = HOLDOUT_DATASET_SUFFIX
) -> Path:
    """Beside the adopted dataset, never on top of it."""
    key = bd.dataset_key(spec, adopted.horizon)
    return adopted.dataset_dir.parent / f"{key}{suffix}"


def build_holdout_panel(
    adopted: AdoptedRun,
    config: LakeConfig,
    boundary: Boundary,
    *,
    write: bool = True,
    suffix: str = HOLDOUT_DATASET_SUFFIX,
) -> tuple[Path, list[Fold], pl.DataFrame]:
    """Build the panel over ``[PERIOD_START, eval_last_formation]``.

    Mirrors ``build_dataset.build_dataset`` — same panel SQL, same interactions,
    same preprocessing — with two differences, both of which are the point:
    the period reaches past the wall, and the folds carry a ``holdout`` role
    whose train end is already purged by ``walk_forward_splits``.

    ``LabelSpec.index_bench`` is forced on: the gate's I, S and D are measured
    against the index (`§3.1`), and ``raw_label_idx_{h}d`` does not exist
    without it. It adds columns beside ``raw_label_{h}d`` rather than replacing
    it, so the training label is the adopted one (`execution_plan` §3.3).

    The wall is crossed only when there is a wall to cross. A dress rehearsal
    ends inside the selection window, so its spec is an ordinary legal one and
    ``_widen_to_holdout`` is never called — which is what makes the rehearsal
    safe to run as often as it needs to be.
    """
    from dataclasses import replace as dc_replace

    h = adopted.horizon
    base = adopted.spec()
    spec = dc_replace(base, label=dc_replace(base.label, index_bench=True))
    if boundary.eval_last_formation > PERIOD_END:
        spec = _widen_to_holdout(spec, boundary.eval_last_formation)
    else:
        spec = dc_replace(spec, period_end=boundary.eval_last_formation)

    columns, materials = bd.panel_feature_columns(spec, h)
    marts = sorted(bd.required_mart_columns([*columns, *materials]))

    con = bd.connect(config)
    register_views(con, config, tables=[bd.PRICE_TABLE, bd.INDEX_TABLE])
    contracts = bd.register_read_only(con, config, [bd.UNIVERSE_VIEW, *marts])
    bd.verify_universe_contract(config, spec)  # D-5: the stored universe is the adopted one
    con.execute(
        f"CREATE OR REPLACE VIEW {bd.LABEL_VIEW} AS {labels_mod.build_label_sql(spec.label)}"
    )
    panel = pl.from_arrow(con.execute(bd.build_panel_sql(spec, h)).arrow())
    con.close()

    interactions = [i for i in bd.fx.INTERACTIONS if i[0] in columns]
    panel = bd.add_interactions(panel, interactions)
    panel = bd._restrict_columns(panel, columns)

    # Row-level purge: the halted names the calendar purge cannot see.
    ends = label_end_dates(config, h)
    panel = panel.join(ends, on=["trade_date", "ticker", "market"], how="left")

    dates = panel.get_column("trade_date").unique().sort().to_list()
    holdout_len = sum(1 for d in dates if str(d) >= boundary.holdout_start)
    folds = walk_forward_splits(
        dates,
        horizon=h,
        embargo=h,
        purge=h,
        n_folds=spec.n_folds,
        holdout_len=holdout_len,
    )
    holdout_fold = next(f for f in folds if f.role == "holdout")

    dataset_dir = holdout_dataset_dir(adopted, spec, suffix=suffix)
    cfg_pp = pp.PreprocessConfig(profile=spec.preprocess_profile)
    if not pp.is_stateless(cfg_pp):
        raise ValueError(
            f"preprocess profile {spec.preprocess_profile!r} is fitted per fold; the "
            "holdout design matrix would have to be fit on the train range only. "
            "The adopted profile is 'rank', which is stateless."
        )
    fitted = pp.fit(panel.drop("label_end_date"), cfg_pp)
    panel_std = fitted.transform(panel.drop("label_end_date")).join(
        panel.select(["trade_date", "ticker", "market", "label_end_date"]),
        on=["trade_date", "ticker", "market"],
        how="left",
    )

    if write:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "feat_panel_std").mkdir(exist_ok=True)
        panel_std.write_parquet(dataset_dir / "feat_panel_std" / "part-000000.parquet")
        panel.write_parquet(dataset_dir / "feat_panel.parquet")
        label_cols = [c for c in panel.columns if c.startswith(("y_", "raw_label", "fwd_ret_"))]
        panel.select([*bd.KEY_COLS, *label_cols, "label_end_date"]).write_parquet(
            dataset_dir / "label_daily.parquet"
        )
        pl.DataFrame([f.as_record() for f in folds]).write_parquet(
            dataset_dir / "split_folds.parquet"
        )
        (dataset_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "model_id": spec.model_id,
                    "snapshot_date": config.snapshot_date,
                    "lake": {
                        "raw": str(config.raw_root),
                        "feature_mart": str(config.feature_mart_root),
                    },
                    "period": {"start": spec.period_start, "end": spec.period_end},
                    "label_spec": {"index_bench": True, "horizons": list(spec.label.horizons)},
                    "row_count": panel.height,
                    "mart_contracts": contracts,
                    "holdout_fold": holdout_fold.as_record(),
                    "holdout_start": boundary.holdout_start,
                    "built_by": "holdout_run",
                    "is_rehearsal": suffix == REHEARSAL_DATASET_SUFFIX,
                },
                indent=2,
                default=str,
            )
        )
    return dataset_dir, folds, panel_std


# --- 3. one fit -----------------------------------------------------------------


def fit_once(
    train_std: pl.DataFrame,
    eval_std: pl.DataFrame,
    adopted: AdoptedRun,
    *,
    design: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Fit the adopted model on ``train_std`` once and score ``eval_std``.

    One call to ``train._fit_fold`` with the stored parameters — the same
    function every validation fold went through, so the holdout's model is not
    a second implementation of the adopted one. No sweep, no refit, no second
    model to choose between.
    """
    config = adopted.train_config()
    design = design or tr.design_columns(train_std, config.target_column)
    params = dict(adopted.best_params)
    predictions, calibrated, empty = tr._fit_fold(train_std, eval_std, design, config, params)
    if calibrated != (config.calibrate != "none"):
        raise AssertionError(
            f"calibration mismatch: config says {config.calibrate!r}, fit returned {calibrated}"
        )
    if empty:
        print(f"  학습 구간에 값이 하나도 없는 설계 컬럼 {len(empty)}개: {sorted(empty)}")
    return predictions, list(empty)


def assert_no_overlap(train_std: pl.DataFrame, eval_std: pl.DataFrame) -> None:
    """The last line of defence, written where a reader will look for it."""
    if train_std.is_empty() or eval_std.is_empty():
        raise ValueError("학습 또는 평가 프레임이 비었다")
    t_max = train_std.get_column("trade_date").max()
    e_min = eval_std.get_column("trade_date").min()
    if t_max >= e_min:
        raise AssertionError(f"학습 끝 {t_max} 가 평가 시작 {e_min} 보다 앞서지 않는다")
    shared = set(train_std.get_column("trade_date").unique().to_list()) & set(
        eval_std.get_column("trade_date").unique().to_list()
    )
    if shared:
        raise AssertionError(f"학습·평가에 겹치는 날짜 {len(shared)}개")


def purge_train_by_label_end(
    train_std: pl.DataFrame, holdout_start: str
) -> tuple[pl.DataFrame, int]:
    """Drop training rows whose label closes at or after the wall.

    The calendar purge removes the last ``h`` formation sessions. This removes
    what it cannot see: a name halted across the boundary, whose ``d_idx + h``
    lands on the far side even though its formation date is years earlier.
    """
    if "label_end_date" not in train_std.columns:
        raise ValueError("label_end_date 컬럼이 없다 — 이 패널은 holdout_run 이 만든 것이 아니다")
    keep = train_std.filter(
        pl.col("label_end_date").is_null()
        | (pl.col("label_end_date") < pl.lit(holdout_start).str.to_date())
    )
    return keep, int(train_std.height - keep.height)


# --- 4. the gate ----------------------------------------------------------------


@dataclass
class GateMetrics:
    """E, E', I, S, D — and nothing that could be mistaken for one of them.

    Every field is a number the frozen rule names (`§3.1`). The verdict reads
    signs only, in the order the rule fixes: C first, then A, then B.
    """

    e: float
    e_prime: float
    i: float
    s: float
    d: float
    d_longest_rebalances: int
    n_rebalances_e: int
    n_rebalances_i: int
    turnover_base: float
    turnover_tradable: float
    n_obs_base: int
    n_obs_tradable: int
    verdict: str
    verdict_reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def gate_metrics(frame: pl.DataFrame, adopted: AdoptedRun) -> GateMetrics:
    """Compute the five, on the two universes and the two benchmarks.

    ``frame`` must already carry ``tradable`` (R1) and both label columns; the
    caller assembles it with ``preopen_measures._with_universe_and_buckets`` so
    the tradable flag is the same one R1 measured on the formation window.
    """
    h, k, cost = adopted.horizon, adopted.k, adopted.cost_bps_roundtrip
    pred = adopted.primary_pred_col
    eqw, idx = adopted.eqw_label, adopted.idx_label
    if idx not in frame.columns:
        raise ValueError(f"{idx!r} 가 없다 — 패널을 index_bench 없이 만들었다. I·S·D 를 잴 수 없다")
    tradable = frame.filter(pl.col("tradable"))

    def topk(part: pl.DataFrame, realized: str):
        return topk_economic_report(
            part, pred_col=pred, realized_col=realized, horizon=h, k=k, cost_bps_roundtrip=cost
        )

    base_eqw = topk(frame, eqw)
    trad_eqw = topk(tradable, eqw)
    trad_idx = topk(tradable, idx)
    spread, _tmb, _hit, _n = _quantile_stats(
        tradable, pred_col=pred, realized_col=idx, date_col="trade_date"
    )
    # S's cost term follows ``topk_economic_report``'s convention exactly: a
    # turnover that could not be measured (one rebalance, so no consecutive
    # pair) costs nothing rather than poisoning the metric. Without this S goes
    # NaN while E does not, and the branch rule would refuse to judge a window
    # it can perfectly well judge.
    turn = trad_idx.turnover
    turn_cost = 0.0 if turn != turn else turn * cost / 10_000.0
    s_net = spread - turn_cost if spread == spread else float("nan")

    e, i, s = base_eqw.cost_adjusted_return, trad_idx.cost_adjusted_return, s_net
    verdict, reason = decide(e, i, s)
    return GateMetrics(
        e=e,
        e_prime=trad_eqw.cost_adjusted_return,
        i=i,
        s=s,
        d=trad_idx.max_drawdown,
        d_longest_rebalances=trad_idx.longest_drawdown_rebalances,
        n_rebalances_e=base_eqw.n_rebalances,
        n_rebalances_i=trad_idx.n_rebalances,
        turnover_base=base_eqw.turnover,
        turnover_tradable=trad_idx.turnover,
        n_obs_base=int(frame.height),
        n_obs_tradable=int(tradable.height),
        verdict=verdict,
        verdict_reason=reason,
    )


def decide(e: float, i: float, s: float) -> tuple[str, str]:
    """The frozen branch rule (`§3.2`), in its fixed order: C, then A, then B.

    Order matters and the plan says why: ``E <= 0`` with ``I > 0`` would satisfy
    both A and C, and R5 measured that the universe beats the index, so that
    combination is not rare. C first means "no stock selection" wins over "the
    universe beat the index".
    """
    for value, name in ((e, "E"), (i, "I"), (s, "S")):
        if value != value:
            return "미판정", f"{name} 가 NaN — 판정하지 않는다"
    if e <= 0:
        return "C", f"E={e:.6f} <= 0"
    if i > 0 and s > 0:
        return "A", f"E={e:.6f} > 0, I={i:.6f} > 0, S={s:.6f} > 0"
    return "B", f"E={e:.6f} > 0 이지만 I={i:.6f}, S={s:.6f} 중 하나가 0 이하"


# --- 5. the full report set -----------------------------------------------------


def write_reports(
    frame: pl.DataFrame,
    adopted: AdoptedRun,
    boundary: Boundary,
    gate: GateMetrics,
    out_dir: Path,
) -> None:
    """R1, R2, R3, T1-1B, T1-2, T1-6 and the rebalance path, in one pass.

    All of them come from ``preopen_measures``, which already computed them on
    the formation window: the holdout has to be read against those numbers, and
    two implementations of R2 would make that comparison meaningless. The frame
    is tagged ``fold_id = 1`` because the holdout is one window, not five.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **adopted.summary,
        "horizon": adopted.horizon,
        "k": adopted.k,
        "cost_bps_roundtrip": adopted.cost_bps_roundtrip,
        "primary_pred_col": adopted.primary_pred_col,
    }
    tagged = frame.with_columns(pl.lit(1).alias("fold_id"))

    def dump(rows: list[dict], name: str) -> None:
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(out_dir / name)
        print(f"  {name}: {len(rows)} rows")

    dump(pm._tradable_rows(tagged, summary), "r1_tradable.parquet")
    dump(pm._bucket_rows(tagged, summary), "r2_by_bucket.parquet")
    dump(pm._drawdown_rows(tagged, summary), "r3_drawdown.parquet")
    dump(pm._neutralization_rows(tagged, summary), "t1_1b_neutralization.parquet")
    dump(pm._hysteresis_rows(tagged, summary), "t1_6_hysteresis.parquet")
    dump(pm._cost_curve_rows(tagged, summary), "t1_2_cost_curve.parquet")

    for label, realized in (("eqw", adopted.eqw_label), ("idx", adopted.idx_label)):
        series = topk_rebalance_series(
            tagged.filter(pl.col("tradable")),
            pred_col=adopted.primary_pred_col,
            realized_col=realized,
            horizon=adopted.horizon,
            k=adopted.k,
            cost_bps_roundtrip=adopted.cost_bps_roundtrip,
        )
        series.write_parquet(out_dir / f"rebalance_returns_{label}.parquet")
        print(f"  rebalance_returns_{label}.parquet: {series.height} rows")

    (out_dir / "gate.json").write_text(json.dumps(gate.as_dict(), indent=2))
    (out_dir / "boundary.json").write_text(json.dumps(asdict(boundary), indent=2, default=str))


def render_summary(adopted: AdoptedRun, boundary: Boundary, gate: GateMetrics) -> str:
    """The one page a reader opens first. Numbers, then their limits."""
    head = (
        f"- snapshot: **{boundary.snapshot_date}** · horizon **h{boundary.horizon}**"
        f" · k={adopted.k} · 비용 {adopted.cost_bps_roundtrip:.0f}bp\n"
        f"- 학습 formation: {boundary.train_first_formation}"
        f" \\~ **{boundary.train_last_formation}** (라벨 종료 {boundary.train_last_label_end})\n"
        f"- 평가 formation: **{boundary.eval_first_formation}"
        f" \\~ {boundary.eval_last_formation}** (라벨 종료 {boundary.eval_last_label_end})\n"
        f"- 리밸런스: 보유 **{boundary.n_rebalances_held}회**"
        f" · 성숙 **{boundary.n_rebalances_matured}회**"
    )
    tail = (
        f"D 는 판정에 쓰지 않는다. 최장 손실 {gate.d_longest_rebalances} 리밸런스,"
        f" 총 {gate.n_rebalances_i} 회다."
    )
    return f"""# holdout 결과 — {adopted.run_id}

{head}

## 판정

| 기호 | 값 | 유니버스 | 벤치 |
|---|---:|---|---|
| **E** | {gate.e:+.6f} | 기본 | 동일가중 |
| **I** | {gate.i:+.6f} | 거래 가능 | 지수 |
| **S** | {gate.s:+.6f} | 거래 가능 | 지수 |
| E′ | {gate.e_prime:+.6f} | 거래 가능 | 동일가중 |
| D | {gate.d:+.6f} | 거래 가능 | 지수 |

**갈래 {gate.verdict}** — {gate.verdict_reason}

{tail}

## 읽을 때의 한계

- **모델02 의 첫 최종 평가이지, 프로젝트 전체의 미사용 구간이 아니다.**
  모델01 이 이미 본 구간과 겹친다.
- 비용은 관측 데이터에 가정한 비용식을 적용한 추정이다. 실제 체결 비용은 아직 재지 않았다.
- D 는 리밸런스 단위 초과수익 경로의 낙폭이고, 계좌의 일별 최대 손실이 아니다.
- 평가 구간의 가격 품질과 입력 커버리지는 `boundary.json` 과 품질 기록을 같이 읽는다.
"""


# --- 6. modes -------------------------------------------------------------------


def preflight(eval_last_formation: str | None, *, snapshot: str | None = None) -> Boundary:
    """Resolve and print. Fits nothing, reads no holdout label, writes one file."""
    adopted = load_adopted()
    config = lake_config()
    if snapshot:
        config = LakeConfig(root=config.root, snapshot_date=snapshot, source=config.source)
    print(f"채택 run  : {adopted.run_id} (h{adopted.horizon}, k={adopted.k}, {adopted.model})")
    print(f"저장 파라미터: {adopted.best_params}")
    print(f"snapshot  : {config.snapshot_date} / {config.source}")
    boundary = resolve_boundary(adopted, config, eval_last_formation=eval_last_formation)

    print("\n세 날짜")
    print(
        f"  snapshot 생성일        : {boundary.snapshot_date}"
        f" (가격 마지막 {boundary.price_data_last_date})"
    )
    print(f"  마지막 평가 formation  : {boundary.eval_last_formation or '(미지정)'}")
    print(f"  마지막 성숙 리밸런스의 라벨 종료일: {boundary.eval_last_label_end or '(없다)'}")
    print("\n학습 경계")
    print(
        f"  마지막 학습 formation  : {boundary.train_last_formation}"
        f" (달력 purge {boundary.purge_sessions} 세션)"
    )
    print(
        f"  그 라벨의 종료일       : {boundary.train_last_label_end} (시장 세션 기준)"
        f" · 정지 지연 최악 {boundary.train_last_label_end_halted}"
    )
    print(
        f"  라벨 종료일로 빼는 행  : {boundary.n_train_rows_dropped_by_label_end:,}행 "
        f"/ {boundary.n_train_tickers_dropped_by_label_end}종목 (거래정지 지연)"
    )
    print("\n평가 격자")
    print(
        f"  세션 {boundary.n_eval_sessions}일 → 리밸런스 보유 {boundary.n_rebalances_held}회 "
        f"· 성숙 {boundary.n_rebalances_matured}회"
    )
    if boundary.unmatured_rebalance_dates:
        print(f"  안 닫힌 리밸런스: {boundary.unmatured_rebalance_dates}")
    for note in boundary.notes:
        print(f"  ! {note}")

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    path = HOLDOUT_DIR / f"preflight_{config.snapshot_date}_{boundary.horizon}.json"
    path.write_text(json.dumps(asdict(boundary), indent=2, default=str))
    print(f"\n  {path.relative_to(RESULTS_ROOT)}")
    return boundary


def rehearse(fold_id: int) -> dict:
    """Rebuild a recorded fold through this runner and compare, row by row.

    The check that matters is not "does it look right" but "is it the same
    model". Same dataset, same design, same stored parameters, same
    ``_fit_fold`` — so the predictions must match the ones the adopted run
    wrote, to the bit. If they do, the only thing the opening changes is which
    rows go in.
    """
    adopted = load_adopted()
    source = tr.DatasetFolds(adopted.dataset_dir)
    train_std, valid_std = source.slices(fold_id)
    design = source.design_columns(adopted.train_config().target_column)
    print(
        f"fold {fold_id}: 학습 {train_std.height:,}행 · 검증 {valid_std.height:,}행"
        f" · 설계 {len(design)}열"
    )
    assert_no_overlap(train_std, valid_std)

    started = time.time()
    predictions, _empty = fit_once(train_std, valid_std, adopted, design=design)
    print(f"  적합 1회 {time.time() - started:.1f}s")

    recorded = pl.read_parquet(adopted.predictions_path).filter(
        pl.col("trade_date").is_between(
            valid_std.get_column("trade_date").min(), valid_std.get_column("trade_date").max()
        )
    )
    joined = predictions.select(["trade_date", "ticker", "market", "p_raw"]).join(
        recorded.select(["trade_date", "ticker", "market", pl.col("p_raw").alias("p_recorded")]),
        on=["trade_date", "ticker", "market"],
        how="inner",
    )
    diff = (joined.get_column("p_raw") - joined.get_column("p_recorded")).abs()
    out = {
        "fold_id": fold_id,
        "n_new": int(predictions.height),
        "n_recorded": int(recorded.height),
        "n_joined": int(joined.height),
        "max_abs_diff": float(diff.max()) if joined.height else float("nan"),
        "n_exact": int((diff == 0).sum()) if joined.height else 0,
    }
    print(f"  대조: 새 {out['n_new']:,} · 기록 {out['n_recorded']:,} · 조인 {out['n_joined']:,}")
    print(f"  최대 절대 차 {out['max_abs_diff']:.3e} · 완전 일치 {out['n_exact']:,}행")
    out["pass"] = bool(
        out["n_joined"] == out["n_recorded"] == out["n_new"] and out["max_abs_diff"] == 0.0
    )
    print(f"  {'통과' if out['pass'] else '불일치 — 학습 경로가 채택 run 과 다르다'}")
    return out


def _lake_for(snapshot: str | None) -> LakeConfig:
    config = lake_config()
    if snapshot:
        config = LakeConfig(root=config.root, snapshot_date=snapshot, source=config.source)
    return config


def run_window(
    *,
    eval_last_formation: str,
    holdout_start: str,
    snapshot: str | None,
    rehearsal: bool,
) -> GateMetrics:
    """Train once, score one window, write the whole report set.

    The opening and the dress rehearsal are the same procedure over different
    dates, so they are the same function. That is the point of the rehearsal:
    a rehearsal that ran a different code path would only prove that the
    rehearsal works.
    """
    adopted = load_adopted()
    config = _lake_for(snapshot)
    boundary = resolve_boundary(
        adopted, config, eval_last_formation=eval_last_formation, holdout_start=holdout_start
    )
    suffix = REHEARSAL_DATASET_SUFFIX if rehearsal else HOLDOUT_DATASET_SUFFIX
    out_dir = HOLDOUT_DIR / (f"rehearsal_{holdout_start}" if rehearsal else adopted.run_id)

    label = "리허설" if rehearsal else "개봉"
    print(f"[{label}] 경계 {holdout_start} · 평가 끝 {boundary.eval_last_formation}")
    print(
        f"  리밸런스 보유 {boundary.n_rebalances_held}회"
        f" · 성숙 {boundary.n_rebalances_matured}회"
    )

    started = time.time()
    dataset_dir, folds, panel_std = build_holdout_panel(adopted, config, boundary, suffix=suffix)
    holdout_fold = next(f for f in folds if f.role == "holdout")
    print(f"  패널 {panel_std.height:,}행 · {time.time() - started:.0f}s → {dataset_dir.name}")
    print(
        f"  마지막 fold: 학습 \\~{holdout_fold.train_end}"
        f" · 평가 {holdout_fold.valid_start}\\~{holdout_fold.valid_end}"
    )

    train_std = panel_std.filter(pl.col("trade_date") <= holdout_fold.train_end)
    eval_std = panel_std.filter(
        pl.col("trade_date").is_between(holdout_fold.valid_start, holdout_fold.valid_end)
    )
    train_std, dropped = purge_train_by_label_end(train_std, boundary.holdout_start)
    boundary.n_train_rows_dropped_by_label_end = dropped
    print(
        f"  라벨 종료일로 뺀 학습 행 {dropped:,}"
        f" · 학습 {train_std.height:,} · 평가 {eval_std.height:,}"
    )
    assert_no_overlap(train_std, eval_std)

    started = time.time()
    predictions, _empty = fit_once(train_std, eval_std, adopted)
    print(f"  적합 1회 {time.time() - started:.1f}s · 예측 {predictions.height:,}행")

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "predictions_rehearsal" if rehearsal else "predictions_holdout"
    pred_path = dataset_dir / f"{prefix}__{adopted.run_id}.parquet"
    predictions.write_parquet(pred_path)

    summary = {**adopted.summary, "dataset_dir": str(dataset_dir)}
    frame = pm._with_cost_inputs(summary, predictions.with_columns(pl.lit(1).alias("fold_id")))
    frame = pm._with_universe_and_buckets(summary, frame)
    frame = frame.join(
        panel_std.select(["trade_date", "ticker", "market", adopted.idx_label]),
        on=["trade_date", "ticker", "market"],
        how="left",
    )
    gate = gate_metrics(frame, adopted)
    write_reports(frame, adopted, boundary, gate, out_dir)
    text = render_summary(adopted, boundary, gate)
    if rehearsal:
        text = (
            "> **리허설이다. 개봉이 아니다.** 경계를 "
            f"{holdout_start} 로 옮겨 formation 구간에서 개봉 절차를 그대로 돌린 것이고,\n"
            "> 아래 숫자는 판정에 쓰지 않는다. 채택 후보를 고를 때 이미 본 구간이다.\n\n"
        ) + text
    (out_dir / "summary.md").write_text(text)
    (out_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "git_sha": current_git_sha(),
                "code_hash": code_hash(),
                "model_code_hash": model_code_hash(),
                "adopted_run": adopted.run_id,
                "best_params": adopted.best_params,
                "grid_points_searched": 0,
                "is_rehearsal": rehearsal,
                "holdout_start": holdout_start,
                "snapshot_date": config.snapshot_date,
                "dataset_dir": str(dataset_dir),
                "predictions_path": str(pred_path),
                "boundary": asdict(boundary),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n갈래 {gate.verdict} — {gate.verdict_reason}")
    print(
        f"  E={gate.e:+.6f} E'={gate.e_prime:+.6f}"
        f" I={gate.i:+.6f} S={gate.s:+.6f} D={gate.d:+.6f}"
    )
    print(f"  {out_dir}")
    return gate


def dress_rehearse(
    pseudo_holdout_start: str,
    eval_last_formation: str = PERIOD_END,
    *,
    snapshot: str | None = None,
) -> GateMetrics:
    """The whole opening procedure, with the wall moved back inside selection.

    Everything the opening does — panel, folds, both purges, the single fit,
    the gate and the report set — over a window the candidate selection already
    saw. The numbers are worthless as evidence and the header says so; what is
    worth having is that the procedure ran end to end before the day it has to.
    """
    if pseudo_holdout_start >= HOLDOUT_START:
        raise ValueError(
            f"리허설 경계 {pseudo_holdout_start!r} 가 진짜 경계 {HOLDOUT_START!r} 앞이 아니다. "
            "리허설은 formation 구간 안에서만 한다"
        )
    if eval_last_formation > PERIOD_END:
        raise ValueError(
            f"리허설 평가 끝 {eval_last_formation!r} 이 선택 구간 끝 {PERIOD_END!r} 을 넘는다"
        )
    return run_window(
        eval_last_formation=eval_last_formation,
        holdout_start=pseudo_holdout_start,
        snapshot=snapshot,
        rehearsal=True,
    )


def open_holdout(eval_last_formation: str, *, snapshot: str | None = None) -> GateMetrics:
    """The opening. Once."""
    return run_window(
        eval_last_formation=eval_last_formation,
        holdout_start=HOLDOUT_START,
        snapshot=snapshot,
        rehearsal=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="날짜·경계·성숙만 푼다 (기본)")
    mode.add_argument(
        "--rehearse-fold", type=int, default=None, help="기록된 fold 로 학습 경로를 대조한다"
    )
    mode.add_argument(
        "--dress-rehearse",
        metavar="PSEUDO_HOLDOUT_START",
        default=None,
        help="개봉 절차 전체를 formation 구간의 가짜 경계로 돌린다 (예: 2024-07-08)",
    )
    mode.add_argument("--open", action="store_true", help="개봉한다 — 한 번만")
    parser.add_argument(
        "--eval-last-formation", default=None, help="마지막으로 평가할 예측일 (YYYY-MM-DD)"
    )
    parser.add_argument("--snapshot", default=None, help="snapshot_date (기본: spec 의 고정값)")
    parser.add_argument(
        "--i-am-opening-the-holdout",
        action="store_true",
        help="--open 에 반드시 같이 준다. 이 플래그가 없으면 개봉하지 않는다",
    )
    args = parser.parse_args(argv)

    if args.rehearse_fold is not None:
        return 0 if rehearse(args.rehearse_fold)["pass"] else 1
    if args.dress_rehearse:
        dress_rehearse(
            args.dress_rehearse,
            args.eval_last_formation or PERIOD_END,
            snapshot=args.snapshot,
        )
        return 0
    if args.open:
        if not args.i_am_opening_the_holdout:
            parser.error("--open 에는 --i-am-opening-the-holdout 이 필요하다. 한 번만 여는 문이다")
        if not args.eval_last_formation:
            parser.error(
                "--open 에는 --eval-last-formation 이 필요하다. snapshot 날짜가 정하지 않는다"
            )
        open_holdout(args.eval_last_formation, snapshot=args.snapshot)
        return 0
    preflight(args.eval_last_formation, snapshot=args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
