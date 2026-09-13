"""train — walk-forward fitting for the three model families (M-A / M-B / M-C).

Plan: ``docs/dev/20260907_model_experiment/04_models_and_calibration.md`` §1
(the families), §2 (the fixed grids), §4 (the per-fold procedure), §6 (seeds).

The grids are module constants because they are preregistered: a grid that can
be widened after seeing a fold's metrics is not a grid, it is a search. `05`
also fixes *which* grid each stage runs — E1 uses the 4-point one, E2 onward the
16-point one — so both are here rather than assembled at the call site.

Selection inside a run is by mean valid log-loss for the two classifiers and by
mean Rank IC for the rank regressor, which is the metric model 01 selected on.
The winner is then **refit** to produce the predictions this module returns.
That costs one extra pass over the folds and saves holding 16 grid points x 5
folds of predictions in memory at once; HGB with a fixed ``random_state`` is
deterministic, so the refit is the same model.

Memory is the other thing shaping this module. A run never holds the whole
design matrix: :class:`DatasetFolds` reads one fold's train and valid slices
from the built dataset, the loop is **fold-outer, grid-inner** so those slices
are read twice per run rather than once per grid point, and each fold's frames
are dropped before the next. Reading the full ``feat_panel_std`` of an FS2 panel
would be ~25GB on a 36GB host; one fold is a few.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from modeler.etl.metrics import PROB_CLIP, binary_log_loss, per_date_rank_ic
from modeler.models._01_20_access_return_rank.train import design_columns
from modeler.models._02_updown_prob import calibrate as cal
from modeler.models._02_updown_prob import features as fx

MODELS: tuple[str, ...] = ("hgb_clf", "logit", "hgb_reg")
TARGETS: tuple[str, ...] = ("y_up", "y_top", "y_rank")

# M-B, the primary model. 16 points (`04` §2.1). min_samples_leaf is fixed at
# 200 so a leaf cannot hang on a handful of names out of ~1,000 per session;
# early stopping is off for determinism; class_weight stays None because
# reweighting the positives is exactly what breaks calibration.
HGB_CLF_GRID: tuple[dict, ...] = tuple(
    {"max_iter": m, "learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
    for m in (200, 400)
    for lr in (0.03, 0.1)
    for leaves in (15, 31)
    for l2 in (0.0, 1.0)
)
# E1's half grid: max_iter 400 and l2 0 fixed, so 4 points (`04` §2.1).
HGB_CLF_GRID_E1: tuple[dict, ...] = tuple(
    p for p in HGB_CLF_GRID if p["max_iter"] == 400 and p["l2_regularization"] == 0.0
)
# M-C (`04` §2.2). Rank features are already on [0,1], so no extra scaling.
LOGIT_GRID: tuple[dict, ...] = tuple({"C": c} for c in (0.1, 1.0, 10.0))
# M-A, model 01's grid verbatim (`04` §2.3).
HGB_REG_GRID: tuple[dict, ...] = tuple(
    {"max_iter": m, "learning_rate": lr} for m in (100, 200) for lr in (0.01, 0.1)
)

HGB_FIXED = {"min_samples_leaf": 200, "max_bins": 255, "early_stopping": False}


@dataclass(frozen=True)
class TrainConfig:
    """One model configuration: family, target, grid, seed, calibration."""

    model: str = "hgb_clf"
    target: str = "y_up"
    horizon: int = 20
    grid: tuple[dict, ...] = HGB_CLF_GRID_E1
    seed: int = 0
    calibrate: str = "none"
    # What the calibrator maps *to*. M-A trains on the rank label but its
    # probability has to be a probability of the same event the classifiers
    # predict, or E0's "probability baseline" would not be comparable to E1
    # (`04` §1, `05` §2 E0). None means "the training target".
    cal_target: str | None = None
    cal_frac: float = cal.DEFAULT_CAL_FRAC
    monotonic: bool = False
    date_col: str = "trade_date"
    id_cols: tuple[str, ...] = ("ticker", "market")

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")
        if self.calibrate not in cal.VALID_METHODS:
            raise ValueError(f"calibrate must be one of {cal.VALID_METHODS}")
        if not self.grid:
            raise ValueError("grid must be non-empty")
        if self.model == "hgb_reg" and self.target != "y_rank":
            raise ValueError("hgb_reg (M-A) is the rank baseline; its target is y_rank")
        if self.model != "hgb_reg" and self.target == "y_rank":
            raise ValueError(f"{self.model} is a classifier; y_rank is not a binary target")
        if self.monotonic and self.model != "hgb_clf":
            raise ValueError("monotonic constraints are an HGB classifier variant (E4b)")
        if self.cal_target is not None and self.cal_target not in ("y_up", "y_top"):
            raise ValueError(f"cal_target must be a binary label, got {self.cal_target!r}")
        if self.model == "hgb_reg" and self.calibrate != "none" and self.cal_target is None:
            raise ValueError(
                "hgb_reg produces a rank score; calibrating it needs an explicit "
                "cal_target (the binary event the probability is about)"
            )

    @property
    def is_classifier(self) -> bool:
        return self.model in ("hgb_clf", "logit")

    @property
    def target_column(self) -> str:
        return f"{self.target}_{self.horizon}d"

    @property
    def cal_target_column(self) -> str:
        """The label the calibrator (and the probability metrics) read."""
        target = self.cal_target or self.target
        return f"{target}_{self.horizon}d"

    @property
    def realized_column(self) -> str:
        return f"raw_label_{self.horizon}d"

    @property
    def selection_metric(self) -> str:
        """What the grid is chosen on — log-loss for probabilities, IC for M-A."""
        return "log_loss" if self.is_classifier else "rank_ic"


class FoldSource(Protocol):
    """Where a run gets its per-fold slices from."""

    def fold_ids(self) -> list[int]: ...

    def design_columns(self, target: str) -> list[str]: ...

    def slices(self, fold_id: int) -> tuple[pl.DataFrame, pl.DataFrame]: ...


@dataclass
class InMemoryFolds:
    """A design matrix already in memory, tagged with ``fold_id``/``fold_role``.

    What a test hands in, and what a caller holding one frame gets for free.
    """

    panel_std: pl.DataFrame

    def fold_ids(self) -> list[int]:
        wf = self.panel_std.filter(pl.col("fold_role").is_in(["train", "valid"]))
        return sorted(wf.get_column("fold_id").unique().to_list())

    def design_columns(self, target: str) -> list[str]:
        return design_columns(self.panel_std, target)

    def slices(self, fold_id: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        wf = self.panel_std.filter(pl.col("fold_id") == fold_id)
        return (
            wf.filter(pl.col("fold_role") == "train"),
            wf.filter(pl.col("fold_role") == "valid"),
        )


@dataclass
class DatasetFolds:
    """One fold at a time, read from a built dataset directory.

    Handles both layouts ``build_dataset`` writes. ``per_fold`` parts are read
    by name. The ``single`` layout — one transformed panel, which a stateless
    profile makes correct — is sliced by the fold table's date ranges, and the
    predicate goes into the parquet scan so only those row groups are read.
    """

    dataset_dir: Path
    std_dir_name: str = "feat_panel_std"

    def __post_init__(self) -> None:
        self._std = Path(self.dataset_dir) / self.std_dir_name
        if not self._std.is_dir() or not any(self._std.glob("*.parquet")):
            raise FileNotFoundError(f"no design matrix under {self._std}")
        self._folds = pl.read_parquet(Path(self.dataset_dir) / "split_folds.parquet")
        self._per_fold = bool(list(self._std.glob("fold*_*.parquet")))

    def fold_ids(self) -> list[int]:
        rows = self._folds.filter(pl.col("role") == "fold")
        return sorted(rows.get_column("fold_id").unique().to_list())

    def design_columns(self, target: str) -> list[str]:
        head = pl.scan_parquet(self._std / "*.parquet").head(0).collect()
        return design_columns(head, target)

    def slices(self, fold_id: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        if self._per_fold:
            return (self._part(fold_id, "train"), self._part(fold_id, "valid"))
        row = self._folds.filter(pl.col("fold_id") == fold_id).row(0, named=True)
        return (
            self._range(row["train_start"], row["train_end"], fold_id, "train"),
            self._range(row["valid_start"], row["valid_end"], fold_id, "valid"),
        )

    def _part(self, fold_id: int, role: str) -> pl.DataFrame:
        path = self._std / f"fold{fold_id}_{role}.parquet"
        return pl.read_parquet(path) if path.is_file() else pl.DataFrame()

    def _range(self, start, end, fold_id: int, role: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(self._std / "*.parquet")
            .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
            .collect()
            .with_columns(
                pl.lit(fold_id).alias("fold_id"),
                pl.lit(role).alias("fold_role"),
            )
        )


def _as_source(panel_std: pl.DataFrame | FoldSource) -> FoldSource:
    return InMemoryFolds(panel_std) if isinstance(panel_std, pl.DataFrame) else panel_std


@dataclass
class FoldFit:
    """One fold's outcome: its metric, and (for the winner) its predictions."""

    fold_id: int
    params: dict
    n_train: int
    n_valid: int
    metric: float
    calibrated: bool
    skipped: str | None = None
    empty_design_columns: list[str] = field(default_factory=list)
    predictions: pl.DataFrame | None = None


@dataclass
class TrainResult:
    """Grid sweep + the refit winner's per-fold predictions."""

    config: TrainConfig
    best_params: dict
    best_metric: float
    grid_metrics: list[dict]
    folds: list[FoldFit]
    design_columns: list[str] = field(default_factory=list)

    @property
    def predictions(self) -> pl.DataFrame:
        """Every fold's valid predictions, stacked."""
        frames = [f.predictions for f in self.folds if f.predictions is not None]
        return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()

    def summary(self) -> dict:
        return {
            "model": self.config.model,
            "target": self.config.target_column,
            "horizon": self.config.horizon,
            "seed": self.config.seed,
            "calibrate": self.config.calibrate,
            "monotonic": self.config.monotonic,
            "grid_size": len(self.config.grid),
            "selection_metric": self.config.selection_metric,
            "best_params": self.best_params,
            "best_metric": self.best_metric,
            "n_folds": len([f for f in self.folds if f.skipped is None]),
            "skipped_folds": {f.fold_id: f.skipped for f in self.folds if f.skipped},
            "empty_design_columns": {
                f.fold_id: f.empty_design_columns for f in self.folds if f.empty_design_columns
            },
        }


def make_model(config: TrainConfig, params: dict, design: list[str]):
    """Instantiate one grid point. Every knob not in the grid is fixed here."""
    if config.model == "logit":
        return LogisticRegression(C=params["C"], solver="lbfgs", max_iter=500)
    if config.model == "hgb_reg":
        return HistGradientBoostingRegressor(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            random_state=config.seed,
            **HGB_FIXED,
        )
    constraints = fx.monotonic_constraints(tuple(design)) if config.monotonic else None
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        class_weight=None,
        monotonic_cst=list(constraints) if constraints is not None else None,
        random_state=config.seed,
        **HGB_FIXED,
    )


def empty_columns(frame: pl.DataFrame, design: list[str]) -> list[str]:
    """Design columns with no observed value at all in ``frame``.

    A real case, not a hypothetical: the short-selling balance families start
    2016-06-30, so an early fold's train slice holds nothing for them. sklearn's
    histogram binner cannot bin an all-missing column — 1.9 raises numpy's
    "window shape cannot be larger than input array shape" from inside
    ``_BinMapper.fit``, which says nothing about the column that caused it.
    """
    if frame.height == 0:
        return []
    counts = frame.select(
        [
            (
                pl.col(c).null_count() + pl.col(c).is_nan().sum()
                if frame.schema[c].is_float()
                else pl.col(c).null_count()
            ).alias(c)
            for c in design
        ]
    ).row(0)
    return [c for c, missing in zip(design, counts, strict=True) if missing == frame.height]


def _matrix(
    frame: pl.DataFrame,
    design: list[str],
    *,
    fill_null: bool,
    fill_columns: tuple[str, ...] = (),
) -> np.ndarray:
    """Design matrix. ``fill_null`` is for the linear model, which cannot take NaN.

    Only flag-shaped columns can still be null at this point: the rank profile
    fills the ranked features, and ``ix_*``/``rg_*`` carry a real "off or
    unknown" state. 0.0 is that state's encoding, and the matching ``*_isna``
    column is what tells the model it was unknown rather than off. HGB reads the
    NaN natively and gets the sharper signal, which is why this is not applied
    to both.

    ``fill_columns`` are the columns :func:`empty_columns` found unobservable in
    this fold's fit slice; they become a constant 0.0 in the fit *and* predict
    matrices. A column with no values carries no information either way, and
    keeping it in the design (rather than dropping it) keeps the matrix — and a
    monotonic constraint vector — the same shape across folds.
    """
    columns = tuple(design) if fill_null else tuple(c for c in fill_columns if c in design)
    if columns:
        frame = frame.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in columns])
    return frame.select(design).to_numpy()


def _predict(model, x: np.ndarray, *, is_classifier: bool) -> np.ndarray:
    if is_classifier:
        return np.clip(model.predict_proba(x)[:, 1], PROB_CLIP, 1.0 - PROB_CLIP)
    return model.predict(x)


def _labelled(frame: pl.DataFrame, target: str) -> pl.DataFrame:
    """Rows with a resolved label — an unresolved one is not a training example."""
    out = frame.filter(pl.col(target).is_not_null())
    if out.schema[target].is_float():
        out = out.filter(pl.col(target).is_not_nan())
    return out


def _fit_fold(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    design: list[str],
    config: TrainConfig,
    params: dict,
) -> tuple[pl.DataFrame, bool, tuple[str, ...]]:
    """Fit one fold: ``(valid predictions, calibrated?, unobservable columns)``.

    With calibration on, the model is fit on the train range *minus* the
    calibration slice and its embargo (`04` §4), so ``p_raw`` and ``p_cal`` come
    from the same model — the alternative, fitting twice, would make the two
    columns incomparable. With calibration off the model gets the whole train
    range, which is why "none" is the default for the primary result.
    """
    target = config.target_column
    cal_target = config.cal_target_column
    labelled = _labelled(train, target)
    calibrator = None
    if config.calibrate != "none":
        split = cal.split_cal_slice(
            labelled.get_column(config.date_col).unique().to_list(),
            config.horizon,
            config.cal_frac,
        )
        fit_rows = labelled.filter(pl.col(config.date_col).is_in(split.model_dates))
        cal_rows = _labelled(
            labelled.filter(pl.col(config.date_col).is_in(split.cal_dates)), cal_target
        )
    else:
        fit_rows, cal_rows = labelled, labelled.clear()

    fill = tuple(empty_columns(fit_rows, design))
    linear = config.model == "logit"
    x = _matrix(fit_rows, design, fill_null=linear, fill_columns=fill)
    y = fit_rows.get_column(target).to_numpy()
    model = make_model(config, params, design)
    model.fit(x, y)

    if cal_rows.height:
        scores = _predict(
            model,
            _matrix(cal_rows, design, fill_null=linear, fill_columns=fill),
            is_classifier=config.is_classifier,
        )
        calibrator = cal.fit_calibrator(
            config.calibrate, scores, cal_rows.get_column(cal_target).cast(pl.Float64).to_numpy()
        )

    p_raw = _predict(
        model,
        _matrix(valid, design, fill_null=linear, fill_columns=fill),
        is_classifier=config.is_classifier,
    )
    keep = [config.date_col, *[c for c in config.id_cols if c in valid.columns]]
    label_cols = list(dict.fromkeys([target, cal_target]))
    out = valid.select([*keep, config.realized_column, *label_cols]).with_columns(
        pl.Series("p_raw", p_raw)
    )
    if calibrator is not None:
        out = out.with_columns(pl.Series("p_cal", calibrator.transform(p_raw)))
    else:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("p_cal"))
    return out, calibrator is not None, fill


def _fold_metric(predictions: pl.DataFrame, config: TrainConfig) -> float:
    """The fold's selection metric: valid log-loss, or Rank IC for M-A."""
    target = config.target_column
    if config.is_classifier:
        scored = predictions.filter(pl.col(target).is_not_null())
        if scored.height == 0:
            return float("nan")
        return binary_log_loss(
            scored.get_column("p_raw").to_numpy(),
            scored.get_column(target).cast(pl.Float64).to_numpy(),
        )
    ic = per_date_rank_ic(
        predictions,
        pred_col="p_raw",
        realized_col=config.realized_column,
        date_col=config.date_col,
    )["rank_ic"].drop_nulls()
    return float(ic.mean()) if ic.len() else float("nan")


def _better(candidate: float, incumbent: float, config: TrainConfig) -> bool:
    """Lower log-loss wins; higher Rank IC wins. NaN never wins."""
    if candidate != candidate:
        return False
    if incumbent != incumbent:
        return True
    return candidate < incumbent if config.is_classifier else candidate > incumbent


def walk_forward(
    panel_std: pl.DataFrame | FoldSource,
    config: TrainConfig | None = None,
    *,
    keep_predictions: bool = True,
) -> TrainResult:
    """Sweep the grid over every fold, then refit and return the winner.

    Takes either a design matrix in memory or a :class:`FoldSource` — a real run
    passes ``DatasetFolds`` so only one fold is resident at a time. Either way
    the fold boundaries come from what ``build_dataset`` computed, so the purge
    and embargo cannot drift from the panel they were cut for.

    The loop is fold-outer / grid-inner: each fold's slices are read once for
    the whole grid instead of once per grid point.
    """
    config = config or TrainConfig()
    source = _as_source(panel_std)
    design = source.design_columns(config.target_column)
    fold_ids = source.fold_ids()

    grid = list(config.grid)
    per_point: list[list[float]] = [[] for _ in grid]
    skipped: dict[int, str] = {}
    sizes: dict[int, tuple[int, int]] = {}

    for fold_id in fold_ids:
        train, valid = source.slices(fold_id)
        sizes[fold_id] = (train.height, valid.height)
        skip = _skip_reason(train, valid, config)
        if skip:
            skipped[fold_id] = skip
            continue
        for index, params in enumerate(grid):
            predictions, _calibrated, _empty = _fit_fold(train, valid, design, config, params)
            per_point[index].append(_fold_metric(predictions, config))
        del train, valid

    grid_metrics: list[dict] = []
    best_params: dict | None = None
    best_metric = float("nan")
    for params, metrics in zip(grid, per_point, strict=True):
        finite = [m for m in metrics if m == m]
        mean = float(np.mean(finite)) if finite else float("nan")
        grid_metrics.append({**params, "mean_metric": mean, "n_folds": len(finite)})
        if _better(mean, best_metric, config):
            best_params, best_metric = params, mean

    if best_params is None:
        raise ValueError(
            "no grid point produced a finite metric on any fold — check the label "
            f"{config.target_column!r} and the fold slices"
        )

    folds: list[FoldFit] = []
    for fold_id in fold_ids:
        n_train, n_valid = sizes[fold_id]
        if fold_id in skipped:
            folds.append(
                FoldFit(
                    fold_id=fold_id,
                    params=best_params,
                    n_train=n_train,
                    n_valid=n_valid,
                    metric=float("nan"),
                    calibrated=False,
                    skipped=skipped[fold_id],
                )
            )
            continue
        train, valid = source.slices(fold_id)
        predictions, calibrated, empty = _fit_fold(train, valid, design, config, best_params)
        folds.append(
            FoldFit(
                fold_id=fold_id,
                params=best_params,
                n_train=n_train,
                n_valid=n_valid,
                metric=_fold_metric(predictions, config),
                calibrated=calibrated,
                empty_design_columns=list(empty),
                predictions=predictions if keep_predictions else None,
            )
        )
        del train, valid

    return TrainResult(
        config=config,
        best_params=best_params,
        best_metric=best_metric,
        grid_metrics=grid_metrics,
        folds=folds,
        design_columns=design,
    )


def _skip_reason(train: pl.DataFrame, valid: pl.DataFrame, config: TrainConfig) -> str | None:
    """Why this fold cannot be fit, or None.

    Reported rather than raised: a thin early fold is a property of the split,
    and the run should record which folds it stands on instead of dying.
    """
    target = config.target_column
    if train.height == 0 or valid.height == 0:
        return "empty slice"
    labelled = _labelled(train, target)
    if labelled.height == 0:
        return "no resolved labels in train"
    if config.is_classifier and labelled.get_column(target).n_unique() < 2:
        return "single class in train"
    return None
