"""preprocess — missing/outlier/scale handling for the linear model (etl_00 §4).

Ridge/ElasticNet impose hard ETL requirements (etl_00 §0): no NaN (L1), scale
sensitivity (L2), fat-tail fragility (L3). This module implements the §4.2/§4.3
recipe over a Polars panel:

  1. add_isna_flags  — one ``<col>_isna`` per feature (L1).
  2. impute          — z-score/ratio/momentum -> 0 (neutral); level -> per-date
                       cross-sectional median (etl_00 §4.2).
  3. per_date_transform — per-date winsorize -> signed-log (configured cols) ->
                       per-date z-score (L2/L3, etl_00 §4.3). "Per-date" =
                       within ``trade_date`` so only the cross-section matters
                       (regime removed), matching the rank label.

Fold-awareness (etl_00 §4.3, §5): winsorize/log/z-score statistics must be fit
on the TRAIN range only and applied to valid/test. Because the standardization
is per-date (within a single day) there is little cross-fold leakage, but the
winsorize quantiles are global-per-column, so :func:`fit_winsor_bounds` is fit on
the train slice and passed to :func:`per_date_transform` for all slices.

``profile`` toggles model-family behavior (00_shared §3.3): ``"linear"`` does the
full impute+standardize; ``"tree"`` keeps NaNs (native) and skips z-score;
``"rank"`` (20260907_model_experiment `02` §4.1) replaces each feature with its
percentile inside ``(trade_date, market)``.

The rank profile has no fitted statistics at all — a percentile is computed
within one day's cross-section, so there is no train-range quantity to leak
across a fold boundary. That is why it exists: the probability model feeds the
same ranks to a tree, a logistic regression and the interaction builder, and
none of the three can see the future through its preprocessing.

See ``etl_00`` §4.2, §4.3, ``00_shared`` §3.3, ``etl_03_implementation_plan.md`` §4 (P5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

KEY_COLS = ("trade_date", "ticker", "market")

_VALID_PROFILES = ("linear", "tree", "rank")

# Profiles with nothing fitted on the train slice. A stateless profile means a
# row transforms to the same value in every fold, so a per-fold copy of the
# design matrix is redundant — which is what lets a build store one transformed
# panel instead of one per fold (the rank percentile is taken within
# ``(trade_date, market)``, and a fold slice is whole dates, so the group a row
# is ranked against is the same either way).
STATELESS_PROFILES: tuple[str, ...] = ("rank",)

# Columns imputed to 0 (neutral) vs per-date median. Heuristic by name suffix/
# content: ratios, momentum, z-scores, returns center at 0; absolute levels
# (turnover, holding shares, balances, prices) use the per-date median.
_ZERO_IMPUTE_HINTS = ("_z_", "ret_", "_chg_", "mom_", "_ratio", "rank", "netbuy")
_MEDIAN_IMPUTE_HINTS = ("turnover", "balance", "holding", "amihud", "avg_price", "volume", "value")

# Binary features, by name. Ranking a 0/1 column inside a cross-section turns
# "halted" into "how many others were halted today", which is a different
# variable — so the rank profile passes these through untouched. Name-based
# because a mart may deliver them as Int8 rather than Boolean; the regime
# columns (``rg_*``, 02 §4.1) are the ones that actually arrive that way.
_FLAG_FEATURE_NAMES: tuple[str, ...] = (
    "px_is_halted",
    "fin_is_negative_equity",
    "fin_has_fs",
)
_FLAG_FEATURE_PREFIXES: tuple[str, ...] = ("rg_",)

# Already a percentile when it reaches preprocessing: ``ix_* = pct_rank(x) x
# regime`` (`02` §3.2). Ranking it again would map a regime-off session's whole
# cross-section of zeros onto 0.5 — the middle of the scale — and the "0 means
# the regime was off" marker the tree splits on would be gone.
_PRERANKED_FEATURE_PREFIXES: tuple[str, ...] = ("ix_",)


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing knobs (etl_00 §4)."""

    profile: str = "linear"  # "linear" | "tree" | "rank"
    winsor: tuple[float, float] = (0.01, 0.99)  # per-column quantile clip (etl_00 §4.3)
    signed_log_cols: tuple[str, ...] = ()  # heavy-tailed levels to signed-log
    zero_impute_extra: tuple[str, ...] = ()
    median_impute_extra: tuple[str, ...] = ()
    # profile="rank" only: what a missing percentile becomes. 0.5 (the neutral
    # middle of the cross-section) is the default because a logistic regression
    # cannot take a NaN; pass None for a tree, which reads the gap natively.
    # Either way the ``*_isna`` flag carries the fact that it was missing.
    rank_null_fill: float | None = 0.5

    def __post_init__(self) -> None:
        if self.profile not in _VALID_PROFILES:
            raise ValueError(f"profile must be one of {_VALID_PROFILES}, got {self.profile!r}")
        lo, hi = self.winsor
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"winsor must be 0<=lo<hi<=1, got {self.winsor}")


def is_stateless(profile: str | PreprocessConfig) -> bool:
    """True if the profile fits nothing on the train slice (see :data:`STATELESS_PROFILES`)."""
    name = profile.profile if isinstance(profile, PreprocessConfig) else profile
    return name in STATELESS_PROFILES


def feature_columns(panel: pl.DataFrame, *, extra_exclude: tuple[str, ...] = ()) -> list[str]:
    """Numeric feature columns: everything except keys, labels, and excludes."""
    exclude = set(KEY_COLS) | set(extra_exclude)
    cols = []
    for name, dtype in zip(panel.columns, panel.dtypes):
        if name in exclude or name.startswith("y_") or name.startswith("raw_label"):
            continue
        if name.startswith("fwd_ret_") or name.startswith("bench_ret_"):
            continue
        if dtype.is_numeric() or dtype == pl.Boolean:
            cols.append(name)
    return cols


def add_isna_flags(panel: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Append a ``<col>_isna`` Int8 flag for each feature (L1, etl_00 §4.2)."""
    flags = [panel[c].is_null().cast(pl.Int8).alias(f"{c}_isna") for c in feature_cols]
    return panel.with_columns(flags)


def is_flag_feature(name: str, dtype: pl.DataType | None = None) -> bool:
    """True for a 0/1 feature: Boolean dtype, a known flag name, or ``rg_*``.

    ``*_isna`` flags are flags too — :func:`add_isna_flags` writes them as Int8
    and nothing downstream should rank or standardize them.
    """
    if dtype == pl.Boolean:
        return True
    if name in _FLAG_FEATURE_NAMES or name.endswith("_isna"):
        return True
    return name.startswith(_FLAG_FEATURE_PREFIXES)


def _cast_bool_features(panel: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Cast Boolean feature columns to Float64 so numeric ops (impute/zscore) work."""
    casts = [
        pl.col(c).cast(pl.Float64).alias(c)
        for c in feature_cols
        if is_flag_feature(c, panel.schema.get(c)) and panel.schema.get(c) != pl.Float64
    ]
    return panel.with_columns(casts) if casts else panel


def _impute_kind(col: str, cfg: PreprocessConfig) -> str:
    """Return 'zero' or 'median' for a feature column (etl_00 §4.2)."""
    if col in cfg.zero_impute_extra:
        return "zero"
    if col in cfg.median_impute_extra:
        return "median"
    lowered = col.lower()
    if any(h in lowered for h in _MEDIAN_IMPUTE_HINTS):
        return "median"
    if any(h in lowered for h in _ZERO_IMPUTE_HINTS):
        return "zero"
    return "zero"  # default neutral


def impute(panel: pl.DataFrame, feature_cols: list[str], cfg: PreprocessConfig) -> pl.DataFrame:
    """Fill nulls: zero-impute or group-based median (etl_00 §4.2).

    Uses ``(trade_date, market)`` as the grouping for median imputation to
    better capture sector/market-specific neutral levels.
    """
    zero_cols = [c for c in feature_cols if _impute_kind(c, cfg) == "zero"]
    median_cols = [c for c in feature_cols if _impute_kind(c, cfg) == "median"]

    exprs = []
    for c in zero_cols:
        exprs.append(pl.col(c).fill_null(0.0).alias(c))

    # Identify grouping columns. 'market' is used if present in the panel.
    group_cols = ["trade_date"]
    if "market" in panel.columns:
        group_cols.append("market")

    for c in median_cols:
        # per-group cross-sectional median, then any residual null -> 0.
        med = pl.col(c).median().over(group_cols)
        exprs.append(pl.col(c).fill_null(med).fill_null(0.0).alias(c))
    return panel.with_columns(exprs)


def fit_winsor_bounds(
    train_panel: pl.DataFrame, feature_cols: list[str], cfg: PreprocessConfig
) -> dict[str, tuple[float, float]]:
    """Fit per-column winsorize bounds on the TRAIN slice only (etl_00 §4.3, §5)."""
    lo_q, hi_q = cfg.winsor
    bounds: dict[str, tuple[float, float]] = {}
    for c in feature_cols:
        col = train_panel[c]
        if not col.dtype.is_numeric():
            continue
        lo = col.quantile(lo_q)
        hi = col.quantile(hi_q)
        if lo is not None and hi is not None:
            bounds[c] = (float(lo), float(hi))
    return bounds


def _signed_log(expr: pl.Expr) -> pl.Expr:
    """signed-log: sign(x) * ln(1+|x|) — tames heavy tails, keeps sign (etl_00 §4.3)."""
    return expr.sign() * (expr.abs() + 1.0).log()


def per_date_transform(
    panel: pl.DataFrame,
    feature_cols: list[str],
    cfg: PreprocessConfig,
    bounds: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    """Winsorize (fitted bounds) -> signed-log (configured) -> per-date z-score.

    Order matters (etl_00 §4.3). For ``profile="tree"`` the z-score step is
    skipped (trees are scale-invariant) but winsorize/log still run.
    """
    # 1) winsorize using train-fitted bounds (clip both tails).
    clip_exprs = []
    for c in feature_cols:
        if c in bounds:
            lo, hi = bounds[c]
            clip_exprs.append(pl.col(c).clip(lo, hi).alias(c))
    panel = panel.with_columns(clip_exprs) if clip_exprs else panel

    # 2) signed-log for configured heavy-tailed level columns.
    if cfg.signed_log_cols:
        log_exprs = [
            _signed_log(pl.col(c)).alias(c) for c in cfg.signed_log_cols if c in feature_cols
        ]
        panel = panel.with_columns(log_exprs) if log_exprs else panel

    # 3) per-date z-score (cross-sectional standardize). Skipped for trees.
    if cfg.profile == "linear":
        z_exprs = []
        for c in feature_cols:
            mean = pl.col(c).mean().over("trade_date")
            std = pl.col(c).std().over("trade_date")
            # std==0 (or null) -> 0 so a constant cross-section becomes neutral.
            z = pl.when(std.is_null() | (std == 0)).then(0.0).otherwise((pl.col(c) - mean) / std)
            z_exprs.append(z.alias(c))
        panel = panel.with_columns(z_exprs)

    return panel


def rankable_columns(panel: pl.DataFrame, feature_cols: list[str]) -> list[str]:
    """Feature columns the rank transform touches — numeric, non-flag ones."""
    out: list[str] = []
    for c in feature_cols:
        dtype = panel.schema.get(c)
        if dtype is None or not dtype.is_numeric():
            continue
        if is_flag_feature(c, dtype) or c.startswith(_PRERANKED_FEATURE_PREFIXES):
            continue
        out.append(c)
    return out


def per_date_rank_expr(col: str, group_cols: list[str]) -> pl.Expr:
    """The percentile of ``col`` inside ``group_cols`` — the one definition.

    ``(rank_avg - 1) / (n_valid - 1)`` over the group's non-null values; a
    one-name group is 0.5; NULL stays NULL. The rank profile and the
    interaction builder (`02` §3.2 "두 곳에서 다른 순위 정의가 생기면 안 된다")
    both go through here.
    """
    n_valid = pl.col(col).count().over(group_cols)
    rank = pl.col(col).rank("average").over(group_cols)
    return (
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(n_valid > 1)
        .then((rank - 1.0) / (n_valid - 1.0))
        .otherwise(0.5)
        .cast(pl.Float64)
    )


def per_date_rank_transform(
    panel: pl.DataFrame,
    feature_cols: list[str],
    *,
    group_cols: tuple[str, ...] = ("trade_date", "market"),
    null_fill: float | None = None,
) -> pl.DataFrame:
    """Replace each feature with its percentile in ``(trade_date, market)``.

    ``(rank_avg - 1) / (n_valid - 1)`` over the non-null values of the group, so
    distinct values span [0, 1] exactly and ties share the average — the same
    definition the interaction builder uses (`02` §3.2), which is why it lives
    here as one function rather than in two places.

    Note this is *average* ties, unlike the label's ``y_rank`` (DuckDB
    ``PERCENT_RANK``, minimum ties). A count feature with a long flat middle —
    ``ev_filing_burst_60d`` — would otherwise push every tied name to the
    bottom of its own tie group.

    A one-name cross-section becomes 0.5: there is no ordering to read, and
    dropping the row is the universe filter's job, not preprocessing's.
    Nulls stay null unless ``null_fill`` is given.
    """
    group = [c for c in group_cols if c in panel.columns]
    if not group:
        raise ValueError(f"none of {group_cols} present in panel columns")
    cols = rankable_columns(panel, feature_cols)
    exprs = []
    for c in cols:
        pct = per_date_rank_expr(c, group)
        if null_fill is not None:
            pct = pct.fill_null(null_fill)
        exprs.append(pct.alias(c))
    return panel.with_columns(exprs) if exprs else panel


def assert_finite(panel: pl.DataFrame, feature_cols: list[str]) -> None:
    """Assert no NaN/null/inf remain in feature columns (L1 gate, etl_00 §4.2)."""
    for c in feature_cols:
        s = panel[c]
        n_null = s.null_count()
        n_nan = int(s.is_nan().sum()) if s.dtype.is_float() else 0
        n_inf = int(s.is_infinite().sum()) if s.dtype.is_float() else 0
        if n_null or n_nan or n_inf:
            raise AssertionError(
                f"feature {c!r} has non-finite values: null={n_null} nan={n_nan} inf={n_inf}"
            )


@dataclass
class FittedPreprocess:
    """Statistics fit on train, applied to any slice (fold-aware, etl_00 §4.3)."""

    feature_cols: list[str]
    cfg: PreprocessConfig
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def transform(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Apply isna -> cast-bool -> impute -> winsor/log/zscore to a panel slice.

        The ``rank`` profile takes a different, stateless route: isna flags,
        bool cast, then the per-(date, market) percentile. No winsorize (a
        percentile has no tails), no z-score (it is already bounded), no
        median impute (``rank_null_fill`` decides).
        """
        out = add_isna_flags(panel, self.feature_cols)
        # Boolean features (e.g. px_is_halted) -> Float64 so numeric ops apply.
        out = _cast_bool_features(out, self.feature_cols)
        if self.cfg.profile == "rank":
            return per_date_rank_transform(
                out, self.feature_cols, null_fill=self.cfg.rank_null_fill
            )
        if self.cfg.profile == "linear":
            out = impute(out, self.feature_cols, self.cfg)
        out = per_date_transform(out, self.feature_cols, self.cfg, self.bounds)
        return out


def fit(
    train_panel: pl.DataFrame, cfg: PreprocessConfig, *, extra_exclude: tuple[str, ...] = ()
) -> FittedPreprocess:
    """Fit preprocessing on the train slice (winsor bounds). Returns a transformer.

    The cross-sectional impute median and z-score mean/std are computed per-date
    at transform time (within-day), so only the winsor bounds need fitting here.
    ``profile="rank"`` fits nothing — the returned transformer carries no
    statistics, so the train slice it was handed cannot influence any other.
    """
    feats = feature_columns(train_panel, extra_exclude=extra_exclude)
    if cfg.profile == "rank":
        return FittedPreprocess(feature_cols=feats, cfg=cfg, bounds={})
    bounds = fit_winsor_bounds(train_panel, feats, cfg)
    return FittedPreprocess(feature_cols=feats, cfg=cfg, bounds=bounds)
