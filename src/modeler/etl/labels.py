# ruff: noqa: E501
"""make_label — the single, parameterized label generator (00_shared §3.1).

Model count == label count, so there is ONE generator. The 20d excess-return
rank (etl_00 §2) is just the ``horizon=20, kind="excess", bench="eqw_market",
outputs=("reg","rank","cls")`` instance; 5d/60d and volatility/MDD are other
parameter sets sharing this code.

Mechanics (etl_00 §2):
  - Forward return uses a per-ticker trading-day index that EXCLUDES halt days
    (``open=high=low=0``), so ``t+H`` is exactly H real sessions later and halt/
    holiday gaps are absorbed (etl_00 §2.1; §9 checklist "거래일 인덱스 기준").
  - Benchmark (``eqw_market``): equal-weighted mean forward return within the
    same ``(trade_date, market)`` — robust, PIT-safe, no external index needed.
    ``excess = fwd - bench``; per (date, market) the mean excess is ~0.
    ``bench="index"`` is reserved for a future market-index swap (one-line, the
    prediction-target doc §1 note) and currently raises NotImplementedError.
  - Outputs (etl_00 §2.2): per-date winsorized regression (``y_reg_*``), per-date
    percentile rank in [0,1] (``y_rank_*`` — the main target), and a 3-class
    label (``y_cls_*`` ∈ {-1,0,1}) thresholded at the 0.2/0.8 ranks.
  - Binary outputs for the probability model (20260907_model_experiment `01` §3):
    ``y_up_*`` = 1[raw_label > 0] (L-A, "beat the market") and ``y_top_*`` =
    1[y_rank >= cls_top] (L-B, "land in the top 20%"). ``y_top`` carries the same
    information as ``y_cls == 1``; it exists so a classifier never has to re-cut
    a 3-class column. Neither is in the default ``outputs`` — model 01's label
    SQL is unchanged by their addition.
  - Auxiliary (etl_00 §2.3): realized volatility / max-drawdown of the t+1..t+H
    daily-return path, produced by :func:`build_risk_label_sql`.

All ranking/winsorizing is WITHIN ``(trade_date, market)`` so regime/market-
direction is removed and the label matches per-date cross-sectional features
(etl_00 §2.2, §4.3).

See ``etl_00`` §2, ``00_shared`` §3.1, and ``etl_03_implementation_plan.md`` §4 (P4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

LABEL_TABLE = "label_daily"
LABEL_SCAN_TABLE = "label_scan"

_VALID_KINDS = ("excess", "abs")
_VALID_BENCH = ("eqw_market", "index")
_VALID_OUTPUTS = ("reg", "rank", "cls", "up", "top")


@dataclass(frozen=True)
class LabelSpec:
    """Parameters for one label instance (00_shared §3.1).

    ``horizons`` is the set of forward windows to emit (the first is the
    "primary"); column suffixes are ``_{h}d``. ``cls_top``/``cls_bottom`` are the
    rank thresholds for the 3-class label.
    """

    horizons: Sequence[int] = (20, 5, 60)
    kind: str = "excess"
    bench: str = "eqw_market"
    outputs: Sequence[str] = ("reg", "rank", "cls")
    winsor: tuple[float, float] = (0.005, 0.995)
    cls_top: float = 0.8
    cls_bottom: float = 0.2
    # Risk labels (y_vol/y_mdd, etl_00 §2.3) are a *separate* regression target.
    # Off by default: the MDD path is expensive and the return-rank model does
    # not use them. Set include_risk=True to emit them into label_daily.
    risk_horizons: Sequence[int] = field(default_factory=lambda: (20,))
    include_risk: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")
        if self.bench not in _VALID_BENCH:
            raise ValueError(f"bench must be one of {_VALID_BENCH}, got {self.bench!r}")
        bad = set(self.outputs) - set(_VALID_OUTPUTS)
        if bad:
            raise ValueError(f"unknown outputs {bad}; valid: {_VALID_OUTPUTS}")
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        lo, hi = self.winsor
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"winsor must be 0<=lo<hi<=1, got {self.winsor}")


def _forward_cte(price_view: str) -> str:
    """CTE ``px``: per-ticker non-halt trading-day index over a price view."""
    return f"""
        px AS (
            SELECT trade_date, ticker, market,
                   CAST(close AS DOUBLE) AS close_d,
                   ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS d_idx
            FROM {price_view}
            WHERE NOT (open = 0 AND high = 0 AND low = 0)   -- exclude halt-day close distortion
        )
    """


def build_label_sql(spec: LabelSpec, price_view: str = "daily_ohlcv") -> str:
    """SQL producing ``label_daily`` (return labels) from a price view.

    ``price_view`` must already be registered. Grain: (trade_date, ticker,
    market). Emits, per horizon h: ``fwd_ret_{h}d``, optionally ``bench_ret_{h}d``,
    ``raw_label_{h}d`` (the regression source), and the requested outputs
    ``y_reg_{h}d`` / ``y_rank_{h}d`` / ``y_cls_{h}d``.
    """
    if spec.bench == "index":
        raise NotImplementedError(
            "bench='index' not implemented; use 'eqw_market' (prediction_target §1). "
            "Swapping to KOSPI/KOSDAQ index returns is a future one-line change."
        )

    px = _forward_cte(price_view)
    primary = spec.horizons[0]

    # One forward-return CTE per horizon, plus an equal-weight benchmark CTE.
    fwd_ctes: list[str] = []
    bench_ctes: list[str] = []
    for h in spec.horizons:
        fwd_ctes.append(f"""
        fwd{h} AS (
            SELECT a.trade_date, a.ticker, a.market,
                   f.close_d / NULLIF(a.close_d, 0) - 1 AS fwd_ret_{h}d
            FROM px a
            JOIN px f
              ON f.ticker = a.ticker AND f.market = a.market
             AND f.d_idx = a.d_idx + {h}
        )""")
        if spec.kind == "excess":
            bench_ctes.append(f"""
        bench{h} AS (
            SELECT trade_date, market, AVG(fwd_ret_{h}d) AS bench_ret_{h}d
            FROM fwd{h} GROUP BY trade_date, market
        )""")

    # fwd{primary} is the base row set; other horizons/benchmarks LEFT JOIN on.
    all_ctes = ",\n".join([px] + fwd_ctes + bench_ctes)
    select_cols = _build_select_cols(spec)
    risk_join = ""
    if spec.include_risk:
        # risk labels (y_vol/y_mdd) joined on the full key; NULL where no path.
        risk_cols = ", ".join(f"risk.y_vol_{h}d, risk.y_mdd_{h}d" for h in spec.risk_horizons)
        select_cols += ",\n            " + risk_cols
        risk_sql = build_risk_label_sql(spec, price_view)
        risk_join = f"\n        LEFT JOIN ({risk_sql}) AS risk USING (trade_date, ticker, market)"
    return f"""
        WITH {all_ctes}
        SELECT
            {select_cols}
        FROM fwd{primary} AS base{_extra_joins(spec, primary)}{risk_join}
    """


def _extra_joins(spec: LabelSpec, primary: int) -> str:
    """JOIN clauses for non-primary horizons + benchmarks (base is fwd{primary})."""
    parts: list[str] = []
    if spec.kind == "excess":
        parts.append(f"JOIN bench{primary} AS m{primary} USING (trade_date, market)")
    for h in spec.horizons:
        if h == primary:
            continue
        parts.append(f"LEFT JOIN fwd{h} AS f{h} USING (trade_date, ticker, market)")
        if spec.kind == "excess":
            parts.append(f"LEFT JOIN bench{h} AS m{h} USING (trade_date, market)")
    if not parts:
        return ""
    return "\n        " + "\n        ".join(parts)


def _build_select_cols(spec: LabelSpec) -> str:
    """Build the final SELECT column list (raw labels + requested outputs)."""
    primary = spec.horizons[0]
    cols: list[str] = ["base.trade_date", "base.ticker", "base.market"]
    lo, hi = spec.winsor

    for h in spec.horizons:
        fref = "base" if h == primary else f"f{h}"
        cols.append(f"{fref}.fwd_ret_{h}d")
        if spec.kind == "excess":
            cols.append(f"m{h}.bench_ret_{h}d")
            raw = f"({fref}.fwd_ret_{h}d - m{h}.bench_ret_{h}d)"
        else:
            raw = f"{fref}.fwd_ret_{h}d"
        cols.append(f"{raw} AS raw_label_{h}d")

        # per-date (within trade_date, market) ranking window.
        # NOTE: PERCENT_RANK() assigns a value to rows whose ORDER BY key is NULL
        # (they sort last), so a horizon with no forward return (e.g. the last H
        # sessions, or a secondary horizon beyond the panel) would get a bogus
        # rank/class. Guard every output with ``WHEN raw IS NULL THEN NULL`` so a
        # missing forward label stays NULL across reg/rank/cls.
        win = "PARTITION BY base.trade_date, base.market"
        null_guard = f"{raw} IS NOT NULL"
        if "rank" in spec.outputs:
            cols.append(
                f"CASE WHEN {null_guard} "
                f"THEN PERCENT_RANK() OVER ({win} ORDER BY {raw}) END AS y_rank_{h}d"
            )
        if "reg" in spec.outputs:
            # per-date winsorize via quantile clip. The same null guard as
            # rank/cls is required: DuckDB's GREATEST/LEAST *skip* NULL
            # arguments, so LEAST(GREATEST(NULL, p_lo), p_hi) returns p_lo — a
            # missing forward return would silently become the bottom clip
            # instead of staying NULL.
            cols.append(
                f"CASE WHEN {null_guard} THEN "
                f"LEAST(GREATEST({raw}, "
                f"QUANTILE_CONT({raw}, {lo}) OVER ({win})), "
                f"QUANTILE_CONT({raw}, {hi}) OVER ({win})) END AS y_reg_{h}d"
            )
        if "cls" in spec.outputs:
            r = f"PERCENT_RANK() OVER ({win} ORDER BY {raw})"
            cols.append(
                f"CASE WHEN {null_guard} THEN "
                f"(CASE WHEN {r} >= {spec.cls_top} THEN 1 "
                f"WHEN {r} <= {spec.cls_bottom} THEN -1 ELSE 0 END) END AS y_cls_{h}d"
            )
        if "up" in spec.outputs:
            # L-A. ``raw = 0`` is a 0: an excess return of exactly zero did not
            # beat the market. Measured to be all but absent in this panel.
            cols.append(f"CASE WHEN {null_guard} THEN CAST({raw} > 0 AS TINYINT) END AS y_up_{h}d")
        if "top" in spec.outputs:
            # L-B, the same cut as ``y_cls == 1`` — kept as its own column so a
            # binary model reads a binary label.
            r = f"PERCENT_RANK() OVER ({win} ORDER BY {raw})"
            cols.append(
                f"CASE WHEN {null_guard} "
                f"THEN CAST({r} >= {spec.cls_top} AS TINYINT) END AS y_top_{h}d"
            )
    return ",\n            ".join(cols)


def build_risk_label_sql(spec: LabelSpec, price_view: str = "daily_ohlcv") -> str:
    """SQL for realized-volatility / max-drawdown labels (etl_00 §2.3).

    For each (t, ticker, market) and horizon h in ``risk_horizons``, over the
    t+1..t+h daily-return path:
      - ``y_vol_{h}d``: stddev of daily log returns (realized volatility)
      - ``y_mdd_{h}d``: max drawdown of the cumulative return path (<= 0)

    Implemented with forward window frames + a single self-join aggregation per
    horizon (not a per-row correlated subquery), so it scales to the full panel.
    Uses the same non-halt d_idx as the return labels.
    """
    px = _forward_cte(price_view)
    rets = """
        rets AS (
            SELECT trade_date, ticker, market, d_idx,
                   ln(close_d / NULLIF(LAG(close_d) OVER (
                       PARTITION BY ticker, market ORDER BY d_idx), 0)) AS r1
            FROM px
        )
    """
    # Forward realized vol per horizon via a FOLLOWING window frame on rets.
    vol_cols = []
    for h in spec.risk_horizons:
        vol_cols.append(
            f"STDDEV_SAMP(r1) OVER (PARTITION BY ticker, market ORDER BY d_idx "
            f"ROWS BETWEEN 1 FOLLOWING AND {h} FOLLOWING) AS y_vol_{h}d"
        )
    vol_cte = f"""
        vol AS (
            SELECT trade_date, ticker, market, d_idx,
                   {", ".join(vol_cols)}
            FROM rets
        )
    """
    # MDD per horizon: join each base row to its forward window rows, build the
    # running cum-sum / peak per (base, horizon), and take the min drawdown.
    # One pass per horizon over the (base x window) pairs — no per-row subquery.
    mdd_ctes = []
    mdd_joins = []
    for h in spec.risk_horizons:
        mdd_ctes.append(f"""
        mdd{h} AS (
            SELECT base_td AS trade_date, ticker, market,
                   MIN(cum - run_max) AS y_mdd_{h}d
            FROM (
                SELECT base_td, ticker, market, w_d_idx,
                       cum,
                       MAX(cum) OVER (PARTITION BY base_td, ticker, market
                                      ORDER BY w_d_idx) AS run_max
                FROM (
                    SELECT a.trade_date AS base_td, a.ticker, a.market, b.d_idx AS w_d_idx,
                           SUM(b.r1) OVER (PARTITION BY a.trade_date, a.ticker, a.market
                                           ORDER BY b.d_idx) AS cum
                    FROM rets a
                    JOIN rets b
                      ON b.ticker = a.ticker AND b.market = a.market
                     AND b.d_idx > a.d_idx AND b.d_idx <= a.d_idx + {h}
                ) s
            ) p
            GROUP BY base_td, ticker, market
        )""")
        mdd_joins.append(f"LEFT JOIN mdd{h} USING (trade_date, ticker, market)")

    all_ctes = ",\n".join([px, rets, vol_cte] + mdd_ctes)
    mdd_select = "".join(f", mdd{h}.y_mdd_{h}d" for h in spec.risk_horizons)
    return f"""
        WITH {all_ctes}
        SELECT vol.trade_date, vol.ticker, vol.market,
               {", ".join(f"vol.y_vol_{h}d" for h in spec.risk_horizons)}{mdd_select}
        FROM vol
        {" ".join(mdd_joins)}
    """


def materialize_label(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    spec: LabelSpec | None = None,
    *,
    price_view: str = "daily_ohlcv",
    force: bool = False,
) -> str:
    """Build + register ``label_daily`` mart view. Returns the view name.

    Requires ``price_view`` registered on ``con``.
    """
    spec = spec or LabelSpec()
    materialize(con, config, LABEL_TABLE, build_label_sql(spec, price_view), force=force)
    return register_mart_view(con, config, LABEL_TABLE)


def build_label_scan_sql(
    price_view: str = "daily_ohlcv",
    *,
    quality_view: str | None = None,
    holdout_start: str = "2025-08-01",
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20, 40, 60, 120),
    buckets: Sequence[tuple[int, int]] = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 120)),
) -> str:
    """Build the preregistered scan labels without changing ``label_daily``.

    The raw forward endpoint and CA count are retained internally so a bucket
    can be valid even when an earlier cumulative horizon was masked. Benchmarks
    and ranks are then computed from the same remaining population.
    """
    hs = tuple(horizons)
    if not hs or any(h <= 0 for h in hs):
        raise ValueError("label scan horizons must be positive")
    q_join = (
        f"LEFT JOIN {quality_view} q USING (trade_date, ticker, market)"
        if quality_view
        else ""
    )
    event_expr = "COALESCE(q.ca_event, FALSE)" if quality_view else "FALSE"
    px = f"""
        px0 AS (
            SELECT p.trade_date, p.ticker, p.market,
                   CAST(p.close AS DOUBLE) AS close_d,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.ticker, p.market ORDER BY p.trade_date
                   ) AS d_idx,
                   SUM(CASE WHEN {event_expr} THEN 1 ELSE 0 END) OVER (
                       PARTITION BY p.ticker, p.market ORDER BY p.trade_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS ca_cum
            FROM {price_view} p {q_join}
            WHERE NOT (p.open = 0 AND p.high = 0 AND p.low = 0)
        )"""
    fwd_ctes = []
    for h in hs:
        fwd_ctes.append(f"""
        fwd_{h} AS (
            SELECT a.trade_date, a.ticker, a.market,
                   f.trade_date AS label_end_date_{h}d,
                   f.close_d / NULLIF(a.close_d, 0) - 1 AS gross_ret_{h}d,
                   f.ca_cum - a.ca_cum AS ca_count_{h}d
            FROM px0 a
            LEFT JOIN px0 f
              ON f.ticker = a.ticker AND f.market = a.market
             AND f.d_idx = a.d_idx + {h}
        )""")
    bench_ctes = []
    for h in hs:
        bench_ctes.append(f"""
        bench_{h} AS (
            SELECT trade_date, market, AVG(fwd_ret_{h}d) AS bench_ret_{h}d
            FROM raw_forward GROUP BY trade_date, market
        )""")
    # The unmasked endpoint columns are needed for bucket-specific CA masks.
    raw_forward = f"""
        raw_forward AS (
            SELECT a.trade_date, a.ticker, a.market, a.close_d,
                   {', '.join(
                       f"CASE WHEN f{h}.ca_count_{h}d = 0 THEN "
                       f"f{h}.gross_ret_{h}d END AS fwd_ret_{h}d, "
                       f"f{h}.gross_ret_{h}d, f{h}.label_end_date_{h}d, f{h}.ca_count_{h}d"
                       for h in hs
                   )}
            FROM px0 a
            {''.join(f'LEFT JOIN fwd_{h} f{h} USING (trade_date, ticker, market) ' for h in hs)}
        )"""
    label_select: list[str] = []
    for h in hs:
        raw = f"(r.fwd_ret_{h}d - b{h}.bench_ret_{h}d)"
        label_select.extend([
            f"r.fwd_ret_{h}d, r.label_end_date_{h}d, b{h}.bench_ret_{h}d, "
            f"{raw} AS raw_label_{h}d",
            f"(r.label_end_date_{h}d < DATE '{holdout_start}' AND {raw} IS NOT NULL) "
            f"AS label_ok_{h}d",
        ])
    bucket_ctes: list[str] = []
    bucket_bench: list[str] = []
    bucket_select: list[str] = []
    for h1, h2 in buckets:
        if h2 not in hs or (h1 and h1 not in hs):
            raise ValueError(f"bucket ({h1},{h2}] requires horizons in the scan grid")
        gross1 = "0.0" if h1 == 0 else f"r.gross_ret_{h1}d"
        ca1 = "0" if h1 == 0 else f"COALESCE(r.ca_count_{h1}d, 0)"
        bucket_ctes.append(f"""
        bucket_{h1}_{h2} AS (
            SELECT r.trade_date, r.ticker, r.market,
                   r.label_end_date_{h2}d AS bucket_end_date_{h1}_{h2}d,
                   CASE WHEN r.ca_count_{h2}d - {ca1} = 0
                        THEN (1 + r.gross_ret_{h2}d) / (1 + {gross1}) - 1 END
                        AS bucket_ret_{h1}_{h2}d
            FROM raw_forward r
        )""")
        bucket_bench.append(f"""
        bucket_bench_{h1}_{h2} AS (
            SELECT trade_date, market, AVG(bucket_ret_{h1}_{h2}d) AS bucket_bench_ret_{h1}_{h2}d
            FROM bucket_{h1}_{h2} GROUP BY trade_date, market
        )""")
        br = f"(bk{h1}_{h2}.bucket_ret_{h1}_{h2}d - bb{h1}_{h2}.bucket_bench_ret_{h1}_{h2}d)"
        bucket_select.extend([
            f"bk{h1}_{h2}.bucket_ret_{h1}_{h2}d, "
            f"bb{h1}_{h2}.bucket_bench_ret_{h1}_{h2}d AS bucket_bench_ret_{h1}_{h2}d, "
            f"{br} AS raw_bucket_label_{h1}_{h2}d, "
            f"bk{h1}_{h2}.bucket_end_date_{h1}_{h2}d",
        ])
    bucket_joins = " ".join(
        f"LEFT JOIN bucket_{h1}_{h2} bk{h1}_{h2} USING (trade_date, ticker, market) "
        f"LEFT JOIN bucket_bench_{h1}_{h2} bb{h1}_{h2} USING (trade_date, market)"
        for h1, h2 in buckets
    )
    survivor_expr = (
        f"(f.common_formation_120d AND r.label_end_date_120d IS NOT NULL "
        f"AND r.label_end_date_120d < DATE '{holdout_start}')"
        if 120 in hs
        else "FALSE"
    )
    joined_bench = ", ".join(
        f"b{h}.bench_ret_{h}d AS bench_ret_{h}d" for h in hs
    )
    joined_buckets = ", ".join(
        f"bk{h1}_{h2}.*, bb{h1}_{h2}.bucket_bench_ret_{h1}_{h2}d "
        f"AS bucket_bench_ret_{h1}_{h2}d"
        for h1, h2 in buckets
    )
    rank_ctes = ", ".join(
        f"rank_{h} AS ("
        f"SELECT trade_date, ticker, market, "
        f"PERCENT_RANK() OVER (PARTITION BY trade_date, market ORDER BY raw_label_{h}d) "
        f"AS y_rank_{h}d FROM joined WHERE raw_label_{h}d IS NOT NULL)"
        for h in hs
    )
    bucket_rank_ctes = ", ".join(
        f"bucket_rank_{h1}_{h2} AS ("
        f"SELECT trade_date, ticker, market, "
        f"PERCENT_RANK() OVER (PARTITION BY trade_date, market ORDER BY raw_bucket_label_{h1}_{h2}d) "
        f"AS y_rank_bucket_{h1}_{h2}d "
        f"FROM joined WHERE raw_bucket_label_{h1}_{h2}d IS NOT NULL)"
        for h1, h2 in buckets
    )
    joined_raw_labels = ", ".join(
        f"(r.fwd_ret_{h}d - b{h}.bench_ret_{h}d) AS raw_label_{h}d" for h in hs
    )
    joined_raw_buckets = ", ".join(
        f"(bk{h1}_{h2}.bucket_ret_{h1}_{h2}d - "
        f"bb{h1}_{h2}.bucket_bench_ret_{h1}_{h2}d) AS raw_bucket_label_{h1}_{h2}d"
        for h1, h2 in buckets
    )
    label_select_j = [item for h in hs for item in [
                          f"j.fwd_ret_{h}d, j.label_end_date_{h}d, j.bench_ret_{h}d, "
                          f"(j.fwd_ret_{h}d - j.bench_ret_{h}d) AS raw_label_{h}d",
                          f"(j.label_end_date_{h}d < DATE '{holdout_start}' AND "
                          f"(j.fwd_ret_{h}d - j.bench_ret_{h}d) IS NOT NULL) AS label_ok_{h}d",
                      ]]
    bucket_select_j = [item for h1, h2 in buckets for item in [
            f"j.bucket_ret_{h1}_{h2}d, j.bucket_bench_ret_{h1}_{h2}d, "
            f"(j.bucket_ret_{h1}_{h2}d - j.bucket_bench_ret_{h1}_{h2}d) "
            f"AS raw_bucket_label_{h1}_{h2}d, j.bucket_end_date_{h1}_{h2}d"
        ]]
    bucket_coverage_j = ", ".join(
        f"(j.bucket_end_date_{h1}_{h2}d < DATE '{holdout_start}' AND "
        f"j.bucket_ret_{h1}_{h2}d IS NOT NULL) AS bucket_ok_{h1}_{h2}d"
        for h1, h2 in buckets
    )
    return f"""
        WITH {px}, {', '.join(fwd_ctes)}, {raw_forward},
        {', '.join(bench_ctes)},
        {', '.join(bucket_ctes)},
        {', '.join(bucket_bench)},
        calendar AS (
            SELECT trade_date, ROW_NUMBER() OVER (ORDER BY trade_date) AS session_idx
            FROM (SELECT DISTINCT trade_date FROM px0)
        ),
        formation AS (
            SELECT c.trade_date,
                   (c120.trade_date < DATE '{holdout_start}') AS common_formation_120d
            FROM calendar c
            LEFT JOIN calendar c120 ON c120.session_idx = c.session_idx + 120
        ),
        joined AS (
            SELECT r.*, {joined_bench}, {joined_buckets},
                   {joined_raw_labels}, {joined_raw_buckets}
            FROM raw_forward r
            {' '.join(f'LEFT JOIN bench_{h} b{h} USING (trade_date, market)' for h in hs)}
            {bucket_joins}
        )
        , {rank_ctes}, {bucket_rank_ctes}
        SELECT j.trade_date, j.ticker, j.market,
               {', '.join(label_select_j)},
               {', '.join(bucket_select_j)}, {bucket_coverage_j},
               f.common_formation_120d,
               {survivor_expr.replace('r.', 'j.')} AS common_survivor_120d,
               {', '.join(f'rank_{h}.y_rank_{h}d' for h in hs)},
               {', '.join(f'bucket_rank_{h1}_{h2}.y_rank_bucket_{h1}_{h2}d' for h1, h2 in buckets)}
        FROM joined j
        {' '.join(f'LEFT JOIN rank_{h} USING (trade_date, ticker, market)' for h in hs)}
        {' '.join(f'LEFT JOIN bucket_rank_{h1}_{h2} USING (trade_date, ticker, market)' for h1, h2 in buckets)}
        LEFT JOIN formation f USING (trade_date)
    """


def materialize_label_scan(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    quality_view: str | None = None,
    holdout_start: str = "2025-08-01",
    force: bool = False,
) -> str:
    materialize(
        con,
        config,
        LABEL_SCAN_TABLE,
        build_label_scan_sql(
            price_view,
            quality_view=quality_view,
            holdout_start=holdout_start,
        ),
        force=force,
    )
    return register_mart_view(con, config, LABEL_SCAN_TABLE)
