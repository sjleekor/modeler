"""C stage 3 — dataset assembly for the next KR model (18 §3 C3-10).

Plan: ``my/milestones/common/20261012_feature_explainer/18_c3_training_test.md`` (C3-2 sample,
C3-3 missingness, C3-4 input timing, C3-10 this module), ``17_c2_feature_set.md`` (the 43 / 30
columns and their marts), ``16_c1_label_benchmark.md`` (label, benchmark, universe).

One module for training and shadow serving. It returns one SQL statement (DuckDB, over
registered views) and a polars wrapper. Pieces, in order:

1. **Marts** — built inline as CTEs from the same ``build_*_sql`` functions the mart
   materializers use, only for the marts the chosen feature set reads (``COLUMN_MART``):
   ``feat_price_v2``, ``feat_price`` (the four legacy columns), ``feat_flow_v2`` (main set only),
   ``feat_fin_v2`` (vintage ``_dl``), ``feat_market_cap`` and ``feat_filing_activity``.
   ``mart_overrides`` swaps a CTE for an already registered mart view.
2. **Universe** (C3-2) — ``build_universe_raw_price_sql(tradable_universe_filter())`` ->
   ``universe_terminal`` exclusion -> ``build_one_per_company_sql``; rows with
   ``keep_one_per_company`` stay. ``in_terminal_tail`` comes from the entry-label CTEs
   (``ent``). In serving mode (``include_labels=False``) only the CTEs up to ``ent`` are
   emitted, nothing that computes a return, label or exit kind.
3. **Input timing** (C3-4) — the columns in :data:`LAGGED_COLS` are replaced by their value on the
   ticker's previous valid (non-halt) session. Price columns and universe flags are same-day.
   The lag runs over the whole panel *before* the one-per-company and date filters, so the
   previous row is the ticker's previous valid session, not the previous kept row.
4. **Date filter** — ``start_date`` (default 2017-01-01, C3-2) and ``end_date`` (inclusive).
5. **Labels** (``include_labels=True``) — ``build_entry_label_sql`` pieces: c1 entry, h20, excess
   vs. the KOSPI 200 fixed benchmark. ``y`` is the percentile rank of
   ``raw_label_k200_c1_20d`` within ``trade_date`` over both markets, among the *trainable* rows
   of the kept sample; ``trainable`` is the label gate (C3-2).

Missing values stay missing: no ``_isna`` column, no fill (C3-3).

Why the lag is here and not in the marts (C3-4): the marts stay as they are (they are the
serving contract, the code is hashed), and the same module serves training and shadow serving.
The per-ticker previous-valid-row convention is the one ``serving/kr_prepare.py`` uses for the
flow lag ("applied once at the per-ticker previous valid row"), now over the joined panel.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import duckdb
import polars as pl

from modeler.etl.corporate_actions import MCAP_DISTORTION_VIEW
from modeler.etl.features.filing_activity import build_filing_activity_sql
from modeler.etl.features.fin_v2 import build_fin_v2_sql
from modeler.etl.features.flow_v2 import build_flow_v2_sql
from modeler.etl.features.market_cap import build_market_cap_sql
from modeler.etl.features.price import build_price_sql
from modeler.etl.features.price_v2 import build_price_v2_sql
from modeler.etl.label_gate import DEFAULT_ALLOWED_KINDS, LABEL_EXIT_KINDS
from modeler.etl.labels import LabelSpec
from modeler.etl.labels_entry import _entry_ctes, build_entry_label_sql
from modeler.etl.one_per_company import KEEP_COL, build_one_per_company_sql
from modeler.etl.universe import UniverseFilter, tradable_universe_filter
from modeler.etl.universe_raw_price import build_universe_raw_price_sql
from modeler.etl.universe_terminal import build_universe_v2_sql
from modeler.models._03_kr_next import feature_set as fs

FeatureSet = Literal["main", "noflow"]

KEYS: tuple[str, ...] = ("trade_date", "ticker", "market")
DEFAULT_START_DATE = "2017-01-01"  # C3-2: before this the financial TTM is not full yet
TAG = "c1"  # entry at the t+1 close
HORIZON = 20

#: Columns whose value is read one valid session late (C3-4, decided 2026-10-18).
#: ``daily_market_cap`` (KRX raw values) reaches the server one day after the price, so every
#: input built from it is shifted; the flow features are shifted as a whole, like the current
#: serving (the source publication time is not recorded).
KRX_RAW_LAGGED_COLS: tuple[str, ...] = (
    "mcap_krx_log",
    "px_turnover_v2",
    "px_turnover_ma20_v2",
    "px_amihud_20d_v2",
    # flow columns whose denominator is a KRX raw value (these are also flow columns)
    "flow_foreign_netbuy_to_listed_5d_v2",
    "flow_foreign_netbuy_to_listed_20d_v2",
    "flow_inst_netbuy_to_listed_5d_v2",
    "flow_inst_netbuy_to_listed_20d_v2",
    "flow_indiv_netbuy_to_listed_5d_v2",
    "flow_indiv_netbuy_to_listed_20d_v2",
    "flow_individual_netbuy_to_volume_5d_v2",
    "flow_individual_netbuy_to_volume_20d_v2",
)
#: Every flow feature (the set's ``FLOW_COLS``): shifted as a whole.
FLOW_LAGGED_COLS: tuple[str, ...] = fs.FLOW_COLS
LAGGED_COLS: tuple[str, ...] = tuple(dict.fromkeys((*KRX_RAW_LAGGED_COLS, *FLOW_LAGGED_COLS)))

#: Universe flags carried to the output.
UNIVERSE_FLAG_COLS: tuple[str, ...] = (
    "in_universe",
    "in_universe_v2",
    "raw_price_missing",
    "keep_one_per_company",
    "corp_code",
)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _ident(value: str, what: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"{what} must be a plain SQL identifier, got {value!r}")
    return value


@dataclass(frozen=True)
class KrNextViews:
    """Names of the registered views the assembly reads (defaults = the lake's names)."""

    daily_ohlcv: str = "daily_ohlcv"
    daily_market_cap: str = "daily_market_cap"
    #: ``dim_price_quality_daily`` (``valid_session_idx``, ``ca_event``)
    quality_view: str = "dim_price_quality_daily"
    #: the PIT view built by ``build_stock_pit_nonempty_sql`` (registered under this name)
    pit_view: str = "dim_stock_pit_daily"
    flow_raw: str = "krx_security_flow_raw"
    #: ``fin_quarterly_metric_vintage_dl`` (C2: the ``_dl`` vintage with delisted names)
    vintage_view: str = "fin_quarterly_metric_vintage_dl"
    #: panel of (trade_date, ticker, market) that gives the filing mart its calendar
    filing_universe_view: str = "dim_universe_broad_daily"
    filing_receipt_view: str = "dart_filing_receipt_raw"
    mcap_distortion_view: str | None = MCAP_DISTORTION_VIEW
    stock_master: str = "stock_master"
    dart_corp_master: str = "dart_corp_master"
    krx_index_daily: str = "krx_index_daily"

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None:
                _ident(value, name)


def feature_columns(feature_set: str) -> tuple[str, ...]:
    """The column list of a feature set: ``main`` (43) or ``noflow`` (30)."""
    if feature_set == "main":
        return fs.MAIN_COLS
    if feature_set == "noflow":
        return fs.NOFLOW_COLS
    raise ValueError(f"feature_set must be 'main' or 'noflow', got {feature_set!r}")


def lagged_columns(feature_set: str) -> tuple[str, ...]:
    """The part of :data:`LAGGED_COLS` that the feature set has, in the set's order."""
    cols = set(LAGGED_COLS)
    return tuple(c for c in feature_columns(feature_set) if c in cols)


def label_columns(horizon: int = HORIZON) -> dict[str, str]:
    """Names of the label columns the assembly emits (``include_labels=True``)."""
    return {
        "raw": f"raw_label_{fs.FIXED_BENCHMARK[0]}_{TAG}_{horizon}d",
        "fwd": f"fwd_ret_{TAG}_{horizon}d",
        "kind": f"label_exit_kind_{TAG}_{horizon}d",
        "end_date": f"label_end_date_{TAG}_{horizon}d",
        "exit_lag": f"label_exit_lag_{TAG}_{horizon}d",
    }


def _mart_sql(mart: str, v: KrNextViews) -> str:
    """The SELECT that materializes ``mart`` (same builder call as the mart's materializer)."""
    if mart == "feat_price_v2":
        return build_price_v2_sql(
            v.daily_ohlcv, quality_view=v.quality_view, market_cap_view=v.daily_market_cap
        )
    if mart == "feat_price":
        return build_price_sql(v.daily_ohlcv, quality_view=v.quality_view)
    if mart == "feat_flow_v2":
        return build_flow_v2_sql(
            v.flow_raw,
            pit_view=v.pit_view,
            quality_view=v.quality_view,
            market_cap_view=v.daily_market_cap,
        )
    if mart == "feat_fin_v2":
        return build_fin_v2_sql(
            pit_view=v.pit_view, quality_view=v.quality_view, vintage_view=v.vintage_view
        )
    if mart == "feat_market_cap":
        return build_market_cap_sql(v.daily_market_cap, distortion_view=v.mcap_distortion_view)
    if mart == "feat_filing_activity":
        return build_filing_activity_sql(v.filing_universe_view, v.filing_receipt_view)
    raise KeyError(f"no builder for mart {mart!r}")


def _terminal_tail_sql(spec: LabelSpec, v: KrNextViews, **kw: int) -> str:
    """``in_terminal_tail`` only: the entry-label CTEs up to ``ent``, no return or kind.

    The CTEs before the first per-horizon one (``fx{h}``) are the price/series/tail part;
    ``ent`` is the last of them and carries ``in_terminal_tail``.
    """
    ctes = _entry_ctes(
        spec,
        TAG,
        v.daily_ohlcv,
        v.quality_view,
        master_view=v.stock_master,
        end_halt_haircut=0.0,
        **kw,
    )
    cut = next((i for i, c in enumerate(ctes) if re.match(r"\s*fx\d+ AS \(", c)), None)
    if cut is None or not re.match(r"\s*ent AS \(", ctes[cut - 1]):
        raise RuntimeError("labels_entry CTE layout changed: cannot cut the tail CTEs at 'ent'")
    body = ",\n".join(ctes[:cut])
    return f"""
        WITH {body}
        SELECT trade_date, ticker, market, in_terminal_tail FROM ent
    """


def build_kr_next_panel_sql(
    feature_set: FeatureSet = "main",
    *,
    views: KrNextViews | None = None,
    include_labels: bool = False,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    horizon: int = HORIZON,
    universe_filter: UniverseFilter | None = None,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
    keep_raw_label: bool = False,
    mart_overrides: Mapping[str, str] | None = None,
    min_cross_section: int = 30,
    min_end_gap_sessions: int = 1,
    end_halt_sessions: int = 20,
    terminal_tail_max: int = 10,
) -> str:
    """One SQL statement: the next KR model's panel (kept universe rows, features, labels).

    Columns, in order: ``trade_date, ticker, market``; the feature set's columns (NULL kept);
    ``in_universe, in_universe_v2, raw_price_missing, keep_one_per_company, corp_code``; with
    ``include_labels`` also ``y``, ``trainable`` and ``label_exit_kind_c1_{h}d``,
    ``label_end_date_c1_{h}d``, ``label_exit_lag_c1_{h}d`` (plus ``fwd_ret_c1_{h}d`` and
    ``raw_label_k200_c1_{h}d`` with ``keep_raw_label``). Rows: ``keep_one_per_company`` rows
    with ``start_date <= trade_date <= end_date``, ordered by the keys. Halted days are not
    in the universe, so they never appear.

    ``y`` = percent rank of ``raw_label_k200_c1_{h}d`` within ``trade_date`` (both markets as
    one cross-section) among the ``trainable`` rows of that date; NULL on other rows.
    ``trainable`` = ``fwd`` not NULL and exit kind in ``allowed_kinds`` (``label_gate``);
    rows are never dropped for it.

    ``mart_overrides``: ``{mart name: registered view}`` to read an already built mart instead
    of building it inline. A mart must be unique on ``(trade_date, ticker, market)``
    (``feat_market_cap`` on ``(trade_date, ticker)``; see below).

    ``feat_market_cap`` is joined on ``(trade_date, ticker)`` only: its ``market`` is the
    market at the time, while ``daily_ohlcv`` carries the current one for the whole history
    (the same reason as ``universe_raw_price`` and ``flow_v2``). If a ticker-day has two rows,
    the smaller market name wins, so the pick is deterministic.
    """
    v = views or KrNextViews()
    cols = feature_columns(feature_set)
    for d, name in ((start_date, "start_date"), (end_date, "end_date")):
        if d is not None and not _DATE.match(d):
            raise ValueError(f"{name} must be 'YYYY-MM-DD', got {d!r}")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive int, got {horizon!r}")
    kinds = tuple(allowed_kinds)
    bad = sorted(set(kinds) - set(LABEL_EXIT_KINDS))
    if bad:
        raise ValueError(f"unknown label_exit_kind {bad}; known: {list(LABEL_EXIT_KINDS)}")
    overrides = dict(mart_overrides or {})
    unknown = sorted(set(overrides) - set(fs.MART_COLUMNS))
    if unknown:
        raise KeyError(f"unknown mart in mart_overrides: {unknown}")
    flt = universe_filter or tradable_universe_filter()

    # --- marts (only those the set reads) -------------------------------------------------
    marts = [m for m in fs.MART_COLUMNS if any(fs.COLUMN_MART[c] == m for c in cols)]
    ctes: list[str] = []
    for m in marts:
        if m in overrides:
            body = f"SELECT * FROM {_ident(overrides[m], 'mart override')}"
        else:
            body = _mart_sql(m, v)
        ctes.append(f"kn_{m} AS (\n{body.strip()}\n)")
    if "feat_market_cap" in marts:
        ctes.append(
            "kn_mcap1 AS (\n"
            "    SELECT trade_date, ticker, mcap_krx_log FROM kn_feat_market_cap\n"
            "    QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_date, ticker ORDER BY market) = 1\n"
            ")"
        )

    # --- label / tail CTE the universe exclusion reads -------------------------------------
    label = label_columns(horizon)
    spec = LabelSpec(horizons=(horizon,), kind="excess", bench="eqw_market", outputs=("rank",))
    terminal_kw = {
        "min_end_gap_sessions": min_end_gap_sessions,
        "end_halt_sessions": end_halt_sessions,
        "terminal_tail_max": terminal_tail_max,
        "min_cross_section": min_cross_section,
    }
    if include_labels:
        label_sql = build_entry_label_sql(
            spec,
            TAG,
            v.daily_ohlcv,
            v.quality_view,
            master_view=v.stock_master,
            end_halt_haircut=0.0,
            fixed_benchmarks=(fs.FIXED_BENCHMARK,),
            fixed_index_view=v.krx_index_daily,
            **terminal_kw,
        )
        ctes.append(f"kn_label AS (\n{label_sql.strip()}\n)")
        tail_relation = "kn_label"
    else:
        ctes.append(f"kn_tail AS (\n{_terminal_tail_sql(spec, v, **terminal_kw).strip()}\n)")
        tail_relation = "kn_tail"

    # --- universe -------------------------------------------------------------------------
    raw_uni = build_universe_raw_price_sql(
        flt, price_view=v.daily_ohlcv, market_cap_view=v.daily_market_cap
    )
    uni_sql = build_one_per_company_sql(
        build_universe_v2_sql(raw_uni, tail_relation),
        corp_view=v.dart_corp_master,
        market_cap_view=v.daily_market_cap,
    )
    ctes.append(f"kn_uni AS (\n{uni_sql.strip()}\n)")

    # --- panel: universe rows + marts, valid (non-halt) sessions only ----------------------
    flag_cols = ", ".join(f"u.{c}" for c in UNIVERSE_FLAG_COLS)
    select_feat: list[str] = []
    joins: list[str] = []
    for m in marts:
        mine = [c for c in cols if fs.COLUMN_MART[c] == m]
        if m == "feat_market_cap":
            joins.append(
                "LEFT JOIN kn_mcap1 m_mcap ON m_mcap.trade_date = u.trade_date "
                "AND m_mcap.ticker = u.ticker"
            )
            select_feat += [f"m_mcap.{c}" for c in mine]
        else:
            a = f"m_{m}"
            joins.append(
                f"LEFT JOIN kn_{m} {a} ON {a}.trade_date = u.trade_date "
                f"AND {a}.ticker = u.ticker AND {a}.market = u.market"
            )
            select_feat += [f"{a}.{c}" for c in mine]
    ctes.append(
        f"""kn_feat AS (
    SELECT u.trade_date, u.ticker, u.market, {flag_cols},
           {", ".join(select_feat)}
    FROM kn_uni u
    {chr(10).join("    " + j for j in joins).strip()}
    WHERE NOT u.is_halted
)"""
    )

    # --- input timing (C3-4): previous valid session of the ticker --------------------------
    lagged = lagged_columns(feature_set)
    replace = ", ".join(f"LAG({c}) OVER w AS {c}" for c in lagged)
    ctes.append(
        f"""kn_lag AS (
    SELECT * REPLACE ({replace})
    FROM kn_feat
    WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
)"""
    )

    # --- sample + date filter ---------------------------------------------------------------
    where = [KEEP_COL, f"trade_date >= DATE '{start_date}'"]
    if end_date is not None:
        where.append(f"trade_date <= DATE '{end_date}'")
    ctes.append(f"kn_sel AS (\n    SELECT * FROM kn_lag WHERE {' AND '.join(where)}\n)")

    out_cols = [*KEYS, *cols, *UNIVERSE_FLAG_COLS]
    if not include_labels:
        final_from, extra = "kn_sel", []
    else:
        kinds_sql = ", ".join(f"'{k}'" for k in kinds)
        ctes.append(
            f"""kn_lab AS (
    SELECT s.*,
           l.{label["fwd"]}, l.{label["raw"]},
           l.{label["kind"]}, l.{label["end_date"]}, l.{label["exit_lag"]}
    FROM kn_sel s
    LEFT JOIN kn_label l
      ON l.trade_date = s.trade_date AND l.ticker = s.ticker AND l.market = s.market
)"""
        )
        # PARTITION BY the flag too: the non-trainable rows form their own partition, so the
        # percent rank of the trainable ones is over the trainable rows only.
        ctes.append(
            f"""kn_y AS (
    SELECT *,
           CASE WHEN trainable
                THEN PERCENT_RANK() OVER (
                    PARTITION BY trade_date, trainable ORDER BY {label["raw"]})
           END AS y
    FROM (
        SELECT *,
               COALESCE({label["fwd"]} IS NOT NULL AND {label["raw"]} IS NOT NULL
                        AND {label["kind"]} IN ({kinds_sql}), FALSE) AS trainable
        FROM kn_lab
    )
)"""
        )
        final_from = "kn_y"
        extra = ["y", "trainable", label["kind"], label["end_date"], label["exit_lag"]]
        if keep_raw_label:
            extra += [label["fwd"], label["raw"]]
    select = ", ".join([*out_cols, *extra])
    return (
        "WITH\n"
        + ",\n".join(ctes)
        + f"\nSELECT {select}\nFROM {final_from}\nORDER BY trade_date, ticker, market"
    )


def build_kr_next_panel(
    con: duckdb.DuckDBPyConnection, feature_set: FeatureSet = "main", **kwargs: object
) -> pl.DataFrame:
    """Run :func:`build_kr_next_panel_sql` on ``con`` and return a polars frame."""
    sql = build_kr_next_panel_sql(feature_set, **kwargs)  # type: ignore[arg-type]
    return con.execute(sql).pl()
