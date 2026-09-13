"""Unit tests for model 02's panel builder (20260907_model_experiment `06` §2.2).

The lake here is synthetic but the *universe* mart is built with the real
``build_universe_sql``, so the D-5 contract check this builder performs is
exercised for real rather than stubbed out.

What these tests are for: the builder is the only place in model 02 that moves a
value in time (flow ``lag1``) or invents a column (the interactions), and both
are easy to get subtly wrong in a way no later stage would notice.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.features.regime import REGIME_TABLE
from modeler.etl.lake import connect, register_views
from modeler.etl.mart import mart_glob, mart_table_dir, materialize
from modeler.etl.universe import UniverseFilter, broad_universe_filter, build_universe_sql
from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob import features as fx
from modeler.models._02_updown_prob.spec import ModelSpec

START = dt.date(2020, 1, 1)
N_SESSIONS = 180
TICKERS = (
    ("A001", "KOSPI"),
    ("A002", "KOSPI"),
    ("A003", "KOSPI"),
    ("A004", "KOSDAQ"),
    ("A005", "KOSDAQ"),
    ("A006", "KOSDAQ"),
)
# One halted session for A001, mid-sample: the flow mart has no row there (the
# real one filters halts out), so the lag has to step over the gap.
HALT_TICKER = "A001"
HALT_INDEX = 100

_DAY = "DATEDIFF('day', DATE '2020-01-01', trade_date)"
_NUM = "CAST(SUBSTR(ticker, 2, 3) AS INTEGER)"

_BOOL_COLUMNS = {"px_is_halted", "fin_is_negative_equity", "fin_has_fs"}
# The regime is NULL before this session, so the "regime unknown -> interaction
# NULL" rule has something to bite on (the real mart's warm-up does this). It
# has to sit past the universe filter's own warm-up — 40 valid days in a 60-row
# window — or there would be no panel rows there to check.
_REGIME_WARMUP = 60
# A single missing material value, for the "material NULL -> interaction NULL"
# rule; placed past the regime warm-up so that rule is what does the work.
_MATERIAL_NULL_TICKER = "A002"
_MATERIAL_NULL_INDEX = 110


def _sessions(count: int = N_SESSIONS, start: dt.date = START) -> list[dt.date]:
    out: list[dt.date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _value_expr(index: int) -> str:
    return f"({_DAY} * 1.0 + {_NUM} * 0.001 + {index} * 0.37)"


def _column_expr(name: str, index: int) -> str:
    if name in _BOOL_COLUMNS:
        return f"(({_DAY} + {_NUM}) % 3 = 0) AS {name}"
    if name.endswith("_lag1"):
        # +1000 so reading the plain column instead of the twin is visible.
        return f"({_value_expr(index)} + 1000.0) AS {name}"
    if name == "px_turnover_shock":
        sessions = _sessions()
        missing = sessions[_MATERIAL_NULL_INDEX]
        return (
            f"CASE WHEN ticker = '{_MATERIAL_NULL_TICKER}' "
            f"AND trade_date = DATE '{missing}' THEN NULL "
            f"ELSE {_value_expr(index)} END AS {name}"
        )
    return f"{_value_expr(index)} AS {name}"


def _regime_expr(name: str, index: int) -> str:
    sessions = _sessions()
    warm = sessions[_REGIME_WARMUP]
    body = f"(({_DAY} + {index}) % 3 <> 0)"
    if name.endswith("_z"):
        body = _value_expr(index)
    return f"CASE WHEN trade_date >= DATE '{warm}' THEN {body} END AS {name}"


def _write_raw_ohlcv(config: LakeConfig, sessions: list[dt.date]) -> None:
    rows = []
    for index, day in enumerate(sessions):
        for ticker, market in TICKERS:
            halted = ticker == HALT_TICKER and index == HALT_INDEX
            close = 100_000 + index * 10 + int(ticker[1:])
            rows.append(
                {
                    "trade_date": day,
                    "ticker": ticker,
                    "market": market,
                    "open": 0 if halted else close,
                    "high": 0 if halted else close,
                    "low": 0 if halted else close,
                    "close": close,
                    "volume": 0 if halted else 2_000,
                }
            )
    path = config.raw_root / "daily_ohlcv"
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path / "part-000000.parquet")


def _synth_marts(
    con: duckdb.DuckDBPyConnection, config: LakeConfig, needed: dict[str, list[str]]
) -> None:
    """One synthetic mart per entry, with exactly the columns asked for."""
    for mart, columns in needed.items():
        if mart == REGIME_TABLE:
            selects = ", ".join(_regime_expr(c, i) for i, c in enumerate(columns))
            sql = f"SELECT DISTINCT trade_date, {selects} FROM daily_ohlcv"
        else:
            selects = ", ".join(_column_expr(c, i) for i, c in enumerate(columns))
            sql = (
                f"SELECT trade_date, ticker, market, {selects} FROM daily_ohlcv "
                "WHERE NOT (open = 0 AND high = 0 AND low = 0)"
            )
        materialize(con, config, mart, sql)


def _lake(
    tmp_path: Path,
    spec: ModelSpec,
    horizon: int,
    *,
    start: dt.date = START,
) -> tuple[LakeConfig, duckdb.DuckDBPyConnection]:
    # root=DataRoot(tmp_path), or a write test lands in the repository's data/ dir
    config = LakeConfig(
        root=DataRoot(tmp_path),
        snapshot_date="2020-12-31",
        source="test",
    )
    _write_raw_ohlcv(config, _sessions(start=start))
    con = connect(config)
    register_views(con, config, tables=["daily_ohlcv"])
    # the real SQL, so verify_universe_contract has something true to verify
    materialize(
        con, config, bd.UNIVERSE_VIEW, build_universe_sql(spec.universe, price_view="daily_ohlcv")
    )
    columns, materials = bd.panel_feature_columns(spec, horizon)
    # the columns the panel SQL will read — the flow mart's _lag1 twins included
    _synth_marts(con, config, bd.mart_source_columns(spec, [*columns, *materials]))
    return config, con


def _spec(**kwargs) -> ModelSpec:
    return ModelSpec(period_start="2020-01-01", period_end="2020-12-31", n_folds=3, **kwargs)


@pytest.fixture()
def fs0_build(tmp_path: Path):
    spec = _spec(feature_set="FS0")
    config, _con = _lake(tmp_path, spec, 5)
    return spec, config, bd.build_dataset(spec, config, horizon=5, write=False)


# --- the panel shape ---------------------------------------------------------


def test_panel_carries_exactly_the_feature_set_plus_keys_and_labels(fs0_build) -> None:
    _spec_used, _config, result = fs0_build
    assert result.panel_rows > 0
    assert result.n_dates > 0
    assert result.n_folds == 3
    # every FS0 column is a design column, and nothing unregistered rode along
    design = set(result.feature_cols)
    assert design == set(fx.FS0_COLS), f"design mismatch: {sorted(design ^ set(fx.FS0_COLS))}"


def test_a_horizon_outside_the_spec_is_refused(fs0_build) -> None:
    spec, config, _result = fs0_build  # noqa: F841 - config is used below
    with pytest.raises(ValueError, match="horizon"):
        bd.build_dataset(spec, config, horizon=10, write=False)


# --- flow timing (`02` §2, D-6) ----------------------------------------------


def _mart_frame(config: LakeConfig, name: str) -> pl.DataFrame:
    return pl.read_parquet(mart_glob(config, name).replace("**/", ""))


def _panel(spec: ModelSpec, config: LakeConfig, horizon: int) -> pl.DataFrame:
    """Re-run the panel query alone, so a test can look at the raw columns."""
    con = connect(config)
    register_views(con, config, tables=["daily_ohlcv"])
    columns, materials = bd.panel_feature_columns(spec, horizon)
    bd.register_read_only(
        con, config, [bd.UNIVERSE_VIEW, *sorted(bd.required_mart_columns([*columns, *materials]))]
    )
    from modeler.etl import labels as labels_mod

    con.execute(
        f"CREATE OR REPLACE VIEW {bd.LABEL_VIEW} AS {labels_mod.build_label_sql(spec.label)}"
    )
    panel = pl.from_arrow(con.execute(bd.build_panel_sql(spec, horizon)).arrow())
    columns, _materials = bd.panel_feature_columns(spec, horizon)
    return bd.add_interactions(panel, [i for i in fx.INTERACTIONS if i[0] in columns])


def test_lag1_reads_the_previous_row_of_the_flow_mart(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS0")
    config, _con = _lake(tmp_path, spec, 5)
    panel = _panel(spec, config, 5)
    flow = _mart_frame(config, fx.GROUP_VIEW["flow"]).sort(["ticker", "market", "trade_date"])
    column = "flow_foreign_netbuy_sum_5d"
    expected = flow.with_columns(
        pl.col(column).shift(1).over(["ticker", "market"]).alias("expected")
    ).select(["trade_date", "ticker", "market", "expected"])
    joined = panel.join(expected, on=["trade_date", "ticker", "market"], how="inner")
    assert joined.height > 0
    mismatched = joined.filter(
        (pl.col(column) != pl.col("expected"))
        | (pl.col(column).is_null() != pl.col("expected").is_null())
    )
    assert mismatched.height == 0


def test_lag1_steps_over_a_halt_instead_of_going_null(tmp_path: Path) -> None:
    """The halted session has no flow row, so the next session must reach past it.

    This is the property A0's ``feat_flow`` provides by filtering on
    ``valid_session_idx`` before its own windows: "previous row" is "previous
    valid session". A builder that lagged over a calendar grid would hand the
    session after a halt a NULL.
    """
    spec = _spec(feature_set="FS0")
    config, _con = _lake(tmp_path, spec, 5)
    panel = _panel(spec, config, 5)
    sessions = _sessions()
    halt_day, before, after = (
        sessions[HALT_INDEX],
        sessions[HALT_INDEX - 1],
        sessions[HALT_INDEX + 1],
    )
    column = "flow_foreign_netbuy_sum_5d"

    # the halted session itself is out of the universe entirely
    assert (
        panel.filter((pl.col("ticker") == HALT_TICKER) & (pl.col("trade_date") == halt_day)).height
        == 0
    )

    flow = _mart_frame(config, fx.GROUP_VIEW["flow"])
    pre_halt = flow.filter((pl.col("ticker") == HALT_TICKER) & (pl.col("trade_date") == before))[
        column
    ][0]
    got = panel.filter((pl.col("ticker") == HALT_TICKER) & (pl.col("trade_date") == after))[column][
        0
    ]
    assert got == pytest.approx(pre_halt)


def test_native_t_reads_the_same_session(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS0", flow_variant="native_t")
    config, _con = _lake(tmp_path, spec, 5)
    panel = _panel(spec, config, 5)
    flow = _mart_frame(config, fx.GROUP_VIEW["flow"])
    column = "flow_foreign_netbuy_sum_5d"
    joined = panel.join(
        flow.select(["trade_date", "ticker", "market", pl.col(column).alias("mart")]),
        on=["trade_date", "ticker", "market"],
    )
    assert joined.height > 0
    assert joined.filter(pl.col(column) != pl.col("mart")).height == 0


def test_the_ratio_pair_is_read_from_its_mart_lag1_twin(tmp_path: Path) -> None:
    """FS1's two individual-flow ratios have a ``_lag1`` column in the mart.

    Lagging the plain column instead would be a different number (the fixture
    offsets the twin by 1000 to make that visible) and, on the real mart, a
    subtly different one: the twin was lagged inside the mart's own valid-session
    window.
    """
    spec = _spec(feature_set="FS1")
    config, _con = _lake(tmp_path, spec, 20)
    panel = _panel(spec, config, 20)
    flow = _mart_frame(config, fx.GROUP_VIEW["flow"])
    for column, twin in fx.FLOW_MART_LAG1_SOURCE.items():
        joined = panel.join(
            flow.select(["trade_date", "ticker", "market", pl.col(twin).alias("twin")]),
            on=["trade_date", "ticker", "market"],
        )
        assert joined.filter(pl.col(column) != pl.col("twin")).height == 0


# --- interactions (`02` §3.2) ------------------------------------------------


def test_interaction_is_the_rank_times_the_regime(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS2")
    config, _con = _lake(tmp_path, spec, 20)
    panel = _panel(spec, config, 20)
    name, material, regime = "ix_tshock_liqhigh", "px_turnover_shock", "rg_liq_high"
    assert name in panel.columns

    from modeler.etl.preprocess import per_date_rank_expr

    checked = panel.with_columns(
        per_date_rank_expr(material, ["trade_date", "market"]).alias("rank_material")
    )
    on = checked.filter(pl.col(regime) & pl.col(material).is_not_null())
    off = checked.filter(~pl.col(regime).fill_null(True))
    assert on.height > 0 and off.height > 0
    # regime 1 -> the percentile itself
    assert on.filter((pl.col(name) - pl.col("rank_material")).abs() > 1e-12).height == 0
    # regime 0 -> exactly zero
    assert off.filter(pl.col(name) != 0.0).height == 0


def test_interaction_is_null_when_the_material_or_the_regime_is_unknown(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS2")
    config, _con = _lake(tmp_path, spec, 20)
    panel = _panel(spec, config, 20)
    sessions = _sessions()

    # material NULL (one ticker, one day) -> interaction NULL, not 0
    row = panel.filter(
        (pl.col("ticker") == _MATERIAL_NULL_TICKER)
        & (pl.col("trade_date") == sessions[_MATERIAL_NULL_INDEX])
    )
    assert row.height == 1
    assert row["px_turnover_shock"][0] is None
    assert row["rg_liq_high"][0] is not None  # the regime is known here
    assert row["ix_tshock_liqhigh"][0] is None  # ...so the material is what NULLs it

    # regime NULL (warm-up) -> interaction NULL for every name that day
    warm = panel.filter(pl.col("trade_date") < sessions[_REGIME_WARMUP])
    assert warm.height > 0
    assert warm["rg_liq_high"].null_count() == warm.height
    assert warm["ix_tshock_liqhigh"].null_count() == warm.height


def test_interaction_materials_never_reach_the_design_matrix(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS2")
    config, _con = _lake(tmp_path, spec, 20)
    result = bd.build_dataset(spec, config, horizon=20, write=False)
    assert "px_turnover_shock" not in result.feature_cols
    assert "flow_foreign_netbuy_to_volume_20d_lag1" not in result.feature_cols
    assert set(result.feature_cols) == set(spec.feature_columns(20))


def test_the_rank_profile_leaves_interactions_alone(tmp_path: Path) -> None:
    """``ix_*`` is already a percentile; re-ranking it would erase the zeros.

    On a regime-off session the whole cross-section is 0. Ranked again, that
    becomes 0.5 everywhere — indistinguishable from a mid-ranked name on a
    regime-on session, and the marker the tree splits on is gone.
    """
    spec = _spec(feature_set="FS2")
    config, _con = _lake(tmp_path, spec, 20)
    bd.build_dataset(spec, config, horizon=20, write=True, created_at="t")
    directory = config.dataset_dir(spec.model_id) / bd.dataset_key(spec, 20)
    panel = pl.read_parquet(directory / "feat_panel.parquet")
    std = pl.read_parquet(directory / bd.STD_DIR / "*.parquet")

    name = "ix_mktbeta_mktup"
    joined = std.join(
        panel.select(["trade_date", "ticker", "market", pl.col(name).alias("raw")]),
        on=["trade_date", "ticker", "market"],
    )
    assert joined.height > 0
    assert joined.filter((pl.col(name) - pl.col("raw")).abs() > 1e-12).height == 0
    # a plain feature *was* ranked, so the profile really did run
    ranked = std["px_ret_20d"]
    assert float(ranked.min()) >= 0.0 and float(ranked.max()) <= 1.0


# --- the marts are read, never rebuilt (`02` §5) ------------------------------


def test_building_does_not_touch_a_single_mart_file(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS2")
    config, _con = _lake(tmp_path, spec, 20)
    marts = [bd.UNIVERSE_VIEW, *sorted(bd.required_mart_columns(list(spec.feature_columns(20))))]
    before = {
        path: path.stat().st_mtime_ns
        for mart in marts
        for path in mart_table_dir(config, mart).rglob("*")
        if path.is_file()
    }
    assert before

    bd.build_dataset(spec, config, horizon=20, write=False)

    after = {path: path.stat().st_mtime_ns for path in before}
    assert after == before


def test_mart_contracts_are_recorded_for_every_mart_read(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS1")
    config, _con = _lake(tmp_path, spec, 20)
    result = bd.build_dataset(spec, config, horizon=20, write=False)
    expected = {bd.UNIVERSE_VIEW, *bd.required_mart_columns(list(spec.feature_columns(20)))}
    assert set(result.mart_contracts) == expected
    for contract in result.mart_contracts.values():
        assert contract is not None
        assert "sql_hash" in contract


def test_a_missing_mart_says_so_instead_of_building_one(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS1")
    config, _con = _lake(tmp_path, spec, 20)
    import shutil

    shutil.rmtree(mart_table_dir(config, fx.GROUP_VIEW["pex"]))
    with pytest.raises(FileNotFoundError, match="never builds a mart"):
        bd.build_dataset(spec, config, horizon=20, write=False)


# --- the D-5 universe contract ------------------------------------------------


def test_a_universe_built_with_another_filter_is_refused(tmp_path: Path) -> None:
    """Reading someone else's ``dim_universe_daily`` is only safe if it matches.

    The thresholds live in the SQL text, so an equal ``sql_hash`` means an equal
    filter — the broad universe (liquidity filter off) does not match, and D-5
    says the model trains on the default filter.
    """
    spec = _spec(feature_set="FS0")
    config, con = _lake(tmp_path, spec, 5)
    materialize(
        con,
        config,
        bd.UNIVERSE_VIEW,
        build_universe_sql(broad_universe_filter(), price_view="daily_ohlcv"),
        force=True,
    )
    with pytest.raises(ValueError, match="UniverseFilter"):
        bd.build_dataset(spec, config, horizon=5, write=False)
    # ...and the check is what refuses it, not a downstream accident
    assert bd.build_dataset(spec, config, horizon=5, write=False, strict_universe=False).panel_rows


def test_the_default_filter_passes_the_check(tmp_path: Path) -> None:
    spec = _spec(feature_set="FS0")
    config, _con = _lake(tmp_path, spec, 5)
    assert bd.verify_universe_contract(config, spec)
    assert spec.universe == UniverseFilter()


# --- the holdout guard --------------------------------------------------------


def test_a_panel_reaching_the_holdout_fails_even_if_the_spec_was_bypassed(
    tmp_path: Path,
) -> None:
    """Defense in depth for `06` §8's named risk.

    ``ModelSpec`` refuses a late ``period_end`` at construction; this is the
    second line, for a spec mutated past that check.
    """
    # a lake that reaches into the holdout window, and a spec mutated to let it in
    spec = ModelSpec(
        period_start="2025-01-02", period_end="2025-07-31", n_folds=3, feature_set="FS0"
    )
    config, _con = _lake(tmp_path, spec, 5, start=dt.date(2025, 1, 1))
    # inside the boundary, the same lake builds fine
    assert bd.build_dataset(spec, config, horizon=5, write=False).panel_rows

    object.__setattr__(spec, "period_end", "2026-06-10")
    with pytest.raises(AssertionError, match="holdout boundary"):
        bd.build_dataset(spec, config, horizon=5, write=False)


# --- the design-matrix layout -------------------------------------------------


def extra_layout(result: bd.BuildResult) -> str:
    return json.loads(result.manifest_path.read_text(encoding="utf-8"))["extra"]["std_layout"]


def test_a_stateless_profile_stores_one_design_matrix_not_one_per_fold(tmp_path: Path) -> None:
    """`02` §4.1: the rank transform depends only on a row's own cross-section.

    A fold slice is whole dates, so every fold would transform a row to the same
    value — five expanding folds would write the early years five times over.
    Measured on E0's FS0 panel: 4.15GB of per-fold copies against ~1.4GB for one.
    """
    spec = _spec(feature_set="FS0")
    config, _con = _lake(tmp_path, spec, 5)
    result = bd.build_dataset(spec, config, horizon=5, write=True)
    assert extra_layout(result) == "single"

    parts = sorted((result.dataset_dir / bd.STD_DIR).glob("*.parquet"))
    assert [p.name for p in parts] == ["part-000000.parquet"]
    std = pl.read_parquet(parts[0])
    panel = pl.read_parquet(result.dataset_dir / "feat_panel.parquet")
    assert std.height == panel.height  # 1x the panel, not 3x
    assert "fold_id" not in std.columns  # the fold table says which dates are whose


def test_a_fold_fitted_profile_stores_one_part_per_slice(tmp_path: Path) -> None:
    """``tree`` fits winsorize bounds per fold, so its slices genuinely differ."""
    spec = _spec(feature_set="FS0", preprocess_profile="tree")
    config, _con = _lake(tmp_path, spec, 5)
    result = bd.build_dataset(spec, config, horizon=5, write=True)
    assert extra_layout(result) == "per_fold"

    names = sorted(p.name for p in (result.dataset_dir / bd.STD_DIR).glob("*.parquet"))
    assert names == [
        "fold1_train.parquet",
        "fold1_valid.parquet",
        "fold2_train.parquet",
        "fold2_valid.parquet",
        "fold3_train.parquet",
        "fold3_valid.parquet",
    ]
    part = pl.read_parquet(result.dataset_dir / bd.STD_DIR / "fold2_train.parquet")
    assert part["fold_id"].unique().to_list() == [2]
    assert part["fold_role"].unique().to_list() == ["train"]


def test_the_stateless_layout_gives_the_same_values_as_per_fold_slicing(tmp_path: Path) -> None:
    """The claim the single layout rests on, checked rather than asserted.

    Transforming the whole panel and then slicing must equal transforming each
    fold's slice — true because the percentile group is ``(trade_date, market)``
    and a slice never splits a date.
    """
    spec = _spec(feature_set="FS0")
    config, _con = _lake(tmp_path, spec, 5)
    result = bd.build_dataset(spec, config, horizon=5, write=True)
    whole = pl.read_parquet(result.dataset_dir / bd.STD_DIR / "part-000000.parquet")
    panel = pl.read_parquet(result.dataset_dir / "feat_panel.parquet")
    folds = pl.read_parquet(result.dataset_dir / "split_folds.parquet")

    from modeler.etl import preprocess as pp

    cfg = pp.PreprocessConfig(profile="rank")
    row = folds.row(0, named=True)
    slice_dates = (pl.col("trade_date") >= row["train_start"]) & (
        pl.col("trade_date") <= row["train_end"]
    )
    per_fold = pp.fit(panel.filter(slice_dates), cfg).transform(panel.filter(slice_dates))
    from_whole = whole.filter(slice_dates)
    columns = [c for c in per_fold.columns if c in from_whole.columns]
    assert (
        per_fold.select(columns)
        .sort(["trade_date", "ticker"])
        .equals(from_whole.select(columns).sort(["trade_date", "ticker"]))
    )


# --- artifacts ----------------------------------------------------------------


def _panel_path(spec: ModelSpec, config: LakeConfig, horizon: int) -> Path:
    return config.dataset_dir(spec.model_id) / bd.dataset_key(spec, horizon) / "feat_panel.parquet"


def test_write_produces_the_artifacts_and_a_manifest_that_pins_the_build(
    tmp_path: Path,
) -> None:
    spec = _spec(feature_set="FS1")
    config, _con = _lake(tmp_path, spec, 20)
    result = bd.build_dataset(spec, config, horizon=20, write=True, created_at="2026-09-09T00:00")

    directory = result.dataset_dir
    assert directory.name == "FS1_h20_lag1_rank"
    for name in ("feat_panel", "label_daily", "split_folds"):
        assert (directory / f"{name}.parquet").is_file()
    parts = sorted((directory / bd.STD_DIR).glob("*.parquet"))
    assert len(parts) == 1  # the rank profile is stateless -> one panel, not one per fold
    assert extra_layout(result) == "single"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    extra = manifest["extra"]
    assert extra["feature_set"] == "FS1"
    assert extra["flow_variant"] == "lag1"
    assert extra["horizon"] == 20
    assert extra["n_folds"] == 3
    assert extra["feature_columns"] == list(spec.feature_columns(20))
    assert set(extra["mart_contracts"]) == {
        bd.UNIVERSE_VIEW,
        *bd.required_mart_columns(list(spec.feature_columns(20))),
    }
    assert set(extra["null_ratios"]) == set(spec.feature_columns(20))
    assert manifest["period"] == {"start": "2020-01-01", "end": "2020-12-31"}

    folds = pl.read_parquet(directory / "split_folds.parquet")
    assert folds.height == 3
    assert set(folds["role"].to_list()) == {"fold"}  # holdout_len=0 (`03` §1)
    # embargo == purge == h (=20), and both gaps sit between the kept train end
    # and the valid start: purge drops the trailing train sessions whose labels
    # reach into the gap, embargo is the gap itself.
    dates = (
        pl.read_parquet(directory / "feat_panel.parquet")["trade_date"].unique().sort().to_list()
    )
    index = {d: i for i, d in enumerate(dates)}
    for row in folds.iter_rows(named=True):
        assert index[row["valid_start"]] - index[row["train_end"]] == 20 + 20 + 1
