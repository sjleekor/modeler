"""Tests for Phase C conditional IC (Stage 1b §4-§7).

Design: ``docs/dev/20260829_macro_features/01_design/03_stage1b_conditional_ic_phase_c.md``
§7.4 lists what has to hold. The fixtures are synthetic daily IC plus a
synthetic regime series — the same two inputs the real run reads, so nothing
here needs a panel or a lake.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from modeler.analysis.horizon_scan_config import CONFIG_PATH, load_config
from modeler.analysis.horizon_scan_daily_ic import DAILY_IC_COLUMNS, DAILY_IC_DIR_NAME
from modeler.analysis.horizon_scan_phase_c import (
    DISCOVERY_SAMPLE_KIND,
    DISCOVERY_UNIVERSE,
    PairSpec,
    apply_phase_c_bh,
    assert_regime_columns_present,
    assign_evidence_grade,
    build_phase_c_run_spec,
    compute_period_sign_pass,
    compute_screen_pass,
    compute_tradable_pass,
    estimate_pair,
    load_daily_ic,
    load_pairs,
    placebo_seed,
    run_regime_placebo,
    select_cell_ic,
)

MACRO_PATH = CONFIG_PATH.with_name("horizon_scan_macro_20260829.yaml")


def _sessions(n: int, start: date = date(2016, 1, 4)) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _pair(**overrides) -> PairSpec:
    kwargs = {
        "pair_id": "P1",
        "role": "primary",
        "family": "px_idio_vol_60d",
        "scan_type": "cum",
        "h_start": 0,
        "h_end": 60,
        "regime_id": "vix_up",
        "direction": "+",
    }
    kwargs.update(overrides)
    return PairSpec(**kwargs)


def _daily_ic(
    values: np.ndarray,
    days: list[date],
    *,
    pair: PairSpec,
    universe: str = DISCOVERY_UNIVERSE,
) -> pl.DataFrame:
    n = len(days)
    return pl.DataFrame(
        {
            "hypothesis_id": [f"{pair.family}|f|cum|0|{pair.h_end}"] * n,
            "family": [pair.family] * n,
            "feature": ["f"] * n,
            "scan_type": [pair.scan_type] * n,
            "h_start": [pair.h_start] * n,
            "h_end": [pair.h_end] * n,
            "universe": [universe] * n,
            "sample_kind": [DISCOVERY_SAMPLE_KIND] * n,
            "hypothesis_role": ["primary"] * n,
            "trade_date": days,
            "formation_session_idx": list(range(1, n + 1)),
            "rank_ic": values,
            "n_obs": [500] * n,
            "rank_ic_kospi": values,
            "n_kospi": [300] * n,
            "rank_ic_kosdaq": values,
            "n_kosdaq": [200] * n,
        }
    ).select(DAILY_IC_COLUMNS)


def _regimes(flags: list[bool], days: list[date], *, regime_id: str = "vix_up") -> pl.DataFrame:
    data = {"trade_date": days, "session_idx": list(range(1, len(days) + 1))}
    from modeler.analysis.horizon_scan_phase_c_regimes import REGIME_IDS

    for rid in REGIME_IDS:
        data[f"s_{rid}"] = flags if rid == regime_id else [None] * len(days)
        data[f"alt_s_{rid}"] = flags if rid == regime_id else [None] * len(days)
    return pl.DataFrame(data)


def _planted(n: int = 1200, delta: float = 0.03, block: int = 60, seed: int = 1):
    """``IC_t = a + delta * s_t + noise`` with a persistent regime."""
    rng = np.random.default_rng(seed)
    days = _sessions(n)
    flags = [((i // block) % 2) == 0 for i in range(n)]
    values = 0.01 + delta * np.array(flags, dtype=float) + rng.normal(0, 0.02, n)
    return days, flags, values


# --- §4 estimation ---


def test_the_planted_delta_is_recovered() -> None:
    days, flags, values = _planted()
    pair = _pair()
    row = estimate_pair(pair, _daily_ic(values, days, pair=pair), _regimes(flags, days))

    assert row["status"] == "valid"
    assert row["delta"] == pytest.approx(0.03, abs=0.005)
    assert row["ic_mean_s1"] - row["ic_mean_s0"] == pytest.approx(row["delta"], abs=1e-12)
    assert row["nw_lag"] == 59  # h_end - 1, the cell's own convention
    assert row["n_dates"] == len(days)
    assert row["n_dates_s1"] + row["n_dates_s0"] == row["n_dates"]


def test_persistence_is_reported_with_the_estimate() -> None:
    """§6.5: how few independent switches the sample really has is part of the
    result, not a footnote — it is what makes G4 necessary."""
    days, flags, values = _planted(block=60)
    pair = _pair()
    row = estimate_pair(pair, _daily_ic(values, days, pair=pair), _regimes(flags, days))

    assert row["mean_run_length_s1"] == pytest.approx(60.0)
    assert row["mean_run_length_s0"] == pytest.approx(60.0)
    assert row["n_regime_transitions"] == len(days) // 60 - 1


def test_g1_occupancy_failure_withholds_the_estimate() -> None:
    """§5: a regime that barely varies has no delta worth publishing. The pair
    keeps its slot in the BH population at p=1.0 — m never shrinks."""
    days = _sessions(1200)
    rng = np.random.default_rng(2)
    flags = [i < 100 for i in range(len(days))]  # 100 of 1200 -> 8% share
    values = 0.01 + rng.normal(0, 0.02, len(days))
    pair = _pair()

    row = estimate_pair(pair, _daily_ic(values, days, pair=pair), _regimes(flags, days))
    assert row["status"] == "insufficient"
    assert row["status_reason"].startswith("g1_occupancy")
    assert row["delta"] is None
    assert row["n_dates_s1"] == 100


def test_the_sample_start_bound_is_applied() -> None:
    days, flags, values = _planted(n=800)
    pair = _pair()
    full = estimate_pair(pair, _daily_ic(values, days, pair=pair), _regimes(flags, days))
    trimmed = estimate_pair(
        pair,
        _daily_ic(values, days, pair=pair),
        _regimes(flags, days),
        sample_start=days[300],
    )
    assert trimmed["n_dates"] == len(days) - 300
    assert trimmed["sample_start"] == days[300]
    assert full["n_dates"] > trimmed["n_dates"]


def test_a_missing_regime_column_is_reported_not_silently_skipped() -> None:
    days, flags, values = _planted(n=400)
    pair = _pair(regime_id="vix_up")
    regimes = _regimes(flags, days).drop("s_vix_up")
    row = estimate_pair(pair, _daily_ic(values, days, pair=pair), regimes)
    assert row["status_reason"] == "unknown_regime:vix_up"


def test_a_cell_that_resolves_to_two_hypotheses_is_rejected() -> None:
    """The pair names exactly one cell (§3.1); an ambiguous match would make
    the conditioned series undefined."""
    days, flags, values = _planted(n=200)
    pair = _pair()
    frame = _daily_ic(values, days, pair=pair)
    doubled = pl.concat([frame, frame.with_columns(pl.lit("other").alias("hypothesis_id"))])
    with pytest.raises(ValueError, match="resolves to 2 hypotheses"):
        select_cell_ic(
            doubled, pair, universe=DISCOVERY_UNIVERSE, sample_kind=DISCOVERY_SAMPLE_KIND
        )


# --- §6.1 G4 placebo ---


def test_placebo_p_is_large_when_two_persistent_series_are_unrelated() -> None:
    """§5.1's whole argument. The regime and the IC are both strongly
    persistent but independent; HAC at lag 59 still leaves a large |t|, and the
    circular shift is what shows it means nothing."""
    rng = np.random.default_rng(17)
    n = 1500
    flags = np.array([((i // 70) % 2) == 0 for i in range(n)])
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.97 * noise[i - 1] + rng.normal(0, 0.01)
    values = 0.01 + noise  # no regime term at all
    sessions = np.arange(1, n + 1)
    pair = _pair()

    result = run_regime_placebo(
        pair, values, flags, sessions, config_hash="cfg", universe="broad", repeats=60
    )
    assert result["placebo_status"] == "ok"
    assert result["placebo_p"] > 0.10
    assert result["placebo_pass"] is False


def test_placebo_p_is_small_when_the_regime_really_drives_the_ic() -> None:
    days, flags, values = _planted(n=1500, delta=0.05, block=70, seed=23)
    pair = _pair()
    result = run_regime_placebo(
        pair,
        values,
        np.array(flags),
        np.arange(1, len(days) + 1),
        config_hash="cfg",
        universe="broad",
        repeats=60,
    )
    assert result["placebo_p"] <= 0.10
    assert result["placebo_pass"] is True


def test_placebo_is_deterministic_across_runs() -> None:
    days, flags, values = _planted(n=1000)
    pair = _pair()
    kwargs = dict(config_hash="cfg", universe="broad", repeats=20)
    a = run_regime_placebo(pair, values, np.array(flags), np.arange(1, len(days) + 1), **kwargs)
    b = run_regime_placebo(pair, values, np.array(flags), np.arange(1, len(days) + 1), **kwargs)
    assert a == b


def test_placebo_seeds_differ_by_pair_universe_and_replicate() -> None:
    base = dict(config_hash="cfg", base_seed=20260829)
    seeds = {
        placebo_seed(replicate_index=0, pair_id="P1", universe="broad", **base),
        placebo_seed(replicate_index=1, pair_id="P1", universe="broad", **base),
        placebo_seed(replicate_index=0, pair_id="P2", universe="broad", **base),
        placebo_seed(replicate_index=0, pair_id="P1", universe="tradable", **base),
        placebo_seed(
            replicate_index=0,
            pair_id="P1",
            universe="broad",
            config_hash="other",
            base_seed=20260829,
        ),
    }
    assert len(seeds) == 5


def test_a_sample_too_short_for_the_minimum_shift_reports_rather_than_shifts() -> None:
    days, flags, values = _planted(n=200)
    pair = _pair()
    result = run_regime_placebo(
        pair,
        values,
        np.array(flags),
        np.arange(1, len(days) + 1),
        config_hash="cfg",
        universe="broad",
        repeats=10,
        min_shift=120,
    )
    assert result["placebo_p"] is None
    assert result["placebo_status"].startswith("sample_too_short")


# --- §4.3 BH, §5 gates ---


def _row(pair_id: str, **overrides) -> dict:
    row = {
        "pair_id": pair_id,
        "regime_role": "primary",
        "status": "valid",
        "p_nw": 0.5,
        "delta": 0.01,
        "direction_preregistered": None,
    }
    row.update(overrides)
    return row


def test_bh_population_is_the_primary_pairs_only() -> None:
    rows = [_row(f"P{i}", p_nw=0.001 if i == 1 else 0.5) for i in range(1, 16)]
    rows += [_row("X1", regime_role="reference", p_nw=0.0001)]
    rows += [_row("E1", regime_role="exploratory", p_nw=0.0001)]

    scored = apply_phase_c_bh(rows, q_threshold=0.10)
    by_id = {r["pair_id"]: r for r in scored}
    assert by_id["P1"]["q_fdr_phase_c"] is not None
    # A reference or exploratory pair is reported, never part of the claim.
    assert by_id["X1"]["q_fdr_phase_c"] is None
    assert by_id["X1"]["discovery"] is False
    assert by_id["E1"]["discovery"] is False


def test_an_insufficient_pair_keeps_its_slot_at_p_one() -> None:
    rows = [
        _row("P1", p_nw=0.001),
        *[_row(f"P{i}", status="insufficient", p_nw=None) for i in range(2, 16)],
    ]
    scored = apply_phase_c_bh(rows, q_threshold=0.10)
    by_id = {r["pair_id"]: r for r in scored}
    # 15 hypotheses in m, not 1 — the smallest q is p * 15 / 1.
    assert by_id["P1"]["q_fdr_phase_c"] == pytest.approx(0.001 * 15)
    assert by_id["P2"]["discovery"] is False


def test_a_wrong_signed_delta_fails_a_direction_fixed_pair() -> None:
    rows = [
        _row("P1", p_nw=1e-6, delta=-0.02, direction_preregistered="+"),
        _row("P2", p_nw=1e-6, delta=-0.02, direction_preregistered="-"),
        _row("P3", p_nw=1e-6, delta=-0.02, direction_preregistered=None),
        *[_row(f"P{i}") for i in range(4, 16)],
    ]
    scored = {r["pair_id"]: r for r in apply_phase_c_bh(rows, q_threshold=0.10)}
    assert scored["P1"]["direction_pass"] is False
    assert scored["P1"]["discovery"] is False
    assert scored["P2"]["direction_pass"] is True
    assert scored["P2"]["discovery"] is True
    # A two-sided pair has no directional gate at all.
    assert scored["P3"]["direction_pass"] is None
    assert scored["P3"]["discovery"] is True


def test_screen_pass_needs_every_gate() -> None:
    passing = {
        "regime_role": "primary",
        "status": "valid",
        "discovery": True,
        "period_sign_pass": True,
        "tradable_pass": True,
        "placebo_pass": True,
    }
    assert compute_screen_pass(passing)["screen_pass"] is True
    for gate in ("discovery", "period_sign_pass", "tradable_pass", "placebo_pass"):
        result = compute_screen_pass({**passing, gate: False})
        assert result["screen_pass"] is False
        assert result["failed_gates"]


def test_period_sign_pass_needs_a_ceiling_half_majority_of_valid_windows() -> None:
    # Four valid windows, two agreeing -> ceil(4/2) = 2, so it passes.
    assert compute_period_sign_pass([0.1, 0.1, -0.1, -0.1], 0.05)["period_sign_pass"] is True
    assert compute_period_sign_pass([0.1, -0.1, -0.1, -0.1], 0.05)["period_sign_pass"] is False
    # None entries are windows too thin on one side; they neither help nor hurt.
    result = compute_period_sign_pass([0.1, 0.1, None, None, None], 0.05)
    assert result["valid_subperiods"] == 2
    assert result["period_sign_pass"] is True


def test_tradable_retention_needs_the_same_sign_and_half_the_magnitude() -> None:
    assert compute_tradable_pass(delta_broad=0.04, delta_tradable=0.03)["tradable_pass"] is True
    assert compute_tradable_pass(delta_broad=0.04, delta_tradable=0.01)["tradable_pass"] is False
    assert compute_tradable_pass(delta_broad=0.04, delta_tradable=-0.03)["tradable_pass"] is False
    # A zero broad delta cannot anchor a ratio; that is a failure, not "N/A".
    zero = compute_tradable_pass(delta_broad=0.0, delta_tradable=0.03)
    assert zero["tradable_retention"] is None
    assert zero["tradable_pass"] is False


def test_evidence_grades_follow_the_rubric() -> None:
    base = {
        "regime_role": "primary",
        "status": "valid",
        "screen_pass": True,
        "valid_subperiods": 5,
        "alt_cut_sign_agree": True,
    }
    assert assign_evidence_grade(base) == "A"
    assert assign_evidence_grade({**base, "valid_subperiods": 3}) == "B"
    assert assign_evidence_grade({**base, "alt_cut_sign_agree": False}) == "B"
    assert assign_evidence_grade({**base, "screen_pass": False}) == "D"
    assert assign_evidence_grade({**base, "status": "insufficient"}) == "C"
    assert assign_evidence_grade({**base, "regime_role": "exploratory"}) == "C"
    assert assign_evidence_grade({**base, "regime_role": "reference"}) == "R"


# --- contract wiring ---


def test_pairs_load_from_the_registered_contract() -> None:
    config = load_config(MACRO_PATH)
    pairs = load_pairs(config.raw)
    assert len(pairs) == 17
    assert sum(1 for p in pairs if p.role == "primary") == 15
    by_id = {p.pair_id: p for p in pairs}
    assert by_id["P1"].family == "px_idio_vol_60d"
    assert by_id["P1"].regime_id == "vix_up"
    assert by_id["P1"].direction == "+"
    assert by_id["P1"].nw_lag == 59
    assert by_id["P3"].nw_lag == 4  # cum 0->5
    assert by_id["P15"].family == "px_market_beta"
    assert by_id["X1"].role == "reference"


def test_every_contract_regime_must_exist_in_the_series() -> None:
    days = _sessions(50)
    complete = _regimes([True] * 50, days)
    assert_regime_columns_present(complete)
    with pytest.raises(ValueError, match="missing columns"):
        assert_regime_columns_present(complete.drop("s_liq_high"))


# --- §7.1 run spec ---


def _write_daily_ic(run_dir: Path, frame: pl.DataFrame, *, under_core: bool) -> None:
    root = (run_dir / "core" / DAILY_IC_DIR_NAME) if under_core else (run_dir / DAILY_IC_DIR_NAME)
    target = root / "family=fam"
    target.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target / "f.parquet")


def test_daily_ic_loads_from_both_the_phase_a_and_phase_b_layouts(tmp_path: Path) -> None:
    """Phase A keeps it under ``core/``, Phase B at the run root; one call has
    to cover the (A, B) pair the CLI is given."""
    days, flags, values = _planted(n=100)
    pair = _pair()
    a_dir, b_dir = tmp_path / "run_id=A", tmp_path / "run_id=B"
    _write_daily_ic(a_dir, _daily_ic(values, days, pair=pair), under_core=True)
    _write_daily_ic(b_dir, _daily_ic(values, days, pair=pair), under_core=False)

    frame = load_daily_ic([a_dir, b_dir])
    assert frame.height == 2 * len(days)

    with pytest.raises(FileNotFoundError, match="no daily_ic"):
        load_daily_ic([tmp_path / "run_id=missing"])


def test_run_spec_records_both_input_runs_with_their_daily_ic_hashes(tmp_path: Path) -> None:
    """§7.1: the run ids are execution facts, not contract — so they live here
    with a content hash, instead of forcing a hash-exclusion rule in config."""
    days, flags, values = _planted(n=100)
    pair = _pair()
    a_dir, b_dir = tmp_path / "run_id=A", tmp_path / "run_id=B"
    _write_daily_ic(a_dir, _daily_ic(values, days, pair=pair), under_core=True)
    _write_daily_ic(b_dir, _daily_ic(values, days, pair=pair), under_core=False)

    config = load_config(MACRO_PATH)
    spec = build_phase_c_run_spec(
        config_hash=config.config_hash,
        phase_c_block=config.raw["phase_c"],
        phase_a_run_dir=a_dir,
        phase_b_run_dir=b_dir,
        snapshot_date="2026-08-23",
        source="sj2_remote",
        sessions=days,
        started_at="2026-08-29T10:00:00+09:00",
        command_line=["horizon_scan_phase_c"],
    )

    assert spec["phase"] == "C"
    assert spec["config_hash"] == config.config_hash
    assert spec["phase_a_run_id"] == "run_id=A"
    assert spec["phase_b_run_id"] == "run_id=B"
    assert set(spec["daily_ic_sha256"]) == {"run_id=A", "run_id=B"}
    assert all(len(h) == 64 for h in spec["daily_ic_sha256"].values())
    assert spec["n_sessions"] == len(days)
    assert spec["placebo_seed"] == 20260829
    # JSON-serializable: it is written before anything else in the run.
    json.dumps(spec, default=str)


def test_the_run_spec_hash_moves_when_the_input_daily_ic_changes(tmp_path: Path) -> None:
    days, flags, values = _planted(n=100)
    pair = _pair()
    config = load_config(MACRO_PATH)

    def _spec(scale: float) -> dict:
        run_dir = tmp_path / f"run_id=A{scale}"
        _write_daily_ic(run_dir, _daily_ic(values * scale, days, pair=pair), under_core=True)
        return build_phase_c_run_spec(
            config_hash=config.config_hash,
            phase_c_block=config.raw["phase_c"],
            phase_a_run_dir=run_dir,
            phase_b_run_dir=None,
            snapshot_date="2026-08-23",
            source="sj2_remote",
            sessions=days,
            started_at="2026-08-29T10:00:00+09:00",
            command_line=[],
        )

    first, second = _spec(1.0), _spec(2.0)
    assert list(first["daily_ic_sha256"].values()) != list(second["daily_ic_sha256"].values())
    assert first["session_grid_sha256"] == second["session_grid_sha256"]


def test_a_config_without_registered_pairs_is_refused() -> None:
    """The base config carries a ``phase_c`` block that is only an open policy;
    running Phase C against it would produce an empty, meaningless result."""
    from modeler.analysis.horizon_scan_phase_c import run_phase_c

    with pytest.raises(ValueError, match="registers no phase_c.pairs"):
        run_phase_c(
            phase_a_run_dir=Path("unused"),
            phase_b_run_dir=None,
            snapshot_date="2026-08-23",
            source="sj2_remote",
            output_root=Path("unused"),
            command_line=[],
            config_path=CONFIG_PATH,
        )


def test_non_session_rows_in_the_regime_series_cannot_reach_the_estimate() -> None:
    """§2.1/§7.4(g): the join is on the scan's own ``trade_date``, so a regime
    row for a date the scan never had is simply not there to condition on."""
    days, flags, values = _planted(n=600)
    pair = _pair()
    extra_days = days + [days[-1] + timedelta(days=1)]
    regimes = _regimes(flags + [True], extra_days)

    row = estimate_pair(pair, _daily_ic(values, days, pair=pair), regimes)
    assert row["n_dates"] == len(days)
    assert row["sample_end"] == days[-1]


def test_the_estimate_uses_only_formation_dates_the_regime_already_knew() -> None:
    """``s_t`` is the regime at formation date t, so shifting the regime one
    session forward has to change the answer — if it did not, the join would be
    ignoring the alignment the contract depends on."""
    days, flags, values = _planted(n=800, delta=0.05, block=50)
    pair = _pair()
    aligned = estimate_pair(pair, _daily_ic(values, days, pair=pair), _regimes(flags, days))
    shifted = estimate_pair(
        pair,
        _daily_ic(values, days, pair=pair),
        _regimes([flags[0]] + flags[:-1], days),
    )
    assert aligned["delta"] != shifted["delta"]
    assert math.isfinite(aligned["delta"])


# --- synthetic end-to-end (§7.4) ---


def _synthetic_lake(tmp_path: Path, days: list[date], flags: list[bool]) -> Path:
    """A lake with just what Phase C reads: label_scan and the common fact.

    Phase C builds no panel, so this is genuinely everything it needs — which
    is the property being demonstrated as much as tested.
    """
    import duckdb
    from modeler.analysis.horizon_scan_phase_c_regimes import SOURCE_FEATURE_CODES

    lake_root = tmp_path / "data_lake"
    mart = lake_root / "derived" / "feature" / "snapshot_date=2026-08-23" / "source=sj2_remote"
    derived = lake_root / "derived" / "metric" / "snapshot_date=2026-08-23" / "source=sj2_remote"

    con = duckdb.connect()
    (mart / "label_scan").mkdir(parents=True, exist_ok=True)
    con.execute(
        "CREATE TABLE label_scan AS SELECT * FROM (VALUES "
        + ",".join(f"(DATE '{d.isoformat()}', 'A', 'KOSPI', true)" for d in days)
        + ") t(trade_date, ticker, market, common_formation_120d)"
    )
    con.execute(
        f"COPY label_scan TO '{(mart / 'label_scan' / 'part-000000.parquet').as_posix()}' "
        "(FORMAT PARQUET)"
    )
    (mart / "label_scan" / "_cache_metadata.json").write_text(
        json.dumps(
            {
                "analysis_config_hash": load_config(MACRO_PATH).config_hash,
                "schema_hash": "x",
                "sql_hash": "y",
            }
        )
    )
    (mart / "_manifests").mkdir(parents=True, exist_ok=True)
    (mart / "_manifests" / "_SUCCESS.json").write_text(
        json.dumps({"config_hash": load_config(MACRO_PATH).config_hash, "status": "success"})
    )

    # VIX rises for the sessions the regime should mark, so `vix_up` reproduces
    # the flags the daily IC was generated against.
    vix = []
    level = 20.0
    for i in range(len(days)):
        level += 0.5 if flags[min(i + 20, len(flags) - 1)] else -0.5
        vix.append(level)
    rows = []
    defaults = {
        "global_vix_level": None,
        "market_kospi_close": 2500.0,
        "market_kospi_turnover_value": 1.0e12,
        "market_kosdaq_turnover_value": 5.0e11,
        "rate_kr_term_spread_10y_3y": 0.5,
        "market_kosdaq_ret_1d": 0.0,
        "market_kospi_ret_1d": 0.0,
        "fx_usdkrw_level": 1300.0,
    }
    for code in SOURCE_FEATURE_CODES:
        for i, d in enumerate(days):
            value = vix[i] if code == "global_vix_level" else defaults[code]
            rows.append((d, code, float(value), d))
    con.execute(
        "CREATE TABLE cfdf (feature_date DATE, feature_code VARCHAR, "
        "value_numeric DOUBLE, asof_available_date DATE)"
    )
    con.executemany("INSERT INTO cfdf VALUES (?,?,?,?)", rows)
    (derived / "common_feature_daily_fact").mkdir(parents=True, exist_ok=True)
    con.execute(
        "COPY cfdf TO "
        f"'{(derived / 'common_feature_daily_fact' / 'part-000000.parquet').as_posix()}' "
        "(FORMAT PARQUET)"
    )
    return lake_root


def test_phase_c_publishes_a_complete_run_directory(tmp_path: Path) -> None:
    from modeler.analysis.horizon_scan_phase_c import (
        CONDITIONAL_IC_TABLE,
        PHASE_C_REPORT_NAME,
        PHASE_C_RUN_SPEC_NAME,
        run_phase_c,
    )
    from modeler.etl.config import DataRoot

    days, flags, values = _planted(n=1400, delta=0.04, block=70)
    lake_root = _synthetic_lake(tmp_path, days, flags)
    root = DataRoot(lake_root)

    # One daily IC series per registered family, at both universes, so every
    # pair resolves rather than reporting no_daily_ic.
    config = load_config(MACRO_PATH)
    frames = []
    for pair in load_pairs(config.raw):
        for universe in ("broad", "tradable"):
            spec = PairSpec(
                pair_id=pair.pair_id,
                role=pair.role,
                family=pair.family,
                scan_type=pair.scan_type,
                h_start=pair.h_start,
                h_end=pair.h_end,
                regime_id=pair.regime_id,
                direction=pair.direction,
            )
            frames.append(_daily_ic(values, days, pair=spec, universe=universe))
    run_dir = tmp_path / "run_id=A"
    _write_daily_ic(
        run_dir,
        pl.concat(frames, how="vertical").unique(
            subset=["hypothesis_id", "universe", "sample_kind", "trade_date"], keep="first"
        ),
        under_core=True,
    )

    published = run_phase_c(
        phase_a_run_dir=run_dir,
        phase_b_run_dir=None,
        snapshot_date="2026-08-23",
        source="sj2_remote",
        output_root=tmp_path / "out",
        command_line=["horizon_scan_phase_c"],
        config_path=MACRO_PATH,
        root=root,
    )

    assert (published / PHASE_C_RUN_SPEC_NAME).is_file()
    assert (published / "manifest.json").is_file()
    assert (published / f"{CONDITIONAL_IC_TABLE}.parquet").is_file()
    assert (published / PHASE_C_REPORT_NAME).is_file()
    assert (published / "_SUCCESS.json").is_file()
    assert f"config_hash={config.config_hash}" in published.as_posix()

    frame = pl.read_parquet(published / f"{CONDITIONAL_IC_TABLE}.parquet")
    assert frame.height == 17
    assert set(frame["regime_role"].to_list()) == {"primary", "reference"}
    assert frame.filter(pl.col("regime_role") == "primary").height == 15
    report = (published / PHASE_C_REPORT_NAME).read_text(encoding="utf-8")
    assert "Phase C 조건부 IC 결과" in report
    assert "읽는 규칙" in report
