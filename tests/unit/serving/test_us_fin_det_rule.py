"""Rule A (det_a) for the 11 US financial features and its serving gate."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.serving import us_daily
from modeler.serving.orchestration import (
    US_PARITY_CRITERIA,
    US_PARITY_EVIDENCE_SCHEMA,
    us_native_block_reason,
    us_parity_failures,
)
from modeler.us.features.fundamentals_ttm import (
    RULE_DET_A,
    RULE_LEGACY,
    flow_ttm,
    instant_latest,
    instant_yoy_pair,
    market_cap,
)

SCHEMA = ["cik", "tag", "start", "end", "fp", "filed", "unit", "val", "accn", "form"]
FILED = date(2021, 1, 1)
PANEL = pl.DataFrame({"date": [date(2021, 1, 2)], "symbol": ["ABC"], "cik": [1], "close": [2.0]})


class Lake:
    def __init__(self, facts: pl.DataFrame):
        self.facts = facts

    def scan(self, table: str) -> pl.LazyFrame:
        assert table == "fundamentals"
        return self.facts.lazy()


def _facts(rows: list[tuple], order: list[int] | None = None) -> pl.DataFrame:
    order = order if order is not None else list(range(len(rows)))
    return pl.DataFrame([rows[i] for i in order], schema=SCHEMA, orient="row")


def _ni(start, end, fp, val, *, filed=FILED, unit="USD", accn="A1", form="10-K"):
    return (1, "NetIncomeLoss", start, end, fp, filed, unit, val, accn, form)


def _clean_year() -> list[tuple]:
    s = date(2020, 1, 1)
    return [
        _ni(s, date(2020, 3, 31), "Q1", 10.0),
        _ni(s, date(2020, 6, 30), "Q2", 30.0),
        _ni(s, date(2020, 9, 30), "Q3", 60.0),
        _ni(s, date(2020, 12, 31), "FY", 100.0),
    ]


def _assets(val, *, unit="USD", accn="A1", form="10-K", tag="Assets"):
    return (1, tag, None, date(2020, 12, 31), "FY", FILED, unit, val, accn, form)


def _ttm(rows, rule=None, order=None):
    lake = Lake(_facts(rows, order))
    out = flow_ttm(PANEL, lake, ["NetIncomeLoss"]) if rule is None else flow_ttm(
        PANEL, lake, ["NetIncomeLoss"], rule)
    return out["value"].item()


def _latest(rows, rule=None, order=None):
    lake = Lake(_facts(rows, order))
    out = instant_latest(PANEL, lake, "Assets") if rule is None else instant_latest(
        PANEL, lake, "Assets", rule)
    return out["value"].item()


def test_legacy_is_the_default_and_unchanged_on_unambiguous_facts() -> None:
    rows = _clean_year() + [_assets(500.0)]
    assert _ttm(rows) == _ttm(rows, RULE_LEGACY) == 100.0
    assert _latest(rows) == _latest(rows, RULE_LEGACY) == 500.0
    # det_a agrees when nothing is ambiguous.
    assert _ttm(rows, RULE_DET_A) == 100.0
    assert _latest(rows, RULE_DET_A) == 500.0
    shares = [(1, "EntityCommonStockSharesOutstanding", None, date(2020, 12, 31), "FY", FILED,
               "shares", 7.0, "A1", "10-K")]
    cap = market_cap(PANEL, Lake(_facts(shares)))
    assert cap["mcap"].item() == 14.0
    assert market_cap(PANEL, Lake(_facts(shares)), RULE_DET_A)["mcap"].item() == 14.0
    pair = instant_yoy_pair(PANEL, Lake(_facts(rows)), "Assets")
    assert pair["cur_val"].item() == 500.0 and pair["prior_val"].item() is None


def test_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="선택 규칙"):
        _ttm(_clean_year(), "rule_b")
    with pytest.raises(ValueError, match="선택 규칙"):
        _latest([_assets(1.0)], "nope")


def test_det_a_same_filed_selection_does_not_depend_on_input_order() -> None:
    # Same slot, USD vs CNY, different accn: the larger accn wins (CNY), in any order.
    rows = _clean_year() + [
        _assets(100.0, unit="USD", accn="0001-21-000001"),
        _assets(700.0, unit="CNY", accn="0001-21-000002"),
    ]
    rng = random.Random(3)
    orders = [list(range(len(rows))), list(reversed(range(len(rows))))]
    for _ in range(25):
        order = list(range(len(rows)))
        rng.shuffle(order)
        orders.append(order)
    assert {_latest(rows, RULE_DET_A, o) for o in orders} == {700.0}
    assert {_ttm(rows, RULE_DET_A, o) for o in orders} == {100.0}


def test_det_a_tiebreaks_form_unit_then_val() -> None:
    same_accn = [
        _assets(700.0, unit="CNY", accn="A9"),
        _assets(100.0, unit="USD", accn="A9"),
    ]
    assert _latest(same_accn, RULE_DET_A) == 100.0  # USD/shares preferred
    assert _latest(list(reversed(same_accn)), RULE_DET_A) == 100.0
    amended = [
        _assets(300.0, accn="A9", form="10-K"),
        _assets(200.0, accn="A9", form="10-K/A"),
    ]
    assert _latest(amended, RULE_DET_A) == 200.0  # amendment wins
    assert _latest(list(reversed(amended)), RULE_DET_A) == 200.0
    same_all = [_assets(5.0, accn="A9"), _assets(9.0, accn="A9")]
    assert _latest(same_all, RULE_DET_A) == 9.0  # val ascending, last
    assert _latest(list(reversed(same_all)), RULE_DET_A) == 9.0


def test_det_a_quarter_end_tie_uses_start_descending_then_value() -> None:
    a = date(2020, 1, 1)
    b = date(2020, 1, 2)
    rows = _clean_year() + [_ni(b, date(2020, 3, 31), "Q1", 11.0, accn="A2")]
    # top4 by q_end: 40 + 30 + 20, then the 2020-03-31 tie (10 from start a, 11 from start b).
    # start descending picks b -> 101, for every input order.
    rng = random.Random(5)
    results = set()
    for _ in range(30):
        order = list(range(len(rows)))
        rng.shuffle(order)
        results.add(_ttm(rows, RULE_DET_A, order))
    results.add(_ttm(rows, RULE_DET_A, list(reversed(range(len(rows))))))
    assert results == {101.0}
    assert a < b


def test_det_a_matches_the_tie_fixture_case_from_the_design_review() -> None:
    rows = [
        _assets(100.0, unit="USD", accn="A1"),
        _assets(700.0, unit="CNY", accn="A1"),
        *[
            _ni(date(2020, 1, 1), end, fp, val, unit=unit, accn="A1")
            for end, fp, val in (
                (date(2020, 3, 31), "Q1", 10.0),
                (date(2020, 6, 30), "Q2", 30.0),
                (date(2020, 9, 30), "Q3", 60.0),
                (date(2020, 12, 31), "FY", 100.0),
            )
            for unit in ("USD",)
        ],
        _ni(date(2020, 1, 1), date(2020, 12, 31), "FY", 700.0, unit="CNY", accn="A1"),
    ]
    forward = (_latest(rows, RULE_DET_A), _ttm(rows, RULE_DET_A))
    backward = (
        _latest(rows, RULE_DET_A, list(reversed(range(len(rows))))),
        _ttm(rows, RULE_DET_A, list(reversed(range(len(rows))))),
    )
    assert forward == backward == (100.0, 100.0)


def test_financial_code_hash_covers_rule_and_callers() -> None:
    digest = us_daily.financial_code_hash()
    assert len(digest) == 64
    assert digest == us_daily.financial_code_hash()


# --- gate evidence ---------------------------------------------------------------------


def _model_metrics() -> dict:
    return {
        "keys_equal": True, "rows": 19862, "min_daily_spearman": 0.999994,
        "min_top50_overlap": 1.0, "min_top100_overlap": 1.0, "p99_abs_rank_shift": 1,
        "frozen_top100_max_abs_rank_shift": 1,
    }


def _evidence(**overrides) -> dict:
    evidence = {
        "schema_version": US_PARITY_EVIDENCE_SCHEMA,
        "status": "score_equivalent",
        "rule": RULE_DET_A,
        "criteria": dict(US_PARITY_CRITERIA),
        "labels_read": False,
        "financial_code_hash": us_daily.financial_code_hash(),
        "inputs": {"bundle_matches_frozen": True},
        "determinism": {"runs": [
            {"name": f"run{i}", "keys_equal": True, "diff_cells": 0} for i in range(4)
        ]},
        "non_financial": {"keys_equal": True, "eligible_equal": True, "diff_cells": 0},
        "models": {"lightgbm": _model_metrics(), "ridge": _model_metrics()},
    }
    evidence.update(overrides)
    return evidence


def test_gate_accepts_the_reviewed_numbers_and_rejects_each_violation() -> None:
    assert us_parity_failures(_evidence()) == []
    cases = {
        ("models", "lightgbm", "min_daily_spearman"): 0.99989,
        ("models", "ridge", "min_top50_overlap"): 0.98,
        ("models", "ridge", "min_top100_overlap"): 0.99,
        ("models", "lightgbm", "p99_abs_rank_shift"): 2,
        ("models", "ridge", "frozen_top100_max_abs_rank_shift"): 2,
        ("models", "lightgbm", "keys_equal"): False,
    }
    for path, value in cases.items():
        bad = _evidence()
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        assert us_parity_failures(bad), path
    bad = _evidence()
    bad["determinism"]["runs"][2]["diff_cells"] = 1
    assert us_parity_failures(bad) == ["determinism:run2"]
    bad = _evidence()
    bad["non_financial"]["diff_cells"] = 1
    assert us_parity_failures(bad) == ["non_financial"]
    bad = _evidence(criteria={**US_PARITY_CRITERIA, "min_daily_spearman": 0.9})
    assert "criteria differ from code constants" in us_parity_failures(bad)
    assert us_parity_failures(_evidence(labels_read=True))
    assert us_parity_failures(_evidence(inputs={"bundle_matches_frozen": False}))
    assert us_parity_failures({"schema_version": "x"}) == ["schema_version"]


def test_gate_status_is_failed_when_metrics_fail_even_if_status_says_ok() -> None:
    bad = _evidence()
    bad["models"]["ridge"]["p99_abs_rank_shift"] = 9
    assert bad["status"] == "score_equivalent"
    assert us_parity_failures(bad)


def test_rank_shift_metrics_identical_and_shifted() -> None:
    symbols = [f"S{i:04d}" for i in range(300)]
    day = date(2024, 1, 2)
    base = pl.DataFrame({"date": [day] * 300, "symbol": symbols, "rank": list(range(1, 301))})
    same = us_daily._rank_shift_metrics(base, base)
    assert same["changed_rows"] == 0 and same["min_daily_spearman"] == 1.0
    assert same["min_top50_overlap"] == 1.0 and same["min_top100_overlap"] == 1.0
    ranks = list(range(1, 301))
    ranks[149], ranks[150] = ranks[150], ranks[149]  # swap two rows far below the top 100
    swapped = base.with_columns(pl.Series("rank", ranks))
    got = us_daily._rank_shift_metrics(base, swapped)
    assert got["changed_rows"] == 2 and got["max_abs_rank_shift"] == 1
    assert got["p99_abs_rank_shift"] <= 1 and got["frozen_top100_max_abs_rank_shift"] == 0
    assert got["min_daily_spearman"] < 1.0
    top = list(range(1, 301))
    top[49], top[50] = top[50], top[49]  # crosses the top-50 boundary
    crossed = us_daily._rank_shift_metrics(base, base.with_columns(pl.Series("rank", top)))
    assert crossed["min_top50_overlap"] == pytest.approx(49 / 50)
    assert crossed["frozen_top100_max_abs_rank_shift"] == 1


# --- serving verdict -------------------------------------------------------------------


def _native(tmp_path: Path, evidence: dict | None = None, **overrides) -> dict:
    evidence = evidence if evidence is not None else _evidence()
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(evidence, sort_keys=True))
    native = {
        "raw_feature_parity_status": "score_equivalent",
        "diagnostic_only": False,
        "serving_eligible": True,
        "financial_selection_rule": RULE_DET_A,
        "financial_code_hash": evidence.get("financial_code_hash"),
        "raw_feature_parity_evidence": str(path),
        "raw_feature_parity_evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    native.update(overrides)
    return native


def test_score_equivalent_native_with_pinned_passing_evidence_is_servable(tmp_path: Path) -> None:
    assert us_native_block_reason(_native(tmp_path)) is None


@pytest.mark.parametrize("override", [
    {"financial_selection_rule": "legacy"},
    {"financial_selection_rule": None},
    {"raw_feature_parity_evidence_sha256": "0" * 64},
    {"raw_feature_parity_evidence_sha256": None},
    {"raw_feature_parity_evidence": None},
    {"raw_feature_parity_evidence": "/nonexistent/parity.json"},
    {"serving_eligible": False},
    {"diagnostic_only": True},
    {"serving_eligible": None},
    {"financial_code_hash": "f" * 64},
    {"raw_feature_parity_status": "failed"},
])
def test_score_equivalent_is_blocked_unless_every_pin_holds(tmp_path: Path, override: dict) -> None:
    assert us_native_block_reason(_native(tmp_path, **override)) is not None


def test_blocked_when_evidence_failed_or_rule_differs_or_stale(tmp_path: Path) -> None:
    failed = _evidence(status="failed")
    assert us_native_block_reason(_native(tmp_path, failed)) is not None
    other_rule = _evidence(rule="legacy")
    assert us_native_block_reason(_native(tmp_path, other_rule)) is not None
    stale = _evidence(financial_code_hash="a" * 64)
    assert us_native_block_reason(_native(tmp_path, stale, financial_code_hash="b" * 64)) is not None
    numbers = _evidence()
    numbers["models"]["lightgbm"]["min_daily_spearman"] = 0.99
    assert us_native_block_reason(_native(tmp_path, numbers)) is not None


def test_historical_blocks_and_absent_fields_behave_as_before() -> None:
    assert us_native_block_reason({"raw_feature_parity_status": "failed"}) is not None
    assert us_native_block_reason({"diagnostic_only": True}) is not None
    assert us_native_block_reason({"serving_eligible": False}) is not None
    assert us_native_block_reason({"raw_feature_parity_status": "unverified",
                                   "serving_eligible": False, "diagnostic_only": True}) is not None
    assert us_native_block_reason({"market": "US"}) is None


def test_prepare_refuses_score_equivalent_without_valid_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires evidence"):
        us_daily.prepare_daily_features(
            date(2026, 9, 28), raw_feature_parity_status="score_equivalent")
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(_evidence()))
    with pytest.raises(ValueError, match="diagnostic_only"):
        us_daily.prepare_daily_features(
            date(2026, 9, 28), diagnostic_only=True,
            raw_feature_parity_status="score_equivalent", raw_feature_parity_evidence=str(path))
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(_evidence(financial_code_hash="0" * 64)))
    with pytest.raises(ValueError, match="재무 코드"):
        us_daily.prepare_daily_features(
            date(2026, 9, 28), raw_feature_parity_status="score_equivalent",
            raw_feature_parity_evidence=str(stale))
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps(_evidence(status="failed")))
    with pytest.raises(ValueError, match="score_equivalent"):
        us_daily.prepare_daily_features(
            date(2026, 9, 28), raw_feature_parity_status="score_equivalent",
            raw_feature_parity_evidence=str(failed))


def test_validate_parity_evidence_accepts_current_passing_rule_a(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(_evidence()))
    assert us_daily.validate_parity_evidence(path)["rule"] == RULE_DET_A
    bad = copy.deepcopy(_evidence())
    bad["models"]["ridge"]["p99_abs_rank_shift"] = 5
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="gate 기준"):
        us_daily.validate_parity_evidence(path)


def test_prepare_cli_score_equivalent_requires_evidence_and_forbids_diagnostic() -> None:
    for extra in (
        ["--raw-feature-parity-status", "score_equivalent"],
        ["--raw-feature-parity-status", "score_equivalent", "--diagnostic-only",
         "--raw-feature-parity-evidence", "x.json"],
    ):
        with pytest.raises(SystemExit) as error:
            us_daily.main(["prepare", "--as-of", "2026-09-25", *extra])
        assert error.value.code == 2


def test_serving_defaults_to_rule_a_and_research_helpers_default_to_legacy() -> None:
    import inspect

    assert us_daily.SERVING_FINANCIAL_RULE == RULE_DET_A
    assert "F8_payout" in us_daily.FINANCIAL_FAMILIES
    assert set(us_daily.RULE_GOVERNED_FEATURES) == {*us_daily.FINANCIAL_FEATURES, "buyback_yield"}
    for fn in (us_daily.build_daily_features, us_daily.build_daily_features_for_dates):
        assert inspect.signature(fn).parameters["financial_selection_rule"].default == RULE_DET_A
    from modeler.us.features import investment, payout, profitability, valuation

    for fn in (valuation.add_valuation, profitability.add_profitability, investment.add_investment,
               payout.add_payout):
        assert inspect.signature(fn).parameters["selection_rule"].default == RULE_LEGACY
    for fn in (flow_ttm, instant_latest, instant_yoy_pair, market_cap):
        assert inspect.signature(fn).parameters["rule"].default == RULE_LEGACY


class _ActionsLake:
    def __init__(self, actions: pl.DataFrame):
        self.actions = actions

    def scan(self, table: str) -> pl.LazyFrame:
        assert table == "corp_actions"
        return self.actions.lazy()


_DIV_AMOUNTS = [0.1, 0.7, 1e16, 0.2, -1e16, 0.3, 1e-3, 0.05, 123456.789, 0.01]


def _dividend_sum(order: list[int], rule: str | None) -> float:
    from modeler.us.features.payout import _dividend_ttm

    actions = pl.DataFrame(
        {
            "symbol": ["ABC"] * len(_DIV_AMOUNTS),
            "kind": ["dividend"] * len(_DIV_AMOUNTS),
            "ex_date": [date(2020, 6, 1 + i) for i in range(len(_DIV_AMOUNTS))],
            "amount": _DIV_AMOUNTS,
        }
    ).select(pl.all().gather(order))
    lake = _ActionsLake(actions)
    out = _dividend_ttm(PANEL, lake) if rule is None else _dividend_ttm(PANEL, lake, rule)
    return out["div_sum"].item()


def test_det_a_dividend_sum_is_bitwise_independent_of_input_row_order() -> None:
    rng = random.Random(7)
    base = list(range(len(_DIV_AMOUNTS)))
    expected = _dividend_sum(base, RULE_DET_A)
    for _ in range(20):
        order = base[:]
        rng.shuffle(order)
        assert _dividend_sum(order, RULE_DET_A).hex() == expected.hex()


def test_legacy_dividend_sum_default_is_unsorted_research_path() -> None:
    rng = random.Random(7)
    base = list(range(len(_DIV_AMOUNTS)))
    det = _dividend_sum(base, RULE_DET_A)
    seen = set()
    for _ in range(20):
        order = base[:]
        rng.shuffle(order)
        default = _dividend_sum(order, None)
        assert default.hex() == _dividend_sum(order, RULE_LEGACY).hex()
        seen.add(default.hex())
    # legacy는 정렬하지 않으므로 입력 순서에 따라 값이 달라진다(연구 경로 동작 불변).
    assert len(seen) > 1 or det.hex() not in seen
