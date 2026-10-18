"""Unit tests for the next KR model's feature sets (C stage 2, 17_c2_feature_set.md).

The lists are a preregistration, so these tests are what stops one from drifting: the counts,
the containment of the no-flow set, the serving map covering exactly the 47 serving columns,
and every column being produced by the mart the table says it comes from. The flow group reads
KRX raw values (17 §8.1: ``to_listed`` and ``to_volume_v2``), never the retro-adjusted Naver
volume or the stale DART float.
"""

from __future__ import annotations

import re

from modeler.etl.features.filing_activity import (
    FILING_ACTIVITY_TABLE,
    build_filing_activity_sql,
)
from modeler.etl.features.fin_v2 import FIN_V2_TABLE, build_fin_v2_sql
from modeler.etl.features.flow_v2 import FLOW_V2_COLUMNS, FLOW_V2_TABLE
from modeler.etl.features.market_cap import MARKET_CAP_TABLE, build_market_cap_sql
from modeler.etl.features.price import PRICE_TABLE, build_price_sql
from modeler.etl.features.price_v2 import PRICE_V2_COLUMNS, PRICE_V2_TABLE
from modeler.models._02_updown_prob.features import feature_columns
from modeler.models._03_kr_next import feature_set as fs
from modeler.serving.kr_model import EXCLUDED_FEATURES


def _serving_cols() -> tuple[str, ...]:
    """What the serving model reads: FS1h at h=20 minus the blocked short-balance columns."""
    return tuple(c for c in feature_columns("FS1h", 20) if c not in EXCLUDED_FEATURES)


# --- counts and containment ---------------------------------------------------


def test_group_and_set_counts() -> None:
    assert [len(g) for g in (fs.PRICE_COLS, fs.FLOW_COLS, fs.FIN_COLS, fs.FILING_COLS)] == [
        17,
        13,
        10,
        3,
    ]
    assert len(fs.MAIN_COLS) == 43
    assert len(fs.NOFLOW_COLS) == 30


def test_no_set_repeats_a_column() -> None:
    assert len(set(fs.MAIN_COLS)) == len(fs.MAIN_COLS)
    assert len(set(fs.NOFLOW_COLS)) == len(fs.NOFLOW_COLS)


def test_noflow_is_main_without_exactly_the_flow_columns() -> None:
    assert set(fs.NOFLOW_COLS) < set(fs.MAIN_COLS)
    assert set(fs.MAIN_COLS) - set(fs.NOFLOW_COLS) == set(fs.FLOW_COLS)
    assert all(c.startswith("flow_") for c in fs.FLOW_COLS)
    assert not any(c.startswith("flow_") for c in fs.NOFLOW_COLS)


def test_flow_group_is_pinned_to_krx_raw_denominators() -> None:
    """17 §8.1: C2-7 and C2-8 swapped the denominators; a drift back would reintroduce the leak."""
    assert fs.FLOW_COLS == (
        "flow_foreign_netbuy_to_listed_5d_v2",
        "flow_foreign_netbuy_to_listed_20d_v2",
        "flow_inst_netbuy_to_listed_5d_v2",
        "flow_inst_netbuy_to_listed_20d_v2",
        "flow_indiv_netbuy_to_listed_5d_v2",
        "flow_indiv_netbuy_to_listed_20d_v2",
        "flow_foreign_netbuy_z_20d_v2",
        "flow_inst_netbuy_z_20d_v2",
        "flow_short_avg_price_v2",
        "flow_short_selling_volume_v2",
        "flow_short_selling_value_v2",
        "flow_individual_netbuy_to_volume_5d_v2",
        "flow_individual_netbuy_to_volume_20d_v2",
    )
    assert not any("_to_float_" in c for c in fs.MAIN_COLS)
    # the legacy feat_flow volume ratios (Naver adjusted denominator) are gone from the set
    for legacy in ("flow_individual_netbuy_to_volume_5d", "flow_individual_netbuy_to_volume_20d"):
        assert legacy not in fs.MAIN_COLS
        assert legacy not in fs.COLUMN_MART


def test_to_float_columns_stay_in_the_mart_for_the_record_only() -> None:
    record_only = [c for c in FLOW_V2_COLUMNS if "_to_float_" in c]
    assert len(record_only) == 6
    assert not set(record_only) & set(fs.MAIN_COLS)
    assert not set(record_only) & set(fs.COLUMN_MART)


def test_group_prefixes() -> None:
    assert all(c.startswith("px_") for c in fs.PRICE_COLS)
    assert all(c.startswith(("fin_", "mcap_")) for c in fs.FIN_COLS)
    assert all(c.startswith(("ev_", "own_")) for c in fs.FILING_COLS)


# --- serving map --------------------------------------------------------------


def test_serving_columns_are_47() -> None:
    assert len(_serving_cols()) == 47


def test_serving_map_covers_exactly_the_serving_columns() -> None:
    assert set(fs.SERVING_MAP) == set(_serving_cols())
    assert len(fs.SERVING_MAP) == 47


def test_serving_map_replacements_are_exactly_main() -> None:
    replaced = [c for cols in fs.SERVING_MAP.values() if cols for c in cols]
    # one-to-one: each main column is the replacement of exactly one serving column
    assert len(replaced) == len(set(replaced)) == 43
    assert set(replaced) == set(fs.MAIN_COLS)


def test_dropped_columns_are_the_none_entries_and_not_in_main() -> None:
    assert {c for c, v in fs.SERVING_MAP.items() if v is None} == set(fs.DROPPED)
    assert set(fs.DROPPED) == {
        "px_is_halted",
        "flow_foreign_holding_chg_5d",
        "flow_foreign_holding_chg_20d",
        "fin_has_fs",
    }
    assert not set(fs.DROPPED) & set(fs.MAIN_COLS)
    assert all(fs.DROPPED.values())


def test_serving_map_flow_replacements_follow_17_section_8_1() -> None:
    m = fs.SERVING_MAP
    for who in ("foreign", "inst", "indiv"):
        for w in ("5d", "20d"):
            assert m[f"flow_{who}_netbuy_sum_{w}"] == (f"flow_{who}_netbuy_to_listed_{w}_v2",)
    for w in ("5d", "20d"):
        assert m[f"flow_individual_netbuy_to_volume_{w}"] == (
            f"flow_individual_netbuy_to_volume_{w}_v2",
        )


def test_serving_blocked_columns_never_enter() -> None:
    assert not set(EXCLUDED_FEATURES) & set(fs.SERVING_MAP)
    assert not set(EXCLUDED_FEATURES) & set(fs.MAIN_COLS)


# --- every column is produced by its mart --------------------------------------


def _as_names(sql: str) -> set[str]:
    return set(re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql))


def test_mart_columns_partition_main() -> None:
    flat = [c for cols in fs.MART_COLUMNS.values() for c in cols]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(fs.MAIN_COLS)
    assert set(fs.COLUMN_MART) == set(fs.MAIN_COLS)


def test_mart_names_are_the_real_table_names() -> None:
    assert set(fs.MART_COLUMNS) == {
        PRICE_V2_TABLE,
        PRICE_TABLE,
        FLOW_V2_TABLE,
        FIN_V2_TABLE,
        MARKET_CAP_TABLE,
        FILING_ACTIVITY_TABLE,
    }


def test_v2_price_and_flow_columns_come_from_the_exported_tuples() -> None:
    for col in fs.MART_COLUMNS[PRICE_V2_TABLE]:
        assert col in PRICE_V2_COLUMNS, col
    for col in fs.MART_COLUMNS[FLOW_V2_TABLE]:
        assert col in FLOW_V2_COLUMNS, col


def test_fin_v2_columns_are_selected_by_the_builder() -> None:
    # fin_v2 exports no column tuple; its final SELECT names every column with ``AS``.
    names = _as_names(build_fin_v2_sql())
    for col in fs.MART_COLUMNS[FIN_V2_TABLE]:
        assert col in names, col


def test_legacy_mart_columns_are_selected_by_their_builders() -> None:
    builders = {
        PRICE_TABLE: build_price_sql("daily_ohlcv"),
        MARKET_CAP_TABLE: build_market_cap_sql(),
        FILING_ACTIVITY_TABLE: build_filing_activity_sql(),
    }
    for mart, sql in builders.items():
        names = _as_names(sql)
        for col in fs.MART_COLUMNS[mart]:
            assert col in names, f"{col} not produced by {mart} builder"


# --- hypothesis and benchmark ---------------------------------------------------


def test_hypothesis_columns_are_inside_main() -> None:
    assert fs.HYPOTHESIS_COLS == {
        "tail_vol": ("px_idio_vol_60d", "px_vol_20d_v2"),
        "tail_loss": ("fin_roa_v2", "fin_capital_impaired_v2"),
    }
    for cols in fs.HYPOTHESIS_COLS.values():
        assert set(cols) <= set(fs.MAIN_COLS)
        assert set(cols) <= set(fs.NOFLOW_COLS), "the hypotheses must not need flow (15 §10.4)"


def test_fixed_benchmark() -> None:
    assert fs.FIXED_BENCHMARK == ("k200", "코스피 200")
