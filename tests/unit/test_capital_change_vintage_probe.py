"""§4.4.1 probe runner — the pre-registered threshold table and the report shape."""

from __future__ import annotations

from datetime import date

import duckdb

from modeler.analysis.capital_change_vintage_probe import decide, run


def _distance(distance: int, *, windows: int, changed: int) -> dict[str, object]:
    return {
        "vintage_distance_years": distance,
        "tickers": 1,
        "compared_windows": windows,
        "changed_windows": changed,
        "feature_changing_rate": changed / windows if windows else None,
    }


def _identity(latest: float, strict: float) -> list[dict[str, object]]:
    return [
        {"vintage_policy": "latest_vintage", "feature_available_rate": latest},
        {"vintage_policy": "strict_pit", "feature_available_rate": strict},
    ]


def test_rate_below_one_percent_adopts_latest_vintage() -> None:
    verdict = decide(
        distance_rows=[_distance(9, windows=1000, changed=5)],
        identity_rows=_identity(0.8, 0.7),
    )
    assert verdict["decision"] == "latest_vintage"


def test_rate_between_one_and_five_percent_adds_a_quality_flag() -> None:
    verdict = decide(
        distance_rows=[_distance(9, windows=1000, changed=30)],
        identity_rows=_identity(0.8, 0.7),
    )
    assert verdict["decision"] == "latest_vintage_with_quality_flag"


def test_rate_above_five_percent_adopts_strict_pit() -> None:
    verdict = decide(
        distance_rows=[_distance(9, windows=1000, changed=80)],
        identity_rows=_identity(0.8, 0.7),
    )
    assert verdict["decision"] == "strict_pit"


def test_coverage_priority_rule_overrides_a_high_disagreement_rate() -> None:
    # strict_pit keeps 0.3 of 0.8 -- below the half-of-latest floor, so it stays
    # a sensitivity analysis no matter how much the vintages disagree.
    verdict = decide(
        distance_rows=[_distance(9, windows=1000, changed=800)],
        identity_rows=_identity(0.8, 0.3),
    )
    assert verdict["decision"] == "latest_vintage"
    assert "priority rule" in verdict["rule"]


def test_verdict_reads_the_largest_distance_not_the_nearest() -> None:
    verdict = decide(
        distance_rows=[
            _distance(1, windows=1000, changed=0),
            _distance(9, windows=1000, changed=80),
        ],
        identity_rows=_identity(0.8, 0.7),
    )
    assert verdict["max_distance_years"] == 9
    assert verdict["decision"] == "strict_pit"


def test_no_second_vintage_is_inconclusive() -> None:
    verdict = decide(distance_rows=[], identity_rows=_identity(0.8, 0.0))
    assert verdict["decision"] == "inconclusive"
    assert verdict["rate_at_max_distance"] is None


def test_run_produces_all_three_sections_on_a_synthetic_lake() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dart_corp_master (ticker VARCHAR, market VARCHAR, corp_code VARCHAR)")
    con.execute(
        "CREATE TABLE dart_share_count_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, se VARCHAR, istc_totqy BIGINT, tesstk_co BIGINT, "
        "now_to_isu_stock_totqy BIGINT, now_to_dcrs_stock_totqy BIGINT, stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_capital_change_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, isu_dcrs_de DATE, isu_dcrs_stle VARCHAR, "
        "isu_dcrs_stock_knd VARCHAR, isu_dcrs_qy BIGINT)"
    )
    con.execute("INSERT INTO dart_corp_master VALUES ('000001','KOSPI','corp1')")
    for year, issued, now_to_isu in ((2021, 1000, 500), (2022, 1100, 600), (2023, 1100, 600)):
        con.execute(
            "INSERT INTO dart_share_count_raw VALUES "
            "('corp1','000001',?,'11011',?,'합계',?,0,?,0,?)",
            [year, f"{year + 1}0310000001", issued, now_to_isu, date(year, 12, 31)],
        )
    for bsns_year, rcept_no in ((2022, "20230310000002"), (2023, "20240310000003")):
        con.execute(
            "INSERT INTO dart_capital_change_raw VALUES "
            "('corp1','000001',?,'11011',?,?,'유상증자(일반공모)','보통주',100)",
            [bsns_year, rcept_no, date(2022, 6, 15)],
        )

    trading_days = [date(2023, 3, 13), date(2024, 3, 11), date(2024, 3, 12)]
    payload = run(con, trading_days=trading_days)

    assert payload["distance_summary"][0]["changed_windows"] == 0
    assert payload["row_summary"][0]["old_events_absent_from_newest"] == 0
    assert {row["vintage_policy"] for row in payload["identity_pass_rate"]} == {
        "latest_vintage",
        "strict_pit",
    }
    assert payload["verdict"]["decision"] == "latest_vintage"
