"""One listed line per company and date (``etl/one_per_company.py``): the SQL and
polars forms on small hand-made frames, pinned to give the same answer."""

from __future__ import annotations

import datetime

import duckdb
import polars as pl
import pytest

from modeler.etl.one_per_company import (
    assert_one_per_company_inputs_unique,
    build_one_per_company_sql,
    with_one_per_company,
)

D1 = datetime.date(2020, 1, 2)
D2 = datetime.date(2020, 1, 3)


def _universe() -> pl.DataFrame:
    rows = [
        # (date, ticker, in_universe_v2)
        (D1, "005930", True),  # common + preferred, both in    -> common kept
        (D1, "005935", True),
        (D1, "000150", False),  # common out, preferred in       -> preferred kept
        (D1, "000157", True),
        (D1, "00680K", True),  # two preferred, no common line  -> larger trading value
        (D1, "006805", True),
        (D1, "999990", True),  # unmapped                       -> kept + flag
        (D1, "99999K", True),  # unmapped, a different "company" with same prefix rule
        (D1, "111110", None),  # NULL in_universe               -> never kept
        (D2, "005930", False),  # common out on D2, preferred in -> preferred kept
        (D2, "005935", True),
        (D2, "000010", True),  # single line                    -> kept
    ]
    return pl.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "ticker": [r[1] for r in rows],
            "market": ["KOSPI"] * len(rows),
            "in_universe_v2": [r[2] for r in rows],
        }
    )


def _corp() -> pl.DataFrame:
    # Common tickers only, as in dart_corp_master. 000010 is a delisted corp.
    return pl.DataFrame(
        {
            "ticker": ["005930", "000150", "006800", "000010", "", None],
            "corp_code": ["00126380", "00111111", "00222222", "00333333", "00999999", "00888888"],
        }
    )


def _market_cap() -> pl.DataFrame:
    rows = [
        (D1, "00680K", 100),
        (D1, "006805", 900),  # larger -> wins among the two preferred
        (D1, "005935", 10**12),  # huge preferred must still lose to the common
        (D1, "005930", 5),
    ]
    return pl.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "ticker": [r[1] for r in rows],
            "trading_value": [r[2] for r in rows],
        }
    )


def _run_sql(universe: pl.DataFrame, corp: pl.DataFrame, mcap: pl.DataFrame) -> pl.DataFrame:
    con = duckdb.connect()
    con.register("dim_u", universe)
    con.register("dart_corp_master", corp)
    con.register("daily_market_cap", mcap)
    return con.execute(build_one_per_company_sql("dim_u")).pl()


def _keep(df: pl.DataFrame) -> dict[tuple[datetime.date, str], bool]:
    return {
        (r["trade_date"], r["ticker"]): r["keep_one_per_company"] for r in df.iter_rows(named=True)
    }


def test_common_preferred_over_preferred_even_with_smaller_trading_value() -> None:
    out = _run_sql(_universe(), _corp(), _market_cap())
    k = _keep(out)
    assert k[(D1, "005930")] is True
    assert k[(D1, "005935")] is False


def test_preferred_kept_when_common_is_not_in_universe() -> None:
    k = _keep(_run_sql(_universe(), _corp(), _market_cap()))
    assert k[(D1, "000150")] is False  # not in the universe: never kept
    assert k[(D1, "000157")] is True
    assert k[(D2, "005930")] is False
    assert k[(D2, "005935")] is True


def test_two_classes_without_common_keep_the_larger_trading_value() -> None:
    k = _keep(_run_sql(_universe(), _corp(), _market_cap()))
    assert k[(D1, "006805")] is True
    assert k[(D1, "00680K")] is False


def test_tie_on_trading_value_goes_to_the_smaller_ticker() -> None:
    mcap = pl.DataFrame(
        {"trade_date": [D1, D1], "ticker": ["00680K", "006805"], "trading_value": [7, 7]}
    )
    k = _keep(_run_sql(_universe(), _corp(), mcap))
    assert k[(D1, "006805")] is True  # '006805' < '00680K'
    assert k[(D1, "00680K")] is False


def test_missing_trading_value_loses_to_a_known_one() -> None:
    mcap = pl.DataFrame({"trade_date": [D1], "ticker": ["00680K"], "trading_value": [1]})
    k = _keep(_run_sql(_universe(), _corp(), mcap))
    assert k[(D1, "00680K")] is True
    assert k[(D1, "006805")] is False


def test_unmapped_tickers_are_kept_and_flagged() -> None:
    out = _run_sql(_universe(), _corp(), _market_cap())
    rows = {(r["trade_date"], r["ticker"]): r for r in out.iter_rows(named=True)}
    for t in ("999990", "99999K"):
        assert rows[(D1, t)]["keep_one_per_company"] is True
        assert rows[(D1, t)]["corp_code_missing"] is True
        assert rows[(D1, t)]["corp_code"] is None
    assert rows[(D1, "005935")]["corp_code_missing"] is False
    assert rows[(D1, "005935")]["corp_code"] == "00126380"
    assert rows[(D1, "005935")]["is_common_stock"] is False
    assert rows[(D1, "005930")]["is_common_stock"] is True
    # blank / NULL tickers of the corp master are ignored, not matched
    assert out["corp_code"].drop_nulls().is_in(["00999999", "00888888"]).sum() == 0


def test_null_universe_flag_is_never_kept_and_one_row_per_date_and_corp() -> None:
    out = _run_sql(_universe(), _corp(), _market_cap())
    assert _keep(out)[(D1, "111110")] is False
    assert out.height == _universe().height  # never multiplied
    kept = out.filter(pl.col("keep_one_per_company") & ~pl.col("corp_code_missing"))
    assert kept.group_by("trade_date", "corp_code").len()["len"].max() == 1
    # every (date, corp) with an in-universe line keeps exactly one
    live = (
        out.filter(pl.col("in_universe_v2").fill_null(False) & ~pl.col("corp_code_missing"))
        .select("trade_date", "corp_code")
        .unique()
    )
    assert live.height == kept.height
    # kept rows are always in the universe
    assert (
        out.filter(
            pl.col("keep_one_per_company") & ~pl.col("in_universe_v2").fill_null(False)
        ).height
        == 0
    )


def test_trailing_window_uses_the_last_20_rows_of_the_ticker() -> None:
    days = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(25)]
    mcap = pl.DataFrame(
        {
            "trade_date": days * 2,
            "ticker": ["00680K"] * 25 + ["006805"] * 25,
            # 00680K: huge early, then small; 006805: steady. (huge days leave the window)
            "trading_value": [1000] * 5 + [1] * 20 + [50] * 25,
        }
    )
    uni = pl.DataFrame(
        {
            "trade_date": [days[0], days[24]] * 2,
            "ticker": ["00680K", "00680K", "006805", "006805"],
            "market": ["KOSPI"] * 4,
            "in_universe_v2": [True] * 4,
        }
    )
    k = _keep(_run_sql(uni, _corp(), mcap))
    assert k[(days[0], "00680K")] is True  # one row of 1000 vs 50
    assert k[(days[24], "00680K")] is False  # mean of last 20 = 1 vs 50
    assert k[(days[24], "006805")] is True


def test_polars_form_equals_sql_form_and_keeps_row_order() -> None:
    uni, corp, mcap = _universe(), _corp(), _market_cap()
    sql = _run_sql(uni, corp, mcap)
    pol = with_one_per_company(uni, corp, mcap)
    assert pol["ticker"].to_list() == uni["ticker"].to_list()
    cols = sorted(sql.columns)
    assert sorted(pol.columns) == cols
    key = ["trade_date", "ticker", "market"]
    a = sql.select(cols).sort(key)
    b = pol.select(cols).sort(key)
    assert a.equals(b)
    lazy = with_one_per_company(uni.lazy(), corp.lazy(), mcap.lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(pol)


def test_polars_form_rejects_existing_columns() -> None:
    uni = _universe().with_columns(pl.lit(1).alias("corp_code"))
    with pytest.raises(ValueError, match="corp_code"):
        with_one_per_company(uni, _corp(), _market_cap())


def test_bad_identifiers_and_window_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_one_per_company_sql("u; drop table x")
    with pytest.raises(ValueError):
        build_one_per_company_sql("u", corp_view="a b")
    with pytest.raises(ValueError):
        build_one_per_company_sql("u", window=0)


def test_input_uniqueness_check_fails_closed() -> None:
    con = duckdb.connect()
    con.register("dim_u", _universe())
    con.register("daily_market_cap", _market_cap())
    assert_one_per_company_inputs_unique(con, "dim_u")
    con.register("daily_market_cap", pl.concat([_market_cap(), _market_cap()]))
    with pytest.raises(AssertionError, match="duplicate"):
        assert_one_per_company_inputs_unique(con, "dim_u")
