"""Synthetic tests of the delisted-keeping vintage marts (``metric_vintages_delisted``).

15_quality_fix_plan.md §10.2/§10.4 (E1): a delisted corp has ``market = NULL`` in
``dart_corp_master`` and drops out of ``stock_metric_vintage_fact``. The ``_dl`` variant
takes ``market`` from ``stock_master`` and ``daily_ohlcv`` when the corp master has none.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import duckdb
import pytest

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.marts import metric_vintages as mv
from modeler.etl.marts.financial_quarters import (
    FQMV_TABLE,
    register_fin_quarterly_metric_vintage_view,
)
from modeler.etl.marts.metric_vintages import SMVF_TABLE, register_stock_metric_vintage_fact_view
from modeler.etl.marts.metric_vintages_delisted import (
    FQMV_DL_TABLE,
    SMVF_DL_TABLE,
    CorpCteDriftError,
    OriginalSqlDriftError,
    build_fin_quarterly_metric_vintage_dl_sql,
    build_stock_metric_vintage_fact_dl_plan,
    build_stock_metric_vintage_fact_dl_sql,
    corp_relation_sql,
    materialize_fin_quarterly_metric_vintage_dl,
    materialize_stock_metric_vintage_fact_dl,
    register_fin_quarterly_metric_vintage_dl_view,
    register_stock_metric_vintage_fact_dl_view,
)


def _weekdays(year: int) -> list[date]:
    out = []
    d = date(year, 1, 1)
    while d.year == year:
        if d.weekday() < 5:
            out.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    return out


_TRADING_DAYS = _weekdays(2023) + _weekdays(2024)

ACTIVE = "005930"  # corp master has its market
KOSDAQ_ACTIVE = "035720"
DELISTED = "900001"  # corp master market NULL, stock_master DELISTED
TRANSFER = "900002"  # corp master market NULL; KOSDAQ DELISTED (2018) then KOSPI ACTIVE (2026)
OHLCV_ONLY = "900003"  # corp master market NULL, not in stock_master, prices on two markets
NO_MARKET = "900004"  # corp master market NULL and no other source
NO_CORP = "900005"  # raw rows but not in the corp master at all
BLANK_SM = "900006"  # stock_master market is blank -> falls through to the prices

_CORP_CODE = {
    ACTIVE: "00126380",
    KOSDAQ_ACTIVE: "00258801",
    DELISTED: "00999001",
    TRANSFER: "00999002",
    OHLCV_ONLY: "00999003",
    NO_MARKET: "00999004",
    NO_CORP: "00999005",
    BLANK_SM: "00999006",
}


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dart_corp_master (ticker VARCHAR, market VARCHAR, corp_code VARCHAR)")
    con.execute(
        "CREATE TABLE dart_financial_statement_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "fs_div VARCHAR, sj_div VARCHAR, account_id VARCHAR, account_nm VARCHAR, "
        "ord INTEGER, thstrm_amount DECIMAL(30,4), currency VARCHAR, rcept_no VARCHAR, "
        "thstrm_add_amount DECIMAL(30,4), frmtrm_q_amount DECIMAL(30,4))"
    )
    con.execute(
        "CREATE TABLE dart_share_count_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, se VARCHAR, istc_totqy BIGINT, tesstk_co BIGINT, stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_shareholder_return_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, statement_type VARCHAR, row_name VARCHAR, stock_knd VARCHAR, "
        "dim1 VARCHAR, dim2 VARCHAR, dim3 VARCHAR, metric_code VARCHAR, "
        "value_numeric DECIMAL(30,4), stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_xbrl_fact_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, concept_id VARCHAR, concept_name VARCHAR, context_id VARCHAR, "
        "context_type VARCHAR, period_start DATE, period_end DATE, instant_date DATE, "
        "dimensions VARCHAR, value_numeric DECIMAL(30,4), label_ko VARCHAR)"
    )
    con.execute(
        "CREATE TABLE dart_filing_receipt_raw "
        "(corp_code VARCHAR, rcept_no VARCHAR, report_nm VARCHAR)"
    )
    con.execute(
        "CREATE TABLE stock_master (ticker VARCHAR, market VARCHAR, name VARCHAR, "
        "status VARCHAR, last_seen_date DATE, source VARCHAR, "
        "updated_at TIMESTAMP WITH TIME ZONE, listing_date DATE, first_seen_date DATE)"
    )
    con.execute(
        "CREATE TABLE daily_ohlcv (ticker VARCHAR, market VARCHAR, trade_date DATE, close DOUBLE)"
    )
    return con


def _corp(con: duckdb.DuckDBPyConnection, ticker: str, market: str | None) -> None:
    con.execute(
        "INSERT INTO dart_corp_master VALUES (?, ?, ?)", [ticker, market, _CORP_CODE[ticker]]
    )


def _sm(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    market: str,
    status: str,
    last_seen: date | None,
    updated: datetime = datetime(2026, 9, 1, 12, 0),
) -> None:
    con.execute(
        "INSERT INTO stock_master VALUES (?, ?, 'x', ?, ?, 'krx', ?, NULL, NULL)",
        [ticker, market, status, last_seen, updated],
    )


def _px(con: duckdb.DuckDBPyConnection, ticker: str, market: str, trade_date: date) -> None:
    con.execute("INSERT INTO daily_ohlcv VALUES (?, ?, ?, 1.0)", [ticker, market, trade_date])


def _fs(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    bsns_year: int,
    reprt_code: str,
    amount: float,
    rcept_no: str,
    add: float | None = None,
    account_id: str = "ifrs-full_Revenue",
    sj_div: str = "IS",
) -> None:
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, account_nm, "
        "ord, thstrm_amount, currency, rcept_no, thstrm_add_amount) "
        "VALUES (?, ?, ?, ?, 'CFS', ?, ?, '', 1, ?, 'KRW', ?, ?)",
        [
            _CORP_CODE[ticker],
            ticker,
            bsns_year,
            reprt_code,
            sj_div,
            account_id,
            amount,
            rcept_no,
            add,
        ],
    )


def _four_reports(con: duckdb.DuckDBPyConnection, ticker: str, tag: int) -> None:
    """Revenue Q1=100, half=150 (cum 250), Q3=140 (cum 390), annual=560 -> TTM 560."""
    _fs(con, ticker, bsns_year=2023, reprt_code="11013", amount=100, rcept_no=f"20230410{tag:06d}")
    _fs(
        con, ticker, bsns_year=2023, reprt_code="11012", amount=150, add=250,
        rcept_no=f"20230810{tag:06d}",
    )  # fmt: skip
    _fs(
        con, ticker, bsns_year=2023, reprt_code="11014", amount=140, add=390,
        rcept_no=f"20231110{tag:06d}",
    )  # fmt: skip
    _fs(con, ticker, bsns_year=2023, reprt_code="11011", amount=560, rcept_no=f"20240315{tag:06d}")
    # an issued-shares row (dart_share_count_raw) for the annual report as well
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES (?, ?, 2023, '11011', ?, '합계', 5000, 10, "
        "DATE '2023-12-31')",
        [_CORP_CODE[ticker], ticker, f"20240315{tag:06d}"],
    )


def _fixture() -> duckdb.DuckDBPyConnection:
    con = _con()
    # --- corp master -------------------------------------------------------------------
    _corp(con, ACTIVE, "KOSPI")
    _corp(con, KOSDAQ_ACTIVE, "KOSDAQ")
    for t in (DELISTED, TRANSFER, OHLCV_ONLY, NO_MARKET, BLANK_SM):
        _corp(con, t, None)
    # --- stock master ------------------------------------------------------------------
    _sm(con, ACTIVE, "KOSPI", "ACTIVE", date(2026, 9, 28))
    _sm(con, KOSDAQ_ACTIVE, "KOSDAQ", "ACTIVE", date(2026, 9, 28))
    _sm(con, DELISTED, "KOSDAQ", "DELISTED", date(2021, 3, 31))
    _sm(con, TRANSFER, "KOSDAQ", "DELISTED", date(2018, 12, 31))
    _sm(con, TRANSFER, "KOSPI", "ACTIVE", date(2026, 9, 28))
    _sm(con, BLANK_SM, "", "DELISTED", date(2020, 1, 31))
    # --- prices ------------------------------------------------------------------------
    for t, m in (
        (ACTIVE, "KOSPI"), (KOSDAQ_ACTIVE, "KOSDAQ"), (DELISTED, "KOSDAQ"),
        (TRANSFER, "KOSPI"), (BLANK_SM, "KOSDAQ"),
    ):  # fmt: skip
        _px(con, t, m, date(2023, 6, 1))
    _px(con, OHLCV_ONLY, "KOSDAQ", date(2019, 5, 2))
    _px(con, OHLCV_ONLY, "KOSPI", date(2022, 5, 2))  # moved later -> the later row wins
    # --- raw DART rows -----------------------------------------------------------------
    for i, t in enumerate(
        (ACTIVE, KOSDAQ_ACTIVE, DELISTED, TRANSFER, OHLCV_ONLY, NO_MARKET, NO_CORP, BLANK_SM), 1
    ):
        _four_reports(con, t, i)
    # a revision of the active corp's annual report, so the lineage columns differ per row
    _fs(con, ACTIVE, bsns_year=2023, reprt_code="11011", amount=570, rcept_no="20240401000001")
    con.execute(
        "INSERT INTO dart_filing_receipt_raw VALUES (?, '20240401000001', '[정정]사업보고서')",
        [_CORP_CODE[ACTIVE]],
    )
    return con


def _register_both(con: duckdb.DuckDBPyConnection) -> None:
    register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS)
    register_stock_metric_vintage_fact_dl_view(con, trading_days=_TRADING_DAYS)


def _markets(con: duckdb.DuckDBPyConnection, view: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for ticker, market in con.execute(f"SELECT DISTINCT ticker, market FROM {view}").fetchall():
        out.setdefault(ticker, set()).add(market)
    return out


# --- the corp relation ------------------------------------------------------------------


def test_delisted_corp_appears_only_in_the_dl_mart() -> None:
    con = _fixture()
    _register_both(con)

    original = _markets(con, SMVF_TABLE)
    variant = _markets(con, SMVF_DL_TABLE)

    assert set(original) == {ACTIVE, KOSDAQ_ACTIVE}
    assert DELISTED not in original
    assert variant[DELISTED] == {"KOSDAQ"}  # stock_master row, status DELISTED


def test_active_corp_rows_are_identical_to_the_original_builder() -> None:
    con = _fixture()
    _register_both(con)

    active = f"('{ACTIVE}', '{KOSDAQ_ACTIVE}')"
    original = con.execute(f"SELECT * FROM {SMVF_TABLE} ORDER BY ALL").fetchall()
    variant = con.execute(
        f"SELECT * FROM {SMVF_DL_TABLE} WHERE ticker IN {active} ORDER BY ALL"
    ).fetchall()

    assert len(original) >= 10  # the fixture is not trivially empty
    assert any(row[9] == "20240401000001" for row in original)  # the revision row is in it
    assert variant == original
    # same columns in the same order, not just the same values
    assert (
        con.execute(f"DESCRIBE {SMVF_DL_TABLE}").fetchall()
        == con.execute(f"DESCRIBE {SMVF_TABLE}").fetchall()
    )


def test_dl_mart_is_the_original_plus_only_the_added_tickers() -> None:
    con = _fixture()
    _register_both(con)

    extra = con.execute(
        f"SELECT DISTINCT ticker FROM {SMVF_DL_TABLE} EXCEPT SELECT ticker FROM {SMVF_TABLE}"
    ).fetchall()
    assert {r[0] for r in extra} == {DELISTED, TRANSFER, OHLCV_ONLY, BLANK_SM}
    # every original row is still there (row-for-row, no loss)
    lost = con.execute(
        f"SELECT count(*) FROM (SELECT * FROM {SMVF_TABLE} EXCEPT SELECT * FROM {SMVF_DL_TABLE})"
    ).fetchone()
    assert lost == (0,)


def test_market_transfer_ticker_takes_the_later_stock_master_row() -> None:
    con = _fixture()
    _register_both(con)

    assert _markets(con, SMVF_DL_TABLE)[TRANSFER] == {"KOSPI"}  # not the 2018 KOSDAQ row
    # and the two stock_master rows did not double the ticker's rows
    per_metric = con.execute(
        f"SELECT metric_code, statement_period_end, fs_basis, rcept_no, count(*) "
        f"FROM {SMVF_DL_TABLE} WHERE ticker = '{TRANSFER}' GROUP BY ALL HAVING count(*) > 1"
    ).fetchall()
    assert per_metric == []
    n_dl = con.execute(f"SELECT count(*) FROM {SMVF_DL_TABLE} WHERE ticker='{TRANSFER}'").fetchone()
    n_delisted = con.execute(
        f"SELECT count(*) FROM {SMVF_DL_TABLE} WHERE ticker='{DELISTED}'"
    ).fetchone()
    assert n_dl == n_delisted  # same filings, same rows


def test_market_falls_back_to_prices_and_blank_stock_master_counts_as_missing() -> None:
    con = _fixture()
    _register_both(con)

    markets = _markets(con, SMVF_DL_TABLE)
    assert markets[OHLCV_ONLY] == {"KOSPI"}  # last price row, not the 2019 KOSDAQ one
    assert markets[BLANK_SM] == {"KOSDAQ"}  # stock_master market '' -> prices


def test_tickers_without_any_market_or_corp_stay_out() -> None:
    con = _fixture()
    _register_both(con)

    markets = _markets(con, SMVF_DL_TABLE)
    assert NO_MARKET not in markets  # no stock_master row, no prices
    assert NO_CORP not in markets  # not in dart_corp_master


def test_corp_master_market_wins_over_stock_master_and_prices() -> None:
    con = _fixture()
    # stock_master and prices say KOSDAQ for the active KOSPI corp: the corp's own value stays
    _sm(con, ACTIVE, "KOSDAQ", "ACTIVE", date(2026, 9, 29))
    _px(con, ACTIVE, "KOSDAQ", date(2024, 1, 2))
    _register_both(con)

    assert _markets(con, SMVF_DL_TABLE)[ACTIVE] == {"KOSPI"}


def test_corp_relation_has_one_row_per_corp_master_row() -> None:
    con = _fixture()
    rel = corp_relation_sql()
    rows = con.execute(f"SELECT ticker, market, corp_code FROM {rel}").fetchall()
    master = con.execute("SELECT count(*) FROM dart_corp_master").fetchone()[0]
    assert len(rows) == master
    by_ticker = {r[0]: r[1] for r in rows}
    assert by_ticker[NO_MARKET] is None
    assert by_ticker[TRANSFER] == "KOSPI"


def test_view_names_are_parameters() -> None:
    con = _fixture()
    con.execute("ALTER TABLE stock_master RENAME TO sm_alt")
    con.execute("ALTER TABLE daily_ohlcv RENAME TO px_alt")
    con.execute("ALTER TABLE dart_corp_master RENAME TO corp_alt")
    register_stock_metric_vintage_fact_dl_view(
        con,
        trading_days=_TRADING_DAYS,
        corp_view="corp_alt",
        stock_master_view="sm_alt",
        ohlcv_view="px_alt",
    )
    assert DELISTED in _markets(con, SMVF_DL_TABLE)


@pytest.mark.parametrize("semantics", ["v1", "v2"])
def test_semantics_follow_the_original(semantics: str) -> None:
    con = _fixture()
    register_stock_metric_vintage_fact_view(
        con, trading_days=_TRADING_DAYS, view_name="orig", semantics=semantics
    )
    register_stock_metric_vintage_fact_dl_view(con, trading_days=_TRADING_DAYS, semantics=semantics)
    active = f"('{ACTIVE}', '{KOSDAQ_ACTIVE}')"
    assert (
        con.execute(
            f"SELECT * FROM {SMVF_DL_TABLE} WHERE ticker IN {active} ORDER BY ALL"
        ).fetchall()
        == con.execute("SELECT * FROM orig ORDER BY ALL").fetchall()
    )


# --- guard against drift of the original text --------------------------------------------


def _drifted(original, old: str, new: str):
    def wrapper(**kw):
        parts, final = original(**kw)
        assert old in parts["corp"]
        return {**parts, "corp": parts["corp"].replace(old, new)}, final

    return wrapper


def test_guard_raises_when_the_original_corp_cte_adds_a_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        mv,
        "_vintage_sql_parts",
        _drifted(
            mv._vintage_sql_parts, "AND market IS NOT NULL", "AND market IS NOT NULL AND is_active"
        ),
    )
    with pytest.raises(CorpCteDriftError, match="corp CTE"):
        build_stock_metric_vintage_fact_dl_sql()


def test_guard_raises_when_the_original_corp_cte_ignores_corp_view(monkeypatch) -> None:
    original = mv._vintage_sql_parts

    def wrapper(**kw):
        parts, final = original(**kw)
        text = parts["corp"].replace(f"FROM {kw['corp_view']}", "FROM dart_corp_master")
        return {**parts, "corp": text}, final

    monkeypatch.setattr(mv, "_vintage_sql_parts", wrapper)
    with pytest.raises(CorpCteDriftError):
        build_stock_metric_vintage_fact_dl_sql()


def test_guard_raises_when_the_staged_plan_loses_the_corp_relation(monkeypatch) -> None:
    original = mv._staged_plan

    def wrapper(semantics, **kw):
        plan = original(semantics, **kw)
        stages = tuple(
            (name, sql.replace(kw["corp_view"], "dart_corp_master")) for name, sql in plan.stages
        )
        import dataclasses

        return dataclasses.replace(plan, stages=stages)

    monkeypatch.setattr(mv, "_staged_plan", wrapper)
    with pytest.raises(CorpCteDriftError, match="stages"):
        build_stock_metric_vintage_fact_dl_plan(plan="staged")


def test_guard_raises_when_the_quarterly_builder_stops_reading_the_view(monkeypatch) -> None:
    from modeler.etl.marts import financial_quarters as fq

    monkeypatch.setattr(
        fq, "build_fin_quarterly_metric_vintage_sql", lambda **kw: "SELECT 1 FROM somewhere\n"
    )
    with pytest.raises(OriginalSqlDriftError, match="fin_quarterly_metric_vintage_dl"):
        build_fin_quarterly_metric_vintage_dl_sql()


def test_unmodified_original_builds_without_error() -> None:
    sql = build_stock_metric_vintage_fact_dl_sql()
    assert sql.count("corp_market_src") == 1  # the relation alias, once
    assert sql.count("market IS NOT NULL") >= 1
    for plan_name in ("single", "staged"):
        plan = build_stock_metric_vintage_fact_dl_plan(plan=plan_name)
        assert dict(plan.metadata)["variant"] == "keep_delisted_corps"


def test_the_original_module_text_is_untouched_by_the_variant() -> None:
    original = mv.build_stock_metric_vintage_fact_sql()
    build_stock_metric_vintage_fact_dl_sql()
    assert mv.build_stock_metric_vintage_fact_sql() == original
    assert "corp_market_src" not in original


# --- downstream ----------------------------------------------------------------------------


def test_quarterly_dl_view_reads_the_dl_fact_and_gives_the_delisted_corp_a_ttm() -> None:
    con = _fixture()
    _register_both(con)
    register_fin_quarterly_metric_vintage_view(con)
    register_fin_quarterly_metric_vintage_dl_view(con)

    sql = build_fin_quarterly_metric_vintage_dl_sql()
    assert "FROM stock_metric_vintage_fact_dl\n" in sql
    assert "FROM stock_metric_vintage_fact\n" not in sql

    def ttm(view: str, ticker: str) -> list[tuple]:
        return con.execute(
            f"SELECT ttm_value, ttm_complete FROM {view} "
            f"WHERE ticker=? AND metric_code='revenue' AND bsns_year=2023 AND quarter='Q4'",
            [ticker],
        ).fetchall()

    assert ttm(FQMV_TABLE, DELISTED) == []  # the original quarterly mart never sees it
    (value, complete) = ttm(FQMV_DL_TABLE, DELISTED)[0]
    assert complete is True
    assert float(value) == 560.0

    # the active corps' quarterly rows are unchanged
    active = f"('{ACTIVE}', '{KOSDAQ_ACTIVE}')"
    assert (
        con.execute(
            f"SELECT * FROM {FQMV_DL_TABLE} WHERE ticker IN {active} ORDER BY ALL"
        ).fetchall()
        == con.execute(f"SELECT * FROM {FQMV_TABLE} ORDER BY ALL").fetchall()
    )


# --- materialisation (parquet cache) ---------------------------------------------------------


def _config(tmp_path) -> LakeConfig:
    return LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-07-30", source="test")


@pytest.mark.parametrize("plan", ["single", "staged"])
def test_materialize_writes_new_names_only_and_records_the_variant(tmp_path, plan: str) -> None:
    con = _fixture()
    config = _config(tmp_path)
    stage_dir = tmp_path / "stages"

    view = materialize_stock_metric_vintage_fact_dl(
        con, config, trading_days=_TRADING_DAYS, plan=plan, stage_dir=stage_dir
    )
    assert view == SMVF_DL_TABLE
    fqmv_view = materialize_fin_quarterly_metric_vintage_dl(con, config)
    assert fqmv_view == FQMV_DL_TABLE

    root = config.feature_mart_root
    assert sorted(p.name for p in root.iterdir()) == sorted([SMVF_DL_TABLE, FQMV_DL_TABLE])
    meta = json.loads((root / SMVF_DL_TABLE / "_cache_metadata.json").read_text())
    assert meta["variant"] == "keep_delisted_corps"
    assert meta["plan"] == plan
    meta_q = json.loads((root / FQMV_DL_TABLE / "_cache_metadata.json").read_text())
    assert meta_q["vintage_view"] == SMVF_DL_TABLE

    assert DELISTED in _markets(con, SMVF_DL_TABLE)
    # a second call reuses the cache (same contract) instead of rebuilding
    before = (root / SMVF_DL_TABLE / "part-000000.parquet").stat().st_mtime_ns
    materialize_stock_metric_vintage_fact_dl(
        con, config, trading_days=_TRADING_DAYS, plan=plan, stage_dir=stage_dir
    )
    assert (root / SMVF_DL_TABLE / "part-000000.parquet").stat().st_mtime_ns == before


def test_staged_and_single_plans_give_the_same_rows(tmp_path) -> None:
    con = _fixture()
    single = materialize_stock_metric_vintage_fact_dl(
        con,
        LakeConfig(root=DataRoot(tmp_path / "a"), snapshot_date="2026-07-30", source="test"),
        trading_days=_TRADING_DAYS,
    )
    rows_single = con.execute(f"SELECT * FROM {single} ORDER BY ALL").fetchall()
    staged = materialize_stock_metric_vintage_fact_dl(
        con,
        LakeConfig(root=DataRoot(tmp_path / "b"), snapshot_date="2026-07-30", source="test"),
        trading_days=_TRADING_DAYS,
        plan="staged",
        stage_dir=tmp_path / "stages",
    )
    rows_staged = con.execute(f"SELECT * FROM {staged} ORDER BY ALL").fetchall()
    assert rows_staged == rows_single
    assert len(rows_single) > 0
