"""Derived views (``views.py``) and the shadow prepare step (``shadow_prepare.py``).

Synthetic raw export only: three tickers over the weekdays of 2024, written as parquet in the
lake layout ``<lake>/raw/raw_postgres/snapshot_date=<d>/source=sj2_remote/<table>/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pytest

from modeler.etl.marts.metric_vintages_delisted import build_fin_quarterly_metric_vintage_dlr_sql
from modeler.etl.stock_pit import PLAN_ASOF_INTERVALS
from modeler.etl.stock_pit_nonempty import build_stock_pit_nonempty_sql
from modeler.etl.universe import UniverseFilter
from modeler.models._03_kr_next import build_panel as bp
from modeler.models._03_kr_next import shadow as sh
from modeler.models._03_kr_next import shadow_prepare as sp
from modeler.models._03_kr_next import views as vw

from .test_kr_next_shadow import SEOUL, bundle_dir  # noqa: F401  (fixture)

SNAPSHOT = "2025-01-01"  # = D of the tests: ``run`` reads snapshot_date=<D>
D = date(2025, 1, 1)
T = date(2024, 12, 31)
SMALL_UNIVERSE = UniverseFilter(
    warmup_window=3,
    warmup_min_valid=1,
    liquidity_window=2,
    min_liquidity_krw=0.0,
    min_close_krw=0.0,
    apply_liquidity_filter=False,
)
TICKERS = {  # ticker -> (market, daily growth, corp_code)
    "000010": ("KOSPI", 0.002, "00000010"),
    "000020": ("KOSDAQ", 0.003, "00000020"),
    "000030": ("KOSDAQ", 0.001, "00000030"),
}
LIQUIDATED = "000030"
REPO_ROOT = Path(__file__).resolve().parents[2]


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


PRICE_DAYS = weekdays(date(2024, 1, 1), T)
SESSIONS = weekdays(date(2024, 1, 1), date(2025, 1, 31))


def _write(raw: Path, table: str, frame: pl.DataFrame) -> None:
    d = raw / table
    d.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(d / "part-000000.parquet")


def make_raw(lake: Path, snapshot: str = SNAPSHOT) -> Path:
    """Write the raw tables the views and the noflow panel read; returns the raw root."""
    raw = lake / "raw" / "raw_postgres" / f"snapshot_date={snapshot}" / "source=sj2_remote"
    ohlcv, mcap, shares, fs = [], [], [], []
    for t, (market, g, corp) in TICKERS.items():
        for i, day in enumerate(PRICE_DAYS):
            close = 1000.0 * (1 + g) ** i
            ohlcv.append((day, t, market, close - 1, close + 1, close - 2, close, 1000.0))
            mcap.append((day, t, market, close, 1000.0, close * 1000.0, close * 1e6, 1e6))
        shares.append((corp, t, 2023, "11011", "20240315000001", "합계", 5000, 10, 4990,
                       date(2023, 12, 31)))  # fmt: skip
        # a later share filing with an empty total: the non-empty PIT must ignore it
        shares.append((corp, t, 2024, "11013", "20240510000002", "합계", None, None, None,
                       date(2024, 3, 31)))  # fmt: skip
        for code, no, amount, add in (
            ("11013", "20240410000001", 100.0, None),
            ("11012", "20240810000001", 150.0, 250.0),
            ("11014", "20241110000001", 140.0, 390.0),
            ("11011", "20240315000009", 560.0, None),
        ):
            fs.append((corp, t, 2023 if code == "11011" else 2024, code, "CFS", "IS",
                       "ifrs-full_Revenue", "", 1, amount, "KRW", no, add, None))  # fmt: skip
    _write(
        raw,
        "daily_ohlcv",
        pl.DataFrame(
            ohlcv,
            schema=["trade_date", "ticker", "market", "open", "high", "low", "close", "volume"],
            orient="row",
        ),
    )
    _write(
        raw,
        "daily_market_cap",
        pl.DataFrame(
            mcap,
            schema=["trade_date", "ticker", "market", "source_close", "volume", "trading_value",
                    "market_cap", "listed_shares"],
            orient="row",
        ),
    )  # fmt: skip
    _write(
        raw,
        "dart_share_count_raw",
        pl.DataFrame(
            shares,
            schema={
                "corp_code": pl.String, "ticker": pl.String, "bsns_year": pl.Int32,
                "reprt_code": pl.String, "rcept_no": pl.String, "se": pl.String,
                "istc_totqy": pl.Int64, "tesstk_co": pl.Int64, "distb_stock_co": pl.Int64,
                "stlm_dt": pl.Date,
            },
            orient="row",
        ),
    )  # fmt: skip
    _write(
        raw,
        "dart_financial_statement_raw",
        pl.DataFrame(
            fs,
            schema={
                "corp_code": pl.String, "ticker": pl.String, "bsns_year": pl.Int32,
                "reprt_code": pl.String, "fs_div": pl.String, "sj_div": pl.String,
                "account_id": pl.String, "account_nm": pl.String, "ord": pl.Int32,
                "thstrm_amount": pl.Decimal(30, 4), "currency": pl.String,
                "rcept_no": pl.String, "thstrm_add_amount": pl.Decimal(30, 4),
                "frmtrm_q_amount": pl.Decimal(30, 4),
            },
            orient="row",
        ),
    )  # fmt: skip
    _write(
        raw,
        "dart_shareholder_return_raw",
        pl.DataFrame(
            schema={
                "corp_code": pl.String, "ticker": pl.String, "bsns_year": pl.Int32,
                "reprt_code": pl.String, "rcept_no": pl.String, "statement_type": pl.String,
                "row_name": pl.String, "stock_knd": pl.String, "dim1": pl.String,
                "dim2": pl.String, "dim3": pl.String, "metric_code": pl.String,
                "value_numeric": pl.Decimal(30, 4), "stlm_dt": pl.Date,
            }
        ),
    )  # fmt: skip
    _write(
        raw,
        "dart_xbrl_fact_raw",
        pl.DataFrame(
            schema={
                "corp_code": pl.String, "ticker": pl.String, "bsns_year": pl.Int32,
                "reprt_code": pl.String, "rcept_no": pl.String, "concept_id": pl.String,
                "concept_name": pl.String, "context_id": pl.String, "context_type": pl.String,
                "period_start": pl.Date, "period_end": pl.Date, "instant_date": pl.Date,
                "dimensions": pl.String, "value_numeric": pl.Decimal(30, 4),
                "label_ko": pl.String,
            }
        ),
    )  # fmt: skip
    _write(
        raw,
        "dart_filing_receipt_raw",
        pl.DataFrame(
            [
                (TICKERS[LIQUIDATED][2], LIQUIDATED, "20241220000001", date(2024, 12, 20),
                 "기타시장안내(정리매매 개시)"),
                (TICKERS["000010"][2], "000010", "20240401000001", date(2024, 4, 1), "사업보고서"),
            ],
            schema={"corp_code": pl.String, "ticker": pl.String, "rcept_no": pl.String,
                    "rcept_dt": pl.Date, "report_nm": pl.String},
            orient="row",
        ),
    )  # fmt: skip
    _write(
        raw,
        "dart_corp_master",
        pl.DataFrame(
            [(t, m, c) for t, (m, _, c) in TICKERS.items()],
            schema=["ticker", "market", "corp_code"],
            orient="row",
        ),
    )
    _write(
        raw,
        "stock_master",
        pl.DataFrame(
            [(t, m, "x", "ACTIVE", T, "krx", None, None, None) for t, (m, _, _) in TICKERS.items()],
            schema={"ticker": pl.String, "market": pl.String, "name": pl.String,
                    "status": pl.String, "last_seen_date": pl.Date, "source": pl.String,
                    "updated_at": pl.Datetime, "listing_date": pl.Date,
                    "first_seen_date": pl.Date},
            orient="row",
        ),
    )  # fmt: skip
    return raw


def sessions_file(path: Path) -> Path:
    path.write_text(json.dumps({"sessions": [d.isoformat() for d in SESSIONS]}))
    return path


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """One raw lake, its views and the prepared panel (shared: the build takes a few seconds)."""
    base = tmp_path_factory.mktemp("prepare")
    raw = make_raw(base / "lake")
    views = base / "views"
    vman = vw.build_views(raw, views, memory_limit="2GB", threads=2)
    out_root = base / "shadow"
    res = sp.prepare_day(
        D,
        raw_root=raw,
        views_dir=views,
        out_root=out_root,
        sessions=SESSIONS,
        memory_limit="2GB",
        threads=2,
        universe_filter=SMALL_UNIVERSE,
    )
    return {"base": base, "raw": raw, "views": views, "vman": vman, "out_root": out_root, **res}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------ views


def test_views_files_manifest_and_hashes(built) -> None:
    views: Path = built["views"]
    man = json.loads((views / "manifest.json").read_text())
    assert man == built["vman"]
    assert (
        set(man["views"])
        == set(vw.VIEW_FILES)
        == {
            "dim_stock_pit_daily",
            "dim_price_quality_daily",
            "stock_metric_vintage_fact_dl",
            "fin_quarterly_metric_vintage_dlr",
        }
    )
    assert set(bp.DERIVED_VIEW_FILES) <= set(man["views"])
    for name, info in man["views"].items():
        f = views / f"{name}.parquet"
        assert info["file"] == f.name and info["sha256"] == _sha(f)
        n = duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{f}')").fetchone()[0]
        assert info["rows"] == n > 0
        assert len(info["sql_sha256"]) == 64
    assert (
        man["raw_root"] == str(built["raw"]) and man["raw_snapshot"] == f"snapshot_date={SNAPSHOT}"
    )
    assert man["through"] is None and man["quality_pit"] == "serving"
    assert man["raw_tables"]["daily_ohlcv"]["files"] == 1
    assert "modeler_commit" in man and "modeler_dirty" in man
    assert man["calendar"]["first"] == "2024-01-02"  # 01-01 is a holiday
    assert man["calendar"]["last"] > PRICE_DAYS[-1].isoformat()  # reaches past the last price day
    assert not list(views.parent.glob(".views.work"))  # work dir removed


def test_views_sql_hashes_are_the_builders_text(built) -> None:
    v = built["vman"]["views"]
    pit = build_stock_pit_nonempty_sql(plan=PLAN_ASOF_INTERVALS)
    assert v["dim_stock_pit_daily"]["sql_sha256"] == hashlib.sha256(pit.encode()).hexdigest()
    dlr = build_fin_quarterly_metric_vintage_dlr_sql(
        vintage_view="stock_metric_vintage_fact_dl", semantics="v1"
    )
    assert (
        v["fin_quarterly_metric_vintage_dlr"]["sql_sha256"]
        == hashlib.sha256(dlr.encode()).hexdigest()
    )
    assert v["stock_metric_vintage_fact_dl"]["sql_sha256"] == built["vman"]["vintage"]["plan_hash"]


def test_pit_view_ignores_the_empty_share_filing(built) -> None:
    pit = built["views"] / "dim_stock_pit_daily.parquet"
    row = (
        duckdb.connect()
        .execute(
            f"SELECT issued_shares_pit FROM read_parquet('{pit}') WHERE ticker='000010' "
            f"AND trade_date = DATE '{T}'"
        )
        .fetchone()
    )
    assert row == (5000,)  # the NULL-total filing of 2024-05-10 did not blank it


def test_views_are_write_once(built, tmp_path) -> None:
    views: Path = built["views"]
    before = {p.name: _sha(p) for p in views.iterdir()}
    with pytest.raises(vw.ViewsRefused, match="not overwriting"):
        vw.build_views(built["raw"], views, memory_limit="2GB", threads=2)
    assert {p.name: _sha(p) for p in views.iterdir()} == before
    # one stray file is enough to refuse, and nothing is written next to it
    stray = tmp_path / "v"
    stray.mkdir()
    (stray / "dim_price_quality_daily.parquet").write_bytes(b"keep")
    with pytest.raises(vw.ViewsRefused):
        vw.build_views(built["raw"], stray, memory_limit="2GB", threads=2)
    assert sorted(p.name for p in stray.iterdir()) == ["dim_price_quality_daily.parquet"]


def test_through_cuts_the_raw_data(tmp_path) -> None:
    raw = make_raw(tmp_path / "lake")
    cut = date(2024, 6, 28)
    man = vw.build_views(raw, tmp_path / "v", through=cut, memory_limit="2GB", threads=2)
    assert man["through"] == cut.isoformat()
    pit = tmp_path / "v" / "dim_stock_pit_daily.parquet"
    last = (
        duckdb.connect().execute(f"SELECT max(trade_date) FROM read_parquet('{pit}')").fetchone()[0]
    )
    assert last == cut
    # the Q3 and annual filings (after the cut) are not in the vintage
    smvf = tmp_path / "v" / "stock_metric_vintage_fact_dl.parquet"
    late = (
        duckdb.connect()
        .execute(f"SELECT count(*) FROM read_parquet('{smvf}') WHERE rcept_no >= '20240701'")
        .fetchone()[0]
    )
    assert late == 0
    assert man["cuts"]["dart_corp_master"] is None and "2024-06-28" in man["cuts"]["daily_ohlcv"]


def test_cut_predicates_match_the_serving_prepare_where_they_overlap() -> None:
    from modeler.serving import kr_live_prepare as live

    for table in live.RAW_INPUTS:
        assert vw.cut_predicate(table, "2024-06-28") == live.cut_predicate(table, "2024-06-28")
    with pytest.raises(vw.ViewsRefused):
        vw.cut_predicate("dart_employee_raw", "2024-06-28")


def test_missing_raw_table_and_bad_arguments_leave_nothing(tmp_path) -> None:
    raw = make_raw(tmp_path / "lake")
    shutil.rmtree(raw / "dart_xbrl_fact_raw")
    out = tmp_path / "v"
    with pytest.raises(vw.ViewsRefused, match="dart_xbrl_fact_raw"):
        vw.build_views(raw, out, memory_limit="2GB", threads=2)
    assert not out.exists() and not list(tmp_path.glob(".v.work"))
    with pytest.raises(vw.ViewsRefused):
        vw.build_views(raw, out, quality_pit="other")
    with pytest.raises(vw.ViewsRefused):
        vw.build_views(tmp_path / "nope", out)


def test_quality_pit_nonempty_is_an_option(tmp_path) -> None:
    raw = make_raw(tmp_path / "lake")
    man = vw.build_views(raw, tmp_path / "v", quality_pit="nonempty", memory_limit="2GB", threads=2)
    assert man["quality_pit"] == "nonempty" and "quality_pit_sql_sha256" not in man


def test_verify_views_detects_a_changed_file(built, tmp_path) -> None:
    copy = tmp_path / "v"
    shutil.copytree(built["views"], copy)
    assert vw.verify_views(copy)["schema"] == vw.SCHEMA
    (copy / "dim_price_quality_daily.parquet").write_bytes(b"changed")
    with pytest.raises(vw.ViewsRefused, match="differs"):
        vw.verify_views(copy)
    (copy / "manifest.json").unlink()
    with pytest.raises(vw.ViewsRefused, match="manifest missing"):
        vw.verify_views(copy)


# ------------------------------------------------------------------------ prepare_day


def test_panel_path_is_the_wrappers_contract(built) -> None:
    out_root: Path = built["out_root"]
    assert built["panel"] == out_root / "panels_in" / f"D={D.isoformat()}" / "panel.parquet"
    assert built["manifest_path"] == built["panel"].with_name("panel_manifest.json")
    assert built["panel"].is_file() and built["manifest_path"].is_file()
    assert sp.panel_dir(out_root, D) == built["dir"]
    assert not list((out_root / "panels_in").glob(".build*"))  # build dir removed
    # the wrapper reads exactly these names and calls this module by default
    script = (REPO_ROOT / "deploy/prod/bin/kr-next-shadow.sh").read_text()
    assert 'panel_dir="$SHADOW_ROOT/panels_in/D=$D"' in script
    assert 'panel="$panel_dir/panel.parquet"' in script
    assert '"$panel_dir/panel_manifest.json"' in script
    assert "modeler.models._03_kr_next.shadow_prepare run" in script
    assert "SHADOW_PREPARE_CMD" in script  # the hook stays


def test_wrapper_script_parses_and_the_hook_still_wins() -> None:
    import subprocess

    path = REPO_ROOT / "deploy/prod/bin/kr-next-shadow.sh"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    script = path.read_text()
    i_hook = script.index('if [ -n "${SHADOW_PREPARE_CMD:-}" ]')
    i_default = script.index("shadow_prepare run --date")
    assert i_hook < i_default  # an explicit command is used before the default
    assert script.index('if [ ! -f "$panel" ]') < i_hook  # an existing panel skips both


def test_panel_is_the_signal_day_in_serving_mode(built) -> None:
    panel = pl.read_parquet(built["panel"])
    assert panel["trade_date"].unique().to_list() == [T]  # t = the session before D
    assert panel.height == built["manifest"]["row_count"] > 0
    assert not [c for c in panel.columns if bp._BANNED_SERVING.search(c)]
    assert {"liq_notice_flag", "liq_notice_rcept_no", "mcap_krx_log"} <= set(panel.columns)
    assert panel["liq_notice_flag"].null_count() == 0
    flagged = panel.filter(pl.col("liq_notice_flag"))["ticker"].to_list()
    assert flagged == [LIQUIDATED]
    assert (
        panel.filter(pl.col("ticker") == LIQUIDATED)["liq_notice_rcept_no"][0] == "20241220000001"
    )


def test_panel_manifest_records_hashes(built) -> None:
    m = json.loads(built["manifest_path"].read_text())
    assert m == built["manifest"] and m["schema"] == sp.SCHEMA
    assert (m["mode"], m["t"], m["D"]) == ("serving", T.isoformat(), D.isoformat())
    assert m["panel_sha256"] == _sha(built["panel"])
    assert m["views_manifest_sha256"] == _sha(built["views"] / "manifest.json")
    assert m["raw_snapshot"] == str(built["raw"]) and m["raw_snapshot_id"] == SNAPSHOT
    assert {"kn_tail", "kn_uni", "kn_liq", "kn_feat_fin_v2", "kn_feat_price_v2"} <= set(
        m["view_sql_sha256"]
    )
    assert "kn_label" not in m["view_sql_sha256"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in m["view_sql_sha256"].values())
    assert set(m["derived_views"]) == set(vw.VIEW_FILES)
    for name, info in m["derived_views"].items():
        assert info["sha256"] == built["vman"]["views"][name]["sha256"]
    assert m["columns"] == pl.read_parquet(built["panel"]).columns
    assert m["through"] is None and "modeler_commit" in m


def test_prepare_is_write_once(built) -> None:
    before = (_sha(built["panel"]), _sha(built["manifest_path"]))
    with pytest.raises(sp.PrepareRefused, match="not overwriting"):
        sp.prepare_day(
            D,
            raw_root=built["raw"],
            views_dir=built["views"],
            out_root=built["out_root"],
            sessions=SESSIONS,
            memory_limit="2GB",
            threads=2,
            universe_filter=SMALL_UNIVERSE,
        )
    assert (_sha(built["panel"]), _sha(built["manifest_path"])) == before


def test_an_orphan_manifest_does_not_block_a_retry(built, tmp_path) -> None:
    out_root = tmp_path / "shadow"
    d = sp.panel_dir(out_root, D)
    d.mkdir(parents=True)
    (d / "panel_manifest.json").write_text("{}")  # a run that died before moving the panel
    res = sp.prepare_day(
        D, raw_root=built["raw"], views_dir=built["views"], out_root=out_root,
        sessions=SESSIONS, memory_limit="2GB", threads=2, universe_filter=SMALL_UNIVERSE,
    )  # fmt: skip
    assert res["manifest"]["schema"] == sp.SCHEMA and res["panel"].is_file()


def test_prepare_refuses_views_from_elsewhere(built, tmp_path) -> None:
    def call(**kw):
        args = dict(
            raw_root=built["raw"], views_dir=built["views"], out_root=tmp_path / "s",
            sessions=SESSIONS, memory_limit="2GB", threads=2, universe_filter=SMALL_UNIVERSE,
        )  # fmt: skip
        args.update(kw)
        return sp.prepare_day(D, **args)

    # another raw snapshot (same layout, other date)
    other = make_raw(tmp_path / "lake2", snapshot="2025-01-02")
    with pytest.raises(sp.PrepareRefused, match="views were built from"):
        call(raw_root=other)
    # a cut the views do not have
    with pytest.raises(sp.PrepareRefused, match="cut at"):
        call(through="2024-12-30")
    # a view file that changed after the build
    bad = tmp_path / "views_bad"
    shutil.copytree(built["views"], bad)
    (bad / "dim_stock_pit_daily.parquet").write_bytes(b"x")
    with pytest.raises(sp.PrepareRefused, match="differs"):
        call(views_dir=bad)
    # a raw root outside the lake layout
    with pytest.raises(sp.PrepareRefused, match="raw root must look like"):
        call(raw_root=tmp_path)
    assert not (tmp_path / "s" / "panels_in" / f"D={D}" / "panel.parquet").exists()


def test_prepare_refuses_a_day_that_is_no_session(built, tmp_path) -> None:
    for bad in (date(2025, 1, 4), SESSIONS[0]):  # a Saturday, the first session
        with pytest.raises(sp.PrepareRefused):
            sp.prepare_day(
                bad, raw_root=built["raw"], views_dir=built["views"], out_root=tmp_path / "s",
                sessions=SESSIONS, universe_filter=SMALL_UNIVERSE,
            )  # fmt: skip
    assert not (tmp_path / "s").exists()


def test_through_panel_equals_the_uncut_panel_on_its_day(tmp_path) -> None:
    """Data after t never reaches the panel of t: the cut run and the whole-snapshot run agree."""
    raw = make_raw(tmp_path / "lake")
    cut = date(2024, 9, 30)  # a Monday; t = the session before D = 2024-10-01
    d = date(2024, 10, 1)
    v_cut = vw.build_views(raw, tmp_path / "v_cut", through=cut, memory_limit="2GB", threads=2)
    v_all = vw.build_views(raw, tmp_path / "v_all", memory_limit="2GB", threads=2)
    assert v_cut["through"] == cut.isoformat() and v_all["through"] is None
    common = dict(sessions=SESSIONS, memory_limit="2GB", threads=2, universe_filter=SMALL_UNIVERSE)
    a = sp.prepare_day(d, raw_root=raw, views_dir=tmp_path / "v_cut", out_root=tmp_path / "a",
                       through=cut, **common)  # fmt: skip
    b = sp.prepare_day(d, raw_root=raw, views_dir=tmp_path / "v_all", out_root=tmp_path / "b",
                       **common)  # fmt: skip
    assert a["manifest"]["through"] == cut.isoformat() and b["manifest"]["through"] is None
    fa = pl.read_parquet(a["panel"]).sort("ticker")
    fb = pl.read_parquet(b["panel"]).sort("ticker")
    assert fa.height == fb.height > 0
    assert fa.equals(fb)


# ------------------------------------------------------------------------------- CLI


def test_cli_run_builds_views_once_then_the_panel(tmp_path, monkeypatch) -> None:
    lake = tmp_path / "lake"
    make_raw(lake)
    cal = sessions_file(tmp_path / "cal.json")
    out_root = tmp_path / "shadow"
    # the real universe filter would need years of warm-up: swap in the small one
    real = bp.build_panel_file
    monkeypatch.setattr(
        bp,
        "build_panel_file",
        lambda con, **kw: real(con, **{**kw, "universe_filter": SMALL_UNIVERSE}),
    )
    argv = ["run", "--date", D.isoformat(), "--raw-lake", str(lake / "raw" / "raw_postgres"),
            "--out-root", str(out_root), "--sessions", str(cal),
            "--memory-limit", "2GB", "--threads", "2"]  # fmt: skip
    assert sp.main(argv) == 0
    panel = sp.panel_dir(out_root, D) / "panel.parquet"
    assert panel.is_file() and (sp.views_dir_for(out_root, D) / "manifest.json").is_file()
    assert sp.main(argv) == 2  # a panel exists: refused, nothing rewritten
    # a raw snapshot that is not there is exit 31 (the wrapper's "panel missing")
    gone = [*argv]
    gone[gone.index("--date") + 1] = "2025-01-02"
    assert sp.main(gone) == sp.EXIT_RAW_MISSING


def test_views_dir_name_has_no_hive_partition_pattern(tmp_path) -> None:
    assert "=" not in sp.views_dir_for(tmp_path, D).name


def test_cli_run_rebuilds_views_without_a_manifest(tmp_path, monkeypatch) -> None:
    lake = tmp_path / "lake"
    make_raw(lake)
    out_root = tmp_path / "shadow"
    stale = sp.views_dir_for(out_root, D)
    stale.mkdir(parents=True)
    (stale / "dim_price_quality_daily.parquet").write_bytes(b"half a build")
    real = bp.build_panel_file
    monkeypatch.setattr(
        bp,
        "build_panel_file",
        lambda con, **kw: real(con, **{**kw, "universe_filter": SMALL_UNIVERSE}),
    )
    res = sp.prepare_from_lake(
        D, raw_lake=lake / "raw" / "raw_postgres", out_root=out_root, sessions=SESSIONS,
        memory_limit="2GB", threads=2,
    )  # fmt: skip
    assert res["panel"].is_file() and (stale / "manifest.json").is_file()


def test_record_carries_the_panel_manifest(built, bundle_dir, tmp_path, capsys) -> None:  # noqa: F811
    """The wrapper's second step: ``shadow score --panel-manifest`` puts it into record.json."""
    cal = sessions_file(tmp_path / "cal.json")
    out = tmp_path / "records"
    # the 30-feature bundle wants every column; the synthetic panel has them all (maybe NULL)
    rc = sh.main(
        ["score", "--date", D.isoformat(), "--panel", str(built["panel"]),
         "--panel-manifest", str(built["manifest_path"]), "--bundle-dir", str(bundle_dir),
         "--out-root", str(out), "--ledger", str(out / "ledger.jsonl"), "--sessions", str(cal)]
    )  # fmt: skip
    assert rc in (0, 3)  # 3 = written after the 15:00 deadline of a 2025 date: invalid, kept
    rec = json.loads((out / f"D={D.isoformat()}" / "record.json").read_text())
    assert rec["panel_manifest"] == built["manifest"]
    assert (
        rec["panel_manifest_sha256"]
        == hashlib.sha256(sh._json_dumps(built["manifest"]).encode()).hexdigest()
    )
    assert SEOUL  # (imported for the fixture module)
