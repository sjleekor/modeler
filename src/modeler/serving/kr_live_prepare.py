"""Build the KR briefing's required marts from one complete sj2 raw snapshot.

This is a serving entry point. It deliberately imports only feature builders;
research scan orchestration and label builders are outside its dependency graph.
The raw export is a separate, completed collector operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import UTC, date, datetime, time, timedelta
from importlib.resources import files
from pathlib import Path
from zoneinfo import ZoneInfo

from collector.kr.shared import METRIC_RULES_MRV1, get_trading_days

from modeler.etl.config import CONFIG_TABLES, RAW_TABLES, DataRoot, EngineOptions, LakeConfig, REMOTE_SOURCE
from modeler.etl.features.filing_activity import materialize_filing_activity
from modeler.etl.features.fin_pit import SEMANTICS_V2 as FIN_PIT_SEMANTICS_V2
from modeler.etl.features.fin_pit import materialize_fin_pit
from modeler.etl.features.fin_scan import SEMANTICS_V2 as FIN_SCAN_SEMANTICS_V2
from modeler.etl.features.fin_scan import check_projection_columns, materialize_fin_scan_daily
from modeler.etl.features.fin_vintage import JOIN_PLAN_COALESCE
from modeler.etl.features.flow import PIVOT_PLAN_WINDOW, materialize_flow
from modeler.etl.features.price import materialize_price
from modeler.etl.lake import (
    _sql_str_literal,
    connect,
    read_derived_plan_record,
    register_derived_marts,
    register_views,
)
from modeler.etl.mart import is_materialized, mart_cache_metadata, materialize, register_mart_view, sql_contract_hash
from modeler.etl.marts.financial_quarters import SEMANTICS_V2 as FQMV_SEMANTICS_V2
from modeler.etl.marts.financial_quarters import materialize_fin_quarterly_metric_vintage
from modeler.etl.marts.metric_vintages import (
    AVAILABILITY_BEYOND_CALENDAR, PLAN_STAGED, SEMANTICS_V2 as SMVF_SEMANTICS_V2,
    materialize_stock_metric_vintage_fact,
)
from modeler.etl.marts.metrics_normalize import PLAN_SPLIT_ARGMIN as SMF_PLAN_SPLIT_ARGMIN
from modeler.etl.quality import QUALITY_TABLE, materialize_price_quality
from modeler.etl.stock_pit import PIT_TABLE, PLAN_ASOF_INTERVALS, materialize_stock_pit
from modeler.etl.trading_panel import MARKET_SEMANTICS_V2
from modeler.etl.universe import broad_universe_filter, build_universe_sql, materialize_universe
from modeler.serving.build_profile import (
    PARTIAL_PROFILE_NAME, PROFILE_NAME, BuildProfiler, NullProfiler, write_json_atomic,
)
from modeler.serving.kr_prepare import (
    BUILD_PROFILE_FULL, BUILD_PROFILE_SERVING, BUILD_PROFILES, DEFAULT_BUILD_PROFILE,
    REQUIRED_PREP_MARTS, KrBriefingSpec, _required_marts, requested_mart_columns,
)

SEOUL = ZoneInfo("Asia/Seoul")
SERVING_MARTS = REQUIRED_PREP_MARTS
# The temp cap is not optional for the serving builder: an unbounded spill filled the
# server disk once. Callers may raise or lower it, never remove it.
DEFAULT_MAX_TEMP_SIZE = "30GB"
TEMP_PARENT_NAME = "_duckdb_tmp"
# Raw export marker policies (collector bin/raw-parquet-export-all.sh). The exported-snapshot
# policy (collector v0.15.17 --consistent-snapshot) reads every table from one PostgreSQL
# snapshot, so each table manifest must name the marker's snapshot id.
RAW_POLICY_PER_CHUNK = "read_committed_per_chunk"
RAW_POLICY_EXPORTED_SNAPSHOT = "repeatable_read_exported_snapshot"
# Which version of each shared mart the serving builder selects. The research and frozen
# paths keep their own defaults (v1 / single / or-join): these are not defaults anywhere else.
# A "plan" computes the same rows faster; a "semantics" version changes what the mart means.
#   dim_stock_pit_daily        v1, asof_intervals plan (winner interval + ASOF join).
#   feat_price                 v2: the market return is summed in a fixed order, so beta, alpha
#                              and resid_ret (px_idio_vol_60d, px_resid_mom_12_1) are
#                              reproducible. v1 differs from it only in the last bits.
#   feat_flow                  v1, window plan (the argmin_pivot plan moved
#                              flow_*_netbuy_z_20d in the last bits; root cause unknown, so it
#                              stays available but is not selected).
#   stock_metric_fact          v1, split_argmin plan (equality/wildcard rule joins, GROUP BY
#                              arg_min winner). Rules mrv1_20260818: the training-time
#                              set, so fin_cash_ratio and its isna flag match the frozen panel.
#                              Research keeps the current rules (mrv2_20260909).
#   feat_fin_pit               v2: the first report wins ties on period_end.
#   stock_metric_vintage_fact  v2 (no look-back fallback at the calendar edge, first report
#                              wins ties), staged plan.
#   fin_quarterly_metric_vintage  v2: a derived available_from (differenced quarter, weighted
#                              share, TTM) is NULL when a contributing vintage's is NULL, so a
#                              filing beyond the calendar cannot be skipped by greatest().
#   feat_fin_scan_daily        v2: z-score means and deviations are summed in ticker order, so
#                              fin_value_z is reproducible (not a model input; v1 moves it by
#                              ~1e-14 between runs). coalesce_join plan (hash-joinable bound).
SERVING_STOCK_PIT_PLAN = PLAN_ASOF_INTERVALS
SERVING_PRICE_SEMANTICS = MARKET_SEMANTICS_V2
SERVING_FLOW_PIVOT_PLAN = PIVOT_PLAN_WINDOW
SERVING_METRIC_FACT_PLAN = SMF_PLAN_SPLIT_ARGMIN
# The rule set the frozen model's training data was built under (107 rules / 29 codes).
# The current set adds rules, which fills fin_cash_ratio where training had NULL.
SERVING_METRIC_RULES_VERSION = METRIC_RULES_MRV1
SERVING_FIN_PIT_SEMANTICS = FIN_PIT_SEMANTICS_V2
SERVING_SMVF_SEMANTICS = SMVF_SEMANTICS_V2
SERVING_SMVF_PLAN = PLAN_STAGED
SERVING_FQMV_SEMANTICS = FQMV_SEMANTICS_V2
SERVING_FIN_SCAN_SEMANTICS = FIN_SCAN_SEMANTICS_V2
SERVING_FIN_SCAN_JOIN_PLAN = JOIN_PLAN_COALESCE
# feat_flow arguments. The research builder (etl/compute_all.py `_build_features`) passes the
# same three views; leaving price_view out silently takes the degraded path in
# etl/features/flow.py, where the volume ratios (flow_*_netbuy_to_volume_*) are all NULL.
SERVING_FLOW_VIEWS = {
    "price_view": "daily_ohlcv", "pit_view": PIT_TABLE, "quality_view": QUALITY_TABLE,
}
# stock_metric_vintage_fact rows whose receipt date lies beyond the calendar are not a build
# failure: fin_quarterly_metric_vintage v2 turns every figure that depends on them into NULL
# availability, which the daily intervals drop. They are counted in the marker and warned about.
# More than this many rows is a data error, not a late filing, so the build still fails. One
# filing yields 35 rows on average and 58 at most (3.71M rows, 106k filings, 2026-09-29), so 500
# rows is 9 to 14 filings, 0.013% of the mart; a calendar or rcept_no parsing fault would
# exceed it by far, while a single bad receipt number (the likely case) stays under it.
DEFAULT_MAX_BEYOND_CALENDAR_ROWS = 500
BEYOND_CALENDAR_SAMPLE = 10
# How many sessions past max(K, last price date) the vintage calendar must reach. A filing
# received on the last day of the calendar has no next session otherwise.
CALENDAR_EXTENSION_SESSIONS = 10
RAW_INPUTS = (
    "daily_ohlcv", "krx_security_flow_raw", "dart_financial_statement_raw",
    "dart_share_count_raw", "dart_shareholder_return_raw", "dart_xbrl_fact_raw",
    "dart_corp_master", "dart_filing_receipt_raw",
)


# How each raw input is cut when the marts must end at a reference session earlier than the
# newest row of the export (K' fallback, see ``kr_reference``).  Same method as the replay tool that
# produced the 2026-10-06 historical units.  Receipt dates come from the first eight digits of
# ``rcept_no``; a row whose date cannot be read is kept, because the marts treat it through their
# documented fallback and hiding it would change the features.
RCEPT_DATE_SQL = "TRY_STRPTIME(NULLIF(SUBSTR(CAST(rcept_no AS VARCHAR), 1, 8), ''), '%Y%m%d')::DATE"
CUT_TRADE_DATE_TABLES = ("daily_ohlcv", "krx_security_flow_raw")
CUT_RCEPT_TABLES = (
    "dart_financial_statement_raw", "dart_share_count_raw", "dart_shareholder_return_raw",
    "dart_xbrl_fact_raw",
)
CUT_RECEIPT_TABLE = "dart_filing_receipt_raw"
CUT_NONE_TABLES = ("dart_corp_master",)  # no event date: a company list as of the snapshot


def cut_predicate(table: str, asof: str) -> str | None:
    """SQL keeping the rows a database would have held at the close of ``asof``; None = uncut."""
    day = f"DATE '{date.fromisoformat(asof).isoformat()}'"
    if table in CUT_TRADE_DATE_TABLES:
        return f"trade_date <= {day}"
    if table in CUT_RCEPT_TABLES:
        return f"COALESCE({RCEPT_DATE_SQL} <= {day}, TRUE)"
    if table == CUT_RECEIPT_TABLE:
        return f"COALESCE({RCEPT_DATE_SQL} <= {day}, TRUE) AND COALESCE(rcept_dt <= {day}, TRUE)"
    if table in CUT_NONE_TABLES:
        return None
    raise ValueError(f"no cut is defined for raw table {table!r}; refusing to leave it uncut")


def register_raw_views(con, config: LakeConfig, *, cut_asof: str | None = None) -> dict[str, str | None]:
    """Register the raw inputs; with ``cut_asof`` every dated one only shows rows up to that day.

    Returns table -> predicate (None = uncut).  Empty when no cut was asked for.
    """
    created = register_views(con, config, tables=RAW_INPUTS)
    cuts: dict[str, str | None] = {}
    if cut_asof is None:
        return cuts
    for table in created:
        predicate = cut_predicate(table, cut_asof)
        cuts[table] = predicate
        if predicate is None:
            continue
        glob = _sql_str_literal(config.table_glob(table))
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet({glob}, hive_partitioning=false) WHERE {predicate}")
    return cuts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


HOLIDAY_CALENDAR_NAME = "collector/kr/infra/calendar/data/holidays_krx.csv"


def _default_holiday_path() -> Path:
    return Path(str(files("collector.kr.infra.calendar") / "data" / "holidays_krx.csv"))


def load_holiday_calendar(
    *, upper: date, feature_asof_date: str, path: Path | None = None
) -> tuple[set[date], dict]:
    """Read the KRX holiday CSV explicitly; fail closed instead of weekend-only fallback.

    The collector loader silently drops to weekends when the CSV is absent (frozen
    releases copy only ``*.py``), which would count holidays as trading days.
    """
    target = path or _default_holiday_path()
    if not target.is_file():
        raise FileNotFoundError(f"KRX holiday calendar CSV is missing: {target}")
    holidays: set[date] = set()
    with target.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                holidays.add(date.fromisoformat(row["date"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"KRX holiday calendar has an invalid row: {row!r}") from exc
    if not holidays:
        raise ValueError(f"KRX holiday calendar has no rows: {target}")
    required_year = max(upper.year, date.fromisoformat(feature_asof_date).year)
    if max(holidays).year < required_year:
        raise ValueError(
            f"KRX holiday calendar ends {max(holidays)} but needs coverage through year {required_year}")
    record = {
        "name": HOLIDAY_CALENDAR_NAME if path is None else target.name,
        "sha256": _sha256(target), "row_count": len(holidays),
        "min_date": min(holidays).isoformat(), "max_date": max(holidays).isoformat(),
    }
    return holidays, record


def serving_trading_days(
    lower: date, upper: date, *, feature_asof_date: str, holidays: set[date],
    extension_sessions: int = CALENDAR_EXTENSION_SESSIONS,
) -> tuple[list[date], dict]:
    """Sessions from ``lower`` to ``extension_sessions`` past ``max(upper, K)``.

    The vintage mart looks up the first session after each receipt date. A calendar that
    stops at the last price date has no such session for a receipt filed that day, so the
    calendar is extended with weekday-minus-holiday sessions from the official CSV. The
    extension stops at the end of the last CSV year: later holidays are unknown, and a
    guessed session would be wrong. If not even one session after ``max(upper, K)`` fits
    inside the CSV's coverage the build fails rather than falling back.
    """
    k = date.fromisoformat(feature_asof_date)
    anchor = max(upper, k)
    coverage_end = date(max(holidays).year, 12, 31)
    # Ten sessions are at most ~25 days (Chuseok plus a weekend); 45 is a safe scan bound.
    scan_end = min(anchor + timedelta(days=45), coverage_end)
    days = list(get_trading_days(lower, scan_end, holidays=holidays))
    after = [day for day in days if day > anchor]
    if not after:
        raise ValueError(
            f"KRX holiday calendar covers through {coverage_end}; no session after "
            f"{anchor} is known, so receipts filed on {anchor} would have no available date")
    last = after[:extension_sessions][-1]
    days = [day for day in days if day <= last]
    digest = hashlib.sha256(",".join(day.isoformat() for day in days).encode()).hexdigest()
    record = {
        "first_session": days[0].isoformat(), "last_session": last.isoformat(),
        "session_count": len(days), "anchor_date": anchor.isoformat(),
        "sessions_after_anchor": min(len(after), extension_sessions),
        "sessions_sha256": digest,
    }
    return days, record


def _check_beyond_calendar(con, *, max_rows: int) -> dict:
    """Count receipts that still have no next session after the extension.

    ``fin_quarterly_metric_vintage`` v2 propagates the unknown availability, so these rows
    no longer need to stop the build: figures that depend on them are simply not available
    yet. The count goes into the marker and a warning to stderr. More than ``max_rows``
    fails the build, so a systematic fault cannot silently mass-null the financial features.
    """
    if max_rows < 0:
        raise ValueError("max_beyond_calendar_rows must not be negative")
    rows, filings = con.execute(
        "SELECT count(*), count(DISTINCT rcept_no) FROM stock_metric_vintage_fact "
        f"WHERE availability_source = '{AVAILABILITY_BEYOND_CALENDAR}'").fetchone()
    sample: list[str] = []
    if rows:
        sample = [row[0] for row in con.execute(
            "SELECT DISTINCT rcept_no FROM stock_metric_vintage_fact "
            f"WHERE availability_source = '{AVAILABILITY_BEYOND_CALENDAR}' "
            f"ORDER BY rcept_no LIMIT {BEYOND_CALENDAR_SAMPLE}").fetchall()]
    record = {"rows": rows, "filings": filings, "sample_rcept_no": sample, "max_rows": max_rows}
    if rows > max_rows:
        raise ValueError(
            f"{rows} stock_metric_vintage_fact rows ({filings} filings) have a receipt date "
            f"beyond the trading calendar, more than the {max_rows} allowed; extend the holiday "
            f"CSV or the calendar extension, or check the rcept_no dates (e.g. {sample[:3]})")
    if rows:
        print(
            f"[kr-build] warning: {rows} stock_metric_vintage_fact rows ({filings} filings) have "
            f"a receipt date beyond the trading calendar; their availability stays unknown "
            f"(e.g. rcept_no {sample[:3]})", file=sys.stderr)
    return record


def _cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("input_cutoff must include a timezone")
    local = parsed.astimezone(SEOUL)
    if local.time().replace(tzinfo=None) != time(9, 30):
        raise ValueError("KR input cutoff must be 09:30 Asia/Seoul")
    return parsed


def verify_raw(config: LakeConfig, *, cutoff: datetime, feature_asof_date: str) -> str:
    """Require the exporter completion marker and a source capture before cutoff."""
    path = config.raw_root / "_manifests" / "_SUCCESS.json"
    if not path.is_file():
        raise FileNotFoundError(f"complete sj2 raw export is missing: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("route") != "remote":
        raise ValueError("KR raw export must use the sj2 direct route")
    expected = set(RAW_TABLES) | set(CONFIG_TABLES)
    if set(body.get("tables", {})) != expected:
        raise ValueError("KR raw export table set is incomplete or unexpected")
    policy = body.get("snapshot_policy")
    pg_snapshot_id = None
    if policy == RAW_POLICY_EXPORTED_SNAPSHOT:
        pg_snapshot_id = body.get("pg_snapshot_id")
        if not pg_snapshot_id:
            raise ValueError("KR raw export marker has no pg_snapshot_id")
    elif policy != RAW_POLICY_PER_CHUNK:
        raise ValueError("KR raw export snapshot policy differs from the collector contract")
    for name, entry in body["tables"].items():
        expected_path = config.raw_root / "_manifests" / "table_manifests" / f"{name}.json"
        if not isinstance(entry, dict) or entry.get("manifest_path") != str(expected_path):
            raise ValueError(f"KR raw export table manifest path does not match snapshot: {name}")
        if not expected_path.is_file():
            raise FileNotFoundError(f"KR raw export table manifest is missing: {name}")
        detail = json.loads(expected_path.read_text(encoding="utf-8"))
        source = detail.get("source", {})
        table = detail.get("table", {})
        if (source.get("name") != config.source or source.get("snapshot_date") != config.snapshot_date
                or table.get("name") != name or entry.get("rows_exported") != table.get("rows_exported")
                or entry.get("schema_hash") != (table.get("schema") or {}).get("hash")):
            raise ValueError(f"KR raw export table manifest does not match snapshot/source: {name}")
        if pg_snapshot_id is not None and (
                source.get("snapshot_policy") != RAW_POLICY_EXPORTED_SNAPSHOT
                or source.get("pg_snapshot_id") != pg_snapshot_id):
            raise ValueError(f"KR raw export table was not read from the marker's snapshot: {name}")
    stamp = datetime.fromisoformat(body.get("finished_at", "").replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None or stamp > cutoff:
        raise ValueError("KR raw export has no verified completion by input cutoff")
    if date.fromisoformat(feature_asof_date) >= cutoff.astimezone(SEOUL).date():
        raise ValueError("feature K must precede the report date")
    return _sha256(path)


def required_serving_marts() -> tuple[str, ...]:
    spec = KrBriefingSpec()
    columns = list(spec.feature_columns())
    from modeler.models._02_updown_prob import features as fx

    materials = [material for output, material, _ in fx.INTERACTIONS if output in columns]
    required = set(_required_marts([*columns, *materials])) | {"dim_universe_daily", "dim_price_quality_daily"}
    if required != set(SERVING_MARTS):
        raise ValueError(f"KR model mart mapping changed: {sorted(required ^ set(SERVING_MARTS))}")
    return SERVING_MARTS


def _build_marts(
    con, config: LakeConfig, feature_asof_date: str, holidays_path: Path | None = None,
    profiler: BuildProfiler | None = None, build_info: dict | None = None,
    *, profile: str = DEFAULT_BUILD_PROFILE,
    max_beyond_calendar_rows: int = DEFAULT_MAX_BEYOND_CALENDAR_ROWS,
    cut_to_asof: bool = False,
) -> dict | None:
    """Topological, strict feature-only build. No best-effort skipped marts.

    ``cut_to_asof`` hides every raw row dated after ``feature_asof_date``, so the marts end at
    the reference session even when the export holds later rows (``build_info["raw_cut"]``).

    ``profile="full"`` builds every mart of the chain. ``profile="serving"`` builds only
    what the model reads: the vintage marts are skipped (they feed nothing but the
    ``feat_fin_scan_daily`` columns the model does not request) and ``feat_fin_scan_daily``
    is a projection onto the requested columns. Without the vintage marts there is no
    calendar to extend, so that step is skipped as well and the return value is ``None``.

    ``build_info`` receives facts the marker records besides the mart list (the calendar
    range actually used); the return value stays the holiday CSV record.
    """
    if profile not in BUILD_PROFILES:
        raise ValueError(f"unknown KR build profile {profile!r}; expected {BUILD_PROFILES}")
    full = profile == BUILD_PROFILE_FULL
    # Fails loudly before any work if the model asks for a column the projection cannot make.
    scan_columns = None if full else _scan_projection_columns()
    profiler = profiler or NullProfiler()
    holiday_record = None
    with profiler.step("raw_views"):
        cuts = register_raw_views(con, config, cut_asof=feature_asof_date if cut_to_asof else None)
        if build_info is not None and cut_to_asof:
            build_info["raw_cut"] = {"cut_asof": feature_asof_date, "predicates": cuts}
    if full:
        with profiler.step("calendar"):
            lower, upper = con.execute(
                "SELECT min(trade_date), max(trade_date) FROM daily_ohlcv").fetchone()
            if lower is None or upper is None:
                raise ValueError("KR price table is empty")
            holidays, holiday_record = load_holiday_calendar(
                upper=upper, feature_asof_date=feature_asof_date, path=holidays_path)
            trading_days, calendar_record = serving_trading_days(
                lower, upper, feature_asof_date=feature_asof_date, holidays=holidays)
            if build_info is not None:
                build_info["trading_calendar"] = calendar_record
    with profiler.step("stock_pit", PIT_TABLE, con):
        materialize_stock_pit(con, config, plan=SERVING_STOCK_PIT_PLAN)
    with profiler.step("price_quality", QUALITY_TABLE, con):
        materialize_price_quality(con, config, pit_view=PIT_TABLE)
    with profiler.step("feat_price", "feat_price", con):
        materialize_price(
            con, config, quality_view=QUALITY_TABLE, semantics=SERVING_PRICE_SEMANTICS)
    with profiler.step("feat_flow", "feat_flow", con):
        materialize_flow(con, config, **SERVING_FLOW_VIEWS, pivot_plan=SERVING_FLOW_PIVOT_PLAN)
    with profiler.step("dim_universe_broad_daily", "dim_universe_broad_daily", con):
        broad_sql = build_universe_sql(broad_universe_filter())
        materialize(con, config, "dim_universe_broad_daily", broad_sql)
        register_mart_view(con, config, "dim_universe_broad_daily")
    with profiler.step("dim_universe_daily", "dim_universe_daily", con):
        materialize_universe(con, config, KrBriefingSpec().universe)
    with profiler.step("stock_metric_fact", "stock_metric_fact", con):
        register_derived_marts(
            con, config, which=("stock_metric_fact",), persist=True,
            metric_fact_plan=SERVING_METRIC_FACT_PLAN,
            metric_rules_version=SERVING_METRIC_RULES_VERSION)
    with profiler.step("feat_fin_pit", "feat_fin_pit", con):
        materialize_fin_pit(con, config, semantics=SERVING_FIN_PIT_SEMANTICS)
    with profiler.step("feat_filing_activity", "feat_filing_activity", con):
        materialize_filing_activity(con, config)
    if full:
        with profiler.step("stock_metric_vintage_fact", "stock_metric_vintage_fact", con):
            materialize_stock_metric_vintage_fact(
                con, config, trading_days=trading_days, semantics=SERVING_SMVF_SEMANTICS,
                plan=SERVING_SMVF_PLAN, stage_dir=Path(config.engine.temp_directory) / "stages")
            beyond = _check_beyond_calendar(con, max_rows=max_beyond_calendar_rows)
            if build_info is not None:
                build_info["beyond_calendar"] = beyond
        with profiler.step("fin_quarterly_metric_vintage", "fin_quarterly_metric_vintage", con):
            materialize_fin_quarterly_metric_vintage(
                con, config, semantics=SERVING_FQMV_SEMANTICS)
        with profiler.step("feat_fin_scan_daily", "feat_fin_scan_daily", con):
            materialize_fin_scan_daily(
                con, config, join_plan=SERVING_FIN_SCAN_JOIN_PLAN,
                semantics=SERVING_FIN_SCAN_SEMANTICS)
    else:
        with profiler.step("feat_fin_scan_daily", "feat_fin_scan_daily", con):
            materialize_fin_scan_daily(con, config, columns=scan_columns)
    return holiday_record


def _scan_projection_columns() -> tuple[str, ...]:
    """The ``feat_fin_scan_daily`` columns the model reads, from its feature list.

    ``check_projection_columns`` raises for a column the projection cannot produce, so a
    new model feature never becomes a silently missing column.
    """
    requested = requested_mart_columns().get("feat_fin_scan_daily", [])
    return check_projection_columns(requested)


def _mart_record(con, config: LakeConfig, name: str, feature_asof_date: str) -> dict:
    if not is_materialized(config, name):
        raise FileNotFoundError(f"KR serving mart missing after build: {name}")
    metadata = mart_cache_metadata(config, name)
    if not metadata or not metadata.get("sql_hash"):
        raise ValueError(f"KR serving mart has no SQL cache contract: {name}")
    count, latest = con.execute(f"SELECT count(*), max(trade_date) FROM {name}").fetchone()
    if count <= 0 or latest is None or latest.isoformat() != feature_asof_date:
        raise ValueError(f"KR serving mart {name} is not complete through K={feature_asof_date}: max={latest}")
    record = {
        "view": name, "row_count": count, "max_trade_date": latest.isoformat(),
        "sql_hash": metadata["sql_hash"],
        **_version_entry(metadata),
    }
    if "projected_columns" in metadata:
        # A reader must not mistake a projection for the full mart.
        record["projected_columns"] = metadata["projected_columns"]
    return record


# Built and versioned but not read by the model: they feed feat_fin_scan_daily, so their
# semantics and plan still belong in the marker.
INTERMEDIATE_VERSIONED_MARTS = ("stock_metric_vintage_fact", "fin_quarterly_metric_vintage")


# Per profile, the marts the marker describes besides the model's seven. The serving profile
# does not build the vintage marts at all.
PROFILE_INTERMEDIATE_MARTS = {
    BUILD_PROFILE_FULL: INTERMEDIATE_VERSIONED_MARTS,
    BUILD_PROFILE_SERVING: (),
}


def _version_entry(metadata: dict) -> dict:
    """semantics / plan / plan hash of one mart; absent keys mean the legacy statement."""
    entry = {
        "semantics_version": metadata.get("semantics_version", "v1"),
        "plan": metadata.get("plan", "single"), "plan_hash": metadata.get("plan_hash"),
    }
    if "projected_columns" in metadata:
        entry["projected_columns"] = metadata["projected_columns"]
    return entry


def _intermediate_versions(config: LakeConfig, profile: str = DEFAULT_BUILD_PROFILE) -> dict:
    """Versions of the marts the model does not read but the marker still has to describe.

    ``stock_metric_fact`` is a derived mart without cache metadata; its plan comes from the
    ``_plan.json`` the persisting step wrote (absent = the default plan).
    """
    versions = {"stock_metric_fact": read_derived_plan_record(config, "stock_metric_fact")}
    for name in PROFILE_INTERMEDIATE_MARTS[profile]:
        metadata = mart_cache_metadata(config, name)
        if not metadata or not metadata.get("sql_hash"):
            raise ValueError(f"KR serving mart has no SQL cache contract: {name}")
        versions[name] = _version_entry(metadata)
    return versions


def _execution_environment(engine: EngineOptions) -> dict:
    """What the floating-point results of the window aggregates depend on besides the input.

    DuckDB's moving-window aggregates (``STDDEV_SAMP``/``AVG`` over ``ROWS BETWEEN n
    PRECEDING``) add values in a tree whose node boundaries follow the row layout, so the
    last bits can change with the engine version, the thread count (how rows are split into
    hash groups) and the other rows in the table. Two builds are bitwise comparable only if
    this record and the input match; E1 treats it as part of the contract, not as noise.
    """
    import duckdb

    return {
        "duckdb": duckdb.__version__, "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "threads": engine.threads, "memory_limit": engine.memory_limit,
    }


def _check_engine_limits(threads: int, memory_limit: str) -> None:
    """The one place that bounds the serving engine; B experiments widen it here."""
    if threads < 1 or threads > 2 or memory_limit != "4GB":
        raise ValueError("KR live mart builder is capped at two threads and 4GB")


def _new_run_dir(root: DataRoot, snapshot_date: str, temp_dir: Path | None) -> Path:
    """Fresh per-run DuckDB spill directory, never inside the code tree or the cwd.

    Default parent is ``<kr>/derived/_duckdb_tmp``: the same volume as the marts (the
    NVMe on the server), so a spill cannot land on the system disk, and it does not
    depend on where the job was started. ``temp_dir`` replaces the parent only; the
    run id subdirectory is always created, so cleanup removes exactly this run's files.
    """
    if temp_dir is not None and not temp_dir.is_absolute():
        raise ValueError(f"temp_dir must be an absolute path, not relative to the cwd: {temp_dir}")
    parent = temp_dir if temp_dir is not None else root.derived / TEMP_PARENT_NAME
    run_id = f"{snapshot_date}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{os.getpid()}"
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_live_marts(
    *, snapshot_date: str, feature_asof_date: str, input_cutoff: str,
    stock_data_root: Path | None = None, threads: int = 2, memory_limit: str = "4GB",
    holidays_csv: Path | None = None, temp_dir: Path | None = None,
    max_temp_size: str = DEFAULT_MAX_TEMP_SIZE, profile: str = DEFAULT_BUILD_PROFILE,
    max_beyond_calendar_rows: int = DEFAULT_MAX_BEYOND_CALENDAR_ROWS,
    cut_to_asof: bool = False,
) -> dict:
    if profile not in BUILD_PROFILES:
        raise ValueError(f"unknown KR build profile {profile!r}; expected {BUILD_PROFILES}")
    if date.fromisoformat(snapshot_date).isoformat() != snapshot_date:
        raise ValueError("snapshot_date must use YYYY-MM-DD")
    if date.fromisoformat(feature_asof_date).isoformat() != feature_asof_date:
        raise ValueError("feature_asof_date must use YYYY-MM-DD")
    if snapshot_date < feature_asof_date:
        raise ValueError("snapshot cannot predate K")
    _check_engine_limits(threads, memory_limit)
    if not max_temp_size or not max_temp_size.strip():
        raise ValueError("max_temp_size must be a DuckDB size such as 30GB")
    root = (DataRoot(base=stock_data_root / "kr") if stock_data_root
            else DataRoot.resolve(market="kr"))
    cutoff = _cutoff(input_cutoff)
    # Checks that can fail run before the spill directory exists, so they leave nothing behind.
    probe = LakeConfig(root=root, snapshot_date=snapshot_date, source=REMOTE_SOURCE)
    raw_hash = verify_raw(probe, cutoff=cutoff, feature_asof_date=feature_asof_date)
    names = required_serving_marts()
    marker = probe.feature_mart_root / "_manifests" / "_SUCCESS.json"
    if marker.exists():
        raise FileExistsError(f"KR serving feature snapshot is already sealed: {marker}")
    run_dir = _new_run_dir(root, snapshot_date, temp_dir)
    engine = EngineOptions(
        threads=threads, memory_limit=memory_limit, temp_directory=str(run_dir),
        max_temp_directory_size=max_temp_size)
    config = LakeConfig(root=root, snapshot_date=snapshot_date, source=REMOTE_SOURCE, engine=engine)
    profiler = BuildProfiler(
        temp_dir=run_dir, engine=engine.as_pragmas(),
        context={"snapshot_date": snapshot_date, "feature_asof_date": feature_asof_date,
                 "run_id": run_dir.name, "profile": profile})
    profiler.start()
    con = None
    try:
        con = connect(config)
        build_info: dict = {}
        holiday_record = _build_marts(
            con, config, feature_asof_date, holidays_csv, profiler, build_info,
            profile=profile, max_beyond_calendar_rows=max_beyond_calendar_rows,
            cut_to_asof=cut_to_asof)
        with profiler.step("verify_marts"):
            records = [_mart_record(con, config, name, feature_asof_date) for name in names]
        con.close()
        con = None
        completed = datetime.now(UTC)
        body = {
            "schema_version": "kr-serving-feature-marts.v1", "status": "success",
            "profile": profile, "snapshot_date": snapshot_date, "source": REMOTE_SOURCE,
            "feature_asof_date": feature_asof_date, "input_cutoff": input_cutoff,
            "raw_marker_sha256": raw_hash, "created_at": completed.isoformat(),
            "universe_sql_hash": sql_contract_hash(build_universe_sql(KrBriefingSpec().universe)),
            "holiday_calendar": holiday_record,
            "trading_calendar": build_info.get("trading_calendar"),
            # Receipts beyond the calendar (full profile only: the serving profile builds no
            # vintage mart, so there is nothing to count).
            "beyond_calendar": build_info.get("beyond_calendar"),
            # None = the export was used as it is; set = rows after the reference session were hidden.
            "raw_cut": build_info.get("raw_cut"),
            "execution_environment": _execution_environment(engine),
            "mart_versions": {
                **{item["view"]: _version_entry(item) for item in records},
                **_intermediate_versions(config, profile),
            },
            "marts": records,
        }
        profiler.finish()
        profiler.stop()
        _write_profile(profiler, marker.parent / PROFILE_NAME)
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name("_SUCCESS.json.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(body, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    except BaseException as exc:
        profiler.fail(exc)
        profiler.stop()
        # The run directory stays for inspection: spill files plus the partial profile.
        _write_profile(profiler, run_dir / PARTIAL_PROFILE_NAME)
        print(f"[kr-build] failed; temp and partial profile kept at {run_dir}", file=sys.stderr)
        raise
    finally:
        if con is not None:
            con.close()
    shutil.rmtree(run_dir, ignore_errors=True)
    return body


def _write_profile(profiler: BuildProfiler, path: Path) -> None:
    """Profiling is diagnostics: a write failure is reported, never raised."""
    try:
        write_json_atomic(path, profiler.to_dict())
    except OSError as exc:
        print(f"[kr-build] could not write build profile {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--feature-asof-date", required=True)
    parser.add_argument("--input-cutoff", required=True)
    parser.add_argument("--stock-data-root", type=Path)
    parser.add_argument("--holidays-csv", type=Path, help="override the packaged KRX holiday CSV")
    parser.add_argument(
        "--temp-dir", type=Path,
        help="parent for the per-run DuckDB spill directory (default: <kr>/derived/_duckdb_tmp)")
    parser.add_argument(
        "--max-temp-size", default=DEFAULT_MAX_TEMP_SIZE,
        help="DuckDB max_temp_directory_size for the whole build (default: %(default)s)")
    parser.add_argument(
        "--profile", choices=BUILD_PROFILES, default=DEFAULT_BUILD_PROFILE,
        help="full: every mart of the chain; serving: only what the model reads "
             "(default: %(default)s)")
    parser.add_argument(
        "--max-beyond-calendar-rows", type=int, default=DEFAULT_MAX_BEYOND_CALENDAR_ROWS,
        help="fail if more stock_metric_vintage_fact rows than this have a receipt date beyond "
             "the calendar (full profile; default: %(default)s)")
    parser.add_argument(
        "--cut-to-asof", action="store_true",
        help="hide raw rows dated after --feature-asof-date (K' fallback; default: use the export as is)")
    args = parser.parse_args()
    result = build_live_marts(**vars(args))
    print(json.dumps({"snapshot_date": result["snapshot_date"], "feature_asof_date": result["feature_asof_date"], "marts": len(result["marts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
