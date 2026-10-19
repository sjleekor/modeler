"""US daily serving: universe version v1 (default) or v2 (``universe_daily_v2``).

합성 parquet만 쓴다. 실제 레이크는 읽지 않는다.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time
from pathlib import Path

import polars as pl
import pytest
from collector.lake import DataRoot

from test_daily_inputs import _hash, _inputs, _write

from modeler.serving import us_daily
from modeler.serving.daily_inputs import select
from modeler.serving.us_daily import (
    SERVING_V2_EXCLUDED_SECURITY_TYPES,
    UNIVERSE_V2_SERVING_RULE,
    _monthly_membership,
    collect_pinned_revisions,
    latest_complete_session,
    sha256_file,
    verify_universe_v2_completion,
)
from modeler.us.lake import US_TABLES, UsLake

SESSIONS = [date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2)]
MONTH_FIRST = date(2026, 10, 1)


class FakeLake:
    def __init__(self, tables: dict[str, pl.DataFrame]) -> None:
        self.tables = tables

    def scan(self, table: str) -> pl.LazyFrame:
        return self.tables[table].lazy()


def _calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2026, 9, 30), MONTH_FIRST, date(2026, 10, 2)],
            "exchange": ["XNYS"] * 3,
            "close_local": [time(16, 0)] * 3,
        }
    )


def _v1(rows: list[tuple]) -> pl.DataFrame:
    """rows: (symbol, in_universe, cik, sic, mcap_rank)"""
    return pl.DataFrame(
        {
            "date": [MONTH_FIRST] * len(rows),
            "symbol": [r[0] for r in rows],
            "cik": [r[2] for r in rows],
            "sic": [r[3] for r in rows],
            "mcap_rank": [r[4] for r in rows],
            "adv_20d": [1_000_000.0 + i for i in range(len(rows))],
            "exchange": ["XNYS"] * len(rows),
            "in_universe": [r[1] for r in rows],
        },
        schema_overrides={"cik": pl.Int64, "sic": pl.String, "mcap_rank": pl.Int32},
    )


def _v2(rows: list[tuple], *, view: list[str] | None = None) -> pl.DataFrame:
    """rows: (symbol, in_universe[, security_type]) -- security_type defaults to ``common``."""
    frame = pl.DataFrame(
        {
            "date": [MONTH_FIRST] * len(rows),
            "symbol": [r[0] for r in rows],
            "security_id": [f"id-{i}" for i in range(len(rows))],
            "security_type": [r[2] if len(r) > 2 else "common" for r in rows],
            "in_universe": [r[1] for r in rows],
        },
        schema_overrides={"security_type": pl.String},
    )
    return frame.with_columns(pl.Series("view", view)) if view is not None else frame


V1_ROWS = [
    ("AAA", True, 1, "3571", 1),
    ("BBB", True, 2, "6020", 2),
    ("ONLYV1", True, 3, "1000", 3),
    ("OUT", False, 4, "2000", 4),
]


def _lake(v2: pl.DataFrame, v1: pl.DataFrame | None = None) -> FakeLake:
    return FakeLake(
        {
            "trading_calendar": _calendar(),
            "universe_daily": v1 if v1 is not None else _v1(V1_ROWS),
            "universe_daily_v2": v2,
        }
    )


# ---- S2: membership ---------------------------------------------------------------------------


def test_v1_membership_is_unchanged_and_v1_is_the_default() -> None:
    lake = _lake(_v2([("AAA", True)]))
    default = _monthly_membership(lake, date(2026, 10, 2))
    assert default.columns == [
        "symbol", "cik", "sic", "mcap_rank", "adv_20d", "exchange", "sic2", "date"]
    assert default["symbol"].sort().to_list() == ["AAA", "BBB", "ONLYV1"]
    assert default.equals(_monthly_membership(lake, date(2026, 10, 2), "v1"))
    assert default["date"].unique().to_list() == [date(2026, 10, 2)]


def test_v2_membership_takes_v2_symbols_with_v1_attributes() -> None:
    # NEWV2 is only in v2 (no v1 row): kept with null attributes.  ONLYV1 is only in v1: dropped.
    # OUT is a v1 non-member that v2 admits: its v1 attributes are used.
    lake = _lake(_v2([("AAA", True), ("BBB", True), ("NEWV2", True), ("OUT", True), ("GONE", False)]))
    members = _monthly_membership(lake, date(2026, 10, 2), "v2").sort("symbol")
    assert members["symbol"].to_list() == ["AAA", "BBB", "NEWV2", "OUT"]
    assert members.columns == _monthly_membership(lake, date(2026, 10, 2)).columns
    aaa = members.filter(pl.col("symbol") == "AAA").row(0, named=True)
    assert (aaa["cik"], aaa["sic"], aaa["sic2"], aaa["mcap_rank"]) == (1, "3571", "35", 1)
    new = members.filter(pl.col("symbol") == "NEWV2").row(0, named=True)
    assert new["cik"] is None and new["mcap_rank"] is None and new["sic2"] is None
    out = members.filter(pl.col("symbol") == "OUT").row(0, named=True)
    assert out["cik"] == 4 and out["sic2"] == "20"
    assert members["date"].unique().to_list() == [date(2026, 10, 2)]


def test_v2_membership_excludes_adr_only_in_v2_and_keeps_other_types_and_null() -> None:
    assert SERVING_V2_EXCLUDED_SECURITY_TYPES == ("adr",)
    assert UNIVERSE_V2_SERVING_RULE == "v2-minus-adr"
    v2 = _v2([("AAA", True, "common"), ("BBB", True, "reit"), ("NEWV2", True, "common_equity"),
              ("ADR1", True, "adr"), ("ADR2", True, "adr"), ("NULLT", True, None),
              ("OUTADR", False, "adr"), ("OUT", False, "common")])
    lake = _lake(v2, _v1(V1_ROWS + [("ADR1", True, 9, "3571", 9)]))
    v2_members = _monthly_membership(lake, date(2026, 10, 2), "v2")
    assert sorted(v2_members["symbol"].to_list()) == ["AAA", "BBB", "NEWV2", "NULLT"]
    # v1 is untouched: it ignores universe_daily_v2 and its own ADR-like rows stay
    v1_members = _monthly_membership(lake, date(2026, 10, 2), "v1")
    assert sorted(v1_members["symbol"].to_list()) == ["AAA", "ADR1", "BBB", "ONLYV1"]
    # a month made only of ADRs is empty after the exclusion
    with pytest.raises(ValueError, match="비었습니다"):
        _monthly_membership(_lake(_v2([("ADR1", True, "adr")])), date(2026, 10, 2), "v2")


def test_research_panel_v2_still_includes_adr() -> None:
    """The serving rule must not leak into the research/forward v2 definition (D2)."""
    from modeler.us import panel

    v2 = _v2([("AAA", True, "common"), ("ADR1", True, "adr")])
    lake = _lake(v2, _v1(V1_ROWS + [("ADR1", True, 9, "3571", 9)]))
    members = panel._members_v2(lake, [MONTH_FIRST]).collect()
    assert sorted(members["symbol"].to_list()) == ["AAA", "ADR1"]


def test_v2_membership_reads_the_first_session_of_the_month_only() -> None:
    v2 = pl.concat(
        [
            _v2([("AAA", True)]),
            _v2([("AAA", True), ("LATER", True)]).with_columns(pl.lit(date(2026, 10, 2)).alias("date")),
        ]
    )
    members = _monthly_membership(_lake(v2), date(2026, 10, 2), "v2")
    assert members["symbol"].to_list() == ["AAA"]


def test_v2_membership_keeps_pit_rows_when_a_view_column_exists() -> None:
    v2 = _v2([("AAA", True), ("BBB", True), ("POST", True)], view=["pit", "pit", "post"])
    members = _monthly_membership(_lake(v2), date(2026, 10, 2), "v2")
    assert sorted(members["symbol"].to_list()) == ["AAA", "BBB"]


def test_v2_membership_refuses_duplicates_and_an_empty_member_list() -> None:
    with pytest.raises(ValueError, match="중복 심볼"):
        _monthly_membership(_lake(_v2([("AAA", True), ("AAA", True)])), date(2026, 10, 2), "v2")
    with pytest.raises(ValueError, match="비었습니다"):
        _monthly_membership(_lake(_v2([("AAA", False)])), date(2026, 10, 2), "v2")
    with pytest.raises(ValueError, match="universe_version"):
        _monthly_membership(_lake(_v2([("AAA", True)])), date(2026, 10, 2), "v3")


def test_the_version_is_validated_on_every_entry_point() -> None:
    with pytest.raises(ValueError, match="universe_version"):
        us_daily.build_daily_panel(_lake(_v2([("AAA", True)])), date(2026, 10, 2), universe_version="x")
    with pytest.raises(ValueError, match="universe_version"):
        us_daily.build_daily_features_for_dates(
            _lake(_v2([("AAA", True)])), [date(2026, 10, 2)], universe_version="x")
    with pytest.raises(ValueError, match="universe_version"):
        us_daily.prepare_daily_features(date(2026, 10, 2), diagnostic_only=True, universe_version="x")
    with pytest.raises(ValueError, match="universe_version"):
        latest_complete_session(date(2026, 10, 2), universe_version="x")


# ---- synthetic lake on disk for pinning / completion / prepare -----------------------------------


def _snapshot(root: DataRoot, table: str, snapshot: str, frame: pl.DataFrame | None = None) -> Path:
    directory = root.derived / "snapshots" / table / f"snapshot_date={snapshot}"
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / "part.parquet"
    (frame if frame is not None else pl.DataFrame({"x": [1]})).write_parquet(part)
    return part


def _disk_lake(
    tmp_path: Path, *, v2_days: list[date] | None = None, v2_extra: dict | None = None
) -> tuple[DataRoot, Path]:
    root = DataRoot(tmp_path / "us")
    for table in US_TABLES:
        _snapshot(root, table, "2026-10-01")
    _snapshot(root, "trading_calendar", "2026-10-05", _calendar().vstack(
        pl.DataFrame({"date": [date(2026, 10, 5)], "exchange": ["XNYS"], "close_local": [time(16, 0)]})))
    days = SESSIONS
    prices = pl.DataFrame({"date": days, "symbol": ["AAA"] * 3, "close": [10.0] * 3})
    _snapshot(root, "prices_daily", "2026-10-03", prices)
    universe = pl.DataFrame({"date": days, "symbol": ["AAA"] * 3, "in_universe": [True] * 3})
    universe_part = _snapshot(root, "universe_daily", "2026-10-03", universe)
    inputs = {"prices_daily": "2026-10-03", "listing_snapshots": "2026-10-01",
              "filings_sub": "2026-10-01", "midas_security_daily": "2026-10-01"}
    ticker = root.base / "raw" / "tickers.json"
    ticker.parent.mkdir(parents=True, exist_ok=True)
    ticker.write_text("{}")
    (universe_part.parent / "completion.json").write_text(json.dumps({
        "schema_version": 1, "table": "universe_daily", "snapshot_date": "2026-10-03",
        "snapshot_sha256": sha256_file(universe_part), "unjudged_months": [],
        "input_snapshots": inputs,
        "input_snapshot_sha256": {
            t: sha256_file(root.derived / "snapshots" / t / f"snapshot_date={d}" / "part.parquet")
            for t, d in inputs.items()},
        "ticker_source_sha256": {"raw/tickers.json": sha256_file(ticker)},
        "previous_month_seed_count": 3}))
    v2_days = v2_days if v2_days is not None else SESSIONS
    v2_frame = pl.DataFrame({"date": v2_days, "symbol": ["AAA"] * len(v2_days),
                             "security_id": ["s1"] * len(v2_days),
                             "in_universe": [True] * len(v2_days)})
    v2_part = _snapshot(root, "universe_daily_v2", "2026-10-04", v2_frame)
    record = {
        "schema_version": 1, "table": "universe_daily_v2", "snapshot_date": "2026-10-04",
        "snapshot_sha256": sha256_file(v2_part), "rule_version": "seg-r3c.1+master-r3c.1+ud2-1",
        "unjudged_months": [],
        "input_snapshots": {"prices_daily": "2026-10-03", "listing_snapshots_v2": "2026-10-03"},
        "input_snapshot_sha256": {"prices_daily": "a" * 64, "listing_snapshots_v2": "b" * 64},
        **(v2_extra or {}),
    }
    (v2_part.parent / "completion.json").write_text(json.dumps(record))
    return root, v2_part


def _patch_root(monkeypatch, root: DataRoot) -> None:
    monkeypatch.setattr(us_daily.DataRoot, "resolve", classmethod(lambda cls, market: root))


# ---- pinning and S4 ---------------------------------------------------------------------------


def test_pinned_revisions_add_v2_only_in_v2_mode(tmp_path: Path) -> None:
    root, _ = _disk_lake(tmp_path)
    v1 = collect_pinned_revisions(root, UsLake(root))
    assert v1 == collect_pinned_revisions(root, UsLake(root), "v1")
    assert "universe_daily_v2" not in v1
    v2 = collect_pinned_revisions(root, UsLake(root), "v2")
    assert v2.pop("universe_daily_v2") == "2026-10-04"
    assert v2 == v1


def test_a_prime_in_v2_mode_is_bounded_by_the_v2_max_date(tmp_path: Path, monkeypatch) -> None:
    root, _ = _disk_lake(tmp_path, v2_days=SESSIONS[:2])  # the v2 build stopped one session early
    _patch_root(monkeypatch, root)
    v1 = latest_complete_session(date(2026, 10, 5))
    assert "universe_v2_max" not in v1 and v1["resolved"] == "2026-10-02"
    v2 = latest_complete_session(date(2026, 10, 5), universe_version="v2")
    assert v2["universe_v2_max"] == "2026-10-01" and v2["resolved"] == "2026-10-01"
    assert v2["stale"] is True
    assert {k: v for k, v in v2.items() if k not in {"universe_v2_max", "resolved"}} == {
        k: v for k, v in v1.items() if k != "resolved"}
    # With the v2 table as current as the rest, both versions agree.
    full_root, _ = _disk_lake(tmp_path / "full")
    _patch_root(monkeypatch, full_root)
    assert (latest_complete_session(date(2026, 10, 5), universe_version="v2")["resolved"]
            == latest_complete_session(date(2026, 10, 5))["resolved"] == "2026-10-02")


# ---- S3: v2 completion ------------------------------------------------------------------------


def test_verify_v2_completion_returns_what_the_manifest_needs(tmp_path: Path) -> None:
    root, v2_part = _disk_lake(tmp_path)
    got = verify_universe_v2_completion(root, "2026-10-04")
    assert got["completion_sha256"] == sha256_file(v2_part.parent / "completion.json")
    assert got["rule_version"] == "seg-r3c.1+master-r3c.1+ud2-1"
    assert got["input_snapshot_sha256"] == {"prices_daily": "a" * 64, "listing_snapshots_v2": "b" * 64}


@pytest.mark.parametrize("patch", [
    {"snapshot_sha256": "0" * 64},
    {"table": "universe_daily"},
    {"unjudged_months": ["2026-09-01"]},
    {"schema_version": 2},
    {"snapshot_date": "2026-10-03"},
    {"rule_version": ""},
    {"rule_version": None},
    {"input_snapshot_sha256": None},
])
def test_verify_v2_completion_rejects_a_bad_record(tmp_path: Path, patch: dict) -> None:
    root, v2_part = _disk_lake(tmp_path)
    marker = v2_part.parent / "completion.json"
    marker.write_text(json.dumps({**json.loads(marker.read_text()), **patch}))
    with pytest.raises(ValueError, match="universe_daily_v2"):
        verify_universe_v2_completion(root, "2026-10-04")


def test_verify_v2_completion_rejects_missing_and_symlinked_files(tmp_path: Path) -> None:
    root, v2_part = _disk_lake(tmp_path)
    marker = v2_part.parent / "completion.json"
    real = tmp_path / "elsewhere.json"
    real.write_text(marker.read_text())
    marker.unlink()
    marker.symlink_to(real)
    with pytest.raises(ValueError, match="universe_daily_v2"):
        verify_universe_v2_completion(root, "2026-10-04")
    marker.unlink()
    with pytest.raises(ValueError, match="universe_daily_v2"):
        verify_universe_v2_completion(root, "2026-10-04")
    marker.write_text(real.read_text())
    moved = tmp_path / "part.parquet"
    os.replace(v2_part, moved)
    v2_part.symlink_to(moved)
    with pytest.raises(ValueError, match="universe_daily_v2"):
        verify_universe_v2_completion(root, "2026-10-04")


# ---- S5: prepare hash and manifest ------------------------------------------------------------


def _prepare(monkeypatch, root: DataRoot, version: str | None) -> dict:
    _patch_root(monkeypatch, root)
    seen: list[str] = []

    def fake_build(lake, scoring_date, *, financial_selection_rule, universe_version="v1"):
        seen.append(universe_version)
        return pl.DataFrame({"date": [scoring_date], "symbol": ["AAA"], "x": [1.0]})

    monkeypatch.setattr(us_daily, "build_daily_features", fake_build)
    kwargs = {} if version is None else {"universe_version": version}
    path = us_daily.prepare_daily_features(date(2026, 10, 2), diagnostic_only=True, **kwargs)
    assert seen == [version or "v1"]
    return json.loads(path.read_text())


def test_v1_prepare_has_no_universe_v2_keys_and_the_default_equals_explicit_v1(
    tmp_path: Path, monkeypatch
) -> None:
    root, _ = _disk_lake(tmp_path)
    default = _prepare(monkeypatch, root, None)
    explicit = _prepare(monkeypatch, root, "v1")
    assert default == explicit  # the same prep_id: nothing in the hash input changed
    assert not [k for k in default if k.startswith("universe_") and k != "universe_completion_sha256"]
    assert "universe_daily_v2" not in default["source_revision"]
    assert "universe_daily_v2" not in default["source_snapshot_files"]


def test_v2_prepare_records_the_v2_snapshot_and_changes_the_prep_id(
    tmp_path: Path, monkeypatch
) -> None:
    root, v2_part = _disk_lake(tmp_path)
    v1 = _prepare(monkeypatch, root, "v1")
    v2 = _prepare(monkeypatch, root, "v2")
    assert v2["prep_id"] != v1["prep_id"]
    assert v2["universe_version"] == "v2"
    assert v2["universe_v2_snapshot_date"] == "2026-10-04"
    assert v2["universe_v2_completion_sha256"] == sha256_file(v2_part.parent / "completion.json")
    assert v2["universe_v2_rule_version"] == "seg-r3c.1+master-r3c.1+ud2-1"
    assert v2["universe_v2_input_snapshot_sha256"] == {
        "prices_daily": "a" * 64, "listing_snapshots_v2": "b" * 64}
    assert v2["source_revision"]["universe_daily_v2"] == "2026-10-04"
    (entry,) = v2["source_snapshot_files"]["universe_daily_v2"]
    assert entry["sha256"] == sha256_file(v2_part)
    only_v2 = set(v2) - set(v1)
    assert only_v2 == {"universe_version", "universe_v2_snapshot_date", "universe_v2_completion_sha256",
                       "universe_v2_rule_version", "universe_v2_input_snapshot_sha256",
                       "universe_v2_serving_rule", "universe_v2_excluded_security_types"}
    assert v2["universe_v2_serving_rule"] == "v2-minus-adr"
    assert v2["universe_v2_excluded_security_types"] == ["adr"]


def test_v2_prepare_stops_on_a_bad_v2_completion(tmp_path: Path, monkeypatch) -> None:
    root, v2_part = _disk_lake(tmp_path, v2_extra={"unjudged_months": ["2026-09-01"]})
    _patch_root(monkeypatch, root)
    with pytest.raises(ValueError, match="unjudged"):
        us_daily.prepare_daily_features(date(2026, 10, 2), diagnostic_only=True, universe_version="v2")


# ---- run-time selection -----------------------------------------------------------------------


def _set_us_version(args: dict, version: str | None) -> None:
    path = next(args["prepared_root"].glob("us/**/manifest.json"))
    native = json.loads(path.read_text())
    if version is not None:
        native["universe_version"] = version
    _write(path, native)
    marker_path = path.parent / "completion.json"
    marker = json.loads(marker_path.read_text())
    marker["native_prepare_manifest_sha256"] = _hash(path)
    _write(marker_path, marker)


@pytest.mark.parametrize("native_version,configured,selected", [
    (None, "v1", True),   # an old native without the field is v1
    ("v1", "v1", True),
    ("v2", "v2", True),
    ("v2", "v1", False),
    (None, "v2", False),
    ("v1", "v2", False),
])
def test_selection_accepts_only_natives_of_the_configured_version(
    tmp_path: Path, native_version: str | None, configured: str, selected: bool
) -> None:
    args = _inputs(tmp_path)
    _set_us_version(args, native_version)
    result = select(**args, us_universe_version=configured)
    assert ("universe_version" in result) is False
    assert result.get("us_universe_version") == (None if configured == "v1" else "v2")
    if selected:
        assert result["markets"]["US"]["status"] == "ok" and len(result["jobs"]) == 3
    else:
        assert result["markets"]["US"] == {"status": "unavailable", "reason": "us_universe_version_mismatch"}
        assert [job["market"] for job in result["jobs"]] == ["KR"]


def test_selection_default_is_v1_and_rejects_unknown_versions(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    _set_us_version(args, "v2")
    assert select(**args)["markets"]["US"]["reason"] == "us_universe_version_mismatch"
    with pytest.raises(ValueError, match="us_universe_version"):
        select(**args, us_universe_version="v3")


def test_coordinator_passes_and_validates_us_universe_version(monkeypatch) -> None:
    from modeler.serving import daily_coordinator as dc

    assert dc.US_UNIVERSE_VERSIONS == ("v1", "v2")
    seen: dict = {}
    monkeypatch.setattr(dc, "select", lambda **kw: seen.update(kw) or {})
    config = {"prepared_root": "/p", "selection_root": "/s", "release_manifest": "/r",
              "kr_calendar": "/k", "us_calendar": "/u", "us_expected_source": "/e"}
    now = datetime(2026, 10, 6, 9, 30)
    dc.select_stage(config, date(2026, 10, 6), now)
    assert seen["us_universe_version"] == "v1"
    dc.select_stage({**config, "us_universe_version": "v2"}, date(2026, 10, 6), now)
    assert seen["us_universe_version"] == "v2"


# ---- us-prepare.sh ----------------------------------------------------------------------------

from test_briefing_wrappers import PREPARE, _calls, _run, serving  # noqa: E402,F401  (fixture)


def _set_ops(serving: dict, **values: object) -> None:
    path = serving["config"] / "ops.json"
    ops = json.loads(path.read_text())
    path.write_text(json.dumps({**ops, **values}))


def _helper_calls(serving: dict, mode: str) -> list[str]:
    return [c for c in _calls(serving) if f" {mode} 2026-" in c]


def test_wrapper_without_the_key_runs_v1_exactly_as_before(serving) -> None:
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"])
    assert done.returncode == 0, done.stderr
    (prepare,) = [c for c in _calls(serving) if "modeler.serving.us_daily" in c]
    assert "--universe-version" not in prepare
    assert "resolve 2026-09-29 v1|" in _helper_calls(serving, "resolve")[0]
    assert all(" prepared-ok 2026-09-29 " in c and " v1|" in c for c in _helper_calls(serving, "prepared-ok"))


def test_wrapper_passes_v2_to_resolve_prepared_ok_and_prepare(serving) -> None:
    _set_ops(serving, us_universe_version="v2")
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"])
    assert done.returncode == 0, done.stderr
    assert "universe version v2" in done.stdout
    (prepare,) = [c for c in _calls(serving) if "modeler.serving.us_daily" in c]
    assert (f"prepare --as-of 2026-09-29 --raw-feature-parity-status score_equivalent "
            f"--raw-feature-parity-evidence {serving['evidence']} --universe-version v2|") in prepare
    assert "resolve 2026-09-29 v2|" in _helper_calls(serving, "resolve")[0]
    assert all(" v2|" in c for c in _helper_calls(serving, "prepared-ok"))


@pytest.mark.parametrize("bad", ["v3", "", None, 2])
def test_wrapper_rejects_an_unknown_version_with_10(serving, bad) -> None:
    _set_ops(serving, us_universe_version=bad)
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"])
    assert done.returncode == 10 and "us_universe_version must be v1 or v2" in done.stderr
    assert _calls(serving) == []


def _embedded_helper() -> str:
    text = PREPARE.read_text()
    start = text.index("helper=$(cat <<'PY'\n") + len("helper=$(cat <<'PY'\n")
    return text[start:text.index("\nPY\n)", start)]


@pytest.mark.parametrize("native_version,wanted,found", [
    (None, "v1", True), ("v1", "v1", True), ("v2", "v2", True),
    ("v2", "v1", False), (None, "v2", False),
])
def test_prepared_ok_requires_the_configured_version(
    tmp_path: Path, monkeypatch, capsys, native_version, wanted, found
) -> None:
    from modeler.serving import orchestration

    stock = tmp_path / "stock"
    monkeypatch.setenv("STOCK_DATA_ROOT", str(stock))
    base = (stock / "us" / "output" / us_daily.SCORING_VERSION / "prepared"
            / "score_date=2026-10-02" / "prep_id=x")
    base.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(base / "features.parquet")
    features_sha = sha256_file(base / "features.parquet")
    native = {"market": "US", "feature_asof_date": "2026-10-02",
              "raw_feature_parity_status": "score_equivalent", "serving_eligible": True,
              "diagnostic_only": False, "code_hash": us_daily.code_tree_hash(),
              "raw_feature_parity_evidence_sha256": "e" * 64, "features_sha256": features_sha,
              "input_sha256": features_sha, "availability_evidence_type": "prepared_features_completion"}
    if native_version is not None:
        native["universe_version"] = native_version
    (base / "manifest.json").write_text(json.dumps(native))
    (base / "completion.json").write_text(json.dumps({
        "schema_version": "prepared-features-completion.v1",
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": features_sha,
        "native_prepare_manifest_sha256": sha256_file(base / "manifest.json"),
        "verified_available_by": "2026-10-05T17:40:00+09:00"}))
    monkeypatch.setattr(orchestration, "us_native_block_reason", lambda native: None)
    monkeypatch.setattr("sys.argv", ["-c", "prepared-ok", "2026-10-02", "e" * 64, wanted])
    with pytest.raises(SystemExit) as stop:
        exec(compile(_embedded_helper(), "<helper>", "exec"), {"__name__": "__helper__"})
    assert stop.value.code == (0 if found else 1)
    assert capsys.readouterr().out.strip().endswith("manifest.json") is found
