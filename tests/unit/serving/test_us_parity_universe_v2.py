"""US parity-check ``--universe-version v2``: the ``universe_v2`` evidence block (S6).

합성 데이터만 쓴다. 실제 레이크는 읽지 않는다.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from collector.lake import DataRoot

from test_us_fin_det_rule import _evidence
from test_us_universe_v2_serving import _disk_lake, _patch_root

from modeler.reporting.markdown import SELECT_REASON_KO
from modeler.serving import us_daily
from modeler.serving.orchestration import (
    US_PARITY_CRITERIA,
    US_PARITY_V2_CRITERIA,
    us_parity_failures,
    us_parity_v2_failures,
)
from modeler.serving.us_daily import (
    FEATURES,
    GOLDEN_DATES,
    UNIVERSE_V2_TABLE,
    sha256_file,
    validate_parity_evidence,
)
from modeler.us.features.fundamentals_ttm import RULE_DET_A

RULE_VERSION = "seg-r3c.1+master-r3c.1+ud2-1"
REVISION = "2026-10-06"
FIN = us_daily.FINANCIAL_FEATURES
NON_FIN = [f for f in FEATURES if f not in FIN]


# ---- evidence gating (no data) ------------------------------------------------------------------


def _block(**overrides) -> dict:
    block = {
        "criteria": dict(US_PARITY_V2_CRITERIA),
        "snapshot_date": REVISION,
        "completion_sha256": "a" * 64,
        "rule_version": RULE_VERSION,
        "per_date": [
            {
                "date": d.isoformat(), "v2_rows": 10, "common_keys": 9, "v1_only": 1, "v2_only": 1,
                "g1_missing_symbols": 0, "g1_extra_symbols": 0, "g1_duplicate_symbols": 0,
                "g3_eligible_mismatch_rows": 0, "g4_unexplained_frozen_only": 0,
                "g4_unexplained_v2_only": 0,
            }
            for d in GOLDEN_DATES
        ],
        "non_financial": {"keys_equal": True, "diff_cells": 0, "features": 42, "common_keys": 54},
        "status": "passed",
        "failures": [],
    }
    block.update(overrides)
    return block


def _with_block(**overrides) -> dict:
    return _evidence(universe_version="v2", universe_v2=_block(**overrides))


def test_v1_evidence_without_the_block_is_unchanged() -> None:
    evidence = _evidence()
    assert "universe_v2" not in evidence
    assert us_parity_failures(evidence) == []
    assert not [k for k in US_PARITY_CRITERIA if "universe" in k or k.startswith("g")]


def test_a_passing_block_adds_no_failures_and_criteria_are_visible() -> None:
    assert us_parity_failures(_with_block()) == []
    assert _with_block()["universe_v2"]["criteria"] == US_PARITY_V2_CRITERIA


def _set(path: tuple, value):
    def apply(block: dict) -> dict:
        node = block
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return block
    return apply


@pytest.mark.parametrize(("mutate", "expected"), [
    (_set(("per_date", 0, "g1_missing_symbols"), 1), "g1_members"),
    (_set(("per_date", 2, "g1_extra_symbols"), 3), "g1_members"),
    (_set(("per_date", 2, "g1_duplicate_symbols"), 1), "g1_members"),
    (_set(("non_financial", "diff_cells"), 1), "g2_non_financial"),
    (_set(("non_financial", "keys_equal"), False), "g2_non_financial"),
    (_set(("per_date", 5, "g3_eligible_mismatch_rows"), 1), "g3_eligible"),
    (_set(("per_date", 1, "g4_unexplained_frozen_only"), 1), "g4_key_accounting"),
    (_set(("per_date", 1, "g4_unexplained_v2_only"), 2), "g4_key_accounting"),
    (_set(("per_date", 3, "common_keys"), 0), "common_keys"),
    (_set(("criteria", "g2_non_financial_diff_cells"), 5), "criteria differ from code constants"),
    (_set(("rule_version",), ""), "rule_version"),
    (_set(("completion_sha256",), "short"), "completion_sha256"),
    (_set(("snapshot_date",), None), "snapshot_date"),
    (_set(("per_date",), []), "per_date missing"),
])
def test_each_violation_fails_the_gate_and_the_top_level_failures(mutate, expected: str) -> None:
    block = mutate(_block())
    assert expected in us_parity_v2_failures(block)
    top = us_parity_failures(_evidence(universe_v2=block))
    assert f"universe_v2:{expected}" in top


def test_status_text_is_not_trusted_and_a_malformed_block_fails() -> None:
    bad = _with_block(status="passed")
    bad["universe_v2"]["non_financial"]["diff_cells"] = 4
    assert bad["universe_v2"]["status"] == "passed"
    assert us_parity_failures(bad) == ["universe_v2:g2_non_financial"]
    assert us_parity_failures(_evidence(universe_v2="x")) == ["universe_v2:malformed"]
    broken = _block()
    del broken["per_date"][0]["g3_eligible_mismatch_rows"]
    assert "malformed" in us_parity_v2_failures(broken)


# ---- validate_parity_evidence -----------------------------------------------------------------------


def _write(tmp_path: Path, evidence: dict, name: str = "e.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(evidence))
    return path


def test_validate_v1_ignores_the_block_and_needs_no_rule_version(tmp_path: Path) -> None:
    assert validate_parity_evidence(_write(tmp_path, _evidence()))["rule"] == RULE_DET_A
    assert validate_parity_evidence(_write(tmp_path, _with_block()))["rule"] == RULE_DET_A


def test_validate_v2_requires_a_passed_block_with_the_same_rule_version(tmp_path: Path) -> None:
    ok = _write(tmp_path, _with_block())
    got = validate_parity_evidence(ok, universe_version="v2", universe_v2_rule_version=RULE_VERSION)
    assert got["universe_v2"]["rule_version"] == RULE_VERSION

    v1_only = _write(tmp_path, _evidence(), "v1.json")
    with pytest.raises(ValueError, match="universe_v2 절"):
        validate_parity_evidence(v1_only, universe_version="v2", universe_v2_rule_version=RULE_VERSION)

    failed_label = _write(tmp_path, _with_block(status="failed"), "label.json")
    with pytest.raises(ValueError, match="통과하지 못했습니다"):
        validate_parity_evidence(failed_label, universe_version="v2", universe_v2_rule_version=RULE_VERSION)

    failed_numbers = _with_block()
    failed_numbers["universe_v2"]["per_date"][0]["g3_eligible_mismatch_rows"] = 2
    with pytest.raises(ValueError, match="gate 기준"):
        validate_parity_evidence(
            _write(tmp_path, failed_numbers, "num.json"),
            universe_version="v2", universe_v2_rule_version=RULE_VERSION)

    with pytest.raises(ValueError, match="rule_version"):
        validate_parity_evidence(ok, universe_version="v2", universe_v2_rule_version="ud2-2")
    with pytest.raises(ValueError, match="rule_version"):
        validate_parity_evidence(ok, universe_version="v2")
    with pytest.raises(ValueError, match="universe_version"):
        validate_parity_evidence(ok, universe_version="v3")


# ---- prepare in v2 mode ---------------------------------------------------------------------------------


def _prepare_v2(monkeypatch, root: DataRoot, evidence: Path | None, version: str = "v2") -> Path:
    _patch_root(monkeypatch, root)

    def fake_build(lake, scoring_date, *, financial_selection_rule, universe_version="v1"):
        return pl.DataFrame({"date": [scoring_date], "symbol": ["AAA"], "x": [1.0]})

    monkeypatch.setattr(us_daily, "build_daily_features", fake_build)
    return us_daily.prepare_daily_features(
        date(2026, 10, 2), raw_feature_parity_status="score_equivalent",
        raw_feature_parity_evidence=str(evidence), universe_version=version)


def test_prepare_v2_rejects_v1_only_evidence_and_a_rule_version_change(tmp_path: Path, monkeypatch) -> None:
    root, _ = _disk_lake(tmp_path / "lake")
    with pytest.raises(ValueError, match="universe_v2 절"):
        _prepare_v2(monkeypatch, root, _write(tmp_path, _evidence(), "v1.json"))
    stale = _write(tmp_path, _with_block(rule_version="seg-r3c.1+master-r3c.1+ud2-0"), "stale.json")
    with pytest.raises(ValueError, match="rule_version"):
        _prepare_v2(monkeypatch, root, stale)
    failed = _with_block()
    failed["universe_v2"]["per_date"][3]["g4_unexplained_v2_only"] = 1
    with pytest.raises(ValueError, match="gate 기준"):
        _prepare_v2(monkeypatch, root, _write(tmp_path, failed, "failed.json"))


def test_prepare_v2_accepts_a_matching_v2_evidence_and_v1_mode_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    root, _ = _disk_lake(tmp_path / "lake")
    evidence = _write(tmp_path, _with_block(rule_version=RULE_VERSION), "ok.json")
    manifest = json.loads(_prepare_v2(monkeypatch, root, evidence).read_text())
    assert manifest["universe_version"] == "v2"
    assert manifest["raw_feature_parity_status"] == "score_equivalent"
    assert manifest["serving_eligible"] is True
    assert manifest["raw_feature_parity_evidence_sha256"] == sha256_file(evidence)
    # v1 mode still takes a v1-only evidence (nothing about universe_v2 is required).
    v1_only = _write(tmp_path, _evidence(), "v1.json")
    v1 = json.loads(_prepare_v2(monkeypatch, root, v1_only, "v1").read_text())
    assert "universe_version" not in v1 and v1["serving_eligible"] is True


# ---- the v2 block builder -----------------------------------------------------------------------------------


def _frame(symbols: list[str], *, bump: dict | None = None, price: dict | None = None) -> pl.DataFrame:
    rows = []
    for day in GOLDEN_DATES:
        for i, sym in enumerate(symbols):
            row = {"date": day, "symbol": sym, "price_ge_5": (price or {}).get(sym, True),
                   "close": 10.0 + i, "adv_20d": 2e6}
            for j, name in enumerate(FEATURES):
                value = None if (name == FEATURES[0] and sym == "AAA") else 0.1 * i + 0.001 * j
                if bump and (sym, name) in bump:
                    value = bump[(sym, name)]
                row[name] = value
                row[f"{name}_isna"] = value is None
            rows.append(row)
    return pl.DataFrame(rows)


def _v2_root(tmp_path: Path, members: list[str], others: dict[str, tuple[str, str]]) -> DataRoot:
    """A lake holding one universe_daily_v2 snapshot: ``members`` in, ``others`` out with a reason."""
    root = DataRoot(tmp_path / "us")
    directory = root.derived / "snapshots" / UNIVERSE_V2_TABLE / f"snapshot_date={REVISION}"
    directory.mkdir(parents=True)
    rows = []
    for day in GOLDEN_DATES:
        rows += [(day, s, "common", True, None) for s in members]
        rows += [(day, s, kind, False, why) for s, (kind, why) in others.items()]
    pl.DataFrame(rows, schema=["date", "symbol", "security_type", "in_universe", "exclusion_reason"],
                 orient="row").write_parquet(directory / "part.parquet")
    (directory / "completion.json").write_text(json.dumps({
        "schema_version": 1, "table": UNIVERSE_V2_TABLE, "snapshot_date": REVISION,
        "snapshot_sha256": sha256_file(directory / "part.parquet"), "rule_version": RULE_VERSION,
        "unjudged_months": [], "input_snapshots": {"prices_daily": "2026-10-03"},
        "input_snapshot_sha256": {"prices_daily": "b" * 64}}))
    return root


def _build_block(monkeypatch, root: DataRoot, built: pl.DataFrame, frozen: pl.DataFrame) -> dict:
    monkeypatch.setattr(us_daily, "_golden_features", lambda *a, **k: built)
    return us_daily._universe_v2_block(
        data_root=root, base_map={}, base_lake=None, flow_lake=None, rule=RULE_DET_A,
        frozen=frozen, revision=REVISION)


def test_block_passes_and_records_counts_breakdowns_and_null_rates(tmp_path: Path, monkeypatch) -> None:
    root = _v2_root(tmp_path, ["AAA", "BBB", "NEW"], {"OLD": ("etf", "fund_name"), "ZZZ": ("adr", "adr")})
    frozen = _frame(["AAA", "BBB", "OLD", "ZZZ", "GONE"])
    # financial features may differ from frozen on the common keys: recorded, not gated
    built = _frame(["AAA", "BBB", "NEW"], bump={("BBB", FIN[0]): 123.0, ("AAA", FIN[1]): 9.0})
    block = _build_block(monkeypatch, root, built, frozen)
    assert block["status"] == "passed" and block["failures"] == []
    assert block["snapshot_date"] == REVISION and block["rule_version"] == RULE_VERSION
    assert len(block["completion_sha256"]) == 64 and block["criteria"] == US_PARITY_V2_CRITERIA
    assert [d["date"] for d in block["per_date"]] == [d.isoformat() for d in sorted(GOLDEN_DATES)]
    for d in block["per_date"]:
        assert (d["v2_rows"], d["common_keys"], d["v1_only"], d["v2_only"]) == (3, 2, 3, 1)
        assert d["v1_only_by_security_type_exclusion_reason"] == {
            "(no v2 row) | (no v2 row)": 1, "adr | adr": 1, "etf | fund_name": 1}
        assert d["v2_only_by_security_type"] == {"common": 1}
    assert block["non_financial"]["diff_cells"] == 0 and block["non_financial"]["common_keys"] == 12
    assert {(b["security_type"], b["exclusion_reason"]): b["rows"] for b in block["v1_only_breakdown"]} == {
        ("etf", "fund_name"): 6, ("adr", "adr"): 6, ("(no v2 row)", "(no v2 row)"): 6}
    assert block["v2_only_breakdown"] == [{"security_type": "common", "rows": 6}]
    fin = block["financial_vs_frozen_on_common_diff_cells"]
    assert fin[FIN[0]] == 6 and fin[FIN[1]] == 6 and sum(fin.values()) == 12
    nulls = block["null_rates"]
    assert (nulls["common_rows"], nulls["v2_only_rows"]) == (12, 6)
    assert nulls["by_feature"][FEATURES[0]] == {"common": 0.5, "v2_only": 0.0}
    json.dumps(block)  # serializable as is


def test_block_g2_fails_on_a_non_financial_cell_on_common_keys(tmp_path: Path, monkeypatch) -> None:
    root = _v2_root(tmp_path, ["AAA", "BBB"], {})
    frozen = _frame(["AAA", "BBB"])
    built = _frame(["AAA", "BBB"], bump={("BBB", NON_FIN[3]): 5.0, ("BBB", NON_FIN[7]): None})
    block = _build_block(monkeypatch, root, built, frozen)
    assert block["status"] == "failed" and block["failures"] == ["g2_non_financial"]
    diffs = block["non_financial"]["diff_by_feature"]
    assert {k: v for k, v in diffs.items() if v} == {NON_FIN[3]: 6, NON_FIN[7]: 6}


def test_block_g3_fails_when_price_ge_5_differs_on_common_keys(tmp_path: Path, monkeypatch) -> None:
    root = _v2_root(tmp_path, ["AAA", "BBB"], {})
    block = _build_block(monkeypatch, root, _frame(["AAA", "BBB"], price={"AAA": False}),
                         _frame(["AAA", "BBB"]))
    assert block["failures"] == ["g3_eligible"]
    assert sum(d["g3_eligible_mismatch_rows"] for d in block["per_date"]) == 6


def test_block_g1_and_g4_fail_when_built_members_differ_from_the_sealed_table(
    tmp_path: Path, monkeypatch
) -> None:
    root = _v2_root(tmp_path, ["AAA", "BBB", "CCC"], {})
    frozen = _frame(["AAA", "BBB", "CCC"])
    # CCC is a v2 member and in the frozen set, yet the panel dropped it; XTRA is not a member.
    block = _build_block(monkeypatch, root, _frame(["AAA", "BBB", "XTRA"]), frozen)
    assert block["status"] == "failed"
    assert {"g1_members", "g4_key_accounting"} <= set(block["failures"])
    d = block["per_date"][0]
    assert (d["g1_missing_symbols"], d["g1_extra_symbols"]) == (1, 1)
    assert d["g4_unexplained_frozen_only"] == 1 and d["g4_unexplained_v2_only"] == 1
    # an unexplained key on the frozen side: panel lacks a frozen key that is a v2 member
    block = _build_block(monkeypatch, root, _frame(["AAA", "BBB"]), frozen)
    assert block["per_date"][0]["g4_unexplained_frozen_only"] == 1
    assert "g4_key_accounting" in block["failures"]


def test_block_reads_pit_rows_only_when_the_table_has_a_view_column(tmp_path: Path, monkeypatch) -> None:
    root = _v2_root(tmp_path, ["AAA", "BBB"], {})
    part = root.derived / "snapshots" / UNIVERSE_V2_TABLE / f"snapshot_date={REVISION}" / "part.parquet"
    table = pl.read_parquet(part)
    extra = table.filter(pl.col("symbol") == "AAA").with_columns(
        pl.lit("BAD").alias("symbol"), pl.lit(True).alias("in_universe"))
    pl.concat([table.with_columns(pl.lit("pit").alias("view")),
               extra.with_columns(pl.lit("post").alias("view"))]).write_parquet(part)
    marker = part.parent / "completion.json"
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "snapshot_sha256": sha256_file(part)}))
    block = _build_block(monkeypatch, root, _frame(["AAA", "BBB"]), _frame(["AAA", "BBB"]))
    assert block["status"] == "passed"


def test_block_refuses_a_bad_completion(tmp_path: Path, monkeypatch) -> None:
    root = _v2_root(tmp_path, ["AAA"], {})
    marker = root.derived / "snapshots" / UNIVERSE_V2_TABLE / f"snapshot_date={REVISION}" / "completion.json"
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "unjudged_months": ["2026-09-01"]}))
    with pytest.raises(ValueError, match="unjudged"):
        _build_block(monkeypatch, root, _frame(["AAA"]), _frame(["AAA"]))


# ---- entry points ---------------------------------------------------------------------------------------------


def test_run_parity_check_rejects_a_revision_without_v2_and_an_unknown_version(tmp_path: Path) -> None:
    kwargs = dict(lake_root=tmp_path, frozen_root=tmp_path, bundle_root=tmp_path, rule=RULE_DET_A,
                  output=tmp_path / "out.json")
    with pytest.raises(ValueError, match="universe_v2_revision"):
        us_daily.run_parity_check(**kwargs, universe_v2_revision=REVISION)
    with pytest.raises(ValueError, match="universe_version"):
        us_daily.run_parity_check(**kwargs, universe_version="v3")


def test_parity_cli_options(tmp_path: Path) -> None:
    base = ["parity-check", "--rule", "det_a", "--output", str(tmp_path / "o.json"),
            "--lake-root", str(tmp_path), "--frozen-root", str(tmp_path), "--bundle-root", str(tmp_path)]
    for extra in (["--universe-v2-revision", REVISION],
                  ["--universe-version", "v2", "--universe-v2-revision", "10-06"],
                  ["--universe-version", "v3"]):
        with pytest.raises(SystemExit) as error:
            us_daily.main([*base, *extra])
        assert error.value.code == 2


def test_select_reason_has_the_universe_version_mismatch_text() -> None:
    assert SELECT_REASON_KO["us_universe_version_mismatch"] == "US 유니버스 버전이 설정과 다른 준비 입력"
