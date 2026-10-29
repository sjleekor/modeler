"""Synthetic-only tests for metadata binding; never open any data lake input."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from modeler.feedback.briefing import read_briefing
from modeler.feedback.core import sha256_bytes
from modeler.reporting.markdown import c, front_matter, unit_sha256

DAY = "2026-10-29"
NOW = "2026-10-30T12:00:00+09:00"
HASH = "a" * 64
INVOCATION = "synthetic-invocation"


def write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(tmp_path, *, us=False):
    root = tmp_path / DAY
    market = {
        "market": "US" if us else "KR",
        "report_date": DAY,
        "decision_at": DAY + "T10:00:00+09:00",
        "feature_asof_date": "2026-10-28",
        "model_id": "us_synthetic" if us else "kr_daily_h20_v1",
        "model_version": "1.0.0",
        "rankings": [
            {
                "symbol": "TEST",
                "rank": 1,
                "score": 0.5,
                "quality_review": True,
                "quality_reasons": ["K_halt_or_unknown"],
            }
        ],
        "provenance": {
            key: HASH
            for key in (
                "bundle_manifest_sha256",
                "input_sha256",
                "code_inventory_sha256",
                "native_prepare_manifest_sha256",
                "bundle_manifest_content_sha256",
            )
        },
    }
    if not us:
        market["rankings"][0]["market"] = "KOSPI"
    report = {
        "schema_version": "1.0",
        "report_date": DAY,
        "markets": [market],
        "decision_at": DAY + "T10:00:00+09:00",
    }
    write(root / f"report-{DAY}.json", report)
    digest = sha256_bytes((root / f"report-{DAY}.json").read_bytes())
    write(root / "run-state.json", {"invocation_id": INVOCATION})
    write(root / "reports-publisher-config.json", {"release": "synthetic-release"})
    write(
        root / "coordinator-inference.json",
        {
            "report_date": DAY,
            "report_sha256": digest,
            "invocation_id": INVOCATION,
            "status": "inferred",
            "this_invocation_completed": True,
        },
    )
    return root, report, digest


def update_report(root, report):
    write(root / f"report-{DAY}.json", report)
    digest = sha256_bytes((root / f"report-{DAY}.json").read_bytes())
    write(
        root / "coordinator-inference.json",
        {
            "report_date": DAY,
            "report_sha256": digest,
            "invocation_id": INVOCATION,
        },
    )
    return digest


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def publication(root, tmp_path, digest, *, revision=1, shown_ranks=None, missing_model_link=False):
    checkout = tmp_path / "reports"
    checkout.mkdir(exist_ok=True)
    if not (checkout / ".git").exists():
        git(checkout, "init")
    files = {}
    report = json.loads((root / f"report-{DAY}.json").read_text())
    for name in ("README.md", "market-sector.md", "kr-stocks.md", "us-stocks.md", "data-status.md"):
        market = {"kr-stocks.md": "KR", "us-stocks.md": "US"}.get(name)
        models = [model for model in report["markets"] if model["market"] == market]
        meta = {
            "schema": "stock-reports.v1",
            "unit": DAY,
            "revision": revision,
            "section": "summary" if name == "README.md" else name.removesuffix(".md"),
            "models": [model["model_id"] for model in models],
            "source": {"release": "synthetic-release", "report_sha256": digest},
        }
        lines = ["Synthetic only", ""]
        for model in models:
            model_id = model["model_id"]
            lines.extend([f"## {model_id}", ""])
            if not missing_model_link:
                lines.extend(
                    [
                        f"- 모델: [Synthetic](../../reference/models/{model_id}.md) (`{model_id}`)",
                        "",
                    ]
                )
            rows = [
                row
                for row in model["rankings"]
                if shown_ranks is None or row["rank"] in shown_ranks.get(model_id, set())
            ]
            if rows:
                lines.extend(["| 순위 | 코드 | 점수 |", "| ---: | --- | ---: |"])
                lines.extend(
                    f"| {row['rank']} | {c(row['symbol'])} | {row['score']} |" for row in rows
                )
                lines.append("")
            else:
                lines.extend(["순위 표를 내지 않았습니다.", ""])
        files[name] = front_matter(meta) + "\n" + "\n".join(lines)
        path = checkout / f"reports/daily-briefing/2026/10/{DAY}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[name])
    git(checkout, "add", "reports")
    git(
        checkout,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=synthetic@example.invalid",
        "commit",
        "-m",
        f"Synthetic revision {revision}",
    )
    commit = git(checkout, "rev-parse", "HEAD")
    write(
        root / "markdown/manifest.json",
        {
            "unit": DAY,
            "release": "synthetic-release",
            "report_sha256": digest,
            "invocation_id": INVOCATION,
            "unit_sha256": unit_sha256(files),
        },
    )
    write(
        root / "coordinator-publication.json",
        {
            "status": "published",
            "report_sha256": digest,
            "reports_commit": commit,
        },
    )
    write(root / "remote-verification.json", {"status": "contained", "reports_commit": commit})
    return checkout


def reasons(batch):
    return {item["reason"] for item in batch.errors}


def build_case(tmp_path):
    """Public synthetic fixture for integration tests."""
    root, _, digest = fixture(tmp_path)
    return root, publication(root, tmp_path, digest)


def test_sidecar_invocation_and_stable_prediction_identity(tmp_path):
    root, _, _ = fixture(tmp_path)
    first = read_briefing(root, observed_at=NOW)
    second = read_briefing(root, observed_at="2026-11-01T13:00:00Z")
    assert not first.errors
    assert len(first.predictions) == 1
    assert first.predictions[0]["prediction_ref"] == second.predictions[0]["prediction_ref"]
    assert first.predictions[0]["evaluation_status"] == "pending"
    assert first.predictions[0]["historical_link"] is True
    assert first.publications[0]["status"] == "publication_unverified"
    assert first.findings[0]["reason_code"] == "K_halt_or_unknown"


def test_us_metadata_is_sealed_and_never_reads_external_input(tmp_path):
    root, report, _ = fixture(tmp_path, us=True)
    report["markets"][0]["provenance"]["price_source"] = str(tmp_path / "labels" / "secret.parquet")
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert len(batch.predictions) == 1
    assert batch.predictions[0]["security_market"] == "US"
    assert batch.predictions[0]["evaluation_status"] == "sealed"
    assert all("parquet" not in source["path"] for source in batch.sources)


def test_inconsistent_sidecar_rejects_every_row_with_explicit_errors(tmp_path):
    root, _, _ = fixture(tmp_path)
    write(root / "run-state.json", {"invocation_id": "old-invocation"})
    batch = read_briefing(root, observed_at=NOW)
    assert batch.predictions == []
    assert "invocation_missing_or_mismatch" in reasons(batch)
    assert any("rankings/0" in error["source"] for error in batch.errors)


def test_same_symbol_different_security_markets_does_not_collide(tmp_path):
    root, report, _ = fixture(tmp_path)
    other = copy.deepcopy(report["markets"][0]["rankings"][0])
    other.update(market="KOSDAQ", rank=2)
    report["markets"][0]["rankings"].append(other)
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert len({row["prediction_ref"] for row in batch.predictions}) == 2


def test_invalid_row_is_retained_as_error_and_non_top_quality_is_not_checked(tmp_path):
    root, report, _ = fixture(tmp_path)
    report["markets"][0]["rankings"].extend(
        [
            {"symbol": "BAD", "market": "KOSPI", "rank": 2, "score": "unknown"},
            {
                "symbol": "LOW",
                "market": "KOSPI",
                "rank": 101,
                "score": 0.1,
                "quality_reasons": ["should_not_read"],
            },
        ]
    )
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert len(batch.predictions) == 2
    assert "ranking_invalid" in reasons(batch)
    assert batch.predictions[1]["quality_review"] == "not_checked"
    assert batch.predictions[1]["quality_reasons"] == []


def test_standalone_cannot_be_mistaken_for_daily_file_hash_producer(tmp_path):
    root, report, _ = fixture(tmp_path)
    write(root / f"report-{DAY}.json", report["markets"][0])
    batch = read_briefing(root, observed_at=NOW)
    assert not batch.predictions
    assert "daily_envelope_required" in reasons(batch)


def test_explain_file_hash_invocation_and_exact_row_binding(tmp_path):
    root, _, digest = fixture(tmp_path)
    explain = {
        "schema": "briefing-explain.v1",
        "unit": DAY,
        "release": "synthetic-release",
        "report_sha256": digest,
        "invocation_id": INVOCATION,
        "models": {
            "kr_daily_h20_v1": {
                "rows": [{"symbol": "TEST", "rank": 1, "score": 0.5}],
                "inputs": {"prepared_sha256": HASH, "bundle_sha256": HASH},
            }
        },
    }
    explain_path = root / f"explain-{DAY}.json"
    write(explain_path, explain)
    explain_hash = sha256_bytes(explain_path.read_bytes())
    write(root / "coordinator-explain.json", {"document_sha256": explain_hash})
    batch = read_briefing(root, observed_at=NOW)
    assert batch.predictions[0]["explain_sha256"] == explain_hash
    explain["invocation_id"] = "stale"
    write(explain_path, explain)
    write(
        root / "coordinator-explain.json",
        {"document_sha256": sha256_bytes(explain_path.read_bytes())},
    )
    batch = read_briefing(root, observed_at=NOW)
    assert batch.predictions[0]["explain_sha256"] is None
    assert "explain_invocation_mismatch" in reasons(batch)


def test_explain_collision_is_not_arbitrarily_assigned(tmp_path):
    root, report, _ = fixture(tmp_path)
    row = copy.deepcopy(report["markets"][0]["rankings"][0])
    row.update(market="KOSDAQ", rank=2)
    report["markets"][0]["rankings"].append(row)
    digest = update_report(root, report)
    write(
        root / f"explain-{DAY}.json",
        {
            "schema": "briefing-explain.v1",
            "unit": DAY,
            "release": "synthetic-release",
            "report_sha256": digest,
            "invocation_id": INVOCATION,
            "models": {
                "kr_daily_h20_v1": {
                    "rows": [{"symbol": "TEST", "rank": 1, "score": 0.5}],
                    "inputs": {"prepared_sha256": HASH, "bundle_sha256": HASH},
                }
            },
        },
    )
    write(
        root / "coordinator-explain.json",
        {"document_sha256": sha256_bytes((root / f"explain-{DAY}.json").read_bytes())},
    )
    batch = read_briefing(root, observed_at=NOW)
    assert all(row["explain_sha256"] is None for row in batch.predictions)
    assert "explain_symbol_ambiguous" in reasons(batch)


def test_local_git_and_old_remote_evidence_have_no_verified_timestamp(tmp_path):
    root, _, digest = fixture(tmp_path)
    checkout = publication(root, tmp_path, digest)
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert not batch.errors
    record = batch.publications[0]
    assert record["status"] == "published_evidence"
    assert record["historical_publication_evidence"] is True
    assert record["verified_by"] is None and record["verified_at"] is None
    assert record["revision"] == 1
    assert batch.predictions[0]["publication_status"] == "published_evidence"
    assert record["prediction_refs"] == [batch.predictions[0]["prediction_ref"]]
    assert sum(source["path"].startswith("git:") for source in batch.sources) == 5


def test_wording_revision_keeps_prediction_changes_publication_reference(tmp_path):
    root, _, digest = fixture(tmp_path)
    checkout = publication(root, tmp_path, digest)
    first = read_briefing(root, observed_at=NOW, checkout=checkout)
    publication(root, tmp_path, digest, revision=2)
    second = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert first.predictions[0]["prediction_ref"] == second.predictions[0]["prediction_ref"]
    assert first.publications[0]["publication_ref"] != second.publications[0]["publication_ref"]


def test_bad_unit_hash_or_stale_remote_never_yields_published_evidence(tmp_path):
    root, _, digest = fixture(tmp_path)
    checkout = publication(root, tmp_path, digest)
    manifest_path = root / "markdown/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unit_sha256"] = "b" * 64
    write(manifest_path, manifest)
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert batch.publications[0]["status"] == "publication_unverified"
    assert "git_unit_hash_mismatch" in reasons(batch)


def test_manifest_symlink_is_not_followed(tmp_path):
    root, _, _ = fixture(tmp_path)
    target = tmp_path / "outside.json"
    write(target, {"unit": DAY})
    (root / "markdown").mkdir()
    (root / "markdown/manifest.json").symlink_to(target)
    batch = read_briefing(root, observed_at=NOW)
    assert batch.publications[0]["status"] == "publication_unverified"
    assert any("symlink" in error["reason"] for error in batch.errors)
    assert all(source["path"] != "markdown/manifest.json" for source in batch.sources)


@pytest.mark.parametrize(
    "changed,expected",
    [
        ({"schema": "random.v1"}, "explain_schema_invalid"),
        ({"unit": "2026-10-28"}, "explain_unit_mismatch"),
        ({"release": "old-release"}, "explain_release_mismatch"),
        ({"prepared_sha256": "b" * 64}, "explain_model_inputs_mismatch"),
        ({"bundle_sha256": "b" * 64}, "explain_model_inputs_mismatch"),
        ({"score": 0.6}, "explain_row_mismatch"),
    ],
)
def test_explain_metadata_must_match_source(tmp_path, changed, expected):
    root, _, digest = fixture(tmp_path)
    document = {
        "schema": "briefing-explain.v1",
        "unit": DAY,
        "release": "synthetic-release",
        "report_sha256": digest,
        "invocation_id": INVOCATION,
        "models": {
            "kr_daily_h20_v1": {
                "rows": [{"symbol": "TEST", "rank": 1, "score": 0.5}],
                "inputs": {"prepared_sha256": HASH, "bundle_sha256": HASH},
            }
        },
    }
    if "prepared_sha256" in changed or "bundle_sha256" in changed:
        document["models"]["kr_daily_h20_v1"]["inputs"].update(changed)
    elif "score" in changed:
        document["models"]["kr_daily_h20_v1"]["rows"][0].update(changed)
    else:
        document.update(changed)
    write(root / f"explain-{DAY}.json", document)
    write(
        root / "coordinator-explain.json",
        {"document_sha256": sha256_bytes((root / f"explain-{DAY}.json").read_bytes())},
    )
    batch = read_briefing(root, observed_at=NOW)
    assert len(batch.predictions) == 1
    assert batch.predictions[0]["explain_sha256"] is None
    assert expected in reasons(batch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_asof_date", DAY),
        ("feature_asof_date", "2026-10-30"),
        ("decision_at", DAY + "T10:00:00"),
        ("decision_at", DAY + "T11:00:00+09:00"),
    ],
)
def test_invalid_model_dates_are_explicitly_rejected(tmp_path, field, value):
    root, report, _ = fixture(tmp_path)
    report["markets"][0][field] = value
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert not batch.predictions
    assert "source_incomplete" in reasons(batch)


def test_new_report_same_day_creates_new_prediction_and_refuses_old_sidecars(tmp_path):
    root, report, _ = fixture(tmp_path)
    first = read_briefing(root, observed_at=NOW)
    report["markets"][0]["rankings"][0]["score"] = 0.8
    update_report(root, report)
    second = read_briefing(root, observed_at=NOW)
    assert first.predictions[0]["prediction_ref"] != second.predictions[0]["prediction_ref"]
    write(
        root / "coordinator-render.json",
        {"report_sha256": first.predictions[0]["report_sha256"], "invocation_id": INVOCATION},
    )
    third = read_briefing(root, observed_at=NOW)
    assert not third.predictions
    assert "report_sha256_mismatch" in reasons(third)


def test_monitor_for_other_date_does_not_confirm_current_publication(tmp_path):
    root, _, digest = fixture(tmp_path)
    checkout = publication(root, tmp_path, digest)
    write(
        root / "monitor-0.json",
        {"report_date": "2026-10-28", "status": "verified", "axes": {"remote": "contained"}},
    )
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert batch.publications[0]["status"] == "publication_unverified"
    assert batch.publications[0]["verified_by"] is None


def test_malformed_quality_has_explicit_invalid_status(tmp_path):
    root, report, _ = fixture(tmp_path)
    report["markets"][0]["rankings"][0]["quality_reasons"] = "bad"
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert len(batch.predictions) == 1
    assert batch.predictions[0]["quality_validation"] == "source_invalid"
    assert "quality_reasons_invalid" in reasons(batch)


def test_invalid_security_market_is_not_used_as_an_identity(tmp_path):
    root, report, _ = fixture(tmp_path)
    report["markets"][0]["rankings"][0]["market"] = "unknown"
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert not batch.predictions
    assert "ranking_invalid" in reasons(batch)


def test_publication_only_links_rows_in_immutable_git_ranking_table(tmp_path):
    root, report, _ = fixture(tmp_path)
    report["markets"][0]["rankings"].append(
        {
            "rank": 2,
            "symbol": "HIDDEN",
            "market": "KOSPI",
            "score": 0.3,
            "quality_reasons": ["K_halt_or_unknown"],
        }
    )
    digest = update_report(root, report)
    checkout = publication(root, tmp_path, digest, shown_ranks={"kr_daily_h20_v1": {1}})
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert not batch.errors
    assert len(batch.predictions) == 2
    assert batch.publications[0]["prediction_refs"] == [batch.predictions[0]["prediction_ref"]]
    assert len(batch.publications[0]["report_prediction_refs"]) == 2
    assert batch.predictions[0]["publication_status"] == "published_evidence"
    assert batch.predictions[1]["publication_status"] == "not_published"
    assert batch.publications[0]["release"] == "synthetic-release"


@pytest.mark.parametrize("status", ["stale", "failed", "unavailable"])
def test_suppressed_or_failed_model_rows_are_not_promoted_from_report(tmp_path, status):
    root, report, _ = fixture(tmp_path)
    report["markets"][0]["status"] = status
    digest = update_report(root, report)
    checkout = publication(root, tmp_path, digest, shown_ranks={})
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert not batch.errors
    assert batch.publications[0]["status"] == "published_evidence"
    assert batch.publications[0]["prediction_refs"] == []
    assert len(batch.predictions) == 1
    assert batch.predictions[0]["publication_status"] == "not_published"


def test_us_table_blocks_bind_each_model_separately(tmp_path):
    root, report, _ = fixture(tmp_path, us=True)
    other = copy.deepcopy(report["markets"][0])
    other["model_id"] = "us_other_model"
    report["markets"].append(other)
    digest = update_report(root, report)
    checkout = publication(root, tmp_path, digest, shown_ranks={"us_other_model": {1}})
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert not batch.errors
    assert len(batch.predictions) == 2
    first, second = batch.predictions
    assert first["publication_status"] == "not_published"
    assert second["publication_status"] == "published_evidence"
    assert batch.publications[0]["prediction_refs"] == [second["prediction_ref"]]


def test_ranking_table_without_model_binding_does_not_promote_rows(tmp_path):
    root, _, digest = fixture(tmp_path)
    checkout = publication(root, tmp_path, digest, missing_model_link=True)
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert batch.publications[0]["prediction_refs"] == []
    assert batch.predictions[0]["publication_status"] == "not_published"
    assert "publication_ranking_model_unverified" in reasons(batch)


def test_ambiguous_published_symbol_and_rank_does_not_choose_security_market(tmp_path):
    root, report, _ = fixture(tmp_path)
    other = copy.deepcopy(report["markets"][0]["rankings"][0])
    other["market"] = "KOSDAQ"
    report["markets"][0]["rankings"].append(other)
    digest = update_report(root, report)
    checkout = publication(root, tmp_path, digest)
    batch = read_briefing(root, observed_at=NOW, checkout=checkout)
    assert len(batch.predictions) == 2
    assert batch.publications[0]["prediction_refs"] == []
    assert all(row["publication_status"] == "not_published" for row in batch.predictions)
    assert "publication_ranking_unmatched_or_ambiguous" in reasons(batch)


def test_invalid_empty_model_is_an_explicit_error(tmp_path):
    root, report, _ = fixture(tmp_path)
    report["markets"][0].update(rankings=[], feature_asof_date=DAY)
    update_report(root, report)
    batch = read_briefing(root, observed_at=NOW)
    assert not batch.predictions
    assert "source_incomplete" in reasons(batch)
    assert any(error["source"].endswith("markets/0") for error in batch.errors)
