"""V7–V10: synthetic-only issue links and internal metadata diagnostics."""

from __future__ import annotations

import copy
import hashlib

import pytest

from modeler.feedback.core import canonical_bytes
from modeler.feedback.issues import build_occurrences, compile_mapping, render_diagnostic

COMMIT = "a" * 40
WARNING = "W20_halt_in_window"


def document(*rows: tuple[str, str, str], state: str = "confirmed") -> bytes:
    header = (
        "# Synthetic issues\n\n"
        "| issue_id | 문제·분류 / 대상 | reason_code | 확인 상태 "
        "| 조치·담당 / 다음 목표 | 완료 기준·근거 |\n"
        "|---|---|---|---|---|---|\n"
    )
    return (
        header
        + "".join(
            f"| {issue} | synthetic problem / {target} | `{reason}` | {state} | "
            "UNDEPLOYED_ACTION | UNVERIFIED_PERFORMANCE_CLAIM |\n"
            for issue, target, reason in rows
        )
    ).encode()


def compile_doc(raw: bytes, *, commit: str = COMMIT) -> dict:
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return compile_mapping(raw, my_commit=commit, expected_blob_sha1=blob)


def mapping() -> dict:
    return compile_doc(document(("KR-FB-0001", "운영 v1 가격 피쳐", WARNING)))


def prediction(**overrides) -> dict:
    return {
        "prediction_ref": "p-synthetic",
        "model_id": "kr_daily_h20_v1",
        "market": "KR",
        "symbol": "SYNTHETIC",
        "feature_asof_date": "2026-01-01",
        "rank": 1,
        "score": 0.5,
        "explain_sha256": "c" * 64,
        "publication_status": "published_verified",
        "quality_reasons": [WARNING],
        "checker_version": "checker-v1",
        "source_evidence_sha256": "b" * 64,
        "observed_at": "2026-01-02T01:00:00Z",
        "evaluation_status": "pending",
        **overrides,
    }


def test_v7_human_state_change_is_not_a_server_state_record():
    before = mapping()
    after = compile_doc(document(("KR-FB-0001", "운영 v1 가격 피쳐", WARNING), state="dismissed"))
    assert before["rules"] == after["rules"]
    assert before["document_blob_sha1"] != after["document_blob_sha1"]
    assert set(before["rules"][0]) == {"issue_id", "target", "reason_code"}
    serialized = canonical_bytes(before)
    for hidden in (b"confirmed", b"UNDEPLOYED_ACTION", b"UNVERIFIED_PERFORMANCE_CLAIM"):
        assert hidden not in serialized
    first = build_occurrences([prediction()], before)[0]
    changed = build_occurrences([prediction(observed_at="2026-01-03T01:00:00Z")], after)[0]
    assert first["occurrence_ref"] == changed["occurrence_ref"]


def test_v7_mapping_change_updates_link_without_increasing_occurrence_identity():
    other = compile_doc(document(("KR-FB-0099", "운영 v1 가격 피쳐", WARNING)), commit="d" * 40)
    first = build_occurrences([prediction()], mapping())[0]
    changed = build_occurrences([prediction()], other)[0]
    assert first["occurrence_ref"] == changed["occurrence_ref"]
    assert (first["issue_id"], changed["issue_id"]) == ("KR-FB-0001", "KR-FB-0099")
    different_day = prediction(prediction_ref="next-day-prediction")
    records = build_occurrences([prediction(), different_day], mapping())
    assert len(records) == 2
    assert {record["issue_id"] for record in records} == {"KR-FB-0001"}


def test_v7_repeated_warnings_and_reordered_input_are_idempotent():
    old = prediction(quality_reasons=[WARNING, WARNING])
    new = prediction(observed_at="2026-01-03T01:00:00Z")
    first = build_occurrences([old, new, old], mapping())
    assert len(first) == 1
    assert first == build_occurrences([new, old], mapping())
    assert first[0]["observed_at"] == old["observed_at"]
    offset = prediction(observed_at="2026-01-02T09:30:00+09:00")
    assert build_occurrences([old, offset], mapping())[0]["observed_at"] == offset["observed_at"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_ref", "different"),
        ("quality_reasons", ["different_warning"]),
        ("checker_version", "checker-v2"),
        ("source_evidence_sha256", "e" * 64),
    ],
)
def test_v7_each_identity_component_distinguishes_occurrences(field, value):
    records = build_occurrences([prediction(), prediction(**{field: value})], mapping())
    assert len(records) == 2


@pytest.mark.parametrize("status", ["generated", "local_ready", "publication_unverified", None])
def test_nonpublished_warning_remains_diagnostic_only(status):
    row = prediction(publication_status=status)
    assert build_occurrences([row], mapping()) == []
    rendered = render_diagnostic([row], [], issue_document_ref="issues.md@synthetic")
    assert WARNING.replace("_", r"\_") in rendered
    assert "KR-FB-0001" not in rendered


def test_publication_evidence_is_accepted():
    assert (
        len(build_occurrences([prediction(publication_status="published_evidence")], mapping()))
        == 1
    )


def test_v8_human_claims_are_excluded_from_mapping_and_diagnostic():
    row = prediction(issue_status="confirmed", action_status="verified", performance=1234567)
    rendered = render_diagnostic(
        [row], build_occurrences([row], mapping()), issue_document_ref="issues.md@synthetic"
    )
    assert "KR-FB-0001" in rendered
    assert "issues\\.md@synthetic" in rendered
    assert "confirmed" not in rendered and "UNDEPLOYED_ACTION" not in rendered
    assert "1234567" not in rendered


def test_v9_provenance_is_preserved_without_inference_or_mutation():
    row = prediction(
        provenance={"fact_semantics": "v3", "quarter_semantics": "v2", "past_version": "unknown"}
    )
    before = copy.deepcopy(row)
    build_occurrences([row], mapping())
    render_diagnostic([row], [], issue_document_ref="issues.md@synthetic")
    assert row == before


def test_v10_unknown_warning_is_kept_unmapped():
    records = build_occurrences([prediction(quality_reasons=[WARNING, "new_warning"])], mapping())
    by_reason = {row["reason_code"]: row for row in records}
    assert by_reason["new_warning"]["mapping_status"] == "unmapped"
    assert by_reason["new_warning"]["issue_id"] is None
    assert by_reason[WARNING]["issue_id"] == "KR-FB-0001"


def test_v10_git_blob_hash_is_checked_instead_of_bare_content_sha1():
    raw = document(("KR-FB-0001", "운영 v1 가격 피쳐", WARNING))
    with pytest.raises(ValueError, match="blob hash mismatch"):
        compile_mapping(raw, my_commit=COMMIT, expected_blob_sha1=hashlib.sha1(raw).hexdigest())
    with pytest.raises(ValueError, match="my_commit"):
        compile_doc(raw, commit="not-a-commit")


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_blob_sha1", "0" * 40),
        ("document_sha256", "0" * 64),
        ("mapping_sha256", "0" * 64),
        ("my_commit", "0" * 40),
    ],
)
def test_v10_tampered_mapping_is_rejected_before_issue_linking(field, value):
    invalid = {**mapping(), field: value}
    with pytest.raises(ValueError, match="digest mismatch"):
        build_occurrences([prediction()], invalid)


def test_v10_ambiguous_rules_and_duplicate_issues_fail_closed():
    with pytest.raises(ValueError, match="ambiguous"):
        compile_doc(document(("A", "model-a", WARNING), ("B", "model-a", WARNING)))
    with pytest.raises(ValueError, match="duplicate issue"):
        compile_doc(document(("A", "model-a", WARNING), ("A", "model-b", "other")))
    with pytest.raises(ValueError, match="unambiguous"):
        compile_doc(document(("A", "model-a", "warning_a,warning_b")))


def test_reason_codes_are_scoped_to_models_and_known_alias_cannot_be_overridden():
    compiled = compile_doc(document(("A", "model-a", WARNING), ("B", "model-b", WARNING)))
    rows = [
        prediction(prediction_ref=model, model_id=model)
        for model in ("model-a", "model-b", "model-c")
    ]
    by_ref = {row["prediction_ref"]: row for row in build_occurrences(rows, compiled)}
    assert [by_ref[model]["issue_id"] for model in ("model-a", "model-b", "model-c")] == [
        "A",
        "B",
        None,
    ]
    wrong = prediction(
        model_id="us_model",
        market="US",
        target="운영 v1 가격 피쳐",
        issue_target="운영 v1 가격 피쳐",
    )
    assert build_occurrences([wrong], mapping())[0]["issue_id"] is None
    wrong_market = prediction(market="US")
    assert build_occurrences([wrong_market], mapping())[0]["issue_id"] is None


def test_unimplemented_reasons_are_omitted():
    compiled = compile_doc(document(("A", "model-a", "—"), ("B", "model-b", "-")))
    assert compiled["rules"] == []
    assert build_occurrences([prediction()], compiled)[0]["mapping_status"] == "unmapped"


def test_diagnostic_escapes_text_and_never_fills_missing_evaluation_with_zero():
    row = prediction(
        model_id="a|b\n<script>x</script>",
        symbol="[link](bad) ~ *x*",
        rank=None,
        score=None,
        evaluation_status="sealed",
        check_ref=None,
        evaluation={"return": 987654321},
        human_state="investigating",
    )
    rendered = render_diagnostic([row], [], issue_document_ref="issues.md@abc|def")
    assert "<script>" not in rendered
    assert "a\\|b &lt;script&gt;x&lt;/script&gt;" in rendered
    assert "\\[link\\]\\(bad\\) \\~ \\*x\\*" in rendered
    assert "| unknown | unknown |" in rendered
    assert "| sealed | unknown |" in rendered
    assert "987654321" not in rendered
    assert "investigating" not in rendered
    assert "| 0 |" not in rendered
    assert "2026-01-01" in rendered


def test_top100_boundary_is_shown_as_not_checked_and_refs_are_not_payloads():
    row = prediction(quality_reasons=[], quality_review="not_checked", rank=101)
    assert "not\\_checked" in render_diagnostic([row], [], issue_document_ref="issues.md")
    with pytest.raises(ValueError, match="scalar metadata"):
        render_diagnostic(
            [prediction(check_ref={"outcome_value": 999})], [], issue_document_ref="issues.md"
        )
