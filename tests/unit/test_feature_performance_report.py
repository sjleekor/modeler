from __future__ import annotations

import json
from pathlib import Path

import pytest
from modeler.analysis.feature_performance_report import (
    ReportContractError,
    _assert_safe_output,
    _cell_label,
    _figure_html,
    align_ic,
    family_name_union,
    paired_delta,
    resolve_and_validate_paths,
    t1_h20_decision,
)


def test_align_ic_preserves_positive_and_flips_negative_expectation() -> None:
    assert align_ic(0.03, "+") == pytest.approx(0.03)
    assert align_ic(-0.03, "-") == pytest.approx(0.03)
    assert align_ic(None, "+") is None
    assert align_ic(float("nan"), "+") is None


def test_family_name_union_uses_a_cards_and_b_summary_names() -> None:
    cards = [{"family": "a_one"}, {"family": "a_two"}]
    assert family_name_union(cards, ["b_one", "b_two"]) == {
        "a_one",
        "a_two",
        "b_one",
        "b_two",
    }


def test_paired_delta_reads_nested_metric() -> None:
    records = [
        {"name": "baseline", "horizon": 20, "economic": {"cost": 0.02}},
        {"name": "candidate", "horizon": 20, "economic": {"cost": 0.015}},
    ]
    assert paired_delta(
        records,
        horizon=20,
        candidate_name="candidate",
        metric_path=("economic", "cost"),
    ) == pytest.approx(-0.005)


def test_paired_delta_fails_when_candidate_is_missing() -> None:
    with pytest.raises(ReportContractError, match="candidate record"):
        paired_delta(
            [{"name": "baseline", "horizon": 20, "value": 1.0}],
            horizon=20,
            candidate_name="candidate",
            metric_path=("value",),
        )


def test_t1_h20_decision_uses_topk_cost_adjusted_return() -> None:
    data = {
        "records": [
            {
                "name": "baseline",
                "horizon": 20,
                "topk": {"cost_adjusted_return": 0.02},
            },
            {
                "name": "candidate",
                "horizon": 20,
                "topk": {"cost_adjusted_return": 0.015},
            },
        ]
    }
    decision, delta = t1_h20_decision(data)
    assert decision == "not_adopted_h20"
    assert delta == pytest.approx(-0.005)


@pytest.mark.parametrize(
    ("index_html", "manifest", "message"),
    [
        ("<script src='https://cdn.example/a.js'></script>", "{}", "외부 asset"),
        ("<p>/Users/someone/secret</p>", "{}", "홈 절대경로"),
        ("<p>ok</p>", '{"dsn":"postgresql://u:p@host/db"}', "비밀정보"),
    ],
)
def test_safe_output_rejects_external_and_sensitive_content(
    index_html: str, manifest: str, message: str
) -> None:
    with pytest.raises(ReportContractError, match=message):
        _assert_safe_output(index_html, manifest)


def test_safe_output_allows_inline_javascript() -> None:
    _assert_safe_output("<script>window.reportReady=true;</script>", '{"path":"docs/a.json"}')


def test_cell_label_uses_scan_type_when_cell_type_is_nan() -> None:
    assert (
        _cell_label({"cell_type": float("nan"), "scan_type": "cum", "h_start": 1, "h_end": 20})
        == "cum 1–20"
    )


def test_figure_html_rejects_webgl_trace() -> None:
    class Trace:
        type = "scattergl"

    class Figure:
        data = [Trace()]

        def to_html(self, **_kwargs):  # pragma: no cover - reject before rendering
            return ""

    with pytest.raises(ReportContractError, match="WebGL trace"):
        _figure_html(Figure())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_run_resolution_rejects_ab_snapshot_mismatch(tmp_path: Path) -> None:
    snapshot = "2026-08-23"
    source = "sj2_remote"
    config_hash = "a" * 64
    ab_run_id = "ab-run"
    ab_dir = (
        tmp_path
        / "horizon_scan/phase=AB"
        / f"snapshot_date={snapshot}"
        / f"source={source}"
        / f"config_hash={config_hash}"
        / f"run_id={ab_run_id}"
    )
    _write_json(
        ab_dir / "manifest.json",
        {
            "phase": "AB",
            "snapshot_date": "2026-08-22",
            "source": source,
            "config_hash": config_hash,
            "run_id": ab_run_id,
        },
    )
    _write_json(ab_dir / "_SUCCESS.json", {"run_id": ab_run_id, "content_hash": "ab-hash"})

    with pytest.raises(ReportContractError, match="AB snapshot"):
        resolve_and_validate_paths(
            tmp_path,
            snapshot_date=snapshot,
            source=source,
            config_hash=config_hash,
            ab_run_id=ab_run_id,
        )
