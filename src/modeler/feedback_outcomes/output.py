"""Immutable outcome runs with a last-step atomic current pointer."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

from modeler.feedback.core import canonical_bytes, sha256_bytes
from modeler.feedback.store import _atomic, _immutable, _writer, check_output_root

from .contracts import RULE_ID, OutcomeError, require_real_review, rule_pin, stamp


def _parquet(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = []
    for row in rows:
        normalized.append(
            {
                key: canonical_bytes(value).decode() if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    table = (
        pa.Table.from_pylist(normalized)
        if normalized
        else pa.table({"status": pa.array([], type=pa.string())})
    )
    stream = io.BytesIO()
    pq.write_table(table, stream, compression="zstd", version="2.6")
    return stream.getvalue()


def render_summary(result):
    text = ["# KR v1 게시 후 가상 보유 진단", ""]
    if result["synthetic_fixture"]:
        text += ["**합성 시험입니다. 실제 예측·가격·성과가 아닙니다.**", ""]
    text += [
        f"- 규칙: `{RULE_ID}`",
        "- 가격수익이며 배당은 제외합니다. 왕복 비용 가정은 60bp입니다.",
        "- 해결된 성숙 cohort의 조건부 평균입니다. 전체 표본 성과로 일반화하지 않습니다.",
        "- 겹치는 cohort는 독립 표본이 아닙니다.",
        "- 계좌 수익·연율 수익·Sharpe·채택 판정은 계산하지 않습니다.",
        "",
        "| 묶음 | 전체 | 성숙 | 해결 | 미해결 | 비용 후 평균 | 기준선 평균 | 초과 평균 | 상태 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("top20", "top100"):
        row = result["summary"][name]

        def value(key):
            return "null" if row[key] is None else f"{row[key]:.6f}"

        text.append(
            f"| {name} | {row['total_cohort_count']} | {row['mature_cohort_count']} | "
            f"{row['resolved_cohort_count']} | {row['unresolved_cohort_count']} | "
            f"{value('mean_net')} | {value('mean_benchmark')} | "
            f"{value('mean_excess')} | {row['sample_status']} |"
        )
    return "\n".join(text) + "\n"


def write_run(root, result, *, inputs, source_roots=(), now=None):
    """Preserve all prior versions; repeated identical inputs expose the same run."""
    root = Path(root)
    synthetic = result.get("synthetic_fixture") is True
    if type(result.get("synthetic_fixture")) is not bool or result.get("rule") != rule_pin():
        raise OutcomeError("output rule and fixture pins are required")
    if not synthetic:
        current = require_real_review(now or datetime.now(UTC))
        if stamp(result["cutoff"]) > current:
            raise OutcomeError("output cutoff has not arrived")
    if synthetic:
        if "stock_data" in root.parts:
            raise OutcomeError("synthetic artifacts cannot be written to real stock_data")
        root = root / "synthetic" / RULE_ID
    elif root.parts[-5:] != ("stock_data", "kr", "output", "prediction_feedback_outcomes", RULE_ID):
        raise OutcomeError("real outcomes require the frozen KR output layout")
    root = check_output_root(root, list(source_roots))
    files = {
        "rows.parquet": _parquet(result["rows"]),
        "cohorts.parquet": _parquet(result["cohorts"]),
        "summary.md": render_summary(result).encode(),
        "decisions.json": canonical_bytes(result["decisions"]),
    }
    code = {
        "feedback_outcomes/" + p.name: sha256_bytes(p.read_bytes())
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    package = Path(__file__).parent.parent
    for name in (
        "serving/calendars.py",
        "feedback/core.py",
        "feedback/store.py",
        "feedback/briefing.py",
        "feedback/capture.py",
        "feedback/captured.py",
        "reporting/markdown.py",
    ):
        code[name] = sha256_bytes((package / name).read_bytes())
    manifest = {
        "schema": "kr-v1-outcome-run.v1",
        "synthetic_fixture": synthetic,
        "rule": rule_pin(),
        "cutoff": result["cutoff"],
        "calendar_sha256": result["calendar_sha256"],
        "inputs": inputs,
        "code_files": code,
        "calendar_code_sha256": code["serving/calendars.py"],
        "summary": result["summary"],
        "nested_columns_format": "canonical JSON strings",
        "files": {name: sha256_bytes(raw) for name, raw in files.items()},
    }
    encoded = canonical_bytes(manifest)
    run_id = sha256_bytes(encoded)
    with _writer(root):
        target = root / "runs" / run_id
        for name, raw in files.items():
            _immutable(target / name, raw)
        _immutable(target / "run_manifest.json", encoded)
        _atomic(
            root / "current.json",
            canonical_bytes({"run_id": run_id, "manifest_sha256": sha256_bytes(encoded)}),
        )
    return {
        "run_id": run_id,
        "path": str(target),
        "synthetic_fixture": synthetic,
        "counts": {key: len(result[key]) for key in ("rows", "cohorts", "decisions")},
    }
