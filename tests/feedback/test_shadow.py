"""Synthetic metadata only; panels stay opaque and evaluation paths stay sealed."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest

from modeler.feedback.core import canonical_bytes, identity, sha256_bytes
from modeler.feedback.shadow import read_shadow

OBSERVED = "2026-10-30T12:00:00+09:00"
BUNDLE = "b" * 64


def _write_record(
    root: Path,
    *,
    d="2026-10-20",
    t="2026-10-19",
    attempt=1,
    valid=True,
    rows=None,
    files_extra=None,
    d0="2026-10-20",
) -> dict:
    rel = f"D={d}" + (f"/attempt-{attempt}" if attempt > 1 else "")
    folder = root / rel
    folder.mkdir(parents=True)
    # Deliberately not valid parquet: the adapter must hash, never parse, panels.
    (folder / "panel.parquet").write_bytes(b"synthetic opaque panel - do not parse")
    rows = (
        rows
        if rows is not None
        else [
            {
                "ticker": "000001",
                "market": "KOSPI",
                "rank": 1,
                "score": 0.42,
                "sealed_return": 999.0,
            }
        ]
    )
    pl.DataFrame(rows).write_parquet(folder / "predictions.parquet")
    files = {
        name: sha256_bytes((folder / name).read_bytes())
        for name in ("panel.parquet", "predictions.parquet")
    }
    files.update(files_extra or {})
    written = f"{d}T{'14' if valid else '16'}:00:00+09:00"
    rec = {
        "schema_version": "kr-next-shadow-record.v1",
        "D": d,
        "t": t,
        "attempt": attempt,
        "valid": valid,
        "d0": d0,
        "is_grid_date": True,
        "grid_k": 0,
        "decision_at": f"{d}T10:00:00+09:00",
        "deadline_at": f"{d}T15:00:00+09:00",
        "written_at": written,
        "files": files,
        "bundle": {"manifest_sha256": BUNDLE},
        "code": {"inventory_sha256": "c" * 64},
        "panel_manifest": {"synthetic": True},
        "panel_manifest_sha256": "d" * 64,
    }
    raw = canonical_bytes(rec)
    (folder / "record.json").write_bytes(raw)
    return {
        "schema": "kr-next-shadow-ledger.v1",
        "D": d,
        "attempt": attempt,
        "dir": rel,
        "record_sha256": sha256_bytes(raw),
        "valid": valid,
        "written_at": written,
        "has_basket": False,
    }


def _ledger(root: Path, lines: list[dict]) -> None:
    previous = "0" * 64
    texts = []
    for line in lines:
        line["prev_ledger_sha256"] = previous
        text = canonical_bytes(line)
        texts.append(text)
        previous = sha256_bytes(text)
    (root / "ledger.jsonl").write_bytes(b"\n".join(texts) + b"\n")


def test_links_all_attempts_and_only_first_valid_counts(tmp_path):
    lines = [
        _write_record(tmp_path, attempt=1, valid=False, d0=None),
        _write_record(tmp_path, attempt=2),
        _write_record(tmp_path, attempt=3),
    ]
    _ledger(tmp_path, lines)
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.errors
    assert len(batch.predictions) == 3
    assert [row["valid"] for row in batch.predictions] == [False, True, True]
    assert [row["counting"] for row in batch.predictions] == [False, True, False]
    assert [row["attempt"] for row in batch.predictions] == [1, 2, 3]
    assert [row["d0"] for row in batch.predictions] == [None, "2026-10-20", "2026-10-20"]
    row = batch.predictions[1]
    assert row["report_date"] == "2026-10-20" and row["feature_asof_date"] == "2026-10-19"
    assert row["market"] == "KR" and row["security_market"] == "KOSPI"
    assert row["bundle_hash"] == BUNDLE and row["evaluation_status"] == "awaiting_check"
    assert row["prediction_ref"] == identity(
        "shadow",
        {
            "bundle_hash": BUNDLE,
            "record_sha256": lines[1]["record_sha256"],
            "ticker": "000001",
            "market": "KOSPI",
        },
    )
    assert "sealed_return" not in row
    assert {s["hash_kind"] for s in batch.sources} == {"bytes_sha256"}
    assert {sha256_bytes(raw) for raw in batch.artifacts.values()} == set(batch.artifacts)
    assert len(batch.artifacts) == 4  # ledger and small record metadata only


@pytest.mark.parametrize("corruption", ["chain", "record", "torn", "order", "attempt"])
def test_ledger_corruption_rejects_connection(tmp_path, corruption):
    first = _write_record(tmp_path)
    second = _write_record(tmp_path, d="2026-10-21", t="2026-10-20", d0="2026-10-20")
    if corruption == "order":
        _ledger(tmp_path, [second, first])
    elif corruption == "attempt":
        second["attempt"] = 2
        _ledger(tmp_path, [first, second])
    else:
        _ledger(tmp_path, [first, second])
        if corruption == "chain":
            raw = (tmp_path / "ledger.jsonl").read_bytes()
            (tmp_path / "ledger.jsonl").write_bytes(
                raw.replace(b'"prev_ledger_sha256":"0', b'"prev_ledger_sha256":"1', 1)
            )
        elif corruption == "record":
            (tmp_path / first["dir"] / "record.json").write_bytes(b"{}")
        else:
            (tmp_path / "ledger.jsonl").write_bytes((tmp_path / "ledger.jsonl").read_bytes()[:-1])
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and batch.errors


def test_missing_first_counting_source_does_not_promote_next_attempt(tmp_path):
    first, second = _write_record(tmp_path), _write_record(tmp_path, attempt=2)
    _ledger(tmp_path, [first, second])
    (tmp_path / first["dir"] / "predictions.parquet").unlink()
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert len(batch.predictions) == 1
    assert batch.predictions[0]["attempt"] == 2 and not batch.predictions[0]["counting"]
    assert batch.errors[0]["reason"] == "source_missing"
    assert batch.errors[0]["record_sha256"] == first["record_sha256"]


def test_missing_ledger_and_record_are_explicit(tmp_path):
    assert read_shadow(tmp_path, observed_at=OBSERVED).errors[0]["reason"] == "source_missing"
    line = _write_record(tmp_path)
    _ledger(tmp_path, [line])
    (tmp_path / line["dir"] / "record.json").unlink()
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and batch.errors[0]["reason"] == "source_missing"


def test_no_evaluation_files_or_non_metadata_prediction_columns_are_read(tmp_path, monkeypatch):
    line = _write_record(tmp_path)
    _ledger(tmp_path, [line])
    for rel in ("forward/points.parquet", "forward/checks/2026-10-30.json", "checks/a.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")
    original_open = Path.open
    opened = []

    def safe_open(path, *args, **kwargs):
        assert not {"checks", "forward"}.intersection(path.parts)
        assert path.name != "points.parquet"
        opened.append(path)
        return original_open(path, *args, **kwargs)

    original_parquet = pl.read_parquet
    projected = []

    def safe_parquet(path, *args, **kwargs):
        assert isinstance(path, io.BytesIO)
        assert kwargs["columns"] == ["ticker", "market", "rank", "score"]
        projected.append(path)
        return original_parquet(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", safe_open)
    monkeypatch.setattr(pl, "read_parquet", safe_parquet)
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.errors and len(batch.predictions) == 1 and len(projected) == 1
    assert any(path.name == "panel.parquet" for path in opened)  # byte hashing only


@pytest.mark.parametrize("extra", ["../forward/points.parquet", "checks/a.json", "secret.json"])
def test_arbitrary_record_file_paths_rejected_before_open(tmp_path, extra, monkeypatch):
    line = _write_record(tmp_path, files_extra={extra: "a" * 64})
    _ledger(tmp_path, [line])
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name not in {"points.parquet", "a.json", "secret.json"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and "undeclared output" in batch.errors[0]["message"]


def test_symlinks_rejected_even_when_target_is_inside_source_root(tmp_path):
    line = _write_record(tmp_path)
    _ledger(tmp_path, [line])
    original = tmp_path / line["dir"] / "predictions.parquet"
    original.rename(tmp_path / "predictions-original.parquet")
    original.symlink_to(tmp_path / "predictions-original.parquet")
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and "symlink" in batch.errors[0]["message"]


def test_source_bytes_mismatch_does_not_parse_prediction(tmp_path, monkeypatch):
    line = _write_record(tmp_path)
    _ledger(tmp_path, [line])
    (tmp_path / line["dir"] / "panel.parquet").write_bytes(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("must verify original files before parsing predictions")

    monkeypatch.setattr(pl, "read_parquet", forbidden)
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and batch.errors


def test_replay_stable_identity_and_dates_separate_reused_ticker(tmp_path):
    first = _write_record(tmp_path)
    second = _write_record(tmp_path, d="2026-10-21", t="2026-10-20")
    _ledger(tmp_path, [first, second])
    initial = read_shadow(tmp_path, observed_at=OBSERVED)
    replay = read_shadow(tmp_path, observed_at="2026-11-01T12:00:00+09:00", historical_link=False)
    assert len({row["prediction_ref"] for row in initial.predictions}) == 2
    assert [r["prediction_ref"] for r in replay.predictions] == [
        r["prediction_ref"] for r in initial.predictions
    ]
    assert all(not row["historical_link"] for row in replay.predictions)


def test_invalid_metadata_does_not_create_partial_rows(tmp_path):
    rows = [
        {"ticker": "000001", "market": "KOSPI", "rank": 1, "score": 0.4},
        {"ticker": "000002", "market": "KOSPI", "rank": 2, "score": float("nan")},
    ]
    line = _write_record(tmp_path, rows=rows)
    _ledger(tmp_path, [line])
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and "finite" in batch.errors[0]["message"]


def test_orphan_record_is_rejected(tmp_path):
    line = _write_record(tmp_path)
    _write_record(tmp_path, attempt=2)
    _ledger(tmp_path, [line])
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and "orphan" in batch.errors[0]["message"]


def test_date_is_not_inferred_from_folder(tmp_path):
    line = _write_record(tmp_path, t="2026-10-20")
    _ledger(tmp_path, [line])
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and "precede" in batch.errors[0]["message"]


def test_prediction_change_after_initial_hash_never_reaches_parser(tmp_path, monkeypatch):
    from modeler.feedback import shadow

    line = _write_record(tmp_path)
    _ledger(tmp_path, [line])
    original = shadow._predictions

    def changed(root, ledger_line, record):
        path = root / ledger_line["dir"] / "predictions.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        return original(root, ledger_line, record)

    def no_parse(*args, **kwargs):
        pytest.fail("changed prediction bytes must be rejected before parsing")

    monkeypatch.setattr(shadow, "_predictions", changed)
    monkeypatch.setattr(pl, "read_parquet", no_parse)
    batch = read_shadow(tmp_path, observed_at=OBSERVED)
    assert not batch.predictions and "changed after" in batch.errors[0]["message"]
