"""Source provenance must identify bytes, not only the snapshot date."""

from datetime import date

import pytest

from collector.lake import DataRoot
from modeler.serving.us_daily import fingerprint_source_snapshots, main


def test_source_fingerprint_covers_every_parquet_and_detects_rewrite(tmp_path):
    root = DataRoot(tmp_path)
    first = root.derived / "snapshots/prices_daily/snapshot_date=2026-09-27/part.parquet"
    second = root.derived / "snapshots/prices_daily/snapshot_date=2026-09-27/extra.parquet"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first version")
    second.write_bytes(b"second file")
    revisions = {"prices_daily": "2026-09-27"}
    before = fingerprint_source_snapshots(root, revisions)
    assert [row["path"] for row in before["prices_daily"]] == [
        "derived/snapshots/prices_daily/snapshot_date=2026-09-27/extra.parquet",
        "derived/snapshots/prices_daily/snapshot_date=2026-09-27/part.parquet",
    ]
    first.write_bytes(b"changed version")
    assert fingerprint_source_snapshots(root, revisions) != before
    second.unlink()
    assert fingerprint_source_snapshots(root, revisions) != before


def test_prepare_cli_cannot_claim_parity_passed_or_evidence_without_diagnostic():
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--as-of", date(2026, 9, 25).isoformat(),
              "--raw-feature-parity-evidence", "some.json"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--as-of", date(2026, 9, 25).isoformat(),
              "--raw-feature-parity-status", "passed"])
    assert error.value.code == 2
