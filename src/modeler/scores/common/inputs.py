"""입력 고정: 스냅샷을 못 박은 레이크 뷰, 파일 해시, git 커밋.

``PinnedScopedLake``는 표마다 ``snapshot_date``를 고정하고 ``prices_daily``·
``corp_actions``를 대상 심볼만 읽게 한다(2,900만 행 전체를 읽지 않으려는 것이다).
``us/prices.py``의 분할 조정 로직이 ``UsLake``를 받으므로 서브클래스로 끼운다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from modeler.us.lake import UsLake

SYMBOL_SCOPED_TABLES = ("prices_daily", "corp_actions")


@dataclass(frozen=True)
class PinnedScopedLake(UsLake):
    snapshots: dict[str, str]
    symbols: tuple[str, ...]

    def latest_snapshot(self, table: str) -> date:
        if table not in self.snapshots:
            return super().latest_snapshot(table)
        return date.fromisoformat(self.snapshots[table])

    def scan_raw(self, table: str, snapshot_date: date | None = None) -> pl.LazyFrame:
        lf = super().scan_raw(table, snapshot_date)
        if table in SYMBOL_SCOPED_TABLES:
            lf = lf.filter(pl.col("symbol").is_in(list(self.symbols)))
        return lf

    def input_files(self, tables: tuple[str, ...]) -> dict[str, list[Path]]:
        return {t: sorted(self.snapshot_dir(t, None).glob("*.parquet")) for t in tables}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame_bytes(path: Path) -> str:
    return sha256_file(path)
