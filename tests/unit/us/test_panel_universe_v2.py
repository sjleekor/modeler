"""``build_panel(universe_version="v2")`` — 멤버십만 v2 (유니버스 v2 설계 §3·§4, T11).

합성 parquet을 ``tmp_path``에 쓴다. 실제 레이크는 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.lake import UsLake
from modeler.us.panel import build_panel, universe_manifest

D = date(2019, 1, 2)


def _snap(root: Path, table: str, frame: pl.DataFrame, snapshot_date: str = "2026-09-18") -> Path:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")
    return directory


_V1_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.String,
    "cik": pl.Int64,
    "sic": pl.String,
    "mcap_rank": pl.Int32,
    "adv_20d": pl.Float64,
    "exchange": pl.String,
    "in_universe": pl.Boolean,
}


def _v1_row(symbol: str, in_universe: bool, **kw: object) -> dict:
    row = {
        "date": D,
        "symbol": symbol,
        "cik": 1,
        "sic": "3571",
        "mcap_rank": 1,
        "adv_20d": 1_000_000.0,
        "exchange": "XNYS",
        "in_universe": in_universe,
    }
    row.update(kw)
    return row


def _v2_frame(rows: list[tuple[str, bool]], *, view: str | None = None) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "date": [D] * len(rows),
            "symbol": [r[0] for r in rows],
            "security_id": [f"{r[0]}#1" for r in rows],
            "in_universe": [r[1] for r in rows],
        }
    )
    if view is not None:
        frame = frame.with_columns(pl.lit(view).alias("view"))
    return frame


def _write_world(root: Path, v1_rows: list[dict], v2: pl.DataFrame, symbols: list[str]) -> None:
    _snap(root, "trading_calendar", pl.DataFrame({"date": [D], "exchange": ["XNYS"]}), "2026-09-19")
    _snap(
        root,
        "corp_actions",
        pl.DataFrame(
            schema={
                "symbol": pl.String,
                "ex_date": pl.Date,
                "kind": pl.String,
                "to_factor": pl.Float64,
                "for_factor": pl.Float64,
            }
        ),
    )
    _snap(root, "universe_daily", pl.DataFrame(v1_rows, schema=_V1_SCHEMA))
    _snap(root, "universe_daily_v2", v2)
    prices = [
        {
            "date": D,
            "symbol": s,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1000.0,
        }
        for s in symbols
    ]
    _snap(
        root,
        "prices_daily",
        pl.DataFrame(
            prices,
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        ),
    )


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def test_v2_membership_comes_from_universe_daily_v2_and_other_columns_from_v1(
    tmp_path: Path, lake: UsLake
) -> None:
    # AAA: 둘 다 멤버. BBB: v1에만. CCC(ADR 같은 v2 전용): v1 행은 있으나 멤버가 아니다.
    _write_world(
        tmp_path,
        [
            _v1_row("AAA", True, cik=11),
            _v1_row("BBB", True, cik=22),
            _v1_row("CCC", False, cik=33, mcap_rank=7, adv_20d=2_500_000.0),
        ],
        _v2_frame([("AAA", True), ("BBB", False), ("CCC", True)]),
        ["AAA", "BBB", "CCC"],
    )
    v1 = build_panel(lake, start=D, end=D)
    v2 = build_panel(lake, start=D, end=D, universe_version="v2")

    assert v1["symbol"].to_list() == ["AAA", "BBB"]
    assert v2["symbol"].to_list() == ["AAA", "CCC"]
    # 스키마(열 이름·순서·타입)가 v1과 같다 — security_id 열이 없다.
    assert v2.schema == v1.schema
    assert "security_id" not in v2.columns
    # v2 전용 멤버의 나머지 열은 v1 universe_daily의 같은 (date, symbol) 행에서 온다.
    ccc = v2.filter(pl.col("symbol") == "CCC").row(0, named=True)
    assert ccc["cik"] == 33 and ccc["mcap_rank"] == 7 and ccc["adv_20d"] == 2_500_000.0
    assert ccc["close"] == 10.0 and ccc["price_ge_5"] is True
    # 공통 멤버 행은 두 버전에서 같다.
    assert (
        v2.filter(pl.col("symbol") == "AAA").to_dicts()
        == v1.filter(pl.col("symbol") == "AAA").to_dicts()
    )


def test_v2_member_without_v1_row_gets_null_columns(tmp_path: Path, lake: UsLake) -> None:
    _write_world(
        tmp_path,
        [_v1_row("AAA", True)],
        _v2_frame([("AAA", True), ("DDD", True)]),
        ["AAA", "DDD"],
    )
    v2 = build_panel(lake, start=D, end=D, universe_version="v2")
    ddd = v2.filter(pl.col("symbol") == "DDD").row(0, named=True)
    assert ddd["cik"] is None and ddd["mcap_rank"] is None and ddd["adv_20d"] is None
    assert ddd["sic2"] is None
    assert ddd["close"] == 10.0  # 가격은 prices_daily에서 온다


def test_default_universe_version_is_v1_and_unchanged(tmp_path: Path, lake: UsLake) -> None:
    _write_world(
        tmp_path,
        [_v1_row("AAA", True), _v1_row("BBB", False)],
        _v2_frame([("AAA", False), ("BBB", True)]),
        ["AAA", "BBB"],
    )
    default = build_panel(lake, start=D, end=D)
    explicit = build_panel(lake, start=D, end=D, universe_version="v1")
    assert default.equals(explicit)
    assert default["symbol"].to_list() == ["AAA"]


def test_unknown_universe_version_is_rejected(tmp_path: Path, lake: UsLake) -> None:
    with pytest.raises(ValueError, match="universe_version"):
        build_panel(lake, universe_version="v3")


def test_v2_member_key_must_be_unique(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "date": [D, D],
            "symbol": ["AAA", "AAA"],
            "security_id": ["AAA#1", "AAA#2"],
            "in_universe": [True, True],
        }
    )
    _write_world(tmp_path, [_v1_row("AAA", True)], frame, ["AAA"])
    with pytest.raises(ValueError, match="겹칩니다"):
        build_panel(lake, start=D, end=D, universe_version="v2")


# --- T11: 사후 보기를 읽지 않는다 -------------------------------------------------------


@dataclass(frozen=True)
class _RecordingLake(UsLake):
    """``scan``으로 읽은 표 이름을 기록하는 리더."""

    log: list[str] = field(default_factory=list)

    def scan(self, table: str, snapshot_date: date | None = None) -> pl.LazyFrame:
        self.log.append(table)
        return super().scan(table, snapshot_date)


def test_t11_v2_panel_never_reads_post_view_or_segment_tables(tmp_path: Path) -> None:
    """v2 패널 로더는 유니버스·가격 표만 읽는다 — 식별·마스터 표는 안 연다.

    ``universe_daily_v2``에 ``view`` 열이 있어도(지금은 없다) ``post`` 행은 멤버십에 못 들어온다.
    """
    frame = pl.concat(
        [
            _v2_frame([("AAA", True)], view="pit"),
            _v2_frame([("POSTONLY", True)], view="post"),  # 사후 보기에만 멤버 — 들어오면 안 된다
        ]
    )
    _write_world(
        tmp_path,
        [_v1_row("AAA", True), _v1_row("POSTONLY", True)],
        frame,
        ["AAA", "POSTONLY"],
    )
    # 읽으면 안 되는 표가 있어도 열지 않는다는 것을 보이려고 일부러 만들어 둔다.
    _snap(
        tmp_path,
        "security_segments",
        pl.DataFrame({"symbol": ["AAA"], "view": ["post"]}),
    )
    rec = _RecordingLake(root=DataRoot(base=tmp_path))

    panel = build_panel(rec, start=D, end=D, universe_version="v2")

    assert panel["symbol"].to_list() == ["AAA"]
    assert "security_segments" not in rec.log
    assert "security_master" not in rec.log
    assert set(rec.log) <= {
        "universe_daily_v2",
        "universe_daily",
        "trading_calendar",
        "prices_daily",
        "corp_actions",
    }


# --- manifest ----------------------------------------------------------------------------


def _completion(directory: Path, **overrides: object) -> None:
    sha = hashlib.sha256((directory / "part.parquet").read_bytes()).hexdigest()
    body = {
        "table": "universe_daily_v2",
        "snapshot_date": "2026-09-18",
        "rule_version": "seg-r3c.1+master-r3c.1+ud2-1",
        "rule_versions": {"segments": "seg-r3c.1"},
        "snapshot_sha256": sha,
        "input_snapshots": {"prices_daily": "2026-09-18"},
        "input_snapshot_sha256": {"prices_daily": "abc"},
        "dolt_stocks": {"commit": "x", "note": None},
        "month_stats": [{"month": "2019-01-01"}],  # manifest로 옮기지 않는다
    }
    body.update(overrides)
    (directory / "completion.json").write_text(json.dumps(body))


def test_universe_manifest_records_v2_completion_and_checks_sha(
    tmp_path: Path, lake: UsLake
) -> None:
    directory = _snap(tmp_path, "universe_daily_v2", _v2_frame([("AAA", True)]))
    _snap(tmp_path, "universe_daily", pl.DataFrame([_v1_row("AAA", True)], schema=_V1_SCHEMA))
    _snap(
        tmp_path,
        "security_segments",
        pl.DataFrame({"symbol": ["AAA"], "view": ["pit"]}),
        "2026-09-17",
    )
    _completion(directory)

    meta = universe_manifest(lake, "v2", security_boundaries=True)

    assert meta["universe_version"] == "v2" and meta["security_boundaries"] is True
    assert meta["universe_tables"] == {
        "universe_daily": "2026-09-18",
        "universe_daily_v2": "2026-09-18",
    }
    assert meta["security_segments_snapshot"] == "2026-09-17"
    completion = meta["universe_v2_completion"]
    assert completion["rule_version"] == "seg-r3c.1+master-r3c.1+ud2-1"
    assert completion["input_snapshot_sha256"] == {"prices_daily": "abc"}
    assert "month_stats" not in completion

    _completion(directory, snapshot_sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        universe_manifest(lake, "v2")


def test_universe_manifest_v1_has_no_v2_fields(tmp_path: Path, lake: UsLake) -> None:
    _snap(tmp_path, "universe_daily", pl.DataFrame([_v1_row("AAA", True)], schema=_V1_SCHEMA))
    meta = universe_manifest(lake, "v1")
    assert meta == {
        "universe_version": "v1",
        "universe_tables": {"universe_daily": "2026-09-18"},
        "security_boundaries": False,
    }


def test_universe_manifest_v2_requires_completion(tmp_path: Path, lake: UsLake) -> None:
    _snap(tmp_path, "universe_daily", pl.DataFrame([_v1_row("AAA", True)], schema=_V1_SCHEMA))
    _snap(tmp_path, "universe_daily_v2", _v2_frame([("AAA", True)]))
    with pytest.raises(FileNotFoundError, match="completion"):
        universe_manifest(lake, "v2")
    assert universe_manifest(lake, "v2", require_completion=False)["universe_version"] == "v2"
