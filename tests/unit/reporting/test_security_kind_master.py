"""US 종류를 증권 마스터로 정하기(유니버스 v2 서빙 W4, 계획 S7).

builder(`security_names`)는 심볼 → universe_daily_v2.security_id(D 달의 첫 세션) → security_master
(D 시점 유효 행) → 표시 종류를 정하고, 렌더러는 `security-names.v2`의 그 값을 그대로 씁니다.
`security-names.v1`(이름 규칙)은 옛 단위를 고정 바이트로 다시 그릴 수 있도록 그대로 그립니다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import md_fixtures as mf
import polars as pl
import pytest
from test_security_kind import RELEASE, STAMP, _no_log, _table_rows

from modeler.reporting import markdown as md
from modeler.reporting import security_names as sn

D = dt.date(2026, 10, 5)
DAY = D.isoformat()


def _d(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def master_row(sid: str, *, valid_from="2020-01-01", valid_to=None, **kw) -> dict:
    row = {
        "security_id": sid,
        "valid_from": _d(valid_from),
        "valid_to": _d(valid_to) if valid_to else None,
        "issuer_kind": "operating",
        "security_type": "common",
        "include": False,
        "reason_code": "unknown",
        "source_code": "none",
        "rule_version": "t",
    }
    row.update(kw)
    return row


def write_lake(
    root: Path,
    *,
    universe: list[tuple] | None,
    master: list[dict] | None,
    listing: list[tuple] | None,
    rule_version: str | None = "seg-test+ud2-1",
) -> None:
    """합성 US 레이크. universe=(date, symbol, security_id), listing=(as_of, kind, symbol, name)."""
    snap = root / "derived/snapshots"
    if listing is not None:
        d = snap / "listing_snapshots_v2/snapshot_date=2026-10-06"
        d.mkdir(parents=True)
        pl.DataFrame(
            {
                "as_of": [_d(r[0]) for r in listing],
                "kind": [r[1] for r in listing],
                "symbol": [r[2] for r in listing],
                "security_name": [r[3] for r in listing],
            },
            schema_overrides={"security_name": pl.String},
        ).write_parquet(d / "part.parquet")
    if universe is not None:
        d = snap / "universe_daily_v2/snapshot_date=2026-10-06"
        d.mkdir(parents=True)
        pl.DataFrame(
            {
                "date": [_d(r[0]) for r in universe],
                "symbol": [r[1] for r in universe],
                "security_id": [r[2] for r in universe],
            },
            schema_overrides={"security_id": pl.String},
        ).write_parquet(d / "part.parquet")
        if rule_version:
            (d / "completion.json").write_text(json.dumps({"rule_version": rule_version}))
    if master is not None:
        d = snap / "security_master/snapshot_date=2026-10-06"
        d.mkdir(parents=True)
        pl.DataFrame(
            master,
            schema_overrides={
                "valid_to": pl.Date,
                "valid_from": pl.Date,
                "include": pl.Boolean,
            },
        ).write_parquet(d / "part.parquet")


# (security_type, issuer_kind, include, reason_code) -> 기대 표시
MAPPING = [
    (dict(include=True, security_type="adr"), "ADR"),
    (dict(include=True, security_type="reit"), "리츠"),
    (dict(include=True, security_type="common"), "보통주"),
    (dict(include=True, security_type="common_equity"), "보통주"),
    # include가 이기므로 spac 발행사여도 include면 보통주
    (dict(include=True, issuer_kind="spac"), "보통주"),
    (dict(issuer_kind="spac", security_type="common"), "SPAC"),
    (dict(issuer_kind="spac", security_type="warrant"), "SPAC"),  # SPAC이 기타보다 먼저
    (dict(issuer_kind="bdc"), "펀드"),
    (dict(issuer_kind="registered_fund"), "펀드"),
    (dict(security_type="fund"), "펀드"),
    (dict(security_type="trust"), "펀드"),
    (dict(security_type="etf"), "펀드"),
    (dict(security_type="bond"), "채권형"),
    (dict(security_type="etn"), "채권형"),
    (dict(security_type="preferred"), "우선주"),
    (dict(security_type="warrant"), "기타"),
    (dict(security_type="right"), "기타"),
    (dict(security_type="unit"), "기타"),
    (dict(security_type="warrant_right_unit"), "기타"),
    (dict(security_type="test"), "기타"),
    (dict(reason_code="issuer_name_noncommon"), "기타"),
    (dict(reason_code="issuer_only_suffix"), "기타"),
    (dict(issuer_kind="unknown", security_type="unknown", reason_code="unknown"), "확인 안 됨"),
    (dict(security_type="f6k_only", reason_code="f6k_only"), "확인 안 됨"),
    (dict(), "확인 안 됨"),
]


@pytest.mark.parametrize(("fields", "kind"), MAPPING)
def test_us_kind_follows_the_s7_table(fields: dict, kind: str) -> None:
    assert sn.us_kind(master_row("X#1", **fields)) == kind
    assert sn.us_kind(None) == "확인 안 됨"


def test_kind_order_has_the_new_kinds_after_the_old_ones() -> None:
    assert md.KIND_ORDER == ("채권형", "우선주", "SPAC", "리츠", "펀드", "ADR", "기타")


def _lake_with_all_kinds(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "us"
    symbols = [f"S{i:02d}" for i in range(len(MAPPING))]
    master = [master_row(f"{s}#1", **fields) for s, (fields, _k) in zip(symbols, MAPPING)]
    universe = [("2026-10-01", s, f"{s}#1") for s in symbols]
    listing = [("2026-10-05", "nasdaqlisted", s, f"Name {s}") for s in symbols]
    write_lake(root, universe=universe, master=master, listing=listing)
    return root, symbols


def _envelope(symbols: list[str]) -> dict:
    return {
        "report_date": DAY,
        "markets": [{"market": "US", "rankings": [{"symbol": s} for s in symbols]}],
    }


def test_from_roots_maps_every_row_of_the_table(tmp_path: Path) -> None:
    root, symbols = _lake_with_all_kinds(tmp_path)
    built = sn.from_roots(_envelope(symbols), us_root=root, kr_root=None)
    assert built.notes == {}
    data = json.loads(built.raw)
    assert data["schema"] == "security-names.v2"
    block = data["markets"]["US"]
    assert block["kind_by"] == "security_master"
    assert {s: block["symbols"][s]["kind"] for s in symbols} == {
        s: k for s, (_f, k) in zip(symbols, MAPPING)
    }
    assert block["source"].startswith("US 레이크 listing_snapshots_v2.security_name")
    assert "snapshot_date=2026-10-06" in block["kind_source"]
    assert "기준일 2026-10-01" in block["kind_source"]
    assert "rule_version seg-test+ud2-1" in block["kind_source"]
    assert built.counts == {"US": (len(symbols), len(symbols))}
    assert built.stats["US"]["kinds"]["ADR"] == 1
    assert sum(built.stats["US"]["kinds"].values()) == len(symbols)
    loaded = md.load_security_names(built.raw)
    assert loaded["US"]["master_kind"] is True
    assert md.row_kind("US", {"symbol": symbols[0]}, loaded) == "ADR"


def test_output_is_deterministic_and_has_no_paths(tmp_path: Path) -> None:
    root, symbols = _lake_with_all_kinds(tmp_path)
    one = sn.from_roots(_envelope(symbols), us_root=root, kr_root=None).raw
    two = sn.from_roots(_envelope(list(reversed(symbols))), us_root=root, kr_root=None).raw
    assert one == two
    assert str(tmp_path) not in one.decode()


def test_identity_date_is_the_first_session_of_the_month(tmp_path: Path) -> None:
    root = tmp_path / "us"
    # 9월 말과 10월 첫 세션에서 같은 심볼이 다른 종목을 가리킵니다(심볼 재사용).
    universe = [
        ("2026-09-30", "AAA", "AAA#1"),
        ("2026-10-01", "AAA", "AAA#2"),
        ("2026-10-02", "AAA", "AAA#3"),
        ("2026-10-05", "AAA", "AAA#3"),
    ]
    master = [
        master_row("AAA#1", security_type="preferred"),
        master_row("AAA#2", include=True, security_type="common"),
        master_row("AAA#3", security_type="warrant"),
    ]
    write_lake(root, universe=universe, master=master, listing=[("2026-10-05", "x", "AAA", "A")])
    block = sn.from_roots(_envelope(["AAA"]), us_root=root, kr_root=None)
    assert json.loads(block.raw)["markets"]["US"]["symbols"]["AAA"]["kind"] == "보통주"
    assert block.stats["US"]["identity_date"] == "2026-10-01"
    assert block.stats["US"]["month_first_session"] is True
    assert block.notes == {}
    # D가 첫 세션 전(10-01 이전, 달 안에 날짜 없음)이면 가장 최근 날짜로 대신하고 적습니다.
    early = sn.from_roots(
        {**_envelope(["AAA"]), "report_date": "2026-09-30"}, us_root=root, kr_root=None
    )
    assert early.stats["US"]["identity_date"] == "2026-09-30"  # 9월 첫 세션은 없어 9-30이 첫 날짜
    # 달 안에 D 이하 날짜가 없는 경우: 10-01이 D보다 뒤
    root2 = tmp_path / "us2"
    write_lake(
        root2,
        universe=[("2026-09-30", "AAA", "AAA#1"), ("2026-10-02", "AAA", "AAA#3")],
        master=master,
        listing=[("2026-09-30", "x", "AAA", "A")],
    )
    got = sn.from_roots(
        {**_envelope(["AAA"]), "report_date": "2026-10-01"}, us_root=root2, kr_root=None
    )
    assert got.stats["US"]["identity_date"] == "2026-09-30"
    assert got.stats["US"]["month_first_session"] is False
    assert "가장 최근 2026-09-30" in got.notes["US"]
    assert json.loads(got.raw)["markets"]["US"]["symbols"]["AAA"]["kind"] == "우선주"


def test_master_valid_to_is_exclusive_and_latest_valid_from_wins(tmp_path: Path) -> None:
    root = tmp_path / "us"
    universe = [("2026-10-01", "AAA", "AAA#1"), ("2026-10-01", "BBB", "BBB#1")]
    master = [
        # AAA: 10-05에 끝나는 행(valid_to = D)은 D에 유효하지 않고, 바로 이어지는 행이 유효합니다
        master_row(
            "AAA#1", valid_from="2020-01-01", valid_to="2026-10-05", security_type="warrant"
        ),
        master_row("AAA#1", valid_from="2026-10-05", security_type="preferred"),
        # BBB: 겹치는 두 행이면 valid_from이 가장 늦은 행
        master_row("BBB#1", valid_from="2020-01-01", security_type="bond"),
        master_row("BBB#1", valid_from="2026-09-01", valid_to="2026-10-06", security_type="etf"),
        # D 뒤에 시작하는 행은 쓰지 않습니다
        master_row("BBB#1", valid_from="2026-10-06", security_type="unit"),
    ]
    write_lake(root, universe=universe, master=master, listing=[("2026-10-05", "x", "AAA", "A")])
    out = json.loads(sn.from_roots(_envelope(["AAA", "BBB"]), us_root=root, kr_root=None).raw)
    symbols = out["markets"]["US"]["symbols"]
    assert symbols["AAA"]["kind"] == "우선주"
    assert symbols["BBB"]["kind"] == "펀드"
    # valid_to가 D보다 하루 뒤면 아직 유효합니다
    earlier = sn.from_roots(
        {**_envelope(["AAA"]), "report_date": "2026-10-04"}, us_root=root, kr_root=None
    )
    assert json.loads(earlier.raw)["markets"]["US"]["symbols"]["AAA"]["kind"] == "기타"


def test_two_security_ids_for_one_symbol_on_the_same_date_are_not_guessed(tmp_path: Path) -> None:
    root = tmp_path / "us"
    universe = [
        ("2026-10-01", "AAA", "AAA#1"),
        ("2026-10-01", "AAA", "AAA#2"),
        ("2026-10-01", "BBB", "BBB#1"),
        ("2026-10-01", "BBB", "BBB#1"),  # 같은 id가 겹쳐 있으면 문제 없음
    ]
    master = [
        master_row("AAA#1", include=True),
        master_row("AAA#2", include=True),
        master_row("BBB#1", include=True),
    ]
    write_lake(root, universe=universe, master=master, listing=[("2026-10-05", "x", "AAA", "A")])
    built = sn.from_roots(_envelope(["AAA", "BBB"]), us_root=root, kr_root=None)
    symbols = json.loads(built.raw)["markets"]["US"]["symbols"]
    assert symbols["AAA"]["kind"] == "확인 안 됨" and symbols["BBB"]["kind"] == "보통주"
    assert built.stats["US"]["ambiguous_v2"] == 1
    assert "서로 다른 security_id" in built.notes["US"]


def test_missing_rows_become_unknown_and_are_counted(tmp_path: Path) -> None:
    root = tmp_path / "us"
    universe = [("2026-10-01", "HAS", "HAS#1"), ("2026-10-01", "NOMASTER", "NOMASTER#1")]
    master = [master_row("HAS#1", include=True), master_row("HAS#1", valid_from="2026-10-06")]
    listing = [
        ("2026-10-05", "nasdaqlisted", "HAS", "Has Inc"),
        ("2026-10-05", "nasdaqlisted", "NOV2", "No v2 row Inc"),  # v2 행은 없고 이름만 있음
        ("2026-10-05", "nasdaqlisted", "NONAME", None),
    ]
    universe.append(("2026-10-01", "NONAME", "NONAME#1"))
    master.append(master_row("NONAME#1", security_type="bond"))
    write_lake(root, universe=universe, master=master, listing=listing)
    env = _envelope(["HAS", "NOMASTER", "NOV2", "NONAME", "GHOST"])
    built = sn.from_roots(env, us_root=root, kr_root=None)
    block = json.loads(built.raw)["markets"]["US"]
    kinds = {s: v["kind"] for s, v in block["symbols"].items()}
    assert kinds == {
        "HAS": "보통주",
        "NOMASTER": "확인 안 됨",
        "NOV2": "확인 안 됨",
        "NONAME": "채권형",
    }
    assert "GHOST" not in kinds  # 이름도 식별자도 없으면 싣지 않습니다
    stats = built.stats["US"]
    assert stats == block["stats"]
    assert (stats["no_v2_row"], stats["no_master_row"], stats["no_name"]) == (2, 1, 3)
    assert stats["kinds"] == {"보통주": 1, "채권형": 1, "확인 안 됨": 3}
    assert built.counts["US"] == (4, 5)


def test_names_come_from_listing_snapshots_v2_with_the_same_as_of_rule(tmp_path: Path) -> None:
    root = tmp_path / "us"
    listing = [
        ("2026-09-01", "nasdaqlisted", "AAA", "Alpha Old"),
        ("2026-10-05", "nasdaqlisted", "AAA", "Alpha New"),
        ("2026-10-06", "nasdaqlisted", "AAA", "Alpha From The Future"),
        ("2026-09-01", "nasdaqlisted", "BBB", "Beta Stale"),  # 마지막 목록에 없음
        ("2026-10-05", "nasdaqlisted", "CCC", "Gamma"),
    ]
    write_lake(
        root,
        universe=[("2026-10-01", "AAA", "AAA#1")],
        master=[master_row("AAA#1", include=True)],
        listing=listing,
    )
    # v1 listing이 있어도 v2를 씁니다
    v1 = root / "derived/snapshots/listing_snapshots/snapshot_date=2026-09-18"
    v1.mkdir(parents=True)
    pl.DataFrame(
        {
            "as_of": [_d("2026-09-18")],
            "kind": ["nasdaqlisted"],
            "symbol": ["AAA"],
            "security_name": ["V1 NAME"],
        }
    ).write_parquet(v1 / "part.parquet")
    built = sn.from_roots(_envelope(["AAA", "BBB"]), us_root=root, kr_root=None)
    symbols = json.loads(built.raw)["markets"]["US"]["symbols"]
    assert symbols["AAA"]["name"] == "Alpha New" and symbols["AAA"]["current"] is True
    assert symbols["BBB"]["name"] == "Beta Stale" and symbols["BBB"]["current"] is False
    # 이름이 낡았어도 종류는 증권 마스터가 정합니다(식별자가 없는 BBB는 확인 안 됨)
    assert symbols["AAA"]["kind"] == "보통주" and symbols["BBB"]["kind"] == "확인 안 됨"


def test_without_v2_tables_the_builder_falls_back_or_notes(tmp_path: Path) -> None:
    # v2 listing은 있고 universe·master가 없으면 이름만 싣고 종류는 확인 안 됨, 사유를 남깁니다
    root = tmp_path / "us"
    write_lake(root, universe=None, master=None, listing=[("2026-10-05", "x", "AAA", "A")])
    built = sn.from_roots(_envelope(["AAA"]), us_root=root, kr_root=None)
    assert "증권 마스터 입력" in built.notes["US"]
    assert json.loads(built.raw)["markets"]["US"]["symbols"]["AAA"]["kind"] == "확인 안 됨"
    # v2 listing이 없으면 v1 listing 방식(이름 규칙용 입력)입니다
    old = tmp_path / "old"
    v1 = old / "derived/snapshots/listing_snapshots/snapshot_date=2026-09-18"
    v1.mkdir(parents=True)
    pl.DataFrame(
        {
            "as_of": [_d("2026-10-01")],
            "kind": ["nasdaqlisted"],
            "symbol": ["AAA"],
            "security_name": ["Alpha Inc. Common Stock"],
        }
    ).write_parquet(v1 / "part.parquet")
    built = sn.from_roots(_envelope(["AAA"]), us_root=old, kr_root=None)
    loaded = md.load_security_names(built.raw)
    assert "master_kind" not in loaded["US"] and built.stats == {}
    assert md.row_kind("US", {"symbol": "AAA"}, loaded) == "보통주"


# --- 렌더러 ------------------------------------------------------------------------------------
def _inputs(tmp_path: Path) -> tuple[dict, bytes]:
    paths = mf.write_all(tmp_path / "fx")
    env_path, ms_path = paths["f1_normal"]
    return json.loads(env_path.read_text(encoding="utf-8")), ms_path.read_bytes()


def _render(env: dict, ms: bytes, names: bytes | None) -> tuple[dict, dict]:
    ctx = md.build_context(
        Path("no-repo"),
        json.dumps(env).encode(),
        ms,
        RELEASE,
        STAMP,
        100,
        _no_log,
        security_names=names,
    )
    return ctx, md.render_unit(ctx)


def _v2_names(env: dict, kinds: list[str | None]) -> bytes:
    lgb = next(m for m in env["markets"] if m["model_id"] == mf.LGB_ID)
    symbols = [r["symbol"] for r in lgb["rankings"]]
    syms = {}
    for i, s in enumerate(symbols):
        syms[s] = {"name": f"Sample {s}", "current": False, "kind": "보통주"}  # 이름 낡음과 무관
    for s, kind in zip(symbols, kinds):
        syms[s]["kind"] = kind
    block = {
        "source": "US 시험 목록 v2",
        "basis": "시험 기준",
        "kind_by": "security_master",
        "kind_source": "US 시험 마스터 (snapshot_date=2026-10-06)",
        "symbols": syms,
    }
    return json.dumps({"schema": "security-names.v2", "markets": {"US": block}}).encode()


def test_v2_file_renders_adr_and_other_from_the_given_kind(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    names = _v2_names(env, ["ADR", "기타", "펀드", "확인 안 됨", "리츠", "엉뚱한 값"])
    ctx, files = _render(env, ms, names)
    text = files["us-stocks.md"]
    rows = _table_rows(text, "| 순위 | 코드 | 종류 |")
    assert [r[2] for r in rows[:6]] == ["ADR", "기타", "펀드", "확인 안 됨", "리츠", "확인 안 됨"]
    assert rows[6][2] == "보통주"  # 이름이 낡았어도(current false) 마스터 종류를 씁니다
    assert "보통주가 아닌 종목 4개 포함 (리츠 1 · 펀드 1 · ADR 1 · 기타 1)." in text
    assert md.KIND_NOTE_MASTER in text and md.KIND_NOTE not in text
    status = files["data-status.md"]
    assert md.US_MASTER_KIND_INTRO in status
    assert "US 시험 마스터 (snapshot_date=2026-10-06)" in status
    assert md.US_MASTER_KIND_RULES in status
    summary = files["README.md"]
    assert md.KIND_NOTE_MIXED in summary  # KR 열도 보이므로 섞인 문구
    assert md.validate_unit_files(ctx["unit"], files) == []
    # 마스터 종류는 순위·점수를 바꾸지 않습니다
    _, plain = _render(env, ms, None)
    before = _table_rows(plain["us-stocks.md"], "| 순위 | 코드 |")
    assert [r[:2] + r[3:] for r in rows if len(r) > 3] == [r[:2] + r[2:] for r in before][
        : len(rows)
    ]


def test_v2_file_without_kind_by_reads_like_v1(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    lgb = next(m for m in env["markets"] if m["model_id"] == mf.LGB_ID)
    symbols = [r["symbol"] for r in lgb["rankings"]]
    syms = {
        s: {"name": "Ford Motor Company 6.500% Notes due 2062", "current": True} for s in symbols
    }
    block = {"source": "s", "basis": "b", "symbols": syms}
    raw = json.dumps({"schema": "security-names.v2", "markets": {"US": block}}).encode()
    _, files = _render(env, ms, raw)
    assert "채권형" in files["us-stocks.md"] and md.KIND_NOTE in files["us-stocks.md"]


def test_kr_is_unchanged_next_to_a_v2_us_block(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    kr = next(m for m in env["markets"] if m["market"] == "KR")
    kr["rankings"][0].update(symbol="473050", name="유안타제15호스팩")
    kr["rankings"][1].update(symbol="005935", name="삼성전자우")
    v2 = json.loads(_v2_names(env, ["ADR"]))
    _, files = _render(env, ms, json.dumps(v2).encode())
    _, ref = _render(env, ms, None)
    assert (
        files["kr-stocks.md"] == ref["kr-stocks.md"]
    )  # KR 표·문구가 이름 입력 없이 그린 것과 같음
    rows = _table_rows(files["kr-stocks.md"], "| 순위 | 코드 | 이름 | 종류 |")
    assert [r[3] for r in rows[:3]] == ["SPAC", "우선주", "보통주"]
    assert md.KIND_NOTE_MIXED in files["README.md"]


# 옛(v1) 입력은 main 렌더러와 바이트가 같아야 합니다. 아래 해시는 이 변경 전 코드(093a6c1)로
# 같은 입력을 그린 파일의 sha256입니다.
V1_GOLDEN = {
    "README.md": "1c56bbd8cf271edd89d392504ed686e5f8034013ac6e16619ad55320b805dbd3",
    "data-status.md": "4164e71cf8c6d5b23db74cc0e5788e7aa06842f6cd631c80ab62e2b179757c67",
    "kr-stocks.md": "ff8b5cdd01010c0c7b41a50e1271af7c32f529594ac205870d6726e93c63d07b",
    "market-sector.md": "d70e8405f770450530af0c892ee0457604718d1e3bb84c0af70df59815f06e1a",
    "us-stocks.md": "5427be61eb27a0603ab82400d188329fced273fe8531c30fe7ba743907485a37",
}


def test_v1_file_renders_byte_identically_to_the_previous_renderer(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    lgb = next(m for m in env["markets"] if m["model_id"] == mf.LGB_ID)
    symbols = [r["symbol"] for r in lgb["rankings"]]
    us = {s: {"name": f"Sample {s} Inc. Common Stock", "current": True} for s in symbols}
    us[symbols[0]] = {"name": "Ford Motor Company 6.500% Notes due 2062", "current": True}
    us[symbols[1]] = {"name": "Blackrock Enhanced Equity Dividend Trust", "current": True}
    us[symbols[2]] = {"name": "Old ETN Name", "current": False}
    us[symbols[3]] = {"name": "Irenic Acquisition Corp. - Class A Ordinary Shares"}
    block = {"source": "US 시험 목록", "basis": "시험 기준", "symbols": us}
    raw = json.dumps({"schema": "security-names.v1", "markets": {"US": block}}).encode()
    _, files = _render(env, ms, raw)
    got = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(files.items())}
    assert got == V1_GOLDEN
