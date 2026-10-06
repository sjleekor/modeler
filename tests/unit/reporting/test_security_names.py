"""security_names: 레이크에서 security-names.v1 만들기(R6 D1)와 publisher·coordinator 연결.

R0 도구 `build_names.py`(pandas)를 시험 안에 오라클로 두고, 같은 입력에서 같은 블록이 나오는지
임의 입력으로 겨룹니다. 실제 레이크와 R0 산출물을 겨룬 결과는 작업 기록에 있고, 여기 시험은 합성
parquet만 씁니다.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from reports_world import RELEASE, World, no_log, pr

from modeler.reporting import markdown as md
from modeler.reporting import security_names as sn
from modeler.serving import daily_coordinator as dc

DAY = dt.date(2026, 10, 7)
D1 = "2026-10-07"


# --- 합성 레이크 ---------------------------------------------------------------------------
def write_listing(us_root: Path, rows: list[tuple], snapshot: str = "2026-09-18") -> Path:
    """rows = (as_of, kind, symbol, security_name)를 레이크 listing_snapshots 자리에 씁니다."""
    directory = us_root / "derived/snapshots/listing_snapshots" / f"snapshot_date={snapshot}"
    directory.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "as_of": [dt.date.fromisoformat(r[0]) for r in rows],
            "kind": [r[1] for r in rows],
            "symbol": [r[2] for r in rows],
            "security_name": [r[3] for r in rows],
            "snapshot": ["x"] * len(rows),
        },
        schema_overrides={"security_name": pl.String},
    )
    frame.write_parquet(directory / "part.parquet")
    return directory


def write_stock_master(
    kr_root: Path, rows: list[tuple], snapshot: str = "2026-10-07", *, marker: bool = True
) -> Path:
    source = kr_root / "raw/raw_postgres" / f"snapshot_date={snapshot}" / "source=sj2_remote"
    table = source / "stock_master/schema_version=1"
    table.mkdir(parents=True)
    pl.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "market": ["KOSPI"] * len(rows),
        }
    ).write_parquet(table / "part-000000.parquet")
    if marker:
        (source / "_manifests").mkdir()
        (source / "_manifests/_SUCCESS.json").write_text("{}")
    return source / "stock_master"


LISTING = [
    ("2026-08-01", "nasdaqlisted", "AAA", "Alpha Inc. Common Stock"),
    ("2026-09-01", "nasdaqlisted", "AAA", "Alpha Inc. Common Stock New"),
    ("2026-08-01", "nasdaqlisted", "BBB", "Old Barclays ETN"),  # 마지막 목록에 없음
    ("2026-09-01", "nasdaqlisted", "CCC", "Gamma N"),
    ("2026-09-01", "otherlisted", "CCC", "Gamma O"),  # 같은 as_of → kind 사전순으로 뒤
    ("2026-09-01", "nasdaqlisted", "EEE", None),
    ("2026-09-01", "otherlisted", "FFF", "Not requested"),
    ("2026-10-10", "nasdaqlisted", "ZZZ", "Future listing"),  # D 뒤 → 쓰지 않음
    ("2026-10-10", "nasdaqlisted", "AAA", "Alpha From The Future"),
]


def envelope(day: str, us: list[str], kr: list[str], *, models: int = 2) -> dict:
    markets = [{"market": "KR", "rankings": [{"symbol": s} for s in kr]}]
    for _ in range(models):
        markets.append({"market": "US", "rankings": [{"symbol": s} for s in us]})
    markets.append({"market": "XX", "rankings": [{"symbol": "IGNORED"}]})
    return {"report_date": day, "markets": markets}


# --- 블록 만들기 ------------------------------------------------------------------------------
def test_us_names_follow_the_r0_rules(tmp_path: Path) -> None:
    listing = write_listing(tmp_path, LISTING)
    block = sn.us_names(listing, {"AAA", "BBB", "CCC", "EEE", "MISSING"}, dt.date(2026, 10, 1))
    assert block["symbols"] == {
        "AAA": {
            "name": "Alpha Inc. Common Stock New",
            "as_of": "2026-09-01",
            "kind": "nasdaqlisted",
            "current": True,
        },
        "BBB": {
            "name": "Old Barclays ETN",
            "as_of": "2026-08-01",
            "kind": "nasdaqlisted",
            "current": False,  # 그 갈래의 마지막 목록(09-01)에 없었습니다
        },
        "CCC": {"name": "Gamma O", "as_of": "2026-09-01", "kind": "otherlisted", "current": True},
        "EEE": {"name": None, "as_of": "2026-09-01", "kind": "nasdaqlisted", "current": True},
    }
    assert block["source"] == "US 레이크 listing_snapshots.security_name (snapshot_date=2026-09-18)"
    assert block["basis"] == (
        "as_of <= 2026-10-01 중 가장 최근 이름. 갈래별 마지막 목록 "
        "nasdaqlisted 2026-09-01·otherlisted 2026-09-01. "
        "마지막 목록에 없던 심볼은 이름이 낡았을 수 있어 판정하지 않음"
    )
    # D가 지나면 미래 행이 들어옵니다(as_of <= D)
    later = sn.us_names(listing, {"AAA", "ZZZ"}, dt.date(2026, 10, 10))
    assert later["symbols"]["AAA"]["name"] == "Alpha From The Future"
    assert later["symbols"]["AAA"]["current"] is True and "ZZZ" in later["symbols"]


def _r0_us_names(listing: Path, symbols: set, day: dt.date) -> dict:
    """R0 `build_names.py`의 us_names를 그대로 옮긴 오라클(pandas)입니다."""
    df = pq.read_table(listing, columns=["as_of", "kind", "symbol", "security_name"]).to_pandas()
    df = df[df["as_of"] <= day]
    last_by_kind = df.groupby("kind")["as_of"].max().to_dict()
    df = df[df["symbol"].isin(symbols)].sort_values(["symbol", "as_of", "kind"])
    latest = df.groupby("symbol").tail(1)
    out = {}
    for row in latest.itertuples(index=False):
        name = row.security_name if isinstance(row.security_name, str) else None
        out[row.symbol] = {
            "name": name,
            "as_of": row.as_of.isoformat(),
            "kind": row.kind,
            "current": bool(row.as_of == last_by_kind[row.kind]),
        }
    return out


def test_us_names_match_the_r0_oracle_on_random_listings(tmp_path: Path) -> None:
    rng = random.Random(7)
    symbols = [f"S{i:03d}" for i in range(60)]
    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=30 * i) for i in range(12)]
    rows = []
    for _ in range(900):
        rows.append(
            (
                rng.choice(dates).isoformat(),
                rng.choice(["nasdaqlisted", "otherlisted"]),
                rng.choice(symbols),
                rng.choice([None, "Name A Common Stock", "Name B Fund", "Name C Notes due 2030"]),
            )
        )
    listing = write_listing(tmp_path, rows)
    asked = set(rng.sample(symbols, 40)) | {"NOPE"}
    for day in (dt.date(2026, 5, 5), dt.date(2026, 10, 7), dt.date(2027, 6, 1)):
        got = sn.us_names(listing, asked, day)["symbols"]
        assert got == _r0_us_names(listing, asked, day)


def test_kr_names_and_envelope_symbols(tmp_path: Path) -> None:
    table = write_stock_master(
        tmp_path, [("005930", "삼성전자"), ("005935", "삼성전자우"), ("000001", None)]
    )
    block = sn.kr_names(table, {"005930", "000001", "999999"})
    assert block["symbols"] == {
        "005930": {"name": "삼성전자", "current": True},
        "000001": {"name": None, "current": True},
    }
    assert block["source"] == "KR raw stock_master.name (snapshot_date=2026-10-07)"
    assert block["basis"] == "수집 시점 현재 이름"
    env = envelope(D1, ["A", "B"], ["005930"])
    assert sn.envelope_symbols(env) == {"KR": {"005930"}, "US": {"A", "B"}}
    env["markets"].append({"market": "US", "rankings": [{"rank": 1}, "x", {"symbol": 5}]})
    assert sn.envelope_symbols(env)["US"] == {"A", "B"}


def test_the_output_is_security_names_v1_and_is_read_by_the_renderer(tmp_path: Path) -> None:
    listing = write_listing(tmp_path / "us", LISTING)
    table = write_stock_master(tmp_path / "kr", [("005930", "삼성전자")])
    env = envelope(D1, ["AAA", "BBB"], ["005930"])
    data = sn.build_security_names(env, us_listing=listing, kr_stock_master=table)
    raw = sn.dumps(data)
    assert raw.endswith(b"\n") and raw == sn.dumps(json.loads(raw))
    assert json.loads(raw)["schema"] == md.SECURITY_NAMES_SCHEMA
    loaded = md.load_security_names(raw)
    assert loaded["US"]["symbols"]["BBB"] == {"name": "Old Barclays ETN", "current": False}
    assert loaded["KR"]["symbols"]["005930"]["name"] == "삼성전자"
    assert str(tmp_path) not in raw.decode()  # 경로를 싣지 않습니다
    # source·basis에 경로나 호스트명이 들어가면 만들지 않습니다
    data["markets"]["US"]["source"] = "/home/whi/data/x"
    with pytest.raises(sn.NamesError, match="경로"):
        sn.dumps(data)


# --- 레이크 root에서 찾기 ------------------------------------------------------------------------
def test_find_us_listing_takes_the_latest_snapshot_that_has_files(tmp_path: Path) -> None:
    with pytest.raises(sn.NamesError, match="listing_snapshots"):
        sn.find_us_listing(tmp_path)
    older = write_listing(tmp_path, LISTING, snapshot="2026-09-18")
    (older.parent / "snapshot_date=2026-10-01").mkdir()  # 비어 있는 snapshot은 건너뜁니다
    assert sn.find_us_listing(tmp_path) == older
    newer = write_listing(tmp_path, LISTING[:3], snapshot="2026-09-29")
    assert sn.find_us_listing(tmp_path) == newer


def test_find_kr_stock_master_needs_the_success_marker_and_prefers_not_after_d(
    tmp_path: Path,
) -> None:
    with pytest.raises(sn.NamesError, match="stock_master"):
        sn.find_kr_stock_master(tmp_path, DAY)
    ok = write_stock_master(tmp_path, [("005930", "삼성전자")], snapshot="2026-10-06")
    write_stock_master(tmp_path, [("005930", "내보내는 중")], snapshot="2026-10-07", marker=False)
    write_stock_master(tmp_path, [("005930", "내일 것")], snapshot="2026-10-08")
    assert sn.find_kr_stock_master(tmp_path, DAY) == ok
    assert "snapshot_date=2026-10-06" in str(ok)
    # D 이전 snapshot이 없으면 D 뒤 가장 이른 것(지난 날짜 단위를 다시 만드는 경우)
    assert sn.find_kr_stock_master(tmp_path, dt.date(2026, 10, 5)) == ok
    only_later = tmp_path / "later"
    write_stock_master(only_later, [("005930", "a")], snapshot="2026-10-09")
    earliest = write_stock_master(only_later, [("005930", "b")], snapshot="2026-10-08")
    assert sn.find_kr_stock_master(only_later, DAY) == earliest


def test_from_roots_builds_both_markets_and_reports_counts(tmp_path: Path) -> None:
    write_listing(tmp_path / "us", LISTING)
    write_stock_master(tmp_path / "kr", [("005930", "삼성전자")])
    env = envelope(D1, ["AAA", "BBB", "MISSING"], ["005930", "000660"])
    built = sn.from_roots(env, us_root=tmp_path / "us", kr_root=tmp_path / "kr")
    assert built.notes == {}
    assert built.counts == {"US": (2, 3), "KR": (1, 2)}
    assert set(md.load_security_names(built.raw)) == {"US", "KR"}


def test_from_roots_keeps_going_when_a_source_is_missing_or_unreadable(tmp_path: Path) -> None:
    env = envelope(D1, ["AAA"], ["005930"])
    # root를 안 준 시장은 사유도 없습니다
    assert sn.from_roots(env, us_root=None, kr_root=None) == sn.Built(None, {}, {})
    # root는 있는데 원천이 없습니다
    built = sn.from_roots(env, us_root=tmp_path / "no-us", kr_root=tmp_path / "no-kr")
    assert built.raw is None
    assert built.notes == {
        "US": "US listing_snapshots가 레이크에 없음",
        "KR": "KR raw snapshot의 stock_master가 없음",
    }
    # US는 읽히고 KR만 못 읽습니다 → US 블록만 나옵니다
    write_listing(tmp_path / "us", LISTING)
    built = sn.from_roots(env, us_root=tmp_path / "us", kr_root=tmp_path / "no-kr")
    assert set(md.load_security_names(built.raw)) == {"US"}
    assert built.notes == {"KR": "KR raw snapshot의 stock_master가 없음"}
    # 깨진 parquet는 예외 이름만 사유에 적고 경로는 적지 않습니다
    listing = next((tmp_path / "us/derived/snapshots/listing_snapshots").iterdir())
    (listing / "part.parquet").write_bytes(b"not parquet")
    built = sn.from_roots(env, us_root=tmp_path / "us", kr_root=None)
    assert built.raw is None
    assert (
        built.notes["US"].startswith("US 이름 원천를 읽지 못함 (")
        and str(tmp_path) not in (built.notes["US"])
    )
    # report_date가 깨진 envelope
    broken = sn.from_roots({"report_date": "x"}, us_root=tmp_path, kr_root=None)
    assert broken.raw is None and "report_date" in broken.notes["US"]


def test_cli_writes_the_same_bytes_as_the_library(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    listing = write_listing(tmp_path / "us", LISTING)
    table = write_stock_master(tmp_path / "kr", [("005930", "삼성전자")])
    env = envelope(D1, ["AAA", "BBB"], ["005930"])
    env_path = tmp_path / "env.json"
    env_path.write_text(json.dumps(env), encoding="utf-8")
    out = tmp_path / "out/names.json"
    code = sn.main(
        ["--envelope", str(env_path), "--us-listing", str(listing), "--kr-stock-master", str(table)]
        + ["--out", str(out)]
    )
    assert code == 0
    assert out.read_bytes() == sn.dumps(
        sn.build_security_names(env, us_listing=listing, kr_stock_master=table)
    )
    assert capsys.readouterr().out.strip() == f"{D1}: US 2/2 (낡음 1) KR 1/1 (낡음 0)"
    # root로도 같은 결과입니다
    out2 = tmp_path / "names2.json"
    argv = ["--envelope", str(env_path), "--us-root", str(tmp_path / "us")]
    assert sn.main([*argv, "--kr-root", str(tmp_path / "kr"), "--out", str(out2)]) == 0
    assert (
        json.loads(out2.read_text())["markets"]["US"]["symbols"]
        == (json.loads(out.read_text())["markets"]["US"]["symbols"])
    )
    assert sn.main(["--envelope", str(env_path), "--out", str(out2)]) == 1  # 원천을 안 줌
    capsys.readouterr()


# --- 렌더러: 사유 ---------------------------------------------------------------------------
def _ctx(world: World, names: bytes | None, notes: dict | None) -> dict:
    run = world.make_run(D1)
    report = (run / f"report-{D1}.json").read_bytes()
    return md.build_context(
        world.checkout, report, None, RELEASE, f"{D1}T10:03:12+09:00", 100, no_log,
        security_names=names, names_notes=notes,
    )  # fmt: skip


@pytest.fixture()
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_notes_explain_the_omitted_column_only_in_data_status(world: World) -> None:
    plain = md.render_unit(_ctx(world, None, None))
    same = md.render_unit(_ctx(world, None, {}))
    assert plain == same  # 사유가 없으면 이전 출력과 같습니다
    notes = {"US": "US listing_snapshots가 레이크에 없음", "KR": "x|y"}
    files = md.render_unit(_ctx(world, None, notes))
    status = files["data-status.md"]
    assert "(사유: US listing_snapshots가 레이크에 없음)" in status
    assert "| 이름 원천이 없어 `종류` 열을 생략했습니다" in status
    # KR은 순위 행에 이름이 있어 열을 보입니다 → 사유를 적지 않습니다
    assert "x\\|y" not in status
    assert {k: v for k, v in files.items() if k != "data-status.md"} == {
        k: v for k, v in plain.items() if k != "data-status.md"
    }
    assert md.validate_unit_files(D1, files) == []


# --- publisher 로컬·동기화 단계 ------------------------------------------------------------------
def _us_symbols(world: World) -> list[str]:
    report = json.loads((world.runs / D1 / f"report-{D1}.json").read_text())
    lgb = next(m for m in report["markets"] if m["market"] == "US")
    return [r["symbol"] for r in lgb["rankings"]]


def _lake(world: World, tmp_path: Path) -> dict:
    """합성 US listing·KR stock_master. 처음 세 종목은 채권형·펀드·우선주 이름입니다."""
    symbols = _us_symbols(world)
    names = {
        symbols[0]: "Ford Motor Company 6.500% Notes due 2062",
        symbols[1]: "Blackrock Enhanced Equity Dividend Trust",
        symbols[2]: "Annaly Capital 6.95% Series F Preferred Stock",
    }
    rows = [
        ("2026-09-01", "nasdaqlisted", s, names.get(s, f"Sample {s} Inc. Common Stock"))
        for s in symbols
    ]
    write_listing(tmp_path / "us", rows)
    write_stock_master(tmp_path / "kr", [("100001", "샘플스팩")])
    return {
        "security_names_us_root": str(tmp_path / "us"),
        "security_names_kr_root": str(tmp_path / "kr"),
    }


def _unit_files(world: World, root: Path | None = None) -> dict:
    directory = root or world.local_unit(D1)
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(directory.iterdir())}


def test_local_step_adds_the_kind_column_and_pins_the_names_input(
    world: World, tmp_path: Path
) -> None:
    world.make_run(D1)
    roots = _lake(world, tmp_path)
    out = pr.local_step(world.config(D1, **roots))
    assert out["status"] == "local_ready"
    files = _unit_files(world)
    assert "| 순위 | 코드 | 종류 |" in files["us-stocks.md"]
    assert "채권형" in files["us-stocks.md"] and "펀드" in files["us-stocks.md"]
    assert "보통주가 아닌 종목" in files["us-stocks.md"]
    assert "listing_snapshots.security_name (snapshot_date=2026-09-18)" in files["data-status.md"]
    assert "증권 이름 입력 sha256" in files["data-status.md"]
    # 고정한 입력과 sha256가 manifest·journal에 같이 남습니다
    names_file = world.runs / D1 / "markdown" / pr.NAMES_FILE
    manifest = json.loads((world.runs / D1 / "markdown/manifest.json").read_text())
    entry = world.journal()["units"][D1]
    for record in (manifest, entry):
        assert record["security_names_path"] == str(names_file)
        digest = pr.hashlib.sha256(names_file.read_bytes()).hexdigest()
        assert record["security_names_sha256"] == digest
        assert record["security_names_notes"] is None
    # 서버 경로를 이름 입력이나 단위에 싣지 않습니다
    assert str(tmp_path) not in names_file.read_text(encoding="utf-8")
    assert all(str(tmp_path) not in text for text in files.values())


def test_sync_step_renders_from_the_pinned_names_and_publishes_the_kind_column(
    world: World, tmp_path: Path
) -> None:
    world.make_run(D1)
    roots = _lake(world, tmp_path)
    result = pr.run_step(world.config(D1, **roots))
    assert result["status"] == "published"
    remote = world.remote_file(f"reports/daily-briefing/2026/10/{D1}/us-stocks.md")
    assert "| 순위 | 코드 | 종류 |" in remote
    # 같은 입력으로 다시 돌면 새 커밋이 없습니다(정정 경로도 같은 입력을 씁니다)
    head = world.remote_head()
    again = pr.run_step(world.config(D1, **roots))
    assert again["status"] in ("unchanged", "nothing_to_do") and world.remote_head() == head
    assert world.remote_file(f"reports/daily-briefing/2026/10/{D1}/us-stocks.md") == remote


def test_correction_rerenders_with_the_pinned_names(world: World, tmp_path: Path) -> None:
    """이름 없이 올린 단위를 이름 입력을 켠 local 단계 뒤 `--correct`로 바로잡습니다(R6 D2)."""
    world.make_run(D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    before = world.remote_file(f"reports/daily-briefing/2026/10/{D1}/us-stocks.md")
    assert "| 순위 | 코드 | 종류 |" not in before
    roots = _lake(world, tmp_path)
    assert pr.local_step(world.config(D1, **roots))["status"] == "local_ready"
    out = pr.sync_step(world.config(D1), correct=D1, reason="증권 종류 열 추가")
    assert out["status"] == "published", out
    after = world.remote_file(f"reports/daily-briefing/2026/10/{D1}/us-stocks.md")
    assert "| 순위 | 코드 | 종류 |" in after
    assert "정정 r2" in world.remote_file(f"reports/daily-briefing/2026/10/{D1}/README.md")


def test_without_roots_the_unit_is_what_the_old_publisher_rendered(world: World) -> None:
    run = world.make_run(D1)
    config = world.config(D1)
    assert pr.local_step(config)["status"] == "local_ready"
    ctx = md.build_context(
        world.checkout,
        (run / f"report-{D1}.json").read_bytes(),
        (run / f"market-sector-{D1}.json").read_bytes(),
        RELEASE,
        config["generated_at"],
        md.DEFAULT_TOP_N,
        no_log,
    )
    assert _unit_files(world) == md.render_unit(ctx)
    assert not (run / "markdown" / pr.NAMES_FILE).exists()
    entry = world.journal()["units"][D1]
    assert entry["security_names_path"] is None and entry["security_names_notes"] is None
    # null로 준 root도 같습니다
    config = world.config(D1, security_names_us_root=None, security_names_kr_root=None)
    assert pr.local_step(config)["status"] == "local_ready"
    assert _unit_files(world) == md.render_unit(ctx)


def test_missing_or_unreadable_roots_keep_the_unit_and_write_the_reason(
    world: World, tmp_path: Path
) -> None:
    world.make_run(D1)
    config = world.config(D1, security_names_us_root=str(tmp_path / "gone"))
    assert pr.local_step(config)["status"] == "local_ready"
    files = _unit_files(world)
    assert "| 순위 | 코드 | 종류 |" not in files["us-stocks.md"]
    assert "(사유: US listing_snapshots가 레이크에 없음)" in files["data-status.md"]
    assert str(tmp_path) not in "".join(files.values())
    assert world.journal()["units"][D1]["security_names_notes"] == {
        "US": "US listing_snapshots가 레이크에 없음"
    }
    assert not (world.runs / D1 / "markdown" / pr.NAMES_FILE).exists()
    # 이어서 동기화해도 같은 사유로 올라갑니다
    assert pr.sync_step(world.config(D1))["status"] == "published"
    status = world.remote_file(f"reports/daily-briefing/2026/10/{D1}/data-status.md")
    assert "(사유: US listing_snapshots가 레이크에 없음)" in status
    # 레이크가 생기면 같은 날 다시 돌아도 이름이 들어옵니다(이전 잔재 없음)
    _lake(world, tmp_path)
    ok = world.config(D1, security_names_us_root=str(tmp_path / "us"))
    assert pr.local_step(ok)["status"] == "local_ready"
    assert "| 순위 | 코드 | 종류 |" in _unit_files(world)["us-stocks.md"]


def test_a_changed_names_file_stops_a_correction(world: World, tmp_path: Path) -> None:
    """정정은 journal에 적은 입력으로 다시 렌더합니다. 그 사이 입력이 바뀌었으면 멈춥니다."""
    world.make_run(D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    roots = _lake(world, tmp_path)
    assert pr.local_step(world.config(D1, **roots))["status"] == "local_ready"
    head = world.remote_head()
    names_file = world.runs / D1 / "markdown" / pr.NAMES_FILE
    original = names_file.read_text(encoding="utf-8")
    names_file.write_text("{}", encoding="utf-8")
    out = pr.sync_step(world.config(D1), correct=D1, reason="증권 종류 열 추가")
    assert out["status"] == "failed" and "security names input changed" in out["detail"]
    names_file.unlink()
    out = pr.sync_step(world.config(D1), correct=D1, reason="증권 종류 열 추가")
    assert out["status"] == "failed" and "security names input" in out["detail"]
    assert world.remote_head() == head
    names_file.write_text(original, encoding="utf-8")
    assert pr.sync_step(world.config(D1), correct=D1, reason="증권 종류 열 추가")["status"] == (
        "published"
    )


def test_config_accepts_only_absolute_names_roots(world: World, tmp_path: Path) -> None:
    world.make_run(D1)
    path = tmp_path / "cfg.json"
    good = world.config(D1, expected_remote_url=pr.EXPECTED_REMOTE_URL)
    path.write_text(json.dumps({**good, "security_names_us_root": "/lake/us"}))
    assert pr.read_config(path, need_local=True)["security_names_us_root"] == "/lake/us"
    path.write_text(json.dumps({**good, "security_names_kr_root": "relative/kr"}))
    with pytest.raises(pr.PublishError, match="absolute"):
        pr.read_config(path, need_local=True)
    path.write_text(json.dumps({**good, "security_names_kr_root": None}))
    assert pr.read_config(path, need_local=True)["security_names_kr_root"] is None


# --- coordinator ------------------------------------------------------------------------------
def test_coordinator_resolves_names_roots_with_the_market_sector_roots_as_fallback() -> None:
    assert dc._security_names_roots({}) == {}
    only_ms = {"market_sector_us_root": "/lake/us", "market_sector_kr_root": "/lake/kr"}
    assert dc._security_names_roots(only_ms) == {
        "security_names_us_root": "/lake/us",
        "security_names_kr_root": "/lake/kr",
    }
    own = {**only_ms, "security_names_us_root": "/names/us"}
    assert dc._security_names_roots(own) == {
        "security_names_us_root": "/names/us",
        "security_names_kr_root": "/lake/kr",
    }
    assert dc._security_names_roots({"security_names_kr_root": "/names/kr"}) == {
        "security_names_kr_root": "/names/kr"
    }
    with pytest.raises(ValueError, match="absolute"):
        dc._security_names_roots({"security_names_us_root": "relative"})


def test_publisher_input_carries_the_roots_only_when_configured() -> None:
    base = {
        "reports_audience": "owner_only",
        "reports_remote_url": "git@github.com:sjleekor/stock_reports.git",
        "reports_branch": "main",
        "release_manifest": "/s/releases/r1/release.json",
        "model_cards_path": "/s/releases/r1/model-cards.json",
        "reports_checkout": "/s/reports-checkout",
    }
    render = {"report_sha256": "a" * 64, "invocation_id": "inv"}
    args = (dt.date(2026, 10, 7), render, Path("/s/runs/2026-10-07"))
    plain = dc._reports_publisher_input(base, *args)
    assert not any(key.startswith("security_names") for key in plain)
    with_roots = dc._reports_publisher_input({**base, "market_sector_us_root": "/lake/us"}, *args)
    assert with_roots == {**plain, "security_names_us_root": "/lake/us"}
    assert set(with_roots) <= pr.ALLOWED_KEYS  # publisher가 받는 키입니다
