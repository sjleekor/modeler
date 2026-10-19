"""순위 종목의 증권 이름 입력(`security-names.v2`, 옛 `v1`도 읽힙니다)을 레이크에서 만듭니다.

R0 임시 도구 `build_names.py`(2026-10-06)의 규칙을 그대로 옮긴 것입니다. 자동 게시 경로가
`markdown.build_context(security_names=)`에 이름을 넘기지 못해 US 순위 표의 `종류` 열이 빠지던 결함
(R6 절차서 D1)을 고칩니다. 규칙을 바꾸면 R0 결과와 어긋나므로 바꾸지 않습니다.

  US(v2, 유니버스 v2 서빙 W4) 이름은 `listing_snapshots_v2.security_name`, 종류는 증권 마스터입니다.
      심볼 → `universe_daily_v2.security_id`(D가 속한 달의 첫 세션, 그 달에 D 이전 날짜가 없으면
      D 이전 가장 최근 날짜) → `security_master`(D 시점 유효 행: valid_from <= D < valid_to,
      valid_to가 비면 열림) → 표시 종류(`us_kind`). 종류는 이름 낡음과 무관합니다.
      v2 listing이 레이크에 없으면 아래 v1 방식으로 되돌아갑니다(옛 레이크·시험용).
  US(v1) 레이크 `listing_snapshots.security_name`. as_of <= D 중 가장 최근 행을 씁니다. 그 행의 갈래
      (nasdaqlisted·otherlisted) 마지막 목록(as_of <= D)에 없던 심볼은 이름이 낡았을 수 있어
      `current=false`로 둡니다(심볼 재사용: BXDC, BNY 등). 판정은 렌더러가 합니다.
  KR  raw snapshot `stock_master.name`(수집 시점 현재 이름). 순위 행에 이름이 있으면 렌더러가 그것을
      먼저 쓰므로 이 입력은 이름이 없는 행(R3 F3 이전 report)만 채웁니다.

읽기만 합니다. 출력에는 서버 경로를 넣지 않습니다. polars는 함수 안에서 불러 이 패키지를 가볍게
둡니다(serving venv에 있습니다).

수동 사용::

    python -m modeler.reporting.security_names --envelope report-D.json \\
        --us-root <stock_data>/us --kr-root <stock_data>/kr --out names-D.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from modeler.reporting.markdown import (
    KIND_ADR,
    KIND_BY_MASTER,
    KIND_COMMON,
    KIND_OTHER,
    KIND_UNKNOWN,
    SECURITY_NAMES_SCHEMA,
    SECURITY_NAMES_SCHEMA_V2,
)

KR_SOURCE = "sj2_remote"
MARKETS = ("US", "KR")
_SNAPSHOT_DIR = re.compile(r"snapshot_date=(\d{4}-\d{2}-\d{2})\Z")
_SNAPSHOT_IN_PATH = re.compile(r"snapshot_date=(\d{4}-\d{2}-\d{2})")
# 우리가 쓰는 문자열(source·basis)에 경로나 호스트명이 들어가면 안 됩니다.
_FORBIDDEN = re.compile(r"/home/|/Users/|/tmp/|sj2")


class NamesError(Exception):
    """이름 입력을 만들 수 없습니다. 메시지에 경로가 없어 데이터 상태에 그대로 적을 수 있습니다."""


# ---------------------------------------------------------------------------
# 입력 찾기
# ---------------------------------------------------------------------------
def _snapshot_dirs(parent: Path) -> list[tuple[str, Path]]:
    """`snapshot_date=YYYY-MM-DD` 디렉터리를 최근 순으로 돌려줍니다. 링크는 건너뜁니다."""
    if not parent.is_dir():
        return []
    found = []
    for child in parent.iterdir():
        match = _SNAPSHOT_DIR.match(child.name)
        if match and child.is_dir() and not child.is_symlink():
            found.append((match.group(1), child))
    return sorted(found, reverse=True)


def _parquet_files(path: Path) -> list[Path]:
    """파일 하나거나, 디렉터리 안의 parquet 전부(숨김·`_`로 시작하는 파일 제외)입니다."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        p
        for p in path.rglob("*.parquet")
        if p.is_file() and not p.is_symlink() and not p.name.startswith((".", "_"))
    )


def find_us_listing(us_root: Path) -> Path:
    """`<us>/derived/snapshots/listing_snapshots/snapshot_date=<가장 최근>/`를 돌려줍니다."""
    parent = us_root / "derived" / "snapshots" / "listing_snapshots"
    for _date, directory in _snapshot_dirs(parent):
        if _parquet_files(directory):
            return directory
    raise NamesError("US listing_snapshots가 레이크에 없음")


def find_kr_stock_master(kr_root: Path, day: dt.date) -> Path:
    """KR raw snapshot의 `stock_master/`를 돌려줍니다. D 이전(포함) 가장 최근 것을 먼저 씁니다.

    D 이전 snapshot이 없으면(서버는 snapshot을 하나만 보존합니다. 지난 날짜 단위를 다시 만들 때)
    D 뒤 가장 이른 것을 씁니다. 이름은 "수집 시점 현재 이름"이고 어느 snapshot인지는 `source`에
    적힙니다. 끝났다는 표시(`_manifests/_SUCCESS.json`)가 없는 snapshot은 쓰지 않습니다
    (내보내는 중일 수 있습니다).
    """
    parent = kr_root / "raw" / "raw_postgres"
    usable = []
    for snapshot, directory in _snapshot_dirs(parent):
        source = directory / f"source={KR_SOURCE}"
        table = source / "stock_master"
        if (source / "_manifests" / "_SUCCESS.json").is_file() and _parquet_files(table):
            usable.append((snapshot, table))
    before = [table for snapshot, table in usable if snapshot <= day.isoformat()]
    after = [table for snapshot, table in usable if snapshot > day.isoformat()]
    if before:
        return before[0]  # 최근 순으로 정렬돼 있습니다
    if after:
        return after[-1]
    raise NamesError("KR raw snapshot의 stock_master가 없음")


# ---------------------------------------------------------------------------
# 이름 만들기 (R0 build_names.py와 같은 규칙)
# ---------------------------------------------------------------------------
def _read(path: Path, columns: list[str]):
    import polars as pl

    files = _parquet_files(path)
    if not files:
        raise NamesError("이름 원천 parquet가 없음")
    return pl.concat([pl.read_parquet(f, columns=columns) for f in files], how="vertical")


def _snapshot_label(path: Path) -> str:
    match = _SNAPSHOT_IN_PATH.search(str(path))
    return f" (snapshot_date={match.group(1)})" if match else ""


def us_names(
    listing: Path, symbols: set[str], day: dt.date, table: str = "listing_snapshots"
) -> dict:
    """US 이름 블록 {"source", "basis", "symbols"}. `listing`은 parquet 파일이나 디렉터리입니다.

    `table`은 source에 적는 표 이름입니다(v1 `listing_snapshots`, v2 `listing_snapshots_v2`).
    """
    import polars as pl

    df = _read(listing, ["as_of", "kind", "symbol", "security_name"])
    df = df.filter(pl.col("as_of") <= day)
    last_by_kind = {
        row["kind"]: row["as_of"]
        for row in df.group_by("kind").agg(pl.col("as_of").max()).iter_rows(named=True)
    }
    # 같은 심볼의 행 중 as_of가 가장 늦은 것. 같으면 kind 사전순으로 뒤, 그다음은 입력 순서입니다.
    latest = (
        df.filter(pl.col("symbol").is_in(sorted(symbols)))
        .sort(["symbol", "as_of", "kind"], maintain_order=True)
        .group_by("symbol", maintain_order=True)
        .last()
    )
    out = {}
    for row in latest.iter_rows(named=True):
        name = row["security_name"]
        out[row["symbol"]] = {
            "name": name if isinstance(name, str) else None,
            "as_of": row["as_of"].isoformat(),
            "kind": row["kind"],
            "current": bool(row["as_of"] == last_by_kind[row["kind"]]),
        }
    basis = (
        f"as_of <= {day} 중 가장 최근 이름. 갈래별 마지막 목록 "
        + "·".join(f"{k} {v}" for k, v in sorted(last_by_kind.items()))
        + ". 마지막 목록에 없던 심볼은 이름이 낡았을 수 있어 판정하지 않음"
    )
    source = f"US 레이크 {table}.security_name" + _snapshot_label(listing)
    return {"source": source, "basis": basis, "symbols": out}


# ---------------------------------------------------------------------------
# US 종류: 증권 마스터 (유니버스 v2 서빙 S7)
# ---------------------------------------------------------------------------
_FUND_ISSUERS = frozenset({"bdc", "registered_fund"})
_FUND_TYPES = frozenset({"fund", "trust", "etf"})
_BOND_TYPES = frozenset({"bond", "etn"})
_OTHER_TYPES = frozenset({"warrant", "right", "unit", "warrant_right_unit", "test"})
_OTHER_REASONS = frozenset({"issuer_name_noncommon", "issuer_only_suffix"})


def us_kind(master: dict | None) -> str:
    """증권 마스터 행 하나의 표시 종류(S7 표, 먼저 걸리는 것이 이김). 행이 없으면 확인 안 됨."""
    if not master:
        return KIND_UNKNOWN
    stype = master.get("security_type")
    issuer = master.get("issuer_kind")
    reason = master.get("reason_code")
    if master.get("include") is True:
        if stype == "adr":
            return KIND_ADR
        if stype == "reit":
            return "리츠"
        return KIND_COMMON
    if issuer == "spac":
        return "SPAC"
    if issuer in _FUND_ISSUERS or stype in _FUND_TYPES:
        return "펀드"
    if stype in _BOND_TYPES:
        return "채권형"
    if stype == "preferred":
        return "우선주"
    if stype in _OTHER_TYPES or reason in _OTHER_REASONS:
        return KIND_OTHER
    return KIND_UNKNOWN


def find_us_v2_snapshot(us_root: Path, table: str) -> Path | None:
    """`<us>/derived/snapshots/<table>/snapshot_date=<가장 최근, 파일 있는 것>/`. 없으면 None."""
    for _date, directory in _snapshot_dirs(us_root / "derived" / "snapshots" / table):
        if _parquet_files(directory):
            return directory
    return None


def _completion_rule_version(directory: Path) -> str | None:
    try:
        data = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        value = data.get("rule_version")
        return value if isinstance(value, str) and value else None
    except (OSError, ValueError, AttributeError):
        return None


def _identity_date(universe: Path, day: dt.date) -> tuple[dt.date | None, bool]:
    """(M, 달 첫 세션 여부). D의 달에서 D 이하인 가장 이른 날짜, 없으면 D 이하 가장 최근 날짜."""
    import polars as pl

    upto = (
        pl.scan_parquet(_parquet_files(universe))
        .select("date")
        .filter(pl.col("date") <= day)
        .unique()
        .collect()["date"]
    )
    if upto.len() == 0:
        return None, False
    in_month = upto.filter((upto.dt.year() == day.year) & (upto.dt.month() == day.month))
    if in_month.len():
        return in_month.min(), True
    return upto.max(), False


def us_session_date(envelope: dict) -> tuple[dt.date | None, str | None]:
    """US 순위 세션 A: US 시장 항목들의 `feature_asof_date`. (A, 사유).

    값이 하나로 모이지 않으면(없음·서로 다름·형식 오류) A는 None이고 사유를 돌려줍니다.
    """
    found: set[dt.date] = set()
    for market in envelope.get("markets") or []:
        if isinstance(market, dict) and market.get("market") == "US":
            raw = market.get("feature_asof_date")
            if raw is None:
                continue
            try:
                found.add(dt.date.fromisoformat(str(raw)))
            except ValueError:
                return None, "US 순위의 feature_asof_date를 읽을 수 없어 report_date로 대신함"
    if len(found) == 1:
        return next(iter(found)), None
    if not found:
        return None, "US 순위에 feature_asof_date가 없어 report_date로 대신함"
    return None, "US 순위의 feature_asof_date가 서로 달라 report_date로 대신함"


def us_names_master(
    listing_v2: Path,
    universe: Path | None,
    master: Path | None,
    symbols: set[str],
    day: dt.date,
    kind_day: dt.date | None = None,
    kind_notes: tuple[str, ...] = (),
) -> tuple[dict, dict]:
    """US v2 블록과 통계. 이름은 listing_v2, 종류는 universe_daily_v2 → security_master입니다.

    이름은 `day`(report_date) 이전 최신 행이고, 종류는 `kind_day`(US 순위 세션 A, 없으면 `day`)
    기준입니다: 식별자는 A가 속한 달의 첫 세션, 마스터는 A 시점 유효 행입니다.

    `universe`·`master`가 None이면(레이크에 없음) 종류는 전부 확인 안 됨이고 통계 notes에 적습니다.
    """
    import polars as pl

    base = us_names(listing_v2, symbols, day, table="listing_snapshots_v2")
    names = base["symbols"]
    kd = kind_day or day
    notes: list[str] = list(kind_notes)
    sec_ids: dict[str, str] = {}
    ambiguous = 0
    identity, first_session = None, False
    rule_version = None
    if universe is None or master is None:
        notes.append("증권 마스터 입력(universe_daily_v2·security_master)이 레이크에 없음")
    else:
        identity, first_session = _identity_date(universe, kd)
        if identity is None:
            notes.append(f"universe_daily_v2에 {kd} 이전 날짜가 없음")
        else:
            if not first_session:
                notes.append(
                    f"universe_daily_v2에 {kd:%Y-%m} 날짜가 없어 가장 최근 {identity}로 대신함"
                )
            rows = (
                pl.scan_parquet(_parquet_files(universe))
                .select("date", "symbol", "security_id")
                .filter((pl.col("date") == identity) & pl.col("symbol").is_in(sorted(symbols)))
                .collect()
            )
            per_symbol: dict[str, set] = {}
            for sym, sid in zip(rows["symbol"], rows["security_id"], strict=True):
                per_symbol.setdefault(sym, set()).add(sid)
            for sym, ids in per_symbol.items():
                good = {i for i in ids if isinstance(i, str)}
                if len(good) == 1:
                    sec_ids[sym] = next(iter(good))
                elif len(good) > 1:
                    ambiguous += 1
            if ambiguous:
                notes.append(
                    f"{identity}에 심볼 하나가 서로 다른 security_id를 가진 경우 {ambiguous}개"
                )
        rule_version = _completion_rule_version(universe)
    masters: dict[str, dict] = {}
    if master is not None and sec_ids:
        ids = sorted(set(sec_ids.values()))
        frame = (
            pl.scan_parquet(_parquet_files(master))
            .filter(pl.col("security_id").is_in(ids))
            .filter(
                (pl.col("valid_from") <= kd)
                & (pl.col("valid_to").is_null() | (pl.col("valid_to") > kd))
            )
            .collect()
            .sort(["security_id", "valid_from"], maintain_order=True)
        )
        for row in frame.iter_rows(named=True):
            masters[row["security_id"]] = (
                row  # 같은 id가 여럿이면 valid_from이 가장 늦은 행이 남습니다
            )
    out: dict[str, dict] = {}
    kinds: dict[str, int] = {}
    no_v2 = no_master = 0
    for sym in sorted(symbols):
        sid = sec_ids.get(sym)
        row = masters.get(sid) if sid else None
        if sid is None:
            no_v2 += 1
        elif row is None:
            no_master += 1
        kind = us_kind(row)
        kinds[kind] = kinds.get(kind, 0) + 1
        name_info = names.get(sym)
        if name_info is None and sid is None:
            continue  # 이름도 식별자도 없는 심볼은 싣지 않습니다
        out[sym] = {
            "name": name_info["name"] if name_info else None,
            "as_of": name_info["as_of"] if name_info else None,
            "listing_kind": name_info["kind"] if name_info else None,
            "current": name_info["current"] if name_info else False,
            "kind": kind,
        }
    parts = []
    if identity is not None:
        parts.append(f"universe_daily_v2{_snapshot_label(universe)} 기준일 {identity}")
    if master is not None and identity is not None:
        parts.append(f"security_master{_snapshot_label(master)} {kd} 시점 유효 행")
    kind_source = "US 레이크 " + " → ".join(parts) if parts else "증권 마스터 없음"
    if rule_version:
        kind_source += f", rule_version {rule_version}"
    stats = {
        "requested": len(symbols),
        "kinds": dict(sorted(kinds.items())),
        "no_v2_row": no_v2,
        "ambiguous_v2": ambiguous,
        "no_master_row": no_master,
        "no_name": sum(1 for s in symbols if (names.get(s) or {}).get("name") is None),
        "kind_date": kd.isoformat(),
        "identity_date": identity.isoformat() if identity else None,
        "month_first_session": first_session,
        "notes": notes,
    }
    block = {
        "source": base["source"],
        "basis": base["basis"],
        "kind_by": KIND_BY_MASTER,
        "kind_source": kind_source,
        "stats": stats,
        "symbols": out,
    }
    return block, stats


def kr_names(stock_master: Path, symbols: set[str]) -> dict:
    """KR 이름 블록. `stock_master`는 parquet 파일이나 그 디렉터리입니다."""
    import polars as pl

    df = _read(stock_master, ["ticker", "name"]).filter(pl.col("ticker").is_in(sorted(symbols)))
    out = {t: {"name": n, "current": True} for t, n in zip(df["ticker"], df["name"], strict=True)}
    source = "KR raw stock_master.name" + _snapshot_label(stock_master)
    return {"source": source, "basis": "수집 시점 현재 이름", "symbols": out}


def envelope_symbols(envelope: dict) -> dict[str, set[str]]:
    """envelope의 순위 행에서 시장별 심볼 집합을 모읍니다(모델 전부, 표시 행 수와 무관)."""
    found: dict[str, set[str]] = {"KR": set(), "US": set()}
    for market in envelope.get("markets") or []:
        if isinstance(market, dict) and market.get("market") in found:
            for row in market.get("rankings") or []:
                if isinstance(row, dict) and isinstance(row.get("symbol"), str):
                    found[market["market"]].add(row["symbol"])
    return found


def _report_date(envelope: dict) -> dt.date:
    try:
        return dt.date.fromisoformat(str(envelope.get("report_date")))
    except ValueError:
        raise NamesError("envelope의 report_date를 읽을 수 없음") from None


def _us_block(
    us_root: Path, symbols: set[str], day: dt.date, envelope: dict | None = None
) -> tuple[dict, dict | None, str | None]:
    """root에서 US 블록·통계·사유를 만듭니다.

    v2 listing이 없으면 v1 이름 규칙 방식이고 통계는 None, 사유에 되돌아갔다고 적습니다.
    """
    listing_v2 = find_us_v2_snapshot(us_root, "listing_snapshots_v2")
    if listing_v2 is None:
        note = "listing_snapshots_v2가 레이크에 없어 v1 이름 규칙으로 판정함"
        return us_names(find_us_listing(us_root), symbols, day), None, note
    kind_day, why = us_session_date(envelope) if envelope is not None else (None, None)
    block, stats = us_names_master(
        listing_v2,
        find_us_v2_snapshot(us_root, "universe_daily_v2"),
        find_us_v2_snapshot(us_root, "security_master"),
        symbols,
        day,
        kind_day,
        (why,) if why else (),
    )
    return block, stats, None


def build_security_names(
    envelope: dict,
    *,
    us_listing: Path | None = None,
    kr_stock_master: Path | None = None,
    us_root: Path | None = None,
) -> dict:
    """이름 입력 본문을 만듭니다. 원천을 준 시장만 블록이 생깁니다.

    `us_listing`(v1 listing 경로)을 주면 이름 규칙용 `security-names.v1`입니다. 그 밖에는
    `security-names.v2`이고, `us_root` 레이크에 v2 표가 있으면 US 블록에 마스터 종류가 실립니다.
    """
    day = _report_date(envelope)
    symbols = envelope_symbols(envelope)
    markets = {}
    schema = SECURITY_NAMES_SCHEMA_V2
    if us_listing is not None:
        markets["US"] = us_names(us_listing, symbols["US"], day)
        schema = SECURITY_NAMES_SCHEMA
    elif us_root is not None:
        markets["US"], _stats, _note = _us_block(us_root, symbols["US"], day, envelope)
    if kr_stock_master is not None:
        markets["KR"] = kr_names(kr_stock_master, symbols["KR"])
    return {"schema": schema, "report_date": day.isoformat(), "markets": markets}


def dumps(data: dict) -> bytes:
    """R0 도구와 같은 형식(정렬, 들여쓰기 1, 끝 줄바꿈)의 UTF-8 바이트입니다."""
    for market, block in data["markets"].items():
        for field_name in ("source", "basis", "kind_source"):
            if _FORBIDDEN.search(block.get(field_name, "")):
                raise NamesError(f"{market}.{field_name}에 경로나 호스트명이 들어감")
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# 레이크 root에서 만들기 (publisher가 씁니다)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Built:
    """`from_roots`의 결과. `raw`가 None이면 이름 입력이 없습니다.

    `notes`는 이름 원천을 쓰지 못한 시장의 사유입니다(경로 없음, 데이터 상태에 적힙니다).
    """

    raw: bytes | None
    notes: dict[str, str] = field(default_factory=dict)
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)  # 시장 -> (찾은 수, 요청 수)
    # 시장 -> 종류 분포·식별자·마스터 행·이름 없는 수 등(US v2만). JSON 블록의 `stats`와 같음
    stats: dict[str, dict] = field(default_factory=dict)


def from_roots(envelope: dict, *, us_root: Path | None, kr_root: Path | None) -> Built:
    """레이크 root(`<stock_data>/us`, `<stock_data>/kr`)에서 이름 입력을 만듭니다.

    root를 주지 않은 시장은 건너뜁니다(사유도 적지 않습니다). root를 줬는데 원천이 없거나 읽지
    못하면 그 시장만 빼고 사유를 `notes`에 남깁니다. 예외를 밖으로 내보내지 않습니다.
    """
    try:
        day = _report_date(envelope)
    except NamesError as exc:
        return Built(None, {m: str(exc) for m, r in (("US", us_root), ("KR", kr_root)) if r})
    symbols = envelope_symbols(envelope)
    markets: dict[str, dict] = {}
    notes: dict[str, str] = {}
    counts: dict[str, tuple[int, int]] = {}
    stats: dict[str, dict] = {}
    for market, root, label in (("US", us_root, "US 이름 원천"), ("KR", kr_root, "KR 이름 원천")):
        if root is None:
            continue
        try:
            if market == "US":
                block, us_stats, us_note = _us_block(root, symbols["US"], day, envelope)
                if us_note:
                    notes["US"] = us_note
                if us_stats is not None:
                    stats["US"] = us_stats
                    if us_stats["notes"]:
                        notes["US"] = "; ".join(us_stats["notes"])
            else:
                block = kr_names(find_kr_stock_master(root, day), symbols["KR"])
        except NamesError as exc:
            notes[market] = str(exc)
            continue
        except Exception as exc:  # 읽기 실패는 단위를 막지 않고 열을 생략합니다
            notes[market] = f"{label}를 읽지 못함 ({type(exc).__name__})"
            continue
        markets[market] = block
        counts[market] = (len(block["symbols"]), len(symbols[market]))
    if not markets:
        return Built(None, notes, counts)
    try:
        raw = dumps(
            {"schema": SECURITY_NAMES_SCHEMA_V2, "report_date": day.isoformat(), "markets": markets}
        )
    except NamesError as exc:
        return Built(None, {m: str(exc) for m in markets} | notes, {})
    return Built(raw, notes, counts, stats)


# ---------------------------------------------------------------------------
# CLI (수동 단위·정정용)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--us-listing", type=Path, help="listing_snapshots parquet(파일·디렉터리)")
    parser.add_argument("--kr-stock-master", type=Path, help="stock_master parquet(파일·디렉터리)")
    parser.add_argument(
        "--us-root",
        type=Path,
        help="<stock_data>/us (v2 표가 있으면 증권 마스터 종류까지, 없으면 v1 listing)",
    )
    parser.add_argument("--kr-root", type=Path, help="<stock_data>/kr (D 이전 snapshot을 찾음)")
    args = parser.parse_args(argv)
    try:
        envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise NamesError("envelope가 JSON 객체가 아님")
        day = _report_date(envelope)
        us_listing = args.us_listing
        kr_master = args.kr_stock_master or (
            find_kr_stock_master(args.kr_root, day) if args.kr_root else None
        )
        if us_listing is None and args.us_root is None and kr_master is None:
            raise NamesError("--us-listing·--us-root·--kr-stock-master·--kr-root 중 하나가 필요함")
        data = build_security_names(
            envelope, us_listing=us_listing, kr_stock_master=kr_master, us_root=args.us_root
        )
        raw = dumps(data)
    except (NamesError, OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    symbols = envelope_symbols(envelope)
    parts = []
    for market, block in data["markets"].items():
        stale = sum(not v["current"] for v in block["symbols"].values())
        parts.append(f"{market} {len(block['symbols'])}/{len(symbols[market])} (낡음 {stale})")
    print(f"{day}: " + " ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
