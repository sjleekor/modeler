"""순위 종목의 증권 이름 입력(`security-names.v1`)을 레이크에서 만듭니다.

R0 임시 도구 `build_names.py`(2026-10-06)의 규칙을 그대로 옮긴 것입니다. 자동 게시 경로가
`markdown.build_context(security_names=)`에 이름을 넘기지 못해 US 순위 표의 `종류` 열이 빠지던 결함
(R6 절차서 D1)을 고칩니다. 규칙을 바꾸면 R0 결과와 어긋나므로 바꾸지 않습니다.

  US  레이크 `listing_snapshots.security_name`. as_of <= D 중 가장 최근 행을 씁니다. 그 행의 갈래
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

from modeler.reporting.markdown import SECURITY_NAMES_SCHEMA

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


def us_names(listing: Path, symbols: set[str], day: dt.date) -> dict:
    """US 이름 블록 {"source", "basis", "symbols"}. `listing`은 parquet 파일이나 디렉터리입니다."""
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
    source = "US 레이크 listing_snapshots.security_name" + _snapshot_label(listing)
    return {"source": source, "basis": basis, "symbols": out}


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


def build_security_names(
    envelope: dict, *, us_listing: Path | None = None, kr_stock_master: Path | None = None
) -> dict:
    """`security-names.v1` 본문을 만듭니다. 원천을 준 시장만 블록이 생깁니다."""
    day = _report_date(envelope)
    symbols = envelope_symbols(envelope)
    markets = {}
    if us_listing is not None:
        markets["US"] = us_names(us_listing, symbols["US"], day)
    if kr_stock_master is not None:
        markets["KR"] = kr_names(kr_stock_master, symbols["KR"])
    return {"schema": SECURITY_NAMES_SCHEMA, "report_date": day.isoformat(), "markets": markets}


def dumps(data: dict) -> bytes:
    """R0 도구와 같은 형식(정렬, 들여쓰기 1, 끝 줄바꿈)의 UTF-8 바이트입니다."""
    for market, block in data["markets"].items():
        for field_name in ("source", "basis"):
            if _FORBIDDEN.search(block[field_name]):
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
    for market, root, label in (("US", us_root, "US 이름 원천"), ("KR", kr_root, "KR 이름 원천")):
        if root is None:
            continue
        try:
            if market == "US":
                block = us_names(find_us_listing(root), symbols["US"], day)
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
            {"schema": SECURITY_NAMES_SCHEMA, "report_date": day.isoformat(), "markets": markets}
        )
    except NamesError as exc:
        return Built(None, {m: str(exc) for m in markets} | notes, {})
    return Built(raw, notes, counts)


# ---------------------------------------------------------------------------
# CLI (수동 단위·정정용)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--us-listing", type=Path, help="listing_snapshots parquet(파일·디렉터리)")
    parser.add_argument("--kr-stock-master", type=Path, help="stock_master parquet(파일·디렉터리)")
    parser.add_argument("--us-root", type=Path, help="<stock_data>/us (가장 최근 listing을 찾음)")
    parser.add_argument("--kr-root", type=Path, help="<stock_data>/kr (D 이전 snapshot을 찾음)")
    args = parser.parse_args(argv)
    try:
        envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise NamesError("envelope가 JSON 객체가 아님")
        day = _report_date(envelope)
        us_listing = args.us_listing or (find_us_listing(args.us_root) if args.us_root else None)
        kr_master = args.kr_stock_master or (
            find_kr_stock_master(args.kr_root, day) if args.kr_root else None
        )
        if us_listing is None and kr_master is None:
            raise NamesError("--us-listing·--us-root·--kr-stock-master·--kr-root 중 하나가 필요함")
        data = build_security_names(envelope, us_listing=us_listing, kr_stock_master=kr_master)
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
