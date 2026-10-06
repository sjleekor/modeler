"""MS1 동결 run을 서빙 bundle로 묶는 계약. 표준 라이브러리만 쓴다.

일일 서빙은 모델을 다시 학습하지 않는다. 동결 run
(`stock_data/{us,kr}/output/market_sector/<run_id>/`)의 고정 fit 모델·OOF reference·run manifest와
**계산 달력 세션 파일**(``<시장>/calendar.json``)을 한 디렉터리로 복사하고, 파일마다 sha256을
`bundle.json`에 적는다. 이 모듈은 그 디렉터리를 **읽고 검증하는 쪽**이다. 만드는 쪽
(`score_daily.build_bundle`)은 달력 세션을 계산해야 해서 polars와 `exchange_calendars`가 필요하므로
거기에 있다. 채점은 `exchange_calendars` 없이 이 달력 파일만 쓴다(운영 venv에 그 패키지가 없다).

select 단계(`serving.daily_inputs`)와 release 빌드(`serving.release_build`)도 이 검증을 쓰므로,
polars·sklearn을 끌어오지 않게 이 파일을 따로 두었다.

bundle.json 구조 (``market-sector-bundle.v2``)::

    schema_version, calendar_range_note, frozen_tag, config_hash, asset_registry{version, hash},
    verdicts, files{상대경로: sha256},
    markets{US|KR: run_id, run_manifest, oof, latest_scores, models{이름: 상대경로},
            calendar{..., file, range_end, file_sessions_sha256}, panel{...},
            frozen_input_pins{...}, assets[...]}

달력 파일 구조 (``market-sector-calendar.v1``)::

    schema_version, market, calendar_id, calendar_basis, generated_by, method, segments[...],
    range_start, range_end, first_session, last_session, n_sessions, sessions_sha256,
    valid_through_note, columns[session, open_utc, close_utc], sessions[[날짜, 개장, 폐장], ...]

``sessions_sha256``은 세션 날짜(ISO)를 줄바꿈으로 이어 붙인 문자열의 sha256이다. 개장·폐장 시각까지
포함한 파일 전체의 sha256은 bundle.json의 ``files``가 고정한다. ``range_end`` 뒤의 결정일은
채점이 거부한다(``calendar_range_exhausted``). 그 뒤로 가려면 bundle을 다시 만들어야 한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

SCHEMA = "market-sector-bundle.v2"
CALENDAR_SCHEMA = "market-sector-calendar.v1"
CALENDAR_COLUMNS = ["session", "open_utc", "close_utc"]
BUNDLE_FILE = "bundle.json"
FROZEN_TAG = "ms-prereg-frozen"
MARKETS = ("US", "KR")
#: 일일 채점에 쓰는 live 모델. LightGBM·섹터 상대 선택(p_mkt_*)은 일일 표에 넣지 않는다.
MODEL_NAMES = ("p_opp_ridge", "p_stab_logit", "b_stab_logit_rvol")
HEX = re.compile(r"[0-9a-f]{64}\Z")


class BundleError(ValueError):
    """bundle.json이 없거나, 파일이 바뀌었거나, 형식이 맞지 않는다."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sessions_sha256(iso_sessions: Iterable[str]) -> str:
    """세션 날짜(ISO 문자열) 목록의 지문: 줄바꿈으로 이어 붙인 문자열의 sha256."""
    return hashlib.sha256("\n".join(iso_sessions).encode()).hexdigest()


def _safe_rel(rel: str) -> str:
    parts = Path(rel).parts
    if not rel or rel.startswith("/") or ".." in parts or rel == BUNDLE_FILE:
        raise BundleError(f"bundle 파일 경로가 올바르지 않습니다: {rel!r}")
    return rel


def read_manifest(bundle_json: Path) -> dict[str, Any]:
    if bundle_json.is_symlink() or not bundle_json.is_file():
        raise BundleError("bundle.json이 없거나 심볼릭 링크입니다")
    try:
        manifest = json.loads(bundle_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError("bundle.json을 읽을 수 없습니다") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA:
        raise BundleError("bundle.json 스키마가 맞지 않습니다")
    return manifest


def read_calendar_file(path: Path) -> dict[str, Any]:
    """계산 달력 파일을 읽고 형식과 내부 지문을 확인한다(표준 라이브러리만).

    확인하는 것: 스키마, 세션 날짜가 오름차순·중복 없음, 개장·폐장 시각 형식, ``n_sessions``,
    ``sessions_sha256``, 첫·마지막 세션, ``range_end``가 마지막 세션 이상. 파일 전체의 sha256은
    bundle.json이 고정하므로 여기서 보지 않는다.
    """
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"계산 달력 파일이 없거나 심볼릭 링크입니다: {path.name}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"계산 달력 파일을 읽을 수 없습니다: {path.name}") from exc
    if not isinstance(body, dict) or body.get("schema_version") != CALENDAR_SCHEMA:
        raise BundleError("계산 달력 파일 스키마가 맞지 않습니다")
    if body.get("columns") != CALENDAR_COLUMNS:
        raise BundleError("계산 달력 파일 열 구성이 맞지 않습니다")
    rows = body.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise BundleError("계산 달력 파일에 세션이 없습니다")
    previous: date | None = None
    iso: list[str] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or not all(isinstance(x, str) for x in row):
            raise BundleError("계산 달력 파일 행 형식이 맞지 않습니다")
        try:
            day = date.fromisoformat(row[0])
            opened, closed = datetime.fromisoformat(row[1]), datetime.fromisoformat(row[2])
        except ValueError as exc:
            raise BundleError("계산 달력 파일 날짜·시각 형식이 맞지 않습니다") from exc
        if opened.tzinfo is None or closed.tzinfo is None or closed <= opened:
            raise BundleError(f"계산 달력 파일 개장·폐장 시각이 올바르지 않습니다: {row[0]}")
        if previous is not None and day <= previous:
            raise BundleError("계산 달력 파일 세션이 오름차순·중복 없음이 아닙니다")
        previous = day
        iso.append(row[0])
    try:
        range_end = date.fromisoformat(body["range_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError("계산 달력 파일에 range_end가 없습니다") from exc
    if (body.get("n_sessions") != len(iso) or body.get("sessions_sha256") != sessions_sha256(iso)
            or body.get("first_session") != iso[0] or body.get("last_session") != iso[-1]):
        raise BundleError("계산 달력 파일 머리말이 세션 목록과 다릅니다")
    if range_end < date.fromisoformat(iso[-1]):
        raise BundleError("계산 달력 파일 range_end가 마지막 세션보다 앞입니다")
    return body


def _check_calendar(directory: Path, market: str, spec: dict[str, Any]) -> None:
    """시장 하나의 달력 파일이 bundle.json 안의 달력 지문과 같은지 확인한다."""
    body = read_calendar_file(directory / _safe_rel(spec["file"]))
    same = (body.get("market") == market and body.get("calendar_id") == spec.get("calendar_id")
            and body.get("calendar_basis") == spec.get("basis")
            and body.get("range_end") == spec.get("range_end")
            and body.get("n_sessions") == spec.get("file_n_sessions")
            and body.get("sessions_sha256") == spec.get("file_sessions_sha256"))
    if not same:
        raise BundleError(f"{market}: 계산 달력 파일이 bundle.json의 달력 항목과 다릅니다")
    # 동결 구간(동결 패널의 첫~마지막 세션)의 세션 목록이 동결 때와 같아야 한다.
    lo, hi = spec.get("first_session"), spec.get("last_session")
    inside = [row[0] for row in body["sessions"] if isinstance(lo, str) and lo <= row[0] <= hi]
    if len(inside) != spec.get("n_sessions") or sessions_sha256(inside) != spec.get(
            "sessions_sha256"):
        raise BundleError(f"{market}: 계산 달력 파일의 동결 구간 세션이 동결 때와 다릅니다")


def verify_bundle_dir(directory: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """파일 목록과 sha256이 bundle.json과 같은지 확인하고 manifest를 돌려준다(``_sha256`` 포함).

    bundle.json에 없는 파일이 디렉터리에 있어도, 적힌 파일이 없거나 바뀌어도 거부한다.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise BundleError("bundle 디렉터리가 없거나 심볼릭 링크입니다")
    bundle_json = directory / BUNDLE_FILE
    manifest = read_manifest(bundle_json)
    manifest_sha = sha256_file(bundle_json)
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise BundleError("bundle.json sha256이 고정된 값과 다릅니다")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError("bundle.json에 files 목록이 없습니다")
    for rel, digest in files.items():
        _safe_rel(rel)
        if not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise BundleError(f"bundle 파일 sha256 형식이 맞지 않습니다: {rel}")
        path = directory / rel
        if path.is_symlink() or not path.is_file():
            raise BundleError(f"bundle 파일이 없거나 심볼릭 링크입니다: {rel}")
        if sha256_file(path) != digest:
            raise BundleError(f"bundle 파일이 바뀌었습니다: {rel}")
    present = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    extra = sorted(present - set(files) - {BUNDLE_FILE})
    if extra:
        raise BundleError(f"bundle.json에 없는 파일이 있습니다: {extra[0]}")
    markets = manifest.get("markets")
    if not isinstance(markets, dict) or set(markets) != set(MARKETS):
        raise BundleError("bundle.json에 US·KR 두 시장이 다 있어야 합니다")
    for market, block in markets.items():
        refs = [block.get("run_manifest"), block.get("oof"), block.get("latest_scores")]
        calendar = block.get("calendar")
        if not isinstance(calendar, dict) or not isinstance(calendar.get("file"), str):
            raise BundleError(f"{market}: bundle.json에 계산 달력 파일 항목이 없습니다")
        refs.append(calendar["file"])
        models = block.get("models")
        if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
            raise BundleError(f"{market}: 모델 목록이 {list(MODEL_NAMES)}와 다릅니다")
        refs += list(models.values())
        for rel in refs:
            if rel not in files:
                raise BundleError(f"{market}: bundle files에 없는 파일을 가리킵니다: {rel}")
        _check_calendar(directory, market, calendar)
    return {**manifest, "_sha256": manifest_sha}
