"""MS1 동결 run을 서빙 bundle로 묶는 계약. 표준 라이브러리만 쓴다.

일일 서빙은 모델을 다시 학습하지 않는다. 동결 run
(`stock_data/{us,kr}/output/market_sector/<run_id>/`)의 고정 fit 모델·OOF reference·run manifest를
한 디렉터리로 복사하고, 파일마다 sha256을 `bundle.json`에 적는다. 이 모듈은 그 디렉터리를
**읽고 검증하는 쪽**이다. 만드는 쪽(`score_daily.build_bundle`)은 달력 세션을 계산해야 해서
polars가 필요하므로 거기에 있다.

select 단계(`serving.daily_inputs`)와 release 빌드(`serving.release_build`)도 이 검증을 쓰므로,
polars·sklearn을 끌어오지 않게 이 파일을 따로 두었다.

bundle.json 구조 (``market-sector-bundle.v1``)::

    schema_version, frozen_tag, config_hash, asset_registry{version, hash}, verdicts,
    files{상대경로: sha256},
    markets{US|KR: run_id, run_manifest, oof, latest_scores, models{이름: 상대경로},
            calendar{...}, panel{...}, frozen_input_pins{...}, assets[...]}
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "market-sector-bundle.v1"
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
        models = block.get("models")
        if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
            raise BundleError(f"{market}: 모델 목록이 {list(MODEL_NAMES)}와 다릅니다")
        refs += list(models.values())
        for rel in refs:
            if rel not in files:
                raise BundleError(f"{market}: bundle files에 없는 파일을 가리킵니다: {rel}")
    return {**manifest, "_sha256": manifest_sha}
