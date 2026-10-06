"""데이터셋 쓰기 — parquet + ``manifest.json``.

**바이트 해시가 아니라 내용 해시를 쓴다.** parquet는 쓰기 시각·라이브러리 버전·행
그룹으로 바이트가 달라지므로, 지우고 다시 만들었을 때 "같은 것을 다시 만들었나"를
확인하려면 파일 바이트가 아니라 정렬된 내용을 봐야 한다 (수집 계획
``08_plan_review.md`` V19, 모델링 계획 ``05_validation_protocol.md`` §6).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from modeler.etl.config import DataRoot


class DirtyWorktreeError(RuntimeError):
    """커밋 안 된 변경으로 데이터셋을 만들려고 했다."""


class DatasetExistsError(FileExistsError):
    """같은 이름의 데이터셋이 이미 있다 — 덮어쓰지 않는다 (유니버스 v2 설계 §3, T9)."""


class DatasetNamingError(ValueError):
    """유니버스 버전·종목 구간 옵션과 데이터셋 이름이 규칙에 안 맞는다."""


#: 동결된 데이터셋 이름. 이 이름이 이미 있는 루트에서는 ``write_dataset``이 쓰기를 거부한다
#: (아무 이름이나 있으면 거부하는 규칙의 일부이고, 시험이 이 목록을 명시해 확인한다).
FROZEN_DATASET_NAMES: frozenset[str] = frozenset(
    {
        "us_panel_v1",
        "us_panel_v2",
        "us_features_v1",
        "us_features_v2",
        "us_features_flow_v1",
        "us_labels_v1",
        "us_labels_v2",
        "us_labels_h5_v1",
        "us_labels_h63_v1",
        "us_labels_h63_v2",
    }
)

#: 유니버스 v2로 만든 데이터셋 이름에 들어가야 하는 토큰(``_u2``). v1에는 없어야 한다.
UNIVERSE_V2_TOKEN = "u2"

#: 4차 전진 등록(§10)의 보조 판정 데이터셋 이름에 들어가는 토큰. 이 이름에는 종목 구간
#: 모드를 못 쓴다 — 보조 판정은 멤버십만 v2이고 피쳐·라벨은 동결 코드(꺼짐)여야 한다.
FORWARD_TOKEN = "_fwd"


def check_dataset_name(
    name: str, *, universe_version: str, security_boundaries: bool = False
) -> None:
    """유니버스 버전·종목 구간 옵션이 데이터셋 이름과 맞는지 본다. 어긋나면 ``DatasetNamingError``.

    * v2 데이터셋 이름에는 ``_u2``가 있어야 하고, v1에는 없어야 한다.
    * 종목 구간 모드(``security_boundaries``)는 v2에서만 쓰고, 이름에 ``_fwd``가 있으면 거부한다.
    """
    if universe_version not in ("v1", "v2"):
        raise DatasetNamingError(f"모르는 유니버스 버전입니다: {universe_version!r}")
    has_u2 = UNIVERSE_V2_TOKEN in name.split("_")
    if universe_version == "v2" and not has_u2:
        raise DatasetNamingError(
            f"유니버스 v2 데이터셋 이름에는 _{UNIVERSE_V2_TOKEN} 토큰이 있어야 합니다: {name!r}"
        )
    if universe_version == "v1" and has_u2:
        raise DatasetNamingError(
            f"유니버스 v1 데이터셋 이름에는 _{UNIVERSE_V2_TOKEN} 토큰이 없어야 합니다: {name!r}"
        )
    if security_boundaries:
        if universe_version != "v2":
            raise DatasetNamingError("종목 구간 모드는 유니버스 v2에서만 켤 수 있습니다.")
        if FORWARD_TOKEN in name:
            raise DatasetNamingError(
                f"{FORWARD_TOKEN} 데이터셋은 동결 코드(종목 구간 모드 꺼짐)로 만들어야 합니다: "
                f"{name!r} (전진 등록 §10)"
            )


def git_commit(repo: Path, *, allow_dirty: bool = False) -> str:
    """``repo``의 HEAD 커밋. **더러우면 거부한다.**

    manifest의 ``modeler_git_commit``은 "이 커밋으로 이 명령을 돌리면 같은
    ``content_hash``가 나온다"는 약속이다. 더러운 트리로 만들면 **그 약속이
    어느 커밋으로도 지켜지지 않는다.**

    실제로 그렇게 됐다. ``us_features_v1``의 manifest가
    ``d38d1d44…-dirty``이고, 지금 코드로 ``sp_ttm``을 다시 계산하면
    168,161행 중 17행이 다르다 (2026-09-21 확인 · 미국 2차 후속 ``02`` §6).

    ``allow_dirty``는 **버리는 실험용**이다. 남길 데이터셋에는 쓰지 않는다.
    """
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not dirty:
        return head
    if not allow_dirty:
        raise DirtyWorktreeError(
            f"{repo} 에 커밋 안 된 변경이 있다. 데이터셋을 만들면 manifest 의 "
            f"커밋({head[:12]})으로 다시 만들 수 없다.\n"
            f"먼저 커밋하거나, 버리는 실험이면 --allow-dirty 를 준다.\n"
            f"{dirty[:500]}"
        )
    return f"{head}-dirty"


def content_hash(df: pl.DataFrame) -> str:
    """키로 정렬한 뒤 컬럼 값의 해시. **parquet 바이트 해시가 아니다.**

    모든 컬럼으로 정렬해 물리적 행 순서와 무관한 정본 순서를 만든 다음,
    행 단위 해시(``hash_rows``, 고정 시드)를 이어붙여 sha256을 낸다. 같은 내용을
    다시 만들면(파일을 지우고 다시 써도) 같은 해시가 나온다 — 파일을 만든 시각이나
    parquet 행 그룹 배치는 이 해시에 영향을 주지 않는다.
    """
    canonical = df.sort(by=df.columns)
    row_hashes = canonical.hash_rows(seed=0)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def assert_dataset_absent(root: DataRoot, name: str) -> None:
    """``root.datasets / name``이 이미 있으면 ``DatasetExistsError``. 빌더는 계산 전에 부른다."""
    dataset_dir = root.datasets / name
    if dataset_dir.exists():
        frozen = " (동결 데이터셋)" if name in FROZEN_DATASET_NAMES else ""
        raise DatasetExistsError(
            f"{dataset_dir} 가 이미 있습니다{frozen} — 덮어쓰지 않습니다. "
            "다른 이름으로 만들거나, 지워도 되는 것이면 사람이 직접 지우십시오."
        )


def write_dataset(df: pl.DataFrame, root: DataRoot, name: str, *, manifest: dict[str, Any]) -> Path:
    """``root.datasets / name``에 ``part.parquet``과 ``manifest.json``을 쓴다.

    ``manifest``는 호출자가 채운 것(입력 표 ``snapshot_date``, collector·modeler
    커밋 해시, 유니버스 필터 설명 등)에 ``row_count``·``content_hash``·
    ``created_at``을 얹어서 쓴다 — 이 셋은 ``df`` 하나로 정해지는 값이라 호출자가
    따로 계산해 넘기면 ``df``와 어긋날 위험이 있어 여기서 직접 채운다.

    **같은 이름의 디렉터리가 이미 있으면 쓰기를 거부한다**(``DatasetExistsError``) — 동결
    데이터셋(``FROZEN_DATASET_NAMES``)을 덮어쓰지 않으려는 규칙이다 (유니버스 v2 설계 §3, T9).
    우회 플래그는 없다. 다시 만들어야 하면 사람이 디렉터리를 지우거나 다른 이름을 쓴다.
    """
    assert_dataset_absent(root, name)
    dataset_dir = root.datasets / name
    dataset_dir.mkdir(parents=True, exist_ok=False)

    parquet_path = dataset_dir / "part.parquet"
    df.write_parquet(parquet_path)

    full_manifest = {
        **manifest,
        "row_count": df.height,
        "content_hash": content_hash(df),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(full_manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )
    return dataset_dir


def read_panel_dataset(root: DataRoot, name: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    """저장된 패널 데이터셋(``part.parquet``)과 manifest를 읽는다.

    manifest에 ``universe_version``이 없으면(동결 패널 ``us_panel_v1``·``us_panel_v2``) ``v1``이다.
    """
    directory = root.datasets / name
    part = directory / "part.parquet"
    manifest_path = directory / "manifest.json"
    if not part.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{name} 데이터셋이 없습니다: {directory}")
    manifest = json.loads(manifest_path.read_text())
    return pl.read_parquet(part), manifest


def manifest_universe_version(manifest: dict[str, Any]) -> str:
    """패널 manifest가 적은 유니버스 버전. 없으면 ``v1``(v2 이전에 만든 동결 데이터셋)."""
    return str(manifest.get("universe_version", "v1"))
