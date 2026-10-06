"""빌더 ``main()``의 데이터셋 이름·덮어쓰기 규칙 (유니버스 v2 설계 §3, T9).

규칙 위반은 **레이크를 읽기 전에** 걸러야 한다 — 수 분짜리 계산 뒤에야 거부하면 안 된다.
``STOCK_DATA_ROOT``는 ``tests/conftest.py``가 ``tmp_path``로 돌려 둔다(``<tmp>/us/datasets``).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from modeler.us import build_features, build_flow_features, build_labels, build_panel
from modeler.us.dataset import DatasetExistsError, DatasetNamingError


def _datasets(tmp_path: Path) -> Path:
    return tmp_path / "us" / "datasets"


def _write_panel(tmp_path: Path, name: str, **manifest: object) -> None:
    directory = _datasets(tmp_path) / name
    directory.mkdir(parents=True)
    pl.DataFrame(
        {"date": [], "symbol": []}, schema={"date": pl.Date, "symbol": pl.String}
    ).write_parquet(directory / "part.parquet")
    (directory / "manifest.json").write_text(json.dumps({"content_hash": "h", **manifest}))


def test_build_panel_v2_needs_a_u2_name(tmp_path: Path) -> None:
    with pytest.raises(DatasetNamingError, match="_u2"):
        build_panel.main(["--universe-version", "v2", "--name", "us_panel_v3"])


def test_build_panel_v2_requires_an_explicit_name(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_panel.main(["--universe-version", "v2"])


def test_build_panel_v1_rejects_a_u2_name(tmp_path: Path) -> None:
    with pytest.raises(DatasetNamingError):
        build_panel.main(["--name", "us_panel_v1_u2"])


def test_build_panel_refuses_an_existing_frozen_name_before_reading_the_lake(
    tmp_path: Path,
) -> None:
    _write_panel(tmp_path, "us_panel_v1")
    with pytest.raises(DatasetExistsError, match="동결 데이터셋"):
        build_panel.main([])  # 기본 이름 us_panel_v1 — 레이크가 비어 있어도 거기까지 가지 않는다


@pytest.mark.parametrize("module", [build_features, build_labels])
def test_feature_and_label_builders_follow_the_input_panel_manifest(
    tmp_path: Path, module: object
) -> None:
    """저장된 패널의 manifest가 v2면 이름에 _u2가 있어야 한다."""
    _write_panel(tmp_path, "us_panel_v3_u2", universe_version="v2")
    with pytest.raises(DatasetNamingError, match="_u2"):
        module.main(["--panel-name", "us_panel_v3_u2", "--name", "out_v3"])
    # v1 패널(manifest에 버전이 없다)로 만들 때는 _u2가 없어야 한다.
    _write_panel(tmp_path, "us_panel_old")
    with pytest.raises(DatasetNamingError):
        module.main(["--panel-name", "us_panel_old", "--name", "out_v3_u2"])


@pytest.mark.parametrize("module", [build_features, build_labels])
def test_security_boundaries_rejects_v1_panel_and_fwd_names(tmp_path: Path, module: object) -> None:
    _write_panel(tmp_path, "us_panel_v3_u2", universe_version="v2")
    with pytest.raises(DatasetNamingError, match="_fwd"):
        module.main(
            [
                "--panel-name",
                "us_panel_v3_u2",
                "--name",
                "out_fwd_u2",
                "--security-boundaries",
            ]
        )
    _write_panel(tmp_path, "us_panel_old")
    with pytest.raises(DatasetNamingError, match="v2에서만"):
        module.main(["--panel-name", "us_panel_old", "--name", "out_v3", "--security-boundaries"])


@pytest.mark.parametrize("module", [build_features, build_labels])
def test_feature_and_label_builders_refuse_an_existing_output(
    tmp_path: Path, module: object
) -> None:
    _write_panel(tmp_path, "us_panel_v3_u2", universe_version="v2")
    _write_panel(tmp_path, "already_u2", universe_version="v2")
    with pytest.raises(DatasetExistsError):
        module.main(["--panel-name", "us_panel_v3_u2", "--name", "already_u2"])


def test_flow_builder_reads_the_source_panel_manifest_for_the_name_rule(tmp_path: Path) -> None:
    _write_panel(tmp_path, "us_panel_v3_u2", universe_version="v2")
    with pytest.raises(DatasetNamingError, match="_u2"):
        build_flow_features.main(["--source-panel", "us_panel_v3_u2", "--name", "flow_v3"])
    with pytest.raises(DatasetNamingError, match="_fwd"):
        build_flow_features.main(
            [
                "--source-panel",
                "us_panel_v3_u2",
                "--name",
                "us_features_flow_v1_fwd_u2",
                "--security-boundaries",
            ]
        )
    # v2 패널은 기본 이름(us_features_flow_v1)을 쓸 수 없다 — 이름을 직접 줘야 한다.
    with pytest.raises(SystemExit):
        build_flow_features.main(["--source-panel", "us_panel_v3_u2"])


def test_flow_builder_existing_output_returns_1_and_does_not_write(tmp_path: Path, capsys) -> None:
    _write_panel(tmp_path, "us_panel_v2")  # 동결 입력 패널(버전 키 없음 = v1)
    _write_panel(tmp_path, "us_features_flow_v1")
    assert build_flow_features.main([]) == 1
    assert "이미 있습니다" in capsys.readouterr().err
