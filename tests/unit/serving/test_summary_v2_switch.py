"""coordinator 설정 `summary_v2_enabled`(첫 화면 v2): 없으면 꺼짐, 있으면 bool, publisher 입력에는
켰을 때만 키를 넣습니다. `macro_context_enabled`와 같은 모양입니다."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import test_daily_multiday_publish as mp

from modeler.serving import daily_coordinator as dc

e2e = mp.e2e
D1 = mp.D1
KEY = "summary_v2_enabled"
BEFORE = "b7c90c4"  # 첫 화면 v2 전 main


def _setup(tmp_path: Path) -> tuple[Path, dict]:
    _release, config_path = e2e._setup(tmp_path)
    mp._enable_publisher(tmp_path, config_path)
    return config_path, json.loads(config_path.read_text())


def test_config_switch_is_off_when_absent_and_a_bool_when_present(tmp_path: Path) -> None:
    config_path, body = _setup(tmp_path)
    body.pop(KEY, None)
    config_path.write_text(json.dumps(body))
    assert KEY not in dc._config(config_path)  # 이 키가 없는 ops.json도 그대로 돕니다
    for bad in ("yes", 1, None):
        config_path.write_text(json.dumps({**body, KEY: bad}))
        with pytest.raises(ValueError, match="summary_v2_enabled must be true or false"):
            dc._config(config_path)
    config_path.write_text(json.dumps({**body, KEY: True}))
    assert dc._config(config_path)[KEY] is True


def test_publisher_input_carries_the_switch_only_when_on(tmp_path: Path) -> None:
    _config_path, base = _setup(tmp_path)
    render = {"report_sha256": "a" * 64, "invocation_id": "i"}
    built = dc._reports_publisher_input({**base, KEY: True}, D1, render, tmp_path / "run")
    assert built[KEY] is True
    for off in ({**base, KEY: False}, {k: v for k, v in base.items() if k != KEY}):
        assert KEY not in dc._reports_publisher_input(off, D1, render, tmp_path / "run")


def _publisher_before(tmp_path: Path):
    done = subprocess.run(
        ["git", "-C", str(e2e.PROJECT), "show", f"{BEFORE}:deploy/reports/publish_reports.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        pytest.skip(f"{BEFORE} is not in this checkout")
    path = tmp_path / "publish_reports_before_v2.py"
    path.write_text(done.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("publish_reports_before_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publisher_before_v2_accepts_the_input_while_the_switch_is_off(tmp_path: Path) -> None:
    old = _publisher_before(tmp_path)
    _config_path, base = _setup(tmp_path)
    render = {"report_sha256": "a" * 64, "invocation_id": "inv"}
    for off in ({**base, KEY: False}, {k: v for k, v in base.items() if k != KEY}):
        built = dc._reports_publisher_input(off, D1, render, tmp_path / "run")
        old.check_config(built, need_local=True, strict_remote=False)  # does not raise
    on = dc._reports_publisher_input({**base, KEY: True}, D1, render, tmp_path / "run")
    with pytest.raises(old.PublishError, match="unknown config keys"):
        old.check_config(on, need_local=True, strict_remote=False)  # needs the new publisher
