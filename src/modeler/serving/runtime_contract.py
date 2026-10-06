"""Check the serving interpreter and model dependencies against a frozen release."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

PACKAGES = ("numpy", "scipy", "pandas", "polars", "pyarrow", "scikit-learn",
            "lightgbm", "joblib", "duckdb")
# The market-sector section needs no extra package: its calculation calendar (MS1's KR panel was
# built with ``exchange_calendars`` XKRX) is a frozen file inside the market-sector bundle, made on
# the Mac, so the serving venv does not carry ``exchange_calendars``.


def probe_code(packages: tuple[str, ...] = PACKAGES) -> str:
    return (
        "import importlib.metadata as m, json, platform; "
        "print(json.dumps({'python':platform.python_version(), "
        "'implementation':platform.python_implementation(), "
        "'system':platform.system(), 'machine':platform.machine(), "
        "'packages':{p:m.version(p) for p in " + repr(tuple(packages)) + "}}, sort_keys=True))"
    )


PROBE = probe_code(PACKAGES)


def verify_runtime(python: Path, manifest_path: Path) -> dict[str, Any]:
    """Compare the interpreter with the frozen runtime manifest.

    The manifest must pin ``PACKAGES``.  It may pin more; every package it lists is checked, so a
    manifest that lists a package the venv lacks (for example ``exchange_calendars``) is refused.
    """
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "daily-briefing-runtime.v1":
        raise ValueError("frozen runtime manifest version mismatch")
    expected = {key: contract[key] for key in ("python", "implementation", "system", "machine", "packages")}
    if not set(PACKAGES) <= set(expected["packages"]):
        raise ValueError("runtime manifest does not pin all model dependencies")
    try:
        probe = subprocess.run([str(python), "-c", probe_code(tuple(sorted(expected["packages"])))],
                               stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("serving interpreter probe failed") from None
    if probe.returncode != 0:
        raise ValueError("serving interpreter probe failed")
    actual = json.loads(probe.stdout)
    if actual != expected:
        raise ValueError("serving interpreter or package versions differ from frozen release")
    return actual
