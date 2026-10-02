"""Check the serving interpreter and model dependencies against a frozen release."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

PACKAGES = ("numpy", "scipy", "pandas", "polars", "pyarrow", "scikit-learn",
            "lightgbm", "joblib", "duckdb")
PROBE = (
    "import importlib.metadata as m, json, platform; "
    "print(json.dumps({'python':platform.python_version(), "
    "'implementation':platform.python_implementation(), "
    "'system':platform.system(), 'machine':platform.machine(), "
    "'packages':{p:m.version(p) for p in " + repr(PACKAGES) + "}}, sort_keys=True))"
)


def verify_runtime(python: Path, manifest_path: Path) -> dict[str, Any]:
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "daily-briefing-runtime.v1":
        raise ValueError("frozen runtime manifest version mismatch")
    expected = {key: contract[key] for key in ("python", "implementation", "system", "machine", "packages")}
    if set(expected["packages"]) != set(PACKAGES):
        raise ValueError("runtime manifest does not pin all model dependencies")
    try:
        probe = subprocess.run([str(python), "-c", PROBE], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("serving interpreter probe failed") from None
    if probe.returncode != 0:
        raise ValueError("serving interpreter probe failed")
    actual = json.loads(probe.stdout)
    if actual != expected:
        raise ValueError("serving interpreter or package versions differ from frozen release")
    return actual
