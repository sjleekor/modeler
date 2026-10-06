"""Check the serving interpreter and model dependencies against a frozen release."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

PACKAGES = ("numpy", "scipy", "pandas", "polars", "pyarrow", "scikit-learn",
            "lightgbm", "joblib", "duckdb")
#: Extra packages a release needs when the market-sector section is on.  MS1's KR panel was built
#: with ``exchange_calendars`` XKRX, and the scoring refuses any other calendar (there is no
#: fallback), so the venv must carry the pinned version and the runtime manifest must list it.
MARKET_SECTOR_PACKAGES = ("exchange_calendars",)


def probe_code(packages: tuple[str, ...] = PACKAGES) -> str:
    return (
        "import importlib.metadata as m, json, platform; "
        "print(json.dumps({'python':platform.python_version(), "
        "'implementation':platform.python_implementation(), "
        "'system':platform.system(), 'machine':platform.machine(), "
        "'packages':{p:m.version(p) for p in " + repr(tuple(packages)) + "}}, sort_keys=True))"
    )


PROBE = probe_code(PACKAGES)


def verify_runtime(python: Path, manifest_path: Path, *,
                   extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Compare the interpreter with the frozen runtime manifest.

    ``extra_packages`` must be pinned in addition to ``PACKAGES`` (``MARKET_SECTOR_PACKAGES`` when
    the section is on).  The manifest may pin more than that (a manifest made for the section keeps
    working with a config that has it off); every package it lists is checked.
    """
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "daily-briefing-runtime.v1":
        raise ValueError("frozen runtime manifest version mismatch")
    expected = {key: contract[key] for key in ("python", "implementation", "system", "machine", "packages")}
    wanted = (*PACKAGES, *extra_packages)
    if not set(wanted) <= set(expected["packages"]):
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
