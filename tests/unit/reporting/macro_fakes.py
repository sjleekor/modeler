"""Stand-ins for the macro-context calculation and renderer modules (M2-a, M2-b).

Those modules are built on other branches.  The wiring (stage, renderer hook, publisher) only uses
their fixed interface, so the tests here run against these small fakes:

* ``modeler.scores.macro_context``: ``SCHEMA``, ``build_document``, ``failure_document``,
  ``document_bytes``;
* ``modeler.reporting.macro``: ``render(body) -> (markdown, {"macro-<name>.svg": svg})``.

``install(monkeypatch)`` puts both into ``sys.modules`` for in-process code;
``RELEASE_SOURCES`` writes them into a synthetic release's ``src`` for child processes.
"""

from __future__ import annotations

import json
import sys
import types

SCHEMA = "briefing-macro.v1"
MS_SELECTION_SCHEMA = "market-sector-selection.v1"
CALC_MODULE = "modeler.scores.macro_context"
RENDER_MODULE = "modeler.reporting.macro"

CALC_SOURCE = '''"""Fake macro-context calculation (tests only)."""
import json

SCHEMA = "briefing-macro.v1"


def _identity(unit, invocation_id, release, selection_sha256):
    return {"schema": SCHEMA, "unit": unit, "invocation_id": invocation_id, "release": release,
            "selection_sha256": selection_sha256}


def build_document(selection, *, unit, invocation_id, release, selection_sha256):
    # Like the real loader: only the market-sector selection document of the same D is accepted.
    if selection.get("schema_version") != "market-sector-selection.v1":
        raise ValueError("not a market-sector selection document")
    if selection.get("report_date") != unit or not selection.get("input_cutoff"):
        raise ValueError("selection is for another day")
    return {**_identity(unit, invocation_id, release, selection_sha256), "state": "ok",
            "seen_selection_schema": selection["schema_version"],
            "header": {"daily_asof_max": unit},
            "indicators": [{"id": "rate_us10y", "label": "US 10Y", "change_1m": -0.12}]}


def failure_document(*, unit, invocation_id, release, selection_sha256, reason):
    return {**_identity(unit, invocation_id, release, selection_sha256), "state": "failed",
            "reason": reason}


def document_bytes(body):
    return (json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\\n").encode()
'''

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="120" viewBox="0 0 320 120">'
    '<title>US 10Y</title><polyline points="0,60 160,40 320,80" fill="none" stroke="#334"/>'
    "</svg>\n"
)

RENDER_SOURCE = f'''"""Fake macro renderer (tests only)."""
SVG = {SVG!r}


def render(body):
    unit = body.get("unit")
    if body.get("state") == "failed":
        reason = body.get("reason")
        return f"# 거시 흐름 — {{unit}}\\n\\n거시 흐름을 내지 못했습니다: {{reason}}.\\n", {{}}
    rows = "\\n".join(f"| {{i['label']}} | {{i['change_1m']:+.2f}} |" for i in body["indicators"])
    text = (f"# 거시 흐름 — {{unit}}\\n\\n표시 전용입니다. 1~3개월 흐름입니다.\\n\\n"
            f"| 지표 | 1개월 |\\n|---|---:|\\n{{rows}}\\n\\n"
            "![US 10Y](assets/macro-rates.svg)\\n\\n"
            "자산별 수익률은 [시장·섹터](market-sector.md)에 있습니다.\\n")
    return text, {{"macro-rates.svg": SVG}}
'''

RELEASE_SOURCES = {
    "modeler/scores/macro_context/__init__.py": CALC_SOURCE,
    "modeler/reporting/macro.py": RENDER_SOURCE,
}


def _module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(compile(source, f"<fake {name}>", "exec"), module.__dict__)
    return module


def install(monkeypatch) -> tuple[types.ModuleType, types.ModuleType]:
    """Put both fakes into ``sys.modules``; returns (calc, render) so a test can patch them."""
    calc = _module(CALC_MODULE, CALC_SOURCE)
    render = _module(RENDER_MODULE, RENDER_SOURCE)
    monkeypatch.setitem(sys.modules, CALC_MODULE, calc)
    monkeypatch.setitem(sys.modules, RENDER_MODULE, render)
    return calc, render


def doc(unit: str, invocation_id: str, /, *, state: str = "ok", **changes) -> dict:
    calc = _module(CALC_MODULE, CALC_SOURCE)
    if state == "failed":
        body = calc.failure_document(
            unit=unit,
            invocation_id=invocation_id,
            release="r-test",
            selection_sha256="a" * 64,
            reason=changes.pop("reason", "timeout"),
        )
    else:
        body = calc.build_document(
            {
                "schema_version": MS_SELECTION_SCHEMA,
                "report_date": unit,
                "input_cutoff": f"{unit}T09:30:00+09:00",
            },
            unit=unit,
            invocation_id=invocation_id,
            release="r-test",
            selection_sha256="a" * 64,
        )
    body.update(changes)
    return body


def dump(body: dict) -> bytes:
    return (
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()
