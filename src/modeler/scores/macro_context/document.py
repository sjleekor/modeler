"""문서 모양과 바이트 규칙. 표준 라이브러리만 쓴다(실패 문서는 계산 의존성 없이 만들어야 한다)."""

from __future__ import annotations

import json
from typing import Any

from modeler.scores.macro_context.spec import SCHEMA


def failure_document(
    *, unit: str, invocation_id: str, release: str, selection_sha256: str | None, reason: str
) -> dict[str, Any]:
    """계산 전이나 중에 실패한 단계의 문서. ``reason``은 경로·원문이 없는 짧은 코드다."""
    return {
        "schema": SCHEMA,
        "state": "failed",
        "unit": unit,
        "invocation_id": invocation_id,
        "release": release,
        "selection_sha256": selection_sha256,
        "reason": reason,
    }


def document_bytes(body: dict[str, Any]) -> bytes:
    """같은 body는 같은 바이트다(키 정렬, NaN 금지, 끝에 줄바꿈 하나)."""
    return (
        json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def failure_reason(exc: BaseException) -> str:
    """예외를 짧은 사유 코드로 바꾼다(경로·원문 메시지는 싣지 않는다)."""
    from modeler.scores.market_sector import inputs_pin as ip

    if isinstance(exc, ip.InputChangedError):
        return "input_changed"
    if isinstance(exc, ip.InputPinError):
        return "input_unavailable"
    if isinstance(exc, (FileNotFoundError, OSError)):
        return "input_unavailable"
    name = type(exc).__name__
    if name == "KrNotSyncedError":
        return "input_unavailable"
    if name == "MacroInputError":
        return "input_missing"
    return "compute_error"
