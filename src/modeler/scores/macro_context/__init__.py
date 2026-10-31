"""일일 리포트 "거시 흐름" 섹션의 계산 모듈(``briefing-macro.v1``). 표시 전용이고 점수·순위에 쓰지 않는다.

* ``build_document``: selection이 고정한 입력(한국 ``common_feature_observation_raw``, 미국 ``macro_series``)만
  sha256을 다시 확인하고 열어 지표·요약·지난 보고서 대비·그림 데이터를 만든다. 실패하면 예외를 올린다.
* ``failure_document``: 계산 의존성 없이 만드는 실패 문서(``state="failed"``, 짧은 사유 코드).
* ``document_bytes``: 문서를 바이트로(키 정렬·NaN 금지·끝 줄바꿈).

``build_document``는 numpy·polars를 쓰므로 부를 때만 불러온다.
"""

from __future__ import annotations

from typing import Any

from modeler.scores.macro_context.document import document_bytes, failure_document, failure_reason
from modeler.scores.macro_context.spec import SCHEMA

__all__ = ["SCHEMA", "build_document", "document_bytes", "failure_document", "failure_reason"]


def build_document(
    selection: dict[str, Any], *, unit: str, invocation_id: str, release: str, selection_sha256: str
) -> dict[str, Any]:
    from modeler.scores.macro_context.build import build_document as _build

    return _build(
        selection,
        unit=unit,
        invocation_id=invocation_id,
        release=release,
        selection_sha256=selection_sha256,
    )
