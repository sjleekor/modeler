"""시장·섹터 일일 문서(``market-sector-D.json``)의 계약. 표준 라이브러리만 쓴다.

문서는 렌더러(`reporting/markdown.py --market-sector`)와 publisher가 읽는다. 채점(`score_daily`)이
정상 문서를 쓰고, 채점을 끝내지 못한 날은 coordinator가 이 모듈의 ``failure_document``로 실패
문서를 쓴다. 실패 문서에는 자산 행이 없고 ``failure``(단계·원인 클래스·사유 코드·timeout)만
있다. 렌더러는 이것을 시장·섹터 섹션의 실패 사유로 적는다.
"""

from __future__ import annotations

from datetime import date
from typing import Any

SCHEMA = "market-sector-daily.v1"
SCHEMA_FAILED_STATUS = "failed"


def failure_document(
    report_date: date,
    *,
    stage: str,
    error_class: str,
    reason: str,
    exit_code: int | None = None,
    timed_out: bool = False,
    markets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """채점을 끝내지 못한 날의 문서.

    ``reason``은 고정된 사유 코드다. 원문 메시지와 경로는 싣지 않는다.
    """
    return {
        "schema_version": SCHEMA,
        "report_date": report_date.isoformat(),
        "historical_replay": False,
        "status": SCHEMA_FAILED_STATUS,
        "assets": [],
        "failure": {
            "stage": stage,
            "error_class": error_class,
            "reason": reason,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "markets": markets or {},
        },
        "notes": [],
    }
