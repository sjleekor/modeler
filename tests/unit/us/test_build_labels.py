"""``modeler.us.build_labels``의 순수 함수 단위 테스트.

레이크를 읽지 않는다 — ``main()``은 ``UsLake.resolve()``를 부르므로 통합
실행(``build_labels.py`` 작업 보고 참고)으로만 검증하고, 여기서는 이름 규칙
(``_dataset_name``)만 본다.
"""

from __future__ import annotations

from modeler.us.build_labels import DATASET_NAME, _dataset_name
from modeler.us.labels import HORIZON_TRADING_DAYS


def test_dataset_name_h21_keeps_the_existing_name() -> None:
    assert _dataset_name(HORIZON_TRADING_DAYS) == DATASET_NAME == "us_labels_v1"


def test_dataset_name_other_horizon_gets_suffix() -> None:
    assert _dataset_name(5) == "us_labels_h5_v1"
    assert _dataset_name(63) == "us_labels_h63_v1"
