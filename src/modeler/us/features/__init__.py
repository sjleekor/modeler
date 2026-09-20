"""피쳐 family — ``04_feature_test_plan.md`` §3의 F1\\~F16.

**파일 하나에 family 하나다** (``06_execution_steps.md`` M2). 검정 대상은
F1\\~F14의 44개고, F15(시장 수준 4개)·F16(캘린더 1개)은 단독 검정을 하지 않는다 —
상호작용·분해 축·정규화에만 쓴다.

각 family 모듈은 ``add_<family>(panel, lake, ...) -> pl.DataFrame``을 내놓고,
피쳐마다 ``<이름>_isna`` 플래그를 같이 낸다. 결측 자체가 정보인 경우가 있다
(``iv_isna``는 유동성 프록시, 재무 결측은 외국발행사 플래그다).
"""

from __future__ import annotations
