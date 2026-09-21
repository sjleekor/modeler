"""``modeler.us.features.FAMILY_ORDER`` 단위 테스트.

레이크를 전혀 안 읽는다 — 매핑 자체(순서·개수·중복 없음·이름-함수 짝)만 본다.
실제 family 계산 로직은 각 ``test_feature_*.py``가 이미 검정한다.
"""

from __future__ import annotations

from modeler.us.features import FAMILY_ORDER
from modeler.us.features.calendar import add_calendar
from modeler.us.features.earnings import add_earnings
from modeler.us.features.filing_activity import add_filing_activity
from modeler.us.features.index_membership import add_index_membership
from modeler.us.features.insider import add_insider
from modeler.us.features.investment import add_investment
from modeler.us.features.liquidity import add_liquidity
from modeler.us.features.market import add_market
from modeler.us.features.momentum import add_momentum
from modeler.us.features.options_iv import add_options_iv
from modeler.us.features.payout import add_payout
from modeler.us.features.profitability import add_profitability
from modeler.us.features.reversal import add_reversal
from modeler.us.features.short import add_short
from modeler.us.features.valuation import add_valuation
from modeler.us.features.volatility import add_volatility

_EXPECTED = (
    ("F1_momentum", add_momentum),
    ("F2_reversal", add_reversal),
    ("F3_volatility", add_volatility),
    ("F4_liquidity", add_liquidity),
    ("F5_valuation", add_valuation),
    ("F6_profitability", add_profitability),
    ("F7_investment", add_investment),
    ("F8_payout", add_payout),
    ("F9_earnings", add_earnings),
    ("F10_insider", add_insider),
    ("F11_short", add_short),
    ("F12_filing_activity", add_filing_activity),
    ("F13_options_iv", add_options_iv),
    ("F14_index_membership", add_index_membership),
    ("F15_market", add_market),
    ("F16_calendar", add_calendar),
)


def test_family_order_has_16_entries() -> None:
    assert len(FAMILY_ORDER) == 16


def test_family_order_names_are_unique() -> None:
    names = [name for name, _ in FAMILY_ORDER]
    assert len(names) == len(set(names))


def test_family_order_matches_f1_through_f16_in_plan_order() -> None:
    """``04_feature_test_plan.md`` §3의 F1~F16 순서 그대로다."""
    assert FAMILY_ORDER == _EXPECTED


def test_family_order_functions_are_the_add_family_functions_themselves() -> None:
    """매핑이 감싸거나 복사한 함수가 아니라 각 모듈의 ``add_<family>`` 그 자체를 가리킨다."""
    for (_, fn), (_, expected_fn) in zip(FAMILY_ORDER, _EXPECTED, strict=True):
        assert fn is expected_fn
