"""Earnings forecast A (research A stage 1): fixed constants and month arithmetic.

Every value here is part of the pre-registered design. Changing one changes the
frozen spec, so do it only through a new preregistration.
"""

from __future__ import annotations

import calendar
import itertools
from datetime import date

LABEL_SPEC_VERSION = "a1-v1"

# Short name -> metric_code in the v4p and dl_v4 marts.
METRICS: dict[str, str] = {
    "oi": "operating_income",
    "rev": "revenue",
    "ni": "net_income",
    "ocf": "operating_cash_flow",
    "ta": "total_assets",
    "te": "total_equity",
    "cash": "cash_and_cash_equivalents",
    "debt_st": "borrowings_short_term",
    "debt_lt": "borrowings_long_term",
    "capex_ppe": "capex_ppe",
    "capex_int": "capex_intangible",
    "div": "dividends_paid",
}

# Feature order is part of the spec (section 5.3 of the preregistration).
FEATURES: tuple[str, ...] = (
    "oi_ttm",
    "oi_q0",
    "oi_q1",
    "oi_q2",
    "oi_q3",
    "oi_chg_4q",
    "rev_ttm",
    "margin_ttm",
    "ocf_ttm",
    "accrual_ttm",
    "ni_ttm",
    "asset_growth",
    "equity_ratio",
    "cash_ratio",
    "debt_ratio",
    "capex_ttm",
    "div_ttm",
    "log_assets",
    "loss_ttm",
    "is_ofs",
    "stale_days",
)
BINARY_FEATURES: tuple[str, ...] = ("loss_ttm", "is_ofs")
CLIP_Q: tuple[float, float] = (0.01, 0.99)

# Universe rules.
SPAC_PATTERNS: tuple[str, ...] = ("스팩", "기업인수목적")
FINANCIAL_KSIC2: frozenset[str] = frozenset({"64", "65", "66"})
HOLDING_KSIC: str = "64992"

# Fiscal year and availability rules.
QUARTER_MONTHS: frozenset[int] = frozenset({3, 6, 9, 12})
A_T_MAX_STALE_MONTHS: int = 12
DEADLINE_DAYS: dict[str, int] = {"interim": 45, "annual": 90}

# Raw snapshot used for the run.
SNAPSHOT: str = "2026-10-23"
RAW_MAX_DATE: str = "2026-10-22"

# Models.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
_HGB_LEARNING_RATES = (0.05, 0.1)
_HGB_MAX_ITER = (100, 300)
_HGB_MAX_LEAF_NODES = (15, 31)
HGB_GRID: tuple[dict, ...] = tuple(
    {
        "learning_rate": lr,
        "max_iter": mi,
        "max_leaf_nodes": ml,
        "min_samples_leaf": 100,
        "early_stopping": False,
        "loss": "absolute_error",
    }
    for lr, mi, ml in itertools.product(_HGB_LEARNING_RATES, _HGB_MAX_ITER, _HGB_MAX_LEAF_NODES)
)
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

# Inner validation dates for hyperparameter choice: 13 quarter-ends, 2019-06-30..2022-06-30.
INNER_VAL_DATES: tuple[date, ...] = (
    date(2019, 6, 30),
    date(2019, 9, 30),
    date(2019, 12, 31),
    date(2020, 3, 31),
    date(2020, 6, 30),
    date(2020, 9, 30),
    date(2020, 12, 31),
    date(2021, 3, 31),
    date(2021, 6, 30),
    date(2021, 9, 30),
    date(2021, 12, 31),
    date(2022, 3, 31),
    date(2022, 6, 30),
)

# Bootstrap for the firm-cluster lower bound.
BOOTSTRAP_SEED: int = 20261103
BOOTSTRAP_DRAWS: int = 10000

REASON_CODES: tuple[str, ...] = (
    "IMMATURE",
    "SRC_MISSING",
    "ORIG_MISSING_CORR_ONLY",
    "MERGED",
    "DELISTED",
    "REPORT_STOPPED",
    "UNCONFIRMED",
)


def month_index(d: date) -> int:
    """Return the absolute month number of ``d``: ``year * 12 + month - 1``.

    Two dates share a month index exactly when they fall in the same calendar
    month, and consecutive months differ by 1. Marts are matched on this index,
    never on ``seq_key``, because the dl_v4 ``seq_key`` is renumbered per fiscal year.
    """
    return d.year * 12 + d.month - 1


def add_months_end(d: date, n: int) -> date:
    """Return the last calendar day of the month ``n`` months after ``d``'s month.

    ``n`` may be negative. The day of ``d`` is ignored.
    Examples: 2024-02-29 + 12 -> 2025-02-28; 2023-11-30 + 3 -> 2024-02-29.
    """
    total = month_index(d) + n
    year, month0 = divmod(total, 12)
    month = month0 + 1
    return date(year, month, calendar.monthrange(year, month)[1])
