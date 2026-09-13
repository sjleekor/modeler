"""``feat_event_scan_daily`` — Phase B B-5 (04_specific_plan_B.md §4.4-§4.5, B-5).

Two independent PIT continuous features, both interval-joined onto the daily
panel the same way as B-4:

    ev_net_share_issuance_yoy   §4.4 — economic net new shares over the
                                 trailing year, using ``dart_capital_change_raw``
                                 (B-PR2) to separate economic issuance (paid-in
                                 capital increases, conversions, option
                                 exercises) from mechanical actions (splits,
                                 bonus issues, stock dividends) that a naive
                                 share-count YoY would misread as issuance.
    ev_payout_yield             §4.5 — (cash dividends + buyback cash) / market
                                 cap, cash dividends preferring a direct total
                                 amount row and falling back to a DPS x
                                 eligible-shares proxy, buyback reusing B-3's
                                 already-TTM'd ``treasury_share_acquisition_amount``.

Why this reads ``dart_share_count_raw``/``dart_shareholder_return_raw`` directly
rather than through B-2/B-3: ``now_to_isu_stock_totqy``/``now_to_dcrs_stock_totqy``
(cumulative-since-listing issuance/decrease totals, used for the identity
check) and the dividend *total* row are not covered by any
``metric_rules.py`` rule, and adding one would touch the shared file B-2/B-3
deliberately avoid (its legacy consumer is frozen against golden-parity
fixtures — see ``metric_vintages.py``'s module docstring). ``dps`` and
``treasury_share_acquisition_amount`` *are* already metric_rules.py-covered,
so those two are read from ``stock_metric_vintage_fact``/
``fin_quarterly_metric_vintage`` instead of re-deriving them.

Identity reconciliation uses only the numeric share-count fields
(``istc_totqy``, ``now_to_isu_stock_totqy``, ``now_to_dcrs_stock_totqy``); the
free-text reason columns (``redc``, ``profit_incnr``, ``rdmstk_repy``, ``etc``)
are never parsed for a reason (§3.4/§4.4: "이 source에 없는 증가 사유를
추정하지 않는다") — they are not read by this module at all.

Dividend row-name matching (best-effort, see ``CASH_DIVIDEND_TOTAL_ROW_NAME``):
this repository has never captured a live OpenDART alotMatter payload with a
total cash-dividend-amount row, so the row_name below is an unverified best
guess. If it never matches, the code does exactly what §4.5 prescribes when
the direct total is unavailable: falls back to the DPS proxy — it does not
silently produce a wrong number.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

EVENT_SCAN_TABLE = "feat_event_scan_daily"
_CAL_TABLE = "_event_scan_calendar"

# §4.4 isu_dcrs_stle classification — unrecognized values fall through to
# 'unclassified' (never guessed) and block ev_net_share_issuance_yoy for that
# trailing-year window (§4.4: "미분류 isu_dcrs_stle ... NULL과 quality flag").
#
# Matching stays EXACT. No normalization, no substring rule: a rule loose enough
# to absorb 무상감자 into 감자(무상) is also loose enough to silently absorb the
# next unseen reason, and §4.4 step 3 forbids inferring a reason this source did
# not give. Every string below was added by reading it, not by pattern.
#
# v2 (2026-08-12) added the four reasons the collected history actually returns
# that v1 had no entry for — 22.4% of annual-vintage events were falling through
# to 'unclassified' purely because the catalog was short. See
# 04_specific_plan_B.md §4.4 "매핑 판단 근거".
#
# v3 (2026-08-15) added 유상감자 to the economic-decrease set. v2 fixed exactly
# this word-order variant for 무상감자 but reviewed only the 259 annual-vintage
# events, where 유상감자 does not appear; over the whole capital-change history
# it is 699 rows / 88 issuers. Until v3 the economic-decrease set matched
# nothing at all (the source never writes 감자(유상)), so the feature could not
# produce a negative value and every window holding a 유상감자 row was dropped
# as unclassified. See 10_known_issues.md I3.
#
# v4 (2026-08-15) is not a catalog change: the position and vintage selectors
# got total-order tiebreakers so the issuance figures stop depending on scan
# order (10_known_issues.md I12). Bumped because the values move.
EVENT_FEATURE_FORMULA_VERSION = "issuance_v4"

# The payout feature shares this module but not this catalog: its formula is the
# dividend/buyback TTM sum over PIT market cap. Fingerprinting it separately
# keeps a catalog bump from implying that payout changed (and vice versa) —
# ``ev_payout_yield``'s card would otherwise read "issuance_vN", which is not
# the rule it was built with.
#
#   payout_v1 — the 2026-08 rules as first scanned (§4.5).
#   payout_v2 — direct dividend totals are unit-checked against the DPS-implied
#               amount and negative totals rejected (10_known_issues.md I5).
#   payout_v3 — dividend rows are restricted to the common share (or an unsplit
#               filing) and every winner-selection ORDER BY is a total order, so
#               the mart stops depending on scan order (10_known_issues.md I12).
PAYOUT_FEATURE_FORMULA_VERSION = "payout_v3"

ECONOMIC_INCREASE_REASONS = frozenset(
    {
        "유상증자(일반공모)",
        "유상증자(주주배정)",
        "유상증자(제3자배정)",
        "유상증자(주주우선공모)",  # v2 — same paid-in increase, different allocation
        "전환권행사",
        "신주인수권행사",  # v2 — warrant exercise, economically identical to 전환권행사
        "주식매수선택권행사",
        "출자전환",  # v2 — debt swapped for new shares; real dilution, see plan §4.4
    }
)
ECONOMIC_DECREASE_REASONS = frozenset(
    {
        "감자(유상)",
        "유상감자",  # v3 — the same action, spelled the other way round
    }
)
MECHANICAL_INCREASE_REASONS = frozenset({"무상증자", "주식배당", "주식분할"})
MECHANICAL_DECREASE_REASONS = frozenset(
    {
        "주식병합",
        "감자(무상)",
        "무상감자",  # v2 — the same action, spelled the other way round
        "소각",
    }
)

# payout_v3: which share class a dividend row describes. ``stock_knd`` is not in
# the row's key, so a company reporting 보통주 and 우선주 separately puts several
# rows in one (ticker, bsns_year, row_name) group. The DPS proxy multiplies the
# winning per-share figure by *all* shares outstanding, so picking the preferred
# row silently prices common shares at the preferred dividend.
#
# Rank 0 is the common share, rank 1 is a filer that did not split by class at
# all (one class, so the figure is the common one). Everything else — preferred
# in its several spellings, 종류주식, and the shareholder-category rows that leak
# into this section — is excluded rather than ranked: we cannot know what share
# count it belongs to. Exact match only, per 04_specific_plan_B.md §4.4.2.
COMMON_STOCK_KINDS = frozenset({"보통주", "보통주식"})
UNSPLIT_STOCK_KINDS = frozenset({"-", ""})
_STOCK_KIND_RANK_SQL = (
    "CASE WHEN sr.stock_knd IN ("
    + ", ".join(f"'{k}'" for k in sorted(COMMON_STOCK_KINDS))
    + ") THEN 0 WHEN sr.stock_knd IS NULL OR sr.stock_knd IN ("
    + ", ".join(f"'{k}'" for k in sorted(UNSPLIT_STOCK_KINDS))
    + ") THEN 1 END"
)

# Unverified best guess — see module docstring. DPS's row_name is the one
# already relied on elsewhere in this codebase (opendart_share_info tests).
CASH_DIVIDEND_TOTAL_ROW_NAME = "현금배당금총액(백만원)"
CASH_DIVIDEND_TOTAL_UNIT_SCALE = 1_000_000  # "백만원" -> won

# Some filers write the raw won amount into the 백만원-labelled field, which the
# unconditional scale above then inflates by 1e6 (일양약품 2019: 2,249,158,200 in
# the 2019 filing vs 2,249 in the 2020 one — the same 22.5억). Only 4 rows of
# 17,859 do it, but an inflated one produces a payout yield in the thousands.
#
# DPS x (issued - treasury) is an independent estimate of the same amount from a
# different row of the same filing, so it can tell a unit slip (1e6x) from a
# real number. Legitimate disagreement between the two is bounded by share-class
# and timing effects (a few x at most), so the multiple below only has to sit
# between "a few" and 1e6 — 100 leaves three orders of magnitude of headroom on
# each side. When the direct total fails the check the row falls through to the
# DPS proxy rather than being corrected by division: the source said 원 or it
# said 백만원, and guessing which is not this layer's job.
DIVIDEND_TOTAL_SANITY_MULTIPLE = 100

# Backstop for the filings that carry no DPS row, where the check above has
# nothing to compare against. Observed distribution: median 0.4%, 99th 13.5%,
# 99.9th 51%; the surviving unit slips sit in the thousands. 10 (= 1000% of
# market cap returned in one year) is far outside anything a filer does and far
# inside the error, so it separates them without touching a real tail.
PAYOUT_YIELD_MAX = 10

# A negative cash dividend total is not a quantity this feature can carry; the
# source has 27 such rows across 12 issuers. Excluded here rather than clipped
# downstream, for the same reason.
_DPS_IMPLIED_TOTAL = "d.dps_raw * (sw.issued_shares - COALESCE(sw.treasury_shares, 0))"
DIRECT_TOTAL_USABLE_SQL = f"""(
                d.dividend_total_raw IS NOT NULL
                AND d.dividend_total_raw >= 0
                AND (
                    d.dps_raw IS NULL OR sw.issued_shares IS NULL
                    OR {_DPS_IMPLIED_TOTAL} <= 0
                    OR d.dividend_total_raw * {CASH_DIVIDEND_TOTAL_UNIT_SCALE}
                       <= {DIVIDEND_TOTAL_SANITY_MULTIPLE} * ({_DPS_IMPLIED_TOTAL})
                )
            )"""
DPS_ROW_NAME = "주당 현금배당금(원)"

DEFAULT_ISSUANCE_IDENTITY_TOLERANCE = 0

# §4.4.1 — irdsSttus returns the whole since-listing history on every report, so
# one real capital action appears once per report vintage stored in
# dart_capital_change_raw. Summing the table as-is multiplies every event by its
# vintage count and breaks the §4.4 step-4 identity for good. The fix is to pick
# ONE vintage per ticker and use only its rows: vintages disagree (000040's
# 4,476,350-share conversion is dated 2021-01-31 in the FY2024 report and
# 2021-01-13 in the FY2025 one), so unioning them by event identity would leave
# a corrected event standing twice. Only the annual report carries the history.
ANNUAL_REPRT_CODE = "11011"

# Which vintage to pick. Both are implemented because the choice between them is
# decided by measurement, not by preference — see 04_specific_plan_B.md §4.4.1
# "vintage distance probe" for the pre-registered thresholds. The probe ran on
# snapshot_date=2026-08-12: the 9-year-distance feature-changing rate is 0.1824,
# far past the 5% row of the threshold table, and strict PIT's identity pass rate
# (0.653) is nowhere near the "half of latest" (0.375) override floor. So (b) is
# the adopted policy, not a provisional one — see 08 §4.1.1 for the measurements.
VINTAGE_POLICY_LATEST = "latest_vintage"
VINTAGE_POLICY_STRICT_PIT = "strict_pit"
VINTAGE_POLICIES = (VINTAGE_POLICY_LATEST, VINTAGE_POLICY_STRICT_PIT)
DEFAULT_VINTAGE_POLICY = VINTAGE_POLICY_STRICT_PIT


def _classification_case(alias: str) -> str:
    def in_list(reasons: frozenset[str]) -> str:
        return "(" + ", ".join(f"'{r}'" for r in sorted(reasons)) + ")"

    economic_inc = in_list(ECONOMIC_INCREASE_REASONS)
    economic_dec = in_list(ECONOMIC_DECREASE_REASONS)
    mechanical_inc = in_list(MECHANICAL_INCREASE_REASONS)
    mechanical_dec = in_list(MECHANICAL_DECREASE_REASONS)
    return f"""CASE
        WHEN {alias}.isu_dcrs_stle IN {economic_inc} THEN 'economic_increase'
        WHEN {alias}.isu_dcrs_stle IN {economic_dec} THEN 'economic_decrease'
        WHEN {alias}.isu_dcrs_stle IN {mechanical_inc} THEN 'mechanical_increase'
        WHEN {alias}.isu_dcrs_stle IN {mechanical_dec} THEN 'mechanical_decrease'
        ELSE 'unclassified'
    END"""


def _quarter_ordinal_case(col: str) -> str:
    return (
        f"CASE {col} WHEN '11013' THEN 1 WHEN '11012' THEN 2 "
        "WHEN '11014' THEN 3 WHEN '11011' THEN 4 END"
    )


def _position_vintage_sql(policy: str) -> str:
    """CTEs picking which capital-change vintage each filing position reads.

    ``latest_vintage`` takes one newest vintage per ticker and uses it for every
    position; ``strict_pit`` takes, per position, the newest vintage already
    disclosed by that position's ``available_from``. Positions with no eligible
    vintage keep a row with NULL vintage keys, so they end up with zero events
    and fail the identity check rather than silently reading someone else's.
    """
    if policy == VINTAGE_POLICY_LATEST:
        return """
    selected_vintage AS (
        SELECT ticker, vintage_bsns_year, vintage_rcept_no, vintage_available_from
        FROM capital_change_vintages
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker
            ORDER BY vintage_bsns_year DESC, vintage_disclosed_date DESC NULLS LAST,
                     vintage_rcept_no DESC
        ) = 1
    ),
    position_vintage AS (
        SELECT
            wp.*,
            sv.vintage_bsns_year AS capital_vintage_bsns_year,
            sv.vintage_rcept_no AS capital_vintage_rcept_no,
            sv.vintage_available_from AS capital_vintage_available_from
        FROM with_prior_year wp
        LEFT JOIN selected_vintage sv ON sv.ticker = wp.ticker
    ),"""
    return """
    position_vintage AS (
        SELECT
            wp.*,
            v.vintage_bsns_year AS capital_vintage_bsns_year,
            v.vintage_rcept_no AS capital_vintage_rcept_no,
            v.vintage_available_from AS capital_vintage_available_from
        FROM with_prior_year wp
        LEFT JOIN capital_change_vintages v
          ON v.ticker = wp.ticker
         AND v.vintage_available_from IS NOT NULL
         AND wp.available_from IS NOT NULL
         AND v.vintage_available_from <= wp.available_from
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY wp.ticker, wp.bsns_year, wp.reprt_code
            ORDER BY v.vintage_available_from DESC NULLS LAST, v.vintage_rcept_no DESC,
                     -- total order (I12): a filer catching up on several years at
                     -- once (001470 filed 2018-2021 on 2023-12-12) puts multiple
                     -- vintages at the same availability.
                     v.vintage_bsns_year DESC NULLS LAST
        ) = 1
    ),"""


def build_issuance_sql(
    *,
    share_count_view: str = "dart_share_count_raw",
    capital_change_view: str = "dart_capital_change_raw",
    corp_view: str = "dart_corp_master",
    calendar_table: str = _CAL_TABLE,
    identity_tolerance: int = DEFAULT_ISSUANCE_IDENTITY_TOLERANCE,
    vintage_policy: str = DEFAULT_VINTAGE_POLICY,
) -> str:
    """One row per (ticker, filing position) with ``ev_net_share_issuance_yoy``.

    Uses the exact seq_key-4 self-join pattern from B-3/B-4 (not LAG) so a
    ticker with a gap in its filing history never silently pairs with the
    wrong prior-year filing.

    ``vintage_policy`` selects between the two §4.4.1 candidates — see
    ``VINTAGE_POLICY_LATEST`` / ``VINTAGE_POLICY_STRICT_PIT``.
    """
    if vintage_policy not in VINTAGE_POLICIES:
        raise ValueError(
            f"unknown vintage_policy {vintage_policy!r}; expected one of {VINTAGE_POLICIES}"
        )
    quarter_ord = _quarter_ordinal_case("s.reprt_code")
    position_vintage = _position_vintage_sql(vintage_policy)
    return f"""
    WITH corp AS (
        SELECT ticker, market, corp_code FROM {corp_view}
        WHERE ticker IS NOT NULL AND ticker <> '' AND market IS NOT NULL
    ),
    positions AS (
        SELECT
            s.ticker, c.market, s.corp_code, s.bsns_year, s.reprt_code,
            s.bsns_year * 4 + {quarter_ord} AS seq_key,
            s.rcept_no, s.istc_totqy, s.now_to_isu_stock_totqy, s.now_to_dcrs_stock_totqy,
            s.stlm_dt,
            CASE WHEN s.rcept_no ~ '^[0-9]{{14}}$'
                 THEN strptime(left(s.rcept_no, 8), '%Y%m%d')::DATE END AS disclosed_date
        FROM {share_count_view} s
        JOIN corp c ON c.ticker = s.ticker
        WHERE s.se = '합계'
    ),
    winners AS (
        SELECT *
        FROM positions
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, bsns_year, reprt_code
            ORDER BY disclosed_date ASC NULLS LAST, rcept_no ASC,
                     -- total order (I12): duplicate '합계' rows under one receipt
                     -- would otherwise resolve by scan order.
                     stlm_dt DESC NULLS LAST, istc_totqy DESC NULLS LAST,
                     now_to_isu_stock_totqy DESC NULLS LAST,
                     now_to_dcrs_stock_totqy DESC NULLS LAST
        ) = 1
    ),
    with_available AS (
        SELECT *,
            CASE WHEN disclosed_date IS NOT NULL
                 THEN (SELECT MIN(d) FROM {calendar_table} WHERE d > disclosed_date)
            END AS available_from
        FROM winners
    ),
    with_prior_year AS (
        SELECT
            cur.ticker, cur.market, cur.corp_code, cur.bsns_year, cur.reprt_code,
            cur.seq_key, cur.available_from, cur.stlm_dt, cur.istc_totqy,
            cur.now_to_isu_stock_totqy, cur.now_to_dcrs_stock_totqy,
            prev.stlm_dt AS stlm_dt_prior, prev.istc_totqy AS istc_totqy_prior,
            prev.now_to_isu_stock_totqy AS now_to_isu_prior,
            prev.now_to_dcrs_stock_totqy AS now_to_dcrs_prior
        FROM with_available cur
        LEFT JOIN with_available prev
          ON prev.ticker = cur.ticker AND prev.seq_key = cur.seq_key - 4
    ),
    capital_change_vintages AS (
        SELECT
            ticker, vintage_bsns_year, vintage_rcept_no, vintage_disclosed_date,
            CASE WHEN vintage_disclosed_date IS NOT NULL
                 THEN (SELECT MIN(d) FROM {calendar_table} WHERE d > vintage_disclosed_date)
            END AS vintage_available_from
        FROM (
            SELECT DISTINCT
                cc.ticker,
                cc.bsns_year AS vintage_bsns_year,
                cc.rcept_no AS vintage_rcept_no,
                CASE WHEN cc.rcept_no ~ '^[0-9]{{14}}$'
                     THEN strptime(left(cc.rcept_no, 8), '%Y%m%d')::DATE END
                    AS vintage_disclosed_date
            FROM {capital_change_view} cc
            JOIN corp c ON c.ticker = cc.ticker
            WHERE cc.reprt_code = '{ANNUAL_REPRT_CODE}'
        )
    ),{position_vintage}
    capital_change_classified AS (
        SELECT
            cc.ticker,
            cc.bsns_year AS vintage_bsns_year,
            cc.rcept_no AS vintage_rcept_no,
            cc.isu_dcrs_de, cc.isu_dcrs_qy,
            {_classification_case("cc")} AS action_class
        FROM {capital_change_view} cc
        JOIN corp c ON c.ticker = cc.ticker
        WHERE cc.reprt_code = '{ANNUAL_REPRT_CODE}'
          AND cc.isu_dcrs_de IS NOT NULL
    ),
    issuance_summary AS (
        SELECT
            wp.ticker, wp.market, wp.corp_code, wp.bsns_year, wp.reprt_code,
            wp.seq_key, wp.available_from, wp.istc_totqy, wp.istc_totqy_prior,
            wp.capital_vintage_bsns_year, wp.capital_vintage_available_from,
            (wp.now_to_isu_prior IS NOT NULL) AS has_prior_year,
            wp.now_to_isu_stock_totqy - wp.now_to_isu_prior AS issuance_delta_1y,
            wp.now_to_dcrs_stock_totqy - wp.now_to_dcrs_prior AS decrease_delta_1y,
            COALESCE(SUM(CASE WHEN cc.action_class = 'economic_increase'
                               THEN cc.isu_dcrs_qy END), 0) AS economic_increase_1y,
            COALESCE(SUM(CASE WHEN cc.action_class = 'mechanical_increase'
                               THEN cc.isu_dcrs_qy END), 0) AS mechanical_increase_1y,
            COALESCE(SUM(CASE WHEN cc.action_class = 'economic_decrease'
                               THEN cc.isu_dcrs_qy END), 0) AS economic_decrease_1y,
            COALESCE(SUM(CASE WHEN cc.action_class = 'mechanical_decrease'
                               THEN cc.isu_dcrs_qy END), 0) AS mechanical_decrease_1y,
            COALESCE(SUM(CASE WHEN cc.action_class = 'unclassified'
                               THEN cc.isu_dcrs_qy END), 0) AS unclassified_1y
        FROM position_vintage wp
        LEFT JOIN capital_change_classified cc
          ON cc.ticker = wp.ticker
         AND cc.vintage_bsns_year = wp.capital_vintage_bsns_year
         AND cc.vintage_rcept_no = wp.capital_vintage_rcept_no
         AND wp.stlm_dt_prior IS NOT NULL
         AND cc.isu_dcrs_de > wp.stlm_dt_prior AND cc.isu_dcrs_de <= wp.stlm_dt
        GROUP BY
            wp.ticker, wp.market, wp.corp_code, wp.bsns_year, wp.reprt_code, wp.seq_key,
            wp.available_from, wp.istc_totqy, wp.istc_totqy_prior, wp.now_to_isu_prior,
            wp.now_to_isu_stock_totqy, wp.now_to_dcrs_stock_totqy, wp.now_to_dcrs_prior,
            wp.capital_vintage_bsns_year, wp.capital_vintage_available_from
    )
    SELECT
        *,
        (unclassified_1y = 0) AS issuance_classification_complete,
        (
            ABS(
                (istc_totqy_prior + economic_increase_1y + mechanical_increase_1y
                 - economic_decrease_1y - mechanical_decrease_1y) - istc_totqy
            ) <= {identity_tolerance}
        ) AS issuance_identity_ok,
        CASE
            WHEN has_prior_year AND unclassified_1y = 0
             AND ABS(
                    (istc_totqy_prior + economic_increase_1y + mechanical_increase_1y
                     - economic_decrease_1y - mechanical_decrease_1y) - istc_totqy
                 ) <= {identity_tolerance}
             AND istc_totqy_prior > 0
            THEN (economic_increase_1y - economic_decrease_1y) / istc_totqy_prior
        END AS ev_net_share_issuance_yoy
    FROM issuance_summary
    -- issuance_v4: one position per availability date, same reason as the payout
    -- side. 001470 filed its 2018-2021 positions together on 2023-12-12, so four
    -- rows shared an availability and which one owned the interval depended on
    -- scan order. Resolved to the latest fiscal period.
    -- See 10_known_issues.md I12.
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY ticker, available_from
        ORDER BY seq_key DESC
    ) = 1
    """


def build_payout_sql(
    *,
    shareholder_return_view: str = "dart_shareholder_return_raw",
    vintage_view: str = "stock_metric_vintage_fact",
    quarterly_view: str = "fin_quarterly_metric_vintage",
    calendar_table: str = _CAL_TABLE,
) -> str:
    """One row per (ticker, annual filing) with dividend/buyback components.

    Dividends are annual-cadence (alotMatter's thstrm/frmtrm/lwfr compare
    fiscal years, not quarters), so this does not need B-3's quarter/TTM
    machinery — only the annual (``reprt_code='11011'``) filing position.
    """
    direct_total_usable = DIRECT_TOTAL_USABLE_SQL
    return f"""
    WITH dividend_rows AS (
        SELECT
            sr.ticker, sr.bsns_year, sr.rcept_no, sr.row_name, sr.value_numeric, sr.value_text,
            sr.stock_knd,
            {_STOCK_KIND_RANK_SQL} AS stock_kind_rank,
            CASE WHEN sr.rcept_no ~ '^[0-9]{{14}}$'
                 THEN strptime(left(sr.rcept_no, 8), '%Y%m%d')::DATE END AS disclosed_date
        FROM {shareholder_return_view} sr
        WHERE sr.statement_type = 'dividend' AND sr.reprt_code = '11011'
          AND sr.metric_code = 'thstrm'
          AND sr.row_name IN ('{CASH_DIVIDEND_TOTAL_ROW_NAME}', '{DPS_ROW_NAME}')
    ),
    dividend_winners AS (
        SELECT *
        FROM dividend_rows
        WHERE stock_kind_rank IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, bsns_year, row_name
            ORDER BY stock_kind_rank ASC,
                     disclosed_date ASC NULLS LAST, rcept_no ASC,
                     -- total order: without these, rows tying on every key above
                     -- (the same filing's several dividend lines) resolve by scan
                     -- order, which is parallelism-dependent. See I12.
                     value_numeric DESC NULLS LAST, stock_knd ASC NULLS LAST,
                     value_text ASC NULLS LAST
        ) = 1
    ),
    dividend_wide AS (
        SELECT
            ticker, bsns_year,
            MAX(CASE WHEN row_name = '{CASH_DIVIDEND_TOTAL_ROW_NAME}' THEN value_numeric END)
                AS dividend_total_raw,
            MAX(CASE WHEN row_name = '{CASH_DIVIDEND_TOTAL_ROW_NAME}' THEN value_text END)
                AS dividend_total_text,
            MAX(CASE WHEN row_name = '{CASH_DIVIDEND_TOTAL_ROW_NAME}' THEN disclosed_date END)
                AS dividend_total_disclosed,
            MAX(CASE WHEN row_name = '{DPS_ROW_NAME}' THEN value_numeric END) AS dps_raw,
            MAX(CASE WHEN row_name = '{DPS_ROW_NAME}' THEN value_text END) AS dps_text,
            MAX(CASE WHEN row_name = '{DPS_ROW_NAME}' THEN disclosed_date END) AS dps_disclosed
        FROM dividend_winners
        GROUP BY ticker, bsns_year
    ),
    shares_annual AS (
        SELECT ticker, bsns_year, metric_code, value_numeric, available_from
        FROM {vintage_view}
        WHERE metric_code IN ('issued_shares', 'treasury_shares') AND reprt_code = '11011'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, bsns_year, metric_code
            ORDER BY COALESCE(is_revision, FALSE) ASC, available_from ASC, rcept_no ASC,
                     -- total order (I12)
                     value_numeric DESC NULLS LAST
        ) = 1
    ),
    shares_wide AS (
        SELECT
            ticker, bsns_year,
            MAX(CASE WHEN metric_code = 'issued_shares' THEN value_numeric END) AS issued_shares,
            MAX(CASE WHEN metric_code = 'treasury_shares' THEN value_numeric END)
                AS treasury_shares
        FROM shares_annual
        GROUP BY ticker, bsns_year
    ),
    dividend_resolved AS (
        SELECT
            d.ticker, d.bsns_year,
            CASE WHEN d.dividend_total_disclosed IS NOT NULL
                 THEN (
                    SELECT MIN(c.d) FROM {calendar_table} c
                    WHERE c.d > d.dividend_total_disclosed
                 )
            END AS dividend_total_available_from,
            CASE
                WHEN {direct_total_usable}
                     THEN d.dividend_total_raw * {CASH_DIVIDEND_TOTAL_UNIT_SCALE}
                WHEN d.dividend_total_text = '-' THEN 0
                WHEN d.dps_raw IS NOT NULL AND sw.issued_shares IS NOT NULL
                     THEN d.dps_raw * (sw.issued_shares - COALESCE(sw.treasury_shares, 0))
                WHEN d.dps_text = '-' THEN 0
            END AS cash_dividends_total,
            CASE
                WHEN {direct_total_usable} OR d.dividend_total_text = '-'
                     THEN 'direct_total'
                WHEN d.dps_raw IS NOT NULL THEN 'dps_proxy'
                WHEN d.dps_text = '-' THEN 'dps_proxy'
            END AS dividend_source,
            CASE WHEN d.dps_disclosed IS NOT NULL
                 THEN (SELECT MIN(c.d) FROM {calendar_table} c WHERE c.d > d.dps_disclosed)
            END AS dps_available_from
        FROM dividend_wide d
        LEFT JOIN shares_wide sw ON sw.ticker = d.ticker AND sw.bsns_year = d.bsns_year
    ),
    buyback AS (
        SELECT
            ticker, bsns_year,
            CASE WHEN fs_basis = 'CFS' THEN ttm_value END AS buyback_cfs,
            CASE WHEN fs_basis = 'OFS' THEN ttm_value END AS buyback_ofs,
            CASE WHEN fs_basis = 'CFS' THEN ttm_available_from END AS buyback_avail_cfs,
            CASE WHEN fs_basis = 'OFS' THEN ttm_available_from END AS buyback_avail_ofs
        FROM {quarterly_view}
        WHERE metric_code = 'treasury_share_acquisition_amount' AND quarter = 'Q4'
    ),
    buyback_wide AS (
        SELECT
            ticker, bsns_year,
            -- payout_v2: a negative treasury *acquisition* amount is not a
            -- quantity this feature can carry (disposals are a different line
            -- item). Dropping it leaves the dividend leg intact rather than
            -- letting the sum go negative.
            CASE WHEN COALESCE(MAX(buyback_cfs), MAX(buyback_ofs)) >= 0
                 THEN COALESCE(MAX(buyback_cfs), MAX(buyback_ofs)) END AS buyback_cash_ttm,
            CASE WHEN COALESCE(MAX(buyback_cfs), MAX(buyback_ofs)) >= 0
                 THEN COALESCE(MAX(buyback_avail_cfs), MAX(buyback_avail_ofs))
            END AS buyback_available_from
        FROM buyback
        GROUP BY ticker, bsns_year
    )
    SELECT
        dr.ticker, dr.bsns_year,
        COALESCE(dr.dividend_total_available_from, dr.dps_available_from)
            AS dividend_available_from,
        dr.cash_dividends_total, dr.dividend_source,
        bw.buyback_cash_ttm, bw.buyback_available_from,
        greatest(
            COALESCE(dr.dividend_total_available_from, dr.dps_available_from),
            bw.buyback_available_from
        ) AS payout_available_from
    FROM dividend_resolved dr
    LEFT JOIN buyback_wide bw ON bw.ticker = dr.ticker AND bw.bsns_year = dr.bsns_year
    -- payout_v3: one position per availability date. A filer catching up files
    -- several years at once, and the daily interval join reads
    -- [payout_available_from, next) — with two rows sharing a date the interval
    -- chain, and so every later date's value, depended on scan order. Resolved
    -- to the latest fiscal year, matching the daily feature's meaning.
    -- See 10_known_issues.md I12.
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY dr.ticker, payout_available_from
        ORDER BY dr.bsns_year DESC
    ) = 1
    """


def build_event_scan_daily_sql(
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    share_count_view: str = "dart_share_count_raw",
    capital_change_view: str = "dart_capital_change_raw",
    shareholder_return_view: str = "dart_shareholder_return_raw",
    vintage_view: str = "stock_metric_vintage_fact",
    quarterly_view: str = "fin_quarterly_metric_vintage",
    corp_view: str = "dart_corp_master",
    calendar_table: str = _CAL_TABLE,
    identity_tolerance: int = DEFAULT_ISSUANCE_IDENTITY_TOLERANCE,
    vintage_policy: str = DEFAULT_VINTAGE_POLICY,
) -> str:
    """SQL producing ``feat_event_scan_daily``: issuance + payout, daily PIT."""
    issuance_sql = build_issuance_sql(
        share_count_view=share_count_view,
        capital_change_view=capital_change_view,
        corp_view=corp_view,
        calendar_table=calendar_table,
        identity_tolerance=identity_tolerance,
        vintage_policy=vintage_policy,
    )
    payout_sql = build_payout_sql(
        shareholder_return_view=shareholder_return_view,
        vintage_view=vintage_view,
        quarterly_view=quarterly_view,
        calendar_table=calendar_table,
    )
    return f"""
    WITH issuance_positions AS ({issuance_sql}),
    issuance_intervals AS (
        SELECT
            *,
            LEAD(available_from) OVER (PARTITION BY ticker ORDER BY available_from)
                AS next_available_from
        FROM issuance_positions
        WHERE available_from IS NOT NULL
    ),
    payout_positions AS ({payout_sql}),
    payout_intervals AS (
        SELECT
            *,
            LEAD(payout_available_from) OVER (PARTITION BY ticker ORDER BY payout_available_from)
                AS next_payout_available_from
        FROM payout_positions
        WHERE payout_available_from IS NOT NULL
    ),
    panel AS (
        SELECT
            pit.trade_date, pit.ticker, pit.market, pit.market_cap_pit,
            pit.shares_is_available, pit.shares_invalid_flag,
            q.is_halted, q.valid_session_idx
        FROM {pit_view} pit
        LEFT JOIN {quality_view} q USING (trade_date, ticker, market)
    ),
    joined AS (
        SELECT
            panel.*,
            iss.ev_net_share_issuance_yoy, iss.available_from AS issuance_available_from,
            iss.issuance_identity_ok, iss.issuance_classification_complete,
            pay.cash_dividends_total, pay.dividend_source, pay.buyback_cash_ttm,
            pay.payout_available_from
        FROM panel
        LEFT JOIN issuance_intervals iss
          ON iss.ticker = panel.ticker
         AND iss.available_from <= panel.trade_date
         AND (iss.next_available_from IS NULL OR panel.trade_date < iss.next_available_from)
        LEFT JOIN payout_intervals pay
          ON pay.ticker = panel.ticker
         AND pay.payout_available_from <= panel.trade_date
         AND (pay.next_payout_available_from IS NULL
              OR panel.trade_date < pay.next_payout_available_from)
    ),
    scored AS (
        SELECT
            *,
            (market_cap_pit IS NOT NULL AND market_cap_pit > 0
             AND shares_is_available AND NOT shares_invalid_flag
             AND NOT COALESCE(is_halted, TRUE) AND valid_session_idx IS NOT NULL
            ) AS base_ok
        FROM joined
    ),
    ratios AS (
        SELECT
            trade_date, ticker, market,
            ev_net_share_issuance_yoy, issuance_available_from,
            issuance_identity_ok, issuance_classification_complete,
            cash_dividends_total, dividend_source, buyback_cash_ttm, payout_available_from,
            -- payout_v2: the DPS cross-check upstream only fires when the
            -- filing also carries a DPS row. This is the backstop for the rest
            -- — a yield above PAYOUT_YIELD_MAX means the company returned more
            -- than 10x its own market value in one year, which no filer does;
            -- it is a unit slip the source did not give us the means to
            -- resolve. Rejected, not clipped, for the same reason as upstream.
            CASE WHEN base_ok AND (cash_dividends_total IS NOT NULL OR buyback_cash_ttm IS NOT NULL)
                      AND (COALESCE(cash_dividends_total, 0) + COALESCE(buyback_cash_ttm, 0))
                          / market_cap_pit <= {PAYOUT_YIELD_MAX}
                 THEN (COALESCE(cash_dividends_total, 0) + COALESCE(buyback_cash_ttm, 0))
                      / market_cap_pit
            END AS ev_payout_yield
        FROM scored
    )
    SELECT
        trade_date, ticker, market,
        ev_net_share_issuance_yoy,
        LAG(ev_net_share_issuance_yoy) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS ev_net_share_issuance_yoy_lag1,
        issuance_available_from, issuance_identity_ok, issuance_classification_complete,
        ev_payout_yield,
        LAG(ev_payout_yield) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS ev_payout_yield_lag1,
        cash_dividends_total, dividend_source, buyback_cash_ttm, payout_available_from
    FROM ratios
    """


def register_event_scan_calendar(
    con: duckdb.DuckDBPyConnection,
    trading_days: Sequence[date],
    *,
    table: str = _CAL_TABLE,
) -> str:
    """(Re)create the KRX session table the ``available_from`` lookups read."""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} (d DATE)")
    con.executemany(f"INSERT INTO {table} VALUES (?)", [(d,) for d in trading_days])
    return table


def register_event_scan_daily_view(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: Sequence[date],
    view_name: str = EVENT_SCAN_TABLE,
    **views: str,
) -> str:
    """Register the KRX session calendar table, then a view over the SQL above."""
    register_event_scan_calendar(con, trading_days)
    sql = build_event_scan_daily_sql(**views)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_event_scan_daily(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    trading_days: Sequence[date],
    force: bool = False,
    **views: str,
) -> str:
    """Build + register ``feat_event_scan_daily`` as a cached parquet mart."""
    register_event_scan_calendar(con, trading_days)
    materialize(
        con,
        config,
        EVENT_SCAN_TABLE,
        build_event_scan_daily_sql(**views),
        force=force,
    )
    return register_mart_view(con, config, EVENT_SCAN_TABLE)
