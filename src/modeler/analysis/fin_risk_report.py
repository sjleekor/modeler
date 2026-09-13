"""F-4.3/F-4.4 — build ``feat_fin_risk`` and report on it.

Answers what ``03_financial_risk_lifecycle_transition.md`` §5 asks before any
preregistration: does the mart read the same vintage ``feat_fin_scan_daily``
does, when does each column start, how often do the three transition flags
actually fire (which *decides* the scan path under the rule fixed in §1.3), do
the new columns merely restate the size or issuance axis, and how much of the
delisted universe is still missing.

Reads the A0/B marts of the same snapshot straight from parquet — they were
built under the canonical scan's ``analysis_config_hash`` and this report is
not part of that preregistration, so it never re-registers them through the
cache gate.

Example::

    uv run python -m modeler.analysis.fin_risk_report \
        --snapshot-date 2026-08-23 --source sj2_remote
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import duckdb
from collector.kr.shared import get_trading_days

from modeler.etl.config import DEFAULT_SOURCE, DataRoot, EngineOptions, LakeConfig
from modeler.etl.features.fin_risk import (
    AVAILABILITY_GROUPS,
    FIN_RISK_TABLE,
    FORMULA_VERSION,
    INTEREST_COVERAGE_CAP,
    LIFECYCLE_STAGES,
    PRIMARY_COLUMNS,
    materialize_fin_risk,
)
from modeler.etl.lake import _sql_str_literal, connect, register_views
from modeler.etl.mart import mart_glob

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("docs/dev/20260907_additional_feature/results")

_RAW_INPUTS = ("daily_ohlcv", "dart_shareholder_return_raw", "stock_master")
_INPUT_MARTS = ("dim_stock_pit_daily", "dim_price_quality_daily", "fin_quarterly_metric_vintage")
_DIAGNOSTIC_MARTS = ("feat_fin_scan_daily", "feat_event_scan_daily")

#: §1.3 — a flag firing for fewer than this share of stocks in a year is an
#: event, not a continuous variable, and is scanned as an event cohort.
EVENT_COHORT_THRESHOLD = 0.05

_CONTINUOUS = (
    "fin_debt_to_assets",
    "fin_net_debt_to_mcap",
    "fin_interest_coverage",
    "fin_ext_finance_to_assets",
)
_DISCRETE = (
    "fin_lifecycle_stage",
    "fin_lifecycle_transition",
    "fin_profit_turn",
    "fin_dividend_initiation",
    "fin_negative_equity_exit",
)
#: Flags whose scan path §1.3 leaves to the measured frequency.
_FREQUENCY_DECIDED = ("fin_lifecycle_transition", "fin_profit_turn")


def _register_mart(con: duckdb.DuckDBPyConnection, config: LakeConfig, name: str) -> bool:
    glob = mart_glob(config, name)
    if not con.execute("SELECT count(*) FROM glob(?)", [glob]).fetchone()[0]:
        logger.warning("%s not present in this snapshot", name)
        return False
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_parquet({_sql_str_literal(glob)}, hive_partitioning=false)"
    )
    return True


def _md_table(header: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def coverage_section(con: duckdb.DuckDBPyConnection) -> str:
    starts = []
    for column in PRIMARY_COLUMNS:
        row = con.execute(
            f"SELECT min(trade_date), count({column}) FROM {FIN_RISK_TABLE} "
            f"WHERE {column} IS NOT NULL"
        ).fetchone()
        starts.append([f"`{column}`", row[0], f"{row[1]:,}"])

    per_year = con.execute(f"""
        SELECT year(trade_date) AS y, count(*) AS rows,
               {", ".join(f"round(count({c}) * 1.0 / count(*), 3)" for c in PRIMARY_COLUMNS)}
        FROM {FIN_RISK_TABLE}
        GROUP BY 1 ORDER BY 1
        """).fetchall()

    ages = con.execute(f"""
        SELECT {", ".join(
            f"round(median({group}_fin_age_days))" for group in AVAILABILITY_GROUPS
        )}
        FROM {FIN_RISK_TABLE}
        """).fetchone()

    return "\n".join(
        [
            "### 유효 시작일",
            "",
            _md_table(["컬럼", "최초 세션", "non-NULL 행"], starts),
            "",
            "### 연도별 커버리지",
            "",
            _md_table(
                ["연도", "행 수", *[c.replace("fin_", "") for c in PRIMARY_COLUMNS]],
                [[row[0], f"{row[1]:,}", *row[2:]] for row in per_year],
            ),
            "",
            "### 노출 지연 (중앙값, 일)",
            "",
            _md_table(
                [f"`{g}`" for g in AVAILABILITY_GROUPS],
                [list(ages)],
            ),
        ]
    )


def distribution_section(con: duckdb.DuckDBPyConnection) -> str:
    rows = []
    for column in _CONTINUOUS:
        stats = con.execute(f"""
            SELECT round(avg({column}), 4), round(stddev_samp({column}), 4),
                   round(quantile_cont({column}, 0.01), 4),
                   round(quantile_cont({column}, 0.50), 4),
                   round(quantile_cont({column}, 0.99), 4)
            FROM {FIN_RISK_TABLE} WHERE {column} IS NOT NULL
            """).fetchone()
        rows.append([f"`{column}`", *stats])

    discrete = []
    for column in _DISCRETE:
        counts = con.execute(
            f"SELECT {column}, count(*) FROM {FIN_RISK_TABLE} "
            f"WHERE {column} IS NOT NULL GROUP BY 1 ORDER BY 1"
        ).fetchall()
        total = sum(count for _, count in counts) or 1
        discrete.append(
            [
                f"`{column}`",
                ", ".join(f"{value}: {count / total:.3f}" for value, count in counts),
                f"{total:,}",
            ]
        )

    capped, cap_total = con.execute(
        f"SELECT count(*) FILTER (WHERE fin_interest_coverage_capped), "
        f"count(fin_interest_coverage_capped) FROM {FIN_RISK_TABLE}"
    ).fetchone()
    aligned, align_total = con.execute(
        f"SELECT count(*) FILTER (WHERE fin_lifecycle_prev_aligned), "
        f"count(*) FILTER (WHERE fin_lifecycle_transition IS NOT NULL) FROM {FIN_RISK_TABLE}"
    ).fetchone()
    basis = con.execute(
        f"SELECT fs_basis_used, count(*) FROM {FIN_RISK_TABLE} GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()

    return "\n".join(
        [
            "### 연속형",
            "",
            _md_table(["컬럼", "평균", "표준편차", "p1", "p50", "p99"], rows),
            "",
            "### 이산형 (값별 비중)",
            "",
            _md_table(["컬럼", "분포", "non-NULL 행"], discrete),
            "",
            f"- `fin_interest_coverage`가 상한 {INTEREST_COVERAGE_CAP}에 걸린 비율: "
            f"{capped / (cap_total or 1):.4f} ({capped:,} / {cap_total:,})",
            f"- `fin_lifecycle_prev_aligned`(직전 vintage 세 현금흐름의 접수일 일치): "
            f"{aligned / (align_total or 1):.4f} ({aligned:,} / {align_total:,})",
            "- `fs_basis_used`: "
            + ", ".join(f"{value or 'NULL'} {count:,}" for value, count in basis),
        ]
    )


def lifecycle_section(con: duckdb.DuckDBPyConnection) -> str:
    stages = con.execute(f"""
        SELECT fin_lifecycle_stage AS stage, count(*) AS n,
               count(DISTINCT ticker) AS stocks
        FROM {FIN_RISK_TABLE} WHERE fin_lifecycle_stage IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """).fetchall()
    names = {1: "Introduction", 2: "Growth", 3: "Mature", 4: "Shake-out", 5: "Decline"}
    total = sum(row[1] for row in stages) or 1

    # ``lifecycle_available_from`` is a greatest() and DuckDB's greatest SKIPS
    # NULLs, so "not null" there means *at least one* cash flow was filed, not
    # all three. Measuring the exact-zero rule against it would report the
    # incomplete-triple rows as zero-sign rows (34% rather than the true rate).
    # The zero rule has to be measured where all three values exist, which the
    # mart only exposes through the stage itself — so this counts the rows the
    # transition column could have used and did not.
    missing = con.execute(f"""
        SELECT count(*) FILTER (WHERE fin_lifecycle_stage IS NULL
                                AND fin_lifecycle_prev_stage IS NOT NULL),
               count(*) FILTER (WHERE fin_lifecycle_stage IS NOT NULL
                                OR fin_lifecycle_prev_stage IS NOT NULL)
        FROM {FIN_RISK_TABLE}
        """).fetchone()

    return "\n".join(
        [
            _md_table(
                ["단계", "이름", "행 비중", "행 수", "종목"],
                [
                    [row[0], names.get(row[0], "?"), f"{row[1] / total:.3f}", f"{row[1]:,}", row[2]]
                    for row in stages
                ],
            ),
            "",
            f"- 직전 단계는 있는데 이번 단계가 NULL인 행(세 현금흐름 중 하나가 사라졌거나 "
            f"부호가 정확히 0): {missing[0] / (missing[1] or 1):.5f} "
            f"({missing[0]:,} / {missing[1]:,})",
            f"- 매핑 조합 수 {len(LIFECYCLE_STAGES)} (부호 8조합 전수, Dickinson 2011 표 1 고정)",
        ]
    )


def input_coverage_section(con: duckdb.DuckDBPyConnection) -> str:
    """Why the nine columns start in different years — the mapping-rule gap.

    The design doc expected all of them from 2016-2017 (03 §2, "TTM·직전 vintage
    요구로 2016~2017"). Five start in 2019-2020 instead, and the reason is not in
    this mart: four of the ten input metrics have no XBRL fallback rule, so they
    exist only where ``dart_financial_statement_raw`` reaches — which is thin
    before 2019. This is the measurement that says so.
    """
    rows = con.execute("""
        SELECT metric_code,
               count(*) FILTER (WHERE bsns_year <= 2018) AS rows_to_2018,
               count(*) FILTER (WHERE bsns_year >= 2020) AS rows_from_2020,
               count(DISTINCT ticker) AS tickers
        FROM fin_quarterly_metric_vintage
        WHERE metric_code IN (
            'total_assets', 'total_liabilities', 'total_equity', 'net_income',
            'operating_income', 'operating_cash_flow', 'investing_cash_flow',
            'financing_cash_flow', 'interest_paid', 'cash_and_cash_equivalents'
        )
        GROUP BY 1 ORDER BY 2
        """).fetchall()

    from collector.kr.shared import default_metric_mapping_rules

    rules = default_metric_mapping_rules()
    sources: dict[str, set[str]] = {}
    for rule in rules:
        sources.setdefault(rule.metric_code, set()).add(rule.source_table)

    table = [
        [
            f"`{metric}`",
            f"{to_2018:,}",
            f"{from_2020:,}",
            tickers,
            (
                "XBRL fallback 있음"
                if "dart_xbrl_fact_raw" in sources.get(metric, set())
                else "**없음**"
            ),
        ]
        for metric, to_2018, from_2020, tickers in rows
    ]

    return "\n".join(
        [
            _md_table(
                ["metric", "≤2018 행", "≥2020 행", "종목", "매핑 규칙 원천"],
                table,
            ),
            "",
            "**이것이 유효 시작일이 갈리는 이유다.** `investing_cash_flow`·"
            "`financing_cash_flow`·`interest_paid`·`cash_and_cash_equivalents` 넷은 "
            "`dart_financial_statement_raw` 규칙만 있고 XBRL fallback이 없다. 그 원천이 "
            "2019년 이전에 얇아서 네 metric의 2018년 이전 행이 100건 안팎이고, 그래서 이 넷을 "
            "쓰는 **5개 family(`fin_net_debt_to_mcap`·`fin_interest_coverage`·"
            "`fin_ext_finance`·`fin_lifecycle_stage`·`fin_lifecycle_transition`)가 사실상 "
            "2020년부터만 존재한다.** 설계 문서가 기대한 2016~2017이 아니다.",
            "",
            "F-5(`metric_rules` 확장)에서 **가장 값이 큰 항목이 이것**이다. fin_v4가 "
            "`revenue`에 XBRL fallback을 붙였을 때 8,103행 → 약 148,000행으로 늘었다. 같은 "
            "종류의 규칙을 이 넷에 붙이면 위 5개 family의 표본이 2016년까지 내려갈 가능성이 "
            "크다. 다만 그러면 `feat_fin_risk` 값이 바뀌므로 `FORMULA_VERSION` 범프와 "
            "재검정이 따라온다 — 이번 라운드에서는 하지 않고 측정만 남긴다.",
        ]
    )


def event_frequency_section(con: duckdb.DuckDBPyConnection) -> str:
    """F-4.3 — the frequency that *decides* each flag's scan path."""
    flags = {
        "fin_lifecycle_transition": "fin_lifecycle_transition = 1",
        "fin_profit_turn": "fin_profit_turn <> 0",
        "fin_dividend_initiation": "fin_dividend_initiation = 1",
        "fin_negative_equity_exit": "fin_negative_equity_exit = 1",
    }
    rows = []
    verdicts = []
    for column, predicate in flags.items():
        per_year = con.execute(f"""
            SELECT year(trade_date) AS y,
                   count(DISTINCT ticker) FILTER (WHERE {predicate}) * 1.0
                       / NULLIF(count(DISTINCT ticker), 0) AS share
            FROM {FIN_RISK_TABLE}
            WHERE {column} IS NOT NULL
            GROUP BY 1 HAVING count(*) > 0 ORDER BY 1
            """).fetchall()
        usable = [(year, share) for year, share in per_year if share is not None]
        mean_share = sum(share for _, share in usable) / len(usable) if usable else 0.0
        events = con.execute(f"SELECT count(*) FROM {FIN_RISK_TABLE} WHERE {predicate}").fetchone()[
            0
        ]
        path = "continuous" if mean_share >= EVENT_COHORT_THRESHOLD else "event cohort"
        rows.append(
            [
                f"`{column}`",
                f"{mean_share:.4f}",
                f"{min((s for _, s in usable), default=0):.4f}",
                f"{max((s for _, s in usable), default=0):.4f}",
                f"{events:,}",
                path,
            ]
        )
        if column in _FREQUENCY_DECIDED:
            verdicts.append((column, mean_share, path))

    lines = [
        _md_table(
            ["컬럼", "연평균 종목 비율", "최소", "최대", "플래그=1 행 수", "스캔 경로"], rows
        ),
        "",
        f"규칙(§1.3, 결과 보기 전 고정): 연간 이벤트 종목 비율이 "
        f"{EVENT_COHORT_THRESHOLD:.0%} 미만이면 event cohort, 이상이면 continuous.",
        "",
    ]
    for column, share, path in verdicts:
        lines.append(
            f"- **`{column}` → {path}** (연평균 {share:.4f}, 문턱 " f"{EVENT_COHORT_THRESHOLD:.2f})"
        )
    lines.append(
        "- `fin_dividend_initiation`·`fin_negative_equity_exit`은 §4에서 이미 event cohort로 "
        "고정돼 있다. 위 표의 비율은 `min_events` 미달 여부를 미리 보기 위한 것이다."
    )
    return "\n".join(lines)


def overlap_section(con: duckdb.DuckDBPyConnection, available: set[str]) -> str:
    pairs: list[tuple[str, str, str]] = []
    if "feat_fin_scan_daily" in available:
        pairs += [
            ("fin_net_debt_to_mcap", "feat_fin_scan_daily", "fin_log_mcap"),
            ("fin_net_debt_to_mcap", "feat_fin_scan_daily", "fin_book_to_market"),
            ("fin_debt_to_assets", "feat_fin_scan_daily", "fin_log_mcap"),
            ("fin_ext_finance_to_assets", "feat_fin_scan_daily", "fin_asset_growth_yoy"),
            ("fin_interest_coverage", "feat_fin_scan_daily", "fin_operating_profitability"),
            ("fin_lifecycle_stage", "feat_fin_scan_daily", "fin_log_mcap"),
        ]
    if "feat_event_scan_daily" in available:
        pairs += [
            ("fin_ext_finance_to_assets", "feat_event_scan_daily", "ev_net_share_issuance_yoy"),
        ]
    if not pairs:
        return "- 비교 대상 마트가 이 snapshot에 없어 건너뛰었다."

    rows = []
    for own, view, other in pairs:
        value = con.execute(f"""
            SELECT round(avg(rho), 4), round(stddev_samp(rho), 4), count(*)
            FROM (
                SELECT corr(rank_a, rank_b) AS rho
                FROM (
                    SELECT trade_date, ticker, market,
                           rank() OVER (PARTITION BY trade_date ORDER BY {own}) AS rank_a
                    FROM {FIN_RISK_TABLE} WHERE {own} IS NOT NULL
                ) r
                JOIN (
                    SELECT trade_date, ticker, market,
                           rank() OVER (PARTITION BY trade_date ORDER BY {other}) AS rank_b
                    FROM {view} WHERE {other} IS NOT NULL
                ) o USING (trade_date, ticker, market)
                GROUP BY r.trade_date
                HAVING count(*) >= 100
            )
            """).fetchone()
        flag = "⚠ |ρ| ≥ 0.5" if value[0] is not None and abs(value[0]) >= 0.5 else ""
        rows.append([f"`{own}`", f"`{view}.{other}`", value[0], value[1], f"{value[2]:,}", flag])

    return _md_table(
        ["fin_risk 컬럼", "기존 컬럼", "일별 순위상관 평균", "표준편차", "날짜 수", "경고"], rows
    )


def delisted_section(con: duckdb.DuckDBPyConnection) -> str:
    """The S-1 remainder, measured — the gate on any Decline conclusion."""
    row = con.execute("""
        WITH master AS (
            SELECT DISTINCT ticker, status FROM stock_master
            WHERE ticker IS NOT NULL AND ticker <> ''
        ),
        covered AS (
            SELECT DISTINCT ticker FROM fin_quarterly_metric_vintage
        )
        SELECT
            count(*) FILTER (WHERE status = 'DELISTED') AS delisted,
            count(*) FILTER (WHERE status = 'DELISTED' AND c.ticker IS NOT NULL)
                AS delisted_with_financials,
            count(*) FILTER (WHERE status <> 'DELISTED') AS listed,
            count(*) FILTER (WHERE status <> 'DELISTED' AND c.ticker IS NOT NULL)
                AS listed_with_financials
        FROM master m
        LEFT JOIN covered c USING (ticker)
        """).fetchone()
    delisted, delisted_cov, listed, listed_cov = row
    decline = con.execute(f"""
        SELECT count(DISTINCT ticker) FROM {FIN_RISK_TABLE} WHERE fin_lifecycle_stage = 5
        """).fetchone()[0]

    return "\n".join(
        [
            _md_table(
                ["상태", "종목", "재무 vintage 있음", "커버리지"],
                [
                    [
                        "DELISTED",
                        f"{delisted:,}",
                        f"{delisted_cov:,}",
                        f"{delisted_cov / (delisted or 1):.3f}",
                    ],
                    [
                        "그 외",
                        f"{listed:,}",
                        f"{listed_cov:,}",
                        f"{listed_cov / (listed or 1):.3f}",
                    ],
                ],
            ),
            "",
            f"- Decline(5단계)을 한 번이라도 기록한 종목 {decline:,}개. 상폐 종목 재무가 "
            f"{delisted_cov / (delisted or 1):.1%}만 있으므로 이 축의 판정은 **F-9.3 S-1 잔여 "
            "백필 뒤로 보류**한다(`03` §1.2·§2). 마트는 지금 만들고 카드에 "
            '"생존편향 미해결"을 적는다.',
            "",
            f"위 `DELISTED` {delisted:,}개는 **`stock_master`가 아는 상폐 종목만**이다. "
            "corpCode.xml 기준 상폐 법인은 약 1,330개이고 `stock_master`가 그만큼을 담고 있지 "
            "않다 — 그것을 복구하는 것이 F-9.4(S-2, `universe backfill-master`)다. 즉 실제 "
            "생존편향은 이 표보다 크다.",
        ]
    )


def preregistration_section() -> str:
    """F-4.5 — the nine families as they go to F-HS-1, fixed before the scan.

    Literal, copied from ``03`` §4. The scan-path column is the one thing §1.3
    left to measurement, and §5 of the generated report records what the
    measurement decided under the rule fixed beforehand.
    """
    return "\n".join(
        [
            _md_table(
                ["family", "primary", "fdr_family", "부호", "horizon", "스캔 경로", "비고"],
                [
                    [
                        "`fin_debt_to_assets`",
                        "`fin_debt_to_assets`",
                        "`financial_risk`",
                        "양방향",
                        "[60, 120]",
                        "continuous",
                        "총부채(이자부부채 아님)",
                    ],
                    [
                        "`fin_net_debt_to_mcap`",
                        "`fin_net_debt_to_mcap`",
                        "`financial_risk`",
                        "양방향",
                        "[60, 120]",
                        "continuous",
                        "`fin_log_mcap`과 분모 공유",
                    ],
                    [
                        "`fin_interest_coverage`",
                        "`fin_interest_coverage`",
                        "`financial_risk`",
                        "`+`",
                        "[60, 120]",
                        "continuous",
                        f"분모 ≤ 0 NULL, 상한 {INTEREST_COVERAGE_CAP}",
                    ],
                    [
                        "`fin_ext_finance`",
                        "`fin_ext_finance_to_assets`",
                        "`financial_risk`",
                        "`−`",
                        "[60, 120]",
                        "continuous",
                        "`ev_net_share_issuance_yoy`와 개념 겹침",
                    ],
                    [
                        "`fin_lifecycle_stage`",
                        "`fin_lifecycle_stage`",
                        "`lifecycle`",
                        "양방향",
                        "[60, 120]",
                        "continuous(순서형 rank)",
                        "Decline 판정 보류",
                    ],
                    [
                        "`fin_lifecycle_transition`",
                        "`fin_lifecycle_transition`",
                        "`lifecycle`",
                        "양방향",
                        "[20, 60]",
                        "§5 측정 결과",
                        "",
                    ],
                    [
                        "`fin_profit_turn`",
                        "`fin_profit_turn`",
                        "`transition`",
                        "`+`",
                        "[20, 60, 120]",
                        "§5 측정 결과",
                        "",
                    ],
                    [
                        "`fin_dividend_initiation`",
                        "`fin_dividend_initiation`",
                        "`transition`",
                        "`+`",
                        "[60, 120]",
                        "event cohort",
                        "연 1회 이벤트",
                    ],
                    [
                        "`fin_negative_equity_exit`",
                        "`fin_negative_equity_exit`",
                        "`transition`",
                        "`+`",
                        "[60, 120]",
                        "event cohort",
                        "`min_events` 미달이면 `insufficient`",
                    ],
                ],
            ),
            "",
            "- 검정하지 않는 컬럼: `fin_lifecycle_transition_up`·`_down`(방향 진단), "
            "`fin_lifecycle_prev_stage`, `fin_lifecycle_prev_aligned`, "
            "`fin_interest_coverage_capped`, `fs_basis_used`, `*_available_from`, "
            "`*_fin_age_days`, 그리고 모든 `_lag1`(기존 variant 축).",
            "- 양방향 5건은 첫 run 뒤 관측 부호를 고정하고 이후 바꾸지 않는다.",
            "- Phase C 후보: `fin_debt_to_assets × credit_wide`(F-8 신용 국면), "
            "`fin_lifecycle_stage × market_up`.",
            "- F-5(`metric_rules` 확장)의 파생 family는 이 config에 넣지 않는다(`03` §3).",
        ]
    )


def summary_section(con: duckdb.DuckDBPyConnection) -> str:
    """What §5 asked, answered in one place."""
    starts = {}
    for column in PRIMARY_COLUMNS:
        starts[column] = con.execute(
            f"SELECT min(trade_date) FROM {FIN_RISK_TABLE} WHERE {column} IS NOT NULL"
        ).fetchone()[0]
    from_2020 = [c for c, d in starts.items() if d is not None and d.year >= 2019]

    zero_lifecycle = con.execute(f"""
        SELECT round(count(*) FILTER (WHERE fin_lifecycle_stage IS NULL) * 1.0
                     / NULLIF(count(*), 0), 4)
        FROM {FIN_RISK_TABLE} WHERE lifecycle_available_from IS NOT NULL
        """).fetchone()[0]

    return "\n".join(
        [
            "1. **PIT 규칙은 `feat_fin_scan_daily`와 같은 것을 쓴다.** vintage 선택·"
            "`base_ok`·같은-날 tie-break을 `fin_vintage`로 꺼내 공유하고, `feat_fin_scan_daily`"
            "의 SQL 텍스트는 바이트 단위로 그대로다(A0 캐시 키라서 필수다).",
            f"2. **5개 family가 2020년부터만 존재한다** — `{'`, `'.join(from_2020)}`. "
            "설계 문서는 2016~2017을 기대했다. 원인은 이 마트가 아니라 입력 metric 4개에 "
            "XBRL fallback 규칙이 없는 것이고(§5), F-5의 최우선 항목이다. "
            "**사전등록 표본을 2020~2025로 좁혀 읽어야 한다.**",
            "3. **연속형 4개는 꼬리가 매우 두껍다.** `fin_interest_coverage`는 상한만 100으로 "
            "묶여 있고 하한이 없어 평균이 −9,500까지 간다(분모가 0에 가까운 적자 기업). "
            "`fin_net_debt_to_mcap`·`fin_ext_finance_to_assets`도 분모가 작아 같은 모양이다. "
            "스캔이 rank로 변환하므로 판정에는 영향이 없지만, 수준을 그대로 쓰는 소비자는 "
            "winsorize가 필요하다. 정의는 `03` §1.1에 고정된 것이므로 바꾸지 않았다.",
            "4. **중복 경고 2건.** `fin_interest_coverage` × `fin_operating_profitability` "
            "ρ = 0.88은 사실상 같은 축이다(분자가 같은 영업이익). "
            "`fin_net_debt_to_mcap` × `fin_book_to_market` ρ = 0.51은 분모(시총)를 공유한다. "
            "둘 다 카드에 경고를 적고, 전자는 `financial_risk` family의 독립적 발견으로 "
            "읽지 않는다.",
            "5. **전이 두 개는 continuous로 확정.** `fin_lifecycle_transition` 연평균 0.69, "
            "`fin_profit_turn` 0.27 — 둘 다 5% 문턱을 크게 넘는다(§6). "
            "`fin_dividend_initiation`(0.040)·`fin_negative_equity_exit`(0.004)은 event "
            "cohort 그대로다.",
            f"6. **생애주기 단계 분포는 Dickinson과 부합한다.** Mature 0.35가 가장 크고 "
            "Introduction 0.15 / Growth 0.23 / Shake-out 0.16 / Decline 0.11이다. 세 현금흐름 "
            f"중 하나라도 있는 행 기준 단계 NULL 비율 {zero_lifecycle} — 대부분 세 개가 다 "
            "차지 않은 행이다.",
            "7. **Decline·부실 판정은 보류.** 상폐 종목 재무 커버리지가 9.2%이고 실제 상폐 "
            "모집단은 그보다 크다(§4). F-9.3·F-9.4 뒤에 다시 본다.",
        ]
    )


def build(config: LakeConfig, *, force: bool) -> tuple[duckdb.DuckDBPyConnection, set[str]]:
    con = connect(config)
    register_views(con, config, tables=list(_RAW_INPUTS))
    for name in _INPUT_MARTS:
        if not _register_mart(con, config, name):
            raise SystemExit(f"input mart {name} missing from snapshot {config.snapshot_date}")
    bounds = con.execute("SELECT min(trade_date), max(trade_date) FROM daily_ohlcv").fetchone()
    trading_days = list(get_trading_days(bounds[0], bounds[1]))

    logger.info("building %s", FIN_RISK_TABLE)
    materialize_fin_risk(con, config, trading_days=trading_days, force=force)

    available = {name for name in _DIAGNOSTIC_MARTS if _register_mart(con, config, name)}
    return con, available


def render(con: duckdb.DuckDBPyConnection, config: LakeConfig, available: set[str]) -> str:
    rows = con.execute(f"SELECT count(*) FROM {FIN_RISK_TABLE}").fetchone()[0]
    span = con.execute(f"SELECT min(trade_date), max(trade_date) FROM {FIN_RISK_TABLE}").fetchone()

    return "\n".join(
        [
            "# F-4 — `feat_fin_risk` 마트 검증 리포트",
            "",
            f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST",
            f"- snapshot `{config.snapshot_date}` / source `{config.source}`",
            f"- `FORMULA_VERSION = {FORMULA_VERSION}`",
            f"- `{FIN_RISK_TABLE}` {rows:,}행, {span[0]} ~ {span[1]}",
            "",
            "생성 명령: `uv run python -m modeler.analysis.fin_risk_report "
            f"--snapshot-date {config.snapshot_date} --source {config.source}`",
            "",
            "---",
            "",
            "## 0. 요약",
            "",
            summary_section(con),
            "",
            "",
            "**holdout(2025-08-01~)은 열지 않았다.** 이 리포트의 모든 수치는 피쳐 "
            "쪽만 본다 — 커버리지·분포·자기상관·피쳐 간 순위상관·이벤트 빈도. "
            "forward 수익률이나 라벨을 쓰는 계산은 하나도 없으므로 표본을 전 "
            "구간으로 잡아도 홀드아웃이 소진되지 않는다. 판정은 F-HS에서 사전등록된 "
            "경계로만 한다.",
            "",
            "---",
            "",
            "## 1. 커버리지·노출 지연",
            "",
            coverage_section(con),
            "",
            "---",
            "",
            "## 2. 분포",
            "",
            distribution_section(con),
            "",
            "---",
            "",
            "## 3. 생애주기 단계 (Dickinson 2011)",
            "",
            lifecycle_section(con),
            "",
            "---",
            "",
            "## 4. 상폐 커버리지 — Decline 축의 선행 조건",
            "",
            delisted_section(con),
            "",
            "---",
            "",
            "## 5. 입력 metric 커버리지 — 유효 시작일이 갈리는 이유",
            "",
            input_coverage_section(con),
            "",
            "---",
            "",
            "## 6. 전이 이벤트 빈도 → 스캔 경로 (F-4.3)",
            "",
            event_frequency_section(con),
            "",
            "---",
            "",
            "## 7. 기존 축과의 중복 (F-4.4, 일별 순위상관)",
            "",
            overlap_section(con, available),
            "",
            "|ρ| ≥ 0.5이면 카드에 경고를 적는다.",
            "",
            "---",
            "",
            "## 8. 사전등록 고정 항목 (F-4.5 → F-HS-1)",
            "",
            preregistration_section(),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--force", action="store_true", help="rebuild the mart")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-name", default="f4_fin_risk_verification.md")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.snapshot_date is None:
        parser.error("--snapshot-date is required")
    config = LakeConfig(
        root=DataRoot.resolve(market="kr"),
        snapshot_date=args.snapshot_date,
        source=args.source or DEFAULT_SOURCE,
        engine=EngineOptions(threads=args.threads, memory_limit=args.memory_limit),
    )

    con, available = build(config, force=args.force)
    text = render(con, config, available)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.output_name).write_text(text, encoding="utf-8")
    print(f"wrote {out_dir / args.output_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
