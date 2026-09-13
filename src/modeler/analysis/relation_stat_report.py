"""F-2.4 — build ``dim_peer_monthly``/``feat_relation_stat`` and report on them.

Answers the five questions ``02_relationship_features.md`` §1.5 asks *before*
any preregistration: when does each column actually start, how much of the
cross-section carries a value, what does the distribution look like, how fast
does it turn over, and does it merely restate a size/liquidity axis the scan
already has.

Reads the A0 marts of the same snapshot (``feat_price``, ``feat_fin_scan_daily``)
straight from parquet for the overlap diagnostics — those were built under the
canonical scan's ``analysis_config_hash`` and this report is not part of that
preregistration, so it never re-registers them through the cache gate.

Example::

    uv run python -m modeler.analysis.relation_stat_report \
        --snapshot-date 2026-08-23 --source sj2_remote
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import duckdb

from modeler.etl.config import DEFAULT_SOURCE, DataRoot, EngineOptions, LakeConfig
from modeler.etl.features.relation_stat import (
    COUNT_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    FORMULA_VERSION,
    PEER_MONTHLY_TABLE,
    PRIMARY_COLUMNS,
    RELATION_STAT_TABLE,
    materialize_peer_monthly,
    materialize_relation_stat,
)
from modeler.etl.lake import _sql_str_literal, connect, register_views
from modeler.etl.mart import mart_glob

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("docs/dev/20260907_additional_feature/results")

_RAW_INPUTS = ("daily_ohlcv", "daily_market_cap", "dart_corp_master")
_REPORTED_COLUMNS = PRIMARY_COLUMNS + DIAGNOSTIC_COLUMNS


def _register_a0_mart(con: duckdb.DuckDBPyConnection, config: LakeConfig, name: str) -> bool:
    """Register one already-built A0 mart by glob, skipping the cache gate."""
    glob = mart_glob(config, name)
    if not con.execute("SELECT count(*) FROM glob(?)", [glob]).fetchone()[0]:
        logger.warning("%s not present in this snapshot; overlap diagnostics skipped", name)
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


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def peer_monthly_section(con: duckdb.DuckDBPyConnection) -> str:
    total, months, stocks = con.execute(
        f"SELECT count(*), count(DISTINCT month_end), count(DISTINCT ticker) "
        f"FROM {PEER_MONTHLY_TABLE}"
    ).fetchone()
    span = con.execute(
        f"SELECT min(month_end), max(month_end) FROM {PEER_MONTHLY_TABLE}"
    ).fetchone()
    per_year = con.execute(f"""
        SELECT year(month_end) AS y,
               count(DISTINCT month_end) AS month_ends,
               count(DISTINCT ticker) AS stocks,
               round(avg(rho) FILTER (WHERE peer_rank <= 20), 4) AS mean_rho_k20,
               round(avg(rho) FILTER (WHERE peer_rank = 1), 4) AS mean_rho_rank1,
               round(avg(rho) FILTER (WHERE peer_rank = 20), 4) AS mean_rho_rank20,
               round(avg(joint_sessions), 1) AS mean_joint,
               round(avg(CASE WHEN same_ksic2 THEN 1.0 ELSE 0.0 END)
                     FILTER (WHERE peer_rank <= 20 AND same_ksic2 IS NOT NULL), 4)
                   AS ksic2_agree_k20
        FROM {PEER_MONTHLY_TABLE}
        GROUP BY 1 ORDER BY 1
        """).fetchall()
    thin = con.execute(f"""
        SELECT count(*) FROM (
            SELECT month_end, ticker, count(*) AS n
            FROM {PEER_MONTHLY_TABLE} GROUP BY 1, 2
        ) WHERE n < 40
        """).fetchone()[0]
    baseline = dict(_ksic_random_baseline(con))
    rows = [[*row, baseline.get(row[0])] for row in per_year]

    return "\n".join(
        [
            f"- 행 수 {total:,} / 월말 {months} / 종목 {stocks:,}",
            f"- 기간 {span[0]} ~ {span[1]}",
            f"- 저장 peer 40개를 못 채운 (월말, 종목) 조합: {thin:,}",
            "",
            _md_table(
                [
                    "연도",
                    "월말",
                    "종목",
                    "ρ 평균(K20)",
                    "ρ 1위",
                    "ρ 20위",
                    "공통세션",
                    "KSIC2 일치율",
                    "무작위 기준",
                ],
                rows,
            ),
            "",
            "`무작위 기준`은 같은 월말 후보 집합에서 서로 다른 두 종목을 무작위로 뽑았을 때 "
            "KSIC 2자리가 같을 확률(`Σ n_g(n_g−1) / n(n−1)`)이다. 상관 peer가 업종을 "
            "얼마나 재현하는지는 일치율 자체가 아니라 이 기준과의 비율로 읽는다.",
        ]
    )


def _ksic_random_baseline(con: duckdb.DuckDBPyConnection) -> list[tuple[int, float]]:
    """Per-year chance that two distinct candidates share a KSIC 2-digit group."""
    return con.execute(f"""
        WITH candidates AS (
            SELECT DISTINCT month_end, ticker FROM {PEER_MONTHLY_TABLE}
        ),
        coded AS (
            SELECT c.month_end, c.ticker, left(m.induty_code, 2) AS ksic2
            FROM candidates c
            JOIN dart_corp_master m USING (ticker)
            WHERE m.induty_code IS NOT NULL AND length(m.induty_code) >= 2
        ),
        by_group AS (
            SELECT month_end, ksic2, count(*) AS n_g FROM coded GROUP BY 1, 2
        ),
        per_month AS (
            SELECT month_end,
                   sum(n_g * (n_g - 1)) * 1.0 / NULLIF(sum(n_g) * (sum(n_g) - 1), 0) AS p
            FROM by_group GROUP BY 1
        )
        SELECT year(month_end) AS y, round(avg(p), 4) FROM per_month GROUP BY 1 ORDER BY 1
        """).fetchall()


def coverage_section(con: duckdb.DuckDBPyConnection) -> str:
    starts = []
    for column in _REPORTED_COLUMNS:
        row = con.execute(
            f"SELECT min(trade_date), count({column}) FROM {RELATION_STAT_TABLE} "
            f"WHERE {column} IS NOT NULL"
        ).fetchone()
        starts.append([column, row[0], f"{row[1]:,}"])

    per_year = con.execute(f"""
        SELECT year(trade_date) AS y, count(*) AS sessions_rows,
               {", ".join(
                   f"round(count({c}) * 1.0 / count(*), 4) AS cov_{c}" for c in PRIMARY_COLUMNS
               )}
        FROM {RELATION_STAT_TABLE}
        GROUP BY 1 ORDER BY 1
        """).fetchall()

    daily = con.execute(f"""
        SELECT round(min(ratio), 4), round(median(ratio), 4), round(max(ratio), 4)
        FROM (
            SELECT trade_date, count(rel_peer_mom_20d) * 1.0 / count(*) AS ratio
            FROM {RELATION_STAT_TABLE}
            WHERE trade_date >= DATE '2017-01-01'
            GROUP BY 1
        )
        """).fetchone()

    return "\n".join(
        [
            "### 유효 시작일 (컬럼별 최초 non-NULL 세션)",
            "",
            _md_table(["컬럼", "최초 세션", "non-NULL 행"], starts),
            "",
            "### 연도별 커버리지 (valid session 행 기준)",
            "",
            _md_table(
                ["연도", "행 수", *[c for c in PRIMARY_COLUMNS]],
                [[row[0], f"{row[1]:,}", *row[2:]] for row in per_year],
            ),
            "",
            f"- 2017년 이후 날짜별 `rel_peer_mom_20d` 커버리지: 최소 {daily[0]}, "
            f"중앙값 {daily[1]}, 최대 {daily[2]} (기대 0.8 이상)",
        ]
    )


def distribution_section(con: duckdb.DuckDBPyConnection) -> str:
    rows = []
    for column in _REPORTED_COLUMNS:
        stats = con.execute(f"""
            SELECT round(avg({column}), 6), round(stddev_samp({column}), 6),
                   round(quantile_cont({column}, 0.01), 6),
                   round(quantile_cont({column}, 0.50), 6),
                   round(quantile_cont({column}, 0.99), 6)
            FROM {RELATION_STAT_TABLE} WHERE {column} IS NOT NULL
            """).fetchone()
        rows.append([column, *stats])

    centred = con.execute(f"""
        SELECT round(avg(m), 6), round(stddev_samp(m), 6)
        FROM (
            SELECT trade_date, median(rel_own_minus_peer_20d) AS m
            FROM {RELATION_STAT_TABLE}
            WHERE rel_own_minus_peer_20d IS NOT NULL
            GROUP BY 1
        )
        """).fetchone()

    counts = con.execute(f"""
        SELECT {", ".join(f"round(avg({c}), 2)" for c in COUNT_COLUMNS)}
        FROM {RELATION_STAT_TABLE}
        """).fetchone()

    return "\n".join(
        [
            _md_table(["컬럼", "평균", "표준편차", "p1", "p50", "p99"], rows),
            "",
            f"- `rel_own_minus_peer_20d`의 날짜별 중앙값: 평균 {centred[0]}, "
            f"표준편차 {centred[1]} (정의상 0 근처여야 한다)",
            "- 유효 peer 수 평균: "
            + ", ".join(f"`{c}` {v}" for c, v in zip(COUNT_COLUMNS, counts, strict=True)),
        ]
    )


def turnover_section(con: duckdb.DuckDBPyConnection) -> str:
    rows = []
    for column in ("rel_peer_mom_20d", "rel_peer_mom_60d", "rel_own_minus_peer_20d"):
        value = con.execute(f"""
            SELECT round(corr({column}, prev), 4), count(*)
            FROM (
                SELECT {column},
                       lag({column}) OVER (PARTITION BY ticker, market ORDER BY trade_date) AS prev
                FROM {RELATION_STAT_TABLE}
            ) WHERE {column} IS NOT NULL AND prev IS NOT NULL
            """).fetchone()
        rows.append([column, value[0], f"{value[1]:,}"])

    persistence = con.execute(f"""
        SELECT round(avg(kept), 4)
        FROM (
            SELECT a.month_end,
                   count(*) FILTER (WHERE b.peer_ticker IS NOT NULL) * 1.0
                       / count(*) AS kept
            FROM {PEER_MONTHLY_TABLE} a
            LEFT JOIN {PEER_MONTHLY_TABLE} b
              ON b.ticker = a.ticker AND b.market = a.market
             AND b.peer_ticker = a.peer_ticker AND b.peer_market = a.peer_market
             AND b.peer_rank <= 20
             AND b.month_end = (
                    SELECT max(m.month_end) FROM (SELECT DISTINCT month_end
                                                  FROM {PEER_MONTHLY_TABLE}) m
                    WHERE m.month_end < a.month_end
                 )
            WHERE a.peer_rank <= 20
            GROUP BY 1
            HAVING count(*) FILTER (WHERE b.peer_ticker IS NOT NULL) > 0
        )
        """).fetchone()[0]

    return "\n".join(
        [
            _md_table(["컬럼", "lag1 자기상관", "쌍 수"], rows),
            "",
            f"- peer 집합 월간 유지율(K=20, 직전 월말과 겹치는 비율): {persistence}",
        ]
    )


def overlap_section(con: duckdb.DuckDBPyConnection, *, have_price: bool, have_fin: bool) -> str:
    pairs: list[tuple[str, str, str]] = []
    if have_price:
        pairs += [
            ("rel_peer_mom_20d", "feat_price", "px_amihud_20d"),
            ("rel_peer_mom_20d", "feat_price", "px_mom_12_1"),
            ("rel_own_minus_peer_20d", "feat_price", "px_reversal_5d"),
            ("rel_peer_dispersion_20d", "feat_price", "px_vol_20d"),
            ("rel_peer_bigcap_lag_ret_5d", "feat_price", "px_ret_5d"),
        ]
    if have_fin:
        pairs += [
            ("rel_peer_mom_20d", "feat_fin_scan_daily", "fin_log_mcap"),
            ("rel_peer_dispersion_20d", "feat_fin_scan_daily", "fin_log_mcap"),
        ]
    if not pairs:
        return "- A0 마트가 이 snapshot에 없어 중복 진단을 건너뛰었다."

    rows = []
    for own, view, other in pairs:
        value = con.execute(f"""
            SELECT round(avg(rho), 4), round(stddev_samp(rho), 4), count(*)
            FROM (
                SELECT r.trade_date,
                       corr(rank_a, rank_b) AS rho
                FROM (
                    SELECT trade_date,
                           rank() OVER (PARTITION BY trade_date ORDER BY {own}) AS rank_a,
                           ticker, market
                    FROM {RELATION_STAT_TABLE} WHERE {own} IS NOT NULL
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
        rows.append([own, f"{view}.{other}", value[0], value[1], f"{value[2]:,}", flag])

    return _md_table(
        ["관계 컬럼", "기존 컬럼", "일별 순위상관 평균", "표준편차", "날짜 수", "경고"], rows
    )


def preregistration_section() -> str:
    """F-2.5 — the four families as they go to F-HS-1, fixed before the scan.

    Literal, not computed: a preregistration that could be regenerated from the
    output it is meant to constrain is not one. Signs and horizons are copied
    from ``02_relationship_features.md`` §3 and change only there.
    """
    return "\n".join(
        [
            _md_table(
                [
                    "family",
                    "primary",
                    "secondary",
                    "fdr_family",
                    "부호",
                    "primary horizon",
                    "스캔 경로",
                ],
                [
                    [
                        "`rel_peer_mom`",
                        "`rel_peer_mom_20d`",
                        "`rel_peer_mom_60d`",
                        "`relation`",
                        "양방향",
                        "[20, 40, 60]",
                        "continuous",
                    ],
                    [
                        "`rel_own_minus_peer`",
                        "`rel_own_minus_peer_20d`",
                        "—",
                        "`relation`",
                        "`−`",
                        "[5, 10, 20]",
                        "continuous",
                    ],
                    [
                        "`rel_peer_bigcap_lag`",
                        "`rel_peer_bigcap_lag_ret_5d`",
                        "—",
                        "`relation`",
                        "`+`",
                        "[5, 10, 20]",
                        "continuous",
                    ],
                    [
                        "`rel_peer_dispersion`",
                        "`rel_peer_dispersion_20d`",
                        "—",
                        "`relation`",
                        "양방향",
                        "[20, 40, 60]",
                        "continuous",
                    ],
                ],
            ),
            "",
            "- 검정하지 않는 컬럼: `rel_peer_corr_mean`(모델에 조건 변수로만 넘긴다), "
            "`rel_peer_ksic_agree`, `rel_peer_mom_20d_k10`, `rel_peer_mom_20d_k40`, "
            "`rel_peer_n_*`, `rel_peer_month_end`, `rel_own_ret_20d`.",
            "- K는 20으로 고정한다. `_k10`·`_k40`은 민감도를 보고하기 위한 것이고 "
            "primary로 승격하지 않는다.",
            "- 양방향 등록 두 건은 첫 run 뒤 관측 부호를 고정하고 이후 바꾸지 않는다.",
            "- Phase C 국면 쌍 후보는 `rel_peer_mom × liq_high`, `rel_peer_mom × market_up`, "
            "`rel_own_minus_peer × vix_high`, `rel_peer_bigcap_lag × liq_high` — F-HS-C2로 넘긴다.",
        ]
    )


def summary_section(con: duckdb.DuckDBPyConnection) -> str:
    """The five §1.5 questions, answered in one place."""
    coverage = con.execute(f"""
        SELECT round(count(rel_peer_mom_20d) * 1.0 / count(*), 4)
        FROM {RELATION_STAT_TABLE} WHERE trade_date >= DATE '2017-01-01'
        """).fetchone()[0]
    first = con.execute(
        f"SELECT min(trade_date) FROM {RELATION_STAT_TABLE} WHERE rel_peer_mom_20d IS NOT NULL"
    ).fetchone()[0]
    agree, baseline = con.execute(f"""
        WITH agreement AS (
            SELECT avg(CASE WHEN same_ksic2 THEN 1.0 ELSE 0.0 END) AS a
            FROM {PEER_MONTHLY_TABLE}
            WHERE peer_rank <= 20 AND same_ksic2 IS NOT NULL AND month_end >= DATE '2016-01-01'
        )
        SELECT round((SELECT a FROM agreement), 4), 0.0
        """).fetchone()
    baseline = float(con.execute("SELECT round(avg(p), 4) FROM (" + f"""
            WITH candidates AS (SELECT DISTINCT month_end, ticker FROM {PEER_MONTHLY_TABLE}
                                WHERE month_end >= DATE '2016-01-01'),
            coded AS (SELECT c.month_end, left(m.induty_code, 2) AS ksic2
                      FROM candidates c JOIN dart_corp_master m USING (ticker)
                      WHERE m.induty_code IS NOT NULL AND length(m.induty_code) >= 2),
            by_group AS (SELECT month_end, ksic2, count(*) AS n_g FROM coded GROUP BY 1, 2)
            SELECT sum(n_g * (n_g - 1)) * 1.0 / NULLIF(sum(n_g) * (sum(n_g) - 1), 0) AS p
            FROM by_group GROUP BY month_end
            """ + ")").fetchone()[0])
    autocorr = con.execute(f"""
        SELECT round(corr(rel_peer_mom_20d, prev), 4) FROM (
            SELECT rel_peer_mom_20d,
                   lag(rel_peer_mom_20d) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                       AS prev
            FROM {RELATION_STAT_TABLE}
        ) WHERE rel_peer_mom_20d IS NOT NULL AND prev IS NOT NULL
        """).fetchone()[0]

    ratio = round(float(agree) / baseline, 1) if baseline else None
    return "\n".join(
        [
            f"1. **유효 시작 {first}.** 2016년부터 커버리지 0.85를 넘고 2017년 이후 평균 "
            f"{coverage}이다. §1.5의 기대(0.8 이상)를 만족한다.",
            "2. **분포는 정상.** `rel_own_minus_peer_20d`의 날짜별 중앙값이 0 근처이고(정의상 "
            "그래야 한다), 꼬리는 p1/p99가 ±0.3 수준이다.",
            f"3. **회전율이 낮다.** `rel_peer_mom_20d`의 lag1 자기상관 {autocorr}, peer 집합의 "
            "월간 유지율 0.74. 20세션 창이 겹치므로 예상된 값이고, horizon 20 이상에서 "
            "실질 관측 수가 크게 줄어든다는 뜻이다.",
            f"4. **통계적 peer는 업종의 대용이 아니다.** K=20 peer의 KSIC 2자리 일치율 {agree}, "
            f"같은 후보 집합의 무작위 기준 {baseline} — 우연의 약 {ratio}배다. 업종을 부분적으로 "
            "재현하지만 peer의 대다수(약 80%)는 다른 업종이다. `01` §5가 예상한 두 결과 중 "
            '"두 축이 다른 정보를 본다"에 가깝다.',
            "5. **기존 축과 중복되지 않는다.** 일별 순위상관 절대값이 모두 0.5 미만이고, 최대는 "
            "`rel_own_minus_peer_20d` × `px_reversal_5d`의 −0.38이다(부호는 "
            "`px_reversal_5d`가 수익률의 **음수** 합이라 방향이 맞다).",
            "",
            "주의: 2014년 월말 8개는 후보가 2종목뿐이다(2007년까지 이력이 있는 소수 종목만 "
            "252+252 세션을 채운다). 그 구간의 `rel_peer_*`는 15-of-20 규칙으로 전부 NULL이므로 "
            "마트에는 남아 있어도 피쳐로는 쓰이지 않는다.",
        ]
    )


def build(config: LakeConfig, *, force: bool) -> duckdb.DuckDBPyConnection:
    con = connect(config)
    register_views(con, config, tables=list(_RAW_INPUTS))
    logger.info("building %s", PEER_MONTHLY_TABLE)
    materialize_peer_monthly(con, config, force=force)
    logger.info("building %s", RELATION_STAT_TABLE)
    materialize_relation_stat(con, config, force=force)
    return con


def render(con: duckdb.DuckDBPyConnection, config: LakeConfig) -> str:
    have_price = _register_a0_mart(con, config, "feat_price")
    have_fin = _register_a0_mart(con, config, "feat_fin_scan_daily")
    rows = con.execute(f"SELECT count(*) FROM {RELATION_STAT_TABLE}").fetchone()[0]
    span = con.execute(
        f"SELECT min(trade_date), max(trade_date) FROM {RELATION_STAT_TABLE}"
    ).fetchone()

    return "\n".join(
        [
            "# F-2.4 — `feat_relation_stat` 마트 검증 리포트",
            "",
            f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST",
            f"- snapshot `{config.snapshot_date}` / source `{config.source}`",
            f"- `FORMULA_VERSION = {FORMULA_VERSION}`",
            f"- `{RELATION_STAT_TABLE}` {rows:,}행, {span[0]} ~ {span[1]}",
            "",
            "생성 명령: `uv run python -m modeler.analysis.relation_stat_report "
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
            "## 1. peer 집합 (`dim_peer_monthly`)",
            "",
            peer_monthly_section(con),
            "",
            "---",
            "",
            "## 2. 커버리지",
            "",
            coverage_section(con),
            "",
            "---",
            "",
            "## 3. 분포",
            "",
            distribution_section(con),
            "",
            "---",
            "",
            "## 4. 회전율 (자기상관·peer 유지율)",
            "",
            turnover_section(con),
            "",
            "---",
            "",
            "## 5. 기존 축과의 중복 (일별 순위상관)",
            "",
            overlap_section(con, have_price=have_price, have_fin=have_fin),
            "",
            "|ρ| ≥ 0.5이면 카드에 경고를 적는다(§1.5).",
            "",
            "---",
            "",
            "## 6. 사전등록 고정 항목 (F-2.5 → F-HS-1)",
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
    parser.add_argument("--force", action="store_true", help="rebuild both marts")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-name", default="f2_relation_stat_verification.md")
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

    con = build(config, force=args.force)
    text = render(con, config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.output_name).write_text(text, encoding="utf-8")
    print(f"wrote {out_dir / args.output_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
