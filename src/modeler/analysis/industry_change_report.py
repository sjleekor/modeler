"""F-1.7 — the monthly ``induty_code`` change-rate report.

``01_industry_pit.md`` §4 turns "can we use backcast industry?" into a
measurement: D-F1 allows F-3's industry relation features onto the scan and
into FS3 only if the **group** change rate averages under 0.3% a month over
three snapshots. Nobody has ever measured that number for the Korean market —
N4's index-membership path is not published, so this report is the only source
of it. The threshold was fixed on 2026-09-07, before any result (§4.2), and is
not revisited here.

Three things this report does that a plain ``GROUP BY`` would not:

**It never averages a seed pair into the verdict.** The 2026-08 rows are copies
of ``dart_corp_master``'s current state, so a difference between them and the
2026-09 snapshot may have happened at any time before it — the DDL comment on
``is_seed`` says so, and F-1.4 measured that pair at 0.0505% / 0.0253% without
being able to date either change. Seed pairs are reported and excluded from the
D-F1 average, which is why the earliest that average can exist is 2026-11.

**It separates a precision change from an industry change.** One of F-1.4's two
code differences was ``262 -> 26293``: the same industry reported at finer
KSIC precision, not a reclassification. A rate that counts those overstates the
leak D-F1 is about, so the code rate is split into prefix-compatible and
genuine.

**It shares one fold-up with the mart.** ``ind_group`` comes from
``industry_pit.compute_industry_observations``, the same call
``dim_industry_pit_daily`` is built from, so the report and the feature cannot
disagree about what group a company was in.

Example::

    uv run python -m modeler.analysis.industry_change_report \
        --snapshot-date 2026-09-08 --source sj2_remote
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import duckdb

from modeler.etl.config import DEFAULT_SOURCE, DataRoot, EngineOptions, LakeConfig
from modeler.etl.features.industry_pit import (
    FORMULA_VERSION,
    INDUSTRY_PIT_TABLE,
    compute_industry_observations,
)
from modeler.etl.lake import connect, register_views
from modeler.etl.mart import is_materialized, register_mart_view

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("docs/dev/20260907_additional_feature/results")

#: D-F1's threshold on the monthly *group* change rate, fixed 2026-09-07 before
#: any result was seen (``01`` §4.2). Not a tuning knob.
GROUP_RATE_THRESHOLD_PCT = 0.3

#: Consecutive non-seed snapshot pairs D-F1 needs before it can be decided.
REQUIRED_PAIRS = 3

_RAW_INPUTS = ("dart_corp_profile_history",)


def _pair_rows(con: duckdb.DuckDBPyConnection, staging: str) -> list[tuple]:
    """One row per consecutive month pair, with both change counts.

    ``prefix_compatible`` is the F-1.4 case: one code is a prefix of the other,
    so the industry did not move — only the reported precision did.
    """
    return con.execute(f"""
        WITH obs AS (
            SELECT
                ticker, observed_month, induty_code, ind_group, is_seed,
                LAG(observed_month) OVER w AS prev_month,
                LAG(induty_code) OVER w AS prev_code,
                LAG(ind_group) OVER w AS prev_group,
                LAG(is_seed) OVER w AS prev_is_seed
            FROM {staging}
            WINDOW w AS (PARTITION BY ticker ORDER BY observed_month)
        ),
        pairs AS (
            SELECT
                prev_month, observed_month,
                (prev_is_seed OR is_seed) AS involves_seed,
                (induty_code IS DISTINCT FROM prev_code) AS code_changed,
                (ind_group IS DISTINCT FROM prev_group) AS group_changed,
                (
                    induty_code IS DISTINCT FROM prev_code
                    AND induty_code IS NOT NULL AND prev_code IS NOT NULL
                    AND (
                        starts_with(induty_code, prev_code)
                        OR starts_with(prev_code, induty_code)
                    )
                ) AS prefix_compatible
            FROM obs
            WHERE prev_month IS NOT NULL
        )
        SELECT
            prev_month, observed_month,
            bool_or(involves_seed) AS involves_seed,
            count(*) AS n_both,
            sum(code_changed::INT) AS code_changes,
            sum(prefix_compatible::INT) AS precision_changes,
            sum((code_changed AND NOT prefix_compatible)::INT) AS genuine_changes,
            sum(group_changed::INT) AS group_changes
        FROM pairs
        GROUP BY 1, 2
        ORDER BY 2
        """).fetchall()


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def monthly_section(pairs: list[tuple]) -> str:
    if not pairs:
        return "관측 쌍이 없다. 스냅샷이 한 달뿐이면 변경률은 정의되지 않는다."
    lines = [
        "| 구간 | 양쪽 법인 | 코드 변경 | 코드 변경률 | 정밀도만 | 실제 코드 변경 "
        "| 그룹 변경 | 그룹 변경률 | seed 포함 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for prev, month, seed, n, code, precision, genuine, group in pairs:
        lines.append(
            f"| {prev} → {month} | {n:,} | {code} | {_pct(code, n):.4f}% | {precision} | "
            f"{genuine} | {group} | {_pct(group, n):.4f}% | {'예' if seed else '아니오'} |"
        )
    return "\n".join(lines)


def precision_section(pairs: list[tuple]) -> str:
    total_code = sum(row[4] for row in pairs)
    total_precision = sum(row[5] for row in pairs)
    total_genuine = sum(row[6] for row in pairs)
    if total_code == 0:
        return "코드가 바뀐 법인이 없다. 정밀도 변경과 실제 변경을 나눌 것도 없다."
    return "\n".join(
        [
            f"- 코드 변경 {total_code}건 = 정밀도 변경 {total_precision}건 + 실제 변경 "
            f"{total_genuine}건",
            "- 정밀도 변경은 한쪽 코드가 다른 쪽의 접두인 경우다(F-1.4의 `262` → `26293`). "
            "업종이 바뀐 것이 아니라 DART가 더 자세히 적은 것이다.",
            f"- D-F1이 보는 것은 그룹 변경률이므로 정밀도 변경 {total_precision}건은 "
            "판정에 들어가지 않는다. 2자리 접두가 같으면 그룹도 같다.",
        ]
    )


def cumulative_section(con: duckdb.DuckDBPyConnection, staging: str) -> str:
    rows = con.execute(f"""
        WITH first_obs AS (
            SELECT
                ticker,
                arg_min(induty_code, observed_month) AS first_code,
                arg_min(ind_group, observed_month) AS first_group,
                min(observed_month) AS first_month
            FROM {staging}
            GROUP BY ticker
        )
        SELECT
            o.observed_month,
            count(*) AS n,
            sum((o.induty_code IS DISTINCT FROM f.first_code)::INT) AS code_moved,
            sum((o.ind_group IS DISTINCT FROM f.first_group)::INT) AS group_moved
        FROM {staging} o
        JOIN first_obs f ON o.ticker = f.ticker
        WHERE o.observed_month > f.first_month
        GROUP BY 1
        ORDER BY 1
        """).fetchall()
    if not rows:
        return "첫 스냅샷 이후의 달이 없다."
    lines = [
        "| 스냅샷 | 양쪽 법인 | 첫 스냅샷 대비 코드 변경 | 비율 | 그룹 변경 | 비율 |",
        "|---|---|---|---|---|---|",
    ]
    for month, n, code_moved, group_moved in rows:
        lines.append(
            f"| {month} | {n:,} | {code_moved} | {_pct(code_moved, n):.4f}% | "
            f"{group_moved} | {_pct(group_moved, n):.4f}% |"
        )
    return "\n".join(lines)


def direction_section(con: duckdb.DuckDBPyConnection, staging: str) -> str:
    rows = con.execute(f"""
        WITH obs AS (
            SELECT
                ticker, observed_month, ind_group,
                LAG(ind_group) OVER w AS prev_group,
                LAG(observed_month) OVER w AS prev_month
            FROM {staging}
            WINDOW w AS (PARTITION BY ticker ORDER BY observed_month)
        )
        SELECT prev_group, ind_group, count(*) AS n, min(observed_month), max(observed_month)
        FROM obs
        WHERE prev_month IS NOT NULL AND ind_group IS DISTINCT FROM prev_group
        GROUP BY 1, 2
        ORDER BY n DESC, 1, 2
        """).fetchall()
    if not rows:
        return "그룹이 바뀐 법인이 없다. 방향을 적을 것이 없다."
    lines = ["| 이전 그룹 | 이후 그룹 | 건수 | 처음 | 마지막 |", "|---|---|---|---|---|"]
    for prev_group, group, n, first, last in rows:
        lines.append(f"| {prev_group} | {group} | {n} | {first} | {last} |")
    return "\n".join(lines)


def verdict_section(pairs: list[tuple]) -> str:
    usable = [row for row in pairs if not row[2]]
    seeded = [row for row in pairs if row[2]]
    lines = [
        f"- 문턱: 월평균 그룹 변경률 **{GROUP_RATE_THRESHOLD_PCT}%** "
        f"(2026-09-07 고정, `01` §4.2). 필요한 비-seed 구간 {REQUIRED_PAIRS}개.",
        f"- 지금 있는 구간: 비-seed {len(usable)}개, seed 포함 {len(seeded)}개.",
    ]
    if seeded:
        rates = ", ".join(f"{row[1]} {_pct(row[7], row[3]):.4f}%" for row in seeded)
        lines.append(
            f"- seed 포함 구간의 그룹 변경률({rates})은 **판정에 넣지 않는다.** seed 행은 "
            "`dart_corp_master`의 현재 상태 복사본이라 변경 시점을 잡을 수 없다."
        )
    if len(usable) < REQUIRED_PAIRS:
        lines.append(
            f"- **판정 보류.** 비-seed 구간이 {len(usable)}/{REQUIRED_PAIRS}개다. "
            "판정은 2026-12이고, 그때까지 소급 업종은 정규화·진단 전용을 유지한다."
        )
        return "\n".join(lines)

    recent = usable[-REQUIRED_PAIRS:]
    average = sum(_pct(row[7], row[3]) for row in recent) / len(recent)
    lines.append(f"- 최근 {REQUIRED_PAIRS}개 구간 그룹 변경률 월평균 **{average:.4f}%**")
    if average < GROUP_RATE_THRESHOLD_PCT:
        lines.append(
            f"- 문턱 미달({average:.4f}% < {GROUP_RATE_THRESHOLD_PCT}%). `01` §4.2대로 "
            "F-3 업종 관계 피쳐를 소급 업종으로 만들어 Horizon Scan 후보와 FS3에 올릴 수 "
            f"있다. 단 카드에 \"업종은 2026-09 이전 소급(연 약 {average * 12:.2f}% 누수)\"을 "
            "적고 `ind_is_backcast`를 같이 넘긴다."
        )
    else:
        lines.append(
            f"- 문턱 초과({average:.4f}% ≥ {GROUP_RATE_THRESHOLD_PCT}%). 소급 업종은 "
            "정규화·진단 전용을 유지한다. 업종 관계 피쳐는 L1 이력이 12개월 이상 쌓인 뒤 "
            "PIT 구간에서만 검정한다(2027-09 이후)."
        )
    return "\n".join(lines)


def mart_section(con: duckdb.DuckDBPyConnection, config: LakeConfig) -> str:
    """Coverage of ``dim_industry_pit_daily``, when this snapshot has it built."""
    if not is_materialized(config, INDUSTRY_PIT_TABLE):
        return (
            f"`{INDUSTRY_PIT_TABLE}`가 이 snapshot에 없다. "
            "`uv run python -m modeler.etl.compute_all --with-features`로 만든다."
        )
    register_mart_view(con, config, INDUSTRY_PIT_TABLE)
    total, first, last = con.execute(
        f"SELECT count(*), min(trade_date), max(trade_date) FROM {INDUSTRY_PIT_TABLE}"
    ).fetchone()
    rows = con.execute(f"""
        SELECT
            CASE
                WHEN ind_is_backcast IS NULL THEN '관측 없음'
                WHEN ind_is_backcast THEN '소급(backcast)'
                ELSE 'PIT'
            END AS segment,
            count(*) AS n,
            count(ind_ksic_code) AS with_code,
            count(DISTINCT ticker) AS tickers,
            min(trade_date) AS first_date,
            max(trade_date) AS last_date
        FROM {INDUSTRY_PIT_TABLE}
        GROUP BY 1
        ORDER BY 2 DESC
        """).fetchall()
    lines = [
        f"`{INDUSTRY_PIT_TABLE}` {total:,}행, {first} ~ {last}, "
        f"`FORMULA_VERSION = {FORMULA_VERSION}`",
        "",
        "| 구간 | 행 | 코드 있음 | 종목 | 처음 | 마지막 |",
        "|---|---|---|---|---|---|",
    ]
    for segment, n, with_code, tickers, first_date, last_date in rows:
        lines.append(
            f"| {segment} | {n:,} | {with_code:,} | {tickers:,} | {first_date} | {last_date} |"
        )
    groups = con.execute(f"""
        SELECT count(DISTINCT ind_group), count(*) FILTER (WHERE ind_group = 'OTHER'),
               count(*) FILTER (WHERE ind_group = 'XX'),
               count(*) FILTER (WHERE ind_group IS NULL)
        FROM {INDUSTRY_PIT_TABLE}
        """).fetchone()
    lines += [
        "",
        f"- 그룹 {groups[0]}개. `OTHER` {groups[1]:,}행, `XX`(코드가 KSIC 형태가 아님) "
        f"{groups[2]:,}행, NULL(관측 자체가 없음) {groups[3]:,}행",
        "- `XX`와 NULL은 다르다. `XX`는 관측된 코드가 2자리 숫자로 시작하지 않는 경우고, "
        "NULL은 그 종목이 한 번도 프로필에 없었던 경우다.",
        "- `PIT` 구간만 look-ahead가 없다. 소급 구간은 첫 스냅샷의 코드를 과거에 붙인 "
        "것이고 `ind_is_backcast = TRUE`가 그 표지다.",
    ]
    return "\n".join(lines)


def render(con: duckdb.DuckDBPyConnection, config: LakeConfig, staging: str) -> str:
    pairs = _pair_rows(con, staging)
    months = con.execute(
        f"SELECT count(DISTINCT observed_month), min(observed_month), max(observed_month), "
        f"count(DISTINCT ticker) FROM {staging}"
    ).fetchone()
    latest = pairs[-1] if pairs else None
    headline = (
        f"최신 구간 {latest[0]} → {latest[1]}: 코드 {_pct(latest[4], latest[3]):.4f}%, "
        f"그룹 **{_pct(latest[7], latest[3]):.4f}%**"
        if latest
        else "구간이 없다"
    )

    return "\n".join(
        [
            f"# F-1.7 — 업종 코드 변경률 리포트 ({months[2]})",
            "",
            f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST",
            f"- snapshot `{config.snapshot_date}` / source `{config.source}`",
            f"- 스냅샷 {months[0]}개({months[1]} ~ {months[2]}), 법인 {months[3]:,}",
            f"- {headline}",
            "",
            "생성 명령: `uv run python -m modeler.analysis.industry_change_report "
            f"--snapshot-date {config.snapshot_date} --source {config.source}`",
            "",
            "---",
            "",
            "## 1. 월별 변경률",
            "",
            monthly_section(pairs),
            "",
            "분모는 두 달 양쪽에 다 있는 ticker 보유 법인이다. `dart_corp_profile_history`는 "
            "`--universe-scope historical`로 받으므로 상폐 법인도 들어 있다.",
            "",
            "---",
            "",
            "## 2. 정밀도 변경과 실제 변경",
            "",
            precision_section(pairs),
            "",
            "---",
            "",
            "## 3. 누적 변경률 (첫 스냅샷 대비)",
            "",
            cumulative_section(con, staging),
            "",
            "---",
            "",
            "## 4. 변경 방향",
            "",
            direction_section(con, staging),
            "",
            "---",
            "",
            "## 5. D-F1 판정 상태",
            "",
            verdict_section(pairs),
            "",
            "---",
            "",
            "## 6. 마트 커버리지 (`dim_industry_pit_daily`)",
            "",
            mart_section(con, config),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--output-name",
        default=None,
        help="default: industry_change_<latest observed_month YYYYMM>.md",
    )
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

    con = connect(config)
    register_views(con, config, tables=list(_RAW_INPUTS))
    staging = compute_industry_observations(con)
    text = render(con, config, staging)

    latest = con.execute(f"SELECT max(observed_month) FROM {staging}").fetchone()[0]
    name = args.output_name or f"industry_change_{latest:%Y%m}.md"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(text, encoding="utf-8")
    print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
