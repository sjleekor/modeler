"""F-6.1 — which XBRL receipts have to be refetched for ``fin_sue`` to exist.

``fin_sue`` is the one axis in the whole scan that has never been measured:
``effective_start`` came out 2025-05-02 with coverage 0.00 and all six event
cells ``insufficient`` (``04_sue_backfill_and_events.md`` §1.1). The cause is
not the formula, it is the input -- XBRL captured against *original* periodic
receipts is too sparse for any ticker to accumulate the history the statistic
needs.

**What the statistic needs, taken from the code rather than the prose.**
``features/sue_event.py`` computes ``seasonal_change = quarterly_eps -
comparative_eps`` where ``comparative_eps`` comes from ``value_lag_4q`` — the
comparative column *inside the same filing* (``comparative_policy =
'as_was_lag4q'``). So one filing yields one ``seasonal_change``, and
``fin_sue`` needs ``history_count >= MIN_SUE_HISTORY`` (8) over the preceding
rows plus the event's own. That is **nine consecutive quarterly filings, each
with XBRL** — not thirteen, which is what "eight quarters of differences" would
mean if the comparative came from the filing four quarters back.

**The narrowing.** Refetching every periodic receipt without XBRL is 27,911
calls, and most of them can never help: a window is only completable if all
nine of its filings were actually *filed*. So a missing receipt is a target
only when it sits inside a nine-quarter window in which every period has an
original receipt on record and at least one of them lacks XBRL. A window with a
genuinely unfiled quarter is skipped -- no amount of fetching closes it.

Output is **JSON lines**, the format ``dart backfill-xbrl-receipts
--targets-file`` actually parses (``cli/app.py``): one object per line with
``ticker``, ``corp_code``, ``bsns_year``, ``reprt_code``, ``rcept_no``. The
plan document's ``--out ...csv`` was wrong about the extension.

Example::

    uv run python -m modeler.analysis.sue_backfill_targets \
        --snapshot-date 2026-09-08 --source sj2_remote \
        --out targets/sue_xbrl_targets.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb

from modeler.etl.config import DEFAULT_SOURCE, DataRoot, EngineOptions, LakeConfig
from modeler.etl.features.sue_event import MIN_SUE_HISTORY
from modeler.etl.lake import connect, register_views

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("docs/dev/20260907_additional_feature/results")

#: Consecutive quarterly filings one ``fin_sue`` value needs: the eight rows of
#: history the code requires plus the event's own filing.
RUN_LENGTH = MIN_SUE_HISTORY + 1

#: Fiscal-period month -> OpenDART ``reprt_code``.
_REPRT_BY_MONTH = {3: "11013", 6: "11012", 9: "11014", 12: "11011"}

_RAW_INPUTS = ("dart_filing_receipt_raw", "dart_xbrl_document")

#: Report names that are an original periodic filing. Corrections
#: (``[기재정정]``/``[첨부정정]``), attachment additions, deadline-extension
#: notices and re-filings of foreign reports are all excluded: the SUE event
#: rule reads ``is_revision = FALSE`` originals only.
_PERIODIC_SQL = """
    (report_nm LIKE '사업보고서 (%'
     OR report_nm LIKE '반기보고서 (%'
     OR report_nm LIKE '분기보고서 (%')
"""


def build_receipt_sql(
    *,
    receipt_view: str = "dart_filing_receipt_raw",
    xbrl_document_view: str = "dart_xbrl_document",
    start_year: int = 2015,
    end_year: int = 2026,
) -> str:
    """One row per original periodic filing, with its period and XBRL presence.

    ``bsns_year``/``reprt_code`` are parsed out of ``report_nm``'s ``(YYYY.MM)``
    suffix rather than taken from the receipt date: a Q1 report is filed in May,
    so the two disagree, and the backfill command keys on the fiscal period.
    """
    reprt_case = " ".join(
        f"WHEN {month} THEN '{code}'" for month, code in sorted(_REPRT_BY_MONTH.items())
    )
    return f"""
        WITH originals AS (
            SELECT
                corp_code,
                any_value(ticker) AS ticker,
                rcept_no,
                rcept_dt,
                CAST(regexp_extract(report_nm, '\\((\\d{{4}})\\.(\\d{{2}})\\)', 1) AS INTEGER)
                    AS bsns_year,
                CAST(regexp_extract(report_nm, '\\((\\d{{4}})\\.(\\d{{2}})\\)', 2) AS INTEGER)
                    AS period_month
            FROM {receipt_view}
            WHERE {_PERIODIC_SQL}
            GROUP BY corp_code, rcept_no, rcept_dt, report_nm
        ),
        typed AS (
            SELECT
                corp_code, ticker, rcept_no, rcept_dt, bsns_year,
                CASE period_month {reprt_case} END AS reprt_code,
                bsns_year * 4 + (period_month / 3) - 1 AS period_index
            FROM originals
            WHERE bsns_year BETWEEN {start_year} AND {end_year}
              AND period_month IN ({", ".join(str(m) for m in sorted(_REPRT_BY_MONTH))})
        )
        SELECT
            t.corp_code, t.ticker, t.rcept_no, t.rcept_dt, t.bsns_year,
            t.reprt_code, CAST(t.period_index AS BIGINT) AS period_index,
            (x.rcept_no IS NOT NULL) AS has_xbrl
        FROM typed t
        LEFT JOIN (SELECT DISTINCT rcept_no FROM {xbrl_document_view}) x
          ON x.rcept_no = t.rcept_no
    """


def select_targets(rows: list[tuple]) -> tuple[list[dict], dict[str, int]]:
    """Pick the receipts that could complete a ``RUN_LENGTH`` window.

    Args:
        rows: ``(corp_code, ticker, rcept_no, rcept_dt, bsns_year, reprt_code,
            period_index, has_xbrl)`` for every original periodic filing.

    Returns:
        ``(targets, stats)``. Each target is the JSON-lines object the backfill
        command reads. ``stats`` carries the counts the measurement document
        reports, including the two rejection reasons — a window that is already
        complete needs nothing, and one with an unfiled quarter cannot be
        completed at all.
    """
    by_corp: dict[str, dict[int, tuple]] = defaultdict(dict)
    for row in rows:
        by_corp[row[0]][int(row[6])] = row

    stats = {
        "corps_seen": len(by_corp),
        "receipts_seen": len(rows),
        "receipts_missing_xbrl": sum(1 for r in rows if not r[7]),
        "windows_complete": 0,
        "windows_completable": 0,
        "windows_unfilable": 0,
    }
    chosen: dict[str, tuple] = {}

    for periods in by_corp.values():
        if not periods:
            continue
        lo, hi = min(periods), max(periods)
        for start in range(lo, hi - RUN_LENGTH + 2):
            window = [periods.get(start + offset) for offset in range(RUN_LENGTH)]
            if any(entry is None for entry in window):
                stats["windows_unfilable"] += 1
                continue
            gaps = [entry for entry in window if not entry[7]]
            if not gaps:
                stats["windows_complete"] += 1
                continue
            stats["windows_completable"] += 1
            for entry in gaps:
                chosen[entry[2]] = entry

    targets = [
        {
            "ticker": entry[1] or "",
            "corp_code": entry[0],
            "bsns_year": int(entry[4]),
            "reprt_code": entry[5],
            "rcept_no": entry[2],
        }
        for entry in sorted(chosen.values(), key=lambda e: (e[0], int(e[6])))
    ]
    stats["targets"] = len(targets)
    stats["target_corps"] = len({t["corp_code"] for t in targets})
    return targets, stats


def _year_histogram(rows: list[tuple], targets: list[dict]) -> list[tuple[int, int, int, int]]:
    """``(bsns_year, filings, missing_xbrl, targets)`` per fiscal year."""
    target_by_receipt = {t["rcept_no"] for t in targets}
    per_year: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        bucket = per_year[int(row[4])]
        bucket[0] += 1
        if not row[7]:
            bucket[1] += 1
        if row[2] in target_by_receipt:
            bucket[2] += 1
    return [(year, *counts) for year, counts in sorted(per_year.items())]


def render(
    config: LakeConfig,
    rows: list[tuple],
    targets: list[dict],
    stats: dict[str, int],
    out_path: Path,
) -> str:
    histogram = _year_histogram(rows, targets)
    missing = stats["receipts_missing_xbrl"]
    saved = missing - stats["targets"]
    lines = [
        "# F-6.1 — `fin_sue` XBRL 백필 대상 측정",
        "",
        f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST",
        f"- snapshot `{config.snapshot_date}` / source `{config.source}`",
        f"- 대상 파일: `{out_path}` (JSON lines)",
        "",
        "생성 명령: `uv run python -m modeler.analysis.sue_backfill_targets "
        f"--snapshot-date {config.snapshot_date} --source {config.source} --out {out_path}`",
        "",
        "---",
        "",
        "## 1. 무엇이 필요한가 — 코드에서 읽은 값",
        "",
        f"`features/sue_event.py`는 `history_count >= MIN_SUE_HISTORY`({MIN_SUE_HISTORY})를 "
        "요구하고, `comparative_eps`는 `value_lag_4q` — **같은 보고서 안의 비교 컬럼**이다"
        "(`comparative_policy='as_was_lag4q'`). 보고서 하나가 `seasonal_change` 하나를 만든다.",
        "",
        f"따라서 `fin_sue` 한 값에 필요한 것은 **연속 {RUN_LENGTH}개 분기 보고서**이고 "
        "각각 XBRL이 있어야 한다. 비교값이 4분기 전 보고서에서 온다면 13개가 필요했겠지만 "
        "그렇지 않다.",
        "",
        "---",
        "",
        "## 2. 규모",
        "",
        f"- 원본 정기보고서 **{stats['receipts_seen']:,}건** / 법인 {stats['corps_seen']:,}",
        f"- XBRL 없는 접수 **{missing:,}건**",
        f"- 그중 **대상 {stats['targets']:,}건** / 법인 {stats['target_corps']:,} "
        f"(제외 {saved:,}건, {100.0 * saved / missing:.1f}%)",
        "",
        f"창 판정: 완성 가능 {stats['windows_completable']:,} / 이미 완성 "
        f"{stats['windows_complete']:,} / **채울 수 없음 {stats['windows_unfilable']:,}**",
        "",
        "마지막 것이 좁히기의 핵심이다. 아홉 분기 중 하나라도 애초에 **제출되지 않았으면** "
        "그 창은 아무리 받아도 완성되지 않으므로 그 안의 결측은 대상이 아니다.",
        "",
        "| 회계연도 | 원본 접수 | XBRL 없음 | 대상 |",
        "|---|---|---|---|",
    ]
    for year, filings, missing_year, targets_year in histogram:
        lines.append(f"| {year} | {filings:,} | {missing_year:,} | {targets_year:,} |")
    lines += [
        "",
        "---",
        "",
        "## 3. 호출 수와 실행 판단",
        "",
        f"- 접수 하나당 XBRL 1회 호출이므로 **약 {stats['targets']:,}회**다. 재시도는 별도다.",
        f"- OpenDART 키당 일 한도 10,000이므로 키 2개면 "
        f"{max(1, -(-stats['targets'] // 20000))}일 안에 끝난다.",
        "- exit 75 재개는 skip-if-present로 동작한다 — 이미 받은 문서는 다시 받지 않는다. "
        "F-9.3처럼 **회수 불가한 대상이 매번 반복되는 문제**는 여기서는 없다: 대상은 "
        "제출된 보고서이고 없는 것은 XBRL 문서뿐이다.",
        f"- slice ledger 사용 여부: 대상이 {stats['targets']:,}건이므로 "
        + (
            "**필요하다**(수만 건, 여러 날에 걸친다)."
            if stats["targets"] > 20000
            else "**불필요하다**(하루 안에 끝난다)."
        ),
        "",
        "---",
        "",
        "## 4. 다음 (F-6.2, F-6.3)",
        "",
        "- F-6.2: prod에서 `dart backfill-xbrl-receipts --targets-file`. `04` §1.3대로 "
        "04:00 체인·23:30 filings와 겹치지 않는 시간대, `opendart` lock 공유",
        "- F-6.3: 새 snapshot 재빌드 → `fin_sue`의 `effective_start`·coverage 확인 → "
        "**같은 사전등록**으로 재판정(격자·부호를 바꾸지 않는다)",
        "",
        "정정: `04` §1.3의 `--out targets/sue_xbrl_targets.csv`는 확장자가 틀렸다. "
        "`backfill-xbrl-receipts`가 파싱하는 것은 **JSON lines**다(`cli/app.py`).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out", default="targets/sue_xbrl_targets.jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-name", default=None)
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

    con: duckdb.DuckDBPyConnection = connect(config)
    register_views(con, config, tables=list(_RAW_INPUTS))
    rows = con.execute(
        build_receipt_sql(start_year=args.start_year, end_year=args.end_year)
    ).fetchall()
    logger.info("original periodic filings: %d", len(rows))

    targets, stats = select_targets(rows)
    logger.info("targets: %d over %d corps", stats["targets"], stats["target_corps"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(target, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(targets)} targets)")

    text = render(config, rows, targets, stats, out_path)
    name = args.output_name or f"sue_backfill_targets_{datetime.now():%Y%m}.md"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(text, encoding="utf-8")
    print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
