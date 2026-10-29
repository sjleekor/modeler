"""사업보고서(b) 해석 표 전수 자동 점검 (12 §6.4). 문턱은 없고 이상 후보 목록과 집계만 낸다.

    python -m modeler.etl.disclosures.biz_interp_qa --dir <출력 디렉터리> [--sample-tsv <대조 표본 tsv>] [--dev-dir <개발 표본 해석 출력>]

``biz_interp_build full``이 쓴 ``kr_bizrpt_*_row_all_candidates_diag.parquet``(또는 ``interp``가 쓴
``kr_bizrpt_*_row.parquet``)를 DuckDB로 집계한다(행 수백만 개를 파이썬 객체로 올리지 않는다).
결과는 ``qa_summary.json``과 ``qa_*.tsv``로 같은 디렉터리에 쓴다.

``--sample-tsv``(대조 표본)의 문서는 집계 숫자에는 섞되, **행·표를 나열하는 목록 TSV에서는 뺀다**
(결과 전에 저장한 대조 표본의 해석 행을 미리 보지 않기 위해서다).

점검 (모두 해석 값만 본다 — 원문·수익률은 열지 않는다)
* 합계 대조: ``total`` 행이 하나, ``subtotal``·``adjust`` 행이 없는 표·기간에서 ``item`` 합 ≈ 합계(±행 수 단위·0.2%)
* 비율 합: ``item`` 비율 합 > 100.5
* 수주 항등식: 수주총액 − 기납품액 = 수주잔고 (±1 또는 0.2%)
* 기간: ``period_conflict``·``period_irregular``·``fiscal_year``가 FY0에서 3 이상 벗어남·연도 NULL
* 단위 분포: 서식 군별, 접수 연도별(전체 krw 비율과 10%p 넘게 벗어나는 연도)
* 시대별 행 비율: 개발 표본 대비 ≥2023 비율이 10%p 넘게 벗어나는 서식 군(개발 표본 30행 이상일 때)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb

TARGETS = ("customer", "orderbook", "segment_sales")
STEMS = {
    "customer": ("kr_bizrpt_customer_row_all_candidates_diag", "kr_bizrpt_customer_row"),
    "orderbook": ("kr_bizrpt_orderbook_row_all_candidates_diag", "kr_bizrpt_orderbook_row"),
    "segment_sales": (
        "kr_bizrpt_segment_sales_row_all_candidates_diag",
        "kr_bizrpt_segment_sales_row",
    ),
}
UNITS = ("krw", "missing", "conflict", "foreign", "nomoney")
NOT_SMP = "rcept_no not in (select rcept_no from smp)"


def _path(d: Path, t: str) -> Path:
    for stem in STEMS[t]:
        p = d / f"{stem}.parquet"
        if p.exists():
            return p
    raise FileNotFoundError(f"{t}: {STEMS[t]}")


def _dump(con: duckdb.DuckDBPyConnection, path: Path, sql: str) -> int:
    cur = con.execute(sql)
    cols = [x[0] for x in cur.description]
    rows = cur.fetchall()
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


def _one(con, sql):
    return con.execute(sql).fetchone()[0]


def _common(con, d: Path, t: str, p: Path) -> dict:
    s: dict = {}
    s["n_rows"], s["n_tables"] = con.execute(
        f"select count(*), count(distinct (rcept_no, section_key, table_idx)) from '{p}'"
    ).fetchone()
    for k, col in (
        ("parse_status_rows", "parse_status"),
        ("unit_status_rows", "unit_status"),
        ("row_kind", "row_kind"),
    ):
        s[k] = dict(con.execute(f"select {col}, count(*) from '{p}' group by 1").fetchall())
    s["rows_by_group"] = {
        f"{g}{' [판정]' if j else ''}": n
        for g, j, n in con.execute(
            f"select tmpl_group, judged, count(*) from '{p}' group by 1,2 order by 3 desc"
        ).fetchall()
    }
    _dump(
        con,
        d / f"qa_unit_status_{t}_by_group.tsv",
        f"""select tmpl_group, judged, count(*) n_rows, count(distinct (rcept_no, section_key, table_idx)) n_tables,
               {", ".join(f"sum((unit_status='{u}')::int) rows_{u}" for u in UNITS)}
        from '{p}' group by 1,2 order by n_rows desc""",
    )
    overall = _one(con, f"select avg((unit_status='krw')::int) from '{p}'")
    _dump(
        con,
        d / f"qa_unit_status_{t}_by_year.tsv",
        f"""select cast(substr(rcept_dt,1,4) as integer) rcpt_year, count(*) n_rows,
               round(avg((unit_status='krw')::int),4) krw_share,
               round(avg((unit_status='krw')::int) - {overall},4) diff_vs_overall,
               abs(avg((unit_status='krw')::int) - {overall}) > 0.10 flag_gt_10pp
        from '{p}' group by 1 order by 1""",
    )
    s["unit_krw_share_overall"] = round(overall, 4)
    s["unit_year_flagged"] = _one(
        con,
        f"select count(*) from (select abs(avg((unit_status='krw')::int) - {overall}) dv from '{p}' group by cast(substr(rcept_dt,1,4) as integer)) where dv > 0.10",
    )
    cols = {r[0] for r in con.execute(f"describe select * from '{p}'").fetchall()}
    lab = "period_label_raw" if "period_label_raw" in cols else "cast(null as varchar)"
    pidx = "period_idx"
    flags = [
        "case when period_conflict then 'period_conflict' end"
        if "period_conflict" in cols
        else "null",
        "case when period_irregular then 'period_irregular' end"
        if "period_irregular" in cols
        else "null",
        "case when fiscal_year is null then 'fiscal_year_null' end",
        "case when fiscal_year is not null and fy0 is not null and abs(fiscal_year - fy0) >= 3 then 'fy_far_from_fy0' end",
    ]
    per_sql = f"""select rcept_no, section_key, table_idx, {pidx}, tmpl_group, {lab} as period_label_raw, fiscal_year, fy0,
               concat_ws(',', {", ".join(flags)}) why
        from (select distinct on (rcept_no, section_key, table_idx, period_idx) * from '{p}') x"""
    s["period_tables_total"] = _one(con, f"select count(*) from ({per_sql})")
    s["period_flags"] = dict(
        con.execute(
            f"select why, count(*) from ({per_sql}) where why <> '' group by 1 order by 2 desc"
        ).fetchall()
    )
    _dump(
        con,
        d / f"qa_{t}_periods.tsv",
        f"select * from ({per_sql}) where why <> '' and {NOT_SMP} order by 1,2,3,4",
    )
    return s


def _orderbook(con, d: Path, p: Path, s: dict) -> None:
    has = "total_amt_num is not null and delivered_amt_num is not null and backlog_amt_num is not null"
    bad_sql = f"""select rcept_no, section_key, table_idx, row_idx, tmpl_group, total_amt_num, delivered_amt_num, backlog_amt_num
        from '{p}' where {has}
        and abs((total_amt_num - delivered_amt_num) - backlog_amt_num) > greatest(1.0, 0.002 * abs(total_amt_num))"""
    s["order_identity_checked_rows"] = _one(con, f"select count(*) from '{p}' where {has}")
    s["order_identity_bad_rows"] = _one(con, f"select count(*) from ({bad_sql})")
    _dump(
        con,
        d / "qa_orderbook_identity.tsv",
        f"select * from ({bad_sql}) where {NOT_SMP} order by 1,2,3,4",
    )


def _sums(con, d: Path, t: str, p: Path, s: dict) -> None:
    base = f"""select rcept_no, section_key, table_idx, period_idx, any_value(tmpl_group) tmpl_group, any_value(judged) judged,
             count(*) filter (where row_kind='item' and amount_num is not null) n_item,
             count(*) filter (where row_kind='total' and amount_num is not null) n_tot,
             count(*) filter (where row_kind in ('subtotal','adjust')) n_sub,
             sum(case when row_kind='item' then amount_num end) item_sum,
             max(case when row_kind='total' then amount_num end) tot,
             sum(case when row_kind='item' then ratio_pct end) ratio_sum
        from '{p}' group by 1,2,3,4"""
    cond = "n_tot = 1 and n_sub = 0 and n_item > 0"
    bad = "abs(item_sum - tot) > n_item * 1.0 + 0.002 * greatest(abs(item_sum), abs(tot))"
    s["sum_check_checked_tables"] = _one(con, f"select count(*) from ({base}) where {cond}")
    s["sum_check_bad_tables"] = _one(con, f"select count(*) from ({base}) where {cond} and {bad}")
    s["ratio_sum_checked_tables"] = _one(
        con, f"select count(*) from ({base}) where ratio_sum is not null"
    )
    s["ratio_sum_gt_100_tables"] = _one(
        con, f"select count(*) from ({base}) where ratio_sum > 100.5"
    )
    s["sum_check_bad_by_group"] = dict(
        con.execute(
            f"select tmpl_group, count(*) from ({base}) where {cond} and {bad} group by 1 order by 2 desc"
        ).fetchall()
    )
    _dump(
        con,
        d / f"qa_{t}_sums.tsv",
        f"""select rcept_no, section_key, table_idx, period_idx, tmpl_group, judged, n_item, item_sum, tot,
               case when {cond} and {bad} then 'item_sum_vs_total' else '' end sum_flag, ratio_sum, ratio_sum > 100.5 ratio_flag
        from ({base}) where (({cond} and {bad}) or ratio_sum > 100.5) and {NOT_SMP} order by 1,2,3,4""",
    )


def _era(con, d: Path, dev_dir: Path | None, summ: dict) -> None:
    q = "select tmpl_group, count(*), avg((cast(substr(rcept_dt,1,4) as integer) >= 2023)::int) from '{}' group by 1"
    out = []
    for t in TARGETS:
        full = {r[0]: (r[1], r[2]) for r in con.execute(q.format(_path(d, t))).fetchall()}
        dev: dict = {}
        if dev_dir:
            try:
                dev = {
                    r[0]: (r[1], r[2]) for r in con.execute(q.format(_path(dev_dir, t))).fetchall()
                }
            except FileNotFoundError:
                pass
        for g, (n, ge) in sorted(full.items(), key=lambda kv: -kv[1][0]):
            nd, ged = dev.get(g, (0, None))
            flag = ged is not None and nd >= 30 and abs(ge - ged) > 0.10
            out.append(
                (t, g, n, round(ge, 4), nd, None if ged is None else round(ged, 4), bool(flag))
            )
    with open(d / "qa_era_ratio.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "target",
                "tmpl_group",
                "n_rows",
                "share_ge2023",
                "n_rows_dev",
                "share_ge2023_dev",
                "flag_gt_10pp_dev_ge30",
            ]
        )
        w.writerows(out)
    summ["era_flagged_groups"] = [r[1] for r in out if r[6]]


def run(d: Path, sample_tsv: Path | None = None, dev_dir: Path | None = None) -> dict:
    con = duckdb.connect()
    con.execute("create temp table smp(rcept_no varchar)")
    n_smp = 0
    if sample_tsv and sample_tsv.exists():
        with open(sample_tsv, encoding="utf-8") as f:
            ids = [r["rcept_no"] for r in csv.DictReader(f, delimiter="\t")]
        con.executemany("insert into smp values (?)", [(x,) for x in ids])
        n_smp = len(ids)
    summ: dict = {"sample_docs_excluded_from_lists": n_smp}
    asg = d / "interp_assignment.parquet"
    if asg.exists():
        rows_sql = " union all ".join(
            f"select '{t}' as target, rcept_no, section_key, table_idx, count(*) n from '{_path(d, t)}' group by 2,3,4"
            for t in TARGETS
        )
        _dump(
            con,
            d / "qa_input_output_tables.tsv",
            f"""select a.target, a.adapter, a.judged, count(*) n_tables_in, sum((a.parse_status='partial')::int) n_partial,
                   sum((coalesce(r.n,0)>0)::int) n_tables_with_rows, sum(coalesce(r.n,0)) n_rows
            from '{asg}' a left join ({rows_sql}) r using (target, rcept_no, section_key, table_idx)
            group by 1,2,3 order by 1,4 desc""",
        )
    for t in TARGETS:
        p = _path(d, t)
        s = _common(con, d, t, p)
        if t == "orderbook":
            _orderbook(con, d, p, s)
        else:
            _sums(con, d, t, p, s)
        summ[t] = s
    _era(con, d, dev_dir, summ)
    (d / "qa_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--sample-tsv", type=Path, default=None)
    ap.add_argument("--dev-dir", type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps(run(args.dir, args.sample_tsv, args.dev_dir), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
