"""사업보고서(b) 해석 표 전수 자동 점검 (12 §6.4). 문턱은 없고 이상 후보 목록과 집계만 낸다.

    python -m modeler.etl.disclosures.biz_interp_qa --dir <interp 출력 디렉터리>

``biz_interp_build interp``가 쓴 ``kr_bizrpt_*_row.parquet``·``interp_assignment_subset.parquet``를 읽고
``qa_summary.json``·``qa_*.tsv``를 같은 디렉터리에 쓴다. 해석 값만 보고 원문·수익률은 열지 않는다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

TOL_REL = 0.002  # 합계 대조 상대 허용(반올림)


def _tables(rows: list[dict]):
    by: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[(r["rcept_no"], r["section_key"], r["table_idx"], r["period_idx"])].append(r)
    return by


def _close(a: float, b: float, n_items: int) -> bool:
    return abs(a - b) <= max(n_items, 1) * 1.0 + TOL_REL * max(abs(a), abs(b))


def check_sums(rows: list[dict], amt: str = "amount_num", ratio: str = "ratio_pct") -> list[dict]:
    """표·기간마다 ``item`` 합 ≈ 합계(``total``) 행, 비율 합 ≤ 100.5를 본다. 어긋나는 표를 낸다."""
    bad = []
    for key, rs in _tables(rows).items():
        items = [r for r in rs if r["row_kind"] == "item" and r.get(amt) is not None]
        tots = [r for r in rs if r["row_kind"] == "total" and r.get(amt) is not None]
        subs = [r for r in rs if r["row_kind"] == "subtotal"]
        if len(tots) == 1 and items and not subs:  # 소계가 섞인 계층 표는 이 단순 대조로 못 본다
            s, t = sum(r[amt] for r in items), tots[-1][amt]
            if not _close(s, t, len(items)):
                bad.append(
                    {
                        "table": "/".join(map(str, key)),
                        "check": "item_sum_vs_total",
                        "item_sum": s,
                        "total": t,
                    }
                )
        rr = [r[ratio] for r in items if r.get(ratio) is not None]
        if rr and sum(rr) > 100.5:
            bad.append(
                {
                    "table": "/".join(map(str, key)),
                    "check": "ratio_sum_gt_100",
                    "item_sum": sum(rr),
                    "total": None,
                }
            )
    return bad


def check_orderbook(rows: list[dict]) -> list[dict]:
    bad = []
    for r in rows:
        t, d, b = r.get("total_amt_num"), r.get("delivered_amt_num"), r.get("backlog_amt_num")
        if None not in (t, d, b) and abs((t - d) - b) > max(1.0, TOL_REL * abs(t)):
            bad.append(
                {
                    "table": f"{r['rcept_no']}/{r['section_key']}/{r['table_idx']}",
                    "row_idx": r["row_idx"],
                    "total": t,
                    "delivered": d,
                    "backlog": b,
                }
            )
    return bad


def check_periods(rows: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for r in rows:
        k = (r["rcept_no"], r["section_key"], r["table_idx"], r["period_idx"])
        if k in seen:
            continue
        seen.add(k)
        fy, fy0 = r.get("fiscal_year"), r.get("fy0")
        why = []
        if r.get("period_conflict"):
            why.append("period_conflict")
        if r.get("period_irregular"):
            why.append("period_irregular")
        if fy is not None and fy0 is not None and abs(fy - fy0) >= 3:
            why.append("fy_far_from_fy0")
        if fy is None:
            why.append("fiscal_year_null")
        if why:
            out.append(
                {
                    "table": "/".join(map(str, k[:3])),
                    "period_idx": k[3],
                    "why": ",".join(why),
                    "label": r.get("period_label_raw"),
                }
            )
    return out


def _write_tsv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0]) if rows else ["empty"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def run(d: Path) -> dict:
    summ: dict = {}
    names = {
        "customer": "kr_bizrpt_customer_row",
        "orderbook": "kr_bizrpt_orderbook_row",
        "segment_sales": "kr_bizrpt_segment_sales_row",
    }
    for t, n in names.items():
        rows = pq.read_table(d / f"{n}.parquet").to_pylist()
        s: dict = {
            "n_rows": len(rows),
            "n_tables": len({(r["rcept_no"], r["section_key"], r["table_idx"]) for r in rows}),
        }
        s["unit_status_rows"] = dict(collections.Counter(r["unit_status"] for r in rows))
        s["row_kind"] = dict(collections.Counter(r["row_kind"] for r in rows))
        s["tmpl_group_rows"] = dict(collections.Counter(r["tmpl_group"] for r in rows))
        s["parse_status"] = dict(collections.Counter(r["parse_status"] for r in rows))
        per = check_periods(rows)
        s["period_flags"] = dict(collections.Counter(x["why"] for x in per))
        _write_tsv(d / f"qa_{t}_periods.tsv", per)
        if t == "orderbook":
            bad = check_orderbook(rows)
            s["order_identity_bad_rows"] = len(bad)
            _write_tsv(d / "qa_orderbook_identity.tsv", bad)
        else:
            bad = check_sums(rows)
            s["sum_check_bad_tables"] = len(bad)
            _write_tsv(d / f"qa_{t}_sums.tsv", bad)
            s["tables_with_total_row"] = len(
                {
                    _k
                    for _k, rs in _tables(rows).items()
                    if any(r["row_kind"] == "total" for r in rs)
                }
            )
        summ[t] = s
    (d / "qa_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.dir), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
