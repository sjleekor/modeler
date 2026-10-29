"""사업보고서(b) 해석 표 실행 — 대상·어댑터 목록(``assign``)과 지정 문서 해석(``interp``).

입력은 승격된 셀 표 ``<root>/kr/derived/disclosure_struct/snapshot_date=<날짜>/kr_bizrpt_table_cell``과
``kr_bizrpt_section_index.parquet``이다(원문 zip은 읽지 않는다). 경로는 ``STOCK_DATA_ROOT``
(기본 ``../stock_data``). ``derived/``에는 쓰지 않고 ``<root>/kr/output/disclosure_interp_v1/``에만 쓴다.

    # 1단계: 모집단 표마다 대상·어댑터만 붙인 목록 (해석 값 없음)
    python -m modeler.etl.disclosures.biz_interp_build assign

    # 지정 문서만 해석 (개발 표본 등). 문서 목록(rcept_no 한 줄씩, 또는 tsv의 rcept_no 열)이 필수다.
    python -m modeler.etl.disclosures.biz_interp_build interp --rcept-list <file> --out <dir>

``assign``은 ``verify_sample/`` 아래 모든 표본 tsv의 ``rcept_no``를 집합 빼기로만 써서 그 문서의
셀을 읽지 않는다. 전체 해석 실행(2단계)은 이 파일에 없다(대조 표본이 정해진 뒤 따로 지시).
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from modeler.etl.disclosures import bizinterp as bi
from modeler.etl.disclosures.build import code_rev, data_root

OUT_NAME = "disclosure_interp_v1"
SNAPSHOT_DEFAULT = "2026-10-28"
SECTIONS = (
    "main_products",
    "raw_materials",
    "sales",
    "orders",
    "major_customer_note_consol",
    "major_customer_note_sep",
)
CHUNK_DOCS = 2500

META_COLS = [
    "rcept_no",
    "rcept_dt",
    "corp_code",
    "stock_code",
    "corp_cls",
    "report_nm",
    "src_zip_sha256",
    "parsed_at",
    "available_from",
]
CELL_COLS = [
    "section_key",
    "table_idx",
    "row_idx",
    "col_idx",
    "rowspan",
    "colspan",
    "is_header",
    "header_path_raw",
    "text_raw",
    "num_value",
    "num_has_pct",
    "unit_raw",
    "unit_src",
    "tbl_header_rule",
]

ASSIGN_FIELDS = [
    ("rcept_no", pa.string()),
    ("section_key", pa.string()),
    ("table_idx", pa.int64()),
    ("target", pa.string()),
    ("adapter", pa.string()),
    ("tmpl_sig", pa.string()),
    ("n_body_rows", pa.int64()),
    ("n_periods", pa.int64()),
    ("unit_status", pa.string()),
    ("rcpt_year", pa.int64()),
    ("corp_cls", pa.string()),
    ("era", pa.string()),
    ("judged", pa.bool_()),
    ("parse_status", pa.string()),
    ("n_lead", pa.int64()),
    ("source", pa.string()),
    ("rcept_dt", pa.string()),
]


def snapshot_dir(root: Path, snapshot: str) -> Path:
    return root / "kr" / "derived" / "disclosure_struct" / f"snapshot_date={snapshot}"


def era_of(rcept_dt: str | None) -> str:
    y = int((rcept_dt or "0000")[:4])
    return "le2022" if y <= 2022 else "ge2023"


def read_rcept_set(path: Path) -> set[str]:
    out: set[str] = set()
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "\t" in first or "rcept_no" in first:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("rcept_no"):
                    out.add(r["rcept_no"].strip())
        else:
            out = {ln.strip() for ln in f if ln.strip()}
    return out


def excluded_docs(root: Path) -> tuple[set[str], list[str]]:
    """``verify_sample/`` 아래 모든 tsv의 ``rcept_no`` 집합(열 하나만 읽는다)."""
    d = root / "kr" / "output" / "disclosure_structuring_v1" / "verify_sample"
    ex: set[str] = set()
    files = []
    for p in sorted(d.glob("*.tsv")):
        with open(p, encoding="utf-8") as f:
            rd = csv.DictReader(f, delimiter="\t")
            if rd.fieldnames and "rcept_no" in rd.fieldnames:
                files.append(p.name)
                ex.update(r["rcept_no"] for r in rd if r.get("rcept_no"))
    return ex, files


def iter_tables(
    con: duckdb.DuckDBPyConnection,
    cell_glob: str,
    docs: list[str],
    sections: tuple[str, ...] = SECTIONS,
):
    """문서 chunk마다 ``(meta, section_key, table_idx, cells)``를 낸다. 표 안은 행·열 순."""
    sec_sql = ",".join(f"'{s}'" for s in sections)
    cols = ",".join(META_COLS + CELL_COLS + ["parse_status"])
    for i in range(0, len(docs), CHUNK_DOCS):
        chunk = docs[i : i + CHUNK_DOCS]
        con.execute("drop table if exists pk")
        con.execute("create temp table pk(rcept_no varchar)")
        con.executemany("insert into pk values (?)", [(x,) for x in chunk])
        cur = con.execute(
            f"select {cols} from read_parquet('{cell_glob}', hive_partitioning=1) "
            f"where rcept_no in (select rcept_no from pk) and section_key in ({sec_sql}) "
            "order by rcept_no, section_key, table_idx, row_idx, col_idx"
        )
        names = [d[0] for d in cur.description]
        cur_key = None
        meta: dict = {}
        cells: list[dict] = []
        while True:
            batch = cur.fetchmany(200000)
            if not batch:
                break
            for r in batch:
                d = dict(zip(names, r))
                key = (d["rcept_no"], d["section_key"], d["table_idx"])
                if key != cur_key:
                    if cur_key is not None:
                        yield meta, cur_key[1], cur_key[2], cells
                    cur_key = key
                    meta = {k: d[k] for k in META_COLS}
                    meta["doc_parse_status"] = d["parse_status"]
                    cells = []
                cells.append(d)
        if cur_key is not None:
            yield meta, cur_key[1], cur_key[2], cells


def population_docs(
    con: duckdb.DuckDBPyConnection, index_path: Path, exclude: set[str]
) -> tuple[list[str], int, int]:
    all_docs = [
        r[0]
        for r in con.execute(
            f"select distinct rcept_no from read_parquet('{index_path}')"
        ).fetchall()
    ]
    docs = sorted(d for d in all_docs if d not in exclude)
    return docs, len(all_docs), len(all_docs) - len(docs)


def _assign_row(meta: dict, section_key: str, table_idx: int, a: dict) -> dict:
    src = bi.SRC_NOTE.get(section_key, section_key)
    return {
        "rcept_no": meta["rcept_no"],
        "section_key": section_key,
        "table_idx": table_idx,
        "target": a["target"],
        "adapter": a["adapter"],
        "tmpl_sig": a["tmpl_sig"],
        "n_body_rows": a["n_body_rows"],
        "n_periods": a["n_periods"],
        "unit_status": a["unit_status"],
        "rcpt_year": int(meta["rcept_dt"][:4]),
        "corp_cls": meta["corp_cls"],
        "era": era_of(meta["rcept_dt"]),
        "judged": a["judged"],
        "parse_status": a["parse_status"],
        "n_lead": a["n_lead"],
        "source": src,
        "rcept_dt": meta["rcept_dt"],
    }


def summarize(rows: list[dict], path: Path) -> list[dict]:
    """대상 × 어댑터별 표 수·비율(대상 안)·본문 행·시대·단위 분포."""
    by_t: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    agg: dict[tuple, dict] = {}
    for r in rows:
        k = (r["target"], r["adapter"])
        by_t[r["target"]][r["adapter"]] += 1
        a = agg.setdefault(
            k,
            {
                "n": 0,
                "rows": 0,
                "docs": set(),
                "le": 0,
                "ge": 0,
                "u": collections.Counter(),
                "judged": r["judged"],
                "partial": 0,
            },
        )
        a["n"] += 1
        a["rows"] += r["n_body_rows"]
        a["docs"].add(r["rcept_no"])
        a["le" if r["era"] == "le2022" else "ge"] += 1
        a["u"][r["unit_status"]] += 1
        a["partial"] += r["parse_status"] == "partial"
    out = []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "target",
                "adapter",
                "judged",
                "n_tables",
                "share_in_target",
                "n_docs",
                "n_body_rows",
                "tables_le2022",
                "tables_ge2023",
                "unit_krw",
                "unit_missing",
                "unit_conflict",
                "unit_foreign",
                "unit_nomoney",
                "parse_partial",
            ]
        )
        for (t, ad), a in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1]["n"])):
            tot = sum(by_t[t].values())
            u = a["u"]
            row = [
                t,
                ad,
                a["judged"],
                a["n"],
                round(a["n"] / tot, 4),
                len(a["docs"]),
                a["rows"],
                a["le"],
                a["ge"],
                u["krw"],
                u["missing"],
                u["conflict"],
                u["foreign"],
                u["nomoney"],
                a["partial"],
            ]
            w.writerow(row)
            out.append(
                dict(zip(["target", "adapter", "judged", "n_tables", "share_in_target"], row[:5]))
            )
    return out


def run_assign(root: Path, snapshot: str, out_dir: Path, limit: int | None = None) -> dict:
    t0 = time.time()
    snap = snapshot_dir(root, snapshot)
    index_path = snap / "kr_bizrpt_section_index.parquet"
    cell_glob = str(snap / "kr_bizrpt_table_cell" / "**" / "*.parquet")
    con = duckdb.connect()
    ex, ex_files = excluded_docs(root)
    docs, n_all, n_excl = population_docs(con, index_path, ex)
    if limit:
        docs = docs[:limit]
    rows: list[dict] = []
    n_tables_seen = 0
    for meta, sk, ti, cells in iter_tables(con, str(cell_glob), docs):
        n_tables_seen += 1
        a = bi.assign_table(sk, cells)
        if a is not None:
            rows.append(_assign_row(meta, sk, ti, a))
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(ASSIGN_FIELDS)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, out_dir / "interp_assignment.parquet", compression="zstd")
    summ = summarize(rows, out_dir / "interp_assignment_summary.tsv")
    manifest = {
        "name": "interp_assignment",
        "interp_version": bi.INTERP_VERSION,
        "code_rev": code_rev(),
        "created_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(
            timespec="seconds"
        ),
        "input_snapshot": str(snap),
        "input_manifest_sha256": hashlib.sha256((snap / "manifest.json").read_bytes()).hexdigest()
        if (snap / "manifest.json").exists()
        else None,
        "n_docs_in_index": n_all,
        "n_docs_excluded_verify_sample": n_excl,
        "excluded_files": ex_files,
        "n_docs_scanned": len(docs),
        "n_tables_scanned": n_tables_seen,
        "n_rows": len(rows),
        "sections_scanned": list(SECTIONS),
        "note": "표마다 대상·어댑터만 붙인 목록. 해석 값(이름·금액·기간 값)은 없다. verify_sample 문서의 셀은 읽지 않았다.",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "interp_assignment.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"manifest": manifest, "summary": summ}


def _arrow_type(col: str):
    if col in bi.BOOL_COLS:
        return pa.bool_()
    if col in bi.INT_COLS:
        return pa.int64()
    if col in bi.FLOAT_COLS:
        return pa.float64()
    return pa.string()


def run_interp(root: Path, snapshot: str, rcept_list: Path, out_dir: Path) -> dict:
    """지정 문서(``rcept_list``)만 해석한다. 목록 없이 전체를 돌리지 않는다."""
    t0 = time.time()
    snap = snapshot_dir(root, snapshot)
    cell_glob = str(snap / "kr_bizrpt_table_cell" / "**" / "*.parquet")
    ex, _ = excluded_docs(root)
    docs = sorted(read_rcept_set(rcept_list))
    bad = [d for d in docs if d in ex]
    if bad:
        raise SystemExit(f"verify_sample 문서 {len(bad)}건이 목록에 있다. 열지 않는다: {bad[:3]}")
    con = duckdb.connect()
    by_target: dict[str, list[dict]] = {t: [] for t in bi.TARGETS}
    assigns: list[dict] = []
    rows_by_table: dict[tuple, int] = collections.defaultdict(int)
    for meta, sk, ti, cells in iter_tables(con, cell_glob, docs):
        a = bi.assign_table(sk, cells)
        if a is None:
            continue
        ar = _assign_row(meta, sk, ti, a)
        assigns.append(ar)
        for r in bi.interp_table(meta, sk, ti, cells, a):
            by_target[a["target"]].append(r)
            rows_by_table[(meta["rcept_no"], sk, ti)] += 1
    bi.mark_customer_dups(by_target["customer"])
    bi.mark_table_dups(
        by_target["segment_sales"], ("cat_path_raw", "period_idx", "amount_raw", "row_kind")
    )
    bi.mark_table_dups(
        by_target["orderbook"],
        ("item_raw", "total_amt_num", "delivered_amt_num", "backlog_amt_num", "row_kind"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    names = {
        "customer": "kr_bizrpt_customer_row",
        "orderbook": "kr_bizrpt_orderbook_row",
        "segment_sales": "kr_bizrpt_segment_sales_row",
    }
    for t, rs in by_target.items():
        cols = bi.TABLE_COLS[t]
        schema = pa.schema([(c, _arrow_type(c)) for c in cols])
        pq.write_table(
            pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in rs], schema=schema),
            out_dir / f"{names[t]}.parquet",
            compression="zstd",
        )
        counts[t] = len(rs)
    doc_rows = bi.build_interp_doc(docs, assigns, rows_by_table)
    pq.write_table(
        pa.Table.from_pylist(doc_rows), out_dir / "kr_bizrpt_interp_doc.parquet", compression="zstd"
    )
    pq.write_table(
        pa.Table.from_pylist(assigns, schema=pa.schema(ASSIGN_FIELDS)),
        out_dir / "interp_assignment_subset.parquet",
        compression="zstd",
    )
    return {
        "rows": counts,
        "n_docs": len(docs),
        "n_tables": len(assigns),
        "elapsed_sec": round(time.time() - t0, 1),
    }


FULL_NAMES = {
    "customer": "kr_bizrpt_customer_row_all_candidates_diag",
    "orderbook": "kr_bizrpt_orderbook_row_all_candidates_diag",
    "segment_sales": "kr_bizrpt_segment_sales_row_all_candidates_diag",
}

PASSED_NAMES = {
    "customer": "kr_bizrpt_customer_row",
    "orderbook": "kr_bizrpt_orderbook_row",
    "segment_sales": "kr_bizrpt_segment_sales_row",
}
VERDICT_DATE = "2026-10-29"


def run_full(
    root: Path,
    snapshot: str,
    out_dir: Path,
    include_verify_sample: bool = False,
    limit: int | None = None,
) -> dict:
    """모집단 전체 해석(2단계). 후보 전부(판정 군·판정 밖 어댑터)를 ``*_all_candidates_diag``로 쓴다.

    ``*_all_candidates_diag``는 진단용이다. 메인 대조 판정(``bizinterp.PASSED``)을 통과한 군만 걸러
    ``kr_bizrpt_*_row.parquet``로 따로 쓴다(``derived/`` 승격 대상, 결정 8). 문서 묶음마다 해석하고
    바로 쓴다(메모리). ``include_verify_sample``을 켜야만 ``verify_sample/`` 문서를 포함한다.
    """
    t0 = time.time()
    snap = snapshot_dir(root, snapshot)
    index_path = snap / "kr_bizrpt_section_index.parquet"
    cell_glob = str(snap / "kr_bizrpt_table_cell" / "**" / "*.parquet")
    con = duckdb.connect()
    ex, ex_files = excluded_docs(root)
    docs, n_all, n_excl = population_docs(con, index_path, set() if include_verify_sample else ex)
    if limit:
        docs = docs[:limit]
    doc_status = dict(
        con.execute(
            "select parse_status, count(distinct rcept_no) from read_parquet(?) group by 1",
            [str(index_path)],
        ).fetchall()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    schemas = {t: pa.schema([(c, _arrow_type(c)) for c in bi.TABLE_COLS[t]]) for t in bi.TARGETS}
    for t in bi.TARGETS:
        writers[t] = pq.ParquetWriter(
            out_dir / f"{FULL_NAMES[t]}.parquet", schemas[t], compression="zstd"
        )
    counts: dict = collections.Counter()
    row_status: dict = collections.Counter()
    assigns: list[dict] = []
    rows_by_table: dict[tuple, int] = collections.defaultdict(int)
    try:
        for i in range(0, len(docs), CHUNK_DOCS):
            chunk = docs[i : i + CHUNK_DOCS]
            by_target: dict[str, list[dict]] = {t: [] for t in bi.TARGETS}
            for meta, sk, ti, cells in iter_tables(con, cell_glob, chunk):
                a = bi.assign_table(sk, cells)
                if a is None:
                    continue
                assigns.append(_assign_row(meta, sk, ti, a))
                for r in bi.interp_table(meta, sk, ti, cells, a):
                    by_target[a["target"]].append(r)
                    rows_by_table[(meta["rcept_no"], sk, ti)] += 1
            bi.mark_customer_dups(by_target["customer"])
            bi.mark_table_dups(
                by_target["segment_sales"], ("cat_path_raw", "period_idx", "amount_raw", "row_kind")
            )
            bi.mark_table_dups(
                by_target["orderbook"],
                ("item_raw", "total_amt_num", "delivered_amt_num", "backlog_amt_num", "row_kind"),
            )
            for t, rs in by_target.items():
                cols = bi.TABLE_COLS[t]
                if rs:
                    writers[t].write_table(
                        pa.Table.from_pylist(
                            [{c: r.get(c) for c in cols} for r in rs], schema=schemas[t]
                        )
                    )
                counts[t] += len(rs)
                for r in rs:
                    row_status[(t, r["parse_status"])] += 1
    finally:
        for w in writers.values():
            w.close()
    pq.write_table(
        pa.Table.from_pylist(assigns, schema=pa.schema(ASSIGN_FIELDS)),
        out_dir / "interp_assignment.parquet",
        compression="zstd",
    )
    summarize(assigns, out_dir / "interp_assignment_summary.tsv")
    doc_rows = bi.build_interp_doc(docs, assigns, rows_by_table)
    pq.write_table(
        pa.Table.from_pylist(doc_rows), out_dir / "kr_bizrpt_interp_doc.parquet", compression="zstd"
    )
    doc_status_counts = collections.Counter((r["target"], r["status"]) for r in doc_rows)
    tbl_ad = collections.Counter((a["target"], a["adapter"]) for a in assigns)
    passed_counts: dict[str, int] = {}
    passed_by_group: dict[str, int] = {}
    for t in bi.TARGETS:
        diag = pq.read_table(out_dir / f"{FULL_NAMES[t]}.parquet")
        sub = diag.filter(pc.equal(diag["passed"], True))
        pq.write_table(sub, out_dir / f"{PASSED_NAMES[t]}.parquet", compression="zstd")
        passed_counts[t] = sub.num_rows
        for g, n in collections.Counter(sub["tmpl_group"].to_pylist()).items():
            passed_by_group[g] = n
    manifest = {
        "name": "interp_full_diag_and_passed_rows",
        "note": (
            "*_all_candidates_diag.parquet = 진단용 전체 후보 행(쓰지 않음, derived/에 안 올린다). "
            "judged 열은 JUDGED 상수(판정 후보 군) 표시일 뿐 판정 결과가 아니다. "
            "kr_bizrpt_{customer,orderbook,segment_sales}_row.parquet = 메인 대조 판정 합격 군만"
            "(passed==True, diag와 같은 스키마, derived 승격 대상)."
        ),
        "verdict_date": VERDICT_DATE,
        "passed_groups": sorted(f"{t}:{a}" for t, a in bi.PASSED),
        "judged_not_passed_groups": sorted(f"{t}:{a}" for t, a in bi.JUDGED - bi.PASSED),
        "passed_rows_by_target": passed_counts,
        "row_counts_by_file": {f"{FULL_NAMES[t]}.parquet": counts[t] for t in bi.TARGETS}
        | {f"{PASSED_NAMES[t]}.parquet": passed_counts[t] for t in bi.TARGETS},
        "passed_rows_by_group": dict(sorted(passed_by_group.items())),
        "interp_version": bi.INTERP_VERSION,
        "code_rev": code_rev(),
        "created_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(
            timespec="seconds"
        ),
        "input_snapshot": str(snap),
        "input_manifest_sha256": hashlib.sha256((snap / "manifest.json").read_bytes()).hexdigest()
        if (snap / "manifest.json").exists()
        else None,
        "n_docs_in_index": n_all,
        "include_verify_sample": include_verify_sample,
        "n_docs_excluded": 0 if include_verify_sample else n_excl,
        "excluded_files": [] if include_verify_sample else ex_files,
        "n_docs_run": len(docs),
        "doc_parse_status_index": {str(k): v for k, v in doc_status.items()},
        "n_tables_assigned": len(assigns),
        "tables_by_target_adapter": {f"{t}:{a}": n for (t, a), n in sorted(tbl_ad.items())},
        "rows_by_target": dict(counts),
        "rows_by_target_parse_status": {f"{t}:{s}": n for (t, s), n in sorted(row_status.items())},
        "interp_doc_status": {f"{t}:{s}": n for (t, s), n in sorted(doc_status_counts.items())},
        "outputs": {t: f"{FULL_NAMES[t]}.parquet" for t in bi.TARGETS}
        | {f"{t}_passed": f"{PASSED_NAMES[t]}.parquet" for t in bi.TARGETS}
        | {"doc": "kr_bizrpt_interp_doc.parquet", "assignment": "interp_assignment.parquet"},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "manifest_full.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assign", help="모집단 표마다 대상·어댑터 목록(값 없음)")
    a.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    a.add_argument("--out", type=Path, default=None)
    a.add_argument("--limit", type=int, default=None, help="시험용: 앞 N 문서만")
    i = sub.add_parser("interp", help="지정 문서만 해석 (문서 목록 필수)")
    i.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    i.add_argument("--rcept-list", type=Path, required=True)
    i.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("full", help="모집단 전체 해석(진단용 *_all_candidates_diag)")
    f.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    f.add_argument("--out", type=Path, default=None)
    f.add_argument(
        "--include-verify-sample",
        action="store_true",
        help="verify_sample/ 문서까지 포함(기본: 제외)",
    )
    f.add_argument("--limit", type=int, default=None, help="시험용: 앞 N 문서만")
    args = ap.parse_args(argv)
    root = data_root()
    if args.cmd == "full":
        out = args.out or (root / "kr" / "output" / OUT_NAME)
        print(
            json.dumps(
                run_full(root, args.snapshot, out, args.include_verify_sample, args.limit),
                ensure_ascii=False,
                indent=1,
            )
        )
        return
    if args.cmd == "assign":
        out = args.out or (root / "kr" / "output" / OUT_NAME)
        res = run_assign(root, args.snapshot, out, args.limit)
        print(json.dumps(res["manifest"], ensure_ascii=False, indent=1))
    else:
        res = run_interp(root, args.snapshot, args.rcept_list, args.out)
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
