"""사업보고서(b) 절 위치 표·표 셀 표 전체 실행.

입력은 ``<root>/kr/raw/dart_doc_original/b`` 의 zip 30,001개(``targets.tsv``), 출력은
``<root>/kr/output/disclosure_structuring_v1/`` 의 ``kr_bizrpt_section_index.parquet``,
``kr_bizrpt_table_cell/rcpt_year=YYYY/part-*.parquet``, ``manifest_bizrpt.json`` 이다.
1단계(a·ga) 표와 ``manifest.json``은 건드리지 않는다. 경로는 ``STOCK_DATA_ROOT``
(기본 ``../stock_data``)로 받고 ``derived/``에는 쓰지 않는다.

    python -m modeler.etl.disclosures.biz_build [--limit N] [--procs 6]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from modeler.etl.disclosures.bizparse import BIZ_PARSER_VERSION, SECTION_KEYS, parse_biz_text
from modeler.etl.disclosures.build import (
    MASTER_SNAPSHOT,
    _ymd,
    code_rev,
    data_root,
    load_trade_dates,
    read_targets,
)
from modeler.etl.disclosures.calendar_util import NextTradingDay
from modeler.etl.disclosures.docparse import read_zip_bytes

OUT_NAME = "disclosure_structuring_v1"
INDEX_NAME = "kr_bizrpt_section_index"
CELL_NAME = "kr_bizrpt_table_cell"
DOC_SECTION = "_doc"  # 문서 전체가 fail일 때 셀 표에 남기는 자리표시 행의 section_key

COMMON = [
    "rcept_no",
    "rcept_dt",
    "corp_code",
    "stock_code",
    "corp_cls",
    "report_nm",
    "src_zip_sha256",
    "src_member",
    "src_encoding",
    "src_declared_encoding",
    "src_encoding_mismatch",
    "parser_version",
    "parse_status",
    "parse_err",
    "parse_fail_reason",
    "parsed_at",
    "available_from",
    "available_from_assumed",
]
INDEX_FIELDS = [
    "section_key",
    "found",
    "method",
    "miss_reason",
    "biz_form",
    "heading_raw",
    "char_start",
    "char_end",
    "char_len",
    "n_tables",
    "n_cells",
    "unit_raw",
    "n_tables_with_unit",
]
CELL_FIELDS = [
    "section_key",
    "table_idx",
    "row_idx",
    "col_idx",
    "rowspan",
    "colspan",
    "is_header",
    "header_path_raw",
    "row_label_raw",
    "text_raw",
    "num_value",
    "num_has_pct",
    "unit_raw",
    "unit_src",
    "tbl_n_rows",
    "tbl_n_cols",
    "tbl_header_rule",
    "tbl_has_nested",
]

_S, _I, _B, _F = pa.string(), pa.int64(), pa.bool_(), pa.float64()
COMMON_TYPES = {
    "src_encoding_mismatch": _B,
    "available_from_assumed": _B,
}
INDEX_TYPES = {
    "found": _B,
    "char_start": _I,
    "char_end": _I,
    "char_len": _I,
    "n_tables": _I,
    "n_cells": _I,
    "n_tables_with_unit": _I,
}
CELL_TYPES = {
    "table_idx": _I,
    "row_idx": _I,
    "col_idx": _I,
    "rowspan": _I,
    "colspan": _I,
    "is_header": _B,
    "num_value": _F,
    "num_has_pct": _B,
    "tbl_n_rows": _I,
    "tbl_n_cols": _I,
    "tbl_has_nested": _B,
}


def _schema(fields: list[str], types: dict) -> pa.Schema:
    return pa.schema([(f, types.get(f, _S)) for f in fields])


COMMON_SCHEMA = _schema(COMMON, COMMON_TYPES)
INDEX_SCHEMA = _schema(COMMON + INDEX_FIELDS, {**COMMON_TYPES, **INDEX_TYPES})
CELL_SCHEMA = _schema(COMMON + CELL_FIELDS, {**COMMON_TYPES, **CELL_TYPES})


def _now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")


def parse_one(args: tuple) -> dict:
    """worker: zip 하나 → ``{"doc", "index"(키별 행), "cells"(튜플)}``. 예외는 fail로 남긴다."""
    meta, zip_path, rev = args
    doc = {
        k: meta.get(k)
        for k in ("rcept_no", "rcept_dt", "corp_code", "stock_code", "corp_cls", "report_nm")
    }
    doc.update(
        src_zip_sha256=None,
        src_member=None,
        src_encoding=None,
        src_declared_encoding=None,
        src_encoding_mismatch=None,
        parser_version=f"{BIZ_PARSER_VERSION}+{rev}",
        parse_status="fail",
        parse_err=None,
        parse_fail_reason=None,
        parsed_at=_now(),
        available_from=meta["available_from"],
        available_from_assumed=meta["available_from_assumed"],
    )
    index: list[dict] = []
    cells: list[tuple] = []
    try:
        data = Path(zip_path).read_bytes()
        doc["src_zip_sha256"] = hashlib.sha256(data).hexdigest()
        raw = read_zip_bytes(data, meta["rcept_no"])
        doc.update(
            src_member=raw.member,
            src_encoding=raw.encoding,
            src_declared_encoding=raw.declared_encoding,
            src_encoding_mismatch=raw.encoding_mismatch,
        )
        out = parse_biz_text(raw.text)
        errs = list(out["flags"])
        if raw.encoding == "replace":
            errs.append("decode_replace")
        if "�" in raw.text:
            errs.append("source_replacement_chars")
        doc["parse_err"] = ",".join(sorted(set(errs))) or None
        partial = {"no_biz_section", "unclosed_table", "num_conv_err", "decode_replace"}
        doc["parse_status"] = "partial" if partial & set(errs) else "ok"
        n_cells: dict[str, int] = {}
        for c in out["cells"]:
            n_cells[c["section_key"]] = n_cells.get(c["section_key"], 0) + 1
            cells.append(tuple(c[f] for f in CELL_FIELDS))
        for key in SECTION_KEYS:
            s = out["sections"][key]
            index.append(
                {
                    "section_key": key,
                    **{k: s[k] for k in INDEX_FIELDS if k in s},
                    "biz_form": out["biz_form"],
                    "n_cells": n_cells.get(key, 0) if s["found"] else None,
                }
            )
    except Exception as e:  # noqa: BLE001 - 어떤 예외든 행은 남긴다
        doc["parse_fail_reason"] = f"{type(e).__name__}: {str(e)[:120]}"
        index = [
            {
                "section_key": key,
                "found": False,
                "method": None,
                "miss_reason": "parse_fail",
            }
            for key in SECTION_KEYS
        ]
        cells = [
            (
                DOC_SECTION,
                -1,
                -1,
                -1,
                1,
                1,
                False,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        ]
    return {"doc": doc, "index": index, "cells": cells}


class _CellWriter:
    """접수 연도별 parquet 조각 쓰기. 메모리를 아끼려고 ``flush_rows`` 넘으면 바로 쓴다."""

    def __init__(self, root: Path, flush_rows: int = 600_000) -> None:
        self.root = root
        self.flush_rows = flush_rows
        self.buf: dict[str, tuple[list[dict], list[tuple]]] = {}
        self.n_rows: dict[str, int] = {}
        self.n_parts: dict[str, int] = {}

    def add(self, year: str, doc: dict, cells: list[tuple]) -> None:
        docs, rows = self.buf.setdefault(year, ([], []))
        di = len(docs)
        docs.append(doc)
        rows.extend((di, *c) for c in cells)
        if len(rows) >= self.flush_rows:
            self.flush(year)

    def flush(self, year: str) -> None:
        docs, rows = self.buf.pop(year, ([], []))
        if not rows:
            return
        di = pa.array([r[0] for r in rows], type=pa.int64())
        cols = []
        for f in COMMON:
            arr = pa.array([d[f] for d in docs], type=CELL_SCHEMA.field(f).type)
            cols.append(arr.take(di))
        for j, f in enumerate(CELL_FIELDS, start=1):
            cols.append(pa.array([r[j] for r in rows], type=CELL_SCHEMA.field(f).type))
        t = pa.Table.from_arrays(cols, schema=CELL_SCHEMA)
        d = self.root / f"rcpt_year={year}"
        d.mkdir(parents=True, exist_ok=True)
        n = self.n_parts.get(year, 0)
        pq.write_table(t, d / f"part-{n:04d}.parquet", compression="zstd")
        self.n_parts[year] = n + 1
        self.n_rows[year] = self.n_rows.get(year, 0) + len(rows)

    def close(self) -> None:
        for y in list(self.buf):
            self.flush(y)


def run(
    limit: int | None = None,
    procs: int = 6,
    out_name: str = OUT_NAME,
    exclude_file: str | None = None,
) -> dict:
    t0 = time.time()
    root = data_root()
    doc_root = root / "kr/raw/dart_doc_original"
    raw_dir = root / f"kr/raw/raw_postgres/snapshot_date={MASTER_SNAPSHOT}/source=sj2_remote"
    out_dir = root / "kr/output" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = out_dir / CELL_NAME
    if cell_dir.exists():  # 이 모듈이 만든 폴더만 지운다
        shutil.rmtree(cell_dir)
    cell_dir.mkdir()
    rev = code_rev()

    targets = read_targets(doc_root / "b" / "targets.tsv")
    n_targets = len(targets)
    if exclude_file:
        ex = set(Path(exclude_file).read_text().split())
        targets = [m for m in targets if m["rcept_no"] not in ex]
    if limit:
        targets = targets[:limit]

    cal = NextTradingDay(load_trade_dates(raw_dir))
    jobs = []
    for m in targets:
        nd, assumed = cal.next_after(_ymd(m["rcept_dt"]))
        m = {**m, "available_from": nd.isoformat(), "available_from_assumed": assumed}
        jobs.append((m, str(doc_root / "b" / "raw" / f"{m['rcept_no']}.zip"), rev))

    writer = _CellWriter(cell_dir)
    index_rows: list[dict] = []
    status: dict[str, int] = {}
    with Pool(procs) as p:
        for r in p.imap(parse_one, jobs, chunksize=8):
            doc = r["doc"]
            status[doc["parse_status"]] = status.get(doc["parse_status"], 0) + 1
            for s in r["index"]:
                index_rows.append({**doc, **s})
            writer.add(doc["rcept_dt"][:4], doc, r["cells"])
    writer.close()
    t_parse = time.time() - t0

    cols = COMMON + INDEX_FIELDS
    tbl = pa.Table.from_pylist(
        [{c: r.get(c) for c in cols} for r in index_rows], schema=INDEX_SCHEMA
    )
    pq.write_table(tbl, out_dir / f"{INDEX_NAME}.parquet", compression="zstd")

    mf = {
        "output": out_name,
        "stage": "2-b (사업보고서 절 위치 표·표 셀 표, 1차 원문 보존)",
        "created_at": _now(),
        "modeler_commit": rev,
        "parser_version": f"{BIZ_PARSER_VERSION}+{rev}",
        "inputs": {
            "doc_root": str(doc_root / "b"),
            "targets": str(doc_root / "b" / "targets.tsv"),
            "zip_sha256_files": {
                "server": str(doc_root / "b_server_sha256.txt"),
                "mac": str(doc_root / "b_mac_sha256.txt"),
            },
            "rsync_done_marker": str(doc_root / ".rsync_b_done"),
            "calendar": "daily_ohlcv.trade_date 고유 날짜 + 마지막 날 이후 평일 가정",
            "calendar_snapshot": MASTER_SNAPSHOT,
        },
        "section_keys": list(SECTION_KEYS),
        "n_targets": n_targets,
        "n_processed": len(targets),
        "exclude_file": exclude_file,
        "limit": limit,
        "rows": {
            INDEX_NAME: len(index_rows),
            CELL_NAME: sum(writer.n_rows.values()),
        },
        "cell_rows_by_year": dict(sorted(writer.n_rows.items())),
        "cell_parts_by_year": dict(sorted(writer.n_parts.items())),
        "parse_status": status,
        "seconds_parse": round(t_parse, 1),
        "seconds_total": round(time.time() - t0, 1),
        "procs": procs,
        "python": sys.version.split()[0],
        "note": "텍스트 모델·LLM 추출 없음. 절 본문 텍스트는 저장하지 않음. derived/ 쓰기 없음.",
    }
    (out_dir / "manifest_bizrpt.json").write_text(
        json.dumps(mf, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return mf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--exclude-file", default=None, help="제외할 rcept_no 목록(개발용)")
    a = ap.parse_args()
    print(
        json.dumps(run(a.limit, a.procs, a.out_name, a.exclude_file), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
