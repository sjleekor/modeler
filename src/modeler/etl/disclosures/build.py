"""a(공급계약 체결)·ga(해지·진행사항) 원문 전체 구조화 실행.

입력은 ``<root>/kr/raw/dart_doc_original/{a,ga}`` 의 zip이고, 출력은
``<root>/kr/output/disclosure_structuring_v1/`` 의 parquet·manifest다. 경로는 환경변수
``STOCK_DATA_ROOT``(기본 ``../stock_data``)로 받는다. 이 모듈은 ``derived/``에 쓰지 않는다.

    python -m modeler.etl.disclosures.build [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from modeler.etl.disclosures import PARSER_VERSION
from modeler.etl.disclosures.calendar_util import NextTradingDay
from modeler.etl.disclosures.contract import core_status, parse_contract_html
from modeler.etl.disclosures.counterparty import build_pit_index, classify_counterparty
from modeler.etl.disclosures.docparse import read_zip_bytes
from modeler.etl.disclosures.linking import Linker, title_class

OUT_NAME = "disclosure_structuring_v1"
MASTER_SNAPSHOT = "2026-10-23"

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
]
EVENT_FIELDS = [
    "form_variant",
    "contract_name",
    "contract_name_src",
    "contract_kind",
    "amount",
    "amount_raw",
    "amount_label",
    "amount_unit",
    "amount_unit_assumed",
    "amount_fixed",
    "amount_fixed_raw",
    "amount_cond",
    "amount_cond_raw",
    "is_conditional",
    "recent_sales",
    "recent_sales_raw",
    "ratio_pct",
    "ratio_pct_raw",
    "large_corp",
    "counterparty_raw",
    "cp_recent_sales",
    "cp_recent_sales_raw",
    "cp_business",
    "cp_relation",
    "cp_prior_3y",
    "region",
    "period_start",
    "period_start_raw",
    "period_end",
    "period_end_raw",
    "terms",
    "advance_flag",
    "payment_terms",
    "mode_inhouse",
    "mode_outsourced",
    "contract_date",
    "contract_date_raw",
    "hold_reason",
    "hold_until",
    "hold_until_raw",
    "subsidiary_notice",
    "other_text",
    "fx_hint",
    "fx_amount_raw",
    "vat_hint",
    "related_raw",
    "n_related_pairs",
    "ratio_check_diff",
    "ratio_check_ok",
]
CANCEL_FIELDS = [
    "form_variant",
    "contract_name",
    "contract_kind",
    "amount",
    "amount_raw",
    "cancel_amount",
    "cancel_amount_raw",
    "cancel_amount_label",
    "cancel_amount_unit",
    "recent_sales",
    "recent_sales_raw",
    "ratio_pct",
    "ratio_pct_raw",
    "counterparty_raw",
    "region",
    "period_start",
    "period_start_raw",
    "period_end",
    "period_end_raw",
    "cancel_date",
    "cancel_date_raw",
    "cancel_reason",
    "subsidiary_notice",
    "other_text",
    "fx_hint",
    "vat_hint",
    "related_raw",
    "n_related_pairs",
    "ratio_check_diff",
    "ratio_check_ok",
]
NOTICE_FIELDS = [
    "title_in_doc",
    "decision_date",
    "decision_date_raw",
    "main_text_len",
    "related_raw",
    "n_related_pairs",
    "notice_kind",
]


def data_root() -> Path:
    return Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data")).resolve()


def code_rev() -> str:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def read_targets(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def group_of(report_nm: str, grp: str) -> str:
    """a → event, ga → cancel(제목에 해지) 또는 notice(진행사항·거래정지·기타)."""
    if grp == "a":
        return "event"
    compact = "".join(report_nm.split())
    return "cancel" if compact.startswith("단일판매") and "해지" in compact else "notice"


def _parse_one(args: tuple) -> dict:
    """worker: zip 하나 → 행 dict. 예외는 fail 행으로 돌려주고 조용히 빠뜨리지 않는다."""
    meta, zip_path, kind, rev = args
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")
    row = {
        k: meta.get(k)
        for k in ("rcept_no", "rcept_dt", "corp_code", "stock_code", "corp_cls", "report_nm")
    }
    row.update(
        src_zip_sha256=None,
        src_member=None,
        src_encoding=None,
        src_declared_encoding=None,
        src_encoding_mismatch=None,
        parser_version=f"{PARSER_VERSION}+{rev}",
        parse_status="fail",
        parse_err=None,
        parse_fail_reason=None,
        parsed_at=now,
        kind=kind,
    )
    try:
        data = Path(zip_path).read_bytes()
        row["src_zip_sha256"] = hashlib.sha256(data).hexdigest()
        doc = read_zip_bytes(data, meta["rcept_no"])
        row.update(
            src_member=doc.member,
            src_encoding=doc.encoding,
            src_declared_encoding=doc.declared_encoding,
            src_encoding_mismatch=doc.encoding_mismatch,
        )
        f = parse_contract_html(doc.text)
        if "\ufffd" in doc.text:  # 원천 파일에 이미 깨진 글자(U+FFFD)가 들어 있다
            f["parse_err"] = ",".join(
                filter(None, [f.get("parse_err"), "source_replacement_chars"])
            )
        f.pop("_labels", None)
        pairs = f.pop("related_pairs")
        f["n_related_pairs"] = len(pairs)
        row["pairs"] = pairs
        row.update(f)
        if kind == "notice":
            row["notice_kind"] = (
                "trading_halt"
                if "거래정지" in meta["report_nm"]
                else "progress"
                if "진행" in meta["report_nm"]
                else "other"
            )
            row["parse_status"] = (
                "ok" if f.get("title_in_doc") is not None or f["n_rows"] else "fail"
            )
            if row["parse_status"] == "fail":
                row["parse_fail_reason"] = "no_rows"
        else:
            row["parse_status"] = core_status(f, kind == "cancel")
            if row["parse_status"] == "fail":
                row["parse_fail_reason"] = "unknown_form_or_no_rows"
        if doc.encoding == "replace":
            row["parse_err"] = ",".join(filter(None, [row.get("parse_err"), "decode_replace"]))
    except Exception as e:  # noqa: BLE001 - 어떤 예외든 행은 남긴다
        row["parse_fail_reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


def load_trade_dates(raw_dir: Path) -> list[dt.date]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    d = ds.dataset(str(raw_dir / "daily_ohlcv"), format="parquet", partitioning="hive")
    col = d.to_table(columns=["trade_date"]).column("trade_date")
    return [pd.Timestamp(x).date() for x in pc.unique(col).to_pylist()]


def _ymd(s: str) -> dt.date:
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def run(limit: int | None = None, procs: int = 4) -> dict:
    t0 = time.time()
    root = data_root()
    doc_root = root / "kr/raw/dart_doc_original"
    raw_dir = root / f"kr/raw/raw_postgres/snapshot_date={MASTER_SNAPSHOT}/source=sj2_remote"
    out_dir = root / "kr/output" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    rev = code_rev()

    jobs, n_targets = [], {}
    for grp in ("a", "ga"):
        tg = read_targets(doc_root / grp / "targets.tsv")
        n_targets[grp] = len(tg)
        for m in tg[:limit] if limit else tg:
            jobs.append(
                (
                    m,
                    str(doc_root / grp / "raw" / f"{m['rcept_no']}.zip"),
                    group_of(m["report_nm"], grp),
                    rev,
                )
            )
    with Pool(procs) as p:
        rows = p.map(_parse_one, jobs, chunksize=200)
    t_parse = time.time() - t0

    cal = NextTradingDay(load_trade_dates(raw_dir))
    for r in rows:
        d = _ymd(r["rcept_dt"])
        nd, assumed = cal.next_after(d)
        r["available_from"] = nd.isoformat()
        r["available_from_assumed"] = assumed

    # 상대방 연결
    dm = pd.read_parquet(
        glob.glob(str(raw_dir / "dart_corp_master/**/*.parquet"), recursive=True)[0],
        columns=["corp_code", "ticker", "corp_name", "stock_name"],
    )
    sm = pd.read_parquet(
        glob.glob(str(raw_dir / "stock_master/**/*.parquet"), recursive=True)[0],
        columns=["ticker", "name"],
    )
    snap_df = pd.read_parquet(raw_dir / "stock_master_snapshot")
    snap_items = pd.read_parquet(
        raw_dir / "stock_master_snapshot_items", columns=["snapshot_id", "ticker", "name"]
    )
    idx = build_pit_index(dm, sm, snap_df, snap_items, MASTER_SNAPSHOT)

    ev_rows = [r for r in rows if r["kind"] == "event"]
    ca_rows = [r for r in rows if r["kind"] == "cancel"]
    no_rows = [r for r in rows if r["kind"] == "notice"]
    linker = Linker([r for r in ev_rows if r["parse_status"] != "fail"], rows)

    cp_out = []
    for r in ev_rows + ca_rows:
        if r["parse_status"] == "fail":
            continue
        for part in classify_counterparty(r.get("counterparty_raw"), idx, r["rcept_dt"]):
            cp_out.append(
                {
                    "rcept_no": r["rcept_no"],
                    "rcept_dt": r["rcept_dt"],
                    "corp_code": r["corp_code"],
                    "stock_code": r["stock_code"],
                    "src": r["kind"],
                    **part,
                    "master_snapshot": MASTER_SNAPSHOT,
                }
            )

    rel_out = []
    for r in rows:
        for i, (d, title) in enumerate(r.get("pairs") or []):
            rn, n = linker.link_related_pair(r["corp_code"], d, title, r["rcept_no"])
            rel_out.append(
                {
                    "rcept_no": r["rcept_no"],
                    "idx": i,
                    "related_date": d,
                    "related_title_raw": title,
                    "title_class": title_class(title),
                    "related_rcept_no": rn,
                    "n_candidates": n,
                }
            )

    for r in ca_rows:
        if r["parse_status"] == "fail":
            r.update(
                first_rcept_no=None, link_method="parse_fail", n_candidates=0, first_pair_date=None
            )
        else:
            r.update(linker.link_cancel(r, r.get("pairs") or []))

    # 자회사·모회사 중복 표시(합산 금지 표시만)
    groups: dict[tuple, list[dict]] = {}
    for r in ev_rows:
        if r["parse_status"] != "fail" and r.get("contract_date") and r.get("amount") is not None:
            nm = "".join((r.get("contract_name") or "").split()).lower()
            groups.setdefault((r["contract_date"], nm, r["amount"]), []).append(r)
    for k, g in groups.items():
        if len({x["corp_code"] for x in g}) > 1 and any(x.get("subsidiary_notice") for x in g):
            gid = hashlib.sha1(repr(k).encode()).hexdigest()[:12]
            for x in g:
                x["dup_group_id"] = gid

    def table(rs, fields, extra=()):
        cols = COMMON + list(fields) + list(extra) + ["available_from", "available_from_assumed"]
        return pd.DataFrame([{c: r.get(c) for c in cols} for r in rs], columns=cols)

    ev = table(ev_rows, EVENT_FIELDS, ["dup_group_id"])
    ev["superseded_by"] = None
    ca = table(
        ca_rows, CANCEL_FIELDS, ["first_rcept_no", "link_method", "n_candidates", "first_pair_date"]
    )
    no = table(no_rows, NOTICE_FIELDS)
    rel = pd.DataFrame(rel_out)
    cp = pd.DataFrame(cp_out)
    for df, name in (
        (ev, "kr_contract_event"),
        (ca, "kr_contract_cancel"),
        (no, "kr_contract_notice_free"),
        (rel, "kr_contract_related"),
        (cp, "kr_contract_counterparty_link"),
    ):
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    mf = {
        "output": OUT_NAME,
        "created_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(
            timespec="seconds"
        ),
        "modeler_commit": rev,
        "parser_version": f"{PARSER_VERSION}+{rev}",
        "inputs": {
            "doc_root": str(doc_root),
            "targets": {g: str(doc_root / g / "targets.tsv") for g in ("a", "ga")},
            "zip_sha256_files": {
                "server": str(doc_root / "server_sha256.txt"),
                "mac": str(doc_root / "mac_sha256.txt"),
            },
            "master_snapshot": MASTER_SNAPSHOT,
            "master_dir": str(raw_dir),
            "stock_master_snapshot": {
                "dir": str(raw_dir / "stock_master_snapshot"),
                "items_dir": str(raw_dir / "stock_master_snapshot_items"),
                "n_snapshots_raw": len(snap_df),
                "n_snapshot_dates_used": len(idx.snaps),
                "as_of_min": snap_df.as_of_date.min().isoformat(),
                "as_of_max": snap_df.as_of_date.max().isoformat(),
                "rule": "접수일 이하 최신 스냅샷(그 전이면 첫 것), 같은 날짜는 fetched_at 최신",
            },
            "calendar": "daily_ohlcv.trade_date 고유 날짜 + 마지막 날 이후 평일 가정",
        },
        "n_targets": n_targets,
        "rows": {
            "kr_contract_event": len(ev),
            "kr_contract_cancel": len(ca),
            "kr_contract_notice_free": len(no),
            "kr_contract_related": len(rel),
            "kr_contract_counterparty_link": len(cp),
        },
        "limit": limit,
        "seconds_parse": round(t_parse, 1),
        "seconds_total": round(time.time() - t0, 1),
        "python": sys.version.split()[0],
        "note": "텍스트 모델·LLM 추출 없음. derived/에는 쓰지 않았다(§7 통과 전).",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(mf, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return mf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(run(a.limit, a.procs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
