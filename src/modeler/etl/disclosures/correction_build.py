"""정정 공시(c) 구조화 실행과 원공시 연결 (계획 13 §3~§6).

입력은 ``<root>/kr/raw/dart_doc_original/c`` 의 zip·``targets.tsv``, 접수 목록(raw_postgres 스냅샷),
a 구조화 산출(``kr/derived/disclosure_struct/snapshot_date=...``, 읽기만)이고, 출력은
``<root>/kr/output/disclosure_correction_v1/`` 다. 이 모듈은 ``derived/``에 쓰지 않는다.
경로는 환경변수로 받는다(기본값만 ``../stock_data``).

    STOCK_DATA_ROOT                   기본 ../stock_data
    DISCLOSURE_CORR_RECEIPT_SNAPSHOT  기본 2026-10-23 (접수 목록 raw 스냅샷 날짜)
    DISCLOSURE_CORR_A_SNAPSHOT        기본 2026-10-28 (a 구조화 스냅샷 날짜, 읽기만)

    python -m modeler.etl.disclosures.correction_build [--limit N]

연결 규칙(§4)은 ``link_correction`` 과 ``follow_chain`` 에 있다.
퍼지·계약명·금액 후보 줄이기는 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from modeler.etl.disclosures.build import _ymd, load_trade_dates
from modeler.etl.disclosures.calendar_util import NextTradingDay
from modeler.etl.disclosures.correction import (
    CORRECTION_PARSER_VERSION,
    classify_item,
    extract_links,
    parse_correction_zip,
    parse_dates,
    title_family,
    to_num,
)
from modeler.etl.disclosures.docparse import read_zip_bytes

OUT_NAME = "disclosure_correction_v1"
FIRST_COLLECTED = "20200305"  # c 수집 시작일. 이 날 앞 정정은 원문이 없다.
MAX_CHAIN = 50  # 결정 2026-11-01: 10 초과 383건이 순환 0이라 상한만 올림
KIND_DIGITS = "8960"  # KIND 번호의 9번째 자리를 이 값으로 바꾸면 DART 번호 후보
MARKET_DIGIT = {"Y": "8", "K": "9"}
FAMILY = ("original", "correction")  # 체결 계열 (해지·첨부·그 밖은 아님)


def data_root() -> Path:
    return Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data")).resolve()


def receipt_snapshot() -> str:
    return os.environ.get("DISCLOSURE_CORR_RECEIPT_SNAPSHOT", "2026-10-23")


def a_snapshot() -> str:
    return os.environ.get("DISCLOSURE_CORR_A_SNAPSHOT", "2026-10-28")


def receipt_path(root: Path) -> Path:
    return (
        root
        / f"kr/raw/raw_postgres/snapshot_date={receipt_snapshot()}/source=sj2_remote"
        / "dart_filing_receipt_raw/schema_version=1/part-000000.parquet"
    )


def a_dir(root: Path) -> Path:
    return root / f"kr/derived/disclosure_struct/snapshot_date={a_snapshot()}"


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True
        )
        return out.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def code_state() -> dict:
    """커밋 해시와 작업 트리(미커밋 변경 + 추적 안 된 파일) 해시."""
    here = Path(__file__).resolve().parent
    top = Path(_git(["rev-parse", "--show-toplevel"], here).strip() or here)
    head = _git(["rev-parse", "HEAD"], top).strip() or "unknown"
    h = hashlib.sha256()
    h.update(_git(["diff", "HEAD"], top).encode())
    for rel in sorted(_git(["ls-files", "--others", "--exclude-standard"], top).split("\n")):
        p = top / rel
        if rel and p.is_file() and "__pycache__" not in rel:
            h.update(rel.encode())
            h.update(p.read_bytes())
    dirty = bool(_git(["status", "--porcelain"], top).strip())
    return {
        "commit": head,
        "worktree_dirty": dirty,
        "worktree_sha256": h.hexdigest() if dirty else None,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], top).strip(),
    }


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")


# ---------------------------------------------------------------- 파싱 (worker)

COMMON = ["rcept_no", "rcept_dt", "corp_code", "stock_code", "corp_cls", "report_nm"]


def _parse_one(args: tuple) -> dict:
    meta, zip_path, version = args
    row = {k: meta.get(k) for k in COMMON}
    row.update(
        src_zip_sha256=None,
        src_member=None,
        src_encoding=None,
        src_declared_encoding=None,
        src_encoding_mismatch=None,
        parser_version=version,
        parse_status="fail",
        parse_err=None,
        parse_fail_reason=None,
        parsed_at=_now(),
    )
    try:
        data = Path(zip_path).read_bytes()
        row["src_zip_sha256"] = hashlib.sha256(data).hexdigest()
        f = parse_correction_zip(data, meta["rcept_no"], meta["report_nm"])
        row.update(f)
        if f["parse_status"] == "fail":
            row["parse_fail_reason"] = f["parse_err"]
    except Exception as e:  # noqa: BLE001 - 어떤 예외든 행은 남긴다
        row["parse_fail_reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


# ---------------------------------------------------------------- 접수 목록 색인


class Receipts:
    """접수 목록 색인. 체결 계열(원공시·기재정정)만 날짜 색인에 올린다."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.by_no: dict[str, tuple[str, str, str, str, str]] = {}
        self.fam_by_corp_date: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.corp_cls: dict[str, str] = {}
        for rn, cc, nm, d, cls in zip(
            df.rcept_no, df.corp_code, df.report_nm, df.rcept_dt, df.corp_cls, strict=True
        ):
            d8 = str(d).replace("-", "")
            fam = title_family(nm)
            self.by_no[rn] = (cc, nm, d8, fam, cls)
            if cls and cc not in self.corp_cls:
                self.corp_cls[cc] = cls
            if fam in FAMILY:
                self.fam_by_corp_date[(cc, d8)].append(rn)
        for v in self.fam_by_corp_date.values():
            v.sort()

    def family(self, rn: str) -> str | None:
        v = self.by_no.get(rn)
        return v[3] if v else None

    def date8(self, rn: str) -> str | None:
        v = self.by_no.get(rn)
        return v[2] if v else None


def kind_variants(no: str) -> list[str]:
    """KIND 접수번호 → DART 접수번호 후보(9번째 자리만 바꿈). 자기 자신이 이미 DART 번호일 수 있다."""  # noqa: E501
    out = []
    for d in KIND_DIGITS:
        v = no[:8] + d + no[9:]
        if v not in out:
            out.append(v)
    return out


def resolve_link(rc: Receipts, corp_code: str, kind_no: str) -> dict:
    """링크 하나 → 같은 회사 접수 목록과의 대응. 회사 조건은 꼭 건다(다른 회사 번호가 33%)."""
    found, other = [], 0
    for v in kind_variants(kind_no):
        rec = rc.by_no.get(v)
        if rec is None:
            continue
        if rec[0] == corp_code:
            found.append(v)
        else:
            other += 1
    fam = [v for v in found if rc.family(v) in FAMILY]
    return {"found": found, "n_other_corp": other, "family": fam}


# ---------------------------------------------------------------- 연결 (§4 1~2)


def link_correction(r: dict, rc: Receipts) -> dict:
    """정정 하나 → 바로 앞 공시(target). 후보가 정확히 하나일 때만 잇는다."""
    out = dict(
        link_stage1=None,
        link_method=None,
        target_rcept_no=None,
        n_link_candidates=0,
        n_date_candidates=0,
        link_rows=[],
    )
    if r["corr_kind"] != "value_correction":
        out["link_method"] = "not_value_correction"
        return out
    if r["parse_status"] == "fail":
        out["link_method"] = "parse_fail"
        return out
    sub = r.get("ref_submit_date")
    if not sub:
        out["link_method"] = "no_submit_date"
        return out
    sub8 = sub.replace("-", "")
    self_no = r["rcept_no"]
    corp = r["corp_code"]
    links = r["rel_links"]
    mark = MARKET_DIGIT.get(r.get("corp_cls") or "")
    matching: list[dict] = []
    for i, (no, text) in enumerate(links):
        res = resolve_link(rc, corp, no)
        td = parse_dates(text)
        d_no = no[:8]
        ok = d_no == sub8 or (td and td[0].replace("-", "") == sub8)
        digits = [v[8] for v in res["found"]]
        out["link_rows"].append(
            dict(
                rcept_no=self_no,
                src="correction",
                idx=i,
                kind_no=no,
                link_text=text,
                number_date=d_no,
                text_date=td[0] if td else None,
                date_match=bool(ok),
                n_found_same_corp=len(res["found"]),
                n_found_other_corp=res["n_other_corp"],
                found_rcept_nos=json.dumps(res["found"]),
                n_family=len(res["family"]),
                found_digit=",".join(digits),
                found_titles=json.dumps([rc.by_no[v][1] for v in res["found"]], ensure_ascii=False),
                market_digit_ok=(None if not (mark and digits) else all(d == mark for d in digits)),
            )
        )
        if ok:
            matching.append(res)
    if not links:
        out["link_stage1"] = "no_link"
    elif not matching:
        out["link_stage1"] = "link_date_mismatch"
    else:
        any_found = any(m["found"] for m in matching)
        cands = sorted(
            {
                v
                for m in matching
                for v in m["family"]
                if v < self_no and (rc.date8(v) or "") <= r["rcept_dt"]
            }
        )
        out["n_link_candidates"] = len(cands)
        if not any_found:
            out["link_stage1"] = "link_no_dart_match"
        elif not cands:
            out["link_stage1"] = "link_non_contract"
        elif len(cands) == 1:
            out.update(link_stage1="link_kind", link_method="link_kind", target_rcept_no=cands[0])
            return out
        else:
            out["link_stage1"] = "link_ambiguous"
    if out["link_stage1"] in ("link_non_contract", "link_ambiguous"):
        out["link_method"] = out["link_stage1"]
        return out
    # §4 2번 날짜 보조: 링크가 없거나(날짜 불일치·변환 결과 없음 포함) 날짜가 안 맞을 때
    if sub8 == r["rcept_dt"]:
        # 결정 2026-11-01: 제출일=접수일이면 같은 날 다른 계약일 수 있어 안 잇는다
        out["link_method"] = "date_self"
        return out
    cands = [v for v in rc.fam_by_corp_date.get((corp, sub8), []) if v < self_no and v != self_no]
    out["n_date_candidates"] = len(cands)
    if len(cands) == 1:
        out.update(link_method="date_unique", target_rcept_no=cands[0])
    elif not cands:
        out["link_method"] = "no_candidate"
    else:
        out["link_method"] = "date_multi"
    return out


# ---------------------------------------------------------------- 사슬 (§4 5번)


def follow_chain(start: str | None, self_no: str, c_target: dict[str, str | None], rc: Receipts):
    """target에서 시작해 원공시까지 올라간다 → (status, first, last, hops).

    ``c_target``은 c에 원문이 있는 정정의 {rcept_no: target 또는 None}이다.
    """
    if start is None:
        return "unlinked", None, None, 0
    seen = [self_no]
    cur, hops = start, 1
    while True:
        if cur in seen or hops > MAX_CHAIN:
            return "chain_loop", None, cur, hops
        fam = rc.family(cur)
        if fam == "original":
            return "reached_original", cur, cur, hops
        # 정정이다
        if cur in c_target:
            nxt = c_target[cur]
            if nxt is None:
                return "chain_broken_unlinked", None, cur, hops
            seen.append(cur)
            cur, hops = nxt, hops + 1
            continue
        d = rc.date8(cur) or ""
        if d < FIRST_COLLECTED:
            return "chain_broken_pre_2020_03_05", None, cur, hops
        return "chain_broken_not_in_c", None, cur, hops


# ---------------------------------------------------------------- 해지 v2 (§4 6번)


def link_cancel_v2(
    cancel: dict, links: list[tuple[str, str]], rc: Receipts, c_chain: dict[str, tuple]
) -> dict:
    """해지의 ``※ 관련공시`` 링크 → 체결 계열 접수 → 사슬 → 원공시. 기존 열은 건드리지 않는다."""
    res = dict(
        first_rcept_no_v2=None,
        link_method_v2=None,
        v2_resolved=None,
        v2_chain_last_rcept_no=None,
        v2_n_links=len(links),
    )
    if not links:
        res["link_method_v2"] = "v2_no_link"
        return res
    corp = cancel["corp_code"]
    mark = MARKET_DIGIT.get(rc.corp_cls.get(corp, ""))
    resolved, any_found = [], False
    link_rows = []
    for i, (no, text) in enumerate(links):
        r = resolve_link(rc, corp, no)
        any_found |= bool(r["found"])
        if len(r["family"]) == 1:
            resolved.append(r["family"][0])
        elif len(r["family"]) > 1:
            resolved.extend(
                r["family"]
            )  # 한 링크가 둘 이상으로 풀리면 후보 전부(아래서 다중이면 불연결)
        digits = [v[8] for v in r["found"]]
        link_rows.append(
            dict(
                rcept_no=cancel["rcept_no"],
                src="cancel",
                idx=i,
                kind_no=no,
                link_text=text,
                number_date=no[:8],
                text_date=(parse_dates(text) or [None])[0],
                date_match=None,
                n_found_same_corp=len(r["found"]),
                n_found_other_corp=r["n_other_corp"],
                found_rcept_nos=json.dumps(r["found"]),
                n_family=len(r["family"]),
                found_digit=",".join(digits),
                found_titles=json.dumps([rc.by_no[v][1] for v in r["found"]], ensure_ascii=False),
                market_digit_ok=(None if not (mark and digits) else all(d == mark for d in digits)),
            )
        )
    res["link_rows"] = link_rows
    resolved = sorted(set(resolved))
    res["v2_resolved"] = json.dumps(resolved)
    if not resolved:
        res["link_method_v2"] = "v2_link_non_contract" if any_found else "v2_link_no_dart_match"
        return res
    firsts, statuses, lasts = set(), [], []
    kinds = set()
    for v in resolved:
        if rc.family(v) == "original":
            firsts.add(v)
            kinds.add("v2_link_original")
            continue
        if v in c_chain:
            st, first, last = c_chain[v][:3]
            if first:
                firsts.add(first)
                kinds.add("v2_link_chain")
            else:
                statuses.append(st)
                lasts.append(last)
        else:
            d = rc.date8(v) or ""
            statuses.append(
                "chain_broken_pre_2020_03_05" if d < FIRST_COLLECTED else "chain_broken_not_in_c"
            )
            lasts.append(v)
    if len(firsts) == 1:
        res["first_rcept_no_v2"] = next(iter(firsts))
        res["link_method_v2"] = (
            "v2_link_original" if "v2_link_original" in kinds else "v2_link_chain"
        )
    elif len(firsts) > 1:
        res["link_method_v2"] = "v2_multi_first"
    else:
        order = [
            "chain_loop",
            "chain_broken_pre_2020_03_05",
            "chain_broken_unlinked",
            "chain_broken_not_in_c",
            "unlinked",
        ]
        st = min(statuses, key=lambda s: order.index(s) if s in order else 99)
        res["link_method_v2"] = "v2_" + st
        res["v2_chain_last_rcept_no"] = lasts[statuses.index(st)]
    return res


# ---------------------------------------------------------------- 실행


def read_targets(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _cancel_links(args: tuple) -> tuple[str, str | None, list]:
    rn, path = args
    try:
        data = Path(path).read_bytes()
        doc = read_zip_bytes(data, rn)
        return rn, hashlib.sha256(data).hexdigest(), extract_links(doc.text)
    except Exception:  # noqa: BLE001
        return rn, None, []


def run(limit: int | None = None, procs: int = 4) -> dict:
    t0 = time.time()
    root = data_root()
    doc_root = root / "kr/raw/dart_doc_original"
    out_dir = root / "kr/output" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = code_state()
    version = f"{CORRECTION_PARSER_VERSION}+{cs['commit'][:7]}" + (
        "+dirty" if cs["worktree_dirty"] else ""
    )

    targets = read_targets(doc_root / "c/targets.tsv")
    n_targets = len(targets)
    jobs, n_nozip = [], 0
    for m in targets[:limit] if limit else targets:
        zp = doc_root / "c/raw" / f"{m['rcept_no']}.zip"
        if not zp.exists():
            n_nozip += 1  # 원문 없음(014 등)은 행을 만들지 않는다
            continue
        jobs.append((m, str(zp), version))
    with Pool(procs) as p:
        rows = p.map(_parse_one, jobs, chunksize=200)
    t_parse = time.time() - t0

    rc_df = pd.read_parquet(
        receipt_path(root), columns=["rcept_no", "corp_code", "report_nm", "rcept_dt", "corp_cls"]
    ).drop_duplicates("rcept_no")
    rc = Receipts(rc_df)

    # 연결
    link_rows: list[dict] = []
    for r in rows:
        if r["parse_status"] == "fail" and "rel_links" not in r:
            r.update(rel_links=[], corr_kind=None, ref_submit_date=None)
        lk = link_correction(r, rc)
        link_rows += lk.pop("link_rows")
        r.update(lk)
    c_target = {
        r["rcept_no"]: r["target_rcept_no"]
        for r in rows
        if r.get("corr_kind") == "value_correction"
    }
    c_chain: dict[str, tuple] = {}
    for r in rows:
        if r.get("corr_kind") != "value_correction":
            r.update(
                chain_status=None, chain_len=None, first_rcept_no=None, chain_last_rcept_no=None
            )
            continue
        st, first, last, hops = follow_chain(r["target_rcept_no"], r["rcept_no"], c_target, rc)
        r.update(chain_status=st, chain_len=hops, first_rcept_no=first, chain_last_rcept_no=last)
        c_chain[r["rcept_no"]] = (st, first, last)

    # 가용 시점 (§6): 접수일 다음 거래일
    cal = NextTradingDay(
        load_trade_dates(
            root / f"kr/raw/raw_postgres/snapshot_date={receipt_snapshot()}/source=sj2_remote"
        )
    )
    for r in rows:
        nd, assumed = cal.next_after(_ymd(r["rcept_dt"]))
        r["available_from"] = nd.isoformat()
        r["available_from_assumed"] = assumed
        cd = r.get("corr_date")
        r["corr_date_ne_rcept"] = None if not cd else cd.replace("-", "") != r["rcept_dt"]

    item_rows = []
    for r in rows:
        for i, (lab, bf, af) in enumerate(r.get("items") or []):
            cls, multi = classify_item(lab)
            item_rows.append(
                dict(
                    rcept_no=r["rcept_no"],
                    item_seq=i,
                    item_label_raw=lab,
                    before_raw=bf,
                    after_raw=af,
                    item_class=cls,
                    item_multi_field=multi,
                    before_num=to_num(bf),
                    after_num=to_num(af),
                )
            )

    # §5 이벤트별 superseded_by
    ev = pd.read_parquet(
        a_dir(root) / "kr_contract_event.parquet", columns=["rcept_no", "corp_code", "rcept_dt"]
    )
    linked = [
        (r["first_rcept_no"], r["rcept_dt"], r["rcept_no"]) for r in rows if r.get("first_rcept_no")
    ]
    last_by_first: dict[str, tuple[str, str]] = {}
    n_by_first: dict[str, int] = defaultdict(int)
    for f, d, rn in sorted(linked):
        n_by_first[f] += 1
        last_by_first[f] = (d, rn)
    unlinked_by_corp: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.get("corr_kind") == "value_correction" and not r.get("first_rcept_no"):
            unlinked_by_corp[r["corp_code"]].append(r["rcept_dt"])
    sup = []
    for e_no, e_corp, e_dt in zip(ev.rcept_no, ev.corp_code, ev.rcept_dt, strict=True):
        lb = last_by_first.get(e_no)
        sup.append(
            dict(
                rcept_no=e_no,
                corp_code=e_corp,
                superseded_by=lb[1] if lb else None,
                superseded_rcept_dt=lb[0] if lb else None,
                n_corrections_linked=n_by_first.get(e_no, 0),
                n_corrections_unlinked=sum(
                    1 for d in unlinked_by_corp.get(e_corp, []) if d >= e_dt
                ),
            )
        )

    # 해지 v2
    cn = pd.read_parquet(
        a_dir(root) / "kr_contract_cancel.parquet",
        columns=[
            "rcept_no",
            "rcept_dt",
            "corp_code",
            "first_rcept_no",
            "link_method",
            "first_pair_date",
            "parse_status",
        ],
    )
    with Pool(procs) as p:
        cl = p.map(
            _cancel_links,
            [(rn, str(doc_root / "ga/raw" / f"{rn}.zip")) for rn in cn.rcept_no],
            chunksize=50,
        )
    cl_map = {rn: (h, ls) for rn, h, ls in cl}
    v2 = []
    for rec in cn.to_dict("records"):
        h, ls = cl_map[rec["rcept_no"]]
        res = link_cancel_v2(rec, ls, rc, c_chain)
        link_rows += res.pop("link_rows", [])
        # 계획 §2.5의 76건: 기존이 못 이은 해지 중 가장 이른 쌍 날짜에 같은 회사 기재정정 체결 공시가 있는 건  # noqa: E501
        in76 = False
        if rec["link_method"] == "unlinked_first_not_collected" and rec["first_pair_date"]:
            d8 = str(rec["first_pair_date"]).replace("-", "")
            in76 = any(
                rc.family(v) == "correction"
                for v in rc.fam_by_corp_date.get((rec["corp_code"], d8), [])
            )
        v2.append(
            {
                **rec,
                **res,
                "src_zip_sha256": h,
                "in_plan_case76": in76,
                "first_rcept_no_old": rec["first_rcept_no"],
                "link_method_old": rec["link_method"],
            }
        )
    cancel_v2 = pd.DataFrame(v2).drop(columns=["first_rcept_no", "link_method"])

    # 저장
    corr_cols = COMMON + [
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
        "corr_kind",
        "corr_date",
        "corr_date_raw",
        "corr_date_ne_rcept",
        "ref_doc_name_raw",
        "ref_submit_raw",
        "ref_submit_date",
        "ref_submit_n_dates",
        "ref_latest_corr_date",
        "reason_raw",
        "n_items",
        "note_raw",
        "n_rel_links",
        "rel_links_raw",
        "rel_text_raw",
        "link_stage1",
        "link_method",
        "target_rcept_no",
        "n_link_candidates",
        "n_date_candidates",
        "chain_status",
        "chain_len",
        "first_rcept_no",
        "chain_last_rcept_no",
        "available_from",
        "available_from_assumed",
    ]
    corr = pd.DataFrame([{c: r.get(c) for c in corr_cols} for r in rows], columns=corr_cols)
    tables = {
        "kr_contract_correction": corr,
        "kr_contract_correction_item": pd.DataFrame(item_rows),
        "kr_contract_correction_link": pd.DataFrame(link_rows),
        "kr_contract_event_supersede": pd.DataFrame(sup),
        "kr_contract_cancel_v2": cancel_v2,
    }
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    inputs = {
        "c_targets": {
            "path": str(doc_root / "c/targets.tsv"),
            "sha256": sha256_path(doc_root / "c/targets.tsv"),
            "n_rows": n_targets,
            "n_no_zip": n_nozip,
        },
        "c_zip_sha256_list": {
            "path": str(doc_root / "c_mac_sha256.txt"),
            "sha256": sha256_path(doc_root / "c_mac_sha256.txt"),
        },
        "receipt_list": {
            "path": str(receipt_path(root)),
            "sha256": sha256_path(receipt_path(root)),
            "snapshot": receipt_snapshot(),
            "n_rows": len(rc_df),
        },
        "a_event": {
            "path": str(a_dir(root) / "kr_contract_event.parquet"),
            "sha256": sha256_path(a_dir(root) / "kr_contract_event.parquet"),
        },
        "a_cancel": {
            "path": str(a_dir(root) / "kr_contract_cancel.parquet"),
            "sha256": sha256_path(a_dir(root) / "kr_contract_cancel.parquet"),
        },
        "ga_zip_dir": str(doc_root / "ga/raw"),
    }
    mf = {
        "output": OUT_NAME,
        "created_at": _now(),
        "code": cs,
        "parser_version": version,
        "inputs": inputs,
        "rows": {k: len(v) for k, v in tables.items()},
        "limit": limit,
        "seconds_parse": round(t_parse, 1),
        "seconds_total": round(time.time() - t0, 1),
        "python": sys.version.split()[0],
        "rules": {
            "link_kind_digits": KIND_DIGITS,
            "family": "체결 계열 = 제목에 체결·공급계약, 해지 아님, 원공시 또는 [기재정정]. [첨부…]는 후보에서 뺀다",  # noqa: E501
            "link_date_match": "링크 번호 앞 8자리 또는 글자의 날짜가 ref_submit_date와 같은 링크만",  # noqa: E501
            "date_stage_fallback": "링크가 없음·날짜 불일치·변환 결과 없음일 때만. link_ambiguous·link_non_contract는 날짜 보조로 안 넘어감. 제출일=접수일이면 date_self(미연결)",  # noqa: E501
            "max_chain": MAX_CHAIN,
            "first_collected": FIRST_COLLECTED,
            "excluded": "20200304900188(대상 밖)은 targets.tsv에 없어 행이 없다",
        },
        "availability_note": "가용 시점 = 정정 접수일 다음 거래일. corr_date가 접수일과 다른 건은 corr_date로 소급하지 않는다(보수적 가정, 계획 13 §6).",  # noqa: E501
        "note": "텍스트 모델·LLM 추출 없음. derived/에는 쓰지 않았다(§7 통과 전). 기존 snapshot_date=2026-10-28은 읽기만.",  # noqa: E501
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
