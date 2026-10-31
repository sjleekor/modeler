"""정정 공시(c) 전수 자동 점검(계획 13 §7.3)과 사람 대조 표본 목록(§7.1).

합격 판정은 하지 않는다. 숫자와 후보 목록만 낸다. 대조는 메인 세션이 원문 표와 파서 값을 맞댄다.
표본은 대조 전에 시드와 함께 ``review/``에 저장하고, 표본 문서를 규칙 고치는 데 쓰지 않는다.
개발 중 규칙을 고치며 본 문서(``review/dev_seen.txt``)와 계획 §2.4의 35건은 표본에서 뺀다.

    python -m modeler.etl.disclosures.correction_qa [--dev-seen FILE] [--prior-excluded FILE]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from modeler.etl.disclosures.correction_build import (
    MAX_CHAIN,
    OUT_NAME,
    a_dir,
    data_root,
)

SEED = 2026110101
PER_STRATUM = 25  # 정정 표 9층 x 25
N_ITEMS_DOCS = 40
N_LINK_PER_STRATUM = 20  # (a) 링크 연결, 9층 x 20
N_DATE_LINK = 80  # (b)
N_UNLINKED = 20  # (c)
N_CHAIN = 40  # (e)
N_DATE_NE = 25  # 정정일자 != 접수일
REF_TABLE_RATE = 15061 / 15141  # 계획 §2.2·§7.3의 99.47%


def _bin(y: int) -> str:
    return "2020-2021" if y <= 2021 else "2022-2023" if y <= 2023 else "2024-2026"


def _cls(c: str | None) -> str:
    return c if c in ("Y", "K") else "EN"


def _rng(name: str) -> random.Random:
    # 11-01 수정 표본은 시드를 올린다: b2는 2026110102, e2는 2026110103
    seed = {"date_unique_v2": SEED + 1, "chain2plus_v2": SEED + 2}.get(name, SEED)
    return random.Random(f"{seed}:{name}")


def _pick(ids: list[str], n: int, name: str) -> list[str]:
    ids = sorted(ids)
    _rng(name).shuffle(ids)
    return ids[:n]


def _read_ids(p: Path | None) -> set[str]:
    if not p or not p.exists():
        return set()
    return {x.split()[0] for x in p.read_text().splitlines() if x.strip()}


def run(
    dev_seen: Path | None = None,
    prior: Path | None = None,
    recheck: Path | None = None,
    write_review: bool = True,
) -> dict:
    root = data_root()
    out = root / "kr/output" / OUT_NAME
    rv = out / "review"
    rv.mkdir(exist_ok=True)
    qd = out / "qa"
    qd.mkdir(exist_ok=True)
    c = pd.read_parquet(out / "kr_contract_correction.parquet")
    it = pd.read_parquet(out / "kr_contract_correction_item.parquet")
    lk = pd.read_parquet(out / "kr_contract_correction_link.parquet")
    sp = pd.read_parquet(out / "kr_contract_event_supersede.parquet")
    cv = pd.read_parquet(out / "kr_contract_cancel_v2.parquet")
    ev = pd.read_parquet(
        a_dir(root) / "kr_contract_event.parquet",
        columns=["rcept_no", "corp_code", "contract_name", "amount", "period_end", "period_start"],
    )
    mf = json.loads((out / "manifest.json").read_text())
    summ: list[tuple[str, str, object]] = []

    def S(sec: str, name: str, val: object) -> None:
        summ.append((sec, name, val))

    # 1. 건수·실패율
    n_in = mf["inputs"]["c_targets"]["n_rows"] - mf["inputs"]["c_targets"]["n_no_zip"]
    S("count", "targets_rows", mf["inputs"]["c_targets"]["n_rows"])
    S("count", "targets_without_zip(014)", mf["inputs"]["c_targets"]["n_no_zip"])
    S("count", "input_docs_with_zip", n_in)
    S("count", "output_rows", len(c))
    S("count", "in_eq_out", bool(n_in == len(c)))
    S("count", "dup_rcept_no", int(c.rcept_no.duplicated().sum()))
    vc = c.parse_status.value_counts()
    for k in ("ok", "partial", "fail"):
        S("status", k, int(vc.get(k, 0)))
    S("status", "fail_ratio", round(float((c.parse_status == "fail").mean()), 5))
    S("status", "fail_threshold(<0.5%)_ok", bool((c.parse_status == "fail").mean() < 0.005))
    for k, v in c.corr_kind.value_counts(dropna=False).items():
        S("corr_kind", str(k), int(v))
    c[c.parse_status != "ok"][
        ["rcept_no", "report_nm", "corr_kind", "parse_status", "parse_err", "parse_fail_reason"]
    ].to_csv(qd / "qa_parse_not_ok.tsv", sep="\t", index=False)
    S("count", "items_rows", len(it))
    S(
        "count",
        "docs_n_items_0(value_correction)",
        int(((c.corr_kind == "value_correction") & (c.n_items == 0)).sum()),
    )

    # 2. 연도·시장별 정정 표 있음 비율
    d = c.assign(year=c.rcept_dt.str[:4], cls=c.corp_cls.map(_cls))
    d["has_table"] = d.corr_kind == "value_correction"
    g = d.groupby(["year", "cls"]).has_table.agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    g["dev_pp"] = (g.rate - REF_TABLE_RATE) * 100
    g["flag_dev10"] = g.dev_pp.abs() >= 10
    g.to_csv(qd / "qa_table_rate_by_year_cls.tsv", sep="\t", index=False)
    S("table_rate", "overall", round(float(d.has_table.mean()), 5))
    S("table_rate", "cells_flag_dev10", int(g.flag_dev10.sum()))
    S("table_rate", "ref_rate", round(REF_TABLE_RATE, 5))

    # 3. 인코딩
    S("encoding", "replace_decode", int((c.src_encoding == "replace").sum()))
    S("encoding", "mismatch_declared_vs_actual", int(c.src_encoding_mismatch.fillna(False).sum()))
    S(
        "encoding",
        "source_replacement_chars",
        int(c.parse_err.fillna("").str.contains("source_replacement").sum()),
    )
    c[
        (c.src_encoding == "replace")
        | c.parse_err.fillna("").str.contains("source_replacement|decode_replace")
    ][["rcept_no", "src_encoding", "parse_err"]].to_csv(
        qd / "qa_encoding_replace.tsv", sep="\t", index=False
    )

    # 4. 링크 변환 (정정 문서 링크)
    lc = lk[lk.src == "correction"]
    n = len(lc)
    S("link_conv", "links_total", n)
    S("link_conv", "same_corp_exactly_1", int((lc.n_found_same_corp == 1).sum()))
    S("link_conv", "same_corp_exactly_1_rate", round(float((lc.n_found_same_corp == 1).mean()), 4))
    S("link_conv", "same_corp_2plus", int((lc.n_found_same_corp >= 2).sum()))
    S("link_conv", "same_corp_0", int((lc.n_found_same_corp == 0).sum()))
    S(
        "link_conv",
        "same_corp_0_but_other_corp_hit",
        int(((lc.n_found_same_corp == 0) & (lc.n_found_other_corp > 0)).sum()),
    )
    dm = lc[lc.date_match]
    S("link_conv", "date_matched_links", len(dm))
    S(
        "link_conv",
        "date_matched_exactly_1_rate",
        round(float((dm.n_found_same_corp == 1).mean()), 4),
    )
    mk = lc[lc.market_digit_ok.eq(False)]
    S("link_conv", "market_digit_mismatch_links(Y!=8,K!=9)", len(mk))
    S("link_conv", "market_digit_checkable_links", int(lc.market_digit_ok.notna().sum()))
    mk.to_csv(qd / "qa_link_market_digit_mismatch.tsv", sep="\t", index=False)
    nc = lc[(lc.n_found_same_corp > 0) & (lc.n_family == 0)]
    S("link_conv", "converted_to_non_contract_title_links", len(nc))
    nc.to_csv(qd / "qa_link_converted_non_contract.tsv", sep="\t", index=False)
    S(
        "link_conv",
        "digit_counts",
        json.dumps(lc.found_digit.str.split(",").explode().value_counts().to_dict()),
    )
    lcn = lk[lk.src == "cancel"]
    S("link_conv", "cancel_links_total", len(lcn))
    S(
        "link_conv",
        "cancel_links_same_corp_exactly_1_rate",
        round(float((lcn.n_found_same_corp == 1).mean()), 4),
    )
    S("link_conv", "cancel_market_digit_mismatch_links", int(lcn.market_digit_ok.eq(False).sum()))
    lcn[lcn.market_digit_ok.eq(False)].to_csv(
        qd / "qa_cancel_link_market_digit_mismatch.tsv", sep="\t", index=False
    )

    # 5. 연결 방법·사슬
    vcc = c[c.corr_kind == "value_correction"]
    for k, v in c.link_method.value_counts(dropna=False).items():
        S("link_method", str(k), int(v))
    for k, v in c.link_stage1.value_counts(dropna=False).items():
        S("link_stage1", str(k), int(v))
    for k, v in vcc.chain_status.value_counts(dropna=False).items():
        S("chain_status", str(k), int(v))
    S("chain", "value_correction_docs", len(vcc))
    S(
        "chain",
        "reached_original_rate",
        round(float((vcc.chain_status == "reached_original").mean()), 4),
    )
    cyc = vcc[(vcc.chain_status == "chain_loop") & (vcc.chain_len <= MAX_CHAIN)]
    long_ = vcc[(vcc.chain_status == "chain_loop") & (vcc.chain_len > MAX_CHAIN)]
    S("chain", "true_cycles(0 expected)", len(cyc))
    S("chain", f"too_long_gt_{MAX_CHAIN}(chain_loop로 표시)", len(long_))
    pd.concat([cyc.assign(why="cycle"), long_.assign(why="too_long")])[
        [
            "rcept_no",
            "corp_code",
            "rcept_dt",
            "chain_len",
            "why",
            "target_rcept_no",
            "chain_last_rcept_no",
        ]
    ].to_csv(qd / "qa_chain_loop.tsv", sep="\t", index=False)
    bk = vcc.assign(year=vcc.rcept_dt.str[:4])
    t = bk.groupby("year").apply(
        lambda x: pd.Series(
            dict(
                n=len(x),
                broken_pre_2020_03_05=int((x.chain_status == "chain_broken_pre_2020_03_05").sum()),
                broken_unlinked=int((x.chain_status == "chain_broken_unlinked").sum()),
                broken_not_in_c=int((x.chain_status == "chain_broken_not_in_c").sum()),
                chain_loop=int((x.chain_status == "chain_loop").sum()),
                unlinked=int((x.chain_status == "unlinked").sum()),
                reached_original=int((x.chain_status == "reached_original").sum()),
            )
        ),
        include_groups=False,
    )
    t["broken_pre_2020_rate"] = (t.broken_pre_2020_03_05 / t.n).round(4)
    t.reset_index().to_csv(qd / "qa_chain_by_year.tsv", sep="\t", index=False)
    S("chain", "broken_pre_2020_03_05", int(t.broken_pre_2020_03_05.sum()))
    S(
        "chain",
        "broken_pre_2020_03_05_rate",
        round(float(t.broken_pre_2020_03_05.sum() / len(vcc)), 4),
    )
    evset = set(ev.rcept_no)
    ro = vcc[vcc.chain_status == "reached_original"]
    inev = ro.first_rcept_no.isin(evset)
    S("chain", "first_rcept_no_in_a_event_rate", round(float(inev.mean()), 4))
    S("chain", "first_rcept_no_not_in_a_event", int((~inev).sum()))
    ro[~inev][["rcept_no", "corp_code", "first_rcept_no", "rcept_dt"]].to_csv(
        qd / "qa_first_not_in_a_event.tsv", sep="\t", index=False
    )
    # 직전 값과 before_raw 일치(보고만): 원공시가 바로 target인 정정, amount·period_end 숫자·날짜 칸
    one = ro[ro.chain_len == 1][["rcept_no", "first_rcept_no"]].merge(
        ev, left_on="first_rcept_no", right_on="rcept_no", suffixes=("", "_e")
    )
    ia = it[(it.item_class == "amount") & ~it.item_multi_field & it.before_num.notna()].merge(
        one, on="rcept_no"
    )
    ia = ia[ia.amount.notna()]
    S("before_match", "amount_rows_checkable", len(ia))
    S(
        "before_match",
        "amount_before_eq_event_amount_rate",
        round(float((ia.before_num == ia.amount).mean()), 4) if len(ia) else None,
    )
    ip = it[(it.item_class == "period_end") & ~it.item_multi_field].merge(one, on="rcept_no")
    ip = ip[ip.period_end.notna()]
    ip = ip[ip.before_raw.str.strip().str.match(r"^\d{4}-\d{2}-\d{2}$")]
    S("before_match", "period_end_rows_checkable", len(ip))
    S(
        "before_match",
        "period_end_before_eq_event_rate",
        round(float((ip.before_raw.str.strip() == ip.period_end).mean()), 4) if len(ip) else None,
    )

    # 6. 정정일자 != 접수일
    ne = c[c.corr_date_ne_rcept.fillna(False)]
    S("corr_date", "ne_rcept_dt", len(ne))
    S("corr_date", "ne_rcept_dt_rate", round(len(ne) / len(vcc), 4))
    ne[["rcept_no", "rcept_dt", "corp_cls", "corr_date", "corr_date_raw"]].to_csv(
        qd / "qa_corr_date_ne_rcept.tsv", sep="\t", index=False
    )

    # 7. 항목 분류 보고
    for k, v in it.item_class.value_counts().items():
        S("item_class", k, f"{int(v)} ({v / len(it):.3f})")
    S("item_class", "multi_field_rows", int(it.item_multi_field.sum()))
    S("item_num", "before_num_filled", f"{int(it.before_num.notna().sum())}/{len(it)}")

    # 8. 이벤트 superseded_by
    S("supersede", "events", len(sp))
    S("supersede", "with_superseded_by", int(sp.superseded_by.notna().sum()))
    S(
        "supersede",
        "events_with_unlinked_corrections_same_corp",
        int((sp.n_corrections_unlinked > 0).sum()),
    )

    # 9. 해지 v2
    for k, v in cv.link_method_v2.value_counts(dropna=False).items():
        S("cancel_v2", str(k), int(v))
    both = cv[cv.first_rcept_no_old.notna() & cv.first_rcept_no_v2.notna()]
    S("cancel_v2", "old_and_v2_both_linked", len(both))
    S("cancel_v2", "old_eq_v2", int((both.first_rcept_no_old == both.first_rcept_no_v2).sum()))
    S(
        "cancel_v2",
        "old_ne_v2(목록)",
        int((both.first_rcept_no_old != both.first_rcept_no_v2).sum()),
    )
    both[both.first_rcept_no_old != both.first_rcept_no_v2].to_csv(
        qd / "qa_cancel_old_ne_v2.tsv", sep="\t", index=False
    )
    cs = cv[cv.in_plan_case76].copy()
    cs["pair_year"] = cs.first_pair_date.astype(str).str[:4].astype(int)
    S("cancel_v2", "plan_case76", len(cs))
    S("cancel_v2", "plan_case76_pair_year_2020plus", int((cs.pair_year >= 2020).sum()))
    S(
        "cancel_v2",
        "plan_case76_cancel_year_2020plus",
        int((cs.rcept_dt.astype(str).str[:4].astype(int) >= 2020).sum()),
    )
    c40 = cs[cs.pair_year >= 2020]
    for k, v in c40.link_method_v2.value_counts(dropna=False).items():
        S("cancel_v2_40", str(k), int(v))
    S("cancel_v2_40", "first_rcept_no_v2_filled", int(c40.first_rcept_no_v2.notna().sum()))
    cx = c40.merge(
        ev.rename(
            columns={
                "rcept_no": "first_rcept_no_v2",
                "corp_code": "v2_corp_code",
                "amount": "v2_event_amount",
                "contract_name": "v2_event_contract_name",
                "period_end": "v2_event_period_end",
                "period_start": "v2_event_period_start",
            }
        ),
        on="first_rcept_no_v2",
        how="left",
    )
    ca_full = pd.read_parquet(
        a_dir(root) / "kr_contract_cancel.parquet",
        columns=["rcept_no", "contract_name", "amount", "cancel_amount", "period_end", "report_nm"],
    ).rename(
        columns={
            "contract_name": "cancel_contract_name",
            "amount": "cancel_doc_amount",
            "period_end": "cancel_period_end",
            "report_nm": "cancel_report_nm",
        }
    )
    cx = cx.merge(ca_full, on="rcept_no", how="left")
    cx["auto_same_corp"] = cx.v2_corp_code.eq(cx.corp_code)
    cx["auto_amount_eq(참고)"] = cx.v2_event_amount.eq(
        cx.cancel_doc_amount
    ) | cx.v2_event_amount.eq(cx.cancel_amount)
    cols = [
        "rcept_no",
        "rcept_dt",
        "corp_code",
        "cancel_report_nm",
        "first_pair_date",
        "link_method_old",
        "first_rcept_no_old",
        "link_method_v2",
        "first_rcept_no_v2",
        "v2_resolved",
        "v2_chain_last_rcept_no",
        "cancel_contract_name",
        "cancel_doc_amount",
        "cancel_amount",
        "cancel_period_end",
        "v2_event_contract_name",
        "v2_event_amount",
        "v2_event_period_end",
        "auto_same_corp",
        "auto_amount_eq(참고)",
    ]
    if write_review:
        cx[cols].to_csv(rv / "sample_d_cancel40_full.tsv", sep="\t", index=False)
    S(
        "cancel_v2_40",
        "v2_filled_auto_same_corp",
        int(cx[cx.first_rcept_no_v2.notna()].auto_same_corp.sum()),
    )
    S(
        "cancel_v2_40",
        "v2_filled_auto_amount_eq(참고)",
        int(cx[cx.first_rcept_no_v2.notna()]["auto_amount_eq(참고)"].sum()),
    )

    pd.DataFrame(summ, columns=["section", "name", "value"]).to_csv(
        qd / "qa_summary.tsv", sep="\t", index=False
    )

    if not write_review:
        return {"summary_rows": len(summ)}
    meta = make_samples(c, it, lk, rv, dev_seen, prior, ne, recheck)
    return {"summary_rows": len(summ), **meta}


def make_samples(c, it, lk, rv: Path, dev_seen, prior, ne, recheck=None) -> dict:
    dev = _read_ids(dev_seen)
    pri = _read_ids(prior)
    rck = _read_ids(recheck)
    excl = dev | pri | rck
    vc_all = c[c.corr_kind == "value_correction"]
    n_vc_all = len(vc_all)
    vc = c[(c.corr_kind == "value_correction") & ~c.rcept_no.isin(excl)].copy()
    vc["stratum"] = vc.rcept_dt.str[:4].astype(int).map(_bin) + "|" + vc.corp_cls.map(_cls)
    meta: dict = {
        "seed": SEED,
        "seed_rule": "random.Random(f'{seed}:{표본이름}'), 모집단은 rcept_no 정렬 뒤 셔플",
        "n_dev_seen_excluded": len(dev),
        "n_prior_plan_2_4_excluded": len(pri),
        "n_main_recheck_400_excluded": len(rck),
        "exclusion_sets": {
            "dev_seen.txt": len(dev),
            "prior_plan_2_4_35.txt": len(pri),
            "main_recheck_400_seed20261031.txt": len(rck),
            "union": len(excl),
            "union_in_value_correction_population": int(vc_all.rcept_no.isin(excl).sum()),
        },
        "population_value_correction_before_exclusion": n_vc_all,
        "excluded_note": "dev_seen.txt(개발 중 본 문서) + 계획 §2.4 35건(시드 2026103101 재현) "
        "+ 메인 §12 재측정 400건(시드 20261031).",
        "samples": {},
    }

    def attach(ids: list[str]) -> pd.DataFrame:
        d = c.set_index("rcept_no").loc[ids].reset_index()
        return d

    # --- 정정 표 필드 표본: 9층 x 25
    parts = []
    for st, g in sorted(vc.groupby("stratum")):
        pick = _pick(list(g.rcept_no), PER_STRATUM, f"fields:{st}")
        d = attach(pick)
        d["stratum"] = st
        d["seq"] = range(1, len(d) + 1)
        parts.append(d)
        meta["samples"].setdefault("fields", {})[st] = {"population": len(g), "listed": len(pick)}
    f = pd.concat(parts, ignore_index=True)
    fcols = [
        "stratum",
        "seq",
        "rcept_no",
        "rcept_dt",
        "corp_code",
        "corp_cls",
        "report_nm",
        "src_zip_sha256",
        "corr_date_raw",
        "corr_date",
        "ref_doc_name_raw",
        "ref_submit_raw",
        "ref_submit_date",
        "ref_latest_corr_date",
        "reason_raw",
        "n_items",
        "parse_status",
        "parse_err",
    ]
    f[fcols].to_csv(rv / "sample_fields_corr_table.tsv", sep="\t", index=False)
    meta["samples"]["fields"]["_n"] = len(f)

    # --- 항목 표본: 위 표본에서 n_items>=2인 문서 40건
    multi = list(f[f.n_items >= 2].rcept_no)
    pick = _pick(multi, N_ITEMS_DOCS, "items")
    n_items_docs = len(pick)
    itp = it[it.rcept_no.isin(pick)].merge(
        c[["rcept_no", "rcept_dt", "corp_code", "report_nm"]], on="rcept_no"
    )
    itp.to_csv(rv / "sample_items_rows.tsv", sep="\t", index=False)
    pd.DataFrame({"rcept_no": sorted(pick)}).to_csv(
        rv / "sample_items_docs.tsv", sep="\t", index=False
    )
    meta["samples"]["items"] = {
        "docs": len(pick),
        "rows": len(itp),
        "from": "fields 표본 중 n_items>=2",
    }

    # --- 연결 표본
    def link_cols(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        info = c.set_index("rcept_no")
        d["target_dt"] = d.target_rcept_no.map(
            lambda x: info.rcept_dt.get(x) if isinstance(x, str) and x in info.index else None
        )
        return d

    lcols = [
        "rcept_no",
        "rcept_dt",
        "corp_code",
        "corp_cls",
        "report_nm",
        "src_zip_sha256",
        "ref_doc_name_raw",
        "ref_submit_raw",
        "ref_submit_date",
        "rel_links_raw",
        "link_stage1",
        "link_method",
        "target_rcept_no",
        "n_link_candidates",
        "n_date_candidates",
        "chain_status",
        "chain_len",
        "first_rcept_no",
        "chain_last_rcept_no",
    ]
    # (a) 링크 연결 9층 x 20
    pa = []
    ga = vc[vc.link_method == "link_kind"]
    for st, g in sorted(ga.groupby("stratum")):
        pick = _pick(list(g.rcept_no), N_LINK_PER_STRATUM, f"link_kind:{st}")
        d = attach(pick)
        d["stratum"] = st
        pa.append(d)
        meta["samples"].setdefault("link_kind", {})[st] = {
            "population": len(g),
            "listed": len(pick),
        }
    a = pd.concat(pa, ignore_index=True)
    a[["stratum", *lcols]].to_csv(rv / "sample_a_link_kind.tsv", sep="\t", index=False)
    # (b) 날짜 보조 80
    gb = vc[vc.link_method == "date_unique"]
    pick = _pick(list(gb.rcept_no), N_DATE_LINK, "date_unique")
    attach(pick)[lcols].to_csv(rv / "sample_b_date_unique.tsv", sep="\t", index=False)
    meta["samples"]["date_unique"] = {"population": len(gb), "listed": len(pick)}
    # (c) 연결 못 한 정정 20건: 이유별 균등 배분
    reasons = [
        "no_candidate",
        "date_multi",
        "no_submit_date",
        "link_non_contract",
        "link_ambiguous",
    ]
    per = N_UNLINKED // len(reasons)
    pc = []
    for rs in reasons:
        g = vc[vc.link_method == rs]
        pick = _pick(list(g.rcept_no), per, f"unlinked:{rs}")
        pc.append(attach(pick))
        meta["samples"].setdefault("unlinked", {})[rs] = {"population": len(g), "listed": len(pick)}
    pd.concat(pc, ignore_index=True)[lcols].to_csv(
        rv / "sample_c_unlinked.tsv", sep="\t", index=False
    )
    # (e) 사슬 두 단계 이상(원공시 도달) 40건
    ge = vc[(vc.chain_status == "reached_original") & (vc.chain_len >= 2)]
    pick = _pick(list(ge.rcept_no), N_CHAIN, "chain2plus")
    ee = attach(pick)
    path = {}
    cm = c.set_index("rcept_no").target_rcept_no.to_dict()
    for rn in pick:
        p, cur = [rn], cm.get(rn)
        while cur and len(p) < 12:
            p.append(cur)
            cur = cm.get(cur)
        path[rn] = ">".join(p)
    ee["chain_path"] = ee.rcept_no.map(path)
    ee[[*lcols, "chain_path"]].to_csv(rv / "sample_e_chain2plus.tsv", sep="\t", index=False)
    meta["samples"]["chain2plus"] = {"population": len(ge), "listed": len(pick)}
    # 정정일자 != 접수일
    gn = ne[~ne.rcept_no.isin(excl)]
    pick = _pick(list(gn.rcept_no), N_DATE_NE, "corr_date_ne")
    attach(pick)[
        [
            "rcept_no",
            "rcept_dt",
            "corp_cls",
            "report_nm",
            "src_zip_sha256",
            "corr_date_raw",
            "corr_date",
        ]
    ].to_csv(rv / "sample_corr_date_ne_rcept.tsv", sep="\t", index=False)
    meta["samples"]["corr_date_ne"] = {"population": len(gn), "listed": len(pick)}
    meta["samples"]["cancel40"] = "sample_d_cancel40_full.tsv (40건 전수, 결정 3번 승인)"
    ex = vc_all[vc_all.rcept_no.isin(excl)]
    meta["excluded_removed_from_population"] = {
        "fields_all_strata": len(ex),
        "link_kind": int((ex.link_method == "link_kind").sum()),
        "date_unique": int((ex.link_method == "date_unique").sum()),
        "unlinked_by_reason": {
            k: int((ex.link_method == k).sum())
            for k in [
                "no_candidate",
                "date_multi",
                "no_submit_date",
                "link_non_contract",
                "link_ambiguous",
            ]
        },
        "chain2plus": int(((ex.chain_status == "reached_original") & (ex.chain_len >= 2)).sum()),
        "corr_date_ne": int(ex.corr_date_ne_rcept.fillna(False).astype(bool).sum()),
    }
    meta["method_note"] = "파서 값만 담았다. 원문과의 대조·합격 판정은 하지 않았다."
    (rv / "sample_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"n_fields": len(f), "n_items_docs": n_items_docs}


def make_v2_samples() -> dict:
    """2026-11-01 규칙 수정(date_self) 뒤 새 표본: b2(날짜 보조 80), e2(사슬 2단계 이상 40).

    기존 표본 파일은 건드리지 않고, 제외는 기존 제외 목록 셋 + 기존 sample_*.tsv의 rcept_no다.
    """
    out = data_root() / "kr/output" / OUT_NAME
    rv = out / "review"
    c = pd.read_parquet(out / "kr_contract_correction.parquet")
    excl: dict[str, set[str]] = {
        n: _read_ids(rv / f"{n}.txt")
        for n in ("dev_seen", "prior_plan_2_4_35", "main_recheck_400_seed20261031")
    }
    old_files = sorted(
        p
        for p in rv.glob("sample_*.tsv")
        if p.name not in ("sample_b2_date_unique.tsv", "sample_e2_chain2plus.tsv")
    )
    old: set[str] = set()
    for p in old_files:
        old |= set(pd.read_csv(p, sep="\t", dtype=str, usecols=["rcept_no"]).rcept_no)
    allx = set().union(*excl.values()) | old
    vc = c[(c.corr_kind == "value_correction") & ~c.rcept_no.isin(allx)]
    lcols = [
        "rcept_no",
        "rcept_dt",
        "corp_code",
        "corp_cls",
        "report_nm",
        "src_zip_sha256",
        "ref_doc_name_raw",
        "ref_submit_raw",
        "ref_submit_date",
        "rel_links_raw",
        "link_stage1",
        "link_method",
        "target_rcept_no",
        "n_link_candidates",
        "n_date_candidates",
        "chain_status",
        "chain_len",
        "first_rcept_no",
        "chain_last_rcept_no",
    ]
    meta: dict = {
        "rule_change": "2026-11-01 date_self: 제출일=접수일이면 날짜 보조로 안 잇는다",
        "excluded": {k: len(v) for k, v in excl.items()},
        "excluded_old_sample_files": [p.name for p in old_files],
        "excluded_old_sample_ids": len(old),
        "excluded_union": len(allx),
    }
    gb = vc[vc.link_method == "date_unique"]
    pick = _pick(list(gb.rcept_no), N_DATE_LINK, "date_unique_v2")  # 시드는 아래에서 바꾼다
    meta["b2"] = {"seed": SEED + 1, "population_after_exclusion": len(gb), "listed": len(pick)}
    c.set_index("rcept_no").loc[pick].reset_index()[lcols].to_csv(
        rv / "sample_b2_date_unique.tsv", sep="\t", index=False
    )
    ge = vc[(vc.chain_status == "reached_original") & (vc.chain_len >= 2)]
    pick = _pick(list(ge.rcept_no), N_CHAIN, "chain2plus_v2")
    ee = c.set_index("rcept_no").loc[pick].reset_index()
    cm = c.set_index("rcept_no").target_rcept_no.to_dict()
    path = {}
    for rn in pick:
        p, cur = [rn], cm.get(rn)
        while cur and len(p) < 60:
            p.append(cur)
            cur = cm.get(cur)
        path[rn] = ">".join(p)
    ee["chain_path"] = ee.rcept_no.map(path)
    ee[[*lcols, "chain_path"]].to_csv(rv / "sample_e2_chain2plus.tsv", sep="\t", index=False)
    meta["e2"] = {"seed": SEED + 2, "population_after_exclusion": len(ge), "listed": len(pick)}
    (rv / "sample_meta_v2.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-seen", type=Path, default=None)
    ap.add_argument("--prior-excluded", type=Path, default=None)
    ap.add_argument("--v2-samples", action="store_true", help="b2·e2만 새로 뽑고 qa 요약만 갱신")
    a = ap.parse_args()
    if a.v2_samples:
        run(write_review=False)
        print(json.dumps(make_v2_samples(), ensure_ascii=False, indent=2))
        return
    out = data_root() / "kr/output" / OUT_NAME / "review"
    out.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            run(
                a.dev_seen or out / "dev_seen.txt",
                a.prior_excluded or out / "prior_plan_2_4_35.txt",
                out / "main_recheck_400_seed20261031.txt",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
