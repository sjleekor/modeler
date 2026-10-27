"""전수 자동 점검(계획 §7.3)과 대조용 표본 목록(§7.1) 만들기.

합격 판정은 하지 않는다. 숫자와 후보 목록만 낸다. 대조는 원문 표와 파서 값을 사람이 맞댄다.

    python -m modeler.etl.disclosures.qa
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from modeler.etl.disclosures.build import OUT_NAME, data_root

SEED = 20261028
PER_STRATUM_CORE = 15  # 12층 x 15 = 180건 (계획 §7.1 "합 180건 안팎")
PER_STRATUM_POOL = 40  # 층마다 무작위 순서로 40건까지 목록을 둔다(core 뒤는 추가 대조용)

# 06 표본(a 200건)의 채움 건수 / 200. 연도별 채움률이 10%p 넘게 벗어나면 서식 변경 후보다.
REF_FILL = {
    "contract_name": 1.0,
    "amount": 197 / 200,
    "recent_sales": 1.0,
    "ratio_pct": 197 / 200,
    "counterparty_raw": 192 / 200,
    "period_start": 182 / 200,
    "period_end": 180 / 200,
    "contract_date": 1.0,
    "region": 192 / 200,
}
FILL_EVENT = [
    "contract_name",
    "contract_kind",
    "amount",
    "recent_sales",
    "ratio_pct",
    "counterparty_raw",
    "region",
    "period_start",
    "period_end",
    "contract_date",
    "terms",
    "other_text",
    "related_raw",
]
FILL_CANCEL = [
    "contract_name",
    "cancel_amount",
    "recent_sales",
    "ratio_pct",
    "counterparty_raw",
    "cancel_date",
    "cancel_reason",
    "period_start",
    "period_end",
    "related_raw",
]
SAMPLE_COLS_EVENT = [
    "contract_name",
    "amount_raw",
    "amount",
    "amount_label",
    "recent_sales_raw",
    "recent_sales",
    "ratio_pct_raw",
    "ratio_pct",
    "counterparty_raw",
    "region",
    "period_start_raw",
    "period_start",
    "period_end_raw",
    "period_end",
    "contract_date_raw",
    "contract_date",
    "ratio_check_ok",
]
SAMPLE_COLS_CANCEL = [
    "contract_name",
    "cancel_amount_raw",
    "cancel_amount",
    "recent_sales_raw",
    "recent_sales",
    "ratio_pct_raw",
    "ratio_pct",
    "counterparty_raw",
    "period_start",
    "period_end",
    "cancel_date_raw",
    "cancel_date",
    "cancel_reason",
    "first_rcept_no",
    "link_method",
    "ratio_check_ok",
]


def _filled(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.notna() & ~s.astype(str).str.strip().isin(["", "-", "None"])
    return s.notna()


def _bin(y: int) -> str:
    return "2014-2017" if y <= 2017 else "2018-2021" if y <= 2021 else "2022-2026"


def run(sample_exclude: Path | None = None) -> dict:
    root = data_root()
    out = root / "kr/output" / OUT_NAME
    ev = pd.read_parquet(out / "kr_contract_event.parquet")
    ca = pd.read_parquet(out / "kr_contract_cancel.parquet")
    no = pd.read_parquet(out / "kr_contract_notice_free.parquet")
    cp = pd.read_parquet(out / "kr_contract_counterparty_link.parquet")
    rel = pd.read_parquet(out / "kr_contract_related.parquet")
    mf = json.loads((out / "manifest.json").read_text())
    summ: list[tuple[str, str, object]] = []

    def S(sec, name, val):
        summ.append((sec, name, val))

    # 1. 입력 건수 = 출력 행 수
    n_a, n_ga = mf["n_targets"]["a"], mf["n_targets"]["ga"]
    S("count", "a_targets", n_a)
    S("count", "a_event_rows", len(ev))
    S("count", "ga_targets", n_ga)
    S("count", "ga_rows", len(ca) + len(no))
    S("count", "ga_cancel_rows", len(ca))
    S("count", "ga_notice_rows", len(no))
    S("count", "in_eq_out", bool(n_a == len(ev) and n_ga == len(ca) + len(no)))
    allr = pd.concat([ev[["rcept_no"]], ca[["rcept_no"]], no[["rcept_no"]]])
    S("count", "dup_rcept_no_all", int(allr.rcept_no.duplicated().sum()))
    for nm, d in (("event", ev), ("cancel", ca), ("notice", no)):
        vc = d.parse_status.value_counts()
        for k in ("ok", "partial", "fail"):
            S("status", f"{nm}_{k}", int(vc.get(k, 0)))
        S("status", f"{nm}_fail_ratio", round(float((d.parse_status == "fail").mean()), 5))
    allx = pd.concat([ev, ca, no])
    S("status", "fail_ratio_all", round(float((allx.parse_status == "fail").mean()), 5))

    # 2. 인코딩·변환 오류
    enc = (
        allx.groupby(["src_encoding", "src_encoding_mismatch"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    enc.to_csv(out / "qa_encoding.tsv", sep="\t", index=False)
    S("encoding", "replace_decode", int((allx.src_encoding == "replace").sum()))
    errs = allx[allx.parse_err.notna()][["rcept_no", "report_nm", "parse_status", "parse_err"]]
    errs.to_csv(out / "qa_parse_err.tsv", sep="\t", index=False)
    S("encoding", "rows_with_parse_err", len(errs))
    S(
        "encoding",
        "number_or_date_ERR_rows",
        int(
            errs.parse_err.str.contains(
                "amount|ratio|recent|period|date|cp_recent|hold", na=False
            ).sum()
        ),
    )
    fails = allx[allx.parse_status == "fail"][
        ["rcept_no", "report_nm", "parse_fail_reason", "parse_err"]
    ]
    fails.to_csv(out / "qa_fail_list.tsv", sep="\t", index=False)

    # 3. 연도별 채움률
    rows = []
    for nm, d, fields in (("event", ev, FILL_EVENT), ("cancel", ca, FILL_CANCEL)):
        dd = d[d.parse_status != "fail"].copy()
        dd["year"] = dd.rcept_dt.str[:4]
        for fld in fields:
            f = _filled(dd[fld])
            by = f.groupby(dd["year"]).agg(["sum", "count"])
            for y, (k, n) in by.iterrows():
                ref = REF_FILL.get(fld) if nm == "event" else None
                dev = (k / n - ref) * 100 if ref is not None else None
                rows.append(
                    dict(
                        table=nm,
                        field=fld,
                        year=y,
                        n=int(n),
                        n_filled=int(k),
                        rate=round(k / n, 4),
                        ref_rate_06=None if ref is None else round(ref, 4),
                        dev_pp=None if dev is None else round(dev, 1),
                        flag_dev10=None if dev is None else bool(abs(dev) >= 10),
                    )
                )
            S("fill_all", f"{nm}.{fld}", f"{int(f.sum())}/{len(dd)} ({f.mean():.3f})")
    pd.DataFrame(rows).to_csv(out / "qa_fill_by_year.tsv", sep="\t", index=False)
    S("fill", "n_cells_dev10", int(sum(1 for r in rows if r["flag_dev10"])))

    # 4. 금액 / 최근매출 대 비율
    for nm, d in (("event", ev), ("cancel", ca)):
        dd = d[d.parse_status != "fail"]
        chk = dd.ratio_check_ok.dropna()
        S("ratio", f"{nm}_checkable", len(chk))
        S("ratio", f"{nm}_ok", int(chk.sum()))
        S("ratio", f"{nm}_ok_rate", round(float(chk.mean()), 4) if len(chk) else None)
    bad = pd.concat(
        [
            ev[ev.ratio_check_ok.eq(False)][
                [
                    "rcept_no",
                    "rcept_dt",
                    "report_nm",
                    "form_variant",
                    "amount_raw",
                    "recent_sales_raw",
                    "ratio_pct_raw",
                    "ratio_check_diff",
                ]
            ].assign(table="event"),
            ca[ca.ratio_check_ok.eq(False)][
                [
                    "rcept_no",
                    "rcept_dt",
                    "report_nm",
                    "form_variant",
                    "cancel_amount_raw",
                    "recent_sales_raw",
                    "ratio_pct_raw",
                    "ratio_check_diff",
                ]
            ]
            .rename(columns={"cancel_amount_raw": "amount_raw"})
            .assign(table="cancel"),
        ]
    )
    bad.to_csv(out / "qa_ratio_mismatch.tsv", sep="\t", index=False)

    # 5. 단위·통화·범위
    unit = []
    for nm, d, col in (("event", ev, "amount"), ("cancel", ca, "cancel_amount")):
        dd = d[d.parse_status != "fail"]
        u = dd[f"{col}_unit"] if f"{col}_unit" in dd else dd["amount_unit"]
        S(
            "unit",
            f"{nm}_unit_counts",
            json.dumps(u.fillna("NONE").value_counts().to_dict(), ensure_ascii=False),
        )
        v = dd[col]
        neg = dd[v < 0]
        zero = dd[v == 0]
        big = dd[v > 1e14]
        small = dd[(v > 0) & (v < 1e6)]
        S("range", f"{nm}_negative", len(neg))
        S("range", f"{nm}_zero", len(zero))
        S("range", f"{nm}_gt_1e14", len(big))
        S("range", f"{nm}_lt_1e6", len(small))
        gt_sales = dd[(dd[col] > dd.recent_sales * 20) & dd.recent_sales.notna()]
        S("range", f"{nm}_amount_gt_20x_recent_sales", len(gt_sales))
        for tag, x in (
            ("negative", neg),
            ("zero", zero),
            ("gt_1e14", big),
            ("lt_1e6", small),
            ("gt_20x_sales", gt_sales),
        ):
            unit.append(
                x[["rcept_no", "rcept_dt", "report_nm", f"{col}_raw", "recent_sales_raw"]]
                .rename(columns={f"{col}_raw": "amount_raw"})
                .assign(table=nm, reason=tag)
            )
        S(
            "unit",
            f"{nm}_unit_assumed",
            int(dd[f"{col}_unit_assumed"].sum()) if f"{col}_unit_assumed" in dd else 0,
        )
        S("unit", f"{nm}_fx_hint_docs", int(dd.fx_hint.notna().sum()) if "fx_hint" in dd else 0)
    pd.concat(unit).to_csv(out / "qa_unit_range_candidates.tsv", sep="\t", index=False)

    # 6. 날짜 순서
    dd = ev[ev.parse_status != "fail"]
    o1 = dd[(dd.period_start.notna()) & (dd.period_end.notna()) & (dd.period_start > dd.period_end)]
    rd = dd.rcept_dt.str[:4] + "-" + dd.rcept_dt.str[4:6] + "-" + dd.rcept_dt.str[6:8]
    o2 = dd[dd.contract_date.notna() & (dd.contract_date > rd)]
    S("date", "event_period_start_gt_end", len(o1))
    S("date", "event_contract_date_gt_rcept_dt(기록용)", len(o2))
    dc = ca[ca.parse_status != "fail"]
    rdc = dc.rcept_dt.str[:4] + "-" + dc.rcept_dt.str[4:6] + "-" + dc.rcept_dt.str[6:8]
    o3 = dc[dc.period_start.notna() & dc.period_end.notna() & (dc.period_start > dc.period_end)]
    o4 = dc[dc.cancel_date.notna() & (dc.cancel_date > rdc)]
    S("date", "cancel_period_start_gt_end", len(o3))
    S("date", "cancel_date_gt_rcept_dt(기록용)", len(o4))
    pd.concat(
        [
            o1[["rcept_no", "period_start_raw", "period_end_raw"]].assign(
                table="event", reason="period_order"
            ),
            o3[["rcept_no", "period_start_raw", "period_end_raw"]].assign(
                table="cancel", reason="period_order"
            ),
        ]
    ).to_csv(out / "qa_date_order.tsv", sep="\t", index=False)
    S(
        "date",
        "contract_date_year_outside_1990_2100",
        int(ev.parse_err.fillna("").str.contains("contract_date").sum()),
    )

    # 7. 해지 → 최초 체결 연결
    vc = ca.link_method.value_counts()
    for k, v in vc.items():
        S("cancel_link", f"method_{k}", int(v))
    withc = ca[
        ca.link_method.isin(
            [
                "unique_corp_date",
                "name_match",
                "amount_match",
                "unlinked_first_not_collected",
                "ambiguous_multiple_candidates",
            ]
        )
    ]
    S("cancel_link", "cancel_with_conclusion_pair", len(withc))
    S("cancel_link", "linked", int(ca.first_rcept_no.notna().sum()))
    S(
        "cancel_link",
        "linked_rate_of_pairs",
        round(float(ca.first_rcept_no.notna().sum() / max(len(withc), 1)), 4),
    )
    S("cancel_link", "linked_rate_of_all_cancel", round(float(ca.first_rcept_no.notna().mean()), 4))
    S("related", "pairs_total", len(rel))
    S("related", "pairs_with_rcept_no", int(rel.related_rcept_no.notna().sum()))
    for k, v in rel.title_class.value_counts().items():
        S("related", f"class_{k}", int(v))
    ca[
        [
            "rcept_no",
            "corp_code",
            "link_method",
            "n_candidates",
            "first_rcept_no",
            "first_pair_date",
        ]
    ].to_csv(out / "qa_cancel_link.tsv", sep="\t", index=False)

    # 8. 상대방 분류
    cls = cp.cp_class.value_counts()
    first = cp[cp.part_idx == 0].cp_class.value_counts()
    pd.DataFrame({"parts": cls, "docs_first_part": first}).fillna(0).astype(int).rename_axis(
        "cp_class"
    ).to_csv(out / "qa_cp_class.tsv", sep="\t")
    for k, v in first.items():
        S("cp_class_doc", k, f"{int(v)} ({v / first.sum():.3f})")
    S("cp_class", "parts_total", len(cp))
    S("cp_class", "docs_with_multi_parts", int((cp.groupby("rcept_no").size() > 1).sum()))
    listed = cp[cp.cp_class.str.startswith("listed")]
    S("cp_class", "listed_parts", len(listed))
    S("cp_class", "listed_docs", int(listed.rcept_no.nunique()))

    pd.DataFrame(summ, columns=["section", "name", "value"]).to_csv(
        out / "qa_summary.tsv", sep="\t", index=False
    )

    # 대조용 표본
    vs = _verify_sample(out, ev, ca, bad, pd.concat(unit), sample_exclude)
    return {"summary_rows": len(summ), **vs}


def _verify_sample(out: Path, ev, ca, bad, unit, exclude_path: Path | None) -> dict:
    vdir = out / "verify_sample"
    vdir.mkdir(exist_ok=True)
    excl: set[str] = set()
    if exclude_path and exclude_path.exists():
        excl = set(pd.read_csv(exclude_path, sep="\t", dtype=str)["rcept_no"])
    rng = random.Random(SEED)
    parts = []
    base = ev[(ev.parse_status != "fail") & ~ev.rcept_no.isin(excl)].copy()
    base["stratum"] = base.rcept_dt.str[:4].astype(int).map(_bin) + "|" + base.form_variant
    cb = ca[(ca.parse_status != "fail") & ~ca.rcept_no.isin(excl)].copy()
    cb["stratum"] = cb.rcept_dt.str[:4].astype(int).map(_bin) + "|cancel"
    layers = {}
    for tbl, d, cols in (("event", base, SAMPLE_COLS_EVENT), ("cancel", cb, SAMPLE_COLS_CANCEL)):
        for st, g in d.groupby("stratum"):
            ids = sorted(g.rcept_no)  # 정렬한 뒤 시드 고정 셔플 → 재현 가능
            rng.shuffle(ids)
            pick = ids[:PER_STRATUM_POOL]
            layers[st] = (tbl, len(g), len(pick))
            sub = g.set_index("rcept_no").loc[pick].reset_index()
            sub["seq"] = range(1, len(sub) + 1)
            sub["tier"] = ["core" if s <= PER_STRATUM_CORE else "extra" for s in sub["seq"]]
            sub["table"] = tbl
            parts.append(
                sub[
                    [
                        "stratum",
                        "seq",
                        "tier",
                        "table",
                        "rcept_no",
                        "rcept_dt",
                        "corp_code",
                        "report_nm",
                        "form_variant",
                        "parse_status",
                        "src_zip_sha256",
                        *cols,
                    ]
                ]
            )
    samp = pd.concat(parts, ignore_index=True)
    samp.to_csv(vdir / "sample_a_ga.tsv", sep="\t", index=False)
    samp[samp.tier == "core"][["stratum", "seq", "rcept_no"]].to_csv(
        vdir / "sample_core_rcept_no.tsv", sep="\t", index=False
    )
    # 오류 후보(자동 점검에서 걸린 것): 층화 표본과 별도로 대조 대상에 더한다
    cand = pd.concat(
        [
            bad[["rcept_no", "table"]].assign(reason="ratio_mismatch"),
            unit[["rcept_no", "table", "reason"]],
        ]
    ).drop_duplicates()
    cand = cand[~cand.rcept_no.isin(excl)]
    cand.to_csv(vdir / "extra_candidates_auto_checks.tsv", sep="\t", index=False)
    meta = {
        "seed": SEED,
        "per_stratum_core": PER_STRATUM_CORE,
        "per_stratum_pool": PER_STRATUM_POOL,
        "strata": {
            k: {"table": v[0], "population": v[1], "listed": v[2]}
            for k, v in sorted(layers.items())
        },
        "n_core": int((samp.tier == "core").sum()),
        "n_listed": len(samp),
        "excluded_06_sample": str(exclude_path) if exclude_path else None,
        "n_excluded_ids": len(excl),
        "method": "층마다 rcept_no 정렬 후 random.Random(seed).shuffle, 앞 40건 목록·앞 15건 core",
        "note": "파서 값만 담았다. 원문과의 대조·합격 판정은 하지 않았다.",
    }
    (vdir / "sample_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"n_core": meta["n_core"], "n_listed": meta["n_listed"], "n_extra_candidates": len(cand)}


def main() -> None:
    ex = data_root() / "kr/output/research_prep_20261027/b_contract_fields/sample_rcept_no.tsv"
    print(json.dumps(run(ex), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
