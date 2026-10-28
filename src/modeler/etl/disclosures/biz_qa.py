"""사업보고서(b) 구조화 전수 자동 점검 (계획 07 §7.3).

``biz_build``가 만든 ``kr_bizrpt_section_index.parquet``·``kr_bizrpt_table_cell/``을 읽어
``qa_b_*.tsv``를 출력 폴더에 쓴다. 문턱은 판정하지 않고 숫자만 낸다(판정은 결과 문서에서).

    python -m modeler.etl.disclosures.biz_qa [--out-name disclosure_structuring_v1]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from modeler.etl.disclosures.biz_build import CELL_NAME, INDEX_NAME, OUT_NAME
from modeler.etl.disclosures.bizparse import SECTION_KEYS
from modeler.etl.disclosures.build import data_root

BIZ_KEYS = ("main_products", "raw_materials", "sales", "orders")
# 06 표본(N=60: 접수 ≤2022 40건 · ≥2023 20건) 탐지 건수. 06 §7 표
BASELINE_06 = {
    "le2022": (40, {"main_products": 26, "raw_materials": 23, "sales": 31, "orders": 21}),
    "ge2023": (20, {"main_products": 19, "raw_materials": 19, "sales": 19, "orders": 19}),
}


def _tsv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _df_tsv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")


def _era(y: int) -> str:
    return "le2022" if y <= 2022 else "ge2023"


def run(out_name: str = OUT_NAME) -> dict:
    root = data_root()
    out = root / "kr/output" / out_name
    doc_root = root / "kr/raw/dart_doc_original/b"
    idx = pd.read_parquet(out / f"{INDEX_NAME}.parquet")
    mf = json.loads((out / "manifest_bizrpt.json").read_text(encoding="utf-8"))
    res: dict = {}

    docs = idx.drop_duplicates("rcept_no").copy()
    docs["year"] = docs["rcept_dt"].str[:4].astype(int)
    n_docs = len(docs)
    with open(doc_root / "targets.tsv", encoding="utf-8") as f:
        n_targets = sum(1 for _ in f) - 1
    res["n_targets_file"] = n_targets
    res["n_processed_manifest"] = mf["n_processed"]
    res["n_docs_in_index"] = n_docs
    res["index_rows"] = len(idx)
    res["index_rows_expected"] = n_docs * len(SECTION_KEYS)
    res["rcept_no_unique"] = bool(docs.rcept_no.is_unique)
    res["index_key_dups"] = int(idx.duplicated(["rcept_no", "section_key"]).sum())
    res["input_equals_output"] = n_docs == mf["n_processed"] and len(idx) == n_docs * len(
        SECTION_KEYS
    )

    # 상태·인코딩·변환 오류
    st = docs.groupby("parse_status").size()
    res["parse_status"] = {k: int(v) for k, v in st.items()}
    res["fail_ratio"] = float((docs.parse_status == "fail").mean())
    res["fail_gate_lt_0.5pct"] = res["fail_ratio"] < 0.005
    res["encoding"] = {
        k: int(v) for k, v in docs.groupby("src_encoding", dropna=False).size().items()
    }
    res["encoding_replace"] = int((docs.src_encoding == "replace").sum())
    res["encoding_mismatch_declared"] = int(docs.src_encoding_mismatch.fillna(False).sum())
    errs = docs.parse_err.dropna().str.split(",").explode()
    res["parse_err_counts"] = {k: int(v) for k, v in errs.value_counts().items()}
    res["num_conv_err_docs"] = int(res["parse_err_counts"].get("num_conv_err", 0))
    fails = docs[docs.parse_status == "fail"][
        ["rcept_no", "rcept_dt", "corp_cls", "report_nm", "parse_fail_reason"]
    ]
    _df_tsv(out / "qa_b_fail_list.tsv", fails)
    odd = docs[(docs.parse_status != "ok") | (docs.src_encoding == "replace")][
        ["rcept_no", "rcept_dt", "corp_cls", "parse_status", "src_encoding", "parse_err"]
    ]
    _df_tsv(out / "qa_b_nonok_list.tsv", odd)

    # 원문 zip 해시 대조
    def _read_sha(p: Path) -> dict[str, str]:
        d = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                d[Path(parts[-1]).stem] = parts[0]
        return d

    mac = _read_sha(doc_root.parent / "b_mac_sha256.txt")
    srv = _read_sha(doc_root.parent / "b_server_sha256.txt")
    ok = docs.dropna(subset=["src_zip_sha256"])
    res["sha_checked"] = len(ok)
    res["sha_mismatch_vs_mac"] = int(
        sum(mac.get(r) != h for r, h in zip(ok.rcept_no, ok.src_zip_sha256))
    )
    res["sha_mismatch_vs_server"] = int(
        sum(srv.get(r) != h for r, h in zip(ok.rcept_no, ok.src_zip_sha256))
    )

    # 절 키별 탐지율 (연도별·시대별)
    idx["year"] = idx["rcept_dt"].str[:4].astype(int)
    idx["found"] = idx["found"].fillna(False).astype(bool)
    g = idx.groupby(["section_key", "year"]).agg(n=("found", "size"), found=("found", "sum"))
    g["rate"] = (g.found / g.n).round(4)
    g = g.reset_index()
    _df_tsv(out / "qa_b_detect_by_year_key.tsv", g)
    rows = []
    for key in SECTION_KEYS:
        for era in ("le2022", "ge2023"):
            sub = idx[(idx.section_key == key) & (idx.year.map(_era) == era)]
            rows.append(
                {
                    "section_key": key,
                    "era": era,
                    "n": len(sub),
                    "found": int(sub.found.sum()),
                    "rate": round(float(sub.found.mean()), 4) if len(sub) else None,
                    "baseline_06_rate": (
                        round(BASELINE_06[era][1][key] / BASELINE_06[era][0], 4)
                        if key in BIZ_KEYS
                        else None
                    ),
                }
            )
    _df_tsv(out / "qa_b_detect_by_era.tsv", pd.DataFrame(rows))
    dev = []
    for _, r in g.iterrows():
        key, y = r.section_key, int(r.year)
        if key not in BIZ_KEYS:
            continue
        base = BASELINE_06[_era(y)]
        b = base[1][key] / base[0]
        if abs(r.rate - b) >= 0.10:
            dev.append(
                {
                    "section_key": key,
                    "year": y,
                    "n": int(r.n),
                    "rate": r.rate,
                    "baseline_06_rate": round(b, 4),
                    "diff_pp": round((r.rate - b) * 100, 1),
                }
            )
    _df_tsv(out / "qa_b_detect_deviation_ge10pp.tsv", pd.DataFrame(dev))
    res["detect_deviation_ge10pp_n"] = len(dev)

    # 방법별
    m = idx[idx.found].groupby(["section_key", "method"]).size().reset_index(name="n")
    _df_tsv(out / "qa_b_method.tsv", m)
    res["method_counts"] = {
        f"{r.section_key}/{r.method}": int(r.n) for r in m.itertuples(index=False)
    }
    miss = idx[~idx.found].groupby(["section_key", "miss_reason"], dropna=False).size()
    res["miss_reason"] = {f"{a}/{b}": int(v) for (a, b), v in miss.items()}

    # 서식: 금융업 등 절 없는 서식
    form = docs.groupby("biz_form", dropna=False).size()
    res["biz_form"] = {str(k): int(v) for k, v in form.items()}
    bizf = idx[idx.section_key.isin(BIZ_KEYS)].groupby("rcept_no").found.sum()
    docs["n_biz_keys_found"] = docs.rcept_no.map(bizf).fillna(0).astype(int)
    z = docs[docs.n_biz_keys_found == 0]
    res["docs_no_biz_key_found"] = int(len(z))
    res["docs_no_biz_key_found_by_form"] = {
        str(k): int(v) for k, v in z.groupby("biz_form", dropna=False).size().items()
    }
    res["docs_no_biz_key_found_by_year"] = {
        int(k): int(v) for k, v in z.groupby("year").size().items()
    }
    _df_tsv(
        out / "qa_b_no_biz_key_found.tsv",
        z[["rcept_no", "rcept_dt", "corp_cls", "biz_form", "parse_status"]],
    )

    # 셀 표
    cell_ds = ds.dataset(str(out / CELL_NAME), format="parquet", partitioning="hive")
    n_cells = cell_ds.count_rows()
    res["cell_rows"] = n_cells
    t0 = cell_ds.to_table(
        columns=[
            "rcept_no",
            "section_key",
            "table_idx",
            "row_idx",
            "col_idx",
            "unit_raw",
            "unit_src",
            "tbl_n_rows",
            "tbl_n_cols",
            "tbl_header_rule",
            "tbl_has_nested",
        ],
        filter=(ds.field("row_idx") == 0) & (ds.field("col_idx") == 0),
    ).to_pandas()
    t0 = t0[t0.section_key != "_doc"]
    res["tables"] = len(t0)
    res["tables_by_key"] = {k: int(v) for k, v in t0.groupby("section_key").size().items()}
    # 인덱스 n_tables 대비 (0,0) 셀이 있는 표: 셀 없는 표 찾기
    nt = idx[idx.found].groupby(["rcept_no", "section_key"]).n_tables.sum().sum()
    res["index_n_tables_sum"] = int(nt)
    res["tables_with_cell00"] = len(t0)
    res["tables_without_cells"] = int(nt) - len(t0)
    res["tables_nested"] = int(t0.tbl_has_nested.fillna(False).sum())
    res["header_rule"] = {k: int(v) for k, v in t0.groupby("tbl_header_rule").size().items()}
    res["unit_src"] = {
        str(k): int(v) for k, v in t0.groupby("unit_src", dropna=False).size().items()
    }
    res["tables_unit_null_ratio"] = float(t0.unit_raw.isna().mean())
    # unit_raw 분포
    top = (
        t0.groupby("unit_raw", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="n")
    )
    _df_tsv(out / "qa_b_unit_top.tsv", top.head(300))
    res["unit_raw_top10"] = [
        [None if pd.isna(a) else a, int(b)] for a, b in top.head(10).itertuples(index=False)
    ]
    res["unit_raw_distinct"] = int(top.shape[0])
    # 단위 모양이 이상한 것(단위 글자가 없음)
    ok_unit = re.compile(
        r"(원|천원|백만원|십억원|억원|만원|백만|천|%|USD|달러|EUR|JPY|CNY|엔|위안|톤|㎡|개|명|주|건|EA|ea|TEU|㎏|kg|Kg|M/T|MT|배럴|ℓ|L|㎘|대|세트|식|매|본|롤|m|M|km|평)"
    )
    weird = top[
        top.unit_raw.notna() & ~top.unit_raw.fillna("").apply(lambda s: bool(ok_unit.search(s)))
    ]
    _df_tsv(out / "qa_b_unit_weird.tsv", weird.head(300))
    res["unit_raw_no_unit_word_distinct"] = int(len(weird))

    # 셀 수 분포
    cols = ["rcept_no", "section_key", "table_idx", "rowspan", "colspan", "is_header", "num_value"]
    per_doc: dict[str, int] = {}
    per_table: dict[tuple, int] = {}
    dup_total = 0
    span_gt1 = 0
    hdr = 0
    numeric = 0
    num_null_digit = 0
    unparsed: dict[str, int] = {}
    digit_re = re.compile(r"^[\(\-△▲−－]?\s*[\d,\.]+\s*[\)%]?$")
    for frag in cell_ds.get_fragments():
        t = frag.to_table(columns=cols + ["row_idx", "col_idx", "text_raw"]).to_pandas()
        t = t[t.section_key != "_doc"]
        dup_total += int(
            t.duplicated(["rcept_no", "section_key", "table_idx", "row_idx", "col_idx"]).sum()
        )
        for k, v in t.groupby("rcept_no").size().items():
            per_doc[k] = per_doc.get(k, 0) + int(v)
        for k, v in t.groupby(["rcept_no", "section_key", "table_idx"]).size().items():
            per_table[k] = int(v)
        span_gt1 += int(((t.rowspan > 1) | (t.colspan > 1)).sum())
        hdr += int(t.is_header.sum())
        numeric += int(t.num_value.notna().sum())
        cand = t[t.num_value.isna() & ~t.is_header & (t.text_raw.str.len() <= 30)]
        mask = cand.text_raw.apply(
            lambda s: bool(digit_re.match(s)) and any(ch.isdigit() for ch in s)
        )
        for s in cand[mask].text_raw:
            unparsed[s] = unparsed.get(s, 0) + 1
            num_null_digit += 1
    res["cell_key_duplicates"] = dup_total
    res["cells_with_span_gt1"] = span_gt1
    res["cells_header"] = hdr
    res["cells_numeric"] = numeric
    res["cells_digit_like_but_null_num"] = num_null_digit
    _df_tsv(
        out / "qa_b_num_unparsed.tsv",
        pd.DataFrame(
            sorted(unparsed.items(), key=lambda kv: -kv[1])[:500], columns=["text_raw", "n"]
        ),
    )
    pdoc = pd.Series(per_doc)
    res["cells_per_doc"] = {
        "docs_with_cells": int(len(pdoc)),
        "docs_zero_cells": int(n_docs - len(pdoc)),
        **{f"q{int(q*100)}": float(pdoc.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "max": int(pdoc.max()),
        "mean": round(float(pdoc.mean()), 1),
    }
    ptab = pd.Series(per_table)
    res["cells_per_table"] = {
        **{f"q{int(q*100)}": float(ptab.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "max": int(ptab.max()),
        "mean": round(float(ptab.mean()), 1),
        "tables_gt_5000_cells": int((ptab > 5000).sum()),
    }
    _df_tsv(
        out / "qa_b_cells_dist.tsv",
        pd.DataFrame(
            [
                {"scope": "doc", **res["cells_per_doc"]},
                {"scope": "table", **res["cells_per_table"]},
            ]
        ),
    )
    # 표 격자 크기 = 셀 수 정합(표 단위): 최소 점검 — 셀 수가 격자보다 크면 안 된다
    tabs = t0.set_index(["rcept_no", "section_key", "table_idx"])
    cnt = ptab.reindex(tabs.index)
    grid = tabs.tbl_n_rows * tabs.tbl_n_cols
    res["tables_cells_gt_grid"] = int((cnt > grid).sum())
    res["tables_cells_eq_grid"] = int((cnt == grid).sum())

    # 같은 문서에서 범위가 같은 절(제품·원재료가 한 절) 수
    f = idx[idx.found].pivot_table(
        index="rcept_no", columns="section_key", values="char_start", aggfunc="first"
    )
    same = (
        int((f.main_products == f.raw_materials).sum())
        if {"main_products", "raw_materials"} <= set(f.columns)
        else 0
    )
    res["docs_products_same_start_as_materials"] = same

    pd.Series(res).to_json(out / "qa_b_summary.json", force_ascii=False, indent=2)
    rows_out = [{"metric": k, "value": json.dumps(v, ensure_ascii=False)} for k, v in res.items()]
    _tsv(out / "qa_b_summary.tsv", rows_out, ["metric", "value"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default=OUT_NAME)
    a = ap.parse_args()
    print(json.dumps(run(a.out_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
