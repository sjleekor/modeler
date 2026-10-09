"""ETF 분류 표와 연도별 비교 그룹 크기 분포를 만들고 사전등록 §6 표와 맞춰 본다.

점수·결과와 연결하지 않는다. 쓰는 칸은 이름·지수명·거래일(종가 유무)뿐이다.

    STOCK_DATA_ROOT=../stock_data python -m modeler.scores.quality.etf_report

경로는 환경변수로 받는다(기본 ``../stock_data``). 산출물은 ``stock_data/``에만 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl

from modeler.scores.quality import etf_classify as ec

DEFAULT_INPUT_REL = "kr/output/etf_krx_research_pull_20261009/etf_daily_2010_20261008.csv.gz"
DEFAULT_OUT_REL = "kr/output/quality_score_prep_20261010"

# 사전등록 §6 "E2 비교 그룹 크기 분포" 표.
# 연도 -> (상장, (가)≥5, (가)≥3, (나)≥5, (나)≥3), 각 칸은 (그룹 수, ETF 수)
SECTION6 = {
    2010: (50, (1, 5), (3, 12), (1, 5), (3, 12)),
    2011: (64, (1, 8), (4, 18), (1, 5), (5, 18)),
    2012: (106, (1, 9), (4, 20), (1, 6), (5, 20)),
    2013: (135, (2, 17), (5, 27), (1, 8), (6, 26)),
    2014: (146, (2, 17), (4, 24), (2, 13), (5, 23)),
    2015: (172, (2, 17), (8, 37), (2, 13), (7, 29)),
    2016: (198, (4, 28), (10, 49), (2, 13), (11, 41)),
    2017: (256, (7, 56), (10, 67), (4, 34), (11, 56)),
    2018: (325, (7, 57), (17, 91), (4, 35), (15, 70)),
    2019: (413, (12, 98), (24, 139), (8, 63), (24, 115)),
    2020: (450, (12, 100), (25, 144), (9, 70), (25, 121)),
    2021: (468, (12, 103), (30, 165), (9, 72), (25, 125)),
    2022: (533, (14, 119), (32, 180), (11, 82), (27, 134)),
    2023: (666, (15, 143), (35, 209), (12, 93), (31, 155)),
    2024: (812, (17, 163), (42, 245), (13, 104), (37, 181)),
    2025: (935, (18, 175), (41, 250), (13, 109), (34, 180)),
    2026: (1058, (18, 184), (39, 253), (15, 123), (33, 184)),
}

# §6 (나)의 "이름 토큰 네 가지"를 글자 그대로 읽은 것(재현용; 파서와 다르다).
_SIMPLE_TOKENS = (
    # 글자 그대로 "(H)"만 보면 2016~2024의 (나)≥3 칸이 §6과 어긋난다.
    # §6 표는 "(합성 H)"도 환헤지로 셌다.
    ("hedge", lambda n: ("(H)" in n) or ("합성 H" in n)),
    ("active", lambda n: "액티브" in n),
    ("levinv", lambda n: ("레버리지" in n) or ("인버스" in n)),
    ("option", lambda n: ("커버드콜" in n) or ("옵션" in n)),
)


def _simple_token_key(n: str | None) -> str:
    n = n or ""
    return "".join("1" if f(n) else "0" for _, f in _SIMPLE_TOKENS)


def first_trading_day_by_year(days: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for d in days:  # 오름차순
        out.setdefault(int(d[:4]), d)
    return out


def _size_cells(keys: list[str | None]) -> dict:
    """키 목록(ETF당 하나, None=보류)에서 크기 ≥3·≥5 그룹 수와 ETF 수."""
    cnt: dict[str, int] = {}
    for k in keys:
        if k is not None:
            cnt[k] = cnt.get(k, 0) + 1
    out = {}
    for t in (5, 3):
        g = [c for c in cnt.values() if c >= t]
        out[t] = (len(g), sum(g))
    out["held"] = sum(1 for k in keys if k is None)
    return out


def distribution(df: pl.DataFrame, cls: pl.DataFrame) -> pl.DataFrame:
    """연도별 분포. 시점 정의별로 여러 방식을 한 표에 낸다.

    reproduce_*  : §6 방식(그 해 첫 거래일 행의 IDX_IND_NM·ISU_NM). §6 표 재현용.
    fixed_*      : §11.1 방식(마지막 거래일 값으로 고정한 분류). 같은 지수명 / +토큰 넷 / 전체 키.
    """
    days = ec.trading_days(df)
    fy = first_trading_day_by_year(days)
    close = df.filter(ec.has_close())
    cls_map = {r["isu_cd"]: r for r in cls.iter_rows(named=True)}
    rows = []
    for year, d in sorted(fy.items()):
        listed = close.filter(pl.col("BAS_DD") == d)
        recs = listed.select("ISU_CD", "ISU_NM", "IDX_IND_NM").iter_rows()
        recs = list(recs)
        row: dict = {"year": year, "first_trading_day": d, "listed": len(recs)}
        # (가)(나) 재현: 그 날 행의 값
        ga = [r[2] if r[2] is not None else None for r in recs]
        na = [None if r[2] is None else f"{r[2]}|{_simple_token_key(r[1])}" for r in recs]
        # 고정(마지막 거래일) 값
        fga = [cls_map[r[0]]["idx_ind_nm"] for r in recs]
        fna = [
            f"{cls_map[r[0]]['idx_ind_nm']}|{_simple_token_key(cls_map[r[0]]['isu_nm'])}"
            for r in recs
        ]
        full = [cls_map[r[0]]["group_key"] for r in recs]
        full_strict = [
            None if cls_map[r[0]]["compare_hold_strict_rt"] else cls_map[r[0]]["group_key"]
            for r in recs
        ]
        # 국내형만(E1·E2 모두 국내형이 주력이므로 참고로 따로)
        for name, keys in (
            ("repro_ga", ga),
            ("repro_na", na),
            ("fixed_ga", fga),
            ("fixed_na", fna),
            ("full_lenient", full),
            ("full_strict_rt", full_strict),
        ):
            c = _size_cells(keys)
            for t in (5, 3):
                row[f"{name}_g{t}"] = c[t][0]
                row[f"{name}_e{t}"] = c[t][1]
            row[f"{name}_held"] = c["held"]
        rows.append(row)
    return pl.DataFrame(rows)


def compare_section6(dist: pl.DataFrame) -> pl.DataFrame:
    """§6 표와 재현(repro_*) 값을 칸별로 비교한다."""
    out = []
    for r in dist.iter_rows(named=True):
        y = r["year"]
        if y not in SECTION6:
            continue
        listed, ga5, ga3, na5, na3 = SECTION6[y]
        mine = {
            "listed": r["listed"],
            "ga5": (r["repro_ga_g5"], r["repro_ga_e5"]),
            "ga3": (r["repro_ga_g3"], r["repro_ga_e3"]),
            "na5": (r["repro_na_g5"], r["repro_na_e5"]),
            "na3": (r["repro_na_g3"], r["repro_na_e3"]),
        }
        doc = {"listed": listed, "ga5": ga5, "ga3": ga3, "na5": na5, "na3": na3}
        out.append(
            {
                "year": y,
                **{f"doc_{k}": str(v) for k, v in doc.items()},
                **{f"mine_{k}": str(v) for k, v in mine.items()},
                "all_match": all(mine[k] == doc[k] for k in doc),
                "mismatch_cells": ",".join(k for k in doc if mine[k] != doc[k]),
            }
        )
    return pl.DataFrame(out)


def _sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.check_output(["git", "-C", str(here), *args], text=True).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--input", default=str(root / DEFAULT_INPUT_REL))
    ap.add_argument("--out-dir", default=str(root / DEFAULT_OUT_REL))
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = ec.read_input(a.input)
    days = ec.trading_days(df)
    cls = ec.classification_table(df)
    dist = distribution(df, cls)
    cmp6 = compare_section6(dist)

    cls.write_csv(out / "etf_classification.csv")
    dist.write_csv(out / "etf_group_size_by_year.csv")
    cmp6.write_csv(out / "etf_group_size_vs_section6.csv")
    cls.select("idx_ind_nm", "base_index", "region", "region_rule").unique().sort(
        "idx_ind_nm"
    ).write_csv(out / "etf_index_region_list.csv")

    src = Path(ec.__file__)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "quality-score v0 ETF 분류 파서 동결 전 준비(사전등록 §11.1·§12.1). "
        "점수·결과 연결 없음.",
        "input": {
            "path": str(a.input),
            "sha256": _sha256(a.input),
            "columns_used": ec.INPUT_COLUMNS,
            "rows": df.height,
            "trading_days": len(days),
            "distinct_isu_cd": df["ISU_CD"].n_unique(),
        },
        "parser": {
            "version": ec.PARSER_VERSION,
            "file": str(src),
            "sha256": _sha256(src),
            "report_file_sha256": _sha256(Path(__file__)),
        },
        "code": {
            "repo": "modeler",
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "worktree_dirty": bool(_git("status", "--porcelain", "--", ".")),
        },
        "outputs": sorted(p.name for p in out.glob("*.csv")),
        "ambiguities": ec.AMBIGUITIES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: manifest[k] for k in ("input", "parser", "code")}, ensure_ascii=False, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
