"""조사 사본 결과와 정본 결과를 대조한다 (09b §12).

``python -m modeler.scores.etf_baseline.compare --canonical-run <정본 결과 폴더>``

결과 폴더 둘(``metrics.csv``·``nav_daily.parquet``)에서 기준선별 연환산 수익·최대낙폭·끝 날짜와
일별 수익 차이 최대값을 비교하고, 입력 층에서 거래일 목록·ETF 일봉(종가·상장좌수)·분배금 행의
차이를 센다. 입력 층은 정본 결과 manifest의 ``source``에 적힌 설정(정본 경로·관측판·끝 날짜)으로
다시 읽는다. 읽기만 하고, 조사 사본 결과 폴더에는 아무것도 쓰지 않는다.
정본 결과 폴더에 ``compare_vs_research.json``을 쓴다(``--no-write``로 끈다).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

from modeler.scores.etf_baseline import canonical as C
from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import run as R

RESEARCH_RUN_DEFAULT = "run_20261018_main"
HEADLINE_LINES = ["B1", "B2", "B3", "B4", "B5", "R1", "R2"]
HEADLINE_UNIVERSE, HEADLINE_COST = "main", "low"  # 주 구간 · 왕복 10bp


def _metrics(run: Path) -> pl.DataFrame:
    return pl.read_csv(run / "metrics.csv")


def compare_metrics(res: Path, can: Path) -> dict:
    a, b = _metrics(res), _metrics(can)
    keys = ["universe", "cost", "line"]
    j = a.join(b, on=keys, how="full", suffix="_can", coalesce=True)
    rows = []
    for r in j.iter_rows(named=True):
        rows.append(
            {
                "universe": r["universe"],
                "cost": r["cost"],
                "line": r["line"],
                "cagr_research": r["cagr"],
                "cagr_canonical": r["cagr_can"],
                "mdd_research": r["mdd"],
                "mdd_canonical": r["mdd_can"],
                "end_research": r["end"],
                "end_canonical": r["end_can"],
            }
        )

    def diff(x, y):
        return None if x is None or y is None else abs(x - y)

    for r in rows:
        r["cagr_absdiff"] = diff(r["cagr_research"], r["cagr_canonical"])
        r["mdd_absdiff"] = diff(r["mdd_research"], r["mdd_canonical"])
    present = [r for r in rows if r["cagr_absdiff"] is not None]
    return {
        "rows": rows,
        "max_cagr_absdiff": max((r["cagr_absdiff"] for r in present), default=None),
        "max_mdd_absdiff": max((r["mdd_absdiff"] for r in present), default=None),
        "lines_only_in_one_side": [r for r in rows if r["cagr_absdiff"] is None],
        "end_mismatch": [r for r in rows if r["end_research"] != r["end_canonical"]],
    }


def compare_daily_returns(res: Path, can: Path) -> dict:
    keys = ["universe", "cost", "line"]
    a = pl.read_parquet(res / "nav_daily.parquet").rename({"nav": "nav_r"})
    b = pl.read_parquet(can / "nav_daily.parquet").rename({"nav": "nav_c"})
    j = a.join(b, on=[*keys, "date"], how="full", coalesce=True).sort([*keys, "date"])
    only_r = j.filter(pl.col("nav_c").is_null()).height
    only_c = j.filter(pl.col("nav_r").is_null()).height
    both = j.drop_nulls(["nav_r", "nav_c"]).with_columns(
        (pl.col("nav_r") / pl.col("nav_r").shift(1).over(keys) - 1).alias("ret_r"),
        (pl.col("nav_c") / pl.col("nav_c").shift(1).over(keys) - 1).alias("ret_c"),
    )
    both = both.with_columns((pl.col("ret_r") - pl.col("ret_c")).abs().alias("d"))
    worst = both.drop_nulls("d").sort("d", descending=True).head(1)
    per = (
        both.drop_nulls("d")
        .group_by(keys)
        .agg(pl.col("d").max().alias("max_abs_daily_ret_diff"), pl.col("nav_r").last())
        .sort(keys)
    )
    nav_last = both.group_by(keys).agg(
        (pl.col("nav_r").last() - pl.col("nav_c").last()).abs().alias("last_nav_absdiff")
    )
    return {
        "dates_only_in_research": only_r,
        "dates_only_in_canonical": only_c,
        "max_abs_daily_ret_diff": None if worst.is_empty() else float(worst["d"][0]),
        "worst": None
        if worst.is_empty()
        else {k: str(v) for k, v in worst.row(0, named=True).items()},
        "per_line": per.drop("nav_r").to_dicts(),
        "max_last_nav_absdiff": float(nav_last["last_nav_absdiff"].max()),
    }


def compare_inputs(root: Path, src: dict) -> dict:
    """입력 층 차이. 정본은 결과 manifest의 source 설정으로 다시 읽는다."""
    end = date.fromisoformat(src["end_date"])
    if "canonical_base" not in src:
        return {"note": "정본 입력을 쓰지 않은 결과입니다 — 입력 층 대조 없음"}
    store = C.CanonicalStore(Path(src["canonical_base"]))
    view = src.get("obs_view", "latest")
    mf = root / I.ETF_DIR / I.MANIFEST_FILE
    days_r = [d for d in I.load_trading_days(mf) if d <= end]
    out: dict = {"obs_view": view, "end_date": end.isoformat()}
    if src["kind"] == "canonical":
        days_c, _ = C.load_trading_days(store, end)
        out["trading_days"] = {
            "research": len(days_r),
            "canonical": len(days_c),
            "only_research": [str(d) for d in sorted(set(days_r) - set(days_c))][:20],
            "only_canonical": [str(d) for d in sorted(set(days_c) - set(days_r))][:20],
        }
        fr = I.load_etf_frames(root / I.ETF_DIR / I.ETF_FILE, R.CODES)
        fc = C.load_etf_frames(store, R.CODES, view)
        per = {}
        for c in R.CODES:
            j = fr[c].join(fc[c], on="date", how="full", suffix="_c", coalesce=True)
            both = j.drop_nulls(["close", "close_c"])
            per[c] = {
                "rows_research": fr[c].height,
                "rows_canonical": fc[c].height,
                "dates_only_research": j.filter(pl.col("close_c").is_null()).height,
                "dates_only_canonical": j.filter(pl.col("close").is_null()).height,
                "close_diff_rows": both.filter(pl.col("close") != pl.col("close_c")).height,
                "shares_diff_rows": j.filter(
                    pl.col("shares").fill_null(-1) != pl.col("shares_c").fill_null(-1)
                ).height,
            }
        out["etf_daily"] = per
    if src["dist_kind"] == "canonical":
        dr = I.load_distributions(root / I.DIST_FILE, root / I.MAP_FILE, R.CODES)
        dc, _ = C.load_distributions(store, R.CODES, end, view)
        rr = (
            dr.rows.with_columns(pl.lit(1).alias("n"))
            .group_by(["code", "record_date", "amount"])
            .len()
        )
        cc = dc.rows.group_by(["code", "record_date", "amount"]).len()
        j = rr.join(
            cc, on=["code", "record_date", "amount"], how="full", suffix="_c", coalesce=True
        )
        only_r = j.filter(pl.col("len_c").is_null() | (pl.col("len_c") != pl.col("len")))
        out["distributions"] = {
            "rows_research": dr.rows.height,
            "rows_canonical": dc.rows.height,
            "rows_differing": only_r.height,
            "examples": [
                {k: str(v) for k, v in r.items()}
                for r in only_r.sort("code", "record_date").head(20).to_dicts()
            ],
            "duplicates_dropped": [dr.duplicates_dropped, dc.duplicates_dropped],
            "liquidation_ignored": [dr.liquidation_rows_ignored, dc.liquidation_rows_ignored],
            "isin_equal": dr.isin_by_code == dc.isin_by_code,
        }
    return out


def _fmt(x, pct=True):
    return "-" if x is None else (f"{x:.2%}" if pct else f"{x:.3g}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--canonical-run", type=Path, required=True, help="정본 결과 폴더(run)")
    ap.add_argument(
        "--research-run",
        type=Path,
        default=None,
        help=f"조사 사본 결과 폴더. 기본 <root>/{I.OUTPUT_DIR}/{RESEARCH_RUN_DEFAULT}",
    )
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    root = I.stock_data_root()
    res = a.research_run or root / I.OUTPUT_DIR / RESEARCH_RUN_DEFAULT
    can = a.canonical_run
    for d in (res, can):
        if not (d / "metrics.csv").is_file():
            raise I.InputError(f"{d}: metrics.csv가 없습니다")
    src = json.loads((can / "manifest.json").read_text())["source"]
    mrep = compare_metrics(res, can)
    drep = compare_daily_returns(res, can)
    irep = compare_inputs(root, src)
    print(
        f"조사 사본 결과: {res}\n정본 결과: {can}\n원천: {src['kind']} / 분배금 {src['dist_kind']}"
    )
    print("\n[주 구간 · 왕복 10bp]  줄  연환산(조사→정본)  최대낙폭(조사→정본)  끝 날짜")
    for r in mrep["rows"]:
        if (
            r["universe"] == HEADLINE_UNIVERSE
            and r["cost"] == HEADLINE_COST
            and r["line"] in HEADLINE_LINES
        ):
            print(
                f"  {r['line']:>3}  {_fmt(r['cagr_research'])} -> {_fmt(r['cagr_canonical'])}  "
                f"{_fmt(r['mdd_research'])} -> {_fmt(r['mdd_canonical'])}  "
                f"{r['end_research']} / {r['end_canonical']}"
            )
    print(
        f"\n전 줄 최대 |연환산 차| = {_fmt(mrep['max_cagr_absdiff'], False)}, "
        f"|최대낙폭 차| = {_fmt(mrep['max_mdd_absdiff'], False)}, "
        f"끝 날짜 불일치 {len(mrep['end_mismatch'])}줄, "
        f"한쪽에만 있는 줄 {len(mrep['lines_only_in_one_side'])}"
    )
    print(
        f"일별 수익 차 최대 = {_fmt(drep['max_abs_daily_ret_diff'], False)}, 마지막 NAV 차 최대 = "
        f"{_fmt(drep['max_last_nav_absdiff'], False)}, 날짜 한쪽에만: "
        f"{drep['dates_only_in_research']}/{drep['dates_only_in_canonical']}"
    )
    print("\n[입력 층]\n" + json.dumps(irep, ensure_ascii=False, indent=1, default=str))
    full = {
        "metrics": mrep,
        "daily": drep,
        "inputs": irep,
        "research_run": str(res),
        "canonical_run": str(can),
    }
    if not a.no_write:
        (can / "compare_vs_research.json").write_text(
            json.dumps(full, ensure_ascii=False, indent=1, default=str)
        )
    identical = (
        (mrep["max_cagr_absdiff"] or 0) == 0
        and (mrep["max_mdd_absdiff"] or 0) == 0
        and not mrep["end_mismatch"]
        and not mrep["lines_only_in_one_side"]
        and (drep["max_abs_daily_ret_diff"] or 0) == 0
    )
    print(
        "\n판정:",
        "결과가 비트까지 같습니다" if identical else "차이가 있습니다 — 위 입력 층 차이를 보십시오",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
