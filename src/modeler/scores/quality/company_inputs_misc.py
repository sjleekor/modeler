"""회사 점수(부분 C) 배당·주식수·소각·사건 해 입력 W2b (사전등록 20261010_quality_score §5).

한 행 = (회사 ``corp_code``, 형성 사업연도 ``fy``). 배당·발행주식수·자사주 표 어디든 그 해
사업보고서(``reprt_code`` 11011, ``bsns_year`` = fy)가 있으면 행이 된다.
분모 필터는 W2a 패널과 묶을 때 한다.
결과 사건(배당 삭감·무배당 전환)·점수는 만들지도 세지도 않는다.
coverage는 입력 존재 개수만 센다(§7.2).

규칙 요약
- 판본(§5.1): 사업연도 y 사업보고서의 판본 중 가용일(접수일의 다음 거래일) ≤ B_t인
  가장 최근 접수번호
  (가용일 내림, 접수번호 내림). raw는 키당 최신 접수번호 하나라 정정이 B_t 뒤면 결측이 된다.
- 배당(§5.2 C3): t 보고서의 ``thstrm``·``frmtrm``·``lwfr`` = dps·dps_p1·dps_p2.
  ``knd_source`` 가 ``other_only``·``none`` 이면 결측(``company_maps.dividend_per_report`` 규칙).
- 발행주식수(§5.2 "발행주식수 원천"): ``istc_totqy``, 보통주 행 우선, 없으면 합계 행.
  shares = fy 보고서,
  shares_p1 = fy-1 보고서(같은 B_t).
- 소각(§5.2 C3): 자사주 표의 ``change_qy_incnr`` > 0 이면 1, 행은 있는데 모두 0·``-`` 이면 0,
  자사주 행이 없으면 결측.
- 사건 해(§5.2 "발행주식수 변동 사건 목록", 정정 C-1·C-2): ``capevt*`` 는 동결 목록
  (증자·감자 표 다섯 유형)으로만
  판정하고, ``capevt_rec*`` 는 액면 변경 해를 더한 기록용 비교다.

사전등록이 정하지 않은 선택은 상수로 두고 ``# CI-<이름>`` 주석을 달았다.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from modeler.scores.quality import company_maps as cm
from modeler.scores.quality.company_common import (
    Lake,
    TradingCalendar,
    base_date,
    code_sha256,
    filing_availability,
    guard_years,
    sha256_file,
)

REPRT_ANNUAL = "11011"  # 사업보고서

# CI-raw-root: ``raw_root`` 를 안 주면 ``lake.root`` 아래 ``kr/raw/raw_postgres`` 를 쓴다.
# (company_maps.default_raw_root 는 환경변수만 보므로 Lake와 어긋날 수 있다.)
CI_RAW_ROOT_FROM_LAKE = True  # CI-raw-root

# CI-se-row: se 값은 공백을 모두 지운 뒤 비교한다. 보통주 행이 *있으면* 그 값이 비어 있어도
# 합계로 넘어가지 않는다(행 존재 기준, 사전등록 문면). 보통주 값이 비고 합계에 값이 있는 보고서 수를
# se 요약에 센다.
SE_COMMON = "보통주"
SE_TOTAL = "합계"
# CI-se-common: 보통주 행 판별은 배당의 stock_knd 규칙(cm.classify_stock_knd == "common",
# 문자열에 `보통`이 있고 `우선`·`종류`·`외`가 없음)과 같게 한다 — `보통주식`·`보 통 주` 같은 표기
# 차이가 합계(우선주 포함)로 넘어가지 않게(메인 검증 10-10, 09-30 기준 `보통주식` 1,378행).


def _is_common_se(se: pl.Series) -> pl.Series:
    """se 값마다 classify_stock_knd로 보통주 행인지(고유값 단위로 계산)."""
    uniq = se.unique().to_list()
    m = {u: cm.classify_stock_knd(u) == "common" for u in uniq}
    return se.replace_strict(m, return_dtype=pl.Boolean)


CI_SE_FALLBACK_ON_NULL_VALUE = False  # CI-se-row

# CI-retire-any-class: 소각은 주식 종류(보통·우선)를 가리지 않는다. 우선주 행에서만 > 0 인 보고서는
# 입력 존재 개수(retire_pref_only)로 센다.
CI_RETIRE_ANY_CLASS = True  # CI-retire-any-class

# CI-latest-order: ``dps_latest`` 의 "가장 늦은 판본"은 (report_year 안에서)
# 접수번호 내림 첫 행이다.
# 접수번호 앞 8자리가 접수일이므로 접수일 순서와 같다(기준일·가용일 필터 없음).
CI_LATEST_ORDER = "rcept_no_desc"  # CI-latest-order

# CI-event-no-base: 사건 해는 기준일(B_t)로 거르지 않는다. 사전등록이 사건 목록에 가용일 규칙을
# 적지 않았고, 사건은 증자·감자 표의 사건일(해)이기 때문이다.
CI_EVENT_NO_BASE = True  # CI-event-no-base

PANEL_COLUMNS = [
    "corp_code",
    "fy",
    "dps",
    "dps_p1",
    "dps_p2",
    "dps_src",
    "dps_rcept_no",
    "shares",
    "shares_p1",
    "shares_src",
    "retire",
    "retire_p1",
    "capevt",
    "capevt_p1",
    "capevt_p2",
    "capevt_rec",
    "capevt_rec_p1",
    "capevt_rec_p2",
]

TABLES_RAW = (
    "dart_shareholder_return_raw",
    "dart_share_count_raw",
    "dart_capital_change_raw",
    "dart_filing_receipt_raw",
)


def _root(lake: Lake, raw_root: str | Path | None) -> Path:
    if raw_root is not None:
        return Path(raw_root)
    return Path(lake.root) / "kr" / "raw" / "raw_postgres"


def _scan(raw_root: Path, snap: str, table: str) -> pl.LazyFrame:
    pattern = str(cm.table_dir(raw_root, snap, table) / "**" / "*.parquet")
    return pl.scan_parquet(pattern, hive_partitioning=False)


# ---------------------------------------------------------------- 숫자 파싱
def parse_qty_expr(col: str) -> pl.Expr:
    """수량 칸 → Int64. 쉼표·공백을 지우고 정수면 값, ``-``·빈칸·그 밖은 null.

    이미 정수 열이면 그대로 쓴다(raw ``istc_totqy`` 는 parquet에서 Int64다).
    """
    c = pl.col(col)
    return c.str.replace_all(r"[\s,]", "").pipe(
        lambda t: pl.when(t.str.contains(r"^-?\d+$")).then(t.cast(pl.Int64, strict=False))
    )


def _qty(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    if df.schema[col] == pl.String:
        return df.with_columns(parse_qty_expr(col).alias(out))
    return df.with_columns(pl.col(col).cast(pl.Int64, strict=False).alias(out))


# ---------------------------------------------------------------- 보고서 단위 표
def dividend_reports(raw_root: Path, snap: str) -> pl.DataFrame:
    """보고서별 DPS. 열: corp_code, report_year, rcept_no, dps, dps_p1, dps_p2, dps_src.

    ``company_maps.dividend_per_report`` 의 NaN은 null로 바꾼다. ``other_only``·``none`` 은
    값이 NaN이라 결측이다.
    """
    d = cm.dividend_per_report(raw_root, snap)
    return d.select(
        "corp_code",
        pl.col("report_year").cast(pl.Int32),
        "rcept_no",
        pl.col("dps_t").fill_nan(None).alias("dps"),
        pl.col("dps_p1").fill_nan(None),
        pl.col("dps_p2").fill_nan(None),
        pl.col("knd_source").alias("dps_src"),
    )


def share_reports(raw_root: Path, snap: str) -> pl.DataFrame:
    """보고서별 보통주 발행주식수. 열: corp_code, report_year, rcept_no, shares, shares_src.

    ``se`` 는 공백을 지워 비교한다. 보통주 행이 있으면 보통주,
    없고 합계 행만 있으면 합계(§5.2, R15).
    같은 (접수번호, se)에 행이 여럿이면 raw_id가 큰 쪽이다.
    """
    df = (
        _scan(raw_root, snap, "dart_share_count_raw")
        .filter(pl.col("reprt_code") == REPRT_ANNUAL)
        .select("raw_id", "corp_code", "bsns_year", "rcept_no", "se", "istc_totqy")
        .collect()
    )
    df = _qty(df, "istc_totqy", "qty")
    df = df.with_columns(pl.col("se").fill_null("").str.replace_all(r"\s", "").alias("se_n"))
    df = df.with_columns(_is_common_se(df["se_n"]).alias("is_common"))
    df = df.filter(pl.col("is_common") | (pl.col("se_n") == SE_TOTAL)).with_columns(
        pl.when(pl.col("is_common")).then(0).otherwise(1).alias("rank")
    )
    # 보통주 행 우선(rank 0), 같은 se 안에서는 raw_id 큰 쪽.
    df = df.sort(
        ["corp_code", "rcept_no", "rank", "raw_id"], descending=[False, False, False, True]
    )
    df = df.unique(subset=["corp_code", "rcept_no"], keep="first", maintain_order=True)
    return df.select(
        "corp_code",
        pl.col("bsns_year").cast(pl.Int32).alias("report_year"),
        "rcept_no",
        pl.col("qty").alias("shares"),
        pl.col("se_n").alias("shares_src"),
    )


def share_se_summary(raw_root: Path, snap: str) -> dict:
    """se 값 종류(11011 행 수), 보통주/합계 선택 건수,
    보통주 값이 비고 합계는 값이 있는 보고서 수."""
    df = (
        _scan(raw_root, snap, "dart_share_count_raw")
        .filter(pl.col("reprt_code") == REPRT_ANNUAL)
        .select("raw_id", "corp_code", "rcept_no", "se", "istc_totqy")
        .collect()
    )
    df = _qty(df, "istc_totqy", "qty").with_columns(
        pl.col("se").fill_null("").str.replace_all(r"\s", "").alias("se_n")
    )
    kinds = {r[0]: r[1] for r in df.group_by("se_n").len().sort("len", descending=True).iter_rows()}
    df = df.with_columns(_is_common_se(df["se_n"]).alias("is_common"))
    per = df.group_by("corp_code", "rcept_no").agg(
        pl.col("is_common").any().alias("has_common"),
        (pl.col("se_n") == SE_TOTAL).any().alias("has_total"),
        pl.col("qty").filter(pl.col("is_common")).max().alias("q_common"),
        pl.col("qty").filter(pl.col("se_n") == SE_TOTAL).max().alias("q_total"),
    )
    return {
        "reports": per.height,
        "se_kinds_rows": kinds,
        "picked_common": int(per["has_common"].sum()),
        "picked_total": int((~per["has_common"] & per["has_total"]).sum()),
        "picked_none": int((~per["has_common"] & ~per["has_total"]).sum()),
        "both_common_and_total": int((per["has_common"] & per["has_total"]).sum()),
        "common_row_value_null_total_has_value": int(
            (per["has_common"] & per["q_common"].is_null() & per["q_total"].is_not_null()).sum()
        ),
    }


def treasury_reports(raw_root: Path, snap: str) -> pl.DataFrame:
    """보고서별 소각 표시. 열: corp_code, report_year, rcept_no, retire, retire_pref_only.

    자사주 표(``statement_type = treasury_stock``, 11011)에서
    ``change_qy_incnr`` > 0 인 행이 하나라도 있으면
    retire = 1, 행은 있는데 모두 0·``-`` 이면 0. 이 표에 행이 없는 보고서는 나오지 않는다(결측).
    retire_pref_only: 소각 > 0 인 행이 있고 그 행이 모두 보통주·표시 없음(``classify_stock_knd`` 의
    common·unmarked)이 아닌 경우(CI-retire-any-class, 입력 존재 개수).
    """
    df = (
        _scan(raw_root, snap, "dart_shareholder_return_raw")
        .filter(
            (pl.col("statement_type") == "treasury_stock") & (pl.col("reprt_code") == REPRT_ANNUAL)
        )
        .select("corp_code", "bsns_year", "rcept_no", "stock_knd", "raw_payload")
        .unique(subset=["corp_code", "rcept_no", "raw_payload"], maintain_order=True)
        .collect()
    )
    df = df.with_columns(
        pl.col("raw_payload").str.json_path_match("$.change_qy_incnr").alias("inc_raw")
    )
    df = parse_incnr(df)
    df = cm._map_unique(df, "stock_knd", cm.classify_stock_knd, "k", pl.String).with_columns(
        pl.col("k").fill_null("unmarked")
    )
    df = df.with_columns(
        (pl.col("inc") > 0).fill_null(False).alias("pos"),
        pl.col("k").is_in(["common", "unmarked"]).alias("is_common"),
    )
    return (
        df.group_by("corp_code", "bsns_year", "rcept_no")
        .agg(
            pl.col("pos").any().alias("any_pos"),
            (pl.col("pos") & pl.col("is_common")).any().alias("pos_common"),
        )
        .select(
            "corp_code",
            pl.col("bsns_year").cast(pl.Int32).alias("report_year"),
            "rcept_no",
            pl.col("any_pos").cast(pl.Int8).alias("retire"),
            (pl.col("any_pos") & ~pl.col("pos_common")).alias("retire_pref_only"),
        )
    )


def parse_incnr(df: pl.DataFrame) -> pl.DataFrame:
    """``inc_raw`` 문자열 → ``inc``(Int64, ``-`` 는 null)."""
    return df.with_columns(parse_qty_expr("inc_raw").alias("inc"))


# ---------------------------------------------------------------- 판본 선택 (§5.1)
def select_versions(
    reports: pl.DataFrame, avail: pl.DataFrame, fys: Sequence[int], lag: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """형성 연도 fy마다 (report_year = fy-lag) 보고서의 판본을 고른다.

    가용일 ≤ B_fy 인 판본 중 (가용일 내림, 접수번호 내림) 첫 행.
    접수 목록에 없는 접수번호는 가용하지
    않다(company_common CI-avail-missing). 반환: (picked, exist). picked는 reports 열 + fy,
    exist는 그 해 보고서가 어떤 판본이든 있는 (corp_code, fy).
    """
    a = avail.select("rcept_no", "avail_date")
    j = reports.join(a, on="rcept_no", how="left")
    picked, exist = [], []
    for fy in fys:
        sub = j.filter(pl.col("report_year") == fy - lag)
        exist.append(
            sub.select("corp_code").unique().with_columns(pl.lit(fy, pl.Int32).alias("fy"))
        )
        ok = sub.filter(
            pl.col("avail_date").is_not_null() & (pl.col("avail_date") <= base_date(fy))
        )
        ok = ok.sort(["avail_date", "rcept_no"], descending=True).unique(
            subset="corp_code", keep="first", maintain_order=True
        )
        picked.append(ok.drop("avail_date").with_columns(pl.lit(fy, pl.Int32).alias("fy")))
    p = (
        pl.concat(picked)
        if picked
        else reports.clear().with_columns(pl.lit(None, pl.Int32).alias("fy"))
    )
    e = pl.concat(exist) if exist else pl.DataFrame(schema={"corp_code": pl.String, "fy": pl.Int32})
    return p, e


# ---------------------------------------------------------------- 사건 해
def _event_flags(
    raw_root: Path, snap: str, sources: tuple[str, ...], suffix: str
) -> dict[str, pl.DataFrame]:
    fl = cm.share_event_flags(raw_root, snap, sources=sources)
    out = {}
    for k, tag in ((0, ""), (1, "_p1"), (2, "_p2")):
        out[f"capevt{suffix}{tag}"] = fl.select(
            "corp_code", (pl.col("year") + k).cast(pl.Int32).alias("fy")
        ).with_columns(pl.lit(True).alias(f"capevt{suffix}{tag}"))
    return out


# ---------------------------------------------------------------- 공개 함수
def build_misc(
    lake: Lake,
    fys: Sequence[int],
    *,
    raw_root: str | Path | None = None,
    avail: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(패널, 진단). 진단 열: corp_code, fy, dps_after_base, shares_after_base, retire_after_base,
    retire_pref_only (입력 존재 표시, coverage용)."""
    fys = [int(y) for y in fys]
    guard_years(fys, "inputs")
    root = _root(lake, raw_root)
    snap = lake.raw_snapshot
    if avail is None:
        avail = filing_availability(lake, TradingCalendar.from_lake(lake))

    dv = dividend_reports(root, snap)
    sh = share_reports(root, snap)
    tr = treasury_reports(root, snap)

    d0, d_ex = select_versions(dv, avail, fys, 0)
    s0, s_ex = select_versions(sh, avail, fys, 0)
    s1, _ = select_versions(sh, avail, fys, 1)
    t0, t_ex = select_versions(tr, avail, fys, 0)
    t1, _ = select_versions(tr, avail, fys, 1)

    # 행 = 그 해(report_year = fy) 사업보고서가 어느 표든 있는 corp-fy (판본 가용 여부와 무관).
    keys = pl.concat([d_ex, s_ex, t_ex]).unique().sort("corp_code", "fy").select("corp_code", "fy")
    panel = (
        keys.join(
            d0.select(
                "corp_code",
                "fy",
                "dps",
                "dps_p1",
                "dps_p2",
                "dps_src",
                pl.col("rcept_no").alias("dps_rcept_no"),
            ),
            on=["corp_code", "fy"],
            how="left",
        )
        .join(
            s0.select("corp_code", "fy", "shares", "shares_src"), on=["corp_code", "fy"], how="left"
        )
        .join(
            s1.select("corp_code", "fy", pl.col("shares").alias("shares_p1")),
            on=["corp_code", "fy"],
            how="left",
        )
        .join(t0.select("corp_code", "fy", "retire"), on=["corp_code", "fy"], how="left")
        .join(
            t1.select("corp_code", "fy", pl.col("retire").alias("retire_p1")),
            on=["corp_code", "fy"],
            how="left",
        )
    )
    flags = {
        **_event_flags(root, snap, cm.JUDGMENT_EVENT_SOURCES, ""),
        **_event_flags(root, snap, cm.RECORD_EVENT_SOURCES, "_rec"),
    }
    for name, f in flags.items():
        panel = panel.join(f, on=["corp_code", "fy"], how="left").with_columns(
            pl.col(name).fill_null(False)
        )
    panel = panel.select(PANEL_COLUMNS)

    def after_base(exist: pl.DataFrame, picked: pl.DataFrame, name: str) -> pl.DataFrame:
        got = picked.select("corp_code", "fy").with_columns(pl.lit(True).alias("_got"))
        return (
            exist.join(got, on=["corp_code", "fy"], how="left")
            .with_columns(pl.col("_got").is_null().alias(name))
            .select("corp_code", "fy", name)
        )

    diag = (
        keys.join(after_base(d_ex, d0, "dps_after_base"), on=["corp_code", "fy"], how="left")
        .join(after_base(s_ex, s0, "shares_after_base"), on=["corp_code", "fy"], how="left")
        .join(after_base(t_ex, t0, "retire_after_base"), on=["corp_code", "fy"], how="left")
        .join(
            tr.join(t0.select("corp_code", "rcept_no", "fy"), on=["corp_code", "rcept_no"]).select(
                "corp_code", "fy", "retire_pref_only"
            ),
            on=["corp_code", "fy"],
            how="left",
        )
        .with_columns(
            pl.col(
                "dps_after_base", "shares_after_base", "retire_after_base", "retire_pref_only"
            ).fill_null(False)
        )
    )
    return panel, diag


def misc_panel(
    lake: Lake, fys: Sequence[int], *, raw_root: str | Path | None = None
) -> pl.DataFrame:
    """배당·주식수·소각·사건 해 패널(열은 ``PANEL_COLUMNS``)."""
    return build_misc(lake, fys, raw_root=raw_root)[0]


def dps_latest(lake: Lake, *, raw_root: str | Path | None = None) -> pl.DataFrame:
    """기준일 없이 가장 늦은 판본의 배당(O3 실현 값용, §5.3).

    열: corp_code, report_year, rcept_no, dps_t, dps_p1, knd_source. (corp, report_year)마다
    접수번호가 가장 큰 보고서(CI-latest-order). NaN은 null.
    """
    d = dividend_reports(_root(lake, raw_root), lake.raw_snapshot)
    d = d.sort(["corp_code", "report_year", "rcept_no"], descending=[False, False, True]).unique(
        subset=["corp_code", "report_year"], keep="first", maintain_order=True
    )
    return d.select(
        "corp_code",
        "report_year",
        "rcept_no",
        pl.col("dps").alias("dps_t"),
        "dps_p1",
        pl.col("dps_src").alias("knd_source"),
    ).sort("corp_code", "report_year")


def coverage_misc(
    lake: Lake,
    fys: Sequence[int],
    *,
    raw_root: str | Path | None = None,
    built: tuple[pl.DataFrame, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """연도별 입력 존재 개수(결과성 숫자는 세지 않는다)."""
    panel, diag = built if built is not None else build_misc(lake, fys, raw_root=raw_root)
    p = panel.join(diag, on=["corp_code", "fy"])
    src = pl.col("dps_src")
    return (
        p.group_by("fy")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("dps").is_null().sum().alias("n_dps_null"),
            (pl.col("dps").is_null().mean()).alias("dps_null_share"),
            (src == "common").sum().alias("dps_src_common"),
            (src == "unmarked").sum().alias("dps_src_unmarked"),
            (src == "other_only").sum().alias("dps_src_other_only"),
            (src == "none").sum().alias("dps_src_none"),
            src.is_null().sum().alias("dps_src_no_version"),
            pl.col("dps_after_base").sum().alias("n_dps_after_base"),
            pl.col("shares_after_base").sum().alias("n_shares_after_base"),
            pl.col("retire_after_base").sum().alias("n_retire_after_base"),
            pl.col("shares").is_not_null().sum().alias("n_shares"),
            pl.col("shares_p1").is_not_null().sum().alias("n_shares_p1"),
            pl.col("retire").is_not_null().sum().alias("n_retire"),
            pl.col("retire_p1").is_not_null().sum().alias("n_retire_p1"),
            (pl.col("shares_src") == SE_COMMON).sum().alias("n_shares_se_common"),
            (pl.col("shares_src") == SE_TOTAL).sum().alias("n_shares_se_total"),
            pl.col("retire_pref_only").sum().alias("n_retire_pref_only"),
            pl.col("capevt").sum().alias("n_capevt"),
            pl.col("capevt_rec").sum().alias("n_capevt_rec"),
        )
        .sort("fy")
    )


# ---------------------------------------------------------------- CLI
def _parse_fys(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _table_info(d: Path) -> dict:
    files = sorted(d.rglob("*.parquet"))
    return {"path": str(d), "files": len(files), "bytes": sum(f.stat().st_size for f in files)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-snapshot", required=True)
    ap.add_argument("--derived-snapshot", required=True)
    ap.add_argument("--fys", required=True, help="예: 2015-2025 또는 2017,2018")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    fys = _parse_fys(args.fys)
    lake = Lake.from_env(raw_snapshot=args.raw_snapshot, derived_snapshot=args.derived_snapshot)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else lake.output_dir(f"quality_score_company_inputs_{args.raw_snapshot}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _root(lake, None)

    built = build_misc(lake, fys)
    panel = built[0]
    cov = coverage_misc(lake, fys, built=built)
    latest = dps_latest(lake)
    se = share_se_summary(root, lake.raw_snapshot)

    outs = {
        "misc_panel.parquet": out_dir / "misc_panel.parquet",
        "dps_latest.parquet": out_dir / "dps_latest.parquet",
        "coverage_misc.tsv": out_dir / "coverage_misc.tsv",
    }
    panel.write_parquet(outs["misc_panel.parquet"])
    latest.write_parquet(outs["dps_latest.parquet"])
    cov.write_csv(outs["coverage_misc.tsv"], separator="\t")

    manifest = {
        "module": "modeler.scores.quality.company_inputs_misc",
        "code_sha256": code_sha256(__file__),
        "maps_code_sha256": code_sha256(cm.__file__),
        "args": {
            "raw_snapshot": args.raw_snapshot,
            "derived_snapshot": args.derived_snapshot,
            "fys": fys,
        },
        "inputs": {
            **{t: _table_info(cm.table_dir(root, lake.raw_snapshot, t)) for t in TABLES_RAW},
            "dim_trading_calendar": _table_info(lake.derived_dir("dim_trading_calendar")),
        },
        "rows": {"misc_panel": panel.height, "dps_latest": latest.height},
        "share_se_summary": se,
        "judgment_event_sources": list(cm.JUDGMENT_EVENT_SOURCES),
        "record_event_sources": list(cm.RECORD_EVENT_SOURCES),
        "ci": {
            "CI-raw-root": CI_RAW_ROOT_FROM_LAKE,
            "CI-se-row": "보통주 행이 있으면 값이 비어도 합계로 안 넘어감",
            "CI-retire-any-class": CI_RETIRE_ANY_CLASS,
            "CI-latest-order": CI_LATEST_ORDER,
            "CI-event-no-base": CI_EVENT_NO_BASE,
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "peak_rss_mb": (
            round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)
            if sys.platform == "darwin"
            else round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        ),
    }
    mp = out_dir / "manifest_misc.json"
    manifest["outputs"] = {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in outs.items()}
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(cov)
    print(json.dumps({"out_dir": str(out_dir), "se": se}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
