"""회사 점수(부분 C) 재무 입력 패널 W2a (사전등록 20261010_quality_score §5.1·§5.2·§7.2).

한 행 = (회사 ``corp_code``, 형성 사업연도 ``fy``). 분모, 판본 후보, 판본 선택, 값 읽기, 기록용 (a),
"가장 늦게 알려진 판본"(결과 변수용), 입력 존재 개수(coverage)를 만든다.
점수·결과 사건은 다루지 않는다. coverage는 값이 있는지(non-null)만 센다(§7.2).

규칙 요약 (모두 §5.1)
- 분모: ``stock_master.market ∈ {KOSPI, KOSDAQ}``(상장폐지 포함). SPAC(이름에 ``스팩``·
  ``기업인수목적``)·금융업(``induty_code`` 앞 두 자리 64\\~66, 현재값)·비12월 결산은 표시만 하고
  ``in_universe_attr`` 에서 뺀다.
- 판본 후보 = vintage 연간(fs_basis CFS/OFS) ∪ raw 연간. 키는 (corp, fy, fs_div, rcept_no).
- 가용일 = 접수 목록 ``rcept_dt`` 의 다음 거래일 ≤ B_t = (t+1)-06-30 (company_common).
- fs_div: 고른 범위 안에서 가장 최근 CFS 판본에 총자산이 있으면 CFS, 아니면 OFS. 그 fs_div에서
  (avail_date, rcept_no) 가장 최근 판본 하나.
- 층: 한 접수번호의 값은 그 접수번호에서만 읽는다. vintage → raw → XBRL 순(CI-layer-fill).
- 전기·전전기 값: 고른 접수번호의 raw ``frmtrm_amount``·``bfefrmtrm_amount``, raw에 그 접수번호가
  없으면 같은 접수번호의 XBRL P·BP(CI-a).

사전등록이 정하지 않은 선택은 상수로 두고 ``# CI-<이름>`` 주석을 달았다.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from modeler.scores.quality.company_common import (
    Lake,
    TradingCalendar,
    code_sha256,
    filing_availability,
    guard_years,
    sha256_file,
)
from modeler.scores.quality.company_xbrl import XBRL_METRICS, xbrl_values

# ---------------------------------------------------------------- 분모 상수 (§5.1)
MARKETS = ("KOSPI", "KOSDAQ")
SPAC_PATTERNS = ("스팩", "기업인수목적")
FINANCIAL_KSIC2 = ("64", "65", "66")
DEC_CLOSING = 12
# CI-dup-ticker: stock_master에 같은 종목코드가 KOSPI·KOSDAQ 두 시장에 있으면(시장 이전, 16개)
# ACTIVE 행을, 없으면 last_seen_date가 늦은 행을 쓴다. SPAC 여부는 두 행의 이름 중 하나라도
# 걸리면 참이다(상폐 행의 이름이 종목코드로 바뀐 경우가 있다).
CI_DUP_TICKER_RULE = "active_then_latest_seen"  # CI-dup-ticker
# CI-acc-null: acc_mt가 없으면 12월 결산으로 확인되지 않으므로 in_universe_attr에서 뺀다.
CI_ACC_NULL_IS_DEC = False  # CI-acc-null

# ---------------------------------------------------------------- 지표 (열 계약, 05 §5)
CUR = ["ta", "tl", "te", "ca", "cl", "re", "oi", "ni", "ocf", "rev", "gp", "ltb"]
IP_PARTS = ["ip_op", "ip_fin", "ip_inv"]  # §5.2 지급이자 세 계정, 이 순서
ALL_CUR = CUR + IP_PARTS + ["cap"]
VINT_NAME = {
    "ta": "total_assets",
    "tl": "total_liabilities",
    "te": "total_equity",
    "ca": "current_assets",
    "cl": "current_liabilities",
    "re": "retained_earnings",
    "oi": "operating_income",
    "ni": "net_income",
    "ocf": "operating_cash_flow",
    "rev": "revenue",
    "gp": "gross_profit",
    "ltb": "borrowings_long_term",
    "ip_op": "interest_paid",
}
VINT_METRICS = list(VINT_NAME)  # 13개
PRIOR1 = ["ta", "te", "ni", "ocf", "ca", "cl", "rev", "gp", "ltb"]
PRIOR2 = ["ta", "te", "ni"]
PRIOR_COLS = [f"{m}_p1" for m in PRIOR1] + [f"{m}_p2" for m in PRIOR2]
PRIOR_ORDER = [
    "ta_p1", "ta_p2", "te_p1", "te_p2", "ni_p1", "ni_p2", "ocf_p1", "ca_p1", "cl_p1",
    "rev_p1", "gp_p1", "ltb_p1",
]  # fmt: skip
assert sorted(PRIOR_ORDER) == sorted(PRIOR_COLS)

# raw FS 계정 매핑: 지표 → (계정 우선순위(소문자·접두어 정규화), 허용 sj_div).
# vintage ``fin.*`` 규칙(mapping_rule_code)과 같은 계정이고 oi는 §5.2 X3의 두 태그 순서다.
# CI-raw-sj: 손익 항목(oi·ni·rev·gp)의 sj_div는 IS·CIS 둘 다 읽는다. vintage fin 규칙은 ni만
# CIS(+드문 IS)이고 oi·rev·gp는 IS뿐이며 나머지는 XBRL(xbrlfb)에서 읽는다 — raw에는 같은 계정이
# CIS에 적힌 회사가 많아서 둘 다 열어야 vintage·XBRL과 같은 값이 나온다(층 불일치 건수가 확인).
_IS = ("IS", "CIS")
RAW_MAP: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ta": (("ifrs-full_assets",), ("BS",)),
    "tl": (("ifrs-full_liabilities",), ("BS",)),
    "te": (("ifrs-full_equity",), ("BS",)),
    "ca": (("ifrs-full_currentassets",), ("BS",)),
    "cl": (("ifrs-full_currentliabilities",), ("BS",)),
    "re": (("ifrs-full_retainedearnings",), ("BS",)),
    "oi": (("dart_operatingincomeloss", "ifrs-full_profitlossfromoperatingactivities"), _IS),
    "ni": (("ifrs-full_profitloss",), _IS),
    "ocf": (("ifrs-full_cashflowsfromusedinoperatingactivities",), ("CF",)),
    "rev": (("ifrs-full_revenue",), _IS),
    "gp": (("ifrs-full_grossprofit",), _IS),
    "ltb": (("dart_longtermborrowingsgross",), ("BS",)),
    "ip_op": (("ifrs-full_interestpaidclassifiedasoperatingactivities",), ("CF",)),
    "ip_fin": (("ifrs-full_interestpaidclassifiedasfinancingactivities",), ("CF",)),
    "ip_inv": (("ifrs-full_interestpaidclassifiedasinvestingactivities",), ("CF",)),
    "cap": (("ifrs-full_issuedcapital",), ("BS",)),  # CI-b: vintage에 없다. 한글명 보충 안 함(§5.1)
}
_NC = "lower(regexp_replace(account_id,'^ifrs[-_](full_)?','ifrs-full_','i'))"  # §5.1 접두어

# CI-layer-fill: 고른 접수번호가 vintage에 있는데 특정 지표만 vintage에 없을 때 같은 접수번호의
# raw → XBRL로 채운다(기본). 같은 접수번호라 층을 섞어도 값이 같다(07 §1.4, W2c V1).
CI_LAYER_FILL = True  # CI-layer-fill
# CI-a: raw에 그 접수번호가 없으면 같은 접수번호의 XBRL P·BP로 전기·전전기를 읽는다(기본).
CI_USE_XBRL_PRIOR = True  # CI-a
# CI-prior-cut: 기록용 (a)에서 t−1·t−2 보고서의 판본은 형성 연도 t의 기준일 B_t 이전에 가용한
# 것 중에서 고른다("같은 B_t 규칙"). B_{t-1}로 자르지 않는다.
CI_PRIOR_CUT = "b_t"  # CI-prior-cut
# CI-fsdiv-nofallback: 가장 최근 CFS 판본에 총자산이 없는데 범위 안에 OFS 판본이 없으면 CFS를
# 그대로 쓴다(다른 지표 값이 있을 수 있다). 사전등록은 이 경우를 정하지 않는다.
CI_FSDIV_NOFALLBACK = "keep_cfs"  # CI-fsdiv-nofallback
# CI-ip-abs: 지급이자는 현금흐름표 부호와 상관없이 절대값으로 낸다.
CI_IP_ABS = True  # CI-ip-abs
# CI-prior-src: prior_src는 고른 접수번호가 raw에 있으면 'raw'(비교 칸이 비어도 'raw'), 없고
# XBRL P·BP가 하나라도 있으면 'xbrl', 아니면 'none'.
CI_MISMATCH_TOL = 0.5  # 층 불일치 허용 오차(원)
# CI-xbrl-join: XBRL 캐시는 (접수번호, fs_div, bsns_year)로 맞춘다(접수번호 하나가 사업연도 둘).
# CI-latest-null-avail: as_of="latest"에서 가용일이 없는 후보(접수 목록에 없음)는 순서 맨 뒤.

OUT_PANEL_COLS = (
    ["corp_code", "fy", "stock_code", "market", "in_universe", "is_spac", "is_financial",
     "acc_mt", "fs_div", "rcept_no", "layer", "avail_date", "currency"]
    + CUR + ["ip", "ip_src", "cap"] + PRIOR_ORDER + ["prior_src"]
)  # fmt: skip
VALUE_COLS = CUR + ["ip", "cap"] + PRIOR_ORDER


# ---------------------------------------------------------------- 분모 (§5.1)
def universe(lake: Lake) -> pl.DataFrame:
    """분모 표 [corp_code, stock_code, market, is_spac, is_financial, acc_mt, in_universe_attr].

    행 = ``stock_master`` 의 KOSPI·KOSDAQ 종목코드 하나당 한 행(상장폐지 포함). 종목코드 ↔ corp_code
    는 ``dart_corp_master.ticker`` 로 잇고, 못 이은 종목은 corp_code가 null이다.
    ``in_universe_attr`` = SPAC 아님 & 금융업 아님 & 12월 결산(그 fy에 판본이 있는지는 패널이 본다).
    """
    sm = (
        pl.scan_parquet(lake.raw_glob("stock_master"))
        .filter(pl.col("market").is_in(MARKETS))
        .select("ticker", "market", "name", "status", "last_seen_date")
        .collect()
    )
    spac = pl.any_horizontal([pl.col("name").str.contains(p, literal=True) for p in SPAC_PATTERNS])
    spac_by_ticker = (
        sm.with_columns(spac.fill_null(False).alias("_s"))
        .group_by("ticker")
        .agg(pl.col("_s").any().alias("is_spac"))
    )
    sm = (
        sm.sort(
            ["ticker", "status", "last_seen_date"],
            descending=[False, False, True],  # ACTIVE < DELISTED 사전순, 최근 확인일 먼저
            nulls_last=True,
        )
        .unique(subset="ticker", keep="first", maintain_order=True)
        .join(spac_by_ticker, on="ticker", how="left")
    )
    cm = (
        pl.scan_parquet(lake.raw_glob("dart_corp_master"))
        .filter(pl.col("ticker").is_not_null())
        .select("corp_code", "ticker", "induty_code", "acc_mt")
        .unique(subset="ticker", keep="first", maintain_order=True)
        .collect()
    )
    u = sm.join(cm, on="ticker", how="left")
    acc = pl.col("acc_mt").cast(pl.Int32, strict=False)
    fin = pl.col("induty_code").str.slice(0, 2).is_in(FINANCIAL_KSIC2).fill_null(False)
    is_dec = (acc == DEC_CLOSING).fill_null(CI_ACC_NULL_IS_DEC)
    return u.select(
        pl.col("corp_code"),
        pl.col("ticker").alias("stock_code"),
        pl.col("market"),
        pl.col("is_spac").fill_null(False),
        fin.alias("is_financial"),
        acc.alias("acc_mt"),
        (
            pl.col("market").is_in(MARKETS) & ~pl.col("is_spac").fill_null(False) & ~fin & is_dec
        ).alias("in_universe_attr"),
        pl.col("status"),
    ).drop("status")


def universe_report(lake: Lake) -> dict[str, object]:
    """분모 매핑 결과와 코넥스·기타법인 제외 확인 (§12.4 C "분모 규칙 확인", 입력 존재 개수)."""
    sm_all = (
        pl.scan_parquet(lake.raw_glob("stock_master"))
        .select("ticker", "market", "status", "name", "listing_date")
        .collect()
    )
    sm = sm_all.filter(pl.col("market").is_in(MARKETS))
    cm = (
        pl.scan_parquet(lake.raw_glob("dart_corp_master"))
        .filter(pl.col("ticker").is_not_null())
        .select("corp_code", "ticker", "corp_cls")
        .collect()
    )
    u = universe(lake)
    mapped = u.filter(pl.col("corp_code").is_not_null())
    unmapped = u.filter(pl.col("corp_code").is_null())
    st = sm.unique(subset="ticker", keep="first").select("ticker", "status")
    unm_status = unmapped.join(
        sm.sort("status").unique(subset="ticker", keep="first").select("ticker", "status"),
        left_on="stock_code",
        right_on="ticker",
        how="left",
    )
    cls = mapped.join(cm.select("corp_code", "corp_cls"), on="corp_code", how="left")
    # raw 재무(11011)에 있는 corp 중 분모 밖
    con = _con("4GB")
    try:
        fs_corps = con.execute(
            f"select distinct corp_code from read_parquet("
            f"'{lake.raw_glob('dart_financial_statement_raw')}', hive_partitioning=true) "
            f"where reprt_code=11011"
        ).pl()
    finally:
        con.close()
    cm_all = (
        pl.scan_parquet(lake.raw_glob("dart_corp_master"))
        .select("corp_code", "corp_cls")
        .unique(subset="corp_code")
        .collect()
    )
    outside = fs_corps.join(mapped.select("corp_code"), on="corp_code", how="anti").join(
        cm_all, on="corp_code", how="left"
    )
    inside_cls = cls.group_by("corp_cls").len().sort("corp_cls", nulls_last=True).rows(named=True)
    return {
        "stock_master_rows_all": sm_all.height,
        "stock_master_tickers_kospi_kosdaq": u.height,
        "stock_master_dup_tickers_across_markets": int(sm.height - sm["ticker"].n_unique()),
        "stock_master_other_markets": sm_all.filter(~pl.col("market").is_in(MARKETS))
        .group_by("market")
        .len()
        .rows(named=True),
        "unmapped_preferred_like_ticker": int(
            unmapped.filter(~pl.col("stock_code").str.ends_with("0")).height
        ),
        "unmapped_listed_2026": int(
            unmapped.join(
                sm_all.filter(pl.col("market").is_in(MARKETS))
                .sort("status")
                .unique(subset="ticker", keep="first")
                .select("ticker", "listing_date"),
                left_on="stock_code",
                right_on="ticker",
                how="left",
            )
            .filter(pl.col("listing_date").dt.year() >= 2026)
            .height
        ),
        "mapped_to_corp_code": mapped.height,
        "unmapped_to_corp_code": unmapped.height,
        "unmapped_by_status": unm_status.group_by("status").len().rows(named=True),
        "mapped_by_status": mapped.join(st, left_on="stock_code", right_on="ticker", how="left")
        .group_by("status")
        .len()
        .rows(named=True),
        "spac_tickers": int(u["is_spac"].sum()),
        "financial_corps": int(mapped["is_financial"].sum()),
        "non_december_corps": int((mapped["acc_mt"] != DEC_CLOSING).fill_null(True).sum()),
        "in_universe_attr_corps": int(mapped["in_universe_attr"].sum()),
        "universe_corps_by_corp_cls": inside_cls,
        "raw_fs_annual_corps": fs_corps.height,
        "raw_fs_corps_outside_denominator": outside.height,
        "raw_fs_corps_outside_by_corp_cls": outside.group_by("corp_cls")
        .len()
        .sort("corp_cls", nulls_last=True)
        .rows(named=True),
        "denominator_corps_with_cls_N": int(cls.filter(pl.col("corp_cls") == "N").height),
        "denominator_corps_with_cls_E": int(cls.filter(pl.col("corp_cls") == "E").height),
    }


# ---------------------------------------------------------------- 후보·값 읽기 (duckdb)
def _con(memory_limit: str = "6GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"set memory_limit='{memory_limit}'")
    return con


def _raw_agg_sql() -> str:
    """raw FS를 (corp, fy, fs_div, rcept_no) 한 행으로 접는 조건부 집계 식."""
    parts = []
    for m, (accts, sjs) in RAW_MAP.items():
        sj = ",".join(f"'{s}'" for s in sjs)

        def agg(col: str, accts=accts, sj=sj) -> str:
            terms = [
                f"max(case when {_NC}='{a}' and sj_div in ({sj}) then {col} end)" for a in accts
            ]
            return "cast(coalesce(" + ",".join(terms) + ") as double)"

        parts.append(f"{agg('thstrm_amount')} as r_{m}")
        if m in PRIOR1:
            parts.append(f"{agg('frmtrm_amount')} as rp1_{m}")
        if m in PRIOR2:
            parts.append(f"{agg('bfefrmtrm_amount')} as rp2_{m}")
    return ",\n ".join(parts)


def load_candidates(
    lake: Lake, corp_codes: Sequence[str], lo: int, hi: int, *, memory_limit: str = "6GB"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(vintage 넓은 표, raw 넓은 표). 키 (corp_code, fy, fs_div, rcept_no), fy ∈ [lo, hi].

    vintage: 연간(reprt_code 11011, period_type annual)·fs_basis CFS/OFS.
    ``v_<지표>``·``v_currency``.
    raw: reprt_code 11011·fs_div CFS/OFS. ``r_<지표>``(당기)·``rp1_``·``rp2_``(전기·전전기)·
    ``r_currency``. 한 접수번호·계정에 같은 sj_div의 서로 다른 값은 없다(max 집계가 안전, 확인함).
    필요한 계정·열만 읽는다.
    """
    con = _con(memory_limit)
    try:
        con.register("ucorp_df", pl.DataFrame({"corp_code": list(corp_codes)}).to_arrow())
        con.execute("create temp table ucorp as select * from ucorp_df")
        vg = lake.derived_glob("stock_metric_vintage_fact")
        vm = ",".join(f"'{v}'" for v in VINT_NAME.values())
        vparts = ",\n ".join(
            f"cast(max(case when metric_code='{v}' then value_numeric end) as double) as v_{m}"
            for m, v in VINT_NAME.items()
        )
        v = con.execute(f"""
            select corp_code, cast(bsns_year as int) as fy, fs_basis as fs_div, rcept_no,
                   max(currency) as v_currency, {vparts}
            from read_parquet('{vg}', hive_partitioning=true)
            where reprt_code='11011' and period_type='annual' and fs_basis in ('CFS','OFS')
              and bsns_year between {lo} and {hi}
              and corp_code in (select corp_code from ucorp)
              and (metric_code in ({vm}) or true)
            group by 1,2,3,4
            """).pl()
        fg = lake.raw_glob("dart_financial_statement_raw")
        r = con.execute(f"""
            select corp_code, cast(bsns_year as int) as fy, fs_div, rcept_no,
                   max(currency) as r_currency, {_raw_agg_sql()}
            from read_parquet('{fg}', hive_partitioning=true)
            where reprt_code=11011 and fs_div in ('CFS','OFS')
              and bsns_year between {lo} and {hi}
              and corp_code in (select corp_code from ucorp)
            group by 1,2,3,4
            """).pl()
    finally:
        con.close()
    return v, r


def xbrl_wide(
    lake: Lake,
    rcept_nos: Sequence[str],
    cache: str | Path | None = None,
    *,
    memory_limit: str = "6GB",
) -> pl.DataFrame:
    """접수번호별 XBRL 넓은 표 [rcept_no, fy, fs_div, x_<지표>(C), xp1_<지표>(P), xp2_<지표>(BP)].

    ``cache`` (W2c가 만든 xbrl_values.parquet)가 있으면 읽고, 없으면 ``company_xbrl.xbrl_values``
    로 만든다(느리다). 필요한 접수번호만 남긴다.
    """
    rc = list(dict.fromkeys(rcept_nos))
    path = Path(cache) if cache else _default_cache(lake)
    if path.exists():
        x = (
            pl.scan_parquet(path)
            .filter(pl.col("rcept_no").is_in(rc))
            .select("rcept_no", "bsns_year", "fs_div", "metric", "period", "value")
            .collect()
        )
    else:
        raw = Path(lake.raw_dir("dart_xbrl_fact_raw"))
        xglob = f"{raw}/**/*.parquet"
        con = _con(memory_limit)
        try:
            x = xbrl_values(xglob, rc, con=con).select(
                "rcept_no", "bsns_year", "fs_div", "metric", "period", "value"
            )
        finally:
            con.close()
    pre = {"C": "x_", "P": "xp1_", "BP": "xp2_"}
    x = x.with_columns(
        (pl.col("period").replace_strict(pre, default=None) + pl.col("metric")).alias("_k")
    ).filter(pl.col("_k").is_not_null())
    w = x.pivot(
        on="_k",
        index=["rcept_no", "bsns_year", "fs_div"],
        values="value",
        aggregate_function="first",
    ).rename({"bsns_year": "fy"})
    return w.with_columns(pl.col("fy").cast(pl.Int32))


def _default_cache(lake: Lake) -> Path:
    return (
        lake.output_dir(f"quality_score_company_cache_{lake.raw_snapshot}") / "xbrl_values.parquet"
    )


# ---------------------------------------------------------------- 입력 묶음
@dataclass
class FsInputs:
    """읽기 단계 결과. 판본·값 규칙은 ``resolve`` 이후 순수 함수가 적용한다."""

    lake: Lake
    fys: list[int]
    universe: pl.DataFrame
    cv: pl.DataFrame  # 후보 행 + v_·r_·rp*_·x_·xp*_ 열 + 가용일 등
    receipts_total: int

    def with_cv(self, cv: pl.DataFrame) -> FsInputs:
        return FsInputs(self.lake, self.fys, self.universe, cv, self.receipts_total)


def build_candidate_frame(
    v: pl.DataFrame,
    r: pl.DataFrame,
    x: pl.DataFrame,
    avail: pl.DataFrame,
    names: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """vintage ∪ raw 후보에 XBRL·가용일을 붙인다 (합성 자료 시험용으로 분리).

    avail: [rcept_no, avail_date]. names: [rcept_no, report_nm] (없으면 정정 여부는 null).
    열: corp_code, fy, fs_div, rcept_no, in_vintage, in_raw, in_xbrl, in_receipt, avail_date,
    is_revision 와 v_·r_·rp*_·x_·xp*_ 열.
    """
    key = ["corp_code", "fy", "fs_div", "rcept_no"]
    v = v.with_columns(pl.lit(True).alias("in_vintage"))
    r = r.with_columns(pl.lit(True).alias("in_raw"))
    c = v.join(r, on=key, how="full", coalesce=True)
    c = c.with_columns(
        pl.col("in_vintage").fill_null(False),
        pl.col("in_raw").fill_null(False),
    )
    x = x.with_columns(pl.lit(True).alias("in_xbrl"))
    c = c.join(x, on=["rcept_no", "fs_div", "fy"], how="left").with_columns(
        pl.col("in_xbrl").fill_null(False)
    )
    c = c.join(
        avail.select("rcept_no", "avail_date").with_columns(pl.lit(True).alias("in_receipt")),
        on="rcept_no",
        how="left",
    ).with_columns(pl.col("in_receipt").fill_null(False))
    if names is not None:
        c = c.join(
            names.select(
                "rcept_no",
                pl.col("report_nm").str.contains("정정", literal=True).alias("is_revision"),
            ),
            on="rcept_no",
            how="left",
        )
    else:
        c = c.with_columns(pl.lit(None, dtype=pl.Boolean).alias("is_revision"))
    # 열 보장: 없는 v_·r_·x_ 열은 null로 만든다(합성 자료·빈 층 대비)
    need = (
        [f"v_{m}" for m in VINT_METRICS]
        + ["v_currency", "r_currency"]
        + [f"r_{m}" for m in ALL_CUR]
        + [f"rp1_{m}" for m in PRIOR1]
        + [f"rp2_{m}" for m in PRIOR2]
        + [f"x_{m}" for m in ALL_CUR]
        + [f"xp1_{m}" for m in PRIOR1]
        + [f"xp2_{m}" for m in PRIOR2]
    )
    add = [
        pl.lit(None, dtype=pl.String if n.endswith("currency") else pl.Float64).alias(n)
        for n in need
        if n not in c.columns
    ]
    return c.with_columns(add) if add else c


def load_inputs(
    lake: Lake,
    fys: Sequence[int],
    *,
    xbrl_cache: str | Path | None = None,
    memory_limit: str = "6GB",
) -> FsInputs:
    """분모·vintage·raw·XBRL·가용일을 읽어 후보 표를 만든다. fy 범위는 (min−2, max)."""
    guard_years(fys, "inputs")
    fys = sorted(set(int(y) for y in fys))
    u = universe(lake)
    corps = u.filter(pl.col("corp_code").is_not_null())["corp_code"].to_list()
    lo, hi = fys[0] - 2, fys[-1]
    v, r = load_candidates(lake, corps, lo, hi, memory_limit=memory_limit)
    rcepts = pl.concat([v["rcept_no"], r["rcept_no"]]).unique().to_list()
    x = xbrl_wide(lake, rcepts, xbrl_cache, memory_limit=memory_limit)
    cal = TradingCalendar.from_lake(lake)
    avail_all = filing_availability(lake, cal)
    avail = avail_all.filter(pl.col("rcept_no").is_in(rcepts))
    names = (
        pl.scan_parquet(lake.raw_glob("dart_filing_receipt_raw"))
        .filter(pl.col("rcept_no").is_in(rcepts))
        .select("rcept_no", "report_nm")
        .unique(subset="rcept_no", keep="first")
        .collect()
    )
    cv = build_candidate_frame(v, r, x, avail, names)
    return FsInputs(lake, fys, u, cv, avail_all.height)


# ---------------------------------------------------------------- 값 읽기 규칙 (§5.1 층·전년)
def resolve_values(
    cv: pl.DataFrame,
    *,
    layer_fill: bool = CI_LAYER_FILL,
    use_xbrl_prior: bool = CI_USE_XBRL_PRIOR,
) -> pl.DataFrame:
    """후보 행마다 당기·지급이자·자본금·전기·전전기 값과 출처를 정한다.

    당기 지표: vintage에 값이 있으면 vintage, 없으면(layer_fill) 같은 접수번호의 raw, 없으면 XBRL C.
    vintage에 없는 접수번호는 raw → XBRL. vintage에 없는 지표(ip_fin·ip_inv·cap)는 항상 raw → XBRL.
    ``src_<지표>`` ∈ vintage/raw/xbrl/null. 지급이자 ``ip`` = ip_op → ip_fin → ip_inv의 첫 값의
    절대값(CI-ip-abs), ``ip_src`` ∈ op/fin/inv/none.
    전기·전전기: 접수번호가 raw에 있으면 raw의 frmtrm·bfefrmtrm, 없으면(use_xbrl_prior) XBRL P·BP.
    """
    exprs: list[pl.Expr] = []
    for m in ALL_CUR:
        v = pl.col(f"v_{m}") if m in VINT_NAME else pl.lit(None, dtype=pl.Float64)
        r, x = pl.col(f"r_{m}"), pl.col(f"x_{m}")
        if layer_fill or m not in VINT_NAME:
            val = pl.coalesce(v, r, x)
            src = (
                pl.when(v.is_not_null())
                .then(pl.lit("vintage"))
                .when(r.is_not_null())
                .then(pl.lit("raw"))
                .when(x.is_not_null())
                .then(pl.lit("xbrl"))
                .otherwise(pl.lit(None, dtype=pl.String))
            )
        else:
            # vintage 접수번호는 vintage 값만. raw 접수번호만 raw → XBRL
            val = pl.when(pl.col("in_vintage")).then(v).otherwise(pl.coalesce(r, x))
            src = (
                pl.when(pl.col("in_vintage"))
                .then(pl.when(v.is_not_null()).then(pl.lit("vintage")))
                .when(r.is_not_null())
                .then(pl.lit("raw"))
                .when(x.is_not_null())
                .then(pl.lit("xbrl"))
                .otherwise(pl.lit(None, dtype=pl.String))
            )
        exprs += [val.alias(m), src.alias(f"src_{m}")]
    df = cv.with_columns(exprs)

    first = pl.coalesce(*[pl.col(p) for p in IP_PARTS])
    ip = first.abs() if CI_IP_ABS else first
    ip_src = (
        pl.when(pl.col("ip_op").is_not_null())
        .then(pl.lit("op"))
        .when(pl.col("ip_fin").is_not_null())
        .then(pl.lit("fin"))
        .when(pl.col("ip_inv").is_not_null())
        .then(pl.lit("inv"))
        .otherwise(pl.lit("none"))
    )
    cur = pl.coalesce(pl.col("v_currency"), pl.col("r_currency"))
    df = df.with_columns(
        ip.alias("ip"), first.alias("ip_signed"), ip_src.alias("ip_src"), cur.alias("currency")
    )

    pri = []
    for m in PRIOR1:
        pri.append(_prior_expr(m, 1, use_xbrl_prior).alias(f"{m}_p1"))
    for m in PRIOR2:
        pri.append(_prior_expr(m, 2, use_xbrl_prior).alias(f"{m}_p2"))
    x_any = pl.any_horizontal(
        [pl.col(f"xp{k}_{m}").is_not_null() for k, ms in ((1, PRIOR1), (2, PRIOR2)) for m in ms]
    )
    psrc = (
        pl.when(pl.col("in_raw"))
        .then(pl.lit("raw"))
        .when(pl.lit(use_xbrl_prior) & x_any)
        .then(pl.lit("xbrl"))
        .otherwise(pl.lit("none"))
    )
    return df.with_columns(pri + [psrc.alias("prior_src")])


def _prior_expr(m: str, k: int, use_xbrl: bool) -> pl.Expr:
    r, x = pl.col(f"rp{k}_{m}"), pl.col(f"xp{k}_{m}")
    if use_xbrl:
        return pl.when(pl.col("in_raw")).then(r).otherwise(x)
    return pl.when(pl.col("in_raw")).then(r).otherwise(pl.lit(None, dtype=pl.Float64))


# ---------------------------------------------------------------- 판본 선택 (§5.1)
def pick_versions(cv: pl.DataFrame, *, shift: int = 0, latest: bool = False) -> pl.DataFrame:
    """보고서 사업연도 ``fy`` 의 판본을 (corp, 형성연도 t = fy + shift)마다 하나 고른다.

    - latest=False: avail_date ≤ B_t = base_date(t)인 후보만(avail_date가 없으면 가용 아님).
      shift=0이면 t = fy(주 판본), shift=1·2는 기록용 (a)의 t−1·t−2 보고서
      (B_t로 자른다, CI-prior-cut).
    - latest=True: 기준일 없이 모든 후보. (avail_date 내림, rcept_no 내림)에서
      가용일 없는 후보는 뒤.
    - fs_div: 그 범위의 가장 최근 CFS 후보에 총자산(ta)이 있으면 CFS, 아니면 OFS. OFS 후보가 없으면
      CFS를 그대로 쓴다(CI-fsdiv-nofallback). 그 fs_div의 첫 판본 하나.
    반환: 입력 열 전부 + t, n_avail(그 범위 후보 수). 판본이 없는 (corp, t)는 행이 없다.
    """
    df = cv.filter(pl.col("fs_div").is_in(["CFS", "OFS"])).with_columns(
        (pl.col("fy") + shift).alias("t")
    )
    if not latest:
        cut = pl.date(pl.col("t") + 1, 6, 30)
        df = df.filter(pl.col("avail_date").is_not_null() & (pl.col("avail_date") <= cut))
    n_av = df.group_by(["corp_code", "t"]).len().rename({"len": "n_avail"})
    df = df.sort(
        ["corp_code", "t", "fs_div", "avail_date", "rcept_no"],
        descending=[False, False, False, True, True],
        nulls_last=True,
    )
    best = df.group_by(["corp_code", "t", "fs_div"], maintain_order=True).first()
    g = ["corp_code", "t"]
    best = best.with_columns(
        (pl.col("fs_div") == "OFS").any().over(g).alias("_has_ofs"),
        ((pl.col("fs_div") == "CFS") & pl.col("ta").is_not_null()).any().over(g).alias("_cfs_ta"),
    )
    keep = ((pl.col("fs_div") == "CFS") & (pl.col("_cfs_ta") | ~pl.col("_has_ofs"))) | (
        (pl.col("fs_div") == "OFS") & ~pl.col("_cfs_ta")
    )
    out = best.filter(keep).drop("_has_ofs", "_cfs_ta")
    return out.join(n_av, on=g, how="left")


# ---------------------------------------------------------------- 패널
def _keys(cv: pl.DataFrame, fys: Sequence[int]) -> pl.DataFrame:
    """(corp, fy)마다 후보 수·vintage 후보 수·B_t 이전 가용 후보 수·정정본만 여부."""
    b = pl.date(pl.col("fy") + 1, 6, 30)
    ok = pl.col("avail_date").is_not_null() & (pl.col("avail_date") <= b)
    return (
        cv.filter(pl.col("fy").is_in(list(fys)))
        .group_by(["corp_code", "fy"])
        .agg(
            pl.len().alias("k_n_cand"),
            pl.col("in_vintage").sum().cast(pl.Int64).alias("k_n_vint"),
            ok.sum().cast(pl.Int64).alias("k_n_avail"),
            (~pl.col("in_receipt")).sum().cast(pl.Int64).alias("k_n_no_receipt"),
            pl.col("is_revision").fill_null(False).all().alias("k_all_revision"),
        )
    )


def _raw_only(cv: pl.DataFrame) -> pl.DataFrame:
    """raw 후보만 남기고 vintage·XBRL 값 열(통화 포함)을 비운다(§5.1 raw 한 층 단독 민감도)."""
    cols = [c for c in cv.columns if c.startswith(("v_", "x_", "xp1_", "xp2_"))]
    return cv.filter(pl.col("in_raw")).with_columns(
        *[pl.lit(None, dtype=cv.schema[c]).alias(c) for c in cols],
        pl.lit(False).alias("in_vintage"),
        pl.lit(False).alias("in_xbrl"),
    )


def _meta(u: pl.DataFrame) -> pl.DataFrame:
    return u.filter(pl.col("corp_code").is_not_null()).select(
        "corp_code", "stock_code", "market", "is_spac", "is_financial", "acc_mt", "in_universe_attr"
    )


def build_panel(
    inp: FsInputs,
    fys: Sequence[int] | None = None,
    *,
    as_of: str = "base",
    prior_source: str = "frmtrm",
    layer_fill: bool = CI_LAYER_FILL,
    use_xbrl_prior: bool = CI_USE_XBRL_PRIOR,
    layers: str = "all",
    cap_xbrl: bool = True,
    diag: bool = False,
    src: bool = False,
) -> pl.DataFrame:
    """패널을 만든다. ``fs_panel`` 의 본체(읽기 단계 결과 재사용).

    ``layers="raw_only"`` 는 §5.1 raw 한 층 단독 민감도다: raw 후보(접수번호)만 두고 값·전기 값·
    자본금·지급이자·통화를 raw에서만 읽는다(vintage·XBRL 안 씀, 가용일 규칙은 같다).
    ``cap_xbrl=False`` 는 자본금을 XBRL로 대체하지 않는다(CI14 문면판, 기록용).
    기본값(``"all"``·True)은 출력이 바뀌지 않는다.
    """
    if layers not in ("all", "raw_only"):
        raise ValueError(f"layers는 'all'|'raw_only': {layers!r}")
    if as_of not in ("base", "latest"):
        raise ValueError(f"as_of는 'base'|'latest': {as_of!r}")
    if prior_source not in ("frmtrm", "t_minus_1_report"):
        raise ValueError(f"prior_source는 'frmtrm'|'t_minus_1_report': {prior_source!r}")
    fys = sorted(set(inp.fys if fys is None else fys))
    guard_years(fys, "inputs")
    cv_in = inp.cv
    if layers == "raw_only":
        cv_in = _raw_only(cv_in)
    if not cap_xbrl:
        cv_in = cv_in.with_columns(pl.lit(None, dtype=pl.Float64).alias("x_cap"))
    cv = resolve_values(cv_in, layer_fill=layer_fill, use_xbrl_prior=use_xbrl_prior)
    latest = as_of == "latest"
    p0 = pick_versions(cv, shift=0, latest=latest)
    keys = _keys(cv, fys)
    sel_cols = (
        ["fs_div", "rcept_no", "avail_date", "currency", "in_vintage", "n_avail"]
        + CUR + ["ip", "ip_src", "cap"] + PRIOR_ORDER + ["prior_src"]
        + [f"src_{m}" for m in ALL_CUR] + ["ip_signed"]
    )  # fmt: skip
    p0s = p0.select(["corp_code", pl.col("t").alias("fy")] + sel_cols)
    df = keys.join(p0s, on=["corp_code", "fy"], how="left")

    if prior_source == "t_minus_1_report":
        df = _apply_prior_a(df, cv, latest=latest)

    df = df.join(_meta(inp.universe), on="corp_code", how="left")
    has = pl.col("rcept_no").is_not_null()
    df = df.with_columns(
        pl.when(~has)
        .then(pl.lit("none"))
        .when(pl.col("in_vintage").fill_null(False))
        .then(pl.lit("vintage"))
        .otherwise(pl.lit("raw"))
        .alias("layer"),
        pl.col("in_universe_attr").fill_null(False).alias("in_universe"),
        pl.col("prior_src").fill_null("none"),
        pl.col("ip_src").fill_null("none"),
    )
    extra = (
        [
            "k_n_cand",
            "k_n_vint",
            "k_n_avail",
            "k_n_no_receipt",
            "k_all_revision",
            "in_universe_attr",
        ]
        if diag
        else []
    )
    src_cols = ([f"src_{m}" for m in ALL_CUR] + ["ip_signed"]) if src else []
    return df.select(OUT_PANEL_COLS + extra + src_cols).sort(["fy", "corp_code"])


def _apply_prior_a(df: pl.DataFrame, cv: pl.DataFrame, *, latest: bool) -> pl.DataFrame:
    """기록용 (a): ``*_p1``·``*_p2`` 를 t−1·t−2 보고서의 판본 당기 값으로 바꾼다.

    판본은 형성 연도 t의 B_t로 자른다(CI-prior-cut). 고른 fs_div가 t와 다르면 그 신호는 결측.
    """
    out = df.drop(PRIOR_COLS)
    fills = []
    for k, ms in ((1, PRIOR1), (2, PRIOR2)):
        pk = pick_versions(cv, shift=k, latest=latest)
        pk = pk.select(
            pl.col("corp_code"),
            pl.col("t").alias("fy"),
            pl.col("fs_div").alias(f"_fs{k}"),
            *[pl.col(m).alias(f"_{m}_p{k}") for m in ms],
        )
        out = out.join(pk, on=["corp_code", "fy"], how="left")
        same = pl.col(f"_fs{k}") == pl.col("fs_div")
        fills += [
            pl.when(same).then(pl.col(f"_{m}_p{k}")).otherwise(None).alias(f"{m}_p{k}") for m in ms
        ]
        out = out.with_columns(pl.col(f"_fs{k}").is_not_null().alias(f"_has{k}"))
    out = out.with_columns(fills)
    psrc = (
        pl.when(pl.col("_has1") | pl.col("_has2"))
        .then(pl.lit("t_minus_1_report"))
        .otherwise(pl.lit("none"))
        .alias("prior_src")
    )
    drop = [c for c in out.columns if c.startswith("_")]
    return out.with_columns(psrc).drop(drop)


# ---------------------------------------------------------------- 공개 함수
def _inputs(lake: Lake, fys: Sequence[int], inputs: FsInputs | None, xbrl_cache) -> FsInputs:
    return inputs if inputs is not None else load_inputs(lake, fys, xbrl_cache=xbrl_cache)


def select_versions(
    lake: Lake,
    fys: Sequence[int],
    as_of: str = "base",
    *,
    xbrl_cache: str | Path | None = None,
    inputs: FsInputs | None = None,
) -> pl.DataFrame:
    """판본 선택 결과 [corp_code, fy, fs_div, rcept_no, layer, avail_date, n_avail].

    as_of="base": 각 fy에서 avail_date ≤ B_t인 후보만. as_of="latest": 기준일 없이 모든 후보.
    둘 다 연결 우선(가장 최근 CFS 판본에 총자산이 있으면 CFS, 아니면 OFS) 뒤 그 fs_div의
    (avail_date 내림, rcept_no 내림) 첫 판본 하나. 판본이 없는 corp-fy는 행이 없다.
    """
    if as_of not in ("base", "latest"):
        raise ValueError(f"as_of는 'base'|'latest': {as_of!r}")
    inp = _inputs(lake, fys, inputs, xbrl_cache)
    cv = resolve_values(inp.cv)
    p = pick_versions(cv, shift=0, latest=as_of == "latest")
    p = p.filter(pl.col("t").is_in(list(fys)))
    return p.select(
        "corp_code",
        pl.col("t").alias("fy"),
        "fs_div",
        "rcept_no",
        pl.when(pl.col("in_vintage"))
        .then(pl.lit("vintage"))
        .otherwise(pl.lit("raw"))
        .alias("layer"),
        "avail_date",
        "n_avail",
    ).sort(["fy", "corp_code"])


def fs_panel(
    lake: Lake,
    fys: Sequence[int],
    *,
    as_of: str = "base",
    prior_source: str = "frmtrm",
    use_xbrl_prior: bool = CI_USE_XBRL_PRIOR,
    layer_fill: bool = CI_LAYER_FILL,
    layers: str = "all",
    cap_xbrl: bool = True,
    xbrl_cache: str | Path | None = None,
    inputs: FsInputs | None = None,
) -> pl.DataFrame:
    """재무 입력 패널 (05 §5 열 계약). 행 = 분모 회사 × fys 중 연간 판본 후보가 있는 corp-fy.

    ``prior_source="frmtrm"``: ``*_p1``·``*_p2`` 를 고른 접수번호의 비교 칸(raw → XBRL)에서(주).
    ``"t_minus_1_report"``: t−1·t−2 보고서의 판본 당기 값에서(기록용 (a), §14 #22).
    ``in_universe`` = 분모 속성(KOSPI·KOSDAQ·SPAC 아님·금융 아님·12월 결산) &
    그 fy에 판본 후보가 있음.
    판본이 가용하지 않은 corp-fy도 행은 남고 값은 전부 결측, ``layer="none"``.
    ``layers``·``cap_xbrl``·``layer_fill`` 은 기록용 판(raw 한 층 단독·문면판)용이다.
    기본값이면 출력은 그대로다.
    """
    inp = _inputs(lake, fys, inputs, xbrl_cache)
    return build_panel(
        inp,
        fys,
        as_of=as_of,
        prior_source=prior_source,
        use_xbrl_prior=use_xbrl_prior,
        layer_fill=layer_fill,
        layers=layers,
        cap_xbrl=cap_xbrl,
    )


def fs_latest(
    lake: Lake,
    years: Sequence[int],
    *,
    xbrl_cache: str | Path | None = None,
    inputs: FsInputs | None = None,
) -> pl.DataFrame:
    """가장 늦게 알려진 판본의 값 [corp_code, year, fs_div, rcept_no, te, cap, ni] (결과 변수용).

    as_of="latest"(기준일 없음)로 같은 연결 우선 규칙을 쓴다. 값 읽기도 같은 층 규칙이다.
    """
    inp = _inputs(lake, years, inputs, xbrl_cache)
    cv = resolve_values(inp.cv)
    p = pick_versions(cv, shift=0, latest=True).filter(pl.col("t").is_in(list(years)))
    return p.select(
        "corp_code", pl.col("t").alias("year"), "fs_div", "rcept_no", "te", "cap", "ni"
    ).sort(["year", "corp_code"])


# ---------------------------------------------------------------- 층 불일치·coverage
def layer_mismatch(cv: pl.DataFrame, fys: Sequence[int]) -> pl.DataFrame:
    """같은 접수번호의 vintage·raw 값 쌍 수와 다른 쌍 수 [fy, metric, pairs, mismatch].

    후보 전체(고른 판본만이 아니라) 기준이다. 허용 오차 ``CI_MISMATCH_TOL`` 원.
    """
    both = cv.filter(pl.col("in_vintage") & pl.col("in_raw") & pl.col("fy").is_in(list(fys)))
    rows = []
    for m in VINT_METRICS:
        d = both.filter(pl.col(f"v_{m}").is_not_null() & pl.col(f"r_{m}").is_not_null())
        g = d.group_by("fy").agg(
            pl.len().alias("pairs"),
            ((pl.col(f"v_{m}") - pl.col(f"r_{m}")).abs() > CI_MISMATCH_TOL)
            .sum()
            .cast(pl.Int64)
            .alias("mismatch"),
        )
        rows.append(g.with_columns(pl.lit(m).alias("metric")))
    return pl.concat(rows).select("fy", "metric", "pairs", "mismatch").sort(["fy", "metric"])


def _count_rows(df: pl.DataFrame, scope: str) -> list[tuple[int, str, str, int]]:
    out: list[tuple[int, str, str, int]] = []
    fy_col = df["fy"].unique().sort().to_list()
    for fy in fy_col:
        d = df.filter(pl.col("fy") == fy)
        if scope == "in_universe":
            d = d.filter(pl.col("in_universe"))
        out.append((fy, scope, "rows", d.height))
        for c in VALUE_COLS:
            out.append((fy, scope, f"nn:{c}", int(d[c].is_not_null().sum())))
        for col in ("layer", "prior_src", "ip_src"):
            for val, n in d.group_by(col).len().iter_rows():
                out.append((fy, scope, f"{col}:{val}", n))
        for val, n in d.group_by(pl.col("src_cap").fill_null("none")).len().iter_rows():
            out.append((fy, scope, f"cap_src:{val}", n))
        vin = d.filter(pl.col("layer") == "vintage")
        rawl = d.filter(pl.col("layer") == "raw")
        for m in CUR + ["ip_op"]:
            sc = f"src_{m}"
            if sc not in d.columns:
                continue
            for s in ("raw", "xbrl"):
                out.append((fy, scope, f"fill_vintage_layer:{m}:{s}", int((vin[sc] == s).sum())))
            out.append((fy, scope, f"fill_raw_layer:{m}:xbrl", int((rawl[sc] == "xbrl").sum())))
        # 지급이자 원값 부호(절대값 취하기 전, 입력 분포 확인용)
        sg = d.filter(pl.col("ip_signed").is_not_null())
        for name, cond in (
            ("neg", pl.col("ip_signed") < 0),
            ("zero", pl.col("ip_signed") == 0),
            ("pos", pl.col("ip_signed") > 0),
        ):
            for s_ in ("op", "fin", "inv"):
                out.append(
                    (
                        fy,
                        scope,
                        f"ip_sign:{s_}:{name}",
                        int(sg.filter(cond & (pl.col("ip_src") == s_)).height),
                    )
                )
        # 키 단위 입력 존재 (원본 없는 키 등)
        raw_only = d.filter(pl.col("k_n_vint") == 0)
        out.append((fy, scope, "raw_only_corp_years", raw_only.height))
        nov = d.filter((pl.col("k_n_avail") == 0) & (pl.col("k_n_cand") > 0))
        out.append((fy, scope, "no_version_at_base", nov.height))
        out.append(
            (
                fy,
                scope,
                "no_version_at_base:missing_receipt",
                int((nov["k_n_no_receipt"] > 0).sum()),
            )
        )
        no_origin = nov.filter(pl.col("k_n_vint") == 0)
        out.append((fy, scope, "no_origin_keys", no_origin.height))
        out.append(
            (fy, scope, "no_origin_keys:revision_only", int(no_origin["k_all_revision"].sum()))
        )
    return out


def key_level_counts(cv: pl.DataFrame, fys: Sequence[int]) -> list[tuple[int, str, str, int]]:
    """(corp, fy, fs_div) 키 단위 "원본 없는 키" 수 — §5.1이 인용한 검토 숫자와 같은 단위.

    key_raw_only: 그 키의 후보가 vintage에 하나도 없는 키. ``:revision`` 은 그 접수가 정정본,
    ``:after_base`` 는 가용일이 B_t 뒤(또는 없음), ``:revision_after_base`` 는 둘 다.
    """
    b = pl.date(pl.col("fy") + 1, 6, 30)
    k = (
        cv.filter(pl.col("fy").is_in(list(fys)) & pl.col("fs_div").is_in(["CFS", "OFS"]))
        .group_by(["corp_code", "fy", "fs_div"])
        .agg(
            pl.col("in_vintage").any().alias("_v"),
            pl.col("is_revision").fill_null(False).all().alias("_rev"),
            (pl.col("avail_date").is_null() | (pl.col("avail_date") > b)).all().alias("_after"),
        )
        .filter(~pl.col("_v"))
    )
    out: list[tuple[int, str, str, int]] = []
    for fy in sorted(k["fy"].unique().to_list()):
        d = k.filter(pl.col("fy") == fy)
        out += [
            (fy, "all", "key_raw_only", d.height),
            (fy, "all", "key_raw_only:revision", int(d["_rev"].sum())),
            (fy, "all", "key_raw_only:after_base", int(d["_after"].sum())),
            (fy, "all", "key_raw_only:revision_after_base", int((d["_rev"] & d["_after"]).sum())),
        ]
    return out


def coverage_from(inp: FsInputs, fys: Sequence[int] | None = None) -> pl.DataFrame:
    """연도별 입력 존재 개수 [fy, scope, item, n] (결과 사건·점수 없음, §7.2).

    scope: all(패널 모든 행) | in_universe. item 예:
    - ``rows``, ``nn:<열>``(값이 있는 행 수), ``layer:*``, ``prior_src:*``, ``ip_src:*``,
      ``cap_src:*``, ``ip_sign:<op|fin|inv>:<neg|zero|pos>``(절대값 전 원값 부호)
    - ``fill_vintage_layer:<지표>:<raw|xbrl>``(CI-layer-fill로 채운 수),
      ``fill_raw_layer:<지표>:xbrl``
    - ``raw_only_corp_years``(vintage 후보가 없는 corp-year), ``no_version_at_base``(B_t까지
      가용한 후보가 없는 행), ``no_origin_keys``(그중 vintage 후보가 없는 행 = §5.1 원본 없는 키)
    - ``key_raw_only*``(corp·fy·fs_div 키 단위), ``layer_pairs:<지표>``·
      ``layer_mismatch:<지표>``(scope=all)
    """
    fys = sorted(set(inp.fys if fys is None else fys))
    panel = build_panel(inp, fys, diag=True, src=True)
    rows = _count_rows(panel, "all") + _count_rows(panel, "in_universe")
    rows += key_level_counts(inp.cv, fys)
    mm = layer_mismatch(inp.cv, fys)
    for fy, m, pairs, bad in mm.iter_rows():
        rows.append((fy, "all", f"layer_pairs:{m}", pairs))
        rows.append((fy, "all", f"layer_mismatch:{m}", bad))
    return pl.DataFrame(rows, schema=["fy", "scope", "item", "n"], orient="row").sort(
        ["fy", "scope", "item"]
    )


def coverage(
    lake: Lake,
    fys: Sequence[int],
    *,
    xbrl_cache: str | Path | None = None,
    inputs: FsInputs | None = None,
) -> pl.DataFrame:
    """``coverage_from`` 의 편의 함수(읽기부터 한다)."""
    return coverage_from(_inputs(lake, fys, inputs, xbrl_cache), fys)


# ---------------------------------------------------------------- CLI
def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if os.uname().sysname == "Darwin" else ru / 1024


def _parse_fys(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _dir_stat(d: Path) -> dict[str, int | str]:
    files = [p for p in d.rglob("*.parquet")] if d.is_dir() else []
    return {"path": str(d), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-snapshot", required=True)
    ap.add_argument("--derived-snapshot", required=True)
    ap.add_argument("--fys", required=True, help="예: 2015-2025 또는 2017,2018")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--xbrl-cache", default=None)
    ap.add_argument("--memory-limit", default="6GB")
    args = ap.parse_args(argv)

    fys = _parse_fys(args.fys)
    lake = Lake.from_env(raw_snapshot=args.raw_snapshot, derived_snapshot=args.derived_snapshot)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else lake.output_dir(f"quality_score_company_inputs_{args.raw_snapshot}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    inp = load_inputs(lake, fys, xbrl_cache=args.xbrl_cache, memory_limit=args.memory_limit)
    panel = build_panel(inp, fys)
    panel_a = build_panel(inp, fys, prior_source="t_minus_1_report")
    latest = fs_latest(lake, fys, inputs=inp)
    cov = coverage_from(inp, fys)
    urep = universe_report(lake)

    files = {
        "fs_panel": out_dir / "fs_panel.parquet",
        "fs_panel_a": out_dir / "fs_panel_a.parquet",
        "fs_latest": out_dir / "fs_latest.parquet",
    }
    panel.write_parquet(files["fs_panel"])
    panel_a.write_parquet(files["fs_panel_a"])
    latest.write_parquet(files["fs_latest"])
    cov_path = out_dir / "coverage_fs.tsv"
    cov.write_csv(cov_path, separator="\t")
    elapsed = time.time() - t0

    xcache = Path(args.xbrl_cache) if args.xbrl_cache else _default_cache(lake)
    manifest = {
        "raw_snapshot": args.raw_snapshot,
        "derived_snapshot": args.derived_snapshot,
        "fys": fys,
        "args": vars(args),
        "inputs": {
            t: _dir_stat(lake.raw_dir(t))
            for t in (
                "stock_master",
                "dart_corp_master",
                "dart_financial_statement_raw",
                "dart_filing_receipt_raw",
            )
        }
        | {
            t: _dir_stat(lake.derived_dir(t))
            for t in ("stock_metric_vintage_fact", "dim_trading_calendar")
        }
        | {
            "xbrl_cache": {
                "path": str(xcache),
                "exists": xcache.exists(),
                "sha256": sha256_file(xcache) if xcache.exists() else None,
            }
        },
        "code_sha256": code_sha256(__file__),
        "constants": {
            "CI_LAYER_FILL": CI_LAYER_FILL,
            "CI_USE_XBRL_PRIOR": CI_USE_XBRL_PRIOR,
            "CI_PRIOR_CUT": CI_PRIOR_CUT,
            "CI_FSDIV_NOFALLBACK": CI_FSDIV_NOFALLBACK,
            "CI_IP_ABS": CI_IP_ABS,
            "CI_DUP_TICKER_RULE": CI_DUP_TICKER_RULE,
            "CI_ACC_NULL_IS_DEC": CI_ACC_NULL_IS_DEC,
            "CI_MISMATCH_TOL": CI_MISMATCH_TOL,
            "RAW_MAP": {k: [list(a), list(s)] for k, (a, s) in RAW_MAP.items()},
            "XBRL_METRICS": {k: list(v) for k, v in XBRL_METRICS.items()},
        },
        "universe_report": urep,
        "outputs": {
            k: {"path": str(p), "rows": n, "sha256": sha256_file(p)}
            for (k, p), n in zip(
                files.items(), (panel.height, panel_a.height, latest.height), strict=True
            )
        }
        | {
            "coverage_fs": {
                "path": str(cov_path),
                "rows": cov.height,
                "sha256": sha256_file(cov_path),
            }
        },
        "elapsed_sec": round(elapsed, 1),
        "peak_rss_mb": round(_peak_rss_mb()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("elapsed_sec", "peak_rss_mb", "outputs")},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
