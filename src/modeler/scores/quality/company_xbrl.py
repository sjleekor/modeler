"""회사 점수(부분 C) XBRL 재무값 읽기 모듈 W2c (사전등록 20261010_quality_score §5.1·§5.2·§7.2).

배경: §5.1 판본 규칙이 고른 접수번호가 raw ``dart_financial_statement_raw``에 없으면(나중에
정정돼 raw에는 최신 접수번호만 남는다) 전기·전전기 비교 값, 자본금(IssuedCapital),
재무·투자활동 분류 지급이자(§5.2 이자보상배율)를 같은 접수번호의 XBRL 사실
(``dart_xbrl_fact_raw``)에서 읽는다. 이 모듈은 **읽기와 입력 일치 검증**만 한다.
점수·결과 변수·수익률은 다루지 않는다(§7.2). 값을 어디에 쓸지는 호출자가 정한다.

규칙 요약
- fs_div: 사실의 dimensions가 ``ConsolidatedAndSeparateFinancialStatementsAxis`` 한 축뿐이면
  그 멤버로 정한다(SeparateMember → OFS, ConsolidatedMember → CFS). 다른 축이 하나라도 더
  붙으면(자본변동표 구성요소 등) 버린다. 축이 없는 사실은 ``fs_div`` 를 정하지 못해 버린다
  (대상 concept 18개에서는 사업보고서 기준 0건이라 규칙이 필요 없다).
- 기간: 접수번호마다 기준 종료일 A(= 대상 사실의 가장 늦은 instant_date/period_end)를 잡고
  A와 같은 결산월이면 C, 12개월 전이면 P, 24개월 전이면 BP. 비12월 결산도 같은 원리다.
  duration은 길이가 약 1년(360~371일)인 것만 쓴다(분기·반기 duration 제외).
- 한 키에 값이 여럿이면 concept 우선순위가 높은 것, 같은 순위면 ifrs-full_·dart_ 접두어를
  ifrs_보다 먼저, 그래도 같으면 context_id 사전순 첫 값(결정적). 건수는 ``diagnostics`` 가 센다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from collections.abc import Sequence
from pathlib import Path

import duckdb
import polars as pl

# ---------------------------------------------------------------- 지표 → concept (우선순위 순)
# vintage ``xbrlfb.*`` 규칙(stock_metric_vintage_fact.mapping_priority)과 같은 concept·순서.
# ifrs-full_·ifrs_ 접두어는 같은 concept으로 본다(ifrs-full_ 먼저).
XBRL_METRICS: dict[str, tuple[str, ...]] = {
    "ta": ("ifrs-full_Assets",),
    "tl": ("ifrs-full_Liabilities",),
    "te": ("ifrs-full_Equity",),
    "ca": ("ifrs-full_CurrentAssets",),
    "cl": ("ifrs-full_CurrentLiabilities",),
    "re": ("ifrs-full_RetainedEarnings",),
    "oi": ("dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"),
    "ni": ("ifrs-full_ProfitLoss",),
    "ocf": ("ifrs-full_CashFlowsFromUsedInOperatingActivities",),
    "rev": ("ifrs-full_Revenue",),
    "gp": ("ifrs-full_GrossProfit",),
    "ltb": ("dart_LongTermBorrowingsGross", "ifrs-full_LongtermBorrowings"),
    # §5.2 C1 이자보상배율의 지급이자 세 계정 (CI-xbrl-interest: 재무·투자 분류는 vintage에 없어
    # 이 모듈이 정한다. 계정 이름은 실제 데이터에 있는지 존재 개수로 확인했다)
    "ip_op": ("ifrs-full_InterestPaidClassifiedAsOperatingActivities",),
    "ip_fin": ("ifrs-full_InterestPaidClassifiedAsFinancingActivities",),
    "ip_inv": ("ifrs-full_InterestPaidClassifiedAsInvestingActivities",),
    # §5.2 자본잠식률의 자본금 (CI-xbrl-capital: vintage에 없음)
    "cap": ("ifrs-full_IssuedCapital",),
}

# vintage metric_code ↔ 이 모듈 지표 (V1 검증용)
VINTAGE_METRIC = {
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

PERIOD_MONTHS = {"C": 0, "P": 12, "BP": 24}
DURATION_DAYS = (360, 371)  # CI-xbrl-duration: 1년 길이로 인정하는 일수 범위(364·365 모두 포함)
ANNUAL_REPRT = 11011
OUT_COLS = [
    "rcept_no",
    "corp_code",
    "bsns_year",
    "fs_div",
    "metric",
    "period",
    "period_date",
    "value",
    "concept_used",
]

_CS_AXIS = r'^\["ifrs(-full)?:ConsolidatedAndSeparateFinancialStatementsAxis=[^,]*"\]$'
_NORM = "lower(regexp_replace(concept_id,'^ifrs[-_](full_)?','ifrs-full_','i'))"


def _concept_table(metrics: Sequence[str]) -> list[tuple[str, str, int]]:
    """(정규화 concept 소문자, metric, 우선순위) 목록."""
    rows = []
    for m in metrics:
        for i, c in enumerate(XBRL_METRICS[m]):
            rows.append((c.lower(), m, i))
    return rows


def _connect(memory_limit: str = "6GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"set memory_limit='{memory_limit}'")
    return con


def build_candidates(
    con: duckdb.DuckDBPyConnection,
    xbrl_glob: str,
    rcept_nos: Sequence[str] | None,
    metrics: Sequence[str] | None = None,
) -> None:
    """임시 테이블 ``cand``를 만든다: 대상 concept의 사실 + fs_div·기간 판정 결과.

    열: rcept_no, corp_code, bsns_year, fs_div, metric, prio, concept_id, context_id,
    ctx_period(context_id 접두어가 말하는 기간), period(날짜로 정한 기간, 못 정하면 NULL),
    d(기준 날짜), value. fs_div가 없거나 축이 더 붙은 사실은 들어오지 않는다.
    """
    metrics = list(metrics) if metrics is not None else list(XBRL_METRICS)
    unknown = [m for m in metrics if m not in XBRL_METRICS]
    if unknown:
        raise KeyError(f"알 수 없는 지표: {unknown}")
    cmap = _concept_table(metrics)
    con.execute("create or replace temp table cmap(nc varchar, metric varchar, prio int)")
    con.executemany("insert into cmap values (?,?,?)", cmap)

    rfilter = ""
    if rcept_nos is not None:
        rsel = pl.DataFrame(
            {"rcept_no": [str(r) for r in dict.fromkeys(rcept_nos)]}, schema={"rcept_no": pl.Utf8}
        )
        con.register("rsel_df", rsel.to_arrow())
        con.execute("create or replace temp table rsel as select * from rsel_df")
        con.unregister("rsel_df")
        rfilter = "and rcept_no in (select rcept_no from rsel)"

    # 1) 대상 concept · 비nil · 값 있음 · fs_div(축 정확히 하나)
    con.execute(f"""
        create or replace temp table raw1 as
        select rcept_no, corp_code, bsns_year, concept_id, context_id, context_type,
               period_start, coalesce(instant_date, period_end) as d,
               case
                 when regexp_matches(dimensions, '{_CS_AXIS}') then
                   case when dimensions like '%SeparateMember"]' then 'OFS'
                        when dimensions like '%ConsolidatedMember"]' then 'CFS' end
               end as fs_div,
               regexp_extract(context_id, '^(BPFY|CFY|PFY)', 1) as ctx_pre,
               cast(value_numeric as double) as value, {_NORM} as nc
        from read_parquet('{xbrl_glob}', hive_partitioning=true)
        where reprt_code = {ANNUAL_REPRT}
          and not coalesce(is_nil, false) and value_numeric is not null
          and {_NORM} in (select nc from cmap) {rfilter}
        """)
    # 기준 종료일 A: 접수번호의 모든 사실 중 CFY 접두어가 붙은 것의 날짜 최빈값(모든 concept).
    # 대상 concept에 당기 값이 없는 접수번호(당기 재무 사실이 빠진 경우)에서도 P·BP를 한 해씩
    # 밀어 잘못 붙이지 않게 하려는 것이다(CI-xbrl-anchor). 각 사실의 기간은 자기 날짜로 정한다.
    con.execute(f"""
        create or replace temp table anchor as
        with cfy as (
          select rcept_no, coalesce(instant_date, period_end) as d, count(*) as n
          from read_parquet('{xbrl_glob}', hive_partitioning=true)
          where reprt_code = {ANNUAL_REPRT} and starts_with(context_id, 'CFY')
                and coalesce(instant_date, period_end) is not null {rfilter}
          group by 1, 2
        ),
        best as (
          select rcept_no, arg_max(d, n * 100000 + epoch(d) / 86400) as a from cfy group by 1
        ),
        fb as (
          select rcept_no, max(d) as a from raw1 where fs_div is not null group by 1
        )
        select rcept_no, coalesce(best.a, fb.a) as a
        from fb left join best using (rcept_no)
        """)
    lo, hi = DURATION_DAYS
    # 2) 기간 판정: 기준 종료일 A와의 개월 차 0/12/24, 말일 규칙 포함, duration은 약 1년
    con.execute(f"""
        create or replace temp table cand0 as
        select r.rcept_no, r.corp_code, r.bsns_year, r.fs_div, m.metric, m.prio, r.concept_id,
               r.context_id, r.d, r.value, r.nc,
               case r.ctx_pre when 'CFY' then 'C' when 'PFY' then 'P' when 'BPFY' then 'BP' end
                 as ctx_period,
               case
                 when r.context_type = 'duration'
                      and not (datediff('day', r.period_start, r.d) between {lo} and {hi})
                   then null
                 when (year(a.a)*12 + month(a.a)) - (year(r.d)*12 + month(r.d)) not in (0,12,24)
                   then null
                 when not (day(r.d) = day(a.a) or (r.d = last_day(r.d) and a.a = last_day(a.a)))
                   then null
                 else case (year(a.a)*12 + month(a.a)) - (year(r.d)*12 + month(r.d))
                        when 0 then 'C' when 12 then 'P' else 'BP' end
               end as period_d
        from raw1 r
        join anchor a using (rcept_no)
        join cmap m on m.nc = r.nc
        where r.fs_div is not null
        """)
    # 날짜로 정한 기간이 context_id 접두어와 어긋나면 버린다(결산기 변경으로 전기가 12개월 전이
    # 아닌 경우 등. CI-xbrl-mismatch-drop). 접두어가 없으면 날짜 판정을 그대로 쓴다.
    con.execute("""
        create or replace temp table cand as
        select *, case when ctx_period is not null and period_d is not null
                            and ctx_period <> period_d then null else period_d end as period
        from cand0
        """)


def _select_sql() -> str:
    # 같은 키 우선순위: concept 순서 → ifrs-full_ 접두어 → context_id 사전순
    return """
        select rcept_no, corp_code, cast(bsns_year as int) as bsns_year, fs_div, metric, period,
               d as period_date, value, concept_id as concept_used
        from (
          select *, row_number() over (
            partition by rcept_no, fs_div, metric, period
            order by prio, (case when lower(concept_id) like 'ifrs\\_%' escape '\\'
                                      and lower(concept_id) not like 'ifrs-full%' then 1
                                 else 0 end),
                     context_id, abs(year(a) - bsns_year), bsns_year) as rn
          from cand join anchor using (rcept_no) where period is not null)
        where rn = 1
        order by rcept_no, fs_div, metric, period
    """


def xbrl_values(
    xbrl_glob: str,
    rcept_nos: Sequence[str] | None,
    metrics: Sequence[str] | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """접수번호별 XBRL 값 표 (§5.1 전년·전전년 출처, §5.2 지급이자·자본금).

    반환 열: rcept_no, corp_code, bsns_year, fs_div, metric, period, period_date, value,
    concept_used. (rcept_no, fs_div, metric, period)당 한 행.
    ``rcept_nos`` 가 None이면 사업보고서(reprt_code=11011) 전체.
    """
    own = con is None
    con = con or _connect()
    try:
        build_candidates(con, xbrl_glob, rcept_nos, metrics)
        df = con.execute(_select_sql()).pl()
    finally:
        if own:
            con.close()
    return df.select(OUT_COLS)


# ---------------------------------------------------------------- 진단
def diagnostics(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """``build_candidates`` 뒤 판정 규칙의 입력 존재·어긋남 개수 (결과 사건 아님)."""
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    out: dict[str, object] = {}
    out["후보행"] = q("select count(*) from cand")[0][0]
    out["기간_최종_미판정"] = q("select count(*) from cand where period is null")[0][0]
    out["접두어_날짜_어긋남(버림)"] = q(
        "select count(*) from cand where period_d is not null and ctx_period is not null "
        "and period_d <> ctx_period"
    )[0][0]
    out["어긋남_접수번호"] = q(
        "select count(distinct rcept_no) from cand where period_d is not null "
        "and ctx_period is not null and period_d <> ctx_period"
    )[0][0]
    out["날짜_미판정(분기·결산기변경 등)"] = q("select count(*) from cand where period_d is null")[
        0
    ][0]
    out["접두어_없는_사실"] = q("select count(*) from cand where ctx_period is null")[0][0]
    out["기준일_해_다른_접수"] = q(
        "select count(*) from anchor a join (select distinct rcept_no, bsns_year from cand) c "
        "using (rcept_no) where year(a.a) <> c.bsns_year"
    )[0][0]
    out["중복키_수"] = q(
        "select count(*) from (select 1 from cand where period is not null "
        "group by rcept_no, fs_div, metric, period having count(*) > 1)"
    )[0][0]
    out["중복키_값상이"] = q(
        "select count(*) from (select 1 from cand where period is not null "
        "group by rcept_no, fs_div, metric, period having count(distinct value) > 1)"
    )[0][0]
    out["중복키_같은_concept_값상이"] = q(
        "select count(*) from (select 1 from cand where period is not null "
        "group by rcept_no, fs_div, metric, period, nc having count(distinct value) > 1)"
    )[0][0]
    return out


# ---------------------------------------------------------------- 검증 V1~V3
def validate_v1(
    con: duckdb.DuckDBPyConnection, vintage_glob: str, xbrl_glob: str | None = None
) -> pl.DataFrame:
    """V1: vintage annual 당기 값 ↔ 이 모듈 C 값. 지표·출처(fin/xbrlfb)별 쌍 수·불일치 수.

    ``build_candidates``(전체 11011, 전 지표)가 먼저 돌아 있어야 한다.
    """
    vm = ",".join(f"('{k}','{v}')" for k, v in VINTAGE_METRIC.items())
    return con.execute(f"""
        with vm(metric, vcode) as (values {vm}),
        x as ({_select_sql()}),
        v as (
          select rcept_no, fs_basis, metric_code, value_numeric,
                 case when mapping_rule_code like 'xbrlfb.%' then 'xbrlfb' else 'fin' end as src
          from read_parquet('{vintage_glob}', hive_partitioning=true)
          where period_type = 'annual' and reprt_code = '11011' and value_numeric is not null
        )
        select vm.metric, v.src, count(*) as pairs,
               sum(case when abs(v.value_numeric - x.value) > 0.5 then 1 else 0 end) as mismatch
        from v join vm on vm.vcode = v.metric_code
        join x on x.rcept_no = v.rcept_no and x.metric = vm.metric and x.period = 'C'
             and x.fs_div = v.fs_basis
        group by 1, 2 order by 1, 2
        """).pl()


def validate_v2(con: duckdb.DuckDBPyConnection, raw_fs_glob: str) -> pl.DataFrame:
    """V2: raw FS의 frmtrm·bfefrmtrm ↔ 이 모듈 P·BP. 지표별 쌍 수·불일치 수.

    BS 지표는 sj_div BS, ni는 IS·CIS, ocf·ip_*는 CF로 좁히고 fs_div(CFS/OFS)도 맞춘다.
    """
    sj = {
        "ta": "BS", "tl": "BS", "te": "BS", "ca": "BS", "cl": "BS", "re": "BS", "cap": "BS",
        "ltb": "BS", "ni": "IS,CIS", "oi": "IS,CIS", "rev": "IS,CIS", "gp": "IS,CIS",
        "ocf": "CF", "ip_op": "CF", "ip_fin": "CF", "ip_inv": "CF",
    }  # fmt: skip
    cm = ",".join(f"('{m}','{c.lower()}','{sj[m]}')" for m in sj for c in XBRL_METRICS[m])
    return con.execute(f"""
        with cm(metric, nc, sjs) as (values {cm}),
        x as ({_select_sql()}),
        r as (
          select rcept_no, fs_div, sj_div,
                 lower(regexp_replace(account_id,'^ifrs[-_](full_)?','ifrs-full_','i')) as nc,
                 try_cast(frmtrm_amount as double) as p, try_cast(bfefrmtrm_amount as double) as bp
          from read_parquet('{raw_fs_glob}', hive_partitioning=true)
          where reprt_code = 11011
        ),
        j as (
          select cm.metric, cm.nc, r.rcept_no, r.fs_div, r.p, r.bp
          from r join cm on cm.nc = r.nc and list_contains(string_split(cm.sjs, ','), r.sj_div)
        ),
        jp as (
          select j.metric, 'P' as period, x.value as xv, j.p as rv from j
          join x on x.rcept_no = j.rcept_no and x.fs_div = j.fs_div and x.metric = j.metric
                 and x.period = 'P' and lower(regexp_replace(x.concept_used, '^ifrs[-_](full_)?',
                 'ifrs-full_', 'i')) = j.nc where j.p is not null
          union all
          select j.metric, 'BP', x.value, j.bp from j
          join x on x.rcept_no = j.rcept_no and x.fs_div = j.fs_div and x.metric = j.metric
                 and x.period = 'BP' and lower(regexp_replace(x.concept_used, '^ifrs[-_](full_)?',
                 'ifrs-full_', 'i')) = j.nc where j.bp is not null
        )
        select metric, period, count(*) as pairs,
               sum(case when abs(xv - rv) > 0.5 then 1 else 0 end) as mismatch
        from jp group by 1, 2 order by 1, 2
        """).pl()


def validate_v3(
    con: duckdb.DuckDBPyConnection, vintage_glob: str, raw_fs_glob: str
) -> pl.DataFrame:
    """V3: vintage 사업보고서 접수번호 중 raw FS에 없는 것(사업연도별)에서 P 값을 줄 수 있는 수."""
    return con.execute(f"""
        with vr as (
          select distinct rcept_no, bsns_year from read_parquet('{vintage_glob}',
            hive_partitioning=true) where period_type='annual' and reprt_code='11011'
        ),
        rr as (
          select distinct rcept_no from read_parquet('{raw_fs_glob}', hive_partitioning=true)
          where reprt_code = 11011
        ),
        t as (select vr.* from vr where rcept_no not in (select rcept_no from rr)),
        x as ({_select_sql()}),
        hv as (
          select rcept_no, metric from x where period = 'P' group by 1, 2
        )
        select t.bsns_year, count(distinct t.rcept_no) as rcepts,
          {",".join(
              f"count(distinct case when hv.metric='{m}' then t.rcept_no end) as has_{m}"
              for m in ("ta", "te", "ni", "ocf", "cap", "ip_op")
          )}
        from t left join hv using (rcept_no)
        group by 1 order by 1
        """).pl()


# ---------------------------------------------------------------- 캐시 CLI
def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if os.uname().sysname == "Darwin" else ru / 1024


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-snapshot", required=True, help="예: 2026-09-30")
    ap.add_argument("--out", default=None, help="출력 parquet 경로(기본: 캐시 디렉터리)")
    args = ap.parse_args(argv)

    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    raw = root / "kr/raw/raw_postgres" / f"snapshot_date={args.raw_snapshot}" / "source=sj2_remote"
    xdir = raw / "dart_xbrl_fact_raw"
    xbrl_glob = f"{xdir}/**/*.parquet"
    files = sorted(xdir.rglob("*.parquet"))
    out_dir = root / "kr/output" / f"quality_score_company_cache_{args.raw_snapshot}"
    out = Path(args.out) if args.out else out_dir / "xbrl_values.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    con = _connect()
    build_candidates(con, xbrl_glob, None)
    diag = diagnostics(con)
    df = con.execute(_select_sql()).pl()
    con.close()
    df.select(OUT_COLS).write_parquet(out)
    elapsed = time.time() - t0

    manifest = {
        "raw_snapshot": args.raw_snapshot,
        "input_glob": xbrl_glob,
        "input_files": len(files),
        "input_bytes": sum(f.stat().st_size for f in files),
        "code_sha256": _sha256_file(Path(__file__)),
        "rows": df.height,
        "rcept_nos": df["rcept_no"].n_unique(),
        "metrics": {k: list(v) for k, v in XBRL_METRICS.items()},
        "diagnostics": diag,
        "elapsed_sec": round(elapsed, 1),
        "peak_rss_mb": round(_peak_rss_mb()),
    }
    (out.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
