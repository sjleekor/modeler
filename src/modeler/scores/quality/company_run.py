"""회사 점수(부분 C) 묶기·실행 W6 (사전등록 20261010_quality_score §5·§7.2·§9·§12.2·§12.4 C).

지금까지 만든 모듈(W0~W5)을 이어 세 가지를 한다.

* ``--period dev`` (기본): 형성 FY2017~2018 디버깅. 점수와 결과를 잇는다(§5.4 개발 구간 허용).
  O1~O3은 t+1, O4는 ``o4_dev_t1``. 판정 통계·게이트(등급은 참고)·차원별 16칸·기록용 비교·
  디버깅 표본(``dev_samples.tsv``)을 낸다.
* ``--period judgment``: 형성 FY2019~2024(O4는 2019~2022). ``QUALITY_C_JUDGMENT_CONFIRMED`` 가
  없으면 **아무 파일도 쓰기 전에** 거부한다(§7.2).
* ``--period checks``: 입력 존재 확인만. 형성 FY2015~2025 전부. 결과 변수 함수는 부르지 않고
  점수는 non-null 수만 센다(§7.2 (a)).

데이터를 읽는 부분(``load_sources``)과 계산 부분(``analyze``·``run_checks``)을 나눴다. 계산 부분은
``Sources`` 프레임 묶음만 받아서 합성 자료로 시험한다.

사전등록 문면이 정하지 않은 선택은 상수로 두고 ``# CI-<이름>`` 주석을 달았다.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import polars as pl

from modeler.scores.quality import company_inputs_fs as fi
from modeler.scores.quality import company_inputs_misc as mi
from modeler.scores.quality import company_judge as cj
from modeler.scores.quality import company_maps as cm
from modeler.scores.quality import company_outcomes as co
from modeler.scores.quality import company_score as cs
from modeler.scores.quality.company_common import (
    DEV_YEARS,
    JUDGMENT_ENV,
    JUDGMENT_YEARS,
    O4_JUDGMENT_YEARS,
    Lake,
    TradingCalendar,
    default_prereg,
    filing_availability,
    git_head,
    guard_years,
    sha256_file,
)

RUN_VERSION = "quality-score-company/company_run/1"

# ---------------------------------------------------------------- 구간·상수
CHECK_YEARS = tuple(range(2015, 2026))  # §5.4 checks: 입력 존재만, FY2015~2025 전부
FS_LATEST_YEARS = tuple(range(2015, 2026))  # 결과 변수용 "가장 늦은 판본" 사업연도
PERIOD_YEARS = {"dev": DEV_YEARS, "judgment": JUDGMENT_YEARS, "checks": CHECK_YEARS}
PERIODS = tuple(PERIOD_YEARS)

SAMPLE_EVENTS = 10  # 디버깅 표본: 결과마다 사건 최대 10건
SAMPLE_NON_EVENTS = 5  # 비사건 최대 5건

# CI-o1-auto: auto는 짝수 사업연도(bsns_year) 감사의견 보고서가 하나라도 있으면 full(§5.3 R03).
# 사전등록은 "있으면"만 적었고 최소 개수를 정하지 않았다. 09-30 raw에는 짝수 해 보고서가 0건이다.
O1_AUTO_MIN_EVEN_REPORTS = 1  # CI-o1-auto

# CI-ccy-nearest: O3 dps 통화는 fs 패널의 (corp, fy = report_year) 통화, 없으면 그 회사에서
# 연도 차이가 가장 작은 해의 통화(차이가 같으면 이른 해). 통화가 하나도 없으면 null(W4가 KRW로 읽음).
CI_CCY_TIE = "earlier_year"  # CI-ccy-nearest

# CI-capevt-nomisc: misc 패널에 행이 없는 (corp, fy)의 사건 해 표시는 False로 읽는다.
CI_CAPEVT_NO_MISC_FALSE = True  # CI-capevt-nomisc

# CI-record-key: 기록용 비교는 결과 이름 O1~O4 아래에 항목 이름을 "차원" 칸으로 달아 낸다
# (``company_judge.record_cells`` 가 O1~O4 키만 읽기 때문).
CELL_DIMS = {"C1": "c1", "C2": "c2", "C3": "c3", "F": "f_sum"}
SCORE_COUNT_COLS = ("c1", "c2", "c3", "c", "f_sum", "f_ltb_req")  # checks: non-null 수만

# 해석 표 대안 기록용 항목 이름(§9, A4·P3: 판정 규칙은 그대로, 대안은 기록용 숫자로만 낸다).
REC_D3 = "interp_d3_dps_multi_missing"
REC_D5 = "interp_d5_anchor_max_off"
REC_C3_LATEST = "record_c3_latest_version"
C3_LATEST_NOTE = (
    "PIT 위반 표시: 배당·주식수·자사주 판본을 B_t 기준일 없이 가장 늦은 판본(접수번호 최대)으로 골랐다. "
    "B_t 뒤 정정본의 값이 들어가므로 판정에 쓰지 않는다. 해석 표 승인 때 사용자에게 제안할 결과 전 기록이다."
)

RAW_TABLES = (
    "stock_master",
    "dart_corp_master",
    "dart_financial_statement_raw",
    "dart_filing_receipt_raw",
    "dart_governance_raw",
    "dart_shareholder_return_raw",
    "dart_share_count_raw",
    "dart_capital_change_raw",
    "dart_xbrl_fact_raw",
)
DERIVED_TABLES = ("stock_metric_vintage_fact", "dim_trading_calendar")

CI_NOTES = {
    "CI-o1-auto": f"짝수 사업연도 감사의견 보고서 {O1_AUTO_MIN_EVEN_REPORTS}건 이상이면 full",
    "CI-ccy-nearest": f"O3 통화: 같은 해, 없으면 가장 가까운 해({CI_CCY_TIE}), 없으면 null",
    "CI-capevt-nomisc": "misc 행이 없는 corp-fy의 사건 해 표시는 False",
    "CI-record-key": "기록용 항목은 record_cells의 차원 칸 이름으로 낸다",
    "CI-sample-seed": "디버깅 표본의 시드는 실행 시드(--seed)와 같다",
    "CI-checks-scores": "checks의 점수 입력 수는 compute_scores를 돌려 non-null만 센다",
}

PANEL_FLAG_COLS = (
    "capevt",
    "capevt_p1",
    "capevt_p2",
    "capevt_rec",
    "capevt_rec_p1",
    "capevt_rec_p2",
)


# ================================================================ 1. 패널 묶기
def assemble_panel(
    fs: pl.DataFrame, misc: pl.DataFrame, *, capevt: str = "judgment"
) -> pl.DataFrame:
    """fs_panel(행 기준) ⟕ misc_panel on (corp_code, fy). ``REQUIRED_COLS`` 를 모두 갖는다(W3 열 계약).

    ``capevt="judgment"`` 는 판정용 동결 목록의 ``capevt*`` 를 그대로 쓴다. ``"record"`` 는
    ``capevt``·``capevt_p1``·``capevt_p2`` 를 misc의 ``capevt_rec*``(액면 변경 해를 더한 기록용,
    정정 C-2)로 바꿔 넣는다. 사건 해 표시의 결측은 False다(CI-capevt-nomisc).
    """
    if capevt not in ("judgment", "record"):
        raise ValueError(f"capevt는 judgment·record 중 하나입니다: {capevt!r}")
    keys = ["corp_code", "fy"]
    f = fs.with_columns(pl.col("fy").cast(pl.Int64))
    m = misc.with_columns(pl.col("fy").cast(pl.Int64))
    if m.select(keys).is_duplicated().any():
        raise ValueError("misc 패널의 (corp_code, fy)가 유일하지 않습니다.")
    overlap = [c for c in m.columns if c in f.columns and c not in keys]
    if overlap:
        raise ValueError(f"fs·misc 패널에 겹치는 열이 있습니다: {overlap}")
    out = f.join(m, on=keys, how="left", maintain_order="left")
    if capevt == "record":
        for tag in ("", "_p1", "_p2"):
            if f"capevt_rec{tag}" not in out.columns:
                raise ValueError(f"기록용 사건 해 열이 없습니다: capevt_rec{tag}")
            out = out.drop(f"capevt{tag}", strict=False).rename(
                {f"capevt_rec{tag}": f"capevt{tag}"}
            )
        # 기록용 열 이름은 보존한다(값은 같다).
        out = out.with_columns(
            *[pl.col(f"capevt{t}").alias(f"capevt_rec{t}") for t in ("", "_p1", "_p2")]
        )
    if CI_CAPEVT_NO_MISC_FALSE:
        fill = [c for c in PANEL_FLAG_COLS if c in out.columns]
        out = out.with_columns(*[pl.col(c).fill_null(False) for c in fill])
    missing = [c for c in cs.REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"묶은 패널에 W3 필수 열이 없습니다: {missing}")
    return out


# ================================================================ 2. 결과 입력 묶기
@dataclass
class OutcomeInputs:
    """W4가 기대하는 프레임 묶음(사전등록 §5.3)."""

    formation: pl.DataFrame  # corp_code, fy, in_universe, currency, te, cap, ni
    fs_latest: pl.DataFrame  # corp_code, year, te, cap, ni
    dps_latest: pl.DataFrame  # corp_code, report_year, currency, dps_t, dps_p1
    opinions: pl.DataFrame
    capevt: pl.DataFrame  # corp_code, year
    universe_years: pl.DataFrame  # corp_code, year
    fs_latest_full: pl.DataFrame | None = None  # + fs_div, rcept_no (디버깅 표본용)
    dps_latest_full: pl.DataFrame | None = None  # + rcept_no, knd_source
    currency_note: dict = field(default_factory=dict)


def formation_frame(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.select(
        pl.col("corp_code"),
        pl.col("fy").cast(pl.Int64),
        pl.col("in_universe").fill_null(False),
        pl.col("currency"),
        pl.col("te"),
        pl.col("cap"),
        pl.col("ni"),
    )


def dps_with_currency(dps_lat: pl.DataFrame, ccy_panel: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """dps_latest에 통화를 단다(CI-ccy-nearest).

    통화는 fs 패널의 (corp_code, fy = report_year) 통화이고, 없으면 그 회사에서 연도 차이가 가장 작은
    해(같으면 이른 해)의 통화다. 회사에 통화가 하나도 없으면 null이다(W4·W3이 KRW로 읽음).
    반환: (``currency`` 열을 단 프레임, {n, n_same_year, n_nearest_year, n_none}).
    """
    ccy = (
        ccy_panel.filter(pl.col("currency").is_not_null())
        .select("corp_code", pl.col("fy").cast(pl.Int64).alias("_fy"), "currency")
        .unique(subset=["corp_code", "_fy"], keep="first", maintain_order=True)
    )
    d = dps_lat.with_columns(pl.col("report_year").cast(pl.Int64))
    near = (
        d.select("corp_code", "report_year")
        .unique()
        .join(ccy, on="corp_code", how="inner")
        .with_columns((pl.col("report_year") - pl.col("_fy")).abs().alias("_gap"))
        .sort(["corp_code", "report_year", "_gap", "_fy"])
        .unique(subset=["corp_code", "report_year"], keep="first", maintain_order=True)
        .select("corp_code", "report_year", "currency", "_gap")
    )
    out = d.join(near, on=["corp_code", "report_year"], how="left", maintain_order="left")
    note = {
        "n": out.height,
        "n_same_year": int((out["_gap"] == 0).sum()),
        "n_nearest_year": int((out["_gap"] > 0).sum()),
        "n_none": int(out["_gap"].is_null().sum()),
        "rule": CI_CCY_TIE,
    }
    return out.drop("_gap"), note


def outcome_inputs(
    panel: pl.DataFrame,
    *,
    fs_lat: pl.DataFrame,
    dps_lat: pl.DataFrame,
    opinions: pl.DataFrame,
    capevt: pl.DataFrame,
    currency_panel: pl.DataFrame | None = None,
) -> OutcomeInputs:
    """묶은 패널과 결과용 원천 프레임 → W4 입력.

    * formation = 패널의 [corp_code, fy, in_universe, currency, te, cap, ni]
    * fs_latest = [corp_code, year, te, cap, ni]
    * dps_latest = [corp_code, report_year, currency, dps_t, dps_p1] (통화는 ``dps_with_currency``)
    * opinions = ``audit_opinion_rows`` 그대로, capevt = ``share_event_flags`` [corp_code, year]
    * universe_years = in_universe 패널 행의 [corp_code, year = fy]
    ``currency_panel`` 은 통화를 찾을 fs 패널(기본: ``panel``).
    """
    dps_full, note = dps_with_currency(dps_lat, panel if currency_panel is None else currency_panel)
    uni = (
        panel.filter(pl.col("in_universe").fill_null(False))
        .select("corp_code", pl.col("fy").cast(pl.Int64).alias("year"))
        .unique()
        .sort("corp_code", "year")
    )
    return OutcomeInputs(
        formation=formation_frame(panel),
        fs_latest=fs_lat.select("corp_code", pl.col("year").cast(pl.Int64), "te", "cap", "ni"),
        dps_latest=dps_full.select("corp_code", "report_year", "currency", "dps_t", "dps_p1"),
        opinions=opinions,
        capevt=capevt.select("corp_code", pl.col("year").cast(pl.Int64)),
        universe_years=uni,
        fs_latest_full=fs_lat,
        dps_latest_full=dps_full,
        currency_note=note,
    )


# ================================================================ 3. O1 모드·짝수 해 규칙
def decide_o1_mode(opinions: pl.DataFrame, requested: str = "auto") -> dict:
    """O1 모드를 정한다(§5.3 R03·§12.4 C). 근거로 짝수·홀수 사업연도 보고서 수를 남긴다.

    ``auto``: raw ``dart_governance_raw`` 감사의견 보고서 중 ``bsns_year`` 가 짝수인 것이
    ``O1_AUTO_MIN_EVEN_REPORTS`` 건 이상이면 full(짝수 해 백필이 들어옴), 아니면 fallback.
    """
    if requested not in ("auto", "full", "fallback"):
        raise ValueError(f"o1 모드는 auto·full·fallback 중 하나입니다: {requested!r}")
    rep = opinions.select("rcept_no", pl.col("report_year").cast(pl.Int64)).unique()
    by = {
        int(y): int(n)
        for y, n in rep.group_by("report_year").agg(pl.len()).sort("report_year").iter_rows()
    }
    n_even = sum(n for y, n in by.items() if y % 2 == 0)
    n_odd = sum(n for y, n in by.items() if y % 2 == 1)
    mode = requested
    if requested == "auto":
        mode = "full" if n_even >= O1_AUTO_MIN_EVEN_REPORTS else "fallback"
    return {
        "requested": requested,
        "mode": mode,
        "n_even_year_reports": n_even,
        "n_odd_year_reports": n_odd,
        "reports_by_bsns_year": {str(k): v for k, v in by.items()},
        "auto_min_even_reports": O1_AUTO_MIN_EVEN_REPORTS,
        "rule": "§5.3 R03: 짝수 해 백필이 들어왔으면 full(6개 연도), 아니면 fallback(2020·2022·2024)",
    }


def opinion_rates_and_flags(
    universe_years: pl.DataFrame, opinions: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """연도별 의견 결측 비율과 짝수 해 3%p 플래그(입력 존재 비율이라 판정 구간도 허용)."""
    rates = co.opinion_missing_rate(universe_years, opinions)
    return rates, co.even_year_flags(rates)


def demoted_formation_years(flags: pl.DataFrame, mode: str) -> list[int]:
    """full 모드에서 짝수 해 격차가 3%p를 넘어 O1 판정에서 기록용으로 내릴 형성 연도 t.

    fallback은 t+1이 홀수인 해만 읽어서 해당 없다."""
    if mode != "full":
        return []
    return sorted(int(t) for t, dm in zip(flags["t"].to_list(), flags["demote"].to_list()) if dm)


# ================================================================ 4. 결과·점수 연결
def compute_outcomes(
    oi: OutcomeInputs,
    years: Sequence[int],
    *,
    period: str,
    o1_mode: str,
    demoted_t: Sequence[int] = (),
    capevt: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """O1~O4 상태 프레임. ``O1_demoted`` 는 짝수 해 규칙으로 내린 형성 연도(있을 때만).

    dev의 O4는 ``o4_dev_t1``(t+1만, §5.4), judgment의 O4는 형성 FY2019~2022의 t+1~t+3.
    """
    years = list(years)
    cap = oi.capevt if capevt is None else capevt
    o1 = co.o1(oi.formation, oi.opinions, years, mode=o1_mode)
    out: dict[str, pl.DataFrame] = {}
    if demoted_t:
        d = list(demoted_t)
        out["O1"] = o1.filter(~pl.col("fy").is_in(d))
        out["O1_demoted"] = o1.filter(pl.col("fy").is_in(d))
    else:
        out["O1"] = o1
    out["O2"] = co.o2(oi.formation, oi.fs_latest, years, kind="partial")
    out["O3"] = co.o3(oi.formation, oi.dps_latest, cap, years)
    if period == "dev":
        out["O4"] = co.o4_dev_t1(oi.formation, oi.fs_latest, years)
    else:
        out["O4"] = co.o4(oi.formation, oi.fs_latest, [y for y in years if y in O4_JUDGMENT_YEARS])
    return out


def link_score(scores: pl.DataFrame, outcome: pl.DataFrame, col: str) -> pl.DataFrame:
    """결과 프레임과 점수 열을 잇는다 → [corp_code, fy, score, event]. 점수와 결과를 잇는 곳이다.

    판정 구간(형성 FY2019 이상)은 확인 환경변수가 있어야 한다(§7.2 ``guard_years`` link).
    점수가 있고 event가 0/1인 행만 남는다."""
    guard_years([int(y) for y in outcome["fy"].unique().to_list()], "link")
    o = outcome.filter(pl.col("event").is_not_null()).select(
        "corp_code", pl.col("fy").cast(pl.Int64), pl.col("event")
    )
    s = (
        scores.filter(pl.col("in_universe") & pl.col(col).is_not_null())
        .select(
            "corp_code", pl.col("fy").cast(pl.Int64), pl.col(col).cast(pl.Float64).alias("score")
        )
        .unique(subset=["corp_code", "fy"], keep="last", maintain_order=True)
    )
    return o.join(s, on=["corp_code", "fy"], how="inner").select(
        "corp_code", "fy", "score", "event"
    )


def link_all(
    scores: pl.DataFrame, outcomes: dict[str, pl.DataFrame], col: str
) -> dict[str, pl.DataFrame]:
    return {o: link_score(scores, outcomes[o], col) for o in cj.OUTCOME_ORDER if o in outcomes}


def status_counts(outcomes: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """결과·연도·상태별 행 수(분모 구성 보고용)."""
    parts = []
    for o in cj.OUTCOME_ORDER:
        if o not in outcomes:
            continue
        parts.append(
            outcomes[o]
            .group_by("fy", "status")
            .agg(pl.len().cast(pl.Int64).alias("n"))
            .with_columns(pl.lit(o).alias("outcome"))
            .select("outcome", "fy", "status", "n")
        )
    return pl.concat(parts).sort("outcome", "fy", "status") if parts else pl.DataFrame()


# ================================================================ 5. 읽기 단계
@dataclass
class Sources:
    """읽기 단계 결과. 계산 단계는 이 프레임만 본다(합성 자료 시험용으로 분리)."""

    lake: Lake
    years: list[int]
    fs_all: pl.DataFrame  # 기준 재무 패널, 형성 FY2015~2025 전부
    misc: pl.DataFrame  # 배당·주식 패널(실행 연도)
    opinions: pl.DataFrame
    fs_prior_a: pl.DataFrame | None = None  # prior_source="t_minus_1_report" 패널(실행 연도)
    fs_latest: pl.DataFrame | None = None
    dps_latest: pl.DataFrame | None = None
    capevt_judgment: pl.DataFrame | None = None
    capevt_record: pl.DataFrame | None = None
    cov_fs: pl.DataFrame | None = None
    cov_misc: pl.DataFrame | None = None
    xbrl_cache: Path | None = None
    # 해석 표 대안 기록용 입력(없으면 해당 항목은 건너뜀)
    dps_multi: pl.DataFrame | None = (
        None  # 보통주 행이 여럿이고 값이 다른 보고서: corp_code, report_year, rcept_no, n_common_rows
    )
    opinions_anchor_off: pl.DataFrame | None = None  # audit_opinion_rows(anchor_max=False)
    misc_latest: pl.DataFrame | None = None  # misc_panel(as_of="latest")


def default_xbrl_cache(lake: Lake) -> Path:
    return (
        lake.output_dir(f"quality_score_company_cache_{lake.raw_snapshot}") / "xbrl_values.parquet"
    )


def ensure_xbrl_cache(lake: Lake, path: Path | None) -> Path:
    """XBRL 캐시 경로. 없으면 ``company_xbrl`` 로 만든다(느리다)."""
    p = Path(path) if path else default_xbrl_cache(lake)
    if not p.exists():
        from modeler.scores.quality import company_xbrl

        rc = company_xbrl.main(["--raw-snapshot", lake.raw_snapshot, "--out", str(p)])
        if rc != 0 or not p.exists():
            raise RuntimeError(f"XBRL 캐시를 만들지 못했습니다: {p}")
    return p


def raw_root_of(lake: Lake) -> Path:
    return Path(lake.root) / "kr" / "raw" / "raw_postgres"


def load_sources(lake: Lake, period: str, *, xbrl_cache: Path | None = None) -> Sources:
    """레이크를 읽어 ``Sources`` 를 만든다. 결과 변수는 만들지 않는다.

    재무 후보는 한 번만 읽는다(``load_inputs``). 기준 패널은 FY2015~2025 전부(통화·분모·의견 결측 비율용),
    misc는 실행 연도만 만든다.
    """
    years = list(PERIOD_YEARS[period])
    cache = ensure_xbrl_cache(lake, xbrl_cache)
    root = raw_root_of(lake)
    inp = fi.load_inputs(lake, CHECK_YEARS, xbrl_cache=cache)
    fs_all = fi.build_panel(inp, CHECK_YEARS)
    avail = filing_availability(lake, TradingCalendar.from_lake(lake))
    built = mi.build_misc(lake, years, raw_root=root, avail=avail)
    opinions = cm.audit_opinion_rows(root, lake.raw_snapshot)
    src = Sources(
        lake=lake, years=years, fs_all=fs_all, misc=built[0], opinions=opinions, xbrl_cache=cache
    )
    # 해석 표 대안 기록용 입력(판정 경로에는 안 쓴다)
    src.dps_multi = dps_multi_reports(root, lake.raw_snapshot)
    src.opinions_anchor_off = cm.audit_opinion_rows(root, lake.raw_snapshot, anchor_max=False)
    src.misc_latest = mi.build_misc(lake, years, raw_root=root, avail=avail, as_of="latest")[0]
    if period == "checks":
        src.cov_fs = fi.coverage_from(inp, CHECK_YEARS)
        src.cov_misc = mi.coverage_misc(lake, years, raw_root=root, built=built)
        return src
    src.fs_prior_a = fi.build_panel(inp, years, prior_source="t_minus_1_report")
    src.fs_latest = fi.fs_latest(lake, FS_LATEST_YEARS, inputs=inp)
    src.dps_latest = mi.dps_latest(lake, raw_root=root)
    src.capevt_judgment = cm.share_event_flags(root, lake.raw_snapshot)
    src.capevt_record = cm.share_event_flags(
        root, lake.raw_snapshot, sources=cm.RECORD_EVENT_SOURCES
    )
    return src


# ================================================================ 5b. 해석 표 대안 (기록용)
def dps_multi_reports(raw_root: Path, snap: str) -> pl.DataFrame:
    """보통주 행이 여럿이고 값이 다른 사업보고서(``common_multi_value``)의 목록(해석 표 D3 대안용).

    열: corp_code, report_year, rcept_no, n_common_rows."""
    d = cm.dividend_per_report(raw_root, snap)
    return d.filter(pl.col("common_multi_value")).select(
        "corp_code", pl.col("report_year").cast(pl.Int32), "rcept_no", "n_common_rows"
    )


def dps_null_for_multi(panel: pl.DataFrame, multi: pl.DataFrame) -> pl.DataFrame:
    """D3 문면판: dps_rcept_no가 multi 보고서인 행의 dps·dps_p1·dps_p2를 결측으로 둔다."""
    bad = pl.col("dps_rcept_no").is_in(multi["rcept_no"].to_list())
    return panel.with_columns(
        *[
            pl.when(bad).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("dps", "dps_p1", "dps_p2")
        ]
    )


def dps_latest_null_for_multi(dps_lat_full: pl.DataFrame, multi: pl.DataFrame) -> pl.DataFrame:
    """D3 문면판의 O3용 dps_latest: 같은 접수번호 행의 dps_t·dps_p1을 결측으로 둔다."""
    bad = pl.col("rcept_no").is_in(multi["rcept_no"].to_list())
    return dps_lat_full.with_columns(
        *[pl.when(bad).then(None).otherwise(pl.col(c)).alias(c) for c in ("dps_t", "dps_p1")]
    )


def d3_counts(src: Sources) -> dict:
    """D3 대상 개수(입력 존재 개수): multi 보고서 수·그 보고서의 보통주 행 수·패널·dps_latest에서 영향받은 수."""
    if src.dps_multi is None:
        return {"skipped": "dps_multi 입력 없음"}
    m = src.dps_multi
    ids = m["rcept_no"].to_list()
    p = src.misc.filter(pl.col("dps_rcept_no").is_in(ids))
    out = {
        "n_multi_reports": m.height,
        "n_multi_common_rows": int(m["n_common_rows"].sum() or 0),
        "n_panel_corp_years": p.height,
        "n_panel_corp_years_dps_non_null": int(p["dps"].is_not_null().sum()),
    }
    if src.dps_latest is not None:
        out["n_dps_latest_rows"] = int(src.dps_latest["rcept_no"].is_in(ids).sum())
    return out


def d5_counts(opinions: pl.DataFrame, off: pl.DataFrame | None) -> dict:
    """D5 영향 행 수(입력 존재 개수): 앵커를 끄면 ``fiscal_year`` 가 바뀌거나 결측이 되는 의견 행."""
    if off is None:
        return {"skipped": "opinions_anchor_off 입력 없음"}
    keys = ["corp_code", "rcept_no", "row_ordinal"]
    j = opinions.select(*keys, "fiscal_year", "opinion_class").join(
        off.select(*keys, pl.col("fiscal_year").alias("fy_off")), on=keys, how="left"
    )
    chg = j.filter(
        pl.col("fiscal_year").is_not_null() & (pl.col("fiscal_year") != pl.col("fy_off"))
    )
    nul = j.filter(pl.col("fiscal_year").is_not_null() & pl.col("fy_off").is_null())
    return {
        "n_rows": opinions.height,
        "n_rows_year_changed": chg.height,
        "n_rows_became_null": nul.height,
        "n_rows_became_null_with_opinion_class": int(nul["opinion_class"].is_not_null().sum()),
        "n_reports_affected": int(
            pl.concat([chg, nul]).select("rcept_no").n_unique() if chg.height + nul.height else 0
        ),
    }


MISC_VALUE_COLS = ("dps", "dps_p1", "dps_p2", "shares", "shares_p1", "retire", "retire_p1")


def c3_latest_counts(base: pl.DataFrame, latest: pl.DataFrame | None) -> dict:
    """C3 latest 영향 개수(입력 존재 개수): 열마다 값이 새로 생긴 corp-year 수와 값이 달라진 수."""
    if latest is None:
        return {"skipped": "misc_latest 입력 없음"}
    keys = ["corp_code", "fy"]
    j = base.select(*keys, *MISC_VALUE_COLS).join(
        latest.select(*keys, *[pl.col(c).alias(f"{c}__l") for c in MISC_VALUE_COLS]),
        on=keys,
        how="full",
        coalesce=True,
    )
    out: dict = {"n_corp_years_base": base.height, "n_corp_years_latest": latest.height}
    for c in MISC_VALUE_COLS:
        b, la = pl.col(c), pl.col(f"{c}__l")
        out[c] = {
            "n_filled_by_latest": int(j.filter(b.is_null() & la.is_not_null()).height),
            "n_value_changed": int(j.filter(b.is_not_null() & la.is_not_null() & (b != la)).height),
        }
    return out


def alt_input_counts(src: Sources) -> dict:
    """세 대안의 입력 존재 개수. checks는 이것만 낸다(결과 변수·AUC 없이)."""
    return {
        REC_D3: d3_counts(src),
        REC_D5: d5_counts(src.opinions, src.opinions_anchor_off),
        REC_C3_LATEST: c3_latest_counts(src.misc, src.misc_latest),
    }


# ================================================================ 6. 디버깅 표본 (dev)
def _opinion_pick(opinions: pl.DataFrame) -> pl.DataFrame:
    """(회사, fiscal_year)마다 ``opinion_by_year`` 가 읽는 보고서(가장 늦은 접수번호)와 그 의견 원문.

    열: corp_code, fiscal_year, rcept_no, raw — raw는 ``라벨:의견 원문(분류)`` 를 `` ; `` 로 이은 문자열."""
    keys = ["corp_code", "fiscal_year"]
    rows = opinions.filter(pl.col("opinion_class").is_not_null())
    last = (
        rows.sort(["report_year", "rcept_no"])
        .with_columns(pl.col("rcept_no").last().over(keys).alias("_last"))
        .filter(pl.col("rcept_no") == pl.col("_last"))
        .with_columns(
            (
                pl.col("label_raw").fill_null("")
                + ":"
                + pl.col("opinion_raw").fill_null("")
                + "("
                + pl.col("opinion_class")
                + ")"
            ).alias("_txt")
        )
    )
    return last.group_by(keys, maintain_order=True).agg(
        pl.col("rcept_no").first().alias("rcept_no"),
        pl.col("_txt").str.join(" ; ").alias("raw"),
    )


def _pick_sample(linked: pl.DataFrame, seed: int) -> pl.DataFrame:
    parts = []
    for ev, k in ((1, SAMPLE_EVENTS), (0, SAMPLE_NON_EVENTS)):
        d = linked.filter(pl.col("event") == ev).sort(["corp_code", "fy"])
        parts.append(d.sample(n=min(k, d.height), seed=seed, shuffle=True))
    return pl.concat(parts).select("corp_code", "fy", "event")


def build_dev_samples(
    *,
    panel: pl.DataFrame,
    scores: pl.DataFrame,
    outcomes: dict[str, pl.DataFrame],
    oi: OutcomeInputs,
    seed: int,
) -> pl.DataFrame:
    """결과마다 사건 최대 10건·비사건 최대 5건(고정 시드)과 그 결과에 쓴 입력값.

    점수(c·c1·c2·c3·f_sum)와 O1(t·t+1 의견 원문·접수번호)·O2(te·cap t·t+1)·
    O3(t+1 보고서 접수번호·dps 전기·당기)·O4(ni t·t+1)를 한 장에 둔다. 메인이 원문과 손으로 대조한다.
    개발 구간 전용이다.
    """
    keys = ["corp_code", "fy"]
    meta = (
        panel.select("corp_code", pl.col("fy").cast(pl.Int64), "stock_code")
        if "stock_code" in panel.columns
        else panel.select("corp_code", pl.col("fy").cast(pl.Int64)).with_columns(
            pl.lit(None, dtype=pl.String).alias("stock_code")
        )
    ).unique(subset=keys, keep="first", maintain_order=True)
    sc = scores.select("corp_code", pl.col("fy").cast(pl.Int64), "c", "c1", "c2", "c3", "f_sum")
    pick = _opinion_pick(oi.opinions)
    fsl = (oi.fs_latest_full if oi.fs_latest_full is not None else oi.fs_latest).with_columns(
        pl.col("year").cast(pl.Int64)
    )
    dps = oi.dps_latest_full if oi.dps_latest_full is not None else oi.dps_latest
    fm = oi.formation.select("corp_code", "fy", "te", "cap", "ni")

    def _op(d: pl.DataFrame, shift: int, tag: str) -> pl.DataFrame:
        j = pick.select(
            "corp_code",
            (pl.col("fiscal_year").cast(pl.Int64) - shift).alias("fy"),
            pl.col("raw").alias(f"op_{tag}_raw"),
            pl.col("rcept_no").alias(f"op_{tag}_rcept"),
        )
        return d.join(j, on=keys, how="left")

    parts = []
    for o in cj.OUTCOME_ORDER:
        if o not in outcomes:
            continue
        linked = link_score(scores, outcomes[o], "c")
        if linked.height == 0:
            continue
        d = (
            _pick_sample(linked, seed)
            .join(outcomes[o].select(*keys, "status"), on=keys, how="left")
            .join(meta, on=keys, how="left")
            .join(sc, on=keys, how="left")
            .with_columns(pl.lit(o).alias("outcome"))
        )
        if o == "O1":
            d = _op(_op(d, 0, "t"), 1, "t1")
        elif o == "O2":
            nxt = fsl.select(
                "corp_code",
                (pl.col("year") - 1).alias("fy"),
                pl.col("te").alias("te_t1"),
                pl.col("cap").alias("cap_t1"),
                *([pl.col("rcept_no").alias("fs_t1_rcept")] if "rcept_no" in fsl.columns else []),
            ).unique(subset=keys, keep="last", maintain_order=True)
            d = d.join(
                fm.select(*keys, pl.col("te").alias("te_t"), pl.col("cap").alias("cap_t")),
                on=keys,
                how="left",
            ).join(nxt, on=keys, how="left")
        elif o == "O3":
            r1 = (
                dps.with_columns(pl.col("report_year").cast(pl.Int64))
                .unique(subset=["corp_code", "report_year"], keep="last", maintain_order=True)
                .select(
                    "corp_code",
                    (pl.col("report_year") - 1).alias("fy"),
                    *(
                        [pl.col("rcept_no").alias("dps_rcept_t1")]
                        if "rcept_no" in dps.columns
                        else []
                    ),
                    pl.col("dps_p1").alias("dps_prev"),
                    pl.col("dps_t").alias("dps_cur"),
                    pl.col("currency").alias("dps_ccy"),
                )
            )
            d = d.join(r1, on=keys, how="left")
        elif o == "O4":
            nxt = fsl.select(
                "corp_code", (pl.col("year") - 1).alias("fy"), pl.col("ni").alias("ni_t1")
            ).unique(subset=keys, keep="last", maintain_order=True)
            d = d.join(fm.select(*keys, pl.col("ni").alias("ni_t")), on=keys, how="left").join(
                nxt, on=keys, how="left"
            )
        parts.append(d)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


# ================================================================ 7. 기록용 비교 (§9, 정정 C-1R·C-2)
@dataclass
class Ctx:
    src: Sources
    period: str
    years: list[int]
    panel: pl.DataFrame
    scores: pl.DataFrame
    oi: OutcomeInputs
    outcomes: dict[str, pl.DataFrame]
    o1_mode: str
    demoted_t: list[int]
    seed: int
    n_boot: int


def _cells(ctx: Ctx, named: dict[str, dict[str, pl.DataFrame]]) -> dict:
    return cj.record_cells(named, seed=ctx.seed, n_boot=ctx.n_boot)


def _universe_variants(panel: pl.DataFrame) -> dict[str, pl.Expr]:
    """분모 규칙을 한 조건만 푼 in_universe 식. 시장·SPAC·금융업·12월 결산 네 조건 중 하나를 푼다."""
    mk = pl.col("market").is_in(list(fi.MARKETS)).fill_null(False)
    spac = ~pl.col("is_spac").fill_null(False)
    fin = ~pl.col("is_financial").fill_null(False)
    dec = (pl.col("acc_mt") == fi.DEC_CLOSING).fill_null(fi.CI_ACC_NULL_IS_DEC)
    return {
        "universe_financial_included": mk & spac & dec,
        "universe_non_december_included": mk & spac & fin,
        "universe_spac_included": mk & fin & dec,
    }


def compute_records(ctx: Ctx) -> dict:
    """기록용 비교(판정에 안 씀). 항목마다 ``record_cells`` 형식(차원 → 결과 → AUC·95% 구간)."""
    src, years, oi = ctx.src, ctx.years, ctx.oi
    dev = ctx.period == "dev"
    rec: dict = {}

    # (a1) 전기 값 출처: t−1 보고서 판본 값으로 만든 점수
    if src.fs_prior_a is not None:
        sc_a = cs.compute_scores(assemble_panel(src.fs_prior_a, src.misc))
        rec["prior_t_minus_1_report"] = _cells(
            ctx,
            {
                "c": link_all(sc_a, ctx.outcomes, "c"),
                "f_sum": link_all(sc_a, ctx.outcomes, "f_sum"),
            },
        )

    # (a2) 사건 해 기록용(액면 변경 해 포함, 정정 C-2): 점수·F와 O3
    if src.capevt_record is not None:
        fs_main = src.fs_all.filter(pl.col("fy").is_in(years))
        sc_r = cs.compute_scores(assemble_panel(fs_main, src.misc, capevt="record"))
        out_r = dict(ctx.outcomes)
        out_r["O3"] = co.o3(oi.formation, oi.dps_latest, src.capevt_record, years)
        rec["capevt_record"] = _cells(
            ctx, {"c": link_all(sc_r, out_r, "c"), "f_sum": link_all(sc_r, out_r, "f_sum")}
        )
        cnt = lambda df: int((df["status"] == co.EXCLUDED_CAPEVT).sum())  # noqa: E731
        rec["capevt_record"]["o3_excluded_capevt"] = {
            "judgment_list": cnt(ctx.outcomes["O3"]),
            "record_list": cnt(out_r["O3"]),
        }

    sc = ctx.scores
    # (b) 순환 성분을 뺀 합성: c_o2는 O2, c_o3는 O3, c_o4는 O4와
    rec["circular_removed"] = _cells(
        ctx,
        {
            "c_o2": {"O2": link_score(sc, ctx.outcomes["O2"], "c_o2")},
            "c_o3": {"O3": link_score(sc, ctx.outcomes["O3"], "c_o3")},
            "c_o4": {"O4": link_score(sc, ctx.outcomes["O4"], "c_o4")},
        },
    )
    # (c) 두 차원 이상 합성 · (d) 장기차입금 값 필수 F
    rec["two_dims"] = _cells(ctx, {"c_2dim": link_all(sc, ctx.outcomes, "c_2dim")})
    rec["f_ltb_required"] = _cells(ctx, {"f_ltb_req": link_all(sc, ctx.outcomes, "f_ltb_req")})

    # (e) O1 폴백 · O2 완전 잠식 · O4 완화판
    o1_fb = co.o1(oi.formation, oi.opinions, years, mode="fallback")
    rec["o1_fallback"] = _cells(ctx, {"c": {"O1": link_score(sc, o1_fb, "c")}})
    # CI-d 대안: 한 해의 의견을 가장 늦은 보고서 대신 모든 보고서에서 비적정 하나라도로 접는다
    o1_any = co.o1(oi.formation, oi.opinions, years, mode=ctx.o1_mode, fold_mode="any_report")
    o1_any = o1_any.filter(~pl.col("fy").is_in(list(ctx.demoted_t)))
    rec["o1_fold_any_report"] = _cells(ctx, {"c": {"O1": link_score(sc, o1_any, "c")}})
    o2_full = co.o2(oi.formation, oi.fs_latest, years, kind="full")
    rec["o2_full"] = _cells(ctx, {"c": {"O2": link_score(sc, o2_full, "c")}})
    if dev:
        rec["o4_relaxed"] = {
            "skipped": "개발 구간은 O4 t+1만 본다(§5.4). t+3이 필요한 완화판은 건너뜀"
        }
    else:
        o4_rel = co.o4_relaxed(
            oi.formation, oi.fs_latest, [y for y in years if y in O4_JUDGMENT_YEARS]
        )
        rec["o4_relaxed"] = _cells(ctx, {"c": {"O4": link_score(sc, o4_rel, "c")}})
    if "O1_demoted" in ctx.outcomes:
        rec["o1_even_year_demoted"] = {
            "formation_years": list(ctx.demoted_t),
            **_cells(ctx, {"c": {"O1": link_score(sc, ctx.outcomes["O1_demoted"], "c")}}),
        }

    # (f) vintage 단독 민감도: layer=="vintage"인 행만 점수 풀에 둔다
    pv = ctx.panel.with_columns(
        (pl.col("in_universe") & (pl.col("layer") == "vintage")).alias("in_universe")
    )
    sc_v = cs.compute_scores(pv)
    rec["vintage_only"] = _cells(ctx, {"c": link_all(sc_v, ctx.outcomes, "c")})

    # (g) 금융업 포함·비12월 포함·SPAC 포함: 그 조건만 풀어 점수와 결과를 다시 만든다
    for name, expr in _universe_variants(ctx.panel).items():
        pvar = ctx.panel.with_columns(expr.alias("in_universe"))
        sc_x = cs.compute_scores(pvar)
        out_x = compute_outcomes(
            replace(oi, formation=formation_frame(pvar)),
            years,
            period=ctx.period,
            o1_mode=ctx.o1_mode,
            demoted_t=ctx.demoted_t,
        )
        rec[name] = _cells(ctx, {"c": link_all(sc_x, out_x, "c")})

    # (h) 해석 표 대안 셋(기록용, 판정 규칙은 그대로)
    rec.update(_interpretation_alts(ctx))
    return rec


def _interpretation_alts(ctx: Ctx) -> dict:
    """해석 표 D3·D5 대안(문면판)과 C3 latest(PIT 위반) 기록. 입력이 없으면 ``skipped``."""
    src, years, oi, sc = ctx.src, ctx.years, ctx.oi, ctx.scores
    fs_main = src.fs_all.filter(pl.col("fy").is_in(years))
    out: dict = {}

    # D3: 보통주 행이 여럿이고 값이 다른 보고서의 DPS를 결측으로 둔 판
    if src.dps_multi is None:
        out[REC_D3] = {"skipped": "dps_multi 입력 없음"}
    else:
        panel_d3 = dps_null_for_multi(ctx.panel, src.dps_multi)
        sc_d3 = cs.compute_scores(panel_d3)
        out_d3 = dict(ctx.outcomes)
        full = oi.dps_latest_full if oi.dps_latest_full is not None else None
        if full is not None and "rcept_no" in full.columns:
            lat = dps_latest_null_for_multi(full, src.dps_multi).select(oi.dps_latest.columns)
            out_d3["O3"] = co.o3(oi.formation, lat, oi.capevt, years)
        out[REC_D3] = {
            "n_affected": d3_counts(src),
            **_cells(
                ctx, {"c": link_all(sc_d3, out_d3, "c"), "f_sum": link_all(sc_d3, out_d3, "f_sum")}
            ),
        }

    # D5: 감사의견 최대 기수 앵커를 끈 판(점수는 기본 그대로, O1과 짝수 해 규칙만 다시)
    if src.opinions_anchor_off is None:
        out[REC_D5] = {"skipped": "opinions_anchor_off 입력 없음"}
    else:
        off = src.opinions_anchor_off
        all_uni = (
            src.fs_all.filter(pl.col("in_universe").fill_null(False))
            .select("corp_code", pl.col("fy").cast(pl.Int64).alias("year"))
            .unique()
        )
        rates, flags = opinion_rates_and_flags(all_uni, off)
        demoted = demoted_formation_years(flags, ctx.o1_mode)
        o1_off = co.o1(oi.formation, off, years, mode=ctx.o1_mode)
        o1_off = o1_off.filter(~pl.col("fy").is_in(demoted))
        out[REC_D5] = {
            "n_affected": d5_counts(src.opinions, off),
            "demoted_formation_years": demoted,
            "default_demoted_formation_years": list(ctx.demoted_t),
            "opinion_missing_rate": rates.to_dicts(),
            "even_year_flags": flags.to_dicts(),
            **_cells(ctx, {"c": {"O1": link_score(sc, o1_off, "c")}}),
        }

    # C3 latest: B_t 기준일 없이 가장 늦은 판본(PIT 위반 — 판정에 쓰지 않는 결과 전 기록)
    if src.misc_latest is None:
        out[REC_C3_LATEST] = {"skipped": "misc_latest 입력 없음", "pit_violation": True}
    else:
        sc_l = cs.compute_scores(assemble_panel(fs_main, src.misc_latest, capevt="judgment"))
        out[REC_C3_LATEST] = {
            "pit_violation": True,
            "note": C3_LATEST_NOTE,
            "n_affected": c3_latest_counts(src.misc, src.misc_latest),
            **_cells(
                ctx,
                {"c": link_all(sc_l, ctx.outcomes, "c"), "c3": link_all(sc_l, ctx.outcomes, "c3")},
            ),
        }
    return out


# ================================================================ 8. dev·judgment 분석
def analyze(src: Sources, period: str, *, o1_info: dict, seed: int, n_boot: int) -> dict:
    """점수 → 결과 → 판정 통계·16칸·기록용 비교. dev·judgment 공통.

    판정 구간 보호는 ``link_score``·결과 함수의 ``guard_years`` 가 맡는다(§7.2)."""
    if period not in ("dev", "judgment"):
        raise ValueError(f"analyze는 dev·judgment만 받습니다: {period!r}")
    years = list(PERIOD_YEARS[period])
    fs_main = src.fs_all.filter(pl.col("fy").is_in(years))
    panel = assemble_panel(fs_main, src.misc, capevt="judgment")
    scores = cs.compute_scores(panel)
    oi = outcome_inputs(
        panel,
        fs_lat=src.fs_latest,
        dps_lat=src.dps_latest,
        opinions=src.opinions,
        capevt=src.capevt_judgment,
        currency_panel=src.fs_all,
    )
    all_uni = (
        src.fs_all.filter(pl.col("in_universe").fill_null(False))
        .select("corp_code", pl.col("fy").cast(pl.Int64).alias("year"))
        .unique()
    )
    rates, flags = opinion_rates_and_flags(all_uni, src.opinions)
    demoted = demoted_formation_years(flags, o1_info["mode"])
    outcomes = compute_outcomes(
        oi, years, period=period, o1_mode=o1_info["mode"], demoted_t=demoted
    )
    frames = link_all(scores, outcomes, "c")
    judged = cj.judge_outcomes(frames, seed=seed, n_boot=n_boot)
    cells = cj.record_cells(
        {dim: link_all(scores, outcomes, col) for dim, col in CELL_DIMS.items()},
        seed=seed,
        n_boot=n_boot,
    )
    ctx = Ctx(
        src, period, years, panel, scores, oi, outcomes, o1_info["mode"], demoted, seed, n_boot
    )
    records = compute_records(ctx)
    records["f_vs_composite"] = {
        o: {
            "c": judged["outcomes"][o]["stats"]["pooled_auc"],
            "f_sum": cells["F"][o]["auc"] if "F" in cells and o in cells["F"] else None,
        }
        for o in judged["order"]
    }
    scored = scores.filter(pl.col("in_universe") & pl.col("c").is_not_null()).select(
        "corp_code", "fy"
    )
    unobserved = co.unobserved_scored_counts(
        {o: outcomes[o] for o in cj.OUTCOME_ORDER if o in outcomes}, scored
    )
    result = {
        "version": RUN_VERSION,
        "period": period,
        "years": years,
        "o4_definition": (
            "t+1 (개발 디버깅, §5.4)" if period == "dev" else "t+1~t+3 중 둘 이상 적자"
        ),
        "o1": {**o1_info, "demoted_formation_years": demoted},
        "judge": judged,
        "cells": cells,
        "records": records,
        "status_counts": status_counts(outcomes).to_dicts(),
        "unobserved_scored": unobserved.to_dicts(),
        "opinion_missing_rate": rates.to_dicts(),
        "even_year_flags": flags.to_dicts(),
        "currency_note": oi.currency_note,
        "ci": CI_NOTES,
    }
    samples = (
        build_dev_samples(panel=panel, scores=scores, outcomes=outcomes, oi=oi, seed=seed)
        if period == "dev"
        else None
    )
    return {"result": result, "samples": samples}


def _auc_row(name: str, o: str, row: dict) -> dict:
    lo, hi = row["ci95"]
    return {
        "item": name,
        "outcome": o,
        "auc": row["auc"],
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n_rows": row["n_rows"],
        "n_events": row["n_events"],
        "n_boot_nan": row["n_boot_nan"],
    }


def _flatten_cells(prefix: str, cells: dict) -> list[dict]:
    out = []
    for dim, per in cells.items():
        if not isinstance(per, dict):
            continue
        for o, row in per.items():
            if o in cj.OUTCOME_ORDER and isinstance(row, dict) and "auc" in row:
                out.append(_auc_row(f"{prefix}{dim}", o, row))
    return out


def result_tables(result: dict) -> dict[str, pl.DataFrame]:
    """결과 JSON을 사람이 읽는 TSV 표로 편다."""
    j = result["judge"]
    rows = []
    byyear = []
    for o in j["order"]:
        b = j["outcomes"][o]
        s, g = b["stats"], b["gates"]
        rows.append(
            {
                "outcome": o,
                "n_rows": s["n_rows"],
                "n_events": s["n_events"],
                "base_rate": s["base_rate"],
                "pooled_auc": s["pooled_auc"],
                "ci95_lo": s["bootstrap"]["ci95"][0],
                "ci95_hi": s["bootstrap"]["ci95"][1],
                "p_value": s["bootstrap"]["p_value"],
                "lift20": s["capture"]["20"]["lift"],
                "capture20": s["capture"]["20"]["capture"],
                "g1": g["g1"]["status"],
                "g2": g["g2"]["status"],
                "g3": g["g3"]["pass"],
                "g4": g["g4"]["pass"],
                "grade": b["grade"],
                "grade_rule": b["grade_rule"],
            }
        )
        for y in s["by_year"]:
            byyear.append({"outcome": o, **y})
    recs = _flatten_cells("cell:", result["cells"])
    for name, rec in result["records"].items():
        if isinstance(rec, dict):
            recs += _flatten_cells(
                f"{name}:", {k: v for k, v in rec.items() if isinstance(v, dict)}
            )
    return {
        "outcomes_summary.tsv": pl.DataFrame(rows),
        "by_year.tsv": pl.DataFrame(byyear),
        "cells_and_records.tsv": pl.DataFrame(recs),
        "status_counts.tsv": pl.DataFrame(result["status_counts"]),
        "unobserved_scored.tsv": pl.DataFrame(result["unobserved_scored"]),
    }


# ================================================================ 9. checks (입력 존재만)
def score_input_counts(scores: pl.DataFrame) -> pl.DataFrame:
    """연도별 점수 입력이 있는 회사 수(분모 안, non-null 수만). 값·분포는 내지 않는다."""
    u = scores.filter(pl.col("in_universe"))
    return (
        u.group_by("fy")
        .agg(
            pl.len().cast(pl.Int64).alias("n_universe_rows"),
            *[
                pl.col(c).is_not_null().sum().cast(pl.Int64).alias(f"n_{c}")
                for c in SCORE_COUNT_COLS
            ],
        )
        .sort("fy")
    )


def prereg_numbers(cov_fs: pl.DataFrame, cov_misc: pl.DataFrame) -> dict:
    """사전등록이 manifest에 적으라 한 입력 존재 숫자(§5.1·§5.2·§12.4 C). 사건 수가 아니다.

    * ``raw_only_corp_years``: vintage에 없고 raw에만 있는 corp-year(scope all·in_universe)
    * ``no_origin_keys``: 원본(vintage) 없는 키, ``no_version_at_base``: B_t까지 가용한 판본이 없는 행
    * ``key_raw_only*``: raw에만 있는 키 중 정정본·B_t 뒤 접수(정정 결측 수)
    * misc: ``dps_null_share``(stock_knd 규칙 적용 뒤 DPS 결측 비율)·``n_*_after_base``(판본이 B_t 뒤)
    """
    out: dict = {}
    for item in (
        "raw_only_corp_years",
        "no_origin_keys",
        "no_origin_keys:revision_only",
        "no_version_at_base",
        "no_version_at_base:missing_receipt",
        "key_raw_only",
        "key_raw_only:revision",
        "key_raw_only:after_base",
        "key_raw_only:revision_after_base",
    ):
        d = cov_fs.filter(pl.col("item") == item)
        out[item] = {
            sc: {str(int(fy)): int(n) for fy, n in g.select("fy", "n").iter_rows()}
            for (sc,), g in d.group_by("scope")
        }
    for col in (
        "n_rows",
        "dps_null_share",
        "n_dps_null",
        "n_dps_after_base",
        "n_shares_after_base",
        "n_retire_after_base",
    ):
        if col in cov_misc.columns:
            out[f"misc:{col}"] = {
                str(int(fy)): v for fy, v in cov_misc.select("fy", col).iter_rows()
            }
    return out


def run_checks(src: Sources, *, o1_info: dict) -> dict[str, object]:
    """입력 존재 확인(§7.2 (a)). 결과 변수 함수·판정 통계는 부르지 않는다.

    점수 입력은 ``compute_scores`` 를 돌려 열별 non-null 수만 센다(CI-checks-scores)."""
    years = list(CHECK_YEARS)
    guard_years(years, "inputs")
    fs_main = src.fs_all.filter(pl.col("fy").is_in(years))
    panel = assemble_panel(fs_main, src.misc, capevt="judgment")
    scores = cs.compute_scores(panel)
    all_uni = (
        panel.filter(pl.col("in_universe").fill_null(False))
        .select("corp_code", pl.col("fy").cast(pl.Int64).alias("year"))
        .unique()
    )
    rates, flags = opinion_rates_and_flags(all_uni, src.opinions)
    tables: dict[str, pl.DataFrame] = {
        "score_inputs.tsv": score_input_counts(scores),
        "opinion_missing_rate.tsv": rates,
        "even_year_flags.tsv": flags,
    }
    nums: dict = {}
    if src.cov_fs is not None:
        tables["coverage_fs.tsv"] = src.cov_fs
    if src.cov_misc is not None:
        tables["coverage_misc.tsv"] = src.cov_misc
    if src.cov_fs is not None and src.cov_misc is not None:
        nums = prereg_numbers(src.cov_fs, src.cov_misc)
    return {
        "tables": tables,
        "numbers": nums,
        "alt_counts": alt_input_counts(src),
        "o1": {
            **o1_info,
            "demoted_formation_years": demoted_formation_years(flags, o1_info["mode"]),
        },
        "years": years,
    }


# ================================================================ 10. manifest (§12.2)
def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / (1 << 20) if sys.platform == "darwin" else ru / 1024, 1)


def table_info(d: Path) -> dict:
    """입력 표 하나: 파일 수·바이트·(상대경로:크기 목록)의 sha256."""
    if not d.is_dir():
        return {"path": str(d), "exists": False, "files": 0, "bytes": 0, "listing_sha256": None}
    files = sorted(d.rglob("*.parquet"))
    h = hashlib.sha256()
    total = 0
    for f in files:
        size = f.stat().st_size
        total += size
        h.update(f"{f.relative_to(d).as_posix()}:{size}\n".encode())
    return {
        "path": str(d),
        "exists": True,
        "files": len(files),
        "bytes": total,
        "listing_sha256": h.hexdigest(),
    }


def build_manifest(
    *,
    period: str,
    lake: Lake,
    args: dict,
    o1_info: dict,
    even_flags: list[dict],
    xbrl_cache: Path | None,
    prereg: Path | None,
    t0: float,
    outputs: dict[str, Path],
    extra: dict | None = None,
) -> dict:
    """§12.2: snapshot 이름·입력 표 파일 수/바이트/목록 해시·XBRL 캐시 해시·코드 해시·modeler HEAD·
    사전등록 해시·시드·B·``uv.lock`` 해시·CLI 인자·O1 모드와 근거·짝수 해 플래그·실행 시간·메모리."""
    here = Path(__file__).resolve()
    repo = here.parents[4]
    prereg_p = Path(prereg) if prereg else None
    uv = repo / "uv.lock"
    man = {
        "version": RUN_VERSION,
        "period": period,
        "raw_snapshot": lake.raw_snapshot,
        "derived_snapshot": lake.derived_snapshot,
        "inputs": {
            **{t: table_info(lake.raw_dir(t)) for t in RAW_TABLES},
            **{t: table_info(lake.derived_dir(t)) for t in DERIVED_TABLES},
        },
        "xbrl_cache": {
            "path": str(xbrl_cache) if xbrl_cache else None,
            "sha256": sha256_file(xbrl_cache) if xbrl_cache and Path(xbrl_cache).exists() else None,
        },
        "code_sha256": {p.name: sha256_file(p) for p in sorted(here.parent.glob("company_*.py"))},
        "modeler_git_head": git_head(repo),
        "prereg": {
            "path": str(prereg_p) if prereg_p else None,
            "sha256": sha256_file(prereg_p) if prereg_p and prereg_p.exists() else None,
        },
        "bootstrap": {"seed": args.get("seed"), "n_boot": args.get("n_boot")},
        "uv_lock_sha256": sha256_file(uv) if uv.exists() else None,
        "args": args,
        "o1": o1_info,
        "even_year_flags": even_flags,
        "outputs": {
            k: {"path": str(p), "sha256": sha256_file(p)} for k, p in outputs.items() if p.exists()
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "peak_rss_mb": _peak_rss_mb(),
    }
    if extra:
        man.update(extra)
    return man


# ================================================================ 11. 실행
def _write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(cj._jsonable(obj), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def default_out_dir(lake: Lake, period: str) -> Path:
    return lake.output_dir(f"quality_score_company_{period}_{lake.raw_snapshot}")


def run(
    period: str,
    lake: Lake,
    out_dir: str | Path | None = None,
    *,
    seed: int = cj.SEED,
    n_boot: int = cj.N_BOOT,
    o1_mode: str = "auto",
    xbrl_cache: str | Path | None = None,
    prereg: str | Path | None = None,
    args: dict | None = None,
    sources: Sources | None = None,
) -> dict:
    """한 기간을 실행하고 ``out_dir`` 에 결과를 쓴다. 쓴 파일 경로 dict를 돌려준다.

    ``sources`` 를 주면 레이크를 읽지 않는다(합성 자료 시험용).
    """
    if period not in PERIODS:
        raise ValueError(f"period는 {PERIODS} 중 하나입니다: {period!r}")
    # §7.2: 판정 구간은 아무 파일도 쓰기 전에, 읽기 전에 멈춘다.
    if period == "judgment" and not os.environ.get(JUDGMENT_ENV):
        raise PermissionError(
            f"판정 구간 실행은 사전등록 동결 확인 뒤 환경변수 {JUDGMENT_ENV}를 설정해야 합니다 (§7.2)."
        )
    guard_years(PERIOD_YEARS[period], "inputs" if period == "checks" else "link")
    t0 = time.time()
    out = Path(out_dir) if out_dir else default_out_dir(lake, period)
    src = (
        sources
        if sources is not None
        else load_sources(lake, period, xbrl_cache=Path(xbrl_cache) if xbrl_cache else None)
    )
    o1_info = decide_o1_mode(src.opinions, o1_mode)
    cli = dict(args or {"period": period, "seed": seed, "n_boot": n_boot, "o1_mode": o1_mode})
    cli.setdefault("seed", seed)
    cli.setdefault("n_boot", n_boot)

    written: dict[str, Path] = {}
    extra: dict = {}
    if period == "checks":
        res = run_checks(src, o1_info=o1_info)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in res["tables"].items():
            written[name] = out / name
            cm.write_tsv(df, written[name])
        written["prereg_numbers.json"] = out / "prereg_numbers.json"
        _write_json(written["prereg_numbers.json"], res["numbers"])
        written["interp_alt_counts.json"] = out / "interp_alt_counts.json"
        _write_json(written["interp_alt_counts.json"], res["alt_counts"])
        even = res["tables"]["even_year_flags.tsv"].to_dicts()
        o1_final = res["o1"]
    else:
        res = analyze(src, period, o1_info=o1_info, seed=seed, n_boot=n_boot)
        out.mkdir(parents=True, exist_ok=True)
        written["result.json"] = out / "result.json"
        _write_json(written["result.json"], res["result"])
        for name, df in result_tables(res["result"]).items():
            written[name] = out / name
            cm.write_tsv(df, written[name])
        if res["samples"] is not None:
            written["dev_samples.tsv"] = out / "dev_samples.tsv"
            cm.write_tsv(res["samples"], written["dev_samples.tsv"])
        even = res["result"]["even_year_flags"]
        o1_final = res["result"]["o1"]
    man = build_manifest(
        period=period,
        lake=lake,
        args=cli,
        o1_info=o1_final,
        even_flags=even,
        xbrl_cache=src.xbrl_cache,
        prereg=Path(prereg) if prereg else default_prereg(),
        t0=t0,
        outputs=written,
        extra=extra,
    )
    written["manifest.json"] = out / "manifest.json"
    _write_json(written["manifest.json"], man)
    return {"out_dir": out, "files": written, "manifest": man, "result": res}


# ================================================================ 12. CLI
def _summary(period: str, r: dict) -> str:
    lines = [f"출력: {r['out_dir']}"]
    man = r["manifest"]
    lines.append(
        f"O1 모드: {man['o1']['mode']} (짝수 해 보고서 {man['o1']['n_even_year_reports']}건)"
    )
    lines.append(f"실행 {man['elapsed_sec']}초, 최대 메모리 {man['peak_rss_mb']}MB")
    if period != "checks":
        j = r["result"]["result"]["judge"]
        for o in j["order"]:
            b = j["outcomes"][o]
            s = b["stats"]
            lines.append(
                f"{o}: 사건 {s['n_events']}/{s['n_rows']} AUC {s['pooled_auc']:.3f} "
                f"p {s['bootstrap']['p_value']:.4f} 등급(참고) {b['grade']}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--period", choices=PERIODS, default="dev")
    ap.add_argument("--raw-snapshot", default="2026-09-30")
    ap.add_argument("--derived-snapshot", default="2026-09-29")
    ap.add_argument("--xbrl-cache", default=None)
    ap.add_argument("--o1-mode", choices=("auto", "full", "fallback"), default="auto")
    ap.add_argument("--seed", type=int, default=cj.SEED)
    ap.add_argument("--n-boot", type=int, default=cj.N_BOOT)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prereg", default=None)
    a = ap.parse_args(argv)
    lake = Lake.from_env(raw_snapshot=a.raw_snapshot, derived_snapshot=a.derived_snapshot)
    try:
        r = run(
            a.period,
            lake,
            a.out_dir,
            seed=a.seed,
            n_boot=a.n_boot,
            o1_mode=a.o1_mode,
            xbrl_cache=a.xbrl_cache,
            prereg=a.prereg,
            args=vars(a),
        )
    except PermissionError as e:
        print(f"거부: {e}", file=sys.stderr)
        return 2
    print(_summary(a.period, r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
