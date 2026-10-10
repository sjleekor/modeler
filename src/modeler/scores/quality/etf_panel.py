"""ETF 일별 패널 모듈과 입력 존재 확인 (사전등록 20261010_quality_score §11.1~11.3, §12.4 E).

E1(상장 유지 위험)·E2(거래 품질)·테마 카드 코드가 이 모듈만 보고 짤 수 있게 재료만 만든다.
**점수(백분위·E1·E2)를 계산하지 않고, 점수와 사건을 잇지도 않는다.** 사건 목록과 입력 개수만 센다.

    STOCK_DATA_ROOT=../stock_data python -m modeler.scores.quality.etf_panel --checks

경로는 환경변수로 받는다(기본 ``../stock_data``). 산출물은 ``stock_data/`` 아래
``kr/output/quality_score_etf_inputs_20261010/panel_checks`` 에만 쓴다.

정의(전부 사전등록 원문 그대로, 동결 파서 ``etf_classify`` 의 거래일 정의와 같다):

- 거래일 = 종가(``TDD_CLSPRC``)가 있는 행이 하나라도 있는 날. 그 밖의 평일(휴장일) 행은 버린다.
- ETF의 첫/마지막 거래일 = 그 ETF의 종가가 있는 첫/마지막 날.
- 사건 상태: 마지막 거래일 뒤 자료 끝까지의 거래일 수 k. k=0 listed, 1..20 pending(확정 보류),
  k>20 disappeared(연속 10거래일 이상 안 나타나고 다시 안 나타남을 자동으로 만족).
- 월말 거래일 = 달력 월마다 마지막 거래일.
- 월말 공식 순자산 = 월말 거래일의 ``INVSTASST_NETASST_TOTAMT``. 0은 누락. 0이거나 없으면 그 월말
  직전 5거래일 안의 0 아닌 값(가장 가까운 날), 없으면 null. ``NAV × 상장좌수`` 로 채우지 않는다.

분류 칸(만기형·연금 부적격·국내/해외·액티브·비교 그룹 키)은 ``etf_classify.classification_table``
결과를 그대로 붙인다. 이 파일은 분류 규칙을 새로 정하지 않는다.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from modeler.scores.quality import etf_classify as ec

DEFAULT_INPUT_REL = "kr/output/etf_krx_research_pull_20261009/etf_daily_2010_20261008.csv.gz"
DEFAULT_OUT_REL = "kr/output/quality_score_etf_inputs_20261010/panel_checks"

PANEL_VERSION = "quality-score-v0/etf_panel/1"

NUMERIC_COLS = [
    "TDD_CLSPRC",
    "CMPPREVDD_PRC",
    "FLUC_RT",
    "NAV",
    "TDD_OPNPRC",
    "TDD_HGPRC",
    "TDD_LWPRC",
    "ACC_TRDVOL",
    "ACC_TRDVAL",
    "MKTCAP",
    "INVSTASST_NETASST_TOTAMT",
    "LIST_SHRS",
    "OBJ_STKPRC_IDX",
    "CMPPREVDD_IDX",
    "FLUC_RT_IDX",
]

# 처음 고른 값(사전등록). 코드에 이름을 붙여 둔다.
PENDING_MAX_DAYS = 20  # §11.3: 마지막 거래일이 자료 끝에서 20거래일 안이면 확정 보류
NETASST_LOOKBACK_DAYS = 5  # §11.1: 월말 순자산이 0이면 직전 5거래일 안의 값
LISTING_MONTHS = 12  # §11.2: 상장 1년 이상
GAP_LONG_DAYS = 10  # §11.3: 연속 10거래일 안 나타남
PRE_DELIST_DAYS = 30  # §12.4 E: 폐지 직전 30거래일


# ---------------------------------------------------------------- 읽기
@dataclass
class Panel:
    """거래일 행만 남긴 일별 패널과 달력.

    rows: 거래일에 속한 행(ETF-일). ``day_idx`` 는 거래일 달력의 0부터 번호.
    calendar: 거래일 표(``date``, ``day_idx``, ``is_month_end``).
    """

    rows: pl.DataFrame
    calendar: pl.DataFrame
    n_rows_raw: int
    n_weekdays_in_file: int
    parse_issues: dict[str, int]

    @property
    def n_days(self) -> int:
        return self.calendar.height

    @property
    def end_idx(self) -> int:
        return self.n_days - 1

    @property
    def end_date(self) -> date:
        return self.calendar["date"][-1]

    @property
    def n_closed_days(self) -> int:
        return self.n_weekdays_in_file - self.n_days

    @property
    def month_ends(self) -> pl.DataFrame:
        """달력 월마다 마지막 거래일(``date``, ``day_idx``)."""
        return self.calendar.filter(pl.col("is_month_end")).select("date", "day_idx")


def read_panel(path: str | Path) -> Panel:
    """CSV(gz)를 읽어 숫자 칸 Float64, ``BAS_DD`` Date로 바꾸고 휴장일 행을 버린다.

    ``ISU_CD`` 는 영숫자가 섞이니 문자열로 둔다. 숫자 칸의 빈 값·``-`` 는 null.
    """
    raw = pl.read_csv(str(path), infer_schema_length=0, null_values=[""])
    n_raw = raw.height
    issues = {}
    for c in NUMERIC_COLS:
        # "-" 외의 글자 때문에 null이 된 값이 있으면 개수를 남긴다(없어야 정상).
        bad = raw.select(
            (
                pl.col(c).is_not_null()
                & (pl.col(c) != "-")
                & pl.col(c).cast(pl.Float64, strict=False).is_null()
            ).sum()
        ).item()
        issues[c] = int(bad)
    df = raw.with_columns(
        pl.col("BAS_DD").str.strptime(pl.Date, "%Y%m%d"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC_COLS],
    )
    weekdays = df["BAS_DD"].n_unique()
    cal = (
        df.filter(pl.col("TDD_CLSPRC").is_not_null())
        .select(pl.col("BAS_DD").unique().sort().alias("date"))
        .with_row_index("day_idx")
        .with_columns(pl.col("day_idx").cast(pl.Int64))
        .with_columns(
            (
                pl.col("date").dt.month_end() != pl.col("date").shift(-1).dt.month_end()
            )
            .fill_null(True)
            .alias("is_month_end")
        )
    )
    rows = (
        df.join(cal.select(pl.col("date").alias("BAS_DD"), "day_idx"), on="BAS_DD", how="inner")
        .rename({"BAS_DD": "date"})
        .sort(["ISU_CD", "day_idx"])
    )
    return Panel(rows, cal, n_raw, weekdays, issues)


# ---------------------------------------------------------------- 분류 표
def classification_from_panel(panel: Panel) -> pl.DataFrame:
    """동결 파서 ``classification_table`` 을 그대로 부른다(종가가 있는 행만 그 입력이다)."""
    s = panel.rows.select(
        pl.col("date").dt.strftime("%Y%m%d").alias("BAS_DD"),
        "ISU_CD",
        "ISU_NM",
        "IDX_IND_NM",
        pl.col("TDD_CLSPRC").cast(pl.String).alias("TDD_CLSPRC"),
    )
    return ec.classification_table(s)


# ---------------------------------------------------------------- 생애표·사건 상태
def lifecycle(panel: Panel, cls: pl.DataFrame | None = None) -> pl.DataFrame:
    """ETF당 한 행. 첫/마지막 거래일, 종가 있는 거래일 수, 중간 공백, 사건 상태, 분류 칸.

    중간 공백 = 첫~마지막 거래일 사이(양 끝 포함 구간)의 거래일 중 종가가 없는 날.
    ``gap_no_row`` 는 그 ETF 행이 아예 없는 날, ``gap_blank_close`` 는 행은 있는데 종가가 빈 날.
    ``max_gap_run`` 은 종가 없는 연속 거래일의 최대 길이.
    ``k_after_last`` = 마지막 거래일 뒤 자료 끝까지의 거래일 수, ``status`` 는 §11.3.
    """
    r = panel.rows
    closes = r.filter(pl.col("TDD_CLSPRC").is_not_null())
    per_close = closes.group_by("ISU_CD").agg(
        pl.col("day_idx").min().alias("first_idx"),
        pl.col("day_idx").max().alias("last_idx"),
        pl.len().alias("n_close_days"),
        (pl.col("day_idx").diff().max() - 1).fill_null(0).alias("max_gap_run"),
    )
    # 첫~마지막 구간 안의 행 수
    joined = r.join(per_close.select("ISU_CD", "first_idx", "last_idx"), on="ISU_CD")
    in_win = joined.filter(pl.col("day_idx").is_between(pl.col("first_idx"), pl.col("last_idx")))
    rows_in = in_win.group_by("ISU_CD").agg(pl.len().alias("n_rows_in_window"))
    out = (
        per_close.join(rows_in, on="ISU_CD", how="left")
        .with_columns(
            (pl.col("last_idx") - pl.col("first_idx") + 1).alias("window_days"),
        )
        .with_columns(
            (pl.col("window_days") - pl.col("n_close_days")).alias("gap_days"),
            (pl.col("window_days") - pl.col("n_rows_in_window")).alias("gap_no_row"),
            (pl.col("n_rows_in_window") - pl.col("n_close_days")).alias("gap_blank_close"),
            (panel.end_idx - pl.col("last_idx")).alias("k_after_last"),
        )
        .with_columns(
            pl.when(pl.col("k_after_last") == 0)
            .then(pl.lit("listed"))
            .when(pl.col("k_after_last") <= PENDING_MAX_DAYS)
            .then(pl.lit("pending"))
            .otherwise(pl.lit("disappeared"))
            .alias("status"),
            (pl.col("gap_days") > 0).alias("has_gap"),
        )
    )
    cal = panel.calendar.select("day_idx", "date")
    out = (
        out.join(cal.rename({"day_idx": "first_idx", "date": "first_date"}), on="first_idx")
        .join(cal.rename({"day_idx": "last_idx", "date": "last_date"}), on="last_idx")
        .rename({"ISU_CD": "isu_cd"})
    )
    if cls is None:
        cls = classification_from_panel(panel)
    cls = cls.drop("first_trade_date", "last_trade_date", "n_trade_days")
    return out.join(cls, on="isu_cd", how="left").sort("isu_cd")


def event_status(k_after_last: int) -> str:
    """사건 상태 한 값(§11.3). ``lifecycle`` 과 같은 규칙의 스칼라 판."""
    if k_after_last <= 0:
        return "listed"
    if k_after_last <= PENDING_MAX_DAYS:
        return "pending"
    return "disappeared"


# ---------------------------------------------------------------- 월말 순자산
def month_end_netassets(panel: Panel, life: pl.DataFrame | None = None) -> pl.DataFrame:
    """월말 공식 순자산(§11.1). 월말에 상장 범위(첫~마지막 거래일) 안인 ETF-월마다 한 행.

    칸: ``isu_cd``, ``month_end``, ``day_idx``, ``netasst``(0 아닌 값 또는 null),
    ``netasst_src_date``(쓴 값의 날), ``lag_days``(월말에서 몇 거래일 되짚었나, 0이면 월말 값),
    ``filled``(lag>0), ``raw_zero_or_null``(월말 당일 값이 0이거나 없었나).
    0 이하 값은 누락이다. ``NAV × 상장좌수`` 로 채우지 않는다.
    """
    if life is None:
        life = lifecycle(panel)
    me = panel.month_ends.rename({"date": "month_end"})
    grid = (
        life.select("isu_cd", "first_idx", "last_idx")
        .join(me, how="cross")
        .filter(pl.col("day_idx").is_between(pl.col("first_idx"), pl.col("last_idx")))
        .select("isu_cd", "month_end", "day_idx")
    )
    valid = (
        panel.rows.filter(pl.col("INVSTASST_NETASST_TOTAMT") > 0)
        .select(
            pl.col("ISU_CD").alias("isu_cd"),
            pl.col("day_idx").alias("src_idx"),
            pl.col("date").alias("netasst_src_date"),
            pl.col("INVSTASST_NETASST_TOTAMT").alias("netasst"),
        )
        .with_columns(pl.col("src_idx").alias("day_idx"))
        .sort("day_idx")
    )
    g = grid.sort("day_idx")
    asof = g.join_asof(
        valid,
        on="day_idx",
        by="isu_cd",
        strategy="backward",
        tolerance=NETASST_LOOKBACK_DAYS,
        coalesce=True,
    )
    return (
        asof.with_columns((pl.col("day_idx") - pl.col("src_idx")).alias("lag_days"))
        .with_columns(
            (pl.col("lag_days") > 0).fill_null(False).alias("filled"),
            (pl.col("lag_days") != 0).fill_null(True).alias("raw_zero_or_null"),
        )
        .drop("src_idx")
        .sort(["isu_cd", "month_end"])
    )


# ---------------------------------------------------------------- 일별 괴리
def daily_gap(panel: Panel) -> pl.DataFrame:
    """종가와 NAV가 있고 NAV > 0인 거래일의 ``|종가 − NAV| ÷ NAV`` (칸 ``gap``).

    NAV가 0이거나 비면 그 날은 없다. ``ACC_TRDVAL`` 도 같이 둬서 E2 재료가 한 표에 있다.
    """
    return panel.rows.filter(
        pl.col("TDD_CLSPRC").is_not_null() & pl.col("NAV").is_not_null() & (pl.col("NAV") > 0)
    ).select(
        pl.col("ISU_CD").alias("isu_cd"),
        "date",
        "day_idx",
        ((pl.col("TDD_CLSPRC") - pl.col("NAV")).abs() / pl.col("NAV")).alias("gap"),
        pl.col("ACC_TRDVAL").alias("trdval"),
    )


# ---------------------------------------------------------------- 변동률 쌍
def return_pairs(panel: Panel) -> pl.DataFrame:
    """NAV 변동률과 기초지수 변동률 쌍(상관계수 재료, §11.2).

    이웃한 두 거래일(달력 번호가 1 차이) 모두 그 ETF의 NAV > 0 이고 ``OBJ_STKPRC_IDX`` 가 있고
    > 0 일 때만 쌍을 만든다. 하나라도 없으면 그 날의 쌍은 없다. 휴장일 행은 이미 빠졌다.
    칸: ``isu_cd``, ``date``, ``day_idx``, ``nav_ret``, ``idx_ret``.
    """
    v = (
        panel.rows.filter(
            pl.col("NAV").is_not_null()
            & (pl.col("NAV") > 0)
            & pl.col("OBJ_STKPRC_IDX").is_not_null()
            & (pl.col("OBJ_STKPRC_IDX") > 0)
        )
        .select("ISU_CD", "date", "day_idx", "NAV", pl.col("OBJ_STKPRC_IDX").alias("IDX"))
        .sort(["ISU_CD", "day_idx"])
    )
    return (
        v.with_columns(
            pl.col("day_idx").shift(1).over("ISU_CD").alias("p_idx"),
            pl.col("NAV").shift(1).over("ISU_CD").alias("p_nav"),
            pl.col("IDX").shift(1).over("ISU_CD").alias("p_ix"),
        )
        .filter(pl.col("day_idx") - pl.col("p_idx") == 1)
        .select(
            pl.col("ISU_CD").alias("isu_cd"),
            "date",
            "day_idx",
            (pl.col("NAV") / pl.col("p_nav") - 1).alias("nav_ret"),
            (pl.col("IDX") / pl.col("p_ix") - 1).alias("idx_ret"),
        )
    )


# ---------------------------------------------------------------- 상장 1년
def add_months(d: date, months: int) -> date:
    """달력 월 더하기. 그 달에 없는 날(2월 29일 등)은 달의 마지막 날로 맞춘다."""
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def is_listed_one_year(month_end: date, first_trade_date: date) -> bool:
    """월말 M에서 ``M ≥ 첫 거래일 + 12개월(달력)`` 이면 참(§11.2).

    2010-01-04부터 있던 ETF도 같은 식이라 2011-01 월말부터 참이다(§7.1과 같음).
    """
    return month_end >= add_months(first_trade_date, LISTING_MONTHS)


def listed_one_year_expr(month_end: str = "month_end", first: str = "first_date") -> pl.Expr:
    """``is_listed_one_year`` 의 polars 식 판(두 날짜 칸을 가진 표에 쓴다)."""
    return pl.col(month_end) >= pl.col(first).dt.offset_by(f"{LISTING_MONTHS}mo")


# ---------------------------------------------------------------- 입력 존재 확인
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


def _dist(s: pl.Series) -> dict:
    if s.len() == 0:
        return {"n": 0}
    return {
        "n": s.len(),
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "p10": round(float(s.quantile(0.10)), 6),
        "p25": round(float(s.quantile(0.25)), 6),
        "p75": round(float(s.quantile(0.75)), 6),
        "p90": round(float(s.quantile(0.90)), 6),
        "share_zero": round(float((s == 0).mean()), 6),
        "share_all": round(float((s == 1).mean()), 6),
        "share_gt_half": round(float((s > 0.5).mean()), 6),
    }


def pre_delist_zero_ratio(panel: Panel, life: pl.DataFrame) -> pl.DataFrame:
    """disappeared ETF마다, 그 ETF의 종가 있는 마지막 30거래일(마지막 거래일 포함) 중
    순자산이 0이거나 없는 날의 비율(R17). 30일이 안 되면 있는 날 전부."""
    dis = life.filter(pl.col("status") == "disappeared").select(
        "isu_cd", "last_date", "exclude_maturity", "is_maturity_type"
    )
    t = (
        panel.rows.filter(pl.col("TDD_CLSPRC").is_not_null())
        .rename({"ISU_CD": "isu_cd"})
        .join(dis.select("isu_cd"), on="isu_cd")
        .sort(["isu_cd", "day_idx"])
        .group_by("isu_cd", maintain_order=True)
        .tail(PRE_DELIST_DAYS)
    )
    z = t.group_by("isu_cd").agg(
        pl.len().alias("n_days"),
        (
            pl.col("INVSTASST_NETASST_TOTAMT").is_null()
            | (pl.col("INVSTASST_NETASST_TOTAMT") == 0)
        )
        .sum()
        .alias("n_zero_or_null"),
    )
    return (
        dis.join(z, on="isu_cd")
        .with_columns((pl.col("n_zero_or_null") / pl.col("n_days")).alias("zero_ratio"))
        .with_columns(pl.col("last_date").dt.year().alias("last_year"))
        .sort("isu_cd")
    )


def run_checks(panel: Panel, input_path: str | Path, out: Path) -> dict:
    """(a)~(g) 입력 존재 확인을 계산해 CSV·``checks.json`` 으로 쓴다."""
    out.mkdir(parents=True, exist_ok=True)
    life = lifecycle(panel)
    res: dict = {}

    # (a) 기본 수
    res["a_basic"] = {
        "rows": panel.n_rows_raw,
        "weekdays_in_file": panel.n_weekdays_in_file,
        "closed_days": panel.n_closed_days,
        "closed_days_expected": 248,
        "trading_days": panel.n_days,
        "first_day": str(panel.calendar["date"][0]),
        "end_day": str(panel.end_date),
        "month_ends": panel.month_ends.height,
        "etfs": life.height,
        "etfs_expected": 1433,
        "status_counts": {
            k: int(v) for k, v in life.group_by("status").len().iter_rows()
        },
        "parse_issues_nonblank_unparsed": {k: v for k, v in panel.parse_issues.items() if v},
    }
    # 동결 파서와 거래일 정의가 같은지 확인(칸 값 일치)
    raw_s = ec.read_input(str(input_path))
    days_ec = ec.trading_days(raw_s)
    cls_ec = ec.classification_table(raw_s).select(
        "isu_cd", "first_trade_date", "last_trade_date", "n_trade_days"
    )
    mine = life.select(
        "isu_cd",
        pl.col("first_date").dt.strftime("%Y%m%d").alias("first_trade_date"),
        pl.col("last_date").dt.strftime("%Y%m%d").alias("last_trade_date"),
        pl.col("n_close_days").alias("n_trade_days"),
    )
    cmp = cls_ec.join(mine, on="isu_cd", how="full", suffix="_m")
    res["a_basic"]["consistency_with_etf_classify"] = {
        "trading_days_equal": days_ec == [d.strftime("%Y%m%d") for d in panel.calendar["date"]],
        "lifecycle_equal": bool(
            cmp.height == life.height
            and (cmp["first_trade_date"] == cmp["first_trade_date_m"]).all()
            and (cmp["last_trade_date"] == cmp["last_trade_date_m"]).all()
            and (cmp["n_trade_days"] == cmp["n_trade_days_m"]).all()
        ),
    }
    life.write_csv(out / "lifecycle.csv")

    # (b) 폐지 직전 30거래일 순자산 0 비율
    zr = pre_delist_zero_ratio(panel, life)
    zr.write_csv(out / "b_pre_delist_zero_ratio.csv")
    by_year = []
    for y, g in zr.group_by("last_year", maintain_order=False):
        for label, sub in (("incl_maturity", g), ("excl_maturity", g.filter(~pl.col("exclude_maturity")))):
            by_year.append({"last_year": y[0], "set": label, **_dist(sub["zero_ratio"])})
    by_year_df = pl.DataFrame(by_year, infer_schema_length=None).sort(["set", "last_year"])
    by_year_df.write_csv(out / "b_pre_delist_zero_ratio_by_year.csv")
    res["b_pre_delist_zero_ratio"] = {
        "window_trading_days": PRE_DELIST_DAYS,
        "incl_maturity": _dist(zr["zero_ratio"]),
        "excl_maturity": _dist(zr.filter(~pl.col("exclude_maturity"))["zero_ratio"]),
        "n_window_lt_30": int((zr["n_days"] < PRE_DELIST_DAYS).sum()),
        "by_year": by_year_df.to_dicts(),
    }

    # (c) 중간 공백
    gap = life.filter(pl.col("has_gap"))
    long_gap = gap.filter(pl.col("max_gap_run") >= GAP_LONG_DAYS)
    gap_cols = [
        "isu_cd", "isu_nm", "first_date", "last_date", "status", "n_close_days",
        "gap_days", "gap_no_row", "gap_blank_close", "max_gap_run", "exclude_maturity",
    ]
    gap.select(gap_cols).write_csv(out / "gap_etfs.csv")
    res["c_gap_etfs"] = {
        "n_with_gap": gap.height,
        "n_gap_run_ge_10": long_gap.height,
        "gap_run_ge_10_list": long_gap.select(
            "isu_cd", "isu_nm", "max_gap_run", "gap_days", "status"
        ).with_columns(pl.col("isu_cd")).to_dicts(),
        "gap_days_total": int(gap["gap_days"].sum() or 0),
        "gap_no_row_total": int(gap["gap_no_row"].sum() or 0),
        "gap_blank_close_total": int(gap["gap_blank_close"].sum() or 0),
        "reappeared_after_gap_by_status": {
            k: int(v) for k, v in gap.group_by("status").len().iter_rows()
        },
    }

    # (d) 만기형 비율
    dis = life.filter(pl.col("status") == "disappeared")
    nd = dis.height
    n_name = int(dis["is_maturity_type"].sum())
    n_excl = int(dis["exclude_maturity"].sum())
    res["d_maturity_share"] = {
        "disappeared": nd,
        "name_yymm_only": {"n": n_name, "share": round(n_name / nd, 4) if nd else None},
        "exclude_maturity_parser": {"n": n_excl, "share": round(n_excl / nd, 4) if nd else None},
        "reference_kind": {"delisted": 225, "maturity": 26, "share": round(26 / 225, 4)},
        "all_etfs_exclude_maturity": int(life["exclude_maturity"].sum()),
    }

    # (e) 월말 순자산 결측
    mn = month_end_netassets(panel, life)
    mn.write_csv(out / "e_month_end_netassets.csv")
    e = (
        mn.with_columns(pl.col("month_end").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("etf_months"),
            pl.col("netasst").is_null().sum().alias("null_after_lookback"),
            pl.col("filled").sum().alias("filled_by_lookback"),
            pl.col("raw_zero_or_null").sum().alias("raw_zero_or_null"),
        )
        .with_columns(
            (pl.col("null_after_lookback") / pl.col("etf_months")).alias("null_share"),
            (pl.col("raw_zero_or_null") / pl.col("etf_months")).alias("raw_zero_or_null_share"),
        )
        .sort("year")
    )
    e.write_csv(out / "e_month_end_netassets_by_year.csv")
    res["e_month_end_netassets"] = {
        "etf_months": mn.height,
        "null_after_lookback": int(mn["netasst"].is_null().sum()),
        "filled_by_lookback": int(mn["filled"].sum()),
        "raw_zero_or_null": int(mn["raw_zero_or_null"].sum()),
        "by_year": e.to_dicts(),
    }

    # (f) 기초지수 종가 결측(국내형, 종가 있는 거래일 기준)
    dom = life.filter(pl.col("region") == "domestic").select("isu_cd")
    f = (
        panel.rows.filter(pl.col("TDD_CLSPRC").is_not_null())
        .join(dom, left_on="ISU_CD", right_on="isu_cd")
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("etf_days"),
            pl.col("OBJ_STKPRC_IDX").is_null().sum().alias("idx_null"),
            (pl.col("OBJ_STKPRC_IDX") == 0).sum().alias("idx_zero"),
        )
        .with_columns((pl.col("idx_null") / pl.col("etf_days")).alias("null_share"))
        .sort("year")
    )
    f.write_csv(out / "f_domestic_index_close_missing_by_year.csv")
    res["f_domestic_index_missing"] = {
        "domestic_etfs": dom.height,
        "etf_days": int(f["etf_days"].sum()),
        "idx_null": int(f["idx_null"].sum()),
        "idx_null_share": round(float(f["idx_null"].sum() / f["etf_days"].sum()), 6),
        "idx_zero": int(f["idx_zero"].sum()),
        "by_year": f.to_dicts(),
    }

    # (g) 목록
    ev_cols = [
        "isu_cd", "isu_nm", "first_date", "last_date", "n_close_days",
        "is_maturity_type", "maturity_by_index_name", "exclude_maturity",
        "pension_ineligible_candidate", "region", "active",
    ]
    dis.select(ev_cols).write_csv(out / "event_list.csv")
    pend = life.filter(pl.col("status") == "pending")
    pend.select(ev_cols + ["k_after_last"]).write_csv(out / "pending.csv")
    res["g_lists"] = {
        "event_list_rows": dis.height,
        "pending_rows": pend.height,
        "gap_etfs_rows": gap.height,
    }

    (out / "checks.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--input", default=str(root / DEFAULT_INPUT_REL))
    ap.add_argument("--out-dir", default=str(root / DEFAULT_OUT_REL))
    ap.add_argument("--checks", action="store_true", help="입력 존재 확인(§12.4 E)을 돌린다")
    a = ap.parse_args(argv)
    if not a.checks:
        ap.print_help()
        return 2

    out = Path(a.out_dir)
    panel = read_panel(a.input)
    res = run_checks(panel, a.input, out)

    src = Path(__file__)
    status = _git("status", "--porcelain")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "quality-score v0 ETF 일별 패널 입력 존재 확인(사전등록 §11.1~11.3, §12.4 E). "
        "점수·사건 연결 없음.",
        "input": {"path": str(a.input), "sha256": _sha256(a.input), "rows": panel.n_rows_raw},
        "module": {"version": PANEL_VERSION, "file": str(src), "sha256": _sha256(src)},
        "frozen_parser_sha256": {
            n: _sha256(src.parent / n) for n in ("etf_classify.py", "etf_theme.py", "power.py")
        },
        "code": {
            "repo": "modeler",
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": status.splitlines(),
            "worktree_dirty": bool(status),
        },
        "runtime": {"python": sys.version.split()[0], "polars": pl.__version__},
        "outputs": sorted(p.name for p in out.glob("*") if p.name != "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "x"}, ensure_ascii=False, indent=1,
                     default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
