"""ETF 테마 상품 카드 (사전등록 20261010_quality_score §11.5, 설명용 기록).

판정이 아니다. 테마마다 지금 상장된 상품을 한 표에 모아 상품 속성과 E1 상장 유지 위험 점수를 같이
보여 준다. **유망성·수익률 판단을 하지 않는다.** 수익률·가격 등락·테마 사이 비교·종가 자체를 넣지
않는다(괴리율만). 사건(상장폐지)과 점수를 잇지 않는다 — 현재 시점 설명값이다.

    STOCK_DATA_ROOT=../stock_data PYTHONPATH=src python -m modeler.scores.quality.etf_theme_card

경로는 환경변수 ``STOCK_DATA_ROOT`` 로 받는다(기본 ``../stock_data``). 산출물은
``kr/output/quality_score_etf_theme_cards_20261010/`` 에만 쓴다. **KRX 약관(제3자 제공 금지) 때문에
산출물을 외부에 올리지 않는다.**

규칙
- 테마 소속은 동결 ``etf_theme.py`` 의 ``membership_table`` 로 다시 계산하고, 동결 때 산출한
  ``etf_theme_membership.csv`` 와 같은지 확인한다. 다르면 멈춘다.
- 카드 대상 = 자료 끝에 상장 중인 ETF(``status == "listed"``). 만기형도 카드에는 남기고(E1 점수 없음,
  이유 ``만기형``) 테마 소속은 동결 표 그대로다. 한 ETF가 여러 테마에 들면 행이 여러 개다.
- E1은 ``etf_e1.monthly_scores`` 를 기본 인자 그대로 불러 기준 월말(2026-09-30) 행을 읽는다.
- 설명값 창 = 자료 끝까지 최근 252거래일. 거래대금 중앙값은 그 창에서 종가가 있는 일의
  ``ACC_TRDVAL`` 중앙값, 괴리율은 ``daily_gap`` 의 \\|종가 − NAV\\| ÷ NAV 일평균이다.
- **합성 표시(``synthetic_display``)**: 동결 파서에 합성 칸이 없다. 카드에 보이려고 이름에 ``합성`` 이
  있으면 참으로 둘 뿐이며 **판정에 쓰지 않는다.**
- **채권혼합 표시(``bond_mix_display``)**: ``synthetic_display`` 와 같은 방식이다. 마지막 거래일 이름
  (``ISU_NM``)에 ``채권혼합`` 이 있으면 참으로 둘 뿐이며 **판정에 쓰지 않는다**(동결 테마 사전은 그대로).
  채권혼합 상품은 주식 비중이 절반 안팎이라 테마 노출이 묽다는 점을 카드에서 알아보게 하는 표시다.
- 총보수는 원천이 정해지지 않아 전부 ``원천 미정``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl

from modeler.scores.quality import etf_e1 as e1
from modeler.scores.quality import etf_panel as ep
from modeler.scores.quality import etf_theme as et

CARD_VERSION = "quality-score-v0/etf_theme_card/1"

DEFAULT_OUT_REL = "kr/output/quality_score_etf_theme_cards_20261010"
DEFAULT_MEMBERSHIP_REL = "kr/output/quality_score_prep_20261010/etf_theme_membership.csv"

MONTH_END = date(2026, 9, 30)  # E1 기준 월말(자료 끝 달은 월말로 안 쓴다, etf_e1 I21)
DESC_WINDOW_SESSIONS = 252  # 설명값 창: 자료 끝까지 최근 252거래일
FEE_PLACEHOLDER = "원천 미정"
SYNTHETIC_TOKEN = "합성"  # 표시용. 판정에 안 씀
BOND_MIX_TOKEN = "채권혼합"  # 표시용. 판정에 안 씀

# 점수 없는 이유(§ 작업 지시)
R_NEW = "상장 1년 미만"
R_CORR = "기초지수 종가 부족(상관 쌍 < 200)"
R_MATURITY = "만기형"
R_PENSION = "연금 부적격 후보"
R_REGION = "국내/해외 미분류"
R_NETASST = "순자산 없음"
R_OTHER = "점수 없음(기타)"
NO_SCORE_REASONS = (R_NEW, R_CORR, R_MATURITY, R_PENSION, R_REGION, R_NETASST)

E1_JUDGMENT_DOMESTIC = "국내형 판정 D(2026-10-10, 사용자 결정: 유지 — 설명값)"
E1_JUDGMENT_FOREIGN = "해외형 기록용(순자산 단독)"
E2_JUDGMENT_NOTE = (
    "E2 괴리 판정 A(같은 비교 그룹 안 이듬해 괴리 순위 상관) — 테마 상품에는 참고로만"
)


def e1_judgment_label(has_score: bool, e1_type: str | None) -> str:
    """E1 판정 표시(정정 E-1 재분류, 동결 규칙 판정). 점수 없는 행은 빈 값."""
    if not has_score:
        return ""
    if e1_type == "domestic":
        return E1_JUDGMENT_DOMESTIC
    if e1_type == "foreign":
        return E1_JUDGMENT_FOREIGN
    return ""


MEMO_NOT_ALLOWED = "연금저축·IRP 불가"
MEMO_SYNTHETIC = "IRP 제한 가능"

CARD_COLUMNS = [
    "theme", "category", "required_by_user",
    "isu_cd", "isu_nm", "first_date",
    "base_index", "idx_ind_nm", "region", "active", "hedge",
    "synthetic_display", "bond_mix_display", "option", "multiplier",
    "pension_eligible_candidate", "account_memo",
    "e1_month_end", "e1_pct", "e1_alert", "e1_type", "e1_no_score_reason", "e1_foreign_record_only",
    "e1_judgment",
    "netasst_won", "trdval_median_252", "gap_mean_252", "n_days_252", "n_gap_days_252",
    "fee",
]  # fmt: skip

# 칸 이름에 수익률·가격이 들어가면 안 된다는 시험용 금지어
FORBIDDEN_COLUMN_WORDS = ("ret", "return", "수익", "close", "clsprc", "price", "가격", "rank_theme")


# ---------------------------------------------------------------- 순수 규칙(시험 대상)
def synthetic_display(isu_nm: str | None) -> bool:
    """이름에 ``합성`` 이 있으면 참. 카드 표시용이며 판정에 안 쓴다."""
    return SYNTHETIC_TOKEN in (isu_nm or "")


def bond_mix_display(isu_nm: str | None) -> bool:
    """이름에 ``채권혼합`` 이 있으면 참. 카드 표시용이며 판정에 안 쓴다."""
    return BOND_MIX_TOKEN in (isu_nm or "")


def account_memo(pension_ineligible: bool, synthetic: bool) -> str:
    """계좌 메모. 레버리지·인버스는 불가, 합성은 IRP 제한 가능. 둘 다면 ``; `` 로 잇는다."""
    parts = []
    if pension_ineligible:
        parts.append(MEMO_NOT_ALLOWED)
    if synthetic:
        parts.append(MEMO_SYNTHETIC)
    return "; ".join(parts)


def no_score_reason(
    *,
    has_score: bool,
    pension_ineligible: bool,
    exclude_maturity: bool,
    region: str | None,
    first_date: date,
    month_end: date,
    netasst: float | None,
) -> str:
    """E1 점수가 없는 이유 한 칸. 점수가 있으면 빈 문자열.

    우선순위: 연금 부적격 후보 → 만기형 → 국내/해외 미분류 → 상장 1년 미만 → 순자산 없음 →
    기초지수 종가 부족(국내형 상관 쌍 200 미만). ``etf_e1.monthly_scores`` 의 대상 조건과 같은 순서다.
    """
    if has_score:
        return ""
    if pension_ineligible:
        return R_PENSION
    if exclude_maturity:
        return R_MATURITY
    if region not in e1.REGIONS:
        return R_REGION
    if not ep.is_listed_one_year(month_end, first_date):
        return R_NEW
    if netasst is None:
        return R_NETASST
    if region == "domestic":
        return R_CORR
    return R_OTHER


def window_start_idx(end_idx: int, sessions: int = DESC_WINDOW_SESSIONS) -> int:
    """자료 끝(end_idx)까지 최근 ``sessions`` 거래일 창의 첫 day_idx."""
    return end_idx - sessions + 1


def window_stats(panel: ep.Panel, sessions: int = DESC_WINDOW_SESSIONS) -> pl.DataFrame:
    """ETF마다 최근 ``sessions`` 거래일의 거래대금 중앙값·괴리율 일평균(설명값).

    칸: ``isu_cd``, ``trdval_median_252``, ``gap_mean_252``, ``n_days_252``(종가 있는 일 수),
    ``n_gap_days_252``(괴리율을 낸 일 수). 종가 자체는 내지 않는다.
    """
    lo = window_start_idx(panel.end_idx, sessions)
    tv = (
        panel.rows.filter((pl.col("day_idx") >= lo) & pl.col("TDD_CLSPRC").is_not_null())
        .group_by("ISU_CD")
        .agg(
            pl.col("ACC_TRDVAL").median().alias("trdval_median_252"),
            pl.len().alias("n_days_252"),
        )
        .rename({"ISU_CD": "isu_cd"})
    )
    gp = (
        ep.daily_gap(panel)
        .filter(pl.col("day_idx") >= lo)
        .group_by("isu_cd")
        .agg(pl.col("gap").mean().alias("gap_mean_252"), pl.len().alias("n_gap_days_252"))
    )
    return tv.join(gp, on="isu_cd", how="left")


# ---------------------------------------------------------------- 동결 소속표 대조
def recompute_membership(panel: ep.Panel) -> pl.DataFrame:
    """동결 ``etf_theme.membership_table`` 을 패널에서 만든 분류 표로 다시 계산한다."""
    cls = ep.classification_from_panel(panel).with_columns(
        pl.col("first_trade_date").cast(pl.Int64), pl.col("last_trade_date").cast(pl.Int64)
    )
    return et.membership_table(cls)


def compare_membership(new: pl.DataFrame, frozen_csv: str | Path) -> dict:
    """다시 계산한 소속표와 동결 때 CSV를 칸 단위로 대조한다. 다르면 ``ok`` 가 거짓."""
    old = pl.read_csv(frozen_csv, infer_schema_length=0, null_values=[])
    cols = ["isu_cd", "listed_now", "themes", "theme_only", "style_only", "method",
            "single_stock", "asset_class"]  # fmt: skip
    a = new.select(cols).with_columns(pl.all().cast(pl.String).fill_null(""))
    b = old.select(cols).with_columns(pl.all().fill_null(""))
    a = a.with_columns(pl.col("listed_now").str.to_lowercase())
    b = b.with_columns(pl.col("listed_now").str.to_lowercase())
    j = a.join(b, on="isu_cd", how="full", suffix="_old", coalesce=True)
    diff_cols = {}
    for c in cols[1:]:
        n = j.filter(pl.col(c).is_null() | pl.col(f"{c}_old").is_null() | (pl.col(c) != pl.col(f"{c}_old"))).height
        diff_cols[c] = n
    only_new = j.filter(pl.col("themes_old").is_null()).height
    only_old = j.filter(pl.col("themes").is_null()).height
    return {
        "ok": not any(diff_cols.values()) and only_new == 0 and only_old == 0,
        "n_new": a.height,
        "n_frozen": b.height,
        "n_diff_by_column": diff_cols,
        "only_in_new": only_new,
        "only_in_frozen": only_old,
    }


# ---------------------------------------------------------------- 카드 만들기
def build_cards(
    panel: ep.Panel,
    life: pl.DataFrame,
    mem: pl.DataFrame,
    scores: pl.DataFrame,
    month_end: date = MONTH_END,
) -> pl.DataFrame:
    """카드 한 행 = (테마, ETF). 대상 = ``status == "listed"``. 테마 → 순자산 내림차순."""
    dict_order = {t.name: i for i, t in enumerate(et.THEMES)}
    cat = {t.name: t.category for t in et.THEMES}
    req = {t.name: t.required for t in et.THEMES}

    listed = life.filter(pl.col("status") == "listed")
    mn = ep.month_end_netassets(panel, life).filter(pl.col("month_end") == month_end).select(
        "isu_cd", "netasst"
    )
    sc = scores.filter(pl.col("month_end") == month_end).select(
        "isu_cd", "e1_pct", "alert", "region", "in_pool"
    ).rename({"region": "e1_type"})
    ws = window_stats(panel)

    base = (
        listed.select(
            "isu_cd", "isu_nm", "first_date", "base_index", "idx_ind_nm", "region", "active",
            "hedge", "option", "multiplier",
            pl.col("exclude_maturity").fill_null(False),
            pl.col("pension_ineligible_candidate").fill_null(False),
        )
        .join(mn, on="isu_cd", how="left")
        .join(sc, on="isu_cd", how="left")
        .join(ws, on="isu_cd", how="left")
    )

    members = {r["isu_cd"]: [t for t in r["themes"].split("|") if t] for r in mem.iter_rows(named=True)}
    rows = []
    for r in base.iter_rows(named=True):
        ths = members.get(r["isu_cd"], [])
        if not ths:
            continue
        syn = synthetic_display(r["isu_nm"])
        has_score = r["e1_pct"] is not None
        reason = no_score_reason(
            has_score=has_score,
            pension_ineligible=r["pension_ineligible_candidate"],
            exclude_maturity=r["exclude_maturity"],
            region=r["region"],
            first_date=r["first_date"],
            month_end=month_end,
            netasst=r["netasst"],
        )
        for th in ths:
            rows.append(
                {
                    "theme": th,
                    "category": cat[th],
                    "required_by_user": req[th],
                    "isu_cd": r["isu_cd"],
                    "isu_nm": r["isu_nm"],
                    "first_date": r["first_date"],
                    "base_index": r["base_index"],
                    "idx_ind_nm": r["idx_ind_nm"],
                    "region": r["region"],
                    "active": r["active"],
                    "hedge": r["hedge"],
                    "synthetic_display": syn,
                    "bond_mix_display": bond_mix_display(r["isu_nm"]),
                    "option": r["option"],
                    "multiplier": r["multiplier"],
                    "pension_eligible_candidate": not r["pension_ineligible_candidate"],
                    "account_memo": account_memo(r["pension_ineligible_candidate"], syn),
                    "e1_month_end": month_end if has_score else None,
                    "e1_pct": r["e1_pct"],
                    "e1_alert": r["alert"] if has_score else None,
                    "e1_type": r["e1_type"] if has_score else None,
                    "e1_no_score_reason": reason,
                    "e1_foreign_record_only": bool(has_score and r["region"] == "foreign"),
                    "e1_judgment": e1_judgment_label(has_score, r["e1_type"]),
                    "netasst_won": r["netasst"],
                    "trdval_median_252": r["trdval_median_252"],
                    "gap_mean_252": r["gap_mean_252"],
                    "n_days_252": r["n_days_252"],
                    "n_gap_days_252": r["n_gap_days_252"],
                    "fee": FEE_PLACEHOLDER,
                    "_ord": dict_order[th],
                }
            )
    schema = {
        "e1_month_end": pl.Date, "e1_pct": pl.Float64, "e1_alert": pl.Boolean, "e1_type": pl.String,
        "netasst_won": pl.Float64, "trdval_median_252": pl.Float64, "gap_mean_252": pl.Float64,
        "n_days_252": pl.Int64, "n_gap_days_252": pl.Int64,
    }  # fmt: skip
    df = pl.DataFrame(rows, schema_overrides=schema, infer_schema_length=None)
    return (
        df.sort(["_ord", "netasst_won", "isu_cd"], descending=[False, True, False], nulls_last=True)
        .select(CARD_COLUMNS)
    )


def build_summary(cards: pl.DataFrame) -> pl.DataFrame:
    """테마별 요약(개수·중앙값만). 테마 사이 순위나 비교 칸은 없다."""
    order = {t.name: i for i, t in enumerate(et.THEMES)}
    aggs = [
        pl.len().alias("n_listed"),
        pl.col("pension_eligible_candidate").sum().alias("n_pension_eligible"),
        pl.col("bond_mix_display").sum().alias("n_bond_mix"),
        pl.col("e1_pct").is_not_null().sum().alias("n_e1_scored"),
        (pl.col("e1_alert") == True).sum().alias("n_alert"),  # noqa: E712
        *[
            (pl.col("e1_no_score_reason") == r).sum().alias(f"n_no_score__{r}")
            for r in NO_SCORE_REASONS
        ],
        pl.col("netasst_won").median().alias("netasst_median_won"),
    ]
    out = cards.group_by("theme", "category", "required_by_user", maintain_order=True).agg(aggs)
    return out.with_columns(pl.col("theme").replace_strict(order, return_dtype=pl.Int64).alias("_o")).sort(
        "_o"
    ).drop("_o")


# ---------------------------------------------------------------- 사용자 일곱 테마 표(md)
def _md(s: object) -> str:
    """표 칸용 이스케이프. 파이프와 물결표."""
    return str("" if s is None else s).replace("|", "\\|").replace("~", "\\~")


def _yn(v: bool | None) -> str:
    return "" if v is None else ("예" if v else "아니오")


def _e1_cell(r: dict) -> str:
    if r["e1_pct"] is None:
        return f"없음: {r['e1_no_score_reason']}"
    s = f"{r['e1_pct']:.1f}"
    if r["e1_alert"]:
        s += " 경보"
    if r["e1_foreign_record_only"]:
        s += " (해외형·순자산 하나로 매긴 기록용)"
    return s


def render_user_theme_md(cards: pl.DataFrame, month_end: date, end_date: date) -> str:
    """사용자 일곱 테마마다 표 하나. 연금 적격 후보만. 숫자만, 추천 문장 없음."""
    lines = [
        "# ETF 테마 상품 카드 — 사용자 일곱 테마 (설명용 기록)",
        "",
        f"E1 기준 월말 {month_end}, 거래대금·괴리율 창은 자료 끝 {end_date}까지 최근 "
        f"{DESC_WINDOW_SESSIONS}거래일입니다. 사전등록 §11.5 설명용 기록이며 판정이 아닙니다. "
        "수익률·유망성·테마 사이 비교는 넣지 않았습니다. 총보수는 원천 미정입니다. "
        "KRX 약관상 외부에 올리지 않습니다.",
        "",
        f"E1 판정 표시: 국내형 E1은 동결 규칙으로 판정해 D였습니다(2026-10-10, 사용자 결정: 유지 — 설명값으로 계속 씁니다). "
        f"해외형은 순자산 하나로 매긴 기록용입니다. {E2_JUDGMENT_NOTE}.",
        "",
        "채권혼합 상품은 주식 비중이 절반 안팎이라 테마 노출이 묽습니다(표시만, 사전은 그대로).",
        "",
    ]
    for th in [t.name for t in et.THEMES if t.required]:
        sub = cards.filter(pl.col("theme") == th)
        elig = sub.filter(pl.col("pension_eligible_candidate"))
        n_alert = int((sub["e1_alert"] == True).sum())  # noqa: E712
        lines += [
            f"## {_md(th)}",
            "",
            f"상장 {sub.height} · 적격 {elig.height} · 경보 {n_alert}",
            "",
            "| 상품 | 기초지수 | 국내/해외 | 액티브 | 환헤지 | 합성 | 채권혼합 | E1 백분위/경보 또는 없는 이유 "
            "| 순자산(억 원) | 거래대금 중앙값(억 원) | 괴리율(%) |",
            "|---|---|---|---|---|---|---|---|---:|---:|---:|",
        ]
        for r in elig.iter_rows(named=True):
            na = "" if r["netasst_won"] is None else f"{r['netasst_won'] / 1e8:,.1f}"
            tv = "" if r["trdval_median_252"] is None else f"{r['trdval_median_252'] / 1e8:,.2f}"
            gp = "" if r["gap_mean_252"] is None else f"{r['gap_mean_252'] * 100:.3f}"
            lines.append(
                f"| {_md(r['isu_cd'])} {_md(r['isu_nm'])} | {_md(r['idx_ind_nm'])} | {_md(r['region'])} "
                f"| {_yn(r['active'])} | {_md(r['hedge'])} | {_yn(r['synthetic_display'])} "
                f"| {_yn(r['bond_mix_display'])} "
                f"| {_md(_e1_cell(r))} | {na} | {tv} | {gp} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 실행
def _sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.check_output(["git", "-C", str(here), *args], text=True).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--input", default=str(root / ep.DEFAULT_INPUT_REL))
    ap.add_argument("--membership", default=str(root / DEFAULT_MEMBERSHIP_REL))
    ap.add_argument("--out-dir", default=str(root / DEFAULT_OUT_REL))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)

    panel = ep.read_panel(a.input)
    life = ep.lifecycle(panel)

    mem = recompute_membership(panel)
    cmp = compare_membership(mem, a.membership)
    if not cmp["ok"]:
        print(json.dumps(cmp, ensure_ascii=False, indent=1))
        raise SystemExit("동결 소속표와 다시 계산한 소속표가 다릅니다. 멈춥니다.")

    scores = e1.monthly_scores(panel, life)  # 기본 인자 그대로
    cards = build_cards(panel, life, mem, scores, MONTH_END)
    summary = build_summary(cards)

    out.mkdir(parents=True, exist_ok=True)
    cards.write_csv(out / "theme_cards.csv")
    summary.write_csv(out / "theme_summary.csv")
    (out / "cards_user7.md").write_text(render_user_theme_md(cards, MONTH_END, panel.end_date))

    src = Path(__file__)
    theme_src = src.parent / "etf_theme.py"
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "ETF 테마 상품 카드(사전등록 §11.5 설명용 기록). 판정 아님. 수익률·유망성·테마 간 비교 없음. "
        "KRX 약관상 외부 제공 금지.",
        "inputs": {
            "panel_csv": {"path": str(a.input), "sha256": _sha256(a.input)},
            "frozen_membership_csv": {"path": str(a.membership), "sha256": _sha256(a.membership)},
        },
        "frozen_dictionary": {"file": str(theme_src), "sha256": _sha256(theme_src)},
        "membership_check": cmp,
        "module": {"file": str(src), "sha256": _sha256(src)},
        "versions": {
            "card": CARD_VERSION,
            "panel": ep.PANEL_VERSION,
            "e1": e1.E1_VERSION,
            "dictionary": et.DICT_VERSION,
            "polars": pl.__version__,
            "python": sys.version.split()[0],
        },
        "code": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain").splitlines(),
        },
        "dates": {
            "month_end": str(MONTH_END),
            "data_end": str(panel.end_date),
            "desc_window_sessions": DESC_WINDOW_SESSIONS,
            "desc_window_first_date": str(
                panel.calendar.filter(
                    pl.col("day_idx") == window_start_idx(panel.end_idx)
                )["date"][0]
            ),
        },
        "n_cards": cards.height,
        "n_etfs": cards["isu_cd"].n_unique(),
        "n_themes": cards["theme"].n_unique(),
        "outputs": sorted(["theme_cards.csv", "theme_summary.csv", "cards_user7.md"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"cards": cards.height, "etfs": manifest["n_etfs"], "membership": cmp}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
