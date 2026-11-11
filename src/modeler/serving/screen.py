"""Daily briefing rank screening (rule ``screen-v1``): which scored stocks are "worth watching".

Spec (canonical): ``my/milestones/common/20261110_briefing_usefulness/06_screen_v1_rules.md``, fixed
2026-11-10 23:05 KST.  The regexes and form lists below are copied verbatim from it; changing any
definition is a new rule version (``screen-v2``) and is never applied to past units.

The stage runs after inference (and market-sector), before explain.  For each of the three models it
reads the report's whole scored cross-section and the pinned raw inputs, and writes ``screen-D.json``
(``briefing-screen.v1``): per row the condition values, the hit reasons, the filing receipt numbers,
plus the input hashes and the dates used.  It never changes a model, a score, a rank or
``report-D.json``.  The screened rank is the original rank order with every row that fails a
condition skipped; nothing is re-scored.

Heavy libraries (polars) are imported inside the functions that need them, so the coordinator can
import the document helpers alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "briefing-screen.v1"
RULE_VERSION = "screen-v1"
SCREEN_TOP = 30  # 선별 상위 30 (세 모델 같음)
WINDOW_SESSIONS = 252
LIQUIDITY_SESSIONS = 20

# ---------------------------------------------------------------- KR (06 §2) — verbatim
KR_PREFERRED_LAST_DIGIT = "0"  # 코드 끝자리가 0이 아니면 우선주 (modeler etl/one_per_company.py 와 같은 규칙)
KR_SPAC_RE = r"스팩|기업인수목적"
KR_REIT_RE = r"리츠|인프라$|맥쿼리인프라"
KR_FILING_INCLUDE_RE = (
    r"상장폐지|실질심사|관리종목|개선기간|정리매매|환기종목|의견거절|부적정|감사의견|자본잠식|"
    r"자기자본10억원미만|회생절차|파산신청|부도발생|매출액50억원미만|파산선고|회생계획인가"
)
KR_FILING_EXCLUDE_RE = (
    r"미지정|기각|사유 해소|대상 제외|상장유지|종속회사|자회사|피보증|미진행|해제|취하|철회|조회공시요구|"
    r"이전상장|해외증권|의안상정|우선주|신주인수권|사채|채권|^상장폐지결정$|사유\s*해소|출자법인|"
    r"관리종목지정우려종목\)?\s*\(?(주가|시가총액)|(주가|시가총액)(요건)? ?미달에 따른"
)
KR_FILING_DAYS = 1095  # 접수일이 (K − 1,095일, K]
KR_HALT_DAYS = 20  # 252세션 안 거래량 0인 날 20일 이상
KR_SESSION_MIN_TICKERS = 1000  # 세션은 종목 1,000개 넘게 시세가 있는 날
KR_MIN_CLOSE = 1000  # 원
KR_MIN_MCAP = 20_000_000_000  # 200억 원
KR_SIZE_RANK = 500
KR_LIQUIDITY_RANK = 500

# ---------------------------------------------------------------- US (06 §3) — verbatim
US_ACQ_A_FORMS = frozenset({"SC 14D9", "SC 14D9/A"})
US_ACQ_B_FORMS = frozenset({"SC TO-T", "SC TO-T/A", "SC 13E3", "SC 13E3/A"})
US_ACQ_B_FILE_PREFIX = "005-"
US_ACQ_C_FORMS = frozenset({"DEFM14A", "DEFM14A/A", "PREM14A", "PREM14A/A"})
US_ACQ_C_ISSUER_FORMS = frozenset({"S-4", "S-4/A", "S-4MEF", "S-4 POS", "F-4", "F-4/A"})
US_ACQ_C_COMPLETION_ITEM = "2.01"  # 8-K
US_ACQ_C_BLOCK_ITEMS = frozenset({"5.01", "3.01"})  # 8-K
US_ACQ_C_BLOCK_FORMS = frozenset({"15-12B", "15-12G"})
US_LISTING_ITEM = "3.01"  # 8-K
US_LISTING_FORMS = frozenset({"15-12G", "15-12G/A"})
US_8K = "8-K"
US_NARROW_RANGE = 0.015
US_NARROW_SESSIONS = 20
US_LIQUIDITY_RANK = 1500
US_SESSION_MIN_SHARE = 0.5  # a thin day is not a session (same rule as explain.us_price_matrix)

KR_REASONS = ("kind_preferred", "kind_spac", "kind_reit", "delisting_filing", "long_halt",
              "close_below_1000", "mcap_below_20b", "short_history", "size_liquidity")
US_REASONS = ("acquisition_target", "narrow_range", "listing_deficiency", "trading_gap",
              "short_history", "liquidity")
KR_RISK = KR_REASONS[:-1]
US_RISK = US_REASONS[:-1]
KR_MODEL_ID = "kr_daily_h20_v1"
US_MODEL_IDS = ("us_exploratory_20260929_r1_lightgbm", "us_exploratory_20260929_r1_ridge")
MODEL_MARKET = {KR_MODEL_ID: "KR", **{m: "US" for m in US_MODEL_IDS}}
KR_TABLES = ("daily_ohlcv", "daily_market_cap", "dart_filing_receipt_raw", "stock_master")
US_TABLES = ("prices_daily", "corp_actions", "filings_index", "universe_daily_v2")


class ScreenError(ValueError):
    """A model's screening cannot be computed; ``args[0]`` is the machine reason."""


# --------------------------------------------------------------------------- generic helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document_bytes(body: dict[str, Any]) -> bytes:
    return (json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def rank_desc(values: list[float | None]) -> list[int | None]:
    """``rank(method="min")`` with the largest value first; a missing value gets no rank."""
    present = sorted((v for v in values if v is not None), reverse=True)
    first: dict[float, int] = {}
    for i, v in enumerate(present):
        first.setdefault(v, i + 1)
    return [None if v is None else first[v] for v in values]


def month_first_sessions(sessions: Iterable[date]) -> list[date]:
    out: list[date] = []
    for d in sorted(sessions):
        if not out or (d.year, d.month) != (out[-1].year, out[-1].month):
            out.append(d)
    return out


def judgement_session(sessions: list[date], asof: date) -> date:
    """The last "first session of a month" on or before ``asof``."""
    firsts = [d for d in month_first_sessions(sessions) if d <= asof]
    if not firsts:
        raise ScreenError("no_month_first_session")
    return firsts[-1]


def previous_month_first(sessions: list[date], day: date) -> date | None:
    firsts = [d for d in month_first_sessions(sessions) if d < day]
    return firsts[-1] if firsts else None


def screen_order(rows: list[dict[str, Any]], passed: list[bool], top: int = SCREEN_TOP) -> list[dict[str, Any]]:
    """Original rank order with failing rows skipped; the first ``top``."""
    out = []
    for row, ok in zip(rows, passed, strict=True):
        if ok:
            out.append({"screen_rank": len(out) + 1, "rank": int(row["rank"]), "symbol": row["symbol"]})
            if len(out) == top:
                break
    return out


def _summary(rows: list[dict[str, Any]], reasons: tuple[str, ...], top_n: int) -> dict[str, Any]:
    by_reason = Counter(r for row in rows for r in row["reasons"])
    top_by_reason = Counter(r for row in rows if row["rank"] <= top_n for r in row["reasons"])
    passed = [row for row in rows if row["pass"]]
    return {
        "total": len(rows),
        "passed": len(passed),
        "excluded": len(rows) - len(passed),
        "by_reason": {r: by_reason.get(r, 0) for r in reasons},
        "top_n": top_n,
        "top_n_by_reason": {r: top_by_reason.get(r, 0) for r in reasons},
        "top_n_kept": sum(1 for row in passed if row["rank"] <= top_n),
    }


# --------------------------------------------------------------------------- KR (pure)
def kr_kind(symbol: str, name: str | None) -> list[str]:
    hits = []
    if not symbol.endswith(KR_PREFERRED_LAST_DIGIT):
        hits.append("kind_preferred")
    text = name or ""
    if re.search(KR_SPAC_RE, text):
        hits.append("kind_spac")
    if re.search(KR_REIT_RE, text):
        hits.append("kind_reit")
    return hits


def kr_filing_hit(title: str | None) -> bool:
    text = title or ""
    return bool(re.search(KR_FILING_INCLUDE_RE, text)) and not re.search(KR_FILING_EXCLUDE_RE, text)


def kr_sessions(ohlcv) -> list[date]:
    """Days on which more than 1,000 stocks have a price row."""
    import polars as pl

    counts = ohlcv.group_by("trade_date").agg(pl.col("ticker").n_unique().alias("n"))
    return sorted(counts.filter(pl.col("n") > KR_SESSION_MIN_TICKERS)["trade_date"].to_list())


def screen_kr(rankings: list[dict[str, Any]], *, asof: date, ohlcv, mcap, filings,
              master=None, top_n: int = 100) -> dict[str, Any]:
    """Screen the KR cross-section.

    ``ohlcv``: ``trade_date, ticker, close, volume`` (rows after ``asof`` are dropped here);
    ``mcap``: ``trade_date, ticker, market_cap, trading_value``; ``filings``: ``ticker, rcept_dt,
    rcept_no, report_nm``; ``master`` (optional): ``ticker, market, name`` for a missing report name.
    """
    import polars as pl

    ohlcv = ohlcv.filter(pl.col("trade_date") <= asof)
    sessions = kr_sessions(ohlcv)
    if not sessions or sessions[-1] != asof:
        raise ScreenError("asof_not_a_session")
    if len(sessions) < WINDOW_SESSIONS:
        raise ScreenError("window_shorter_than_252_sessions")
    window = sessions[-WINDOW_SESSIONS:]
    w = (ohlcv.filter(pl.col("trade_date").is_in(window))
         .unique(subset=["trade_date", "ticker"], keep="first", maintain_order=True))
    per = (w.group_by("ticker").agg(
        pl.len().alias("rows"),
        (pl.col("volume") == 0).sum().alias("zero_volume_days"),
        pl.col("close").filter(pl.col("trade_date") == asof).first().alias("close_k"),
    ))
    hist = {t: (n, z, c) for t, n, z, c in per.iter_rows()}

    mcap = mcap.filter(pl.col("trade_date") <= asof)
    mdates = sorted(mcap["trade_date"].unique().to_list())
    if not mdates:
        raise ScreenError("market_cap_empty")
    risk_mdate = mdates[-1]
    risk_cap = dict(mcap.filter(pl.col("trade_date") == risk_mdate).select("ticker", "market_cap").iter_rows())
    judge = judgement_session(sessions, asof)
    size_from = judge
    if judge not in set(mdates):  # 판정일 값이 스냅샷에 아직 없으면 전달 첫 세션 값 (06 §2 ③)
        size_from = previous_month_first(sessions, judge)
        if size_from is None:
            raise ScreenError("no_previous_month_first_session")
    size_dates = [d for d in mdates if d <= size_from]
    if not size_dates:
        raise ScreenError("market_cap_before_judgement_missing")
    size_mdate = size_dates[-1]
    liq_dates = size_dates[-LIQUIDITY_SESSIONS:]
    size_cap = dict(mcap.filter(pl.col("trade_date") == size_mdate).select("ticker", "market_cap").iter_rows())
    liq = dict(mcap.filter(pl.col("trade_date").is_in(liq_dates))
               .unique(subset=["trade_date", "ticker"], keep="first", maintain_order=True)
               .group_by("ticker").agg(pl.col("trading_value").fill_null(0).sum()).iter_rows())

    start = asof - timedelta(days=KR_FILING_DAYS)
    in_window = (filings.filter((pl.col("rcept_dt") > start) & (pl.col("rcept_dt") <= asof))
                 .sort("rcept_dt", "rcept_no"))
    hit_by: dict[str, list[dict[str, Any]]] = {}
    # Python ``re`` on purpose: the regexes are the spec's text, matched with the engine the tests use.
    for ticker, dt, no, title in in_window.select("ticker", "rcept_dt", "rcept_no", "report_nm").iter_rows():
        if kr_filing_hit(title):
            hit_by.setdefault(ticker, []).append({"rcept_no": no, "rcept_dt": dt.isoformat(), "report_nm": title})
    filing_basis = filings.filter(pl.col("rcept_dt") <= asof)["rcept_dt"].max()

    names = {}
    if master is not None:
        for t, mk, nm in master.select("ticker", "market", "name").iter_rows():
            names.setdefault((t, mk), nm)

    caps = [size_cap.get(r["symbol"]) for r in rankings]
    liqs = [(liq[r["symbol"]] / LIQUIDITY_SESSIONS) if r["symbol"] in liq else None for r in rankings]
    cap_rank, liq_rank = rank_desc(caps), rank_desc(liqs)
    rows = []
    for i, r in enumerate(rankings):
        sym = r["symbol"]
        name = r.get("name") or names.get((sym, r.get("market")))
        n_rows, zero, close_k = hist.get(sym, (0, 0, None))
        cap_k = risk_cap.get(sym)
        filed = hit_by.get(sym, [])
        reasons = kr_kind(sym, name)
        if filed:
            reasons.append("delisting_filing")
        if zero >= KR_HALT_DAYS:
            reasons.append("long_halt")
        if close_k is None or close_k < KR_MIN_CLOSE:
            reasons.append("close_below_1000")
        if cap_k is None or cap_k < KR_MIN_MCAP:
            reasons.append("mcap_below_20b")
        if n_rows < WINDOW_SESSIONS:
            reasons.append("short_history")
        size_ok = (cap_rank[i] is not None and cap_rank[i] <= KR_SIZE_RANK
                   and liq_rank[i] is not None and liq_rank[i] <= KR_LIQUIDITY_RANK)
        if not size_ok:
            reasons.append("size_liquidity")
        rows.append({
            "rank": int(r["rank"]), "symbol": sym, "market": r.get("market"), "pass": not reasons,
            "reasons": reasons,
            "values": {"name": name, "close_k": close_k, "market_cap_k": cap_k, "zero_volume_days": zero,
                       "price_rows": n_rows, "market_cap_size": caps[i], "market_cap_rank": cap_rank[i],
                       "trading_value_20": liqs[i], "trading_value_rank": liq_rank[i]},
            "filings": [f["rcept_no"] for f in filed],
        })
    passed = [row["pass"] for row in rows]
    return {
        "market": "KR", "status": "ok", "reason": None, "asof": asof.isoformat(),
        "dates": {"window_first_session": window[0].isoformat(), "window_last_session": window[-1].isoformat(),
                  "judgement_session": judge.isoformat(), "size_market_cap_date": size_mdate.isoformat(),
                  "liquidity_dates": [liq_dates[0].isoformat(), liq_dates[-1].isoformat()],
                  "risk_market_cap_date": risk_mdate.isoformat(),
                  "filing_window": [(start + timedelta(days=1)).isoformat(), asof.isoformat()],
                  "filing_basis_date": filing_basis.isoformat() if filing_basis else None},
        "summary": _summary(rows, KR_REASONS, top_n),
        "screened": screen_order(rows, passed),
        "filings": {sym: hit_by[sym] for sym in sorted({row["symbol"] for row in rows} & set(hit_by))},
        "rows": rows,
    }


# --------------------------------------------------------------------------- US (pure)
def us_sessions(prices) -> list[date]:

    per_day = prices.group_by("date").len().sort("date")
    if per_day.height == 0:
        raise ScreenError("price_rows_empty")
    median = float(per_day["len"].median())
    return [d for d, n in per_day.select("date", "len").iter_rows() if n >= US_SESSION_MIN_SHARE * median]


def _items(raw: str | None) -> set[str]:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


def us_filing_flags(filings) -> dict[int, dict[str, Any]]:
    """Per ``cik`` within the window: acquisition (target side) and listing-deficiency hits.

    ``filings``: ``cik, form, filing_date, file_number, items, accession`` already cut to the window.
    """
    by_cik: dict[int, list[tuple]] = {}
    for cik, form, day, number, items, acc in filings.select(
            "cik", "form", "filing_date", "file_number", "items", "accession").iter_rows():
        if cik is None:
            continue
        by_cik.setdefault(int(cik), []).append((form, day, number or "", _items(items), acc))
    out = {}
    for cik, rows in by_cik.items():
        rows.sort(key=lambda t: (t[1], t[4] or ""))
        a = [t for t in rows if t[0] in US_ACQ_A_FORMS]
        b = [t for t in rows if t[0] in US_ACQ_B_FORMS and t[2].startswith(US_ACQ_B_FILE_PREFIX)]
        c = [t for t in rows if t[0] in US_ACQ_C_FORMS]
        c_counts = False
        if c:
            issuer = any(t[0] in US_ACQ_C_ISSUER_FORMS for t in rows)
            first = min(t[1] for t in a + b + c)
            completed = any(t[0] == US_8K and US_ACQ_C_COMPLETION_ITEM in t[3] and t[1] > first for t in rows)
            blocked = any((t[0] == US_8K and t[3] & US_ACQ_C_BLOCK_ITEMS) or t[0] in US_ACQ_C_BLOCK_FORMS
                          for t in rows)
            c_counts = not issuer and not (completed and not blocked)
        acq = a + b + (c if c_counts else [])
        listing = [t for t in rows if (t[0] == US_8K and US_LISTING_ITEM in t[3]) or t[0] in US_LISTING_FORMS]
        if acq or listing:
            out[cik] = {
                "acquisition": sorted({t[4] for t in acq}),
                "acquisition_rules": [k for k, hit in (("a", a), ("b", b), ("c", c_counts)) if hit],
                "listing": sorted({t[4] for t in listing}),
            }
    return out


def us_cik_map(universe_at_a, symbols: list[str]) -> dict[str, int | None]:
    """``universe_daily_v2`` A-day ``cik``; a dotted class ticker without one takes its base ticker's."""
    known = {}
    for sym, cik in universe_at_a.select("symbol", "cik").iter_rows():
        if cik is not None:
            known.setdefault(sym, int(cik))
    out = {}
    for sym in symbols:
        cik = known.get(sym)
        if cik is None and "." in sym:
            cik = known.get(sym.split(".", 1)[0])
        out[sym] = cik
    return out


def screen_us(rankings: list[dict[str, Any]], *, asof: date, prices, adjusted, filings, universe,
              top_n: int = 100, filings_basis: date | None = None) -> dict[str, Any]:
    """Screen one US model's cross-section.

    ``prices``: raw ``date, symbol, close, volume``; ``adjusted``: ``date, symbol, adj_close, adj_volume``
    (split adjusted to ``asof``); ``filings``: ``cik, form, filing_date, file_number, items, accession``;
    ``universe``: ``universe_daily_v2`` rows ``date, symbol, cik``.
    """
    import polars as pl

    prices = prices.filter(pl.col("date") <= asof)
    sessions = us_sessions(prices)
    if not sessions or sessions[-1] != asof:
        raise ScreenError("asof_not_a_session")
    if len(sessions) < WINDOW_SESSIONS:
        raise ScreenError("window_shorter_than_252_sessions")
    window = sessions[-WINDOW_SESSIONS:]
    symbols = [r["symbol"] for r in rankings]
    w = (prices.filter(pl.col("date").is_in(window) & pl.col("symbol").is_in(symbols))
         .unique(subset=["date", "symbol"], keep="first", maintain_order=True)
         .with_columns(pl.col("close").cast(pl.Float64), pl.col("volume").cast(pl.Float64)))
    first = dict(w.group_by("symbol").agg(pl.col("date").min()).iter_rows())
    pos = {d: i for i, d in enumerate(window)}
    gap_stats = {}
    for sym, dates, closes, vols in (w.group_by("symbol").agg("date", "close", "volume").iter_rows()):
        start = pos[min(dates)]
        bad_rows = sum(1 for c, v in zip(closes, vols, strict=True)
                       if c is None or v is None or c <= 0 or v <= 0)
        missing = (len(window) - start) - len(dates)
        gap_stats[sym] = (missing, bad_rows)

    judge = judgement_session(sessions, asof)
    liq_window = [d for d in sessions if d <= judge][-LIQUIDITY_SESSIONS:]
    dv = dict(prices.filter(pl.col("date").is_in(liq_window) & pl.col("symbol").is_in(symbols))
              .unique(subset=["date", "symbol"], keep="first", maintain_order=True)
              .group_by("symbol")
              .agg((pl.col("close").cast(pl.Float64) * pl.col("volume").cast(pl.Float64)).fill_null(0).sum())
              .iter_rows())
    narrow_window = window[-US_NARROW_SESSIONS:]
    a = (adjusted.filter(pl.col("date").is_in(narrow_window) & pl.col("symbol").is_in(symbols))
         .unique(subset=["date", "symbol"], keep="first", maintain_order=True)
         .with_columns(pl.col("adj_close").cast(pl.Float64), pl.col("adj_volume").cast(pl.Float64)))
    narrow = {}
    for sym, dates, closes, vols in a.group_by("symbol").agg("date", "adj_close", "adj_volume").iter_rows():
        if len(set(dates)) != US_NARROW_SESSIONS or any(v is None or v <= 0 for v in vols) \
                or any(c is None or c <= 0 for c in closes):
            continue
        last = closes[dates.index(asof)]
        narrow[sym] = (max(closes) - min(closes)) / last

    universe_a = universe.filter(pl.col("date") == asof)
    if universe_a.height == 0:
        raise ScreenError("universe_has_no_asof_rows")
    ciks = us_cik_map(universe_a, symbols)
    fw = filings.filter((pl.col("filing_date") >= window[0]) & (pl.col("filing_date") <= asof))
    flags = us_filing_flags(fw)
    basis = filings_basis or filings.filter(pl.col("filing_date") <= asof)["filing_date"].max()

    dollar = [(dv.get(s, 0.0) / LIQUIDITY_SESSIONS) for s in symbols]
    liq_rank = rank_desc(dollar)
    rows = []
    for i, r in enumerate(rankings):
        sym = r["symbol"]
        cik = ciks.get(sym)
        flag = flags.get(cik, {}) if cik is not None else {}
        reasons = []
        if flag.get("acquisition"):
            reasons.append("acquisition_target")
        ratio = narrow.get(sym)
        if ratio is not None and ratio < US_NARROW_RANGE:
            reasons.append("narrow_range")
        if flag.get("listing"):
            reasons.append("listing_deficiency")
        first_day = first.get(sym)
        missing, bad = gap_stats.get(sym, (0, 0))
        if first_day is not None and (missing > 0 or bad > 0):
            reasons.append("trading_gap")
        if first_day is None or first_day > window[0]:
            reasons.append("short_history")
        if not (dollar[i] > 0 and liq_rank[i] is not None and liq_rank[i] <= US_LIQUIDITY_RANK):
            reasons.append("liquidity")
        rows.append({
            "rank": int(r["rank"]), "symbol": sym, "pass": not reasons, "reasons": reasons,
            "values": {"cik": cik, "filings_checked": cik is not None, "range_20": ratio,
                       "first_price_date": first_day.isoformat() if first_day else None,
                       "missing_sessions": missing, "bad_rows": bad, "dollar_volume_20": dollar[i],
                       "dollar_volume_rank": liq_rank[i], "acquisition_rules": flag.get("acquisition_rules", [])},
            "filings": sorted(set(flag.get("acquisition", [])) | set(flag.get("listing", []))),
        })
    passed = [row["pass"] for row in rows]
    summary = _summary(rows, US_REASONS, top_n)
    summary["filings_unchecked"] = sum(1 for row in rows if not row["values"]["filings_checked"])
    return {
        "market": "US", "status": "ok", "reason": None, "asof": asof.isoformat(),
        "dates": {"window_first_session": window[0].isoformat(), "window_last_session": window[-1].isoformat(),
                  "judgement_session": judge.isoformat(),
                  "liquidity_dates": [liq_window[0].isoformat(), liq_window[-1].isoformat()],
                  "narrow_dates": [narrow_window[0].isoformat(), narrow_window[-1].isoformat()],
                  "filing_basis_date": basis.isoformat() if basis else None},
        "summary": summary,
        "screened": screen_order(rows, passed),
        "rows": rows,
    }


# --------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class JobPin:
    market: str
    model_id: str
    native_manifest: Path
    input_sha256: str


def read_job_pins(jobs_config: Path, prepared_root: Path) -> dict[str, JobPin]:
    body = json.loads(Path(jobs_config).read_text(encoding="utf-8"))
    out = {}
    for raw in body.get("jobs", []):
        path = Path(raw["native_manifest"])
        out[raw["model_id"]] = JobPin(market=raw["market"], model_id=raw["model_id"],
                                      native_manifest=path if path.is_absolute() else Path(prepared_root) / path,
                                      input_sha256=raw["input_sha256"])
    return out


def _listing_sha(directory: Path) -> str:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise ScreenError(f"{directory.name}_files_missing")
    lines = [f"{p.relative_to(directory).as_posix()}\t{p.stat().st_size}" for p in files]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def kr_table_dirs(kr_root: Path, native: dict[str, Any]) -> dict[str, Path]:
    snapshot, source = native.get("snapshot_date"), native.get("source")
    if not snapshot or not source:
        raise ScreenError("native_manifest_without_snapshot")
    base = Path(kr_root) / "raw" / "raw_postgres" / f"snapshot_date={snapshot}" / f"source={source}"
    return {t: base / t for t in KR_TABLES}


def kr_input_record(directory: Path) -> dict[str, str]:
    """Hash of one KR raw table: its export table manifest, else a file listing (as explain does)."""
    manifest = directory.parent / "_manifests" / "table_manifests" / f"{directory.name}.json"
    if manifest.is_file():
        return {"path": str(directory), "sha256": sha256_file(manifest), "kind": "table_manifest"}
    return {"path": str(directory), "sha256": _listing_sha(directory), "kind": "file_listing"}


@dataclass
class KrInputs:
    ohlcv: Any
    mcap: Any
    filings: Any
    master: Any
    records: dict[str, dict[str, str]] = field(default_factory=dict)


def load_kr_inputs(dirs: dict[str, Path], asof: date) -> KrInputs:
    import polars as pl

    records = {t: kr_input_record(Path(d)) for t, d in dirs.items()}
    start = asof - timedelta(days=430)  # 252 sessions + the previous month's first session

    def scan(t: str):
        return pl.scan_parquet(str(Path(dirs[t])) + "/**/*.parquet")

    ohlcv = (scan("daily_ohlcv").filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= asof))
             .select("trade_date", "ticker", "close", "volume").collect())
    mcap = (scan("daily_market_cap").filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= asof))
            .select("trade_date", "ticker", "market_cap", "trading_value").collect())
    fstart = asof - timedelta(days=KR_FILING_DAYS + 1)
    filings = (scan("dart_filing_receipt_raw").filter((pl.col("rcept_dt") > fstart) & (pl.col("rcept_dt") <= asof))
               .select("ticker", "rcept_dt", "rcept_no", "report_nm").collect())
    master = scan("stock_master").select("ticker", "market", "name").collect()
    return KrInputs(ohlcv, mcap, filings, master, records)


def us_snapshot_files(native: dict[str, Any], us_root: Path) -> dict[str, dict[str, str]]:
    """The four pinned US tables, checked against the prepare manifest's file hashes."""
    files = native.get("source_snapshot_files") or {}
    out = {}
    for table in US_TABLES:
        entries = files.get(table)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ScreenError(f"{table}_snapshot_not_single_file")
        path = Path(us_root) / entries[0]["path"]
        if sha256_file(path) != entries[0]["sha256"]:
            raise ScreenError(f"{table}_changed_since_prepare")
        out[table] = {"path": entries[0]["path"], "sha256": entries[0]["sha256"], "kind": "file"}
        out[table]["_abs"] = str(path)
    return out


@dataclass
class UsInputs:
    prices: Any
    adjusted: Any
    filings: Any
    universe: Any
    basis: date | None
    records: dict[str, dict[str, str]] = field(default_factory=dict)


def load_us_inputs(paths: dict[str, Path], asof: date, records: dict[str, dict[str, str]] | None = None) -> UsInputs:
    import polars as pl

    from modeler.serving.explain import _SliceLake
    from modeler.us.prices import adjusted_daily

    start = asof - timedelta(days=430)
    px = (pl.scan_parquet(paths["prices_daily"]).filter((pl.col("date") >= start) & (pl.col("date") <= asof))
          .collect())
    nstart = asof - timedelta(days=60)
    ca = (pl.scan_parquet(paths["corp_actions"])
          .filter((pl.col("kind") == "split") & (pl.col("ex_date") >= nstart)).collect())
    recent = px.filter(pl.col("date") >= nstart)
    adjusted = (adjusted_daily(_SliceLake(recent, ca), base_date=asof, security_boundaries=False)
                .select("date", "symbol", "adj_close", "adj_volume").collect())
    prices = px.select("date", "symbol", "close", "volume")
    del px, recent
    fl = pl.scan_parquet(paths["filings_index"])
    basis = fl.filter(pl.col("filing_date") <= asof).select(pl.col("filing_date").max()).collect().item()
    forms = (US_ACQ_A_FORMS | US_ACQ_B_FORMS | US_ACQ_C_FORMS | US_ACQ_C_ISSUER_FORMS | US_ACQ_C_BLOCK_FORMS
             | US_LISTING_FORMS | {US_8K})
    filings = (fl.filter((pl.col("filing_date") >= start) & (pl.col("filing_date") <= asof)
                         & pl.col("form").is_in(sorted(forms)))
               .select("cik", "form", "filing_date", "file_number", "items", "accession").collect())
    universe = (pl.scan_parquet(paths["universe_daily_v2"]).filter(pl.col("date") == asof)
                .select("date", "symbol", "cik").collect())
    return UsInputs(prices, adjusted, filings, universe, basis, dict(records or {}))


# --------------------------------------------------------------------------- document
def _report_model(report: dict[str, Any], market: str, model_id: str) -> dict[str, Any]:
    for item in report.get("markets", []):
        if item.get("market") == market and item.get("model_id") == model_id:
            if not item.get("rankings"):
                raise ScreenError("report_has_no_rankings")
            return item
    raise ScreenError("model_not_in_report")


def failed_model(market: str, reason: str, asof: str | None = None) -> dict[str, Any]:
    return {"market": market, "status": "failed", "reason": reason, "asof": asof, "inputs": {}, "dates": {},
            "summary": {}, "screened": [], "rows": []}


def _reason(exc: BaseException) -> str:
    if isinstance(exc, ScreenError):
        return str(exc.args[0])
    return type(exc).__name__


def rollup_status(models: dict[str, dict[str, Any]]) -> str:
    states = [m["status"] for m in models.values()]
    if states and all(s == "ok" for s in states):
        return "ok"
    if any(s == "ok" for s in states):
        return "partial"
    return "failed"


def build_document(*, unit: str, invocation_id: str, report_sha256: str, release: str,
                   models: dict[str, dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    return {"schema": SCHEMA, "rule_version": RULE_VERSION, "screen_top": SCREEN_TOP, "unit": unit,
            "invocation_id": invocation_id, "report_sha256": report_sha256, "release": release,
            "status": rollup_status(models), "reason": reason, "models": models}


def failure_document(*, unit: str, invocation_id: str, report_sha256: str, release: str, reason: str,
                     models: tuple[str, ...] = tuple(MODEL_MARKET)) -> dict[str, Any]:
    """Stdlib-only document for a stage that failed before or while computing (timeout, crash)."""
    body = {m: failed_model(MODEL_MARKET[m], reason) for m in models}
    doc = build_document(unit=unit, invocation_id=invocation_id, report_sha256=report_sha256, release=release,
                         models=body, reason=reason)
    doc["status"] = "failed"
    return doc


def _public_records(records: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {t: {k: v for k, v in rec.items() if not k.startswith("_")} for t, rec in sorted(records.items())}


def screen_report(*, unit: str, invocation_id: str, report_bytes: bytes, release: str,
                  pins: dict[str, JobPin] | None, kr_root: Path | None, us_root: Path | None,
                  top_n: int = 100, kr_dirs: dict[str, Path] | None = None,
                  us_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Whole document for one day. One model never stops another.

    ``kr_dirs`` / ``us_paths`` replace the lake paths (copied inputs, tests): then the pins are not
    needed for that market and the hashes are taken from the given files.
    """
    report = json.loads(report_bytes)
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    models: dict[str, dict[str, Any]] = {}
    us_cache: dict[str, UsInputs] = {}
    for model_id, market in MODEL_MARKET.items():
        asof_text = None
        try:
            report_model = _report_model(report, market, model_id)
            asof_text = str(report_model["feature_asof_date"])
            asof = date.fromisoformat(asof_text)
            rankings = report_model["rankings"]
            if market == "KR":
                if kr_dirs is None:
                    pin = (pins or {}).get(model_id)
                    if pin is None:
                        raise ScreenError("model_not_pinned")
                    if kr_root is None:
                        raise ScreenError("kr_root_not_configured")
                    native = json.loads(pin.native_manifest.read_text(encoding="utf-8"))
                    dirs = kr_table_dirs(kr_root, native)
                else:
                    dirs = kr_dirs
                inputs = load_kr_inputs(dirs, asof)
                block = screen_kr(rankings, asof=asof, ohlcv=inputs.ohlcv, mcap=inputs.mcap,
                                  filings=inputs.filings, master=inputs.master, top_n=top_n)
                block["inputs"] = _public_records(inputs.records)
            else:
                if us_paths is None:
                    pin = (pins or {}).get(model_id)
                    if pin is None:
                        raise ScreenError("model_not_pinned")
                    if us_root is None:
                        raise ScreenError("us_root_not_configured")
                    native = json.loads(pin.native_manifest.read_text(encoding="utf-8"))
                    key = pin.input_sha256
                    if key not in us_cache:
                        us_cache.clear()
                        recs = us_snapshot_files(native, us_root)
                        paths = {t: Path(rec["_abs"]) for t, rec in recs.items()}
                        us_cache[key] = load_us_inputs(paths, asof, recs)
                else:
                    key = "given"
                    if key not in us_cache:
                        recs = {t: {"path": str(p), "sha256": sha256_file(Path(p)), "kind": "file"}
                                for t, p in us_paths.items()}
                        us_cache[key] = load_us_inputs({t: Path(p) for t, p in us_paths.items()}, asof, recs)
                shared = us_cache[key]
                block = screen_us(rankings, asof=asof, prices=shared.prices, adjusted=shared.adjusted,
                                  filings=shared.filings, universe=shared.universe, top_n=top_n,
                                  filings_basis=shared.basis)
                block["inputs"] = _public_records(shared.records)
            models[model_id] = block
        except Exception as exc:
            models[model_id] = failed_model(market, _reason(exc), asof_text)
    return build_document(unit=unit, invocation_id=invocation_id, report_sha256=report_sha, release=release,
                          models=models)


def atomic_write(path: Path, body: dict[str, Any]) -> str:
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = document_bytes(body)
    fd, temp = tempfile.mkstemp(prefix=".screen-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- reading (render, explain)
def screened_targets(doc: dict[str, Any] | None, model_id: str) -> list[dict[str, Any]]:
    """The screened top rows ``[{screen_rank, rank, symbol}]`` of one usable model block, else []."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("rule_version") != RULE_VERSION:
        return []
    block = (doc.get("models") or {}).get(model_id)
    if not isinstance(block, dict) or block.get("status") != "ok":
        return []
    out = []
    for item in block.get("screened") or []:
        if isinstance(item, dict) and isinstance(item.get("rank"), int) and isinstance(item.get("symbol"), str):
            out.append(item)
    return out


# --------------------------------------------------------------------------- CLI (child process)
def _run(args: argparse.Namespace) -> int:
    report_bytes = Path(args.report).read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != args.report_sha256:
        print(json.dumps({"status": "failed", "reason": "report_changed"}))
        return 2
    if sha256_file(Path(args.jobs_config)) != args.jobs_config_sha256:
        print(json.dumps({"status": "failed", "reason": "jobs_config_changed"}))
        return 2
    pins = read_job_pins(Path(args.jobs_config), Path(args.prepared_root))
    document = screen_report(
        unit=args.report_date, invocation_id=args.invocation_id, report_bytes=report_bytes,
        release=args.release, pins=pins, kr_root=Path(args.kr_root) if args.kr_root else None,
        us_root=Path(args.us_root) if args.us_root else None, top_n=args.top_n)
    atomic_write(Path(args.output), document)
    print(json.dumps({"status": document["status"],
                      "models": {k: v["status"] for k, v in document["models"].items()}}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-screen")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute screen-D.json into --output")
    run.add_argument("--report-date", required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--report-sha256", required=True)
    run.add_argument("--invocation-id", required=True)
    run.add_argument("--release", required=True)
    run.add_argument("--jobs-config", type=Path, required=True)
    run.add_argument("--jobs-config-sha256", required=True)
    run.add_argument("--prepared-root", type=Path, required=True)
    run.add_argument("--kr-root", type=Path)
    run.add_argument("--us-root", type=Path)
    run.add_argument("--top-n", type=int, default=100)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.report_date):
        return 2
    try:
        return _run(args)
    except Exception as exc:  # only the class goes out; paths may be in the message
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
