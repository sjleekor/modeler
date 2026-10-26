"""Daily briefing 'explain' stage: why a stock sits where it does, and how its price moved.

Plan: ``my/milestones/common/20261026_briefing_charts_and_reasons`` (§4 method, §5.2.1 format).

For the three briefing models this module reads the SAME pinned inputs inference used, checks that
it reproduces the report's scores, and writes ``explain-<D>.json`` (``briefing-explain.v1``):

* per top-100 stock, the score change ``Δ_g`` when one factor group's design columns are replaced by
  the day's baseline (mean of the comparison set; mode for 0/1, missing-flag and two-valued columns);
* the most extreme feature percentiles (ties above 5% excluded);
* a 61-session price index (start = 100), a comparison-set equal-weight mean line, and 5/20/60
  session returns, all cut at the model's own as-of session (never later).

It never changes a model, a score or a rank, and never touches ``report-<D>.json``.
The document is the only interface to the render side.  Heavy libraries (numpy, polars, joblib) are
imported inside the functions that need them, so the coordinator can import this module's
document helpers without paying for them.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "briefing-explain.v1"
METHOD_VERSION = "occlusion-mean-mode-v1"
GROUP_MAP_VERSION = "groups-2026-10-26"
SCORE_ATOL = 1e-12
TOP_N = 100
N_SESSIONS = 61
RETURN_HORIZONS = (5, 20, 60)
TIE_MAX = 0.05
TOP_FEATURES_K = 4
KR_JUMP = (0.67, 1.33)  # price limit 30% x 1.10, as ``etl/quality.py``
US_JUMP = (0.5, 2.0)
KR_MODEL_ID = "kr_daily_h20_v1"
US_MODEL_IDS = ("us_exploratory_20260929_r1_lightgbm", "us_exploratory_20260929_r1_ridge")
MODEL_MARKET = {KR_MODEL_ID: "KR", **{m: "US" for m in US_MODEL_IDS}}

GROUPS = (
    ("price", "가격 흐름"),
    ("risk", "변동성·위험"),
    ("liq", "거래·유동성·규모"),
    ("flow", "수급"),
    ("value", "가치·수익성"),
    ("event", "이벤트·공시"),
    ("holders", "지분·기관"),
    ("micro", "옵션·호가 구조"),
)
GID = tuple(g for g, _ in GROUPS)
GLABEL = dict(GROUPS)


def _m(gid: str, names: str) -> dict[str, str]:
    return {n: gid for n in names.split()}


# Explicit feature -> group tables. A feature and its ``_isna``/``_rank`` design columns share a group.
# KR has no macro_beta_* and no hc_* feature, and only one own_* feature (P1 F1).
KR_FEATURE_GROUP = {
    **_m("price", "px_ret_1d px_ret_5d px_ret_20d px_ret_60d px_mom_20_60 px_gap_vs_ma20 px_dist_52w_high "
                  "px_reversal_5d px_maxret_20d"),
    **_m("risk", "px_vol_20d px_vol_60d px_high_low_range_20d px_idio_vol_60d px_is_halted px_halt_ratio_20d"),
    **_m("liq", "px_turnover px_turnover_ma20 px_amihud_20d fin_log_mcap"),
    **_m("flow", "flow_foreign_netbuy_sum_5d flow_foreign_netbuy_sum_20d flow_inst_netbuy_sum_5d "
                 "flow_inst_netbuy_sum_20d flow_indiv_netbuy_sum_5d flow_indiv_netbuy_sum_20d "
                 "flow_foreign_holding_chg_5d flow_foreign_holding_chg_20d flow_foreign_netbuy_z_20d "
                 "flow_inst_netbuy_z_20d flow_short_avg_price flow_short_selling_volume "
                 "flow_short_selling_value flow_individual_netbuy_to_volume_5d "
                 "flow_individual_netbuy_to_volume_20d"),
    **_m("value", "fin_roa fin_roe fin_operating_margin fin_debt_to_equity fin_equity_ratio fin_ocf_to_assets "
                  "fin_cash_ratio fin_asset_turnover fin_is_negative_equity fin_has_fs"),
    **_m("event", "ev_amendment_ratio_1y ev_filing_burst_60d"),
    **_m("holders", "own_major_filing_60d"),
}
US_FEATURE_GROUP = {
    **_m("price", "mom_12_1 mom_6_1 mom_1m rev_1w max_ret_1m"),
    **_m("risk", "rv_20 rv_60 idio_vol_60 beta_252"),
    **_m("liq", "log_dvol_20 amihud_20 mcap_rank"),
    **_m("flow", "si_ratio dtc si_chg sv_share_20 ftd_share_20 ftd_days_20 ftd_chg"),
    **_m("value", "bm ep_ttm cfp_ttm sp_ttm roa_ttm roe_ttm gpa opm_ttm asset_growth accruals net_issuance "
                  "div_yield buyback_yield"),
    **_m("event", "sue_last days_since_earn days_to_earn n_estimates n_8k_90 filing_lag"),
    **_m("holders", "ins_netbuy_90 ins_cluster_90 ins_officer_buy_90 sp500_member sp500_days_since_add "
                    "inst_n_log inst_breadth_chg inst_shares_chg"),
    **_m("micro", "iv_rank iv_hv_spread iv_isna cancel_ratio_20 hidden_share_20 oddlot_share_20 fill_ratio_20"),
}
# 0/1 design columns that the KR rank transform leaves as raw flags: baseline = mode, never a rank.
KR_FLAG_FEATURES = frozenset({"px_is_halted", "fin_is_negative_equity", "fin_has_fs"})

KR_NAMES = {
    "px_ret_1d": "하루 수익률", "px_ret_5d": "5일 수익률", "px_ret_20d": "20일 수익률", "px_ret_60d": "60일 수익률",
    "px_mom_20_60": "20일·60일 수익률 차이", "px_vol_20d": "최근 한 달 변동성", "px_vol_60d": "최근 석 달 변동성",
    "px_high_low_range_20d": "한 달 고저 폭", "px_turnover": "당일 거래대금", "px_turnover_ma20": "한 달 평균 거래대금",
    "px_amihud_20d": "거래 대비 가격 충격(아미후드)", "px_gap_vs_ma20": "한 달 평균가 대비 위치",
    "px_dist_52w_high": "52주 고가와의 거리", "px_is_halted": "거래정지 여부", "px_halt_ratio_20d": "한 달 거래정지 비율",
    "flow_foreign_netbuy_sum_5d": "5일 외국인 순매수", "flow_foreign_netbuy_sum_20d": "20일 외국인 순매수",
    "flow_inst_netbuy_sum_5d": "5일 기관 순매수", "flow_inst_netbuy_sum_20d": "20일 기관 순매수",
    "flow_indiv_netbuy_sum_5d": "5일 개인 순매수", "flow_indiv_netbuy_sum_20d": "20일 개인 순매수",
    "flow_foreign_holding_chg_5d": "5일 외국인 보유 변화", "flow_foreign_holding_chg_20d": "20일 외국인 보유 변화",
    "flow_foreign_netbuy_z_20d": "외국인 순매수 튐 정도", "flow_inst_netbuy_z_20d": "기관 순매수 튐 정도",
    "flow_short_avg_price": "공매도 평균 체결가", "flow_short_selling_volume": "공매도 수량",
    "flow_short_selling_value": "공매도 금액", "fin_roa": "총자산이익률", "fin_roe": "자기자본이익률",
    "fin_operating_margin": "영업이익률", "fin_debt_to_equity": "부채비율", "fin_equity_ratio": "자기자본비율",
    "fin_ocf_to_assets": "영업현금흐름/자산", "fin_cash_ratio": "현금 비중", "fin_asset_turnover": "자산회전율",
    "fin_is_negative_equity": "자본잠식 여부", "fin_has_fs": "재무제표 유무", "px_reversal_5d": "5일 되돌림",
    "px_maxret_20d": "한 달 중 최대 일간 상승률", "px_idio_vol_60d": "종목 고유 변동성",
    "flow_individual_netbuy_to_volume_5d": "5일 개인 순매수/거래량",
    "flow_individual_netbuy_to_volume_20d": "20일 개인 순매수/거래량", "fin_log_mcap": "시가총액 규모",
    "ev_amendment_ratio_1y": "정정 공시 비율", "ev_filing_burst_60d": "공시 몰림",
    "own_major_filing_60d": "대량보유 공시",
}
US_NAMES = {
    "mom_12_1": "12개월 수익률(최근 1개월 제외)", "mom_6_1": "6개월 수익률(최근 1개월 제외)", "mom_1m": "1개월 수익률",
    "rev_1w": "1주 되돌림", "max_ret_1m": "한 달 중 최대 일간 상승률", "rv_20": "20일 변동성", "rv_60": "60일 변동성",
    "idio_vol_60": "종목 고유 변동성", "beta_252": "시장 민감도(베타)", "log_dvol_20": "20일 거래대금",
    "amihud_20": "거래 대비 가격 충격", "mcap_rank": "시가총액 순위", "bm": "장부가/시가", "ep_ttm": "이익/주가",
    "cfp_ttm": "현금흐름/주가", "sp_ttm": "매출/주가", "roa_ttm": "총자산이익률", "roe_ttm": "자기자본이익률",
    "gpa": "매출총이익/자산", "opm_ttm": "영업이익률", "asset_growth": "자산 증가율", "accruals": "발생액",
    "net_issuance": "순주식발행", "div_yield": "배당수익률", "buyback_yield": "자사주매입 수익률",
    "sue_last": "직전 실적 서프라이즈", "days_since_earn": "직전 실적발표 후 경과일",
    "days_to_earn": "다음 실적발표까지 일수", "n_estimates": "애널리스트 추정 수", "ins_netbuy_90": "내부자 90일 순매수",
    "ins_cluster_90": "내부자 매수 집중", "ins_officer_buy_90": "임원 90일 매수", "si_ratio": "공매도 잔고 비율",
    "dtc": "공매도 청산 소요일", "si_chg": "공매도 잔고 변화", "sv_share_20": "20일 공매도 거래 비중",
    "n_8k_90": "90일 8-K 공시 수", "filing_lag": "공시 지연", "iv_rank": "내재변동성 순위",
    "iv_hv_spread": "내재-실현 변동성 차이", "iv_isna": "옵션 데이터 없음 표시", "sp500_member": "S&P500 편입 여부",
    "sp500_days_since_add": "S&P500 편입 후 경과일", "ftd_share_20": "20일 결제실패 비중",
    "ftd_days_20": "20일 결제실패 일수", "ftd_chg": "결제실패 변화", "cancel_ratio_20": "20일 주문 취소 비율",
    "hidden_share_20": "20일 숨은 주문 비중", "oddlot_share_20": "20일 소액 주문 비중",
    "fill_ratio_20": "20일 체결 비율", "inst_n_log": "기관 보유자 수", "inst_breadth_chg": "기관 보유 폭 변화",
    "inst_shares_chg": "기관 보유 주식 변화",
}
KR_SCOPE = {"KOSPI": "같은 시장(코스피)", "KOSDAQ": "같은 시장(코스닥)"}
US_SCOPE = "US 적격 종목"

RET_NOTE = {
    "price_jump_suspect_in_span": "의심일을 가로지름",
    "missing_asof": "기준일 가격 없음(정지·누락)",
    "missing_start_or_short_history": "시작일 가격 없음 또는 이력 부족",
    "window_short": "이력 부족",
}
START_HALTED_NOTE = "정지 중 시작"


class ExplainError(ValueError):
    """A model's explain (or price) part cannot be computed; ``args[0]`` is the machine reason."""


# --------------------------------------------------------------------------- generic helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, body: Any) -> str:
    """Write ``body`` next to ``path`` and rename it into place; returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fd, temp = tempfile.mkstemp(prefix=".explain-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- groups / baseline
def base_feature(design_col: str, features: tuple[str, ...] | list[str]) -> str:
    """The feature a design column belongs to (``f``, ``f_isna`` or, for US, ``f_rank``)."""
    known = set(features)
    if design_col in known:
        return design_col
    for suffix in ("_isna", "_rank"):
        if design_col.endswith(suffix) and design_col[: -len(suffix)] in known:
            return design_col[: -len(suffix)]
    raise ExplainError(f"design column has no feature: {design_col}")


def group_columns(design, features, feature_group: dict[str, str]) -> dict[str, Any]:
    """Explicit design column -> group map. Every column lands in exactly one group, or this raises."""
    import numpy as np

    if set(feature_group) != set(features):
        raise ExplainError("group table and model features differ: "
                           f"{sorted(set(feature_group) ^ set(features))[:5]}")
    unknown = set(feature_group.values()) - set(GID)
    if unknown:
        raise ExplainError(f"unknown group id: {sorted(unknown)}")
    seen: list[int] = []
    idx: dict[str, list[int]] = {g: [] for g in GID}
    for j, col in enumerate(design):
        idx[feature_group[base_feature(col, features)]].append(j)
        seen.append(j)
    if sorted(seen) != list(range(len(design))) or sum(len(v) for v in idx.values()) != len(design):
        raise ExplainError("a design column is mapped zero or several times")
    return {g: np.array(v, dtype=int) for g, v in idx.items()}


def baseline(X, design, mode_cols=frozenset()):
    """Per-column baseline of one comparison set.

    Mean for continuous columns. Mode (NaN excluded, smallest value on a tie) for ``*_isna`` columns,
    for the 0/1 flags in ``mode_cols`` and for any column with at most two distinct values (P1 F6) —
    so no state that the real inputs do not contain is ever created.
    """
    import numpy as np

    out = np.full(X.shape[1], np.nan)
    for j, col_name in enumerate(design):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        values, counts = np.unique(col, return_counts=True)
        if col_name.endswith("_isna") or col_name in mode_cols or len(values) <= 2:
            out[j] = values[int(np.argmax(counts))]
        else:
            out[j] = float(col.mean())
    return out


def occlude(predict: Callable[[Any], Any], Xtop, Btop, gcols: dict[str, Any]):
    """``f0`` (n,) and ``Δ_g`` (n, len(GID)) in ``GID`` order; NaN for a group with no columns."""
    import numpy as np

    n = len(Xtop)
    blocks, used = [Xtop], []
    for g in GID:
        cols = gcols[g]
        if len(cols) == 0:
            continue
        masked = Xtop.copy()
        masked[:, cols] = Btop[:, cols]
        blocks.append(masked)
        used.append(g)
    pred = np.asarray(predict(np.vstack(blocks)), dtype=float)
    f0 = pred[:n]
    delta = np.full((n, len(GID)), np.nan)
    for k, g in enumerate(used):
        delta[:, GID.index(g)] = f0 - pred[(1 + k) * n:(2 + k) * n]
    return f0, delta


def tie_shares(X, cols: list[int], labels):
    """Share of the comparison set holding the same value, per cell of the given columns."""
    import numpy as np

    shares = np.zeros(X.shape)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        for j in cols:
            _, inverse, counts = np.unique(X[idx, j], return_inverse=True, return_counts=True)
            shares[idx, j] = counts[inverse.reshape(-1)] / len(idx)
    return shares


def top_features(xrow, rank_cols: dict[str, int], isna_cols: dict[str, int], names: dict[str, str],
                 scope: str, excluded: set[str], tie_row, k: int = TOP_FEATURES_K) -> list[dict[str, Any]]:
    """The k most extreme observed percentiles of one stock.

    Skipped: values that were missing (``*_isna`` = 1), 0/1 and two-valued columns, and columns where more
    than 5% of the comparison set share the value (P1 F5: ``rank(min)`` makes 'lower 1%' out of ties).
    """
    import numpy as np

    cands = []
    for feature, j in rank_cols.items():
        if feature in excluded:
            continue
        if isna_cols.get(feature) is not None and xrow[isna_cols[feature]] == 1:
            continue
        p = float(xrow[j])
        if np.isnan(p) or tie_row[j] > TIE_MAX:
            continue
        cands.append((abs(p - 0.5), feature, p))
    cands.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for _, feature, p in cands[:k]:
        side = "low" if p < 0.5 else "high"
        out.append({"feature": feature, "label": names.get(feature, feature),
                    "pct": p if side == "low" else 1.0 - p, "side": side, "scope": scope})
    return out


@dataclass
class CrossSection:
    f0: Any
    delta: Any  # (n_top, len(GID))
    top_index: list[int]
    top_feats: list[list[dict[str, Any]]]
    max_abs_score_diff: float = 0.0


def explain_cross_section(*, predict: Callable[[Any], Any], X, design, features, feature_group: dict[str, str],
                          keys: list[Any], symbols: list[str], comparison, report_keys: list[Any],
                          report_scores: list[float], names: dict[str, str], scope_for: Callable[[Any], str],
                          flag_features: frozenset[str] = frozenset(), top_n: int = TOP_N) -> CrossSection:
    """Reproduce the report's scores, then occlude the top ``top_n``. Raises ``ExplainError`` on mismatch.

    ``keys[i]`` identifies matrix row ``i`` (KR ``(ticker, market)``, US ``symbol``); ``report_keys`` and
    ``report_scores`` are the report's ranking in order. ``comparison[i]`` is the comparison set of the row
    (KR market, US one set): the baseline and tie shares are taken inside it.
    """
    import numpy as np

    gcols = group_columns(design, features, feature_group)
    scores = np.asarray(predict(X), dtype=float)
    if len(scores) != len(keys):
        raise ExplainError("score_row_count_mismatch")
    if len(set(keys)) != len(keys) or set(keys) != set(report_keys) or len(report_keys) != len(keys):
        raise ExplainError("report_keys_differ_from_input")
    by_key = dict(zip(report_keys, report_scores, strict=True))
    diff = float(max(abs(scores[i] - by_key[keys[i]]) for i in range(len(keys))))
    if not diff <= SCORE_ATOL:
        raise ExplainError(f"score_mismatch:{diff:.3e}")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], symbols[i]))
    if [keys[i] for i in order] != list(report_keys):
        raise ExplainError("rank_order_mismatch")
    position = {key: i for i, key in enumerate(keys)}
    top_index = [position[key] for key in report_keys[:top_n]]

    comparison = np.asarray(comparison)
    base_by = {label: baseline(X[comparison == label], design, {c for c in design if c in flag_features})
               for label in np.unique(comparison)}
    Xtop = X[top_index]
    Btop = np.vstack([base_by[comparison[i]] for i in top_index])
    f0, delta = occlude(predict, Xtop, Btop, gcols)
    top_diff = float(np.max(np.abs(f0 - scores[top_index]))) if len(top_index) else 0.0
    if not top_diff <= SCORE_ATOL:
        raise ExplainError(f"occlusion_baseline_mismatch:{top_diff:.3e}")

    rank_cols = {f: design.index(f) if f in design else design.index(f"{f}_rank") for f in features}
    isna_cols = {f: design.index(f"{f}_isna") for f in features if f"{f}_isna" in design}
    ties = tie_shares(X, sorted(rank_cols.values()), comparison)
    binary = set(flag_features)
    for feature, j in rank_cols.items():
        col = X[:, j]
        if len(np.unique(col[~np.isnan(col)])) <= 2:
            binary.add(feature)
    feats = [top_features(X[i], rank_cols, isna_cols, names, scope_for(comparison[i]), binary, ties[i])
             for i in top_index]
    return CrossSection(f0=f0, delta=delta, top_index=top_index, top_feats=feats,
                        max_abs_score_diff=max(diff, top_diff))


# --------------------------------------------------------------------------- price paths
def suspect_matrix(C, lo: float, hi: float):
    """Mark the day on which a valid observation jumps by a ratio outside (lo, hi) vs the previous one."""
    import numpy as np

    flags = np.zeros(C.shape, dtype=bool)
    for i in range(C.shape[0]):
        valid = np.where(~np.isnan(C[i]))[0]
        for a, b in zip(valid[:-1], valid[1:]):
            ratio = C[i, b] / C[i, a]
            if ratio <= lo or ratio >= hi:
                flags[i, b] = True
    return flags


def benchmark_series(C, members, lo: float, hi: float):
    """Equal-weight mean of adjacent-session simple returns over ``members``, chained from 100.

    A stock with no value that day (or the day before) and a day flagged as a jump drop out of that
    day's mean. No member left means a 0 return for the line only (stock rows are never zero-filled).
    """
    import numpy as np

    sub = C[members]
    ratio = sub[:, 1:] / sub[:, :-1]
    ratio = np.where((ratio <= lo) | (ratio >= hi), np.nan, ratio) - 1.0
    count = (~np.isnan(ratio)).sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(ratio, axis=0)
    mean = np.where(count > 0, mean, 0.0)
    line = np.empty(C.shape[1])
    line[0] = 100.0
    line[1:] = 100.0 * np.cumprod(1.0 + mean)
    return line, count


def returns_for(C_row, carry_row, susp_row, horizons=RETURN_HORIZONS):
    """``(ret, why, carried)`` per horizon. Blank (None) with a reason, never zero-filled.

    Start price: the day's own close, or — when the start day was a halt — the carried close of the
    last trade (P1 F7). The end day (as-of) halted means blank.
    """
    import numpy as np

    T = len(C_row)
    asof = T - 1
    ret: dict[int, float | None] = {}
    why: dict[int, str | None] = {}
    carried: dict[int, bool] = {}
    for h in horizons:
        t0 = asof - h
        carried[h] = False
        if t0 < 0:
            ret[h], why[h] = None, "window_short"
        elif np.isnan(C_row[asof]):
            ret[h], why[h] = None, "missing_asof"
        else:
            start = C_row[t0]
            if np.isnan(start) and not np.isnan(carry_row[t0]):
                start, carried[h] = carry_row[t0], True
            if np.isnan(start):
                ret[h], why[h] = None, "missing_start_or_short_history"
            elif susp_row[t0 + 1:asof + 1].any():
                ret[h], why[h] = None, "price_jump_suspect_in_span"
            else:
                ret[h], why[h] = float(C_row[asof] / start - 1.0), None
    return ret, why, carried


def price_row(C_row, carry_row, halt_row, susp_row, bench_line, dates: list[str]) -> dict[str, Any]:
    """One stock's §5.2.1 price fields from its row of the (stocks x 61) price matrices."""
    import numpy as np

    T = len(C_row)
    valid = np.where(~np.isnan(C_row))[0]
    n_obs = int(len(valid))
    if n_obs == 0:
        return {"price_status": "failed", "price_reason": "가격 없음", "path": [None] * T,
                "bench_path": [None] * T, "ret": {str(h): None for h in RETURN_HORIZONS}, "ret_note": {},
                "gaps": [], "n_obs": 0}
    if not np.isnan(C_row[0]):
        base_idx, base = 0, C_row[0]
    elif not np.isnan(carry_row[0]):
        base_idx, base = 0, carry_row[0]  # started during a halt: the carried close is the start (F7)
    else:
        base_idx = int(valid[0])
        base = C_row[base_idx]
    path = [None if (np.isnan(C_row[t]) or susp_row[t]) else float(100.0 * C_row[t] / base) for t in range(T)]
    bench_path = [None if t < base_idx else float(100.0 * bench_line[t] / bench_line[base_idx]) for t in range(T)]
    gaps = []
    for t in range(T):
        if susp_row[t]:
            gaps.append({"date": dates[t], "kind": "ca_suspect"})
        elif np.isnan(C_row[t]):
            gaps.append({"date": dates[t], "kind": "halt" if halt_row[t] else "missing"})
    ret, why, carried = returns_for(C_row, carry_row, susp_row)
    notes = {}
    for h in RETURN_HORIZONS:
        if why[h]:
            notes[str(h)] = RET_NOTE[why[h]]
        elif carried[h]:
            notes[str(h)] = START_HALTED_NOTE
    blanks = [str(h) for h in RETURN_HORIZONS if ret[h] is None]
    parts = []
    n_halt = sum(1 for g in gaps if g["kind"] == "halt")
    n_miss = sum(1 for g in gaps if g["kind"] == "missing")
    n_susp = sum(1 for g in gaps if g["kind"] == "ca_suspect")
    if n_halt or n_miss:
        parts.append(f"거래정지·누락 {n_halt + n_miss}일")
    if n_susp:
        parts.append(f"가격 급변 의심 {n_susp}일(선 끊음)")
    if blanks:
        parts.append(f"수익률 비움({','.join(blanks)})")
    return {"price_status": "partial" if parts else "ok", "price_reason": " · ".join(parts) or None,
            "path": path, "bench_path": bench_path,
            "ret": {str(h): ret[h] for h in RETURN_HORIZONS}, "ret_note": notes, "gaps": gaps, "n_obs": n_obs}


@dataclass
class PriceMatrix:
    """Price matrices of the comparison set, rows in ``keys`` order, columns the 61 sessions."""

    dates: list[date]
    C: Any  # valid close (NaN on halt / missing)
    carry: Any  # close carried through a halt row (NaN when the row has none)
    halt: Any  # bool
    source: str
    sha256: str
    notes: dict[str, Any] = field(default_factory=dict)


def price_block(pm: PriceMatrix, *, top_index: list[int], members_by_row, lo: float, hi: float) -> dict[str, Any]:
    """Price fields for the top rows. ``members_by_row[i]`` is row ``i``'s benchmark member array."""

    dates = [d.isoformat() for d in pm.dates]
    susp = suspect_matrix(pm.C[top_index], lo, hi)
    lines: dict[int, Any] = {}
    rows = []
    for k, i in enumerate(top_index):
        label = id(members_by_row[i])
        if label not in lines:
            lines[label] = benchmark_series(pm.C, members_by_row[i], lo, hi)[0]
        rows.append(price_row(pm.C[i], pm.carry[i], pm.halt[i], susp[k], lines[label], dates))
    return {"dates": dates, "rows": rows}


def kr_price_matrix(px, keys: list[tuple[str, str]], asof: date, *, source: str, sha256: str) -> PriceMatrix:
    """KR ``daily_ohlcv`` rows (``trade_date, ticker, market, open, high, low, close, volume``)."""
    import numpy as np
    import polars as pl

    px = px.filter(pl.col("trade_date") <= asof)
    sessions = sorted(px["trade_date"].unique().to_list())
    axis = sessions[-N_SESSIONS:]
    if len(axis) < N_SESSIONS or axis[-1] != asof:
        raise ExplainError("price_axis_not_ending_at_asof")
    index = pl.DataFrame({"ticker": [k[0] for k in keys], "market": [k[1] for k in keys],
                          "ri": list(range(len(keys)))})
    zero_ohlc = (pl.col("open") == 0) & (pl.col("high") == 0) & (pl.col("low") == 0)
    w = (px.filter(pl.col("trade_date").is_in(axis)).join(index, on=["ticker", "market"])
         .with_columns((zero_ohlc | (pl.col("volume") == 0)).alias("halt")))
    pos = {d: t for t, d in enumerate(axis)}
    ri = w["ri"].to_numpy()
    ti = np.array([pos[d] for d in w["trade_date"].to_list()], dtype=int)
    close = w["close"].to_numpy().astype(float)
    halt = w["halt"].to_numpy()
    C = np.full((len(keys), N_SESSIONS), np.nan)
    carry = np.full_like(C, np.nan)
    H = np.zeros(C.shape, dtype=bool)
    ok = (~halt) & (close > 0)
    C[ri[ok], ti[ok]] = close[ok]
    hh = halt & (close > 0)
    carry[ri[hh], ti[hh]] = close[hh]
    H[ri[halt], ti[halt]] = True
    return PriceMatrix(axis, C, carry, H, source, sha256)


def us_price_matrix(adj, symbols: list[str], asof: date, *, source: str, sha256: str) -> PriceMatrix:
    """US split-adjusted rows (``date, symbol, adj_close, adj_volume``) restricted to ``symbols``."""
    import numpy as np
    import polars as pl

    adj = adj.filter(pl.col("date") <= asof)
    per_day = adj.group_by("date").len().sort("date")
    if per_day.height == 0:
        raise ExplainError("price_rows_empty")
    median = float(per_day["len"].median())
    sessions = [d for d, n in per_day.rows() if n >= 0.5 * median]  # a thin day is not a session
    axis = sessions[-N_SESSIONS:]
    if len(axis) < N_SESSIONS or axis[-1] != asof:
        raise ExplainError("price_axis_not_ending_at_asof")
    index = {s: i for i, s in enumerate(symbols)}
    w = adj.filter(pl.col("date").is_in(axis) & pl.col("symbol").is_in(symbols))
    pos = {d: t for t, d in enumerate(axis)}
    ri = np.array([index[s] for s in w["symbol"].to_list()], dtype=int)
    ti = np.array([pos[d] for d in w["date"].to_list()], dtype=int)
    close = w["adj_close"].to_numpy().astype(float)
    volume = w["adj_volume"].to_numpy().astype(float)
    good = np.isfinite(close) & (close > 0)
    halt = good & (volume <= 0)
    ok = good & (volume > 0)
    C = np.full((len(symbols), N_SESSIONS), np.nan)
    carry = np.full_like(C, np.nan)
    H = np.zeros(C.shape, dtype=bool)
    C[ri[ok], ti[ok]] = close[ok]
    carry[ri[halt], ti[halt]] = close[halt]
    H[ri[halt], ti[halt]] = True
    return PriceMatrix(axis, C, carry, H, source, sha256)


# --------------------------------------------------------------------------- inputs of one model
@dataclass(frozen=True)
class JobPin:
    """What the 09:30 selection pinned for one model (a row of ``jobs.json``)."""

    market: str
    model_id: str
    prepared_input: Path
    input_sha256: str
    native_manifest: Path
    bundle_manifest: Path
    bundle_sha256: str


def read_job_pins(jobs_config: Path, prepared_root: Path) -> dict[str, JobPin]:
    body = json.loads(Path(jobs_config).read_text(encoding="utf-8"))
    out = {}
    for raw in body.get("jobs", []):
        def resolve(name: str) -> Path:
            path = Path(raw[name])
            return path if path.is_absolute() else Path(prepared_root) / path

        out[raw["model_id"]] = JobPin(
            market=raw["market"], model_id=raw["model_id"], prepared_input=resolve("prepared_input"),
            input_sha256=raw["input_sha256"], native_manifest=resolve("native_manifest"),
            bundle_manifest=Path(raw["bundle_path"]), bundle_sha256=raw["bundle_sha256"])
    return out


def _report_model(report: dict[str, Any], market: str, model_id: str) -> dict[str, Any]:
    for item in report.get("markets", []):
        if item.get("market") == market and item.get("model_id") == model_id:
            if not item.get("rankings"):
                raise ExplainError("report_has_no_rankings")
            return item
    raise ExplainError("model_not_in_report")


def _check_pins(pin: JobPin, report_model: dict[str, Any]) -> None:
    """The files read now are the files inference scored: job pin, report provenance and disk all agree."""
    prov = report_model.get("provenance", {})
    if prov.get("input_sha256") != pin.input_sha256:
        raise ExplainError("report_input_differs_from_pin")
    if prov.get("bundle_manifest_sha256") != pin.bundle_sha256:
        raise ExplainError("report_bundle_differs_from_pin")
    if sha256_file(pin.prepared_input) != pin.input_sha256:
        raise ExplainError("prepared_input_changed")
    if sha256_file(pin.bundle_manifest) != pin.bundle_sha256:
        raise ExplainError("bundle_changed")


def _dir_listing_sha(files: list[Path], root: Path) -> str:
    lines = [f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}" for p in sorted(files)]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def _scope_for_kr(label: Any) -> str:
    return KR_SCOPE.get(str(label), "같은 시장")


def _assemble(*, model_id: str, market: str, asof: str, inputs: dict[str, str], group_ids: list[str],
              top: list[dict[str, Any]], section: CrossSection | None, explain_error: str | None,
              prices: dict[str, Any] | None, price_error: str | None) -> dict[str, Any]:
    """Model block of the document. Explain and price fail separately; rows carry both."""
    import numpy as np

    rows = []
    for k, item in enumerate(top):
        row = {"rank": int(item["rank"]), "symbol": item["symbol"], "score": float(item["score"])}
        if section is not None:
            delta = {g: float(section.delta[k, GID.index(g)]) for g in group_ids}
            row.update(explain_status="ok", explain_reason=None, delta=delta, top_features=section.top_feats[k])
        else:
            row.update(explain_status="failed", explain_reason=explain_error, delta={}, top_features=[])
        if prices is not None:
            row.update(prices["rows"][k])
        else:
            row.update(price_status="failed", price_reason=price_error, path=[], bench_path=[],
                       ret={str(h): None for h in RETURN_HORIZONS}, ret_note={}, gaps=[], n_obs=0)
        rows.append(row)
    mean_delta = {}
    if section is not None:
        for g in group_ids:
            mean_delta[g] = float(np.mean(section.delta[:, GID.index(g)]))
    reasons = []
    if explain_error:
        reasons.append(f"explain:{explain_error}")
    if price_error:
        reasons.append(f"price:{price_error}")
    row_failed = [r for r in rows if r["explain_status"] == "failed"]
    if section is None:
        status = "failed"
    elif price_error or row_failed:
        status = "partial"
    else:
        status = "ok"
    return {"market": market, "status": status, "reason": "; ".join(reasons) or None, "asof": asof,
            "inputs": inputs, "groups": [{"id": g, "label": GLABEL[g]} for g in group_ids],
            "dates": prices["dates"] if prices else [], "top100_mean_delta": mean_delta, "rows": rows}


def failed_model(market: str, reason: str, asof: str | None = None) -> dict[str, Any]:
    return {"market": market, "status": "failed", "reason": reason, "asof": asof,
            "inputs": {"prepared_sha256": "", "bundle_sha256": "", "price_source": "", "price_sha256": "",
                       "benchmark": ""}, "groups": [], "dates": [], "top100_mean_delta": {}, "rows": []}


def _top_of_report(report_model: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    return [{"rank": int(r["rank"]), "symbol": r["symbol"], "market": r.get("market"), "score": float(r["score"])}
            for r in report_model["rankings"][:top_n]]


# --------------------------------------------------------------------------- KR
def explain_kr(pin: JobPin, report_model: dict[str, Any], *, kr_root: Path | None,
               price_dir: Path | None = None, top_n: int = TOP_N) -> dict[str, Any]:
    """KR model block. ``price_dir`` overrides the lake path (tests, copied inputs)."""
    import numpy as np
    import polars as pl

    from modeler.serving.kr_model import load_bundle
    from modeler.serving.kr_serving import build_design_matrix

    asof = str(report_model["feature_asof_date"])
    asof_date = date.fromisoformat(asof)
    top = _top_of_report(report_model, top_n)
    inputs = {"prepared_sha256": pin.input_sha256, "bundle_sha256": pin.bundle_sha256, "price_source": "",
              "price_sha256": "", "benchmark": "same-market equal-weight mean of adjacent-session simple "
                                               "returns (price only), chained, start=100"}
    keys_all = [(r["symbol"], r["market"]) for r in report_model["rankings"]]
    report_scores = [float(r["score"]) for r in report_model["rankings"]]

    section, explain_error = None, None
    try:
        _check_pins(pin, report_model)
        native = json.loads(pin.native_manifest.read_text(encoding="utf-8"))
        if native.get("feature_asof_date") != asof:
            raise ExplainError("native_asof_differs_from_report")
        bundle = load_bundle(pin.bundle_manifest.parent, verify_golden=False)
        if bundle.manifest.get("model_id") != KR_MODEL_ID:
            raise ExplainError("bundle_identity")
        panel = pl.read_parquet(pin.prepared_input)
        excluded = [c for c in ("flow_short_balance_qty", "flow_short_balance_chg_20d") if c in panel.columns]
        selected, X = build_design_matrix(bundle, panel.drop(excluded))
        del panel
        design, features = list(bundle.design_columns), list(bundle.feature_columns)
        tickers = selected["ticker"].cast(pl.String).to_list()
        markets = selected["market"].cast(pl.String).to_list()
        section = explain_cross_section(
            predict=lambda M: bundle.model.predict_proba(M)[:, 1], X=X, design=design, features=features,
            feature_group=KR_FEATURE_GROUP, keys=list(zip(tickers, markets, strict=True)), symbols=tickers,
            comparison=np.array(markets), report_keys=keys_all, report_scores=report_scores,
            names=KR_NAMES, scope_for=_scope_for_kr, flag_features=KR_FLAG_FEATURES, top_n=top_n)
        group_ids = [g for g, cols in group_columns(design, features, KR_FEATURE_GROUP).items() if len(cols)]
    except Exception as exc:  # explain fails alone; the price part below still runs
        section, explain_error = None, _reason(exc)
        group_ids = [g for g in GID if g in set(KR_FEATURE_GROUP.values())]

    prices, price_error = None, None
    try:
        native = json.loads(pin.native_manifest.read_text(encoding="utf-8"))
        if price_dir is None and kr_root is None:
            raise ExplainError("kr_root_not_configured")
        directory = Path(price_dir) if price_dir else _kr_price_dir(Path(kr_root), native)
        files = sorted(directory.rglob("*.parquet"))
        if not files:
            raise ExplainError("price_files_missing")
        manifest = directory.parent / "_manifests" / "table_manifests" / "daily_ohlcv.json"
        digest = sha256_file(manifest) if manifest.is_file() else _dir_listing_sha(files, directory)
        inputs["price_source"] = (f"kr daily_ohlcv snapshot_date={native.get('snapshot_date')} "
                                  f"source={native.get('source')} ({'table manifest' if manifest.is_file() else 'file listing'})")
        inputs["price_sha256"] = digest
        start = asof_date - timedelta(days=200)
        px = (pl.scan_parquet(str(directory) + "/**/*.parquet")
              .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= asof_date))
              .select("trade_date", "ticker", "market", "open", "high", "low", "close", "volume").collect())
        pm = kr_price_matrix(px, keys_all, asof_date, source=inputs["price_source"], sha256=digest)
        del px
        pos = {k: i for i, k in enumerate(keys_all)}
        top_index = [pos[(r["symbol"], r["market"])] for r in top]
        mk = np.array([k[1] for k in keys_all])
        members = {m: np.where(mk == m)[0] for m in np.unique(mk)}
        by_row = [members[m] for m in mk]
        _check_kr_simple_return(pm, report_model, top_index)
        prices = price_block(pm, top_index=top_index, members_by_row=by_row, lo=KR_JUMP[0], hi=KR_JUMP[1])
    except Exception as exc:
        prices, price_error = None, _reason(exc)
    return _assemble(model_id=pin.model_id, market="KR", asof=asof, inputs=inputs, group_ids=group_ids,
                     top=top, section=section if explain_error is None else None, explain_error=explain_error,
                     prices=prices, price_error=price_error)


def _kr_price_dir(kr_root: Path, native: dict[str, Any]) -> Path:
    snapshot, source = native.get("snapshot_date"), native.get("source")
    if not snapshot or not source:
        raise ExplainError("native_manifest_without_snapshot")
    return Path(kr_root) / "raw" / "raw_postgres" / f"snapshot_date={snapshot}" / f"source={source}" / "daily_ohlcv"


def _check_kr_simple_return(pm: PriceMatrix, report_model: dict[str, Any], top_index: list[int]) -> None:
    """K's one-day return from this price file must match the one the prepared panel used."""
    import numpy as np

    rows = report_model["rankings"]
    bad = checked = 0
    for i in top_index:
        ref = rows[i].get("K_simple_return")
        a, b = pm.C[i, -1], pm.C[i, -2]
        if ref is None or np.isnan(a) or np.isnan(b):
            continue
        checked += 1
        bad += abs(a / b - 1.0 - float(ref)) > 1e-9
    if checked and bad > 0.02 * checked:
        raise ExplainError("price_system_mismatch")


# --------------------------------------------------------------------------- US
class _SliceLake:
    """The two tables ``adjusted_daily`` reads, already narrowed to the window."""

    security_boundaries = False

    def __init__(self, prices, corp_actions) -> None:
        self._tables = {"prices_daily": prices, "corp_actions": corp_actions}

    def scan(self, table: str):
        return self._tables[table].lazy()


@dataclass
class UsShared:
    """Inputs the two US models share, loaded once."""

    transformed: Any = None
    matrix: Any = None
    symbols: list[str] = field(default_factory=list)
    input_error: str | None = None
    prices: PriceMatrix | None = None
    price_error: str | None = None
    price_inputs: dict[str, str] = field(default_factory=dict)


def load_us_shared(pin: JobPin, asof: date, *, us_root: Path | None,
                   price_files: tuple[Path, Path] | None = None) -> UsShared:
    """Feature matrix and split-adjusted prices of the US eligible set. ``price_files`` overrides the lake."""
    import numpy as np
    import polars as pl

    from modeler.serving import us_daily
    from modeler.us.prices import adjusted_daily

    shared = UsShared()
    try:
        shared.transformed, shared.matrix = us_daily.transform_daily_features(pl.read_parquet(pin.prepared_input))
        shared.symbols = shared.transformed["symbol"].to_list()
        if np.isnan(shared.matrix).any():
            raise ExplainError("design_matrix_has_nan")
    except Exception as exc:
        shared.input_error = _reason(exc)
        return shared
    try:
        native = json.loads(pin.native_manifest.read_text(encoding="utf-8"))
        if price_files is None:
            files = native["source_snapshot_files"]
            resolved = []
            if us_root is None:
                raise ExplainError("us_root_not_configured")
            for table in ("prices_daily", "corp_actions"):
                entries = files[table]
                if len(entries) != 1:
                    raise ExplainError(f"{table}_snapshot_not_single_file")
                path = Path(us_root) / entries[0]["path"]
                if sha256_file(path) != entries[0]["sha256"]:
                    raise ExplainError(f"{table}_changed_since_prepare")
                resolved.append((path, entries[0]["sha256"], entries[0]["path"]))
            prices_path, ca_path = resolved[0][0], resolved[1][0]
            sha_a, sha_b = resolved[0][1], resolved[1][1]
            rel = (resolved[0][2], resolved[1][2])
        else:
            prices_path, ca_path = price_files
            sha_a, sha_b = sha256_file(prices_path), sha256_file(ca_path)
            rel = (prices_path.name, ca_path.name)
        start = asof - timedelta(days=190)
        px = (pl.scan_parquet(prices_path).filter((pl.col("date") >= start) & (pl.col("date") <= asof)).collect())
        ca = (pl.scan_parquet(ca_path).filter((pl.col("kind") == "split") & (pl.col("ex_date") >= start)).collect())
        adj = (adjusted_daily(_SliceLake(px, ca), base_date=asof, security_boundaries=False)
               .select("date", "symbol", "close", "adj_close", "adj_volume").collect())
        del px, ca
        source = f"us prices_daily={rel[0]} + corp_actions={rel[1]} (split adjusted, price only)"
        combined = hashlib.sha256(f"prices_daily={sha_a}\ncorp_actions={sha_b}\n".encode()).hexdigest()
        shared.prices = us_price_matrix(adj, shared.symbols, asof, source=source, sha256=combined)
        shared.price_inputs = {"price_source": source, "price_sha256": combined}
        _check_us_adj_close(shared, adj)
        del adj
    except Exception as exc:
        shared.prices, shared.price_error = None, _reason(exc)
    return shared


def _check_us_adj_close(shared: UsShared, adj) -> None:
    """The as-of adjusted close must equal the one the prepared features carry (same split table)."""
    import numpy as np
    import polars as pl

    if "adj_close" not in shared.transformed.columns:
        return
    ref = shared.transformed.select("symbol", pl.col("adj_close").alias("_ref"))
    last = shared.prices.dates[-1]
    mine = adj.filter(pl.col("date") == last).select("symbol", "adj_close")
    joined = ref.join(mine, on="symbol").drop_nulls()
    if joined.height == 0:
        return
    rel = np.abs(joined["adj_close"].to_numpy() / joined["_ref"].to_numpy() - 1.0)
    if (rel > 1e-9).sum() > 0.01 * joined.height:
        raise ExplainError("price_system_mismatch")


def _load_us_estimator(pin: JobPin):
    """Model of one US bundle, hash-checked against the pinned manifest (inference ran the golden test)."""
    import joblib

    from modeler.serving import us_daily

    manifest = json.loads(pin.bundle_manifest.read_text(encoding="utf-8"))
    if manifest.get("model_id") != us_daily.MODEL_ID or manifest.get("variant") not in {"lightgbm", "ridge"}:
        raise ExplainError("bundle_identity")
    model_path = pin.bundle_manifest.parent / "model.joblib"
    if sha256_file(model_path) != manifest.get("model_sha256"):
        raise ExplainError("model_changed")
    if tuple(manifest.get("features", ())) != us_daily.FEATURES or \
            tuple(manifest.get("design_columns", ())) != us_daily.DESIGN_COLUMNS:
        raise ExplainError("bundle_feature_contract")
    bundle = joblib.load(model_path)
    if tuple(bundle.get("features", ())) != us_daily.FEATURES:
        raise ExplainError("bundle_feature_contract")
    return bundle["model"]


def explain_us(pin: JobPin, report_model: dict[str, Any], shared: UsShared, *, top_n: int = TOP_N) -> dict[str, Any]:
    import numpy as np

    from modeler.serving import us_daily

    asof = str(report_model["feature_asof_date"])
    top = _top_of_report(report_model, top_n)
    inputs = {"prepared_sha256": pin.input_sha256, "bundle_sha256": pin.bundle_sha256,
              "price_source": shared.price_inputs.get("price_source", ""),
              "price_sha256": shared.price_inputs.get("price_sha256", ""),
              "benchmark": "US eligible set equal-weight mean of adjacent-session split-adjusted simple "
                           "returns (price only, no dividends), chained, start=100"}
    design, features = list(us_daily.DESIGN_COLUMNS), list(us_daily.FEATURES)
    section, explain_error = None, None
    try:
        _check_pins(pin, report_model)
        if shared.input_error:
            raise ExplainError(shared.input_error)
        native = json.loads(pin.native_manifest.read_text(encoding="utf-8"))
        if native.get("feature_asof_date") != asof:
            raise ExplainError("native_asof_differs_from_report")
        estimator = _load_us_estimator(pin)
        keys = list(shared.symbols)
        section = explain_cross_section(
            predict=estimator.predict, X=shared.matrix, design=design, features=features,
            feature_group=US_FEATURE_GROUP, keys=keys, symbols=keys, comparison=np.zeros(len(keys), dtype=int),
            report_keys=[r["symbol"] for r in report_model["rankings"]],
            report_scores=[float(r["score"]) for r in report_model["rankings"]], names=US_NAMES,
            scope_for=lambda _label: US_SCOPE, flag_features=frozenset(), top_n=top_n)
        del estimator
    except Exception as exc:
        section, explain_error = None, _reason(exc)
    group_ids = [g for g in GID if g in set(US_FEATURE_GROUP.values())]

    prices, price_error = None, None
    try:
        if shared.prices is None:
            raise ExplainError(shared.price_error or shared.input_error or "price_unavailable")
        pos = {s: i for i, s in enumerate(shared.symbols)}
        top_index = [pos[r["symbol"]] for r in top]
        everyone = np.arange(len(shared.symbols))
        prices = price_block(shared.prices, top_index=top_index, members_by_row=[everyone] * len(shared.symbols),
                             lo=US_JUMP[0], hi=US_JUMP[1])
    except Exception as exc:
        prices, price_error = None, _reason(exc)
    return _assemble(model_id=pin.model_id, market="US", asof=asof, inputs=inputs, group_ids=group_ids, top=top,
                     section=section, explain_error=explain_error, prices=prices, price_error=price_error)


def _reason(exc: BaseException) -> str:
    """Short, path-free cause: ``ExplainError`` carries its own machine reason, others the class name."""
    if isinstance(exc, ExplainError):
        return str(exc.args[0])
    return type(exc).__name__


# --------------------------------------------------------------------------- document
def rollup_status(models: dict[str, dict[str, Any]]) -> str:
    states = [m["status"] for m in models.values()]
    if states and all(s == "ok" for s in states):
        return "ok"
    if any(s in {"ok", "partial"} for s in states):
        return "partial"
    return "failed"


def build_document(*, unit: str, invocation_id: str, report_sha256: str, release: str,
                   models: dict[str, dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    return {"schema": SCHEMA, "method_version": METHOD_VERSION, "group_map_version": GROUP_MAP_VERSION,
            "unit": unit, "invocation_id": invocation_id, "report_sha256": report_sha256, "release": release,
            "status": rollup_status(models), "reason": reason, "models": models}


def failure_document(*, unit: str, invocation_id: str, report_sha256: str, release: str, reason: str,
                     models: tuple[str, ...] = tuple(MODEL_MARKET)) -> dict[str, Any]:
    """Stdlib-only document for a stage that failed before or while computing (timeout, crash)."""
    body = {m: failed_model(MODEL_MARKET[m], reason) for m in models}
    doc = build_document(unit=unit, invocation_id=invocation_id, report_sha256=report_sha256,
                         release=release, models=body, reason=reason)
    doc["status"] = "failed"
    return doc


def explain_report(*, unit: str, invocation_id: str, report_bytes: bytes, release: str,
                   pins: dict[str, JobPin], kr_root: Path | None, us_root: Path | None,
                   top_n: int = TOP_N, kr_price_dir: Path | None = None,
                   us_price_files: tuple[Path, Path] | None = None) -> dict[str, Any]:
    """Whole document for one day. One model never stops another.

    ``kr_price_dir`` / ``us_price_files`` replace the lake paths (copied inputs, tests).
    """
    report = json.loads(report_bytes)
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    models: dict[str, dict[str, Any]] = {}
    shared_us: dict[str, UsShared] = {}
    for model_id, market in MODEL_MARKET.items():
        try:
            pin = pins.get(model_id)
            if pin is None:
                raise ExplainError("model_not_pinned")
            report_model = _report_model(report, market, model_id)
            if market == "KR":
                models[model_id] = explain_kr(pin, report_model, kr_root=kr_root, price_dir=kr_price_dir, top_n=top_n)
            else:
                key = pin.input_sha256
                if key not in shared_us:
                    shared_us.clear()  # one US input at a time keeps memory down
                    gc.collect()
                    shared_us[key] = load_us_shared(pin, date.fromisoformat(str(report_model["feature_asof_date"])),
                                                    us_root=us_root, price_files=us_price_files)
                models[model_id] = explain_us(pin, report_model, shared_us[key], top_n=top_n)
        except Exception as exc:
            models[model_id] = failed_model(market, _reason(exc))
        gc.collect()
    return _clean(build_document(unit=unit, invocation_id=invocation_id, report_sha256=report_sha,
                                 release=release, models=models))


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
    document = explain_report(
        unit=args.report_date, invocation_id=args.invocation_id, report_bytes=report_bytes,
        release=args.release, pins=pins, kr_root=Path(args.kr_root) if args.kr_root else None,
        us_root=Path(args.us_root) if args.us_root else None, top_n=args.top_n)
    atomic_write_json(Path(args.output), document)
    print(json.dumps({"status": document["status"],
                      "models": {k: v["status"] for k, v in document["models"].items()}}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-explain")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute explain-D.json into --output")
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
    run.add_argument("--top-n", type=int, default=TOP_N)
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
