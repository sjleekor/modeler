"""시장·섹터 60세션 라벨 (README §4, 사양 01 §5.1).

모두 **총수익 경로 ``V``** 위에서, 진입 종가(t+1)를 1로 놓고 계산한다::

    excess_return_60d_vs_cash   = V(exit)/V(entry) - 1 - cash_proxy_return(entry, exit)
    excess_return_60d_vs_market = 자산 60d 총수익 - 부모 벤치마크 60d 총수익 (같은 진입/만기)
    entry_loss_60d              = min_{entry<=u<=exit} V(u)/V(entry) - 1      (<= 0, 진입일 포함)
    loss_event_60d_8pct         = 1[entry_loss_60d <= -0.08]
    future_max_drawdown_60d     = 구간 [entry, exit]의 고점 대비 최대 낙폭 (진입가 손실과 다르다)

"+20% 뒤 -10%": ``entry_loss_60d`` = 0(진입가 아래로 간 적이 없다), ``future_max_drawdown_60d``
= -10%. 두 라벨을 섞지 않는다.

**누락 규칙.** 진입 또는 만기 종가가 없으면 라벨 전부 null + ``label_missing_reason``.
가까운 날짜로 옮기지 않는다. 만기 세션이 스냅샷 마지막 관측 세션 뒤이면
``label_matured=False``, 사유 ``not_matured``. 현금·시장 대비 라벨은 각자 사유
(``cash_label_missing_reason``, ``market_label_missing_reason``)를 따로 낸다.
구간 안쪽에 가격이 빠진 세션이 있으면 관측된 점만으로 계산하되 ``window_missing_sessions``에 센다.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from modeler.scores.common.calendar import HORIZON_SESSIONS, UTC_TS, SessionCalendar
from modeler.scores.common.cash import CODE_NO_RATE, CODE_OK, CODE_STALE, CashAccount

LOSS_THRESHOLD = -0.08
_EPS = 1e-12

R_NOT_MATURED = "not_matured"
R_ENTRY_MISSING = "entry_close_missing"
R_EXIT_MISSING = "exit_close_missing"
R_BEYOND_CALENDAR = "exit_beyond_calendar"
R_CASH_SERIES_MISSING = "cash_series_missing"
R_CASH_STALE = "cash_rate_stale"
R_CASH_NO_RATE = "cash_rate_not_available_yet"
R_NO_PARENT = "no_parent_benchmark"
R_PARENT_MISSING = "parent_close_missing"


def align_to_calendar(
    cal: SessionCalendar, path: pl.DataFrame, col: str = "tr_index"
) -> np.ndarray:
    """경로의 ``col``을 달력 인덱스 순서 배열로. 관측 없는 세션은 NaN."""
    out = np.full(len(cal), np.nan)
    idx = (
        cal.session_table()
        .select("idx", "session")
        .join(path.select("session", col), on="session", how="inner")
    )
    out[idx["idx"].to_numpy()] = idx[col].to_numpy()
    return out


def _series(a: np.ndarray, dtype: pl.DataType = pl.Float64) -> pl.Series:
    return pl.Series(a, dtype=dtype, nan_to_null=True)


def compute_labels(
    asset_id: str,
    cal: SessionCalendar,
    path: pl.DataFrame,
    *,
    cash: CashAccount | None,
    cash_missing_reason: str = R_CASH_SERIES_MISSING,
    parent_path: pl.DataFrame | None = None,
    has_parent: bool = False,
    horizon: int = HORIZON_SESSIONS,
    loss_threshold: float = LOSS_THRESHOLD,
) -> pl.DataFrame:
    """결정 세션 t(첫 관측 ~ 마지막 관측 세션)마다 한 행. ``build_asset_panel``의 행과 같다."""
    n = len(cal)
    v = align_to_calendar(cal, path)
    observed = ~np.isnan(v)
    if not observed.any():
        raise ValueError(f"{asset_id}: 관측된 경로가 없습니다")
    first_obs = int(np.argmax(observed))
    last_obs = int(n - 1 - np.argmax(observed[::-1]))

    t_idx = np.arange(first_obs, min(last_obs, n - 2) + 1)  # t+1이 달력에 있어야 한다
    e_idx = t_idx + 1
    x_idx = e_idx + horizon
    in_cal = x_idx < n
    x_safe = np.where(in_cal, x_idx, n - 1)

    ve = v[e_idx]
    vx = np.where(in_cal, v[x_safe], np.nan)
    entry_ok = ~np.isnan(ve)
    exit_ok = ~np.isnan(vx)

    reason = np.full(len(t_idx), None, dtype=object)
    reason[~in_cal] = R_BEYOND_CALENDAR
    m = (reason == None) & (e_idx <= last_obs) & ~entry_ok  # noqa: E711
    reason[m] = R_ENTRY_MISSING
    m = (reason == None) & (e_idx > last_obs)  # noqa: E711
    reason[m] = R_NOT_MATURED
    m = (reason == None) & (x_idx > last_obs)  # noqa: E711
    reason[m] = R_NOT_MATURED
    m = (reason == None) & ~exit_ok  # noqa: E711
    reason[m] = R_EXIT_MISSING
    ok = reason == None  # noqa: E711

    total = np.full(len(t_idx), np.nan)
    loss = np.full(len(t_idx), np.nan)
    mdd = np.full(len(t_idx), np.nan)
    win_missing = np.full(len(t_idx), np.nan)
    if ok.any():
        vpad = np.concatenate([v, np.full(horizon + 2, np.nan)])
        windows = sliding_window_view(vpad, horizon + 1)[e_idx[ok]]
        rel = windows / ve[ok][:, None]
        total[ok] = vx[ok] / ve[ok] - 1.0
        loss[ok] = np.nanmin(rel, axis=1) - 1.0
        runmax = np.fmax.accumulate(rel, axis=1)
        mdd[ok] = np.nanmin(rel / runmax, axis=1) - 1.0
        win_missing[ok] = np.isnan(windows).sum(axis=1)

    # 현금 대비
    cash_ret = np.full(len(t_idx), np.nan)
    cash_reason = reason.copy()
    if cash is None:
        cash_reason[ok] = cash_missing_reason
    else:
        if cash.frame.height != n:
            raise ValueError("현금 계정 세션 수가 자산 달력과 다릅니다")
        cr, code = cash.interval_return(e_idx, x_safe)
        cr = np.where(ok, cr, np.nan)
        cash_ret = cr
        cash_reason[ok & (code == CODE_STALE)] = R_CASH_STALE
        cash_reason[ok & (code == CODE_NO_RATE)] = R_CASH_NO_RATE
        assert CODE_OK == 0
    excess_cash = np.where(np.isnan(cash_ret), np.nan, total - cash_ret)

    # 시장(부모) 대비
    parent_ret = np.full(len(t_idx), np.nan)
    market_reason = reason.copy()
    if not has_parent:
        market_reason[ok] = R_NO_PARENT
    else:
        assert parent_path is not None
        pv = align_to_calendar(cal, parent_path)
        pe, px = pv[e_idx], np.where(in_cal, pv[x_safe], np.nan)
        pok = ok & ~np.isnan(pe) & ~np.isnan(px)
        parent_ret[pok] = px[pok] / pe[pok] - 1.0
        market_reason[ok & ~pok] = R_PARENT_MISSING
    excess_mkt = np.where(np.isnan(parent_ret), np.nan, total - parent_ret)

    sess_tab = cal.session_table()
    close_at = sess_tab["close_at"]
    label_end = close_at.gather(x_safe.tolist()).to_list()
    label_end = [d if c else None for d, c in zip(label_end, in_cal, strict=True)]
    loss_event = np.where(np.isnan(loss), np.nan, (loss <= loss_threshold + _EPS).astype(float))

    return pl.DataFrame(
        {
            "asset_id": [asset_id] * len(t_idx),
            "session": [cal.sessions[i] for i in t_idx],
            "entry_session": [cal.sessions[i] for i in e_idx],
            "exit_session": [
                cal.sessions[i] if c else None for i, c in zip(x_safe, in_cal, strict=True)
            ],
            "label_end_at": pl.Series(label_end, dtype=UTC_TS),
            "label_matured": pl.Series(ok, dtype=pl.Boolean),
            "total_return_60d": _series(total),
            "cash_proxy_return_60d": _series(cash_ret),
            "parent_return_60d": _series(parent_ret),
            "excess_return_60d_vs_cash": _series(excess_cash),
            "excess_return_60d_vs_market": _series(excess_mkt),
            "entry_loss_60d": _series(loss),
            "loss_event_60d_8pct": pl.Series(loss_event, nan_to_null=True).cast(pl.Boolean),
            "future_max_drawdown_60d": _series(mdd),
            "window_missing_sessions": _series(win_missing).cast(pl.Int32),
            "label_missing_reason": pl.Series(reason.tolist(), dtype=pl.String),
            "cash_label_missing_reason": pl.Series(cash_reason.tolist(), dtype=pl.String),
            "market_label_missing_reason": pl.Series(market_reason.tolist(), dtype=pl.String),
        }
    )
