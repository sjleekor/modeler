"""합성 경로·달력 도우미."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from modeler.scores.common.calendar import SessionCalendar


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_cal(n: int = 30, start: date = date(2024, 1, 2), cid: str = "XNYS") -> SessionCalendar:
    return SessionCalendar.from_sessions(cid, weekdays(start, n), calendar_basis="synthetic")


def make_path(
    cal: SessionCalendar, tr: dict[int, float] | list[float], basis: str = "total_return"
) -> pl.DataFrame:
    """``tr``: 달력 인덱스 -> tr_index (dict) 또는 0부터 연속 리스트."""
    items = tr.items() if isinstance(tr, dict) else enumerate(tr)
    rows = [(cal.sessions[i], v) for i, v in items]
    return pl.DataFrame(
        {
            "session": [r[0] for r in rows],
            "px_raw": [float(r[1]) * 100 for r in rows],
            "px_adj": [float(r[1]) * 100 for r in rows],
            "tr_index": [float(r[1]) for r in rows],
            "return_basis": [basis] * len(rows),
        }
    )


# --------------------------------------------------------------------------- layer 2
def make_layer2_frame(n_sessions: int = 1100, seed: int = 0, assets=("us_spx", "us_ndx", "us_fin")):
    """합성 경로에서 레이어 1(패널·라벨)과 레이어 2 피쳐를 실제 함수로 만든 조인 프레임."""
    import numpy as np

    from modeler.scores.common.assets import get_asset
    from modeler.scores.market_sector.build_panel import assemble
    from modeler.scores.market_sector.config import MsConfig
    from modeler.scores.market_sector.features import (
        MACRO_SERIES_IDS,
        build_features,
        fred_available_at,
    )

    rng = np.random.default_rng(seed)
    cal = make_cal(n_sessions, start=date(2015, 1, 1))
    paths = {}
    for a in assets:
        r = rng.normal(0.0003, 0.02, n_sessions)
        paths[a] = make_path(cal, list(100 * np.exp(np.cumsum(r))))
    panel, labels = assemble([get_asset(a) for a in assets], paths, cal, None)
    rows = []
    for sid in MACRO_SERIES_IDS:
        base = 20.0 if sid == "VIXCLS" else 3.0
        v = base + np.cumsum(rng.normal(0, 0.1, n_sessions))
        v = np.abs(v) + 1.0
        for d, x in zip(cal.sessions, v, strict=True):
            rows.append((sid, d, d, float(x)))
    macro = fred_available_at(
        pl.DataFrame(rows, schema=["series_id", "date", "realtime_start", "value"], orient="row")
    )
    cfg = MsConfig()
    feats = build_features(panel, macro, cfg, market="US")
    lab = labels.select(
        "asset_id",
        "session",
        "label_end_at",
        "label_matured",
        "total_return_60d",
        "excess_return_60d_vs_cash",
        "excess_return_60d_vs_market",
        "loss_event_60d_8pct",
    )
    return feats.join(lab, on=["asset_id", "session"], how="left"), macro, panel


def small_cfg(**kw):
    from dataclasses import replace

    from modeler.scores.market_sector.config import MsConfig

    base = dict(
        first_test_year={"US": 2017, "KR": 2017},
        train_start={"US": date(2015, 1, 1), "KR": date(2015, 1, 1)},
        lgbm_n_estimators=15,
        min_train_rows=200,
        min_train_events_per_asset=5,
        bootstrap_resamples=40,
        opportunity_reference_min_oof=50,
        block_sensitivity=(20, 40),
        block_sessions=30,
    )
    base.update(kw)
    return replace(MsConfig(), **base)
