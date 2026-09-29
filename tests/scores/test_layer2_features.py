from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from modeler.scores.common.calendar import UTC_TS
from modeler.scores.market_sector.build_panel import KrNotSyncedError
from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.features import (
    CONTINUOUS_FEATURES,
    PitSeries,
    build_features,
    fred_available_at,
    kr_foreign_net_20_over_trdval,
    model_feature_columns,
    usdkrw_ret_60,
)

from ._helpers import make_layer2_frame


def _rows(pairs):
    return fred_available_at(
        pl.DataFrame(pairs, schema=["date", "realtime_start", "value"], orient="row")
    )


def test_fred_available_at_is_17et_and_tz_aware():
    r = _rows(
        [(date(2024, 7, 1), date(2024, 7, 1), 1.0), (date(2024, 1, 2), date(2024, 1, 3), 2.0)]
    )
    a = r["available_at"].to_list()
    assert a[0] == datetime(2024, 7, 1, 21, 0, tzinfo=UTC)  # EDT 17:00 = 21:00Z
    assert a[1] == datetime(2024, 1, 3, 22, 0, tzinfo=UTC)  # EST, realtime_start가 더 늦다


def test_pit_series_uses_only_known_rows_and_latest_vintage():
    rows = _rows(
        [
            (date(2024, 1, 2), date(2024, 1, 2), 10.0),
            (date(2024, 1, 3), date(2024, 1, 3), 11.0),
            (date(2024, 1, 3), date(2024, 1, 10), 99.0),  # 나중 개정
            (date(2024, 1, 4), date(2024, 1, 8), 12.0),  # 늦게 공표
        ]
    )
    ps = PitSeries(rows)
    d = lambda day, h: datetime(2024, 1, day, h, tzinfo=UTC)  # noqa: E731
    res = ps.lookup([d(4, 15), d(9, 15), d(11, 15)], lags=(0, 1))
    # 1/4 15:00Z: 1/3 관측(11.0)까지만 알려져 있다
    assert res[0]["value"][0] == 11.0 and res[1]["value"][0] == 10.0
    # 1/9: 1/4 관측(12.0)이 공표됨, 1/3은 아직 개정 전 vintage
    assert res[0]["value"][1] == 12.0 and res[1]["value"][1] == 11.0
    # 1/11: 1/3의 개정본(99.0)이 최신 vintage
    assert res[1]["value"][2] == 99.0
    for j in range(3):
        for k in (0, 1):
            av = res[k]["avail"][j]
            assert av is None or av <= [d(4, 15), d(9, 15), d(11, 15)][j]


def test_kr_features_generic_functions():
    ats_at = lambda i: datetime(2024, 1, 1, 8, 30, tzinfo=UTC) + timedelta(days=i)  # noqa: E731
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(70)]
    av = pl.Series([ats_at(i) for i in range(70)], dtype=UTC_TS)
    fx = pl.DataFrame({"date": days, "value": [1000.0 + i for i in range(70)], "available_at": av})
    out = usdkrw_ret_60(fx, [ats_at(69) + timedelta(hours=1)])
    assert out["usdkrw_ret_60"][0] == pytest.approx(__import__("math").log(1069 / 1009))
    flow = fx.with_columns(pl.lit(2.0).alias("value"))
    trd = fx.with_columns(pl.lit(10.0).alias("value"))
    out = kr_foreign_net_20_over_trdval(flow, trd, [ats_at(30) + timedelta(hours=1)])
    assert out["kr_foreign_net_20_over_trdval"][0] == pytest.approx(0.2)
    early = kr_foreign_net_20_over_trdval(flow, trd, [ats_at(5)])
    assert early["kr_foreign_net_20_over_trdval"][0] is None


def test_kr_market_fails_loudly():
    frame, macro, panel = make_layer2_frame(n_sessions=300)
    with pytest.raises(KrNotSyncedError):
        build_features(panel, macro, MsConfig(), market="KR")


def test_feature_frame_pit_and_no_lookahead():
    frame, macro, panel = make_layer2_frame(n_sessions=600)
    assert (frame["feature_available_at"] <= frame["decision_at"]).all()
    assert (frame["decision_at"] < frame["entry_at"]).all()
    cols = model_feature_columns(sorted(frame["asset_id"].unique().to_list()))
    assert len(CONTINUOUS_FEATURES) == 18
    assert all(c in frame.columns for c in cols)
    # 미래 가격을 바꿔도 과거 행의 피쳐는 그대로다
    cfg = MsConfig()
    from modeler.scores.market_sector.features import compute_price_features

    base = compute_price_features(panel, cfg)
    cut = 400
    sess = sorted(panel["session"].unique().to_list())[cut]
    mod = panel.with_columns(
        pl.when(pl.col("session") > sess)
        .then(pl.col("tr_index_t") * 3.7)
        .otherwise(pl.col("tr_index_t"))
        .alias("tr_index_t")
    )
    other = compute_price_features(mod, cfg)
    a = base.filter(pl.col("session") <= sess).drop("feature_ready")
    b = other.filter(pl.col("session") <= sess).drop("feature_ready")
    assert a.equals(b)


def test_warmup_rows_not_ready():
    frame, _, _ = make_layer2_frame(n_sessions=400)
    n_ready = frame.filter(pl.col("feature_ready")).height
    assert n_ready == frame.height - 3 * 252
