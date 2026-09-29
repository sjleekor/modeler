import hashlib
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from modeler.scores.common import panel as panel_mod
from modeler.scores.common.assets import get_asset
from modeler.scores.common.cash import build_cash_account
from modeler.scores.common.panel import PitViolationError, assert_pit, build_asset_panel
from modeler.scores.market_sector.build_panel import assemble, write_outputs
from modeler.scores.market_sector.labels import compute_labels
from tests.scores._helpers import make_cal, make_path


def _row(lab: pl.DataFrame, cal, i: int) -> dict:
    return lab.filter(pl.col("session") == cal.sessions[i]).row(0, named=True)


def test_entry_loss_differs_from_future_max_drawdown():
    cal = make_cal(12)
    # 진입 = 세션1(V=1) -> +20% -> -10%(고점 대비)
    path = make_path(cal, {0: 1.0, 1: 1.0, 2: 1.2, 3: 1.08, 4: 1.08, 5: 1.08})
    lab = compute_labels("a", cal, path, cash=None, horizon=3)
    r = _row(lab, cal, 0)
    assert r["label_matured"] is True
    assert r["entry_loss_60d"] == pytest.approx(0.0)  # 진입가 아래로 간 적 없다
    assert r["entry_loss_60d"] > -0.08 and r["loss_event_60d_8pct"] is False
    assert r["future_max_drawdown_60d"] == pytest.approx(1.08 / 1.2 - 1)  # -10%
    assert r["future_max_drawdown_60d"] <= -0.083
    assert r["total_return_60d"] == pytest.approx(0.08)


def test_loss_event_threshold_and_window_bounds():
    cal = make_cal(12)
    path = make_path(cal, {0: 1, 1: 1, 2: 0.91, 3: 0.95, 4: 1.0, 5: 0.5})  # 0.5는 창 밖
    lab = compute_labels("a", cal, path, cash=None, horizon=3)
    r = _row(lab, cal, 0)
    assert r["entry_loss_60d"] == pytest.approx(-0.09) and r["loss_event_60d_8pct"] is True
    edge = make_path(cal, {0: 1, 1: 1, 2: 0.92, 3: 1.0, 4: 1.0})
    assert (
        _row(compute_labels("a", cal, edge, cash=None, horizon=3), cal, 0)["loss_event_60d_8pct"]
        is True
    )  # -8% 정확히 경계는 사건


def test_missing_exit_close_gives_null_labels_and_no_shift():
    cal = make_cal(12)
    # 만기(진입=1, 만기=1+3=4) 종가 없음. 세션 5는 있다 -> 5로 옮기면 안 된다
    path = make_path(cal, {0: 1, 1: 1, 2: 1.1, 3: 1.2, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0})
    lab = compute_labels("a", cal, path, cash=None, horizon=3)
    r = _row(lab, cal, 0)
    assert r["label_missing_reason"] == "exit_close_missing" and r["label_matured"] is False
    for c in (
        "total_return_60d",
        "entry_loss_60d",
        "loss_event_60d_8pct",
        "future_max_drawdown_60d",
    ):
        assert r[c] is None
    # 진입 종가 없음
    path2 = make_path(cal, {0: 1, 2: 1.1, 3: 1.2, 4: 1.2, 5: 1.2, 6: 1.2, 7: 1.2})
    r2 = _row(compute_labels("a", cal, path2, cash=None, horizon=3), cal, 0)
    assert r2["label_missing_reason"] == "entry_close_missing" and r2["entry_loss_60d"] is None


def test_immature_labels_and_label_end_at():
    cal = make_cal(12)
    path = make_path(cal, [1.0] * 6)  # 관측은 세션 5까지
    lab = compute_labels("a", cal, path, cash=None, horizon=3)
    assert _row(lab, cal, 1)["label_matured"] is True  # 진입2, 만기5
    r = _row(lab, cal, 2)  # 만기 6은 미관측
    assert r["label_matured"] is False and r["label_missing_reason"] == "not_matured"
    assert r["label_end_at"] == cal.close_at(cal.sessions[6])  # 만기 시각 자체는 안다


def test_cash_series_missing_reason_and_other_labels_populated():
    cal = make_cal(12)
    path = make_path(cal, [1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0])
    lab = compute_labels("a", cal, path, cash=None, horizon=3)
    r = _row(lab, cal, 0)
    assert (
        r["excess_return_60d_vs_cash"] is None
        and r["cash_label_missing_reason"] == "cash_series_missing"
    )
    assert r["entry_loss_60d"] is not None and r["label_missing_reason"] is None


def test_excess_vs_cash_and_market():
    cal = make_cal(12)
    from datetime import timedelta as td

    d0 = cal.sessions[0] - td(days=1)
    cash = build_cash_account(
        pl.DataFrame(
            {"date": [d0], "realtime_start": [d0], "value": [3.65]},
            schema={"date": pl.Date, "realtime_start": pl.Date, "value": pl.Float64},
        ),
        cal.sessions,
        series_id="T",
        staleness_days=1000,
    )
    path = make_path(cal, [1, 1, 1.05, 1.1, 1.1, 1.1, 1.1, 1.1])
    parent = make_path(cal, [1, 1, 1.0, 1.02, 1.02, 1.02, 1.02, 1.02])
    lab = compute_labels("a", cal, path, cash=cash, parent_path=parent, has_parent=True, horizon=3)
    r = _row(lab, cal, 0)
    days = (cal.sessions[4] - cal.sessions[1]).days
    g = np.prod(
        [1 + 0.0365 * (cal.sessions[k + 1] - cal.sessions[k]).days / 365 for k in (1, 2, 3)]
    )
    assert r["cash_proxy_return_60d"] == pytest.approx(g - 1)
    assert r["excess_return_60d_vs_cash"] == pytest.approx(0.10 - (g - 1))
    assert r["excess_return_60d_vs_market"] == pytest.approx(0.10 - 0.02)
    assert days >= 3
    nop = compute_labels("a", cal, path, cash=cash, horizon=3)
    assert _row(nop, cal, 0)["market_label_missing_reason"] == "no_parent_benchmark"


def test_pit_assertion_fails_loudly():
    cal = make_cal(10)
    p = build_asset_panel("a", cal, make_path(cal, [1.0] * 8), horizon=3)
    assert_pit(p, ["price_available_at"])  # 통과
    bad = p.with_columns((pl.col("decision_at") + timedelta(hours=1)).alias("price_available_at"))
    with pytest.raises(PitViolationError, match="price_available_at > decision_at"):
        assert_pit(bad, ["price_available_at"])
    bad2 = p.with_columns(pl.col("decision_at").alias("entry_at"))
    with pytest.raises(PitViolationError, match="decision_at >= entry_at"):
        assert_pit(bad2, ["price_available_at"])


def test_panel_builder_rejects_too_late_availability(monkeypatch):
    cal = make_cal(10)
    monkeypatch.setattr(panel_mod, "PRICE_AVAILABILITY_BUFFER", timedelta(hours=30))
    with pytest.raises(PitViolationError):
        build_asset_panel("a", cal, make_path(cal, [1.0] * 8), horizon=3)


def test_panel_time_contract_and_stale_row():
    cal = make_cal(10)
    p = build_asset_panel("a", cal, make_path(cal, {0: 1, 1: 1.1, 3: 1.2, 4: 1.3}), horizon=3)
    assert (p["price_available_at"] <= p["decision_at"]).all()
    assert (p["decision_at"] < p["entry_at"]).all()
    miss = p.filter(pl.col("session") == cal.sessions[2]).row(0, named=True)
    assert miss["dq_price_missing_at_t"] is True and miss["last_price_session"] == cal.sessions[1]
    assert miss["px_raw_t"] == pytest.approx(110.0)  # as-of, 이동 없이 직전 관측값


def test_build_is_deterministic(tmp_path):
    cal = make_cal(40)
    rng = np.random.default_rng(0)
    idx = np.cumprod(1 + rng.normal(0, 0.01, 40))
    assets = [get_asset("us_spx"), get_asset("us_ndx")]
    paths = {
        "us_spx": make_path(cal, list(idx)),
        "us_ndx": make_path(cal, list(idx**1.3)),
    }
    hashes = []
    for k in range(2):
        panel, labels = assemble(assets, paths, cal, None, horizon=5)
        hashes.append(write_outputs(panel, labels, tmp_path / f"run{k}"))
    assert hashes[0] == hashes[1]
    assert len(hashes[0]["panel.parquet"]) == 64
    a = (tmp_path / "run0" / "labels.parquet").read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashes[0]["labels.parquet"]
    labels = pl.read_parquet(tmp_path / "run0" / "labels.parquet")
    assert (
        labels.filter(pl.col("asset_id") == "us_spx")["excess_return_60d_vs_market"].null_count()
        == labels.filter(pl.col("asset_id") == "us_spx").height
    )  # 부모 없음 -> 전부 null
