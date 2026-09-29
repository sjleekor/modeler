from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from modeler.scores.market_sector import baselines as bl
from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.evaluate import (
    block_ids,
    block_matrix,
    bootstrap_ci,
    draw_counts,
    mean_asset_skill,
)
from modeler.scores.market_sector.scoring import (
    opportunity_scores,
    percentile_against,
    stability_scores,
)

from ._helpers import small_cfg


def test_opportunity_percentile_uses_only_previous_folds():
    rng = np.random.default_rng(0)
    n = 300
    fold = np.repeat([2016, 2017, 2018], 100)
    market = np.array(["US"] * n, dtype=object)
    pred = rng.normal(size=n)
    s1, st1, rn1 = opportunity_scores(pred, fold, market, 50)
    assert np.isnan(s1[fold == 2016]).all() and set(st1[fold == 2016]) == {"warmup"}
    assert (rn1[fold == 2017] == 100).all() and (rn1[fold == 2018] == 200).all()
    # 2017 점수는 2016 예측 분포만으로 정해진다
    ref = pred[fold == 2016]
    expect = percentile_against(ref, pred[fold == 2017], 50)
    assert np.allclose(s1[fold == 2017], expect)
    # 자기 fold 예측을 바꿔도 자기 점수의 reference는 안 변한다
    pred2 = pred.copy()
    pred2[fold == 2018] = pred2[fold == 2018] * 100 + 5
    s2, _, _ = opportunity_scores(pred2, fold, market, 50)
    assert np.allclose(s1[fold == 2017], s2[fold == 2017])
    assert np.allclose(
        s2[fold == 2018], percentile_against(pred2[fold <= 2017], pred2[fold == 2018], 50)
    )
    # 이전 fold 예측이 바뀌면 이후 fold 점수는 바뀐다
    pred3 = pred.copy()
    pred3[fold == 2016] += 10
    s3, _, _ = opportunity_scores(pred3, fold, market, 50)
    assert not np.allclose(s1[fold == 2017], s3[fold == 2017])


def test_opportunity_reference_is_per_market():
    fold = np.array([2016] * 60 + [2017] * 60 + [2016] * 5 + [2017] * 5)
    market = np.array(["US"] * 120 + ["KR"] * 10, dtype=object)
    pred = np.random.default_rng(1).normal(size=130)
    s, st, rn = opportunity_scores(pred, fold, market, 50)
    kr17 = (market == "KR") & (fold == 2017)
    assert np.isnan(s[kr17]).all() and set(st[kr17]) == {"warmup"}  # KR reference는 5개뿐
    assert not np.isnan(s[(market == "US") & (fold == 2017)]).any()


def test_stability_score_is_raw_probability_transform():
    p = np.array([0.0, 0.1, 0.5, np.nan])
    s = stability_scores(p)
    assert s[:3].tolist() == [100.0, 90.0, 50.0] and np.isnan(s[3])


def _toy_pool(n_events):
    """한 자산 200행, 사건 ``n_events`` 개. label_end_at은 결정 시각 + 1."""
    n = 200
    y = np.zeros(n)
    y[:n_events] = 1
    return (
        np.array(["a"] * n, dtype=object),
        np.arange(n, dtype=np.int64) * 10,
        np.arange(n, dtype=np.int64) * 10 + 1,
        y,
    )


def test_stability_null_when_fewer_than_30_events():
    cfg = MsConfig()
    for n_ev, reason in (
        (10, "events_in_train<30"),
        (0, "single_class_in_train"),
        (200, "single_class_in_train"),
        (40, None),
    ):
        asset, dec, end, y = _toy_pool(n_ev)
        st = bl.train_event_status(asset, y, np.arange(200), cfg)
        assert st["a"] == reason


def test_pit_baselines_use_only_matured_labels():
    asset = np.array(["a"] * 6, dtype=object)
    dec = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    end = dec + 25  # 라벨은 25 뒤에 만기
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pool = np.ones(6, dtype=bool)
    m = bl.pit_expanding_mean(asset, dec, end, y, pool)
    # 결정 30: end<30 인 행은 idx0(end=25)만 -> 1.0 ; 결정 50: idx0,1,2(end=25,35,45)
    assert np.isnan(m[0]) and np.isnan(m[2]) and m[3] == 1.0
    assert m[5] == pytest.approx(2.0)
    ev = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    r = bl.pit_smoothed_rate(asset, dec, end, ev, pool, k=60)
    # 결정 30: 자산 n=1 events=1 pooled=1 -> (1+60*1)/(1+60)=1
    assert r["asset_rate"][3] == pytest.approx(1.0)
    assert r["n_asset"][5] == 3


def test_block_bootstrap_resamples_whole_blocks_all_assets_together():
    rng = np.random.default_rng(3)
    n_days, n_assets = 90, 3
    sessions = np.repeat(np.arange(n_days), n_assets)
    a_idx = np.tile(np.arange(n_assets), n_days)
    vals = np.column_stack(
        [rng.random(len(sessions)), rng.random(len(sessions)) + 1, np.ones(len(sessions))]
    )
    block = block_ids(sessions, 30)
    assert block.max() == 2
    m = block_matrix(block, a_idx, vals, n_assets)
    counts = draw_counts(3, 5, seed=0)
    for c in counts:
        assert c.sum() == 3
        # 행 단위로 블록을 이어붙인 것과 합계 행렬로 뽑은 것이 같다 -> 블록 단위, 모든 자산 동반
        rows = np.concatenate([np.flatnonzero(block == b) for b in range(3) for _ in range(c[b])])
        tot_rows = np.zeros((n_assets, 3))
        np.add.at(tot_rows, a_idx[rows], vals[rows])
        tot_blocks = np.tensordot(c, m, axes=(0, 0))
        assert np.allclose(tot_rows, tot_blocks)
        assert all((a_idx[rows] == a).sum() == (a_idx[rows] == 0).sum() for a in range(n_assets))
    out = bootstrap_ci(m, mean_asset_skill, 200, 0)
    assert out["n_blocks"] == 3 and out["lo"] is not None and out["lo"] <= out["hi"]
    # 시드가 같으면 같다
    assert bootstrap_ci(m, mean_asset_skill, 200, 0) == out


def test_config_hash_changes_with_values():
    a, b = MsConfig(), small_cfg()
    assert a.config_hash() != b.config_hash()
    assert a.config_hash() == MsConfig().config_hash()
    d = a.as_dict()
    assert d["horizon_sessions"] == 60 and d["block_sessions"] == 60 and d["ridge_alpha"] == 10.0
    assert d["train_start"] == {"US": "2011-01-03", "KR": "2010-01-04"}
    assert a.first_test_year == {"US": 2016, "KR": 2015}


_ = pl
