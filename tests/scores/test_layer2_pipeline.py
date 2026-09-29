from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from modeler.scores.market_sector.run import (
    OPP_FALLBACK_LABEL,
    run_pipeline,
    write_outputs,
)

from ._helpers import make_layer2_frame, small_cfg


@pytest.fixture(scope="module")
def frame():
    return make_layer2_frame()[0]


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_pipeline_end_to_end_and_deterministic(frame, tmp_path):
    cfg = small_cfg()
    r1 = run_pipeline(frame, cfg, "US", models_dir=tmp_path / "m1")
    r2 = run_pipeline(frame.sample(fraction=1.0, shuffle=True, seed=5), cfg, "US")
    h1 = write_outputs(r1, tmp_path / "o1")
    h2 = write_outputs(r2, tmp_path / "o2")
    assert h1 == h2  # 입력 행 순서와 무관하게 같은 바이트
    assert r1.opportunity_target == OPP_FALLBACK_LABEL  # 합성 패널은 현금 라벨이 없다
    assert r1.reload_verified and all(r1.reload_verified.values())
    assert _sha(tmp_path / "o1" / "oof_predictions.parquet") == h1["oof_predictions.parquet"]
    # 산출물 구조
    oof, sc = r1.oof, r1.scores
    assert {"p_opp_ridge", "p_stab_logit", "b_stab_logit_rvol", "fold_year"} <= set(oof.columns)
    assert oof["fold_year"].min() == cfg.first_test_year["US"]
    stab = sc.filter(pl.col("score_type") == "stability")
    assert set(stab["calibration_method"].unique()) == {"none"}
    assert set(stab["validation_status"].unique()) == {"research"}
    p = stab.filter(pl.col("model") == "stab_logit").drop_nulls("raw_prediction")
    assert np.allclose(p["score"].to_numpy(), 100 * (1 - p["raw_prediction"].to_numpy()))
    # 첫 fold의 Opportunity 점수는 warm-up
    first = sc.filter((pl.col("score_type") == "opportunity") & (pl.col("fold_year") == 2017))
    assert first["score"].null_count() == first.height
    assert set(first["score_status"].unique()) == {"warmup"}
    assert r1.latest.height == 3 and r1.latest["stab_logit_p_hat"].null_count() == 0
    assert r1.metrics["stability"]["logit"]["n_rows"] > 0


def test_train_predictions_only_from_past_labels(frame):
    """OOF 행의 학습 경계: 예측이 있는 모든 fold의 학습 행 수가 fold마다 늘어난다."""
    r = run_pipeline(frame, small_cfg(), "US")
    n_train = [f["n_train"] for f in r.fold_table if not f["skipped_reason"]]
    assert n_train == sorted(n_train) and len(set(n_train)) == len(n_train)


def test_kr_run_requires_kr_data():
    from modeler.scores.market_sector.build_panel import KrNotSyncedError
    from modeler.scores.market_sector.run import main

    with pytest.raises(KrNotSyncedError):
        main(["--market", "kr"])
