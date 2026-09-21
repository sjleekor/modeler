"""``modeler.us.m6_run`` 단위 테스트 — 전부 합성 데이터다. 실제 레이크를
읽지 않는다(``06_execution_steps.md`` M6 §6 지시).

CPCV가 ``inputs.dates``(holdout 벽을 이미 통과한 값) 밖의 날짜를 절대
만들어내지 않는지, 그리고 DSR의 시행 수(N) 집계가 model_id별 grid_search
크기를 정확히 세는지를 검사한다.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.m4_run import M4Inputs
from modeler.us.m6_run import CPCV_N_FOLDS, CPCV_N_TEST_FOLDS, _gather_trials, run_cpcv


def _make_inputs(n_months: int = 48, n_symbols: int = 20) -> M4Inputs:
    dates: list[date] = []
    y, mo = 2020, 1
    for _ in range(n_months):
        dates.append(date(y, mo, 1))
        mo += 1
        if mo > 12:
            mo = 1
            y += 1

    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        for s in range(n_symbols):
            f1 = rng.normal()
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{s:03d}",
                    "f1_rank": float(s) / (n_symbols - 1),
                    "f1_isna": False,
                    "L0": float(f1) * 0.01,
                    "y_rank": float(s) / (n_symbols - 1),
                }
            )
    core = pl.DataFrame(rows)
    return M4Inputs(
        core=core,
        dates=dates,
        model_features=["f1"],
        m3_manifest_path=Path("dummy.json"),
        features_content_hash=None,
        labels_content_hash=None,
        features_row_count=core.height,
        labels_row_count=core.height,
    )


def test_run_cpcv_never_uses_dates_outside_inputs_dates() -> None:
    inputs = _make_inputs()
    result = run_cpcv(inputs)

    expected_n_splits = math.comb(CPCV_N_FOLDS, CPCV_N_TEST_FOLDS)
    assert result["n_paths"] == expected_n_splits
    assert len(result["paths"]) == expected_n_splits

    dev_dates = set(inputs.dates)
    for path in result["paths"]:
        assert date.fromisoformat(path["test_date_min"]) in dev_dates
        assert date.fromisoformat(path["test_date_max"]) in dev_dates
        # holdout 벽은 build_m4_inputs가 이미 지킨다 — CPCV는 inputs.dates의
        # 인덱스만 재배열하므로 그 범위를 벗어난 날짜를 만들 수 없다.
        assert date.fromisoformat(path["test_date_min"]) <= date.fromisoformat(
            path["test_date_max"]
        )


def test_run_cpcv_reports_configured_purge_embargo() -> None:
    inputs = _make_inputs()
    result = run_cpcv(inputs)
    assert result["n_folds"] == CPCV_N_FOLDS
    assert result["n_test_folds"] == CPCV_N_TEST_FOLDS
    assert result["pbo_fraction_sharpe_nonpositive"] is not None
    assert 0.0 <= result["pbo_fraction_sharpe_nonpositive"] <= 1.0


# --- 2. _gather_trials — DSR의 N 세는 법 -----------------------------------------


def _write_run(
    root: DataRoot, run_id: str, *, grid_search: list[dict] | None, rank_ic_mean: float
) -> None:
    out_dir = root.output / "model_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rank_ic_mean": rank_ic_mean}
    if grid_search is not None:
        payload["grid_search"] = grid_search
    (out_dir / "metrics.json").write_text(json.dumps(payload))


def test_gather_trials_counts_grid_search_len_or_one_scalar_trial(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    _write_run(
        root,
        "m4_ridge_20260921",
        grid_search=[{"mean_ic": 0.01}, {"mean_ic": 0.02}, {"mean_ic": 0.03}, {"mean_ic": 0.04}],
        rank_ic_mean=0.04,
    )
    _write_run(
        root,
        "m4_enet_20260921",
        grid_search=[{"mean_ic": v} for v in range(12)],
        rank_ic_mean=0.05,
    )
    _write_run(root, "m4_ols3_20260921", grid_search=None, rank_ic_mean=0.009)
    _write_run(
        root,
        "m4_lgbm_20260921",
        grid_search=[{"mean_ic": v} for v in range(24)],
        rank_ic_mean=0.056,
    )
    _write_run(root, "m5_ensemble_20260921", grid_search=None, rank_ic_mean=0.0565)

    trials = _gather_trials(root)

    assert len(trials["m4_ridge"]) == 4
    assert len(trials["m4_enet"]) == 12
    assert trials["m4_ols3"] == [0.009]
    assert len(trials["m4_lgbm"]) == 24
    assert trials["m5_ensemble"] == [0.0565]
    total = sum(len(v) for v in trials.values())
    assert total == 42  # 06 M6 지시가 준 값과 같다


def test_gather_trials_missing_run_raises(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    with pytest.raises(FileNotFoundError):
        _gather_trials(root)
