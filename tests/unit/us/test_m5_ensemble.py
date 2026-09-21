"""``modeler.us.m5_ensemble`` 단위 테스트 — 전부 합성 데이터다. 실제 레이크를
읽지 않는다(M5 지시 §6 "실제 레이크를 읽는 pytest를 만들지 마라").

M-E 정의(``03_model_candidates.md`` §4, M5 지시 §2)의 핵심 두 가지를 검사한다:

1. 순위 평균이 시드별 예측 스케일에 흔들리지 않는가(백분위로 바꾼 뒤에만
   평균한다).
2. fold별 rank IC 집계, 채택 규칙(``03`` §5) 판정 로직.

LightGBM을 실제로 학습시키는 ``run_mg_seed``/``run_ensemble`` 전체 파이프라인은
여기서 돌리지 않는다 — ``min_data_in_leaf=200``짜리 실제 모델 적합에는
합성으로 만들기엔 과한 데이터가 필요하고, ``test_m4_run.py``도 같은 이유로
``run_lgbm`` 전체를 단위 테스트하지 않는다. 대신 ``build_ensemble_oof``·
``fold_rank_ics``·``adoption_verdict``·``load_m4_comparison``처럼 모델 적합과
무관한 데이터 변환·판정 로직을 직접 검사한다.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.m4_splits import build_wf_folds
from modeler.us.m4_transform import cross_sectional_percentile
from modeler.us.m5_ensemble import (
    DEV_END,
    adoption_verdict,
    build_ensemble_oof,
    fold_rank_ics,
    load_m4_comparison,
)
from modeler.us.scan import DEV_END as SCAN_DEV_END

D1 = date(2020, 1, 2)
D2 = date(2020, 2, 3)


def _seed_oof(dates: list[date], preds_by_symbol: dict[str, list[float]]) -> pl.DataFrame:
    """symbol별 (날짜 수만큼의) pred 리스트로 시드 하나의 OOF 프레임을 만든다.

    L2·y_rank는 이 테스트들에서 핵심이 아니라 symbol 순번을 그대로 쓴다.
    """
    symbols = sorted(preds_by_symbol)
    rows = []
    for date_idx, d in enumerate(dates):
        for sym_idx, sym in enumerate(symbols):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "month_idx": date_idx + 1,
                    "fold_id": 1,
                    "L2": float(sym_idx),
                    "y_rank": float(sym_idx) / max(len(symbols) - 1, 1),
                    "pred": preds_by_symbol[sym][date_idx],
                }
            )
    return pl.DataFrame(rows)


# --- 1. build_ensemble_oof — 정의대로 계산되는가 --------------------------------


def test_build_ensemble_oof_matches_manual_percentile_mean() -> None:
    """2 시드 · 1 날짜 · 4 종목. 손으로 백분위 평균을 계산해 맞춰 본다."""
    seed0 = _seed_oof([D1], {"A": [10.0], "B": [40.0], "C": [20.0], "D": [30.0]})
    seed1 = _seed_oof([D1], {"A": [1.0], "B": [2.0], "C": [4.0], "D": [3.0]})

    out = build_ensemble_oof({0: seed0, 1: seed1})

    # seed0 순위(percentile): A=0,C=1/3,D=2/3,B=1 / seed1 순위: A=0,B=1/3,D=2/3,C=1
    # 평균: A=0, B=(1+1/3)/2=2/3, C=(1/3+1)/2=2/3, D=(2/3+2/3)/2=2/3
    got_pctile0 = dict(zip(out["symbol"].to_list(), out["pctile_seed0"].to_list(), strict=True))
    got_pctile1 = dict(zip(out["symbol"].to_list(), out["pctile_seed1"].to_list(), strict=True))
    got_mean = dict(
        zip(out["symbol"].to_list(), out["ensemble_pctile_mean"].to_list(), strict=True)
    )
    assert got_pctile0 == pytest.approx({"A": 0.0, "C": 1 / 3, "D": 2 / 3, "B": 1.0})
    assert got_pctile1 == pytest.approx({"A": 0.0, "B": 1 / 3, "D": 2 / 3, "C": 1.0})
    assert got_mean == pytest.approx({"A": 0.0, "B": 2 / 3, "C": 2 / 3, "D": 2 / 3})

    # 평균을 다시 그날 횡단면 순위로 본다: A가 최소(0)라 pred=0.0, 나머지 셋(B,C,D)은
    # 값이 같아(2/3) 동순위 평균 순위를 나눠 갖는다 — 순서는 무관하고 A만 확정된다.
    assert out.filter(pl.col("symbol") == "A")["pred"].item() == pytest.approx(0.0)


def test_build_ensemble_oof_is_invariant_to_per_seed_monotonic_rescaling() -> None:
    """M5 지시 §2 핵심 — "시드마다 예측 스케일이 다를 수 있다."

    한 시드의 예측을 통째로 스케일·이동(단조변환)해도 백분위는 그대로이므로
    최종 앙상블 ``pred``가 바뀌면 안 된다.
    """
    dates = [D1, D2]
    base_seed0 = _seed_oof(
        dates, {"A": [10.0, 1.0], "B": [40.0, 4.0], "C": [20.0, 3.0], "D": [30.0, 2.0]}
    )
    base_seed1 = _seed_oof(
        dates, {"A": [5.0, 9.0], "B": [1.0, 2.0], "C": [9.0, 1.0], "D": [3.0, 5.0]}
    )

    rescaled_seed1 = base_seed1.with_columns((pl.col("pred") * 1000.0 + 7.0).alias("pred"))

    out_base = build_ensemble_oof({0: base_seed0, 1: base_seed1}).sort("date", "symbol")
    out_rescaled = build_ensemble_oof({0: base_seed0, 1: rescaled_seed1}).sort("date", "symbol")

    assert out_base["pred"].to_list() == pytest.approx(out_rescaled["pred"].to_list())
    assert out_base["ensemble_pctile_mean"].to_list() == pytest.approx(
        out_rescaled["ensemble_pctile_mean"].to_list()
    )


def test_build_ensemble_oof_seed_pctile_columns_match_cross_sectional_percentile() -> None:
    seed0 = _seed_oof([D1], {"A": [1.0], "B": [2.0], "C": [3.0]})
    seed1 = _seed_oof([D1], {"A": [9.0], "B": [8.0], "C": [7.0]})

    out = build_ensemble_oof({0: seed0, 1: seed1})

    expected0 = cross_sectional_percentile(seed0, "pred", date_col="date", out_col="pctile_seed0")
    expected1 = cross_sectional_percentile(seed1, "pred", date_col="date", out_col="pctile_seed1")
    exp0 = dict(
        zip(expected0["symbol"].to_list(), expected0["pctile_seed0"].to_list(), strict=True)
    )
    exp1 = dict(
        zip(expected1["symbol"].to_list(), expected1["pctile_seed1"].to_list(), strict=True)
    )
    got0 = dict(zip(out["symbol"].to_list(), out["pctile_seed0"].to_list(), strict=True))
    got1 = dict(zip(out["symbol"].to_list(), out["pctile_seed1"].to_list(), strict=True))
    assert got0 == pytest.approx(exp0)
    assert got1 == pytest.approx(exp1)


def test_build_ensemble_oof_requires_at_least_two_seeds() -> None:
    seed0 = _seed_oof([D1], {"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="2개 이상"):
        build_ensemble_oof({0: seed0})


def test_build_ensemble_oof_rejects_mismatched_seed_row_counts() -> None:
    seed0 = _seed_oof([D1, D2], {"A": [1.0, 2.0], "B": [2.0, 3.0]})
    seed1 = _seed_oof([D1], {"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="행수"):
        build_ensemble_oof({0: seed0, 1: seed1})


# --- 2. fold_rank_ics -----------------------------------------------------------


def test_fold_rank_ics_computes_ic_per_fold() -> None:
    n = 20
    rows = []
    for fold_id, sign in ((1, 1.0), (2, -1.0)):
        for s in range(n):
            rows.append(
                {
                    "date": D1 if fold_id == 1 else D2,
                    "fold_id": fold_id,
                    "pred": sign * s,
                    "L2": float(s),
                }
            )
    oof = pl.DataFrame(rows)

    fold_ics = fold_rank_ics(oof)

    assert fold_ics == pytest.approx([1.0, -1.0])


def test_fold_rank_ics_below_min_names_is_nan() -> None:
    rows = [{"date": D1, "fold_id": 1, "pred": float(s), "L2": float(s)} for s in range(5)]
    oof = pl.DataFrame(rows)

    fold_ics = fold_rank_ics(oof)

    assert len(fold_ics) == 1
    assert math.isnan(fold_ics[0])


# --- 3. adoption_verdict — 03 §5 채택 규칙 ---------------------------------------


def test_adoption_verdict_me_beats_both_ml() -> None:
    comparison = {
        "m4_ridge": {"rank_ic_mean": 0.05},
        "m4_enet": {"rank_ic_mean": 0.04},
    }
    result = adoption_verdict(0.06, comparison)
    assert result["beats_m4_ridge"] is True
    assert result["beats_m4_enet"] is True
    assert "M-E 채택" in result["verdict"]


def test_adoption_verdict_me_loses_to_both_ml() -> None:
    """M4 실측(2026-09-21): M-E ≈ M-G(0.0567)는 M-L Ridge(0.0591)·ENet(0.0597)
    둘 다보다 낮다 — 이 테스트는 그 모양(둘 다 짐)을 판정 로직으로 검사한다."""
    comparison = {
        "m4_ridge": {"rank_ic_mean": 0.0591},
        "m4_enet": {"rank_ic_mean": 0.0597},
    }
    result = adoption_verdict(0.0567, comparison)
    assert result["beats_m4_ridge"] is False
    assert result["beats_m4_enet"] is False
    assert "M-L 채택" in result["verdict"]


def test_adoption_verdict_split_result() -> None:
    comparison = {
        "m4_ridge": {"rank_ic_mean": 0.05},
        "m4_enet": {"rank_ic_mean": 0.07},
    }
    result = adoption_verdict(0.06, comparison)
    assert result["beats_m4_ridge"] is True
    assert result["beats_m4_enet"] is False
    assert "엇갈림" in result["verdict"]


def test_adoption_verdict_missing_comparison_data_is_not_silently_a_win() -> None:
    """빈 dict에서 ``all([])``이 True라 데이터가 없는데 "둘 다 이겼다"로 잘못
    읽힐 수 있었다 — 회귀 테스트."""
    result = adoption_verdict(0.06, {})
    assert result["beats_m4_ridge"] is None
    assert result["beats_m4_enet"] is None
    assert "판정 불가" in result["verdict"]


# --- 4. load_m4_comparison — M4 산출물을 다시 도는 대신 읽기만 한다 ----------------


def _write_m4_run(root: DataRoot, model_id: str, snapshot: str, metrics: dict) -> None:
    d = root.output / "model_runs" / f"{model_id}_{snapshot}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))


def test_load_m4_comparison_reads_latest_snapshot_per_model(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    _write_m4_run(root, "m4_ridge", "20260901", {"rank_ic_mean": 0.01, "selected_params": {}})
    _write_m4_run(
        root,
        "m4_ridge",
        "20260921",
        {"rank_ic_mean": 0.0591, "selected_params": {"alpha": 100.0}},
    )
    _write_m4_run(root, "m4_enet", "20260921", {"rank_ic_mean": 0.0597, "selected_params": {}})
    _write_m4_run(root, "m4_lgbm", "20260921", {"rank_ic_mean": 0.0567, "selected_params": {}})

    comparison = load_m4_comparison(root)

    assert comparison["m4_ridge"]["rank_ic_mean"] == pytest.approx(0.0591)
    assert "20260921" in comparison["m4_ridge"]["run_dir"]
    assert comparison["m4_enet"]["rank_ic_mean"] == pytest.approx(0.0597)
    assert comparison["m4_lgbm"]["rank_ic_mean"] == pytest.approx(0.0567)


def test_load_m4_comparison_missing_run_reports_error(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    comparison = load_m4_comparison(root)
    assert "error" in comparison["m4_ridge"]


def test_load_m4_comparison_does_not_confuse_lgbm_with_lgbm_on_l1(tmp_path: Path) -> None:
    """회귀 테스트 — 2026-09-21 M5 첫 실행 실측 버그.

    ``m4_lgbm_on_l1_20260921``(대조군)이 ``m4_lgbm_``으로 시작해 접두어가
    충돌하고, 문자열 정렬상 ``m4_lgbm_on_l1_...``이 ``m4_lgbm_2026...``보다
    뒤라 "최신"으로 잘못 뽑혔다. ``m4_lgbm`` 조회는 정확히
    ``m4_lgbm_<날짜>``만 골라야 한다.
    """
    root = DataRoot(base=tmp_path)
    _write_m4_run(root, "m4_lgbm", "20260921", {"rank_ic_mean": 0.0567, "selected_params": {}})
    _write_m4_run(
        root, "m4_lgbm_on_l1", "20260921", {"rank_ic_mean": 0.0517, "selected_params": {}}
    )

    comparison = load_m4_comparison(root)

    assert comparison["m4_lgbm"]["rank_ic_mean"] == pytest.approx(0.0567)
    assert "m4_lgbm_on_l1" not in comparison["m4_lgbm"]["run_dir"]


# --- 5. holdout 벽 — scan.DEV_END를 그대로 재사용하는가 (별도 정의가 아닌지) -------


def test_m5_ensemble_reuses_scan_dev_end_directly() -> None:
    assert DEV_END is SCAN_DEV_END


def test_m5_ensemble_build_wf_folds_rejects_holdout_dates() -> None:
    """이 모듈이 fold를 만들 때 쓰는 ``build_wf_folds``(``m4_splits``에서 그대로
    가져온 것)가 holdout 날짜를 넘기면 여전히 거부하는지 — M5가 자기만의
    (약해진) 버전을 만들지 않았는지 확인한다."""
    holdout_date = date(2025, 7, 1)
    assert holdout_date > DEV_END
    with pytest.raises(ValueError, match="holdout"):
        build_wf_folds([holdout_date], dev_end=DEV_END)
