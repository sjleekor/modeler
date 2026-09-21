"""M7 — holdout 개봉 (``05_validation_protocol.md`` §5, ``06_execution_steps.md`` M7).

    uv run --offline python -m modeler.us.m7_run

채택 모델(**M-L Ridge, alpha=100** — M5 §5 "채택")을 개발 구간
**전체**(``modeler.us.scan.DEV_END`` 이전, 2018-09-07~2025-06-30)로 **한 번만**
재학습해 holdout(``HOLDOUT_START``~``HOLDOUT_END``, 2025-07-01~2026-06-30,
12회 리밸런스)에 예측한다. **walk-forward가 아니다** — fold를 나누지 않고
``build_m4_inputs``가 조립한 개발 구간 전체 프레임 하나로 ``fit``, holdout
프레임 하나에 ``predict``를 딱 한 번씩만 부른다.

**딱 한 번만 연다.** 이 파일은 채택 설정(Ridge alpha=100, 입력 피쳐 6개)
말고 다른 모델·다른 그리드를 holdout에 돌리는 경로를 갖지 않는다 — 그리드
서치 함수(``m4_models.ridge_grid`` 등)를 import하지 않는다.

**holdout 벽을 여기서 의도적으로 넘는다.** ``load_holdout_frame``이 그
벽 너머(``HOLDOUT_START``~``HOLDOUT_END``)만 읽고, 학습 프레임은 그대로
``modeler.us.scan.assert_dev_window``(``DEV_END`` 이전)를 거친다 —
``assert_no_window_overlap``이 두 프레임에 겹치는 날짜가 없는지 마지막으로
한 번 더 확인한다.

**지표·분해는 M6이 만든 것을 그대로 재사용한다** (``06`` M7 지시).
``modeler.us.m6_run.compute_model_metrics``가 ``05`` §4 표 전부(``E``·
``E_ew``·``I``·``S``·``MDD``·``turnover``·``breakeven_Q``·``hit_top``·R2
분해·top-100 시총 분포·Q×k 민감도)를 낸다 — 여기서 다시 짜지 않는다. 이
파일이 새로 더하는 것은 셋뿐이다: holdout 재학습·예측 루프, 갈래 판정
(``00`` §3.2, 부호만), 갈래 A 귀무 확률(``06`` M8 재료, 부호 뒤섞기
permutation).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.us import benchmark as bm
from modeler.us import metrics as m
from modeler.us.lake import UsLake
from modeler.us.m4_models import fit_predict_ridge
from modeler.us.m4_run import _dataset_manifest_field, build_m4_inputs
from modeler.us.m4_transform import rank_transform, to_design_arrays
from modeler.us.m6_run import (
    ADOPTED_RIDGE_ALPHA,
    bucket_universe,
    compute_model_metrics,
    joined_frame,
)
from modeler.us.scan import DEV_END, DEV_START, assert_dev_window

# --- 0. 상수 --------------------------------------------------------------------

#: holdout 구간 — ``05`` §1. ``HOLDOUT_END`` 뒤(2026-07-01~)는 빼는 이유가
#: 거기 적혀 있다: ``filings_sub``·``midas``·``insider_trans``가 2026-06-30에
#: 끝나 업종·``mcap_rank``가 없다.
HOLDOUT_START: date = date(2025, 7, 1)
HOLDOUT_END: date = date(2026, 6, 30)

#: 기대하는 리밸런스 횟수 — ``05`` §1 "12회". 다르면 경고만 하고 멈추지
#: 않는다(레이크가 그새 갱신됐을 수 있다 — 부호 판정 자체는 개수와 무관하다).
EXPECTED_HOLDOUT_MONTHS = 12

#: 갈래 A 귀무 확률(``06`` M8) — 부호 뒤섞기 permutation 횟수·시드.
#: 시드는 이 계획을 실행한 날짜(``scan.PLACEBO_SAMPLE_SEED``와 같은 관례) —
#: 재현성 자체가 목적이라 값이 무엇이든 상관없고, 한 번 정하면 바꾸지 않는다.
PERMUTATION_N = 1000
PERMUTATION_SEED = 20260921

#: M3 산출물에서 읽은 모델 입력 6개를 손으로 적지 않는다 — 여기 상수로
#: 박지 않고 항상 ``m4_run.load_m3_model_input_features``로 읽는다(모듈
#: docstring 참고, R6 동결).


# --- 1. holdout 날짜 벽 ----------------------------------------------------------


def enforce_holdout_window(
    df: pl.DataFrame, *, start: date = HOLDOUT_START, end: date = HOLDOUT_END
) -> pl.DataFrame:
    """``date`` 컬럼이 ``[start, end]`` 밖인 행을 버린다 — holdout 벽 그 자체.

    ``scan.enforce_dev_window``와 대칭이다: 그쪽이 ``dev_end`` 이전만 남기듯
    이 함수는 holdout 구간만 남긴다.
    """
    if "date" not in df.columns:
        raise ValueError("enforce_holdout_window: 'date' 컬럼이 없습니다")
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def assert_holdout_window(
    df: pl.DataFrame,
    *,
    start: date = HOLDOUT_START,
    end: date = HOLDOUT_END,
    dev_end: date = DEV_END,
) -> None:
    """``df``에 개발 구간 날짜나 지정 구간 밖 날짜가 섞였으면 예외.

    ``scan.assert_dev_window``와 짝이다 — 그쪽은 학습 프레임을, 이 함수는
    예측(holdout) 프레임을 마지막에 다시 확인하는 방어선이다.
    """
    if df.height == 0:
        raise ValueError(f"holdout 프레임이 비어 있습니다 ([{start}, {end}])")
    dmin, dmax = df["date"].min(), df["date"].max()
    if dmin <= dev_end:
        raise ValueError(
            f"holdout 프레임에 개발 구간 날짜가 섞였습니다: min(date)={dmin} <= dev_end={dev_end}"
        )
    if dmin < start or dmax > end:
        raise ValueError(f"holdout 프레임이 [{start}, {end}] 밖입니다: [{dmin}, {dmax}]")


def load_holdout_frame(
    root: DataRoot, name: str, *, start: date = HOLDOUT_START, end: date = HOLDOUT_END
) -> pl.DataFrame:
    """``root.datasets/<name>/part.parquet``를 읽고 holdout 구간만 남긴다.

    ``scan.load_dev_frame``과 대칭이다. 원본 데이터셋은 개발 구간 뒤로도
    행이 있다(레이크가 계속 갱신되기 때문) — 이 함수가 그중 holdout
    구간만 잘라낸다.
    """
    path = root.datasets / name / "part.parquet"
    df = pl.read_parquet(path)
    win = enforce_holdout_window(df, start=start, end=end)
    assert_holdout_window(win, start=start, end=end)
    return win


def assert_no_window_overlap(train_dates: list[date], holdout_dates: list[date]) -> None:
    """학습 프레임과 예측 프레임에 겹치는 날짜가 없는지 마지막으로 다시 확인한다.

    ``build_m4_inputs``가 이미 ``assert_dev_window``로, ``load_holdout_frame``이
    이미 ``assert_holdout_window``로 각자의 구간을 지키지만, "학습 프레임에
    holdout이 한 행도 섞이면 안 된다"는 지시(``06`` M7)를 코드 한 곳에
    명시적으로 다시 건다.
    """
    overlap = set(train_dates) & set(holdout_dates)
    if overlap:
        raise ValueError(f"학습·예측 프레임에 겹치는 날짜가 있습니다: {sorted(overlap)}")
    if train_dates and holdout_dates and max(train_dates) >= min(holdout_dates):
        raise ValueError(
            f"학습 구간 끝({max(train_dates)})이 예측 구간 시작({min(holdout_dates)})보다 "
            "앞서지 않습니다"
        )


# --- 2. holdout 설계 프레임 ------------------------------------------------------


def build_holdout_core(
    features_holdout: pl.DataFrame, labels_holdout: pl.DataFrame, model_features: list[str]
) -> pl.DataFrame:
    """holdout 설계 프레임 — ``build_m4_inputs``와 같은 조립(순위 변환 · 조인)을
    holdout 창 하나, 모델 입력 6개에만 적용한다.

    OLS-3 대조군 컬럼은 붙이지 않는다 — M7은 채택 모델(Ridge) 하나만 연다
    (``06`` M7 "다른 모델을 holdout에 돌리지 않는다").
    """
    isna_cols = [f"{c}_isna" for c in model_features]
    base_cols = ["date", "symbol", *model_features, *isna_cols]
    core = features_holdout.select(base_cols).join(
        labels_holdout.select("date", "symbol", "L0", "L1", "L2", "y_rank", "y_up"),
        on=["date", "symbol"],
        how="inner",
    )
    assert_holdout_window(core)
    core = rank_transform(core, model_features)
    return core.with_columns(pl.col("date").rank(method="dense").cast(pl.Int64).alias("month_idx"))


def fit_predict_holdout(
    train_core: pl.DataFrame,
    holdout_core: pl.DataFrame,
    model_features: list[str],
    *,
    alpha: float = ADOPTED_RIDGE_ALPHA,
) -> np.ndarray:
    """채택 모델(Ridge, ``alpha``)을 개발 구간 전체로 **한 번** 학습해 holdout에 예측한다.

    이 함수 호출 1회가 재학습 1회다 — walk-forward 루프가 없다(``05`` §5
    "재학습은 개발 구간 전체로 한 번").
    """
    x_train, _ = to_design_arrays(train_core, model_features)
    y_train = train_core["y_rank"].to_numpy()
    x_holdout, _ = to_design_arrays(holdout_core, model_features)
    return fit_predict_ridge({"alpha": alpha}, x_train, y_train, x_holdout)


def build_predictions(holdout_core: pl.DataFrame, preds: np.ndarray) -> pl.DataFrame:
    return holdout_core.select("date", "symbol", "month_idx", "L2", "y_rank").with_columns(
        pl.Series("pred", preds)
    )


# --- 3. 갈래 판정 — 부호만 (``00`` §3.2) -----------------------------------------


@dataclass
class BranchDecision:
    branch: str
    E: float
    E_ew: float
    I: float  # noqa: E741 - 05 §4 지표 이름 그대로
    S: float

    def to_dict(self) -> dict:
        return {"branch": self.branch, "E": self.E, "E_ew": self.E_ew, "I": self.I, "S": self.S}


def decide_branch(metrics_result) -> BranchDecision:
    """``00`` §3.2 갈래 판정 — 부호만 본다. 문턱을 두지 않는다.

    C: ``E_ew <= 0`` (종목 선별력이 없다)
    A: ``E > 0 ∧ E_ew > 0 ∧ I > 0 ∧ S > 0``
    B: 그 외
    """
    e, e_ew, i, s = (
        metrics_result.E,
        metrics_result.E_ew,
        metrics_result.I,
        metrics_result.S,
    )
    if e_ew <= 0:
        branch = "C"
    elif e > 0 and e_ew > 0 and i > 0 and s > 0:
        branch = "A"
    else:
        branch = "B"
    return BranchDecision(branch=branch, E=e, E_ew=e_ew, I=i, S=s)


# --- 4. 월별 시계열 넷 — 갈래 A 귀무 확률 재료 (``06`` M8) ------------------------


def monthly_series_frame(
    joined: pl.DataFrame, *, universe_ew: pl.DataFrame, spy: pl.DataFrame
) -> pl.DataFrame:
    """``E``·``E_ew``·``I``·``S`` 월별 시계열 하나 — ``date, E, E_ew, I, S``.

    넷 다 같은 ``joined``(holdout 예측 + 유니버스)에서 뽑는다. ``E``·``E_ew``는
    top-100(전체 유니버스) 궤적, ``I``는 ``price_ge_5`` 유니버스 궤적,
    ``S``는 예측 10분위 스프레드다 — ``compute_model_metrics``가 계산하는
    스칼라 평균과 같은 정의의 시계열판이다.
    """
    track = m.portfolio_track(joined)
    track_ge5 = m.portfolio_track(joined.filter(pl.col("price_ge_5")))

    e_series = m.excess_over_series(
        track, spy, return_col="net_return", benchmark_col="spy_h21_return"
    ).rename({"excess": "E"})
    e_ew_series = m.excess_over_series(
        track, universe_ew, return_col="net_return", benchmark_col="ew_l0_h21_return"
    ).rename({"excess": "E_ew"})
    i_series = m.excess_over_series(
        track_ge5, spy, return_col="net_return", benchmark_col="spy_h21_return"
    ).rename({"excess": "I"})
    s_monthly = m.s_spread_long_short(joined)
    s_series = s_monthly.select("date", pl.col("spread").alias("S"))

    return (
        e_series.join(e_ew_series, on="date", how="inner")
        .join(i_series, on="date", how="inner")
        .join(s_series, on="date", how="inner")
        .sort("date")
    )


# --- 5. 출력 ---------------------------------------------------------------------


def _git_commit(repo: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def _window_summary(df: pl.DataFrame, dates: list[date]) -> dict:
    return {
        "n_rows": df.height,
        "n_months": len(dates),
        "date_min": dates[0].isoformat() if dates else None,
        "date_max": dates[-1].isoformat() if dates else None,
    }


def closed_by_reason_counts(labels_holdout: pl.DataFrame) -> dict[str, int]:
    """holdout 구간에서 가격이 끊겨 닫은 (date, symbol) 수 — 사유별(``05`` §4)."""
    counts = (
        labels_holdout.filter(pl.col("close_reason").is_not_null())
        .group_by("close_reason")
        .agg(pl.len().alias("n"))
    )
    return {row["close_reason"]: row["n"] for row in counts.iter_rows(named=True)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--n-permutation", type=int, default=PERMUTATION_N)
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    args = parser.parse_args(argv)
    snapshot_date = args.snapshot_date or date.today().isoformat()

    t0 = time.monotonic()
    root = DataRoot.resolve(market="us")
    modeler_repo = Path(__file__).resolve().parents[3]
    modeler_git_commit = _git_commit(modeler_repo)

    # 1. 학습 — 개발 구간 전체, 한 번 --------------------------------------------
    print("학습 입력 조립 (개발 구간 전체) ...", flush=True)
    train_inputs = build_m4_inputs(root)  # build_wf_folds를 안 써서 walk-forward가 아니다
    assert_dev_window(train_inputs.core)  # 재확인 — holdout이 섞였으면 여기서 죽는다
    model_features = train_inputs.model_features
    train_dates = train_inputs.dates
    print(f"  모델 입력 6개: {model_features} (M3: {train_inputs.m3_manifest_path})")
    print(
        f"  학습 프레임: {train_inputs.core.height}행, "
        f"{train_dates[0]}~{train_dates[-1]} ({len(train_dates)}개월)"
    )

    # 2. holdout 프레임 조립 -------------------------------------------------------
    print("holdout 프레임 조립 ...", flush=True)
    features_holdout = load_holdout_frame(root, "us_features_v1")
    labels_holdout = load_holdout_frame(root, "us_labels_v1")
    holdout_core = build_holdout_core(features_holdout, labels_holdout, model_features)
    holdout_dates = sorted(holdout_core["date"].unique().to_list())
    assert_holdout_window(holdout_core)
    assert_no_window_overlap(train_dates, holdout_dates)
    print(
        f"  holdout 프레임: {holdout_core.height}행, "
        f"{holdout_dates[0]}~{holdout_dates[-1]} ({len(holdout_dates)}개월)"
    )
    if len(holdout_dates) != EXPECTED_HOLDOUT_MONTHS:
        print(
            f"  *** 경고: holdout 리밸런스가 {EXPECTED_HOLDOUT_MONTHS}회가 아니라 "
            f"{len(holdout_dates)}회입니다 ***",
            flush=True,
        )

    # 3. 재학습 한 번 · 예측 -------------------------------------------------------
    print("Ridge(alpha=100) 재학습 (개발 구간 전체, 1회) · holdout 예측 ...", flush=True)
    preds = fit_predict_holdout(train_inputs.core, holdout_core, model_features)
    pred_df = build_predictions(holdout_core, preds)

    # 4. 유니버스 · 벤치마크 --------------------------------------------------------
    lake = UsLake.resolve()
    universe = bucket_universe(lake, labels_holdout)
    joined = joined_frame(pred_df, universe)

    universe_ew = bm.universe_equal_weight_monthly(universe)
    spy = bm.spy_monthly_return(lake, holdout_dates)
    ew_minus_spy = universe_ew.join(spy, on="date", how="inner").with_columns(
        (pl.col("ew_l0_h21_return") - pl.col("spy_h21_return")).alias("ew_minus_spy")
    )
    ew_minus_spy_mean = (
        float(ew_minus_spy["ew_minus_spy"].mean()) if ew_minus_spy.height else float("nan")
    )

    # 5. 05 §4 지표표 — m6_run.compute_model_metrics 그대로 재사용 ------------------
    metrics_result = compute_model_metrics(
        "m7_holdout",
        joined,
        universe_ew=universe_ew,
        spy=spy,
        ew_minus_spy_mean=ew_minus_spy_mean,
    )

    # 6. 갈래 판정 — 부호만 --------------------------------------------------------
    branch = decide_branch(metrics_result)
    print(
        f"갈래 {branch.branch}: E={branch.E:.4f} E_ew={branch.E_ew:.4f} "
        f"I={branch.I:.4f} S={branch.S:.4f}",
        flush=True,
    )

    # 7. S 롱/숏 분해 --------------------------------------------------------------
    s_monthly = m.s_spread_long_short(joined)
    s_summary = m.s_long_short_summary(s_monthly)

    # 8. 월별 시계열 넷 · 갈래 A 귀무 확률(permutation, M8 재료) --------------------
    monthly = monthly_series_frame(joined, universe_ew=universe_ew, spy=spy)
    perm_input = {col: monthly[col].to_list() for col in ("E", "E_ew", "I", "S")}
    permutation = m.sign_flip_permutation_probability(
        perm_input, n_perm=args.n_permutation, seed=args.permutation_seed
    )
    print(
        f"갈래 A 귀무 확률 (permutation {permutation['n_perm']}회, "
        f"{permutation['n_months']}개월): {permutation['probability_all_positive']:.4f}",
        flush=True,
    )

    # 9. closed_by_reason -----------------------------------------------------------
    closed_by_reason = closed_by_reason_counts(labels_holdout)

    # 10. 산출물 ----------------------------------------------------------------------
    run_id = f"m7_holdout_{snapshot_date.replace('-', '')}"
    out_dir = root.output / "model_runs" / run_id
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(pred_dir / "holdout.parquet")
    monthly.write_parquet(out_dir / "monthly_series.parquet")

    config = {
        "run_id": run_id,
        "adopted_model": "M-L Ridge (M5 §5 채택)",
        "alpha": ADOPTED_RIDGE_ALPHA,
        "feature_cols": model_features,
        "design_matrix": "rank([0,1], 결측 0.5) + _isna 플래그, 06 §2",
        "m3_manifest_path": str(train_inputs.m3_manifest_path),
        "model_input_features_source": "us_features_v1 manifest.model_input_features.all (M3)",
        "walk_forward": False,
        "retrain_count": 1,
        "train_window": {
            "dev_wall_start": DEV_START.isoformat(),
            "dev_wall_end": DEV_END.isoformat(),
            **_window_summary(train_inputs.core, train_dates),
        },
        "predict_window": {
            "holdout_wall_start": HOLDOUT_START.isoformat(),
            "holdout_wall_end": HOLDOUT_END.isoformat(),
            **_window_summary(holdout_core, holdout_dates),
        },
        "input_datasets": {
            "us_features_v1": {
                "content_hash": _dataset_manifest_field(root, "us_features_v1", "content_hash"),
                "train_row_count": train_inputs.features_row_count,
            },
            "us_labels_v1": {
                "content_hash": _dataset_manifest_field(root, "us_labels_v1", "content_hash"),
                "train_row_count": train_inputs.labels_row_count,
            },
        },
        "permutation": {"n_perm": args.n_permutation, "seed": args.permutation_seed},
        "modeler_git_commit": modeler_git_commit,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )

    payload = {
        "snapshot_date": snapshot_date,
        "branch": branch.to_dict(),
        "metrics": metrics_result.to_dict(),
        "s_long_short": s_summary,
        "permutation_null_probability": permutation,
        "closed_by_reason": closed_by_reason,
        "ew_minus_spy_mean": ew_minus_spy_mean,
        "train_window": config["train_window"],
        "predict_window": config["predict_window"],
        "modeler_git_commit": modeler_git_commit,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )

    print(f"\n산출물: {out_dir}", flush=True)
    print(f"총 소요 시간: {time.monotonic() - t0:.1f}초", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
