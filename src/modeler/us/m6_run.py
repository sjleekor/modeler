"""M6 — 비용·분해·CPCV 과적합 확률 (``06_execution_steps.md`` M6).

    uv run python -m modeler.us.m6_run

M4·M5가 저장한 OOF 예측(``m4_ridge_*``·``m4_enet_*``·``m4_ols3_*``·
``m4_lgbm_*``·``m4_lgbm_on_l1_*``·``m5_ensemble_*``)을 **다시 학습하지 않고**
읽어 ``05_validation_protocol.md`` §4 지표표, R2 분해(``mcap_rank`` decile ·
``adv_20d`` 3분위 · ``sic2`` · 연도), Y16 민감도표(Q×k · 스프레드 2배)를
낸다. **CPCV만 예외다** — 채택 설정(Ridge alpha=100) 하나를
``skfolio.CombinatorialPurgedCV``로 28개 경로에서 재학습해 PBO·DSR을 낸다
(``05`` §3이 "경로별 재학습이 필요하다"고 명시한 유일한 예외).

**비용·벤치마크 로직은 새로 짜지 않는다** — ``modeler.us.cost``·
``modeler.us.benchmark``·``modeler.us.metrics``를 그대로 쓴다.

**개발 구간만 읽는다.** ``build_m4_inputs``·``load_dev_frame``을 그대로
재사용하므로 holdout 날짜 벽(``modeler.us.scan.DEV_END``·
``assert_dev_window``)이 그대로 적용된다 — 이 파일 안에 holdout을 열
경로가 없다. **지표 구간은 OOF 검증 fold가 있는 2020-05~2025-04(60개월)
뿐이다** — 개발 구간 전체(2018-09~2025-06, 82개월)가 아니다. 첫 fold의
학습 구간(2018-09~2020-03)은 어느 모델도 검증받지 않았기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from skfolio.model_selection import CombinatorialPurgedCV

from modeler.etl.config import DataRoot
from modeler.etl.metrics import drawdown_stats, newey_west_tstat
from modeler.us import benchmark as bm
from modeler.us import cost as cost_mod
from modeler.us import metrics as m
from modeler.us.lake import UsLake
from modeler.us.m4_models import fit_predict_ridge
from modeler.us.m4_run import M4Inputs, build_m4_inputs
from modeler.us.m4_transform import cross_sectional_percentile, to_design_arrays
from modeler.us.m5_ensemble import _latest_run_dir  # 모델 id 접두어 충돌 방어 재사용 (2026-09-21)
from modeler.us.scan import HAC_LAG, MIN_NAMES, load_dev_frame, monthly_rank_ic

# --- 0. 상수 ------------------------------------------------------------------

#: 지표를 낼 세 필수 run — M6 지시 §1. 있으면 같이 낸다(``M4_OPTIONAL_RUN_IDS``).
REQUIRED_RUN_IDS: tuple[str, ...] = (
    "m4_ridge_20260921",
    "m4_lgbm_20260921",
    "m4_lgbm_on_l1_20260921",
)
OPTIONAL_RUN_IDS: tuple[str, ...] = ("m4_enet_20260921", "m5_ensemble_20260921")

#: 채택 모델 — M5 §5 "M-L Ridge alpha=100".
ADOPTED_RUN_ID = "m4_ridge_20260921"
ADOPTED_RIDGE_ALPHA = 100.0

#: CPCV — ``05`` §3.
CPCV_N_FOLDS = 8
CPCV_N_TEST_FOLDS = 2
#: purge=embargo=21거래일 ≈ 리밸런스 1회(월간 격자에서의 근사) —
#: ``m4_splits.py`` 모듈 docstring과 같은 이유(리밸런스가 월 1회라 h=21
#: 거래일 자체가 "그 달 하나"에 해당한다)로 월 단위 1을 쓴다. skfolio는
#: purge와 embargo를 따로 받으므로 여기서는 둘 다 명시적으로 지정한다.
CPCV_PURGED_SIZE = 1
CPCV_EMBARGO_SIZE = 1

#: DSR — N을 세는 다섯 model_id. Ridge 4 + ENet 12 + OLS-3 1 + M-G 24 + M-E 1
#: = 42(``06`` M6 지시가 준 값과 같다). ``m4_lgbm_on_l1``은 대조군(하이퍼
#: 파라미터 탐색이 아니라 M-G 설정 재사용)이라 시행 수에 안 넣는다.
DSR_TRIAL_MODEL_IDS: tuple[str, ...] = ("m4_ridge", "m4_enet", "m4_ols3", "m4_lgbm", "m5_ensemble")

#: R2 분해 — mcap_rank decile(1 최소~10 최대, ``02`` §2.1) · adv_20d 3분위 ·
#: sic2_bucket · 연도.
DECOMPOSITION_AXES: tuple[str, ...] = ("mcap_rank_bucket", "adv_tertile", "sic2_bucket", "year")


# --- 1. 유니버스 조립 ----------------------------------------------------------


def bucket_universe(lake: UsLake, labels_df: pl.DataFrame) -> pl.DataFrame:
    """라벨 프레임에 ``sigma_daily``와 R2 분해축 버킷을 붙인다 — 개발·holdout 공용.

    ``sigma_daily``는 라벨 데이터셋에 저장돼 있지 않다(``build_labels.py``가
    manifest의 비용 격자 계산에만 썼다) — ``cost.daily_volatility``로 다시
    조인한다(공식은 그대로 ``cost.py``의 것).

    ``labels_df``가 어느 날짜창(개발 구간 ``load_dev_frame`` 또는 holdout
    ``m7_run.load_holdout_frame``)에서 왔는지는 이 함수가 상관하지 않는다 —
    받은 프레임 그대로에 sigma·버킷만 붙인다(``m7_run``이 같은 버킷 정의를
    holdout에 재사용한다, ``06`` M7).
    """
    sigma = cost_mod.daily_volatility(lake).select("date", "symbol", "sigma_daily").collect()
    universe = labels_df.join(sigma, on=["date", "symbol"], how="left")
    universe = cross_sectional_percentile(universe, "adv_20d", date_col="date", out_col="_adv_pct")
    universe = universe.with_columns(
        pl.when(pl.col("_adv_pct") < 1.0 / 3.0)
        .then(pl.lit("1_저유동"))
        .when(pl.col("_adv_pct") < 2.0 / 3.0)
        .then(pl.lit("2_중유동"))
        .otherwise(pl.lit("3_고유동"))
        .alias("adv_tertile"),
        pl.col("mcap_rank")
        .cast(pl.Int64)
        .cast(pl.Utf8)
        .fill_null("결측(mcap_rank 없음)")
        .alias("mcap_rank_bucket"),
        pl.col("date").dt.year().alias("year"),
    ).drop("_adv_pct")
    return universe


def build_universe(root: DataRoot) -> tuple[UsLake, pl.DataFrame]:
    """``us_labels_v1``(개발 구간)에 ``sigma_daily``를 붙인 유니버스 프레임."""
    lake = UsLake.resolve()
    labels_dev = load_dev_frame(root, "us_labels_v1")
    return lake, bucket_universe(lake, labels_dev)


def load_oof(root: DataRoot, run_id: str) -> pl.DataFrame:
    pred_dir = root.output / "model_runs" / run_id / "predictions"
    parts = sorted(pred_dir.glob("fold_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{pred_dir}에 fold_*.parquet가 없습니다")
    frames = [pl.read_parquet(p) for p in parts]
    return pl.concat(frames).select("date", "symbol", "pred", "fold_id", "L2", "y_rank")


_JOIN_COLS = (
    "date",
    "symbol",
    "L0",
    "L1",
    "close",
    "adj_close",
    "mcap_rank",
    "mcap_rank_bucket",
    "adv_20d",
    "adv_tertile",
    "sic2_bucket",
    "price_ge_5",
    "sigma_daily",
    "year",
)


def joined_frame(oof: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    out = oof.join(universe.select(*_JOIN_COLS), on=["date", "symbol"], how="inner")
    if out.height != oof.height:
        raise ValueError(
            f"OOF({oof.height}행)와 유니버스 조인 후 행수({out.height})가 다릅니다 — "
            "라벨 데이터셋 스냅샷이 OOF를 만들 때와 달라졌을 수 있습니다"
        )
    return out


# --- 2. 05 §4 지표표 -----------------------------------------------------------


@dataclass
class ModelMetrics:
    run_id: str
    n_months: int
    rank_ic_mean: float
    rank_ic_t_hac: float
    S: float
    S_n_months: int
    E_gross: float
    E: float
    E_ew: float
    I: float  # noqa: E741 - 05 §4 지표 이름 그대로
    ew_minus_spy: float
    turnover_mean: float
    max_drawdown: float
    longest_drawdown_months: int
    breakeven_q_dollar: float | None
    hit_top: float
    mean_names_held: float
    sensitivity_qk: pl.DataFrame
    sensitivity_spread_2x_E: float
    mcap_bucket_distribution: pl.DataFrame
    decomposition: dict[str, pl.DataFrame]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_months": self.n_months,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_t_hac": self.rank_ic_t_hac,
            "S": self.S,
            "S_n_months": self.S_n_months,
            "E_gross": self.E_gross,
            "E": self.E,
            "E_ew": self.E_ew,
            "I": self.I,
            "ew_minus_spy": self.ew_minus_spy,
            "turnover_mean": self.turnover_mean,
            "max_drawdown": self.max_drawdown,
            "longest_drawdown_months": self.longest_drawdown_months,
            "breakeven_q_dollar": self.breakeven_q_dollar,
            "hit_top": self.hit_top,
            "mean_names_held": self.mean_names_held,
            "sensitivity_qk": self.sensitivity_qk.to_dicts(),
            "sensitivity_spread_2x_E": self.sensitivity_spread_2x_E,
            "mcap_bucket_distribution": self.mcap_bucket_distribution.to_dicts(),
            "decomposition": {k: v.to_dicts() for k, v in self.decomposition.items()},
        }


def _decompose_axis(joined: pl.DataFrame, picked: pl.DataFrame, *, axis_col: str) -> pl.DataFrame:
    """축 ``axis_col``의 버킷별: top-k 비중 · top-k 평균 L0 · 그 버킷 안 rank IC.

    IC는 ``(date, 버킷)``을 합성 그룹키로 묶어 그 횡단면(그 버킷에 속한
    이름들만)에서 스피어만 상관을 낸 뒤 버킷별로 월 평균한다 — "어느
    구간에서 스킬이 나오는가"를 보는 것이라 top-k 픽에 한정하지 않는다.
    """
    total_picks = picked.height
    composition = (
        picked.group_by(axis_col)
        .agg(pl.len().alias("n_picks"), pl.col("L0").mean().alias("mean_l0_topk"))
        .with_columns((pl.col("n_picks") / total_picks).alias("share_of_topk"))
    )
    key = pl.concat_str(
        [pl.col("date").cast(pl.Utf8), pl.lit("|"), pl.col(axis_col).cast(pl.Utf8)]
    ).alias("_bucket_date")
    tagged = joined.with_columns(key)
    ic_table = monthly_rank_ic(
        tagged, x_col="pred", y_col="L2", group_col="_bucket_date", min_names=MIN_NAMES
    )
    lookup = tagged.select("_bucket_date", axis_col).unique()
    ic_by_bucket = (
        ic_table.join(lookup, on="_bucket_date", how="left")
        .group_by(axis_col)
        .agg(pl.col("ic").mean().alias("rank_ic_mean"), pl.len().alias("n_months_with_ic"))
    )
    return composition.join(ic_by_bucket, on=axis_col, how="left").sort(axis_col)


def compute_model_metrics(
    run_id: str,
    joined: pl.DataFrame,
    *,
    universe_ew: pl.DataFrame,
    spy: pl.DataFrame,
    ew_minus_spy_mean: float,
) -> ModelMetrics:
    n_months = joined["date"].n_unique()
    ic_table = monthly_rank_ic(joined, x_col="pred", y_col="L2", group_col="date")
    ic_mean = float(ic_table["ic"].drop_nulls().mean()) if ic_table.height else float("nan")
    month_index = {d: i for i, d in enumerate(sorted(ic_table["date"].to_list()))}
    idx = np.array([month_index[d] for d in ic_table["date"].to_list()])
    t_hac = newey_west_tstat(ic_table["ic"].to_numpy(), idx, HAC_LAG)

    picked = m.topk_rows(joined, k=m.TOP_K)
    track = m.portfolio_track(joined)
    turnover_map = {row["date"]: row["turnover"] for row in track.iter_rows(named=True)}

    e_gross, _ = m.excess_over(
        track, spy, return_col="gross_return", benchmark_col="spy_h21_return"
    )
    e_net, _ = m.excess_over(track, spy, return_col="net_return", benchmark_col="spy_h21_return")
    e_ew_net, _ = m.excess_over(
        track, universe_ew, return_col="net_return", benchmark_col="ew_l0_h21_return"
    )

    joined_ge5 = joined.filter(pl.col("price_ge_5"))
    track_ge5 = m.portfolio_track(joined_ge5)
    i_net, _ = m.excess_over(
        track_ge5, spy, return_col="net_return", benchmark_col="spy_h21_return"
    )

    s_value, s_n = m.s_spread(joined)
    turnover_values = [v for v in turnover_map.values() if math.isfinite(v)]
    turnover_mean = float(np.mean(turnover_values)) if turnover_values else float("nan")
    dd = drawdown_stats(track["net_return"].to_list())
    hit = m.hit_rate(picked, value_col="L2")
    mean_names_held = float(track["n_held"].mean()) if track.height else float("nan")

    breakeven = m.breakeven_q_dollar(e_gross, picked, turnover_map)

    sensitivity = m.sensitivity_grid(
        picked,
        turnover_map,
        gross_return_by_date=track.select("date", "gross_return"),
        benchmark=spy,
        benchmark_col="spy_h21_return",
    )
    spread2x_cost = m.monthly_cost_drag(
        picked,
        turnover=turnover_map,
        q_dollar=cost_mod.DEFAULT_Q_DOLLAR,
        k=cost_mod.DEFAULT_K,
        spread_multiplier=m.SPREAD_SENSITIVITY_MULTIPLIER,
    )
    spread2x_net = (
        track.select("date", "gross_return")
        .join(spread2x_cost, on="date", how="left")
        .with_columns(
            (pl.col("gross_return") - pl.col("cost_drag").fill_null(0.0)).alias("net_return")
        )
    )
    e_spread2x, _ = m.excess_over(
        spread2x_net, spy, return_col="net_return", benchmark_col="spy_h21_return"
    )

    decomposition = {
        axis: _decompose_axis(joined, picked, axis_col=axis) for axis in DECOMPOSITION_AXES
    }
    mcap_dist = decomposition["mcap_rank_bucket"].select(
        "mcap_rank_bucket", "n_picks", "share_of_topk"
    )

    return ModelMetrics(
        run_id=run_id,
        n_months=n_months,
        rank_ic_mean=ic_mean,
        rank_ic_t_hac=t_hac,
        S=s_value,
        S_n_months=s_n,
        E_gross=e_gross,
        E=e_net,
        E_ew=e_ew_net,
        I=i_net,
        ew_minus_spy=ew_minus_spy_mean,
        turnover_mean=turnover_mean,
        max_drawdown=dd.max_drawdown,
        longest_drawdown_months=dd.longest_underwater,
        breakeven_q_dollar=breakeven,
        hit_top=hit,
        mean_names_held=mean_names_held,
        sensitivity_qk=sensitivity,
        sensitivity_spread_2x_E=e_spread2x,
        mcap_bucket_distribution=mcap_dist,
        decomposition=decomposition,
    )


# --- 3. CPCV -> PBO ------------------------------------------------------------


def run_cpcv(inputs: M4Inputs) -> dict:
    dates = inputs.dates
    n = len(dates)
    cv = CombinatorialPurgedCV(
        n_folds=CPCV_N_FOLDS,
        n_test_folds=CPCV_N_TEST_FOLDS,
        purged_size=CPCV_PURGED_SIZE,
        embargo_size=CPCV_EMBARGO_SIZE,
    )
    x_idx = np.arange(n).reshape(-1, 1)
    paths: list[dict] = []
    for split_id, (train_idx, test_idx_groups) in enumerate(cv.split(x_idx)):
        train_dates = [dates[i] for i in train_idx]
        test_dates = sorted({dates[i] for grp in test_idx_groups for i in grp})
        train_df = inputs.core.filter(pl.col("date").is_in(train_dates))
        test_df = inputs.core.filter(pl.col("date").is_in(test_dates))
        x_train, _ = to_design_arrays(train_df, inputs.model_features)
        y_train = train_df["y_rank"].to_numpy()
        x_test, _ = to_design_arrays(test_df, inputs.model_features)
        preds = fit_predict_ridge({"alpha": ADOPTED_RIDGE_ALPHA}, x_train, y_train, x_test)
        pred_df = test_df.select("date", "symbol", "L0").with_columns(pl.Series("pred", preds))
        picked = m.topk_rows(pred_df, k=m.TOP_K)
        monthly = (
            picked.group_by("date").agg(pl.col("L0").mean().alias("gross_return")).sort("date")
        )
        returns = monthly["gross_return"].to_list()
        paths.append(
            {
                "split_id": split_id,
                "n_train_months": len(train_dates),
                "n_test_months": len(test_dates),
                "test_date_min": min(test_dates).isoformat() if test_dates else None,
                "test_date_max": max(test_dates).isoformat() if test_dates else None,
                "sharpe_monthly": m.sharpe_ratio(returns, periods_per_year=None),
                "sharpe_annualized": m.sharpe_ratio(returns, periods_per_year=12),
            }
        )
    pbo = m.pbo_fraction_nonpositive([p["sharpe_monthly"] for p in paths])
    return {
        "n_folds": CPCV_N_FOLDS,
        "n_test_folds": CPCV_N_TEST_FOLDS,
        "purged_size_months": CPCV_PURGED_SIZE,
        "embargo_size_months": CPCV_EMBARGO_SIZE,
        "n_dev_months": n,
        "n_paths": len(paths),
        "pbo_fraction_sharpe_nonpositive": pbo,
        "paths": paths,
    }


# --- 4. DSR --------------------------------------------------------------------


def _gather_trials(root: DataRoot) -> dict[str, list[float]]:
    def _metrics(model_id: str) -> dict:
        run_dir = _latest_run_dir(root, model_id)
        if run_dir is None:
            raise FileNotFoundError(f"{model_id}_* 산출물이 없습니다 — M4/M5를 먼저 돌리십시오")
        return json.loads((run_dir / "metrics.json").read_text())

    trials: dict[str, list[float]] = {}
    for model_id in DSR_TRIAL_MODEL_IDS:
        payload = _metrics(model_id)
        grid = payload.get("grid_search")
        if grid:
            trials[model_id] = [g["mean_ic"] for g in grid]
        else:
            trials[model_id] = [payload["rank_ic_mean"]]
    return trials


def run_dsr(root: DataRoot, ridge_track: pl.DataFrame, ridge_ic_hat: float) -> dict:
    """DSR — 채택 모델(Ridge)의 실제 top-100 월수익으로 ``SR_hat``·왜도·첨도를
    재고, ``N``개 시행의 IC를 Sharpe 척도로 근사해 ``sr_var_across_trials``를
    낸다(``metrics.deflated_sharpe_ratio`` 참고).

    IC→Sharpe 근사는 Fundamental Law의 배도(breadth) 가정을 새로 들여오지
    않는다 — 대신 채택 모델 하나에서 **실측한** ``SR_hat_월간 / IC_hat``
    비율(``scale``)을 다른 41개 시행의 IC에 그대로 곱한다. 이 시행들이
    "IC-Sharpe 관계가 채택 모델과 같은 선형"이라는 근사를 깐다는 뜻이고,
    보고서에 이 가정을 적는다.

    ElasticNet 그리드 12개 중 2개는 계수가 완전히 죽어(``m4_models.py``
    docstring) fold IC가 전부 NaN이라 ``m4_run._safe_mean_ic``가 ``-inf``를
    남겼다 — 이 둘은 **시행 수 N에는 그대로 넣고**(실제로 시도했고 실패한
    것도 다중비교의 일부다), 분산 계산에서는 뺀다(``-inf``가 분산 자체를
    정의 불능으로 만든다).
    """
    returns = ridge_track["gross_return"].to_list()
    sr_hat_monthly = m.sharpe_ratio(returns, periods_per_year=None)
    sr_hat_annual = m.sharpe_ratio(returns, periods_per_year=12)
    skewness, kurt = m.sample_skew_kurtosis(returns)
    n_obs = len([r for r in returns if r is not None and math.isfinite(r)])

    trials = _gather_trials(root)
    all_ics = [ic for lst in trials.values() for ic in lst]
    n_trials = len(all_ics)
    finite_ics = [ic for ic in all_ics if math.isfinite(ic)]
    n_degenerate_trials = n_trials - len(finite_ics)
    scale = sr_hat_monthly / ridge_ic_hat if ridge_ic_hat else float("nan")
    trial_sharpes = [scale * ic for ic in finite_ics]
    sr_var = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else float("nan")

    dsr = m.deflated_sharpe_ratio(
        sr_hat_monthly,
        n_trials=n_trials,
        sr_var_across_trials=sr_var,
        skewness=skewness,
        kurtosis=kurt,
        n_obs=n_obs,
    )
    return {
        "sr_hat_monthly": sr_hat_monthly,
        "sr_hat_annualized": sr_hat_annual,
        "skewness": skewness,
        "kurtosis": kurt,
        "n_obs": n_obs,
        "n_trials": n_trials,
        "n_degenerate_trials_excluded_from_variance": n_degenerate_trials,
        "trial_counts_by_model": {k: len(v) for k, v in trials.items()},
        "ic_to_sharpe_scale": scale,
        "sr_var_across_trials": sr_var,
        **dsr,
    }


# --- 5. 출력 -------------------------------------------------------------------


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


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    args = parser.parse_args(argv)
    snapshot_date = args.snapshot_date or date.today().isoformat()

    t0 = time.monotonic()
    root = DataRoot.resolve(market="us")
    modeler_repo = Path(__file__).resolve().parents[3]

    lake, universe = build_universe(root)
    dev_dates = sorted(universe["date"].unique().to_list())
    universe_ew_all = bm.universe_equal_weight_monthly(universe)
    spy_all = bm.spy_monthly_return(lake, dev_dates)
    ew_minus_spy_all = universe_ew_all.join(spy_all, on="date", how="inner").with_columns(
        (pl.col("ew_l0_h21_return") - pl.col("spy_h21_return")).alias("ew_minus_spy")
    )

    run_ids = list(REQUIRED_RUN_IDS)
    for rid in OPTIONAL_RUN_IDS:
        if (root.output / "model_runs" / rid).is_dir():
            run_ids.append(rid)

    results: dict[str, ModelMetrics] = {}
    oof_dates_by_run: dict[str, list] = {}
    for run_id in run_ids:
        print(f"=== {run_id} ===", flush=True)
        oof = load_oof(root, run_id)
        joined = joined_frame(oof, universe)
        oof_dates = sorted(joined["date"].unique().to_list())
        oof_dates_by_run[run_id] = oof_dates
        ew_window = ew_minus_spy_all.filter(pl.col("date").is_in(oof_dates))
        ew_minus_spy_mean = (
            float(ew_window["ew_minus_spy"].mean()) if ew_window.height else float("nan")
        )
        metrics_result = compute_model_metrics(
            run_id,
            joined,
            universe_ew=universe_ew_all,
            spy=spy_all,
            ew_minus_spy_mean=ew_minus_spy_mean,
        )
        results[run_id] = metrics_result
        print(
            f"  n_months={metrics_result.n_months} "
            f"rank_ic={_fmt(metrics_result.rank_ic_mean)} "
            f"S={_fmt(metrics_result.S)} E_gross={_fmt(metrics_result.E_gross)} "
            f"E={_fmt(metrics_result.E)} E_ew={_fmt(metrics_result.E_ew)} "
            f"I={_fmt(metrics_result.I)} "
            f"turnover={_fmt(metrics_result.turnover_mean, 3)} "
            f"MDD={_fmt(metrics_result.max_drawdown)} "
            f"hit_top={_fmt(metrics_result.hit_top, 3)}",
            flush=True,
        )

    print("=== CPCV (Ridge alpha=100) ===", flush=True)
    inputs = build_m4_inputs(root)
    cpcv_result = run_cpcv(inputs)
    print(
        f"  n_paths={cpcv_result['n_paths']} "
        f"PBO(Sharpe<=0 비율)={_fmt(cpcv_result['pbo_fraction_sharpe_nonpositive'], 3)}",
        flush=True,
    )

    print("=== DSR ===", flush=True)
    adopted_track = m.portfolio_track(joined_frame(load_oof(root, ADOPTED_RUN_ID), universe))
    ridge_ic_hat = results[ADOPTED_RUN_ID].rank_ic_mean
    dsr_result = run_dsr(root, adopted_track, ridge_ic_hat)
    print(
        f"  N={dsr_result['n_trials']} SR_hat(월)={_fmt(dsr_result['sr_hat_monthly'])} "
        f"DSR={_fmt(dsr_result['dsr'], 4)}",
        flush=True,
    )

    if cpcv_result["pbo_fraction_sharpe_nonpositive"] > 0.5:
        print(
            f"\n*** PBO {cpcv_result['pbo_fraction_sharpe_nonpositive']:.3f} > 0.5"
            f" — 보고서 첫 줄에 적을 것 (05 §3) ***\n",
            flush=True,
        )

    out_dir = root.output / "m6_metrics" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_date": snapshot_date,
        "modeler_git_commit": _git_commit(modeler_repo),
        "oof_window": {
            run_id: {"n_months": len(d), "start": d[0].isoformat(), "end": d[-1].isoformat()}
            for run_id, d in oof_dates_by_run.items()
        },
        "models": {run_id: r.to_dict() for run_id, r in results.items()},
        "cpcv": cpcv_result,
        "dsr": dsr_result,
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
