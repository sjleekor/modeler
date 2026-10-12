"""US 층 3 진단 — 탐색 모델(Ridge·LightGBM)이 각 피쳐에 어떻게 반응하나.

계획: ``my/milestones/common/20261012_feature_explainer/08_layer3_plan.md`` §3·§4.
**진단이지 판정이 아니다.** 모델·피쳐를 고르는 근거로 쓰지 않는다.

- 입력은 동결 ``us_features_v2`` + ``us_features_flow_v1``의 월초 12개 횡단면
  (2025-07-01~2026-06-01)이다. **라벨 데이터셋은 읽지 않는다**
  (``exploratory_run.load_inputs``도 부르지 않는다 — 그 함수가 라벨을 연다).
- 재학습하지 않는다. 저장된 모델 두 개를 그대로 쓴다.
- LightGBM: ``predict(pred_contrib=True)``(TreeSHAP, path-dependent) + 직접 격자 PDP/ICE.
- Ridge: 정확한 선형 기여 ``coef_j * (x_j - 표본 평균_j)``.
- 모양 분류·안정성 규칙(§4)은 kr.py와 같은 상수로 따로 구현한다(서로 import하지 않는다).

실행::

    python -m modeler.analysis.feature_response.us run [--out-root DIR] [--dates-limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.stats import rankdata

from modeler.etl.config import DataRoot
from modeler.us.exploratory_run import FEATURES, prepare_features
from modeler.us.m4_transform import to_design_arrays

# ---------------------------------------------------------------- 고정 상수 (계획 §3·§4)

MODEL_RUN_TAG = "us_exploratory_20260929_r1"
#: us_serving.md:23 의 원본 모델 sha256. 다르면 멈춘다.
EXPECTED_MODEL_SHA256 = {
    "lightgbm": "02c14a2a008743536160bfc6dc059e3208bc38e5ca9646d4aff4bdbfb613c2eb",
    "ridge": "06288b14fd5f96172897c947bfdd9cf4ac50901f4a42366df365fc87d355332e",
}
#: 라벨이 없는 두 피쳐 데이터셋만 읽는다.
FEATURE_DATASETS = ("us_features_v2", "us_features_flow_v1")
SAMPLE_START = date(2025, 7, 1)
SAMPLE_END = date(2026, 6, 1)
EXPECTED_N_DATES = 12
FIRST_HALF_LAST_DATE = date(2025, 12, 1)  # 앞 절반 = 2025-07-01~2025-12-01
SECOND_HALF_FIRST_DATE = date(2026, 1, 2)  # 뒤 절반 = 2026-01-02~2026-06-01

RHO_THRESHOLD = 0.7
N_BOOTSTRAP = 200
SEED = 20261012
N_GRID = 21
GRID = np.linspace(0.0, 1.0, N_GRID)
N_ICE = 1000
PDP_THREADS = 8

FLAT_FRACTION = 0.25  # 평평: 폭 < 전 피쳐 PDP 폭 중앙값의 25%
MONOTONE_SHARE = 0.80  # 단조: 인접 차이 부호가 80% 이상 같음
TOP_K = 10  # 안정: 두 절반 모두 상위 10 또는 두 절반 모두 하위 절반

SHAPE_FLAT = "flat"
SHAPE_UP = "monotone_increasing"
SHAPE_DOWN = "monotone_decreasing"
SHAPE_U = "U"
SHAPE_INV_U = "inverted_U"
SHAPE_OTHER = "기타"

HALVES = ("all", "first", "second")
COMPONENTS = ("rank", "isna", "combined")


# ---------------------------------------------------------------- 순수 함수 (시험 대상)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 상관. 한쪽이 상수면 nan."""
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = rankdata(a)
    rb = rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def classify_shape(curve: Sequence[float], median_range: float) -> str:
    """PDP 곡선 모양 (계획 §4.4). ``median_range``는 그 모델 전 피쳐 PDP 폭의 중앙값."""
    y = np.asarray(curve, dtype=float)
    if y.size < 3 or not np.all(np.isfinite(y)):
        return SHAPE_OTHER
    if float(y.max() - y.min()) < FLAT_FRACTION * median_range:
        return SHAPE_FLAT
    d = np.diff(y)
    if (d > 0).mean() >= MONOTONE_SHARE:
        return SHAPE_UP
    if (d < 0).mean() >= MONOTONE_SHARE:
        return SHAPE_DOWN
    i_min, i_max = int(np.argmin(y)), int(np.argmax(y))
    last = y.size - 1
    if 0 < i_min < last and y[0] > y[i_min] and y[last] > y[i_min]:
        return SHAPE_U
    if 0 < i_max < last and y[0] < y[i_max] and y[last] < y[i_max]:
        return SHAPE_INV_U
    return SHAPE_OTHER


def rank_desc(values: np.ndarray) -> np.ndarray:
    """큰 값이 1위. 동률은 ``min`` 순위."""
    return rankdata(-np.asarray(values, dtype=float), method="min").astype(int)


def is_stable(
    rank_first: int, rank_second: int, n_features: int, dir_first: float, dir_second: float
) -> tuple[bool, str]:
    """안정성 (계획 §4.1).

    (두 절반 모두 상위 10 또는 두 절반 모두 하위 절반) **그리고** 방향 부호가 같다.
    """
    top_both = rank_first <= TOP_K and rank_second <= TOP_K
    bottom_both = rank_first > n_features / 2 and rank_second > n_features / 2
    if not (top_both or bottom_both):
        return False, "크기 순위가 두 절반에서 일관되지 않음"
    if not (np.isfinite(dir_first) and np.isfinite(dir_second)):
        return False, "방향을 계산할 수 없음"
    if np.sign(dir_first) == 0 or np.sign(dir_first) != np.sign(dir_second):
        return False, "방향 부호가 두 절반에서 다름"
    return True, "상위 10 두 절반" if top_both else "하위 절반 두 절반"


def union_find_groups(names: Sequence[str], rho: np.ndarray, threshold: float) -> list[list[str]]:
    """|rho| >= threshold 이면 같은 묶음 (union-find). nan은 무시. 이름 순서 유지."""
    n = len(names)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            r = rho[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return list(groups.values())


def daily_mean_spearman(x_rank: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """날짜별 열 간 Spearman을 구해 날짜 평균(nanmean)을 낸다. ``x_rank``는 (행, 피쳐)."""
    uniq = np.unique(dates)
    mats = []
    for d in uniq:
        block = x_rank[dates == d]
        ranks = np.apply_along_axis(rankdata, 0, block)
        with np.errstate(invalid="ignore", divide="ignore"):
            mats.append(np.corrcoef(ranks, rowvar=False))
    stack = np.stack(mats)
    with np.errstate(all="ignore"):
        out = np.nanmean(stack, axis=0)
    np.fill_diagonal(out, 1.0)
    return out


def bootstrap_mean_abs(
    abs_by_date_sum: np.ndarray,
    n_by_date: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOTSTRAP,
) -> tuple[np.ndarray, np.ndarray]:
    """날짜를 복원 추출해 평균 |기여|의 95% 구간. ``abs_by_date_sum``은 (날짜, 피쳐)."""
    n_dates = abs_by_date_sum.shape[0]
    idx = rng.integers(0, n_dates, size=(n_boot, n_dates))
    boot = abs_by_date_sum[idx].sum(axis=1) / n_by_date[idx].sum(axis=1)[:, None]
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def lgbm_contrib(
    model, x: np.ndarray, threads: int = PDP_THREADS
) -> tuple[np.ndarray, float, float]:
    """LightGBM TreeSHAP 기여(행, 열)와 편향, 가산성 최대 오차 (sum == raw predict)."""
    booster = model.booster_
    raw = booster.predict(x, pred_contrib=True, num_threads=threads)
    contrib, bias = raw[:, :-1], raw[:, -1]
    pred = booster.predict(x, raw_score=True, num_threads=threads)
    err = float(np.max(np.abs(contrib.sum(axis=1) + bias - pred)))
    return contrib, float(bias[0]), err


def ridge_contrib(model, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ridge 정확한 선형 기여 ``coef_j*(x_j - mean_j)``. 기준값(=평균 점수)과 가산성 오차."""
    coef = np.asarray(model.coef_, dtype=float).ravel()
    mean = x.mean(axis=0)
    contrib = (x - mean) * coef
    base = float(model.intercept_) + float(coef @ mean)
    err = float(np.max(np.abs(contrib.sum(axis=1) + base - model.predict(x))))
    return contrib, base, err


def split_components(contrib: np.ndarray, n_feat: int) -> dict[str, np.ndarray]:
    """열 순서(rank n개 → isna n개)에서 rank·isna·피쳐별 합계 기여를 나눈다."""
    rank_c, isna_c = contrib[:, :n_feat], contrib[:, n_feat:]
    return {"rank": rank_c, "isna": isna_c, "combined": rank_c + isna_c}


def make_groups(rho: np.ndarray, features: Sequence[str]) -> list[list[str]]:
    return union_find_groups(list(features), rho, RHO_THRESHOLD)


def pdp_sweep(predict, x: np.ndarray, j: int, n_feat: int, grid: np.ndarray = GRID) -> np.ndarray:
    """피쳐 j의 rank 열을 격자값으로, isna 열을 0으로 바꿔 행별 예측 (격자, 행)을 구한다."""
    out = np.empty((grid.size, x.shape[0]))
    saved_rank, saved_isna = x[:, j].copy(), x[:, n_feat + j].copy()
    x[:, n_feat + j] = 0.0
    try:
        for k, g in enumerate(grid):
            x[:, j] = g
            out[k] = predict(x)
    finally:
        x[:, j], x[:, n_feat + j] = saved_rank, saved_isna
    return out


# ---------------------------------------------------------------- 입출력


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _forbid_label_path(path: Path) -> None:
    if "labels" in str(path).lower():
        raise RuntimeError(f"라벨 경로는 읽지 않습니다: {path}")


def load_model(models_dir: Path, name: str) -> tuple[dict, str]:
    path = models_dir / name / "model.joblib"
    _forbid_label_path(path)
    digest = sha256_file(path)
    if digest != EXPECTED_MODEL_SHA256[name]:
        raise RuntimeError(f"{name} 모델 sha256 불일치: {digest}")
    bundle = joblib.load(path)
    if list(bundle["features"]) != list(FEATURES):
        raise RuntimeError(f"{name} 모델 features가 exploratory_run.FEATURES와 다릅니다")
    return bundle, digest


def load_feature_rows(
    datasets_dir: Path, features: Sequence[str], dates_limit: int | None
) -> tuple[pl.DataFrame, list[date], dict]:
    """라벨 없이 두 피쳐 데이터셋을 이어 월초 12개 횡단면의 rank·isna 열을 만든다."""
    paths = {n: datasets_dir / n / "part.parquet" for n in FEATURE_DATASETS}
    for p in paths.values():
        _forbid_label_path(p)
    in_window = pl.col("date").is_between(SAMPLE_START, SAMPLE_END)
    base_all = pl.scan_parquet(paths[FEATURE_DATASETS[0]])
    flow_all = pl.scan_parquet(paths[FEATURE_DATASETS[1]])
    base_cols = [c for c in features if c in base_all.collect_schema().names()]
    flow_cols = [c for c in features if c not in base_cols]
    base = base_all.filter(in_window).select("date", "symbol", "price_ge_5", *base_cols).collect()
    flow = flow_all.filter(in_window).select("date", "symbol", *flow_cols).collect()
    raw = base.join(flow, on=["date", "symbol"], how="left", validate="1:1")
    if raw.height != flow.height:
        raise ValueError("피쳐 데이터셋 키가 다릅니다")
    dates = sorted(raw["date"].unique().to_list())
    if dates_limit is None:
        if len(dates) != EXPECTED_N_DATES:
            raise AssertionError(f"날짜 {len(dates)}개 (기대 {EXPECTED_N_DATES}): {dates}")
        if not (dates[0] >= SAMPLE_START and dates[-1] < date(2026, 7, 1)):
            raise AssertionError("표본 구간을 벗어난 날짜")
        if dates[5] > FIRST_HALF_LAST_DATE or dates[6] < SECOND_HALF_FIRST_DATE:
            raise AssertionError(f"절반 경계가 계획과 다릅니다: {dates}")
    else:
        # 스모크: 앞 N/2 + 뒤 N/2 개를 골라 두 절반 모두 채운다.
        half = max(1, dates_limit // 2)
        dates = dates[:half] + dates[-(dates_limit - half) :] if dates_limit > 1 else dates[:1]
        raw = raw.filter(pl.col("date").is_in(dates))
    prepared = prepare_features(raw, tuple(features))
    info = {
        "input_part_sha256": {n: sha256_file(p) for n, p in paths.items()},
        "input_manifest_sha256": {
            n: sha256_file(datasets_dir / n / "manifest.json") for n in FEATURE_DATASETS
        },
        "input_paths": {n: str(p) for n, p in paths.items()},
    }
    return prepared.sort("date", "symbol"), dates, info


def package_versions() -> dict:
    out = {}
    for pkg in ("numpy", "scikit-learn", "lightgbm", "polars", "scipy", "shap"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def git_state(path: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    try:
        return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"head": None, "dirty": None}


# ---------------------------------------------------------------- 분석 본체


def summarize_model(
    name: str,
    contrib: np.ndarray,
    x: np.ndarray,
    dates: np.ndarray,
    features: Sequence[str],
    half_masks: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """모델 하나의 피쳐 × 성분(rank·isna·combined) 요약 행."""
    n_feat = len(features)
    comps = split_components(contrib, n_feat)
    uniq_dates = np.unique(dates)
    date_idx = {d: i for i, d in enumerate(uniq_dates)}
    di = np.array([date_idx[d] for d in dates])
    n_by_date = np.bincount(di, minlength=len(uniq_dates)).astype(float)
    rows: list[dict] = []
    for comp in COMPONENTS:
        c = comps[comp]
        abs_c = np.abs(c)
        mean_abs = {h: abs_c[m].mean(axis=0) for h, m in half_masks.items()}
        ranks = {h: rank_desc(v) for h, v in mean_abs.items()}
        sums = np.zeros((len(uniq_dates), n_feat))
        np.add.at(sums, di, abs_c)
        lo, hi = bootstrap_mean_abs(sums, n_by_date, rng)
        for j, feat in enumerate(features):
            xj = x[:, j]  # 방향의 기준은 항상 그 피쳐의 rank 값
            xv = xj if comp != "isna" else x[:, n_feat + j]
            direction = {h: spearman(xv[m], c[m, j]) for h, m in half_masks.items()}
            stable, reason = (None, None)
            if comp == "combined":
                stable, reason = is_stable(
                    int(ranks["first"][j]),
                    int(ranks["second"][j]),
                    n_feat,
                    direction["first"],
                    direction["second"],
                )
            rows.append(
                {
                    "model": name,
                    "feature": feat,
                    "component": comp,
                    "mean_abs": float(mean_abs["all"][j]),
                    "mean_abs_first": float(mean_abs["first"][j]),
                    "mean_abs_second": float(mean_abs["second"][j]),
                    "ci95_lo": float(lo[j]),
                    "ci95_hi": float(hi[j]),
                    "rank_all": int(ranks["all"][j]),
                    "rank_first": int(ranks["first"][j]),
                    "rank_second": int(ranks["second"][j]),
                    "direction": direction["all"],
                    "direction_first": direction["first"],
                    "direction_second": direction["second"],
                    "stable": stable,
                    "stability_reason": reason,
                    "expected_sign": "",  # 층 2 카드 기대 부호 — 메인 세션이 채운다
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def group_report(rho: np.ndarray, features: Sequence[str], contribs: dict[str, np.ndarray]) -> dict:
    n_feat = len(features)
    groups = make_groups(rho, features)
    iu = np.triu_indices(n_feat, k=1)
    pair_rho = rho[iu]
    finite = pair_rho[np.isfinite(pair_rho)]
    pairs = [
        {"a": features[i], "b": features[j], "rho": float(rho[i, j])}
        for i, j in zip(*iu, strict=True)
        if np.isfinite(rho[i, j]) and abs(rho[i, j]) >= RHO_THRESHOLD
    ]
    out_groups = []
    for members in groups:
        if len(members) < 2:
            continue
        idx = [features.index(m) for m in members]
        entry = {"members": members, "mean_abs_group_sum": {}, "sum_member_mean_abs": {}}
        for model_name, contrib in contribs.items():
            comb = split_components(contrib, n_feat)["combined"][:, idx]
            entry["mean_abs_group_sum"][model_name] = float(np.abs(comb.sum(axis=1)).mean())
            entry["sum_member_mean_abs"][model_name] = float(np.abs(comb).mean(axis=0).sum())
        out_groups.append(entry)
    return {
        "threshold": RHO_THRESHOLD,
        "method": "날짜별 Spearman(rank 열 53개)을 12개 날짜 평균, |rho|>=0.7 union-find",
        "n_features": n_feat,
        "n_singletons": sum(1 for g in groups if len(g) == 1),
        "groups": out_groups,
        "rho_summary": {
            "n_pairs": int(pair_rho.size),
            "n_finite": int(finite.size),
            "n_pairs_ge_threshold": len(pairs),
            "max_abs": float(np.abs(finite).max()) if finite.size else None,
            "mean_abs": float(np.abs(finite).mean()) if finite.size else None,
            "quantiles_abs": (
                {str(q): float(np.quantile(np.abs(finite), q)) for q in (0.5, 0.9, 0.99)}
                if finite.size
                else {}
            ),
        },
        "pairs_ge_threshold": pairs,
    }


def run_pdp(
    model,
    x: np.ndarray,
    features: Sequence[str],
    half_masks: dict[str, np.ndarray],
    ice_rows: np.ndarray,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, str]], dict[str, float]]:
    n_feat = len(features)
    booster = model.booster_

    def predict(a: np.ndarray) -> np.ndarray:
        return booster.predict(a, raw_score=True, num_threads=PDP_THREADS)

    curves: dict[str, dict[str, np.ndarray]] = {h: {} for h in HALVES}
    pdp_rows, ice_parts = [], []
    for j, feat in enumerate(features):
        preds = pdp_sweep(predict, x, j, n_feat)  # (격자, 행)
        for h, m in half_masks.items():
            curve = preds[:, m].mean(axis=1)
            curves[h][feat] = curve
            for k, g in enumerate(GRID):
                pdp_rows.append(
                    {
                        "model": "lightgbm",
                        "feature": feat,
                        "half": h,
                        "grid": float(g),
                        "pdp": float(curve[k]),
                    }
                )
        sub = preds[:, ice_rows]
        ice_parts.append(
            pl.DataFrame(
                {
                    "model": "lightgbm",
                    "feature": feat,
                    "row": np.tile(ice_rows, N_GRID).astype(np.int64),
                    "grid": np.repeat(GRID, ice_rows.size),
                    "ice": sub.ravel(),
                }
            )
        )
    shapes: dict[str, dict[str, str]] = {}
    ranges: dict[str, float] = {}
    for h in HALVES:
        widths = {f: float(np.ptp(curves[h][f])) for f in features}
        med = float(np.median(list(widths.values())))
        for f in features:
            shapes.setdefault(f, {})[h] = classify_shape(curves[h][f], med)
            if h == "all":
                ranges[f] = widths[f]
    return pl.DataFrame(pdp_rows), pl.concat(ice_parts), shapes, ranges


def write_report(
    out: Path, summary: pl.DataFrame, groups: dict, manifest: dict, shapes: dict
) -> None:
    lines = [
        "# US 층 3 — 모델 반응 자동 요약",
        "",
        f"- run_id: `{manifest['run_id']}` · 표본 날짜 {len(manifest['dates'])}개 · "
        f"행 {manifest['n_rows']:,} · labels_read: {manifest['labels_read']}",
        "- 진단이지 판정이 아닙니다. '수익에 기여한다'는 말은 쓰지 않습니다.",
        f"- 가산성 최대 오차: {manifest['additivity_max_error']}",
        "",
    ]
    for model_name in ("lightgbm", "ridge"):
        comb = summary.filter((pl.col("model") == model_name) & (pl.col("component") == "combined"))
        n_stable = int(comb["stable"].sum())
        lines += [
            f"## {model_name} — 피쳐별 평균 |기여| 상위 15 (rank+isna 합산)",
            "",
            f"안정 {n_stable}개 / {comb.height}개",
            "",
            "| 순위 | 피쳐 | 평균 \\|기여\\| | 95% 구간 | 방향(ρ) | 모양 | 안정 | 기대 부호 |",
            "|---:|---|---:|---|---:|---|---|---|",
        ]
        for r in comb.sort("rank_all").head(15).iter_rows(named=True):
            shape = (
                shapes.get(r["feature"], {}).get("all", "") if model_name == "lightgbm" else "선형"
            )
            lines.append(
                f"| {r['rank_all']} | `{r['feature']}` | {r['mean_abs']:.5f} | "
                f"{r['ci95_lo']:.5f}\\~{r['ci95_hi']:.5f} | {r['direction']:+.2f} | {shape} | "
                f"{'안정' if r['stable'] else '불안정'} | {r['expected_sign']} |"
            )
        lines.append("")
    lines += ["## 상관 묶음 (|ρ| ≥ 0.7)", ""]
    if not groups["groups"]:
        lines.append("묶인 피쳐가 없습니다.")
    for g in groups["groups"]:
        lines.append(
            f"- {', '.join(f'`{m}`' for m in g['members'])} — 묶음 합계 평균|기여| "
            + ", ".join(f"{k}: {v:.5f}" for k, v in g["mean_abs_group_sum"].items())
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(out_root: Path | None, dates_limit: int | None) -> Path:
    t_start = time.monotonic()
    timings: dict[str, float] = {}
    now = datetime.now(UTC)
    run_id = f"us_l3_{now.strftime('%Y%m%dT%H%M%SZ')}"
    if out_root is None:
        base = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
        us_root = DataRoot(base=base / "us")
        out_base = us_root.output / "feature_response_l3"
    else:
        us_root = DataRoot(base=Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data")) / "us")
        out_base = out_root
    out = out_base / run_id
    out.mkdir(parents=True, exist_ok=False)

    models_dir = us_root.output / "exploratory_models" / MODEL_RUN_TAG
    bundles, model_sha = {}, {}
    for name in ("lightgbm", "ridge"):
        bundles[name], model_sha[name] = load_model(models_dir, name)
        print(f"model {name} sha256 {model_sha[name]} OK", flush=True)
    features = list(bundles["lightgbm"]["features"])
    n_feat = len(features)

    t = time.monotonic()
    frame, dates, input_info = load_feature_rows(us_root.datasets, features, dates_limit)
    x, cols = to_design_arrays(frame, features)
    expected_cols = [f"{f}_rank" for f in features] + [f"{f}_isna" for f in features]
    if cols != expected_cols or x.shape[1] != 2 * n_feat:
        raise AssertionError("설계행렬 열 순서가 모델 features 순서와 다릅니다")
    if not np.isfinite(x).all():
        raise AssertionError("설계행렬에 비유한 값이 있습니다")
    row_dates = np.array(frame["date"].to_list())
    mid = len(dates) // 2
    half_masks = {
        "all": np.ones(len(row_dates), dtype=bool),
        "first": np.isin(row_dates, dates[:mid]),
        "second": np.isin(row_dates, dates[mid:]),
    }
    timings["load_s"] = time.monotonic() - t
    print(f"rows {x.shape[0]:,} dates {len(dates)}", flush=True)

    rng = np.random.default_rng(SEED)
    t = time.monotonic()
    lgb_c, lgb_bias, lgb_err = lgbm_contrib(bundles["lightgbm"]["model"], x)
    rdg_c, rdg_base, rdg_err = ridge_contrib(bundles["ridge"]["model"], x)
    timings["contrib_s"] = time.monotonic() - t
    print(f"additivity max err lightgbm {lgb_err:.3e} ridge {rdg_err:.3e}", flush=True)

    t = time.monotonic()
    summary = pl.concat(
        [
            summarize_model("lightgbm", lgb_c, x, row_dates, features, half_masks, rng),
            summarize_model("ridge", rdg_c, x, row_dates, features, half_masks, rng),
        ]
    )
    timings["summary_s"] = time.monotonic() - t

    t = time.monotonic()
    rho = daily_mean_spearman(x[:, :n_feat], row_dates)
    groups = group_report(rho, features, {"lightgbm": lgb_c, "ridge": rdg_c})
    timings["groups_s"] = time.monotonic() - t

    t = time.monotonic()
    ice_rows = np.sort(rng.choice(x.shape[0], size=min(N_ICE, x.shape[0]), replace=False))
    pdp, ice, shapes, pdp_range = run_pdp(
        bundles["lightgbm"]["model"], x, features, half_masks, ice_rows
    )
    timings["pdp_s"] = time.monotonic() - t

    # 모양 열을 요약표에 붙인다 (LightGBM만 · rank PDP 기준, Ridge는 선형이라 비움)
    shape_rows = pl.DataFrame(
        {
            "model": "lightgbm",
            "feature": features,
            "shape_all": [shapes[f]["all"] for f in features],
            "shape_first": [shapes[f]["first"] for f in features],
            "shape_second": [shapes[f]["second"] for f in features],
            "pdp_range": [pdp_range[f] for f in features],
        }
    )
    summary = summary.join(shape_rows, on=["model", "feature"], how="left")
    group_of = {m: i for i, g in enumerate(groups["groups"]) for m in g["members"]}
    summary = summary.with_columns(
        pl.col("feature")
        .replace_strict(group_of, default=None, return_dtype=pl.Int64)
        .alias("group_id")
    )

    ridge_model = bundles["ridge"]["model"]
    coef = np.asarray(ridge_model.coef_, dtype=float).ravel()
    ridge_coef = pl.DataFrame(
        {
            "feature": features,
            "rank_coef": coef[:n_feat],
            "isna_coef": coef[n_feat:],
            "intercept": float(ridge_model.intercept_),
        }
    )

    summary.write_parquet(out / "shap_summary.parquet")
    pdp.write_parquet(out / "pdp.parquet")
    ice.with_columns(
        pl.col("row")
        .replace_strict({int(r): i for i, r in enumerate(ice_rows)}, return_dtype=pl.Int64)
        .alias("sample_idx")
    ).write_parquet(out / "ice_sample.parquet")
    ridge_coef.write_parquet(out / "ridge_coef.parquet")
    (out / "groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n")

    timings["total_s"] = time.monotonic() - t_start
    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "kind": "us_layer3_feature_response",
        "plan": "my/milestones/common/20261012_feature_explainer/08_layer3_plan.md",
        "labels_read": False,
        "model_run_tag": MODEL_RUN_TAG,
        "model_sha256": model_sha,
        "model_sha256_expected": EXPECTED_MODEL_SHA256,
        "model_train_end": {k: str(v["train_end"]) for k, v in bundles.items()},
        "model_max_training_terminal": {
            k: str(v["max_training_terminal"]) for k, v in bundles.items()
        },
        **input_info,
        "dates": [d.isoformat() for d in dates],
        "first_half_dates": [d.isoformat() for d in dates[: len(dates) // 2]],
        "second_half_dates": [d.isoformat() for d in dates[len(dates) // 2 :]],
        "dates_limit": dates_limit,
        "n_rows": int(x.shape[0]),
        "n_features": n_feat,
        "design_columns": 2 * n_feat,
        "shap_method": "lightgbm pred_contrib (tree_path_dependent)",
        "ridge_method": "exact linear contribution coef_j*(x_j - sample mean_j), sample = all rows",
        "ridge_base_value": rdg_base,
        "lightgbm_base_value": lgb_bias,
        "additivity_max_error": {"lightgbm": lgb_err, "ridge": rdg_err},
        "pdp": {"grid": GRID.tolist(), "isna_fixed": 0, "n_ice": int(ice_rows.size)},
        "seeds": {"numpy_rng": SEED, "n_bootstrap": N_BOOTSTRAP},
        "rho_threshold": RHO_THRESHOLD,
        "modeler": git_state(Path(__file__).resolve().parents[4]),
        "package_versions": package_versions(),
        "python": sys.version.split()[0],
        "timings_s": timings,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    write_report(out, summary, groups, manifest, shapes)
    (out / ".done").write_text(datetime.now(UTC).isoformat() + "\n")
    print(f"done -> {out}  ({timings['total_s']:.1f}s)", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--dates-limit", type=int, default=None)
    args = parser.parse_args(argv)
    run(args.out_root, args.dates_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
