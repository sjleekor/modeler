"""OOS 평가 (사양 02 §5.3).

불확실성은 **날짜 블록** 부트스트랩이다. 결정 세션을 ``block_sessions``개씩 연속 블록으로
묶고, 블록을 통째로(같은 시기의 모든 자산 함께) 복원추출한다. 통계는 블록×자산의 합계 행렬로
미리 만들어 뒀다가 뽑힌 횟수를 곱해 다시 합친다 -> 블록을 쪼갤 수 없다.

* Opportunity: 자산별 squared-error skill(1 - SSE_model/SSE_baseline)을 자산 평균, 자산별 시계열
  Spearman IC. 섹터 상대 선택은 같은 날짜 순위 IC와 상위-하위 시장 대비 수익(decile 없음).
* Stability: 자산별 Brier skill(대 baseline b 주, a 보조)을 자산 평균, reliability, PR-AUC.
* 시장별·자산별·연도별 표, 사건률 높은 해를 하나씩 뺀 민감도.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from modeler.scores.market_sector.config import MsConfig


# --------------------------------------------------------------------------- bootstrap
def block_ids(sessions: np.ndarray, block_len: int) -> np.ndarray:
    """행마다 블록 번호. 고유 결정 세션을 정렬해 ``block_len``개씩 묶는다."""
    uniq = np.unique(sessions)
    return np.searchsorted(uniq, sessions) // block_len


def block_matrix(
    block: np.ndarray, asset_idx: np.ndarray, values: np.ndarray, n_assets: int
) -> np.ndarray:
    """``[n_blocks, n_assets, k]`` 합계 행렬."""
    nb = int(block.max()) + 1 if len(block) else 0
    m = np.zeros((nb, n_assets, values.shape[1]))
    np.add.at(m, (block, asset_idx), values)
    return m


def draw_counts(n_blocks: int, n_resamples: int, seed: int) -> np.ndarray:
    """``[R, n_blocks]``: 각 재표본에서 블록이 뽑힌 횟수(블록 단위 복원추출)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    return np.stack([np.bincount(d, minlength=n_blocks) for d in draws])


def bootstrap_ci(
    m: np.ndarray, stat_fn, n_resamples: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    """``stat_fn(tot[n_assets, k]) -> float``. 점추정·백분위 CI·0 초과 비율."""
    nb = m.shape[0]
    point = stat_fn(m.sum(axis=0))
    if nb < 2:
        return {"point": point, "lo": None, "hi": None, "frac_gt0": None, "n_blocks": nb}
    counts = draw_counts(nb, n_resamples, seed)
    tot = np.einsum("rb,bak->rak", counts, m)
    stats = np.array([stat_fn(t) for t in tot])
    stats = stats[~np.isnan(stats)]
    a = (1 - level) / 2
    return {
        "point": point,
        "lo": float(np.quantile(stats, a)) if len(stats) else None,
        "hi": float(np.quantile(stats, 1 - a)) if len(stats) else None,
        "frac_gt0": float((stats > 0).mean()) if len(stats) else None,
        "n_blocks": nb,
        "n_resamples": int(len(stats)),
    }


def mean_asset_skill(tot: np.ndarray) -> float:
    """``tot[:, 0]``=SSE(model), ``[:, 1]``=SSE(base), ``[:, 2]``=n. 자산 평균 skill."""
    ok = (tot[:, 2] > 0) & (tot[:, 1] > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(1.0 - tot[ok, 0] / tot[ok, 1]))


# --------------------------------------------------------------------------- helpers
def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def _skill(sm: float, sb: float) -> float | None:
    return None if sb <= 0 else 1.0 - sm / sb


def skill_report(ev: pl.DataFrame, em: str, eb: str, cfg: MsConfig) -> dict[str, Any]:
    """``ev``: ``asset_id, session, year, market`` + 오차 열 ``em``(모델)·``eb``(baseline)."""
    assets = sorted(ev["asset_id"].unique().to_list())
    a_idx = np.searchsorted(np.array(assets), ev["asset_id"].to_numpy())
    vals = np.column_stack([ev[em].to_numpy(), ev[eb].to_numpy(), np.ones(ev.height)])
    sess = ev["session"].to_numpy()
    out: dict[str, Any] = {"n_rows": ev.height, "assets": assets}
    for bl in (cfg.block_sessions, *cfg.block_sensitivity):
        m = block_matrix(block_ids(sess, bl), a_idx, vals, len(assets))
        key = "main" if bl == cfg.block_sessions else f"block_{bl}"
        out[key] = {
            "block_sessions": bl,
            **bootstrap_ci(m, mean_asset_skill, cfg.bootstrap_resamples, cfg.bootstrap_seed),
        }
    g = ev.group_by("asset_id").agg(
        pl.col(em).sum().alias("sm"), pl.col(eb).sum().alias("sb"), pl.len().alias("n")
    )
    out["per_asset"] = {
        r["asset_id"]: {"skill": _skill(r["sm"], r["sb"]), "n": r["n"]}
        for r in g.sort("asset_id").to_dicts()
    }
    gy = ev.group_by(["year", "asset_id"]).agg(
        pl.col(em).sum().alias("sm"), pl.col(eb).sum().alias("sb")
    )
    gy = gy.with_columns((1 - pl.col("sm") / pl.col("sb")).alias("skill"))
    out["per_year"] = {
        int(r["year"]): r["skill"]
        for r in gy.group_by("year").agg(pl.col("skill").mean()).sort("year").to_dicts()
    }
    gm = ev.group_by(["market", "asset_id"]).agg(
        pl.col(em).sum().alias("sm"), pl.col(eb).sum().alias("sb")
    )
    gm = gm.with_columns((1 - pl.col("sm") / pl.col("sb")).alias("skill"))
    out["per_market"] = {
        r["market"]: r["skill"]
        for r in gm.group_by("market").agg(pl.col("skill").mean()).sort("market").to_dicts()
    }
    return out


def _point_skill(ev: pl.DataFrame, em: str, eb: str) -> float | None:
    if ev.height == 0:
        return None
    g = ev.group_by("asset_id").agg(pl.col(em).sum().alias("sm"), pl.col(eb).sum().alias("sb"))
    s = [1 - r["sm"] / r["sb"] for r in g.to_dicts() if r["sb"] > 0]
    return float(np.mean(s)) if s else None


# --------------------------------------------------------------------------- Opportunity
def evaluate_opportunity(oof: pl.DataFrame, cfg: MsConfig, model: str) -> dict[str, Any]:
    p, b = f"p_opp_{model}", "b_opp_mean"
    ev = oof.filter(
        pl.col("label_matured")
        & pl.col("y_opp").is_not_null()
        & pl.col(p).is_not_null()
        & pl.col(b).is_not_null()
    ).with_columns(
        pl.col("session").dt.year().alias("year"),
        ((pl.col("y_opp") - pl.col(p)) ** 2).alias("_em"),
        ((pl.col("y_opp") - pl.col(b)) ** 2).alias("_eb"),
    )
    if ev.height == 0:
        return {"n_rows": 0}
    rep = skill_report(ev, "_em", "_eb", cfg)
    ic_asset, ic_year = {}, {}
    for aid, sub in ev.group_by("asset_id", maintain_order=True):
        ic_asset[aid[0]] = _spearman(sub[p].to_numpy(), sub["y_opp"].to_numpy())
    for (yr, aid), sub in ev.group_by(["year", "asset_id"], maintain_order=True):
        ic_year.setdefault(int(yr), []).append(
            _spearman(sub[p].to_numpy(), sub["y_opp"].to_numpy())
        )
    rep["ic_per_asset"] = ic_asset
    rep["ic_mean_over_assets"] = _nanmean(list(ic_asset.values()))
    rep["ic_per_year_mean_over_assets"] = {y: _nanmean(v) for y, v in sorted(ic_year.items())}
    rep["ic_positive_years"] = int(
        sum(1 for v in rep["ic_per_year_mean_over_assets"].values() if v is not None and v > 0)
    )
    return rep


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r = spearmanr(a, b).statistic
    return None if np.isnan(r) else float(r)


def _nanmean(v: list) -> float | None:
    x = [t for t in v if t is not None]
    return float(np.mean(x)) if x else None


def evaluate_sector_selection(oof: pl.DataFrame, cfg: MsConfig, pred_col: str) -> dict[str, Any]:
    """같은 날짜·같은 시장의 부모가 있는 자산 사이 순위 IC와 상위-하위 시장 대비 수익."""
    ev = oof.filter(
        pl.col("label_matured")
        & (pl.col("has_parent") == 1)
        & pl.col("y_mkt").is_not_null()
        & pl.col(pred_col).is_not_null()
    )
    if ev.height == 0:
        return {"n_dates": 0}
    d = (
        ev.group_by(["market", "session"])
        .agg(
            pl.corr(pl.col(pred_col).rank(), pl.col("y_mkt").rank()).alias("ic"),
            (
                pl.col("y_mkt").sort_by(pred_col).last() - pl.col("y_mkt").sort_by(pred_col).first()
            ).alias("tmb"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 3)
        .drop_nulls(["ic"])
        .sort(["market", "session"])
    )
    if d.height == 0:
        return {"n_dates": 0}
    sess = d["session"].to_numpy()
    block = block_ids(sess, cfg.block_sessions)
    res: dict[str, Any] = {"n_dates": d.height}
    for name in ("ic", "tmb"):
        vals = np.column_stack([d[name].to_numpy(), np.ones(d.height)])
        m = block_matrix(block, np.zeros(d.height, dtype=int), vals, 1)
        res[name] = bootstrap_ci(
            m,
            lambda t: float(t[0, 0] / t[0, 1]) if t[0, 1] > 0 else float("nan"),
            cfg.bootstrap_resamples,
            cfg.bootstrap_seed,
        )
    res["per_year"] = {
        int(r["year"]): {"ic": r["ic"], "tmb": r["tmb"]}
        for r in d.with_columns(pl.col("session").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("ic").mean(), pl.col("tmb").mean())
        .sort("year")
        .to_dicts()
    }
    res["per_market"] = {
        r["market"]: {"ic": r["ic"], "tmb": r["tmb"]}
        for r in d.group_by("market").agg(pl.col("ic").mean(), pl.col("tmb").mean()).to_dicts()
    }
    return res


# --------------------------------------------------------------------------- Stability
def _reliability(p: np.ndarray, y: np.ndarray, nbins: int) -> list[dict[str, Any]]:
    if len(p) < nbins:
        return []
    order = np.argsort(p, kind="stable")
    rows = []
    for i, ix in enumerate(np.array_split(order, nbins)):
        rows.append(
            {
                "bin": i + 1,
                "n": int(len(ix)),
                "mean_pred": float(p[ix].mean()),
                "event_rate": float(y[ix].mean()),
            }
        )
    return rows


def evaluate_stability(
    oof: pl.DataFrame, cfg: MsConfig, model: str, loco_years: list[int] | None = None
) -> dict[str, Any]:
    p = f"p_stab_{model}"
    need = [p, "b_stab_asset_rate", "b_stab_logit_rvol", "b_stab_pooled"]
    ev = oof.filter(
        pl.col("label_matured")
        & pl.col("y_loss").is_not_null()
        & pl.all_horizontal([pl.col(c).is_not_null() for c in need])
    ).with_columns(
        pl.col("session").dt.year().alias("year"),
        ((pl.col(p) - pl.col("y_loss")) ** 2).alias("_em"),
        ((pl.col("b_stab_asset_rate") - pl.col("y_loss")) ** 2).alias("_ea"),
        ((pl.col("b_stab_logit_rvol") - pl.col("y_loss")) ** 2).alias("_eb"),
        ((pl.col("b_stab_pooled") - pl.col("y_loss")) ** 2).alias("_ep"),
    )
    total_matured = oof.filter(pl.col("label_matured") & pl.col("y_loss").is_not_null()).height
    res: dict[str, Any] = {
        "n_rows": ev.height,
        "n_matured_rows_before_null_rule": total_matured,
        "n_rows_dropped_null_rule": total_matured - ev.height,
    }
    if ev.height == 0:
        return res
    res["skill_vs_baseline_b_asset_intercept_rvol"] = skill_report(ev, "_em", "_eb", cfg)
    res["skill_vs_baseline_a_asset_base_rate"] = skill_report(ev, "_em", "_ea", cfg)
    res["baseline_b_vs_a"] = skill_report(ev, "_eb", "_ea", cfg)["main"]
    res["baseline_a_vs_pooled"] = skill_report(ev, "_ea", "_ep", cfg)["main"]
    y, pr = ev["y_loss"].to_numpy(), ev[p].to_numpy()
    res["reliability_pooled"] = _reliability(pr, y, cfg.reliability_bins)
    per_asset: dict[str, Any] = {}
    for aid, sub in ev.group_by("asset_id", maintain_order=True):
        ya, pa = sub["y_loss"].to_numpy(), sub[p].to_numpy()
        rel = _reliability(pa, ya, cfg.reliability_bins_per_asset)
        per_asset[aid[0]] = {
            "n": int(len(ya)),
            "event_rate": float(ya.mean()),
            "pr_auc": float(average_precision_score(ya, pa)) if 0 < ya.sum() < len(ya) else None,
            "reliability": rel,
            "low_bin_below_high_bin": bool(rel and rel[0]["event_rate"] < rel[-1]["event_rate"]),
        }
    res["per_asset_diagnostics"] = per_asset
    res["pr_auc_pooled"] = float(average_precision_score(y, pr)) if 0 < y.sum() < len(y) else None
    res["prevalence_pooled"] = float(y.mean())
    # 위기 하나씩 빼기: 사건률이 가장 높은 해 top-N
    yr = ev.group_by("year").agg(pl.col("y_loss").mean().alias("rate"), pl.len().alias("n"))
    top = yr.sort("rate", descending=True).head(cfg.crisis_top_n)
    years = loco_years if loco_years is not None else top["year"].to_list()
    res["leave_one_crisis_out"] = {
        "event_rate_by_year": {int(r["year"]): r["rate"] for r in yr.sort("year").to_dicts()},
        "dropped_years": [int(v) for v in years],
        "mean_skill_vs_b_without_year": {
            int(v): _point_skill(ev.filter(pl.col("year") != v), "_em", "_eb") for v in years
        },
        "mean_skill_vs_a_without_year": {
            int(v): _point_skill(ev.filter(pl.col("year") != v), "_em", "_ea") for v in years
        },
    }
    return res


# --------------------------------------------------------------------------- top level
def evaluate_all(oof: pl.DataFrame, cfg: MsConfig, *, opportunity_target: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "opportunity_target": opportunity_target,
        "opportunity": {m: evaluate_opportunity(oof, cfg, m) for m in ("ridge", "lgbm")},
        "sector_selection": {
            "ridge": evaluate_sector_selection(oof, cfg, "p_mkt_ridge"),
            "lgbm": evaluate_sector_selection(oof, cfg, "p_mkt_lgbm"),
            "baseline_hist_mean": evaluate_sector_selection(oof, cfg, "b_mkt_mean"),
        },
        "stability": {m: evaluate_stability(oof, cfg, m) for m in ("logit", "lgbm")},
    }
    return _clean(metrics)


def render_report(metrics: dict[str, Any], header: dict[str, Any], fold_table: list[dict]) -> str:
    def f(x: Any, nd: int = 4) -> str:
        return "-" if x is None else f"{x:.{nd}f}"

    def ci(d: dict | None) -> str:
        if not d or d.get("point") is None:
            return "-"
        return f"{f(d['point'])} [{f(d.get('lo'))}, {f(d.get('hi'))}] (blocks={d.get('n_blocks')})"

    L = [f"# 시장·섹터 레이어 2 — {header['run_id']}", ""]
    L += [
        f"- market: {header['market']}  panel: {header['panel_version']}  smoke: {header['smoke']}"
    ]
    L += [f"- opportunity_target: `{metrics['opportunity_target']}`"]
    L += [f"- return_basis: `{header['return_basis']}`  cash_basis: `{header['cash_basis']}`"]
    L += ["- 사전등록 전 개발 실행이다. 결과를 튜닝에 쓰지 않는다.", ""]
    L += ["## Fold", "", "| fold | n_train | n_test | n_test_matured | first | last | note |"]
    L += ["|---|---:|---:|---:|---|---|---|"]
    for r in fold_table:
        L += [
            f"| {r['year']} | {r['n_train']} | {r['n_test']} | {r['n_test_matured']} | "
            f"{str(r['first_decision'])[:10]} | {str(r['last_decision'])[:10]} | "
            f"{r.get('skipped_reason') or ''} |"
        ]
    L += ["", "## Opportunity (squared-error skill vs 자산별 과거 평균, 자산 평균)", ""]
    for m, r in metrics["opportunity"].items():
        if not r.get("n_rows"):
            L += [f"- {m}: 표본 없음"]
            continue
        L += [f"### {m}", "", f"- skill (block {r['main']['block_sessions']}): {ci(r['main'])}"]
        for k in r:
            if k.startswith("block_") and isinstance(r[k], dict):
                L += [f"- skill (block {r[k]['block_sessions']}): {ci(r[k])}"]
        L += [f"- IC 자산 평균: {f(r['ic_mean_over_assets'])}, IC>0인 해: {r['ic_positive_years']}"]
        L += ["", "| asset | skill | IC | n |", "|---|---:|---:|---:|"]
        for a, v in r["per_asset"].items():
            L += [f"| {a} | {f(v['skill'])} | {f(r['ic_per_asset'].get(a))} | {v['n']} |"]
        L += ["", "| year | skill(자산 평균) | IC(자산 평균) |", "|---|---:|---:|"]
        for y, v in r["per_year"].items():
            L += [f"| {y} | {f(v)} | {f(r['ic_per_year_mean_over_assets'].get(y))} |"]
        L += [""]
    L += ["## 섹터 상대 선택 (같은 날짜 순위 IC · 상위-하위 시장 대비 수익)", ""]
    for m, r in metrics["sector_selection"].items():
        if not r.get("n_dates"):
            L += [f"- {m}: 표본 없음"]
            continue
        L += [f"- {m}: IC {ci(r['ic'])} · 상위-하위 {ci(r['tmb'])} · 날짜 {r['n_dates']}"]
    L += ["", "## Stability (Brier skill, 자산 평균)", ""]
    for m, r in metrics["stability"].items():
        L += [f"### {m}", ""]
        if not r.get("n_rows"):
            L += ["- 표본 없음", ""]
            continue
        L += [f"- 표본 {r['n_rows']}행 (null 규칙으로 뺀 행 {r['n_rows_dropped_null_rule']})"]
        b = r["skill_vs_baseline_b_asset_intercept_rvol"]
        a = r["skill_vs_baseline_a_asset_base_rate"]
        L += [f"- vs baseline b (자산 절편+rvol): {ci(b['main'])}"]
        for k in b:
            if k.startswith("block_") and isinstance(b[k], dict):
                L += [f"  - block {b[k]['block_sessions']}: {ci(b[k])}"]
        L += [f"- vs baseline a (자산 기저율): {ci(a['main'])}"]
        L += [f"- baseline b vs a: {ci(r['baseline_b_vs_a'])}"]
        L += [f"- baseline a vs pooled: {ci(r['baseline_a_vs_pooled'])}"]
        L += [f"- PR-AUC 전체: {f(r['pr_auc_pooled'])} (사건률 {f(r['prevalence_pooled'])})", ""]
        L += ["| bin | n | mean_pred | event_rate |", "|---:|---:|---:|---:|"]
        for x in r["reliability_pooled"]:
            L += [f"| {x['bin']} | {x['n']} | {f(x['mean_pred'])} | {f(x['event_rate'])} |"]
        L += ["", "| asset | n | 사건률 | PR-AUC | 낮은 구간<높은 구간 | skill vs b |"]
        L += ["|---|---:|---:|---:|---|---:|"]
        for aid, v in r["per_asset_diagnostics"].items():
            sk = b["per_asset"].get(aid, {}).get("skill")
            L += [
                f"| {aid} | {v['n']} | {f(v['event_rate'])} | {f(v['pr_auc'])} | "
                f"{v['low_bin_below_high_bin']} | {f(sk)} |"
            ]
        L += ["", "| year | skill vs b (자산 평균) | 사건률 |", "|---|---:|---:|"]
        rates = r["leave_one_crisis_out"]["event_rate_by_year"]
        for y, v in b["per_year"].items():
            L += [f"| {y} | {f(v)} | {f(rates.get(y))} |"]
        lo = r["leave_one_crisis_out"]
        L += [
            "",
            f"- 위기 하나씩 제외(사건률 상위 {len(lo['dropped_years'])}개 해): "
            f"{lo['dropped_years']}",
        ]
        for y in lo["dropped_years"]:
            L += [
                f"  - {y} 제외: skill vs b {f(lo['mean_skill_vs_b_without_year'].get(str(y)))}, "
                f"vs a {f(lo['mean_skill_vs_a_without_year'].get(str(y)))}"
            ]
        L += [""]
    return "\n".join(L) + "\n"
