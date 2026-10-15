"""Distribution on the full non-overlapping grid: box statistics, hit rates, path bands."""

from __future__ import annotations

import numpy as np
import polars as pl

from .select import group_names

QS = (0.05, 0.25, 0.5, 0.75, 0.95)
METRICS = ("raw_label_20d", "fwd_ret_20d")


def group_order(direction: int, mode: str = "tie", good_group: str | None = None) -> list[str]:
    a, b = group_names(direction, mode, good_group)
    return [a, b] if mode == "two" else [a, "mid", b]


def box_stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    out = {"n": int(v.size), "n_missing": int(values.size - v.size)}
    if v.size == 0:
        return out
    q = np.quantile(v, QS)
    out.update(
        mean=float(v.mean()),
        q05=float(q[0]),
        q25=float(q[1]),
        q50=float(q[2]),
        q75=float(q[3]),
        q95=float(q[4]),
        hit_rate=float((v > 0).mean()),
    )
    return out


def distribution_rows(
    feature: str,
    direction: int,
    grid: pl.DataFrame,
    paths: np.ndarray,
    mode: str = "tie",
    good_group: str | None = None,
) -> list[dict]:
    """``grid``: grid (date, ticker) rows with group and labels; ``paths`` aligned (R, 21)."""
    rows: list[dict] = []
    groups = grid["group"].to_numpy()
    for g in group_order(direction, mode, good_group):
        sel = groups == g
        ndates = grid.filter(pl.col("group") == g)["trade_date"].n_unique()
        for metric in METRICS:
            st = box_stats(grid[metric].to_numpy()[sel].astype(float))
            st["n_dates"] = int(ndates)
            for stat, val in st.items():
                rows.append(
                    {
                        "feature": feature,
                        "group": g,
                        "metric": metric,
                        "stat": stat,
                        "k": None,
                        "value": float(val),
                    }
                )
        p = paths[sel]
        for k in range(21):
            col = p[:, k]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            for stat, q in (("q25", 0.25), ("q50", 0.5), ("q75", 0.75)):
                rows.append(
                    {
                        "feature": feature,
                        "group": g,
                        "metric": "cum_excess_path",
                        "stat": stat,
                        "k": k,
                        "value": float(np.quantile(col, q)),
                    }
                )
            rows.append(
                {
                    "feature": feature,
                    "group": g,
                    "metric": "cum_excess_path",
                    "stat": "n",
                    "k": k,
                    "value": float(col.size),
                }
            )
    return rows
