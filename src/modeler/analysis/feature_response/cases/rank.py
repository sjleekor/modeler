"""Ranked comparison of the 47 features: one number pair per feature, computed from the
distribution table (metric ``raw_label_20d``) only. Viewing order, not a verdict."""

from __future__ import annotations

import re

import polars as pl

from . import config as C
from .config import FeatureSpec
from .select import group_names, has_good_side

METRIC = "raw_label_20d"
BIG = 5.0  # |hit-rate gap| in %p: at or above is clear
MID = 2.0  # at or above is middling, below is small
SMALL_N = 2000
SMALL_RATIO = 0.3
REF_NAME = {
    "card": "카드 기대",
    "plan": "계획 표",
    "obs": "관측 부호",
    "explore": "모델 방향(탐색)",
}
KIND_LABEL = {
    "card": "카드 기대",
    "plan": "계획 표",
    "obs": "관측",
    "explore": "탐색",
    "none": "없음",
}


def basis_kind(spec: FeatureSpec) -> str:
    b = spec.basis
    if b.startswith("좋은 쪽 정하지 않음"):
        return "none"
    if b.startswith("카드 기대"):
        return "card"
    if b.startswith("계획"):
        return "plan"
    if b.startswith("단변량 관측"):
        return "obs"
    if b.startswith("층 3 모델 반응"):
        return "explore"
    raise ValueError(f"unclassified basis: {spec.name}: {b}")


def pair_names(spec: FeatureSpec) -> tuple[str, str, str | None]:
    """(first, second, middle) group names. First is the good side when there is one."""
    a, b = group_names(spec.direction, spec.mode, spec.good_group)
    if spec.mode == "two":
        if not has_good_side(spec.direction, spec.mode, spec.good_group):
            return "other", "modal", None  # no direction: other vs modal
        return a, b, None
    return a, b, "mid"


def pair_labels(spec: FeatureSpec) -> tuple[str, str]:
    first, second, _ = pair_names(spec)
    return {
        "good": ("좋은 쪽", "반대쪽"),
        "top": ("상위", "하위"),
        "other": ("그 외", "가장 흔한 값"),
    }[first if first in ("good", "top", "other") else "good"]


def _stat(df: pl.DataFrame, feature: str, group: str, stat: str) -> float | None:
    r = df.filter(
        (pl.col("feature") == feature)
        & (pl.col("group") == group)
        & (pl.col("metric") == METRIC)
        & (pl.col("stat") == stat)
        & pl.col("k").is_null()
    )["value"]
    return float(r[0]) if r.len() else None


def direction_text(spec: FeatureSpec) -> str:
    if not has_good_side(spec.direction, spec.mode, spec.good_group):
        return "좋은 쪽 없음"
    if spec.mode == "two":
        return "가장 흔한 값이 좋음" if spec.good_group == "modal" else "그 외가 좋음"
    return "높은 쪽이 좋음" if spec.direction > 0 else "낮은 쪽이 좋음"


def summary_text(row: dict) -> str:
    """Neutral one-line reading of the numbers."""
    d = row["hit_diff"]
    if d is None:
        return "무리를 만들 수 없어 비교 없음"
    a = abs(d)
    first, second = row["labels"]
    if a < MID:
        return f"양 끝 차이 {max(1, round(a))}%p 안팎"
    mag = "뚜렷" if a >= BIG else "중간"
    leader = first if d > 0 else second
    if row["good_side"]:
        other = second if d > 0 else first
        txt = f"{leader}이 {other}보다 적중률 {a:.1f}%p 앞섬({mag})"
        if row["kind"] != "none":
            ref = REF_NAME[row["kind"]]
            jo = "과" if ref.endswith(")") else "와"
            txt += f" · {ref}{jo} " + ("같은 쪽" if d > 0 else "반대 쪽")
        return txt
    j = "가" if leader in ("상위", "하위") else ("가" if leader == "그 외" else "이")
    return f"{leader}{j} 적중률 {a:.1f}%p 앞섬({mag}, 방향 없음)"


def first_clause(short: str) -> str:
    s = re.split(r"(?<=[.])\s", short.strip())[0].rstrip(".")
    return s


def rank_rows(
    specs: list[FeatureSpec], dist: pl.DataFrame, unformable: dict[str, str] | None = None
) -> list[dict]:
    """One row per feature, already in default order (|hit-rate gap| descending, no-group last)."""
    unformable = unformable or {}
    grp_of = {c: i for i, (c, _) in enumerate(C.CARD_GROUPS)}
    rows = []
    for sp in specs:
        first, second, mid = pair_names(sp)
        gs = has_good_side(sp.direction, sp.mode, sp.good_group)
        row = {
            "name": sp.name,
            "spec": sp,
            "kind": basis_kind(sp),
            "good_side": gs,
            "grp": grp_of[sp.card],
            "labels": pair_labels(sp),
            "has_mid": mid is not None,
            "hit": [None, None, None],
            "n": [None, None, None],
            "hit_diff": None,
            "mean_diff": None,
            "sign": "nogroup",
            "small": False,
        }
        h1, h2 = _stat(dist, sp.name, first, "hit_rate"), _stat(dist, sp.name, second, "hit_rate")
        if sp.name in unformable or h1 is None or h2 is None:
            rows.append(row)
            continue
        m1, m2 = _stat(dist, sp.name, first, "mean"), _stat(dist, sp.name, second, "mean")
        n1, n2 = _stat(dist, sp.name, first, "n"), _stat(dist, sp.name, second, "n")
        hm = nm = None
        if mid:
            hm, nm = _stat(dist, sp.name, mid, "hit_rate"), _stat(dist, sp.name, mid, "n")
        row["hit"] = [h1, hm, h2]
        row["n"] = [n1, nm, n2]
        row["hit_diff"] = (h1 - h2) * 100
        row["mean_diff"] = (m1 - m2) * 100 if m1 is not None and m2 is not None else None
        if gs:
            row["sign"] = "lead" if row["hit_diff"] > 0 else "trail"
        else:
            row["sign"] = "none"
        lo, hi = sorted([n1 or 0, n2 or 0])
        row["small"] = lo < SMALL_N or (hi > 0 and lo / hi < SMALL_RATIO)
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["hit_diff"] is None,
            -abs(r["hit_diff"]) if r["hit_diff"] is not None else 0.0,
            r["name"],
        )
    )
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
