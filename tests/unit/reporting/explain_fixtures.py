"""근거 입력(`briefing-explain.v1`, 그날 `explain-D.json`) 시험용 합성 fixture.

serving의 explain 단계가 쓸 파일 모양(20261026 계획 §5.2.1)을 따릅니다. 값은 합성이라 실제 시장과
무관합니다. report 바이트에서 report sha256·순위를 읽어 짝을 맞춥니다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import date, timedelta

KR_GROUPS = [
    ("price", "가격 흐름"),
    ("risk", "변동성·위험"),
    ("liq", "거래·유동성·규모"),
    ("flow", "수급"),
    ("value", "가치·수익성"),
    ("event", "이벤트·공시"),
    ("holders", "지분·기관"),
]
US_GROUPS = KR_GROUPS + [("micro", "옵션·호가 구조")]


def sessions(end: str, count: int = 61) -> list:
    """end 이하 평일 count개(시작일~end). 합성 달력입니다."""
    day = date.fromisoformat(end)
    out = []
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day -= timedelta(days=1)
    return out[::-1]


def _path(rng: random.Random, n: int, gaps: set) -> list:
    level, out = 100.0, []
    for t in range(n):
        if t:
            level *= 1.0 + rng.uniform(-0.03, 0.03)
        out.append(None if t in gaps else round(level, 3))
    return out


def _row(rng: random.Random, row: dict, groups: list, dates: list, market: str) -> dict:
    n = len(dates)
    gaps = set(range(20, 25)) if row["rank"] in (2, 7) else set()
    path = _path(rng, n, gaps)
    bench = _path(rng, n, set())
    ret = {}
    for h in ("5", "20", "60"):
        start = n - 1 - int(h)
        ret[h] = None if path[start] is None else path[-1] / path[start] - 1.0
    scope = "같은 시장(코스닥)" if market == "KR" else "US 적격 종목"
    return {
        "rank": row["rank"],
        "symbol": row["symbol"],
        "score": row["score"],
        "explain_status": "ok",
        "explain_reason": None,
        "delta": {g: rng.uniform(-0.01, 0.03) for g, _ in groups},
        "top_features": [
            {
                "feature": "f_low",
                "label": "60일 수익률",
                "pct": 0.02,
                "side": "low",
                "scope": scope,
            },
            {
                "feature": "f_high",
                "label": "한 달 고저 폭",
                "pct": 0.99,
                "side": "high",
                "scope": scope,
            },
        ],
        "price_status": "partial" if gaps else "ok",
        "price_reason": None,
        "path": path,
        "bench_path": bench,
        "ret": ret,
        "ret_note": {"20": "정지 중 시작"} if row["rank"] == 3 else {},
        "gaps": [{"date": dates[t], "kind": "halt"} for t in sorted(gaps)],
        "n_obs": n - len(gaps),
    }


def explain_doc(report_bytes: bytes, invocation_id: str, top_n: int = 100) -> dict:
    """report(envelope) 바이트와 짝이 맞는 정상 근거 입력."""
    env = json.loads(report_bytes)
    models = {}
    for report in env["markets"]:
        rng = random.Random(report["model_id"] + env["report_date"])
        market = report["market"]
        groups = KR_GROUPS if market == "KR" else US_GROUPS
        asof = report["feature_asof_date"]
        dates = sessions(asof)
        rows = [_row(rng, r, groups, dates, market) for r in report["rankings"][:top_n]]
        mean = {g: sum(r["delta"][g] for r in rows) / len(rows) for g, _ in groups}
        models[report["model_id"]] = {
            "market": market,
            "status": "ok",
            "reason": None,
            "asof": asof,
            "inputs": {"prepared_sha256": "0" * 64, "bundle_sha256": "1" * 64},
            "groups": [{"id": g, "label": label} for g, label in groups],
            "dates": dates,
            "top100_mean_delta": mean,
            "rows": rows,
        }
    return {
        "schema": "briefing-explain.v1",
        "method_version": "occlusion-mean-mode-v1",
        "group_map_version": "groups-2026-10-26",
        "unit": env["report_date"],
        "invocation_id": invocation_id,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "release": "r20261005",
        "status": "ok",
        "reason": None,
        "models": models,
    }


def dump(doc: dict) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def variant(doc: dict) -> dict:
    return copy.deepcopy(doc)
