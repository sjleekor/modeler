"""회사 품질 점수 부분 C 판정 통계 — AUC·분위·포착률·클러스터 부트스트랩·게이트·등급·manifest.

사전등록 20261010_quality_score §5.4(통계량·신뢰구간·다중검정·p값), §5.5(G1~G4·등급·요약 판정),
§5.3(O1 폴백 3개 연도·O4 4개 연도), §9(기록 칸), §10(중단 제안 문구).

**데이터를 읽지 않는 순수 함수 모듈이다.** 입력은 ``pl.DataFrame[corp_code, fy, score, event]`` 이다.
``score`` 는 높을수록 좋은 점수이고 위험 점수 = 100 − score 이다(§5.4). ``event`` 는 0/1 이다.
결측(null·nan) 행은 함수 안에서도 뺀다. 뺀 행 수는 결과에 ``n_dropped`` 로 적는다.

사전등록 문면이 정하지 않은 선택은 상수로 두고 ``# CI-<이름>`` 주석을 달았다(해석 표 후보).
"""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

JUDGE_VERSION = "quality-score-v0/company_judge/1"

# ---- 사전등록에 적힌 값
OUTCOME_ORDER = ("O1", "O2", "O3", "O4")  # §5.4 고정 순서
N_BOOT = 2000  # §5.4
SEED = 20261010  # §5.4 고정 시드(ETF I14 선례와 같은 값)
ALPHA = 0.05  # §5.4 단측
G1_MIN = 30  # §5.5 G1 표본 부족
G1_FULL = 50  # §5.5 30~49건은 "탐색 판정"
G2_AUC = 0.60  # §5.5 G2
G3_LIFT = 1.5  # §5.5 G3
G4_SHARE = (2, 3)  # §5.5 G4 2/3 이상
G4_MIN_EVENTS = 10  # §5.5 G4 연도별 사건 10건 미만인 해는 세지 않는다
QUANTILES = 5  # §5.4 합성 5분위
CAPTURE_FRACS = (0.10, 0.20)  # §5.4 포착률 하위 10%·20%

TEST_SKIPPED = "검정 안 함(앞 단계 탈락)"  # §5.4
STOP_TEXT = "중단 조건 충족 — 사용자 확인 대기"  # §10
ADOPT_TEXT = "재무 건전성 점수로 채택 후보"  # §5.5
FOLD_TEXT = "접기 제안 — 중단 조건 충족, 사용자 확인 대기"  # §5.5·§10

# ---- 사전등록 문면이 둘로 읽히거나 정하지 않은 선택(해석 표 후보)
# CI-QUANTILE-CUT: 5분위·포착률을 "연도 안 순위로 자른 뒤 풀링"(기본) vs "풀링 점수 순위로 자름".
DEFAULT_CUT = "within_year"
# CI-G1-GRADE: G1 탈락(30건 미만)의 등급 표기. 기본안 = ETF I28 선례, 등급 없음 + §10 제안 문구는 그대로.
G1_FAIL_GRADE_TEXT = "등급 없음(표본 부족)"
# CI-G4-DENOM: G4 분모. "counted" = 사건 10건 이상인 해만(기본), "all" = 판정 연도 전체.
DEFAULT_G4_DENOM = "counted"
# CI-SEQ-P-ONLY: 고정 순서 검정은 G1 결과와 무관하게 p값만으로 진행한다(표본 부족이어도 p가 있으면 순서 유지).
# CI-SEED-PER-OUTCOME: 결과 변수마다 같은 시드로 새 난수열을 만든다(O 하나만 다시 돌려도 같은 값).
# CI-QUINTILE-TIE: 5분위 번호 = floor(5 × (평균 순위 − 0.5) / n). 동률 묶음은 같은 분위에 들어간다.
# CI-CAPTURE-RULE: 하위 비율 f 집합 = 평균 순위 ≤ f × n 인 행(경계 동률은 묶음 통째로 평균 순위로 정함).
# CI-AUC-FOR-BOOT: 부트스트랩은 가중(뽑힌 횟수) Mann–Whitney로 계산한다. 행을 복제한 AUC와 같다.


# ---------------------------------------------------------------- 입력 정리
def _clean(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """null·nan·사건 값이 0/1이 아닌 행을 빼고 (정리된 프레임, 뺀 수)를 돌려준다.

    ``risk`` = 100 − score 열을 더한다. 형식: corp_code, fy, risk, event."""
    n0 = df.height
    d = df.select(
        pl.col("corp_code").cast(pl.String),
        pl.col("fy").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
        pl.col("event").cast(pl.Int64, strict=False),
    ).drop_nulls()
    d = d.filter(pl.col("score").is_finite() & pl.col("event").is_in([0, 1]))
    d = d.with_columns((100.0 - pl.col("score")).alias("risk"))
    return d.select("corp_code", "fy", "risk", "event"), n0 - d.height


# ---------------------------------------------------------------- AUC
def auc(risk, event) -> float:
    """Mann–Whitney AUC. 사건 위험 점수가 비사건보다 높을 확률, 동률은 0.5.

    sklearn ``roc_auc_score`` 와 같다. 사건·비사건 중 하나가 0이면 nan."""
    r = np.asarray(risk, dtype=float)
    e = np.asarray(event).astype(int)
    n1 = int(e.sum())
    n0 = int(len(e) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(r)
    return float((rk[e == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def yearly_auc(d: pl.DataFrame) -> list[dict]:
    """연도별 AUC와 연도별 사건 수·전체 수. ``d`` 는 ``_clean`` 결과."""
    out = []
    for fy in sorted(d["fy"].unique().to_list()):
        s = d.filter(pl.col("fy") == fy)
        out.append(
            {
                "fy": int(fy),
                "n": s.height,
                "n_events": int(s["event"].sum()),
                "auc": auc(s["risk"].to_numpy(), s["event"].to_numpy()),
            }
        )
    return out


def leave_one_year_out(d: pl.DataFrame) -> dict:
    """한 해씩 뺀 풀링 AUC(§5.4 민감도). 최소·최대와 해별 값."""
    by = {}
    for fy in sorted(d["fy"].unique().to_list()):
        s = d.filter(pl.col("fy") != fy)
        by[str(int(fy))] = (
            auc(s["risk"].to_numpy(), s["event"].to_numpy()) if s.height else float("nan")
        )
    vals = [v for v in by.values() if not math.isnan(v)]
    return {
        "min": min(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
        "by_year": by,
    }


# ---------------------------------------------------------------- 분위·포착률
def _groups(d: pl.DataFrame, cut: str) -> list[pl.DataFrame]:
    if cut == "within_year":
        return [d.filter(pl.col("fy") == fy) for fy in sorted(d["fy"].unique().to_list())]
    if cut == "pooled":
        return [d]
    raise ValueError(f"cut must be 'within_year' or 'pooled', got {cut!r}")


def quintile_rates(d: pl.DataFrame, cut: str = DEFAULT_CUT, k: int = QUANTILES) -> dict:
    """점수 분위별 사건율과 기저율 대비 배율(lift). 1분위 = 점수가 가장 낮은(위험한) 쪽.

    ``cut="within_year"`` 는 연도 안 점수 순위(동률 평균 순위)로 자른 뒤 풀링한다(CI-QUANTILE-CUT)."""
    n = np.zeros(k, dtype=int)
    ev = np.zeros(k, dtype=int)
    for g in _groups(d, cut):
        if g.height == 0:
            continue
        rk = rankdata(-g["risk"].to_numpy())  # 점수 오름차순 순위(위험 내림차순)
        q = np.minimum(k - 1, np.floor(k * (rk - 0.5) / g.height).astype(int))
        e = g["event"].to_numpy()
        n += np.bincount(q, minlength=k)
        ev += np.bincount(q, weights=e, minlength=k).astype(int)
    base = ev.sum() / n.sum() if n.sum() else float("nan")
    rows = []
    for i in range(k):
        rate = ev[i] / n[i] if n[i] else float("nan")
        rows.append(
            {
                "quintile": i + 1,
                "n": int(n[i]),
                "n_events": int(ev[i]),
                "event_rate": float(rate),
                "lift": float(rate / base) if base and not math.isnan(rate) else float("nan"),
            }
        )
    return {"cut": cut, "base_rate": float(base), "quintiles": rows}


def capture(d: pl.DataFrame, frac: float, cut: str = DEFAULT_CUT) -> dict:
    """점수 하위 ``frac`` 의 포착률(그 집합의 사건 ÷ 전체 사건)과 lift(집합 사건율 ÷ 기저율).

    집합 = 점수 평균 순위 ≤ frac × n (CI-CAPTURE-RULE). 동률이 없고 frac × n이 정수면
    lift = 포착률 / frac 이다."""
    n_sel = 0
    ev_sel = 0
    n_all = 0
    ev_all = 0
    for g in _groups(d, cut):
        if g.height == 0:
            continue
        rk = rankdata(-g["risk"].to_numpy())
        m = rk <= frac * g.height + 1e-9
        e = g["event"].to_numpy()
        n_sel += int(m.sum())
        ev_sel += int(e[m].sum())
        n_all += g.height
        ev_all += int(e.sum())
    base = ev_all / n_all if n_all else float("nan")
    rate = ev_sel / n_sel if n_sel else float("nan")
    return {
        "frac": frac,
        "cut": cut,
        "n_selected": n_sel,
        "n_events_selected": ev_sel,
        "capture": float(ev_sel / ev_all) if ev_all else float("nan"),
        "event_rate": float(rate),
        "base_rate": float(base),
        "lift": float(rate / base) if base and not math.isnan(rate) else float("nan"),
    }


# ---------------------------------------------------------------- 클러스터 부트스트랩
class _WeightedAuc:
    """행을 위험 점수로 한 번 정렬해 두고, 회사 가중치(뽑힌 횟수)만 바꿔 AUC를 빠르게 구한다."""

    def __init__(self, d: pl.DataFrame):
        corp = d["corp_code"].to_numpy()
        self.corp_uniq, inv = np.unique(corp, return_inverse=True)
        risk = d["risk"].to_numpy().astype(float)
        order = np.argsort(risk, kind="stable")
        r = risk[order]
        self.ev = d["event"].to_numpy().astype(float)[order]
        self.corp_idx = inv[order]
        new = np.concatenate([[True], r[1:] != r[:-1]]) if len(r) else np.array([], dtype=bool)
        self.gid = np.cumsum(new) - 1
        self.n_groups = int(self.gid[-1]) + 1 if len(r) else 0

    @property
    def n_corps(self) -> int:
        return len(self.corp_uniq)

    def auc(self, counts: np.ndarray) -> float:
        w = counts[self.corp_idx].astype(float)
        e = np.bincount(self.gid, weights=w * self.ev, minlength=self.n_groups)
        ne = np.bincount(self.gid, weights=w * (1.0 - self.ev), minlength=self.n_groups)
        et, nt = e.sum(), ne.sum()
        if et <= 0 or nt <= 0:
            return float("nan")
        below = np.cumsum(ne) - ne
        return float((e * (below + 0.5 * ne)).sum() / (et * nt))


def cluster_bootstrap_auc(d: pl.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """회사(corp_code)를 복원 추출하고 한 회사의 모든 연도 행을 한 덩어리로 뽑는다(§5.4).

    각 회차 풀링 AUC를 돌려준다. 사건 또는 비사건이 하나도 안 뽑힌 회차는 nan이다."""
    out = np.full(n_boot, np.nan)
    if d.height == 0 or n_boot <= 0:
        return out
    wa = _WeightedAuc(d)
    rng = np.random.default_rng(seed)
    k = wa.n_corps
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, k, size=k), minlength=k)
        out[b] = wa.auc(counts)
    return out


def boot_summary(boot: np.ndarray) -> dict:
    """p값 = (AUC_b ≤ 0.5 회차 수 + 1) ÷ (유효 회차 수 + 1), 단측(§5.4). nan 회차는 분자·분모에서 뺀다.

    ETF I25 선례: nan 회차 수를 같이 적는다. 95% 백분위 구간을 더한다."""
    b = np.asarray(boot, dtype=float)
    f = b[np.isfinite(b)]
    res = {"b": int(len(b)), "n_valid": int(len(f)), "n_nan": int(len(b) - len(f))}
    if len(f) == 0:
        res.update(
            mean=float("nan"), ci95=[float("nan"), float("nan")], n_le_half=0, p_value=float("nan")
        )
        return res
    k = int((f <= 0.5).sum())
    res.update(
        mean=float(f.mean()),
        ci95=[float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))],
        n_le_half=k,
        p_value=float((k + 1) / (len(f) + 1)),
    )
    return res


# ---------------------------------------------------------------- 고정 순서 검정
def fixed_sequence(
    p_values: dict[str, float], alpha: float = ALPHA, order=OUTCOME_ORDER
) -> dict[str, dict]:
    """O1 → O2 → O3 → O4 순서로 단측 α에서 검정한다(§5.4). p < α이면 기각하고 다음으로 간다.

    기각 못 하면 거기서 멈추고 나머지는 "검정 안 함(앞 단계 탈락)"이다. p가 nan이면 기각 못 한 것으로 본다.
    ``p_values`` 에 없는 결과 변수는 건너뛴다(CI-SEQ-P-ONLY: G1 결과와 무관)."""
    out: dict[str, dict] = {}
    go = True
    step = 0
    for name in order:
        if name not in p_values:
            continue
        step += 1
        p = p_values[name]
        if not go:
            out[name] = {
                "step": step,
                "p": p,
                "tested": False,
                "rejected": None,
                "status": TEST_SKIPPED,
            }
            continue
        rej = bool(p is not None and not math.isnan(p) and p < alpha)
        out[name] = {
            "step": step,
            "p": p,
            "tested": True,
            "rejected": rej,
            "status": "기각" if rej else "기각 못 함",
        }
        go = rej
    return out


# ---------------------------------------------------------------- 게이트·등급
def g1_gate(n_events: int) -> dict:
    """G1: 풀링 사건 < 30 → 표본 부족(판정 안 함), 30~49 → 탐색 판정, 50 이상 → 판정."""
    if n_events < G1_MIN:
        status = "표본 부족"
    elif n_events < G1_FULL:
        status = "탐색 판정"
    else:
        status = "판정"
    return {
        "n_events": int(n_events),
        "status": status,
        "pass": n_events >= G1_MIN,
        "exploratory": status == "탐색 판정",
    }


def g2_gate(pooled_auc: float, seq: dict) -> dict:
    """G2: 풀링 AUC ≥ 0.60 그리고 고정 순서 단계에서 p < 0.05. 검정 안 함이면 판정도 "검정 안 함"."""
    point_ok = bool(not math.isnan(pooled_auc) and pooled_auc >= G2_AUC)
    if not seq["tested"]:
        return {
            "auc": pooled_auc,
            "point_ok": point_ok,
            "test_status": seq["status"],
            "pass": None,
            "status": "검정 안 함",
        }
    ok = bool(point_ok and seq["rejected"])
    return {
        "auc": pooled_auc,
        "point_ok": point_ok,
        "test_status": seq["status"],
        "p": seq["p"],
        "pass": ok,
        "status": "통과" if ok else "탈락",
    }


def g3_gate(lift20: float) -> dict:
    """G3: 하위 20% 사건율 ÷ 기저율 ≥ 1.5(점추정)."""
    ok = bool(not math.isnan(lift20) and lift20 >= G3_LIFT)
    return {"lift20": lift20, "threshold": G3_LIFT, "pass": ok}


def g4_gate(years: list[dict], denom: str = DEFAULT_G4_DENOM) -> dict:
    """G4: 판정 연도 중 연도별 AUC > 0.5인 해가 2/3 이상. 사건 10건 미만인 해는 세지 않는다.

    ``denom="counted"`` 는 센 해가 분모(기본), ``"all"`` 은 판정 연도 전체가 분모(CI-G4-DENOM).
    센 해가 0이면 탈락. 사전등록 예: 6개 → 4, O4 4개 → 3, O1 폴백 3개 → 2."""
    if denom not in ("counted", "all"):
        raise ValueError(f"denom must be 'counted' or 'all', got {denom!r}")
    counted = [y for y in years if y["n_events"] >= G4_MIN_EVENTS and not math.isnan(y["auc"])]
    pos = sum(1 for y in counted if y["auc"] > 0.5)
    den = len(counted) if denom == "counted" else len(years)
    ok = bool(len(counted) > 0 and den > 0 and pos * G4_SHARE[1] >= den * G4_SHARE[0])
    need = math.ceil(den * G4_SHARE[0] / G4_SHARE[1]) if den else 0
    return {
        "denominator_rule": denom,
        "n_years": len(years),
        "n_counted": len(counted),
        "n_positive": pos,
        "denominator": den,
        "needed": need,
        "skipped_years": [y["fy"] for y in years if y not in counted],
        "pass": ok,
    }


def assign_grade(g1: bool, g2: bool | None, g3: bool, g4: bool) -> str:
    """D = G1 또는 G2 탈락 · C = G2 통과, G3 탈락 · B = G2·G3 통과, G4 탈락 · A = 전부 통과(§5.5)."""
    if not g1 or not g2:
        return "D"
    if not g3:
        return "C"
    return "B" if not g4 else "A"


def grade_of(gates: dict) -> dict:
    """게이트 dict로 등급을 정한다. G2가 "검정 안 함"이면 등급 칸에 "검정 안 함(앞 단계 탈락)"을 적는다.

    G1 탈락(표본 부족)은 ``grade`` 를 ``G1_FAIL_GRADE_TEXT`` 로 적고 규칙상 등급(D)은 ``grade_rule`` 에 둔다(CI-G1-GRADE).
    """
    g1, g2, g3, g4 = gates["g1"], gates["g2"], gates["g3"], gates["g4"]
    if g2["pass"] is None:
        return {
            "grade": TEST_SKIPPED,
            "grade_rule": None,
            "stop_proposal": False,
            "note": "점추정·구간만 기록",
        }
    rule = assign_grade(g1["pass"], g2["pass"], g3["pass"], g4["pass"])
    if not g1["pass"]:
        return {
            "grade": G1_FAIL_GRADE_TEXT,
            "grade_rule": rule,
            "stop_proposal": True,
            "note": "G1 탈락, 규칙상 D",
        }
    return {"grade": rule, "grade_rule": rule, "stop_proposal": rule in ("C", "D"), "note": None}


def summary_verdict(results: dict[str, dict]) -> dict:
    """부분 C 요약 판정(§5.5). O1이 A 또는 B이면 채택 후보, 아니면 접기 제안(§10과 같은 한 문장).

    O1이 C·D거나 등급 없음(표본 부족)이면 접기 제안이다. O2~O4 중 A·B는 보조 근거로 같이 적는다."""
    o1 = results.get("O1")
    o1_grade = o1["grade"] if o1 else None
    adopt = o1_grade in ("A", "B")
    aux = [k for k in ("O2", "O3", "O4") if k in results and results[k]["grade"] in ("A", "B")]
    return {
        "o1_grade": o1_grade,
        "verdict": ADOPT_TEXT if adopt else FOLD_TEXT,
        "adopt_candidate": adopt,
        "stop_proposal": not adopt,
        "stop_text": None if adopt else STOP_TEXT,
        "aux_ab_outcomes": aux,
        "aux_has_ab": bool(aux),
    }


# ---------------------------------------------------------------- 결과 변수 하나
def outcome_stats(
    df: pl.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    cut: str = DEFAULT_CUT,
) -> dict:
    """결과 변수 하나의 통계량(§5.4). 게이트·등급은 ``judge_outcomes`` 가 붙인다."""
    d, n_drop = _clean(df)
    n_ev = int(d["event"].sum()) if d.height else 0
    risk = d["risk"].to_numpy()
    ev = d["event"].to_numpy()
    boot = cluster_bootstrap_auc(d, n_boot=n_boot, seed=seed)
    return {
        "n_rows": d.height,
        "n_dropped": n_drop,
        "n_events": n_ev,
        "n_corps": int(d["corp_code"].n_unique()) if d.height else 0,
        "base_rate": float(n_ev / d.height) if d.height else float("nan"),
        "pooled_auc": auc(risk, ev),
        "by_year": yearly_auc(d),
        "quintiles": quintile_rates(d, cut=cut),
        "capture": {f"{int(round(f * 100))}": capture(d, f, cut=cut) for f in CAPTURE_FRACS},
        "bootstrap": boot_summary(boot),
        "leave_one_year_out": leave_one_year_out(d),
        "seed": seed,
    }


def judge_outcomes(
    frames: dict[str, pl.DataFrame],
    seed: int = SEED,
    n_boot: int = N_BOOT,
    cut: str = DEFAULT_CUT,
    g4_denom: str = DEFAULT_G4_DENOM,
    alpha: float = ALPHA,
) -> dict:
    """결과 변수 묶음 판정. ``frames`` 키는 O1~O4(없는 것은 건너뜀), 순서는 O1→O4로 고정.

    반환::

        {"order": [...], "seed", "n_boot", "cut", "g4_denom", "alpha",
         "outcomes": {O: {"stats": {...}, "test": {...}, "gates": {"g1","g2","g3","g4"},
                          "grade", "grade_rule", "stop_proposal", "note"}},
         "summary": {...}}
    """
    unknown = set(frames) - set(OUTCOME_ORDER)
    if unknown:
        raise ValueError(f"unknown outcome keys: {sorted(unknown)}")
    names = [o for o in OUTCOME_ORDER if o in frames]
    stats = {o: outcome_stats(frames[o], n_boot=n_boot, seed=seed, cut=cut) for o in names}
    seq = fixed_sequence({o: stats[o]["bootstrap"]["p_value"] for o in names}, alpha=alpha)
    outcomes: dict[str, dict] = {}
    for o in names:
        s = stats[o]
        gates = {
            "g1": g1_gate(s["n_events"]),
            "g2": g2_gate(s["pooled_auc"], seq[o]),
            "g3": g3_gate(s["capture"]["20"]["lift"]),
            "g4": g4_gate(s["by_year"], denom=g4_denom),
        }
        outcomes[o] = {"stats": s, "test": seq[o], "gates": gates, **grade_of(gates)}
    return {
        "version": JUDGE_VERSION,
        "order": names,
        "seed": seed,
        "n_boot": n_boot,
        "cut": cut,
        "g4_denom": g4_denom,
        "alpha": alpha,
        "outcomes": outcomes,
        "summary": summary_verdict(outcomes),
    }


# ---------------------------------------------------------------- 기록 칸
def record_cells(
    cells: dict[str, dict[str, pl.DataFrame]],
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> dict:
    """차원별(C1·C2·C3·F) × O1~O4 16칸의 AUC와 95% 구간(§5.4, §9 기록 칸). 판정에 안 쓴다.

    ``cells[차원][결과]`` = ``[corp_code, fy, score, event]`` 프레임. 구간은 회사 클러스터 부트스트랩 백분위."""
    out: dict[str, dict] = {}
    for dim, per in cells.items():
        out[dim] = {}
        for o in OUTCOME_ORDER:
            if o not in per:
                continue
            d, n_drop = _clean(per[o])
            b = boot_summary(cluster_bootstrap_auc(d, n_boot=n_boot, seed=seed))
            out[dim][o] = {
                "auc": (
                    auc(d["risk"].to_numpy(), d["event"].to_numpy()) if d.height else float("nan")
                ),
                "ci95": b["ci95"],
                "n_rows": d.height,
                "n_dropped": n_drop,
                "n_events": int(d["event"].sum()) if d.height else 0,
                "n_boot_nan": b["n_nan"],
            }
    return out


# ---------------------------------------------------------------- manifest
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


def code_sha256(path: str | Path | None = None) -> str:
    """코드 파일의 sha256. 기본은 이 모듈."""
    p = Path(path) if path else Path(__file__)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_manifest(path: str | Path, result: dict, extra: dict | None = None) -> dict:
    """판정 결과와 시드·B·코드 sha256을 JSON으로 쓴다(§12.2). 쓰기만 하고 쓴 manifest dict를 돌려준다.

    ``extra`` 는 호출한 쪽이 더하는 칸(snapshot·입력 sha256·사전등록 sha256 등)이다."""
    man = {
        "version": JUDGE_VERSION,
        "seed": result.get("seed"),
        "n_boot": result.get("n_boot"),
        "code_sha256": code_sha256(),
        "result": result,
    }
    if extra:
        man["extra"] = extra
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(man), ensure_ascii=False, indent=2), encoding="utf-8")
    return man
