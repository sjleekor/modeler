"""ETF E3 보유 비용 — 실행 진입점 (사양 §6~§8).

두 갈래다.

``prefreeze`` (동결 전에 돈다): 개발 구간 분배락일 점검(§4), 판정 구간 **입력 존재 개수**, 검정력 모의(§7).
판정 구간의 G·H·RTR·C·β̂·이득·보수 값은 계산하지도 출력하지도 않는다(이 경로는 값 함수를 부르지 않는다).

``judge`` (동결 뒤에만 돈다): 태그 ``etf-holding-cost-e3-frozen`` 이 modeler 저장소에 있고 HEAD가 그 태그를 품을 때만
돈다(``require_frozen``). 아니면 거절한다. 동결 전에는 이 진입점을 돌리지 않는다.

    STOCK_DATA_ROOT=../stock_data PYTHONPATH=src python -m modeler.scores.quality.etf_e3_run prefreeze
    STOCK_DATA_ROOT=../stock_data PYTHONPATH=src python -m modeler.scores.quality.etf_e3_run judge

경로는 환경변수(기본 ``../stock_data``). 산출물은 ``stock_data/kr/output/etf_holding_cost_e3_<종류>_<날짜>/`` 에
manifest(입력 sha256·코드 커밋·시드)와 함께 쓴다. 저장소 안에는 아무것도 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.quality import etf_e3_exdate as ex
from modeler.scores.quality import etf_e3_fees as fees
from modeler.scores.quality import etf_e3_inputs as inp
from modeler.scores.quality import etf_e3_power as pw
from modeler.scores.quality import etf_e3_sample as sm
from modeler.scores.quality import etf_e3_stat as st

FROZEN_TAG = "etf-holding-cost-e3-frozen"

NOT_IMPLEMENTED = {
    "R1_unexplained_difference": "TODO 기록용 — 보수로 설명 안 되는 차이(R1). 판정 안 넣음",
    "R7_general_account_15.4": "TODO 기록용 — 일반 계좌 15.4% 세후",
    "R8_close_based": "TODO 기록용 — 종가 기준 수익률",
    "R9_index_tr_tracking": "TODO 기록용 — 지수 TR 절대 추적차이",
    "R11_delisted_partial_window": "TODO 기록용 — 폐지 ETF 부분 창",
    "R12_baseline": "TODO 기록용 — 기준선",
    "R13_forward_confirmation": "TODO 전진 확인(2025 형성)은 2027-01 이후",
    "delisted_merged_by_F_band": "TODO 기록용 — 폐지·합병 ETF-연 수를 F 구간별로(F 구간 문턱 사양에 없음)",
    "foreign_exposure_nav_lag": "구현은 국내 노출만 쓴 변형(region=domestic)으로 대신함",
}


class FrozenGuardError(RuntimeError):
    """동결 태그가 없거나 HEAD가 품지 않을 때."""


def _git(repo: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip()


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "uv.lock").exists():
            return d
    return here.parents[4]


def require_frozen(repo: Path | None = None, tag: str = FROZEN_TAG) -> str:
    """태그가 있고 HEAD가 그 태그의 커밋을 품으면 태그 커밋을 돌려준다. 아니면 ``FrozenGuardError``."""
    repo = repo or repo_root()
    rc, sha = _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    if rc != 0 or not sha:
        raise FrozenGuardError(
            f"판정 거절: 태그 {tag} 가 {repo} 에 없다. 동결(태그)을 먼저 하십시오."
        )
    rc, _ = _git(repo, "merge-base", "--is-ancestor", sha, "HEAD")
    if rc != 0:
        raise FrozenGuardError(f"판정 거절: HEAD가 태그 {tag}({sha[:8]})를 품고 있지 않다.")
    return sha


# ---------------------------------------------------------------- manifest
def manifest(out: Path, ctx: sm.Ctx | None, purpose: str, extra: dict) -> dict:
    repo = repo_root()
    _, head = _git(repo, "rev-parse", "HEAD")
    _, branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _, status = _git(repo, "status", "--porcelain")
    mods = {}
    for m in ("inputs", "dense", "exdate", "fees", "sample", "returns", "stat", "power", "run"):
        f = Path(__file__).with_name(f"etf_e3_{m}.py")
        if f.exists():
            mods[f.name] = inp.sha256_file(f)
    m = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": purpose,
        "version": inp.E3_VERSION,
        "inputs_sha256": ctx.input_sha if ctx else {},
        "frozen_constants": {
            "exdate_rule": ex.FROZEN_RULE,
            "nav_shift": ex.FROZEN_NAV_SHIFT,
            "fee_lag_days": fees.LAG_DAYS,
            "seeds": {
                "main": st.SEED,
                "boot": st.SEED + 1,
                "beta": st.SEED + 2,
                "gain": st.SEED + 3,
            },
            "n_perm": st.B_PERM,
            "n_boot": st.B_BOOT,
        },
        "modules_sha256": mods,
        "code": {
            "repo": "modeler",
            "branch": branch,
            "commit": head,
            "git_status_porcelain": status.splitlines(),
            "worktree_dirty": bool(status),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "polars": pl.__version__,
            "numpy": np.__version__,
        },
        **extra,
        "outputs": sorted(p.name for p in out.glob("*") if p.name != "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, default=str))
    return m


def _nan(o):
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, dict):
        return {str(k): _nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _nan(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(o, path: Path) -> None:
    path.write_text(json.dumps(_nan(o), ensure_ascii=False, indent=2, default=str))


# ================================================================ 동결 전: 개수만
def structure_counts(ey: pl.DataFrame) -> dict:
    g = ey.group_by("year", "gkey").len()
    n = g["len"]
    return {
        "etf_years": int(ey.height),
        "group_years": int(g.height),
        "groups": int(g["gkey"].n_unique()),
        "pairs": int((n * (n - 1) // 2).sum()),
    }


def prefreeze_counts(ctx: sm.Ctx, years=sm.JUDGMENT_YEARS) -> dict:
    """판정 구간 입력 존재 개수. 값(수익률·보수·분배 크기)은 쓰지 않는다. 연도별·단계별 개수만 돌려준다."""
    s = sm.build_sample(ctx, years)
    ey = s.ey
    out: dict = {"start_structure": structure_counts(ey), "exclusion_counts": s.counts.to_dicts()}
    lvl = Counter(ctx.code_level.get(c, "none") for c in ey["isu_cd"].to_list())
    out["start_dis_map_level_etf_years"] = dict(lvl)

    # 분배락일 정합성: 개수 구간별
    def bucket(n, r):
        if n == 0 or r is None:
            return "no_distribution"
        if r >= 1.0:
            return "100%"
        if r >= ex.CONSISTENCY_MIN:
            return "90~<100%"
        if r >= 0.5:
            return "50~<90%"
        return "<50%"

    bk = Counter()
    for y, n, r in zip(ey["year"].to_list(), ey["n_dist"].to_list(), ey["ratio"].to_list()):
        bk[(y, bucket(n, r))] += 1
    out["exdate_consistency_buckets"] = [
        {"year": y, "bucket": b, "n_etf_years": n} for (y, b), n in sorted(bk.items())
    ]
    # F/A 존재(값은 안 읽음)
    fa = (
        ey.group_by("year")
        .agg(
            pl.len().alias("n_etf_years"),
            pl.col("f_available").sum().alias("n_f_available"),
            (pl.col("a_months") >= fees.A_MIN_MONTHS).sum().alias("n_a_months_ge10"),
            (pl.col("f_available") & (pl.col("a_months") >= fees.A_MIN_MONTHS))
            .sum()
            .alias("n_f_and_a"),
            pl.col("x_dis_map").sum().alias("n_no_dis_map"),
        )
        .sort("year")
    )
    out["fee_availability"] = fa.to_dicts()
    out["_sample"] = s
    return out


def run_dev_check(ctx: sm.Ctx, root: Path) -> dict:
    """§4 개발 구간 점검. 판정 구간 행은 쓰지 않는다(RGT_STD_DT ≤ 2015-12-31, 청산분배·금액 0 제외)."""
    d = ctx.dense
    seibro, cats = inp.load_seibro(root / inp.SEIBRO_REL)
    dev = ex.dev_distributions(d, inp.distributions_for_use(seibro))
    adj = ex.krx_adjust_days(d)
    rates = ex.candidate_match_rates(d, adj, dev)
    choice = ex.choose_rule(rates)
    nav = {r: ex.nav_alignment(d, dev, r) for r in ex.RULE_SHIFT}
    nav_chosen = nav[choice["chosen"]] if choice["chosen"] else None
    return {
        "seibro_categories": cats,
        "dev_distributions": int(dev.height),
        "candidate_match": rates,
        "choice": choice,
        "nav_alignment_by_candidate": nav,
        "nav_alignment_chosen": nav_chosen,
        "frozen_in_code": {"rule": ex.FROZEN_RULE, "nav_shift": ex.FROZEN_NAV_SHIFT},
        "frozen_matches_dev_check": bool(
            choice["chosen"] == ex.FROZEN_RULE
            and nav_chosen is not None
            and nav_chosen["majority_offset"] == ex.FROZEN_NAV_SHIFT
        ),
    }


def run_prefreeze(out: Path, root: Path, n_rep: int, workers: int, verify: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ctx = sm.load_ctx(root, verify=verify)
    dev = run_dev_check(ctx, root)
    write_json(dev, out / "a_exdate_dev_check.json")
    cnt = prefreeze_counts(ctx)
    sample = cnt.pop("_sample")
    write_json(cnt, out / "b_input_existence_counts.json")
    sample.counts.write_csv(out / "b_exclusion_counts_by_year.csv")
    # c) 검정력: 구조는 개수만
    struct = pw.structure_from_counts(sample.ey)
    sk = pw.skeleton_from_structure(struct)
    rows = pw.run_power(sk, pw.scenario_list(), n_rep=n_rep, workers=workers)
    pw.write_rows(rows, out / "c_power_table.csv")
    manifest(
        out,
        ctx,
        "E3 동결 전: 개발 구간 분배락일 점검 · 판정 구간 입력 존재 개수 · 검정력 모의. 판정 구간 G·H·RTR·C·β̂·이득·보수 값 없음.",
        {
            "power": {
                "n_rep": n_rep,
                "n_scenarios": len(rows),
                "sim_perm": pw.SIM_PERM,
                "sim_boot": pw.SIM_BOOT,
                "structure": structure_counts_from_struct(struct),
                "base_seed": 20261012,
            },
            "not_computed": [
                "G",
                "H",
                "RTR",
                "C",
                "C_a",
                "beta_hat",
                "gain",
                "TER 분포",
                "쌍 차이",
            ],
        },
    )
    return {"dev": dev, "counts": cnt, "power": rows}


def structure_counts_from_struct(struct) -> dict:
    sizes = [s[2] for s in struct]
    return {
        "group_years": len(struct),
        "groups": len({s[0] for s in struct}),
        "etf_years": int(sum(sizes)),
        "pairs": int(sum(n * (n - 1) // 2 for n in sizes)),
        "size_distribution": dict(sorted(Counter(sizes).items())),
    }


# ================================================================ 동결 뒤: 판정
def _month_end_after(std: date) -> date:
    """기준일 ``std`` 다음 달의 달력 말일."""
    y, m = (std.year + 1, 1) if std.month == 12 else (std.year, std.month + 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def lags_from_watch(path: str | Path) -> dict:
    """하루 1건 조회 기록(watch.jsonl) → 지연 실측.

    - 기준일(standardDt)마다 처음 나온 날 − 기준일 = 지연(일). **단 첫 조회에서 이미 보인 기준일은 빼고**(그 전에 공개돼
      지연의 상한일 뿐이다), 앞선 조회에서 더 이른 기준일이 보였던 경우만 센다. 쉬는 날(10-15~18) 뒤 처음 보이면
      처음 본 날로 센다(지연을 크게 잡는 쪽).
    - ``censored_over``: 마지막으로 보인 기준일의 다음 달 말일이 아직 안 나왔으면, 마지막 조회일 − 그 말일(일).
    메인 정정(10-11): 처음 판은 첫 조회의 기준일(이미 공개된 08-31)을 "그날 처음 나옴"으로 세어 가짜 지연을 만들었다.
    """
    obs: list[tuple[str, str]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "std" and r.get("standardDt"):
            obs.append((r["day"], r["standardDt"]))
    obs.sort()
    first: dict[str, str] = {}
    seen_before: set[str] = set()
    lags = []
    for k, (day, std) in enumerate(obs):
        if std not in first:
            first[std] = day
            if k > 0 and any(s < std for s in seen_before):
                d0 = datetime.strptime(std, "%Y%m%d").date()
                lags.append((date.fromisoformat(day) - d0).days)
        seen_before.add(std)
    censored = None
    if obs:
        last_day, _ = obs[-1]
        latest = max(s for _, s in obs)
        pending = _month_end_after(datetime.strptime(latest, "%Y%m%d").date())
        el = (date.fromisoformat(last_day) - pending).days
        censored = el if el > 0 else None
    return {"lags": lags, "censored_over": censored, "n_obs": len(obs)}


def build_inputs(
    ctx: sm.Ctx, sample: sm.Sample, fee_field: str = "ter", lag_days: int = fees.LAG_DAYS
) -> dict:
    """판정 값 계산(동결 뒤 전용). E3b′·E3a(C_a) 의 ``StatData`` 와 β̂·이득 입력."""
    from modeler.scores.quality import etf_e3_returns as rt

    vals = [rt.gy_values(ctx, gy, fee_field, lag_days) for gy in sample.gys]
    keys = sorted({gy.key for gy in sample.gys})
    gid = {k: i for i, k in enumerate(keys)}
    sizes = [len(gy.cols) for gy in sample.gys]
    years = [gy.year for gy in sample.gys]
    group = [gid[gy.key] for gy in sample.gys]
    sd_b = st.make_stat_data(
        sizes,
        years,
        group,
        keys,
        [v.G - v.G.T for v in vals],
        [v.H for v in vals],
        [gy.pair_ok for gy in sample.gys],
    )
    sd_a = st.make_stat_data(
        sizes,
        years,
        group,
        keys,
        [v.F[None, :] - v.F[:, None] for v in vals],
        [v.R for v in vals],
        [gy.pair_ok for gy in sample.gys],
        score_thr=st.F_THR_PCT,
    )
    xs, ys, Fs, Rs = [], [], [], []
    for v in vals:
        ok = ~v.own_lost & np.isfinite(v.A) & np.isfinite(v.own_y1) & np.isfinite(v.F)
        xs.append(-v.A[ok] / 100.0)
        ys.append(v.own_y1[ok])
        Fs.append(v.F[ok])
        Rs.append(v.own_y1[ok])
    return {
        "vals": vals,
        "keys": keys,
        "group": np.array(group),
        "sd_b": sd_b,
        "sd_a": sd_a,
        "xs": xs,
        "ys": ys,
        "Fs": Fs,
        "Rs": Rs,
    }


def _modal_operator_groups(sample: sm.Sample, vals) -> tuple[np.ndarray, list[str]]:
    labels = []
    for v in vals:
        ops = [o for o in v.operator if o]
        labels.append(Counter(ops).most_common(1)[0][0] if ops else "?")
    keys = sorted(set(labels))
    return np.array([keys.index(x) for x in labels]), keys


def freq_class(n: int) -> str:
    return "monthly" if n >= 10 else "quarterly" if n >= 3 else "annual" if n >= 1 else "none"


def judge_all(
    ctx: sm.Ctx,
    years=sm.JUDGMENT_YEARS,
    n_perm: int = st.B_PERM,
    n_boot: int = st.B_BOOT,
    seed: int = st.SEED,
    lag: dict | None = None,
    variants: bool = True,
) -> dict:
    """판정 계산 전부(가드 없음 — 호출하는 ``run_judgment`` 가 가드를 건다). 합성 자료 테스트도 이 함수를 부른다."""
    sample = sm.build_sample(ctx, years)
    inpt = build_inputs(ctx, sample)
    sd_b, sd_a = inpt["sd_b"], inpt["sd_a"]
    # E3a (기록용 첫 칸)
    beta = st.beta_hat(inpt["xs"], inpt["ys"], sd_b.group, n_boot, seed + 2)
    gain = st.lowest_fee_gain(inpt["Fs"], inpt["Rs"], sd_b.group, n_boot, seed + 3)
    ca = st.c_only(sd_a, n_boot, seed)
    # E3b′ (판정)
    r = st.judge_c(sd_b, n_perm, n_boot, seed)
    g = r["gates"]
    lag = lag or fees.lag_summary(None)
    gr = st.grade(g, cap_1a=beta["rule_1a"], lag=lag)
    res = {
        "E3a_record": {"beta_hat": beta, "lowest_fee_gain": gain, "C_a": ca},
        "E3b_prime": {
            "C": r["C"],
            "p_perm": r["p"],
            "n_pairs_cond": r["n_pairs_cond"],
            "n_groups_with_pairs": r["n_groups_with_pairs"],
            "gates": g,
            "grade": gr,
            "C_by_year": st.c_by_year(sd_b, r["rs"]),
        },
        "lag": lag,
        "sample_counts": sample.counts.to_dicts(),
        "not_implemented": NOT_IMPLEMENTED,
    }
    if variants:
        res["records"] = record_variants(ctx, years, sample, inpt, r["rs"], n_boot, seed)
    return res


def record_variants(ctx, years, sample, inpt, rs, n_boot, seed) -> dict:
    """기록용 변형(m에 안 넣음). C 와 부트스트랩 구간만 낸다."""
    out: dict = {}
    sd_b = inpt["sd_b"]
    vals = inpt["vals"]

    def run(label, **kw):
        s2 = sm.build_sample(
            ctx,
            years,
            **{
                k: v
                for k, v in kw.items()
                if k in ("key_mode", "min_group", "fee_field", "lag_days", "region")
            },
        )
        if not s2.gys:
            out[label] = {"note": "표본 없음"}
            return
        i2 = build_inputs(ctx, s2, kw.get("fee_field", "ter"), kw.get("lag_days", fees.LAG_DAYS))
        out[label] = st.c_only(i2["sd_b"], n_boot, seed)

    run("group_size_ge3", min_group=3)
    run("group_size_ge5", min_group=5)
    run("key_without_return_type_R4", key_mode="no_rt")
    run("fee_lag_61d", lag_days=61)
    run("fee_lag_0d", lag_days=0)
    run("fee_sum_fee", fee_field="sum_fee")
    run("fee_ter_plus_brokerage", fee_field="ter+brokerage")
    run("domestic_only", region="domestic")
    # 운용사 클러스터(그룹-연도의 최빈 운용사로 묶는다)
    grp, ops = _modal_operator_groups(sample, vals)
    sd_op = dataclasses.replace(sd_b, group=grp, group_keys=ops)
    ng, wg = st.group_sums(rs["num"], rs["w"], sd_op)
    bo = st.cluster_boot(ng, wg, n_boot, seed + 1)
    b = bo[np.isfinite(bo)]
    out["operator_cluster_bootstrap"] = {
        "note": "그룹-연도의 최빈 운용사로 묶은 근사",
        "boot_lower_95": float(np.percentile(b, 5)) if len(b) else None,
        "boot_upper_95": float(np.percentile(b, 95)) if len(b) else None,
    }
    # 분배 빈도가 섞인 그룹-연도
    d = ctx.dense
    dist = ctx.dist
    mixed = []
    for gy in sample.gys:
        s0, e0 = d.year_range(gy.year)
        cls = []
        for c in gy.cols:
            n = int(
                (
                    (dist["j"].to_numpy() == c)
                    & (dist["ri"].to_numpy() >= s0)
                    & (dist["ri"].to_numpy() <= e0)
                ).sum()
            )
            cls.append(freq_class(n))
        mixed.append(len(set(cls)) > 1)
    mixed = np.array(mixed)
    out["distribution_frequency_mixed_groups"] = {
        "n_group_years_mixed": int(mixed.sum()),
        "C_mixed": st.subset_c(rs["num"], rs["w"], mixed),
        "C_unmixed": st.subset_c(rs["num"], rs["w"], ~mixed),
    }
    # 보수 변경 ETF-연 수와 폭(연도별)
    chg: dict[int, list[float]] = {}
    tot: Counter = Counter()
    for gy, v in zip(sample.gys, vals):
        for c, isu in zip(gy.cols, gy.isu):
            fund = ctx.code_map.get(isu)
            lo, hi = fees.outcome_range(ctx.fee, fund, gy.year + 1)
            tot[gy.year] += 1
            if np.isfinite(lo) and hi > lo:
                chg.setdefault(gy.year, []).append((hi - lo) * 100.0)
    out["fee_changed_in_outcome_year"] = {
        int(y): {
            "n_etf_years": int(tot[y]),
            "n_changed": len(chg.get(y, [])),
            "width_bp_median": float(np.median(chg[y])) if y in chg else None,
            "width_bp_max": float(np.max(chg[y])) if y in chg else None,
        }
        for y in sorted(tot)
    }
    return out


def run_judgment(
    out: Path,
    root: Path,
    lag_watch: str | None,
    repo: Path | None = None,
    n_perm: int = st.B_PERM,
    n_boot: int = st.B_BOOT,
) -> dict:
    """판정 진입점. **태그 ``etf-holding-cost-e3-frozen`` 가 없으면 거절한다.**"""
    tag_commit = require_frozen(repo)
    out.mkdir(parents=True, exist_ok=True)
    ctx = sm.load_ctx(root, verify=True)
    if lag_watch:
        lw = lags_from_watch(lag_watch)
        lag = fees.lag_summary(lw["lags"], lw["censored_over"])
    else:
        lag = fees.lag_summary(None)
    res = judge_all(ctx, n_perm=n_perm, n_boot=n_boot, lag=lag)
    write_json(res, out / "judgment_results.json")
    manifest(
        out,
        ctx,
        "E3 판정(동결 뒤). E3a 기록 첫 칸 + E3b′ 판정.",
        {"frozen_tag": FROZEN_TAG, "frozen_tag_commit": tag_commit, "lag_watch": lag_watch},
    )
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("prefreeze", help="동결 전: 개발 구간 점검·입력 존재 개수·검정력")
    pf.add_argument("--out-dir", default=None)
    pf.add_argument("--n-rep", type=int, default=300)
    pf.add_argument("--workers", type=int, default=6)
    pf.add_argument("--no-verify", action="store_true")
    jd = sub.add_parser("judge", help="동결 뒤에만: 판정 (태그 가드)")
    jd.add_argument("--out-dir", default=None)
    jd.add_argument("--lag-watch", default=None, help="DIS 하루 1건 조회 기록(watch.jsonl) 사본")
    a = ap.parse_args(argv)
    root = inp.stock_data_root()
    today = datetime.now().strftime("%Y%m%d")
    if a.cmd == "prefreeze":
        out = Path(a.out_dir or root / f"kr/output/etf_holding_cost_e3_prefreeze_{today}")
        run_prefreeze(out, root, a.n_rep, a.workers, verify=not a.no_verify)
        print(f"완료: {out}")
        return 0
    out = Path(a.out_dir or root / f"kr/output/etf_holding_cost_e3_judgment_{today}")
    try:
        run_judgment(out, root, a.lag_watch)
    except FrozenGuardError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"완료: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
