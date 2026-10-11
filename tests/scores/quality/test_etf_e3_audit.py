"""E3 감사 보강: 거절 경로, 지연 등급 end-to-end, 쌍 규칙·순열, watch 경계, 문턱 경계, own_lost."""

import itertools
import json
import subprocess
from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e3_returns as rt
from modeler.scores.quality import etf_e3_run as run
from modeler.scores.quality import etf_e3_sample as sm
from modeler.scores.quality import etf_e3_stat as st

from .e3_synth import DAYS, make_ctx, quarter_end_idx

BASE = dict(idx="합성지수", nav_noise=0.0005)


def git(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
    ).stdout.strip()


def mkrepo(path, files):
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "t@t")
    git(path, "config", "user.name", "t")
    for n, c in files.items():
        (path / n).write_text(c)
        git(path, "add", n)
    git(path, "commit", "-qm", "c1")
    return path


@pytest.fixture()
def world(tmp_path):
    mod = mkrepo(tmp_path / "modeler", {"a.txt": "1"})
    git(mod, "tag", run.FROZEN_TAG)
    my = mkrepo(tmp_path / "my", {"prereg.md": "v1"})
    git(my, "tag", run.FROZEN_TAG)
    lag = tmp_path / "watch.jsonl"
    lag.write_text("")
    return mod, my, lag


class Loaded(Exception):
    pass


def call(world, tmp_path, **kw):
    mod, my, lag = world
    args = dict(lag_watch=lag, prereg=my / "prereg.md", repo=mod)
    args.update(kw)
    return run.run_judgment(
        tmp_path / "out", tmp_path / "root", args.pop("lag_watch"), args.pop("prereg"), **args
    )


def test_all_guards_pass_then_reaches_input_loading(world, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise Loaded

    monkeypatch.setattr(sm, "load_ctx", boom)
    with pytest.raises(Loaded):
        call(world, tmp_path)


def test_refuse_dirty_tree(world, tmp_path):
    mod = world[0]
    (mod / "a.txt").write_text("changed")
    with pytest.raises(run.FrozenGuardError, match="추적 파일 수정"):
        call(world, tmp_path)
    assert not (tmp_path / "out").exists()


def test_untracked_file_does_not_block(world, tmp_path, monkeypatch):
    (world[0] / "new.txt").write_text("x")
    monkeypatch.setattr(sm, "load_ctx", lambda *a, **k: (_ for _ in ()).throw(Loaded()))
    with pytest.raises(Loaded):
        call(world, tmp_path)


def test_refuse_missing_prereg(world, tmp_path):
    with pytest.raises(run.FrozenGuardError, match="--prereg"):
        call(world, tmp_path, prereg=None)


def test_refuse_prereg_differs_from_tag(world, tmp_path):
    my = world[1]
    (my / "prereg.md").write_text("v2 수정")
    with pytest.raises(run.FrozenGuardError, match="태그 시점 내용과 다르다"):
        call(world, tmp_path)


def test_refuse_prereg_without_tag_in_its_repo(world, tmp_path):
    git(world[1], "tag", "-d", run.FROZEN_TAG)
    with pytest.raises(run.FrozenGuardError, match="태그"):
        call(world, tmp_path)


def test_refuse_missing_lag_watch(world, tmp_path):
    with pytest.raises(run.FrozenGuardError, match="--lag-watch"):
        call(world, tmp_path, lag_watch=None)
    with pytest.raises(run.FrozenGuardError, match="--lag-watch"):
        call(world, tmp_path, lag_watch=tmp_path / "없는파일")


def test_parser_hash_mismatch(tmp_path):
    f = tmp_path / "etf_classify.py"
    f.write_text("# 바뀐 파서")
    with pytest.raises(run.FrozenGuardError, match="1e81d6a3"):
        run.check_parser_hash(f)
    assert run.check_parser_hash().startswith("1e81d6a3")  # 실제 동결 파서는 통과


def test_cli_requires_prereg_and_lag_watch():
    with pytest.raises(SystemExit):
        run.main(["judge"])


# ---------------------------------------------------------------- 지연 등급 end-to-end
def strong_ctx(tmp_path):
    etfs = {}
    for g in range(6):
        for k, a in enumerate((0.06, 0.02, -0.02)):
            etfs[f"{g}{k}A000"[:6]] = dict(
                name=f"KODEX G{g}{k}", alpha=a, idx=f"합성지수{g}", nav_noise=0.0003
            )
    return make_ctx(
        tmp_path, etfs, {k: 0.2 for k in etfs}
    )  # 보수 같음 → β̂ 정의 안 됨(1-a 안 걸림)


def test_lag_cap_lowers_g2_pass_to_c_with_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "G1_PAIRS", 1)
    monkeypatch.setattr(st, "G1_GROUPS", 1)
    monkeypatch.setattr(st, "SPLIT_YEAR", 2014)
    ctx = strong_ctx(tmp_path)
    ok = {"cap": False, "status": "실측"}
    res = run.judge_all(
        ctx, years=(2014, 2015, 2016), n_perm=199, n_boot=100, lag=ok, variants=False
    )
    assert res["E3b_prime"]["grade"]["grade"] == "A"
    capped = dict(ok, cap=True, max=93)
    res2 = run.judge_all(
        ctx, years=(2014, 2015, 2016), n_perm=199, n_boot=100, lag=capped, variants=False
    )
    g = res2["E3b_prime"]["grade"]
    assert g["grade"] == "C" and "지연 규칙 부족" in g["reason"]


# ---------------------------------------------------------------- 쌍 규칙과 순열
def brute_perm_values(sd):
    """순열 하나하나를 사양대로 직접 계산(그룹-연도 하나짜리). 정의 안 되면 nan."""
    n = int(sd.n[0])
    hm = sd.hmat_flat.reshape(n, n)
    out = set()
    for pi in itertools.permutations(range(n)):
        k = c = 0
        for a, b, gs in zip(sd.pa, sd.pb, sd.gsign):
            if gs == 0:
                continue
            hd = hm[pi[a], pi[b]] - hm[pi[b], pi[a]]
            if not np.isfinite(hd) or abs(hd) < sd.tie_thr - st.EPS:
                continue
            k += 1
            c += int(np.sign(hd) == gs)
        out.add(round(c / k, 9) if k else None)
    return out


def test_failed_pair_is_out_of_real_stat_and_every_permutation():
    n = 3
    G = np.array([0.0, 0.01, 0.02])
    # H[a, b]: 쌍 (0, 2) 는 Y+1 창 규칙 실패 → NaN. 쌍 (0, 1) 은 실제 통계량에서 빠진다(pair_valid False)
    H = np.array([[0.0, 0.0, np.nan], [0.01, 0.01, 0.01], [np.nan, 0.02, 0.02]])
    valid = np.ones((n, n), bool)
    valid[0, 1] = valid[1, 0] = False
    sd = st.make_stat_data([n], [2015], [0], ["g"], [np.subtract.outer(G, G)], [H], [valid])
    # 실제 통계량: 쌍 (0,1) 은 조건 아님, (0,2) 는 H 가 NaN → 빠짐, 남는 것 (1,2) 하나
    assert sd.gsign.tolist() == [0, -1, -1]  # 쌍 순서 (0,1)(0,2)(1,2)
    rs = st.real_stat(sd)
    assert rs["k"].tolist() == [1.0]
    _, cs = st.perm_pvalue(sd, rs["C"], n_perm=300, seed=3)
    expect = brute_perm_values(sd)
    got = {None if np.isnan(c) else round(float(c), 9) for c in cs}
    assert got <= expect  # NaN 쌍은 그 순열에서만 빠진다
    assert len(got) >= 2


def test_pair_failing_y1_window_is_dropped_in_sample(tmp_path):
    qe = quarter_end_idx(DAYS)
    e = next(i for i in qe if DAYS[i].year == 2016 and DAYS[i].month == 3)
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.05, **BASE),
        "002A00": dict(name="KODEX B", alpha=0.0, **BASE),
        "003A00": dict(name="KODEX C", alpha=-0.05, nav_missing={DAYS[e - 2]}, **BASE),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    s = sm.build_sample(ctx, (2015,))
    gy = s.gys[0]
    k = gy.isu.index("003A00")
    v = rt.gy_values(ctx, gy)
    assert not v.ok_y1[k, gy.isu.index("001A00")] and v.ok_y[k, gy.isu.index("001A00")]
    assert not gy.pair_ok[k, gy.isu.index("001A00")]  # 두 창 모두 통과해야 실제 쌍
    assert np.isnan(v.H[k, gy.isu.index("001A00")]) and np.isfinite(v.G[k, gy.isu.index("001A00")])


# ---------------------------------------------------------------- watch 경계
def write_watch(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def std(day, s):
    return {"kind": "std", "day": day, "standardDt": s}


def test_watch_empty_file(tmp_path):
    p = tmp_path / "w.jsonl"
    write_watch(p, [])
    assert run.lags_from_watch(p) == {"lags": [], "censored_over": None, "n_obs": 0}


def test_watch_error_rows_are_ignored_and_first_seen_excluded(tmp_path):
    p = tmp_path / "w.jsonl"
    write_watch(
        p,
        [
            std("2026-10-12", "20260831"),
            {"kind": "error", "day": "2026-10-13", "attempt": 1, "error": "x"},
            {"kind": "start", "at": "2026-10-12T12:00"},
            std("2026-10-14", "20260831"),
            std("2026-11-02", "20260930"),
        ],
    )
    r = run.lags_from_watch(p)
    assert r["lags"] == [33] and r["n_obs"] == 3  # 첫 조회의 08-31 은 지연으로 안 센다
    assert r["censored_over"] == 2  # 다음 달 말일 10-31 이 11-02 까지 안 나옴


def test_watch_first_seen_after_holiday_counts_from_first_seen_day(tmp_path):
    p = tmp_path / "w.jsonl"
    write_watch(
        p,
        [
            std("2026-10-12", "20260831"),
            std("2026-10-14", "20260831"),
            std("2026-10-19", "20260930"),
        ],
    )
    assert run.lags_from_watch(p)["lags"] == [19]  # 쉬는 날(10-15~18) 뒤 처음 본 날로 셈


def test_watch_non_month_end_standard_date(tmp_path):
    p = tmp_path / "w.jsonl"
    write_watch(p, [std("2026-10-12", "20260831"), std("2026-11-03", "20260929")])
    r = run.lags_from_watch(p)
    assert r["lags"] == [35]  # 09-29 기준 → 11-03 − 09-29
    # 다음 달(10월) 말일 10-31 이 11-03 까지 안 나왔으면 경과일 3일
    assert r["censored_over"] == 3


# ---------------------------------------------------------------- 문턱 경계
def with_missing(tmp_path, n_days, sub):
    days = [d for d in DAYS if d.year == 2015 and d.month == 2 and d.weekday() < 5][:n_days]
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.0, **BASE),
        "002A00": dict(name="KODEX B", alpha=0.0, nav_missing=set(days), **BASE),
        "003A00": dict(name="KODEX C", alpha=0.0, **BASE),
    }
    ctx = (
        make_ctx(tmp_path / sub, etfs, {k: 0.2 for k in etfs})
        if (tmp_path / sub).mkdir() is None
        else None
    )
    s = sm.build_sample(ctx, (2015,))
    return dict(zip(s.ey["isu_cd"], s.ey["excluded_at"]))


def test_nav_missing_exactly_10_passes_and_11_excluded(tmp_path):
    assert with_missing(tmp_path, 10, "a")["002A00"] == ""
    assert with_missing(tmp_path, 11, "b")["002A00"] == "3_nav_missing"


def test_outlier_threshold_30pct_inclusive():
    m = sm.outlier_mask(np.array([0.30, -0.30, 0.2999, -0.29, 0.7 / 1.0 - 1.0, np.nan]))
    assert m.tolist() == [True, True, False, False, True, False]


def test_consistency_exactly_90pct_passes():
    from modeler.scores.quality import etf_e3_exdate as ex

    assert not (9 / 10 < ex.CONSISTENCY_MIN - 1e-12)
    assert 0.89 < ex.CONSISTENCY_MIN - 1e-12


def test_g_5bp_and_h_half_bp_boundaries():
    # G 차이 정확히 5bp → 조건 충족, 4.9bp → 아님
    sd = st.make_stat_data(
        [2],
        [2015],
        [0],
        ["g"],
        [np.array([[0.0, -0.0005], [0.0005, 0.0]])],
        [np.array([[0.0, 0.0], [0.01, 0.01]])],
        [np.ones((2, 2), bool)],
    )
    assert sd.gsign.tolist() == [-1]
    sd2 = st.make_stat_data(
        [2],
        [2015],
        [0],
        ["g"],
        [np.array([[0.0, -0.00049], [0.00049, 0.0]])],
        [np.array([[0.0, 0.0], [0.01, 0.01]])],
        [np.ones((2, 2), bool)],
    )
    assert sd2.gsign.tolist() == [0]

    # H 차이 정확히 0.5bp 는 동률이 아니다, 0.49bp 는 동률
    def stat(hdiff):
        return st.real_stat(
            st.make_stat_data(
                [2],
                [2015],
                [0],
                ["g"],
                [np.array([[0.0, 0.1], [-0.1, 0.0]])],
                [np.array([[0.0, 0.0], [hdiff, hdiff]])],
                [np.ones((2, 2), bool)],
            )
        )

    assert stat(0.00005)["k"].tolist() == [1.0]
    assert stat(0.000049)["k"].tolist() == [0.0]


# ---------------------------------------------------------------- own_lost
def test_own_lost_etf_year_leaves_beta_and_gain_samples(tmp_path):
    qe = quarter_end_idx(DAYS)
    e = next(i for i in qe if DAYS[i].year == 2016 and DAYS[i].month == 3)
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.05, **BASE),
        "002A00": dict(name="KODEX B", alpha=0.0, **BASE),
        "003A00": dict(name="KODEX C", alpha=-0.05, nav_missing={DAYS[e - 2]}, **BASE),
    }
    ctx = make_ctx(tmp_path, etfs, {"001A00": 0.1, "002A00": 0.3, "003A00": 0.5})
    s = sm.build_sample(ctx, (2015,))
    inp = run.build_inputs(ctx, s)
    v = inp["vals"][0]
    k = s.gys[0].isu.index("003A00")
    assert v.own_lost[k] and not v.own_lost[s.gys[0].isu.index("001A00")]
    assert len(inp["xs"][0]) == 2 and len(inp["Fs"][0]) == 2 and len(inp["ys"][0]) == 2
    assert pl.Series(inp["Fs"][0]).min() == pytest.approx(0.1)


def test_min_group_applies_after_exclusions(tmp_path):
    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.05, **BASE),
        "002A00": dict(name="KODEX B", alpha=0.0, **BASE),
        "003A00": dict(
            name="KODEX C", alpha=0.0, idx_before="옛지수", idx_change_at=date(2015, 6, 1), **BASE
        ),
    }
    ctx = make_ctx(tmp_path, etfs, {k: 0.2 for k in etfs})
    assert len(sm.build_sample(ctx, (2015,), min_group=2).gys) == 1
    s3 = sm.build_sample(ctx, (2015,), min_group=3)  # 시작 3 → 제외 뒤 2 < 3 이라 빠진다
    assert s3.counts["n_start"][0] == 3 and s3.gys == []
