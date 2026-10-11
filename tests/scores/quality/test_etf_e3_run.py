"""E3 실행: 동결 태그 가드(거절), 동결 전 경로가 값 함수를 안 부르는지, 합성 자료 판정 일주."""

import subprocess

import pytest

from modeler.scores.quality import etf_e3_returns as rt
from modeler.scores.quality import etf_e3_run as run
from modeler.scores.quality import etf_e3_stat as st

from .e3_synth import make_ctx


def git(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("1")
    git(r, "add", "a.txt")
    git(r, "commit", "-qm", "c1")
    return r


def test_guard_refuses_without_tag(repo):
    with pytest.raises(run.FrozenGuardError, match="태그"):
        run.require_frozen(repo)


def test_guard_accepts_tag_contained_in_head(repo):
    git(repo, "tag", run.FROZEN_TAG)
    sha = run.require_frozen(repo)
    assert sha == git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("2")
    git(repo, "commit", "-qam", "c2")  # HEAD 가 태그 뒤로 가도 품고 있다
    assert run.require_frozen(repo) == sha


def test_guard_refuses_when_head_does_not_contain_tag(repo):
    git(repo, "checkout", "-qb", "side")
    (repo / "a.txt").write_text("side")
    git(repo, "commit", "-qam", "side")
    git(repo, "tag", run.FROZEN_TAG)  # 태그는 side 가지에
    git(repo, "checkout", "-q", "main")
    with pytest.raises(run.FrozenGuardError, match="품고 있지 않다"):
        run.require_frozen(repo)


def test_judgment_entry_refuses_and_writes_nothing(repo, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(run.FrozenGuardError):
        run.run_judgment(out, tmp_path / "no_such_root", None, repo=repo)
    assert not out.exists()  # 입력을 읽기 전에, 출력 디렉터리를 만들기 전에 거절한다


def test_prefreeze_counts_never_calls_value_functions(tmp_path, monkeypatch):
    from .e3_synth import make_ctx as mk

    etfs = {
        "001A00": dict(name="KODEX A", alpha=0.04, idx="합성지수", nav_noise=0.0005),
        "002A00": dict(name="KODEX B", alpha=0.0, idx="합성지수", nav_noise=0.0005),
    }
    ctx = mk(tmp_path, etfs, {k: 0.2 for k in etfs})

    def boom(*a, **k):
        raise AssertionError("동결 전 경로가 값 함수를 불렀다")

    for name in ("pair_rtr", "own_rtr", "gy_values"):
        monkeypatch.setattr(rt, name, boom)
    for name in ("real_stat", "judge_c", "perm_pvalue", "beta_hat", "lowest_fee_gain", "c_only"):
        monkeypatch.setattr(st, name, boom)
    monkeypatch.setattr(run, "judge_all", boom)
    out = run.prefreeze_counts(ctx, years=(2014, 2015))
    assert out["start_structure"]["etf_years"] == 4
    assert {r["year"] for r in out["fee_availability"]} == {2014, 2015}
    assert all(set(r) >= {"n_f_available", "n_a_months_ge10"} for r in out["fee_availability"])
    # 개수만: 구간 이름과 개수
    assert all(
        set(r) == {"year", "bucket", "n_etf_years"} for r in out["exdate_consistency_buckets"]
    )


def test_judge_all_on_synthetic(tmp_path):
    etfs = {}
    for g in range(6):
        for k, a in enumerate((0.06, 0.02, -0.02)):
            etfs[f"{g}{k}A000"[:6]] = dict(
                name=f"KODEX G{g}{k}", alpha=a, idx=f"합성지수{g}", nav_noise=0.0003
            )
    ters = {k: 0.1 + 0.2 * (i % 3) for i, k in enumerate(etfs)}
    ctx = make_ctx(tmp_path, etfs, ters)
    res = run.judge_all(ctx, years=(2014, 2015, 2016), n_perm=199, n_boot=200)
    assert list(res)[0] == "E3a_record"  # 결과 문서 맨 앞 = E3a 기록
    b = res["E3b_prime"]
    assert b["C"] == pytest.approx(1.0)  # 알파 순서가 두 해 내내 같다
    assert b["gates"]["G1"] is False  # 쌍 수 < 300
    assert b["grade"]["grade"] == "D"
    assert res["E3a_record"]["C_a"]["n_pairs_cond"] >= 0
    assert "records" in res and "group_size_ge3" in res["records"]
    assert res["not_implemented"]
