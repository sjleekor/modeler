"""전진(forward) 평가 진입점 시험 — 합성 자료만 쓴다. 실제 stock_data는 어디서도 읽지 않는다.

* 진단 경로 불변: ``run_full`` 결과 지문을 변경 **전** 코드(병합 커밋)에서 구해 두었다.
* 전진 결정은 같은 월말의 진단 결정과 비트까지 같다(``decide(t)``는 t까지만 본다).
* 전진 로더·러너의 멈춤 조건, 출력 폴더 거부, 가드 순서.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from modeler.scores.etf_baseline import canonical as C
from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import lines as L
from modeler.scores.pension_alloc import data as D
from modeler.scores.pension_alloc import guard
from modeler.scores.pension_alloc import run as RUN
from modeler.scores.pension_alloc.jump_model import JumpModelRule
from modeler.scores.pension_alloc.rsp import SoftmaxPolicyRule
from tests.scores.etf_baseline.test_canonical import Builder, _dist, _etf, _log
from tests.scores.pension_alloc._synth import bdays
from tests.scores.pension_alloc.test_run_full_e2e import (  # noqa: F401
    T_GAMMAS,
    bundle,
)

# --------------------------------------------------------------------------- 진단 경로 지문
# 이 변경 전(병합 커밋 HEAD)의 코드로 ``test_run_full_e2e.bundle``을 돌려 구한 값이다.
# 같은 지문이 나와야 진단 경로의 결과가 바이트 단위로 안 바뀐 것이다.
GOLDEN = {
    "jm_m.npy": "73ab61f7b6cafa3719b9edab93aa8a2eb143fd659270eaf1434813d4dc75f8a5",
    "rsp_weights.npy": "c684834c154fa89f33d7021f298a89247ddfed912894900ff2aff2a6cf70ef4f",
    "rsp_seed_weights.npy": "07380d7d1429da71b15cd8d94ff01994c101ea237be7fb5be746f8e3c4b39ef3",
    "result.json(stable)": "d42404d9a6b04a6aa90281405f733d017869fa2636693ea46e03f6444f3ed63a",
}


def _fingerprint(out_dir):
    fp = {}
    for name in ("jm_m.npy", "rsp_weights.npy", "rsp_seed_weights.npy"):
        fp[name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
    d = json.loads((out_dir / "result.json").read_text())
    d["manifest"].pop("finished")
    d["manifest"].pop("run_id")
    d["manifest"].pop("input_paths")  # 임시 폴더 경로라 실행마다 다르다
    fp["result.json(stable)"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return fp


def test_diagnostic_run_full_is_byte_identical_to_pre_change(bundle):  # noqa: F811
    inp, gi, tmp = bundle
    RUN.run_full(inp, tmp / "out" / "g", gi)
    fp = _fingerprint(tmp / "out" / "g")
    assert fp == GOLDEN


def test_diagnostic_constants_untouched():
    assert D.END_DATE == date(2026, 10, 8) and D.KR_SNAPSHOT == "2026-10-18"
    assert RUN.OUTPUT_DIR == "kr/output/pension_alloc_20261019"
    assert I.ETF_FILE == "etf_daily_2010_20261008.csv.gz"
    assert RUN.COSTS == {"low": 0.0005, "high": 0.0030}
    assert RUN.FORWARD_FIRST_DECISION == date(2026, 10, 30)


# --------------------------------------------------------------------------- 러너 (합성 배열)
def _forward_from(inp, e, kr="synthetic"):
    """진단 입력 inp를 인덱스 e까지 자른 전진 입력."""
    ok = inp.kospi_available_ok[: e + 1].copy()
    ok[e] = False
    return D.ForwardInputs(
        days=inp.days[: e + 1],
        kospi_ret=inp.kospi_ret[: e + 1],
        cash_ret=inp.cash_ret[: e + 1],
        etf_ret=inp.etf_ret[: e + 1],
        end_idx=e,
        month_ends=L.month_end_indices(inp.days[: e + 1]),
        kospi_available_ok=ok,
        files=inp.files,
        snapshots=inp.snapshots,
        diagnostics=dict(inp.diagnostics),
        stops=[],
        end_date=inp.days[e],
        kr_snapshot=kr,
        obs_view="first",
        canonical={"base": "synthetic"},
    )


@pytest.fixture
def diag(bundle):  # noqa: F811
    """진단 실행 한 번 + 그 결정 목록."""
    inp, gi, tmp = bundle
    out = tmp / "out" / "diag"
    result = RUN.run_full(inp, out, gi)
    d = result["manifest"]["diagnostics"]
    me = inp.month_ends
    start = inp.days.index(d["diagnostic_start"])
    decisions = [t for t in me if t >= start and t + 1 <= inp.end_idx]
    return inp, gi, tmp, out, decisions


def test_forward_matches_diagnostic_on_shared_month_ends(diag):
    inp, gi, tmp, out, decisions = diag
    w_rsp = np.load(out / "rsp_weights.npy")
    sw = np.load(out / "rsp_seed_weights.npy")
    m = np.load(out / "jm_m.npy")
    e = decisions[-5] + 7  # 월 중간 (월말 아님)
    assert e not in inp.month_ends
    first = inp.days[decisions[-8]]
    fwd = _forward_from(inp, e)
    res = RUN.run_forward(fwd, tmp / "fwd" / "a", gi, first)
    ev_dates = res["manifest"]["evaluated_decision_dates"]
    exp = [t for t in decisions if inp.days[t] >= first and t + 1 <= e]
    assert ev_dates == [inp.days[t] for t in exp] and len(exp) >= 4
    assert res["manifest"]["pending_decision_date"] is None
    rows = [decisions.index(t) for t in exp]
    fo = tmp / "fwd" / "a"
    assert np.array_equal(np.load(fo / "rsp_weights.npy"), w_rsp[rows])
    assert np.array_equal(np.load(fo / "rsp_seed_weights.npy"), sw[rows])
    assert np.array_equal(np.load(fo / "jm_m.npy"), m[rows])
    # 평가 결과: 세 후보 × 두 비용, 기준선 B1~B5, placebo 없음
    assert set(res["candidates"]) == {"R-JM", "R-SP", "R-SP-IRP"}
    for cand, per in res["candidates"].items():
        assert set(per) == {"low", "high"}
        assert all(v["placebo"] is None for v in per.values())
    assert set(res["baselines"]["low"]) == {"B1", "B2", "B3", "B4", "B5"}
    assert sorted(p.name for p in fo.iterdir()) == [
        "forward_result.json",
        "jm_m.npy",
        "jm_weights.npy",
        "manifest.json",
        "rsp_irp_weights.npy",
        "rsp_seed_weights.npy",
        "rsp_weights.npy",
    ]
    man = json.loads((fo / "forward_result.json").read_text())["manifest"]
    for k in (
        "run_id",
        "guard",
        "code_sha",
        "kr_snapshot",
        "obs_view",
        "end_date",
        "first_forward_decision_on_or_after",
        "evaluated_decision_dates",
        "input_sha256",
        "params",
        "canonical",
    ):
        assert k in man
    assert man["obs_view"] == "first" and man["params"]["placebo"] is False
    assert man["code_sha"] == "deadbeef" and man["end_date"] == inp.days[e].isoformat()


def test_forward_pending_decision_on_month_end_day(diag):
    """끝 날짜 자체가 월말이면 그날의 결정 비중이 나오고, 진단의 같은 결정과 같다."""
    inp, gi, tmp, out, decisions = diag
    w_rsp = np.load(out / "rsp_weights.npy")
    m = np.load(out / "jm_m.npy")
    k = len(decisions) - 4
    e = decisions[k]
    first = inp.days[decisions[k - 3]]
    res = RUN.run_forward(_forward_from(inp, e), tmp / "fwd" / "p", gi, first)
    man = res["manifest"]
    assert man["pending_decision_date"] == inp.days[e]
    assert man["evaluated_decision_dates"] == [inp.days[t] for t in decisions[k - 3 : k]]
    pend = res["pending_decision"]
    assert pend["date"] == inp.days[e]
    assert np.array_equal(np.array(pend["R-SP"]), w_rsp[k])
    b4 = L.line_specs(L.US_MAIN)["B4"]
    assert np.array_equal(np.array(pend["R-JM"]), b4.weights(m[k]))
    fo = tmp / "fwd" / "p"
    assert np.load(fo / "rsp_weights.npy").shape[0] == 4  # 평가 3 + 대기 1
    assert np.array_equal(np.load(fo / "rsp_weights.npy")[-1], w_rsp[k])
    assert res["status"] == "evaluated"


def test_forward_pending_only(diag):
    """첫 전진 월말 당일: 평가할 결정은 없고 대기 결정만 있다 (10-30 당일 상황)."""
    inp, gi, tmp, out, decisions = diag
    w_rsp = np.load(out / "rsp_weights.npy")
    k = len(decisions) - 3
    e = decisions[k]
    res = RUN.run_forward(_forward_from(inp, e), tmp / "fwd" / "q", gi, inp.days[e])
    assert res["status"] == "pending_only" and res["candidates"] is None
    assert res["manifest"]["evaluated_decision_dates"] == []
    assert np.array_equal(np.array(res["pending_decision"]["R-SP"]), w_rsp[k])


def test_forward_no_eligible_month_end_exits_cleanly(diag):
    inp, gi, tmp, out, decisions = diag
    e = decisions[-4] + 5  # 월 중간
    res = RUN.run_forward(
        _forward_from(inp, e), tmp / "fwd" / "n", gi, inp.days[e] + timedelta(days=40)
    )
    assert res["status"] == "no_decision_yet"
    fo = tmp / "fwd" / "n"
    assert [p.name for p in fo.iterdir()] == ["forward_result.json"]
    man = json.loads((fo / "forward_result.json").read_text())["manifest"]
    assert man["status"] == "no_decision_yet" and man["evaluated_decision_dates"] == []
    assert man["pending_decision_date"] is None and man["obs_view"] == "first"


def test_forward_stops_before_compute(diag):
    inp, gi, tmp, out, decisions = diag
    fwd = _forward_from(inp, decisions[-4] + 5)
    fwd.stops.append("테스트: 상장 뒤 종가 없는 세션")
    with pytest.raises(I.InputError, match="STOP"):
        RUN.run_forward(fwd, tmp / "fwd" / "s1", gi, inp.days[decisions[-8]])
    assert not (tmp / "fwd" / "s1").exists()
    fwd.stops.clear()
    fwd.kospi_available_ok[decisions[-6] : decisions[-4]] = False  # 월말 가용 시각 위반
    with pytest.raises(I.InputError, match="가용하지 않은"):
        RUN.run_forward(fwd, tmp / "fwd" / "s2", gi, inp.days[decisions[-8]])


def test_forward_stops_if_first_decision_before_rule_start(diag):
    inp, gi, tmp, out, decisions = diag
    early = inp.days[decisions[0] - 40]
    with pytest.raises(I.InputError, match="규칙 첫 결정일"):
        RUN.run_forward(_forward_from(inp, decisions[-4] + 5), tmp / "fwd" / "e", gi, early)


def test_forward_refuses_nonempty_output_folder(diag):
    inp, gi, tmp, out, decisions = diag
    fwd = _forward_from(inp, decisions[-4] + 5)
    d = tmp / "fwd" / "x"
    d.mkdir(parents=True)
    (d / "old.txt").write_text("prior")
    with pytest.raises(FileExistsError):
        RUN.run_forward(fwd, d, gi, inp.days[decisions[-6]])
    assert [p.name for p in d.iterdir()] == ["old.txt"]  # 건드리지 않았다
    # 비어 있는 폴더는 괜찮다
    d2 = tmp / "fwd" / "y"
    d2.mkdir()
    RUN.run_forward(fwd, d2, gi, inp.days[decisions[-6]])
    assert (d2 / "forward_result.json").exists()
    # 같은 폴더에 다시 쓰면 거부
    with pytest.raises(FileExistsError):
        RUN.run_forward(fwd, d2, gi, inp.days[decisions[-6]])


def test_forward_month_ends_rule():
    days = bdays(60, date(2026, 9, 1))
    i = days.index(date(2026, 9, 30))
    assert i not in L.month_end_indices(days[: i + 1])
    assert i in RUN.forward_month_ends(days[: i + 1], i)  # 끝 날짜가 월 마지막 평일
    j = days.index(date(2026, 9, 29))
    assert j not in RUN.forward_month_ends(days[: j + 1], j)
    # 다음 세션이 있으면 기존 월말 목록과 같다
    k = days.index(date(2026, 10, 5))
    assert RUN.forward_month_ends(days[: k + 1], k) == L.month_end_indices(days[: k + 1])


# --------------------------------------------------------------------------- 절단 성질 (두 규칙)
def test_decide_on_truncated_data_equals_longer_series(diag):
    """decide(t)는 t까지의 자료만 쓴다: t에서 자른 자료와 긴 자료의 결과가 같다."""
    inp, gi, tmp, out, decisions = diag
    cfg = RUN.jm_config()
    slot_full = np.column_stack([inp.etf_ret, inp.cash_ret])
    first_ok = max(int(np.flatnonzero(~np.isnan(inp.etf_ret[:, k]))[0]) for k in range(5)) + 63
    me = inp.month_ends
    jm_full = JumpModelRule(inp.kospi_ret, inp.cash_ret, me, cfg)
    me_rsp_full = [d for d in me if d >= first_ok]
    rsp_full = SoftmaxPolicyRule(slot_full, me_rsp_full, T_GAMMAS, RUN.RSP_SEEDS, "main")
    for t in (decisions[-6], decisions[-3]):
        n = t + 1
        me_cut = RUN.forward_month_ends(inp.days[:n], t)
        jm_cut = JumpModelRule(inp.kospi_ret[:n], inp.cash_ret[:n], me_cut, cfg)
        a, b = jm_full.decide(t), jm_cut.decide(t)
        assert (a["m"], a["lambda"], a["is_risk"], a["val_sharpe"]) == (
            b["m"],
            b["lambda"],
            b["is_risk"],
            b["val_sharpe"],
        )
        me_rsp_cut = [d for d in me_cut if d >= first_ok]
        rsp_cut = SoftmaxPolicyRule(slot_full[:n], me_rsp_cut, T_GAMMAS, RUN.RSP_SEEDS, "main")
        c = rsp_full.decide(me_rsp_full.index(t))
        d_ = rsp_cut.decide(me_rsp_cut.index(t))
        assert np.array_equal(c["weights"], d_["weights"])
        assert np.array_equal(c["seed_weights"], d_["seed_weights"])
        assert c["gamma"] == d_["gamma"]


# ------------------------------------------------------------- 로더 (합성 정본 + 합성 KR 스냅샷)
CODES = D.ETF_CODES
N_W = 40
START = date(2020, 1, 6)


def _write_kr(root, snap, sessions, kospi_through=None, cd91_through=None):
    base = root / "kr/raw/raw_postgres" / f"snapshot_date={snap}" / "source=sj2_remote"
    tdir = base / D.OBS_TABLE
    tdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, through, v0, step in (
        (D.KOSPI_SERIES, kospi_through, 2000.0, 3.0),
        (D.CD91_SERIES, cd91_through, 3.0, 0.0001),
    ):
        for i, d in enumerate(sessions):
            if through is not None and d > through:
                continue
            nxt = sessions[i + 1] if i + 1 < len(sessions) else d + timedelta(days=1)
            rows.append(
                (
                    sid,
                    d,
                    Decimal(f"{v0 + step * i + (i % 3) * 0.5 * (sid == D.KOSPI_SERIES):.8f}"),
                    nxt,
                    datetime(2026, 9, 29, tzinfo=UTC),
                )
            )
    pl.DataFrame(
        rows,
        schema={
            "series_id": pl.String,
            "observation_date": pl.Date,
            "value_numeric": pl.Decimal(20, 8),
            "available_from_date": pl.Date,
            "fetched_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    ).write_parquet(tdir / "part-0.parquet")
    (base / "_manifests").mkdir(parents=True, exist_ok=True)
    (base / "_manifests" / "_SUCCESS.json").write_text(
        json.dumps({"tables": {D.OBS_TABLE: {"rows": len(rows)}}})
    )


def _world(
    root,
    monkeypatch,
    snap="2020-03-31",
    holiday_idx=None,
    drop_last_close=None,
    kospi_through=None,
    etf_obs_extra=(),
):
    monkeypatch.setattr(C, "HISTORY_START", START)
    sessions = bdays(N_W, START)
    _write_kr(root, snap, sessions, kospi_through=kospi_through)
    b = Builder(root / "kr/raw/krx_baseline")
    wd = list(sessions)
    hol = wd[holiday_idx] if holiday_idx is not None else None
    logs = [_log(d, "no_price" if d == hol else "trading") for d in wd]
    rows = []
    for ci, c in enumerate(CODES):
        for i, d in enumerate(wd):
            if d == hol or (drop_last_close == c and i == N_W - 1):
                continue
            rows.append(_etf(d, code=c, close=str(1000 + 7 * i + 3 * ci + (i % 4)), shares="100"))
    rows += list(etf_obs_extra)
    isin = "KR7" + CODES[0] + "007"
    dist = [_dist(isin, f"{sessions[10]:%Y%m%d}", "5")]
    wins = [
        {
            "request_key": "window=20030101_20200331",
            "fetched_at": datetime(2020, 4, 1, tzinfo=UTC),
            "result": "new_obs",
            "day_kind": "complete",
        }
    ]
    b.commit(
        obs=[("etf_daily", 2020, rows), ("etf_distribution", 2020, dist)],
        logs=[("etf_bydd_trd", logs), ("seibro_dist", wins)],
    )
    return sessions


def test_loader_builds_inputs_with_first_view(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch)
    end = s[-1]
    inp = D.load_inputs_forward(tmp_path, end, "2020-03-31")
    assert inp.stops == [] and inp.days == s and inp.end_idx == N_W - 1
    assert inp.end_date == end and inp.obs_view == "first" and inp.kr_snapshot == "2020-03-31"
    assert inp.etf_ret.shape == (N_W, 5) and np.isfinite(inp.etf_ret[1:]).all()
    assert inp.canonical["manifests"]["count"] == 1
    assert (
        inp.canonical["verified_sha256"] and inp.canonical["calendar"]["trading_days_to_end"] == N_W
    )
    assert inp.snapshots == {D.OBS_TABLE: "2020-03-31"}
    assert all("snapshot_date=2020-03-31" in str(p) for p in inp.files.values())
    assert not np.isnan(inp.cash_ret[-1])


def test_loader_uses_first_observation_not_latest(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch)
    d = s[-1]
    first = D.load_inputs_forward(tmp_path, d, "2020-03-31").etf_ret.copy()
    # 마지막 날 첫 종목 종가를 나중에 정정한다 (obs_seq=2)
    b = Builder.__new__(Builder)
    b.base, b.n = tmp_path / "kr/raw/krx_baseline", 1
    b.commit(
        obs=[("etf_daily", 2020, [_etf(d, code=CODES[0], close="9999", shares="100", obs_seq=2)])]
    )
    again = D.load_inputs_forward(tmp_path, d, "2020-03-31")
    assert np.array_equal(again.etf_ret, first, equal_nan=True)  # 최초 관측본이라 같다
    latest = C.load_etf_frames(C.CanonicalStore(b.base), CODES, "latest")[CODES[0]]
    assert latest["close"][-1] == 9999.0


def test_loader_stops_if_end_date_not_common_session(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch, holiday_idx=N_W - 3)
    # ETF 거래일이 아닌 날 (KOSPI에는 세션이 있다)
    with pytest.raises(I.InputError, match="끝 날짜"):
        D.load_inputs_forward(tmp_path, s[N_W - 3], "2020-03-31")
    # KOSPI 세션이 아닌 날(토요일)
    with pytest.raises(I.InputError, match="KOSPI 세션이 아닙니다"):
        D.load_inputs_forward(tmp_path, s[-1] + timedelta(days=1), "2020-03-31")


def test_loader_stops_if_series_ends_before_end_date(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch, drop_last_close=CODES[2])
    with pytest.raises(I.InputError, match=f"{CODES[2]}: 마지막 날"):
        D.load_inputs_forward(tmp_path, s[-1], "2020-03-31")


def test_loader_stops_if_kospi_ends_before_end_date(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch, kospi_through=bdays(N_W, START)[-3])
    with pytest.raises(I.InputError, match="KOSPI 세션이 아닙니다"):
        D.load_inputs_forward(tmp_path, s[-1], "2020-03-31")


def test_loader_collects_session_mismatch_stops(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch, holiday_idx=20)
    inp = D.load_inputs_forward(tmp_path, s[-1], "2020-03-31")
    assert any("ETF 거래일에 없는 KOSPI 세션" in x for x in inp.stops)
    with pytest.raises(I.InputError, match="STOP"):
        RUN.run_forward(inp, tmp_path / "o", {"code_sha": "x"})
    assert not (tmp_path / "o").exists()


def test_loader_then_runner_no_decision_yet(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch)
    inp = D.load_inputs_forward(tmp_path, s[-1], "2020-03-31")
    res = RUN.run_forward(inp, tmp_path / "o", {"code_sha": "abc"})  # 첫 결정 2026-10-30 > 끝 날짜
    assert res["status"] == "no_decision_yet"
    man = json.loads((tmp_path / "o" / "forward_result.json").read_text())["manifest"]
    assert man["kr_snapshot"] == "2020-03-31" and man["obs_view"] == "first"
    assert man["canonical"]["manifests"]["sha256"] and man["canonical"]["verified_sha256"]
    assert set(man["input_sha256"]) == set(inp.files) and man["end_date"] == s[-1].isoformat()
    assert man["first_forward_decision_on_or_after"] == "2026-10-30"


def test_schema_only_forward_prints_structure_not_values(tmp_path, monkeypatch):
    s = _world(tmp_path, monkeypatch)
    txt = D.schema_only_forward(tmp_path, s[-1], "2020-03-31")
    assert "end_date_is_trading_day=True" in txt and "obs_view=first" in txt
    assert f"code {CODES[0]}: rows={N_W} first_date={s[0]} last_date={s[-1]}" in txt
    assert "series market_kospi_ecos" in txt and "series rate_kr_cd91" in txt
    # 값이 새지 않는다: 합성 종가(1000대 정수)·KOSPI 수준(2000대)·분배금 500이 출력에 없다
    for token in ("1007", "1010", "2003.", "2000.0"):
        assert token not in txt
    bad = D.schema_only_forward(tmp_path, s[-1] + timedelta(days=3), "2020-03-31")
    assert "STOP" in bad and str(s[-1]) in bad


# --------------------------------------------------------------------------- 진입점 순서
def test_main_forward_blocked_by_guard_before_any_data(monkeypatch):
    import pathlib

    def boom(*a, **k):
        raise AssertionError("자료를 읽었습니다")

    monkeypatch.setattr(RUN.D, "load_inputs_forward", boom)
    monkeypatch.setattr(RUN.D, "load_inputs", boom)
    monkeypatch.setattr(RUN.D, "schema_only_forward", boom)
    monkeypatch.setattr(RUN.I, "stock_data_root", boom)
    monkeypatch.setattr(guard, "_repo_root", lambda: pathlib.Path("/nonexistent-x"))
    with pytest.raises(guard.NotFrozenError):
        RUN.main(["--forward", "--end-date", "2026-10-30", "--kr-snapshot", "2026-10-31"])


def test_main_forward_calls_guard_first_then_loader_then_runner(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(
        guard, "require_frozen_tag", lambda: order.append("guard") or {"code_sha": "x"}
    )
    monkeypatch.setattr(RUN.I, "stock_data_root", lambda: tmp_path)

    def load(root, end, snap):
        order.append(("load", end, snap))
        return "INP"

    def run(inp, out, gi):
        order.append(("run", inp, out.parent.name, gi["code_sha"]))
        return {}

    monkeypatch.setattr(RUN.D, "load_inputs_forward", load)
    monkeypatch.setattr(RUN, "run_forward", run)
    assert (
        RUN.main(
            [
                "--forward",
                "--end-date",
                "2026-10-30",
                "--kr-snapshot",
                "2026-10-31",
                "--run-id",
                "r1",
            ]
        )
        == 0
    )
    assert order == [
        "guard",
        ("load", date(2026, 10, 30), "2026-10-31"),
        ("run", "INP", "pension_alloc_forward", "x"),
    ]


def test_main_forward_schema_only_needs_no_guard(monkeypatch, capsys):
    def boom():
        raise AssertionError("가드를 불렀습니다")

    monkeypatch.setattr(guard, "require_frozen_tag", boom)
    monkeypatch.setattr(RUN.D, "schema_only_forward", lambda root, e, s: f"SCHEMA {e} {s}")
    monkeypatch.setattr(RUN.I, "stock_data_root", lambda: None)
    assert (
        RUN.main(
            [
                "--schema-only",
                "--forward",
                "--end-date",
                "2026-10-22",
                "--kr-snapshot",
                "2026-10-23",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "SCHEMA 2026-10-22 2026-10-23"


def test_main_argument_validation(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("가드·자료 접근")

    monkeypatch.setattr(guard, "require_frozen_tag", boom)
    monkeypatch.setattr(RUN.I, "stock_data_root", boom)
    for argv in (
        ["--forward"],
        ["--forward", "--end-date", "2026-10-30"],
        ["--forward", "--kr-snapshot", "2026-10-31"],
        ["--end-date", "2026-10-30"],
        ["--kr-snapshot", "2026-10-31"],
        ["--forward", "--end-date", "2026-13-40", "--kr-snapshot", "x"],
    ):
        with pytest.raises(SystemExit):
            RUN.main(argv)
