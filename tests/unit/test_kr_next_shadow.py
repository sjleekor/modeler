"""Shadow serving of the next KR model (``models/_03_kr_next/shadow.py``). Synthetic data only."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import evaluate as ev
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import preprocess as pp
from modeler.models._03_kr_next import shadow as sh

SEOUL = ZoneInfo("Asia/Seoul")
COLS = list(fs.NOFLOW_COLS)


def _sessions(n: int = 80, start: date = date(2026, 11, 2)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


SESSIONS = _sessions()


def make_panel(
    t: date, n: int = 60, seed: int = 0, flagged: tuple[str, ...] = (), with_liq: bool = True
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"{i:06d}" for i in range(n)]
    data: dict[str, object] = {
        "trade_date": [t] * n,
        "ticker": tickers,
        "market": ["KOSPI" if i % 3 else "KOSDAQ" for i in range(n)],
    }
    for c in COLS:
        data[c] = rng.normal(size=n)
    data["fin_capital_impaired_v2"] = (rng.random(n) < 0.1).tolist()
    data["mcap_krx_log"] = rng.normal(26.0, 1.0, size=n)
    data["px_turnover_ma20_v2"] = np.exp(rng.normal(20.0, 1.0, size=n))
    data.update(
        in_universe=[True] * n,
        in_universe_v2=[True] * n,
        raw_price_missing=[False] * n,
        keep_one_per_company=[True] * n,
        corp_code=[f"c{i}" for i in range(n)],
    )
    if with_liq:
        data[LIQ] = [x in flagged for x in tickers]
        data[RCEPT] = [("20261029000123" if x in flagged else None) for x in tickers]
    return pl.DataFrame(data)


LIQ, RCEPT = sh.LIQ_FLAG_COL, sh.LIQ_RCEPT_COL


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("bundle")
    frames = [
        make_panel(date(2026, 10, 1) + timedelta(days=i), n=80, seed=100 + i) for i in range(6)
    ]
    big = pl.concat(frames)
    ranked = pp.rank_transform(big.with_columns([pl.col(c).cast(pl.Float64) for c in COLS]), COLS)
    rng = np.random.default_rng(5)
    X = bundle_mod.model_matrix(ranked, COLS)
    y = rng.normal(size=ranked.height)
    model = HistGradientBoostingRegressor(max_iter=10, random_state=0).fit(X, y)
    ridge = Ridge(alpha=100.0).fit(bundle_mod.ridge_matrix(ranked, COLS), y)
    return bundle_mod.write_bundle(
        base / "bundle",
        model=model,
        ridge=ridge,
        ridge_alpha=100.0,
        feature_cols=COLS,
        golden_frame=ranked,
        params={"learning_rate": 0.1},
        max_iter=10,
        score_matrix={"scores": [[0.0]]},
        determinism={"identical": True},
        snapshot_id="snap-test",
        prereg_tag="kr-next-frozen",
        prereg_commit="abc",
        modeler_commit="def",
        prereg_doc_sha256="0" * 64,
        preprocessing={"rank": "per-date"},
    )


def at(d: date, hh: int = 8, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=SEOUL)


class Env:
    def __init__(self, root: Path, bundle: Path):
        self.root, self.bundle = root, bundle
        self.out = root / "shadow"
        self.ledger = self.out / "ledger.jsonl"

    def run(self, d: date, *, n=60, seed=None, flagged=(), now=None, panel=None, **kw):
        t = sh.previous_session(SESSIONS, d)
        if panel is None:
            panel = make_panel(
                t, n=n, seed=seed if seed is not None else d.toordinal(), flagged=flagged
            )
        return sh.score_day(
            d,
            bundle_dir=self.bundle,
            out_root=self.out,
            ledger_path=self.ledger,
            sessions=SESSIONS,
            now=now or at(d),
            panel=panel,
            **kw,
        )


@pytest.fixture
def env(tmp_path, bundle_dir) -> Env:
    return Env(tmp_path, bundle_dir)


# ----------------------------------------------------------------- candidates, ranks, tails


def test_liq_flagged_names_are_not_candidates(env):
    res = env.run(SESSIONS[5], flagged=("000003", "000010"))
    pred = pl.read_parquet(res["dir"] / "predictions.parquet")
    assert pred.height == 58
    assert not {"000003", "000010"} & set(pred["ticker"].to_list())
    assert not pred[LIQ].any()
    panel = pl.read_parquet(res["dir"] / "panel.parquet")
    assert panel.height == 60 and panel[LIQ].sum() == 2  # the input is kept as used
    c = res["record"]["counts"]
    assert (c["panel_rows"], c["liq_flagged_excluded"], c["candidates"]) == (60, 2, 58)


def test_contract_columns_required(env):
    t = sh.previous_session(SESSIONS, SESSIONS[5])
    with pytest.raises(sh.ShadowContractError, match="liq_notice"):
        env.run(SESSIONS[5], panel=make_panel(t, with_liq=False))
    assert not env.ledger.exists()
    with pytest.raises(sh.ShadowContractError, match="px_vol_20d_v2"):
        env.run(SESSIONS[5], panel=make_panel(t).drop("px_vol_20d_v2"))


def test_ranks_are_unique_and_follow_score_then_tv20_then_ticker(env):
    res = env.run(SESSIONS[5])
    pred = pl.read_parquet(res["dir"] / "predictions.parquet")
    assert pred["rank"].to_list() == list(range(1, pred.height + 1))
    assert pred["score"].to_list() == sorted(pred["score"].to_list(), reverse=True)
    for col in ("score", "ridge_score"):
        assert pred[col].null_count() == 0


def _stub_bundle(levels: int = 3):
    def predict(frame: pl.DataFrame) -> np.ndarray:
        return np.floor(frame[COLS[0]].fill_null(0.0).to_numpy() * levels)

    return SimpleNamespace(feature_cols=COLS, predict=predict, ridge=None)


def test_rank_ties_break_by_tv20_then_ticker():
    t = date(2026, 11, 5)
    panel = make_panel(t, n=40, seed=3)
    # equal tv20 inside the tie groups for some names, so the ticker decides there
    tv = np.exp(np.round(np.log(panel["px_turnover_ma20_v2"].to_numpy()), 0))
    panel = panel.with_columns(pl.Series("px_turnover_ma20_v2", tv))
    pred, _ = sh._predictions(panel, _stub_bundle(), ds.lagged_columns("noflow"), date(2026, 11, 6))
    assert pred["score"].n_unique() <= 4  # real ties
    rows = pred.select("score", "tv20", "ticker", "rank").rows()
    assert rows == sorted(rows, key=lambda r: (-r[0], -r[1], r[2]))
    assert [r[3] for r in rows] == list(range(1, 41))
    assert pred["ridge_score"].null_count() == pred.height  # stub has no Ridge


def test_tail_flags_and_inputs_recorded(env):
    res = env.run(SESSIONS[5])
    pred = pl.read_parquet(res["dir"] / "predictions.parquet")
    for c in (
        "tail1",
        "tail2",
        "tail",
        *sh.TAIL_INPUT_COLS,
        "px_vol_20d_v2_pct",
        "tv20",
        "mcap_krx",
        LIQ,
        RCEPT,
        "lagnull_mcap_krx_log",
    ):
        assert c in pred.columns, c
    for c in ("tail1", "tail2", "tail"):
        assert pred[c].dtype == pl.Boolean and pred[c].null_count() == 0
    # recomputing on the same inputs gives the same flags; the record is the source later
    again = ev.tail_flags(pred.select("trade_date", "market", "ticker", *sh.TAIL_INPUT_COLS))
    assert again["tail"].to_list() == pred["tail"].to_list()
    top = pred.sort("px_vol_20d_v2", descending=True).row(0, named=True)
    assert top["tail1"] is True
    assert (pred["tail"] == (pred["tail1"] | pred["tail2"])).all()
    expect_mcap = np.exp(
        pl.read_parquet(res["dir"] / "panel.parquet").sort("ticker")["mcap_krx_log"].to_numpy()
    )
    got = pred.sort("ticker")["mcap_krx"].to_numpy()
    assert np.allclose(got, expect_mcap)


def test_rank_transform_is_as_training(env):
    t = sh.previous_session(SESSIONS, SESSIONS[5])
    panel = make_panel(t, n=60, seed=1, flagged=("000001",))
    res = env.run(SESSIONS[5], panel=panel)
    bundle = bundle_mod.load_bundle(env.bundle)
    cands = panel.filter(~pl.col(LIQ))
    ranked = pp.rank_transform(
        cands.select(*sh.KEYS, *COLS).with_columns([pl.col(c).cast(pl.Float64) for c in COLS]),
        COLS,
    )
    want = dict(zip(ranked["ticker"].to_list(), bundle.predict(ranked), strict=True))
    pred = pl.read_parquet(res["dir"] / "predictions.parquet")
    for tk, s in zip(pred["ticker"].to_list(), pred["score"].to_list(), strict=True):
        assert s == pytest.approx(want[tk])


# --------------------------------------------------------------------------- grid and basket


def test_grid_and_non_grid_days(env):
    d0, d1, d20 = SESSIONS[5], SESSIONS[6], SESSIONS[25]
    r0 = env.run(d0, n=120)
    assert r0["record"]["is_grid_date"] and r0["record"]["d0"] == d0.isoformat()
    assert (r0["dir"] / "basket.parquet").is_file()
    r1 = env.run(d1, n=120)
    assert not r1["record"]["is_grid_date"] and r1["record"]["basket"] is None
    assert not (r1["dir"] / "basket.parquet").exists()
    r20 = env.run(d20, n=120)
    assert r20["record"]["is_grid_date"] and r20["record"]["grid_k"] == 1
    off = env.run(SESSIONS[24], n=120)
    assert not off["record"]["is_grid_date"]
    lines = sh.read_ledger(env.ledger)
    assert [(x["D"], x["has_basket"]) for x in lines] == [
        (d0.isoformat(), True),
        (d1.isoformat(), False),
        (d20.isoformat(), True),
        (SESSIONS[24].isoformat(), False),
    ]


def test_basket_is_30_names_capped_and_continues_from_previous_target(env):
    d0, d20 = SESSIONS[5], SESSIONS[25]
    r0 = env.run(d0, n=120)
    b0 = pl.read_parquet(r0["dir"] / "basket.parquet")
    assert b0.height == 30 and b0["target_weight"].sum() == pytest.approx(1.0)
    assert b0["target_weight"].max() <= 0.10 + 1e-12
    assert r0["record"]["basket"]["first_build"] is True
    assert not b0["was_held"].any()
    assert set(b0["rank"].to_list()) == set(range(1, 31))

    r20 = env.run(d20, n=120)
    b20 = pl.read_parquet(r20["dir"] / "basket.parquet")
    pred = pl.read_parquet(r20["dir"] / "predictions.parquet")
    held = set(b0["ticker"].to_list())
    keep = pred.filter(pl.col("ticker").is_in(list(held)) & (pl.col("rank") <= ev.BASKET_K_OUT))
    assert set(keep["ticker"].to_list()) <= set(b20["ticker"].to_list())
    assert b20.height == 30 and b20["was_held"].sum() == keep.height
    # same as the evaluation module run by hand with the previous target as holdings
    want = ev.build_basket(
        dict(zip(b0["ticker"].to_list(), b0["target_weight"].to_list(), strict=True)),
        pred.select("ticker", "score", "tv20", pl.col("mcap_krx").alias("mcap")),
    )
    got = dict(zip(b20["ticker"].to_list(), b20["target_weight"].to_list(), strict=True))
    assert got.keys() == want.keys()
    assert all(got[k] == pytest.approx(want[k]) for k in got)
    prev = r20["record"]["basket"]["previous"]
    assert (
        prev["D"] == d0.isoformat() and prev["record_sha256"] == r0["ledger_line"]["record_sha256"]
    )
    # market caps are the lagged KRX values, exp(mcap_krx_log)
    assert (b20["mcap_krx"] > 0).all()


def test_basket_holds_all_when_fewer_than_30_candidates(env):
    res = env.run(SESSIONS[5], n=20)
    b = pl.read_parquet(res["dir"] / "basket.parquet")
    assert b.height == 20 and res["record"]["basket"]["fewer_than_k_candidates"]


def test_missing_grid_point_is_carried_over(env):
    d0, d40 = SESSIONS[5], SESSIONS[45]
    r0 = env.run(d0, n=120)
    # the point at D0+20 was never written; the next grid date still uses the D0 basket
    r40 = env.run(d40, n=120)
    assert r40["record"]["grid_k"] == 2
    assert r40["record"]["basket"]["previous"]["D"] == d0.isoformat()
    assert r0["record"]["is_grid_date"]


# --------------------------------------------------------------------------- validity and attempts


def test_valid_flag_follows_the_1500_deadline(env):
    d = SESSIONS[5]
    ok = env.run(d, now=at(d, 14, 59))
    assert ok["record"]["valid"] is True
    d2 = SESSIONS[6]
    late = env.run(d2, now=at(d2, 15, 1))
    assert late["record"]["valid"] is False
    assert late["record"]["deadline_at"].startswith(f"{d2.isoformat()}T15:00:00")
    assert late["record"]["decision_at"].startswith(f"{d2.isoformat()}T10:00:00")
    d3 = SESSIONS[7]
    exact = env.run(d3, now=at(d3, 15, 0))
    assert exact["record"]["valid"] is True
    nxt = env.run(SESSIONS[8], now=at(SESSIONS[8]) + timedelta(days=1))  # next day
    assert nxt["record"]["valid"] is False


def test_invalid_first_record_is_not_d0(env):
    d = SESSIONS[5]
    late = env.run(d, now=at(d, 16))
    assert late["record"]["d0"] is None and late["record"]["is_grid_date"] is False
    d2 = SESSIONS[6]
    first = env.run(d2)
    assert first["record"]["d0"] == d2.isoformat() and first["record"]["is_grid_date"]
    assert sh.counting_records(sh.read_ledger(env.ledger)).keys() == {d2.isoformat()}


def test_rerun_never_overwrites_and_goes_to_attempt_dir(env):
    d = SESSIONS[5]
    r1 = env.run(d, n=120)
    first_hash = sh.sha256_file(r1["dir"] / "record.json")
    r2 = env.run(d, n=120, flagged=("000001",))
    r3 = env.run(d, n=120)
    assert r1["dir"].name == f"D={d.isoformat()}"
    assert r2["dir"] == r1["dir"] / "attempt-2" and r3["dir"] == r1["dir"] / "attempt-3"
    assert sh.sha256_file(r1["dir"] / "record.json") == first_hash
    lines = sh.read_ledger(env.ledger)
    assert [(x["D"], x["attempt"], x["dir"]) for x in lines] == [
        (d.isoformat(), 1, f"D={d.isoformat()}"),
        (d.isoformat(), 2, f"D={d.isoformat()}/attempt-2"),
        (d.isoformat(), 3, f"D={d.isoformat()}/attempt-3"),
    ]
    # only the first valid attempt counts, and it does not become a new D0
    assert sh.counting_records(lines)[d.isoformat()]["attempt"] == 1
    assert r2["record"]["d0"] == d.isoformat()
    assert sh.verify_ledger(env.ledger)["ok"]


def test_attempt_two_does_not_change_holdings_of_the_next_grid_point(env):
    d0, d20 = SESSIONS[5], SESSIONS[25]
    r0 = env.run(d0, n=120)
    env.run(d0, n=120, flagged=tuple(f"{i:06d}" for i in range(0, 40)))  # a different attempt-2
    r20 = env.run(d20, n=120)
    assert (
        r20["record"]["basket"]["previous"]["record_sha256"] == r0["ledger_line"]["record_sha256"]
    )


def test_orphan_directory_is_reported_and_next_run_uses_attempt_dir(env):
    d = SESSIONS[5]
    (env.out / f"D={d.isoformat()}").mkdir(parents=True)  # crash left a directory, no ledger line
    assert sh.verify_ledger(env.ledger, env.out)["orphans"] == [f"D={d.isoformat()}"]
    res = env.run(d)
    assert res["dir"].name == "attempt-2"


# --------------------------------------------------------------------------- ledger chain


def _three(env):
    for i in (5, 6, 7):
        env.run(SESSIONS[i], n=60)


def test_ledger_chain_verifies_and_chains_the_previous_line(env):
    _three(env)
    rep = sh.verify_ledger(env.ledger)
    assert rep == {"ok": True, "n_lines": 3, "breaks": [], "orphans": []}
    texts = env.ledger.read_text().splitlines()
    objs = [json.loads(x) for x in texts]
    assert objs[0]["prev_ledger_sha256"] == "0" * 64
    for i in (1, 2):
        assert objs[i]["prev_ledger_sha256"] == sh.sha256_bytes(texts[i - 1].encode())
    assert set(objs[0]) >= {"D", "record_sha256", "prev_ledger_sha256", "valid"}


def test_tampering_is_detected(env, tmp_path):
    _three(env)
    ledger_text = env.ledger.read_text()

    # 1. edit a record.json
    rec = env.out / f"D={SESSIONS[6].isoformat()}" / "record.json"
    original = rec.read_text()
    rec.write_text(original.replace('"valid": true', '"valid": false'))
    rep = sh.verify_ledger(env.ledger)
    assert not rep["ok"] and any("line 2" in b and "record.json" in b for b in rep["breaks"])
    rec.write_text(original)
    assert sh.verify_ledger(env.ledger)["ok"]

    # 2. edit a prediction file (record.json still lists the old hash)
    pf = env.out / f"D={SESSIONS[7].isoformat()}" / "predictions.parquet"
    saved = pf.read_bytes()
    pl.read_parquet(pf).with_columns(pl.col("score") + 1).write_parquet(pf)
    rep = sh.verify_ledger(env.ledger)
    assert any("line 3" in b and "predictions.parquet" in b for b in rep["breaks"])
    pf.write_bytes(saved)

    # 3. edit a ledger line
    lines = ledger_text.splitlines()
    env.ledger.write_text(
        "\n".join([lines[0], lines[1].replace('"valid":true', '"valid":false'), lines[2]]) + "\n"
    )
    assert any(
        "line 3" in b and "prev_ledger_sha256" in b for b in sh.verify_ledger(env.ledger)["breaks"]
    )

    # 4. drop the middle line
    env.ledger.write_text("\n".join([lines[0], lines[2]]) + "\n")
    rep = sh.verify_ledger(env.ledger)
    assert not rep["ok"] and rep["breaks"] and rep["orphans"]

    # 5. drop the last line (the chain alone cannot see it; the directory shows up as an orphan)
    env.ledger.write_text("\n".join(lines[:2]) + "\n")
    rep = sh.verify_ledger(env.ledger)
    assert rep["orphans"] == [f"D={SESSIONS[7].isoformat()}"]

    # 6. reorder
    env.ledger.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    assert not sh.verify_ledger(env.ledger)["ok"]


def test_broken_ledger_blocks_new_records(env):
    _three(env)
    lines = env.ledger.read_text().splitlines()
    env.ledger.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(sh.ShadowError, match="ledger chain is broken"):
        env.run(SESSIONS[8])
    assert not (env.out / f"D={SESSIONS[8].isoformat()}").exists()


def test_failed_run_leaves_no_trace(env):
    t = sh.previous_session(SESSIONS, SESSIONS[5])
    bad = make_panel(t, n=60).with_columns(pl.col(LIQ).alias(LIQ))
    all_flagged = bad.with_columns(pl.lit(True).alias(LIQ))
    with pytest.raises(sh.ShadowError, match="no candidates"):
        env.run(SESSIONS[5], panel=all_flagged)
    assert not env.ledger.exists()
    leftovers = [p.name for p in env.out.iterdir()] if env.out.exists() else []
    assert not [n for n in leftovers if n.startswith(".tmp") or n.startswith("D=")]


def test_lock_is_time_limited(env):
    env.out.mkdir(parents=True)
    with sh._lock(env.ledger.with_name("ledger.jsonl.lock")):
        with pytest.raises(sh.ShadowError, match="lock busy"):
            env.run(SESSIONS[5], lock_timeout=0.4)


def test_unknown_session_and_missing_signal_rows(env):
    with pytest.raises(sh.ShadowError, match="not a session"):
        env.run(date(2026, 11, 7), panel=make_panel(date(2026, 11, 6)))  # Saturday
    t = sh.previous_session(SESSIONS, SESSIONS[5])
    with pytest.raises(sh.ShadowError, match="no rows on the signal date"):
        env.run(SESSIONS[5], panel=make_panel(t - timedelta(days=9)))


def test_bundle_feature_list_must_match_the_feature_set(env):
    with pytest.raises(sh.ShadowError, match="feature list"):
        env.run(SESSIONS[5], feature_set="main")


def test_cli_score_and_verify(env, tmp_path, capsys):
    d = SESSIONS[5]
    t = sh.previous_session(SESSIONS, d)
    panel_path = tmp_path / "in.parquet"
    make_panel(t, n=60, seed=9).write_parquet(panel_path)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"market": "KR", "sessions": [s.isoformat() for s in SESSIONS]}))
    args = [
        "score",
        "--date",
        d.isoformat(),
        "--panel",
        str(panel_path),
        "--bundle-dir",
        str(env.bundle),
        "--out-root",
        str(env.out),
        "--ledger",
        str(env.ledger),
        "--sessions",
        str(cal),
    ]
    rc = sh.main(args)
    # the real clock is later than the 15:00 deadline of a 2026-11 date or earlier; either is fine
    assert rc in (0, 3)
    assert (env.out / f"D={d.isoformat()}" / "record.json").is_file()
    assert sh.main(["verify", "--ledger", str(env.ledger)]) == 0
    capsys.readouterr()
