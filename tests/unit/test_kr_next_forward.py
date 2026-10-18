"""Forward evaluation job (``models/_03_kr_next/forward.py``). Synthetic data only.

The ledger comes from ``shadow.score_day`` with a fake bundle; the lake is a small DuckDB with
the four views the job reads. Every expected number is computed here from the raw price table
(not through the code under test).
"""

from __future__ import annotations

import inspect
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import evaluate as ev
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import forward as fw
from modeler.models._03_kr_next import preprocess as pp
from modeler.models._03_kr_next import shadow as sh

SEOUL = ZoneInfo("Asia/Seoul")
COLS = list(fs.NOFLOW_COLS)
TAG_TIME = datetime(2026, 10, 31, 12, 0, tzinfo=SEOUL)
GRID0 = 5  # session index of D0
N_SESSIONS = 260
COST = 0.0065


def _sessions(n: int = N_SESSIONS, start: date = date(2026, 11, 2)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


S = _sessions()


def at(d: date, hh: int = 8, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=SEOUL)


# --------------------------------------------------------------------------- ledger side


def make_panel(t: date, n: int = 60, seed: int = 0) -> pl.DataFrame:
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
    data[sh.LIQ_FLAG_COL] = [False] * n
    data[sh.LIQ_RCEPT_COL] = [None] * n
    return pl.DataFrame(data)


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


class Env:
    def __init__(self, root: Path, bundle: Path):
        self.root, self.bundle = root, bundle
        self.out = root / "shadow"
        self.ledger = self.out / "ledger.jsonl"
        self.fdir = self.out / "forward"

    def run(self, idx: int, *, now: datetime | None = None, n: int = 60) -> dict:
        d = S[idx]
        t = sh.previous_session(S, d)
        return sh.score_day(
            d,
            bundle_dir=self.bundle,
            out_root=self.out,
            ledger_path=self.ledger,
            sessions=S,
            now=now or at(d),
            panel=make_panel(t, n=n, seed=d.toordinal()),
        )

    def grid(self, idxs) -> None:
        for i in idxs:
            self.run(i)

    def day(self, idx: int) -> dict:
        d = self.out / f"D={S[idx].isoformat()}"
        return {
            "preds": pl.read_parquet(d / "predictions.parquet"),
            "basket": pl.read_parquet(d / "basket.parquet"),
        }

    def manifest(self, **over) -> dict:
        m = json.loads((self.bundle / "manifest.json").read_text())
        m["bundle_tag_time"] = TAG_TIME.isoformat()
        m.update(over)
        return m


@pytest.fixture
def env(tmp_path, bundle_dir) -> Env:
    return Env(tmp_path, bundle_dir)


# --------------------------------------------------------------------------- lake side


class Lake:
    """Synthetic lake: smooth price paths with per-window rates and a few special cases."""

    def __init__(
        self,
        n_tickers: int = 60,
        last_idx: int = N_SESSIONS - 1,
        window_rate=None,
        k200_rate: float = 0.0004,
        halted: dict[str, list[tuple[int, int]]] | None = None,
        end_idx: dict[str, int] | None = None,
        ca: set[tuple[str, int]] | None = None,
        bump: dict[tuple[str, int], float] | None = None,
    ):
        self.tickers = [f"{i:06d}" for i in range(n_tickers)]
        self.last_idx = last_idx
        self.k200_rate = k200_rate
        self.halted = halted or {}
        self.end_idx = end_idx or {}
        self.ca = ca or set()
        self.px: dict[str, list[float]] = {}
        for i, t in enumerate(self.tickers):
            path = [100.0]
            for j in range(1, N_SESSIONS):
                k = (j - 1 - GRID0) // 20 if j - 1 >= GRID0 else -1
                r = window_rate(t, k) if window_rate else 0.0005 * ((i % 9) - 4)
                path.append(path[-1] * (1.0 + r))
            self.px[t] = path
        for (t, j), f in (bump or {}).items():
            self.px[t][j] *= f
        self.k200 = [300.0 * (1.0 + k200_rate) ** j for j in range(N_SESSIONS)]

    def is_halted(self, t: str, j: int) -> bool:
        return any(a <= j <= b for a, b in self.halted.get(t, []))

    def connect(self, path: str | None = None) -> duckdb.DuckDBPyConnection:
        rows, qrows = [], []
        for t in self.tickers:
            i = int(t)
            for j in range(0, self.last_idx + 1):
                if j > self.end_idx.get(t, 10**9):
                    break
                c = self.px[t][j]
                if self.is_halted(t, j):
                    prev = self.px[t][j - 1]
                    rows.append((S[j], t, "KOSPI" if i % 3 else "KOSDAQ", 0.0, 0.0, 0.0, prev, 0))
                else:
                    rows.append(
                        (S[j], t, "KOSPI" if i % 3 else "KOSDAQ", c, c * 1.01, c * 0.99, c, 1000)
                    )
                qrows.append((S[j], t, "KOSPI" if i % 3 else "KOSDAQ", (t, j) in self.ca))
        con = duckdb.connect(path) if path else duckdb.connect()
        con.execute(
            "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
            "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT)"
        )
        con.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
        con.execute(
            "CREATE TABLE dim_price_quality_daily (trade_date DATE, ticker VARCHAR, "
            "market VARCHAR, ca_event BOOLEAN)"
        )
        con.executemany("INSERT INTO dim_price_quality_daily VALUES (?,?,?,?)", qrows)
        con.execute("CREATE TABLE stock_master (ticker VARCHAR, market VARCHAR, status VARCHAR)")
        con.executemany(
            "INSERT INTO stock_master VALUES (?,?,?)",
            [
                (
                    t,
                    "KOSPI" if int(t) % 3 else "KOSDAQ",
                    "DELISTED" if t in self.end_idx else "ACTIVE",
                )
                for t in self.tickers
            ],
        )
        con.execute(
            "CREATE TABLE krx_index_daily (bas_dd DATE, index_group VARCHAR, idx_clss VARCHAR, "
            "idx_nm VARCHAR, open_idx DOUBLE, close_idx DOUBLE)"
        )
        irows = []
        for j in range(0, self.last_idx + 1):
            irows.append((S[j], "kospi", "KOSPI", "코스피 200", self.k200[j], self.k200[j]))
            irows.append((S[j], "kospi", "KOSPI", "코스피", 1.0, 1.0))  # decoy
        con.executemany("INSERT INTO krx_index_daily VALUES (?,?,?,?,?,?)", irows)
        return con

    def ret(self, t: str, a: int, b: int) -> float:
        """Close-to-close return, frozen at the last non-halt close <= b, None-safe."""
        j = b
        while j > a and (self.is_halted(t, j) or j > self.end_idx.get(t, 10**9)):
            j -= 1
        return self.px[t][j] / self.px[t][a] - 1.0

    def bench(self, a: int, b: int) -> float:
        return self.k200[b] / self.k200[a] - 1.0


def run(env: Env, lake: Lake | duckdb.DuckDBPyConnection, **kw) -> dict:
    con = lake.connect() if isinstance(lake, Lake) else lake
    return fw.run_forward(
        con=con,
        out_root=env.out,
        ledger_path=env.ledger,
        bundle_manifest=kw.pop("manifest", env.manifest()),
        sessions=S,
        forward_dir=env.fdir,
        n_draws=kw.pop("n_draws", 200),
        **kw,
    )


def points(env: Env) -> pl.DataFrame:
    return fw.read_points(env.fdir / fw.POINTS_FILE)


def mpoints(env: Env) -> pl.DataFrame:
    """Matured rows only (the file also holds one row per missing grid date up to the lake end)."""
    return points(env).filter(pl.col("status") == "matured").sort("grid_date", "revision")


def expected_g(lake: Lake, target: dict[str, float], a: int, b: int, turnover: float) -> dict:
    ok = {t: w for t, w in target.items() if t not in lake.untradable_at(a)}
    tot = sum(ok.values())
    gross = sum(w * lake.ret(t, a, b) for t, w in ok.items()) / tot
    bench = lake.bench(a, b)
    return {
        "gross": gross,
        "bench": bench,
        "cost": COST * turnover,
        "g": gross - bench - COST * turnover,
    }


def _untradable_at(self: Lake, j: int) -> set[str]:
    return {t for t in self.tickers if self.is_halted(t, j)}


Lake.untradable_at = _untradable_at  # type: ignore[attr-defined]


def w_of(basket: pl.DataFrame) -> dict[str, float]:
    return dict(zip(basket["ticker"].to_list(), basket["target_weight"].to_list(), strict=True))


# --------------------------------------------------------------------------- pins


def test_label_params_are_the_dataset_defaults():
    sig = inspect.signature(ds.build_kr_next_panel_sql)
    for k, v in fw.LABEL_PARAMS.items():
        assert sig.parameters[k].default == v, k
    assert fw.END_HALT_HAIRCUT == 0.0
    assert (ds.TAG, ds.HORIZON) == ("c1", 20)
    assert fs.FIXED_BENCHMARK == ("k200", "코스피 200")


# --------------------------------------------------------------------------- G


def test_g_arithmetic_turnover_and_cost_match_hand_computation(env):
    env.grid([GRID0, GRID0 + 20, GRID0 + 40])
    lake = Lake()
    res = run(env, lake)
    assert res["new_rows"] >= 3
    pts = mpoints(env)
    assert pts.height == 3
    assert pts["revision"].to_list() == [1, 1, 1]

    b0, b1, b2 = (w_of(env.day(i)["basket"]) for i in (5, 25, 45))
    # point 0: first build, turnover 1
    e0 = expected_g(lake, b0, 5, 25, 1.0)
    r0 = pts.row(0, named=True)
    assert r0["first_build"] is True and r0["turnover"] == pytest.approx(1.0)
    for key in ("gross", "bench", "cost", "g"):
        assert r0[key] == pytest.approx(e0[key], abs=1e-12), key
    assert r0["cost"] == pytest.approx(0.0065)
    # point 1: drift of basket 0 over D0 -> D1 against basket 1
    drifted = {t: w * (1 + lake.ret(t, 5, 25)) for t, w in b0.items()}
    s = sum(drifted.values())
    drifted = {t: v / s for t, v in drifted.items()}
    names = set(b1) | set(drifted)
    tn1 = sum(abs(b1.get(n, 0.0) - drifted.get(n, 0.0)) for n in names) / 2
    assert 0.0 < tn1 < 1.0
    e1 = expected_g(lake, b1, 25, 45, tn1)
    r1 = pts.row(1, named=True)
    assert r1["first_build"] is False and r1["carried_span"] is False
    assert r1["prev_basket_date"] == S[5]
    assert r1["turnover"] == pytest.approx(tn1, abs=1e-12)
    for key in ("gross", "bench", "cost", "g"):
        assert r1[key] == pytest.approx(e1[key], abs=1e-12), key
    # point 2 uses D2 -> D3 = idx 65 although no record exists at idx 65
    assert pts.row(2, named=True)["next_grid_date"] == S[65]
    assert b2  # basket of the third point exists
    # matured the session after the next grid date (K200 lags one session)
    assert pts["matured_at"].to_list() == [S[26], S[46], S[66]]
    assert pts["g_ready_at"].to_list() == [S[26], S[46], S[66]]
    assert (pts["computed_data_date"] == S[N_SESSIONS - 1]).all()
    # turnover warning is the mean of the last six points (only 3 here: mean of what exists)
    assert pts["turnover_warning"].to_list() == [
        ev.turnover_warning([1.0]),
        ev.turnover_warning([1.0, tn1]),
        ev.turnover_warning([1.0, tn1, pts["turnover"][2]]),
    ]


def test_missing_grid_point_is_recorded_and_the_basket_drifts_over_the_longer_span(env):
    env.run(GRID0)
    env.run(GRID0 + 40)  # the point at idx 25 was never written
    lake = Lake()
    run(env, lake)
    pts = points(env).sort("grid_date", "revision").filter(pl.col("grid_date") <= S[45])
    assert pts["status"].to_list() == ["matured", "missing", "matured"]
    miss = pts.row(1, named=True)
    assert (
        miss["grid_date"] == S[25] and miss["reason"] == "no_record" and miss["matured_at"] is None
    )
    p0, p2 = pts.row(0, named=True), pts.row(2, named=True)
    # D_{k+1} of point 0 is idx 25 although that record is missing
    assert p0["next_grid_date"] == S[25]
    assert p2["carried_span"] is True and p2["prev_basket_date"] == S[5]
    b0, b2 = w_of(env.day(5)["basket"]), w_of(env.day(45)["basket"])
    drifted = {t: w * (1 + lake.ret(t, 5, 45)) for t, w in b0.items()}  # one span of 40 sessions
    s = sum(drifted.values())
    drifted = {t: v / s for t, v in drifted.items()}
    tn = sum(abs(b2.get(n, 0.0) - drifted.get(n, 0.0)) for n in set(b2) | set(drifted)) / 2
    assert p2["turnover"] == pytest.approx(tn, abs=1e-12)
    # the missing row is not a point: a second run adds nothing
    assert run(env, lake)["new_rows"] == 0


def test_late_attempt_is_a_missing_point_with_its_own_reason(env):
    env.run(GRID0)
    env.run(GRID0 + 20, now=at(S[25], 16, 0))  # after 15:00: valid = False
    env.run(GRID0 + 40)
    run(env, Lake(last_idx=60))
    pts = points(env).sort("grid_date")
    miss = pts.filter(pl.col("status") == "missing")
    assert miss["grid_date"].to_list() == [S[25]]
    assert miss["reason"].to_list() == ["only_invalid_attempts"]


# --------------------------------------------------------------------------- special cases


def test_delisting_untradable_ca_mask_in_g_tail_and_rank_ic(env):
    env.grid([GRID0, GRID0 + 20])
    d0 = env.day(5)
    basket = w_of(d0["basket"])
    preds = d0["preds"]
    in_basket = [t for t in preds.sort("rank")["ticker"].to_list() if t in basket]
    del_t, untr_t = in_basket[0], in_basket[1]
    rest = [t for t in preds["ticker"].to_list() if t not in basket]
    tails = set(preds.filter(pl.col("tail"))["ticker"].to_list())
    ca_t = next(t for t in rest if t in tails)  # a tail name that the CA mask drops
    lake = Lake(
        halted={untr_t: [(5, 5)]},  # halted on the entry day: untradable
        end_idx={del_t: 18},  # series ends inside the window, master says DELISTED
        ca={(ca_t, 10)},
    )
    run(env, lake)
    p0 = mpoints(env).row(0, named=True)

    # G: untradable dropped and the weights renormalized; delisted frozen at its last close
    ok = {t: w for t, w in basket.items() if t != untr_t}
    gross = sum(w * lake.ret(t, 5, 25) for t, w in ok.items()) / sum(ok.values())
    assert lake.ret(del_t, 5, 25) == pytest.approx(lake.px[del_t][18] / lake.px[del_t][5] - 1)
    assert p0["gross"] == pytest.approx(gross, abs=1e-12)
    assert p0["n_excluded"] == 1
    assert p0["excluded_weight"] == pytest.approx(basket[untr_t])
    assert p0["g"] == pytest.approx(gross - lake.bench(5, 25) - COST, abs=1e-12)

    # tail: recorded flags, per-group means, excluded rows counted per group and kind
    excluded = {untr_t: "untradable", ca_t: "ca_masked"}
    fwd = {}
    for t in preds["ticker"].to_list():
        if t in excluded:
            continue
        fwd[t] = lake.ret(t, 5, 25)
    tl = [fwd[t] for t in fwd if t in tails]
    rs = [fwd[t] for t in fwd if t not in tails]
    assert p0["tail_diff"] == pytest.approx(np.mean(tl) - np.mean(rs), abs=1e-12)
    assert (p0["n_tail"], p0["n_rest"]) == (len(tl), len(rs))
    ex_tail = {
        k: sum(1 for t, v in excluded.items() if v == k and t in tails)
        for k in set(excluded.values())
    }
    ex_rest = {
        k: sum(1 for t, v in excluded.items() if v == k and t not in tails)
        for k in set(excluded.values())
    }
    assert json.loads(p0["excluded_tail"]) == {k: v for k, v in ex_tail.items() if v}
    assert json.loads(p0["excluded_rest"]) == {k: v for k, v in ex_rest.items() if v}
    kinds = json.loads(p0["exit_kind_counts"])
    assert kinds["untradable"] == 1 and kinds["ca_masked"] == 1 and kinds["terminal"] == 1
    assert kinds["regular"] == preds.height - 3

    # rank IC over the gated names; the delisted name's raw label uses the index at its own exit
    scores, raws = [], []
    for t, sc in zip(preds["ticker"].to_list(), preds["score"].to_list(), strict=True):
        if t in excluded:
            continue
        end = 18 if t == del_t else 25
        scores.append(sc)
        raws.append(lake.ret(t, 5, end) - lake.bench(5, end))
    assert p0["rank_ic"] == pytest.approx(stats.spearmanr(scores, raws).statistic, abs=1e-9)
    assert p0["n_ic"] == len(scores)

    # random basket: same candidates, same window, the percentile of the model gross excess
    hold = ev.holding_returns(
        pl.DataFrame(
            [
                (S[j], t, lake.px[t][j], lake.is_halted(t, j))
                for t in preds["ticker"].to_list()
                for j in range(5, 26)
                if j <= lake.end_idx.get(t, 10**9)
            ],
            schema={
                "trade_date": pl.Date,
                "ticker": pl.String,
                "close": pl.Float64,
                "is_halted": pl.Boolean,
            },
            orient="row",
        ),
        S[5],
        S[25],
        preds["ticker"].to_list(),
    )
    rb = ev.random_basket(
        preds.select("ticker", pl.col("mcap_krx").alias("mcap")),
        S[5],
        hold,
        lake.bench(5, 25),
        model_excess=gross - lake.bench(5, 25),
        n_draws=200,
    )
    assert p0["random_seed"] == ev.date_seed(S[5])
    assert p0["random_n_draws"] == 200
    assert p0["random_percentile"] == pytest.approx(rb["percentile"])
    assert 0.0 <= p0["random_percentile"] <= 1.0
    # matured the session after the later of the label exit and the index date
    assert p0["matured_at"] == S[26] and p0["n_pending"] == 0


def test_tail_flags_are_the_recorded_ones_not_recomputed(env):
    env.grid([GRID0])
    d = S[GRID0]
    path = env.out / f"D={d.isoformat()}" / "predictions.parquet"
    preds = pl.read_parquet(path)
    # the job uses the stored flags: break the inputs the flags came from in a *copy* and compare
    recomputed = ev.tail_flags(preds.select("trade_date", "market", "ticker", *sh.TAIL_INPUT_COLS))
    assert recomputed["tail"].to_list() == preds["tail"].to_list()
    run(env, Lake())
    p0 = mpoints(env).row(0, named=True)
    assert p0["n_tail"] == preds.filter(pl.col("tail")).height
    assert p0["n_rest"] == preds.filter(~pl.col("tail")).height


# --------------------------------------------------------------------------- maturity


def test_maturity_function_is_prices_only():
    assert fw.maturity(S, S[25]) == S[26]  # K200 lags one session
    assert fw.maturity(S, S[25], index_lag=0) == S[25]
    assert fw.maturity(S[:26], S[25]) is None  # the calendar does not reach the session
    assert fw.maturity(S, date(2026, 11, 7)) is None  # not a session
    assert [fw.pending_kind(k) for k in (None, "censored", "late", "regular", "untradable")] == [
        "pending",
        "pending",
        "late",
        "regular",
        "untradable",
    ]


def test_point_is_not_matured_until_next_grid_close_and_index_are_in(env):
    env.grid([GRID0])
    # lake through idx 24: the next grid close (idx 25) is not there
    run(env, Lake(last_idx=24))
    assert mpoints(env).height == 0
    # through idx 25: prices are in, but the index lags one session: not matured
    run(env, Lake(last_idx=25))
    assert mpoints(env).height == 0
    # through idx 26: matured, at idx 26
    run(env, Lake(last_idx=26))
    p = mpoints(env)
    assert p.height == 1 and p["matured_at"][0] == S[26]


def _hand_cells(lake, preds, a, b, drop):
    """Tail diff and rank IC by hand over the names not in ``drop`` (regular exits)."""
    tails = set(preds.filter(pl.col("tail"))["ticker"].to_list())
    fwd = {t: lake.ret(t, a, b) for t in preds["ticker"].to_list() if t not in drop}
    tl = [v for t, v in fwd.items() if t in tails]
    rs = [v for t, v in fwd.items() if t not in tails]
    sc = dict(zip(preds["ticker"].to_list(), preds["score"].to_list(), strict=True))
    ic = stats.spearmanr(
        [sc[t] for t in fwd], [v - lake.bench(a, b) for v in fwd.values()]
    ).statistic
    return np.mean(tl) - np.mean(rs), ic, len(tl), len(rs), tails


def test_halted_candidate_beyond_the_lake_end_does_not_block_maturity(env):
    env.grid([GRID0])
    d0 = env.day(5)
    b0 = w_of(d0["basket"])
    halt_t = sorted(b0)[0]
    halted = {halt_t: [(15, 32)]}  # resumes at idx 33; label exit = idx 43 (20th non-halt session)
    lake30 = Lake(last_idx=30, halted=halted)
    res = run(env, lake30)
    p = mpoints(env)
    first_rows = p
    # matured at idx 26 although a candidate is still halted and undecided
    assert p.height == 1 and res["revisions"][0][2] == 1
    r1 = p.row(0, named=True)
    assert r1["matured_at"] == S[26] and r1["n_pending"] == 1
    assert json.loads(r1["exit_kind_counts"])["pending"] == 1
    # G values the halted name at its last non-halt close (idx 14)
    assert lake30.ret(halt_t, 5, 25) == pytest.approx(
        lake30.px[halt_t][14] / lake30.px[halt_t][5] - 1
    )
    assert r1["gross"] == pytest.approx(
        sum(w * lake30.ret(t, 5, 25) for t, w in b0.items()), abs=1e-12
    )
    # tail and rank IC from the decided rows; the pending row is counted per group and kind
    preds = d0["preds"]
    diff, ic, ntl, nrs, tails = _hand_cells(lake30, preds, 5, 25, {halt_t})
    assert r1["tail_diff"] == pytest.approx(diff, abs=1e-12)
    assert r1["rank_ic"] == pytest.approx(ic, abs=1e-9)
    assert (r1["n_tail"], r1["n_rest"], r1["n_ic"]) == (ntl, nrs, ntl + nrs)
    key = "excluded_tail" if halt_t in tails else "excluded_rest"
    assert json.loads(r1[key]) == {"pending": 1}
    # the lake grows to idx 43: the label decides ('late'), a new revision with final values
    lake43 = Lake(last_idx=43, halted=halted)
    res = run(env, lake43)
    p = mpoints(env)
    assert p.height == 2 and p["revision"].to_list() == [1, 2]
    assert p.head(1).equals(first_rows)
    r2 = p.row(1, named=True)
    assert r2["n_pending"] == 0 and r2["matured_at"] == S[26]
    assert json.loads(r2["exit_kind_counts"])["late"] == 1
    assert r2["g"] == pytest.approx(r1["g"], abs=1e-12)  # G did not change, cells did
    assert r2["n_tail"] + r2["n_rest"] == preds.height
    fwd_h = lake43.px[halt_t][43] / lake43.px[halt_t][5] - 1  # late exit at idx 43
    all_fwd = {t: lake43.ret(t, 5, 25) for t in preds["ticker"].to_list()}
    all_fwd[halt_t] = fwd_h
    tl = [v for t, v in all_fwd.items() if t in tails]
    rs = [v for t, v in all_fwd.items() if t not in tails]
    assert r2["tail_diff"] == pytest.approx(np.mean(tl) - np.mean(rs), abs=1e-12)
    # nothing changes any more: no revision 3
    assert run(env, lake43)["new_rows"] == 0


def test_check_freezes_at_check_values_and_shows_difference_from_final(env):
    env.grid([GRID0])
    d0 = env.day(5)
    b0 = w_of(d0["basket"])
    halt_t = sorted(b0)[0]
    lake = Lake(halted={halt_t: [(15, 32)]})
    cd = S[30]
    res = run(env, lake, check_dates=[cd], allowed_check_dates=[cd])
    assert res["checks_written"]
    path = env.fdir / "checks" / f"{cd.isoformat()}.json"
    rep = json.loads(path.read_text())
    # at the check the point is matured with its pending candidate; there is no provisional list
    assert "provisional" not in rep and len(rep["points"]) == 1
    pt = rep["points"][0]
    assert pt["n_pending_at_check"] == 1 and pt["pending_tickers_at_check"] == [halt_t]
    e = expected_g(lake, b0, 5, 25, 1.0)
    assert pt["g_at_check"] == pytest.approx(e["g"], abs=1e-12)
    diff, ic, *_ = _hand_cells(lake, d0["preds"], 5, 25, {halt_t})
    assert pt["tail_diff_at_check"] == pytest.approx(diff, abs=1e-12)
    assert pt["rank_ic_at_check"] == pytest.approx(ic, abs=1e-9)
    assert rep["decision"]["n_matured"] == 1
    assert rep["decision"]["mean_g"] == pytest.approx(e["g"], abs=1e-12)
    assert rep["decision"]["model"] == ev.STATUS_INSUFFICIENT  # one point is fewer than six
    # final values (full lake, label decided at idx 43) differ in the cells but not in G
    fin = mpoints(env).row(0, named=True)
    assert fin["n_pending"] == 0 and fin["matured_at"] == S[26]
    assert pt["final"]["n_pending"] == 0 and pt["final_revision"] == 1
    assert pt["differs_from_final"] == {
        "g": False,
        "tail_diff": True,
        "rank_ic": True,
        "n_pending": True,
    }
    # the check is never rewritten, even when the lake changed
    before = path.read_bytes()
    res2 = run(env, Lake(), check_dates=[cd], allowed_check_dates=[cd])
    assert res2["checks_existing"] and not res2["checks_written"]
    assert path.read_bytes() == before


def test_check_sees_the_lake_only_up_to_the_check_date(env):
    env.grid([GRID0, GRID0 + 20])
    cd = S[60]
    run(env, Lake(), check_dates=[cd], allowed_check_dates=[cd])
    rep = json.loads((env.fdir / "checks" / f"{cd.isoformat()}.json").read_text())
    # as of idx 60: point 0 matured (idx 26), point 1 (next idx 45 -> idx 46) matured too
    assert [p["grid_date"] for p in rep["points"]] == [S[5].isoformat(), S[25].isoformat()]
    assert rep["data_date"] == S[60].isoformat()
    # a check date later than the lake is refused
    with pytest.raises(fw.ForwardError, match="after the last lake date"):
        run(env, Lake(last_idx=50), check_dates=[S[70]], allowed_check_dates=[S[70]])
    with pytest.raises(fw.ForwardError, match="not a check date"):
        run(env, Lake(), check_dates=[S[61]], allowed_check_dates=[S[60]])


def test_g_at_check_differing_from_the_final_value_is_flagged():
    rows = [
        {
            "state": "matured",
            "grid_date": S[5],
            "matured_at": S[26],
            "g": 0.01,
            "tail_diff": -0.02,
            "rank_ic": 0.1,
            "n_pending": 2,
            "excluded_tail": "{}",
            "excluded_rest": "{}",
            "turnover": 1.0,
        }
    ]
    base = {c: None for c in fw.SCHEMA}
    final = pl.DataFrame(
        [
            {
                **base,
                "grid_date": S[5],
                "status": "matured",
                "revision": 2,
                "g": 0.011,
                "tail_diff": -0.02,
                "rank_ic": 0.1,
                "n_pending": 0,
            }
        ],
        schema=fw.SCHEMA,
    )
    rep = fw.build_check(
        S[30],
        rows,
        final,
        tag_time=TAG_TIME,
        prereg_tag="t",
        ledger_lines=1,
        data_date=S[30],
        code_sha="x",
    )
    pt = rep["points"][0]
    assert pt["g_at_check"] == 0.01 and pt["g_final"] == 0.011 and pt["g_final_differs"] is True
    assert pt["final_revision"] == 2
    assert pt["differs_from_final"] == {
        "g": True,
        "tail_diff": False,
        "rank_ic": False,
        "n_pending": True,
    }


# --------------------------------------------------------------------------- append-only


def test_append_only_rows_and_revisions(env):
    env.grid([GRID0, GRID0 + 20])
    run(env, Lake())
    first = points(env)
    assert mpoints(env)["revision"].to_list() == [1, 1]
    # nothing changed: nothing new
    assert run(env, Lake())["new_rows"] == 0
    assert points(env).equals(first)
    # a later final value differs (the idx 25 close is revised): new revision rows, old rows kept
    bump = {(env.day(5)["basket"]["ticker"][0], 25): 1.05}
    res = run(env, Lake(bump=bump))
    after = points(env)
    assert res["new_rows"] >= 1 and after.height == first.height + res["new_rows"]
    assert after.head(first.height).equals(first)
    new = after.tail(res["new_rows"])
    assert (new["revision"] == 2).all()
    old_g = first.filter(pl.col("grid_date") == S[5])["g"][0]
    assert new.filter(pl.col("grid_date") == S[5])["g"][0] != pytest.approx(old_g, abs=1e-9)
    # same data again: no third revision
    assert run(env, Lake(bump=bump))["new_rows"] == 0


def test_plan_appends_rules():
    base = {c: None for c in fw.SCHEMA}
    old = pl.DataFrame(
        [
            {
                **base,
                "grid_date": S[5],
                "status": "matured",
                "revision": 3,
                "g": 0.1,
                "matured_at": S[26],
            }
        ],
        schema=fw.SCHEMA,
    )
    same = {
        **base,
        "state": "matured",
        "grid_date": S[5],
        "status": "matured",
        "g": 0.1 + 1e-15,
        "matured_at": S[26],
    }
    diff = {**same, "g": 0.2}
    prov = {**same, "state": "provisional", "g": 0.5}
    new_pt = {**same, "grid_date": S[25]}
    miss = {**base, "state": "missing", "grid_date": S[45], "status": "missing"}
    assert fw.plan_appends(old, [same, prov]) == []
    out = fw.plan_appends(old, [diff, new_pt, miss, prov])
    assert [(r["grid_date"], r["revision"]) for r in out] == [(S[5], 4), (S[25], 1), (S[45], 1)]
    old2 = pl.concat([old, fw._frame([{**miss, "revision": 1}])])
    assert fw.plan_appends(old2, [miss]) == []


# --------------------------------------------------------------------------- checks


def _window_rate(env: Env, sign: float, scale: float = 0.002):
    """Per-window rate = sign * scale * z(score at D_k): IC against the label is +-1."""
    z: dict[tuple[int, str], float] = {}
    for k in range(9):
        idx = GRID0 + 20 * k
        p = env.day(idx)["preds"]
        s = p["score"].to_numpy()
        zz = (s - s.mean()) / s.std()
        for t, v in zip(p["ticker"].to_list(), zz, strict=True):
            z[(k, t)] = float(v)

    def rate(t: str, k: int) -> float:
        return sign * scale * z.get((k, t), 0.0)

    return rate


@pytest.fixture
def nine(env):
    env.grid([GRID0 + 20 * k for k in range(9)])
    return env


@pytest.mark.parametrize(
    "sign,k200,model,stop",
    [
        (-1.0, 0.0, ev.STATUS_STOP_PROPOSAL, True),
        (+1.0, 0.0, ev.STATUS_CONTINUE, False),
        (+1.0, 0.01, ev.STATUS_UNDECIDED, False),
    ],
)
def test_check_reports_insufficient_then_the_c5_rule(nine, sign, k200, model, stop):
    lake = Lake(window_rate=_window_rate(nine, sign), k200_rate=k200)
    jan, jul = ev.check_dates(sessions=S)[:2]
    res = run(nine, lake, check_dates=[jan, jul])
    assert len(res["checks_written"]) == 2
    rj = json.loads((nine.fdir / "checks" / f"{jan.isoformat()}.json").read_text())
    rl = json.loads((nine.fdir / "checks" / f"{jul.isoformat()}.json").read_text())
    # January: fewer than 6 matured points
    assert rj["decision"]["n_matured"] < 6
    assert (
        rj["decision"]["model"] == ev.STATUS_INSUFFICIENT
        and rj["decision"]["tail"] == ev.STATUS_INSUFFICIENT
    )
    assert rj["decision"]["stop_proposal"] is False and rj["requires_user_confirmation"] is False
    # July: at least 6, the literal rule
    d = rl["decision"]
    assert d["n_matured"] >= 6
    assert d["model"] == model and d["stop_proposal"] is stop
    assert rl["requires_user_confirmation"] is stop
    assert d["mean_rank_ic"] == pytest.approx(sign, abs=1e-9)
    assert (d["mean_g"] <= 0) == (model != ev.STATUS_CONTINUE)
    assert "탐색" in rl["note"]
    # the points of the report are the points.parquet values (nothing in the lake changed)
    final = points(nine).filter(pl.col("matured_at") <= jul).sort("grid_date")
    assert [p["g_at_check"] for p in rl["points"]] == pytest.approx(final["g"].to_list(), abs=1e-12)
    assert not any(p["g_final_differs"] for p in rl["points"])
    # fixed-n tests wait for 24 points
    assert d["significance_g"]["n24"]["status"] == ev.STATUS_HOLD


# --------------------------------------------------------------------------- guard


class _NoLake:
    """A connection that fails on any use: the guard must act before the lake is read."""

    def execute(self, *a, **k):
        raise AssertionError("the lake was read")

    executemany = execute


def _lines(env: Env) -> list[dict]:
    return sh.read_ledger(env.ledger)


def test_require_forward_only_accepts_forward_records(env):
    env.grid([GRID0])
    tt = fw.require_forward_only(_lines(env), env.manifest(), env.out)
    assert tt == TAG_TIME


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda m: m.pop("prereg_tag"), "prereg_tag"),
        (lambda m: m.update(prereg_tag=""), "prereg_tag"),
        (lambda m: m.pop("bundle_tag_time"), "bundle_tag_time"),
        (lambda m: m.update(bundle_tag_time="not a time"), "ISO"),
        (lambda m: m.update(bundle_tag_time="2026-10-31T12:00:00"), "time zone"),
        (lambda m: m.update(bundle_tag="other-tag"), "expected"),
        (lambda m: m.update(prereg_tag="kr-next-other"), "prereg tag"),
    ],
)
def test_guard_refuses_incomplete_or_foreign_manifest(env, mutate, match):
    env.grid([GRID0])
    m = env.manifest()
    mutate(m)
    with pytest.raises(fw.ForwardOnlyError, match=match):
        run(env, _NoLake(), manifest=m)


def test_guard_refuses_points_before_the_bundle_tag(env):
    d = S[GRID0]
    env.run(GRID0, now=at(d, 14, 0))  # valid (before 15:00), written after the 10:00 decision
    env.run(GRID0 + 20)
    # the tag came after D0's decision time (10:00) but before its write (14:00): D0 is historical
    tag = at(d, 12, 0)
    with pytest.raises(fw.ForwardOnlyError, match="earlier than the bundle tag time"):
        run(env, _NoLake(), manifest=env.manifest(bundle_tag_time=tag.isoformat()))
    # a tag before the decision is fine
    ok = at(d, 9, 0)
    assert fw.require_forward_only(_lines(env), env.manifest(bundle_tag_time=ok.isoformat()))


def test_guard_refuses_records_written_before_the_tag(env):
    d = S[GRID0]
    env.run(GRID0, now=at(d, 8, 0))
    tag = at(d, 8, 30) - timedelta(days=0)  # after the write at 08:00, before the 10:00 decision
    with pytest.raises(fw.ForwardOnlyError, match="written at"):
        fw.require_forward_only(_lines(env), env.manifest(bundle_tag_time=tag.isoformat()), env.out)


def test_guard_point_level_check_and_no_lake_access_on_refusal(env):
    env.grid([GRID0])
    with pytest.raises(fw.ForwardOnlyError):
        fw.assert_forward_point(S[GRID0], at(S[GRID0], 11, 0))
    fw.assert_forward_point(S[GRID0], at(S[GRID0], 10, 0))  # equal is forward
    # historical labels are never opened: no label SQL is built when the guard refuses
    called = []
    orig = fw.compute_labels
    fw.compute_labels = lambda *a, **k: called.append(1) or orig(*a, **k)
    try:
        with pytest.raises(fw.ForwardOnlyError):
            run(env, Lake(), manifest=env.manifest(bundle_tag_time=at(S[GRID0], 12, 0).isoformat()))
    finally:
        fw.compute_labels = orig
    assert not called and not (env.fdir / fw.POINTS_FILE).exists()


def test_broken_ledger_is_refused_before_anything_is_read(env):
    env.grid([GRID0, GRID0 + 20])
    rec = env.out / f"D={S[GRID0].isoformat()}" / "record.json"
    rec.write_text(rec.read_text() + " ")
    with pytest.raises(fw.ForwardError, match="ledger chain is broken"):
        run(env, _NoLake())
    assert not (env.fdir / fw.POINTS_FILE).exists()


def test_empty_ledger_is_a_no_op(env):
    env.out.mkdir(parents=True)
    res = run(env, _NoLake())
    assert res["n_counting"] == 0 and res["new_rows"] == 0


# --------------------------------------------------------------------------- command line


def test_cli_writes_points_and_check(env, tmp_path):
    env.grid([GRID0, GRID0 + 20])
    lake_path = tmp_path / "lake.duckdb"
    Lake().connect(str(lake_path)).close()
    sess = tmp_path / "sessions.json"
    sess.write_text(json.dumps([d.isoformat() for d in S]))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"bundle_tag": "kr-next-bundle-frozen", "bundle_tag_time": TAG_TIME.isoformat()})
    )
    cd = ev.check_dates(sessions=S)[0]
    code = fw.main(
        [
            "--out-root", str(env.out),
            "--lake", str(lake_path),
            "--bundle", str(env.bundle),
            "--freeze-info", str(freeze),
            "--sessions", str(sess),
            "--check-date", cd.isoformat(),
        ]
    )  # fmt: skip
    assert code == 0
    assert mpoints(env).height == 2
    assert (env.fdir / "checks" / f"{cd.isoformat()}.json").is_file()
    # without the freeze info the manifest has no tag time: refused
    bad = fw.main(
        [
            "--out-root",
            str(env.out),
            "--lake",
            str(lake_path),
            "--bundle",
            str(env.bundle),
            "--sessions",
            str(sess),
        ]
    )
    assert bad == 1
