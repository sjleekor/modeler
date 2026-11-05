"""Explain (why a stock ranks where it does, its price path): method and document, on tiny synthetic data."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from modeler.serving import explain as ex

FEATURES = ("f_mom", "f_vol", "f_liq", "f_flow", "f_val", "f_bin")
GROUP = {"f_mom": "price", "f_vol": "risk", "f_liq": "liq", "f_flow": "flow", "f_val": "value", "f_bin": "micro"}
DESIGN = [*(f"{f}_rank" for f in FEATURES), *(f"{f}_isna" for f in FEATURES)]
N = 60


def _world(seed: int = 3):
    """US-shaped matrix: rank columns in (0, 1), 0/1 ``_isna`` columns, one two-valued rank column."""
    rng = np.random.default_rng(seed)
    n = N
    ranks = np.column_stack([(np.argsort(np.argsort(rng.normal(size=n))) + 1) / n for _ in FEATURES])
    ranks[:, FEATURES.index("f_bin")] = np.where(rng.random(n) < 0.8, 0.25, 0.9)  # values: two
    isna = (rng.random((n, len(FEATURES))) < 0.15).astype(float)
    X = np.hstack([ranks, isna])
    y = X @ rng.normal(size=X.shape[1]) + rng.normal(scale=0.1, size=n)
    return X, y


def _independent_baseline(X, design):
    """Hand-written rule (not the module's code): mode for isna / two-valued, mean otherwise."""
    out = []
    for j, name in enumerate(design):
        col = [v for v in X[:, j] if not np.isnan(v)]
        counts = Counter(col)
        if name.endswith("_isna") or len(counts) <= 2:
            top = max(counts.values())
            out.append(min(v for v, c in counts.items() if c == top))
        else:
            out.append(sum(col) / len(col))
    return np.array(out)


def _section(est, X, *, top_n=10, scores=None, keys=None):
    n = len(X)
    scores = est.predict(X) if scores is None else scores
    keys = keys or [f"S{i:02d}" for i in range(n)]
    order = sorted(range(n), key=lambda i: (-scores[i], keys[i]))
    return ex.explain_cross_section(
        predict=est.predict, X=X, design=DESIGN, features=list(FEATURES), feature_group=GROUP, keys=keys,
        symbols=keys, comparison=np.zeros(n, dtype=int), report_keys=[keys[i] for i in order],
        report_scores=[float(scores[i]) for i in order], names={f: f"label {f}" for f in FEATURES},
        scope_for=lambda _: "scope", top_n=top_n), keys, order


# --------------------------------------------------------------------------- group map
def test_group_map_covers_every_design_column_exactly_once():
    cols = ex.group_columns(DESIGN, FEATURES, GROUP)
    flat = sorted(j for v in cols.values() for j in v)
    assert flat == list(range(len(DESIGN)))
    for j, name in enumerate(DESIGN):  # a feature and its _isna column share the group
        assert j in cols[GROUP[ex.base_feature(name, FEATURES)]]


def test_group_map_rejects_unmapped_or_foreign_features():
    with pytest.raises(ex.ExplainError):
        ex.group_columns(DESIGN, FEATURES, {k: v for k, v in GROUP.items() if k != "f_val"})
    with pytest.raises(ex.ExplainError):
        ex.group_columns([*DESIGN, "stray_isna"], FEATURES, GROUP)
    with pytest.raises(ex.ExplainError):
        ex.group_columns(DESIGN, FEATURES, {**GROUP, "f_val": "nonsense"})


def test_real_group_tables_match_the_models_contracts():
    from modeler.models._02_updown_prob import features as fx
    from modeler.serving import us_daily
    from modeler.serving.kr_model import EXCLUDED_FEATURES

    us = ex.group_columns(list(us_daily.DESIGN_COLUMNS), us_daily.FEATURES, ex.US_FEATURE_GROUP)
    assert sum(len(v) for v in us.values()) == 106 and all(len(us[g]) for g in ex.GID)
    kr_features = [c for c in fx.feature_columns("FS1h", 20) if c not in EXCLUDED_FEATURES]
    kr_design = [*kr_features, *(f"{c}_isna" for c in kr_features)]
    kr = ex.group_columns(kr_design, kr_features, ex.KR_FEATURE_GROUP)
    assert sum(len(v) for v in kr.values()) == 94 and len(kr["micro"]) == 0  # no macro_beta / hc features
    assert set(ex.KR_FLAG_FEATURES) <= set(kr_features)


# --------------------------------------------------------------------------- baseline (F2, F6)
def test_baseline_mean_for_continuous_mode_for_flags_and_two_valued_columns():
    X, _ = _world()
    got = ex.baseline(X, DESIGN)
    assert np.allclose(got, _independent_baseline(X, DESIGN), atol=1e-12, rtol=0)
    j_bin = DESIGN.index("f_bin_rank")
    assert got[j_bin] == 0.25 and got[j_bin] != np.nanmean(X[:, j_bin])  # F6: not the unreal mean
    assert got[DESIGN.index("f_mom_rank")] == pytest.approx(np.mean(X[:, 0]))
    assert set(got[len(FEATURES):]) <= {0.0, 1.0}


def test_baseline_ignores_nan_when_taking_the_mode_of_a_flag_column():
    design = ["fin_has_fs", "other"]
    X = np.array([[1.0, 0.2], [np.nan, 0.4], [1.0, 0.9], [0.0, 0.1], [np.nan, 0.5], [1.0, 0.6]])
    got = ex.baseline(X, design, mode_cols={"fin_has_fs"})
    assert got[0] == 1.0  # F2: NaN is not a state; mode of the values that exist
    assert got[1] == pytest.approx(X[:, 1].mean())


# --------------------------------------------------------------------------- occlusion
def test_ridge_occlusion_equals_the_linear_formula():
    from sklearn.linear_model import Ridge

    X, y = _world()
    est = Ridge(alpha=1.0).fit(X, y)
    section, keys, order = _section(est, X)
    base = _independent_baseline(X, DESIGN)
    cols = ex.group_columns(DESIGN, FEATURES, GROUP)
    for k, i in enumerate(section.top_index):
        for g in ex.GID:
            got = section.delta[k, ex.GID.index(g)]
            if len(cols[g]) == 0:
                assert np.isnan(got)
                continue
            expected = float(sum(est.coef_[j] * (X[i, j] - base[j]) for j in cols[g]))
            assert got == pytest.approx(expected, abs=1e-12)
    assert section.max_abs_score_diff <= ex.SCORE_ATOL
    assert [keys[i] for i in section.top_index] == [keys[i] for i in order[:10]]


def test_lightgbm_contributions_reconstruct_the_prediction_and_occlusion_runs():
    lgb = pytest.importorskip("lightgbm")
    X, y = _world()
    est = lgb.LGBMRegressor(n_estimators=25, num_leaves=7, min_child_samples=3, verbose=-1, random_state=0).fit(X, y)
    contrib = est.predict(X[:12], pred_contrib=True)
    assert np.allclose(contrib.sum(axis=1), est.predict(X[:12]), atol=1e-9)  # the P1 reconstruction check
    section, _, _ = _section(est, X)
    assert np.isfinite(section.delta[:, [ex.GID.index(g) for g in GROUP.values()]]).all()
    assert np.allclose(section.f0, est.predict(X[section.top_index]), atol=ex.SCORE_ATOL)


def test_hist_gradient_boosting_with_nan_columns_runs():
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _world()
    X = X.copy()
    X[::7, DESIGN.index("f_val_rank")] = np.nan  # KR leaves NaN in some design columns
    est = HistGradientBoostingClassifier(max_iter=15, random_state=0).fit(X, y > np.median(y))

    class _P:  # probability of the positive class, as the KR adapter scores it
        predict = staticmethod(lambda M: est.predict_proba(M)[:, 1])

    section, _, _ = _section(_P, X)
    assert np.isfinite(section.delta[:, ex.GID.index("price")]).all()


@pytest.mark.parametrize("defect, reason", [
    ("score", "score_mismatch"), ("order", "rank_order_mismatch"), ("keys", "report_keys_differ_from_input")])
def test_a_report_that_the_inputs_cannot_reproduce_fails_the_model(defect, reason):
    from sklearn.linear_model import Ridge

    X, y = _world()
    est = Ridge().fit(X, y)
    scores = est.predict(X)
    keys = [f"S{i:02d}" for i in range(N)]
    order = sorted(range(N), key=lambda i: (-scores[i], keys[i]))
    rk, rs = [keys[i] for i in order], [float(scores[i]) for i in order]
    if defect == "score":
        rs[3] += 1e-9  # beyond atol 1e-12
    elif defect == "order":
        rk[0], rk[1] = rk[1], rk[0]
        rs[0], rs[1] = rs[1], rs[0]
    else:
        rk[0] = "OTHER"
    with pytest.raises(ex.ExplainError, match=reason):
        ex.explain_cross_section(
            predict=est.predict, X=X, design=DESIGN, features=list(FEATURES), feature_group=GROUP, keys=keys,
            symbols=keys, comparison=np.zeros(N, dtype=int), report_keys=rk, report_scores=rs, names={},
            scope_for=lambda _: "s", top_n=5)


# --------------------------------------------------------------------------- top features (F5, F6)
def test_ties_over_five_percent_and_two_valued_columns_are_not_extreme_features():
    n = 40
    rng = np.random.default_rng(0)
    base = (np.argsort(np.argsort(rng.normal(size=n))) + 1) / n
    X = np.column_stack([base, np.where(np.arange(n) < 12, 0.01, base), np.where(np.arange(n) < 2, 0.01, base),
                         np.where(rng.random(n) < 0.5, 0.2, 0.8), np.zeros(n)])
    features = ["a", "tied", "rare", "bin", "isna_pad"]
    ties = ex.tie_shares(X, [0, 1, 2, 3], np.zeros(n, dtype=int))
    assert ties[0, 1] == 12 / n and ties[0, 2] == 2 / n
    row = X[0].copy()
    row[0] = 0.01  # 'a' is genuinely extreme
    got = ex.top_features(row, dict(zip(features[:4], range(4))), {}, {"a": "A"}, "scope", {"bin"}, ties[0], k=4)
    names = [t["feature"] for t in got]
    assert "a" in names and "tied" not in names and "bin" not in names  # F5 + binary exclusion
    assert "rare" in names  # 2/40 = 5% is not over the limit
    first = got[0]
    assert first["side"] == "low" and first["pct"] == pytest.approx(0.01) and first["scope"] == "scope"


def test_top_features_skip_missing_values_and_report_the_tail_share():
    X = np.array([[0.97, 0.5, 0.5], [0.2, 0.5, 0.5], [0.4, 0.5, 0.5]])
    ties = ex.tie_shares(X, [0, 1], np.zeros(3, dtype=int))
    got = ex.top_features(X[0], {"a": 0, "b": 1}, {"a": 2}, {}, "s", set(), np.zeros(X.shape[1]), k=3)
    assert [t["feature"] for t in got][0] == "a" and got[0]["side"] == "high"
    assert got[0]["pct"] == pytest.approx(0.03) and got[0]["label"] == "a"
    X[0, 2] = 1.0  # a's value was missing (isna flag)
    got = ex.top_features(X[0], {"a": 0, "b": 1}, {"a": 2}, {}, "s", set(), np.zeros(X.shape[1]), k=3)
    assert "a" not in [t["feature"] for t in got]
    assert ties.shape == X.shape


@pytest.mark.parametrize("bad", sorted(ex.CONTAMINATED_DISPLAY_FEATURES))
def test_top_features_never_show_contaminated_financial_features(bad):
    # the contaminated feature is the most extreme of all; the next ones fill the list
    X = np.array([[0.001, 0.02, 0.9, 0.5]])
    rank_cols = {bad: 0, "next1": 1, "next2": 2, "mid": 3}
    got = ex.top_features(X[0], rank_cols, {}, {}, "s", set(), np.zeros(4), k=3)
    names = [t["feature"] for t in got]
    assert bad not in names
    assert names == ["next1", "next2", "mid"]


def test_contaminated_display_set_is_exactly_the_four_ttm_ratio_features():
    assert ex.CONTAMINATED_DISPLAY_FEATURES == {
        "fin_operating_margin", "fin_asset_turnover", "fin_operating_margin_v2", "fin_asset_turnover_v2"}


# --------------------------------------------------------------------------- prices
def _axis(n=61, end=date(2026, 10, 23)):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _flat_price(n=61, start=100.0):
    return np.linspace(start, start * 1.3, n)


def test_start_halted_uses_the_carried_close_and_end_halted_is_blank():
    C = _flat_price()
    carry = np.full(61, np.nan)
    halt = np.zeros(61, dtype=bool)
    C[40], carry[40], halt[40] = np.nan, 110.0, True  # 20 sessions before the as-of: halted
    susp = np.zeros(61, dtype=bool)
    ret, why, carried = ex.returns_for(C, carry, susp)
    assert ret[20] == pytest.approx(C[-1] / 110.0 - 1.0) and carried[20] and why[20] is None
    assert not carried[5] and not carried[60]
    # without a carried close the start is blank
    ret2, why2, _ = ex.returns_for(C, np.full(61, np.nan), susp)
    assert ret2[20] is None and why2[20] == "missing_start_or_short_history"
    # the as-of session itself halted: nothing is claimed
    C[-1] = np.nan
    ret3, why3, _ = ex.returns_for(C, carry, susp)
    assert set(ret3.values()) == {None} and set(why3.values()) == {"missing_asof"}


def test_price_row_gaps_are_null_not_zero_and_notes_are_written():
    dates = [d.isoformat() for d in _axis()]
    C = _flat_price()
    carry = np.full(61, np.nan)
    halt = np.zeros(61, dtype=bool)
    C[10:13] = np.nan
    halt[10:12] = True  # two halt days, one plain missing day
    C[40], carry[40], halt[40] = np.nan, 110.0, True
    bench = np.linspace(100, 105, 61)
    susp = np.zeros(61, dtype=bool)
    susp[50] = True  # a flagged jump
    row = ex.price_row(C, carry, halt, susp, bench, dates)
    assert row["path"][0] == 100.0 and row["path"][10] is None and row["path"][12] is None
    assert row["path"][50] is None and 0.0 not in row["path"]
    kinds = {g["date"]: g["kind"] for g in row["gaps"]}
    assert kinds[dates[10]] == "halt" and kinds[dates[12]] == "missing" and kinds[dates[50]] == "ca_suspect"
    assert row["ret"]["20"] is None and "의심일" in row["ret_note"]["20"]  # the flagged day is inside the span
    assert row["ret"]["5"] is not None and row["price_status"] == "partial"
    assert row["bench_path"][0] == 100.0 and len(row["bench_path"]) == 61 and row["n_obs"] == 61 - 4


def test_price_row_started_during_a_halt_uses_the_carried_close_as_start():
    dates = [d.isoformat() for d in _axis()]
    C = _flat_price(start=50.0)
    carry = np.full(61, np.nan)
    C[0], carry[0] = np.nan, 40.0
    halt = np.zeros(61, dtype=bool)
    halt[0] = True
    row = ex.price_row(C, carry, halt, np.zeros(61, dtype=bool), np.full(61, 100.0), dates)
    assert row["path"][0] is None and row["path"][1] == pytest.approx(100.0 * C[1] / 40.0)
    assert row["ret"]["60"] == pytest.approx(C[-1] / 40.0 - 1.0) and row["ret_note"]["60"] == "정지 중 시작"


def test_benchmark_leaves_out_missing_and_jump_days_and_never_zero_fills_a_stock():
    C = np.array([[100.0, 101.0, 102.0, 103.0], [200.0, np.nan, 204.0, 206.0], [10.0, 10.0, 30.0, 30.0]])
    line, count = ex.benchmark_series(C, np.arange(3), 0.5, 2.0)
    # day1: A +1%, B missing, C 0% -> mean 0.5%; day2: A +0.99%, B missing(prev missing), C jump out
    assert count.tolist() == [2, 1, 3]
    assert line[1] == pytest.approx(100 * (1 + (0.01 + 0.0) / 2))
    assert line[2] == pytest.approx(line[1] * (1 + 102 / 101 - 1))
    assert line[3] == pytest.approx(line[2] * (1 + np.mean([103 / 102 - 1, 206 / 204 - 1, 0.0])))


def _kr_px(days, tickers=("A", "B"), market="KOSPI"):
    rows = []
    for t, d in enumerate(days):
        for i, tk in enumerate(tickers):
            price = 100 + t + 10 * i
            rows.append((d, tk, market, price, price, price, price, 1000 + t))
    return pl.DataFrame(rows, schema=["trade_date", "ticker", "market", "open", "high", "low", "close", "volume"],
                        orient="row")


def test_kr_price_matrix_cuts_at_the_asof_and_never_reads_later_sessions():
    days = _axis(70, end=date(2026, 10, 30))  # 5 sessions after the as-of 10-23
    asof = date(2026, 10, 23)
    px = _kr_px(days)
    later = px["trade_date"] > asof
    px = px.with_columns(pl.when(later).then(pl.lit(999999)).otherwise(pl.col("close")).alias("close"))
    pm = ex.kr_price_matrix(px, [("A", "KOSPI"), ("B", "KOSPI")], asof, source="s", sha256="h")
    assert pm.dates[-1] == asof and len(pm.dates) == 61 and max(pm.dates) <= asof
    assert np.nanmax(pm.C) < 1000
    with pytest.raises(ex.ExplainError, match="price_axis_not_ending_at_asof"):
        ex.kr_price_matrix(px.filter(pl.col("trade_date") != asof), [("A", "KOSPI")], asof, source="s", sha256="h")


def test_kr_halt_rules_zero_ohlc_or_zero_volume_are_gaps_with_a_carried_close():
    days = _axis(61)
    px = _kr_px(days, tickers=("A",))
    px = px.with_columns(
        pl.when(pl.col("trade_date") == days[20]).then(0).otherwise(pl.col("open")).alias("open"),
        pl.when(pl.col("trade_date") == days[20]).then(0).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("trade_date") == days[20]).then(0).otherwise(pl.col("low")).alias("low"),
        pl.when(pl.col("trade_date") == days[30]).then(0).otherwise(pl.col("volume")).alias("volume"))
    pm = ex.kr_price_matrix(px, [("A", "KOSPI")], days[-1], source="s", sha256="h")
    for t in (20, 30):  # F3: both kinds of halt cut the line; the carried close is kept apart
        assert np.isnan(pm.C[0, t]) and pm.halt[0, t] and pm.carry[0, t] > 0


def test_us_price_matrix_drops_thin_days_and_cuts_at_the_asof():
    days = _axis(70, end=date(2026, 10, 30))
    asof = date(2026, 10, 23)
    symbols = [f"S{i}" for i in range(10)]
    rows = [(d, s, 10.0 + i, 10.0 + i, 1000.0) for d in days for i, s in enumerate(symbols)]
    adj = pl.DataFrame(rows, schema=["date", "symbol", "close", "adj_close", "adj_volume"], orient="row")
    adj = adj.filter(~((pl.col("date") == days[-8]) & (pl.col("symbol") != "S0")))  # one thin day inside the window
    adj = adj.with_columns(pl.when(pl.col("date") > asof).then(1e9).otherwise(pl.col("adj_close")).alias("adj_close"))
    pm = ex.us_price_matrix(adj, symbols, asof, source="s", sha256="h")
    assert pm.dates[-1] == asof and days[-8] not in pm.dates and np.nanmax(pm.C) < 100
    zero_vol = adj.with_columns(pl.when(pl.col("date") == asof).then(0.0).otherwise(pl.col("adj_volume")).alias("adj_volume"))
    pm2 = ex.us_price_matrix(zero_vol, symbols, asof, source="s", sha256="h")
    assert np.isnan(pm2.C[:, -1]).all() and pm2.halt[:, -1].all() and (pm2.carry[:, -1] > 0).all()


def test_suspect_matrix_flags_the_day_of_the_jump_only():
    C = np.array([[100.0, 101.0, 55.0, 56.0], [100.0, np.nan, 105.0, 106.0]])
    flags = ex.suspect_matrix(C, 0.67, 1.33)
    assert flags.tolist() == [[False, False, True, False], [False, False, False, False]]


# --------------------------------------------------------------------------- document shape
ROW_KEYS = {"rank", "symbol", "score", "explain_status", "explain_reason", "delta", "top_features",
            "price_status", "price_reason", "path", "bench_path", "ret", "ret_note", "gaps", "n_obs"}
MODEL_KEYS = {"market", "status", "reason", "asof", "inputs", "groups", "dates", "top100_mean_delta", "rows"}
DOC_KEYS = {"schema", "method_version", "group_map_version", "unit", "invocation_id", "report_sha256",
            "release", "status", "reason", "models"}


def _model_block(*, explain_ok=True, price_ok=True):
    from sklearn.linear_model import Ridge

    X, y = _world()
    section, keys, order = _section(Ridge().fit(X, y), X, top_n=5)
    days = _axis()
    C = np.tile(_flat_price(), (N, 1))
    pm = ex.PriceMatrix(days, C, np.full_like(C, np.nan), np.zeros(C.shape, dtype=bool), "src", "sha")
    top = [{"rank": r + 1, "symbol": keys[i], "score": 0.5} for r, i in enumerate(section.top_index)]
    prices = ex.price_block(pm, top_index=section.top_index, members_by_row=[np.arange(N)] * N, lo=0.5, hi=2.0)
    return ex._assemble(
        model_id="m", market="US", asof="2026-10-23", inputs={"prepared_sha256": "a", "bundle_sha256": "b",
                                                              "price_source": "s", "price_sha256": "c", "benchmark": "d"},
        group_ids=[g for g in ex.GID if g in set(GROUP.values())], top=top,
        section=section if explain_ok else None, explain_error=None if explain_ok else "score_mismatch:1e-3",
        prices=prices if price_ok else None, price_error=None if price_ok else "price_files_missing")


def test_model_block_has_the_fixed_v1_fields_and_status_rules():
    block = _model_block()
    assert set(block) == MODEL_KEYS and block["status"] == "ok" and len(block["dates"]) == 61
    assert all(set(r) == ROW_KEYS for r in block["rows"])
    assert set(block["rows"][0]["delta"]) == {"price", "risk", "liq", "flow", "value", "micro"}  # groups with columns
    assert set(block["top100_mean_delta"]) == set(block["rows"][0]["delta"])
    assert set(block["rows"][0]["ret"]) == {"5", "20", "60"}
    # price fails alone -> the explain part stands; explain fails alone -> prices stand, model failed
    only_explain = _model_block(price_ok=False)
    assert only_explain["status"] == "partial" and only_explain["rows"][0]["price_status"] == "failed"
    assert only_explain["rows"][0]["explain_status"] == "ok" and "price:price_files_missing" in only_explain["reason"]
    only_price = _model_block(explain_ok=False)
    assert only_price["status"] == "failed" and only_price["rows"][0]["price_status"] == "ok"
    assert only_price["rows"][0]["delta"] == {} and only_price["reason"].startswith("explain:score_mismatch")
    assert set(only_price["rows"][0]) == ROW_KEYS


def test_document_rollup_serialises_without_nan_and_failure_document_is_stdlib_shaped():
    ok, bad = _model_block(), ex.failed_model("KR", "model_not_in_report")
    doc = ex.build_document(unit="2026-10-26", invocation_id="inv", report_sha256="r" * 64, release="rel",
                            models={"m1": ok, "m2": bad})
    assert set(doc) == DOC_KEYS and doc["schema"] == "briefing-explain.v1" and doc["status"] == "partial"
    assert doc["method_version"] == "occlusion-mean-mode-v1" and doc["group_map_version"] == "groups-2026-10-26"
    assert ex.rollup_status({"a": bad}) == "failed" and ex.rollup_status({"a": ok}) == "ok"
    json.dumps(ex._clean(doc), allow_nan=False)
    failure = ex.failure_document(unit="2026-10-26", invocation_id="inv", report_sha256="r" * 64, release="rel",
                                  reason="timeout")
    assert failure["status"] == "failed" and set(failure["models"]) == set(ex.MODEL_MARKET)
    assert all(m["status"] == "failed" and m["reason"] == "timeout" and m["rows"] == [] for m in failure["models"].values())
    json.dumps(failure, allow_nan=False)


def test_atomic_write_json_leaves_no_temp_file_and_replaces_whole(tmp_path):
    target = tmp_path / "explain-2026-10-26.json"
    sha = ex.atomic_write_json(target, {"a": [1, None]})
    assert json.loads(target.read_text()) == {"a": [1, None]} and sha
    ex.atomic_write_json(target, {"b": 2})
    assert json.loads(target.read_text()) == {"b": 2} and [p.name for p in tmp_path.iterdir()] == [target.name]
    with pytest.raises(ValueError):
        ex.atomic_write_json(target, {"x": float("nan")})
    assert json.loads(target.read_text()) == {"b": 2}  # a refused write leaves the old file whole


def test_child_refuses_a_report_or_jobs_file_that_changed(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text("{}")
    jobs = tmp_path / "jobs.json"
    jobs.write_text('{"jobs": []}')
    args = ["run", "--report-date", "2026-10-26", "--report", str(report), "--report-sha256", "0" * 64,
            "--invocation-id", "i", "--release", "r", "--jobs-config", str(jobs), "--jobs-config-sha256", "0" * 64,
            "--prepared-root", str(tmp_path), "--output", str(tmp_path / "out.json")]
    assert ex.main(args) == 2 and "report_changed" in capsys.readouterr().out
    args[args.index("--report-sha256") + 1] = ex.sha256_file(report)
    assert ex.main(args) == 2 and "jobs_config_changed" in capsys.readouterr().out
    assert not (tmp_path / "out.json").exists()


def test_a_model_without_a_pin_or_report_entry_fails_alone_and_the_document_is_whole(tmp_path):
    report = json.dumps({"markets": [], "failures": []}).encode()
    doc = ex.explain_report(unit="2026-10-26", invocation_id="inv", report_bytes=report, release="rel", pins={},
                            kr_root=None, us_root=None)
    assert doc["status"] == "failed" and set(doc["models"]) == set(ex.MODEL_MARKET)
    assert {m["reason"] for m in doc["models"].values()} == {"model_not_pinned"}
