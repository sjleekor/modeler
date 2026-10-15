"""Synthetic tests for the KR case gallery pilot (no real data is read)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from modeler.analysis.feature_response.cases import config as C
from modeler.analysis.feature_response.cases.guards import (
    DateGuardError,
    HoldoutAccessError,
    OutcomeLeakError,
    RepoWriteError,
    check_formation_date,
    check_label_end,
    guard_not_in_repo,
    guard_path,
)
from modeler.analysis.feature_response.cases.select import (
    add_pct,
    build_grid,
    choose_dates,
    group_size_table,
    hit_flag,
    select_cases,
    with_groups,
)


def _calendar(n: int = 2700, start: dt.date = dt.date(2015, 1, 2)) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _day_frame(d: dt.date, n: int = 200) -> pl.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "trade_date": d,
                "ticker": f"{i:06d}",
                "market": "KOSPI" if i % 2 == 0 else "KOSDAQ",
                "value": float((i * 37) % n),
            }
        )
    return pl.DataFrame(rows)


def test_holdout_path_refused():
    for p in ("/x/holdout/feat.parquet", "a/Holdout_2025/b", "feat_panel_HOLDOUT.parquet"):
        with pytest.raises(HoldoutAccessError):
            guard_path(p)
    assert guard_path("/x/dev/feat_panel.parquet")


def test_no_write_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(RepoWriteError):
        guard_not_in_repo(repo / "out", repo)
    assert guard_not_in_repo(tmp_path / "elsewhere", repo)


def test_date_guards():
    assert check_formation_date(C.FORMATION_START)
    assert check_formation_date(C.FORMATION_END)
    with pytest.raises(DateGuardError):
        check_formation_date(dt.date(2015, 12, 31))
    with pytest.raises(DateGuardError):
        check_formation_date(dt.date(2025, 7, 1))
    assert check_label_end(C.DATA_END)
    with pytest.raises(DateGuardError):
        check_label_end(dt.date(2025, 8, 1))


def test_grid_non_overlapping_and_in_window():
    cal = [d for d in _calendar() if d <= C.DATA_END]
    grid = build_grid(cal, data_end=C.DATA_END)
    assert grid[0][0] >= C.FORMATION_START
    idx = [i for _, i, _ in grid]
    assert all(b - a == 20 for a, b in zip(idx, idx[1:], strict=False))
    for d, _, end in grid:
        assert C.FORMATION_START <= d <= C.FORMATION_END
        assert end <= C.DATA_END
    # label windows [i, i+20) never share a day
    spans = [(i, i + 20) for i in idx]
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_grid_refuses_window_past_data_end():
    cal = _calendar(2800)  # runs well past 2025-07-31
    with pytest.raises(DateGuardError):
        build_grid(cal, last_formation=dt.date(2026, 12, 31), data_end=C.DATA_END)


def test_choose_dates_reproducible_one_per_year():
    cal = [d for d in _calendar() if d <= C.DATA_END]
    gd = [d for d, _, _ in build_grid(cal, data_end=C.DATA_END)]
    a = choose_dates(gd, C.SEED)
    assert a == choose_dates(gd, C.SEED)
    assert a != choose_dates(gd, C.SEED + 1)
    assert len(a) == 12 and len(set(a)) == 12
    assert all(any(d.year == y for d in a) for y in C.YEARS)


def test_pct_is_per_market():
    d = dt.date(2020, 1, 2)
    df = pl.DataFrame(
        {
            "trade_date": [d] * 4,
            "ticker": list("abcd"),
            "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ"],
            "value": [1.0, 2.0, 100.0, 200.0],
        }
    )
    out = add_pct(df).sort("ticker")
    assert out["pct"].to_list() == [0.5, 1.0, 0.5, 1.0]


def test_groups_follow_direction():
    d = dt.date(2020, 1, 2)
    pct = add_pct(_day_frame(d))
    hi = with_groups(pct, +1)
    assert set(hi.filter(pl.col("pct") >= 0.9)["group"]) == {"good"}
    assert set(hi.filter(pl.col("pct") <= 0.1)["group"]) == {"opp"}
    lo = with_groups(pct, -1)
    assert set(lo.filter(pl.col("pct") <= 0.1)["group"]) == {"good"}
    none = with_groups(pct, 0)
    assert set(none["group"]) == {"top", "mid", "bottom"}


def test_selection_reproducible_and_counts():
    dates = [dt.date(2020, 1, 2), dt.date(2021, 1, 4)]
    pct = add_pct(pl.concat([_day_frame(d) for d in dates]))
    a = select_cases(pct, 1, C.SEED, "f")
    b = select_cases(pct.sort("ticker", descending=True), 1, C.SEED, "f")
    assert a.sort(["trade_date", "ticker"]).equals(b.sort(["trade_date", "ticker"]))
    per = a.group_by("trade_date", "group").len()
    got = {(r["group"]): r["len"] for r in per.filter(pl.col("trade_date") == dates[0]).to_dicts()}
    assert got == {"good": 3, "opp": 3, "all": 2}
    assert a.filter(pl.col("group") == "good")["pct"].min() >= 0.9
    assert a.filter(pl.col("group") == "opp")["pct"].max() <= 0.1
    assert a.select("trade_date", "ticker").unique().height == a.height  # no duplicates
    other = select_cases(pct, 1, C.SEED + 1, "f")
    assert not a.sort("ticker").equals(other.sort("ticker"))


def test_selection_is_outcome_blind():
    d = dt.date(2020, 1, 2)
    df = _day_frame(d).with_columns(pl.lit(0.1).alias("raw_label_20d"))
    with pytest.raises(OutcomeLeakError):
        add_pct(df)
    pct = add_pct(_day_frame(d)).with_columns(pl.lit(0.1).alias("fwd_ret_20d"))
    with pytest.raises(OutcomeLeakError):
        select_cases(pct, 1, C.SEED, "f")
    with pytest.raises(OutcomeLeakError):
        select_cases(pct.rename({"fwd_ret_20d": "outcome"}), 1, C.SEED, "f")


def test_hit_rule():
    assert hit_flag(0.01, "good") is True
    assert hit_flag(0.0, "good") is False
    assert hit_flag(-0.01, "good") is False
    assert hit_flag(0.5, "opp") is None
    assert hit_flag(0.5, "all") is None
    assert hit_flag(None, "good") is None


# ---------------------------------------------------------------------------------------------
# commentary and captions (synthetic)
# ---------------------------------------------------------------------------------------------
def _case_row(**kw) -> dict:
    base = {
        "label_missing": False,
        "group": "good",
        "hit": True,
        "fwd_ret_20d": 0.05,
        "raw_label_20d": 0.03,
        "halt_fwd": 0,
        "halt_prev20": 0,
        "limit_hits": 0,
        "jump_over_limit": 0,
        "preferred": False,
        "neg_equity": False,
        "corp_action": "",
    }
    base.update(kw)
    return base


def _series() -> dict:
    rel = list(range(-C.PRE_DAYS, 21))
    close = [100.0 * (0.88 if r == -C.PRE_DAYS else 1.0) for r in rel]
    close = [None if False else v for v in close]
    by = {r: v for r, v in zip(rel, close, strict=True)}
    by[-5] = 90.0
    return {"rel": rel, "close_norm": [by[r] for r in rel]}


def test_case_caption_numbers_and_flags():
    from modeler.analysis.feature_response.cases.render import case_caption

    spec = next(s for s in C.FEATURES.values() if s.window and s.window <= C.PRE_DAYS)
    out = case_caption(_case_row(preferred=True, halt_fwd=2), _series(), spec)
    assert f"formation 전 {C.PRE_DAYS}일 +14%" in out
    assert "이후 20일 +5%(시장 대비 +3%), 맞음" in out
    assert "우선주" in out and "이후 거래정지 2일" in out
    miss = case_caption(_case_row(hit=False, raw_label_20d=-0.02), None, spec)
    assert "틀림" in miss and "formation 전" not in miss
    gone = case_caption(_case_row(label_missing=True), None, spec)
    assert "라벨 없음" in gone


def test_commentary_loading_and_render(tmp_path):
    from modeler.analysis.feature_response.cases.render import (
        load_commentary,
        safe_html,
    )

    assert load_commentary("nope", tmp_path) == {}
    (tmp_path / "f.json").write_text(
        '{"feature":"f","dist_raw":"<b>좋습니다</b>. 둘째 문장.","path":"","x":"무시",'
        '"dates":{"2020-01-02":"하루"}}',
        encoding="utf-8",
    )
    com = load_commentary("f", tmp_path)
    assert set(com) == {"dist_raw", "dates"}
    assert com["dates"] == {"2020-01-02": "하루"}
    assert safe_html("<b>a</b><script>") == "<b>a</b>&lt;script&gt;"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert load_commentary("bad", tmp_path) == {}


# ---- plan section 11: two-group and tie-aware modes ---------------------------------------


def _two_frame(d: dt.date, n: int = 200, other_every: int = 10) -> pl.DataFrame:
    """Binary feature: every ``other_every``-th ticker is 1, the rest 0."""
    df = _day_frame(d, n).drop("value")
    return df.with_columns(
        pl.Series("value", [1.0 if i % other_every == 0 else 0.0 for i in range(n)])
    )


def test_two_group_modal_vs_other_per_market():
    d = dt.date(2020, 1, 2)
    pct = add_pct(_two_frame(d))
    g = with_groups(pct, 0, mode="two", good_group=None)
    assert set(g["group"]) == {"modal", "other"}
    assert set(g.filter(pl.col("group") == "modal")["value"]) == {0.0}
    assert set(g.filter(pl.col("group") == "other")["value"]) == {1.0}
    # good side names follow good_group
    gm = with_groups(pct, 0, mode="two", good_group="modal")
    assert set(gm.filter(pl.col("value") == 0.0)["group"]) == {"good"}
    go = with_groups(pct, 0, mode="two", good_group="other")
    assert set(go.filter(pl.col("value") == 1.0)["group"]) == {"good"}


def test_two_group_modal_is_per_date_and_market_with_small_value_tiebreak():
    d1, d2 = dt.date(2020, 1, 2), dt.date(2020, 1, 3)
    rows = []
    for d, vals in ((d1, [0, 0, 1]), (d2, [1, 1, 0]), (d1, [5, 7])):
        for i, v in enumerate(vals):
            rows.append(
                {
                    "trade_date": d,
                    "ticker": f"{d.day}{len(rows)}",
                    "market": "KOSPI",
                    "value": float(v),
                }
            )
    # d1 holds 0,0,1,5,7 -> modal 0; d2 holds 1,1,0 -> modal 1
    g = with_groups(add_pct(pl.DataFrame(rows)), 0, mode="two")
    assert set(g.filter((pl.col("trade_date") == d1) & (pl.col("group") == "modal"))["value"]) == {
        0.0
    }
    assert set(g.filter((pl.col("trade_date") == d2) & (pl.col("group") == "modal"))["value"]) == {
        1.0
    }
    # tie in count: smaller value wins
    t = pl.DataFrame(
        {
            "trade_date": [d1] * 4,
            "ticker": list("abcd"),
            "market": ["KOSPI"] * 4,
            "value": [3.0, 3.0, 2.0, 2.0],
        }
    )
    assert set(
        with_groups(add_pct(t), 0, mode="two").filter(pl.col("group") == "modal")["value"]
    ) == {2.0}


def test_two_group_selection_counts_and_empty_group_is_skipped():
    dates = [dt.date(2020, 1, 2), dt.date(2021, 1, 4)]
    first = _two_frame(dates[0])
    second = _two_frame(dates[1]).with_columns(pl.lit(0.0).alias("value"))  # no "other" at all
    pct = add_pct(pl.concat([first, second]))
    cases = select_cases(pct, 0, C.SEED, "fin_has_fs", mode="two", good_group=None)
    got0 = {
        r["group"]: r["len"]
        for r in cases.filter(pl.col("trade_date") == dates[0]).group_by("group").len().to_dicts()
    }
    assert got0 == {"modal": 3, "other": 3, "all": 2}
    got1 = {
        r["group"]: r["len"]
        for r in cases.filter(pl.col("trade_date") == dates[1]).group_by("group").len().to_dicts()
    }
    assert got1 == {"modal": 3, "all": 2}  # empty group: no cases, no error
    again = select_cases(pct, 0, C.SEED, "fin_has_fs", mode="two", good_group=None)
    assert cases.sort(["trade_date", "ticker"]).equals(again.sort(["trade_date", "ticker"]))
    # with a good side the groups are named good/opp
    gcases = select_cases(pct, 0, C.SEED, "own", mode="two", good_group="modal")
    assert set(gcases["group"]) == {"good", "opp", "all"}
    assert set(gcases.filter(pl.col("group") == "good")["value"]) == {0.0}


def test_two_group_size_table_records_empty_dates():
    dates = [dt.date(2020, 1, 2), dt.date(2021, 1, 4)]
    pct = add_pct(
        pl.concat(
            [_two_frame(dates[0]), _two_frame(dates[1]).with_columns(pl.lit(0.0).alias("value"))]
        )
    )
    g = with_groups(pct, 0, mode="two")
    rows = {
        r["group"]: r
        for r in group_size_table(g.select("trade_date", "group"), dates, ("modal", "other"))
    }
    assert rows["modal"]["rows"] == 180 + 200 and rows["other"]["rows"] == 20
    assert rows["other"]["dates_nonempty"] == 1
    assert rows["other"]["drawn_dates_empty"] == [dates[1].isoformat()]
    assert rows["modal"]["drawn_dates_empty"] == []


def test_tie_mode_keeps_ties_together_and_matches_old_pct():
    d = dt.date(2020, 1, 2)
    # 40 zeros, then 60 distinct values: zeros share an average rank percentile of 0.205
    vals = [0.0] * 40 + [float(i) for i in range(1, 61)]
    df = pl.DataFrame(
        {
            "trade_date": [d] * 100,
            "ticker": [f"{i:06d}" for i in range(100)],
            "market": ["KOSPI"] * 100,
            "value": vals,
        }
    )
    pct = add_pct(df)
    zeros = pct.filter(pl.col("value") == 0.0)["pct"].unique().to_list()
    assert zeros == [0.205]  # average rank 20.5 / count 100
    g = with_groups(pct, 1)
    assert g.filter(pl.col("group") == "opp").height == 0  # ties pushed above 0.1: group empty
    top = g.filter(pl.col("group") == "good")
    # rank 90/100 = 0.9 is inside (>=)
    assert top.height == 11 and top["pct"].min() >= 0.9
    # the same value never lands in two groups
    g2 = with_groups(pct, -1)
    assert set(g2.filter(pl.col("value") == 0.0)["group"]) == {"mid"}
    rows = group_size_table(g.select("trade_date", "group"), [d], ("good", "opp"))
    by = {r["group"]: r for r in rows}
    assert (
        by["good"]["rows"] == 11
        and by["opp"]["rows"] == 0
        and by["opp"]["drawn_dates_empty"] == [d.isoformat()]
    )
    # selection on the empty side yields no cases for it and does not fail
    cases = select_cases(pct, 1, C.SEED, "flow_short_selling_volume")
    assert "opp" not in set(cases["group"])


def test_feature_specs_cover_all_47_with_frozen_modes():
    assert len(C.FEATURES) == 47 and len(C.PILOT_FEATURES) == 9
    two = {n for n, s in C.FEATURES.items() if s.mode == "two"}
    assert two == {
        "px_is_halted",
        "fin_is_negative_equity",
        "px_halt_ratio_20d",
        "fin_has_fs",
        "own_major_filing_60d",
    }
    assert C.FEATURES["own_major_filing_60d"].good_group == "modal"
    assert C.FEATURES["px_halt_ratio_20d"].good_group == "other"
    assert all(
        C.FEATURES[n].good_group is None
        for n in two - {"own_major_filing_60d", "px_halt_ratio_20d"}
    )
    assert all(s.good_group is None for s in C.FEATURES.values() if s.mode == "tie")
    # plan section 11 direction table spot checks
    d = {n: s.direction for n, s in C.FEATURES.items()}
    assert d["px_maxret_20d"] == -1 and d["px_amihud_20d"] == 1 and d["fin_log_mcap"] == -1
    assert (
        d["px_vol_20d"] == 1 and d["px_gap_vs_ma20"] == -1 and d["flow_foreign_netbuy_sum_20d"] == 0
    )
    assert d["fin_cash_ratio"] == 0 and d["fin_asset_turnover"] == 1
    # the nine pilot specs are untouched
    assert [C.FEATURES[n].direction for n in C.PILOT_FEATURES] == [-1, -1, 1, 1, 1, 1, 1, 1, 0]


def _dist_rows(feature: str, groups: dict[str, tuple[float, float, float]]) -> list[dict]:
    out = []
    for g, (hit, mean, n) in groups.items():
        for stat, v in (("hit_rate", hit), ("mean", mean), ("n", n)):
            out.append(
                {
                    "feature": feature,
                    "group": g,
                    "metric": "raw_label_20d",
                    "stat": stat,
                    "k": None,
                    "value": v,
                }
            )
        out.append(  # other metric must be ignored
            {
                "feature": feature,
                "group": g,
                "metric": "fwd_ret_20d",
                "stat": "hit_rate",
                "k": None,
                "value": 0.99,
            }
        )
    return out


def test_rank_rows_pairs_sort_and_flags():
    from modeler.analysis.feature_response.cases.rank import pair_names, rank_rows, summary_text

    names = ["px_idio_vol_60d", "px_ret_60d", "px_halt_ratio_20d", "px_is_halted", "fin_has_fs"]
    specs = [C.FEATURES[n] for n in names]
    assert pair_names(C.FEATURES["px_idio_vol_60d"]) == ("good", "opp", "mid")
    assert pair_names(C.FEATURES["px_ret_60d"]) == ("top", "bottom", "mid")
    assert pair_names(C.FEATURES["px_halt_ratio_20d"]) == ("good", "opp", None)
    assert pair_names(C.FEATURES["fin_has_fs"]) == ("other", "modal", None)
    rows_ = (
        _dist_rows(
            "px_idio_vol_60d",
            {"good": (0.44, 0.02, 20000), "mid": (0.41, 0.0, 180000), "opp": (0.35, -0.01, 20000)},
        )
        + _dist_rows(
            "px_ret_60d",
            {"top": (0.36, -0.02, 20000), "mid": (0.41, 0.0, 180000), "bottom": (0.42, 0.0, 20000)},
        )
        + _dist_rows("px_halt_ratio_20d", {"good": (0.35, -0.03, 909), "opp": (0.41, 0.0, 237000)})
        + _dist_rows("fin_has_fs", {"other": (0.40, 0.0, 5000), "modal": (0.405, 0.0, 200000)})
    )
    dist = pl.DataFrame(rows_)
    rows = rank_rows(specs, dist, {"px_is_halted": "x"})
    assert [r["name"] for r in rows] == [
        "px_idio_vol_60d",
        "px_halt_ratio_20d",
        "px_ret_60d",
        "fin_has_fs",
        "px_is_halted",
    ]
    by = {r["name"]: r for r in rows}
    assert by["px_idio_vol_60d"]["hit_diff"] == pytest.approx(9.0)
    assert by["px_idio_vol_60d"]["sign"] == "lead"
    assert by["px_halt_ratio_20d"]["hit_diff"] == pytest.approx(-6.0)
    assert by["px_halt_ratio_20d"]["sign"] == "trail"
    assert by["px_halt_ratio_20d"]["small"] and not by["px_idio_vol_60d"]["small"]
    assert by["px_ret_60d"]["sign"] == "none"
    assert by["px_ret_60d"]["hit_diff"] == pytest.approx(-6.0)
    assert by["fin_has_fs"]["hit_diff"] == pytest.approx(-0.5)
    assert by["px_is_halted"]["hit_diff"] is None and by["px_is_halted"]["sign"] == "nogroup"
    assert "반대 쪽" in summary_text(by["px_halt_ratio_20d"])
    assert "같은 쪽" in summary_text(by["px_idio_vol_60d"])
    assert summary_text(by["fin_has_fs"]) == "양 끝 차이 1%p 안팎"
