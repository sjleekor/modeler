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
