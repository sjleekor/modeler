"""회사 점수 결과 변수 W4 시험 (합성 자료만, 사전등록 §5.3)."""

from __future__ import annotations

import polars as pl
import pytest

from modeler.scores.quality import company_outcomes as co
from modeler.scores.quality.company_common import JUDGMENT_ENV


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(JUDGMENT_ENV, raising=False)


def formation(rows):
    """rows: (corp, fy, te, cap, ni[, in_universe])"""
    out = []
    for r in rows:
        in_u = r[5] if len(r) > 5 else True
        out.append(
            dict(
                corp_code=r[0],
                fy=r[1],
                in_universe=in_u,
                currency="KRW",
                te=r[2],
                cap=r[3],
                ni=r[4],
            )
        )
    return pl.DataFrame(
        out,
        schema={
            "corp_code": pl.Utf8,
            "fy": pl.Int64,
            "in_universe": pl.Boolean,
            "currency": pl.Utf8,
            "te": pl.Float64,
            "cap": pl.Float64,
            "ni": pl.Float64,
        },
    )


def fs(rows):
    return pl.DataFrame(
        rows,
        schema={
            "corp_code": pl.Utf8,
            "year": pl.Int64,
            "te": pl.Float64,
            "cap": pl.Float64,
            "ni": pl.Float64,
        },
        orient="row",
    )


def ops(rows):
    return pl.DataFrame(
        rows,
        schema={
            "corp_code": pl.Utf8,
            "rcept_no": pl.Utf8,
            "report_year": pl.Int64,
            "fiscal_year": pl.Int64,
            "opinion_class": pl.Utf8,
        },
        orient="row",
    )


def dps(rows):
    return pl.DataFrame(
        rows,
        schema={
            "corp_code": pl.Utf8,
            "report_year": pl.Int64,
            "currency": pl.Utf8,
            "dps_t": pl.Float64,
            "dps_p1": pl.Float64,
        },
        orient="row",
    )


CAPEVT0 = pl.DataFrame(schema={"corp_code": pl.Utf8, "year": pl.Int64})


def st(df, corp, fy=None):
    q = df.filter(pl.col("corp_code") == corp)
    if fy is not None:
        q = q.filter(pl.col("fy") == fy)
    assert q.height == 1, q
    r = q.row(0, named=True)
    return r["status"], r["event"]


# ------------------------------------------------------------------ O1
def test_o1_basic():
    f = formation([(c, 2017, 1, 1, 1) for c in "ABCDE"])
    o = ops(
        [
            ("A", "r1", 2018, 2018, "non_clean"),  # t+1 비적정 -> 사건
            ("B", "r1", 2018, 2018, "clean"),
            ("B", "r0", 2017, 2017, "clean"),
            ("C", "r0", 2017, 2017, "non_clean"),  # t 비적정
            ("C", "r1", 2018, 2018, "clean"),
            ("D", "r0", 2017, 2017, "clean"),  # t+1 없음
            ("E", "r1", 2018, 2018, "clean"),  # t 없음 -> 제외 안 함
        ]
    )
    r = co.o1(f, o, [2017])
    assert list(r.columns) == ["corp_code", "fy", "status", "event"]
    assert st(r, "A") == ("event", 1)
    assert st(r, "B") == ("non_event", 0)
    assert st(r, "C") == ("excluded_at_t", None)
    assert st(r, "D") == ("unobserved", None)
    assert st(r, "E") == ("non_event", 0)


def test_o1_not_in_universe_dropped():
    f = formation([("A", 2017, 1, 1, 1, False), ("B", 2017, 1, 1, 1)])
    o = ops([("B", "r", 2018, 2018, "clean")])
    assert co.o1(f, o, [2017])["corp_code"].to_list() == ["B"]


def test_opinion_latest_report_wins():
    o = ops(
        [
            ("A", "20190401", 2019, 2018, "clean"),  # 전기 라벨, 더 늦은 보고서
            ("A", "20180401", 2018, 2018, "non_clean"),
            ("B", "20180401", 2018, 2018, "clean"),
            ("B", "20180401", 2018, 2018, "non_clean"),  # 같은 보고서 안: 비적정 우선
            ("C", "20190401", 2019, 2018, None),  # 분류 None은 무시 -> 앞 보고서
            ("C", "20180401", 2018, 2018, "non_clean"),
        ]
    )
    r = co.opinion_by_year(o).filter(pl.col("fiscal_year") == 2018)
    assert dict(zip(r["corp_code"], r["opinion"])) == {
        "A": "clean",
        "B": "non_clean",
        "C": "non_clean",
    }
    r2 = co.opinion_by_year(o, mode="any_report")
    assert dict(zip(r2["corp_code"], r2["opinion"]))["A"] == "non_clean"


def test_opinion_same_year_tiebreak_by_rcept_no():
    o = ops(
        [
            ("A", "20190401000001", 2019, 2018, "non_clean"),
            ("A", "20190401000002", 2019, 2018, "clean"),
        ]
    )
    assert co.opinion_by_year(o)["opinion"].to_list() == ["clean"]


def test_o1_fallback_years(monkeypatch):
    assert co.o1_years([2017, 2018], "fallback") == [2018]
    assert co.o1_years([2017, 2018], "full") == [2017, 2018]
    monkeypatch.setenv(JUDGMENT_ENV, "1")
    assert co.o1_years(range(2019, 2025), "fallback") == [2020, 2022, 2024]
    f = formation([("A", 2017, 1, 1, 1), ("A", 2018, 1, 1, 1)])
    o = ops([("A", "r", 2019, 2019, "clean")])
    r = co.o1(f, o, [2017, 2018], mode="fallback")
    assert r["fy"].to_list() == [2018]
    with pytest.raises(ValueError):
        co.o1_years([2018], "bogus")


# ------------------------------------------------------------------ O2
def test_o2():
    f = formation([(c, 2017, 100, 50, 1) for c in "ABCDEF"])
    f = f.with_columns(
        pl.when(pl.col("corp_code") == "C").then(40.0).otherwise(pl.col("te")).alias("te")
    )  # C: 형성 시 이미 잠식
    r = fs(
        [
            ("A", 2018, 30.0, 50.0, 0.0),
            ("B", 2018, 50.0, 50.0, 0.0),  # te == cap
            ("C", 2018, 10.0, 50.0, 0.0),
            ("E", 2018, -5.0, 50.0, 0.0),  # 완전 잠식
            ("F", 2018, 100.0, None, 0.0),  # cap 결측
        ]
    )  # D: t+1 행 없음
    out = co.o2(f, r, [2017])
    assert st(out, "A") == ("event", 1)
    assert st(out, "B") == ("non_event", 0)
    assert st(out, "C") == ("excluded_at_t", None)
    assert st(out, "D") == ("unobserved", None)
    assert st(out, "E") == ("event", 1)
    assert st(out, "F") == ("unobserved", None)
    full = co.o2(f, r, [2017], kind="full")
    assert st(full, "A") == ("non_event", 0)
    assert st(full, "E") == ("event", 1)


def test_o2_nonpositive_cap_missing():
    f = formation([("A", 2017, 100, 50, 1)])
    r = fs([("A", 2018, -1.0, 0.0, 0.0)])
    assert st(co.o2(f, r, [2017]), "A") == ("unobserved", None)


# ------------------------------------------------------------------ O3
def test_o3_single_report():
    f = formation([(c, 2017, 1, 1, 1) for c in "ABCDEFGHI"])
    d = dps(
        [
            ("A", 2018, "KRW", 80.0, 100.0),  # 100->80 사건
            ("B", 2018, "KRW", 81.0, 100.0),  # 아님
            ("C", 2018, "KRW", 0.0, 100.0),  # 중단 사건
            ("D", 2018, "KRW", 0.0, 40.0),  # 50원 미만 제외
            ("E", 2018, "USD", 0.0, 0.5),  # USD 0.5->0 사건
            ("F", 2018, "KRW", 50.0, 0.0),  # 무배당 제외
            ("G", 2018, "KRW", 50.0, None),  # t DPS 행 없음 -> 제외
            # H: 보고서 없음 -> unobserved
            ("I", 2018, "KRW", 10.0, 100.0),  # t+1 capevt
        ]
    )
    cap = pl.DataFrame({"corp_code": ["I"], "year": [2018]})
    out = co.o3(f, d, cap, [2017])
    assert st(out, "A") == ("event", 1)
    assert st(out, "B") == ("non_event", 0)
    assert st(out, "C") == ("event", 1)
    assert st(out, "D") == ("excluded_at_t", None)
    assert st(out, "E") == ("event", 1)
    assert st(out, "F") == ("excluded_at_t", None)
    assert st(out, "G") == ("excluded_at_t", None)
    assert st(out, "H") == ("unobserved", None)
    assert st(out, "I") == ("excluded_capevt", None)


def test_o3_boundary_and_foreign_ccy():
    f = formation([("A", 2017, 1, 1, 1), ("B", 2017, 1, 1, 1), ("C", 2017, 1, 1, 1)])
    d = dps(
        [
            ("A", 2018, "KRW", 100.0 * 0.8, 100.0),
            ("B", 2018, "USD", 0.0, 0.0),  # 0 이하 -> 제외
            ("C", 2018, "KRW", 0.0, 50.0),  # 정확히 50원은 통과
        ]
    )
    out = co.o3(f, d, CAPEVT0, [2017])
    assert st(out, "A") == ("event", 1)
    assert st(out, "B") == ("excluded_at_t", None)
    assert st(out, "C") == ("event", 1)


def test_o3_alt_source_and_capevt_year():
    f = formation([("A", 2017, 1, 1, 1), ("B", 2017, 1, 1, 1), ("C", 2017, 1, 1, 1)])
    d = dps(
        [
            ("A", 2017, "KRW", 100.0, 0.0),  # t 보고서 thstrm = 100
            ("A", 2018, "KRW", 70.0, 999.0),  # 대안에서는 frmtrm(999) 무시
            ("B", 2018, "KRW", 70.0, 100.0),  # t 보고서 없음 -> 제외
            ("C", 2017, "KRW", 100.0, 0.0),
            ("C", 2018, "KRW", 90.0, 100.0),
        ]
    )
    alt = co.o3(f, d, CAPEVT0, [2017], source="two_reports")
    assert st(alt, "A") == ("event", 1)
    assert st(alt, "B") == ("excluded_at_t", None)
    assert st(alt, "C") == ("non_event", 0)
    # 기본안: A는 frmtrm=999 -> 70 <= 799 사건, B는 100->70 사건
    base = co.o3(f, d, CAPEVT0, [2017])
    assert st(base, "B") == ("event", 1)
    # t 해 capevt: 기본안은 보지 않고 t_or_t1은 제외
    cap = pl.DataFrame({"corp_code": ["C"], "year": [2017]})
    assert st(co.o3(f, d, cap, [2017]), "C") == ("non_event", 0)
    assert st(co.o3(f, d, cap, [2017], capevt_year="t_or_t1"), "C") == (
        "excluded_capevt",
        None,
    )
    with pytest.raises(ValueError):
        co.o3(f, d, cap, [2017], source="x")


# ------------------------------------------------------------------ O4
def test_o4():
    f = formation([(c, 2018, 1, 1, 10) for c in "ABCDE"])
    f = f.with_columns(
        pl.when(pl.col("corp_code") == "E").then(-1.0).otherwise(pl.col("ni")).alias("ni")
    )  # E: 형성 ni <= 0
    rows = []
    for c, v in {
        "A": (-1, -1, 1),
        "B": (-1, 1, 1),
        "C": (-1, -1, None),  # 한 해 결측
        "E": (-1, -1, -1),
    }.items():
        for k, x in enumerate(v, start=1):
            if x is not None:
                rows.append((c, 2018 + k, 1.0, 1.0, float(x)))
    for k in (1, 2):  # D: t+3 없음, 적자 없음
        rows.append(("D", 2018 + k, 1.0, 1.0, 1.0))
    r = fs(rows)
    out = co.o4(f, r, [2018])
    assert st(out, "A") == ("event", 1)
    assert st(out, "B") == ("non_event", 0)
    assert st(out, "C") == ("unobserved", None)
    assert st(out, "D") == ("unobserved", None)
    assert st(out, "E") == ("excluded_at_t", None)
    rel = co.o4_relaxed(f, r, [2018])
    assert st(rel, "A") == ("event", 1)
    assert st(rel, "B") == ("non_event", 0)
    assert st(rel, "C") == ("event", 1)  # 관측 2해 중 적자 2
    assert st(rel, "D") == ("non_event", 0)  # 관측 2해, 적자 0
    assert st(rel, "E") == ("excluded_at_t", None)


def test_o4_relaxed_under_two_observed():
    f = formation([("A", 2018, 1, 1, 10)])
    r = fs([("A", 2019, 1.0, 1.0, -5.0)])
    assert st(co.o4_relaxed(f, r, [2018]), "A") == ("unobserved", None)


def test_o4_dev_t1():
    f = formation([(c, 2017, 1, 1, 10) for c in "ABCD"])
    r = fs(
        [
            ("A", 2018, 1.0, 1.0, -1.0),
            ("B", 2018, 1.0, 1.0, 1.0),
            ("C", 2018, 1.0, 1.0, 0.0),  # 0은 음수 아님
        ]
    )
    out = co.o4_dev_t1(f, r, [2017])
    assert st(out, "A") == ("event", 1)
    assert st(out, "B") == ("non_event", 0)
    assert st(out, "C") == ("non_event", 0)
    assert st(out, "D") == ("unobserved", None)


def test_o4_year_limit(monkeypatch):
    monkeypatch.setenv(JUDGMENT_ENV, "1")
    f = formation([("A", 2023, 1, 1, 1)])
    with pytest.raises(ValueError):
        co.o4(f, fs([]), [2023])


# ------------------------------------------------------------------ 분모에서 빠진 수
def test_unobserved_counts():
    out = pl.DataFrame(
        {
            "corp_code": ["A", "B", "C", "D"],
            "fy": [2017] * 4,
            "status": ["event", "unobserved", "unobserved", "excluded_at_t"],
            "event": [1, None, None, None],
        }
    )
    scored = pl.DataFrame({"corp_code": ["A", "B", "D"], "fy": [2017] * 3})
    r = co.unobserved_scored_counts({"O1": out}, scored)
    assert r.columns == ["outcome", "fy", "n_scored", "n_unobserved"]
    assert r.row(0) == ("O1", 2017, 3, 1)  # C는 점수 없음


# ------------------------------------------------------------------ 보호
def test_guard_blocks_judgment(monkeypatch):
    f = formation([("A", 2019, 1, 1, 1)])
    with pytest.raises(PermissionError):
        co.o1(f, ops([]), [2019])
    with pytest.raises(PermissionError):
        co.o2(f, fs([]), [2019])
    with pytest.raises(PermissionError):
        co.o3(f, dps([]), CAPEVT0, [2019])
    with pytest.raises(PermissionError):
        co.o4(f, fs([]), [2019])
    with pytest.raises(PermissionError):
        co.o4_relaxed(f, fs([]), [2019])
    with pytest.raises(PermissionError):
        co.o4_dev_t1(f, fs([]), [2019])
    with pytest.raises(PermissionError):
        co.unobserved_scored_counts(
            {
                "O1": pl.DataFrame(
                    {"corp_code": ["A"], "fy": [2019], "status": ["event"], "event": [1]}
                )
            },
            f,
        )
    monkeypatch.setenv(JUDGMENT_ENV, "1")
    assert co.o1(f, ops([]), [2019]).height == 1


# ------------------------------------------------------------------ 결측 비율 · 짝수 해 규칙
def test_missing_rate_and_even_flags():
    # 2018: 10곳 중 의견 있음 7 (결측 30%), 2019/2017: 결측 0%
    uni = pl.DataFrame(
        [(f"c{i}", y) for y in (2017, 2018, 2019) for i in range(10)],
        schema={"corp_code": pl.Utf8, "year": pl.Int64},
        orient="row",
    )
    rows = [(f"c{i}", "r", 2019, y, "clean") for y in (2017, 2019) for i in range(10)]
    rows += [(f"c{i}", "r", 2019, 2018, "clean") for i in range(7)]
    rows += [("c9", "r", 2019, 2018, None)]  # 분류 None은 없는 것
    rates = co.opinion_missing_rate(uni, ops(rows))
    d = dict(zip(rates["year"], rates["rate"]))
    assert d == {2017: 0.0, 2018: 0.3, 2019: 0.0}
    fl = co.even_year_flags(rates)
    r = fl.row(0, named=True)
    assert (r["year"], r["t"], r["demote"]) == (2018, 2017, True)
    assert r["ref_rate"] == 0.0


def test_even_flags_edges():
    def mk(d):
        ys = sorted(d)
        return pl.DataFrame({"year": ys, "rate": [d[y] for y in ys]})

    # 한쪽만 있을 때 그 해와 비교, 정확히 3%p는 내리지 않는다
    fl = co.even_year_flags(mk({2018: 0.114, 2019: 0.084}))
    assert fl["demote"].to_list() == [False]
    fl = co.even_year_flags(mk({2018: 0.114, 2019: 0.048}))
    assert fl["demote"].to_list() == [True]
    # 양쪽 평균 비교: 2020 0.084 vs (0.048 + 0.047)/2
    fl = co.even_year_flags(mk({2019: 0.048, 2020: 0.084, 2021: 0.047}))
    assert fl["demote"].to_list() == [True]
    # 홀수 이웃이 없으면 null
    fl = co.even_year_flags(mk({2018: 0.1}))
    assert fl["demote"].to_list() == [None]


def test_missing_rate_not_guarded():
    uni = pl.DataFrame({"corp_code": ["A"], "year": [2022]})
    r = co.opinion_missing_rate(uni, ops([]))
    assert r.row(0) == (2022, 1, 1, 1.0)
