"""회사 점수 묶기·실행 W6 시험 (합성 자료만, 사전등록 §5·§7.2·§12.2)."""

# ruff: noqa: E501
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import company_judge as cj
from modeler.scores.quality import company_outcomes as co
from modeler.scores.quality import company_run as cr
from modeler.scores.quality import company_score as cs
from modeler.scores.quality.company_common import JUDGMENT_ENV, Lake


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(JUDGMENT_ENV, raising=False)
    monkeypatch.delenv(cr.INTERP_TABLE_ENV, raising=False)


def _approve(tmp_path, monkeypatch, text="승인본 v1\n"):
    """해석 표 승인본 사본을 만들고 환경변수로 가리킨다."""
    p = tmp_path / "interp_table_v1_approved.md"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setenv(cr.INTERP_TABLE_ENV, str(p))
    return p


# ---------------------------------------------------------------- 합성 자료
CUR = ["ta", "tl", "te", "ca", "cl", "re", "oi", "ni", "ocf", "rev", "gp", "ltb", "ip", "cap"]
PRIOR = ["ta_p1", "ta_p2", "te_p1", "te_p2", "ni_p1", "ni_p2", "ocf_p1", "ca_p1", "cl_p1",
         "rev_p1", "gp_p1", "ltb_p1"]  # fmt: skip
ALL_YEARS = list(range(2015, 2026))


def make_fs(n: int, years, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        corp = f"{i:08d}"
        for fy in years:
            ta = float(rng.lognormal(24, 1))
            r = dict(
                corp_code=corp, fy=fy, stock_code=f"{i:06d}", market="KOSPI",
                in_universe=bool(rng.random() < 0.93), is_spac=False,
                is_financial=bool(rng.random() < 0.05), acc_mt=12 if rng.random() < 0.9 else 3,
                fs_div="CFS", rcept_no=f"{fy + 1}0331{i:06d}",
                layer="vintage" if rng.random() < 0.8 else "raw",
                avail_date=date(fy + 1, 4, 1), currency="KRW",
            )  # fmt: skip
            r.update(ta=ta, tl=ta * rng.uniform(0.2, 0.9), ca=ta * rng.uniform(0.2, 0.6))
            r["te"] = ta - r["tl"] if rng.random() > 0.03 else -ta * 0.1
            r.update(
                cl=r["ca"] * rng.uniform(0.3, 1.5), re=ta * rng.uniform(-0.1, 0.4),
                oi=ta * rng.normal(0.03, 0.05), ni=ta * rng.normal(0.02, 0.05),
                ocf=ta * rng.normal(0.04, 0.05), rev=ta * rng.uniform(0.3, 1.5),
                gp=ta * rng.uniform(0.05, 0.4),
                ltb=ta * rng.uniform(0, 0.3) if rng.random() < 0.6 else None,
                ip=ta * rng.uniform(0, 0.02), cap=ta * 0.05,
            )  # fmt: skip
            for c in PRIOR:
                base = c.split("_")[0]
                r[c] = r[base] * rng.uniform(0.7, 1.2) if r.get(base) is not None else None
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("fy").cast(pl.Int32))


def make_misc(fs: pl.DataFrame, years, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for corp, fy in (
        fs.filter(pl.col("fy").is_in(list(years))).select("corp_code", "fy").iter_rows()
    ):
        d = float(rng.choice([0, 50, 100, 200, 400]))
        ev = bool(rng.random() < 0.05)
        evp = bool(rng.random() < 0.05)
        sh = float(rng.integers(1_000_000, 50_000_000))
        rows.append(
            dict(
                corp_code=corp, fy=fy, dps=d, dps_p1=d * rng.uniform(0.5, 1.5),
                dps_p2=d * rng.uniform(0.5, 1.5), dps_src="common", dps_rcept_no=f"{fy}X{corp}",
                shares=sh, shares_p1=sh * rng.uniform(0.95, 1.05), shares_src="보통주",
                retire=float(rng.integers(0, 2)), retire_p1=float(rng.integers(0, 2)),
                capevt=ev, capevt_p1=evp, capevt_p2=False,
                capevt_rec=ev or bool(rng.random() < 0.1), capevt_rec_p1=evp, capevt_rec_p2=False,
            )
        )  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("fy").cast(pl.Int32))


def make_outcome_sources(n: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    fsl, dps, ops = [], [], []
    for i in range(n):
        corp = f"{i:08d}"
        for y in ALL_YEARS:
            ta = float(rng.lognormal(24, 1))
            fsl.append(
                dict(corp_code=corp, year=y, fs_div="CFS", rcept_no=f"{y + 1}0401{i:06d}",
                     te=ta * rng.uniform(-0.05, 0.6), cap=ta * 0.2, ni=ta * rng.normal(0.0, 0.05))
            )  # fmt: skip
        for ry in range(2016, 2027):
            d = float(rng.choice([0, 30, 50, 100, 200]))
            dps.append(
                dict(corp_code=corp, report_year=ry, rcept_no=f"{ry + 1}0331{i:06d}",
                     dps_t=d, dps_p1=float(rng.choice([0, 50, 100, 200])), knd_source="common")
            )  # fmt: skip
        for ry in (2017, 2019, 2021, 2023, 2025):
            for k, lab in enumerate(("당기", "전기", "전전기")):
                cls = "non_clean" if rng.random() < 0.15 else "clean"
                ops.append(
                    dict(corp_code=corp, rcept_no=f"{ry + 1}0330{i:06d}", report_year=ry,
                         row_ordinal=k, label_raw=lab, fiscal_year=ry - k,
                         opinion_raw="한정" if cls == "non_clean" else "적정",
                         opinion_norm="x", opinion_class=cls)
                )  # fmt: skip
    schema = {"report_year": pl.Int32, "fiscal_year": pl.Int32, "row_ordinal": pl.Int32}
    return (
        pl.DataFrame(fsl),
        pl.DataFrame(dps).with_columns(pl.col("report_year").cast(pl.Int32)),
        pl.DataFrame(ops).with_columns(*[pl.col(k).cast(v) for k, v in schema.items()]),
    )


def make_sources(tmp_path, period: str, n: int = 150, with_outcomes: bool = True) -> cr.Sources:
    years = list(cr.PERIOD_YEARS[period])
    fs_all = make_fs(n, ALL_YEARS)
    misc = make_misc(fs_all, years)
    fsl, dps, ops = make_outcome_sources(n)
    ev_j = misc.filter(pl.col("capevt")).select("corp_code", pl.col("fy").alias("year")).unique()
    ev_r = (
        misc.filter(pl.col("capevt_rec")).select("corp_code", pl.col("fy").alias("year")).unique()
    )
    src = cr.Sources(
        lake=Lake(root=tmp_path / "lake", raw_snapshot="2026-09-30", derived_snapshot="2026-09-29"),
        years=years, fs_all=fs_all, misc=misc, opinions=ops,
    )  # fmt: skip
    if with_outcomes:
        fs_main = fs_all.filter(pl.col("fy").is_in(years)).with_columns(
            (pl.col("ni_p1") * 1.1).alias("ni_p1")
        )
        src.fs_prior_a = fs_main
        src.fs_latest, src.dps_latest = fsl, dps
        src.capevt_judgment, src.capevt_record = ev_j, ev_r
        # 해석 표 대안 기록용 입력: 앞 8개 회사의 해당 연도 보고서가 multi, 전전기 행은 앵커 없이 결측
        src.dps_multi = misc.filter(pl.col("corp_code") < "00000008").select(
            "corp_code", pl.col("fy").alias("report_year"), pl.col("dps_rcept_no").alias("rcept_no"),
            pl.lit(2, pl.Int64).alias("n_common_rows"),
        )  # fmt: skip
        src.opinions_anchor_off = ops.with_columns(
            pl.when(pl.col("label_raw") == "전전기").then(None).otherwise(pl.col("fiscal_year"))
            .alias("fiscal_year")
        )  # fmt: skip
        src.misc_latest = misc.with_columns((pl.col("shares_p1") * 1.1).alias("shares_p1"))
        # 기록용 판 입력: raw 한 층 단독(앞 40개 회사는 행이 없음)·문면판(자본금을 비움)
        src.fs_raw_only = fs_main.filter(pl.col("corp_code") >= "00000040").with_columns(
            pl.lit("raw").alias("layer")
        )
        src.fs_literal = fs_main.with_columns(
            pl.when(pl.col("corp_code") < "00000010")
            .then(None)
            .otherwise(pl.col("cap"))
            .alias("cap")
        )
    src.availability_mismatch = {"vintage_rcept": 100, "vintage_mismatch": 3}
    return src


# ---------------------------------------------------------------- assemble_panel
def test_assemble_panel_contract():
    fs = make_fs(20, [2017, 2018])
    misc = make_misc(fs, [2017, 2018])
    p = cr.assemble_panel(fs, misc)
    assert p.height == fs.height
    assert all(c in p.columns for c in cs.REQUIRED_COLS)
    assert p["fy"].dtype == pl.Int64
    assert p.select("corp_code", "fy").is_duplicated().sum() == 0
    cs.compute_scores(p)  # W3 열 계약을 실제로 통과한다


def test_assemble_panel_capevt_judgment_vs_record():
    fs = make_fs(30, [2017])
    misc = make_misc(fs, [2017]).with_columns(
        pl.when(pl.col("corp_code") == "00000000").then(False).otherwise(pl.col("capevt")).alias("capevt"),
        pl.when(pl.col("corp_code") == "00000000").then(True).otherwise(pl.col("capevt_rec")).alias("capevt_rec"),
        pl.when(pl.col("corp_code") == "00000000").then(True).otherwise(pl.col("capevt_rec_p1")).alias("capevt_rec_p1"),
    )  # fmt: skip
    j = cr.assemble_panel(fs, misc, capevt="judgment")
    r = cr.assemble_panel(fs, misc, capevt="record")
    a = j.filter(pl.col("corp_code") == "00000000").row(0, named=True)
    b = r.filter(pl.col("corp_code") == "00000000").row(0, named=True)
    assert a["capevt"] is False and b["capevt"] is True
    assert b["capevt_p1"] is True and a["capevt_p1"] == misc["capevt_p1"][0]
    for t in ("", "_p1", "_p2"):
        assert r[f"capevt{t}"].to_list() == r[f"capevt_rec{t}"].to_list()
        assert j[f"capevt{t}"].to_list() == misc[f"capevt{t}"].to_list()
    with pytest.raises(ValueError):
        cr.assemble_panel(fs, misc, capevt="x")


def test_assemble_panel_rows_without_misc_and_errors():
    fs = make_fs(10, [2017, 2018])
    misc = make_misc(fs, [2017])  # 2018은 misc 행이 없다
    p = cr.assemble_panel(fs, misc)
    assert p.height == fs.height  # 행 기준은 fs
    r18 = p.filter(pl.col("fy") == 2018)
    assert r18["dps"].null_count() == 10
    assert r18["capevt"].to_list() == [False] * 10  # CI-capevt-nomisc
    with pytest.raises(ValueError, match="유일"):
        cr.assemble_panel(fs, pl.concat([misc, misc]))
    with pytest.raises(ValueError, match="겹치는"):
        cr.assemble_panel(fs, misc.with_columns(pl.lit(1.0).alias("ta")))
    with pytest.raises(ValueError, match="필수 열"):
        cr.assemble_panel(fs, misc.drop("shares"))


# ---------------------------------------------------------------- 결과 입력 묶기
def test_dps_with_currency_same_nearest_none():
    ccy = pl.DataFrame(
        {
            "corp_code": ["A", "A", "B", "C"],
            "fy": [2017, 2019, 2018, 2018],
            "currency": ["KRW", "USD", "USD", None],
        }
    )
    dps = pl.DataFrame(
        {
            "corp_code": ["A", "A", "A", "B", "C", "D"],
            "report_year": [2017, 2018, 2020, 2020, 2019, 2019],
            "dps_t": [1.0] * 6,
        }
    )
    out, note = cr.dps_with_currency(dps, ccy)
    got = {(r["corp_code"], r["report_year"]): r["currency"] for r in out.iter_rows(named=True)}
    assert got[("A", 2017)] == "KRW"  # 같은 해
    assert got[("A", 2018)] == "KRW"  # 2017과 2019가 같은 거리 → 이른 해
    assert got[("A", 2020)] == "USD"  # 2019가 가장 가깝다
    assert got[("B", 2020)] == "USD"
    assert got[("C", 2019)] is None and got[("D", 2019)] is None
    assert note["n_same_year"] == 1 and note["n_nearest_year"] == 3 and note["n_none"] == 2
    assert out.height == dps.height


def test_outcome_inputs_columns():
    fs = make_fs(25, [2017, 2018])
    misc = make_misc(fs, [2017, 2018])
    panel = cr.assemble_panel(fs, misc)
    fsl, dps, ops = make_outcome_sources(25)
    cap = pl.DataFrame({"corp_code": ["00000001"], "year": [2018]})
    oi = cr.outcome_inputs(panel, fs_lat=fsl, dps_lat=dps, opinions=ops, capevt=cap)
    assert oi.formation.columns == ["corp_code", "fy", "in_universe", "currency", "te", "cap", "ni"]
    assert oi.fs_latest.columns == ["corp_code", "year", "te", "cap", "ni"]
    assert oi.dps_latest.columns == ["corp_code", "report_year", "currency", "dps_t", "dps_p1"]
    assert oi.capevt.columns == ["corp_code", "year"]
    assert oi.universe_years.columns == ["corp_code", "year"]
    inu = panel.filter(pl.col("in_universe")).select("corp_code", "fy").unique().height
    assert oi.universe_years.height == inu
    assert oi.dps_latest.height == dps.height
    assert oi.fs_latest_full is not None and "rcept_no" in oi.fs_latest_full.columns
    # W4가 그대로 받는다
    out = co.o2(oi.formation, oi.fs_latest, [2017, 2018])
    assert out.height > 0


# ---------------------------------------------------------------- O1 모드
def _ops(report_years):
    return pl.DataFrame(
        {
            "rcept_no": [f"r{y}{k}" for y in report_years for k in (0, 1)],
            "report_year": pl.Series([y for y in report_years for _ in (0, 1)], dtype=pl.Int32),
        }
    )


def test_decide_o1_mode_auto():
    odd = cr.decide_o1_mode(_ops([2017, 2019, 2021]))
    assert odd["mode"] == "fallback" and odd["n_even_year_reports"] == 0
    assert odd["n_odd_year_reports"] == 6 and odd["reports_by_bsns_year"]["2019"] == 2
    # 개발 구간(2017·2018)은 짝수 해 2018만 필요
    full = cr.decide_o1_mode(_ops([2017, 2018, 2019]), fys=cr.DEV_YEARS)
    assert full["mode"] == "full" and full["n_even_year_reports"] == 2
    assert full["needed_even_years"] == [2018] and full["missing_even_years"] == []
    # 판정 구간은 2020·2022·2024 모두 있어야 full — 일부 연도만 백필되면 fallback(CI-o1-auto)
    part = cr.decide_o1_mode(_ops([2019, 2020, 2021, 2022, 2023]))
    assert part["mode"] == "fallback" and part["missing_even_years"] == [2024]
    allj = cr.decide_o1_mode(_ops([2019, 2020, 2021, 2022, 2023, 2024, 2025]))
    assert allj["mode"] == "full" and allj["needed_even_years"] == [2020, 2022, 2024]
    assert odd["requested"] == "auto"
    # 강제 지정은 근거만 남기고 따른다
    forced = cr.decide_o1_mode(_ops([2017, 2019]), "full")
    assert forced["mode"] == "full" and forced["requested"] == "full"
    assert cr.decide_o1_mode(_ops([2018]), "fallback")["mode"] == "fallback"
    with pytest.raises(ValueError):
        cr.decide_o1_mode(_ops([2017]), "both")


def test_demoted_years_only_in_full_mode():
    flags = pl.DataFrame(
        {"year": [2018, 2020], "t": [2017, 2019], "rate": [0.1, 0.02], "ref_rate": [0.05, 0.02],
         "gap": [0.05, 0.0], "demote": [True, False]}
    )  # fmt: skip
    assert cr.demoted_formation_years(flags, "full") == [2017]
    assert cr.demoted_formation_years(flags, "fallback") == []


# ---------------------------------------------------------------- 판정 보호 (§7.2)
def test_judgment_refused_without_env_writes_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("읽기·계산 전에 멈춰야 합니다")

    monkeypatch.setattr(cr, "load_sources", boom)
    lake = Lake(root=tmp_path / "lake")
    out = tmp_path / "out_judgment"
    with pytest.raises(PermissionError, match=JUDGMENT_ENV):
        cr.run("judgment", lake, out)
    assert not out.exists()
    assert not (tmp_path / "lake").exists()
    # 기본 출력 경로에도 아무것도 안 생긴다
    with pytest.raises(PermissionError):
        cr.run("judgment", lake)
    assert not (tmp_path / "lake").exists()
    # CLI는 거부 메시지를 내고 2로 끝난다
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path / "lake2"))
    assert cr.main(["--period", "judgment", "--out-dir", str(tmp_path / "o2")]) == 2
    assert not (tmp_path / "o2").exists() and not (tmp_path / "lake2").exists()


def test_link_score_guards_judgment_years():
    oc = pl.DataFrame(
        {"corp_code": ["a"], "fy": [2019], "status": ["event"], "event": [1]},
        schema={"corp_code": pl.String, "fy": pl.Int64, "status": pl.String, "event": pl.Int64},
    )
    sc = pl.DataFrame({"corp_code": ["a"], "fy": [2019], "in_universe": [True], "c": [10.0]})
    with pytest.raises(PermissionError):
        cr.link_score(sc, oc, "c")
    dev = oc.with_columns(pl.lit(2018).alias("fy"))
    sc2 = sc.with_columns(pl.lit(2018).alias("fy"))
    out = cr.link_score(sc2, dev, "c")
    assert out.columns == ["corp_code", "fy", "score", "event"] and out.height == 1


# ---------------------------------------------------------------- checks
def test_checks_never_call_outcome_functions(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("checks가 결과·판정 함수를 불렀습니다")

    for name in ("o1", "o2", "o3", "o4", "o4_relaxed", "o4_dev_t1", "unobserved_scored_counts"):
        monkeypatch.setattr(co, name, boom)
    for name in ("judge_outcomes", "record_cells", "outcome_stats"):
        monkeypatch.setattr(cj, name, boom)
    monkeypatch.setattr(cr, "link_score", boom)
    monkeypatch.setattr(cr, "compute_outcomes", boom)

    src = make_sources(tmp_path, "checks", n=60, with_outcomes=False)
    cov_fs = pl.DataFrame(
        {"fy": [2019, 2019], "scope": ["all", "in_universe"],
         "item": ["raw_only_corp_years", "raw_only_corp_years"], "n": [5, 4]}
    )  # fmt: skip
    cov_misc = pl.DataFrame({"fy": [2019], "n_rows": [10], "dps_null_share": [0.1]})
    src.cov_fs, src.cov_misc = cov_fs, cov_misc
    out = tmp_path / "chk"
    r = cr.run("checks", src.lake, out, sources=src, n_boot=5)
    names = {p.name for p in out.iterdir()}
    assert {"score_inputs.tsv", "opinion_missing_rate.tsv", "even_year_flags.tsv", "manifest.json",
            "coverage_fs.tsv", "coverage_misc.tsv", "prereg_numbers.json"} <= names  # fmt: skip
    assert "result.json" not in names and "dev_samples.tsv" not in names
    si = pl.read_csv(out / "score_inputs.tsv", separator="\t")
    assert set(si.columns) == {"fy", "n_universe_rows", "n_c1", "n_c2", "n_c3", "n_c", "n_f_sum",
                               "n_f_ltb_req"}  # fmt: skip
    assert si["fy"].to_list() == list(cr.CHECK_YEARS)  # 값·분포 열이 없다(수만)
    nums = json.loads((out / "prereg_numbers.json").read_text())
    assert nums["raw_only_corp_years"]["all"]["2019"] == 5
    assert nums["misc:dps_null_share"]["2019"] == pytest.approx(0.1)
    man = json.loads((out / "manifest.json").read_text())
    assert man["period"] == "checks" and man["o1"]["mode"] == "fallback"
    assert r["manifest"]["bootstrap"] == {"seed": cj.SEED, "n_boot": 5}


def test_score_input_counts_only_counts():
    sc = pl.DataFrame(
        {
            "fy": [2018, 2018, 2018],
            "in_universe": [True, True, False],
            "c1": [1.0, None, 5.0], "c2": [1.0, 2.0, 3.0], "c3": [None, None, 1.0],
            "c": [1.0, None, 1.0], "f_sum": [3, 4, 5], "f_ltb_req": [None, 2, 2],
        }
    )  # fmt: skip
    out = cr.score_input_counts(sc)
    assert out.row(0, named=True) == {
        "fy": 2018, "n_universe_rows": 2, "n_c1": 1, "n_c2": 2, "n_c3": 0, "n_c": 1,
        "n_f_sum": 2, "n_f_ltb_req": 1,
    }  # fmt: skip


def test_prereg_numbers_shape():
    cov = pl.DataFrame(
        {
            "fy": [2018, 2018, 2018], "scope": ["all", "all", "in_universe"],
            "item": ["no_origin_keys", "key_raw_only:revision_after_base", "no_origin_keys"],
            "n": [3, 2, 1],
        }
    )  # fmt: skip
    m = pl.DataFrame(
        {"fy": [2018], "n_rows": [9], "dps_null_share": [0.25], "n_dps_after_base": [4]}
    )
    out = cr.prereg_numbers(cov, m)
    assert out["no_origin_keys"] == {"all": {"2018": 3}, "in_universe": {"2018": 1}}
    assert out["key_raw_only:revision_after_base"] == {"all": {"2018": 2}}
    assert out["misc:dps_null_share"] == {"2018": 0.25}
    assert out["misc:n_dps_after_base"] == {"2018": 4}


# ---------------------------------------------------------------- dev 끝에서 끝까지
@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dev")
    src = make_sources(tmp, "dev", n=170)
    out = tmp / "out"
    r = cr.run("dev", src.lake, out, sources=src, n_boot=30, seed=7)
    return src, out, r


def test_dev_run_files_and_result(dev_run):
    src, out, r = dev_run
    names = {p.name for p in out.iterdir()}
    assert {"result.json", "manifest.json", "dev_samples.tsv", "outcomes_summary.tsv", "by_year.tsv",
            "cells_and_records.tsv", "status_counts.tsv", "unobserved_scored.tsv"} <= names  # fmt: skip
    res = json.loads((out / "result.json").read_text())
    assert res["period"] == "dev" and res["years"] == [2017, 2018]
    assert res["judge"]["order"] == ["O1", "O2", "O3", "O4"]
    assert res["o1"]["mode"] == "fallback" and res["o1"]["demoted_formation_years"] == []
    assert set(res["cells"]) == {"C1", "C2", "C3", "F"}
    assert all(set(v) == {"O1", "O2", "O3", "O4"} for v in res["cells"].values())
    rec = res["records"]
    for k in ("prior_t_minus_1_report", "capevt_record", "circular_removed", "two_dims",
              "f_ltb_required", "o1_fallback", "o2_full", "o4_relaxed", "vintage_only",
              "universe_financial_included", "universe_non_december_included",
              "universe_spac_included", "f_vs_composite", "o1_fold_any_report"):  # fmt: skip
        assert k in rec, k
    assert "skipped" in rec["o4_relaxed"]  # dev는 t+3이 없어 건너뜀
    assert "todo_interpretation_alternatives" not in rec
    for k in (cr.REC_D3, cr.REC_D5, cr.REC_C3_LATEST):
        assert k in rec and "skipped" not in rec[k], k
    assert set(rec[cr.REC_D3]) >= {"c", "f_sum", "n_affected"}
    assert set(rec[cr.REC_D3]["c"]) == {"O1", "O2", "O3", "O4"}
    assert rec[cr.REC_D3]["n_affected"]["n_multi_reports"] == src.dps_multi.height
    assert (
        set(rec[cr.REC_D5]["c"]) == {"O1"}
        and rec[cr.REC_D5]["n_affected"]["n_rows_became_null"] > 0
    )
    assert rec[cr.REC_C3_LATEST]["pit_violation"] is True and rec[cr.REC_C3_LATEST]["note"]
    assert rec[cr.REC_C3_LATEST]["n_affected"]["shares_p1"]["n_value_changed"] > 0
    assert set(rec["circular_removed"]) == {"c_o2", "c_o3", "c_o4"}
    assert set(rec["circular_removed"]["c_o2"]) == {"O2"}
    assert rec["capevt_record"]["o3_excluded_capevt"]["record_list"] >= (
        rec["capevt_record"]["o3_excluded_capevt"]["judgment_list"]
    )
    # dev의 O4는 t+1
    assert "t+1" in res["o4_definition"]
    assert {s["fy"] for s in res["status_counts"] if s["outcome"] == "O4"} <= {2017, 2018}


def test_dev_run_default_path_ignores_alt_inputs(dev_run, tmp_path):
    """대안 입력을 없애도 판정 통계·16칸·분모 구성은 한 글자도 안 바뀐다(기록 항목만 건너뜀)."""
    src, out, _ = dev_run
    base = json.loads((out / "result.json").read_text())
    bare = make_sources(tmp_path, "dev", n=170)
    bare.dps_multi = bare.opinions_anchor_off = bare.misc_latest = None
    out2 = tmp_path / "out2"
    cr.run("dev", bare.lake, out2, sources=bare, n_boot=30, seed=7)
    other = json.loads((out2 / "result.json").read_text())
    assert all("skipped" in other["records"][k] for k in (cr.REC_D3, cr.REC_D5, cr.REC_C3_LATEST))
    for k in base:
        if k != "records":
            assert base[k] == other[k], k
    for k, v in base["records"].items():
        if k not in (cr.REC_D3, cr.REC_D5, cr.REC_C3_LATEST):
            assert other["records"][k] == v, k
    for name in ("outcomes_summary.tsv", "by_year.tsv", "status_counts.tsv"):
        assert (out / name).read_text() == (out2 / name).read_text(), name


def test_dps_null_for_multi_panel_and_latest():
    panel = pl.DataFrame(
        {"corp_code": ["A", "B"], "fy": [2017, 2017], "dps_rcept_no": ["r1", "r2"],
         "dps": [100.0, 50.0], "dps_p1": [90.0, 40.0], "dps_p2": [80.0, 30.0], "c": [1, 2]}
    )  # fmt: skip
    multi = pl.DataFrame({"rcept_no": ["r1"]})
    out = cr.dps_null_for_multi(panel, multi)
    assert out.row(0, named=True) == {
        "corp_code": "A", "fy": 2017, "dps_rcept_no": "r1", "dps": None, "dps_p1": None,
        "dps_p2": None, "c": 1,
    }  # fmt: skip
    assert out.row(1, named=True)["dps"] == 50.0 and out.row(1, named=True)["dps_p2"] == 30.0
    lat = pl.DataFrame(
        {"corp_code": ["A", "B"], "report_year": [2017, 2017], "rcept_no": ["r1", "r2"],
         "dps_t": [100.0, 50.0], "dps_p1": [90.0, 40.0]}
    )  # fmt: skip
    o = cr.dps_latest_null_for_multi(lat, multi)
    assert o["dps_t"].to_list() == [None, 50.0] and o["dps_p1"].to_list() == [None, 40.0]
    # 대상이 없으면 그대로
    none = cr.dps_null_for_multi(
        panel, pl.DataFrame({"rcept_no": []}, schema={"rcept_no": pl.String})
    )
    assert none.equals(panel)


def test_d5_and_c3_counts():
    op = pl.DataFrame(
        {"corp_code": ["A", "A", "A"], "rcept_no": ["r", "r", "q"], "row_ordinal": [0, 1, 0],
         "fiscal_year": pl.Series([2020, 2019, 2018], dtype=pl.Int32),
         "opinion_class": ["clean", None, "clean"]}
    )  # fmt: skip
    off = op.with_columns(pl.Series("fiscal_year", [2020, None, 2018], dtype=pl.Int32))
    c = cr.d5_counts(op, off)
    assert c["n_rows_became_null"] == 1 and c["n_rows_year_changed"] == 0
    assert c["n_rows_became_null_with_opinion_class"] == 0 and c["n_reports_affected"] == 1
    base = pl.DataFrame({"corp_code": ["A", "B"], "fy": [2017, 2017]}).with_columns(
        *[pl.lit(None, pl.Float64).alias(k) for k in cr.MISC_VALUE_COLS]
    ).with_columns(pl.Series("shares", [10.0, 20.0]))  # fmt: skip
    late = base.with_columns(
        pl.Series("shares", [10.0, 25.0]), pl.Series("dps", [1.0, None], dtype=pl.Float64)
    )
    cc = cr.c3_latest_counts(base, late)
    assert cc["dps"] == {"n_filled_by_latest": 1, "n_value_changed": 0}
    assert cc["shares"] == {"n_filled_by_latest": 0, "n_value_changed": 1}


def test_dev_run_manifest(dev_run):
    _, out, _ = dev_run
    man = json.loads((out / "manifest.json").read_text())
    for k in ("raw_snapshot", "derived_snapshot", "inputs", "xbrl_cache", "code_sha256",
              "modeler_git_head", "prereg", "bootstrap", "uv_lock_sha256", "args", "o1",
              "even_year_flags", "elapsed_sec", "peak_rss_mb", "outputs"):  # fmt: skip
        assert k in man, k
    assert man["bootstrap"] == {"seed": 7, "n_boot": 30}
    assert "company_run.py" in man["code_sha256"] and "company_score.py" in man["code_sha256"]
    assert len(man["code_sha256"]["company_run.py"]) == 64
    assert man["inputs"]["stock_master"]["exists"] is False  # 합성 레이크는 파일이 없다
    assert set(man["inputs"]) >= set(cr.RAW_TABLES) | set(cr.DERIVED_TABLES)
    assert "result.json" in man["outputs"] and len(man["outputs"]["result.json"]["sha256"]) == 64


def test_dev_samples_shape_and_determinism(dev_run, tmp_path):
    src, out, _ = dev_run
    s = pl.read_csv(out / "dev_samples.tsv", separator="\t", infer_schema_length=0)
    for o in cj.OUTCOME_ORDER:
        d = s.filter(pl.col("outcome") == o)
        assert d.filter(pl.col("event") == "1").height <= cr.SAMPLE_EVENTS
        assert d.filter(pl.col("event") == "0").height <= cr.SAMPLE_NON_EVENTS
        assert d.height > 0
    need = {"corp_code", "stock_code", "fy", "c", "c1", "c2", "c3", "f_sum", "op_t_raw",
            "op_t_rcept", "op_t1_raw", "op_t1_rcept", "te_t", "cap_t", "te_t1", "cap_t1",
            "dps_rcept_t1", "dps_prev", "dps_cur", "ni_t", "ni_t1"}  # fmt: skip
    assert need <= set(s.columns)
    o1 = s.filter(pl.col("outcome") == "O1")
    assert o1["op_t1_raw"].null_count() == 0 and o1["op_t_rcept"].null_count() == 0
    o4 = s.filter(pl.col("outcome") == "O4")
    assert o4["ni_t1"].null_count() == 0 and o4["op_t_raw"].null_count() == o4.height
    # 같은 시드 → 같은 표본
    r2 = cr.run("dev", src.lake, tmp_path / "again", sources=src, n_boot=3, seed=7)
    s2 = pl.read_csv(r2["files"]["dev_samples.tsv"], separator="\t", infer_schema_length=0)
    assert s.select("outcome", "corp_code", "fy").equals(s2.select("outcome", "corp_code", "fy"))


def test_dev_samples_values_match_inputs(dev_run):
    src, out, _ = dev_run
    s = pl.read_csv(out / "dev_samples.tsv", separator="\t", infer_schema_length=0)
    o4 = s.filter(pl.col("outcome") == "O4").with_columns(
        pl.col("fy").cast(pl.Int64), pl.col("ni_t1").cast(pl.Float64)
    )
    ref = src.fs_latest.select(
        "corp_code", (pl.col("year") - 1).alias("fy"), pl.col("ni").alias("r")
    )
    j = o4.join(ref, on=["corp_code", "fy"])
    assert j.height == o4.height
    assert (j["ni_t1"] - j["r"]).abs().max() < 1e-3
    # 사건이면 ni_t1 < 0, 비사건이면 ≥ 0 (정의와 표본이 맞는다)
    assert all((v < 0) == (e == "1") for v, e in zip(j["ni_t1"], j["event"]))


def test_dev_no_judgment_years_in_outputs(dev_run):
    _, out, _ = dev_run
    for name in ("by_year.tsv", "status_counts.tsv", "unobserved_scored.tsv", "dev_samples.tsv"):
        d = pl.read_csv(out / name, separator="\t", infer_schema_length=0)
        assert set(d["fy"].cast(pl.Int64).to_list()) <= {2017, 2018}, name


def test_empty_outcome_frame_does_not_break_judge():
    empty = pl.DataFrame(
        schema={"corp_code": pl.String, "fy": pl.Int64, "score": pl.Float64, "event": pl.Int64}
    )
    res = cj.judge_outcomes({"O1": empty}, n_boot=5)
    assert res["outcomes"]["O1"]["stats"]["n_events"] == 0


# ---------------------------------------------------------------- judgment (합성 자료, 환경변수 있음)
def test_judgment_synthetic_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv(JUDGMENT_ENV, "1")
    ap = _approve(tmp_path, monkeypatch)
    src = make_sources(tmp_path, "judgment", n=130)
    out = tmp_path / "j"
    r = cr.run("judgment", src.lake, out, sources=src, n_boot=10, seed=3)
    res = json.loads((out / "result.json").read_text())
    assert res["period"] == "judgment" and res["years"] == list(range(2019, 2025))
    assert not (out / "dev_samples.tsv").exists()  # 판정 구간 행은 표본으로 내지 않는다
    assert "t+3" in res["o4_definition"]
    o4_years = {s["fy"] for s in res["status_counts"] if s["outcome"] == "O4"}
    assert o4_years <= {2019, 2020, 2021, 2022}
    assert (
        not isinstance(res["records"]["o4_relaxed"], str)
        and "skipped" not in res["records"]["o4_relaxed"]
    )
    assert r["manifest"]["period"] == "judgment"
    ia = r["manifest"]["interp_table_approved"]
    assert ia["path"] == str(ap) and ia["sha256"] == cr.sha256_file(ap)
    assert "o4_continuous" in res["records"] and "o4_continuous_dev_t1" not in res["records"]
    assert res["extra_records_enabled"] == []


def test_judgment_refused_without_approved_interp_table(tmp_path, monkeypatch):
    """환경변수(동결 확인)가 있어도 승인본 사본이 없으면 파일을 쓰기 전에 거부, 둘 다 있으면 진행."""

    def boom(*a, **k):
        raise AssertionError("읽기·계산 전에 멈춰야 합니다")

    monkeypatch.setenv(JUDGMENT_ENV, "1")
    lake = Lake(root=tmp_path / "lake")
    out = tmp_path / "out_j"
    monkeypatch.setattr(cr, "load_sources", boom)
    with pytest.raises(PermissionError, match="interp_table_v1_approved.md"):
        cr.run("judgment", lake, out)
    assert not out.exists() and not (tmp_path / "lake").exists()
    # 환경변수가 가리키는 파일이 없어도 거부, 메시지에 그 경로
    monkeypatch.setenv(cr.INTERP_TABLE_ENV, str(tmp_path / "nope.md"))
    with pytest.raises(PermissionError, match="nope.md"):
        cr.run("judgment", lake, out)
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path / "lake2"))
    assert cr.main(["--period", "judgment", "--out-dir", str(out)]) == 2
    assert not out.exists()
    monkeypatch.undo()
    # dev는 거부하지 않고 manifest에 sha256 null로만 적는다
    monkeypatch.delenv(cr.INTERP_TABLE_ENV, raising=False)
    src = make_sources(tmp_path, "dev", n=60)
    r = cr.run("dev", src.lake, tmp_path / "dv", sources=src, n_boot=5, seed=1)
    ia = r["manifest"]["interp_table_approved"]
    assert ia["sha256"] is None and ia["path"].endswith("interp_table_v1_approved.md")


def test_default_interp_table_approved_path(tmp_path, monkeypatch):
    lake = Lake(root=tmp_path, raw_snapshot="2026-09-30")
    assert cr.default_interp_table_approved(lake) == (
        tmp_path / "kr/output/quality_score_company_provenance/interp_table_v1_approved.md"
    )
    monkeypatch.setenv(cr.INTERP_TABLE_ENV, "/x/y.md")
    assert str(cr.default_interp_table_approved(lake)) == "/x/y.md"


def test_cli_dev_arguments(tmp_path, monkeypatch):
    seen = {}

    def fake_run(period, lake, out_dir, **kw):
        seen.update(
            period=period, raw=lake.raw_snapshot, der=lake.derived_snapshot, out=out_dir, **kw
        )
        return {
            "out_dir": tmp_path,
            "manifest": {"o1": {"mode": "fallback", "n_even_year_reports": 0},
                         "elapsed_sec": 1, "peak_rss_mb": 2},
            "result": {"result": {"judge": {"order": []}}},
        }  # fmt: skip

    monkeypatch.setattr(cr, "run", fake_run)
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    rc = cr.main(["--period", "dev", "--raw-snapshot", "2026-09-30", "--derived-snapshot", "2026-09-29",
                  "--seed", "5", "--n-boot", "40", "--o1-mode", "full", "--out-dir", "X"])  # fmt: skip
    assert rc == 0
    assert seen["period"] == "dev" and seen["seed"] == 5 and seen["n_boot"] == 40
    assert seen["o1_mode"] == "full" and seen["out"] == "X" and seen["raw"] == "2026-09-30"
    assert seen["args"]["period"] == "dev"


def test_table_info_and_default_paths(tmp_path):
    d = tmp_path / "t"
    (d / "p=1").mkdir(parents=True)
    (d / "p=1" / "a.parquet").write_bytes(b"123")
    (d / "b.parquet").write_bytes(b"45")
    a = cr.table_info(d)
    assert a["files"] == 2 and a["bytes"] == 5 and len(a["listing_sha256"]) == 64
    (d / "c.parquet").write_bytes(b"6")
    assert (
        cr.table_info(d)["listing_sha256"] != a["listing_sha256"]
    )  # 파일 목록이 바뀌면 해시도 바뀐다
    assert cr.table_info(tmp_path / "none")["exists"] is False
    lake = Lake(root=tmp_path, raw_snapshot="2026-09-30")
    assert (
        cr.default_out_dir(lake, "dev")
        == tmp_path / "kr/output/quality_score_company_dev_2026-09-30"
    )
    assert cr.default_xbrl_cache(lake) == (
        tmp_path / "kr/output/quality_score_company_cache_2026-09-30/xbrl_values.parquet"
    )


# ---------------------------------------------------------------- §12.4 C 확인표 (W8)
def test_checklist_c_has_all_keys_and_missing_without_dirs(tmp_path):
    lake = Lake(root=tmp_path / "nolake", raw_snapshot="2026-10-18", derived_snapshot="2026-10-17")
    o1 = cr.decide_o1_mode(make_outcome_sources(5)[2])
    items = cr.build_checklist_c(
        lake, o1_info=o1, even_flags=[], numbers={}, maps_dir=tmp_path / "nomaps",
        xbrl_cache=None, prereg=tmp_path / "nofile.md",
    )  # fmt: skip
    by = {i["item"]: i for i in items}
    assert set(cr.CHECKLIST_KEYS) <= set(by)
    assert all(set(i) == {"item", "value", "status", "source"} for i in items)
    assert all(i["status"] in {"ok", "check", "missing"} for i in items)
    for k in ("raw_snapshot", "derived_snapshot", "raw_success_marker", "derived_success_marker",
              "calendar_last_day", "receipt_last_date", "universe_spac", "maps_manifest_sha256",
              "maps_new_strings", "par_change_summary", "prereg_numbers_summary", "prereg_sha256",
              "xbrl_cache"):  # fmt: skip
        assert by[k]["status"] == "missing", k
    assert by["o1_mode"]["value"]["mode"] == "fallback"
    assert by["modeler_git_head"]["status"] in {"ok", "missing"}


def test_checklist_c_written_by_checks_run(tmp_path):
    src = make_sources(tmp_path, "checks", n=40, with_outcomes=False)
    src.cov_fs = pl.DataFrame(
        {"fy": [2019], "scope": ["all"], "item": ["raw_only_corp_years"], "n": [5]}
    )
    src.cov_misc = pl.DataFrame({"fy": [2019], "n_rows": [10], "dps_null_share": [0.1]})
    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "manifest.json").write_text(
        json.dumps({"rules_version": "x", "raw_snapshot": "2026-09-30", "module_sha256": "y",
                    "stats": {"par_change_check": {"n_par_changed_corp_years": 3}}})
    )  # fmt: skip
    (maps / "compare_summary.json").write_text(
        json.dumps({"n_new_strings_total": 2, "n_class_changed_total": 0,
                    "old_raw_snapshot": "2026-09-30", "rules_version": {"same": True}})
    )  # fmt: skip
    out = tmp_path / "chk"
    r = cr.run("checks", src.lake, out, sources=src, n_boot=5, maps_dir=maps)
    assert {"checklist_c.json", "checklist_c.tsv"} <= {p.name for p in out.iterdir()}
    assert "checklist_c.json" in r["manifest"]["outputs"]
    doc = json.loads((out / "checklist_c.json").read_text())
    by = {i["item"]: i for i in doc["items"]}
    assert set(cr.CHECKLIST_KEYS) <= set(by)
    assert by["maps_new_strings"]["status"] == "check"  # 새 문자열 2건
    assert by["maps_new_strings"]["value"]["n_new_strings_total"] == 2
    assert by["maps_rules_version"]["status"] == "check"  # 코드의 규칙 버전과 다르다
    assert by["par_change_summary"]["status"] == "ok"
    assert by["prereg_numbers_summary"]["value"]["raw_only_corp_years"] == {"all": 5}
    assert by["raw_snapshot"]["status"] == "missing"  # 합성 레이크에는 디렉터리가 없다
    assert doc["status_counts"]["missing"] >= 1
    tsv = (out / "checklist_c.tsv").read_text().splitlines()
    assert tsv[0] == "item\tstatus\tvalue\tsource" and len(tsv) == 1 + len(cr.CHECKLIST_KEYS)


# ---------------------------------------------------------------- 기록용 항목 넷 (항상 켬)
def test_dev_new_records_always_on_and_extra_off_by_default(dev_run):
    src, out, r = dev_run
    res = json.loads((out / "result.json").read_text())
    rec = res["records"]
    # 1. raw 한 층 단독: 점수만 바꿔 O1~O4, 영향 개수
    ro = rec["raw_layer_only"]
    assert set(ro["c"]) == {"O1", "O2", "O3", "O4"} and "skipped" not in ro
    na = ro["n_affected"]
    assert na["n_scored_variant"] < na["n_scored_default"] and na["n_scored_default_only"] > 0
    assert na["n_scored_both"] + na["n_scored_default_only"] == na["n_scored_default"]
    # 2. O4 연속 판: 개발은 t+1만
    assert "o4_continuous" not in rec and "o4_continuous_dev_t1" in rec
    oc = rec["o4_continuous_dev_t1"]
    assert oc["formation_years"] == [2017, 2018] and "t+1" in oc["definition"]
    assert oc["n_rows"] > 0 and -1 <= oc["rho_pooled"] <= 1
    assert oc["ci95"][0] <= oc["rho_pooled"] <= oc["ci95"][1] or oc["n_boot"] < 100
    assert [y["fy"] for y in oc["by_year"]] == [2017, 2018]
    # 3. 가용일 어긋남은 manifest에
    assert r["manifest"]["availability_mismatch"] == {"vintage_rcept": 100, "vintage_mismatch": 3}
    # 4. 쓴 성분 수: 값별 회사 수
    cc = pl.DataFrame(res["score_component_counts"])
    assert set(cc["component"]) == {"c1_n", "c2_n", "c3_n"} and set(cc["fy"]) == {2017, 2018}
    assert (out / "score_component_counts.tsv").exists()
    # 켜지 않은 제안 판은 없다. 켠 목록은 빈 목록
    for k in cr.EXTRA_RECORD_NAMES:
        assert k not in rec
    assert res["extra_records_enabled"] == [] and r["manifest"]["extra_records_enabled"] == []
    # 새 항목은 기존 cells_and_records.tsv가 아니라 새 표로
    old = (out / "cells_and_records.tsv").read_text()
    new = (out / "cells_and_records_extra.tsv").read_text()
    assert "vintage_only:" in old and "raw_layer_only:" not in old
    assert "raw_layer_only:" in new


def test_o4_continuous_exact_values_and_exclusions():
    from types import SimpleNamespace

    corps = [f"c{i}" for i in range(6)]
    # 점수 c가 ROE와 같은 순서인 해(2019): ρ = 1. c5는 t+2 te ≤ 0이라 제외, c4는 t+3 관측 없음
    rows = [(c, 2019, float(10 * i)) for i, c in enumerate(corps)]
    scores = pl.DataFrame(rows, schema=["corp_code", "fy", "c"], orient="row").with_columns(
        pl.lit(True).alias("in_universe")
    )
    fl = []
    for i, c in enumerate(corps):
        for k in (1, 2, 3):
            te = -1.0 if (c == "c5" and k == 2) else 100.0
            if c == "c4" and k == 3:
                continue
            fl.append(
                {"corp_code": c, "year": 2019 + k, "te": te, "cap": 1.0, "ni": float(i + 1) * 10}
            )
    ctx = SimpleNamespace(
        period="judgment", years=[2019], scores=scores, seed=1, n_boot=50,
        oi=SimpleNamespace(fs_latest=pl.DataFrame(fl)),
    )  # fmt: skip
    import os

    os.environ[JUDGMENT_ENV] = "1"
    try:
        rec = cr.o4_continuous_record(ctx)
    finally:
        del os.environ[JUDGMENT_ENV]
    assert rec["n_scored_candidates"] == 6 and rec["n_rows"] == 4  # c4·c5 제외
    assert rec["rho_pooled"] == pytest.approx(1.0)
    assert rec["by_year"] == [{"fy": 2019, "n": 4, "rho": pytest.approx(1.0)}]
    assert rec["ci95"] == [pytest.approx(1.0), pytest.approx(1.0)] or rec["n_boot_nan"] > 0


def test_o4_continuous_guards_judgment_years():
    from types import SimpleNamespace

    ctx = SimpleNamespace(period="judgment", years=[2019], scores=None, seed=1, n_boot=2, oi=None)
    with pytest.raises(PermissionError):  # 확인 환경변수 없이는 결과 변수 계산 금지(§7.2)
        cr.o4_continuous_record(ctx)


# ---------------------------------------------------------------- 승인 대기 제안 판 (스위치)
def test_extra_records_on_when_named(tmp_path):
    src = make_sources(tmp_path, "dev", n=120)
    out = tmp_path / "ex"
    r = cr.run(
        "dev", src.lake, out, sources=src, n_boot=10, seed=5,
        extra_records=("layer_literal", "pooled_cut", "g4_all_years"),
    )  # fmt: skip
    res = json.loads((out / "result.json").read_text())
    rec = res["records"]
    assert res["extra_records_enabled"] == ["layer_literal", "pooled_cut", "g4_all_years"]
    assert r["manifest"]["extra_records_enabled"] == res["extra_records_enabled"]
    ll = rec["layer_literal"]
    assert set(ll["c"]) == {"O1", "O2", "O3", "O4"} and ll["n_affected"]["n_scored_default"] > 0
    assert ll["n_affected"]["n_scored_both"] > 0 and "definition" in ll
    pc = rec["pooled_cut"]
    assert pc["cut"] == "pooled" and set(pc) >= {"O1", "O2", "O3", "O4"}
    assert pc["O2"]["quintiles"]["cut"] == "pooled" and pc["O2"]["capture"]["20"]["cut"] == "pooled"
    assert "pass" in pc["O2"]["g3"]
    g4 = rec["g4_all_years"]
    assert (
        g4["O2"]["all"]["denominator_rule"] == "all"
        and g4["O2"]["default"]["denominator_rule"] == "counted"
    )
    assert g4["O2"]["all"]["denominator"] == g4["O2"]["all"]["n_years"]
    # 기본 판정 통계는 그대로(켜도 판정 규칙·기본 출력 불변)
    out0 = tmp_path / "ex0"
    cr.run("dev", src.lake, out0, sources=src, n_boot=10, seed=5)
    base = json.loads((out0 / "result.json").read_text())
    for k in ("judge", "cells", "status_counts", "o1"):
        assert base[k] == res[k], k
    for name in ("outcomes_summary.tsv", "by_year.tsv", "cells_and_records.tsv"):
        assert (out0 / name).read_text() == (out / name).read_text(), name
    # 알 수 없는 이름은 거부
    with pytest.raises(ValueError):
        cr.run("dev", src.lake, tmp_path / "bad", sources=src, extra_records=("nope",))


def test_approved_extra_records_constant_and_cli(tmp_path, monkeypatch):
    src = make_sources(tmp_path, "dev", n=80)
    monkeypatch.setattr(cr, "APPROVED_EXTRA_RECORDS", ("pooled_cut",))
    r = cr.run(
        "dev", src.lake, tmp_path / "a", sources=src, n_boot=5, extra_records=("g4_all_years",)
    )
    rec = r["result"]["result"]["records"]
    assert "pooled_cut" in rec and "g4_all_years" in rec and "layer_literal" not in rec
    # CLI: 쉼표 목록을 읽고 모르는 이름은 종료 코드 2(argparse)
    seen = {}

    def fake_run(period, lake, out_dir, **kw):
        seen.update(kw)
        return {"out_dir": tmp_path, "manifest": {"o1": {"mode": "fallback", "n_even_year_reports": 0},
                                                   "elapsed_sec": 1, "peak_rss_mb": 2},
                "result": {"result": {"judge": {"order": []}}}}  # fmt: skip

    monkeypatch.setattr(cr, "run", fake_run)
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    assert cr.main(["--extra-records", "layer_literal, pooled_cut"]) == 0
    assert seen["extra_records"] == ("layer_literal", "pooled_cut")
    with pytest.raises(SystemExit) as e:
        cr.main(["--extra-records", "nope"])
    assert e.value.code == 2


# ---------------------------------------------------------------- checks: 가용일·성분 수·승인본
def test_checks_new_files_and_checklist_items(tmp_path, monkeypatch):
    src = make_sources(tmp_path, "checks", n=40, with_outcomes=False)
    out = tmp_path / "chk2"
    ap = _approve(tmp_path, monkeypatch)
    r = cr.run("checks", src.lake, out, sources=src, n_boot=5)
    am = json.loads((out / "availability_mismatch.json").read_text())
    assert am == {"vintage_rcept": 100, "vintage_mismatch": 3}
    assert r["manifest"]["availability_mismatch"] == am
    cc = pl.read_csv(out / "score_component_counts.tsv", separator="\t")
    assert cc.columns == ["fy", "component", "n_components", "n_corps"]
    assert set(cc["fy"]) == set(cr.CHECK_YEARS) and set(cc["component"]) == {"c1_n", "c2_n", "c3_n"}
    # in_universe 행 수와 맞는다: 연도·성분마다 n_corps 합 = n_universe_rows
    si = pl.read_csv(out / "score_inputs.tsv", separator="\t")
    tot = (
        cc.group_by("fy", "component")
        .agg(pl.col("n_corps").sum())
        .filter(pl.col("component") == "c1_n")
    )
    assert tot.sort("fy")["n_corps"].to_list() == si.sort("fy")["n_universe_rows"].to_list()
    nums = (
        json.loads((out / "prereg_numbers.json").read_text())
        if (out / "prereg_numbers.json").exists()
        else {}
    )
    assert "vintage_mismatch" not in nums  # prereg_numbers.json은 그대로
    doc = json.loads((out / "checklist_c.json").read_text())
    by = {i["item"]: i for i in doc["items"]}
    assert (
        by["availability_mismatch"]["status"] == "ok" and by["availability_mismatch"]["value"] == am
    )
    assert by["interp_table_approved"]["status"] == "check"
    assert by["interp_table_approved"]["value"] == {"path": str(ap), "sha256": cr.sha256_file(ap)}
    # 승인본이 없으면 missing (checks는 거부하지 않는다)
    monkeypatch.setenv(cr.INTERP_TABLE_ENV, str(tmp_path / "none.md"))
    out2 = tmp_path / "chk3"
    cr.run("checks", src.lake, out2, sources=src, n_boot=5)
    by2 = {i["item"]: i for i in json.loads((out2 / "checklist_c.json").read_text())["items"]}
    assert by2["interp_table_approved"]["status"] == "missing"
    assert by2["interp_table_approved"]["value"]["sha256"] is None
