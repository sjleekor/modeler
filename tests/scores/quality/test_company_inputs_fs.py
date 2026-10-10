"""company_inputs_fs 시험 (사전등록 20261010_quality_score §5.1·§5.2·§7.2).

합성 프레임으로 판본 선택(가용일 경계·연결 우선·정정본만 있는 키)·층 규칙·지급이자 순서·전기 값
출처·기록용 (a)를 확인하고, 레이크가 있으면 07 §3의 002210 시험과 vintage·raw 대조를 돌린다.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from modeler.scores.quality import company_inputs_fs as F
from modeler.scores.quality.company_common import Lake

CORP = "00000001"


# ---------------------------------------------------------------- 합성 자료 도우미
def _v(rows: list[dict]) -> pl.DataFrame:
    base = {"corp_code": CORP, "fy": 2024, "fs_div": "CFS", "rcept_no": "20250101000001"}
    df = pl.DataFrame([base | r for r in rows])
    return df.with_columns(pl.col("fy").cast(pl.Int32))


def _empty_r() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "corp_code": pl.String,
            "fy": pl.Int32,
            "fs_div": pl.String,
            "rcept_no": pl.String,
        }
    )


def _empty_x() -> pl.DataFrame:
    return pl.DataFrame(schema={"rcept_no": pl.String, "fy": pl.Int32, "fs_div": pl.String})


def _avail(pairs: dict[str, date]) -> pl.DataFrame:
    return pl.DataFrame(
        {"rcept_no": list(pairs), "avail_date": list(pairs.values())},
        schema={"rcept_no": pl.String, "avail_date": pl.Date},
    )


def _universe(*corps: str, in_attr: bool = True) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "corp_code": list(corps),
            "stock_code": [f"{i:06d}" for i in range(len(corps))],
            "market": ["KOSPI"] * len(corps),
            "is_spac": [False] * len(corps),
            "is_financial": [False] * len(corps),
            "acc_mt": [12] * len(corps),
            "in_universe_attr": [in_attr] * len(corps),
        }
    )


def _inputs(cv: pl.DataFrame, fys=(2024,), uni=None) -> F.FsInputs:
    return F.FsInputs(
        lake=None,  # type: ignore[arg-type]
        fys=list(fys),
        universe=uni if uni is not None else _universe(CORP),
        cv=cv,
        receipts_total=0,
    )


def _cands(v_rows=(), r_rows=(), x=None, avail=None) -> pl.DataFrame:
    v = _v(list(v_rows)) if v_rows else _v([{}]).clear()
    r = _v(list(r_rows)) if r_rows else _empty_r()
    return F.build_candidate_frame(v, r, x if x is not None else _empty_x(), avail)


# ---------------------------------------------------------------- 판본 선택
def test_base_date_boundary_inclusive_and_next_day_excluded():
    """B_t 당일 가용은 포함, 다음 날은 제외(§5.1)."""
    cv = _cands(
        v_rows=[
            {"rcept_no": "20250101000001", "v_ta": 1.0},  # B_t 당일 가용
            {"rcept_no": "20250101000002", "v_ta": 2.0},  # 다음 날 가용
        ],
        avail=_avail({"20250101000001": date(2025, 6, 30), "20250101000002": date(2025, 7, 1)}),
    )
    panel = F.build_panel(_inputs(cv))
    assert panel["rcept_no"].to_list() == ["20250101000001"]
    assert panel["ta"].to_list() == [1.0]
    latest = F.build_panel(_inputs(cv), as_of="latest")
    assert latest["rcept_no"].to_list() == ["20250101000002"]  # 기준일 없이 가장 늦은 판본


def test_latest_ties_broken_by_rcept_no_desc():
    cv = _cands(
        v_rows=[
            {"rcept_no": "20250101000001", "v_ta": 1.0},
            {"rcept_no": "20250101000009", "v_ta": 9.0},
        ],
        avail=_avail({"20250101000001": date(2025, 3, 3), "20250101000009": date(2025, 3, 3)}),
    )
    assert F.build_panel(_inputs(cv))["ta"].to_list() == [9.0]


def test_consolidated_first_when_cfs_has_total_assets():
    cv = _cands(
        v_rows=[
            {"fs_div": "CFS", "rcept_no": "20250101000001", "v_ta": 10.0},
            {"fs_div": "OFS", "rcept_no": "20250101000002", "v_ta": 20.0},
        ],
        avail=_avail({"20250101000001": date(2025, 3, 3), "20250101000002": date(2025, 3, 3)}),
    )
    p = F.build_panel(_inputs(cv))
    assert (p["fs_div"][0], p["ta"][0]) == ("CFS", 10.0)


def test_falls_back_to_ofs_when_cfs_lacks_total_assets():
    cv = _cands(
        v_rows=[
            {"fs_div": "CFS", "rcept_no": "20250101000001", "v_ni": 5.0},  # ta 없음
            {"fs_div": "OFS", "rcept_no": "20250101000002", "v_ta": 20.0, "v_ni": 7.0},
        ],
        avail=_avail({"20250101000001": date(2025, 3, 3), "20250101000002": date(2025, 3, 3)}),
    )
    p = F.build_panel(_inputs(cv))
    assert (p["fs_div"][0], p["ta"][0], p["ni"][0]) == ("OFS", 20.0, 7.0)  # 한 fs_div에서만


def test_cfs_without_ta_and_no_ofs_is_kept():
    """CI-fsdiv-nofallback: OFS 후보가 없으면 CFS를 그대로 쓴다."""
    cv = _cands(
        v_rows=[{"fs_div": "CFS", "rcept_no": "20250101000001", "v_ni": 5.0}],
        avail=_avail({"20250101000001": date(2025, 3, 3)}),
    )
    p = F.build_panel(_inputs(cv))
    assert (p["fs_div"][0], p["ni"][0], p["layer"][0]) == ("CFS", 5.0, "vintage")


def test_cfs_judged_within_scope_not_over_all_versions():
    """가장 최근 CFS 판본은 B_t 이전 범위 안에서 본다. 범위 밖 판본(B_t 뒤)의 ta는 안 쓴다."""
    cv = _cands(
        v_rows=[
            {"fs_div": "CFS", "rcept_no": "20250101000001", "v_ni": 1.0},  # 범위 안, ta 없음
            {"fs_div": "CFS", "rcept_no": "20250801000001", "v_ta": 99.0},  # 범위 밖
            {"fs_div": "OFS", "rcept_no": "20250101000002", "v_ta": 20.0},
        ],
        avail=_avail(
            {
                "20250101000001": date(2025, 3, 3),
                "20250801000001": date(2025, 8, 4),
                "20250101000002": date(2025, 3, 3),
            }
        ),
    )
    p = F.build_panel(_inputs(cv))
    assert (p["fs_div"][0], p["ta"][0]) == ("OFS", 20.0)
    pl_ = F.build_panel(_inputs(cv), as_of="latest")
    assert (pl_["fs_div"][0], pl_["ta"][0]) == ("CFS", 99.0)


def test_revision_only_key_missing_when_after_base_but_row_kept():
    """원본 없는 키: raw만 있고 유일한 판본이 B_t 뒤 접수된 정정본이면 값은 결측, 행은 남는다."""
    cv = _cands(
        r_rows=[{"rcept_no": "20250901000001", "r_ta": 5.0}],
        avail=_avail({"20250901000001": date(2025, 9, 2)}),
    )
    cv = cv.with_columns(pl.lit(True).alias("is_revision"))
    inp = _inputs(cv)
    p = F.build_panel(inp, diag=True)
    assert p.height == 1
    assert p["layer"][0] == "none" and p["ta"][0] is None and p["rcept_no"][0] is None
    assert p["in_universe"][0] is True  # 판본 후보가 있어 행이 있다
    cov = F.coverage_from(inp, [2024]).filter(pl.col("scope") == "all")
    got = {r["item"]: r["n"] for r in cov.iter_rows(named=True)}
    assert got["no_origin_keys"] == 1 and got["no_origin_keys:revision_only"] == 1
    assert got["key_raw_only:revision_after_base"] == 1
    assert got["raw_only_corp_years"] == 1


def test_missing_receipt_is_not_available():
    cv = _cands(v_rows=[{"v_ta": 1.0}], avail=_avail({}))
    p = F.build_panel(_inputs(cv), diag=True)
    assert p["layer"][0] == "none" and p["k_n_no_receipt"][0] == 1


def test_select_versions_frame_shape():
    cv = _cands(v_rows=[{"v_ta": 1.0}], avail=_avail({"20250101000001": date(2025, 3, 3)}))
    inp = _inputs(cv)
    p = F.pick_versions(F.resolve_values(inp.cv), shift=0)
    assert p["fs_div"].to_list() == ["CFS"] and p["t"].to_list() == [2024]


# ---------------------------------------------------------------- 층 규칙·지급이자
def test_layer_fill_vintage_then_raw_then_xbrl():
    """vintage에 없는 지표는 같은 접수번호의 raw → XBRL로 채운다(CI-layer-fill)."""
    rc = "20250101000001"
    x = pl.DataFrame(
        {"rcept_no": [rc], "fy": [2024], "fs_div": ["CFS"], "x_oi": [3.0], "x_rev": [4.0]},
        schema_overrides={"fy": pl.Int32},
    )
    cv = _cands(
        v_rows=[{"rcept_no": rc, "v_ta": 1.0}],
        r_rows=[{"rcept_no": rc, "r_ta": 100.0, "r_oi": 2.0}],
        x=x,
        avail=_avail({rc: date(2025, 3, 3)}),
    )
    p = F.build_panel(_inputs(cv), src=True)
    assert p["ta"][0] == 1.0 and p["src_ta"][0] == "vintage"  # vintage가 먼저
    assert p["oi"][0] == 2.0 and p["src_oi"][0] == "raw"
    assert p["rev"][0] == 4.0 and p["src_rev"][0] == "xbrl"
    cvr = F.resolve_values(cv, layer_fill=False)
    assert cvr["oi"][0] is None and cvr["rev"][0] is None and cvr["ta"][0] == 1.0


def test_raw_only_candidate_reads_raw_then_xbrl_and_cap_from_raw_then_xbrl():
    rc = "20250101000001"
    x = pl.DataFrame(
        {"rcept_no": [rc], "fy": [2024], "fs_div": ["CFS"], "x_cap": [7.0], "x_te": [8.0]},
        schema_overrides={"fy": pl.Int32},
    )
    cv = _cands(r_rows=[{"rcept_no": rc, "r_ta": 100.0}], x=x, avail=_avail({rc: date(2025, 3, 3)}))
    p = F.build_panel(_inputs(cv), src=True)
    assert p["layer"][0] == "raw" and p["ta"][0] == 100.0
    assert p["te"][0] == 8.0 and p["src_te"][0] == "xbrl"
    assert p["cap"][0] == 7.0 and p["src_cap"][0] == "xbrl"
    cv2 = _cands(r_rows=[{"rcept_no": rc, "r_cap": 5.0}], x=x, avail=_avail({rc: date(2025, 3, 3)}))
    assert F.build_panel(_inputs(cv2), src=True)["src_cap"][0] == "raw"


def test_layers_raw_only_reads_raw_only_and_default_unchanged():
    """§5.1 raw 한 층 단독 민감도: raw 후보만, 값·전기·자본금·지급이자·통화를 raw에서만 읽는다."""
    rc1, rc2 = "20250101000001", "20250101000002"
    x = pl.DataFrame(
        {"rcept_no": [rc1, rc2], "fy": [2024, 2024], "fs_div": ["CFS", "CFS"],
         "x_rev": [4.0, 5.0], "x_cap": [7.0, 8.0], "xp1_ni": [11.0, 12.0]},
        schema_overrides={"fy": pl.Int32},
    )  # fmt: skip
    # rc1: vintage+raw 둘 다, rc2: vintage만(raw에 없음)
    cv = _cands(
        v_rows=[
            {"rcept_no": rc1, "v_ta": 1.0, "v_oi": 9.0, "v_currency": "USD"},
            {"rcept_no": rc2, "v_ta": 2.0},
        ],
        r_rows=[{"rcept_no": rc1, "r_ta": 100.0, "r_currency": "KRW", "rp1_ni": 1.5}],
        x=x,
        avail=_avail({rc1: date(2025, 3, 3), rc2: date(2025, 3, 4)}),
    )
    base = F.build_panel(_inputs(cv))
    assert base["rcept_no"].to_list() == [rc2] and base["layer"].to_list() == ["vintage"]
    raw = F.build_panel(_inputs(cv), layers="raw_only", src=True)
    assert raw["rcept_no"].to_list() == [rc1] and raw["layer"].to_list() == ["raw"]  # 후보도 raw만
    assert raw["ta"][0] == 100.0 and raw["src_ta"][0] == "raw"
    assert raw["oi"][0] is None  # vintage 값은 안 읽는다
    assert raw["rev"][0] is None and raw["cap"][0] is None  # XBRL도 안 읽는다
    assert raw["currency"][0] == "KRW"
    assert raw["ni_p1"][0] == 1.5 and raw["prior_src"][0] == "raw"
    with pytest.raises(ValueError):
        F.build_panel(_inputs(cv), layers="x")


def test_cap_xbrl_off_and_literal_flags():
    rc = "20250101000001"
    x = pl.DataFrame(
        {"rcept_no": [rc], "fy": [2024], "fs_div": ["CFS"], "x_cap": [7.0], "x_rev": [4.0]},
        schema_overrides={"fy": pl.Int32},
    )
    cv = _cands(r_rows=[{"rcept_no": rc, "r_ta": 100.0}], x=x, avail=_avail({rc: date(2025, 3, 3)}))
    assert F.build_panel(_inputs(cv))["cap"][0] == 7.0  # 기본: XBRL 대체
    off = F.build_panel(_inputs(cv), cap_xbrl=False)
    assert off["cap"][0] is None and off["rev"][0] == 4.0  # 자본금만 끈다


def test_ip_order_op_fin_inv_and_absolute_value():
    rc = "20250101000001"
    av = _avail({rc: date(2025, 3, 3)})
    # 영업활동이 있으면 그것(음수 → 절대값)
    cv = _cands(v_rows=[{"rcept_no": rc, "v_ip_op": -10.0}], avail=av)
    p = F.build_panel(_inputs(cv), src=True)
    assert (p["ip"][0], p["ip_src"][0], p["ip_signed"][0]) == (10.0, "op", -10.0)
    # 영업이 없으면 재무, 재무도 없으면 투자
    cv = _cands(
        v_rows=[{"rcept_no": rc, "v_ta": 1.0}],
        r_rows=[{"rcept_no": rc, "r_ip_fin": -3.0, "r_ip_inv": 9.0}],
        avail=av,
    )
    p = F.build_panel(_inputs(cv))
    assert (p["ip"][0], p["ip_src"][0]) == (3.0, "fin")
    cv = _cands(
        v_rows=[{"rcept_no": rc, "v_ta": 1.0}],
        r_rows=[{"rcept_no": rc, "r_ip_inv": -9.0}],
        avail=av,
    )
    p = F.build_panel(_inputs(cv))
    assert (p["ip"][0], p["ip_src"][0]) == (9.0, "inv")
    # 영업 0은 값이다(재무로 넘어가지 않는다)
    cv = _cands(
        v_rows=[{"rcept_no": rc, "v_ip_op": 0.0}],
        r_rows=[{"rcept_no": rc, "r_ip_fin": 5.0}],
        avail=av,
    )
    p = F.build_panel(_inputs(cv))
    assert (p["ip"][0], p["ip_src"][0]) == (0.0, "op")
    # 셋 다 없으면 결측
    cv = _cands(v_rows=[{"rcept_no": rc, "v_ta": 1.0}], avail=av)
    p = F.build_panel(_inputs(cv))
    assert p["ip"][0] is None and p["ip_src"][0] == "none"


# ---------------------------------------------------------------- 전기·전전기
def test_prior_values_raw_when_raw_has_rcept_else_xbrl_else_none():
    av = _avail(
        {
            "20250101000001": date(2025, 3, 3),
            "20250101000002": date(2025, 3, 3),
            "20250101000003": date(2025, 3, 3),
        }
    )

    def xp(rc):
        return pl.DataFrame(
            {"rcept_no": [rc], "fy": [2024], "fs_div": ["CFS"], "xp1_ni": [11.0], "xp2_ni": [12.0]},
            schema_overrides={"fy": pl.Int32},
        )

    # raw에 접수번호가 있다 → raw 비교 칸 (XBRL 값이 달라도 raw)
    cv = _cands(
        v_rows=[{"rcept_no": "20250101000001", "v_ta": 1.0}],
        r_rows=[{"rcept_no": "20250101000001", "rp1_ni": 1.5, "rp2_ni": 2.5}],
        x=xp("20250101000001"),
        avail=av,
    )
    p = F.build_panel(_inputs(cv))
    assert (p["ni_p1"][0], p["ni_p2"][0], p["prior_src"][0]) == (1.5, 2.5, "raw")
    # raw에 없다 → XBRL P·BP
    cv = _cands(
        v_rows=[{"rcept_no": "20250101000002", "v_ta": 1.0}], x=xp("20250101000002"), avail=av
    )
    p = F.build_panel(_inputs(cv))
    assert (p["ni_p1"][0], p["ni_p2"][0], p["prior_src"][0]) == (11.0, 12.0, "xbrl")
    # CI-a 끄기: 문면대로 결측
    p = F.build_panel(_inputs(cv), use_xbrl_prior=False)
    assert p["ni_p1"][0] is None and p["prior_src"][0] == "none"
    # 둘 다 없다
    cv = _cands(v_rows=[{"rcept_no": "20250101000003", "v_ta": 1.0}], avail=av)
    assert F.build_panel(_inputs(cv))["prior_src"][0] == "none"


def test_ltb_missing_stays_missing():
    cv = _cands(v_rows=[{"v_ta": 1.0}], avail=_avail({"20250101000001": date(2025, 3, 3)}))
    p = F.build_panel(_inputs(cv))
    assert p["ltb"][0] is None and p["ltb_p1"][0] is None


# ---------------------------------------------------------------- 기록용 (a)
def test_prior_a_uses_t_minus_1_and_2_versions_cut_at_b_t_and_same_fs_div():
    """t−1·t−2 보고서의 판본을 B_t로 잘라 읽고, fs_div가 t와 다르면 그 신호는 결측."""
    rows = [
        # FY2024 (t): CFS
        {"fy": 2024, "fs_div": "CFS", "rcept_no": "20250101000001", "v_ta": 30.0, "v_ni": 3.0},
        # FY2023: 원본(2024-03)과 B_t(2025-06-30) 이전 정정(2025-05) — 정정이 B_2024 이전이라 정정값
        {"fy": 2023, "fs_div": "CFS", "rcept_no": "20240101000001", "v_ta": 20.0, "v_ni": 2.0},
        {"fy": 2023, "fs_div": "CFS", "rcept_no": "20250201000002", "v_ta": 21.0, "v_ni": 2.1},
        # 2025-07 정정은 B_2024 뒤라 안 쓴다
        {"fy": 2023, "fs_div": "CFS", "rcept_no": "20250801000003", "v_ta": 22.0, "v_ni": 2.2},
        # FY2022: OFS만 있다 → t(CFS)와 fs_div가 달라 p2는 결측
        {"fy": 2022, "fs_div": "OFS", "rcept_no": "20230101000001", "v_ta": 10.0, "v_ni": 1.0},
    ]
    av = _avail(
        {
            "20250101000001": date(2025, 3, 20),
            "20240101000001": date(2024, 3, 20),
            "20250201000002": date(2025, 5, 2),
            "20250801000003": date(2025, 8, 4),
            "20230101000001": date(2023, 3, 20),
        }
    )
    cv = _cands(v_rows=rows, avail=av)
    inp = _inputs(cv, fys=(2024,))
    a = F.build_panel(inp, prior_source="t_minus_1_report")
    row = a.filter(pl.col("fy") == 2024)
    assert row["ta_p1"][0] == 21.0 and row["ni_p1"][0] == 2.1
    assert row["ta_p2"][0] is None and row["ni_p2"][0] is None  # fs_div 불일치
    assert row["prior_src"][0] == "t_minus_1_report"
    assert row["ocf_p1"][0] is None  # 그 열은 값이 없다
    # 주 방식은 비교 칸(raw/xbrl 없음 → 결측)이라 다르다
    assert F.build_panel(inp)["ta_p1"][0] is None


# ---------------------------------------------------------------- 층 불일치·coverage
def test_layer_mismatch_counts_pairs_and_tolerance():
    cv = _cands(
        v_rows=[
            {"rcept_no": "20250101000001", "v_ta": 100.0, "v_ni": 5.0},
            {"rcept_no": "20250101000002", "fs_div": "OFS", "v_ta": 100.0},
        ],
        r_rows=[
            {
                "rcept_no": "20250101000001",
                "r_ta": 100.3,
                "r_ni": 6.0,
            },  # ta 허용 오차 안, ni 불일치
            {"rcept_no": "20250101000002", "fs_div": "OFS", "r_ta": 200.0},
        ],
        avail=_avail({"20250101000001": date(2025, 3, 3), "20250101000002": date(2025, 3, 3)}),
    )
    mm = F.layer_mismatch(cv, [2024])
    d = {r["metric"]: (r["pairs"], r["mismatch"]) for r in mm.iter_rows(named=True)}
    assert d["ta"] == (2, 1) and d["ni"] == (1, 1)


def test_coverage_counts_only_presence_not_outcomes():
    cv = _cands(
        v_rows=[{"v_ta": 1.0, "v_te": -5.0, "v_ni": -3.0}],
        avail=_avail({"20250101000001": date(2025, 3, 3)}),
    )
    inp = _inputs(cv)
    c = F.coverage_from(inp, [2024])
    items = set(c["item"].to_list())
    assert "nn:te" in items and "nn:ni" in items
    # 결과 사건·조건부 수는 세지 않는다(§7.2)
    assert not any(("neg" in i and not i.startswith("ip_sign")) or "impair" in i for i in items)
    got = {
        (r["scope"], r["item"]): r["n"]
        for r in c.filter(pl.col("fy") == 2024).iter_rows(named=True)
    }
    assert got[("in_universe", "rows")] == 1 and got[("in_universe", "nn:te")] == 1
    assert got[("all", "layer:vintage")] == 1


def test_in_universe_requires_attribute_and_row():
    cv = _cands(v_rows=[{"v_ta": 1.0}], avail=_avail({"20250101000001": date(2025, 3, 3)}))
    p = F.build_panel(_inputs(cv, uni=_universe(CORP, in_attr=False)))
    assert p["in_universe"].to_list() == [False]
    assert F.OUT_PANEL_COLS == [c for c in p.columns if c in F.OUT_PANEL_COLS]


def test_panel_columns_cover_company_score_contract():
    from modeler.scores.quality.company_score import REQUIRED_COLS

    fs_cols = set(F.OUT_PANEL_COLS)
    in_w2a = {c for c in REQUIRED_COLS if not c.startswith(("dps", "shares", "retire", "capevt"))}
    assert in_w2a <= fs_cols


def test_parse_fys():
    assert F._parse_fys("2015-2017,2020") == [2015, 2016, 2017, 2020]


# ---------------------------------------------------------------- 분모 (합성 parquet)
def _write(lake: Lake, table: str, df: pl.DataFrame) -> None:
    d = lake.raw_dir(table)
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part.parquet")


def test_universe_rules_with_synthetic_lake(tmp_path: Path):
    lake = Lake(root=tmp_path)
    sm = pl.DataFrame(
        {
            "ticker": ["000010", "000020", "000030", "000040", "000050", "000060", "000060"],
            "market": ["KOSPI", "KOSDAQ", "KOSDAQ", "KOSPI", "KOSDAQ", "KOSPI", "KOSDAQ"],
            "name": ["일반", "한국제1호스팩", "은행", "삼월결산", "미매핑", "이전", "000060"],
            "status": ["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "DELISTED"],
            "last_seen_date": [date(2026, 9, 28)] * 7,
        }
    )
    sm = pl.concat(
        [
            sm,
            pl.DataFrame(
                {
                    "ticker": ["000070"],
                    "market": ["KONEX"],
                    "name": ["코넥스"],
                    "status": ["ACTIVE"],
                    "last_seen_date": [date(2026, 9, 28)],
                }
            ),
        ]
    )
    cm = pl.DataFrame(
        {
            "corp_code": ["c10", "c20", "c30", "c40", "c60", "c70"],
            "ticker": ["000010", "000020", "000030", "000040", "000060", "000070"],
            "induty_code": ["26110", "64992", "64121", "26110", "26110", "26110"],
            "acc_mt": ["12", "12", "12", "03", "12", "12"],
        }
    )
    _write(lake, "stock_master", sm)
    _write(lake, "dart_corp_master", cm)
    u = F.universe(lake).sort("stock_code")
    assert u["stock_code"].to_list() == [
        f"{i:06d}" for i in (10, 20, 30, 40, 50, 60)
    ]  # 코넥스 제외
    rec = {r["stock_code"]: r for r in u.iter_rows(named=True)}
    assert rec["000010"]["in_universe_attr"] is True
    assert rec["000020"]["is_spac"] is True and rec["000020"]["in_universe_attr"] is False
    assert rec["000030"]["is_financial"] is True and rec["000030"]["in_universe_attr"] is False
    assert rec["000040"]["acc_mt"] == 3 and rec["000040"]["in_universe_attr"] is False
    assert rec["000050"]["corp_code"] is None  # 매핑 안 됨
    assert rec["000060"]["market"] == "KOSPI"  # 두 시장 중 ACTIVE 행
    assert u["stock_code"].n_unique() == u.height


# ---------------------------------------------------------------- 실데이터 (STOCK_DATA_ROOT 필요)
ROOT = os.environ.get("STOCK_DATA_ROOT")
_have_lake = False
if ROOT:
    _lk = Lake(root=Path(ROOT))
    _have_lake = (
        _lk.raw_dir("dart_financial_statement_raw").is_dir()
        and _lk.derived_dir("stock_metric_vintage_fact").is_dir()
    )
real = pytest.mark.skipif(not _have_lake, reason="STOCK_DATA_ROOT 레이크가 없습니다")


@pytest.fixture(scope="module")
def real_inputs() -> F.FsInputs:
    return F.load_inputs(Lake(root=Path(ROOT)), range(2017, 2026))


@real
def test_002210_fy2024_net_income_base_vs_latest(real_inputs):
    """07 §3: 기준일 2025-06-30 → 최초값, 판본 제한 없이 읽으면 정정값. fs_div는 OFS."""
    corp = "00116268"
    base = F.build_panel(real_inputs, [2024]).filter(pl.col("corp_code") == corp)
    assert base["fs_div"][0] == "OFS" and base["layer"][0] == "vintage"
    assert base["rcept_no"][0] == "20250321001375"
    assert base["ni"][0] == pytest.approx(-7_253_006_357)
    lat = F.build_panel(real_inputs, [2024], as_of="latest").filter(pl.col("corp_code") == corp)
    assert lat["rcept_no"][0] == "20260506000539"
    assert lat["ni"][0] == pytest.approx(-14_571_588_545)
    fl = F.fs_latest(None, [2024], inputs=real_inputs).filter(pl.col("corp_code") == corp)  # type: ignore[arg-type]
    assert fl["ni"][0] == pytest.approx(-14_571_588_545) and fl["fs_div"][0] == "OFS"
    sv = F.select_versions(None, [2024], "base", inputs=real_inputs)  # type: ignore[arg-type]
    assert sv.filter(pl.col("corp_code") == corp)["rcept_no"][0] == "20250321001375"


@real
def test_panel_values_equal_vintage_for_selected_rcept(real_inputs):
    lake = Lake(root=Path(ROOT))
    p = F.build_panel(real_inputs, range(2019, 2025))
    p = p.filter(pl.col("layer") == "vintage").sample(300, seed=1)
    con = duckdb.connect()
    con.register(
        "pp", p.select("corp_code", "fy", "fs_div", "rcept_no", "ni", "ta", "te").to_arrow()
    )
    g = lake.derived_glob("stock_metric_vintage_fact")
    bad = con.execute(f"""
        select count(*) from pp join read_parquet('{g}', hive_partitioning=true) v
          on v.rcept_no = pp.rcept_no and v.fs_basis = pp.fs_div and v.corp_code = pp.corp_code
         and v.reprt_code='11011' and v.period_type='annual'
        where (v.metric_code='net_income' and abs(cast(v.value_numeric as double) - pp.ni) > 0.5)
           or (v.metric_code='total_assets' and abs(cast(v.value_numeric as double) - pp.ta) > 0.5)
           or (v.metric_code='total_equity' and abs(cast(v.value_numeric as double) - pp.te) > 0.5)
        """).fetchone()[0]
    assert bad == 0


@real
def test_prior_values_equal_raw_frmtrm(real_inputs):
    lake = Lake(root=Path(ROOT))
    p = F.build_panel(real_inputs, range(2019, 2025))
    p = p.filter((pl.col("prior_src") == "raw") & pl.col("ni_p1").is_not_null()).sample(300, seed=2)
    con = duckdb.connect()
    con.register("pp", p.select("fs_div", "rcept_no", "ni_p1", "ni_p2", "te_p1").to_arrow())
    g = lake.raw_glob("dart_financial_statement_raw")
    bad = con.execute(f"""
        select count(*) from pp join read_parquet('{g}', hive_partitioning=true) r
          on r.rcept_no = pp.rcept_no and r.fs_div = pp.fs_div
        where r.reprt_code = 11011 and r.sj_div in ('IS','CIS')
          and lower(regexp_replace(r.account_id,'^ifrs[-_](full_)?','ifrs-full_','i'))
              = 'ifrs-full_profitloss'
          and (abs(cast(r.frmtrm_amount as double) - pp.ni_p1) > 0.5
               or abs(cast(r.bfefrmtrm_amount as double) - pp.ni_p2) > 0.5)
        """).fetchone()[0]
    assert bad == 0


@real
def test_other_company_vintage_equal_and_panel_one_row_per_corp_fy(real_inputs):
    p = F.build_panel(real_inputs, range(2017, 2026))
    assert p.select(["corp_code", "fy"]).is_duplicated().sum() == 0
    assert set(F.OUT_PANEL_COLS) == set(p.columns)
    # 삼성전자(00126380) FY2023 연결 순이익은 사업보고서 접수번호 하나에서 vintage와 같다
    s = p.filter((pl.col("corp_code") == "00126380") & (pl.col("fy") == 2023))
    assert s.height == 1 and s["fs_div"][0] == "CFS" and s["layer"][0] == "vintage"


@real
def test_universe_report_real_lake_excludes_konex_and_others():
    rep = F.universe_report(Lake(root=Path(ROOT)))
    assert rep["denominator_corps_with_cls_N"] == 0  # 코넥스 법인은 분모에 없다
    assert rep["spac_tickers"] > 0
    assert rep["mapped_to_corp_code"] + rep["unmapped_to_corp_code"] == (
        rep["stock_master_tickers_kospi_kosdaq"]
    )
