"""F3 report on synthetic frames (no real data)."""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import checks as C
from modeler.scores.earnings_a import universe as U
from modeler.scores.earnings_a.spec import FEATURES

D_DEV = date(2016, 3, 31)
D_SEALED = date(2023, 6, 30)


def _urow(decision_date, ticker, **kw):
    row = {c: None for c in U.COLUMNS}
    row.update(
        decision_date=decision_date,
        session_date=decision_date,
        ticker=ticker,
        corp_code="c" + ticker,
        market="KOSPI",
        ksic2="26",
        induty_code="26100",
        is_halted=False,
        is_financial=False,
        is_holding=False,
        is_spac=False,
        liq_notice=False,
        is_non_krw=False,
        p6_tradable=False,
        only_preferred_listed=False,
        in_universe=False,
        exclusion_reason=None,
    )
    row.update(kw)
    return row


def _frow(decision_date, ticker, **kw):
    row = {"decision_date": pd.Timestamp(decision_date), "ticker": ticker}
    row.update({name: 0.1 for name in FEATURES})
    row.update(
        in_universe=False,
        a_t_status="ok",
        fye_on_grid=True,
        fye_month=12,
        basis="CFS",
        basis_switched=False,
        oi_ttm_raw=1.0,
        input_eligible=False,
        input_eligible_incl_holding=False,
        ttm_includes_conflict=False,
        p6_tradable=False,
    )
    row.update(kw)
    return row


def _dev_frames():
    """One DEV date with eight universe rows and four feature rows (hand-counted)."""
    u = pd.DataFrame(
        [
            _urow(D_DEV, "A", in_universe=True, p6_tradable=True),
            _urow(D_DEV, "B", market="KOSDAQ", is_halted=True, in_universe=True),
            _urow(D_DEV, "H", in_universe=True, p6_tradable=True),
            _urow(D_DEV, "C", exclusion_reason="only_preferred"),
            _urow(D_DEV, "D", is_financial=True, is_holding=True, exclusion_reason="financial"),
            _urow(D_DEV, "E", market="KOSDAQ", is_spac=True, exclusion_reason="spac"),
            _urow(D_DEV, "F", liq_notice=True, exclusion_reason="liq_notice", p6_tradable=True),
            _urow(
                D_DEV,
                "G",
                market="KOSDAQ",
                is_non_krw=True,
                p6_tradable=True,
                exclusion_reason="non_krw",
            ),
        ],
        columns=U.COLUMNS,
    )
    f = pd.DataFrame(
        [
            _frow(
                D_DEV,
                "A",
                in_universe=True,
                input_eligible=True,
                input_eligible_incl_holding=True,
                ttm_includes_conflict=True,
                p6_tradable=True,
            ),
            _frow(
                D_DEV,
                "B",
                in_universe=True,
                a_t_status="stale",
                fye_on_grid=False,
                fye_month=4,
                basis="OFS",
                basis_switched=True,
                oi_ttm_raw=np.nan,
                ttm_includes_conflict=None,
            ),
            _frow(
                D_DEV,
                "H",
                in_universe=True,
                a_t_status="nonpositive",
                fye_month=3,
                oi_ttm_raw=0.5,
            ),
            _frow(
                D_DEV,
                "D",
                in_universe=False,
                oi_ttm_raw=2.0,
                input_eligible_incl_holding=True,
            ),
        ]
    )
    return u, f


def _sealed_frames():
    """One SEALED date: two eligible rows with NaN in one feature each, one ineligible row full of NaN."""
    d = D_SEALED
    u = pd.DataFrame(
        [
            _urow(d, "X", in_universe=True, p6_tradable=True),
            _urow(d, "Y", in_universe=True),
            _urow(d, "Z", exclusion_reason="financial", is_financial=True),
        ],
        columns=U.COLUMNS,
    )
    x = _frow(d, "X", in_universe=True, input_eligible=True, oi_chg_4q=np.nan, stale_days=np.nan)
    y = _frow(d, "Y", in_universe=True, input_eligible=True)
    z = _frow(d, "Z", in_universe=False, a_t_status="absent", fye_on_grid=False, fye_month=pd.NA)
    z.update({name: np.nan for name in FEATURES})
    f = pd.DataFrame([x, y, z])
    return u, f


def test_columns_exact_and_order():
    u, f = _dev_frames()
    rep = C.f3_report(u, f)
    expected = [
        "decision_date",
        "window",
        "session_date",
        "n_companies",
        "n_kospi",
        "n_kosdaq",
        "n_halted",
        "n_excl_only_preferred",
        "n_excl_financial",
        "n_holding_64992",
        "n_excl_spac",
        "n_excl_liq_notice",
        "n_excl_non_krw",
        "n_spac_flag",
        "n_in_universe",
        "n_p6_tradable_in_universe",
        "n_a_t_ok",
        "n_a_t_stale",
        "n_a_t_nonpositive",
        "n_a_t_absent",
        "n_fye_offgrid",
        "n_fye_dec",
        "n_fye_other_grid",
        "n_basis_cfs",
        "n_basis_ofs",
        "n_basis_switched",
        "n_oi_ttm_present",
        "n_input_eligible",
        "n_input_eligible_incl_holding",
        "n_ttm_includes_conflict",
        "n_p6_tradable_eligible",
    ] + [f"miss_{name}" for name in FEATURES]
    assert list(rep.columns) == expected
    assert list(rep.columns) == C.F3_COLUMNS


def test_window_mapping_and_sorted_rows():
    dates = [
        date(2026, 12, 31),  # OTHER
        date(2025, 9, 30),  # CONFIRM
        date(2023, 6, 30),  # SEALED
        date(2022, 9, 30),  # GAP
        date(2022, 6, 30),  # DEV (last DEV quarter)
        date(2016, 3, 31),  # DEV (first)
    ]
    rows = [_urow(d, "A") for d in dates]
    u = pd.DataFrame(rows, columns=U.COLUMNS)
    f = pd.DataFrame(columns=list(_frow(D_DEV, "x").keys()))
    rep = C.f3_report(u, f)
    assert list(rep["decision_date"]) == sorted(d.isoformat() for d in dates)
    got = dict(zip(rep["decision_date"], rep["window"], strict=True))
    assert got == {
        "2016-03-31": "DEV",
        "2022-06-30": "DEV",
        "2022-09-30": "GAP",
        "2023-06-30": "SEALED",
        "2025-09-30": "CONFIRM",
        "2026-12-31": "OTHER",
    }
    assert C.window_of(date(2024, 12, 31)) == "SEALED"
    assert C.window_of(date(2023, 3, 31)) == "GAP"


def test_counts_on_hand_built_example():
    u, f = _dev_frames()
    rep = C.f3_report(u, f)
    assert len(rep) == 1
    r = rep.iloc[0]
    assert r["decision_date"] == "2016-03-31"
    assert r["window"] == "DEV"
    assert r["session_date"] == "2016-03-31"

    expected_universe = {
        "n_companies": 8,
        "n_kospi": 5,  # A, H, C, D, F
        "n_kosdaq": 3,  # B, E, G
        "n_halted": 1,
        "n_excl_only_preferred": 1,
        "n_excl_financial": 1,
        "n_holding_64992": 1,
        "n_excl_spac": 1,
        "n_excl_liq_notice": 1,
        "n_excl_non_krw": 1,
        "n_spac_flag": 1,
        "n_in_universe": 3,  # A, B, H
        "n_p6_tradable_in_universe": 2,  # A, H (B is not p6 tradable)
    }
    expected_features = {
        "n_a_t_ok": 1,  # A
        "n_a_t_stale": 1,  # B
        "n_a_t_nonpositive": 1,  # H
        "n_a_t_absent": 0,
        "n_fye_offgrid": 1,  # B: month 4 is off the grid
        "n_fye_dec": 1,  # A
        "n_fye_other_grid": 1,  # H: month 3
        "n_basis_cfs": 2,  # A, H
        "n_basis_ofs": 1,  # B
        "n_basis_switched": 1,  # B
        "n_oi_ttm_present": 2,  # A, H (B is NaN)
        "n_input_eligible": 1,  # A
        "n_input_eligible_incl_holding": 2,  # A and D (whole frame)
        "n_ttm_includes_conflict": 1,  # A
        "n_p6_tradable_eligible": 1,  # A
    }
    for key, value in {**expected_universe, **expected_features}.items():
        assert r[key] == value, key


def test_miss_shares_over_input_eligible_rows():
    u, f = _sealed_frames()
    rep = C.f3_report(u, f)
    r = rep.iloc[0]
    assert r["window"] == "SEALED"
    # Eligible rows are X and Y only. Z is not eligible, so its NaNs do not count.
    assert r["n_input_eligible"] == 2
    assert r["miss_oi_chg_4q"] == 0.5
    assert r["miss_stale_days"] == 0.5
    for name in FEATURES:
        if name not in ("oi_chg_4q", "stale_days"):
            assert r[f"miss_{name}"] == 0.0, name
    assert r["n_a_t_absent"] == 0  # Z is not in the universe, so it is not counted here
    assert r["n_a_t_ok"] == 2


def test_miss_share_is_nan_without_eligible_rows():
    u, f = _dev_frames()
    f = f.assign(input_eligible=False)
    rep = C.f3_report(u, f)
    assert np.isnan(rep.iloc[0]["miss_oi_ttm"])


def test_values_are_python_ints_and_strings():
    u, f = _dev_frames()
    rec = C.f3_report(u, f).to_dict(orient="records")[0]
    for k, v in rec.items():
        if k.startswith("n_"):
            assert type(v) is int or isinstance(v, int), k


def test_sign_table_counts_on_complete_rows_by_year():
    rows = pd.DataFrame(
        [
            {
                "metric_code": "capex_ppe",
                "statement_period_end": "2019-12-31",
                "ttm_complete": True,
                "ttm_value": -5.0,
            },
            {
                "metric_code": "capex_ppe",
                "statement_period_end": "2019-03-31",
                "ttm_complete": True,
                "ttm_value": 0.0,
            },
            {
                "metric_code": "capex_ppe",
                "statement_period_end": "2019-09-30",
                "ttm_complete": True,
                "ttm_value": np.nan,
            },
            {
                "metric_code": "capex_ppe",
                "statement_period_end": "2019-06-30",
                "ttm_complete": False,
                "ttm_value": -9.0,
            },
            {
                "metric_code": "dividends_paid",
                "statement_period_end": "2020-12-31",
                "ttm_complete": True,
                "ttm_value": 3.0,
            },
            {
                "metric_code": "operating_income",
                "statement_period_end": "2020-12-31",
                "ttm_complete": True,
                "ttm_value": -1.0,
            },
        ]
    )
    sg = C.sign_table(rows)
    assert list(sg.columns) == C.SIGN_COLUMNS
    got = {
        (r.metric_code, r.group_value): (r.n_lt0, r.n_eq0, r.n_gt0, r.n_nan)
        for r in sg.itertuples()
    }
    assert got == {
        ("capex_ppe", "2019"): (1, 1, 0, 1),
        ("dividends_paid", "2020"): (0, 0, 1, 0),
    }
    assert set(sg["group_by"]) == {"period_year"}


def test_sign_table_uses_bsns_year_when_present():
    rows = pd.DataFrame(
        [
            {
                "metric_code": "capex_intangible",
                "bsns_year": 2018,
                "statement_period_end": "2018-12-31",
                "ttm_complete": True,
                "ttm_value": -1.0,
            },
        ]
    )
    sg = C.sign_table(rows)
    assert list(sg["group_by"]) == ["bsns_year"]
    assert list(sg["group_value"]) == ["2018"]
    assert int(sg.iloc[0]["n_lt0"]) == 1


def test_write_f3_writes_three_files(tmp_path):
    u, f = _dev_frames()
    # No eligible rows, so every miss share is NaN. The JSON must write those as null.
    rep = C.f3_report(u, f.assign(input_eligible=False))
    sg = C.sign_table(
        pd.DataFrame(columns=["metric_code", "statement_period_end", "ttm_complete", "ttm_value"])
    )
    out = tmp_path / "f3_out"
    paths = C.write_f3(rep, sg, out)
    assert (out / "f3_report.tsv").is_file()
    assert (out / "f3_report.json").is_file()
    assert (out / "f3_signs.tsv").is_file()
    assert set(paths) == {"report_tsv", "report_json", "signs_tsv"}
    header = (out / "f3_report.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == C.F3_COLUMNS
    recs = json.loads((out / "f3_report.json").read_text(encoding="utf-8"))
    assert len(recs) == 1
    assert recs[0]["n_input_eligible"] == 0
    assert recs[0]["miss_oi_chg_4q"] is None  # NaN share is written as null


def test_write_once_refuses_existing_dir(tmp_path):
    u, f = _dev_frames()
    rep = C.f3_report(u, f)
    sg = C.sign_table(
        pd.DataFrame(columns=["metric_code", "statement_period_end", "ttm_complete", "ttm_value"])
    )
    out = tmp_path / "once"
    C.write_f3(rep, sg, out)
    with pytest.raises(FileExistsError):
        C.write_f3(rep, sg, out)


def test_refuses_inside_repo_with_git_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n", encoding="utf-8")
    u, f = _dev_frames()
    rep = C.f3_report(u, f)
    sg = C.sign_table(
        pd.DataFrame(columns=["metric_code", "statement_period_end", "ttm_complete", "ttm_value"])
    )
    out = repo / "nested" / "out"
    with pytest.raises(ValueError):
        C.write_f3(rep, sg, out)
    assert not out.exists()
    assert not (repo / "nested").exists()


def test_no_forbidden_names_in_columns():
    u, f = _dev_frames()
    rep = C.f3_report(u, f)
    sg = C.sign_table(
        pd.DataFrame(columns=["metric_code", "statement_period_end", "ttm_complete", "ttm_value"])
    )
    # 'y_' as a token prefix only: a plain substring check would hit n_excl_only_preferred.
    pattern = re.compile(r"(^|_)y_|label|oi_sum|mae")
    for col in list(rep.columns) + list(sg.columns):
        assert not pattern.search(col), col
