"""Synthetic-only tests for earnings A labels. The real r2 mart is never touched."""

from dataclasses import replace
from datetime import date
from decimal import Decimal

import duckdb
import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import labels, seal
from modeler.scores.earnings_a.paths import DataPaths
from modeler.scores.earnings_a.seal import SealViolation

T = date(2021, 9, 30)


def _row(ticker, metric, basis, pe, val, avail, rcept, reason=None, seq=1):
    return {
        "ticker": ticker,
        "corp_code": "C" + ticker,
        "metric_code": metric,
        "fs_basis": basis,
        "statement_period_end": pe,
        "rcept_no": rcept,
        "available_from": avail,
        "standalone_value": val,
        "standalone_null_reason": reason,
        "seq_key": seq,
    }


def _write(tmp_path, rows):
    df = pd.DataFrame(rows)
    df["statement_period_end"] = pd.to_datetime(df["statement_period_end"]).dt.date
    df["available_from"] = pd.to_datetime(df["available_from"]).dt.date
    con = duckdb.connect(":memory:")
    con.register("df", df)
    out = tmp_path / "r2.parquet"
    con.execute(
        "COPY (SELECT * REPLACE (CAST(standalone_value AS DECIMAL(38,4)) AS standalone_value)"
        f" FROM df) TO '{out}' (FORMAT PARQUET)"
    )
    con.close()
    paths = DataPaths.resolve(env={"STOCK_DATA_ROOT": str(tmp_path)})
    return replace(paths, r2_parquet=out)


def _anchor(tickers=("A",), basis="CFS", at="1000.0000", dates=(T,), eligible=None):
    rows = []
    for d in dates:
        for i, tk in enumerate(tickers):
            r = {
                "decision_date": d,
                "ticker": tk,
                "corp_code": "C" + tk,
                "basis": basis,
                "a_t_dec": at,
            }
            if eligible is not None:
                r["input_eligible"] = eligible[i]
            rows.append(r)
    return pd.DataFrame(rows)


RECEIPTS = pd.DataFrame({"rcept_no": ["R1", "R2", "R3", "R4"], "report_nm": ["분기보고서"] * 4})
PES = ["2021-12-31", "2022-03-31", "2022-06-30", "2022-09-30"]


def _four(ticker="A", basis="CFS", vals=("1.5", "2.5", "3.5", "4.5"), pes=PES, **kw):
    return [
        _row(ticker, "operating_income", basis, pe, v, f"2022-0{i + 1}-15", f"R{i + 1}", **kw)
        for i, (pe, v) in enumerate(zip(pes, vals, strict=True))
    ]


def _run(tmp_path, rows, anchor=None, receipts=RECEIPTS, target="oi_y1", dates=(T,)):
    paths = _write(tmp_path, rows)
    g = seal.grant_dev()
    r2 = labels.load_label_rows(
        paths, decision_dates=list(dates), grant=g, horizon=labels.TARGETS[target][1]
    )
    anchor = _anchor(dates=dates) if anchor is None else anchor
    return labels.build_labels(
        r2, anchor, receipts, decision_dates=list(dates), grant=g, target=target
    )


def test_basic_sum_and_pub_date(tmp_path):
    out = _run(tmp_path, _four(vals=("0.1", "0.2", "0.3", "0.4")))
    r = out.iloc[0]
    assert r.label_ok and r.oi_sum_dec == "1.0000"
    assert r.y == pytest.approx(0.001)
    assert r.label_pub_date == date(2022, 4, 15)  # max of the four available_from
    assert r.label_last_quarter_end == date(2022, 9, 30)
    assert (r.rcept_no_1, r.rcept_no_4) == ("R1", "R4")
    assert r.missing_k is None and r.missing_kind is None


def test_decimal_exact(tmp_path):
    vals = ("0.1", "0.2", "0.0001", "0.0002")
    out = _run(tmp_path, _four(vals=vals))
    assert out.iloc[0].oi_sum_dec == "0.3003"
    assert Decimal(out.iloc[0].oi_sum_dec) == Decimal("0.3003")


def test_month_index_match_with_0330_and_seq_key_ignored(tmp_path):
    pes = ["2021-12-31", "2022-03-30", "2022-06-30", "2022-09-30"]
    rows = _four(pes=pes)
    for i, r in enumerate(rows):
        r["seq_key"] = 100 - i  # disagrees with the calendar
    out = _run(tmp_path, rows)
    assert out.iloc[0].label_ok and out.iloc[0].oi_sum_dec == "12.0000"


def test_fiscal_march_firm(tmp_path):
    # t = 2021-03-31: targets Jun/Sep/Dec/Mar; fiscal-March firm reports those quarter ends
    pes = ["2021-06-30", "2021-09-30", "2021-12-31", "2022-03-31"]
    out = _run(
        tmp_path,
        _four(pes=pes),
        anchor=_anchor(dates=(date(2021, 3, 31),)),
        dates=(date(2021, 3, 31),),
    )
    assert out.iloc[0].label_ok
    assert out.iloc[0].label_last_quarter_end == date(2022, 3, 31)


def test_missing_row_no_row(tmp_path):
    rows = _four()
    del rows[2]
    out = _run(tmp_path, rows).iloc[0]
    assert not out.label_ok and out.missing_k == 3 and out.missing_kind == "NO_ROW"
    assert out.oi_sum_dec is None and np.isnan(out.y)
    assert out.label_pub_date is None


def test_corr_only(tmp_path):
    rc = pd.DataFrame(
        {"rcept_no": ["R1", "R2", "R3", "R4"], "report_nm": ["a", "[기재정정]분기보고서", "a", "a"]}
    )
    out = _run(tmp_path, _four(), receipts=rc).iloc[0]
    assert not out.label_ok and out.missing_kind == "CORR_ONLY" and out.missing_k == 2


def test_null_reason(tmp_path):
    rows = _four()
    rows[1]["standalone_value"] = None
    rows[1]["standalone_null_reason"] = "NO_CUMULATIVE"
    out = _run(tmp_path, rows).iloc[0]
    assert not out.label_ok and out.missing_kind == "MART_NULL:NO_CUMULATIVE"
    rows = _four()
    rows[3]["standalone_null_reason"] = "CONFLICT"  # value present but reason set
    out = _run(tmp_path, rows).iloc[0]
    assert not out.label_ok and out.missing_k == 4


def test_basis_changed(tmp_path):
    rows = _four()[:2] + _four(basis="OFS")[2:]
    out = _run(tmp_path, rows).iloc[0]
    assert not out.label_ok and out.missing_k == 3 and out.missing_kind == "BASIS_CHANGED"
    paths = _write(tmp_path, rows)
    g = seal.grant_dev()
    r2 = labels.load_label_rows(paths, decision_dates=[T], grant=g)
    lq = labels.build_label_quarters(r2, _anchor(), RECEIPTS, decision_dates=[T], grant=g)
    assert lq["other_basis_present"].tolist() == [False, False, True, True]
    assert lq["row_present"].tolist() == [True, True, False, False]
    assert "value_dec" not in labels.presence_view(lq).columns


def test_target_months_and_other_metrics_ignored(tmp_path):
    rows = _four() + [_row("A", "total_assets", "CFS", pe, "9", "2022-01-01", "RX") for pe in PES]
    paths = _write(tmp_path, rows)
    g = seal.grant_dev()
    r2 = labels.load_label_rows(paths, decision_dates=[T], grant=g, metrics=("operating_income",))
    lq = labels.build_label_quarters(r2, _anchor(), RECEIPTS, decision_dates=[T], grant=g)
    assert lq["target_pe"].tolist() == [
        date(2021, 12, 31),
        date(2022, 3, 31),
        date(2022, 6, 30),
        date(2022, 9, 30),
    ]
    assert lq["value_dec"].tolist() == ["1.5000", "2.5000", "3.5000", "4.5000"]


def test_load_window_caps_rows(tmp_path):
    far = _row("A", "operating_income", "CFS", "2024-03-31", "1", "2024-05-01", "RF")
    early = _row("A", "operating_income", "CFS", "2021-06-30", "1", "2021-08-01", "RE")
    paths = _write(tmp_path, _four() + [far, early])
    r2 = labels.load_label_rows(paths, decision_dates=[T], grant=seal.grant_dev())
    assert set(r2["rcept_no"]) == {"R1", "R2", "R3", "R4"}


def test_y_clip_only_eligible_ok(tmp_path):
    tickers = [f"T{i:03d}" for i in range(101)]
    rows = []
    for i, tk in enumerate(tickers):
        # sum = i, A_t = 1000 -> y = i / 1000 ; T100 is huge but ineligible
        v = "5000" if i == 100 else str(i)
        rows += [
            _row(tk, "operating_income", "CFS", pe, v if j == 0 else "0", "2022-01-01", f"R{j + 1}")
            for j, pe in enumerate(PES)
        ]
    eligible = [i != 100 for i in range(101)]
    an = _anchor(tickers=tickers, eligible=eligible)
    out = _run(tmp_path, rows, anchor=an).set_index("ticker")
    pool = np.arange(100) / 1000.0
    lo, hi = np.nanquantile(pool, (0.01, 0.99), method="linear")
    assert out.loc["T099", "y_clip"] == pytest.approx(hi)
    assert out.loc["T000", "y_clip"] == pytest.approx(lo)
    assert out.loc["T100", "y"] == pytest.approx(5.0)
    assert out.loc["T100", "y_clip"] == pytest.approx(hi)  # clipped by eligible bounds
    assert out.loc["T050", "y_clip"] == pytest.approx(0.05)


def test_clip_helper_per_date_and_missing_eligible():
    y = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0, 6.0])
    d = pd.Series(["a", "a", "a", "a", "b", "b"])
    ok = pd.Series([True, True, True, False, True, True])
    el = pd.Series([True] * 6)
    out = labels._clip_y(y, d, el, ok)
    assert np.isnan(out.iloc[3])
    assert out.iloc[0] == pytest.approx(np.nanquantile([1, 2, 3], 0.01))
    assert out.iloc[4] == pytest.approx(np.nanquantile([5, 6], 0.01))


def test_no_input_eligible_column_means_all(tmp_path):
    out = _run(tmp_path, _four())
    assert out.iloc[0].y_clip == pytest.approx(out.iloc[0].y)


def test_dev_blocks_sealed_date_before_read(tmp_path, monkeypatch):
    paths = _write(tmp_path, _four())
    calls = []
    monkeypatch.setattr(labels.duckdb, "connect", lambda *a, **k: calls.append(1))
    with pytest.raises(SealViolation):
        labels.load_label_rows(paths, decision_dates=[date(2022, 9, 30)], grant=seal.grant_dev())
    assert calls == []


def test_dev_horizon2_guard(tmp_path):
    paths = _write(tmp_path, _four())
    g = seal.grant_dev()
    r2 = labels.load_label_rows(paths, decision_dates=[T], grant=g, horizon=2)
    assert r2["statement_period_end"].max() <= date(2023, 6, 30)
    with pytest.raises(SealViolation, match="2023-09-30"):
        labels.build_labels(r2, _anchor(), RECEIPTS, decision_dates=[T], grant=g, target="oi_y2")
    # horizon 2 is fine for an earlier date: t=2021-06-30 -> last quarter 2023-06-30
    d2 = date(2021, 6, 30)
    out = labels.build_labels(
        r2, _anchor(dates=(d2,)), RECEIPTS, decision_dates=[d2], grant=g, target="oi_y2"
    )
    assert out.iloc[0].label_last_quarter_end == date(2023, 6, 30)


def test_requires_grant(tmp_path):
    paths = _write(tmp_path, _four())
    with pytest.raises(SealViolation):
        labels.load_label_rows(paths, decision_dates=[T])
    with pytest.raises(SealViolation):
        labels.build_labels(pd.DataFrame(), _anchor(), RECEIPTS, decision_dates=[T], target="oi_y1")


def test_digest_only_sealed_no_tag(tmp_path):
    d = seal.SEALED_DATES[0]  # 2023-06-30 -> quarters 2023-09 .. 2024-06
    pes = ["2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    paths = _write(tmp_path, _four(pes=pes))
    an = _anchor(dates=(d,))

    def builder(g):
        r2 = labels.load_label_rows(paths, decision_dates=[d], grant=g)
        return labels.build_labels(r2, an, RECEIPTS, decision_dates=[d], grant=g)

    out = seal.digest_only(builder, exact_cols=["decision_date", "ticker", "oi_sum_dec"])
    assert set(out) >= {"exact_sha256", "n_rows"}
    assert out["n_rows"] == 1
    assert "oi_sum_dec" not in str(out.get("n_rows_by"))
