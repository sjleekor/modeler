from datetime import date, datetime

import pandas as pd
import pytest

from modeler.scores.earnings_a.reasons import build_reason_dictionary, summarize_reasons

RUN = datetime(2026, 11, 3, 9, 0, 0)
D = date(2026, 3, 31)


def pres(ticker="000010", pe=date(2024, 3, 31), **kw):
    row = {
        "decision_date": D,
        "ticker": ticker,
        "corp_code": "C" + ticker,
        "basis": "CFS",
        "k": 1,
        "target_pe": pe,
        "row_present": False,
        "available_from": None,
        "rcept_no": None,
        "null_reason": None,
        "is_corr_only": False,
        "other_basis_present": False,
    }
    row.update(kw)
    return row


def rcpt(corp, no, nm):
    return {"rcept_no": no, "corp_code": corp, "report_nm": nm}


def sm(ticker, status="ACTIVE", seen=date(2026, 10, 22)):
    return {"ticker": ticker, "status": status, "name": "x", "last_seen_date": seen}


def lt(ticker, d):
    return {"ticker": ticker, "last_trade_date": d}


def run(rows, receipts=(), masters=(), trades=(), **kw):
    r = pd.DataFrame(list(receipts), columns=["rcept_no", "corp_code", "report_nm"])
    m = pd.DataFrame(list(masters), columns=["ticker", "status", "name", "last_seen_date"])
    t = pd.DataFrame(list(trades), columns=["ticker", "last_trade_date"])
    return build_reason_dictionary(pd.DataFrame(rows), r, m, t, run_at=RUN, **kw)


PER = rcpt("C000010", "20240515000001", "분기보고서 (2024.03)")


def test_ok_rows_dropped():
    out = run([pres(row_present=True)])
    assert out.empty


def test_immature():
    out = run([pres(pe=date(2026, 9, 30))], masters=[sm("000010")])
    assert out.reason_code.tolist() == ["IMMATURE"]
    assert out.target_kind.tolist() == ["interim"]
    assert not out.is_complete.any()
    assert out.confirmed_at[0] == RUN.isoformat()
    assert out.source_snapshot[0] == "2026-10-23"


def test_immature_boundary_annual_vs_interim():
    # raw max 2026-10-22. interim pe 2026-09-07 +45 = 2026-10-22 -> not immature.
    # annual pe 2026-07-24 (fye 7) +90 = 2026-10-22 -> not immature; one day later is.
    out = run(
        [
            pres("000010", pe=date(2026, 9, 7)),
            pres("000011", pe=date(2026, 9, 8)),
            pres("000012", pe=date(2026, 7, 24), fye_month=7),
            pres("000013", pe=date(2026, 7, 25), fye_month=7),
            pres("000014", pe=date(2026, 7, 25), fye_month=12),
        ]
    ).set_index("ticker")
    assert out.loc["000010", "reason_code"] == "UNCONFIRMED"
    assert out.loc["000011", "reason_code"] == "IMMATURE"
    assert out.loc["000012", "reason_code"] == "UNCONFIRMED"
    assert out.loc["000013", "reason_code"] == "IMMATURE"
    assert out.loc["000012", "target_kind"] == "annual"
    assert out.loc["000014", "target_kind"] == "interim"
    assert out.loc["000014", "reason_code"] == "UNCONFIRMED"  # +45d passed


def test_src_missing_details():
    rows = [
        pres("000010", row_present=True, null_reason="NOT_REPORTED"),
        pres("000020", other_basis_present=True),
        pres("000030"),
    ]
    rc = [
        PER,
        rcpt("C000020", "20240515000002", "분기보고서 (2024.03)"),
        rcpt("C000030", "20240515000003", "분기보고서 (2024.03)"),
    ]
    ms = [sm("000010"), sm("000020"), sm("000030")]
    tr = [lt(t, date(2026, 10, 1)) for t in ("000010", "000020", "000030")]
    out = run(rows, rc, ms, tr).set_index("ticker")
    assert (out.reason_code == "SRC_MISSING").all()
    assert out.loc["000010", "reason_detail"] == "MART_NULL:NOT_REPORTED"
    assert out.loc["000020", "reason_detail"] == "BASIS_CHANGED"
    assert out.loc["000030", "reason_detail"] == "NO_ROW"
    assert out.loc["000030", "evidence_date"] == "2024-05-15"
    assert out.loc["000030", "reason_evidence"].startswith("20240515000003")


def test_corr_only():
    rows = [pres(row_present=True, is_corr_only=True, rcept_no="20240601000009")]
    rc = [rcpt("C000010", "20240601000009", "[기재정정]분기보고서 (2024.03)")]
    out = run(rows, rc, [sm("000010")], [lt("000010", date(2026, 10, 1))])
    assert out.reason_code.tolist() == ["ORIG_MISSING_CORR_ONLY"]
    assert out.reason_evidence[0] == "20240601000009"


def test_merged_and_own_company_filter():
    seen = date(2024, 4, 10)
    rows = [pres("000010"), pres("000020")]
    rc = [
        rcpt("C000010", "20240101000001", "회사합병결정(종속회사의주요경영사항)"),
        rcpt("C000010", "20240201000002", "주요사항보고서(회사합병결정)"),
        rcpt("C000020", "20240201000003", "회사합병결정(자회사의 주요경영사항)"),
    ]
    ms = [sm("000010", "DELISTED", seen), sm("000020", "DELISTED", seen)]
    tr = [lt("000010", seen), lt("000020", seen)]
    out = run(rows, rc, ms, tr).set_index("ticker")
    assert out.loc["000010", "reason_code"] == "MERGED"
    assert out.loc["000010", "reason_evidence"] == "20240201000002"
    assert out.loc["000010", "evidence_date"] == "2024-02-01"
    assert out.loc["000020", "reason_code"] == "DELISTED"


def test_merger_outside_window_not_merged():
    seen = date(2024, 4, 10)
    rc = [rcpt("C000010", "20230101000001", "회사합병결정")]
    out = run([pres()], rc, [sm("000010", "DELISTED", seen)], [lt("000010", seen)])
    assert out.reason_code.tolist() == ["DELISTED"]


def test_delisted_and_active_row_blocks():
    seen = date(2024, 4, 10)
    ms = [sm("000010", "DELISTED", seen), sm("000010", "ACTIVE", date(2026, 10, 22))]
    out = run([pres()], [], ms, [lt("000010", seen)])
    assert out.reason_code.tolist() == ["UNCONFIRMED"]


def test_delisted_keeps_mart_null_detail():
    seen = date(2024, 4, 1)
    out = run(
        [pres(row_present=True, null_reason="X")],
        [],
        [sm("000010", "DELISTED", seen)],
        [lt("000010", seen)],
    )
    assert out.reason_code.tolist() == ["DELISTED"]
    assert out.reason_detail.tolist() == ["MART_NULL:X"]


def test_report_stopped():
    out = run([pres()], [], [sm("000010")], [lt("000010", date(2026, 10, 1))])
    assert out.reason_code.tolist() == ["REPORT_STOPPED"]


def test_unconfirmed():
    out = run([pres()], [], [sm("000010")], [lt("000010", date(2024, 5, 1))])
    assert out.reason_code.tolist() == ["UNCONFIRMED"]
    assert out.reason_detail.isna().all()


def test_first_match_src_missing_beats_delisted():
    # delisted late (last_seen after deadline), traded after, receipt exists.
    seen = date(2025, 6, 1)
    out = run(
        [pres()],
        [PER],
        [sm("000010", "DELISTED", seen)],
        [lt("000010", seen)],
    )
    assert out.reason_code.tolist() == ["SRC_MISSING"]


def test_value_columns_rejected():
    for c in ("value_dec", "y", "y_clip", "oi_sum_dec", "standalone_value", "ttm_value"):
        df = pd.DataFrame([pres()])
        df[c] = "1"
        with pytest.raises(ValueError):
            build_reason_dictionary(
                df,
                pd.DataFrame(columns=["rcept_no", "corp_code", "report_nm"]),
                pd.DataFrame(columns=["ticker", "status", "name", "last_seen_date"]),
                pd.DataFrame(columns=["ticker", "last_trade_date"]),
                run_at=RUN,
            )


def test_summarize_counts():
    out = run(
        [pres("000010"), pres("000011"), pres("000012", pe=date(2026, 9, 30))],
        [],
        [sm("000010"), sm("000011"), sm("000012")],
    )
    s = summarize_reasons(out)
    got = {r.reason_code: r.n for r in s.itertuples()}
    assert got == {"UNCONFIRMED": 2, "IMMATURE": 1}
    assert s["n"].dtype.kind == "i"
