"""테마 카드 총보수 붙이기 시험. 합성 XML·표만 쓴다."""

import gzip
from datetime import date

import polars as pl
import pytest

from modeler.scores.quality import etf_theme_card as tc
from modeler.scores.quality import etf_theme_fee as ef


def _row(code, a, ter, op="X"):
    def tag(i, v):
        return f"<tmpV{i}>{v}</tmpV{i}>" if v is not None else f"<tmpV{i} />"

    return (
        "<selectMeta>" + tag(1, op) + tag(9, a) + tag(12, ter) + tag(15, code) + "<tmpV17 /></selectMeta>"
    )


def test_parse_duplicates_and_empty_tags():
    xml = "<root>" + _row("K1", "0.15", "0.17") + _row("K1", "9.99", "9.99") + _row("K2", None, "0.2") \
        + _row("", "1", "1") + "</root>"  # fmt: skip
    rows = ef.parse_dis_xml(xml)
    assert [r["fund_code"] for r in rows] == ["K1", "K1", "K2"]  # 코드 없는 행은 버린다
    first = ef.first_per_code(rows)
    assert first["K1"]["sum_fee"] == 0.15 and first["K1"]["ter"] == 0.17  # 중복은 첫 행
    assert first["K2"]["sum_fee"] is None and first["K2"]["ter"] == 0.2


def test_pick_basis_date_cutoff():
    ms = [date(2026, 6, 30), date(2026, 7, 31), date(2026, 8, 31), date(2026, 9, 30)]
    assert ef.pick_basis_date(ms, date(2026, 10, 10)) == date(2026, 9, 30)
    assert ef.pick_basis_date(ms[:3], date(2026, 10, 10)) == date(2026, 8, 31)
    assert ef.pick_basis_date(ms, date(2026, 9, 29)) == date(2026, 8, 31)
    assert ef.pick_basis_date(ms, date(2026, 8, 31)) == date(2026, 8, 31)  # 당일 포함
    with pytest.raises(ValueError):
        ef.pick_basis_date(ms, date(2026, 6, 29))


def test_load_month_and_available(tmp_path):
    (tmp_path / "20260731.xml.gz").write_bytes(gzip.compress(("<root>" + _row("K1", "0.1", "0.1") + "</root>").encode()))
    (tmp_path / "20260831.xml.gz").write_bytes(
        gzip.compress(("<root>" + _row("K1", "0.2", "0.25") + _row("K1", "7", "7") + "</root>").encode())
    )
    ms = ef.available_months(tmp_path)
    assert ms == [date(2026, 7, 31), date(2026, 8, 31)]
    m = ef.load_month(tmp_path, ef.pick_basis_date(ms))
    assert m["K1"]["ter"] == 0.25


def test_code_map_rule(tmp_path):
    (tmp_path / "map.csv").write_text(
        "fund_code,isin,short_code,confidence\n"
        "K1,KR7000001,000001,high\n"
        "K2,KR7000002,000002,medium\n"
        "K3,KR7000003,000003,low\n"
        "K4,,,none\n"
    )
    (tmp_path / "man.csv").write_text("short_code,fund_code\n000003,K3\n")
    cmap, known = ef.load_code_map(tmp_path / "map.csv", tmp_path / "man.csv")
    conf = dict(zip(cmap["short_code"], cmap["fund_code"]))
    assert conf == {"000001": "K1", "000003": "K3"}  # high + 수동 확인만
    assert known == {"000001", "000002", "000003"}


def test_classify_four_statuses():
    conf = {"000001": "K1", "000003": "K3", "000004": "K4"}
    known = {"000001", "000002", "000003", "000004"}
    month = {"K1": {"fund_code": "K1", "sum_fee": 0.15, "ter": 0.17}, "K4": {"fund_code": "K4", "sum_fee": None, "ter": None}}
    assert ef.classify("000001", conf, known, month) == (ef.ST_OK, 0.15, 0.17)
    assert ef.classify("000002", conf, known, month) == (ef.ST_UNCONFIRMED, None, None)
    assert ef.classify("000003", conf, known, month) == (ef.ST_NOT_IN_MONTH, None, None)  # 행 없음
    assert ef.classify("000004", conf, known, month) == (ef.ST_NOT_IN_MONTH, None, None)  # 값이 둘 다 빔
    assert ef.classify("999999", conf, known, month) == (ef.ST_NOT_IN_MAP, None, None)


def test_attach_keeps_old_columns_and_order():
    cards = pl.DataFrame({"theme": ["a", "b", "a"], "isu_cd": ["000001", "000001", "000002"], "fee": ["원천 미정"] * 3})
    fees = ef.fee_table(["000001", "000002"], {"000001": "K1"}, {"000001", "000002"},
                        {"K1": {"fund_code": "K1", "sum_fee": 0.5, "ter": 0.6}}, date(2026, 8, 31))  # fmt: skip
    out = ef.attach_fees(cards, fees)
    assert out.columns == ["theme", "isu_cd", "fee", *ef.FEE_COLUMNS]
    assert out.select("theme", "isu_cd", "fee").equals(cards)
    assert out["fee_status"].to_list() == [ef.ST_OK, ef.ST_OK, ef.ST_UNCONFIRMED]
    assert out["fee_basis_date"].unique().to_list() == [date(2026, 8, 31)]
    c = ef.summarize_status(out)
    assert c["rows"] == {ef.ST_OK: 2, ef.ST_UNCONFIRMED: 1} and c["unique_etfs"] == {ef.ST_OK: 1, ef.ST_UNCONFIRMED: 1}
    with pytest.raises(ValueError):
        ef.attach_fees(cards, pl.concat([fees, fees]))


def test_summary_and_md_with_fee_columns():
    base = dict(
        required_by_user=True, category="테마", isu_nm="N", idx_ind_nm="I", region="domestic", active=False,
        hedge="none", synthetic_display=False, bond_mix_display=False, pension_eligible_candidate=True,
        e1_pct=50.0, e1_alert=False, e1_foreign_record_only=False, e1_no_score_reason="",
        netasst_won=1e10, trdval_median_252=1e9, gap_mean_252=0.001,
    )  # fmt: skip
    cards = pl.DataFrame(
        [
            {**base, "theme": "반도체", "isu_cd": "000001", "fee_total_pct": 0.15, "ter_pct": 0.17,
             "fee_basis_date": date(2026, 8, 31), "fee_status": ef.ST_OK},
            {**base, "theme": "반도체", "isu_cd": "000002", "fee_total_pct": None, "ter_pct": None,
             "fee_basis_date": date(2026, 8, 31), "fee_status": ef.ST_UNCONFIRMED},
        ],
        schema_overrides={"e1_alert": pl.Boolean, "fee_total_pct": pl.Float64, "ter_pct": pl.Float64},
    )  # fmt: skip
    s = tc.build_summary(cards).row(0, named=True)
    assert s["n_fee_value"] == 1 and s["n_listed"] == 2
    md = tc.render_user_theme_md(cards, date(2026, 9, 30), date(2026, 10, 8))
    assert "총보수(%) | TER(%)" in md and "0.1500 | 0.1700" in md and "대응 미확인" in md
    assert "원천 미정" not in md and "2026-08-31" in md
