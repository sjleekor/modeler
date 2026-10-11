"""E3 입력 읽기: DIS XML(자기닫힘 태그), 대응표, SEIBro, sha256 검증."""

import json

import pytest

from modeler.scores.quality import etf_e3_inputs as inp

XML = """<root><selectMeta>
<tmpV1>가운용</tmpV1><tmpV2>펀드A증권상장지수투자신탁</tmpV2><tmpV9>.405</tmpV9><tmpV12>.47</tmpV12>
<tmpV15>K55207EE5208</tmpV15><tmpV16>.1009</tmpV16><tmpV17 /><tmpV18 />
</selectMeta>
<selectMeta>
<tmpV1>나운용</tmpV1><tmpV9 /><tmpV12>.1</tmpV12><tmpV15>KR5000000001</tmpV15><tmpV16></tmpV16><tmpV17 />
</selectMeta>
<selectMeta><tmpV1>코드없음</tmpV1><tmpV12>.3</tmpV12><tmpV15 /></selectMeta></root>"""


def test_parse_dis_xml_self_closing_and_blank():
    rows = inp.parse_dis_xml(XML)
    assert len(rows) == 2  # 표준코드 없는 행은 버린다
    a, b = rows
    assert a["fund_code"] == "K55207EE5208" and a["operator"] == "가운용"
    assert a["ter"] == pytest.approx(0.47) and a["sum_fee"] == pytest.approx(0.405)
    assert a["brokerage"] == pytest.approx(0.1009)
    assert b["sum_fee"] is None and b["brokerage"] is None and b["ter"] == pytest.approx(0.1)


def test_load_dis_fees_reads_gz(tmp_path):
    import gzip

    (tmp_path / "20160831.xml.gz").write_bytes(gzip.compress(XML.encode()))
    (tmp_path / "20160930.xml.gz").write_bytes(gzip.compress(XML.encode()))
    df = inp.load_dis_fees(tmp_path)
    assert df.height == 4 and df["base_date"].n_unique() == 2


def test_code_map_high_plus_manual(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text(
        "fund_code,isin,short_code,confidence\nF1,i1,111111,high\nF2,i2,222222,medium\nF3,i3,333333,high\n"
    )
    c = tmp_path / "c.csv"
    c.write_text("short_code,fund_code\n222222,F2\n")
    cm = inp.load_code_map(m, c)
    assert dict(zip(cm["short_code"], cm["level"])) == {
        "111111": "high",
        "333333": "high",
        "222222": "manual",
    }
    c.write_text("short_code,fund_code\n444444,F1\n")  # 표준코드 F1이 두 단축코드에 붙는다
    with pytest.raises(ValueError):
        inp.load_code_map(m, c)


def test_seibro_isin_slice_and_liquidation(tmp_path):
    rows = [
        {
            "ISIN": "KR7069500007",
            "RGT_STD_DT": "20130630",
            "ESTM_STDPRC": "150",
            "RGT_RSN_DTAIL_NM": "이익분배",
        },
        {
            "ISIN": "KR7069500007",
            "RGT_STD_DT": "20130930",
            "ESTM_STDPRC": "0",
            "RGT_RSN_DTAIL_NM": "이익분배",
        },
        {
            "ISIN": "KR7144750007",
            "RGT_STD_DT": "20130626",
            "ESTM_STDPRC": "14600.4",
            "RGT_RSN_DTAIL_NM": "청산분배",
        },
    ]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(rows))
    df, cats = inp.load_seibro(p)
    assert cats == {"이익분배": 2, "청산분배": 1}
    assert df["isu_cd"].to_list()[0] == "069500"
    use = inp.distributions_for_use(df)
    assert use.height == 1 and use["amount"][0] == 150.0


def test_verify_inputs_rejects_mismatch(tmp_path):
    for rel in inp.EXPECTED_SHA256:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    with pytest.raises(ValueError, match="sha256 불일치"):
        inp.verify_inputs(tmp_path, check_dis_raw=False)
