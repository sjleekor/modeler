"""사전등록 §5.2·§5.3·정정 C-1 규칙마다 작은 합성 예시. 레이크 의존 시험은 없으면 건너뛴다."""

import json
import math
import os
from pathlib import Path

import polars as pl
import pytest

from modeler.scores.quality import company_maps as cm

# ------------------------------------------------------------------ O1 의견


@pytest.mark.parametrize(
    "s",
    ["적 정", "적정의견", "적정 주1)", '"적정"', "공정(적정)", "개별-적정", "적정　", "적정(*)"],
)
def test_opinion_clean(s):
    assert cm.classify_opinion(s) == "clean"


@pytest.mark.parametrize(
    "s",
    [
        "한정",
        "부적정",  # `적정` 을 포함하지만 ① 이 먼저다
        "의견거절(*1)",
        "(별도)의견거절(주1)\n(연결)의견거절(주2)",
        "의결거절",
        "임의감사 (한정)",  # 정정 C-1 ②: 괄호 안 다른 글자는 남긴다
        "감사의견 : 적정\n반기검토의견 : 범위제한한정",  # D1: 문면대로 비적정
        "비적정",
        "(*1, *2) 반기검토 의견거절",
    ],
)
def test_opinion_non_clean(s):
    assert cm.classify_opinion(s) == "non_clean"


@pytest.mark.parametrize(
    "s",
    [
        "삼일회계법인",
        "〃",
        "＂",
        '""',
        "검토",
        "감사",
        "없음",
        "-",
        "작정",
        "적장",
        "적절",
        "",
        None,
    ],
)
def test_opinion_missing(s):
    assert cm.classify_opinion(s) is None


def test_normalize_opinion_removes_only_notes():
    assert cm.normalize_opinion("적 정 (주1)") == "적정"
    assert cm.normalize_opinion("적정 주1)") == "적정"
    assert cm.normalize_opinion("의견거절(*1, *2)") == "의견거절"
    assert cm.normalize_opinion("한정(*주5)") == "한정"
    assert cm.normalize_opinion("적정(주)") == "적정"
    assert cm.normalize_opinion("임의감사 (한정)") == "임의감사(한정)"
    assert cm.normalize_opinion("적정　\n") == "적정"
    assert cm.normalize_opinion(None) == ""


# ------------------------------------------------------------------ 연도 풀기


def test_fiscal_year_markers():
    f = cm.opinion_fiscal_year
    assert f("제22기(당기)", 2023) == 2023
    assert f("제26기\n(전기)", 2025) == 2024
    assert f("전전기", 2019) == 2017
    assert f("15년", 2017) == 2015
    assert f("2013", 2015) == 2013
    assert f("-", 2015) is None
    assert f(None, 2015) is None


def test_fiscal_year_ki_needs_anchor():
    same = ["제10기(당기)", "제9기(전기)", "제8기"]
    assert cm.opinion_fiscal_year("제8기", 2015, same) == 2013
    # CI-anchor-max: 당기 표시가 없으면 최대 기수가 앵커라 `제8기` 하나뿐이면 당기로 본다
    assert cm.opinion_fiscal_year("제8기", 2015, ["제8기"]) == 2015
    assert cm._anchor_period(["제8기"], anchor_max=False) is None
    # CI-anchor: 당기 표시가 서로 다른 M 을 가리키면 앵커 없음
    assert cm.opinion_fiscal_year("제8기", 2015, ["제10기(당기)", "제11기(당기)"]) is None


def test_fiscal_year_out_of_range_is_none():
    assert cm.opinion_fiscal_year("2009", 2015) is None
    assert cm.opinion_fiscal_year("2016", 2015) is None
    assert cm.opinion_fiscal_year("제3기", 2015, ["제10기(당기)"]) is None


# ------------------------------------------------------------------ stock_knd


@pytest.mark.parametrize(
    "s", ["보통주", "보 통 주", "의결권 있는 보통주", "보통주(대주주)", "보통주-", "보통부"]
)
def test_knd_common(s):
    assert cm.classify_stock_knd(s) == "common"


@pytest.mark.parametrize("s", ["우선주", "보통주외", "보통주 외", "1종 종류주식"])
def test_knd_excluded(s):
    assert cm.classify_stock_knd(s) == "excluded"


@pytest.mark.parametrize("s", ["-", "--", "*", "", None, "　-", " * "])
def test_knd_unmarked(s):
    assert cm.classify_stock_knd(s) == "unmarked"


@pytest.mark.parametrize("s", ["보동주", "결산배당", "의결권 있는 주식", "0"])
def test_knd_other(s):
    assert cm.classify_stock_knd(s) == "other"


# ------------------------------------------------------------------ DPS


def test_parse_dps_cell():
    assert cm.parse_dps_cell("1,250") == 1250.0
    assert cm.parse_dps_cell("-") == 0.0
    assert cm.parse_dps_cell("12.5") == 12.5
    assert cm.parse_dps_cell("해당없음") is None
    assert cm.parse_dps_cell(None) is None


def test_select_dps_common_max_per_column():
    r = cm.select_dps([("common", 25.0, 25.0, 25.0), ("common", 0.0, 15.0, None)])
    assert (r["dps_t"], r["dps_p1"], r["dps_p2"]) == (25.0, 25.0, 25.0)
    assert r["knd_source"] == "common" and r["n_common_rows"] == 2 and r["common_multi_value"]
    r = cm.select_dps([("common", 100.0, 100.0, None), ("common", 150.0, 0.0, None)])
    assert (r["dps_t"], r["dps_p1"]) == (150.0, 100.0)
    assert math.isnan(r["dps_p2"])


def test_select_dps_common_beats_unmarked_and_dash_is_zero():
    r = cm.select_dps([("unmarked", 0.0, 0.0, 0.0), ("common", 500.0, 400.0, 300.0)])
    assert r["knd_source"] == "common" and r["dps_t"] == 500.0 and not r["common_multi_value"]
    r = cm.select_dps([("unmarked", 0.0, 0.0, 0.0)])
    assert r["knd_source"] == "unmarked" and r["dps_t"] == 0.0


def test_select_dps_other_only_and_none():
    r = cm.select_dps([("other", 100.0, 100.0, 100.0), ("excluded", 50.0, 50.0, 50.0)])
    assert r["knd_source"] == "other_only" and math.isnan(r["dps_t"])
    r = cm.select_dps([])
    assert r["knd_source"] == "none" and math.isnan(r["dps_t"])
    assert cm.select_dps([("excluded", 5.0, 5.0, 5.0)])["knd_source"] == "none"


# ------------------------------------------------------------------ 사건 목록


def test_event_list_is_five_types():
    assert cm.CAPITAL_EVENT_TYPES == {"무상증자", "주식분할", "무상감자", "유상감자", "주식배당"}
    assert "유상증자(주주배정)" not in cm.CAPITAL_EVENT_TYPES


def test_date_parsing_and_anomaly():
    assert cm.parse_event_date("2015.03.02").year == 2015
    assert cm.parse_event_date("2015-03-02").month == 3
    assert cm.parse_event_date("2015.02.30") is None
    assert cm.parse_event_date("-") is None
    assert cm.date_anomaly_reason("2923.10.06", 2026) == "year_above_snapshot"
    assert cm.date_anomaly_reason("1989.12.31", 2026) == "year_below_min"
    assert cm.date_anomaly_reason("20150302", 2026) == "unparsable"
    assert cm.date_anomaly_reason("2026.09.01", 2026) is None


def _write_capital(root: Path, rows: list[dict]) -> None:
    import json

    d = root / "snapshot_date=2026-09-30" / "source=sj2_remote" / "dart_capital_change_raw"
    d.mkdir(parents=True)
    recs = []
    for i, r in enumerate(rows):
        payload = {
            "isu_dcrs_de": r["de"],
            "isu_dcrs_stle": r["stle"],
            "isu_dcrs_qy": r.get("qy", "1,000"),
        }
        recs.append(
            {"corp_code": r["corp"], "rcept_no": r["rc"], "raw_payload": json.dumps(payload)}
        )
    pl.DataFrame(recs).write_parquet(d / "part-0.parquet")


def test_capital_events_filter_dedupe_anomaly(tmp_path):
    rows = [
        {"corp": "A", "rc": "2", "de": "2016.05.01", "stle": "무상증자"},
        {"corp": "A", "rc": "1", "de": "2016.05.01", "stle": "무상증자"},  # 되풀이 → 한 건
        {"corp": "A", "rc": "1", "de": "2016.05.01", "stle": "전환권행사"},  # 목록 밖
        {"corp": "B", "rc": "3", "de": "2923.10.06", "stle": "주식분할"},  # 이상값
        {"corp": "B", "rc": "3", "de": "2017-01-02", "stle": "무상감자", "qy": "-"},
        {"corp": "C", "rc": "4", "de": "-", "stle": "-"},  # 비사건
        {"corp": "C", "rc": "4", "de": "-", "stle": "주식배당"},  # 목록 유형인데 날짜 없음
    ]
    _write_capital(tmp_path, rows)
    ev, an = cm.capital_events_and_anomalies(tmp_path, "2026-09-30")
    assert sorted(ev["event_type"].to_list()) == ["무상감자", "무상증자"]
    assert ev.filter(pl.col("corp_code") == "A")["rcept_no"].to_list() == ["1"]
    assert ev["event_year"].to_list() == [2016, 2017]
    assert sorted(an["reason"].to_list()) == ["unparsable", "year_above_snapshot"]
    assert an["corp_code"].to_list() == ["B", "C"]


# ------------------------------------------------------------------ 레이크 (없으면 skip)

_LAKE = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data")) / "kr/raw/raw_postgres"
_SNAP = "2026-09-30"
_needs_lake = pytest.mark.skipif(
    not cm.table_dir(_LAKE, _SNAP, "dart_governance_raw").exists(), reason="raw 레이크 없음"
)


@_needs_lake
def test_lake_opinion_rows_shape():
    op = cm.audit_opinion_rows(_LAKE, _SNAP)
    assert op.columns == [
        "corp_code",
        "rcept_no",
        "report_year",
        "row_ordinal",
        "label_raw",
        "fiscal_year",
        "opinion_raw",
        "opinion_norm",
        "opinion_class",
    ]
    assert op.select(["rcept_no", "row_ordinal"]).n_unique() == op.height
    ok = op.filter(pl.col("fiscal_year").is_not_null())
    assert (
        (ok["fiscal_year"] <= ok["report_year"]) & (ok["fiscal_year"] >= ok["report_year"] - 2)
    ).all()


# ------------------------------------------------------------------ CI-anchor-max


def test_anchor_max_without_dang_label():
    same = ["제10기", "제9기", "제8기"]
    f = cm.opinion_fiscal_year
    assert f("제10기", 2015, same) == 2015
    assert f("제9기", 2015, same) == 2014
    assert f("제8기", 2015, same) == 2013
    # 앵커 최댓값을 끄면 풀리지 않는다
    assert cm._anchor_period(same, anchor_max=False) is None
    # (당기) 앵커가 있으면 최댓값보다 앵커가 우선
    assert cm._anchor_period(["제10기(당기)", "제12기"]) == 10
    # 앵커가 여럿이면 그대로 None
    assert cm._anchor_period(["제10기(당기)", "제11기(당기)", "제12기"]) is None


def test_audit_opinion_rows_anchor_max_arg(tmp_path):
    """anchor_max 기본값은 지금과 같고, False면 (당기) 표시 없는 `제N기`는 연도가 None이다."""
    d = tmp_path / "snapshot_date=2026-09-30" / "source=sj2_remote" / "dart_governance_raw"
    d.mkdir(parents=True)
    rows = []
    for i, lab in enumerate(["제10기", "제9기", "제8기"]):
        payload = json.dumps({"bsns_year": lab, "adt_opinion": "적정의견"})
        rows.append({"corp_code": "A", "rcept_no": "R1", "bsns_year": 2015, "row_ordinal": i,
                     "statement_type": "audit_opinion", "raw_payload": payload})  # fmt: skip
    rows.append({"corp_code": "B", "rcept_no": "R2", "bsns_year": 2015, "row_ordinal": 0,
                 "statement_type": "audit_opinion",
                 "raw_payload": json.dumps({"bsns_year": "제10기(당기)", "adt_opinion": "적정"})
                 })  # fmt: skip
    pl.DataFrame(rows).with_columns(pl.col("bsns_year").cast(pl.Int32)).write_parquet(
        d / "p.parquet"
    )
    default = cm.audit_opinion_rows(tmp_path, "2026-09-30")
    assert default.equals(cm.audit_opinion_rows(tmp_path, "2026-09-30", anchor_max=True))
    a = default.filter(pl.col("corp_code") == "A").sort("row_ordinal")
    assert a["fiscal_year"].to_list() == [2015, 2014, 2013]
    off = cm.audit_opinion_rows(tmp_path, "2026-09-30", anchor_max=False)
    assert off.filter(pl.col("corp_code") == "A")["fiscal_year"].to_list() == [None, None, None]
    # (당기) 앵커가 있는 보고서는 두 판이 같다
    b = lambda df: df.filter(pl.col("corp_code") == "B")["fiscal_year"].to_list()  # noqa: E731
    assert b(default) == b(off) == [2015]


# ------------------------------------------------------------------ 액면 변경 해 (정정 C-2)


def _write_div(root: Path, rows: list[dict]) -> None:
    import json

    d = root / "snapshot_date=2026-09-30" / "source=sj2_remote" / "dart_shareholder_return_raw"
    d.mkdir(parents=True)
    recs = []
    for r in rows:
        payload = {"thstrm": r["t"], "frmtrm": r["p1"], "lwfr": r["p2"]}
        recs.append(
            {
                "corp_code": r["corp"],
                "rcept_no": r["rc"],
                "bsns_year": r["y"],
                "statement_type": "dividend",
                "reprt_code": "11011",
                "row_name": "주당액면가액(원)",
                "stock_knd": r.get("knd", "-"),
                "raw_payload": json.dumps(payload),
            }
        )
    pl.DataFrame(recs).write_parquet(d / "part-0.parquet")


def test_share_event_years_par_change(tmp_path):
    div = tmp_path / "div"
    cap = tmp_path / "cap"
    _write_div(
        div,
        [
            {"corp": "A", "rc": "1", "y": 2017, "t": "500", "p1": "5,000", "p2": "5,000"},  # t
            {"corp": "B", "rc": "2", "y": 2017, "t": "500", "p1": "500", "p2": "5,000"},  # t-1
            {"corp": "C", "rc": "3", "y": 2017, "t": "500", "p1": "500", "p2": "500"},  # 없음
            {"corp": "D", "rc": "4", "y": 2017, "t": "-", "p1": "-", "p2": "500"},  # `-` 없음
            {"corp": "E", "rc": "5", "y": 2017, "t": "100", "p1": "500", "p2": "500"},
            {"corp": "E", "rc": "6", "y": 2019, "t": "50", "p1": "100", "p2": "100"},  # 합집합
            {
                "corp": "F",
                "rc": "7",
                "y": 2017,
                "t": "100",
                "p1": "500",
                "p2": "500",
                "knd": "우선주",
            },
        ],
    )
    # 같은 snapshot 디렉터리 구조에 capital 표도 둔다(분리 root 두 개는 불가라 한 root 에 합친다).
    _write_capital(div, [{"corp": "A", "rc": "9", "de": "2017.03.01", "stle": "무상증자"}])
    ey = cm.share_event_years(div, "2026-09-30")
    par = ey.filter(pl.col("source") == "par_change")
    got = sorted(zip(par["corp_code"], par["event_year"], strict=True))
    assert got == [("A", 2017), ("B", 2016), ("E", 2017), ("E", 2019)]
    assert set(par["event_type"]) == {"액면변경"}
    assert ey.filter(pl.col("source") == "capital_change").height == 1
    fl = cm.share_event_flags(div, "2026-09-30", sources=cm.RECORD_EVENT_SOURCES)
    assert fl.filter(pl.col("corp_code") == "A").height == 1  # 두 원천이 같은 해여도 한 행
    assert fl.columns == ["corp_code", "year"]
    # 기본(판정용)은 동결 목록뿐 — 액면 변경 해는 기록용에만(정정 C-2)
    fj = cm.share_event_flags(div, "2026-09-30")
    assert sorted(zip(fj["corp_code"], fj["year"], strict=True)) == [("A", 2017)]
    assert cap is not None


# ------------------------------------------------------------------ 매핑표 비교 (W8)
def _write_fake_maps(d: Path, *, opinions, knds, events, ses, snapshot="2026-09-30", ver="v1"):
    d.mkdir(parents=True, exist_ok=True)
    cm.write_tsv(
        pl.DataFrame(
            opinions, schema=["opinion_raw", "opinion_norm", "opinion_class"], orient="row"
        ),
        d / "opinion_map.tsv",
    )
    cm.write_tsv(
        pl.DataFrame(
            knds, schema=["stock_knd", "knd_norm", "knd_class", "n_reports"], orient="row"
        ),
        d / "stock_knd_map.tsv",
    )
    cm.write_tsv(
        pl.DataFrame(events, schema=["event_type", "in_list"], orient="row"),
        d / "capital_event_types.tsv",
    )
    if ses is not None:
        cm.write_tsv(
            pl.DataFrame(ses, schema=["se_raw", "se_norm", "knd_class"], orient="row"),
            d / "share_count_se_map.tsv",
        )
    (d / "manifest.json").write_text(
        json.dumps({"rules_version": ver, "raw_snapshot": snapshot, "module_sha256": "x"})
    )


def _fake_pair(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write_fake_maps(
        old,
        opinions=[("적정", "적정", "clean"), ("옛 의견\n줄바꿈", "옛의견줄바꿈", None)],
        knds=[("보통주", "보통주", "common", 5)],
        events=[("무상증자", True)],
        ses=[("보통주", "보통주", "common"), ("우선주", "우선주", "excluded")],
    )
    _write_fake_maps(
        new,
        opinions=[
            ("적정", "적정", "clean"),
            ("한정 (주1)", "한정", "non_clean"),  # 새 문자열
        ],
        knds=[("보통주", "보통주", "excluded", 5), ("신종 보통", "신종보통", "common", 1)],
        events=[("무상증자", True), ("신규유형", False)],
        ses=[("보통주", "보통주", "common"), ("신보통주", "신보통주", "common")],
        snapshot="2026-10-18",
    )
    return new, old


def test_compare_maps_catches_new_removed_and_class_change(tmp_path):
    new, old = _fake_pair(tmp_path)
    rows, s = cm.compare_maps(new, old)
    key = {(r["kind"], r["change"], r["raw"]) for r in rows}
    assert ("opinion", "new", "한정 (주1)") in key
    assert s["opinion"]["n_new_strings"] == 1 and s["opinion"]["n_removed_strings"] == 1
    assert s["opinion"]["n_new_by_class"] == {"non_clean": 1}
    # 같은 원문인데 분류가 바뀌면 따로 잡는다(new 가 아니다)
    assert ("stock_knd", "class_changed", "보통주") in key
    assert s["stock_knd"]["n_class_changed"] == 1 and s["stock_knd"]["n_new_strings"] == 1
    assert s["capital_event_type"]["n_new_strings"] == 1
    assert s["share_count_se"]["n_new_strings"] == 1 and s["share_count_se"]["n_new_common"] == 1
    assert s["share_count_se"]["n_removed_strings"] == 1
    assert s["n_new_strings_total"] == 4 and s["n_class_changed_total"] == 1
    assert s["old_raw_snapshot"] == "2026-09-30" and s["new_raw_snapshot"] == "2026-10-18"
    assert s["rules_version"]["same"] is True and s["module_sha256"]["same"] is True
    # 건수 열이 없다(§7.2): 비교 행은 문자열·분류뿐
    assert set(rows[0]) == set(cm.COMPARE_COLUMNS)


def test_compare_maps_same_maps_zero(tmp_path):
    new, old = _fake_pair(tmp_path)
    _, s = cm.compare_maps(old, old)
    assert s["n_new_strings_total"] == 0 and s["n_class_changed_total"] == 0
    assert all(s[k]["n_removed_strings"] == 0 for k in ("opinion", "stock_knd", "share_count_se"))


def test_compare_maps_old_without_se_map_is_unavailable(tmp_path):
    new, old = _fake_pair(tmp_path)
    (old / "share_count_se_map.tsv").unlink()
    _, s = cm.compare_maps(new, old)  # raw_root 가 없어 다시 만들 수 없다
    assert s["share_count_se"]["available"] is False
    assert s["share_count_se"]["old_source"] == "unavailable"


def test_compare_maps_old_without_se_map_recomputed_from_raw(tmp_path):
    new, old = _fake_pair(tmp_path)
    (old / "share_count_se_map.tsv").unlink()
    d = cm.table_dir(tmp_path / "raw", "2026-09-30", "dart_share_count_raw")
    d.mkdir(parents=True)
    pl.DataFrame(
        {"reprt_code": ["11011", "11011", "11012"], "se": ["보통주", "우선주", "기타"]}
    ).write_parquet(d / "p.parquet")
    _, s = cm.compare_maps(new, old, raw_root=tmp_path / "raw")
    se = s["share_count_se"]
    assert se["old_source"] == "recomputed_from_raw" and se["n_new_strings"] == 1  # 11011 만
    assert se["n_removed_strings"] == 1


def test_write_compare_files_and_cli(tmp_path):
    new, old = _fake_pair(tmp_path)
    tsv, sp, _ = cm.write_compare(new, old)
    assert tsv.name == "compare_to_2026-09-30.tsv" and sp.name == "compare_summary.json"
    assert json.loads(sp.read_text())["n_new_strings_total"] == 4
    head, *body = tsv.read_text().splitlines()
    assert head.split("\t") == list(cm.COMPARE_COLUMNS) and len(body) == 7


def test_share_count_se_map_common_check(tmp_path):
    d = cm.table_dir(tmp_path, "2026-09-30", "dart_share_count_raw")
    d.mkdir(parents=True)
    pl.DataFrame(
        {"reprt_code": ["11011"] * 4, "se": ["보통주", " 우선주 ", "합계", None]}
    ).write_parquet(d / "p.parquet")
    m = cm.share_count_se_map(tmp_path, "2026-09-30")
    assert dict(zip(m["se_raw"], m["knd_class"], strict=True)) == {
        "보통주": "common", " 우선주 ": "excluded", "합계": "other",
    }  # fmt: skip
    assert cm.share_count_se_map(tmp_path, "1999-01-01").height == 0  # 표 없음 → 빈 표
