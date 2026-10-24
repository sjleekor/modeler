"""정본(krx_baseline) 읽기 시험 — 합성 parquet. collector를 import하지 않는다."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modeler.scores.etf_baseline import canonical as C
from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import run as R

ETF_COLS = ["BAS_DD", "ISU_CD", "TDD_CLSPRC", "LIST_SHRS"]
DIST_COLS = ["ISIN", "RGT_STD_DT", "TH1_PAY_TERM_BEGIN_DT", "ESTM_STDPRC", "RGT_RSN_DTAIL_NM"]


class Builder:
    """manifest 계약대로 파일을 쓰는 최소 작성기."""

    def __init__(self, base: Path):
        self.base = base
        self.n = 0
        (base / "_manifest").mkdir(parents=True)

    def _write(self, rel: str, table: pa.Table) -> dict:
        path = self.base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        data = path.read_bytes()
        return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def commit(self, obs=(), logs=(), orphan=False) -> None:
        """obs: (table, year, rows) · logs: (service, rows). ``orphan``이면 manifest를 안 쓴다."""
        self.n += 1
        stem = f"run__001__{self.n:05d}"
        files = []
        for table, year, rows in obs:
            tbl = pa.table(
                {
                    **{
                        k: pa.array([r.get(k) for r in rows], pa.string())
                        for k in rows[0]
                        if k not in ("obs_seq", "obs_kind")
                    },
                    "obs_seq": pa.array([r.get("obs_seq", 1) for r in rows], pa.int64()),
                    "obs_kind": pa.array([r.get("obs_kind", "value") for r in rows], pa.string()),
                }
            )
            e = self._write(f"{table}/year={year}/{stem}.parquet", tbl)
            files.append({"kind": "observation", "table": table, "rows": len(rows), **e})
        for service, rows in logs:
            tbl = pa.table(
                {
                    "request_key": pa.array([r["request_key"] for r in rows], pa.string()),
                    "fetched_at": pa.array(
                        [r["fetched_at"] for r in rows], pa.timestamp("us", tz="UTC")
                    ),
                    "result": pa.array([r.get("result", "new_obs") for r in rows], pa.string()),
                    "day_kind": pa.array([r.get("day_kind") for r in rows], pa.string()),
                }
            )
            e = self._write(f"fetch_log/{service}/{stem}.parquet", tbl)
            files.append({"kind": "fetch_log", "service": service, "rows": len(rows), **e})
        if orphan:
            return
        m = {"schema_version": 1, "run_id": "run", "files": files}
        (self.base / "_manifest" / f"{stem}.json").write_text(json.dumps(m))


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d = date.fromordinal(d.toordinal() + 1)


def _log(day: date, kind="trading", result="new_obs", hour=12):
    return {
        "request_key": f"bas_dd={day:%Y%m%d}",
        "fetched_at": datetime(2026, 10, 9, hour, tzinfo=UTC),
        "result": result,
        "day_kind": kind,
    }


def _etf(day: date, code="069500", close="100", shares="10", **kw):
    return {
        "BAS_DD": f"{day:%Y%m%d}",
        "ISU_CD": code,
        "TDD_CLSPRC": close,
        "LIST_SHRS": shares,
        **kw,
    }


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "HISTORY_START", date(2020, 1, 6))
    return tmp_path / "krx_baseline"


def _basic(store_dir, end=date(2020, 1, 10)):
    b = Builder(store_dir)
    days = list(_weekdays(date(2020, 1, 6), end))
    holiday = days[2]
    logs = [_log(d, "no_price" if d == holiday else "trading") for d in days]
    rows = [_etf(d, close=str(100 + i)) for i, d in enumerate(days) if d != holiday]
    b.commit(obs=[("etf_daily", 2020, rows)], logs=[("etf_bydd_trd", logs)])
    return b, days, holiday


def test_trading_days_from_fetch_log_day_kind(store_dir):
    _, days, holiday = _basic(store_dir)
    got, info = C.load_trading_days(C.CanonicalStore(store_dir), date(2020, 1, 10))
    assert got == [d for d in days if d != holiday]
    assert info["day_kind_counts"] == {"no_price": 1, "trading": 4}


def test_end_date_cuts_calendar_but_must_be_trading(store_dir):
    _, days, holiday = _basic(store_dir)
    st = C.CanonicalStore(store_dir)
    got, _ = C.load_trading_days(st, days[3])
    assert got[-1] == days[3] and days[4] not in got
    with pytest.raises(I.InputError, match="끝 날짜"):
        C.load_trading_days(st, holiday)


def test_last_successful_record_wins_and_failed_ignored(store_dir):
    b, days, holiday = _basic(store_dir)
    # 나중에 다시 받으니 거래일이 됐다 + 더 늦은 실패 기록은 무시한다
    b.commit(
        logs=[
            (
                "etf_bydd_trd",
                [_log(holiday, "trading", hour=15), _log(holiday, "empty", "failed", hour=20)],
            )
        ]
    )
    got, _ = C.load_trading_days(C.CanonicalStore(store_dir), date(2020, 1, 10))
    assert holiday in got


def test_missing_weekday_record_stops(store_dir):
    b = Builder(store_dir)
    days = list(_weekdays(date(2020, 1, 6), date(2020, 1, 10)))
    b.commit(logs=[("etf_bydd_trd", [_log(d) for d in days if d != days[1]])])
    with pytest.raises(I.InputError, match="성공한 요청 기록이 없는 날 1개"):
        C.load_trading_days(C.CanonicalStore(store_dir), date(2020, 1, 10))


def test_history_start_missing_stops(store_dir):
    b = Builder(store_dir)
    days = list(_weekdays(date(2020, 1, 7), date(2020, 1, 10)))
    b.commit(logs=[("etf_bydd_trd", [_log(d) for d in days])])
    with pytest.raises(I.InputError, match="성공한 요청 기록이 없는 날"):
        C.load_trading_days(C.CanonicalStore(store_dir), date(2020, 1, 10))


def test_only_manifest_committed_files_are_read(store_dir):
    b, days, _ = _basic(store_dir)
    b.commit(obs=[("etf_daily", 2020, [_etf(days[0], close="999")])], orphan=True)
    fr = C.load_etf_frames(C.CanonicalStore(store_dir), ["069500"])
    assert fr["069500"]["close"][0] == 100.0


def test_manifest_size_and_sha_mismatch_stop(store_dir):
    _basic(store_dir)
    victim = next((store_dir / "etf_daily").rglob("*.parquet"))
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(I.InputError, match="크기가 manifest와 다릅니다"):
        C.load_etf_frames(C.CanonicalStore(store_dir), ["069500"])


def test_no_manifest_stops(tmp_path):
    (tmp_path / "krx_baseline" / "_manifest").mkdir(parents=True)
    with pytest.raises(I.InputError, match="완료 manifest가 하나도 없습니다"):
        C.CanonicalStore(tmp_path / "krx_baseline")
    with pytest.raises(I.InputError, match="_manifest"):
        C.CanonicalStore(tmp_path / "nowhere")


def test_views_first_latest_and_absent(store_dir):
    b, days, _ = _basic(store_dir)
    d0, d1 = days[0], days[1]
    b.commit(
        obs=[
            (
                "etf_daily",
                2020,
                [
                    _etf(d0, close="777", obs_seq=2),  # 정정
                    _etf(d1, close="", shares="", obs_seq=2, obs_kind="absent"),  # 사라짐
                ],
            )
        ]
    )
    st = C.CanonicalStore(store_dir)
    latest = C.load_etf_frames(st, ["069500"], "latest")["069500"]
    first = C.load_etf_frames(st, ["069500"], "first")["069500"]
    assert latest.filter(pl.col("date") == d0)["close"][0] == 777.0
    assert first.filter(pl.col("date") == d0)["close"][0] == 100.0
    assert latest.filter(pl.col("date") == d1).is_empty()  # absent는 뺀다
    assert not first.filter(pl.col("date") == d1).is_empty()


def test_empty_etf_table_stops(store_dir):
    b = Builder(store_dir)
    days = list(_weekdays(date(2020, 1, 6), date(2020, 1, 7)))
    b.commit(logs=[("etf_bydd_trd", [_log(d) for d in days])])
    with pytest.raises(I.InputError, match="etf_daily 관측 표가 비었습니다"):
        C.load_etf_frames(C.CanonicalStore(store_dir), ["069500"])


def test_frames_match_research_conversion(store_dir):
    _, days, holiday = _basic(store_dir)
    st = C.CanonicalStore(store_dir)
    got = C.load_etf_frames(st, ["069500"])["069500"]
    raw = pl.DataFrame([_etf(d, close=f"{100 + i:,}") for i, d in enumerate(days) if d != holiday])
    want = I.frames_from_strings(raw, ["069500"])["069500"]
    assert got.shape == want.shape and got["date"].to_list() == want["date"].to_list()
    assert got["close"].to_list() == [float(100 + i) for i, d in enumerate(days) if d != holiday]


def _dist(isin, rgt, amt, pay="20200110", kind="이익분배", **kw):
    return {
        "ISIN": isin,
        "RGT_STD_DT": rgt,
        "TH1_PAY_TERM_BEGIN_DT": pay,
        "ESTM_STDPRC": amt,
        "RGT_RSN_DTAIL_NM": kind,
        **kw,
    }


def _dist_store(store_dir, rows, windows=(("20030101", "20200111"),)):
    b = Builder(store_dir)
    logs = [
        {
            "request_key": f"window={a}_{z}",
            "fetched_at": datetime(2020, 1, 11, tzinfo=UTC),
            "result": "new_obs",
            "day_kind": "complete",
        }
        for a, z in windows
    ]
    obs = [("etf_distribution", 2019, rows)] if rows else []
    b.commit(obs=obs, logs=[("seibro_dist", logs)])
    return C.CanonicalStore(store_dir)


def test_distributions_code_from_isin_and_filters(store_dir):
    rows = [
        _dist("KR7069500007", "20191231", "1,000"),
        _dist("KR7069500007", "20191231", "1,050", obs_seq=2),  # 같은 키의 정정판 — 최신이 이긴다
        _dist("KR7069500007", "20200101", "5", kind="청산분배"),
        _dist("KR7999999999", "20191231", "7"),  # 대상 아님
    ]
    st = _dist_store(store_dir, rows)
    info, win = C.load_distributions(st, ["069500", "114260"], date(2020, 1, 10))
    assert info.isin_by_code == {"069500": "KR7069500007", "114260": ""}
    assert info.rows["amount"].to_list() == [1050.0]
    assert info.duplicates_dropped == 0 and info.liquidation_rows_ignored == 1
    first, _ = C.load_distributions(st, ["069500"], date(2020, 1, 10), "first")
    assert first.rows["amount"].to_list() == [1000.0]
    assert win["last_complete_window_end"] == date(2020, 1, 11)


def test_distribution_window_checks_stop(store_dir):
    rows = [_dist("KR7069500007", "20191231", "5")]
    st = _dist_store(store_dir, rows, windows=(("20200101", "20200111"),))
    with pytest.raises(I.InputError, match="전체 받기 완료 창이 없습니다"):
        C.load_distributions(st, ["069500"], date(2020, 1, 10))


def test_distribution_window_too_early_stops(tmp_path):
    st = _dist_store(
        tmp_path / "a",
        [_dist("KR7069500007", "20191231", "5")],
        windows=(("20030101", "20200105"),),
    )
    with pytest.raises(I.InputError, match="마지막 완료 창 끝"):
        C.load_distributions(st, ["069500"], date(2020, 1, 10))


def test_distribution_empty_and_two_isins_stop(tmp_path):
    st = _dist_store(tmp_path / "a", [])
    with pytest.raises(I.InputError, match="etf_distribution 관측 표가 비었습니다"):
        C.load_distributions(st, ["069500"], date(2020, 1, 10))
    st = _dist_store(
        tmp_path / "b",
        [_dist("KR7069500007", "20191231", "5"), _dist("KR8069500001", "20191230", "5")],
    )
    with pytest.raises(I.InputError, match="ISIN이 둘 이상"):
        C.load_distributions(st, ["069500"], date(2020, 1, 10))


def test_manifest_digest_changes_with_new_manifest(store_dir):
    b, days, _ = _basic(store_dir)
    d1 = C.CanonicalStore(store_dir).manifest_digest()
    b.commit(logs=[("etf_bydd_trd", [_log(days[0], hour=18)])])
    d2 = C.CanonicalStore(store_dir).manifest_digest()
    assert d1["count"] == 1 and d2["count"] == 2 and d1["sha256"] != d2["sha256"]


def _prepared(root: Path, canonical: bool) -> R.Prepared:
    src = {"kind": "canonical" if canonical else "research", "dist_kind": "research"}
    return R.Prepared(root, [], None, None, None, None, 0, {}, source=src)


def test_output_dir_never_inside_frozen_research_results(tmp_path):
    root = tmp_path
    frozen = root / I.OUTPUT_DIR
    (frozen / "run_20261018_main").mkdir(parents=True)
    p = _prepared(root, True)
    with pytest.raises(I.InputError, match="조사 사본 결과 폴더"):
        R.resolve_out_dir(p, "x", frozen / "run_20261018_main")
    with pytest.raises(I.InputError, match="조사 사본 결과 폴더"):
        R.resolve_out_dir(p, "x", frozen)
    out = R.resolve_out_dir(p, "canon_1", None)
    assert out == root / R.CANON_OUTPUT_DIR / "canon_1" and frozen not in out.parents


def test_canonical_output_dir_refuses_existing_results(tmp_path):
    p = _prepared(tmp_path, True)
    out = tmp_path / R.CANON_OUTPUT_DIR / "canon_1"
    out.mkdir(parents=True)
    (out / "metrics.csv").write_text("x")
    with pytest.raises(I.InputError, match="이미 결과가 있습니다"):
        R.resolve_out_dir(p, "canon_1", None)


def test_research_is_default_source_and_old_output_dir(tmp_path):
    ap = __import__("argparse").ArgumentParser()
    R.add_source_args(ap)
    a = ap.parse_args([])
    assert (a.source, a.dist_source, a.obs_view, a.end_date) == (
        "research",
        None,
        "latest",
        R.END_DATE,
    )
    p = _prepared(tmp_path, False)
    assert R.resolve_out_dir(p, "run_x", None) == tmp_path / I.OUTPUT_DIR / "run_x"
