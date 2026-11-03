"""F4/F3 end to end on a synthetic lake. The real marts are never touched."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import cli, digest, manifest, seal
from modeler.scores.earnings_a import paths as P

FLOATISH = re.compile(r"-?\d+\.\d{3,}")
# keys whose string values may legitimately look like long decimals (versions, paths)
ALLOWED_KEYS = {
    "env",
    "inputs",
    "repo",
    "describe",
    "branch",
    "head",
    "clip_q",
    "hgb_grid",
    "ridge_alphas",
}
RUN_AT = datetime(2026, 11, 3, tzinfo=UTC)
TICKERS = {"000010": "C1", "000020": "C2"}


def _qe(start_year=2015, end=date(2026, 6, 30)):
    out = []
    for y in range(start_year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            if date(y, m, d) <= end:
                out.append(date(y, m, d))
    return out


def _w(path, df, casts=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.register("df", df)
    con.execute(f"COPY (SELECT *{casts} FROM df) TO '{path}' (FORMAT PARQUET)")
    con.close()


def _mart_rows(shuffle_seed=None, with_ttm=True):
    rng = np.random.default_rng(7)
    rows = []
    for ti, (tk, cc) in enumerate(TICKERS.items()):
        for pe in _qe():
            avail = pe + timedelta(days=45)
            for metric, base in (
                ("total_assets", 1000.0 + 50 * ti),
                ("operating_income", 30.0 + 5 * ti),
                ("operating_cash_flow", 40.0),
            ):
                v = base * (1 + 0.1 * rng.standard_normal()) + 0.1234
                rows.append(
                    {
                        "ticker": tk,
                        "corp_code": cc,
                        "metric_code": metric,
                        "fs_basis": "CFS",
                        "statement_period_end": pe,
                        "available_from": avail,
                        "standalone_value": round(v, 4),
                        "ttm_value": round(4 * v, 4) if with_ttm else None,
                        "ttm_complete": True,
                        "ttm_available_from": avail,
                        "ttm_includes_conflict": False,
                        "fiscal_year_end": date(pe.year, 12, 31),
                        "rcept_no": f"{avail:%Y%m%d}{ti:06d}",
                        "standalone_null_reason": None,
                    }
                )
    df = pd.DataFrame(rows)
    for c in ("statement_period_end", "available_from", "ttm_available_from", "fiscal_year_end"):
        df[c] = pd.to_datetime(df[c]).dt.date
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    return df


CAST = (
    " REPLACE (CAST(standalone_value AS DECIMAL(38,4)) AS standalone_value,"
    " CAST(ttm_value AS DECIMAL(38,4)) AS ttm_value)"
)


def _drop_oi(df, tickers):
    """Remove operating_income rows for ``tickers``: their OI is missing, so no row is eligible."""
    if not tickers:
        return df
    keep = ~(df["ticker"].isin(tickers) & (df["metric_code"] == "operating_income"))
    return df[keep].reset_index(drop=True)


def build_lake(root, monkeypatch, r2_shuffle=None, no_oi_for=()):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(root))
    paths = P.DataPaths.resolve()
    full = _drop_oi(_mart_rows(), no_oi_for)
    v4p = full.drop(columns=["rcept_no", "standalone_null_reason"])
    _w(paths.v4p_parquet, v4p, CAST)
    r2 = _drop_oi(_mart_rows(r2_shuffle), no_oi_for)[
        [
            "ticker",
            "corp_code",
            "metric_code",
            "fs_basis",
            "statement_period_end",
            "available_from",
            "standalone_value",
            "rcept_no",
            "standalone_null_reason",
        ]
    ]
    _w(
        paths.r2_parquet,
        r2,
        " REPLACE (CAST(standalone_value AS DECIMAL(38,4)) AS standalone_value)",
    )

    # raw tables: one trading day on every quarter-end
    days = _qe(2015, date(2026, 10, 22))
    oh = pd.DataFrame(
        [(d, tk, "KOSPI", 5000, 5100, 4900, 5000, 100000) for d in days for tk in TICKERS],
        columns=["trade_date", "ticker", "market", "open", "high", "low", "close", "volume"],
    )
    _w(paths.raw_root / "daily_ohlcv/schema_version=1/p.parquet", oh)
    mc = oh[["trade_date", "ticker", "market", "close", "volume"]].rename(
        columns={"close": "source_close"}
    )
    mc["trading_value"] = mc["source_close"] * mc["volume"]
    _w(paths.raw_root / "daily_market_cap/schema_version=1/p.parquet", mc)
    sm = pd.DataFrame(
        {
            "ticker": list(TICKERS),
            "market": "KOSPI",
            "name": ["alpha", "beta"],
            "status": "ACTIVE",
            "last_seen_date": date(2026, 10, 22),
        }
    )
    _w(paths.raw_root / "stock_master/schema_version=1/p.parquet", sm)
    corp = pd.DataFrame(
        {
            "ticker": list(TICKERS),
            "corp_code": list(TICKERS.values()),
            "induty_code": "26000",
            "corp_name": ["alpha", "beta"],
        }
    )
    _w(paths.raw_root / "dart_corp_master/schema_version=1/p.parquet", corp)
    rc = full[["ticker", "corp_code", "rcept_no"]].drop_duplicates("rcept_no").copy()
    rc["rcept_dt"] = pd.to_datetime(rc["rcept_no"].str[:8]).dt.date
    rc["report_nm"] = "분기보고서 (2020.03)"
    rc.iloc[3, rc.columns.get_loc("report_nm")] = "[기재정정]분기보고서 (2020.03)"
    _w(paths.raw_root / "dart_filing_receipt_raw/schema_version=1/p.parquet", rc)
    fs = pd.DataFrame({"corp_code": list(TICKERS.values()), "currency": "KRW"})
    _w(paths.raw_root / "dart_financial_statement_raw/schema_version=1/p.parquet", fs)

    for k, p in (("v4p", paths.v4p_parquet), ("r2", paths.r2_parquet)):
        monkeypatch.setitem(P.EXPECTED_SHA256, k, P.sha256_file(p))
    return paths


def _run_cli(args, capsys):
    rc = cli.main(args)
    out = capsys.readouterr().out.strip().splitlines()
    return rc, out


def _scan(obj, key=None, hits=None):
    hits = [] if hits is None else hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            _scan(v, k if key is None else key, hits)
    elif isinstance(obj, list):
        for v in obj:
            _scan(v, key, hits)
    elif isinstance(obj, float):
        hits.append((key, repr(obj)))
    elif isinstance(obj, str) and FLOATISH.search(obj):
        hits.append((key, "<str>"))
    return [h for h in hits if h[0] not in ALLOWED_KEYS]


def test_f4_rehearsal_end_to_end(tmp_path, monkeypatch, capsys):
    build_lake(tmp_path / "lake", monkeypatch)
    rc, out = _run_cli(["f4", "--rehearsal"], capsys)
    assert rc == 0 and len(out) == 2
    out_dir, mpath = out
    assert mpath.endswith("manifest_f4.json")
    m = json.loads(open(mpath).read())
    assert m["kind"] == "F4_hash_only_REHEARSAL"
    assert set(m["digests"]) >= {
        "DEV.oi_y1",
        "GAP.oi_y1",
        "SEALED.oi_y1",
        "DEV.oi_y2",
        "SEALED.ocf_y1",
    }
    assert all(len(d["exact_sha256"]) == 64 for d in m["digests"].values())
    assert set(m["reason_digest"]) == {"DEV", "GAP", "SEALED"}
    assert m["decision_dates"]["CONFIRM"][-1] == "2026-06-30"
    assert m["inputs"]["r2"]["rows"] > 0
    # no value-looking numbers anywhere in the written files
    import pathlib

    files = list(pathlib.Path(out_dir).rglob("*"))
    assert [f.name for f in files if f.is_file()] == ["manifest_f4.json"]
    assert _scan(m) == []
    for f in files:
        for hit in FLOATISH.findall(f.read_text()):
            pytest.fail(f"decimal-looking token in {f.name}")  # pragma: no cover
    # write-once
    with pytest.raises(FileExistsError):
        manifest.write_once(mpath, {})
    # out dir must not exist
    with pytest.raises(FileExistsError):
        cli.main(["f4", "--rehearsal", "--out", out_dir])


def test_f4_refuses_dirty_without_rehearsal(tmp_path, monkeypatch):
    build_lake(tmp_path / "lake", monkeypatch)
    monkeypatch.setattr(
        manifest,
        "git_state",
        lambda repo: {"head": "x", "branch": "b", "dirty": True, "describe": "d"},
    )
    with pytest.raises(SystemExit):
        cli.main(["f4"])
    assert not list((tmp_path / "lake").glob("kr/output/earnings_forecast_a/f4_*"))


def test_f4_clean_kind(tmp_path, monkeypatch, capsys):
    build_lake(tmp_path / "lake", monkeypatch)
    monkeypatch.setattr(
        manifest,
        "git_state",
        lambda repo: {"head": "x", "branch": "b", "dirty": False, "describe": "d"},
    )
    _, out = _run_cli(["f4"], capsys)
    assert json.loads(open(out[1]).read())["kind"] == "F4_hash_only"


def test_input_hash_mismatch_stops(tmp_path, monkeypatch):
    build_lake(tmp_path / "lake", monkeypatch)
    monkeypatch.setitem(P.EXPECTED_SHA256, "r2", "0" * 64)
    with pytest.raises(RuntimeError):
        cli.main(["f4", "--rehearsal"])


def test_digest_stable_under_r2_row_order(tmp_path, monkeypatch):
    paths = build_lake(tmp_path / "a", monkeypatch, r2_shuffle=None)
    d1 = digest.run_f4(paths, P.connect_inputs(paths), run_at=RUN_AT)
    paths = build_lake(tmp_path / "b", monkeypatch, r2_shuffle=3)
    d2 = digest.run_f4(paths, P.connect_inputs(paths), run_at=RUN_AT)
    assert d1 == d2
    assert d1["digests"]["DEV.oi_y1"]["n_rows"] > 0


def test_anchor_excludes_rows_not_input_eligible_incl_holding(tmp_path, monkeypatch):
    # 000020 has no OI, so its features rows are not input_eligible_incl_holding (§5.4).
    paths = build_lake(tmp_path / "lake", monkeypatch, no_oi_for=("000020",))
    _, _, feats = digest.prepare_inputs(paths, P.connect_inputs(paths))
    dev_set = set(seal.DEV_DATES)
    dev = feats[[seal._to_date(d) in dev_set for d in feats["decision_date"]]]
    flag = dev["input_eligible_incl_holding"].fillna(False).astype(bool)
    excluded = dev[~flag]
    assert set(excluded["ticker"]) == {"000020"} and set(dev[flag]["ticker"]) == {"000010"}
    expected_n = int(flag.sum())  # DEV rows with input_eligible_incl_holding, counted from features
    assert 0 < expected_n < len(dev)

    seen = []
    real = digest.reasons.build_reason_dictionary

    def spy(presence, *a, **k):
        seen.append(presence[["decision_date", "ticker"]].copy())
        return real(presence, *a, **k)

    monkeypatch.setattr(digest.reasons, "build_reason_dictionary", spy)
    res = digest.run_f4(paths, P.connect_inputs(paths), run_at=RUN_AT)
    assert res["digests"]["DEV.oi_y1"]["n_rows"] == expected_n
    pres = pd.concat(seen)
    pres = pres[[seal._to_date(d) in dev_set for d in pres["decision_date"]]]
    assert set(pres["ticker"]) == {"000010"}


def test_f4_only_json_scalars_and_hash_grant_blocks_confirm(tmp_path, monkeypatch):
    paths = build_lake(tmp_path / "lake", monkeypatch)
    res = digest.run_f4(paths, P.connect_inputs(paths), run_at=RUN_AT)
    json.dumps(res)
    assert not any(k.startswith("CONFIRM") for k in res["digests"])
    with pytest.raises(seal.SealViolation):
        seal.digest_only(
            lambda g: __import__(
                "modeler.scores.earnings_a.labels", fromlist=["x"]
            ).load_label_rows(paths, decision_dates=list(seal.CONFIRM_DATES), grant=g),
            exact_cols=["ticker"],
        )


def test_f3_end_to_end(tmp_path, monkeypatch, capsys):
    pytest.importorskip("modeler.scores.earnings_a.checks")
    build_lake(tmp_path / "lake", monkeypatch)
    rc, out = _run_cli(["f3"], capsys)
    assert rc == 0 and len(out) == 2
    m = json.loads(open(out[1]).read())
    assert m["kind"] == "F3_label_free"
    assert "r2" not in m["inputs"]
    assert _scan(m) == []


def test_manifest_helpers():
    assert all(len(v) == 64 for v in manifest.package_file_sha256().values())
    assert "digest.py" in manifest.package_file_sha256()
    env = manifest.env_versions()
    assert {"python", "numpy", "pandas", "polars", "duckdb", "scikit-learn", "platform"} <= set(env)
    g = manifest.git_state(manifest.repo_root())
    assert set(g) == {"head", "branch", "dirty", "describe"} and isinstance(g["dirty"], bool)
