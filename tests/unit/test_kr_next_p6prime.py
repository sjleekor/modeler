"""P6′ wrapper: tag guards (T12), the view-folder helpers and the label-free checks.

Everything runs on temporary git repositories, fake panels and fake marts; no lake is read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from modeler.etl.marts import financial_quarters_v4p as fq4p
from modeler.models._03_kr_next import build_panel as p6_bp
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import forward as p6_forward
from modeler.models._03_kr_next import train as p6_train
from modeler.models._03_kr_next import views as p6_views
from modeler.models._04_kr_next_p6prime import build_panel as bp
from modeler.models._04_kr_next_p6prime import checks, inputs, tags
from modeler.models._04_kr_next_p6prime import train as tr

from .test_financial_quarters_v4p import _FACT, _RCPT, _con, _december_company, _march_to_december
from .test_kr_next_train import make_panel

# ----------------------------------------------------------------------------- helpers


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / "doc.md").write_text(text)
    _git(repo, "add", "doc.md")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", text)
    return _git(repo, "rev-parse", "HEAD")


def _repo(path: Path, *, p6: bool = True, p6prime: bool = False) -> Path:
    """A prereg repository: the P6 tag on the first commit, the P6′ tag on a second one."""
    path.mkdir()
    _git(path, "init", "-q")
    _commit(path, "P6 prereg")
    if p6:
        _git(path, "-c", "tag.gpgsign=false", "tag", tags.P6_PREREG_TAG)
    _commit(path, "P6' prereg")
    if p6prime:
        _git(path, "-c", "tag.gpgsign=false", "tag", tags.PREREG_TAG)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _labeled_panel(folder: Path, *, prereg_tag: str, tamper: bool = False) -> Path:
    """A fake labeled panel (random numbers, no real label) with a builder-style manifest."""
    folder.mkdir(parents=True)
    panel = make_panel(
        cols=list(fs.NOFLOW_COLS),
        label_name="label_end_date_c1_20d",
        extra_flag=True,
        n_dates=230,
        n_tickers=16,
    )
    path, man = p6_bp.output_paths(folder, "labeled")
    panel.write_parquet(path)
    sha = "0" * 64 if tamper else _sha(path)
    man.write_text(json.dumps({"mode": "labeled", "prereg_tag": prereg_tag, "panel_sha256": sha}))
    return path


def _bp_args(tmp: Path, repo: Path, *extra: str, mode: str = "labeled") -> list[str]:
    return [
        "--mode", mode, "--lake", str(tmp / "lake"), "--snapshot", "2026-10-23",
        "--end", "2026-10-22", "--out-dir", str(tmp / "out"), "--prereg-repo", str(repo), *extra,
    ]  # fmt: skip


# ----------------------------------------------------------------------------- tags (T12)


def test_require_prereg_tag_does_not_accept_the_p6_tag(tmp_path) -> None:
    repo = _repo(tmp_path / "my", p6=True, p6prime=False)
    with pytest.raises(tags.FrozenTagMissing, match=tags.PREREG_TAG):
        tags.require_prereg_tag(repo)
    _git(repo, "-c", "tag.gpgsign=false", "tag", f"{tags.PREREG_TAG}-x")  # a near miss
    with pytest.raises(tags.FrozenTagMissing):
        tags.require_prereg_tag(repo)
    _git(repo, "-c", "tag.gpgsign=false", "tag", tags.PREREG_TAG)
    assert tags.require_prereg_tag(repo) == _git(repo, "rev-parse", "HEAD")
    assert tags.require_prereg_tag(repo) != _git(repo, "rev-parse", tags.P6_PREREG_TAG)


def test_labeled_panel_is_refused_with_only_the_p6_tag(tmp_path, capsys) -> None:
    repo = _repo(tmp_path / "my", p6=True, p6prime=False)
    assert bp.main(_bp_args(tmp_path, repo)) == 2
    assert tags.PREREG_TAG in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".*duckdb_tmp"))  # no connection was even prepared
    # the gap this closes: the P6 builder lets the same repository through its tag guard and only
    # stops later, at the (here missing) lake
    assert p6_bp.main(_bp_args(tmp_path, repo)) == 2
    err = capsys.readouterr().err
    assert "git tag" not in err and "no parquet files" in err


def test_labeled_panel_with_the_p6prime_tag_needs_a_p6prime_view_folder(tmp_path, capsys) -> None:
    repo = _repo(tmp_path / "my", p6=False, p6prime=True)
    views = tmp_path / "views"
    views.mkdir()
    f = views / "dim_stock_pit_daily.parquet"
    f.write_bytes(b"x")
    manifest = {"views": {"dim_stock_pit_daily": {"file": f.name, "sha256": _sha(f)}}}
    (views / "manifest.json").write_text(json.dumps(manifest))  # a P6-style manifest
    assert bp.main(_bp_args(tmp_path, repo, "--views-dir", str(views))) == 2
    assert "not a P6′ view folder" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_panel_output_inside_the_p6_dataset_folder_is_refused(tmp_path, capsys) -> None:
    repo = _repo(tmp_path / "my", p6prime=True)
    args = _bp_args(tmp_path, repo, mode="serving")
    args[args.index("--out-dir") + 1] = str(tmp_path / "lake" / "datasets" / "kr_next" / "x")
    assert bp.main(args) == 2
    assert "P6 dataset folder" in capsys.readouterr().err


def test_training_is_refused_with_only_the_p6_tag(tmp_path, capsys) -> None:
    repo = _repo(tmp_path / "my", p6=True, p6prime=False)
    panel = _labeled_panel(tmp_path / "panel", prereg_tag=tags.PREREG_TAG)
    out = tmp_path / "bundle"
    args = ["--lake", str(tmp_path), "--snapshot", "s", "--out", str(out),
            "--prereg-repo", str(repo), "--panel", str(panel)]  # fmt: skip
    assert tr.main(args) == 2
    assert tags.PREREG_TAG in capsys.readouterr().err
    assert not out.exists()
    # the P6 CLI's own guard is satisfied by the same repository (the gap this closes)
    p6_train.require_frozen_tag(repo)


@pytest.mark.parametrize(
    "case", ["no_panel", "p6_panel", "tampered", "not_a_builder_output", "existing_out"]
)
def test_training_needs_a_p6prime_labeled_panel(tmp_path, capsys, case) -> None:
    repo = _repo(tmp_path / "my", p6prime=True)
    out = tmp_path / "bundle"
    tag = tags.P6_PREREG_TAG if case == "p6_panel" else tags.PREREG_TAG
    panel = _labeled_panel(tmp_path / "panel", prereg_tag=tag, tamper=case == "tampered")
    if case == "not_a_builder_output":
        panel = panel.rename(panel.with_name("other.parquet"))
    if case == "existing_out":
        out.mkdir()
    args = ["--lake", str(tmp_path), "--snapshot", "s", "--out", str(out),
            "--prereg-repo", str(repo)]  # fmt: skip
    if case != "no_panel":
        args += ["--panel", str(panel)]
    assert tr.main(args) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert (
        (out.exists() and not list(out.iterdir())) if case == "existing_out" else not out.exists()
    )


def test_training_end_to_end_records_the_p6prime_tag_and_restores_p6(tmp_path) -> None:
    repo = _repo(tmp_path / "my", p6=True, p6prime=True)
    panel = _labeled_panel(tmp_path / "panel", prereg_tag=tags.PREREG_TAG)
    out = tmp_path / "bundle"
    before = (p6_train.PREREG_TAG, p6_train.require_frozen_tag)
    rc = tr.main(
        ["--lake", str(tmp_path), "--snapshot", "snap-synth", "--out", str(out),
         "--prereg-repo", str(repo), "--panel", str(panel)]
    )  # fmt: skip
    assert rc == 0
    m = bundle_mod.load_bundle(out).manifest
    assert m["prereg_tag"] == tags.PREREG_TAG
    assert m["prereg_commit"] == _git(repo, "rev-parse", f"refs/tags/{tags.PREREG_TAG}^{{commit}}")
    assert m["prereg_commit"] != _git(repo, "rev-parse", tags.P6_PREREG_TAG)
    assert m["trials"] == {"hgb_grid_points": 16, "ridge": 1}  # the P6 procedure, unchanged
    assert (p6_train.PREREG_TAG, p6_train.require_frozen_tag) == before
    assert p6_train.PREREG_TAG == "kr-next-frozen"


def test_swaps_restore_on_error_and_are_not_reentrant() -> None:
    original = p6_train.require_frozen_tag
    with pytest.raises(ValueError, match="boom"), tags.p6_train_with_p6prime_tag():
        assert p6_train.PREREG_TAG == tags.PREREG_TAG
        raise ValueError("boom")
    assert p6_train.require_frozen_tag is original and p6_train.PREREG_TAG == tags.P6_PREREG_TAG
    with tags.p6_train_with_p6prime_tag():
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with tags.p6_forward_with_p6prime_tag():
                pass
    assert p6_forward.BUNDLE_TAG_NAME == tags.P6_BUNDLE_TAG


def test_the_swapped_names_are_all_the_tag_references_of_the_p6_code() -> None:
    """If the frozen P6 code read a tag some other way, the swap would miss it."""
    assert tags.tag_references(p6_train.main) == {
        "names": {"require_frozen_tag", "FrozenTagMissing", "PREREG_TAG"},
        "literals": set(),
    }
    refs = tags.tag_references(p6_forward.bundle_tag_time)
    assert refs["names"] - {"tag"} == {"BUNDLE_TAG_NAME"} and refs["literals"] == set()


def test_forward_swap_accepts_only_the_p6prime_bundle_tag() -> None:
    def man(bundle_tag: str) -> dict:
        return {
            "prereg_tag": "x",
            "bundle_tag": bundle_tag,
            "bundle_tag_time": "2026-11-20T22:00:00+09:00",
        }

    with pytest.raises(p6_forward.ForwardOnlyError):
        p6_forward.bundle_tag_time(man(tags.BUNDLE_TAG))
    with tags.p6_forward_with_p6prime_tag():
        assert p6_forward.bundle_tag_time(man(tags.BUNDLE_TAG)).year == 2026
        with pytest.raises(p6_forward.ForwardOnlyError):
            p6_forward.bundle_tag_time(man(tags.P6_BUNDLE_TAG))
    assert p6_forward.bundle_tag_time(man(tags.P6_BUNDLE_TAG)).year == 2026


# ----------------------------------------------------------------------------- inputs


def test_session_calendar_reaches_ten_sessions_past_the_anchor() -> None:
    holidays = {date(2026, 10, 26), date(2027, 12, 31)}
    days = inputs.session_calendar(date(2026, 10, 1), date(2026, 10, 22), holidays)
    assert days[0] == date(2026, 10, 1)
    after = [d for d in days if d > date(2026, 10, 22)]
    assert len(after) == 10 and date(2026, 10, 26) not in days
    assert all(d.weekday() < 5 for d in days)
    with pytest.raises(inputs.InputsRefused):
        inputs.session_calendar(date(2026, 10, 1), date(2026, 12, 28), {date(2026, 1, 1)})


def test_content_digest_ignores_row_order_but_not_values(tmp_path) -> None:
    con = duckdb.connect()
    a, b, c = (tmp_path / f"{n}.parquet" for n in "abc")
    rows = "(VALUES (1, 'x', NULL::DOUBLE), (2, NULL, 2.5), (3, 'z', 3.0)) t(k, s, v)"
    con.execute(f"COPY (SELECT * FROM {rows} ORDER BY k) TO '{a}' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM {rows} ORDER BY k DESC) TO '{b}' (FORMAT PARQUET)")
    con.execute(
        f"COPY (SELECT k, s, CASE WHEN k = 3 THEN 3.5 ELSE v END v FROM {rows}) "
        f"TO '{c}' (FORMAT PARQUET)"
    )
    assert _sha(a) != _sha(b)
    assert inputs.content_digest(con, a) == inputs.content_digest(con, b)
    assert inputs.content_digest(con, a) != inputs.content_digest(con, c)


def _fake_p6prime_folder(folder: Path, *, block: bool = True) -> Path:
    folder.mkdir()
    views = {}
    for name in inputs.VIEW_FILES:
        f = folder / f"{name}.parquet"
        f.write_bytes(name.encode())
        views[name] = {"file": f.name, "sha256": _sha(f), "content_digest": "d" + name}
    man = {"views": views, "calendar": {"sessions_sha256": "c"}}
    if block:
        man["p6prime"] = {
            "fin_file_holds": fq4p.FQMV_DLR_V4P_TABLE,
            "fact_cache_metadata": {"sql_hash": inputs.PINS["fact_sql_hash"]},
            "fin_cache_metadata": {"sql_hash": "other"},
        }
    (folder / inputs.MANIFEST_NAME).write_text(json.dumps(man))
    return folder


def test_p6prime_folder_is_recognised_and_rehashed(tmp_path) -> None:
    good = _fake_p6prime_folder(tmp_path / "good")
    assert inputs.load_p6prime_manifest(good)["p6prime"]["fin_file_holds"].endswith("_v4p")
    with pytest.raises(inputs.InputsRefused, match="not a P6′"):
        inputs.load_p6prime_manifest(_fake_p6prime_folder(tmp_path / "p6", block=False))
    (good / f"{inputs.FIN_FILE}.parquet").write_bytes(b"changed")
    with pytest.raises(inputs.InputsRefused, match="differs"):
        inputs.load_p6prime_manifest(good)
    with pytest.raises(inputs.InputsRefused, match="already exists"):
        inputs.check_outputs_free(good)


def test_compare_pins_reports_each_pin(tmp_path) -> None:
    man = json.loads(
        (_fake_p6prime_folder(tmp_path / "f") / inputs.MANIFEST_NAME).read_text("utf-8")
    )
    pins = inputs.compare_pins(man)
    assert set(pins) == set(inputs.PINS) | set(inputs.RESEARCH_FILE_SHA256)
    assert pins["fact_sql_hash"]["same"] is True
    assert pins["fin_sql_hash"] == {
        "expected": inputs.PINS["fin_sql_hash"],
        "actual": "other",
        "same": False,
    }


def test_view_folder_names_match_what_the_p6_builder_reads() -> None:
    assert set(p6_bp.DERIVED_VIEW_FILES) <= set(inputs.VIEW_FILES)
    assert inputs.FIN_FILE == "fin_quarterly_metric_vintage_dlr"
    assert set(inputs.COPIED_FROM_P6) == {p6_views.PIT_VIEW, p6_views.QUALITY_TABLE}
    assert inputs.FACT_TABLE == fq4p.DEFAULT_VINTAGE_VIEW == "stock_metric_vintage_fact_dl_v3"


# ----------------------------------------------------------------------------- checks


def _mart(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE v4p AS "
        + fq4p.build_fin_quarterly_metric_vintage_dlr_v4p_sql(vintage_view=_FACT)
    )


def test_pit_cut_passes_on_a_point_in_time_mart() -> None:
    con = _con()
    _december_company(con)
    _march_to_december(con)
    _mart(con)
    res = checks.pit_cut(
        con,
        fact_view=_FACT,
        receipt_view=_RCPT,
        mart_view="v4p",
        cut_dates=("2020-12-31", "2021-12-31", "2022-06-30"),
    )
    assert res["pass"], res
    d = {r["t"]: r for r in res["per_date"]}
    assert all(r["placement_differs"] == 0 and r["identity_differs"] == 0 for r in d.values())
    assert d["2021-12-31"]["cut_rows"] > 0


def test_pit_cut_flags_a_mart_that_knows_the_future() -> None:
    con = _con()
    _december_company(con)
    _mart(con)
    # a "full" mart whose 2020 Q1 value differs from what was filed by the cut date
    con.execute(
        "UPDATE v4p SET standalone_value = standalone_value + 1 "
        "WHERE statement_period_end = DATE '2020-03-31'"
    )
    res = checks.pit_cut(
        con, fact_view=_FACT, receipt_view=_RCPT, mart_view="v4p", cut_dates=("2020-12-31",)
    )
    assert not res["pass"] and res["per_date"][0]["value_differs_unexplained"] > 0


def test_outlier_r1_and_non_december_checks_on_a_small_mart() -> None:
    con = _con()
    _december_company(con, conflict_q2=True)
    _march_to_december(con)
    _mart(con)
    con.execute("CREATE TABLE p6 AS SELECT * FROM v4p")
    assert checks.outlier_rows(con, p6_dlr_view="p6", mart_view="v4p")["pass"]
    r1 = checks.r1_conflicts(con, mart_view="v4p")
    assert r1["pass"] and r1["by_quarter"] == [{"quarter": "Q2", "n": 1, "n_null": 0}]
    nd = checks.non_december(con, p6_dlr_view="p6", mart_view="v4p")
    assert nd["pass"] and nd["tickers"] == 1


def _panel(path: Path, rows: list[dict]) -> Path:
    pl.DataFrame(rows).write_parquet(path)
    return path


def _panel_rows(fin_value: float | None, px: float = 1.0) -> list[dict]:
    base = {c: 0.5 for c in fs.NOFLOW_COLS}
    rows = []
    for i, t in enumerate(("A", "B", "C", "D")):
        r = {"trade_date": date(2020, 1, 2), "ticker": t, "market": "KOSPI", **base}
        r["px_ret_1d_v2"] = px if i == 0 else 0.0
        r["fin_roa_v2"] = fin_value if i == 0 else 0.1
        r["fin_ttm_conflict_v2"] = False
        r["liq_notice_flag"] = False
        rows.append(r)
    return rows


def test_compare_serving_panels(tmp_path) -> None:
    con = duckdb.connect()
    a = _panel(tmp_path / "a.parquet", _panel_rows(0.1))
    b = _panel(tmp_path / "b.parquet", _panel_rows(None))
    res = checks.compare_serving_panels(con, a, b)
    assert res["pass"] and res["keys_only_in_p6"] == 0
    roa = res["fin_columns"]["fin_roa_v2"]
    assert roa["changed_rows"] == 1 and roa["changed_share"] == 0.25 and roa["value_to_null"] == 1
    assert "mcap_krx_log" in res["non_fin_columns_compared"]
    c = _panel(tmp_path / "c.parquet", _panel_rows(0.1, px=2.0))
    bad = checks.compare_serving_panels(con, a, c)
    assert not bad["pass"] and bad["non_fin_rows_differ_beyond_tolerance"]["px_ret_1d_v2"] == 1
    # float noise of a parallel sum: not bit-equal, but inside the golden tolerance
    d = _panel(tmp_path / "d.parquet", _panel_rows(0.1, px=1.0 + 1e-15))
    noise = checks.compare_serving_panels(con, a, d)
    assert noise["pass"] and not noise["pass_exact"]
    assert noise["non_fin_rows_differ"]["px_ret_1d_v2"] == 1
    assert noise["non_fin_max_abs_diff"]["px_ret_1d_v2"] < 1e-14


def test_missing_gap(tmp_path) -> None:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE stock_master AS SELECT * FROM (VALUES ('A', 'DELISTED'), ('B', 'ACTIVE'), "
        "('C', 'ACTIVE'), ('C', 'DELISTED'), ('D', 'ACTIVE')) t(ticker, status)"
    )
    rows = _panel_rows(None)  # A (later delisted) misses fin_roa_v2, the rest have it
    p = _panel(tmp_path / "p.parquet", rows)
    res = checks.missing_gap(con, p, "stock_master", start="2020-01-01", end="2020-12-31")
    assert res["later_delisted_tickers"] == 1 and res["rows"] == 4
    assert res["gap_pp"]["fin_roa_v2"] == pytest.approx(100.0)
    assert res["columns_over"] == ["fin_roa_v2"] and not res["pass"]
    assert res["gap_pp"]["px_ret_1d_v2"] == 0


def test_panel_checks_refuse_a_labeled_panel(tmp_path) -> None:
    rows = _panel_rows(0.1)
    for r in rows:
        r["y"] = 0.5
    p = _panel(tmp_path / "labeled.parquet", rows)
    with pytest.raises(ValueError, match="serving panels only"):
        checks.run_panel_checks(p, p, tmp_path, temp_dir=tmp_path / "tmp")


def test_december_attribution_on_a_small_mart() -> None:
    con = _con()
    _december_company(con)
    _mart(con)
    con.execute(
        "CREATE TABLE p6 AS SELECT * EXCLUDE (standalone_null_reason, fiscal_year_end) FROM v4p"
    )
    res = checks.december_attribution(
        con, p6_dlr_view="p6", fact_view=_FACT, receipt_view=_RCPT, mart_view="v4p"
    )
    assert res["december_tickers"] == 1
    assert res["totals"] == {"same": 8} and res["pass"]


def test_t1_cut_dates_are_the_draft_dates() -> None:
    assert checks.PIT_CUT_DATES == (
        "2016-06-30",
        "2018-03-31",
        "2020-12-31",
        "2022-08-31",
        "2024-05-31",
    )
