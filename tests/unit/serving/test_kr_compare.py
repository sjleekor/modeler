from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.serving import kr_compare as cmp
from modeler.serving.kr_model import LoadedBundle

DAY = date(2026, 9, 29)


def _daily(**overrides) -> pl.DataFrame:
    frame = {
        "trade_date": [DAY] * 3,
        "ticker": ["000001", "000002", "000003"],
        "market": ["KOSPI"] * 3,
        "px_ret_1d": [0.10, 0.20, 0.30],
        "flag": [True, False, True],
    }
    frame.update(overrides)
    return pl.DataFrame(frame, schema_overrides={"trade_date": pl.Date})


def _mart(root: Path, name: str, frame: pl.DataFrame) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part-0.parquet")


def _run(tmp_path: Path, a: dict[str, pl.DataFrame], b: dict[str, pl.DataFrame], *extra: str):
    for side, marts in (("a", a), ("b", b)):
        for name, frame in marts.items():
            _mart(tmp_path / side, name, frame)
    report_path = tmp_path / "report.json"
    code = cmp.main([
        "marts", "--a-root", str(tmp_path / "a"), "--b-root", str(tmp_path / "b"),
        "--report", str(report_path), "--temp-dir", str(tmp_path / "tmp"), *extra,
    ])
    return code, json.loads(report_path.read_text(encoding="utf-8"))


def test_identical_marts_exit_zero(tmp_path: Path, capsys) -> None:
    code, report = _run(tmp_path, {"feat_price": _daily()}, {"feat_price": _daily()})
    assert code == 0 and report["equal"] is True
    result = report["marts"]["feat_price"]
    assert result["row_count"] == {"a": 3, "b": 3}
    assert result["key"] == ["trade_date", "ticker", "market"]
    assert result["except_all"] == {"a_minus_b": 0, "b_minus_a": 0}
    assert "RESULT equal" in capsys.readouterr().out
    assert list((tmp_path / "tmp").iterdir()) == []  # the spill directory is removed


def test_changed_value_is_found_and_named(tmp_path: Path) -> None:
    changed = _daily(px_ret_1d=[0.10, 0.2000001, 0.30])
    code, report = _run(tmp_path, {"feat_price": _daily()}, {"feat_price": changed})
    result = report["marts"]["feat_price"]
    assert code == 1 and result["equal"] is False
    assert result["except_all"] == {"a_minus_b": 1, "b_minus_a": 1}
    assert set(result["keyed"]["columns"]) == {"px_ret_1d"}
    assert result["keyed"]["columns"]["px_ret_1d"]["over_tolerance"] == 1
    assert result["samples"]["a_minus_b"][0]["ticker"] == "000002"


def test_float_tolerance_uses_absolute_and_relative_thresholds(tmp_path: Path) -> None:
    a = _daily(px_ret_1d=[1000.0, 0.0, 0.30])
    b = _daily(px_ret_1d=[1000.4, 0.0000004, 0.30])
    # 0.4 on 1000 passes only through the relative term, 4e-7 near 0 only through the absolute term.
    code, report = _run(tmp_path, {"m": a}, {"m": b}, "--abs-tol", "1e-6", "--rel-tol", "1e-3")
    assert code == 0 and report["marts"]["m"]["except_all"]["a_minus_b"] == 2
    assert report["marts"]["m"]["keyed"]["clean"] is True
    code, report = _run(tmp_path, {"m": a}, {"m": b}, "--abs-tol", "1e-6", "--rel-tol", "1e-5")
    assert code == 1
    assert report["marts"]["m"]["keyed"]["columns"]["px_ret_1d"]["over_tolerance"] == 1
    code, _ = _run(tmp_path, {"m": a}, {"m": b})
    assert code == 1


def test_null_moved_to_another_row_is_a_difference_even_within_tolerance(tmp_path: Path) -> None:
    a = _daily(px_ret_1d=[None, 0.20, 0.30])
    b = _daily(px_ret_1d=[0.10, None, 0.30])
    code, report = _run(tmp_path, {"m": a}, {"m": b}, "--abs-tol", "1000", "--rel-tol", "1")
    columns = report["marts"]["m"]["keyed"]["columns"]
    assert code == 1 and columns["px_ret_1d"]["null_mismatch"] == 2
    assert columns["px_ret_1d"]["over_tolerance"] == 0


def test_nan_and_inf_positions_must_match(tmp_path: Path) -> None:
    nan, inf = float("nan"), float("inf")
    a = _daily(px_ret_1d=[nan, inf, 0.30])
    same = _daily(px_ret_1d=[nan, inf, 0.30])
    code, _ = _run(tmp_path, {"m": a}, {"m": same})
    assert code == 0
    moved = _daily(px_ret_1d=[inf, nan, 0.30])
    code, report = _run(tmp_path, {"m": a}, {"m": moved}, "--abs-tol", "1", "--rel-tol", "1")
    stats = report["marts"]["m"]["keyed"]["columns"]["px_ret_1d"]
    assert code == 1 and stats["nan_mismatch"] == 2 and stats["inf_mismatch"] == 2
    sign = _daily(px_ret_1d=[nan, -inf, 0.30])
    code, report = _run(tmp_path, {"m": a}, {"m": sign}, "--abs-tol", "1", "--rel-tol", "1")
    assert code == 1
    assert report["marts"]["m"]["keyed"]["columns"]["px_ret_1d"]["inf_mismatch"] == 1


def test_duplicate_key_and_row_count_are_reported(tmp_path: Path) -> None:
    duplicated = pl.concat([_daily(), _daily().head(1)])
    code, report = _run(tmp_path, {"m": _daily()}, {"m": duplicated})
    result = report["marts"]["m"]
    assert code == 1
    assert result["duplicates"]["a"] == {"duplicate_keys": 0, "excess_rows": 0}
    assert result["duplicates"]["b"] == {"duplicate_keys": 1, "excess_rows": 1}
    assert result["row_count"] == {"a": 3, "b": 4}
    assert result["except_all"] == {"a_minus_b": 0, "b_minus_a": 1}
    # Identical duplicates on both sides are the same data: reported, not a difference.
    code, report = _run(tmp_path, {"m": duplicated}, {"m": duplicated})
    assert code == 0 and report["marts"]["m"]["duplicates"]["a"]["duplicate_keys"] == 1


def test_schema_type_and_order_differences(tmp_path: Path) -> None:
    reordered = _daily().select(["trade_date", "ticker", "market", "flag", "px_ret_1d"])
    code, report = _run(tmp_path, {"m": _daily()}, {"m": reordered})
    assert code == 1 and report["marts"]["m"]["schema"]["order_differs"] is True
    assert report["marts"]["m"]["except_all"] == {"a_minus_b": 0, "b_minus_a": 0}
    retyped = _daily().with_columns(pl.col("px_ret_1d").cast(pl.Float32))
    code, report = _run(tmp_path, {"m": _daily()}, {"m": retyped})
    mismatch = report["marts"]["m"]["schema"]["type_mismatch"]
    assert code == 1 and mismatch == [{"column": "px_ret_1d", "a": "DOUBLE", "b": "FLOAT"}]


def test_non_daily_mart_uses_the_key_table_and_override(tmp_path: Path) -> None:
    facts = pl.DataFrame({
        "ticker": ["1", "1", "2"], "metric_code": ["rev", "rev", "rev"],
        "bsns_year": [2025, 2026, 2026], "reprt_code": ["11011"] * 3,
        "value_numeric": [1.0, 2.0, 3.0],
    })
    code, report = _run(tmp_path, {"stock_metric_fact": facts}, {"stock_metric_fact": facts})
    assert code == 0
    assert report["marts"]["stock_metric_fact"]["key"] == [
        "ticker", "metric_code", "bsns_year", "reprt_code"]
    odd = pl.DataFrame({"code": ["a", "a"], "v": [1.0, 1.0]})
    code, report = _run(tmp_path, {"odd": odd}, {"odd": odd})
    assert code == 1 and "key columns are missing" in report["marts"]["odd"]["problems"][0]
    code, report = _run(tmp_path, {"odd": odd}, {"odd": odd}, "--key", "odd=code")
    assert code == 0 and report["marts"]["odd"]["duplicates"]["a"]["duplicate_keys"] == 1


def test_mart_missing_on_one_side_and_usage_errors(tmp_path: Path, capsys) -> None:
    code, report = _run(tmp_path, {"m": _daily(), "extra": _daily()}, {"m": _daily()})
    assert code == 1 and report["marts"]["extra"]["problems"] == ["missing in b"]
    assert cmp.main(["marts", "--a-root", str(tmp_path / "none"), "--b-root", str(tmp_path / "b"),
                     "--report", str(tmp_path / "r.json")]) == 2
    assert cmp.main(["prepared", "--a-dir", str(tmp_path), "--b-dir", str(tmp_path),
                     "--report", str(tmp_path / "r.json")]) == 2
    assert "error:" in capsys.readouterr().err
    assert cmp.main(["marts", "--a-root", str(tmp_path / "a"), "--b-root", str(tmp_path / "b"),
                     "--report", str(tmp_path / "r.json"), "--abs-tol", "-1"]) == 2


class _RankModel:
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        score = matrix[:, 0]
        return np.column_stack((1.0 - score, score))


def _bundle() -> LoadedBundle:
    return LoadedBundle(
        directory=Path("/unused"),
        manifest={
            "manifest_sha256": "bundle-hash",
            "feature_contract": {
                "feature_columns": ["px_ret_1d"],
                "design_columns": ["px_ret_1d", "px_ret_1d_isna"],
            },
        },
        model=_RankModel(), golden={},
    )


def _prepared(root: Path, name: str, frame: pl.DataFrame, **manifest) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    frame.write_parquet(directory / "feature_panel.parquet")
    body = {"snapshot_date": "2026-09-29", "generated_at": f"volatile-{name}", **manifest}
    (directory / "prepare_manifest.json").write_text(json.dumps(body), encoding="utf-8")
    return directory


def _compare_prepared(tmp_path: Path, a: pl.DataFrame, b: pl.DataFrame, *, bundle: bool = True,
                      a_manifest: dict | None = None, b_manifest: dict | None = None):
    a_dir = _prepared(tmp_path, "a", a, **(a_manifest or {}))
    b_dir = _prepared(tmp_path, "b", b, **(b_manifest or {}))
    report_path = tmp_path / "prepared.json"
    argv = ["prepared", "--a-dir", str(a_dir), "--b-dir", str(b_dir), "--report", str(report_path)]
    if bundle:
        argv += ["--bundle", str(tmp_path / "bundle")]
    return cmp.main(argv), json.loads(report_path.read_text(encoding="utf-8"))


@pytest.fixture
def fake_bundle(monkeypatch) -> None:
    monkeypatch.setattr("modeler.serving.kr_model.load_bundle", lambda *_a, **_k: _bundle())


def test_prepared_identical_scores_bitwise_equal(tmp_path: Path, fake_bundle) -> None:
    code, report = _compare_prepared(tmp_path, _daily(), _daily())
    assert code == 0 and report["equal"] is True
    scoring = report["scoring"]
    assert scoring["matrix"]["bitwise_equal"] and scoring["p_raw"]["bitwise_equal"]
    assert scoring["ranks"]["equal"] and scoring["top100"]["set_equal"]
    assert report["manifest_differences"] == []  # the volatile generated_at is ignored


def test_prepared_value_change_flips_rank_and_matrix(tmp_path: Path, fake_bundle) -> None:
    swapped = _daily(px_ret_1d=[0.30, 0.20, 0.10])
    code, report = _compare_prepared(tmp_path, _daily(), swapped)
    scoring = report["scoring"]
    assert code == 1
    assert report["panel"]["equal"] is False
    assert scoring["matrix"]["differing_rows"] == 2
    assert scoring["matrix"]["differing_columns"] == ["px_ret_1d"]
    assert scoring["p_raw"]["bitwise_equal"] is False and scoring["p_raw"]["max_abs_diff"] > 0
    assert scoring["ranks"]["max_rank_shift"] == 2
    assert scoring["top100"]["set_equal"] is True and scoring["top100"]["order_equal"] is False


def test_prepared_universe_difference_and_ties(tmp_path: Path, fake_bundle) -> None:
    code, report = _compare_prepared(tmp_path, _daily(), _daily().head(2))
    assert code == 1
    assert report["panel"]["keyed"]["only_in_a"] == 1
    assert report["scoring"]["rows_only_in_a"] == 1
    tied = _daily(px_ret_1d=[0.5, 0.5, 0.1])
    (tmp_path / "t").mkdir()
    code, report = _compare_prepared(tmp_path / "t", tied, tied)
    assert code == 0 and report["scoring"]["ranks"]["tied_rows"] == {"a": 2, "b": 2}


def test_prepared_without_bundle_skips_scoring_and_lists_manifest_changes(tmp_path: Path) -> None:
    code, report = _compare_prepared(
        tmp_path, _daily(), _daily(), bundle=False,
        a_manifest={"mart_contracts": {"feat_price": "h1"}},
        b_manifest={"mart_contracts": {"feat_price": "h2"}})
    assert code == 0
    assert "skipped" in report["scoring"]
    assert report["manifest_differences"] == [
        {"path": "mart_contracts.feat_price", "a": "h1", "b": "h2"}]
