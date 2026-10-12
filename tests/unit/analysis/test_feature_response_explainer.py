"""Pure-function tests of the feature explainer (synthetic inputs, no stock_data needed)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from modeler.analysis.feature_response.explainer import cli, render
from modeler.analysis.feature_response.explainer.common import (
    clean_json,
    pick_example,
    quantiles,
    strip_md,
    supp_shape,
    table_cells,
    unescape_md,
)
from modeler.analysis.feature_response.explainer.config import (
    CONFIG_NAMES,
    ConfigError,
    load_config,
    resolve_config,
    validate_config,
    validate_principle,
)
from modeler.analysis.feature_response.explainer.data_kr import parse_cards_md, parse_result_md
from modeler.analysis.feature_response.explainer.data_us import parse_us_result_md
from modeler.analysis.feature_response.explainer.paths import (
    LabelAccessError,
    Roots,
    guard_not_label,
    latest_match,
    read_parquet,
    resolve_roots,
)


def test_markdown_cleanup():
    assert strip_md("`px_ret_1d` 부호 \\~ 반대 ") == "px_ret_1d 부호 ~ 반대"
    assert unescape_md("`SHORT` +1.2 \\~ 3") == "`SHORT` +1.2 ~ 3"
    cells = table_cells("| 1 | `mcap_rank` | 0.0058 | 0.0057\\~0.0058 |")
    assert cells == ["1", "mcap_rank", "0.0058", "0.0057~0.0058"]


def test_supp_shape_counts_only_changed_steps():
    flat, up, verdict = supp_shape([0.4, 0.4, 0.41, 0.41, 0.42])
    assert (flat, up, verdict) == (2, 1.0, "증가")
    assert supp_shape([0.5, 0.5, 0.5])[2] == "변화 없음"
    assert supp_shape([0.4, 0.5, 0.4, 0.5])[2] == "단조 아님"
    assert supp_shape([0.5, 0.4, 0.3, 0.2, 0.1])[2] == "감소"


def test_clean_json_and_quantiles():
    assert clean_json({"a": [float("nan"), 1.0, float("inf")]}) == {"a": [None, 1.0, None]}
    assert quantiles(np.array([np.nan, 1.0, 2.0, 3.0]), [0.0, 0.5, 1.0]) == [1.0, 2.0, 3.0]
    assert quantiles(np.array([np.nan]), [0.5]) == [None]


def _cand(rows):
    return pl.DataFrame(
        rows, schema={"pct": pl.Float64, "size": pl.Float64, "t": pl.String}, orient="row"
    )


def test_pick_example_prefers_close_then_large():
    df = _cand([(0.951, 5.0, "a"), (0.949, 9.0, "b"), (0.90, 99.0, "c")])
    row, fallback = pick_example(df, 0.95)
    assert row["t"] == "b" and fallback is False
    # ties on pct (mcap_rank-like): all rows equal, pick the biggest
    tied = _cand([(0.5, 1.0, "x"), (0.5, 7.0, "y")])
    row, fallback = pick_example(tied, 0.95)
    assert row["t"] == "y" and fallback is True
    assert pick_example(_cand([]), 0.95) == (None, False)


def test_label_guard(tmp_path: Path):
    with pytest.raises(LabelAccessError):
        guard_not_label("/data/kr/datasets/x/label_daily.parquet")
    with pytest.raises(LabelAccessError):
        guard_not_label(Path("/data/us/datasets/us_labels_v2/part.parquet"))
    with pytest.raises(LabelAccessError):
        read_parquet(tmp_path / "us_labels_h5_v1" / "part.parquet")
    assert guard_not_label("/data/us/datasets/us_features_v2/part.parquet")


def test_roots_env_and_rel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sd, my = tmp_path / "sd", tmp_path / "my"
    sd.mkdir()
    my.mkdir()
    monkeypatch.setenv("STOCK_DATA_ROOT", str(sd))
    monkeypatch.setenv("MY_ROOT", str(my))
    roots = resolve_roots()
    assert roots.stock_data == sd.resolve() and roots.my == my.resolve()
    assert roots.rel(sd.resolve() / "kr" / "a.parquet") == "stock_data/kr/a.parquet"
    assert roots.rel(my.resolve() / "x.html") == "my/x.html"
    explicit = resolve_roots(stock_data=sd, my=my)
    assert explicit == roots


def test_latest_match_takes_newest_snapshot(tmp_path: Path):
    for d in ("snapshot_date=2026-09-18", "snapshot_date=2026-10-06"):
        (tmp_path / "t" / d).mkdir(parents=True)
        (tmp_path / "t" / d / "part.parquet").write_bytes(b"")
    hit = latest_match(tmp_path, "t/snapshot_date=*/part.parquet")
    assert "2026-10-06" in str(hit)
    with pytest.raises(FileNotFoundError):
        latest_match(tmp_path, "nope/*.parquet")


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_bundled_configs_load_and_validate(name: str):
    cfg = load_config(name)
    assert validate_config(cfg) == []
    feats = [f for _, _, fs in cfg.THEMES for f in fs]
    assert feats and all(f in cfg.CARDS and f in cfg.FMT for f in feats)
    assert cfg.OUT_NAME.endswith(".html") and cfg.DOC_MD.endswith(".md")
    for f in feats:
        win = cfg.CARDS[f]["win"]
        assert {"axis", "formula", "rows", "marks"} <= set(win)


def test_resolve_config_uses_roots_only(tmp_path: Path):
    roots = Roots(stock_data=tmp_path / "sd", my=tmp_path / "my")
    cfg = resolve_config(load_config("kr_flow"), roots)
    assert str(cfg.OUT_HTML).startswith(str(roots.my))
    assert str(cfg.RUN_DIR).startswith(str(roots.stock_data))
    assert cfg.RUN_DIR.name == cfg.RUN_ID
    out = resolve_config(load_config("kr_flow"), roots, out=tmp_path / "x.html")
    assert out.OUT_HTML == tmp_path / "x.html"
    with pytest.raises(FileNotFoundError):  # US needs the newest snapshot below stock_data
        resolve_config(load_config("us_flow_holders"), roots)


def test_validate_config_reports_missing_fields():
    bad = validate_config(SimpleNamespace(MARKET="kr"))
    assert any("missing field THEMES" in b for b in bad)
    with pytest.raises(ConfigError):
        resolve_config(SimpleNamespace(MARKET="kr"), Roots(Path("a"), Path("b")))
    with pytest.raises(ConfigError):
        load_config("no_such_config")


def test_validate_principle():
    flow = {"type": "flow", "steps": [{"title": "a"}, {"title": "b", "hl": True}]}
    assert validate_principle("f", flow) == []
    assert validate_principle("f", {"type": "flow", "steps": [{"title": "a"}]})
    tl = {"type": "timeline", "events": [{"at": 0, "label": "x"}, {"at": 45, "label": "y"}]}
    assert validate_principle("f", tl) == []
    assert validate_principle("f", {"type": "timeline", "events": [{"at": 0}]})
    ratio = {"type": "ratio", "rows": [{"label": "r", "num": -30, "den": -10}]}
    assert validate_principle("f", ratio) == []
    assert validate_principle("f", {"type": "ratio", "rows": [{"num": 1, "den": 0}]})
    assert validate_principle("f", {"type": "bar"})


def test_parse_kr_tables():
    result = (
        "| 1 | `px_a` | 0.0642 | 0.06\\~0.07 | -0.96 (-0.97/-0.96) | 단조 감소 | Y | `−` "
        "| 기대와 같음 |\n"
        "| 3 | `px_b` | 0.0300 | x | y | z | N | 미등록 | 기대 없음 |\n"
        "| 1 | `px_a` | 0.0814 | 단조 감소 | 1/20 | 5% | 감소 |\n"
    )
    contrast, supp = parse_result_md(result)
    assert contrast["px_a"] == {"rank": 1, "expected": "−", "result": "기대와 같음"}
    assert supp["px_a"] == {"flat": 1, "up": "5%", "verdict": "감소"}
    cards = (
        "### `px_a` — 제목\n| 모델 반응 (층 3) | 섞기 ΔrankIC +0.05473 (단독 47개 중 1위, 안정) "
        "· SHAP 1위, 방향 ρ −0.96, 모양 단조 감소(보조 감소), 안정 · 기대와 같음. 진단\n"
    )
    summary, line = parse_cards_md(cards)
    assert summary == {} and line["px_a"]["perm_rank"] == 1
    assert line["px_a"]["dic"] == pytest.approx(0.05473) and line["px_a"]["supp"] == "감소"


def test_parse_us_result_table():
    md = (
        "## 1. 읽는 법\n### 1.1 보조 판정\n| 1 | `mcap_rank` | 기타 | 7/20 | 0% | 감소 |\n"
        "## 2. LightGBM\n| 1 | `mcap_rank` | 0.0058 | 0.0057\\~0.0058 | -0.90 | 기타 "
        "| Y (상위 10 두 절반) | − | 기대와 같음 |\n"
        "## 3. Ridge\n| 1 | `mcap_rank` | 0.0118 | 0.01\\~0.02 | -0.99 | 선형 "
        "| Y (상위 10 두 절반) | − | 기대와 같음 | -0.068 | +0.005 |\n## 4. 묶음\n"
    )
    contr, supp = parse_us_result_md(md)
    assert contr["lightgbm"]["mcap_rank"]["rank"] == 1
    assert contr["ridge"]["mcap_rank"]["result"] == "기대와 같음"
    assert supp["mcap_rank"] == {"flat": 7, "up": "0%", "verdict": "감소"}


def _tiny_cfg() -> SimpleNamespace:
    win = {"axis": 8, "formula": "f", "rows": [], "marks": [], "note": ""}
    card = {
        "title": "t",
        "basis": "valid",
        "intuition": "i",
        "win": win,
        "what": "w",
        "model": "m",
        "caution": "c",
    }
    return SimpleNamespace(
        MARKET="kr",
        PAGE_TITLE="시험",
        THEMES=[("소주제", "설명", ["px_a"])],
        CARDS={"px_a": card},
        FMT={"px_a": "pct"},
        BASIS={"valid": "유효"},
        N_ICE=3,
        MODELER_COMMIT="abc1234",
        RUN_DIR=Path("run"),
        PANEL=Path("panel.parquet"),
        RAW=Path("raw"),
        MD_CARDS=Path("01_x.md"),
        MD_RESULT=Path("09_kr_result.md"),
        HTML_OVERRIDE={"FOOT": "<li>맨 위 문단</li>"},
    )


def test_render_keeps_common_footer_when_foot_is_overridden():
    data = {"market": "kr", "meta": {"run_id": "r1", "ref_date": "2026-08-28"}, "features": []}
    roots = Roots(Path("/sd"), Path("/my"))
    html = render.render_html(_tiny_cfg(), data, "kr_x", roots, dt.date(2026, 10, 12))
    assert "<li>맨 위 문단</li>" in html
    assert "<b>입력 파일</b>" in html and "r1" in html
    assert "explainer build kr_x" in html and "2026-10-12" in html
    assert "__FOOT__" not in html and "scratchpad" not in html
    assert render.js({"a": "</script>"}) == '{"a":"<\\/script>"}'


def test_cli_list_and_cache_dir(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["list"]) == 0
    assert capsys.readouterr().out.split() == list(CONFIG_NAMES)
    cache = Path(cli.tmp_cache())
    assert cache.is_dir()
    cache.rmdir()
    roots = Roots(Path("/sd"), Path("/my"))
    assert cli.default_cache_dir(roots, "us") == Path(
        "/sd/us/output/feature_response_l3/explainer_cache"
    )
    assert cli._problems(["EX x", "MISMATCH a", "no md row b"]) == ["MISMATCH a", "no md row b"]
