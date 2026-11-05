"""근거·그래프 표시(explain-D.json v1)와 그림 파일(assets/*.svg)의 렌더·비교·검증 시험.

계획: my/milestones/common/20261026_briefing_charts_and_reasons/README.md §5.2.1·§5.4·§5.5.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import explain_fixtures as ef
import md_fixtures as mf
import pytest

from modeler.reporting import markdown as md
from modeler.reporting import svg as reporting_svg

RELEASE = "r20261005"
CARDS = Path(__file__).resolve().parents[3] / "deploy" / "prod" / "model-cards.json"
SVG = "assets/kr-top20.svg"
STAMP = "2026-10-07T10:03:12+09:00"
INVOCATION = "inv-test"
ASSETS = {
    "assets/kr-top20.svg",
    "assets/us-lightgbm-top20.svg",
    "assets/us-ridge-top20.svg",
}
# 근거 기능을 넣기 전 코드(modeler 0ebe1c8)로 같은 fixture를 렌더한 단위 해시입니다.
# f1·f3은 R-7 문구(briefing-r7-kr-halt 4d7ac26)를 합친 뒤 다시 구했습니다 — 바뀐 곳은 kr-stocks.md 상태 줄 하나입니다.
# 근거 입력이 없으면(absent·disabled) 출력이 이것과 바이트까지 같아야 합니다(계획 C7·§5.4).
BEFORE_EXPLAIN = {
    "f1_normal": "3b53f45cbd94265a0a390331889fcd529802f50d9ecbcfb818e7792c1d7fbb58",
    "f2_kr_unavailable_us_stale": (
        "0ef37ca6d6f47a57c9098168d236b95318896ae3499d629d614b818a3c3b7eae"
    ),
    "f3_replay": "a3dc506be03fe395fd2b75573021b1ccfcfe5df5ef7340e5dc9e724b7b43f05c",
    "f4_all_failed": "47abc436c539e2fa0a81030313c59c4b1c60a7c22edab118945fa027228d8f8d",
    "f5_lag_over_limit": "3fdbd7a5d75ff0ebc136f7d122415772f6432968bf385c1091526a4ea5508cbe",
}


def _no_log(_message: str) -> None:
    return None


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    paths = mf.write_all(tmp_path_factory.mktemp("fx"))
    return {
        name: (env.read_bytes(), ms.read_bytes() if ms else None)
        for name, (env, ms) in paths.items()
    }


def _ctx(tmp_path: Path, env: bytes, ms: bytes | None, explain: bytes | None = None, **kw):
    return md.build_context(
        tmp_path / "no-repo",
        env,
        ms,
        RELEASE,
        kw.pop("stamp", STAMP),
        100,
        _no_log,
        explain=explain,
        invocation_id=kw.pop("invocation_id", INVOCATION),
        **kw,
    )


def _render(tmp_path: Path, inputs: dict, doc: dict | None, name: str = "f1_normal", **kw):
    env, ms = inputs[name]
    raw = ef.dump(doc) if doc is not None else None
    ctx = _ctx(tmp_path, env, ms, raw, **kw)
    return ctx, md.render_unit(ctx)


def _doc(inputs: dict, name: str = "f1_normal") -> dict:
    return ef.explain_doc(inputs[name][0], INVOCATION)


# --- 근거 입력이 없으면 지금과 바이트까지 같습니다 ------------------------------------------
@pytest.mark.parametrize("name", sorted(BEFORE_EXPLAIN))
def test_absent_explain_is_byte_identical_to_before(
    tmp_path: Path, inputs: dict, name: str
) -> None:
    env, ms = inputs[name]
    plain = md.render_unit(
        md.build_context(tmp_path / "no-repo", env, ms, RELEASE, STAMP, 100, _no_log)
    )
    assert md.unit_sha256(plain) == BEFORE_EXPLAIN[name]
    assert all(isinstance(v, str) for v in plain.values()) and len(plain) == 5
    # 근거 입력 자리를 None으로 넘겨도(absent·disabled) 같습니다.
    _, absent = _render(tmp_path, inputs, None, name)
    assert absent == plain


# --- 정상 근거 ------------------------------------------------------------------------
def test_ok_explain_adds_columns_details_and_three_svgs(tmp_path: Path, inputs: dict) -> None:
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    assert set(files) == set(md.SECTION_FILES.values()) | ASSETS
    assert all(isinstance(files[a], bytes) for a in ASSETS)
    assert md.validate_unit_files(ctx["unit"], files) == []
    kr = files["kr-stocks.md"]
    assert "- 근거·그래프: 정상 (그래프 " in kr
    assert "| 60세션 흐름 | 5세션 | 20세션 | 60세션 | 점수를 올린 요인 | 품질 |" in kr
    assert "![KR 상위 20 최근 60세션 지수](assets/kr-top20.svg)" in kr
    assert kr.count("<details>") == 1 and kr.count("#### ") == 20
    assert "상위 100 평균 Δ" in kr and "하위 2%" in kr and "상위 1%" in kr
    assert "같은 시장(코스닥) 안 하위 2%" in kr
    us = files["us-stocks.md"]
    assert "](assets/us-lightgbm-top20.svg)" in us and "](assets/us-ridge-top20.svg)" in us
    assert us.count("<details>") == 2
    readme = files["README.md"]
    assert "| 60세션 흐름 | 점수를 올린 요인 |" in readme
    meta, _ = md.parse_front_matter(kr)
    assert meta["source"]["explain_sha256"] == md.sha256_bytes(ef.dump(_doc(inputs)))
    # 정지 구간은 선을 끊고 글자 그림에 · 로 보입니다. 수익률 사유는 표 아래에 있습니다.
    assert "- 3위 `100003` 20세션: 정지 중 시작" in kr
    assert "표시 주의: 거래정지 5일" in kr
    for name in md.SECTION_FILES.values():
        assert len(files[name].encode()) < md.MAX_FILE_BYTES
        assert "~" not in files[name].replace("\\~", "")
    for name in ASSETS:
        assert len(files[name]) < md.SVG_MAX_BYTES


def test_svg_bytes_are_deterministic_and_change_the_unit_hash(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    _, a = _render(tmp_path, inputs, doc)
    _, b = _render(tmp_path, inputs, doc, stamp="2026-10-07T10:40:00+09:00")
    assert all(a[n] == b[n] for n in ASSETS)
    assert md.same_unit_content(a, b)
    changed = ef.variant(doc)
    changed["models"][mf.KR_ID]["rows"][0]["path"][-1] = 150.0  # 그림만 바뀌게 합니다
    _, c = _render(tmp_path, inputs, changed)
    assert c["assets/kr-top20.svg"] != a["assets/kr-top20.svg"]
    assert md.unit_sha256(c) != md.unit_sha256(a)
    assert not md.same_unit_content(a, c)
    # 그림 바이트 하나만 달라도 같은 단위가 아닙니다(front matter 정규화를 하지 않습니다).
    tweaked = dict(a)
    tweaked["assets/kr-top20.svg"] = a["assets/kr-top20.svg"].replace(b"<title>", b"<title> ", 1)
    assert not md.same_unit_content(a, tweaked)
    assert md.unit_sha256(tweaked) != md.unit_sha256(a)


def test_svg_module_is_plain_and_breaks_lines_at_gaps() -> None:
    panels = [
        {
            "label": "#1 <A&B>",
            "sub": "x",
            "stock": [100.0, 101.0, None, 99.0, 98.0, 97.0],
            "bench": [100.0, 100.5, 101.0, 100.0, 99.5, 99.0],
            "breaks": [4],
        },
        {"label": "#2", "sub": "", "stock": [], "bench": []},
    ]
    one = reporting_svg.small_multiples("제목", "부제", panels)
    assert one == reporting_svg.small_multiples("제목", "부제", panels)
    text = one.decode()
    assert text.startswith("<svg ") and "&lt;A&amp;B&gt;" in text
    # 종목 선은 None과 의심일(4)에서 끊깁니다: 선 두 토막 + 점 하나, 비교 선 하나.
    assert text.count("<polyline") == 3 and text.count("<circle") == 1
    assert "가격 경로 없음" in text and 'stroke="#dc2626"' in text
    assert md.svg_violations("x.svg", one) == []


# --- 실패·부분 --------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["report_sha256", "invocation_id", "unit"])
def test_mismatched_link_renders_failed_line_and_keeps_todays_table(
    tmp_path: Path, inputs: dict, field: str
) -> None:
    doc = _doc(inputs)
    doc[field] = {"report_sha256": "f" * 64, "invocation_id": "older", "unit": "2026-10-06"}[field]
    ctx, files = _render(tmp_path, inputs, doc)
    assert set(files) == set(md.SECTION_FILES.values())
    _, plain = _render(tmp_path, inputs, None)
    kr, base = files["kr-stocks.md"], plain["kr-stocks.md"]
    line = "- 근거·그래프: 실패 — 입력 불일치. 표는 근거 없이 냈습니다."
    assert line in kr and kr.count(line) == 1
    assert files["us-stocks.md"].count(line) == 2
    # 실패 줄을 빼면 지금 리포트와 같습니다. 입력 불일치면 front matter에 근거 hash도 없습니다.
    assert kr.replace(line + "\n", "") == base
    assert "explain_sha256" not in kr
    assert "KR 근거·그래프를 계산하지 못했습니다: 입력 불일치." in files["README.md"]
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_missing_invocation_means_mismatch(tmp_path: Path, inputs: dict) -> None:
    _, files = _render(tmp_path, inputs, _doc(inputs), invocation_id=None)
    assert "실패 — 입력 불일치" in files["kr-stocks.md"]


@pytest.mark.parametrize(
    "raw,reason",
    [(b"", "근거 파일 형식 오류"), (b"{", "근거 파일 형식 오류"), (b"[]", "근거 파일 형식 오류")],
)
def test_unreadable_explain_is_failed_not_fatal(
    tmp_path: Path, inputs: dict, raw: bytes, reason: str
) -> None:
    env, ms = inputs["f1_normal"]
    files = md.render_unit(_ctx(tmp_path, env, ms, raw))
    assert f"실패 — {reason}" in files["kr-stocks.md"]


def test_explain_failed_status_and_reason(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    doc["status"], doc["reason"] = "failed", "timeout"
    _, files = _render(tmp_path, inputs, doc)
    assert "- 근거·그래프: 실패 — 시간 초과." in files["kr-stocks.md"]
    meta, _ = md.parse_front_matter(files["kr-stocks.md"])
    assert "explain_sha256" in meta["source"]  # 짝은 맞는 파일이라 hash를 남깁니다


def test_partial_model_and_rows(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    del doc["models"][mf.RDG_ID]
    rows = doc["models"][mf.KR_ID]["rows"]
    rows[1].update(explain_status="failed", explain_reason="score_mismatch")
    rows[4].update(price_status="failed", price_reason="price_unavailable")
    del rows[9]
    ctx, files = _render(tmp_path, inputs, doc)
    kr, us = files["kr-stocks.md"], files["us-stocks.md"]
    assert "- 근거·그래프: 일부만 계산" in kr
    assert "- 2위 `100002` 근거: 점수 재현 실패" in kr
    assert "- 5위 `100005` 가격: 가격 입력 없음" in kr
    assert "- 10위 `100010` 근거: 근거 행 없음" in kr and "- 10위 `100010` 가격: 근거 행 없음" in kr
    assert "근거를 내지 못했습니다: 점수 재현 실패" in kr
    assert "- 근거·그래프: 실패 — 이 모델의 근거가 없습니다." in us
    assert "assets/us-ridge-top20.svg" not in files and "assets/us-lightgbm-top20.svg" in files
    assert "](assets/us-ridge-top20.svg)" not in us
    assert md.validate_unit_files(ctx["unit"], files) == []


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (
            lambda m: m["dates"].__setitem__(-1, "2099-01-01"),
            "입력 불일치(그래프 날짜가 기준일 뒤)",
        ),
        (lambda m: m.__setitem__("asof", "2026-01-01"), "입력 불일치(기준일)"),
        (lambda m: m["rows"][0].__setitem__("symbol", "OTHER"), "입력 불일치(순위)"),
        (lambda m: m.__setitem__("groups", []), "근거 파일 형식 오류"),
    ],
)
def test_model_level_guards(tmp_path: Path, inputs: dict, mutate, reason: str) -> None:
    doc = _doc(inputs)
    mutate(doc["models"][mf.KR_ID])
    _, files = _render(tmp_path, inputs, doc)
    assert f"- 근거·그래프: 실패 — {reason}" in files["kr-stocks.md"]
    assert "assets/kr-top20.svg" not in files


def test_up_groups_rounding_and_order(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    row = doc["models"][mf.KR_ID]["rows"][0]
    row["delta"] = {g: -0.01 for g, _ in ef.KR_GROUPS}
    row["delta"].update(price=0.00004, risk=0.02, liq=0.03, flow=0.00006)
    _, files = _render(tmp_path, inputs, doc)
    first = next(x for x in files["kr-stocks.md"].split("\n") if x.startswith("| 1 |"))
    assert "| 거래·유동성·규모 · 변동성·위험 |" in first
    detail = files["kr-stocks.md"].split("#### 1위")[1].split("####")[0]
    assert "| 가격 흐름 | 0.0000 |" in detail and "| 수급 | +0.0001 |" in detail
    row["delta"] = {g: (0.00004 if g == "price" else -0.01) for g, _ in ef.KR_GROUPS}
    _, files = _render(tmp_path, inputs, doc)
    first = next(x for x in files["kr-stocks.md"].split("\n") if x.startswith("| 1 |"))
    assert first.endswith("| 없음 | 이상 없음 |")


def test_sparkline_scale_and_gaps() -> None:
    assert md.sparkline(None) == md.BLANK
    assert md.sparkline([100.0] * 61) == "`" + "▄" * 10 + "`"
    rising = [100.0 + t for t in range(61)]
    assert md.sparkline(rising) == "`▁▁▂▃▄▅▆▇██`"
    gap = rising[:]
    gap[7:13] = [None] * 6
    assert md.sparkline(gap)[2] == md.SPARK_GAP


# --- 같은 내용 비교와 기존 단위 읽기 ------------------------------------------------------
def test_generate_rerun_with_svg_is_unchanged_and_correction_drops_old_assets(
    tmp_path: Path, inputs: dict
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env, ms = inputs["f1_normal"]
    paths = {"env": tmp_path / "env.json", "ms": tmp_path / "ms.json", "ex": tmp_path / "ex.json"}
    paths["env"].write_bytes(env)
    paths["ms"].write_bytes(ms)
    paths["ex"].write_bytes(ef.dump(_doc(inputs)))

    def run(stamp: str, **kw) -> str:
        return md.generate(
            repo,
            paths["env"],
            paths["ms"],
            RELEASE,
            stamp,
            100,
            kw.pop("reason", None),
            None,
            False,
            _no_log,
            explain_path=kw.pop("explain", paths["ex"]),
            invocation_id=INVOCATION,
        )

    assert run(STAMP) == "created"
    udir = md.unit_dir(repo, "2026-10-07")
    assert sorted(p.name for p in (udir / "assets").iterdir()) == sorted(
        a.split("/")[1] for a in ASSETS
    )
    existing = md.read_existing(udir)
    assert set(existing) == set(md.SECTION_FILES.values()) | ASSETS
    assert run("2026-10-07T10:40:00+09:00") == "unchanged"
    # 근거 없이 다시 내면(정정) 옛 그림을 지웁니다.
    assert run("2026-10-07T11:00:00+09:00", explain=None, reason="근거 끔") == "corrected"
    assert not (udir / "assets").exists()
    assert md.validate(repo, paths=[f"{md.unit_rel('2026-10-07')}/README.md"]) == []


def test_read_existing_marks_symlinks_and_strays(tmp_path: Path) -> None:
    udir = tmp_path / "unit"
    (udir / "assets").mkdir(parents=True)
    (udir / "README.md").write_text("x", encoding="utf-8")
    (udir / "assets" / "a.svg").write_bytes(b"<svg/>")
    (udir / "assets" / "b.svg").symlink_to(udir / "assets" / "a.svg")
    (udir / "assets" / "c.png").write_bytes(b"x")
    found = md.read_existing(udir)
    assert found["assets/a.svg"] == b"<svg/>"
    assert found["assets/b.svg"] == md.INVALID_ASSET and found["assets/c.png"] == md.INVALID_ASSET


# --- 검증기 -----------------------------------------------------------------------------
def _ok_files(tmp_path: Path, inputs: dict) -> tuple[str, dict]:
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    return ctx["unit"], files


def test_unit_validator_rejects_bad_images_and_svgs(tmp_path: Path, inputs: dict) -> None:
    unit, files = _ok_files(tmp_path, inputs)

    def problems(changes: dict) -> list:
        broken = dict(files)
        for name, value in changes.items():
            if value is None:
                broken.pop(name)
            else:
                broken[name] = value
        return md.validate_unit_files(unit, broken)

    kr = files["kr-stocks.md"]
    link = "](assets/kr-top20.svg)"
    assert any("그림 대상이 없습니다" in p for p in problems({SVG: None}))
    for wrong in ("](../assets/kr-top20.svg)", "](assets/../../x.svg)", "](kr-top20.svg)"):
        found = problems({"kr-stocks.md": kr.replace(link, wrong)})
        assert any("같은 단위의 assets/" in p for p in found), wrong
    assert any(
        "상대 경로만" in p
        for p in problems({"kr-stocks.md": kr.replace(link, "](https://x/a.svg)")})
    )
    assert any(
        "절대 경로 그림" in p for p in problems({"kr-stocks.md": kr.replace(link, "](/a.svg)")})
    )
    svg = files["assets/kr-top20.svg"]
    big = svg.replace(b"</svg>", b"<!--" + b"a" * md.SVG_MAX_BYTES + b"--></svg>")
    assert any("200KB" in p for p in problems({SVG: big}))
    for bad in (
        b"<script>alert(1)</script>",
        b'<rect onload="x()"/>',
        b'<a href="https://example.invalid">x</a>',
        b"<foreignObject></foreignObject>",
        b'<image x="0"/>',
        b"/home/whi/secret",
    ):
        found = problems({SVG: svg.replace(b"</svg>", bad + b"</svg>")})
        assert found, bad
    assert any("SVG 파일이 아닙니다" in p for p in problems({SVG: b"<html/>"}))
    assert any("UTF-8" in p for p in problems({SVG: b"<svg>\xff</svg>"}))
    # 그림은 markdown 규칙(front matter)으로 보지 않습니다.
    assert not any("front matter" in p for p in md.validate_unit_files(unit, files))
    stray = dict(files)
    stray["assets/x.png"] = b"x"
    assert any("둘 수 없는 파일" in p for p in md.validate_unit_files(unit, stray))


def test_tree_validator_checks_images_svgs_and_symlinks(tmp_path: Path, inputs: dict) -> None:
    unit, files = _ok_files(tmp_path, inputs)
    repo = tmp_path / "repo"
    md.init_repo(repo, CARDS, _no_log)
    udir = md.unit_dir(repo, unit)
    for name, value in files.items():
        (udir / name).parent.mkdir(parents=True, exist_ok=True)
        (udir / name).write_bytes(md.as_bytes(value))
    rels = [f"{md.unit_rel(unit)}/{name}" for name in files]
    assert md.validate(repo, paths=rels) == []
    kr_rel = f"{md.unit_rel(unit)}/kr-stocks.md"
    svg_rel = f"{md.unit_rel(unit)}/assets/kr-top20.svg"
    (udir / "assets" / "kr-top20.svg").unlink()
    assert any("그림 대상이 없습니다" in p for p in md.validate(repo, paths=[kr_rel]))
    os.symlink(udir / "assets" / "us-ridge-top20.svg", udir / "assets" / "kr-top20.svg")
    assert any("symlink" in p for p in md.validate(repo, paths=[svg_rel]))
    (udir / "assets" / "kr-top20.svg").unlink()
    (udir / "assets" / "kr-top20.svg").write_bytes(b"<svg><script/></svg>")
    assert any("스크립트" in p for p in md.validate(repo, paths=[svg_rel]))
    # assets/ 자체가 symlink면 거부합니다.
    real = tmp_path / "elsewhere"
    (udir / "assets").rename(real)
    os.symlink(real, udir / "assets")
    found = md.validate(repo, paths=[f"{md.unit_rel(unit)}/assets", svg_rel])
    assert any("symlink" in p for p in found)


def test_change_set_allows_asset_deletion_only_for_corrected_units(tmp_path: Path) -> None:
    import sys

    reports_dir = Path(__file__).resolve().parents[3] / "deploy" / "reports"
    if str(reports_dir) not in sys.path:
        sys.path.insert(0, str(reports_dir))
    import validate_reports as vr

    rel = f"{md.unit_rel('2026-10-07')}/assets/kr-top20.svg"
    change = [("D", rel, "")]
    assert vr.check_change_set(tmp_path, change, new_units=["2026-10-07"]) != []
    assert vr.check_change_set(tmp_path, change, new_units=[], corrected_units=["2026-10-07"]) == []
    md_rel = f"{md.unit_rel('2026-10-07')}/kr-stocks.md"
    assert (
        vr.check_change_set(
            tmp_path, [("D", md_rel, "")], new_units=[], corrected_units=["2026-10-07"]
        )
        != []
    )
    assert vr.own_models_of({"assets/kr-top20.svg": b"<svg/>", "x.md": "no fm"}) == set()


def test_cli_accepts_explain_and_invocation(tmp_path: Path, inputs: dict) -> None:
    repo = tmp_path / "repo"
    assert (
        md.main(["--repo", str(repo), "--init", "--no-validate", "--model-cards", str(CARDS)]) == 0
    )
    env, ms = inputs["f1_normal"]
    (tmp_path / "env.json").write_bytes(env)
    (tmp_path / "ex.json").write_bytes(ef.dump(_doc(inputs)))
    code = md.main(
        [
            "--repo",
            str(repo),
            "--release",
            RELEASE,
            "--generated-at",
            STAMP,
            "--envelope",
            str(tmp_path / "env.json"),
            "--explain",
            str(tmp_path / "ex.json"),
            "--invocation-id",
            INVOCATION,
        ]
    )
    assert code == 0
    assert (md.unit_dir(repo, "2026-10-07") / "assets" / "kr-top20.svg").is_file()
    assert json.loads((tmp_path / "ex.json").read_text())["schema"] == md.EXPLAIN_SCHEMA


# --- serving 쪽 v1 사실(46bc08e)에 맞춘 경우 ----------------------------------------------
def test_price_failure_for_a_whole_model_keeps_the_reasons(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    model = doc["models"][mf.KR_ID]
    model["dates"] = []
    for row in model["rows"]:
        row.update(price_status="failed", price_reason="가격 입력 없음", path=[], bench_path=[])
    ctx, files = _render(tmp_path, inputs, doc)
    kr = files["kr-stocks.md"]
    assert "- 근거·그래프: 일부만 계산 (가격 경로 없음)." in kr
    assert "| 60세션 흐름 | 5세션 | 20세션 | 60세션 | 점수를 올린 요인 | 품질 |" in kr
    first = next(x for x in kr.split("\n") if x.startswith("| 1 |"))
    assert "| — | — | — | — |" in first and not first.endswith("| — | 이상 없음 |")
    assert "assets/kr-top20.svg" not in files
    assert "그릴 가격 경로가 없어 그래프를 만들지 않았습니다." in kr
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_skipped_status(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    doc["status"] = "skipped"
    _, files = _render(tmp_path, inputs, doc)
    assert md.unit_sha256(files) == BEFORE_EXPLAIN["f1_normal"]
    doc = _doc(inputs)
    doc["models"][mf.KR_ID]["status"] = "skipped"
    _, files = _render(tmp_path, inputs, doc)
    assert "- 근거·그래프: 실패 — 근거를 계산하지 않았습니다." in files["kr-stocks.md"]


def test_partial_price_reason_is_shown_as_written(tmp_path: Path, inputs: dict) -> None:
    doc = _doc(inputs)
    doc["models"][mf.KR_ID]["rows"][1]["price_reason"] = "거래정지·누락 5일 · 수익률 비움(20)"
    _, files = _render(tmp_path, inputs, doc)
    assert (
        "표시 주의: 거래정지·누락 5일 · 수익률 비움(20) · 관측 56/61세션" in files["kr-stocks.md"]
    )
