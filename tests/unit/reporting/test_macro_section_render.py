"""거시 흐름(macro-D.json → macro.md·assets/macro-*.svg)을 단위에 싣는 렌더 쪽 시험.

계획: my/milestones/common/20261031_briefing_macro_context/README.md §5.5·§9 M2-c.
계산 모듈과 렌더러는 다른 브랜치에서 만들므로 `macro_fakes`의 가짜로 고정 인터페이스만 씁니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import macro_fakes as mk
import md_fixtures as mf
import pytest
from reports_world import vr
from test_explain_render import BEFORE_EXPLAIN

from modeler.reporting import markdown as md

RELEASE = "r20261005"
STAMP = "2026-10-07T10:03:12+09:00"
INVOCATION = "inv-test"
UNIT = "2026-10-07"
SVG = "assets/macro-rates.svg"
FIVE = set(md.SECTION_FILES.values())
LINK = "거시 흐름은 [macro.md](macro.md)"
ROW = "| [거시 흐름](macro.md) |"


def _no_log(_message: str) -> None:
    return None


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    paths = mf.write_all(tmp_path_factory.mktemp("fx"))
    return {
        name: (env.read_bytes(), ms.read_bytes() if ms else None)
        for name, (env, ms) in paths.items()
    }


@pytest.fixture()
def fakes(monkeypatch):
    return mk.install(monkeypatch)


def _render(tmp_path: Path, inputs: dict, body: dict | bytes | None, name: str = "f1_normal", **kw):
    env, ms = inputs[name]
    raw = mk.dump(body) if isinstance(body, dict) else body
    ctx = md.build_context(
        tmp_path / "no-repo",
        env,
        ms,
        RELEASE,
        kw.pop("stamp", STAMP),
        100,
        _no_log,
        invocation_id=kw.pop("invocation_id", INVOCATION),
        macro=raw,
        **kw,
    )
    return ctx, md.render_unit(ctx)


def _unit(inputs: dict, name: str = "f1_normal") -> str:
    return json.loads(inputs[name][0])["report_date"]


def _doc(inputs: dict, name: str = "f1_normal", **kw) -> dict:
    return mk.doc(_unit(inputs, name), INVOCATION, **kw)


# --- (a) 입력이 없으면 지금과 바이트까지 같습니다 ------------------------------------------
@pytest.mark.parametrize("name", sorted(BEFORE_EXPLAIN))
def test_absent_macro_is_byte_identical_to_before(
    tmp_path: Path, inputs: dict, name: str, fakes
) -> None:
    _, files = _render(tmp_path, inputs, None, name)
    assert md.unit_sha256(files) == BEFORE_EXPLAIN[name]
    assert set(files) == FIVE
    assert LINK not in files["market-sector.md"] and ROW not in files["README.md"]


# --- (b) 정상 문서 ------------------------------------------------------------------------
def test_ok_macro_adds_page_svg_and_links_only(tmp_path: Path, inputs: dict, fakes) -> None:
    body = _doc(inputs)
    ctx, files = _render(tmp_path, inputs, body)
    _, plain = _render(tmp_path, inputs, None)
    assert set(files) == FIVE | {md.MACRO_FILE, SVG}
    assert isinstance(files[SVG], bytes) and files[SVG] == mk.SVG.encode()
    assert md.validate_unit_files(ctx["unit"], files) == []
    page = files[md.MACRO_FILE]
    meta, text = md.parse_front_matter(page)
    assert meta["section"] == "macro" and meta["status"] == "ok"
    assert meta["source"]["macro_sha256"] == md.sha256_bytes(mk.dump(body))
    assert meta["source"]["report_sha256"] == ctx["sha"] and meta["models"] == []
    assert "](assets/macro-rates.svg)" in text and "1\\~3개월" in text
    # 순위·데이터 상태 파일은 그대로이고, README·시장·섹터에는 링크만 더해집니다.
    for name in ("kr-stocks.md", "us-stocks.md", "data-status.md"):
        assert files[name] == plain[name]
    assert files["market-sector.md"].replace("\n" + LINK + "\n", "") == plain["market-sector.md"]
    readme = [line for line in files["README.md"].split("\n") if not line.startswith(ROW)]
    assert "\n".join(readme) == plain["README.md"]
    assert f"{ROW} 정상 | {UNIT} |" in files["README.md"]  # header.daily_asof_max
    assert files["market-sector.md"].rstrip("\n").endswith(LINK)


def test_link_sits_at_the_bottom_of_a_failed_market_sector_too(
    tmp_path: Path, inputs: dict, fakes
) -> None:
    name = "f4_all_failed"
    _, files = _render(tmp_path, inputs, _doc(inputs, name), name)
    assert files["market-sector.md"].rstrip("\n").endswith(LINK)
    assert md.validate_unit_files(_unit(inputs, name), files) == []


def test_svg_bytes_are_deterministic_and_the_rerun_is_the_same_unit(
    tmp_path: Path, inputs: dict, fakes
) -> None:
    body = _doc(inputs)
    _, a = _render(tmp_path, inputs, body)
    _, b = _render(tmp_path, inputs, body, stamp="2026-10-07T10:40:00+09:00")
    assert a[SVG] == b[SVG] and md.same_unit_content(a, b)
    tweaked = dict(a)
    tweaked[SVG] = a[SVG].replace(b"<title>", b"<title> ", 1)
    assert not md.same_unit_content(a, tweaked)


# --- (c) 실패 문서·입력 불일치 ------------------------------------------------------------
def test_failed_document_renders_the_failure_notice(tmp_path: Path, inputs: dict, fakes) -> None:
    ctx, files = _render(tmp_path, inputs, _doc(inputs, state="failed", reason="timeout"))
    assert set(files) == FIVE | {md.MACRO_FILE}
    meta, text = md.parse_front_matter(files[md.MACRO_FILE])
    assert meta["status"] == "failed" and "macro_sha256" in meta["source"]
    assert "거시 흐름을 내지 못했습니다: timeout." in text
    assert f"{ROW} 실패 | — |" in files["README.md"]  # a failure document has no header
    assert ctx["macro"]["state"] == "failed" and ctx["macro"]["accepted"] is True
    # 단위 상태는 거시 흐름과 무관합니다.
    _, plain = _render(tmp_path, inputs, None)
    assert md.parse_front_matter(files["README.md"])[0]["status"] == (
        md.parse_front_matter(plain["README.md"])[0]["status"]
    )
    assert md.validate_unit_files(ctx["unit"], files) == []


@pytest.mark.parametrize("field", ["invocation_id", "unit"])
def test_document_of_another_run_is_shown_as_a_mismatch(
    tmp_path: Path, inputs: dict, field: str, fakes
) -> None:
    body = _doc(inputs, **{field: {"invocation_id": "older", "unit": "2026-10-06"}[field]})
    ctx, files = _render(tmp_path, inputs, body)
    meta, text = md.parse_front_matter(files[md.MACRO_FILE])
    assert "input_mismatch" in text and meta["status"] == "failed"
    assert "macro_sha256" not in meta["source"] and SVG not in files
    assert ctx["macro"]["accepted"] is False
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_missing_invocation_means_mismatch(tmp_path: Path, inputs: dict, fakes) -> None:
    _, files = _render(tmp_path, inputs, _doc(inputs), invocation_id=None)
    assert "input_mismatch" in files[md.MACRO_FILE]


@pytest.mark.parametrize("raw", [b"", b"{", b"[]", b'{"schema": "other"}'])
def test_unreadable_document_is_a_failure_notice_not_a_blocked_unit(
    tmp_path: Path, inputs: dict, raw: bytes, fakes
) -> None:
    ctx, files = _render(tmp_path, inputs, raw)
    assert "format_error" in files[md.MACRO_FILE] and SVG not in files
    assert md.validate_unit_files(ctx["unit"], files) == []


# --- (f) 렌더 결과가 단위 규칙에 걸리면 실패 안내로 바꿉니다 ------------------------------
BAD_RENDERS = {
    "bad_name": lambda svg: ("![x](assets/rates.svg)\n", {"rates.svg": svg}),
    "nested_name": lambda svg: ("본문\n", {"macro-a/b.svg": svg}),
    "too_big": lambda svg: (
        "![x](assets/macro-big.svg)\n",
        {
            "macro-big.svg": svg.replace(
                "</svg>", "<!--" + "x" * (md.SVG_MAX_BYTES + 1) + "--></svg>"
            )
        },
    ),
    "script": lambda svg: (
        "본문\n",
        {"macro-s.svg": svg.replace("</svg>", "<script>1</script></svg>")},
    ),
    "external_href": lambda svg: (
        "본문\n",
        {"macro-h.svg": svg.replace("</svg>", '<a href="https://x.invalid">x</a></svg>')},
    ),
    "server_path_in_svg": lambda svg: (
        "본문\n",
        {"macro-p.svg": svg.replace("US 10Y", "/home/whi/data")},
    ),
    "missing_link_target": lambda svg: ("[없음](nowhere.md)\n", {}),
    "outside_image": lambda svg: ("![x](../2026-10-06/assets/macro-rates.svg)\n", {}),
    "not_text": lambda svg: (b"bytes", {}),
}


@pytest.mark.parametrize("case", sorted(BAD_RENDERS))
def test_render_breaking_unit_rules_becomes_a_failure_notice(
    tmp_path: Path, inputs: dict, fakes, monkeypatch, case: str
) -> None:
    _, render_module = fakes
    good = render_module.render

    def render(body):
        if body.get("state") == "failed":
            return good(body)
        return BAD_RENDERS[case](mk.SVG)

    monkeypatch.setattr(render_module, "render", render)
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    assert set(files) == FIVE | {md.MACRO_FILE}
    assert "render_rejected" in files[md.MACRO_FILE]
    assert ctx["macro"]["state"] == "failed"
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_render_crash_and_missing_modules_never_block_the_unit(
    tmp_path: Path, inputs: dict, fakes, monkeypatch
) -> None:
    _, render_module = fakes

    def boom(body):
        raise RuntimeError("renderer bug")

    monkeypatch.setattr(render_module, "render", boom)
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    assert files[md.MACRO_FILE].endswith(md.MACRO_FALLBACK)
    assert md.validate_unit_files(ctx["unit"], files) == []
    # 렌더러 모듈 자체가 없어도(합치기 전 release) 고정 문구로 냅니다.
    monkeypatch.setitem(__import__("sys").modules, mk.RENDER_MODULE, None)
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    assert files[md.MACRO_FILE].endswith(md.MACRO_FALLBACK)
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_server_path_in_the_page_is_scrubbed(
    tmp_path: Path, inputs: dict, fakes, monkeypatch
) -> None:
    _, render_module = fakes
    monkeypatch.setattr(render_module, "render", lambda body: ("출처 /home/whi/x.json\n", {}))
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    assert "/home/" not in files[md.MACRO_FILE] and "(경로 생략)" in files[md.MACRO_FILE]
    assert md.validate_unit_files(ctx["unit"], files) == []


# --- 단위 검증기는 거시 흐름 파일을 받되 그림 규칙을 같이 적용합니다 ----------------------
def test_unit_validator_accepts_macro_and_rejects_bad_svgs(
    tmp_path: Path, inputs: dict, fakes
) -> None:
    ctx, files = _render(tmp_path, inputs, _doc(inputs))
    unit = ctx["unit"]
    assert md.validate_unit_files(unit, files) == []
    big = dict(files)
    big[SVG] = files[SVG].replace(b"</svg>", b"<!--" + b"x" * md.SVG_MAX_BYTES + b"--></svg>")
    assert any("200KB" in v for v in md.validate_unit_files(unit, big))
    script = dict(files)
    script[SVG] = files[SVG].replace(b"</svg>", b"<script>1</script></svg>")
    assert any("스크립트" in v for v in md.validate_unit_files(unit, script))
    stray = dict(files)
    stray["macro.svg"] = files[SVG]
    assert any("둘 수 없는 파일" in v for v in md.validate_unit_files(unit, stray))
    wrong = dict(files)
    wrong[md.MACRO_FILE] = files[md.MACRO_FILE].replace("section: macro", "section: summary")
    assert any("section이 파일 이름(macro)" in v for v in md.validate_unit_files(unit, wrong))
    # macro.md는 선택 파일입니다: 없어도 단위는 완전합니다.
    without = {k: v for k, v in files.items() if k not in (md.MACRO_FILE, SVG)}
    without["README.md"] = without["README.md"].replace("[거시 흐름](macro.md)", "거시 흐름")
    without["market-sector.md"] = without["market-sector.md"].replace(LINK, "")
    assert md.validate_unit_files(unit, without) == []


# --- (e) 같은 단위 재실행·정정 -------------------------------------------------------------
def test_generate_rerun_is_unchanged_and_correction_without_macro_drops_the_page(
    tmp_path: Path, inputs: dict, fakes
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    md.init_repo(
        repo, Path(__file__).resolve().parents[3] / "deploy/prod/model-cards.json", _no_log
    )
    paths = mf.write_all(tmp_path / "fx")
    env_path, ms_path = paths["f1_normal"]
    macro_path = tmp_path / "macro.json"
    macro_path.write_bytes(mk.dump(_doc(inputs)))
    args = (repo, env_path, ms_path, RELEASE, STAMP, 100)
    kw = {"invocation_id": INVOCATION}
    assert md.generate(*args, None, None, False, _no_log, macro_path=macro_path, **kw) == "created"
    udir = md.unit_dir(repo, UNIT)
    assert (udir / md.MACRO_FILE).is_file() and (udir / SVG).is_file()
    assert md.validate(repo) == []
    again = md.generate(
        repo,
        env_path,
        ms_path,
        RELEASE,
        "2026-10-07T10:40:00+09:00",
        100,
        None,
        None,
        False,
        _no_log,
        macro_path=macro_path,
        **kw,
    )
    assert again == "unchanged"
    fixed = md.generate(*args, "거시 흐름 끔", "abc1234", False, _no_log, **kw)
    assert fixed == "corrected"
    assert not (udir / md.MACRO_FILE).exists() and not (udir / "assets").exists()
    assert LINK not in (udir / "market-sector.md").read_text()
    assert md.validate(repo) == []


def test_change_set_allows_macro_deletion_only_for_corrected_units(tmp_path: Path) -> None:
    rel = f"{md.unit_rel(UNIT)}/{md.MACRO_FILE}"
    changes = [("D", rel, "100644")]
    assert vr.check_change_set(tmp_path, changes, new_units=[], corrected_units=[UNIT]) == []
    blocked = vr.check_change_set(tmp_path, changes, new_units=[], corrected_units=[])
    assert blocked and "삭제는 허용하지 않습니다" in blocked[0]
    other = [("D", f"{md.unit_rel(UNIT)}/kr-stocks.md", "100644")]
    assert vr.check_change_set(tmp_path, other, new_units=[], corrected_units=[UNIT])


def test_cli_accepts_macro(tmp_path: Path, inputs: dict, fakes) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    cards = Path(__file__).resolve().parents[3] / "deploy/prod/model-cards.json"
    assert md.main(["--repo", str(repo), "--init", "--model-cards", str(cards)]) == 0
    paths = mf.write_all(tmp_path / "fx")
    env_path, ms_path = paths["f1_normal"]
    macro_path = tmp_path / "macro.json"
    macro_path.write_bytes(mk.dump(_doc(inputs)))
    code = md.main(
        [
            "--repo",
            str(repo),
            "--release",
            RELEASE,
            "--envelope",
            str(env_path),
            "--market-sector",
            str(ms_path),
            "--generated-at",
            STAMP,
            "--macro",
            str(macro_path),
            "--invocation-id",
            INVOCATION,
        ]
    )
    assert code == 0
    assert (md.unit_dir(repo, UNIT) / SVG).is_file()


@pytest.mark.parametrize(
    "header,cell",
    [
        ({"daily_asof_max": "2026-10-05"}, "2026-10-05"),
        ({"daily_asof_max": None}, "—"),
        ({"daily_asof_max": "not-a-date"}, "—"),
        (None, "—"),
    ],
)
def test_readme_row_shows_the_last_daily_asof(tmp_path: Path, inputs: dict, fakes, header, cell):
    body = _doc(inputs)
    if header is None:
        body.pop("header")
    else:
        body["header"] = header
    _, files = _render(tmp_path, inputs, body)
    assert f"{ROW} 정상 | {cell} |" in files["README.md"]
    assert md.validate_unit_files(UNIT, files) == []
