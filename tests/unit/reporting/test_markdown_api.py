"""publisher가 쓰는 markdown 모듈의 함수들: 단위 검증, 변경 파일만 검사, 모델 카드, 정정 렌더."""

from __future__ import annotations

import json
from pathlib import Path

import md_fixtures as mf

from modeler.reporting import markdown as md

PROJECT = Path(__file__).resolve().parents[3]
CARDS = PROJECT / "deploy" / "prod" / "model-cards.json"
RELEASE = "r20261005"


def _no_log(_message: str) -> None:
    return None


def _normal_inputs(tmp_path: Path) -> tuple[bytes, bytes]:
    paths = mf.write_all(tmp_path / "fx")
    env_path, ms_path = paths["f1_normal"]
    return env_path.read_bytes(), ms_path.read_bytes()


def _render(tmp_path: Path, repo: Path, **kw) -> tuple[dict, dict]:
    env, ms = _normal_inputs(tmp_path)
    ctx = md.build_context(repo, env, ms, RELEASE, "2026-10-07T10:03:12+09:00", 100, _no_log)
    return ctx, md.render_unit(ctx)


def test_unit_files_pass_the_local_unit_check(tmp_path: Path) -> None:
    ctx, files = _render(tmp_path, tmp_path / "no-repo")
    assert sorted(files) == sorted(md.FILE_SECTIONS)
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_local_unit_check_catches_each_rule(tmp_path: Path) -> None:
    ctx, files = _render(tmp_path, tmp_path / "no-repo")
    unit = ctx["unit"]

    def check(name: str, old: str, new: str) -> list:
        broken = dict(files)
        assert old in broken[name]
        broken[name] = broken[name].replace(old, new, 1)
        return md.validate_unit_files(unit, broken)

    assert any("front matter 필드 없음" in p for p in check("kr-stocks.md", "revision: 1\n", ""))
    assert any("금지 문자열" in p for p in check("kr-stocks.md", "# KR", "/home/x\n# KR"))
    assert any(
        "링크 대상이 없습니다" in p for p in check("README.md", "(market-sector.md)", "(nope.md)")
    )
    assert any(
        "절대 경로 링크" in p for p in check("README.md", "(market-sector.md)", "(/etc/x.md)")
    )
    big = dict(files)
    big["data-status.md"] += "가" * md.MAX_FILE_BYTES
    assert any("1MB" in p for p in md.validate_unit_files(unit, big))
    missing = {k: v for k, v in files.items() if k != "us-stocks.md"}
    assert any("단위에 파일이 빠졌습니다" in p for p in md.validate_unit_files(unit, missing))


def test_synthetic_unit_needs_explicit_allowance(tmp_path: Path) -> None:
    env, ms = _normal_inputs(tmp_path)
    body = json.loads(env)
    body["synthetic_fixture"] = True
    ctx = md.build_context(
        tmp_path / "x",
        json.dumps(body).encode(),
        ms,
        RELEASE,
        "2026-10-07T10:03:12+09:00",
        100,
        _no_log,
    )
    files = md.render_unit(ctx)
    assert any("합성 fixture" in p for p in md.validate_unit_files(ctx["unit"], files))
    assert md.validate_unit_files(ctx["unit"], files, allow_synthetic=True) == []


def test_tree_check_only_looks_at_given_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    md.init_repo(repo, CARDS, _no_log)
    env, ms = _normal_inputs(tmp_path)
    ctx = md.build_context(repo, env, ms, RELEASE, "2026-10-07T10:03:12+09:00", 100, _no_log)
    for name, text in md.render_unit(ctx).items():
        (md.unit_dir(repo, ctx["unit"]) / name).parent.mkdir(parents=True, exist_ok=True)
        (md.unit_dir(repo, ctx["unit"]) / name).write_text(text, encoding="utf-8")
    md.reindex(repo, _no_log)
    assert md.validate(repo) == []
    # 사람이 고친 다른 문서가 깨져도, 그 파일을 넘기지 않으면 보지 않습니다.
    (repo / "reference" / "glossary.md").write_text("서버 /home/whi\n", encoding="utf-8")
    assert md.validate(repo) != []
    unit_paths = [f"{md.unit_rel(ctx['unit'])}/{name}" for name in md.SECTION_FILES.values()]
    assert md.validate(repo, paths=unit_paths) == []
    assert md.validate(repo, paths=["reference/glossary.md"]) != []
    # 단위 안의 파일 하나만 넘겨도 단위 구성은 디스크 기준으로 전부 봅니다.
    (md.unit_dir(repo, ctx["unit"]) / "us-stocks.md").unlink()
    problems = md.validate(repo, paths=[unit_paths[0]])
    assert any("단위에 파일이 빠졌습니다" in p for p in problems)


def test_svg_allowance_and_limits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assets = md.unit_dir(repo, "2026-10-07") / "assets"
    assets.mkdir(parents=True)
    ok = assets / "a.svg"
    ok.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    rel = f"{md.unit_rel('2026-10-07')}/assets/a.svg"
    assert [p for p in md.validate(repo, paths=[rel]) if "단위에" not in p] == []
    ok.write_text("<svg onload='x()'/>", encoding="utf-8")
    assert any("SVG" in p for p in md.validate(repo, paths=[rel]))
    ok.write_text("<svg>" + "a" * md.SVG_MAX_BYTES + "</svg>", encoding="utf-8")
    assert any("200KB" in p for p in md.validate(repo, paths=[rel]))
    (assets / "b.png").write_bytes(b"x")
    assert any(".md만" in p for p in md.validate(repo, paths=[rel.replace("a.svg", "b.png")]))


def test_ensure_model_cards_creates_only_missing_cards(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    created = md.ensure_model_cards(repo, cards, _no_log)
    assert sorted(created) == sorted(
        [f"reference/models/{m}.md" for m in cards] + [f"reference/models/{md.MS_MODEL}.md"]
    )
    edited = repo / "reference" / "models" / "kr_daily_h20_v1.md"
    edited.write_text("# 내가 고친 카드\n", encoding="utf-8")
    assert md.ensure_model_cards(repo, cards, _no_log) == []
    assert edited.read_text(encoding="utf-8") == "# 내가 고친 카드\n"


def test_unit_hash_is_stable_and_ignores_nothing_but_names(tmp_path: Path) -> None:
    _, files = _render(tmp_path, tmp_path / "no-repo")
    assert md.unit_sha256(files) == md.unit_sha256(dict(reversed(list(files.items()))))
    other = dict(files)
    other["kr-stocks.md"] += "x"
    assert md.unit_sha256(other) != md.unit_sha256(files)


def test_same_content_ignores_generated_at_and_correction_adds_revision(tmp_path: Path) -> None:
    env, ms = _normal_inputs(tmp_path)
    repo = tmp_path / "no-repo"
    first = md.build_context(repo, env, ms, RELEASE, "2026-10-07T10:03:12+09:00", 100, _no_log)
    later = md.build_context(repo, env, ms, RELEASE, "2026-10-07T10:30:00+09:00", 100, _no_log)
    a, b = md.render_unit(first), md.render_unit(later)
    assert a != b and md.same_unit_content(a, b)
    changed = json.loads(env)
    changed["markets"][0]["rankings"][0]["score"] = 0.123456
    third = md.build_context(
        repo,
        json.dumps(changed).encode(),
        ms,
        RELEASE,
        "2026-10-07T11:00:00+09:00",
        100,
        _no_log,
    )
    c = md.render_unit(third)
    assert not md.same_unit_content(a, c)
    fixed = md.correct_unit_files(third, a, "KR 점수 재계산", "abc1234")
    assert md.existing_revision(fixed) == 2
    summary = fixed["README.md"]
    assert "> 정정 r2 (2026-10-07T11:00:00+09:00): KR 점수 재계산. 이전 판: abc1234" in summary
    assert summary.index("정정 r2") < summary.index("## 섹션 상태")
    assert all(md.parse_front_matter(t)[0]["revision"] == 2 for t in fixed.values())
    # 한 번 더 고치면 정정 줄이 쌓입니다.
    again = md.correct_unit_files(third, fixed, "두 번째 정정", None)
    assert md.existing_revision(again) == 3
    assert again["README.md"].index("정정 r3") < again["README.md"].index("정정 r2")


def test_scrub_hits_are_counted_per_render(tmp_path: Path) -> None:
    md.reset_scrub_hits()
    env, ms = _normal_inputs(tmp_path)
    body = json.loads(env)
    body["markets"][0]["rankings"][0]["name"] = "누수/home/whi/key"
    ctx = md.build_context(
        tmp_path / "x",
        json.dumps(body).encode(),
        ms,
        RELEASE,
        "2026-10-07T10:03:12+09:00",
        100,
        _no_log,
    )
    files = md.render_unit(ctx)
    assert md.SCRUB_HITS
    assert "/home/" not in "".join(files.values())
    md.reset_scrub_hits()
    assert md.SCRUB_HITS == []
