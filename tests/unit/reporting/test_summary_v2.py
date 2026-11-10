"""첫 화면 v2와 순위 섹션 목록 요약(설정 `summary_v2_enabled`)의 렌더 시험.

계획: my/milestones/common/20261110_briefing_usefulness/01_plan.md §4.2-§4.4·§6·§7 P2.
꺼져 있으면 지금과 바이트까지 같고, 켜도 순위 표·`- 모델:` 줄·front matter·파일 집합·피드백 게시 행
집합은 그대로인지 봅니다. 입력은 합성 fixture라 실제 시장과 무관합니다.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import explain_fixtures as ef
import macro_fakes as mk
import md_fixtures as mf
import pytest
from test_explain_render import BEFORE_EXPLAIN

from modeler.feedback.briefing import _exposed_predictions
from modeler.reporting import markdown as md

RELEASE = "r20261005"
STAMP = "2026-10-07T10:03:12+09:00"
INVOCATION = "inv-test"
CARDS = Path(__file__).resolve().parents[3] / "deploy" / "prod" / "model-cards.json"
FIVE = set(md.SECTION_FILES.values())
RANK_HEAD = re.compile(r"^\|\s*순위\s*\|\s*코드\s*\|")
MACRO_TEXT = """# 거시 흐름 · {unit}

점수·순위에 쓰지 않습니다. 매매 신호가 아닙니다.

일간 지표 기준일 10/01\\~10/06 · 갱신 지연 0개

## 이번에 눈여겨볼 변화

| 영역 | 관찰 | 대표 수치 | 기준일 |
| --- | --- | --- | --- |
| 금리 | 합성 금리: 1개월 급락 | 1개월 −0.30%p | 2026-10-06 |
| 원자재 | 합성 원자재: 1주 급등 | 1주 +5.00% | 2026-10-02 (갱신 지연) |
| 위험 선호 | 합성 변동성: 1개월 급등 | 1개월 +30.00% | 2026-10-06 |
| 물가·경기 (월간) | 합성 물가: 전월비 급등 | 전월비 +0.50% | 2026-08 |

선정 규칙: 합성입니다.

## 지난 보고서 대비

이전 보고서 날짜는 2026-10-06 (화)입니다.

**새로 나타남**

- 합성 금리: 1개월 급락

**방향 차이 새로 생김**

- 없음

**이어짐 (이번 보고서 포함, 20회에 닿으면 20회 이상으로 표시)**

- 합성 변동성: 1개월 급등 3회

**해제**

- 합성 환율: 1주 급등
- 합성 유가: 3개월 급락

이전 상태는 합성입니다.

## 영역별 상세

합성 상세입니다.
"""


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
def macro_render(monkeypatch):
    """`macro.md` 모양(머리·요약 표·지난 보고서 대비)을 흉내 내는 가짜 렌더러."""
    module = types.ModuleType("modeler.reporting.macro")

    def render(body):
        if body.get("state") == "failed":
            return f"# 거시 흐름 · {body.get('unit')}\n\n거시 흐름을 내지 못했습니다.\n", {}
        return MACRO_TEXT.format(unit=body["unit"]), {}

    module.render = render
    calc, _ = mk.install(monkeypatch)
    monkeypatch.setitem(sys.modules, "modeler.reporting.macro", module)
    return module


def _unit(env: bytes) -> str:
    return json.loads(env)["report_date"]


def explain_doc(env: bytes, *, narrow: bool = True) -> dict:
    """정상 근거 입력. 비교 대상 경로를 시장 키마다 같게 맞추고, 가격 범위가 좁은 행을 만듭니다.

    - US LightGBM 4위: 최근 20세션이 거의 평평합니다 → 가격 범위 좁음
    - US LightGBM 5위: 평평하지만 창 안에 가격 의심일이 있습니다 → 뺌
    - KR 3위: 평평하지만 품질 보류(거래정지)입니다 → 뺌
    """
    doc = ef.explain_doc(env, INVOCATION)
    data = json.loads(env)
    boards = {
        (rep["model_id"], r["symbol"]): r.get("market")
        for rep in data["markets"]
        for r in rep["rankings"]
    }
    for model_id, model in doc["models"].items():
        n = len(model["dates"])
        slopes = {"KOSPI": 0.002, "KOSDAQ": -0.003, "US": 0.001}
        common = {
            key: [round(100.0 * (1 + slope) ** t, 6) for t in range(n)]
            for key, slope in slopes.items()
        }
        for row in model["rows"]:
            key = boards[(model_id, row["symbol"])] if model["market"] == "KR" else "US"
            row["bench_path"] = list(common[key])
            flat = narrow and (
                (model_id == mf.LGB_ID and row["rank"] in (4, 5))
                or (model["market"] == "KR" and row["rank"] == 3)
            )
            if flat:
                row["path"][-md.NARROW_WINDOW :] = [
                    50.0 + 0.01 * i for i in range(md.NARROW_WINDOW)
                ]
            if model_id == mf.LGB_ID and row["rank"] == 5 and narrow:
                row["gaps"] = [{"date": model["dates"][-3], "kind": "ca_suspect"}]
                row["price_status"] = "partial"
    return doc


def macro_doc(env: bytes, **kw) -> bytes:
    unit = _unit(env)
    body = mk.doc(unit, INVOCATION, **kw)
    if body.get("state") != "failed":
        body["header"] = {"daily_asof_min": "2026-10-01", "daily_asof_max": "2026-10-06"}
    return mk.dump(body)


def render(
    tmp_path: Path,
    env: bytes,
    ms: bytes | None,
    *,
    v2: bool,
    explain: dict | bytes | None = None,
    macro: bytes | None = None,
    repo: Path | None = None,
    **kw,
):
    raw = ef.dump(explain) if isinstance(explain, dict) else explain
    extra = {"summary_v2": True} if v2 else {}
    ctx = md.build_context(
        repo or tmp_path / "no-repo",
        env,
        ms,
        RELEASE,
        kw.pop("stamp", STAMP),
        100,
        _no_log,
        explain=raw,
        invocation_id=INVOCATION,
        macro=macro,
        **extra,
        **kw,
    )
    return ctx, md.render_unit(ctx)


def body_lines(text: str) -> list:
    """front matter 뒤 본문 줄(바로 뒤 빈 줄 포함, 마지막 줄바꿈 제외)."""
    _, body = md.split_front_matter(text)
    return body.rstrip("\n").split("\n")


def rank_table_lines(text: str) -> list:
    """머리 첫 두 칸이 `순위`·`코드`인 표의 줄(머리·구분선·행)."""
    out, on = [], False
    for line in text.split("\n"):
        if line.startswith("|"):
            on = on or bool(RANK_HEAD.match(line))
            if on:
                out.append(line)
        else:
            on = False
    return out


def is_subsequence(short: list, long: list) -> bool:
    it = iter(long)
    return all(any(x == y for y in it) for x in short)


def write_prev_unit(repo: Path, unit: str, data_asof: dict | None, raw: str | None = None) -> None:
    udir = md.unit_dir(repo, unit)
    udir.mkdir(parents=True, exist_ok=True)
    if raw is None:
        meta = {"schema": md.SCHEMA, "family": md.FAMILY, "unit": unit, "section": "summary"}
        meta["data_asof"] = data_asof
        raw = md.front_matter(meta) + "\n# 일일 브리핑\n"
    (udir / "README.md").write_text(raw, encoding="utf-8")


def batch_for(env: bytes) -> SimpleNamespace:
    preds = [
        {
            "market": rep["market"],
            "model_id": rep["model_id"],
            "rank": r["rank"],
            "symbol": r["symbol"],
            "prediction_ref": f"{rep['model_id']}:{r['rank']}",
        }
        for rep in json.loads(env)["markets"]
        for r in rep["rankings"]
    ]
    return SimpleNamespace(predictions=preds, errors=[])


# --- 꺼짐: 지금과 바이트까지 같습니다 ------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BEFORE_EXPLAIN))
def test_off_is_byte_identical_to_before(tmp_path: Path, inputs: dict, name: str) -> None:
    env, ms = inputs[name]
    _, files = render(tmp_path, env, ms, v2=False)
    assert md.unit_sha256(files) == BEFORE_EXPLAIN[name]
    ctx = md.build_context(
        tmp_path / "no-repo", env, ms, RELEASE, STAMP, 100, _no_log, summary_v2=False
    )
    assert "summary_v2" not in ctx and "prev_unit" not in ctx
    assert md.unit_sha256(md.render_unit(ctx)) == BEFORE_EXPLAIN[name]


def test_off_with_explain_and_macro_equals_no_key(
    tmp_path: Path, inputs: dict, macro_render
) -> None:
    env, ms = inputs["f1_normal"]
    repo = tmp_path / "repo"
    write_prev_unit(repo, "2026-10-06", md.all_data_asof(render(tmp_path, env, ms, v2=False)[0]))
    kw = {"explain": explain_doc(env), "macro": macro_doc(env)}
    _, plain = render(tmp_path, env, ms, v2=False, **kw)
    ctx = md.build_context(
        repo,
        env,
        ms,
        RELEASE,
        STAMP,
        100,
        _no_log,
        explain=ef.dump(kw["explain"]),
        invocation_id=INVOCATION,
        macro=kw["macro"],
        summary_v2=False,
    )
    assert md.render_unit(ctx) == plain
    assert "목록 요약" not in plain["kr-stocks.md"] + plain["us-stocks.md"]


# --- 켬: 순위 표·파일 집합·피드백 행은 그대로 ------------------------------------------------
@pytest.fixture()
def pair(tmp_path: Path, inputs: dict, macro_render) -> tuple:
    env, ms = inputs["f1_normal"]
    kw = {"explain": explain_doc(env), "macro": macro_doc(env)}
    v1_ctx, v1 = render(tmp_path, env, ms, v2=False, **kw)
    v2_ctx, v2 = render(tmp_path, env, ms, v2=True, **kw)
    return env, v1_ctx, v1, v2_ctx, v2


def test_on_keeps_files_front_matter_rank_tables_and_other_sections(pair) -> None:
    env, v1_ctx, v1, v2_ctx, v2 = pair
    assert set(v1) == set(v2) and FIVE | {md.MACRO_FILE} <= set(v2)
    assert v2_ctx["sha"] == v1_ctx["sha"] == md.sha256_bytes(env)
    for name in v1:
        if name in ("README.md", "kr-stocks.md", "us-stocks.md"):
            continue
        assert v2[name] == v1[name], name  # 시장·섹터·데이터 상태·거시 흐름·그림은 그대로
    for name in ("README.md", "kr-stocks.md", "us-stocks.md"):
        assert md.split_front_matter(v2[name])[0] == md.split_front_matter(v1[name])[0]
        meta, _ = md.parse_front_matter(v2[name])
        assert meta["source"]["report_sha256"] == md.sha256_bytes(env)
    for name in ("kr-stocks.md", "us-stocks.md"):
        old, new = v1[name].split("\n"), v2[name].split("\n")
        assert rank_table_lines(v2[name]) == rank_table_lines(v1[name]) != []
        assert is_subsequence(old, new) and len(new) > len(old)  # 넣기만 했습니다
        assert [x for x in old if x.startswith("- 모델:")] == [
            x for x in new if x.startswith("- 모델:")
        ]
        added = [x for x in new if x not in set(old)]
        assert not any(RANK_HEAD.match(x) for x in added)
        heads = [sum(1 for x in t.split("\n") if RANK_HEAD.match(x)) for t in (v1[name], v2[name])]
        assert heads[0] == heads[1]
    # README는 상위 10 표를 빼서 `순위 | 코드` 표가 없습니다.
    assert not any(RANK_HEAD.match(x) for x in v2["README.md"].split("\n"))
    assert md.validate_unit_files(v2_ctx["unit"], v2) == []
    for name, data in v2.items():
        size = len(data.encode() if isinstance(data, str) else data)
        assert size < (md.SVG_MAX_BYTES if name.endswith(".svg") else md.MAX_FILE_BYTES)


def test_on_feedback_exposed_rows_are_identical_to_v1(pair) -> None:
    env, _, v1, _, v2 = pair
    got = {}
    for label, files in (("v1", v1), ("v2", v2)):
        batch = batch_for(env)
        raw = {k: v.encode() for k, v in files.items() if isinstance(v, str)}
        got[label] = _exposed_predictions(raw, batch)
        assert batch.errors == []
    assert got["v1"] == got["v2"] and len(got["v2"]) == 300


def test_on_readme_order_length_and_values(pair) -> None:
    _, _, _, ctx, v2 = pair
    lines = body_lines(v2["README.md"])
    assert len(lines) <= 65
    heads = [(i + 1, x) for i, x in enumerate(lines) if x.startswith("## ")]
    assert [h for _, h in heads] == [
        "## 시장",
        "## 거시 — 눈여겨볼 변화",
        "## 종목 순위 — 어떤 목록인가",
        "## 상태",
    ]
    macro_end = heads[2][0] - 2  # 순위 절 바로 앞 빈 줄 앞이 거시 절의 마지막 줄
    assert macro_end <= 35
    text = "\n".join(lines)
    assert "단위 상태: 부분 완료 (오늘의 경고 0건 · 상시 한계 1건)." in text
    assert "> " + md.RANKING_STATUS_ALL in text
    assert "| 모델 | 기준일 | 전체 → 표시 | 첫째 요인 (상위 100) |" in text
    assert "- 상시 한계: KR은 현재 관리종목·거래정지 상태를 확인하지 못해 부분 완료로" in text
    assert "- 오늘의 경고: 없습니다." in text
    assert "## 경고" not in text and "## KR 상위 10" not in text
    # 거시 표와 문장은 macro.md 그대로이고, 지난 보고서 대비는 개수와 이름만 한 줄입니다.
    macro = v2[md.MACRO_FILE]
    table = [x for x in lines if x.startswith("| 금리 |") or x.startswith("| 원자재 |")]
    assert table and all(x in macro.split("\n") for x in table)
    assert (
        "지난 보고서(2026-10-06) 대비: 새로 나타남 1 (합성 금리: 1개월 급락) · 방향 차이 새로 "
        "생김 0 · 해제 2 (합성 환율: 1주 급등, 합성 유가: 3개월 급락). 상세는 [거시 흐름](macro.md)에 "
        "있습니다." in text
    )
    assert "기준일: KR 종목 " in text and "거시 2026-10-01\\~2026-10-06" in text
    # 적격 종목 평균: 시장 키마다 같은 경로라 한 줄에 셋이 다 나옵니다.
    avg = next(x for x in lines if x.startswith("- 적격 종목 동일가중 평균"))
    assert "코스피 +0.2% · +1.0% · +4.1% · +12.7%" in avg and "지수가 아닙니다." in avg
    assert "코스닥 -0.3% · -1.5% · -5.8% · -16.5%" in avg and "US +0.1%" in avg
    # 지수 표는 SPY·QQQ·코스피·코스닥 중 입력에 있는 시장 지수, 섹터 줄은 시장별 최고·최저입니다.
    assert any(
        x.startswith("| 지수 | 기준 종가일 | 1세션 | 5세션 | 20세션 | 60세션 |") for x in lines
    )
    # 새로 실은 날짜는 모두 단위 날짜 이하입니다.
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    assert dates and max(dates) <= ctx["unit"]


def test_on_list_summary_blocks(pair) -> None:
    env, _, _, ctx, v2 = pair
    kr, us = v2["kr-stocks.md"], v2["us-stocks.md"]
    assert kr.count("### 목록 요약") == 1 and us.count("### 목록 요약") == 2
    assert "| 항목 | 값 |" in kr
    kr_m = ctx["kr"]["models"][0]
    boards = [r["market"] for r in kr_m["rows"]]
    mix = f"| 시장 구성 | 코스피 {boards.count('KOSPI')} · 코스닥 {boards.count('KOSDAQ')} |"
    assert mix in kr
    assert "| 표시 | 품질 보류 3 · " in kr and "가격 범위 좁음 0 |" in kr  # KR 3위는 보류라 뺌
    lgb = us[us.index("## LightGBM") : us.index("## Ridge")]
    sym4 = ctx["us"]["models"][0]["rows"][3]["symbol"]
    assert f"가격 범위 좁음 1 — 4위 {sym4} |" in lgb  # 5위는 창 안 의심일이라 뺌
    assert "| 비교 대상 평균 (5 · 20 · 60세션) | +0.5% · +2.0% · +6.2% |" in lgb
    # 중앙값은 소수 1자리(게시 표와 같음)
    med = re.search(r"\| 수익률 중앙값 \(5 · 20 · 60세션\) \| (.+) \|", lgb).group(1)
    assert all(re.fullmatch(r"[+-]\d+\.\d%", x) for x in med.split(" · "))
    assert "기준일 A(" in lgb and "A 뒤 가격은 쓰지 않았습니다." in lgb
    assert md.NARROW_NOTE in lgb and md.FIRST_FACTOR_NOTE in lgb
    # 블록은 `전체 N개 중` 줄 바로 다음, 순위 표보다 위에 있습니다.
    assert kr.index("전체 ") < kr.index("### 목록 요약") < kr.index("## 품질 보류 요약")
    first = lgb.index("### 목록 요약")
    assert lgb.index("전체 ") < first < lgb.index("| 순위 | 코드 |")


def test_first_factor_counts_follow_the_largest_delta(pair) -> None:
    env, _, _, ctx, v2 = pair
    doc = explain_doc(env)["models"][mf.KR_ID]
    labels = {g["id"]: g["label"] for g in doc["groups"]}
    order = [g["id"] for g in doc["groups"]]
    counts: dict = {}
    for row in doc["rows"]:
        best = max(order, key=lambda g: (row["delta"][g], -order.index(g)))
        counts[labels[best]] = counts.get(labels[best], 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], order.index(next(g for g in order if labels[g] == kv[0]))),
    )
    want = " · ".join(f"{k} {v}" for k, v in ranked)
    assert f"| 첫째 요인 | {want} |" in v2["kr-stocks.md"]
    top3 = " · ".join(f"{k} {v}" for k, v in ranked[:3])
    assert f"| {top3} |" in v2["README.md"]


# --- 같은 기준 세션 ---------------------------------------------------------------------
def test_same_base_session_marks_only_the_matching_market(tmp_path: Path, inputs: dict) -> None:
    env, ms = inputs["f1_normal"]
    asof = md.all_data_asof(render(tmp_path, env, ms, v2=False)[0])
    repo = tmp_path / "repo"
    write_prev_unit(repo, "2026-10-05", {**asof, "kr_features": "2026-09-30"})
    write_prev_unit(repo, "2026-10-06", {**asof, "kr_features": "2026-10-01"})
    _, files = render(tmp_path, env, ms, v2=True, repo=repo)
    readme, us, kr = files["README.md"], files["us-stocks.md"], files["kr-stocks.md"]
    a = asof["us_features"]
    assert f"US 종목 {a} (**전 단위 2026-10-06와 같은 기준 세션**)" in readme
    assert readme.count("(전 단위와 같음)") == 2  # US 두 모델 행
    sentence = (
        f"기준 세션 A({a})가 전 단위(2026-10-06)와 같습니다. 새 US 자료가 없어 같은 입력에서 "
        "나온 순위입니다."
    )
    assert us.count(sentence) == 1
    assert us.index(md.RANKING_STATUS_US) < us.index(sentence) < us.index("## LightGBM")
    assert "전 단위" not in kr


def test_same_base_session_kr(tmp_path: Path, inputs: dict) -> None:
    env, ms = inputs["f1_normal"]
    asof = md.all_data_asof(render(tmp_path, env, ms, v2=False)[0])
    repo = tmp_path / "repo"
    write_prev_unit(repo, "2026-10-06", {**asof, "us_features": "2026-10-01"})
    _, files = render(tmp_path, env, ms, v2=True, repo=repo)
    kr = files["kr-stocks.md"]
    sentence = f"기준일({asof['kr_features']})이 전 단위(2026-10-06)와 같습니다."
    assert kr.count(sentence) == 1
    assert kr.index(md.RANKING_STATUS_KR) < kr.index(sentence) < kr.index("전체 ")
    assert "전 단위" not in files["us-stocks.md"]


def test_no_marker_when_previous_unit_is_missing_or_unreadable(
    tmp_path: Path, inputs: dict
) -> None:
    env, ms = inputs["f1_normal"]
    asof = md.all_data_asof(render(tmp_path, env, ms, v2=False)[0])
    _, none = render(tmp_path, env, ms, v2=True, repo=tmp_path / "empty")
    repo = tmp_path / "repo"
    write_prev_unit(repo, "2026-10-05", asof)  # 같지만 바로 앞 단위가 아닙니다
    write_prev_unit(repo, "2026-10-06", None, raw="front matter 없음\n")
    _, broken = render(tmp_path, env, ms, v2=True, repo=repo)
    for files in (none, broken):
        assert "전 단위" not in files["README.md"] + files["us-stocks.md"] + files["kr-stocks.md"]
    later = tmp_path / "later"
    write_prev_unit(later, "2026-10-08", asof)  # 이번 단위보다 뒤는 보지 않습니다
    _, files = render(tmp_path, env, ms, v2=True, repo=later)
    assert "전 단위" not in files["README.md"]


# --- 입력이 빠진 경우 --------------------------------------------------------------------
def test_macro_absent_or_failed_omits_the_macro_section(
    tmp_path: Path, inputs: dict, macro_render
) -> None:
    env, ms = inputs["f1_normal"]
    _, absent = render(tmp_path, env, ms, v2=True)
    _, failed = render(tmp_path, env, ms, v2=True, macro=macro_doc(env, state="failed"))
    for files in (absent, failed):
        assert "## 거시" not in files["README.md"]
        assert md.validate_unit_files(_unit(env), files) == []
    assert "| [거시 흐름](macro.md) | 실패 |" in failed["README.md"]
    assert "거시 " not in absent["README.md"].split("기준일: ", 1)[1].split("\n", 1)[0]


@pytest.mark.parametrize("case", ["absent", "mismatch", "model_failed"])
def test_explain_missing_or_failed_degrades(tmp_path: Path, inputs: dict, case: str) -> None:
    env, ms = inputs["f1_normal"]
    doc = explain_doc(env)
    if case == "mismatch":
        doc["report_sha256"] = "0" * 64
    if case == "model_failed":
        doc["models"][mf.RDG_ID] = {"status": "failed", "reason": "timeout"}
    ctx, files = render(tmp_path, env, ms, v2=True, explain=None if case == "absent" else doc)
    kr, us, readme = files["kr-stocks.md"], files["us-stocks.md"], files["README.md"]
    assert md.validate_unit_files(ctx["unit"], files) == []
    ridge = us[us.index("## Ridge") :]
    assert "| 첫째 요인 | 확인되지 않음 |" in ridge
    assert "| 비교 대상 평균 (5 · 20 · 60세션) | 확인되지 않음 |" in ridge
    assert "가격 범위 좁음 확인되지 않음" in ridge
    row = next(x for x in readme.split("\n") if x.startswith("| US Ridge |"))
    assert "확인되지 않음" in row
    if case == "model_failed":
        assert "| 첫째 요인 | 확인되지 않음 |" not in kr
        assert "- 오늘의 경고: US Ridge 근거·그래프를 계산하지 못했습니다" in readme
        assert "(오늘의 경고 1건 · 상시 한계 1건)" in readme
    else:
        assert "적격 종목 동일가중 평균" not in readme
        assert "| 첫째 요인 | 확인되지 않음 |" in kr
    if case == "mismatch":
        assert "(오늘의 경고 3건 · 상시 한계 1건)" in readme
        assert "- 오늘의 경고:\n  - KR 근거·그래프를 계산하지 못했습니다" in readme


def test_bench_line_needs_one_path_per_market(tmp_path: Path, inputs: dict) -> None:
    env, ms = inputs["f1_normal"]
    _, files = render(tmp_path, env, ms, v2=True, explain=ef.explain_doc(env, INVOCATION))
    # 합성 근거의 비교 대상 경로는 행마다 달라 시장 평균이 아닙니다 → 평균 줄을 내지 않습니다.
    assert "적격 종목 동일가중 평균" not in files["README.md"]
    assert "| 비교 대상 평균 (5 · 20 · 60세션) | 확인되지 않음 |" in files["kr-stocks.md"]


@pytest.mark.parametrize(
    "name", ["f2_kr_unavailable_us_stale", "f3_replay", "f4_all_failed", "f5_lag_over_limit"]
)
def test_other_scenarios_render_without_crash(tmp_path: Path, inputs: dict, name: str) -> None:
    env, ms = inputs[name]
    explain = explain_doc(env) if name != "f4_all_failed" else None
    v1_ctx, v1 = render(tmp_path, env, ms, v2=False, explain=explain)
    ctx, v2 = render(tmp_path, env, ms, v2=True, explain=explain)
    assert set(v1) == set(v2)
    assert md.validate_unit_files(ctx["unit"], v2, allow_synthetic=True) == []
    for f in ("kr-stocks.md", "us-stocks.md"):
        assert rank_table_lines(v2[f]) == rank_table_lines(v1[f])
    readme = v2["README.md"]
    assert "## 종목 순위 — 어떤 목록인가" in readme and "## 상태" in readme
    warnings = [t for k, t in md.summary_warnings(ctx) if k == md.WARN_TODAY]
    assert f"(오늘의 경고 {len(warnings)}건 " in readme
    for m in ctx["kr"]["models"] + ctx["us"]["models"]:
        if m["status"] == "failed":
            label = "KR" if m["market"] == "KR" else "US " + md.US_SHORT[m["model_id"]]
            assert f"- {label} 상위 10: 순위를 내지 못했습니다." in readme


def test_model_with_no_rankings(tmp_path: Path) -> None:
    day = "2026-10-07"
    reports = [
        mf.kr_report(day, "2026-10-06"),
        mf.us_report(day, mf.LGB_ID, "2026-10-06", "2026-10-06", "2026-10-06"),
        mf.us_report(day, mf.RDG_ID, "2026-10-06", "2026-10-06", "2026-10-06"),
    ]
    full = (json.dumps(mf.envelope(day, reports, []), sort_keys=True) + "\n").encode()
    doc = explain_doc(full)
    reports[2]["rankings"] = []
    env = (json.dumps(mf.envelope(day, reports, []), sort_keys=True) + "\n").encode()
    doc["models"][mf.RDG_ID]["rows"] = []
    doc["report_sha256"] = md.sha256_bytes(env)
    ms = json.dumps(mf.ms_input(day, kr_asof="2026-10-06", us_asof="2026-10-06")).encode()
    ctx, files = render(tmp_path, env, ms, v2=True, explain=doc)
    us, readme = files["us-stocks.md"], files["README.md"]
    assert us.count("### 목록 요약") == 1
    assert "| US Ridge | 2026-10-06 | 이번 판에 순위가 없습니다 | - | - | - |" in readme
    assert "- US Ridge 상위 10: 이번 판에 순위가 없습니다." in readme
    assert "두 US 모델" not in readme
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_correction_line_stays_on_top_and_counts_as_todays_warning(
    tmp_path: Path, inputs: dict
) -> None:
    env, ms = inputs["f1_normal"]
    ctx, files = render(tmp_path, env, ms, v2=True, explain=explain_doc(env))
    fixed = md.correct_unit_files(ctx, files, "표시 오류", "abc1234")
    readme = fixed["README.md"]
    assert readme.index(md.CORR_BEGIN) < readme.index("단위 상태:")
    assert "- 오늘의 경고: 이 단위는 정정판입니다." in readme


# --- 검증기: 저장소에 써도 규칙에 맞습니다 ----------------------------------------------------
def test_generated_v2_unit_passes_the_repository_validator(
    tmp_path: Path, inputs: dict, macro_render
) -> None:
    env, ms = inputs["f1_normal"]
    repo = tmp_path / "repo"
    md.init_repo(repo, CARDS, _no_log)
    paths = {}
    for name, data in (
        ("env.json", env),
        ("ms.json", ms),
        ("explain.json", ef.dump(explain_doc(env))),
        ("macro.json", macro_doc(env)),
    ):
        paths[name] = tmp_path / name
        paths[name].write_bytes(data)
    action = md.generate(
        repo,
        paths["env.json"],
        paths["ms.json"],
        RELEASE,
        STAMP,
        100,
        None,
        None,
        False,
        _no_log,
        explain_path=paths["explain.json"],
        invocation_id=INVOCATION,
        macro_path=paths["macro.json"],
        summary_v2=True,
    )
    assert action == "created"
    assert md.validate(repo) == []
    udir = md.unit_dir(repo, _unit(env))
    assert "## 종목 순위 — 어떤 목록인가" in (udir / "README.md").read_text(encoding="utf-8")
    # 같은 입력으로 다시 돌리면 내용이 같아 건드리지 않습니다.
    again = md.generate(
        repo,
        paths["env.json"],
        paths["ms.json"],
        RELEASE,
        "2026-10-07T10:40:00+09:00",
        100,
        None,
        None,
        False,
        _no_log,
        explain_path=paths["explain.json"],
        invocation_id=INVOCATION,
        macro_path=paths["macro.json"],
        summary_v2=True,
    )
    assert again == "unchanged"
