#!/usr/bin/env python3
"""modeler.reporting.markdown 시험.

임시 디렉터리에 시험 저장소(REPO)를 한 번 만들고, 변경을 가하는 시험은 복사본(WORK)에서 합니다.
이 저장소의 어떤 경로도 건드리지 않습니다. 원본 일회성 변환기의 시험 45개를 옮긴 것입니다.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import md_fixtures as mf

from modeler.reporting import markdown as bm

PROJECT = Path(__file__).resolve().parents[3]
CARDS = PROJECT / "deploy" / "prod" / "model-cards.json"
_TMP = Path(tempfile.mkdtemp(prefix="stock-reports-md-"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
REPO = _TMP / "repo"
WORK = _TMP / "work"
INDIR = _TMP / "inputs"
FIX = _TMP / "fixtures"
RELEASE = "r20261005"
# 단위마다 고정한 generated_at (결정성 시험용)
GENERATED = {
    "f3_replay": "2026-10-06T14:40:00+09:00",
    "f1_normal": "2026-10-07T10:03:12+09:00",
    "f2_kr_unavailable_us_stale": "2026-10-08T10:03:40+09:00",
    "f4_all_failed": "2026-10-12T10:02:55+09:00",
    "f5_lag_over_limit": "2026-10-14T10:03:01+09:00",
}
ORDER = [
    "f3_replay",
    "f1_normal",
    "f2_kr_unavailable_us_stale",
    "f4_all_failed",
    "f5_lag_over_limit",
]
UNITS = {
    "replay": "2026-09-30",
    "normal": "2026-10-07",
    "kr_unavail": "2026-10-08",
    "all_failed": "2026-10-12",
    "lag6": "2026-10-14",
}
FIVE = ["README.md", "market-sector.md", "kr-stocks.md", "us-stocks.md", "data-status.md"]


def build(repo: Path, fixtures: Path) -> int:
    """fixture 5개로 시험 저장소를 처음부터 만듭니다."""
    paths = mf.write_all(fixtures)
    if repo.exists():
        shutil.rmtree(repo)
    code = bm.main(["--repo", str(repo), "--init", "--no-validate", "--model-cards", str(CARDS)])
    for name in ORDER:
        env_path, ms_path = paths[name]
        argv = [
            "--repo",
            str(repo),
            "--release",
            RELEASE,
            "--generated-at",
            GENERATED[name],
            "--envelope",
            str(env_path),
            "--no-validate",
        ]
        if ms_path:
            argv += ["--market-sector", str(ms_path)]
        code |= bm.main(argv)
    return code | bm.main(["--repo", str(repo), "--validate"])


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = bm.main(argv)
    return code, out.getvalue(), err.getvalue()


def unit_path(repo: Path, unit: str) -> Path:
    return repo / "reports" / "daily-briefing" / unit[:4] / unit[5:7] / unit


def snapshot(repo: Path) -> dict:
    snap = {}
    for p in sorted(repo.rglob("*")):
        if p.is_file():
            snap[p.relative_to(repo).as_posix()] = (
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_mtime_ns,
            )
    return snap


def text(repo: Path, unit: str, name: str) -> str:
    return (unit_path(repo, unit) / name).read_text(encoding="utf-8")


def outside_auto(t: str) -> str:
    a, b = t.find(bm.AUTO_BEGIN), t.find(bm.AUTO_END)
    return t[:a] + t[b + len(bm.AUTO_END) :] if a != -1 and b != -1 else t


def gen_args(repo, name, **kw):
    env_path, ms_path = (FIX / (name + "_envelope.json"), FIX / (name + "_ms.json"))
    argv = [
        "--repo",
        str(repo),
        "--release",
        RELEASE,
        "--generated-at",
        kw.pop("generated_at", GENERATED.get(name, "2026-10-20T10:00:00+09:00")),
        "--envelope",
        str(env_path),
    ]
    if ms_path.exists():
        argv += ["--market-sector", str(ms_path)]
    for k, v in kw.items():
        argv += ["--" + k.replace("_", "-")] + ([] if v is True else [str(v)])
    return argv


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = build(REPO, FIX)
        assert code == 0, "초기 빌드가 실패했습니다: " + out.getvalue()
        if WORK.exists():
            shutil.rmtree(WORK)

    @classmethod
    def tearDownClass(cls):
        for d in (WORK, INDIR):
            if d.exists():
                shutil.rmtree(d)

    def fresh_copy(self) -> Path:
        for d in (WORK, INDIR):
            if d.exists():
                shutil.rmtree(d)
        INDIR.mkdir()
        shutil.copytree(REPO, WORK)
        return WORK


class TestBuild(Base):
    def test_validate_passes(self):
        self.assertEqual(bm.validate(REPO), [])

    def test_tree_has_all_files(self):
        for unit in UNITS.values():
            for name in FIVE:
                self.assertTrue((unit_path(REPO, unit) / name).is_file(), (unit, name))
        for rel in (
            "README.md",
            "CONVENTIONS.md",
            "reference/README.md",
            "reference/glossary.md",
            "reference/models/kr_daily_h20_v1.md",
            "reference/models/ms1_market_sector.md",
            "reference/models/us_exploratory_20260929_r1_lightgbm.md",
            "reference/models/us_exploratory_20260929_r1_ridge.md",
            "reports/daily-briefing/README.md",
            "reports/daily-briefing/2026/10/README.md",
            "reports/daily-briefing/2026/09/README.md",
        ):
            self.assertTrue((REPO / rel).is_file(), rel)

    def test_front_matter_all_fields(self):
        for unit in UNITS.values():
            for name in FIVE:
                meta, _ = bm.parse_front_matter(text(REPO, unit, name))
                for field in bm.UNIT_FM_FIELDS:
                    self.assertIn(field, meta, (unit, name, field))
                self.assertEqual(meta["schema"], "stock-reports.v1")
                self.assertEqual(meta["family"], "daily-briefing")
                self.assertEqual(meta["unit"], unit)
                self.assertEqual(meta["section"], bm.FILE_SECTIONS[name])
                self.assertEqual(meta["revision"], 1)
                self.assertIn(meta["status"], ("ok", "partial", "stale", "failed"))
                self.assertTrue(bm.is_hex64(meta["source"]["report_sha256"]))
                self.assertEqual(meta["source"]["release"], "r20261005")
                self.assertTrue(meta["decision_at"].endswith("T10:00:00+09:00"))

    def test_status_mapping(self):
        def st(unit):
            return {n: bm.parse_front_matter(text(REPO, unit, n))[0]["status"] for n in FIVE}

        normal = st(UNITS["normal"])
        self.assertEqual(
            (
                normal["market-sector.md"],
                normal["kr-stocks.md"],
                normal["us-stocks.md"],
                normal["README.md"],
            ),
            ("ok", "partial", "ok", "partial"),
        )
        kr = st(UNITS["kr_unavail"])
        self.assertEqual(
            (kr["kr-stocks.md"], kr["us-stocks.md"], kr["README.md"]),
            ("failed", "stale", "partial"),
        )
        failed = st(UNITS["all_failed"])
        self.assertEqual(set(failed.values()), {"failed"})
        lag6 = st(UNITS["lag6"])
        self.assertEqual((lag6["kr-stocks.md"], lag6["us-stocks.md"]), ("stale", "stale"))

    def test_internal_to_front_matter_mapping(self):
        self.assertEqual(
            [
                bm.STATUS_MAP[s]
                for s in ("ok", "partial", "stale", "withheld", "unavailable", "failed")
            ],
            ["ok", "partial", "stale", "failed", "failed", "failed"],
        )
        self.assertEqual(bm.unit_status(["ok", "ok", "ok"]), "ok")
        self.assertEqual(bm.unit_status(["failed", "failed", "failed"]), "failed")
        self.assertEqual(bm.unit_status(["ok", "failed", "ok"]), "partial")
        self.assertEqual(bm.unit_status(["ok", "stale", "ok"]), "partial")
        self.assertEqual(bm.aggregate_status(["ok", "stale"]), "stale")
        self.assertEqual(bm.aggregate_status(["ok", "failed"]), "partial")
        self.assertEqual(bm.aggregate_status(["failed", "failed"]), "failed")


class TestContent(Base):
    def test_kr_top100_rows_and_wording(self):
        t = text(REPO, UNITS["normal"], "kr-stocks.md")
        rows = [ln for ln in t.split("\n") if re.match(r"\| \d+ \| \d{6} \|", ln)]
        self.assertEqual(len(rows), 100)
        self.assertIn("전체 120개 중 상위 100개", t)
        self.assertIn("순위용 점수 — 확률 아님", t)
        self.assertTrue(all(re.search(r"\| \d\.\d{4} \|", ln) for ln in rows))
        self.assertIn("품질 보류: K 기준 거래정지 또는 상태 미확인", t)
        self.assertIn("상위 100개 중 품질 보류 3개", t)
        self.assertNotIn("100101", t)  # 101위(상위 100 밖)는 나오지 않습니다

    def test_us_two_models_and_overlap(self):
        t = text(REPO, UNITS["normal"], "us-stocks.md")
        self.assertIn("## LightGBM", t)
        self.assertIn("## Ridge", t)
        rows = [
            ln for ln in t.split("\n") if re.match(r"\| \d+ \| [A-Z]{2}\d{3} \| -?\d\.\d{4} \|", ln)
        ]
        self.assertEqual(len(rows), 200)
        self.assertIn("두 모델 상위 100의 겹침: 70개", t)
        self.assertIn("회사명 원천이 없어 이름은 코드(symbol)로 대신 표시합니다", t)
        self.assertIn("기준 세션 A: 2026-10-06", t)
        self.assertIn("순위용 점수 — 확률 아님", t)

    def test_table_escape(self):
        t = text(REPO, UNITS["normal"], "kr-stocks.md")
        self.assertIn("샘플\\|기업<br>005", t)
        self.assertIn("샘플\\[링크\\](x)\\*기업", t)
        header_pipes = None
        for line in t.split("\n"):
            if line.startswith("| 순위 | 코드"):
                header_pipes = len(re.findall(r"(?<!\\)\|", line))
            if header_pipes and re.match(r"\| \d+ \| \d{6} \|", line):
                self.assertEqual(len(re.findall(r"(?<!\\)\|", line)), header_pipes, line)

    def test_no_unescaped_tilde_anywhere(self):
        for p in REPO.rglob("*.md"):
            body = bm.strip_code(p.read_text(encoding="utf-8"))
            self.assertIsNone(re.search(r"(?<!\\)~", body), p.relative_to(REPO))
        # 사양의 점검 명령(grep '[^\\]~')과 같은 결과인지 코드 블록 밖에서 직접 봅니다.
        self.assertIn("0\\~100", text(REPO, UNITS["normal"], "market-sector.md"))
        self.assertIn("09:30\\~10:00", text(REPO, UNITS["replay"], "data-status.md"))

    def test_no_recommendation_words(self):
        # 운영 모델 카드 원문의 "매수 가능 가격을 뜻하지 않습니다"는 부정 문장이라
        # reference/는 뺍니다.
        for p in (REPO / "reports").rglob("*.md"):
            t = p.read_text(encoding="utf-8")
            for word in ("매수", "추천", "사세요", "유망"):
                self.assertNotIn(word, t, (p.relative_to(REPO), word))

    def test_market_sector_layout(self):
        t = text(REPO, UNITS["normal"], "market-sector.md")
        self.assertNotIn("5432.1", t)  # 지수 종가 수준(fixture의 close)은 나오면 안 됩니다
        self.assertNotIn("2711.9", t)
        self.assertNotIn("close", t)
        head, details = t.split("<details>", 1)
        self.assertIn("## 시장 상태", head)
        self.assertIn("## baseline", head)
        self.assertIn("b_opp_mean", head)
        self.assertIn("b_stab_logit_rvol", head)
        self.assertNotIn("Opportunity", head)  # 모델 점수는 본문에 나오지 않습니다
        self.assertTrue(t.rstrip().endswith("</details>"))  # 문서 맨 아래
        self.assertIn(
            "<summary>MS1 연구용 점수 — 판정 실패·채택 없음, 의사결정에 쓰지 않음</summary>",
            details,
        )
        pos = {
            k: details.find(k)
            for k in ("확률로 읽지 않습니다", "**목표별 판정**", "**연구용 점수**")
        }
        self.assertTrue(
            0 < pos["확률로 읽지 않습니다"] < pos["**목표별 판정**"] < pos["**연구용 점수**"], pos
        )
        self.assertIn("| 섹터 상대 선택 (Ridge) | 보류 | 실패 |", details)
        self.assertIn("판정 6개 중 통과 0, 보류 1, 실패 5입니다.", details)
        self.assertEqual(
            len(re.findall(r"^\| (?:SPY|QQQ|XL|코스|KRX)", head, flags=re.M)), 14 * 2
        )  # 상태 표 + baseline 표
        self.assertIn("현금 금리(DGS3MO): 4.20%", head)
        self.assertIn("현금 금리(CD91): 3.45%", head)

    def test_summary_content(self):
        t = text(REPO, UNITS["normal"], "README.md")
        for needle in (
            "## 섹션 상태",
            "## 경고",
            "## 대표지수",
            "## KR 상위 10",
            "## US 상위 10",
            "[시장·섹터](market-sector.md)",
            "[데이터 상태](data-status.md)",
        ):
            self.assertIn(needle, t)
        self.assertEqual(
            len([ln for ln in t.split("\n") if re.match(r"\| \d+ \| 1000\d\d \|", ln)]), 10
        )
        self.assertEqual(len(re.findall(r"^\| (?:SPY|QQQ|코스피|코스닥)", t, flags=re.M)), 4)

    def test_kr_unavailable_and_us_stale_lag3(self):
        u = UNITS["kr_unavail"]
        kr = text(REPO, u, "kr-stocks.md")
        self.assertIn("순위를 내지 못했습니다.", kr)
        self.assertIn("`ValueError`", kr)
        self.assertNotIn("| 순위 | 코드", kr)
        self.assertIn("마지막 정상 단위: [2026-10-07](../2026-10-07/kr-stocks.md)", kr)
        us = text(REPO, u, "us-stocks.md")
        self.assertIn("지연 세션 수: 3", us)
        self.assertIn("> 경고: 미국 입력이 3세션 늦었습니다.", us)
        self.assertEqual(
            len([ln for ln in us.split("\n") if re.match(r"\| \d+ \| [A-Z]{2}\d{3} \|", ln)]), 200
        )
        readme = text(REPO, u, "README.md")
        self.assertIn("KR: 순위를 내지 못했습니다", readme)
        self.assertIn("US LightGBM·Ridge: 입력이 3세션 늦습니다.", readme)

    def test_lag_over_limit_suppresses_tables(self):
        u = UNITS["lag6"]
        kr = text(REPO, u, "kr-stocks.md")
        us = text(REPO, u, "us-stocks.md")
        for t in (kr, us):
            self.assertNotIn("| 순위 | 코드", t)
            self.assertIn("5세션 초과", t)
        self.assertIn("[2026-10-07](../2026-10-07/kr-stocks.md)", kr)
        self.assertIn("[2026-10-07](../2026-10-07/us-stocks.md)", us)
        self.assertNotIn(
            "6세션 전 기준 순위입니다", kr
        )  # 표를 내지 않으면 "순위입니다" 배너도 없습니다
        self.assertIn("K′", kr)

    def test_replay(self):
        u = UNITS["replay"]
        for name in FIVE:
            t = text(REPO, u, name)
            meta, body = bm.parse_front_matter(t)
            self.assertIs(meta["historical_replay"], True, name)
            self.assertTrue(meta["title"].endswith(" (재현)"), name)
            self.assertIn(bm.REPLAY_NOTICE, body, name)
        first_block = text(REPO, u, "README.md").split("\n# ", 1)[1].split("\n\n")[:2]
        self.assertTrue(first_block[1].startswith("> 재현 리포트입니다."))
        ds = text(REPO, u, "data-status.md")
        self.assertIn("## 재현 정보", ds)
        for needle in (
            "| K (KR 기준 세션) | 2026-09-29 |",
            "| A (US 기준 세션) | 2026-09-29 |",
            "입력 snapshot",
            "2026-10-06 14:20:11 KST",
            "건너뛴 시각 검사",
            "입력 완료 시각 \\<= D 09:30",
        ):
            self.assertIn(needle, ds)
        for unit in (UNITS["normal"], UNITS["kr_unavail"]):
            meta, _ = bm.parse_front_matter(text(REPO, unit, "README.md"))
            self.assertNotIn("historical_replay", meta)
            self.assertNotIn("(재현)", meta["title"])

    def test_market_sector_missing_input(self):
        u = UNITS["all_failed"]
        t = text(REPO, u, "market-sector.md")
        self.assertIn("입력 없음", t)
        meta, _ = bm.parse_front_matter(t)
        self.assertEqual((meta["status"], meta["data_asof"]), ("failed", {}))

    def test_data_status_fields(self):
        t = text(REPO, UNITS["normal"], "data-status.md")
        for needle in (
            "## 섹션 상태",
            "## 입력 기준일과 신선도",
            "## 실패",
            "## 게이트와 품질 값",
            "## 출처",
            "`r20261005`",
            "입력 cutoff: 2026-10-07 09:30:00 KST",
            "bundle manifest",
            "prepared manifest",
            "공개 게이트(`publication.status`): `unresolved`",
            "KIS 장중 관측(opening)은 이 리포트 범위 밖입니다.",
        ):
            self.assertIn(needle, t)
        self.assertNotIn("## 재현 정보", t)
        failed = text(REPO, UNITS["all_failed"], "data-status.md")
        self.assertIn("`MissingInference`", failed)
        self.assertIn("`TimeoutError`", failed)

    def test_indexes(self):
        root = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn(bm.AUTO_BEGIN, root)
        self.assertIn("본인 전용", root.split(bm.AUTO_BEGIN)[0])
        self.assertIn("[2026-10-14](reports/daily-briefing/2026/10/2026-10-14/README.md)", root)
        fam = (REPO / "reports/daily-briefing/README.md").read_text(encoding="utf-8")
        meta, _ = bm.parse_front_matter(fam)
        self.assertEqual(
            (meta["cadence"], meta["unit_id_format"], meta["family"]),
            ("daily", "YYYY/MM/YYYY-MM-DD/", "daily-briefing"),
        )
        self.assertIn("owner", meta)
        self.assertEqual(len(re.findall(r"^\| \[\d{4}-\d{2}-\d{2}\]", fam, flags=re.M)), 5)
        month = (REPO / "reports/daily-briefing/2026/10/README.md").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"^\| \[2026-10-\d{2}\]", month, flags=re.M)), 4)
        dates = re.findall(r"^\| \[(\d{4}-\d{2}-\d{2})\]", fam, flags=re.M)
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_reference_files(self):
        glossary = (REPO / "reference/glossary.md").read_text(encoding="utf-8")
        for term in (
            "K′",
            "stale",
            "순위용 점수",
            "historical_replay",
            "cutoff",
            "A",
            "지연 세션 수",
        ):
            self.assertIn(f"| {term} |", glossary)
        card = (REPO / "reference/models/us_exploratory_20260929_r1_ridge.md").read_text(
            encoding="utf-8"
        )
        cards = json.loads(CARDS.read_text(encoding="utf-8"))
        self.assertIn(cards["us_exploratory_20260929_r1_ridge"]["summary"], card)
        ms1 = (REPO / "reference/models/ms1_market_sector.md").read_text(encoding="utf-8")
        self.assertIn("통과 0, 보류 1, 실패 5", ms1)
        self.assertIn("Brier skill -0.255", ms1)


class TestDeterminism(Base):
    def test_rerun_same_inputs_changes_nothing(self):
        repo = self.fresh_copy()
        before = snapshot(repo)
        for name in ORDER:
            code, out, _ = run_main(gen_args(repo, name) + ["--no-validate"])
            self.assertEqual(code, 0)
            self.assertIn("unchanged", out)
        self.assertEqual(snapshot(repo), before)  # 내용과 mtime 모두 그대로

    def test_rerun_with_other_generated_at_changes_nothing(self):
        repo = self.fresh_copy()
        before = snapshot(repo)
        code, out, _ = run_main(
            gen_args(repo, "f1_normal", generated_at="2026-10-07T18:00:00+09:00")
        )
        self.assertEqual(code, 0)
        self.assertIn("unchanged", out)
        self.assertEqual(snapshot(repo), before)

    def test_only_auto_blocks_change_when_a_unit_is_added(self):
        repo = self.fresh_copy()
        for rel, add in (
            ("README.md", "\n사람이 쓴 메모입니다. ~~취소선~~ 50KB~5MB\n"),
            ("reports/daily-briefing/README.md", "\n사람이 쓴 종류 메모입니다.\n"),
            ("reports/daily-briefing/2026/10/README.md", "\n사람이 쓴 월 메모입니다.\n"),
        ):
            p = repo / rel
            t = p.read_text(encoding="utf-8")
            # 자동 구간 앞에 사람이 쓴 줄을 끼워 넣습니다.
            p.write_text(t.replace(bm.AUTO_BEGIN, add + "\n" + bm.AUTO_BEGIN, 1), encoding="utf-8")
        before = snapshot(repo)
        before_text = {p: (repo / p).read_text(encoding="utf-8") for p in before}
        env, ms = mf.scenario_normal()
        day = "2026-10-13"
        env2 = json.loads(
            json.dumps(env).replace("2026-10-07", day).replace("2026-10-06", "2026-10-12")
        )
        ms2 = json.loads(
            json.dumps(ms).replace("2026-10-07", day).replace("2026-10-06", "2026-10-12")
        )
        (INDIR / "e.json").write_text(json.dumps(env2), encoding="utf-8")
        (INDIR / "m.json").write_text(json.dumps(ms2), encoding="utf-8")
        code, out, err = run_main(
            [
                "--repo",
                str(repo),
                "--release",
                "r20261005",
                "--generated-at",
                "2026-10-13T10:03:00+09:00",
                "--envelope",
                str(INDIR / "e.json"),
                "--market-sector",
                str(INDIR / "m.json"),
            ]
        )
        self.assertEqual(code, 1 if "링크 대상이 없습니다" in err else 0, err)
        after = snapshot(repo)
        changed = {p for p in after if before.get(p, (None,))[0] != after[p][0]}
        new_files = {p for p in after if p not in before}
        self.assertEqual(len(new_files), 5)
        self.assertTrue(all("2026-10-13/" in p for p in new_files))
        self.assertEqual(
            changed - new_files,
            {
                "README.md",
                "reports/daily-briefing/README.md",
                "reports/daily-briefing/2026/10/README.md",
            },
        )
        for rel in (
            "README.md",
            "reports/daily-briefing/README.md",
            "reports/daily-briefing/2026/10/README.md",
        ):
            self.assertEqual(
                outside_auto(before_text[rel]),
                outside_auto((repo / rel).read_text(encoding="utf-8")),
                rel,
            )
        self.assertIn(
            "~~취소선~~ 50KB~5MB", (repo / "README.md").read_text(encoding="utf-8")
        )  # 사람이 쓴 곳은 이스케이프하지 않습니다
        self.assertIn(
            "| modeler.reporting.markdown | 6 |", (repo / "README.md").read_text(encoding="utf-8")
        )  # 단위 수 5 -> 6
        self.assertIn(
            "[2026-10-13](2026-10-13/README.md)",
            (repo / "reports/daily-briefing/2026/10/README.md").read_text(encoding="utf-8"),
        )

    def test_conflict_then_correction(self):
        repo = self.fresh_copy()
        env, ms = mf.scenario_normal()
        env["markets"][0]["rankings"][0]["score"] = 0.123456
        e = INDIR / "e.json"
        e.write_text(json.dumps(env), encoding="utf-8")
        m = FIX / "f1_normal_ms.json"
        base = [
            "--repo",
            str(repo),
            "--release",
            "r20261005",
            "--generated-at",
            "2026-10-07T11:00:00+09:00",
            "--envelope",
            str(e),
            "--market-sector",
            str(m),
        ]
        before = snapshot(repo)
        code, out, err = run_main(base)
        self.assertEqual(code, 2)
        self.assertIn("--reason", err)
        self.assertEqual(
            {k: v[0] for k, v in snapshot(repo).items()}, {k: v[0] for k, v in before.items()}
        )
        code, out, err = run_main(
            base + ["--reason", "KR 점수 재계산", "--previous-commit", "abc1234"]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("corrected", out)
        for name in FIVE:
            meta, _ = bm.parse_front_matter(text(repo, "2026-10-07", name))
            self.assertEqual(meta["revision"], 2, name)
        readme = text(repo, "2026-10-07", "README.md")
        self.assertIn(
            "> 정정 r2 (2026-10-07T11:00:00+09:00): KR 점수 재계산. 이전 판: abc1234", readme
        )
        self.assertLess(readme.index("정정 r2"), readme.index("## 섹션 상태"))
        # 같은 입력으로 다시 돌리면 이제는 바뀌는 것이 없습니다.
        before = snapshot(repo)
        code, out, _ = run_main(base)
        self.assertEqual((code, "unchanged" in out), (0, True))
        self.assertEqual(snapshot(repo), before)
        # 한 번 더 고치면 r3이고 정정 줄이 쌓입니다.
        env["markets"][0]["rankings"][1]["score"] = 0.111111
        e.write_text(json.dumps(env), encoding="utf-8")
        code, _, err = run_main(base + ["--reason", "두 번째 정정"])
        self.assertEqual(code, 0, err)
        readme = text(repo, "2026-10-07", "README.md")
        self.assertIn("정정 r3", readme)
        self.assertIn("정정 r2", readme)
        self.assertLess(readme.index("정정 r3"), readme.index("정정 r2"))
        fam = (repo / "reports/daily-briefing/README.md").read_text(encoding="utf-8")
        self.assertIn("r3", fam)
        self.assertEqual(bm.validate(repo), [])
        # --overwrite는 revision을 올리지 않습니다.
        env["markets"][0]["rankings"][2]["score"] = 0.101010
        e.write_text(json.dumps(env), encoding="utf-8")
        code, out, _ = run_main(base + ["--overwrite"])
        self.assertEqual(code, 0)
        self.assertEqual(
            bm.parse_front_matter(text(repo, "2026-10-07", "kr-stocks.md"))[0]["revision"], 3
        )


class TestInputs(Base):
    def run_env(self, env, ms=None, name="x"):
        repo = self.fresh_copy()
        d = INDIR
        e = d / "e.json"
        e.write_text(json.dumps(env), encoding="utf-8")
        argv = [
            "--repo",
            str(repo),
            "--release",
            "r20261005",
            "--generated-at",
            "2026-10-20T10:00:00+09:00",
            "--envelope",
            str(e),
        ]
        if ms is not None:
            m = d / "m.json"
            m.write_text(json.dumps(ms), encoding="utf-8")
            argv += ["--market-sector", str(m)]
        code, out, err = run_main(argv)
        return repo, code, out, err

    def test_internal_states_withheld_and_unavailable(self):
        day = "2026-10-15"
        kr = mf.kr_report(day, "2026-10-14", status="partial")
        kr["status"] = "withheld"
        us1 = mf.us_report(day, mf.LGB_ID, "2026-10-14", "2026-10-14", "2026-10-14")
        us2 = mf.us_report(day, mf.RDG_ID, "2026-10-14", "2026-10-14", "2026-10-14", status="ok")
        us2["status"] = "unavailable"
        us2["rankings"] = []
        us2["provenance"]["freshness"]["reason"] = "US delivery lag reached the stop limit"
        env = mf.envelope(day, [kr, us1, us2], [])
        repo, code, out, err = self.run_env(env, mf.ms_input(day))
        self.assertEqual(code, 0, err)
        meta, _ = bm.parse_front_matter(text(repo, day, "kr-stocks.md"))
        self.assertEqual(meta["status"], "failed")
        self.assertIn("공개 보류", text(repo, day, "kr-stocks.md"))
        self.assertNotIn("| 순위 | 코드", text(repo, day, "kr-stocks.md"))
        us = text(repo, day, "us-stocks.md")
        self.assertEqual(
            bm.parse_front_matter(us)[0]["status"], "partial"
        )  # 한 모델 정상 + 한 모델 failed
        self.assertIn("미국 도착 지연이 중단 한도에 닿았습니다", us)
        self.assertIn("(내부 상태: 자료 없음)", us)

    def test_invalid_report_becomes_failure_not_crash(self):
        day = "2026-10-15"
        bad = mf.kr_report(day, "2026-10-14")
        bad["rankings"][1]["rank"] = 1  # 순위가 증가하지 않음
        env = {
            "schema_version": "1.0",
            "report_date": day,
            "decision_at": mf.decision_at(day).isoformat(),
            "status": "failed",
            "markets": [bad],
            "failures": [],
            "opening": {},
            "synthetic_fixture": False,
            "historical_replay": False,
        }
        repo, code, out, err = self.run_env(env)
        self.assertEqual(code, 0, err)
        self.assertIn("InvalidReport", text(repo, day, "data-status.md"))
        self.assertEqual(
            bm.parse_front_matter(text(repo, day, "kr-stocks.md"))[0]["status"], "failed"
        )

    def test_schema_compat_with_ops_validate_report(self):
        for builder in mf.SCENARIOS.values():
            env, _ = builder()
            for report in env["markets"]:
                bm.validate_report_min(report)
                mf.OPS.validate_report(report)
        report = copy.deepcopy(mf.scenario_normal()[0]["markets"][0])
        report["rankings"][1]["rank"] = 1
        with self.assertRaises(ValueError):
            mf.OPS.validate_report(report)
        with self.assertRaises(ValueError):
            bm.validate_report_min(report)

    def test_market_sector_edge_cases(self):
        day = "2026-10-15"
        env = mf.envelope(
            day,
            [
                mf.kr_report(day, "2026-10-14"),
                mf.us_report(day, mf.LGB_ID, "2026-10-14", "2026-10-14", "2026-10-14"),
                mf.us_report(day, mf.RDG_ID, "2026-10-14", "2026-10-14", "2026-10-14"),
            ],
            [],
        )
        ms = mf.ms_input(day)
        bad_date = copy.deepcopy(ms)
        bad_date["report_date"] = "2026-10-14"
        repo, code, _, _ = self.run_env(env, bad_date)
        meta, _ = bm.parse_front_matter(text(repo, day, "market-sector.md"))
        self.assertEqual(meta["status"], "failed")
        self.assertIn("입력 날짜 불일치", text(repo, day, "market-sector.md"))
        no_kr = copy.deepcopy(ms)
        no_kr["assets"] = [a for a in no_kr["assets"] if a["market"] == "US"]
        repo, code, _, _ = self.run_env(env, no_kr)
        self.assertEqual(
            bm.parse_front_matter(text(repo, day, "market-sector.md"))[0]["status"], "partial"
        )
        self.assertIn("KR 자산 행이 없습니다", text(repo, day, "market-sector.md"))
        null_vals = copy.deepcopy(ms)
        null_vals["assets"][2]["ret_20"] = None
        repo, code, _, _ = self.run_env(env, null_vals)
        t = text(repo, day, "market-sector.md")
        self.assertEqual(bm.parse_front_matter(t)[0]["status"], "partial")
        self.assertIn("ret_20 값이 없습니다", t)

    def test_score_asof_date_override(self):
        day = "2026-10-15"
        env = mf.envelope(day, [mf.kr_report(day, "2026-10-14")], [])
        ms = mf.ms_input(day)
        for a in ms["assets"]:
            a["score_asof_date"] = "2026-09-25" if a["market"] == "US" else "2026-09-28"
        repo, code, _, err = self.run_env(env, ms)
        self.assertEqual(code, 0, err)
        t = text(repo, day, "market-sector.md")
        details = t.split("<details>", 1)[1]
        self.assertIn("| SPY (S&P 500) | US | 2026-09-25 |", details)
        self.assertIn("| 코스피 | KR | 2026-09-28 |", details)
        self.assertIn("점수 기준일이 위 시장 상태 표의 기준 종가일과 다릅니다", details)
        self.assertIn("| SPY (S&P 500) | 배당 포함 | 2026-10-15 |", t.split("<details>", 1)[0])

    def test_scrub_leaks(self):
        day = "2026-10-15"
        kr = mf.kr_report(day, "2026-10-14")
        kr["rankings"][0]["name"] = "누수/home/whi/secret/key.pem"
        kr["rankings"][1]["name"] = "host sj2-server ghp_abcdef api_key=zzz"
        env = mf.envelope(day, [kr], [])
        repo, code, out, err = self.run_env(env, mf.ms_input(day))
        self.assertEqual(code, 0, err)
        self.assertIn("본문에서 지웠습니다", err)
        allt = "".join(p.read_text(encoding="utf-8") for p in unit_path(repo, day).glob("*.md"))
        for needle in ("/home/", "sj2", "ghp_", "api_key"):
            self.assertNotIn(needle, allt)
        self.assertEqual(bm.validate(repo), [])

    def test_decision_time_must_be_10am(self):
        env, _ = mf.scenario_normal()
        env["decision_at"] = "2026-10-07T09:00:00+09:00"
        repo, code, out, err = self.run_env(env)
        self.assertEqual(code, 2)
        self.assertIn("10:00 KST", err)

    def test_top_n_option(self):
        repo = self.fresh_copy()
        code, out, err = run_main(
            gen_args(repo, "f1_normal", generated_at="2026-10-07T10:03:12+09:00", top_n=30)
            + ["--overwrite"]
        )
        self.assertEqual(code, 0, err)
        t = text(repo, "2026-10-07", "kr-stocks.md")
        self.assertEqual(
            len([ln for ln in t.split("\n") if re.match(r"\| \d+ \| \d{6} \|", ln)]), 30
        )
        self.assertIn("두 모델 상위 30의 겹침", text(repo, "2026-10-07", "us-stocks.md"))


class TestValidate(Base):
    def broken(self, mutate):
        repo = self.fresh_copy()
        mutate(repo)
        return bm.validate(repo)

    def append(self, rel, extra):
        def mutate(repo):
            p = repo / rel
            p.write_text(p.read_text(encoding="utf-8") + extra, encoding="utf-8")

        return mutate

    def test_forbidden_strings(self):
        rel = "reports/daily-briefing/2026/10/2026-10-07/kr-stocks.md"
        for extra, label in (
            ("\n경로 /home/whi/apps\n", "/home/"),
            ("\n/Users/whishaw/x\n", "/Users/"),
            ("\n/private/tmp/x\n", "/private/"),
            ("\nssh sj2-server\n", "sj2"),
            ("\ntoken ghp_abcdef\n", "gh"),
            ("\n-----BEGIN PRIVATE KEY-----\n", "PEM"),
            ("\nAPI_KEY=1\n", "api_key"),
        ):
            problems = self.broken(self.append(rel, extra))
            self.assertTrue(
                any("금지 문자열" in p and rel in p for p in problems), (label, problems)
            )

    def test_path_and_extension_rules(self):
        problems = self.broken(lambda r: (r / "data.csv").write_text("a,b\n"))
        self.assertTrue(any("허용 경로 밖" in p for p in problems), problems)
        problems = self.broken(
            lambda r: (r / "reports/daily-briefing/2026/10/2026-10-07/x.json").write_text("{}")
        )
        self.assertTrue(any(".md만" in p for p in problems), problems)
        problems = self.broken(
            lambda r: (r / "reports/other-family").mkdir()
            or (r / "reports/other-family/README.md").write_text("x")
        )
        self.assertTrue(any("허용 경로 밖" in p for p in problems), problems)
        problems = self.broken(lambda r: (r / "reference/img.png").write_bytes(b"x"))
        self.assertTrue(any(".md만" in p for p in problems), problems)

    def test_size_limit(self):
        problems = self.broken(lambda r: (r / "reference/big.md").write_text("가" * (400 * 1024)))
        self.assertTrue(any("1MB" in p for p in problems), problems)
        problems = self.broken(lambda r: (r / "reference/ok.md").write_text("a" * (1024 * 1024)))
        self.assertEqual(problems, [])  # 정확히 1MB는 통과

    def test_symlink(self):
        def mutate(repo):
            (repo / "reference/link.md").symlink_to(repo / "README.md")

        self.assertTrue(any("symlink" in p for p in self.broken(mutate)))

    def test_front_matter_rules(self):
        rel = "reports/daily-briefing/2026/10/2026-10-07/kr-stocks.md"

        def sub(old, new):
            def mutate(repo):
                p = repo / rel
                t = p.read_text(encoding="utf-8")
                assert old in t, old
                p.write_text(t.replace(old, new, 1), encoding="utf-8")

            return mutate

        cases = (
            (("family: daily-briefing", "family: other"), "family"),
            (("unit: 2026-10-07", "unit: 2026-10-08"), "unit"),
            (("schema: stock-reports.v1", "schema: stock-reports.v2"), "schema"),
            (("status: partial", "status: done"), "status"),
            (("section: kr-stocks", "section: us-stocks"), "section"),
            (("revision: 1", "revision: 0"), "revision"),
            (("markets: [KR]", "markets: [JP]"), "markets"),
            (
                (
                    "decision_at: 2026-10-07T10:00:00+09:00",
                    "decision_at: 2026-10-07T11:00:00+09:00",
                ),
                "decision_at",
            ),
        )
        for (old, new), key in cases:
            problems = self.broken(sub(old, new))
            self.assertTrue(any(rel in p and key in p for p in problems), (key, problems))
        problems = self.broken(sub("models: [kr_daily_h20_v1]\n", ""))
        self.assertTrue(any("필드 없음: models" in p for p in problems), problems)
        problems = self.broken(sub("  report_sha256: ", "  report_sha256x: "))
        self.assertTrue(any("source" in p for p in problems), problems)

        def drop_fm(repo):
            p = repo / rel
            p.write_text(p.read_text(encoding="utf-8").split("\n---\n", 1)[1], encoding="utf-8")

        self.assertTrue(any("front matter가 없습니다" in p for p in self.broken(drop_fm)))

    def test_family_readme_fields(self):
        def mutate(repo):
            p = repo / "reports/daily-briefing/README.md"
            p.write_text(
                p.read_text(encoding="utf-8").replace("cadence: daily\n", ""), encoding="utf-8"
            )

        self.assertTrue(any("필드 없음: cadence" in p for p in self.broken(mutate)))

    def test_links(self):
        rel = "reports/daily-briefing/2026/10/2026-10-07/README.md"
        problems = self.broken(self.append(rel, "\n[없는 파일](nope.md)\n"))
        self.assertTrue(any("링크 대상이 없습니다: nope.md" in p for p in problems), problems)
        problems = self.broken(self.append(rel, "\n[밖](../../../../../../x.md)\n"))
        self.assertTrue(any("링크 대상이 없습니다" in p for p in problems), problems)
        problems = self.broken(self.append(rel, "\n[절대](/reference/README.md)\n"))
        self.assertTrue(any("절대 경로 링크" in p for p in problems), problems)
        problems = self.broken(
            self.append(rel, "\n[앵커](#a) [외부](https://example.com/x) [ok](kr-stocks.md#top)\n")
        )
        self.assertEqual(problems, [])
        problems = self.broken(
            self.append(rel, "\n```\n[코드 안](nope.md)\n```\n`[인라인](nope2.md)`\n")
        )
        self.assertEqual(problems, [])
        problems = self.broken(lambda r: (r / "reference/models/kr_daily_h20_v1.md").unlink())
        self.assertTrue(any("kr_daily_h20_v1.md" in p for p in problems), problems)

    def test_missing_unit_file(self):
        problems = self.broken(
            lambda r: (r / "reports/daily-briefing/2026/10/2026-10-07/us-stocks.md").unlink()
        )
        self.assertTrue(
            any("단위에 파일이 빠졌습니다: us-stocks.md" in p for p in problems), problems
        )

    def test_mixed_replay_flags(self):
        def mutate(repo):
            p = repo / "reports/daily-briefing/2026/10/2026-10-07/kr-stocks.md"
            p.write_text(
                p.read_text(encoding="utf-8").replace(
                    "---\n\n# ", "historical_replay: true\n---\n\n# ", 1
                ),
                encoding="utf-8",
            )

        problems = self.broken(mutate)
        self.assertTrue(
            any("historical_replay가 파일마다 다릅니다" in p for p in problems), problems
        )

    def test_cli_exit_codes(self):
        repo = self.fresh_copy()
        env = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
        cmd = [sys.executable, "-m", "modeler.reporting.markdown"]
        ok = subprocess.run(
            [*cmd, "--repo", str(repo), "--validate"], capture_output=True, text=True, env=env
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        (repo / "reference/glossary.md").write_text("서버 /home/whi\n", encoding="utf-8")
        bad = subprocess.run(
            [*cmd, "--repo", str(repo), "--validate"], capture_output=True, text=True, env=env
        )
        self.assertEqual(bad.returncode, 1)
        self.assertIn("검증 위반", bad.stderr)
        usage = subprocess.run([*cmd, "--repo", str(repo)], capture_output=True, text=True, env=env)
        self.assertEqual(usage.returncode, 2)
        norel = subprocess.run(
            [*cmd, "--repo", str(repo), "--envelope", str(FIX / "f1_normal_envelope.json")],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(norel.returncode, 2)

    def test_init_does_not_overwrite(self):
        repo = self.fresh_copy()
        (repo / "CONVENTIONS.md").write_text("# 내가 고친 규칙\n", encoding="utf-8")
        (repo / "reference/glossary.md").write_text("# 내가 고친 용어\n", encoding="utf-8")
        code, out, _ = run_main(["--repo", str(repo), "--init", "--model-cards", str(CARDS)])
        self.assertEqual(code, 0)
        self.assertIn("이미 있어 건너뜁니다: CONVENTIONS.md", out)
        self.assertEqual(
            (repo / "CONVENTIONS.md").read_text(encoding="utf-8"), "# 내가 고친 규칙\n"
        )
        self.assertEqual(
            (repo / "reference/glossary.md").read_text(encoding="utf-8"), "# 내가 고친 용어\n"
        )


class TestHelpers(unittest.TestCase):
    def test_front_matter_roundtrip(self):
        meta = {
            "schema": "stock-reports.v1",
            "title": '제목: 콜론 "따옴표" — 2026',
            "markets": ["KR", "US"],
            "models": [],
            "data_asof": {},
            "revision": 3,
            "source": {"release": "r1", "report_sha256": "a" * 64},
            "flag": True,
        }
        parsed, body = bm.parse_front_matter(bm.front_matter(meta) + "\n# 본문\n")
        self.assertEqual(parsed, meta)
        self.assertEqual(body, "\n# 본문\n")

    def test_escape_helpers(self):
        self.assertEqual(bm.c("a|b\nc"), "a\\|b<br>c")
        self.assertEqual(bm.c(""), "-")
        self.assertEqual(bm.escape_tilde("50KB~5MB `a~b` ~~x~~"), "50KB\\~5MB `a~b` \\~\\~x\\~\\~")
        self.assertEqual(bm.escape_tilde("```\na~b\n```\nc~d"), "```\na~b\n```\nc\\~d")
        self.assertEqual(bm.escape_tilde("이미 \\~ 처리됨"), "이미 \\~ 처리됨")
        self.assertEqual(bm.pct(0.0012), "+0.12%")
        self.assertEqual(bm.pct(-0.031), "-3.10%")
        self.assertEqual(bm.pct_plain(1.1), "1.10%")
        self.assertEqual(bm.pct(None), "-")

    def test_replay_aliases(self):
        rows = dict(
            bm.replay_rows(
                {
                    "k": "2026-09-29",
                    "a": "2026-09-29",
                    "snapshot": "2026-10-06",
                    "ran_at": "2026-10-06T14:20:11+09:00",
                    "skipped_checks": ["x"],
                    "extra": 1,
                }
            )
        )
        self.assertEqual(rows["K (KR 기준 세션)"], "2026-09-29")
        self.assertEqual(rows["실제 실행 시각"], "2026-10-06 14:20:11 KST")
        self.assertEqual(rows["건너뛴 시각 검사"], "x")
        self.assertEqual(rows["`extra`"], "1")
        self.assertEqual(dict(bm.replay_rows({}))["입력 snapshot"], "기록 없음")


if __name__ == "__main__":
    unittest.main(verbosity=2)
