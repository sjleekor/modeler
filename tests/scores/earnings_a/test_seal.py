"""Synthetic-only tests for the research A seal guard."""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.seal import Phase, SealViolation


def test_window_counts():
    assert len(seal.DEV_DATES) == 26
    assert len(seal.GAP_DATES) == 3
    assert len(seal.SEALED_DATES) == 9
    assert len(seal.CONFIRM_DATES) == 4
    assert seal.GAP_DATES[0] == date(2022, 9, 30)
    assert seal.SEALED_DATES[-1] == date(2025, 6, 30)


def test_allowed_sets():
    assert len(seal.allowed_decision_dates(Phase.DEV)) == 26
    assert len(seal.allowed_decision_dates(Phase.SEALED_RUN)) == 38
    assert len(seal.allowed_decision_dates(Phase.CONFIRM_RUN)) == 42
    with pytest.raises(NotImplementedError):
        seal.allowed_decision_dates(Phase.FORWARD)


def test_phase_accept_reject():
    seal.check_decision_dates([date(2016, 3, 31), date(2022, 6, 30)], Phase.DEV)
    for d in (date(2022, 9, 30), date(2023, 6, 30), date(2025, 9, 30)):
        with pytest.raises(SealViolation, match=d.isoformat()):
            seal.check_decision_dates([d], Phase.DEV)
    seal.check_decision_dates([date(2022, 12, 31), date(2023, 6, 30)], Phase.SEALED_RUN)
    with pytest.raises(SealViolation):
        seal.check_decision_dates([date(2025, 9, 30)], Phase.SEALED_RUN)
    seal.check_decision_dates([date(2026, 6, 30)], Phase.CONFIRM_RUN)
    with pytest.raises(SealViolation):
        seal.check_decision_dates([date(2026, 9, 30)], Phase.CONFIRM_RUN)
    with pytest.raises(SealViolation):
        seal.check_decision_dates([date(2015, 12, 31)], Phase.CONFIRM_RUN)


def test_non_quarter_end_rejected():
    with pytest.raises(SealViolation):
        seal.check_decision_dates([date(2020, 3, 30)], Phase.DEV)


def test_input_types():
    seal.check_decision_dates(
        [
            "2020-03-31",
            pd.Timestamp("2020-06-30"),
            np.datetime64("2020-09-30"),
            datetime(2020, 12, 31, 12, 0),
        ],
        Phase.DEV,
    )
    with pytest.raises(SealViolation):
        seal.check_decision_dates(["2023-06-30"], Phase.DEV)


def test_check_frame():
    ok = pd.DataFrame({"decision_date": pd.to_datetime(["2020-03-31"]), "secret": [1.5]})
    seal.check_frame(ok, Phase.DEV)
    with pytest.raises(SealViolation, match="missing"):
        seal.check_frame(ok.drop(columns="decision_date"), Phase.DEV)
    bad = pd.DataFrame({"decision_date": pd.to_datetime(["2023-06-30"]), "secret": [987.654]})
    with pytest.raises(SealViolation) as ei:
        seal.check_frame(bad, Phase.DEV)
    assert "987.654" not in str(ei.value)
    empty = pd.DataFrame({"decision_date": pd.to_datetime(["2020-03-31", None])})
    with pytest.raises(SealViolation, match="empty"):
        seal.check_frame(empty, Phase.DEV)


def test_label_quarters_boundary():
    ok = pd.DataFrame({"label_last_quarter_end": pd.to_datetime(["2023-06-30", "2020-03-31"])})
    seal.check_label_quarters(ok, Phase.DEV)
    bad = pd.DataFrame({"label_last_quarter_end": pd.to_datetime(["2023-09-30"])})
    with pytest.raises(SealViolation, match="2023-09-30"):
        seal.check_label_quarters(bad, Phase.DEV)
    seal.check_label_quarters(bad, Phase.SEALED_RUN)


def _runner(tag_present=True, annotated=True):
    def run(args):
        a = list(args)
        if a[0] == "tag":
            return 0, (f"{seal.SEAL_TAG}\n" if tag_present else "")
        if a[0] == "cat-file":
            return 0, ("tag\n" if annotated else "commit\n")
        if a[0] == "rev-list":
            return 0, "aaaa\n"
        if a[0] == "rev-parse":
            return 0, "bbbb\n"
        return 1, ""

    return run


def test_open_seal_missing_tag(tmp_path):
    led = tmp_path / "l.jsonl"
    with pytest.raises(SealViolation, match="not found"):
        seal.open_seal(
            Phase.SEALED_RUN,
            repo_dir=tmp_path,
            ledger_path=led,
            git_runner=_runner(tag_present=False),
        )
    assert not led.exists()
    with pytest.raises(SealViolation, match="annotated"):
        seal.open_seal(
            Phase.SEALED_RUN,
            repo_dir=tmp_path,
            ledger_path=led,
            git_runner=_runner(annotated=False),
        )


def test_open_seal_flow(tmp_path):
    led = tmp_path / "sub" / "l.jsonl"
    kw = dict(repo_dir=tmp_path, ledger_path=led, git_runner=_runner())
    rec = seal.open_seal(Phase.SEALED_RUN, **kw)
    assert rec["reopen"] is False
    assert rec["tag_commit"] == "aaaa" and rec["head_commit"] == "bbbb"
    assert len(led.read_text().splitlines()) == 1
    with pytest.raises(SealViolation, match="already opened once"):
        seal.open_seal(Phase.SEALED_RUN, **kw)
    # other phase is independent
    assert seal.open_seal(Phase.CONFIRM_RUN, **kw)["reopen"] is False
    rec2 = seal.open_seal(Phase.SEALED_RUN, allow_reopen=True, **kw)
    assert rec2["reopen"] is True
    assert len(led.read_text().splitlines()) == 3
    with pytest.raises(SealViolation):
        seal.open_seal(Phase.DEV, **kw)


def test_sealed_digest():
    df = pd.DataFrame({"a": [1, 2, 3], "y": [0.1, 0.2, 0.30000000000000004]})
    d = seal.sealed_digest(df, ["a", "y"])
    assert len(d) == 64
    assert d == seal.sealed_digest(df.iloc[[2, 0, 1]], ["a", "y"])
    ch = df.copy()
    ch.loc[0, "y"] = 0.1000000001
    assert d != seal.sealed_digest(ch, ["a", "y"])
    assert d != seal.sealed_digest(df, ["a"])


def test_ledger_path_required_and_outside_repo(tmp_path, monkeypatch):
    import inspect

    sig = inspect.signature(seal.open_seal)
    assert sig.parameters["ledger_path"].default is inspect.Parameter.empty
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path / "sd"))
    p = seal.default_ledger_path()
    assert p == tmp_path / "sd/kr/output/earnings_forecast_a/seal_ledger.jsonl"
    repo = Path(seal.__file__).resolve().parents[4]
    assert repo not in p.parents


def test_open_seal_real_git(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    led = tmp_path / "led.jsonl"
    git("tag", seal.SEAL_TAG)  # lightweight tag is refused
    with pytest.raises(SealViolation, match="annotated"):
        seal.open_seal(Phase.SEALED_RUN, repo_dir=repo, ledger_path=led)
    git("tag", "-d", seal.SEAL_TAG)
    git("-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", seal.SEAL_TAG, "-m", "frozen")
    rec = seal.open_seal(Phase.SEALED_RUN, repo_dir=repo, ledger_path=led)
    assert rec["tag_commit"] == rec["head_commit"] and len(rec["tag_commit"]) == 40
    with pytest.raises(SealViolation, match="already opened once"):
        seal.open_seal(Phase.SEALED_RUN, repo_dir=repo, ledger_path=led)
