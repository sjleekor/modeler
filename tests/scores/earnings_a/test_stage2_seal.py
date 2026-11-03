"""Synthetic-only tests for the stage 2 seal phases (STAGE2_PAST and FORWARD)."""

import json
from datetime import date

import pandas as pd
import pytest

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.seal import Phase, SealViolation


def _runner(tag=seal.STAGE2_TAG, present=True, annotated=True):
    def run(args):
        a = list(args)
        if a[0] == "tag":
            return 0, (f"{tag}\n" if present and a[-1] == tag else "")
        if a[0] == "cat-file":
            return 0, ("tag\n" if annotated else "commit\n")
        if a[0] == "rev-list":
            return 0, "aaaa\n"
        if a[0] == "rev-parse":
            return 0, "bbbb\n"
        return 1, ""

    return run


def test_stage2_past_dates_are_25_quarter_ends_and_train_rows_stop_at_sealed():
    d = seal.STAGE2_PAST_DATES
    assert len(d) == 25 and d[0] == date(2019, 6, 30) and d[-1] == date(2025, 6, 30)
    allowed = seal.allowed_decision_dates(Phase.STAGE2_PAST)
    assert allowed == seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES
    assert set(d) <= set(allowed)
    assert not set(seal.CONFIRM_DATES) & set(allowed)
    with pytest.raises(SealViolation, match="2025-09-30"):
        seal.check_decision_dates([date(2025, 9, 30)], Phase.STAGE2_PAST)


def test_forward_eval_dates_follow_today():
    f = seal.forward_eval_dates
    assert f(date(2026, 9, 29)) == ()
    assert f(date(2026, 9, 30)) == (date(2026, 9, 30),)
    assert f(date(2026, 11, 3)) == (date(2026, 9, 30),)
    assert f(date(2027, 1, 1)) == (date(2026, 9, 30), date(2026, 12, 31))
    assert f(date(2027, 4, 1))[-1] == date(2027, 3, 31)


def test_stage1_phases_unchanged():
    assert seal.allowed_decision_dates(Phase.DEV) == seal.DEV_DATES
    assert len(seal.allowed_decision_dates(Phase.SEALED_RUN)) == 38
    assert len(seal.allowed_decision_dates(Phase.CONFIRM_RUN)) == 42
    assert seal.SEAL_TAG == "earnings-a-frozen"
    assert seal.DEV_DATES[0] == date(2016, 3, 31) and seal.CONFIRM_DATES[-1] == date(2026, 6, 30)


def test_stage2_past_needs_its_own_tag(tmp_path):
    led = tmp_path / "l.jsonl"
    # the stage 1 tag does not open stage 2
    with pytest.raises(SealViolation, match="earnings-a2-frozen"):
        seal.open_seal(
            Phase.STAGE2_PAST,
            repo_dir=tmp_path,
            ledger_path=led,
            git_runner=_runner(tag=seal.SEAL_TAG),
        )
    assert not led.exists()
    with pytest.raises(SealViolation, match="annotated"):
        seal.open_seal(
            Phase.STAGE2_PAST,
            repo_dir=tmp_path,
            ledger_path=led,
            git_runner=_runner(annotated=False),
        )
    assert not led.exists()


def test_stage2_past_opens_once_unless_reopen_is_recorded(tmp_path):
    led = tmp_path / "sub" / "l.jsonl"
    kw = dict(repo_dir=tmp_path, ledger_path=led, git_runner=_runner())
    rec = seal.open_seal(Phase.STAGE2_PAST, **kw)
    assert rec["phase"] == "STAGE2_PAST" and rec["tag"] == "earnings-a2-frozen"
    assert rec["reopen"] is False and rec["tag_commit"] == "aaaa"
    with pytest.raises(SealViolation, match="already opened once"):
        seal.open_seal(Phase.STAGE2_PAST, **kw)
    rec2 = seal.open_seal(Phase.STAGE2_PAST, allow_reopen=True, **kw)
    assert rec2["reopen"] is True
    lines = [json.loads(x) for x in led.read_text().splitlines()]
    assert [r["phase"] for r in lines] == ["STAGE2_PAST", "STAGE2_PAST"]
    # stage 1 phases keep their own counters in the same ledger
    r1 = seal.open_seal(
        Phase.SEALED_RUN, repo_dir=tmp_path, ledger_path=led, git_runner=_runner(tag=seal.SEAL_TAG)
    )
    assert r1["reopen"] is False and r1["tag"] == "earnings-a-frozen"


def test_stage2_past_grant_allows_train_dates_only(tmp_path):
    g = seal.open_seal_grant(
        Phase.STAGE2_PAST,
        repo_dir=tmp_path,
        ledger_path=tmp_path / "l.jsonl",
        git_runner=_runner(),
    )
    assert g.allowed_dates == seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES

    @seal.guarded
    def build(*, decision_dates, grant):
        return pd.DataFrame({"decision_date": list(decision_dates)})

    build(decision_dates=list(seal.STAGE2_PAST_DATES), grant=g)
    with pytest.raises(SealViolation, match="2025-09-30"):
        build(decision_dates=[date(2025, 9, 30)], grant=g)


def test_forward_grant_one_record_per_decision_date(tmp_path):
    led = tmp_path / "l.jsonl"
    kw = dict(repo_dir=tmp_path, ledger_path=led, git_runner=_runner(), today=date(2027, 1, 5))
    r1 = seal.open_forward(date(2026, 9, 30), **kw)
    assert r1["phase"] == "FORWARD" and r1["decision_date"] == "2026-09-30"
    with pytest.raises(SealViolation, match="already granted once"):
        seal.open_forward(date(2026, 9, 30), **kw)
    r2 = seal.open_forward(date(2026, 12, 31), **kw)  # another date is independent
    assert r2["reopen"] is False
    r3 = seal.open_forward(date(2026, 9, 30), allow_reopen=True, **kw)
    assert r3["reopen"] is True
    rows = [json.loads(x) for x in led.read_text().splitlines()]
    assert [(r["phase"], r["decision_date"]) for r in rows] == [
        ("FORWARD", "2026-09-30"),
        ("FORWARD", "2026-12-31"),
        ("FORWARD", "2026-09-30"),
    ]


def test_forward_rejects_non_forward_dates_and_missing_tag(tmp_path):
    led = tmp_path / "l.jsonl"
    kw = dict(repo_dir=tmp_path, ledger_path=led, git_runner=_runner(), today=date(2026, 11, 3))
    for bad in (date(2026, 6, 30), date(2026, 12, 31), date(2026, 10, 1)):
        with pytest.raises(SealViolation, match="not a FORWARD decision date"):
            seal.open_forward(bad, **kw)
    with pytest.raises(SealViolation, match="not found"):
        seal.open_forward(date(2026, 9, 30), **{**kw, "git_runner": _runner(present=False)})
    with pytest.raises(SealViolation, match="earnings-a2-frozen"):
        seal.open_forward(date(2026, 9, 30), **{**kw, "git_runner": _runner(tag=seal.SEAL_TAG)})
    assert not led.exists()


def test_forward_grant_dates_exclude_confirm_unless_ledger_covers_it(tmp_path):
    led = tmp_path / "l.jsonl"
    kw = dict(repo_dir=tmp_path, ledger_path=led, git_runner=_runner(), today=date(2027, 1, 5))
    g = seal.open_forward_grant(date(2026, 9, 30), **kw)
    base = seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES
    assert g.allowed_dates == base + (date(2026, 9, 30),)  # no CONFIRM date, no later forward date
    assert not set(seal.CONFIRM_DATES) & set(g.allowed_dates)
    assert date(2026, 12, 31) not in g.allowed_dates
    # a CONFIRM_RUN record that lists a scored date covers exactly that date
    led.write_text(
        json.dumps({"phase": "CONFIRM_RUN", "scored_dates": ["2025-09-30"]}) + "\n",
        encoding="utf-8",
    )
    assert seal.covered_confirm_dates(led) == (date(2025, 9, 30),)
    g2 = seal.open_forward_grant(date(2026, 12, 31), **kw)
    assert date(2025, 9, 30) in g2.allowed_dates and date(2025, 12, 31) not in g2.allowed_dates
    # a CONFIRM_RUN record without scored_dates covers nothing
    led2 = tmp_path / "l2.jsonl"
    led2.write_text(json.dumps({"phase": "CONFIRM_RUN"}) + "\n", encoding="utf-8")
    assert seal.covered_confirm_dates(led2) == ()


def test_forward_grant_is_guarded_to_its_own_date(tmp_path):
    g = seal.open_forward_grant(
        date(2026, 9, 30),
        repo_dir=tmp_path,
        ledger_path=tmp_path / "l.jsonl",
        git_runner=_runner(),
        today=date(2027, 1, 5),
    )

    @seal.guarded
    def build(*, decision_dates, grant):
        return pd.DataFrame({"decision_date": list(decision_dates)})

    build(decision_dates=[date(2023, 6, 30), date(2026, 9, 30)], grant=g)
    for bad in (date(2026, 12, 31), date(2025, 9, 30)):
        with pytest.raises(SealViolation, match=bad.isoformat()):
            build(decision_dates=[bad], grant=g)


def test_grant_dev_still_works_for_reproduction():
    g = seal.grant_dev()
    assert g.phase is Phase.DEV and g.allowed_dates == seal.DEV_DATES


def test_forward_grant_keeps_earlier_forward_dates_as_training_rows(tmp_path):
    kw = dict(repo_dir=tmp_path, ledger_path=tmp_path / "l.jsonl", git_runner=_runner())
    today = date(2028, 4, 5)
    g = seal.open_forward_grant(date(2028, 3, 31), today=today, **kw)
    fwd = (date(2026, 9, 30), date(2026, 12, 31), date(2027, 3, 31), date(2027, 6, 30),
           date(2027, 9, 30), date(2027, 12, 31), date(2028, 3, 31))  # fmt: skip
    base = seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES
    assert g.allowed_dates == base + fwd
    train = seal.forward_train_dates(date(2028, 3, 31), today=today, ledger_path=kw["ledger_path"])
    assert train == base + fwd[:-1]
    # a middle date does not see later forward dates
    g2 = seal.open_forward_grant(date(2027, 3, 31), today=today, **kw)
    assert g2.allowed_dates == base + fwd[:3]
