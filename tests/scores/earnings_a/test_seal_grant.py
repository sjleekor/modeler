"""Synthetic-only tests for SealGrant, guarded and digest_only."""

import dataclasses
import gc
import json
import weakref
from datetime import date

import pandas as pd
import pytest

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.seal import Phase, SealGrant, SealViolation

DEV = seal.DEV_DATES[0]
GAP = seal.GAP_DATES[0]
SEALED = seal.SEALED_DATES[0]
CONFIRM = seal.CONFIRM_DATES[0]


def _frame(dates, **extra):
    df = pd.DataFrame(
        {"decision_date": list(dates), "ticker": [f"T{i}" for i in range(len(dates))]}
    )
    for k, v in extra.items():
        df[k] = v
    return df


def test_forged_grant_raises():
    with pytest.raises(SealViolation):
        SealGrant(Phase.SEALED_RUN, None)
    with pytest.raises(SealViolation):
        SealGrant(Phase.DEV, None, object())


def test_allowed_dates():
    assert seal.grant_dev().allowed_dates == seal.DEV_DATES
    assert len(seal._mint(seal.SealGrant, Phase.SEALED_RUN, None).allowed_dates) == 38
    assert len(seal._mint(seal.SealGrant, Phase.CONFIRM_RUN, None).allowed_dates) == 42


def _make_guarded():
    state = {"ran": False}

    @seal.guarded
    def build(*, decision_dates, grant, frame=None):
        state["ran"] = True
        return frame if frame is not None else _frame(decision_dates)

    return build, state


def test_guarded_rejects_missing_or_fake_grant():
    build, state = _make_guarded()
    with pytest.raises(SealViolation):
        build(decision_dates=[DEV])
    with pytest.raises(SealViolation):
        build(decision_dates=[DEV], grant="x")
    with pytest.raises(SealViolation):
        build(decision_dates=[DEV], grant=object())
    assert not state["ran"]


def test_guarded_requires_decision_dates():
    build, state = _make_guarded()
    with pytest.raises(SealViolation):
        build(grant=seal.grant_dev())
    assert not state["ran"]


@pytest.mark.parametrize("bad", [GAP, SEALED, CONFIRM])
def test_dev_grant_rejects_other_dates_before_body(bad):
    build, state = _make_guarded()
    with pytest.raises(SealViolation) as ei:
        build(decision_dates=[DEV, bad], grant=seal.grant_dev())
    assert bad.isoformat() in str(ei.value)
    assert not state["ran"]


def test_guarded_ok_dev():
    build, state = _make_guarded()
    out = build(decision_dates=[DEV], grant=seal.grant_dev())
    assert state["ran"] and len(out) == 1


def test_dev_label_quarter_past_cutoff_raises():
    build, _ = _make_guarded()
    fr = _frame([DEV], label_last_quarter_end=[date(2023, 9, 30)])
    with pytest.raises(SealViolation):
        build(decision_dates=[DEV], grant=seal.grant_dev(), frame=fr)
    ok = _frame([DEV], label_last_quarter_end=[date(2023, 6, 30)])
    build(decision_dates=[DEV], grant=seal.grant_dev(), frame=ok)


def test_output_dates_must_be_subset_of_passed():
    build, _ = _make_guarded()
    fr = _frame([DEV, seal.DEV_DATES[1]])
    with pytest.raises(SealViolation):
        build(decision_dates=[DEV], grant=seal.grant_dev(), frame=fr)


def test_output_null_date_raises():
    build, _ = _make_guarded()
    fr = _frame([DEV, None])
    with pytest.raises(SealViolation):
        build(decision_dates=[DEV], grant=seal.grant_dev(), frame=fr)


def test_guarded_registered():
    @seal.guarded
    def some_fn(*, decision_dates, grant):
        return None

    assert f"{some_fn.__module__}.{some_fn.__qualname__}" in seal.GUARDED
    assert some_fn.__name__ == "some_fn"


def test_open_seal_grant(tmp_path):
    def no_tag(args):
        return 0, ""

    with pytest.raises(SealViolation):
        seal.open_seal_grant(
            Phase.SEALED_RUN, repo_dir=tmp_path, ledger_path=tmp_path / "l.jsonl", git_runner=no_tag
        )

    def with_tag(args):
        if args[0] == "tag":
            return 0, seal.SEAL_TAG + "\n"
        if args[0] == "cat-file":
            return 0, "tag\n"
        return 0, "abc123\n"

    g = seal.open_seal_grant(
        Phase.SEALED_RUN, repo_dir=tmp_path, ledger_path=tmp_path / "l.jsonl", git_runner=with_tag
    )
    assert isinstance(g, SealGrant)
    assert g.record["tag_commit"] == "abc123"
    assert SEALED in g.allowed_dates and CONFIRM not in g.allowed_dates


def test_digest_only_json_and_no_tag_needed():
    seen = {}

    @seal.guarded
    def build(*, decision_dates, grant):
        seen["grant"] = grant
        return _frame(decision_dates, y=[0.1234567 * i for i in range(len(decision_dates))])

    res = seal.digest_only(
        lambda g: build(decision_dates=[DEV, SEALED, GAP], grant=g),
        exact_cols=["decision_date", "ticker"],
        float_cols=["y"],
    )
    json.dumps(res)
    assert isinstance(seen["grant"], seal._HashGrant)
    assert res["n_rows"] == 3
    assert sum(res["n_rows_by"].values()) == 3
    assert len(res["exact_sha256"]) == 64 and len(res["float_sha256"]) == 64


def test_digest_only_rejects_confirm():
    @seal.guarded
    def build(*, decision_dates, grant):
        return _frame(decision_dates)

    with pytest.raises(SealViolation):
        seal.digest_only(
            lambda g: build(decision_dates=[CONFIRM], grant=g), exact_cols=["decision_date"]
        )


def test_digest_only_hides_exception_text():
    def builder(g):
        raise ValueError("secret 987.654")

    with pytest.raises(SealViolation) as ei:
        seal.digest_only(builder, exact_cols=["decision_date"])
    assert "987.654" not in str(ei.value) and "secret" not in str(ei.value)
    assert "ValueError" in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_digest_only_rejects_non_frame():
    with pytest.raises(SealViolation):
        seal.digest_only(lambda g: {"a": 1}, exact_cols=["a"])


def test_digest_only_frees_frame():
    ref = {}

    def builder(g):
        df = _frame([DEV, SEALED])
        ref["r"] = weakref.ref(df)
        return df

    seal.digest_only(builder, exact_cols=["decision_date", "ticker"])
    gc.collect()
    assert ref["r"]() is None


def test_digest_stable_under_permutation():
    base = _frame(seal.DEV_DATES[:5], y=[1.5, 2.5, 3.5, 4.5, 5.5])
    shuffled = base.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = seal.digest_only(lambda g: base, exact_cols=["decision_date", "ticker"], float_cols=["y"])
    b = seal.digest_only(
        lambda g: shuffled, exact_cols=["decision_date", "ticker"], float_cols=["y"]
    )
    assert a == b


def test_replace_and_setattr_forgeries_rejected():
    build, state = _make_guarded()
    g = seal.grant_dev()
    forged = dataclasses.replace(g, phase=Phase.SEALED_RUN)
    with pytest.raises(SealViolation):
        build(decision_dates=[SEALED], grant=forged)
    g2 = seal.grant_dev()
    object.__setattr__(g2, "phase", Phase.SEALED_RUN)
    with pytest.raises(SealViolation):
        build(decision_dates=[SEALED], grant=g2)
    assert not state["ran"]


def test_leaked_hash_grant_rejected_after_digest_only():
    build, state = _make_guarded()
    leaked = {}

    def builder(g):
        leaked["g"] = g
        return _frame([DEV])

    seal.digest_only(builder, exact_cols=["decision_date"])
    with pytest.raises(SealViolation):
        build(decision_dates=[SEALED], grant=leaked["g"])
    assert not state["ran"]
