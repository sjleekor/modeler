"""Seal guard for research A stage 1 (next-4-quarter operating income / total assets).

The model is evaluated once on a sealed historical window. Until the preregistration is
frozen (git tag ``earnings-a-frozen``) nobody may read labels of sealed decision dates.
Label and training code must call this module before touching any label table.

Decision dates are calendar quarter-ends. Windows (inclusive):

* DEV      2016-03-31 .. 2022-06-30  (26 dates)
* GAP      2022-09-30, 2022-12-31, 2023-03-31 (3 dates, never evaluated)
* SEALED   2023-06-30 .. 2025-06-30  (9 dates)
* CONFIRM  2025-09-30 .. 2026-06-30  (4 dates)

Phases: ``DEV`` allows DEV only. ``SEALED_RUN`` retrains walk-forward, so it allows
DEV + GAP + SEALED. ``CONFIRM_RUN`` allows DEV + GAP + SEALED + CONFIRM.

Stage 2 (``my/milestones/kr/modeling/plan/20261103_earnings_a_stage2/01_preregistration_draft.md``):
``STAGE2_PAST`` evaluates the 25 quarter-ends 2019-06-30..2025-06-30 once (tag
``earnings-a2-frozen``, one ledger record); its training and allowed rows are DEV + GAP +
SEALED. ``FORWARD`` evaluates one quarter-end at a time (>= 2026-09-30 and <= the latest quarter
end on or before today); one grant and one ledger record per decision date. A CONFIRM date is a
training row only when the ledger holds a ``CONFIRM_RUN`` record whose ``scored_dates`` lists it.

Boundaries come from the preregistration draft
``my/milestones/common/scores/20261102_earnings_forecast_a/01_preregistration_draft.md``
and the design doc
``my/milestones/common/research/20261027_investment_research_agenda/14_plan_a_long_term_earnings.md``
section 2.3 and section 8.

This module never writes inside the repository: the ledger path is supplied by the caller.
Error messages list dates only, never other column values.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEAL_TAG = "earnings-a-frozen"
STAGE2_TAG = "earnings-a2-frozen"
DEV_LABEL_END_MAX = date(2023, 6, 30)  # DEV labels must not reach 2023Q3
_QUARTER_END_MD = {(3, 31), (6, 30), (9, 30), (12, 31)}


class SealViolation(RuntimeError):
    """A guarded read was attempted that the seal does not allow."""


class Phase(Enum):
    DEV = "DEV"
    SEALED_RUN = "SEALED_RUN"
    CONFIRM_RUN = "CONFIRM_RUN"
    FORWARD = "FORWARD"
    STAGE2_PAST = "STAGE2_PAST"


def _quarter_ends(start: date, end: date) -> tuple[date, ...]:
    out = []
    for y in range(start.year, end.year + 1):
        for m, d in sorted(_QUARTER_END_MD):
            c = date(y, m, d)
            if start <= c <= end:
                out.append(c)
    return tuple(out)


DEV_DATES = _quarter_ends(date(2016, 3, 31), date(2022, 6, 30))
GAP_DATES = _quarter_ends(date(2022, 9, 30), date(2023, 3, 31))
SEALED_DATES = _quarter_ends(date(2023, 6, 30), date(2025, 6, 30))
CONFIRM_DATES = _quarter_ends(date(2025, 9, 30), date(2026, 6, 30))

# Stage 2. Evaluation dates of the one-shot past run (25 quarter-ends).
STAGE2_PAST_DATES = _quarter_ends(date(2019, 6, 30), date(2025, 6, 30))
FORWARD_START = date(2026, 9, 30)


def forward_eval_dates(today: date) -> tuple[date, ...]:
    """Quarter-ends >= 2026-09-30 and <= ``today`` (the latest one on or before today)."""
    return _quarter_ends(FORWARD_START, today) if today >= FORWARD_START else ()


def allowed_decision_dates(
    phase: Phase,
    *,
    today: date | None = None,
    confirm_dates: Iterable[date] = (),
) -> tuple[date, ...]:
    """Rows a phase may read. ``FORWARD`` needs ``today`` (forward quarter-ends are added) and
    adds the ``confirm_dates`` the ledger covers (see :func:`covered_confirm_dates`)."""
    if phase is Phase.DEV:
        return DEV_DATES
    if phase in (Phase.SEALED_RUN, Phase.STAGE2_PAST):
        return DEV_DATES + GAP_DATES + SEALED_DATES
    if phase is Phase.CONFIRM_RUN:
        return DEV_DATES + GAP_DATES + SEALED_DATES + CONFIRM_DATES
    if today is None:
        raise SealViolation("FORWARD phase: allowed dates need the 'today' date")
    keep = {d for d in confirm_dates if d in set(CONFIRM_DATES)}
    return (
        DEV_DATES
        + GAP_DATES
        + SEALED_DATES
        + tuple(d for d in CONFIRM_DATES if d in keep)
        + forward_eval_dates(today)
    )


def _to_date(v: Any) -> date:
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).date()
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    raise TypeError(f"unsupported date type: {type(v).__name__}")


def check_decision_dates(dates: Iterable[Any], phase: Phase) -> None:
    """Raise ``SealViolation`` listing every date not allowed in ``phase``."""
    allowed = set(allowed_decision_dates(phase))
    bad = sorted({d for d in map(_to_date, dates) if d not in allowed})
    if bad:
        raise SealViolation(
            f"{len(bad)} decision date(s) not allowed in {phase.value}: "
            + ", ".join(d.isoformat() for d in bad)
        )


def check_frame(df: pd.DataFrame, phase: Phase, date_col: str = "decision_date") -> None:
    if date_col not in df.columns:
        raise SealViolation(f"column {date_col!r} missing; cannot verify the seal")
    n_null = int(df[date_col].isna().sum())
    if n_null:
        raise SealViolation(f"{n_null} row(s) with empty {date_col!r}; cannot verify the seal")
    check_decision_dates(df[date_col].unique(), phase)


def check_label_quarters(
    df: pd.DataFrame, phase: Phase, end_col: str = "label_last_quarter_end"
) -> None:
    """In DEV, every label's last target quarter end must be <= 2023-06-30."""
    if end_col not in df.columns:
        raise SealViolation(f"column {end_col!r} missing; cannot verify the seal")
    if phase is not Phase.DEV:
        return
    bad = sorted({d for d in map(_to_date, df[end_col].dropna().unique()) if d > DEV_LABEL_END_MAX})
    if bad:
        raise SealViolation(
            f"DEV labels reach past {DEV_LABEL_END_MAX.isoformat()}: "
            + ", ".join(d.isoformat() for d in bad)
        )


GitRunner = Callable[[Sequence[str]], "tuple[int, str]"]
"""``runner(git_args) -> (returncode, stdout)``; git_args exclude ``git -C <repo>``."""


def _subprocess_runner(repo_dir: Path) -> GitRunner:
    def run(args: Sequence[str]) -> tuple[int, str]:
        p = subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
        )
        return p.returncode, p.stdout

    return run


def default_ledger_path() -> Path:
    """``<STOCK_DATA_ROOT>/kr/output/earnings_forecast_a/seal_ledger.jsonl``.

    Same convention as the other modules: ``STOCK_DATA_ROOT``, default ``../stock_data``
    relative to the working directory (run from the modeler repo root).
    """
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data")).resolve()
    return root / "kr" / "output" / "earnings_forecast_a" / "seal_ledger.jsonl"


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _check_tag(run: GitRunner, repo: Path, tag: str) -> tuple[str, str]:
    """Return (tag commit, HEAD commit) of an existing annotated ``tag``; else raise."""
    rc, out = run(["tag", "-l", tag])
    if rc != 0 or out.strip() != tag:
        raise SealViolation(f"git tag {tag!r} not found in {repo}; seal stays closed")
    rc, kind = run(["cat-file", "-t", f"refs/tags/{tag}"])
    if rc != 0 or kind.strip() != "tag":
        raise SealViolation(f"git tag {tag!r} is not an annotated tag")
    rc, tag_commit = run(["rev-list", "-n", "1", tag])
    if rc != 0 or not tag_commit.strip():
        raise SealViolation(f"cannot resolve commit of tag {tag!r}")
    rc, head = run(["rev-parse", "HEAD"])
    if rc != 0 or not head.strip():
        raise SealViolation("cannot resolve HEAD commit")
    return tag_commit.strip(), head.strip()


def _append_ledger(ledger: Path, record: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


_OPEN_TAGS = {
    Phase.SEALED_RUN: SEAL_TAG,
    Phase.CONFIRM_RUN: SEAL_TAG,
    Phase.STAGE2_PAST: STAGE2_TAG,
}


def open_seal(
    phase: Phase,
    *,
    repo_dir: str | Path,
    ledger_path: str | Path,
    allow_reopen: bool = False,
    git_runner: GitRunner | None = None,
) -> dict[str, Any]:
    """Verify the freeze tag, check the ledger, append and return an open record.

    ``SEALED_RUN``/``CONFIRM_RUN`` need tag ``earnings-a-frozen``; ``STAGE2_PAST`` needs
    ``earnings-a2-frozen``. Each phase opens once unless ``allow_reopen`` (recorded).
    """
    if phase not in _OPEN_TAGS:
        raise SealViolation(
            f"open_seal applies to SEALED_RUN/CONFIRM_RUN/STAGE2_PAST, not {phase.value}"
        )
    tag = _OPEN_TAGS[phase]
    repo = Path(repo_dir)
    run = git_runner or _subprocess_runner(repo)
    tag_commit, head = _check_tag(run, repo, tag)

    ledger = Path(ledger_path)
    prior = [r for r in _read_ledger(ledger) if r.get("phase") == phase.value]
    if prior and not allow_reopen:
        raise SealViolation(f"{phase.value} already opened once")

    record = {
        "phase": phase.value,
        "opened_at": datetime.now(UTC).isoformat(),
        "tag": tag,
        "tag_commit": tag_commit,
        "head_commit": head,
        "reopen": bool(prior),
    }
    _append_ledger(ledger, record)
    return record


def covered_confirm_dates(ledger_path: str | Path) -> tuple[date, ...]:
    """CONFIRM dates a ``CONFIRM_RUN`` ledger record lists in ``scored_dates`` (none today)."""
    out: set[date] = set()
    for r in _read_ledger(Path(ledger_path)):
        if r.get("phase") == Phase.CONFIRM_RUN.value:
            out.update(date.fromisoformat(str(d)[:10]) for d in r.get("scored_dates", []))
    return tuple(d for d in CONFIRM_DATES if d in out)


def forward_train_dates(
    decision_date: date, *, today: date, ledger_path: str | Path
) -> tuple[date, ...]:
    """Dates a FORWARD grant for ``decision_date`` may read as training rows: DEV + GAP + SEALED,
    covered CONFIRM dates and earlier forward quarter-ends (not the decision date itself)."""
    d = _to_date(decision_date)
    allowed = allowed_decision_dates(
        Phase.FORWARD, today=today, confirm_dates=covered_confirm_dates(ledger_path)
    )
    return tuple(x for x in allowed if x < d)


def open_forward(
    decision_date: date,
    *,
    repo_dir: str | Path,
    ledger_path: str | Path,
    today: date | None = None,
    allow_reopen: bool = False,
    git_runner: GitRunner | None = None,
) -> dict[str, Any]:
    """One FORWARD grant record per decision date; needs tag ``earnings-a2-frozen``."""
    today = today or datetime.now(UTC).date()
    d = _to_date(decision_date)
    if d not in set(forward_eval_dates(today)):
        raise SealViolation(
            f"{d.isoformat()} is not a FORWARD decision date as of {today.isoformat()}"
        )
    repo = Path(repo_dir)
    run = git_runner or _subprocess_runner(repo)
    tag_commit, head = _check_tag(run, repo, STAGE2_TAG)
    ledger = Path(ledger_path)
    rows = _read_ledger(ledger)
    prior = [
        r
        for r in rows
        if r.get("phase") == Phase.FORWARD.value and r.get("decision_date") == d.isoformat()
    ]
    if prior and not allow_reopen:
        raise SealViolation(f"FORWARD {d.isoformat()} already granted once")
    confirm = covered_confirm_dates(ledger)
    allowed = allowed_decision_dates(Phase.FORWARD, today=today, confirm_dates=confirm)
    # earlier forward quarter-ends stay readable as training rows (rule A20 decides by
    # label_pub_date <= d whether one is actually used); later ones do not
    allowed = tuple(x for x in allowed if x <= d)
    record = {
        "phase": Phase.FORWARD.value,
        "decision_date": d.isoformat(),
        "opened_at": datetime.now(UTC).isoformat(),
        "tag": STAGE2_TAG,
        "tag_commit": tag_commit,
        "head_commit": head,
        "reopen": bool(prior),
        "confirm_dates": [x.isoformat() for x in confirm],
        "allowed_dates": [x.isoformat() for x in allowed],
    }
    _append_ledger(ledger, record)
    return record


def sealed_digest(df: pd.DataFrame, cols: Sequence[str]) -> str:
    """Order-independent sha256 (hex) of ``cols``; no values are exposed."""
    cols = list(cols)
    sub = df[cols].sort_values(cols, kind="mergesort", na_position="last").reset_index(drop=True)
    csv = sub.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Grants and the ``guarded`` decorator
# ---------------------------------------------------------------------------

_KEY = object()  # module-private; only this module can mint a SealGrant


@dataclass(frozen=True)
class SealGrant:
    """Proof that a phase was opened. Constructible only inside this module."""

    phase: Phase
    record: dict[str, Any] | None = None
    _token: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _KEY:
            raise SealViolation("SealGrant cannot be constructed directly")

    @property
    def allowed_dates(self) -> tuple[date, ...]:
        if self.phase is Phase.FORWARD:
            return tuple(date.fromisoformat(x) for x in (self.record or {})["allowed_dates"])
        return allowed_decision_dates(self.phase)


class _HashGrant(SealGrant):
    """Hash-only grant: DEV + GAP + SEALED dates, never CONFIRM. Made in digest_only."""

    @property
    def allowed_dates(self) -> tuple[date, ...]:
        return DEV_DATES + GAP_DATES + SEALED_DATES


_MINTED: dict[int, tuple[SealGrant, Phase, type]] = {}


def _mint(cls: type[SealGrant], phase: Phase, record: dict[str, Any] | None) -> SealGrant:
    g = cls(phase, record, _KEY)
    _MINTED[id(g)] = (g, phase, cls)
    return g


def _is_genuine(grant: Any) -> bool:
    if not isinstance(grant, SealGrant) or getattr(grant, "_token", None) is not _KEY:
        return False
    entry = _MINTED.get(id(grant))
    return (
        entry is not None
        and entry[0] is grant
        and entry[1] is grant.phase
        and entry[2] is type(grant)
    )


def grant_dev() -> SealGrant:
    return _mint(SealGrant, Phase.DEV, None)


def open_forward_grant(
    decision_date: date,
    *,
    repo_dir: str | Path,
    ledger_path: str | Path,
    today: date | None = None,
    allow_reopen: bool = False,
    git_runner: GitRunner | None = None,
) -> SealGrant:
    record = open_forward(
        decision_date,
        repo_dir=repo_dir,
        ledger_path=ledger_path,
        today=today,
        allow_reopen=allow_reopen,
        git_runner=git_runner,
    )
    return _mint(SealGrant, Phase.FORWARD, record)


def open_seal_grant(
    phase: Phase,
    *,
    repo_dir: str | Path,
    ledger_path: str | Path,
    allow_reopen: bool = False,
    git_runner: GitRunner | None = None,
) -> SealGrant:
    record = open_seal(
        phase,
        repo_dir=repo_dir,
        ledger_path=ledger_path,
        allow_reopen=allow_reopen,
        git_runner=git_runner,
    )
    return _mint(SealGrant, phase, record)


GUARDED: set[str] = set()


def _check_dates_in(dates: Iterable[Any], allowed: Iterable[date]) -> None:
    allow = set(allowed)
    bad = sorted({d for d in map(_to_date, dates) if d not in allow})
    if bad:
        raise SealViolation(
            f"{len(bad)} decision date(s) not allowed by grant: "
            + ", ".join(d.isoformat() for d in bad)
        )


def guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Require a genuine ``grant`` and allowed ``decision_dates`` keyword arguments."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        grant = kwargs.get("grant")
        if not _is_genuine(grant):
            raise SealViolation("a genuine SealGrant is required as keyword 'grant'")
        if "decision_dates" not in kwargs:
            raise SealViolation("keyword 'decision_dates' is required")
        passed = list(kwargs["decision_dates"])
        _check_dates_in(passed, grant.allowed_dates)
        result = fn(*args, **kwargs)
        if isinstance(result, pd.DataFrame) and "decision_date" in result.columns:
            n_null = int(result["decision_date"].isna().sum())
            if n_null:
                raise SealViolation(
                    f"{n_null} row(s) with empty 'decision_date'; cannot verify the seal"
                )
            out_dates = list(result["decision_date"].unique())
            _check_dates_in(out_dates, grant.allowed_dates)
            _check_dates_in(out_dates, [_to_date(d) for d in passed])
            if (
                grant.phase is Phase.DEV
                and not isinstance(grant, _HashGrant)
                and "label_last_quarter_end" in result.columns
            ):
                check_label_quarters(result, Phase.DEV)
        return result

    GUARDED.add(f"{fn.__module__}.{fn.__qualname__}")
    return wrapper


def digest_only(
    builder: Callable[[SealGrant], pd.DataFrame],
    *,
    exact_cols: Sequence[str],
    float_cols: Sequence[str] = (),
    count_by: Sequence[str] = ("decision_date",),
) -> dict[str, Any]:
    """Build a frame under a hash-only grant and return digests and counts only."""
    grant = _mint(_HashGrant, Phase.SEALED_RUN, None)
    try:
        df = builder(grant)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("builder must return a DataFrame")
        out: dict[str, Any] = {
            "exact_cols": [str(c) for c in exact_cols],
            "float_cols": [str(c) for c in float_cols],
            "exact_sha256": sealed_digest(df, exact_cols),
            "n_rows": int(len(df)),
        }
        if float_cols:
            out["float_sha256"] = sealed_digest(df, float_cols)
        counts = df.groupby(list(count_by), dropna=False).size()
        out["n_rows_by"] = {
            (str(k[0]) if isinstance(k, tuple) and len(k) == 1 else str(k)): int(v)
            for k, v in counts.items()
        }
        del df
    except Exception as e:
        raise SealViolation(f"hash-only build failed: {type(e).__name__}") from None
    finally:
        _MINTED.pop(id(grant), None)
    return out
