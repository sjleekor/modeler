"""WP9: label / training / judging code cannot bypass the seal guard.

Static (AST) checks over the package source plus behaviour checks with spies.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from modeler.scores.earnings_a import evaluate, labels, seal, train
from modeler.scores.earnings_a.seal import SealViolation

PKG_DIR = Path(seal.__file__).parent
GUARDED_MODULES = ("labels", "train", "evaluate")

# Public names in labels/train/evaluate that are deliberately NOT guarded.
# presence_view: drops value columns from an already-guarded frame (pure).
# select_spec: pure selection on predictions already joined with labels (no data access).
# ModelSpec: frozen dataclass of hyper-parameters, no data access.
UNGUARDED_ALLOW = {
    "labels": {"presence_view"},
    "train": {"select_spec", "ModelSpec"},
    "evaluate": set(),
}

REQUIRED_GUARDED = {
    "modeler.scores.earnings_a.labels.load_label_rows",
    "modeler.scores.earnings_a.labels.build_label_quarters",
    "modeler.scores.earnings_a.labels.build_labels",
    "modeler.scores.earnings_a.train.walk_forward",
    "modeler.scores.earnings_a.train.fit_predict_date",
    "modeler.scores.earnings_a.evaluate.h1_h2",
}

READ_ATTRS = {"read_parquet", "read_csv", "read_json", "scan_parquet", "scan_csv", "read_table"}


def _sources() -> dict[str, ast.Module]:
    return {
        p.stem: ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for p in sorted(PKG_DIR.glob("*.py"))
    }


SOURCES = _sources()


def _is_guarded_deco(d: ast.expr) -> bool:
    if isinstance(d, ast.Attribute):
        return d.attr == "guarded"
    if isinstance(d, ast.Name):
        return d.id == "guarded"
    return False


def _module_funcs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _guarded(fn: ast.FunctionDef) -> bool:
    return any(_is_guarded_deco(d) for d in fn.decorator_list)


# --------------------------------------------------------------------------- 1 registry


def test_registry_contains_required_functions():
    assert REQUIRED_GUARDED <= seal.GUARDED, REQUIRED_GUARDED - seal.GUARDED


# --------------------------------------------------------------------------- 2 AST


def test_r2_strings_only_in_labels_and_paths():
    needles = ("fin_quarterly_metric_vintage_dl_v4", "mart_r2")
    hits = []
    for name, tree in SOURCES.items():
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                if any(s in n.value for s in needles) and name not in {"labels", "paths"}:
                    hits.append((name, n.lineno))
    assert not hits, hits
    # paths.py really does define the r2 location (otherwise this check is vacuous)
    assert any(
        isinstance(n, ast.Constant) and isinstance(n.value, str) and "mart_r2" in n.value
        for n in ast.walk(SOURCES["paths"])
    )


def _attr_parents(tree: ast.Module) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    return parents


def _enclosing_func(parents: dict[int, ast.AST], node: ast.AST) -> str | None:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.FunctionDef):
            return cur.name
        cur = parents.get(id(cur))
    return None


def test_r2_parquet_attribute_access_restricted():
    flagged = []
    for name, tree in SOURCES.items():
        if name in {"labels", "paths"}:
            continue
        parents = _attr_parents(tree)
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Attribute) and n.attr == "r2_parquet"):
                continue
            par = parents.get(id(n))
            ok = False
            # sha256_file(paths.r2_parquet) is allowed in digest/manifest/cli
            if name in {"digest", "manifest", "cli"}:
                if isinstance(par, ast.Call):
                    fn = par.func
                    fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                    ok = fname == "sha256_file"
                # row counting helper inside manifest.py
                if not ok and name == "manifest":
                    ef = _enclosing_func(parents, n) or ""
                    ok = bool(re.search(r"row_?count|count_?rows|n_rows", ef))
            if not ok:
                flagged.append((name, n.lineno))
    assert not flagged, flagged


def test_public_functions_guarded_or_allowlisted():
    problems = []
    for mod in GUARDED_MODULES:
        tree = SOURCES[mod]
        for n in tree.body:
            if isinstance(n, ast.FunctionDef | ast.ClassDef) and not n.name.startswith("_"):
                if n.name in UNGUARDED_ALLOW[mod]:
                    continue
                if not (isinstance(n, ast.FunctionDef) and _guarded(n)):
                    problems.append(f"{mod}.{n.name}")
    assert not problems, problems


def _read_sites(fn: ast.FunctionDef) -> list[int]:
    lines = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in READ_ATTRS:
                lines.append(n.lineno)
            elif isinstance(f, ast.Attribute):
                if f.attr in READ_ATTRS:
                    lines.append(n.lineno)
                elif (
                    f.attr in {"connect", "sql", "query", "execute"}
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "duckdb"
                ):
                    lines.append(n.lineno)
        # SQL text that opens a parquet is also a read site
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and "read_parquet" in n.value:
            lines.append(n.lineno)
    return lines


def test_reads_only_inside_guarded_call_chains():
    problems = []
    for mod in GUARDED_MODULES:
        funcs = _module_funcs(SOURCES[mod])
        callers: dict[str, set[str]] = {k: set() for k in funcs}
        for cname, c in funcs.items():
            for n in ast.walk(c):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs:
                    callers[n.func.id].add(cname)
                # a function passed around as a value counts as a use too
                if isinstance(n, ast.Name) and n.id in funcs and n.id != cname:
                    callers[n.id].add(cname)

        def safe(name: str, seen: frozenset[str] = frozenset()) -> bool:
            if _guarded(funcs[name]):
                return True
            if not name.startswith("_") or name in seen:
                return False
            cs = callers[name]
            return bool(cs) and all(safe(c, seen | {name}) for c in cs)

        for name, fn in funcs.items():
            sites = _read_sites(fn)
            if sites and not safe(name):
                problems.append(f"{mod}.{name} lines {sites}")
        # module-level (outside any function) reads
        for n in SOURCES[mod].body:
            if not isinstance(n, ast.FunctionDef | ast.ClassDef | ast.Import | ast.ImportFrom):
                holder = ast.FunctionDef(
                    name="<module>", args=None, body=[n], decorator_list=[], lineno=n.lineno
                )
                if _read_sites(holder):
                    problems.append(f"{mod}.<module> line {n.lineno}")
    assert not problems, problems


def test_features_universe_checks_do_not_import_downstream():
    banned = {"labels", "train", "evaluate", "reasons"}
    problems = []
    for name in ("features", "universe", "checks"):
        tree = SOURCES.get(name)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    parts = a.name.split(".")
                    if "earnings_a" in parts and banned & set(parts):
                        problems.append((name, n.lineno, a.name))
            elif isinstance(n, ast.ImportFrom):
                parts = (n.module or "").split(".")
                pkg_local = n.level > 0 or "earnings_a" in parts
                if pkg_local and (banned & set(parts) or any(a.name in banned for a in n.names)):
                    problems.append((name, n.lineno, n.module))
    assert not problems, problems


def test_only_seal_mentions_grant_internals():
    names = {"_HashGrant", "_KEY", "_MINTED", "_mint"}
    problems = []
    for name, tree in SOURCES.items():
        if name == "seal":
            continue
        for n in ast.walk(tree):
            hit = None
            if isinstance(n, ast.Name) and n.id in names:
                hit = n.id
            elif isinstance(n, ast.Attribute) and n.attr in names:
                hit = n.attr
            elif isinstance(n, ast.alias) and n.name in names:
                hit = n.name
            elif isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in names:
                hit = n.name
            if hit:
                problems.append((name, getattr(n, "lineno", 0), hit))
    assert not problems, problems


# --------------------------------------------------------------------------- 3 behaviour


def _guarded_callables():
    out = []
    for mod in (labels, train, evaluate):
        for qn in sorted(seal.GUARDED):
            prefix = mod.__name__ + "."
            if qn.startswith(prefix) and "." not in qn[len(prefix) :]:
                out.append(pytest.param(getattr(mod, qn[len(prefix) :]), id=qn.split(".", 3)[-1]))
    return out


def _dummy_call(fn, _omit=(), **extra):
    """Call ``fn`` with None placeholders for everything the guard should stop first."""
    sig = inspect.signature(fn)
    args = []
    kwargs = {}
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty or p.kind is p.VAR_KEYWORD:
            continue
        if p.kind is p.KEYWORD_ONLY:
            kwargs[p.name] = None
        elif p.kind is p.POSITIONAL_OR_KEYWORD:
            args.append(None)
    kwargs.update(extra)
    for k in _omit:
        kwargs.pop(k, None)
    return fn(*args, **kwargs)


@pytest.fixture
def spies(monkeypatch):
    calls: list[str] = []

    def make(name):
        def spy(*a, **k):
            calls.append(name)
            raise AssertionError(f"{name} ran behind the guard")

        return spy

    monkeypatch.setattr(labels, "_read_r2", make("labels._read_r2"))
    monkeypatch.setattr(duckdb, "connect", make("duckdb.connect"))
    monkeypatch.setattr(pd, "read_parquet", make("pd.read_parquet"))
    monkeypatch.setattr(train, "_one_date", make("train._one_date"))
    monkeypatch.setattr(train, "_check_ds", make("train._check_ds"))
    monkeypatch.setattr(evaluate, "_prepare", make("evaluate._prepare"))
    return calls


def _forged_grants():
    sealed = seal.SEALED_DATES[0]
    ns = SimpleNamespace(
        phase=seal.Phase.SEALED_RUN,
        record=None,
        allowed_dates=seal.DEV_DATES + seal.GAP_DATES + seal.SEALED_DATES,
    )
    cloned = dataclasses.replace(seal.grant_dev(), phase=seal.Phase.SEALED_RUN)
    # even holding the private key, an object that was never minted is refused
    unminted = object.__new__(seal.SealGrant)
    object.__setattr__(unminted, "phase", seal.Phase.CONFIRM_RUN)
    object.__setattr__(unminted, "record", None)
    object.__setattr__(unminted, "_token", seal._KEY)
    return sealed, {"namespace": ns, "replace_phase": cloned, "unminted": unminted}


def test_guard_registry_not_empty_for_parametrisation():
    assert len(_guarded_callables()) >= 6


@pytest.mark.parametrize("fn", _guarded_callables())
def test_no_grant_raises_before_body(fn, spies):
    with pytest.raises(SealViolation):
        _dummy_call(fn, decision_dates=[date(2021, 9, 30)])
    with pytest.raises(SealViolation):
        _dummy_call(fn, decision_dates=[date(2021, 9, 30)], grant=None)
    assert spies == []


@pytest.mark.parametrize("fn", _guarded_callables())
def test_forged_grants_raise_before_body(fn, spies):
    sealed, forged = _forged_grants()
    for kind, g in forged.items():
        with pytest.raises(SealViolation):
            _dummy_call(fn, decision_dates=[sealed], grant=g)
        assert spies == [], kind


@pytest.mark.parametrize("fn", _guarded_callables())
def test_dev_grant_with_sealed_or_confirm_date_raises_before_body(fn, spies):
    g = seal.grant_dev()
    for d in (seal.SEALED_DATES[0], seal.GAP_DATES[0], seal.CONFIRM_DATES[0]):
        with pytest.raises(SealViolation):
            _dummy_call(fn, decision_dates=[d], grant=g)
    with pytest.raises(SealViolation):  # missing decision_dates keyword
        _dummy_call(fn, _omit=("decision_dates",), grant=g)
    assert spies == []


def test_open_seal_grant_without_tag_raises_and_writes_nothing(tmp_path):
    ledger = tmp_path / "sub" / "seal_ledger.jsonl"
    seen = []

    def runner(args):
        seen.append(tuple(args))
        return 0, ""  # `git tag -l` prints nothing: tag is missing

    with pytest.raises(SealViolation):
        seal.open_seal_grant(
            seal.Phase.SEALED_RUN, repo_dir=tmp_path, ledger_path=ledger, git_runner=runner
        )
    assert seen, "runner was not consulted"
    assert not ledger.exists() and not ledger.parent.exists()


def test_dev_label_ending_after_cutoff_raises():
    g = seal.grant_dev()
    t = date(2021, 9, 30)
    r2 = pd.DataFrame(
        {
            "ticker": pd.Series([], dtype=object),
            "metric_code": pd.Series([], dtype=object),
            "fs_basis": pd.Series([], dtype=object),
            "statement_period_end": pd.Series([], dtype=object),
            "pe_month": pd.Series([], dtype="int64"),
            "rcept_no": pd.Series([], dtype=object),
            "available_from": pd.Series([], dtype=object),
            "value_dec": pd.Series([], dtype=object),
            "standalone_null_reason": pd.Series([], dtype=object),
        }
    )
    anchor = _anchor([t])
    receipts = pd.DataFrame({"rcept_no": ["R1"], "report_nm": ["x"]})
    with pytest.raises(SealViolation, match="2023-09-30"):
        labels.build_labels(r2, anchor, receipts, decision_dates=[t], grant=g, target="oi_y2")
    # horizon 1 from the same date ends 2022-09-30 and is fine
    out = labels.build_labels(r2, anchor, receipts, decision_dates=[t], grant=g, target="oi_y1")
    assert len(out) == 1


def _anchor(dates):
    return pd.DataFrame(
        {
            "decision_date": list(dates),
            "ticker": ["A"] * len(dates),
            "corp_code": ["CA"] * len(dates),
            "basis": ["CFS"] * len(dates),
            "a_t_dec": ["1000.0000"] * len(dates),
        }
    )


# --------------------------------------------------------------------------- 4 digest_only


def _empty_r2():
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype=object),
            "metric_code": pd.Series([], dtype=object),
            "fs_basis": pd.Series([], dtype=object),
            "statement_period_end": pd.Series([], dtype=object),
            "pe_month": pd.Series([], dtype="int64"),
            "rcept_no": pd.Series([], dtype=object),
            "available_from": pd.Series([], dtype=object),
            "value_dec": pd.Series([], dtype=object),
            "standalone_null_reason": pd.Series([], dtype=object),
        }
    )


def _json_only(v):
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_only(x) for k, x in v.items())
    if isinstance(v, list):
        return all(_json_only(x) for x in v)
    return type(v) in (str, int)


def test_digest_only_sealed_dates_without_tag_json_only():
    d = seal.SEALED_DATES[0]
    receipts = pd.DataFrame({"rcept_no": ["R1"], "report_nm": ["x"]})

    def builder(g):
        return labels.build_labels(_empty_r2(), _anchor([d]), receipts, decision_dates=[d], grant=g)

    out = seal.digest_only(builder, exact_cols=["decision_date", "ticker", "label_ok"])
    assert _json_only(out), out
    json.dumps(out)
    assert out["n_rows"] == 1
    assert set(out) >= {"exact_sha256", "n_rows", "n_rows_by"}


def test_digest_only_confirm_date_raises():
    d = seal.CONFIRM_DATES[0]
    receipts = pd.DataFrame({"rcept_no": ["R1"], "report_nm": ["x"]})
    inner: list[Exception] = []

    def builder(g):
        try:
            return labels.build_labels(
                _empty_r2(), _anchor([d]), receipts, decision_dates=[d], grant=g
            )
        except Exception as e:
            inner.append(e)
            raise

    with pytest.raises(SealViolation):
        seal.digest_only(builder, exact_cols=["decision_date", "ticker"])
    assert inner and isinstance(inner[0], SealViolation)
    assert "2025-09-30" in str(inner[0])


# --------------------------------------------------------------------------- 5 cli


def test_cli_has_no_sealed_open_subcommand():
    tree = SOURCES.get("cli")
    if tree is None:
        pytest.skip("cli.py not present")
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr == "add_parser" and n.args:
                a = n.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    names.append(a.value)
    # "sealed" (2026-11-03, main session request) is the one-shot verdict runner: it checks the
    # chosen spec and the F4 digest before it opens the seal itself, in one process.
    # There is still no bare open/seal/confirm subcommand.
    bad = [x for x in names if re.search(r"seal|open|confirm", x, re.I) and x != "sealed"]
    assert not bad, bad
