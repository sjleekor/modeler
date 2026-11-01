"""P6′ freeze tags and the guards that check them instead of the P6 tags (decision O-7).

The P6 code has its tag names as module constants (``_03_kr_next.train.PREREG_TAG``,
``build_panel.main``'s literal, ``forward.BUNDLE_TAG_NAME``). Run as is, the P6 CLIs would let a
P6′ label panel or training through as soon as the P6 tag ``kr-next-frozen`` exists -- which it
does. The P6 files are frozen and are not edited, so P6′ checks its own tags in three ways, in
order of preference:

1. **pass the tag as an argument** where the P6 function takes one:
   :func:`require_prereg_tag` calls ``train.require_frozen_tag(repo, tag=PREREG_TAG)``;
2. **check in the wrapper**: :mod:`.build_panel` has its own ``main`` that calls (1) and then the
   P6 building functions (``register_inputs``, ``build_panel_file``), which take no tag;
3. **swap the module globals for one call** (last resort), where the P6 code reads a constant at
   run time and has no argument for it: ``train.main`` calls ``require_frozen_tag(repo)`` with
   the default tag and writes ``PREREG_TAG`` into the bundle manifest; ``forward.bundle_tag_time``
   compares with ``BUNDLE_TAG_NAME``. :func:`p6_train_with_p6prime_tag` and
   :func:`p6_forward_with_p6prime_tag` replace those names in the P6 module for the duration of a
   ``with`` block and put the originals back on exit (also on an exception).

Risks of (3), and what limits them:

* the swap is process-wide: another thread calling P6 code during the block would see the P6′
  tag. The swaps take a lock, are not re-entrant, and the CLIs here are single-threaded;
* a default argument is bound when the function is defined, so changing ``PREREG_TAG`` alone
  would not change what ``require_frozen_tag(repo)`` checks; the swap therefore replaces the
  function itself with one that always checks the P6′ tag;
* a future edit of the P6 file could add another reference the swap misses. The P6 files are
  frozen; :func:`tag_references` reads ``train.main``'s and ``forward.bundle_tag_time``'s source
  so a test fails when their tag references are not exactly the names swapped here.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from modeler.models._03_kr_next import forward as p6_forward
from modeler.models._03_kr_next import train as p6_train

#: P6′ preregistration tag (repository ``my``); labels stay closed until it exists
PREREG_TAG = "kr-next-p6prime-frozen"
#: P6′ bundle tag; forward points are judged only after it
BUNDLE_TAG = "kr-next-p6prime-bundle-frozen"
#: P6 tags, which must never open a P6′ label
P6_PREREG_TAG = p6_train.PREREG_TAG
P6_BUNDLE_TAG = p6_forward.BUNDLE_TAG_NAME

FrozenTagMissing = p6_train.FrozenTagMissing

_SWAP_LOCK = threading.Lock()


def require_prereg_tag(prereg_repo: str | Path) -> str:
    """Commit of ``kr-next-p6prime-frozen`` in ``prereg_repo``; ``FrozenTagMissing`` without it.

    The P6 tag does not count: only the exact P6′ tag ref is looked up.
    """
    return p6_train.require_frozen_tag(prereg_repo, tag=PREREG_TAG)


@contextmanager
def _swapped(module: object, **names: object) -> Iterator[None]:
    if not _SWAP_LOCK.acquire(blocking=False):
        raise RuntimeError("a P6 module is already swapped to the P6′ tags; not re-entrant")
    saved = {k: getattr(module, k) for k in names}
    try:
        for k, v in names.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)
        _SWAP_LOCK.release()


@contextmanager
def p6_train_with_p6prime_tag() -> Iterator[None]:
    """Inside the block ``_03_kr_next.train`` checks and records the P6′ preregistration tag.

    ``train.require_frozen_tag`` becomes a function that looks up ``kr-next-p6prime-frozen``
    whatever tag it is asked for, and ``train.PREREG_TAG`` (written to the bundle manifest) is
    the P6′ tag. Both are restored on exit.
    """
    original = p6_train.require_frozen_tag

    def require_p6prime(prereg_repo: str | Path, tag: str | None = None) -> str:
        return original(prereg_repo, tag=PREREG_TAG)

    with _swapped(p6_train, PREREG_TAG=PREREG_TAG, require_frozen_tag=require_p6prime):
        yield


@contextmanager
def p6_forward_with_p6prime_tag() -> Iterator[None]:
    """Inside the block ``_03_kr_next.forward`` accepts only the P6′ bundle tag name."""
    with _swapped(p6_forward, BUNDLE_TAG_NAME=BUNDLE_TAG):
        yield


def tag_references(func: object) -> dict[str, set[str]]:
    """What ``func`` reads that concerns a tag, from its source (for the guard tests).

    ``{"names": {...}, "literals": {...}}``: every name it loads whose spelling contains ``tag``
    or ``frozen``, and every string constant (docstring excluded) that contains ``kr-next``.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))  # type: ignore[arg-type]
    body = tree.body[0].body  # type: ignore[attr-defined]
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        del body[0]  # the docstring is not code
    names = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
        and isinstance(n.ctx, ast.Load)
        and ("tag" in n.id.lower() or "frozen" in n.id.lower())
    }
    literals = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "kr-next" in n.value
    }
    return {"names": names, "literals": literals}
