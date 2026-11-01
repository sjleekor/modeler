"""P6′ training CLI: the P6 training run unchanged, behind the P6′ tag (decisions O-4, O-7).

O-4: P6′ runs the same grid CV, selection, final fit and bundle as P6 -- no code change. So this
module does not copy the training steps; it runs ``_03_kr_next.train.main`` itself. What differs
is the guard, in three layers:

1. **here, before anything is read**: ``--prereg-repo`` must have ``kr-next-p6prime-frozen``
   (``tags.require_prereg_tag``; the P6 tag does not count), ``--out`` must not exist;
2. **the panel**: ``--panel`` is required (P6's default path is P6's panel) and must sit next to
   its ``manifest_labeled.json`` written by the P6′ builder: ``mode`` labeled, ``prereg_tag`` the
   P6′ tag, ``panel_sha256`` equal to the file's. A P6 panel has the P6 tag there and is refused;
3. **inside the P6 run**: ``tags.p6_train_with_p6prime_tag`` swaps ``train.require_frozen_tag``
   (it then checks the P6′ tag whatever it is asked) and ``train.PREREG_TAG`` (written to the
   bundle manifest as ``prereg_tag``) for the duration of the call. See :mod:`.tags` for the
   risks of that swap and the test that pins ``train.main``'s tag references.

Usage::

    python -m modeler.models._04_kr_next_p6prime.train --lake <stock_data/kr> --snapshot S \
        --out <new bundle dir> --prereg-repo <my repo> --panel <p6prime panel_labeled.parquet> \
        --prereg-doc <P6′ prereg doc> [--r1-sensitivity]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from modeler.models._03_kr_next import build_panel as p6_bp
from modeler.models._03_kr_next import train as p6_train
from modeler.models._03_kr_next import views as p6_views
from modeler.models._04_kr_next_p6prime import tags


class PanelRefused(RuntimeError):
    """The panel is not a P6′ labeled panel (or does not match its manifest)."""


def check_panel(panel: str | Path) -> dict:
    """The panel's builder manifest, after checking it is a P6′ labeled panel."""
    path = Path(panel)
    if not path.is_file():
        raise PanelRefused(f"labeled panel not found: {path}")
    expected, man_path = p6_bp.output_paths(path.parent, "labeled")
    if path.name != expected.name or not man_path.is_file():
        raise PanelRefused(f"not a builder output with its manifest: {path} ({man_path})")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("mode") != "labeled":
        raise PanelRefused(f"panel manifest mode is {man.get('mode')!r}, not 'labeled'")
    if man.get("prereg_tag") != tags.PREREG_TAG:
        raise PanelRefused(
            f"panel was built under prereg tag {man.get('prereg_tag')!r}, not {tags.PREREG_TAG!r}"
        )
    if p6_views.sha256_file(path) != man.get("panel_sha256"):
        raise PanelRefused(f"panel sha256 differs from its manifest: {path}")
    return man


def _parse_known(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prereg-repo")
    ap.add_argument("--out")
    ap.add_argument("--panel")
    return ap.parse_known_args(argv)[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Same arguments as ``_03_kr_next.train`` (``--panel`` required)."""
    args = list(sys.argv[1:] if argv is None else argv)
    known = _parse_known(args)
    if not known.prereg_repo or not known.out:
        # let the P6 parser print its usage error for the missing required options
        p6_train._parse_args(args)
        return 2
    # 1) the P6′ tag before anything is read
    try:
        tags.require_prereg_tag(known.prereg_repo)
    except tags.FrozenTagMissing as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if Path(known.out).exists():
        print(f"REFUSED: --out already exists, not overwriting: {known.out}", file=sys.stderr)
        return 2
    # 2) a P6′ labeled panel, named explicitly
    if not known.panel:
        print("REFUSED: --panel is required (the default path is P6's panel)", file=sys.stderr)
        return 2
    try:
        check_panel(known.panel)
    except PanelRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    # 3) the P6 run, with the P6′ tag swapped in for its own guard and manifest
    with tags.p6_train_with_p6prime_tag():
        return p6_train.main(args)


if __name__ == "__main__":
    sys.exit(main())
