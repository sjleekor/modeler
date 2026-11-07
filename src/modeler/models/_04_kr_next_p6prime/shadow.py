"""P6′ shadow serving: score with the P6′ bundle into the P6′ ledger, through the P6 code.

Preregistration ``my/milestones/kr/modeling/plan/20261102_p6prime_prereg/
01_preregistration_draft.md`` §1.1 (serving 정리매매 rule and record format = P6 §5.3/§11, own
ledger, own release, own Cronicle event) and §5.2 (P6′ ``out_root``, ledger and lock separate
from P6; P6′ never writes or reads the P6 ledger). Nothing in ``_03_kr_next`` is modified: the
daily call is ``_03_kr_next.shadow.score_day`` itself (same candidates, ranks, tail flags, grid,
basket, ``record.json``, hash-chained ``ledger.jsonl``, deadline D 15:00 KST, exit codes). This
module only adds guards around it and puts the P6′ provenance into the one free-form field of the
P6 record, ``panel_manifest``:

* **P6′ bundle only.** The bundle's ``manifest.json`` must carry ``prereg_tag`` =
  ``kr-next-p6prime-frozen`` (the P6′ training writes it, :mod:`.tags`); a P6 bundle
  (``kr-next-frozen``) is refused before anything is written. The P6 shadow checks no tag here
  (it relies on the bundle's golden check), so this is an addition, not a swap.
* **P6′ root only.** ``out_root`` (records) must resolve to a path with a component containing
  ``p6prime``, and the ledger must lie inside it. The P6 default
  (``/home/whi/apps/market-briefing/shadow/records``) is refused without being read. The ledger
  lock is P6's ``<ledger>.lock``, so it is a different file from P6's.
* **P6′ panel only.** The panel directory must hold the P6 ``panel_manifest.json`` *and* the
  P6′ sidecar ``p6prime_panel.json`` written by :mod:`.shadow_prepare`. They must name the same
  D and the same P6′ view-folder manifest; the sidecar must say the financial file holds
  ``fin_quarterly_metric_vintage_dlr_v4p``. A panel prepared by the P6 step has no sidecar and
  is refused.

What the record holds beyond P6: ``panel_manifest`` = the P6 panel manifest plus one key
``p6prime`` = ``{"panel": <sidecar>, "score": {"bundle_tag", "prereg_tag",
"bundle_manifest_sha256", "code": {files, inventory_sha256}}}`` (``code`` = the ``*.py`` of this
package; ``record.code`` stays the P6 package, which is the code that scores). All other keys,
the record schema ``kr-next-shadow-record.v1`` and the ledger schema are P6's.

``verify`` = the P6 ledger check plus, for every line, that the record's bundle tag is the P6′
tag and its ``panel_manifest`` has the ``p6prime`` block (``p6prime_breaks``).

Commands (same arguments and exit codes as ``_03_kr_next.shadow``; ``--panel-manifest`` is
required)::

    python -m modeler.models._04_kr_next_p6prime.shadow score --date D --panel <p> \
        --panel-manifest <m> --bundle-dir <b> --out-root <r> --ledger <r>/ledger.jsonl \
        --sessions <calendar json>
    python -m modeler.models._04_kr_next_p6prime.shadow verify --ledger <l> --out-root <r>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from modeler.etl.marts import financial_quarters_v4p as fq4p
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import shadow as p6
from modeler.models._04_kr_next_p6prime import tags

MODEL = "kr-next-p6prime"
ROOT_MARK = "p6prime"
SIDECAR_FILE = "p6prime_panel.json"
SIDECAR_SCHEMA = "kr-next-p6prime-panel-sidecar.v1"
PANEL_MANIFEST_KEY = "p6prime"
ShadowError = p6.ShadowError


class P6RootRefused(ShadowError):
    """A path that is not a P6′ shadow path (for instance the P6 one) was given."""


def require_p6prime_root(path: str | Path, what: str = "out_root") -> Path:
    """``path`` resolved; refuses unless one of its components contains ``p6prime``.

    The P6 shadow root and records have no such component, so they cannot be passed by
    mistake. Nothing under ``path`` is read.
    """
    resolved = Path(path).expanduser().resolve()
    if not any(ROOT_MARK in part for part in resolved.parts):
        raise P6RootRefused(
            f"{what} {resolved} is not a P6′ path (no component contains {ROOT_MARK!r}); "
            "P6′ never writes into the P6 shadow root"
        )
    return resolved


def code_files() -> dict[str, str]:
    """``{file name: sha256}`` of every ``*.py`` of this package (``_04_kr_next_p6prime``)."""
    return p6.code_files(Path(__file__).resolve().parent)


def bundle_manifest_tag(bundle_dir: str | Path) -> tuple[str, str]:
    """``(prereg_tag, manifest sha256)`` of a bundle; refuses a bundle that is not P6′.

    Only ``prereg_tag`` is read from the manifest (nothing of it is printed).
    """
    path = Path(bundle_dir) / bundle_mod.MANIFEST_FILE
    if not path.is_file():
        raise ShadowError(f"bundle manifest missing: {path}")
    tag = json.loads(path.read_text(encoding="utf-8")).get("prereg_tag")
    if tag != tags.PREREG_TAG:
        raise ShadowError(
            f"bundle prereg_tag is {tag!r}, expected {tags.PREREG_TAG!r}: not a P6′ bundle"
        )
    return tag, p6.sha256_file(path)


def check_sidecar(
    d: date, panel_manifest: Mapping[str, Any] | None, sidecar: Mapping[str, Any] | None
) -> None:
    """Refuse a panel that was not prepared by the P6′ step (see the module docstring)."""
    if not panel_manifest:
        raise ShadowError("P6′ scoring needs the panel manifest (--panel-manifest)")
    if not sidecar:
        raise ShadowError(f"P6′ panel sidecar {SIDECAR_FILE} missing: not a P6′ panel")
    if sidecar.get("schema") != SIDECAR_SCHEMA or sidecar.get("model") != MODEL:
        raise ShadowError(f"{SIDECAR_FILE} is not a P6′ sidecar")
    if sidecar.get("fin_file_holds") != fq4p.FQMV_DLR_V4P_TABLE:
        raise ShadowError(f"{SIDECAR_FILE}: the financial file is not {fq4p.FQMV_DLR_V4P_TABLE}")
    if sidecar.get("D") != d.isoformat() or panel_manifest.get("D") != d.isoformat():
        raise ShadowError(f"panel manifest / sidecar are not of D={d}")
    if sidecar.get("views_manifest_sha256") != panel_manifest.get("views_manifest_sha256"):
        raise ShadowError("panel manifest and P6′ sidecar name different view folders")
    fin = (panel_manifest.get("derived_views") or {}).get("fin_quarterly_metric_vintage_dlr")
    if not fin or fin.get("sha256") != sidecar.get("fin_sha256"):
        raise ShadowError("the panel's financial view is not the P6′ (v4p) file of the sidecar")


def score_day(
    d: date,
    *,
    bundle_dir: str | Path,
    out_root: str | Path,
    ledger_path: str | Path,
    sessions: Sequence[date],
    panel: pl.DataFrame,
    panel_manifest: Mapping[str, Any] | None,
    sidecar: Mapping[str, Any] | None,
    now: datetime | None = None,
    lock_timeout: float = p6.LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Guards, then ``_03_kr_next.shadow.score_day`` (noflow, prebuilt panel). Same return."""
    root = require_p6prime_root(out_root)
    ledger = Path(ledger_path).expanduser().resolve()
    if root not in ledger.parents:
        raise P6RootRefused(f"ledger {ledger} is not inside the P6′ out_root {root}")
    prereg_tag, manifest_sha = bundle_manifest_tag(bundle_dir)
    check_sidecar(d, panel_manifest, sidecar)
    assert panel_manifest is not None and sidecar is not None
    cfiles = code_files()
    merged = {
        **dict(panel_manifest),
        PANEL_MANIFEST_KEY: {
            "panel": dict(sidecar),
            "score": {
                "bundle_tag": tags.BUNDLE_TAG,
                "prereg_tag": prereg_tag,
                "bundle_manifest_sha256": manifest_sha,
                "code": {"files": cfiles, "inventory_sha256": p6.code_inventory_sha256(cfiles)},
            },
        },
    }
    return p6.score_day(
        d,
        bundle_dir=bundle_dir,
        out_root=root,
        ledger_path=ledger,
        sessions=sessions,
        now=now,
        panel=panel,
        panel_manifest=merged,
        feature_set="noflow",
        lock_timeout=lock_timeout,
    )


def verify_ledger(path: str | Path, out_root: str | Path | None = None) -> dict[str, Any]:
    """P6's ledger check plus the P6′ checks of every record (``p6prime_breaks``)."""
    ledger = Path(path)
    root = require_p6prime_root(out_root if out_root is not None else ledger.parent)
    report = p6.verify_ledger(ledger, root)
    extra: list[str] = []
    try:
        lines = p6.read_ledger(ledger)
    except ShadowError as exc:
        lines, extra = [], [str(exc)]
    for n, line in enumerate(lines, 1):
        rec_path = root / str(line.get("dir")) / p6.FILE_RECORD
        if not rec_path.is_file():
            continue  # reported by the P6 check
        try:
            rec = json.loads(rec_path.read_bytes())
        except json.JSONDecodeError:
            continue  # reported by the P6 check
        if (rec.get("bundle") or {}).get("prereg_tag") != tags.PREREG_TAG:
            extra.append(f"line {n}: record bundle prereg_tag is not {tags.PREREG_TAG}")
        block = (rec.get("panel_manifest") or {}).get(PANEL_MANIFEST_KEY) or {}
        if (block.get("panel") or {}).get("fin_file_holds") != fq4p.FQMV_DLR_V4P_TABLE:
            extra.append(f"line {n}: record panel_manifest has no P6′ (v4p) block")
    report["p6prime_breaks"] = extra
    report["ok"] = bool(report["ok"]) and not extra
    return report


def read_sidecar(panel_dir: Path) -> dict[str, Any] | None:
    path = Path(panel_dir) / SIDECAR_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kr_next_p6prime_shadow", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--date", required=True, type=date.fromisoformat, help="report date D")
    s.add_argument("--panel", required=True, type=Path, help="serving-mode panel parquet (day t)")
    s.add_argument("--panel-manifest", required=True, type=Path)
    s.add_argument(
        "--sidecar", type=Path, default=None, help=f"default: <panel dir>/{SIDECAR_FILE}"
    )
    s.add_argument("--bundle-dir", required=True, type=Path)
    s.add_argument("--out-root", required=True, type=Path)
    s.add_argument("--ledger", required=True, type=Path)
    s.add_argument("--sessions", required=True, type=Path, help="JSON list or calendar manifest")
    v = sub.add_parser("verify")
    v.add_argument("--ledger", required=True, type=Path)
    v.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        if args.cmd == "verify":
            report = verify_ledger(args.ledger, args.out_root)
            print(p6._json_dumps(report, indent=2))
            return 0 if report["ok"] else 1
        manifest = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
        sidecar = (
            json.loads(args.sidecar.read_text(encoding="utf-8"))
            if args.sidecar
            else read_sidecar(args.panel.parent)
        )
        res = score_day(
            args.date,
            bundle_dir=args.bundle_dir,
            out_root=args.out_root,
            ledger_path=args.ledger,
            sessions=p6._load_sessions(args.sessions),
            panel=pl.read_parquet(args.panel),
            panel_manifest=manifest,
            sidecar=sidecar,
        )
    except (ShadowError, bundle_mod.BundleError, ValueError, OSError) as exc:
        print(f"kr_next_p6prime_shadow failed: {exc}", file=sys.stderr)
        return 1
    rec = res["record"]
    print(
        p6._json_dumps(
            {
                "dir": str(res["dir"]),
                "valid": rec["valid"],
                "attempt": rec["attempt"],
                "is_grid_date": rec["is_grid_date"],
                "counts": rec["counts"],
            },
            indent=2,
        )
    )
    return 0 if rec["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
