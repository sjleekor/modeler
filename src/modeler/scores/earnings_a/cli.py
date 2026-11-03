"""Command line: ``python -m modeler.scores.earnings_a.cli {f3,f4} [--out DIR] [--rehearsal]``.

stdout carries only the output directory and the manifest path. Errors go to stderr.
F4 refuses a dirty ``earnings_a`` tree unless ``--rehearsal`` (then the manifest kind says so).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from modeler.scores.earnings_a import digest, manifest
from modeler.scores.earnings_a import paths as P


def _out_dir(paths: P.DataPaths, cmd: str, out: str | None, *, create: bool = True) -> Path:
    """New output directory; it must not exist. F3 lets ``checks.write_f3`` create it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out) if out else paths.out_root / f"{cmd}_{stamp}"
    if create:
        d.mkdir(parents=True, exist_ok=False)
    elif d.exists():
        raise FileExistsError(str(d))
    return d


def run_f3(paths: P.DataPaths, out: str | None) -> tuple[Path, Path]:
    from modeler.scores.earnings_a import checks  # label-free report, imported lazily

    con = P.connect_inputs(paths)
    try:
        uni, rows, feats = digest.prepare_inputs(paths, con)
    finally:
        con.close()
    report = checks.f3_report(uni, feats)
    signs = checks.sign_table(rows)
    out_dir = _out_dir(paths, "f3", out, create=False)
    checks.write_f3(report, signs, out_dir)
    m = manifest.build_manifest(
        "F3_label_free",
        paths,
        code=manifest.git_state(manifest.repo_root()),
        inputs=manifest.describe_inputs(paths, include_r2=False),
    )
    return out_dir, manifest.write_once(out_dir / "manifest_f3.json", m)


def run_f4_cmd(paths: P.DataPaths, out: str | None, rehearsal: bool) -> tuple[Path, Path]:
    git = manifest.git_state(manifest.repo_root())
    if git["dirty"] and not rehearsal:
        raise SystemExit("refused: earnings_a code or tests are dirty (use --rehearsal)")
    P.verify_input_hashes(paths)
    out_dir = _out_dir(paths, "f4", out)
    run_at = datetime.now(UTC)
    con = P.connect_inputs(paths)
    try:
        result = digest.run_f4(paths, con, run_at=run_at)
    finally:
        con.close()
    kind = "F4_hash_only_REHEARSAL" if rehearsal else "F4_hash_only"
    reason_digests = result["reason_digests"]
    extra = {
        "digests": result["digests"],
        "reason_digest": {
            w: {k: v for k, v in d.items() if k != "n_rows_by"} for w, d in reason_digests.items()
        },
        "reason_counts_by_date": {w: d["n_rows_by"] for w, d in reason_digests.items()},
        "raw_max_trade_date": result["raw_max_trade_date"],
        "dirty_recorded": bool(git["dirty"]),
    }
    m = manifest.build_manifest(kind, paths, code=git, extra=extra)
    return out_dir, manifest.write_once(out_dir / "manifest_f4.json", m)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="modeler.scores.earnings_a.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("f3").add_argument("--out")
    p4 = sub.add_parser("f4")
    p4.add_argument("--out")
    p4.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args(argv)
    paths = P.DataPaths.resolve()
    if args.cmd == "f3":
        out_dir, mpath = run_f3(paths, args.out)
    else:
        out_dir, mpath = run_f4_cmd(paths, args.out, args.rehearsal)
    sys.stdout.write(f"{out_dir}\n{mpath}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
