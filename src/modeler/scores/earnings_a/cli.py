"""Command line: ``python -m modeler.scores.earnings_a.cli {f3,f4,dev,sealed} [--out DIR] [--rehearsal | --dry-run-dev]``.

Stage 2: ``stage2-check``, ``stage2-repro`` (DEV grant only), ``stage2-past`` (opens STAGE2_PAST once),
``stage2-forward --decision-date D`` (opens FORWARD for D), ``stage2-score --meta FILE...``.

stdout carries only the output directory and the manifest path. Errors go to stderr.
F4 refuses a dirty ``earnings_a`` tree unless ``--rehearsal`` (then the manifest kind says so).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from modeler.scores.earnings_a import dev_select, digest, manifest, sealed_run
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


def run_dev_cmd(paths: P.DataPaths, out: str | None, rehearsal: bool) -> tuple[Path, Path]:
    git = manifest.git_state(manifest.repo_root())
    if git["dirty"] and not rehearsal:
        raise SystemExit("refused: earnings_a code or tests are dirty (use --rehearsal)")
    P.verify_input_hashes(paths)
    out_dir = _out_dir(paths, "dev", out)
    run_at = datetime.now(UTC)
    con = P.connect_inputs(paths)
    try:
        res = dev_select.run_dev_selection(paths, con, run_at=run_at)
    finally:
        con.close()
    files = dev_select.write_outputs(out_dir, res)
    kind = "DEV_SELECTION_REHEARSAL" if rehearsal else "DEV_SELECTION"
    extra = {
        "run_at": run_at.isoformat(timespec="seconds"),
        "elapsed_s": round((datetime.now(UTC) - run_at).total_seconds(), 1),
        "chosen": {k: {"spec_id": s.spec_id, "params": s.params} for k, s in res["chosen"].items()},
        "output_sha256": {k: dev_select.file_sha256(p) for k, p in files.items()},
        "n_dev_rows": res["n_dev_rows"],
        "dataset_key_sha256": res["dataset_key_sha256"],
        "hgb_seed_average": res["seeds"],
        "dirty_recorded": bool(git["dirty"]),
    }
    m = manifest.build_manifest(kind, paths, code=git, extra=extra)
    return out_dir, manifest.write_once(out_dir / "manifest_dev.json", m)


def run_sealed_cmd(paths: P.DataPaths, out: str | None, dry_run_dev: bool) -> tuple[Path, Path]:
    """One-shot sealed verdict (or its DEV dry run); all checks and the single open are inside."""
    mode = sealed_run.make_mode(dry_run_dev)
    return sealed_run.run_sealed(
        paths, sealed_run.make_out_dir(paths, mode, out), dry_run_dev=dry_run_dev
    )


def run_stage2_cmd(paths: P.DataPaths, args: argparse.Namespace) -> tuple[Path, Path]:
    from datetime import date

    from modeler.scores.earnings_a import stage2_run as S2

    if args.cmd == "stage2-check":
        return S2.run_check(paths, args.out)
    if args.cmd == "stage2-past":
        return S2.run_past(paths, args.out, allow_reopen=args.allow_reopen)
    if args.cmd == "stage2-forward":
        return S2.run_forward(
            paths,
            date.fromisoformat(args.decision_date),
            args.out,
            allow_reopen=args.allow_reopen,
        )
    past = Path(args.past_summary) if args.past_summary else None
    return S2.run_score(paths, [Path(m) for m in args.meta], args.out, past)


def run_stage2_repro(paths: P.DataPaths) -> str:
    """Reproduction check under the DEV grant: prints the max abs MAE differences only."""
    import json as _json

    from modeler.scores.earnings_a import stage2_run as S2
    from modeler.scores.earnings_a.sealed_run import CHOSEN_REL, CHOSEN_SHA256, _load_chosen

    P.verify_input_hashes(paths)
    chosen = _load_chosen(paths.out_root / CHOSEN_REL, CHOSEN_SHA256)
    con = S2.connect_stage2(paths)
    try:
        res = S2.reproduce_dev(
            paths, con, chosen, paths.out_root / "dev_20261103T031237Z" / "per_date.tsv"
        )
    finally:
        con.close()
    return _json.dumps(res, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="modeler.scores.earnings_a.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("f3").add_argument("--out")
    p4 = sub.add_parser("f4")
    p4.add_argument("--out")
    p4.add_argument("--rehearsal", action="store_true")
    pd_ = sub.add_parser("dev")
    pd_.add_argument("--out")
    pd_.add_argument("--rehearsal", action="store_true")
    ps = sub.add_parser("sealed")
    ps.add_argument("--out")
    ps.add_argument("--dry-run-dev", action="store_true")
    sub.add_parser("stage2-check").add_argument("--out")
    sub.add_parser("stage2-repro")
    for name in ("stage2-past", "stage2-forward"):
        sp = sub.add_parser(name)
        sp.add_argument("--out")
        sp.add_argument("--allow-reopen", action="store_true")
        if name == "stage2-forward":
            sp.add_argument("--decision-date", required=True)
    sc = sub.add_parser("stage2-score")
    sc.add_argument("--out")
    sc.add_argument("--meta", nargs="+", required=True)
    sc.add_argument("--past-summary")
    args = ap.parse_args(argv)
    paths = P.DataPaths.resolve()
    if args.cmd == "stage2-repro":
        sys.stdout.write(run_stage2_repro(paths) + "\n")
        return 0
    if args.cmd.startswith("stage2-"):
        out_dir, mpath = run_stage2_cmd(paths, args)
        sys.stdout.write(f"{out_dir}\n{mpath}\n")
        return 0
    if args.cmd == "f3":
        out_dir, mpath = run_f3(paths, args.out)
    elif args.cmd == "sealed":
        out_dir, mpath = run_sealed_cmd(paths, args.out, args.dry_run_dev)
    elif args.cmd == "dev":
        out_dir, mpath = run_dev_cmd(paths, args.out, args.rehearsal)
    else:
        out_dir, mpath = run_f4_cmd(paths, args.out, args.rehearsal)
    sys.stdout.write(f"{out_dir}\n{mpath}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
