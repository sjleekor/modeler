"""CLI to train the frozen KR h20 serving bundle on sj2-server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modeler.serving.kr_model import build_label_end_dates, train_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path, help="transferred frozen feat_panel.parquet")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--price-glob", required=True, help="frozen daily_ohlcv parquet glob")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-code-revision", required=True)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = args.work_dir / "kr_h20_label_end_dates.parquet"
    if terminal_path.exists():
        raise FileExistsError(f"terminal-date output already exists: {terminal_path}")
    terminal = build_label_end_dates(args.price_glob)
    terminal.write_parquet(terminal_path, compression="zstd")
    manifest = train_bundle(
        panel_path=args.panel,
        label_ends_path=terminal_path,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        price_source=args.price_glob,
        source_code_revision=args.source_code_revision,
    )
    print(json.dumps({"bundle_dir": str(args.output_dir), "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
