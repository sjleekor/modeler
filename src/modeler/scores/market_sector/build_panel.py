"""시장·섹터 자산 패널·라벨 데이터셋을 만든다 (레이어 1: 피쳐·모델 없음).

    uv run python -m modeler.scores.market_sector.build_panel --market us \\
        --version ms_panel_v1 [--snapshot-date YYYY-MM-DD] [--assets us_spx us_ndx ...]

출력 ``stock_data/us/datasets/market_sector/<version>/``: ``panel.parquet``,
``labels.parquet``, ``manifest.json``, ``readiness.json``. 이미 있는 버전은 덮지 않는다
(``--overwrite`` 없이는 거부) — 동결본을 실수로 덮지 않으려는 것이다.

``raw/``·``derived/``에는 쓰지 않는다(미러가 ``--delete``로 덮는다).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from modeler.etl.config import REPO_ROOT, DataRoot
from modeler.scores.common.assets import (
    ASSET_REGISTRY_VERSION,
    Asset,
    assets_for_market,
    get_asset,
    registry_hash,
)
from modeler.scores.common.calendar import HORIZON_SESSIONS, SessionCalendar
from modeler.scores.common.cash import (
    CASH_BASIS,
    DEFAULT_STALENESS_DAYS,
    CashAccount,
    build_cash_account,
    load_us_rates,
)
from modeler.scores.common.inputs import PinnedScopedLake, sha256_file
from modeler.scores.common.panel import (
    AVAILABLE_AT_BASIS,
    PRICE_AVAILABILITY_BUFFER,
    build_asset_panel,
)
from modeler.scores.common.total_return import load_us_total_return
from modeler.scores.market_sector.labels import LOSS_THRESHOLD, compute_labels
from modeler.us.dataset import content_hash, git_commit

logger = logging.getLogger(__name__)

DEFAULT_VERSION = "ms_panel_v1"
US_TABLES_USED = ("prices_daily", "corp_actions", "macro_series", "trading_calendar")
DEFAULT_CASH_SERIES = {"us": "DGS3MO", "kr": "rate_kr_cd91"}


class KrNotSyncedError(RuntimeError):
    """KR 지수·CD91 데이터가 아직 맥에 sync되지 않았다."""


def _select_assets(market: str, ids: list[str] | None) -> list[Asset]:
    chosen = [get_asset(i) for i in ids] if ids else list(assets_for_market(market))
    bad = [a.asset_id for a in chosen if a.market != market.upper()]
    if bad:
        raise ValueError(f"{market.upper()} 시장이 아닌 자산: {bad}")
    return chosen


def assemble(
    assets: list[Asset],
    paths: dict[str, pl.DataFrame],
    cal: SessionCalendar,
    cash: CashAccount | None,
    *,
    horizon: int = HORIZON_SESSIONS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """순수 조립: 경로 -> (panel, labels). 부모 경로는 ``paths``에 있어야 한다."""
    panels, labels = [], []
    for a in assets:
        path = paths[a.asset_id]
        panels.append(build_asset_panel(a.asset_id, cal, path, horizon=horizon))
        parent = paths[a.parent_benchmark] if a.parent_benchmark else None
        labels.append(
            compute_labels(
                a.asset_id,
                cal,
                path,
                cash=cash,
                parent_path=parent,
                has_parent=a.parent_benchmark is not None,
                horizon=horizon,
            )
        )
    return (
        pl.concat(panels).sort(["asset_id", "session"]),
        pl.concat(labels).sort(["asset_id", "session"]),
    )


def write_outputs(panel: pl.DataFrame, labels: pl.DataFrame, out_dir: Path) -> dict[str, str]:
    """parquet를 결정적으로 쓴다(같은 입력 -> 같은 바이트). 파일 sha256을 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, df in (("panel", panel), ("labels", labels)):
        p = out_dir / f"{name}.parquet"
        df.write_parquet(p, compression="zstd", statistics=True)
        hashes[f"{name}.parquet"] = sha256_file(p)
    return hashes


def readiness_table(
    assets: list[Asset],
    paths: dict[str, pl.DataFrame],
    diags: dict[str, dict[str, Any]],
    labels: pl.DataFrame,
    cal: SessionCalendar,
    *,
    horizon: int = HORIZON_SESSIONS,
) -> list[dict[str, Any]]:
    rows = []
    for a in assets:
        p = paths[a.asset_id]
        d = diags[a.asset_id]
        first, last = p["session"].min(), p["session"].max()
        expected = [s for s in cal.sessions if first <= s <= last]
        obs = set(p["session"].to_list())
        lab = labels.filter((pl.col("asset_id") == a.asset_id) & pl.col("label_matured"))
        n_win = 0
        if lab.height:
            i0 = cal.index_of(lab["session"].min())
            i1 = cal.index_of(lab["session"].max())
            n_win = (i1 - i0) // horizon + 1
        rows.append(
            {
                "asset_id": a.asset_id,
                "proxy": a.proxy,
                "first_session": first,
                "last_session": last,
                "sessions_observed": p.height,
                "sessions_in_calendar_range": len(expected),
                "missing_vs_calendar": len([s for s in expected if s not in obs]),
                "duplicate_price_dates": d["duplicate_price_dates"],
                "prices_off_calendar": d["prices_off_calendar"],
                "dividend_rows": d["dividend_rows"],
                "ex_date_min": d["ex_date_min"],
                "ex_date_max": d["ex_date_max"],
                "dividends_possibly_missing_recent": d["possibly_missing_recent"],
                "return_basis": p["return_basis"][0],
                "first_matured_decision": lab["session"].min() if lab.height else None,
                "last_matured_decision": lab["session"].max() if lab.height else None,
                "matured_labels": lab.height,
                "loss_event_rate_8pct": (
                    float(lab["loss_event_60d_8pct"].cast(pl.Float64).mean())
                    if lab.height
                    else None
                ),
                "nonoverlapping_60d_windows": n_win,
                "excess_vs_cash_non_null": int(
                    labels.filter(pl.col("asset_id") == a.asset_id)["excess_return_60d_vs_cash"]
                    .is_not_null()
                    .sum()
                ),
                "excess_vs_market_non_null": int(
                    labels.filter(pl.col("asset_id") == a.asset_id)["excess_return_60d_vs_market"]
                    .is_not_null()
                    .sum()
                ),
            }
        )
    return rows


def format_readiness(rows: list[dict[str, Any]]) -> str:
    df = pl.DataFrame(rows)
    with pl.Config(tbl_rows=50, tbl_cols=50, tbl_width_chars=250, fmt_str_lengths=40):
        return str(
            df.transpose(
                include_header=True,
                header_name="field",
                column_names=df["asset_id"].cast(pl.String).to_list(),
            )
        )


def _json_default(o: Any) -> Any:
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def build_us(args: argparse.Namespace) -> Path:
    root = DataRoot.resolve("us")
    out_dir = root.datasets / "market_sector" / args.version
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} 가 이미 있습니다. 새 버전 이름을 쓰거나 --overwrite 를 주십시오."
        )

    selected = _select_assets("us", args.assets)
    needed = {a.asset_id: a for a in selected}
    for a in selected:
        if a.parent_benchmark:
            needed.setdefault(a.parent_benchmark, get_asset(a.parent_benchmark))
    symbols = {aid: a.proxy for aid, a in needed.items()}

    base_lake = PinnedScopedLake(root=root, snapshots={}, symbols=tuple(symbols.values()))
    snaps: dict[str, str] = {}
    for t in US_TABLES_USED:
        snaps[t] = (
            args.snapshot_date
            if args.snapshot_date
            and (root.derived / "snapshots" / t / f"snapshot_date={args.snapshot_date}").is_dir()
            else base_lake.latest_snapshot(t).isoformat()
        )
    lake = PinnedScopedLake(root=root, snapshots=snaps, symbols=tuple(symbols.values()))

    cal = SessionCalendar.from_us_lake(lake)
    paths, diags = load_us_total_return(lake, symbols, sessions=frozenset(cal.sessions))

    rates = load_us_rates(lake, args.cash_series)
    cash = (
        build_cash_account(
            rates, cal.sessions, series_id=args.cash_series, staleness_days=args.staleness_days
        )
        if rates is not None
        else None
    )
    if cash is None:
        logger.warning(
            "현금 시리즈 %s 가 레이크에 없다 — 현금 대비 라벨은 null(cash_series_missing)",
            args.cash_series,
        )

    panel, labels = assemble(selected, paths, cal, cash)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    out_hashes = write_outputs(panel, labels, tmp)

    inputs = {}
    for t, files in lake.input_files(US_TABLES_USED).items():
        inputs[t] = {
            "snapshot_date": snaps[t],
            "files": {f.name: {"sha256": sha256_file(f), "bytes": f.stat().st_size} for f in files},
        }
    readiness = readiness_table(selected, paths, diags, labels, cal)
    manifest = {
        "dataset": args.version,
        "market": "US",
        "layer": "market_sector_layer1_panel_labels",
        "modeler_git_commit": git_commit(REPO_ROOT, allow_dirty=args.allow_dirty),
        "asset_registry_version": ASSET_REGISTRY_VERSION,
        "asset_registry_hash": registry_hash(),
        "assets": [
            {
                "asset_id": a.asset_id,
                "proxy": a.proxy,
                "asset_type": a.asset_type,
                "parent_benchmark": a.parent_benchmark,
                "definition_version": a.definition_version,
                "return_basis": paths[a.asset_id]["return_basis"][0],
                "dividend_coverage": {
                    k: diags[a.asset_id][k]
                    for k in (
                        "dividend_rows",
                        "ex_date_min",
                        "ex_date_max",
                        "median_gap_days_recent8",
                        "days_since_last_ex_date",
                        "possibly_missing_recent",
                        "dividends_unattributed",
                        "dividends_shifted",
                        "prices_off_calendar",
                        "prices_off_calendar_dates",
                    )
                },
            }
            for a in needed.values()
        ],
        "inputs": inputs,
        "cash": {
            "cash_basis": CASH_BASIS,
            "series_id": args.cash_series,
            "series_present_in_lake": rates is not None,
            "staleness_days": args.staleness_days,
        },
        "calendar": {
            "calendar_id": cal.calendar_id,
            "calendar_basis": cal.calendar_basis,
            "first_session": cal.sessions[0],
            "last_session": cal.sessions[-1],
            "decision_rule": "next session open - 30 minutes",
        },
        "time_contract": {
            "price_available_at_basis": AVAILABLE_AT_BASIS,
            "price_availability_buffer_minutes": PRICE_AVAILABILITY_BUFFER.total_seconds() / 60,
            "entry": "close of t+1",
            "exit": f"close of entry + {HORIZON_SESSIONS} sessions",
            "px_adj_t_note": (
                "px_adj_t는 스냅샷 끝 기준 분할 조정이라 수준이 아니라 비율로만 쓴다. "
                "PIT 수준은 px_raw_t."
            ),
        },
        "labels": {"horizon_sessions": HORIZON_SESSIONS, "loss_threshold": LOSS_THRESHOLD},
        "outputs": {
            **out_hashes,
            "panel_content_hash": content_hash(panel),
            "labels_content_hash": content_hash(labels),
            "panel_rows": panel.height,
            "labels_rows": labels.height,
        },
        "env": {"polars": pl.__version__, "python": platform.python_version()},
    }
    (tmp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        + "\n"
    )
    (tmp / "readiness.json").write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        + "\n"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)

    print(format_readiness(readiness))
    print(f"\n출력: {out_dir}")
    return out_dir


def build_kr(args: argparse.Namespace) -> Path:
    raise KrNotSyncedError(
        "KR 지수(KRX 지수 일별)·CD91일(ECOS) 데이터가 아직 맥에 sync되지 않았습니다. "
        "collector 백필 후 `collector db sync-remote`로 받은 뒤 로더를 붙이십시오."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--market", choices=["us", "kr"], required=True)
    ap.add_argument(
        "--snapshot-date", default=None, help="표마다 이 날짜가 있으면 쓰고 없으면 최신"
    )
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--cash-series", default=None)
    ap.add_argument("--staleness-days", type=int, default=DEFAULT_STALENESS_DAYS)
    ap.add_argument(
        "--allow-dirty", action="store_true", help="커밋 안 된 트리 허용(manifest에 -dirty 표시)"
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cash_series is None:
        args.cash_series = DEFAULT_CASH_SERIES[args.market]
    (build_us if args.market == "us" else build_kr)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
