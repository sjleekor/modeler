"""합성 US 레이크·KR raw snapshot·시장·섹터 bundle. score_daily와 서빙 통합 시험이 같이 쓴다.

실제 거래 달력(``exchange_calendars`` XNYS·XKRX)으로 세션을 만들고, 가격은 시드가 고정된
랜덤 워크다. bundle의 모델은 합성 데이터로 맞춘 작은 Ridge·Logit이다(점수 값이 아니라 파이프라인
구조를 시험한다).
동결 run과 같은 점수를 내는지는 실제 ``stock_data``로 ``score_daily verify-frozen``이 본다.

기본 날짜: 리포트 D=2026-10-07(수). US 가격은 직전 완료 세션 A=10-06까지, KR 지수는
K=10-06보다 한 세션 이른 10-02까지(KRX Open API T+1 공표, 10-05는 대체공휴일)다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import polars as pl

from modeler.scores.common.assets import assets_for_market, kr_index_key
from modeler.scores.common.calendar import SessionCalendar
from modeler.scores.market_sector.bundle import MODEL_NAMES
from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.models import fit_model
from modeler.scores.market_sector.score_daily import (
    bundle_manifest,
    calendar_spec,
)

SEOUL = ZoneInfo("Asia/Seoul")
D = date(2026, 10, 7)
K = date(2026, 10, 6)
A = date(2026, 10, 6)
START = date(2024, 4, 1)
KR_LAST = date(2026, 10, 2)
US_SNAP = "2026-10-06"
MACRO_SNAP = "2026-10-05"
CAL_SNAP = "2026-09-22"
KR_SNAP = "2026-10-07"
US_DONE = datetime(2026, 10, 6, 15, 30, tzinfo=SEOUL)
KR_DONE = "2026-10-07T04:29:15+0900"
SERIES = ("VIXCLS", "DGS10", "DGS2", "DCOILWTICO", "BAA10Y", "DGS3MO")
RUN_IDS = {"US": "ms_us_synthetic", "KR": "ms_kr_synthetic"}


@dataclass
class World:
    root: Path
    cfg: MsConfig
    us_sessions: list[date]
    kr_sessions: list[date]

    @property
    def us_root(self) -> Path:
        return self.root / "us"

    @property
    def kr_root(self) -> Path:
        return self.root / "kr"

    def kr_snapshot_dir(self, snap: str = KR_SNAP) -> Path:
        return self.kr_root / "raw" / "raw_postgres" / f"snapshot_date={snap}" / "source=sj2_remote"

    def us_table_file(self, table: str, snap: str) -> Path:
        snapshot = f"snapshot_date={snap}"
        return self.us_root / "derived" / "snapshots" / table / snapshot / "part.parquet"


def sessions_of(calendar_id: str, start: date, end: date) -> list[date]:
    cal = SessionCalendar.from_exchange_calendars(calendar_id, start, end)
    assert cal is not None, "exchange_calendars가 있어야 합니다"
    return list(cal.sessions)


def _walk(
    seed: int, n: int, start: float = 100.0, drift: float = 0.0003, vol: float = 0.01
) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(start * np.exp(np.cumsum(rng.normal(drift, vol, n))))


def set_mtime(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def write_us_table(world_root: Path, table: str, snap: str, df: pl.DataFrame,
                   done: datetime = US_DONE) -> Path:
    d = world_root / "us" / "derived" / "snapshots" / table / f"snapshot_date={snap}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "part.parquet"
    df.write_parquet(path)
    set_mtime(path, done)
    return path


def write_kr_snapshot(
    world_root: Path, snap: str, finished_at: str, kr_sessions: list[date], *, seed: int = 1
) -> Path:
    src = world_root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={snap}" / "source=sj2_remote"
    specs = {}
    for k, a in enumerate(assets_for_market("KR")):
        group, name = kr_index_key(a.asset_id)
        specs[a.asset_id] = (group, name, _walk(seed + k, len(kr_sessions), 1000.0 + 50 * k))
    rows = []
    for _aid, (group, name, px) in specs.items():
        for d, v in zip(kr_sessions, px, strict=True):
            rows.append((d, group, group.upper(), name, Decimal(f"{v:.2f}")))
    index = pl.DataFrame(
        rows,
        schema={"bas_dd": pl.Date, "index_group": pl.String, "idx_clss": pl.String,
                "idx_nm": pl.String, "close_idx": pl.Decimal(18, 2)},
        orient="row",
    )
    for group in sorted(index["index_group"].unique().to_list()):
        d = src / "krx_index_daily" / "schema_version=1" / f"index_group={group}"
        d.mkdir(parents=True, exist_ok=True)
        index.filter(pl.col("index_group") == group).write_parquet(d / "part-000000.parquet")
    obs_rows = []
    series = {"rate_kr_cd91": (3.0, 0.0001), "fx_usdkrw_ecos": (1300.0, 0.4),
              "foreign_net_kospi_ecos": (100.0, 1.0), "trdval_kospi_ecos": (10000.0, 1.0)}
    horizon = [*kr_sessions, kr_sessions[-1] + timedelta(days=3)]
    for sid, (base, step) in series.items():
        for i, d in enumerate(kr_sessions):
            obs_rows.append((sid, d, Decimal(f"{base + step * i:.8f}"), horizon[i + 1],
                             datetime(2026, 10, 1, tzinfo=UTC)))
    obs = pl.DataFrame(
        obs_rows,
        schema={"series_id": pl.String, "observation_date": pl.Date,
                "value_numeric": pl.Decimal(20, 8), "available_from_date": pl.Date,
                "fetched_at": pl.Datetime("us", "UTC")},
        orient="row",
    )
    d = src / "common_feature_observation_raw" / "schema_version=1" / "source=ECOS"
    d.mkdir(parents=True, exist_ok=True)
    obs.write_parquet(d / "part-000000.parquet")
    manifests = src / "_manifests"
    (manifests / "table_manifests").mkdir(parents=True, exist_ok=True)
    pg = "00000017-00000127-1"
    tables = {}
    for table in ("krx_index_daily", "common_feature_observation_raw"):
        tm = manifests / "table_manifests" / f"{table}.json"
        tm.write_text(json.dumps({"source": {"name": "sj2_remote", "pg_snapshot_id": pg},
                                  "table": {"name": table}}))
        tables[table] = {"manifest_path": str(tm), "rows_exported": 1}
    (manifests / "_SUCCESS.json").write_text(json.dumps({
        "route": "remote", "started_at": finished_at, "finished_at": finished_at,
        "pg_snapshot_id": pg, "tables": tables}))
    return src


def build_world(
    tmp_path: Path, *, cfg: MsConfig | None = None, kr_last: date = KR_LAST
) -> World:
    cfg = cfg or MsConfig()
    root = tmp_path / "world"
    us_sessions = [s for s in sessions_of("XNYS", START, date(2027, 3, 31)) if s <= A]
    cal_sessions = sessions_of("XNYS", START, date(2027, 3, 31))
    symbols = {a.asset_id: a.proxy for a in assets_for_market("US")}
    price_rows, div_rows = [], []
    for k, (aid, sym) in enumerate(sorted(symbols.items())):
        px = _walk(10 + k, len(us_sessions), 80.0 + 10 * k)
        for d, v in zip(us_sessions, px, strict=True):
            price_rows.append((d, sym, Decimal(f"{v:.4f}"), Decimal(f"{v * 1.01:.4f}"),
                               Decimal(f"{v * 0.99:.4f}"), Decimal(f"{v:.4f}"), 1_000_000))
        for q in range(0, len(us_sessions), 63):
            div_rows.append((sym, us_sessions[q], "dividend", None, None, Decimal("0.50000")))
    prices = pl.DataFrame(
        price_rows,
        schema={"date": pl.Date, "symbol": pl.String, "open": pl.Decimal(14, 4),
                "high": pl.Decimal(14, 4), "low": pl.Decimal(14, 4), "close": pl.Decimal(14, 4),
                "volume": pl.Int64},
        orient="row",
    )
    actions = pl.DataFrame(
        div_rows,
        schema={"symbol": pl.String, "ex_date": pl.Date, "kind": pl.String,
                "to_factor": pl.Decimal(10, 5), "for_factor": pl.Decimal(10, 5),
                "amount": pl.Decimal(10, 5)},
        orient="row",
    )
    weekdays = [START - timedelta(days=30) + timedelta(days=i) for i in range(1200)]
    weekdays = [d for d in weekdays if d.weekday() < 5 and d <= date(2026, 10, 5)]
    macro_rows = []
    for k, sid in enumerate(SERIES):
        v = _walk(100 + k, len(weekdays), 20.0 if sid == "VIXCLS" else 4.0, 0.0, 0.003)
        for d, x in zip(weekdays, v, strict=True):
            macro_rows.append((sid, d, d, float(x)))
    macro = pl.DataFrame(
        macro_rows, schema=["series_id", "date", "realtime_start", "value"], orient="row")
    calendar = pl.DataFrame(
        {"date": cal_sessions, "exchange": ["XNYS"] * len(cal_sessions),
         "close_local": [time(16, 0)] * len(cal_sessions)})
    write_us_table(root, "prices_daily", US_SNAP, prices)
    write_us_table(root, "corp_actions", US_SNAP, actions)
    write_us_table(root, "macro_series", MACRO_SNAP, macro,
                   datetime(2026, 10, 5, 23, 40, tzinfo=SEOUL))
    write_us_table(root, "trading_calendar", CAL_SNAP, calendar,
                   datetime(2026, 9, 22, 0, 20, tzinfo=SEOUL))
    kr_sessions = [s for s in sessions_of("XKRX", START, date(2026, 12, 31)) if s <= kr_last]
    write_kr_snapshot(root, KR_SNAP, KR_DONE, kr_sessions)
    return World(root, cfg, us_sessions, kr_sessions)


# --------------------------------------------------------------------------- bundle
def make_bundle(world: World, out_dir: Path, *, null_stability_asset: str | None = None) -> Path:
    """합성 bundle. 계산 달력 지문은 이 세계의 달력에서 계산한다(동결 구간 = 합성 데이터 구간)."""
    cfg = world.cfg
    out_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    markets: dict[str, dict] = {}
    from modeler.scores.market_sector.bundle import sha256_file

    for market in ("US", "KR"):
        m = market.lower()
        asset_ids = sorted(a.asset_id for a in assets_for_market(market))
        n_cols = 18 + 2 + len(asset_ids)  # model_feature_columns: 연속형 18 + 지표 2 + 자산 원-핫
        rng = np.random.default_rng(7 if market == "US" else 8)
        X = rng.normal(size=(400, n_cols))
        y = rng.normal(size=400)
        yb = (rng.random(400) < 0.2).astype(float)
        w = np.ones(400)
        Xb = np.concatenate([X[:, 4:5], X[:, -len(asset_ids):]], axis=1)  # rvol_20 + 자산 원-핫
        models = {"p_opp_ridge": fit_model("ridge", X, y, w, cfg),
                  "p_stab_logit": fit_model("logit", X, yb, w, cfg),
                  "b_stab_logit_rvol": fit_model("logit", Xb, yb, w, cfg)}
        rel = {"run_manifest": f"{m}/manifest.json", "oof": f"{m}/oof_predictions.parquet",
               "latest_scores": f"{m}/latest_scores.json"}
        model_rel = {}
        (out_dir / m / "models" / "live").mkdir(parents=True)
        for name in MODEL_NAMES:
            model_rel[name] = f"{m}/models/live/{name}.joblib"
            joblib.dump(models[name], out_dir / model_rel[name])
        (out_dir / rel["run_manifest"]).write_text(json.dumps({"run_id": RUN_IDS[market]}))
        pl.DataFrame({"market": [market] * 400, "p_opp_ridge": rng.normal(size=400)}).write_parquet(
            out_dir / rel["oof"])
        latest = [{"asset_id": a, "session": "2026-09-28",
                   "stab_null_reason": "events_in_train<30" if a == null_stability_asset else None,
                   "model_train_boundary_label_end_before": "2026-09-28T23:30:00+00:00",
                   "n_train": 1234} for a in asset_ids]
        (out_dir / rel["latest_scores"]).write_text(json.dumps(latest))
        for r in [*rel.values(), *model_rel.values()]:
            files[r] = sha256_file(out_dir / r)
        if market == "US":
            sessions = [s for s in sessions_of("XNYS", START, A) if s >= world.us_sessions[0]]
            cal_man = {"calendar_id": "XNYS", "calendar_basis": f"lake_trading_calendar@{CAL_SNAP}",
                       "first_session": world.us_sessions[0].isoformat()}
        else:
            sessions = list(world.kr_sessions)
            from modeler.scores.market_sector.score_daily import xcals_basis

            cal_man = {"calendar_id": "XKRX", "calendar_basis": xcals_basis(),
                       "first_session": world.kr_sessions[0].isoformat()}
        markets[market] = {
            "run_id": RUN_IDS[market], **rel, "models": model_rel,
            "modeler_git_commit": "0" * 40,
            "panel": {"version": "synthetic", "manifest_sha256": "0" * 64, "features": "synthetic"},
            "calendar": calendar_spec(cal_man, sessions),
            "frozen_input_pins": {}, "frozen_latest_session": "2026-09-28",
            "assets": asset_ids,
            "train": {"boundary_label_end_before": "2026-09-28T23:30:00+00:00", "n_train": 1234},
        }
    (out_dir / "bundle.json").write_text(
        json.dumps(bundle_manifest(files, markets, cfg), indent=2, sort_keys=True) + "\n")
    return out_dir
