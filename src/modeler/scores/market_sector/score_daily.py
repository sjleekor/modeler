"""MS1 시장·섹터 일일 채점. 재학습 없이 동결 run의 고정 fit 모델로 채점만 한다.

``scores/``의 동결 코드(태그 ``ms-prereg-frozen``)는 고치지 않고 import해서 쓴다. 이 파일은
새 진입점이다. ``run.py``는 실행마다 재학습하고 기존 버전 디렉터리를 거부해서 매일 쓸 수
없다(계획 03 §4).

한 번의 실행이 하는 일

1. **bundle**: 동결 run 산출물(모델·OOF reference·run manifest)과 계산 달력 세션 sha256을 묶은
   디렉터리를 sha256으로 확인한다(``bundle.py``). 설정 해시·자산 레지스트리가 현재 코드와
   같아야 한다.
2. **입력**: ``ms-selection.json``(``inputs_pin.py``)이 고정한 snapshot 파일만 연다. 열기 전에
   sha256과 파일 목록을 다시 확인한다. 레이크의 "최신 snapshot"을 스스로 찾지 않는다.
3. **계산 달력**: MS1 패널은 KR을 ``exchange_calendars`` XKRX로, US를 레이크 ``trading_calendar``로
   만들었다. 이 세션 목록은 bundle의 달력 파일(``<시장>/calendar.json``)에 들어 있고, 채점은
   ``exchange_calendars``나 레이크 달력 대신 **이 파일만** 쓴다(운영 venv에 그 패키지가 없다).
   달력 기준·동결 구간 세션 sha256이 동결 때와 다르면 거부하고, 결정일 + 20일이 파일 범위
   (2027-12-31)를 넘으면 ``calendar_range_exhausted``로 거부한다. 관측 가격일 달력으로 대신하지
   않는다. 운영 달력(리포트를 만들지, K가 무엇인지)은 별개다(계획 03 §6.2).
4. **피쳐**: 가격 경로는 전체 이력으로 만든다(라벨·``b_opp_mean``이 전체 이력의 평균이다).
   피쳐는 최근 창(252세션 + 여유)만으로 만들고, 마지막 행의 가격 피쳐가 전체 패널 피쳐와
   같은지 확인한다.
5. **채점**: 시장 상태 파생값 + baseline 2개 + 연구용 점수 2개를 ``market-sector-D.json``으로
   낸다. 지수 종가 수준은 내지 않는다(Q1).

CLI (``python -m modeler.scores.market_sector.score_daily``)::

    score         --report-date D --selection SEL.json --bundle DIR --output OUT.json
    build-bundle  --stock-data-root ROOT --output DIR     # 동결 run -> bundle (읽기만, 맥에서)
    verify-frozen --stock-data-root ROOT --bundle DIR     # 동결 run과 같은 입력으로 재현 검사

종료 코드: 0 문서를 썼다(시장 하나만 실패해도 0), 1 시장 둘 다 실패(문서 없음), 2 사용법·입력
오류.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import math
import platform
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn

from modeler.etl.config import DataRoot
from modeler.scores.common.assets import (
    assets_for_market,
    kr_index_key,
    registry_hash,
    registry_version,
)
from modeler.scores.common.calendar import SessionCalendar
from modeler.scores.common.cash import build_cash_account, load_us_rates, rate_available_at
from modeler.scores.common.inputs import PinnedScopedLake
from modeler.scores.common.kr_inputs import (
    KrLake,
    load_kr_index_paths,
    load_kr_macro,
    load_kr_rates,
)
from modeler.scores.common.total_return import load_us_total_return
from modeler.scores.market_sector import baselines as bl
from modeler.scores.market_sector.build_panel import assemble
from modeler.scores.market_sector.bundle import (
    CALENDAR_COLUMNS,
    CALENDAR_SCHEMA,
    FROZEN_TAG,
    MARKETS,
    MODEL_NAMES,
    BundleError,
    read_calendar_file,
    sessions_sha256,
    sha256_file,
    verify_bundle_dir,
)
from modeler.scores.market_sector.bundle import (
    SCHEMA as BUNDLE_SCHEMA,
)
from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.daily_doc import SCHEMA
from modeler.scores.market_sector.features import (
    MACRO_SERIES_IDS,
    PRICE_FEATURES,
    PitSeries,
    build_features,
    compute_price_features,
    load_us_macro,
    model_feature_columns,
)
from modeler.scores.market_sector.inputs_pin import (
    US_TABLES,
    US_TABLES_FOR,
    InputChangedError,
    InputPinError,
    describe_kr_snapshot,
    describe_us_table,
    verify_pins,
)
from modeler.scores.market_sector.models import predict
from modeler.scores.market_sector.scoring import percentile_against, stability_scores

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
log = logging.getLogger("score_daily")

FROZEN_RUN_ID = {"US": "ms_us_202609300940", "KR": "ms_kr_202609300941"}
FROZEN_PANEL = {"US": "ms_panel_v3", "KR": "ms_panel_kr_v2"}
FROZEN_FEATURES = {"US": "ms_feat_v2", "KR": "ms_feat_kr_v1"}
#: 계산 달력을 만드는 ``exchange_calendars`` 버전. MS1 동결 때와 같다(uv.lock).
#: 다르면 build-bundle이 멈춘다.
XCALS_PIN = "exchange_calendars==4.13.2"
CASH_SERIES = {"US": "DGS3MO", "KR": "rate_kr_cd91"}
#: 피쳐 창: 가장 긴 lookback(252세션) + 현재 행 + 여유. 여유는 창 피쳐와 전체 패널 피쳐를
#: 비교하는 구간이다.
WINDOW_MARGIN_SESSIONS = 64
#: 창 피쳐와 전체 패널 피쳐가 같다고 보는 허용 오차(rolling 합의 부동소수 오차 수준).
WINDOW_TOLERANCE = 1e-9
#: 이 값을 넘게 늦은 입력은 section status를 stale로 둔다. KR 지수는 T+1 공표라 1세션
#: 늦은 것이 정상이다.
NOMINAL_LAG_SESSIONS = {"US": 0, "KR": 1}
#: 계산 달력 파일이 덮는 마지막 날. 이 날 뒤로 가려면 bundle을 다시 만들어야 한다.
CALENDAR_RANGE_END = date(2027, 12, 31)
#: 결정일 + 이 일수가 달력 범위 끝 안에 있어야 채점한다(다음 세션 개장 시각·진입일 계산 여유).
CALENDAR_TAIL_DAYS = 20
CALENDAR_GENERATED_BY = "modeler.scores.market_sector.score_daily build-bundle (맥, 네트워크 없음)"

LAB_COLS = [
    "asset_id",
    "session",
    "label_end_at",
    "label_matured",
    "total_return_60d",
    "excess_return_60d_vs_cash",
    "excess_return_60d_vs_market",
    "loss_event_60d_8pct",
]
TABLE_DISPLAY = {
    "us_spx": "SPY (S&P 500)",
    "us_ndx": "QQQ (Nasdaq 100)",
    "us_fin": "XLF (금융)",
    "us_hlth": "XLV (헬스케어)",
    "us_ind": "XLI (산업재)",
    "us_enrg": "XLE (에너지)",
    "us_tech": "XLK (정보기술)",
    "kr_kospi": "코스피",
    "kr_kosdaq": "코스닥",
    "kr_fin": "KRX 은행",
    "kr_hlth": "KRX 헬스케어",
    "kr_ind": "KRX 기계장비",
    "kr_enrg": "KRX 에너지화학",
    "kr_tech": "KRX 반도체",
}
#: MS1 판정 (05_result.md, 2026-09-30). 렌더러가 읽는 한글 판정과 사유. bundle.json에도 들어간다.
VERDICT_SOURCE = (
    "milestones/common/scores/20260929_market_sector_modeling/05_result.md (2026-09-30)"
)
VERDICTS = {
    "US": {
        "opportunity": ("실패", "Ridge skill -0.053 [-0.231, +0.080] (baseline: 자산별 과거 평균)"),
        "sector_relative": (
            "보류", "Ridge IC +0.100 [-0.016, +0.200], baseline IC +0.093과 거의 같음"),
        "stability": ("실패", "Logistic Brier skill -0.255 [-0.469, -0.139], 예측이 과신"),
    },
    "KR": {
        "opportunity": ("실패", "Ridge skill -0.091 [-0.273, +0.049] (price_only)"),
        "sector_relative": ("실패", "Ridge IC -0.061 [-0.154, +0.034]"),
        "stability": ("실패", "Logistic Brier skill -0.118 [-0.187, -0.046], 예측이 과신"),
    },
}


# --------------------------------------------------------------------------- 예외
class CalendarMismatchError(RuntimeError):
    """계산 달력이 동결 때와 다르다(버전·기준·동결 구간 세션 sha256). 대신 쓸 달력은 없다."""


class CalendarRangeExhaustedError(RuntimeError):
    """결정일(+20일)이 bundle 달력 파일의 범위 끝을 넘었다. bundle을 다시 만들어야 한다."""


class WindowMismatchError(RuntimeError):
    """창 피쳐가 전체 패널 피쳐와 다르다."""


class MarketInputError(RuntimeError):
    """입력 표에 채점에 필요한 값이 없다."""


#: 실패 사유 코드. 문서에는 코드와 예외 클래스 이름만 적는다(경로·원문 메시지는 싣지 않는다).
def failure_reason(exc: BaseException) -> str:
    if isinstance(exc, InputChangedError):
        return "input_changed"
    if isinstance(exc, InputPinError):
        return "input_unavailable"
    if isinstance(exc, CalendarRangeExhaustedError):
        return "calendar_range_exhausted"
    if isinstance(exc, CalendarMismatchError):
        return "calendar_mismatch"
    if isinstance(exc, WindowMismatchError):
        return "window_mismatch"
    if isinstance(exc, BundleError):
        return "bundle_invalid"
    if isinstance(exc, MarketInputError):
        return "input_missing"
    return "compute_error"


# --------------------------------------------------------------------------- 공용
def sessions_sha(sessions: Sequence[date]) -> str:
    return sessions_sha256(d.isoformat() for d in sessions)


def fnum(x: Any, nd: int | None = None) -> float | None:
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd) if nd is not None else x


def _jdefault(o: Any) -> Any:
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _json_text(body: Any) -> str:
    return json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2, default=_jdefault) + "\n"


def _write_text_atomic(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{id(data)}.tmp")
    try:
        temp.write_bytes(data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- bundle 읽기
@dataclass
class MarketBundle:
    market: str
    directory: Path
    spec: dict[str, Any]
    oof_ref: np.ndarray
    oof_rows: int
    latest: dict[str, dict[str, Any]]
    _models: dict[str, Any] = field(default_factory=dict)
    _calendar: SessionCalendar | None = None

    def model(self, name: str) -> Any:
        if name not in self._models:
            self._models[name] = joblib.load(self.directory / self.spec["models"][name])
        return self._models[name]

    @property
    def calendar(self) -> dict[str, Any]:
        return self.spec["calendar"]

    def session_calendar(self) -> SessionCalendar:
        """bundle 달력 파일의 계산 달력. 파일 형식과 내부 지문은 읽을 때 확인한다."""
        if self._calendar is None:
            self._calendar = calendar_from_file(self.directory / self.calendar["file"])
        return self._calendar

    def model_sha(self) -> dict[str, str]:
        files = self.spec["_files"]
        return {name: files[rel] for name, rel in sorted(self.spec["models"].items())}


@dataclass
class Bundle:
    directory: Path
    manifest: dict[str, Any]
    sha256: str
    cfg: MsConfig
    markets: dict[str, MarketBundle]


def load_bundle(
    path: Path, *, expected_sha256: str | None = None, cfg: MsConfig | None = None
) -> Bundle:
    """bundle 디렉터리(또는 그 안의 bundle.json)를 읽고 모든 파일의 sha256을 확인한다."""
    path = Path(path)
    directory = path.parent if path.name == "bundle.json" else path
    cfg = cfg or MsConfig()
    manifest = verify_bundle_dir(directory, expected_sha256=expected_sha256)
    if manifest["config_hash"] != cfg.config_hash():
        raise BundleError("bundle 설정 해시가 현재 MsConfig와 다릅니다")
    reg = manifest["asset_registry"]
    if reg["version"] != registry_version() or reg["hash"] != registry_hash():
        raise BundleError("bundle 자산 레지스트리가 현재 코드와 다릅니다")
    markets: dict[str, MarketBundle] = {}
    for market in MARKETS:
        spec = {**manifest["markets"][market], "_files": manifest["files"]}
        oof = pl.read_parquet(directory / spec["oof"])
        ref = oof.filter(pl.col("market") == market)["p_opp_ridge"].to_numpy()
        latest = json.loads((directory / spec["latest_scores"]).read_text(encoding="utf-8"))
        markets[market] = MarketBundle(
            market=market,
            directory=directory,
            spec=spec,
            oof_ref=ref,
            oof_rows=int(oof.filter(pl.col("market") == market).height),
            latest={r["asset_id"]: r for r in latest},
        )
    return Bundle(directory, {k: v for k, v in manifest.items() if k != "_sha256"},
                  manifest["_sha256"], cfg, markets)


# --------------------------------------------------------------------------- 계산 달력
def verify_calendar(cal: SessionCalendar, spec: dict[str, Any]) -> dict[str, Any]:
    """달력 기준과 동결 구간 세션 sha256이 bundle과 같은지 확인한다. 다르면 거부한다."""
    if spec["kind"] == "exchange_calendars":
        if cal.calendar_basis != spec["basis"]:
            raise CalendarMismatchError(
                f"계산 달력 기준이 {cal.calendar_basis!r}입니다. "
                f"동결 패널은 {spec['basis']!r}입니다"
            )
    elif not cal.calendar_basis.startswith(spec["basis_prefix"]):
        raise CalendarMismatchError(
            f"계산 달력 기준이 {cal.calendar_basis!r}입니다. {spec['basis_prefix']!r}여야 합니다"
        )
    if cal.calendar_id != spec["calendar_id"]:
        raise CalendarMismatchError("계산 달력 id가 동결 때와 다릅니다")
    lo, hi = date.fromisoformat(spec["first_session"]), date.fromisoformat(spec["last_session"])
    inside = [s for s in cal.sessions if lo <= s <= hi]
    sha = sessions_sha(inside)
    if len(inside) != spec["n_sessions"] or sha != spec["sessions_sha256"]:
        raise CalendarMismatchError(
            f"{cal.calendar_id} 동결 구간 {lo}~{hi}의 세션 목록이 동결 때와 다릅니다 "
            f"(세션 {len(inside)}개, sha256 {sha[:12]})"
        )
    return {
        "calendar_id": cal.calendar_id,
        "basis": cal.calendar_basis,
        "frozen_range": [lo.isoformat(), hi.isoformat()],
        "frozen_sessions": len(inside),
        "frozen_sessions_sha256": sha,
    }


def calendar_from_file(path: Path) -> SessionCalendar:
    """달력 파일(``market-sector-calendar.v1``)을 ``SessionCalendar``로 읽는다.

    ``exchange_calendars``는 쓰지 않는다.
    """
    body = read_calendar_file(path)
    rows = body["sessions"]
    return SessionCalendar(
        body["calendar_id"],
        body["calendar_basis"],
        tuple(date.fromisoformat(r[0]) for r in rows),
        tuple(datetime.fromisoformat(r[1]).astimezone(UTC) for r in rows),
        tuple(datetime.fromisoformat(r[2]).astimezone(UTC) for r in rows),
    )


def bundle_calendar(bundle_m: MarketBundle) -> tuple[SessionCalendar, dict[str, Any]]:
    """bundle 달력 파일의 계산 달력과, 동결 때와 같은지 확인한 결과(문서 provenance)."""
    cal = bundle_m.session_calendar()
    check = verify_calendar(cal, bundle_m.calendar)
    file_rel = bundle_m.calendar["file"]
    check.update(
        source="bundle_calendar_file",
        file=file_rel,
        file_sha256=bundle_m.spec["_files"][file_rel],
        file_range_end=bundle_m.calendar["range_end"],
        file_sessions=len(cal.sessions),
    )
    return cal, check


def ensure_calendar_range(spec: dict[str, Any], market: str, decision: date) -> None:
    """결정일 + ``CALENDAR_TAIL_DAYS``가 달력 파일 범위 안에 있어야 한다. 아니면 거부한다."""
    range_end = date.fromisoformat(spec["range_end"])
    if decision + timedelta(days=CALENDAR_TAIL_DAYS) > range_end:
        raise CalendarRangeExhaustedError(
            f"{market} 계산 달력 파일은 {range_end}까지입니다. 결정일 {decision} + "
            f"{CALENDAR_TAIL_DAYS}일이 그 뒤입니다. bundle을 다시 만드십시오"
        )


# --------------------------------------------------------------------------- 시장 입력
@dataclass
class Frames:
    """한 시장의 계산 입력. 가격 경로·라벨은 전체 이력이고 피쳐는 창에서 만든다."""

    market: str
    cal: SessionCalendar
    calendar_check: dict[str, Any]
    panel: pl.DataFrame
    labels: pl.DataFrame
    rates: pl.DataFrame
    macro_rows: pl.DataFrame
    kr_macro: Any
    inputs: dict[str, Any]
    max_dates: dict[str, Any]


def _input_summary(table_rec: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in table_rec.items()
        if k in ("snapshot_date", "completed_at", "export_finished_at", "pg_snapshot_id")
    } | {"files": {n: f["sha256"] for n, f in table_rec["files"].items()}}


def _macro_max_dates(macro_rows: pl.DataFrame) -> dict[str, Any]:
    latest = (
        macro_rows.group_by("series_id")
        .agg(pl.col("date").max().alias("max_obs"),
             pl.col("realtime_start").max().alias("max_rt"))
        .sort("series_id")
    )
    return {r[0]: {"max_obs": r[1], "max_realtime_start": r[2]} for r in latest.rows()}


def build_frames(
    market: str, selection: dict[str, Any], bundle_m: MarketBundle, cfg: MsConfig
) -> Frames:
    """selection이 고정한 입력만 연다. 먼저 sha256과 파일 목록을 다시 확인한다."""
    decision = [date.fromisoformat(selection["report_date"])]
    if selection["limits"].get(market):
        decision.append(date.fromisoformat(selection["limits"][market]))
    ensure_calendar_range(bundle_m.calendar, market, max(decision))  # 싼 검사를 먼저 한다
    verify_pins(selection, market)
    roots = selection["roots"]
    us_tables = selection["us"]["tables"]
    assets = list(assets_for_market(market))
    if market == "US":
        us_root = DataRoot(Path(roots["us"]))
        symbols = {a.asset_id: a.proxy for a in assets}
        snaps = {t: us_tables[t]["snapshot_date"] for t in US_TABLES_FOR["US"]}
        lake = PinnedScopedLake(root=us_root, snapshots=snaps, symbols=tuple(symbols.values()))
        cal, check = bundle_calendar(bundle_m)
        paths, _ = load_us_total_return(lake, symbols, sessions=frozenset(cal.sessions))
        rates = load_us_rates(lake, CASH_SERIES["US"])
        if rates is None:
            raise MarketInputError("macro_series에 DGS3MO가 없습니다")
        kr_macro = None
        macro_rows = load_us_macro(lake)
        inputs = {t: _input_summary(us_tables[t]) for t in US_TABLES_FOR["US"]}
        ca_max = lake.scan("corp_actions").select(pl.col("ex_date").max()).collect().item()
        max_dates = {
            "prices_daily_by_asset": {a: paths[a]["session"].max() for a in paths},
            "corp_actions_max_ex_date": ca_max,
            "macro_series": _macro_max_dates(macro_rows),
            "calendar": {"first": cal.sessions[0], "last": cal.sessions[-1]},
        }
    else:
        kr_root = DataRoot(Path(roots["kr"]))
        kr = selection["kr"]
        kr_lake = KrLake.resolve(kr_root, snapshot_date=kr["snapshot_date"])
        keys = {a.asset_id: kr_index_key(a.asset_id) for a in assets}
        cal, check = bundle_calendar(bundle_m)
        paths, _ = load_kr_index_paths(kr_lake, keys, sessions=frozenset(cal.sessions))
        rates = load_kr_rates(kr_lake, CASH_SERIES["KR"], cal.sessions)
        if rates is None:
            raise MarketInputError("common_feature_observation_raw에 rate_kr_cd91이 없습니다")
        kr_macro = load_kr_macro(kr_lake, cal.sessions)
        macro_snap = us_tables["macro_series"]["snapshot_date"]
        us_lake = PinnedScopedLake(
            root=DataRoot(Path(roots["us"])), snapshots={"macro_series": macro_snap}, symbols=()
        )
        macro_rows = load_us_macro(us_lake)
        inputs = {t: _input_summary({**kr["tables"][t], "snapshot_date": kr["snapshot_date"],
                                     "export_finished_at": kr["export_finished_at"],
                                     "pg_snapshot_id": kr["pg_snapshot_id"]})
                  for t in kr["tables"]}
        inputs["macro_series(US, KR 피쳐용)"] = _input_summary(us_tables["macro_series"])
        co = (
            kr_lake.scan("common_feature_observation_raw")
            .filter(pl.col("series_id").is_in(
                ["rate_kr_cd91", "fx_usdkrw_ecos", "foreign_net_kospi_ecos", "trdval_kospi_ecos"]))
            .group_by("series_id")
            .agg(pl.col("observation_date").cast(pl.Date).max().alias("max_obs"),
                 pl.col("available_from_date").cast(pl.Date).max().alias("max_avail"))
            .sort("series_id")
            .collect()
        )
        max_dates = {
            "krx_index_daily_by_asset": {a: paths[a]["session"].max() for a in paths},
            "common_feature_observation_raw": {
                r[0]: {"max_obs": r[1], "max_available_from": r[2]} for r in co.rows()},
            "macro_series(US)": _macro_max_dates(macro_rows),
            "calendar": {"first": cal.sessions[0], "last": cal.sessions[-1]},
        }
    cash = build_cash_account(
        rates, cal.sessions, series_id=CASH_SERIES[market], staleness_days=cfg.cash_staleness_days
    )
    panel, labels = assemble(assets, paths, cal, cash)
    return Frames(market, cal, check, panel, labels, rates, macro_rows, kr_macro, inputs, max_dates)


# --------------------------------------------------------------------------- 피쳐: 창과 전체
def window_panel(panel: pl.DataFrame, limit: date, rows: int) -> pl.DataFrame:
    """자산마다 ``session <= limit``인 마지막 ``rows``행."""
    return (
        panel.sort(["asset_id", "session"])
        .filter(pl.col("session") <= limit)
        .group_by("asset_id", maintain_order=True)
        .tail(rows)
    )


def full_price_frame(frames: Frames, cfg: MsConfig) -> pl.DataFrame:
    """전체 이력의 가격 피쳐·라벨·PIT ``b_opp_mean``. ``b_opp_mean``은 전체 라벨 이력의 평균이다.

    ``baselines.py`` 정의 그대로다: 그 행의 결정 시각 이전에 만기된(``label_end_at <
    decision_at``) 성숙 라벨 중 학습 후보 행(``feature_ready``이고 ``session >= train_start``)의
    자산별 평균.
    """
    pf = compute_price_features(frames.panel, cfg).sort(["asset_id", "session"])
    frame = pf.join(frames.labels.select(LAB_COLS), on=["asset_id", "session"], how="left").sort(
        ["asset_id", "session"]
    )
    y = frame["excess_return_60d_vs_cash"].cast(pl.Float64).to_numpy()
    pool = (
        frame["label_matured"].fill_null(False)
        & frame["feature_ready"].fill_null(False)
        & (frame["session"] >= cfg.train_start[frames.market])
    ).to_numpy()
    b_opp = bl.pit_expanding_mean(
        frame["asset_id"].to_numpy(),
        bl.epoch_us(frame["decision_at"]),
        bl.epoch_us(frame["label_end_at"]),
        y,
        pool,
    )
    return frame.with_columns(pl.Series("b_opp_mean", b_opp))


def window_features(frames: Frames, limit: date, cfg: MsConfig) -> pl.DataFrame:
    """최근 창만으로 만든 피쳐 프레임(자산·세션당 한 행).

    거시 피쳐 계산이 이력 길이에 비례해 느려서 창만 쓴다.
    """
    win = window_panel(
        frames.panel, limit, cfg.max_lookback_sessions + 1 + WINDOW_MARGIN_SESSIONS
    )
    return build_features(
        win, frames.macro_rows, cfg, market=frames.market, kr_macro=frames.kr_macro
    ).sort(["asset_id", "session"])


def window_vs_full_diff(
    feats_w: pl.DataFrame, full: pl.DataFrame, last_n: int | None = None
) -> dict[str, Any]:
    """창 가격 피쳐와 전체 패널 가격 피쳐를 (자산, 세션)으로 맞춰 비교한다.

    ``last_n``: 자산마다 창의 마지막 ``last_n``행만(None이면 마지막 행 하나). 창 앞부분은 워밍업이라
    비교하지 않는다. 반환: 비교한 행 수, 최대 절대 차이, null 불일치 수, feature_ready 불일치 수.
    """
    n = 1 if last_n is None else last_n
    w = feats_w.group_by("asset_id", maintain_order=True).tail(n)
    j = w.select("asset_id", "session", "feature_ready", *PRICE_FEATURES).join(
        full.select("asset_id", "session", pl.col("feature_ready").alias("_fr_full"),
                    *[pl.col(c).alias(f"_full_{c}") for c in PRICE_FEATURES]),
        on=["asset_id", "session"], how="left",
    )
    worst, null_mismatch = 0.0, 0
    for c in PRICE_FEATURES:
        a, b = j[c], j[f"_full_{c}"]
        null_mismatch += int((a.is_null() != b.is_null()).sum())
        both = a.is_not_null() & b.is_not_null()
        if both.any():
            d = (a.filter(both).cast(pl.Float64) - b.filter(both).cast(pl.Float64)).abs().max()
            worst = max(worst, float(d))
    ready_w = j["feature_ready"].fill_null(False)
    ready_mismatch = int((ready_w != j["_fr_full"].fill_null(False)).sum())
    return {"rows": j.height, "max_abs_diff": worst, "null_mismatch": null_mismatch,
            "feature_ready_mismatch": ready_mismatch}


# --------------------------------------------------------------------------- 채점
def _pct_or_none(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def requested_session(cal: SessionCalendar, limit: date) -> date:
    k = bisect.bisect_right(cal.sessions, limit) - 1
    if k < 0:
        raise ValueError(f"{cal.calendar_id}: {limit} 이하 세션이 없습니다")
    return cal.sessions[k]


def _x_matrix(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def macro_frontier(
    macro_rows: pl.DataFrame, decision_at: datetime, ids: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sid in ids:
        sub = macro_rows.filter(pl.col("series_id") == sid)
        if sub.height == 0:
            out[sid] = {"obs_date": None, "age_days": None}
            continue
        r = PitSeries(sub).lookup([decision_at], (0,))[0]
        od = r["obs_date"][0]
        age = (decision_at.astimezone(UTC).date() - od).days if od else None
        out[sid] = {"obs_date": od, "age_days": age}
    return out


@dataclass
class MarketResult:
    market: str
    rows: list[dict[str, Any]]
    meta: dict[str, Any]
    warnings: list[str]
    frames: Frames
    macro_front: dict[str, dict[str, Any]]
    kr_series_obs: dict[str, Any]
    window_check: dict[str, Any]
    #: 반올림 전 원값: asset_id -> p_opp, p_stab, opportunity_score, stability_score, ...
    raw: dict[str, dict[str, Any]] = field(default_factory=dict)


def score_market(
    frames: Frames, bundle_m: MarketBundle, cfg: MsConfig, limit: date
) -> MarketResult:
    """고정 fit 채점. 자산마다 ``session <= limit``인 마지막 행의 값을 낸다."""
    market = frames.market
    full = full_price_frame(frames, cfg)
    feats_w = window_features(frames, limit, cfg)
    check = window_vs_full_diff(feats_w, full)
    if (check["max_abs_diff"] > WINDOW_TOLERANCE or check["null_mismatch"]
            or check["feature_ready_mismatch"]):
        raise WindowMismatchError(
            f"창 피쳐가 전체 패널 피쳐와 다릅니다 (최대 차이 {check['max_abs_diff']:.3e}, "
            f"null 불일치 {check['null_mismatch']}, ready 불일치 {check['feature_ready_mismatch']})"
        )
    registry = list(assets_for_market(market))
    asset_ids = sorted(a.asset_id for a in registry)
    last = feats_w.group_by("asset_id", maintain_order=True).tail(1).sort("asset_id")
    if last["asset_id"].to_list() != asset_ids:
        raise MarketInputError("창에 자산 행이 모자랍니다: " + ", ".join(sorted(
            set(asset_ids) - set(last["asset_id"].to_list()))))
    cols = model_feature_columns(asset_ids)
    bcols = ["rvol_20", *[c for c in cols if c.startswith("asset_")]]
    X, Xb = _x_matrix(last, cols), _x_matrix(last, bcols)
    ready = last["feature_ready"].fill_null(False).to_numpy()
    p_opp = predict(bundle_m.model("p_opp_ridge"), X, "ridge")
    p_stab = predict(bundle_m.model("p_stab_logit"), X, "logit")
    p_b = predict(bundle_m.model("b_stab_logit_rvol"), Xb, "logit")
    score_opp = percentile_against(bundle_m.oof_ref, p_opp, cfg.opportunity_reference_min_oof)
    score_stab = stability_scores(p_stab)
    b_opp_by = {
        r["asset_id"]: r["b_opp_mean"]
        for r in last.select("asset_id", "session").join(
            full.select("asset_id", "session", "b_opp_mean"), on=["asset_id", "session"], how="left"
        ).iter_rows(named=True)
    }
    scored: dict[str, dict[str, Any]] = {}
    for k, aid in enumerate(asset_ids):
        reason = bundle_m.latest.get(aid, {}).get("stab_null_reason")
        ok = bool(ready[k])
        stab_ok = ok and reason is None
        b_opp = b_opp_by[aid]
        scored[aid] = {
            "feature_ready": ok,
            "p_opp_ridge": float(p_opp[k]),
            "p_stab_logit": float(p_stab[k]),
            "opportunity_score": float(score_opp[k]) if ok and not np.isnan(score_opp[k]) else None,
            "stability_score": float(score_stab[k]) if stab_ok else None,
            "b_stab_prob": float(p_b[k]) if stab_ok else None,
            "b_opp_mean": None if b_opp is None or np.isnan(b_opp) else float(b_opp),
            "stab_null_reason": reason,
        }
    # 시장 상태 표: 단순수익률은 tr_index_t 비율이다(창 안에서 60행 뒤를 본다)
    window_rows = cfg.max_lookback_sessions + 1 + WINDOW_MARGIN_SESSIONS
    tr = pl.col("tr_index_t")
    state = window_panel(frames.panel, limit, window_rows).select(
        "asset_id", "session", "last_price_session", "return_basis",
        *[(tr / tr.shift(k).over("asset_id") - 1.0).alias(f"sret_{k}") for k in (1, 5, 20, 60)],
    )
    state_by = {(r["asset_id"], r["session"]): r for r in state.iter_rows(named=True)}
    cal = frames.cal
    req = requested_session(cal, limit)
    warns: list[str] = []
    rows: list[dict[str, Any]] = []
    asof_dates: list[date] = []
    max_lag = 0
    dec_ref: datetime | None = None
    last_by = {r["asset_id"]: r for r in last.iter_rows(named=True)}
    for a in registry:
        aid = a.asset_id
        feat = last_by[aid]
        sess = feat["session"]
        srow = state_by[(aid, sess)]
        asof_d = srow["last_price_session"]
        lag = cal.index_of(req) - cal.index_of(asof_d)
        max_lag = max(max_lag, lag)
        asof_dates.append(asof_d)
        if dec_ref is None:
            dec_ref = feat["decision_at"]
        sc = scored[aid]
        ra = rate_available_at(frames.rates, [asof_d]).row(0, named=True)
        cash_rate, cash_obs, cash_status = None, ra["rate_obs_date"], "ok"
        if ra["rate_pct"] is None:
            cash_status = "no_rate_yet"
        elif (asof_d - ra["rate_obs_date"]).days > cfg.cash_staleness_days:
            cash_status = "stale"
        else:
            cash_rate = ra["rate_pct"] / 100.0
        rows.append({
            "asset_id": aid,
            "name": TABLE_DISPLAY[aid],
            "market": market,
            "group": a.asset_type,
            "return_basis": srow["return_basis"],
            "asof_date": asof_d,
            "ret_1": fnum(srow["sret_1"], 6),
            "ret_5": fnum(srow["sret_5"], 6),
            "ret_20": fnum(srow["sret_20"], 6),
            "ret_60": fnum(srow["sret_60"], 6),
            "rvol_20": fnum(feat["rvol_20"], 6),
            "dd_252": fnum(feat["dd_252"], 6),
            "cash_rate": fnum(cash_rate, 6),
            "b_opp_mean_pct": fnum(_pct_or_none(sc["b_opp_mean"]), 4),
            "b_stab_prob_pct": fnum(_pct_or_none(sc["b_stab_prob"]), 4),
            "opportunity_score": fnum(sc["opportunity_score"], 4),
            "stability_score": fnum(sc["stability_score"], 4),
            "cash_rate_obs_date": cash_obs,
            "cash_status": cash_status,
            "lag_sessions": lag,
        })
        if not sc["feature_ready"]:
            warns.append(f"{aid}: feature_ready=False (252세션 이력 부족) — 점수 null")
        if sc["stab_null_reason"]:
            warns.append(f"{aid}: Stability null ({sc['stab_null_reason']})")
        if cash_status != "ok":
            warns.append(f"{aid}: cash_rate null ({cash_status})")
    assert dec_ref is not None
    macro_front = macro_frontier(frames.macro_rows, dec_ref, MACRO_SERIES_IDS)
    kr_series: dict[str, Any] = {}
    if market == "KR":
        kr_macro = frames.kr_macro
        for nm, series in (("usdkrw", kr_macro.fx), ("foreign_net", kr_macro.foreign_net),
                           ("trdval", kr_macro.trdval)):
            kr_series[nm] = PitSeries(series).lookup([dec_ref], (0,))[0]["obs_date"][0]
    meta = {
        "requested_limit": limit,
        "requested_session": req,
        "asof_min": min(asof_dates),
        "asof_max": max(asof_dates),
        "lag_sessions": max_lag,
        "decision_at": dec_ref,
    }
    return MarketResult(market, rows, meta, warns, frames, macro_front, kr_series, check, scored)


# --------------------------------------------------------------------------- 문서
def verdict_block(bundle: Bundle) -> tuple[dict[str, Any], dict[str, Any]]:
    """(렌더러가 읽는 한글 판정, 판정 근거 전체)."""
    v = bundle.manifest["verdicts"]
    short = {m: {k: d["label"] for k, d in v[m].items()} for m in MARKETS}
    return short, v


def compose_document(
    *,
    report_date: date,
    bundle: Bundle,
    selection: dict[str, Any],
    selection_sha256: str,
    results: dict[str, MarketResult],
    failures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cfg = bundle.cfg
    us, kr = results.get("US"), results.get("KR")
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    warns: list[str] = []
    for r in (us, kr):
        if r is not None:
            rows += r.rows
            warns += r.warnings
    asof: dict[str, Any] = {}
    asof_detail: dict[str, Any] = {}
    stale = False
    if kr is not None:
        asof["KR"] = kr.meta["asof_max"]
        asof_detail["KR"] = {
            "requested_K": kr.meta["requested_limit"],
            "requested_session": kr.meta["requested_session"],
            "asof_min": kr.meta["asof_min"],
            "lag_sessions": kr.meta["lag_sessions"],
            "decision_at": kr.meta["decision_at"],
            "series_obs": {**{s: v["obs_date"] for s, v in kr.macro_front.items()},
                           **kr.kr_series_obs},
        }
        stale |= kr.meta["lag_sessions"] > NOMINAL_LAG_SESSIONS["KR"]
    if us is not None:
        feat_obs = [v["obs_date"] for v in us.macro_front.values() if v["obs_date"]]
        asof["US"] = us.meta["asof_max"]
        asof["US_macro"] = max(feat_obs) if feat_obs else None
        asof_detail["US"] = {
            "requested_A": us.meta["requested_limit"],
            "requested_session": us.meta["requested_session"],
            "asof_min": us.meta["asof_min"],
            "lag_sessions": us.meta["lag_sessions"],
            "decision_at": us.meta["decision_at"],
            "macro_frontier": us.macro_front,
        }
        stale |= us.meta["lag_sessions"] > NOMINAL_LAG_SESSIONS["US"]
    runs = {m: bundle.manifest["markets"][m] for m in MARKETS}
    boundary = {m: runs[m]["train"] for m in MARKETS}
    notes.append(
        "모델 점수는 MS1 동결 run({}·{})의 고정 fit을 다시 학습하지 않고 채점한 연구용 값입니다. "
        "학습 경계는 US label_end_at < {}(행 {:,}개), KR < {}(행 {:,}개)이고, 리포트 날짜가 "
        "지나도 모델은 바뀌지 않습니다.".format(
            runs["US"]["run_id"], runs["KR"]["run_id"],
            boundary["US"]["boundary_label_end_before"], boundary["US"]["n_train"],
            boundary["KR"]["boundary_label_end_before"], boundary["KR"]["n_train"]))
    kr_xcals = runs["KR"]["calendar"]["basis"].split("==")[-1]
    notes += [
        "ret_k는 계산 달력 기준 k세션 단순수익률(TR_t/TR_(t-k)-1)입니다. US는 총수익(배당 포함), "
        "KR은 price_only(배당 제외)입니다. MS1 모델 입력 ret_20·ret_60은 같은 값의 "
        "로그수익률입니다.",
        "rvol_20은 일별 log 수익률 20개의 표준편차(ddof=1)에 sqrt(252)를 곱한 연율화 "
        "값입니다(MS1 features 정의). dd_252는 TR 지수의 252세션 최고점 대비 낙폭입니다. "
        "지수 종가 수준은 출력하지 않습니다.",
        f"KR 계산 달력(exchange_calendars XKRX {kr_xcals})에는 2026-06-03·07-17이 세션으로 "
        "들어 있고 앞 값을 채웁니다(MS1 동결 방식). 이 두 날을 포함하는 창(ret_60 등)은 실제 "
        "거래일 수가 하루나 이틀 적습니다.",
        "cash_rate는 연 금리의 소수 표기(0.042=4.2%)입니다. MS1 cash 모듈 규칙대로 각 시장 기준 "
        "세션 날짜에 이용 가능했던 가장 최근 값(US DGS3MO, KR CD91)이고 "
        f"{cfg.cash_staleness_days}일 넘게 묵으면 null입니다.",
        "b_opp_mean_pct는 60일 현금 대비 초과수익의 PIT 자산별 과거 평균(%)입니다. 결정 시각 "
        "이전에 만기된 성숙 라벨만 쓰고 전체 이력으로 계산합니다. b_stab_prob_pct는 "
        "b_stab_logit_rvol live 모델의 60일 안 -8% 손실 확률(%)입니다. 둘 다 0에서 100 사이로 "
        "바꾸는 변환을 만들지 않고 원래 단위입니다.",
    ]
    kr_sel, us_tables = selection["kr"], selection["us"]["tables"]
    parts = []
    if kr is not None:
        parts.append("KR raw snapshot {} (export 완료 {})".format(
            kr_sel["snapshot_date"], kr_sel["export_finished_at"]))
    if us is not None:
        parts.append("US 가격·배당 snapshot {}".format(us_tables["prices_daily"]["snapshot_date"]))
    if parts:
        notes.append(
            "입력은 select 단계가 D 09:30 이전에 끝난 snapshot으로 고정했습니다: "
            + ", ".join(parts) + ". 입력 파일은 고정한 뒤 sha256을 다시 확인하고 읽었습니다.")
    if kr is not None and kr.meta["lag_sessions"] > NOMINAL_LAG_SESSIONS["KR"]:
        lag = kr.meta["lag_sessions"]
        notes.append(f"KR 지수가 기준 세션보다 {lag}세션 늦습니다(정상은 1세션).")
    if us is not None and us.meta["lag_sessions"] > NOMINAL_LAG_SESSIONS["US"]:
        notes.append(f"US 가격이 기준 세션보다 {us.meta['lag_sessions']}세션 늦습니다.")
    stale_macro = sorted(
        {s for r in (us, kr) if r is not None for s, v in r.macro_front.items()
         if v["age_days"] is None or v["age_days"] > cfg.macro_staleness_days})
    if stale_macro:
        notes.append(
            f"거시 입력 {stale_macro}가 {cfg.macro_staleness_days}일 넘게 묵어 해당 피쳐는 "
            "null이고 학습 중앙값으로 대치돼 채점됩니다.")
    notes += warns
    short_verdicts, verdict_detail = verdict_block(bundle)
    status = "ok"
    if failures:
        status = "partial"
    elif stale:
        status = "stale"
    inputs_doc: dict[str, Any] = {}
    max_dates: dict[str, Any] = {}
    calendars: dict[str, Any] = {}
    for m, r in results.items():
        inputs_doc[m] = r.frames.inputs
        max_dates[m] = r.frames.max_dates
        calendars[m] = r.frames.calendar_check
    return {
        "schema_version": SCHEMA,
        "report_date": report_date,
        "historical_replay": False,
        "status": status,
        "asof": asof,
        "asof_detail": asof_detail,
        "assets": rows,
        "failures": failures,
        "verdicts": short_verdicts,
        "verdict_details": verdict_detail,
        "provenance": {
            "ms_runs": [runs["US"]["run_id"], runs["KR"]["run_id"]],
            "config_hash": cfg.config_hash(),
            "calendar_basis": {m: runs[m]["calendar"]["basis"] for m in MARKETS},
            "calendar_check": calendars,
            "bundle": {
                "sha256": bundle.sha256,
                "frozen_tag": bundle.manifest["frozen_tag"],
                "runs": {m: {
                    "run_id": runs[m]["run_id"],
                    "manifest_sha256": bundle.manifest["files"][runs[m]["run_manifest"]],
                    "oof_sha256": bundle.manifest["files"][runs[m]["oof"]],
                    "models_sha256": bundle.markets[m].model_sha(),
                    "modeler_git_commit": runs[m]["modeler_git_commit"],
                    "train": runs[m]["train"],
                    "opportunity_reference_n": bundle.markets[m].oof_rows,
                } for m in MARKETS},
            },
            "selection": {
                "sha256": selection_sha256,
                "selected_at": selection["selected_at"],
                "selection_mode": selection["selection_mode"],
                "input_cutoff": selection["input_cutoff"],
            },
            "input_snapshots": inputs_doc,
            "input_max_dates": max_dates,
            "window_check": {m: {"rows": r.window_check["rows"],
                                 "max_abs_diff": r.window_check["max_abs_diff"]}
                             for m, r in results.items()},
            "asset_registry": {"version": registry_version(), "hash": registry_hash()},
            "env": {
                "python": platform.python_version(),
                "polars": pl.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "rounding": "ret/rvol/dd/cash 6자리, 퍼센트·점수 4자리",
        },
        "notes": notes,
    }


def _xcals_version() -> str | None:
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    return getattr(xcals, "__version__", "?")


def xcals_basis() -> str:
    """``SessionCalendar.from_exchange_calendars``가 쓰는 달력 기준 문자열."""
    return f"exchange_calendars=={_xcals_version()}"


# --------------------------------------------------------------------------- score
def run_score(
    *,
    report_date: date,
    selection_path: Path,
    selection_sha256: str | None,
    bundle_path: Path,
    bundle_sha256: str | None,
    output: Path,
    cfg: MsConfig | None = None,
) -> dict[str, Any]:
    """한 날의 시장·섹터를 채점해 ``output``에 쓴다. 요약(JSON 직렬화 가능)을 돌려준다."""
    cfg = cfg or MsConfig()
    selection_raw = Path(selection_path).read_bytes()
    actual_sha = hashlib.sha256(selection_raw).hexdigest()
    if selection_sha256 is not None and actual_sha != selection_sha256:
        raise InputChangedError("ms-selection.json이 고정한 뒤에 바뀌었습니다")
    selection = json.loads(selection_raw)
    if selection.get("report_date") != report_date.isoformat():
        raise InputPinError("selection 날짜가 리포트 날짜와 다릅니다")
    pin = bundle_sha256 or selection["bundle"]["sha256"]
    bundle = load_bundle(bundle_path, expected_sha256=pin, cfg=cfg)
    if bundle.sha256 != selection["bundle"]["sha256"]:
        raise BundleError("bundle이 selection이 고정한 것과 다릅니다")
    results: dict[str, MarketResult] = {}
    failures: dict[str, dict[str, Any]] = {}
    for market in ("US", "KR"):
        limit_raw = selection["limits"].get(market)
        try:
            if not limit_raw:
                raise InputPinError("기준 상한 날짜가 없습니다")
            frames = build_frames(market, selection, bundle.markets[market], cfg)
            results[market] = score_market(
                frames, bundle.markets[market], cfg, date.fromisoformat(limit_raw))
        except Exception as exc:
            failures[market] = {"reason": failure_reason(exc), "error_class": type(exc).__name__}
            log.warning("%s 시장·섹터 실패: %s: %s", market, type(exc).__name__, exc)
    summary: dict[str, Any] = {
        "report_date": report_date.isoformat(),
        "markets": {m: ("ok" if m in results else failures[m]["reason"]) for m in ("US", "KR")},
        "failures": failures,
    }
    if not results:
        summary["status"] = "failed"
        summary["reason"] = "all_markets_failed"
        return summary
    doc = compose_document(report_date=report_date, bundle=bundle, selection=selection,
                           selection_sha256=actual_sha, results=results, failures=failures)
    summary["status"] = doc["status"]
    summary["output_sha256"] = _write_text_atomic(Path(output), _json_text(doc))
    return summary


# --------------------------------------------------------------------------- bundle 만들기
def _copy_checked(src: Path, dst: Path, expected: str | None = None) -> str:
    digest = sha256_file(src)
    if expected is not None and digest != expected:
        raise BundleError(f"원본 sha256이 manifest와 다릅니다: {src.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha256_file(dst) != digest:
        raise BundleError(f"복사 뒤 sha256이 다릅니다: {dst.name}")
    return digest


def calendar_spec(
    cal_man: dict[str, Any],
    sessions: list[date],
    *,
    file: str | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """동결 구간(첫 자산 패널의 첫~마지막 세션)의 계산 달력 지문.

    달력 파일이 있으면(``file``, ``body``) 그 항목도 적는다.
    """
    basis = cal_man["calendar_basis"]
    lake = basis.startswith("lake_trading_calendar@")
    spec = {
        "calendar_id": cal_man["calendar_id"],
        "kind": "lake_trading_calendar" if lake else "exchange_calendars",
        "basis": basis,
        "calendar_start": cal_man["first_session"],
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "n_sessions": len(sessions),
        "sessions_sha256": sessions_sha(sessions),
    }
    if lake:
        spec["basis_prefix"] = "lake_trading_calendar@"
    if file is not None and body is not None:
        spec.update(
            file=file,
            range_end=body["range_end"],
            file_n_sessions=body["n_sessions"],
            file_sessions_sha256=body["sessions_sha256"],
        )
    return spec


# --- 계산 달력 파일 ------------------------------------------------------------------
def _calendar_rows(cal: SessionCalendar) -> list[list[str]]:
    return [
        [s.isoformat(), o.astimezone(UTC).isoformat(), c.astimezone(UTC).isoformat()]
        for s, o, c in zip(cal.sessions, cal.opens, cal.closes, strict=True)
    ]


def _segment(source: str, cal: SessionCalendar, first: int, last: int) -> dict[str, Any]:
    return {"source": source, "first_session": cal.sessions[first].isoformat(),
            "last_session": cal.sessions[last - 1].isoformat(), "n_sessions": last - first}


def calendar_file_body(
    market: str,
    cal: SessionCalendar,
    *,
    range_end: date,
    method: str,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """달력 파일(``market-sector-calendar.v1``)의 내용. 세션·개장·폐장은 UTC ISO 시각이다."""
    return {
        "schema_version": CALENDAR_SCHEMA,
        "market": market,
        "calendar_id": cal.calendar_id,
        "calendar_basis": cal.calendar_basis,
        "generated_by": CALENDAR_GENERATED_BY,
        "method": method,
        "segments": segments,
        "range_start": cal.sessions[0].isoformat(),
        "range_end": range_end.isoformat(),
        "first_session": cal.sessions[0].isoformat(),
        "last_session": cal.sessions[-1].isoformat(),
        "n_sessions": len(cal.sessions),
        "sessions_sha256": sessions_sha(cal.sessions),
        "valid_through_note": (
            f"결정일 + {CALENDAR_TAIL_DAYS}일이 range_end({range_end.isoformat()})를 넘으면 채점이 "
            "calendar_range_exhausted로 거부한다. 그 뒤로 가려면 build-bundle로 bundle을 다시 "
            "만들고 release를 새로 빌드해야 한다."
        ),
        "columns": CALENDAR_COLUMNS,
        "sessions": _calendar_rows(cal),
    }


def calendar_file_text(body: dict[str, Any]) -> str:
    """머리말은 키 이름순, 세션은 한 줄에 하나. 같은 내용이면 같은 바이트다."""
    head = {k: v for k, v in body.items() if k != "sessions"}
    text = json.dumps(head, ensure_ascii=False, sort_keys=True, indent=2)
    rows = ",\n".join("    " + json.dumps(r, separators=(",", ":")) for r in body["sessions"])
    return text[:-2] + ',\n  "sessions": [\n' + rows + "\n  ]\n}\n"


def kr_calendar_file(
    cal_man: dict[str, Any], frozen_sessions: list[date]
) -> tuple[SessionCalendar, dict[str, Any]]:
    """KR: ``exchange_calendars`` XKRX를 동결 패널과 같은 버전으로 계산해 파일 내용을 만든다.

    라이브러리가 없거나 버전이 동결 패널과 다르거나, 동결 구간 세션이 동결 때와 다르면 멈춘다.
    """
    start = date.fromisoformat(cal_man["first_session"])
    cal = SessionCalendar.from_exchange_calendars("XKRX", start, CALENDAR_RANGE_END)
    if cal is None:
        raise BundleError("exchange_calendars를 불러올 수 없어 KR 계산 달력을 만들지 못합니다")
    if cal.calendar_basis != cal_man["calendar_basis"]:
        raise BundleError(
            f"exchange_calendars 버전이 동결 패널과 다릅니다: 지금 {cal.calendar_basis}, "
            f"동결 {cal_man['calendar_basis']}"
        )
    verify_calendar(cal, calendar_spec(cal_man, frozen_sessions))
    source = (f"{cal.calendar_basis} get_calendar('XKRX', start={start.isoformat()}, "
              f"end={CALENDAR_RANGE_END.isoformat()}).schedule")
    body = calendar_file_body(
        "KR", cal, range_end=CALENDAR_RANGE_END,
        method=(
            f"맥 modeler venv의 {cal.calendar_basis}가 계산한 XKRX 세션과 개장·폐장 시각(UTC)을 "
            "그대로 적었다. 동결 구간의 세션 목록이 동결 패널과 같은지 build-bundle이 확인했다. "
            "2026-06-03·07-17처럼 라이브러리에만 있는 세션도 그대로 둔다(동결 방식)."
        ),
        segments=[_segment(source, cal, 0, len(cal.sessions))],
    )
    return cal, body


def us_calendar_file(
    root: Path, cal_man: dict[str, Any], panel_man: dict[str, Any], frozen_sessions: list[date]
) -> tuple[SessionCalendar, dict[str, Any]]:
    """US: 동결 패널이 쓴 레이크 ``trading_calendar`` snapshot의 세션 목록으로 파일 내용을 만든다.

    snapshot은 2027-09-22까지라서, 그 뒤 ``CALENDAR_RANGE_END``까지는 같은 버전
    ``exchange_calendars``로 이어 붙인다. 이어 붙이기 전에 snapshot이 그 라이브러리와 겹치는
    구간에서 세션·개장·폐장이 전부 같은지 확인한다(레이크 표가 ``exchange_calendars 4.13.2``로
    만든 것이다). 다르면 멈춘다.
    """
    basis = cal_man["calendar_basis"]
    snap = basis.split("@", 1)[1]
    pin = panel_man["inputs"]["trading_calendar"]
    part = (root / "us" / "derived" / "snapshots" / "trading_calendar"
            / f"snapshot_date={snap}" / "part.parquet")
    if pin["snapshot_date"] != snap or sha256_file(part) != pin["files"]["part.parquet"]["sha256"]:
        raise BundleError("US trading_calendar snapshot이 동결 패널이 쓴 파일과 다릅니다")
    tab = (
        pl.read_parquet(part)
        .filter(pl.col("exchange") == cal_man["calendar_id"])
        .unique(subset=["date"], keep="last")
        .sort("date")
    )
    lake = SessionCalendar.from_sessions(
        cal_man["calendar_id"], tab["date"].to_list(), calendar_basis=basis,
        close_local=tab["close_local"].to_list(),
    )
    verify_calendar(lake, calendar_spec(cal_man, frozen_sessions))
    lib = SessionCalendar.from_exchange_calendars(
        cal_man["calendar_id"], lake.sessions[0], CALENDAR_RANGE_END)
    if lib is None:
        raise BundleError("exchange_calendars를 불러올 수 없어 US 계산 달력을 이어 붙이지 못합니다")
    if lib.calendar_basis != XCALS_PIN:
        raise BundleError(f"exchange_calendars 버전이 {XCALS_PIN}가 아닙니다: {lib.calendar_basis}")
    k = bisect.bisect_right(lib.sessions, lake.sessions[-1])
    if not (k == len(lake.sessions) and lib.sessions[:k] == lake.sessions
            and lib.opens[:k] == lake.opens and lib.closes[:k] == lake.closes):
        raise BundleError(
            "레이크 trading_calendar snapshot이 exchange_calendars와 겹치는 구간에서 다릅니다")
    cal = SessionCalendar(
        lake.calendar_id, basis, lake.sessions + lib.sessions[k:], lake.opens + lib.opens[k:],
        lake.closes + lib.closes[k:])
    revs = sorted(set(tab["source_rev"].to_list())) if "source_rev" in tab.columns else []
    segments = [
        _segment(f"{basis} part.parquet sha256 {pin['files']['part.parquet']['sha256']} "
                 f"(source_rev {', '.join(revs) or '-'})", cal, 0, k),
        _segment(f"{lib.calendar_basis} get_calendar('{lake.calendar_id}', "
                 f"start={lake.sessions[0].isoformat()}, end={CALENDAR_RANGE_END.isoformat()})"
                 ".schedule, 레이크 snapshot 뒤", cal, k, len(cal.sessions)),
    ]
    body = calendar_file_body(
        "US", cal, range_end=CALENDAR_RANGE_END,
        method=(
            f"동결 패널이 쓴 레이크 {basis}의 세션과 폐장 시각(UTC)을 그대로 적고, snapshot이 끝난 "
            f"뒤부터 {CALENDAR_RANGE_END.isoformat()}까지는 {lib.calendar_basis}로 이어 붙였다. "
            "이어 붙이기 전에 snapshot이 그 라이브러리와 겹치는 구간에서 세션·개장·폐장이 같은지 "
            "build-bundle이 확인했다. 동결 구간의 세션 목록이 동결 패널과 같은지도 확인했다."
        ),
        segments=segments,
    )
    return cal, body


def bundle_manifest(
    files: dict[str, str], markets: dict[str, Any], cfg: MsConfig
) -> dict[str, Any]:
    verdicts: dict[str, Any] = {"source": VERDICT_SOURCE, "adopted_model": None,
                                "baseline_kept": True, "validation_status": "research"}
    for market, block in VERDICTS.items():
        verdicts[market] = {k: {"label": lab, "detail": det} for k, (lab, det) in block.items()}
    return {
        "schema_version": BUNDLE_SCHEMA,
        "calendar_range_note": (
            f"계산 달력 파일은 {CALENDAR_RANGE_END.isoformat()}까지다. "
            f"결정일 + {CALENDAR_TAIL_DAYS}일이 그 뒤이면 채점이 calendar_range_exhausted로 "
            "거부한다. 그 전에 build-bundle로 bundle을 다시 만들고 release를 새로 빌드해야 한다."
        ),
        "frozen_tag": FROZEN_TAG,
        "config_hash": cfg.config_hash(),
        "asset_registry": {"version": registry_version(), "hash": registry_hash()},
        "verdicts": verdicts,
        "files": dict(sorted(files.items())),
        "markets": markets,
    }


def build_bundle(
    stock_data_root: Path, output: Path, cfg: MsConfig | None = None
) -> dict[str, Any]:
    """동결 run을 bundle 디렉터리로 복사한다.

    ``stock_data``는 읽기만 하고 ``output``은 새로 만든다.
    """
    cfg = cfg or MsConfig()
    root, output = Path(stock_data_root), Path(output)
    if output.exists():
        raise FileExistsError(f"{output} 가 이미 있습니다")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
    try:
        files: dict[str, str] = {}
        markets: dict[str, Any] = {}
        calendars: dict[str, Any] = {}
        for market in MARKETS:
            m = market.lower()
            run = root / m / "output" / "market_sector" / FROZEN_RUN_ID[market]
            man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if man["config_hash"] != cfg.config_hash():
                raise BundleError(f"{market}: 동결 run 설정 해시가 현재 MsConfig와 다릅니다")
            if man.get("git_dirty"):
                raise BundleError(f"{market}: 동결 run이 dirty 트리에서 만들어졌습니다")
            dataset = root / m / "datasets" / "market_sector"
            panel_dir = dataset / FROZEN_PANEL[market]
            panel_man = json.loads((panel_dir / "manifest.json").read_text(encoding="utf-8"))
            feat_man = json.loads(
                (dataset / FROZEN_FEATURES[market] / "manifest.json").read_text(encoding="utf-8"))
            rel = {"run_manifest": f"{m}/manifest.json", "oof": f"{m}/oof_predictions.parquet",
                   "latest_scores": f"{m}/latest_scores.json"}
            files[rel["run_manifest"]] = _copy_checked(
                run / "manifest.json", stage / rel["run_manifest"])
            files[rel["oof"]] = _copy_checked(
                run / "oof_predictions.parquet", stage / rel["oof"],
                man["outputs"]["oof_predictions.parquet"])
            files[rel["latest_scores"]] = _copy_checked(
                run / "latest_scores.json", stage / rel["latest_scores"])
            model_rel = {}
            for name in MODEL_NAMES:
                model_rel[name] = f"{m}/models/live/{name}.joblib"
                files[model_rel[name]] = _copy_checked(
                    run / "models" / "live" / f"{name}.joblib", stage / model_rel[name])
            latest = json.loads((run / "latest_scores.json").read_text(encoding="utf-8"))
            if not latest:
                raise BundleError(f"{market}: latest_scores.json이 비었습니다")
            panel = pl.read_parquet(panel_dir / "panel.parquet")
            first_asset = sorted(panel["asset_id"].unique().to_list())[0]
            sessions = sorted(panel.filter(pl.col("asset_id") == first_asset)["session"].to_list())
            frozen_pins: dict[str, Any] = {
                "inputs": {t: {"snapshot_date": v["snapshot_date"],
                               "files": {n: f["sha256"] for n, f in v["files"].items()}}
                           for t, v in panel_man["inputs"].items()}}
            if market == "KR":
                frozen_pins["us_macro_series"] = {
                    "snapshot_date": feat_man["macro_series_snapshot_date"],
                    "files": feat_man["macro_series_files"]}
            cal_rel = f"{m}/calendar.json"
            if market == "KR":
                _, cal_body = kr_calendar_file(panel_man["calendar"], sessions)
            else:
                _, cal_body = us_calendar_file(root, panel_man["calendar"], panel_man, sessions)
            cal_text = calendar_file_text(cal_body)
            files[cal_rel] = _write_text_atomic(stage / cal_rel, cal_text)
            calendars[market] = {k: cal_body[k] for k in (
                "calendar_basis", "range_start", "range_end", "first_session", "last_session",
                "n_sessions", "sessions_sha256")} | {"file_sha256": files[cal_rel]}
            any_latest = latest[0]
            markets[market] = {
                "run_id": FROZEN_RUN_ID[market],
                **rel,
                "models": model_rel,
                "modeler_git_commit": man["modeler_git_commit"],
                "panel": {"version": FROZEN_PANEL[market],
                          "manifest_sha256": sha256_file(panel_dir / "manifest.json"),
                          "features": FROZEN_FEATURES[market]},
                "calendar": calendar_spec(panel_man["calendar"], sessions, file=cal_rel,
                                          body=cal_body),
                "frozen_input_pins": frozen_pins,
                "frozen_latest_session": any_latest["session"],
                "assets": sorted(r["asset_id"] for r in latest),
                "train": {
                    "boundary_label_end_before":
                        any_latest["model_train_boundary_label_end_before"],
                    "n_train": any_latest["n_train"]},
            }
        (stage / "bundle.json").write_text(
            _json_text(bundle_manifest(files, markets, cfg)), encoding="utf-8")
        verify_bundle_dir(stage)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"bundle": str(output), "bundle_sha256": sha256_file(output / "bundle.json"),
            "files": len(files), "calendars": calendars}


# --------------------------------------------------------------------------- 동결 재현
def _frozen_selection(root: Path, bundle: Bundle, market: str) -> dict[str, Any]:
    """동결 패널이 쓴 snapshot을 selection 모양으로 만든다.

    맥 ``stock_data``에 같은 바이트가 있어야 한다. cutoff 규칙을 피하려고 리포트 날짜를 먼
    날(2026-12-31)로 둔다. 동결 입력은 그 전에 끝난 파일이다.
    """
    pins = bundle.manifest["markets"][market]["frozen_input_pins"]
    sel: dict[str, Any] = {
        "schema_version": "market-sector-selection.v1",
        "report_date": "2026-12-31",
        "selected_at": "2026-12-31T09:31:00+09:00",
        "selection_mode": "scheduled",
        "input_cutoff": "2026-12-31T09:30:00+09:00",
        "bundle": {"path": str(bundle.directory / "bundle.json"), "sha256": bundle.sha256},
        "roots": {"kr": str(root / "kr"), "us": str(root / "us")},
        "limits": {"KR": None, "US": None},
        "kr": {"status": "unavailable", "reason": "frozen_replay"},
        "us": {"status": "selected", "tables": {}},
    }
    expected: dict[str, dict[str, str]] = {}
    if market == "US":
        for t in US_TABLES:
            snap = pins["inputs"][t]["snapshot_date"]
            sel["us"]["tables"][t] = {
                "status": "selected", **describe_us_table(root / "us", t, snap)}
            expected[f"US {t}"] = pins["inputs"][t]["files"]
    else:
        snap = pins["inputs"]["krx_index_daily"]["snapshot_date"]
        sel["kr"] = {"status": "selected", **describe_kr_snapshot(root / "kr", snap)}
        for t in sel["kr"]["tables"]:
            expected[f"KR {t}"] = pins["inputs"][t]["files"]
        macro = pins["us_macro_series"]
        macro_snap = macro["snapshot_date"]
        sel["us"]["tables"]["macro_series"] = {
            "status": "selected", **describe_us_table(root / "us", "macro_series", macro_snap)}
        expected["US macro_series"] = macro["files"]
    for key, files in expected.items():
        rec = (sel["kr"]["tables"][key[3:]] if key.startswith("KR ")
               else sel["us"]["tables"][key[3:]])
        # 동결 패널 manifest는 파일 이름만 키로 써서, 파티션 디렉터리가 있는 KR 표는
        # 마지막 파일 하나만 남았다.
        live = {Path(n).name: f["sha256"] for n, f in sorted(rec["files"].items())}
        if live != files:
            raise InputChangedError(f"{key}: 동결 패널이 쓴 파일과 다릅니다")
    return sel


def verify_frozen(
    stock_data_root: Path, bundle_path: Path, tolerance: float = 1e-9
) -> dict[str, Any]:
    """동결 run과 같은 입력 snapshot으로 채점해 ``latest_scores.json``과 점수를 비교한다.

    점수는 반올림 전 원값끼리 비교한다. 창 피쳐는 전체 패널 피쳐와 창의 마지막
    ``WINDOW_MARGIN_SESSIONS``행에서 비교한다.
    """
    cfg = MsConfig()
    root = Path(stock_data_root)
    bundle = load_bundle(bundle_path, cfg=cfg)
    report: dict[str, Any] = {}
    for market in MARKETS:
        mb = bundle.markets[market]
        sel = _frozen_selection(root, bundle, market)
        limit = date.fromisoformat(mb.spec["frozen_latest_session"])
        sel["limits"][market] = limit.isoformat()
        frames = build_frames(market, sel, mb, cfg)
        res = score_market(frames, mb, cfg, limit)
        wdiff = window_vs_full_diff(
            window_features(frames, limit, cfg), full_price_frame(frames, cfg),
            last_n=WINDOW_MARGIN_SESSIONS)
        # b_opp_mean은 동결 run의 OOF에 같은 (자산, 세션) 행이 있으면 그 값과도 비교한다
        oof = pl.read_parquet(mb.directory / mb.spec["oof"]).filter(
            pl.col("market") == market, pl.col("session") == limit)
        oof_b = {r["asset_id"]: r["b_opp_mean"] for r in oof.iter_rows(named=True)}
        b_worst = max((abs(res.raw[a]["b_opp_mean"] - v) for a, v in oof_b.items()
                       if v is not None and res.raw[a]["b_opp_mean"] is not None), default=None)
        worst = 0.0
        compare = []
        for aid in sorted(res.raw):
            lr, raw = mb.latest[aid], res.raw[aid]
            diffs = {
                "opportunity_score": abs(raw["opportunity_score"] - lr["opp_ridge_score"]),
                "stability_score": abs(raw["stability_score"] - lr["stab_logit_score"]),
                "raw_opp": abs(raw["p_opp_ridge"] - lr["opp_ridge_raw"]),
                "raw_stab": abs(raw["p_stab_logit"] - lr["stab_logit_p_hat"]),
            }
            worst = max(worst, *diffs.values())
            compare.append({"asset_id": aid, "session": lr["session"], **diffs})
        report[market] = {
            "limit": limit.isoformat(),
            "rows": compare,
            "max_abs_diff": worst,
            "tolerance": tolerance,
            "score_pass": worst <= tolerance and (b_worst is None or b_worst <= tolerance),
            "b_opp_mean_vs_oof": {"rows": len(oof_b), "max_abs_diff": b_worst},
            "window_vs_full_last_n": {"n": WINDOW_MARGIN_SESSIONS, **wdiff},
            "window_pass": wdiff["max_abs_diff"] <= WINDOW_TOLERANCE
            and wdiff["null_mismatch"] == 0 and wdiff["feature_ready_mismatch"] == 0,
            "calendar": frames.calendar_check,
            "lag_sessions": res.meta["lag_sessions"],
        }
    report["overall_pass"] = all(
        report[m]["score_pass"] and report[m]["window_pass"] for m in MARKETS)
    return report


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler.scores.market_sector.score_daily",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score")
    p.add_argument("--report-date", type=date.fromisoformat, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--selection-sha256")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--bundle-sha256")
    p.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("build-bundle")
    b.add_argument("--stock-data-root", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify-frozen")
    v.add_argument("--stock-data-root", type=Path, required=True)
    v.add_argument("--bundle", type=Path, required=True)
    v.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "score":
            summary = run_score(
                report_date=args.report_date, selection_path=args.selection,
                selection_sha256=args.selection_sha256, bundle_path=args.bundle,
                bundle_sha256=args.bundle_sha256, output=args.output)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=_jdefault))
            return 0 if summary["status"] != "failed" else 1
        if args.command == "build-bundle":
            print(json.dumps(build_bundle(args.stock_data_root, args.output), sort_keys=True))
            return 0
        report = verify_frozen(args.stock_data_root, args.bundle, args.tolerance)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=_jdefault))
        return 0 if report["overall_pass"] else 1
    except Exception as exc:  # the coordinator reads one JSON line whatever went wrong
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__,
                          "reason": failure_reason(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
