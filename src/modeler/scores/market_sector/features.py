"""시장·섹터 피쳐 (레이어 2 MS1): 10~20개의 해석 가능한 피쳐.

모든 피쳐는 ``available_at <= decision_at``인 입력만 쓴다 (사양 01 §5).

* 가격 피쳐: 레이어 1 패널의 총수익 경로 ``tr_index_t``의 과거 값. 입력 가용 시각은
  패널의 ``price_available_at``(세션 폐장 + 60분)이다.
* 거시 피쳐(US 일별 FRED/ALFRED): 행마다 ``available_at = 17:00 America/New_York``
  on ``max(관측일, realtime_start)``. ALFRED ``realtime_start``가 관측일보다 늦으면
  그 날짜 이후에만 그 값이 존재한 것으로 본다. tz-aware라 KR 08:30 KST 결정에도 그대로 쓴다.
  각 결정 시각마다 ``available_at <= decision_at``인 행 중 **관측일이 가장 늦은** 값
  (같은 관측일이면 최신 vintage)을 쓴다. 나이가 ``macro_staleness_days``를 넘으면 null.
* KR 입력은 아직 맥에 없다. KR 실행은 ``KrNotSyncedError``로 멈춘다(0으로 채우지 않는다).
  KR 피쳐 함수는 일반 long 프레임(``date, value, available_at``)을 받아 sync 뒤 바로 쓴다.

**경고.** ``macro_series``에서 일부 계열은 초기 ``realtime_start``가 백필 시점이다
(BAA10Y 2014-01-27, WTI 2011-04-06, VIX 2010-11-22 ...). strict PIT 규칙대로 그 전 결정일에는
null이다. 모델 입력에서는 학습 구간 중앙값으로 채운다(``models.py``).
"""

from __future__ import annotations

import json
import math
import platform
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from modeler.scores.common.assets import Asset, get_asset
from modeler.scores.common.calendar import UTC_TS
from modeler.scores.common.panel import assert_pit
from modeler.scores.market_sector.build_panel import KrNotSyncedError
from modeler.scores.market_sector.config import MsConfig

__all__ = [
    "KrNotSyncedError",
    "FEATURE_SPECS",
    "PitSeries",
    "asset_onehot_columns",
    "build_features",
    "compute_price_features",
    "fred_available_at",
    "kr_foreign_net_20_over_trdval",
    "load_us_macro",
    "macro_features",
    "model_feature_columns",
    "usdkrw_ret_60",
]

TRADING_DAYS = 252
MACRO_SERIES_IDS = ("VIXCLS", "DGS10", "DGS2", "DCOILWTICO", "BAA10Y")
FRED_AVAILABLE_BASIS = "17:00 America/New_York on max(observation_date, realtime_start)"
PRICE_BASIS = "session_close_plus_60min (panel price_available_at)"

PRICE_FEATURES = (
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_252",
    "trend_ma200",
    "dd_252",
    "rvol_20",
    "rvol_60",
    "rvol_ratio_20_252",
    "rel_strength_60",
)
MACRO_FEATURES = (
    "vix_log",
    "vix_chg_20",
    "us_term_spread",
    "us_rate10_chg_20",
    "wti_ret_60",
    "credit_baa10y_chg_20",
)
KR_FEATURES = ("usdkrw_ret_60", "kr_foreign_net_20_over_trdval")
INDICATORS = ("has_parent", "market_is_kr")
#: 연속형 피쳐 18개 (이름 고정).
CONTINUOUS_FEATURES = PRICE_FEATURES + MACRO_FEATURES + KR_FEATURES


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    inputs: str
    available_at_basis: str
    definition: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    *(
        FeatureSpec(
            f"ret_{k}",
            "price",
            "panel.tr_index_t",
            PRICE_BASIS,
            f"log(TR_t / TR_(t-{k} sessions))",
        )
        for k in (20, 60, 120, 252)
    ),
    FeatureSpec("trend_ma200", "price", "panel.tr_index_t", PRICE_BASIS, "TR_t / MA200(TR) - 1"),
    FeatureSpec(
        "dd_252", "price", "panel.tr_index_t", PRICE_BASIS, "TR_t / max(TR, 252 sessions) - 1"
    ),
    FeatureSpec(
        "rvol_20", "price", "panel.tr_index_t", PRICE_BASIS, "std(daily log TR, 20) * sqrt(252)"
    ),
    FeatureSpec(
        "rvol_60", "price", "panel.tr_index_t", PRICE_BASIS, "std(daily log TR, 60) * sqrt(252)"
    ),
    FeatureSpec(
        "rvol_ratio_20_252",
        "price",
        "panel.tr_index_t",
        PRICE_BASIS,
        "rvol_20 / (std(daily log TR, 252) * sqrt(252))",
    ),
    FeatureSpec(
        "rel_strength_60",
        "price",
        "panel.tr_index_t of asset and its parent_benchmark",
        PRICE_BASIS,
        "ret_60(asset) - ret_60(parent); 0 when no parent (has_parent=0)",
    ),
    FeatureSpec(
        "vix_log", "macro", "macro_series VIXCLS", FRED_AVAILABLE_BASIS, "log(VIXCLS latest known)"
    ),
    FeatureSpec(
        "vix_chg_20",
        "macro",
        "macro_series VIXCLS",
        FRED_AVAILABLE_BASIS,
        "log(VIX_latest / VIX_20 observations earlier)",
    ),
    FeatureSpec(
        "us_term_spread",
        "macro",
        "macro_series DGS10, DGS2",
        FRED_AVAILABLE_BASIS,
        "DGS10 - DGS2 (percentage points, each latest known)",
    ),
    FeatureSpec(
        "us_rate10_chg_20",
        "macro",
        "macro_series DGS10",
        FRED_AVAILABLE_BASIS,
        "DGS10_latest - DGS10_20 observations earlier (pp)",
    ),
    FeatureSpec(
        "wti_ret_60",
        "macro",
        "macro_series DCOILWTICO",
        FRED_AVAILABLE_BASIS,
        "(P - P_60obs) / max(|P_60obs|, 1)  (WTI went negative in 2020-04, so no log)",
    ),
    FeatureSpec(
        "credit_baa10y_chg_20",
        "macro",
        "macro_series BAA10Y",
        FRED_AVAILABLE_BASIS,
        "BAA10Y_latest - BAA10Y_20 observations earlier (pp)",
    ),
    FeatureSpec(
        "usdkrw_ret_60",
        "market_specific_kr",
        "KR USD/KRW (not synced)",
        "per-row available_at supplied by the KR loader",
        "log(FX_latest / FX_60obs); US market: 0",
    ),
    FeatureSpec(
        "kr_foreign_net_20_over_trdval",
        "market_specific_kr",
        "KR foreign net purchase value, KR trading value (not synced)",
        "per-row available_at supplied by the KR loader",
        "sum(foreign_net, 20 obs) / sum(trdval, 20 obs); US market: 0",
    ),
    FeatureSpec("has_parent", "indicator", "asset registry", "static", "1 if parent_benchmark"),
    FeatureSpec("market_is_kr", "indicator", "asset registry", "static", "1 if KR asset"),
)


# --------------------------------------------------------------------------- PIT series
def fred_available_at(rows: pl.DataFrame) -> pl.DataFrame:
    """``rows``(``date, realtime_start``)에 ``available_at``(UTC)을 붙인다."""
    eff = pl.when(pl.col("realtime_start") > pl.col("date")).then(pl.col("realtime_start"))
    eff = eff.otherwise(pl.col("date"))
    return rows.with_columns(
        (eff.cast(pl.Datetime("us")) + pl.duration(hours=17))
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
        .cast(UTC_TS)
        .alias("available_at")
    )


class PitSeries:
    """관측일 ``date``·값 ``value``·가용 시각 ``available_at``(tz-aware) 행의 시점 조회기.

    ``lookup(decision_ats, lags)``: 각 결정 시각 ``D``에서 ``available_at <= D``인 행 중 관측일이
    가장 늦은 값(frontier)과, frontier에서 ``k``개 관측일 앞선 날짜의 그 시점 최신 vintage 값.
    """

    def __init__(self, rows: pl.DataFrame):
        r = rows.drop_nulls(["date", "available_at", "value"]).sort(["available_at", "date"])
        self._avail: list[datetime] = r["available_at"].to_list()
        self._date: list[date] = r["date"].to_list()
        self._val: list[float] = r["value"].to_list()
        self._all_dates = sorted(set(self._date))
        self._pos = {d: i for i, d in enumerate(self._all_dates)}

    def lookup(
        self, decision_ats: Sequence[datetime], lags: Sequence[int] = (0,)
    ) -> dict[int, dict[str, list]]:
        order = sorted(range(len(decision_ats)), key=lambda i: decision_ats[i])
        n = len(decision_ats)
        out = {k: {"value": [None] * n, "obs_date": [None] * n, "avail": [None] * n} for k in lags}
        latest: dict[date, tuple[datetime, float]] = {}
        frontier: date | None = None
        i, m = 0, len(self._avail)
        for j in order:
            d_at = decision_ats[j]
            while i < m and self._avail[i] <= d_at:
                d = self._date[i]
                latest[d] = (self._avail[i], self._val[i])
                if frontier is None or d > frontier:
                    frontier = d
                i += 1
            if frontier is None:
                continue
            for k in lags:
                p = self._pos[frontier] - k
                if p < 0:
                    continue
                d2 = self._all_dates[p]
                hit = latest.get(d2)
                if hit is None:
                    continue
                out[k]["value"][j] = hit[1]
                out[k]["obs_date"][j] = d2
                out[k]["avail"][j] = hit[0]
        return out


def _age_days(decision_at: datetime, obs: date | None) -> int | None:
    if obs is None:
        return None
    return (decision_at.astimezone(UTC).date() - obs).days


def _maxdt(*xs: datetime | None) -> datetime | None:
    v = [x for x in xs if x is not None]
    return max(v) if v else None


def _feature_from_series(
    series: PitSeries,
    decision_ats: Sequence[datetime],
    lag: int,
    fn,
    staleness_days: int | None,
) -> tuple[list[float | None], list[datetime | None], list[int | None]]:
    """``fn(v0, vlag) -> float|None``. 반환: 값, 사용한 입력의 최대 available_at, 나이(일)."""
    res = series.lookup(decision_ats, (0, lag) if lag else (0,))
    vals, avails, ages = [], [], []
    for j, d_at in enumerate(decision_ats):
        v0 = res[0]["value"][j]
        age = _age_days(d_at, res[0]["obs_date"][j])
        vk = res[lag]["value"][j] if lag else None
        ok = v0 is not None and (lag == 0 or vk is not None)
        if ok and staleness_days is not None and age is not None and age > staleness_days:
            ok = False
        val = fn(v0, vk) if ok else None
        if val is not None and not math.isfinite(val):
            val = None
        vals.append(val)
        avails.append(
            _maxdt(res[0]["avail"][j], res[lag]["avail"][j] if lag else None)
            if val is not None
            else None
        )
        ages.append(age if val is not None else None)
    return vals, avails, ages


def load_us_macro(lake, series_ids: Sequence[str] = MACRO_SERIES_IDS) -> pl.DataFrame:
    """레이크 ``macro_series``에서 ``series_id, date, realtime_start, value, available_at``."""
    rows = (
        lake.scan("macro_series")
        .filter(pl.col("series_id").is_in(list(series_ids)))
        .select("series_id", "date", "realtime_start", "value")
        .collect()
    )
    return fred_available_at(rows)


def macro_features(
    macro_rows: pl.DataFrame, decision_ats: Sequence[datetime], cfg: MsConfig
) -> pl.DataFrame:
    """결정 시각(고유·tz-aware)마다 거시 피쳐 6개. 입력 계열이 없으면 ``KeyError``."""
    ats = list(decision_ats)
    stale = cfg.macro_staleness_days
    ps = {}
    for sid in MACRO_SERIES_IDS:
        sub = macro_rows.filter(pl.col("series_id") == sid)
        if sub.height == 0:
            raise KeyError(f"macro_series에 {sid} 가 없습니다")
        ps[sid] = PitSeries(sub)

    def lg(a: float, b: float) -> float | None:
        return math.log(a / b) if a > 0 and b > 0 else None

    cols: dict[str, tuple[list, list, list]] = {}
    cols["vix_log"] = _feature_from_series(
        ps["VIXCLS"], ats, 0, lambda v, _: math.log(v) if v > 0 else None, stale
    )
    cols["vix_chg_20"] = _feature_from_series(ps["VIXCLS"], ats, 20, lg, stale)
    # 스프레드: 두 계열 각각의 최신 값 (나이 검사도 각각)
    v10 = _feature_from_series(ps["DGS10"], ats, 0, lambda v, _: v, stale)
    v2 = _feature_from_series(ps["DGS2"], ats, 0, lambda v, _: v, stale)
    sp = [None if a is None or b is None else a - b for a, b in zip(v10[0], v2[0], strict=True)]
    sp_av = [
        _maxdt(a, b) if s is not None else None for a, b, s in zip(v10[1], v2[1], sp, strict=True)
    ]
    sp_age = [
        max(a, b) if s is not None else None for a, b, s in zip(v10[2], v2[2], sp, strict=True)
    ]
    cols["us_term_spread"] = (sp, sp_av, sp_age)
    cols["us_rate10_chg_20"] = _feature_from_series(ps["DGS10"], ats, 20, lambda a, b: a - b, stale)
    cols["wti_ret_60"] = _feature_from_series(
        ps["DCOILWTICO"], ats, 60, lambda a, b: (a - b) / max(abs(b), 1.0), stale
    )
    cols["credit_baa10y_chg_20"] = _feature_from_series(
        ps["BAA10Y"], ats, 20, lambda a, b: a - b, stale
    )
    data: dict[str, Any] = {"decision_at": pl.Series(ats, dtype=UTC_TS)}
    for name in MACRO_FEATURES:
        data[name] = pl.Series(cols[name][0], dtype=pl.Float64)
    avail_cols = [cols[n][1] for n in MACRO_FEATURES]
    age_cols = [cols[n][2] for n in MACRO_FEATURES]
    data["macro_available_at_max"] = pl.Series(
        [_maxdt(*[c[j] for c in avail_cols]) for j in range(len(ats))], dtype=UTC_TS
    )
    data["macro_age_days_max"] = pl.Series(
        [max([c[j] for c in age_cols if c[j] is not None], default=None) for j in range(len(ats))],
        dtype=pl.Int64,
    )
    return pl.DataFrame(data)


# --------------------------------------------------------------------------- KR (generic)
def usdkrw_ret_60(
    fx_rows: pl.DataFrame, decision_ats: Sequence[datetime], staleness_days: int | None = 10
) -> pl.DataFrame:
    """``fx_rows``: ``date, value, available_at``(tz-aware). ``log(FX / FX_60 관측 전)``."""
    ps = PitSeries(fx_rows)
    vals, avails, _ = _feature_from_series(
        ps,
        list(decision_ats),
        60,
        lambda a, b: math.log(a / b) if a > 0 and b > 0 else None,
        staleness_days,
    )
    return pl.DataFrame(
        {
            "decision_at": pl.Series(list(decision_ats), dtype=UTC_TS),
            "usdkrw_ret_60": pl.Series(vals, dtype=pl.Float64),
            "usdkrw_available_at": pl.Series(avails, dtype=UTC_TS),
        }
    )


def kr_foreign_net_20_over_trdval(
    flow_rows: pl.DataFrame,
    trdval_rows: pl.DataFrame,
    decision_ats: Sequence[datetime],
    window: int = 20,
) -> pl.DataFrame:
    """외국인 순매수(원)·거래대금(원) 각 ``date, value, available_at`` long 프레임.

    결정 시각 ``D``에서 ``available_at <= D``인 관측 중 날짜가 가장 늦은 ``window``개의 합 비율.
    두 계열의 관측일 집합이 달라 창이 안 맞으면 null(채우지 않는다). ``available_at``이
    날짜순으로 단조가 아니면 예외.
    """

    def prep(rows: pl.DataFrame) -> tuple[list[date], list[float], list[datetime]]:
        r = rows.drop_nulls(["date", "value", "available_at"]).sort("date")
        av = r["available_at"].to_list()
        if any(b < a for a, b in zip(av, av[1:], strict=False)):
            raise ValueError("available_at이 날짜순으로 단조가 아닙니다")
        return r["date"].to_list(), r["value"].to_list(), av

    fd, fv, fa = prep(flow_rows)
    td, tv, ta = prep(trdval_rows)
    vals, avails = [], []
    for d_at in decision_ats:
        kf = bisect_right(fa, d_at)
        kt = bisect_right(ta, d_at)
        if kf < window or kt < window or fd[kf - window : kf] != td[kt - window : kt]:
            vals.append(None)
            avails.append(None)
            continue
        den = sum(tv[kt - window : kt])
        if den == 0:
            vals.append(None)
            avails.append(None)
            continue
        vals.append(sum(fv[kf - window : kf]) / den)
        avails.append(max(fa[kf - 1], ta[kt - 1]))
    return pl.DataFrame(
        {
            "decision_at": pl.Series(list(decision_ats), dtype=UTC_TS),
            "kr_foreign_net_20_over_trdval": pl.Series(vals, dtype=pl.Float64),
            "kr_flow_available_at": pl.Series(avails, dtype=UTC_TS),
        }
    )


# --------------------------------------------------------------------------- price features
def compute_price_features(panel: pl.DataFrame, cfg: MsConfig) -> pl.DataFrame:
    """패널(``asset_id, session, decision_at, tr_index_t, price_available_at, ...``)에서 가격 피쳐.

    과거 행만 쓰는 shift·rolling(뒤쪽 창)만 쓴다. 창이 다 안 찬 앞부분은 null이다.
    ``asset_row``: 자산 패널 안 행 번호(0부터).
    """
    p = panel.sort(["asset_id", "session"]).with_columns(
        pl.col("tr_index_t").log().alias("_lt"),
        pl.int_range(pl.len()).over("asset_id").alias("asset_row"),
    )
    p = p.with_columns((pl.col("_lt") - pl.col("_lt").shift(1).over("asset_id")).alias("_r1"))
    exprs = [
        (pl.col("_lt") - pl.col("_lt").shift(k).over("asset_id")).alias(f"ret_{k}")
        for k in (20, 60, 120, 252)
    ]
    ann = math.sqrt(TRADING_DAYS)
    exprs += [
        (pl.col("tr_index_t") / pl.col("tr_index_t").rolling_mean(200).over("asset_id") - 1).alias(
            "trend_ma200"
        ),
        (pl.col("tr_index_t") / pl.col("tr_index_t").rolling_max(252).over("asset_id") - 1).alias(
            "dd_252"
        ),
        (pl.col("_r1").rolling_std(20).over("asset_id") * ann).alias("rvol_20"),
        (pl.col("_r1").rolling_std(60).over("asset_id") * ann).alias("rvol_60"),
        (pl.col("_r1").rolling_std(252).over("asset_id") * ann).alias("_rvol_252"),
    ]
    p = p.with_columns(exprs).with_columns(
        (pl.col("rvol_20") / pl.col("_rvol_252")).alias("rvol_ratio_20_252")
    )
    parent_of = {a: get_asset(a).parent_benchmark for a in p["asset_id"].unique().to_list()}
    pm = pl.DataFrame(
        {
            "asset_id": list(parent_of),
            "_parent": list(parent_of.values()),
        },
        schema={"asset_id": pl.String, "_parent": pl.String},
    )
    p = p.join(pm, on="asset_id", how="left")
    parent_ret = p.select(
        pl.col("asset_id").alias("_parent"), "session", pl.col("ret_60").alias("_parent_ret_60")
    )
    p = p.join(parent_ret, on=["_parent", "session"], how="left").with_columns(
        pl.col("_parent").is_not_null().cast(pl.Int8).alias("has_parent"),
    )
    p = p.with_columns(
        pl.when(pl.col("_parent").is_null())
        .then(0.0)
        .otherwise(pl.col("ret_60") - pl.col("_parent_ret_60"))
        .alias("rel_strength_60")
    )
    # 미래를 안 본다는 보장: 룩백보다 짧은 행은 feature_ready=False
    ready = (
        (pl.col("asset_row") >= cfg.max_lookback_sessions)
        & pl.col("ret_252").is_not_null()
        & pl.col("rvol_ratio_20_252").is_not_null()
        & pl.col("rel_strength_60").is_not_null()
    )
    keep = [
        "asset_id",
        "session",
        "decision_at",
        "entry_at",
        "price_available_at",
        "asset_row",
        *PRICE_FEATURES,
        "has_parent",
    ]
    return p.with_columns(ready.alias("feature_ready")).select(*keep, "feature_ready")


def asset_onehot_columns(asset_ids: Sequence[str]) -> list[str]:
    return [f"asset_{a}" for a in sorted(asset_ids)]


def model_feature_columns(asset_ids: Sequence[str]) -> list[str]:
    """모델 입력 열 순서: 연속형 18 + has_parent + market_is_kr + 자산 원-핫."""
    return [*CONTINUOUS_FEATURES, *INDICATORS, *asset_onehot_columns(asset_ids)]


# --------------------------------------------------------------------------- assemble
def build_features(
    panel: pl.DataFrame,
    macro_rows: pl.DataFrame | None,
    cfg: MsConfig,
    *,
    market: str,
) -> pl.DataFrame:
    """패널 -> 피쳐 프레임 (asset_id, session 한 행). US만 지원, KR은 ``KrNotSyncedError``."""
    if market.upper() == "KR":
        raise KrNotSyncedError(
            "KR 입력(USD/KRW·외국인 순매수·거래대금·KR 지수 패널)이 아직 맥에 sync되지 않았습니다. "
            "collector 백필 후 `collector db sync-remote`로 받은 뒤 KR 로더를 붙이십시오. "
            "0으로 채우지 않고 여기서 멈춥니다."
        )
    if macro_rows is None:
        raise ValueError("US 피쳐에는 macro_rows가 필요합니다")
    pf = compute_price_features(panel, cfg)
    ats = sorted(pf["decision_at"].unique().to_list())
    mf = macro_features(macro_rows, ats, cfg)
    f = pf.join(mf, on="decision_at", how="left")
    asset_ids = sorted(f["asset_id"].unique().to_list())
    f = f.with_columns(
        pl.lit(0.0).alias("usdkrw_ret_60"),
        pl.lit(0.0).alias("kr_foreign_net_20_over_trdval"),
        pl.lit(0, dtype=pl.Int8).alias("market_is_kr"),
        pl.max_horizontal("price_available_at", "macro_available_at_max").alias(
            "feature_available_at"
        ),
        *[(pl.col("asset_id") == a).cast(pl.Int8).alias(f"asset_{a}") for a in asset_ids],
    )
    f = f.sort(["asset_id", "session"])
    assert_pit(f, ["feature_available_at"])
    # 연속형 피쳐의 기대 열이 다 있는지
    missing = [c for c in model_feature_columns(asset_ids) if c not in f.columns]
    if missing:  # pragma: no cover
        raise AssertionError(f"피쳐 열 누락: {missing}")
    return f


def null_rates(f: pl.DataFrame, asset_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    """feature_ready 행 기준 피쳐별 null 비율 + 전체 행 기준(warm-up 포함)."""
    cols = [*CONTINUOUS_FEATURES, "has_parent", "market_is_kr"]
    ready = f.filter(pl.col("feature_ready"))
    return {
        "ready_rows": {c: float(ready[c].null_count() / max(ready.height, 1)) for c in cols},
        "all_rows": {c: float(f[c].null_count() / max(f.height, 1)) for c in cols},
    }


def feature_manifest(
    f: pl.DataFrame, cfg: MsConfig, *, asset_ids: Sequence[str], extra: dict[str, Any]
) -> dict[str, Any]:
    return {
        "layer": "market_sector_layer2_features",
        "features": [
            {
                "name": s.name,
                "group": s.group,
                "inputs": s.inputs,
                "available_at_basis": s.available_at_basis,
                "definition": s.definition,
            }
            for s in FEATURE_SPECS
        ],
        "asset_onehot_columns": asset_onehot_columns(asset_ids),
        "feature_config": cfg.feature_config(),
        "feature_config_hash": cfg.feature_config_hash(),
        "rows": f.height,
        "feature_ready_rows": int(f["feature_ready"].sum()),
        "null_rates": null_rates(f, asset_ids),
        "env": {"polars": pl.__version__, "python": platform.python_version()},
        **extra,
    }


def write_features(f: pl.DataFrame, out_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    from modeler.scores.common.inputs import sha256_file
    from modeler.us.dataset import content_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "features.parquet"
    f.write_parquet(p, compression="zstd", statistics=True)
    manifest = {
        **manifest,
        "outputs": {
            "features.parquet": sha256_file(p),
            "features_content_hash": content_hash(f),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )
    return manifest["outputs"]


def registry_assets(asset_ids: Sequence[str]) -> list[Asset]:
    return [get_asset(a) for a in asset_ids]
