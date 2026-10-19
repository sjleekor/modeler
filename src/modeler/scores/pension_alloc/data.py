"""실제 자료 읽기와 ``--schema-only`` 점검 (사전등록 §4.1·F7·F9).

* :func:`schema_only`는 **값을 읽지 않는다.** 파일 경로, 열 이름, 자료형, 첫날·끝날, 행 수만
  돌려준다(동결 태그 없이 허용). 수익률·지수·금리·종가 값이나 그 집계는 어디에도 만들지 않는다.
* :func:`load_inputs`는 값을 읽는다. 동결 가드를 통과한 ``run.main``에서만 부른다.

세션 달력은 하나다: KOSPI 가격지수(ECOS ``market_kospi_ecos``) 관측일 전부(MRS와 같다). 월말도
이 목록에서 구한다. 현금은 기준선 함수(``build_cash_account``, 오래된 값 한도 14일 — 기준선과
같은 기본값)이고 일 현금수익은 ``daily_cash_returns``다. ETF는 ``etf_baseline`` 적재 함수를
그대로 쓴다. KOSPI 값의 가용 시각은 ``common.kr_inputs``의 ``available_from_date`` 08:30 KST
규칙이고, 월말 결정일 t의 값이 결정 시각(다음 세션 개장 30분 전)까지 가용한지 센다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.etf_baseline import inputs as I
from modeler.scores.etf_baseline import lines as L

KR_SNAPSHOT = "2026-10-18"
KOSPI_SERIES = "market_kospi_ecos"
CD91_SERIES = "rate_kr_cd91"
OBS_TABLE = "common_feature_observation_raw"
SOURCE = "source=sj2_remote"
END_DATE = date(2026, 10, 8)
ETF_CODES = ["069500", "143850", "114260", "148070", "132030"]  # R-SP 슬롯 순서(현금은 따로)
CASH_STALENESS_DAYS = 14  # build_cash_account 기본값 = 기준선과 같다


def obs_dir(root: Path, snapshot: str = KR_SNAPSHOT) -> Path:
    return root / "kr/raw/raw_postgres" / f"snapshot_date={snapshot}" / SOURCE / OBS_TABLE


# --------------------------------------------------------------------------- 스키마만
def schema_only(root: Path) -> str:
    """입력마다 경로·열 이름·자료형·첫날·끝날·행 수만 문자열로 낸다. 값은 읽지 않는다."""
    out: list[str] = []
    od = obs_dir(root)
    files = sorted(od.rglob("*.parquet"))
    out.append(f"[KR raw {OBS_TABLE}] snapshot={KR_SNAPSHOT} files={len(files)}")
    for f in files:
        out.append(f"  file: {f}")
    lf = pl.scan_parquet([str(f) for f in files], hive_partitioning=False)
    sch = lf.collect_schema()
    out.append("  columns: " + ", ".join(f"{k}:{v}" for k, v in sch.items()))
    # 날짜·계열 이름 열만 읽는다(값 열은 읽지 않는다).
    meta = lf.select("series_id", pl.col("observation_date").cast(pl.Date).alias("d")).collect()
    for sid in (KOSPI_SERIES, CD91_SERIES):
        s = meta.filter(pl.col("series_id") == sid)
        out.append(
            f"  series {sid}: rows={s.height} first_date={s['d'].min()} last_date={s['d'].max()}"
            f" distinct_dates={s['d'].n_unique()}"
        )
    avail = [c for c in ("available_from_date", "fetched_at") if c in sch.names()]
    for c in avail:
        a = (
            lf.filter(pl.col("series_id").is_in([KOSPI_SERIES, CD91_SERIES]))
            .select(pl.col(c).is_null().sum().alias("nulls"))
            .collect()
        )
        out.append(f"  column {c}: nulls_in_two_series={a['nulls'][0]}")

    etf_dir = root / I.ETF_DIR
    csv = etf_dir / I.ETF_FILE
    out.append(f"[ETF research copy] {csv} bytes={csv.stat().st_size}")
    hdr = pl.read_csv(csv, n_rows=0, infer_schema_length=0)
    out.append("  columns(all string as loaded by etf_baseline): " + ", ".join(hdr.columns))
    raw = pl.read_csv(csv, columns=["BAS_DD", "ISU_CD"], infer_schema_length=0)
    for c in ETF_CODES:
        s = raw.filter(pl.col("ISU_CD") == c)
        d = s["BAS_DD"].str.strptime(pl.Date, "%Y%m%d")
        out.append(f"  code {c}: rows={s.height} first_date={d.min()} last_date={d.max()}")
    mf = etf_dir / I.MANIFEST_FILE
    m = pl.read_csv(mf, separator="\t", infer_schema_length=0)
    days = I.load_trading_days(mf)
    out.append(f"[ETF trading-day manifest] {mf}")
    out.append(
        f"  columns: {', '.join(m.columns)} rows={m.height} trading_days={len(days)}"
        f" first_date={days[0]} last_date={days[-1]}"
    )
    dj = root / I.DIST_FILE
    import json

    data = json.loads(dj.read_text())
    keys = sorted(data[0].keys()) if data else []
    out.append(f"[SEIBRO distributions] {dj} bytes={dj.stat().st_size}")
    out.append(f"  top-level={type(data).__name__} records={len(data)} record_keys={keys}")
    mp = root / I.MAP_FILE
    mm = pl.read_csv(mp, infer_schema_length=0)
    out.append(f"[ETF code map] {mp} columns: {', '.join(mm.columns)} rows={mm.height}")
    return "\n".join(out)


# --------------------------------------------------------------------------- 현금 (F7)
def cash_returns(rates: pl.DataFrame, days: list[date]) -> np.ndarray:
    """기준선 장부와 같은 일 현금수익 (길이 len(days), 세션 i = i−1 -> i).

    금리(연 %)를 앞 값으로 채워(그 세션까지 가용한 가장 최근 관측) 달력일 수로 다시 복리 계산한다.
    관측이 ``CASH_STALENESS_DAYS``(14일, 기준선 기본값)보다 오래됐거나 아직 없으면 NaN이고
    채우지 않는다. 0일은 NaN이다. ``rates``: ``date, realtime_start, value``.
    """
    from modeler.scores.common.cash import build_cash_account

    acct = build_cash_account(
        rates, days, series_id=CD91_SERIES, staleness_days=CASH_STALENESS_DAYS
    )
    return I.daily_cash_returns(acct, len(days))


# --------------------------------------------------------------------------- 값 읽기
@dataclass
class RealInputs:
    days: list[date]  # 마스터 세션(KOSPI 관측일), END_DATE까지
    kospi_ret: np.ndarray  # (T,) 가격지수 일수익, 0일은 NaN
    cash_ret: np.ndarray  # (T,) 기준선과 같은 일 현금수익(불가·0일 NaN)
    etf_ret: np.ndarray  # (T, 5) 총수익(상장 전 NaN)
    end_idx: int
    month_ends: list[int]  # 마스터 달력의 월 마지막 거래일 인덱스
    kospi_available_ok: np.ndarray  # (T,) 날 d의 KOSPI 값이 d의 결정 시각까지 가용한가
    files: dict[str, Path]
    snapshots: dict[str, str]
    diagnostics: dict = field(default_factory=dict)
    stops: list[str] = field(default_factory=list)


def load_inputs(root: Path) -> RealInputs:
    """값을 읽는다. 동결 가드를 통과한 진입점에서만 부른다."""
    from modeler.etl.config import DataRoot
    from modeler.scores.common.calendar import SessionCalendar
    from modeler.scores.common.kr_inputs import (
        COMMON_OBS_TABLE,
        KrLake,
        _load_series,
        _with_available_at,
        load_kr_rates,
    )

    lake = KrLake.resolve(
        DataRoot(base=root / "kr"), snapshot_date=KR_SNAPSHOT, tables=(COMMON_OBS_TABLE,)
    )
    stops: list[str] = []
    first = _load_series(lake, KOSPI_SERIES, ())
    if first is None:
        raise I.InputError(f"{KOSPI_SERIES}가 없습니다")
    all_dates = sorted(set(first["date"].to_list()))
    raw = _load_series(lake, KOSPI_SERIES, all_dates)
    assert raw is not None
    raw = raw.filter(pl.col("date") <= END_DATE)
    days = raw["date"].to_list()
    T = len(days)
    if days[-1] != END_DATE:
        stops.append(f"KOSPI 마지막 날 {days[-1]} != {END_DATE}")
    lvl = raw["value"].to_numpy()
    kospi_ret = np.full(T, np.nan)
    kospi_ret[1:] = lvl[1:] / lvl[:-1] - 1.0

    # 현금: 기준선과 같은 함수
    rates = load_kr_rates(lake, CD91_SERIES, days)
    if rates is None:
        raise I.InputError("rate_kr_cd91이 없습니다")
    cash_ret = cash_returns(rates, days)

    # ETF: 기준선 적재 함수. 마스터 세션 위에서 만든다.
    etf_dir = root / I.ETF_DIR
    csv, mf = etf_dir / I.ETF_FILE, etf_dir / I.MANIFEST_FILE
    etf_days = [d for d in I.load_trading_days(mf) if d <= END_DATE]
    pos = {d: i for i, d in enumerate(days)}
    not_in_master = [d for d in etf_days if d not in pos]
    if not_in_master:
        stops.append(
            f"ETF 거래일 중 KOSPI 세션에 없는 날 {len(not_in_master)}개 (첫 {not_in_master[0]})"
        )
    etf_set = set(etf_days)
    extra = [d for d in days if d >= etf_days[0] and d not in etf_set]
    if extra:
        stops.append(f"ETF 구간 안에서 ETF 거래일에 없는 KOSPI 세션 {len(extra)}개 (첫 {extra[0]})")
    sub = [d for d in days if d >= etf_days[0]]
    off = pos[sub[0]]
    frames = I.load_etf_frames(csv, ETF_CODES)
    dist = I.load_distributions(root / I.DIST_FILE, root / I.MAP_FILE, ETF_CODES)
    panel = I.build_return_panel(sub, frames, dist.rows)
    etf_ret = np.full((T, len(ETF_CODES)), np.nan)
    for k, c in enumerate(ETF_CODES):
        etf_ret[off:, k] = panel.ret[c]
        miss = [d for d in panel.missing_days[c]]
        if miss:
            stops.append(f"{c}: 상장 뒤 종가 없는 세션 {len(miss)}개 (첫 {miss[0]})")
        if panel.last_date[c] != END_DATE:
            stops.append(f"{c}: 마지막 날 {panel.last_date[c]} != {END_DATE}")
    for j in panel.jumps:
        stops.append(f"{j['code']} {j['date']} 보정 뒤에도 하루 수익 {j['ret']:.1%}")

    # KOSPI 가용성: 날 d의 값이 d의 결정 시각(다음 세션 개장 30분 전)까지 가용한가
    av = _with_available_at(raw)
    cal = SessionCalendar.from_observed_dates("XKRX", days)
    ok = np.zeros(T, dtype=bool)
    avail_at = av["available_at"].to_list()
    for i in range(T - 1):
        ok[i] = avail_at[i] <= cal.decision_at(days[i])
    ok[T - 1] = False  # 마지막 날은 결정 시각이 없다

    files = {
        **{f"kr_obs_{p.name}": p for p in sorted(obs_dir(root).rglob("*.parquet"))},
        "etf_daily": csv,
        "manifest_files": mf,
        "seibro_dist": root / I.DIST_FILE,
        "dist_code_map": root / I.MAP_FILE,
    }
    return RealInputs(
        days=days,
        kospi_ret=kospi_ret,
        cash_ret=cash_ret,
        etf_ret=etf_ret,
        end_idx=T - 1,
        month_ends=L.month_end_indices(days),
        kospi_available_ok=ok,
        files=files,
        snapshots={OBS_TABLE: KR_SNAPSHOT},
        diagnostics={
            "kospi_sessions": T,
            "kospi_first": days[0],
            "kospi_last": days[-1],
            "cash_null_sessions": int(np.isnan(cash_ret[1:]).sum()),
            "cash_first_valid": next(
                (days[i] for i in range(T) if not np.isnan(cash_ret[i])), None
            ),
            "kospi_available_late_sessions": int((~ok[:-1]).sum()),
            "etf_first_date": {c: panel.first_date[c] for c in ETF_CODES},
            "dist_unplaced": len(panel.dist_unplaced),
        },
        stops=stops,
    )
