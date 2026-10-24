"""정본(collector ``krx_baseline`` 저장소) 읽기 — ETF 일봉·거래일·SEIBro 분배금.

collector를 import하지 않는다(modeler가 의존하는 collector 판에 이 저장소 코드가 없을 수 있다).
저장 계약(collector ``kr/baseline``, 계획 04 §3.1)에서 읽는 쪽에 필요한 최소 규칙만 옮겼다.

* 위치: ``<STOCK_DATA_ROOT>/kr/raw/krx_baseline`` (서버 미러 ``kr-baseline-mirror.sh``).
* **완료 manifest(``_manifest/*.json``)가 가리키는 파일만 읽는다.** manifest가 없는 parquet은
  없는 것이다. 읽기 전에 manifest의 바이트 수·sha256과 맞는지 확인한다.
* 관측 표 ``etf_daily``(키 ``BAS_DD, ISU_CD``)·``etf_distribution``(키 ``ISIN, RGT_STD_DT,
  RGT_RSN_DTAIL_NM``)는 ``<표>/year=YYYY/<run_id>__<attempt>__<batch>.parquet``이고 같은 키가
  여러 번 관측될 수 있다. ``obs_seq``가 1이면 최초 관측본, 키별 최대 ``obs_seq``가 최신본이며
  그것이 ``obs_kind=absent``면 뺀다. 경로의 ``year=``는 열이 아니다(``pq.ParquetFile``로 읽는다).
* 날짜 상태·거래일은 요청 기록 ``fetch_log/etf_bydd_trd``의 ``day_kind``(``trading``)이 정본이다.
  요청 키 ``bas_dd=YYYYMMDD``마다 ``result != failed``인 **마지막** 기록을 본다.
* 분배금 종목 코드는 ``ISIN[3:9]``다(대응표 CSV가 필요 없다).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from modeler.scores.etf_baseline import inputs as I

CANON_REL = "kr/raw/krx_baseline"
MANIFEST_DIR = "_manifest"
ETF_SERVICE = "etf_bydd_trd"
ETF_TABLE = "etf_daily"
DIST_SERVICE = "seibro_dist"
DIST_TABLE = "etf_distribution"
HISTORY_START = date(2010, 1, 4)  # 조사 사본과 같은 시작일 — 이보다 늦으면 멈춘다
DIST_FULL_START = "20030101"  # SEIBro 전체 받기 창 시작 (collector EARLIEST_RGT_STD_DT)
VIEWS = ("latest", "first")

ETF_READ_COLS = ["BAS_DD", "ISU_CD", "TDD_CLSPRC", "LIST_SHRS"]
DIST_READ_COLS = [
    "ISIN",
    "RGT_STD_DT",
    "TH1_PAY_TERM_BEGIN_DT",
    "ESTM_STDPRC",
    "RGT_RSN_DTAIL_NM",
]
_OBS_COLS = ["obs_seq", "obs_kind"]


def canonical_base(root: Path, override: Path | None = None) -> Path:
    return Path(override) if override is not None else root / CANON_REL


def _sha256_file(path: Path) -> str:
    return I.sha256_file(path)


@dataclass
class CanonicalStore:
    base: Path
    #: 읽은 parquet의 sha256 (manifest 기록값과 대조한 뒤의 값). 결과 manifest용.
    verified: dict[str, str]

    def __init__(self, base: Path):
        self.base = Path(base)
        self.verified = {}
        if not (self.base / MANIFEST_DIR).is_dir():
            raise I.InputError(
                f"{self.base}: 정본 _manifest 디렉터리가 없습니다 (미러를 돌렸습니까)"
            )
        self._manifests = [
            (p, json.loads(p.read_text("utf-8")))
            for p in sorted((self.base / MANIFEST_DIR).glob("*.json"))
        ]
        if not self._manifests:
            raise I.InputError(f"{self.base}: 완료 manifest가 하나도 없습니다")

    # ---------------------------------------------------------------- manifest
    def manifest_digest(self) -> dict:
        """완료 manifest 전체의 지문: 파일별 sha256을 이름순으로 이어 다시 sha256."""
        h = hashlib.sha256()
        for p, _ in self._manifests:
            h.update(p.name.encode() + b"\0" + _sha256_file(p).encode() + b"\n")
        return {
            "count": len(self._manifests),
            "first": self._manifests[0][0].name,
            "last": self._manifests[-1][0].name,
            "sha256": h.hexdigest(),
        }

    def committed(self, kind: str, name: str) -> list[Path]:
        out = []
        for _, m in self._manifests:
            for e in m["files"]:
                if e["kind"] == kind and e.get("table", e.get("service")) == name:
                    out.append((e, self.base / e["path"]))
        return out

    def _check(self, entry: dict, path: Path) -> None:
        if not path.is_file():
            raise I.InputError(f"{entry['path']}: manifest에 있는데 파일이 없습니다")
        if path.stat().st_size != entry["bytes"]:
            raise I.InputError(f"{entry['path']}: 크기가 manifest와 다릅니다")
        sha = _sha256_file(path)
        if sha != entry["sha256"]:
            raise I.InputError(f"{entry['path']}: sha256이 manifest와 다릅니다")
        self.verified[entry["path"]] = sha

    def _read(self, kind: str, name: str, columns: list[str]) -> pl.DataFrame:
        tables = []
        for entry, path in self.committed(kind, name):
            self._check(entry, path)
            tables.append(pq.ParquetFile(path).read(columns=columns))
        if not tables:
            return pl.DataFrame({c: [] for c in columns}, schema={c: pl.String for c in columns})
        return pl.from_arrow(pa.concat_tables(tables))

    # ---------------------------------------------------------------- 관측 표
    def observations(
        self, table: str, columns: list[str], keys: list[str], view: str = "latest"
    ) -> pl.DataFrame:
        if view not in VIEWS:
            raise ValueError(f"view는 {VIEWS} 중 하나입니다: {view!r}")
        df = self._read("observation", table, list(dict.fromkeys([*columns, *_OBS_COLS, *keys])))
        df = df.select(list(dict.fromkeys([*columns, *keys, *_OBS_COLS])))
        if view == "first":
            df = df.filter(pl.col("obs_seq") == 1)
        else:
            df = (
                df.sort("obs_seq")
                .unique(subset=keys, keep="last", maintain_order=True)
                .filter(pl.col("obs_kind") != "absent")
            )
        return df.select(columns)

    def fetch_log(self, service: str) -> pl.DataFrame:
        cols = ["request_key", "fetched_at", "result", "day_kind"]
        return self._read("fetch_log", service, cols)


# --------------------------------------------------------------------------- 거래일
def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def load_trading_days(store: CanonicalStore, end_date: date) -> tuple[list[date], dict]:
    """``day_kind=trading`` 날짜(요청 키별 성공한 마지막 기록). ``end_date`` 이하만.

    ``2010-01-04``부터 ``end_date``까지 평일 전부에 성공한 기록이 있어야 한다. 하나라도 비면
    거래일 달력이 조용히 짧아지므로 ``InputError``로 멈춘다.
    """
    log = store.fetch_log(ETF_SERVICE)
    if log.is_empty():
        raise I.InputError(f"fetch_log/{ETF_SERVICE}가 비었습니다")
    ok = log.filter(pl.col("result") != "failed").sort("fetched_at")
    last = ok.unique(subset=["request_key"], keep="last", maintain_order=True)
    kind_by_day: dict[date, str | None] = {}
    for key, kind in zip(last["request_key"].to_list(), last["day_kind"].to_list()):
        if not key.startswith("bas_dd="):
            raise I.InputError(f"{ETF_SERVICE}: 모르는 요청 키 {key!r}")
        kind_by_day[datetime.strptime(key[7:], "%Y%m%d").date()] = kind
    missing = [d for d in _weekdays(HISTORY_START, end_date) if d not in kind_by_day]
    if missing:
        raise I.InputError(
            f"{ETF_SERVICE}: {HISTORY_START}~{end_date} 평일 중 성공한 요청 기록이 없는 날 "
            f"{len(missing)}개 (첫 {missing[0]}, 끝 {missing[-1]})"
        )
    days = sorted(d for d, k in kind_by_day.items() if k == "trading" and d <= end_date)
    if not days or days[-1] != end_date:
        raise I.InputError(
            f"{ETF_SERVICE}: 끝 날짜 {end_date}가 거래일이 아니거나 없습니다 "
            f"(정본 마지막 거래일 {days[-1] if days else None})"
        )
    info = {
        "calendar_last_trading_day": max(
            (d for d, k in kind_by_day.items() if k == "trading"), default=None
        ),
        "requested_days": len(kind_by_day),
        "trading_days_to_end": len(days),
        "day_kind_counts": {
            k: sum(1 for v in kind_by_day.values() if v == k)
            for k in sorted({str(v) for v in kind_by_day.values()})
        },
    }
    return days, info


# --------------------------------------------------------------------------- 일봉
def load_etf_frames(store: CanonicalStore, codes: list[str], view: str = "latest"):
    df = store.observations(ETF_TABLE, ETF_READ_COLS, ["BAS_DD", "ISU_CD"], view)
    if df.is_empty():
        raise I.InputError(f"{ETF_TABLE} 관측 표가 비었습니다")
    return I.frames_from_strings(df, codes)


# --------------------------------------------------------------------------- 분배금
def check_dist_windows(store: CanonicalStore, end_date: date) -> dict:
    """SEIBro 전체 받기 창이 끝났고 마지막 완료 창이 ``end_date`` 이후까지 덮는지."""
    log = store.fetch_log(DIST_SERVICE)
    done = log.filter(
        (pl.col("day_kind") == "complete") & pl.col("request_key").str.starts_with("window=")
    )
    if done.is_empty():
        raise I.InputError(f"{DIST_SERVICE}: 완료된 창 요약 기록이 없습니다")
    wins = [k[7:].split("_") for k in done["request_key"].to_list()]
    if not any(a == DIST_FULL_START for a, _ in wins):
        raise I.InputError(f"{DIST_SERVICE}: {DIST_FULL_START}부터의 전체 받기 완료 창이 없습니다")
    last_end = max(datetime.strptime(b, "%Y%m%d").date() for _, b in wins)
    if last_end < end_date:
        raise I.InputError(
            f"{DIST_SERVICE}: 마지막 완료 창 끝 {last_end}가 끝 날짜 {end_date}보다 이릅니다"
        )
    return {"complete_windows": len(wins), "last_complete_window_end": last_end}


def load_distributions(
    store: CanonicalStore, codes: list[str], end_date: date, view: str = "latest"
) -> tuple[I.DistInfo, dict]:
    windows = check_dist_windows(store, end_date)
    keys = ["ISIN", "RGT_STD_DT", "RGT_RSN_DTAIL_NM"]
    df = store.observations(DIST_TABLE, DIST_READ_COLS, keys, view)
    if df.is_empty():
        raise I.InputError(f"{DIST_TABLE} 관측 표가 비었습니다")
    isin_by_code: dict[str, str] = {}
    for c in codes:
        isins = sorted({i for i in df["ISIN"].to_list() if i and i[3:9] == c})
        if len(isins) > 1:
            raise I.InputError(f"{c}: 분배금 ISIN이 둘 이상입니다 {isins}")
        isin_by_code[c] = isins[0] if isins else ""  # 분배 기록이 한 번도 없는 종목
    info = I.distributions_from_records(df.to_dicts(), codes, isin_by_code)
    return info, windows
