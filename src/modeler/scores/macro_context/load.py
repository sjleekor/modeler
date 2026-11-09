"""입력 고정 확인과 읽기.

selection(``market-sector-selection.v1``, 시장·섹터와 같은 것)이 고정한 입력만 연다. 열기 전에 두 가지를
다시 확인한다 — ``common_feature_observation_raw``(한국 raw snapshot)와 ``macro_series``(미국 레이크).
레이크의 "최신 snapshot"을 스스로 찾지 않는다.

날짜 규칙(리포트 날짜 D = unit)

* 한국: ``available_from_date <= D`` 행만. 같은 관측일이 여러 행이면 가용일이 가장 늦은 행이 값이다.
* 미국: 빈 값을 뺀 뒤 날짜마다 ``realtime_start <= D`` 중 가장 늦은 vintage.
* 지난 보고서 대비로 다시 계산하는 앞 날짜 d(< D)에서 미국은 ``realtime_start < d``다. 미국 날짜 d에
  나온 vintage는 d 09:30 KST 보고서 뒤에 나왔기 때문이다. D 자체는 고정 snapshot이 D 09:30 KST
  전에 끝나 ``realtime_start == D``인 행이 들어올 수 없으므로 ``<= D`` 그대로 둔다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from modeler.etl.config import DataRoot
from modeler.scores.common.inputs import PinnedScopedLake
from modeler.scores.common.kr_inputs import COMMON_OBS_TABLE, KrLake
from modeler.scores.macro_context.spec import KR_IDS, US_IDS
from modeler.scores.market_sector import inputs_pin as ip

US_TABLE = "macro_series"


@dataclass
class MacroData:
    """읽은 원자료. 행은 ``(관측일, 값, 가용일)`` 오름차순이고 D 이후 가용분은 들어 있지 않다."""

    D: dt.date
    kr_rows: dict[str, list[tuple[dt.date, float, dt.date]]]
    us_rows: dict[str, list[tuple[dt.date, float, dt.date]]]
    kr_excluded: int
    us_snapshot: str
    inputs: dict[str, Any]


def verify_macro_pins(selection: dict[str, Any], D: dt.date) -> dict[str, Any]:
    """거시 섹션에 필요한 입력이 selection이 고정한 그대로인지 확인한다. 아니면 예외.

    ``InputPinError``(고정 기록이 없거나 형식이 다름), ``InputChangedError``(고정 뒤 파일이 바뀜).
    sha256을 다시 계산한다. 반환값은 문서 ``inputs``에 쓸 기록이다.
    """
    if selection.get("schema_version") != ip.SCHEMA:
        raise ip.InputPinError("selection 스키마가 맞지 않습니다")
    if dt.date.fromisoformat(selection["report_date"]) != D:
        raise ip.InputPinError("selection 리포트 날짜가 unit과 다릅니다")
    cutoff = ip._aware(selection["input_cutoff"], "input_cutoff")
    if cutoff != ip.input_cutoff(D):
        raise ip.InputPinError("input_cutoff이 D 09:30 KST가 아닙니다")
    roots = selection["roots"]
    kr = selection["kr"]
    if kr.get("status") != "selected":
        raise ip.InputPinError("KR snapshot이 selection에 없습니다")
    if ip._aware(kr["export_finished_at"], "export_finished_at") > cutoff:
        raise ip.InputPinError("KR export가 D 09:30 이후에 끝났습니다")
    rec_kr = kr["tables"].get(COMMON_OBS_TABLE)
    if rec_kr is None:
        raise ip.InputPinError(f"KR {COMMON_OBS_TABLE}가 selection에 없습니다")
    kr_root = Path(roots["kr"])
    source_dir = ip.kr_snapshot_dir(kr_root, kr["snapshot_date"])
    marker = ip.kr_marker(source_dir)
    if ip.sha256_file(marker["marker_path"]) != kr["success_marker_sha256"]:
        raise ip.InputChangedError("KR export 완료 marker가 selection 뒤에 바뀌었습니다")
    manifest = source_dir / "_manifests" / "table_manifests" / f"{COMMON_OBS_TABLE}.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ip.InputPinError(f"{COMMON_OBS_TABLE}: table manifest가 없습니다")
    if ip.sha256_file(manifest) != rec_kr["manifest_sha256"]:
        raise ip.InputChangedError(
            f"KR {COMMON_OBS_TABLE}: table manifest가 selection 뒤에 바뀌었습니다"
        )
    ip._same_files(
        f"KR {COMMON_OBS_TABLE}",
        rec_kr["files"],
        ip._records(source_dir / COMMON_OBS_TABLE, recursive=True),
    )
    rec_us = selection["us"]["tables"].get(US_TABLE, {})
    if rec_us.get("status") != "selected":
        raise ip.InputPinError(f"US {US_TABLE} snapshot이 selection에 없습니다")
    if ip._aware(rec_us["completed_at"], "completed_at") > cutoff:
        raise ip.InputPinError(f"US {US_TABLE}가 D 09:30 이후에 끝났습니다")
    live_us = ip.describe_us_table(Path(roots["us"]), US_TABLE, rec_us["snapshot_date"])
    ip._same_files(f"US {US_TABLE}", rec_us["files"], live_us["files"])
    return {
        "kr": {
            "table": COMMON_OBS_TABLE,
            "source": kr["source"],
            "snapshot_date": kr["snapshot_date"],
            "dir": f"raw/raw_postgres/snapshot_date={kr['snapshot_date']}/source={kr['source']}/{COMMON_OBS_TABLE}",
            "export_finished_at": kr["export_finished_at"],
            "pg_snapshot_id": kr.get("pg_snapshot_id"),
            "success_marker_sha256": kr["success_marker_sha256"],
            "manifest_sha256": rec_kr["manifest_sha256"],
            "files": {n: f["sha256"] for n, f in sorted(rec_kr["files"].items())},
            "series": list(KR_IDS),
        },
        "us": {
            "table": US_TABLE,
            "snapshot_date": rec_us["snapshot_date"],
            "dir": f"derived/snapshots/{US_TABLE}/snapshot_date={rec_us['snapshot_date']}",
            "completed_at": rec_us["completed_at"],
            "files": {n: f["sha256"] for n, f in sorted(rec_us["files"].items())},
            "series": list(US_IDS),
        },
        "selection": {
            "schema_version": selection["schema_version"],
            "report_date": selection["report_date"],
            "selection_mode": selection.get("selection_mode"),
            "input_cutoff": selection["input_cutoff"],
        },
    }


def _group(rows: list[tuple]) -> dict[str, list[tuple[dt.date, float, dt.date]]]:
    out: dict[str, list[tuple[dt.date, float, dt.date]]] = {}
    for sid, d, v, af in rows:
        out.setdefault(sid, []).append((d, v, af))
    return out


def load_macro_data(selection: dict[str, Any], D: dt.date) -> MacroData:
    """고정을 확인하고 두 표에서 필요한 series의 ``D`` 가용분을 읽는다."""
    inputs = verify_macro_pins(selection, D)
    roots = selection["roots"]
    kr_lake = KrLake.resolve(
        DataRoot(Path(roots["kr"])),
        snapshot_date=selection["kr"]["snapshot_date"],
        tables=(COMMON_OBS_TABLE,),
    )
    kr_all = (
        kr_lake.scan(COMMON_OBS_TABLE)
        .filter(
            pl.col("series_id").is_in(list(KR_IDS))
            & pl.col("value_numeric").is_not_null()
            & pl.col("available_from_date").is_not_null()
        )
        .select(
            pl.col("series_id").cast(pl.String),
            pl.col("observation_date").cast(pl.Date).alias("d"),
            pl.col("value_numeric").cast(pl.Float64).alias("v"),
            pl.col("available_from_date").cast(pl.Date).alias("af"),
        )
        .collect()
    )
    kr_excluded = int(kr_all.filter(pl.col("af") > D).height)
    kr_df = kr_all.filter(pl.col("af") <= D).sort(["series_id", "d", "af", "v"])
    us_snap = selection["us"]["tables"][US_TABLE]["snapshot_date"]
    us_lake = PinnedScopedLake(
        root=DataRoot(Path(roots["us"])), snapshots={US_TABLE: us_snap}, symbols=()
    )
    us_df = (
        us_lake.scan(US_TABLE)
        .filter(
            pl.col("series_id").is_in(list(US_IDS))
            & pl.col("value").is_not_null()
            & (pl.col("realtime_start") <= D)
        )
        .select(
            pl.col("series_id").cast(pl.String),
            pl.col("date").cast(pl.Date).alias("d"),
            pl.col("value").cast(pl.Float64).alias("v"),
            pl.col("realtime_start").alias("af"),
        )
        .collect()
        .sort(["series_id", "d", "af", "v"])
    )
    return MacroData(
        D=D,
        kr_rows=_group(kr_df.rows()),
        us_rows=_group(us_df.rows()),
        kr_excluded=kr_excluded,
        us_snapshot=us_snap,
        inputs=inputs,
    )


def view(rows: dict[str, list[tuple[dt.date, float, dt.date]]], d: dt.date, *,
         before: bool = False):
    """``d`` 가용분의 보기. ``(최신 값 보기, 처음 값이 생긴 가용일 보기)``.

    행이 (관측일, 가용일) 오름차순이라 마지막 행이 가장 늦은 가용분, 처음 본 행이 첫 가용일이다.
    ``before``이면 가용일이 ``d``보다 앞선 행만 쓴다(``가용일 < d``, 지난 보고서 대비의 미국 행).
    """
    cur: dict[str, dict[dt.date, tuple[float, dt.date]]] = {}
    first: dict[str, dict[dt.date, dt.date]] = {}
    for sid, lst in rows.items():
        c: dict[dt.date, tuple[float, dt.date]] = {}
        f: dict[dt.date, dt.date] = {}
        for obs, v, af in lst:
            if af > d or (before and af == d):
                continue
            c[obs] = (v, af)
            f.setdefault(obs, af)
        if c:
            cur[sid] = c
            first[sid] = f
    return cur, first
