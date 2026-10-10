"""회사 점수(부분 C) 분류 규칙·매핑표 (사전등록 20261010_quality_score §5.2·§5.3, 동결 대상).

이 파일 하나가 동결 대상이다. 규칙 문장(O1 의견 문자열, `stock_knd`, 발행주식수 변동 사건 목록)이
전부 여기 있고, 매핑표 생성 명령(`python -m modeler.scores.quality.company_maps`)이 같은 파일의
sha256을 manifest에 적는다.

들어오는 것은 raw 표 셋(`dart_governance_raw`·`dart_shareholder_return_raw`·
`dart_capital_change_raw`)뿐이다. 결과 변수(O1\\~O4)나 점수와 잇지 않고, 사건 수도 세지 않는다.
매핑표에는 문자열과 분류만 적고 **연도별 건수를 적지 않는다**(비적정 문자열의 연도별 건수가 곧
판정 구간 사건 수다). 연도별로는 분류 결측 비율만 적는다(§5.3).

사전등록 문면을 넘는 구현 선택은 상수와 ``# CI-<이름>`` 주석으로 남겼다. 규칙을 "더 맞게" 고치지
않았다. 규칙 문면대로 내용상 이상한 분류(예: ``감사의견 : 적정⏎반기검토의견 : 범위제한한정``이
비적정)도 그대로 둔다(작업 계획 D1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path

import polars as pl

from modeler.scores.quality.company_common import stock_data_root

RULES_VERSION = "quality-score-company/company_maps/1"

DIVIDEND_REPRT_CODE = "11011"  # 사업보고서
DPS_ROW_NAME = "주당 현금배당금(원)"
PAR_ROW_NAME = "주당액면가액(원)"

# CI-date: "날짜 이상값" 의 연도 하한. 사전등록은 "2923년 등 28행" 만 말하고 하한을 적지 않았다.
# 하한 1990·상한 raw snapshot 연도는 작업 지시서가 정했다.
DATE_YEAR_MIN = 1990

# §5.2 발행주식수 변동 사건 목록 — C3·O3·F7 공통.
CAPITAL_EVENT_TYPES: frozenset[str] = frozenset(
    {"무상증자", "주식분할", "무상감자", "유상감자", "주식배당"}
)
# 주식병합 확인(§12.4 C)에서 "감자 유형"으로 보는 둘.
REDUCTION_EVENT_TYPES: frozenset[str] = frozenset({"무상감자", "유상감자"})

# ---------------------------------------------------------------- 공통
_RE_WS = re.compile(r"\s+")  # 파이썬 \s 는 전각 공백(U+3000)·줄바꿈·탭을 포함한다.


def _strip_ws(s: str) -> str:
    return _RE_WS.sub("", s)


def default_raw_root() -> Path:
    """raw 표 루트 ``<STOCK_DATA_ROOT>/kr/raw/raw_postgres``."""
    return stock_data_root() / "kr" / "raw" / "raw_postgres"


def table_dir(raw_root: str | Path, raw_snapshot: str, table: str) -> Path:
    """한 raw 표의 디렉터리. snapshot 날짜·소스 파티션은 인자로 조립한다."""
    return Path(raw_root) / f"snapshot_date={raw_snapshot}" / "source=sj2_remote" / table


def _scan(raw_root: str | Path, raw_snapshot: str, table: str) -> pl.LazyFrame:
    pattern = str(table_dir(raw_root, raw_snapshot, table) / "**" / "*.parquet")
    return pl.scan_parquet(pattern, hive_partitioning=False)


def input_files(raw_root: str | Path, raw_snapshot: str, table: str) -> list[Path]:
    return sorted(table_dir(raw_root, raw_snapshot, table).rglob("*.parquet"))


def _map_unique(df: pl.DataFrame, col: str, fn: Callable, out: str, dtype) -> pl.DataFrame:
    """서로 다른 값에만 ``fn`` 을 돌려 ``out`` 열로 붙인다(수십 만 행에서도 빠르게)."""
    uniq = df.select(col).unique().drop_nulls()
    mapped = pl.DataFrame(
        {col: uniq[col].to_list(), out: [fn(v) for v in uniq[col].to_list()]},
        schema={col: df.schema[col], out: dtype},
    )
    return df.join(mapped, on=col, how="left")


# ================================================================ 1. O1 의견 분류
# 주석 표시: (주1) (*1) (*) (*1,*2) (*주5) (주) (주1,주2) — 괄호 안이 `*`·`주`와 숫자뿐인 꼴.
_RE_NOTE_PAREN = re.compile(r"\((?:[*주]+\d*)(?:,(?:[*주]+\d*|\d+))*\)")
_RE_NOTE_BARE = re.compile(r"주\d+\)")  # `적정 주1)` 처럼 여는 괄호가 없는 꼴

# CI-opinion-notes: 주석 표시 꼴의 정확한 범위(위 두 정규식)는 사전등록이 예시로만 적었다.
# 괄호 안 다른 글자는 남긴다(정정 C-1 ②): `임의감사 (한정)` 은 비적정.
NON_CLEAN_TOKENS = ("부적정", "비적정", "한정", "거절", "거부")
CLEAN_TOKENS = ("적정", "공정")


def normalize_opinion(s: str | None) -> str:
    """O1 의견 문자열 정규화(§5.3 + 정정 C-1 ②).

    모든 공백(전각 U+3000 포함)·줄바꿈을 지운 뒤 **주석 표시만** 지운다.
    ``None`` 은 빈 문자열이다.
    """
    if s is None:
        return ""
    t = _strip_ws(s)
    t = _RE_NOTE_PAREN.sub("", t)
    t = _RE_NOTE_BARE.sub("", t)
    return t


def classify_opinion(s: str | None) -> str | None:
    """O1 의견 문자열 문장 규칙(§5.3 "O1 의견 문자열 분류 — 동결 대상").

    ① 정규화 문자열에 ``부적정``·``비적정``·``한정``·``거절``·``거부`` 가 있으면 ``non_clean``.
    ② ①이 아니고 ``적정``·``공정`` 이 있으면 ``clean``.
    ③ 나머지(회계법인명·같음 표시·공란·검토·감사·없음·오타)는 ``None``(결측).
    같음 표시는 앞 행 값으로 채우지 않는다.
    """
    t = normalize_opinion(s)
    if any(tok in t for tok in NON_CLEAN_TOKENS):
        return "non_clean"
    if any(tok in t for tok in CLEAN_TOKENS):
        return "clean"
    return None


# ================================================================ 2. 의견 행의 연도 풀기
_RE_YEAR4 = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_RE_YEAR2 = re.compile(r"(?<!\d)(\d{2})년")
_RE_KI = re.compile(r"제?(\d{1,3})기")  # `제22기`·`22기`


def _label_marker(label: str) -> str | None:
    """라벨의 상대 표시. 전전기를 먼저 검사한다(``전전기`` 는 ``전기`` 를 포함한다)."""
    if "전전기" in label:
        return "p2"
    if "전기" in label:
        return "p1"
    if "당기" in label:
        return "cur"
    return None


# CI-anchor-max (사용자 승인 2026-10-10): `(당기)` 표시 행이 없는 보고서는 같은 접수번호 안의
# 가장 큰 기수 N(제N기)을 당기로 본다. (당기) 앵커가 있으면 그것을 쓴다.
ANCHOR_MAX = True


def _anchor_period(labels: Iterable[str | None], anchor_max: bool = ANCHOR_MAX) -> int | None:
    """같은 접수번호 안의 ``제M기(당기)`` 에서 M을 구한다.

    CI-anchor: 당기 표시 행이 가리키는 M이 하나가 아니면(연결·별도가 다르게 적었을 때) 앵커 없음.
    CI-anchor-max: 당기 표시 행이 아예 없으면 ``anchor_max`` 일 때 기수 최댓값을 앵커로 쓴다.
    """
    ms: set[int] = set()
    all_n: list[int] = []
    for lab in labels:
        if lab is None:
            continue
        t = _strip_ws(lab)
        m = _RE_KI.search(t)
        if m:
            all_n.append(int(m.group(1)))
        if _label_marker(t) == "cur" and m:
            ms.add(int(m.group(1)))
    if ms:
        return next(iter(ms)) if len(ms) == 1 else None
    if anchor_max and all_n:
        return max(all_n)
    return None


def _fiscal_year(label: str | None, report_year: int, anchor: int | None) -> int | None:
    if label is None:
        return None
    t = _strip_ws(label)
    marker = _label_marker(t)
    fy: int | None
    if marker == "p2":
        fy = report_year - 2
    elif marker == "p1":
        fy = report_year - 1
    elif marker == "cur":
        fy = report_year
    elif (m := _RE_YEAR4.search(t)) is not None:
        fy = int(m.group(1))
    elif (m := _RE_YEAR2.search(t)) is not None:
        fy = 2000 + int(m.group(1))
    elif (m := _RE_KI.search(t)) is not None and anchor is not None:
        # CI-label-first: 제N기가 여러 번 나오면 첫 번째를 쓴다.
        fy = report_year - (anchor - int(m.group(1)))
    else:
        return None
    # 범위 밖은 연도 미상(CI 로 보고): 한 보고서는 당기·전기·전전기 세 해만 담는다.
    return fy if report_year - 2 <= fy <= report_year else None


def opinion_fiscal_year(
    label: str | None, report_year: int, same_report_labels: Iterable[str | None] = ()
) -> int | None:
    """의견 행의 라벨을 사업연도로 푼다(§5.3 작은 것 4).

    ``전전기`` → report_year−2, ``전기`` → −1, ``당기`` → report_year, 4자리 연도 → 그 해,
    ``NN년`` → 2000+NN. ``제N기`` 만 있으면 같은 접수번호의 ``제M기(당기)`` 에서
    report_year − (M − N), 앵커가 없으면 ``None``. [report_year−2, report_year] 밖이면 ``None``.
    """
    return _fiscal_year(label, report_year, _anchor_period(same_report_labels))


def audit_opinion_rows(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    """``audit_opinion`` 행을 연도·분류와 함께 낸다.

    열: corp_code, rcept_no, report_year, row_ordinal, label_raw, fiscal_year, opinion_raw,
    opinion_norm, opinion_class.

    CI-g: 같은 접수번호·같은 payload 는 한 행으로 합친다(row_ordinal 은 가장 작은 값).
    2025 사업연도 행에 중복이 많다(17,646행 중 서로 다른 payload 12,689).
    """
    df = (
        _scan(raw_root, raw_snapshot, "dart_governance_raw")
        .filter(pl.col("statement_type") == "audit_opinion")
        .select("corp_code", "rcept_no", "bsns_year", "row_ordinal", "raw_payload")
        .collect()
        .sort("row_ordinal")
        .unique(subset=["corp_code", "rcept_no", "raw_payload"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("bsns_year").alias("report_year"),
            pl.col("raw_payload").str.json_path_match("$.bsns_year").alias("label_raw"),
            pl.col("raw_payload").str.json_path_match("$.adt_opinion").alias("opinion_raw"),
        )
        .drop("raw_payload", "bsns_year")
    )
    anchors: dict[str, int | None] = {}
    for rcept, labs in df.group_by("rcept_no").agg(pl.col("label_raw")).iter_rows():
        anchors[rcept] = _anchor_period(labs)
    fy = [
        _fiscal_year(lab, int(ry), anchors[rc])
        for lab, ry, rc in zip(df["label_raw"], df["report_year"], df["rcept_no"], strict=True)
    ]
    df = df.with_columns(pl.Series("fiscal_year", fy, dtype=pl.Int32))
    df = _map_unique(df, "opinion_raw", normalize_opinion, "opinion_norm", pl.String)
    df = _map_unique(df, "opinion_raw", classify_opinion, "opinion_class", pl.String)
    return df.select(
        "corp_code",
        "rcept_no",
        "report_year",
        "row_ordinal",
        "label_raw",
        "fiscal_year",
        "opinion_raw",
        "opinion_norm",
        "opinion_class",
    ).sort("corp_code", "rcept_no", "row_ordinal")


# ================================================================ 3. stock_knd 분류
def normalize_knd(s: str | None) -> str:
    """``stock_knd`` 정규화: 공백·전각 공백·줄바꿈 제거. ``None`` 은 빈 문자열."""
    return "" if s is None else _strip_ws(s)


_UNMARKED = frozenset({"", "-", "--", "*"})


def classify_stock_knd(s: str | None) -> str:
    """``stock_knd`` 분류(§5.2 + 정정 C-1 ① + 작업 계획 D4).

    (1) ``우선``·``종류``·``외`` 포함 → ``excluded``. (2) ``보통`` 포함 → ``common``.
    (3) ``-``·``--``·``*``·빈칸·``None`` → ``unmarked``. (4) 나머지 → ``other``(읽지 않음).
    """
    t = normalize_knd(s)
    if any(tok in t for tok in ("우선", "종류", "외")):
        return "excluded"
    if "보통" in t:
        return "common"
    if t in _UNMARKED:
        return "unmarked"
    return "other"


_RE_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_dps_cell(s: str | None) -> float | None:
    """배당 표 숫자 칸. 쉼표 제거 후 숫자, ``-`` 는 0.0, 그 밖은 ``None``(NaN, CI 로 보고).

    CI-dps-cell: 행이 있고 숫자 칸이 ``-`` 이면 0원(§5.2). ``None``(칸 자체가 없음)은 결측이다.
    """
    if s is None:
        return None
    t = _strip_ws(s).replace(",", "")
    if t == "-":
        return 0.0
    if _RE_NUM.match(t):
        return float(t)
    return None


def _max_or_none(vals: Sequence[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return max(xs) if xs else None


def select_dps(rows: Sequence[tuple[str, float | None, float | None, float | None]]) -> dict:
    """한 보고서의 DPS 행들에서 보통주 DPS를 고른다.

    ``rows`` 는 (knd_class, thstrm, frmtrm, lwfr). 보고서에 common 행이 있으면 common 행만,
    칸별(thstrm/frmtrm/lwfr) 최댓값(정정 C-1 ①). common 이 없고 unmarked 만 있으면 unmarked 행
    (CI-unmarked-multi: 여럿이고 값이 다르면 같은 방식으로 칸별 최댓값). 둘 다 없으면 NaN,
    other 가 있으면 ``other_only``, 아니면 ``none``(excluded 만 있는 경우 포함, CI-knd-none).
    """
    by: dict[str, list[tuple[float | None, float | None, float | None]]] = {}
    for k, t, p1, p2 in rows:
        by.setdefault(k, []).append((t, p1, p2))
    common = by.get("common", [])
    unmarked = by.get("unmarked", [])
    if common:
        source, use = "common", common
    elif unmarked:
        source, use = "unmarked", unmarked
    else:
        source, use = ("other_only" if by.get("other") else "none"), []
    nan = float("nan")
    if use:
        vals = [_max_or_none([r[i] for r in use]) for i in range(3)]
        t, p1, p2 = (nan if v is None else v for v in vals)
    else:
        t = p1 = p2 = nan
    return {
        "dps_t": t,
        "dps_p1": p1,
        "dps_p2": p2,
        "knd_source": source,
        "n_common_rows": len(common),
        "common_multi_value": len(set(common)) > 1,
        "n_unmarked_rows": len(unmarked),
        "unmarked_multi_value": len(set(unmarked)) > 1,
    }


def _dividend_rows(
    raw_root: str | Path, raw_snapshot: str, row_names: Sequence[str]
) -> pl.DataFrame:
    """사업보고서 배당 표의 지정 행. 같은 접수번호·같은 payload 는 한 행(긴 형식 중복 제거).

    ``dart_shareholder_return_raw`` 는 한 payload 가 metric_code(thstrm/frmtrm/lwfr) 셋으로
    반복된다. (corp_code, rcept_no, raw_payload) 로 합친다(CI-dps-dedupe).
    """
    df = (
        _scan(raw_root, raw_snapshot, "dart_shareholder_return_raw")
        .filter(
            (pl.col("statement_type") == "dividend")
            & (pl.col("reprt_code") == DIVIDEND_REPRT_CODE)
            & pl.col("row_name").is_in(list(row_names))
        )
        .select("corp_code", "rcept_no", "bsns_year", "row_name", "stock_knd", "raw_payload")
        .unique(subset=["corp_code", "rcept_no", "raw_payload"], maintain_order=True)
        .collect()
    )
    return df.with_columns(
        pl.col("bsns_year").alias("report_year"),
        pl.col("raw_payload").str.json_path_match("$.thstrm").alias("c_t"),
        pl.col("raw_payload").str.json_path_match("$.frmtrm").alias("c_p1"),
        pl.col("raw_payload").str.json_path_match("$.lwfr").alias("c_p2"),
    ).drop("raw_payload", "bsns_year")


def dividend_per_report(raw_root: str | Path, raw_snapshot: str, *, return_unparsed: bool = False):
    """보고서별 보통주 DPS(§5.2, 정정 C-1 ①).

    열: corp_code, rcept_no, report_year, dps_t, dps_p1, dps_p2, knd_source, n_common_rows,
    common_multi_value, n_unmarked_rows, unmarked_multi_value. dps_t=thstrm, dps_p1=frmtrm,
    dps_p2=lwfr. 배당 표(11011)가 있는 보고서는 모두 나오고, ``주당 현금배당금(원)`` 행이 없으면
    NaN·``none`` 이다. ``return_unparsed=True`` 면 (표, 숫자로 읽히지 않은 칸 문자열 Counter).
    """
    rows = _dividend_rows(raw_root, raw_snapshot, [DPS_ROW_NAME])
    all_reports = (
        _scan(raw_root, raw_snapshot, "dart_shareholder_return_raw")
        .filter(
            (pl.col("statement_type") == "dividend") & (pl.col("reprt_code") == DIVIDEND_REPRT_CODE)
        )
        .select("corp_code", "rcept_no", pl.col("bsns_year").alias("report_year"))
        .unique()
        .collect()
    )
    unparsed: Counter = Counter()
    groups: dict[tuple[str, str], list[tuple[str, float | None, float | None, float | None]]] = {}
    for corp, rc, knd, ct, cp1, cp2 in rows.select(
        "corp_code", "rcept_no", "stock_knd", "c_t", "c_p1", "c_p2"
    ).iter_rows():
        vals = []
        for raw in (ct, cp1, cp2):
            v = parse_dps_cell(raw)
            if v is None:
                unparsed["<null>" if raw is None else raw] += 1
            vals.append(v)
        groups.setdefault((corp, rc), []).append((classify_stock_knd(knd), *vals))
    out = []
    for corp, rc, ry in all_reports.sort("corp_code", "rcept_no").iter_rows():
        rec = select_dps(groups.get((corp, rc), []))
        out.append({"corp_code": corp, "rcept_no": rc, "report_year": ry, **rec})
    schema = {
        "corp_code": pl.String,
        "rcept_no": pl.String,
        "report_year": pl.Int32,
        "dps_t": pl.Float64,
        "dps_p1": pl.Float64,
        "dps_p2": pl.Float64,
        "knd_source": pl.String,
        "n_common_rows": pl.Int64,
        "common_multi_value": pl.Boolean,
        "n_unmarked_rows": pl.Int64,
        "unmarked_multi_value": pl.Boolean,
    }
    res = pl.DataFrame(out, schema=schema)
    return (res, unparsed) if return_unparsed else res


# ================================================================ 4. 발행주식수 변동 사건
def parse_event_date(s: str | None) -> date | None:
    """``YYYY.MM.DD``·``YYYY-MM-DD`` 만 읽는다. 그 밖의 꼴·없는 날짜는 ``None``."""
    if s is None:
        return None
    m = re.match(r"^(\d{4})[.-](\d{2})[.-](\d{2})$", s.strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def date_anomaly_reason(raw: str | None, snapshot_year: int) -> str | None:
    """날짜 이상값 사유. 정상이면 ``None``(CI-date)."""
    d = parse_event_date(raw)
    if d is None:
        return "unparsable"
    if d.year < DATE_YEAR_MIN:
        return "year_below_min"
    if d.year > snapshot_year:
        return "year_above_snapshot"
    return None


def _snapshot_year(raw_snapshot: str) -> int:
    return int(raw_snapshot[:4])


def _capital_rows(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    df = (
        _scan(raw_root, raw_snapshot, "dart_capital_change_raw")
        .select("corp_code", "rcept_no", "raw_payload")
        .collect()
        .with_columns(
            pl.col("raw_payload").str.json_path_match("$.isu_dcrs_de").alias("date_raw"),
            pl.col("raw_payload").str.json_path_match("$.isu_dcrs_stle").alias("event_type"),
            pl.col("raw_payload").str.json_path_match("$.isu_dcrs_qy").alias("qty_raw"),
        )
        .drop("raw_payload")
    )
    qty = pl.col("qty_raw").str.replace_all(r"[\s,]", "")
    return df.with_columns(
        pl.when(qty.str.contains(r"^-?\d+$")).then(qty.cast(pl.Int64, strict=False)).alias("qty"),
    )


def capital_event_type_table(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    """``isu_dcrs_stle`` 서로 다른 값 전부와 목록 포함 여부(열: event_type, in_list)."""
    df = _capital_rows(raw_root, raw_snapshot)
    types = sorted(df["event_type"].drop_nulls().unique().to_list())
    return pl.DataFrame(
        {"event_type": types, "in_list": [t in CAPITAL_EVENT_TYPES for t in types]},
        schema={"event_type": pl.String, "in_list": pl.Boolean},
    )


def capital_events_and_anomalies(
    raw_root: str | Path, raw_snapshot: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(사건표, 날짜 이상값표).

    사건표 열: corp_code, event_date, event_year, event_type, qty, rcept_no. 목록 유형만.
    (corp, 날짜, 유형, 수량) 중복은 하나로(여러 보고서가 같은 사건을 되풀이해 적음), rcept_no 는
    가장 작은 값. 날짜 이상값은 사건표에서 제외한다(CI-date). event_year = 날짜의 해(CI-f).
    이상값표 열: corp_code, rcept_no, date_raw, event_type, qty, reason, in_list. 유형이
    ``-`` 이고 날짜도 ``-`` 인 비사건 행은 이상값이 아니다.
    """
    df = _capital_rows(raw_root, raw_snapshot)
    snap_year = _snapshot_year(raw_snapshot)
    df = _map_unique(
        df, "date_raw", lambda s: date_anomaly_reason(s, snap_year), "reason", pl.String
    )
    df = df.with_columns(
        pl.when(pl.col("date_raw").is_null())
        .then(pl.lit("unparsable"))
        .otherwise(pl.col("reason"))
        .alias("reason"),
        pl.col("event_type").is_in(list(CAPITAL_EVENT_TYPES)).alias("in_list"),
    )
    # 이상값: 목록 유형이거나 날짜가 `-` 가 아닌 행 중 사유가 있는 것.
    anomalies = (
        df.filter(
            pl.col("reason").is_not_null()
            & (pl.col("in_list") | (pl.col("date_raw").fill_null("-") != "-"))
        )
        .group_by("corp_code", "date_raw", "event_type", "qty", "reason", "in_list")
        .agg(pl.col("rcept_no").min())
        .select("corp_code", "rcept_no", "date_raw", "event_type", "qty", "reason", "in_list")
        .sort("corp_code", "date_raw", "event_type")
    )
    ev = (
        df.filter(pl.col("in_list") & pl.col("reason").is_null())
        .with_columns(
            pl.col("date_raw")
            .str.replace_all(r"\.", "-")
            .str.to_date("%Y-%m-%d")
            .alias("event_date")
        )
        .group_by("corp_code", "event_date", "event_type", "qty")
        .agg(pl.col("rcept_no").min())
        .with_columns(pl.col("event_date").dt.year().cast(pl.Int32).alias("event_year"))
        .select("corp_code", "event_date", "event_year", "event_type", "qty", "rcept_no")
        .sort("corp_code", "event_date", "event_type", "qty")
    )
    return ev, anomalies


def capital_events(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    """발행주식수 변동 사건(§5.2 목록 다섯 유형만, 날짜 이상값 제외, 중복 제거)."""
    return capital_events_and_anomalies(raw_root, raw_snapshot)[0]


# ================================================================ 5. 주식병합 확인 (§12.4 C)
def par_value_changes(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    """사업보고서 배당 표 ``주당액면가액(원)`` 에서 thstrm ≠ frmtrm 인 corp-year.

    열: corp_code, year(report_year), par_t, par_p1. 둘 다 숫자로 읽힌 행만 본다.
    CI-par: 액면이 바뀐 해 = 보고서 사업연도(thstrm 해). 같은 corp-year 에 행·보고서가 여럿이면
    하나라도 다르면 바뀐 것으로 본다(비교 가능한 corp-year 는 :func:`par_comparable`).
    """
    return _par_rows(raw_root, raw_snapshot).filter(pl.col("par_t") != pl.col("par_p1"))


def _par_rows(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    rows = _dividend_rows(raw_root, raw_snapshot, [PAR_ROW_NAME])

    def num(c: str) -> pl.Expr:
        t = pl.col(c).str.replace_all(r"[\s,]", "")
        return pl.when(t.str.contains(r"^\d+(?:\.\d+)?$")).then(t.cast(pl.Float64, strict=False))

    return (
        rows.with_columns(num("c_t").alias("par_t"), num("c_p1").alias("par_p1"))
        .filter(pl.col("par_t").is_not_null() & pl.col("par_p1").is_not_null())
        .select("corp_code", pl.col("report_year").alias("year"), "par_t", "par_p1")
        .unique()
        .sort("corp_code", "year", "par_t", "par_p1")
    )


def share_event_years(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    """발행주식수 변동 사건 해(정정 C-2: 목록 사건 + 액면 변경 해).

    열: corp_code, event_year, source(``capital_change``·``par_change``), event_type.
    ``par_change`` 는 사업보고서 배당 표 ``주당액면가액(원)`` 행에서 한 보고서의 thstrm ≠ frmtrm
    이면 그 보고서 사업연도 t, frmtrm ≠ lwfr 이면 t−1 을 사건 해로 보고(event_type ``액면변경``),
    여러 보고서의 해는 합집합이다. 행 선택은 DPS와 같은 stock_knd 규칙(common 우선, 없으면
    unmarked)이고, 두 칸이 모두 숫자일 때만 비교한다(``-``·결측은 변경이 아니다).
    ``capital_change`` 는 :func:`capital_events` 의 사건 연도다. 한 (corp, year)에 두 원천이
    모두 있으면 행이 둘이다.
    """
    cap = (
        capital_events(raw_root, raw_snapshot)
        .select(
            "corp_code",
            pl.col("event_year"),
            pl.lit("capital_change").alias("source"),
            pl.col("event_type"),
        )
        .unique()
    )
    rows = _dividend_rows(raw_root, raw_snapshot, [PAR_ROW_NAME])
    rows = _map_unique(rows, "stock_knd", classify_stock_knd, "k", pl.String).with_columns(
        pl.col("k").fill_null("unmarked")  # stock_knd 가 null 이면 빈칸과 같다
    )
    rows = rows.filter(pl.col("k").is_in(["common", "unmarked"]))
    # 보고서마다 common 이 있으면 common 행만, 없으면 unmarked 행.
    has_common = rows.group_by("rcept_no").agg((pl.col("k") == "common").any().alias("hc"))
    rows = rows.join(has_common, on="rcept_no").filter(
        pl.when(pl.col("hc")).then(pl.col("k") == "common").otherwise(True)
    )

    def num(c: str) -> pl.Expr:
        t = pl.col(c).str.replace_all(r"[\s,]", "")
        return pl.when(t.str.contains(r"^\d+(?:\.\d+)?$")).then(t.cast(pl.Float64, strict=False))

    rows = rows.with_columns(
        num("c_t").alias("t"), num("c_p1").alias("p1"), num("c_p2").alias("p2")
    )
    par = pl.concat(
        [
            rows.filter(pl.col("t") != pl.col("p1")).select(
                "corp_code", pl.col("report_year").alias("event_year")
            ),
            rows.filter(pl.col("p1") != pl.col("p2")).select(
                "corp_code", (pl.col("report_year") - 1).alias("event_year")
            ),
        ]
    ).unique()
    par = par.select(
        "corp_code",
        pl.col("event_year").cast(pl.Int32),
        pl.lit("par_change").alias("source"),
        pl.lit("액면변경").alias("event_type"),
    )
    return pl.concat([cap.with_columns(pl.col("event_year").cast(pl.Int32)), par]).sort(
        "corp_code", "event_year", "source", "event_type"
    )


# 판정에 쓰는 사건 원천은 동결 목록(증자·감자 표의 다섯 유형)뿐이다. 액면 변경 해("par_change")는
# 사전등록 정정 C-2(결과 전 기록용, 02_rules A4·00_plan P3)에 따라 기록용 비교에만 더한다.
JUDGMENT_EVENT_SOURCES = ("capital_change",)
RECORD_EVENT_SOURCES = ("capital_change", "par_change")


def share_event_flags(
    raw_root: str | Path,
    raw_snapshot: str,
    sources: tuple[str, ...] = JUDGMENT_EVENT_SOURCES,
) -> pl.DataFrame:
    """사건 해가 하나라도 있는 (corp_code, year)만 고유하게 낸다(C3·O3·F7 공통 표시).

    기본은 판정용 동결 목록(``JUDGMENT_EVENT_SOURCES``). 기록용 비교는
    ``sources=RECORD_EVENT_SOURCES``(액면 변경 해 포함, 정정 C-2).
    """
    return (
        share_event_years(raw_root, raw_snapshot)
        .filter(pl.col("source").is_in(list(sources)))
        .select("corp_code", pl.col("event_year").alias("year"))
        .unique()
        .sort("corp_code", "year")
    )


def par_change_check(raw_root: str | Path, raw_snapshot: str) -> tuple[pl.DataFrame, dict]:
    """주식병합 확인. (액면이 바뀐 corp-year 표, 요약 dict).

    표 열: corp_code, year, par_t, par_p1, has_listed_event, listed_event_types.
    요약: 액면 변경 해 수, 그중 목록 사건 있는/없는 수, 비교 가능한 corp-year 중 감자
    (무상·유상) 사건 해의 액면 변경 비율. 입력(사건 목록) 확인이라 점수·결과와 잇지 않는다.
    """
    par = _par_rows(raw_root, raw_snapshot)
    ev = capital_events(raw_root, raw_snapshot)
    ev_year = (
        ev.group_by("corp_code", pl.col("event_year").alias("year"))
        .agg(pl.col("event_type").unique().sort().alias("types"))
        .with_columns(pl.col("types").list.join("|").alias("listed_event_types"))
    )
    # corp-year 단위로 합친다: 바뀐 행이 하나라도 있으면 변경.
    cy = (
        par.with_columns((pl.col("par_t") != pl.col("par_p1")).alias("chg"))
        .group_by("corp_code", "year")
        .agg(
            pl.col("chg").any().alias("changed"),
            pl.col("par_t").filter(pl.col("chg")).first().alias("par_t"),
            pl.col("par_p1").filter(pl.col("chg")).first().alias("par_p1"),
        )
        .join(ev_year, on=["corp_code", "year"], how="left")
        .with_columns(
            pl.col("listed_event_types").is_not_null().alias("has_listed_event"),
            pl.col("types")
            .list.eval(pl.element().is_in(list(REDUCTION_EVENT_TYPES)))
            .list.any()
            .fill_null(False)
            .alias("has_reduction_event"),
        )
    )
    changed = cy.filter(pl.col("changed"))
    red = cy.filter(pl.col("has_reduction_event"))
    summary = {
        "n_comparable_corp_years": cy.height,
        "n_par_changed_corp_years": changed.height,
        "n_changed_with_listed_event": int(changed["has_listed_event"].sum()),
        "n_changed_without_listed_event": int((~changed["has_listed_event"]).sum()),
        "n_reduction_event_corp_years_comparable": red.height,
        "n_reduction_event_corp_years_par_changed": int(red["changed"].sum()),
        "reduction_event_par_changed_ratio": (
            float(red["changed"].sum()) / red.height if red.height else math.nan
        ),
    }
    table = changed.select(
        "corp_code",
        "year",
        "par_t",
        "par_p1",
        "has_listed_event",
        pl.col("listed_event_types").fill_null(""),
    ).sort("corp_code", "year")
    return table, summary


# ================================================================ 6. 매핑표 생성
def _esc(v) -> str:
    """TSV 셀. 줄바꿈·탭을 이스케이프한다(``\\n``)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(v)
    return (
        str(v).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    )


def write_tsv(df: pl.DataFrame, path: Path) -> None:
    lines = ["\t".join(df.columns)]
    for row in df.iter_rows():
        lines.append("\t".join(_esc(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def opinion_map(op: pl.DataFrame) -> pl.DataFrame:
    """서로 다른 의견 원문 전부와 정규화·분류. **건수 열이 없다**(§5.3)."""
    return (
        op.select("opinion_raw", "opinion_norm", "opinion_class")
        .drop_nulls("opinion_raw")
        .unique()
        .sort("opinion_raw")
    )


def opinion_missing_by_year(op: pl.DataFrame) -> pl.DataFrame:
    """연도별 행 수·연도 미상·분류 결측과 비율. 비적정·적정 건수는 넣지 않는다.

    열: by, year, n_rows, n_year_unknown, n_class_missing, class_missing_ratio.
    ``by=report_year`` 행은 보고서 연도별, ``by=fiscal_year`` 행은 풀린 사업연도별이다
    (``fiscal_year`` 행의 ``n_year_unknown`` 은 비운다).
    """
    a = (
        op.group_by("report_year")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("fiscal_year").is_null().sum().alias("n_year_unknown"),
            pl.col("opinion_class").is_null().sum().alias("n_class_missing"),
        )
        .with_columns(pl.lit("report_year").alias("by"), pl.col("report_year").alias("year"))
        .drop("report_year")
    )
    b = (
        op.filter(pl.col("fiscal_year").is_not_null())
        .group_by("fiscal_year")
        .agg(
            pl.len().alias("n_rows"),
            pl.lit(None, dtype=pl.UInt32).alias("n_year_unknown"),
            pl.col("opinion_class").is_null().sum().alias("n_class_missing"),
        )
        .with_columns(pl.lit("fiscal_year").alias("by"), pl.col("fiscal_year").alias("year"))
        .drop("fiscal_year")
    )
    cols = ["by", "year", "n_rows", "n_year_unknown", "n_class_missing"]
    out = pl.concat(
        [
            a.select(cols).with_columns(pl.col("year").cast(pl.Int32)),
            b.select(cols).with_columns(pl.col("year").cast(pl.Int32)),
        ]
    )
    return out.with_columns(
        (pl.col("n_class_missing") / pl.col("n_rows")).alias("class_missing_ratio")
    ).sort("by", "year", descending=[True, False])


def stock_knd_map(raw_root: str | Path, raw_snapshot: str) -> pl.DataFrame:
    """서로 다른 ``stock_knd`` 원문 전부와 정규화·분류, 그 값을 가진 사업보고서 수.

    사업보고서 수는 ``주당 현금배당금(원)`` 행이 있는 접수번호 수다(배당 건수는 사건이 아니다).
    """
    rows = _dividend_rows(raw_root, raw_snapshot, [DPS_ROW_NAME])
    cnt = rows.group_by("stock_knd").agg(pl.col("rcept_no").n_unique().alias("n_reports"))
    cnt = _map_unique(cnt, "stock_knd", normalize_knd, "knd_norm", pl.String)
    cnt = _map_unique(cnt, "stock_knd", classify_stock_knd, "knd_class", pl.String)
    # `stock_knd` 가 null 인 행은 빈칸과 같은 unmarked 로 읽는다.
    cnt = cnt.with_columns(
        pl.when(pl.col("stock_knd").is_null())
        .then(pl.lit(""))
        .otherwise(pl.col("knd_norm"))
        .alias("knd_norm"),
        pl.when(pl.col("stock_knd").is_null())
        .then(pl.lit("unmarked"))
        .otherwise(pl.col("knd_class"))
        .alias("knd_class"),
    )
    return cnt.select("stock_knd", "knd_norm", "knd_class", "n_reports").sort("stock_knd")


def other_only_reports(raw_root: str | Path, raw_snapshot: str, div: pl.DataFrame) -> pl.DataFrame:
    """규칙 (1)~(3) 밖의 ``stock_knd`` 만 있는 보고서 목록(corp_code·rcept_no·연도·값)."""
    rows = _dividend_rows(raw_root, raw_snapshot, [DPS_ROW_NAME])
    keys = div.filter(pl.col("knd_source") == "other_only").select("corp_code", "rcept_no")
    vals = (
        rows.join(keys, on=["corp_code", "rcept_no"])
        .with_columns(
            pl.col("stock_knd").map_elements(classify_stock_knd, return_dtype=pl.String).alias("k")
        )
        .filter(pl.col("k") == "other")
        .group_by("corp_code", "rcept_no", "report_year")
        .agg(pl.col("stock_knd").unique().sort().str.join("|").alias("values"))
    )
    return vals.sort("corp_code", "rcept_no")


def dps_multi_value_reports(
    raw_root: str | Path, raw_snapshot: str, div: pl.DataFrame
) -> pl.DataFrame:
    """보통주(또는 미표기) 행이 여럿이고 값이 다른 보고서 목록과 행별 값."""
    flagged = div.filter(pl.col("common_multi_value") | pl.col("unmarked_multi_value")).select(
        "corp_code", "rcept_no", "report_year", "knd_source", "dps_t", "dps_p1", "dps_p2"
    )
    rows = _dividend_rows(raw_root, raw_snapshot, [DPS_ROW_NAME]).with_columns(
        pl.col("stock_knd").map_elements(classify_stock_knd, return_dtype=pl.String).alias("k")
    )
    rows = rows.filter(pl.col("k").is_in(["common", "unmarked"])).with_columns(
        (
            pl.col("stock_knd").fill_null("")
            + ":"
            + pl.col("c_t").fill_null("")
            + "/"
            + pl.col("c_p1").fill_null("")
            + "/"
            + pl.col("c_p2").fill_null("")
        ).alias("row")
    )
    per = rows.group_by("corp_code", "rcept_no", "k").agg(
        pl.col("row").sort().str.join(" | ").alias("rows")
    )
    return (
        flagged.join(
            per,
            left_on=["corp_code", "rcept_no", "knd_source"],
            right_on=["corp_code", "rcept_no", "k"],
            how="left",
        )
        .rename({"dps_t": "chosen_t", "dps_p1": "chosen_p1", "dps_p2": "chosen_p2"})
        .sort("corp_code", "rcept_no")
    )


def build_maps(raw_root: str | Path, raw_snapshot: str, out_dir: Path) -> dict:
    """매핑표 파일과 manifest 를 쓰고 manifest dict 를 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, pl.DataFrame] = {}
    stats: dict = {}

    op = audit_opinion_rows(raw_root, raw_snapshot)
    files["opinion_map.tsv"] = opinion_map(op)
    files["opinion_missing_by_year.tsv"] = opinion_missing_by_year(op)
    om = files["opinion_map.tsv"]
    stats["opinion"] = {
        "n_rows_after_dedupe": op.height,
        "n_distinct_opinion_strings": om.height,
        "n_strings_by_class": {
            str(k): v for k, v in om.group_by("opinion_class").len().iter_rows()
        },
        "n_rows_year_unknown": int(op["fiscal_year"].is_null().sum()),
        "ratio_rows_year_unknown": float(op["fiscal_year"].is_null().mean()),
        "ratio_rows_class_missing": float(op["opinion_class"].is_null().mean()),
    }

    div, unparsed = dividend_per_report(raw_root, raw_snapshot, return_unparsed=True)
    km = stock_knd_map(raw_root, raw_snapshot)
    files["stock_knd_map.tsv"] = km
    files["stock_knd_other_only_reports.tsv"] = other_only_reports(raw_root, raw_snapshot, div)
    files["dps_multi_value_reports.tsv"] = dps_multi_value_reports(raw_root, raw_snapshot, div)
    files["dps_nonnumeric_cells.tsv"] = pl.DataFrame(
        {"cell": list(unparsed.keys()), "n_cells": list(unparsed.values())},
        schema={"cell": pl.String, "n_cells": pl.Int64},
    ).sort("n_cells", descending=True)
    stats["stock_knd"] = {
        "n_distinct_values": km.height,
        "n_values_by_class": {str(k): v for k, v in km.group_by("knd_class").len().iter_rows()},
        "n_reports_with_dividend_table": div.height,
        "n_other_only_reports": files["stock_knd_other_only_reports.tsv"].height,
        "n_common_multi_value_reports": int(div["common_multi_value"].sum()),
        "n_unmarked_multi_value_reports": int(div["unmarked_multi_value"].sum()),
        "n_reports_by_knd_source": {
            str(k): v for k, v in div.group_by("knd_source").len().iter_rows()
        },
        "n_nonnumeric_cell_strings": len(unparsed),
    }

    types = capital_event_type_table(raw_root, raw_snapshot)
    ev, anomalies = capital_events_and_anomalies(raw_root, raw_snapshot)
    files["capital_event_types.tsv"] = types
    files["capital_event_date_anomalies.tsv"] = anomalies
    stats["capital_events"] = {
        "n_types": types.height,
        "n_events_after_dedupe": ev.height,
        "n_date_anomaly_rows_all_types": anomalies.height,
        "n_date_anomaly_rows_listed_types": int(anomalies["in_list"].sum()),
    }

    par_table, par_summary = par_change_check(raw_root, raw_snapshot)
    files["par_change_check.tsv"] = par_table
    stats["par_change_check"] = par_summary

    sey = share_event_years(raw_root, raw_snapshot)
    files["share_event_years.tsv"] = sey
    cap_keys = sey.filter(pl.col("source") == "capital_change").select("corp_code", "event_year")
    par_keys = sey.filter(pl.col("source") == "par_change").select("corp_code", "event_year")
    stats["share_event_years"] = {
        "n_corp_years_capital_change": cap_keys.unique().height,
        "n_corp_years_par_change": par_keys.unique().height,
        "n_corp_years_both": cap_keys.join(par_keys, on=["corp_code", "event_year"])
        .unique()
        .height,
        "n_corp_years_union": share_event_flags(
            raw_root, raw_snapshot, sources=RECORD_EVENT_SOURCES
        ).height,
        "n_par_change_corp_years_by_year": {
            str(k): v
            for k, v in par_keys.group_by("event_year").len().sort("event_year").iter_rows()
        },
    }

    hashes: dict[str, str] = {}
    for name, df in files.items():
        p = out_dir / name
        write_tsv(df, p)
        hashes[name] = sha256_file(p)

    inputs: dict[str, dict] = {}
    for tbl in ("dart_governance_raw", "dart_shareholder_return_raw", "dart_capital_change_raw"):
        fl = input_files(raw_root, raw_snapshot, tbl)
        listing = "\n".join(
            f"{f.relative_to(table_dir(raw_root, raw_snapshot, tbl))}:{f.stat().st_size}"
            for f in fl
        )
        inputs[tbl] = {
            "n_files": len(fl),
            "bytes": sum(f.stat().st_size for f in fl),
            "sha256_of_path_size_listing": hashlib.sha256(listing.encode()).hexdigest(),
        }
    manifest = {
        "rules_version": RULES_VERSION,
        "raw_snapshot": raw_snapshot,
        "module_file": Path(__file__).name,
        "module_sha256": sha256_file(Path(__file__)),
        "files_sha256": hashes,
        "inputs": inputs,
        "stats": stats,
        "ci_notes": CI_NOTES,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest


# 사전등록이 정하지 않은 구현 선택(작업 계획 §7 C 해석 표 후보). 코드의 ``# CI-<이름>`` 과 같다.
CI_NOTES = {
    "CI-g": "감사의견 같은 접수번호·같은 payload 는 한 행(row_ordinal 최소)",
    "CI-anchor": "제N기만 있는 라벨은 같은 접수번호의 `제M기(당기)` 가 하나로 정해질 때만 푼다",
    "CI-label-first": "제N기가 한 라벨에 여럿이면 첫 번째",
    "CI-opinion-notes": "주석 표시 꼴은 `(주N)`·`(*N)`·`(*)`·`(*N,*M)`·`(*주N)`·`(주)`·`주N)`",
    "CI-dps-dedupe": "배당 표는 (corp_code, rcept_no, raw_payload)로 중복 제거",
    "CI-dps-cell": "숫자 칸 `-`=0.0, 읽히지 않는 문자열=NaN(목록은 dps_nonnumeric_cells.tsv)",
    "CI-unmarked-multi": "미표기 행이 여럿이고 값이 다르면 칸별 최댓값",
    "CI-knd-none": "excluded 만 있는 보고서는 knd_source='none'",
    "CI-date": f"날짜 이상값: 읽히지 않음·연도<{DATE_YEAR_MIN}·연도>snapshot 연도",
    "CI-f": "사건 연도 = 날짜의 해",
    "CI-par": "액면 변경 해 = 보고서 사업연도, corp-year 에 하나라도 다르면 변경",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-snapshot", required=True, help="raw snapshot 날짜 YYYY-MM-DD")
    ap.add_argument("--raw-root", default=None, help="기본 <STOCK_DATA_ROOT>/kr/raw/raw_postgres")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root) if args.raw_root else default_raw_root()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else stock_data_root() / "kr" / "output" / f"quality_score_company_maps_{args.raw_snapshot}"
    )
    manifest = build_maps(raw_root, args.raw_snapshot, out_dir)
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2, default=str))
    print(f"출력: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
