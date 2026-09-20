"""미국 18표 parquet 레이크 리더.

레이크 구조 (2026-09-20 실측, ``01_data_readiness.md`` §1)::

    $STOCK_DATA_ROOT/us/derived/snapshots/<table>/snapshot_date=YYYY-MM-DD/*.parquet

표마다 스냅샷이 하나씩 있다 (2026-09-18 또는 2026-09-19). ``scan_raw()``는
원본 그대로를, ``scan()``은 ``01_data_readiness.md`` §6 정제 규칙을 적용한
것을 준다 — **모델 코드는 ``scan()``만 쓴다.**

이 모듈은 parquet를 직접 읽는다. ``collector.us.*``는 import하지 않는다
(``pyproject.toml``의 banned-api가 ``collector.us.sources|store|cli|paths``를
막는다). ``DataRoot``만 ``collector.lake``에서 재노출한
``modeler.etl.config``를 통해 가져온다 — 저장소 관례다.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot

#: 18표. 알파벳 순 — ``01_data_readiness.md`` §1 실측 그대로다.
US_TABLES: tuple[str, ...] = (
    "company_meta",
    "corp_actions",
    "earnings_calendar",
    "filings_index",
    "filings_sub",
    "fundamentals",
    "index_constituents",
    "insider_owners",
    "insider_trans",
    "listing_snapshots",
    "macro_series",
    "midas_security_daily",
    "prices_daily",
    "short_interest",
    "short_volume",
    "trading_calendar",
    "universe_daily",
    "volatility_daily",
)

#: 표 -> as-of 축 컬럼명 (``01_data_readiness.md`` §2 표 그대로).
#:
#: 모델 코드는 이 컬럼 하나만 보고 시점을 잘라야 한다 — 다른 날짜 컬럼으로
#: 자르면 미래를 본다. 축이 ``None``인 표 셋은 이유가 각각 다르다:
#:
#: - ``company_meta``: CIK 현재값 스냅샷 1컷이다. 과거 시점 업종·필터에 쓰면
#:   안 된다 (현재 필터에만 쓴다).
#: - ``insider_owners``: 자기 날짜 컬럼이 없다. ``accession``으로
#:   ``insider_trans``에 join해 그 표의 ``filing_date``를 빌려 쓴다.
#: - ``trading_calendar``: (date, exchange) 자체가 거래일 참조표라 as-of
#:   개념이 없다.
ASOF_AXIS: dict[str, str | None] = {
    "company_meta": None,
    "corp_actions": "ex_date",
    "earnings_calendar": "date",
    "filings_index": "acceptance_datetime",
    "filings_sub": "filed",
    "fundamentals": "filed",
    "index_constituents": "as_of",
    "insider_owners": None,
    "insider_trans": "filing_date",
    "listing_snapshots": "as_of",
    "macro_series": "realtime_start",
    "midas_security_daily": "date",
    "prices_daily": "date",
    "short_interest": "settlement_date",
    "short_volume": "date",
    "trading_calendar": None,
    "universe_daily": "date",
    "volatility_daily": "date",
}

#: 재무 피쳐가 쓰는 form. 정정본을 같이 남기고 ``filed <= t`` 최신 규칙(M2)이
#: 고르게 한다 — ``_clean_fundamentals`` 주석 참고.
FUNDAMENTAL_FORMS: tuple[str, ...] = ("10-K", "10-Q", "10-K/A", "10-Q/A")

#: as-of 축에 **더해야 하는** 공표 지연 (거래일).
#:
#: ``ASOF_AXIS``의 축 값만 보고 ``축 <= t``로 자르면 미래를 본다 — 축이 사건이
#: 일어난 날이고 그 값이 공표된 날이 아니기 때문이다. ``축 + 지연 <= t``여야
#: t 시점에 실제로 알 수 있던 정보다.
#:
#: ``short_interest``: FINRA가 **결제일 + 7영업일**에 공표하고(원문 확인
#: 2026-09-20), 나스닥이 그날 16:00 ET 뒤 배포한다. 종가 기준 판정이라
#: 8거래일째부터 쓸 수 있고, 영업일과 거래일이 어긋나는 경우까지 덮으려고
#: 10으로 잡았다 (``07_risks.md`` Y10). **표에 공표일 컬럼이 없어 이 상수가
#: 유일한 방어다.** 규정이 바뀔 수 있으니 상수를 여기 한 곳에 둔다.
ASOF_LAG_TRADING_DAYS: dict[str, int] = {
    "short_interest": 10,
}

_SNAPSHOT_DIR_RE = re.compile(r"^snapshot_date=(\d{4}-\d{2}-\d{2})$")
_EASTERN_TZ = "America/New_York"


def _require_known_table(table: str) -> None:
    if table not in US_TABLES:
        raise KeyError(f"모르는 미국 표입니다: {table!r}. US_TABLES 18개 중 하나여야 합니다.")


def acceptance_datetime_to_et(expr: pl.Expr) -> pl.Expr:
    """``filings_index.acceptance_datetime``(UTC)을 미국 동부시간(ET)으로 바꾼다.

    10-Q의 46.7%가 ET 16:00 마감 뒤에 접수된다 (``01_data_readiness.md`` §2).
    날짜만 보고 자르면 다음 거래일로 밀어야 할 공시를 당일 것으로 잘못 본다.
    서머타임(EDT/EST) 전환은 시간대 변환이 달력에 맞춰 처리한다.

    "16:00 뒤면 다음 거래일" 판정 자체는 여기서 하지 않는다 — 규칙이 이 표에
    매여 있어 헬퍼는 ``lake.py``에 두지만, 실제 적용(``trading_calendar``와
    맞춰 다음 거래일을 정하는 일)은 M2(피쳐) 몫이다.
    """
    return expr.dt.convert_time_zone(_EASTERN_TZ)


def _clean_fundamentals(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        # 원천 파싱 오류로 보이는 미래 end 값. 2026-09-20 duckdb 실측 502행
        # (``01_data_readiness.md`` §6) — 수집 계획에는 없던 발견이다.
        (pl.col("end") <= date(2030, 12, 31))
        # 국내 정기 보고서와 그 정정본만 쓴다. 20-F·40-F(외국발행사)는 연 1회라
        # PIT 지연이 다르므로 1차에서는 뺀다 — 그 종목은 재무 피쳐가 결측이 되고
        # ``_isna``가 곧 "외국발행사" 플래그가 된다 (§3).
        #
        # **정정본(/A)을 남기는 이유.** ``01_data_readiness.md`` §6은 "정정본은
        # filed 규칙으로 자연히 덮인다"고 적었는데, 정정본을 필터에서 빼면 덮을
        # 행 자체가 없어 그 문장이 성립하지 않는다. 2026-09-20 실측으로
        # **(cik, end, tag) 조합 527,329개가 정정본에만 있었다** — 원본이 그 값을
        # 낸 적이 없다. 그리고 t 시점에 정정본이 이미 공시돼 있었다면 그것이
        # 그때 알 수 있던 값이므로, 남겨 두고 ``filed <= t`` 최신 규칙(M2)이
        # 고르게 하는 쪽이 PIT에 맞다.
        & pl.col("form").is_in(FUNDAMENTAL_FORMS)
    )


def _clean_midas_security_daily(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ETF는 quartile 눈금이라 종목 rank와 섞을 수 없다 (§6). rank 넷은 원천에서
    # 이미 int32라 따로 캐스팅하지 않는다.
    return lf.filter(pl.col("security_type") == "Stock")


def _clean_macro_series(lf: pl.LazyFrame) -> pl.LazyFrame:
    # SP500 계열은 vintage(realtime_start)가 없어 PIT를 보장 못한다. 지수는
    # prices_daily의 SPY 가격으로 대신한다 (§2, §6).
    return lf.filter(pl.col("series_id") != "SP500")


def _clean_short_interest(lf: pl.LazyFrame) -> pl.LazyFrame:
    # 같은 settlement_date·symbol에 정정 행이 있으면 그것을 쓴다.
    # ``revision_flag``는 계획 문서 표기(``'R'``)와 달리 **Boolean**이다
    # (2026-09-20 실측: False 3,659,633 / True 24,753).
    # 그것을 쓴다 (수집 계획 03 §4.4, ``01_data_readiness.md`` §6). 정정이
    # 없는 조합은 원래 행이 그대로 남는다. revision_flag를 내림차순으로 정렬해
    # 정정 행을 그룹 맨 앞에 두고 첫 행만 남긴다.
    return lf.sort(
        ["settlement_date", "symbol", "revision_flag"],
        descending=[False, False, True],
    ).unique(subset=["settlement_date", "symbol"], keep="first", maintain_order=True)


#: 표 -> 정제 함수. 없는 표는 ``scan()``이 ``scan_raw()``와 같은 것을 준다.
_CLEANERS: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    "fundamentals": _clean_fundamentals,
    "midas_security_daily": _clean_midas_security_daily,
    "macro_series": _clean_macro_series,
    "short_interest": _clean_short_interest,
}


@dataclass(frozen=True)
class UsLake:
    """미국 레이크 리더 하나. ``root``가 가리키는 ``$STOCK_DATA_ROOT/us``를 읽는다."""

    root: DataRoot

    @classmethod
    def resolve(cls) -> UsLake:
        """``$STOCK_DATA_ROOT/us``를 가리키는 ``UsLake``.

        환경변수는 ``DataRoot.resolve()`` 안에서만 읽는다 — 모듈 최상위에서
        읽으면 ``STOCK_DATA_ROOT``가 없는 채로 ``import modeler.us``만 해도
        죽는다 (한국에서 이 문제로 전체가 죽은 적이 있다).
        """
        return cls(root=DataRoot.resolve(market="us"))

    def _table_dir(self, table: str) -> Path:
        _require_known_table(table)
        return self.root.derived / "snapshots" / table

    def latest_snapshot(self, table: str) -> date:
        """``table`` 아래 ``snapshot_date=`` 디렉터리 중 가장 최신 날짜.

        표 디렉터리 자체가 없거나 ``snapshot_date=`` 형식 하위 디렉터리가
        하나도 없으면 ``FileNotFoundError``를 던진다.
        """
        table_dir = self._table_dir(table)
        if not table_dir.is_dir():
            raise FileNotFoundError(f"{table} 표의 스냅샷 디렉터리가 없습니다: {table_dir}")
        dates: list[date] = []
        for child in table_dir.iterdir():
            if not child.is_dir():
                continue
            match = _SNAPSHOT_DIR_RE.match(child.name)
            if match:
                dates.append(date.fromisoformat(match.group(1)))
        if not dates:
            raise FileNotFoundError(
                f"{table} 아래 snapshot_date=YYYY-MM-DD 디렉터리가 없습니다: {table_dir}"
            )
        return max(dates)

    def snapshot_dir(self, table: str, snapshot_date: date | None = None) -> Path:
        """``table``의 스냅샷 디렉터리. ``snapshot_date``를 안 주면 최신을 쓴다.

        경로만 조립한다 — 디스크에 실제로 있는지는 보지 않는다. 존재 확인은
        ``scan_raw()``가 parquet 파일을 찾을 때 한다.
        """
        resolved = snapshot_date if snapshot_date is not None else self.latest_snapshot(table)
        return self._table_dir(table) / f"snapshot_date={resolved.isoformat()}"

    def scan_raw(self, table: str, snapshot_date: date | None = None) -> pl.LazyFrame:
        """정제 규칙을 적용하지 않은 원본 ``LazyFrame``."""
        directory = self.snapshot_dir(table, snapshot_date)
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{table}의 parquet 파일이 없습니다: {directory}")
        return pl.scan_parquet(files)

    def scan(self, table: str, snapshot_date: date | None = None) -> pl.LazyFrame:
        """정제 규칙(``01_data_readiness.md`` §6)을 적용한 ``LazyFrame``.

        모델 코드는 이것만 쓴다. ``filings_index``의 16:00 ET 규칙,
        ``Revenues`` 합치기, ``GrossProfit`` 폴백은 여기 넣지 않는다 — "이
        표를 읽을 때 항상 참인 것"만 여기서 걸고, 나머지는 M2(피쳐)의 일이다.
        """
        lf = self.scan_raw(table, snapshot_date)
        cleaner = _CLEANERS.get(table)
        if cleaner is not None:
            lf = cleaner(lf)
        return lf

    def snapshot_manifest(self) -> dict[str, str]:
        """표 이름 -> 쓴 snapshot_date(ISO). 데이터셋 manifest에 그대로 들어간다."""
        return {table: self.latest_snapshot(table).isoformat() for table in US_TABLES}
