"""F19 기관보유(13F, 조건부) [``inst_holdings_q``] — ``inst_n_log`` ·
``inst_breadth_chg`` · ``inst_shares_chg``.

``us4_flow_features/00_draft.md`` §5.3. 정의::

    inst_n_log       = log(1 + n_holders), 최근 사용 가능한 분기
    inst_breadth_chg = (n_holders - 직전 분기 n_holders) / 그 분기 전체 13F-HR filer 수
    inst_shares_chg  = (shares_total - 직전 분기 shares_total) / 20일 평균 거래량(주식 수)

**조건부 family다(U-D8).** 10/11까지 서버에 ``inst_holdings_q`` 표가 굳지
않으면 사전등록에서 이 family를 "미검정"으로 남긴다 — 이 모듈은 표가 없으면
(``inst_holdings_q``·``thirteenf_submissions``·``cusip_symbol_pit`` 중 하나라도)
**조용히 건너뛰지 않고 그대로 예외를 던진다.** ``UsLake.scan_raw``가 스냅샷
디렉터리를 못 찾으면 ``FileNotFoundError``를 던지는 동작을 그대로 쓴다 — 이
모듈이 따로 잡지 않는다. ``build_flow_features.py``도 마찬가지로 잡지 않는다.

**원천은 이미 정정·필터를 끝낸 집계표다.** ``collector``의
``sources/sec_13f.extract_13f``가 ``inst_holdings_q``를 만들 때 이미:

* ``13F-NT``/``13F-NT/A``를 뺐고
* ``13F-HR/A``(정정)로 같은 ``(filer_cik, period_of_report)``의 앞 제출을 대체했고
* ``SSHPRNAMTTYPE = 'SH'`` ∧ ``PUTCALL`` 빈 값만 남겼고(옵션 보유 제외)
* ``OTHERMANAGER``가 찬 공동 보유의 주식 수 중복을 걷어냈다

그래서 이 모듈은 ``thirteenf_submissions``를 직접 읽지 않는다 — ``inst_holdings_q``
한 표(``cusip, period_of_report, n_holders, shares_total,
n_filers_total_that_period``)만으로 셋 다 계산할 수 있다. ``thirteenf_submissions``는
``lake.py``에 표로는 등록해 두되(§7 지시), 이 모듈이 쓰는 것은 ``inst_holdings_q``뿐이다.

**CUSIP → 심볼.** ``inst_holdings_q``의 키는 CUSIP이라 패널의 ``symbol``로
바꿔야 한다. ``cusip_symbol_pit``(``ftd_fails``로 만든 다리, 01_sec_ftd.md §6)에서
``first_seen <= period_of_report <= last_seen``인 쌍을 고른다 — 여럿이면
``n_settlement_dates``가 큰 쌍 하나만 남긴다(초안 §5.3). CUSIP 쪽에서 고른 뒤
심볼 쪽에서도 한 번 더 같은 규칙으로 중복을 접는다 — 서로 다른 CUSIP이 같은
분기에 우연히 같은 심볼로 풀리는 경우(합병·재상장)를 방어하는 안전장치다(스펙에
없는 임의 결정 — 데이터로 확인되면 바뀔 수 있다).

**"직전 분기"는 심볼이 아니라 CUSIP 기준이다(초안 §5.3).** CUSIP이 바뀌면(합병
등) 연속성을 보장할 수 없으므로, ``_prev_*``·연속 분기 판정은 심볼로 바꾸기
**전에** ``inst_holdings_q``를 CUSIP으로 정렬해 계산한다. "분기 하나 건너뛰면
null"은 ``period_of_report``의 연·월 차이가 정확히 3개월(같은 분기 주기)인지로
판정한다 — 13F의 ``period_of_report``가 분기말 날짜라 이 방식이 윤년·연말 경계에서도
정확하다.

**지연.** 사용 가능일 = ``period_of_report`` + ``LAG_13F_DAYS``(달력일). 법정
마감이 45일이라 **60일**을 임시값으로 둔다 — 초안 §5.4가 "미측정"이라 적어 둔
그대로다. 서버에서 ``FILING_DATE`` 분포를 재면(U-D8 데드라인 전) 이 상수를
바꾼다. ``ftd.py``·``order_flow.py``처럼 ``join_asof``로 시점을 지킨다.

**``inst_shares_chg``의 분모.** "20일 평균 거래량 주식 수"를 문자 그대로
``prices_daily.volume``(조정 전, F17과 같은 단위 관례)의 20거래일 **평균**으로
계산한다 — F17의 비율(20일 합/20일 합)과 달리 이건 그 자체로 평균이다. 창은
``period_of_report``(공시 시점이 아니라 측정 대상 분기 자체) 기준으로 잡는다 —
F17이 ``settlement_date`` 창을 쓰지 ``available_date`` 창을 쓰지 않는 것과 같은
이유다: 보유 변화의 크기를 그 분기 당시의 유동성으로 정규화하는 것이지, 60일
뒤 알게 된 시점의 유동성으로 정규화하는 게 아니다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._daily import panel_symbols
from modeler.us.lake import UsLake

#: ``period_of_report`` + 이 값(달력일)부터 그 분기 13F 보유 표를 알 수 있다.
#: 초안 §5.3·§5.4 — 법정 45일, 임시값(미측정 · U-D8 데드라인 전에 실측 예정).
LAG_13F_DAYS = 60

#: ``inst_shares_chg`` 분모(20거래일 평균 거래량 주식 수)의 창·최소 유효 일수 —
#: §5 인트로 일반 규칙("창 안 유효 일수가 10 미만이면 null")과 같은 상수.
_VOLUME_WINDOW = 20
_VOLUME_MIN_VALID_DAYS = 10

_FEATURES = ("inst_n_log", "inst_breadth_chg", "inst_shares_chg")


def _quarters_apart(period_col: pl.Expr, prev_period_col: pl.Expr) -> pl.Expr:
    """두 분기말 날짜의 연·월 차이가 정확히 3개월(=한 분기)인가.

    일수 차이(꽉 찬 91~92일)로 재면 윤년·월 길이 차이로 흔들린다 — 연·월을
    ``year*12+month``로 펴서 빼면 분기말 날짜끼리는 항상 정수 3이 나온다.
    ``prev_period_col``이 null이면(그 CUSIP의 첫 관측) 비교 결과도 null이 되고,
    호출자가 ``fill_null(False)``로 "연속 아님"으로 접는다.
    """
    months = period_col.dt.year() * 12 + period_col.dt.month()
    prev_months = prev_period_col.dt.year() * 12 + prev_period_col.dt.month()
    return (months - prev_months) == 3


def _holdings_with_quarter_over_quarter(
    lake: UsLake, relevant_cusips: pl.LazyFrame
) -> pl.LazyFrame:
    """``inst_holdings_q``를 CUSIP별로 정렬해 ``inst_n_log``·직전 분기 대비 변화를 낸다.

    "직전 분기"는 심볼이 아니라 **CUSIP** 기준이다(모듈독스트링 참고) — 그래서
    심볼로 바꾸기 전에 이 단계에서 전부 계산해 둔다.
    """
    holdings = (
        lake.scan("inst_holdings_q")
        .join(relevant_cusips, on="cusip", how="inner")
        .select(
            "cusip",
            "period_of_report",
            pl.col("n_holders").cast(pl.Float64),
            pl.col("shares_total").cast(pl.Float64),
            pl.col("n_filers_total_that_period").cast(pl.Float64),
        )
        .sort(["cusip", "period_of_report"])
    )

    prev_period = pl.col("period_of_report").shift(1).over("cusip")
    prev_n_holders = pl.col("n_holders").shift(1).over("cusip")
    prev_shares_total = pl.col("shares_total").shift(1).over("cusip")
    is_consecutive_quarter = _quarters_apart(pl.col("period_of_report"), prev_period).fill_null(
        False
    )

    return holdings.with_columns(
        (pl.col("n_holders") + 1.0).log().alias("inst_n_log"),
        pl.when(is_consecutive_quarter & (pl.col("n_filers_total_that_period") > 0))
        .then((pl.col("n_holders") - prev_n_holders) / pl.col("n_filers_total_that_period"))
        .otherwise(None)
        .alias("inst_breadth_chg"),
        pl.when(is_consecutive_quarter)
        .then(pl.col("shares_total") - prev_shares_total)
        .otherwise(None)
        .alias("_shares_diff"),
    ).select("cusip", "period_of_report", "inst_n_log", "inst_breadth_chg", "_shares_diff")


def _resolve_symbol(lake: UsLake, symbols: list[str]) -> pl.LazyFrame:
    """``cusip_symbol_pit``에서 패널 종목만 남긴다. 범위 필터·선택은 호출자가 한다."""
    return (
        lake.scan("cusip_symbol_pit")
        .filter(pl.col("symbol").is_in(symbols))
        .select("cusip", "symbol", "first_seen", "last_seen", "n_settlement_dates")
    )


def _pick_best_pair(frame: pl.LazyFrame, *, group_keys: list[str], tie_break: str) -> pl.LazyFrame:
    """``group_keys``마다 ``tie_break``(예: ``n_settlement_dates``) 내림차순 첫 행만 남긴다."""
    return frame.sort(
        [*group_keys, tie_break], descending=[*([False] * len(group_keys)), True]
    ).unique(subset=group_keys, keep="first", maintain_order=True)


def add_institutional(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F19 기관보유 피쳐 + ``_isna``를 붙인다.

    ``inst_holdings_q``·``cusip_symbol_pit``·``prices_daily`` 중 하나라도
    레이크에 없으면 ``FileNotFoundError``가 그대로 올라온다(``UsLake.scan_raw``
    출처) — 이 함수는 잡지 않는다.
    """
    symbols = panel_symbols(panel)

    cusip_map = _resolve_symbol(lake, symbols)
    relevant_cusips = cusip_map.select("cusip").unique()

    holdings = _holdings_with_quarter_over_quarter(lake, relevant_cusips)

    # CUSIP -> 심볼: first_seen <= period_of_report <= last_seen 인 쌍만, 여럿이면
    # n_settlement_dates 최댓값 하나. 그다음 심볼 쪽에서도 한 번 더 같은 규칙으로
    # 접는다(서로 다른 CUSIP이 같은 분기·같은 심볼로 풀리는 경우의 방어, 모듈독스트링).
    with_symbol = holdings.join(cusip_map, on="cusip", how="inner").filter(
        (pl.col("first_seen") <= pl.col("period_of_report"))
        & (pl.col("period_of_report") <= pl.col("last_seen"))
    )
    with_symbol = _pick_best_pair(
        with_symbol, group_keys=["cusip", "period_of_report"], tie_break="n_settlement_dates"
    )
    with_symbol = _pick_best_pair(
        with_symbol, group_keys=["symbol", "period_of_report"], tie_break="n_settlement_dates"
    )

    raw_volume = (
        lake.scan("prices_daily")
        .filter(pl.col("symbol").is_in(symbols))
        .select("date", "symbol", pl.col("volume").cast(pl.Float64))
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("volume")
            .rolling_mean(window_size=_VOLUME_WINDOW, min_samples=_VOLUME_MIN_VALID_DAYS)
            .over("symbol")
            .alias("avg_volume_20")
        )
        .select("symbol", "date", "avg_volume_20")
    )

    with_volume = with_symbol.sort(["symbol", "period_of_report"]).join_asof(
        raw_volume.sort(["symbol", "date"]),
        left_on="period_of_report",
        right_on="date",
        by="symbol",
        strategy="backward",
    )

    quarterly = with_volume.with_columns(
        pl.when(pl.col("avg_volume_20") > 0)
        .then(pl.col("_shares_diff") / pl.col("avg_volume_20"))
        .otherwise(None)
        .alias("inst_shares_chg"),
        (pl.col("period_of_report") + pl.duration(days=LAG_13F_DAYS)).alias("available_date"),
    ).select("symbol", "period_of_report", "available_date", *_FEATURES)

    panel_lf = panel.lazy().sort(["symbol", "date"])
    joined = panel_lf.join_asof(
        quarterly.sort(["symbol", "available_date"]),
        left_on="date",
        right_on="available_date",
        by="symbol",
        strategy="backward",
    )

    isna_flags = [pl.col(c).is_null().alias(f"{c}_isna") for c in _FEATURES]
    result = joined.with_columns(isna_flags)

    keep = [*panel.columns]
    for c in _FEATURES:
        keep.extend([c, f"{c}_isna"])
    return result.select(keep).sort(["date", "symbol"]).collect()
