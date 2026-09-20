"""F10 내부자 — ``insider_trans``(``filing_date``) · ``insider_owners``.

``04_feature_test_plan.md`` §3 F10. 세 피쳐: ``ins_netbuy_90`` ·
``ins_cluster_90`` · ``ins_officer_buy_90``.

**PIT의 축은 반드시 ``filing_date``다.** 거래일(``trans_date``)로 자르면
안 된다 — Form 5는 P99 지연이 304일이라, 거래가 실제로 일어난 뒤 304일이나
지나서야 공시되는 경우가 있다. ``trans_date``로 자르면 아직 공시되지 않은
(그래서 시장 참여자가 알 수 없던) 거래를 미리 아는 셈이 된다
(``04`` §3 F10, 이 계획 지시문 §2③). **Form 4만 쓰는 변형은 만들지 않는다**
— 결과를 보고 고르면 p-hacking이다.

``insider_owners``는 자기 날짜가 없다 — ``accession``으로 ``insider_trans``에
join해 그 표의 ``filing_date``를 빌려 쓴다. **join에 실패한 행(그 accession이
``insider_trans``에 없는 경우)은 버린다** — 이 파일은 항상 ``insider_trans``
쪽에서 시작해 ``insider_owners``를 붙이므로 이 규칙이 자동으로 지켜진다.

패널의 ``cik``로 ``insider_trans.issuer_cik``에 join한다 — **``issuer_symbol``을
키로 쓰면 안 된다**(CIK 1,955곳이 표기를 둘 이상 쓴다, 수집 계획 ``03`` §4.10).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from modeler.us.lake import UsLake

#: 90일 트레일링 윈도(달력일). ``04`` §3이 "90일"이라고만 적어 거래일이 아니라
#: 달력일로 읽었다 — F9의 ``days_since_earn``처럼 "거래일 수"라고 명시된
#: 자리만 거래일 단위다.
_WINDOW_DAYS = 90

#: 매수·매도 거래 코드 (수집 계획 ``03`` §4.10 실측 분포).
_BUY_CODE = "P"
_SELL_CODE = "S"


def _table_start(lake: UsLake) -> date:
    """``insider_trans``가 실제로 적재된 첫 ``filing_date``.

    ``01_data_readiness.md`` §4가 2018-07-02라고 적었지만, 매직 넘버를 코드에
    박지 않고 표에서 직접 구한다 — 표가 바뀌면(수집 계획에 요청해 2017
    분기를 더 받으면, §4가 예고한 대로) 이 함수가 자동으로 따라간다.
    """
    row = lake.scan("insider_trans").select(pl.col("filing_date").min().alias("d")).collect()
    return row.item()


def _first_filing_by_cik(lake: UsLake) -> pl.LazyFrame:
    """issuer_cik -> 그 발행사의 가장 이른 ``filing_date``.

    "이 회사가 시점 t까지 Section 16 공시를 한 번이라도 했는가"를 판정하는
    데 쓴다 — 외국발행사(Section 16 비대상)는 전체 이력에 한 번도 안
    나타나므로 이 값 자체가 null이다.
    """
    return (
        lake.scan("insider_trans")
        .group_by("issuer_cik")
        .agg(pl.col("filing_date").min().alias("_first_filing_date"))
    )


def _trades(lake: UsLake) -> pl.LazyFrame:
    """매수·매도(``P``·``S``) 거래 1건 1행 — accession · issuer_cik · filing_date ·
    거래금액(``_dollar`` = 주수 × 주당가) · 매수/매도 분리 컬럼."""
    return (
        lake.scan("insider_trans")
        .filter(
            pl.col("trans_code").is_in([_BUY_CODE, _SELL_CODE])
            & pl.col("trans_shares").is_not_null()
            & pl.col("trans_pricepershare").is_not_null()
        )
        .with_columns((pl.col("trans_shares") * pl.col("trans_pricepershare")).alias("_dollar"))
        .with_columns(
            pl.when(pl.col("trans_code") == _BUY_CODE)
            .then(pl.col("_dollar"))
            .otherwise(0.0)
            .alias("_buy_dollar"),
            pl.when(pl.col("trans_code") == _SELL_CODE)
            .then(pl.col("_dollar"))
            .otherwise(0.0)
            .alias("_sell_dollar"),
        )
        .select(
            "accession", "issuer_cik", "filing_date", "trans_code", "_buy_dollar", "_sell_dollar"
        )
    )


def _owner_role_flags(lake: UsLake) -> pl.LazyFrame:
    """accession -> 그 공시에 임원·이사가 한 명이라도 있는가.

    한 accession(공시)에 신고인이 최대 10명이라(수집 계획 ``03`` §4.10),
    공동 신고에서 아무나 임원·이사면 그 공시의 매수 건을
    ``ins_officer_buy_90``에 센다.
    """
    return (
        lake.scan("insider_owners")
        .group_by("accession")
        .agg(
            (
                pl.col("is_officer").fill_null(False).any()
                | pl.col("is_director").fill_null(False).any()
            ).alias("_officer_or_director")
        )
    )


def _owner_accession_pairs(lake: UsLake) -> pl.LazyFrame:
    """accession -> owner_cik. 공동 신고인 수를 세는 재료(``ins_cluster_90``)."""
    return lake.scan("insider_owners").select("accession", "owner_cik").unique()


def _window_sum_by_cik(
    panel_cik: pl.LazyFrame, events: pl.LazyFrame, *aggs: pl.Expr
) -> pl.LazyFrame:
    """(date, cik) 마다 ``events``의 ``[date-90일, date]`` 구간을 ``aggs``로 접는다.

    ``join_where``(부등식 join)로 윈도 안 행만 골라 붙인 뒤 그룹으로 접는다
    — 두 표를 통째로 cross join하는 대신, 매치되는 조합만 만든다.

    ``join_where``의 부등식 조건에 산술식(``date - 90일``)을 바로 넣으면
    polars가 "join key에 alias를 쓸 수 없다"는 에러를 낸다 — 조건은 반드시
    컬럼 대 컬럼이어야 해서, 윈도 시작일을 먼저 ``with_columns``로 만든
    컬럼(``_window_start``)으로 박아 둔다.
    """
    panel_with_window = panel_cik.with_columns(
        (pl.col("date") - pl.duration(days=_WINDOW_DAYS)).alias("_window_start")
    )
    matched = panel_with_window.join_where(
        events,
        pl.col("cik") == pl.col("issuer_cik"),
        pl.col("filing_date") <= pl.col("date"),
        pl.col("filing_date") > pl.col("_window_start"),
    )
    return matched.group_by(["date", "cik"]).agg(*aggs)


def add_insider(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F10(``ins_netbuy_90`` · ``ins_cluster_90`` · ``ins_officer_buy_90``)를 붙인다."""
    table_start = _table_start(lake)

    panel_cik = (
        panel.lazy()
        .select("date", "cik")
        .filter(pl.col("cik").is_not_null())
        .unique(subset=["date", "cik"])
    )

    trades = _trades(lake)
    # 매수·매도 금액을 한 번의 group_by로 같이 합친다(``ins_netbuy_90``의 분자·분모).
    buysell_agg = _window_sum_by_cik(
        panel_cik,
        trades,
        pl.col("_buy_dollar").sum().alias("_buy_sum"),
        pl.col("_sell_dollar").sum().alias("_sell_sum"),
    )

    buys = trades.filter(pl.col("trans_code") == _BUY_CODE).select(
        "accession", "issuer_cik", "filing_date"
    )

    owner_pairs = _owner_accession_pairs(lake)
    buy_owner_events = buys.join(owner_pairs, on="accession", how="inner")
    cluster_agg = _window_sum_by_cik(
        panel_cik,
        buy_owner_events,
        pl.col("owner_cik").n_unique().alias("ins_cluster_90"),
    )

    role_flags = _owner_role_flags(lake)
    officer_buys = buys.join(role_flags, on="accession", how="inner").filter(
        pl.col("_officer_or_director")
    )
    officer_agg = _window_sum_by_cik(
        panel_cik,
        officer_buys,
        pl.len().alias("ins_officer_buy_90"),
    )

    first_filing = _first_filing_by_cik(lake)

    base = (
        panel.lazy()
        .select("date", "symbol", "cik")
        .join(buysell_agg, on=["date", "cik"], how="left")
        .join(cluster_agg, on=["date", "cik"], how="left")
        .join(officer_agg, on=["date", "cik"], how="left")
        .join(first_filing, left_on="cik", right_on="issuer_cik", how="left")
        .with_columns(
            pl.col("_buy_sum").fill_null(0.0),
            pl.col("_sell_sum").fill_null(0.0),
            pl.col("ins_cluster_90").fill_null(0).cast(pl.Int64),
            pl.col("ins_officer_buy_90").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (
                pl.col("cik").is_not_null()
                & pl.col("_first_filing_date").is_not_null()
                & (pl.col("_first_filing_date") <= pl.col("date"))
            ).alias("_has_history"),
            ((pl.col("date") - pl.duration(days=_WINDOW_DAYS)) >= table_start).alias(
                "_window_full"
            ),
        )
        .with_columns((pl.col("_has_history") & pl.col("_window_full")).alias("_eligible"))
        .with_columns(
            pl.when(pl.col("_eligible") & ((pl.col("_buy_sum") + pl.col("_sell_sum")) > 0))
            .then(
                (pl.col("_buy_sum") - pl.col("_sell_sum"))
                / (pl.col("_buy_sum") + pl.col("_sell_sum"))
            )
            .otherwise(None)
            .alias("ins_netbuy_90"),
            pl.when(pl.col("_eligible"))
            .then(pl.col("ins_cluster_90"))
            .otherwise(None)
            .alias("ins_cluster_90"),
            pl.when(pl.col("_eligible"))
            .then(pl.col("ins_officer_buy_90"))
            .otherwise(None)
            .alias("ins_officer_buy_90"),
        )
        .with_columns(
            pl.col("ins_netbuy_90").is_null().alias("ins_netbuy_90_isna"),
            pl.col("ins_cluster_90").is_null().alias("ins_cluster_90_isna"),
            pl.col("ins_officer_buy_90").is_null().alias("ins_officer_buy_90_isna"),
        )
        .select(
            "date",
            "symbol",
            "ins_netbuy_90",
            "ins_netbuy_90_isna",
            "ins_cluster_90",
            "ins_cluster_90_isna",
            "ins_officer_buy_90",
            "ins_officer_buy_90_isna",
        )
    )

    result = panel.lazy().join(base, on=["date", "symbol"], how="left").collect()
    return result.sort(["date", "symbol"])
