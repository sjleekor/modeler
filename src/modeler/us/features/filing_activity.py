"""F12 공시 활동 — ``filings_index``.

``04_feature_test_plan.md`` §3 F12. 두 피쳐: ``n_8k_90`` · ``filing_lag``.

**PIT의 핵심(``07_risks.md`` Y4).** ``filings_index``의 as-of 축은
``acceptance_datetime``(UTC 초)이지 ``filing_date``(달력일)가 아니다. 10-Q의
46.7%가 ET 16:00 마감 뒤에 접수된다 — ``filing_date``만 보면 당일 종가에
아직 못 쓸 정보를 쓰게 된다. 그래서 두 피쳐 다 "이 공시를 언제부터 알 수
있는가"를 ``features._trading_days.filings_effective_date``(ET 변환 → 16:00
규칙 → 다음 거래일 올림)로 정하고, 그 유효일이 ``t`` 이하인 것만 쓴다.

``filing_lag``(``filed - period`` 일수) 자체의 **값**은 ``filing_date``(달력일)
``report_date``(그 공시가 보고하는 기간의 끝)를 그대로 뺀 것이다 — 여기는
그 공시가 "얼마나 늦게 냈는가"를 재는 서술 통계라 시점 판정과 무관하다.
어떤 공시를 "최근 것"으로 골라 그 값을 쓸지만 유효일 규칙을 따른다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._trading_days import filings_effective_date
from modeler.us.lake import UsLake

#: 90일 트레일링 윈도(달력일) — ``ins_netbuy_90``과 같은 관례.
_WINDOW_DAYS = 90

#: ``filing_lag``이 보는 form. ``04`` §3이 "10-K/10-Q"라고만 적어 정정본
#: (``/A``)은 포함하지 않는다 — F5(재무)의 "정정본을 남긴다" 규칙과 다른
#: family다.
_LAG_FORMS = ("10-K", "10-Q")


def _filings_with_effective_date(lake: UsLake) -> pl.LazyFrame:
    return filings_effective_date(
        lake,
        lake.scan("filings_index").select(
            "cik", "form", "filing_date", "report_date", "acceptance_datetime"
        ),
        "acceptance_datetime",
        out_col="effective_date",
    )


def _n_8k_90(panel: pl.DataFrame, lake: UsLake) -> pl.LazyFrame:
    # join_where 술어에서 이름이 같은 컬럼끼리 비교하면 어느 쪽인지 모호해진다
    # (insider.py의 issuer_cik처럼) — 오른쪽 것을 다른 이름으로 미리 바꾼다.
    eightk = (
        _filings_with_effective_date(lake)
        .filter(pl.col("form") == "8-K")
        .select(pl.col("cik").alias("_filing_cik"), "effective_date")
    )

    panel_cik = (
        panel.lazy()
        .select("date", "cik")
        .filter(pl.col("cik").is_not_null())
        .unique(subset=["date", "cik"])
        .with_columns((pl.col("date") - pl.duration(days=_WINDOW_DAYS)).alias("_window_start"))
    )

    matched = panel_cik.join_where(
        eightk,
        pl.col("cik") == pl.col("_filing_cik"),
        pl.col("effective_date") <= pl.col("date"),
        pl.col("effective_date") > pl.col("_window_start"),
    )
    agg = matched.group_by(["date", "cik"]).agg(pl.len().alias("n_8k_90"))

    return (
        panel.lazy()
        .select("date", "symbol", "cik")
        .join(agg, on=["date", "cik"], how="left")
        .with_columns(
            pl.when(pl.col("cik").is_not_null())
            .then(pl.col("n_8k_90").fill_null(0))
            .otherwise(None)
            .alias("n_8k_90")
        )
        .select("date", "symbol", "n_8k_90")
    )


def _filing_lag(panel: pl.DataFrame, lake: UsLake) -> pl.LazyFrame:
    lag_filings = (
        _filings_with_effective_date(lake)
        .filter(pl.col("form").is_in(_LAG_FORMS) & pl.col("report_date").is_not_null())
        .with_columns(
            (pl.col("filing_date") - pl.col("report_date")).dt.total_days().alias("filing_lag")
        )
        .select("cik", "effective_date", "filing_lag")
        .sort(["cik", "effective_date"])
    )

    panel_cik = (
        panel.lazy()
        .select("date", "symbol", "cik")
        .filter(pl.col("cik").is_not_null())
        .sort(["cik", "date"])
    )
    matched = panel_cik.join_asof(
        lag_filings,
        left_on="date",
        right_on="effective_date",
        by="cik",
        strategy="backward",
    )
    without_cik = (
        panel.lazy()
        .select("date", "symbol", "cik")
        .filter(pl.col("cik").is_null())
        .with_columns(pl.lit(None, dtype=pl.Int64).alias("filing_lag"))
    )
    return pl.concat(
        [
            matched.select("date", "symbol", "filing_lag"),
            without_cik.select("date", "symbol", "filing_lag"),
        ],
        how="vertical_relaxed",
    )


def add_filing_activity(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F12(``n_8k_90`` · ``filing_lag``)를 붙인다."""
    n_8k = _n_8k_90(panel, lake)
    lag = _filing_lag(panel, lake)

    result = (
        panel.lazy()
        .join(n_8k, on=["date", "symbol"], how="left")
        .join(lag, on=["date", "symbol"], how="left")
        .with_columns(
            pl.col("n_8k_90").is_null().alias("n_8k_90_isna"),
            pl.col("filing_lag").is_null().alias("filing_lag_isna"),
        )
        .collect()
    )
    return result.sort(["date", "symbol"])
