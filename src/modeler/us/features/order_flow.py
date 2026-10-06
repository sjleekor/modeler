"""F18 주문흐름 [``midas_security_daily`` 12칸] — ``cancel_ratio_20`` ·
``hidden_share_20`` · ``oddlot_share_20`` · ``fill_ratio_20``.

``us4_flow_features/00_draft.md`` §5.2. 정의(전부 20거래일 창 합의 비율)::

    cancel_ratio_20  = Σ20 cancels / Σ20 lit_trades
    hidden_share_20  = Σ20 hidden_vol_k / Σ20 trade_vol_for_hidden_k
    oddlot_share_20  = Σ20 odd_lot_vol_k / Σ20 trade_vol_for_odd_lots_k
    fill_ratio_20    = Σ20 lit_vol_k / Σ20 order_vol_k

부호는 넷 다 미등록(U-Q3) — 이 모듈은 값만 낸다. ``security_type = 'Stock'``만
쓰는데, ``lake.scan("midas_security_daily")``가 ``_clean_midas_security_daily``에서
이미 그 필터를 걸어 준다 — 여기서 다시 거르지 않는다.

**PIT — 사용 가능일 (``liquidity.py``의 실수를 반복하지 않는다).**
``liquidity.py``의 ``turnover_rank``는 ``midas_security_daily``를 ``date``로
그대로 조인한다 — MIDAS는 **분기가 끝나야 나오는 파일**이라 이 조인은 t
시점에 아직 모르는 그 분기 값을 쓰는 룩어헤드다(초안 §2.1). 이 모듈은
``short.py``의 선례(발행 지연 상수 + ``join_asof``)를 따라, ``date``가 속한
**분기 끝 + 사용 가능해지기까지의 달력일**부터 그 값을 쓴다.

지연은 상수 하나가 아니라 **분기별 실측 표**(``MIDAS_AVAILABLE_FROM``)다 —
``02_lag_constants.md`` §3(2026-09-28 sj2-server ``curl -I`` 실측, MIDAS
32개 zip)이 분기마다 발행 시각 편차가 너무 커서(2021\\~2024는 21\\~59일,
2025년 이후는 128\\~289일) 상수 하나로 못 정한다고 결론 냈다. 표에 없는
분기(2018q3\\~2020q3 — SEC 사이트 이전으로 ``Last-Modified``를 못 잼·2022q4 —
빈 값)는 ``MIDAS_FALLBACK_LAG_DAYS``(분기 끝 + 60일, 측정된 2020q4\\~2024q4
체제의 최댓값 59일을 덮는 가정)로 대신한다.

**창 유효 일수.** §5 인트로 일반 규칙 — 20거래일 창 안에 값 있는 날이 10
미만이면 null. ``momentum.py``·``liquidity.py``와 같은 이유로
``_daily.mask_ticker_reuse_gap``도 건다 — MIDAS 이력이 2018-07\\~2026-06으로
짧아 위험은 낮지만, 같은 티커가 몇 해 뒤 재사용된 경우를 방어하는 것은
이 저장소의 일관된 관례다.

**커버리지.** MIDAS는 REIT·SPAC·ADR·CEF/BDC·LP/MLP 다섯 부류를 안 싣는다 —
그 부류의 유니버스 종목은 넷 다 ``_isna``가 항상 True다(초안 §5.2,
[07_midas_unused_columns.md](../../../../../my/milestones/us/research/data/source_expansion/07_midas_unused_columns.md)
§8.7). 미래 정보는 아니지만 이 다섯 부류의 지시변수가 된다는 점을 기록해 둔다.

**종목 구간 모드** (``lake.security_boundaries``, 유니버스 v2 설계 §3). MIDAS 행에
``(symbol, date)``로 ``security_id``를 붙여 20거래일 합을 구간 안에서만 굴리고, 패널과도
구간 단위로 ``join_asof``한다. ``min_samples``는 꺼짐과 같고(``_MIN_VALID_DAYS``), 대신 구간 안
행 위치(0부터)가 19 미만인 첫 19행은 비운다(``segments.mask_warmup``). 창 안에 빈 값이 있어도
위치가 19 이상이면 값이 나온다.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from modeler.us.features._daily import mask_ticker_reuse_gap, panel_symbols
from modeler.us.lake import UsLake
from modeler.us.segments import attach_security_id, group_key, mask_warmup

#: 창(거래일). 창 안 유효 일수가 이 값 미만이면 null — §5 인트로 일반 규칙.
_WINDOW = 20
_MIN_VALID_DAYS = 10

#: 표에 없는 분기의 기본 지연(달력일) — ``02_lag_constants.md`` §1·§3.2.
MIDAS_FALLBACK_LAG_DAYS = 60

#: 분기별 실측 사용 가능일 — (year, quarter) -> 그 분기 zip을 실제로 받을 수
#: 있게 된 날. ``02_lag_constants.md`` §3.2 표 그대로다(``Last-Modified`` 실측 +
#: 2025년 세 분기는 Wayback으로 첫 등장을 확인해 보정). 2018q3~2020q3(사이트
#: 이전으로 ``Last-Modified``가 전부 2020-12-19로 덮임)·2022q4(빈 값)는 표에
#: 없다 — ``MIDAS_FALLBACK_LAG_DAYS``가 대신한다.
MIDAS_AVAILABLE_FROM: dict[tuple[int, int], date] = {
    (2020, 4): date(2021, 1, 28),
    (2021, 1): date(2021, 4, 28),
    (2021, 2): date(2021, 8, 13),
    (2021, 3): date(2021, 10, 25),
    (2021, 4): date(2022, 1, 27),
    (2022, 1): date(2022, 5, 3),
    (2022, 2): date(2022, 7, 27),
    (2022, 3): date(2022, 10, 21),
    (2023, 1): date(2023, 4, 28),
    (2023, 2): date(2023, 8, 16),
    (2023, 3): date(2023, 10, 27),
    (2023, 4): date(2024, 1, 31),
    (2024, 1): date(2024, 5, 2),
    (2024, 2): date(2024, 8, 26),
    (2024, 3): date(2024, 10, 31),
    (2024, 4): date(2025, 2, 28),
    (2025, 1): date(2025, 12, 22),
    (2025, 2): date(2026, 1, 14),
    (2025, 3): date(2026, 3, 12),
    (2025, 4): date(2026, 7, 17),
    (2026, 1): date(2026, 8, 6),
    (2026, 2): date(2026, 8, 6),
}

_FEATURES = ("cancel_ratio_20", "hidden_share_20", "oddlot_share_20", "fill_ratio_20")

#: (분자 20일합 컬럼, 분모 20일합 컬럼, 결과 이름).
_RATIO_SPECS = (
    ("cancels", "lit_trades", "cancel_ratio_20"),
    ("hidden_vol_k", "trade_vol_for_hidden_k", "hidden_share_20"),
    ("odd_lot_vol_k", "trade_vol_for_odd_lots_k", "oddlot_share_20"),
    ("lit_vol_k", "order_vol_k", "fill_ratio_20"),
)


def _available_from_table() -> pl.DataFrame:
    rows = [
        {"_year": year, "_quarter": quarter, "_table_available_from": available_from}
        for (year, quarter), available_from in MIDAS_AVAILABLE_FROM.items()
    ]
    return pl.DataFrame(
        rows,
        schema={"_year": pl.Int32, "_quarter": pl.Int32, "_table_available_from": pl.Date},
    )


def _with_available_date(midas: pl.LazyFrame) -> pl.LazyFrame:
    """``date``가 속한 분기 끝을 계산하고, 표 또는 폴백으로 ``available_date``를 붙인다."""
    lookup = _available_from_table().lazy()
    quarter = ((pl.col("date").dt.month() - 1) // 3 + 1).cast(pl.Int32)
    quarter_end = pl.date(pl.col("date").dt.year(), quarter * 3, 1).dt.month_end()
    return (
        midas.with_columns(
            pl.col("date").dt.year().cast(pl.Int32).alias("_year"),
            quarter.alias("_quarter"),
            quarter_end.alias("_quarter_end"),
        )
        .join(lookup, on=["_year", "_quarter"], how="left")
        .with_columns(
            pl.when(pl.col("_table_available_from").is_not_null())
            .then(pl.col("_table_available_from"))
            .otherwise(pl.col("_quarter_end") + pl.duration(days=MIDAS_FALLBACK_LAG_DAYS))
            .alias("available_date")
        )
        .drop("_year", "_quarter", "_quarter_end", "_table_available_from")
    )


def add_order_flow(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F18 주문흐름 피쳐 + ``_isna``를 붙인다."""
    symbols = panel_symbols(panel)
    by = group_key(lake)
    segmented = lake.security_boundaries

    midas = (
        # security_type == 'Stock'만 (lake._clean_midas_security_daily가 이미 건다).
        lake.scan("midas_security_daily")
        .filter(pl.col("ticker").is_in(symbols))
        .select(
            "date",
            pl.col("ticker").alias("symbol"),
            "cancels",
            "lit_trades",
            "hidden_vol_k",
            "trade_vol_for_hidden_k",
            "odd_lot_vol_k",
            "trade_vol_for_odd_lots_k",
            "lit_vol_k",
            "order_vol_k",
        )
    )
    midas = _with_available_date(midas)
    if segmented:
        midas = attach_security_id(midas, lake)

    def _warm(expr: pl.Expr) -> pl.Expr:
        return mask_warmup(expr, by, _WINDOW) if segmented else expr

    rolling_cols = []
    ratio_terms = []
    for numer_col, denom_col, feature_name in _RATIO_SPECS:
        numer_20 = f"_{numer_col}_20"
        denom_20 = f"_{denom_col}_20"
        rolling_cols.append(
            mask_ticker_reuse_gap(
                _warm(
                    pl.col(numer_col)
                    .cast(pl.Float64)
                    .rolling_sum(window_size=_WINDOW, min_samples=_MIN_VALID_DAYS)
                    .over(by)
                ),
                pl.col("date"),
                _WINDOW - 1,
                by=by,
            ).alias(numer_20)
        )
        rolling_cols.append(
            mask_ticker_reuse_gap(
                _warm(
                    pl.col(denom_col)
                    .cast(pl.Float64)
                    .rolling_sum(window_size=_WINDOW, min_samples=_MIN_VALID_DAYS)
                    .over(by)
                ),
                pl.col("date"),
                _WINDOW - 1,
                by=by,
            ).alias(denom_20)
        )
        ratio_terms.append(
            pl.when(pl.col(denom_20) > 0)
            .then(pl.col(numer_20) / pl.col(denom_20))
            .otherwise(None)
            .alias(feature_name)
        )

    daily = (
        midas.sort([by, "date"])
        .with_columns(rolling_cols)
        .with_columns(ratio_terms)
        .select(*dict.fromkeys(["symbol", by]), "date", "available_date", *_FEATURES)
    )

    panel_lf = panel.lazy().sort(["symbol", "date"])
    if segmented:
        panel_lf = attach_security_id(panel_lf, lake)
    joined = panel_lf.join_asof(
        daily.sort([by, "available_date"]),
        left_on="date",
        right_on="available_date",
        by=by,
        strategy="backward",
    )

    isna_flags = [pl.col(c).is_null().alias(f"{c}_isna") for c in _FEATURES]
    result = joined.with_columns(isna_flags)

    keep = [*panel.columns]
    for c in _FEATURES:
        keep.extend([c, f"{c}_isna"])
    return result.select(keep).sort(["date", "symbol"]).collect()
