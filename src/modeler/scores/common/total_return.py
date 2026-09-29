"""총수익 경로 (사양 01 §5.1).

분할만 조정한 가격 ``P*``와 같은 주식 수 기준으로 조정한 주당 배당 ``D*``::

    V(entry) = 1
    V(u) / V(u_prev) = (P*(u) + D*(u)) / P*(u_prev),   u > entry

* 배당은 **배당락일 종가에 재투자**한다. 진입일 배당은 받지 않는다 — 진입일의
  ``V``가 기준(1)이고, 진입 이후 세션의 비율만 쓰기 때문이다. (전체 경로 지수
  ``tr_index``는 계열 첫 세션에서 1로 시작하는 연속 지수이며, 라벨은 항상
  ``tr_index(u)/tr_index(entry)``로 쓴다. 그러면 진입일의 배당은 분자·분모에 모두
  들어가 상쇄된다.)
* ``corp_actions.amount``는 **배당락일 당시 주식 수 기준의 원 금액**이다(2026-09-29
  실측: XLK는 2025-12-05 2:1 분할 전 0.40, 후 0.22 수준으로 가격과 같은 스케일).
  그래서 ``D* = amount × Π(for/to) (분할 ex_date > 배당 ex_date)`` — 가격 조정(``us/prices.py``)과
  같은 "엄격히 이후" 규칙이다.
* 배당락일이 가격 세션이 아니면 그 이후 첫 관측 세션에 붙인다.
* 첫 관측 세션 이전·마지막 관측 세션 이후의 배당은 경로에 넣지 않는다(수량을 남긴다).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import polars as pl

from modeler.scores.common.inputs import PinnedScopedLake
from modeler.us.prices import adjusted_daily, split_factors

logger = logging.getLogger(__name__)

RETURN_BASIS_TOTAL = "total_return"
RETURN_BASIS_PRICE = "price_only"

#: 최근 배당이 빠졌을 가능성 표시: 마지막 배당락일 이후 경과일이 최근 배당 간격 중앙값의
#: 이 배수를 넘으면 표시한다.
STALE_DIVIDEND_GAP_FACTOR = 1.25


def split_adjusted_dividends(
    lake: PinnedScopedLake, *, base_date: date | None = None
) -> pl.DataFrame:
    """``symbol, ex_date, amount, div_adj`` — 같은 (symbol, ex_date)는 합친다."""
    divs = (
        lake.scan("corp_actions")
        .filter(pl.col("kind") == "dividend")
        .select(
            "symbol",
            "ex_date",
            pl.col("amount").cast(pl.Float64).alias("amount"),
        )
        .filter(pl.col("amount").is_not_null() & (pl.col("amount") > 0))
        .group_by(["symbol", "ex_date"])
        .agg(pl.col("amount").sum())
        .sort(["symbol", "ex_date"])
        .collect()
    )
    if base_date is not None:
        divs = divs.filter(pl.col("ex_date") <= base_date)
    if divs.height == 0:
        return divs.with_columns(pl.lit(None, dtype=pl.Float64).alias("div_adj"))

    sf = (
        split_factors(lake, base_date=base_date)
        .with_columns((pl.col("date") - timedelta(days=1)).alias("_boundary"))
        .select("symbol", "_boundary", "split_factor")
        .sort(["symbol", "_boundary"])
        .collect()
    )
    # 분할 ex_date > 배당 ex_date  <=>  분할 ex_date - 1일 >= 배당 ex_date
    joined = divs.sort(["symbol", "ex_date"]).join_asof(
        sf, left_on="ex_date", right_on="_boundary", by="symbol", strategy="forward"
    )
    return joined.with_columns(
        (pl.col("amount") * pl.col("split_factor").fill_null(1.0)).alias("div_adj")
    ).select("symbol", "ex_date", "amount", "div_adj")


def compute_total_return_path(
    prices: pl.DataFrame, dividends: pl.DataFrame | None
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """순수 함수. ``prices``: ``session, px_adj``(+선택 ``px_raw``),
    ``dividends``: ``ex_date, div_adj``.

    반환: (``session, [px_raw,] px_adj, div_adj, tr_index, return_basis``, 진단 dict).
    ``dividends``가 ``None``이거나 행이 0이면 ``return_basis=price_only``.
    """
    px = prices.sort("session").unique(subset=["session"], keep="last", maintain_order=True)
    diag: dict[str, Any] = {"dividends_unattributed": 0, "dividends_shifted": 0}
    have_div = dividends is not None and dividends.height > 0
    if have_div:
        assert dividends is not None
        sess = px.select("session").sort("session")
        d = (
            dividends.select("ex_date", "div_adj")
            .group_by("ex_date")
            .agg(pl.col("div_adj").sum())
            .sort("ex_date")
            .join_asof(
                sess.with_columns(pl.col("session").alias("_sess")),
                left_on="ex_date",
                right_on="session",
                strategy="forward",
            )
        )
        diag["dividends_unattributed"] = d.filter(pl.col("_sess").is_null()).height
        diag["dividends_shifted"] = d.filter(
            pl.col("_sess").is_not_null() & (pl.col("_sess") != pl.col("ex_date"))
        ).height
        per_session = (
            d.filter(pl.col("_sess").is_not_null())
            .group_by("_sess")
            .agg(pl.col("div_adj").sum())
            .rename({"_sess": "session"})
        )
        px = px.join(per_session, on="session", how="left")
    else:
        px = px.with_columns(pl.lit(None, dtype=pl.Float64).alias("div_adj"))

    px = px.sort("session").with_columns(pl.col("div_adj").fill_null(0.0))
    ratio = ((pl.col("px_adj") + pl.col("div_adj")) / pl.col("px_adj").shift(1)).fill_null(1.0)
    px = px.with_columns(ratio.cum_prod().alias("tr_index")).with_columns(
        pl.lit(RETURN_BASIS_TOTAL if have_div else RETURN_BASIS_PRICE).alias("return_basis")
    )
    cols = ["session"] + (["px_raw"] if "px_raw" in px.columns else [])
    return px.select(*cols, "px_adj", "div_adj", "tr_index", "return_basis"), diag


def dividend_coverage(
    dividends: pl.DataFrame | None, *, last_price_session: date | None
) -> dict[str, Any]:
    """배당 행 수·범위·최신성. readiness 표에 들어간다.

    ``possibly_missing_recent``: 마지막 배당락일 이후 경과일이 최근 배당 간격 중앙값의
    1.25배를 넘으면 참 — 행이 있다는 사실만으로 최근 분배금까지 빠짐없다고 볼 수 없다
    (사양 01 §2.2).
    """
    if dividends is None or dividends.height == 0:
        return {
            "dividend_rows": 0,
            "ex_date_min": None,
            "ex_date_max": None,
            "median_gap_days_recent8": None,
            "days_since_last_ex_date": None,
            "possibly_missing_recent": None,
        }
    ex = dividends["ex_date"].unique().sort()
    gaps = ex.diff().dt.total_days().drop_nulls().tail(8)
    med = float(gaps.median()) if len(gaps) else None  # type: ignore[arg-type]
    since = (last_price_session - ex.max()).days if last_price_session else None  # type: ignore[operator]
    stale = None if med is None or since is None else since > STALE_DIVIDEND_GAP_FACTOR * med
    return {
        "dividend_rows": int(dividends.height),
        "ex_date_min": ex.min(),
        "ex_date_max": ex.max(),
        "median_gap_days_recent8": med,
        "days_since_last_ex_date": since,
        "possibly_missing_recent": stale,
    }


def load_us_total_return(
    lake: PinnedScopedLake,
    symbols_by_asset: dict[str, str],
    *,
    sessions: frozenset[date] | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, Any]]]:
    """자산별 총수익 경로와 진단(배당 커버리지·가격 행/중복 수).

    ``symbols_by_asset``: ``asset_id -> 심볼``. ``sessions``를 주면 달력에 없는 날짜의
    가격 행(휴장일 행 등)은 경로에서 뺀다 — 세션이 아닌 날을 세션으로 세지 않고, 이웃
    날짜로 옮기지도 않는다. 뺀 수와 날짜는 진단에 남는다. 반환 경로는 ``session, px_raw, px_adj,
    div_adj, tr_index, return_basis``.
    """
    symbols = tuple(symbols_by_asset.values())
    adj = (
        adjusted_daily(lake)
        .filter(pl.col("symbol").is_in(list(symbols)))
        .filter(pl.col("adj_close").is_not_null() & (pl.col("adj_close") > 0))
        .select("symbol", pl.col("date").alias("session"), "close", "adj_close")
        .collect()
    )
    base_date = adj["session"].max()
    divs = split_adjusted_dividends(lake, base_date=base_date)
    paths: dict[str, pl.DataFrame] = {}
    diags: dict[str, dict[str, Any]] = {}
    for asset_id, sym in symbols_by_asset.items():
        p = adj.filter(pl.col("symbol") == sym).select(
            "session",
            pl.col("close").alias("px_raw"),
            pl.col("adj_close").alias("px_adj"),
        )
        raw_rows = p.height
        dup = raw_rows - p["session"].n_unique()
        off_dates: list[date] = []
        if sessions is not None:
            off = p.filter(~pl.col("session").is_in(list(sessions)))
            off_dates = sorted(off["session"].unique().to_list())
            p = p.filter(pl.col("session").is_in(list(sessions)))
        d_sym = divs.filter(pl.col("symbol") == sym)
        path, diag = compute_total_return_path(p, d_sym if d_sym.height else None)
        last = path["session"].max() if path.height else None
        diag.update(
            {
                "price_rows": raw_rows,
                "duplicate_price_dates": dup,
                "prices_off_calendar": len(off_dates),
                "prices_off_calendar_dates": [d.isoformat() for d in off_dates],
                **dividend_coverage(d_sym if d_sym.height else None, last_price_session=last),
            }
        )
        paths[asset_id] = path
        diags[asset_id] = diag
        logger.info(
            "%s(%s): 가격 %d행 · 배당 %s행 [%s ~ %s] · %s",
            asset_id,
            sym,
            raw_rows,
            diag["dividend_rows"],
            diag["ex_date_min"],
            diag["ex_date_max"],
            path["return_basis"][0] if path.height else "n/a",
        )
    return paths, diags
