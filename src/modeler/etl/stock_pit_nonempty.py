"""``dim_stock_pit_daily`` that ignores empty share-count filings.

``dart_share_count_raw`` '합계' rows carry ``istc_totqy`` NULL (payload ``"-"``) at a rising
rate (by ``stlm_dt`` year: 2022 4.7%, 2023 8.9%, 2024 11.8%, 2025 14.6%, 2026 21%).
``stock_pit.build_stock_pit_sql`` picks the filing with the latest ``stlm_dt``, so an empty
later filing overrides an earlier complete one and the shares (and everything built on them:
market cap, float, ``BASE_OK``) go missing until the next complete filing.

Here the share view is filtered to rows with ``istc_totqy > 0`` before the unchanged PIT SQL
runs, so the latest *complete* filing wins. Use it as ``pit_view`` for ``flow_v2`` /
``fin_v2`` in the next preregistration. The existing ``dim_stock_pit_daily`` (serving, frozen
``sql_hash``) is unchanged; nothing in ``stock_pit.py`` is modified.
"""

from __future__ import annotations

from modeler.etl.stock_pit import PLAN_SINGLE, build_stock_pit_sql


def build_nonempty_share_view_sql(share_view: str = "dart_share_count_raw") -> str:
    """``SELECT *`` from ``share_view`` keeping only rows with ``istc_totqy > 0``.

    The cast is ``TRY_CAST(... AS DOUBLE)`` as in ``build_stock_pit_sql``, so a non-numeric
    payload counts as empty. All columns pass through.
    """
    return f"""
        SELECT *
        FROM {share_view}
        WHERE TRY_CAST(istc_totqy AS DOUBLE) > 0
    """


def build_stock_pit_nonempty_sql(
    *,
    price_view: str = "daily_ohlcv",
    share_view: str = "dart_share_count_raw",
    plan: str = PLAN_SINGLE,
) -> str:
    """``build_stock_pit_sql`` over the non-empty share rows (inline subquery as the view)."""
    return build_stock_pit_sql(
        price_view=price_view,
        share_view=f"({build_nonempty_share_view_sql(share_view)})",
        plan=plan,
    )
