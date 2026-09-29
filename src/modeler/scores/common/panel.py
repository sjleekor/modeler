"""PIT 자산 패널: (asset_id, 결정 세션 t) 한 행.

계약(사양 01 §5): 입력마다 ``available_at <= decision_at < entry_at``.

가격 ``available_at``: 사용한 가격 세션(``last_price_session``)의 폐장 시각 + 60분
(``available_at_basis="session_close_plus_60min"``). 종가 확정·수집·배포 지연을 뭉뚱그린
보수적 여유다. 15:30·16:00 정각에 종가와 수급이 다 나왔다고 가정하지 않는다. 다음 세션
개장 30분 전이 결정 시각이므로 여유가 크게 남는다. 스냅샷의 ``observed_at``(수집 시각)은
과거 백필 때문에 PIT 근거가 못 된다 — 쓰지 않는다.

KR 지수는 다르다. KRX Open API가 T+1로 공표하므로(당일 행이 23:00 KST에도 없다, 2026-09-29
실측) ``build_asset_panel(available_at_fn=..., available_at_basis=...)``로 세션별 가용 시각을
받는다(``common/kr_inputs.py``). 기본 동작(위 60분 규칙)은 그대로다.

t 세션에 가격이 없으면(``dq_price_missing_at_t``) 마지막으로 관측된 세션의 값을 그대로
들고 오되(as-of), ``last_price_session``이 t보다 앞선다. 라벨은 그런 이동을 하지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta

import polars as pl

from modeler.scores.common.calendar import HORIZON_SESSIONS, UTC_TS, SessionCalendar

PRICE_AVAILABILITY_BUFFER = timedelta(minutes=60)
AVAILABLE_AT_BASIS = "session_close_plus_60min"


class PitViolationError(AssertionError):
    """입력의 ``available_at``이 ``decision_at``보다 늦거나 ``decision_at >= entry_at``."""


def assert_pit(
    df: pl.DataFrame,
    available_cols: list[str],
    *,
    decision_col: str = "decision_at",
    entry_col: str = "entry_at",
) -> None:
    """모든 행에서 ``available_at <= decision_at < entry_at``를 확인한다. 어기면 예외."""
    problems: list[str] = []
    for col in available_cols:
        bad = df.filter(pl.col(col) > pl.col(decision_col))
        if bad.height:
            first = bad.row(0, named=True)
            problems.append(
                f"{col} > {decision_col}: {bad.height}행 (첫 행 {first.get('session')}: "
                f"{first[col]} > {first[decision_col]})"
            )
    bad = df.filter(pl.col(decision_col) >= pl.col(entry_col))
    if bad.height:
        problems.append(f"{decision_col} >= {entry_col}: {bad.height}행")
    if problems:
        raise PitViolationError("PIT 위반 — " + "; ".join(problems))


def build_asset_panel(
    asset_id: str,
    cal: SessionCalendar,
    path: pl.DataFrame,
    *,
    horizon: int = HORIZON_SESSIONS,
    available_at_fn: Callable[[date], datetime] | None = None,
    available_at_basis: str = AVAILABLE_AT_BASIS,
) -> pl.DataFrame:
    """한 자산의 패널.

    ``path``: ``compute_total_return_path`` 출력(``session, px_raw, px_adj, tr_index, ...``).
    행 범위는 첫 관측 세션 ~ 마지막 관측 세션이다(그 뒤에는 입력 가격이 없다).

    ``available_at_fn``: 가격 세션 -> 그 가격을 알 수 있는 시각(tz-aware). 없으면 폐장 + 60분.
    """
    if path.height == 0:
        raise ValueError(f"{asset_id}: 가격 경로가 비었습니다")
    tab = cal.session_table()
    unknown = path.join(tab.select("session"), on="session", how="anti")
    if unknown.height:
        raise ValueError(
            f"{asset_id}: 달력에 없는 가격 세션 {unknown.height}개 ({unknown['session'][0]} ...)"
        )
    first, last = path["session"].min(), path["session"].max()
    basis = path["return_basis"][0]

    t = tab.join(
        path.select(
            "session",
            pl.col("session").alias("_obs"),
            "px_raw",
            "px_adj",
            "tr_index",
        ),
        on="session",
        how="left",
    ).with_columns(
        pl.col("close_at").shift(-1).alias("entry_at"),
        pl.col("session").shift(-1).alias("entry_session"),
        pl.col("close_at").shift(-(1 + horizon)).alias("exit_at"),
        pl.col("session").shift(-(1 + horizon)).alias("exit_session"),
        pl.col("_obs").is_null().alias("dq_price_missing_at_t"),
    )
    t = t.with_columns(
        pl.col("_obs").forward_fill().alias("last_price_session"),
        pl.col("px_raw").forward_fill().alias("px_raw_t"),
        pl.col("px_adj").forward_fill().alias("px_adj_t"),
        pl.col("tr_index").forward_fill().alias("tr_index_t"),
    )
    t = t.filter((pl.col("session") >= first) & (pl.col("session") <= last))
    if available_at_fn is None:
        t = t.join(
            tab.select(
                pl.col("session").alias("last_price_session"),
                pl.col("close_at").alias("_lp_close"),
            ),
            on="last_price_session",
            how="left",
        ).with_columns(
            (pl.col("_lp_close") + PRICE_AVAILABILITY_BUFFER)
            .cast(UTC_TS)
            .alias("price_available_at")
        )
    else:
        sess = t["last_price_session"].drop_nulls().unique().sort().to_list()
        avail = pl.DataFrame(
            {
                "last_price_session": sess,
                "price_available_at": pl.Series([available_at_fn(d) for d in sess], dtype=UTC_TS),
            },
            schema={"last_price_session": pl.Date, "price_available_at": UTC_TS},
        )
        t = t.join(avail, on="last_price_session", how="left")
    out = t.select(
        pl.lit(asset_id).alias("asset_id"),
        "session",
        "decision_at",
        "entry_session",
        "entry_at",
        "exit_session",
        "exit_at",
        "last_price_session",
        "price_available_at",
        pl.lit(available_at_basis).alias("available_at_basis"),
        "px_raw_t",
        "px_adj_t",
        "tr_index_t",
        "dq_price_missing_at_t",
        pl.lit(basis).alias("return_basis"),
        pl.lit(cal.calendar_basis).alias("calendar_basis"),
    ).sort("session")
    # 마지막 세션 다음이 달력에 없는 행은 만들 수 없다(달력이 가격보다 앞서 끝난 경우).
    out = out.filter(pl.col("decision_at").is_not_null())
    assert_pit(out, ["price_available_at"])
    return out
