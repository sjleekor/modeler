"""F14 지수 편입 — ``index_constituents``.

``04_feature_test_plan.md`` §3 F14. 두 피쳐: ``sp500_member`` ·
``sp500_days_since_add``.

``index_constituents.as_of``는 위키 리비전이 기록된 시각(datetime)이다.
``ASOF_AXIS``가 다른 표들과 같이 "날짜" 수준으로 다룬다 — 시각까지 정밀하게
볼 근거가 없어(위키 편집 시각과 시장 개장 여부는 무관하다) ``as_of``를
날짜로 깎아 ``date <= t``로 판정한다.

**표본이 실제로는 주 1회가 아니다** — 최대 47일 간격(``01_data_readiness.md``
§2, ``04`` §3 F14). ``sp500_days_since_add``는 그 오차를 그대로 물려받는다.

**왼쪽이 잘려 있다(left-censored).** 첫 리비전이 2018-01-04다. 그 리비전에
이미 있던 종목은 "언제 편입됐는지" 이 표만으로는 알 수 없다 — 2018-01-04
이전에 편입됐을 수도 있고, 우연히 그 직전에 편입됐을 수도 있다. 이 구분을
못 하면서 억지로 날짜를 만들면(예: 2018-01-04를 편입일로 놓기) 조작한
값이 된다. 그래서 **"편입"이 리비전 사이에서 실제로 관측된 경우에만**
``sp500_days_since_add``를 계산하고, 첫 리비전부터 계속 있던 종목은
``_isna``로 남긴다 — 이번 구현의 판단이다(보고에 적는다).
"""

from __future__ import annotations

import polars as pl

from modeler.us.lake import UsLake


def _membership_revisions(lake: UsLake) -> pl.LazyFrame:
    """(revision_index, as_of_date, symbol) — S&P 500 리비전마다 한 행.

    같은 위키 리비전(``as_of`` 값이 완전히 같다)에 속한 행들이 그 리비전의
    구성 종목이다. 전역 리비전 순번(0부터)을 매겨 "바로 앞 리비전에도
    있었는가"를 판정하는 재료로 쓴다.
    """
    raw = (
        lake.scan("index_constituents")
        .filter(pl.col("index_id") == "SP500")
        .select(pl.col("as_of").dt.date().alias("as_of_date"), "symbol")
        .unique()
    )
    revision_dates = (
        raw.select("as_of_date").unique().sort("as_of_date").with_row_index("revision_index")
    )
    return raw.join(revision_dates, on="as_of_date", how="left").with_columns(
        pl.col("revision_index").cast(pl.Int64)
    )


def _membership_spells(lake: UsLake) -> pl.LazyFrame:
    """(as_of_date, symbol, add_date) — 그 리비전 시점에 그 종목이 속한 "연속 편입 구간"의 시작일.

    ``add_date``는 심볼별로 리비전 순번이 끊기지 않고 이어지는 구간(연속
    편입)의 첫 리비전 날짜다. 그 구간이 **아예 첫 리비전(순번 0)에서
    시작하면** — 즉 2018-01-04부터 계속 있었던 것처럼 보이면 — 실제 편입일을
    모르므로 ``add_date``를 null로 둔다(위 모듈 docstring의 판단).
    """
    revisions = _membership_revisions(lake)
    with_prev = revisions.sort(["symbol", "revision_index"]).with_columns(
        pl.col("revision_index").shift(1).over("symbol").alias("_prev_index")
    )
    # 새 구간의 시작: 이 심볼의 첫 등장이거나(``_prev_index`` null), 바로 전
    # 리비전에는 없었다가(순번이 1보다 크게 뜀) 다시 나타난 경우.
    with_prev = with_prev.with_columns(
        (
            pl.col("_prev_index").is_null()
            | ((pl.col("revision_index") - pl.col("_prev_index")) > 1)
        ).alias("_is_spell_start")
    )
    with_spell_id = with_prev.with_columns(
        pl.col("_is_spell_start").cast(pl.Int64).cum_sum().over("symbol").alias("_spell_id")
    )
    with_add_date = with_spell_id.with_columns(
        pl.when(pl.col("revision_index").min().over(["symbol", "_spell_id"]) == 0)
        .then(None)
        .otherwise(pl.col("as_of_date").min().over(["symbol", "_spell_id"]))
        .alias("add_date")
    )
    return with_add_date.select("as_of_date", "symbol", "add_date")


def add_index_membership(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F14(``sp500_member`` · ``sp500_days_since_add``)를 붙인다."""
    spells = _membership_spells(lake).sort("as_of_date")
    revision_dates = spells.select("as_of_date").unique().sort("as_of_date")

    panel_lf = panel.lazy().sort("date")
    # 리비전 날짜 하나를 골라 그 리비전의 구성 종목 집합을 판다 — 심볼별
    # asof가 아니라 "리비전 자체의" asof다(모든 종목이 같은 리비전 날짜를 쓴다).
    current_revision = (
        panel_lf.select("date")
        .unique()
        .join_asof(revision_dates, left_on="date", right_on="as_of_date", strategy="backward")
    )

    joined = (
        panel.lazy()
        .join(current_revision, on="date", how="left")
        .join(spells, on="as_of_date", how="left", suffix="_spell")
    )
    # spells의 symbol과 패널의 symbol이 같아야 "그 리비전에 이 종목이 있다".
    # ``add_date``는 편입일을 몰라도(왼쪽 잘림) null일 수 있으므로, "멤버인가"는
    # 매치 자체(``_is_member``)로 따로 표시한다 — add_date의 null 여부와
    # 섞으면 왼쪽 잘린 멤버가 비멤버로 잘못 보인다.
    is_member = joined.filter(pl.col("symbol") == pl.col("symbol_spell")).select(
        "date", "symbol", pl.lit(True).alias("_is_member"), "add_date"
    )

    # 첫 리비전(2018-01-04)보다 앞선 날짜는 리비전 자체가 없어 "멤버가 아니다"와
    # "모른다"를 구분해야 한다 — 패널이 2018-09-07부터라 실제로는 안 걸리지만,
    # 더 이른 날짜로 쓰였을 때를 대비해 남겨 둔다.
    no_revision_dates = current_revision.filter(pl.col("as_of_date").is_null()).select("date")

    result = (
        panel.lazy()
        .join(is_member, on=["date", "symbol"], how="left")
        .join(
            no_revision_dates.with_columns(pl.lit(True).alias("_no_revision")),
            on="date",
            how="left",
        )
        .with_columns(pl.col("_is_member").fill_null(False).alias("sp500_member"))
        .with_columns(
            pl.when(pl.col("sp500_member") & pl.col("add_date").is_not_null())
            .then((pl.col("date") - pl.col("add_date")).dt.total_days())
            .otherwise(None)
            .alias("sp500_days_since_add")
        )
        .with_columns(
            pl.col("_no_revision").fill_null(False).alias("sp500_member_isna"),
            pl.col("sp500_days_since_add").is_null().alias("sp500_days_since_add_isna"),
        )
        .drop("_is_member", "add_date", "_no_revision")
        .collect()
    )
    return result.sort(["date", "symbol"])
