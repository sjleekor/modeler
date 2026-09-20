"""라벨 L0·L1·L2 — ``02_labels_universe_benchmark.md`` §2.

| 층 | 정의 |
|---|---|
| L0 | ``adj_close(t+21)/adj_close(t) - 1``. 거래일 21일 뒤(``HORIZON_TRADING_DAYS``) |
| L1 | ``L0 - mean_universe(L0)`` (그날 유니버스 동일가중 — 라벨 벤치마크, ``02`` §3) |
| L2 | ``x=log(adv_20d)``·``y=L1``의 백분위 순위로 ``y ~ x + x^2 + sic2``를 회귀한 잔차 |
| y_rank | L2의 그날 횡단면 백분위 순위 [0,1] |
| y_up | ``L2 > 0`` |

**L2를 수준(level) 공간이 아니라 순위(rank) 공간에서 중립화한다.** 처음 구현은
``log(adv_20d)`` 10분위 더미로 중립화했는데, M1 완료 판정(``02`` §7)이 요구하는
``max|ρ(L2, log(adv_20d))| < 0.02``에 크게 못 미쳤다(실측 0.169, ``mcap_rank``는
0.244 — 레이크 결함을 고친 뒤에도 그대로였다). 원인은 10분위 더미가 데실
"사이" 평균차만 걷고 데실 "안"의 연속적 크기효과를 못 걷는 것, 그리고 게이트가
재는 것은 rank 상관인데 회귀는 수준에서 한다는 것 — 정의가 서로 안 맞았다.
그래서 ``x``·``y`` 모두 그날 횡단면 백분위 순위로 바꾸고, 순위상 비선형
크기효과까지 걷도록 ``x^2``을 더했다. 게이트 기준도 함께 바꿨다 — 종목 약
4,000개면 순위상관의 표준오차가 ``1/√4000 ≈ 0.016``이라, 완벽히 중립화된
데이터도 96개월 중 일부 달은 ``|ρ|``가 0.03~0.05까지 나온다. ``max|ρ| < 0.02``는
잡음 바닥보다 낮은 값을 96개월 전부에 요구한 셈이라 애초에 달성 불가능했다 —
새 기준은 ``mean|ρ| < 0.03``(두 축 모두)이고 ``max|ρ|``는 기록만 한다.

**가격이 끊긴 종목** (``t``에 유니버스에 있었는데 ``t+21``에 가격이 정확히 없는 종목)은
사유별로 닫는다 (``02`` §2.1): ``listing_snapshots.financial_status``가 마지막
스냅샷(``as_of <= t+21``)에서 부실이면 마지막 체결가에 −30%(Shumway 1997), 그
외는 마지막 체결가 그대로. "마지막 체결가"는 이 모듈에서 **조정 종가**
(``adj_close``)를 쓴다 — ``adj_close(t)``와 같은 스케일이어야 수익률 비율이
맞기 때문이다(원 계획 문서는 원시/조정을 명시하지 않는다 — 판단 근거는
``build_labels.py`` 보고 참고).

**t+21이 레이크의 가격 데이터 밖인 리밸런스일은 통째로 뺀다** — 그 달은 어느
종목도 라벨을 계산할 근거(미래 가격)가 없다는 뜻이라, 개별 종목의 "닫힘"과는
다른 사유다 (``dropped_rebalance_dates``로 남긴다).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

from modeler.us.lake import UsLake
from modeler.us.panel import build_panel
from modeler.us.prices import adjusted_daily

#: 리밸런스 t에서 라벨 만기까지의 거래일 수 (``02`` §1 h21 — 주 horizon).
HORIZON_TRADING_DAYS = 21

#: ``trading_calendar``에 실제 있는 거래소 하나. ``panel.py``의 ``_EXCHANGE``와
#: 같은 값이다 (2026-09-20 실측 — XNYS 하나뿐, ``08_plan_review.md``).
_EXCHANGE = "XNYS"

#: 소수 셀 규칙 — 그달 유니버스에서 이 미만인 ``sic2``는 "기타"로 묶는다
#: (``02`` §2, 계획 검토 V12). 안 묶으면 종목 1~2개짜리 업종은 회귀가 완전
#: 적합돼 잔차가 0이 되고 그 라벨이 사라진다.
MIN_SIC2_GROUP_SIZE = 20

#: 미분류(``sic2`` 결측) 더미 이름.
UNCLASSIFIED_SIC2 = "미분류"

#: 소수 셀을 묶는 더미 이름.
OTHER_SIC2 = "기타"

#: 부실 상폐 시 마지막 체결가에 적용하는 충격 (Shumway 1997 CRSP 관행).
DISTRESS_SHOCK = 0.30

#: ``listing_snapshots.financial_status`` 중 부실로 보는 코드.
#:
#: 나스닥 Financial Status Indicator 정의: N=정상, D=Deficient(요건 미달),
#: E=Delinquent(보고 지연), Q=Bankrupt(파산), 그리고 그 조합
#: G(D+Q)·H(D+E)·J(E+Q)·K(D+E+Q). ``null``은 이 지표 자체가 없는 종목
#: (``otherlisted``, 대개 NYSE)이라 부실 여부를 알 수 없다 — "그 외"로 둔다.
#: 2026-09-20 실측 분포(``listing_snapshots`` 전체 행 기준)는
#: N 285,787 · null 236,021 · D 14,960 · E 1,617 · H 519 · G 17 · K 8 · Q 8 · J 4다.
DISTRESS_FINANCIAL_STATUS = frozenset({"D", "E", "G", "H", "J", "K", "Q"})

#: ``|L0|``가 이 값을 넘으면 라벨에서 뺀다 — corp_actions 중복 분할 행이 만드는
#: 조정가 불연속(20~3000배 어긋남)을 거르는 최소한의 방어선이다. 21거래일에
#: 1000%(10배) 넘는 수익은 조직적 데이터 결함일 가능성이 실제 극단적 시세보다
#: 훨씬 크다고 보고 정했다 — 근거는 ``build_labels.py`` 실행 보고 참고.
MAX_PLAUSIBLE_ABS_L0 = 10.0

#: 같은 심볼의 연속 관측 사이 간격(달력일)이 이 값을 넘으면 "같은 종목이
#: 이어졌다"고 보지 않는다 — **티커 재사용** 방어선이다. ``prices_daily``는
#: 상폐 뒤 같은 티커를 다른 회사가 쓰는 계열을 구분 없이 담고 있다
#: (2026-09-20 실측 — 패널 종목 9,383개 중 634개가 1년 넘는 가격 공백을
#: 갖는다. 예: JONE이 2018-11-26 $2.13에서 2026-09-03 $9.84로 "이어진다").
#: 21거래일은 달력으로 약 29~31일이라, 그 두 배 가까이(60일)를 문턱으로 잡아
#: 정상적인 거래정지·휴장 몰림은 통과시키고 몇 달 이상의 공백만 걸러낸다.
#: 걸리면 사유를 ``TICKER_REUSE_GAP``으로 따로 세어 manifest에 남긴다 —
#: features 쪽 ``mask_ticker_reuse_gap``(``features/_daily.py``, 다른
#: 에이전트 소유)과 같은 문제의식이지만, 여기서는 라벨 쪽에 독립적으로 짠다.
MAX_TICKER_GAP_DAYS = 60

#: 공백 때문에 닫힌 종목의 종가 사유.
TICKER_REUSE_GAP = "ticker_reuse_gap"


def trading_day_offsets(lake: UsLake, dates: Sequence[date], n: int) -> dict[date, date | None]:
    """``dates`` 각각 -> ``n`` 거래일 뒤 날짜. 거래일 달력 밖이면 ``None``.

    ``trading_calendar``(XNYS) 전체를 한 번 읽어 색인한다. ``dates``는 이미
    거래일이어야 한다 — 리밸런스일이 그렇듯 ``month_first_trading_days``가 주는
    값이면 된다.
    """
    calendar = (
        lake.scan("trading_calendar")
        .filter(pl.col("exchange") == _EXCHANGE)
        .select("date")
        .sort("date")
        .collect()["date"]
        .to_list()
    )
    index = {d: i for i, d in enumerate(calendar)}
    result: dict[date, date | None] = {}
    for d in dates:
        i = index.get(d)
        if i is None or i + n >= len(calendar):
            result[d] = None
        else:
            result[d] = calendar[i + n]
    return result


def bucket_sic2(df: pl.DataFrame, *, min_group_size: int = MIN_SIC2_GROUP_SIZE) -> pl.DataFrame:
    """``sic2`` -> ``sic2_bucket`` 컬럼을 붙인다.

    결측은 ``UNCLASSIFIED_SIC2``("미분류"), 그날 유니버스에서 ``min_group_size``
    미만인 ``sic2``는 ``OTHER_SIC2``("기타")로 묶는다. **``df``는 이미 하나의
    횡단면(한 날짜)이어야 한다** — 그룹 크기는 그날 유니버스 기준이다.

    ``group_by().join()``이 아니라 ``.over()``로 그룹 크기를 구한다 — join은
    행 순서를 보장하지 않아, join 결과에 원래 순서로 계산한 다른 컬럼을 붙이면
    조용히 어긋난다 (예전 구현의 ``adv_decile``이 실제로 이 문제를 겪었다 — 지금은
    ``neutralize_cross_section``이 순위를 elementwise로만 계산해 이 문제가 없다).
    """
    return df.with_columns(
        pl.when(pl.col("sic2").is_null())
        .then(pl.lit(UNCLASSIFIED_SIC2))
        .when(pl.len().over("sic2") < min_group_size)
        .then(pl.lit(OTHER_SIC2))
        .otherwise(pl.col("sic2"))
        .alias("sic2_bucket")
    )


def _percentile_rank(series: pl.Series) -> pl.Series:
    """SQL ``PERCENT_RANK()``와 같은 정의: ``(rank-1)/(n-1)``, 동순위는 최소 rank. [0,1].

    회귀 입력(``x``·``y``)과 최종 출력 ``y_rank`` 모두 이 정의 하나로 통일한다 —
    ``modeler.etl.labels``의 한국 라벨과 같은 관례다. 순위는 단조변환에 불변이라
    ``log(adv_20d)``의 순위와 ``adv_20d``의 순위는 같다 — 그래도 정의(``x=log(adv_20d)``의
    백분위)를 코드에 그대로 드러내려고 ``log()``를 명시해서 넘긴다.
    """
    n = series.len()
    denom = max(n - 1, 1)
    return (series.rank(method="min").cast(pl.Float64) - 1) / denom


def neutralize_cross_section(df: pl.DataFrame) -> pl.DataFrame:
    """한 날짜(횡단면)의 ``L1``을 중립화한 잔차 ``L2``와 ``y_rank``·``y_up``을 붙인다.

    **순위(rank) 공간에서 중립화한다** (2026-09-20 정정 — 모듈 docstring 참고).
    ``x``=``log(adv_20d)``의 그날 횡단면 백분위 순위, ``y``=``L1``의 그날 횡단면
    백분위 순위(둘 다 ``_percentile_rank``, [0,1])로 바꾼 뒤, 회귀는 절편 +
    ``x`` + ``x^2``(순위상 비선형 크기효과까지 걷는다) + (``sic2_bucket`` 더미,
    기준 하나 드롭)이고 ``numpy.linalg.lstsq``로 푼다. 절편이 있으므로 잔차의
    합은 부동소수점 오차 안에서 0이다 (``02`` §7 완료 판정).

    ``lstsq``는 SVD 기반이라 ``x``·``x^2``가 극단(순위 0·1 부근)에서 거의
    공선(共線)이 되거나 사실상 특이행렬이 되는 경우도 최소노름해로 처리한다 —
    별도 예외 처리가 필요 없다(더미는 ``drop_first``로 완전공선만 미리 없앤다).

    ``df``는 ``date``가 하나뿐이어야 하고 ``adv_20d``·``sic2``·``L1`` 컬럼이
    있어야 한다.
    """
    n = df.height
    bucketed = bucket_sic2(df)

    x = _percentile_rank(bucketed["adv_20d"].log())
    y = _percentile_rank(bucketed["L1"])
    bucketed = bucketed.with_columns(x.alias("adv_rank"))

    dummy_cols = bucketed.select(["sic2_bucket"]).to_dummies(
        columns=["sic2_bucket"], drop_first=True
    )
    # ``sic2_bucket``의 고유값이 하나뿐이면(그날 횡단면 전부 같은 버킷 — 실데이터에는
    # 거의 없지만 방어한다) polars ``to_dummies(drop_first=True)``가 (0, 0)짜리
    # DataFrame을 준다((n, 0)이 아니다 — 실측 polars 1.44.2). 그대로 column_stack에
    # 넣으면 행 수가 안 맞아 죽는다. 더미가 없다는 뜻이니 (n, 0) 배열로 바로잡는다.
    if dummy_cols.width == 0:
        dummy_arr = np.empty((n, 0), dtype=np.float64)
    else:
        dummy_arr = dummy_cols.to_numpy().astype(np.float64)
    x_arr = x.to_numpy().astype(np.float64)
    design = np.column_stack(
        [
            np.ones(n, dtype=np.float64),
            x_arr,
            x_arr**2,
            dummy_arr,
        ]
    )
    y_arr = y.to_numpy().astype(np.float64)
    beta, _residuals, _rank, _sv = np.linalg.lstsq(design, y_arr, rcond=None)
    residual = y_arr - design @ beta

    l2 = pl.Series("L2", residual)
    percent_rank = _percentile_rank(l2)

    return bucketed.with_columns(
        l2,
        percent_rank.alias("y_rank"),
        (l2 > 0).alias("y_up"),
    )


def add_l2(df: pl.DataFrame) -> pl.DataFrame:
    """``date``별로 ``neutralize_cross_section``을 적용해 이어붙인다.

    ``df``가 빈 경우(그달 라벨이 전부 걸러진 경우 등) ``partition_by``가 파티션을
    하나도 안 줘 ``pl.concat([])``이 죽는다 — ``neutralize_cross_section``이
    추가했을 컬럼들을 빈 값으로 채워 스키마만 맞춰 돌려준다.
    """
    parts = df.partition_by("date", maintain_order=True)
    if not parts:
        return df.with_columns(
            pl.lit(None, dtype=pl.String).alias("sic2_bucket"),
            pl.lit(None, dtype=pl.Float64).alias("adv_rank"),
            pl.lit(None, dtype=pl.Float64).alias("L2"),
            pl.lit(None, dtype=pl.Float64).alias("y_rank"),
            pl.lit(None, dtype=pl.Boolean).alias("y_up"),
        )
    return pl.concat([neutralize_cross_section(part) for part in parts])


def _segment_terminal(
    daily: pl.DataFrame,
    anchor: pl.DataFrame,
    t21: date,
    *,
    max_gap_days: int = MAX_TICKER_GAP_DAYS,
) -> pl.DataFrame:
    """``t``(anchor)부터 ``t21``까지 **끊기지 않고 이어진** 마지막 관측을 찾는다.

    ``anchor``는 ``symbol, date(모두 t), adj_close`` — 그 심볼의 ``t`` 시점
    값이다(패널이 주는 값을 그대로 쓴다). ``t`` 다음 관측부터 ``t21``까지
    ``daily``에서 훑으면서, 연속 관측 사이 간격(달력일)이 ``max_gap_days``를
    넘는 지점(티커 재사용 의심, 모듈 docstring·``MAX_TICKER_GAP_DAYS`` 참고)을
    만나면 그 **앞**에서 멈춘다.

    반환: ``symbol, terminal_date, terminal_adj_close, had_gap``.
    ``had_gap``이 참이면 ``terminal_date``가 공백 앞의 마지막 관측일이고,
    ``t21``보다 이르다 — 그 심볼은 정확히 ``t21``에 값이 있어도 공백을 건넌
    값이므로 이어짐으로 치지 않는다(호출부가 그 값을 안 쓴다).
    """
    symbols = anchor["symbol"].to_list()
    if not symbols:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "terminal_date": pl.Date,
                "terminal_adj_close": pl.Float64,
                "had_gap": pl.Boolean,
            }
        )
    t = anchor["date"][0]

    window = daily.filter(
        pl.col("symbol").is_in(symbols) & (pl.col("date") > t) & (pl.col("date") <= t21)
    ).select("symbol", "date", "adj_close")

    combined = (
        pl.concat([anchor.select("symbol", "date", "adj_close"), window])
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("date") - pl.col("date").shift(1).over("symbol"))
            .dt.total_days()
            .fill_null(0)
            .alias("_gap_days")
        )
        .with_columns((pl.col("_gap_days") > max_gap_days).alias("_is_break"))
        .with_columns(pl.col("_is_break").cast(pl.Int32).cum_sum().over("symbol").alias("_segment"))
    )

    had_gap = combined.group_by("symbol").agg(pl.col("_is_break").any().alias("had_gap"))
    terminal = (
        combined.filter(pl.col("_segment") == 0)
        .sort(["symbol", "date"])
        .group_by("symbol", maintain_order=True)
        .agg(
            pl.col("date").last().alias("terminal_date"),
            pl.col("adj_close").last().alias("terminal_adj_close"),
        )
    )
    return terminal.join(had_gap, on="symbol", how="left")


def _distress_at(
    listing_snapshots: pl.DataFrame, symbols: Sequence[str], cutoff: date
) -> pl.DataFrame:
    """``symbols`` 각각의 ``as_of <= cutoff`` 중 마지막 스냅샷이 부실인지.

    ``listing_snapshots``는 ``symbol, as_of, financial_status`` 컬럼이 있는
    전체 표(eager)다. 스냅샷이 아예 없는 종목(``otherlisted``라 지표가 없거나,
    ``cutoff`` 이전 캡처가 없는 경우)은 부실 여부를 알 수 없으므로 ``False``
    ("그 외")로 둔다.
    """
    if not symbols:
        return pl.DataFrame(schema={"symbol": pl.String, "is_distress": pl.Boolean})

    scoped = listing_snapshots.filter(
        pl.col("symbol").is_in(list(symbols)) & (pl.col("as_of") <= cutoff)
    )
    if scoped.height == 0:
        return pl.DataFrame({"symbol": list(symbols), "is_distress": [False] * len(symbols)})

    last_as_of = scoped.group_by("symbol").agg(pl.col("as_of").max().alias("as_of"))
    last_rows = scoped.join(last_as_of, on=["symbol", "as_of"], how="inner")
    distress = last_rows.group_by("symbol").agg(
        pl.col("financial_status").is_in(list(DISTRESS_FINANCIAL_STATUS)).any().alias("is_distress")
    )
    missing = [s for s in symbols if s not in set(distress["symbol"].to_list())]
    if missing:
        distress = pl.concat(
            [distress, pl.DataFrame({"symbol": missing, "is_distress": [False] * len(missing)})]
        )
    return distress


def _label_one_date(
    universe_t: pl.DataFrame,
    t21: date,
    daily: pl.DataFrame,
    listing_snapshots: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """``universe_t``(그 날 유니버스, panel 행) 하나에 ``L0``와 종가 사유를 붙인다.

    ``t``부터 ``t21``까지 조정 종가가 ``MAX_TICKER_GAP_DAYS``보다 크게 끊기면
    (티커 재사용 의심 — 모듈 docstring 참고) **정확히 ``t21``에 값이 있어도**
    이어짐으로 보지 않고 공백 앞의 마지막 값으로 닫는다(``TICKER_REUSE_GAP``).
    그렇지 않은데 ``t21``에 값이 없으면 기존대로 사유별로 닫는다(``02`` §2.1).

    반환: (``L0``가 붙은 DataFrame,
    {"distress_delisted": n, "other": n, "ticker_reuse_gap": n}).
    """
    anchor = universe_t.select("symbol", "date", "adj_close")
    terminal = _segment_terminal(daily, anchor, t21)
    joined = universe_t.join(terminal, on="symbol", how="left")

    gapped = joined.filter(pl.col("had_gap"))
    continuous = joined.filter(~pl.col("had_gap"))

    priced = continuous.filter(pl.col("terminal_date") == t21).with_columns(
        (pl.col("terminal_adj_close") / pl.col("adj_close") - 1).alias("L0"),
        pl.lit(None, dtype=pl.String).alias("close_reason"),
    )
    output_columns = priced.columns

    counts = {"distress_delisted": 0, "other": 0, "ticker_reuse_gap": 0}
    closed_parts: list[pl.DataFrame] = []

    if gapped.height:
        gapped = gapped.with_columns(
            (pl.col("terminal_adj_close") / pl.col("adj_close") - 1).alias("L0"),
            pl.lit(TICKER_REUSE_GAP, dtype=pl.String).alias("close_reason"),
        )
        counts["ticker_reuse_gap"] = gapped.height
        closed_parts.append(gapped.select(output_columns))

    other_candidates = continuous.filter(pl.col("terminal_date") != t21)
    if other_candidates.height:
        symbols = other_candidates["symbol"].to_list()
        distress = _distress_at(listing_snapshots, symbols, t21)
        closed = other_candidates.join(distress, on="symbol", how="left").with_columns(
            pl.when(pl.col("is_distress"))
            .then(pl.lit("distress_delisted"))
            .otherwise(pl.lit("other"))
            .alias("close_reason")
        )
        closed = closed.with_columns(
            (
                pl.col("terminal_adj_close")
                * pl.when(pl.col("is_distress")).then(1 - DISTRESS_SHOCK).otherwise(1.0)
                / pl.col("adj_close")
                - 1
            ).alias("L0")
        )
        counts["distress_delisted"] = int((closed["close_reason"] == "distress_delisted").sum())
        counts["other"] = int((closed["close_reason"] == "other").sum())
        closed_parts.append(closed.select(output_columns))

    combined = pl.concat([priced, *closed_parts]) if closed_parts else priced
    return combined, counts


def build_labels(
    lake: UsLake, *, panel: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, dict[str, object]]:
    """``L0``·``L1``·``L2``·``y_rank``·``y_up``이 붙은 라벨 DataFrame을 만든다.

    ``panel``을 안 주면 ``panel.build_panel(lake)``로 만든다.

    반환: (라벨 DataFrame, diagnostics). diagnostics에는
    ``rebalance_dates_total``·``rebalance_dates_usable``·
    ``dropped_rebalance_dates``(``t+21``이 데이터 밖이라 뺀 리밸런스일)·
    ``closed_by_reason``(사유별 종가 처리 수)이 있다.
    """
    if panel is None:
        panel = build_panel(lake)

    rebalance_dates = sorted(panel["date"].unique().to_list())
    offsets = trading_day_offsets(lake, rebalance_dates, HORIZON_TRADING_DAYS)

    max_price_date = lake.scan("prices_daily").select(pl.col("date").max()).collect().item()

    daily = adjusted_daily(lake).select("date", "symbol", "adj_close").collect()
    listing_snapshots = (
        lake.scan("listing_snapshots").select("symbol", "as_of", "financial_status").collect()
    )

    usable_dates: list[date] = []
    dropped_dates: list[date] = []
    for t in rebalance_dates:
        t21 = offsets.get(t)
        if t21 is None or max_price_date is None or t21 > max_price_date:
            dropped_dates.append(t)
        else:
            usable_dates.append(t)

    parts: list[pl.DataFrame] = []
    closed_by_reason = {"distress_delisted": 0, "other": 0, "ticker_reuse_gap": 0}
    for t in usable_dates:
        universe_t = panel.filter(pl.col("date") == t)
        t21 = offsets[t]
        assert t21 is not None
        labeled_t, counts = _label_one_date(universe_t, t21, daily, listing_snapshots)
        for reason in closed_by_reason:
            closed_by_reason[reason] += counts[reason]
        parts.append(labeled_t)

    if parts:
        l0_df = pl.concat(parts)
    else:
        # 사용 가능한 리밸런스일이 하나도 없다(전부 데이터 밖) — panel 스키마에
        # _label_one_date가 붙이는 컬럼을 더해 빈 DataFrame으로 스키마만 맞춘다.
        l0_df = panel.clear().with_columns(
            pl.lit(None, dtype=pl.Date).alias("terminal_date"),
            pl.lit(None, dtype=pl.Float64).alias("terminal_adj_close"),
            pl.lit(None, dtype=pl.Boolean).alias("had_gap"),
            pl.lit(None, dtype=pl.Float64).alias("L0"),
            pl.lit(None, dtype=pl.String).alias("close_reason"),
        )

    # 레이크 결함 방어: corp_actions에 to_factor=0인 분할 행이 있는 종목
    # (2026-09-20 실측 — AIV·DWDP·IEP·TRI·PHG·SINT)은 prices.split_factors의
    # for_factor/to_factor 나눗셈이 0으로 나눠 NaN/Inf를 내고, cum_prod가 그
    # 종목의 adj_close 전체 이력을 NaN으로 오염시킨다(prices.py는 이 계획에서
    # 고치지 않는다 — 보고만 한다). 이 L0를 그대로 두면 L1이 ``mean().over("date")``
    # 를 쓰므로 NaN 하나가 그 달 전체 종목의 L1·L2를 NaN으로 감염시킨다. 그래서
    # non-finite L0 행은 라벨에서 아예 뺀다(그 종목·그 달만 결측이 된다).
    #
    # ``adj_close``(t 시점, panel이 준 값)도 같이 본다 — 오염된 split이 t와
    # t+21 "사이"에 걸리면 t의 adj_close만 inf가 되고 t+21은 멀쩡해서,
    # L0 = 유한값/inf = 0 근처로 "깨끗하게" 계산돼 ``is_finite()``를 통과해
    # 버린다(실제로 겪었다 — 마치 -100% 수익처럼 보이는 값이 나온다). t의
    # adj_close 자체가 non-finite면 L0의 finite 여부와 무관하게 같이 뺀다.
    finite_mask = pl.col("adj_close").is_finite() & pl.col("L0").is_finite()
    non_finite = l0_df.filter(~finite_mask)
    if non_finite.height:
        l0_df = l0_df.filter(finite_mask)

    # 레이크 결함 방어 (2): corp_actions에 **중복 분할 행**이 있다 — 같은 종목에
    # 거의 같은 배율의 split이 며칠 간격으로 두 번 찍혀 있다(2026-09-20 실측
    # 109건, 예: AMZN이 진짜 20:1 분할(2022-06-06) 말고 11일 전(2022-05-26)에도
    # 똑같이 20:1 행이 하나 더 있다 — 그 날 가격에는 실제 분할 흔적이 전혀 없다).
    # ``prices.split_factors``의 누적곱이 그 여분의 배율까지 곱해서, 그 가짜
    # ex_date 이전 딱 하루치 구간에 20배(때로는 3000배) 어긋난 조정가를 낸다.
    # (t, t+21)이 그 하루를 걸치면 L0가 수천~수만 %로 튄다 — 실제로 겪은 값:
    # AMZN·GOOG·GOOGL이 18배, TTSH가 1,718배. **이 표는 이 계획에서 고치지
    # 않는다**(``prices.py``가 읽는 입력이다) — 대신 명백히 말이 안 되는 L0만
    # 걸러내고 사유를 남긴다. 배율이 작은(2~5배) 중복 분할은 이 문턱을 넘지
    # 않아 걸러지지 않는다 — 실제 극단적 수익률과 구분할 수단이 없다는 뜻이라
    # 그대로 보고한다.
    implausible = l0_df.filter(pl.col("L0").abs() > MAX_PLAUSIBLE_ABS_L0)
    if implausible.height:
        l0_df = l0_df.filter(pl.col("L0").abs() <= MAX_PLAUSIBLE_ABS_L0)

    l1_df = l0_df.with_columns((pl.col("L0") - pl.col("L0").mean().over("date")).alias("L1"))
    labeled = add_l2(l1_df).sort(["date", "symbol"])

    diagnostics: dict[str, object] = {
        "rebalance_dates_total": len(rebalance_dates),
        "rebalance_dates_usable": len(usable_dates),
        "dropped_rebalance_dates": [d.isoformat() for d in dropped_dates],
        "closed_by_reason": closed_by_reason,
        "excluded_non_finite_l0": {
            "count": non_finite.height,
            "symbols": sorted(non_finite["symbol"].unique().to_list()),
        },
        "excluded_implausible_l0": {
            "threshold_abs_l0": MAX_PLAUSIBLE_ABS_L0,
            "count": implausible.height,
            "symbols": sorted(implausible["symbol"].unique().to_list()),
        },
    }
    return labeled, diagnostics
