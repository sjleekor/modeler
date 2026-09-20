"""``fundamentals`` 분기화 공용 헬퍼 — F5(valuation)·F6(profitability)·F7(investment)·
F8(payout) 네 모듈이 이것을 import해서 쓴다. TTM 로직을 네 번 쓰지 않는다
(``04_feature_test_plan.md`` §3 F5 · ``06_execution_steps.md`` M2).

**문제.** ``fundamentals``는 XBRL 사실 1건이 1행이다. 10-Q는 분기값과 누적값이
섞여 있고(예: "3개월간"과 "9개월간"이 같은 태그로 둘 다 잡힌다), 10-K는 연간값이다.
TTM을 만들려면 먼저 "이 값이 정확히 어느 분기(또는 몇 분기 누적)를 가리키는가"를
가려야 한다.

**실측(2026-09-20, duckdb/polars, ``fundamentals`` 1억 2,548만 행)** — ``fp``만
보고 가르면 안 된다.

1. ``fp``는 사실 하나하나가 아니라 **그 사실이 속한 filing(accn) 전체**에 붙는
   값이다. 같은 accn 안에 전혀 다른 기간을 가리키는 행이 같은 ``fp``를 달고
   같이 들어온다 — 예: ``form=10-K, fp=FY``인 ``NetIncomeLoss`` 447,746행 중
   **48.6%(226,243행)가 span(=end−start) 315\\~410일(약 1년) 밖이다.** 실측 최빈값은
   90일(분기 노트나 footnote 비교치로 보인다) — 10-K 한 건에 "분기별 요약" 각주가
   함께 XBRL 태깅되면 그 분기값도 같은 ``fp=FY``로 들어온다. 극단값은 span 1일
   (하루짜리 이벤트성 수치)부터 30,997일(기원이 불명확한 "설립 이후 누계"로 보이는
   오염)까지 있었다.
2. 그래서 **``fp`` 와 ``span`` 을 같이 봐야 한다.** 어느 한쪽만으로는 가려지지
   않는다. 아래 표가 그 결합 규칙이다. 10-Q의 ``fp=Q2``인데 span이 90일대인
   행(그 분기만 따로 낸 값)과 180일대인 행(반기 누적)이 실측으로 둘 다
   흔하다 — 회사마다 어느 쪽을 태깅하는지 다르다. 두 값을 굳이 가려 쓰지 않고
   **항상 "회계연도 시작부터의 누적값" 쪽(span이 넓은 쪽)만 골라 인접 분기와
   빼서 분기값을 만든다** — 그러면 회사가 분기 단독값을 따로 안 냈어도(즉
   span 90일대 행이 없어도) 항상 계산할 수 있다.
3. 이 결합 규칙(``fp`` 일치 + span 허용범위)을 걸면 AAPL·NetIncomeLoss FY2020\\~
   FY2024 5개 회계연도, AAPL·MSFT 두 태그(순이익·매출)에서 **잔차(Q1+Q2+Q3+Q4−FY)가
   전부 정확히 0**이었다 — 아래 §PIT 검산.

| ``fp`` | 의미 | span 허용(일) — 분기 경계는 ~91일 |
|---|---|---|
| ``Q1`` | 회계연도 1분기 누적(=1분기 그 자체) | 45\\~135 |
| ``Q2`` | 회계연도 반기 누적 | 135\\~225 |
| ``Q3`` | 회계연도 3분기 누적 | 225\\~315 |
| ``FY`` | 회계연도 전체(10-K) | 315\\~410 |

경계값은 느슨하게 잡았다(±45일) — 52/53주 회계연도, 인수합병 직후 변칙
회계연도 같은 정상적인 변형을 걸러내지 않기 위해서다. 실측으로 315\\~410
구간의 span 분포는 365일 근방에 매끈하게 몰려 있고 경계 부근에 이상한 뭉침이
없었다(양끝 각각 연 20\\~30건 수준) — 이 범위가 인위적으로 자르는 게 아니라는
뜻이다.

**Q4는 절대 직접 공시되지 않는다.** 10-K는 연간값만 내고 10-Q는 4분기를
만들지 않는다. 그래서 ``Q4 = FY − (Q1 누적+2분기 차이+3분기 차이) = FY − 9개월
누적``으로만 얻을 수 있다 — 이게 ``06`` M2가 "M2에서 제일 어려운 로직"이라고
지목한 자리다.

**PIT.** 같은 (cik, tag, start, end) 조합에 여럿이면 ``filed``가 최신인 것을
쓴다(정정본이 원본을 덮는다, ``01_data_readiness.md`` §6). 계획 문서 §1은 이
키를 (cik, tag, end)로 적었지만, 그러면 같은 ``end``를 갖는 "그 분기만"과
"그 분기까지 누적"이 서로 다른 값인데도 섞여 버린다 — 그래서 여기서는
(cik, tag, start, end)로 정밀화했다(§7 안 맞거나 판단이 필요한 것 참고).

각 (cik, tag가 고른 태그, fy 앵커=start) 아래 최근 4분기가 서로 인접하지
않으면(중간에 빠진 분기가 있으면) TTM을 내지 않는다 — 빈 분기를 건너뛰고
합치면 12개월이 아닌 다른 기간이 되기 때문이다.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from modeler.us.lake import UsLake

#: 매출 태그 폴백 우선순위 — ``Revenues``가 없으면 ASC 606 이후 태그로 갈아탄
#: filer의 값을 쓴다(``01_data_readiness.md`` §3·§6). F5(sp_ttm)·F6(opm_ttm·gpa)가
#: 공유한다.
REVENUE_TAGS: tuple[str, ...] = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")

#: fp -> (qseq, span 하한, span 상한). 모듈 docstring 표 그대로.
_QUARTER_SPAN_DAYS: dict[str, tuple[int, int, int]] = {
    "Q1": (1, 45, 135),
    "Q2": (2, 135, 225),
    "Q3": (3, 225, 315),
    "FY": (4, 315, 410),
}

#: 최근 4분기가 "인접하다"고 볼 end 간격(가장 이른 것과 가장 늦은 것의 일수 차).
#: 4분기 = 약 3×91일 간격이므로 200~420일이면 정상 범위다. 이 밖이면 중간에
#: 빠진 분기가 있다는 뜻이라 TTM을 내지 않는다.
_TTM_SPAN_DAYS = (200, 420)


def _qseq_expr() -> pl.Expr:
    span = (pl.col("end") - pl.col("start")).dt.total_days()
    return (
        pl.when((pl.col("fp") == "Q1") & span.is_between(45, 135))
        .then(pl.lit(1))
        .when((pl.col("fp") == "Q2") & span.is_between(135, 225))
        .then(pl.lit(2))
        .when((pl.col("fp") == "Q3") & span.is_between(225, 315))
        .then(pl.lit(3))
        .when((pl.col("fp") == "FY") & span.is_between(315, 410))
        .then(pl.lit(4))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("qseq")
    )


def _dates_cik(panel: pl.DataFrame) -> pl.DataFrame:
    """패널의 (date, cik) 고유 조합. ``cik``가 없는 행(23%, 01 §2.1)은 뺀다 —
    join에서 자연히 매치되지 않아 결과가 null(→ ``_isna``)이 되지만, 후보
    테이블을 줄여 두는 편이 join 비용이 싸다."""
    return panel.select("date", "cik").filter(pl.col("cik").is_not_null()).unique()


def _quarter_facts(lake: UsLake, tags: Sequence[str]) -> pl.DataFrame:
    """분기 판정을 마친 (cik, start, end, qseq, filed, val) 후보.

    ``tags``는 폴백 우선순위(앞이 우선)다 — 매출의 ``Revenues`` ∪
    ``RevenueFromContractWithCustomerExcludingAssessedTax``처럼, 같은
    (cik, start, end, qseq)에 두 태그가 다 있으면 앞쪽 태그를 쓴다.

    같은 슬롯(cik, start, end, qseq)에 여러 ``filed``가 남아 있을 수 있다
    (정정본) — 여기서는 지우지 않는다. "이 시점에 무엇이 보였나"는 호출자가
    ``t``별로 고른다(``_asof_latest``).
    """
    lf = lake.scan("fundamentals").filter(
        pl.col("tag").is_in(list(tags)) & pl.col("start").is_not_null()
    )
    df = (
        lf.select("cik", "tag", "fp", "start", "end", "val", "filed")
        .with_columns(_qseq_expr())
        .filter(pl.col("qseq").is_not_null())
        .collect()
    )
    if df.is_empty():
        return df.select("cik", "start", "end", "qseq", "filed", "val")

    tag_rank = {tag: rank for rank, tag in enumerate(tags)}
    df = df.with_columns(
        pl.col("tag").replace_strict(tag_rank, default=len(tags)).alias("_tag_rank")
    )
    best_rank = df.group_by(["cik", "start", "end", "qseq"]).agg(
        pl.col("_tag_rank").min().alias("_best_rank")
    )
    df = df.join(best_rank, on=["cik", "start", "end", "qseq"], how="left").filter(
        pl.col("_tag_rank") == pl.col("_best_rank")
    )
    return df.select("cik", "start", "end", "qseq", "filed", "val")


def _asof_latest(dates_cik: pl.DataFrame, facts: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    """``dates_cik``(date, cik)마다, ``key_cols``로 식별되는 슬롯별로
    ``filed <= date``인 것 중 ``filed``가 최신인 행을 고른다.

    ``dates_cik``와 ``facts``를 ``cik``로 join한 뒤(같은 회사의 전체 신고
    이력이 각 날짜에 다 붙는다) ``filed <= date``로 자르고, 슬롯별 최신을
    남긴다 — 이것이 PIT 규칙("filed 최신인 것을 쓴다")의 구현이다.
    """
    if facts.is_empty() or dates_cik.is_empty():
        return facts.clear(0).join(dates_cik.clear(0), on="cik", how="inner")

    candidates = dates_cik.join(facts, on="cik", how="inner").filter(
        pl.col("filed") <= pl.col("date")
    )
    candidates = candidates.sort(["date", "cik", *key_cols, "filed"])
    return candidates.group_by(["date", "cik", *key_cols], maintain_order=True).last()


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"date": pl.Date, "symbol": pl.String, "value": pl.Float64, "isna": pl.Boolean}
    )


def flow_ttm(panel: pl.DataFrame, lake: UsLake, tags: Sequence[str]) -> pl.DataFrame:
    """(date, symbol)별 trailing-4-quarter 합.

    ``tags``는 폴백 우선순위 리스트(앞이 우선) — 매출처럼 태그가 갈리는
    경우에 쓴다. 단일 태그면 길이 1 리스트를 준다.

    반환: ``date, symbol, value, isna``. ``value``는 4분기가 인접하게
    갖춰지지 않으면(워밍업·결측·중간에 빈 분기) null이고 ``isna``가 True다.
    """
    dates_cik = _dates_cik(panel)
    facts = _quarter_facts(lake, tags)
    best = _asof_latest(dates_cik, facts, ["start", "end", "qseq"])
    if not best.is_empty():
        # 드물게 같은 (date, cik, start, qseq)에 end가 다른 두 슬롯이 남을 수
        # 있다(예: 회계연도 끝이 며칠 밀린 변칙 신고) — pivot이 셀당 값 하나를
        # 요구하므로, 더 넓은(가장 늦은 end) 쪽을 대표값으로 남긴다.
        best = (
            best.sort(["date", "cik", "start", "qseq", "end"])
            .group_by(["date", "cik", "start", "qseq"], maintain_order=True)
            .last()
        )

    base_cols = ["date", "cik", "start"]
    if best.is_empty():
        combined = pl.DataFrame(
            schema={
                "date": pl.Date,
                "cik": pl.Int64,
                "start": pl.Date,
                "q1cum": pl.Float64,
                "q2cum": pl.Float64,
                "q3cum": pl.Float64,
                "fy_val": pl.Float64,
                "end1": pl.Date,
                "end2": pl.Date,
                "end3": pl.Date,
                "end4": pl.Date,
            }
        )
    else:
        val_piv = best.pivot(values="val", index=base_cols, on="qseq")
        end_piv = best.pivot(values="end", index=base_cols, on="qseq")
        val_names = {"1": "q1cum", "2": "q2cum", "3": "q3cum", "4": "fy_val"}
        end_names = {"1": "end1", "2": "end2", "3": "end3", "4": "end4"}
        for src, dst in val_names.items():
            val_piv = (
                val_piv.rename({src: dst})
                if src in val_piv.columns
                else val_piv.with_columns(pl.lit(None, dtype=pl.Float64).alias(dst))
            )
        for src, dst in end_names.items():
            end_piv = (
                end_piv.rename({src: dst})
                if src in end_piv.columns
                else end_piv.with_columns(pl.lit(None, dtype=pl.Date).alias(dst))
            )
        combined = val_piv.join(end_piv, on=base_cols, how="left")

    combined = combined.with_columns(
        pl.col("q1cum").alias("q1"),
        (pl.col("q2cum") - pl.col("q1cum")).alias("q2"),
        (pl.col("q3cum") - pl.col("q2cum")).alias("q3"),
        (pl.col("fy_val") - pl.col("q3cum")).alias("q4"),
    )

    long = pl.concat(
        [
            combined.select(
                "date", "cik", pl.col(end_col).alias("q_end"), pl.col(val_col).alias("q_val")
            )
            for val_col, end_col in [("q1", "end1"), ("q2", "end2"), ("q3", "end3"), ("q4", "end4")]
        ]
    ).filter(pl.col("q_val").is_not_null() & pl.col("q_end").is_not_null())

    if long.is_empty():
        ttm = pl.DataFrame(schema={"date": pl.Date, "cik": pl.Int64, "value": pl.Float64})
    else:
        long = long.sort(["date", "cik", "q_end"], descending=[False, False, True])
        top4 = long.group_by(["date", "cik"], maintain_order=True).head(4)
        agg = top4.group_by(["date", "cik"]).agg(
            pl.col("q_val").sum().alias("value"),
            pl.col("q_end").max().alias("_max_end"),
            pl.col("q_end").min().alias("_min_end"),
            pl.len().alias("_n"),
        )
        lo, hi = _TTM_SPAN_DAYS
        ttm = agg.with_columns(
            pl.when(
                (pl.col("_n") == 4)
                & (pl.col("_max_end") - pl.col("_min_end")).dt.total_days().is_between(lo, hi)
            )
            .then(pl.col("value"))
            .otherwise(None)
            .alias("value")
        ).select("date", "cik", "value")

    result = panel.select("date", "symbol", "cik").join(ttm, on=["date", "cik"], how="left")
    return result.with_columns(pl.col("value").is_null().alias("isna")).select(
        "date", "symbol", "value", "isna"
    )


def _instant_facts(lake: UsLake, tag: str) -> pl.DataFrame:
    """(cik, end, filed, val) — 잔액표(순간값) 태그. ``start``가 null인 행만이다."""
    lf = lake.scan("fundamentals").filter((pl.col("tag") == tag) & pl.col("start").is_null())
    return lf.select("cik", "end", "val", "filed").collect()


def instant_latest(panel: pl.DataFrame, lake: UsLake, tag: str) -> pl.DataFrame:
    """(date, symbol)별 t 시점에 알려진 가장 최근 보고기간의 잔액표 값.

    반환: ``date, symbol, value, isna``.
    """
    dates_cik = _dates_cik(panel)
    facts = _instant_facts(lake, tag)
    best = _asof_latest(dates_cik, facts, ["end"])
    if best.is_empty():
        latest = pl.DataFrame(schema={"date": pl.Date, "cik": pl.Int64, "value": pl.Float64})
    else:
        latest = (
            best.sort(["date", "cik", "end"])
            .group_by(["date", "cik"], maintain_order=True)
            .last()
            .select("date", "cik", pl.col("val").alias("value"))
        )
    result = panel.select("date", "symbol", "cik").join(latest, on=["date", "cik"], how="left")
    return result.with_columns(pl.col("value").is_null().alias("isna")).select(
        "date", "symbol", "value", "isna"
    )


def instant_yoy_pair(panel: pl.DataFrame, lake: UsLake, tag: str) -> pl.DataFrame:
    """(date, symbol)별 최신 잔액표 값과 그 약 1년 전 값.

    "1년 전"은 달력 365일이 아니라, 최신 보고기간(``end``)보다 300\\~430일
    이른 것 중 가장 가까운(가장 늦은) 보고기간이다 — 회사마다 회계연도
    끝이 달라 정확히 365일이 아닐 수 있다.

    반환: ``date, symbol, cur_val, prior_val`` — 성장률·변화율은 호출자가
    계산한다(분모가 0이거나 음수인 경우의 처리가 피쳐마다 다를 수 있어서다).
    """
    dates_cik = _dates_cik(panel)
    facts = _instant_facts(lake, tag)
    best = _asof_latest(dates_cik, facts, ["end"])
    if best.is_empty():
        current = pl.DataFrame(
            schema={"date": pl.Date, "cik": pl.Int64, "cur_end": pl.Date, "cur_val": pl.Float64}
        )
        prior = pl.DataFrame(schema={"date": pl.Date, "cik": pl.Int64, "prior_val": pl.Float64})
    else:
        current = (
            best.sort(["date", "cik", "end"])
            .group_by(["date", "cik"], maintain_order=True)
            .last()
            .select("date", "cik", pl.col("end").alias("cur_end"), pl.col("val").alias("cur_val"))
        )
        joined = current.join(
            best.select("date", "cik", "end", "val"), on=["date", "cik"], how="inner"
        ).filter((pl.col("cur_end") - pl.col("end")).dt.total_days().is_between(300, 430))
        prior = (
            joined.sort(["date", "cik", "end"])
            .group_by(["date", "cik"], maintain_order=True)
            .last()
            .select("date", "cik", pl.col("val").alias("prior_val"))
        )
    out = current.join(prior, on=["date", "cik"], how="left")
    result = panel.select("date", "symbol", "cik").join(out, on=["date", "cik"], how="left")
    return result.select("date", "symbol", "cur_val", "prior_val")


def market_cap(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """(date, symbol)별 시총 근사 — ``EntityCommonStockSharesOutstanding``(최신
    ``filed <= t``) × 그날 원시 종가(``panel.close``). 분기 계단이 있으므로
    (``07_risks.md`` Y6) 절대값 피쳐로 쓰지 않는다 — 순위화는 호출자 몫이다.

    반환: ``date, symbol, mcap, isna``.
    """
    shares = instant_latest(panel, lake, "EntityCommonStockSharesOutstanding")
    out = panel.select("date", "symbol", "close").join(
        shares.select("date", "symbol", "value"), on=["date", "symbol"], how="left"
    )
    out = out.with_columns((pl.col("value") * pl.col("close")).alias("mcap"))
    return out.with_columns(pl.col("mcap").is_null().alias("isna")).select(
        "date", "symbol", "mcap", "isna"
    )


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """분모가 null이거나 0이면 null. 둘 다 있으면 numerator/denominator."""
    return (
        pl.when(denominator.is_null() | numerator.is_null() | (denominator == 0))
        .then(None)
        .otherwise(numerator / denominator)
    )
