"""동점 순서를 고정한다 (2026-09-22 · 3차 `01_tie_break.md`).

**`rank(method="ordinal")` 은 동점을 행 순서로 가르는데 polars 가 그 순서를
보장하지 않는다.** 같은 코드·같은 데이터·같은 시드로 두 번 돌렸더니
`filing_lag` 의 placebo 가 1.7648 → 1.5570, `placebo_p` 가 0.098 → 0.118 로
달라졌다. 등급은 우연히 안 바뀌었지만 G3 문턱(0.05) 근처였다면 뒤집힌다.

그리고 44개 중 **23개가 동점투성이**다 — `iv_isna` 는 동점 묶음이 평균
**2,010종목**이라 "top-100" 이 그 중 아무 100개였다.

시험이 지키는 것 셋:

1. **입력 행 순서를 섞어도 답이 같다** — 이게 핵심이다
2. **키가 종목마다 고정이다** — 달마다 새로 뽑으면 회전율이 95% 가 되어
   G1(비용)이 전부 죽는다
3. 동점이 없으면 옛 동작과 같다 — 고친 것이 동점 처리뿐이다
"""

from __future__ import annotations

import polars as pl

from modeler.us import scan_long


def _frame(n_per_month: int = 300, months: int = 4) -> pl.DataFrame:
    """값이 셋뿐인 피쳐 — 한 달에 100종목씩 동점으로 묶인다."""
    rows = []
    for m in range(months):
        for i in range(n_per_month):
            rows.append(
                {
                    "month_idx": m,
                    "symbol": f"SYM{i:04d}",
                    "f": float(i % 3),  # 값 셋 → 동점 묶음 100개씩
                    "L0": (i * 7 + m * 13) % 100 / 1000.0,
                }
            )
    return pl.DataFrame(rows)


def test_shuffling_the_input_does_not_change_the_answer():
    """**핵심.** 같은 데이터를 다른 순서로 주면 같은 답이 나와야 한다."""
    base = _frame()
    out = None
    for seed in (0, 1, 2, 3, 4):
        shuffled = base.sample(fraction=1.0, shuffle=True, seed=seed)
        got = scan_long.long_short_monthly(
            shuffled, feature_col="f", sign="+", value_col="L0", min_names=10, top_k=100
        ).sort("month_idx")
        if out is None:
            out = got
        else:
            assert got.equals(out), f"seed={seed} 에서 답이 달라졌다"


def test_tie_key_is_fixed_per_symbol_not_per_row():
    """**달마다 새로 뽑으면 안 된다.** 회전율이 95% 가 되어 G1 이 다 죽는다."""
    a = scan_long.with_tie_break(pl.DataFrame({"symbol": ["AAA", "BBB", "CCC"]}))
    b = scan_long.with_tie_break(pl.DataFrame({"symbol": ["CCC", "AAA", "BBB"]}))
    m_a = dict(zip(a["symbol"].to_list(), a["_tie"].to_list(), strict=True))
    m_b = dict(zip(b["symbol"].to_list(), b["_tie"].to_list(), strict=True))
    assert m_a == m_b


def test_tie_break_is_stable_across_processes():
    """`hashlib` 이라 판·플랫폼이 바뀌어도 같다. polars `.hash()` 는 보장이 없다."""
    assert scan_long._tie_key("AAPL") == scan_long._tie_key("AAPL")
    assert scan_long._tie_key("AAPL") != scan_long._tie_key("MSFT")


def test_with_tie_break_is_idempotent():
    df = scan_long.with_tie_break(pl.DataFrame({"symbol": ["AAA", "BBB"]}))
    again = scan_long.with_tie_break(df)
    assert again.equals(df)


def test_basket_is_the_same_hundred_names_every_month_when_the_group_is_stable():
    """동점 묶음이 안 변하면 **바스켓도 안 변한다** — 회전율이 낮게 유지된다."""
    base = _frame(n_per_month=300, months=3)
    ranked = scan_long.ordinal_rank_stable(
        base.with_columns(scan_long.scored_column("f", "+").alias("_score"))
    )
    picked = {
        m: set(
            ranked.filter((pl.col("month_idx") == m) & (pl.col("_r") > pl.col("_n") - 100))[
                "symbol"
            ].to_list()
        )
        for m in range(3)
    }
    assert picked[0] == picked[1] == picked[2]
    assert len(picked[0]) == 100


def test_no_ties_behaves_like_plain_ordinal_rank():
    """동점이 없으면 고치기 전과 같다 — 바꾼 것이 동점 처리뿐이다."""
    df = pl.DataFrame(
        {
            "month_idx": [0] * 5,
            "symbol": [f"S{i}" for i in range(5)],
            "_score": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
    )
    got = scan_long.ordinal_rank_stable(df).sort("symbol")
    want = df.with_columns(
        pl.col("_score").rank(method="ordinal").over("month_idx").cast(pl.Int64).alias("_r")
    ).sort("symbol")
    assert got["_r"].to_list() == want["_r"].to_list()


def test_rank_covers_every_row_exactly_once():
    base = _frame(n_per_month=50, months=2)
    ranked = scan_long.ordinal_rank_stable(
        base.with_columns(scan_long.scored_column("f", "+").alias("_score"))
    )
    for m in range(2):
        rs = sorted(ranked.filter(pl.col("month_idx") == m)["_r"].to_list())
        assert rs == list(range(1, 51))
