"""Label-free technical checks of the P6′ preregistration draft §3 (T1-T5, T8, T9).

Every check reads input marts or serving-mode panels (no label, no return, no score) and returns
a JSON-able dict with the numbers and a ``pass`` flag. The CLI writes them to one report file.

* **T1** :func:`pit_cut` -- cut the fact at ``disclosed_date <= T`` and the receipt list at the
  receipt date ``<= T``, rebuild the placement audit and the mart, and compare with the full build:
  the fiscal placement of every cut row must equal the full build's, every cut mart row must exist
  in the full mart, a full mart row available by ``T`` must exist in the cut mart, the placing
  columns (:data:`IDENTITY_COLS`) must be equal, and a row may differ only by what a vintage
  disclosed after ``T`` adds: a TTM window or year-ago value completed by a later neighbour (the
  full build's ``ttm_available_from`` after ``T``; the year-ago row available after ``T``), or a
  Q4 standalone that was NULL at ``T`` for want of the Q3 cumulative
  (``q4_missing_q3_cumulative``) and is completed later (``available_from`` after ``T``).
* **T2** :func:`december_attribution` -- December companies: every row that differs from P6's
  ``_dlr`` is attributed to KR-FB-0002 (v3 fact), the quarterly statement v1 -> v2, KR-FB-0004
  (1e14 block) or KR-FB-0003 (placement/TTM), through the intermediate marts; "other" must be 0.
* **T3** :func:`non_december` -- non-December companies: TTM windows over non-consecutive period
  ends and Q4 rows whose period end is not the fiscal year end must be 0.
* **T4** :func:`outlier_rows` -- duration standalone values with ``abs > 1e14`` must be 0.
* **T5** :func:`r1_conflicts` -- R1: conflicting Q2/Q3 keep their value; NULL only for 1e14.
* **T8** :func:`compare_serving_panels` -- P6 and P6′ serving panels: same keys, same values in
  the 21 non-financial feature columns and the universe/notice columns (floats within the bundle's
  golden tolerance, NULL pattern exact), and per financial column the share of rows that changed
  (recorded, not judged).
* **T9** :func:`missing_gap` -- P6 §7.5 rule M-3: missing rate of later-delisted tickers
  (``stock_master`` DELISTED and never ACTIVE) minus the rest, per feature, in percentage points;
  a column over :data:`feature_set.MISSING_GAP_PP` is reported (the features are not changed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.marts import financial_quarters_v4p as fq4p
from modeler.etl.marts import metric_vintages_delisted as dl
from modeler.models._03_kr_next import build_panel as p6_bp
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import views as p6_views
from modeler.models._04_kr_next_p6prime import inputs

PIT_CUT_DATES: tuple[str, ...] = (
    "2016-06-30",
    "2018-03-31",
    "2020-12-31",
    "2022-08-31",
    "2024-05-31",
)
#: rows of the P6 mart contract (the 22 columns minus the key), compared in T2
P6_VALUE_COLS: tuple[str, ...] = (
    "market", "corp_code", "quarter_ordinal", "seq_key", "statement_period_end", "rcept_no",
    "available_from", "metric_kind", "standalone_value", "standalone_source_conflict",
    "cumulative_derived_value", "comparative_q_amount", "value_lag_4q", "ttm_value",
    "ttm_complete", "ttm_available_from", "ttm_includes_conflict",
)  # fmt: skip
P6_KEY: tuple[str, ...] = ("ticker", "metric_code", "fs_basis", "bsns_year", "quarter")
MART_KEY: tuple[str, ...] = ("ticker", "metric_code", "fs_basis", "seq_key")
AUDIT_KEY: tuple[str, ...] = (
    "ticker", "metric_code", "fs_basis", "rcept_no", "source_key", "reprt_code",
)  # fmt: skip
#: mart columns a later neighbour can complete (TTM window, the year-ago value)
NEIGHBOUR_COLS: tuple[str, ...] = (
    "value_lag_4q", "ttm_value", "ttm_complete", "ttm_available_from", "ttm_includes_conflict",
)  # fmt: skip
CUT_COMPARE_COLS: tuple[str, ...] = (
    "bsns_year", "quarter", "fiscal_year_end", "statement_period_end", "rcept_no",
    "available_from", "standalone_value", "standalone_null_reason", *NEIGHBOUR_COLS,
)  # fmt: skip
#: mart columns that place a row; they must be equal between a cut build and the full build
IDENTITY_COLS: tuple[str, ...] = (
    "bsns_year", "quarter", "fiscal_year_end", "statement_period_end", "rcept_no",
)  # fmt: skip
Q4_MISSING_Q3 = "q4_missing_q3_cumulative"
#: float comparison of T8 = the bundle's golden tolerance
FLOAT_RTOL = bundle_mod.GOLDEN_RTOL
FLOAT_ATOL = bundle_mod.GOLDEN_ATOL
DURATION_KINDS: tuple[str, ...] = ("direct_interim", "cumulative_reported")
FIN_FEATURES: tuple[str, ...] = tuple(c for c in fs.FIN_COLS if c.startswith("fin_"))
NON_FIN_FEATURES: tuple[str, ...] = tuple(c for c in fs.NOFLOW_COLS if c not in FIN_FEATURES)
_RECEIPT_DATE = (
    "CASE WHEN rcept_no ~ '^[0-9]{14}$' THEN strptime(left(rcept_no, 8), '%Y%m%d')::DATE "
    "ELSE rcept_dt END"
)
_ANNUAL_NAME = r"^(\[[^\]]*\])?\s*사업보고서\s*\((\d{4})\.(\d{2})\)\s*$"


def _lit(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _row_md5(cols: Iterable[str], alias: str = "") -> str:
    a = f"{alias}." if alias else ""
    parts = ", ".join(f"COALESCE(CAST({a}{c} AS VARCHAR), chr(0))" for c in cols)
    return f"md5(concat_ws(chr(31), {parts}))"


def _one(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [
        {k: (v.isoformat() if isinstance(v, date) else v) for k, v in zip(names, r, strict=True)}
        for r in cur.fetchall()
    ]


# --------------------------------------------------------------------------- T1


def pit_cut(
    con: duckdb.DuckDBPyConnection,
    *,
    fact_view: str,
    receipt_view: str,
    mart_view: str,
    cut_dates: Sequence[str] = PIT_CUT_DATES,
) -> dict[str, Any]:
    """T1. ``mart_view`` is the full v4p mart built from ``fact_view`` and ``receipt_view``."""
    key = ", ".join(AUDIT_KEY)
    audit = fq4p.build_pit_fiscal_position_audit_sql
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE _t1_full_audit AS SELECT {key}, fiscal_position_status st, "
        f"fiscal_year_end fye, fiscal_reprt_code rc FROM ("
        f"{audit(vintage_view=fact_view, filing_receipt_view=receipt_view)})"
    )
    mkey = ", ".join(MART_KEY)
    per_date: list[dict[str, Any]] = []
    for t in cut_dates:
        day = f"DATE '{date.fromisoformat(t).isoformat()}'"
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW _t1_fact AS "
            f"SELECT * FROM {fact_view} WHERE disclosed_date <= {day}"
        )
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW _t1_rcpt AS "
            f"SELECT * FROM {receipt_view} WHERE {_RECEIPT_DATE} <= {day}"
        )
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE _t1_cut_audit AS SELECT {key}, fiscal_position_status "
            f"st, fiscal_year_end fye, fiscal_reprt_code rc FROM ("
            f"{audit(vintage_view='_t1_fact', filing_receipt_view='_t1_rcpt')})"
        )
        a_rows, a_missing, a_diff = con.execute(
            f"""SELECT count(*), count(*) FILTER (WHERE f.st IS NULL),
                count(*) FILTER (WHERE f.st IS NOT NULL AND (f.st <> x.st
                    OR f.fye IS DISTINCT FROM x.fye OR f.rc IS DISTINCT FROM x.rc))
            FROM _t1_cut_audit x LEFT JOIN _t1_full_audit f USING ({key})"""
        ).fetchone()
        sql = fq4p.build_fin_quarterly_metric_vintage_dlr_v4p_sql(
            vintage_view="_t1_fact", filing_receipt_view="_t1_rcpt"
        )
        con.execute(f"CREATE OR REPLACE TEMP TABLE _t1_cut AS {sql}")

        def any_diff(cs: Iterable[str]) -> str:
            return "(" + " OR ".join(f"x.{c} IS DISTINCT FROM f.{c}" for c in cs) + ")"

        differ = any_diff(CUT_COMPARE_COLS)
        ident = any_diff(IDENTITY_COLS)
        value = any_diff(c for c in CUT_COMPARE_COLS if c not in (*IDENTITY_COLS, *NEIGHBOUR_COLS))
        ttm = any_diff(c for c in NEIGHBOUR_COLS if c != "value_lag_4q")
        lag = "(x.value_lag_4q IS DISTINCT FROM f.value_lag_4q)"
        # a Q4 standalone that was NULL at T for want of the Q3 cumulative and that the full
        # build completes with a later vintage: its availability is that later date
        q4_later = (
            f"(x.standalone_value IS NULL AND x.standalone_null_reason = {_lit(Q4_MISSING_Q3)} "
            f"AND f.standalone_value IS NOT NULL AND f.available_from > {day})"
        )
        # the year-ago row of the full build (its availability decides whether the lag was known)
        prior_known = (
            f"EXISTS (SELECT 1 FROM {mart_view} p WHERE p.ticker = f.ticker "
            f"AND p.metric_code = f.metric_code AND p.fs_basis = f.fs_basis "
            f"AND p.seq_key = f.seq_key - 4 AND p.available_from <= {day})"
        )
        both = "f.ticker IS NOT NULL"
        m = _rows(
            con,
            f"""SELECT count(*) AS cut_rows,
                count(*) FILTER (WHERE f.ticker IS NULL) AS cut_rows_not_in_full,
                count(*) FILTER (WHERE {both} AND {differ}) AS rows_differ,
                count(*) FILTER (WHERE {both} AND {ident}) AS identity_differs,
                count(*) FILTER (WHERE {both} AND {value} AND {q4_later})
                    AS q4_completed_after_t,
                count(*) FILTER (WHERE {both} AND {value} AND NOT {q4_later})
                    AS value_differs_unexplained,
                count(*) FILTER (WHERE {both} AND {ttm}) AS ttm_differs,
                count(*) FILTER (WHERE {both} AND {ttm}
                                 AND NOT COALESCE(f.ttm_available_from > {day}, FALSE))
                    AS ttm_differs_full_known_by_t,
                count(*) FILTER (WHERE {both} AND {lag}) AS lag_differs,
                count(*) FILTER (WHERE {both} AND {lag} AND {prior_known})
                    AS lag_differs_prior_known_by_t
            FROM _t1_cut x LEFT JOIN {mart_view} f USING ({mkey})""",
        )[0]
        full_known_missing = _one(
            con,
            f"""SELECT count(*) FROM {mart_view} f LEFT JOIN _t1_cut x USING ({mkey})
            WHERE f.available_from <= {day} AND x.ticker IS NULL""",
        )
        ok = (
            a_missing == 0
            and a_diff == 0
            and m["cut_rows_not_in_full"] == 0
            and m["identity_differs"] == 0
            and m["value_differs_unexplained"] == 0
            and m["ttm_differs_full_known_by_t"] == 0
            and m["lag_differs_prior_known_by_t"] == 0
            and full_known_missing == 0
        )
        per_date.append(
            {
                "t": t,
                "audit_rows": a_rows,
                "audit_rows_not_in_full": a_missing,
                "placement_differs": a_diff,
                **m,
                "full_rows_available_by_t_missing_in_cut": full_known_missing,
                "pass": ok,
            }
        )
    return {"per_date": per_date, "pass": all(d["pass"] for d in per_date)}


# --------------------------------------------------------------------------- T2


def _annual_name_class_sql(receipt_view: str) -> str:
    """corp_code -> 'dec_only' | 'nondec_constant' | 'changed_or_short' (annual-report names)."""
    return f"""
    WITH n AS (
        SELECT corp_code, last_day(make_date(
            CAST(regexp_extract(report_nm, '\\((\\d{{4}})\\.(\\d{{2}})\\)', 1) AS INT),
            CAST(regexp_extract(report_nm, '\\((\\d{{4}})\\.(\\d{{2}})\\)', 2) AS INT), 1)) fye
        FROM {receipt_view} WHERE regexp_matches(report_nm, '{_ANNUAL_NAME}') GROUP BY ALL),
    g AS (SELECT corp_code, fye, date_diff('month',
            lag(fye) OVER (PARTITION BY corp_code ORDER BY fye), fye) m FROM n)
    SELECT corp_code, CASE
        WHEN bool_and(month(fye) = 12) THEN 'dec_only'
        WHEN count(DISTINCT month(fye)) = 1 AND bool_and(m IS NULL OR m % 12 = 0)
            THEN 'nondec_constant'
        ELSE 'changed_or_short' END cls
    FROM g GROUP BY 1"""


def december_attribution(
    con: duckdb.DuckDBPyConnection,
    *,
    p6_dlr_view: str,
    fact_view: str,
    receipt_view: str,
    mart_view: str,
) -> dict[str, Any]:
    """T2. Attribute every December-company row that differs from P6 ``_dlr``.

    A December company: every annual-report name in December (or none), every v4p fiscal year
    end in December, every P6 Q4 period end in December. Variants on the v3 fact: B = ``_dlr``
    (statement v1), B2 = ``_dlr`` (statement v2), C = v4p without the 1e14 block, D = v4p. A row
    (P6 key) is 'same' when A = D, else the first step where it changes: A->B '0002_v3_fact',
    B->B2 'quarterly_v1_to_v2', C->D '0004_outlier_block', B2->C '0003_placement_ttm', else
    'other'.
    """
    variants = {
        "a": f"SELECT * FROM {p6_dlr_view}",
        "b": dl.build_fin_quarterly_metric_vintage_dlr_sql(vintage_view=fact_view, semantics="v1"),
        "b2": dl.build_fin_quarterly_metric_vintage_dlr_sql(vintage_view=fact_view, semantics="v2"),
        "c": fq4p.build_fin_quarterly_metric_vintage_dlr_v4p_sql(
            vintage_view=fact_view, filing_receipt_view=receipt_view, outlier_block=False
        ),
        "d": f"SELECT * FROM {mart_view}",
    }
    k = ", ".join(P6_KEY)
    for name, sql in variants.items():
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE _t2_h{name} AS "
            f"SELECT {k}, {_row_md5(P6_VALUE_COLS)} h FROM ({sql})"
        )
    con.execute(f"CREATE OR REPLACE TEMP TABLE _t2_cls AS {_annual_name_class_sql(receipt_view)}")
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE _t2_tk AS
        WITH t AS (SELECT DISTINCT ticker, corp_code FROM {p6_dlr_view}
                   UNION SELECT DISTINCT ticker, corp_code FROM {mart_view}),
        x_names AS (SELECT ticker FROM t LEFT JOIN _t2_cls USING (corp_code) GROUP BY ticker
                    HAVING bool_and(COALESCE(cls, 'no_annual_name')
                                    IN ('dec_only', 'no_annual_name'))),
        x_new AS (SELECT ticker FROM {mart_view} GROUP BY ticker
                  HAVING bool_and(month(fiscal_year_end) = 12)),
        x_p6 AS (SELECT ticker FROM {p6_dlr_view} GROUP BY ticker
                 HAVING bool_and(quarter <> 'Q4' OR month(statement_period_end) = 12))
        SELECT DISTINCT ticker, (ticker IN (SELECT * FROM x_names)
            AND ticker IN (SELECT * FROM x_new) AND ticker IN (SELECT * FROM x_p6)) is_dec
        FROM t"""
    )
    dup = {
        n: _one(
            con,
            f"SELECT count(*) FROM (SELECT {k} FROM _t2_h{n} JOIN _t2_tk USING (ticker) "
            f"WHERE is_dec GROUP BY ALL HAVING count(*) > 1)",
        )
        for n in variants
    }
    coalesced = ", ".join(f"COALESCE(ha.{c}, hd.{c}) AS {c}" for c in P6_KEY)
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE _t2_att AS
        SELECT {coalesced}, ha.h a, hb.h b, hb2.h b2, hc.h c, hd.h d
        FROM _t2_ha ha FULL JOIN _t2_hd hd USING ({k})
        LEFT JOIN _t2_hb hb USING ({k}) LEFT JOIN _t2_hb2 hb2 USING ({k})
        LEFT JOIN _t2_hc hc USING ({k})"""
    )
    by_cat = _rows(
        con,
        """SELECT cat, presence, count(*) n, count(DISTINCT ticker) tickers FROM (
            SELECT ticker, CASE
                WHEN a IS NOT NULL AND a = d THEN 'same'
                WHEN a IS DISTINCT FROM b THEN '0002_v3_fact'
                WHEN b IS DISTINCT FROM b2 THEN 'quarterly_v1_to_v2'
                WHEN c IS DISTINCT FROM d THEN '0004_outlier_block'
                WHEN b2 IS DISTINCT FROM c THEN '0003_placement_ttm'
                ELSE 'other' END cat,
              CASE WHEN a IS NULL THEN 'only_p6prime' WHEN d IS NULL THEN 'only_p6'
                   ELSE 'both' END presence
            FROM _t2_att JOIN _t2_tk USING (ticker) WHERE _t2_tk.is_dec)
        GROUP BY ALL ORDER BY cat, presence""",
    )
    totals: dict[str, int] = {}
    for r in by_cat:
        totals[r["cat"]] = totals.get(r["cat"], 0) + r["n"]
    tickers = con.execute(
        "SELECT count(*) FILTER (WHERE is_dec), count(*) FILTER (WHERE NOT is_dec) FROM _t2_tk"
    ).fetchone()
    return {
        "december_tickers": tickers[0],
        "other_tickers": tickers[1],
        "duplicate_p6_keys_in_december_rows": dup,
        "rows_by_cause": by_cat,
        "totals": totals,
        "pass": totals.get("other", 0) == 0 and not any(dup.values()),
    }


# --------------------------------------------------------------------------- T3-T5


def non_december(
    con: duckdb.DuckDBPyConnection, *, p6_dlr_view: str, mart_view: str
) -> dict[str, Any]:
    """T3. TTM over non-consecutive period ends; Q4 period end not the fiscal year end."""
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE _t3_tk AS
        SELECT DISTINCT ticker FROM {mart_view} WHERE month(fiscal_year_end) <> 12
        UNION SELECT DISTINCT ticker FROM {p6_dlr_view}
        WHERE quarter = 'Q4' AND month(statement_period_end) <> 12"""
    )
    q = """WITH x AS (SELECT a.ticker, a.statement_period_end pe, n1.statement_period_end p1,
            n3.statement_period_end p3 FROM {v} a
        LEFT JOIN {v} n1 ON n1.ticker = a.ticker AND n1.metric_code = a.metric_code
            AND n1.fs_basis = a.fs_basis AND n1.seq_key = a.seq_key - 1
        LEFT JOIN {v} n3 ON n3.ticker = a.ticker AND n3.metric_code = a.metric_code
            AND n3.fs_basis = a.fs_basis AND n3.seq_key = a.seq_key - 3
        WHERE a.ttm_complete AND a.ticker IN (SELECT ticker FROM _t3_tk))
        SELECT count(*) ttm_rows, count(*) FILTER (WHERE date_diff('month', p3, pe) <> 9
            OR date_diff('month', p1, pe) <> 3) non_consecutive FROM x"""
    p6 = _rows(con, q.format(v=p6_dlr_view))[0]
    new = _rows(con, q.format(v=mart_view))[0]
    p6_q4 = _one(
        con,
        f"SELECT count(*) FROM {p6_dlr_view} WHERE quarter = 'Q4' "
        f"AND month(statement_period_end) <> 12",
    )
    new_q4 = _one(
        con,
        f"SELECT count(*) FROM {mart_view} WHERE quarter = 'Q4' "
        f"AND statement_period_end <> fiscal_year_end",
    )
    return {
        "tickers": _one(con, "SELECT count(*) FROM _t3_tk"),
        "p6": {**p6, "q4_period_end_not_december": p6_q4},
        "p6prime": {**new, "q4_period_end_not_fiscal_year_end": new_q4},
        "pass": new["non_consecutive"] == 0 and new_q4 == 0,
    }


def outlier_rows(
    con: duckdb.DuckDBPyConnection, *, p6_dlr_view: str, mart_view: str
) -> dict[str, Any]:
    """T4. Standalone duration values with ``abs > 1e14`` (by ``metric_kind``)."""
    lim = fq4p.OUTLIER_ABS_LIMIT
    q = (
        "SELECT metric_kind, count(*) FILTER (WHERE abs(standalone_value) > {lim}) n "
        "FROM {v} GROUP BY 1 ORDER BY 1"
    )
    p6 = _rows(con, q.format(v=p6_dlr_view, lim=lim))
    new = _rows(con, q.format(v=mart_view, lim=lim))
    dur = sum(r["n"] for r in new if r["metric_kind"] in DURATION_KINDS)
    return {
        "limit": lim,
        "duration_kinds": list(DURATION_KINDS),
        "p6_by_kind": p6,
        "p6prime_by_kind": new,
        "p6_duration_rows": sum(r["n"] for r in p6 if r["metric_kind"] in DURATION_KINDS),
        "p6prime_duration_rows": dur,
        "pass": dur == 0,
    }


def r1_conflicts(con: duckdb.DuckDBPyConnection, *, mart_view: str) -> dict[str, Any]:
    """T5. Conflicting direct-interim quarters: kept value, NULL only with the 1e14 reason."""
    by_q = _rows(
        con,
        f"""SELECT quarter, count(*) n, count(*) FILTER (WHERE standalone_value IS NULL) n_null
        FROM {mart_view} WHERE standalone_source_conflict GROUP BY 1 ORDER BY 1""",
    )
    reasons = _rows(
        con,
        f"""SELECT quarter, standalone_null_reason, count(*) n FROM {mart_view}
        WHERE standalone_source_conflict AND standalone_value IS NULL GROUP BY ALL ORDER BY ALL""",
    )
    bad = sum(r["n"] for r in reasons if r["standalone_null_reason"] != fq4p.REASON_OUTLIER)
    other_q = sum(r["n"] for r in by_q if r["quarter"] not in ("Q2", "Q3"))
    return {
        "by_quarter": by_q,
        "null_reasons": reasons,
        "null_not_1e14": bad,
        "conflicts_outside_q2_q3": other_q,
        "pass": bad == 0,
    }


# --------------------------------------------------------------------------- T8, T9


def compare_serving_panels(
    con: duckdb.DuckDBPyConnection, p6_panel: str | Path, p6prime_panel: str | Path
) -> dict[str, Any]:
    """T8. Keys, non-financial columns and universe columns equal; changed share per fin col.

    Floating-point columns are compared exactly and within :data:`FLOAT_RTOL`/:data:`FLOAT_ATOL`
    (the bundle's golden tolerance): the panel builder is not bit-for-bit deterministic for every
    float column between runs (summation order of parallel aggregates), so "the same value" means
    the same NULL pattern and a difference inside the tolerance. Other columns must be equal.
    """
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW _t8_a AS SELECT * FROM read_parquet({_lit(p6_panel)})"
    )
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW _t8_b AS SELECT * FROM read_parquet({_lit(p6prime_panel)})"
    )
    ca = con.execute("DESCRIBE _t8_a").fetchall()
    cb = con.execute("DESCRIBE _t8_b").fetchall()
    cols = [r[0] for r in ca]
    floats = {r[0] for r in ca if r[1] in ("DOUBLE", "FLOAT")}
    key = ", ".join(ds.KEYS)
    rows_a = _one(con, "SELECT count(*) FROM _t8_a")
    rows_b = _one(con, "SELECT count(*) FROM _t8_b")
    a_only = _one(
        con, f"SELECT count(*) FROM (SELECT {key} FROM _t8_a EXCEPT SELECT {key} FROM _t8_b)"
    )
    b_only = _one(
        con, f"SELECT count(*) FROM (SELECT {key} FROM _t8_b EXCEPT SELECT {key} FROM _t8_a)"
    )
    dup = _one(con, f"SELECT count(*) - count(DISTINCT ({key})) FROM _t8_b")
    fin = [*FIN_FEATURES, *[c for c in ds.MART_EXTRA_COLS.get("feat_fin_v2", ()) if c in cols]]
    same_cols = [c for c in cols if c not in ds.KEYS and c not in fin]

    def beyond(c: str) -> str:
        if c not in floats:
            return f"a.{c} IS DISTINCT FROM b.{c}"
        return (
            f"((a.{c} IS NULL) <> (b.{c} IS NULL) OR "
            f"abs(a.{c} - b.{c}) > {FLOAT_ATOL!r} + {FLOAT_RTOL!r} * abs(a.{c}))"
        )

    parts = []
    for c in same_cols:
        parts.append(f"count(*) FILTER (WHERE a.{c} IS DISTINCT FROM b.{c}) AS {c}__exact")
        parts.append(f"count(*) FILTER (WHERE {beyond(c)}) AS {c}__tol")
        if c in floats:
            parts.append(f"max(abs(a.{c} - b.{c})) AS {c}__max_abs")
    r = _rows(con, f"SELECT {', '.join(parts)} FROM _t8_a a JOIN _t8_b b USING ({key})")[0]
    exact = {c: r[f"{c}__exact"] for c in same_cols}
    tol = {c: r[f"{c}__tol"] for c in same_cols}
    max_abs = {c: r[f"{c}__max_abs"] for c in same_cols if c in floats and exact[c]}
    fin_parts = []
    for c in fin:
        fin_parts += [
            f"count(*) FILTER (WHERE a.{c} IS DISTINCT FROM b.{c}) AS {c}__changed",
            f"count(*) FILTER (WHERE {beyond(c)}) AS {c}__changed_tol",
            f"count(*) FILTER (WHERE a.{c} IS NULL AND b.{c} IS NOT NULL) AS {c}__null_to_value",
            f"count(*) FILTER (WHERE a.{c} IS NOT NULL AND b.{c} IS NULL) AS {c}__value_to_null",
            f"avg((a.{c} IS NULL)::INT) AS {c}__p6_missing",
            f"avg((b.{c} IS NULL)::INT) AS {c}__p6prime_missing",
        ]
    f = _rows(
        con, f"SELECT count(*) AS n, {', '.join(fin_parts)} FROM _t8_a a JOIN _t8_b b USING ({key})"
    )[0]
    n = f["n"]
    fin_out = {
        c: {
            "changed_rows": f[f"{c}__changed"],
            "changed_share": f[f"{c}__changed"] / n if n else None,
            "changed_rows_beyond_tolerance": f[f"{c}__changed_tol"],
            "changed_share_beyond_tolerance": f[f"{c}__changed_tol"] / n if n else None,
            "null_to_value": f[f"{c}__null_to_value"],
            "value_to_null": f[f"{c}__value_to_null"],
            "p6_missing": f[f"{c}__p6_missing"],
            "p6prime_missing": f[f"{c}__p6prime_missing"],
        }
        for c in fin
    }
    schema_same = [(r[0], r[1]) for r in ca] == [(r[0], r[1]) for r in cb]
    keys_ok = schema_same and a_only == 0 and b_only == 0 and dup == 0 and rows_a == rows_b
    return {
        "rows": {"p6": rows_a, "p6prime": rows_b},
        "schema_same": schema_same,
        "keys_only_in_p6": a_only,
        "keys_only_in_p6prime": b_only,
        "duplicate_keys_p6prime": dup,
        "non_fin_columns_compared": same_cols,
        "float_tolerance": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL},
        "non_fin_rows_differ": exact,
        "non_fin_rows_differ_beyond_tolerance": tol,
        "non_fin_max_abs_diff": max_abs,
        "fin_columns": fin_out,
        "pass_exact": keys_ok and not any(exact.values()),
        "pass": keys_ok and not any(tol.values()),
    }


def missing_gap(
    con: duckdb.DuckDBPyConnection,
    panel: str | Path,
    stock_master_view: str,
    *,
    start: str = "2017-01-01",
    end: str = "2025-07-31",
    columns: Sequence[str] = fs.NOFLOW_COLS,
) -> dict[str, Any]:
    """T9. M-3: missing rate (later delisted) − (rest), percentage points, per column."""
    s, e = date.fromisoformat(start).isoformat(), date.fromisoformat(end).isoformat()
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE _t9 AS
        SELECT p.*, p.ticker IN (
            SELECT ticker FROM {stock_master_view} GROUP BY ticker
            HAVING bool_or(status = 'DELISTED') AND NOT bool_or(status = 'ACTIVE')
        ) AS later_delisted
        FROM read_parquet({_lit(panel)}) p
        WHERE trade_date BETWEEN DATE '{s}' AND DATE '{e}'"""
    )
    rows, n_del, tk_del = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE later_delisted), "
        "count(DISTINCT ticker) FILTER (WHERE later_delisted) FROM _t9"
    ).fetchone()
    parts = ", ".join(
        f"100 * (avg(({c} IS NULL)::INT) FILTER (WHERE later_delisted) - "
        f"avg(({c} IS NULL)::INT) FILTER (WHERE NOT later_delisted)) AS {c}"
        for c in columns
    )
    gap = _rows(con, f"SELECT {parts} FROM _t9")[0]
    by_year = _rows(con, f"SELECT year(trade_date) AS year, {parts} FROM _t9 GROUP BY 1 ORDER BY 1")
    yearly_max = {
        c: max((r[c] for r in by_year if r[c] is not None), default=None) for c in columns
    }
    over = [c for c in columns if gap[c] is not None and gap[c] > fs.MISSING_GAP_PP]
    return {
        "start": s,
        "end": e,
        "rows": rows,
        "later_delisted_rows": n_del,
        "later_delisted_share": n_del / rows if rows else None,
        "later_delisted_tickers": tk_del,
        "threshold_pp": fs.MISSING_GAP_PP,
        "gap_pp": gap,
        "yearly_max_gap_pp": yearly_max,
        "columns_over": over,
        "pass": not over,
    }


# --------------------------------------------------------------------------- CLI


def _connect(memory_limit: str, threads: int, temp_dir: Path) -> duckdb.DuckDBPyConnection:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit={_lit(memory_limit)}")
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET temp_directory={_lit(temp_dir)}")
    return con


def run_input_checks(
    views_dir: Path,
    p6_views_dir: Path,
    raw_root: Path,
    *,
    memory_limit: str = "10GB",
    threads: int = 8,
    temp_dir: Path,
    cut_dates: Sequence[str] = PIT_CUT_DATES,
) -> dict[str, Any]:
    """T1-T5 on a P6′ view folder (its fact and v4p files) against the P6 folder's ``_dlr``."""
    inputs.load_p6prime_manifest(views_dir)
    p6_views.verify_views(p6_views_dir)
    con = _connect(memory_limit, threads, temp_dir)
    try:
        p6_views.register_raw(con, raw_root, ["dart_filing_receipt_raw"])
        for name, path in (
            ("fact_v3", views_dir / f"{inputs.FACT_TABLE}.parquet"),
            ("v4p", views_dir / f"{inputs.FIN_FILE}.parquet"),
            ("p6_dlr", p6_views_dir / f"{inputs.FIN_FILE}.parquet"),
        ):
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({_lit(path)})")
        out: dict[str, Any] = {}
        rcpt = "dart_filing_receipt_raw"
        out["T1"] = pit_cut(
            con, fact_view="fact_v3", receipt_view=rcpt, mart_view="v4p", cut_dates=cut_dates
        )
        out["T2"] = december_attribution(
            con, p6_dlr_view="p6_dlr", fact_view="fact_v3", receipt_view=rcpt, mart_view="v4p"
        )
        out["T3"] = non_december(con, p6_dlr_view="p6_dlr", mart_view="v4p")
        out["T4"] = outlier_rows(con, p6_dlr_view="p6_dlr", mart_view="v4p")
        out["T5"] = r1_conflicts(con, mart_view="v4p")
    finally:
        con.close()
    return out


def run_panel_checks(
    p6_panel: Path,
    p6prime_panel: Path,
    raw_root: Path,
    *,
    memory_limit: str = "10GB",
    threads: int = 8,
    temp_dir: Path,
) -> dict[str, Any]:
    """T8 and T9 on two serving-mode panels (they must not carry a label column)."""
    con = _connect(memory_limit, threads, temp_dir)
    try:
        for p in (p6_panel, p6prime_panel):
            names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {_lit(p)}").fetchall()]
            bad = [c for c in names if p6_bp._BANNED_SERVING.search(c)]
            if bad:
                raise ValueError(f"{p} carries label/return columns {bad}; serving panels only")
        p6_views.register_raw(con, raw_root, ["stock_master"])
        return {
            "T8": compare_serving_panels(con, p6_panel, p6prime_panel),
            "T9": {
                "p6": missing_gap(con, p6_panel, "stock_master"),
                "p6prime": missing_gap(con, p6prime_panel, "stock_master"),
            },
        }
    finally:
        con.close()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m modeler.models._04_kr_next_p6prime.checks",
        description="Label-free technical checks of the P6′ draft §3 (T1-T5 inputs, T8-T9 panels)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inputs", help="T1-T5 on the P6′ view folder")
    a.add_argument("--views-dir", required=True)
    a.add_argument("--p6-views-dir", required=True)
    a.add_argument("--raw-root", required=True)
    b = sub.add_parser("panels", help="T8-T9 on two serving panels")
    b.add_argument("--p6-panel", required=True)
    b.add_argument("--p6prime-panel", required=True)
    b.add_argument("--raw-root", required=True)
    for p in (a, b):
        p.add_argument("--out", required=True, help="report JSON (must not exist)")
        p.add_argument("--memory-limit", default="10GB")
        p.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"REFUSED: report exists, not overwriting: {out}", file=sys.stderr)
        return 2
    tmp = out.with_name(f".{out.name}.duckdb_tmp")
    started = datetime.now(UTC)
    try:
        if args.cmd == "inputs":
            res = run_input_checks(
                Path(args.views_dir),
                Path(args.p6_views_dir),
                Path(args.raw_root),
                memory_limit=args.memory_limit,
                threads=args.threads,
                temp_dir=tmp,
            )
        else:
            res = run_panel_checks(
                Path(args.p6_panel),
                Path(args.p6prime_panel),
                Path(args.raw_root),
                memory_limit=args.memory_limit,
                threads=args.threads,
                temp_dir=tmp,
            )
    except (inputs.InputsRefused, p6_views.ViewsRefused, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    finally:
        if tmp.exists():
            for p in tmp.glob("*"):
                p.unlink(missing_ok=True)
            tmp.rmdir()
    report = {
        "cmd": args.cmd,
        "args": {k: v for k, v in vars(args).items() if k != "cmd"},
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duckdb_version": duckdb.__version__,
        "results": res,
        "pass": {k: v.get("pass") if "pass" in v else {s: x["pass"] for s, x in v.items()}
                 for k, v in res.items()},
    }  # fmt: skip
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(report["pass"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
