"""``dim_industry_group`` — one normalisation group per ticker (N2-9).

The grouping rules live in :mod:`collector.kr.definitions.industry_groups`, a
pure-data module the marts import the same way they import ``metric_rules``.
They are not reimplemented in SQL, because the fold-up is *data-dependent*: a
2-digit KSIC group with fewer than 20 members folds into its section, and a
section still short of 20 folds into ``OTHER``. That loop needs the membership
counts, so it runs in Python and its answer is handed to DuckDB as a lookup.

Two limits, both deliberate and both the reason this is diagnostic-only.

**It is not point-in-time.** ``induty_code`` comes from ``company.json``, which
reports today's industry with no change history. A company that switched
business lines carries its current industry backwards. That is a look-ahead,
which is why `11_feature_taxonomy.md` §6.3 confines industry to a normalisation
group and a diagnostic axis rather than an alpha feature.

**Groups are resolved once, not per date.** ``resolve_groups`` is documented as
a per-date call because membership moves with listings, and a per-date
resolution would be the right thing for a feature. For a diagnostic sitting on
a non-PIT input it would add cost and a moving group label without removing the
look-ahead that already dominates, so the whole universe is resolved together.
"""

from __future__ import annotations

import duckdb
from collector.kr.shared import (
    MIN_GROUP_SIZE,
    OTHER_GROUP,
    UNKNOWN_GROUP,
    resolve_groups,
)

INDUSTRY_GROUP_VIEW = "dim_industry_group"

__all__ = [
    "INDUSTRY_GROUP_VIEW",
    "MIN_GROUP_SIZE",
    "OTHER_GROUP",
    "UNKNOWN_GROUP",
    "register_industry_group_view",
]


def register_industry_group_view(
    con: duckdb.DuckDBPyConnection,
    *,
    corp_master_view: str = "dart_corp_master",
    view_name: str = INDUSTRY_GROUP_VIEW,
    min_group_size: int = MIN_GROUP_SIZE,
) -> str:
    """Register a ``(ticker, industry_group)`` view over the corp master.

    Args:
        con: DuckDB connection with *corp_master_view* registered.
        corp_master_view: Source view carrying ``ticker`` and ``induty_code``.
        view_name: Name to register.
        min_group_size: Smallest group allowed to stand on its own.

    Returns:
        The registered view name.

    Raises:
        RuntimeError: When the corp master has no ``induty_code`` column. That
            is the expected state on a lake snapshot taken before 2026-08-15,
            and it must fail loudly: silently grouping everything as unknown
            would produce an industry-neutral variant identical to the plain
            one, which reads as "industry does not matter".
    """
    columns = {
        row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {corp_master_view}").fetchall()
    }
    if "induty_code" not in columns:
        raise RuntimeError(
            f"{corp_master_view} has no induty_code column; it was added on 2026-08-15 "
            "(N2). Refresh the lake before building the industry-neutral variant."
        )

    rows = con.execute(f"""
        SELECT ticker, induty_code
        FROM {corp_master_view}
        WHERE ticker IS NOT NULL AND ticker <> ''
        """).fetchall()
    codes_by_ticker = {str(ticker): code for ticker, code in rows}
    groups = resolve_groups(codes_by_ticker, min_group_size=min_group_size)

    con.execute(f"CREATE OR REPLACE TEMP TABLE _{view_name}_src(ticker VARCHAR, group_ VARCHAR)")
    if groups:
        con.executemany(
            f"INSERT INTO _{view_name}_src VALUES (?, ?)",
            sorted(groups.items()),
        )
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"SELECT ticker, group_ AS industry_group FROM _{view_name}_src"
    )
    return view_name
