# ruff: noqa: E501
"""``dim_peer_monthly`` / ``feat_relation_stat`` — statistical peer relations (F-2).

Design: ``docs/dev/20260907_additional_feature/02_relationship_features.md`` §1.

The cross-section had no relation axis at all (``11_feature_taxonomy.md`` §7,
C5 = 0). Industry is the obvious grouping and it is the one we cannot have
point-in-time — ``induty_code`` carries today's industry backwards with no
change history (F-1 starts that versioning, and only from 2026-09 forwards).
So the peer set here is built from returns instead: structurally PIT,
available over the whole 2015- sample, and needing no new collection.

**Peer set** (§1.1). At each month end ``t0`` the correlation of market-model
*residual* returns is measured over the trailing 252 sessions, and the top
``PEER_K`` = 20 names become that stock's peers for every session from
``t0 + 1`` through the next month end. Residuals, not raw returns: raw-return
correlation is dominated by the market factor, which would make every stock a
peer of every other. The residual is
``trading_panel.build_market_model_sql``'s — the same one ``feat_price``'s
``px_idio_vol_60d`` / ``px_resid_mom_12_1`` and ``feat_macro_exposure``'s betas
are statements about, reused rather than redefined (F-2.1; ``feat_price``'s SQL
text is an A0 cache key and is untouched by this module).

**PIT rule.** Everything a session-``t`` row reads was published before
session ``t``:

- the peer set comes from a month end strictly before ``t``, chosen from
  returns through that month end only;
- the big-cap subset is ranked on the last ``daily_market_cap`` row strictly
  *before* ``t0``. That endpoint publishes T+1 (``CLAUDE.md``, "KRX 접근 제한"),
  so a session's own market cap is not readable until the next evening; taking
  the row before ``t0`` means the ranking was known by ``t0``'s close, hence
  before the first session the peer set is used on;
- ``rel_peer_bigcap_lag_ret_5d`` reads sessions ``t-5 .. t-1`` (Hou 2007's
  lead-lag is a claim about *yesterday's* large caps); every other column reads
  a window ending at ``t``.

**Valid from.** ``resid_ret`` needs 252 prior sessions before it exists at all,
the correlation window needs 252 sessions of it, and the peer 60-session return
needs 60 more — so the first populated session is roughly 252 + 252 sessions
after ``daily_ohlcv`` starts (2014-01). The measured effective start per column
is recorded in ``docs/dev/20260907_additional_feature/results/``.

**Grain** ``(trade_date, ticker, market)``, valid sessions only — halt rows
(``open=high=low=0``) never enter, exactly as in ``build_valid_session_sql``.

The diagnostic columns (``_k10`` / ``_k40`` / ``rel_peer_ksic_agree`` /
``rel_peer_corr_mean``) exist to be reported on, not preregistered. K is fixed
at 20 and the alternatives are never promoted — publishing both says how
sensitive the answer is to K, it does not license picking the best one after
seeing the result (§1.2, "K는 사전등록값이고 바꾸지 않는다").
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import duckdb
import numpy as np

from modeler.etl.config import LakeConfig
from modeler.etl.mart import (
    _metadata_path,
    _sql_hash,
    is_materialized,
    materialize,
    materialize_in_parts,
    register_mart_view,
)
from modeler.etl.trading_panel import build_market_model_sql, build_valid_session_sql

logger = logging.getLogger(__name__)

PEER_MONTHLY_TABLE = "dim_peer_monthly"
RELATION_STAT_TABLE = "feat_relation_stat"

#: Bump when the peer construction OR any column formula below changes. Neither
#: is covered by ``config_hash`` (the scan YAML), so without this a re-run could
#: reuse an artifact of an identical fingerprint while producing different
#: numbers — the role ``FIN_FEATURE_FORMULA_VERSION`` plays for
#: ``feat_fin_scan_daily``. It is part of both marts' cache contract text.
#:
#:   relation_stat_v1 — the 2026-09 rules as first built (02 §1).
#:   relation_stat_v2 — adds the ``_lag1`` execution variant of every
#:     ``PRIMARY_COLUMNS`` entry. No peer rule and no column formula changed;
#:     ``dim_peer_monthly`` rebuilds bit-identically. The scan config requires a
#:     ``variant_columns.lag1`` mapping per family and every other registered
#:     mart (``feat_market_cap``, ``feat_filing_activity``,
#:     ``feat_periodic_extras``, ``feat_macro_exposure``, ``feat_fin_risk``)
#:     carries real lag1 columns for its registered primaries; this one did not,
#:     so F-HS-1 would have had to name columns that do not exist.
FORMULA_VERSION = "relation_stat_v2"

# --- peer-set construction (frozen before any result was seen; §1.1) --------

#: Trailing sessions the residual correlation is measured over.
CORR_WINDOW_SESSIONS = 252
#: A stock is a candidate only with this many usable residuals in the window,
#: and a pair is correlated only over this many jointly-usable sessions (75%).
MIN_VALID_SESSIONS = 189
#: The preregistered peer-set size. Not a tuning knob — see module docstring.
PEER_K = 20
#: Peers stored per stock. 40 so the ``_k10``/``_k40`` sensitivity diagnostics
#: are a rank filter on the same table rather than a second build.
PEER_K_STORED = 40
#: Diagnostic peer-set size below K.
PEER_K_SMALL = 10
#: Large caps within the K=20 set that ``rel_peer_bigcap_lag_ret_5d`` reads.
BIGCAP_N = 5

# --- NULL rules (frozen; §1.2 "peer 중 그 날 유효 세션이 15개 미만이면 NULL") --

#: Of the 20 peers, how many must carry a usable return on session t.
MIN_PEER_VALID = 15
#: The same 75% floor on the diagnostic K's and on the 5 large caps (3 of 5).
#: Fixed here rather than after looking at coverage.
MIN_PEER_VALID_K10 = 8
MIN_PEER_VALID_K40 = 30
MIN_BIGCAP_VALID = 3

#: How far back a ``_lag1`` predecessor may sit. A session whose previous
#: valid session is more than a year away was suspended, and calling that a
#: one-session execution delay would be false — NULL is the honest answer.
#: Stating the bound also makes the column partition-independent: a part reads
#: its own year plus the previous one, so every predecessor this admits is
#: inside the part, and the parts reproduce the unpartitioned contract exactly.
#: ``feat_market_cap``/``feat_fin_risk`` lag unguarded only because they are
#: built in one shot and never had to answer this.
LAG1_MAX_GAP_DAYS = 365

#: Return windows. 20/60 end at session t; the big-cap window ends at t-1.
MOM_WINDOW_SHORT = 20
MOM_WINDOW_LONG = 60
BIGCAP_LAG_WINDOW = 5

#: The four preregistered families' primary/secondary columns (§3).
PRIMARY_COLUMNS: tuple[str, ...] = (
    "rel_peer_mom_20d",
    "rel_peer_mom_60d",
    "rel_own_minus_peer_20d",
    "rel_peer_bigcap_lag_ret_5d",
    "rel_peer_dispersion_20d",
)
#: Reported, never preregistered (§1.4). ``rel_peer_corr_mean`` is additionally
#: handed to the model as a conditioning variable without being scanned.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "rel_peer_corr_mean",
    "rel_peer_ksic_agree",
    "rel_peer_mom_20d_k10",
    "rel_peer_mom_20d_k40",
)
#: How many peers actually carried a value — the honest denominator behind each
#: NULL rule, kept so a thin session reads as thin rather than as a confident 0.
COUNT_COLUMNS: tuple[str, ...] = (
    "rel_peer_n_20d",
    "rel_peer_n_60d",
    "rel_peer_n_bigcap",
)
FEATURE_COLUMNS: tuple[str, ...] = PRIMARY_COLUMNS + DIAGNOSTIC_COLUMNS

_KEY_SEP = "\x1f"
_STAGING_TABLE = "_peer_monthly_staging"


# ---------------------------------------------------------------------------
# stage (a): dim_peer_monthly
# ---------------------------------------------------------------------------


def build_resid_panel_sql(price_view: str = "daily_ohlcv") -> str:
    """``(trade_date, ticker, market, resid_ret)`` on valid sessions.

    The residual definition is imported, not restated: ``build_market_model_sql``
    returns ``feat_price``'s own text verbatim, so the peer correlation and
    ``px_idio_vol_60d`` cannot drift into two ideas of "residual return".
    """
    return f"""
        WITH valid AS (
            {build_valid_session_sql(price_view)}
        ),
        {build_market_model_sql("valid")}
        SELECT trade_date, ticker, market, resid_ret
        FROM residuals
        WHERE resid_ret IS NOT NULL
    """


def build_month_end_sql(price_view: str = "daily_ohlcv") -> str:
    """The last valid KRX session of each calendar month.

    Read off the panel rather than from a holiday calendar: a month whose final
    weekday had no session must resolve to the session before it, and
    ``src/collector/kr/infra/calendar/data/holidays_krx.csv`` only covers 2024-2026 (F-9.2).
    """
    return f"""
        WITH valid AS (
            {build_valid_session_sql(price_view)}
        )
        SELECT max(trade_date) AS month_end
        FROM (SELECT DISTINCT trade_date FROM valid)
        GROUP BY date_trunc('month', trade_date)
        ORDER BY 1
    """


def build_peer_mcap_sql(
    *,
    price_view: str = "daily_ohlcv",
    market_cap_view: str = "daily_market_cap",
) -> str:
    """One market cap per ``(month_end, ticker, market)`` for the big-cap rank.

    The value is the last ``daily_market_cap`` row *strictly before* the month
    end — see the module docstring on why not the month end's own row. An ASOF
    join over the (month end x listed name) grid rather than an inequality join
    over the whole 7M-row table.
    """
    return f"""
        WITH month_ends AS (
            {build_month_end_sql(price_view)}
        ),
        caps AS (
            SELECT trade_date, ticker, market, CAST(market_cap AS DOUBLE) AS market_cap
            FROM {market_cap_view}
            WHERE market_cap IS NOT NULL AND market_cap > 0
        ),
        grid AS (
            SELECT m.month_end, n.ticker, n.market
            FROM month_ends m
            CROSS JOIN (SELECT DISTINCT ticker, market FROM caps) n
        )
        SELECT g.month_end, g.ticker, g.market, c.market_cap
        FROM grid g
        ASOF LEFT JOIN caps c
          ON g.ticker = c.ticker AND g.market = c.market AND c.trade_date < g.month_end
        WHERE c.market_cap IS NOT NULL
    """


@dataclass(frozen=True)
class _Panel:
    """A dense ``(session, name)`` residual matrix and its two axes."""

    #: Ascending session axis, as days since 1970-01-01.
    days: np.ndarray
    #: Name axis: ``(ticker, market)`` pairs, in the matrix's column order.
    names: list[tuple[str, str]]
    #: ``(len(days), len(names))`` float64, NaN where there is no residual.
    values: np.ndarray


def _epoch_days(value: object) -> int:
    """Days since 1970-01-01 for a ``datetime.date`` (the panel's date key)."""
    return int(np.datetime64(value, "D").astype("int64"))


def _load_resid_panel(con: duckdb.DuckDBPyConnection, price_view: str) -> _Panel:
    """Pivot the residual panel into one dense matrix, once.

    Dense rather than per-window: ~2,900 sessions x ~3,500 names is 80MB in
    float64, and every month end is then a contiguous row slice instead of its
    own pivot.
    """
    frame = con.execute(
        "SELECT date_diff('day', DATE '1970-01-01', trade_date)::INTEGER AS d, "
        "ticker, market, resid_ret "
        f"FROM ({build_resid_panel_sql(price_view)})"
    ).pl()
    if frame.height == 0:
        return _Panel(days=np.empty(0, dtype=np.int64), names=[], values=np.empty((0, 0)))

    keys = (frame["ticker"] + _KEY_SEP + frame["market"]).to_numpy()
    uniq_keys, name_idx = np.unique(keys, return_inverse=True)
    names = [tuple(key.split(_KEY_SEP)) for key in uniq_keys]

    days = np.unique(frame["d"].to_numpy().astype(np.int64))
    date_idx = np.searchsorted(days, frame["d"].to_numpy().astype(np.int64))

    values = np.full((days.size, len(names)), np.nan, dtype=np.float64)
    values[date_idx, name_idx] = frame["resid_ret"].to_numpy().astype(np.float64)
    return _Panel(days=days, names=names, values=values)  # type: ignore[arg-type]


def _pairwise_complete_corr(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson correlation, and the joint-observation counts.

    ``window`` is ``(sessions, names)`` with NaN for a missing session. Pairs
    are correlated on the sessions where *both* are present, which is the only
    honest option here: a stock listed mid-window, or halted for a fortnight,
    has real gaps, and filling them with zero would turn a short overlap into a
    confidently low correlation. Computed from cross-moment matrices — five
    BLAS matmuls per month end rather than ~4M individual correlations.
    """
    mask = np.isfinite(window)
    z = np.where(mask, window, 0.0)
    m = mask.astype(np.float64)

    joint = m.T @ m
    sx = z.T @ m  # sx[i, j] = sum of x_i over the sessions where j is present
    sxx = (z * z).T @ m
    sxy = z.T @ z

    # ``joint`` is symmetric and the y-side moments are the x-side ones
    # transposed (``sy[i, j] == sx[j, i]``), so the second variance is a
    # transposed view rather than three more N x N arrays — this is a ~2,800
    # wide matrix per month end and the temporaries dominate the memory.
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = joint * sxy
        cov -= sx * sx.T
        var_x = joint * sxx
        var_x -= sx * sx
        rho = cov / np.sqrt(var_x * var_x.T)
    rho[~np.isfinite(rho)] = np.nan
    return rho, joint


def _top_peers(
    rho: np.ndarray, joint: np.ndarray, k: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(row, peer columns, rho, joint sessions)`` in descending ρ order.

    A pair below ``MIN_VALID_SESSIONS`` joint sessions is not a candidate at
    all — its correlation is not measured on enough overlap to rank on.
    """
    eligible = joint >= MIN_VALID_SESSIONS
    np.fill_diagonal(eligible, False)
    scored = np.where(eligible, rho, np.nan)

    for i in range(scored.shape[0]):
        row = scored[i]
        usable = np.flatnonzero(np.isfinite(row))
        if usable.size == 0:
            continue
        take = min(k, usable.size)
        # argpartition finds the k best in O(n); only those k are then sorted.
        part = usable[np.argpartition(-row[usable], take - 1)[:take]]
        order = part[np.argsort(-row[part], kind="stable")]
        yield i, order, row[order], joint[i, order]


@dataclass
class _MonthBatch:
    """Flat column buffers for one month end's peer rows."""

    self_pos: list[int]
    peer_pos: list[int]
    rho: list[float]
    rank: list[int]
    joint: list[int]
    mcap_rank: list[int | None]

    @classmethod
    def empty(cls) -> _MonthBatch:
        return cls([], [], [], [], [], [])

    def __len__(self) -> int:
        return len(self.self_pos)


def _collect_month(rho: np.ndarray, joint: np.ndarray, caps: np.ndarray) -> _MonthBatch:
    """Select each candidate's peers and rank the large caps inside its K=20.

    The big-cap rank is taken within the K=20 set only, because that is the set
    the feature reads: ranking over all 40 stored peers and then filtering to
    rank <= 20 would silently leave some stocks fewer than five large caps.
    """
    batch = _MonthBatch.empty()
    for i, order, rhos, joints in _top_peers(rho, joint, PEER_K_STORED):
        n_peers = order.size
        k20 = min(PEER_K, n_peers)
        head_caps = caps[order[:k20]]
        ranked = np.flatnonzero(np.isfinite(head_caps))
        ranked = ranked[np.argsort(-head_caps[ranked], kind="stable")]
        rank_of_pos = {int(pos): rank for rank, pos in enumerate(ranked, start=1)}

        batch.self_pos.extend([i] * n_peers)
        batch.peer_pos.extend(int(j) for j in order)
        batch.rho.extend(float(v) for v in rhos)
        batch.rank.extend(range(1, n_peers + 1))
        batch.joint.extend(int(v) for v in joints)
        batch.mcap_rank.extend(rank_of_pos.get(pos) for pos in range(n_peers))
    return batch


def _month_caps_array(
    candidate_names: list[tuple[str, str]], month_caps: dict[tuple[str, str], float]
) -> np.ndarray:
    return np.array([month_caps.get(name, np.nan) for name in candidate_names], dtype=np.float64)


def _same_ksic2(
    candidate_names: list[tuple[str, str]],
    ksic2: dict[str, str],
    self_pos: list[int],
    peer_pos: list[int],
) -> list[bool | None]:
    """Per-peer "shares my KSIC 2-digit" flag, NULL when either side is unknown."""
    codes = [ksic2.get(ticker) for ticker, _ in candidate_names]
    out: list[bool | None] = []
    for i, j in zip(self_pos, peer_pos, strict=True):
        own, peer = codes[i], codes[j]
        out.append(None if own is None or peer is None else own == peer)
    return out


def compute_peer_monthly(
    con: duckdb.DuckDBPyConnection,
    *,
    price_view: str = "daily_ohlcv",
    market_cap_view: str = "daily_market_cap",
    corp_view: str = "dart_corp_master",
    table_name: str = _STAGING_TABLE,
) -> str:
    """Compute the monthly peer sets into a DuckDB table. Returns its name.

    One pass per month end, each reading only the trailing
    ``CORR_WINDOW_SESSIONS`` rows of the residual panel — so no month end can
    see a return that follows it. This is the PIT guarantee the whole family
    rests on, and the one ``test_relation_stat.py`` pins by mutating post-``t0``
    returns and asserting the peer sets do not move.
    """
    import polars as pl

    panel = _load_resid_panel(con, price_view)
    month_ends = [
        row[0]
        for row in con.execute(
            f"SELECT month_end FROM ({build_month_end_sql(price_view)})"
        ).fetchall()
    ]
    cap_by_month = _load_month_caps(con, price_view=price_view, market_cap_view=market_cap_view)
    ksic2 = _load_ksic2(con, corp_view)

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"""
        CREATE TABLE {table_name} (
            month_end DATE, ticker VARCHAR, market VARCHAR,
            peer_ticker VARCHAR, peer_market VARCHAR,
            rho DOUBLE, peer_rank INTEGER, joint_sessions INTEGER,
            peer_market_cap DOUBLE, peer_mcap_rank_k20 INTEGER, same_ksic2 BOOLEAN
        )
        """)

    total = 0
    for month_end in month_ends:
        end_pos = int(np.searchsorted(panel.days, _epoch_days(month_end), side="right"))
        if end_pos < CORR_WINDOW_SESSIONS:
            continue
        window = panel.values[end_pos - CORR_WINDOW_SESSIONS : end_pos, :]
        candidates = np.flatnonzero(np.isfinite(window).sum(axis=0) >= MIN_VALID_SESSIONS)
        if candidates.size < 2:
            continue

        names = [panel.names[c] for c in candidates]
        rho, joint = _pairwise_complete_corr(window[:, candidates])
        caps = _month_caps_array(names, cap_by_month.get(month_end, {}))
        batch = _collect_month(rho, joint, caps)
        if not batch:
            continue

        frame = pl.DataFrame(
            {
                "month_end": [month_end] * len(batch),
                "ticker": [names[i][0] for i in batch.self_pos],
                "market": [names[i][1] for i in batch.self_pos],
                "peer_ticker": [names[j][0] for j in batch.peer_pos],
                "peer_market": [names[j][1] for j in batch.peer_pos],
                "rho": batch.rho,
                "peer_rank": batch.rank,
                "joint_sessions": batch.joint,
                "peer_market_cap": [
                    None if not np.isfinite(caps[j]) else float(caps[j]) for j in batch.peer_pos
                ],
                "peer_mcap_rank_k20": batch.mcap_rank,
                "same_ksic2": _same_ksic2(names, ksic2, batch.self_pos, batch.peer_pos),
            }
        )
        con.register("_peer_batch", frame)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM _peer_batch")
        con.unregister("_peer_batch")
        total += len(batch)
        logger.debug(
            "peer month %s: %d candidates, %d rows", month_end, candidates.size, len(batch)
        )

    logger.info("%s: %d rows over %d month ends", table_name, total, len(month_ends))
    return table_name


def _load_month_caps(
    con: duckdb.DuckDBPyConnection,
    *,
    price_view: str,
    market_cap_view: str,
) -> dict[object, dict[tuple[str, str], float]]:
    """``month_end -> {(ticker, market): market cap}``, or empty without the view."""
    try:
        rows = con.execute(
            "SELECT month_end, ticker, market, market_cap FROM ("
            f"{build_peer_mcap_sql(price_view=price_view, market_cap_view=market_cap_view)})"
        ).fetchall()
    except duckdb.Error:
        logger.warning("%s unavailable; rel_peer_bigcap_lag_ret_5d will be NULL", market_cap_view)
        return {}
    by_month: dict[object, dict[tuple[str, str], float]] = {}
    for month_end, ticker, market, market_cap in rows:
        by_month.setdefault(month_end, {})[(ticker, market)] = float(market_cap)
    return by_month


def _load_ksic2(con: duckdb.DuckDBPyConnection, corp_view: str) -> dict[str, str]:
    """``ticker -> KSIC 2-digit prefix`` from the corp master, or empty.

    Backcast by construction (one current code per corporation, no history), so
    it only ever feeds ``rel_peer_ksic_agree``, a diagnostic. A lake without the
    column yields no agreement column rather than failing the build — the
    opposite of ``register_industry_group_view``, which must fail loudly because
    *its* consumer is a feature.
    """
    try:
        rows = con.execute(
            f"SELECT ticker, induty_code FROM {corp_view} "
            "WHERE ticker IS NOT NULL AND ticker <> '' AND induty_code IS NOT NULL"
        ).fetchall()
    except duckdb.Error:
        logger.warning("%s has no induty_code; rel_peer_ksic_agree will be NULL", corp_view)
        return {}
    return {str(ticker): str(code)[:2] for ticker, code in rows if len(str(code)) >= 2}


def peer_monthly_contract_sql(staging_table: str = _STAGING_TABLE) -> str:
    """The cache-contract text for ``dim_peer_monthly``.

    The peer sets come out of numpy, not SQL, so this is what stands in for the
    formula: every frozen constant appears in it, and ``FORMULA_VERSION`` covers
    the rest. Changing any of them changes ``mart._sql_hash`` and forces the
    rebuild that a changed peer definition demands.
    """
    return f"""
        -- {FORMULA_VERSION}: window={CORR_WINDOW_SESSIONS} min_valid={MIN_VALID_SESSIONS}
        -- k={PEER_K} stored={PEER_K_STORED} bigcap={BIGCAP_N}
        SELECT month_end, ticker, market, peer_ticker, peer_market,
               rho, peer_rank, joint_sessions, peer_market_cap, peer_mcap_rank_k20, same_ksic2
        FROM {staging_table}
    """


def _peer_monthly_is_current(con: duckdb.DuckDBPyConnection, config: LakeConfig) -> bool:
    """True when the on-disk peer mart was built by these exact rules."""
    del con
    if not is_materialized(config, PEER_MONTHLY_TABLE):
        return False
    path = _metadata_path(config, PEER_MONTHLY_TABLE)
    if not path.is_file():
        return False
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stored.get("analysis_config_hash") != config.analysis_config_hash:
        return False
    return stored.get("sql_hash") == _sql_hash(peer_monthly_contract_sql())


def materialize_peer_monthly(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    market_cap_view: str = "daily_market_cap",
    corp_view: str = "dart_corp_master",
    force: bool = False,
) -> str:
    """Build + register ``dim_peer_monthly``. Returns the view name.

    Requires ``price_view`` (and, for the big-cap rank, ``market_cap_view``)
    registered on ``con``. Idempotent: an existing mart built under the same
    ``FORMULA_VERSION`` and constants is reused, and the several minutes of
    correlation work is skipped.
    """
    if not force and _peer_monthly_is_current(con, config):
        return register_mart_view(con, config, PEER_MONTHLY_TABLE)

    staging = compute_peer_monthly(
        con,
        price_view=price_view,
        market_cap_view=market_cap_view,
        corp_view=corp_view,
    )
    materialize(con, config, PEER_MONTHLY_TABLE, peer_monthly_contract_sql(staging), force=True)
    con.execute(f"DROP TABLE IF EXISTS {staging}")
    return register_mart_view(con, config, PEER_MONTHLY_TABLE)


# ---------------------------------------------------------------------------
# stage (b): feat_relation_stat
# ---------------------------------------------------------------------------


def build_return_panel_sql(price_view: str = "daily_ohlcv") -> str:
    """Per-ticker trailing log returns, on that ticker's own valid sessions.

    Each window is required to be *complete* (``COUNT(log_ret) = n``) rather
    than partial: a peer average that silently mixes a 20-session return with a
    6-session one is not the quantity §1.2 defines. ``ret_5d_prior`` ends one
    session back — the lead-lag column reads yesterday's large caps.
    """
    return f"""
        WITH valid AS (
            {build_valid_session_sql(price_view)}
        )
        SELECT
            trade_date, ticker, market,
            CASE WHEN COUNT(log_ret) OVER w_short = {MOM_WINDOW_SHORT}
                 THEN SUM(log_ret) OVER w_short END AS ret_20d,
            CASE WHEN COUNT(log_ret) OVER w_long = {MOM_WINDOW_LONG}
                 THEN SUM(log_ret) OVER w_long END AS ret_60d,
            CASE WHEN COUNT(log_ret) OVER w_prior = {BIGCAP_LAG_WINDOW}
                 THEN SUM(log_ret) OVER w_prior END AS ret_5d_prior
        FROM valid
        WINDOW
            w_short AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN {MOM_WINDOW_SHORT - 1} PRECEDING AND CURRENT ROW),
            w_long AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN {MOM_WINDOW_LONG - 1} PRECEDING AND CURRENT ROW),
            w_prior AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN {BIGCAP_LAG_WINDOW} PRECEDING AND 1 PRECEDING)
    """


def build_relation_stat_sql(
    *,
    price_view: str = "daily_ohlcv",
    peer_view: str = PEER_MONTHLY_TABLE,
    return_panel_view: str | None = None,
    session_filter: str | None = None,
    emit_filter: str | None = None,
) -> str:
    """SQL producing ``feat_relation_stat``.

    Args:
        price_view: Daily OHLCV view.
        peer_view: ``dim_peer_monthly``.
        return_panel_view: When given, read the trailing returns from this
            already-materialized relation instead of recomputing them inline.
            The inline form is the *contract* — what the cache hash is taken
            from — and both run the same text either way, since
            ``build_return_panel_sql`` is what gets embedded and what the temp
            table is built from.
        session_filter: An extra predicate on the self rows, used to bound the
            peer join when the mart is written a year at a time. It restricts
            which rows are *computed*, never which history the rolling windows
            see.
        emit_filter: An extra predicate applied *after* the ``_lag1`` window,
            so a part can compute a session's predecessor and still emit only
            its own year. A part therefore reads one year wider than it writes:
            filtering on ``session_filter`` alone would hand every year's first
            session a NULL lag1, since SQL applies WHERE before the window.
    """
    panel_cte = (
        f"SELECT * FROM {return_panel_view}"
        if return_panel_view
        else build_return_panel_sql(price_view)
    )
    where_self = f"WHERE {session_filter}" if session_filter else ""
    where_emit = f"WHERE {emit_filter}" if emit_filter else ""
    lag1 = ",\n                ".join(
        f"CASE WHEN date_diff('day', LAG(trade_date) OVER w, trade_date)"
        f" <= {LAG1_MAX_GAP_DAYS}\n                     THEN LAG({column}) OVER w END"
        f" AS {column}_lag1"
        for column in PRIMARY_COLUMNS
    )
    return f"""
        -- {FORMULA_VERSION}
        WITH rets AS (
            {panel_cte}
        ),
        month_ends AS (
            SELECT DISTINCT month_end FROM {peer_view}
        ),
        -- The peer set of the last month end STRICTLY BEFORE t. So t0 + 1 is
        -- the first session to use t0's set and t0 itself still uses the
        -- previous month's: a peer set is never read on a session whose own
        -- return helped choose it.
        self_rows AS (
            SELECT r.*, m.month_end AS peer_month_end
            FROM rets r
            ASOF LEFT JOIN month_ends m ON m.month_end < r.trade_date
            {where_self}
        ),
        peer_joined AS (
            SELECT
                s.trade_date, s.ticker, s.market,
                p.peer_rank, p.rho, p.same_ksic2, p.peer_mcap_rank_k20,
                pr.ret_20d AS peer_ret_20d,
                pr.ret_60d AS peer_ret_60d,
                pr.ret_5d_prior AS peer_ret_5d_prior
            FROM self_rows s
            JOIN {peer_view} p
              ON p.month_end = s.peer_month_end
             AND p.ticker = s.ticker AND p.market = s.market
            LEFT JOIN rets pr
              ON pr.trade_date = s.trade_date
             AND pr.ticker = p.peer_ticker AND pr.market = p.peer_market
        ),
        peer_agg AS (
            SELECT
                trade_date, ticker, market,
                count(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K}) AS n_20d,
                avg(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K}) AS mean_20d,
                stddev_samp(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K}) AS disp_20d,
                count(peer_ret_60d) FILTER (WHERE peer_rank <= {PEER_K}) AS n_60d,
                avg(peer_ret_60d) FILTER (WHERE peer_rank <= {PEER_K}) AS mean_60d,
                count(peer_ret_5d_prior) FILTER (
                    WHERE peer_rank <= {PEER_K} AND peer_mcap_rank_k20 <= {BIGCAP_N}
                ) AS n_bigcap,
                avg(peer_ret_5d_prior) FILTER (
                    WHERE peer_rank <= {PEER_K} AND peer_mcap_rank_k20 <= {BIGCAP_N}
                ) AS mean_bigcap,
                avg(rho) FILTER (WHERE peer_rank <= {PEER_K}) AS corr_mean,
                avg(CASE WHEN same_ksic2 THEN 1.0 ELSE 0.0 END)
                    FILTER (WHERE peer_rank <= {PEER_K}) AS ksic_agree,
                count(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K_SMALL}) AS n_20d_k10,
                avg(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K_SMALL}) AS mean_20d_k10,
                count(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K_STORED}) AS n_20d_k40,
                avg(peer_ret_20d) FILTER (WHERE peer_rank <= {PEER_K_STORED}) AS mean_20d_k40
            FROM peer_joined
            GROUP BY trade_date, ticker, market
        ),
        vals AS (
        SELECT
            s.trade_date, s.ticker, s.market,
            CASE WHEN a.n_20d >= {MIN_PEER_VALID} THEN a.mean_20d END AS rel_peer_mom_20d,
            CASE WHEN a.n_60d >= {MIN_PEER_VALID} THEN a.mean_60d END AS rel_peer_mom_60d,
            CASE WHEN a.n_20d >= {MIN_PEER_VALID} AND s.ret_20d IS NOT NULL
                 THEN s.ret_20d - a.mean_20d END AS rel_own_minus_peer_20d,
            CASE WHEN a.n_bigcap >= {MIN_BIGCAP_VALID} THEN a.mean_bigcap END
                AS rel_peer_bigcap_lag_ret_5d,
            CASE WHEN a.n_20d >= {MIN_PEER_VALID} THEN a.disp_20d END
                AS rel_peer_dispersion_20d,
            a.corr_mean AS rel_peer_corr_mean,
            a.ksic_agree AS rel_peer_ksic_agree,
            CASE WHEN a.n_20d_k10 >= {MIN_PEER_VALID_K10} THEN a.mean_20d_k10 END
                AS rel_peer_mom_20d_k10,
            CASE WHEN a.n_20d_k40 >= {MIN_PEER_VALID_K40} THEN a.mean_20d_k40 END
                AS rel_peer_mom_20d_k40,
            a.n_20d AS rel_peer_n_20d,
            a.n_60d AS rel_peer_n_60d,
            a.n_bigcap AS rel_peer_n_bigcap,
            s.peer_month_end AS rel_peer_month_end,
            s.ret_20d AS rel_own_ret_20d
        FROM self_rows s
        LEFT JOIN peer_agg a USING (trade_date, ticker, market)
        ),
        -- The execution-delay variant the scan config requires a mapping for.
        -- Lagged over the ticker's own emitted sessions, so a halted session
        -- that never reaches the mart is not counted as a day of delay, and
        -- bounded by LAG1_MAX_GAP_DAYS so a suspension is not read as one.
        lagged AS (
            SELECT
                *,
                {lag1}
            FROM vals
            WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
        )
        SELECT * FROM lagged
        {where_emit}
    """


def register_relation_stat_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = RELATION_STAT_TABLE,
    price_view: str = "daily_ohlcv",
    peer_view: str = PEER_MONTHLY_TABLE,
) -> str:
    """Register a DuckDB view over the SQL above (no parquet — tests/parity)."""
    sql = build_relation_stat_sql(price_view=price_view, peer_view=peer_view)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_relation_stat(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    peer_view: str = PEER_MONTHLY_TABLE,
    force: bool = False,
) -> str:
    """Build + register ``feat_relation_stat``, one parquet part per year.

    Requires ``price_view`` and ``peer_view`` registered on ``con``. The join
    fans every ticker-session out to 40 peer rows — ~280M rows over the full
    history for a ~7M-row output — so it is written a year at a time; the
    returns panel is computed once into a temp table and shared across parts.
    """
    contract_sql = build_relation_stat_sql(price_view=price_view, peer_view=peer_view)
    if is_materialized(config, RELATION_STAT_TABLE) and not force:
        # Same cache check as a one-shot build; the parts are never needed.
        materialize_in_parts(con, config, RELATION_STAT_TABLE, contract_sql, (), force=False)
        return register_mart_view(con, config, RELATION_STAT_TABLE)

    panel_table = "_relation_return_panel"
    con.execute(f"DROP TABLE IF EXISTS {panel_table}")
    con.execute(f"CREATE TEMP TABLE {panel_table} AS {build_return_panel_sql(price_view)}")
    years = [
        int(row[0])
        for row in con.execute(
            f"SELECT DISTINCT year(trade_date) FROM {panel_table} ORDER BY 1"
        ).fetchall()
    ]
    # Each part computes its own year plus the one before it and emits only its
    # own, so the first session of a year gets a real ``_lag1`` from the last
    # session of the previous year instead of a NULL at 20 part boundaries.
    parts = [
        (
            f"part-{year}",
            build_relation_stat_sql(
                price_view=price_view,
                peer_view=peer_view,
                return_panel_view=panel_table,
                session_filter=f"year(r.trade_date) IN ({year - 1}, {year})",
                emit_filter=f"year(trade_date) = {year}",
            ),
        )
        for year in years
    ]
    materialize_in_parts(con, config, RELATION_STAT_TABLE, contract_sql, parts, force=True)
    con.execute(f"DROP TABLE IF EXISTS {panel_table}")
    return register_mart_view(con, config, RELATION_STAT_TABLE)
