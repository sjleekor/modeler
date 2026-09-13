-- Path B: raw DART account PIT alignment template.
--
-- This is for account-level follow-up after Path A. Replace ACCOUNT_NAME_FILTER
-- or remove that predicate to inspect a broader account set.
--
-- Important PIT rules:
--   1. disclosed_date comes from rcept_no[1:8], not fetched_at.
--   2. a filing is usable only from the next trading day after disclosed_date.
--   3. revisions are intervalized per raw account key; do not globally keep
--      the latest rcept_no.
--   4. account_nm and ord are preserved because many rows share
--      account_id = '-표준계정코드 미사용-'.

WITH trading_days AS (
    SELECT DISTINCT trade_date
    FROM daily_ohlcv
),
raw_account AS (
    SELECT
        fs.corp_code,
        fs.ticker,
        COALESCE(sm.market, 'UNKNOWN') AS market,
        fs.bsns_year,
        fs.reprt_code,
        fs.fs_div,
        fs.sj_div,
        fs.sj_nm,
        fs.account_id,
        fs.account_nm,
        fs.ord,
        CAST(fs.thstrm_amount AS DOUBLE) AS account_value,
        fs.currency,
        fs.rcept_no,
        strptime(left(fs.rcept_no, 8), '%Y%m%d')::date AS disclosed_date
    FROM dart_financial_statement_raw AS fs
    LEFT JOIN stock_master AS sm
      USING (ticker)
    WHERE fs.thstrm_amount IS NOT NULL
      -- AND fs.account_nm = 'ACCOUNT_NAME_FILTER'
),
available AS (
    SELECT
        r.*,
        (
            SELECT min(td.trade_date)
            FROM trading_days AS td
            WHERE td.trade_date > r.disclosed_date
        ) AS available_from
    FROM raw_account AS r
    WHERE r.disclosed_date IS NOT NULL
),
intervalized AS (
    SELECT
        *,
        lead(available_from) OVER (
            PARTITION BY
                corp_code,
                ticker,
                bsns_year,
                reprt_code,
                fs_div,
                sj_div,
                account_id,
                account_nm,
                ord
            ORDER BY available_from, rcept_no
        ) AS next_from
    FROM available
    WHERE available_from IS NOT NULL
),
pit_account AS (
    SELECT
        u.trade_date,
        u.ticker,
        u.market,
        i.corp_code,
        i.bsns_year,
        i.reprt_code,
        i.fs_div,
        i.sj_div,
        i.sj_nm,
        i.account_id,
        i.account_nm,
        i.ord,
        i.account_value,
        i.currency,
        i.rcept_no,
        i.disclosed_date,
        i.available_from
    FROM dim_universe_daily AS u
    JOIN intervalized AS i
      ON i.ticker = u.ticker
     AND i.market = u.market
     AND i.available_from <= u.trade_date
     AND (i.next_from IS NULL OR u.trade_date < i.next_from)
    WHERE u.in_universe
)
SELECT *
FROM pit_account;
