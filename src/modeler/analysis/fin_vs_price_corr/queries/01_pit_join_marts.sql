-- Path A: PIT financial feature mart joined to forward-return labels.
-- `feat_price.px_ret_*` are backward-looking features, not targets.

WITH base AS (
    SELECT
        f.trade_date,
        f.ticker,
        f.market,
        f.fin_is_negative_equity,
        f.fin_has_fs,
        f.fin_roa,
        f.fin_roe,
        f.fin_debt_to_equity,
        f.fin_equity_ratio,
        f.fin_ocf_to_assets,
        f.fin_cash_ratio,
        f.fin_asset_turnover,
        f.fin_operating_margin,
        l.fwd_ret_5d,
        l.fwd_ret_20d,
        l.fwd_ret_60d,
        l.raw_label_5d,
        l.raw_label_20d,
        l.raw_label_60d,
        p.px_is_halted
    FROM feat_fin_pit AS f
    JOIN label_daily AS l
      USING (trade_date, ticker, market)
    LEFT JOIN feat_price AS p
      USING (trade_date, ticker, market)
    WHERE p.px_is_halted IS NOT TRUE
)
SELECT *
FROM base;
