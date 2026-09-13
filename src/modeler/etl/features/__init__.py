"""Feature builders (mart, L2a) — one module per source group.

Each builder reads lake/canonical views and writes a ``feat_*`` mart table of
*pre-standardization* derived features (etl_00 §3). Per-date winsorize/log/
z-score is deferred to the model preprocess stage (P5, etl_00 §4.3).

Column naming uses group prefixes for ablation toggles (00_shared §2, etl_00 §4.5):
``px_`` price, ``flow_`` flow, ``fin_`` financial, ``ev_`` event, ``cf_`` common.
"""
