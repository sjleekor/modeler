"""Feature explainer pages (``01_kr_price.html`` ... ``06_us_flow_holders.html``).

One self-contained HTML page per card document. The numbers come from the layer-3 run
(``feature_response_l3``), the pooled raw-value quantiles of the frozen feature panel and,
for the example charts, daily prices up to the reference date. No label file is ever opened.

Run ``python -m modeler.analysis.feature_response.explainer --help``.
"""
