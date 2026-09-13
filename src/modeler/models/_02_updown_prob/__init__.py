"""Model 02 — probability of beating the market over h sessions.

Plan: ``docs/dev/20260907_model_experiment/``. This package is additive: model
01 (``_01_20_access_return_rank``) is read-only here and the shared libraries
(``research/etl/labels.py``, ``preprocess.py``, ``metrics.py``, ``splits.py``)
are extended, never re-defaulted, so 01's published gate numbers still hold.
"""
