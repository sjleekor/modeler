"""Model 01 — 20d market-relative excess-return rank (Ridge/ElasticNet).

First instance of the shared ETL platform (00_shared §7). Composes the
model-agnostic ``modeler.etl`` layer into a per-model dataset + training run.

The package dir uses a leading underscore (``_01_...``) so it is a valid Python
identifier; the model_id (00_shared §2) keeps the canonical ``01_20_access_return_rank``.
"""

MODEL_ID = "01_20_access_return_rank"
