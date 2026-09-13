"""Deterministic row-mapping contracts for Horizon Scan null experiments."""

from __future__ import annotations

import hashlib
import struct
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

JOINT_CS_MAPPING_CONTRACT = "joint_cs_v2"
_SEPARATOR = "\x1f"


def _iso(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _canonical_mapping_frame(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"trade_date", "market", "ticker"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"mapping frame is missing columns: {missing}")
    keys = ["trade_date", "market", "ticker"]
    if frame.select(keys).is_duplicated().any():
        raise ValueError("mapping frame has duplicate (trade_date, market, ticker) rows")
    return frame.sort(keys)


def mapping_seed_sequence(
    *,
    mapping_contract_version: str,
    replicate_index: int,
    config_hash: str,
    trade_date: Any,
    market: Any,
) -> np.random.SeedSequence:
    """Derive the full-width SHA-256 entropy specified by ``joint_cs_v2``."""
    key = _SEPARATOR.join(
        [
            mapping_contract_version,
            str(replicate_index),
            config_hash,
            _iso(trade_date),
            str(market),
        ]
    ).encode("utf-8")
    entropy = int.from_bytes(hashlib.sha256(key).digest(), byteorder="big")
    return np.random.SeedSequence(entropy)


def build_group_permutation_mapping(
    frame: pl.DataFrame,
    *,
    replicate_index: int,
    config_hash: str,
    mapping_contract_version: str = JOINT_CS_MAPPING_CONTRACT,
) -> tuple[dict[tuple[Any, Any], np.ndarray], str]:
    """Build one deterministic ticker mapping per date×market group."""
    canonical = _canonical_mapping_frame(frame)
    return _build_group_permutation_mapping(
        canonical,
        replicate_index=replicate_index,
        config_hash=config_hash,
        mapping_contract_version=mapping_contract_version,
    )


def _build_group_permutation_mapping(
    canonical: pl.DataFrame,
    *,
    replicate_index: int,
    config_hash: str,
    mapping_contract_version: str,
) -> tuple[dict[tuple[Any, Any], np.ndarray], str]:
    """Build a mapping from an already canonicalized frame."""
    mappings: dict[tuple[Any, Any], np.ndarray] = {}
    digest = hashlib.sha256()
    for (trade_date, market), group in canonical.group_by(
        ["trade_date", "market"], maintain_order=True
    ):
        permutation = (
            np.random.default_rng(
                mapping_seed_sequence(
                    mapping_contract_version=mapping_contract_version,
                    replicate_index=replicate_index,
                    config_hash=config_hash,
                    trade_date=trade_date,
                    market=market,
                )
            )
            .permutation(group.height)
            .astype(np.int32, copy=False)
        )
        key = (trade_date, market)
        mappings[key] = permutation
        tickers = group["ticker"].cast(pl.Utf8).to_list()
        digest.update(_iso(trade_date).encode("utf-8"))
        digest.update(str(market).encode("utf-8"))
        digest.update(struct.pack("<I", group.height))
        for ticker in tickers:
            encoded = str(ticker).encode("utf-8")
            if len(encoded) > 0xFFFF:
                raise ValueError("ticker key is longer than the mapping contract allows")
            digest.update(struct.pack("<H", len(encoded)))
            digest.update(encoded)
        digest.update(permutation.tobytes(order="C"))
    return mappings, digest.hexdigest()


def apply_group_permutation(
    frame: pl.DataFrame,
    *,
    permute_cols: list[str],
    mappings: dict[tuple[Any, Any], np.ndarray],
) -> pl.DataFrame:
    """Apply one mapping to every requested feature column."""
    canonical = _canonical_mapping_frame(frame)
    return _apply_group_permutation_canonical(
        canonical, permute_cols=permute_cols, mappings=mappings
    )


def _apply_group_permutation_canonical(
    canonical: pl.DataFrame,
    *,
    permute_cols: list[str],
    mappings: dict[tuple[Any, Any], np.ndarray],
) -> pl.DataFrame:
    """Apply mappings without sorting or concatenating one frame per group."""
    missing = sorted(set(permute_cols) - set(canonical.columns))
    if missing:
        raise ValueError(f"permutation columns are missing: {missing}")
    fixed_cols = [col for col in canonical.columns if col not in permute_cols]
    source_indices: list[int] = []
    row_start = 0
    for key, group in canonical.group_by(["trade_date", "market"], maintain_order=True):
        try:
            permutation = mappings[key]
        except KeyError as exc:
            raise ValueError(f"mapping has no group {key!r}") from exc
        if len(permutation) != group.height:
            raise ValueError(f"mapping length mismatch for group {key!r}")
        source_indices.extend((row_start + permutation).tolist())
        row_start += group.height
    permuted = canonical.select(permute_cols)[source_indices]
    return canonical.select(fixed_cols).hstack(permuted).select(canonical.columns)


def build_and_apply_group_permutation(
    frame: pl.DataFrame,
    *,
    permute_cols: list[str],
    replicate_index: int,
    config_hash: str,
    mapping_contract_version: str = JOINT_CS_MAPPING_CONTRACT,
) -> tuple[pl.DataFrame, str]:
    canonical = _canonical_mapping_frame(frame)
    mappings, mapping_hash = _build_group_permutation_mapping(
        canonical,
        replicate_index=replicate_index,
        config_hash=config_hash,
        mapping_contract_version=mapping_contract_version,
    )
    return (
        _apply_group_permutation_canonical(canonical, permute_cols=permute_cols, mappings=mappings),
        mapping_hash,
    )
