"""Reranking configuration used by the optional improved service mode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankWeights:
    lambda_emb: float = 0.7
