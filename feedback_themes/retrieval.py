from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .domain import ContractError, Taxonomy

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RETRIEVAL_TOP_K = 12


class RetrievalError(RuntimeError):
    """Raised when local semantic retrieval cannot produce valid candidates."""


@dataclass(frozen=True)
class ThemeCandidate:
    specific_theme_id: str
    score: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "specific_theme_id": self.specific_theme_id,
            "score": round(self.score, 6),
        }


class ThemeRetriever(Protocol):
    model_name: str

    def retrieve(
        self,
        reviews: list[dict[str, Any]],
        taxonomy: Taxonomy,
        *,
        top_k: int,
    ) -> dict[str, list[ThemeCandidate]]: ...


class EmbeddingBackend(Protocol):
    def query_embed(self, texts: Iterable[str]) -> Iterable[Sequence[float]]: ...

    def passage_embed(self, texts: Iterable[str]) -> Iterable[Sequence[float]]: ...


class FastEmbedThemeRetriever:
    """Rank fixed taxonomy leaves with a small local ONNX embedding model."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path | None = None,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self.model_name = model_name
        if backend is not None:
            self._backend = backend
            return

        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RetrievalError(
                "fastembed is required for hybrid retrieval; install the "
                "project dependencies before running"
            ) from error

        arguments: dict[str, Any] = {"model_name": model_name}
        if cache_dir is not None:
            arguments["cache_dir"] = str(cache_dir)
        try:
            self._backend = TextEmbedding(**arguments)
        except Exception as error:
            raise RetrievalError(
                f"could not initialize embedding model {model_name!r}: {error}"
            ) from error

    def retrieve(
        self,
        reviews: list[dict[str, Any]],
        taxonomy: Taxonomy,
        *,
        top_k: int,
    ) -> dict[str, list[ThemeCandidate]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not reviews:
            return {}

        theme_ids, theme_documents = _theme_documents(taxonomy)
        selected_count = min(top_k, len(theme_ids))
        review_ids: list[str] = []
        review_queries: list[str] = []
        for index, review in enumerate(reviews):
            if not isinstance(review, dict):
                raise ContractError(f"reviews[{index}] must be an object")
            review_id = review.get("id")
            content = review.get("content_en")
            if not isinstance(review_id, str) or not review_id.strip():
                raise ContractError(
                    f"reviews[{index}].id must be a non-empty string"
                )
            if not isinstance(content, str) or not content.strip():
                raise ContractError(
                    f"reviews[{index}].content_en must be a non-empty string"
                )
            title = review.get("title")
            title_text = title.strip() if isinstance(title, str) else ""
            query = (
                f"Customer feedback title: {title_text}\n"
                f"Customer feedback: {content.strip()}"
            )
            review_ids.append(review_id)
            review_queries.append(query)

        try:
            theme_vectors = list(
                self._backend.passage_embed(theme_documents)
            )
            review_vectors = list(
                self._backend.query_embed(review_queries)
            )
        except Exception as error:
            raise RetrievalError(
                f"embedding inference failed for {self.model_name!r}: {error}"
            ) from error

        if len(theme_vectors) != len(theme_ids):
            raise RetrievalError(
                "embedding backend returned the wrong number of theme vectors"
            )
        if len(review_vectors) != len(review_ids):
            raise RetrievalError(
                "embedding backend returned the wrong number of review vectors"
            )

        rankings: dict[str, list[ThemeCandidate]] = {}
        for review_id, review_vector in zip(
            review_ids, review_vectors, strict=True
        ):
            scored = [
                ThemeCandidate(
                    specific_theme_id=theme_id,
                    score=_cosine_similarity(review_vector, theme_vector),
                )
                for theme_id, theme_vector in zip(
                    theme_ids, theme_vectors, strict=True
                )
            ]
            rankings[review_id] = sorted(
                scored,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.specific_theme_id,
                ),
            )[:selected_count]
        return rankings


def _theme_documents(taxonomy: Taxonomy) -> tuple[list[str], list[str]]:
    definitions: dict[str, str] = {}
    for strategic in taxonomy.source["strategic_themes"]:
        for midlevel in strategic["midlevel_themes"]:
            for specific in midlevel["specific_themes"]:
                definitions[specific["id"]] = specific["definition"]

    theme_ids: list[str] = []
    documents: list[str] = []
    for theme_id, path in taxonomy.leaves.items():
        definition = definitions.get(theme_id)
        if not isinstance(definition, str) or not definition.strip():
            raise ContractError(f"taxonomy definition missing for {theme_id!r}")
        theme_ids.append(theme_id)
        documents.append(
            "Feedback theme: "
            f"{path.strategic_label} > {path.midlevel_label} > "
            f"{path.specific_label}. Definition: {definition}"
        )
    return theme_ids, documents


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or len(left) == 0:
        raise RetrievalError("embedding vectors must have equal non-zero dimensions")
    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        left_float = float(left_value)
        right_float = float(right_value)
        dot_product += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm == 0.0 or right_norm == 0.0:
        raise RetrievalError("embedding vectors must have non-zero magnitude")
    return dot_product / math.sqrt(left_norm * right_norm)
