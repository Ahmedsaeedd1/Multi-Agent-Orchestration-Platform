"""
Phase 4 — Long-Term Semantic Memory (Qdrant)

Uses the collections created in Phase 0.  Stores text + metadata as payload,
embeds queries via a user-supplied callable, and filters results by ``user_id``
to enforce memory isolation between users.
"""

import logging
import uuid
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


class QdrantStore:
    """Vector memory store backed by Qdrant."""

    def __init__(self, url: str, collection: str, embedding_fn: Callable[[str], list[float]]):
        self.collection = collection
        self.embedding_fn = embedding_fn
        self._client = QdrantClient(url=url)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_memory(self, user_id: str, text: str, metadata: dict) -> str:
        """
        Embed *text*, upsert a point into the collection, and return the
        generated point ID.

        The ``user_id`` is stored in the payload so ``retrieve_memory`` can
        filter by it.
        """
        vector = self.embedding_fn(text)
        point_id = str(uuid.uuid4())

        payload = {**metadata, "text": text, "user_id": user_id}

        self._client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Embed *query* and search the collection, scoped to *user_id*.

        Returns a list of dicts, each containing::

            {"text": str, "score": float, "metadata": dict}

        Only results with ``score >= score_threshold`` are included.
        Results below threshold are silently dropped.
        """
        vector = self.embedding_fn(query)

        results = self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
            score_threshold=score_threshold,
            with_payload=True,
        )

        out: list[dict] = []
        for hit in results:
            out.append({
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {
                    k: v for k, v in hit.payload.items()
                    if k not in ("text", "user_id")
                },
            })
        return out