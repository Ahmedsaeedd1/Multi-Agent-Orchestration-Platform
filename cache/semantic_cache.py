"""
Layer 1 (Redis): exact SHA256 match on (agent_name, messages).
Layer 2 (Qdrant): semantic match on the last user message.

Fail-open: if Redis or Qdrant is unreachable, log the error and fall
through to ``call_fn`` — the cache never blocks an LLM call.

Design note
-----------
The Qdrant tier embeds the *query* (last user message), not the response.
The actual LLM response is stored in ``metadata["response"]`` so it can
be retrieved on a semantic hit.  This keeps query vectors semantically
comparable to future query vectors, which is what makes recall actually work.

call_fn contract
----------------
``call_fn`` MUST be a callable that accepts (agent_name, messages) and
returns a plain **string** — not a ChatCompletion object.  Example wiring:

    cache = SemanticCache(redis_store, qdrant_store)
    result = cache.get_cached_or_call(
        agent_name="researcher",
        messages=messages,
        call_fn=lambda name, msgs: router.call(name, msgs),
    )

If call_fn returns a ChatCompletion object the cache will store and return
that object on future hits, which will break every downstream string
operation silently.
"""

import hashlib
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class SemanticCache:
    """Two-tier cache backed by Redis (exact) and Qdrant (semantic)."""

    def __init__(
        self,
        redis_store,
        qdrant_store,
        enabled: bool = True,
        exact_ttl: int = 3600,
        semantic_threshold: float = 0.92,
    ):
        self.redis = redis_store
        self.qdrant = qdrant_store
        self.enabled = enabled
        self.exact_ttl = exact_ttl
        self.semantic_threshold = semantic_threshold

    # ------------------------------------------------------------------
    # Key builder
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(agent_name: str, messages: list, **kwargs) -> str:
        """
        Constructs a cache key using the agent name, kwargs, and the full task text
        to prevent collision issues from hashing or truncation.
        """
        last_content = messages[-1]["content"] if messages else ""
        kwargs_str = json.dumps(kwargs, sort_keys=True) if kwargs else "{}"
        return f"cache:exact:{agent_name}:{kwargs_str}:{last_content}"

    # ------------------------------------------------------------------
    # Core orchestrator
    # ------------------------------------------------------------------

    def get_cached_or_call(
        self,
        agent_name: str,
        messages: list,
        call_fn: Callable,
        **kwargs,
    ) -> str:
        """
        Return a cached response (str) if available, otherwise call
        *call_fn*, cache the result, and return it.

        call_fn must return a plain string — see module docstring.

        Steps
        -----
        1. If ``self.enabled`` is False → skip straight to *call_fn*.
        2. Redis exact-match lookup → return immediately if hit.
        3. Qdrant semantic match on last user message → return if
           score >= threshold.
        4. Miss: invoke *call_fn*, assert result is str, write to both tiers.
        5. Any Redis or Qdrant error → log + continue (fail-open).
        """
        if not self.enabled:
            logger.debug("Cache disabled — calling call_fn directly")
            return call_fn(agent_name, messages)

        last_content = messages[-1]["content"] if messages else ""

        # ── Layer 1: exact match ──────────────────────────────────────
        key = self.cache_key(agent_name, messages, **kwargs)
        try:
            cached = self.redis.read(key)
            if cached is not None:
                logger.debug("Cache HIT (exact) key=%s", key)
                # cached is always stored as {"response": str}
                return cached["response"] if isinstance(cached, dict) else str(cached)
        except Exception as e:
            logger.warning("Redis cache read failed (fail-open): %s", e)

        # ── Layer 2: semantic match (scoped per agent to prevent cross-topic poisoning) ────
        cache_ns = f"__cache__{agent_name}__"
        try:
            results = self.qdrant.retrieve_memory(
                user_id=cache_ns,
                query=last_content,
                top_k=1,
                score_threshold=self.semantic_threshold,
            )
            if results:
                cached_response = results[0]["metadata"].get("response")
                if cached_response is not None:
                    logger.info(
                        "Cache HIT (semantic) agent=%s score=%.4f threshold=%.2f",
                        agent_name,
                        results[0]["score"],
                        self.semantic_threshold,
                    )
                    return str(cached_response)
        except Exception as e:
            logger.warning("Qdrant cache read failed (fail-open): %s", e)

        # ── Miss: invoke the real LLM call ────────────────────────────
        response = call_fn(agent_name, messages)

        # Guard: call_fn must return a string — catch misuse early
        if not isinstance(response, str):
            raise TypeError(
                f"call_fn for agent '{agent_name}' returned {type(response).__name__!r}, "
                "expected str. Wrap router.call() in a lambda that extracts the content string."
            )

        # ── Write back to Redis (exact) ───────────────────────────────
        try:
            self.redis.write(key, {"response": response}, ttl=self.exact_ttl)
        except Exception as e:
            logger.warning("Redis cache write failed (fail-open): %s", e)

        # ── Write back to Qdrant (semantic, scoped per agent) ────────
        try:
            self.qdrant.write_memory(
                user_id=cache_ns,
                text=last_content,
                metadata={
                    "agent_name": agent_name,
                    "cache_key": key,
                    "response": response,
                },
            )
        except Exception as e:
            logger.warning("Qdrant cache write failed (fail-open): %s", e)

        return response