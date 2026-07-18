"""Unit tests for cache/semantic_cache.py — Phase 5.

All tests use ``pytest.mark.unit`` (mocked, no live Docker required).
"""

import sys
import os
from unittest.mock import MagicMock, call
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cache.semantic_cache import SemanticCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    r = MagicMock()
    r.read.return_value = None  # cache miss by default
    return r


@pytest.fixture
def mock_qdrant():
    q = MagicMock()
    q.retrieve_memory.return_value = []  # cache miss by default
    return q


@pytest.fixture
def call_fn():
    fn = MagicMock()
    fn.return_value = "fresh response from LLM"
    return fn


@pytest.fixture
def cache(mock_redis, mock_qdrant):
    return SemanticCache(
        redis_store=mock_redis,
        qdrant_store=mock_qdrant,
        enabled=True,
        exact_ttl=3600,
        semantic_threshold=0.92,
    )


@pytest.fixture
def messages():
    return [{"role": "user", "content": "What is the capital of France?"}]


# ===================================================================
# Tests
# ===================================================================

@pytest.mark.unit
class TestSemanticCache:

    # ------------------------------------------------------------------
    # Exact hit
    # ------------------------------------------------------------------

    def test_exact_hit_does_not_call_llm(self, cache, mock_redis, call_fn, messages):
        """
        Redis returns a cached entry → call_fn must not be invoked at all.
        """
        mock_redis.read.return_value = {"response": "cached response"}

        result = cache.get_cached_or_call("researcher", messages, call_fn)

        assert result == "cached response"
        call_fn.assert_not_called()
        mock_redis.read.assert_called_once()

    # ------------------------------------------------------------------
    # Semantic hit
    # ------------------------------------------------------------------

    def test_semantic_hit_does_not_call_llm(
        self, cache, mock_redis, mock_qdrant, call_fn, messages,
    ):
        """
        First call is a miss → invokes call_fn.  Second call with a
        paraphrased message hits Qdrant (score >= threshold) and must
        not invoke call_fn again.
        """
        # First call: miss → go to LLM
        call_fn.return_value = "Paris is the capital of France."
        result1 = cache.get_cached_or_call("researcher", messages, call_fn)
        assert result1 == "Paris is the capital of France."
        assert call_fn.call_count == 1

        # Reset Redis to miss on second call so we exercise the semantic path
        mock_redis.read.return_value = None

        # Second call: paraphrased query → Qdrant hit
        # Note: response is in metadata["response"], not in "text"
        mock_qdrant.retrieve_memory.return_value = [
            {
                "text": "France's capital city?",
                "score": 0.95,
                "metadata": {"response": "Paris is the capital of France."},
            }
        ]
        para_messages = [{"role": "user", "content": "France's capital city?"}]

        result2 = cache.get_cached_or_call("researcher", para_messages, call_fn)

        assert result2 == "Paris is the capital of France."
        assert call_fn.call_count == 1  # still only called once

    # ------------------------------------------------------------------
    # Fail-open: Redis down
    # ------------------------------------------------------------------

    def test_fail_open_on_redis_down(
        self, cache, mock_redis, mock_qdrant, call_fn, messages,
    ):
        """
        Redis.read raises → fall through to Qdrant, then to call_fn.
        Must still return a valid response, never raise.
        """
        mock_redis.read.side_effect = ConnectionError("Redis unreachable")
        mock_qdrant.retrieve_memory.side_effect = ConnectionError("Qdrant unreachable")

        result = cache.get_cached_or_call("researcher", messages, call_fn)

        assert result == "fresh response from LLM"
        call_fn.assert_called_once()

    # ------------------------------------------------------------------
    # Fail-open: Qdrant down
    # ------------------------------------------------------------------

    def test_fail_open_on_qdrant_down(
        self, cache, mock_redis, mock_qdrant, call_fn, messages,
    ):
        """
        Qdrant.retrieve_memory raises → fall through to call_fn.
        Must still return a valid response, never raise.
        """
        mock_redis.read.return_value = None
        mock_qdrant.retrieve_memory.side_effect = ConnectionError("Qdrant unreachable")

        result = cache.get_cached_or_call("researcher", messages, call_fn)

        assert result == "fresh response from LLM"
        call_fn.assert_called_once()

    # ------------------------------------------------------------------
    # CACHE_ENABLED=false
    # ------------------------------------------------------------------

    def test_disabled_cache_always_calls_llm(
        self, mock_redis, mock_qdrant, call_fn, messages,
    ):
        """
        When enabled=False, every call goes straight to call_fn.
        Redis and Qdrant must never be touched.
        """
        disabled_cache = SemanticCache(
            redis_store=mock_redis,
            qdrant_store=mock_qdrant,
            enabled=False,
        )

        r1 = disabled_cache.get_cached_or_call("researcher", messages, call_fn)
        r2 = disabled_cache.get_cached_or_call("researcher", messages, call_fn)

        assert r1 == "fresh response from LLM"
        assert r2 == "fresh response from LLM"
        assert call_fn.call_count == 2
        mock_redis.read.assert_not_called()
        mock_qdrant.retrieve_memory.assert_not_called()

    # ------------------------------------------------------------------
    # Write-back correctness: query embedded, response in metadata
    # ------------------------------------------------------------------

    def test_write_back_embeds_query_not_response(
        self, cache, mock_redis, mock_qdrant, call_fn, messages,
    ):
        """
        On a cache miss, write_memory must be called with the QUERY as
        ``text`` and the LLM response in ``metadata["response"]``.
        This is what makes semantic recall actually work — embedding the
        query means future similar queries land nearby in vector space.
        """
        call_fn.return_value = "Paris is the capital of France."

        cache.get_cached_or_call("researcher", messages, call_fn)

        mock_qdrant.write_memory.assert_called_once()
        _, kwargs = mock_qdrant.write_memory.call_args
        # text must be the query, not the LLM response
        assert kwargs["text"] == messages[-1]["content"]
        # response must be stored in metadata
        assert kwargs["metadata"]["response"] == "Paris is the capital of France."