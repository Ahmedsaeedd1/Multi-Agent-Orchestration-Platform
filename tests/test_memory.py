"""Integration tests for memory stores — Phase 4."""

import os
import time
import pytest
import requests

from memory.redis_store import RedisStore
from memory.qdrant_store import QdrantStore

# Known user_ids used across these tests — cleaned up before each test run
TEST_USER_IDS = ["test_user_same", "test_user_a", "test_user_b", "test_user_para"]


# ---------------------------------------------------------------------------
# Embedding helper for Qdrant tests
# ---------------------------------------------------------------------------

def _ollama_embed(text: str) -> list[float]:
    """Embed *text* via the local Ollama nomic-embed-text model."""
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def redis_store():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return RedisStore(url=url, default_ttl=60)


@pytest.fixture(scope="module")
def qdrant_store():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Use the task_history collection (created in Phase 0)
    return QdrantStore(url=url, collection="task_history", embedding_fn=_ollama_embed)


@pytest.fixture(autouse=True)
def clean_collection(qdrant_store):
    """
    Remove all points belonging to the known test user_ids before each
    test, so stale data from prior runs never leaks into assertions and
    the collection doesn't grow unbounded across test runs.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchAny

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    def _purge():
        try:
            client.delete(
                collection_name="task_history",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchAny(any=TEST_USER_IDS),
                        )
                    ]
                ),
            )
        except Exception:
            # Collection may not exist yet on first run, or delete-by-filter
            # may no-op if there's nothing to remove — either is fine here.
            pass

    _purge()   # clean before the test
    yield
    _purge()   # clean after the test too, so failures don't leave residue


# ===================================================================
# Redis tests
# ===================================================================

@pytest.mark.integration
class TestRedisStore:
    """Redis short-term memory tests."""

    def test_write_and_read_dict(self, redis_store):
        """Write a dict, read it back, confirm exact match."""
        key = "test:write_read"
        data = {"task": "test", "count": 42, "tags": ["a", "b"]}
        try:
            redis_store.write(key, data, ttl=60)
            result = redis_store.read(key)
            assert result == data
        finally:
            redis_store.delete(key)  # always clean up, even on assert failure

    def test_ttl_expiry(self, redis_store):
        """Write with a 1s TTL, wait 2s, confirm read() returns None."""
        key = "test:ttl_expiry"
        redis_store.write(key, {"temp": True}, ttl=1)
        # Confirm it's there immediately
        assert redis_store.read(key) is not None
        time.sleep(2)
        assert redis_store.read(key) is None

    def test_build_key_state(self):
        """build_key("user123", "state") → "session:user123:state"."""
        assert RedisStore.build_key("user123", "state") == "session:user123:state"

    def test_build_key_agent(self):
        """build_key("user123", "agent", "planner") → "session:user123:agent:planner"."""
        assert (
            RedisStore.build_key("user123", "agent", "planner")
            == "session:user123:agent:planner"
        )


# ===================================================================
# Qdrant tests
# ===================================================================

@pytest.mark.integration
class TestQdrantStore:
    """Qdrant long-term semantic memory tests."""

    def test_write_and_retrieve_same_text(self, qdrant_store):
        """
        Write a memory, retrieve it with the same text as query.
        Confirm the result comes back with score >= 0.5 and correct user_id.
        """
        user_id = "test_user_same"
        text = "The capital of France is Paris."
        pid = qdrant_store.write_memory(user_id, text, {"source": "test"})
        assert pid is not None

        results = qdrant_store.retrieve_memory(
            user_id, query=text, top_k=5, score_threshold=0.5
        )

        assert len(results) >= 1
        best = results[0]
        assert best["score"] >= 0.5
        assert best["text"] == text

    def test_user_id_isolation(self, qdrant_store):
        """
        Write two memories under different user_ids, query under one,
        confirm the other user's memory does not leak through.
        """
        user_a = "test_user_a"
        user_b = "test_user_b"

        qdrant_store.write_memory(user_a, "Secret A: pricing model", {"type": "pricing"})
        qdrant_store.write_memory(user_b, "Secret B: launch date", {"type": "launch"})

        # Query as user_a
        results = qdrant_store.retrieve_memory(
            user_a, query="pricing", top_k=10, score_threshold=0.0
        )

        texts_found = [r["text"] for r in results]
        assert any("Secret A" in t for t in texts_found), (
            "User A should find their own memory"
        )
        assert all("Secret B" not in t for t in texts_found), (
            "User A should NOT see User B's memory"
        )

    def test_paraphrased_query(self, qdrant_store):
        """
        Write a memory, retrieve with a semantically similar but different
        query, confirm it still comes back above threshold.
        """
        user_id = "test_user_para"
        text = "Python is a versatile programming language used for web development."
        qdrant_store.write_memory(user_id, text, {"topic": "python"})

        # Paraphrased query
        results = qdrant_store.retrieve_memory(
            user_id,
            query="Python language for building websites",
            top_k=5,
            score_threshold=0.5,
        )

        assert len(results) >= 1
        best = results[0]
        assert best["score"] >= 0.5
        assert "Python" in best["text"] or "python" in best["text"]