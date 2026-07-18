"""
Phase 4 — Short-Term / Working Memory (Redis)

Key schema (from Section 2.4):
    session:{user_id}:state        → current AgentState JSON
    session:{user_id}:history      → list of past messages (JSON array)
    session:{user_id}:agent:{name} → last output of a specific agent
"""

import json
import logging
from typing import Any

import redis as redis_lib

logger = logging.getLogger(__name__)


class RedisStore:
    """Thin wrapper around Redis for session-scoped state."""

    def __init__(self, url: str, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        try:
            self._client = redis_lib.from_url(url, decode_responses=True)
            self._client.ping()
        except Exception as e:
            logger.error("Failed to connect to Redis at %s: %s", url, e)
            raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def write(self, key: str, value: dict | list, ttl: int | None = None) -> None:
        """
        Serialise *value* as JSON and store at *key* with an optional TTL
        (falls back to ``self.default_ttl``).
        """
        payload = json.dumps(value)
        self._client.setex(key, ttl or self.default_ttl, payload)

    def read(self, key: str) -> dict | list | None:
        """
        Retrieve and deserialise *key*.  Returns ``None`` if the key does
        not exist or has expired.
        """
        raw = self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def delete(self, key: str) -> None:
        """Remove *key* from Redis."""
        self._client.delete(key)

    # ------------------------------------------------------------------
    # Key builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_key(user_id: str, kind: str, agent_name: str | None = None) -> str:
        """
        Build a namespaced Redis key.

        Examples::

            RedisStore.build_key("user123", "state")          → "session:user123:state"
            RedisStore.build_key("user123", "agent", "coder") → "session:user123:agent:coder"
        """
        if agent_name:
            return f"session:{user_id}:{kind}:{agent_name}"
        return f"session:{user_id}:{kind}"