"""
Global pytest configuration.

Defines the ``integration`` marker and auto-skips integration tests when
the required external services (Redis, Qdrant, Ollama) are not available.
"""

import os

import pytest
import requests
from dotenv import load_dotenv

# Load .env so that integration-test markers can detect REDIS_URL / QDRANT_URL
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def pytest_configure(config):
    """Register the ``integration`` marker."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live Docker services "
        "(Redis + Qdrant) and a running Ollama instance. Skipped "
        "automatically when REDIS_URL / QDRANT_URL are not set or "
        "Ollama is unreachable.",
    )


def _ollama_reachable() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """
    Skip integration tests if the required environment variables are not
    set, or if Ollama (needed for embeddings) is not reachable.
    """
    redis_url = os.getenv("REDIS_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    ollama_up = _ollama_reachable()

    missing = []
    if not redis_url:
        missing.append("REDIS_URL")
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not ollama_up:
        missing.append("Ollama (localhost:11434 unreachable)")

    if missing:
        skip_integration = pytest.mark.skip(
            reason=f"Skipping integration test: missing/unreachable -> {', '.join(missing)}"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)