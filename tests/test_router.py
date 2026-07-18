"""Tests for router.py — Phase 2."""

import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Anchor to project root regardless of where pytest is invoked from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from router import ModelRouter, RateLimitOrProviderError, strip_thinking


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def disable_cache(monkeypatch):
    """Ensure the router does not hit the semantic cache during these unit tests."""
    monkeypatch.setenv("CACHE_ENABLED", "false")

@pytest.fixture
def mock_clients():
    """
    Replace every entry in router.CLIENTS with a MagicMock so no real
    HTTP calls are made.
    """
    import router as rtr

    mocks = {}
    for provider in list(rtr.CLIENTS.keys()):
        m = MagicMock()
        rtr.CLIENTS[provider] = m
        mocks[provider] = m
    yield mocks


class FakeResponse:
    """Minimal stand-in for an OpenAI ChatCompletion response."""

    def __init__(self, content: str):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content


# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    """Unit tests for the ``strip_thinking`` helper."""

    def test_strips_think_block(self):
        """Feed a response with <think> reasoning </think> actual answer → 'actual answer'."""
        raw = "<think>some reasoning</think>actual answer"
        assert strip_thinking(raw) == "actual answer"

    def test_strips_multiline_think_block(self):
        """Multi-line thinking blocks are removed entirely."""
        raw = "<think>\nstep 1\nstep 2\n</think>\nThe result is X."
        assert strip_thinking(raw) == "The result is X."

    def test_no_think_block(self):
        """No <think> tag → returned unchanged."""
        assert strip_thinking("Hello, world.") == "Hello, world."

    def test_empty_string(self):
        assert strip_thinking("") == ""

    def test_only_think_block(self):
        assert strip_thinking("<think>nothing</think>") == ""

    def test_trailing_whitespace_is_stripped(self):
        assert strip_thinking("<think>r</think>   answer   ") == "answer"


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class TestModelRouter:
    """Unit tests for the fallback chain and thinking-tag stripping."""

    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "agents.yaml")

    # ------------------------------------------------------------------
    # Spec test 1: primary 429 → fall through to fallback
    # ------------------------------------------------------------------

    def test_fallback_on_429(self, mock_clients):
        """
        Mock the primary provider to raise RateLimitOrProviderError,
        confirm the router falls through to the fallback and returns
        a successful response.

        Uses researcher: primary=groq, fallback=openrouter.
        Raises RateLimitOrProviderError directly so the test is not
        coupled to any specific exception-message format in router.py.
        """
        router = ModelRouter(config_path=self.CONFIG_PATH)

        groq_mock = mock_clients["groq"]
        or_mock = mock_clients["openrouter"]

        # Raise RateLimitOrProviderError directly — no string-parsing dependency
        groq_mock.chat.completions.create.side_effect = RateLimitOrProviderError(
            "429 rate limited"
        )

        # OpenRouter (fallback) succeeds
        or_mock.chat.completions.create.return_value = FakeResponse(
            "fallback response"
        )

        result = router.call("researcher", [{"role": "user", "content": "hi"}])

        assert result == "fallback response"
        groq_mock.chat.completions.create.assert_called()
        or_mock.chat.completions.create.assert_called()

    # ------------------------------------------------------------------
    # Spec test 2: strip_thinking removes think block from response
    # ------------------------------------------------------------------

    def test_thinking_model_gets_stripped(self, mock_clients):
        """
        Planner primary is deepseek-ai/DeepSeek-R1:novita (huggingface).
        Router must strip <think>...</think> before returning.
        """
        router = ModelRouter(config_path=self.CONFIG_PATH)
        router.config["planner"] = {
            "primary": {"provider": "huggingface", "model": "deepseek-ai/DeepSeek-R1:novita"},
            "fallback": []
        }

        hf_mock = mock_clients["huggingface"]
        hf_mock.chat.completions.create.return_value = FakeResponse(
            "<think>Let me reason about this step by step</think>The answer is 42."
        )

        result = router.call("planner", [{"role": "user", "content": "plan"}])

        assert result == "The answer is 42."
        assert "<think>" not in result

    # ------------------------------------------------------------------
    # Extra test: all providers fail → raises cleanly
    # ------------------------------------------------------------------

    def test_all_providers_fail_raises(self, mock_clients):
        """
        When primary and every fallback raise RateLimitOrProviderError,
        the router should raise rather than return None or loop forever.
        """
        router = ModelRouter(config_path=self.CONFIG_PATH)

        # Make every provider fail
        for mock in mock_clients.values():
            mock.chat.completions.create.side_effect = RateLimitOrProviderError(
                "all down"
            )

        with pytest.raises(Exception):
            router.call("researcher", [{"role": "user", "content": "hi"}])

    # ------------------------------------------------------------------
    # Regression: explicit temperature override must be honored
    # ------------------------------------------------------------------

    def test_explicit_temperature_override_is_honored(self, mock_clients):
        """
        Calling router.call(..., temperature=0.9) must actually use 0.9,
        not silently fall back to the agent's configured default.
        """
        router = ModelRouter(config_path=self.CONFIG_PATH)
        groq_mock = mock_clients["groq"]
        groq_mock.chat.completions.create.return_value = FakeResponse("ok")

        router.call("researcher", [{"role": "user", "content": "hi"}], temperature=0.9)

        _, kwargs = groq_mock.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.9
