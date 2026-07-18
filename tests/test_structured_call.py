"""Tests for agents/structured_call.py — Phase 3."""

import sys
import os
from unittest.mock import MagicMock
import pytest
import json

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.schemas import PlannerOutput
from agents.structured_call import call_agent_structured, StructuredCallError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_router():
    """Return a ModelRouter-like object whose ``call`` method is a MagicMock."""
    router = MagicMock()
    router.config = {"planner": {"primary": {"provider": "groq", "model": "llama-3.3-70b-versatile"}}}
    return router


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCallAgentStructured:
    """Three test cases for the repair loop."""

    # ------------------------------------------------------------------
    # Test 1 — happy path
    # ------------------------------------------------------------------

    def test_happy_path_returns_valid_planner_output(self, mock_router):
        """
        Mock router.call to return valid JSON matching PlannerOutput.
        Confirm the returned object is a PlannerOutput with correct fields.
        """
        valid_json = json.dumps({
            "tasks": ["Research A", "Code B", "Analyze C"],
            "reasoning": "Need to research first, then build, then analyze.",
            "priority_order": [0, 1, 2],
        })
        mock_router.call.return_value = valid_json

        result = call_agent_structured(
            router=mock_router,
            agent_name="planner",
            messages=[{"role": "user", "content": "plan this"}],
            schema=PlannerOutput,
        )

        assert isinstance(result, PlannerOutput)
        assert result.tasks == ["Research A", "Code B", "Analyze C"]
        assert "Need to research" in result.reasoning
        assert result.priority_order == [0, 1, 2]
        mock_router.call.assert_called_once()

    # ------------------------------------------------------------------
    # Test 2 — one repair
    # ------------------------------------------------------------------

    def test_one_repair_recovers(self, mock_router):
        """
        Mock router.call to return malformed JSON on the first call,
        then valid JSON on the second call.  Confirm it returns
        successfully after exactly one repair.
        """
        responses = iter([
            "not valid json at all{{{",
            json.dumps({
                "tasks": ["Fix bug", "Write test"],
                "reasoning": "Fix the bug first, then cover with tests.",
                "priority_order": [0, 1],
            }),
        ])
        mock_router.call.side_effect = lambda *a, **kw: next(responses)

        # messages list should be mutated (repair msg appended) after first fail
        messages = [{"role": "user", "content": "debug this"}]

        result = call_agent_structured(
            router=mock_router,
            agent_name="planner",
            messages=messages,
            schema=PlannerOutput,
        )

        assert isinstance(result, PlannerOutput)
        assert result.tasks == ["Fix bug", "Write test"]

        # The router should have been called twice (initial + 1 repair)
        assert mock_router.call.call_count == 2  # called 2 times total

        # messages should have grown by 1 (the repair prompt)
        assert len(messages) == 2
        assert "Fix ONLY" in messages[1]["content"]

    # ------------------------------------------------------------------
    # Test 3 — exhaust repairs
    # ------------------------------------------------------------------

    def test_exhaust_repairs_raises_structured_call_error(self, mock_router):
        """
        Mock router.call to always return malformed JSON.
        Confirm StructuredCallError is raised after exactly max_repairs attempts,
        not before.
        """
        always_bad = "this is not json"
        mock_router.call.return_value = always_bad

        max_repairs = 3

        with pytest.raises(StructuredCallError) as exc_info:
            call_agent_structured(
                router=mock_router,
                agent_name="planner",
                messages=[{"role": "user", "content": "do it"}],
                schema=PlannerOutput,
                max_repairs=max_repairs,
            )

        # The router should have been called exactly max_repairs + 1 times
        # (initial call + 3 repair attempts)
        assert mock_router.call.call_count == max_repairs + 1

        err = exc_info.value
        assert err.agent_name == "planner"
        assert err.raw_response == always_bad
        assert err.validation_error is not None
        assert "failed to produce valid" in err.args[0]
