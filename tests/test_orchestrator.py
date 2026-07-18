"""Tests for agents/orchestrator.py — Phase 8."""

import json
import sys
import os
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch ModelRouter before the import so it doesn't try to load agents.yaml
with patch("agents.orchestrator.ModelRouter") as MockRouter:
    MockRouter.return_value = MagicMock()
    from agents.orchestrator import orchestrator_node, _fallback


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    """Return a minimal AgentState with sensible defaults."""
    state = {
        "task": "Research competitors and write a summary.",
        "subtasks": [],
        "assignments": {},
        "research_notes": [],
        "code": "",
        "analysis": "",
        "review_feedback": "",
        "final_output": "",
        "run_id": "test_run_001",
        "review_cycles": 0,
    }
    state.update(overrides)
    return state


def mock_orchestrator_output(**kwargs) -> MagicMock:
    """Return a MagicMock that quacks like an OrchestratorOutput."""
    defaults = {
        "subtasks": ["Research competitors", "Write summary"],
        "assignments": {"0": "researcher", "1": "coder"},
        "reasoning": "Research first, then write.",
        "max_review_cycles": 3,
    }
    defaults.update(kwargs)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ===================================================================
# Tests
# ===================================================================

class TestOrchestratorNode:
    """orchestrator_node behaviour."""

    def test_orchestrator_returns_subtasks(self):
        """
        Mock ``call_agent_structured`` to return a valid OrchestratorOutput.
        Confirm the state update has ``subtasks`` and ``assignments``.
        """
        with patch("agents.orchestrator.call_agent_structured") as mock_call:
            mock_call.return_value = mock_orchestrator_output()

            state = make_state(task="Research competitors and write a summary.")
            result = orchestrator_node(state)

            assert "subtasks" in result
            assert "assignments" in result
            assert result["subtasks"] == ["Research competitors", "Write summary"]
            assert result["assignments"]["0"] == "researcher"
            assert result["assignments"]["1"] == "coder"

    def test_orchestrator_fallback_on_error(self):
        """
        Mock ``call_agent_structured`` to raise ``StructuredCallError``.
        Confirm the fallback is returned instead of crashing.
        """
        from agents.structured_call import StructuredCallError

        with patch("agents.orchestrator.call_agent_structured") as mock_call:
            mock_call.side_effect = StructuredCallError(
                agent_name="orchestrator",
                raw_response="bad json",
                validation_error=ValueError("invalid"),
            )

            state = make_state(task="Do everything requested by the user.")
            result = orchestrator_node(state)

            assert result["subtasks"] == ["Do everything requested by the user."]
            assert result["assignments"]["0"] == "researcher"

    def test_fallback_function(self):
        """_fallback returns a single subtask assigned to researcher."""
        result = _fallback("test task")
        assert result["subtasks"] == ["test task"]
        assert result["assignments"] == {"0": "researcher"}

    def test_empty_task_uses_fallback(self):
        """An empty task should trigger the fallback path."""
        state = make_state(task="")
        result = orchestrator_node(state)
        assert result["subtasks"] == [""]
        assert result["assignments"] == {"0": "researcher"}