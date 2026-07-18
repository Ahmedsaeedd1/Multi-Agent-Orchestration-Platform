"""Tests for agents/planner.py."""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

with patch("agents.planner.ModelRouter") as MockRouter:
    MockRouter.return_value = MagicMock()
    from agents.planner import planner_node


def make_state(**overrides) -> dict:
    state = {
        "task": "Do it",
        "subtasks": ["subtask1", "subtask2"],
        "assignments": {"0": "coder", "1": "researcher"},
    }
    state.update(overrides)
    return state


def mock_planner_output(**kwargs) -> MagicMock:
    defaults = {
        "tasks": ["Step 1", "Step 2", "Step 3"],
        "reasoning": "This is logical.",
        "priority_order": [0, 1, 2],
    }
    defaults.update(kwargs)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


class TestPlannerNode:
    def test_planner_refines_subtasks(self):
        with patch("agents.planner.call_agent_structured") as mock_call:
            mock_call.return_value = mock_planner_output()

            state = make_state(subtasks=["old_task_1", "old_task_2"])
            result = planner_node(state)

            assert "subtasks" in result
            assert result["subtasks"] == ["Step 1", "Step 2", "Step 3"]

    def test_planner_fallback(self):
        from agents.structured_call import StructuredCallError

        with patch("agents.planner.call_agent_structured") as mock_call:
            mock_call.side_effect = StructuredCallError(
                agent_name="planner",
                raw_response="bad json",
                validation_error=ValueError("invalid"),
            )

            state = make_state(subtasks=["original_1", "original_2"])
            result = planner_node(state)

            assert result["subtasks"] == ["original_1", "original_2"]
