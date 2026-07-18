"""Tests for agents/coder.py — Phase 8."""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Patch heavy module-level initialisations before importing coder
# ---------------------------------------------------------------------------

with patch("agents.coder.ModelRouter") as _MockRouter, \
     patch("agents.coder.PermissionLayer") as _MockPL, \
     patch("agents.coder.build_registry") as _MockBuildReg:

    _MockRouter.return_value = MagicMock()
    _MockPL.return_value = MagicMock()

    _fake_registry = MagicMock()
    _fake_registry.catalog_for.return_value = [
        {"name": "run_python", "description": "Execute Python code.", "parameters": {}},
        {"name": "read_file",  "description": "Read a file.",        "parameters": {}},
        {"name": "write_file", "description": "Write a file.",       "parameters": {}},
        {"name": "memory_write", "description": "Write memory.",     "parameters": {}},
    ]
    _MockBuildReg.return_value = _fake_registry

    from agents.coder import coder_node, _parse_tool_calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    state = {
        "task": "Write a function that reverses a string.",
        "subtasks": ["Write code", "Test code"],
        "assignments": {"0": "coder", "1": "reviewer"},
        "research_notes": ["Python has built-in slicing: s[::-1]"],
        "code": "",
        "run_id": "run_test_coder",
    }
    state.update(overrides)
    return state


def mock_coder_output(**kwargs) -> MagicMock:
    defaults = {
        "code": "def reverse(s: str) -> str:\n    return s[::-1]",
        "language": "python",
        "explanation": "Uses Python slicing to reverse the string.",
    }
    defaults.update(kwargs)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ===================================================================
# Tests: happy path
# ===================================================================

class TestCoderNodeHappyPath:

    def test_returns_code_from_structured_output(self):
        """No tool calls → straight to structured call → returns code."""
        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured") as mock_struct:

            mock_router.call.return_value = "Here is the code (no tool calls)."
            mock_struct.return_value = mock_coder_output()

            result = coder_node(make_state())

            assert result["code"] == "def reverse(s: str) -> str:\n    return s[::-1]"
            mock_struct.assert_called_once()

    def test_agentic_loop_executes_tool_calls(self):
        """Model requests run_python → execute_tool is called → loop continues."""
        tool_response = '{"tool_calls": [{"name": "run_python", "args": {"code": "print(1+1)"}}]}'
        no_tool_response = "Done testing."

        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured") as mock_struct, \
             patch("agents.coder.execute_tool") as mock_exec:

            # First call: tool use. Second call: no tool use (break).
            mock_router.call.side_effect = [tool_response, no_tool_response]
            mock_exec.return_value = "2\n"
            mock_struct.return_value = mock_coder_output()

            result = coder_node(make_state())

            mock_exec.assert_called_once()
            assert result["code"]

    def test_agentic_loop_caps_at_max_iterations(self):
        """Even if every response has tool calls, loop stops at 3."""
        tool_response = '{"tool_calls": [{"name": "run_python", "args": {"code": "x"}}]}'

        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured") as mock_struct, \
             patch("agents.coder.execute_tool") as mock_exec:

            mock_router.call.return_value = tool_response
            mock_exec.return_value = "ok"
            mock_struct.return_value = mock_coder_output()

            coder_node(make_state())

            # router.call invoked 3 times (agentic loop), then structured call
            assert mock_router.call.call_count == 3

    def test_tool_results_passed_to_structured_call(self):
        """Tool results must appear in messages for the final structured call."""
        tool_response = '{"tool_calls": [{"name": "run_python", "args": {"code": "print(42)"}}]}'

        captured_messages = []

        def capture(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return mock_coder_output()

        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured", side_effect=capture), \
             patch("agents.coder.execute_tool", return_value="42\n"):

            mock_router.call.side_effect = [tool_response, "No more tools."]
            coder_node(make_state())

        contents = [m["content"] for m in captured_messages]
        assert any("42" in c for c in contents)

    def test_research_notes_in_user_message(self):
        """Research context must be passed to the model."""
        captured_messages = []

        def capture(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return mock_coder_output()

        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured", side_effect=capture):

            mock_router.call.return_value = "No tools needed."
            coder_node(make_state(research_notes=["slicing is fast"]))

        user_msgs = [m["content"] for m in captured_messages if m["role"] == "user"]
        assert any("slicing is fast" in c for c in user_msgs)


# ===================================================================
# Tests: fallback paths
# ===================================================================

class TestCoderNodeFallback:

    def test_fallback_on_structured_call_error(self):
        from agents.structured_call import StructuredCallError

        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured") as mock_struct:

            mock_router.call.return_value = "no tools"
            mock_struct.side_effect = StructuredCallError(
                agent_name="coder",
                raw_response="{}",
                validation_error=ValueError("bad"),
            )

            result = coder_node(make_state())
            assert result["code"] == "# Code generation failed"

    def test_fallback_on_router_error(self):
        with patch("agents.coder.router") as mock_router:
            mock_router.call.side_effect = RuntimeError("All providers failed")
            result = coder_node(make_state())
            assert result["code"] == "# Code generation failed"

    def test_blocked_tool_does_not_crash(self):
        """If a tool is not in _CODER_TOOLS, an error message is appended but no crash."""
        bad_tool_response = '{"tool_calls": [{"name": "web_search", "args": {"query": "x"}}]}'

        with patch("agents.coder.router") as mock_router, \
             patch("agents.coder.call_agent_structured") as mock_struct:

            mock_router.call.side_effect = [bad_tool_response, "done"]
            mock_struct.return_value = mock_coder_output()

            result = coder_node(make_state())
            assert result["code"]  # should still succeed
