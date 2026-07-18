"""Tests for agents/researcher.py — Phase 8."""

import sys
import os
from unittest.mock import patch, MagicMock, call
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Patch heavy module-level initialisations before importing researcher
# ---------------------------------------------------------------------------

with patch("agents.researcher.ModelRouter") as _MockRouter, \
     patch("agents.researcher.PermissionLayer") as _MockPL, \
     patch("agents.researcher.build_registry") as _MockBuildReg:

    _MockRouter.return_value = MagicMock()
    _MockPL.return_value = MagicMock()

    # Build a minimal fake registry so catalog_for() works
    _fake_registry = MagicMock()
    _fake_registry.catalog_for.return_value = [
        {"name": "web_search", "description": "Search the web.", "parameters": {}},
        {"name": "web_fetch",  "description": "Fetch a URL.",    "parameters": {}},
        {"name": "memory_write", "description": "Write memory.", "parameters": {}},
    ]
    _MockBuildReg.return_value = _fake_registry

    from agents.researcher import (
        researcher_node,
        _parse_tool_calls,
        _execute_tool_calls,
        _write_findings_to_memory,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    state = {
        "task": "Research AI trends in 2025.",
        "subtasks": ["Find AI trends", "Summarise findings"],
        "assignments": {"0": "researcher", "1": "coder"},
        "research_notes": [],
        "run_id": "run_test_001",
    }
    state.update(overrides)
    return state


def mock_researcher_output(**kwargs) -> MagicMock:
    defaults = {
        "findings": ["LLMs are growing fast.", "Agentic AI is emerging."],
        "sources": ["https://example.com/ai-trends"],
        "confidence": 0.85,
    }
    defaults.update(kwargs)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ===================================================================
# Unit tests: _parse_tool_calls
# ===================================================================

class TestParseToolCalls:
    def test_parses_valid_tool_call(self):
        content = '{"tool_calls": [{"name": "web_search", "args": {"query": "AI 2025"}}]}'
        result = _parse_tool_calls(content)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"] == {"query": "AI 2025"}

    def test_parses_multiple_tool_calls(self):
        content = (
            '{"tool_calls": ['
            '  {"name": "web_search", "args": {"query": "q1"}},'
            '  {"name": "web_fetch",  "args": {"url": "https://example.com"}}'
            ']}'
        )
        result = _parse_tool_calls(content)
        assert len(result) == 2
        assert result[1]["name"] == "web_fetch"

    def test_returns_empty_on_no_json(self):
        assert _parse_tool_calls("No JSON here, just prose.") == []

    def test_returns_empty_on_empty_tool_calls(self):
        assert _parse_tool_calls('{"tool_calls": []}') == []

    def test_returns_empty_on_bad_json(self):
        assert _parse_tool_calls("{bad json!!!}") == []

    def test_supports_arguments_key_alias(self):
        """Some models emit 'arguments' instead of 'args'."""
        content = '{"tool_calls": [{"name": "web_search", "arguments": {"query": "x"}}]}'
        result = _parse_tool_calls(content)
        assert result[0]["args"] == {"query": "x"}

    def test_ignores_surrounding_prose(self):
        """JSON embedded in surrounding text should still be parsed."""
        content = 'Thinking...\n{"tool_calls": [{"name": "web_search", "args": {"query": "q"}}]}\nDone.'
        result = _parse_tool_calls(content)
        assert len(result) == 1


# ===================================================================
# Unit tests: researcher_node happy path
# ===================================================================

class TestResearcherNodeHappyPath:

    def test_returns_research_notes_from_findings(self):
        """
        Mock both router.call and call_agent_structured.
        Confirm research_notes come from output.findings.
        """
        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher.call_agent_structured") as mock_struct, \
             patch("agents.researcher._write_findings_to_memory"):

            mock_router.call.return_value = '{"tool_calls": []}'
            mock_struct.return_value = mock_researcher_output()

            state = make_state()
            result = researcher_node(state)

            assert result["research_notes"] == [
                "LLMs are growing fast.",
                "Agentic AI is emerging.",
            ]

    def test_tool_calls_are_executed(self):
        """
        When the first router response contains tool calls, execute_tool
        must be invoked for each call.
        """
        tool_response = (
            '{"tool_calls": ['
            '  {"name": "web_search", "args": {"query": "AI 2025"}}'
            ']}'
        )

        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher.call_agent_structured") as mock_struct, \
             patch("agents.researcher.execute_tool") as mock_exec, \
             patch("agents.researcher._write_findings_to_memory"):

            mock_router.call.return_value = tool_response
            mock_exec.return_value = "Top 3 results for AI 2025 ..."
            mock_struct.return_value = mock_researcher_output()

            state = make_state()
            result = researcher_node(state)

            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs["tool_name"] == "web_search"
            assert result["research_notes"]  # non-empty

    def test_second_call_passes_tool_results_in_messages(self):
        """
        Tool results must appear in the message list given to
        call_agent_structured (so the model sees what the tools returned).
        """
        tool_response = '{"tool_calls": [{"name": "web_search", "args": {"query": "AI"}}]}'

        captured_messages = []

        def capture_messages(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return mock_researcher_output()

        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher.call_agent_structured", side_effect=capture_messages), \
             patch("agents.researcher.execute_tool", return_value="search results here"), \
             patch("agents.researcher._write_findings_to_memory"):

            mock_router.call.return_value = tool_response
            researcher_node(make_state())

        contents = [m["content"] for m in captured_messages]
        assert any("search results here" in c for c in contents), (
            "Tool results should appear in messages passed to structured call"
        )

    def test_findings_written_to_memory(self):
        """_write_findings_to_memory must be called once on success."""
        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher.call_agent_structured") as mock_struct, \
             patch("agents.researcher._write_findings_to_memory") as mock_write:

            mock_router.call.return_value = '{"tool_calls": []}'
            mock_struct.return_value = mock_researcher_output()

            researcher_node(make_state())
            mock_write.assert_called_once()


# ===================================================================
# Unit tests: researcher_node fallback paths
# ===================================================================

class TestResearcherNodeFallback:

    def test_fallback_on_structured_call_error(self):
        """StructuredCallError → fallback research_notes returned."""
        from agents.structured_call import StructuredCallError

        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher.call_agent_structured") as mock_struct, \
             patch("agents.researcher._write_findings_to_memory"):

            mock_router.call.return_value = '{"tool_calls": []}'
            mock_struct.side_effect = StructuredCallError(
                agent_name="researcher",
                raw_response="{}",
                validation_error=ValueError("missing fields"),
            )

            result = researcher_node(make_state())
            assert result["research_notes"] == ["Research failed — no findings"]

    def test_fallback_on_router_error(self):
        """If router.call() itself raises, fallback is returned."""
        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher._write_findings_to_memory"):

            mock_router.call.side_effect = RuntimeError("All providers failed")
            result = researcher_node(make_state())
            assert result["research_notes"] == ["Research failed — no findings"]

    def test_memory_write_failure_does_not_crash(self):
        """
        _write_findings_to_memory absorbs its own errors internally.
        The node must still return actual findings, not the fallback.
        """
        with patch("agents.researcher.router") as mock_router, \
             patch("agents.researcher.call_agent_structured") as mock_struct, \
             patch("agents.researcher._write_findings_to_memory",
                   side_effect=ConnectionError("Qdrant down")):

            mock_router.call.return_value = '{"tool_calls": []}'
            mock_struct.return_value = mock_researcher_output()

            result = researcher_node(make_state())

            # Memory failure is absorbed — real findings still returned
            assert result["research_notes"] == [
                "LLMs are growing fast.",
                "Agentic AI is emerging.",
            ]