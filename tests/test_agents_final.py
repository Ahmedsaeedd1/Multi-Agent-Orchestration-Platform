"""Tests for the final three agents: data_analyst, reviewer, aggregator."""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Patch module-level initialisations
# ---------------------------------------------------------------------------

with patch("agents.data_analyst.ModelRouter") as _MockRouterDA, \
     patch("agents.data_analyst.PermissionLayer") as _MockPLDA, \
     patch("agents.data_analyst.build_registry") as _MockBuildRegDA, \
     patch("agents.reviewer.ModelRouter") as _MockRouterRev, \
     patch("agents.aggregator.ModelRouter") as _MockRouterAgg:

    _MockRouterDA.return_value = MagicMock()
    _MockPLDA.return_value = MagicMock()
    _fake_registry = MagicMock()
    _fake_registry.catalog_for.return_value = []
    _MockBuildRegDA.return_value = _fake_registry

    _MockRouterRev.return_value = MagicMock()
    _MockRouterAgg.return_value = MagicMock()

    from agents.data_analyst import data_analyst_node
    from agents.reviewer import reviewer_node
    from agents.aggregator import aggregator_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    state = {
        "task": "Test task",
        "research_notes": ["Note 1"],
        "code": "print(1)",
        "analysis": "Some analysis",
        "review_cycles": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Data Analyst Tests
# ---------------------------------------------------------------------------

def mock_analyst_output() -> MagicMock:
    m = MagicMock()
    m.summary = "Data is good."
    m.insights = ["Trend is up."]
    m.code_used = None
    return m

def test_data_analyst_returns_analysis():
    with patch("agents.data_analyst.router") as mock_router, \
         patch("agents.data_analyst.call_agent_structured") as mock_struct:
        
        mock_router.call.return_value = "No tool calls."
        mock_struct.return_value = mock_analyst_output()

        result = data_analyst_node(make_state())
        assert result["analysis"] == "Data is good."


# ---------------------------------------------------------------------------
# Reviewer Tests
# ---------------------------------------------------------------------------

def mock_reviewer_output(verdict="approved") -> MagicMock:
    m = MagicMock()
    m.verdict = verdict
    m.feedback = "Looks good"
    return m

def test_reviewer_approves():
    with patch("agents.reviewer.call_agent_structured") as mock_struct:
        mock_struct.return_value = mock_reviewer_output("approved")
        
        result = reviewer_node(make_state(review_cycles=1, code_verified=True))
        
        assert result["review_feedback"] == "approved"
        assert result["review_cycles"] == 2

def test_reviewer_fallback():
    with patch("agents.reviewer.call_agent_structured") as mock_struct:
        mock_struct.side_effect = Exception("Crash")
        
        result = reviewer_node(make_state(review_cycles=2, code_verified=True))
        
        assert result["review_feedback"] == "needs_revision"
        assert result["review_cycles"] == 3
        assert "error" in result
        assert "Crash" in result["error"]


# ---------------------------------------------------------------------------
# Aggregator Tests
# ---------------------------------------------------------------------------

def mock_aggregator_output() -> MagicMock:
    m = MagicMock()
    m.final_answer = "# Final Answer"
    m.sources_used = []
    m.agent_steps = []
    return m

def test_aggregator_combines():
    with patch("agents.aggregator.call_agent_structured") as mock_struct:
        mock_struct.return_value = mock_aggregator_output()
        
        result = aggregator_node(make_state())
        
        assert result["final_output"] == "# Final Answer"
