"""Tests for researcher memory retrieval."""

import json
from unittest.mock import patch, MagicMock

from agents.researcher import researcher_node

def make_state(**overrides) -> dict:
    state = {
        "task": "Test research task",
        "user_id": "test_user_001",
        "subtasks": ["subtask 1"],
        "assignments": {"0": "researcher"},
        "run_id": "run_001"
    }
    state.update(overrides)
    return state

@patch("agents.researcher._task_history_store.retrieve_memory")
@patch("agents.researcher.router.call")
@patch("agents.researcher.call_agent_structured")
@patch("agents.researcher._write_findings_to_memory")
def test_memory_retrieval_called(mock_write, mock_structured, mock_call, mock_retrieve):
    """Confirm retrieve_memory is called with the correct parameters."""
    mock_retrieve.return_value = []
    
    # Mock structured output
    mock_out = MagicMock()
    mock_out.findings = ["Finding 1"]
    mock_out.sources = ["http://source.com"]
    mock_out.confidence = 0.9
    mock_structured.return_value = mock_out

    state = make_state()
    researcher_node(state)

    mock_retrieve.assert_called_once_with(
        user_id="test_user_001",
        query="Test research task",
        top_k=3,
        score_threshold=0.75
    )

@patch("agents.researcher._task_history_store.retrieve_memory")
@patch("agents.researcher.router.call")
@patch("agents.researcher.call_agent_structured")
@patch("agents.researcher._write_findings_to_memory")
def test_memory_context_injected(mock_write, mock_structured, mock_call, mock_retrieve):
    """Confirm past findings are injected into the system prompt."""
    mock_retrieve.return_value = [
        {"text": "Past finding 1", "score": 0.9},
        {"text": "Past finding 2", "score": 0.85}
    ]
    
    # Mock structured output
    mock_out = MagicMock()
    mock_out.findings = ["Finding 1"]
    mock_out.sources = ["http://source.com"]
    mock_out.confidence = 0.9
    mock_structured.return_value = mock_out

    state = make_state()
    researcher_node(state)

    # Check the messages passed to router.call
    messages = mock_call.call_args[0][1]
    system_content = messages[0]["content"]
    
    assert "Relevant findings from past research" in system_content
    assert "- Past finding 1" in system_content
    assert "- Past finding 2" in system_content

@patch("agents.researcher._task_history_store.retrieve_memory")
@patch("agents.researcher.router.call")
@patch("agents.researcher.call_agent_structured")
@patch("agents.researcher._write_findings_to_memory")
def test_memory_failure_non_fatal(mock_write, mock_structured, mock_call, mock_retrieve):
    """Confirm researcher_node completes normally even if memory retrieval fails."""
    mock_retrieve.side_effect = Exception("Database connection failed")
    
    # Mock structured output
    mock_out = MagicMock()
    mock_out.findings = ["Finding 1"]
    mock_out.sources = ["http://source.com"]
    mock_out.confidence = 0.9
    mock_structured.return_value = mock_out

    state = make_state()
    result = researcher_node(state)

    # The node should complete normally despite the Exception
    assert result == {"research_notes": ["Finding 1"]}
    
    # The system message should not contain past memory
    messages = mock_call.call_args[0][1]
    system_content = messages[0]["content"]
    assert "Relevant findings from past research" not in system_content
