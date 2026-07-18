import pytest
from unittest.mock import patch

from agents.orchestrator import orchestrator_node, _is_vague_task

def test_vague_task_patterns():
    vague_tasks = [
        "help",
        "help me with my project",
        "can you help",
        "do something useful",
        "i need help",
        "what should i do",
    ]
    for task in vague_tasks:
        assert _is_vague_task(task) is True, f"Failed to detect vague task: {task}"

@patch("agents.orchestrator.call_agent_structured")
def test_vague_task_short_circuits(mock_call):
    state = {"task": "Help me with my project"}
    result = orchestrator_node(state)
    
    assert result.get("needs_clarification") is True
    assert "subtasks" in result and result["subtasks"] == []
    assert "final_output" in result
    assert "I'd like to help!" in result["final_output"]
    
    # Ensure LLM was not called
    mock_call.assert_not_called()

@patch("agents.orchestrator.call_agent_structured")
def test_normal_task_not_vague(mock_call):
    # Mock a valid LLM response
    from agents.orchestrator import OrchestratorOutput
    mock_call.return_value = OrchestratorOutput(
        subtasks=["Write code"],
        assignments={"0": "coder"}
    )
    
    state = {"task": "Write a Python fibonacci function"}
    result = orchestrator_node(state)
    
    assert result.get("needs_clarification", False) is False
    # Ensure LLM WAS called
    mock_call.assert_called_once()
