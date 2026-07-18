import pytest
from unittest.mock import patch, MagicMock
from tools.code_eval import run_static_analysis, run_sandboxed_with_edge_cases
from agents.code_evaluator import code_evaluator_node, CodeEvalOutput, EdgeCaseSet

def test_run_static_analysis_detects_issues():
    # Pass code with an obvious lint error (unused import)
    code = "import os\nprint('hello')\n"
    result = run_static_analysis(code)
    assert not result["clean"]
    assert len(result["issues"]) > 0
    # Should catch unused import F401
    assert any(issue["code"] == "F401" for issue in result["issues"])

def test_run_static_analysis_clean_code():
    code = "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n"
    result = run_static_analysis(code)
    assert result["clean"] is True
    assert len(result["issues"]) == 0

def test_run_sandboxed_with_edge_cases_catches_failure():
    # Code that fails on empty input
    code = "import sys\ndata = sys.stdin.read()\nassert len(data) > 0\n"
    edge_cases = [{"input": ""}]
    result = run_sandboxed_with_edge_cases(code, edge_cases)
    assert not result["all_passed"]
    assert len(result["results"]) == 1
    assert not result["results"][0]["passed"]
    assert "AssertionError" in str(result["results"][0]["error"])

def test_run_sandboxed_with_edge_cases_all_pass():
    code = "import sys\ndata = sys.stdin.read().strip()\nprint(data[::-1])\n"
    edge_cases = [{"input": "hello"}, {"input": "world"}]
    result = run_sandboxed_with_edge_cases(code, edge_cases)
    assert result["all_passed"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["passed"]
    assert result["results"][0]["output"].strip() == "olleh"
    assert result["results"][1]["passed"]
    assert result["results"][1]["output"].strip() == "dlrow"

def test_sandboxed_execution_respects_timeout():
    # Infinite loop
    code = "while True: pass\n"
    edge_cases = [{"input": ""}]
    # We monkeypatch the subprocess.run timeout down to 1s for the test so it doesn't take 10s
    with patch("subprocess.run") as mock_run:
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired(cmd="python", timeout=1)
        result = run_sandboxed_with_edge_cases(code, edge_cases)
        assert not result["all_passed"]
        assert "Execution timed out" in result["results"][0]["error"]

@patch("agents.code_evaluator.call_agent_structured")
@patch("agents.code_evaluator.run_static_analysis")
@patch("agents.code_evaluator.run_sandboxed_with_edge_cases")
def test_code_evaluator_node_isolated(mock_run_sandboxed, mock_run_static, mock_call_agent):
    mock_run_static.return_value = {"issues": [], "clean": True}
    mock_run_sandboxed.return_value = {
        "results": [{"input": "a", "output": "a", "error": None, "passed": True}],
        "all_passed": True
    }
    
    # First call generates edge cases, second call generates verdict
    mock_call_agent.side_effect = [
        EdgeCaseSet(edge_cases=[{"description": "basic", "input": "a"}]),
        CodeEvalOutput(
            verdict="pass",
            static_issues_summary="None",
            edge_case_summary="Passed basic",
            overall_summary="Everything passed"
        )
    ]
    
    state = {
        "task": "Echo input",
        "code_output": {"code": "print(input())"}
    }
    
    result = code_evaluator_node(state)
    assert "code_eval_output" in result
    output = result["code_eval_output"]
    assert output["verdict"] == "pass"
    assert "Everything passed" in output["summary"]

@patch("agents.code_evaluator.call_agent_structured")
@patch("agents.code_evaluator.run_static_analysis")
@patch("agents.code_evaluator.run_sandboxed_with_edge_cases")
def test_verdict_fails_on_real_error_not_style_warning(mock_run_sandboxed, mock_run_static, mock_call_agent):
    # Setup: Real edge case failure
    mock_run_static.return_value = {"issues": [], "clean": True}
    mock_run_sandboxed.return_value = {
        "results": [{"input": "", "output": "", "error": "Crash", "passed": False}],
        "all_passed": False
    }
    
    # The LLM hallucinates a "pass" despite the crash
    mock_call_agent.side_effect = [
        EdgeCaseSet(edge_cases=[{"description": "empty", "input": ""}]),
        CodeEvalOutput(
            verdict="pass",
            static_issues_summary="Clean",
            edge_case_summary="Failed but it's ok",
            overall_summary="Pass anyway"
        )
    ]
    
    state = {
        "task": "Crash",
        "code_output": {"code": "raise Exception()"}
    }
    
    result = code_evaluator_node(state)
    output = result["code_eval_output"]
    # The agent should force verdict to "fail" because passed=False in edge case results
    assert output["verdict"] == "fail"
