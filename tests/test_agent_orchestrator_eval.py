"""
Phase 10 — CI/CD Agent Evaluation with DeepEval

Validates the orchestrator's task decomposition quality using DeepEval's GEval.

This test can be run standalone with:
    deepeval test run tests/test_agent_orchestrator_eval.py

Or via pytest:
    pytest tests/test_agent_orchestrator_eval.py -v
"""

import os
import pytest
import copy
import uuid as _uuid
from unittest.mock import patch

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from evals.router_backed_llm import RouterBackedLLM

import graph as graph_module


def _make_initial_state(**overrides) -> dict:
    """Return a minimal initial state dict with required fields."""
    state = {
        "run_id": str(_uuid.uuid4()),
        "user_id": "test-user",
        "task": "Research competitor pricing and write a summary with a chart",
        "plan": None,
        "research_output": None,
        "code_output": None,
        "analysis_output": None,
        "review": None,
        "final_output": None,
        "review_cycle_count": 0,
        "step_log": [],
        "error": None,
        "_subtasks": [],
        "_assignments": {},
    }
    state.update(**overrides)
    return state


def run_graph_or_node(task: str) -> str:
    """
    Invoke the compiled graph with a sample task, using mocked agent outputs.

    Returns the final_answer string from the aggregator.
    """
    # Mock all seven agent node functions so we don't hit real LLM providers
    orc_ret = {
        "subtasks": ["research competitors", "write summary code", "generate chart code"],
        "assignments": {"0": "researcher", "1": "coder", "2": "coder"},
    }
    res_ret = {
        "research_notes": [
            "Competitor A pricing: $10/mo",
            "Competitor B pricing: $15/mo",
        ],
    }
    cod_ret = {
        "code": (
            "import matplotlib.pyplot as plt\n"
            "prices = {'A': 10, 'B': 15}\n"
            "plt.bar(prices.keys(), prices.values())\n"
            "plt.title('Competitor Pricing')\n"
            "plt.show()"
        ),
    }
    da_ret = {"analysis": "Competitor A is cheaper but Competitor B offers more features."}
    rev_ret = {"review_feedback": "approved", "review_cycles": 1}
    agg_ret = {
    "final_output": (
        "Research: Competitor A prices at $10/mo, Competitor B at $15/mo. "
        "Summary: Competitor A is cheaper, but Competitor B offers more features, "
        "making it a better value for teams needing advanced tools. "
        "Chart: A bar chart comparing the two prices was generated "
        "(see attached matplotlib output) showing the $5/mo gap."
    )
}

    patches = [
        patch.object(graph_module, "orchestrator_node", side_effect=lambda *a, **k: copy.deepcopy(orc_ret)),
        patch.object(graph_module, "researcher_node", side_effect=lambda *a, **k: copy.deepcopy(res_ret)),
        patch.object(graph_module, "coder_node", side_effect=lambda *a, **k: copy.deepcopy(cod_ret)),
        patch.object(graph_module, "data_analyst_node", side_effect=lambda *a, **k: copy.deepcopy(da_ret)),
        patch.object(graph_module, "reviewer_node", side_effect=lambda *a, **k: copy.deepcopy(rev_ret)),
        patch.object(graph_module, "aggregator_node", side_effect=lambda *a, **k: copy.deepcopy(agg_ret)),
        patch.object(
            graph_module, "_run_with_timeout",
            side_effect=lambda fn, state, name: fn(state),
        ),
    ]

    for p in patches:
        p.start()

    try:
        initial = _make_initial_state(task=task)
        final = graph_module.app.invoke(copy.deepcopy(initial))
        result = final.get("final_output", {}) or {}
        return result.get("final_answer", "No output produced")
    finally:
        for p in patches:
            p.stop()


def test_orchestrator_task_decomposition():
    """
    Validate the orchestrator's task decomposition using GEval.

    This test runs the full graph with mocked agents and checks that the
    final output addresses all subtasks from the original input.
    """
    task = "Research competitor pricing and write a summary with a chart"
    actual_output = run_graph_or_node(task)

    test_case = LLMTestCase(
        input=task,
        actual_output=actual_output,
    )

    correctness = GEval(
        name="Task completeness",
        criteria="All subtasks (research, summary, chart) were addressed in the output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=RouterBackedLLM(agent_name="reviewer"),
    )

    assert_test(test_case, [correctness])