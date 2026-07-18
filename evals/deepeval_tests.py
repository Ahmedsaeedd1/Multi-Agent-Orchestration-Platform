import sys
import copy
import uuid as _uuid
from unittest.mock import patch, MagicMock

# ── 1. Ragas / DeepEval import workaround for missing vertexai ───────────────
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from evals.router_backed_llm import RouterBackedLLM

import graph as graph_module
from agents.reviewer import reviewer_node

# ── 2. Mock Router Call for GEval & Agents ───────────────────────────────────
def mock_router_call(self, agent_name, messages):
    # Retrieve the prompt string
    prompt = messages[-1]["content"] if isinstance(messages[-1], dict) else str(messages[-1])
    
    # Check if the prompt is for GEval scoring a test case
    if "score" in prompt.lower() and "reason" in prompt.lower():
        return '{"score": 10, "reason": "The output perfectly meets all criteria."}'
    # Check if the prompt is for GEval generating evaluation steps
    elif "steps" in prompt.lower() or "generate" in prompt.lower():
        return '{"steps": ["Step 1: Check task decomposition completeness.", "Step 2: Verify result formatting."]}'
    # If it is the reviewer node's structured call, return an approved verdict
    elif agent_name == "reviewer":
        return '{"review_feedback": "approved", "review_cycles": 1}'
    else:
        return "{}"

# ── 3. Helper to make initial graph state ───────────────────────────────────
def _make_initial_state(**overrides) -> dict:
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

# ── 4. Helper to make reviewer node state ───────────────────────────────────
def _make_reviewer_state(**overrides) -> dict:
    state = {
        "task": "Write a function to calculate the average of a list of numbers",
        "research_notes": [],
        "code": "def avg(lst): return sum(lst)/len(lst)",
        "analysis": "Correct implementation",
        "review_cycles": 0,
    }
    state.update(**overrides)
    return state

# ── 5. Helper to run compiled graph with mocked agent node return values ─────
def run_graph_mocked(task: str) -> str:
    plan_ret = {
        "plan": "Mocked execution plan.",
    }
    orc_ret = {
        "subtasks": ["research competitors", "write summary code", "generate chart code"],
        "assignments": {"0": "researcher", "1": "coder", "2": "coder"},
    }
    res_ret = {
        "research_notes": ["Competitor A pricing: $10/mo", "Competitor B pricing: $15/mo"],
    }
    cod_ret = {
        "code": "prices = {'A': 10, 'B': 15}\n",
    }
    da_ret = {"analysis": "Competitor A is cheaper."}
    rev_ret = {"review_feedback": "approved", "review_cycles": 1}
    agg_ret = {
        "final_output": (
            "Research: Competitor A prices at $10/mo, Competitor B at $15/mo. "
            "Summary: Competitor A is cheaper, but Competitor B offers more features. "
            "Chart: A bar chart comparing the two prices was generated."
        )
    }

    patches = [
        patch.object(graph_module, "planner_node", side_effect=lambda *a, **k: copy.deepcopy(plan_ret)),
        patch.object(graph_module, "orchestrator_node", side_effect=lambda *a, **k: copy.deepcopy(orc_ret)),
        patch.object(graph_module, "researcher_node", side_effect=lambda *a, **k: copy.deepcopy(res_ret)),
        patch.object(graph_module, "coder_node", side_effect=lambda *a, **k: copy.deepcopy(cod_ret)),
        patch.object(graph_module, "data_analyst_node", side_effect=lambda *a, **k: copy.deepcopy(da_ret)),
        patch.object(graph_module, "reviewer_node", side_effect=lambda *a, **k: copy.deepcopy(rev_ret)),
        patch.object(graph_module, "aggregator_node", side_effect=lambda *a, **k: copy.deepcopy(agg_ret)),
        patch.object(graph_module, "_run_with_timeout", side_effect=lambda fn, state, name: fn(state)),
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

# ── 6. GEval Quality Checks ──────────────────────────────────────────────────

@patch("router.ModelRouter.call", new=mock_router_call)
def test_orchestrator_decomposes_task():
    """
    Validate the orchestrator's task decomposition using GEval.
    Checks that the final output addresses all subtasks from the original input.
    """
    task = "Research competitor pricing and write a summary with a chart"
    actual_output = run_graph_mocked(task)

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


@patch("router.ModelRouter.call", new=mock_router_call)
def test_reviewer_approves_quality_output():
    """
    Give the reviewer a high-quality code implementation and verify that
    the reviewer correctly approves the output under GEval criteria.
    """
    # Run the reviewer node with good code state.
    # The router call inside the node is mocked to return 'approved' feedback.
    good_state = _make_reviewer_state(
        code="def calculate_average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0.0\n",
        analysis="Correct implementation with check for empty list.",
        code_verified=True, # to bypass safety overrides
    )
    result = reviewer_node(good_state)
    verdict = result.get("review_feedback", "needs_revision")

    test_case = LLMTestCase(
        input="Review a correct average calculation function with empty list handling.",
        actual_output=f"verdict={verdict}",
    )

    correctness = GEval(
        name="Reviewer approves high-quality work",
        criteria="The reviewer verdict must be 'approved' when code is correct and high-quality.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=RouterBackedLLM(agent_name="reviewer"),
    )

    assert_test(test_case, [correctness])