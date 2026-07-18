"""
Phase 9 — LangGraph Assembly Tests

Tests validate graph topology, control flow, cycle caps, fan-out selectivity,
node timeouts, and step_log population — all with mocked agent functions.

Updated for planner insertion: orchestrator -> planner -> fan-out.
Every test that asserts on step_log contents, sequence, or length now
accounts for the mandatory planner step between orchestrator and the
specialist fan-out. planner_node is mocked the same way every other
node is — these are topology/control-flow tests, not planner quality
tests (see evals/ for LLM-quality checks).
"""

import copy
import time
import uuid as _uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, ANY

import pytest

import graph as graph_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_state(**overrides) -> dict:
    """Return a minimal initial state dict with required fields."""
    state = {
        "run_id": str(_uuid.uuid4()),
        "user_id": "test-user",
        "task": "Write a Python script to calculate fibonacci numbers",
        "plan": None,
        "plan_reasoning": "",
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


def _default_planner_return(subtasks_in: list) -> dict:
    """
    Default mocked planner_node return: pass subtasks through unchanged
    (same count, so the task-count mismatch guard in planner_node never
    triggers), with a plausible reasoning string.
    """
    return {
        "subtasks": list(subtasks_in) if subtasks_in else ["research fibonacci"],
        "plan_reasoning": "Single research step, no dependencies to order.",
    }


def _run_with_patched_nodes(
    initial_state: dict,
    orchestrator_return: dict | None = None,
    planner_return: dict | None = None,
    researcher_return: dict | None = None,
    coder_return: dict | None = None,
    data_analyst_return: dict | None = None,
    reviewer_return: dict | None = None,
    aggregator_return: dict | None = None,
    timeout_return: callable = None,
):
    """
    Patch all node functions in graph.py (and optionally _run_with_timeout)
    then invoke the compiled graph.  Returns (final_state, mocks_dict).
    """
    orc_ret = orchestrator_return or {
        "subtasks": ["research fibonacci"],
        "assignments": {"0": "researcher"},
    }
    plan_ret = planner_return or _default_planner_return(orc_ret.get("subtasks", []))
    res_ret = researcher_return or {"research_notes": ["Fibonacci is Fn = Fn-1 + Fn-2"]}
    cod_ret = coder_return or {
        "code": "def fib(n): ...",
        "code_verified": True,
        "code_exec_output": "(mocked — not actually executed)",
    }
    da_ret = data_analyst_return or {"analysis": "O(2^n) naive, O(n) iterative"}
    rev_ret = reviewer_return or {
        "review_feedback": "approved",
        "review_feedback_text": "Looks correct.",
        "review_cycles": 1,
    }
    agg_ret = aggregator_return or {"final_output": "# Fibonacci\n\ndef fib(n): ..."}

    if timeout_return is None:
        timeout_return = lambda fn, state, name: fn(state)

    patches = [
        patch.object(graph_module, "orchestrator_node", return_value=orc_ret),
        patch.object(graph_module, "planner_node", return_value=plan_ret),
        patch.object(graph_module, "researcher_node", return_value=res_ret),
        patch.object(graph_module, "coder_node", return_value=cod_ret),
        patch.object(graph_module, "data_analyst_node", return_value=da_ret),
        patch.object(graph_module, "reviewer_node", return_value=rev_ret),
        patch.object(graph_module, "aggregator_node", return_value=agg_ret),
        patch.object(graph_module, "_run_with_timeout", side_effect=timeout_return),
    ]

    mocks = {}
    mock_names = [
        "orchestrator_node", "planner_node", "researcher_node", "coder_node",
        "data_analyst_node", "reviewer_node", "aggregator_node",
        "_run_with_timeout",
    ]
    for p, name in zip(patches, mock_names):
        mocks[name] = p.start()

    try:
        final_state = graph_module.app.invoke(copy.deepcopy(initial_state))
    finally:
        for p in patches:
            p.stop()

    return final_state, mocks


# ===================================================================
# Tests
# ===================================================================


class TestGraphHappyPath:
    """Full happy path — reviewer approves on first pass."""

    def test_ends_at_aggregator(self):
        """After the happy path, final_output must be populated."""
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        assert final.get("final_output") is not None, (
            f"Expected final_output to be set, got: {final.get('final_output')}"
        )
        answer = final["final_output"]
        assert isinstance(answer, dict)
        assert "final_answer" in answer

    def test_approves_on_first_cycle(self):
        """review_cycle_count should be 1 after first-pass approval."""
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        assert final["review_cycle_count"] == 1, (
            f"Expected 1 review cycle, got {final['review_cycle_count']}"
        )

    def test_step_log_has_all_nodes(self):
        """
        All 5 invoked nodes appear: orchestrator, planner, researcher,
        reviewer, aggregator. Planner is now mandatory between
        orchestrator and the specialist fan-out.
        """
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        step_log = final.get("step_log", [])
        node_names = [e["node"] for e in step_log]

        expected = ["orchestrator", "planner", "researcher", "reviewer", "aggregator"]
        for name in expected:
            assert name in node_names, (
                f"step_log missing '{name}'. Got: {node_names}"
            )
        assert len(step_log) == 5, (
            f"Expected exactly 5 step_log entries (with planner), got {len(step_log)}: {node_names}"
        )

    def test_planner_runs_after_orchestrator_before_specialists(self):
        """
        Explicit ordering check: planner must run strictly after
        orchestrator and strictly before any specialist node.
        """
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        node_sequence = [e["node"] for e in final.get("step_log", [])]
        orch_idx = node_sequence.index("orchestrator")
        planner_idx = node_sequence.index("planner")
        researcher_idx = node_sequence.index("researcher")

        assert orch_idx < planner_idx < researcher_idx, (
            f"Expected orchestrator < planner < researcher, got sequence: {node_sequence}"
        )

    def test_plan_reasoning_populated(self):
        """planner's reasoning must be forwarded into graph state."""
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        assert final.get("plan_reasoning"), (
            f"Expected non-empty plan_reasoning, got: {final.get('plan_reasoning')!r}"
        )


class TestPlannerFailureHandling:
    """Planner failing must be non-fatal — orchestrator's plan survives."""

    def test_planner_error_falls_back_to_orchestrator_subtasks(self):
        """
        If planner_node's wrapped call returns an error (e.g. timeout),
        the original orchestrator subtasks must still be used for
        fan-out — the run should not crash or lose the plan entirely.
        """
        orc_ret = {
            "subtasks": ["research fibonacci"],
            "assignments": {"0": "researcher"},
        }

        def _timeout_with_planner_failing(fn, state, name):
            if name == "planner":
                return {"error": "Node 'planner' timed out after 30s"}
            return fn(state)

        final, mocks = _run_with_patched_nodes(
            _make_initial_state(),
            orchestrator_return=orc_ret,
            timeout_return=_timeout_with_planner_failing,
        )

        # Fan-out must still have happened based on orchestrator's
        # original _assignments, even though planner errored.
        assert mocks["researcher_node"].call_count == 1, (
            "Expected researcher to still be invoked despite planner failure"
        )
        assert final.get("final_output") is not None, (
            "Run should still complete even if planner fails"
        )


class TestCycleCapEnforcement:
    """Graph must not loop forever when reviewer always returns needs_revision."""

    def test_cycle_cap_and_call_counts(self):
        """
        Reviewer always returns needs_revision; graph still reaches aggregator/END
        and the reviewer mock is called exactly MAX_REVIEW_CYCLES times.
        Each cycle now includes a planner pass (orchestrator -> planner ->
        specialists -> reviewer), so planner is also called MAX_REVIEW_CYCLES times.
        """
        max_cycles = graph_module.MAX_REVIEW_CYCLES  # default 3

        reviewer_mock = MagicMock()
        call_count = [0]

        def _reviewer_side_effect(state):
            cycles = state.get("review_cycles", call_count[0])
            call_count[0] += 1
            return {
                "review_feedback": "needs_revision",
                "review_feedback_text": "Not good enough yet.",
                "review_cycles": cycles + 1,
            }

        reviewer_mock.side_effect = _reviewer_side_effect

        orc_ret = {
            "subtasks": ["task1"],
            "assignments": {"0": "researcher", "1": "coder"},
        }
        planner_mock = MagicMock(
            side_effect=lambda state: _default_planner_return(state.get("subtasks", ["task1"]))
        )
        res_ret = {"research_notes": ["finding"]}
        cod_ret = {"code": "# code", "code_verified": True, "code_exec_output": "(mocked)"}
        da_ret = {"analysis": "analysis"}
        agg_ret = {"final_output": "Final answer after max cycles"}

        patches = [
            patch.object(graph_module, "orchestrator_node", return_value=orc_ret),
            patch.object(graph_module, "planner_node", planner_mock),
            patch.object(graph_module, "researcher_node", return_value=res_ret),
            patch.object(graph_module, "coder_node", return_value=cod_ret),
            patch.object(graph_module, "data_analyst_node", return_value=da_ret),
            patch.object(graph_module, "reviewer_node", reviewer_mock),
            patch.object(graph_module, "aggregator_node", return_value=agg_ret),
            patch.object(graph_module, "_run_with_timeout",
                         side_effect=lambda fn, state, name: fn(state)),
        ]

        mocks = {}
        mock_names = [
            "orchestrator_node", "planner_node", "researcher_node", "coder_node",
            "data_analyst_node", "reviewer_node", "aggregator_node",
            "_run_with_timeout",
        ]
        for p, name in zip(patches, mock_names):
            mocks[name] = p.start()

        try:
            initial = _make_initial_state()
            final = graph_module.app.invoke(copy.deepcopy(initial))
        finally:
            for p in patches:
                p.stop()

        assert call_count[0] == max_cycles, (
            f"Expected reviewer to be called {max_cycles} times, got {call_count[0]}"
        )
        assert reviewer_mock.call_count == max_cycles

        # Planner runs once per cycle too, since orchestrator -> planner
        # is a mandatory plain edge re-entered on every needs_revision loop.
        assert planner_mock.call_count == max_cycles, (
            f"Expected planner to be called {max_cycles} times (once per "
            f"cycle), got {planner_mock.call_count}"
        )

        assert final.get("final_output") is not None, (
            "Graph should reach aggregator even after max cycles"
        )
        assert final["final_output"]["final_answer"] == "Final answer after max cycles"
        assert final["review_cycle_count"] == max_cycles

        # Step count: rather than hand-deriving a formula for how many
        # times each node ran (easy to get subtly wrong when reasoning
        # about exactly which cycle triggers the cycle-cap forced exit),
        # derive the expected total directly from the mocks' own
        # call_count — the mocks are the ground truth for how many times
        # each node actually ran, so step_log's length must match their sum.
        step_log = final.get("step_log", [])
        node_sequence = [e["node"] for e in step_log]

        expected_total = (
            mocks["orchestrator_node"].call_count
            + planner_mock.call_count
            + mocks["researcher_node"].call_count
            + mocks["coder_node"].call_count
            + reviewer_mock.call_count
            + mocks["aggregator_node"].call_count
        )
        assert len(step_log) == expected_total, (
            f"step_log length ({len(step_log)}) must equal the sum of all "
            f"node call_counts ({expected_total}). Sequence was: {node_sequence}"
        )

        # Structural invariants that must hold regardless of the exact
        # cycle-cap arithmetic:
        assert node_sequence[0] == "orchestrator", "Must start with orchestrator"
        assert node_sequence[1] == "planner", "planner must immediately follow orchestrator"
        assert node_sequence[-1] == "aggregator", "Must end with aggregator"
        assert node_sequence.count("aggregator") == 1, "aggregator must run exactly once"

        # Every orchestrator run must be immediately followed by a
        # planner run — this is a plain mandatory edge, never skipped.
        orch_positions = [i for i, n in enumerate(node_sequence) if n == "orchestrator"]
        for pos in orch_positions:
            assert node_sequence[pos + 1] == "planner", (
                f"orchestrator at position {pos} not immediately followed by "
                f"planner. Sequence: {node_sequence}"
            )


class TestFanOutSelectivity:
    """Only the specialist agents requested by the orchestrator are invoked."""

    def test_only_researcher_invoked(self):
        """When assignments only mention researcher, coder+data_analyst are not called."""
        orc_ret = {
            "subtasks": ["search topic"],
            "assignments": {"0": "researcher"},
        }
        res_ret = {"research_notes": ["found info"]}
        agg_ret = {"final_output": "Output with research only"}

        patches = [
            patch.object(graph_module, "orchestrator_node", return_value=orc_ret),
            patch.object(graph_module, "planner_node",
                         side_effect=lambda state: _default_planner_return(state.get("subtasks", []))),
            patch.object(graph_module, "researcher_node", return_value=res_ret),
            patch.object(graph_module, "coder_node", return_value={"code": "# should not be called"}),
            patch.object(graph_module, "data_analyst_node",
                         return_value={"analysis": "should not be called"}),
            patch.object(graph_module, "reviewer_node",
                         return_value={"review_feedback": "approved", "review_feedback_text": "ok", "review_cycles": 1}),
            patch.object(graph_module, "aggregator_node", return_value=agg_ret),
            patch.object(graph_module, "_run_with_timeout",
                         side_effect=lambda fn, state, name: fn(state)),
        ]

        mock_names = [
            "orchestrator_node", "planner_node", "researcher_node", "coder_node",
            "data_analyst_node", "reviewer_node", "aggregator_node",
            "_run_with_timeout",
        ]
        mocks = {}
        for p, name in zip(patches, mock_names):
            mocks[name] = p.start()

        try:
            initial = _make_initial_state()
            final = graph_module.app.invoke(copy.deepcopy(initial))
        finally:
            for p in patches:
                p.stop()

        assert mocks["researcher_node"].call_count == 1, (
            f"Expected researcher call_count=1, got {mocks['researcher_node'].call_count}"
        )
        assert mocks["coder_node"].call_count == 0, (
            f"Expected coder call_count=0, got {mocks['coder_node'].call_count}"
        )
        assert mocks["data_analyst_node"].call_count == 0, (
            f"Expected data_analyst call_count=0, got {mocks['data_analyst_node'].call_count}"
        )
        assert final.get("final_output") is not None

    def test_all_three_specialists_invoked(self):
        """When assignments mention all three, all are called."""
        orc_ret = {
            "subtasks": ["research", "code", "analyze"],
            "assignments": {"0": "researcher", "1": "coder", "2": "data_analyst"},
        }
        res_ret = {"research_notes": ["info"]}
        cod_ret = {"code": "# code", "code_verified": True, "code_exec_output": "(mocked)"}
        da_ret = {"analysis": "analysis"}
        agg_ret = {"final_output": "Full output"}

        patches = [
            patch.object(graph_module, "orchestrator_node", return_value=orc_ret),
            patch.object(graph_module, "planner_node",
                         side_effect=lambda state: _default_planner_return(state.get("subtasks", []))),
            patch.object(graph_module, "researcher_node", return_value=res_ret),
            patch.object(graph_module, "coder_node", return_value=cod_ret),
            patch.object(graph_module, "data_analyst_node", return_value=da_ret),
            patch.object(graph_module, "reviewer_node",
                         return_value={"review_feedback": "approved", "review_feedback_text": "ok", "review_cycles": 1}),
            patch.object(graph_module, "aggregator_node", return_value=agg_ret),
            patch.object(graph_module, "_run_with_timeout",
                         side_effect=lambda fn, state, name: fn(state)),
        ]

        mock_names = [
            "orchestrator_node", "planner_node", "researcher_node", "coder_node",
            "data_analyst_node", "reviewer_node", "aggregator_node",
            "_run_with_timeout",
        ]
        mocks = {}
        for p, name in zip(patches, mock_names):
            mocks[name] = p.start()

        try:
            initial = _make_initial_state()
            final = graph_module.app.invoke(copy.deepcopy(initial))
        finally:
            for p in patches:
                p.stop()

        assert mocks["researcher_node"].call_count == 1
        assert mocks["coder_node"].call_count == 1
        assert mocks["data_analyst_node"].call_count == 1
        assert final.get("final_output") is not None


class TestNodeTimeout:
    """When a node times out, the graph must not hang — error is set and routes to aggregator."""

    def test_coder_timeout_sets_error_and_routes_to_aggregator(self):
        """If coder sleeps past its timeout, state['error'] is set and graph reaches END."""
        orc_ret = {
            "subtasks": ["research", "code"],
            "assignments": {"0": "researcher", "1": "coder"},
        }
        res_ret = {"research_notes": ["info"]}
        cod_ret = {"code": "# should never return"}
        agg_ret = {"final_output": "Partial output after timeout"}

        patches = [
            patch.object(graph_module, "orchestrator_node", return_value=orc_ret),
            patch.object(graph_module, "planner_node",
                         side_effect=lambda state: _default_planner_return(state.get("subtasks", []))),
            patch.object(graph_module, "researcher_node", return_value=res_ret),
            patch.object(
                graph_module, "coder_node",
                side_effect=lambda state: time.sleep(30) or cod_ret,
            ),
            patch.object(graph_module, "data_analyst_node",
                         return_value={"analysis": "should not be called"}),
            patch.object(graph_module, "reviewer_node",
                         return_value={"review_feedback": "approved", "review_feedback_text": "ok", "review_cycles": 1}),
            patch.object(graph_module, "aggregator_node", return_value=agg_ret),
            patch.object(
                graph_module, "NODE_TIMEOUTS",
                {"orchestrator": 60, "planner": 60, "researcher": 60, "coder": 1,
                 "data_analyst": 60, "reviewer": 60, "aggregator": 60},
            ),
        ]

        mock_names = [
            "orchestrator_node", "planner_node", "researcher_node", "coder_node",
            "data_analyst_node", "reviewer_node", "aggregator_node",
            "NODE_TIMEOUTS",
        ]
        mocks = {}
        for p, name in zip(patches, mock_names):
            mocks[name] = p.start()

        try:
            initial = _make_initial_state()
            final = graph_module.app.invoke(copy.deepcopy(initial))
        finally:
            for p in patches:
                p.stop()

        assert final.get("error") is not None, (
            f"Expected 'error' to be set after coder timeout, got: {final.get('error')}"
        )
        assert final.get("final_output") is not None, (
            "Graph should still reach aggregator after a node timeout"
        )


class TestStepLogPopulation:
    """Every node invocation appends exactly one entry to step_log."""

    def test_step_log_entries_have_required_fields(self):
        """Each step_log entry has step_id (uuid), node, and timestamp (iso8601)."""
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        step_log = final.get("step_log", [])
        assert len(step_log) > 0, "step_log is empty"

        for entry in step_log:
            assert "step_id" in entry, f"Missing 'step_id' in {entry}"
            assert "node" in entry, f"Missing 'node' in {entry}"
            assert "timestamp" in entry, f"Missing 'timestamp' in {entry}"

            _uuid.UUID(entry["step_id"])
            datetime.fromisoformat(entry["timestamp"])

    def test_step_log_correct_order(self):
        """Entries in step_log appear in the order nodes were invoked."""
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        step_log = final.get("step_log", [])
        node_sequence = [e["node"] for e in step_log]

        # Happy path now includes planner: orchestrator -> planner ->
        # researcher -> reviewer -> aggregator
        expected_subsequence = ["orchestrator", "planner", "researcher", "reviewer", "aggregator"]

        seq_iter = iter(node_sequence)
        for expected_node in expected_subsequence:
            found = False
            for actual_node in seq_iter:
                if actual_node == expected_node:
                    found = True
                    break
            assert found, (
                f"Expected node '{expected_node}' not found in order after previous nodes. "
                f"Full sequence: {node_sequence}"
            )

    def test_every_node_adds_one_entry(self):
        """Each node called during a run adds exactly one step_log entry."""
        initial = _make_initial_state()
        final, _ = _run_with_patched_nodes(initial)

        step_log = final.get("step_log", [])
        node_counts: dict[str, int] = {}
        for entry in step_log:
            node_counts[entry["node"]] = node_counts.get(entry["node"], 0) + 1

        assert node_counts.get("orchestrator", 0) == 1
        assert node_counts.get("planner", 0) == 1
        assert node_counts.get("researcher", 0) == 1
        assert node_counts.get("reviewer", 0) == 1
        assert node_counts.get("aggregator", 0) == 1


class TestGraphEndToEnd:
    """End-to-end validation of graph.execution."""

    def test_equivalent_to_invoke(self):
        """Using build_graph + compile gives same result as module-level app."""
        initial = _make_initial_state()

        final1, _ = _run_with_patched_nodes(initial)

        orc_ret = {
            "subtasks": ["research fibonacci"],
            "assignments": {"0": "researcher"},
        }
        res_ret = {"research_notes": ["Fibonacci is Fn = Fn-1 + Fn-2"]}
        agg_ret = {"final_output": "# Fibonacci\n\ndef fib(n): ..."}

        patches = [
            patch.object(graph_module, "orchestrator_node", return_value=orc_ret),
            patch.object(graph_module, "planner_node",
                         side_effect=lambda state: _default_planner_return(state.get("subtasks", []))),
            patch.object(graph_module, "researcher_node", return_value=res_ret),
            patch.object(graph_module, "coder_node", return_value={"code": "", "code_verified": True, "code_exec_output": ""}),
            patch.object(graph_module, "data_analyst_node", return_value={"analysis": ""}),
            patch.object(graph_module, "reviewer_node",
                         return_value={"review_feedback": "approved", "review_feedback_text": "ok", "review_cycles": 1}),
            patch.object(graph_module, "aggregator_node", return_value=agg_ret),
            patch.object(graph_module, "_run_with_timeout",
                         side_effect=lambda fn, state, name: fn(state)),
        ]
        for p in patches:
            p.start()
        try:
            builder = graph_module.build_graph()
            fresh_app = builder.compile()
            final2 = fresh_app.invoke(copy.deepcopy(initial))
        finally:
            for p in patches:
                p.stop()

        assert final1.get("final_output") == final2.get("final_output"), (
            "Module-level app and fresh build produce different results"
        )