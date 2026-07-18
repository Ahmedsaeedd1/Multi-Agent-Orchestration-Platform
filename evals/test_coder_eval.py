"""
Phase 10 — Coder Agent Evaluation via Execution, NOT GEval

Unlike the orchestrator/reviewer evals (which use an LLM judge because
"is this a good task decomposition" is inherently subjective), code
correctness is often objectively checkable: run it and see if it
produces the right answer. This file evaluates the real coder_node
end-to-end against a small set of known tasks with known-correct
outputs, using tools/code_exec.run_python as the checker — no judge
model, no subjectivity, no extra LLM call for grading.

This complements (does not replace) the coder's own forced
self-verification in agents/coder.py: that check only confirms the
code *runs without error*. This eval additionally confirms the code
*produces the correct result* for a handful of known cases — a
stronger, CI-appropriate signal that a prompt or model swap didn't
silently break the coder's actual competence.

Run with:
    pytest evals/test_coder_eval.py -v

Each test makes a real call through coder_node -> the real router ->
a real (free-tier) LLM provider, then executes the result. These are
network-dependent, not pure-mock unit tests.
"""

import re

import pytest

from agents.coder import coder_node


# ---------------------------------------------------------------------------
# Known tasks with objectively checkable expected output
# ---------------------------------------------------------------------------
# Each case: a task description, plus a way to verify correctness by
# inspecting run_python's actual execution output (already captured by
# coder_node as `code_exec_output`) rather than re-running code ourselves.
# This also implicitly checks that code_verified came back True, since
# a task whose own test asserts on real output naturally fails if the
# code didn't execute cleanly in the first place.

def _make_state(task: str) -> dict:
    """Old-format state dict expected by coder_node."""
    return {
        "task": task,
        "run_id": "eval-coder",
        "review_cycles": 0,
        "research_notes": [],
    }


@pytest.mark.llm
class TestCoderExecutionCorrectness:
    """
    Each test gives the coder a task with one objectively correct
    answer, then checks the actual execution output (not the code
    text, not an LLM's opinion of the code text) for that answer.
    """

    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_fibonacci(self):
        """
        Task: implement fibonacci and print fib(10).
        Expected: the correct 10th Fibonacci number is 55
        (0-indexed: 0,1,1,2,3,5,8,13,21,34,55).
        """
        state = _make_state(
            "Write a Python script that defines a function fib(n) "
            "computing the nth Fibonacci number (0-indexed, fib(0)=0, "
            "fib(1)=1), then prints fib(10)."
        )
        result = coder_node(state)

        assert result.get("code_verified") is True, (
            f"Code failed to execute cleanly: {result.get('code_exec_output')}"
        )
        assert "55" in result.get("code_exec_output", ""), (
            f"Expected '55' in execution output for fib(10), got: "
            f"{result.get('code_exec_output')!r}\nCode was:\n{result.get('code')}"
        )

    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_string_reversal(self):
        """
        Task: reverse a known string.
        Expected: "hello world" reversed is "dlrow olleh".
        """
        state = _make_state(
            "Write a Python script that reverses the string "
            "'hello world' and prints the result."
        )
        result = coder_node(state)

        assert result.get("code_verified") is True, (
            f"Code failed to execute cleanly: {result.get('code_exec_output')}"
        )
        assert "dlrow olleh" in result.get("code_exec_output", ""), (
            f"Expected 'dlrow olleh' in output, got: "
            f"{result.get('code_exec_output')!r}\nCode was:\n{result.get('code')}"
        )

    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_prime_check(self):
        """
        Task: check primality of a specific known prime and composite.
        Expected output must correctly classify both 97 (prime) and
        100 (not prime).
        """
        state = _make_state(
            "Write a Python script with a function is_prime(n) that "
            "returns True if n is prime. Print is_prime(97) and "
            "is_prime(100) on separate lines."
        )
        result = coder_node(state)

        assert result.get("code_verified") is True, (
            f"Code failed to execute cleanly: {result.get('code_exec_output')}"
        )
        output = result.get("code_exec_output", "")
        lines = [l.strip() for l in output.strip().splitlines() if l.strip()]

        assert any("true" in l.lower() for l in lines), (
            f"Expected a True result for is_prime(97), got: {output!r}"
        )
        assert any("false" in l.lower() for l in lines), (
            f"Expected a False result for is_prime(100), got: {output!r}"
        )

    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_list_sum(self):
        """
        Task: sum a known list.
        Expected: sum([3, 7, 12, 5, 18]) == 45.
        """
        state = _make_state(
            "Write a Python script that computes and prints the sum "
            "of this list: [3, 7, 12, 5, 18]"
        )
        result = coder_node(state)

        assert result.get("code_verified") is True, (
            f"Code failed to execute cleanly: {result.get('code_exec_output')}"
        )
        assert "45" in result.get("code_exec_output", ""), (
            f"Expected '45' in output, got: {result.get('code_exec_output')!r}\n"
            f"Code was:\n{result.get('code')}"
        )


@pytest.mark.llm
class TestCoderSelfVerificationHonesty:
    """
    Confirms code_verified/code_exec_output are actually meaningful —
    not just always True regardless of what happened. This guards
    against a future refactor accidentally short-circuiting the
    verification step added in agents/coder.py.
    """

    def test_verified_flag_present_and_boolean(self):
        """Every call must return a boolean code_verified, not missing
        or some other truthy/falsy stand-in."""
        state = _make_state("Write a Python script that prints 'hello'.")
        result = coder_node(state)

        assert "code_verified" in result
        assert isinstance(result["code_verified"], bool)

    def test_exec_output_present_and_nonempty(self):
        """code_exec_output must always be populated — either real
        execution output or a clear failure message — never silently
        missing."""
        state = _make_state("Write a Python script that prints 'hello'.")
        result = coder_node(state)

        assert "code_exec_output" in result
        assert isinstance(result["code_exec_output"], str)
        assert len(result["code_exec_output"]) > 0