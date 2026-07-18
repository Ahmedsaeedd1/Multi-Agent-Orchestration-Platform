"""
Phase 10 — Researcher Agent Faithfulness Evaluation

Unlike the coder (where execution is the right checker) or the
orchestrator/reviewer (where GEval judges subjective quality), the
researcher's core risk is hallucination: claiming a "finding" that was
never actually present in anything web_search/web_fetch returned.

GEval is the right tool here because faithfulness-to-context is
exactly the kind of judgment an LLM judge is good at — "does every
claim in this text trace back to this source material" is a natural
language entailment problem, not something deterministically checkable
like arithmetic.

Approach
--------
1. Mock web_search/web_fetch to return KNOWN, fixed content (so we
   control exactly what the researcher was given to work with).
2. Run the real researcher_node — real router call, real tool-call
   parsing, real structured output — so the eval exercises actual
   agent behavior, not a rewritten simulation of it.
3. Feed the researcher's actual findings + the known tool content into
   a GEval faithfulness rubric: every claim in `findings` must be
   traceable to the retrieved content; anything not grounded in it is
   a hallucination and should fail the rubric.

This deliberately mirrors RAGAS's "faithfulness" metric conceptually,
but implemented as a GEval rubric using RouterBackedLLM (free-tier,
no Ollama/OpenAI dependency) rather than pulling in RAGAS's own
faithfulness metric, which expects a slightly different data shape
(retrieved_contexts/reference) than what this test naturally produces.

Run with:
    pytest evals/test_researcher_eval.py -v -m llm
"""

from unittest.mock import patch

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evals.router_backed_llm import RouterBackedLLM
from agents.researcher import researcher_node

_JUDGE = RouterBackedLLM(agent_name="reviewer")


def _make_state(task: str, **overrides) -> dict:
    state = {
        "task": task,
        "run_id": "eval-researcher",
        "assignments": {"0": "researcher"},
    }
    state.update(overrides)
    return state


def _run_researcher_with_fixed_tool_content(task: str, tool_content: str) -> tuple[list[str], str]:
    """
    Run the real researcher_node, but with web_search/web_fetch patched
    to always return *tool_content* regardless of the query — this
    pins down exactly what "ground truth" the researcher had access to,
    so the faithfulness check has a fixed reference to judge against.

    Also patches the Qdrant memory-write helper to a no-op, since this
    eval doesn't need or want a live Qdrant/Ollama dependency just to
    check findings quality.

    Returns (findings, tool_content) for use in the GEval test case.
    """
    with patch("agents.researcher.execute_tool", return_value=tool_content), \
         patch("agents.researcher._write_findings_to_memory", return_value=None):
        result = researcher_node(_make_state(task))

    findings = result.get("research_notes", [])
    return findings, tool_content


@pytest.mark.llm
class TestResearcherFaithfulness:
    """
    Each test pins the researcher's tool results to known content, then
    checks that the researcher's reported findings are faithful to that
    content — no invented facts, no numbers/claims that weren't there.
    """

    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_findings_grounded_in_retrieved_content(self):
        """
        Give the researcher a single, narrow fact to find. Its findings
        must reflect that fact and must not introduce claims (e.g.
        different numbers, unrelated details) that weren't present in
        what it actually retrieved.
        """
        tool_content = (
            "[Tool result: web_search]\n"
            "Source: official product page.\n"
            "The Zenith X200 laptop has a battery life of 14 hours and "
            "weighs 1.2 kg. It was released in March 2025 and costs $899."
        )

        findings, ground_truth = _run_researcher_with_fixed_tool_content(
            task="Research the Zenith X200 laptop's battery life and price.",
            tool_content=tool_content,
        )

        assert findings, "Researcher returned no findings at all"

        findings_text = "\n".join(f"- {f}" for f in findings)

        test_case = LLMTestCase(
            input=(
                f"Retrieved source content:\n{ground_truth}\n\n"
                f"Researcher's reported findings:\n{findings_text}"
            ),
            actual_output=findings_text,
        )

        faithfulness = GEval(
            name="Researcher faithfulness",
            criteria=(
                "Every factual claim in the researcher's reported findings "
                "must be directly supported by the retrieved source content "
                "provided in the input. If the findings state any number, "
                "date, or fact that contradicts or is not present in the "
                "retrieved source content, this is a hallucination and the "
                "output fails this criterion. Paraphrasing the same facts "
                "is fine; inventing new ones is not."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=_JUDGE,
        )

        assert_test(test_case, [faithfulness])

    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_findings_do_not_invent_unavailable_data(self):
        """
        Give the researcher deliberately sparse/incomplete tool content
        (missing a detail the task asks about). A faithful researcher
        should either omit that detail or flag it as unavailable —
        NOT invent a plausible-sounding number for it.
        """
        tool_content = (
            "[Tool result: web_search]\n"
            "Source: news article.\n"
            "The startup Nimbus Analytics announced a new product line "
            "today. The announcement did not include pricing details."
        )

        findings, ground_truth = _run_researcher_with_fixed_tool_content(
            task="Research Nimbus Analytics' new product pricing.",
            tool_content=tool_content,
        )

        assert findings, "Researcher returned no findings at all"

        findings_text = "\n".join(f"- {f}" for f in findings)

        test_case = LLMTestCase(
            input=(
                f"Retrieved source content:\n{ground_truth}\n\n"
                f"Researcher's reported findings:\n{findings_text}"
            ),
            actual_output=findings_text,
        )

        no_fabrication = GEval(
            name="Researcher does not fabricate missing data",
            criteria=(
                "The retrieved source content explicitly does NOT include "
                "pricing information. The researcher's findings must not "
                "state a specific price or invented pricing detail for "
                "the product. Findings that correctly note pricing is "
                "unavailable, or that omit pricing entirely, satisfy this "
                "criterion. Findings that state any specific price fail it."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=_JUDGE,
        )

        assert_test(test_case, [no_fabrication])