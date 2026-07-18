"""
Phase 8 — Reviewer Agent Node
"""

import logging

from agents.orchestrator import AgentState
from agents.schemas import ReviewerOutput
from agents.structured_call import call_agent_structured, StructuredCallError
from router import ModelRouter

logger = logging.getLogger(__name__)
router = ModelRouter()

SYSTEM_PROMPT = (
    "You are the reviewer. Review the outputs below for quality and completeness. "
    "Return approved or needs_revision with specific feedback. "
    "Return valid JSON only.\n\n"
    "IMPORTANT: If code is present, it includes a self-verification result "
    "showing whether it was actually executed and whether that execution "
    "succeeded. If code_verified is False, you MUST return needs_revision "
    "regardless of how correct the code looks — code that fails to execute "
    "is not acceptable, no matter how clean it reads."
)


def reviewer_node(state: AgentState) -> dict:
    research_notes = state.get("research_notes", [])
    code = state.get("code", "")
    analysis = state.get("analysis", "")
    fact_check_output = state.get("fact_check_output")
    review_cycles = state.get("review_cycles", 0)

    # Coder's forced self-verification result (see agents/coder.py).
    # Defaults are conservative: if code is present but this key is
    # somehow missing, treat it as unverified rather than assuming pass.
    code_verified = state.get("code_verified", False if code else True)
    code_exec_output = state.get("code_exec_output", "")

    code_verification_block = ""
    if code:
        code_verification_block = (
            f"\n\nCode self-verification: "
            f"{'PASSED — code executed without error' if code_verified else 'FAILED — code did not execute cleanly'}\n"
            f"Execution output:\n{code_exec_output}"
        )

    fact_check_block = ""
    if fact_check_output:
        fact_check_block = (
            f"\n\nFact-Check Report:\n"
            f"Summary: {fact_check_output.get('confidence_summary', '')}\n"
            f"Contradictions Found: {fact_check_output.get('contradictions', [])}\n"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research: {chr(10).join(research_notes)}\n"
                f"{fact_check_block}\n"
                f"Code:\n{code}"
                f"{code_verification_block}\n\n"
                f"Analysis:\n{analysis}"
            ),
        },
    ]

    try:
        output: ReviewerOutput = call_agent_structured(
            router=router,
            agent_name="reviewer",
            messages=messages,
            schema=ReviewerOutput,
            max_repairs=2,
        )

        # Defense-in-depth: enforce the hard rule in code too, don't rely
        # solely on the LLM following the system prompt's instruction.
        verdict = output.verdict
        if code and not code_verified and verdict == "approved":
            logger.warning(
                "Reviewer approved unverified code — overriding to "
                "needs_revision (code_verified=False takes precedence)"
            )
            verdict = "needs_revision"

        return {
            "review_feedback": verdict,
            "review_feedback_text": output.feedback,
            "review_cycles": review_cycles + 1,
        }
    except Exception as e:
        logger.error("Reviewer failed: %s", e)
        # Fail CLOSED: mark as needing revision, don't silently approve.
        # Also surface as a graph-level error so the review-cycle-cap and
        # error-routing logic in graph.py can short-circuit to aggregator
        # instead of masquerading as a normal approved review.
        return {
            "review_feedback": "needs_revision",
            "review_cycles": review_cycles + 1,
            "error": f"Reviewer failed: {e}",
        }