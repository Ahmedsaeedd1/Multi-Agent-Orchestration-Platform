"""
Phase 8 — Orchestrator Agent Node

The orchestrator has NO tools.  Its only job is to decompose the incoming
task into subtasks and assign each to a specialist agent (researcher, coder,
data_analyst).  It uses ``call_agent_structured`` for reliable JSON output.
"""

import logging
import re
from typing import TypedDict

from agents.schemas import OrchestratorOutput
from agents.structured_call import call_agent_structured, StructuredCallError
from router import ModelRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local AgentState definition (mirrors what graph.py will define in Phase 9)
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Minimal state for the orchestrator node.  Extended in graph.py."""
    task: str
    subtasks: list[str]
    assignments: dict[str, str]       # subtask index -> agent name
    research_notes: list[str]
    code: str
    analysis: str
    review_feedback: str
    final_output: str
    run_id: str
    review_cycles: int
    needs_clarification: bool


# ---------------------------------------------------------------------------
# Module-level router instance
# ---------------------------------------------------------------------------

router = ModelRouter()


# ---------------------------------------------------------------------------
# Vagueness Pre-Check
# ---------------------------------------------------------------------------

VAGUE_TASK_PATTERNS = [
    r'^help me with my (project|work|task)s?\.?$',
    r'^(i need|can you) help\.?$',
    r'^do something (useful|helpful)\.?$',
    r'^help\.?$',
    r'^what (can|should) (i|you) do\.?$',
]

def _is_vague_task(task: str) -> bool:
    """
    Detect tasks with no concrete subject before calling any LLM.
    Catches obviously vague inputs deterministically rather than
    relying on model instruction-following.
    """
    normalized = task.strip().lower()
    if len(normalized) < 15:  # very short tasks are almost always vague
        return True
    for pattern in VAGUE_TASK_PATTERNS:
        if re.match(pattern, normalized):
            return True
    return False


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the orchestrator of a multi-agent system. Decompose the "
    "user task into concrete subtasks and assign each to one of these "
    "agents: researcher, coder, data_analyst, sql_assistant.\n\n"
    "Agent selection rules — follow strictly:\n"
    "- researcher: for ANY question asking about facts, comparisons, explanations, "
    "concepts, recommendations, or 'what/why/how' questions. This is the DEFAULT agent.\n"
    "- coder: Assign whenever the task requires writing, debugging, executing code, or creating code examples/endpoints.\n"
    "- data_analyst: ONLY when the task explicitly requires analysing a dataset, "
    "running statistics, or producing charts.\n"
    "- sql_assistant: ONLY when the task explicitly requires querying a SQL database, "
    "retrieving database schemas, or extracting data from a relational database.\n"
    "- Do NOT assign coder or data_analyst to research/explanation tasks — "
    "they will waste time and resources.\n\n"
    "Return ONLY a raw JSON object (no markdown, no code fences) in this exact shape:\n"
    '{"subtasks": ["subtask description 1", ...], '
    '"assignments": {"0": "researcher", "1": "researcher", ...}}\n\n'
    "Rules:\n"
    "- assignments keys are string indices (\"0\", \"1\", ...) matching subtasks positions.\n"
    "- agent values must be one of: researcher, coder, data_analyst, sql_assistant.\n"
    "- For simple fact-based questions ('what is X', 'explain Y'), return a SINGLE, unified subtask.\n"
    "- If the task asks for a comparative claim requiring quantitative data (e.g., price-to-performance, cost comparison, benchmark comparison), you MUST explicitly create a subtask to gather comparable figures and assign it to data_analyst (or researcher).\n"
    "- Use data_analyst for CSV/JSON/API data processing, sql_assistant for database queries.\n"
    "- Do NOT assign researcher to pure coding tasks unless the user explicitly asks for research/explanation.\n"
    "- For most questions, ALL subtasks should go to researcher.\n"
    "- Do NOT include any text outside the JSON object."
)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState) -> dict:
    """
    LangGraph node that decomposes *state["task"]* into subtasks.

    Returns an ``AgentState`` update dict with ``subtasks`` and ``assignments``.
    On failure, returns a safe fallback (single subtask → researcher).
    """
    task = state.get("task", "")
    if not task:
        logger.warning("Orchestrator received empty task — using fallback")
        return _fallback(task)
    
    if _is_vague_task(task):
        logger.info("Task detected as vague — requesting clarification")
        return {
            "subtasks": [],
            "assignments": {},
            "final_output": (
                "I'd like to help! Could you give me more detail? For example:\n"
                "- What is your project about?\n"
                "- Are you looking for research, code, analysis, or a mix?\n"
                "- What's the specific problem you're trying to solve?"
            ),
            "needs_clarification": True,
        }

    user_content = task
    review = state.get("review") or {}
    if review.get("verdict") == "needs_revision" and review.get("feedback"):
        logger.info("Cycle %d: Injecting reviewer feedback into orchestrator prompt", state.get("review_cycle_count", 0))
        user_content += f"\n\nPREVIOUS ATTEMPT FAILED. Reviewer feedback to fix:\n{review['feedback']}"

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        output: OrchestratorOutput = call_agent_structured(
            router=router,
            agent_name="orchestrator",
            messages=messages,
            schema=OrchestratorOutput,
            max_repairs=2,
        )
    except (StructuredCallError, Exception) as e:
        logger.error("Orchestrator failed to decompose task: %s", e)
        return _fallback(task)

    if output.subtasks and output.subtasks[0].startswith("CLARIFICATION_NEEDED"):
        logger.info("Task too vague — requesting clarification")
        return {
            "subtasks": [],
            "assignments": {},
            "final_output": output.subtasks[0].replace("CLARIFICATION_NEEDED:", "").strip(),
            "needs_clarification": True,
        }

    n = len(output.subtasks)
    logger.info("Orchestrator decomposed task into %d subtasks", n)
    logger.debug("Subtasks: %s", output.subtasks)
    logger.debug("Assignments: %s", output.assignments)

    return {
        "subtasks": output.subtasks,
        "assignments": output.assignments,
    }


def _fallback(task: str) -> dict:
    """Return a safe single-subtask fallback."""
    task_lower = task.lower()
    is_code = any(kw in task_lower for kw in ["code", "python", "script", "function", "bug", "implement"])
    return {
        "subtasks": [task],
        "assignments": {"0": "coder" if is_code else "researcher"},
    }