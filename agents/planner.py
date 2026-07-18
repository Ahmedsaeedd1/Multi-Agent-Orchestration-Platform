"""
Phase 8 — Planner Agent Node
"""

import logging
from agents.orchestrator import AgentState
from agents.schemas import PlannerOutput
from agents.structured_call import call_agent_structured, StructuredCallError
from router import ModelRouter

logger = logging.getLogger(__name__)

router = ModelRouter()

SYSTEM_PROMPT = (
    "You are the planner. Given a list of subtasks, order them for efficient execution.\n\n"
    "Return ONLY a raw JSON object (no markdown, no code fences) in this exact shape:\n"
    '{"tasks": ["step 1 description", "step 2 description", ...]}\n\n'
    "Rules:\n"
    "- tasks is an ordered list of the subtask descriptions, most important first.\n"
    "- Do NOT include any text outside the JSON object."
)

def planner_node(state: AgentState) -> dict:
    subtasks = state.get("subtasks", [])
    assignments = state.get("assignments", {})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Subtasks: {subtasks}\nAssignments: {assignments}"},
    ]

    try:
        output: PlannerOutput = call_agent_structured(
            router=router,
            agent_name="planner",
            messages=messages,
            schema=PlannerOutput,
            max_repairs=2,
        )
    except (StructuredCallError, Exception) as e:
        logger.error("Planner failed to create execution plan: %s", e)
        return {"subtasks": subtasks}

    logger.info("Planner refined task into %d subtasks", len(output.tasks))
    logger.debug("Refined tasks: %s", output.tasks)
    
    return {"subtasks": output.tasks, "plan_reasoning": output.reasoning}
