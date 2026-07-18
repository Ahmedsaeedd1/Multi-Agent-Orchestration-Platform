import os
import time
import uuid
import logging
import concurrent.futures
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from agents.orchestrator import orchestrator_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.coder import coder_node
from agents.data_analyst import data_analyst_node
from agents.sql_assistant import sql_assistant_node
from agents.reviewer import reviewer_node
from agents.aggregator import aggregator_node

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_REVIEW_CYCLES = int(os.getenv("MAX_REVIEW_CYCLES", "3"))
_DEFAULT_NODE_TIMEOUT = int(os.getenv("MAX_AGENT_TIMEOUT", "60"))

NODE_TIMEOUTS: dict[str, int] = {
    "orchestrator": int(os.getenv("ORCHESTRATOR_TIMEOUT", "30")),
    "planner":      int(os.getenv("PLANNER_TIMEOUT",      "30")),
    "researcher":   int(os.getenv("RESEARCHER_TIMEOUT",   str(_DEFAULT_NODE_TIMEOUT))),
    "coder":        int(os.getenv("CODER_TIMEOUT",        str(_DEFAULT_NODE_TIMEOUT))),
    "data_analyst": int(os.getenv("DATA_ANALYST_TIMEOUT", str(_DEFAULT_NODE_TIMEOUT))),
    "sql_assistant": int(os.getenv("SQL_ASSISTANT_TIMEOUT", str(_DEFAULT_NODE_TIMEOUT))),
    # reviewer and aggregator need generous limits: max_repairs=3 plus
    # Groq rate-limit back-offs can each take ~30 s, so 30 s total is
    # virtually guaranteed to expire before the LLM finishes.
    "reviewer":     int(os.getenv("REVIEWER_TIMEOUT",     "60")),
    "aggregator":   int(os.getenv("AGGREGATOR_TIMEOUT",   str(_DEFAULT_NODE_TIMEOUT))),
}

# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------

def append_step_log(left: list, right: list) -> list:
    """Reducer: append step-log entries (never replace)."""
    return left + right

# ---------------------------------------------------------------------------
# AgentState — the full graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Full graph state traversed by the multi-agent graph."""

    run_id: str
    user_id: str
    task: str
    plan: dict | None                  # orchestrator output (subtasks + assignments)
    plan_reasoning: str                # planner's ordering/priority rationale
    research_output: dict | None       # {"findings": [...]}
    code_output: dict | None           # {"code": "..."}
    analysis_output: dict | None       # {"summary": "..."}
    sql_output: dict | None            # new field
    review: dict | None                # {"verdict": ..., "feedback": ...}
    final_output: dict | None          # {"final_answer": ...}
    review_cycle_count: int
    step_log: Annotated[list, append_step_log]
    error: str | None
    needs_clarification: bool
    # Internal routing fields (not exposed externally)
    _subtasks: list[str]
    _assignments: dict[str, str]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_step_entry(node_name: str) -> dict:
    return {
        "step_id": str(uuid.uuid4()),
        "node": node_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_with_timeout(
    node_func,
    state: dict,
    node_name: str,
) -> dict:
    """Execute *node_func(state)* with a per-node timeout.

    Returns the node's result dict on success, or an ``{"error": ...}`` dict
    on timeout or any other exception.
    
    Note: Since threads cannot be forcibly hard-killed in Python, a genuinely
    hung node will leak a background thread even after timing out.
    """
    timeout = NODE_TIMEOUTS.get(node_name, _DEFAULT_NODE_TIMEOUT)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(node_func, state)
    
    # ── Metric: latency tracking ───────────────────────────────────────
    start_ts = time.monotonic()
    
    try:
        result = future.result(timeout=timeout)
        elapsed = time.monotonic() - start_ts
        logger.info(
            "METRIC latency node=%s duration=%.3fs status=success",
            node_name, elapsed,
        )
    except concurrent.futures.TimeoutError:
        elapsed = time.monotonic() - start_ts
        logger.error(
            "METRIC latency node=%s duration=%.3fs status=timeout",
            node_name, elapsed,
        )
        pool.shutdown(wait=False)  # Don't block on the orphaned thread
        return {"error": f"Node '{node_name}' timed out after {timeout}s"}
    except Exception as exc:
        elapsed = time.monotonic() - start_ts
        logger.error(
            "METRIC latency node=%s duration=%.3fs status=error error=%s",
            node_name, elapsed, exc,
        )
        pool.shutdown(wait=False)
        return {"error": f"Node '{node_name}' failed: {exc}"}
        
    pool.shutdown(wait=False)
    return result


def _build_old_state(state: AgentState) -> dict:
    """Build a dict with the field names the Phase-8 agent nodes expect."""
    old: dict = {
        "task": state.get("task", ""),
        "run_id": state.get("run_id", ""),
        "review_cycles": state.get("review_cycle_count", 0),
    }
    if state.get("_subtasks"):
        old["subtasks"] = state["_subtasks"]
    if state.get("_assignments"):
        old["assignments"] = state["_assignments"]
    if state.get("plan_reasoning"):
        old["plan_reasoning"] = state["plan_reasoning"]
    if state.get("research_output") and "findings" in (state["research_output"] or {}):
        old["research_notes"] = state["research_output"]["findings"]
    if state.get("code_output") and "code" in (state["code_output"] or {}):
        old["code"] = state["code_output"]["code"]
        old["code_verified"] = state["code_output"].get("code_verified", False)
        old["code_exec_output"] = state["code_output"].get("code_exec_output", "")
    if state.get("analysis_output") and "summary" in (state["analysis_output"] or {}):
        old["analysis"] = state["analysis_output"]["summary"]
    return old

# ---------------------------------------------------------------------------
# Wrapped node functions  (bridge between graph AgentState and existing nodes)
# ---------------------------------------------------------------------------

def sql_assistant_wrapper(state: AgentState) -> dict:
    step_entry = _make_step_entry("sql_assistant")
    old_state = _build_old_state(state)

    result = _run_with_timeout(sql_assistant_node, old_state, "sql_assistant")
    result["step_log"] = [step_entry]

    if "error" in result:
        result["sql_output"] = None
        return result

    sql_output = result.pop("sql_output", None)
    result["sql_output"] = sql_output
    return result

def orchestrator_wrapper(state: AgentState) -> dict:
    step_entry = _make_step_entry("orchestrator")
    old_state = _build_old_state(state)

    result = _run_with_timeout(orchestrator_node, old_state, "orchestrator")
    result["step_log"] = [step_entry]

    if "error" in result:
        result["plan"] = None
        result["_subtasks"] = []
        result["_assignments"] = {}
        return result

    subtasks = result.pop("subtasks", [])
    assignments = result.pop("assignments", {})
    result["plan"] = {"subtasks": subtasks, "assignments": assignments}
    result["_subtasks"] = subtasks
    result["_assignments"] = assignments
    return result


def planner_wrapper(state: AgentState) -> dict:
    """
    Refines orchestrator's subtasks with explicit ordering + reasoning.
    Does NOT touch _assignments — agent routing stays entirely
    orchestrator's decision, so route_orchestrator_to_specialists'
    fan-out logic (which reads _assignments) is unaffected by planner.
    """
    step_entry = _make_step_entry("planner")
    old_state = _build_old_state(state)

    result = _run_with_timeout(planner_node, old_state, "planner")
    result["step_log"] = [step_entry]

    if "error" in result:
        # Planner failing is non-fatal to the run — keep whatever
        # subtasks orchestrator already produced rather than losing
        # the plan entirely.
        return result

    subtasks = result.pop("subtasks", state.get("_subtasks", []))
    plan_reasoning = result.pop("plan_reasoning", "")

    result["_subtasks"] = subtasks
    result["plan_reasoning"] = plan_reasoning
    # Keep `plan` in sync so anything reading state["plan"] downstream
    # (e.g. logging, the aggregator, future debugging) sees the
    # planner-refined subtasks, not the pre-refinement ones.
    result["plan"] = {
        "subtasks": subtasks,
        "assignments": state.get("_assignments", {}),
    }
    return result


def researcher_wrapper(state: AgentState) -> dict:
    step_entry = _make_step_entry("researcher")
    old_state = _build_old_state(state)

    result = _run_with_timeout(researcher_node, old_state, "researcher")
    result["step_log"] = [step_entry]

    if "error" in result:
        result["research_output"] = None
        return result

    findings = result.pop("research_notes", [])
    result["research_output"] = {"findings": findings}
    return result


def coder_wrapper(state: AgentState) -> dict:
    step_entry = _make_step_entry("coder")
    old_state = _build_old_state(state)

    result = _run_with_timeout(coder_node, old_state, "coder")
    result["step_log"] = [step_entry]

    if "error" in result:
        result["code_output"] = None
        return result

    code = result.pop("code", "")
    code_verified = result.pop("code_verified", False)
    code_exec_output = result.pop("code_exec_output", "")
    result["code_output"] = {
        "code": code,
        "code_verified": code_verified,
        "code_exec_output": code_exec_output,
    }
    return result


def data_analyst_wrapper(state: AgentState) -> dict:
    step_entry = _make_step_entry("data_analyst")
    old_state = _build_old_state(state)

    result = _run_with_timeout(data_analyst_node, old_state, "data_analyst")
    result["step_log"] = [step_entry]

    if "error" in result:
        result["analysis_output"] = None
        return result

    analysis = result.pop("analysis", "")
    result["analysis_output"] = {"summary": analysis}
    return result


def reviewer_wrapper(state: AgentState) -> dict:
    # Defensive check: if a parallel specialist crashed, bypass the reviewer LLM 
    # to avoid race conditions with LangGraph edge evaluation
    if state.get("error"):
        logger.warning("Error present in state — bypassing reviewer agent.")
        return {
            "review": {"verdict": "error"},
            "review_cycle_count": state.get("review_cycle_count", 0)
        }

    step_entry = _make_step_entry("reviewer")
    old_state = _build_old_state(state)

    result = _run_with_timeout(reviewer_node, old_state, "reviewer")
    result["step_log"] = [step_entry]

    if "error" in result:
        result["review"] = None
        result["review_cycle_count"] = state.get("review_cycle_count", 0) + 1
        return result

    review_feedback = result.pop("review_feedback", "approved")
    review_feedback_text = result.pop("review_feedback_text", "")
    review_cycles = result.pop("review_cycles", state.get("review_cycle_count", 0) + 1)

    result["review"] = {"verdict": review_feedback, "feedback": review_feedback_text}
    result["review_cycle_count"] = review_cycles
    return result


def aggregator_wrapper(state: AgentState) -> dict:
    step_entry = _make_step_entry("aggregator")
    old_state = _build_old_state(state)

    result = _run_with_timeout(aggregator_node, old_state, "aggregator")
    result["step_log"] = [step_entry]

    # ── Metric: step count at end of run ──────────────────────────────
    total_steps = len(state.get("step_log", [])) + 1  # +1 for this aggregator entry
    review_cycles = state.get("review_cycle_count", 0)
    logger.info(
        "METRIC step_count total_steps=%d review_cycles=%d run_id=%s",
        total_steps, review_cycles, state.get("run_id", "unknown"),
    )

    if "error" in result:
        result["final_output"] = None
        return result

    # aggregator_node returns {"final_output": "<string>"}.
    # Store as {"final_answer": "<string>"} so app.py can always read
    # final_result.get("final_answer") without special-casing.
    final_output = result.pop("final_output", "")
    if isinstance(final_output, dict):
        # Defensive: already unwrapped by something upstream
        result["final_output"] = final_output
    else:
        result["final_output"] = {"final_answer": final_output}
    return result

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_orchestrator(state: AgentState):
    """Route after orchestrator: handle clarification, planner bypass, or normal planner."""
    if state.get("needs_clarification"):
        return "end"
    subtasks = state.get("_subtasks", [])
    assignments = state.get("_assignments", {})
    if len(subtasks) == 1 and list(assignments.values()) == ["researcher"]:
        logger.info("Fast-path: bypassing planner for simple research task.")
        return "researcher"
    return "planner"

def route_specialist(state: AgentState) -> Literal["reviewer", "aggregator"]:
    """Bypass reviewer for pure informational queries."""
    assignments = set(state.get("_assignments", {}).values())
    if "coder" in assignments or "data_analyst" in assignments:
        return "reviewer"
    logger.info("Fast-path: bypassing reviewer for pure research task.")
    return "aggregator"

def route_orchestrator_to_specialists(state: AgentState) -> list[Send]:
    """Fan out to **only** the specialist nodes the plan calls for."""
    assignments = state.get("_assignments", {})
    agents = set(assignments.values())

    sends: list[Send] = []
    for agent_name in ("researcher", "coder", "data_analyst", "sql_assistant"):
        if agent_name in agents:
            sends.append(Send(agent_name, state))

    if not sends:
        logger.warning("No specialists assigned — defaulting to researcher")
        sends.append(Send("researcher", state))

    logger.debug("Fan-out sends: %s", [s.node for s in sends])
    return sends


def route_reviewer(state: AgentState) -> Literal["aggregator", "orchestrator"]:
    """Decide next step after review."""
    # Catch any bypass overrides
    if state.get("error"):
        return "aggregator"
        
    review = state.get("review", {})
    verdict = (review or {}).get("verdict", "needs_revision")
    cycle = state.get("review_cycle_count", 0)

    if verdict == "approved":
        logger.info("Review approved (cycle %d/%d) → aggregator", cycle, MAX_REVIEW_CYCLES)
        return "aggregator"

    if cycle >= MAX_REVIEW_CYCLES:
        logger.info(
            "Cycle cap reached (%d >= %d) → forcing to aggregator",
            cycle, MAX_REVIEW_CYCLES,
        )
        return "aggregator"

    logger.info(
        "METRIC feedback_loop trigger=needs_revision cycle=%d/%d route=orchestrator",
        cycle, MAX_REVIEW_CYCLES,
    )
    return "orchestrator"

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct the full multi-agent ``StateGraph``."""
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_wrapper)
    builder.add_node("planner", planner_wrapper)
    builder.add_node("researcher", researcher_wrapper)
    builder.add_node("coder", coder_wrapper)
    builder.add_node("data_analyst", data_analyst_wrapper)
    builder.add_node("sql_assistant", sql_assistant_wrapper)
    builder.add_node("reviewer", reviewer_wrapper)
    builder.add_node("aggregator", aggregator_wrapper)

    builder.set_entry_point("orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"planner": "planner", "researcher": "researcher", "end": END}
    )

    # Planner is the actual fan-out point now. Reused unchanged from
    # before: it only reads state["_assignments"], which planner never
    # modifies, so this fan-out logic is unaffected by planner's insertion.
    builder.add_conditional_edges(
        "planner",
        route_orchestrator_to_specialists,
    )

    for specialist in ("researcher", "coder", "data_analyst"):
        builder.add_conditional_edges(specialist, route_specialist)
        
    builder.add_edge("sql_assistant", "reviewer")

    builder.add_conditional_edges(
        "reviewer",
        route_reviewer,
        {
            "aggregator": "aggregator",
            "orchestrator": "orchestrator",
        },
    )

    builder.add_edge("aggregator", END)

    return builder


def compile_graph():
    """Build and compile the graph; return the compiled ``CompiledGraph``."""
    builder = build_graph()
    return builder.compile()


app = compile_graph()