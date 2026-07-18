"""
Phase 8 — Coder Agent Node

The coder has tools: run_python, read_file, write_file, memory_write.

Agentic loop
------------
Up to 3 iterations of tool use (run_python / read_file / write_file)
before a final structured call produces ``CoderOutput``.

Self-verification
------------------
Before returning, the coder's final code is ALWAYS executed once via
run_python — this is not optional tool use the model may or may not
choose to invoke, it's a forced deterministic check. This catches the
most common failure mode (code that reads plausibly but doesn't
actually run) cheaply and without an extra LLM call, before the code
ever reaches the reviewer agent. The reviewer still makes the final
approve/needs_revision judgment call, but now with an objective
execution result attached instead of having to guess from text alone.
"""

import json
import logging

from agents.orchestrator import AgentState
from agents.schemas import CoderOutput
from agents.structured_call import call_agent_structured, StructuredCallError
from router import ModelRouter
from security.permissions import PermissionLayer, execute_tool
from tools.registry import build_registry
from tools.code_exec import run_python

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

router = ModelRouter()
permission_layer = PermissionLayer()
registry = build_registry()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the coder. Write clean, working Python code to complete "
    "the assigned task. You can run_python to test your code before "
    "returning it. Return valid JSON only."
)

_MAX_TOOL_ITERATIONS = 3

# Markers that indicate run_python's execution actually failed, as opposed
# to succeeding but merely printing text that happens to be unrelated.
# These match the exact strings tools/code_exec.py returns on failure.
_EXEC_FAILURE_MARKERS = ("TimeoutError:", "ExecutionError:", "SyntaxError", "AttributeError:")

# ---------------------------------------------------------------------------
# Tool-call parsing (reused from researcher pattern)
# ---------------------------------------------------------------------------

_CODER_TOOLS = frozenset({"run_python", "read_file", "write_file", "memory_write"})


def _parse_tool_calls(content: str) -> list[dict]:
    """Extract tool call requests from the model's plain-text response."""
    content = content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        return []

    try:
        parsed = json.loads(content[start:end])
    except json.JSONDecodeError:
        return []

    calls = parsed.get("tool_calls", [])
    if not isinstance(calls, list):
        return []

    result = []
    for call in calls:
        if isinstance(call, dict) and "name" in call:
            result.append({
                "name": call["name"],
                "args": call.get("args", call.get("arguments", {})),
            })
    return result


def _execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute each tool call and return message dicts with results."""
    new_messages: list[dict] = []
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})

        if tool_name not in _CODER_TOOLS:
            new_messages.append({
                "role": "user",
                "content": f"[Tool error: {tool_name}] Tool not available to coder.",
            })
            continue

        try:
            result = execute_tool(
                agent_name="coder",
                tool_name=tool_name,
                args=args,
                registry=registry,
                permission_layer=permission_layer,
            )
            logger.info("Tool '%s' succeeded (args=%s)", tool_name, str(args)[:120])
        except PermissionError as e:
            result = f"PermissionError: {e}"
            logger.warning("Tool '%s' blocked: %s", tool_name, e)
        except KeyError as e:
            result = f"ToolNotFoundError: {e}"
            logger.warning("Tool '%s' not found: %s", tool_name, e)
        except Exception as e:
            result = f"ToolError: {e}"
            logger.error("Tool '%s' raised: %s", tool_name, e)

        new_messages.append({
            "role": "user",
            "content": f"[Tool result: {tool_name}]\n{result}",
        })
    return new_messages


def _verify_code(code: str) -> tuple[bool, str]:
    """
    Force-execute *code* via run_python exactly once, regardless of
    whether the model chose to test it during the agentic loop.

    Returns (verified, exec_output) where *verified* is False if the
    execution result contains any of the known failure markers that
    tools/code_exec.py returns on timeout/exception/syntax error.

    This is deliberately NOT model-judged — it's a plain string check
    against run_python's own documented failure-output format, so it
    costs zero extra LLM calls and can't itself hallucinate a result.
    """
    if not code or not code.strip():
        return False, "No code produced — nothing to execute."

    exec_output = run_python(code)
    verified = not any(marker in exec_output for marker in _EXEC_FAILURE_MARKERS)

    if verified:
        logger.info("Coder self-verification PASSED")
    else:
        logger.warning("Coder self-verification FAILED: %s", exec_output[:300])

    return verified, exec_output


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def coder_node(state: AgentState) -> dict:
    """
    LangGraph node — runs the coder agent.

    Returns::

        {
            "code": str,
            "code_verified": bool,       # did run_python execute it cleanly?
            "code_exec_output": str,     # truncated run_python output/error
        }

    On any failure returns a code_verified=False fallback so downstream
    agents (reviewer) can see the coder did not produce working code,
    rather than silently treating a failure the same as success.
    """
    task = state.get("task", "")
    research_notes = state.get("research_notes", [])

    # ── Build tool catalog ────────────────────────────────────────────
    tool_catalog = registry.catalog_for("coder", permission_layer)
    tool_descriptions = "\n".join(
        f"- {t['name']}: {(t['description'] or '').strip().splitlines()[0]}"
        for t in tool_catalog
    )

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Available tools (respond with JSON {{\"tool_calls\": [...]}}):\n"
        f"{tool_descriptions}\n\n"
        "After tool use, you will be asked for a final structured JSON response."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Task: {task}\n"
                f"Research context: {chr(10).join(research_notes)}"
            ),
        },
    ]

    try:
        # ── Agentic loop: up to _MAX_TOOL_ITERATIONS ──────────────────
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response: str = router.call("coder", messages)
            logger.debug("Coder iteration %d response: %s", iteration, response[:300])

            messages.append({"role": "assistant", "content": response})

            tool_calls = _parse_tool_calls(response)
            logger.info(
                "METRIC tool_call_accuracy agent=coder iteration=%d tool_attempts=%d",
                iteration, len(tool_calls),
            )
            if not tool_calls:
                logger.debug("Coder: no tool calls on iteration %d — breaking", iteration)
                break

            tool_result_msgs = _execute_tool_calls(tool_calls)
            messages.extend(tool_result_msgs)
            for tc in tool_calls:
                logger.info(
                    "METRIC tool_call agent=coder iteration=%d tool=%s args=%s",
                    iteration, tc["name"], str(tc.get("args", {}))[:120],
                )

        # ── Final structured call ─────────────────────────────────────
        messages.append({
            "role": "user",
            "content": (
                "Now return your final code as valid JSON matching this schema:\n"
                '{"code": "...", "language": "python", "explanation": "..."}'
            ),
        })

        output: CoderOutput = call_agent_structured(
            router=router,
            agent_name="coder",
            messages=messages,
            schema=CoderOutput,
            max_repairs=2,
        )

        logger.info("Coder produced %d chars of %s code", len(output.code), output.language)

        # ── Mandatory self-verification (forced, not model-optional) ──
        # Only meaningfully verifiable for Python — other languages skip
        # execution but are still marked unverified rather than assumed-good.
        if output.language.strip().lower() == "python":
            verified, exec_output = _verify_code(output.code)
        else:
            verified, exec_output = False, (
                f"Skipped execution — self-verification only supports "
                f"Python, got language='{output.language}'."
            )
            logger.info(
                "Coder self-verification skipped: non-Python language '%s'",
                output.language,
            )

        return {
            "code": output.code,
            "code_verified": verified,
            "code_exec_output": exec_output[:500],
        }

    except Exception as e:
        logger.error("Coder node failed: %s", e)
        return {
            "code": "# Code generation failed",
            "code_verified": False,
            "code_exec_output": f"Coder node exception: {e}",
        }