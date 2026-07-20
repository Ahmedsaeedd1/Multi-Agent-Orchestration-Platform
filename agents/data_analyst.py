"""
Phase 8 — Data Analyst Agent Node
"""

import json
import logging

from agents.orchestrator import AgentState
from agents.schemas import DataAnalystOutput
from agents.structured_call import call_agent_structured, StructuredCallError
from router import ModelRouter
from security.permissions import PermissionLayer, execute_tool
from tools.registry import build_registry

logger = logging.getLogger(__name__)

router = ModelRouter()
permission_layer = PermissionLayer()
registry = build_registry()

SYSTEM_PROMPT = (
    "You are the data analyst. Analyze the data and code provided. "
    "You can run_python to execute data analysis scripts. "
    "Return valid JSON only."
)

_MAX_TOOL_ITERATIONS = 2
_ANALYST_TOOLS = frozenset({"run_python", "read_file", "memory_write"})


def _parse_tool_calls(content: str) -> list[dict]:
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
    new_messages: list[dict] = []
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})

        if tool_name not in _ANALYST_TOOLS:
            new_messages.append({
                "role": "user",
                "content": f"[Tool error: {tool_name}] Tool not available to data analyst.",
            })
            continue

        try:
            result = execute_tool(
                agent_name="data_analyst",
                tool_name=tool_name,
                args=args,
                registry=registry,
                permission_layer=permission_layer,
            )
            logger.info("Tool '%s' succeeded", tool_name)
        except PermissionError as e:
            result = f"PermissionError: {e}"
            logger.warning("Tool '%s' blocked: %s", tool_name, e)
        except Exception as e:
            result = f"ToolError: {e}"
            logger.warning("Tool '%s' failed: %s", tool_name, e)

        new_messages.append({
            "role": "user",
            "content": f"[Tool result: {tool_name}]\n{result}",
        })
    return new_messages


def data_analyst_node(state: AgentState) -> dict:
    task = state.get("task", "")
    research_notes = state.get("research_notes", [])
    code = state.get("code", "")

    tool_catalog = registry.catalog_for("data_analyst", permission_layer)
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

    user_content = (
        f"Task: {task}\n"
        f"Research context: {chr(10).join(research_notes)}\n"
        f"Code: {code}"
    )

    review = state.get("review") or {}
    if review.get("verdict") == "needs_revision" and review.get("feedback"):
        user_content += f"\n\nPREVIOUS ATTEMPT FAILED. Reviewer feedback to fix:\n{review['feedback']}"

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response: str = router.call("data_analyst", messages)
            messages.append({"role": "assistant", "content": response})

            tool_calls = _parse_tool_calls(response)
            if not tool_calls:
                break

            tool_result_msgs = _execute_tool_calls(tool_calls)
            messages.extend(tool_result_msgs)

        messages.append({
            "role": "user",
            "content": (
                "Now return your final analysis as valid JSON matching this schema:\n"
                '{"summary": "...", "insights": ["..."], "code_used": "..."}'
            ),
        })

        output: DataAnalystOutput = call_agent_structured(
            router=router,
            agent_name="data_analyst",
            messages=messages,
            schema=DataAnalystOutput,
            max_repairs=2,
        )

        return {"analysis": output.summary}

    except Exception as e:
        logger.error("Data analyst failed: %s", e)
        return {"analysis": ""}
