"""
Phase 8 — Researcher Agent Node

The researcher has tools: web_search, web_fetch, memory_write.

Two-phase execution
-------------------
1. First router.call() → plain text / tool-call JSON.
   Parse any requested tool calls, execute them, append results.
2. Second call_agent_structured() → ResearcherOutput (structured findings).

Findings are written to Qdrant under ``user_id = state["run_id"]`` so
other agents (reviewer, aggregator) can retrieve them later.
"""

import json
import logging
from typing import Any

from agents.orchestrator import AgentState
from agents.schemas import ResearcherOutput
from agents.structured_call import call_agent_structured, StructuredCallError
from router import ModelRouter
from security.permissions import PermissionLayer, execute_tool
from tools.registry import build_registry
from memory.qdrant_store import QdrantStore
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

router = ModelRouter()
permission_layer = PermissionLayer()
registry = build_registry()

def _embed(text: str) -> list[float]:
    import httpx
    resp = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

_task_history_store = QdrantStore(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection="task_history",
    embedding_fn=_embed,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional research analyst. Your job is to produce "
    "comprehensive, well-structured research reports.\n\n"
    "IMPORTANT: When you want to call tools, respond with ONLY a JSON object in this exact format:\n"
    '{"tool_calls": [{"name": "web_search", "args": {"query": "your search query"}}]}\n'
    "Do NOT include any prose, code fences, or explanation — ONLY the JSON object.\n"
    "Each tool call must be an object with 'name' (string) and 'args' (dict of parameters).\n\n"
    "Research standards you must follow:\n"
    "- Find {min_findings}-{max_findings} distinct, substantive findings per query — \n"
    "  NEVER exceed {max_findings} findings even if more information is available.\n"
    "  Prioritize quality and depth over quantity.\n"
    "- Each finding must be a complete sentence with specific details, "
    "numbers, dates, or technical specifics — not vague summaries\n"
    "- Every finding must be traceable to a source URL\n"
    "- Never hallucinate sources — only include URLs you actually fetched\n"
    "- Cover multiple angles: technical details, comparisons, trade-offs, "
    "real-world usage, and recent developments\n"
    "- If the first web search is insufficient, use web_fetch on the most "
    "promising URLs to get deeper content\n"
    "ABSOLUTE RULE: Only include a source URL if you actually called "
    "web_search or web_fetch and received it in a tool result. NEVER "
    "write a citation like 'Author (Year)' or invent a source you did "
    "not retrieve. If you have no real sources for a finding, mark the "
    "finding's source as 'no source retrieved' rather than inventing one.\n"
    "You MUST call web_search before drawing any conclusions."
)

# ---------------------------------------------------------------------------
# Tool-call parsing helpers
# ---------------------------------------------------------------------------

def _parse_tool_calls(content: str) -> list[dict]:
    """
    Extract tool call requests from the model's plain-text response.

    Handles two response formats the LLM may produce:

    1. Structured (preferred)::

        {"tool_calls": [{"name": "web_search", "args": {"query": "..."}}]}

    2. Flat string list (fallback — model listed names without args)::

        {"tool_calls": ["web_search", "web_fetch"]}

    When the flat format is detected, a default query derived from the
    surrounding text is injected so a real search can still be attempted.

    Returns a (possibly empty) list of ``{"name": str, "args": dict}`` dicts.
    Falls back to ``[]`` on any parse error so the caller can proceed to the
    structured phase regardless.
    """
    content = content.strip()

    # Scan ALL JSON objects in the response and pick the one with tool_calls.
    # The model sometimes wraps JSON in prose / code fences, so we must
    # find every '{' and try each candidate rather than trusting the first.
    parsed = None
    pos = 0
    while pos < len(content):
        start = content.find("{", pos)
        if start == -1:
            break
        # Walk forward to find the matching closing brace
        depth = 0
        end = start
        in_string = False
        escape_next = False
        for i, ch in enumerate(content[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = content[start:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "tool_calls" in obj:
                parsed = obj
                break  # found our target
        except json.JSONDecodeError:
            pass
        pos = start + 1

    if parsed is None:
        return []

    calls = parsed.get("tool_calls", [])
    if not isinstance(calls, list):
        return []

    result = []
    for call in calls:
        if isinstance(call, dict) and "name" in call:
            # Standard structured format
            result.append({
                "name": call["name"],
                "args": call.get("args", call.get("arguments", {})),
            })
        elif isinstance(call, str):
            # Flat string format: model listed tool names without args.
            # Only auto-trigger web_search (skip memory_write/web_fetch
            # as they need specific arguments we don't have here).
            if call == "web_search":
                # Extract a query hint from the surrounding prose if possible.
                # The task text is not available here, so we use a sentinel;
                # the caller will substitute the real task text.
                result.append({"name": "web_search", "args": {"query": "__TASK__"}})
            # Ignore other string-only entries (no usable args)
    return result


def _execute_tool_calls(
    tool_calls: list[dict],
    agent_name: str,
) -> list[dict]:
    """
    Execute each tool call in *tool_calls* and return a list of assistant
    + tool-result message pairs to append to the conversation.

    Errors are caught per-call so one failed tool doesn't abort the rest.
    """
    new_messages: list[dict] = []
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        try:
            result = execute_tool(
                agent_name=agent_name,
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


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> dict:
    """
    LangGraph node — runs the researcher agent.

    Returns ``{"research_notes": list[str]}``.
    On any failure returns ``{"research_notes": ["Research failed — no findings"]}``.
    """
    task = state.get("task", "")
    assignments = state.get("assignments", {})
    run_id = state.get("run_id", "__default__")
    subtasks = state.get("subtasks", [])

    # My assigned subtask keys — accept both integer-string keys ("0", "1")
    # produced by orchestrator_node and label keys ("task_1") used in tests.
    my_subtask_keys = [k for k, v in assignments.items() if v == "researcher"]

    COMPREHENSIVE_KEYWORDS = [
        "history of", "all major", "comprehensive", "timeline", 
        "evolution of", "complete guide", "everything about",
    ]
    is_comprehensive = any(kw in task.lower() for kw in COMPREHENSIVE_KEYWORDS)
    max_findings = 15 if is_comprehensive else 10
    max_tool_calls = 8 if is_comprehensive else 5
    min_findings = max_findings - 4

    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace(
        "{min_findings}", str(min_findings)
    ).replace(
        "{max_findings}", str(max_findings)
    )

    # ── Build tool catalog for researcher ────────────────────────────────
    tool_catalog = registry.catalog_for("researcher", permission_layer)
    tool_descriptions = "\n".join(
        f"- {t['name']}: {(t['description'] or '').strip().splitlines()[0]}"
        for t in tool_catalog
    )

    system_content = (
        f"{system_prompt}\n\n"
        f"Available tools (respond with JSON {{\"tool_calls\": [...]}}):\n"
        f"{tool_descriptions}\n\n"
        "After tool use, you will be asked for a final structured JSON response."
    )

    # Build a human-readable list of my subtasks for the prompt
    my_subtask_descriptions: list[str] = []
    for k in my_subtask_keys:
        # Integer-string keys ("0") → look up in subtasks list
        if k.isdigit() and int(k) < len(subtasks):
            my_subtask_descriptions.append(subtasks[int(k)])
        else:
            # Label key (e.g. "task_1") — use the task itself as the description
            my_subtask_descriptions.append(task)

    subtask_text = "\n".join(f"- {d}" for d in my_subtask_descriptions) or f"- {task}"

    past_findings = []
    try:
        past_findings = _task_history_store.retrieve_memory(
            user_id=state.get("user_id", "default"),
            query=task,
            top_k=3,
            score_threshold=0.75,
        )
    except Exception as e:
        logger.warning("Memory retrieval failed (non-fatal): %s", e)

    memory_context = ""
    if past_findings:
        notes = "\n".join(f"- {f['text']}" for f in past_findings)
        memory_context = (
            f"\n\nRelevant findings from past research (verify and expand, "
            f"don't just repeat):\n{notes}"
        )
        logger.info("Retrieved %d past findings from memory", len(past_findings))

    system_content += memory_context

    user_content = (
        f"Task: {task}\n\n"
        f"Your subtasks:\n{subtask_text}\n\n"
    )
    
    review = state.get("review") or {}
    if review.get("verdict") == "needs_revision" and review.get("feedback"):
        user_content += f"PREVIOUS ATTEMPT FAILED. Reviewer feedback to fix:\n{review['feedback']}\n\n"

    user_content += (
        "Call web_search NOW with a relevant query. "
        "Respond with ONLY a JSON object: "
        '{"tool_calls": [{"name": "web_search", "args": {"query": "..."}}]}'
    )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        # ── Phase 1: tool-use call ────────────────────────────────────────
        first_response: str = router.call("researcher", messages)
        logger.debug("Researcher first response: %s", first_response[:300])

        messages.append({"role": "assistant", "content": first_response})

        tool_calls = _parse_tool_calls(first_response)

        if len(tool_calls) > max_tool_calls:
            logger.warning("Researcher requested %d tool calls, capping at %d", 
                            len(tool_calls), max_tool_calls)
            tool_calls = tool_calls[:max_tool_calls]

        # Substitute the __TASK__ sentinel injected by the flat-string fallback
        # so the web_search actually receives the real task as its query.
        for tc in tool_calls:
            if tc.get("name") == "web_search":
                args = tc.get("args", {})
                if args.get("query") == "__TASK__":
                    args["query"] = task

        logger.info(
            "METRIC tool_call_accuracy agent=researcher tool_attempts=%d",
            len(tool_calls),
        )
        if tool_calls:
            tool_result_msgs = _execute_tool_calls(tool_calls, agent_name="researcher")
            messages.extend(tool_result_msgs)
            for tc in tool_calls:
                logger.info("METRIC tool_call agent=researcher tool=%s args=%s", tc["name"], str(tc.get("args", {}))[:120])
        else:
            # Hard fallback: if the model still produced no tool calls at all,
            # run web_search directly with the task as the query.
            logger.warning("Researcher: no tool calls parsed — forcing web_search fallback")
            forced_calls = [{"name": "web_search", "args": {"query": task}}]
            tool_result_msgs = _execute_tool_calls(forced_calls, agent_name="researcher")
            messages.extend(tool_result_msgs)
            logger.info("METRIC tool_call agent=researcher tool=web_search args={'query': '%s'}", task[:80])

        # ── Phase 2: structured findings call ──────────────────────────────────
        messages.append({
            "role": "user",
            "content": (
                "Now write a comprehensive research report as valid JSON.\n\n"
                "Requirements:\n"
                f"- findings: at least {min_findings} detailed bullet points, each 2-3 sentences "
                "long with specific technical details\n"
                "- sources: list must have the SAME length as findings list — one URL per finding, in matching order. "
                "Do NOT just provide a summary list of URLs at the end.\n"
                "- confidence: your confidence score 0.0-1.0\n\n"
                "JSON schema:\n"
                '{"findings": ["detailed finding 1...", "detailed finding 2...", ...], '
                '"sources": ["https://...", "https://...", ...], '
                '"confidence": 0.85}\n\n'
                "Do NOT summarize vaguely. Each finding should contain enough detail "
                "that someone reading it learns something specific and actionable."
            ),
        })

        output: ResearcherOutput = call_agent_structured(
            router=router,
            agent_name="researcher",
            messages=messages,
            schema=ResearcherOutput,
            max_repairs=3,
        )

        # Reject fabricated academic-style citations (Name (Year) pattern)
        import re
        fake_citation_pattern = re.compile(r'^[A-Z][a-z]+\s*\(\d{4}\)$')
        for i, source in enumerate(output.sources):
            if fake_citation_pattern.match(source.strip()):
                logger.warning(
                    "Researcher produced fake citation '%s' instead of a "
                    "real URL — this finding may be hallucinated", source
                )

        logger.info(
            "Researcher produced %d findings with %d sources (confidence=%.2f)",
            len(output.findings),
            len(output.sources),
            output.confidence,
        )
        findings = output.findings

    except (StructuredCallError, Exception) as e:
        logger.error("Researcher node failed: %s", e)
        return {"research_notes": []}

    # Outside try/except — memory failure never triggers fallback
    user_id = state.get("user_id", "default")
    try:
        _write_findings_to_memory(output, user_id=user_id, run_id=run_id, task=task)
    except Exception as mem_err:
        logger.warning("Researcher: memory write failed (non-fatal): %s", mem_err)

    return {"research_notes": findings}


# ---------------------------------------------------------------------------
# Memory write helper (separated for easy mocking in tests)
# ---------------------------------------------------------------------------

def _write_findings_to_memory(
    output: ResearcherOutput,
    user_id: str,
    run_id: str,
    task: str,
) -> None:
    """
    Write each finding to Qdrant keyed by *user_id* so future calls by the
    same user can retrieve them via retrieve_memory(user_id=user_id, ...).

    Errors are caught and logged so a memory failure never prevents findings
    from being returned.
    """
    try:
        # Import here to allow tests to patch before module-level init
        from memory.qdrant_store import QdrantStore
        import os

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        # Lazy embedding function — avoids mandatory Ollama at import time
        def _embed(text: str) -> list[float]:
            import httpx as _httpx
            resp = _httpx.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

        store = QdrantStore(
            url=qdrant_url,
            collection="task_history",
            embedding_fn=_embed,
        )
        for finding in output.findings:
            store.write_memory(
                user_id=user_id,          # persistent key — retrievable by same user
                text=finding,
                metadata={
                    "agent": "researcher",
                    "run_id": run_id,     # kept for audit/traceability
                    "task": task[:200],
                    "sources": output.sources,
                    "confidence": output.confidence,
                },
            )
        logger.info(
            "Researcher wrote %d findings to memory (user_id=%s, run_id=%s)",
            len(output.findings), user_id, run_id,
        )
    except Exception as e:
        logger.warning("Researcher: failed to write findings to Qdrant: %s", e)
