"""
Phase 3 — Structured I/O with Repair-on-Failure Loop
"""

import re
from typing import Type

from pydantic import BaseModel, ValidationError

from router import ModelRouter


class StructuredCallError(Exception):
    def __init__(self, agent_name: str, raw_response: str, validation_error: Exception):
        self.agent_name = agent_name
        self.raw_response = raw_response
        self.validation_error = validation_error
        super().__init__(
            f"Agent '{agent_name}' failed to produce valid structured output "
            f"after max repair attempts. Last raw response:\n{raw_response[:500]}"
        )


_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)```\s*$", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    return _THINK_RE.sub("", text)


def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences ONLY when they wrap the entire response.

    Uses a fully-anchored pattern (^ ... $) so that inner ```python``` blocks
    embedded inside a JSON string value — e.g.::

        {"final_answer": "### Code\n\n```python\n...\n```"}

    — are not matched.  Without anchoring, re.search() would find the first
    inner fence and return garbled content instead of the surrounding JSON.

    If no outer fence is found the original text is returned stripped.
    """
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def _format_validation_errors(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = " -> ".join(str(x) for x in e["loc"])
        lines.append(f"  - `{loc}`: {e['msg']}")
    return "\n".join(lines)


def _recover_truncated_json(text: str) -> str:
    """
    Attempt to close truncated JSON from providers that hit output token limits,
    and escape literal newlines inside JSON strings so that they parse correctly.
    """
    if not text.strip().startswith("{"):
        return text

    in_string = False
    escape_next = False
    stack: list[str] = []  # tracks '{' and '[' nesting
    fixed_chars = []

    for ch in text:
        if escape_next:
            escape_next = False
            fixed_chars.append(ch)
            continue
        if ch == "\\":
            escape_next = True
            fixed_chars.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            fixed_chars.append(ch)
            continue
            
        if in_string and ch == '\n':
            # Escape literal newlines inside JSON strings
            fixed_chars.append('\\')
            fixed_chars.append('n')
            continue

        fixed_chars.append(ch)

        if not in_string:
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    fixed_text = "".join(fixed_chars)

    suffix = ""
    if in_string:
        suffix += '"'  # close the unterminated string value
    # close any open objects/arrays in reverse order
    for opener in reversed(stack):
        suffix += "}" if opener == "{" else "]"

    if suffix:
        return fixed_text + suffix
    return fixed_text


def call_agent_structured(
    router: ModelRouter,
    agent_name: str,
    messages: list,
    schema: Type[BaseModel],
    max_repairs: int = 3,
) -> BaseModel:
    import json
    
    # Inject schema into the messages automatically to ensure the LLM knows the expected output format
    local_messages = list(messages)
    schema_instr = (
        "You must respond ONLY with valid JSON that matches the following JSON Schema:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
        "Do not include any explanation or conversational text outside the JSON."
    )
    
    if local_messages and local_messages[-1]["role"] == "user":
        # Append to the last user message
        local_messages[-1] = {
            "role": "user",
            "content": local_messages[-1]["content"] + "\n\n" + schema_instr
        }
    else:
        local_messages.append({"role": "user", "content": schema_instr})

    last_error = None
    last_response = ""

    for attempt in range(max_repairs + 1):
        # router.call() already returns the extracted content string —
        # NOT a raw OpenAI completion object. Do not re-access
        # .choices[0].message.content here; that was the bug (double
        # unwrapping a str, causing "'str' object has no attribute 'choices'").
        last_response = router.call(agent_name, local_messages) or ""

        # Strip <think> blocks from model output (DeepSeek-R1/Qwen3 emit these)
        clean = _strip_think_blocks(last_response)

        # Strip markdown code fences that many LLMs wrap around their JSON.
        clean = _strip_fences(clean)

        # Recover truncated JSON — some free-tier providers cut responses
        # mid-string when they hit their output token cap.  Try to close
        # any unclosed strings / objects before attempting validation so that
        # the repair loop isn't wasted on a parse error we can fix ourselves.
        clean = _recover_truncated_json(clean)

        try:
            return schema.model_validate_json(clean)
        except (ValidationError, ValueError) as e:
            last_error = e
            if attempt < max_repairs:
                error_detail = (
                    _format_validation_errors(e)
                    if isinstance(e, ValidationError)
                    else str(e)
                )
                repair_msg = (
                    "Your previous response failed schema validation. "
                    "Fix ONLY the following fields and return valid JSON "
                    f"matching the expected schema.\n\n{error_detail}"
                )
                local_messages.append({"role": "assistant", "content": last_response})
                local_messages.append({"role": "user", "content": repair_msg})

    raise StructuredCallError(
        agent_name=agent_name,
        raw_response=last_response,
        validation_error=last_error,
    )