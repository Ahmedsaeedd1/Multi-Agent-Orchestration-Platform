"""
Phase 6 — Tool Registry

Central registry for all agent-accessible tools.  Every tool is a
``ToolSpec`` containing its name, description, implementation function,
and the permission required to call it.

Permission-based design (matches config/permissions.yaml)
---------------------------------------------------------
Tools declare a ``required_permission`` string (e.g. "web_access").
Agents are granted a list of permissions in permissions.yaml.
The PermissionLayer (security/permissions.py) enforces this at
call-time — the registry only handles discovery and catalog building.

This means adding a new agent never requires editing ToolSpec definitions —
only permissions.yaml needs updating.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolSpec:
    """Specification for a single tool available to agents."""

    name: str                            # unique tool identifier
    description: str                     # shown to the LLM in the system prompt
    fn: Callable                         # the actual Python function to execute
    required_permission: str             # e.g. "web_access", "code_execution"
    parameters: dict = field(default_factory=dict)  # JSON schema for LLM tool-call
    requires_confirmation: bool = False  # reserved for human-in-the-loop


class ToolRegistry:
    """Central registry of all tools."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Add a ToolSpec. Raises ValueError if name already registered."""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec

    def get(self, tool_name: str) -> ToolSpec:
        """
        Return the ToolSpec for *tool_name*.
        Raises KeyError if not registered.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")
        return self._tools[tool_name]

    def catalog_for(self, agent_name: str, permission_layer=None) -> list[dict]:
        """
        Return tool parameter schemas for tools this agent is permitted
        to see, based on the permission layer.

        This is what gets injected into the agent's system prompt as
        the available tools list — agents never see tools they can't call.
        """
        if permission_layer is None:
            from security.permissions import PermissionLayer
            permission_layer = PermissionLayer()
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
            if permission_layer.allows(agent_name, t.required_permission)
        ]

    def all_tools(self) -> list[ToolSpec]:
        """Return all registered ToolSpecs (for audit/debug purposes)."""
        return list(self._tools.values())


def build_registry() -> "ToolRegistry":
    """
    Build and return a ToolRegistry with all 7 tools registered.

    Import deferred inside the function to avoid circular imports —
    this module must not import from tools.web / tools.code_exec etc.
    at module load time because those modules may import from here.

    All agents should call this once at module level::

        from tools.registry import build_registry
        registry = build_registry()
    """
    from tools.web import web_search, web_fetch
    from tools.code_exec import run_python
    from tools.files import read_file, write_file
    from tools.sql import execute_sql, get_schema
    from tools.code_eval import run_static_analysis, run_sandboxed_with_edge_cases

    r = ToolRegistry()

    r.register(ToolSpec(
        name="web_search",
        description=web_search.__doc__ or "Search the web.",
        fn=web_search,
        required_permission="web_access",
        parameters={"query": {"type": "string", "description": "Search query"}},
    ))
    r.register(ToolSpec(
        name="web_fetch",
        description=web_fetch.__doc__ or "Fetch a URL.",
        fn=web_fetch,
        required_permission="web_access",
        parameters={"url": {"type": "string", "description": "URL to fetch"}},
    ))
    r.register(ToolSpec(
        name="run_python",
        description=run_python.__doc__ or "Execute Python code.",
        fn=run_python,
        required_permission="code_execution",
        parameters={"code": {"type": "string", "description": "Python code to execute"}},
    ))
    r.register(ToolSpec(
        name="read_file",
        description=read_file.__doc__ or "Read a file.",
        fn=read_file,
        required_permission="file_read",
        parameters={"path": {"type": "string", "description": "Relative file path"}},
    ))
    r.register(ToolSpec(
        name="write_file",
        description=write_file.__doc__ or "Write a file.",
        fn=write_file,
        required_permission="file_write",
        parameters={
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Content to write"},
        },
    ))
    r.register(ToolSpec(
        name="memory_read",
        description="Read from long-term semantic memory for the current user.",
        fn=lambda **kwargs: None,   # wired to QdrantStore.retrieve_memory in Phase 9
        required_permission="memory_read",
        parameters={
            "query": {"type": "string", "description": "Search query for memory"},
        },
    ))
    r.register(ToolSpec(
        name="memory_write",
        description="Write to long-term semantic memory for the current user.",
        fn=lambda **kwargs: None,   # wired to QdrantStore.write_memory in Phase 9
        required_permission="memory_write",
        parameters={
            "text": {"type": "string", "description": "Text to store in memory"},
        },
    ))

    r.register(ToolSpec(
        name="execute_sql",
        description=execute_sql.__doc__ or "Execute a READ-ONLY SQL query.",
        fn=execute_sql,
        required_permission="sql_read",
        parameters={
            "query": {"type": "string", "description": "SQL query to execute"},
            "db_path": {"type": "string", "description": "Path to SQLite database (optional)"}
        },
    ))
    r.register(ToolSpec(
        name="get_schema",
        description=get_schema.__doc__ or "Get schema information.",
        fn=get_schema,
        required_permission="sql_read",
        parameters={
            "table_name": {"type": "string", "description": "Specific table name (optional)"},
            "db_path": {"type": "string", "description": "Path to SQLite database (optional)"}
        },
    ))
    r.register(ToolSpec(
        name="run_static_analysis",
        description=run_static_analysis.__doc__ or "Run static analysis on code.",
        fn=run_static_analysis,
        required_permission="code_execution",
        parameters={
            "code": {"type": "string", "description": "Python code to analyze"}
        },
    ))
    r.register(ToolSpec(
        name="run_sandboxed_with_edge_cases",
        description=run_sandboxed_with_edge_cases.__doc__ or "Run code in a sandbox with specific edge cases.",
        fn=run_sandboxed_with_edge_cases,
        required_permission="code_execution",
        parameters={
            "code": {"type": "string", "description": "Python code to execute"},
            "edge_case_inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "input": {"description": "The input to pass to the code"}
                    }
                },
                "description": "List of edge cases to run"
            }
        },
    ))

    return r