"""
Tool registry initialisation.

Registers all 7 tools that agents can use.  Memory tools (memory_read,
memory_write) are stubs that will be wired to ``QdrantStore`` in Phase 9.
"""

from tools.registry import ToolRegistry, ToolSpec
from tools.web import web_search, web_fetch
from tools.code_exec import run_python
from tools.files import read_file, write_file

registry = ToolRegistry()

registry.register(ToolSpec(
    name="web_search",
    description=web_search.__doc__,
    fn=web_search,
    required_permission="web_access",
))
registry.register(ToolSpec(
    name="web_fetch",
    description=web_fetch.__doc__,
    fn=web_fetch,
    required_permission="web_access",
))
registry.register(ToolSpec(
    name="run_python",
    description=run_python.__doc__,
    fn=run_python,
    required_permission="code_execution",
))
registry.register(ToolSpec(
    name="read_file",
    description=read_file.__doc__,
    fn=read_file,
    required_permission="file_read",
))
registry.register(ToolSpec(
    name="write_file",
    description=write_file.__doc__,
    fn=write_file,
    required_permission="file_write",
))
registry.register(ToolSpec(
    name="memory_read",
    description="Read from long-term semantic memory for the current user.",
    fn=lambda **kwargs: None,   # wired to QdrantStore.retrieve_memory in Phase 9
    required_permission="memory_read",
))
registry.register(ToolSpec(
    name="memory_write",
    description="Write to long-term semantic memory for the current user.",
    fn=lambda **kwargs: None,   # wired to QdrantStore.write_memory in Phase 9
    required_permission="memory_write",
))

__all__ = ["registry", "ToolRegistry", "ToolSpec"]