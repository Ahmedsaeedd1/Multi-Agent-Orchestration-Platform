"""Tests for tools/ — registry scoping, tool execution, path traversal protection."""

import sys
import os
import tempfile
import time
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.registry import ToolRegistry, ToolSpec
from tools.code_exec import run_python
from tools.files import read_file, write_file


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def fresh_registry():
    """Return an empty ToolRegistry with a known set of tools registered."""
    r = ToolRegistry()
    r.register(ToolSpec("tool_a", "Does A", lambda: "A", ["agent1"]))
    r.register(ToolSpec("tool_b", "Does B", lambda: "B", ["agent2"]))
    r.register(ToolSpec("tool_c", "Does C", lambda: "C", ["agent1", "agent2"]))
    return r


# ===================================================================
# Registry scoping (unit — fully mocked)
# ===================================================================

@pytest.mark.unit
class TestRegistryScoping:
    """catalog_for and get — no external dependencies."""

    def test_reviewer_only_gets_memory_read(self):
        """catalog_for('reviewer') returns only memory_read."""
        from tools import registry
        catalog = registry.catalog_for("reviewer")
        names = [t["name"] if isinstance(t, dict) else t.name for t in catalog]
        assert names == ["memory_read"]

    def test_coder_gets_no_web_tools(self):
        """catalog_for('coder') returns run_python, read_file, write_file, memory_write — no web."""
        from tools import registry
        catalog = registry.catalog_for("coder")
        names = set(t["name"] if isinstance(t, dict) else t.name for t in catalog)
        assert "run_python" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "memory_write" in names
        assert "web_search" not in names
        assert "web_fetch" not in names

    def test_researcher_gets_no_file_or_code_tools(self):
        """catalog_for('researcher') returns web_search, web_fetch, memory_write — no file/code."""
        from tools import registry
        catalog = registry.catalog_for("researcher")
        names = set(t["name"] if isinstance(t, dict) else t.name for t in catalog)
        assert "web_search" in names
        assert "web_fetch" in names
        assert "memory_write" in names
        assert "run_python" not in names
        assert "read_file" not in names
        assert "write_file" not in names

    def test_get_by_name(self, fresh_registry):
        """get('tool_a') returns the correct ToolSpec."""
        spec = fresh_registry.get("tool_a")
        assert spec.name == "tool_a"
        assert spec.fn() == "A"

    def test_get_nonexistent_raises_key_error(self, fresh_registry):
        """get('nonexistent') raises KeyError."""
        with pytest.raises(KeyError):
            fresh_registry.get("nonexistent")

    def test_register_duplicate_raises(self, fresh_registry):
        """Registering a duplicate name raises ValueError."""
        dup = ToolSpec("tool_a", "Duplicate", lambda: None, ["agent1"])
        with pytest.raises(ValueError, match="already registered"):
            fresh_registry.register(dup)

    def test_catalog_for_returns_empty_when_no_tools_assigned(self, fresh_registry):
        """An agent with no tools gets an empty list."""
        assert fresh_registry.catalog_for("agent_with_none") == []


# ===================================================================
# run_python execution (integration — spawns real subprocess)
# ===================================================================

@pytest.mark.integration
class TestRunPython:
    """Sandboxed code execution tests."""

    def test_basic_print(self):
        """run_python('print("hello")') returns a string containing 'hello'."""
        result = run_python('print("hello")')
        assert "hello" in result

    def test_timeout(self):
        """run_python with sleep(15) returns a TimeoutError."""
        result = run_python("import time; time.sleep(15)")
        assert "TimeoutError" in result

    def test_stdout_and_stderr(self):
        """Both stdout and stderr are captured."""
        code = "import sys; print('out'); sys.stderr.write('err')"
        result = run_python(code)
        assert "out" in result
        assert "err" in result

    def test_syntax_error(self):
        """A syntax error is captured and returned."""
        result = run_python("if True")
        assert "SyntaxError" in result or "ExecutionError" in result

    def test_network_disabled(self):
        """Importing 'socket' inside sandbox should fail."""
        result = run_python("import socket")
        assert "disabled" in result or "ImportError" in result
    def test_sandbox_internals_not_leaked(self):
        """User code must not have direct access to the sandbox's internal
        _socket reference — that would let it bypass the import guard
        entirely by using an already-imported module object."""
        result = run_python("print(_socket)")
        assert "NameError" in result

    def test_original_import_not_accessible(self):
        """User code must not be able to call the unrestricted
        _original_import directly to sidestep _safe_import."""
        result = run_python("_original_import('socket')")
        assert "NameError" in result


# ===================================================================
# File tools path-traversal protection (unit — no real files needed)
# ===================================================================

@pytest.mark.unit
class TestFileToolsPathTraversal:
    """read_file / write_file path-escaping protection."""

    @pytest.mark.skipif(os.name != "nt", reason="backslash is only a path separator on Windows")
    def test_read_file_outside_cwd_raises_permission_error(self):
        """read_file with a path outside CWD raises PermissionError."""
        with pytest.raises(PermissionError):
            read_file("..\\..\\etc\\passwd")

    def test_read_file_outside_cwd_forward_slash(self):
        """read_file with Unix-style path traversal raises PermissionError."""
        with pytest.raises(PermissionError):
            read_file("../../../etc/passwd")

    

    def test_write_file_outside_cwd_forward_slash(self):
        """write_file with Unix-style path traversal raises PermissionError."""
        with pytest.raises(PermissionError):
            write_file("../../../tmp/evil", "bad")