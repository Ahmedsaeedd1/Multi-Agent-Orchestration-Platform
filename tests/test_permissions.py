"""Tests for security/permissions.py — Phase 7."""

import sys
import os
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from security.permissions import PermissionLayer, execute_tool
from tools.registry import ToolSpec, ToolRegistry


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def perms():
    """PermissionLayer loaded from the real config/permissions.yaml."""
    return PermissionLayer()


@pytest.fixture
def mock_run_python():
    """A MagicMock that simulates run_python."""
    return MagicMock(return_value="executed successfully")


@pytest.fixture
def registry_with_exec_tool(mock_run_python):
    """A ToolRegistry with run_python and write_file registered (no other tools)."""
    r = ToolRegistry()
    r.register(ToolSpec(
        name="run_python",
        description="Execute Python code",
        fn=mock_run_python,
        required_permission="code_execution",
    ))
    r.register(ToolSpec(
        name="write_file",
        description="Write a file",
        fn=MagicMock(return_value="ok: test.txt"),  # should never be called
        required_permission="file_write",
    ))
    return r


# ===================================================================
# Tests
# ===================================================================

class TestPermissionLayer:
    """PermissionLayer.allows and enforce."""

    def test_allows_granted(self, perms):
        """researcher can use web_access."""
        assert perms.allows("researcher", "web_access") is True

    def test_allows_denied(self, perms):
        """reviewer cannot use code_execution."""
        assert perms.allows("reviewer", "code_execution") is False

    def test_enforce_allowed_does_not_raise(self, perms):
        """coder can use code_execution — enforce should not raise."""
        perms.enforce("coder", "code_execution")  # should not raise

    def test_enforce_denied_raises(self, perms):
        """reviewer using file_write should raise PermissionError."""
        with pytest.raises(PermissionError, match="reviewer"):
            perms.enforce("reviewer", "file_write")


class TestExecuteTool:
    """execute_tool function — enforcement boundary."""

    def test_execute_tool_blocked(self, perms, registry_with_exec_tool, mock_run_python):
        """
        Calling write_file as reviewer raises PermissionError BEFORE
        the function executes.  Confirm the mock fn was never called.
        """
        write_file_spec = registry_with_exec_tool.get("write_file")
        with pytest.raises(PermissionError, match="reviewer"):
            execute_tool(
                agent_name="reviewer",
                tool_name="write_file",
                args={"path": "test.txt", "content": "data"},
                registry=registry_with_exec_tool,
                permission_layer=perms,
            )
        # The mock fn should never have been called
        write_file_spec.fn.assert_not_called()

    def test_execute_tool_allowed(self, perms, registry_with_exec_tool, mock_run_python):
        """
        Coder calling run_python should succeed and return the result.
        """
        result = execute_tool(
            agent_name="coder",
            tool_name="run_python",
            args={"code": "print('hello')"},
            registry=registry_with_exec_tool,
            permission_layer=perms,
        )
        assert result == "executed successfully"
        mock_run_python.assert_called_once_with(code="print('hello')")