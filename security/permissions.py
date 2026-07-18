"""
Phase 7 — Permission / Security Layer

Enforces least-privilege / least-agency at the tool-execution boundary.
Loads agent → permission grants from ``config/permissions.yaml``.
"""

import logging
import os

import yaml

logger = logging.getLogger(__name__)


class PermissionLayer:
    """
    Reads the agent → permissions mapping from YAML and provides
    ``allows()`` and ``enforce()`` checks.

    Usage::

        perms = PermissionLayer()
        perms.enforce("reviewer", some_tool_spec)  # may raise PermissionError
    """

    def __init__(self, config_path: str = "config/permissions.yaml"):
        config_full_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        with open(config_full_path) as f:
            self.grants: dict[str, list[str]] = yaml.safe_load(f)

    def allows(self, agent_name: str, permission: str) -> bool:
        """
        Return ``True`` if *agent_name* has been granted *permission*.
        """
        return permission in self.grants.get(agent_name, [])

    def enforce(self, agent_name: str, required_permission: str) -> None:
        """
        Raise ``PermissionError`` if *agent_name* has not been granted
        *required_permission*.
        """
        if not self.allows(agent_name, required_permission):
            raise PermissionError(
                f"Agent '{agent_name}' is not authorized for "
                f"'{required_permission}'"
            )


# ---------------------------------------------------------------------------
# Arg normalisation — handle common LLM aliasing mistakes
# ---------------------------------------------------------------------------

_ARG_ALIASES: dict[str, dict[str, str]] = {
    # tool_name -> {alias -> canonical_param_name}
    "run_python": {
        "script": "code",
        "python_code": "code",
        "source": "code",
        "python": "code",
        "program": "code",
    },
    "web_search": {
        "q": "query",
        "search_query": "query",
        "term": "query",
    },
    "web_fetch": {
        "link": "url",
        "uri": "url",
        "href": "url",
    },
}


def _normalize_args(tool_name: str, args: dict) -> dict:
    """
    Rename any aliased keys in *args* to the canonical parameter name
    expected by the tool function.

    For example, if the LLM calls run_python with ``{"script": "..."}``
    instead of ``{"code": "..."}``, this returns ``{"code": "..."}``.
    """
    aliases = _ARG_ALIASES.get(tool_name, {})
    if not aliases:
        return args
    normalized = {}
    for k, v in args.items():
        canonical = aliases.get(k, k)  # remap if alias known, else keep
        normalized[canonical] = v
    return normalized



def execute_tool(
    agent_name: str,
    tool_name: str,
    args: dict,
    registry,
    permission_layer: PermissionLayer,
) -> str:
    """
    Look up *tool_name* in *registry*, verify *agent_name* has the
    required permission, execute the tool function, and return its
    result.

    Raises
    ------
    KeyError
        If *tool_name* is not registered.
    PermissionError
        If *agent_name* is not authorised for the tool's permission.
    """
    spec = registry.get(tool_name)

    # Enforce BEFORE execution — defence in depth
    permission_layer.enforce(agent_name, spec.required_permission)

    # Log the call
    args_str = str(args)
    if len(args_str) > 200:
        args_str = args_str[:200] + "..."
    logger.info(
        "Tool call — agent=%s tool=%s args=%s", agent_name, tool_name, args_str
    )

    # Normalise arg key aliases (e.g. LLM sends "script" instead of "code")
    args = _normalize_args(tool_name, args)

    # Execute
    result = spec.fn(**args)

    result_str = str(result)
    if len(result_str) > 200:
        result_str = result_str[:200] + "..."
    logger.info(
        "Tool result — agent=%s tool=%s result=%s", agent_name, tool_name, result_str
    )

    return result