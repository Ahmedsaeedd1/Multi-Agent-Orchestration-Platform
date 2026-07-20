"""
Sandboxed Python execution for agents: ``run_python``.

Network access mitigation uses two layers of defence:

1. **Import guard** — ``builtins.__import__`` is replaced with a wrapper
   that blocks any network-related module by matching its top-level name
   against a blocklist (catches ``urllib.request``, ``http.server``, etc.).

2. **Runtime probe (placeholder)** — a reachability check runs before user
   code executes. NOTE: this subprocess is not yet wrapped in real OS-level
   network isolation (no ``docker --network none`` / ``unshare --net``),
   so the probe does not yet raise on successful connectivity. Once real
   isolation wraps subprocess.run, flip the probe to raise.

Windows note
------------
safe_env preserves the environment variables Python needs to locate
its DLLs and stdlib on Windows (PYTHONPATH, PYTHONHOME, LOCALAPPDATA,
APPDATA, USERPROFILE). Without these, subprocess.run fails to import
stdlib modules even when python.exe is on PATH.
"""

import subprocess
import sys
import os
import textwrap
import logging

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 5_000
_TIMEOUT_SECONDS = 10

_SANDBOX_PREAMBLE = textwrap.dedent("""\
def _install_sandbox():
    import builtins, importlib, importlib.util

    _importlib_util = importlib.util
    _socket = __import__("socket")
    _original_import = builtins.__import__

    _BLOCKED_TOP_LEVELS = frozenset({
        'socket', 'http', 'urllib', 'requests', 'httpx', 'ssl',
        'ftplib', 'smtplib', 'telnetlib', 'poplib', 'imaplib',
    })

    def _safe_import(name, *args, **kwargs):
        top_level = name.split('.')[0]
        if top_level in _BLOCKED_TOP_LEVELS:
            raise ImportError(
                f"Network module '{name}' is disabled in sandbox"
            )
        return _original_import(name, *args, **kwargs)

    builtins.__import__ = _safe_import

    def _check_network():
        try:
            spec = _importlib_util.find_spec("netifaces")
            if spec is not None:
                _netifaces = _original_import("netifaces")
                interfaces = _netifaces.interfaces()
                for iface in interfaces:
                    addrs = _netifaces.ifaddresses(iface).get(_netifaces.AF_INET, [])
                    for a in addrs:
                        ip = a.get("addr", "")
                        if ip and not ip.startswith("127."):
                            raise RuntimeError(
                                f"Sandbox network check failed: interface "
                                f"{iface} has non-loopback IP {ip}"
                            )
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(2)
            try:
                s.connect(("1.1.1.1", 53))
                s.close()
                # Placeholder: not raising until real OS-level isolation
                # (docker --network none / unshare --net) wraps subprocess.run.
                # Once isolated, replace this comment with:
                # raise RuntimeError("Sandbox: unexpected network connectivity")
            except (_socket.timeout, OSError):
                pass
            finally:
                s.close()
        except Exception:
            raise

    _check_network()

_install_sandbox()
del _install_sandbox
""")

# Environment variables required for Python to function correctly on Windows.
# Without PYTHONHOME/PYTHONPATH the subprocess cannot find stdlib modules.
# Without LOCALAPPDATA/APPDATA some packages (e.g. pip, certifi) fail to
# locate their data directories.
_WINDOWS_REQUIRED_VARS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "LOCALAPPDATA",
    "APPDATA",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "COMSPEC",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
)

_POSIX_REQUIRED_VARS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
)


def _build_safe_env() -> dict:
    """
    Build a minimal environment for the sandboxed subprocess.

    Includes PATH (so python.exe is findable) plus the platform-specific
    vars Python needs to import its stdlib, but strips all API keys and
    secrets from the parent environment.
    """
    safe_env = {}

    # Always include PATH
    if "PATH" in os.environ:
        safe_env["PATH"] = os.environ["PATH"]

    platform_vars = _WINDOWS_REQUIRED_VARS if os.name == "nt" else _POSIX_REQUIRED_VARS
    for k in platform_vars:
        if k in os.environ:
            safe_env[k] = os.environ[k]

    return safe_env


def run_python(code: str) -> dict:
    """
    Execute Python code in a sandboxed subprocess with a 10-second timeout.
    Network-related imports are disabled via an import guard.

    Returns a structured dictionary with execution results.
    """
    full_code = _SANDBOX_PREAMBLE + "\n" + code

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            env=_build_safe_env(),
        )
        output = result.stdout + result.stderr
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n\n... (truncated)"
        if not output:
            output = "(no output)"
            
        if result.returncode == 0:
            return {"success": True, "output": output, "error_type": None}
        else:
            return {"success": False, "output": output, "error_type": "ExecutionError"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"TimeoutError: execution exceeded {_TIMEOUT_SECONDS}s", "error_type": "TimeoutError"}
    except Exception as e:
        return {"success": False, "output": str(e), "error_type": e.__class__.__name__}