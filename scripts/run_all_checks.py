#!/usr/bin/env python3
"""
Phase 0 — Run All Checks (0.2 through 0.6)

Orchestrates the sub-scripts in order:
  1. verify_providers.py  (0.2)
  2. verify_embeddings.py (0.4)
  3. init_qdrant.py       (0.5)
  4. verify_redis.py      (0.6)

Prints a clear PASS/FAIL summary table at the end.

Note: Step 0.1 (containers up + healthy) is assumed already done before running this.
"""

import subprocess
import sys
import time

SCRIPTS_DIR = __file__.rstrip("run_all_checks.py")

CHECKS = [
    ("0.2 — Provider Verification",       "verify_providers.py"),
    ("0.4 — Embedding Pipeline",          "verify_embeddings.py"),
    ("0.5 — Qdrant Collections",          "init_qdrant.py"),
    ("0.6 — Redis Smoke Test",            "verify_redis.py"),
]

PASS = 0
FAIL = 1


def run_check(description: str, script: str) -> dict:
    """Run a single check script and return its result."""
    print()
    print("=" * 70)
    print(f"  RUNNING: {description}")
    print("=" * 70)
    print()

    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, script],
        cwd=SCRIPTS_DIR,
        capture_output=False,
        text=True,
    )
    elapsed = time.monotonic() - start

    ok = result.returncode == 0
    return {
        "description": description,
        "script": script,
        "ok": ok,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 2),
    }


def print_summary(results: list[dict]) -> None:
    """Print a PASS/FAIL summary table."""
    print()
    print("=" * 70)
    print("  PHASE 0 — SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Check':<40} {'Result':<10} {'Time'}")
    print("-" * 62)

    all_ok = True
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        time_str = f"{r['elapsed_s']}s"
        print(f"{r['description']:<40} {status:<10} {time_str}")
        if not r["ok"]:
            all_ok = False

    print()
    print(f"{'TOTAL':<40} {'ALL PASS' if all_ok else 'SOME FAILED':<10}")
    print()
    if all_ok:
        print("  ✓ All Phase 0 checks passed. Ready to proceed to Phase 1.")
    else:
        print("  ✗ One or more checks failed. Review output above for details.")
    print()


def main() -> int:
    results = []
    overall_ok = True

    for description, script in CHECKS:
        r = run_check(description, script)
        results.append(r)
        if not r["ok"]:
            overall_ok = False

    print_summary(results)
    return PASS if overall_ok else FAIL


if __name__ == "__main__":
    sys.exit(main())