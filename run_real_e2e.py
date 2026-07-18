"""
Real end-to-end run — NO MOCKS.

This is deliberately not a pytest test. It's a standalone script meant
to be run once, by hand, to answer one question: does the full pipeline
actually work when every agent makes a real call through the real
router to a real (free-tier) provider?

This matters because the structured_call.py bug found earlier in this
project meant every agent using call_agent_structured (coder, reviewer,
aggregator, researcher's 2nd phase, orchestrator, planner) was silently
crashing on every real call and falling into each agent's own
exception-handler fallback — e.g. "# Code generation failed",
"Analysis failed", forced needs_revision. All of that was invisible
in the test suite because every test up to that point mocked
call_agent_structured directly, so nothing ever exercised the real
path until the coder eval did.

This script is the first true "did we actually fix it" check at the
full-pipeline level, not just the unit level.

Run with:
    python run_real_e2e.py

Read the output carefully — a run that "completes without crashing"
is not the same as a run that actually worked. Check for:
  1. Did every step_log entry NOT correspond to an exception fallback?
  2. Is code_verified True (if a coder step ran)?
  3. Does final_output actually look like a real answer, not a
     generic fallback string?
  4. Did review_cycle_count stay at 1 (approved first try) or hit
     MAX_REVIEW_CYCLES (something kept getting rejected)?
"""

import copy
import json
import logging
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

# ── 1. UTF-8 stdout — prevents emoji (⚠️ ✅) from crashing on Windows cp1252 ──
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 2. Suppress known third-party warnings before any imports fire them ────────
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# ── 3. Redirect ALL logging to a file — keeps the terminal output clean ───────
_LOG_FILE = Path(__file__).parent / f"e2e_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
logging.getLogger().handlers = [h for h in logging.getLogger().handlers
                                 if isinstance(h, logging.FileHandler)]
print(f"  (Detailed logs → {_LOG_FILE.name})\n")

from graph import app, MAX_REVIEW_CYCLES

# ---------------------------------------------------------------------------
# A real, concrete task — deliberately simple so a first real run is
# easy to sanity-check by eye. Something that should map to a single
# specialist (coder) so this is a small, fast, cheap first probe.
# ---------------------------------------------------------------------------

TASK = "Write a Python function that checks if a number is a palindrome, and test it on 12321."

initial_state = {
    "run_id": str(uuid.uuid4()),
    "user_id": "e2e-smoke-test",
    "task": TASK,
    "plan": None,
    "plan_reasoning": "",
    "research_output": None,
    "code_output": None,
    "analysis_output": None,
    "review": None,
    "final_output": None,
    "review_cycle_count": 0,
    "step_log": [],
    "error": None,
    "_subtasks": [],
    "_assignments": {},
}

print("=" * 70)
print(f"TASK: {TASK}")
print("=" * 70)
print()

final_state = app.invoke(copy.deepcopy(initial_state))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP LOG (order agents actually ran in)")
print("=" * 70)
for entry in final_state.get("step_log", []):
    print(f"  {entry['timestamp']}  {entry['node']}")

print("\n" + "=" * 70)
print("PLAN (orchestrator + planner output)")
print("=" * 70)
print(json.dumps(final_state.get("plan"), indent=2))
print(f"\nplan_reasoning: {final_state.get('plan_reasoning')!r}")

if final_state.get("code_output"):
    print("\n" + "=" * 70)
    print("CODE OUTPUT")
    print("=" * 70)
    print(final_state["code_output"].get("code", ""))
    print(f"\ncode_verified: {final_state['code_output'].get('code_verified')}")
    print(f"code_exec_output: {final_state['code_output'].get('code_exec_output')}")

if final_state.get("research_output"):
    print("\n" + "=" * 70)
    print("RESEARCH OUTPUT")
    print("=" * 70)
    print(json.dumps(final_state["research_output"], indent=2))

if final_state.get("analysis_output"):
    print("\n" + "=" * 70)
    print("ANALYSIS OUTPUT")
    print("=" * 70)
    print(json.dumps(final_state["analysis_output"], indent=2))

print("\n" + "=" * 70)
print("REVIEW")
print("=" * 70)
print(json.dumps(final_state.get("review"), indent=2))
print(f"review_cycle_count: {final_state.get('review_cycle_count')} (cap: {MAX_REVIEW_CYCLES})")

print("\n" + "=" * 70)
print("ERROR STATE")
print("=" * 70)
print(final_state.get("error"))

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)
final_output = final_state.get("final_output") or {}
print(final_output.get("final_answer", "<<< NO FINAL ANSWER >>>"))

# ---------------------------------------------------------------------------
# Automated sanity flags — things that indicate a fallback fired
# somewhere, even if the run didn't crash outright.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)

warnings = []

if final_state.get("error"):
    warnings.append(f"state['error'] is set: {final_state['error']}")

code_output = final_state.get("code_output") or {}
if code_output and code_output.get("code") == "# Code generation failed":
    warnings.append("Coder hit its top-level exception fallback (structured call likely still broken)")
if code_output and code_output.get("code_verified") is False:
    warnings.append(f"code_verified is False — code did not execute cleanly: {code_output.get('code_exec_output')}")

analysis_output = final_state.get("analysis_output") or {}
if analysis_output.get("summary") == "Analysis failed":
    warnings.append("Data analyst hit its exception fallback")

review = final_state.get("review") or {}
if review.get("verdict") == "error":
    warnings.append("Reviewer was bypassed due to an upstream error (reviewer_wrapper's error-bypass fired)")

final_answer = final_output.get("final_answer", "")
if final_answer.startswith("⚠️ Aggregation failed"):
    warnings.append("Aggregator hit its exception fallback — final answer is raw unformatted content")
if final_answer == "<<< NO FINAL ANSWER >>>" or not final_answer:
    warnings.append("No final answer was produced at all")

if final_state.get("review_cycle_count", 0) >= MAX_REVIEW_CYCLES:
    warnings.append(
        f"review_cycle_count hit the cap ({MAX_REVIEW_CYCLES}) — reviewer "
        f"kept rejecting output, or reviewer itself kept failing"
    )

if warnings:
    print("⚠️  Issues detected:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("✅ No fallback/exception indicators detected — run appears genuinely healthy.")

print("=" * 70)