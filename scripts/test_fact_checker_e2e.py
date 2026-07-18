import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph import app
from tools.registry import build_registry

def test_research_e2e():
    print("Testing Research E2E (Japan population)...")
    initial_state = {
        "task": "Research the current population of Japan and summarize it",
        "run_id": "test_fact_checker_e2e_run",
        "user_id": "tester",
        "review_cycle_count": 0,
        "step_log": [],
        "plan": None,
        "_subtasks": [],
        "_assignments": {},
        "error": None,
        "needs_clarification": False
    }

    try:
        final_state = app.invoke(initial_state)
        step_log = [s["node"] for s in final_state.get("step_log", [])]
        print(f"Step Log: {step_log}")
        
        # Verify fact_checker ran after researcher
        if "researcher" in step_log and "fact_checker" in step_log:
            idx_researcher = step_log.index("researcher")
            idx_fact_checker = step_log.index("fact_checker")
            if idx_fact_checker == idx_researcher + 1:
                print("SUCCESS: fact_checker ran immediately after researcher")
            else:
                print("ERROR: fact_checker did not run immediately after researcher")
        else:
            print("ERROR: Missing researcher or fact_checker in step log")

        fc_out = final_state.get("fact_check_output")
        if fc_out:
            print(f"Fact Check Summary: {fc_out.get('confidence_summary')}")
            print(f"Claims checked: {len(fc_out.get('claims_checked', []))}")
        else:
            print("ERROR: Missing fact_check_output")
            
        print("\n\n")
    except Exception as e:
        print(f"E2E test failed: {e}")

def test_non_research_e2e():
    print("Testing Non-Research E2E (SQL task)...")
    initial_state = {
        "task": "How many tables are in the database and what are their names?",
        "run_id": "test_sql_bypass",
        "user_id": "tester",
        "review_cycle_count": 0,
        "step_log": [],
        "plan": None,
        "_subtasks": [],
        "_assignments": {},
        "error": None,
        "needs_clarification": False
    }

    try:
        final_state = app.invoke(initial_state)
        step_log = [s["node"] for s in final_state.get("step_log", [])]
        print(f"Step Log: {step_log}")
        
        # Verify fact_checker is absent
        if "fact_checker" not in step_log:
            print("SUCCESS: fact_checker skipped for SQL task")
        else:
            print("ERROR: fact_checker ran for SQL task")
        print("\n\n")
    except Exception as e:
        print(f"E2E non-research failed: {e}")

def test_claim_limit():
    from agents.fact_checker import fact_checker_node
    
    print("Testing Claim Limit (8+ obvious claims)...")
    findings = [
        "The population of Earth is 8 billion.",
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The speed of light is 299,792,458 m/s.",
        "Mount Everest is 8,848 meters tall.",
        "The Amazon river is 6,992 km long.",
        "The moon orbits Earth.",
        "Saturn has rings.",
        "Mars is called the Red Planet.",
        "Jupiter is the largest planet."
    ]
    
    state = {
        "research_output": {"findings": findings}
    }
    
    result = fact_checker_node(state)
    fc_out = result.get("fact_check_output", {})
    claims_checked = len(fc_out.get("claims_checked", []))
    print(f"Claims checked: {claims_checked}")
    if claims_checked <= 5:
        print("SUCCESS: Claim limit enforced (<= 5)")
    else:
        print("ERROR: Claim limit exceeded")
    print("\n\n")

if __name__ == "__main__":
    print("Starting Fact Checker Verification")
    test_claim_limit()
    test_non_research_e2e()
    test_research_e2e()
