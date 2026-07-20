import sys
import copy
import uuid
import json
import traceback

from graph import app

def run_test(prompt: str):
    initial_state = {
        "run_id": str(uuid.uuid4()),
        "user_id": "test",
        "task": prompt,
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
    
    try:
        final_state = app.invoke(copy.deepcopy(initial_state))
        
        result = {
            "task": prompt,
            "step_log": [
                f"{entry['timestamp']} {entry['agent']}" for entry in final_state.get("step_log", [])
            ],
            "plan": {
                "subtasks": final_state.get("_subtasks"),
                "assignments": final_state.get("_assignments")
            },
            "error": final_state.get("error"),
            "code_output": final_state.get("code_output"),
            "review": final_state.get("review"),
            "final_output": final_state.get("final_output", ""),
            "review_cycle_count": final_state.get("review_cycle_count")
        }
        
        # also snag sql output if any
        if final_state.get("sql_output"):
            result["sql_output"] = final_state.get("sql_output")
            
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"error_running_graph": str(e), "traceback": traceback.format_exc()}))

if __name__ == "__main__":
    prompt = sys.argv[1]
    run_test(prompt)
