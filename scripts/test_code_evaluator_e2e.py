import json
from pprint import pprint
from graph import app

def test_code_evaluator_e2e():
    print("Testing string reversal task...")
    
    # Run a task that requires code
    state = {
        "run_id": "e2e-code-eval-1",
        "user_id": "test-user",
        "task": "Write a Python function that reverses a string, and show it working",
        "step_log": [],
        "review_cycle_count": 0,
    }
    
    final_state = app.invoke(state)
    
    print("\n--- Step Log ---")
    step_nodes = [s["node"] for s in final_state.get("step_log", [])]
    print(step_nodes)
    
    assert "code_evaluator" in step_nodes, "code_evaluator did not run"
    
    # Check that code_evaluator immediately follows coder
    coder_idx = step_nodes.index("coder")
    assert step_nodes[coder_idx + 1] == "code_evaluator", "code_evaluator did not immediately follow coder"
    
    eval_output = final_state.get("code_eval_output")
    print("\n--- Code Eval Output ---")
    pprint(eval_output)
    
    assert eval_output is not None, "code_eval_output is None"
    assert eval_output["verdict"] in ["pass", "fail"]
    
    print("\nSUCCESS: Code evaluator E2E passed!")

if __name__ == "__main__":
    test_code_evaluator_e2e()
