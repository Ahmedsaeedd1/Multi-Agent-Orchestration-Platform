from graph import app

def test_non_code_task():
    print("Testing pure research task...")
    
    state = {
        "run_id": "e2e-non-code-1",
        "user_id": "test-user",
        "task": "What is the capital of France?",
        "step_log": [],
        "review_cycle_count": 0,
    }
    
    final_state = app.invoke(state)
    
    step_nodes = [s["node"] for s in final_state.get("step_log", [])]
    print(step_nodes)
    
    assert "code_evaluator" not in step_nodes, "code_evaluator ran on a non-code task!"
    print("\nSUCCESS: Non-code E2E passed!")

if __name__ == "__main__":
    test_non_code_task()
