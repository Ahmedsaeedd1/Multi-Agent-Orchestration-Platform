from agents.coder import coder_node

state = {
    "task": "Write a Python script that computes and prints the sum of this list: [3, 7, 12, 5, 18]",
    "run_id": "debug",
    "review_cycles": 0,
    "research_notes": [],
}
result = coder_node(state)

print("CODE:\n", result["code"])
print("\nVERIFIED:", result["code_verified"])
print("\nOUTPUT:\n", result["code_exec_output"])