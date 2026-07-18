import sys
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG)

from agents.researcher import researcher_node

state = {
    "task": "What are the top 3 vector databases available in 2025?",
    "run_id": "test_run",
    "assignments": {"task_1": "researcher"},
}

result = researcher_node(state)
print("RESULT:", json.dumps(result, indent=2))
