import os
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import orchestrator_node

def test_orch():
    res = orchestrator_node({"task": "How many tables are in the database and what are their names?"})
    print(res)

if __name__ == "__main__":
    test_orch()
