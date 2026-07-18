import os
import sys

os.environ["CACHE_ENABLED"] = "false"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.fact_checker import fact_checker_node

state = {
    "research_output": {
        "findings": [
            "Japan's population is approximately 125 million.",
            "Tokyo is the capital of Japan.",
            "The currency of Japan is the Yen."
        ]
    }
}

print("Running fact_checker_node directly...")
result = fact_checker_node(state)
print("Result:")
import pprint
pprint.pprint(result)
