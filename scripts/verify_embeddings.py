#!/usr/bin/env python3
"""
Phase 0.4 — Verify Embedding Pipeline

Embeds a test string using nomic-embed-text via Ollama's OpenAI-compatible
/v1/embeddings endpoint. Prints the returned vector dimension.

Expected dimension for nomic-embed-text: 768.
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PASS = 0
FAIL = 1


def main() -> int:
    print("=" * 60)
    print("Phase 0.4 — Embedding Verification")
    print("=" * 60)
    print()

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = "nomic-embed-text"
    test_string = "The quick brown fox jumps over the lazy dog."

    print(f"Ollama URL: {ollama_url}")
    print(f"Model:      {model}")
    print(f"Test text:  {test_string!r}")
    print()

    try:
        client = OpenAI(base_url=ollama_url, api_key="ollama")
        resp = client.embeddings.create(model=model, input=test_string)
        vector = resp.data[0].embedding
        dim = len(vector)
        print(f"Vector dimension: {dim}")
        print(f"First 5 values:    {vector[:5]}")
        print(f"Min value:         {min(vector):.6f}")
        print(f"Max value:         {max(vector):.6f}")
        print()

        if dim == 768:
            print("PASS — dimension matches nomic-embed-text expected size (768).")
            return PASS
        else:
            print(f"WARNING — dimension {dim} differs from expected 768.")
            print(f"This value ({dim}) MUST be used as vector_size in Qdrant collections.")
            return PASS  # Not a hard fail; we use whatever dim is returned

    except Exception as e:
        print(f"FAIL — could not get embedding: {e}")
        return FAIL


if __name__ == "__main__":
    sys.exit(main())