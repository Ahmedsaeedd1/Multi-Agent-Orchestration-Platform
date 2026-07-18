#!/usr/bin/env python3
"""
Phase 0.5 — Initialize Qdrant Collections

Creates four collections:
  - task_history
  - domain_knowledge
  - agent_learnings
  - cache_responses

All with vector_size from verify_embeddings.py output and Cosine distance.
Then upserts one dummy point per collection and queries it back.
"""

import os
import sys
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PASS = 0
FAIL = 1

COLLECTIONS = [
    "task_history",
    "domain_knowledge",
    "agent_learnings",
    "cache_responses",
]


def get_vector_size() -> int:
    """Run verify_embeddings.py inline to get the actual dimension."""
    from openai import OpenAI

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = "nomic-embed-text"
    client = OpenAI(base_url=ollama_url, api_key="ollama")
    resp = client.embeddings.create(model=model, input="vector size probe")
    return len(resp.data[0].embedding)


def main() -> int:
    print("=" * 60)
    print("Phase 0.5 — Qdrant Collection Initialization")
    print("=" * 60)
    print()

    # Connect
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    print(f"Connected to Qdrant at {qdrant_url}")
    print()

    # Get vector size from embedding model
    try:
        vector_size = get_vector_size()
        print(f"Vector size from embedding model: {vector_size}")
    except Exception as e:
        print(f"FAIL — could not get vector size from embedding model: {e}")
        return FAIL
    print()

    # Create collections
    for coll in COLLECTIONS:
        try:
            # Recreate: delete if exists, then create fresh
            try:
                client.delete_collection(coll)
                print(f"  Deleted existing collection '{coll}'")
            except Exception:
                pass  # Didn't exist

            client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"  Created collection '{coll}' (size={vector_size}, distance=Cosine)")
        except Exception as e:
            print(f"  FAIL — could not create collection '{coll}': {e}")
            return FAIL

    print()

    # Upsert one dummy point per collection and query it back
    dummy_vector = [0.01] * vector_size
    all_ok = True
    for coll in COLLECTIONS:
        try:
            point_id = 1
            client.upsert(
                collection_name=coll,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=dummy_vector,
                        payload={"text": f"dummy entry for {coll}", "source": "phase0_init"},
                    )
                ],
            )
            # Query back
            results = client.search(
                collection_name=coll,
                query_vector=dummy_vector,
                limit=1,
            )
            if results and results[0].score > 0.001:
                print(f"  {coll:<25} score={results[0].score:.6f}  PASS")
            else:
                print(f"  {coll:<25} score={results[0].score if results else 0:.6f}  FAIL (no hit)")
                all_ok = False
        except Exception as e:
            print(f"  {coll:<25} FAIL — {e}")
            all_ok = False

    print()
    if all_ok:
        print("All collections created, populated, and verified.")
        return PASS
    else:
        print("Some checks failed. See above.")
        return FAIL


if __name__ == "__main__":
    sys.exit(main())