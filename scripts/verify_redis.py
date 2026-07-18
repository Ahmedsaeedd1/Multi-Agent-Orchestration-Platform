import os
import sys
import time
from dotenv import load_dotenv
import redis as redis_lib

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PASS = 0
FAIL = 1


def main() -> int:
    print("=" * 60)
    print("Phase 0.6 — Redis Schema Smoke Test")
    print("=" * 60)
    print()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    print(f"Redis URL: {redis_url}")

    try:
        r = redis_lib.from_url(redis_url, decode_responses=True)
        r.ping()
        print("Redis connection OK")
    except Exception as e:
        print(f"FAIL — could not connect to Redis: {e}")
        return FAIL
    print()

    # --- Set key with TTL ---
    key = "run:test:state"
    value = "phase0_smoke_test"
    ttl = 5  # seconds

    r.set(key, value, ex=ttl)
    print(f"Set key '{key}' = '{value}' with TTL={ttl}s")

    # --- Read back immediately ---
    read_back = r.get(key)
    if read_back == value:
        print(f"Immediate read-back: '{read_back}'  PASS")
    else:
        print(f"Immediate read-back: '{read_back}' (expected '{value}')  FAIL")
        return FAIL

    # --- Wait for expiry ---
    print(f"Waiting {ttl + 2}s for TTL to expire...")
    time.sleep(ttl + 2)

    after_expiry = r.get(key)
    if after_expiry is None:
        print(f"After expiry: key is gone (None)  PASS")
    else:
        print(f"After expiry: key still exists = '{after_expiry}'  FAIL")
        return FAIL

    # --- Cleanup ---
    r.delete(key)

    print()
    print("Redis smoke test passed.")
    return PASS


if __name__ == "__main__":
    sys.exit(main())