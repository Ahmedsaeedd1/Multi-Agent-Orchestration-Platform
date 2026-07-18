#!/usr/bin/env python3
"""
Phase 0.2 — Verify LLM Provider Availability
Tests all providers across the free stack.
"""

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PASS = 0
FAIL = 1
results: list[dict] = []
REQUEST_TIMEOUT = 45


def try_model(label: str, base_url: str, api_key: str, model: str,
              extra_headers: dict | None = None) -> dict:
    start = time.monotonic()
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",
            timeout=REQUEST_TIMEOUT,
            default_headers=extra_headers or {},
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Respond with the word OK only."}],
            max_tokens=5,
            temperature=0.0,
            stop=[".", "\n"],
        )
        elapsed = time.monotonic() - start
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Empty response — likely rate-limited or queued")
        return {
            "label": label,
            "model": model,
            "status": "OK",
            "latency_s": round(elapsed, 2),
            "response": content.strip()[:40],
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "label": label,
            "model": model,
            "status": "FAIL",
            "latency_s": round(elapsed, 2),
            "error": str(e),
        }


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main() -> int:
    print("=" * 60)
    print("Phase 0.2 — Provider Verification")
    print("=" * 60)

    # ── 1. GROQ — permanently free ──────────────────────────────
    section("Groq (permanently free)")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("  WARNING: GROQ_API_KEY is blank")

    groq_models = [
        "llama-3.3-70b-versatile",   # orchestrator, reviewer
        "llama-3.1-8b-instant",      # researcher, aggregator
    ]
    for model in groq_models:
        print(f"  → {model} ", end="", flush=True)
        r = try_model("Groq", "https://api.groq.com/openai/v1", groq_key, model)
        results.append(r)
        print(f"{r['status']} ({r['latency_s']}s)")

    # ── 2. OPENROUTER — free models ──────────────────────────────
    section("OpenRouter (free tier — :free models)")
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    if not or_key:
        print("  WARNING: OPENROUTER_API_KEY is blank")

    or_headers = {
        "HTTP-Referer": "https://github.com/multi-agent-orchestrator",
        "X-Title": "multi-agent-orchestrator",
    }
    or_models = [
    #    ("openai/gpt-oss-120b:free",                  "strong general — orchestrator fallback"),
    #    ("qwen/qwen3-coder:free",                      "best free coder, 1M ctx"),
    #    ("qwen/qwen3-next-80b-a3b-instruct:free",      "planner/analyst fallback"),
        ("google/gemma-4-26b-a4b-it:free",             "lightweight multimodal fallback"),
    ]
    for model, role in or_models:
        print(f"  → {model} ({role}) ", end="", flush=True)
        r = try_model("OpenRouter", "https://openrouter.ai/api/v1",
                      or_key, model, extra_headers=or_headers)
        results.append(r)
        print(f"{r['status']} ({r['latency_s']}s)")

    # ── 3. DEEPSEEK — 5M token trial (30 days) ──────────────────
 

    # ── 4. Z.AI (GLM) — permanently free flash model ─────────────
    section("Z.ai / GLM (glm-4.7-flash permanently free)")
    zai_key = os.getenv("ZAI_API_KEY", "")
    if not zai_key:
        print("  WARNING: ZAI_API_KEY blank — sign up at z.ai/chat → API Keys")

    zai_models = [
        "glm-4.7-flash",   # permanently free, ~1000 req/day
    ]
    for model in zai_models:
        print(f"  → {model} ", end="", flush=True)
        r = try_model("Z.ai", "https://api.z.ai/api/paas/v4",
                      zai_key, model)
        results.append(r)
        print(f"{r['status']} ({r['latency_s']}s)")

    # ── 5. HUGGINGFACE — Inference Providers router (free tier) ──
    section("HuggingFace Inference Providers (router.huggingface.co)")
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("  WARNING: HF_TOKEN blank — sign up at huggingface.co, no card needed")
    else:
        print("  NOTE: token must have 'Make calls to Inference Providers' permission")
        print("        enabled at https://huggingface.co/settings/tokens")
 
    hf_base_url = "https://router.huggingface.co/v1"
    hf_models = [
        ("Qwen/Qwen3-8B:featherless-ai", "general fallback"),
        ("deepseek-ai/DeepSeek-R1:novita",       "reasoning fallback — Novita free tier"),
    ]
    for model, role in hf_models:
        print(f"  → {model} ({role}) ", end="", flush=True)
        r = try_model("HuggingFace", hf_base_url, hf_token, model)
        results.append(r)
        print(f"{r['status']} ({r['latency_s']}s)")
 
    # ── Results table ────────────────────────────────────────────
    print()
    print(f"\n{'='*90}")
    print(f"{'Provider':<15} {'Model':<45} {'Status':<8} {'Latency'}")
    print(f"{'─'*90}")
 
    exit_code = PASS
    for r in results:
        print(f"{r['label']:<15} {r['model']:<45} {r['status']:<8} {r['latency_s']}s")
        if r["status"] == "FAIL":
            print(f"  └─ Error: {r.get('error', 'unknown')[:100]}")
            exit_code = FAIL
        else:
            print(f"  └─ Response: {r.get('response', '')}")

    # ── Results table ────────────────────────────────────────────
    print()
    print(f"\n{'='*90}")
    print(f"{'Provider':<15} {'Model':<45} {'Status':<8} {'Latency'}")
    print(f"{'─'*90}")

    exit_code = PASS
    for r in results:
        print(f"{r['label']:<15} {r['model']:<45} {r['status']:<8} {r['latency_s']}s")
        if r["status"] == "FAIL":
            print(f"  └─ Error: {r.get('error', 'unknown')[:100]}")
            exit_code = FAIL
        else:
            print(f"  └─ Response: {r.get('response', '')}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "OK")

    # Groq is the only hard requirement — everything else is a fallback
    groq_passed = sum(1 for r in results
                      if r["label"] == "Groq" and r["status"] == "OK")
    groq_total = sum(1 for r in results if r["label"] == "Groq")

    print(f"\nResults: {passed}/{total} models OK")
    print(f"Groq (required): {groq_passed}/{groq_total}")

    if groq_passed < groq_total:
        summary = f"FAIL — Groq not fully passing ({groq_passed}/{groq_total}). Fix keys."
        print(summary)
        return FAIL
    elif passed < total:
        summary = (f"WARN — {passed}/{total} models OK. "
                   f"Groq all passing. Other providers rate-limited or key missing — non-blocking.")
        print(summary)
        return PASS   # Groq is primary — other failures are non-blocking
    else:
        summary = f"OK — {passed}/{total} models responded across all providers"
        print(summary)
        return PASS


if __name__ == "__main__":
    sys.exit(main())