# Multi-Agent Orchestrator — Phased Task Plan

Source docs: `multi_agent_production_architecture.md`, `docker-compose.yml`, `.env`
Purpose: sequential task list for an autonomous coding agent (Cline / Antigravity). Each phase has a **gate** — do not start the next phase until the gate passes.

Conventions:
- `[ ]` = task the agent should execute and check off
- **Verify:** = concrete pass/fail check, not "looks right"
- Agent should commit after each phase with message `phase-N: <summary>`

---

## Phase 0 — Infrastructure & Model Availability ✅ COMPLETED

Verified passing models (from `scripts/verify_providers.py`):

| Provider | Model | Role | Latency |
|---|---|---|---|
| Groq | llama-3.3-70b-versatile | orchestrator / reviewer | 1.09s |
| Groq | llama-3.1-8b-instant | researcher / aggregator | 0.34s |
| OpenRouter | google/gemma-4-26b-a4b-it:free | lightweight fallback | 1.12s |
| Z.ai | glm-4.7-flash | general fallback | 11.67s |
| HuggingFace | Qwen/Qwen3-8B:featherless-ai | planner / coder fallback | 3.67s |
| HuggingFace | deepseek-ai/DeepSeek-R1:novita | reasoning (planner / analyst) | 0.98s |

Redis, Qdrant, and `nomic-embed-text` embedding pipeline all confirmed healthy.

**Do not re-run Phase 0 — proceed to Phase 1. Re-run `verify_providers.py` only at the final Phase 13 gate.**

---

## Phase 1 — Repo scaffolding & config

- [ ] Create the directory structure from the architecture doc's "Suggested repo structure" (config/, agents/, memory/, cache/, tools/, security/, evals/, router.py, graph.py, app.py)
- [ ] `config/agents.yaml` — use the confirmed model assignments below (do not copy Section 1.1 verbatim — those used models that failed Phase 0):

  ```yaml
  orchestrator:
    primary:   { provider: groq,        model: llama-3.3-70b-versatile }
    fallback:  [{ provider: openrouter, model: google/gemma-4-26b-a4b-it:free }]
    temperature: 0.2

  planner:
    primary:   { provider: huggingface, model: deepseek-ai/DeepSeek-R1:novita }
    fallback:  [{ provider: huggingface, model: Qwen/Qwen3-8B:featherless-ai }]
    temperature: 0.3

  researcher:
    primary:   { provider: groq,        model: llama-3.1-8b-instant }
    fallback:  [{ provider: openrouter, model: google/gemma-4-26b-a4b-it:free }]
    temperature: 0.4
    tools: [web_search, web_fetch, memory_write]

  coder:
    primary:   { provider: huggingface, model: Qwen/Qwen3-8B:featherless-ai }
    fallback:  [{ provider: zai,        model: glm-4.7-flash }]
    temperature: 0.1
    tools: [run_python, read_file, write_file, memory_write]

  data_analyst:
    primary:   { provider: huggingface, model: deepseek-ai/DeepSeek-R1:novita }
    fallback:  [{ provider: groq,       model: llama-3.1-8b-instant }]
    temperature: 0.2
    tools: [run_python, read_file, memory_write]

  reviewer:
    primary:   { provider: groq,        model: llama-3.3-70b-versatile }
    fallback:  [{ provider: zai,        model: glm-4.7-flash }]
    temperature: 0.0
    tools: [memory_read]

  results_aggregator:
    primary:   { provider: groq,        model: llama-3.1-8b-instant }
    fallback:  [{ provider: openrouter, model: google/gemma-4-26b-a4b-it:free }]
    temperature: 0.3
    tools: [memory_read]
  ```

  > **Note on R1/Qwen3 thinking output:** DeepSeek-R1 and Qwen3-8B return `<think>...</think>` blocks before the actual response. The router must strip everything up to and including the closing `</think>` tag before passing output to the next agent or a Pydantic schema. Add a `strip_thinking()` helper in `router.py` and call it on every response from these two models.
- [ ] `config/permissions.yaml` — copy from Section 5
- [ ] `requirements.txt` / `pyproject.toml`: `langgraph`, `langchain-core`, `openai`, `qdrant-client`, `redis`, `pydantic`, `tenacity`, `pyyaml`, `gradio`, `python-dotenv`, `ragas`, `deepeval`, `huggingface-hub`
  - The HuggingFace and Z.ai providers both use the OpenAI-compatible client pointed at their respective base URLs — no extra SDK needed beyond `openai` and `huggingface-hub`
- [ ] **Verify:** `pip install -r requirements.txt` completes clean in a fresh venv

---

## Phase 2 — Model router with fallback

- [ ] Implement `router.py` per Section 1.2 (`ModelRouter`, `RateLimitOrProviderError`, retry/backoff) with the following confirmed provider client map:

  ```python
  CLIENTS = {
      "groq":         OpenAI(base_url="https://api.groq.com/openai/v1",          api_key=GROQ_KEY),
      "openrouter":   OpenAI(base_url="https://openrouter.ai/api/v1",             api_key=OR_KEY,
                             default_headers={"HTTP-Referer": "https://github.com/multi-agent-orchestrator",
                                              "X-Title": "multi-agent-orchestrator"}),
      "huggingface":  OpenAI(base_url="https://router.huggingface.co/v1",         api_key=HF_TOKEN),
      "zai":          OpenAI(base_url="https://api.z.ai/api/paas/v4",             api_key=ZAI_KEY),
  }
  ```

- [ ] Add `strip_thinking(text: str) -> str` helper that removes `<think>...</think>` blocks — apply it to every response from `deepseek-ai/DeepSeek-R1:novita` and `Qwen/Qwen3-8B:featherless-ai` before returning from `router.call()`. Both models confirmed returning `<think>` prefixes in Phase 0.
- [ ] Unit test: force a fake 429 on the primary provider, confirm it falls through to the fallback
- [ ] Unit test: feed a response with a `<think>` block, confirm `strip_thinking()` returns only the content after `</think>`
- [ ] **Verify:** `pytest tests/test_router.py` green

---

## Phase 3 — Structured I/O contracts

- [ ] Define Pydantic schemas per agent handoff (start with `PlannerOutput` from Section 1.3; add one per agent that hands off state)
- [ ] Implement `call_agent_structured()` with repair-on-failure loop
- [ ] **Verify:** test with a deliberately malformed JSON response (mock the router) and confirm it repairs within `max_repairs` or raises cleanly

---

## Phase 4 — Memory layer

- [ ] `memory/redis_store.py` — short-term state read/write per Section 2.4 key schema
- [ ] `memory/qdrant_store.py` — `write_memory()` / `retrieve_memory()` per Section 2.4, scoped by `user_id`
- [ ] **Verify:** write a memory, retrieve it via semantic search with a paraphrased query, confirm it comes back above a reasonable score threshold

---

## Phase 5 — Caching layer

- [ ] `cache/semantic_cache.py` — two-tier cache per Section 3 (`cache_key()`, `get_cached_or_call()`)
- [ ] Wire in `CACHE_ENABLED`, `CACHE_EXACT_TTL`, `CACHE_SEMANTIC_THRESHOLD` from `.env`
- [ ] **Verify:** call the same message twice — second call should be a Redis hit (check latency drop + no new provider call in logs); call a paraphrased version — should hit Qdrant, not miss entirely
- [ ] **Verify:** kill Redis/Qdrant mid-test, confirm the system still calls the LLM (fail-open, not fail-closed)

---

## Phase 6 — Tool registry

- [ ] `tools/registry.py` — `ToolSpec`, `ToolRegistry` per Section 4
- [ ] `tools/web.py` (web_search, web_fetch), `tools/code_exec.py` (sandboxed run_python), `tools/files.py` (read_file/write_file)
- [ ] Register all tools listed in Section 1.6's table
- [ ] **Verify:** `registry.catalog_for("reviewer", permission_layer)` returns an empty or read-only-only list — confirms scoping works before Phase 7 even runs

---

## Phase 7 — Permission / security layer

- [ ] `security/permissions.py` — `PermissionLayer`, `execute_tool()` per Section 5
- [ ] Enforce at the tool-execution boundary, not just catalog-filtering
- [ ] **Verify:** attempt to call `write_file` as `reviewer` — must raise `PermissionError`, confirming defense-in-depth even if the model "requests" a disallowed tool
- [ ] **Verify:** append every tool call (agent, tool, args, result) to an append-only JSONL audit log

---

## Phase 8 — Agent nodes

Build one agent at a time in this order: `orchestrator` → `planner` → `researcher` → `coder` → `data_analyst` → `reviewer` → `aggregator`.

Quick model reference (already in `agents.yaml` from Phase 1):
- **orchestrator** → Groq llama-3.3-70b-versatile — no tools, structured JSON output only
- **planner** → DeepSeek-R1:novita — no tools, strip `<think>` before schema parse
- **researcher** → Groq llama-3.1-8b-instant — tools: web_search, web_fetch, memory_write
- **coder** → Qwen3-8B:featherless-ai — tools: run_python (sandboxed), read_file, write_file, memory_write; strip `<think>` before returning code
- **data_analyst** → DeepSeek-R1:novita — tools: run_python (sandboxed), read_file, memory_write; strip `<think>`
- **reviewer** → Groq llama-3.3-70b-versatile — tools: memory_read only; returns `approved` or `needs_revision` + feedback string
- **aggregator** → Groq llama-3.1-8b-instant — tools: memory_read only; produces final output

For each agent:
- [ ] Implement the node function using the router (Phase 2), tool catalog scoped by permissions (Phase 6/7), and structured output where it hands off (Phase 3)
- [ ] **Verify:** call the node in isolation with a hand-built `AgentState` fixture, confirm output shape matches its schema
- [ ] **Verify for R1/Qwen3 agents specifically:** no `<think>` content leaks into the `AgentState` or downstream agent input

---

## Phase 9 — LangGraph assembly

- [ ] `graph.py` — `AgentState` TypedDict, fan-out from orchestrator to researcher/coder/data_analyst, join at reviewer, conditional edge back to orchestrator or to aggregator (Section 1.4)
- [ ] Add timeouts (30–60s) per node, `run_id`/`step_id` logging, and cap review cycles at `MAX_REVIEW_CYCLES` from `.env`
- [ ] **Verify:** run one full task end-to-end through `app.stream()`, confirm it terminates (not stuck in orchestrator↔reviewer loop) and hits `END`
- [ ] **Verify:** deliberately make the reviewer always return `needs_revision` in a test build, confirm it stops at the cycle cap instead of looping forever

---

## Phase 10 — Evaluation & traceability

- [ ] Set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` — confirm a trace appears in LangSmith for a real run (no code changes needed, per Section 6)
- [ ] `evals/ragas_eval.py` — faithfulness / context precision-recall / answer relevancy against the Qdrant retrieval from Phase 4
- [ ] `evals/deepeval_tests.py` — at least one `GEval` pytest-style test per critical agent (start with orchestrator task decomposition, Section 6.3)
- [ ] **Verify:** `pytest evals/` runs as part of a CI step (even a simple GitHub Actions workflow) and fails the build on a broken prompt/model swap

---

## Phase 11 — Interface

- [ ] `app.py` — Gradio `ChatInterface` per Section 7, streaming intermediate agent steps
- [ ] **Verify:** launch locally, run a real multi-step query end-to-end through the UI, confirm streamed step visibility (not just a final blob)

---

## Phase 12 — Input/output contract

- [ ] Implement the request/response shape from Section 8.1/8.2 as the actual API surface (wrap `app.py` or add a thin FastAPI layer if you want it callable outside Gradio)
- [ ] **Verify:** a sample request produces a response matching the documented schema, including `agent_trace` with a real LangSmith URL and `metadata` with real token/latency counts (not placeholders)

---

## Phase 13 — Integration pass

- [ ] Run 3–5 varied real tasks end-to-end (mix of research-only, code-only, and full fan-out tasks)
- [ ] Confirm cache hit rates, provider fallback triggers, and audit log are all populated as expected
- [ ] Write a short `README.md`: setup steps (this plan's Phase 0 checks, condensed), how to run, how to add a new agent

**Final gate:** all of Phase 0–12's individual verifications still pass together, not just in isolation — rerun Phase 0's scripts once more at the end to catch anything that broke along the way (expired free-tier model, Qdrant collection drift, etc).

---

## Notes for the executing agent

- Phase 0 is done — start at Phase 1. Do not re-run Phase 0 scripts unless a model returns 401/404 during Phases 1–12, which should be treated as a Phase-0-level blocker (fix `agents.yaml`, don't patch around it with retries).
- **Thinking-tag discipline:** DeepSeek-R1:novita and Qwen3-8B:featherless-ai both emit `<think>...</think>` before their actual response. `strip_thinking()` in the router (Phase 2) is the single point of removal — never strip in individual agent nodes or you'll miss it when a new model is added later.
- **Z.ai latency:** glm-4.7-flash responded at 11.67s in Phase 0 — it is a last-resort fallback only, not a primary. Do not assign it as primary for any agent.
- **OpenRouter gemma-4-26b:** confirmed OK but it is a multimodal lightweight model — suitable as a fallback for simple aggregation or routing tasks, not for complex reasoning or code generation.
- If a free-tier model goes stale mid-build (404 / model-not-found), update `agents.yaml` primary/fallback and re-run `verify_providers.py` before continuing — do not build on a dead primary.
- Least agency (Section 1.6/5) applies to every new tool added later — new tool = new entry in `config/permissions.yaml`, no exceptions.
