# Multi-Agent Orchestration System — Production Architecture (Free LLMs Only)

This document goes one level deeper than the initial plan: how to implement agents at a production standard (including exactly which tools each one should have), how to structure shared memory and which vector database to use, how to add a caching layer, a centralized tool registry, and a permission/security layer, how to add evaluation and traceability, what interface to build, and the exact input/output contract for the system.

---

## 1. Implementing the agents at production level

The single biggest mistake in DIY multi-agent projects is hardcoding each agent's model, prompt, and error handling directly into its function. That works for a demo and breaks the moment a free model gets rate-limited or an OpenRouter `:free` model rotates out. Build these four layers instead.

### 1.1 Config-driven agent definitions

Keep every agent's model, provider, and parameters in one YAML file — never in code.

```yaml
# config/agents.yaml
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

Swapping a model, adding a fallback, or moving an agent to local-only is a one-line change, not a code change. This matters a lot with free tiers since availability shifts.

### 1.2 A model router with automatic fallback

Every agent calls the same router instead of a raw client. This is the single highest-leverage piece of infrastructure in a free-model system.

```python
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import yaml

CLIENTS = {
    "groq": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY),
    "openrouter": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY),
    "ollama": OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
}

class RateLimitOrProviderError(Exception): pass

class ModelRouter:
    def __init__(self, config_path="config/agents.yaml"):
        self.config = yaml.safe_load(open(config_path))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
           retry=retry_if_exception_type(RateLimitOrProviderError))
    def _call(self, provider, model, messages, **kw):
        try:
            return CLIENTS[provider].chat.completions.create(model=model, messages=messages, **kw)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                raise RateLimitOrProviderError(str(e))
            raise

    def call(self, agent_name, messages, **kw):
        cfg = self.config[agent_name]
        chain = [cfg["primary"]] + cfg.get("fallback", [])
        last_err = None
        for target in chain:
            try:
                return self._call(target["provider"], target["model"], messages,
                                   temperature=cfg.get("temperature", 0.3), **kw)
            except Exception as e:
                last_err = e
                continue  # move to next provider in the chain
        raise RuntimeError(f"All providers failed for {agent_name}: {last_err}")
```

This gives you exactly the resilience a free-tier stack needs: Groq rate-limited → OpenRouter → local Ollama, automatically, per agent, without touching orchestration logic.

### 1.3 Structured I/O contracts per agent

Free models are less reliable than GPT-4/Claude at strict JSON. Don't trust raw output — validate it and repair on failure.

```python
from pydantic import BaseModel, ValidationError

class PlannerOutput(BaseModel):
    subtasks: list[str]
    assigned_agent: dict[str, str]  # subtask -> agent name
    reasoning: str

def call_agent_structured(router, agent_name, messages, schema: type[BaseModel], max_repairs=2):
    for attempt in range(max_repairs + 1):
        resp = router.call(agent_name, messages, response_format={"type": "json_object"})
        try:
            return schema.model_validate_json(resp.choices[0].message.content)
        except ValidationError as e:
            messages.append({"role": "user",
                "content": f"Your last output failed schema validation: {e}. Return valid JSON only."})
    raise RuntimeError(f"{agent_name} failed to produce valid structured output after retries")
```

Every agent that hands off to another agent (orchestrator → specialists, reviewer → orchestrator) should go through a Pydantic schema like this. This is what separates "usually works" from production-grade.

### 1.4 Orchestration and concurrency in LangGraph

Model the graph as a state machine, not a chain. Independent specialists (researcher, data analyst) can run in parallel branches that LangGraph fans out and joins automatically — this cuts latency meaningfully since you're not paying Groq's per-request latency serially five times.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    task: str
    subtasks: list[str]
    research_notes: Annotated[list[str], operator.add]
    code: str
    analysis: str
    review_feedback: str
    final_output: str

graph = StateGraph(AgentState)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)
graph.add_node("data_analyst", data_analyst_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("aggregator", aggregator_node)

graph.set_entry_point("orchestrator")
graph.add_edge("orchestrator", "researcher")   # fan out
graph.add_edge("orchestrator", "coder")
graph.add_edge("orchestrator", "data_analyst")
graph.add_edge(["researcher", "coder", "data_analyst"], "reviewer")  # join

def route_after_review(state: AgentState):
    return "orchestrator" if state["review_feedback"] == "needs_revision" else "aggregator"

graph.add_conditional_edges("reviewer", route_after_review, {"orchestrator": "orchestrator", "aggregator": "aggregator"})
graph.add_edge("aggregator", END)
app = graph.compile()
```

The conditional edge from `reviewer` is your feedback loop from the diagram — implemented as actual control flow, not just a picture.

### 1.5 Guardrails, timeouts, and idempotency

- Wrap every agent node with a hard timeout (30–60s) — a stuck free-tier call shouldn't hang the whole graph.
- Log a `run_id` and `step_id` on every state mutation so a crashed run can resume instead of restarting from scratch.
- Cap iterations on the feedback loop (e.g. max 3 review cycles) to prevent infinite orchestrator↔reviewer ping-pong, which is the most common multi-agent failure mode.

### 1.6 Tools per agent — what each one actually needs

Every extra tool you give an agent is extra blast radius, not extra capability you're likely to use. The governing rule (this shows up again in Section 5) is **least agency**: give each agent the minimum tool set its role requires, nothing more — an agent that can summarize research doesn't also need the ability to delete files, even if it would technically know how to call that tool.

| Agent | Tools it should have | Why (and why not more) |
|---|---|---|
| Orchestrator | *none* — structured planning output only | Its job is delegation, not action. Tool access here only widens the blast radius for no functional benefit |
| Planner | *none* — pure reasoning against task + memory context | Same logic — it decomposes the task, it doesn't act on the world |
| Researcher | `web_search`, `web_fetch`, `memory_write` | Needs to reach the outside world and persist findings; no code execution, no file write |
| Coder | `run_python` (sandboxed), `read_file`, `write_file`, `memory_write` | The only agent that should touch the filesystem or execute arbitrary code |
| Data analyst | `run_python` (sandboxed, for pandas/analysis), `read_file`, `memory_write` | Needs execution for computation but not file write — it shouldn't be creating arbitrary files on disk |
| Reviewer | `memory_read` only | Deliberately read-only. A reviewer that can write is a reviewer that can rubber-stamp its own edits |
| Results aggregator | `memory_read` only | Combines what specialists already produced; no reason to reach further |

Every agent also has implicit access to its own scratch state in short-term memory — that's not a "tool" in the registry sense, it's just how the graph passes state, and it's still gated by the same permission layer as everything else below.

---

## 2. Shared memory: architecture and vector database choice

### 2.1 Two-tier design

Don't put everything in one store. Split by access pattern:

- **Short-term / working memory — Redis.** Per-run state, agent-to-agent messages, task queues. Fast (sub-ms), ephemeral, TTL-based (expire after the run completes or after N hours). This is what agents read/write *during* a single task execution.
- **Long-term / semantic memory — vector database.** Cross-run knowledge: past task outcomes, domain facts, reusable research. This is what makes the system get better over time instead of starting from zero on every run.

### 2.2 Vector database: use Qdrant, not Chroma, for production

Chroma is genuinely the fastest way to prototype (embedded, zero setup) — keep using it while you're building. But for the production version, move to **Qdrant**:

- Rust-based, lowest latency among open-source vector DBs (~4–12ms p50 vs Chroma's ~30–100ms)
- Free and fully self-hostable (Docker, one container, no license)
- Strong metadata filtering (scope retrieval by `user_id`, `session_id`, `agent_name`, `task_type`) — essential once more than one person or one task type shares the store
- Binary quantization can cut memory usage significantly for larger collections, which matters when you're running everything on your own hardware for free
- Clean path: prototype in Chroma → migrate to Qdrant with a one-time re-embed once you have real usage patterns

Skip Pinecone/Weaviate/Milvus for this project specifically — Pinecone is managed-only (not free at scale), Weaviate's main advantage (built-in vectorizers) just calls the same embedding APIs you'd call yourself, and Milvus is built for billion-scale workloads you don't have.

### 2.3 Embedding model (must also be free)

Since OpenAI/Gemini embeddings are off the table, embed locally:

- **nomic-embed-text** (via Ollama) — the standard free choice, good general-purpose quality, runs on modest hardware
- **mxbai-embed-large** — alternative with slightly higher quality, marginally heavier

Both are pulled the same way as your LLMs: `ollama pull nomic-embed-text`.

### 2.4 Collections and Redis schema

```
Qdrant collections:
  task_history      → {task_summary, outcome, agent_path, embedding}  filtered by user_id, date
  domain_knowledge   → {fact, source, embedding}                       filtered by domain, freshness
  agent_learnings    → {failure_pattern, correction, embedding}        filtered by agent_name

Redis keys (TTL-based):
  run:{run_id}:state       → serialized AgentState, TTL = run duration + buffer
  run:{run_id}:messages    → list of inter-agent messages for this run
  queue:tasks               → pending task queue (if you support async/batch runs)
```

```python
import qdrant_client
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

qc = qdrant_client.QdrantClient(url="http://localhost:6333")

def write_memory(collection, text, embedding, metadata):
    qc.upsert(collection, points=[PointStruct(id=uuid4().hex, vector=embedding,
              payload={"text": text, **metadata})])

def retrieve_memory(collection, query_embedding, user_id, top_k=5):
    return qc.search(collection, query_vector=query_embedding, limit=top_k,
        query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]))
```

---

## 3. Caching layer

Free-tier rate limits make caching more than a cost optimization here — it's latency and rate-limit budget, not just money. The pattern nearly every production LLM cache converges on is two layers, and you already have both pieces of infrastructure from Section 2:

- **Layer 1 — exact match (Redis).** Hash key of `agent_name + model + temperature + normalized_messages` → cached response. O(1) lookup, zero embedding cost. This alone typically catches a large share of traffic, since agent frameworks resend near-identical tool descriptions and system prompts on every call.
- **Layer 2 — semantic match (Qdrant).** Embed the query with the same free local embedding model you're already using for long-term memory, search a dedicated `cache_responses` collection, and treat anything above a conservative similarity threshold (~0.90) as a hit. This catches "same question, different words," which exact match misses.

```python
import hashlib, json

def cache_key(agent_name, model, messages, temperature):
    normalized = json.dumps(messages, sort_keys=True)
    raw = f"{agent_name}:{model}:{temperature}:{normalized}"
    return "cache:exact:" + hashlib.sha256(raw.encode()).hexdigest()

def get_cached_or_call(router, agent_name, messages, embed_fn, redis_client, qdrant_client):
    cfg = router.config[agent_name]
    key = cache_key(agent_name, cfg["primary"]["model"], messages, cfg.get("temperature", 0.3))

    if hit := redis_client.get(key):
        return json.loads(hit)                              # Layer 1 hit

    query_embedding = embed_fn(messages[-1]["content"])
    semantic_hits = qdrant_client.search("cache_responses", query_vector=query_embedding, limit=1,
        query_filter=Filter(must=[FieldCondition(key="agent_name", match=MatchValue(value=agent_name))]))
    if semantic_hits and semantic_hits[0].score > 0.90:
        return semantic_hits[0].payload["response"]          # Layer 2 hit

    response = router.call(agent_name, messages)              # cache miss — real call
    redis_client.setex(key, 3600, json.dumps(response))
    qdrant_client.upsert("cache_responses", points=[PointStruct(id=uuid4().hex, vector=query_embedding,
        payload={"agent_name": agent_name, "response": response})])
    return response
```

A few things that matter more than the code above:

- **Scope the cache key correctly.** It must include `agent_name`, `model`, and `temperature`. Skip any of these and you'll serve the Orchestrator's response to the Coder, or a creative 0.8-temperature output to a deterministic 0.0-temperature Reviewer call — both silently wrong.
- **Fail open, not closed.** If Redis or Qdrant is slow or down, call the LLM anyway. A cache should never be a single point of failure for the whole system.
- **Don't cache everything.** Skip caching for the final aggregator's user-facing output (it's often personalized) and for genuinely novel orchestrator decompositions. Cache hardest where variance is low: the researcher's raw tool outputs, repeated planner subtask patterns, coder boilerplate.
- **If you'd rather not hand-roll this**, GPTCache (Zilliz's open-source library) does the same two-layer job out of the box, supports Redis and Qdrant as backends, and wraps your OpenAI-compatible client in a couple of lines — a reasonable shortcut if you don't want to own the cache logic yourself.
- **Track hit rate per agent, not just globally** — a 20% overall hit rate can hide a researcher agent hitting 60% and an orchestrator hitting 2%, which tells you very different things about where the cache is actually earning its keep.

---

## 4. Tool registry

Without a registry, tool definitions get copy-pasted into every agent's prompt, implementations get duplicated across agent files, and there's no single place to answer "what can any agent in this system actually do?" That question needs to have one obvious answer, especially once the permission layer (Section 5) depends on it.

```python
# tool_registry.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable
    schema: dict                  # JSON schema for arguments, given to the model
    required_permission: str      # e.g. "web_access", "code_execution", "file_write"

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def catalog_for(self, agent_name: str, permission_layer) -> list[dict]:
        """Return only the tool schemas this agent is permitted to see."""
        return [t.schema for t in self._tools.values()
                if permission_layer.allows(agent_name, t.required_permission)]

registry = ToolRegistry()
registry.register(ToolSpec("web_search", "Search the web", web_search_fn, WEB_SEARCH_SCHEMA, "web_access"))
registry.register(ToolSpec("run_python", "Execute Python in a sandbox", run_python_fn, RUN_PYTHON_SCHEMA, "code_execution"))
registry.register(ToolSpec("read_file", "Read a project file", read_file_fn, READ_FILE_SCHEMA, "file_read"))
registry.register(ToolSpec("write_file", "Write a project file", write_file_fn, WRITE_FILE_SCHEMA, "file_write"))
registry.register(ToolSpec("memory_search", "Search long-term memory", memory_search_fn, MEMORY_SEARCH_SCHEMA, "memory_read"))
registry.register(ToolSpec("memory_write", "Write to long-term memory", memory_write_fn, MEMORY_WRITE_SCHEMA, "memory_write"))
```

Each agent node calls `registry.catalog_for(agent_name, permission_layer)` to get exactly the tool schemas it's allowed to be offered — never the full list. This is what makes the permission layer actually enforceable rather than just documented in a comment somewhere.

Worth knowing: this is a scaled-down version of what MCP (Model Context Protocol) servers plus an MCP gateway/registry do at infrastructure scale in 2026 — centralized discovery, auth, and audit logging, the pattern enterprises use tools like Kong's MCP Gateway for. For a personal or learning-scale project, the plain Python registry above is the right size. If you later want tools to be independently deployable and reusable across other agent projects, wrapping them as local MCP servers is the natural next step — the registry pattern above maps directly onto that transition.

---

## 5. Permission and security layer

Two principles do the actual design work here — worth naming explicitly rather than treating this as just an access-control checklist:

- **Least privilege** — what data or credentials can this agent's identity access?
- **Least agency** — what is this agent allowed to *decide and do*, even with valid credentials? A researcher agent might legitimately have file-read access, but should never be allowed to decide to delete a file, even if it technically knows how to call that tool. This distinction is what the OWASP Agentic Security Initiative's "Excessive Agency" category is specifically about.

```yaml
# config/permissions.yaml
orchestrator:      []                              # plans only — no tool access at all
planner:            []                              # reasons only
researcher:         [web_access, memory_read, memory_write]
coder:              [code_execution, file_read, file_write, memory_write]
data_analyst:       [code_execution, file_read, memory_write]
reviewer:           [memory_read]                   # read-only — never mutates anything
results_aggregator: [memory_read]
```

```python
class PermissionLayer:
    def __init__(self, config_path="config/permissions.yaml"):
        self.grants = yaml.safe_load(open(config_path))

    def allows(self, agent_name: str, permission: str) -> bool:
        return permission in self.grants.get(agent_name, [])

    def enforce(self, agent_name: str, tool_spec: ToolSpec):
        if not self.allows(agent_name, tool_spec.required_permission):
            raise PermissionError(
                f"{agent_name} is not authorized to use '{tool_spec.name}' "
                f"(requires '{tool_spec.required_permission}')")

def execute_tool(agent_name, tool_name, args, registry, permission_layer):
    spec = registry.get(tool_name)
    permission_layer.enforce(agent_name, spec)      # raises before execution if unauthorized
    return spec.fn(**args)
```

Enforce this at the tool-execution boundary, not just at the point where you build the tool catalog shown to the model. Defense-in-depth matters: even if a prompt injection tricks a model into *requesting* a disallowed tool, the call should still be rejected before it runs — never trust the model's own restraint as the only line of defense.

A few things worth adding as the system matures, drawn from what the current OWASP Agentic Security Initiative Top 10 flags as the common real-world failure modes:

- **Human-in-the-loop for destructive actions** — anything that writes or deletes outside the project sandbox should pause for confirmation rather than auto-executing
- **Audit log** — every tool call, by every agent, with arguments and result, written somewhere append-only (a JSONL file alongside your LangSmith traces is enough for a personal project)
- **Sandboxed code execution** — the coder and data-analyst agents' `run_python` tool should run in a container or restricted subprocess, never in the same process as your orchestration code

NeMo Guardrails (NVIDIA, free and open source) is worth knowing about if you want programmable guardrails beyond this — input, dialog, and execution "rails" — but at this project's scale, the permission layer above plus a hard tool-execution boundary covers the actual risk.

---

## 6. Evaluation and traceability

This is the layer that turns "it worked when I tested it" into "I know how it behaves in production," and it's where most hobby multi-agent projects fall short.

### 3.1 LangSmith for tracing (do this first — it's nearly free to add)

LangSmith is framework-agnostic but has native, near-automatic tracing for LangGraph. Set three environment variables and every LLM call, tool call, and node transition is captured as a trace tree:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=multi-agent-orchestrator
```

That's it — no code changes needed for LangGraph apps. The free Developer tier includes a monthly trace allowance, which is enough for a personal project. You get:

- **LangGraph Studio** — visualizes your graph as an inspectable tree; watch exactly which node ran, in what order, with what input/output, and where a loop or wrong branch happened
- **Datasets** — sample real runs (especially failures) into a permanent regression test set with one click
- **Evaluators** — LLM-as-judge, custom code checks, or human annotation queues, run against your dataset to catch regressions before you change a prompt and ship it blind

### 3.2 RAGAS — evaluate your shared memory retrieval

Your shared-memory lookups are functionally a RAG pipeline, so evaluate them like one. RAGAS gives you ground-truth-free metrics:

- **Faithfulness** — does the agent's output actually match what was retrieved from Qdrant?
- **Context precision/recall** — is retrieval pulling the right memories, or noise?
- **Answer relevancy** — does the final output address the actual task?

Run this in development to tune your retrieval (chunk size, top_k, embedding model choice) before it becomes a production problem.

### 3.3 DeepEval — CI/CD gate

DeepEval is pytest-shaped, so agent evaluation becomes a real test that can fail your build:

```python
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

def test_orchestrator_task_decomposition():
    test_case = LLMTestCase(
        input="Research competitor pricing and write a summary with a chart",
        actual_output=run_graph(task)["final_output"],
    )
    correctness = GEval(name="Task completeness", criteria="All subtasks were addressed")
    assert_test(test_case, [correctness])
```

Wire this into GitHub Actions (or whatever CI you use) so a prompt change or model swap that tanks quality fails the pipeline instead of silently shipping.

### 3.4 Agent-level benchmarks (optional, for rigor)

If you want to benchmark the system's raw capability rather than just your own eval set:
- **tau-bench** — realistic multi-turn agentic tasks with tool use
- **AgentBench** — broad agent-capability benchmark across environments
- **SWE-bench** — if the coder agent is a major focus, this measures real-world bug-fixing capability

These are heavier lifts — use them if you want a defensible "how good is this system" number, not for everyday iteration.

### 3.5 What to actually monitor

| Metric | Why it matters |
|---|---|
| Task success rate | Did the final output satisfy the original request |
| Tool-call accuracy | Did each agent use the right tool, with the right arguments |
| Step count vs. minimum | Catches inefficient loops/re-planning |
| Feedback loop trigger rate | If the reviewer sends work back constantly, a specialist prompt needs work |
| Latency & cost per agent | Free tiers have rate limits, not zero cost in time — track it |
| Faithfulness/hallucination score | From RAGAS, on the shared-memory-grounded outputs |

Offline evals (Layer 1–2: unit + regression suite) catch known failure modes before you ship. Online eval (Layer 3: sampling live traces in LangSmith) catches the drift and edge cases your test set never anticipated. You need both.

---

## 7. Interface

**Recommendation: Gradio.** As of 2026 it has the deepest AI-specific infrastructure (native MCP/tool exposure, works cleanly with Ollama and any OpenAI-compatible endpoint, concurrency handling for multi-user use), and unlike some alternatives it's actively and heavily developed. A basic chat interface with streaming and step visibility is genuinely a few dozen lines:

```python
import gradio as gr

def run_agent_system(message, history):
    result = app.stream({"task": message})  # LangGraph streaming
    for step in result:
        yield format_step_for_display(step)  # show intermediate agent activity
    yield step["final_output"]

gr.ChatInterface(
    run_agent_system,
    title="Multi-agent orchestrator",
    description="Free-model multi-agent system — orchestrator, planner, researcher, coder, data analyst, reviewer",
).launch()
```

**Alternatives, if your priorities differ:**
- **Streamlit** — better if you want dashboards (memory contents, trace charts, cost graphs) alongside the chat, not just chat
- **Chainlit** — purpose-built for chat with nice built-in step visualization, but worth knowing its core team stepped back from active development in 2025 and it's now community-maintained with a couple of disclosed vulnerabilities — fine for a personal project, worth reconsidering for anything public-facing
- **FastAPI + a thin frontend** — if you eventually want full control (custom auth, a non-chat UI element, embedding this in another product)

---

## 8. Input and output specification

### 8.1 Input

```json
{
  "session_id": "sess_8f2a1c",
  "user_id": "user_123",
  "query": "Research our top 3 competitors' pricing, write a Python script to scrape their public pricing pages, and summarize the findings with a comparison chart.",
  "attachments": [
    {"type": "file", "filename": "competitor_list.csv", "content_base64": "..."}
  ],
  "output_format": "markdown",
  "constraints": {
    "max_review_cycles": 3,
    "preferred_providers": ["groq", "ollama"]
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `session_id` | yes | Groups this run's Redis state and LangSmith trace |
| `user_id` | yes | Scopes long-term memory retrieval in Qdrant |
| `query` | yes | The natural-language task — this is what the orchestrator decomposes |
| `attachments` | no | Files the researcher/coder/analyst agents may need |
| `output_format` | no | `markdown` \| `json` \| `plain_text` — defaults to markdown |
| `constraints` | no | Caps and provider preferences per run |

### 8.2 Output

```json
{
  "session_id": "sess_8f2a1c",
  "status": "completed",
  "final_output": "## Competitor pricing summary\n\n...markdown content...",
  "artifacts": [
    {"type": "code", "filename": "scrape_pricing.py", "language": "python"},
    {"type": "chart", "filename": "pricing_comparison.png"}
  ],
  "sources": ["https://competitor-a.com/pricing", "https://competitor-b.com/pricing"],
  "agent_trace": {
    "langsmith_trace_url": "https://smith.langchain.com/...",
    "review_cycles": 1,
    "agents_invoked": ["orchestrator", "researcher", "coder", "reviewer", "aggregator"]
  },
  "metadata": {
    "total_latency_ms": 18420,
    "total_tokens": 14200,
    "providers_used": {"groq": 6, "ollama": 2}
  }
}
```

| Field | Notes |
|---|---|
| `final_output` | The aggregator's combined result, in the requested `output_format` |
| `artifacts` | Non-text outputs the coder/analyst agents produced |
| `sources` | Citations from the researcher agent, for traceability |
| `agent_trace` | Link to the full LangSmith trace plus a summary — this is what makes a result auditable, not just plausible |
| `metadata` | Cost/latency accounting — important to watch given free-tier rate limits |

---

## Suggested repo structure

```
multi-agent-orchestrator/
├── config/
│   ├── agents.yaml
│   └── permissions.yaml
├── agents/
│   ├── orchestrator.py
│   ├── planner.py
│   ├── researcher.py
│   ├── coder.py
│   ├── data_analyst.py
│   ├── reviewer.py
│   └── aggregator.py
├── memory/
│   ├── redis_store.py
│   └── qdrant_store.py
├── cache/
│   └── semantic_cache.py    # two-layer Redis + Qdrant cache
├── tools/
│   ├── registry.py           # ToolRegistry + ToolSpec
│   ├── web.py
│   ├── code_exec.py
│   └── files.py
├── security/
│   └── permissions.py        # PermissionLayer, enforce()
├── router.py                 # model router + fallback chain
├── graph.py                   # LangGraph definition
├── evals/
│   ├── deepeval_tests.py
│   └── ragas_eval.py
├── app.py                     # Gradio interface
└── .env                       # API keys (Groq, OpenRouter, LangSmith)
```

Want me to scaffold this as an actual runnable starter repo — the router, one working agent end-to-end with LangSmith tracing wired in, and the Gradio interface — so you have something you can `python app.py` and iterate on?
