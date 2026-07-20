"""
Phase 2 — Model Router with Automatic Fallback

Every agent calls this single router instead of a raw OpenAI client.
On RateLimitOrProviderError, tenacity retries with exponential backoff;
if the primary provider still fails, the call chain falls through to
fallback providers defined in config/agents.yaml.
"""

import os
import re
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from cache.semantic_cache import SemanticCache
from memory.redis_store import RedisStore
from memory.qdrant_store import QdrantStore

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def _embed(text: str) -> list[float]:
    import httpx
    resp = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

_redis_store = RedisStore(url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
_qdrant_cache_store = QdrantStore(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection="cache_responses",
    embedding_fn=_embed,
)
_cache = SemanticCache(
    redis_store=_redis_store,
    qdrant_store=_qdrant_cache_store,
    enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
    exact_ttl=int(os.getenv("CACHE_EXACT_TTL", "3600")),
    semantic_threshold=float(os.getenv("CACHE_SEMANTIC_THRESHOLD", "0.90")),
)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RateLimitOrProviderError(Exception):
    """Raised when a provider returns 429 or another transient error."""


# ---------------------------------------------------------------------------
# Thinking-tag stripper (single point of removal)
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def strip_thinking(text: str) -> str:
    """
    Remove the first ``<think>...</think>`` block (including the closing
    tag).  Content after the tag is the actual response.

    Safe to call on any string — if no ``<think>`` block exists the
    original string is returned unchanged.
    """
    return _THINK_PATTERN.sub("", text, count=1).strip()


# ---------------------------------------------------------------------------
# Provider clients  (exact map from the Phase 2 spec)
# ---------------------------------------------------------------------------

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
OR_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ZAI_KEY = os.getenv("ZAI_API_KEY", "") or os.getenv("ZAI_KEY", "")
DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")

CLIENTS = {
    "groq": OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_KEY,
    ),
    "openrouter": OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OR_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/multi-agent-orchestrator",
            "X-Title": "multi-agent-orchestrator",
        },
    ),
    "huggingface": OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    ),
    "zai": OpenAI(
        base_url="https://api.z.ai/api/paas/v4",
        api_key=ZAI_KEY,
    ),
    "deepseek": OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=DS_KEY,
    ),
}

# Models that emit <think>...</think> blocks and need stripping.
# Covers direct DeepSeek API, OpenRouter paths, and HuggingFace router paths.
_THINKING_MODELS = frozenset({
    # HuggingFace router
    "deepseek-ai/DeepSeek-R1:novita",
    "Qwen/Qwen3-8B:featherless-ai",
    # DeepSeek direct API
    "deepseek-reasoner",
    # OpenRouter paths
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "qwen/qwen3-30b-a3b",
})


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Loads agent configuration from ``config/agents.yaml`` and routes
    completion calls through a primary → fallback chain.

    Usage::

        router = ModelRouter()
        resp = router.call("researcher", messages)
    """

    def __init__(self, config_path: str = "config/agents.yaml"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_full_path = os.path.join(base_dir, config_path)
        with open(config_full_path) as f:
            self.config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Internal: single-provider call with tenacity retry
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(RateLimitOrProviderError),
    )
    def _call(self, provider: str, model: str, messages: list, **kwargs) -> str:
        """Make one completion call and return the content string."""
        client = CLIENTS.get(provider)
        if client is None:
            raise ValueError(f"Unknown provider '{provider}' — check CLIENTS map")

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=20.0,
                **kwargs,
            )
        except Exception as e:
            err_str = str(e).lower()
            if (
                "429" in err_str or "rate" in err_str or "503" in err_str or
                "timeout" in err_str or "timed out" in err_str or
                "connection" in err_str or "api_timeout" in err_str
            ):
                raise RateLimitOrProviderError(str(e)) from e
            raise  # non-retryable error propagates immediately

        content = resp.choices[0].message.content
        if content is None:
            raise ValueError(f"Empty response from {provider}/{model}")

        # Strip thinking tags if the model is known to emit them
        if model in _THINKING_MODELS:
            content = strip_thinking(content)

        return content

    # ------------------------------------------------------------------
    # Public: primary + fallback chain
    # ------------------------------------------------------------------

    def call(
        self,
        agent_name: str,
        messages: list,
        **kwargs,
    ) -> str:
        """
        Route a completion for *agent_name* through its configured
        primary provider, falling back through the fallback list if
        every retry of the current provider fails.

        Returns the content string of the first successful response.
        """
        cfg = self.config.get(agent_name)
        if cfg is None:
            raise KeyError(
                f"Agent '{agent_name}' not found in config/agents.yaml"
            )

        chain: list[dict] = [cfg["primary"]] + cfg.get("fallback", [])
        
        temperature = kwargs.pop("temperature", cfg.get("temperature", 0.3))
        max_tokens = kwargs.pop("max_tokens", cfg.get("max_tokens", 2048))

        def _do_call(agent: str, msgs: list) -> str:
            last_error: Exception | None = None
            for target in chain:
                provider = target["provider"]
                model = target["model"]
                try:
                    return self._call(
                        provider,
                        model,
                        msgs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except (RateLimitOrProviderError, Exception) as e:
                    last_error = e
                    continue  # try next provider in the chain

            raise RuntimeError(
                f"All providers failed for agent '{agent}': {last_error}"
            )

        return _cache.get_cached_or_call(
            agent_name, 
            messages, 
            _do_call, 
            temperature=temperature, 
            max_tokens=max_tokens
        )