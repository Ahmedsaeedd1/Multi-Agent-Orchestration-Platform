"""
DeepEval judge-model adapter backed by the project's own ModelRouter.

Avoids adding a new dependency (Ollama + local model pull) for the sole
purpose of running GEval — this reuses the same free-tier providers
already configured and verified in Phase 0 (Groq / HuggingFace /
OpenRouter / Z.ai).

Default judge: "reviewer" role → Groq llama-3.3-70b-versatile
(config/agents.yaml). This is a good default judge model — it's the
strongest/fastest primary in the whole config (1.09s latency in the
Phase 0 verification table) and already has a configured fallback
chain, so a transient Groq outage doesn't take the eval suite down
with it.
"""

import logging

from deepeval.models import DeepEvalBaseLLM

from router import ModelRouter

logger = logging.getLogger(__name__)


class RouterBackedLLM(DeepEvalBaseLLM):
    """
    Adapts ModelRouter to DeepEval's judge-model interface so GEval
    (and any other DeepEval metric that needs an LLM judge) can run
    against your existing Groq/HF/OpenRouter setup instead of OpenAI
    or a locally-pulled Ollama model.
    """

    def __init__(self, agent_name: str = "reviewer", router: ModelRouter | None = None):
        self.agent_name = agent_name
        self.router = router or ModelRouter()

    def load_model(self):
        return self.agent_name

    def generate(self, prompt: str, *args, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.router.call(self.agent_name, messages)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        # DeepEval passes schema=Steps when it expects a Pydantic model back.
        # We only return strings, so raise TypeError to force DeepEval's
        # JSON-parsing fallback (trimAndLoadJson) instead.
        if "schema" in kwargs:
            raise TypeError(
                "RouterBackedLLM does not support schema-based generation. "
                "DeepEval will fall back to JSON parsing."
            )
        return self.generate(prompt, *args, **kwargs)

    def get_model_name(self):
        return f"router:{self.agent_name}"