"""
Phase 3 — Structured I/O Contracts (Pydantic Schemas)

Every agent that hands off structured state to the next node uses one of
these schemas.  The repair loop in ``structured_call.py`` validates against
these schemas and re-prompts on failure.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class OrchestratorOutput(BaseModel):
    """Orchestrator → specialists: task decomposition."""
    subtasks: list[str]               # concrete subtasks to delegate
    assignments: dict[str, str]       # subtask -> agent_name
    reasoning: str = ""              # why this decomposition (optional)
    max_review_cycles: int = 3


class PlannerOutput(BaseModel):
    """Planner → specialists: decomposed task plan."""
    tasks: list[str]                   # decomposed subtasks
    reasoning: str = ""               # why this decomposition (optional)
    priority_order: list[int] = []    # indices into tasks[], highest priority first (optional)


class ResearcherOutput(BaseModel):
    """Researcher → reviewer/aggregator: findings from web search."""
    findings: list[str]            # bullet-level findings
    sources: list[str] = []        # URLs or memory keys (optional)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("findings", mode="before")
    @classmethod
    def coerce_findings_to_strings(cls, v: Any) -> list[str]:
        """Accept list[str] or list[dict] — coerce dicts to their JSON repr."""
        if not isinstance(v, list):
            return [str(v)]
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Flatten dict to a readable finding string
                result.append(", ".join(f"{k}: {val}" for k, val in item.items()))
            else:
                result.append(str(item))
        return result


class CoderOutput(BaseModel):
    """Coder → reviewer/aggregator: generated code."""
    code: str                      # the generated code
    language: str                  # e.g. "python"
    explanation: str


class DataAnalystOutput(BaseModel):
    """Data analyst → reviewer/aggregator: analysis results."""
    summary: str
    insights: list[str]
    code_used: str | None = None


class ReviewerOutput(BaseModel):
    """Reviewer → orchestrator or aggregator: pass/fail verdict."""
    verdict: Literal["approved", "needs_revision"]
    feedback: str = ""               # empty string if approved (optional)


class AggregatorOutput(BaseModel):
    """Aggregator → end user: final assembled answer."""
    final_answer: str
    sources_used: list[str] = []     # optional — LLMs rarely populate this
    agent_steps: list[str] = []      # optional — human-readable trace