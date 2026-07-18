import json
import logging
from typing import Literal
from pydantic import BaseModel
from router import ModelRouter
from tools.registry import build_registry
from agents.structured_call import call_agent_structured

logger = logging.getLogger(__name__)

class ExtractedClaims(BaseModel):
    claims: list[str]

class ClaimVerdict(BaseModel):
    claim: str
    status: Literal["confirmed", "contradicted", "unverifiable"]
    evidence: str

class FactCheckOutput(BaseModel):
    claims_checked: list[ClaimVerdict]
    contradictions: list[str]
    confidence_summary: str

def fact_checker_node(state: dict) -> dict:
    """
    1. Reads state['research_output']['findings'] (the Researcher's output)
    2. Generates a list of specific, checkable claims extracted from those findings
       via the model (e.g. "Company X's revenue was $Y in year Z" - not vague
       statements like "the market is growing")
    3. For each claim, calls web_search to independently verify it
    4. Strips <think> blocks from model output (handled in call_agent_structured)
    5. Classifies each claim: "confirmed", "contradicted", or "unverifiable"
    6. Returns {"fact_check_output": {"claims_checked": [...],
                                      "contradictions": [...],
                                      "confidence_summary": str}}
    """
    router = ModelRouter()
    registry = build_registry()
    web_search = registry.get("web_search").fn

    research_output = state.get("research_output", {})
    findings = research_output.get("findings", [])
    if not findings:
        return {"fact_check_output": None}

    # Format findings into a string
    findings_text = ""
    for i, f in enumerate(findings, 1):
        if isinstance(f, dict):
            findings_text += f"{i}. {json.dumps(f)}\n"
        else:
            findings_text += f"{i}. {f}\n"

    # Stage 1: Extract Claims
    sys_prompt = (
        "You are a rigorous Fact-Checker. Your goal is to verify the claims made in research findings.\n"
        "Extract up to 5 specific, checkable claims from the provided findings. Focus on hard numbers, "
        "dates, names, and definitive statements. Avoid vague or subjective statements.\n"
        "Output valid JSON matching the ExtractedClaims schema."
    )
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Extract claims from these findings (MAX 5 claims):\n\n{findings_text}"}
    ]

    try:
        extracted: ExtractedClaims = call_agent_structured(
            router=router,
            agent_name="fact_checker",
            messages=messages,
            schema=ExtractedClaims
        )
        claims = extracted.claims[:5]  # Enforce max 5 claims
    except Exception as e:
        logger.error(f"Failed to extract claims: {e}")
        return {
            "fact_check_output": {
                "claims_checked": [],
                "contradictions": [],
                "confidence_summary": f"Fact-checking failed during claim extraction: {e}"
            }
        }

    if not claims:
        return {
            "fact_check_output": {
                "claims_checked": [],
                "contradictions": [],
                "confidence_summary": "No specific claims could be extracted for verification."
            }
        }

    # Append first response to history
    messages.append({
        "role": "assistant",
        "content": extracted.model_dump_json(indent=2)
    })

    # Stage 2: Verify Claims via Web Search
    logger.info(f"Verifying {len(claims)} claims...")
    search_results_context = []
    
    for i, claim in enumerate(claims, 1):
        try:
            # Query the web for each claim
            result = web_search(query=claim)
            search_results_context.append(f"Claim {i}: {claim}\nSearch Result: {result}\n")
        except Exception as e:
            logger.error(f"Web search failed for claim: {claim}. Error: {e}")
            search_results_context.append(f"Claim {i}: {claim}\nSearch Result: [Search Failed: {e}]\n")

    # Stage 3: Classify Claims
    eval_prompt = (
        "Now, look at the web search results for each claim you extracted.\n\n"
        f"Search Results:\n{''.join(search_results_context)}\n"
        "You must format your response EXACTLY matching the FactCheckOutput schema. For example:\n"
        "```json\n"
        "{\n"
        '  "claims_checked": [\n'
        '    {"claim": "The population is 8 billion.", "status": "confirmed", "evidence": "Source X says 8 billion."}\n'
        "  ],\n"
        '  "contradictions": [],\n'
        '  "confidence_summary": "All claims were verified."\n'
        "}\n"
        "```\n"
    )
    eval_messages = [
        {"role": "system", "content": "You are a rigorous Fact-Checker."},
        {"role": "user", "content": eval_prompt}
    ]

    try:
        fact_check_output: FactCheckOutput = call_agent_structured(
            router=router,
            agent_name="fact_checker",
            messages=eval_messages,
            schema=FactCheckOutput
        )
        return {"fact_check_output": fact_check_output.model_dump()}
    except Exception as e:
        logger.error(f"Failed to generate fact-check verdict: {e}")
        return {
            "fact_check_output": {
                "claims_checked": [],
                "contradictions": [],
                "confidence_summary": f"Failed to generate evaluation summary: {e}"
            }
        }
