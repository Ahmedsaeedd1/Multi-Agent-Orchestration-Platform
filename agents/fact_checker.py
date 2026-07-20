import json
import logging
from typing import Literal
from pydantic import BaseModel
from router import ModelRouter
from tools.registry import build_registry
from agents.structured_call import call_agent_structured

logger = logging.getLogger(__name__)

class ExtractedClaims(BaseModel):
    qualitative_claims: list[str]
    numeric_claims: list[str]

class NumericConsistencyOutput(BaseModel):
    contradictions: list[str]

class ClaimVerdict(BaseModel):
    claim: str
    status: Literal["confirmed", "contradicted", "unverifiable"]
    evidence: str

class FactCheckOutput(BaseModel):
    claims_checked: list[ClaimVerdict]
    contradictions: list[str]
    confidence_summary: str

def fact_checker_node(state: dict) -> dict:
    router = ModelRouter()
    registry = build_registry()
    web_search = registry.get("web_search").fn

    research_output = state.get("research_output", {})
    findings = research_output.get("findings", [])
    if not findings:
        return {"fact_check_output": None}

    findings_text = ""
    for i, f in enumerate(findings, 1):
        if isinstance(f, dict):
            findings_text += f"{i}. {json.dumps(f)}\n"
        else:
            findings_text += f"{i}. {f}\n"

    # Stage 1: Extract Claims
    sys_prompt = (
        "You are a rigorous Fact-Checker. Your goal is to verify the claims made in research findings.\n"
        "Extract up to 5 specific, checkable claims from the provided findings. Separate them into "
        "qualitative_claims and numeric_claims (e.g. 'AWS holds 29% market share', 'Revenue was $5B').\n"
        "Focus on hard numbers, dates, names, and definitive statements. Avoid vague or subjective statements.\n"
        "Output valid JSON matching the ExtractedClaims schema."
    )
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Extract claims from these findings (MAX 5 claims total):\n\n{findings_text}"}
    ]

    try:
        extracted: ExtractedClaims = call_agent_structured(
            router=router,
            agent_name="fact_checker",
            messages=messages,
            schema=ExtractedClaims
        )
        qual_claims = extracted.qualitative_claims or []
        num_claims = extracted.numeric_claims or []
        claims = (qual_claims + num_claims)[:5]
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

    messages.append({
        "role": "assistant",
        "content": extracted.model_dump_json(indent=2)
    })

    # Stage 1.5: Numeric Internal Consistency Check
    internal_contradictions = []
    if len(num_claims) > 1:
        consistency_prompt = (
            "You are an internal consistency checker. Look at the following numeric claims extracted from a single document. "
            "Check if any two claims about the SAME entity disagree with each other (e.g., 'AWS has 32%' vs 'AWS market share is 29%'). "
            "Output any disagreements found in the 'contradictions' list. If none, return an empty list."
        )
        consistency_msg = [
            {"role": "system", "content": consistency_prompt},
            {"role": "user", "content": f"Numeric claims:\n{chr(10).join(num_claims)}"}
        ]
        try:
            consistency_result: NumericConsistencyOutput = call_agent_structured(
                router=router,
                agent_name="fact_checker",
                messages=consistency_msg,
                schema=NumericConsistencyOutput
            )
            internal_contradictions = consistency_result.contradictions
        except Exception as e:
            logger.error(f"Failed numeric consistency check: {e}")

    # Stage 2: Verify Claims via Web Search
    logger.info(f"Verifying {len(claims)} claims...")
    search_results_context = []
    
    for i, claim in enumerate(claims, 1):
        try:
            result = web_search(query=claim)
            search_results_context.append(f"Claim {i}: {claim}\nSearch Result: {result}\n")
        except Exception as e:
            logger.error(f"Web search failed for claim: {claim}. Error: {e}")
            search_results_context.append(f"Claim {i}: {claim}\nSearch Result: [Search Failed: {e}]\n")

    if internal_contradictions:
        search_results_context.append(
            "\nCRITICAL INTERNAL CONTRADICTIONS DETECTED IN NUMERIC CLAIMS:\n" + 
            "\n".join(f"- {c}" for c in internal_contradictions) +
            "\nThese internal contradictions MUST be marked as 'contradicted' in the final output."
        )

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
