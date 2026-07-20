import json
import logging
from typing import Literal
from pydantic import BaseModel

from agents.orchestrator import AgentState
from agents.structured_call import call_agent_structured
from router import ModelRouter
from tools.code_eval import run_static_analysis, run_sandboxed_with_edge_cases

logger = logging.getLogger(__name__)

router = ModelRouter()

class EdgeCaseSet(BaseModel):
    edge_cases: list[dict]   # list of {"description": str, "input": dict}

class CodeEvalOutput(BaseModel):
    verdict: Literal["pass", "fail"]
    static_issues_summary: str
    edge_case_summary: str
    overall_summary: str


def code_evaluator_node(state: AgentState) -> dict:
    """
    1. Reads state['code_output']['code'] (the Coder's output)
    2. Calls run_static_analysis()
    3. Generates 3-5 edge case inputs via the model
    4. Calls run_sandboxed_with_edge_cases()
    5. Produces a structured verdict combining static + dynamic results
    """
    code = state.get("code_output", {}).get("code", "")
    if not code:
        logger.warning("Code Evaluator found no code to evaluate.")
        return {
            "code_eval_output": {
                "verdict": "fail",
                "static_issues": [],
                "edge_case_results": [],
                "summary": "No code was provided by the coder."
            }
        }

    task = state.get("task", "")

    # 1. Run static analysis
    logger.info("Running static analysis...")
    static_results = run_static_analysis(code)
    static_issues = static_results.get("issues", [])

    # 2. Generate edge cases via LLM
    logger.info("Generating edge cases...")
    edge_case_prompt = (
        f"Task:\n{task}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        "Generate 3 to 5 edge case inputs to test this code. "
        "Consider empty inputs, boundary values, incorrect types, or large inputs where applicable. "
        "Return valid JSON matching the EdgeCaseSet schema."
    )

    eval_messages = [
        {"role": "system", "content": "You are an expert QA engineer. Generate edge cases to test the provided code. Output JSON only."},
        {"role": "user", "content": edge_case_prompt}
    ]

    try:
        edge_case_set: EdgeCaseSet = call_agent_structured(
            router=router,
            agent_name="code_evaluator",
            messages=eval_messages,
            schema=EdgeCaseSet,
            max_repairs=2
        )
        edge_cases = edge_case_set.edge_cases
        edge_cases = edge_case_set
    except Exception as e:
        logger.error(f"Failed to generate edge cases: {e}")
        # Fallback empty case to ensure we still run at least one basic test if generation fails completely
        edge_cases = EdgeCaseSet(edge_cases=[])
        
    if not edge_cases.edge_cases:
        logger.warning("No edge cases generated, skipping evaluation.")
        return {
            "code_eval_output": {
                "verdict": "fail",
                "static_issues": static_issues,
                "edge_case_results": [],
                "summary": "No edge cases could be generated or executed — evaluation incomplete, not a genuine pass."
            }
        }
        
    eval_messages.append({"role": "assistant", "content": edge_cases.model_dump_json(indent=2)})

    # 3. Run sandboxed execution with edge cases
    logger.info("Running sandboxed edge case tests...")
    edge_case_results_dict = run_sandboxed_with_edge_cases(code, edge_cases.edge_cases)
    edge_case_results = edge_case_results_dict.get("results", [])

    # 4. Generate summary and verdict via LLM
    logger.info("Generating code evaluation summary and verdict...")
    static_issues_str = json.dumps(static_issues, indent=2)
    edge_case_results_str = json.dumps(edge_case_results, indent=2)

    verdict_prompt = (
        f"Task:\n{task}\n\n"
        f"Static Analysis Issues:\n{static_issues_str}\n\n"
        f"Edge Case Execution Results:\n{edge_case_results_str}\n\n"
        "Review the static analysis issues and edge case execution results. "
        "Rule: The verdict MUST be 'fail' if any edge case errors unexpectedly, "
        "if any execution times out, or if static analysis finds issues classified as errors. "
        "Style-only lint warnings alone should NOT force a fail. "
        "Ground your overall_summary in the actual results provided above. "
        "Return valid JSON matching the CodeEvalOutput schema."
    )

    try:
        eval_output: CodeEvalOutput = call_agent_structured(
            router=router,
            agent_name="code_evaluator",
            messages=[
                {"role": "system", "content": "You are an expert QA engineer. Evaluate the results of static analysis and dynamic execution tests. Output JSON only."},
                {"role": "user", "content": verdict_prompt}
            ],
            schema=CodeEvalOutput,
            max_repairs=2
        )
        
        # Enforce hard fail conditions just in case the LLM hallucinates a pass
        has_real_error = False
        for case in edge_case_results:
            if not case.get("passed", False):
                has_real_error = True
        for issue in static_issues:
            if issue.get("code") in ("E999", "TIMEOUT"):
                has_real_error = True
            elif issue.get("code", "").startswith("E") or issue.get("code", "").startswith("F"):
                # E and F usually mean real errors in Ruff (Syntax, undefined name, etc.)
                # W are warnings
                has_real_error = True

        final_verdict = eval_output.verdict
        if has_real_error:
            final_verdict = "fail"

        summary = f"Verdict: {final_verdict.upper()}\n\nStatic Analysis: {eval_output.static_issues_summary}\n\nEdge Cases: {eval_output.edge_case_summary}\n\nOverall: {eval_output.overall_summary}"

        return {
            "code_eval_output": {
                "verdict": final_verdict,
                "static_issues": static_issues,
                "edge_case_results": edge_case_results,
                "summary": summary
            }
        }
    except Exception as e:
        logger.error(f"Failed to generate evaluation summary: {e}")
        return {
            "code_eval_output": {
                "verdict": "fail",
                "static_issues": static_issues,
                "edge_case_results": edge_case_results,
                "summary": f"Failed to parse evaluation results: {e}"
            }
        }
