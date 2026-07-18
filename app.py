import os
import uuid
import logging
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

from graph import app as agent_graph

def run_agent_system(message, history):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    state = {
        # ── Required AgentState fields ──────────────────────────────
        "task": message,
        "run_id": run_id,
        "user_id": "gradio_user",
        # ── Structured output slots (None = not yet produced) ───────
        "plan": None,
        "plan_reasoning": "",
        "research_output": None,
        "code_output": None,
        "analysis_output": None,
        "review": None,
        "final_output": None,
        # ── Internal routing (set by orchestrator_wrapper) ──────────
        "_subtasks": [],
        "_assignments": {},
        # ── Counters & audit ────────────────────────────────────────
        "review_cycle_count": 0,
        "step_log": [],
        "error": None,
    }

    output_text = ""
    final_result = ""

    for step in agent_graph.stream(state, stream_mode="updates"):
        for node_name, node_state in step.items():
            if node_name == "planner":
                output_text += "🧠 **Planner**: Formulated plan.\n"
            elif node_name == "orchestrator":
                output_text += "🎯 **Orchestrator**: Decomposed task.\n"
            elif node_name == "researcher":
                # researcher_wrapper stores findings in research_output.findings
                research_out = node_state.get('research_output') or {}
                findings = research_out.get('findings', []) if isinstance(research_out, dict) else []
                num_notes = len(findings)
                output_text += f"🔍 **Researcher**: found {num_notes} findings\n"
            elif node_name == "coder":
                output_text += "💻 **Coder**: Generated code.\n"
            elif node_name == "data_analyst":
                output_text += "📊 **Data Analyst**: Analyzed data.\n"
            elif node_name == "reviewer":
                output_text += "✅ **Reviewer**: Reviewed implementation.\n"
            elif node_name == "aggregator":
                agg_out = node_state.get("final_output")
                if agg_out is None and node_state.get("error"):
                    output_text += f"📝 **Aggregator**: ⚠️ timed out — {node_state['error']}\n"
                else:
                    output_text += "📝 **Aggregator**: Compiled response.\n"
            else:
                output_text += f"⚙️ **{node_name}**: Completed step.\n"

            yield output_text

            if "final_output" in node_state and node_state["final_output"]:
                final_result = node_state["final_output"]
            # Capture the last known research findings for the timeout fallback
            if "research_output" in node_state and isinstance(node_state.get("research_output"), dict):
                _last_findings = node_state["research_output"].get("findings", [])

    if final_result:
        # aggregator_wrapper wraps the text as {"final_answer": "..."};
        # handle both that shape and a raw string gracefully.
        if isinstance(final_result, dict):
            answer_text = final_result.get("final_answer", str(final_result))
            sources = final_result.get("sources_used", [])
        elif hasattr(final_result, "final_answer"):
            answer_text = final_result.final_answer
            sources = getattr(final_result, "sources_used", [])
        else:
            answer_text = str(final_result)
            sources = []

        output_text += f"\n---\n### Final Output\n\n{answer_text}\n"
        if sources:
            output_text += "\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources) + "\n"
    else:
        # Aggregator timed out or failed — surface raw findings so the user
        # isn't left with a completely blank response.
        findings = locals().get("_last_findings", [])
        if findings:
            output_text += (
                "\n---\n### ⚠️ Aggregation timed out — Raw Research Findings\n\n"
                + "\n".join(f"- {f}" for f in findings)
                + "\n\n*Tip: set `AGGREGATOR_TIMEOUT` higher in `.env` if this happens often.*\n"
            )
        else:
            output_text += "\n---\n**Final Output**: No output generated (aggregator timed out).\n"
    
    yield output_text

if __name__ == "__main__":
    import socket, subprocess, sys

    PORT = int(os.getenv("APP_PORT", "7860"))

    # ── Kill any process already occupying PORT so we always start fresh ──
    def _free_port(port: int) -> None:
        """Best-effort: release PORT on Windows before Gradio binds it."""
        try:
            # Check if port is actually in use first
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return  # port is free already
            # netstat + taskkill to find & kill the occupying PID
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5,
                        )
                        logging.info("Freed port %d (killed PID %s)", port, pid)
        except Exception as exc:
            logging.warning("Could not free port %d: %s", port, exc)

    _free_port(PORT)

    demo = gr.ChatInterface(
        fn=run_agent_system,
        title="Multi-Agent Orchestrator",
        description="Chat with the multi-agent system.",
    )
    demo.launch(
        server_port=PORT,     # always bind to PORT — never auto-increment
        server_name="127.0.0.1",
    )