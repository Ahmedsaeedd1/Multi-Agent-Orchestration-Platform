import json
import re
from pydantic import BaseModel
from agents.structured_call import call_agent_structured
from router import ModelRouter
from tools.sql import get_schema, execute_sql
from tools.registry import build_registry

class SQLAssistantOutput(BaseModel):
    query: str
    explanation: str

class SQLSummaryOutput(BaseModel):
    result_summary: str

def sql_assistant_node(state: dict) -> dict:
    """
    1. Calls get_schema() to load table/column context
    2. Stores schema in Qdrant (memory_write) under collection 'domain_knowledge'
       so future runs don't re-fetch unchanged schemas
    3. Takes state['task'] + schema context, generates a SQL query via the model
    4. Strips <think>...</think> blocks from model output (DeepSeek-R1 emits these)
       (Note: call_agent_structured/router.py handles stripping automatically before Pydantic parsing)
    5. Validates the query is read-only before execution
    6. Calls execute_sql() with the generated query
    7. Makes a second model call to summarize the actual results
    8. Returns {"sql_output": {"query": str, "result": dict, "summary": str}}
    """
    registry = build_registry()
    memory_write_spec = registry.get("memory_write")
    memory_write = memory_write_spec.fn
    
    schema_info = get_schema()
    
    # Store schema in memory
    memory_write(
        text=f"Database Schema:\n{json.dumps(schema_info, indent=2)}", 
        collection="domain_knowledge"
    )
    
    router = ModelRouter()
    system_prompt = (
        "You are a SQL assistant. Generate a read-only SQL query to answer the user's task.\n"
        "You are querying a Postgres database. Use Postgres syntax (e.g., use information_schema.tables instead of sqlite_master).\n"
        f"Schema Context:\n{json.dumps(schema_info, indent=2)}\n\n"
        "Output valid JSON matching the SQLAssistantOutput schema. "
        "Do NOT output anything except the JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.get("task", "")}
    ]
    
    output: SQLAssistantOutput = call_agent_structured(
        router=router,
        agent_name="sql_assistant",
        messages=messages,
        schema=SQLAssistantOutput
    )
    
    query = output.query
    
    # Extra protection to strip think blocks if they leaked into the query field
    query = re.sub(r"<think>.*?</think>\s*", "", query, flags=re.DOTALL).strip()
    
    # Validate read-only
    query_upper = query.upper()
    forbidden_keywords = ['INSERT ', 'UPDATE ', 'DELETE ', 'DROP ', 'ALTER ', 'CREATE ']
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            raise PermissionError("Only read-only SELECT queries are allowed.")
            
    result = execute_sql(query)
    
    # Second model call to summarize the actual results
    summary_prompt = (
        "You are a SQL data summarizer. Given the user's task, the SQL query executed, and the actual result rows returned, "
        "write a concise summary that directly answers the user's task using the actual data.\n"
        "Output valid JSON matching the SQLSummaryOutput schema. Do NOT output anything except the JSON."
    )
    summary_messages = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": f"Task: {state.get('task', '')}\nQuery: {query}\nResult: {json.dumps(result, default=str)}"}
    ]
    
    summary_output: SQLSummaryOutput = call_agent_structured(
        router=router,
        agent_name="sql_assistant",
        messages=summary_messages,
        schema=SQLSummaryOutput
    )
    
    return {
        "sql_output": {
            "query": query,
            "result": result,
            "summary": summary_output.result_summary
        }
    }
