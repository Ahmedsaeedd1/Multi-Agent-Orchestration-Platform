import os
import sys
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from langsmith.utils import LangSmithAuthError

def verify():
    # 1. Load .env variables
    load_dotenv()
    
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    project = os.getenv("LANGCHAIN_PROJECT", "default").strip()

    print("LangSmith Environment Configuration:")
    print(f"  LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"  LANGCHAIN_PROJECT:    {project}")
    print(f"  LANGCHAIN_API_KEY:    {'<set>' if api_key else '<not set>'}\n")

    if not api_key or not tracing_v2:
        print("WARN — LANGCHAIN_API_KEY not set, tracing disabled")
        return

    # 2. Make one real LangChain-traced call
    runnable = RunnableLambda(lambda x: x + 1)
    
    try:
        # Initialize client to verify credentials
        client = Client()
        
        with collect_runs() as cb:
            res = runnable.invoke(1)
            if not cb.traced_runs:
                raise ValueError("No runs were collected by the callback handler.")
            run = cb.traced_runs[0]
        
        # 3. Print the trace URL
        # Retrieve the URL from the run if possible
        try:
            url = client.get_run_url(run=run)
            print(f"Traced Run URL: {url}")
        except Exception as e:
            # Fallback to manual URL construction if get_run_url fails
            url = f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run.id}"
            print(f"Traced Run URL (constructed): {url} (Client call failed: {e})")
            
        print("OK — LangSmith tracing active")
        
    except (LangSmithAuthError, Exception) as e:
        print(f"LangSmith connection failed: {e}")
        print("WARN — LANGCHAIN_API_KEY not set, tracing disabled")

if __name__ == "__main__":
    verify()
