import json
import subprocess
import sys
import tempfile
import os

def run_static_analysis(code: str) -> dict:
    """
    Write code to a temp file, run `ruff check` and `ruff format --check` against it.
    Returns {"issues": [{"line": int, "code": str, "message": str}], "clean": bool}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    issues = []
    clean = True

    try:
        # Run ruff check
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--output-format=json", temp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.stdout.strip():
            try:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues:
                    issues.append({
                        "line": issue.get("location", {}).get("row", 0),
                        "code": issue.get("code", "error"),
                        "message": issue.get("message", "Unknown error")
                    })
                    clean = False
            except json.JSONDecodeError:
                issues.append({"line": 0, "code": "E999", "message": f"Failed to parse ruff output: {result.stdout}"})
                clean = False
        elif result.returncode != 0 and not result.stdout.strip():
            issues.append({"line": 0, "code": "E999", "message": f"Ruff check failed: {result.stderr}"})
            clean = False

        # Run ruff format
        format_result = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--check", temp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if format_result.returncode != 0:
            issues.append({"line": 0, "code": "FMT", "message": "Code is not formatted correctly according to ruff."})

    except subprocess.TimeoutExpired:
        issues.append({"line": 0, "code": "TIMEOUT", "message": "Static analysis timed out."})
        clean = False
    finally:
        os.remove(temp_path)

    return {"issues": issues, "clean": clean}

def run_sandboxed_with_edge_cases(code: str, edge_case_inputs: list[dict]) -> dict:
    """
    Execute the code in a subprocess once per edge case input.
    Returns {"results": [{"input": ..., "output": ..., "error": str | None, "passed": bool}],
             "all_passed": bool}
    """
    results = []
    all_passed = True

    for case in edge_case_inputs:
        input_data = case.get("input", "")
        if not isinstance(input_data, str):
            input_data = json.dumps(input_data)
        
        passed = False
        error = None
        output = ""

        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=10
            )
            output = res.stdout
            if res.returncode != 0:
                error = res.stderr or "Non-zero exit code"
            else:
                passed = True
        except subprocess.TimeoutExpired:
            error = "Execution timed out"
        except Exception as e:
            error = str(e)

        if not passed:
            all_passed = False

        results.append({
            "input": input_data,
            "output": output,
            "error": error,
            "passed": passed
        })

    return {"results": results, "all_passed": all_passed}
