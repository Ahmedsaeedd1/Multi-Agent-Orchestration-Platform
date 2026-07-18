# Pending Test Suite Fixes

- **`test_graph.py` failures:** There are 7 failing tests related to `TestCycleCapEnforcement` and `TestStepLogPopulation`. 
  - **Root Cause:** A shared mutable mock dictionary (`orc_ret`-style module-level dicts) is being mutated across sequential test functions. Specifically, `result.pop()` permanently deletes keys from the mock dictionary in the first test, which breaks the exact same dictionary when reused in the subsequent tests.
  - **Fix:** Update the test mocks to use a factory function (e.g. `def _make_orc_ret(): return {...}`) or `copy.deepcopy()` per test to ensure isolated, fresh state instead of shared module-level mutable dicts.
- **`test_graph.py` step-count & mock updates needed:** The addition of `code_evaluator` (routed `coder -> code_evaluator -> reviewer`) broke test_graph.py's expected step_log counts. Furthermore, `code_evaluator_node` is currently unmocked in test_graph.py, causing the tests to hit the real LLM/sandboxed execution and take 2+ minutes.
  - **Fix:** Add `code_evaluator_node` to the patch list in `_run_with_patched_nodes` and update the expected step counts/node sequences (e.g. `max_cycles * (1+2+1)+1`) to account for the new node.

- **[2026-07-18] `test_graph.py` failure count increased to 8:** The addition of the `fact_checker` node (routed `researcher -> fact_checker -> reviewer/aggregator`) further broke `test_graph.py` step counts. The failure count increased from 7 to 8, now including `TestFanOutSelectivity::test_all_three_specialists_invoked`.
  - **Fix:** Mock `fact_checker_node` in the graph tests and update sequence assertions to account for it running after the researcher node.
