# Pending Test Suite Fixes

- **`test_graph.py` failures:** There are 7 failing tests related to `TestCycleCapEnforcement` and `TestStepLogPopulation`. 
  - **Root Cause:** A shared mutable mock dictionary (`orc_ret`-style module-level dicts) is being mutated across sequential test functions. Specifically, `result.pop()` permanently deletes keys from the mock dictionary in the first test, which breaks the exact same dictionary when reused in the subsequent tests.
  - **Fix:** Update the test mocks to use a factory function (e.g. `def _make_orc_ret(): return {...}`) or `copy.deepcopy()` per test to ensure isolated, fresh state instead of shared module-level mutable dicts.
