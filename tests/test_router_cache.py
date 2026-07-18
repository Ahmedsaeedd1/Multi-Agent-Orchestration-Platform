import pytest
import uuid
from unittest.mock import patch
from router import ModelRouter

@pytest.fixture
def unique_messages():
    return [{"role": "user", "content": f"Test message {uuid.uuid4()}"}]

def test_cache_hit_skips_llm_call(unique_messages):
    router = ModelRouter()
    
    with patch.object(router, "_call", return_value="Mocked LLM Response") as mock_call:
        # 1st call: Cache miss, should invoke _call (via _do_call)
        res1 = router.call("aggregator", unique_messages)
        assert res1 == "Mocked LLM Response"
        assert mock_call.call_count == 1
        
        # 2nd call: Cache hit, should NOT invoke _call
        res2 = router.call("aggregator", unique_messages)
        assert res2 == "Mocked LLM Response"
        assert mock_call.call_count == 1

def test_cache_disabled_always_calls(unique_messages):
    router = ModelRouter()
    
    with patch("router._cache.enabled", False):
        with patch.object(router, "_call", return_value="Mocked LLM Response Disabled") as mock_call:
            # 1st call
            res1 = router.call("aggregator", unique_messages)
            assert mock_call.call_count == 1
            
            # 2nd call
            res2 = router.call("aggregator", unique_messages)
            assert mock_call.call_count == 2
