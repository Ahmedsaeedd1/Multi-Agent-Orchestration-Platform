import pytest
from unittest.mock import patch, MagicMock

from agents.fact_checker import fact_checker_node, ExtractedClaims, ClaimVerdict, FactCheckOutput

@pytest.fixture
def mock_findings():
    return {
        "research_output": {
            "findings": [
                "Company X's revenue was $5B in 2023.",
                "The CEO is Jane Doe.",
                "The market is growing rapidly.",
                "They acquired Company Y on June 1st.",
                "Their headquarters are in Tokyo.",
                "They have 10,000 employees.",
                "The stock price doubled."
            ]
        }
    }

@patch("agents.fact_checker.call_agent_structured")
@patch("agents.fact_checker.build_registry")
def test_claim_extraction_isolated(mock_build_registry, mock_call_agent, mock_findings):
    # Mock web search (shouldn't be reached if we only care about extraction, but good to have)
    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    mock_registry.get.return_value.fn = lambda query: "Search result"

    # We mock call_agent_structured to return different things on successive calls
    mock_call_agent.side_effect = [
        ExtractedClaims(claims=["Company X's revenue was $5B in 2023.", "The CEO is Jane Doe."]),
        FactCheckOutput(
            claims_checked=[
                ClaimVerdict(claim="Company X's revenue was $5B in 2023.", status="confirmed", evidence="..."),
                ClaimVerdict(claim="The CEO is Jane Doe.", status="confirmed", evidence="...")
            ],
            contradictions=[],
            confidence_summary="All good."
        )
    ]

    result = fact_checker_node(mock_findings)
    assert result["fact_check_output"] is not None
    assert len(result["fact_check_output"]["claims_checked"]) == 2

@patch("agents.fact_checker.call_agent_structured")
@patch("agents.fact_checker.build_registry")
def test_verification_classifies_correctly(mock_build_registry, mock_call_agent, mock_findings):
    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    
    # We mock web search to always return a contradiction for the first claim
    def mock_web_search(query):
        if "revenue" in query:
            return "Company X's revenue was actually $2B."
        return "Jane Doe is the CEO."
    mock_registry.get.return_value.fn = mock_web_search

    mock_call_agent.side_effect = [
        ExtractedClaims(claims=["Company X's revenue was $5B in 2023.", "The CEO is Jane Doe."]),
        FactCheckOutput(
            claims_checked=[
                ClaimVerdict(claim="Company X's revenue was $5B in 2023.", status="contradicted", evidence="Search says $2B"),
                ClaimVerdict(claim="The CEO is Jane Doe.", status="confirmed", evidence="Search confirmed")
            ],
            contradictions=["Company X's revenue was $5B in 2023."],
            confidence_summary="Found a contradiction on revenue."
        )
    ]

    result = fact_checker_node(mock_findings)
    fc_out = result["fact_check_output"]
    assert "contradicted" in [c["status"] for c in fc_out["claims_checked"]]
    assert len(fc_out["contradictions"]) == 1

@patch("agents.fact_checker.call_agent_structured")
@patch("agents.fact_checker.build_registry")
def test_message_history_continuity(mock_build_registry, mock_call_agent, mock_findings):
    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    mock_registry.get.return_value.fn = lambda q: "Search result"

    mock_call_agent.side_effect = [
        ExtractedClaims(claims=["Claim 1"]),
        FactCheckOutput(claims_checked=[], contradictions=[], confidence_summary="Done")
    ]

    fact_checker_node(mock_findings)
    
    # Check the second call to call_agent_structured
    assert mock_call_agent.call_count == 2
    second_call_kwargs = mock_call_agent.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]
    
    # The messages array must contain the assistant's previous response
    # Index 0: System prompt
    # Index 1: User prompt
    # Index 2: Assistant response (from call 1)
    # Index 3: User prompt (for call 2)
    assert len(messages) == 4
    assert messages[2]["role"] == "assistant"
    assert "Claim 1" in messages[2]["content"]

@patch("agents.fact_checker.call_agent_structured")
@patch("agents.fact_checker.build_registry")
def test_claim_limit_enforced(mock_build_registry, mock_call_agent, mock_findings):
    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    
    # Track how many times web_search is called
    search_mock = MagicMock(return_value="Result")
    mock_registry.get.return_value.fn = search_mock

    # The LLM extracts 7 claims
    mock_call_agent.side_effect = [
        ExtractedClaims(claims=[f"Claim {i}" for i in range(7)]),
        FactCheckOutput(claims_checked=[], contradictions=[], confidence_summary="Done")
    ]

    fact_checker_node(mock_findings)
    
    # search_mock should only be called 5 times because of the limit
    assert search_mock.call_count == 5

@patch("agents.fact_checker.call_agent_structured")
@patch("agents.fact_checker.build_registry")
def test_fact_checker_node_isolated(mock_build_registry, mock_call_agent, mock_findings):
    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    mock_registry.get.return_value.fn = lambda q: "Result"

    mock_call_agent.side_effect = [
        ExtractedClaims(claims=["Claim 1"]),
        FactCheckOutput(
            claims_checked=[ClaimVerdict(claim="Claim 1", status="confirmed", evidence="Evidence")],
            contradictions=[],
            confidence_summary="Summary text"
        )
    ]

    result = fact_checker_node(mock_findings)
    
    assert "fact_check_output" in result
    out = result["fact_check_output"]
    assert "claims_checked" in out
    assert "contradictions" in out
    assert "confidence_summary" in out
    assert out["confidence_summary"] == "Summary text"

@patch("agents.fact_checker.call_agent_structured")
@patch("agents.fact_checker.build_registry")
def test_confidence_summary_grounded_in_verdicts(mock_build_registry, mock_call_agent, mock_findings):
    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    mock_registry.get.return_value.fn = lambda q: "Result"

    mock_call_agent.side_effect = [
        ExtractedClaims(claims=["Claim 1"]),
        FactCheckOutput(
            claims_checked=[ClaimVerdict(claim="Claim 1", status="contradicted", evidence="Bad")],
            contradictions=["Claim 1"],
            confidence_summary="Found a contradiction on Claim 1."
        )
    ]

    result = fact_checker_node(mock_findings)
    
    assert "contradicted" in [c["status"] for c in result["fact_check_output"]["claims_checked"]]
    assert result["fact_check_output"]["confidence_summary"] == "Found a contradiction on Claim 1."
