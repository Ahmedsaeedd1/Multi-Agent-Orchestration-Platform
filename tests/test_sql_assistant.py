import pytest
from unittest.mock import patch, MagicMock
from tools.sql import execute_sql, get_schema
from agents.sql_assistant import sql_assistant_node, SQLAssistantOutput
import psycopg2
import os

@pytest.fixture(scope="module")
def postgres_connection():
    # Rely on init_db.py having initialized the DB via the env vars
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5434"),
        dbname=os.getenv("POSTGRES_DB", "multi_agent_db"),
        user=os.getenv("POSTGRES_USER", "multi_agent_user"),
        password=os.getenv("POSTGRES_PASSWORD", "a3f9c1e8b2d4f6a0c7e9b1d3f5a7c9e1")
    )
    yield conn
    conn.close()

def test_execute_sql_returns_correct_shape(postgres_connection):
    result = execute_sql("SELECT * FROM schema_registry")
    assert "columns" in result
    assert "rows" in result
    assert "row_count" in result
    assert isinstance(result["columns"], list)
    assert isinstance(result["rows"], list)
    assert isinstance(result["row_count"], int)

def test_execute_sql_blocks_write_operations(postgres_connection):
    with pytest.raises(PermissionError):
        execute_sql("INSERT INTO schema_registry (table_name) VALUES ('test')")

def test_get_schema_returns_all_tables(postgres_connection):
    result = get_schema(table_name=None)
    assert "tables" in result
    assert result["tables"]
    assert "products" in result["tables"]

def test_execute_sql_readonly_session_enforced(postgres_connection):
    # This query bypasses the keyword filter by using a newline instead of a space,
    # but the read-only session should catch it anyway at the database level.
    with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
        execute_sql("INSERT\nINTO schema_registry (table_name) VALUES ('test')")

@patch("agents.sql_assistant.call_agent_structured")
@patch("agents.sql_assistant.get_schema")
@patch("agents.sql_assistant.execute_sql")
@patch("agents.sql_assistant.build_registry")
def test_sql_assistant_node_isolated(mock_registry, mock_execute, mock_schema, mock_call):
    mock_schema.return_value = {"tables": {"users": []}}
    mock_execute.return_value = {"columns": ["id"], "rows": [{"id": 1}], "row_count": 1}
    
    mock_output_query = SQLAssistantOutput(
        query="SELECT * FROM users",
        explanation="Retrieve all users"
    )
    from agents.sql_assistant import SQLSummaryOutput
    mock_output_summary = SQLSummaryOutput(
        result_summary="There is 1 user"
    )
    mock_call.side_effect = [mock_output_query, mock_output_summary]
    
    # Mock registry for memory_write
    mock_reg = MagicMock()
    mock_spec = MagicMock()
    mock_spec.fn = MagicMock()
    mock_reg.get.return_value = mock_spec
    mock_registry.return_value = mock_reg
    
    state = {"task": "Get users"}
    result = sql_assistant_node(state)
    
    assert "sql_output" in result
    sql_out = result["sql_output"]
    assert sql_out["query"] == "SELECT * FROM users"
    assert sql_out["result"] == {"columns": ["id"], "rows": [{"id": 1}], "row_count": 1}
    assert sql_out["summary"] == "There is 1 user"
    
    mock_schema.assert_called_once()
    mock_execute.assert_called_once_with("SELECT * FROM users")
    assert mock_call.call_count == 2
    mock_spec.fn.assert_called_once() # memory_write
