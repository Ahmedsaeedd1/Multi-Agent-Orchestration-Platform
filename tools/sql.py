import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize connection pool globally so it's loaded once per module load
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=5,
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)

def execute_sql(query: str) -> dict:
    """
    Execute a READ-ONLY SQL query against the Postgres database.
    Raises PermissionError if query contains INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE.
    Returns {"columns": [...], "rows": [...], "row_count": int}
    """
    query_upper = query.upper()
    forbidden_keywords = [
        'INSERT ', 'UPDATE ', 'DELETE ', 'DROP ', 'ALTER ', 'CREATE ',
        'TRUNCATE ', 'GRANT ', 'REVOKE '
    ]
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            raise PermissionError("Only read-only SELECT queries are allowed.")
    
    conn = db_pool.getconn()
    try:
        # Enforce read-only at the database level
        conn.set_session(readonly=True)
        cursor = conn.cursor()
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        
        # Convert row tuples to dictionaries for JSON serialization
        row_dicts = [dict(zip(columns, row)) for row in rows]
        
        return {
            "columns": columns,
            "rows": row_dicts,
            "row_count": len(rows)
        }
    finally:
        db_pool.putconn(conn)

def get_schema(table_name: str = None) -> dict:
    """
    Return schema information from actual database tables (information_schema),
    enriched with descriptions from schema_registry.
    If table_name is None, returns all tables.
    Returns {"tables": {table_name: [{"column": ..., "type": ..., "description": ...}]}}
    """
    conn = db_pool.getconn()
    try:
        # Enforce read-only at the database level
        conn.set_session(readonly=True)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                c.table_name, 
                c.column_name, 
                c.data_type, 
                sr.description
            FROM information_schema.columns c
            JOIN information_schema.tables t ON c.table_name = t.table_name AND c.table_schema = t.table_schema
            LEFT JOIN schema_registry sr ON c.table_name = sr.table_name AND c.column_name = sr.column_name
            WHERE c.table_schema = 'public'
              AND t.table_type = 'BASE TABLE'
        """
        params = ()
        if table_name:
            query += " AND c.table_name = %s"
            params = (table_name,)
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        schema = {}
        for row in rows:
            t_name, c_name, d_type, desc = row
            if t_name not in schema:
                schema[t_name] = []
            schema[t_name].append({
                "column": c_name,
                "type": d_type,
                "description": desc
            })
            
        return {"tables": schema}
    finally:
        db_pool.putconn(conn)
