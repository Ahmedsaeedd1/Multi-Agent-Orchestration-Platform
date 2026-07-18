import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_registry (
            table_name TEXT,
            column_name TEXT,
            data_type TEXT,
            description TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name TEXT,
            price NUMERIC
        )
    """)
    
    # Clear existing sample data if we want it to be idempotent
    cursor.execute("TRUNCATE TABLE schema_registry")
    cursor.execute("TRUNCATE TABLE products")
    
    sample_data = [
        ('schema_registry', 'table_name', 'TEXT', 'Name of the database table'),
        ('schema_registry', 'column_name', 'TEXT', 'Name of the column in the table'),
        ('schema_registry', 'data_type', 'TEXT', 'Data type of the column'),
        ('schema_registry', 'description', 'TEXT', 'Human-readable description of the column purpose'),
        ('products', 'id', 'INTEGER', 'Primary key for products'),
        ('products', 'name', 'TEXT', 'Name of the product'),
        ('products', 'price', 'NUMERIC', 'Price of the product')
    ]
    
    cursor.executemany("""
        INSERT INTO schema_registry (table_name, column_name, data_type, description)
        VALUES (%s, %s, %s, %s)
    """, sample_data)
    
    product_data = [
        ('Laptop', 999.99),
        ('Mouse', 25.50),
        ('Keyboard', 45.00)
    ]
    
    cursor.executemany("""
        INSERT INTO products (name, price)
        VALUES (%s, %s)
    """, product_data)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"OK — Postgres schema initialized at {host}:{port}/{db}")

if __name__ == "__main__":
    init_db()
