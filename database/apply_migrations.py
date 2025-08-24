import psycopg2
from dotenv import load_dotenv
import os
from pathlib import Path

def run_migration():
    """
    Apply SQL migrations to create the glovo_ai schema and products table
    with pgvector support for hybrid search.
    """
    
    # Load environment variables from .env
    load_dotenv()
    
    # Database connection parameters
    db_config = {
        'user': os.getenv("user"),
        'password': os.getenv("password"),
        'host': os.getenv("host"),
        'port': os.getenv("port"),
        'dbname': os.getenv("dbname")
    }
    
    # Migration files in execution order
    migration_files = [
        "database/migrations/00_enable_extensions.sql",
        "database/migrations/01_create_schema.sql", 
        "database/migrations/02_tables.sql",
        "database/migrations/03_indexes_hnsw.sql",
        "database/migrations/04_security_basics.sql"
    ]
    
    connection = None
    cursor = None
    
    try:
        # Connect to PostgreSQL
        print("Connecting to database...")
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        
        # Set autocommit for DDL operations
        connection.autocommit = True
        
        print("Database connection successful!")
        
        # Execute each migration file
        for migration_file in migration_files:
            print(f"\nExecuting {migration_file}...")
            
            # Read SQL file content
            sql_path = Path(migration_file)
            if not sql_path.exists():
                print(f"Warning: {migration_file} not found, skipping...")
                continue
                
            with open(sql_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Skip empty files or files with only comments/whitespace
            sql_content_clean = sql_content.strip()
            if not sql_content_clean or all(line.strip().startswith('--') or line.strip() == '' 
                                          for line in sql_content_clean.split('\n')):
                print(f"✓ {migration_file} skipped (no executable SQL)")
                continue
            
            # Execute the SQL
            try:
                cursor.execute(sql_content)
                print(f"✓ {migration_file} executed successfully")
            except psycopg2.Error as e:
                print(f"✗ Error executing {migration_file}: {e}")
                raise
        
        # Verify the setup
        print("\nVerifying setup...")
        
        # Check if vector extension is available
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print("✓ pgvector extension enabled")
        else:
            print("✗ pgvector extension not found")
        
        # Check if schema exists
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'glovo_ai';")
        if cursor.fetchone():
            print("✓ glovo_ai schema created")
        else:
            print("✗ glovo_ai schema not found")
        
        # Check if table exists
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'glovo_ai' AND table_name = 'products';
        """)
        if cursor.fetchone():
            print("✓ glovo_ai.products table created")
        else:
            print("✗ glovo_ai.products table not found")
        
        # Check HNSW indexes
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE schemaname = 'glovo_ai' AND tablename = 'products'
            AND indexname LIKE '%hnsw%';
        """)
        indexes = cursor.fetchall()
        if len(indexes) >= 2:
            print(f"✓ HNSW indexes created: {[idx[0] for idx in indexes]}")
        else:
            print(f"⚠ Only {len(indexes)} HNSW indexes found (expected 2)")
        
        print("\n🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Load sample data into glovo_ai.products")
        print("2. Generate embeddings for text_emb and image_emb columns")
        print("3. Test hybrid search queries")
        
    except psycopg2.Error as db_error:
        print(f"\n❌ Database error: {db_error}")
        if connection:
            connection.rollback()
        raise
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
        
    finally:
        # Clean up connections
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("\nDatabase connection closed.")

if __name__ == "__main__":
    run_migration()
