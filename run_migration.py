"""
Run PostgreSQL migrations for Second Brain.
Usage: python run_migration.py
"""

import asyncio
import os
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv()


async def run_migration():
    try:
        import asyncpg
    except ImportError:
        print("Installing asyncpg...")
        import subprocess
        subprocess.run(["pip", "install", "asyncpg"], check=True)
        import asyncpg
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env file")
        return False
    
    print(f"Connecting to PostgreSQL...")
    
    try:
        conn = await asyncpg.connect(database_url)
        print("Connected successfully!")
        
        # Read migration SQL
        migration_path = Path(__file__).parent / "prisma" / "migrations" / "init.sql"
        sql = migration_path.read_text()
        
        print("Running migration...")
        await conn.execute(sql)
        
        print("Migration completed successfully!")
        
        # Verify tables created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        print(f"\nTables created ({len(tables)}):")
        for t in tables:
            print(f"  - {t['table_name']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_migration())
