"""
Simple script to test PostgreSQL connection directly.
This bypasses the Flask application and tests the connection string directly.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_connection():
    """Test direct connection to PostgreSQL"""
    # Get database URL from environment
    db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        return False
    
    print(f"Attempting to connect using: {db_url}")
    
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(db_url)
        
        # Create a cursor
        cur = conn.cursor()
        
        # Execute a test query
        cur.execute('SELECT version();')
        
        # Get the database version
        db_version = cur.fetchone()
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        print(f"Connection successful! PostgreSQL version: {db_version[0]}")
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == '__main__':
    success = test_connection()
    if success:
        print("Database connection test passed!")
    else:
        print("Database connection test failed!")
