"""
Comprehensive database setup script for Sarv Marg application.
This script will:
1. Create all necessary tables
2. Add sample users and posts
3. Set up initial data for testing
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time

# Load environment variables from .env file
load_dotenv()

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app, db
from app.models.user import User
from app.models.post import Post

def create_upload_directory():
    """Create upload directory for images if it doesn't exist"""
    uploads_dir = os.path.join(os.path.dirname(__file__), 'app', 'static', 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created uploads directory: {uploads_dir}")
    else:
        print(f"Uploads directory already exists: {uploads_dir}")
    return uploads_dir

def create_database_if_not_exists():
    """Create the PostgreSQL database if it doesn't exist"""
    db_url = os.environ.get('DATABASE_URL')
    db_name = db_url.split('/')[-1]
    
    # Create a connection string to the postgres database
    conn_string = db_url.replace(db_name, 'postgres')
    
    try:
        # Connect to the postgres database
        conn = psycopg2.connect(conn_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Create a cursor
        cur = conn.cursor()
        
        # Check if our database exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()
        
        if not exists:
            print(f"Creating database {db_name}...")
            cur.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created successfully!")
        else:
            print(f"Database {db_name} already exists.")
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def setup_database():
    """Set up the database with tables and sample data"""
    app = create_app()
    
    with app.app_context():
        print("Setting up database tables...")
        try:
            # Create all tables
            db.create_all()
            print("Tables created successfully!")
            
            # Check if admin user exists
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                print("Creating admin user...")
                admin = User(
                    username='admin',
                    email='admin@sarvmarg.com',
                    is_admin=True
                )
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                print("Admin user created successfully!")
            
            # Check if test user exists
            test_user = User.query.filter_by(username='testuser').first()
            if not test_user:
                print("Creating test user...")
                test_user = User(
                    username='testuser',
                    email='test@sarvmarg.com',
                    is_admin=False
                )
                test_user.set_password('test123')
                db.session.add(test_user)
                db.session.commit()
                print("Test user created successfully!")
            
            # Check if sample post exists
            sample_post = Post.query.filter_by(caption='Major accident on highway').first()
            if not sample_post:
                print("Creating sample post...")
                sample_post = Post(
                    caption='Major accident on highway',
                    latitude=28.6139,
                    longitude=77.2090,
                    estimated_blockage_time=120,
                    is_authentic=True,
                    user_id=test_user.id
                )
                db.session.add(sample_post)
                db.session.commit()
                print("Sample post created successfully!")
            
            # Check if tables were created by querying them
            user_count = User.query.count()
            post_count = Post.query.count()
            print(f"Database contains {user_count} users and {post_count} posts.")
            
            print("Database setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    print("Starting Sarv Marg database setup...")
    
    # Create uploads directory
    create_upload_directory()
    
    # Create database if it doesn't exist
    if create_database_if_not_exists():
        # Wait a moment for the database to be ready
        time.sleep(1)
        
        # Set up database tables and sample data
        if setup_database():
            print("\n===== DATABASE SETUP COMPLETE =====")
            print("You can now run the application with 'python run.py'")
        else:
            print("\n===== DATABASE SETUP FAILED =====")
            print("Please check the error messages above and try again.")
    else:
        print("\n===== DATABASE CREATION FAILED =====")
        print("Please check your PostgreSQL configuration and try again.")
