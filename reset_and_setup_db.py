"""
Reset and setup database for Sarv Marg application.
This script will:
1. Drop all existing tables
2. Create all tables from scratch
3. Add sample users and posts
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
    
    # Create a sample image file for testing
    sample_image_path = os.path.join(uploads_dir, 'sample_blockage.jpg')
    if not os.path.exists(sample_image_path):
        # Create an empty file
        with open(sample_image_path, 'w') as f:
            f.write("Sample image file")
        print(f"Created sample image file: {sample_image_path}")
    
    return uploads_dir, 'sample_blockage.jpg'

def reset_and_setup_database():
    """Reset and set up the database with tables and sample data"""
    app = create_app()
    
    with app.app_context():
        print("Dropping all existing tables...")
        try:
            db.drop_all()
            print("All tables dropped successfully!")
            
            print("Creating tables with fresh schema...")
            db.create_all()
            print("Tables created successfully!")
            
            # Create admin user
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
            
            # Create test user
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
            
            # Create sample post with image path
            _, sample_image = create_upload_directory()
            image_path = f'uploads/{sample_image}'
            
            print("Creating sample post...")
            sample_post = Post(
                image_path=image_path,
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
            
            print("Database reset and setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error resetting and setting up database: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    print("Starting Sarv Marg database reset and setup...")
    
    # Create uploads directory and sample image
    create_upload_directory()
    
    # Reset and set up database
    if reset_and_setup_database():
        print("\n===== DATABASE RESET AND SETUP COMPLETE =====")
        print("You can now run the application with 'python run.py'")
        print("\nAdmin credentials:")
        print("Username: admin")
        print("Password: admin123")
        print("\nTest user credentials:")
        print("Username: testuser")
        print("Password: test123")
    else:
        print("\n===== DATABASE RESET AND SETUP FAILED =====")
        print("Please check the error messages above and try again.")
