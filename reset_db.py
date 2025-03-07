"""
Script to reset the database and recreate tables with proper schema.
This script will drop all tables and recreate them with the updated models.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app, db
from app.models.user import User
from app.models.post import Post

def reset_database():
    """Drop all tables and recreate them"""
    app = create_app()
    
    with app.app_context():
        print("Connecting to database...")
        try:
            print("Dropping all tables...")
            db.drop_all()
            print("Creating tables with updated schema...")
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
            
            # Create test user
            print("Creating test user...")
            test_user = User(
                username='testuser',
                email='test@sarvmarg.com',
                is_admin=False
            )
            test_user.set_password('test123')
            db.session.add(test_user)
            
            # Commit the users
            db.session.commit()
            print("Users created successfully!")
            
            # Create sample post
            print("Creating sample post...")
            sample_post = Post(
                caption='Major accident on highway',
                latitude=28.6139,
                longitude=77.2090,
                estimated_blockage_time=120,
                is_authentic=True,
                user_id=2  # test_user's ID
            )
            db.session.add(sample_post)
            db.session.commit()
            print("Sample post created successfully!")
            
            # Verify database setup
            user_count = User.query.count()
            post_count = Post.query.count()
            print(f"Database contains {user_count} users and {post_count} posts.")
            
            print("Database reset and setup completed successfully!")
            
        except Exception as e:
            print(f"Error resetting database: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    reset_database()
