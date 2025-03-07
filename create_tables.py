"""
Script to create database tables directly using SQLAlchemy.
This bypasses Flask-Migrate and directly creates the tables in the database.
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

def create_tables():
    """Create database tables directly using SQLAlchemy"""
    app = create_app()
    
    with app.app_context():
        print("Creating database tables...")
        try:
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
            
            # Check if tables were created by querying them
            user_count = User.query.count()
            post_count = Post.query.count()
            print(f"Database contains {user_count} users and {post_count} posts.")
            
            print("Database setup completed successfully!")
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    create_tables()
