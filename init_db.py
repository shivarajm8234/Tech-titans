"""
Database initialization script for Sarv Marg application.
This script creates the initial database tables, admin user, and sample data.
Uses the PostgreSQL database connection specified in the .env file.
"""

import os
import sys
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app, db
from app.models.user import User
from app.models.post import Post

def init_db():
    """Initialize the database with tables and sample data"""
    app = create_app()
    
    with app.app_context():
        # Create all tables
        print("Creating database tables...")
        db.create_all()
        print("Tables created successfully!")
        
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("Creating admin user...")
            admin = User(
                username='admin',
                email='admin@sarvmarg.com',
                is_admin=True,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            admin.set_password('admin123')
            db.session.add(admin)
        
        # Create a regular test user if it doesn't exist
        test_user = User.query.filter_by(username='testuser').first()
        if not test_user:
            print("Creating test user...")
            test_user = User(
                username='testuser',
                email='test@sarvmarg.com',
                is_admin=False,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            test_user.set_password('test123')
            db.session.add(test_user)
        
        # Commit users to get their IDs
        db.session.commit()
        
        # Create sample posts if there are none
        if Post.query.count() == 0:
            print("Creating sample posts...")
            
            # Create directory for sample images if it doesn't exist
            upload_folder = os.path.join(app.root_path, 'static/uploads')
            os.makedirs(upload_folder, exist_ok=True)
            
            # Sample post data
            sample_posts = [
                {
                    'caption': 'Major accident on highway, multiple vehicles involved. Traffic completely blocked.',
                    'latitude': 28.6139,
                    'longitude': 77.2090,
                    'estimated_blockage_time': 120,
                    'is_authentic': True,
                    'user_id': test_user.id
                },
                {
                    'caption': 'Road construction work in progress. One lane closed, expect delays.',
                    'latitude': 28.6304,
                    'longitude': 77.2177,
                    'estimated_blockage_time': 180,
                    'is_authentic': True,
                    'user_id': test_user.id
                },
                {
                    'caption': 'Fallen tree blocking the road. Municipal workers are clearing it.',
                    'latitude': 28.6229,
                    'longitude': 77.2080,
                    'estimated_blockage_time': 60,
                    'is_authentic': True,
                    'user_id': admin.id
                },
                {
                    'caption': 'Minor traffic jam due to VIP movement. Should clear in 30 minutes.',
                    'latitude': 28.6129,
                    'longitude': 77.2295,
                    'estimated_blockage_time': 30,
                    'is_authentic': True,
                    'user_id': admin.id
                },
                {
                    'caption': 'Water logging after heavy rain. Road partially submerged.',
                    'latitude': 28.5921,
                    'longitude': 77.2290,
                    'estimated_blockage_time': 240,
                    'is_authentic': True,
                    'user_id': test_user.id
                }
            ]
            
            # Create posts with timestamps spread over the last week
            now = datetime.utcnow()
            for i, post_data in enumerate(sample_posts):
                # Create a timestamp between now and 7 days ago
                days_ago = random.randint(0, 6)
                hours_ago = random.randint(0, 23)
                timestamp = now - timedelta(days=days_ago, hours=hours_ago)
                
                # Create image path (in a real app, these would be actual images)
                image_path = f'uploads/sample_blockage_{i+1}.jpg'
                
                # Create post
                post = Post(
                    image_path=image_path,
                    caption=post_data['caption'],
                    latitude=post_data['latitude'],
                    longitude=post_data['longitude'],
                    estimated_blockage_time=post_data['estimated_blockage_time'],
                    is_authentic=post_data['is_authentic'],
                    created_at=timestamp,
                    user_id=post_data['user_id']
                )
                db.session.add(post)
            
            # Commit all posts
            db.session.commit()
            print("Sample data created successfully!")
        else:
            print("Database already contains posts, skipping sample data creation.")

if __name__ == '__main__':
    init_db()
    print("Database initialization completed!")
