from app import db
from datetime import datetime

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=True)
    caption = db.Column(db.Text, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    estimated_blockage_time = db.Column(db.Integer)  # Estimated time in minutes
    is_authentic = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    upload_time = db.Column(db.Float, nullable=True)  # Time taken to upload in seconds
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_analysis = db.Column(db.Text, nullable=True)  # JSON string with image analysis data
    blockage_analysis = db.Column(db.Text, nullable=True)  # JSON string with blockage analysis from Groq
    post_metadata = db.Column(db.Text, nullable=True)  # JSON string with additional metadata
    
    def __repr__(self):
        return f'<Post {self.id}>'
    
    def to_dict(self):
        from flask_login import current_user
        
        # Parse post metadata if available
        metadata = {}
        if self.post_metadata:
            try:
                import json
                metadata = json.loads(self.post_metadata)
            except:
                pass
                
        return {
            'id': self.id,
            'image_path': self.image_path,
            'caption': self.caption,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'estimated_blockage_time': self.estimated_blockage_time,
            'is_authentic': self.is_authentic,
            'created_at': self.created_at.isoformat(),
            'upload_time': self.upload_time,
            'user_id': self.user_id,
            'has_analysis': bool(self.blockage_analysis),  # Flag to indicate if detailed analysis is available
            'metadata': metadata,
            'is_owner': current_user.is_authenticated and current_user.id == self.user_id
        }
