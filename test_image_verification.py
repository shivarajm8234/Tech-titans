#!/usr/bin/env python3
import os
import sys
from app.utils.image_processor import verify_image_authenticity

def test_image_verification():
    """
    Test the image verification function on sample images
    """
    # Get all sample images from the uploads directory
    upload_dir = "app/static/uploads"
    
    if not os.path.exists(upload_dir):
        print(f"Error: Upload directory {upload_dir} does not exist")
        return
    
    image_files = [f for f in os.listdir(upload_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {upload_dir}")
        return
    
    print(f"Testing {len(image_files)} images...")
    
    success_count = 0
    fail_count = 0
    
    for img_file in image_files:
        img_path = os.path.join(upload_dir, img_file)
        is_authentic, error_msg = verify_image_authenticity(img_path)
        
        if is_authentic:
            print(f"✅ {img_file}: Verified successfully")
            success_count += 1
        else:
            print(f"❌ {img_file}: Failed verification - {error_msg}")
            fail_count += 1
    
    print(f"\nSummary: {success_count} passed, {fail_count} failed")

if __name__ == "__main__":
    test_image_verification()
