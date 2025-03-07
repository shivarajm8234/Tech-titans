"""
Generate sample images for road blockages in the Sarv Marg application.
This script creates simple images with text to simulate road blockage photos.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import random

def generate_sample_images():
    """Generate sample road blockage images"""
    # Create uploads directory if it doesn't exist
    upload_dir = os.path.join('app', 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Sample blockage types
    blockage_types = [
        "Accident",
        "Construction",
        "Fallen Tree",
        "Traffic Jam",
        "Water Logging"
    ]
    
    # Generate 5 sample images
    for i in range(5):
        # Create a blank image with random background color
        img_width, img_height = 800, 600
        bg_color = (
            random.randint(200, 255),  # R
            random.randint(200, 255),  # G
            random.randint(200, 255)   # B
        )
        img = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw blockage type text
        blockage_text = blockage_types[i]
        text_width = draw.textlength(blockage_text, font=font)
        draw.text(
            ((img_width - text_width) / 2, 100),
            blockage_text,
            fill=(0, 0, 0),
            font=font
        )
        
        # Draw "Sample Image" text
        sample_text = "Sample Road Blockage Image"
        sample_width = draw.textlength(sample_text, font=font)
        draw.text(
            ((img_width - sample_width) / 2, 200),
            sample_text,
            fill=(0, 0, 0),
            font=font
        )
        
        # Draw a rectangle to represent a road
        draw.rectangle(
            [(100, 300), (700, 400)],
            fill=(100, 100, 100),
            outline=(255, 255, 255)
        )
        
        # Draw blockage symbol
        if blockage_text == "Accident":
            # Draw car accident symbol
            draw.ellipse([(350, 320), (450, 380)], fill=(255, 0, 0))
            draw.line([(300, 350), (500, 350)], fill=(255, 0, 0), width=5)
        elif blockage_text == "Construction":
            # Draw construction cones
            for x in range(300, 501, 50):
                draw.polygon(
                    [(x, 320), (x+20, 320), (x+10, 380)],
                    fill=(255, 165, 0)
                )
        elif blockage_text == "Fallen Tree":
            # Draw fallen tree
            draw.rectangle([(350, 330), (450, 350)], fill=(139, 69, 19))
            draw.ellipse([(370, 300), (430, 340)], fill=(0, 128, 0))
        elif blockage_text == "Traffic Jam":
            # Draw traffic jam (multiple small rectangles)
            for x in range(200, 601, 60):
                draw.rectangle(
                    [(x, 330), (x+40, 370)],
                    fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                )
        elif blockage_text == "Water Logging":
            # Draw water waves
            for y in range(320, 381, 15):
                for x in range(150, 651, 100):
                    draw.arc(
                        [(x, y), (x+50, y+10)],
                        0, 180,
                        fill=(0, 0, 255),
                        width=2
                    )
        
        # Add coordinates at the bottom right with the last digit in light red
        coords = f"28.61{i}9, 77.20{i}0"
        coords_width = draw.textlength(coords, font=small_font)
        
        # Draw all characters except the last one in dark red
        draw.text(
            ((img_width - coords_width), img_height - 30),
            coords[:-1],
            fill=(220, 53, 69),  # Dark red
            font=small_font
        )
        
        # Draw the last character in light red (for authenticity verification)
        last_char = coords[-1]
        last_char_width = draw.textlength(last_char, font=small_font)
        draw.text(
            ((img_width - last_char_width), img_height - 30),
            last_char,
            fill=(220, 53, 69, 128),  # Light red (semi-transparent)
            font=small_font
        )
        
        # Save the image
        file_path = os.path.join(upload_dir, f"sample_blockage_{i+1}.jpg")
        img.save(file_path)
        print(f"Generated sample image: {file_path}")

if __name__ == "__main__":
    generate_sample_images()
    print("Sample images generated successfully!")
