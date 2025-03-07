#!/bin/bash

echo "Setting up Sarv Marg application..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize the database
echo "Initializing the database..."
python3 init_db.py

echo "Setup complete! You can now run the application with:"
echo "source venv/bin/activate && python3 run.py"
