# Sarv Marg - Smart Road Navigation Platform

Sarv Marg is a community-driven platform for real-time road blockage reporting and smart navigation. The application allows users to report road blockages with verified images and provides optimal routes based on real-time road conditions.

## Features

- **User Authentication**: Secure login and registration system with role-based access control
- **Admin Dashboard**: Comprehensive dashboard for administrators to monitor user activity and manage content
- **Real-time Map**: Interactive map showing road blockages and traffic conditions
- **Image Verification**: CNN-based verification system to ensure authenticity of reported blockages
- **Smart Navigation**: Dijkstra's algorithm for calculating optimal routes avoiding reported blockages
- **AI Analysis**: Integration with Groq AI to estimate blockage duration based on user descriptions

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Leaflet.js for maps
- **Backend**: Flask (Python web framework)
- **Database**: PostgreSQL for data storage
- **AI/ML**: TensorFlow for image verification, Groq AI for natural language processing
- **Authentication**: Flask-Login for user authentication and session management

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sarv-marg.git
   cd sarv-marg
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   SECRET_KEY=your_secret_key
   DATABASE_URL=postgresql://username:password@localhost/sarv_marg
   GROQ_API_KEY=your_groq_api_key
   ```

5. Initialize the database:
   ```
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

6. Run the application:
   ```
   python run.py
   ```

7. Access the application at `http://localhost:5000`

## Project Structure

```
sarv-marg/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── user.py
│   │   └── post.py
│   ├── routes/
│   │   ├── auth.py
│   │   ├── main.py
│   │   └── admin.py
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── images/
│   │   └── uploads/
│   ├── templates/
│   │   ├── auth/
│   │   ├── main/
│   │   └── admin/
│   └── utils/
│       ├── image_processor.py
│       ├── ai_analyzer.py
│       └── route_calculator.py
├── run.py
├── requirements.txt
└── README.md
```

## Usage

### User Features

1. **Registration and Login**: Create an account and log in to access all features
2. **Report Road Blockage**: Use the "+" button to report a road blockage
   - Allow camera and location access
   - Take a photo of the blockage
   - Add a descriptive caption
   - Submit the report
3. **Get Directions**: Use the directions button to find optimal routes
   - Set start and end locations
   - View the calculated route avoiding blockages
   - See estimated travel time and distance

### Admin Features

1. **Dashboard**: View statistics about users and posts
2. **User Management**: Manage users, assign admin privileges
3. **Post Management**: View and manage all reported blockages

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Leaflet.js for the interactive maps
- Bootstrap for the UI components
- TensorFlow for image processing capabilities
- Groq AI for natural language processing
