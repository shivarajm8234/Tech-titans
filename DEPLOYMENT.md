# Deploying Sarv Marg to Koyeb

This guide provides step-by-step instructions for deploying the Sarv Marg application to Koyeb, a developer-friendly serverless platform.

## Prerequisites

1. A Koyeb account (sign up at [koyeb.com](https://www.koyeb.com))
2. The Koyeb CLI installed on your machine
3. Git repository with your Sarv Marg application code

## Deployment Steps

### 1. Prepare Your Application

Ensure your application is ready for deployment:

- Make sure your `requirements.txt` file includes all necessary dependencies
- Set the `PORT` environment variable in your application to use the port provided by Koyeb
- Update the `run.py` file to bind to the correct host and port

```python
# Update in run.py
import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
```

### 2. Create a Procfile

Create a `Procfile` in the root of your project:

```
web: gunicorn run:app
```

### 3. Deploy Using the Koyeb Web Interface

1. Log in to your Koyeb account
2. Click on "Create App"
3. Choose "GitHub" as your deployment method
4. Select your repository and branch
5. Configure the following settings:
   - Name: `sarv-marg`
   - Region: Choose the region closest to your users
   - Instance Type: Select an appropriate instance type
   - Environment Variables:
     - `SECRET_KEY`: Your secret key
     - `DATABASE_URL`: Your PostgreSQL connection string
     - `GROQ_API_KEY`: Your Groq API key

6. Click "Deploy"

### 4. Deploy Using the Koyeb CLI

Alternatively, you can deploy using the Koyeb CLI:

1. Log in to the Koyeb CLI:
   ```
   koyeb login
   ```

2. Deploy your application:
   ```
   koyeb app create sarv-marg \
     --git github.com/yourusername/sarv-marg \
     --git-branch main \
     --ports 5000:http \
     --routes /:5000 \
     --env SECRET_KEY=your_secret_key \
     --env DATABASE_URL=your_database_url \
     --env GROQ_API_KEY=your_groq_api_key
   ```

### 5. Set Up a PostgreSQL Database

For the database, you have several options:

1. **Use Koyeb's Database Service**:
   - Go to the Koyeb dashboard
   - Create a new PostgreSQL database
   - Connect your app to the database using the provided connection string

2. **Use an External Database Provider** (recommended for production):
   - Set up a PostgreSQL database with a provider like Supabase, Neon, or AWS RDS
   - Update your `DATABASE_URL` environment variable with the connection string

### 6. Initialize the Database

After deployment, you'll need to initialize your database:

1. Connect to your Koyeb app's shell:
   ```
   koyeb service shell sarv-marg
   ```

2. Run the database initialization script:
   ```
   python init_db.py
   ```

### 7. Verify Deployment

1. Access your application at the URL provided by Koyeb (typically `https://sarv-marg-yourusername.koyeb.app`)
2. Verify that all features are working correctly
3. Check the logs for any errors:
   ```
   koyeb service logs sarv-marg
   ```

## Continuous Deployment

Koyeb automatically sets up continuous deployment from your GitHub repository. Any changes pushed to your main branch will trigger a new deployment.

## Scaling

To scale your application:

1. Go to the Koyeb dashboard
2. Select your `sarv-marg` application
3. Click on "Scale"
4. Adjust the number of instances or instance type as needed

## Troubleshooting

If you encounter issues with your deployment:

1. Check the application logs for error messages
2. Verify that all environment variables are set correctly
3. Ensure your database connection is working properly
4. Check that your application is binding to the correct port

For more help, refer to the [Koyeb documentation](https://www.koyeb.com/docs) or contact Koyeb support.
