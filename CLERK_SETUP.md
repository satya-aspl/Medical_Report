# Clerk Multi-User Setup Guide

This application now supports multiple users with Clerk authentication. Each user's data is isolated and secure.

## Prerequisites

1. Create a free Clerk account at [clerk.com](https://clerk.com)
2. Create a new Clerk application in your dashboard

## Environment Configuration

Add these keys to your `.env` file:

```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_[your-publishable-key]
CLERK_SECRET_KEY=sk_test_[your-secret-key]
```

You can find these keys in your Clerk Dashboard under **API Keys**.

## Database Reset (for existing installations)

If you have an existing `health_data.db` file with data, you have two options:

### Option 1: Keep Existing Data (Manual Migration)
The schema has changed to include `user_id`. You'll need to:
1. Back up your existing database
2. Manually add the `user_id` column to existing tables
3. Update all existing records with a test user_id

### Option 2: Fresh Start (Recommended for Testing)
1. Stop the app if running
2. Delete the existing `health_data.db` file
3. Start the app - a new database with the correct schema will be created automatically

## Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:5000`

## User Flow

1. **Login**: Users click "Login" and authenticate via Clerk
2. **Upload**: Users can upload PDF lab reports
3. **Data Isolation**: Each user only sees their own reports and data
4. **Analysis**: AI-powered analysis is performed on user-specific data

## API Endpoints

All endpoints except `/login` and `/api/investigations` (when called for the first time) require authentication via JWT token in the `Authorization: Bearer [token]` header.

### Protected Endpoints:
- `GET /api/investigations` - Get user's investigation list
- `GET /api/investigation/<name>` - Get specific investigation data
- `POST /upload` - Upload new lab report
- `POST /assessment` - Get AI assessment for investigation

## Frontend Authentication

The frontend automatically:
1. Loads Clerk.js SDK
2. Checks if user is logged in
3. Retrieves JWT token for API calls
4. Passes token to all API requests
5. Handles session expiration and re-authentication

## Troubleshooting

### Database Errors
If you get database schema errors, delete `health_data.db` and restart the app.

### Authentication Errors (401)
- Check that `CLERK_SECRET_KEY` is set correctly in `.env`
- Clear browser cache and re-login
- Verify the JWT token is being sent with API requests

### CORS Errors
The app is configured to accept requests from `localhost:3000` and `localhost:5000`. Update CORS settings in `app.py` if using a different frontend domain.
