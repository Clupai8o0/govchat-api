# GovChat API

A simple FastAPI application with query and similar endpoints.

## Features

- `/query` endpoint: Accepts a string parameter `q` for searching
- `/similar` endpoint: Returns similar items
- Automatic API documentation with Swagger UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### GET /
Root endpoint with API information.

### GET /query
Search endpoint that accepts a query string parameter.

**Parameters:**
- `q` (required): Query string to search for

**Example:**
```
GET /query?q=hello world
```

### GET /similar
Returns a list of similar items.

**Example:**
```
GET /similar
```

## Documentation

Once the server is running, you can access:
- Swagger UI documentation: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc
