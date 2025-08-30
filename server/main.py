from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI(
    title="GovChat API",
    description="A simple FastAPI application with query and similar endpoints",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with basic information about the API."""
    return {
        "message": "Welcome to GovChat API",
        "endpoints": {
            "/query": "Search with query parameter 'q'",
            "/similar": "Find similar items"
        }
    }

@app.get("/query")
async def query_endpoint(q: str = Query(..., description="Query string parameter")):
    """
    Query endpoint that accepts a string parameter 'q'.
    
    Args:
        q: The query string to search for
    
    Returns:
        dict: Response containing the query and placeholder results
    """
    return {
        "query": q,
        "results": f"Search results for: {q}",
        "status": "success",
        "count": 1
    }

@app.get("/similar")
async def similar_endpoint():
    """
    Similar endpoint for finding similar items.
    
    Returns:
        dict: Response with similar items data
    """
    return {
        "similar_items": [
            {"id": 1, "title": "Sample Item 1", "similarity": 0.95},
            {"id": 2, "title": "Sample Item 2", "similarity": 0.87},
            {"id": 3, "title": "Sample Item 3", "similarity": 0.76}
        ],
        "status": "success",
        "count": 3
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
