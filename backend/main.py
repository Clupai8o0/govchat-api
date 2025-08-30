"""
FastAPI server for dataset retrieval API.
"""

from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any
import retrieval

# Initialize FastAPI app
app = FastAPI(
    title="Dataset RAG API",
    description="Retrieval-Augmented Generation API for government datasets",
    version="1.0.0"
)


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return "pong"


@app.get("/query")
async def query_datasets(q: str = Query(..., description="Natural language query")) -> Dict[str, Any]:
    """
    Query datasets using natural language.
    
    Args:
        q: Natural language query string
        
    Returns:
        JSON with top-4 matching datasets
    """
    try:
        # Search for similar datasets
        hits = retrieval.search_similar_datasets(q, top_k=4)
        
        return {
            "query": q,
            "hits": hits,
            "count": len(hits)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/similar/{dataset_id}")
async def get_similar_datasets(dataset_id: str) -> Dict[str, Any]:
    """
    Find datasets similar to a given dataset ID.
    
    Args:
        dataset_id: ID of the dataset to find similar items for
        
    Returns:
        JSON with up to 3 similar datasets
    """
    try:
        # Find similar datasets
        hits = retrieval.find_similar_by_id(dataset_id, top_k=3)
        
        return {
            "dataset_id": dataset_id,
            "similar": hits,
            "count": len(hits)
        }
        
    except Exception as e:
        # Return 404 if dataset not found, 500 for other errors
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        else:
            raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
