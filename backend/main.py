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
        JSON with top-4 matching datasets, friendly GPT response, trust score, and sources
    """
    try:
        # Handle empty query case
        if not q.strip():
            return {
                "query": q,
                "answer": "Please provide a specific query to search for datasets.",
                "sources": [],
                "trust": {"score": 0.0, "factors": [], "checks": [], "audit_id": ""},
                "hits": [],
                "count": 0
            }
        
        # Search for similar datasets
        hits = retrieval.search_similar_datasets(q, top_k=4)
        
        # Handle no results case
        if not hits:
            trust_score = retrieval.calculate_trust_score("", [], [], metadata_only=True)
            retrieval.create_audit_record(
                trust_score["audit_id"], q, 
                "No datasets found matching your query. Please try refining your search terms.",
                [], trust_score
            )
            return {
                "query": q,
                "answer": "No datasets found matching your query. Please try refining your search terms.",
                "sources": [],
                "trust": trust_score,
                "hits": [],
                "count": 0
            }
        
        # Generate friendly response
        friendly_response = retrieval.generate_friendly_response(q, hits)
        
        # Enforce metadata grounding
        clean_answer = retrieval.enforce_metadata_grounding(friendly_response, hits)
        
        # Extract sources
        sources = retrieval.extract_sources(hits)
        
        # Calculate trust score
        trust_score = retrieval.calculate_trust_score(clean_answer, hits, sources, metadata_only=True)
        
        # Create audit record
        retrieval.create_audit_record(trust_score["audit_id"], q, clean_answer, hits, trust_score)
        
        return {
            "query": q,
            "answer": clean_answer,
            "sources": sources,
            "trust": trust_score,
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
        JSON with up to 3 similar datasets, friendly GPT response, trust score, and sources
    """
    try:
        # Find similar datasets
        hits = retrieval.find_similar_by_id(dataset_id, top_k=3)
        
        # Handle no results case
        if not hits:
            trust_score = retrieval.calculate_trust_score("", [], [], metadata_only=True)
            retrieval.create_audit_record(
                trust_score["audit_id"], f"similar:{dataset_id}",
                f"No similar datasets found for {dataset_id}.",
                [], trust_score
            )
            return {
                "dataset_id": dataset_id,
                "answer": f"No similar datasets found for {dataset_id}.",
                "sources": [],
                "trust": trust_score,
                "similar": [],
                "count": 0
            }
        
        # Generate friendly response
        friendly_response = retrieval.generate_similar_response(dataset_id, hits)
        
        # Enforce metadata grounding
        clean_answer = retrieval.enforce_metadata_grounding(friendly_response, hits)
        
        # Extract sources
        sources = retrieval.extract_sources(hits)
        
        # Calculate trust score
        trust_score = retrieval.calculate_trust_score(clean_answer, hits, sources, metadata_only=True)
        
        # Create audit record
        retrieval.create_audit_record(
            trust_score["audit_id"], f"similar:{dataset_id}", 
            clean_answer, hits, trust_score
        )
        
        return {
            "dataset_id": dataset_id,
            "answer": clean_answer,
            "sources": sources,
            "trust": trust_score,
            "similar": hits,
            "count": len(hits)
        }
        
    except Exception as e:
        # Return 404 if dataset not found, 500 for other errors
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        else:
            raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


@app.get("/audit/{audit_id}")
async def get_audit_record(audit_id: str) -> Dict[str, Any]:
    """
    Retrieve audit record by ID for transparency.
    
    Args:
        audit_id: Audit record identifier
        
    Returns:
        JSON with audit details
    """
    try:
        audit_record = retrieval.get_audit_record(audit_id)
        if not audit_record:
            raise HTTPException(status_code=404, detail=f"Audit record {audit_id} not found")
        return audit_record
        
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Audit record {audit_id} not found")
        else:
            raise HTTPException(status_code=500, detail=f"Audit retrieval failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
