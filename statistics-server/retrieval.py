"""
Vector search functions for dataset retrieval.
"""

import os
import json
import time
import re
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_store")


def get_collection():
    """Get the datasets collection from ChromaDB."""
    try:
        collection = chroma_client.get_collection("datasets")
        return collection
    except Exception as e:
        raise Exception(f"Failed to get ChromaDB collection: {e}")


def embed_text(text: str) -> List[float]:
    """Create embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Failed to create embedding: {e}")


def generate_friendly_response(query: str, hits: List[Dict[str, Any]]) -> str:
    """Generate a friendly GPT response for the query and results."""
    try:
        # Build context from the top results
        context_parts = []
        for i, hit in enumerate(hits, 1):  # Use top 3 for context
            context_parts.append(f"{i}. {hit['title']} - {hit['agency']}")
        
        context = "\n".join(context_parts)
        
        # Create GPT prompt
        prompt = f"""You are a helpful government data assistant. A user asked: "{query}"

I found these relevant datasets:
{context}

Write a brief, friendly response (2-3 sentences) that:
1. Acknowledges their query
2. Mentions how many datasets were found
3. Briefly describes what kind of data is available

Keep it conversational and helpful."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback response if GPT fails
        return f"I found {len(hits)} datasets related to '{query}'. Here are the most relevant results:"


def search_similar_datasets(query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Search for datasets similar to query text.
    
    Args:
        query_text: Natural language query
        top_k: Number of results to return
        
    Returns:
        List of dataset hits with id, metadata, and similarity score
    """
    try:
        # Get embedding for query
        query_embedding = embed_text(query_text)
        
        # Search in ChromaDB
        collection = get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        hits = []
        for i in range(len(results["ids"][0])):
            # Convert ChromaDB distance to similarity score
            distance = max(0.0, float(results["distances"][0][i]))
            # For normalized embeddings: cosine_similarity ≈ 1 - (L2_distance² / 2)
            similarity_score = max(0.0, min(1.0, 1.0 - (distance * distance / 2.0)))
            
            hit = {
                "id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "description": results["metadatas"][0][i]["description"],
                "agency": results["metadatas"][0][i]["agency"],
                "api_url": results["metadatas"][0][i]["api_url"],
                "url": results["metadatas"][0][i].get("url", ""),
                "similarity_score": similarity_score
            }
            hits.append(hit)
        
        return hits
        
    except Exception as e:
        raise Exception(f"Search failed: {e}")


def generate_similar_response(dataset_id: str, hits: List[Dict[str, Any]]) -> str:
    """Generate a friendly GPT response for similar dataset results."""
    try:
        # Build context from the similar results
        context_parts = []
        for i, hit in enumerate(hits[:3], 1):  # Use top 3 for context
            context_parts.append(f"{i}. {hit['title']} - {hit['agency']}")
        
        context = "\n".join(context_parts)
        
        # Create GPT prompt
        prompt = f"""You are a helpful government data assistant. A user asked for datasets similar to dataset ID: "{dataset_id}"

I found these similar datasets:
{context}

Write a brief, friendly response (2-3 sentences) that:
1. Acknowledges they're looking for similar datasets
2. Mentions how many similar datasets were found
3. Briefly explains what makes these datasets related or useful together

Keep it conversational and helpful."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback response if GPT fails
        return f"I found {len(hits)} datasets similar to '{dataset_id}'. These datasets share related themes and might be useful for your analysis:"


def extract_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract clean source list from hits."""
    sources = []
    for hit in hits:
        source = {
            "title": hit.get("title", ""),
            "agency": hit.get("agency", ""),
            "api_url": hit.get("api_url", ""),
            "url": hit.get("url", ""),
            "similarity": round(hit.get("similarity_score", 0.0), 3)
        }
        sources.append(source)
    return sources


def detect_numeric_claims(text: str) -> List[str]:
    """Detect numeric claims in text (for future numeric mode)."""
    # Simple regex for numbers with optional units
    numeric_pattern = r'\b\d+(?:\.\d+)?(?:\s*[%$]|\s*(?:million|billion|thousand|index|rate|percent))\b'
    claims = re.findall(numeric_pattern, text, re.IGNORECASE)
    return claims


def calculate_trust_score(
    answer: str, 
    hits: List[Dict[str, Any]], 
    sources: List[Dict[str, Any]],
    metadata_only: bool = True
) -> Dict[str, Any]:
    """
    Calculate trust score using geometric blend of factors.
    
    Returns trust object with score, factors, checks, and audit_id.
    """
    # Factor weights (sum to 1.0)
    weights = {
        "grounding": 0.30,
        "provenance": 0.20, 
        "retrieval": 0.20,
        "verification": 0.20,
        "recency": 0.10
    }
    
    # Calculate individual factors
    factors = {}
    
    # Grounding (g): 1.0 if answer uses only metadata, 0.0 if fabricated
    numeric_claims = detect_numeric_claims(answer)
    if metadata_only and len(numeric_claims) > 0:
        factors["grounding"] = 0.0  # Numbers detected in metadata-only mode
    else:
        factors["grounding"] = 1.0  # Clean metadata-only answer
    
    # Provenance (p): 1.0 if sources exist with valid items
    if sources and len(sources) > 0 and sources[0].get("title"):
        factors["provenance"] = 1.0
    else:
        factors["provenance"] = 0.0
    
    # Retrieval (r): max similarity among hits, clamped at 0.90 -> 1.0
    if hits and len(hits) > 0:
        max_sim = max(hit.get("similarity_score", 0.0) for hit in hits)
        factors["retrieval"] = min(1.0, max_sim / 0.90) if max_sim >= 0.90 else max_sim
    else:
        factors["retrieval"] = 0.0
    
    # Verification (v): 1.0 in metadata-only mode
    if metadata_only:
        factors["verification"] = 1.0
    else:
        # Future: implement numeric verification
        factors["verification"] = 0.0
    
    # Recency (c): 1.0 for now (no freshness tracking yet)
    factors["recency"] = 1.0
    
    # Clamp all factors to [0,1]
    for key in factors:
        factors[key] = max(0.0, min(1.0, factors[key]))
    
    # Calculate geometric mean: score = Π (factor^weight)
    score = 1.0
    factor_list = []
    for factor_name, weight in weights.items():
        factor_value = factors[factor_name]
        score *= (factor_value ** weight)
        factor_list.append({"name": factor_name, "value": round(factor_value, 3)})
    
    # Round final score to 3 decimals
    score = round(score, 3)
    
    # Checks (empty for metadata-only mode)
    checks = []
    
    # Generate audit ID
    audit_id = str(uuid.uuid4())[:8]
    
    return {
        "score": score,
        "factors": factor_list,
        "checks": checks,
        "audit_id": audit_id
    }


def create_audit_record(
    audit_id: str,
    query: str,
    answer: str,
    hits: List[Dict[str, Any]],
    trust: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Create and persist audit record."""
    audit_record = {
        "audit_id": audit_id,
        "timestamp": int(time.time()),
        "query": query,
        "answer": answer,
        "raw_hits": hits,
        "trust": trust,
        "extra": extra or {}
    }
    
    # Ensure audit directory exists
    os.makedirs("audit_logs", exist_ok=True)
    
    # Save audit record
    audit_file = f"audit_logs/{audit_id}.json"
    with open(audit_file, "w") as f:
        json.dump(audit_record, f, indent=2)


def get_audit_record(audit_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve audit record by ID."""
    audit_file = f"audit_logs/{audit_id}.json"
    try:
        with open(audit_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def enforce_metadata_grounding(answer: str, hits: List[Dict[str, Any]]) -> str:
    """
    Enforce grounding rules by checking answer against metadata.
    Strip or replace problematic content.
    """
    # Remove explicit numbers in metadata-only mode
    # Keep qualitative terms like "latest", "recent", "comprehensive"
    cleaned_answer = re.sub(
        r'\b\d+(?:\.\d+)?(?:\s*[%$]|\s*(?:million|billion|thousand))\b',
        '[data available]',
        answer,
        flags=re.IGNORECASE
    )
    
    return cleaned_answer


def find_similar_by_id(dataset_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Find datasets similar to a given dataset ID.
    
    Args:
        dataset_id: ID of the dataset to find similar items for
        top_k: Number of similar results to return (excluding the dataset itself)
        
    Returns:
        List of similar dataset hits
    """
    try:
        collection = get_collection()
        
        # Get the target dataset's document text
        target_result = collection.get(
            ids=[dataset_id],
            include=["documents"]
        )
        
        if not target_result["documents"]:
            raise Exception(f"Dataset with ID {dataset_id} not found")
        
        target_text = target_result["documents"][0]
        
        # Create embedding for the target dataset's text
        target_embedding = embed_text(target_text)
        
        # Search for similar datasets (get more than needed to filter out self)
        results = collection.query(
            query_embeddings=[target_embedding],
            n_results=top_k + 1,  # Get one extra to account for self-match
            include=["metadatas", "documents", "distances"]
        )
        
        # Filter out the original dataset and format results
        hits = []
        for i in range(len(results["ids"][0])):
            result_id = results["ids"][0][i]
            
            # Skip if this is the original dataset
            if result_id == dataset_id:
                continue
                
            # Stop if we have enough results
            if len(hits) >= top_k:
                break
            
            # Convert ChromaDB distance to similarity score
            distance = max(0.0, float(results["distances"][0][i]))
            # For normalized embeddings: cosine_similarity ≈ 1 - (L2_distance² / 2)
            similarity_score = max(0.0, min(1.0, 1.0 - (distance * distance / 2.0)))
                
            hit = {
                "id": result_id,
                "title": results["metadatas"][0][i]["title"],
                "description": results["metadatas"][0][i]["description"],
                "agency": results["metadatas"][0][i]["agency"],
                "api_url": results["metadatas"][0][i]["api_url"],
                "url": results["metadatas"][0][i].get("url", ""),
                "similarity_score": similarity_score
            }
            hits.append(hit)
        
        return hits
        
    except Exception as e:
        raise Exception(f"Similar search failed: {e}")
