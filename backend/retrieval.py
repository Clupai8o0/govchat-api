"""
Vector search functions for dataset retrieval.
"""

import os
from typing import List, Dict, Any
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
                "similarity_score": similarity_score
            }
            hits.append(hit)
        
        return hits
        
    except Exception as e:
        raise Exception(f"Search failed: {e}")


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
                "similarity_score": similarity_score
            }
            hits.append(hit)
        
        return hits
        
    except Exception as e:
        raise Exception(f"Similar search failed: {e}")
