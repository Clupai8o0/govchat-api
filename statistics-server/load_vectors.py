"""
Load vectors from embeddings.pkl and datasets.csv into ChromaDB.
Run this once to set up the vector store.
"""

import os
import pickle
import pandas as pd
import chromadb
from chromadb.config import Settings


def load_data():
    """Load datasets and embeddings from files."""
    # Use combined dataset if it exists, otherwise fall back to original
    if os.path.exists("combined_datasets.csv"):
        df = pd.read_csv("combined_datasets.csv")
        print(f"Loaded {len(df)} rows from combined_datasets.csv")
    else:
        df = pd.read_csv("datasets.csv")
        print(f"Loaded {len(df)} rows from datasets.csv")
    
    # Load embeddings
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    print(f"Loaded {len(embeddings)} embeddings from embeddings.pkl")
    
    return df, embeddings


def setup_chromadb():
    """Initialize ChromaDB client and collection."""
    # Create persistent client
    client = chromadb.PersistentClient(path="./vector_store")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("datasets")
        print("Deleted existing collection")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="datasets",
        metadata={"description": "Government datasets with embeddings"}
    )
    print("Created new ChromaDB collection")
    
    return collection


def populate_collection(collection, df, embeddings):
    """Add all datasets and embeddings to ChromaDB collection."""
    # Prepare data for ChromaDB
    ids = []
    vectors = []
    metadatas = []
    documents = []
    
    for _, row in df.iterrows():
        dataset_id = str(row["id"])
        
        # Skip if no embedding available
        if dataset_id not in embeddings:
            continue
            
        # Build document text (same as used for embedding)
        doc_text = " | ".join([
            str(row.get("title", "") or ""),
            str(row.get("description", "") or ""),
            f"Agency: {row.get('agency','') or ''}",
            f"Api Url: {row.get('api_url','') or ''}",
            f"Collected: {row.get('collected','') or ''}",
            f"Tags: {row.get('tags','') or ''}"
        ])
        
        ids.append(dataset_id)
        vectors.append(embeddings[dataset_id])
        documents.append(doc_text)
        # Handle NaN values properly
        def safe_str(value):
            if pd.isna(value):
                return ""
            return str(value)
        
        metadatas.append({
            "title": safe_str(row.get("title", "")),
            "description": safe_str(row.get("description", "")),
            "agency": safe_str(row.get("agency", "")),
            "api_url": safe_str(row.get("api_url", "")),
            "collected": safe_str(row.get("collected", "")),
            "tags": safe_str(row.get("tags", ""))
        })
    
    # Add to collection in batch
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Added {len(ids)} datasets to ChromaDB collection")


def main():
    """Main function to load data and populate ChromaDB."""
    print("Loading datasets and embeddings...")
    df, embeddings = load_data()
    
    print("Setting up ChromaDB...")
    collection = setup_chromadb()
    
    print("Populating collection...")
    populate_collection(collection, df, embeddings)
    
    print("✅ Vector store setup complete!")


if __name__ == "__main__":
    main()
