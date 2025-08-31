"""
Create embeddings for the combined dataset (original + education data).
This will add embeddings for the new education records to the existing embeddings.pkl
"""

import os
import pickle
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embedding(text: str) -> list:
    """Create embedding for a single text."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def create_combined_embeddings():
    """Create embeddings for combined dataset."""
    
    # Load combined dataset
    if not os.path.exists("combined_datasets.csv"):
        print("Error: combined_datasets.csv not found. Run prepare_combined_data.py first.")
        return False
    
    df = pd.read_csv("combined_datasets.csv")
    print(f"Loaded {len(df)} records from combined_datasets.csv")
    
    # Load existing embeddings
    existing_embeddings = {}
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            existing_embeddings = pickle.load(f)
        print(f"Loaded {len(existing_embeddings)} existing embeddings")
    
    # Create embeddings for missing records
    new_embeddings_count = 0
    all_embeddings = existing_embeddings.copy()
    
    for _, row in df.iterrows():
        dataset_id = str(row["id"])
        
        # Skip if embedding already exists
        if dataset_id in all_embeddings:
            continue
        
        # Build document text (same format as original)
        doc_text = " | ".join([
            str(row.get("title", "") or ""),
            str(row.get("description", "") or ""),
            f"Agency: {row.get('agency','') or ''}",
            f"Tags: {row.get('tags','') or ''}"
        ])
        
        print(f"Creating embedding for: {dataset_id}")
        embedding = create_embedding(doc_text)
        
        if embedding is not None:
            all_embeddings[dataset_id] = embedding
            new_embeddings_count += 1
            
            # Add small delay to respect rate limits
            time.sleep(0.1)
        else:
            print(f"Failed to create embedding for {dataset_id}")
    
    # Save updated embeddings
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(all_embeddings, f)
    
    print(f"✅ Successfully created {new_embeddings_count} new embeddings")
    print(f"Total embeddings: {len(all_embeddings)}")
    
    return True

if __name__ == "__main__":
    create_combined_embeddings()
