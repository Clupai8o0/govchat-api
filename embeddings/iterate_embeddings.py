import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingsIterator:
    """A class to iterate over and work with embeddings from embeddings.pkl"""
    
    def __init__(self, embeddings_path: str = "embeddings.pkl", csv_path: str = "datasets.csv"):
        self.embeddings_path = embeddings_path
        self.csv_path = csv_path
        self.embeddings = None
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load embeddings and original CSV data"""
        # Load embeddings
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded {len(self.embeddings)} embeddings from {self.embeddings_path}")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        
        # Load original CSV data for metadata
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} rows from {self.csv_path}")
        else:
            print(f"Warning: CSV file not found: {self.csv_path}")
    
    def get_embedding_info(self) -> Dict:
        """Get basic information about the embeddings"""
        if not self.embeddings:
            return {}
        
        first_embedding = next(iter(self.embeddings.values()))
        return {
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": len(first_embedding),
            "sample_ids": list(self.embeddings.keys())[:5],
            "data_type": type(first_embedding[0]).__name__
        }
    
    def iterate_embeddings(self, batch_size: int = None):
        """Generator to iterate through embeddings in batches or individually"""
        items = list(self.embeddings.items())
        
        if batch_size is None:
            # Iterate individually
            for doc_id, embedding in items:
                yield doc_id, embedding
        else:
            # Iterate in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                yield batch
    
    def get_embedding_by_id(self, doc_id: str):
        """Get a specific embedding by document ID"""
        return self.embeddings.get(doc_id)
    
    def get_document_metadata(self, doc_id: str) -> Dict:
        """Get metadata for a document from the original CSV"""
        if self.df is None:
            return {}
        
        row = self.df[self.df['id'] == doc_id]
        if row.empty:
            return {}
        
        return row.iloc[0].to_dict()
    
    def find_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar documents to a given document ID"""
        if doc_id not in self.embeddings:
            return []
        
        query_embedding = np.array(self.embeddings[doc_id]).reshape(1, -1)
        similarities = []
        
        for other_id, other_embedding in self.embeddings.items():
            if other_id != doc_id:
                other_emb = np.array(other_embedding).reshape(1, -1)
                sim = cosine_similarity(query_embedding, other_emb)[0][0]
                similarities.append((other_id, sim))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_embeddings_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Convert embeddings to a matrix format for batch operations"""
        ids = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[doc_id] for doc_id in ids])
        return embeddings_matrix, ids
    
    def search_by_text_embedding(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents using a query embedding"""
        query_emb = np.array(query_embedding).reshape(1, -1)
        similarities = []
        
        for doc_id, doc_embedding in self.embeddings.items():
            doc_emb = np.array(doc_embedding).reshape(1, -1)
            sim = cosine_similarity(query_emb, doc_emb)[0][0]
            similarities.append((doc_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def export_embeddings_subset(self, doc_ids: List[str], output_path: str):
        """Export a subset of embeddings to a new pickle file"""
        subset = {doc_id: self.embeddings[doc_id] for doc_id in doc_ids if doc_id in self.embeddings}
        
        with open(output_path, "wb") as f:
            pickle.dump(subset, f)
        
        print(f"Exported {len(subset)} embeddings to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistical information about the embeddings"""
        if not self.embeddings:
            return {}
        
        embeddings_array = np.array(list(self.embeddings.values()))
        
        return {
            "mean_values": embeddings_array.mean(axis=0)[:5].tolist(),  # First 5 dimensions
            "std_values": embeddings_array.std(axis=0)[:5].tolist(),    # First 5 dimensions
            "min_value": embeddings_array.min(),
            "max_value": embeddings_array.max(),
            "shape": embeddings_array.shape
        }


def main():
    """Example usage of the EmbeddingsIterator"""
    
    # Initialize the iterator
    iterator = EmbeddingsIterator()
    
    # Get basic info
    info = iterator.get_embedding_info()
    print("\n=== Embedding Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Get statistics
    stats = iterator.get_statistics()
    print("\n=== Embedding Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example: Iterate through first 5 embeddings
    print("\n=== Sample Iteration ===")
    count = 0
    for doc_id, embedding in iterator.iterate_embeddings():
        if count >= 5:
            break
        print(f"Document ID: {doc_id}")
        print(f"Embedding shape: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Get metadata if available
        metadata = iterator.get_document_metadata(doc_id)
        if metadata:
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Agency: {metadata.get('agency', 'N/A')}")
        print("-" * 50)
        count += 1
    
    # Example: Find similar documents
    if info.get("sample_ids"):
        sample_id = info["sample_ids"][0]
        print(f"\n=== Similar Documents to {sample_id} ===")
        similar_docs = iterator.find_similar_documents(sample_id, top_k=3)
        
        for sim_id, similarity in similar_docs:
            print(f"ID: {sim_id}, Similarity: {similarity:.4f}")
            metadata = iterator.get_document_metadata(sim_id)
            if metadata:
                print(f"  Title: {metadata.get('title', 'N/A')}")
    
    # Example: Batch iteration
    print(f"\n=== Batch Iteration (batch size 10) ===")
    batch_count = 0
    for batch in iterator.iterate_embeddings(batch_size=10):
        if batch_count >= 2:  # Only show first 2 batches
            break
        print(f"Batch {batch_count + 1}: {len(batch)} items")
        print(f"  First item ID: {batch[0][0]}")
        print(f"  Last item ID: {batch[-1][0]}")
        batch_count += 1


if __name__ == "__main__":
    main()
