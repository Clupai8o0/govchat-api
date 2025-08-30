#!/usr/bin/env python3
"""
Example script showing common operations with the embeddings
"""

from iterate_embeddings import EmbeddingsIterator
import numpy as np

def example_similarity_search():
    """Example: Find documents similar to a specific topic"""
    print("=== Example: Similarity Search ===")
    
    iterator = EmbeddingsIterator()
    
    # Find documents related to "ABORIGINAL" topics
    aboriginal_docs = [doc_id for doc_id in iterator.embeddings.keys() 
                      if "ABORIGINAL" in doc_id.upper()]
    
    print(f"Found {len(aboriginal_docs)} documents with 'ABORIGINAL' in ID")
    
    if aboriginal_docs:
        # Find similar documents to the first aboriginal document
        sample_doc = aboriginal_docs[0]
        similar_docs = iterator.find_similar_documents(sample_doc, top_k=5)
        
        print(f"\nDocuments similar to '{sample_doc}':")
        for doc_id, similarity in similar_docs:
            metadata = iterator.get_document_metadata(doc_id)
            print(f"  {similarity:.4f} - {doc_id}")
            print(f"    Title: {metadata.get('title', 'N/A')[:80]}...")

def example_batch_processing():
    """Example: Process embeddings in batches"""
    print("\n=== Example: Batch Processing ===")
    
    iterator = EmbeddingsIterator()
    
    # Process embeddings in batches to compute average similarity
    batch_similarities = []
    
    for batch in iterator.iterate_embeddings(batch_size=50):
        if len(batch_similarities) >= 3:  # Only process first 3 batches for demo
            break
            
        # Compute average pairwise similarity within batch
        embeddings = np.array([emb for _, emb in batch])
        
        # Compute cosine similarity matrix
        dot_products = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = dot_products / np.outer(norms, norms)
        
        # Get average similarity (excluding diagonal)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        batch_similarities.append(avg_similarity)
        
        print(f"Batch {len(batch_similarities)}: {len(batch)} docs, avg similarity: {avg_similarity:.4f}")

def example_filter_by_agency():
    """Example: Filter embeddings by agency"""
    print("\n=== Example: Filter by Agency ===")
    
    iterator = EmbeddingsIterator()
    
    # Get unique agencies
    agencies = iterator.df['agency'].value_counts()
    print(f"Found {len(agencies)} unique agencies:")
    for agency, count in agencies.head(5).items():
        print(f"  {agency}: {count} datasets")
    
    # Filter embeddings for a specific agency
    abs_docs = iterator.df[iterator.df['agency'] == 'ABS']['id'].tolist()
    print(f"\nABS has {len(abs_docs)} datasets")
    
    # Export subset for further analysis
    iterator.export_embeddings_subset(abs_docs[:10], "abs_embeddings_sample.pkl")

def example_embedding_analysis():
    """Example: Analyze embedding properties"""
    print("\n=== Example: Embedding Analysis ===")
    
    iterator = EmbeddingsIterator()
    
    # Convert to matrix for analysis
    embeddings_matrix, ids = iterator.get_embeddings_matrix()
    
    # Find most and least "central" documents
    centroid = embeddings_matrix.mean(axis=0)
    distances_to_centroid = np.linalg.norm(embeddings_matrix - centroid, axis=1)
    
    # Most central (closest to average)
    most_central_idx = np.argmin(distances_to_centroid)
    most_central_id = ids[most_central_idx]
    
    # Most outlier (farthest from average)
    most_outlier_idx = np.argmax(distances_to_centroid)
    most_outlier_id = ids[most_outlier_idx]
    
    print(f"Most central document: {most_central_id}")
    metadata = iterator.get_document_metadata(most_central_id)
    print(f"  Title: {metadata.get('title', 'N/A')[:80]}...")
    
    print(f"\nMost outlier document: {most_outlier_id}")
    metadata = iterator.get_document_metadata(most_outlier_id)
    print(f"  Title: {metadata.get('title', 'N/A')[:80]}...")

def example_search_by_keywords():
    """Example: Search documents by keywords in metadata"""
    print("\n=== Example: Keyword Search ===")
    
    iterator = EmbeddingsIterator()
    
    keywords = ["population", "health", "economic", "education"]
    
    for keyword in keywords:
        # Find documents with keyword in title or description
        matching_docs = []
        for _, row in iterator.df.iterrows():
            title = str(row.get('title', '')).lower()
            description = str(row.get('description', '')).lower()
            if keyword in title or keyword in description:
                matching_docs.append(row['id'])
        
        print(f"\nDocuments containing '{keyword}': {len(matching_docs)}")
        
        if matching_docs:
            # Show first few matches
            for doc_id in matching_docs[:3]:
                metadata = iterator.get_document_metadata(doc_id)
                print(f"  {doc_id}: {metadata.get('title', 'N/A')[:60]}...")

def main():
    """Run all examples"""
    print("Running embedding analysis examples...")
    print("=" * 60)
    
    try:
        example_similarity_search()
        example_batch_processing()
        example_filter_by_agency()
        example_embedding_analysis()
        example_search_by_keywords()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    main()
