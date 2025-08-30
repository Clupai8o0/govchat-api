#Not needed for running the API, gives a basic understanding of the embeddings.pkl file
"""
Simple utility script to quickly explore embeddings.pkl
"""

import pickle
import numpy as np
import pandas as pd

def load_embeddings(path="embeddings.pkl"):
    """Load embeddings from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)

def quick_stats(embeddings_dict):
    """Print quick statistics about embeddings"""
    print(f"Total embeddings: {len(embeddings_dict)}")
    
    if embeddings_dict:
        first_embedding = next(iter(embeddings_dict.values()))
        print(f"Embedding dimension: {len(first_embedding)}")
        print(f"Sample IDs: {list(embeddings_dict.keys())[:5]}")
        
        # Convert to numpy array for stats
        all_embeddings = np.array(list(embeddings_dict.values()))
        print(f"Shape: {all_embeddings.shape}")
        print(f"Mean magnitude: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")
        print(f"Min value: {all_embeddings.min():.4f}")
        print(f"Max value: {all_embeddings.max():.4f}")

def list_all_ids(embeddings_dict):
    """List all document IDs"""
    print("All document IDs:")
    for i, doc_id in enumerate(embeddings_dict.keys(), 1):
        print(f"{i:4d}: {doc_id}")

def show_embedding(embeddings_dict, doc_id):
    """Show details of a specific embedding"""
    if doc_id in embeddings_dict:
        embedding = embeddings_dict[doc_id]
        print(f"Embedding for document: {doc_id}")
        print(f"Dimension: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Last 10 values: {embedding[-10:]}")
        print(f"Magnitude: {np.linalg.norm(embedding):.4f}")
    else:
        print(f"Document ID '{doc_id}' not found")
        print("Available IDs:", list(embeddings_dict.keys())[:10])

def search_ids_by_pattern(embeddings_dict, pattern):
    """Search for document IDs containing a pattern"""
    matching_ids = [doc_id for doc_id in embeddings_dict.keys() if pattern.lower() in doc_id.lower()]
    print(f"IDs containing '{pattern}': {len(matching_ids)}")
    for doc_id in matching_ids[:20]:  # Show first 20 matches
        print(f"  {doc_id}")
    if len(matching_ids) > 20:
        print(f"  ... and {len(matching_ids) - 20} more")

def export_to_csv(embeddings_dict, output_path="embeddings_export.csv"):
    """Export embeddings to CSV format"""
    data = []
    for doc_id, embedding in embeddings_dict.items():
        row = {"id": doc_id}
        for i, val in enumerate(embedding):
            row[f"dim_{i:03d}"] = val
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(data)} embeddings to {output_path}")

def main():
    """Interactive CLI for exploring embeddings"""
    import sys
    
    # Load embeddings
    try:
        embeddings = load_embeddings()
        print("Successfully loaded embeddings.pkl")
        quick_stats(embeddings)
    except FileNotFoundError:
        print("Error: embeddings.pkl not found. Run make_embeddings_openai.py first.")
        return
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "stats":
            quick_stats(embeddings)
        
        elif command == "list":
            list_all_ids(embeddings)
        
        elif command == "show" and len(sys.argv) > 2:
            doc_id = sys.argv[2]
            show_embedding(embeddings, doc_id)
        
        elif command == "search" and len(sys.argv) > 2:
            pattern = sys.argv[2]
            search_ids_by_pattern(embeddings, pattern)
        
        elif command == "export":
            output_path = sys.argv[2] if len(sys.argv) > 2 else "embeddings_export.csv"
            export_to_csv(embeddings, output_path)
        
        else:
            print("Usage:")
            print("  python embedding_utils.py stats          - Show statistics")
            print("  python embedding_utils.py list           - List all document IDs")
            print("  python embedding_utils.py show <doc_id>  - Show specific embedding")
            print("  python embedding_utils.py search <pattern> - Search IDs by pattern")
            print("  python embedding_utils.py export [file]  - Export to CSV")
    
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("Interactive Embeddings Explorer")
        print("Commands: stats, list, show <id>, search <pattern>, export [file], quit")
        print("="*50)
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == "quit" or cmd[0] == "exit":
                    break
                elif cmd[0] == "stats":
                    quick_stats(embeddings)
                elif cmd[0] == "list":
                    list_all_ids(embeddings)
                elif cmd[0] == "show" and len(cmd) > 1:
                    show_embedding(embeddings, cmd[1])
                elif cmd[0] == "search" and len(cmd) > 1:
                    search_ids_by_pattern(embeddings, cmd[1])
                elif cmd[0] == "export":
                    output_path = cmd[1] if len(cmd) > 1 else "embeddings_export.csv"
                    export_to_csv(embeddings, output_path)
                else:
                    print("Unknown command. Available: stats, list, show <id>, search <pattern>, export [file], quit")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
