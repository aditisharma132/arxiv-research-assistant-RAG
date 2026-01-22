# embed_and_store.py
import pickle
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

def create_vector_db(chunks, persist_dir="chroma_db", batch_size=5000):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate all embeddings (still in memory)
    print("Generating embeddings...")
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    
    print(f"Saving {len(chunks)} chunks to ChromaDB in batches of {batch_size}...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "research_papers"
    
    # Delete if exists (for clean re-run)
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add in batches
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        print(f"Adding batch {i//batch_size + 1}: chunks {i} to {end_idx}")
        
        batch_embeddings = embeddings[i:end_idx].tolist()
        batch_docs = [c["text"] for c in chunks[i:end_idx]]
        batch_metadatas = [{"paper_id": c["paper_id"], "chunk_id": c["chunk_id"]} for c in chunks[i:end_idx]]
        batch_ids = [f"{c['paper_id']}_{c['chunk_id']}" for c in chunks[i:end_idx]]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB.")

if __name__ == "__main__":
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    create_vector_db(chunks)