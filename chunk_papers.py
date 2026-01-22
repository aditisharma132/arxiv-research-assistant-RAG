# chunk_from_metadata.py
import os
import json
import pickle

def create_chunks_from_metadata(metadata_dir="papers_metadata"):
    chunks = []
    for filename in os.listdir(metadata_dir):
        if filename.endswith(".json"):
            with open(os.path.join(metadata_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Create a rich text chunk
            text = f"Title: {data['title']}\n\nAbstract: {data['abstract']}"
            chunks.append({
                "text": text,
                "paper_id": data["paper_id"],
                "chunk_id": 0
            })
    return chunks

if __name__ == "__main__":
    chunks = create_chunks_from_metadata()
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Created {len(chunks)} chunks from metadata.")