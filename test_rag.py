import pickle
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="research_papers")

def retrieve(query, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode(query)
    results = collection.query(query_embeddings=emb.tolist(), n_results=top_k)

    retrieved = []
    for i in range(len(results["ids"][0])):
        doc = {
            "paper_id": results["metadatas"][0][i]["paper_id"],
            "text": results["documents"][0][i]
        }
        retrieved.append(doc)
    return retrieved
def ask(query, model="llama3:8b"):
    docs = retrieve(query, top_k=5)
    context = "\n\n".join([d["text"] for d in docs])
    prompt = f"""You are an expert AI research assistant. Use ONLY the provided context to answer the question.
- If the context contains relevant information, summarize key points clearly.
- If the context does NOT contain enough detail, say: "The retrieved papers mention this topic but do not provide specific details."
- Do NOT invent facts.

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.generate(model=model, prompt=prompt)
    return response['response'], docs
if __name__ == "__main__":
    question = "What are efficient methods in NLP mentioned in recent papers?"
    answer, sources = ask(question, model="phi3")
    print("?", question)
    print("||", answer)
    print("\n | Sources:")
    for s in sources:
        print("-", s["paper_id"])