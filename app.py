# app.py
import streamlit as st
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

# --- Load once at startup (use caching) ---
@st.cache_resource
def load_components():
    chunks = pickle.load(open("chunks.pkl", "rb"))
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("research_papers")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load LLM
    llm = Llama(
        model_path="Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=2048,
        n_threads=4,  # Adjust based on available CPU
        verbose=False
    )
    return chunks, collection, encoder, llm

chunks, collection, encoder, llm = load_components()

def retrieve(query, top_k=5):
    emb = encoder.encode(query)
    results = collection.query(query_embeddings=[emb.tolist()], n_results=top_k)
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "paper_id": results["metadatas"][0][i]["paper_id"]
        })
    return docs

def ask(question):
    docs = retrieve(question)
    context = "\n\n".join([d["text"] for d in docs])
    
    prompt = f"""You are an expert AI research assistant. Use ONLY the provided context to answer the question.
- If the context contains relevant information, summarize key points clearly.
- If the context does NOT contain enough detail, say: "The retrieved papers mention this topic but do not provide specific details."
- Do NOT invent facts.

Context:
{context}

Question: {question}

Answer:"""
    
    output = llm(prompt, max_tokens=512, stop=["\n\n"], echo=False)
    answer = output["choices"][0]["text"].strip()
    return answer, docs

# --- Streamlit UI ---
st.set_page_config(page_title="AI Research Assistant", page_icon="📚")
st.title("📚 AI Research Assistant")
st.caption("Ask questions about 1000+ recent AI/ML papers — powered by RAG + Phi-3")

question = st.text_input("Enter your question:", placeholder="What are efficient methods in NLP?")

if st.button("Ask") or question:
    if question.strip():
        with st.spinner("Searching papers and generating answer..."):
            answer, sources = ask(question)
        
        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Sources")
        for s in sources[:3]:
            st.caption(f"📄 {s['paper_id']}")
    else:
        st.warning("Please enter a question.")