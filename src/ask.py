"""
Simple CLI RAG query script
Usage: python -m src.ask
"""
import json
import faiss
import pickle
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    API_KEY, LLM_URL, LLM_MODEL,
    MODEL_PATH, INDEX_PATH, METADATA_PATH
)


def get_query_embedding(text, model):
    """Generate embedding for query text"""
    text = "query: " + text
    emb = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb.astype("float32")


def ask(user_question: str, k: int = 10) -> str:
    """Query the RAG system"""
    # Load model and data
    print("⏳ Loading model and database...")
    model = SentenceTransformer(MODEL_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print("✅ System loaded.")

    # Search
    print("🔍 Searching...")
    query_vector = get_query_embedding(user_question, model).reshape(1, -1)
    D, I = index.search(query_vector, k)

    # Build context
    context = "\n\n".join(
        f"[منبع: {metadata[i]['source_file']}]\n{metadata[i]['text']}"
        for i in I[0]
    )

    # Build prompt
    prompt = f"""Context:
{context}

Question:
{user_question}

Answer:"""

    # Call LLM
    print("🤖 Calling LLM...")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(LLM_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("=" * 50)
    print("   RAG CLI - پرسش و پاسخ مستندات")
    print("=" * 50)

    while True:
        q = input("\n❓ سوال خود را بپرسید (یا 'exit' برای خروج): ").strip()
        if q.lower() in ['exit', 'quit', 'خروج']:
            break
        if not q:
            continue

        print("\n⏳ در حال جستجو...")
        try:
            answer = ask(q)
            print("\n" + "=" * 50)
            print("📝 پاسخ:")
            print(answer)
            print("=" * 50)
        except Exception as e:
            print(f"\n❌ خطا: {e}")