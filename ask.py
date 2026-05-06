import faiss
import pickle
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

API_KEY = "sk-DAThtIkyTqQE3paeoAigBagEMcp9tDe3vviB76xkeScGmLeg"
LLM_URL = "https://api.gapgpt.app/v1/chat/completions"


MODEL_PATH = "/mnt/storage-1/home/sadra/AISadra/user-4/e5-embedding/e5-model"

# load embedding model once
model = SentenceTransformer(MODEL_PATH)


def get_query_embedding(text):
    text = "query: " + text
    emb = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb.astype("float32")


def ask(user_question, k=10):
    index = faiss.read_index("vector.index")

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query_vector = get_query_embedding(user_question).reshape(1, -1)

    D, I = index.search(query_vector, k)

    context = "\n\n".join(
        f"[منبع: {metadata[i]['source_file']}]\n{metadata[i]['text']}"
        for i in I[0]
    )

    prompt = f"""
Context:
{context}

Question:
{user_question}

Answer:
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gapgpt-qwen-3.5",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(LLM_URL, headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    q = input("سوال خود را بپرسید: ")
    print("⏳ در حال جستجو...")
    print(ask(q))
