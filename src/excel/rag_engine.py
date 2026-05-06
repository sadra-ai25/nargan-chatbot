# src/excel/rag_engine.py - FIXED VERSION
"""
Excel RAG Engine - Moved from src/rag_engine.py
Core retrieval and generation logic for Excel data
"""
import re
import faiss
import pickle
import requests
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from src.config import (
    EXCEL_API_KEY,
    EXCEL_BASE_URL,
    EXCEL_LLM_MODEL,
    E5_MODEL_PATH,
    EXCEL_INDEX_PATH,
    EXCEL_METADATA_PATH,
    DEFAULT_K,
)


class ExcelRAGEngine:
    """Main Excel RAG engine class"""

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.metadata_db: list = []
        self.conversations: dict = {}
        self._loaded = False

    def load(self):
        """Load all required models and data"""
        if self._loaded:
            return

        print("⏳ Loading Excel embedding model...")
        self.embedding_model = SentenceTransformer(E5_MODEL_PATH)

        print("⏳ Loading Excel FAISS index...")
        self.faiss_index = faiss.read_index(str(EXCEL_INDEX_PATH))

        print("⏳ Loading Excel metadata...")
        with open(EXCEL_METADATA_PATH, "rb") as f:
            self.metadata_db = pickle.load(f)

        self._loaded = True
        print("✅ Excel RAG Engine loaded successfully!")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def document_count(self) -> int:
        return len(self.metadata_db)

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for query"""
        text = "query: " + text
        emb = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return emb.astype("float32")

    @staticmethod
    def normalize(x: str) -> str:
        """Normalize text for comparison"""
        return x.replace(" ", "").replace("_", "").replace("-", "").lower()

    @staticmethod
    def extract_entities(text: str) -> list:
        """Extract PT-XXXX patterns"""
        return re.findall(r'\bPT-\d+\b', text)

    def query(
        self,
        user_question: str,
        chat_id: str = "default",
        k: int = DEFAULT_K
    ) -> dict:
        """
        Main Excel RAG query method
        
        Returns:
            dict with keys: answer, sources, remembered_entities, results_count
        """
        # Get conversation state
        conv = self.conversations.get(chat_id, {
            "messages": [],
            "last_sources": [],
            "last_entities": []
        })
        history_messages = conv["messages"]
        last_sources = conv["last_sources"]
        last_entities = conv["last_entities"]

        # Query rewriting based on previous entities
        if last_entities:
            has_entity = bool(re.findall(r'\bPT-\d+\b', user_question))
            if not has_entity:
                user_question = f"{user_question} برای {last_entities[0]}"
                print(f"🔄 Query Rewritten: {user_question}")

        # Extract search tags
        words = user_question.split()
        tags_to_search = []
        for w in words:
            clean_w = w.strip(".,?!;()[]{}:'\"/\\")
            if re.search(r'[A-Za-z]', clean_w) and re.search(r'\d', clean_w):
                tags_to_search.append(clean_w.lower())

        # Find number matches
        number_matches = re.findall(r'\d+\.?\d*', user_question)

        # Reverse hits (text contains number)
        reverse_hits = []
        if number_matches:
            for num in number_matches:
                for i, meta in enumerate(self.metadata_db):
                    if num in meta['text']:
                        reverse_hits.append({
                            "index": i,
                            "score": 5.0,
                            "text": meta['text'],
                            "source_file": meta['source_file']
                        })

        # Exact matches (tag search)
        exact_matches = []
        if tags_to_search:
            for i, meta in enumerate(self.metadata_db):
                text_lower = meta['text'].lower()
                match_count = sum(1 for tag in tags_to_search if tag in text_lower)
                if match_count > 0:
                    exact_matches.append({
                        "index": i,
                        "score": match_count * 2.0,
                        "text": meta['text'],
                        "source_file": meta['source_file']
                    })

        # Vector search
        query_vector = self.get_query_embedding(user_question).reshape(1, -1)

        if not tags_to_search and last_sources:
            # Search only in previous sources
            normalized_last = [self.normalize(s) for s in last_sources]
            candidate_indices = [
                i for i, m in enumerate(self.metadata_db)
                if self.normalize(m['source_file']) in normalized_last
            ]

            if candidate_indices:
                all_vecs = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
                all_vecs = np.array(all_vecs, dtype="float32")
                sub_vecs = np.array([all_vecs[i] for i in candidate_indices], dtype="float32")
                sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
                sub_index.add(sub_vecs)
                D_sub, I_sub = sub_index.search(query_vector, k * 2)

                vector_results = []
                for dist, local_idx in zip(D_sub[0], I_sub[0]):
                    if local_idx == -1:
                        continue
                    real_idx = candidate_indices[local_idx]
                    vector_results.append({
                        "index": int(real_idx),
                        "score": float(dist),
                        "text": self.metadata_db[real_idx]['text'],
                        "source_file": self.metadata_db[real_idx]['source_file']
                    })
            else:
                vector_results = []
        else:
            # Full search
            D, I = self.faiss_index.search(query_vector, k * 2)
            vector_results = []
            for dist, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                vector_results.append({
                    "index": int(idx),
                    "score": float(dist),
                    "text": self.metadata_db[idx]['text'],
                    "source_file": self.metadata_db[idx]['source_file']
                })

        # Combine results
        combined_results = {}

        for res in reverse_hits:
            combined_results[res["index"]] = res

        for res in exact_matches:
            idx = res["index"]
            combined_results[idx] = combined_results.get(idx, {"score": 0}) | res
            combined_results[idx]["score"] += res["score"]

        for res in vector_results:
            idx = res["index"]
            combined_results[idx] = combined_results.get(idx, {"score": 0}) | res
            combined_results[idx]["score"] += res["score"]

        # Sort and get top k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        top_k_results = sorted_results[:k]

        # Build context
        context = "\n\n".join(
            f"[منبع: {res['source_file']}]\n{res['text']}"
            for res in top_k_results
        )

        # Build history text
        history_text = ""
        if history_messages:
            last_history = history_messages[-4:]
            history_text = "\n\nPrevious dialog:\n"
            for msg in last_history:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # Build prompt
        prompt = f"""You are an engineering assistant answering ONLY from Context.
Rules:
- Answer concisely (max 3 sentences)
- In Persian
- No intro, no summary
- If not found say: در مستندات پیدا نشد

Context:
{context}
{history_text}

Question:
{user_question}

Answer:"""

        # Call LLM
                # Call LLM
        headers = {
            "Authorization": f"Bearer {EXCEL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": EXCEL_LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }

        base_url = EXCEL_BASE_URL.rstrip("/")
        chat_endpoint = f"{base_url}/chat/completions"


        try:
            print(f"🔗 Calling LLM at: {chat_endpoint}")   # برای دیباگ مفید
            response = requests.post(chat_endpoint, headers=headers, json=payload, timeout=60)
            
            print(f"🔍 LLM Status: {response.status_code}")
            print(f"🔍 LLM Response preview: {response.text[:800]}...")  
            
            response_data = response.json()
            
            # استخراج جواب (چند ساختار رایج رو هندل می‌کنه)
            if "choices" in response_data and response_data["choices"]:
                answer_text = response_data["choices"][0]["message"]["content"]
            elif "response" in response_data:
                answer_text = response_data["response"]
            elif "output" in response_data:
                answer_text = response_data["output"]
            elif "text" in response_data:
                answer_text = response_data["text"]
            else:
                answer_text = f"Unexpected response format from LLM: {str(response_data)[:300]}"
                
        except requests.exceptions.RequestException as e:
            answer_text = f"LLM Connection Error: {str(e)}"
            print(f"❌ LLM Request Error: {e}")
        except Exception as e:
            answer_text = f"LLM Error: {str(e)}"
            print(f"❌ LLM Parsing Error: {e}")

        # Update entities and sources
        current_entities = self.extract_entities(user_question)
        if not current_entities:
            current_entities = last_entities

        used_sources = [res['source_file'] for res in top_k_results]

        # Update conversation history
        history_messages.append({"role": "user", "content": user_question})
        history_messages.append({"role": "assistant", "content": answer_text})

        self.conversations[chat_id] = {
            "messages": history_messages,
            "last_sources": used_sources,
            "last_entities": current_entities
        }

        return {
            "answer": answer_text,
            "sources": used_sources,
            "remembered_entities": current_entities,
            "results_count": len(top_k_results)
        }


# Global instance
excel_rag_engine = ExcelRAGEngine()
