import faiss
import pickle
import requests
import numpy as np
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

API_KEY = "sk-kFIVdGGKBeqL3Eb7nZ67CVBcNiuo5KEid6S8meSfNzcQPrsZ"
LLM_URL = "https://api.gapgpt.app/v1/chat/completions"
MODEL_PATH = "/mnt/storage-1/home/sadra/AISadra/user-4/e5-embedding/e5-model"

app = FastAPI(title="Engineering RAG System")

print("⏳ Loading embedding model and database...")
embedding_model = SentenceTransformer(MODEL_PATH)
faiss_index = faiss.read_index("vector.index")
with open("metadata.pkl", "rb") as f:
    metadata_db = pickle.load(f)
print("✅ System loaded.")

CONVERSATIONS = {}

class QueryRequest(BaseModel):
    question: str
    chat_id: str = "default"

def get_query_embedding(text):
    text = "query: " + text
    emb = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb.astype("float32")

def normalize(x):
    return x.replace(" ", "").replace("_", "").replace("-", "").lower()

def extract_entities(text):
    return re.findall(r'\bPT-\d+\b', text)

# ============================================================
# 🎨 قالب HTML جدید - طراحی زیبا با فونت فارسی
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تحلیلگر مستندات نارگان</title>
    <style>
        @font-face {
            font-family: "BRoya";
            src: url("BRoya.ttf");
        }
        @font-face {
            font-family: "BNazanin";
            src: url("BNazanin.ttf");
        }
        * {
            box-sizing: border-box;
        }
        body {
            font-family: "BRoya", "BNazanin", Tahoma, sans-serif;
            direction: rtl;
            background: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        .container {
            width: 60%;
            max-width: 900px;
            min-width: 320px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px #ccc;
        }
        .header {
            font-size: 22px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
            color: #2e7d32;
            padding-bottom: 15px;
            border-bottom: 2px solid #4caf50;
        }
        #chat {
            height: 420px;
            overflow-y: auto;
            background: #fafafa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .msg {
            padding: 12px 16px;
            margin-bottom: 12px;
            border-radius: 10px;
            max-width: 80%;
            line-height: 1.7;
            word-wrap: break-word;
        }
        .user {
            background: #d1f0ff;
            margin-right: auto;
            border: none;
        }
        .bot {
            background: #e2ffe2;
            border: 1px solid #c9f4c9;
            margin-left: auto;
        }
        .sources {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed #aaa;
            font-size: 12px;
            color: #666;
        }
        .sources strong {
            color: #2e7d32;
        }
        .sources ul {
            margin: 5px 0;
            padding-right: 20px;
        }
        .sources li {
            margin: 2px 0;
        }
        .box {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-family: inherit;
            font-size: 14px;
        }
        input:focus {
            outline: none;
            border-color: #4caf50;
        }
        button {
            padding: 12px 25px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 14px;
        }
        button:hover {
            background: #3d8b40;
        }
        button:disabled {
            background: #aaa;
            cursor: not-allowed;
        }
        .loader {
            text-align: center;
            color: #4caf50;
            font-size: 13px;
            display: none;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">تحلیلگر مستندات نارگان</div>
        <div id="chat"></div>
        <div class="loader" id="loader">در حال پردازش...</div>
        <div class="box">
            <input id="question" placeholder="سؤال خود را بنویسید..." 
                   onkeydown="if(event.key === 'Enter') sendMsg()">
            <button onclick="sendMsg()" id="sendBtn">ارسال</button>
        </div>
    </div>

    <script>
        // ⭐ تولید شناسه یکتا برای هر نشست مرورگر
        const SESSION_ID = "chat_" + Math.random().toString(36).substr(2, 9);
        
        async function sendMsg() {
            const input = document.getElementById("question");
            const btn = document.getElementById("sendBtn");
            const loader = document.getElementById("loader");
            
            let q = input.value.trim();
            if (!q) return;
            
            // نمایش پیام کاربر
            addMsg(q, "user");
            input.value = "";
            
            // فعال کردن حالت لودینگ
            btn.disabled = true;
            loader.style.display = "block";
            
            try {
                let res = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        question: q,
                        chat_id: SESSION_ID
                    })
                });
                
                let data = await res.json();
                
                // نمایش پاسخ به همراه منابع
                addMsg(data.answer || data.error || "خطایی رخ داد", "bot", data.sources);
                
            } catch (e) {
                addMsg("خطا در ارتباط با سرور", "bot", []);
            }
            
            // غیرفعال کردن حالت لودینگ
            btn.disabled = false;
            loader.style.display = "none";
        }
        
        function addMsg(text, type, sources = null) {
            let box = document.getElementById("chat");
            let div = document.createElement("div");
            div.className = "msg " + type;
            
            // اگر پاسخ ربات باشد و منابع وجود داشته باشد، منابع را نمایش بده
            let html = "";
            if (type === "bot") {
                html = `<div>${text}</div>`;
                if (sources && sources.length > 0) {
                    // حذف منابع تکراری
                    const uniqueSources = [...new Set(sources)];
                    html += `<div class="sources"><strong>منابع:</strong><ul>`;
                    uniqueSources.forEach(s => {
                        html += `<li>${s}</li>`;
                    });
                    html += `</ul></div>`;
                }
                div.innerHTML = html;
            } else {
                div.textContent = text;
            }
            
            box.appendChild(div);
            box.scrollTop = box.scrollHeight;
        }
    </script>
</body>
</html>"""

# ============================================================
# بقیه کدهای FastAPI بدون تغییر می‌مانند
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_TEMPLATE

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    user_question = request.question
    chat_id = request.chat_id
    k = 10
    
    conv = CONVERSATIONS.get(chat_id, {
        "messages": [],
        "last_sources": [],
        "last_entities": []
    })
    
    history_messages = conv["messages"]
    last_sources = conv["last_sources"]
    last_entities = conv["last_entities"]
    
    if last_entities:
        has_entity = bool(re.findall(r'\bPT-\d+\b', user_question))
        if not has_entity:
            user_question = f"{user_question} برای {last_entities[0]}"
            print("🔄 Query Rewritten as:", user_question)
    
    words = user_question.split()
    tags_to_search = []
    for w in words:
        clean_w = w.strip(".,?!;()[]{}:'\"/\\")
        if re.search(r'[A-Za-z]', clean_w) and re.search(r'\d', clean_w):
            tags_to_search.append(clean_w.lower())
    
    number_matches = re.findall(r'\d+\.?\d*', user_question)
    
    reverse_hits = []
    if number_matches:
        for num in number_matches:
            for i, meta in enumerate(metadata_db):
                if num in meta['text']:
                    reverse_hits.append({
                        "index": i,
                        "score": 5.0,
                        "text": meta['text'],
                        "source_file": meta['source_file']
                    })
    
    exact_matches = []
    if tags_to_search:
        for i, meta in enumerate(metadata_db):
            text_lower = meta['text'].lower()
            match_count = sum(1 for tag in tags_to_search if tag in text_lower)
            if match_count > 0:
                exact_matches.append({
                    "index": i,
                    "score": match_count * 2.0,
                    "text": meta['text'],
                    "source_file": meta['source_file']
                })
    
    query_vector = get_query_embedding(user_question).reshape(1, -1)
    
    if not tags_to_search and last_sources:
        normalized_last = [normalize(s) for s in last_sources]
        candidate_indices = [
            i for i, m in enumerate(metadata_db)
            if normalize(m['source_file']) in normalized_last
        ]
        if candidate_indices:
            all_vecs = faiss_index.reconstruct_n(0, faiss_index.ntotal)
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
                    "text": metadata_db[real_idx]['text'],
                    "source_file": metadata_db[real_idx]['source_file']
                })
        else:
            vector_results = []
    else:
        D, I = faiss_index.search(query_vector, k * 2)
        vector_results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            vector_results.append({
                "index": int(idx),
                "score": float(dist),
                "text": metadata_db[idx]['text'],
                "source_file": metadata_db[idx]['source_file']
            })
    
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
    
    sorted_results = sorted(
        combined_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    top_k_results = sorted_results[:k]
    
    context = "\n\n".join(
        f"[منبع: {res['source_file']}]\n{res['text']}"
        for res in top_k_results
    )
    
    history_text = ""
    if history_messages:
        last_history = history_messages[-4:]
        history_text = "\n\nPrevious dialog:\n"
        for msg in last_history:
            history_text += f"{msg['role']}: {msg['content']}\n"
    
    prompt = f"""
You are an engineering assistant answering ONLY from Context.
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
Answer:
"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gapgpt-qwen-3.5",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(LLM_URL, headers=headers, json=payload)
        response_data = response.json()
        answer_text = response_data["choices"][0]["message"]["content"]
    except Exception as e:
        answer_text = f"LLM Error: {str(e)}"
    
    current_entities = extract_entities(user_question)
    if not current_entities:
        current_entities = last_entities
    
    used_sources = [res['source_file'] for res in top_k_results]
    history_messages.append({"role": "user", "content": user_question})
    history_messages.append({"role": "assistant", "content": answer_text})
    
    CONVERSATIONS[chat_id] = {
        "messages": history_messages,
        "last_sources": used_sources,
        "last_entities": current_entities
    }
    
    return {
        "answer": answer_text,
        "sources": used_sources,
        "remembered_entities": current_entities
    }