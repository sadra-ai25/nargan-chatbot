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


# -------------------------
# 🔵 normalize (برای حافظه فایل‌ها)
# -------------------------
def normalize(x):
    return x.replace(" ", "").replace("_", "").replace("-", "").lower()


# -------------------------
# 🟢 Entity Extractor (حافظه تجهیز)
# -------------------------
def extract_entities(text):
    # در صورت نیاز regex را ارتقا می‌دهیم
    return re.findall(r'\bPT-\d+\b', text)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>سیستم هوشمند پرسش و پاسخ نارگان</title>

<style>
body{
margin:0;
background:#f1f4f9;
font-family:tahoma, sans-serif;
display:flex;
justify-content:center;
height:100vh;
}

.container{
width:900px;
max-width:95%;
display:flex;
flex-direction:column;
background:white;
border-radius:14px;
box-shadow:0 10px 35px rgba(0,0,0,0.12);
overflow:hidden;
}

.header{
background:#0d6efd;
color:white;
padding:18px;
font-size:20px;
text-align:center;
}

.chat-area{
flex:1;
overflow-y:auto;
padding:20px;
background:#f7f9fc;
}

.message{
margin-bottom:18px;
max-width:80%;
padding:12px 16px;
border-radius:10px;
line-height:1.7;
font-size:14px;
}

.user{
background:#dbe9ff;
margin-right:auto;
}

.bot{
background:#ffffff;
border:1px solid #e3e6ee;
margin-left:auto;
}

.sources{
margin-top:10px;
font-size:12px;
color:#666;
}

.sources ul{
padding-right:18px;
margin:5px 0;
}

.input-area{
display:flex;
padding:15px;
gap:10px;
border-top:1px solid #eee;
background:white;
}

input{
flex:1;
padding:12px;
border-radius:8px;
border:1px solid #ccc;
font-size:14px;
}

button{
background:#0d6efd;
border:none;
color:white;
padding:12px 20px;
border-radius:8px;
cursor:pointer;
font-size:14px;
}

button:hover{
background:#0b5ed7;
}

.loader{
text-align:center;
font-size:14px;
color:#0d6efd;
margin-bottom:15px;
display:none;
}
</style>
</head>

<body>

<div class="container">
<div class="header">سیستم پرسش و پاسخ هوشمند مستندات نارگان</div>
<div class="chat-area" id="chatArea"></div>
<div class="loader" id="loader">در حال تحلیل مستندات ...</div>
<div class="input-area">
<input
id="questionInput"
placeholder="سوال خود را درباره تجهیزات، تگ‌ها یا مستندات بپرسید..."
onkeypress="handleKeyPress(event)"
>
<button onclick="askQuestion()">ارسال</button>
</div>
</div>

<script>
// ⭐ تولید شناسه یکتا برای تب فعلی مرورگر
const SESSION_ID = "chat_" + Math.random().toString(36).substr(2, 9);

function handleKeyPress(e){
    if(e.key === "Enter"){
        askQuestion()
    }
}

function addUserMessage(text){
    const chat=document.getElementById("chatArea")
    const div=document.createElement("div")
    div.className="message user"
    div.innerText=text
    chat.appendChild(div)
    chat.scrollTop=chat.scrollHeight
}

function addBotMessage(answer,sources){
    const chat=document.getElementById("chatArea")
    const div=document.createElement("div")
    div.className="message bot"
    let html="<div>"+answer+"</div>"
    
    if(sources && sources.length>0){
        html+=`<div class="sources">
        <b>منابع:</b>
        <ul>
        `
        const unique=[...new Set(sources)]
        unique.forEach(s=>{
            html+="<li>"+s+"</li>"
        })
        html+="</ul></div>"
    }
    
    div.innerHTML=html
    chat.appendChild(div)
    chat.scrollTop=chat.scrollHeight
}

async function askQuestion(){
    const input=document.getElementById("questionInput")
    const question=input.value.trim()
    
    if(!question)return
    
    addUserMessage(question)
    input.value=""
    document.getElementById("loader").style.display="block"

    try{
        const response=await fetch("/ask",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                question: question,
                chat_id: SESSION_ID // ⭐ ارسال شناسه نشست به سرور
            })
        })

        const data=await response.json()
        addBotMessage(data.answer,data.sources)

    }catch(e){
        addBotMessage("خطا در ارتباط با سرور",[])
    }

    document.getElementById("loader").style.display="none"
}
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_TEMPLATE


# -------------------------------------------------
# 🔥 نسخه جدید و کامل /ask با حافظه تجهیز اصلاح شده
# -------------------------------------------------
@app.post("/ask")
async def ask_rag(request: QueryRequest):

    user_question = request.question
    chat_id = request.chat_id
    k = 10

    # -------------------------
    # 🧠 Load conversation memory
    # -------------------------
    conv = CONVERSATIONS.get(chat_id, {
        "messages": [],
        "last_sources": [],
        "last_entities": []
    })

    history_messages = conv["messages"]
    last_sources = conv["last_sources"]
    last_entities = conv["last_entities"]

    # -------------------------
    # 🟣 اگر سؤال موجودیت (equipment) ندارد → از حافظه استفاده کن
    # -------------------------
    if last_entities:
        has_entity = bool(re.findall(r'\bPT-\d+\b', user_question))
        if not has_entity:
            user_question = f"{user_question} برای {last_entities[0]}"
            print("🔄 Query Rewritten as:", user_question)

    # -------------------------
    # تشخیص تگ‌ها
    # -------------------------
    words = user_question.split()
    tags_to_search = []

    for w in words:
        clean_w = w.strip(".,?!;()[]{}:'\"/\\")
        if re.search(r'[A-Za-z]', clean_w) and re.search(r'\d', clean_w):
            tags_to_search.append(clean_w.lower())

    number_matches = re.findall(r'\d+\.?\d*', user_question)

    # -------------------------
    # Reverse lookup
    # -------------------------
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

    # -------------------------
    # Exact tag matching
    # -------------------------
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

    # -------------------------
    # Embedding search
    # -------------------------
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

    # -------------------------
    # Combine Results
    # -------------------------
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

    # -------------------------
    # History (full convo)
    # -------------------------
    history_text = ""
    if history_messages:
        last_history = history_messages[-4:]
        history_text = "\n\nPrevious dialog:\n"
        for msg in last_history:
            history_text += f"{msg['role']}: {msg['content']}\n"

    # -------------------------
    # LLM prompt
    # -------------------------
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

    # -------------------------
    # ⭐ استخراج موجودیت‌ها فقط از سوال فعلی
    # -------------------------
    current_entities = extract_entities(user_question)
    
    # اگر کاربر در این سوال تگی نگفته بود، همان تگ‌های قبلی را در حافظه نگه دار
    if not current_entities:
        current_entities = last_entities

    # -------------------------
    # Save memory
    # -------------------------
    used_sources = [res['source_file'] for res in top_k_results]

    history_messages.append({"role": "user", "content": user_question})
    history_messages.append({"role": "assistant", "content": answer_text})

    CONVERSATIONS[chat_id] = {
        "messages": history_messages,
        "last_sources": used_sources,
        "last_entities": current_entities   # ⭐ لیست مرتب و دقیق
    }

    return {
        "answer": answer_text,
        "sources": used_sources,
        "remembered_entities": current_entities   
    }
