"""
FastAPI Main Application - Integrated Documents & Excel
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import traceback

from src.documents.rag_engine import doc_rag_engine
from src.excel.rag_engine import excel_rag_engine
from src.token_db import init_database
from api.routes import documents, excel

app = FastAPI(
    title="Nargan AI Platform",
    description="سیستم هوشمند تحلیل مستندات و اکسل نارگان",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents")
app.include_router(excel.router, prefix="/api/excel")

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ----------------------------------------------------
# STARTUP EVENT
# ----------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Load all engines and init token DB"""

    # Token DB Init
    try:
        init_database()
        print("✅ Token database initialized")
    except Exception as e:
        print(f"⚠️ Token DB init warning: {e}")

    # Load Document Engine (ASYNC)
    try:
        await doc_rag_engine.load()
        print("📘 Document engine loaded successfully")
    except Exception as e:
        print(f"⚠️ Document engine load warning: {e}")
        traceback.print_exc()

    # Load Excel Engine (SYNC)
    try:
        excel_rag_engine.load()
        print("📊 Excel engine loaded successfully")
    except Exception as e:
        print(f"⚠️ Excel engine load warning: {e}")
        traceback.print_exc()


# ----------------------------------------------------
# ROOT PAGE
# ----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))

    return HTMLResponse(
        """
        <html><body>
        <h1>Nargan AI Platform is Running!</h1>
        <p>UI file not found. Please ensure templates/index.html exists.</p>
        </body></html>
        """,
        status_code=200
    )


# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }


# ----------------------------------------------------
# LEGACY DOCUMENT API
# ----------------------------------------------------
@app.get("/api/token/status/documents")
async def legacy_doc_token_status(chat_id: str = "default"):
    from src.token_db import get_or_create_user, get_user_tokens
    user_data = get_or_create_user(chat_id)
    user_id = user_data['user']['id']
    token_info = get_user_tokens(user_id)

    remaining = token_info['total_tokens'] - token_info['used_tokens']

    return {
        "remaining_tokens": remaining,
        "max_tokens": token_info['total_tokens'],
        "used_tokens": token_info['used_tokens'],
        "module": "documents"
    }


@app.post("/api/ask/documents")
async def legacy_ask_documents(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "")
        chat_id = body.get("chat_id", "default")

        print(f"📥 Document query: {question[:50]}...")

        # Token system
        from src.token_db import get_or_create_user, use_tokens, get_user_tokens

        user_data = get_or_create_user(chat_id)
        user_id = user_data['user']['id']
        token_info = get_user_tokens(user_id)

        remaining = token_info['total_tokens'] - token_info['used_tokens']
        if remaining < 100:
            raise HTTPException(403, "Insufficient tokens")

        # Main RAG call
        result = await doc_rag_engine.query(question)

        token_result = use_tokens(user_id, question, result)

        if "error" in token_result:
            raise HTTPException(403, token_result["error"])

        return {
            "answer": result,
            "tokens_used": token_result["tokens_used"],
            "remaining_tokens": token_result["remaining"]
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error in legacy_ask_documents: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Internal error: {str(e)}")


# ----------------------------------------------------
# LEGACY EXCEL API
# ----------------------------------------------------
@app.get("/api/token/status/excel")
async def legacy_excel_token_status(chat_id: str = "default"):
    from src.token_db import get_or_create_user, get_user_tokens

    user_data = get_or_create_user(chat_id)
    user_id = user_data["user"]["id"]
    token_info = get_user_tokens(user_id)

    remaining = token_info["total_tokens"] - token_info["used_tokens"]

    return {
        "remaining_tokens": remaining,
        "max_tokens": token_info["total_tokens"],
        "used_tokens": token_info["used_tokens"],
        "module": "excel"
    }


@app.post("/api/ask/excel")
async def legacy_ask_excel(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "")
        chat_id = body.get("chat_id", "default")

        print(f"📥 Excel query: {question[:50]}...")

        # Token system
        from src.token_db import get_or_create_user, use_tokens, get_user_tokens

        user_data = get_or_create_user(chat_id)
        user_id = user_data["user"]["id"]
        token_info = get_user_tokens(user_id)

        remaining = token_info["total_tokens"] - token_info["used_tokens"]
        if remaining < 100:
            raise HTTPException(403, "Insufficient tokens")

        # Excel RAG call
        result = excel_rag_engine.query(
            user_question=question,
            chat_id=chat_id,
            k=10
        )

        token_result = use_tokens(user_id, question, result["answer"])

        if "error" in token_result:
            raise HTTPException(403, token_result["error"])

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "remembered_entities": result.get("remembered_entities", []),
            "results_count": result.get("results_count", 0),
            "tokens_used": token_result["tokens_used"],
            "remaining_tokens": token_result["remaining"]
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error in legacy_ask_excel: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Internal error: {str(e)}")


# ----------------------------------------------------
# STATIC FILES
# ----------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
