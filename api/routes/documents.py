# api/routes/documents.py - FIXED IMPORTS
"""
Document RAG Routes - Compatible with original IP-based token_db
"""
from fastapi import APIRouter, HTTPException, Request

from src.documents.rag_engine import doc_rag_engine  # FIXED: src not data
from src.token_db import (
    get_or_create_user, 
    use_tokens, 
    get_user_tokens,
    calculate_tokens
)
from src.models.schemas import DocumentQueryRequest, DocumentQueryResponse

router = APIRouter()


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    if hasattr(request, 'client') and request.client:
        return request.client.host
    
    return "unknown"


@router.post("/ask", response_model=DocumentQueryResponse)
async def ask_documents(request: Request, query: DocumentQueryRequest):
    """Ask question to document RAG"""
    if not doc_rag_engine.is_loaded:
        raise HTTPException(503, "Document engine not loaded")
    
    # Get client IP and user
    client_ip = get_client_ip(request)
    user_data = get_or_create_user(client_ip)
    user_id = user_data['user']['id']
    
    # Check tokens
    token_info = get_user_tokens(user_id)
    remaining = token_info['total_tokens'] - token_info['used_tokens']
    if remaining < 100:
        raise HTTPException(403, "Insufficient tokens")
    
    # Process query
    try:
        result = doc_rag_engine.query(query.question, top_k=query.top_k)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")
    
    # Use tokens
    token_result = use_tokens(user_id, query.question, result["answer"])
    if 'error' in token_result:
        raise HTTPException(403, token_result['error'])
    
    return DocumentQueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        tokens_used=token_result['tokens_used'],
        remaining_tokens=token_result['remaining']
    )


@router.get("/token/status")
async def doc_token_status(request: Request, chat_id: str):
    """Get token status for documents module (IP-based)"""
    client_ip = get_client_ip(request)
    user_data = get_or_create_user(client_ip)
    user_id = user_data['user']['id']
    token_info = get_user_tokens(user_id)
    
    remaining = token_info['total_tokens'] - token_info['used_tokens']
    
    return {
        "remaining_tokens": remaining,
        "max_tokens": token_info['total_tokens'],
        "used_tokens": token_info['used_tokens'],
        "module": "documents"
    }