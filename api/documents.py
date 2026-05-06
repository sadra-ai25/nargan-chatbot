"""
Document RAG Routes
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict

from data.documents.rag_engine import doc_rag_engine
from src.token_db import token_manager
from src.models.schemas import QueryRequest, QueryResponse, TokenStatusResponse

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_documents(query: QueryRequest):
    """Ask question to document RAG"""
    if not doc_rag_engine.is_loaded:
        raise HTTPException(503, "Document engine not loaded")
    
    # Check tokens
    remaining = await token_manager.get_balance(query.chat_id, "documents")
    if remaining < 100:
        raise HTTPException(403, "Insufficient tokens")
    
    # Process query
    try:
        result = doc_rag_engine.query(query.question, top_k=query.top_k or 15)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")
    
    # Calculate tokens (approximate)
    tokens_used = len(query.question) // 4 + len(result["answer"]) // 4
    
    # Deduct tokens
    new_balance = await token_manager.deduct(query.chat_id, "documents", tokens_used)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        tokens_used=tokens_used,
        remaining_tokens=new_balance
    )


@router.get("/token/status")
async def doc_token_status(chat_id: str):
    """Get token status for documents module"""
    balance = await token_manager.get_balance(chat_id, "documents")
    return TokenStatusResponse(
        remaining_tokens=balance,
        max_tokens=10000,
        module="documents"
    )