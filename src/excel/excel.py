"""
Excel RAG Routes
"""
from fastapi import APIRouter, HTTPException

from src.excel.rag_engine import excel_rag_engine
from src.token_db import token_manager
from src.models.schemas import QueryRequest, QueryResponse, TokenStatusResponse

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_excel(query: QueryRequest):
    """Ask question to Excel RAG"""
    if not excel_rag_engine.is_loaded:
        raise HTTPException(503, "Excel engine not loaded")
    
    # Check tokens
    remaining = await token_manager.get_balance(query.chat_id, "excel")
    if remaining < 100:
        raise HTTPException(403, "Insufficient tokens")
    
    # Process query
    try:
        result = excel_rag_engine.query(query.question, top_k=query.top_k or 10)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")
    
    # Calculate tokens
    tokens_used = len(query.question) // 4 + len(result["answer"]) // 4
    
    # Deduct tokens
    new_balance = await token_manager.deduct(query.chat_id, "excel", tokens_used)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        tokens_used=tokens_used,
        remaining_tokens=new_balance
    )


@router.get("/token/status")
async def excel_token_status(chat_id: str):
    """Get token status for excel module"""
    balance = await token_manager.get_balance(chat_id, "excel")
    return TokenStatusResponse(
        remaining_tokens=balance,
        max_tokens=10000,
        module="excel"
    )