"""
RAG API Routes with Token Management
"""
from fastapi import APIRouter, HTTPException, Request
from src.models.schemas import QueryRequest, QueryResponse
from src.rag_engine import rag_engine
from src.token_db import (
    get_client_ip,
    get_or_create_user,
    get_user_tokens,
    use_tokens,
    DEFAULT_TOKENS
)

router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/ask", response_model=QueryResponse)
async def ask_rag(request: Request, query: QueryRequest):
    # ... (ابتدای تابع بدون تغییر) ...
    
    # Get client IP and create/get user
    client_ip = get_client_ip(request)
    client_id = query.chat_id
    
    user_data = get_or_create_user(client_ip, client_id)
    user_id = user_data['user']['id']
    
    # Check token balance
    token_info = get_user_tokens(user_id)
    remaining = token_info['total_tokens'] - token_info['used_tokens']
    
    # Rough estimate of tokens needed
    estimated_tokens = max(100, len(query.question) // 4 + 500)
    
    if remaining < estimated_tokens:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "Insufficient tokens",
                "remaining": remaining,
                "required": estimated_tokens
            }
        )

    try:
        # Get RAG response
        result = rag_engine.query(
            user_question=query.question,
            chat_id=query.chat_id,
            k=query.k
        )
        
        # Deduct tokens
        token_result = use_tokens(
            user_id=user_id,
            question=query.question,
            answer=result['answer']
        )
        
        # Add remaining tokens to response
        result['remaining_tokens'] = token_result.get('remaining', remaining)
        result['tokens_used'] = token_result.get('tokens_used', 0)
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/token/status")
async def get_token_status(request: Request):
    """Get current user's token status"""
    client_ip = get_client_ip(request)
    user_data = get_or_create_user(client_ip)
    user_id = user_data['user']['id']
    
    token_info = get_user_tokens(user_id)
    
    return {
        "ip_address": client_ip,
        "max_tokens": token_info['total_tokens'],
        "used_tokens": token_info['used_tokens'],
        "remaining_tokens": token_info['total_tokens'] - token_info['used_tokens'],
        "default_tokens": DEFAULT_TOKENS
    }


@router.post("/token/reset/{user_id}")
async def reset_tokens(user_id: int, new_amount: int = DEFAULT_TOKENS):
    """Reset user tokens (admin only - should add auth)"""
    from src.token_db import reset_user_tokens
    
    result = reset_user_tokens(user_id, new_amount)
    return {
        "message": "Tokens reset successfully",
        **result
    }


@router.get("/token/history")
async def get_token_history(request: Request, limit: int = 20):
    """Get user's token usage history"""
    client_ip = get_client_ip(request)
    user_data = get_or_create_user(client_ip)
    user_id = user_data['user']['id']
    
    from src.token_db import get_user_history
    history = get_user_history(user_id, limit)
    
    return {"history": history}


@router.get("/stats")
async def get_stats():
    """Get RAG system statistics"""
    return {
        "document_count": rag_engine.document_count,
        "is_loaded": rag_engine.is_loaded,
        "active_conversations": len(rag_engine.conversations)
    }


@router.post("/reset/{chat_id}")
async def reset_conversation(chat_id: str):
    """Reset conversation history for a specific chat ID"""
    if chat_id in rag_engine.conversations:
        del rag_engine.conversations[chat_id]
        return {"message": f"Conversation {chat_id} reset successfully"}
    return {"message": f"No conversation found for {chat_id}"}