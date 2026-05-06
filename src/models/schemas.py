"""Pydantic schemas for API"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ==================== Common ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    documents_loaded: bool
    excel_loaded: bool
    documents_count: int
    excel_count: int


class TokenStatusResponse(BaseModel):
    """Token status response"""
    remaining_tokens: int
    max_tokens: int
    used_tokens: int
    module: str


# ==================== Documents Module ====================

class DocumentQueryRequest(BaseModel):
    """Request schema for Document RAG query"""
    question: str = Field(..., min_length=1, description="User's question")
    chat_id: str = Field(default="default", description="Chat session ID")
    top_k: int = Field(default=15, ge=1, le=50, description="Number of results to return")


class DocumentSource(BaseModel):
    """Document source information"""
    pdf_name: str
    page: int


class DocumentQueryResponse(BaseModel):
    """Response schema for Document RAG query"""
    answer: str
    sources: List[DocumentSource]
    tokens_used: int
    remaining_tokens: int


# ==================== Excel Module ====================

class ExcelQueryRequest(BaseModel):
    """Request schema for Excel RAG query"""
    question: str = Field(..., min_length=1, description="User's question")
    chat_id: str = Field(default="default", description="Chat session ID")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")


class ExcelQueryResponse(BaseModel):
    """Response schema for Excel RAG query"""
    answer: str
    sources: List[str]
    remembered_entities: List[str]
    results_count: int
    tokens_used: int
    remaining_tokens: int


# ==================== Generic (for frontend compatibility) ====================

class QueryRequest(BaseModel):
    """Generic request (used by frontend)"""
    question: str = Field(..., min_length=1)
    chat_id: str = Field(default="default")
    top_k: Optional[int] = 10


class QueryResponse(BaseModel):
    """Generic response (used by frontend)"""
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int
    remaining_tokens: int