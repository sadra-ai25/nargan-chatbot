#!/bin/bash

# ============================================
# RAG System - Startup Script (Offline Mode)
# ============================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   RAG System - Startup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if virtualenv exists
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtualenv
echo -e "\n${YELLOW}🔄 Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies WITHOUT internet (use existing packages)
echo -e "\n${YELLOW}📥 Installing dependencies...${NC}"

pip install --upgrade pip --quiet 2>/dev/null || true

# Install packages - ignore errors if already installed
pip install faiss-cpu --quiet 2>/dev/null || true
pip install numpy --quiet 2>/dev/null || true
pip install sentence-transformers --quiet 2>/dev/null || true
pip install fastapi --quiet 2>/dev/null || true
pip install uvicorn --quiet 2>/dev/null || true
pip install pydantic --quiet 2>/dev/null || true
pip install httpx --quiet 2>/dev/null || true
pip install requests --quiet 2>/dev/null || true

echo -e "${GREEN}✅ Dependencies ready${NC}"

# Check data files
echo -e "\n${YELLOW}🔍 Checking data files...${NC}"

# Check if embeddings exist
if [ ! -f "data/final_embeddings.jsonl" ]; then
    echo -e "${YELLOW}⚠️  Embeddings not found!${NC}"
    echo -e "${YELLOW}   Run: python3 -m src.embed${NC}"
else
    echo -e "${GREEN}   ✅ Embeddings found${NC}"
fi

# Check if index exists
if [ ! -f "data/vector.index" ]; then
    echo -e "${YELLOW}⚠️  FAISS index not found!${NC}"
    echo -e "${YELLOW}   Run: python3 -m src.index_builder${NC}"
else
    echo -e "${GREEN}   ✅ FAISS index found${NC}"
fi

# Start the server
echo -e "\n${GREEN}🚀 Starting RAG API Server...${NC}"
echo -e "${GREEN}   API: http://localhost:9005${NC}"
echo -e "${GREEN}   Docs: http://localhost:9005/docs${NC}"
echo -e "${GREEN}========================================${NC}\n"

uvicorn api.main:app --host 0.0.0.0 --port 9005 --reload