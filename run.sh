#!/bin/bash

# ============================================
# Nargan AI Platform - Startup Script
# ============================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Nargan AI Platform v2.0${NC}"
echo -e "${GREEN}========================================${NC}"

# Check virtualenv
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

source venv/bin/activate

# Install dependencies
echo -e "\n${YELLOW}📥 Checking dependencies...${NC}"
pip install -q -r requirements.txt || {
    echo -e "${YELLOW}⚠️  Some packages may already be installed${NC}"
}

# Check directories
echo -e "\n${YELLOW}🔍 Checking data directories...${NC}"
mkdir -p data/documents/raw data/documents/faiss_index data/excel

# Check Excel index
if [ ! -f "data/excel/vector.index" ]; then
    echo -e "${RED}⚠️  Excel FAISS index not found!${NC}"
    echo -e "${YELLOW}   Run: python3 -m src.index_builder${NC}"
else
    echo -e "${GREEN}   ✅ Excel index found${NC}"
fi

# Check Document index
if [ ! -f "data/documents/faiss_index/index.faiss" ]; then
    echo -e "${YELLOW}⚠️  Document FAISS index not found!${NC}"
    echo -e "${YELLOW}   PDFs will be processed on first run${NC}"
else
    echo -e "${GREEN}   ✅ Document index found${NC}"
fi

# Check model
if [ ! -d "/mnt/storage-1/home/sadra/AISadra/user-4/e5-embedding/e5-model" ]; then
    echo -e "${RED}❌ E5 model not found at expected path!${NC}"
    exit 1
else
    echo -e "${GREEN}   ✅ E5 model found${NC}"
fi

# Start server
echo -e "\n${GREEN}🚀 Starting Nargan AI Platform...${NC}"
echo -e "${GREEN}   API: http://localhost:9005${NC}"
echo -e "${GREEN}   Docs: http://localhost:9005/docs${NC}"
echo -e "${GREEN}   Modules: /api/documents, /api/excel${NC}"
echo -e "${GREEN}========================================${NC}\n"

export E5_MODEL_PATH=/mnt/storage-1/home/sadra/AISadra/user-4/e5-embedding/e5-model

uvicorn api.main:app --host 0.0.0.0 --port 9005 --reload