# src/config.py
"""Unified Configuration for Excel RAG + Document RAG systems"""

import os
from pathlib import Path

# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# API + LLM CONFIG
# ============================================================

# ============================================================
# DOCUMENT RAG (AvalAI)
# ============================================================

DOC_API_KEY = os.getenv(
    "DOC_API_KEY",
    "aa-PAjywPshbJP8wsilGuB5r3SAPDMjb4fawYQ5VDGHtC8ZroQj"
)

DOC_BASE_URL = os.getenv(
    "DOC_BASE_URL",
    "https://api.avalai.ir/v1"
)

DOC_LLM_MODEL = os.getenv(
    "DOC_LLM_MODEL",
    "gpt-4o-mini"
)

DOC_EMBEDDING_MODEL = os.getenv(
    "DOC_EMBEDDING_MODEL",
    "text-embedding-3-large"
)

# ============================================================
# EXCEL RAG (GapGPT)
# ============================================================

EXCEL_API_KEY = os.getenv(
    "EXCEL_API_KEY",
    "aa-PAjywPshbJP8wsilGuB5r3SAPDMjb4fawYQ5VDGHtC8ZroQj"
    #"sk-kFIVdGGKBeqL3Eb7nZ67CVBcNiuo5KEid6S8meSfNzcQPrsZ"
)

EXCEL_BASE_URL = os.getenv(
    "EXCEL_BASE_URL",
    "https://api.avalai.ir/v1"
    #"https://api.gapgpt.app/v1"
)

EXCEL_LLM_MODEL = os.getenv(
    "EXCEL_LLM_MODEL",
    "gpt-4o-mini"
    #"gapgpt-qwen-3.5"
)

# Excel embedding (local E5)
E5_MODEL_PATH = os.getenv(
    "E5_MODEL_PATH",
    "/mnt/storage-1/home/sadra/AISadra/user-4/e5-embedding/e5-model"
)

# ============================================================
# EXCEL RAG CONFIG (LEGACY)
# ============================================================

EXCEL_INDEX_PATH = DATA_DIR / "vector.index"
EXCEL_METADATA_PATH = DATA_DIR / "metadata.pkl"
EXCEL_EMBEDDINGS_PATH = DATA_DIR / "final_embeddings.jsonl"

EXCEL_SOURCE_FILES = [
    DATA_DIR / "PT - 3607-32-74-ED-IN-DS-7301-A4-rag_chunks.jsonl",
    DATA_DIR / "CV - 3607-32-74-ED-IN-DS-7401-A3g-rag_chunks.jsonl",
]

# ============================================================
# DOCUMENT RAG CONFIG
# ============================================================

DOCUMENTS_INDEX_DIR = DATA_DIR / "documents" / "faiss_index"
DOCUMENTS_PDF_DIR = DATA_DIR / "documents" / "raw"

DOCUMENTS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_PDF_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# GENERAL RAG SETTINGS
# ============================================================

DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

DEFAULT_TOKENS = int(os.getenv("DEFAULT_TOKENS", "10000"))
TOKEN_WARNING_THRESHOLD = int(os.getenv("TOKEN_WARNING_THRESHOLD", "3000"))
