# Nargan RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue) ![E5-Embedding](https://img.shields.io/badge/E5-Multilingual-orange) ![Docker](https://img.shields.io/badge/Docker-Compose-blue)

Retrieval-Augmented Generation (RAG) chatbot for Nargan company engineering documents. Indexes Excel files and text documents using multilingual E5 embeddings + FAISS, then answers natural-language questions by retrieving the most relevant context and calling an external LLM API.

## Features

- **Document ingestion** — upload Excel sheets or text documents via REST API; auto-indexed into FAISS
- **Excel-aware RAG** — special pipeline for structured Excel engineering data with row/column context
- **Multilingual E5 embeddings** — local embedding model for Persian and English queries
- **FAISS vector search** — sub-millisecond nearest-neighbor retrieval over large document collections
- **External LLM integration** — sends retrieved context + query to configurable LLM endpoint (e.g., GapGPT)
- **Token-based access** — lightweight token DB for API authentication
- **Fully Dockerized** — runs locally without internet dependency (except LLM API calls)

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | Multilingual E5 (local model) |
| Vector Store | FAISS |
| LLM Backend | External API (GapGPT / OpenAI-compatible) |
| API Server | FastAPI + Uvicorn |
| Containerization | Docker Compose |

## Architecture

```
User Query (Persian/English)
        │
        ▼
   E5 Embedding Model  (local)
        │  query vector
        ▼
   FAISS Index
        │  top-K relevant chunks
        ▼
   Context Builder
        │  prompt = context + query
        ▼
   LLM API  (GapGPT / OpenAI-compatible)
        │  generated answer
        ▼
   FastAPI Response  →  User

Documents (Excel / Text)
        │  on upload
        ▼
   Chunker → E5 Embed → FAISS Index (persisted to disk)
```

## Prerequisites

- Docker & Docker Compose
- E5 multilingual embedding model downloaded locally
- Access to an OpenAI-compatible LLM API (e.g., GapGPT, Ollama, OpenAI)

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/sadra-ai25/nargan-chatbot.git
cd nargan-chatbot

# 2. Download E5 embedding model
# Place the model directory at the path specified in MODEL_PATH in .env

# 3. Configure environment
cp .env.example .env   # edit with your values

# 4. Start services
docker compose up -d --build
```

## Configuration

| Key | Description | Example |
|---|---|---|
| `API_KEY` | LLM API authentication key | `your_api_key_here` |
| `LLM_URL` | LLM API endpoint | `https://api.gapgpt.app/v1/chat/completions` |
| `LLM_MODEL` | Model name to use | `gapgpt-qwen-3.5` |
| `MODEL_PATH` | Local path to E5 embedding model | `/models/e5-multilingual` |
| `DATA_DIR` | Directory for uploaded documents | `./data` |
| `INDEX_PATH` | FAISS index file path | `./data/vector.index` |
| `METADATA_PATH` | Index metadata file | `./data/metadata.pkl` |
| `DEFAULT_K` | Number of chunks to retrieve | `10` |
| `EMBEDDING_BATCH_SIZE` | Embedding batch size | `32` |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI (browser-based chat interface) |
| `GET` | `/health` | Service health check |
| `POST` | `/api/ask/documents` | Ask a question against indexed text/PDF documents |
| `POST` | `/api/ask/excel` | Ask a question against indexed Excel data |
| `GET` | `/api/token/status/documents` | Token balance for document queries |
| `GET` | `/api/token/status/excel` | Token balance for Excel queries |
| `POST` | `/api/documents/ask` | (Router) Document Q&A with response model |
| `POST` | `/api/excel/ask` | (Router) Excel Q&A with response model |

### Example: Ask About Documents

```bash
curl -X POST http://localhost:8000/api/ask/documents \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the maximum pressure rating for valve V-101?",
    "chat_id": "user123"
  }'
```

**Response:**

```json
{
  "answer": "Valve V-101 has a maximum pressure rating of 150 PSI according to the P&ID sheet.",
  "tokens_used": 245,
  "remaining_tokens": 9755
}
```

### Example: Ask About Excel Data

```bash
curl -X POST http://localhost:8000/api/ask/excel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the flow rates for pump P-201?",
    "chat_id": "user123"
  }'
```

## Building the Index (CLI)

```bash
# Embed and index documents manually
python embed.py --data_dir ./data/documents

# Test FAISS retrieval
python faiss-1.py --query "pressure rating for V-101"

# Ask a question directly (CLI)
python ask.py "What are the flow rates for pump P-201?"
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
