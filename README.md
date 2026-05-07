# Agentic RAG Engine

A small RAG API built with FastAPI, Pinecone, LangChain, and OpenRouter.

## Features

- Upload PDFs and ingest them into Pinecone
- Query the indexed documents with an OpenRouter-backed chat model
- Parse scanned/image-heavy PDFs with LlamaParse before chunking
- Multi-document retrieval with source-aware metadata filters
- Simple FastAPI endpoints for health, ingest, and query

## Tech Stack

- FastAPI
- LangChain
- Pinecone
- OpenRouter
- LlamaParse
- sentence-transformers

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Add your API keys to `.env`:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=rag-index
PINECONE_NAMESPACE=llamaparse-v1
PINECONE_CLEAR_NAMESPACE_BEFORE_INGEST=false
PINECONE_REPLACE_SOURCE_ON_INGEST=true
LLAMA_CLOUD_API_KEY=your_key_here
LLAMAPARSE_RESULT_TYPE=markdown
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
EMBEDDING_NORMALIZE=true
RETRIEVER_TOP_K=5
RETRIEVER_FETCH_K=20
ENABLE_RERANKING=true
RERANKER_MODEL=BAAI/bge-reranker-base
OPENROUTER_MODEL=mistralai/mistral-7b-instruct-v0.3
OPENROUTER_HTTP_REFERER=http://localhost:8000
OPENROUTER_APP_TITLE=Agentic RAG API
OPENROUTER_API_KEY=your_key_here
```

4. Start the server:

```powershell
uvicorn main:app --reload
```

## Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`

## Usage

- Open `http://127.0.0.1:8000/docs`
- Upload a PDF to `/ingest`
- Ask questions through `/query`
- `/` now redirects directly to `/docs`
- `/ingest` now uses LlamaParse (no PyPDFLoader fallback)
- Ingestion writes `source` + `source_id` metadata for each chunk
- `PINECONE_REPLACE_SOURCE_ON_INGEST=true` replaces only the re-uploaded document vectors (good for multi-document indexes)

`/query` supports metadata filtering and retrieval controls:

```json
{
  "question": "What is the OSI model?",
  "source": "CN.pdf",
  "page_from": 18,
  "page_to": 30,
  "top_k": 5,
  "rerank": true
}
```

`question` is required. `source`, `page_from`, `page_to`, and `top_k` are optional.
If you send placeholder values from Swagger (for example `"source": "string"` or `0` for page/top_k),
they are ignored and query runs with defaults.

## Chunking experiment

Run the experiment script to compare chunk sizes and overlaps against `CN.pdf`:

```powershell
python chunking_experiment.py
```

If you want the answer-quality portion of the experiment, install the optional Anthropic SDK:

```powershell
pip install anthropic
```

After picking the best settings, update `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` before ingesting documents.
