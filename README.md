# Agentic RAG Engine

A small RAG API built with FastAPI, Pinecone, LangChain, and OpenRouter.

## Features

- Upload PDFs and ingest them into Pinecone
- Query the indexed documents with an OpenRouter-backed chat model
- Simple FastAPI endpoints for health, ingest, and query

## Tech Stack

- FastAPI
- LangChain
- Pinecone
- OpenRouter
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
CHUNK_SIZE=500
CHUNK_OVERLAP=50
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
