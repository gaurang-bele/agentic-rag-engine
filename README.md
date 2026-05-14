# Agentic RAG Engine

A small RAG API built with FastAPI, Pinecone, LangChain, and OpenRouter.

## Features

- Upload PDFs and ingest them into Pinecone
- Query the indexed documents with an OpenRouter-backed chat model
- Parse scanned/image-heavy PDFs with LlamaParse before chunking
- Multi-document retrieval with source-aware metadata filters
- Agentic web search tool via Tavily (`web_search`)
- Agentic file tools via boto3 S3 client (`read_file`, `write_file`)
- Agentic document tools: summarization and comparison (`summarize_document`, `compare_documents`)
- AgentExecutor ReAct loop (`Reason -> Act -> Observe -> Repeat`) via `/agent`
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
TAVILY_API_KEY=your_key_here
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_DEPTH=advanced
WEB_SEARCH_TOPIC=general
WEB_SEARCH_INCLUDE_ANSWER=true
WEB_SEARCH_INCLUDE_RAW_CONTENT=false
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your_bucket_name
S3_CONNECT_TIMEOUT_SECONDS=10
S3_READ_TIMEOUT_SECONDS=30
S3_MAX_ATTEMPTS=3
SUMMARIZE_MAP_REDUCE_THRESHOLD=1000
SUMMARIZE_CHUNK_SIZE=2000
SUMMARIZE_CHUNK_OVERLAP=100
COMPARE_SUMMARY_THRESHOLD=1000
AGENT_MAX_ITERATIONS=8
```

4. Start the server:

```powershell
uvicorn main:app --reload
```

## Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`
- `POST /agent`

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

## Agentic tool: web_search

`tools/web_search.py` provides a Tavily-backed tool function and a LangChain `StructuredTool`:

```python
from tools import web_search, get_web_search_tool

result = web_search("latest updates on retrieval-augmented generation")
tool = get_web_search_tool()
```

## Agentic tools: read_file and write_file

`tools/file_io.py` provides S3/local read and S3 write tools:

```python
from tools import read_file, write_file, get_read_file_tool, get_write_file_tool

content = read_file("notes.txt", source="local")
s3_obj = read_file("s3://your-bucket/path/input.txt", source="s3")
saved = write_file("hello", s3_key="outputs/hello.txt")
```

`read_file(..., source="auto")` checks local first, then falls back to S3 if AWS credentials are configured.

## Agentic tools: summarize_document and compare_documents

```python
from tools import summarize_document, compare_documents

summary = summarize_document("docs/design.txt", source="local", max_words=200)
comparison = compare_documents(
    doc1_key="docs/v1.txt",
    doc2_key="docs/v2.txt",
    doc1_source="local",
    doc2_source="local",
    focus="both",
)
```

## Agent endpoint (ReAct with verbose tracing)

`/agent` runs a LangChain `AgentExecutor` with `verbose=True` and tools:
`web_search`, `read_file`, `write_file`, `summarize_document`, `compare_documents`, `answer_question`.

Example payload:

```json
{
  "input": "Read docs/design_v2.txt, summarize it, and compare with docs/design_v1.txt"
}
```

Response includes:
- `output` (final answer)
- `intermediate_steps` (tool-by-tool reasoning trace for debugging)
