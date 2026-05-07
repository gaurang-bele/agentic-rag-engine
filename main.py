import logging
import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ingest import ingest_pdf
from rag_chain import answer

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Agentic RAG API")

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    source: str | None = Field(default=None, description="Optional source filename filter")
    page_from: int | None = Field(default=None, description="Optional start page (1-based)")
    page_to: int | None = Field(default=None, description="Optional end page (1-based)")
    top_k: int | None = Field(default=None, description="Optional number of chunks to use")
    rerank: bool = Field(default=True, description="Enable reranking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the OSI model?",
                "source": "temp_CN.pdf",
                "page_from": 18,
                "page_to": 30,
                "top_k": 5,
                "rerank": True,
            }
        }
    }

def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"string", "none", "null"}:
        return None
    return normalized

def _normalize_optional_positive_int(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value

def build_metadata_filter(
    source: str | None,
    page_from: int | None,
    page_to: int | None,
) -> dict | None:
    metadata_filter: dict = {}

    if source:
        metadata_filter["source"] = {"$eq": source}

    page_filter: dict = {}
    if page_from is not None:
        page_filter["$gte"] = page_from
    if page_to is not None:
        page_filter["$lte"] = page_to
    if page_filter:
        metadata_filter["page"] = page_filter

    return metadata_filter or None

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_pdf(temp_path, source_name=file.filename)
        return {"message": f"Successfully ingested {file.filename}"}
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        source = _normalize_optional_string(request.source)
        page_from = _normalize_optional_positive_int(request.page_from)
        page_to = _normalize_optional_positive_int(request.page_to)
        top_k = _normalize_optional_positive_int(request.top_k)

        if page_from is not None and page_to is not None and page_from > page_to:
            raise HTTPException(status_code=400, detail="page_from must be less than or equal to page_to")

        metadata_filter = build_metadata_filter(source, page_from, page_to)
        result = answer(
            question=request.question,
            metadata_filter=metadata_filter,
            top_k=top_k,
            rerank=request.rerank,
        )
        return {
            "answer": result["result"],
            "sources": [
                {
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "content": doc.page_content[:200],
                }
                for doc in result["source_documents"]
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/", include_in_schema=False)
async def root(request: Request):
    return RedirectResponse(url=str(request.url_for("swagger_ui_html")), status_code=302)

@app.get("/health")
async def health():
    return {"status": "running"}
