import logging
import os
import shutil

import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from agent import run_agent
from ingest import ingest_pdf
from rag_chain import answer

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





ingestion_jobs: dict[str, dict] = {}

def _run_background_ingest(job_id: str, temp_path: str, filename: str, parser_type: str):
    def update_progress(percent: int, message: str):
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id]["progress"] = percent
            ingestion_jobs[job_id]["message"] = message

    try:
        update_progress(5, "Started ingestion task...")
        res = ingest_pdf(
            pdf_path=temp_path,
            source_name=filename,
            parser_type=parser_type,
            progress_callback=update_progress,
        )
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id].update({
                "status": res.get("status", "completed"),
                "progress": 100,
                "message": f"Successfully ingested {filename}",
                "result": res,
                "completed_at": datetime.now().isoformat(),
            })
    except Exception as exc:
        logger.exception("Background ingestion failed")
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id].update({
                "status": "failed",
                "progress": 100,
                "message": str(exc),
                "error": str(exc),
                "failed_at": datetime.now().isoformat(),
            })
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    source: str | None = Field(default=None, description="Optional source filename filter (leave null/empty for all files)")
    page_from: int | None = Field(default=None, description="Optional start page (1-based)")
    page_to: int | None = Field(default=None, description="Optional end page (1-based)")
    top_k: int | None = Field(default=None, description="Optional number of chunks to use")
    rerank: bool = Field(default=True, description="Enable BGE CrossEncoder reranking for maximum accuracy")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the OSI model?",
                "source": None,
                "page_from": None,
                "page_to": None,
                "top_k": 5,
                "rerank": True,
            }
        }
    }

class AgentRequest(BaseModel):
    input: str = Field(..., description="Natural language instruction for the agent")

def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"string", "none", "null", "temp_cn.pdf"}:
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
def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    parser_type: str = "llamaparse",
    wait: bool = True,
):
    job_id = str(uuid.uuid4())[:8]
    temp_path = f"temp_{job_id}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if wait:
            # 1-Step Direct Sync Ingestion (Default)
            res = ingest_pdf(
                pdf_path=temp_path,
                source_name=file.filename,
                parser_type=parser_type,
            )
            return {
                "message": f"Successfully ingested {file.filename}",
                "status": "completed",
                "result": res,
            }

        # Background Task Ingestion (Optional)
        ingestion_jobs[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "parser_type": parser_type,
            "status": "processing",
            "progress": 0,
            "message": "File received and queued for ingestion.",
            "created_at": datetime.now().isoformat(),
        }

        background_tasks.add_task(_run_background_ingest, job_id, temp_path, file.filename, parser_type)
        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing",
            "message": f"Ingestion started in background for {file.filename}. Check progress at /ingest/status/{job_id}",
        }
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if wait and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.get("/ingest/status/{job_id}")
def get_ingest_status(job_id: str):
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return ingestion_jobs[job_id]

@app.get("/ingest/jobs")
def list_ingest_jobs():
    return list(ingestion_jobs.values())


@app.post("/query")
def query_document(request: QueryRequest):
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

@app.post("/agent")
def run_agent_endpoint(request: AgentRequest):
    try:
        result = run_agent(request.input)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

from export_utils import generate_markdown_report, generate_pdf_report, save_and_upload_report


class ExportRequest(BaseModel):
    title: str = "Agentic RAG Technical Report"
    query: str
    answer: str
    sources: list[dict] | None = None
    steps: list[dict] | None = None
    format: str = "markdown"


@app.post("/export")
def export_report_endpoint(request: ExportRequest):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fmt = request.format.lower().strip()

        if fmt == "pdf":
            filename = f"rag_report_{ts}.pdf"
            content_bytes = generate_pdf_report(
                title=request.title,
                query=request.query,
                answer=request.answer,
                sources=request.sources,
                steps=request.steps,
            )
            media_type = "application/pdf"
        else:
            filename = f"rag_report_{ts}.md"
            md_text = generate_markdown_report(
                title=request.title,
                query=request.query,
                answer=request.answer,
                sources=request.sources,
                steps=request.steps,
            )
            content_bytes = md_text.encode("utf-8")
            media_type = "text/markdown"

        storage_res = save_and_upload_report(filename, content_bytes, media_type)
        return {
            "status": "success",
            "filename": filename,
            "format": fmt,
            "storage": storage_res,
            "download_url": f"/outputs/{filename}",
        }
    except Exception as exc:
        logger.exception("Export report failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if os.path.exists("outputs"):
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs_static")

@app.get("/", include_in_schema=False)
def root(request: Request):
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return RedirectResponse(url=str(request.url_for("swagger_ui_html")), status_code=302)

@app.get("/health")
def health():
    return {"status": "running"}


