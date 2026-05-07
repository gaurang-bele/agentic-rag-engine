import logging
import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ingest import ingest_pdf
from rag_chain import answer

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Agentic RAG API")

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_pdf(temp_path)
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
        result = answer(request.question)
        return {
            "answer": result["result"],
            "sources": [
                {
                    "page": doc.metadata.get("page"),
                    "content": doc.page_content[:200]
                }
                for doc in result["source_documents"]
            ]
        }
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/")
async def root():
    return {"message": "OK", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "running"}