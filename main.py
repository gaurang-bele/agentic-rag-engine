from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
from ingest import ingest_pdf
from rag_chain import answer

app = FastAPI(title="Agentic RAG API")

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ingest_pdf(temp_path)
    return {"message": f"Successfully ingested {file.filename}"}

@app.post("/query")
async def query_document(request: QueryRequest):
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

@app.get("/health")
async def health():
    return {"status": "running"}