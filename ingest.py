import os
from hashlib import sha1
from pathlib import Path
from typing import Callable

import pinecone
from dotenv import load_dotenv
from embedding_config import get_embedding_model_name, get_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore as PineconeStore
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec

from pypdf import PdfReader


load_dotenv(override=True)

INGESTED_HASHES_CACHE: set[str] = set()

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def compute_file_sha256(file_path: str) -> str:
    hasher = sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_llamaparse_api_key() -> str:
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing LlamaParse API key. Set LLAMA_CLOUD_API_KEY (or LLAMAPARSE_API_KEY) in .env."
        )
    return api_key

def get_pinecone_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "llamaparse-v1")

def get_clear_namespace_before_ingest() -> bool:
    if os.getenv("PINECONE_CLEAR_NAMESPACE_BEFORE_INGEST") is not None:
        return _env_bool("PINECONE_CLEAR_NAMESPACE_BEFORE_INGEST", False)
    if os.getenv("PINECONE_CLEAR_BEFORE_INGEST") is not None:
        return _env_bool("PINECONE_CLEAR_BEFORE_INGEST", False)
    return False

def get_replace_source_before_ingest() -> bool:
    return _env_bool("PINECONE_REPLACE_SOURCE_ON_INGEST", True)

def _normalize_source_name(source_name: str | None, pdf_path: str) -> str:
    candidate = (source_name or "").strip()
    if not candidate:
        candidate = Path(pdf_path).name
    return candidate

def _build_source_id(source_name: str) -> str:
    return sha1(source_name.lower().encode("utf-8")).hexdigest()[:16]

def _normalize_page_number(raw_page: object, fallback_page: int, zero_based: bool) -> int:
    if isinstance(raw_page, int):
        return raw_page + 1 if zero_based else raw_page
    return fallback_page

def load_pdf_with_pypdf(pdf_path: str, source_name: str, source_id: str) -> list[Document]:
    reader = PdfReader(pdf_path)
    pages: list[Document] = []
    for index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        metadata = {
            "source": source_name,
            "source_id": source_id,
            "page": index + 1,
            "parser": "pypdf",
        }
        pages.append(Document(page_content=text, metadata=metadata))
    print(f"Loaded {len(pages)} pages with fast PyPDF parser")
    return pages

def load_pdf_with_llamaparse(pdf_path: str, source_name: str, source_id: str) -> list[Document]:
    parser = LlamaParse(
        api_key=get_llamaparse_api_key(),
        result_type=os.getenv("LLAMAPARSE_RESULT_TYPE", "markdown"),
    )
    parsed_docs = parser.load_data(pdf_path)

    raw_page_values = [
        getattr(parsed_doc, "metadata", {}).get("page")
        for parsed_doc in parsed_docs
        if isinstance(getattr(parsed_doc, "metadata", {}).get("page"), int)
    ]
    zero_based_pages = bool(raw_page_values) and min(raw_page_values) == 0

    pages: list[Document] = []
    for index, parsed_doc in enumerate(parsed_docs):
        text = (getattr(parsed_doc, "text", "") or "").strip()
        if not text:
            continue

        metadata = dict(getattr(parsed_doc, "metadata", {}) or {})
        metadata["source"] = source_name
        metadata["source_id"] = source_id
        metadata["parser"] = "llamaparse"
        metadata["page"] = _normalize_page_number(
            metadata.get("page"),
            fallback_page=index + 1,
            zero_based=zero_based_pages,
        )
        pages.append(Document(page_content=text, metadata=metadata))

    if not pages:
        raise ValueError("LlamaParse did not return any extractable text for this PDF.")

    print(f"Loaded {len(pages)} parsed sections with LlamaParse")
    return pages

def load_pdf_documents(
    pdf_path: str,
    source_name: str,
    source_id: str,
    parser_type: str = "llamaparse",
) -> tuple[list[Document], str]:
    mode = (parser_type or "llamaparse").strip().lower()
    if mode == "pypdf":
        pages = load_pdf_with_pypdf(pdf_path, source_name, source_id)
        if not pages:
            raise ValueError("PyPDF could not extract any text from this PDF.")
        return pages, "pypdf"

    # Default & Primary: LlamaParse for maximum Markdown quality & table preservation
    return load_pdf_with_llamaparse(pdf_path, source_name, source_id), "llamaparse"



def get_chunking_settings(chunk_size: int | None = None, chunk_overlap: int | None = None) -> tuple[int, int]:
    size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE", "500"))
    overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", "50"))

    if size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero")
    if overlap < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative")
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return size, overlap

def get_embedding_dimension(embeddings) -> int:
    return len(embeddings.embed_query("dimension probe"))

def ingest_pdf(
    pdf_path: str,
    source_name: str | None = None,
    parser_type: str = "llamaparse",
    progress_callback: Callable | None = None,
) -> dict:
    def _notify(progress: int, message: str):
        if progress_callback:
            try:
                progress_callback(progress, message)
            except Exception as exc:
                print(f"Progress callback warning: {exc}")

    normalized_source_name = _normalize_source_name(source_name, pdf_path)
    source_id = _build_source_id(normalized_source_name)
    file_hash = compute_file_sha256(pdf_path)

    if file_hash in INGESTED_HASHES_CACHE:
        _notify(100, f"File '{normalized_source_name}' already ingested (SHA-256 hash match).")
        print(f"Skipping re-ingestion for '{normalized_source_name}' (hash match)")
        return {
            "status": "already_ingested",
            "source": normalized_source_name,
            "source_id": source_id,
            "chunks_count": 0,
        }

    _notify(15, f"Reading and parsing PDF '{normalized_source_name}'...")
    # 1. Load PDF
    pages, used_parser = load_pdf_documents(
        pdf_path=pdf_path,
        source_name=normalized_source_name,
        source_id=source_id,
        parser_type=parser_type,
    )
    _notify(45, f"Extracted {len(pages)} pages using {used_parser}. Splitting text into chunks...")

    # 2. Split into chunks
    chunk_size, chunk_overlap = get_chunking_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    _notify(65, f"Generated {len(chunks)} chunks. Initializing Pinecone vectorstore...")

    # 3. Create embeddings model
    embeddings = get_embeddings()
    embedding_dimension = get_embedding_dimension(embeddings)

    # 4. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    namespace = get_pinecone_namespace()

    # Compatibility shim for langchain_community (expects module-level pinecone.Index type)
    if not hasattr(pinecone, "list_indexes"):
        pinecone.list_indexes = pc.list_indexes

    index_name = os.getenv("PINECONE_INDEX", "rag-index")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)
    if not hasattr(pinecone, "Index") or not isinstance(pinecone.Index, type):
        pinecone.Index = type(index)
    stats = index.describe_index_stats() or {}
    index_dimension = stats.get("dimension") if isinstance(stats, dict) else None
    if index_dimension and index_dimension != embedding_dimension:
        raise ValueError(
            f"Embedding model '{get_embedding_model_name()}' has dimension {embedding_dimension}, "
            f"but Pinecone index '{index_name}' is {index_dimension}. Use a compatible embedding model "
            "or create a new Pinecone index."
        )

    namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else {}
    namespace_exists = namespaces.get(namespace, {}).get("vector_count", 0) > 0

    if get_clear_namespace_before_ingest():
        if namespace_exists:
            index.delete(delete_all=True, namespace=namespace)
            print(f"Cleared namespace '{namespace}' before ingestion")
        else:
            print(f"Namespace '{namespace}' is empty or missing; skipping delete")
    elif get_replace_source_before_ingest() and namespace_exists:
        index.delete(
            namespace=namespace,
            filter={"source_id": {"$eq": source_id}},
        )
        print(f"Removed existing vectors for source '{normalized_source_name}'")

    _notify(80, f"Upserting {len(chunks)} vectors into Pinecone namespace '{namespace}'...")
    # 5. Store chunks in Pinecone
    chunk_ids = [
        sha1(
            f"{chunk.metadata.get('source_id', source_id)}|"
            f"{chunk.metadata.get('page', 'na')}|{idx}".encode("utf-8")
        ).hexdigest()
        for idx, chunk in enumerate(chunks)
    ]

    vectorstore = PineconeStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name,
        namespace=namespace,
        ids=chunk_ids,
    )
    INGESTED_HASHES_CACHE.add(file_hash)

    _notify(100, f"Successfully stored {len(chunks)} chunks in Pinecone namespace '{namespace}'.")
    print(
        f"Stored {len(chunks)} chunks in namespace '{namespace}' "
        f"for source '{normalized_source_name}' using {used_parser}"
    )
    return {
        "status": "completed",
        "source": normalized_source_name,
        "source_id": source_id,
        "parser": used_parser,
        "pages_count": len(pages),
        "chunks_count": len(chunks),
        "file_hash": file_hash,
    }

if __name__ == "__main__":
    ingest_pdf("your_document.pdf")

