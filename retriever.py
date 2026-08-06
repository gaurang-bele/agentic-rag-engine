import os
from functools import lru_cache

import pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore as PineconeStore
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from embedding_config import get_embeddings

load_dotenv(override=True)

def get_pinecone_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "llamaparse-v1")

def get_top_k() -> int:
    value = int(os.getenv("RETRIEVER_TOP_K", "5"))
    return max(value, 1)

def get_fetch_k() -> int:
    value = int(os.getenv("RETRIEVER_FETCH_K", "20"))
    return max(value, 1)

def get_reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

def get_reranker_enabled() -> bool:
    value = os.getenv("ENABLE_RERANKING", "true")
    return value.strip().lower() in {"1", "true", "yes", "on"}

@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeStore:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Compatibility shim for langchain_community (expects module-level pinecone.Index type)
    if not hasattr(pinecone, "list_indexes"):
        pinecone.list_indexes = pc.list_indexes

    index_name = os.getenv("PINECONE_INDEX", "rag-index")
    index = pc.Index(index_name)
    if not hasattr(pinecone, "Index") or not isinstance(pinecone.Index, type):
        pinecone.Index = type(index)

    embeddings = get_embeddings()
    namespace = get_pinecone_namespace()
    
    vectorstore = PineconeStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(get_reranker_model_name())

def rerank_documents(question: str, documents: list, top_k: int) -> list:
    if not documents:
        return []

    try:
        pairs = [(question, doc.page_content) for doc in documents]
        scores = get_reranker().predict(pairs)
        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        best_docs = []
        for doc, score in ranked[:top_k]:
            doc.metadata = dict(doc.metadata)
            doc.metadata["rerank_score"] = round(float(score), 4)
            best_docs.append(doc)
        return best_docs
    except Exception as exc:
        print(f"Warning: Reranking failed ({exc}); falling back to vector similarity results.")
        return documents[:top_k]

def get_adaptive_fetch_k(target_top_k: int, estimated_pages: int = 10) -> int:
    # Smart Adaptive fetch_k:
    # If estimated document page count > 50, fetch 20 candidates for deep coverage.
    # If standard document <= 50 pages, fetch 10 candidates for 2x faster reranking speed.
    if estimated_pages > 50:
        return max(20, target_top_k * 4)
    return max(10, target_top_k * 2)

def retrieve_documents(
    question: str,
    metadata_filter: dict | None = None,
    top_k: int | None = None,
    rerank: bool | None = None,
) -> list:
    final_top_k = max(top_k or get_top_k(), 1)
    
    # Adaptive candidate fetch
    fetch_k = get_adaptive_fetch_k(target_top_k=final_top_k, estimated_pages=15)

    docs = get_vectorstore().similarity_search(
        question,
        k=fetch_k,
        filter=metadata_filter,
    )
    if not docs:
        return []

    # If initial vector search returns > 15 chunks (large doc), dynamically upgrade candidate pool
    if len(docs) >= 15 and fetch_k < 20:
        fetch_k = 20
        docs = get_vectorstore().similarity_search(
            question,
            k=fetch_k,
            filter=metadata_filter,
        )

    should_rerank = get_reranker_enabled() if rerank is None else rerank
    if should_rerank:
        return rerank_documents(question, docs, final_top_k)
    return docs[:final_top_k]

