import os
from functools import lru_cache

import pinecone
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from embedding_config import get_embeddings

load_dotenv()

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

def retrieve_documents(
    question: str,
    metadata_filter: dict | None = None,
    top_k: int | None = None,
    rerank: bool | None = None,
) -> list:
    final_top_k = max(top_k or get_top_k(), 1)
    fetch_k = max(get_fetch_k(), final_top_k)

    docs = get_vectorstore().similarity_search(
        question,
        k=fetch_k,
        filter=metadata_filter,
    )
    if not docs:
        return []

    should_rerank = get_reranker_enabled() if rerank is None else rerank
    if should_rerank:
        return rerank_documents(question, docs, final_top_k)
    return docs[:final_top_k]
