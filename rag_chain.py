import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from retriever import retrieve_documents

load_dotenv(override=True)

def get_llm():
    # OpenRouter uses OpenAI-compatible API
    model_name = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct-v0.3")
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Agentic RAG API"),
        },
        temperature=0,
    )

def _build_prompt(question: str, source_documents: list) -> str:
    if not source_documents:
        return (
            "You are a helpful assistant. No relevant context was retrieved from the knowledge base. "
            "Tell the user you could not find enough relevant information and ask for a clearer question.\n\n"
            f"Question: {question}"
        )

    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in source_documents
    )
    return (
        "You are a helpful assistant. Use only the provided context to answer the question. "
        "If the context is insufficient, explicitly say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def _extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)

def answer(
    question: str,
    metadata_filter: dict | None = None,
    top_k: int | None = None,
    rerank: bool | None = None,
):
    source_documents = retrieve_documents(
        question=question,
        metadata_filter=metadata_filter,
        top_k=top_k,
        rerank=rerank,
    )
    llm = get_llm()
    prompt = _build_prompt(question, source_documents)
    response = llm.invoke(prompt)
    result_text = _extract_text(response)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result_text}")
    print(f"\nSources:")
    for doc in source_documents:
        print(f"  - Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")
    
    return {
        "result": result_text,
        "source_documents": source_documents,
    }

if __name__ == "__main__":
    answer("What is this document about?")
