import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from retriever import get_retriever

load_dotenv()

def get_rag_chain():
    # OpenRouter uses OpenAI-compatible API
    model_name = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct-v0.3")
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Agentic RAG API"),
        },
        temperature=0
    )

    retriever = get_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # stuffs all chunks into context
        retriever=retriever,
        return_source_documents=True  # shows which chunks were used
    )
    return chain

def answer(question: str):
    chain = get_rag_chain()
    result = chain({"query": question})
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['result']}")
    print(f"\nSources:")
    for doc in result['source_documents']:
        print(f"  - Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")
    
    return result

if __name__ == "__main__":
    answer("What is this document about?")
