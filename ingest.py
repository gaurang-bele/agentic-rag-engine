import os

import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

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

def ingest_pdf(pdf_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # 2. Split into chunks
    chunk_size, chunk_overlap = get_chunking_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # 3. Create embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Compatibility shim for langchain_community (expects module-level pinecone.Index type)
    if not hasattr(pinecone, "list_indexes"):
        pinecone.list_indexes = pc.list_indexes
    
    # Create index if it doesn't exist
    index_name = os.getenv("PINECONE_INDEX", "rag-index")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # MiniLM outputs 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Ensure pinecone.Index is a type (langchain_community uses isinstance checks)
    index = pc.Index(index_name)
    if not hasattr(pinecone, "Index") or not isinstance(pinecone.Index, type):
        pinecone.Index = type(index)

    # 5. Store chunks in Pinecone
    vectorstore = PineconeStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    )
    print(f"Stored {len(chunks)} chunks in Pinecone")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf("your_document.pdf")
