import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def ingest_pdf(pdf_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Create index if it doesn't exist
    index_name = os.getenv("PINECONE_INDEX", "rag-index")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # MiniLM outputs 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

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
