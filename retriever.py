import os

import pinecone
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone

load_dotenv()

def get_retriever():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Compatibility shim for langchain_community (expects module-level pinecone.Index type)
    if not hasattr(pinecone, "list_indexes"):
        pinecone.list_indexes = pc.list_indexes

    index_name = os.getenv("PINECONE_INDEX", "rag-index")
    index = pc.Index(index_name)
    if not hasattr(pinecone, "Index") or not isinstance(pinecone.Index, type):
        pinecone.Index = type(index)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = PineconeStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Returns top 3 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})
