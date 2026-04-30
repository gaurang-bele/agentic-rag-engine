import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore

load_dotenv()

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = PineconeStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX", "rag-index"),
        embedding=embeddings
    )
    
    # Returns top 3 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})
