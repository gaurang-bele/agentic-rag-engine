import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def get_embedding_device() -> str:
    return os.getenv("EMBEDDING_DEVICE", "cpu")


def get_normalize_embeddings() -> bool:
    return _env_bool("EMBEDDING_NORMALIZE", True)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=get_embedding_model_name(),
        model_kwargs={"device": get_embedding_device()},
        encode_kwargs={"normalize_embeddings": get_normalize_embeddings()},
    )
