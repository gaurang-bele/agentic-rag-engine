import os

from dotenv import load_dotenv
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from rag_chain import get_llm

from .file_io import read_file

load_dotenv()


def _count_words(text: str) -> int:
    return len(text.split())


def _extract_llm_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text.strip())
        if parts:
            return "\n".join(parts).strip()

    return str(content).strip()


def _resolve_source_label(read_result: dict, fallback: str) -> str:
    source_kind = read_result.get("source")
    if source_kind == "local":
        return read_result.get("path", fallback)
    if source_kind == "s3":
        return f"s3://{read_result.get('bucket')}/{read_result.get('key')}"
    return fallback


def _get_map_reduce_threshold() -> int:
    value = int(os.getenv("SUMMARIZE_MAP_REDUCE_THRESHOLD", "1000"))
    if value <= 0:
        raise ValueError("SUMMARIZE_MAP_REDUCE_THRESHOLD must be greater than zero")
    return value


def _get_splitter_chunk_size() -> int:
    value = int(os.getenv("SUMMARIZE_CHUNK_SIZE", "2000"))
    if value <= 0:
        raise ValueError("SUMMARIZE_CHUNK_SIZE must be greater than zero")
    return value


def _get_splitter_chunk_overlap() -> int:
    value = int(os.getenv("SUMMARIZE_CHUNK_OVERLAP", "100"))
    if value < 0:
        raise ValueError("SUMMARIZE_CHUNK_OVERLAP cannot be negative")
    if value >= _get_splitter_chunk_size():
        raise ValueError("SUMMARIZE_CHUNK_OVERLAP must be smaller than SUMMARIZE_CHUNK_SIZE")
    return value


def _build_summary_prompt(max_words: int, content: str) -> str:
    return (
        "Summarize the following content in a concise, factually accurate manner. "
        "Focus on: key ideas, important findings, technical details, actionable insights. "
        "Avoid repetition and speculation. "
        f"Keep the summary under {max_words} words.\n\n"
        f"Content:\n{content}\n\n"
        "Summary:"
    )


def _run_map_reduce_summary(content: str, max_words: int) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_get_splitter_chunk_size(),
        chunk_overlap=_get_splitter_chunk_overlap(),
    )
    documents = splitter.create_documents([content])
    if not documents:
        raise ValueError("Unable to split document for map-reduce summarization")

    llm = get_llm()
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Summarize this chunk in a concise, factually accurate way. "
            "Focus on key ideas, important findings, technical details, actionable insights. "
            "Avoid repetition and speculation.\n\n"
            "{text}\n\nChunk summary:"
        ),
    )
    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are combining chunk summaries into one final summary.\n"
            "Create a concise, factually accurate final summary focused on key ideas, "
            "important findings, technical details, and actionable insights.\n"
            "Avoid repetition and speculation.\n"
            f"Keep the final summary under {max_words} words.\n\n"
            "{text}\n\nFinal summary:"
        ),
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )

    if hasattr(chain, "invoke"):
        result = chain.invoke({"input_documents": documents})
        if isinstance(result, dict) and "output_text" in result:
            return str(result["output_text"]).strip()
        if isinstance(result, str):
            return result.strip()
    if hasattr(chain, "run"):
        run_result = chain.run(documents)
        return str(run_result).strip()

    raise ValueError("Map-reduce summarization chain returned an unsupported output format")


def summarize_document(
    path_or_key: str,
    source: str = "auto",
    max_words: int = 200,
) -> dict:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    read_result = read_file(path_or_key=path_or_key, source=source)
    if read_result.get("error"):
        return {
            "source": _resolve_source_label(read_result=read_result, fallback=path_or_key),
            "error": read_result.get("error"),
            "message": read_result.get("message"),
            "hint": read_result.get("hint"),
        }

    content = read_result.get("content")
    if not isinstance(content, str) or not content.strip():
        return {
            "source": _resolve_source_label(read_result=read_result, fallback=path_or_key),
            "error": "EmptyContent",
            "message": "summarize_document requires non-empty text content",
        }

    word_count_original = _count_words(content)
    if word_count_original < _get_map_reduce_threshold():
        llm = get_llm()
        response = llm.invoke(_build_summary_prompt(max_words=max_words, content=content))
        summary = _extract_llm_text(response)
    else:
        summary = _run_map_reduce_summary(content=content, max_words=max_words)

    if not summary:
        raise ValueError("Summarization returned empty output")

    word_count_summary = _count_words(summary)
    compression_ratio = round(word_count_summary / max(word_count_original, 1), 4)

    return {
        "source": _resolve_source_label(read_result=read_result, fallback=path_or_key),
        "word_count_original": word_count_original,
        "word_count_summary": word_count_summary,
        "compression_ratio": compression_ratio,
        "summary": summary,
    }


class SummarizeInput(BaseModel):
    path_or_key: str = Field(..., description="Local path, S3 key, or s3://bucket/key")
    source: str = Field(default="auto", description="auto, local, or s3")
    max_words: int = Field(default=200, description="Maximum words in summary")


def get_summarize_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="summarize_document",
        description=(
            "Summarize a document from local file or S3. "
            "Uses single-pass summarization for short content and map-reduce for long content."
        ),
        func=summarize_document,
        args_schema=SummarizeInput,
    )
