import json
import os
import re

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from rag_chain import get_llm

from .file_io import read_file
from .summarize import summarize_document

load_dotenv()


def _count_words(text: str) -> int:
    return len(text.split())


def _resolve_source_label(read_result: dict, fallback: str) -> str:
    source_kind = read_result.get("source")
    if source_kind == "local":
        return read_result.get("path", fallback)
    if source_kind == "s3":
        return f"s3://{read_result.get('bucket')}/{read_result.get('key')}"
    return fallback


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


def _normalize_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _get_compare_summary_threshold() -> int:
    value = int(os.getenv("COMPARE_SUMMARY_THRESHOLD", "1000"))
    if value <= 0:
        raise ValueError("COMPARE_SUMMARY_THRESHOLD must be greater than zero")
    return value


def _extract_json_object(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*|^```|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM comparison output did not contain a JSON object")

    json_str = cleaned[start : end + 1]
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError("LLM comparison JSON must be an object")
    return parsed


def _prepare_document_for_comparison(doc_key: str, doc_source: str) -> tuple[str, str] | tuple[None, dict]:
    read_result = read_file(path_or_key=doc_key, source=doc_source)
    if read_result.get("error"):
        return None, {
            "source": _resolve_source_label(read_result=read_result, fallback=doc_key),
            "error": read_result.get("error"),
            "message": read_result.get("message"),
            "hint": read_result.get("hint"),
            "nearby_keys": read_result.get("nearby_keys", []),
        }

    content = read_result.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, {
            "source": _resolve_source_label(read_result=read_result, fallback=doc_key),
            "error": "EmptyContent",
            "message": f"Document is empty or invalid: {doc_key}",
        }

    source_label = _resolve_source_label(read_result=read_result, fallback=doc_key)
    if _count_words(content) > _get_compare_summary_threshold():
        summarized = summarize_document(path_or_key=doc_key, source=doc_source)
        if summarized.get("error"):
            return None, {
                "source": source_label,
                "error": summarized.get("error"),
                "message": summarized.get("message"),
                "hint": summarized.get("hint"),
            }

        summary_text = summarized.get("summary", "")
        if not isinstance(summary_text, str) or not summary_text.strip():
            return None, {
                "source": source_label,
                "error": "SummarizationFailed",
                "message": f"Summarization failed for document: {doc_key}",
            }
        return source_label, summary_text

    return source_label, content


def _build_compare_prompt(doc1_text: str, doc2_text: str, focus: str) -> str:
    return (
        "Compare the following two documents.\n"
        "Identify: major similarities, key differences, contradictory statements, missing information.\n"
        f"Comparison focus: {focus}.\n"
        "Return ONLY valid JSON using this exact schema:\n"
        "{\n"
        '  "similarities": ["..."],\n'
        '  "differences": ["..."],\n'
        '  "contradictions": ["..."],\n'
        '  "overall_assessment": "..."\n'
        "}\n\n"
        f"Document 1:\n{doc1_text}\n\n"
        f"Document 2:\n{doc2_text}\n"
    )


def compare_documents(
    doc1_key: str,
    doc2_key: str | None = None,
    doc1_source: str = "auto",
    doc2_source: str = "auto",
    focus: str = "differences",
) -> dict:
    doc1_value = doc1_key
    doc2_value = doc2_key
    doc1_source_value = doc1_source
    doc2_source_value = doc2_source
    focus_value = focus

    if not doc2_value:
        stripped = (doc1_key or "").strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                parsed_doc1 = parsed.get("doc1_key")
                if isinstance(parsed_doc1, str):
                    doc1_value = parsed_doc1

                parsed_doc2 = parsed.get("doc2_key")
                if isinstance(parsed_doc2, str):
                    doc2_value = parsed_doc2

                parsed_doc1_source = parsed.get("doc1_source")
                if isinstance(parsed_doc1_source, str) and parsed_doc1_source.strip():
                    doc1_source_value = parsed_doc1_source

                parsed_doc2_source = parsed.get("doc2_source")
                if isinstance(parsed_doc2_source, str) and parsed_doc2_source.strip():
                    doc2_source_value = parsed_doc2_source

                parsed_focus = parsed.get("focus")
                if isinstance(parsed_focus, str) and parsed_focus.strip():
                    focus_value = parsed_focus

    if not doc2_value:
        raise ValueError("doc2_key is required")

    normalized_focus = focus_value.strip().lower()
    if normalized_focus not in {"differences", "similarities", "both"}:
        raise ValueError("focus must be one of: differences, similarities, both")

    doc1_label, doc1_payload = _prepare_document_for_comparison(doc1_value, doc1_source_value)
    if doc1_label is None:
        return {"error": "ReadFailed", "doc": "doc1", **doc1_payload}

    doc2_label, doc2_payload = _prepare_document_for_comparison(doc2_value, doc2_source_value)
    if doc2_label is None:
        return {"error": "ReadFailed", "doc": "doc2", **doc2_payload}

    doc1_text = doc1_payload
    doc2_text = doc2_payload

    llm = get_llm()
    response = llm.invoke(_build_compare_prompt(doc1_text=doc1_text, doc2_text=doc2_text, focus=normalized_focus))
    parsed = _extract_json_object(_extract_llm_text(response))

    similarities = _normalize_string_list(parsed.get("similarities"))
    differences = _normalize_string_list(parsed.get("differences"))
    contradictions = _normalize_string_list(parsed.get("contradictions"))
    overall_assessment = str(parsed.get("overall_assessment", "")).strip()

    if normalized_focus == "similarities":
        differences = []
        contradictions = []
    elif normalized_focus == "differences":
        similarities = []

    return {
        "doc1": doc1_label,
        "doc2": doc2_label,
        "focus": normalized_focus,
        "similarities": similarities,
        "differences": differences,
        "contradictions": contradictions,
        "overall_assessment": overall_assessment,
        "doc1_chars": len(doc1_text),
        "doc2_chars": len(doc2_text),
    }


class CompareDocumentsInput(BaseModel):
    doc1_key: str = Field(..., description="Local path, S3 key, or s3://bucket/key for document 1")
    doc2_key: str = Field(..., description="Local path, S3 key, or s3://bucket/key for document 2")
    doc1_source: str = Field(default="auto", description="auto, local, or s3")
    doc2_source: str = Field(default="auto", description="auto, local, or s3")
    focus: str = Field(default="differences", description="differences, similarities, or both")

    @model_validator(mode="before")
    @classmethod
    def _recover_from_nested_json_string(cls, data):
        if not isinstance(data, dict):
            return data

        if data.get("doc1_key") and data.get("doc2_key"):
            return data

        raw = data.get("doc1_key")
        if not isinstance(raw, str):
            return data

        stripped = raw.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return data

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return data

        if not isinstance(parsed, dict):
            return data

        recovered = dict(data)
        for key in ("doc1_key", "doc2_key", "doc1_source", "doc2_source", "focus"):
            value = recovered.get(key)
            if (value is None or value == "") and key in parsed:
                recovered[key] = parsed[key]
        return recovered


def get_compare_documents_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="compare_documents",
        description=(
            "Compare two documents from local file or S3 and return structured similarities, "
            "differences, contradictions, and an overall assessment."
        ),
        func=compare_documents,
        args_schema=CompareDocumentsInput,
    )
