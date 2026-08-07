import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_tavily_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in .env")
    return api_key


def get_default_max_results() -> int:
    value = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    return max(value, 1)


def get_default_search_depth() -> str:
    value = os.getenv("WEB_SEARCH_DEPTH", "basic").strip().lower()
    if value not in {"basic", "advanced"}:
        return "basic"
    return value



def get_default_topic() -> str:
    value = os.getenv("WEB_SEARCH_TOPIC", "general").strip().lower()
    if value not in {"general", "news"}:
        return "general"
    return value


def get_default_days() -> int | None:
    raw = os.getenv("WEB_SEARCH_DAYS", "").strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None

def get_response_output_path() -> Path:
    return Path(__file__).resolve().parent / "response.txt"

def _format_response(query: str, answer: str | None, results: list[dict]) -> str:
    lines = [
        "WEB SEARCH RESULTS",
        f"Query: {query}",
    ]

    if answer:
        lines.extend(["", "Answer:", answer])

    lines.append("")
    lines.append("Results:")
    if not results:
        lines.append("  (no results)")
        return "\n".join(lines) + "\n"

    for idx, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        content = item.get("content") or ""
        score = item.get("score")
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if score is not None:
            lines.append(f"   Score: {score}")
        if content:
            lines.append(f"   Summary: {content}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def web_search(
    query: str,
    max_results: int | None = None,
    search_depth: str | None = None,
    topic: str | None = None,
) -> dict:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("web_search requires a non-empty query")

    # Truncate query to 350 chars max for Tavily API query limits
    if len(normalized_query) > 350:
        normalized_query = normalized_query[:350].rsplit(" ", 1)[0]

    final_max_results = max_results if max_results is not None else get_default_max_results()
    final_max_results = max(final_max_results, 1)

    final_search_depth = (search_depth or get_default_search_depth()).strip().lower()
    if final_search_depth not in {"basic", "advanced"}:
        raise ValueError("search_depth must be 'basic' or 'advanced'")

    final_topic = (topic or get_default_topic()).strip().lower()
    if final_topic not in {"general", "news"}:
        raise ValueError("topic must be 'general' or 'news'")

    client = TavilyClient(api_key=get_tavily_api_key())
    response = client.search(
        query=normalized_query,
        search_depth=final_search_depth,
        topic=final_topic,
        max_results=final_max_results,
        include_answer=_env_bool("WEB_SEARCH_INCLUDE_ANSWER", True),

        include_raw_content=_env_bool("WEB_SEARCH_INCLUDE_RAW_CONTENT", False),
        days=get_default_days(),
    )

    results = [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
            "score": item.get("score"),
        }
        for item in response.get("results", [])
    ]

    output_text = _format_response(normalized_query, response.get("answer"), results)
    output_path = get_response_output_path()
    output_path.write_text(output_text, encoding="utf-8")

    return {
        "query": normalized_query,
        "answer": response.get("answer"),
        "results": results,
        "output_path": str(output_path),
    }


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int | None = Field(default=None, description="Max results to return")
    search_depth: str | None = Field(default=None, description="basic or advanced")
    topic: str | None = Field(default=None, description="general or news")


def get_web_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="web_search",
        description=(
            "Search the web using Tavily and return relevant results. "
            "Use for current events, external facts, or information not in the vector store."
        ),
        func=web_search,
        args_schema=WebSearchInput,
    )
