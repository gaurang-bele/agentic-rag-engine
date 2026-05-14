import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from rag_chain import get_llm
from retriever import retrieve_documents
from tools import (
    get_compare_documents_tool,
    get_read_file_tool,
    get_summarize_tool,
    get_web_search_tool,
    get_write_file_tool,
)

load_dotenv()


def _extract_text(response) -> str:
    content = getattr(response, "content", response)
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


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"string", "none", "null"}:
        return None
    return normalized


def _normalize_optional_positive_int(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _build_metadata_filter(source: str | None, page_from: int | None, page_to: int | None) -> dict | None:
    metadata_filter: dict = {}
    if source:
        metadata_filter["source"] = {"$eq": source}

    page_filter: dict = {}
    if page_from is not None:
        page_filter["$gte"] = page_from
    if page_to is not None:
        page_filter["$lte"] = page_to
    if page_filter:
        metadata_filter["page"] = page_filter

    return metadata_filter or None


def _build_answer_prompt(question: str, source_documents: list) -> str:
    if not source_documents:
        return (
            "You are a helpful assistant. No relevant context was retrieved from the knowledge base. "
            "Tell the user you could not find enough relevant information and ask for a clearer question.\n\n"
            f"Question: {question}"
        )

    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')} | Source {doc.metadata.get('source', '?')}]\n{doc.page_content}"
        for doc in source_documents
    )
    return (
        "You are a helpful assistant. Use only the provided context to answer the question. "
        "If the context is insufficient, explicitly say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def answer_question(
    question: str,
    source: str | None = None,
    page_from: int | None = None,
    page_to: int | None = None,
    top_k: int | None = None,
    rerank: bool = True,
) -> dict:
    normalized_source = _normalize_optional_string(source)
    normalized_page_from = _normalize_optional_positive_int(page_from)
    normalized_page_to = _normalize_optional_positive_int(page_to)
    normalized_top_k = _normalize_optional_positive_int(top_k)

    if normalized_page_from is not None and normalized_page_to is not None and normalized_page_from > normalized_page_to:
        raise ValueError("page_from must be less than or equal to page_to")

    metadata_filter = _build_metadata_filter(
        source=normalized_source,
        page_from=normalized_page_from,
        page_to=normalized_page_to,
    )
    source_documents = retrieve_documents(
        question=question,
        metadata_filter=metadata_filter,
        top_k=normalized_top_k,
        rerank=rerank,
    )
    llm = get_llm()
    response = llm.invoke(_build_answer_prompt(question=question, source_documents=source_documents))
    answer_text = _extract_text(response).strip()

    return {
        "answer": answer_text,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "content": doc.page_content[:400],
            }
            for doc in source_documents
        ],
    }


class AnswerQuestionInput(BaseModel):
    question: str = Field(..., description="Question to answer from retrieved knowledge")
    source: str | None = Field(default=None, description="Optional source filename filter")
    page_from: int | None = Field(default=None, description="Optional start page (1-based)")
    page_to: int | None = Field(default=None, description="Optional end page (1-based)")
    top_k: int | None = Field(default=None, description="Optional number of chunks to use")
    rerank: bool = Field(default=True, description="Enable reranking")


def get_answer_question_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="answer_question",
        description=(
            "Answer a question using the internal RAG knowledge base. "
            "Supports optional source and page metadata filters."
        ),
        func=answer_question,
        args_schema=AnswerQuestionInput,
    )


def get_agent_tools() -> list:
    return [
        get_web_search_tool(),
        get_read_file_tool(),
        get_write_file_tool(),
        get_summarize_tool(),
        get_compare_documents_tool(),
        get_answer_question_tool(),
    ]


def _get_agent_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """You are an agentic RAG assistant.
You can reason step by step, call tools, observe results, and continue until you can provide the best final answer.

You have access to the following tools:
{tools}

You MUST respond using ONLY one of these two formats.

Format A (tool use):
Question: the user input question
Thought: think about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action (use JSON for tool args when appropriate)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat many times)

Format B (final answer):
Question: the user input question
Thought: I now know the final answer
Final Answer: the final answer to the user

Rules:
- ALWAYS put Action or Final Answer on its own line immediately after Thought.
- NEVER output Action Input without an Action line directly above it.
- NEVER wrap your response in code fences or JSON objects with "thought"/"action" keys.
- If you are done, use "Final Answer:" (do NOT write "Action: Final Answer").

Question: {input}
Thought:{agent_scratchpad}"""
    )


def _get_max_iterations() -> int:
    value = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
    if value <= 0:
        raise ValueError("AGENT_MAX_ITERATIONS must be greater than zero")
    return value


def _build_agent_executor(verbose: bool) -> AgentExecutor:
    llm = get_llm()
    tools = get_agent_tools()
    agent = create_react_agent(llm=llm, tools=tools, prompt=_get_agent_prompt())
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=_get_max_iterations(),
    )


@lru_cache(maxsize=1)
def _get_verbose_agent_executor() -> AgentExecutor:
    return _build_agent_executor(verbose=True)


def get_agent_executor(verbose: bool = True) -> AgentExecutor:
    if verbose:
        return _get_verbose_agent_executor()
    return _build_agent_executor(verbose=False)


def _serialize_intermediate_steps(steps) -> list[dict]:
    serialized: list[dict] = []
    for action, observation in steps:
        serialized.append(
            {
                "tool": getattr(action, "tool", None),
                "tool_input": getattr(action, "tool_input", None),
                "log": getattr(action, "log", None),
                "observation": observation,
            }
        )
    return serialized


def run_agent(user_input: str) -> dict:
    normalized_input = user_input.strip()
    if not normalized_input:
        raise ValueError("Agent input cannot be empty")

    executor = get_agent_executor(verbose=True)
    result = executor.invoke({"input": normalized_input})
    return {
        "output": result.get("output", ""),
        "intermediate_steps": _serialize_intermediate_steps(result.get("intermediate_steps", [])),
    }
