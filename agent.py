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


from tools.gmail import get_send_email_tool


def get_agent_tools() -> list:
    return [
        get_web_search_tool(),
        get_read_file_tool(),
        get_write_file_tool(),
        get_summarize_tool(),
        get_compare_documents_tool(),
        get_answer_question_tool(),
        get_send_email_tool(),
    ]



def _get_agent_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """You are an expert autonomous RAG and Web Agent.
You can reason step by step, call tools, observe results, and continue until you provide the final answer.

You have access to the following tools:
{tools}

You MUST respond using ONLY one of these two formats.

Format A (tool use):
Question: the user input question
Thought: think about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON or plain string on a single line)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation cycle can repeat)

Format B (final answer):
Question: the user input question
Thought: I now know the final answer
Final Answer: the final answer to the user

EXAMPLES:

Example 1:
Question: Search web for AI news and save to a file
Thought: I will first search the web for AI news.
Action: web_search
Action Input: {{"query": "latest AI news 2026"}}
Observation: WEB SEARCH RESULTS...
Thought: Now I will save this summary to a file.
Action: write_file
Action Input: {{"content": "Summary of AI news...", "s3_key": "outputs/ai_news.txt"}}
Observation: File successfully saved locally to outputs/ai_news.txt
Thought: I now know the final answer
Final Answer: I have searched the web for AI news, summarized the findings, and saved the result to outputs/ai_news.txt.

Rules:
- ALWAYS put Action Input directly below Action.
- NEVER wrap Action Input in markdown ```json ``` code fences.
- ALWAYS use valid tool names from [{tool_names}].
- If you are finished, use "Final Answer:".

Question: {input}
Thought:{agent_scratchpad}"""
    )


def _get_max_iterations() -> int:
    value = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
    if value <= 0:
        raise ValueError("AGENT_MAX_ITERATIONS must be greater than zero")
    return value


import re
import json
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class RobustReActOutputParser(BaseOutputParser[AgentAction | AgentFinish]):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        cleaned = text.strip()

        # Check for Final Answer
        if "Final Answer:" in cleaned:
            answer = cleaned.split("Final Answer:", 1)[1].strip()
            return AgentFinish(return_values={"output": answer}, log=text)

        # Regex search for Action and Action Input
        action_match = re.search(r"Action:\s*([^\n]+)", cleaned)
        action_input_match = re.search(r"Action Input:\s*([\s\S]+)", cleaned)

        if not action_match or not action_input_match:
            # Fallback if text contains an answer without "Final Answer:" prefix
            if "Thought:" in cleaned and not action_match:
                thought_parts = cleaned.split("Thought:")
                return AgentFinish(return_values={"output": thought_parts[-1].strip()}, log=text)
            raise OutputParserException(f"Could not parse LLM output: {text}")

        action = action_match.group(1).strip().strip("`").strip("'").strip('"')
        raw_input = action_input_match.group(1).strip()

        # Strip code fences ```json ... ```
        if "```" in raw_input:
            raw_input = re.sub(r"```(?:json)?", "", raw_input).strip("` \n")

        # Extract first valid JSON object or clean string
        if raw_input.startswith("{") and "}" in raw_input:
            end_idx = raw_input.rfind("}") + 1
            json_str = raw_input[:end_idx]
            try:
                action_input = json.loads(json_str)
            except json.JSONDecodeError:
                action_input = raw_input
        else:
            action_input = raw_input

        # Log format MUST start after Question to prevent scratchpad duplication loops
        log_text = cleaned
        if "Thought:" in cleaned:
            log_text = "Thought:" + cleaned.split("Thought:", 1)[1]

        return AgentAction(tool=action, tool_input=action_input, log=log_text)

    @property
    def _type(self) -> str:
        return "robust_react"



from rag_chain import get_llm, get_agent_llm


def _build_agent_executor(verbose: bool) -> AgentExecutor:
    llm = get_agent_llm()
    tools = get_agent_tools()
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=_get_agent_prompt(),
        output_parser=RobustReActOutputParser(),
    )
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
        tool_name = getattr(action, "tool", None)
        # Filter out internal parser exception steps
        if tool_name == "_Exception":
            continue
        serialized.append(
            {
                "tool": tool_name,
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

