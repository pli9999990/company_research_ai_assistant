"""LangGraph implementation of the multi-agent company research assistant."""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from company_research_assistant.llm import default_openai_llm
from company_research_assistant.search import (
    CompanySearchTool,
    SearchResult,
    StockQuoteTool,
    default_search_tool,
    default_stock_quote_tool,
)

ClarityStatus = Literal["clear", "needs_clarification"]
ValidationResult = Literal["sufficient", "insufficient"]
MEMORY_WINDOW = 10


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    effective_query: str
    active_company: str | None
    last_company: str | None
    clarity_status: ClarityStatus
    clarification_request: str | None
    research_findings: list[dict[str, Any]]
    stock_quote: dict[str, Any] | None
    confidence_score: float
    validation_result: ValidationResult
    attempts: int
    max_attempts: int
    final_answer: str
    conversation_turns: list[dict[str, str]]
    conversation_summary: str
    turn_count: int


COMMON_COMPANY_SUFFIXES = {
    "Inc",
    "Inc.",
    "Corp",
    "Corp.",
    "Corporation",
    "Company",
    "Co",
    "Co.",
    "Ltd",
    "Ltd.",
    "LLC",
    "PLC",
    "AG",
    "SA",
}

QUESTION_WORDS = {
    "Research",
    "Tell",
    "What",
    "Who",
    "Which",
    "How",
    "Give",
    "Find",
    "Show",
    "Summarize",
    "Compare",
    "Analyze",
    "Latest",
    "Recent",
}

LEADING_NON_COMPANY_WORDS = QUESTION_WORDS | {
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "about",
    "for",
    "on",
}

FOLLOW_UP_MARKERS = {
    "who",
    "what",
    "when",
    "where",
    "which",
    "they",
    "their",
    "company",
    "it",
    "its",
    "them",
    "competitors",
    "ceo",
    "leadership",
    "financials",
    "stock",
    "revenue",
    "founded",
    "founder",
    "history",
    "more",
}

NON_COMPANY_TERMS = {
    "CEO",
    "CFO",
    "CTO",
    "COO",
    "CIO",
    "CMO",
    "CPO",
    "EVP",
    "SVP",
    "President",
    "Founder",
    "Chairman",
    "Chairwoman",
    "Leadership",
    "Competitors",
    "Stock",
    "Price",
}

STOCK_MARKERS = {
    "stock",
    "share",
    "shares",
    "quote",
    "price",
    "market cap",
    "ticker",
}

KNOWN_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "berkshire hathaway": "BRK-B",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "walmart": "WMT",
    "exxon": "XOM",
    "exxon mobil": "XOM",
}


def create_graph(
    search_tool: CompanySearchTool | None = None,
    stock_quote_tool: StockQuoteTool | None = None,
    checkpointer: MemorySaver | None = None,
    debug_log_path: str | Path = "debug.log",
    summary_llm: Any | None = None,
    answer_llm: Any | None = None,
    company_llm: Any | None = None,
    llm_model: str | None = None,
):
    """Create and compile the company research graph."""

    tool = search_tool or default_search_tool()
    quote_tool = stock_quote_tool or default_stock_quote_tool()
    configured_llm = default_openai_llm(llm_model)
    summary_model = summary_llm if summary_llm is not None else configured_llm
    answer_model = answer_llm if answer_llm is not None else configured_llm
    company_model = company_llm if company_llm is not None else configured_llm
    log_path = Path(debug_log_path)

    def clarity_agent(state: ResearchState) -> ResearchState:
        query = _current_query(state)
        company = _extract_company(query, company_model)
        last_company = state.get("last_company")
        is_company_discovery = _is_company_discovery_query(query)

        if not company and is_company_discovery:
            company = "global companies"
        elif not company:
            company = _resolve_follow_up_company(query, state, company_model)

        if company:
            response = {
                "clarity_status": "clear",
                "active_company": company,
                "reason": "Company found in the query or resolved from conversation memory.",
            }
            _log_llm_trace(
                log_path,
                "Clarity Agent",
                "Decide whether the user query names a company or needs clarification.",
                _build_agent_user_prompt(state, query),
                response,
            )
            return {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "effective_query": query,
                "active_company": company,
                "last_company": None if is_company_discovery else company,
                "clarity_status": "clear",
                "clarification_request": None,
                "stock_quote": None,
                "attempts": 0,
                "validation_result": "insufficient",
                "max_attempts": state.get("max_attempts", 3),
            }

        response = {
            "clarity_status": "needs_clarification",
            "clarification_request": "Which company are you asking about?",
            "reason": "No company was found and no prior company context could be resolved.",
        }
        _log_llm_trace(
            log_path,
            "Clarity Agent",
            "Decide whether the user query names a company or needs clarification.",
            _build_agent_user_prompt(state, query),
            response,
        )
        return {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "clarity_status": "needs_clarification",
            "clarification_request": "Which company are you asking about?",
            "stock_quote": None,
            "attempts": 0,
            "validation_result": "insufficient",
            "max_attempts": state.get("max_attempts", 3),
        }

    def clarification_interrupt_agent(state: ResearchState) -> ResearchState:
        clarification = interrupt(
            {
                "question": state.get(
                    "clarification_request",
                    "Which company are you asking about?",
                ),
                "original_query": state.get("query", ""),
            }
        )
        company = str(clarification).strip()
        original_query = state.get("query", "")
        effective_query = f"{original_query} Company: {company}".strip()
        _log_llm_trace(
            log_path,
            "Human Clarification Interrupt",
            "Resume the graph with the human-provided company clarification.",
            f"Original query: {original_query}\nClarification: {company}",
            {"active_company": company, "effective_query": effective_query},
        )
        return {
            "messages": [HumanMessage(content=company)],
            "effective_query": effective_query,
            "active_company": company,
            "last_company": company,
            "clarity_status": "clear",
            "clarification_request": None,
        }

    def research_agent(state: ResearchState) -> ResearchState:
        attempts = int(state.get("attempts", 0)) + 1
        company = state.get("active_company") or ""
        query = state.get("effective_query") or state.get("query") or ""
        research_prompt = _build_research_prompt(company, query, attempts, state)
        search_query = _build_search_query(company, query, attempts)

        results = tool.search(search_query, max_results=5)
        findings = [_finding_from_result(result) for result in results]
        stock_quote = None
        if _is_stock_question(query):
            symbol = _extract_ticker(query, company)
            if symbol:
                stock_quote = asdict(quote_tool.quote(symbol))
            else:
                stock_quote = {
                    "symbol": "",
                    "price": None,
                    "source": "unavailable",
                    "error": "Could not determine a ticker symbol for this company.",
                }
        confidence = _score_confidence(findings, company, attempts)
        _log_llm_trace(
            log_path,
            "Research Agent",
            "Gather company information using search results and assign a confidence score.",
            research_prompt,
            {
                "search_query": search_query,
                "findings_count": len(findings),
                "confidence_score": confidence,
                "stock_quote": stock_quote,
            },
        )

        return {
            "research_findings": findings,
            "stock_quote": stock_quote,
            "confidence_score": confidence,
            "attempts": attempts,
        }

    def validator_agent(state: ResearchState) -> ResearchState:
        findings = state.get("research_findings", [])
        confidence = float(state.get("confidence_score", 0))
        company = state.get("active_company") or ""
        has_company_context = any(
            company.lower() in str(item.get("content", "")).lower()
            or company.lower() in str(item.get("title", "")).lower()
            for item in findings
        )
        sufficient = len(findings) >= 2 and confidence >= 4.5 and has_company_context
        response = {
            "validation_result": "sufficient" if sufficient else "insufficient",
            "findings_count": len(findings),
            "confidence_score": confidence,
            "has_company_context": has_company_context,
        }
        _log_llm_trace(
            log_path,
            "Validator Agent",
            "Review whether the research is sufficient to answer the user's question.",
            _build_agent_user_prompt(state, state.get("query", "")),
            response,
        )
        return {
            "validation_result": response["validation_result"],
        }

    def synthesis_agent(state: ResearchState) -> ResearchState:
        answer = _synthesize(state, answer_model)
        resolved_company = _resolve_company_after_answer(state, answer, company_model)
        memory_state = dict(state)
        if resolved_company:
            memory_state["active_company"] = resolved_company
        _log_llm_trace(
            log_path,
            "Synthesis Agent",
            "Create a coherent, user-friendly company research answer using conversation context.",
            _build_agent_user_prompt(state, state.get("query", "")),
            answer,
        )
        memory = _update_conversation_memory(memory_state, answer, log_path, summary_model)
        state_update: ResearchState = {
            "final_answer": answer,
            "conversation_turns": memory["conversation_turns"],
            "conversation_summary": memory["conversation_summary"],
            "turn_count": memory["turn_count"],
            "messages": [AIMessage(content=answer)],
        }
        if resolved_company:
            state_update["active_company"] = resolved_company
            state_update["last_company"] = resolved_company
        elif state.get("active_company") and not _is_placeholder_company(state["active_company"]):
            state_update["last_company"] = state.get("active_company")
        return {
            **state_update,
        }

    builder = StateGraph(ResearchState)
    builder.add_node("clarity", clarity_agent)
    builder.add_node("clarification_interrupt", clarification_interrupt_agent)
    builder.add_node("research", research_agent)
    builder.add_node("validator", validator_agent)
    builder.add_node("synthesis", synthesis_agent)

    builder.add_edge(START, "clarity")
    builder.add_conditional_edges(
        "clarity",
        _route_from_clarity,
        {
            "clarification_interrupt": "clarification_interrupt",
            "research": "research",
        },
    )
    builder.add_edge("clarification_interrupt", "research")
    builder.add_conditional_edges(
        "research",
        _route_from_research,
        {
            "validator": "validator",
            "synthesis": "synthesis",
        },
    )
    builder.add_conditional_edges(
        "validator",
        _route_from_validator,
        {
            "research": "research",
            "synthesis": "synthesis",
        },
    )
    builder.add_edge("synthesis", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def _route_from_clarity(state: ResearchState) -> Literal["clarification_interrupt", "research"]:
    if state.get("clarity_status") == "clear":
        return "research"
    return "clarification_interrupt"


def _route_from_research(state: ResearchState) -> Literal["validator", "synthesis"]:
    if float(state.get("confidence_score", 0)) < 6:
        return "validator"
    return "synthesis"


def _route_from_validator(state: ResearchState) -> Literal["research", "synthesis"]:
    attempts = int(state.get("attempts", 0))
    max_attempts = int(state.get("max_attempts", 3))
    if state.get("validation_result") == "insufficient" and attempts < max_attempts:
        return "research"
    return "synthesis"


def _current_query(state: ResearchState) -> str:
    if state.get("query"):
        return str(state["query"]).strip()
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
        if getattr(message, "type", None) == "human":
            return str(message.content).strip()
    return ""


def _extract_company(query: str, company_llm: Any | None = None) -> str | None:
    quoted = re.search(r"['\"]([^'\"]{2,80})['\"]", query)
    if quoted:
        return quoted.group(1).strip()

    known_company = _extract_known_company(query)
    if known_company:
        return known_company

    possessive = re.search(r"\b([A-Z][A-Za-z0-9&.\- ]{1,60})'s\b", query)
    if possessive:
        candidate = _clean_company_candidate(possessive.group(1))
        if candidate:
            return candidate

    suffix_pattern = "|".join(re.escape(suffix) for suffix in COMMON_COMPANY_SUFFIXES)
    suffixed = re.search(
        rf"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*\s+(?:{suffix_pattern}))\b",
        query,
    )
    if suffixed:
        return _clean_company_candidate(suffixed.group(1))

    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3}\b",
        query,
    )
    for candidate in candidates:
        cleaned = _clean_company_candidate(candidate)
        if cleaned:
            return cleaned
    return _llm_extract_company(query, company_llm)


def _extract_known_company(query: str) -> str | None:
    lower_query = query.lower()
    sorted_names = sorted(KNOWN_TICKERS, key=len, reverse=True)
    for company_name in sorted_names:
        match = re.search(rf"\b{re.escape(company_name)}\b", lower_query)
        if match:
            original_match = query[match.start() : match.end()]
            if original_match.isupper():
                return original_match
            return _display_company_name(company_name)
    return None


def _display_company_name(company_name: str) -> str:
    special_cases = {
        "jpmorgan": "JPMorgan",
        "jp morgan": "JP Morgan",
    }
    if company_name in special_cases:
        return special_cases[company_name]
    return " ".join(word.capitalize() for word in company_name.split())


def _llm_extract_company(query: str, company_llm: Any | None) -> str | None:
    if company_llm is None:
        return None

    system_prompt = (
        "Extract the company name from the user's query. Return only the company "
        "name. If no company is explicitly mentioned, return NONE. Do not infer "
        "a company from pronouns or vague follow-up questions."
    )
    user_prompt = f"User query: {query}"
    try:
        response = company_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception:
        return None

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    company = str(content).strip().strip('"').strip("'").strip()
    if not company or company.upper() == "NONE":
        return None
    return _clean_company_candidate(company) or company


def _clean_company_candidate(candidate: str) -> str | None:
    words = candidate.strip().split()
    while words and words[0].strip(" .,:;").lower() in {
        word.lower() for word in LEADING_NON_COMPANY_WORDS
    }:
        words.pop(0)
    if not words:
        return None
    cleaned = " ".join(words).strip(" .,:;")
    if len(cleaned) < 2 or cleaned in QUESTION_WORDS or cleaned in NON_COMPANY_TERMS:
        return None
    return cleaned


def _is_company_discovery_query(query: str) -> bool:
    lower_query = query.lower()
    discovery_patterns = [
        r"\bwhich company\b",
        r"\bwhat company\b",
        r"\bmost valuable company\b",
        r"\blargest company\b",
        r"\bbiggest company\b",
        r"\btop company\b",
        r"\bcompany in the world\b",
    ]
    return any(re.search(pattern, lower_query) for pattern in discovery_patterns)


def _resolve_follow_up_company(
    query: str,
    state: ResearchState,
    company_llm: Any | None,
) -> str | None:
    last_company = state.get("last_company")
    if company_llm is not None:
        company = _llm_resolve_follow_up_company(query, state, company_llm)
        if company:
            return company

    if last_company and _looks_like_follow_up_fallback(query):
        return last_company
    return None


def _llm_resolve_follow_up_company(
    query: str,
    state: ResearchState,
    company_llm: Any,
) -> str | None:
    system_prompt = (
        "Resolve whether the user's latest question refers to a company from "
        "the previous conversation. Return only the company name. If the user "
        "is asking a general question or no company can be resolved, return NONE."
    )
    user_prompt = (
        f"Conversation context:\n{_conversation_context(state)}\n\n"
        f"Last known company: {state.get('last_company') or '(none)'}\n"
        f"Latest user question: {query}"
    )
    try:
        response = company_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception:
        return None

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    company = str(content).strip().strip('"').strip("'").strip()
    if not company or company.upper() == "NONE":
        return None
    return _clean_company_candidate(company) or company


def _looks_like_follow_up_fallback(query: str) -> bool:
    lower_query = query.lower()
    if re.search(r"\b(the|this|that|their|its|it|they|them)\s+company\b", lower_query):
        return True
    if re.search(r"\b(ceo|competitors?|founder|founded|history|leadership|revenue|stock)\b", lower_query):
        return True
    if re.search(r"\b(who|what|when|where|tell me more|more about)\b", lower_query):
        return True
    return False


def _is_placeholder_company(company: str) -> bool:
    return company.lower().strip() in {"global companies", "public companies"}


def _resolve_company_after_answer(
    state: ResearchState,
    answer: str,
    company_llm: Any | None,
) -> str | None:
    query = state.get("query", "")
    if not _is_company_discovery_query(query):
        return None

    company = _extract_known_company(answer)
    if company:
        return company

    return _llm_extract_company(answer, company_llm)


def _is_stock_question(query: str) -> bool:
    lower_query = query.lower()
    return any(marker in lower_query for marker in STOCK_MARKERS)


def _extract_ticker(query: str, company: str) -> str | None:
    ticker_match = re.search(r"\b(?:NYSE|NASDAQ|ticker|symbol)[:\s]+([A-Z]{1,5}(?:-[A-Z])?)\b", query)
    if ticker_match:
        return ticker_match.group(1)

    parenthesized = re.search(r"\(([A-Z]{1,5}(?:-[A-Z])?)\)", query)
    if parenthesized:
        return parenthesized.group(1)

    standalone = re.search(r"\b[A-Z]{2,5}(?:-[A-Z])?\b", query)
    if standalone and standalone.group(0) not in QUESTION_WORDS:
        return standalone.group(0)

    normalized_company = company.lower().strip()
    if normalized_company in KNOWN_TICKERS:
        return KNOWN_TICKERS[normalized_company]

    simplified_company = re.sub(r"\b(inc|corp|corporation|company|co|ltd|llc|plc)\.?\b", "", normalized_company)
    simplified_company = re.sub(r"\s+", " ", simplified_company).strip()
    return KNOWN_TICKERS.get(simplified_company)


def _build_research_prompt(
    company: str,
    query: str,
    attempts: int,
    state: ResearchState,
) -> str:
    focus = "news financials recent developments"
    if attempts == 2:
        focus = "latest company filings leadership competitive landscape"
    elif attempts >= 3:
        focus = "comprehensive company profile market performance strategy"
    memory_context = _conversation_context(state)
    return (
        f"company research: {company}\n"
        f"previous question and answer context:\n{memory_context}\n"
        f"new user question: {query}\n"
        f"focus: {focus}"
    )


def _build_search_query(company: str, query: str, attempts: int) -> str:
    focus = "news financials recent developments"
    if attempts == 2:
        focus = "company filings leadership competitive landscape"
    elif attempts >= 3:
        focus = "company profile market performance strategy"

    search_query = f"{company} {query} {focus}".strip()
    return _shorten_for_search(search_query, 380)


def _finding_from_result(result: SearchResult) -> dict[str, Any]:
    return asdict(result)


def _score_confidence(
    findings: list[dict[str, Any]],
    company: str,
    attempts: int,
) -> float:
    if not findings:
        return 0.0

    avg_score = sum(float(item.get("score") or 0) for item in findings) / len(findings)
    result_depth = min(len(findings), 5) * 1.1
    company_mentions = sum(
        1
        for item in findings
        if company.lower() in str(item.get("title", "") + item.get("content", "")).lower()
    )
    company_bonus = min(company_mentions, 3) * 0.9
    retry_penalty = max(0, attempts - 1) * 0.4
    normalized_search_score = min(avg_score * 3, 2.5)
    return round(min(10.0, result_depth + company_bonus + normalized_search_score - retry_penalty), 1)


def _synthesize(state: ResearchState, answer_llm: Any | None = None) -> str:
    company = state.get("active_company") or "the company"
    query = state.get("query") or "your question"
    confidence = float(state.get("confidence_score", 0))
    validation = state.get("validation_result", "sufficient")
    findings = state.get("research_findings", [])
    stock_quote = state.get("stock_quote")

    llm_answer = _llm_synthesize(state, answer_llm)
    if llm_answer:
        return llm_answer

    direct_answer = _direct_answer(company, query, findings, stock_quote)
    if direct_answer:
        return direct_answer

    lines = [
        f"## {company} research summary",
        "",
        f"Query: {query}",
        f"Confidence: {confidence:.1f}/10",
    ]
    if validation == "insufficient":
        lines.append("Validation note: available results were limited, so treat this as a starting point.")

    if stock_quote:
        lines.extend(["", "Current stock quote:", _format_stock_quote(stock_quote)])

    lines.extend(["", "Key findings:"])
    if findings:
        for finding in findings[:5]:
            title = finding.get("title", "Untitled result")
            content = str(finding.get("content", "")).strip()
            url = finding.get("url", "")
            snippet = content[:260] + ("..." if len(content) > 260 else "")
            source = f" ({url})" if url else ""
            lines.append(f"- {title}: {snippet}{source}")
    else:
        lines.append("- No search results were returned.")

    lines.extend(
        [
            "",
            "Bottom line:",
            _bottom_line(company, query, findings),
        ]
    )
    return "\n".join(lines)


def _llm_synthesize(state: ResearchState, answer_llm: Any | None) -> str | None:
    if answer_llm is None:
        return None

    system_prompt = (
        "You are a concise company research assistant. Answer the user's latest "
        "question directly using the provided search findings, stock quote, and "
        "conversation context. Keep the answer short. For narrow factual "
        "questions, respond in one sentence. For broader research questions, use "
        "at most 3 bullets. Do not dump raw search findings. If evidence is "
        "insufficient, say what is missing briefly."
    )
    user_prompt = (
        f"Company: {state.get('active_company') or 'unknown'}\n"
        f"Latest user question: {state.get('query') or ''}\n\n"
        f"Conversation context:\n{_conversation_context(state)}\n\n"
        f"Stock quote:\n{state.get('stock_quote') or '(none)'}\n\n"
        f"Research findings:\n{_format_findings_for_prompt(state.get('research_findings', []))}\n\n"
        "Write the final answer only."
    )

    try:
        response = answer_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        return f"LLM synthesis unavailable: {exc}"

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    answer = str(content).strip()
    return answer or None


def _format_findings_for_prompt(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "(none)"
    lines = []
    for index, finding in enumerate(findings[:5], start=1):
        lines.append(
            f"{index}. Title: {finding.get('title', '')}\n"
            f"   URL: {finding.get('url', '')}\n"
            f"   Content: {_shorten(str(finding.get('content', '')), 500)}"
        )
    return "\n".join(lines)


def _direct_answer(
    company: str,
    query: str,
    findings: list[dict[str, Any]],
    stock_quote: dict[str, Any] | None,
) -> str | None:
    lower_query = query.lower()
    if "ceo" in lower_query:
        ceo = _extract_ceo_name(findings)
        if ceo:
            return f"{company}'s CEO is {ceo}."
        return f"I could not confirm {company}'s CEO from the available search results."

    if stock_quote and _is_stock_question(query):
        quote_line = _format_stock_quote(stock_quote).removeprefix("- ").rstrip(".")
        return f"{quote_line}."

    return None


def _extract_ceo_name(findings: list[dict[str, Any]]) -> str | None:
    text = "\n".join(
        f"{finding.get('title', '')}\n{finding.get('content', '')}"
        for finding in findings
    )
    patterns = [
        r"\bfounder,\s+President\s+and\s+CEO\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3})",
        r"\b([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,3}),?\s+(?:founder,\s+)?(?:President\s+and\s+)?CEO\b",
        r"\bCEO\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3})",
        r"\bchief executive officer\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3})",
    ]
    rejected_first_words = {
        "says",
        "said",
        "discusses",
        "joins",
        "reveals",
        "announces",
        "expects",
        "sees",
        "warns",
    }
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = _clean_person_name(match.group(1))
            if not name:
                continue
            first_word = name.split()[0].lower()
            if (
                name.lower() not in {"the company", "nvidia ceo"}
                and first_word not in rejected_first_words
            ):
                return name
    return None


def _clean_person_name(name: str) -> str:
    words = re.sub(r"\s+", " ", name.strip(" .,:;")).split()
    kept = []
    for word in words:
        cleaned = word.strip(" .,:;")
        if not cleaned:
            continue
        if not cleaned[0].isupper():
            break
        kept.append(cleaned)
    return " ".join(kept)


def _format_stock_quote(stock_quote: dict[str, Any]) -> str:
    symbol = stock_quote.get("symbol") or "unknown ticker"
    price = stock_quote.get("price")
    source = stock_quote.get("source") or "unknown source"
    error = stock_quote.get("error")
    if price is None:
        return f"- {symbol}: live quote unavailable from {source}. {error or ''}".strip()

    currency = stock_quote.get("currency") or ""
    change = stock_quote.get("change")
    change_percent = stock_quote.get("change_percent")
    market_time = stock_quote.get("market_time")
    change_text = ""
    if change is not None and change_percent is not None:
        change_text = f" ({change:+.2f}, {change_percent:+.2f}%)"
    time_text = f" as of {market_time}" if market_time else ""
    return f"- {symbol}: {price:.2f} {currency}{change_text}{time_text} via {source}."


def _bottom_line(company: str, query: str, findings: list[dict[str, Any]]) -> str:
    if not findings:
        return f"I could not gather enough evidence to answer confidently about {company}."
    lower_query = query.lower()
    if "competitor" in lower_query:
        return f"{company}'s competitive position should be evaluated against established peers, emerging challengers, and the markets highlighted in the search results."
    if "ceo" in lower_query or "leadership" in lower_query:
        return f"The leadership angle for {company} needs source-backed confirmation from current company pages or reliable recent coverage."
    if "stock" in lower_query or "quote" in lower_query or "share" in lower_query:
        return f"For {company}'s stock, use the quote as a point-in-time market snapshot and the research findings for business context."
    return f"{company} appears to be shaped by recent operating updates, market positioning, financial trends, and competitive pressure surfaced by the collected results."


def _update_conversation_memory(
    state: ResearchState,
    answer: str,
    debug_log_path: Path,
    summary_llm: Any | None,
) -> dict[str, Any]:
    turns = list(state.get("conversation_turns", []))
    summary = state.get("conversation_summary", "")
    turn = {
        "question": state.get("query", ""),
        "answer": answer,
        "company": state.get("active_company") or "",
    }
    turns.append(turn)

    if len(turns) > MEMORY_WINDOW:
        overflow = turns[:-MEMORY_WINDOW]
        turns = turns[-MEMORY_WINDOW:]
        summary = _summarize_prior_conversation(
            summary,
            overflow,
            debug_log_path,
            summary_llm,
        )

    return {
        "conversation_turns": turns,
        "conversation_summary": summary,
        "turn_count": int(state.get("turn_count", 0)) + 1,
    }


def _summarize_prior_conversation(
    existing_summary: str,
    overflow: list[dict[str, str]],
    debug_log_path: Path,
    summary_llm: Any | None,
) -> str:
    system_prompt = (
        "Summarize prior company-research Q&A turns into compact memory for "
        "future follow-up questions. Preserve companies, tickers, user intent, "
        "important answers, and unresolved caveats."
    )
    user_prompt = _format_turns_for_prompt(overflow)
    llm_user_prompt = (
        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        f"Turns to summarize:\n{user_prompt}"
    )
    summary = _llm_summarize(system_prompt, llm_user_prompt, summary_llm)
    if not summary:
        summary = _compact_summary(existing_summary, overflow)
    _log_llm_trace(
        debug_log_path,
        "Conversation Summary Agent",
        system_prompt,
        llm_user_prompt,
        summary,
    )
    return summary


def _llm_summarize(
    system_prompt: str,
    user_prompt: str,
    summary_llm: Any | None,
) -> str | None:
    if summary_llm is None:
        return None

    try:
        response = summary_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        return f"Summary LLM unavailable: {exc}"

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    return str(content).strip() or None


def _compact_summary(existing_summary: str, overflow: list[dict[str, str]]) -> str:
    summary_parts = [existing_summary.strip()] if existing_summary.strip() else []
    for turn in overflow:
        question = _shorten(turn.get("question", ""), 120)
        answer = _shorten(_plain_text(turn.get("answer", "")), 220)
        company = turn.get("company", "")
        prefix = f"{company}: " if company else ""
        summary_parts.append(f"{prefix}Q: {question} A: {answer}")
    return "\n".join(part for part in summary_parts if part)


def _conversation_context(state: ResearchState) -> str:
    parts = []
    summary = state.get("conversation_summary", "").strip()
    if summary:
        parts.append(f"Compact prior summary:\n{summary}")

    turns = state.get("conversation_turns", [])
    if turns:
        parts.append(f"Recent Q&A turns:\n{_format_turns_for_prompt(turns)}")

    return "\n\n".join(parts) if parts else "(no prior conversation)"


def _format_turns_for_prompt(turns: list[dict[str, str]]) -> str:
    lines = []
    for index, turn in enumerate(turns, start=1):
        company = turn.get("company", "")
        company_text = f"Company: {company}\n" if company else ""
        lines.append(
            f"Turn {index}:\n"
            f"{company_text}"
            f"User: {turn.get('question', '')}\n"
            f"Assistant: {_shorten(turn.get('answer', ''), 600)}"
        )
    return "\n\n".join(lines)


def _build_agent_user_prompt(state: ResearchState, query: str) -> str:
    return (
        f"Previous conversation context:\n{_conversation_context(state)}\n\n"
        f"New user question:\n{query}"
    )


def _log_llm_trace(
    debug_log_path: Path,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    llm_response: Any,
) -> None:
    response_text = llm_response if isinstance(llm_response, str) else repr(llm_response)
    entry = (
        f"\n{'=' * 80}\n"
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        f"agent: {agent_name}\n"
        f"system prompt:\n{system_prompt}\n\n"
        f"user prompt:\n{user_prompt}\n\n"
        f"llm response:\n{response_text}\n"
    )
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("#", "").replace("*", "")).strip()


def _shorten(value: str, limit: int) -> str:
    cleaned = _plain_text(value)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _shorten_for_search(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].strip()
