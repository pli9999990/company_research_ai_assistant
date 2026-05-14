from __future__ import annotations

import sys

from langgraph.types import Command

from company_research_assistant.graph import create_graph
from company_research_assistant.search import SearchResult, StockQuote, YahooStockQuoteTool


class StubSearchTool:
    def __init__(self, result_batches):
        self.result_batches = list(result_batches)
        self.queries = []

    def search(self, query: str, max_results: int = 5):
        self.queries.append(query)
        if self.result_batches:
            return self.result_batches.pop(0)
        return []


class StubStockQuoteTool:
    def __init__(self):
        self.symbols = []

    def quote(self, symbol: str):
        self.symbols.append(symbol)
        return StockQuote(
            symbol=symbol,
            price=199.25,
            currency="USD",
            change=1.5,
            change_percent=0.76,
            market_time="2026-05-13T16:00:00+00:00",
            source="test quote feed",
        )


class StubSummaryLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return "LLM summary: prior discussion was about Nvidia stock."


class StubAnswerLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return "NVIDIA's CEO is Jensen Huang."


class StubSequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "No more stubbed responses."


class StubCompanyLLM:
    def __init__(self, company: str):
        self.company = company
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.company


class StubCompanyResolverLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "NONE"


def results_for(company: str, count: int = 3, score: float = 0.8):
    return [
        SearchResult(
            title=f"{company} result {index}",
            url=f"https://example.com/{index}",
            content=f"{company} has relevant news, financial context, and recent developments.",
            score=score,
        )
        for index in range(count)
    ]


def test_clear_query_routes_to_research_and_synthesis():
    tool = StubSearchTool([results_for("Apple")])
    graph = create_graph(search_tool=tool)
    config = {"configurable": {"thread_id": "clear-query"}}

    result = graph.invoke({"query": "Research Apple's recent developments"}, config=config)

    assert result["clarity_status"] == "clear"
    assert result["active_company"] == "Apple"
    assert result["confidence_score"] >= 6
    assert "Apple research summary" in result["final_answer"]


def test_unclear_query_interrupts_and_resumes_with_clarification():
    tool = StubSearchTool([results_for("Microsoft")])
    graph = create_graph(search_tool=tool)
    config = {"configurable": {"thread_id": "needs-clarification"}}

    interrupted = graph.invoke({"query": "Tell me recent news"}, config=config)

    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["question"] == "Which company are you asking about?"

    resumed = graph.invoke(Command(resume="Microsoft"), config=config)

    assert resumed["clarity_status"] == "clear"
    assert resumed["active_company"] == "Microsoft"
    assert "Microsoft research summary" in resumed["final_answer"]


def test_follow_up_uses_previous_company_context(tmp_path):
    tool = StubSearchTool([results_for("Nvidia"), results_for("Nvidia")])
    graph = create_graph(search_tool=tool, debug_log_path=tmp_path / "debug.log")
    config = {"configurable": {"thread_id": "follow-up"}}

    first = graph.invoke({"query": "Research Nvidia"}, config=config)
    follow_up = graph.invoke({"query": "What about their competitors?"}, config=config)

    assert first["last_company"] == "Nvidia"
    assert follow_up["active_company"] == "Nvidia"
    assert "competitors" in tool.queries[-1].lower()
    assert "previous question and answer context" in (tmp_path / "debug.log").read_text().lower()


def test_ceo_follow_up_uses_previous_stock_question_context(tmp_path):
    ceo_results = [
        SearchResult(
            title="Nvidia CEO discusses AI demand",
            url="https://example.com/nvidia-ceo",
            content="NVIDIA founder, President and CEO Jensen Huang discussed AI demand.",
            score=0.92,
        ),
        SearchResult(
            title="NVIDIA leadership",
            url="https://example.com/leadership",
            content="The company is led by CEO Jensen Huang.",
            score=0.88,
        ),
    ]
    search_tool = StubSearchTool([results_for("NVIDIA"), ceo_results])
    quote_tool = StubStockQuoteTool()
    graph = create_graph(
        search_tool=search_tool,
        stock_quote_tool=quote_tool,
        debug_log_path=tmp_path / "debug.log",
    )
    config = {"configurable": {"thread_id": "ceo-follow-up"}}

    stock_result = graph.invoke(
        {"query": "what is the stock price of NVIDIA (NVDA)?"},
        config=config,
    )
    ceo_result = graph.invoke({"query": "who is CEO?"}, config=config)

    assert stock_result["active_company"] == "NVIDIA"
    assert ceo_result["active_company"] == "NVIDIA"
    assert "CEO" in ceo_result["query"]
    assert ceo_result["final_answer"] == "NVIDIA's CEO is Jensen Huang."
    debug_log = (tmp_path / "debug.log").read_text()
    assert "what is the stock price of NVIDIA" in debug_log
    assert "agent: Clarity Agent" in debug_log
    assert "system prompt:" in debug_log
    assert "user prompt:" in debug_log
    assert "llm response:" in debug_log


def test_validator_loops_until_sufficient_or_max_attempts():
    tool = StubSearchTool(
        [
            [],
            [SearchResult("Weak", "https://example.com/weak", "No useful context", 0.1)],
            results_for("Tesla", count=2, score=0.7),
        ]
    )
    graph = create_graph(search_tool=tool)
    config = {"configurable": {"thread_id": "validator-loop"}}

    result = graph.invoke({"query": "Research Tesla"}, config=config)

    assert result["attempts"] == 3
    assert result["validation_result"] == "sufficient"
    assert "Tesla research summary" in result["final_answer"]


def test_stock_question_adds_current_quote_to_answer():
    search_tool = StubSearchTool([results_for("Apple")])
    quote_tool = StubStockQuoteTool()
    graph = create_graph(search_tool=search_tool, stock_quote_tool=quote_tool)
    config = {"configurable": {"thread_id": "stock-quote"}}

    result = graph.invoke({"query": "What is Apple's stock price?"}, config=config)

    assert quote_tool.symbols == ["AAPL"]
    assert result["stock_quote"]["price"] == 199.25
    assert "AAPL: 199.25 USD" in result["final_answer"]


def test_lowercase_known_company_is_extracted_without_llm():
    search_tool = StubSearchTool([results_for("Nvidia")])
    quote_tool = StubStockQuoteTool()
    company_llm = StubCompanyLLM("Wrong Company")
    graph = create_graph(
        search_tool=search_tool,
        stock_quote_tool=quote_tool,
        company_llm=company_llm,
    )
    config = {"configurable": {"thread_id": "lowercase-nvidia"}}

    result = graph.invoke({"query": "what is the stock price of nvidia"}, config=config)

    assert result["active_company"] == "Nvidia"
    assert quote_tool.symbols == ["NVDA"]
    assert company_llm.calls == []


def test_company_discovery_updates_last_company_for_follow_up():
    answer_llm = StubSequenceLLM(
        [
            "Apple was founded on April 1, 1976.",
            "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald G. Wayne.",
            "As of now, Nvidia is the most valuable company in the world.",
            "NVIDIA's CEO is Jensen Huang.",
        ]
    )
    search_tool = StubSearchTool(
        [
            results_for("Apple"),
            results_for("Apple"),
            results_for("Nvidia"),
            results_for("NVIDIA"),
        ]
    )
    graph = create_graph(search_tool=search_tool, answer_llm=answer_llm)
    config = {"configurable": {"thread_id": "company-discovery-follow-up"}}

    first = graph.invoke({"query": "how old is apple as a company?"}, config=config)
    second = graph.invoke({"query": "who found the company"}, config=config)
    discovery = graph.invoke(
        {"query": "can you find out which company is most valuable company in the world right now?"},
        config=config,
    )
    ceo = graph.invoke({"query": "who is the CEO?"}, config=config)

    assert first["last_company"] == "Apple"
    assert second["active_company"] == "Apple"
    assert discovery["active_company"] == "Nvidia"
    assert discovery["last_company"] == "Nvidia"
    assert ceo["active_company"] == "Nvidia"
    assert "NVIDIA's CEO is Jensen Huang." == ceo["final_answer"]


def test_company_founded_follow_up_uses_last_company_without_interrupt():
    answer_llm = StubSequenceLLM(
        [
            "The stock price of Nvidia (NVDA) is $225.83 USD.",
            "The CEO of Nvidia is Jensen Huang.",
            "Nvidia's key competitors include AMD, Intel, and Broadcom.",
            "Jensen Huang is the CEO and co-founder of Nvidia.",
            "Nvidia was founded in 1993.",
        ]
    )
    graph = create_graph(
        search_tool=StubSearchTool([results_for("Nvidia") for _ in range(5)]),
        stock_quote_tool=StubStockQuoteTool(),
        answer_llm=answer_llm,
    )
    config = {"configurable": {"thread_id": "nvidia-follow-up-founded"}}

    graph.invoke({"query": "what is the stock price of nvidia"}, config=config)
    graph.invoke({"query": "who is CEO"}, config=config)
    graph.invoke({"query": "who are the competitor"}, config=config)
    graph.invoke({"query": "tell more about the CEO"}, config=config)
    founded = graph.invoke({"query": "when was the company founded."}, config=config)

    assert "__interrupt__" not in founded
    assert founded["active_company"] == "Nvidia"
    assert founded["last_company"] == "Nvidia"
    assert founded["final_answer"] == "Nvidia was founded in 1993."


def test_follow_up_company_can_be_resolved_by_llm():
    company_llm = StubCompanyResolverLLM(["NONE", "Nvidia"])
    graph = create_graph(
        search_tool=StubSearchTool([results_for("Nvidia"), results_for("Nvidia")]),
        company_llm=company_llm,
    )
    config = {"configurable": {"thread_id": "llm-follow-up-company"}}

    first = graph.invoke({"query": "Research Nvidia"}, config=config)
    follow_up = graph.invoke({"query": "when was it founded?"}, config=config)

    assert first["active_company"] == "Nvidia"
    assert follow_up["active_company"] == "Nvidia"
    assert len(company_llm.calls) == 2
    assert "User query: when was it founded?" in company_llm.calls[0][1].content
    assert "Latest user question: when was it founded?" in company_llm.calls[1][1].content


def test_conversation_memory_keeps_recent_turns_and_compacts_old_ones():
    batches = [results_for("Nvidia") for _ in range(12)]
    graph = create_graph(search_tool=StubSearchTool(batches))
    config = {"configurable": {"thread_id": "memory-compaction"}}

    result = None
    for index in range(12):
        result = graph.invoke({"query": f"Research Nvidia update {index}"}, config=config)

    assert result is not None
    assert result["turn_count"] == 12
    assert len(result["conversation_turns"]) == 10
    assert result["conversation_turns"][0]["question"] == "Research Nvidia update 2"
    assert "Research Nvidia update 0" in result["conversation_summary"]
    assert "Research Nvidia update 1" in result["conversation_summary"]


def test_conversation_summary_uses_llm_when_provided(tmp_path):
    summary_llm = StubSummaryLLM()
    batches = [results_for("Nvidia") for _ in range(11)]
    graph = create_graph(
        search_tool=StubSearchTool(batches),
        debug_log_path=tmp_path / "debug.log",
        summary_llm=summary_llm,
    )
    config = {"configurable": {"thread_id": "llm-summary"}}

    result = None
    for index in range(11):
        result = graph.invoke({"query": f"Research Nvidia update {index}"}, config=config)

    assert result is not None
    assert summary_llm.calls
    assert result["conversation_summary"] == "LLM summary: prior discussion was about Nvidia stock."


def test_synthesis_uses_answer_llm_when_provided(tmp_path):
    answer_llm = StubAnswerLLM()
    graph = create_graph(
        search_tool=StubSearchTool([results_for("NVIDIA")]),
        answer_llm=answer_llm,
        debug_log_path=tmp_path / "debug.log",
    )
    config = {"configurable": {"thread_id": "answer-llm"}}

    result = graph.invoke({"query": "Research NVIDIA's CEO"}, config=config)

    assert answer_llm.calls
    assert result["final_answer"] == "NVIDIA's CEO is Jensen Huang."
    prompt = answer_llm.calls[0][1].content
    assert "Latest user question: Research NVIDIA's CEO" in prompt
    assert "Research findings:" in prompt


def test_company_extraction_uses_llm_fallback_when_regex_fails(tmp_path):
    company_llm = StubCompanyLLM("Advanced Micro Devices")
    graph = create_graph(
        search_tool=StubSearchTool([results_for("Advanced Micro Devices")]),
        company_llm=company_llm,
        debug_log_path=tmp_path / "debug.log",
    )
    config = {"configurable": {"thread_id": "company-llm-fallback"}}

    result = graph.invoke({"query": "research amd"}, config=config)

    assert company_llm.calls
    assert result["active_company"] == "Advanced Micro Devices"
    assert "Advanced Micro Devices research summary" in result["final_answer"]


def test_ceo_extractor_does_not_crash_on_empty_cleaned_name():
    graph = create_graph(
        search_tool=StubSearchTool(
            [
                [
                    SearchResult(
                        title="NVIDIA CEO discusses AI roadmap",
                        url="https://example.com/ceo",
                        content="The CEO discussed AI demand but the article snippet omitted the executive name.",
                        score=0.8,
                    ),
                    SearchResult(
                        title="NVIDIA leadership update",
                        url="https://example.com/leadership",
                        content="More context about NVIDIA leadership.",
                        score=0.8,
                    ),
                ]
            ]
        )
    )
    config = {"configurable": {"thread_id": "empty-ceo-name"}}

    result = graph.invoke({"query": "Research NVIDIA's CEO"}, config=config)

    assert "could not confirm NVIDIA's CEO" in result["final_answer"]


def test_search_query_stays_under_tavily_limit_with_memory_context():
    batches = [results_for("Nvidia") for _ in range(3)]
    tool = StubSearchTool(batches)
    graph = create_graph(search_tool=tool)
    config = {"configurable": {"thread_id": "search-query-limit"}}

    graph.invoke(
        {
            "query": (
                "Research Nvidia with a very detailed explanation request about "
                "stock price, data center revenue, AI chips, gross margin, CEO, "
                "competitors, supply chain, and recent news"
            )
        },
        config=config,
    )
    graph.invoke({"query": "who is CEO"}, config=config)

    assert tool.queries
    assert all(len(query) <= 400 for query in tool.queries)
    assert "Nvidia who is CEO" in tool.queries[-1]


def test_yahoo_quote_tool_falls_back_to_stooq_after_rate_limit(monkeypatch):
    class Response:
        def __init__(self, text="", status_code=200):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} Client Error: Too Many Requests")

        def json(self):
            return {}

    class Requests:
        calls = []

        @classmethod
        def get(cls, url, params=None, headers=None, timeout=None):
            cls.calls.append((url, params))
            if "finance/chart" in url:
                return Response(status_code=429)
            return Response(
                "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
                "NVDA.US,2026-05-13,20:59:59,180,182,179,181.55,1000\n"
            )

    monkeypatch.setitem(sys.modules, "requests", Requests)

    quote = YahooStockQuoteTool().quote("NVDA")

    assert quote.symbol == "NVDA"
    assert quote.price == 181.55
    assert quote.source == "Stooq"
