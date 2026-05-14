"""Interactive CLI for the company research assistant."""

from __future__ import annotations

from uuid import uuid4

from dotenv import load_dotenv
from langgraph.types import Command

from company_research_assistant.graph import create_graph


GRAPH_ASCII = r"""
LangGraph State Diagram

START
  |
  v
Clarity Agent
  |-- needs_clarification --> Human Clarification Interrupt
  |                             |
  |                             v
  `-- clear -----------------> Research Agent
                                |
                                |-- confidence_score < 6 --> Validator Agent
                                |                       |
                                |                       |-- insufficient and attempts < 3 --.
                                |                       |                                      |
                                |                       `-- sufficient or max attempts --------|
                                |                                                              |
                                `-- confidence_score >= 6 --------------------------------------v
                                                                                         Synthesis Agent
                                                                                              |
                                                                                              v
                                                                                             END
"""


def main() -> None:
    load_dotenv()
    graph = create_graph()
    config = {"configurable": {"thread_id": f"company-research-{uuid4()}"}}
    print("Company Research Assistant")
    print(GRAPH_ASCII)
    print("Type a company research question, or 'quit' to exit.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue

        result = graph.invoke({"query": query}, config=config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            clarification = input(f"{payload['question']} ").strip()
            result = graph.invoke(Command(resume=clarification), config=config)

        print(f"\nAssistant:\n{result['final_answer']}")


if __name__ == "__main__":
    main()
