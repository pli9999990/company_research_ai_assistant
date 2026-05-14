"""LLM helpers for optional OpenAI-backed synthesis and memory."""

from __future__ import annotations

import os
from typing import Any


def default_openai_llm(model: str | None = None) -> Any | None:
    """Return a ChatOpenAI model when OPENAI_API_KEY is configured.

    The import stays lazy so local tests and offline runs can still use the
    deterministic fallback path without installing or configuring OpenAI.
    """

    if not os.environ.get("OPENAI_API_KEY"):
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install langchain-openai to use OpenAI-backed synthesis."
        ) from exc

    return ChatOpenAI(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
        temperature=0.2,
        max_tokens=500,
    )
