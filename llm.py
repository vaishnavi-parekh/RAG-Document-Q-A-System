"""Chat model factory: OpenAI, Groq (free tier, fast) or Ollama (fully local)."""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

KEY_FOR_PROVIDER = {"OpenAI": "OPENAI_API_KEY", "Groq": "GROQ_API_KEY", "Ollama": None}


def provider_ready(provider: str) -> tuple[bool, str]:
    """Return (ready, message) so the UI can explain what's missing."""
    key = KEY_FOR_PROVIDER[provider]
    if key and not os.getenv(key):
        return False, f"Add {key} to your .env file, then restart the app."
    return True, ""


def get_chat_model(provider: str, model: str, temperature: float = 0.1) -> BaseChatModel:
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)
    if provider == "Ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature,
                          base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise ValueError(f"Unknown provider: {provider}")
