"""Central configuration, read from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default model per LLM provider (all can be overridden in the UI or .env)
DEFAULT_MODELS = {
    "OpenAI": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "Groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "Ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
}

EMBEDDING_MODELS = {
    "Local (FastEmbed, free)": os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    "OpenAI": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
}


def _detect_default_llm() -> str:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit in DEFAULT_MODELS:
        return explicit
    if os.getenv("GROQ_API_KEY"):
        return "Groq"
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI"
    return "Ollama"


@dataclass
class Settings:
    llm_provider: str = field(default_factory=_detect_default_llm)
    llm_model: str = ""
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "Local (FastEmbed, free)")
    vector_store: str = os.getenv("VECTOR_STORE", "ChromaDB")  # "ChromaDB" or "Pinecone"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    pinecone_index: str = os.getenv("PINECONE_INDEX", "campus-assistant")

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = DEFAULT_MODELS[self.llm_provider]
