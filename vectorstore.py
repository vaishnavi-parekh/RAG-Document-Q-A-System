"""Vector store layer: ChromaDB (local, default) or Pinecone (cloud).

A small JSON manifest tracks which files are indexed, so the UI can list,
de-duplicate and delete documents the same way for both backends.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .config import DATA_DIR, EMBEDDING_MODELS


# ---------------------------------------------------------------- embeddings
def get_embeddings(provider: str) -> Embeddings:
    model = EMBEDDING_MODELS[provider]
    if provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI embeddings need OPENAI_API_KEY in your .env file.")
        return OpenAIEmbeddings(model=model)

    from langchain_community.embeddings import FastEmbedEmbeddings

    # Runs on CPU, downloads a ~130 MB model once, then works offline.
    try:
        return FastEmbedEmbeddings(model_name=model, cache_dir=str(DATA_DIR / "models"))
    except Exception as exc:
        raise RuntimeError(
            "Couldn't download the local embedding model. The first run needs internet "
            "access (about 130 MB, downloaded once). Check your connection and restart, "
            "or set EMBEDDING_PROVIDER=OpenAI in .env."
        ) from exc


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------- store
class KnowledgeBase:
    """Wraps a LangChain vector store plus a manifest of indexed files."""

    def __init__(self, backend: str, embedding_provider: str, embeddings: Embeddings,
                 pinecone_index: str = "campus-assistant"):
        self.backend = backend
        self.embeddings = embeddings
        key = f"{_slug(backend)}_{_slug(embedding_provider)[:12]}"
        self.manifest_path = DATA_DIR / f"manifest_{key}.json"
        self.manifest: dict = self._read_manifest()

        if backend == "Pinecone":
            self.store = self._init_pinecone(pinecone_index)
        else:
            from langchain_chroma import Chroma

            self.store: VectorStore = Chroma(
                collection_name=f"campus_{key}",
                embedding_function=embeddings,
                persist_directory=str(DATA_DIR / "chroma"),
                collection_metadata={"hnsw:space": "cosine"},
            )

    # -- pinecone setup
    def _init_pinecone(self, index_name: str) -> VectorStore:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Pinecone needs PINECONE_API_KEY in your .env file.")
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        pc = Pinecone(api_key=api_key)
        if index_name not in pc.list_indexes().names():
            dim = len(self.embeddings.embed_query("dimension probe"))
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                    region=os.getenv("PINECONE_REGION", "us-east-1")),
            )
        return PineconeVectorStore(index=pc.Index(index_name), embedding=self.embeddings)

    # -- manifest
    def _read_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    # -- public API
    def has_file(self, fhash: str) -> bool:
        return fhash in self.manifest

    def add_file(self, fhash: str, name: str, chunks: list[Document], pages: int) -> int:
        ids = [str(uuid.uuid4()) for _ in chunks]
        for c in chunks:
            c.metadata["file_id"] = fhash
        for start in range(0, len(chunks), 100):  # batch to stay under API limits
            self.store.add_documents(chunks[start:start + 100], ids=ids[start:start + 100])
        self.manifest[fhash] = {
            "name": name,
            "chunks": len(chunks),
            "pages": pages,
            "ids": ids,
            "added": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        self._save_manifest()
        return len(chunks)

    def delete_file(self, fhash: str) -> None:
        entry = self.manifest.pop(fhash, None)
        if entry and entry["ids"]:
            self.store.delete(ids=entry["ids"])
        self._save_manifest()

    def files(self) -> list[dict]:
        return [{"id": k, **{x: v[x] for x in ("name", "chunks", "pages", "added")}}
                for k, v in self.manifest.items()]

    @property
    def chunk_count(self) -> int:
        return sum(v["chunks"] for v in self.manifest.values())

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self.manifest:
            return []
        return self.store.similarity_search_with_relevance_scores(query, k=k)
