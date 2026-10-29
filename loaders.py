"""Turn uploaded files into clean, chunked LangChain Documents."""
from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n(?=[a-z0-9(])", " ", text)  # re-flow PDF lines broken mid-sentence
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: Path, display_name: str | None = None) -> list[Document]:
    """Load one file into page-level Documents with tidy metadata."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif ext in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    else:
        raise ValueError(f"{ext} files aren't supported. Use PDF, DOCX, TXT or MD.")

    name = display_name or path.name
    cleaned = []
    for d in docs:
        content = _clean(d.page_content)
        if not content:
            continue
        page = d.metadata.get("page")
        cleaned.append(
            Document(
                page_content=content,
                metadata={"source": name, "page": (page + 1) if isinstance(page, int) else None},
            )
        )
    return cleaned


def load_bytes(data: bytes, filename: str) -> list[Document]:
    """Load an in-memory upload (e.g. from Streamlit) via a temp file."""
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return load_file(tmp_path, display_name=filename)
    finally:
        tmp_path.unlink(missing_ok=True)


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
        # Chroma rejects None metadata values
        c.metadata = {k: v for k, v in c.metadata.items() if v is not None}
    return chunks
