"""Bulk-index a folder of documents from the command line.

Usage:
    python scripts/ingest.py sample_docs
    python scripts/ingest.py path/to/your/docs --store Pinecone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.loaders import SUPPORTED_EXTENSIONS, file_hash, load_file, split_documents  # noqa: E402
from src.vectorstore import KnowledgeBase, get_embeddings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--store", choices=["ChromaDB", "Pinecone"], default=None)
    args = ap.parse_args()

    s = Settings()
    if args.store:
        s.vector_store = args.store
    kb = KnowledgeBase(s.vector_store, s.embedding_provider,
                       get_embeddings(s.embedding_provider), s.pinecone_index)

    files = [p for p in sorted(args.folder.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        sys.exit(f"No PDF/DOCX/TXT/MD files found in {args.folder}")

    for path in files:
        fid = file_hash(path.read_bytes())
        if kb.has_file(fid):
            print(f"  skip   {path.name} (already indexed)")
            continue
        pages = load_file(path)
        chunks = split_documents(pages, s.chunk_size, s.chunk_overlap)
        n = kb.add_file(fid, path.name, chunks, pages=len(pages))
        print(f"  added  {path.name}: {len(pages)} pages -> {n} chunks")
    print(f"Done. {len(kb.files())} files, {kb.chunk_count} chunks in {s.vector_store}.")


if __name__ == "__main__":
    main()
