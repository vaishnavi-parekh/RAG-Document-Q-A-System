"""The retrieval-augmented generation pipeline.

    question ──► (rewrite follow-ups using chat history)
             ──► vector search (top-k chunks)
             ──► grounded prompt with numbered sources
             ──► LLM answer that cites [1], [2] ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .vectorstore import KnowledgeBase

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Campus Assistant, a helpful guide that answers questions about the "
     "documents a student or staff member has uploaded (handbooks, policies, "
     "syllabi, research papers).\n\n"
     "Rules:\n"
     "1. Answer ONLY from the numbered context below. Never invent facts, dates, "
     "fees or rules.\n"
     "2. Cite every factual sentence with its source number in square brackets, "
     "e.g. [1] or [2][3].\n"
     "3. If the context does not contain the answer, say so plainly and suggest "
     "what document or office might have it. Do not guess.\n"
     "4. Be concise and well structured: short paragraphs, bullet points for "
     "lists or steps, bold for key numbers and deadlines.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest message as a single standalone search query that "
     "makes sense without the conversation. Keep names, numbers and terms. "
     "Return only the rewritten query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

SUGGEST_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Read these excerpts from a user's documents and write 4 short, specific "
     "questions (max 9 words each) that the excerpts can clearly answer. "
     "One question per line, no numbering, no extra text."),
    ("human", "{excerpts}"),
])


@dataclass
class Source:
    number: int
    name: str
    page: int | None
    score: float
    text: str


@dataclass
class RAGResponse:
    stream: Iterator[str]
    sources: list[Source] = field(default_factory=list)
    search_query: str = ""


def _history_messages(history: list[dict], limit: int = 6):
    msgs = []
    for m in history[-limit:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    return msgs


def _format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        where = f"{s.name}, page {s.page}" if s.page else s.name
        blocks.append(f"[{s.number}] ({where})\n{s.text}")
    return "\n\n---\n\n".join(blocks)


class CampusRAG:
    def __init__(self, kb: KnowledgeBase, llm: BaseChatModel, top_k: int = 4):
        self.kb = kb
        self.llm = llm
        self.top_k = top_k
        self.parser = StrOutputParser()

    def _standalone(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        chain = CONDENSE_PROMPT | self.llm | self.parser
        rewritten = chain.invoke({"history": _history_messages(history), "question": question})
        return rewritten.strip().strip('"') or question

    def retrieve(self, query: str) -> list[Source]:
        hits: list[tuple[Document, float]] = self.kb.search(query, k=self.top_k)
        return [
            Source(number=i + 1, name=d.metadata.get("source", "document"),
                   page=d.metadata.get("page"), score=max(0.0, min(1.0, score)),
                   text=d.page_content)
            for i, (d, score) in enumerate(hits)
        ]

    def ask(self, question: str, history: list[dict]) -> RAGResponse:
        query = self._standalone(question, history)
        sources = self.retrieve(query)
        context = _format_context(sources) if sources else "(no relevant passages found)"
        chain = ANSWER_PROMPT | self.llm | self.parser
        stream = chain.stream({
            "context": context,
            "history": _history_messages(history),
            "question": question,
        })
        return RAGResponse(stream=stream, sources=sources, search_query=query)

    def suggest_questions(self, max_chunks: int = 6) -> list[str]:
        hits = self.kb.search("overview important rules deadlines contacts", k=max_chunks)
        if not hits:
            return []
        excerpts = "\n\n".join(d.page_content[:600] for d, _ in hits)
        raw = (SUGGEST_PROMPT | self.llm | self.parser).invoke({"excerpts": excerpts})
        lines = [re.sub(r"^[\s\-\d.*)]+", "", l).strip() for l in raw.splitlines()]
        return [l for l in lines if 8 < len(l) < 90][:4]


def cited_numbers(answer: str) -> set[int]:
    """Which source numbers the answer actually cites, e.g. {1, 3}."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
