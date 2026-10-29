
from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

from src import llm as llm_mod
from src import vectorstore as vs_mod
from src.config import DEFAULT_MODELS, EMBEDDING_MODELS, ROOT_DIR, Settings
from src.loaders import SUPPORTED_EXTENSIONS, file_hash, load_bytes, split_documents
from src.rag import CampusRAG, cited_numbers

st.set_page_config(page_title="Campus Assistant", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

SAMPLE_DIR = ROOT_DIR / "sample_docs"

# ───────────────────────────────────────────────────────────── styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root{
  --ink:#1B2340; --muted:#69708A; --line:#DFE4EE; --paper:#FFFFFF; --bg:#F4F6FA;
  --green:#1F6F5C; --green-soft:#E3F1EC; --marker:#FFE27A; --marker-soft:#FFF6CF;
}
html, body, [class*="css"], .stApp, button, input, textarea, select { font-family:'Manrope',sans-serif; }
.stApp{ background:var(--bg); color:var(--ink); }
header[data-testid="stHeader"]{ background:transparent; }
[data-testid="stToolbar"]{ visibility:hidden; }
.block-container{ max-width:920px; padding-top:2.2rem; padding-bottom:7rem; }

/* sidebar */
[data-testid="stSidebar"]{ background:#1B2340; }
[data-testid="stSidebar"] *{ color:#E8EBF5; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small{ color:#A9B0C8 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.05); border:1.5px dashed rgba(255,255,255,.25); border-radius:14px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{
  background:#fff; color:var(--ink) !important; border:none; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *{ color:var(--ink) !important; }
[data-testid="stSidebar"] .stButton>button{
  width:100%; border-radius:10px; border:1px solid rgba(255,255,255,.18);
  background:rgba(255,255,255,.06); font-weight:600; }
[data-testid="stSidebar"] .stButton>button:hover{ border-color:var(--marker); color:var(--marker); }
[data-testid="stSidebar"] .stButton>button[kind="primary"]{ background:var(--marker); border:none; }
[data-testid="stSidebar"] .stButton>button[kind="primary"] *{ color:var(--ink) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details{ border-color:rgba(255,255,255,.15); border-radius:12px; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:rgba(255,255,255,.08) !important; color:#fff !important; }
[data-testid="stSidebar"] .stButton>button[kind="tertiary"]{ background:transparent; border:none; width:auto; }
[data-testid="stSidebar"] .stButton>button[kind="tertiary"] *{ color:#A9B0C8 !important; font-size:1rem; }
[data-testid="stSidebar"] .stButton>button[kind="tertiary"]:hover *{ color:#FF9C9C !important; }
.brand{ display:flex; gap:.7rem; align-items:center; margin:.2rem 0 1.4rem; }
.brand-mark{ width:40px; height:40px; border-radius:11px; background:var(--marker);
  display:grid; place-items:center; font-size:21px; }
.brand-name{ font-weight:800; font-size:1.12rem; letter-spacing:-.01em; line-height:1.1; }
.brand-sub{ font-size:.78rem; color:#A9B0C8; }
.side-h{ font-weight:700; font-size:.95rem; margin:1.3rem 0 .5rem; }
.doc-row{ padding:.55rem .7rem; border-radius:10px; background:rgba(255,255,255,.06); margin-bottom:.15rem; }
.doc-name{ font-weight:600; font-size:.86rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.doc-meta{ font-size:.74rem; color:#A9B0C8; }

/* hero */
.hero h1{ font-size:2.35rem; font-weight:800; letter-spacing:-.035em; margin:0 0 .35rem; color:var(--ink); }
.hero p{ color:var(--muted); font-size:1.02rem; margin:0; max-width:620px; }
.status{ display:flex; gap:.5rem; flex-wrap:wrap; margin:1rem 0 1.6rem; }
.pill{ font-size:.78rem; font-weight:600; padding:.32rem .7rem; border-radius:999px;
  background:var(--paper); border:1px solid var(--line); color:var(--muted); }
.pill b{ color:var(--ink); }
.pill.ok{ background:var(--green-soft); border-color:transparent; color:var(--green); }
.pill.warn{ background:#FDECEC; border-color:transparent; color:#A33A3A; }

/* empty state */
.empty{ background:var(--paper); border:1px solid var(--line); border-radius:20px; padding:2rem 2.2rem; }
.empty h3{ margin:0 0 .4rem; font-weight:800; letter-spacing:-.02em; }
.empty p{ color:var(--muted); margin:0 0 .3rem; }
.empty ol{ color:var(--ink); padding-left:1.2rem; margin:.8rem 0 0; }
.empty li{ margin:.25rem 0; }
.suggest-h{ font-weight:700; margin:.4rem 0 .6rem; color:var(--ink); }
[data-testid="stMain"] div[data-testid="stHorizontalBlock"] .stButton>button{
  width:100%; text-align:left; justify-content:flex-start; background:var(--paper);
  border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem; min-height:3.4rem;
  font-weight:600; color:var(--ink); }
[data-testid="stMain"] div[data-testid="stHorizontalBlock"] .stButton>button:hover{ border-color:var(--green); color:var(--green); }

/* chat */
[data-testid="stChatMessage"]{ background:transparent; padding:.4rem 0; gap:.9rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"]{
  font-weight:600; font-size:1.02rem; }
.answer{ font-family:'Source Serif 4',Georgia,serif; font-size:1.07rem; line-height:1.68; color:var(--ink); }
.answer strong{ font-weight:600; background:linear-gradient(transparent 62%, var(--marker-soft) 62%); }
.cite{ display:inline-block; font-family:'Manrope',sans-serif; font-size:.68rem; font-weight:800;
  min-width:1.25rem; text-align:center; padding:.05rem .32rem; margin:0 .08rem; border-radius:6px;
  background:var(--marker); color:var(--ink); vertical-align:.18em; line-height:1.3; }
.searched{ font-size:.78rem; color:var(--muted); margin:.2rem 0 .2rem; }
.src{ background:var(--paper); border:1px solid var(--line); border-radius:14px;
  padding:.85rem 1rem; margin:.5rem 0; }
.src.unused{ opacity:.6; }
.src-top{ display:flex; align-items:center; gap:.6rem; }
.src-num{ font-weight:800; font-size:.78rem; background:var(--marker); border-radius:7px; padding:.12rem .45rem; }
.src.unused .src-num{ background:var(--line); }
.src-name{ font-weight:700; font-size:.88rem; flex:1; word-break:break-all; }
.src-score{ font-size:.74rem; color:var(--muted); white-space:nowrap; }
.bar{ height:4px; background:var(--line); border-radius:4px; margin:.55rem 0 .6rem; overflow:hidden; }
.bar span{ display:block; height:100%; background:var(--green); border-radius:4px; }
.src-text{ font-family:'Source Serif 4',Georgia,serif; font-size:.92rem; line-height:1.55; color:#3C4560;
  max-height:9.5rem; overflow:auto; }
[data-testid="stExpander"] details{ border-radius:14px; border-color:var(--line); background:transparent; }
[data-testid="stChatInput"]{ border-radius:16px; }
[data-testid="stChatInput"] textarea{ font-size:1rem; }
@media (max-width:640px){ .hero h1{ font-size:1.8rem; } .empty{ padding:1.4rem; } }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────── state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("settings", Settings())
    ss.setdefault("messages", [])
    ss.setdefault("suggestions", None)
    ss.setdefault("pending_question", None)
    ss.setdefault("uploader_key", 0)


init_state()
S: Settings = st.session_state.settings


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads it once)…")
def cached_embeddings(provider: str):
    return vs_mod.get_embeddings(provider)


@st.cache_resource(show_spinner=False)
def cached_kb(backend: str, embedding_provider: str, index: str):
    return vs_mod.KnowledgeBase(backend, embedding_provider,
                                cached_embeddings(embedding_provider), pinecone_index=index)


def get_kb():
    return cached_kb(S.vector_store, S.embedding_provider, S.pinecone_index)


# ───────────────────────────────────────────────────────────── helpers
def index_upload(kb, data: bytes, name: str) -> str:
    fid = file_hash(data)
    if kb.has_file(fid):
        return f"**{name}** is already in your library."
    pages = load_bytes(data, name)
    if not pages:
        return f"**{name}** has no readable text (scanned PDFs need OCR first)."
    chunks = split_documents(pages, S.chunk_size, S.chunk_overlap)
    n = kb.add_file(fid, name, chunks, pages=len(pages))
    return f"Added **{name}** — {len(pages)} pages, {n} passages."


def ingest(files: list[tuple[str, bytes]]) -> None:
    kb = get_kb()
    with st.status(f"Adding {len(files)} file(s) to your library…", expanded=True) as status:
        for name, data in files:
            st.write(f"Reading and indexing {name}")
            try:
                st.write(index_upload(kb, data, name))
            except Exception as exc:  # show the reason, keep going with other files
                st.write(f"Couldn't add {name}: {exc}")
        status.update(label="Library updated", state="complete", expanded=False)
    st.session_state.suggestions = None


def render_answer(text: str) -> str:
    """Markdown answer → HTML-safe markdown with highlighted citation chips."""
    safe = html.escape(text, quote=False)
    safe = re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', safe)
    return f'<div class="answer">\n\n{safe}\n\n</div>'


def render_sources(sources: list[dict], answer: str, query: str, question: str) -> None:
    if not sources:
        return
    cited = cited_numbers(answer)
    used = sum(1 for s in sources if s["number"] in cited)
    label = f"Sources · {used} cited of {len(sources)} retrieved" if cited else f"Sources · {len(sources)} passages searched"
    with st.expander(label):
        if query and query.strip().lower() != question.strip().lower():
            st.markdown(f'<div class="searched">Searched for: “{html.escape(query)}”</div>',
                        unsafe_allow_html=True)
        for s in sources:
            where = f"page {s['page']}" if s.get("page") else "text"
            pct = int(round(s["score"] * 100))
            body = html.escape(re.sub(r"^#+\s*", "", s["text"], flags=re.M))
            body = re.sub(r"\n\s*\n+", "<br><br>", body).replace("\n", "<br>")
            cls = "src" if (not cited or s["number"] in cited) else "src unused"
            st.markdown(
                f'<div class="{cls}"><div class="src-top"><span class="src-num">{s["number"]}</span>'
                f'<span class="src-name">{html.escape(s["name"])} · {where}</span>'
                f'<span class="src-score">{pct}% match</span></div>'
                f'<div class="bar"><span style="width:{pct}%"></span></div>'
                f'<div class="src-text">{body}</div></div>', unsafe_allow_html=True)


def build_rag():
    ready, msg = llm_mod.provider_ready(S.llm_provider)
    if not ready:
        raise RuntimeError(msg)
    model = llm_mod.get_chat_model(S.llm_provider, S.llm_model, S.temperature)
    return CampusRAG(get_kb(), model, top_k=S.top_k)


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "Connection refused" in text or "ConnectError" in type(exc).__name__ or "11434" in text:
        return ("Couldn't reach Ollama. Start it with `ollama serve` and pull the model "
                f"with `ollama pull {S.llm_model}`, or pick another provider in Settings.")
    if "401" in text or "invalid_api_key" in text.lower() or "Incorrect API key" in text:
        return f"The {S.llm_provider} API key was rejected. Check it in your .env file."
    if "model" in text.lower() and ("not found" in text.lower() or "does not exist" in text.lower()):
        return f"The model “{S.llm_model}” isn't available on {S.llm_provider}. Change it in Settings."
    return text


# ───────────────────────────────────────────────────────────── sidebar
try:
    kb = get_kb()
    kb_error = None
except Exception as exc:
    kb, kb_error = None, str(exc)

with st.sidebar:
    st.markdown("""
<div class="brand"><div class="brand-mark">🎓</div>
<div><div class="brand-name">Campus Assistant</div>
<div class="brand-sub">Answers from your documents</div></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="side-h">Add documents</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "PDF, DOCX, TXT or MD", type=[e.strip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True, key=f"up_{st.session_state.uploader_key}",
        label_visibility="collapsed")
    if uploads and kb and st.button(f"Add {len(uploads)} file(s) to library", type="primary"):
        ingest([(u.name, u.getvalue()) for u in uploads])
        st.session_state.uploader_key += 1
        st.rerun()

    if kb:
        files = kb.files()
        st.markdown(f'<div class="side-h">Library · {len(files)} file(s)</div>', unsafe_allow_html=True)
        if not files:
            st.caption("Nothing here yet. Upload a file or load the sample handbook.")
            if SAMPLE_DIR.exists() and st.button("Load sample handbook"):
                ingest([(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir())
                        if p.suffix.lower() in SUPPORTED_EXTENSIONS])
                st.rerun()
        for f in files:
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.markdown(f"""<div class="doc-row"><div class="doc-name" title="{html.escape(f['name'])}">📄 {html.escape(f['name'])}</div>
<div class="doc-meta">{f['pages']} pages · {f['chunks']} passages · {f['added']}</div></div>""",
                        unsafe_allow_html=True)
            if c2.button("✕", key=f"del_{f['id']}", help=f"Remove {f['name']}", type="tertiary"):
                kb.delete_file(f["id"])
                st.session_state.suggestions = None
                st.rerun()

    st.markdown('<div class="side-h">Conversation</div>', unsafe_allow_html=True)
    if st.button("Start a new chat", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Settings"):
        providers = list(DEFAULT_MODELS)
        new_provider = st.selectbox("Language model provider", providers,
                                    index=providers.index(S.llm_provider))
        if new_provider != S.llm_provider:
            S.llm_provider, S.llm_model = new_provider, DEFAULT_MODELS[new_provider]
            st.rerun()
        S.llm_model = st.text_input("Model name", S.llm_model)
        S.temperature = st.slider("Creativity", 0.0, 1.0, S.temperature, 0.05,
                                  help="Lower keeps answers closer to the documents.")
        S.top_k = st.slider("Passages to retrieve", 2, 10, S.top_k)
        embs = list(EMBEDDING_MODELS)
        S.embedding_provider = st.selectbox(
            "Embeddings", embs, index=embs.index(S.embedding_provider),
            help="Switching embeddings uses a separate library; re-add your files.")
        stores = ["ChromaDB", "Pinecone"]
        S.vector_store = st.selectbox("Vector database", stores, index=stores.index(S.vector_store))
        st.caption("Chunk size and overlap are set in .env (CHUNK_SIZE, CHUNK_OVERLAP).")

# ───────────────────────────────────────────────────────────── main
ready, ready_msg = llm_mod.provider_ready(S.llm_provider)
n_files = len(kb.files()) if kb else 0

st.markdown(f"""
<div class="hero">
  <h1>Ask your campus documents</h1>
  <p>Handbooks, fee structures, exam rules, syllabi, research papers. Every answer
  is grounded in your files and shows the exact passages it came from.</p>
</div>
<div class="status">
  <span class="pill {'ok' if ready else 'warn'}">{'●' if ready else '○'} <b>{S.llm_provider}</b>&nbsp;{html.escape(S.llm_model)}</span>
  <span class="pill"><b>{n_files}</b>&nbsp;document{'s' if n_files != 1 else ''}</span>
  <span class="pill"><b>{S.vector_store}</b></span>
  <span class="pill">top-{S.top_k} retrieval</span>
</div>""", unsafe_allow_html=True)

if kb_error:
    st.error(f"The vector database couldn't start: {kb_error}")
    st.stop()
if not ready:
    st.warning(ready_msg)

if n_files == 0:
    st.markdown("""
<div class="empty">
  <h3>Start by adding a document</h3>
  <p>Upload a PDF, Word file or text file from the sidebar. The assistant reads it,
  splits it into passages and indexes them so you can ask questions in plain language.</p>
  <ol>
    <li>Drop files into <b>Add documents</b> on the left</li>
    <li>Click <b>Add to library</b> and wait a few seconds</li>
    <li>Ask a question below, like “What is the minimum attendance requirement?”</li>
  </ol>
</div>""", unsafe_allow_html=True)
    if SAMPLE_DIR.exists():
        st.write("")
        if st.button("Try it with the sample college handbook", type="primary"):
            ingest([(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir())
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS])
            st.rerun()

# suggested questions
elif not st.session_state.messages and not st.session_state.pending_question and ready:
    if st.session_state.suggestions is None:
        try:
            with st.spinner("Reading your library for question ideas…"):
                st.session_state.suggestions = build_rag().suggest_questions()
        except Exception:
            st.session_state.suggestions = []
    sugg = st.session_state.suggestions or [
        "Summarise the key rules in these documents",
        "What deadlines should I know about?",
        "Who do I contact for help?",
        "What happens if I break a rule?",
    ]
    st.markdown('<div class="suggest-h">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(sugg[:4]):
        if cols[i % 2].button(q, key=f"sugg_{i}", width="stretch"):
            st.session_state.pending_question = q
            st.rerun()

# history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            st.markdown(render_answer(m["content"]), unsafe_allow_html=True)
            render_sources(m.get("sources", []), m["content"], m.get("query", ""), m.get("question", ""))

# input
placeholder_text = "Ask about your documents…" if n_files else "Add a document first, then ask away"
typed = st.chat_input(placeholder_text, disabled=(n_files == 0 or not ready))
question = typed or st.session_state.pop("pending_question", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎓"):
        box = st.empty()
        box.markdown('<div class="searched">Searching your library…</div>', unsafe_allow_html=True)
        try:
            rag = build_rag()
            resp = rag.ask(question, history)
            answer = ""
            for token in resp.stream:
                answer += token
                box.markdown(render_answer(answer + " ▍"), unsafe_allow_html=True)
            if not answer.strip():
                answer = "I couldn't generate an answer. Try rephrasing the question."
            box.markdown(render_answer(answer), unsafe_allow_html=True)
            sources = [s.__dict__ for s in resp.sources]
            render_sources(sources, answer, resp.search_query, question)
            st.session_state.messages.append({"role": "assistant", "content": answer,
                                              "sources": sources, "query": resp.search_query,
                                              "question": question})
        except Exception as exc:
            box.empty()
            st.error(friendly_error(exc))
            st.session_state.messages.pop()  # drop the unanswered question
