# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system where you upload a document and ask questions. It answers using the relevant parts of the document instead of guessing.

## How it works
1. The uploaded document is split into chunks.
2. Each chunk is converted into an embedding (a numerical representation of its meaning).
3. The question is also converted into an embedding, and the most similar chunks are retrieved.
4. [The retrieved chunks and the question are used to generate an answer grounded in the document.]

## Tech stack
- Language: [Python]
- Embeddings: [library or model, e.g. sentence-transformers]
- Vector store: [FAISS / Chroma / Pinecone, or in-memory]
- Answer generation: [model used, local or API, or "retrieval only"]
- Interface: [command line / Streamlit / Flask]

## How to run
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. [If an API key is needed: create a `.env` file with `YOUR_KEY=...`]
4. Run: `[python app.py]`

## Limitations and future work
- If the wrong chunk is retrieved, the answer can be wrong.
- Tables and images inside documents are not handled well.
- Planned: re-ranking results, showing source citations and evaluating answer quality.
