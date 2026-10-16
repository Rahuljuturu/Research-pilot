# ResearchPilot V3 — Hybrid PDF Search

A **local retrieval system**, not an AI agent or answer-generating RAG chatbot. Upload text-based PDFs, build an ephemeral index and compare **Sentence Transformers + Chroma** semantic retrieval, **BM25** lexical retrieval and **Reciprocal Rank Fusion** hybrid retrieval. Streamlit UI, PDF/page citations and evaluation CSV included. **No paid API or API key.**

## Quick start on Mac (Apple Silicon)

Use Python 3.11 or 3.12.

```bash
cd researchpilot_v3_hybrid
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

First run needs internet once to download Python packages and the `all-MiniLM-L6-v2` embedding model. After downloading, local processing needs no external service. Indexing speed depends on your Mac and number of pages.

## Test in 3 minutes

1. Upload `1706.03762v7.pdf` (Transformer) and `2005.11401v4.pdf` (RAG), if available.
2. Click **Build / replace index**.
3. Ask `What is multi-head attention?` in Semantic, BM25 and Hybrid, observing differences in file, page and ranking.
4. Ask `What are RAG-Sequence and RAG-Token?`, and check the RAG paper is retrieved.
5. Remove a PDF and rebuild. Ensure its passages **never** reappear. Click **Clear index** to reset.
6. Download the evaluation CSV template, populate expected pages from your actual documents, then run evaluations under each search method.

## Technical details

- **PDF parsing**: PyMuPDF, preserves original 1-indexed page number.
- **Chunks**: 900-character windows, 150-character overlap (may cut a sentence).
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`, encoded locally, normalised vectors.
- **Vector search**: Chroma EphemeralClient cosine distance. Lower is closer, **not probability**.
- **Lexical search**: In-repo Python BM25 implementation (`hybrid.py`); no extra BM25 dependency.
- **Hybrid ranking**: reciprocal rank fusion `1 / (60 + rank)` per retrieval list.
- **Scope**: only PDFs selected for the active index. The index is in memory; no old Chroma storage is loaded. Change the upload selection → rebuild before searching.
- **Evaluation**: `Hit Rate@k` and `MRR@k` using expected filename and optionally expected page. This is a weak relevance proxy when no page is specified, **not full information retrieval ground truth**.

## Limitations

- This is *hybrid information retrieval*, not agentic RAG; it does not generate answers or call an LLM.
- Scanned image PDFs require OCR, not included.
- Hybrid fusion uses the rankings even when a vector result is semantically weak; always inspect the actual passages.
- Session-memory retrieval means restarting the Streamlit process will require reindexing.
- Streamlit session state is per browser session; it is not an access-control or data-security boundary for a deployed multi-user server.
- Not designed for sensitive documents or untrusted public deployments.

## Resume (use only once tested)

**ResearchPilot V3 — Hybrid Research Document Search** | Python, Sentence Transformers, ChromaDB, BM25, Streamlit

- Built a multi-PDF information retrieval app integrating local transformer embeddings, BM25 keyword matching and Reciprocal Rank Fusion.
- Added session-scoped document indexing, page-level evidence links and a benchmark view comparing search modes with Hit Rate@k and MRR@k.

Report measured results from your own evaluation set; do not invent performance numbers.
