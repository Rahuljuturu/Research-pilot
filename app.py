"""ResearchPilot V3: Local hybrid PDF retrieval, no LLM or API keys."""
import csv
import io
import streamlit as st
from search_core import SessionIndex, MODEL_NAME, file_signature

st.set_page_config(page_title="ResearchPilot V3 | Hybrid Search", page_icon="🔎", layout="wide")
st.markdown('''<style>
.stApp {background:#0b1220;color:#e7eef9}
[data-testid="stSidebar"] {background:#101b30}
[data-testid="stMetric"] {background:#182841;border:1px solid #2c4264;border-radius:12px;padding:12px}
.hero {background:linear-gradient(115deg,#183353,#29234c);border-radius:18px;border:1px solid #344869;padding:24px 28px;margin-bottom:22px}
.hero h1 {color:white;font-size:2rem;margin:0}
.hero p {color:#c4d6eb;margin:8px 0 0}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading local embedding model (first run may download weights)…")
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")

if "idx" not in st.session_state:
    st.session_state.idx = None
if "results" not in st.session_state:
    st.session_state.results = None

st.markdown('<div class="hero"><h1>🔎 ResearchPilot V3</h1><p>Hybrid PDF Search · BM25 + Vector Embeddings + Reciprocal Rank Fusion · No API key</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("1 · Your documents")
    uploads = st.file_uploader("Upload searchable PDFs", type=["pdf"], accept_multiple_files=True)
    selected = [(f.name, f.getvalue()) for f in (uploads or [])]
    signature = file_signature(selected)
    if st.button("Build / replace index", type="primary", use_container_width=True, disabled=not selected):
        st.session_state.idx = None  # A failure cannot leave old documents searchable.
        st.session_state.results = None
        try:
            with st.spinner("Extracting pages and embedding passages…"):
                idx = SessionIndex(get_model())
                idx.ingest(selected)
                st.session_state.idx = idx
            st.rerun()
        except Exception as exc:
            st.error(f"Index build failed: {exc}")
    if st.button("Clear index", use_container_width=True):
        st.session_state.idx = None
        st.session_state.results = None
        st.rerun()
    st.divider()
    st.header("2 · Retrieval settings")
    mode = st.radio("Search method", ["Hybrid", "Semantic", "BM25"], horizontal=False,
                    help="Hybrid combines vector and exact-term rankings with RRF")
    top_k = st.slider("Top results", 1, 15, 5)
    st.caption("All PDFs and embeddings remain in this running app's memory; no persistent Chroma directory is used. The first model download needs internet access.")

idx = st.session_state.idx
ready = idx is not None and idx.signature == signature
c1, c2, c3 = st.columns(3)
c1.metric("Indexed PDFs", len(idx.files) if ready else 0)
c2.metric("Passages", idx.count() if ready else 0)
c3.metric("Index", "Ready" if ready else "Not ready")
if idx is not None and not ready:
    st.warning("Your selected PDFs changed. Build / replace the index before searching. Old results are disabled.")
if not selected:
    st.info("Upload PDFs using the sidebar, then select Build / replace index.")
if ready:
    st.caption("Active documents: " + ", ".join(idx.files))
    for warning in idx.skipped:
        st.warning(warning)

search_tab, eval_tab = st.tabs(["🔍 Explore results", "📊 Retrieval evaluation"])
with search_tab:
    with st.form("search"):
        question = st.text_input("Research question", placeholder="How does multi-head attention work?")
        scope = st.selectbox("Restrict to", ["All PDFs"] + (idx.files if ready else []))
        submitted = st.form_submit_button("Search", type="primary", disabled=not ready)
    if submitted:
        st.session_state.results = (signature, question, mode, scope, top_k,
            idx.ranked_search(question, mode=mode, k=top_k,
                              filename=None if scope == "All PDFs" else scope))
    result = st.session_state.results
    if result and ready and result[:1] == (signature,):
        if result[2:5] != (mode, scope, top_k):
            st.info("Search settings changed. Press Search again to refresh the results.")
        else:
            hits = result[5]
            st.subheader(f"{len(hits)} matching passages")
            if not hits:
                st.warning("No lexical matches. Try Semantic or Hybrid search, or a different query.")
            for pos, hit in enumerate(hits, 1):
                with st.container(border=True):
                    st.markdown(f"**{pos}. {hit['filename']} — page {hit['page']}**")
                    metrics = []
                    if hit['distance'] is not None:
                        metrics.append(f"Cosine distance: {hit['distance']:.4f}")
                    if hit['bm25'] is not None:
                        metrics.append(f"BM25: {hit['bm25']:.3f}")
                    if hit['rrf'] is not None:
                        metrics.append(f"RRF: {hit['rrf']:.5f}")
                    st.caption(" · ".join(metrics) + " · Scores are NOT accuracy percentages")
                    st.write(hit["text"])
    st.caption("Retrieval only: these are original PDF excerpts, not LLM-generated answers.")

with eval_tab:
    st.write("Evaluate retrieval using known questions and source pages. Upload a CSV with `question`, `expected_file`, and optional `expected_page` columns.")
    template = "question,expected_file,expected_page\nHow does multi-head attention work?,1706.03762v7.pdf,\nWhat are RAG-Token and RAG-Sequence?,2005.11401v4.pdf,\n"
    st.download_button("Download test CSV template", template, file_name="retrieval_tests.csv", mime="text/csv")
    evaluation_file = st.file_uploader("Upload evaluation CSV", type=["csv"], key="eval")
    if st.button("Run evaluation", disabled=not ready or evaluation_file is None):
        try:
            content = evaluation_file.getvalue().decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(content)))
            if not rows or not {"question", "expected_file"}.issubset(rows[0]):
                raise ValueError("CSV must include question and expected_file columns")
            detail = []
            for row in rows:
                q = (row.get("question") or "").strip()
                filename = (row.get("expected_file") or "").strip()
                page = (row.get("expected_page") or "").strip()
                if not q or not filename:
                    continue
                hits = idx.ranked_search(q, mode=mode, k=top_k)
                relevant = [i for i, hit in enumerate(hits, 1) if hit['filename'] == filename and
                            (not page or str(hit['page']) == page)]
                rank = min(relevant) if relevant else None
                detail.append({"question": q, "expected_file": filename, "expected_page": page,
                               "first_relevant_rank": rank, "hit_at_k": int(rank is not None),
                               "reciprocal_rank": 1 / rank if rank else 0.0})
            if not detail:
                raise ValueError("No valid evaluation rows")
            recall = sum(x["hit_at_k"] for x in detail) / len(detail)
            mrr = sum(x["reciprocal_rank"] for x in detail) / len(detail)
            a, b = st.columns(2)
            a.metric(f"Hit Rate@{top_k}", f"{recall:.1%}")
            b.metric(f"MRR@{top_k}", f"{mrr:.3f}")
            st.dataframe(detail, use_container_width=True)
            st.caption("Important: file-level matches are weaker than passage-level relevance. Fill expected_page for a stricter test. This reports Hit Rate@k, not strict passage Recall@k.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
