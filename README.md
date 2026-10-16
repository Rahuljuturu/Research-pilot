# ResearchPilot — Hybrid Document Search Engine

ResearchPilot is a Python-based application that allows users to search multiple PDF documents using natural-language queries. It combines semantic search and keyword-based search to retrieve relevant information from research papers and other documents.

The application runs locally without requiring an external API key.

## Features

- Upload and search multiple PDF documents.
- Semantic search using Sentence Transformers.
- Keyword-based search using BM25.
- Hybrid search using Reciprocal Rank Fusion (RRF).
- Display relevant passages with PDF names and page numbers.
- Compare different retrieval methods.
- Evaluate search performance using Hit Rate@K and MRR.

## Technologies Used

- **Python** — Core application development
- **Streamlit** — Web interface
- **Sentence Transformers** — Text embeddings
- **ChromaDB** — Vector database
- **BM25** — Keyword-based retrieval
- **PyMuPDF** — PDF text extraction

## How It Works

1. Users upload one or more PDF documents.
2. The application extracts text and divides it into smaller passages.
3. Text embeddings are generated for semantic search.
4. BM25 and semantic search retrieve relevant passages.
5. Hybrid search combines both methods to improve ranking.
6. Results are displayed with source document names and page numbers.

## Installation

Clone the repository:

```bash
git clone https://github.com/Rahuljuturu/Research-pilot.git
cd Research-pilot
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Example Queries

- What is the Transformer architecture?
- How does multi-head attention work?
- What is Retrieval-Augmented Generation?
- Why do language models struggle with long contexts?

## Future Improvements

- Integrate a local LLM for question answering.
- Add AI-generated summaries with citations.
- Improve retrieval using advanced reranking techniques.

## Author

**Rahul Juturu**  
Master of Information Technology — University of Technology Sydney (UTS)
