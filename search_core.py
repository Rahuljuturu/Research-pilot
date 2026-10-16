"""Session-scoped, in-memory semantic PDF search. Never reads persistent Chroma data."""
import hashlib
import re
from pathlib import Path

import chromadb
import fitz
from hybrid import BM25, reciprocal_rank_fusion

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    if not (0 <= overlap < size):
        raise ValueError("Chunk overlap must be smaller than chunk size")
    clean = " ".join(text.split())
    if not clean:
        return []
    result = []
    step = size - overlap
    for start in range(0, len(clean), step):
        part = clean[start:start + size]
        if len(part) >= 40:
            result.append(part)
        if start + size >= len(clean):
            break
    return result


def file_signature(files) -> tuple:
    """The currently selected file set; sensitive to changed bytes and filenames."""
    return tuple(sorted((name, hashlib.sha256(data).hexdigest()) for name, data in files))


class SessionIndex:
    def __init__(self, model):
        self.model = model
        # A NEW client per instance, not a process-global cached client.
        self.client = chromadb.EphemeralClient()
        # Unique collection to prevent cross-session mixing when Chroma shares a backend.
        self.name = "session_" + __import__("uuid").uuid4().hex
        self.collection = self.client.create_collection(name=self.name, metadata={"hnsw:space": "cosine"})
        self.files = []
        self.skipped = []
        self.signature = ()
        self.records = []
        self.bm25 = None

    def ingest(self, files: list[tuple[str, bytes]]) -> None:
        """Index the complete current selection on a fresh instance; deduplicate identical PDFs."""
        if not files:
            raise ValueError("Select at least one PDF")
        self.signature = file_signature(files)
        seen = set()
        records = []
        for filename, payload in files:
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen:
                self.skipped.append(f"{filename}: duplicate content")
                continue
            seen.add(digest)
            try:
                with fitz.open(stream=payload, filetype="pdf") as doc:
                    parts_count = 0
                    for page_number, page in enumerate(doc, 1):
                        for part_idx, part in enumerate(chunk_text(page.get_text("text"))):
                            parts_count += 1
                            records.append((f"{digest[:20]}-{page_number}-{part_idx}", part, {
                                "filename": Path(filename).name,
                                "page": page_number,
                                "sha256": digest,
                            }))
                if parts_count:
                    self.files.append(filename)
                else:
                    self.skipped.append(f"{filename}: no extractable text (scanned PDF?)")
            except Exception as exc:
                self.skipped.append(f"{filename}: invalid/unreadable PDF ({type(exc).__name__})")
        if not records:
            raise ValueError("No searchable text in the selected PDFs. Use text-based PDFs rather than scanned images.")
        self.records = [{"id": x[0], "text": x[1], **x[2]} for x in records]
        self.bm25 = BM25([x["text"] for x in self.records])
        for pos in range(0, len(records), 40):
            batch = records[pos:pos + 40]
            vectors = self.model.encode([x[1] for x in batch], normalize_embeddings=True)
            self.collection.add(
                ids=[x[0] for x in batch],
                documents=[x[1] for x in batch],
                metadatas=[x[2] for x in batch],
                embeddings=vectors.tolist(),
            )

    def count(self) -> int:
        return self.collection.count()

    def search(self, question: str, k: int = 5, filename: str | None = None) -> list[dict]:
        question = question.strip()
        if not question or not self.count():
            return []
        where = {"filename": filename} if filename else None
        count = self.count()
        if filename:
            count = len(self.collection.get(where=where, include=[])["ids"])
        if not count:
            return []
        query = self.model.encode([question], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=query,
            n_results=min(k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [{
            "filename": meta["filename"],
            "page": meta["page"],
            "text": text,
            "distance": float(dist),
        } for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]

    def ranked_search(self, question: str, mode="Hybrid", k=5, filename=None) -> list[dict]:
        """Rank only the current-session documents with dense, lexical, or fused retrieval."""
        question = question.strip()
        if not question or not self.records:
            return []
        candidates = [r for r in self.records if filename is None or r["filename"] == filename]
        if not candidates:
            return []
        ids = {r["id"] for r in candidates}
        by_id = {r["id"]: r for r in candidates}
        # Request all candidates for exact RRF ranking, not just a small top-k subset.
        dense_ids = []
        distances = {}
        if mode in ("Semantic", "Hybrid"):
            where = {"filename": filename} if filename else None
            result = self.collection.query(
                query_embeddings=self.model.encode([question], normalize_embeddings=True).tolist(),
                n_results=len(candidates), where=where,
                include=["distances"],
            )
            dense_ids = [rid for rid in result["ids"][0] if rid in ids]
            distances = dict(zip(result["ids"][0], result["distances"][0]))
        lexical_ids = []
        lexical_scores = {}
        if mode in ("BM25", "Hybrid"):
            score_vector = self.bm25.scores(question)
            lexical_scores = {r["id"]: score_vector[i] for i, r in enumerate(self.records) if r["id"] in ids}
            # No token overlap: BM25 must not return unrelated passages as relevant.
            lexical_ids = sorted((rid for rid in ids if lexical_scores[rid] > 0),
                                 key=lambda rid: (-lexical_scores[rid], rid))
        if mode == "Semantic":
            ranked, fused = dense_ids, {}
        elif mode == "BM25":
            ranked, fused = lexical_ids, {}
        elif mode == "Hybrid":
            ranked, fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
        else:
            raise ValueError("Unknown search mode")
        output = []
        for rid in ranked[:k]:
            row = dict(by_id[rid])
            row["distance"] = float(distances[rid]) if rid in distances else None
            row["bm25"] = lexical_scores.get(rid)
            row["rrf"] = fused.get(rid)
            output.append(row)
        return output
