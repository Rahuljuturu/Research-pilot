"""Lightweight BM25 and reciprocal rank fusion; no cloud services."""
from collections import Counter
from math import log
import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.tokens = [tokenize(d) for d in documents]
        self.freqs = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avglen = sum(self.lengths) / max(len(self.lengths), 1)
        self.n = len(self.tokens)
        self.df = Counter()
        for freq in self.freqs:
            self.df.update(freq.keys())
        self.k1, self.b = k1, b

    def scores(self, query):
        result = [0.0] * self.n
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, freq in enumerate(self.freqs):
                tf = freq.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avglen, 1))
                    result[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return result


def reciprocal_rank_fusion(lists, k=60):
    """Each ranked list contains IDs; ties are broken deterministically by ID."""
    scores = {}
    for ids in lists:
        for position, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + position)
    return sorted(scores, key=lambda x: (-scores[x], x)), scores
