from __future__ import annotations
import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[a-zA-Z]+")

class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.n_docs = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(str(text).lower())

    def fit(self, texts: Iterable[str]) -> "BM25Scorer":
        docs = [self.tokenize(t) for t in texts]
        self.n_docs = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n_docs, 1)
        df = Counter()
        for doc in docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return self

    def score_tokens(self, doc_tokens: list[str], query_tokens: list[str]) -> float:
        if not doc_tokens or not query_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, math.log(1 + (self.n_docs + 0.5) / 0.5))
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
        return float(score)

    def phrase_bonus(self, text_lower: str, phrase: str) -> float:
        phrase = phrase.lower().strip()
        if len(phrase.split()) <= 1:
            return 0.0
        return 1.0 if phrase in text_lower else 0.0

    def score_phrase(self, text: str, phrase: str) -> float:
        tokens = self.tokenize(text)
        q_tokens = self.tokenize(phrase)
        base = self.score_tokens(tokens, q_tokens)
        return base + self.phrase_bonus(str(text).lower(), phrase)
