from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .data import ALIAS_TO_ID, BOOKS

@dataclass
class Candidate:
    verse_id: str
    book: str
    chapter: int
    verse: int
    text: str
    score: float
    method: str

_ORDINALS = [(r"\bfirst\b", "1"), (r"\bsecond\b", "2"), (r"\bthird\b", "3")]
def _normalize(text: str) -> str:
    t = text.lower()
    for pat, rep in _ORDINALS:
        t = re.sub(pat, rep, t)
    return t

def _alias_pattern(alias: str) -> str:
    m = re.match(r"^([1-3])(.+)$", alias)
    if m: return re.escape(m.group(1)) + r"\s*" + re.escape(m.group(2))
    return re.escape(alias)

_ALIASES = sorted({a for _, _, al in BOOKS for a in al}, key=len, reverse=True)
_BOOK_RE = re.compile(r"\b(?P<book>" + "|".join(_alias_pattern(a) for a in _ALIASES) + r")\b", re.IGNORECASE)
_SEP = r"(?:\s*[:.]\s*|\s+verse\s+|\s+)"
_TAIL_RE = re.compile(r"\s*(?:chapter\s+)?(?P<ch>\d{1,3})(?:" + _SEP + r"(?P<v>\d{1,3})(?:\s*(?:-|–|—|through|to)\s*(?P<v2>\d{1,3}))?)?", re.IGNORECASE)

class ScriptureDetector:
    def __init__(self, verses: pd.DataFrame, embeddings: Optional[np.ndarray] = None, model=None, threshold: float = 0.55):
        self.verses = verses.reset_index(drop=True)
        self.emb = embeddings.astype(np.float32) if embeddings is not None else None
        self.model = model
        self.threshold = threshold
        self._idx = {(r.book_id, r.chapter, r.verse): i for i, r in enumerate(self.verses.itertuples(index=False))}
        self._chapters = {(r.book_id, r.chapter) for r in self.verses.itertuples(index=False)}

    def parse_references(self, text: str) -> list[dict]:
        t = _normalize(text)
        out = []
        for m in _BOOK_RE.finditer(t):
            alias = re.sub(r"\s+", "", m.group("book"))
            bid = ALIAS_TO_ID.get(alias)
            if not bid: continue
            tail = _TAIL_RE.match(t, m.end())
            if not tail: continue
            ch = int(tail.group("ch"))
            v = int(tail.group("v")) if tail.group("v") else None
            v2 = int(tail.group("v2")) if tail.group("v2") else None
            if v is not None and (bid, ch, v) in self._idx:
                last = v2 if (v2 and (bid, ch, v2) in self._idx) else v
                out.append({"book_id": bid, "chapter": ch, "verses": list(range(v, last + 1))})
            elif (bid, ch) in self._chapters:
                out.append({"book_id": bid, "chapter": ch, "verses": []})
        return out

    def semantic_search(self, text: str, top_k: int = 5) -> list[tuple[int, float]]:
        if self.emb is None or self.model is None: return []
        q = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        scores = self.emb @ q
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def detect(self, text: str, top_k: int = 5, threshold: Optional[float] = None) -> list[Candidate]:
        threshold = self.threshold if threshold is None else threshold
        found = {}
        for ref in self.parse_references(text):
            if ref["verses"]:
                for v in ref["verses"]:
                    i = self._idx[(ref["book_id"], ref["chapter"], v)]
                    row = self.verses.iloc[i]
                    found[row.verse_id] = Candidate(row.verse_id, row.book, row.chapter, row.verse, row.text, 0.99, "reference")
            else:
                row = self.verses[(self.verses.book_id == ref["book_id"]) & (self.verses.chapter == ref["chapter"])].iloc[0]
                found[row.verse_id] = Candidate(row.verse_id, row.book, row.chapter, 0, f"{row.book} {row.chapter} (chapter ref)", 0.85, "reference_chapter")

        for i, s in self.semantic_search(text, top_k=top_k):
            if s < threshold: continue
            row = self.verses.iloc[i]
            if row.verse_id not in found:
                found[row.verse_id] = Candidate(row.verse_id, row.book, row.chapter, row.verse, row.text, s, "semantic")
        return sorted(found.values(), key=lambda c: -c.score)[:top_k]