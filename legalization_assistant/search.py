"""Keyword retrieval over the chunk corpus, backed by minsearch."""

import re
from functools import lru_cache

from minsearch import Index

from .config import DEFAULT_BOOSTS, DEFAULT_NUM_RESULTS, MAX_PARTS_PER_ARTICLE, OVERFETCH_FACTOR
from .ingest import load_corpus

TEXT_FIELDS = ["title", "chapter", "section", "text"]
KEYWORD_FIELDS = ["doc_id", "article"]

# Everyday words the statute writes differently: it says "expulsion", never
# "deportation". Matches are appended to the query, not substituted.
STATUTORY_SYNONYMS = {
    "deport": "expulsion",
    "deported": "expulsion",
    "deportation": "expulsion",
    "deportations": "expulsion",
    "foreigner": "alien",
    "foreigners": "aliens",
    "immigrant": "alien",
    "immigrants": "aliens",
    "expat": "alien",
    "expats": "aliens",
    "type": "category",
    "types": "categories",
    "cost": "fee charge",
    "costs": "fees charges",
    "price": "fee charge",
    "fine": "liability",
    "fines": "liability",
    "overstay": "termination of the period of stay",
    "id": "identity card",
    "kids": "children",
    "job": "labour activities work",
    "jobs": "labour activities work",
    "study": "education",
    "studying": "education",
    "marriage": "marital family relations",
    "banned": "prohibition ban",
}


def expand_query(query: str) -> str:
    """Append the statutory equivalents of any everyday terms in the query."""
    additions: list[str] = []
    for token in re.findall(r"[a-z]+", query.lower()):
        synonym = STATUTORY_SYNONYMS.get(token)
        if synonym and synonym not in additions:
            additions.append(synonym)
    return f"{query} {' '.join(additions)}" if additions else query


class LegalSearch:
    """A TF-IDF index over article chunks, with filtering by source document."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
        self.index.fit(documents)

    def search(
        self,
        query: str,
        num_results: int = DEFAULT_NUM_RESULTS,
        boosts: dict[str, float] | None = None,
        doc_id: str | None = None,
        max_per_article: int = MAX_PARTS_PER_ARTICLE,
    ) -> list[dict]:
        if not query.strip():
            return []
        results = self.index.search(
            expand_query(query),
            filter_dict={"doc_id": doc_id} if doc_id else {},
            boost_dict=boosts if boosts is not None else DEFAULT_BOOSTS,
            num_results=num_results * OVERFETCH_FACTOR,
        )

        seen: dict[tuple[str, str], int] = {}
        diversified: list[dict] = []
        for result in results:
            key = (result["doc_id"], result["article"])
            if seen.get(key, 0) >= max_per_article:
                continue
            seen[key] = seen.get(key, 0) + 1
            diversified.append(result)
            if len(diversified) == num_results:
                break
        return diversified

    @property
    def doc_ids(self) -> list[str]:
        return sorted({document["doc_id"] for document in self.documents})


@lru_cache(maxsize=1)
def get_search() -> LegalSearch:
    """The process-wide index."""
    return LegalSearch(load_corpus())
