"""Retrieval module for SettleIN pipeline.

Provides BM25 (sparse) and dense (BAAI/bge-small-en-v1.5) retrieval
over the passage collection. Used by both the generation pipeline and
the Streamlit chatbot.
"""

from pathlib import Path
import json
import re
import math
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
COLLECTION_PATH = REPO_ROOT / "data" / "collection.jsonl"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_collection() -> list[dict]:
    """Load all passages from collection.jsonl."""
    passages = []
    with open(COLLECTION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


# ── BM25 Implementation ─────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer with basic cleaning."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


class BM25Index:
    """Pure-Python BM25 index over passage contents."""

    def __init__(self, passages: list[dict], k1: float = 1.2, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b

        # Build index
        self.doc_tokens = [_tokenize(p["contents"]) for p in passages]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0
        self.n_docs = len(passages)

        # Build inverted index: term -> list of (doc_idx, term_freq)
        self.inverted: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, tokens in enumerate(self.doc_tokens):
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for term, freq in tf.items():
                if term not in self.inverted:
                    self.inverted[term] = []
                self.inverted[term].append((doc_idx, freq))

        # IDF for each term
        self.idf: dict[str, float] = {}
        for term, postings in self.inverted.items():
            df = len(postings)
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Return top_k passages with BM25 scores."""
        query_tokens = _tokenize(query)
        scores = [0.0] * self.n_docs

        for token in query_tokens:
            if token not in self.inverted:
                continue
            idf = self.idf[token]
            for doc_idx, tf in self.inverted[token]:
                dl = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * numerator / denominator

        # Get top_k
        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.passages[i], scores[i]) for i in ranked if scores[i] > 0]


# ── Dense Retrieval ──────────────────────────────────────────────────────────

class DenseIndex:
    """Dense retrieval using sentence-transformers."""

    def __init__(self, passages: list[dict], model_name: str = "BAAI/bge-small-en-v1.5"):
        self.passages = passages
        self.model_name = model_name
        self._model = None
        self._embeddings = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def _build_embeddings(self):
        if self._embeddings is not None:
            return

        # Try loading cached embeddings
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = EMBEDDINGS_DIR / "passage_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "passage_ids.json"

        if cache_path.exists() and ids_path.exists():
            with open(ids_path, "r") as f:
                cached_ids = json.load(f)
            current_ids = [p["id"] for p in self.passages]
            if cached_ids == current_ids:
                self._embeddings = np.load(str(cache_path))
                print(f"  Loaded cached embeddings: {self._embeddings.shape}")
                return

        # Build fresh embeddings
        self._load_model()
        texts = [p["contents"] for p in self.passages]
        print(f"  Encoding {len(texts)} passages with {self.model_name}...")
        self._embeddings = self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        # Cache
        np.save(str(cache_path), self._embeddings)
        with open(ids_path, "w") as f:
            json.dump([p["id"] for p in self.passages], f)
        print(f"  Embeddings cached to {cache_path}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Return top_k passages by cosine similarity."""
        self._build_embeddings()
        self._load_model()

        query_emb = self._model.encode([query], normalize_embeddings=True)
        scores = np.dot(self._embeddings, query_emb.T).flatten()

        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.passages[i], float(scores[i])) for i in ranked if scores[i] > 0]


# ── Unified Search Interface ────────────────────────────────────────────────

class SearchEngine:
    """Unified search engine supporting BM25 and dense retrieval."""

    def __init__(self, passages: list[dict] | None = None):
        if passages is None:
            passages = load_collection()
        self.passages = passages

        cfg = load_config()
        bm25_cfg = cfg.get("retrieval", {}).get("bm25", {})
        dense_cfg = cfg.get("retrieval", {}).get("dense", {})

        self.bm25 = BM25Index(
            passages,
            k1=bm25_cfg.get("k1", 1.2),
            b=bm25_cfg.get("b", 0.75),
        )

        self.dense = DenseIndex(
            passages,
            model_name=dense_cfg.get("model", "BAAI/bge-small-en-v1.5"),
        )

    def search(self, query: str, method: str = "bm25", top_k: int = 5) -> list[tuple[dict, float]]:
        """Search using specified method."""
        if method == "bm25":
            return self.bm25.search(query, top_k)
        elif method == "dense":
            return self.dense.search(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
