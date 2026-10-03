"""Cross-encoder reranking for SettleIN.

A bi-encoder (bge-small) embeds query and passage separately; a cross-encoder
reads the (query, passage) pair together, which is slower but much better at
judging whether a passage actually answers the question. With ~100 passages
we can afford to rerank the top candidates from first-stage retrieval.

The cross-encoder score is also a usable answerability signal: when even the
best candidate scores low, the knowledge base probably does not cover the
question (see the threshold gate in src/experiments/rerank_experiment.py).
"""

import re

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


def passage_text(passage: dict) -> str:
    """Passage text shown to the reranker: {title} > {heading}: {contents}."""
    heading = re.sub(r"\(#[^)]*\)", "", passage.get("heading", "")).strip()
    prefix = " > ".join(x for x in (passage.get("title", ""), heading) if x)
    contents = passage.get("contents", "")
    return f"{prefix}: {contents}" if prefix else contents


class CrossEncoderReranker:
    """Rerank (passage, score) candidates with a sentence-transformers CrossEncoder.

    Scores are passed through a sigmoid so they lie in [0, 1] and can be
    compared against a fixed threshold.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length,
                activation_fn=torch.nn.Sigmoid(),
            )

    def rerank(self, query: str, candidates: list[tuple[dict, float]],
               top_k: int | None = None) -> list[tuple[dict, float]]:
        """Return candidates re-sorted by cross-encoder score (highest first)."""
        if not candidates:
            return []
        self._load_model()
        pairs = [(query, passage_text(p)) for p, _ in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        out = [(p, float(s)) for (p, _), s in ranked]
        return out[:top_k] if top_k else out
