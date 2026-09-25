"""Generation module for SettleIN pipeline.

Provides RAG prompt construction and LLM generation via Ollama.
Supports three prompt variants: settlein, walert, closed_book.
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Prompt Templates ─────────────────────────────────────────────────────────

SETTLEIN_SYSTEM = """You are SettleIN, a helpful assistant for international students settling in Australia.
You answer questions about visas, workplace rights, Medicare, transport, and student life.

RULES:
1. Answer ONLY based on the provided source passages below.
2. If the passages don't contain enough information, say "I don't have information about that in my sources."
3. Be concise and direct. Cite which source the information comes from.
4. If the question is about something outside your knowledge base, politely decline.
5. Never fabricate information."""

SETTLEIN_USER_TEMPLATE = """Here are the relevant source passages:

{passages}

Question: {question}

Answer based ONLY on the passages above:"""


WALERT_SYSTEM = """You are an information retrieval assistant. Answer the user's question based strictly on the provided passages. If the answer cannot be found in the passages, state that clearly."""

WALERT_USER_TEMPLATE = """Passages:
{passages}

Question: {question}

Answer:"""


CLOSED_BOOK_SYSTEM = """You are a helpful assistant for international students in Australia. Answer the following question to the best of your knowledge."""

CLOSED_BOOK_USER_TEMPLATE = """Question: {question}

Answer:"""


def format_passages(results: list[tuple[dict, float]]) -> str:
    """Format retrieved passages for prompt insertion."""
    parts = []
    for i, (passage, score) in enumerate(results, 1):
        source = passage.get("source_id", "unknown")
        heading = passage.get("heading", "")
        title = passage.get("title", "")
        text = passage.get("contents", "")
        label = f"[Source {source}: {title}]" if title else f"[Source {source}]"
        if heading:
            label += f" — {heading}"
        parts.append(f"Passage {i} {label}:\n{text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(
    question: str,
    results: list[tuple[dict, float]] | None = None,
    variant: str = "settlein",
) -> tuple[str, str]:
    """Build system + user prompt for LLM generation.

    Returns: (system_prompt, user_prompt)
    """
    if variant == "closed_book" or results is None:
        return CLOSED_BOOK_SYSTEM, CLOSED_BOOK_USER_TEMPLATE.format(question=question)

    passages_text = format_passages(results)

    if variant == "walert":
        return WALERT_SYSTEM, WALERT_USER_TEMPLATE.format(
            passages=passages_text, question=question
        )

    # Default: settlein
    return SETTLEIN_SYSTEM, SETTLEIN_USER_TEMPLATE.format(
        passages=passages_text, question=question
    )


def generate_answer(
    question: str,
    results: list[tuple[dict, float]] | None = None,
    variant: str = "settlein",
    config: dict | None = None,
) -> str:
    """Generate an answer using Ollama."""
    if config is None:
        config = load_config()

    gen_cfg = config.get("generation", {})
    model = gen_cfg.get("model", "llama3.2:3b")
    temperature = gen_cfg.get("temperature", 0.0)
    num_predict = gen_cfg.get("num_predict", 256)

    system_prompt, user_prompt = build_prompt(question, results, variant)

    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": num_predict,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Generation failed: {e}\n\nPlease ensure Ollama is running with: ollama serve\nAnd the model is pulled: ollama pull {model}"
