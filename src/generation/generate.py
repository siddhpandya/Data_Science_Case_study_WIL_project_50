"""Generation module for SettleIN pipeline.

Provides RAG prompt construction and LLM generation via Ollama.
Supports prompt variants defined in prompts.py.

Citation format: passages labelled as [S03_002], parsed with \\[S\\d{2}_\\d{3}\\].
Refusal: strict (exact string) and lenient (contains decline phrase).
"""

import re
from pathlib import Path
import yaml

from src.generation.prompts import get_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# The exact refusal string required for refused_strict.
REFUSAL_STRING = "I don't have information about that in my sources."

# Phrases that trigger refused_lenient.
LENIENT_PHRASES = [
    "i don't have information",
    "i don't have specific information",
    "don't have that information",
    "don't have information about that",
    "outside my knowledge",
    "cannot find",
    "not covered in my sources",
    "no relevant passage",
    "not have specific information",
]

# Regex for passage-level citation IDs.
CITATION_REGEX = re.compile(r"\[S\d{2}_\d{3}\]")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Passage formatting ───────────────────────────────────────────────────────

def format_passages_v3(results: list[tuple[dict, float]]) -> str:
    """Format passages with exact IDs for v3 prompt.

    Each passage is labelled [S03_002] so the model can cite it verbatim.
    """
    parts = []
    for passage, score in results:
        pid = passage.get("id", "unknown")       # e.g. S03_002
        heading = passage.get("heading", "")
        text = passage.get("contents", "")
        header = f"[{pid}]"
        if heading:
            header += f" {heading}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


def format_passages_legacy(results: list[tuple[dict, float]]) -> str:
    """Format passages for v1/v2 and walert prompts (numbered, source-level)."""
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


# ── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    results: list[tuple[dict, float]] | None = None,
    variant: str = "settlein_v3",
) -> tuple[str, str]:
    """Build system + user prompt for LLM generation.

    Returns: (system_prompt, user_prompt)
    """
    prompt = get_prompt(variant)

    if variant == "closed_book" or results is None:
        return prompt["system"], prompt["user"].format(question=question)

    # Use v3/v4 formatter for v3/v4 prompts, legacy for everything else.
    if variant in ("settlein_v3", "settlein_v4", "settlein"):
        passages_text = format_passages_v3(results)
    else:
        passages_text = format_passages_legacy(results)

    return prompt["system"], prompt["user"].format(
        passages=passages_text, question=question
    )


# ── Citation parsing ─────────────────────────────────────────────────────────

def parse_citations(
    answer: str,
    retrieved_ids: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Parse cited passage IDs from the answer.

    Primary: regex \\[S\\d{2}_\\d{3}\\] → list of IDs.
    Fallback: if none found, map "Passage N" to the Nth retrieved ID.

    Returns: (cited_ids, citation_mapped)
        citation_mapped is True only when the fallback was used.
    """
    # Primary: exact passage IDs
    matches = CITATION_REGEX.findall(answer)
    cited = [m.strip("[]") for m in matches]

    if cited:
        return list(dict.fromkeys(cited)), False  # deduplicate, preserve order

    # Fallback: "Passage N" mapping
    if retrieved_ids:
        passage_refs = re.findall(r"Passage\s+(\d+)", answer)
        mapped = []
        for ref in passage_refs:
            idx = int(ref) - 1
            if 0 <= idx < len(retrieved_ids):
                mapped.append(retrieved_ids[idx])
        if mapped:
            return list(dict.fromkeys(mapped)), True

    return [], False


def validate_citations(
    cited_ids: list[str],
    retrieved_ids: list[str],
    collection_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Check cited IDs against retrieved and collection.

    Returns: (cited_not_retrieved, cited_nonexistent)
        cited_not_retrieved: IDs cited but not among retrieved passages.
        cited_nonexistent: IDs cited but not in the collection at all.
    """
    retrieved_set = set(retrieved_ids)
    not_retrieved = [c for c in cited_ids if c not in retrieved_set]

    nonexistent = []
    if collection_ids is not None:
        nonexistent = [c for c in cited_ids if c not in collection_ids]

    return not_retrieved, nonexistent


# ── Refusal detection ────────────────────────────────────────────────────────

def detect_refusal(answer: str) -> tuple[bool, bool]:
    """Detect refusal in the answer.

    Returns: (refused_strict, refused_lenient)
        refused_strict: entire answer equals the exact refusal string (after strip).
        refused_lenient: answer contains any decline phrase.
    """
    trimmed = answer.strip()
    refused_strict = trimmed == REFUSAL_STRING
    refused_lenient = any(phrase in trimmed.lower() for phrase in LENIENT_PHRASES)
    return refused_strict, refused_lenient


# ── Generation ───────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    results: list[tuple[dict, float]] | None = None,
    variant: str = "settlein_v4",
    config: dict | None = None,
) -> dict:
    """Generate an answer using Ollama.

    Returns a dict with keys:
        answer: str
        prompt_eval_count: int (number of prompt tokens evaluated)
        eval_count: int (number of generated tokens)
    """
    if config is None:
        config = load_config()

    gen_cfg = config.get("generation", {})
    model = gen_cfg.get("model", "llama3.2:3b")
    temperature = gen_cfg.get("temperature", 0.0)
    num_predict = gen_cfg.get("num_predict", 256)
    num_ctx = gen_cfg.get("num_ctx", 4096)
    seed = gen_cfg.get("seed", 42)

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
                "num_ctx": num_ctx,
                "seed": seed,
            },
        )
        return {
            "answer": response["message"]["content"],
            "prompt_eval_count": response.get("prompt_eval_count", 0),
            "eval_count": response.get("eval_count", 0),
        }
    except Exception as e:
        return {
            "answer": f"⚠️ Generation failed: {e}\n\nPlease ensure Ollama is running with: ollama serve\nAnd the model is pulled: ollama pull {model}",
            "prompt_eval_count": 0,
            "eval_count": 0,
        }
