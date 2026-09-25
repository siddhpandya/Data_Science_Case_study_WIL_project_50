"""SettleIN – RAG Chatbot for International Students in Australia.

Interactive Streamlit chatbot with:
- BM25 and Dense retrieval options
- Evidence panel showing retrieved passages
- Source attribution and confidence scoring
- Responsive chat interface
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.search import SearchEngine, load_collection
from src.generation.generate import generate_answer, load_config, format_passages

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SettleIN – Student Settlement Assistant",
    page_icon="🇦🇺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS (dark theme friendly) ────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        opacity: 0.6;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize State ─────────────────────────────────────────────────────────

@st.cache_resource
def init_search_engine():
    """Load collection and build search indices (cached)."""
    passages = load_collection()
    engine = SearchEngine(passages)
    return engine, passages


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "evidence" not in st.session_state:
        st.session_state.evidence = []


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(passages):
    with st.sidebar:
        st.header("⚙️ Settings")

        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["dense", "bm25"],
            index=0,
            help="Dense = semantic similarity (recommended), BM25 = keyword matching",
        )

        st.caption("📐 Passage count is **adaptive** — automatically picks relevant passages.")

        prompt_variant = st.selectbox(
            "Prompt Style",
            ["settlein", "walert", "closed_book"],
            index=0,
            help="settlein: full RAG, walert: minimal RAG, closed_book: no retrieval",
        )

        st.divider()

        # Corpus stats
        st.header("📊 Corpus")
        sources = set(p["source_id"] for p in passages)
        col1, col2 = st.columns(2)
        col1.metric("Passages", len(passages))
        col2.metric("Sources", len(sources))

        st.divider()

        # Source list
        st.header("📚 Sources")
        shown = set()
        for p in passages:
            sid = p["source_id"]
            if sid not in shown:
                shown.add(sid)
                title = p.get("title", sid)
                short = title[:50] + ("…" if len(title) > 50 else "")
                st.caption(f"**{sid}** — {short}")

        st.divider()
        st.caption("Built by **Group 50** · RMIT COSC2669 WIL Project")

        return retrieval_method, prompt_variant


# ── Evidence Panel ───────────────────────────────────────────────────────────

def render_evidence(evidence: list[tuple[dict, float]]):
    """Render the evidence panel with retrieved passages."""
    if not evidence:
        return

    st.subheader("📎 Retrieved Evidence")

    for i, (passage, score) in enumerate(evidence, 1):
        source_id = passage.get("source_id", "?")
        heading = passage.get("heading", "")
        title = passage.get("title", "")
        text = passage.get("contents", "")
        url = passage.get("url", "")

        display_text = text[:500] + "…" if len(text) > 500 else text
        label = f"Passage {i} · **{source_id}** · {heading[:60]}"

        with st.expander(label, expanded=(i == 1)):
            st.caption(f"Score: `{score:.4f}` · Source: **{source_id}**")
            if title:
                st.markdown(f"**{title}**")
            if heading:
                st.markdown(f"*{heading}*")
            st.markdown(display_text)
            if url:
                st.markdown(f"[🔗 Open source]({url})")


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    init_session()

    # Load engine
    try:
        engine, passages = init_search_engine()
    except FileNotFoundError:
        st.error("❌ Collection not found! Run extraction and build_collection first.")
        st.code(
            "python -m src.ingest.extract --draft\n"
            "python -m src.ingest.build_collection --draft",
            language="bash",
        )
        return

    config = load_config()

    # Header
    st.markdown('<h1 class="main-header">🇦🇺 SettleIN</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Your AI assistant for settling in Australia — '
        "visas, work rights, Medicare, transport & more</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    retrieval_method, prompt_variant = render_sidebar(passages)

    # Layout: Chat + Evidence
    chat_col, evidence_col = st.columns([3, 2])

    with chat_col:
        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if question := st.chat_input("Ask about visas, work rights, Medicare, transport…"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Retrieve (adaptive: auto-selects how many passages are relevant)
            with st.spinner("🔍 Searching…"):
                if prompt_variant == "closed_book":
                    results = None
                    st.session_state.evidence = []
                else:
                    results = engine.adaptive_search(question, method=retrieval_method)
                    st.session_state.evidence = results

            # Generate
            with st.spinner("💭 Generating answer…"):
                answer = generate_answer(
                    question=question,
                    results=results,
                    variant=prompt_variant,
                    config=config,
                )

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

    with evidence_col:
        render_evidence(st.session_state.evidence)

    # Example questions on first load
    if not st.session_state.messages:
        st.divider()
        st.subheader("💡 Try asking")
        examples = [
            "What are the work limitations under visa condition 8105?",
            "How do I enrol in Medicare as an international student?",
            "What is a myki card and how do I get one?",
            "How many hours can I work while studying?",
            "What should I do if my employer is not paying me correctly?",
            "How do I apply for an ABN?",
            "What is bulk billing?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                st.markdown(f"- *{ex}*")


if __name__ == "__main__":
    main()
