"""Streamlit chat frontend for the legalization assistant."""

import streamlit as st

from .config import DEFAULT_NUM_RESULTS, MODEL
from .rag import retrieve, stream_answer
from .search import get_search

EXAMPLE_QUESTIONS = [
    "What types of residence permits exist in Georgia?",
    "On what grounds can a Georgian visa be refused?",
    "How is the period of stay in Georgia terminated?",
    "What rights does a residence permit give me at work?",
]

st.set_page_config(page_title="Georgia Visa & Legalization Assistant", page_icon="🇬🇪")


def render_sources(sources: list[dict]) -> None:
    """Show what the answer was grounded in, so the user can check it."""
    if not sources:
        st.info("No matching articles were found in the indexed legislation.")
        return
    with st.expander(f"Sources ({len(sources)} article excerpts)"):
        for source in sources:
            label = f"**{source['article']} — {source['title']}**"
            if source.get("n_parts", 1) > 1:
                label += f"  ·  part {source['part']} of {source['n_parts']}"
            st.markdown(label)
            amendments = source.get("amendments") or []
            st.caption(
                f"{source['doc_title']} · {source['chapter']} · p. {source['page']}"
                + (f" · last amended: {amendments[-1]}" if amendments else "")
            )
            st.markdown(
                f"> {source['text'][:600]}{'…' if len(source['text']) > 600 else ''}".replace(
                    "\n", "\n> "
                )
            )
            st.divider()


def main() -> None:
    with st.spinner("Building the search index..."):
        index = get_search()

    st.title("🇬🇪 Georgia Visa & Legalization Assistant")
    st.caption(
        "Answers are drawn only from the Georgian legislation indexed below, with the "
        "articles they rely on. This is legal information, not legal advice."
    )

    with st.sidebar:
        st.header("Settings")
        num_results = st.slider(
            "Article excerpts to retrieve",
            min_value=1,
            max_value=10,
            value=DEFAULT_NUM_RESULTS,
            help=(
                "More excerpts give broader coverage but a noisier prompt. Whatever "
                "you pick, only as many as fit the token quota are used — the "
                "expander under each answer lists the ones it was grounded in."
            ),
        )
        doc_ids = index.doc_ids
        doc_choice = st.selectbox(
            "Source document",
            options=["All documents", *doc_ids],
            help="Restrict retrieval to one law.",
        )
        translate = st.toggle(
            "Translate non-English questions",
            value=True,
            help=(
                "The legislation is indexed in English. When on, questions written in "
                "another script are turned into English search terms first."
            ),
        )
        st.divider()
        st.caption(f"Answer model: `{MODEL}`")
        st.caption(f"Indexed chunks: {len(index.documents)}")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.session_state.setdefault("messages", [])

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        columns = st.columns(2)
        for position, example in enumerate(EXAMPLE_QUESTIONS):
            if columns[position % 2].button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask about visas, residence permits, or your stay in Georgia")
    if pending := st.session_state.pop("pending", None):
        question = pending
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the legislation..."):
                sources = retrieve(
                    question,
                    num_results=num_results,
                    doc_id=None if doc_choice == "All documents" else doc_choice,
                    translate=translate,
                )
            answer = st.write_stream(stream_answer(question, sources, history=history))
        except RuntimeError as error:
            st.error(str(error))
            st.session_state.messages.pop()
            return
        except Exception as error:
            st.error(f"The request failed: {error}")
            st.session_state.messages.pop()
            return
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
