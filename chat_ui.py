from __future__ import annotations

from typing import Dict, List

import os
import streamlit as st
import pandas as pd

from storage import get_user_history, get_analysis_extracted_text
from chat_rag import answer_question_with_rag


def _get_chat_messages_key(analysis_id: int) -> str:
    return f"chat_messages_{analysis_id}"


def render_chat_with_paper() -> None:
    st.subheader("💬 Chat with Paper")
    st.write("Discuss the paper's contents with the AI. 🤖")
    
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        st.error("The Gemini API Key is missing from the server environment. Please configure it in .env.")
        return

    user_id = int(st.session_state["auth"]["user_id"])

    hist = get_user_history(user_id)
    if hist.empty:
        st.info("No analyses yet. Upload a paper first, then come back to chat.")
        return

    # Pick which analysis/paper to chat about.
    display = hist.copy()
    display["label"] = display["paper_name"].astype(str) + " (id: " + display["analysis_id"].astype(str) + ")"
    # Default to latest (first row after sorting in storage).
    latest_row = display.iloc[0] if not display.empty else None

    analysis_id = st.selectbox(
        "Choose paper for chat",
        options=display["analysis_id"].tolist(),
        format_func=lambda x: display.loc[display["analysis_id"] == x, "label"].iloc[0],
        index=0,
    )

    chat_messages_key = _get_chat_messages_key(int(analysis_id))
    st.session_state.setdefault(chat_messages_key, [])

    # Render chat history.
    for m in st.session_state[chat_messages_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if st.button("Clear chat", type="secondary"):
        st.session_state[chat_messages_key] = []
        st.rerun()

    prompt = st.chat_input("Ask a question about the selected paper...")
    if not prompt:
        return

    # Append user message.
    st.session_state[chat_messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    extracted_text = get_analysis_extracted_text(int(analysis_id), user_id=user_id)
    if not extracted_text or not extracted_text.strip():
        assistant_text = "I cannot answer because no paper text was found for this analysis."
        st.session_state[chat_messages_key].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
        return

    # Prepare chat history payload for the RAG/LLM layer.
    chat_history_payload: List[Dict[str, str]] = st.session_state[chat_messages_key]

    with st.chat_message("assistant"):
        with st.spinner("Searching relevant parts of the paper and generating an answer..."):
            result = answer_question_with_rag(
                analysis_id=int(analysis_id),
                extracted_text=extracted_text,
                question=prompt,
                chat_history=chat_history_payload,
                top_k=5,
            )
        assistant_text = str(result.get("answer", "")).strip()
        st.markdown(assistant_text)

    # Append assistant message.
    st.session_state[chat_messages_key].append({"role": "assistant", "content": assistant_text})

