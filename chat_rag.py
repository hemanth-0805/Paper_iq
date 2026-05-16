from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from chat_llm import generate_answer


_EMBEDDING_CACHE: Dict[str, Tuple[List[str], np.ndarray]] = {}


def _text_sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def chunk_text(
    text: str,
    *,
    chunk_size_chars: int = 1200,
    overlap_chars: int = 180,
) -> List[str]:
    """
    Chunk paper text into overlapping windows to improve retrieval.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Prefer paragraph boundaries when possible.
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        c = buf.strip()
        if c:
            chunks.append(c)
        buf = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if len(buf) + len(p) + 2 <= chunk_size_chars:
            buf = (buf + "\n\n" + p) if buf else p
            continue

        # Current buffer is too big once we add p.
        flush()
        buf = p

        # If a single paragraph is extremely large, split it.
        while len(buf) > chunk_size_chars:
            chunk = buf[:chunk_size_chars]
            chunks.append(chunk.strip())
            buf = buf[chunk_size_chars - overlap_chars :]

    flush()

    # Final overlap trimming pass (optional).
    # Ensure chunks are not empty and somewhat sized.
    return [c for c in chunks if len(c) >= 150]


def _cosine_sim_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Assumes both query_vec and matrix rows are normalized to unit length.
    return matrix @ query_vec


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def embed_chunks(chunks: List[str], *, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed chunks using sentence-transformers.
    """
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    # Try sentence-transformers first.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        # Cache model inside the function (simple module-level caching would also work).
        # Streamlit hot-reloads can re-import, so this keeps it robust.
        if not hasattr(embed_chunks, "_st_model"):
            embed_chunks._st_model = SentenceTransformer(model_name)  # type: ignore[attr-defined]

        model = embed_chunks._st_model  # type: ignore[attr-defined]
        vecs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers embeddings are required for the chatbot. "
            "Install with `pip install sentence-transformers`."
        ) from e


def _resolve_followup_query(chat_history: List[Dict[str, str]], question: str) -> str:
    """
    Very lightweight follow-up handling: if question uses vague pronouns,
    incorporate the previous user turn into the retrieval query.
    """
    q = (question or "").lower()
    followup_pronouns = ["it", "this", "that", "these", "those", "they", "them", "there", "its", "as mentioned"]

    if not any(p in q.split() for p in followup_pronouns):
        return question

    # Find last user message in history
    last_user = None
    for m in reversed(chat_history):
        if m.get("role") == "user":
            last_user = m.get("content")
            break
    if last_user and last_user.strip() and last_user.strip() != question.strip():
        return f"{last_user}\n\nFollow-up: {question}"
    return question


def retrieve_top_k_chunks(
    question: str,
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    *,
    top_k: int = 5,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> List[Tuple[int, str]]:
    if not chunks:
        return []

    # Embed query into the same vector space.
    # sentence-transformers path returns normalized vectors; TF-IDF path normalizes too.
    try:
        # Use the same embedding logic but only for the query.
        query_vec = embed_chunks([question], model_name=embed_model_name)[0]
    except Exception:
        query_vec = embed_chunks([question], model_name=embed_model_name)[0]

    query_vec = _normalize_rows(np.asarray([query_vec], dtype=np.float32))[0]
    sims = _cosine_sim_matrix(query_vec, _normalize_rows(chunk_embeddings))
    idxs = np.argsort(sims)[::-1][: max(1, top_k)]
    return [(int(i), chunks[i]) for i in idxs]


def build_context_from_chunks(chunks: List[Tuple[int, str]], *, max_chunks: int = 5) -> str:
    """
    Build the context string that will be provided to the LLM.
    """
    selected = chunks[:max_chunks]
    parts = []
    for rank, (i, c) in enumerate(selected, start=1):
        c_clean = (c or "").strip()
        if not c_clean:
            continue
        parts.append(f"[CHUNK {rank}]\n{c_clean}")
    return "\n\n".join(parts).strip()


def answer_question_with_rag(
    *,
    analysis_id: Optional[int],
    extracted_text: str,
    question: str,
    chat_history: List[Dict[str, str]],
    top_k: int = 5,
) -> Dict[str, object]:
    """
    Returns:
      - answer: str
      - retrieved_chunk_count: int
    """
    if not extracted_text.strip():
        return {
            "answer": "I cannot answer because no paper text was found for this analysis.",
            "retrieved_chunk_count": 0,
        }

    chunks = chunk_text(extracted_text)
    if not chunks:
        return {
            "answer": "I cannot answer because the paper text could not be chunked.",
            "retrieved_chunk_count": 0,
        }

    cache_key = f"{analysis_id or 'text'}:{_text_sha256(extracted_text)[:16]}:chunks={len(chunks)}"
    if cache_key in _EMBEDDING_CACHE:
        _, embeddings = _EMBEDDING_CACHE[cache_key]
    else:
        embeddings = embed_chunks(chunks)
        _EMBEDDING_CACHE[cache_key] = (chunks, embeddings)

    # In cache case, chunks list is recomputed above; that's fine for safety.
    retrieved = retrieve_top_k_chunks(question=_resolve_followup_query(chat_history, question), chunks=chunks, chunk_embeddings=embeddings, top_k=top_k)
    context = build_context_from_chunks(retrieved, max_chunks=top_k)

    try:
        answer = generate_answer(question=question, context=context, chat_history=chat_history)
        if answer.strip() == "I cannot find that in the uploaded paper.":
            # Fallback to the entire paper text since Gemini has a large context window
            answer = generate_answer(question=question, context=extracted_text, chat_history=chat_history)
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "GEMINI_API_KEY" in error_msg:
            msg = "Sorry, there is an issue with your Gemini API key. Please verify it is correct."
        elif "quota" in error_msg.lower():
            msg = "Sorry, your Gemini API key has exceeded its quota. Please check your billing details."
        else:
            msg = f"I can’t generate an LLM answer right now ({e})."

        if not context.strip():
            answer = f"{msg}\n\nI cannot find that in the uploaded paper."
        else:
            answer = f"{msg}\n\nHere are the most relevant excerpts from the paper instead:\n\n" + context

    return {
        "answer": answer,
        "retrieved_chunk_count": len(retrieved),
    }

