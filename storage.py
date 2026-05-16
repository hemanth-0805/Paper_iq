import re
from datetime import datetime
from typing import Optional

import pandas as pd

from db import _get_analysis_col, _next_sequence, get_mongo_db


def build_insights_text(engine: "InsightEngine", paper_name: str, keywords_df: Optional[pd.DataFrame] = None) -> str:
    lines = []
    lines.append(f"Paper: {paper_name}")
    lines.append("")

    s = engine.stats
    lines.append("== Basic Stats ==")
    lines.append(f"Words: {int(s.get('word_count') or s.get('words') or 0)}")
    lines.append(f"Sentences: {int(s.get('sentence_count') or 0)}")
    if "avg_sentence_len" in s:
        lines.append(f"Avg sentence length: {float(s.get('avg_sentence_len') or 0.0):.2f}")
    if "avg_word_len" in s:
        lines.append(f"Avg word length: {float(s.get('avg_word_len') or 0.0):.2f}")
    if "vocab_diversity" in s:
        lines.append(f"Vocabulary diversity: {float(s.get('vocab_diversity') or 0.0):.3f}")
    if "complex_word_ratio" in s:
        lines.append(f"Complex word ratio: {float(s.get('complex_word_ratio') or 0.0):.3f}")
    lines.append("")

    lines.append("== Scores (0..100) ==")
    for k in ["Composite", "Language", "Coherence", "Reasoning", "Sophistication", "Readability"]:
        if k in engine.scores:
            lines.append(f"{k}: {float(engine.scores.get(k) or 0.0):.2f}")
    lines.append("")

    lines.append("== Domain ==")
    lines.append(engine.domain or "General Research")
    lines.append("")

    lines.append("== Top Keywords ==")
    kw_df = keywords_df
    if isinstance(kw_df, pd.DataFrame) and not kw_df.empty:
        for _, r in kw_df.iterrows():
            lines.append(f"- {r['keyword']} ({float(r['score']):.4f})")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("== Core Section Summaries (Medium) ==")
    for sec in ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"]:
        lines.append(f"[{sec}]")
        lines.append((engine.ai_summaries.get(sec, {}).get("Medium") or "N/A").strip())
        lines.append("")

    return "\n".join(lines).strip()


def save_analysis(user_id: int, paper_name: str, engine: "InsightEngine", keywords_df: Optional[pd.DataFrame] = None) -> int:
    insights_text = build_insights_text(engine, paper_name=paper_name, keywords_df=keywords_df)
    extracted_words = len(re.findall(r"\b\w+\b", insights_text))
    total_words = int(engine.stats.get("word_count") or engine.stats.get("words") or 0)

    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)

    analysis_id = _next_sequence(db, "analysis_history")
    analysis_col.insert_one(
        {
            "analysis_id": analysis_id,
            "user_id": int(user_id),
            "paper_name": paper_name,
            "upload_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total_words": int(total_words),
            "extracted_words": int(extracted_words),
            "readability_score": float(engine.scores.get("Readability", 0.0)),
            "coherence_score": float(engine.scores.get("Coherence", 0.0)),
            "sophistication_score": float(engine.scores.get("Sophistication", 0.0)),
            "insights_text": insights_text,
            # Store the extracted paper text so "files" are persisted in MongoDB.
            # Note: MongoDB documents have a 16MB limit; if your PDFs are huge,
            # we'll need GridFS to store raw bytes/text safely.
            "extracted_text": engine.full_text or "",
        }
    )
    return int(analysis_id)


def get_user_history(user_id: int) -> pd.DataFrame:
    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)

    # Fetch only metadata for the table (avoid pulling large extracted_text).
    docs = list(
        analysis_col.find(
            {"user_id": int(user_id)},
            {
                "_id": 0,
                "analysis_id": 1,
                "paper_name": 1,
                "upload_time": 1,
                "total_words": 1,
                "extracted_words": 1,
                "readability_score": 1,
                "coherence_score": 1,
                "sophistication_score": 1,
            },
        ).sort("analysis_id", -1)
    )
    if not docs:
        return pd.DataFrame(
            columns=[
                "analysis_id",
                "paper_name",
                "upload_time",
                "total_words",
                "extracted_words",
                "readability_score",
                "coherence_score",
                "sophistication_score",
            ]
        )

    df = pd.DataFrame(docs)
    keep = [
        "analysis_id",
        "paper_name",
        "upload_time",
        "total_words",
        "extracted_words",
        "readability_score",
        "coherence_score",
        "sophistication_score",
    ]
    return df[keep]


def get_analysis_record(analysis_id: int, user_id: Optional[int] = None) -> Optional[dict]:
    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)

    query = {"analysis_id": int(analysis_id)}
    if user_id is not None:
        query["user_id"] = int(user_id)

    # Fetch only what the UI needs for an analysis detail page.
    # extracted_text is intentionally excluded to keep pages fast.
    doc = analysis_col.find_one(
        query,
        {
            "_id": 0,
            "analysis_id": 1,
            "user_id": 1,
            "paper_name": 1,
            "upload_time": 1,
            "insights_text": 1,
            "readability_score": 1,
            "coherence_score": 1,
            "sophistication_score": 1,
        },
    )
    if not doc:
        return None
    return doc


def get_analysis_extracted_text(analysis_id: int, user_id: Optional[int] = None) -> Optional[str]:
    """
    Fetch the stored extracted paper text for RAG/chat.

    We keep this separate from `get_analysis_record()` so the rest of the UI stays fast.
    """
    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)

    query = {"analysis_id": int(analysis_id)}
    if user_id is not None:
        query["user_id"] = int(user_id)

    doc = analysis_col.find_one(query, {"_id": 0, "extracted_text": 1})
    if not doc:
        return None
    return doc.get("extracted_text") or ""

