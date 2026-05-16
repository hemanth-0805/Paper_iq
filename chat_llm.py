from __future__ import annotations

import os
from typing import List, Dict, Optional


def _build_messages(
    question: str,
    context: str,
    chat_history: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    messages = []
    
    system_prompt = (
        "You are a helpful research assistant for the user.\n"
        "IMPORTANT: Answer using ONLY the provided paper context.\n"
        "If the answer is not present in the context, say exactly: "
        "'I cannot find that in the uploaded paper.'\n"
        "Do not hallucinate. Do not use outside knowledge.\n"
    )

    if chat_history:
        # Keep the prompt smaller by including only the last ~6 messages.
        relevant = chat_history[-6:]
        for m in relevant:
            role = "model" if m.get("role") == "assistant" else "user"
            content = (m.get("content") or "").strip()
            if not content:
                continue
            messages.append({"role": role, "parts": [content]})
            
    user_prompt = (
        f"{system_prompt}\n\n"
        "PAPER CONTEXT (retrieved):\n"
        f"{context}\n\n"
        f"USER QUESTION:\n{question}\n"
    )

    messages.append({"role": "user", "parts": [user_prompt]})

    return messages


def generate_answer(
    question: str,
    context: str,
    chat_history: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Generate the final answer from the LLM.

    If Gemini is not configured/available, the function raises.
    """
    if not context.strip():
        return "I cannot find that in the uploaded paper."

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Install/config `google-generativeai` and set GEMINI_API_KEY.")

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError("Google Generative AI package not available. Run `pip install google-generativeai`.") from e

    chosen_model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    genai.configure(api_key=gemini_api_key)

    messages = _build_messages(question=question, context=context, chat_history=chat_history)

    generative_model = genai.GenerativeModel(
        model_name=chosen_model,
        generation_config=genai.GenerationConfig(temperature=temperature),
    )

    resp = generative_model.generate_content(messages)

    return (resp.text or "").strip()

